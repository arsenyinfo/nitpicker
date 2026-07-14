use clap::{Args as ClapArgs, Parser, Subcommand};
use eyre::Result;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

use nitpicker_agent::{config, openrouter};

mod debate;
mod detect;
#[cfg(feature = "antigravity")]
mod gemini_proxy;
mod output;
mod pr;
mod progress;
mod prompts;
mod reflect;
mod review;

/// Flags shared between the default review mode and the ask subcommand.
#[derive(Debug, ClapArgs)]
struct CommonArgs {
    #[arg(long, default_value = ".")]
    repo: PathBuf,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long, short)]
    verbose: bool,
}

#[derive(Debug, Parser)]
#[command(name = "nitpicker")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    common: CommonArgs,

    #[arg(
        long,
        help = "Additional review instructions appended to the diff context (use `ask` for fully custom prompts)"
    )]
    prompt: Option<String>,

    /// Analyze existing code instead of reviewing changes
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    analyze: Option<PathBuf>,

    /// Disable actor-critic debate and use parallel aggregation instead
    #[arg(long)]
    no_debate: bool,

    /// Mix all reviewer models into a shared pool; each LLM call picks one at random
    #[arg(long)]
    alloy: bool,

    /// Maximum debate rounds
    #[arg(long, default_value = "5")]
    rounds: usize,

    /// Maximum tool-use turns per agent or debate turn
    #[arg(long, value_parser = parse_positive_usize)]
    max_turns: Option<usize>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate a nitpicker config template
    Init {
        /// Write to ~/.nitpicker/config.toml instead of ./nitpicker.toml
        #[arg(long)]
        global: bool,

        /// Prefer OpenRouter experimental free models in the generated config
        #[arg(long)]
        free: bool,
    },
    /// Ask multiple LLM agents a free-form question about the codebase
    Ask {
        #[command(flatten)]
        common: CommonArgs,
        /// Question or topic to discuss
        topic: String,
        /// Disable actor-critic debate and use parallel aggregation instead
        #[arg(long)]
        no_debate: bool,
        /// Mix all reviewer models into a shared pool; each LLM call picks one at random
        #[arg(long)]
        alloy: bool,
        /// Maximum debate rounds
        #[arg(long, default_value = "5")]
        rounds: usize,
        /// Maximum tool-use turns per agent or debate turn
        #[arg(long, value_parser = parse_positive_usize)]
        max_turns: Option<usize>,
    },
    /// Review a GitHub PR (current branch's PR or a remote PR by URL)
    Pr(pr::PrArgs),
    /// Reflect on past nitpicker sessions to identify patterns and friction points
    Reflect {
        /// Directory containing sessions (default: ~/.nitpicker/sessions)
        #[arg(long)]
        sessions_dir: Option<PathBuf>,
        /// Number of most recent sessions to analyze
        #[arg(long, default_value = "20")]
        n: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let verbose = args.common.verbose
        || matches!(&args.command, Some(Command::Ask { common, .. }) if common.verbose)
        || matches!(&args.command, Some(Command::Pr(a)) if a.common.verbose);
    let is_reflect = matches!(&args.command, Some(Command::Reflect { .. }));
    let default_level = if verbose || is_reflect {
        "info"
    } else {
        "warn"
    };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
    // logs are never the report — keep stdout reserved for the deliverable so
    // `pr --format json` emits a clean single JSON object.
    tracing_subscriber::fmt()
        .with_writer(progress::stderr_log_writer)
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_level(true)
        .with_ansi(progress::stderr_supports_color())
        .compact()
        .init();

    // note: no json panic hook. reviewer work runs in tokio::spawn tasks whose
    // panics are caught as JoinError and folded into a degraded report (exit 3
    // for review/ask, status ok for pr); a process-wide hook would double-emit
    // there. a genuine top-level panic aborts non-zero with a stderr message,
    // which is an acceptable catastrophic-failure signal for the consumer.
    match args.command {
        Some(Command::Init { global, free }) => {
            let path = init_config_path(global)?;
            if path.exists() {
                eyre::bail!("{} already exists", path.display());
            }
            run_init(path, free).await?;
            return Ok(());
        }
        Some(Command::Ask {
            common,
            topic,
            no_debate,
            alloy,
            rounds,
            max_turns,
        }) => {
            let repo = common.repo.canonicalize()?;
            if !repo.join(".git").is_dir() {
                eyre::bail!("--repo must point to a git repository (missing .git)");
            }
            let config = load_resolved_config(common.config.as_deref(), &repo).await?;
            let max_turns = config.max_turns(max_turns)?;
            let use_alloy = alloy || config.default_alloy();
            config.validate_alloy(use_alloy)?;

            if use_alloy && no_debate {
                eprintln!("warning: --alloy has no effect with --no-debate");
            }

            if !no_debate && config.default_debate() {
                if config.reviewer.len() < 2 {
                    eyre::bail!(
                        "debate mode requires at least 2 reviewers, found {} — add another reviewer or set debate = false in [defaults]",
                        config.reviewer.len()
                    );
                }
                let outcome = debate::run_debate(
                    &repo,
                    &topic,
                    &config,
                    debate::DebateOptions {
                        max_rounds: rounds,
                        max_turns,
                        verbose: common.verbose,
                        mode: debate::DebateMode::Topic,
                        alloy: use_alloy,
                        format: output::OutputFormat::Text,
                    },
                )
                .await?;
                println!("{}", outcome.report);
                if common.verbose {
                    eprintln!("\nTranscript saved to: {}", outcome.transcript_path.display());
                }
                exit_if_degraded(outcome.degraded);
                return Ok(());
            }

            let outcome = review::run_review(
                &repo,
                &topic,
                &config,
                max_turns,
                common.verbose,
                review::TaskMode::Ask,
            )
            .await?;
            println!("{}", outcome.report);
            exit_if_degraded(outcome.degraded);
            return Ok(());
        }
        Some(Command::Pr(pr_args)) => {
            // config loading happens inside run_pr so its failures honor --format json too
            return pr::run_pr(pr_args).await;
        }
        Some(Command::Reflect { sessions_dir, n }) => {
            let repo = args.common.repo.canonicalize()?;
            let config = load_resolved_config(args.common.config.as_deref(), &repo).await?;
            return reflect::run_reflect(reflect::ReflectArgs {
                sessions_dir,
                n,
                repo,
                config,
            })
            .await;
        }
        None => {}
    }

    let repo = args.common.repo.canonicalize()?;
    if !repo.join(".git").is_dir() {
        eyre::bail!("--repo must point to a git repository (missing .git)");
    }

    let config = load_resolved_config(args.common.config.as_deref(), &repo).await?;
    let max_turns = config.max_turns(args.max_turns)?;

    let scope = match args.analyze {
        Some(_) => prompts::ReviewScope::Static,
        None => prompts::ReviewScope::Diff,
    };
    let prompt = if let Some(path) = args.analyze {
        let path_opt = if path.as_os_str().is_empty() {
            None
        } else {
            Some(path.as_path())
        };
        build_analysis_prompt(path_opt, args.prompt.as_deref())
    } else {
        let base = detect_diff_context(&repo)?;
        match args.prompt {
            Some(p) => format!("{base}\n\nAdditional instructions: {p}"),
            None => base,
        }
    };

    let use_alloy = args.alloy || config.default_alloy();
    config.validate_alloy(use_alloy)?;
    if use_alloy && args.no_debate {
        eprintln!("warning: --alloy has no effect with --no-debate");
    }

    if !args.no_debate && config.default_debate() {
        if config.reviewer.len() < 2 {
            eyre::bail!(
                "debate mode requires at least 2 reviewers, found {} — add another reviewer or set debate = false in [defaults]",
                config.reviewer.len()
            );
        }
        let outcome = debate::run_debate(
            &repo,
            &prompt,
            &config,
            debate::DebateOptions {
                max_rounds: args.rounds,
                max_turns,
                verbose: args.common.verbose,
                mode: debate::DebateMode::Review(scope),
                alloy: use_alloy,
                format: output::OutputFormat::Text,
            },
        )
        .await?;
        println!("{}", outcome.report);
        if args.common.verbose {
            eprintln!("\nTranscript saved to: {}", outcome.transcript_path.display());
        }
        exit_if_degraded(outcome.degraded);
        Ok(())
    } else {
        let outcome = review::run_review(
            &repo,
            &prompt,
            &config,
            max_turns,
            args.common.verbose,
            review::TaskMode::Review(scope),
        )
        .await?;
        println!("{}", outcome.report);
        exit_if_degraded(outcome.degraded);
        Ok(())
    }
}

/// Exit-code contract for the default-review and `ask` arms: 0 = clean verdict,
/// 1 = hard failure (no verdict), 3 = degraded verdict (report printed, but at least one
/// reviewer or debate turn failed or fell back). 2 is deliberately unused — clap exits 2
/// on usage errors, and the whole point is an unambiguous subprocess signal.
/// `pr` keeps its own JSON/text contract.
fn exit_if_degraded(degraded: bool) {
    if !degraded {
        return;
    }
    // process::exit skips stdout teardown; without a flush a piped report would be lost
    // (same reasoning as output::emit_json).
    match std::io::Write::flush(&mut std::io::stdout()) {
        Ok(()) => {}
        Err(err) => {
            eprintln!("error: failed to flush report to stdout: {err}");
            std::process::exit(1);
        }
    }
    eprintln!(
        "warning: degraded verdict — a reviewer or debate turn failed or ended without submit_verdict (exit code 3)"
    );
    std::process::exit(3);
}

fn load_config(explicit_path: Option<&Path>, repo: &Path) -> Result<config::Config> {
    let config: config::Config = if let Some(path) = explicit_path {
        let content = std::fs::read_to_string(path)
            .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", path))?;
        toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"))?
    } else if repo.join("nitpicker.toml").exists() {
        let path = repo.join("nitpicker.toml");
        let content = std::fs::read_to_string(&path)
            .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", path))?;
        toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"))?
    } else if let Some(home) = dirs::home_dir() {
        let path = home.join(".nitpicker").join("config.toml");
        if path.exists() {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", path))?;
            toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"))?
        } else {
            eyre::bail!("no config found — run `nitpicker init [--global]` to generate one")
        }
    } else {
        eyre::bail!("no config found — run `nitpicker init [--global]` to generate one")
    };
    config.validate()?;
    Ok(config)
}

pub(crate) async fn load_resolved_config(
    explicit_path: Option<&Path>,
    repo: &Path,
) -> Result<config::Config> {
    let mut config = load_config(explicit_path, repo)?;
    openrouter::resolve_free_models(&mut config).await?;
    Ok(config)
}

async fn run_init(path: PathBuf, prefer_free: bool) -> eyre::Result<()> {
    println!("Detecting available providers...\n");
    let detected = detect::detect_all().await;

    if detected.is_empty() {
        eyre::bail!(
            "no providers detected — set at least one of: \
             ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, \
             OPENROUTER_API_KEY, KIMI_API_KEY, ZAI_API_KEY, MINIMAX_API_KEY, MISTRAL_API_KEY, \
             DATABRICKS_TOKEN (with DATABRICKS_HOST or ~/.databrickscfg)"
        );
    }

    println!("Detected providers:");
    for d in &detected {
        let key_info = match d.api_key_env {
            Some(env) => env.to_string(),
            None => d.auth.unwrap_or("api_key").to_string(),
        };
        println!("  ✓ {} ({}) via {}", d.name, key_info, d.source);
    }

    let use_openrouter_free = should_prefer_openrouter_free(&detected, prefer_free);
    if prefer_free && !use_openrouter_free {
        println!(
            "\nWarning: `--free` prefers OpenRouter free models, but OPENROUTER_API_KEY is not set; using the normal provider order."
        );
    }

    let prioritized = prioritize_init_detected(&detected, use_openrouter_free);
    let config = build_init_config(&prioritized, use_openrouter_free);
    let mut toml_str = toml::to_string_pretty(&config)
        .map_err(|e| eyre::eyre!("failed to serialize config: {e}"))?;

    let active_names: std::collections::HashSet<&str> = config
        .reviewer
        .iter()
        .map(|r| r.name.as_str())
        .chain(std::iter::once(prioritized[0].name))
        .collect();
    let extras: Vec<&detect::Detected> = detected
        .iter()
        .filter(|d| !active_names.contains(d.name))
        .collect();
    if !extras.is_empty() {
        toml_str.push_str("\n# Other detected providers — uncomment to add as a reviewer:\n");
        for d in extras {
            toml_str.push('\n');
            toml_str.push_str(&format_commented_reviewer(d));
            toml_str.push('\n');
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, &toml_str)?;
    println!("\nCreated {}", path.display());

    print_init_hints(&detected);
    Ok(())
}

fn format_commented_reviewer(d: &detect::Detected) -> String {
    let mut lines = vec![
        "# [[reviewer]]".to_string(),
        format!("# name = \"{}\"", d.name),
        format!("# model = \"{}\"", d.model),
        format!("# provider = \"{}\"", d.provider),
    ];
    if let Some(url) = &d.base_url {
        lines.push(format!("# base_url = \"{url}\""));
    }
    if let Some(env) = d.api_key_env {
        if d.local_server {
            lines.push(format!(
                "# api_key_env = \"{env}\"  # set to any non-empty value"
            ));
        } else {
            lines.push(format!("# api_key_env = \"{env}\""));
        }
    }
    if let Some(auth) = d.auth {
        lines.push(format!("# auth = \"{auth}\""));
    }
    lines.join("\n")
}

fn build_init_config(
    detected: &[&detect::Detected],
    prefer_openrouter_free: bool,
) -> config::Config {
    let non_local_count = detected.iter().filter(|d| !d.local_server).count();
    let debate = non_local_count >= 2;

    // aggregator: highest priority (list is already sorted)
    let agg = detected[0];
    let aggregator = config::AggregatorConfig {
        model: init_model_for_detected(agg, prefer_openrouter_free),
        provider: parse_provider_type(agg.provider),
        base_url: agg.base_url.clone(),
        api_key_env: agg.api_key_env.map(str::to_string),
        max_tokens: None,
        auth: agg.auth.map(str::to_string),
        azure_scope: None,
        azure_credentials: None,
    };

    let reviewer_slots = if debate { 2 } else { 1 };
    let reviewers = pick_reviewers(detected, reviewer_slots, prefer_openrouter_free);

    config::Config {
        defaults: Some(config::DefaultsConfig {
            debate: Some(debate),
            alloy: None,
            max_turns: Some(config::DEFAULT_MAX_TURNS),
            compact_threshold: Some(100_000),
            log_trajectories: Some(false),
        }),
        aggregator,
        reviewer: reviewers,
    }
}

fn pick_reviewers(
    detected: &[&detect::Detected],
    count: usize,
    prefer_openrouter_free: bool,
) -> Vec<config::ReviewerConfig> {
    if prefer_openrouter_free {
        return detected
            .first()
            .into_iter()
            .cycle()
            .take(count)
            .map(|d| make_reviewer(d, prefer_openrouter_free))
            .collect();
    }

    let mut result = Vec::new();
    let mut seen_names: std::collections::HashSet<&str> = Default::default();

    // first pass: diverse provider names
    for d in detected {
        if result.len() >= count {
            break;
        }
        if seen_names.insert(d.name) {
            result.push(make_reviewer(d, prefer_openrouter_free));
        }
    }

    // second pass: fill remaining slots with any provider
    for d in detected {
        if result.len() >= count {
            break;
        }
        if result
            .iter()
            .all(|r: &config::ReviewerConfig| r.name != d.name)
        {
            result.push(make_reviewer(d, prefer_openrouter_free));
        }
    }

    result
}

fn make_reviewer(d: &detect::Detected, prefer_openrouter_free: bool) -> config::ReviewerConfig {
    config::ReviewerConfig {
        name: d.name.to_string(),
        model: init_model_for_detected(d, prefer_openrouter_free),
        provider: parse_provider_type(d.provider),
        base_url: d.base_url.clone(),
        api_key_env: d.api_key_env.map(str::to_string),
        compact_threshold: None,
        auth: d.auth.map(str::to_string),
        azure_scope: None,
        azure_credentials: None,
    }
}

fn should_prefer_openrouter_free(detected: &[detect::Detected], prefer_free: bool) -> bool {
    if !prefer_free {
        return false;
    }

    let has_openrouter = detected.iter().any(|d| d.name == "openrouter");
    has_openrouter && std::env::var("OPENROUTER_API_KEY").is_ok()
}

fn prioritize_init_detected(
    detected: &[detect::Detected],
    prefer_openrouter_free: bool,
) -> Vec<&detect::Detected> {
    let mut prioritized: Vec<&detect::Detected> = detected.iter().collect();
    if prefer_openrouter_free {
        prioritized.sort_by_key(|d| if d.name == "openrouter" { 0 } else { 1 });
    }
    prioritized
}

fn init_model_for_detected(d: &detect::Detected, prefer_openrouter_free: bool) -> String {
    if prefer_openrouter_free && d.name == "openrouter" {
        return "free".to_string();
    }

    d.model.clone()
}

fn parse_provider_type(s: &str) -> config::ProviderType {
    match s {
        "anthropic" => config::ProviderType::Anthropic,
        "gemini" => config::ProviderType::Gemini,
        "openrouter" => config::ProviderType::OpenRouter,
        _ => config::ProviderType::OpenAi,
    }
}

fn print_init_hints(detected: &[detect::Detected]) {
    let unset: Vec<&detect::Detected> = detected
        .iter()
        .filter(|d| {
            !d.local_server
                && d.api_key_env
                    .map(|env| std::env::var(env).is_err())
                    .unwrap_or(false)
        })
        .collect();

    if !unset.is_empty() {
        println!("\nProviders detected but env vars not yet set:");
        for d in unset {
            println!(
                "  export {}=...  # found via {}",
                d.api_key_env.unwrap(),
                d.source
            );
        }
    }

    let has_google_ai_key =
        std::env::var("GOOGLE_AI_API_KEY").is_ok() && std::env::var("GEMINI_API_KEY").is_err();
    if has_google_ai_key {
        println!("\n  Note: found GOOGLE_AI_API_KEY — the gemini client reads GEMINI_API_KEY;");
        println!("  add `export GEMINI_API_KEY=$GOOGLE_AI_API_KEY` to your shell profile.");
    }
}

fn init_config_path(global: bool) -> Result<PathBuf> {
    if global {
        let home =
            dirs::home_dir().ok_or_else(|| eyre::eyre!("failed to resolve home directory"))?;
        Ok(home.join(".nitpicker").join("config.toml"))
    } else {
        Ok(Path::new("nitpicker.toml").to_path_buf())
    }
}

pub(crate) fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid positive integer: {value}"))?;

    if parsed == 0 {
        return Err("value must be greater than 0".to_string());
    }

    Ok(parsed)
}

fn build_analysis_prompt(path: Option<&Path>, custom_prompt: Option<&str>) -> String {
    let target = match path {
        Some(p) => format!("`{}`", p.display()),
        None => "the entire repository".to_string(),
    };
    let base = format!(
        "Analyze the following code for issues and improvement opportunities:\n\
         - Target: {}\n\
         - Focus: correctness, security, performance, maintainability",
        target
    );
    match custom_prompt {
        Some(p) if !p.trim().is_empty() => {
            format!("{}\n\nAdditional instructions: {}", base, p)
        }
        _ => base,
    }
}

pub(crate) struct BaseBranch {
    pub(crate) name: String,
    pub(crate) revision: String,
}

pub fn detect_diff_context(repo: &Path) -> Result<String> {
    let branch = run_git(repo, &["rev-parse", "--abbrev-ref", "HEAD"])?;
    let branch = branch.trim();

    if branch == "HEAD" {
        eyre::bail!("detached HEAD state: checkout a branch before running nitpicker");
    }

    let base = detect_base_branch(repo);

    let has_uncommitted = !run_git(repo, &["status", "--porcelain"])
        .unwrap_or_default()
        .trim()
        .is_empty();

    let has_branch_commits = match base.as_ref() {
        Some(base) if branch != base.name => !run_git(
            repo,
            &["log", &format!("{}..HEAD", base.revision), "--oneline"],
        )?
        .trim()
        .is_empty(),
        _ => false,
    };

    if !has_uncommitted && !has_branch_commits {
        if let Some(base) = base.as_ref() {
            eyre::bail!(
                "no changes to review: no uncommitted changes and no branch commits vs {}",
                base.name
            );
        }
        eyre::bail!(
            "no changes to review: no uncommitted changes and no detectable base branch commits"
        );
    }

    let mut parts = Vec::new();
    if has_uncommitted {
        parts.push("- uncommitted changes (`git diff HEAD`)".to_string());
    }
    if has_branch_commits {
        let base = base
            .as_ref()
            .ok_or_else(|| eyre::eyre!("base branch required when branch commits are present"))?;
        parts.push(format!(
            "- commits on this branch vs {} (`git log {}..HEAD`, `git diff {}...HEAD`)",
            base.name, base.revision, base.revision
        ));
    }

    Ok(format!(
        "Review the following changes:\n{}",
        parts.join("\n")
    ))
}

pub(crate) fn detect_base_branch(repo: &Path) -> Option<BaseBranch> {
    run_git(repo, &["symbolic-ref", "refs/remotes/origin/HEAD"])
        .ok()
        .and_then(|s| {
            s.trim()
                .strip_prefix("refs/remotes/origin/")
                .map(str::to_string)
        })
        .and_then(|branch| resolve_base_branch(repo, &branch))
        .or_else(|| {
            ["main", "master"]
                .into_iter()
                .find_map(|branch| resolve_base_branch(repo, branch))
        })
}

fn resolve_base_branch(repo: &Path, branch: &str) -> Option<BaseBranch> {
    let local = format!("refs/heads/{branch}");
    if run_git(repo, &["rev-parse", "--verify", &local]).is_ok() {
        return Some(BaseBranch {
            name: branch.to_string(),
            revision: branch.to_string(),
        });
    }

    let remote = format!("refs/remotes/origin/{branch}");
    if run_git(repo, &["rev-parse", "--verify", &remote]).is_ok() {
        return Some(BaseBranch {
            name: branch.to_string(),
            revision: format!("origin/{branch}"),
        });
    }

    None
}

fn run_git(repo: &Path, args: &[&str]) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eyre::bail!("git {}: {}", args.join(" "), stderr.trim());
    }
    Ok(String::from_utf8(output.stdout)?)
}
