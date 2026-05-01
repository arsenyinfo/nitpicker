use clap::{Args as ClapArgs, Parser, Subcommand};
use eyre::Result;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

mod agent;
mod compact;
mod config;
mod debate;
mod detect;
mod gemini_proxy;
mod llm;
mod openrouter;
mod pr;
mod prompts;
mod provider;
mod review;
mod session;
mod tools;

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

    #[arg(long = "gemini-oauth")]
    gemini_oauth: bool,

    /// Analyze existing code instead of reviewing changes
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    analyze: Option<PathBuf>,

    /// Disable actor-critic debate and use parallel aggregation instead
    #[arg(long)]
    no_debate: bool,

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
        /// Maximum debate rounds
        #[arg(long, default_value = "5")]
        rounds: usize,
        /// Maximum tool-use turns per agent or debate turn
        #[arg(long, value_parser = parse_positive_usize)]
        max_turns: Option<usize>,
    },
    /// Review a GitHub PR (current branch's PR or a remote PR by URL)
    Pr(pr::PrArgs),
}


#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let verbose = args.common.verbose
        || matches!(&args.command, Some(Command::Ask { common, .. }) if common.verbose)
        || matches!(&args.command, Some(Command::Pr(a)) if a.common.verbose);
    let default_level = if verbose { "info" } else { "warn" };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_level(true)
        .with_ansi(true)
        .compact()
        .init();

    match args.command {
        Some(Command::Init { global }) => {
            let path = init_config_path(global)?;
            if path.exists() {
                eyre::bail!("{} already exists", path.display());
            }
            run_init(path).await?;
            return Ok(());
        }
        Some(Command::Ask {
            common,
            topic,
            no_debate,
            rounds,
            max_turns,
        }) => {
            let repo = common.repo.canonicalize()?;
            if !repo.join(".git").is_dir() {
                eyre::bail!("--repo must point to a git repository (missing .git)");
            }
            let mut config = load_config(common.config.as_deref(), &repo)?;
            openrouter::resolve_free_models(&mut config).await?;
            let max_turns = config.max_turns(max_turns)?;

            if !no_debate && config.default_debate() {
                if config.reviewer.len() < 2 {
                    eyre::bail!(
                        "debate mode requires at least 2 reviewers, found {} — add another reviewer or set debate = false in [defaults]",
                        config.reviewer.len()
                    );
                }
                let report = debate::run_debate(
                    &repo,
                    &topic,
                    &config,
                    rounds,
                    max_turns,
                    common.verbose,
                    debate::DebateMode::Topic,
                )
                .await?;
                println!("{report}");
                return Ok(());
            }

            let report = review::run_review(
                &repo,
                &topic,
                &config,
                max_turns,
                common.verbose,
                review::TaskMode::Ask,
            )
            .await?;
            println!("{report}");
            return Ok(());
        }
        Some(Command::Pr(pr_args)) => {
            let mut config = load_config(pr_args.common.config.as_deref(), &pr_args.common.repo)?;
            openrouter::resolve_free_models(&mut config).await?;
            return pr::run_pr(pr_args, config).await;
        }
        None => {}
    }

    if args.gemini_oauth {
        println!("Starting Gemini OAuth authentication flow...");
        let proxy_client = gemini_proxy::GeminiProxyClient::new().await?;
        match proxy_client.check_auth_status()? {
            gemini_proxy::AuthStatus::Valid => {
                println!("✓ Authentication successful! Token is valid.");
            }
            gemini_proxy::AuthStatus::ExpiredButRefreshable => {
                println!("⚠ Token expired but can be refreshed on next use.");
            }
            _ => {
                println!("✗ Authentication failed.");
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    let repo = args.common.repo.canonicalize()?;
    if !repo.join(".git").is_dir() {
        eyre::bail!("--repo must point to a git repository (missing .git)");
    }

    let mut config = load_config(args.common.config.as_deref(), &repo)?;
    openrouter::resolve_free_models(&mut config).await?;
    let max_turns = config.max_turns(args.max_turns)?;

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

    if !args.no_debate && config.default_debate() {
        if config.reviewer.len() < 2 {
            eyre::bail!(
                "debate mode requires at least 2 reviewers, found {} — add another reviewer or set debate = false in [defaults]",
                config.reviewer.len()
            );
        }
        let report = debate::run_debate(
            &repo,
            &prompt,
            &config,
            args.rounds,
            max_turns,
            args.common.verbose,
            debate::DebateMode::Review,
        )
        .await?;
        println!("{report}");
        Ok(())
    } else {
        let report = review::run_review(
            &repo,
            &prompt,
            &config,
            max_turns,
            args.common.verbose,
            review::TaskMode::Review,
        )
        .await?;
        println!("{report}");
        Ok(())
    }
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

async fn run_init(path: PathBuf) -> eyre::Result<()> {
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

    let config = build_init_config(&detected);
    let mut toml_str =
        toml::to_string_pretty(&config).map_err(|e| eyre::eyre!("failed to serialize config: {e}"))?;

    let active_names: std::collections::HashSet<&str> = config
        .reviewer
        .iter()
        .map(|r| r.name.as_str())
        .chain(std::iter::once(detected[0].name))
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
            lines.push(format!("# api_key_env = \"{env}\"  # set to any non-empty value"));
        } else {
            lines.push(format!("# api_key_env = \"{env}\""));
        }
    }
    if let Some(auth) = d.auth {
        lines.push(format!("# auth = \"{auth}\""));
    }
    lines.join("\n")
}

fn build_init_config(detected: &[detect::Detected]) -> config::Config {
    let non_local_count = detected.iter().filter(|d| !d.local_server).count();
    let debate = non_local_count >= 2;

    // aggregator: highest priority (list is already sorted)
    let agg = &detected[0];
    let aggregator = config::AggregatorConfig {
        model: agg.model.clone(),
        provider: parse_provider_type(agg.provider),
        base_url: agg.base_url.clone(),
        api_key_env: agg.api_key_env.map(str::to_string),
        max_tokens: None,
        auth: agg.auth.map(str::to_string),
    };

    let reviewer_slots = if debate { 2 } else { 1 };
    let reviewers = pick_reviewers(detected, reviewer_slots);

    config::Config {
        defaults: Some(config::DefaultsConfig {
            debate: Some(debate),
            max_turns: Some(config::DEFAULT_MAX_TURNS),
            compact_threshold: Some(100_000),
            log_trajectories: Some(false),
        }),
        aggregator,
        reviewer: reviewers,
    }
}

fn pick_reviewers(detected: &[detect::Detected], count: usize) -> Vec<config::ReviewerConfig> {
    let mut result = Vec::new();
    let mut seen_names: std::collections::HashSet<&str> = Default::default();

    // first pass: diverse provider names
    for d in detected {
        if result.len() >= count {
            break;
        }
        if seen_names.insert(d.name) {
            result.push(make_reviewer(d));
        }
    }

    // second pass: fill remaining slots with any provider
    for d in detected {
        if result.len() >= count {
            break;
        }
        if result.iter().all(|r: &config::ReviewerConfig| r.name != d.name) {
            result.push(make_reviewer(d));
        }
    }

    result
}

fn make_reviewer(d: &detect::Detected) -> config::ReviewerConfig {
    config::ReviewerConfig {
        name: d.name.to_string(),
        model: d.model.clone(),
        provider: parse_provider_type(d.provider),
        base_url: d.base_url.clone(),
        api_key_env: d.api_key_env.map(str::to_string),
        compact_threshold: None,
        auth: d.auth.map(str::to_string),
    }
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
            println!("  export {}=...  # found via {}", d.api_key_env.unwrap(), d.source);
        }
    }

    let has_google_ai_key =
        std::env::var("GOOGLE_AI_API_KEY").is_ok() && std::env::var("GEMINI_API_KEY").is_err();
    if has_google_ai_key {
        println!("\n  Note: found GOOGLE_AI_API_KEY — the gemini client reads GEMINI_API_KEY;");
        println!("  add `export GEMINI_API_KEY=$GOOGLE_AI_API_KEY` to your shell profile.");
    }

    if detected.iter().any(|d| d.auth == Some("oauth")) {
        println!(
            "\n  Gemini OAuth: run `nitpicker --gemini-oauth` to authenticate if not already done."
        );
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
