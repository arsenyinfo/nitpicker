use clap::{Args as ClapArgs, Parser, Subcommand};
use eyre::{Result, WrapErr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tracing::field::Empty;
use tracing::{Instrument, info_span};

use nitpicker_agent::{config, openrouter, tools::floor_char_boundary};

mod context;
mod debate;
mod detect;
#[cfg(feature = "antigravity")]
mod gemini_proxy;
mod output;
mod pr;
mod presets;
mod progress;
mod prompts;
mod proxy;
mod reflect;
mod review;
mod telemetry;

/// Flags shared across the default review mode and the subcommands. Declared once here and
/// marked `global`, so they are accepted before or after a subcommand and always land in
/// `Args.common` — per-subcommand copies would be independent namespaces, and a flag parsed
/// into the copy an arm doesn't read would be silently dropped.
#[derive(Debug, ClapArgs)]
struct CommonArgs {
    #[arg(long, global = true, default_value = ".")]
    repo: PathBuf,

    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[arg(long, short, global = true)]
    verbose: bool,

    /// Try the next configured reviewer when the selected model fails
    #[arg(long, global = true)]
    fallback: bool,

    /// Maximum wall-clock seconds for each parallel review job
    #[arg(long, global = true, value_parser = parse_positive_u64)]
    review_timeout_seconds: Option<u64>,
}

/// `--context-file`, kept out of the global `CommonArgs` deliberately: clap propagates a global
/// arg by keeping one winning occurrence list (the subcommand's), so a repeatable flag split
/// around the subcommand would silently drop the root's values. Instead this struct is flattened
/// at the root and into `ask`/`pr`, and the two vectors are concatenated root-first (= the
/// command-line order) at each use site.
#[derive(Debug, ClapArgs)]
struct ContextFileArgs {
    /// Read a file into the prompt verbatim; repeatable. Unlike the agents' own tools, this is not
    /// confined to the repo, so it can carry design notes or working docs that live outside it.
    #[arg(long = "context-file", value_name = "PATH")]
    context_file: Vec<PathBuf>,
}

fn merged_context_files(root: &ContextFileArgs, sub: &ContextFileArgs) -> Vec<PathBuf> {
    root.context_file
        .iter()
        .chain(&sub.context_file)
        .cloned()
        .collect()
}

/// `--preset`, shaped exactly like `ContextFileArgs` and for the same reason: a repeatable
/// flag must not be `global` (clap would keep only the subcommand's occurrence list), so it
/// is flattened at the root and into `pr`, and merged root-first at each use site.
#[derive(Debug, ClapArgs)]
struct PresetArgs {
    /// Review preset(s) to run — repeatable and comma-separated (e.g. --preset security,ml-rigor).
    /// Replaces the configured `[defaults].presets` list for this run.
    #[arg(long = "preset", value_name = "NAME", value_delimiter = ',')]
    preset: Vec<String>,
}

fn merged_presets(root: &PresetArgs, sub: &PresetArgs) -> Vec<String> {
    root.preset.iter().chain(&sub.preset).cloned().collect()
}

/// Presets pick review rubrics; `ask`/`init`/`reflect` have none. Root-position `--preset`
/// parses fine before any subcommand, so without an explicit rejection it would be silently
/// discarded there (`pr` and the default review arms consume it).
fn presets_allowed(command: &Option<Command>) -> bool {
    match command {
        None | Some(Command::Pr(_)) => true,
        Some(Command::Ask { .. } | Command::Init { .. } | Command::Reflect { .. }) => false,
    }
}

fn fallback_allowed(command: &Option<Command>) -> bool {
    matches!(command, None | Some(Command::Ask { .. } | Command::Pr(_)))
}

pub(crate) fn resolve_routing_modes(
    config: &config::Config,
    cli_alloy: bool,
    cli_fallback: bool,
) -> Result<(bool, bool)> {
    let alloy = cli_alloy || config.default_alloy();
    config.validate_alloy(alloy)?;
    let fallback = cli_fallback || config.default_fallback();
    config.validate_fallback(fallback)?;
    Ok((alloy, fallback))
}

#[derive(Debug, Parser)]
#[command(name = "nitpicker")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    common: CommonArgs,

    #[command(flatten)]
    context: ContextFileArgs,

    #[command(flatten)]
    presets: PresetArgs,

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
        /// Write to ~/.nitpicker/config.toml instead of <repo>/nitpicker.toml
        #[arg(long)]
        global: bool,

        /// Prefer OpenRouter experimental free models in the generated config
        #[arg(long)]
        free: bool,
    },
    /// Ask multiple LLM agents a free-form question about the codebase
    Ask {
        #[command(flatten)]
        context: ContextFileArgs,
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

/// Outcome of a run, mapped to the exit-code contract by [`finish`]: 0 = clean verdict,
/// 1 = hard failure (no verdict), 3 = degraded verdict (report printed, but at least one
/// reviewer or debate turn failed). 2 is deliberately unused — clap exits 2 on usage errors,
/// and the whole point is an unambiguous subprocess signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exit {
    Clean,
    Degraded,
    /// The failure was already reported to the consumer (a `pr --json` error envelope), so
    /// nothing more is printed.
    Failed,
}

impl Exit {
    pub(crate) fn from_degraded(degraded: bool) -> Self {
        match degraded {
            true => Self::Degraded,
            false => Self::Clean,
        }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();

    let verbose = args.common.verbose;
    let is_reflect = matches!(&args.command, Some(Command::Reflect { .. }));
    let default_level = if verbose || is_reflect {
        "info"
    } else {
        "warn"
    };
    let telemetry = telemetry::init(default_level);

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("Error: failed to start the async runtime: {err}");
            return ExitCode::from(1);
        }
    };
    let outcome = runtime.block_on(run(args));
    // the runtime goes first so no task is still producing spans while the exporter drains;
    // every guard inside `run` (checkout restore, PR lock) has already dropped with it
    drop(runtime);
    telemetry.shutdown();
    ExitCode::from(finish(outcome, &mut std::io::stdout()))
}

/// Map a run's outcome to its exit code, reporting what the consumer has not seen yet: a
/// degraded verdict gets its warning, an unreported error is rendered the way eyre's default
/// `main` handler would. The report on stdout is flushed explicitly so a piped consumer never
/// loses its tail; a flush failure is a hard failure, not a degraded verdict.
fn finish(outcome: Result<Exit>, stdout: &mut impl std::io::Write) -> u8 {
    match outcome {
        Ok(Exit::Clean) => 0,
        Ok(Exit::Failed) => 1,
        Ok(Exit::Degraded) => match stdout.flush() {
            Ok(()) => {
                eprintln!(
                    "warning: degraded verdict — a reviewer or debate turn failed (exit code 3)"
                );
                3
            }
            Err(err) => {
                eprintln!("error: failed to flush report to stdout: {err}");
                1
            }
        },
        Err(err) => {
            eprintln!("Error: {err:?}");
            1
        }
    }
}

/// The root span of every run; the orchestration spans below it are per-mode.
async fn run(args: Args) -> Result<Exit> {
    let command = match &args.command {
        None => "review",
        Some(Command::Init { .. }) => "init",
        Some(Command::Ask { .. }) => "ask",
        Some(Command::Pr(_)) => "pr",
        Some(Command::Reflect { .. }) => "reflect",
    };
    let span = info_span!(
        "nitpicker.run",
        otel.status_code = Empty,
        nitpicker.command = command,
        nitpicker.degraded = Empty,
        nitpicker.pr.number = Empty,
    );
    let outcome = dispatch(args).instrument(span.clone()).await;
    match &outcome {
        Ok(Exit::Clean) => {}
        Ok(Exit::Degraded) => {
            span.record("nitpicker.degraded", true);
        }
        Ok(Exit::Failed) | Err(_) => {
            span.record("otel.status_code", "ERROR");
        }
    }
    outcome
}

async fn dispatch(args: Args) -> Result<Exit> {
    if !presets_allowed(&args.command) && !args.presets.preset.is_empty() {
        eyre::bail!("--preset applies to review modes only (default review, --analyze, pr)");
    }
    if args.common.fallback && !fallback_allowed(&args.command) {
        eyre::bail!("--fallback applies to review and ask modes only");
    }

    // note: no json panic hook. reviewer work runs in tokio::spawn tasks whose
    // panics are caught as JoinError and folded into a degraded report (exit 3
    // for review/ask, status ok for pr); a process-wide hook would double-emit
    // there. a genuine top-level panic aborts non-zero with a stderr message,
    // which is an acceptable catastrophic-failure signal for the consumer.
    match args.command {
        Some(Command::Init { global, free }) => {
            // the global flags parse here too; `init` generates a config, so honoring --config
            // would be nonsense — reject it rather than silently ignore it
            if args.common.config.is_some() {
                eyre::bail!("--config has no effect on init, which generates a config file");
            }
            let path = init_config_path(global, &args.common.repo)?;
            if path.exists() {
                eyre::bail!("{} already exists", path.display());
            }
            run_init(path, free).await?;
            return Ok(Exit::Clean);
        }
        Some(Command::Ask {
            context,
            topic,
            no_debate,
            alloy,
            rounds,
            max_turns,
        }) => {
            let repo = resolve_repo_root(&args.common.repo)?;
            let mut config = load_config(args.common.config.as_deref(), &repo)?;
            // CLI-only routing validation must precede free-model smoke completions.
            let (use_alloy, use_fallback) =
                resolve_routing_modes(&config, alloy, args.common.fallback)?;
            finalize_routing_config(&mut config, use_fallback).await?;
            let config = config;
            let topic = context::append_to_prompt(
                topic,
                &context::load_context_files(&merged_context_files(&args.context, &context))?,
            );
            let max_turns = config.max_turns(max_turns)?;

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
                        verbose: args.common.verbose,
                        task: prompts::RunTask::Ask,
                        alloy: use_alloy,
                        fallback: use_fallback,
                        format: output::OutputFormat::Text,
                    },
                )
                .await?;
                println!("{}", outcome.report);
                if args.common.verbose {
                    eprintln!(
                        "\nTranscript saved to: {}",
                        outcome.transcript_path.display()
                    );
                }
                return Ok(Exit::from_degraded(outcome.degraded));
            }

            let outcome = review::run_review(
                &repo,
                &topic,
                &config,
                review::ReviewOptions {
                    max_turns,
                    timeout_seconds: args.common.review_timeout_seconds,
                    verbose: args.common.verbose,
                    task: prompts::RunTask::Ask,
                    fallback: use_fallback,
                    format: output::OutputFormat::Text,
                },
            )
            .await?;
            println!("{}", outcome.report);
            return Ok(Exit::from_degraded(outcome.degraded));
        }
        Some(Command::Pr(pr_args)) => {
            let context_files = merged_context_files(&args.context, &pr_args.context);
            let preset_names = merged_presets(&args.presets, &pr_args.presets);
            // config loading happens inside run_pr so its failures honor --format json too
            return pr::run_pr(pr_args, args.common, context_files, preset_names).await;
        }
        Some(Command::Reflect { sessions_dir, n }) => {
            let repo = resolve_repo_root(&args.common.repo)?;
            let config = load_resolved_config(args.common.config.as_deref(), &repo).await?;
            reflect::run_reflect(reflect::ReflectArgs {
                sessions_dir,
                n,
                repo,
                config,
            })
            .await?;
            return Ok(Exit::Clean);
        }
        None => {}
    }

    let repo = resolve_repo_root(&args.common.repo)?;

    let mut config = load_config(args.common.config.as_deref(), &repo)?;
    // Resolve presets and CLI-only routing validation before free-model resolution: pure usage
    // errors must fail before any network call (the resolver can run live smoke completions).
    let presets = presets::resolve(&args.presets.preset, &config)?;
    let (use_alloy, use_fallback) =
        resolve_routing_modes(&config, args.alloy, args.common.fallback)?;
    finalize_routing_config(&mut config, use_fallback).await?;
    let config = config;
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
    let prompt = context::append_to_prompt(
        prompt,
        &context::load_context_files(&args.context.context_file)?,
    );

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
                task: prompts::RunTask::Review {
                    scope,
                    presets: &presets,
                },
                alloy: use_alloy,
                fallback: use_fallback,
                format: output::OutputFormat::Text,
            },
        )
        .await?;
        println!("{}", outcome.report);
        if args.common.verbose {
            eprintln!(
                "\nTranscript saved to: {}",
                outcome.transcript_path.display()
            );
        }
        Ok(Exit::from_degraded(outcome.degraded))
    } else {
        let outcome = review::run_review(
            &repo,
            &prompt,
            &config,
            review::ReviewOptions {
                max_turns,
                timeout_seconds: args.common.review_timeout_seconds,
                verbose: args.common.verbose,
                task: prompts::RunTask::Review {
                    scope,
                    presets: &presets,
                },
                fallback: use_fallback,
                format: output::OutputFormat::Text,
            },
        )
        .await?;
        println!("{}", outcome.report);
        Ok(Exit::from_degraded(outcome.degraded))
    }
}

pub(crate) fn load_config(explicit_path: Option<&Path>, repo: &Path) -> Result<config::Config> {
    let config: config::Config = if let Some(path) = explicit_path {
        let content = std::fs::read_to_string(path)
            .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", path))?;
        toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"))?
    } else if repo.join("nitpicker.toml").exists() {
        let path = repo.join("nitpicker.toml");
        let content = std::fs::read_to_string(&path)
            .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", path))?;
        toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"))?
    } else {
        return load_global_config();
    };
    config.validate_structure()?;
    Ok(config)
}

/// Resolve the worktree containing `path` through Git itself. Linked worktrees and checked-out
/// submodules represent `.git` as a file, so its filesystem shape is not a repository invariant.
pub(crate) fn git_worktree_root(path: &Path) -> Option<PathBuf> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let root = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
    root.canonicalize().ok()
}

fn resolve_repo_root(path: &Path) -> Result<PathBuf> {
    let canonical = path
        .canonicalize()
        .wrap_err("failed to canonicalize --repo path")?;
    git_worktree_root(&canonical)
        .ok_or_else(|| eyre::eyre!("--repo must point inside a Git worktree"))
}

/// The `~/.nitpicker/config.toml` fallback alone — `pr` mode reaches for this directly,
/// since its repo-level config comes from the PR base branch blob, never the working tree.
pub(crate) fn load_global_config() -> Result<config::Config> {
    let path = dirs::home_dir()
        .map(|home| home.join(".nitpicker").join("config.toml"))
        .filter(|path| path.exists())
        .ok_or_else(|| {
            eyre::eyre!("no config found — run `nitpicker init [--global]` to generate one")
        })?;
    let content = std::fs::read_to_string(&path)
        .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", path))?;
    let config: config::Config =
        toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"))?;
    config.validate_structure()?;
    Ok(config)
}

pub(crate) async fn load_resolved_config(
    explicit_path: Option<&Path>,
    repo: &Path,
) -> Result<config::Config> {
    let mut config = load_config(explicit_path, repo)?;
    finalize_routing_config(&mut config, false).await?;
    Ok(config)
}

/// Finish config validation and experimental route resolution after the caller has resolved the
/// effective CLI/config fallback mode. Strict execution requires every credential up front;
/// fallback execution lets route construction skip unusable entries.
pub(crate) async fn finalize_routing_config(
    config: &mut config::Config,
    fallback: bool,
) -> Result<()> {
    if !fallback {
        config.validate_credentials()?;
    }
    openrouter::resolve_free_models_with_fallback(config, fallback).await
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

    // Fallback needs a second route even when debate is disabled. OpenRouter free selection can
    // produce two distinct model routes from its one detected credential.
    let reviewer_slots = if detected.len() >= 2 || prefer_openrouter_free {
        2
    } else {
        1
    };
    let reviewers = pick_reviewers(detected, reviewer_slots, prefer_openrouter_free);
    let fallback = reviewers.len() >= 2;

    config::Config {
        defaults: Some(config::DefaultsConfig {
            debate: Some(debate),
            alloy: None,
            fallback: Some(fallback),
            max_turns: Some(config::DEFAULT_MAX_TURNS),
            compact_threshold: Some(100_000),
            log_trajectories: Some(false),
            presets: None,
        }),
        aggregator,
        reviewer: reviewers,
        presets: None,
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
        max_tokens: None,
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

fn init_config_path(global: bool, repo: &Path) -> Result<PathBuf> {
    if global {
        let home =
            dirs::home_dir().ok_or_else(|| eyre::eyre!("failed to resolve home directory"))?;
        Ok(home.join(".nitpicker").join("config.toml"))
    } else {
        // --repo is a global flag, so `init --repo <dir>` must target that repo's root
        // rather than silently writing into the cwd
        Ok(repo.join("nitpicker.toml"))
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

fn parse_positive_u64(value: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("invalid positive integer: {value}"))?;

    if parsed == 0 {
        return Err("value must be greater than 0".to_string());
    }

    Ok(parsed)
}

// Describes only the target and optional user instructions — the investigation angles come
// from the resolved presets, not from a focus list baked in here.
fn build_analysis_prompt(path: Option<&Path>, custom_prompt: Option<&str>) -> String {
    let target = match path {
        Some(p) => format!("`{}`", p.display()),
        None => "the entire repository".to_string(),
    };
    let base = format!(
        "Analyze the following code for issues and improvement opportunities:\n\
         - Target: {}",
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

const MAX_SNAPSHOT_FILES: usize = 500;
const MAX_SNAPSHOT_BYTES: usize = 64 * 1024;

pub fn detect_diff_context(repo: &Path) -> Result<String> {
    let branch = run_git(repo, &["rev-parse", "--abbrev-ref", "HEAD"])?;
    let branch = branch.trim();

    if branch == "HEAD" {
        eyre::bail!("detached HEAD state: checkout a branch before running nitpicker");
    }

    let base = detect_base_branch(repo);
    let head_sha = run_git(repo, &["rev-parse", "HEAD"])?.trim().to_string();
    let working_tree = run_git(repo, &["status", "--short"])?;
    let working_tree_files = nonempty_lines(&working_tree);
    let has_uncommitted = !working_tree_files.is_empty();

    let base_snapshot = match base.as_ref() {
        Some(base) => {
            let base_sha = run_git(repo, &["rev-parse", &base.revision])?
                .trim()
                .to_string();
            let merge_base = run_git_optional(repo, &["merge-base", &base_sha, &head_sha])?
                .map(|sha| sha.trim().to_string())
                .filter(|sha| !sha.is_empty());
            let comparison_base = merge_base.as_deref().unwrap_or(&base_sha);
            let committed = run_git(
                repo,
                &[
                    "diff",
                    "--name-status",
                    "-M",
                    comparison_base,
                    &head_sha,
                    "--",
                ],
            )?;
            Some((base, base_sha, merge_base, nonempty_lines(&committed)))
        }
        None => None,
    };
    let has_committed_changes = base_snapshot
        .as_ref()
        .is_some_and(|(_, _, _, files)| !files.is_empty());

    if !has_uncommitted && !has_committed_changes {
        if let Some(base) = base.as_ref() {
            eyre::bail!(
                "no changes to review: no uncommitted changes and no committed changes vs {}",
                base.name
            );
        }
        eyre::bail!(
            "no changes to review: no uncommitted changes and no detectable base branch commits"
        );
    }

    let mut snapshot = format!(
        "Review the following changes. This review snapshot was captured once before reviewers \
fan out; treat its revisions and file maps as immutable orientation for this run.\n\n\
## Frozen review snapshot\n\
- Branch: `{branch}`\n\
- HEAD: `{head_sha}`\n"
    );

    match base_snapshot.as_ref() {
        Some((base, base_sha, merge_base, _)) => {
            let comparison_base = merge_base.as_deref().unwrap_or(base_sha);
            snapshot.push_str(&format!(
                "- Base: `{}` @ `{base_sha}`\n\
                 - Merge base: {}\n\
                 - Committed comparison: `git diff {comparison_base} {head_sha} --`\n",
                base.revision,
                merge_base
                    .as_deref()
                    .map(|sha| format!("`{sha}`"))
                    .unwrap_or_else(
                        || "unavailable; using a direct base/HEAD tree comparison".to_string()
                    )
            ));
        }
        None => {
            snapshot.push_str("- Base: not detected\n- Merge base: unavailable\n");
        }
    }

    snapshot.push_str(&format!(
        "- Working tree: {}\n\
         - Tracked working-tree comparison: `git diff {head_sha} --`\n\
         - Untracked files: included in the status map below\n",
        if has_uncommitted { "dirty" } else { "clean" }
    ));

    append_snapshot_file_sections(
        &mut snapshot,
        base_snapshot
            .as_ref()
            .map(|(_, _, _, committed_files)| committed_files.as_slice()),
        &working_tree_files,
    );

    if snapshot.len() > MAX_SNAPSHOT_BYTES {
        snapshot.truncate(floor_char_boundary(&snapshot, MAX_SNAPSHOT_BYTES));
    }

    Ok(snapshot)
}

fn nonempty_lines(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect()
}

fn render_snapshot_files(files: &[String], budget: usize) -> String {
    if files.is_empty() {
        return "(none)\n".to_string();
    }

    let mut text = String::new();
    let mut included = 0;
    for file in files.iter().take(MAX_SNAPSHOT_FILES) {
        let line = format!("- `{file}`\n");
        let next_included = included + 1;
        let omitted = files.len().saturating_sub(next_included);
        let marker = snapshot_omission_marker(omitted);
        if text.len() + line.len() + marker.len() > budget {
            break;
        }
        text.push_str(&line);
        included = next_included;
    }

    let omitted = files.len().saturating_sub(included);
    let marker = snapshot_omission_marker(omitted);
    if text.len() + marker.len() <= budget {
        text.push_str(&marker);
    }
    text
}

fn snapshot_omission_marker(omitted: usize) -> String {
    match omitted {
        0 => String::new(),
        count => format!(
            "- … {count} additional files omitted from the prompt; inspect the pinned comparison if needed\n"
        ),
    }
}

fn append_snapshot_file_sections(
    output: &mut String,
    committed_files: Option<&[String]>,
    working_tree_files: &[String],
) {
    const COMMITTED_HEADING: &str = "\n### Committed changed files (name-status)\n";
    const WORKING_TREE_HEADING: &str = "\n### Uncommitted changed files (`git status --short`)\n";
    const NO_BASE: &str = "(comparison unavailable: no base detected)\n";

    let fixed_len = COMMITTED_HEADING.len() + WORKING_TREE_HEADING.len();
    let content_budget = MAX_SNAPSHOT_BYTES
        .saturating_sub(output.len())
        .saturating_sub(fixed_len);
    let committed_budget = content_budget / 2;
    let working_tree_budget = content_budget.saturating_sub(committed_budget);

    let committed = committed_files
        .map(|files| render_snapshot_files(files, committed_budget))
        .unwrap_or_else(|| NO_BASE.to_string());
    let working_tree = render_snapshot_files(working_tree_files, working_tree_budget);

    output.push_str(COMMITTED_HEADING);
    output.push_str(&committed);
    output.push_str(WORKING_TREE_HEADING);
    output.push_str(&working_tree);
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
    let remote = format!("refs/remotes/origin/{branch}");
    let has_local = run_git(repo, &["rev-parse", "--verify", &local]).is_ok();
    let has_remote = run_git(repo, &["rev-parse", "--verify", &remote]).is_ok();
    let use_remote = match (has_local, has_remote) {
        (false, false) => return None,
        (true, false) => false,
        (false, true) => true,
        (true, true) => {
            let behind = local_base_is_behind_remote(repo, &local, &remote);
            if behind {
                tracing::warn!(
                    "local {branch} is behind origin/{branch}; using origin/{branch} as the review base"
                );
            }
            behind
        }
    };
    Some(BaseBranch {
        name: branch.to_string(),
        revision: match use_remote {
            true => format!("origin/{branch}"),
            false => branch.to_string(),
        },
    })
}

/// A branch cut from `origin/main` while local `main` lags behind it has an older merge-base with
/// local `main`, so the "committed comparison" would include upstream commits that are not part
/// of the change. Prefer the candidate whose merge-base with HEAD is the newer one; equal or
/// diverged merge-bases keep the local ref. No network involved: only remote-tracking refs are
/// consulted.
fn local_base_is_behind_remote(repo: &Path, local: &str, remote: &str) -> bool {
    let merge_base = |candidate: &str| {
        run_git(repo, &["merge-base", "HEAD", candidate]).map(|sha| sha.trim().to_string())
    };
    match (merge_base(local), merge_base(remote)) {
        (Ok(local_base), Ok(remote_base)) if local_base != remote_base => run_git(
            repo,
            &["merge-base", "--is-ancestor", &local_base, &remote_base],
        )
        .is_ok(),
        _ => false,
    }
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

/// Some Git plumbing uses exit 1 for a valid negative result. Preserve every other failure.
fn run_git_optional(repo: &Path, args: &[&str]) -> Result<Option<String>> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()?;
    match output.status.code() {
        Some(0) => Ok(Some(String::from_utf8(output.stdout)?)),
        Some(1) => Ok(None),
        _ => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eyre::bail!("git {}: {}", args.join(" "), stderr.trim());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    struct FailingFlush;

    impl std::io::Write for FailingFlush {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("closed pipe"))
        }
    }

    /// 0 / 1 / 3 is a subprocess contract; a degraded verdict whose report cannot be flushed is
    /// a hard failure, not a degraded one, and an already-reported failure adds no output.
    #[test]
    fn exit_codes_follow_the_contract() {
        assert_eq!(finish(Ok(Exit::Clean), &mut Vec::new()), 0);
        assert_eq!(finish(Ok(Exit::Degraded), &mut Vec::new()), 3);
        assert_eq!(finish(Ok(Exit::Failed), &mut Vec::new()), 1);
        assert_eq!(finish(Err(eyre::eyre!("boom")), &mut Vec::new()), 1);
        assert_eq!(finish(Ok(Exit::Degraded), &mut FailingFlush), 1);
        assert_eq!(finish(Ok(Exit::Clean), &mut FailingFlush), 0);
    }

    #[test]
    fn cli_definition_is_valid() {
        Args::command().debug_assert();
    }

    fn parse(argv: &[&str]) -> Args {
        Args::try_parse_from(argv).expect("argv parses")
    }

    fn ask_context(args: &Args) -> Vec<PathBuf> {
        match &args.command {
            Some(Command::Ask { context, .. }) => merged_context_files(&args.context, context),
            _ => panic!("expected ask subcommand"),
        }
    }

    fn pr_context(args: &Args) -> Vec<PathBuf> {
        match &args.command {
            Some(Command::Pr(pr_args)) => merged_context_files(&args.context, &pr_args.context),
            _ => panic!("expected pr subcommand"),
        }
    }

    #[test]
    fn context_file_before_the_subcommand_reaches_ask() {
        let args = parse(&["nitpicker", "--context-file", "/a", "ask", "topic"]);
        assert_eq!(ask_context(&args), [PathBuf::from("/a")]);
    }

    #[test]
    fn context_file_after_the_subcommand_reaches_ask() {
        let args = parse(&["nitpicker", "ask", "--context-file", "/a", "topic"]);
        assert_eq!(ask_context(&args), [PathBuf::from("/a")]);
    }

    #[test]
    fn context_files_split_around_the_subcommand_merge_in_cli_order() {
        let args = parse(&[
            "nitpicker",
            "--context-file",
            "/a",
            "ask",
            "--context-file",
            "/b",
            "topic",
        ]);
        assert_eq!(
            ask_context(&args),
            [PathBuf::from("/a"), PathBuf::from("/b")]
        );

        let args = parse(&[
            "nitpicker",
            "--context-file",
            "/a",
            "pr",
            "--context-file",
            "/b",
        ]);
        assert_eq!(
            pr_context(&args),
            [PathBuf::from("/a"), PathBuf::from("/b")]
        );
    }

    #[test]
    fn global_scalars_land_in_common_from_either_side_of_the_subcommand() {
        let args = parse(&["nitpicker", "-v", "--repo", "/x", "ask", "topic"]);
        assert!(args.common.verbose);
        assert_eq!(args.common.repo, PathBuf::from("/x"));

        let args = parse(&["nitpicker", "ask", "topic", "--repo", "/x", "-v"]);
        assert!(args.common.verbose);
        assert_eq!(args.common.repo, PathBuf::from("/x"));

        let args = parse(&["nitpicker", "pr", "--repo", "/x", "--config", "/c.toml"]);
        assert!(!args.common.verbose);
        assert_eq!(args.common.repo, PathBuf::from("/x"));
        assert_eq!(args.common.config, Some(PathBuf::from("/c.toml")));

        let args = parse(&["nitpicker", "--fallback", "ask", "topic"]);
        assert!(args.common.fallback);
        let args = parse(&["nitpicker", "pr", "--fallback"]);
        assert!(args.common.fallback);

        let args = parse(&["nitpicker", "--review-timeout-seconds", "420", "pr"]);
        assert_eq!(args.common.review_timeout_seconds, Some(420));
        let args = parse(&["nitpicker", "pr", "--review-timeout-seconds", "420"]);
        assert_eq!(args.common.review_timeout_seconds, Some(420));
    }

    #[test]
    fn review_timeout_must_be_positive() {
        let error =
            Args::try_parse_from(["nitpicker", "--no-debate", "--review-timeout-seconds", "0"])
                .unwrap_err();

        assert!(error.to_string().contains("value must be greater than 0"));
    }

    #[test]
    fn init_writes_into_the_repo_named_by_the_global_repo_flag() {
        let path = init_config_path(false, Path::new("/some/repo")).unwrap();
        assert_eq!(path, PathBuf::from("/some/repo/nitpicker.toml"));
    }

    fn detected_provider(name: &'static str, provider: &'static str) -> detect::Detected {
        detect::Detected {
            name,
            provider,
            model: format!("{name}-model"),
            base_url: None,
            api_key_env: None,
            auth: None,
            source: "test",
            local_server: false,
        }
    }

    #[test]
    fn init_enables_fallback_when_it_can_generate_two_routes() {
        let first = detected_provider("first", "openai");
        let second = detected_provider("second", "anthropic");
        let config = build_init_config(&[&first, &second], false);

        assert_eq!(config.reviewer.len(), 2);
        assert!(config.default_fallback());
        assert!(
            toml::to_string_pretty(&config)
                .unwrap()
                .contains("fallback = true")
        );
    }

    #[test]
    fn free_init_generates_two_fallback_routes_from_openrouter() {
        let openrouter = detected_provider("openrouter", "openrouter");
        let config = build_init_config(&[&openrouter], true);

        assert_eq!(config.reviewer.len(), 2);
        assert!(config.reviewer.iter().all(|route| route.model == "free"));
        assert!(config.default_fallback());
        assert!(
            toml::to_string_pretty(&config)
                .unwrap()
                .contains("fallback = true")
        );
    }

    #[test]
    fn init_does_not_enable_impossible_single_route_fallback() {
        let only = detected_provider("only", "openai");
        let config = build_init_config(&[&only], false);

        assert_eq!(config.reviewer.len(), 1);
        assert!(!config.default_fallback());
    }

    #[test]
    fn git_discovers_primary_and_linked_worktree_roots() {
        let dir = tempfile::tempdir().unwrap();
        let primary = dir.path().join("primary");
        std::fs::create_dir(&primary).unwrap();
        run_git(&primary, &["init", "-b", "main"]).unwrap();
        run_git(
            &primary,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--allow-empty",
                "-m",
                "initial",
            ],
        )
        .unwrap();

        let linked = dir.path().join("linked");
        run_git(
            &primary,
            &["worktree", "add", "-b", "feature", linked.to_str().unwrap()],
        )
        .unwrap();

        let primary = primary.canonicalize().unwrap();
        let linked = linked.canonicalize().unwrap();
        assert!(primary.join(".git").is_dir());
        assert!(linked.join(".git").is_file());
        assert_eq!(git_worktree_root(&primary).as_deref(), Some(&*primary));
        assert_eq!(git_worktree_root(&linked).as_deref(), Some(&*linked));

        let nested = linked.join("nested");
        std::fs::create_dir(&nested).unwrap();
        assert_eq!(git_worktree_root(&nested).as_deref(), Some(&*linked));

        let fake = dir.path().join("fake");
        std::fs::create_dir(&fake).unwrap();
        std::fs::write(fake.join(".git"), "not a gitdir pointer").unwrap();
        assert!(git_worktree_root(&fake).is_none());
    }

    #[test]
    fn base_branch_prefers_origin_only_when_local_is_behind_it() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path();
        let commit = |msg: &str| {
            run_git(
                repo,
                &[
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "--allow-empty",
                    "-m",
                    msg,
                ],
            )
            .unwrap()
        };
        let sha = || {
            run_git(repo, &["rev-parse", "HEAD"])
                .unwrap()
                .trim()
                .to_string()
        };
        run_git(repo, &["init", "-b", "main"]).unwrap();
        commit("a");
        let a = sha();
        commit("b");
        let b = sha();
        run_git(repo, &["switch", "-c", "feature"]).unwrap();
        commit("c");
        let revision = |remote: Option<&str>| {
            match remote {
                Some(sha) => run_git(repo, &["update-ref", "refs/remotes/origin/main", sha]),
                None => run_git(repo, &["update-ref", "-d", "refs/remotes/origin/main"]),
            }
            .unwrap();
            resolve_base_branch(repo, "main").unwrap().revision
        };

        // feature was cut from b, but local main was never fast-forwarded past a
        run_git(repo, &["update-ref", "refs/heads/main", &a]).unwrap();
        assert_eq!(revision(Some(&b)), "origin/main");
        assert_eq!(revision(Some(&a)), "main");
        assert_eq!(revision(None), "main");

        // local main ahead of, or equal to, origin keeps the local ref
        run_git(repo, &["update-ref", "refs/heads/main", &b]).unwrap();
        assert_eq!(revision(Some(&a)), "main");
        assert_eq!(revision(Some(&b)), "main");
    }

    #[test]
    fn diff_context_is_a_frozen_revision_and_file_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path();
        run_git(repo, &["init", "-b", "main"]).unwrap();
        std::fs::write(repo.join("base.txt"), "base\n").unwrap();
        run_git(repo, &["add", "base.txt"]).unwrap();
        run_git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "initial",
            ],
        )
        .unwrap();
        let base_sha = run_git(repo, &["rev-parse", "HEAD"])
            .unwrap()
            .trim()
            .to_string();

        run_git(repo, &["switch", "-c", "feature"]).unwrap();
        std::fs::write(repo.join("committed.txt"), "committed\n").unwrap();
        run_git(repo, &["add", "committed.txt"]).unwrap();
        run_git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "feature",
            ],
        )
        .unwrap();
        let head_sha = run_git(repo, &["rev-parse", "HEAD"])
            .unwrap()
            .trim()
            .to_string();
        std::fs::write(repo.join("committed.txt"), "working tree\n").unwrap();
        std::fs::write(repo.join("untracked.txt"), "untracked\n").unwrap();

        let context = detect_diff_context(repo).unwrap();
        for expected in [
            "## Frozen review snapshot".to_string(),
            "- Branch: `feature`".to_string(),
            format!("- HEAD: `{head_sha}`"),
            format!("- Base: `main` @ `{base_sha}`"),
            format!("- Merge base: `{base_sha}`"),
            format!("git diff {base_sha} {head_sha} --"),
            "- Working tree: dirty".to_string(),
            "- `A\tcommitted.txt`".to_string(),
            "- ` M committed.txt`".to_string(),
            "- `?? untracked.txt`".to_string(),
        ] {
            assert!(
                context.contains(&expected),
                "missing {expected:?}:\n{context}"
            );
        }
    }

    #[test]
    fn diff_context_falls_back_to_direct_tree_comparison_for_unrelated_histories() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path();
        run_git(repo, &["init", "-b", "main"]).unwrap();
        std::fs::write(repo.join("base.txt"), "base\n").unwrap();
        run_git(repo, &["add", "base.txt"]).unwrap();
        run_git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "base",
            ],
        )
        .unwrap();

        run_git(repo, &["switch", "--orphan", "feature"]).unwrap();
        std::fs::write(repo.join("feature.txt"), "feature\n").unwrap();
        run_git(repo, &["add", "--all"]).unwrap();
        run_git(
            repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "feature",
            ],
        )
        .unwrap();

        assert!(run_git(repo, &["merge-base", "main", "HEAD"]).is_err());
        let context = detect_diff_context(repo).unwrap();
        assert!(context.len() <= MAX_SNAPSHOT_BYTES);
    }

    #[test]
    fn snapshot_file_maps_share_a_fixed_rendered_byte_budget() {
        let files = (0..1_000)
            .map(|index| format!("{index}-{}", "x".repeat(4_000)))
            .collect::<Vec<_>>();
        let rendered = render_snapshot_files(&files, 16 * 1024);
        assert!(rendered.len() <= 16 * 1024);
        let rendered_files = rendered
            .lines()
            .filter(|line| line.starts_with("- `"))
            .count();
        assert!(rendered_files < files.len());
        let omitted = files.len() - rendered_files;
        assert!(
            rendered
                .lines()
                .last()
                .is_some_and(|marker| marker.contains(&omitted.to_string()))
        );

        let mut snapshot = String::new();
        append_snapshot_file_sections(&mut snapshot, Some(&files), &files);
        assert!(snapshot.len() <= MAX_SNAPSHOT_BYTES);
    }

    #[test]
    fn subcommands_without_context_files_reject_the_flag() {
        for argv in [
            ["nitpicker", "reflect", "--context-file", "/a"],
            ["nitpicker", "init", "--context-file", "/a"],
        ] {
            assert!(Args::try_parse_from(argv).is_err());
        }
    }

    fn pr_presets(args: &Args) -> Vec<String> {
        match &args.command {
            Some(Command::Pr(pr_args)) => merged_presets(&args.presets, &pr_args.presets),
            _ => panic!("expected pr subcommand"),
        }
    }

    #[test]
    fn preset_reaches_pr_from_either_side_of_the_subcommand() {
        let args = parse(&["nitpicker", "--preset", "security", "pr"]);
        assert_eq!(pr_presets(&args), ["security"]);

        let args = parse(&["nitpicker", "pr", "--preset", "security"]);
        assert_eq!(pr_presets(&args), ["security"]);
    }

    /// Repeated flags append, commas split within one occurrence, and values split around
    /// the subcommand merge root-first (= command-line order) — same contract as
    /// `--context-file`, and the reason `--preset` is not a clap `global`.
    #[test]
    fn presets_split_around_the_subcommand_merge_in_cli_order_with_commas_expanded() {
        let args = parse(&[
            "nitpicker",
            "--preset",
            "security,ml-rigor",
            "pr",
            "--preset",
            "tone",
        ]);
        assert_eq!(pr_presets(&args), ["security", "ml-rigor", "tone"]);
    }

    #[test]
    fn repeated_preset_flags_append_on_the_root_review_path() {
        let args = parse(&["nitpicker", "--preset", "security", "--preset", "tone"]);
        assert!(args.command.is_none());
        assert_eq!(args.presets.preset, ["security", "tone"]);
    }

    #[test]
    fn subcommands_without_presets_reject_the_flag() {
        let cases: [&[&str]; 3] = [
            &["nitpicker", "ask", "topic", "--preset", "security"],
            &["nitpicker", "reflect", "--preset", "security"],
            &["nitpicker", "init", "--preset", "security"],
        ];
        for argv in cases {
            assert!(Args::try_parse_from(argv).is_err(), "argv: {argv:?}");
        }
    }

    /// Root-position `--preset` parses before any subcommand, so the non-review arms must
    /// reject it explicitly instead of silently discarding it.
    #[test]
    fn root_position_presets_are_rejected_for_non_review_subcommands() {
        let cases: [&[&str]; 3] = [
            &["nitpicker", "--preset", "security", "ask", "topic"],
            &["nitpicker", "--preset", "security", "init"],
            &["nitpicker", "--preset", "security", "reflect"],
        ];
        for argv in cases {
            let args = parse(argv);
            assert!(!presets_allowed(&args.command), "argv: {argv:?}");
        }

        let args = parse(&["nitpicker", "--preset", "security", "pr"]);
        assert!(presets_allowed(&args.command));
        let args = parse(&["nitpicker", "--preset", "security"]);
        assert!(presets_allowed(&args.command));
    }

    #[test]
    fn fallback_is_scoped_to_review_and_ask_commands() {
        for argv in [
            &["nitpicker", "--fallback"][..],
            &["nitpicker", "--fallback", "ask", "topic"][..],
            &["nitpicker", "pr", "--fallback"][..],
        ] {
            let args = parse(argv);
            assert!(fallback_allowed(&args.command), "argv: {argv:?}");
        }
        for argv in [
            &["nitpicker", "init", "--fallback"][..],
            &["nitpicker", "reflect", "--fallback"][..],
        ] {
            let args = parse(argv);
            assert!(!fallback_allowed(&args.command), "argv: {argv:?}");
        }
    }

    #[test]
    fn routing_modes_reject_cli_pooling_with_one_reviewer() {
        let config: config::Config = toml::from_str(
            r#"
                [aggregator]
                model = "m"
                provider = "openai"
                auth = "codex"

                [[reviewer]]
                model = "m"
                provider = "openai"
                auth = "codex"
            "#,
        )
        .unwrap();

        let err = resolve_routing_modes(&config, false, true).unwrap_err();
        assert!(format!("{err:#}").contains("requires at least 2 reviewers"));
        let err = resolve_routing_modes(&config, true, false).unwrap_err();
        assert!(format!("{err:#}").contains("--alloy requires at least 2 reviewers"));
        assert_eq!(
            resolve_routing_modes(&config, false, false).unwrap(),
            (false, false)
        );
    }

    /// The config file shape for presets: `[presets.<name>]` tables and the
    /// `[defaults].presets` selection list round-trip through the library's Config.
    #[test]
    fn preset_config_tables_parse_and_validate() {
        let toml_str = r#"
            [defaults]
            presets = ["tone", "security"]

            [aggregator]
            model = "m"
            provider = "openai"
            auth = "codex"

            [[reviewer]]
            name = "r"
            model = "m"
            provider = "openai"
            auth = "codex"

            [presets.tone]
            prompt = "review the docs for tone"
        "#;
        let config: config::Config = toml::from_str(toml_str).expect("parses");
        config.validate().expect("validates");
        let defaults = config.defaults.as_ref().expect("defaults present");
        assert_eq!(
            defaults.presets.as_deref(),
            Some(&["tone".to_string(), "security".to_string()][..])
        );
        let presets = config.presets.as_ref().expect("presets present");
        assert_eq!(presets["tone"].prompt, "review the docs for tone");
    }

    #[test]
    fn unknown_fields_inside_a_preset_table_are_rejected() {
        let toml_str = r#"
            [aggregator]
            model = "m"
            provider = "openai"
            auth = "codex"

            [[reviewer]]
            model = "m"
            provider = "openai"
            auth = "codex"

            [presets.tone]
            prompt = "p"
            model = "sneaky-per-preset-model"
        "#;
        assert!(toml::from_str::<config::Config>(toml_str).is_err());
    }

    #[test]
    fn blank_preset_prompts_fail_validation() {
        let toml_str = r#"
            [aggregator]
            model = "m"
            provider = "openai"
            auth = "codex"

            [[reviewer]]
            model = "m"
            provider = "openai"
            auth = "codex"

            [presets.tone]
            prompt = "   "
        "#;
        let config: config::Config = toml::from_str(toml_str).expect("parses");
        let err = config.validate().expect_err("blank prompt");
        assert!(format!("{err:#}").contains("[presets.tone].prompt"));
    }
}
