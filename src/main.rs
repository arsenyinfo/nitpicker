use clap::{Args as ClapArgs, Parser, Subcommand};
use eyre::Result;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

mod agent;
mod config;
mod debate;
mod gemini_proxy;
mod llm;
mod pr;
mod prompts;
mod review;
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

    /// Use actor-critic debate instead of parallel aggregation
    #[arg(long)]
    debate: bool,

    /// Maximum debate rounds (only with --debate)
    #[arg(long, default_value = "5")]
    rounds: usize,
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
        /// Use actor-critic debate instead of parallel aggregation
        #[arg(long)]
        debate: bool,
        /// Maximum debate rounds (only with --debate)
        #[arg(long, default_value = "5")]
        rounds: usize,
    },
    /// Review a GitHub PR (current branch's PR or a remote PR by URL)
    Pr(pr::PrArgs),
}

const INIT_TEMPLATE: &str = r#"[aggregator]
model = "claude-sonnet-4-6"
provider = "anthropic"

[defaults]
debate = false

[[reviewer]]
name = "claude"
model = "claude-sonnet-4-6"
provider = "anthropic"

[[reviewer]]
name = "gemini"
model = "gemini-3-flash-preview"
provider = "gemini"
auth = "oauth"
"#;

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
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, INIT_TEMPLATE)?;
            println!("Created {}", path.display());
            return Ok(());
        }
        Some(Command::Ask {
            common,
            topic,
            debate,
            rounds,
        }) => {
            let repo = common.repo.canonicalize()?;
            if !repo.join(".git").is_dir() {
                eyre::bail!("--repo must point to a git repository (missing .git)");
            }
            let config = load_config(common.config.as_deref(), &repo)?;
            if debate || config.default_debate() {
                return debate::run_debate(
                    &repo,
                    &topic,
                    &config,
                    rounds,
                    common.verbose,
                    debate::DebateMode::Topic,
                )
                .await;
            } else {
                let report = review::run_review(
                    &repo,
                    &topic,
                    &config,
                    common.verbose,
                    review::TaskMode::Ask,
                )
                .await?;
                println!("{report}");
                return Ok(());
            }
        }
        Some(Command::Pr(pr_args)) => {
            let config = load_config(pr_args.common.config.as_deref(), &pr_args.common.repo)?;
            return pr::run_pr(pr_args, config).await;
        }
        None => {}
    }

    // Handle OAuth login if requested (before validating repo or config)
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

    let config = load_config(args.common.config.as_deref(), &repo)?;

    let prompt = if let Some(path) = args.analyze {
        // Empty string means analyze entire repo (from default_missing_value)
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

    if args.debate || config.default_debate() {
        debate::run_debate(
            &repo,
            &prompt,
            &config,
            args.rounds,
            args.common.verbose,
            debate::DebateMode::Review,
        )
        .await
    } else {
        let report = review::run_review(
            &repo,
            &prompt,
            &config,
            args.common.verbose,
            review::TaskMode::Review,
        )
        .await?;
        println!("{report}");
        Ok(())
    }
}

fn load_config(explicit_path: Option<&Path>, repo: &Path) -> Result<config::Config> {
    // 1. explicit --config flag
    if let Some(path) = explicit_path {
        let content = std::fs::read_to_string(path)
            .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", path))?;
        return toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"));
    }

    // 2. repo-level nitpicker.toml
    let repo_config = repo.join("nitpicker.toml");
    if repo_config.exists() {
        let content = std::fs::read_to_string(&repo_config)
            .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", repo_config))?;
        return toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"));
    }

    // 3. global config: ~/.nitpicker/config.toml (same dir as oauth token)
    if let Some(home) = dirs::home_dir() {
        let global_config = home.join(".nitpicker").join("config.toml");
        if global_config.exists() {
            let content = std::fs::read_to_string(&global_config)
                .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", global_config))?;
            return toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"));
        }
    }

    eyre::bail!(
        "no config found. create one with:\n  \
         nitpicker init\n\n\
         or at global location:\n  \
         ~/.nitpicker/config.toml"
    )
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

    let has_branch_commits = if branch != base {
        !run_git(repo, &["log", &format!("{base}..HEAD"), "--oneline"])
            .unwrap_or_default()
            .trim()
            .is_empty()
    } else {
        false
    };

    if !has_uncommitted && !has_branch_commits {
        eyre::bail!("no changes to review: no uncommitted changes and no branch commits vs {base}");
    }

    let mut parts = Vec::new();
    if has_uncommitted {
        parts.push("- uncommitted changes (`git diff HEAD`)".to_string());
    }
    if has_branch_commits {
        parts.push(format!(
            "- commits on this branch vs {base} (`git log {base}..HEAD`, `git diff {base}..HEAD`)"
        ));
    }

    Ok(format!(
        "Review the following changes:\n{}",
        parts.join("\n")
    ))
}

fn detect_base_branch(repo: &Path) -> String {
    run_git(repo, &["symbolic-ref", "refs/remotes/origin/HEAD"])
        .ok()
        .and_then(|s| {
            s.trim()
                .strip_prefix("refs/remotes/origin/")
                .map(str::to_string)
        })
        .unwrap_or_else(|| "main".to_string())
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
