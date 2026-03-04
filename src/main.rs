use clap::{Parser, Subcommand};
use eyre::Result;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

mod agent;
mod config;
mod gemini_proxy;
mod llm;
mod review;
mod tools;

#[derive(Debug, Parser)]
#[command(name = "nitpicker")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    #[arg(long, default_value = ".")]
    repo: PathBuf,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long = "gemini-oauth")]
    gemini_oauth: bool,

    #[arg(long, short)]
    verbose: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate a nitpicker.toml template in the current directory
    Init,
}

const INIT_TEMPLATE: &str = r#"[aggregator]
model = "claude-sonnet-4-6"
provider = "anthropic"

[[reviewer]]
name = "claude"
model = "claude-sonnet-4-6"
provider = "anthropic"

[[reviewer]]
name = "gemini"
model = "gemini-3-flash-previewå"
provider = "gemini"
auth = "oauth"
"#;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let default_level = if args.verbose { "info" } else { "warn" };
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

    if let Some(Command::Init) = args.command {
        let path = Path::new("nitpicker.toml");
        if path.exists() {
            eyre::bail!("nitpicker.toml already exists");
        }
        std::fs::write(path, INIT_TEMPLATE)?;
        println!("Created nitpicker.toml");
        return Ok(());
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

    let repo = args.repo.canonicalize()?;
    if !repo.join(".git").is_dir() {
        eyre::bail!("--repo must point to a git repository (missing .git)");
    }

    let config_path = args.config.unwrap_or_else(|| repo.join("nitpicker.toml"));
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", config_path))?;
    let config: config::Config =
        toml::from_str(&config_str).map_err(|e| eyre::eyre!("invalid config: {e}"))?;

    let prompt = match args.prompt {
        Some(p) => p,
        None => detect_diff_context(&repo)?,
    };

    let report = review::run_review(&repo, &prompt, &config, args.verbose).await?;
    println!("{report}");
    Ok(())
}

fn detect_diff_context(repo: &Path) -> Result<String> {
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
