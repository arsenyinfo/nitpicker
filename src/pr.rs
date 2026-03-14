use crate::config::Config;
use crate::debate::{self, DebateMode};
use crate::review::{self, TaskMode};
use eyre::{Result, WrapErr};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, clap::Args)]
pub struct PrArgs {
    /// Full GitHub PR URL (https://github.com/owner/repo/pull/N)
    pub url: Option<String>,
    #[command(flatten)]
    pub common: crate::CommonArgs,
    #[arg(long)]
    pub prompt: Option<String>,
    #[arg(long)]
    pub debate: bool,
    #[arg(long, default_value = "5")]
    pub rounds: usize,
    /// Skip posting review as a PR comment
    #[arg(long)]
    pub no_comment: bool,
}

#[derive(Deserialize)]
struct PrMeta {
    title: String,
    body: String,
    commits: Vec<serde_json::Value>,
}

pub fn check_gh() -> Result<()> {
    let status = Command::new("gh").arg("--version").output();
    match status {
        Ok(o) if o.status.success() => Ok(()),
        _ => eyre::bail!(
            "`gh` CLI not found or not working. Install it from https://cli.github.com/ and run `gh auth login`."
        ),
    }
}

fn fetch_pr_meta(url: Option<&str>, repo: &Path) -> Result<PrMeta> {
    let mut cmd = Command::new("gh");
    cmd.args(["pr", "view", "--json", "title,body,commits"])
        .current_dir(repo);
    if let Some(u) = url {
        cmd.arg(u);
    }
    let out = cmd.output().wrap_err("failed to run gh pr view")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("gh pr view failed: {}", stderr.trim());
    }
    serde_json::from_slice(&out.stdout).wrap_err("failed to parse gh pr view output")
}

pub fn parse_pr_url(url: &str) -> Result<(String, u32)> {
    // expects https://github.com/owner/repo/pull/N
    let url_obj = url::Url::parse(url).wrap_err("invalid URL")?;
    let segments: Vec<&str> = url_obj
        .path_segments()
        .ok_or_else(|| eyre::eyre!("URL has no path"))?
        .filter(|s| !s.is_empty())
        .collect();
    match segments.as_slice() {
        [owner, repo, "pull", n] => {
            let pr_number: u32 = n
                .parse()
                .wrap_err_with(|| format!("PR number `{n}` is not a valid integer"))?;
            Ok((format!("{owner}/{repo}"), pr_number))
        }
        _ => eyre::bail!(
            "expected a URL like https://github.com/owner/repo/pull/N, got: {url}"
        ),
    }
}

/// Extracts `owner/repo` from a git remote URL (https or ssh).
fn slug_from_remote_url(url: &str) -> Option<String> {
    let url = url.trim().trim_end_matches(".git");
    if url.contains("://") {
        // https://github.com/owner/repo
        let parsed = url::Url::parse(url).ok()?;
        let path = parsed.path().trim_start_matches('/');
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.len() >= 2 {
            return Some(format!("{}/{}", parts[0], parts[1]));
        }
    } else if let Some(colon) = url.find(':') {
        // git@github.com:owner/repo
        let after = &url[colon + 1..];
        let parts: Vec<&str> = after.split('/').filter(|s| !s.is_empty()).collect();
        if parts.len() >= 2 {
            return Some(format!("{}/{}", parts[0], parts[1]));
        }
    }
    None
}

fn get_origin_slug(repo: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(repo)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    slug_from_remote_url(&String::from_utf8_lossy(&out.stdout))
}

fn get_current_branch(repo: &Path) -> Result<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to get current branch")?;
    if !out.status.success() {
        eyre::bail!("failed to get current branch");
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn restore_branch(repo: &Path, branch: &str) {
    match Command::new("git")
        .args(["checkout", branch])
        .current_dir(repo)
        .output()
    {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!("warning: failed to restore branch '{branch}': {}", stderr.trim());
        }
        Err(e) => eprintln!("warning: failed to restore branch '{branch}': {e}"),
    }
}

/// Fetches the PR head and checks it out as a local branch `pr-{pr_number}`.
/// Works in both full clones (same-repo) and shallow clones (temp dir).
fn checkout_pr_branch(repo: &Path, pr_number: u32) -> Result<()> {
    let refspec = format!("refs/pull/{pr_number}/head");
    let out = Command::new("git")
        .args(["fetch", "origin", &refspec])
        .current_dir(repo)
        .output()
        .wrap_err("failed to fetch PR branch")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("git fetch failed: {}", stderr.trim());
    }

    let branch = format!("pr-{pr_number}");
    // delete stale local branch if present
    let _ = Command::new("git")
        .args(["branch", "-D", &branch])
        .current_dir(repo)
        .output();

    let out = Command::new("git")
        .args(["checkout", "-b", &branch, "FETCH_HEAD"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to checkout PR branch")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!(
            "git checkout failed: {}\nMake sure your working tree is clean before reviewing a PR.",
            stderr.trim()
        );
    }
    Ok(())
}

fn clone_pr(repo_slug: &str, pr_number: u32, pr_commit_count: usize, dir: &Path) -> Result<()> {
    let clone_url = format!("https://github.com/{repo_slug}.git");
    // use a generous depth: PR commits + buffer for base branch reachability and merge commits
    let depth = std::cmp::max(50, pr_commit_count * 2 + 20).to_string();
    let out = Command::new("git")
        .args(["clone", "--depth", &depth, &clone_url, "."])
        .current_dir(dir)
        .output()
        .wrap_err("failed to run git clone")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("git clone failed: {}", stderr.trim());
    }
    checkout_pr_branch(dir, pr_number)
}

fn post_comment(url: Option<&str>, repo: &Path, body: &str) -> Result<()> {
    let mut cmd = Command::new("gh");
    cmd.args(["pr", "comment", "--body", body])
        .current_dir(repo);
    if let Some(u) = url {
        cmd.arg(u);
    }
    let out = cmd.output().wrap_err("failed to run gh pr comment")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("gh pr comment failed: {}", stderr.trim());
    }
    Ok(())
}

fn build_pr_prompt(meta: &PrMeta, diff_context: &str, extra: Option<&str>) -> String {
    let mut parts = vec![format!("## PR: {}", meta.title)];
    if !meta.body.trim().is_empty() {
        parts.push(meta.body.trim().to_string());
    }
    parts.push(diff_context.to_string());
    if let Some(p) = extra {
        if !p.trim().is_empty() {
            parts.push(format!("Additional instructions: {p}"));
        }
    }
    parts.join("\n\n")
}

pub async fn run_pr(args: PrArgs, config: Config) -> Result<()> {
    check_gh()?;

    let verbose = args.common.verbose;
    let _tmpdir_guard: Option<tempfile::TempDir>;

    // original_branch is set when we checkout a PR branch in the user's own repo
    // so we can restore it after the review
    let original_branch: Option<(PathBuf, String)>;

    let (repo, url_for_gh): (PathBuf, Option<String>) = if let Some(ref u) = args.url {
        let (repo_slug, pr_number) = parse_pr_url(u)?;
        let repo_raw = &args.common.repo;
        if !repo_raw.join(".git").is_dir() {
            eyre::bail!("--repo must point to a git repository (missing .git)");
        }
        let repo = repo_raw.canonicalize()?;

        if get_origin_slug(&repo).as_deref() == Some(&repo_slug) {
            // same repo — check out the PR branch in place
            let branch = get_current_branch(&repo)?;
            checkout_pr_branch(&repo, pr_number).map_err(|e| {
                restore_branch(&repo, &branch);
                e
            })?;
            original_branch = Some((repo.clone(), branch));
            _tmpdir_guard = None;
            (repo, Some(u.clone()))
        } else {
            // different repo — clone into a temp dir
            let cwd = std::env::current_dir().wrap_err("failed to get current directory")?;
            let meta_for_depth = fetch_pr_meta(Some(u), &cwd)?;
            let pr_commit_count = meta_for_depth.commits.len();
            let tmpdir = tempfile::TempDir::new().wrap_err("failed to create temp dir")?;
            let path = tmpdir.path().to_path_buf();
            clone_pr(&repo_slug, pr_number, pr_commit_count, &path)?;
            _tmpdir_guard = Some(tmpdir);
            original_branch = None;
            (path, Some(u.clone()))
        }
    } else {
        _tmpdir_guard = None;
        original_branch = None;
        let repo_raw = &args.common.repo;
        if !repo_raw.join(".git").is_dir() {
            eyre::bail!("--repo must point to a git repository (missing .git)");
        }
        (repo_raw.canonicalize()?, None)
    };

    let result = run_review_inner(&repo, url_for_gh.as_deref(), &args, &config, verbose).await;

    if let Some((ref restore_repo, ref branch)) = original_branch {
        restore_branch(restore_repo, branch);
    }

    result
}

async fn run_review_inner(
    repo: &Path,
    url_for_gh: Option<&str>,
    args: &PrArgs,
    config: &Config,
    verbose: bool,
) -> Result<()> {
    let meta = fetch_pr_meta(url_for_gh, repo)?;
    let diff_context = crate::detect_diff_context(repo)?;
    let full_prompt = build_pr_prompt(&meta, &diff_context, args.prompt.as_deref());

    if args.debate || config.default_debate() {
        if !args.no_comment {
            eprintln!("note: --debate mode output will be printed but not posted as a PR comment (not yet supported)");
        }
        debate::run_debate(
            repo,
            &full_prompt,
            config,
            args.rounds,
            verbose,
            DebateMode::Review,
        )
        .await?;
    } else {
        let report = review::run_review(repo, &full_prompt, config, verbose, TaskMode::Review)
            .await?;

        println!("{report}");

        if !args.no_comment {
            let comment = format!(
                "{report}\n\n---\n🔍 Reviewed by [nitpicker](https://github.com/arsenyinfo/nitpicker)"
            );
            post_comment(url_for_gh, repo, &comment)?;
        }
    }

    Ok(())
}
