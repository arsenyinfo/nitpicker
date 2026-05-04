use crate::config::Config;
use crate::debate::{self, DebateMode};
use crate::review::{self, TaskMode};
use eyre::{Result, WrapErr};
use serde::Deserialize;
use sha2::Digest;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn lock_path(repo: &Path) -> PathBuf {
    let canonical = repo.canonicalize().unwrap_or_else(|_| repo.to_path_buf());
    let hash = sha2::Sha256::digest(canonical.as_os_str().as_encoded_bytes());
    let hash_hex = format!("{:x}", hash);
    std::env::temp_dir().join(format!("nitpicker-pr-review-{hash_hex}.lock"))
}

struct PrLock {
    path: PathBuf,
}

impl PrLock {
    fn acquire(repo: &Path) -> Result<Self> {
        let path = lock_path(repo);

        if path.exists() {
            if let Ok(contents) = fs::read_to_string(&path) {
                if let Ok(pid) = contents.trim().parse::<u32>() {
                    if !is_process_running(pid) {
                        let _ = fs::remove_file(&path);
                    }
                }
            }
        }

        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                eyre::bail!(
                    "Another PR review is already running for this repo (lock file: {}). Wait for it to finish or remove the lock file if it crashed.",
                    path.display()
                );
            }
            Err(e) => return Err(e).wrap_err("failed to create lock file"),
        };

        let pid = std::process::id();
        writeln!(file, "{pid}").wrap_err("failed to write to lock file")?;
        Ok(Self { path })
    }
}

fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        std::path::Path::new("/proc")
            .join(pid.to_string())
            .join("stat")
            .exists()
    }
    #[cfg(not(unix))]
    {
        // on non-unix, assume the process is still running to be safe
        true
    }
}

impl Drop for PrLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, clap::Args)]
pub struct PrArgs {
    /// Full GitHub PR URL (https://github.com/owner/repo/pull/N)
    pub url: Option<String>,
    #[command(flatten)]
    pub common: crate::CommonArgs,
    #[arg(long)]
    pub prompt: Option<String>,
    #[arg(long)]
    pub no_debate: bool,
    #[arg(long, default_value = "5")]
    pub rounds: usize,
    #[arg(long, value_parser = crate::parse_positive_usize)]
    pub max_turns: Option<usize>,
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

#[derive(Deserialize)]
struct PrComment {
    author: PrCommentAuthor,
    body: String,
}

#[derive(Deserialize)]
struct PrCommentAuthor {
    login: String,
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

fn fetch_pr_comments(url: Option<&str>, repo: &Path) -> Result<Vec<PrComment>> {
    let mut cmd = Command::new("gh");
    cmd.args(["pr", "view", "--json", "comments"])
        .current_dir(repo);
    if let Some(u) = url {
        cmd.arg(u);
    }
    let out = cmd
        .output()
        .wrap_err("failed to run gh pr view --json comments")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("gh pr view comments failed: {}", stderr.trim());
    }
    #[derive(Deserialize)]
    struct CommentsWrapper {
        comments: Vec<PrComment>,
    }
    let wrapper: CommentsWrapper =
        serde_json::from_slice(&out.stdout).wrap_err("failed to parse gh pr comments output")?;
    Ok(wrapper.comments)
}

pub fn parse_pr_url(url: &str) -> Result<(String, u32)> {
    // expects https://github.com/owner/repo/pull/N
    let url_obj = url::Url::parse(url).wrap_err("invalid URL")?;
    match url_obj.host_str() {
        Some("github.com") => {}
        Some(host) => eyre::bail!("only GitHub PRs are supported (got host: {host})"),
        None => eyre::bail!("URL has no host"),
    }
    let segments: Vec<&str> = url_obj
        .path_segments()
        .ok_or_else(|| eyre::eyre!("URL has no path"))?
        .filter(|s| !s.is_empty())
        .collect();
    match segments.as_slice() {
        [owner, repo, "pull", n, ..] => {
            let pr_number: u32 = n
                .parse()
                .wrap_err_with(|| format!("PR number `{n}` is not a valid integer"))?;
            Ok((format!("{owner}/{repo}"), pr_number))
        }
        _ => eyre::bail!("expected a URL like https://github.com/owner/repo/pull/N, got: {url}"),
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
        // git@github.com:owner/repo or hostname:owner/repo (custom ~/.ssh/config alias)
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
    // symbolic-ref succeeds for attached HEAD, fails for detached
    let out = Command::new("git")
        .args(["symbolic-ref", "-q", "--short", "HEAD"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to get current branch")?;
    if out.status.success() {
        return Ok(String::from_utf8_lossy(&out.stdout).trim().to_string());
    }
    // detached HEAD — return commit hash so restore_branch can check it out
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to get current commit")?;
    if !out.status.success() {
        eyre::bail!("failed to get current branch or commit");
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
            eprintln!(
                "warning: failed to restore branch '{branch}': {}",
                stderr.trim()
            );
        }
        Err(e) => eprintln!("warning: failed to restore branch '{branch}': {e}"),
    }
}

/// Fetches the PR head and checks it out as a local branch `nitpicker/pr-{pr_number}`.
/// Uses a namespaced name to avoid colliding with or deleting any user-owned branch.
/// Always fetches so the review is always against the actual PR head, not stale local state.
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

    // namespaced branch — safe to delete, will never match a user branch
    let branch = format!("nitpicker/pr-{pr_number}");

    if get_current_branch(repo).ok().as_deref() == Some(branch.as_str()) {
        // already on our tracking branch — fast-forward only, so uncommitted changes cause a safe failure
        let out = Command::new("git")
            .args(["merge", "--ff-only", "FETCH_HEAD"])
            .current_dir(repo)
            .output()
            .wrap_err("failed to fast-forward PR branch")?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eyre::bail!(
                "could not fast-forward to PR head: {}\nMake sure your working tree is clean before reviewing a PR.",
                stderr.trim()
            );
        }
        return Ok(());
    }

    // delete stale local branch if present (safe: it's our namespace, not a user branch)
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

fn build_pr_prompt(
    meta: &PrMeta,
    comments: &[PrComment],
    diff_context: &str,
    extra: Option<&str>,
) -> String {
    let mut parts = vec![format!("## PR: {}", meta.title)];
    if !meta.body.trim().is_empty() {
        parts.push(meta.body.trim().to_string());
    }
    if !comments.is_empty() {
        let mut comments_section = String::from("## PR Comments\n");
        for comment in comments {
            comments_section.push_str(&format!(
                "**{}**: {}\n\n",
                comment.author.login, comment.body
            ));
        }
        parts.push(comments_section);
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

    let (repo, url_for_gh, meta): (PathBuf, Option<String>, PrMeta) = if let Some(ref u) = args.url
    {
        let (repo_slug, pr_number) = parse_pr_url(u)?;
        let repo_raw = &args.common.repo;
        let repo = repo_raw.canonicalize()?;
        let in_place = repo.join(".git").exists()
            && get_origin_slug(&repo).as_deref() == Some(&repo_slug);

        if in_place {
            // fetch meta first — a failure here leaves the branch untouched
            let meta = fetch_pr_meta(Some(u), &repo)?;
            let branch = get_current_branch(&repo)?;
            checkout_pr_branch(&repo, pr_number).inspect_err(|_e| {
                restore_branch(&repo, &branch);
            })?;
            original_branch = Some((repo.clone(), branch));
            _tmpdir_guard = None;
            (repo, Some(u.clone()), meta)
        } else {
            // different repo — clone into a temp dir
            let cwd = std::env::current_dir().wrap_err("failed to get current directory")?;
            let meta = fetch_pr_meta(Some(u), &cwd)?;
            let pr_commit_count = meta.commits.len();
            let tmpdir = tempfile::TempDir::new().wrap_err("failed to create temp dir")?;
            let path = tmpdir.path().to_path_buf();
            clone_pr(&repo_slug, pr_number, pr_commit_count, &path)?;
            _tmpdir_guard = Some(tmpdir);
            original_branch = None;
            (path, Some(u.clone()), meta)
        }
    } else {
        _tmpdir_guard = None;
        original_branch = None;
        let repo_raw = &args.common.repo;
        if !repo_raw.join(".git").exists() {
            eyre::bail!("--repo must point to a git repository (missing .git)");
        }
        let repo = repo_raw.canonicalize()?;
        let meta = fetch_pr_meta(None, &repo)?;
        (repo, None, meta)
    };

    let _lock = PrLock::acquire(&repo)?;

    let comments = fetch_pr_comments(url_for_gh.as_deref(), &repo)?;

    let result = run_review_inner(
        &repo,
        url_for_gh.as_deref(),
        &args,
        &config,
        verbose,
        &meta,
        &comments,
    )
    .await;

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
    meta: &PrMeta,
    comments: &[PrComment],
) -> Result<()> {
    const FOOTER: &str =
        "\n\n---\n🔍 Reviewed by [nitpicker](https://github.com/arsenyinfo/nitpicker)";

    let diff_context = crate::detect_diff_context(repo)?;
    let full_prompt = build_pr_prompt(meta, comments, &diff_context, args.prompt.as_deref());
    let max_turns = config.max_turns(args.max_turns)?;

    let report = if !args.no_debate && config.default_debate() {
        debate::run_debate(
            repo,
            &full_prompt,
            config,
            args.rounds,
            max_turns,
            verbose,
            DebateMode::Review,
        )
        .await?
    } else {
        review::run_review(
            repo,
            &full_prompt,
            config,
            max_turns,
            verbose,
            TaskMode::Review,
        )
        .await?
    };

    if !args.no_comment {
        post_comment(url_for_gh, repo, &format!("{report}{FOOTER}"))?;
    }

    Ok(())
}
