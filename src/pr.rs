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
    let hash_hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    std::env::temp_dir().join(format!("nitpicker-pr-review-{hash_hex}.lock"))
}

struct PrLock {
    path: PathBuf,
}

impl PrLock {
    fn acquire(repo: &Path) -> Result<Self> {
        let path = lock_path(repo);

        // Treat the lock as stale (and remove it) when:
        //   - the holding pid is no longer alive, OR
        //   - the file is unreadable / empty / non-numeric — which happens when a previous
        //     process was killed between create_new() and writeln!(pid), leaving a 0-byte
        //     file that would otherwise deadlock all future invocations.
        if path.exists() {
            let stale = match fs::read_to_string(&path) {
                Ok(contents) => match contents.trim().parse::<u32>() {
                    Ok(pid) => !is_process_running(pid),
                    Err(_) => true,
                },
                Err(_) => true,
            };
            if stale {
                let _ = fs::remove_file(&path);
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
        // kill(pid, 0) does no signaling — only validates pid lookup and permissions.
        // 0 => exists and we can signal it. EPERM => exists but we can't (still running).
        // ESRCH => no such process. anything else: assume still running (safer than racing).
        let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
        match (rc, std::io::Error::last_os_error().raw_os_error()) {
            (0, _) => true,
            (_, Some(e)) if e == libc::EPERM => true,
            (_, Some(e)) if e == libc::ESRCH => false,
            _ => true,
        }
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
    #[arg(long)]
    pub alloy: bool,
    #[arg(long, default_value = "5")]
    pub rounds: usize,
    #[arg(long, value_parser = crate::parse_positive_usize)]
    pub max_turns: Option<usize>,
    /// Skip posting review as a PR comment
    #[arg(long)]
    pub no_comment: bool,
    /// Always review in a fresh temp clone, even if the current repo matches the PR's origin
    #[arg(long)]
    pub clone: bool,
}

#[derive(Deserialize)]
struct PrMeta {
    title: String,
    body: String,
    #[serde(rename = "headRefOid")]
    head_ref_oid: String,
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
    cmd.args(["pr", "view", "--json", "title,body,headRefOid"])
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

fn get_head_commit(repo: &Path) -> Result<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to get HEAD commit")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("git rev-parse HEAD failed: {}", stderr.trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
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
    let result = Command::new("git")
        .args(["switch", "--", branch])
        .current_dir(repo)
        .output();
    let detail = match result {
        Ok(out) if out.status.success() => return,
        Ok(out) => String::from_utf8_lossy(&out.stderr).trim().to_string(),
        Err(e) => e.to_string(),
    };
    eprintln!(
        "warning: could not restore branch '{branch}' in {repo}; run `git switch -- {branch}` to recover ({detail})",
        repo = repo.display(),
    );
}

fn assert_clean_working_tree(repo: &Path) -> Result<()> {
    let out = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to run git status")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("git status failed: {}", stderr.trim());
    }
    let porcelain = String::from_utf8_lossy(&out.stdout);
    if !porcelain.trim().is_empty() {
        eyre::bail!(
            "working tree has uncommitted changes; commit or stash before running `nitpicker pr` against a different PR:\n{}",
            porcelain.trim_end()
        );
    }
    Ok(())
}

/// Refreshes remote-tracking branches so `detect_base_branch` doesn't operate on stale state.
/// Fatal: if origin can't be fetched, the diff base would be stale and the review unreliable.
fn refresh_remote_branches(repo: &Path) -> Result<()> {
    let out = Command::new("git")
        .args(["fetch", "origin", "--prune"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to refresh remote branches")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!(
            "git fetch origin failed: {}\nReview would run against a stale base branch. Fix the remote or pass `--clone` to review in a fresh temp clone.",
            stderr.trim()
        );
    }
    Ok(())
}

/// Fetches the PR head and checks it out as a local branch `nitpicker/pr-{pr_number}`.
/// Uses a namespaced name to avoid colliding with or deleting any user-owned branch.
/// Always fetches so the review is always against the actual PR head, not stale local state.
fn checkout_pr_branch(repo: &Path, pr_number: u32) -> Result<()> {
    assert_clean_working_tree(repo)?;

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
        // already on our tracking branch — fast-forward only
        let out = Command::new("git")
            .args(["merge", "--ff-only", "FETCH_HEAD"])
            .current_dir(repo)
            .output()
            .wrap_err("failed to fast-forward PR branch")?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eyre::bail!("could not fast-forward to PR head: {}", stderr.trim());
        }
        return Ok(());
    }

    // delete stale local branch if present (safe: it's our namespace, not a user branch)
    let _ = Command::new("git")
        .args(["branch", "-D", &branch])
        .current_dir(repo)
        .output();

    let out = Command::new("git")
        .args(["switch", "-c", &branch, "--no-track", "FETCH_HEAD"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to checkout PR branch")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("git switch failed: {}", stderr.trim());
    }
    Ok(())
}

fn clone_pr(repo_slug: &str, pr_number: u32, dir: &Path) -> Result<()> {
    let clone_url = format!("https://github.com/{repo_slug}.git");
    // partial clone: full commit/tree history (so merge-base with the PR is always reachable),
    // blobs fetched lazily on demand. Avoids the shallow-depth merge-base gap.
    let out = Command::new("git")
        .args(["clone", "--filter=blob:none", &clone_url, "."])
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

enum PrFlow {
    /// no URL — review current branch against its PR, no checkout needed
    CurrentBranch,
    /// URL points to a PR in the user's own repo and `--clone` was not set
    InPlace { url: String, pr_number: u32 },
    /// URL points elsewhere, or `--clone` forces a fresh clone
    TempClone {
        url: String,
        slug: String,
        pr_number: u32,
    },
}

pub async fn run_pr(args: PrArgs, config: Config) -> Result<()> {
    check_gh()?;

    let verbose = args.common.verbose;
    let user_repo = args
        .common
        .repo
        .canonicalize()
        .wrap_err("failed to canonicalize --repo path")?;
    let user_has_git = user_repo.join(".git").exists();

    let flow = match args.url.as_deref() {
        None => {
            if !user_has_git {
                eyre::bail!("--repo must point to a git repository (missing .git)");
            }
            PrFlow::CurrentBranch
        }
        Some(u) => {
            let (slug, pr_number) = parse_pr_url(u)?;
            let in_place = !args.clone
                && user_has_git
                && get_origin_slug(&user_repo).as_deref() == Some(&slug);
            match in_place {
                true => PrFlow::InPlace {
                    url: u.to_string(),
                    pr_number,
                },
                false => PrFlow::TempClone {
                    url: u.to_string(),
                    slug,
                    pr_number,
                },
            }
        }
    };

    // Acquire the lock BEFORE any git mutation when we'll touch the user's working tree.
    // Temp clones use a fresh, unique temp dir per process, so concurrent runs don't conflict.
    let _lock = match &flow {
        PrFlow::InPlace { .. } | PrFlow::CurrentBranch => Some(PrLock::acquire(&user_repo)?),
        PrFlow::TempClone { .. } => None,
    };

    let _tmpdir_guard: Option<tempfile::TempDir>;
    // set when we switch branches in the user's own repo so we can restore on the way out
    let original_branch: Option<(PathBuf, String)>;

    let (repo, url_for_gh, meta): (PathBuf, Option<String>, PrMeta) = match flow {
        PrFlow::InPlace { url, pr_number } => {
            // fetch meta first — a failure here leaves the branch untouched
            let meta = fetch_pr_meta(Some(&url), &user_repo)?;
            // refresh remote-tracking branches so the diff is computed against an up-to-date base
            refresh_remote_branches(&user_repo)?;
            // if HEAD already matches the PR head, skip the fetch+checkout dance
            let head_matches = get_head_commit(&user_repo)
                .ok()
                .is_some_and(|sha| sha == meta.head_ref_oid);
            match head_matches {
                true => {
                    original_branch = None;
                }
                false => {
                    let branch = get_current_branch(&user_repo)?;
                    checkout_pr_branch(&user_repo, pr_number).inspect_err(|_e| {
                        restore_branch(&user_repo, &branch);
                    })?;
                    original_branch = Some((user_repo.clone(), branch));
                }
            }
            _tmpdir_guard = None;
            (user_repo, Some(url), meta)
        }
        PrFlow::TempClone {
            url,
            slug,
            pr_number,
        } => {
            let cwd = std::env::current_dir().wrap_err("failed to get current directory")?;
            let meta = fetch_pr_meta(Some(&url), &cwd)?;
            let tmpdir = tempfile::TempDir::new().wrap_err("failed to create temp dir")?;
            let path = tmpdir.path().to_path_buf();
            clone_pr(&slug, pr_number, &path)?;
            _tmpdir_guard = Some(tmpdir);
            original_branch = None;
            (path, Some(url), meta)
        }
        PrFlow::CurrentBranch => {
            let meta = fetch_pr_meta(None, &user_repo)?;
            // refresh remote-tracking branches so the diff is computed against an up-to-date base
            refresh_remote_branches(&user_repo)?;
            _tmpdir_guard = None;
            original_branch = None;
            (user_repo, None, meta)
        }
    };

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

    let use_alloy = args.alloy || config.default_alloy();
    config.validate_alloy(use_alloy)?;
    if use_alloy && args.no_debate {
        eprintln!("warning: --alloy has no effect with --no-debate");
    }
    let (report, transcript_path) = if !args.no_debate && config.default_debate() {
        debate::run_debate(
            repo,
            &full_prompt,
            config,
            debate::DebateOptions {
                max_rounds: args.rounds,
                max_turns,
                verbose,
                mode: DebateMode::Review,
                alloy: use_alloy,
            },
        )
        .await?
    } else {
        let report = review::run_review(
            repo,
            &full_prompt,
            config,
            max_turns,
            verbose,
            TaskMode::Review,
        )
        .await?;
        (report, std::path::PathBuf::new())
    };

    println!("{report}");
    if !args.no_comment {
        post_comment(url_for_gh, repo, &format!("{report}{FOOTER}"))?;
    }
    if verbose && !transcript_path.as_os_str().is_empty() {
        eprintln!("\nTranscript saved to: {}", transcript_path.display());
    }

    Ok(())
}
