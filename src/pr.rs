use nitpicker_agent::config::Config;
use crate::debate::{self, DebateMode};
use crate::prompts::ReviewScope;
use crate::review::{self, TaskMode};
use eyre::{Result, WrapErr};
use serde::Deserialize;
use sha2::Digest;
use std::path::{Path, PathBuf};
use std::process::Command;

fn lock_path(repo: &Path) -> PathBuf {
    let canonical = repo.canonicalize().unwrap_or_else(|_| repo.to_path_buf());
    let hash = sha2::Sha256::digest(canonical.as_os_str().as_encoded_bytes());
    let hash_hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    std::env::temp_dir().join(format!("nitpicker-pr-review-{hash_hex}.lock"))
}

struct PrLock {
    // unix: an open fd holding an advisory `flock` for the process lifetime (the lock is the fd, not
    // the file's existence). non-unix: the exclusively-created lock file's path, removed on drop.
    #[cfg(unix)]
    _file: std::fs::File,
    #[cfg(not(unix))]
    path: PathBuf,
}

impl PrLock {
    #[cfg(unix)]
    fn acquire(repo: &Path) -> Result<Self> {
        use std::os::unix::io::AsRawFd;
        let path = lock_path(repo);
        // Advisory `flock(LOCK_EX|LOCK_NB)` held for the process lifetime. The kernel releases it
        // when the fd closes — normal return, `?`, or crash — so there is no stale-pid bookkeeping
        // and no TOCTOU window between a staleness check and an exclusive create (the previous
        // pid-file scheme had both, letting two racers each delete and recreate the lock). The lock
        // file lives at a fixed per-repo path and is never unlinked: unlinking it would let a racing
        // process create a *new* inode and lock that independently. (Advisory flock is unreliable
        // over NFS; PR review runs against a local working tree, so this is acceptable.)
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .wrap_err("failed to open PR review lock file")?;
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            match err.raw_os_error() {
                Some(code) if code == libc::EWOULDBLOCK => eyre::bail!(
                    "Another PR review is already running for this repo (lock file: {}). Wait for it to finish.",
                    path.display()
                ),
                _ => return Err(err).wrap_err("failed to acquire PR review lock"),
            }
        }
        Ok(Self { _file: file })
    }

    #[cfg(not(unix))]
    fn acquire(repo: &Path) -> Result<Self> {
        // No flock: fall back to exclusive create. A crashed run leaves a lock the user must remove
        // by hand (no crash-release), but nitpicker targets unix in practice.
        let path = lock_path(repo);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => Ok(Self { path }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => eyre::bail!(
                "Another PR review is already running for this repo (lock file: {}). Remove it if it crashed.",
                path.display()
            ),
            Err(e) => Err(e).wrap_err("failed to create lock file"),
        }
    }
}

impl Drop for PrLock {
    fn drop(&mut self) {
        // unix: dropping `_file` closes the fd and releases the flock; the lock file is left in
        // place by design. non-unix: remove the exclusive-create lock file.
        #[cfg(not(unix))]
        let _ = std::fs::remove_file(&self.path);
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
    /// Emit a single machine-readable JSON object on stdout (for embedding); human output goes to stderr
    #[arg(long)]
    pub json: bool,
}

impl PrArgs {
    fn output_format(&self) -> crate::output::OutputFormat {
        match self.json {
            true => crate::output::OutputFormat::Json,
            false => crate::output::OutputFormat::Text,
        }
    }
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
    // A comment from a deleted/ghost GitHub account has `author: null`; keep it optional so one
    // such comment doesn't fail the whole comments-list deserialize and abort the run.
    author: Option<PrCommentAuthor>,
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

/// Where HEAD pointed before nitpicker checked out the PR branch, so it can be restored on exit.
enum HeadState {
    Branch(String),
    Detached(String),
}

fn get_head_state(repo: &Path) -> Result<HeadState> {
    // symbolic-ref succeeds for attached HEAD, fails for detached
    let out = Command::new("git")
        .args(["symbolic-ref", "-q", "--short", "HEAD"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to get current branch")?;
    if out.status.success() {
        return Ok(HeadState::Branch(
            String::from_utf8_lossy(&out.stdout).trim().to_string(),
        ));
    }
    // detached HEAD — capture the commit so restore can re-detach onto it
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()
        .wrap_err("failed to get current commit")?;
    if !out.status.success() {
        eyre::bail!("failed to get current branch or commit");
    }
    Ok(HeadState::Detached(
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    ))
}

fn restore_head(repo: &Path, head: &HeadState) {
    // a detached HEAD must be restored with `switch --detach`: `git switch -- <sha>` refuses a bare
    // commit ("a branch is expected, got commit"), which would silently strand the user on
    // nitpicker's PR branch.
    let (args, recover): (Vec<&str>, String) = match head {
        HeadState::Branch(b) => (vec!["switch", "--", b], format!("git switch -- {b}")),
        HeadState::Detached(sha) => (
            vec!["switch", "--detach", sha],
            format!("git switch --detach {sha}"),
        ),
    };
    let result = Command::new("git").args(&args).current_dir(repo).output();
    let detail = match result {
        Ok(out) if out.status.success() => return,
        Ok(out) => String::from_utf8_lossy(&out.stderr).trim().to_string(),
        Err(e) => e.to_string(),
    };
    eprintln!(
        "warning: could not restore HEAD in {repo}; run `{recover}` to recover ({detail})",
        repo = repo.display(),
    );
}

/// Restores the user's original HEAD when dropped, so a panic or early error anywhere in the review
/// path (not just a clean return) can't strand them on nitpicker's PR branch.
struct BranchRestoreGuard {
    repo: PathBuf,
    head: HeadState,
}

impl Drop for BranchRestoreGuard {
    fn drop(&mut self) {
        restore_head(&self.repo, &self.head);
    }
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

    // Fetch into a private named ref instead of relying on the shared `.git/FETCH_HEAD`, which any
    // other fetch in the repo (IDE autofetch, a shell prompt plugin, cron) could rewrite between the
    // fetch and the checkout — silently pointing the review branch at unrelated code. The leading
    // `+` forces the update so a force-pushed PR head still lands. The ref is reused (force-updated)
    // each run, so it is namespace litter at worst, never stale.
    let fetch_ref = format!("refs/nitpicker/pr-{pr_number}-head");
    let refspec = format!("+refs/pull/{pr_number}/head:{fetch_ref}");
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

    if matches!(get_head_state(repo), Ok(HeadState::Branch(b)) if b == branch) {
        // already on our tracking branch — fast-forward only
        let out = Command::new("git")
            .args(["merge", "--ff-only", &fetch_ref])
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
        .args(["switch", "-c", &branch, "--no-track", &fetch_ref])
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
            let login = comment
                .author
                .as_ref()
                .map(|a| a.login.as_str())
                .unwrap_or("[deleted]");
            comments_section.push_str(&format!("**{}**: {}\n\n", login, comment.body));
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

pub async fn run_pr(args: PrArgs) -> Result<()> {
    let start = std::time::Instant::now();
    let format = args.output_format();
    // in json mode every failure (incl. config loading) must still leave a
    // parseable object on stdout and exit non-zero; text mode keeps the
    // eyre-to-stderr behavior.
    match run_pr_inner(args, start).await {
        Ok(()) => Ok(()),
        Err(e) => match format {
            crate::output::OutputFormat::Text => Err(e),
            crate::output::OutputFormat::Json => {
                let env = crate::output::PrReviewOutput::error(
                    format!("{e:#}"),
                    start.elapsed().as_millis() as u64,
                );
                let _ = crate::output::emit_json(&env);
                std::process::exit(1);
            }
        },
    }
}

async fn run_pr_inner(args: PrArgs, start: std::time::Instant) -> Result<()> {
    let config =
        crate::load_resolved_config(args.common.config.as_deref(), &args.common.repo).await?;
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
    // set when we switch branches in the user's own repo; its Drop restores HEAD on the way out
    // (including on panic/early-error), so it must outlive the whole review below
    let _restore_guard: Option<BranchRestoreGuard>;

    // pr number is not part of PrMeta; carry it out of the flow for the json envelope
    let (repo, url_for_gh, meta, pr_number): (PathBuf, Option<String>, PrMeta, Option<u32>) =
        match flow {
            PrFlow::InPlace { url, pr_number } => {
                // fetch meta first — a failure here leaves the branch untouched
                let meta = fetch_pr_meta(Some(&url), &user_repo)?;
                // refresh remote-tracking branches so the diff is computed against an up-to-date base
                refresh_remote_branches(&user_repo)?;
                // Skip the fetch+checkout dance only when HEAD already points at the PR head AND we
                // are on a real branch with a clean tree. Otherwise fall through to checkout:
                //   - a detached HEAD at the PR head must still get a named branch (else
                //     `detect_diff_context` bails) and a restore guard to re-detach afterwards;
                //   - a dirty tree must be rejected by `assert_clean_working_tree`, never reviewed —
                //     skipping would feed local uncommitted WIP into the review and the PR comment.
                let head_matches = get_head_commit(&user_repo)
                    .ok()
                    .is_some_and(|sha| sha == meta.head_ref_oid);
                let head = get_head_state(&user_repo)?;
                let can_skip = head_matches && matches!(head, HeadState::Branch(_));
                match can_skip {
                    true => {
                        assert_clean_working_tree(&user_repo)?;
                        _restore_guard = None;
                    }
                    false => {
                        checkout_pr_branch(&user_repo, pr_number).inspect_err(|_e| {
                            restore_head(&user_repo, &head);
                        })?;
                        _restore_guard = Some(BranchRestoreGuard {
                            repo: user_repo.clone(),
                            head,
                        });
                    }
                }
                _tmpdir_guard = None;
                (user_repo, Some(url), meta, Some(pr_number))
            }
            PrFlow::TempClone {
                url,
                slug,
                pr_number,
            } => {
                let cwd = std::env::current_dir().wrap_err("failed to get current directory")?;
                let meta = fetch_pr_meta(Some(&url), &cwd)?;
                let tmpdir = tempfile::TempDir::new().wrap_err("failed to create temp dir")?;
                // canonicalize so the workspace root matches the canonical paths the tools
                // resolve files to (on macOS the temp dir lives under /var → /private/var)
                let path = tmpdir
                    .path()
                    .canonicalize()
                    .wrap_err("failed to canonicalize temp dir path")?;
                clone_pr(&slug, pr_number, &path)?;
                _tmpdir_guard = Some(tmpdir);
                _restore_guard = None;
                (path, Some(url), meta, Some(pr_number))
            }
            PrFlow::CurrentBranch => {
                let meta = fetch_pr_meta(None, &user_repo)?;
                // refresh remote-tracking branches so the diff is computed against an up-to-date base
                refresh_remote_branches(&user_repo)?;
                _tmpdir_guard = None;
                _restore_guard = None;
                (user_repo, None, meta, None)
            }
        };

    let comments = fetch_pr_comments(url_for_gh.as_deref(), &repo)?;

    // _restore_guard's Drop restores the user's HEAD (panic- and early-error-safe). It drops at the
    // end of this scope, after the review completes and before `_lock`, so restore happens while
    // the lock is still held.
    run_review_inner(
        &repo,
        url_for_gh.as_deref(),
        pr_number,
        &args,
        &config,
        verbose,
        &meta,
        &comments,
        start,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_review_inner(
    repo: &Path,
    url_for_gh: Option<&str>,
    pr_number: Option<u32>,
    args: &PrArgs,
    config: &Config,
    verbose: bool,
    meta: &PrMeta,
    comments: &[PrComment],
    start: std::time::Instant,
) -> Result<()> {
    use crate::output::{OutputFormat, PrInfo, PrReviewOutput, ReviewMode, Status};

    const FOOTER: &str =
        "\n\n---\n🔍 Reviewed by [nitpicker](https://github.com/arsenyinfo/nitpicker)";

    let format = args.output_format();
    let diff_context = crate::detect_diff_context(repo)?;
    let full_prompt = build_pr_prompt(meta, comments, &diff_context, args.prompt.as_deref());
    let max_turns = config.max_turns(args.max_turns)?;

    let use_alloy = args.alloy || config.default_alloy();
    config.validate_alloy(use_alloy)?;
    if use_alloy && args.no_debate {
        eprintln!("warning: --alloy has no effect with --no-debate");
    }
    let debate = !args.no_debate && config.default_debate();
    let (report, transcript_path, usage) = if debate {
        let outcome = debate::run_debate(
            repo,
            &full_prompt,
            config,
            debate::DebateOptions {
                max_rounds: args.rounds,
                max_turns,
                verbose,
                mode: DebateMode::Review(ReviewScope::Diff),
                alloy: use_alloy,
                format,
            },
        )
        .await?;
        (outcome.report, outcome.transcript_path, outcome.usage)
    } else {
        let outcome = review::run_review(
            repo,
            &full_prompt,
            config,
            max_turns,
            verbose,
            TaskMode::Review(ReviewScope::Diff),
        )
        .await?;
        (outcome.report, std::path::PathBuf::new(), outcome.usage)
    };

    match format {
        OutputFormat::Text => {
            println!("{report}");
            if !args.no_comment {
                post_comment(url_for_gh, repo, &format!("{report}{FOOTER}"))?;
            }
            if verbose && !transcript_path.as_os_str().is_empty() {
                eprintln!("\nTranscript saved to: {}", transcript_path.display());
            }
        }
        OutputFormat::Json => {
            // post the comment before emitting so its outcome is reflected in the
            // envelope rather than arriving after a success-looking object. a
            // posting failure is non-fatal here — the report itself succeeded.
            let comment_posted = match args.no_comment {
                true => false,
                false => match post_comment(url_for_gh, repo, &format!("{report}{FOOTER}")) {
                    Ok(()) => true,
                    Err(e) => {
                        tracing::warn!("failed to post PR comment: {e:#}");
                        false
                    }
                },
            };
            let envelope = PrReviewOutput {
                schema_version: crate::output::SCHEMA_VERSION,
                status: Status::Ok,
                pr: Some(PrInfo {
                    url: url_for_gh.map(str::to_string),
                    number: pr_number,
                    title: meta.title.clone(),
                    // report the commit actually reviewed (HEAD), not the oid from `gh pr view`,
                    // which can be stale if the PR was force-pushed between metadata fetch and
                    // checkout. Fall back to the metadata oid if HEAD can't be resolved.
                    head_sha: get_head_commit(repo).unwrap_or_else(|_| meta.head_ref_oid.clone()),
                }),
                mode: Some(match debate {
                    true => ReviewMode::Debate,
                    false => ReviewMode::Parallel,
                }),
                models: Some(crate::output::Models {
                    reviewers: config.reviewer.iter().map(|r| r.model.clone()).collect(),
                    aggregator: config.aggregator.model.clone(),
                }),
                report_markdown: Some(report),
                usage: Some(usage),
                comment_posted,
                duration_ms: start.elapsed().as_millis() as u64,
                error: None,
            };
            crate::output::emit_json(&envelope)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::PrComment;

    #[test]
    fn pr_comments_tolerate_null_author() {
        // gh represents a deleted/ghost account's comment as `author: null`; one such comment must
        // not fail the whole comments-list deserialize and abort the run.
        let comments: Vec<PrComment> = serde_json::from_str(
            r#"[{"author":null,"body":"hi"},{"author":{"login":"octocat"},"body":"yo"}]"#,
        )
        .expect("null author must deserialize");
        assert!(comments[0].author.is_none());
        assert_eq!(comments[1].author.as_ref().unwrap().login, "octocat");
    }
}
