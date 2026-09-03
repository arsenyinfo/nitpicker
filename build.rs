use std::path::PathBuf;
use std::process::Command;

fn git_stdout(manifest_dir: &str, args: &[&str]) -> Option<String> {
    Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|stdout| stdout.trim().to_string())
        .filter(|stdout| !stdout.is_empty())
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let Some(mut revision) = git_stdout(&manifest_dir, &["rev-parse", "--verify", "HEAD"]) else {
        return;
    };

    // cargo only reruns a build script on package-file changes by default; a bare commit or
    // checkout would otherwise leave a stale revision baked in. Resolve the git dir so linked
    // worktrees (where `.git` is a file) get real paths, and skip files that don't exist since
    // a missing rerun path makes cargo rerun on every build.
    if let Some(git_dir) = git_stdout(&manifest_dir, &["rev-parse", "--absolute-git-dir"]) {
        for name in ["HEAD", "index"] {
            let path = PathBuf::from(&git_dir).join(name);
            if path.is_file() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }

    let dirty = Command::new("git")
        .args([
            "-C",
            &manifest_dir,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| !output.stdout.is_empty());
    if dirty {
        revision.push_str("-dirty");
    }
    println!("cargo:rustc-env=NITPICKER_BUILD_REVISION={revision}");
}
