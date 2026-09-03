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

    // Declaring any rerun-if-changed disables cargo's default rerun-on-package-change, so the
    // package sources are re-registered explicitly (dirty detection) alongside the git state that
    // moves on commit or checkout: HEAD, the branch it points at (a commit on a branch never
    // touches HEAD itself), packed-refs, and the index. Paths are resolved through git so linked
    // worktrees get real files; a missing rerun path would make cargo rerun on every build.
    for dir in ["Cargo.toml", "Cargo.lock", "build.rs", "src", "prompts", "crates"] {
        println!("cargo:rerun-if-changed={manifest_dir}/{dir}");
    }
    let mut git_paths = vec!["HEAD".to_string(), "packed-refs".to_string(), "index".to_string()];
    if let Some(branch_ref) = git_stdout(&manifest_dir, &["symbolic-ref", "-q", "HEAD"]) {
        git_paths.push(branch_ref);
    }
    for git_path in git_paths {
        let resolved = git_stdout(&manifest_dir, &["rev-parse", "--git-path", &git_path])
            .map(|path| {
                let path = PathBuf::from(path);
                if path.is_absolute() {
                    path
                } else {
                    PathBuf::from(&manifest_dir).join(path)
                }
            });
        match resolved {
            Some(path) if path.is_file() => println!("cargo:rerun-if-changed={}", path.display()),
            _ => {}
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
