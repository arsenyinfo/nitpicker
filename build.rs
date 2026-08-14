use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let revision = Command::new("git")
        .args(["-C", &manifest_dir, "rev-parse", "--verify", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|revision| revision.trim().to_string())
        .filter(|revision| !revision.is_empty());

    if let Some(mut revision) = revision {
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
}
