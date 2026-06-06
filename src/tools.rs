use eyre::Result;
use glob::glob;
use regex::Regex;
use rig_core::completion::ToolDefinition;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tokio::fs;
use tracing::warn;

/// Find a valid UTF-8 character boundary at or before the given position.
/// This is a polyfill for `str::floor_char_boundary` which requires Rust 1.91.
pub fn floor_char_boundary(s: &str, pos: usize) -> usize {
    let pos = pos.min(s.len());
    // UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF)
    // We need to find a byte that is NOT a continuation byte
    let bytes = s.as_bytes();
    for i in (0..=pos).rev() {
        if i == 0 || (bytes[i] & 0xC0) != 0x80 {
            return i;
        }
    }
    0
}

/// Allowlisted read-only git subcommands. Ref listing is served by the `for-each-ref`/`show-ref`
/// plumbing (no ref-creating/deleting mode → safe by construction for any args), deliberately not
/// the `branch`/`tag` porcelain whose read and write modes are indistinguishable by arguments.
const ALLOWED_GIT_SUBCOMMANDS: &[&str] = &[
    "diff",
    "log",
    "show",
    "blame",
    "status",
    "rev-parse",
    "shortlog",
    "ls-files",
    "for-each-ref",
    "show-ref",
];

/// Reject git invocations that aren't genuinely read-only. The subcommand allowlist alone is not
/// enough: `diff`/`log`/`show` accept `--output=<path>` (write-anywhere, traversal-capable). Ref
/// listing is served by the read-only plumbing `for-each-ref`/`show-ref` (no ref-creating mode),
/// not the `branch`/`tag` porcelain, so no per-subcommand argument validation is needed beyond
/// blocking the output-to-file flag. Returns `Err(message)` for a rejected invocation.
fn ensure_readonly_git(rest: &[&str]) -> std::result::Result<(), String> {
    // `--output`/`-o <file>` writes command output to an arbitrary path on every subcommand. git
    // also accepts unambiguous long-option abbreviations, so reject any `--…` prefix of `--output`
    // too (e.g. `--out`), alongside the exact, `=`-glued, and short (`-o`/`-o<file>`) forms.
    for token in rest {
        let abbreviates_output = token
            .strip_prefix("--")
            .map(|long| long.split('=').next().unwrap_or(long))
            .is_some_and(|stem| !stem.is_empty() && "output".starts_with(stem));
        if abbreviates_output
            || *token == "-o"
            || (token.starts_with("-o") && !token.starts_with("--") && token.len() > 2)
        {
            return Err("Error: writing git output to a file (--output/-o) is not allowed".into());
        }
    }
    Ok(())
}

pub trait Tool: Send + Sync {
    fn name(&self) -> String;
    fn definition(&self) -> ToolDefinition;
    fn call(
        &self,
        args: Value,
        work_dir: PathBuf,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>;
}

pub fn all_tools() -> HashMap<String, Arc<dyn Tool>> {
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool),
        Arc::new(GlobTool),
        Arc::new(GrepTool),
        Arc::new(GitTool),
    ];
    tools.into_iter().map(|tool| (tool.name(), tool)).collect()
}

pub fn reflect_tools() -> HashMap<String, Arc<dyn Tool>> {
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool),
        Arc::new(GlobTool),
        Arc::new(GrepTool),
    ];
    tools.into_iter().map(|tool| (tool.name(), tool)).collect()
}

pub fn tool_definitions(tools: &HashMap<String, Arc<dyn Tool>>) -> Vec<ToolDefinition> {
    tools.values().map(|tool| tool.definition()).collect()
}

pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> String {
        "read_file".to_string()
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description:
                "Read a text file inside the workspace and return numbered lines. Use this after glob or grep to inspect specific files; prefer start_line/end_line for focused reads."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path to read."
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "First line to include. Omit to start at line 1."
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Last line to include. Omit to read to the end of the file."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    fn call(
        &self,
        args: Value,
        work_dir: PathBuf,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>> {
        Box::pin(async move {
            let path = args
                .get("path")
                .and_then(|value| value.as_str())
                .ok_or_else(|| eyre::eyre!("missing path"))?;
            let start_line = args
                .get("start_line")
                .and_then(|value| value.as_u64())
                .unwrap_or(1) as usize;
            let end_line = args
                .get("end_line")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize);
            let full_path = work_dir.join(path);
            let full_path = full_path
                .canonicalize()
                .map_err(|e| eyre::eyre!("cannot resolve path {path:?}: {e}. Only files within {} are accessible.", work_dir.display()))?;
            if !full_path.starts_with(&work_dir) {
                eyre::bail!("access denied: {path:?} is outside the allowed workspace ({}). Only project files are accessible.", work_dir.display());
            }
            let content = fs::read_to_string(&full_path).await?;
            let lines = content.lines().collect::<Vec<_>>();
            let total = lines.len();
            let start = start_line.max(1).min(total.max(1));
            let end = end_line.unwrap_or(total).max(start).min(total);
            let relative = full_path
                .strip_prefix(&work_dir)
                .unwrap_or(&full_path)
                .display()
                .to_string();
            let mut output = format!("File: {relative}\nLines: {start}-{end} of {total}\n\n");
            for (idx, line) in lines.iter().enumerate() {
                let line_num = idx + 1;
                if line_num < start || line_num > end {
                    continue;
                }
                output.push_str(&format!("{line_num:>4} {line}\n"));
            }
            Ok(output)
        })
    }
}

pub struct GlobTool;

impl Tool for GlobTool {
    fn name(&self) -> String {
        "glob".to_string()
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob".to_string(),
            description:
                "Find workspace-relative file paths by glob pattern. Use this when you know the file name or extension pattern but not the exact path; returns at most 200 matches."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Workspace-relative glob such as 'src/**/*.rs' or '**/*.toml'."
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        }
    }

    fn call(
        &self,
        args: Value,
        work_dir: PathBuf,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>> {
        Box::pin(async move {
            let pattern = args
                .get("pattern")
                .and_then(|value| value.as_str())
                .ok_or_else(|| eyre::eyre!("missing pattern"))?;
            let pattern_path = Path::new(pattern);
            if pattern_path.is_absolute()
                || pattern_path
                    .components()
                    .any(|c| c == std::path::Component::ParentDir)
            {
                eyre::bail!("access denied: glob pattern {pattern:?} must be relative to the workspace ({}). Absolute paths and parent-directory traversal are not allowed.", work_dir.display());
            }
            let mut results = Vec::new();
            let full_pattern = work_dir.join(pattern);
            let full_pattern = full_pattern.to_string_lossy();
            for entry in glob(&full_pattern)? {
                if let Ok(path) = entry {
                    if let Ok(relative) = path.strip_prefix(&work_dir) {
                        results.push(relative.display().to_string());
                    }
                }
                if results.len() >= 200 {
                    break;
                }
            }
            if results.is_empty() {
                return Ok(format!("No files matched pattern: {pattern}"));
            }
            Ok(results.join("\n"))
        })
    }
}

pub struct GrepTool;

impl Tool for GrepTool {
    fn name(&self) -> String {
        "grep".to_string()
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".to_string(),
            description: "Search text files for a regex and return path:line:content matches. Use this to locate relevant code before calling read_file; optionally limit by path or file_glob, and expect at most 100 matches."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for in file contents."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional workspace-relative file or directory to search within."
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Optional filename filter such as '*.rs'; matched against file names, not full paths."
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        }
    }

    fn call(
        &self,
        args: Value,
        work_dir: PathBuf,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>> {
        Box::pin(async move {
            let pattern = args
                .get("pattern")
                .and_then(|value| value.as_str())
                .ok_or_else(|| eyre::eyre!("missing pattern"))?;
            let base_path = args
                .get("path")
                .and_then(|value| value.as_str())
                .map(|value| {
                    let p = work_dir.join(value);
                    // canonicalize to resolve symlinks and `..` before the workspace check
                    p.canonicalize()
                        .map_err(|e| eyre::eyre!("cannot resolve path {value:?}: {e}. Only files within {} are accessible.", work_dir.display()))
                })
                .transpose()?
                .unwrap_or_else(|| work_dir.clone());
            if !base_path.starts_with(&work_dir) {
                let path_arg = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                eyre::bail!("access denied: {path_arg:?} is outside the allowed workspace ({}). Only project files are accessible.", work_dir.display());
            }
            let file_glob = args
                .get("file_glob")
                .and_then(|value| value.as_str())
                .map(glob_to_regex)
                .transpose()?;
            let regex = Regex::new(pattern)
                .map_err(|e| eyre::eyre!("invalid regex {pattern:?}: {e}"))?;
            let mut results = Vec::new();
            let mut skipped_files = 0usize;
            if base_path.is_file() {
                if let Err(e) = search_file(&base_path, &regex, &work_dir, &mut results).await {
                    warn!("skipping file {}: {e}", base_path.display());
                    skipped_files += 1;
                }
            } else {
                let mut stack = vec![base_path];
                while let Some(path) = stack.pop() {
                    let entries = match fs::read_dir(&path).await {
                        Ok(entries) => entries,
                        Err(e) => {
                            warn!("skipping unreadable dir {}: {e}", path.display());
                            continue;
                        }
                    };
                    let mut entries = entries;
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        let entry_path = entry.path();
                        let file_type = match entry.file_type().await {
                            Ok(file_type) => file_type,
                            Err(e) => {
                                warn!("skipping {}: {e}", entry_path.display());
                                continue;
                            }
                        };
                        let name = entry.file_name();
                        let name = name.to_string_lossy();
                        if file_type.is_dir() {
                            if name.starts_with('.') || name == "target" || name == "node_modules" {
                                continue;
                            }
                            stack.push(entry_path);
                        } else if file_type.is_file() {
                            if let Some(filter) = &file_glob {
                                if !filter.is_match(&name) {
                                    continue;
                                }
                            }
                            match search_file(&entry_path, &regex, &work_dir, &mut results).await {
                                Ok(_) => {}
                                Err(e) => {
                                    warn!("skipping file {}: {e}", entry_path.display());
                                    skipped_files += 1;
                                }
                            }
                            if results.len() >= 100 {
                                break;
                            }
                        }
                        if results.len() >= 100 {
                            break;
                        }
                    }
                    if results.len() >= 100 {
                        break;
                    }
                }
            }
            if results.is_empty() {
                let mut output = format!("No matches for regex: {pattern}");
                if skipped_files > 0 {
                    output.push_str(&format!("\nSkipped unreadable files: {skipped_files}"));
                }
                return Ok(output);
            }
            let mut output = format!("Matches: {}\n", results.len());
            if skipped_files > 0 {
                output.push_str(&format!("Skipped unreadable files: {skipped_files}\n"));
            }
            output.push('\n');
            output.push_str(&results.join("\n"));
            Ok(output)
        })
    }
}

async fn search_file(
    path: &PathBuf,
    regex: &Regex,
    work_dir: &Path,
    results: &mut Vec<String>,
) -> Result<()> {
    use tokio::io::AsyncReadExt;

    // Open file and check first 8KB for binary content before reading full file
    let mut file = fs::File::open(path).await?;
    let mut buffer = [0u8; 8192];
    let bytes_read = file.read(&mut buffer).await?;

    // Check for null bytes in the sample (binary file indicator)
    if buffer[..bytes_read].contains(&0) {
        return Ok(()); // Skip binary files silently
    }

    // Read the rest of the file
    let mut remaining = Vec::new();
    file.read_to_end(&mut remaining).await?;

    // Combine sample + remaining into full content
    let mut full_content = Vec::with_capacity(bytes_read + remaining.len());
    full_content.extend_from_slice(&buffer[..bytes_read]);
    full_content.extend_from_slice(&remaining);

    // Convert to string and search
    let content = String::from_utf8_lossy(&full_content);
    let relative = path.strip_prefix(work_dir).unwrap_or(path);
    for (idx, line) in content.lines().enumerate() {
        if regex.is_match(line) {
            results.push(format!("{}:{}:{}", relative.display(), idx + 1, line));
            if results.len() >= 100 {
                break;
            }
        }
    }
    Ok(())
}

fn glob_to_regex(pattern: &str) -> Result<Regex> {
    let mut escaped = String::new();
    for ch in pattern.chars() {
        match ch {
            '.' => escaped.push_str("\\."),
            '*' => escaped.push_str(".*"),
            '?' => escaped.push('.'),
            other => escaped.push(other),
        }
    }
    Regex::new(&format!("^{}$", escaped))
        .map_err(|e| eyre::eyre!("invalid file_glob {pattern:?}: {e}"))
}

/// Check if a file is binary by reading the first 8 KiB and checking for null bytes.
/// Returns `true` if binary, `false` if text, or an error if the file cannot be read.
pub async fn is_binary_file(path: &Path) -> std::io::Result<bool> {
    use tokio::io::AsyncReadExt;
    let mut file = fs::File::open(path).await?;
    let mut buffer = [0u8; 8192];
    let bytes_read = file.read(&mut buffer).await?;
    Ok(buffer[..bytes_read].contains(&0)) // null byte = binary
}

pub struct GitTool;

impl Tool for GitTool {
    fn name(&self) -> String {
        "git".to_string()
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "git".to_string(),
            description: "Run an allowlisted read-only git command for review context: diff, log, show, blame, status, rev-parse, shortlog, ls-files, for-each-ref, show-ref. To list or query branches/tags use for-each-ref (e.g. `for-each-ref --contains <sha> refs/heads/`), not branch/tag. Use this for repository history or patch context, not for general file search."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Read-only git command to run, for example 'diff --stat HEAD~1' or 'log --oneline -n 20'."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    fn call(
        &self,
        args: Value,
        work_dir: PathBuf,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>> {
        Box::pin(async move {
            let command = args
                .get("command")
                .and_then(|value| value.as_str())
                .ok_or_else(|| eyre::eyre!("missing command"))?;
            let tokens = command.split_whitespace().collect::<Vec<_>>();
            let Some((subcommand, rest)) = tokens.split_first() else {
                return Ok("Error: empty git command".to_string());
            };
            if !ALLOWED_GIT_SUBCOMMANDS.contains(subcommand) {
                return Ok(format!("Error: git subcommand '{subcommand}' not allowed"));
            }
            if let Err(msg) = ensure_readonly_git(rest) {
                return Ok(msg);
            }
            let output = tokio::process::Command::new("git")
                .args(tokens)
                .current_dir(&work_dir)
                .output()
                .await?;
            let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if stdout.len() > 50_000 {
                let original_len = stdout.len();
                let boundary = floor_char_boundary(&stdout, 50_000);
                stdout.truncate(boundary);
                stdout.push_str(&format!(
                    "\n... truncated after 50,000 chars; {} chars omitted",
                    original_len.saturating_sub(boundary)
                ));
            }
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Ok(format!("Error: {stderr}"));
            }
            Ok(stdout)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ALLOWED_GIT_SUBCOMMANDS, ensure_readonly_git};

    /// Mirror the two gates GitTool applies: subcommand allowlist, then read-only argument check.
    fn check(cmd: &str) -> Result<(), String> {
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        let (sub, rest) = tokens.split_first().expect("non-empty command");
        if !ALLOWED_GIT_SUBCOMMANDS.contains(sub) {
            return Err(format!("subcommand '{sub}' not allowed"));
        }
        ensure_readonly_git(rest)
    }

    #[test]
    fn rejects_output_file_write() {
        // --output / -o turn a read-only command into a write-anywhere primitive
        assert!(check("diff HEAD~1 --output=/tmp/evil").is_err());
        assert!(check("diff --output ../escape.txt HEAD~1").is_err());
        assert!(check("log --oneline -o /tmp/x").is_err());
        assert!(check("show -o/tmp/glued HEAD").is_err());
        assert!(check("diff --out=/tmp/abbrev HEAD~1").is_err()); // long-option abbreviation
    }

    #[test]
    fn allows_plain_read_commands() {
        assert!(check("diff HEAD~1").is_ok());
        assert!(check("log --oneline -n 5").is_ok());
        assert!(check("show HEAD").is_ok());
        assert!(check("status --porcelain").is_ok());
        // --oneline must not be mistaken for the -o output flag
        assert!(check("log --oneline").is_ok());
    }

    #[test]
    fn rejects_branch_and_tag_porcelain() {
        // branch/tag conflate read and write and are not on the allowlist at all; ref listing must
        // go through the for-each-ref/show-ref plumbing instead.
        assert!(check("branch -D some-branch").is_err());
        assert!(check("branch new-branch").is_err());
        assert!(check("branch --set-upstream-to=origin/main").is_err());
        assert!(check("branch -- newbranch").is_err());
        assert!(check("tag v9.9.9").is_err());
        assert!(check("tag -d v1").is_err());
        // even pure listing forms are rejected — the model is steered to for-each-ref
        assert!(check("branch --list").is_err());
    }

    #[test]
    fn allows_readonly_ref_plumbing() {
        // for-each-ref / show-ref have no ref-creating mode, so any arg shape is safe.
        assert!(check("for-each-ref refs/heads/").is_ok());
        assert!(check("for-each-ref --contains HEAD refs/heads/").is_ok());
        assert!(check("for-each-ref --format=%(refname) --sort=-creatordate").is_ok());
        assert!(check("show-ref --tags").is_ok());
        assert!(check("show-ref --heads").is_ok());
        // but the global output-to-file block still applies to them
        assert!(check("for-each-ref --output=/tmp/evil").is_err());
    }
}
