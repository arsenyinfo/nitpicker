use eyre::Result;
use glob::glob;
use regex::Regex;
use rig::completion::ToolDefinition;
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
                "Read a file and return numbered lines. Use start_line/end_line to limit output for large files."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "start_line": { "type": "integer", "minimum": 1 },
                    "end_line": { "type": "integer", "minimum": 1 }
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
            let mut output = String::new();
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
                "Find files by glob pattern relative to the work directory (max 200 results)."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" }
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
            description: "Search files by regex. Returns matches as path:line:content (max 100). Use file_glob to filter file names."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" },
                    "file_glob": { "type": "string" }
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
            if base_path.is_file() {
                if let Err(e) = search_file(&base_path, &regex, &work_dir, &mut results).await {
                    warn!("skipping file {}: {e}", base_path.display());
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
            Ok(results.join("\n"))
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
            description: "Run read-only git commands (allowlisted subcommands only).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
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
            let Some((subcommand, _rest)) = tokens.split_first() else {
                return Ok("Error: empty git command".to_string());
            };
            let allowed = [
                "diff",
                "log",
                "show",
                "blame",
                "status",
                "branch",
                "tag",
                "rev-parse",
                "shortlog",
                "ls-files",
            ];
            if !allowed.contains(subcommand) {
                return Ok(format!("Error: git subcommand '{subcommand}' not allowed"));
            }
            let output = tokio::process::Command::new("git")
                .args(tokens)
                .current_dir(&work_dir)
                .output()
                .await?;
            let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if stdout.len() > 50_000 {
                let boundary = floor_char_boundary(&stdout, 50_000);
                stdout.truncate(boundary);
                stdout.push_str("\n... truncated (50k chars)");
            }
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Ok(format!("Error: {stderr}"));
            }
            Ok(stdout)
        })
    }
}
