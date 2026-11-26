//! Zero-dependency `.env` file loader and fallback lookup for API keys.
//!
//! Process environment variables always take precedence over loaded env files.
//! Keeping the loaded values in an in-process map avoids `unsafe { std::env::set_var }`
//! (which is unsound once Tokio worker threads exist) and scopes env files strictly to
//! callers that opt in via [`var`], rather than letting a file override proxy, telemetry,
//! or runtime settings.

use eyre::Result;
use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::{OnceLock, RwLock};

static FALLBACK_ENV: OnceLock<RwLock<HashMap<String, String>>> = OnceLock::new();

fn store() -> &'static RwLock<HashMap<String, String>> {
    FALLBACK_ENV.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Look up an environment variable, falling back to values loaded by [`load_files`].
///
/// Real process environment variables always win over values from env files.
pub fn var(name: &str) -> Result<String, std::env::VarError> {
    match std::env::var(name) {
        Ok(val) => Ok(val),
        Err(std::env::VarError::NotUnicode(raw)) => Err(std::env::VarError::NotUnicode(raw)),
        Err(std::env::VarError::NotPresent) => {
            let map = store().read().unwrap_or_else(|e| e.into_inner());
            map.get(name)
                .cloned()
                .ok_or(std::env::VarError::NotPresent)
        }
    }
}

/// Load key-value pairs from `paths` into the fallback lookup map, replacing any
/// previously loaded entries.
///
/// Paths are ordered from highest to lowest precedence (e.g. `[<repo>/.env.nitpicker,
/// ~/.nitpicker/.env]`). Missing files are ignored; I/O errors other than `NotFound`
/// and syntax errors are returned.
pub fn load_files(paths: &[&Path]) -> Result<()> {
    let merged = read_files(paths)?;
    let mut guard = store().write().unwrap_or_else(|e| e.into_inner());
    *guard = merged;
    Ok(())
}

/// Read and merge `paths` (highest precedence first) without mutating global state.
pub fn read_files(paths: &[&Path]) -> Result<HashMap<String, String>> {
    let mut merged = HashMap::new();
    for path in paths {
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => {
                eyre::bail!("failed to read env file {}: {err}", path.display());
            }
        };
        let entries = parse_dotenv(&content)
            .map_err(|err| eyre::eyre!("invalid env file {}: {err}", path.display()))?;
        for (key, value) in entries {
            merged.entry(key).or_insert(value);
        }
    }
    Ok(merged)
}

/// Parse a `.env` file into key-value pairs.
///
/// Supports `#` comments, optional `export ` prefixes, single-quoted literals,
/// double-quoted strings with basic escapes (`\n`, `\r`, `\t`, `\"`, `\\`), and
/// inline ` # comment` suffixes on unquoted values. Empty values are treated as unset.
pub fn parse_dotenv(content: &str) -> Result<HashMap<String, String>> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let mut entries = HashMap::new();

    for (idx, raw_line) in content.lines().enumerate() {
        let line_num = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let line = strip_export_prefix(line);
        let Some((raw_key, raw_val)) = line.split_once('=') else {
            eyre::bail!("line {line_num}: expected KEY=VALUE");
        };

        let key = raw_key.trim();
        if !is_valid_env_key(key) {
            eyre::bail!("line {line_num}: invalid variable name '{key}'");
        }

        let value = parse_value(raw_val.trim_start(), key, line_num)?;
        if value.is_empty() {
            entries.remove(key);
        } else {
            entries.insert(key.to_string(), value);
        }
    }

    Ok(entries)
}

fn strip_export_prefix(line: &str) -> &str {
    match line.strip_prefix("export") {
        Some(rest) if rest.starts_with([' ', '\t']) => rest.trim_start(),
        _ => line,
    }
}

fn is_valid_env_key(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn parse_value(raw: &str, key: &str, line_num: usize) -> Result<String> {
    if let Some(rest) = raw.strip_prefix('"') {
        parse_double_quoted(rest, key, line_num)
    } else if let Some(rest) = raw.strip_prefix('\'') {
        parse_single_quoted(rest, key, line_num)
    } else {
        Ok(strip_unquoted_comment(raw).trim_end().to_string())
    }
}

fn parse_double_quoted(rest: &str, key: &str, line_num: usize) -> Result<String> {
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                let trailing = chars.as_str().trim();
                if !trailing.is_empty() && !trailing.starts_with('#') {
                    eyre::bail!(
                        "line {line_num}: unexpected characters after closing quote for '{key}'"
                    );
                }
                return Ok(out);
            }
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => {
                    eyre::bail!("line {line_num}: unterminated double-quoted value for '{key}'")
                }
            },
            other => out.push(other),
        }
    }
    eyre::bail!("line {line_num}: unterminated double-quoted value for '{key}'")
}

fn parse_single_quoted(rest: &str, key: &str, line_num: usize) -> Result<String> {
    let Some((value, after)) = rest.split_once('\'') else {
        eyre::bail!("line {line_num}: unterminated single-quoted value for '{key}'");
    };
    let trailing = after.trim();
    if !trailing.is_empty() && !trailing.starts_with('#') {
        eyre::bail!("line {line_num}: unexpected characters after closing quote for '{key}'");
    }
    Ok(value.to_string())
}

/// Strip an inline comment (` # ...`) from an unquoted value. A `#` without preceding
/// whitespace is kept verbatim so keys/tokens containing `#` are not truncated.
fn strip_unquoted_comment(raw: &str) -> &str {
    let bytes = raw.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'#' && i > 0 && bytes[i - 1].is_ascii_whitespace() {
            return &raw[..i];
        }
    }
    raw
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn parses_standard_dotenv_forms() {
        let input = "\u{feff}\
# leading comment
   # indented comment

OPENAI_API_KEY=sk-plain
export ANTHROPIC_API_KEY=\"sk-ant-quoted\" # inline comment
GEMINI_API_KEY = 'sk-gem-#literal' # single quoted
ESCAPED=\"line1\\nline2\\t\\\"q\\\"\\\\\"
INLINE_HASH=tok#not-a-comment # real comment
SPACED =   hello world   # trailing comment
EMPTY=
EMPTY_QUOTED=\"\"
";
        let parsed = parse_dotenv(input).unwrap();
        assert_eq!(parsed.get("OPENAI_API_KEY").unwrap(), "sk-plain");
        assert_eq!(parsed.get("ANTHROPIC_API_KEY").unwrap(), "sk-ant-quoted");
        assert_eq!(parsed.get("GEMINI_API_KEY").unwrap(), "sk-gem-#literal");
        assert_eq!(parsed.get("ESCAPED").unwrap(), "line1\nline2\t\"q\"\\");
        assert_eq!(parsed.get("INLINE_HASH").unwrap(), "tok#not-a-comment");
        assert_eq!(parsed.get("SPACED").unwrap(), "hello world");
        assert!(!parsed.contains_key("EMPTY"));
        assert!(!parsed.contains_key("EMPTY_QUOTED"));
    }

    #[test]
    fn does_not_strip_export_from_key_name() {
        let parsed = parse_dotenv("export_token=abc\n").unwrap();
        assert_eq!(parsed.get("export_token").unwrap(), "abc");
    }

    #[test]
    fn later_line_in_same_file_wins() {
        let parsed = parse_dotenv("KEY=first\nKEY=second\n").unwrap();
        assert_eq!(parsed.get("KEY").unwrap(), "second");
    }

    #[test]
    fn rejects_malformed_lines() {
        assert!(parse_dotenv("NO_EQUALS_SIGN\n").is_err());
        assert!(parse_dotenv("123BAD=val\n").is_err());
        assert!(parse_dotenv("BAD KEY=val\n").is_err());
        assert!(parse_dotenv("KEY=\"unterminated\n").is_err());
        assert!(parse_dotenv("KEY='unterminated\n").is_err());
        assert!(parse_dotenv("KEY=\"ok\" trailing\n").is_err());
        assert!(parse_dotenv("KEY='ok' trailing\n").is_err());
    }

    #[test]
    fn earlier_file_takes_precedence_and_missing_files_are_ignored() {
        let dir = TempDir::new().unwrap();
        let repo_env = dir.path().join(".env.nitpicker");
        let global_env = dir.path().join("global.env");
        let missing_env = dir.path().join("missing.env");

        std::fs::write(&repo_env, "SHARED=from-repo\nREPO_ONLY=1\nEMPTY_OVERRIDE=\n").unwrap();
        std::fs::write(
            &global_env,
            "SHARED=from-global\nGLOBAL_ONLY=2\nEMPTY_OVERRIDE=from-global\n",
        )
        .unwrap();

        let merged = read_files(&[&missing_env, &repo_env, &global_env]).unwrap();
        assert_eq!(merged.get("SHARED").unwrap(), "from-repo");
        assert_eq!(merged.get("REPO_ONLY").unwrap(), "1");
        assert_eq!(merged.get("GLOBAL_ONLY").unwrap(), "2");
        assert_eq!(merged.get("EMPTY_OVERRIDE").unwrap(), "from-global");
    }

    #[test]
    fn load_files_populates_fallback_without_overriding_process_env() {
        let dir = TempDir::new().unwrap();
        let env_file = dir.path().join(".env.nitpicker");
        std::fs::write(
            &env_file,
            "NITPICKER_TEST_ENV_FILE_ONLY=from-file\nPATH=should-not-override\n",
        )
        .unwrap();

        load_files(&[&env_file]).unwrap();
        assert_eq!(var("NITPICKER_TEST_ENV_FILE_ONLY").unwrap(), "from-file");
        // Process env (PATH is always set) must win over the file entry.
        assert_ne!(var("PATH").unwrap(), "should-not-override");

        // Replacing with an empty list clears previously loaded entries.
        load_files(&[]).unwrap();
        assert!(var("NITPICKER_TEST_ENV_FILE_ONLY").is_err());
    }
}
