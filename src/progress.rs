use indicatif::MultiProgress;
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, Weak, mpsc};
use std::thread;
use std::time::Duration;
use unicode_width::UnicodeWidthStr;

const DEFAULT_TERMINAL_COLUMNS: usize = 80;
const PROGRESS_BAR_RESERVED_COLUMNS: usize = 15;
const MAX_MESSAGE_COLUMNS: usize = 120;
const MAX_TERMINAL_PROJECT_CHARS: usize = 24;
const TERMINAL_TITLE_SPINNER_INTERVAL: Duration = Duration::from_millis(200);
// Keep the target's outer circle fixed so the repository title does not appear to shift.
const TERMINAL_TITLE_SPINNER_FRAMES: [&str; 4] = ["⊙", "⊕", "⊗", "⊕"];

static ACTIVE_PROGRESS: OnceLock<Mutex<Option<Weak<MultiProgress>>>> = OnceLock::new();

pub(crate) struct ActiveProgressGuard {
    previous: Option<Weak<MultiProgress>>,
}

pub(crate) fn set_active_progress(progress: &Arc<MultiProgress>) -> ActiveProgressGuard {
    let mut active = active_progress().lock().unwrap_or_else(|e| e.into_inner());
    let previous = active.replace(Arc::downgrade(progress));
    ActiveProgressGuard { previous }
}

impl Drop for ActiveProgressGuard {
    fn drop(&mut self) {
        let mut active = active_progress().lock().unwrap_or_else(|e| e.into_inner());
        *active = self.previous.take();
    }
}

/// Keeps the terminal tab/window title animated for the lifetime of a review run.
///
/// The title is written only for human text output when stdout is a terminal. That keeps
/// redirected reports and the `pr --json` contract byte-for-byte clean while matching the surface
/// a user is actually looking at during an interactive run. Dropping the guard stops the worker
/// and clears the title we set; terminal titles cannot be read back portably, so restoring an
/// earlier value is not possible.
pub(crate) struct TerminalTitleGuard {
    stop: mpsc::Sender<()>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Drop for TerminalTitleGuard {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Start nitpicker's fixed-outline target `spinner project` indicator in the terminal header.
pub(crate) fn start_terminal_title(
    repo: &Path,
    format: crate::output::OutputFormat,
) -> Option<TerminalTitleGuard> {
    // JSON owns the entire stdout byte stream even when a caller allocates a PTY. Check the
    // output contract before TTY detection so no OSC title bytes can precede its one object.
    let term = std::env::var("TERM").ok();
    if !terminal_title_enabled(format, io::stdout().is_terminal(), term.as_deref()) {
        return None;
    }

    let project = terminal_project_name(repo);
    let (stop, stopped) = mpsc::channel();
    let worker = thread::Builder::new()
        .name("nitpicker-terminal-title".to_string())
        .spawn(move || {
            let mut frame = 0usize;
            loop {
                if write_osc_title(&format!(
                    "{} {project}",
                    TERMINAL_TITLE_SPINNER_FRAMES[frame]
                ))
                .is_err()
                {
                    break;
                }
                match stopped.recv_timeout(TERMINAL_TITLE_SPINNER_INTERVAL) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        frame = (frame + 1) % TERMINAL_TITLE_SPINNER_FRAMES.len();
                    }
                }
            }
            let _ = clear_terminal_title();
        })
        .ok()?;

    Some(TerminalTitleGuard {
        stop,
        worker: Some(worker),
    })
}

fn terminal_title_enabled(
    format: crate::output::OutputFormat,
    stdout_is_terminal: bool,
    term: Option<&str>,
) -> bool {
    format == crate::output::OutputFormat::Text
        && stdout_is_terminal
        && !matches!(term, Some("dumb" | "linux"))
}

fn terminal_project_name(repo: &Path) -> String {
    let raw = repo
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "nitpicker".to_string());
    let project = sanitize_terminal_project(&raw);
    match project.is_empty() {
        true => "nitpicker".to_string(),
        false => project,
    }
}

fn clear_terminal_title() -> io::Result<()> {
    write_osc_title("")
}

fn write_osc_title(title: &str) -> io::Result<()> {
    let mut out = io::stdout().lock();
    write_osc_title_to(&mut out, title)
}

fn write_osc_title_to(out: &mut impl Write, title: &str) -> io::Result<()> {
    write!(out, "\x1b]0;{title}\x07")?;
    out.flush()
}

/// Normalize and bound an untrusted path component before placing it inside an OSC sequence.
fn sanitize_terminal_project(project: &str) -> String {
    let mut sanitized = String::new();
    let mut chars_written = 0usize;
    let mut pending_space = false;

    for ch in project.chars() {
        if ch.is_whitespace() {
            pending_space = !sanitized.is_empty();
            continue;
        }
        if is_disallowed_terminal_title_char(ch) {
            continue;
        }
        if pending_space && chars_written + 1 < MAX_TERMINAL_PROJECT_CHARS {
            sanitized.push(' ');
            chars_written += 1;
            pending_space = false;
        }
        if chars_written >= MAX_TERMINAL_PROJECT_CHARS {
            break;
        }
        sanitized.push(ch);
        chars_written += 1;
    }
    sanitized
}

fn is_disallowed_terminal_title_char(ch: char) -> bool {
    ch.is_control()
        || matches!(
            ch,
            '\u{00AD}'
                | '\u{034F}'
                | '\u{061C}'
                | '\u{180E}'
                | '\u{200B}'..='\u{200F}'
                | '\u{202A}'..='\u{202E}'
                | '\u{2060}'..='\u{206F}'
                | '\u{FE00}'..='\u{FE0F}'
                | '\u{FEFF}'
                | '\u{FFF9}'..='\u{FFFB}'
                | '\u{1BCA0}'..='\u{1BCA3}'
                | '\u{E0100}'..='\u{E01EF}'
        )
}

pub(crate) fn stderr_log_writer() -> ProgressLogWriter {
    ProgressLogWriter
}

pub(crate) fn stderr_is_terminal() -> bool {
    io::stderr().is_terminal()
}

pub(crate) fn stderr_supports_color() -> bool {
    stderr_is_terminal() && color_env_allows()
}

pub(crate) fn color_env_allows() -> bool {
    std::env::var_os("NO_COLOR").is_none() && std::env::var("TERM").as_deref() != Ok("dumb")
}

pub(crate) struct ProgressLogWriter;

impl Write for ProgressLogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        with_active_progress(|| io::stderr().write(buf))
    }

    fn flush(&mut self) -> io::Result<()> {
        with_active_progress(|| io::stderr().flush())
    }
}

fn active_progress() -> &'static Mutex<Option<Weak<MultiProgress>>> {
    ACTIVE_PROGRESS.get_or_init(|| Mutex::new(None))
}

fn with_active_progress<T>(f: impl FnOnce() -> io::Result<T>) -> io::Result<T> {
    let progress = active_progress()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(Weak::upgrade);
    match progress {
        Some(progress) => progress.suspend(f),
        None => f(),
    }
}

pub(crate) fn bar_message(message: impl AsRef<str>) -> String {
    bar_message_for_columns(message.as_ref(), terminal_columns())
}

pub(crate) fn detail_message(prefix: &str, detail: Option<&str>) -> String {
    detail_message_for_columns(prefix, detail, terminal_columns())
}

fn bar_message_for_columns(message: &str, columns: usize) -> String {
    truncate_single_line(
        message,
        columns
            .saturating_sub(PROGRESS_BAR_RESERVED_COLUMNS)
            .min(MAX_MESSAGE_COLUMNS),
    )
}

fn detail_message_for_columns(prefix: &str, detail: Option<&str>, columns: usize) -> String {
    let Some(detail) = detail else {
        return String::new();
    };
    let detail = truncate_single_line(
        detail,
        columns
            .saturating_sub(UnicodeWidthStr::width(prefix))
            .min(MAX_MESSAGE_COLUMNS),
    );
    match detail.is_empty() {
        true => String::new(),
        false => format!("{prefix}{detail}"),
    }
}

/// Token counts for a progress line, where columns are scarce: `1038095` becomes `1.0M`. Exact
/// counts stay in the `info!` logs and the `pr --json` usage block.
pub(crate) fn compact_tokens(tokens: u64) -> String {
    match tokens {
        0..=999 => tokens.to_string(),
        // one decimal only while it carries information: 3.5k, then 938k, then 1.0M
        1_000..=9_999 => format!("{:.1}k", tokens as f64 / 1_000.0),
        10_000..=999_999 => format!("{}k", tokens / 1_000),
        _ => format!("{:.1}M", tokens as f64 / 1_000_000.0),
    }
}

/// The prompt-cache share, as `1.0M in · 90% cached` — a ratio because it answers "is caching
/// working" in a third of the width. Drops the ratio when no prompt was reported at all.
pub(crate) fn input_with_cache_share(input_tokens: u64, cached_input_tokens: u64) -> String {
    let compact = compact_tokens(input_tokens);
    match input_tokens {
        0 => format!("{compact} in"),
        _ => {
            let share = cached_input_tokens.min(input_tokens) * 100 / input_tokens;
            format!("{compact} in · {share}% cached")
        }
    }
}

/// `COLUMNS` is a *shell* variable that zsh and bash do not export, so a child process almost
/// never sees it — reading only that pinned every message to the 80-column fallback regardless of
/// the real terminal, which is what truncated the completed-round line. The env var stays as a
/// first-choice override (tests, CI, deliberate narrowing); otherwise ask the terminal itself.
fn terminal_columns() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|columns| *columns > 0)
        .or_else(|| {
            console::Term::stderr()
                .size_checked()
                .map(|(_rows, columns)| columns as usize)
                .filter(|columns| *columns > 0)
        })
        .unwrap_or(DEFAULT_TERMINAL_COLUMNS)
}

fn truncate_single_line(message: &str, max_columns: usize) -> String {
    let normalized = normalize_whitespace(message);
    if UnicodeWidthStr::width(normalized.as_str()) <= max_columns {
        return normalized;
    }
    if max_columns <= 3 {
        let boundary = floor_display_width_boundary(&normalized, max_columns);
        return normalized[..boundary].to_string();
    }
    let boundary = floor_display_width_boundary(&normalized, max_columns.saturating_sub(3));
    let mut truncated = normalized[..boundary].to_string();
    truncated.push_str("...");
    truncated
}

fn floor_display_width_boundary(message: &str, max_columns: usize) -> usize {
    let mut boundary = 0;
    for (idx, ch) in message.char_indices() {
        let next = idx + ch.len_utf8();
        if UnicodeWidthStr::width(&message[..next]) > max_columns {
            break;
        }
        boundary = next;
    }
    boundary
}

fn normalize_whitespace(message: &str) -> String {
    let mut normalized = String::new();
    for part in message.split_whitespace() {
        if !normalized.is_empty() {
            normalized.push(' ');
        }
        normalized.push_str(part);
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn terminal_title_sanitizes_controls_invisible_text_and_whitespace() {
        assert_eq!(
            sanitize_terminal_project("  revi\t\u{202e}ewer\n\x1b]0;owned\x07  "),
            "revi ewer ]0;owned"
        );
    }

    #[test]
    fn terminal_project_bound_counts_only_sanitized_visible_characters() {
        let project = format!("{}reviewer", "\u{202e}".repeat(40));
        assert_eq!(sanitize_terminal_project(&project), "reviewer");
    }

    #[test]
    fn json_output_disables_terminal_titles_even_inside_a_pty() {
        assert!(!terminal_title_enabled(
            crate::output::OutputFormat::Json,
            true,
            Some("xterm-256color")
        ));
        assert!(terminal_title_enabled(
            crate::output::OutputFormat::Text,
            true,
            Some("xterm-256color")
        ));
    }

    #[test]
    fn terminal_title_excludes_terms_without_window_titles() {
        for term in ["dumb", "linux"] {
            assert!(!terminal_title_enabled(
                crate::output::OutputFormat::Text,
                true,
                Some(term)
            ));
        }
    }

    #[test]
    fn terminal_project_name_is_bounded_and_has_a_root_fallback() {
        assert_eq!(terminal_project_name(Path::new("/")), "nitpicker");
        assert_eq!(
            terminal_project_name(Path::new(
                "/tmp/a-project-name-that-is-longer-than-twenty-four-characters"
            )),
            "a-project-name-that-is-l"
        );
    }

    #[test]
    fn osc_title_uses_bel_terminated_osc_zero() {
        let mut bytes = Vec::new();
        write_osc_title_to(&mut bytes, "⊙ reviewer").unwrap();
        assert_eq!(bytes, b"\x1b]0;\xe2\x8a\x99 reviewer\x07");
    }

    #[test]
    fn compact_tokens_switches_unit_at_each_boundary() {
        assert_eq!(compact_tokens(0), "0");
        assert_eq!(compact_tokens(999), "999");
        assert_eq!(compact_tokens(1_000), "1.0k");
        assert_eq!(compact_tokens(3_509), "3.5k");
        assert_eq!(compact_tokens(9_999), "10.0k");
        assert_eq!(compact_tokens(10_000), "10k");
        assert_eq!(compact_tokens(938_000), "938k");
        assert_eq!(compact_tokens(999_999), "999k");
        assert_eq!(compact_tokens(1_038_095), "1.0M");
        assert_eq!(compact_tokens(2_275_854), "2.3M");
    }

    #[test]
    fn input_with_cache_share_covers_ratio_zero_prompt_and_clamp() {
        let cases = [
            (1_038_095, 938_000, "1.0M in · 90% cached"),
            (1_000, 0, "1.0k in · 0% cached"),
            // no prompt at all (a failed or unmetered turn): "0% cached" of nothing would read as
            // a caching problem rather than as missing data, so the ratio is dropped
            (0, 0, "0 in"),
            // a provider contradicting itself is reported verbatim by `TokenUsage`, so the share
            // still has to be a percentage rather than exceed 100
            (10, 5_000, "10 in · 100% cached"),
        ];
        for (input, cached, expected) in cases {
            assert_eq!(input_with_cache_share(input, cached), expected);
        }
    }

    #[test]
    fn bar_message_reserves_progress_columns() {
        assert_eq!(bar_message_for_columns("abcdefghij", 20), "ab...");
    }

    #[test]
    fn detail_message_collapses_whitespace_and_fits_columns() {
        assert_eq!(
            detail_message_for_columns("    -> ", Some("one\ntwo\tthree"), 18),
            "    -> one two ..."
        );
    }

    #[test]
    fn detail_message_omits_empty_detail() {
        assert_eq!(
            detail_message_for_columns("    -> ", Some(" \n\t "), 80),
            ""
        );
        assert_eq!(detail_message_for_columns("    -> ", None, 80), "");
    }

    #[test]
    fn truncates_by_display_width() {
        let wide = "\u{8868}";
        assert_eq!(truncate_single_line(&format!("ab{wide}cd"), 5), "ab...");
        assert_eq!(truncate_single_line(&format!("{wide}abc"), 1), "");
        assert_eq!(
            detail_message_for_columns("-> ", Some(&format!("{wide}abc")), 5),
            format!("-> {wide}")
        );
    }
}
