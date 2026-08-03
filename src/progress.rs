use indicatif::MultiProgress;
use std::io::{self, IsTerminal, Write};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use unicode_width::UnicodeWidthStr;

const DEFAULT_TERMINAL_COLUMNS: usize = 80;
const PROGRESS_BAR_RESERVED_COLUMNS: usize = 15;
const MAX_MESSAGE_COLUMNS: usize = 120;

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

/// Token counts for a progress line, where columns are scarce: `1038095` becomes `1.0M`.
/// Exact counts stay available in the `info!` logs and the `pr --json` usage block.
pub(crate) fn compact_tokens(tokens: u64) -> String {
    match tokens {
        0..=999 => tokens.to_string(),
        // one decimal only while it carries information: 3.5k, then 938k, then 1.0M
        1_000..=9_999 => format!("{:.1}k", tokens as f64 / 1_000.0),
        10_000..=999_999 => format!("{}k", tokens / 1_000),
        _ => format!("{:.1}M", tokens as f64 / 1_000_000.0),
    }
}

/// The prompt-cache share of a run, as `1.0M in · 90% cached`. A ratio rather than a second token
/// count: it is what answers "is caching working", and it costs a third of the width. Returns just
/// the token count when the provider reported no prompt at all, since `0% cached` of nothing is
/// noise rather than a signal.
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
    fn input_with_cache_share_reports_the_ratio() {
        assert_eq!(
            input_with_cache_share(1_038_095, 938_000),
            "1.0M in · 90% cached"
        );
        assert_eq!(input_with_cache_share(1_000, 0), "1.0k in · 0% cached");
    }

    /// A failed or unmetered turn reports no prompt at all; "0% cached" of nothing would read as a
    /// caching problem rather than as missing data.
    #[test]
    fn input_with_cache_share_omits_the_ratio_without_a_prompt() {
        assert_eq!(input_with_cache_share(0, 0), "0 in");
    }

    /// A provider contradicting itself (more cache than prompt) is reported verbatim by
    /// `TokenUsage`, so the share must still be a percentage rather than exceeding 100.
    #[test]
    fn input_with_cache_share_cannot_exceed_one_hundred_percent() {
        assert_eq!(input_with_cache_share(10, 5_000), "10 in · 100% cached");
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
