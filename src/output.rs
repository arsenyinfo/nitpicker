//! machine-readable output contract for the `pr` subcommand.
//!
//! when `--format json` is set, `nitpicker pr` emits exactly one [`PrReviewOutput`]
//! line on stdout and nothing else; all human output (logs, spinners, debate
//! chatter) is routed to stderr. see the "server / embedding" section of the README.

use eyre::Result;
use serde::Serialize;
use std::io::Write;

/// stdout shape for the `pr` subcommand. `Text` keeps the legacy human output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Ok,
    Error,
}

/// debate (actor/critic) vs. parallel aggregation.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewMode {
    Debate,
    Parallel,
}

#[derive(Debug, Serialize)]
pub struct PrInfo {
    pub url: Option<String>,
    pub number: Option<u32>,
    pub title: String,
    pub head_sha: String,
}

#[derive(Debug, Serialize)]
pub struct Models {
    pub reviewers: Vec<String>,
    pub aggregator: String,
}

/// the single JSON object written to stdout in `--format json` mode.
///
/// `status: ok` means the review process ran to completion and produced a report;
/// the report body may still contain per-reviewer failure notes (partial failures
/// are folded into the markdown, not surfaced as a separate field in v1).
#[derive(Debug, Serialize)]
pub struct PrReviewOutput {
    pub schema_version: u32,
    pub status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pr: Option<PrInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<ReviewMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Models>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_markdown: Option<String>,
    pub comment_posted: bool,
    pub duration_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub const SCHEMA_VERSION: u32 = 1;

impl PrReviewOutput {
    pub fn error(message: String, duration_ms: u64) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            status: Status::Error,
            pr: None,
            mode: None,
            models: None,
            report_markdown: None,
            comment_posted: false,
            duration_ms,
            error: Some(message),
        }
    }
}

/// serialize `value` as one line on stdout and flush before returning.
///
/// flushing explicitly matters because callers exit via `std::process::exit`,
/// which skips the normal stdout teardown that would otherwise drain the buffer.
pub fn emit_json<T: Serialize>(value: &T) -> Result<()> {
    let line = serde_json::to_string(value)?;
    let mut out = std::io::stdout().lock();
    writeln!(out, "{line}")?;
    out.flush()?;
    Ok(())
}
