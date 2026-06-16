//! machine-readable output contract for the `pr` subcommand.
//!
//! when `--format json` is set, `nitpicker pr` emits exactly one [`PrReviewOutput`]
//! line on stdout and nothing else; all human output (logs, spinners, debate
//! chatter) is routed to stderr. see the "server / embedding" section of the README.

use nitpicker_agent::llm::TokenUsage;
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

/// aggregate token + subagent usage for one review run.
///
/// best-effort metering: totals count tokens reported by *successful* LLM
/// completions only (reviewers/debate turns plus the aggregator/meta call). A
/// failed reviewer, subagent, or discarded retry contributes nothing, so this
/// is a lower bound on actual spend, not an exact meter.
#[derive(Debug, Clone, Default, Serialize)]
pub struct UsageReport {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub subagents_spawned: usize,
}

impl UsageReport {
    /// fold one completion's token usage (and any subagents it spawned) into the totals.
    pub fn add(&mut self, usage: TokenUsage, subagents_spawned: usize) {
        self.input_tokens = self.input_tokens.saturating_add(usage.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(usage.output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(usage.total_tokens);
        self.subagents_spawned += subagents_spawned;
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageReport>,
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
            usage: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_add_folds_tokens_and_subagents() {
        let mut usage = UsageReport::default();
        usage.add(TokenUsage::new(100, 30), 2);
        usage.add(TokenUsage::new(10, 5), 1);
        assert_eq!(usage.input_tokens, 110);
        assert_eq!(usage.output_tokens, 35);
        assert_eq!(usage.total_tokens, 145);
        assert_eq!(usage.subagents_spawned, 3);
    }

    #[test]
    fn usage_add_saturates_token_overflow() {
        let mut usage = UsageReport {
            input_tokens: u64::MAX,
            ..Default::default()
        };
        usage.add(TokenUsage::new(5, 0), 0);
        assert_eq!(usage.input_tokens, u64::MAX);
    }

    #[test]
    fn ok_envelope_serializes_usage_block() {
        let mut usage = UsageReport::default();
        usage.add(TokenUsage::new(120000, 8000), 6);
        let envelope = PrReviewOutput {
            schema_version: SCHEMA_VERSION,
            status: Status::Ok,
            pr: None,
            mode: None,
            models: None,
            report_markdown: None,
            usage: Some(usage),
            comment_posted: false,
            duration_ms: 1,
            error: None,
        };
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&envelope).unwrap()).unwrap();
        assert_eq!(json["usage"]["input_tokens"], 120000);
        assert_eq!(json["usage"]["output_tokens"], 8000);
        assert_eq!(json["usage"]["total_tokens"], 128000);
        assert_eq!(json["usage"]["subagents_spawned"], 6);
    }

    #[test]
    fn error_envelope_omits_usage() {
        let envelope = PrReviewOutput::error("boom".to_string(), 1);
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&envelope).unwrap()).unwrap();
        assert!(json.get("usage").is_none(), "usage must be omitted on error");
    }
}
