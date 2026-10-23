//! Mechanics shared by the parallel review and debate orchestrators: session setup, the
//! progress display, and the final synthesis completion with its aggregation record.

use crate::progress::ActiveProgressGuard;
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nitpicker_agent::config::Config;
use nitpicker_agent::llm::{
    Completion, CompletionResponse, FinishReason, LLMClientDyn, TokenUsage, throttled_completion,
};
use nitpicker_agent::session::{AggregationRecord, SessionLogger};
use nitpicker_agent::tools::floor_char_boundary;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{Instrument, info, warn};

/// Open the trajectory session (when enabled) and persist its attribution sidecar.
pub(crate) async fn open_session(config: &Config) -> Result<Option<SessionLogger>> {
    let logger = SessionLogger::maybe_new(config.log_trajectories())?;
    if let Some(logger) = &logger {
        info!(path = %logger.root().display(), "trajectory logging enabled");
        tracing::Span::current().record("nitpicker.session.id", logger.id());
        if let Err(err) = logger
            .write_attribution(&crate::prompts::session_attribution())
            .await
        {
            warn!(error = ?err, "failed to persist session attribution");
        }
    }
    Ok(logger)
}

/// The run's progress rows: hidden under `--verbose`, and registered as the active display
/// (so log lines don't tear it) only when stderr is a terminal.
pub(crate) fn progress_display(verbose: bool) -> (Arc<MultiProgress>, Option<ActiveProgressGuard>) {
    let mp = Arc::new(MultiProgress::new());
    if verbose {
        mp.set_draw_target(ProgressDrawTarget::hidden());
    }
    let guard = (!verbose && crate::progress::stderr_is_terminal())
        .then(|| crate::progress::set_active_progress(&mp));
    (mp, guard)
}

/// The run's `aggregation.json` minus its outcome: `base` carries the configured aggregator
/// model, an empty `text`, and no `error`.
pub(crate) struct RunRecord<'a> {
    pub(crate) logger: Option<&'a SessionLogger>,
    pub(crate) base: AggregationRecord,
}

impl RunRecord<'_> {
    /// Best-effort: a run that failed is the one most in need of a durable record, but a
    /// failed write must not replace the error that ended the run.
    pub(crate) async fn persist_failure(self, err: &eyre::Report, what: &str) {
        let written = match self.logger {
            Some(logger) => {
                logger
                    .write_aggregation(&failure_record(self.base, err))
                    .await
            }
            None => Ok(()),
        };
        match written {
            Ok(()) => {}
            Err(write_err) => warn!(error = ?write_err, "failed to persist {what} record"),
        }
    }

    pub(crate) async fn persist_success(self, model: String, text: String) -> Result<()> {
        match self.logger {
            Some(logger) => {
                logger
                    .write_aggregation(&success_record(self.base, model, text))
                    .await
            }
            None => Ok(()),
        }
    }
}

fn failure_record(base: AggregationRecord, err: &eyre::Report) -> AggregationRecord {
    AggregationRecord {
        error: Some(bounded_error_string(err)),
        ..base
    }
}

fn success_record(base: AggregationRecord, model: String, text: String) -> AggregationRecord {
    AggregationRecord {
        model,
        text,
        ..base
    }
}

/// One non-agentic synthesis completion over the collected evidence.
pub(crate) struct Synthesis<'a> {
    /// An `Err` is an aggregator that could not be built; it fails the synthesis unwrapped.
    pub(crate) client: Result<Arc<dyn LLMClientDyn>>,
    pub(crate) completion: Completion,
    /// Names the step when a response is rejected ("aggregator", "meta-review").
    pub(crate) role: &'static str,
    /// Preset runs wrap completion failures with their run context; Ask/Topic propagate the
    /// provider error untouched.
    pub(crate) failure_context: Option<String>,
    pub(crate) progress: ProgressBar,
    pub(crate) done_style: ProgressStyle,
    pub(crate) llm_semaphore: &'a Semaphore,
}

pub(crate) struct SynthesisOutput {
    pub(crate) text: String,
    pub(crate) model: String,
    pub(crate) usage: TokenUsage,
}

pub(crate) async fn synthesize(
    synthesis: Synthesis<'_>,
    record: RunRecord<'_>,
) -> Result<SynthesisOutput> {
    let Synthesis {
        client,
        completion,
        role,
        failure_context,
        progress,
        done_style,
        llm_semaphore,
    } = synthesis;
    let configured_model = completion.model.clone();
    let span = crate::telemetry::synthesis_span(&configured_model);
    let result = async {
        let client = client?;
        // every job/lane has finished, so the permit is free; routing through the chokepoint
        // keeps "every completion is a chat span" true for the synthesis too
        throttled_completion(llm_semaphore, &client, completion)
            .await
            .and_then(|response| {
                validate_synthesis_response(&response, role)?;
                Ok(response)
            })
            .map_err(|err| match failure_context {
                Some(context) => crate::presets::synthesis_failure(err, context),
                None => err,
            })
    }
    .instrument(span.clone())
    .await;
    progress.set_style(done_style);
    match result {
        Ok(response) => {
            progress.finish_with_message("✓ done");
            let model = synthesis_model(&response, &configured_model);
            let text = response.text();
            record.persist_success(model.clone(), text.clone()).await?;
            Ok(SynthesisOutput {
                text,
                model,
                usage: response.usage,
            })
        }
        Err(err) => {
            span.record("otel.status_code", "ERROR");
            progress.finish_with_message(crate::progress::bar_message("✗ synthesis failed"));
            record.persist_failure(&err, "synthesis-failure").await;
            Err(err)
        }
    }
}

fn validate_synthesis_response(response: &CompletionResponse, role: &str) -> Result<()> {
    match &response.finish_reason {
        FinishReason::ToolUse => eyre::bail!("{role} returned tool calls unexpectedly"),
        FinishReason::MaxTokens => {
            let model = response.selected_model.as_deref().unwrap_or("unknown");
            eyre::bail!(
                "{role} model '{model}' reached its output token limit and returned a truncated verdict"
            )
        }
        FinishReason::None | FinishReason::Stop | FinishReason::Other(_) => Ok(()),
    }
}

fn synthesis_model(response: &CompletionResponse, configured_model: &str) -> String {
    response
        .selected_model
        .clone()
        .unwrap_or_else(|| configured_model.to_string())
}

/// Bounded `{err:#}` chain for persistence — a provider error body can be huge, and the
/// session record is not the place to store it whole.
fn bounded_error_string(err: &eyre::Report) -> String {
    const MAX: usize = 4000;
    let full = format!("{err:#}");
    match full.len() <= MAX {
        true => full,
        false => format!("{}…", &full[..floor_char_boundary(&full, MAX)]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nitpicker_agent::llm::LLMClient;
    use nitpicker_agent::session::{JobRecord, LaneRecord, VerdictRecord};
    use rig_core::OneOrMany;
    use rig_core::completion::{AssistantContent, Message};
    use rig_core::providers::openai;

    fn synthesis_response(finish_reason: FinishReason) -> CompletionResponse {
        CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text("partial verdict")),
            finish_reason,
            usage: TokenUsage::default(),
            selected_model: Some("fallback-model".to_string()),
        }
    }

    #[test]
    fn synthesis_rejects_truncated_or_tool_call_responses() {
        let truncated = synthesis_response(FinishReason::MaxTokens);
        let err = validate_synthesis_response(&truncated, "aggregator").unwrap_err();
        assert!(format!("{err:#}").contains("truncated verdict"));
        assert!(format!("{err:#}").contains("fallback-model"));

        let tool_call = synthesis_response(FinishReason::ToolUse);
        assert!(validate_synthesis_response(&tool_call, "aggregator").is_err());

        let complete = synthesis_response(FinishReason::Stop);
        assert!(validate_synthesis_response(&complete, "aggregator").is_ok());
        assert_eq!(synthesis_model(&complete, "configured"), "fallback-model");

        let mut without_selected_model = complete;
        without_selected_model.selected_model = None;
        assert_eq!(
            synthesis_model(&without_selected_model, "configured"),
            "configured"
        );
    }

    fn base_record() -> AggregationRecord {
        AggregationRecord {
            kind: "aggregation".to_string(),
            model: "configured-aggregator".to_string(),
            text: String::new(),
            error: None,
            rounds: Some(2),
            converged: Some(false),
            presets: Some(vec!["correctness".to_string()]),
            lanes: Some(vec![LaneRecord {
                preset: "correctness".to_string(),
                rounds: 2,
                converged: false,
                degraded: true,
            }]),
            verdicts: vec![VerdictRecord {
                lens: Some("correctness".to_string()),
                stage: "Reviewer · round 1".to_string(),
                text: "verdict".to_string(),
                ok: true,
            }],
            jobs: Some(vec![JobRecord {
                label: "correctness · a".to_string(),
                preset: Some("correctness".to_string()),
                ok: false,
            }]),
        }
    }

    #[test]
    fn failure_record_keeps_run_metadata_and_configured_model() {
        let err = eyre::eyre!("provider down");
        let record = serde_json::to_value(failure_record(base_record(), &err)).unwrap();
        let mut expected = serde_json::to_value(base_record()).unwrap();
        expected["error"] = serde_json::Value::String(bounded_error_string(&err));
        assert_eq!(record, expected);
    }

    #[test]
    fn success_record_reports_the_route_that_answered() {
        let record = serde_json::to_value(success_record(
            base_record(),
            "fallback-route".to_string(),
            "final".to_string(),
        ))
        .unwrap();
        let mut expected = serde_json::to_value(base_record()).unwrap();
        expected["model"] = "fallback-route".into();
        expected["text"] = "final".into();
        assert_eq!(record, expected);
        assert!(record.get("error").is_none());
    }

    #[test]
    fn bounded_error_string_truncates_on_a_char_boundary() {
        let err = eyre::eyre!("é".repeat(3000));
        let bounded = bounded_error_string(&err);
        assert!(bounded.len() <= 4000 + "…".len());
        assert!(bounded.ends_with('…'));
        let short = eyre::eyre!("short");
        assert_eq!(bounded_error_string(&short), "short");
    }

    fn completion() -> Completion {
        Completion {
            model: "configured-aggregator".to_string(),
            prompt: Message::user("synthesize"),
            preamble: None,
            history: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            additional_params: None,
        }
    }

    fn synthesis<'a>(
        client: Result<Arc<dyn LLMClientDyn>>,
        failure_context: Option<&str>,
        semaphore: &'a Semaphore,
    ) -> Synthesis<'a> {
        Synthesis {
            client,
            completion: completion(),
            role: "aggregator",
            failure_context: failure_context.map(str::to_string),
            progress: ProgressBar::hidden(),
            done_style: ProgressStyle::default_spinner(),
            llm_semaphore: semaphore,
        }
    }

    fn unreachable_client() -> Arc<dyn LLMClientDyn> {
        openai::CompletionsClient::builder()
            .api_key("unused")
            .base_url("http://127.0.0.1:1/v1")
            .build()
            .unwrap()
            .into_arc()
    }

    fn no_session() -> RunRecord<'static> {
        RunRecord {
            logger: None,
            base: base_record(),
        }
    }

    #[tokio::test]
    async fn aggregator_build_failure_fails_synthesis_unwrapped() {
        let semaphore = Semaphore::new(1);
        let err = synthesize(
            synthesis(
                Err(eyre::eyre!("cannot build aggregator")),
                Some("run context"),
                &semaphore,
            ),
            no_session(),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(err.chain().count(), 1);
        assert_eq!(err.to_string(), "cannot build aggregator");
    }

    #[tokio::test]
    async fn completion_failure_is_wrapped_only_with_run_context() {
        let semaphore = Semaphore::new(1);
        let wrapped = synthesize(
            synthesis(Ok(unreachable_client()), Some("run context"), &semaphore),
            no_session(),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(wrapped.to_string(), "run context");

        let untouched = synthesize(
            synthesis(Ok(unreachable_client()), None, &semaphore),
            no_session(),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(untouched.chain().count() + 1, wrapped.chain().count());
        assert_ne!(untouched.to_string(), "run context");
    }
}
