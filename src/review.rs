use crate::output::UsageReport;
pub use crate::prompts::TaskMode;
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nitpicker_agent::agent::{
    AgentConfig, AgentDepth, AgentProgress, AgentResult, MAX_CONCURRENT_LLM_CALLS,
    add_spawn_subagent_tool, run_agent,
};
use nitpicker_agent::config::{Config, ReviewerConfig};
use nitpicker_agent::llm::{Completion, FinishReason};
use nitpicker_agent::provider::{build_aggregator_client, build_reviewer_client};
use nitpicker_agent::session::{AggregationRecord, SessionLogger, sanitize_path_component};
use nitpicker_agent::tools::all_tools;
use rig_core::completion::Message;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use tokio::sync::Semaphore;

const MAX_CONCURRENT_REVIEWERS: usize = 8;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

pub struct ReviewOutcome {
    pub report: String,
    pub usage: UsageReport,
    /// At least one reviewer failed; the report is synthesized from the survivors.
    /// Surfaced as exit code 3 in the default-review/`ask` CLI arms.
    pub degraded: bool,
}

pub async fn run_review(
    repo: &Path,
    user_prompt: &str,
    config: &Config,
    max_turns: usize,
    verbose: bool,
    mode: TaskMode,
) -> Result<ReviewOutcome> {
    let mut tools = all_tools();
    add_spawn_subagent_tool(&mut tools);
    let session_logger = SessionLogger::maybe_new(config.log_trajectories())?;
    if let Some(logger) = &session_logger {
        info!(path = %logger.root().display(), "trajectory logging enabled");
    }
    let context = crate::context::build_context(repo).await;
    let system_prompt = mode.system_prompt();
    let initial_message = mode.initial_message(user_prompt);
    let mut handles = Vec::new();
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REVIEWERS));
    // shared across every reviewer + their subagents to cap account-wide in-flight LLM calls
    let llm_semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_LLM_CALLS));

    let mp = Arc::new(MultiProgress::new());
    if verbose {
        mp.set_draw_target(ProgressDrawTarget::hidden());
    }
    let _progress_guard = (!verbose && crate::progress::stderr_is_terminal())
        .then(|| crate::progress::set_active_progress(&mp));
    let spinner_style = ProgressStyle::with_template("{spinner:.cyan} {prefix:<12} {msg}")
        .unwrap()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", ""]);
    let done_style = ProgressStyle::with_template("  {prefix:<12} {msg}").unwrap();

    // the proxy handle stays bound for the whole function so its local server outlives the
    // reviewers; only its base URL is threaded downstream (see build_reviewer_client).
    let gemini_proxy = crate::proxy::GeminiProxy::maybe_start(config).await?;
    let proxy_url = gemini_proxy.url();

    for (index, reviewer) in config.reviewer.iter().enumerate() {
        let tools_map = tools.clone();
        let repo = repo.to_path_buf();
        let name = reviewer.name.clone();
        let subagent_counter = Arc::new(AtomicUsize::new(0));
        let session_agent = reviewer_session_agent(index, &name);
        let session_writer = session_logger
            .as_ref()
            .map(|logger| logger.child(format!("{session_agent}.jsonl")));
        let agent_config = build_agent_config(
            config,
            reviewer,
            session_agent,
            &system_prompt,
            max_turns,
            proxy_url.as_deref(),
            Arc::clone(&subagent_counter),
            Arc::clone(&llm_semaphore),
            session_writer,
        );
        info!(reviewer = %name, "spawning agent");

        let pb = mp.add(ProgressBar::new_spinner());
        pb.set_style(spinner_style.clone());
        pb.set_prefix(name.clone());
        pb.set_message(crate::progress::bar_message("reviewing…"));
        pb.enable_steady_tick(Duration::from_millis(80));

        let sub_pb = mp.insert_after(&pb, ProgressBar::new_spinner());
        sub_pb.set_style(ProgressStyle::with_template("{msg}").unwrap());

        let done = done_style.clone();
        let initial_message = initial_message.clone();
        let context = context.clone();
        let sem = Arc::clone(&semaphore);
        let handle: JoinHandle<Result<AgentResult>> = tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");
            let mut config = match agent_config {
                Ok(config) => config,
                Err(err) => {
                    pb.set_style(done.clone());
                    pb.finish_with_message(crate::progress::bar_message(format!("✗ error: {err}")));
                    sub_pb.finish_and_clear();
                    return Err(err);
                }
            };
            config.project_context = Some(context);
            if !verbose {
                let progress_pb = pb.clone();
                let progress_sub_pb = sub_pb.clone();
                config.progress = Some(Arc::new(move |progress: AgentProgress| {
                    progress_pb.set_message(crate::progress::bar_message(format!(
                        "reviewing… ({} turns, {} tool calls, {} subagents)",
                        progress.turns, progress.tool_calls, progress.subagents_spawned
                    )));
                    progress_sub_pb.set_message(crate::progress::detail_message(
                        "    ↳ ",
                        progress.last_subagent.as_deref(),
                    ));
                }));
            }
            let start = Instant::now();
            let result = run_agent(config, &initial_message, &tools_map, &repo).await;
            let elapsed = start.elapsed().as_secs();
            sub_pb.finish_and_clear();
            pb.set_style(done);
            match &result {
                Ok(r) => pb.finish_with_message(crate::progress::bar_message(format!(
                    "✓ done ({elapsed}s, {} turns, {} tool calls, {} subagents, {}, {} out)",
                    r.turns,
                    r.tool_calls,
                    r.subagents_spawned,
                    crate::progress::input_with_cache_share(
                        r.usage.input_tokens,
                        r.usage.cached_input_tokens
                    ),
                    crate::progress::compact_tokens(r.usage.output_tokens)
                ))),
                Err(e) => {
                    pb.finish_with_message(crate::progress::bar_message(format!("✗ failed: {e}")))
                }
            }
            result
        });
        handles.push((name, handle));
    }

    let reviewer_count = handles.len();
    let mut usage = UsageReport::default();
    let mut rendered = Vec::new();
    let mut success_count = 0usize;
    for (name, handle) in handles {
        match handle.await {
            Ok(Ok(result)) => {
                usage.add(result.usage, result.subagents_spawned);
                rendered.push(format!("## {name} review\n\n{}", result.text));
                success_count += 1;
                info!(reviewer = %name, "review completed");
            }
            Ok(Err(err)) => {
                rendered.push(format!("## {name} review\n\n*Failed: {err:#}*"));
                warn!(reviewer = %name, error = ?err, "review failed");
            }
            Err(err) => {
                rendered.push(format!(
                    "## {name} review\n\n*Failed (task panicked): {err:#}*"
                ));
                error!(reviewer = %name, error = ?err, "reviewer task panicked");
            }
        }
    }

    // Refuse to synthesize a verdict out of nothing but failures: the aggregator would hallucinate
    // a confident review from error notes, and `pr` would post it. A total failure is an error, not
    // an "ok" report with empty findings.
    if success_count == 0 {
        eyre::bail!("all {reviewer_count} reviewer(s) failed; refusing to synthesize a verdict");
    }

    let combined = rendered.join("\n\n---\n\n");
    let reduce_prompt = mode.reduce_prompt(user_prompt, &combined);

    let pb_agg = mp.add(ProgressBar::new_spinner());
    pb_agg.set_style(spinner_style);
    pb_agg.set_prefix("aggregator");
    pb_agg.set_message(crate::progress::bar_message("synthesizing…"));
    pb_agg.enable_steady_tick(Duration::from_millis(80));

    let agg = &config.aggregator;
    let client = build_aggregator_client(agg, proxy_url.as_deref())?;
    let completion = Completion {
        model: agg.model.clone(),
        prompt: Message::user(reduce_prompt),
        preamble: Some(mode.aggregator_preamble().to_string()),
        history: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        max_tokens: Some(config.aggregator_max_tokens()),
        additional_params: None,
    };
    let response = client.completion(completion).await?;
    usage.add(response.usage, 0);
    pb_agg.set_style(done_style);
    if response.finish_reason == FinishReason::ToolUse {
        pb_agg.finish_with_message(crate::progress::bar_message(
            "✗ failed: unexpected tool call",
        ));
        return Err(eyre::eyre!("aggregator returned tool calls unexpectedly"));
    }
    pb_agg.finish_with_message("✓ done");
    let text = response.text();
    if let Some(logger) = &session_logger {
        logger
            .write_aggregation(&AggregationRecord {
                kind: "aggregation".to_string(),
                model: agg.model.clone(),
                text: text.clone(),
                rounds: None,
                converged: None,
            })
            .await?;
    }
    Ok(ReviewOutcome {
        report: text,
        usage,
        degraded: success_count < reviewer_count,
    })
}

/// Trajectory identity for one reviewer: the file stem and every record's `agent` field. The
/// config index keeps it unique even when reviewer names collide or sanitize to the same stem —
/// `reflect` merges all of a session's files by timestamp, so the label is the only separator.
fn reviewer_session_agent(index: usize, name: &str) -> String {
    format!("reviewer-{}-{}", index + 1, sanitize_path_component(name))
}

// internal single-call-site builder; the args are distinct per-reviewer handles, not worth a struct
#[allow(clippy::too_many_arguments)]
fn build_agent_config(
    config: &Config,
    reviewer: &ReviewerConfig,
    session_agent: String,
    system_prompt: &str,
    max_turns: usize,
    proxy_url: Option<&str>,
    subagent_counter: Arc<AtomicUsize>,
    llm_semaphore: Arc<Semaphore>,
    session_writer: Option<nitpicker_agent::session::SessionWriter>,
) -> Result<AgentConfig> {
    let client = build_reviewer_client(reviewer, proxy_url)?;
    let compact_threshold = config.reviewer_compact_threshold(reviewer);

    Ok(AgentConfig {
        name: reviewer.name.clone(),
        session_agent,
        model: reviewer.model.clone(),
        max_turns,
        max_tokens: reviewer.max_tokens,
        compact_threshold,
        system_prompt: system_prompt.to_string(),
        subagent_system_prompt: None,
        client,
        depth: AgentDepth::TopLevel,
        terminal_tools: Vec::new(),
        empty_response_nudge: None,
        max_empty_responses: 0,
        subagent_counter,
        llm_semaphore,
        progress: None,
        project_context: None,
        session_writer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two unnamed (or same-named) reviewers must still get distinct trajectory identities —
    /// before the index they shared one file stem AND one record label, interleaving
    /// indistinguishably.
    #[test]
    fn colliding_reviewer_names_get_distinct_session_agents() {
        assert_ne!(reviewer_session_agent(0, ""), reviewer_session_agent(1, ""));
        assert_ne!(
            reviewer_session_agent(0, "claude"),
            reviewer_session_agent(1, "claude")
        );
    }
}
