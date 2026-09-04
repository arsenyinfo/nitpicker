use crate::output::{OutputFormat, PresetCoverage, UsageReport};
use crate::prompts::RunTask;
use crate::telemetry::{self, RunMode};
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nitpicker_agent::agent::{
    AgentConfig, AgentDepth, AgentProgress, AgentResult, MAX_CONCURRENT_LLM_CALLS,
    add_spawn_subagent_tool, run_agent,
};
use nitpicker_agent::config::{Config, ReviewerConfig};
use nitpicker_agent::llm::{
    AlloyClient, AlloySlot, Completion, CompletionResponse, FallbackSlot, FinishReason,
    LLMClientDyn, PriorityClient, throttled_completion,
};
use nitpicker_agent::provider::{build_aggregator_client, build_reviewer_client};
use nitpicker_agent::session::{
    AggregationRecord, JobRecord, SessionLogger, VerdictRecord, sanitize_path_component,
};
use nitpicker_agent::telemetry::bounded;
use nitpicker_agent::tools::{all_tools, floor_char_boundary};
use rig_core::completion::Message;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use tokio::sync::Semaphore;

use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tracing::field::Empty;
use tracing::{Instrument, error, info, info_span, warn};

/// The pre-preset whole-agent concurrency bound, kept for the Ask path only.
const LEGACY_ASK_CONCURRENT_AGENTS: usize = 8;

pub struct ReviewOutcome {
    pub report: String,
    pub usage: UsageReport,
    /// At least one review job failed; the report is synthesized from the survivors.
    /// Surfaced as exit code 3 in the default-review/`ask`/`pr` CLI arms.
    pub degraded: bool,
    /// Per-preset attempted/succeeded job counts in resolution order. `None` on the Ask path.
    pub coverage: Option<Vec<PresetCoverage>>,
}

pub(crate) type ReviewerClientPool = Vec<std::result::Result<FallbackSlot, String>>;

pub(crate) fn build_reviewer_pool(
    config: &Config,
    gemini_proxy: &crate::proxy::GeminiProxy,
    proxy_url: Option<&str>,
) -> ReviewerClientPool {
    config
        .reviewer
        .iter()
        .map(|reviewer| {
            if nitpicker_agent::openrouter::is_unresolved_free_route(
                &reviewer.provider,
                &reviewer.model,
            ) {
                return Err("experimental OpenRouter free route could not be resolved".to_string());
            }
            gemini_proxy
                .annotate(build_reviewer_client(reviewer, proxy_url))
                .map(|client| {
                    FallbackSlot::new(client, reviewer.model.clone(), reviewer.max_tokens)
                })
                .map_err(|err| format!("{err:#}"))
        })
        .collect()
}

/// A fallback/alloy agent may run on any healthy reviewer route, while compaction happens before
/// route selection. Use the smallest configured threshold that any usable route needs.
pub(crate) fn reviewer_pool_compact_threshold(
    config: &Config,
    pool: &[std::result::Result<FallbackSlot, String>],
) -> Option<u64> {
    config
        .reviewer
        .iter()
        .zip(pool)
        .filter(|(_, route)| route.is_ok())
        .filter_map(|(reviewer, _)| config.reviewer_compact_threshold(reviewer))
        .min()
}

/// Resolve one logical reviewer to either its own client or an ordered priority ring. The ring
/// starts at that reviewer and wraps through the declaration order, so every role keeps its normal
/// primary while sharing the same simple fallback policy.
pub(crate) fn reviewer_client(
    pool: &ReviewerClientPool,
    primary_index: usize,
    fallback: bool,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    reviewer_client_with_deferred_route(pool, primary_index, None, fallback)
}

/// Give an independently progressing agent its own mutable routing state. Priority clients fork
/// their sticky index while retaining the shared route availability carried by `FallbackSlot`;
/// stateless provider and Alloy clients are safe to share directly.
pub(crate) fn independent_agent_client(template: &Arc<dyn LLMClientDyn>) -> Arc<dyn LLMClientDyn> {
    template
        .fork_for_agent()
        .unwrap_or_else(|| Arc::clone(template))
}

/// Build one side of a debate while preserving model diversity for as long as possible. The
/// counterpart's configured primary is still a valid last resort, but spare reviewers are tried
/// first so one failed side does not immediately collapse both roles onto the same model.
pub(crate) fn debate_reviewer_client(
    pool: &ReviewerClientPool,
    primary_index: usize,
    counterpart_primary_index: usize,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    reviewer_client_with_deferred_route(pool, primary_index, Some(counterpart_primary_index), true)
}

fn reviewer_client_with_deferred_route(
    pool: &ReviewerClientPool,
    primary_index: usize,
    deferred_index: Option<usize>,
    fallback: bool,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    if !fallback {
        return match &pool[primary_index] {
            Ok(slot) => Ok(slot.client()),
            Err(message) => Err(eyre::eyre!(message.clone())),
        };
    }

    let mut slots = Vec::with_capacity(pool.len());
    let mut build_errors = Vec::new();
    for index in reviewer_route_order(pool.len(), primary_index, deferred_index) {
        match &pool[index] {
            Ok(slot) => slots.push(slot.clone()),
            Err(message) => {
                warn!(
                    reviewer_index = index,
                    error = %message,
                    "reviewer route unavailable; trying next configured reviewer"
                );
                build_errors.push(message.as_str());
            }
        }
    }
    if slots.is_empty() {
        eyre::bail!(
            "no reviewer client could be built: {}",
            build_errors.join("; ")
        );
    }
    Ok(Arc::new(PriorityClient::new(slots)?))
}

fn reviewer_route_order(
    reviewer_count: usize,
    primary_index: usize,
    deferred_index: Option<usize>,
) -> Vec<usize> {
    let mut order = (0..reviewer_count)
        .map(|offset| (primary_index + offset) % reviewer_count)
        .collect::<Vec<_>>();
    if let Some(deferred_index) = deferred_index
        && deferred_index != primary_index
        && let Some(position) = order.iter().position(|&index| index == deferred_index)
    {
        order.remove(position);
        order.push(deferred_index);
    }
    order
}

pub(crate) fn alloy_client(
    pool: &ReviewerClientPool,
    fallback: bool,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    let mut slots = Vec::with_capacity(pool.len());
    for route in pool {
        match route {
            Ok(slot) => slots.push(slot.clone()),
            Err(message) if fallback => {
                warn!(error = %message, "alloy route unavailable; continuing with healthy reviewers")
            }
            Err(message) => return Err(eyre::eyre!(message.clone())),
        }
    }
    let client = match fallback {
        true => AlloyClient::new_with_fallback_routes(slots)?,
        false => AlloyClient::new(slots.iter().map(AlloySlot::from).collect())?,
    };
    Ok(Arc::new(client))
}

pub(crate) fn aggregator_client(
    config: &Config,
    gemini_proxy: &crate::proxy::GeminiProxy,
    proxy_url: Option<&str>,
    reviewer_pool: &ReviewerClientPool,
    fallback: bool,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    let primary = if nitpicker_agent::openrouter::is_unresolved_free_route(
        &config.aggregator.provider,
        &config.aggregator.model,
    ) {
        Err(eyre::eyre!(
            "experimental OpenRouter free aggregator could not be resolved"
        ))
    } else {
        gemini_proxy.annotate(build_aggregator_client(&config.aggregator, proxy_url))
    };
    if !fallback {
        return primary;
    }

    let mut slots = Vec::with_capacity(1 + reviewer_pool.len());
    match primary {
        Ok(client) => slots.push(FallbackSlot::new(
            client,
            config.aggregator.model.clone(),
            Some(config.aggregator_max_tokens()),
        )),
        Err(err) => warn!(error = %err, "aggregator route unavailable; trying reviewer pool"),
    }
    for route in reviewer_pool {
        match route {
            Ok(slot) => slots.push(slot.clone()),
            Err(message) => warn!(
                error = %message,
                "reviewer route unavailable for aggregator fallback"
            ),
        }
    }
    if slots.is_empty() {
        eyre::bail!("no aggregator or reviewer fallback client could be built");
    }
    Ok(Arc::new(PriorityClient::new(slots)?))
}

pub(crate) fn validate_synthesis_response(response: &CompletionResponse, role: &str) -> Result<()> {
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

pub(crate) fn synthesis_model(response: &CompletionResponse, configured_model: &str) -> String {
    response
        .selected_model
        .clone()
        .unwrap_or_else(|| configured_model.to_string())
}

/// Mirrors `DebateOptions`: the trailing run-control params bundled so `run_review` stays
/// under clippy's argument limit, symmetric with the debate entry point.
pub struct ReviewOptions<'a> {
    pub max_turns: usize,
    pub verbose: bool,
    pub task: RunTask<'a>,
    pub fallback: bool,
    pub format: OutputFormat,
}

pub async fn run_review(
    repo: &Path,
    user_prompt: &str,
    config: &Config,
    opts: ReviewOptions<'_>,
) -> Result<ReviewOutcome> {
    let span = telemetry::run_span(RunMode::Parallel, &opts.task, config.reviewer.len());
    telemetry::record_run(
        span,
        run_review_inner(repo, user_prompt, config, opts),
        |o| (o.degraded, &o.usage),
    )
    .await
}

async fn run_review_inner(
    repo: &Path,
    user_prompt: &str,
    config: &Config,
    opts: ReviewOptions<'_>,
) -> Result<ReviewOutcome> {
    let ReviewOptions {
        max_turns,
        verbose,
        task,
        fallback,
        format,
    } = opts;
    let session_attribution = crate::prompts::session_attribution();
    let _terminal_title = crate::progress::start_terminal_title(repo, format);
    let presets = task.presets();
    let lanes = task.lanes();
    let mut tools = all_tools();
    add_spawn_subagent_tool(&mut tools);
    let session_logger = SessionLogger::maybe_new(config.log_trajectories())?;
    if let Some(logger) = &session_logger {
        info!(path = %logger.root().display(), "trajectory logging enabled");
        tracing::Span::current().record("nitpicker.session.id", logger.id());
        if let Err(err) = logger.write_attribution(&session_attribution).await {
            warn!(error = ?err, "failed to persist session attribution");
        }
    }
    let context = crate::context::build_context(repo).await;
    let initial_message = task.initial_message(user_prompt);
    let reviewer_names: Vec<&str> = config.reviewer.iter().map(|r| r.name.as_str()).collect();
    let jobs = plan_jobs(&reviewer_names, presets);
    // per-preset prompts are identical across reviewers — compose each once, not per job
    let system_prompts: Vec<String> = lanes.iter().map(|lane| lane.reviewer_system()).collect();
    let subagent_prompts: Vec<Option<String>> =
        lanes.iter().map(|lane| lane.subagent_prompt()).collect();
    let mut handles = Vec::new();
    // Preset jobs all spawn eagerly and run concurrently — the account-wide cap on in-flight
    // LLM calls below is their only concurrency bound, shared with every subagent. The Ask
    // path keeps its legacy whole-agent bound: its behavior (incl. >8-reviewer concurrency
    // semantics) is a compatibility surface.
    let ask_agent_semaphore = match presets {
        None => Some(Arc::new(Semaphore::new(LEGACY_ASK_CONCURRENT_AGENTS))),
        Some(_) => None,
    };
    let llm_semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_LLM_CALLS));

    let mp = Arc::new(MultiProgress::new());
    if verbose {
        mp.set_draw_target(ProgressDrawTarget::hidden());
    }
    let _progress_guard = (!verbose && crate::progress::stderr_is_terminal())
        .then(|| crate::progress::set_active_progress(&mp));
    let prefix_width = jobs
        .iter()
        .map(|job| job.label.chars().count())
        .chain([12])
        .max()
        .expect("non-empty iterator")
        .min(32);
    let spinner_style = ProgressStyle::with_template(&format!(
        "{{spinner:.cyan}} {{prefix:<{prefix_width}}} {{msg}}"
    ))
    .unwrap()
    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", ""]);
    let done_style =
        ProgressStyle::with_template(&format!("  {{prefix:<{prefix_width}}} {{msg}}")).unwrap();

    // the proxy handle stays bound for the whole function so its local server outlives the
    // reviewers; only its base URL is threaded downstream (see build_reviewer_client).
    let gemini_proxy = crate::proxy::GeminiProxy::maybe_start(config).await;
    let proxy_url = gemini_proxy.url();

    // Build one pristine routing template per reviewer. Every job forks its template before use,
    // so sticky failover remains agent-local while cloned FallbackSlots retain run-wide route
    // availability. eyre::Report is not Clone, so a build failure is kept as its rendered message
    // and re-raised per job — one broken reviewer fails its own jobs while the others proceed.
    // A dead Gemini proxy degrades only its own jobs normally; fallback mode skips that route and
    // continues through the configured reviewer order, with the startup cause still logged.
    let reviewer_clients = build_reviewer_pool(config, &gemini_proxy, proxy_url.as_deref());
    let fallback_compact_threshold = reviewer_pool_compact_threshold(config, &reviewer_clients);
    let reviewer_client_templates = (0..config.reviewer.len())
        .map(|reviewer_index| {
            reviewer_client(&reviewer_clients, reviewer_index, fallback)
                .map_err(|err| format!("{err:#}"))
        })
        .collect::<Vec<_>>();

    for job in &jobs {
        let reviewer = &config.reviewer[job.reviewer_index];
        let compact_threshold = match fallback {
            true => fallback_compact_threshold,
            false => config.reviewer_compact_threshold(reviewer),
        };
        let prompt_index = match (presets, job.preset_index) {
            (Some(_), Some(j)) => j,
            (None, None) => 0,
            (Some(_), None) | (None, Some(_)) => {
                unreachable!("job planning matches the run's preset mode")
            }
        };
        let tools_map = tools.clone();
        let repo = repo.to_path_buf();
        let label = job.label.clone();
        let subagent_counter = Arc::new(AtomicUsize::new(0));
        let session_writer = session_logger
            .as_ref()
            .map(|logger| logger.child(format!("{}.jsonl", job.session_agent)));
        let agent_config = match &reviewer_client_templates[job.reviewer_index] {
            Ok(template) => Ok(build_agent_config(
                reviewer,
                independent_agent_client(template),
                compact_threshold,
                job.session_agent.clone(),
                system_prompts[prompt_index].clone(),
                subagent_prompts[prompt_index].clone(),
                max_turns,
                Arc::clone(&subagent_counter),
                Arc::clone(&llm_semaphore),
                session_writer,
            )),
            // re-raise the message verbatim: the Ask path folds this into its aggregator
            // input, whose bytes must not drift from the pre-fan-out rendering
            Err(message) => Err(eyre::eyre!(message.clone())),
        };
        info!(job = %label, "spawning agent");

        let pb = mp.add(ProgressBar::new_spinner());
        pb.set_style(spinner_style.clone());
        pb.set_prefix(label.clone());
        pb.set_message(crate::progress::bar_message("reviewing…"));
        pb.enable_steady_tick(Duration::from_millis(80));

        let sub_pb = mp.insert_after(&pb, ProgressBar::new_spinner());
        sub_pb.set_style(ProgressStyle::with_template("{msg}").unwrap());

        let done = done_style.clone();
        let initial_message = initial_message.clone();
        let context = context.clone();
        let agent_sem = ask_agent_semaphore.clone();
        let preset_name = match (presets, job.preset_index) {
            (Some(presets), Some(j)) => bounded(&presets[j].name),
            _ => "",
        };
        // created outside the spawn so it parents under `review`: spawned tasks don't inherit
        let job_span = info_span!(
            "review.job",
            otel.status_code = Empty,
            nitpicker.job = %bounded(&label),
            nitpicker.reviewer = %bounded(&reviewer.name),
            nitpicker.preset = preset_name,
            gen_ai.request.model = %bounded(&reviewer.model),
        );
        let handle: JoinHandle<Result<AgentResult>> = tokio::spawn(
            async move {
                let _agent_permit = match &agent_sem {
                    Some(sem) => Some(sem.acquire().await.expect("semaphore closed")),
                    None => None,
                };
                let mut config = match agent_config {
                    Ok(config) => config,
                    Err(err) => {
                        pb.set_style(done.clone());
                        pb.finish_with_message(crate::progress::bar_message(format!(
                            "✗ error: {err}"
                        )));
                        sub_pb.finish_and_clear();
                        tracing::Span::current().record("otel.status_code", "ERROR");
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
                    Err(e) => pb.finish_with_message(crate::progress::bar_message(format!(
                        "✗ failed: {e}"
                    ))),
                }
                if result.is_err() {
                    tracing::Span::current().record("otel.status_code", "ERROR");
                }
                result
            }
            .instrument(job_span),
        );
        handles.push((label, job.preset_index, handle));
    }

    let job_count = handles.len();
    let preset_run = presets.is_some();
    let mut usage = UsageReport::default();
    let mut rendered = Vec::new();
    let mut success_count = 0usize;
    let mut surviving_preset_indices = std::collections::HashSet::new();
    let mut job_records: Vec<JobRecord> = Vec::new();
    let mut verdict_records: Vec<VerdictRecord> = Vec::new();
    for (label, preset_index, handle) in handles {
        let preset_name = match (presets, preset_index) {
            (Some(ps), Some(j)) => Some(ps[j].name.clone()),
            _ => None,
        };
        let ok = match handle.await {
            Ok(Ok(result)) => {
                usage.add(result.usage, result.subagents_spawned);
                rendered.extend(rendered_section(&label, Ok(&result.text), preset_run));
                verdict_records.push(VerdictRecord {
                    lens: preset_name.clone(),
                    stage: label.clone(),
                    text: result.text,
                    ok: true,
                });
                success_count += 1;
                if let Some(j) = preset_index {
                    surviving_preset_indices.insert(j);
                }
                info!(job = %label, "review completed");
                true
            }
            Ok(Err(err)) => {
                let stub = format!("*Failed: {err:#}*");
                rendered.extend(rendered_section(&label, Err(&stub), preset_run));
                verdict_records.push(VerdictRecord {
                    lens: preset_name.clone(),
                    stage: label.clone(),
                    text: stub,
                    ok: false,
                });
                warn!(job = %label, error = ?err, "review failed");
                false
            }
            Err(err) => {
                let stub = format!("*Failed (task panicked): {err:#}*");
                rendered.extend(rendered_section(&label, Err(&stub), preset_run));
                verdict_records.push(VerdictRecord {
                    lens: preset_name.clone(),
                    stage: label.clone(),
                    text: stub,
                    ok: false,
                });
                error!(job = %label, error = ?err, "review task panicked");
                false
            }
        };
        job_records.push(JobRecord {
            label,
            preset: preset_name,
            ok,
        });
    }

    let coverage = presets.map(|ps| preset_coverage(ps, &job_records));

    // Refuse to synthesize a verdict out of nothing but failures: the aggregator would hallucinate
    // a confident review from error notes, and `pr` would post it. A total failure is an error, not
    // an "ok" report with empty findings. The job outcomes are persisted first — a run where
    // everything failed is the one that most needs a durable record, and client-build failures
    // leave no trajectory file at all.
    if success_count == 0 {
        let err =
            eyre::eyre!("all {job_count} review job(s) failed; refusing to synthesize a verdict");
        if let Some(logger) = &session_logger {
            let record = AggregationRecord {
                kind: "aggregation".to_string(),
                model: config.aggregator.model.clone(),
                text: String::new(),
                error: Some(bounded_error_string(&err)),
                rounds: None,
                converged: None,
                presets: presets.map(|ps| ps.iter().map(|p| p.name.clone()).collect()),
                lanes: None,
                verdicts: verdict_records,
                jobs: Some(job_records),
            };
            match logger.write_aggregation(&record).await {
                Ok(()) => {}
                Err(write_err) => {
                    warn!(error = ?write_err, "failed to persist all-jobs-failed record");
                }
            }
        }
        return Err(err);
    }

    let combined = rendered.join("\n\n---\n\n");
    // The synthesis roster covers only presets with at least one surviving job — a rubric
    // with no matching report would read as an angle that was reviewed and found clean.
    // (The session record and `pr --json` keep the FULL resolved list: they document the
    // run's resolution, not its coverage.)
    let surviving_presets: Vec<crate::presets::ReviewPreset> = presets
        .map(|ps| {
            ps.iter()
                .enumerate()
                .filter(|(j, _)| surviving_preset_indices.contains(j))
                .map(|(_, p)| p.clone())
                .collect()
        })
        .unwrap_or_default();
    let reduce_prompt = match task {
        RunTask::Ask => crate::prompts::ask_reduce_prompt(user_prompt, &combined),
        RunTask::Review { .. } => {
            crate::prompts::review_reduce_prompt(user_prompt, &combined, &surviving_presets)
        }
    };

    let pb_agg = mp.add(ProgressBar::new_spinner());
    pb_agg.set_style(spinner_style);
    pb_agg.set_prefix("aggregator");
    pb_agg.set_message(crate::progress::bar_message("synthesizing…"));
    pb_agg.enable_steady_tick(Duration::from_millis(80));

    let agg = &config.aggregator;
    let synthesis_span = telemetry::synthesis_span(&agg.model);
    let synthesis: Result<(String, String)> = async {
        let client = aggregator_client(
            config,
            &gemini_proxy,
            proxy_url.as_deref(),
            &reviewer_clients,
            fallback,
        )?;
        let completion = Completion {
            model: agg.model.clone(),
            prompt: Message::user(reduce_prompt),
            preamble: Some(task.aggregator_preamble()),
            history: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: Some(config.aggregator_max_tokens()),
            additional_params: None,
        };
        // preset runs get the count/context wrapping; the Ask path propagates the provider
        // error untouched, as it did before the fan-out
        // every job has finished, so the permit is free; routing through the chokepoint keeps
        // "every completion is a chat span" true for the synthesis too
        let response = throttled_completion(&llm_semaphore, &client, completion)
            .await
            .and_then(|response| {
                validate_synthesis_response(&response, "aggregator")?;
                Ok(response)
            })
            .map_err(|err| match presets {
                Some(presets) => crate::presets::synthesis_failure(
                    err,
                    format!(
                        "final aggregation failed over {success_count} surviving review job(s) across {} preset(s)",
                        presets.len()
                    ),
                ),
                None => err,
            })?;
        usage.add(response.usage, 0);
        let model = synthesis_model(&response, &agg.model);
        Ok((response.text(), model))
    }
    .instrument(synthesis_span.clone())
    .await;
    if synthesis.is_err() {
        synthesis_span.record("otel.status_code", "ERROR");
    }
    pb_agg.set_style(done_style);
    // A post-collection synthesis failure still persists the per-job outcomes: the jobs
    // list is the durable record of what ran, and losing it because the aggregator died
    // is exactly when a post-mortem needs it. The record carries `error` and an empty
    // `text` so consumers (reflect) don't render it as a verdict.
    let (text, aggregation_model) = match synthesis {
        Ok(result) => {
            pb_agg.finish_with_message("✓ done");
            result
        }
        Err(err) => {
            pb_agg.finish_with_message(crate::progress::bar_message("✗ synthesis failed"));
            if let Some(logger) = &session_logger {
                let record = AggregationRecord {
                    kind: "aggregation".to_string(),
                    model: agg.model.clone(),
                    text: String::new(),
                    error: Some(bounded_error_string(&err)),
                    rounds: None,
                    converged: None,
                    presets: presets.map(|ps| ps.iter().map(|p| p.name.clone()).collect()),
                    lanes: None,
                    verdicts: verdict_records,
                    jobs: Some(job_records),
                };
                match logger.write_aggregation(&record).await {
                    Ok(()) => {}
                    Err(write_err) => {
                        warn!(error = ?write_err, "failed to persist synthesis-failure record");
                    }
                }
            }
            return Err(err);
        }
    };
    if let Some(logger) = &session_logger {
        logger
            .write_aggregation(&AggregationRecord {
                kind: "aggregation".to_string(),
                model: aggregation_model,
                text: text.clone(),
                error: None,
                rounds: None,
                converged: None,
                presets: presets.map(|ps| ps.iter().map(|p| p.name.clone()).collect()),
                lanes: None,
                verdicts: verdict_records,
                jobs: Some(job_records),
            })
            .await?;
    }
    Ok(ReviewOutcome {
        report: text,
        usage,
        degraded: success_count < job_count,
        coverage,
    })
}

/// One spawned review job. `reviewer_index` picks the client/config slot; `preset_index`
/// (preset runs only) picks the rubric. `label` heads the rendered report section and the
/// progress bar; `session_agent` names the trajectory file and every record in it.
struct ReviewJob {
    reviewer_index: usize,
    preset_index: Option<usize>,
    label: String,
    session_agent: String,
}

/// Plan the job matrix: one job per reviewer for `ask` (`presets: None`), reviewers ×
/// presets for review runs. Labels must be unique — the aggregator attributes reports by
/// them: duplicate reviewer names get an ` #<index>` suffix, and any residual collision
/// (a crafted name that embeds the suffix) falls back to the globally-unique job ordinal.
/// Per-preset attempted/succeeded counts in resolution order, from the drained job list.
fn preset_coverage(
    presets: &[crate::presets::ReviewPreset],
    job_records: &[JobRecord],
) -> Vec<PresetCoverage> {
    presets
        .iter()
        .map(|p| {
            let (attempted, succeeded) = job_records
                .iter()
                .filter(|j| j.preset.as_deref() == Some(p.name.as_str()))
                .fold((0, 0), |(a, s), j| (a + 1, s + usize::from(j.ok)));
            PresetCoverage {
                preset: p.name.clone(),
                attempted,
                succeeded,
            }
        })
        .collect()
}

/// Bounded `{err:#}` chain for persistence — a provider error body can be huge, and the
/// session record is not the place to store it whole.
pub(crate) fn bounded_error_string(err: &eyre::Report) -> String {
    const MAX: usize = 4000;
    let full = format!("{err:#}");
    match full.len() <= MAX {
        true => full,
        false => format!("{}…", &full[..floor_char_boundary(&full, MAX)]),
    }
}

fn plan_jobs(
    reviewer_names: &[&str],
    presets: Option<&[crate::presets::ReviewPreset]>,
) -> Vec<ReviewJob> {
    let duplicated: Vec<bool> = reviewer_names
        .iter()
        .map(|name| reviewer_names.iter().filter(|other| other == &name).count() > 1)
        .collect();
    let mut jobs = Vec::new();
    match presets {
        None => {
            for (i, name) in reviewer_names.iter().enumerate() {
                jobs.push(ReviewJob {
                    reviewer_index: i,
                    preset_index: None,
                    label: name.to_string(),
                    session_agent: reviewer_session_agent(i, name),
                });
            }
        }
        Some(presets) => {
            for (j, preset) in presets.iter().enumerate() {
                for (i, name) in reviewer_names.iter().enumerate() {
                    let label = match duplicated[i] {
                        true => format!("{} · {} #{}", preset.name, name, i + 1),
                        false => format!("{} · {}", preset.name, name),
                    };
                    jobs.push(ReviewJob {
                        reviewer_index: i,
                        preset_index: Some(j),
                        label,
                        session_agent: preset_session_agent(i, name, j, &preset.name),
                    });
                }
            }
            // collision repair is preset-run-only: the Ask path must keep its legacy labels
            // byte-for-byte, duplicate reviewer names included (session agents stay unique)
            let mut seen = std::collections::HashSet::new();
            for (ordinal, job) in jobs.iter_mut().enumerate() {
                let mut label = job.label.clone();
                while !seen.insert(label.clone()) {
                    label = format!("{label} ({})", ordinal + 1);
                }
                job.label = label;
            }
        }
    }
    jobs
}

/// One finished job's contribution to the synthesis input. Successes always render;
/// failure stubs render only outside preset runs — for Review fan-out an error note is
/// execution noise, not review evidence (degraded accounting and the logs carry it), while
/// the Ask path keeps its pre-fan-out stubs byte-for-byte.
fn rendered_section(
    label: &str,
    outcome: std::result::Result<&str, &str>,
    preset_run: bool,
) -> Option<String> {
    let header = match preset_run {
        true => format!("## {label}"),
        false => format!("## {label} review"),
    };
    match (outcome, preset_run) {
        (Ok(text), _) => Some(format!("{header}\n\n{text}")),
        (Err(stub), false) => Some(format!("{header}\n\n{stub}")),
        (Err(_), true) => None,
    }
}

/// Trajectory identity for one reviewer: the file stem and every record's `agent` field. The
/// config index keeps it unique even when reviewer names collide or sanitize to the same stem —
/// `reflect` merges all of a session's files by timestamp, so the label is the only separator.
fn reviewer_session_agent(index: usize, name: &str) -> String {
    format!("reviewer-{}-{}", index + 1, sanitize_path_component(name))
}

/// Preset-run variant: both indices are load-bearing for uniqueness, since reviewer names
/// AND preset names can each sanitize (or truncate) to identical stems.
fn preset_session_agent(
    reviewer_index: usize,
    name: &str,
    preset_index: usize,
    preset_name: &str,
) -> String {
    format!(
        "reviewer-{}-{}-{}-{}",
        reviewer_index + 1,
        sanitize_path_component(name),
        preset_index + 1,
        crate::presets::path_slug(preset_name)
    )
}

// internal single-call-site builder; the args are distinct per-job handles, not worth a struct
#[allow(clippy::too_many_arguments)]
fn build_agent_config(
    reviewer: &ReviewerConfig,
    client: Arc<dyn nitpicker_agent::llm::LLMClientDyn>,
    compact_threshold: Option<u64>,
    session_agent: String,
    system_prompt: String,
    subagent_system_prompt: Option<String>,
    max_turns: usize,
    subagent_counter: Arc<AtomicUsize>,
    llm_semaphore: Arc<Semaphore>,
    session_writer: Option<nitpicker_agent::session::SessionWriter>,
) -> AgentConfig {
    AgentConfig {
        name: reviewer.name.clone(),
        session_agent,
        model: reviewer.model.clone(),
        max_turns,
        max_tokens: reviewer.max_tokens,
        compact_threshold,
        system_prompt,
        subagent_system_prompt,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::ReviewPreset;
    use nitpicker_agent::llm::{LLMClient, TokenUsage};
    use rig_core::OneOrMany;
    use rig_core::completion::AssistantContent;

    struct UnusedClient;

    impl LLMClient for UnusedClient {
        async fn completion(&self, _completion: Completion) -> Result<CompletionResponse> {
            unreachable!("threshold tests never call the model")
        }
    }

    fn usable_slot(model: &str) -> std::result::Result<FallbackSlot, String> {
        Ok(FallbackSlot::new(Arc::new(UnusedClient), model, None))
    }

    fn synthesis_response(finish_reason: FinishReason) -> CompletionResponse {
        CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text("partial verdict")),
            finish_reason,
            usage: TokenUsage::default(),
            selected_model: Some("fallback-model".to_string()),
        }
    }

    #[test]
    fn fallback_compaction_uses_smallest_usable_route_threshold() {
        let config: Config = toml::from_str(
            r#"
                [aggregator]
                provider = "openai"

                [[reviewer]]
                model = "large"
                provider = "openai"
                compact_threshold = 120000

                [[reviewer]]
                model = "small"
                provider = "openai"
                compact_threshold = 24000

                [[reviewer]]
                model = "unset"
                provider = "openai"
            "#,
        )
        .unwrap();

        let all_usable = vec![
            usable_slot("large"),
            usable_slot("small"),
            usable_slot("unset"),
        ];
        assert_eq!(
            reviewer_pool_compact_threshold(&config, &all_usable),
            Some(24_000)
        );

        let small_failed = vec![
            usable_slot("large"),
            Err("small failed to build".to_string()),
            usable_slot("unset"),
        ];
        assert_eq!(
            reviewer_pool_compact_threshold(&config, &small_failed),
            Some(120_000)
        );

        let only_unset_usable = vec![
            Err("large failed to build".to_string()),
            Err("small failed to build".to_string()),
            usable_slot("unset"),
        ];
        assert_eq!(
            reviewer_pool_compact_threshold(&config, &only_unset_usable),
            None
        );

        let defaulted: Config = toml::from_str(
            r#"
                [defaults]
                compact_threshold = 50000

                [aggregator]
                provider = "openai"

                [[reviewer]]
                model = "defaulted"
                provider = "openai"
            "#,
        )
        .unwrap();
        assert_eq!(
            reviewer_pool_compact_threshold(&defaulted, &[usable_slot("defaulted")]),
            Some(50_000)
        );
    }

    #[test]
    fn debate_fallback_defers_the_counterpart_primary() {
        assert_eq!(reviewer_route_order(4, 0, Some(1)), vec![0, 2, 3, 1]);
        assert_eq!(reviewer_route_order(4, 1, Some(0)), vec![1, 2, 3, 0]);

        // With no spare route, using the counterpart is still preferable to aborting the run.
        assert_eq!(reviewer_route_order(2, 0, Some(1)), vec![0, 1]);
        assert_eq!(reviewer_route_order(2, 1, Some(0)), vec![1, 0]);
    }

    #[test]
    fn ordinary_review_fallback_keeps_declaration_order() {
        assert_eq!(reviewer_route_order(4, 0, None), vec![0, 1, 2, 3]);
        assert_eq!(reviewer_route_order(4, 2, None), vec![2, 3, 0, 1]);
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

    fn presets(names: &[&str]) -> Vec<ReviewPreset> {
        names
            .iter()
            .map(|n| ReviewPreset {
                name: n.to_string(),
                prompt: format!("rubric for {n}"),
            })
            .collect()
    }

    fn all_unique(values: impl IntoIterator<Item = String>) -> bool {
        let mut seen = std::collections::HashSet::new();
        values.into_iter().all(|v| seen.insert(v))
    }

    /// The preset fan-out is reviewers × presets, ordered preset-major, and the Ask path
    /// (no presets) stays one job per reviewer.
    #[test]
    fn job_count_is_reviewers_times_presets() {
        let ps = presets(&["security", "tone"]);
        let jobs = plan_jobs(&["claude", "gpt", "gemini"], Some(&ps));
        assert_eq!(jobs.len(), 6);
        assert_eq!(jobs[0].label, "security · claude");
        assert_eq!(jobs[5].label, "tone · gemini");

        let ask_jobs = plan_jobs(&["claude", "gpt"], None);
        assert_eq!(ask_jobs.len(), 2);
        assert_eq!(ask_jobs[0].label, "claude");
    }

    /// Duplicate reviewer names must stay distinguishable in report headings and progress —
    /// the aggregator attributes reports by label alone.
    #[test]
    fn duplicate_reviewer_names_get_indexed_labels() {
        let ps = presets(&["security"]);
        let jobs = plan_jobs(&["claude", "claude"], Some(&ps));
        assert_eq!(jobs[0].label, "security · claude #1");
        assert_eq!(jobs[1].label, "security · claude #2");
        assert!(all_unique(jobs.iter().map(|j| j.label.clone())));
    }

    /// A crafted reviewer name that embeds the dedup suffix cannot force two jobs to share
    /// a label — the job-ordinal fallback keeps the full set unique.
    #[test]
    fn crafted_names_cannot_collide_labels() {
        let ps = presets(&["security"]);
        let jobs = plan_jobs(&["claude", "claude", "claude #2"], Some(&ps));
        assert!(all_unique(jobs.iter().map(|j| j.label.clone())));
    }

    /// Preset and reviewer names that sanitize identically must still produce distinct
    /// trajectory identities (both indices are baked into the stem).
    #[test]
    fn preset_session_agents_stay_unique_under_sanitization_collisions() {
        let ps = presets(&["a/b", "a?b"]);
        let jobs = plan_jobs(&["r!", "r?"], Some(&ps));
        assert!(all_unique(jobs.iter().map(|j| j.session_agent.clone())));
    }

    /// The Ask path is a byte-compatibility surface: duplicate reviewer names keep their
    /// legacy identical labels there (collision repair is preset-run-only).
    #[test]
    fn ask_jobs_keep_legacy_labels_even_when_names_collide() {
        let jobs = plan_jobs(&["claude", "claude"], None);
        assert_eq!(jobs[0].label, "claude");
        assert_eq!(jobs[1].label, "claude");
    }

    /// A preset reviewed by 1 of its N planned jobs must be distinguishable from a fully
    /// covered one — the envelope's `coverage` key carries exactly these counts, including
    /// zero-survivor presets that a names-only summary would drop entirely.
    #[test]
    fn preset_coverage_counts_attempted_and_succeeded_per_preset() {
        let presets = [
            crate::presets::ReviewPreset {
                name: "security".to_string(),
                prompt: "r".to_string(),
            },
            crate::presets::ReviewPreset {
                name: "tone".to_string(),
                prompt: "r".to_string(),
            },
        ];
        let job = |preset: &str, ok: bool| JobRecord {
            label: format!("{preset} · claude"),
            preset: Some(preset.to_string()),
            ok,
        };
        let records = [
            job("security", true),
            job("security", false),
            job("tone", false),
        ];
        assert_eq!(
            preset_coverage(&presets, &records),
            vec![
                PresetCoverage {
                    preset: "security".to_string(),
                    attempted: 2,
                    succeeded: 1,
                },
                PresetCoverage {
                    preset: "tone".to_string(),
                    attempted: 1,
                    succeeded: 0,
                },
            ]
        );
    }

    /// Failure stubs reach the synthesis input only on the Ask path; preset runs drop them
    /// (execution noise, not review evidence) while successes always render.
    #[test]
    fn failure_stubs_are_review_gated_in_the_synthesis_input() {
        assert_eq!(
            rendered_section("security · claude", Ok("finding"), true),
            Some("## security · claude\n\nfinding".to_string())
        );
        assert_eq!(
            rendered_section("security · claude", Err("*Failed: x*"), true),
            None
        );
        assert_eq!(
            rendered_section("claude", Err("*Failed: x*"), false),
            Some("## claude review\n\n*Failed: x*".to_string())
        );
    }
}
