use crate::output::{OutputFormat, PresetCoverage, UsageReport};
use crate::prompts::RunTask;
use crate::routing::{
    aggregator_client, build_reviewer_pool, independent_agent_client, reviewer_client,
    reviewer_pool_compact_threshold,
};
use crate::telemetry::{self, RunMode};
use eyre::Result;
use indicatif::{ProgressBar, ProgressStyle};
use nitpicker_agent::agent::{
    AgentConfig, AgentDepth, AgentProgress, AgentResult, MAX_CONCURRENT_LLM_CALLS,
    add_spawn_subagent_tool, run_agent,
};
use nitpicker_agent::config::{Config, ReviewerConfig};
use nitpicker_agent::llm::Completion;
use nitpicker_agent::session::{
    AggregationRecord, JobRecord, VerdictRecord, sanitize_path_component,
};
use nitpicker_agent::telemetry::bounded;
use nitpicker_agent::tools::all_tools;
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
    let _terminal_title = crate::progress::start_terminal_title(repo, format);
    let presets = task.presets();
    let lanes = task.lanes();
    let mut tools = all_tools();
    add_spawn_subagent_tool(&mut tools);
    let session_logger = crate::run::open_session(config).await?;
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

    let (mp, _progress_guard) = crate::run::progress_display(verbose);
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
    let CollectedJobs {
        mut usage,
        rendered,
        job_records,
        verdict_records,
    } = collect_jobs(handles, presets).await;
    let success_count = job_records.iter().filter(|job| job.ok).count();
    let coverage = presets.map(|ps| preset_coverage(ps, &job_records));

    let record = crate::run::RunRecord {
        logger: session_logger.as_ref(),
        base: AggregationRecord {
            kind: "aggregation".to_string(),
            model: config.aggregator.model.clone(),
            text: String::new(),
            error: None,
            rounds: None,
            converged: None,
            presets: presets.map(|ps| ps.iter().map(|p| p.name.clone()).collect()),
            lanes: None,
            verdicts: verdict_records,
            jobs: Some(job_records),
        },
    };

    // Refuse to synthesize a verdict out of nothing but failures: the aggregator would hallucinate
    // a confident review from error notes, and `pr` would post it. A total failure is an error, not
    // an "ok" report with empty findings. The job outcomes are persisted first — a run where
    // everything failed is the one that most needs a durable record, and client-build failures
    // leave no trajectory file at all.
    if success_count == 0 {
        let err =
            eyre::eyre!("all {job_count} review job(s) failed; refusing to synthesize a verdict");
        record.persist_failure(&err, "all-jobs-failed").await;
        return Err(err);
    }

    // The synthesis roster covers only presets with at least one surviving job — a rubric
    // with no matching report would read as an angle that was reviewed and found clean.
    // (The session record and `pr --json` keep the FULL resolved list: they document the
    // run's resolution, not its coverage.)
    let surviving_presets = surviving_presets(presets, coverage.as_deref());
    let reduce_prompt = reduce_prompt(
        task,
        user_prompt,
        &rendered,
        &surviving_presets,
        coverage.as_deref(),
    );

    let pb_agg = mp.add(ProgressBar::new_spinner());
    pb_agg.set_style(spinner_style);
    pb_agg.set_prefix("aggregator");
    pb_agg.set_message(crate::progress::bar_message("synthesizing…"));
    pb_agg.enable_steady_tick(Duration::from_millis(80));

    let synthesis = crate::run::synthesize(
        crate::run::Synthesis {
            // built only now, after every job: a build failure is reported (and recorded) as the
            // synthesis failure, unwrapped
            client: aggregator_client(
                config,
                &gemini_proxy,
                proxy_url.as_deref(),
                &reviewer_clients,
                fallback,
            ),
            completion: Completion {
                model: config.aggregator.model.clone(),
                prompt: Message::user(reduce_prompt),
                preamble: Some(task.aggregator_preamble()),
                history: Vec::new(),
                tools: Vec::new(),
                tool_choice: None,
                max_tokens: Some(config.aggregator_max_tokens()),
                additional_params: None,
            },
            role: "aggregator",
            failure_context: presets.map(|presets| {
                format!(
                    "final aggregation failed over {success_count} surviving review job(s) across {} preset(s)",
                    presets.len()
                )
            }),
            progress: pb_agg,
            done_style,
            llm_semaphore: &llm_semaphore,
        },
        record,
    )
    .await?;
    usage.add(synthesis.usage, 0);
    let text = synthesis.text;
    let report = match &coverage {
        Some(coverage) => crate::output::with_partial_review_note(
            text,
            job_count - success_count,
            job_count,
            "review jobs",
            coverage,
        ),
        None => text,
    };
    Ok(ReviewOutcome {
        report,
        usage,
        degraded: success_count < job_count,
        coverage,
    })
}

fn reduce_prompt(
    task: RunTask<'_>,
    user_prompt: &str,
    rendered: &[String],
    surviving_presets: &[crate::presets::ReviewPreset],
    coverage: Option<&[PresetCoverage]>,
) -> String {
    let combined = rendered.join("\n\n---\n\n");
    match task {
        RunTask::Ask => crate::prompts::ask_reduce_prompt(user_prompt, &combined),
        RunTask::Review { .. } => crate::prompts::review_reduce_prompt(
            user_prompt,
            &combined,
            surviving_presets,
            coverage.unwrap_or_default(),
        ),
    }
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

type JobHandle = (String, Option<usize>, JoinHandle<Result<AgentResult>>);

/// Every job's contribution to the synthesis input and the session record, in spawn order.
struct CollectedJobs {
    usage: UsageReport,
    rendered: Vec<String>,
    job_records: Vec<JobRecord>,
    verdict_records: Vec<VerdictRecord>,
}

async fn collect_jobs(
    handles: Vec<JobHandle>,
    presets: Option<&[crate::presets::ReviewPreset]>,
) -> CollectedJobs {
    let mut collected = CollectedJobs {
        usage: UsageReport::default(),
        rendered: Vec::new(),
        job_records: Vec::new(),
        verdict_records: Vec::new(),
    };
    for (label, preset_index, handle) in handles {
        let preset_name = match (presets, preset_index) {
            (Some(ps), Some(j)) => Some(ps[j].name.clone()),
            _ => None,
        };
        let outcome = match handle.await {
            Ok(Ok(result)) => {
                collected.usage.add(result.usage, result.subagents_spawned);
                info!(job = %label, "review completed");
                Ok(result.text)
            }
            Ok(Err(err)) => {
                warn!(job = %label, error = ?err, "review failed");
                Err(format!("*Failed: {err:#}*"))
            }
            Err(err) => {
                error!(job = %label, error = ?err, "review task panicked");
                Err(format!("*Failed (task panicked): {err:#}*"))
            }
        };
        collected.rendered.extend(rendered_section(
            &label,
            outcome.as_deref().map_err(String::as_str),
            presets.is_some(),
        ));
        let ok = outcome.is_ok();
        collected.verdict_records.push(VerdictRecord {
            lens: preset_name.clone(),
            stage: label.clone(),
            text: outcome.unwrap_or_else(|stub| stub),
            ok,
        });
        collected.job_records.push(JobRecord {
            label,
            preset: preset_name,
            ok,
        });
    }
    collected
}

/// The synthesis roster: presets with at least one surviving job, in resolution order. A
/// rubric with no matching report would read as an angle that was reviewed and found clean.
fn surviving_presets(
    presets: Option<&[crate::presets::ReviewPreset]>,
    coverage: Option<&[PresetCoverage]>,
) -> Vec<crate::presets::ReviewPreset> {
    match (presets, coverage) {
        (Some(presets), Some(coverage)) => presets
            .iter()
            .zip(coverage)
            .filter(|(_, c)| c.succeeded > 0)
            .map(|(p, _)| p.clone())
            .collect(),
        _ => Vec::new(),
    }
}

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

/// Plan the job matrix: one job per reviewer for `ask` (`presets: None`), reviewers ×
/// presets for review runs. Labels must be unique — the aggregator attributes reports by
/// them: duplicate reviewer names get an ` #<index>` suffix, and any residual collision
/// (a crafted name that embeds the suffix) falls back to the globally-unique job ordinal.
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
/// execution noise, not review evidence (degraded accounting and the logs carry it; the
/// synthesizer learns of the gap through `prompts::coverage_gaps`, and the report through
/// `output::with_partial_review_note`), while the Ask path keeps its pre-fan-out stubs
/// byte-for-byte.
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

    fn spawned(
        label: &str,
        preset_index: Option<usize>,
        outcome: Option<Result<&'static str>>,
    ) -> JobHandle {
        let handle = tokio::spawn(async move {
            match outcome {
                Some(Ok(text)) => Ok(AgentResult {
                    text: text.to_string(),
                    turns: 1,
                    tool_calls: 0,
                    subagents_spawned: 2,
                    usage: nitpicker_agent::llm::TokenUsage::default(),
                }),
                Some(Err(err)) => Err(err),
                None => panic!("job panicked"),
            }
        });
        (label.to_string(), preset_index, handle)
    }

    #[tokio::test]
    async fn collected_jobs_drive_success_count_and_surviving_roster() {
        let presets = presets(&["security", "tone", "simplicity"]);
        let collected = collect_jobs(
            vec![
                spawned("security · a", Some(0), Some(Ok("finding"))),
                spawned("security · b", Some(0), Some(Err(eyre::eyre!("boom")))),
                spawned("tone · a", Some(1), None),
                spawned("simplicity · a", Some(2), Some(Ok("clean"))),
            ],
            Some(&presets),
        )
        .await;
        let oks: Vec<bool> = collected.job_records.iter().map(|j| j.ok).collect();
        assert_eq!(oks, [true, false, false, true]);
        let verdict_oks: Vec<bool> = collected.verdict_records.iter().map(|v| v.ok).collect();
        assert_eq!(verdict_oks, oks);
        assert_eq!(collected.rendered.len(), 2);
        assert_eq!(collected.usage.subagents_spawned, 4);

        let coverage = preset_coverage(&presets, &collected.job_records);
        let roster: Vec<String> = surviving_presets(Some(&presets), Some(&coverage))
            .into_iter()
            .map(|p| p.name)
            .collect();
        assert_eq!(roster, ["security", "simplicity"]);
    }

    #[tokio::test]
    async fn ask_jobs_keep_failure_stubs_in_the_synthesis_input() {
        let collected = collect_jobs(
            vec![
                spawned("a", None, Some(Ok("answer"))),
                spawned("b", None, Some(Err(eyre::eyre!("boom")))),
            ],
            None,
        )
        .await;
        let oks: Vec<bool> = collected.job_records.iter().map(|j| j.ok).collect();
        assert_eq!(oks, [true, false]);
        assert_eq!(collected.rendered.len(), 2);
        assert!(surviving_presets(None, None).is_empty());
    }
}
