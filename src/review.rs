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
use nitpicker_agent::session::{
    AggregationRecord, JobRecord, SessionLogger, sanitize_path_component,
};
use nitpicker_agent::tools::{all_tools, floor_char_boundary};
use rig_core::completion::Message;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use tokio::sync::Semaphore;

use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

/// The pre-preset whole-agent concurrency bound, kept for the Ask path only.
const LEGACY_ASK_CONCURRENT_AGENTS: usize = 8;

pub struct ReviewOutcome {
    pub report: String,
    pub usage: UsageReport,
    /// At least one review job failed; the report is synthesized from the survivors.
    /// Surfaced as exit code 3 in the default-review/`ask`/`pr` CLI arms.
    pub degraded: bool,
    /// Presets with at least one surviving job — the angles the synthesis evidence actually
    /// covered, where the resolved list documents only the selection. `None` on the Ask path.
    pub covered_presets: Option<Vec<String>>,
}

pub async fn run_review(
    repo: &Path,
    user_prompt: &str,
    config: &Config,
    max_turns: usize,
    verbose: bool,
    mode: TaskMode,
    presets: Option<&[crate::presets::ReviewPreset]>,
) -> Result<ReviewOutcome> {
    match (&mode, presets) {
        (TaskMode::Review(_), Some(_)) | (TaskMode::Ask, None) => {}
        (TaskMode::Review(_), None) | (TaskMode::Ask, Some(_)) => {
            unreachable!("Review runs take the resolved presets; Ask takes none")
        }
    }
    let mut tools = all_tools();
    add_spawn_subagent_tool(&mut tools);
    let session_logger = SessionLogger::maybe_new(config.log_trajectories())?;
    if let Some(logger) = &session_logger {
        info!(path = %logger.root().display(), "trajectory logging enabled");
    }
    let context = crate::context::build_context(repo).await;
    let initial_message = mode.initial_message(user_prompt);
    let reviewer_names: Vec<&str> = config.reviewer.iter().map(|r| r.name.as_str()).collect();
    let jobs = plan_jobs(&reviewer_names, presets);
    // per-preset prompts are identical across reviewers — compose each once, not per job
    let system_prompts: Vec<String> = match presets {
        Some(presets) => presets
            .iter()
            .map(|p| mode.system_prompt(Some(p)))
            .collect(),
        None => vec![mode.system_prompt(None)],
    };
    let subagent_prompts: Vec<Option<String>> = match presets {
        Some(presets) => presets
            .iter()
            .map(|p| Some(crate::prompts::preset_subagent_prompt(p)))
            .collect(),
        None => vec![None],
    };
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
    let gemini_proxy = crate::proxy::GeminiProxy::maybe_start(config).await?;
    let proxy_url = gemini_proxy.url();

    // One client per reviewer, shared by all its preset jobs. eyre::Report is not Clone, so
    // a build failure is kept as its rendered message and re-raised per job — one broken
    // reviewer fails its own jobs while the others proceed, exactly as before the fan-out.
    let reviewer_clients: Vec<std::result::Result<_, String>> = config
        .reviewer
        .iter()
        .map(|r| build_reviewer_client(r, proxy_url.as_deref()).map_err(|e| format!("{e:#}")))
        .collect();

    for job in &jobs {
        let reviewer = &config.reviewer[job.reviewer_index];
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
        let agent_config = match &reviewer_clients[job.reviewer_index] {
            Ok(client) => Ok(build_agent_config(
                config,
                reviewer,
                Arc::clone(client),
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
            Err(msg) => Err(eyre::eyre!("{msg}")),
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
        let handle: JoinHandle<Result<AgentResult>> = tokio::spawn(async move {
            let _agent_permit = match &agent_sem {
                Some(sem) => Some(sem.acquire().await.expect("semaphore closed")),
                None => None,
            };
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
        handles.push((label, job.preset_index, handle));
    }

    let job_count = handles.len();
    let preset_run = presets.is_some();
    let mut usage = UsageReport::default();
    let mut rendered = Vec::new();
    let mut success_count = 0usize;
    let mut surviving_preset_indices = std::collections::HashSet::new();
    let mut job_records: Vec<JobRecord> = Vec::new();
    for (label, preset_index, handle) in handles {
        let preset_name = match (presets, preset_index) {
            (Some(ps), Some(j)) => Some(ps[j].name.clone()),
            _ => None,
        };
        let ok = match handle.await {
            Ok(Ok(result)) => {
                usage.add(result.usage, result.subagents_spawned);
                rendered.extend(rendered_section(&label, Ok(&result.text), preset_run));
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
                warn!(job = %label, error = ?err, "review failed");
                false
            }
            Err(err) => {
                let stub = format!("*Failed (task panicked): {err:#}*");
                rendered.extend(rendered_section(&label, Err(&stub), preset_run));
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

    // Refuse to synthesize a verdict out of nothing but failures: the aggregator would hallucinate
    // a confident review from error notes, and `pr` would post it. A total failure is an error, not
    // an "ok" report with empty findings.
    if success_count == 0 {
        eyre::bail!("all {job_count} review job(s) failed; refusing to synthesize a verdict");
    }

    let combined = rendered.join("\n\n---\n\n");
    // The synthesis roster covers only presets with at least one surviving job — a rubric
    // with no matching report would read as an angle that was reviewed and found clean.
    // (The session record and `pr --json` keep the FULL resolved list: they document the
    // run's resolution, not its coverage.)
    let surviving_presets: Option<Vec<crate::presets::ReviewPreset>> = presets.map(|ps| {
        ps.iter()
            .enumerate()
            .filter(|(j, _)| surviving_preset_indices.contains(j))
            .map(|(_, p)| p.clone())
            .collect()
    });
    let reduce_prompt = mode.reduce_prompt(user_prompt, &combined, surviving_presets.as_deref());

    let pb_agg = mp.add(ProgressBar::new_spinner());
    pb_agg.set_style(spinner_style);
    pb_agg.set_prefix("aggregator");
    pb_agg.set_message(crate::progress::bar_message("synthesizing…"));
    pb_agg.enable_steady_tick(Duration::from_millis(80));

    let agg = &config.aggregator;
    let synthesis: Result<String> = async {
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
        // preset runs get the count/context wrapping; the Ask path propagates the provider
        // error untouched, as it did before the fan-out
        let response = client.completion(completion).await.map_err(|err| match presets {
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
        if response.finish_reason == FinishReason::ToolUse {
            eyre::bail!("aggregator returned tool calls unexpectedly");
        }
        Ok(response.text())
    }
    .await;
    pb_agg.set_style(done_style);
    // A post-collection synthesis failure still persists the per-job outcomes: the jobs
    // list is the durable record of what ran, and losing it because the aggregator died
    // is exactly when a post-mortem needs it. The record carries `error` and an empty
    // `text` so consumers (reflect) don't render it as a verdict.
    let text = match synthesis {
        Ok(text) => {
            pb_agg.finish_with_message("✓ done");
            text
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
                model: agg.model.clone(),
                text: text.clone(),
                error: None,
                rounds: None,
                converged: None,
                presets: presets.map(|ps| ps.iter().map(|p| p.name.clone()).collect()),
                lanes: None,
                jobs: Some(job_records),
            })
            .await?;
    }
    Ok(ReviewOutcome {
        report: text,
        usage,
        degraded: success_count < job_count,
        covered_presets: surviving_presets.map(|ps| ps.into_iter().map(|p| p.name).collect()),
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
    config: &Config,
    reviewer: &ReviewerConfig,
    client: Arc<dyn nitpicker_agent::llm::LLMClientDyn>,
    session_agent: String,
    system_prompt: String,
    subagent_system_prompt: Option<String>,
    max_turns: usize,
    subagent_counter: Arc<AtomicUsize>,
    llm_semaphore: Arc<Semaphore>,
    session_writer: Option<nitpicker_agent::session::SessionWriter>,
) -> AgentConfig {
    let compact_threshold = config.reviewer_compact_threshold(reviewer);

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
