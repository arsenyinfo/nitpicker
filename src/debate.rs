use crate::output::UsageReport;
use crate::prompts::RunTask;
use crate::telemetry::{self, RunMode};
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use nitpicker_agent::agent::{
    AgentConfig, AgentDepth, AgentProgress, MAX_CONCURRENT_LLM_CALLS, add_spawn_subagent_tool,
    run_agent,
};
use nitpicker_agent::config::Config;
use nitpicker_agent::llm::{Completion, LLMClientDyn, TokenUsage, is_operational_limit_error};
use nitpicker_agent::provider::{build_aggregator_client, build_reviewer_client};
use nitpicker_agent::session::{
    AggregationRecord, LaneRecord, SessionLogger, SessionWriter, VerdictRecord,
    sanitize_path_component,
};
use nitpicker_agent::telemetry::bounded;
use nitpicker_agent::tools::{Tool, all_tools};
use rig_core::completion::Message;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use termimad::MadSkin;
use tracing::field::Empty;
use tracing::{Instrument, info_span, warn};

struct ModelLabel {
    alias: String, // short name used in agent name / logs
    full: String,  // full identifier used in cast line and transcript
}

impl ModelLabel {
    fn plain(model: &str) -> Self {
        Self {
            alias: model.to_string(),
            full: model.to_string(),
        }
    }

    fn alloy(models: impl Iterator<Item = impl AsRef<str>>) -> Self {
        let joined = models
            .map(|m| m.as_ref().to_string())
            .collect::<Vec<_>>()
            .join(" + ");
        Self {
            alias: "alloy".to_string(),
            full: format!("Alloy ({joined})"),
        }
    }
}

struct DebateVerdict {
    text: String,
    agree: bool,
}

struct DebateTurnResult {
    verdict: DebateVerdict,
    turns: usize,
    tool_calls: usize,
    subagents_spawned: usize,
    usage: TokenUsage,
    /// The agent errored and `verdict` is a synthesized failure stub rather than a real verdict.
    agent_failed: bool,
}

struct LaneVerdict {
    role: String,
    round: usize,
    text: String,
    agent_failed: bool,
}

struct DebateTurnRequest<'a> {
    client: Arc<dyn LLMClientDyn>,
    compact_threshold: Option<u64>,
    max_tokens: Option<u64>,
    model: &'a str,
    system_prompt: &'a str,
    /// Preset lanes hand their subagents a rubric-aware prompt; Topic leaves the generic one.
    subagent_system_prompt: Option<String>,
    initial_message: &'a str,
    max_turns: usize,
    work_dir: &'a Path,
    /// One per run, shared by every lane and subagent — concurrent lanes must split the
    /// account-wide in-flight cap, not multiply it.
    llm_semaphore: Arc<tokio::sync::Semaphore>,
    /// One user-facing failure warning per run; full per-turn errors remain available at debug.
    failure_warning_emitted: Arc<AtomicBool>,
    progress: Option<Arc<dyn Fn(AgentProgress) + Send + Sync>>,
    project_context: Option<String>,
    /// Trajectory identity (`[lane-<j>-<preset>-]<side>-<round>`, the writer's file stem):
    /// `reflect` merges all of a session's files by timestamp, so the record label is what
    /// keeps turns distinguishable.
    session_agent: String,
    session_writer: Option<SessionWriter>,
}

struct SubmitVerdictTool {
    verdict: Arc<Mutex<Option<DebateVerdict>>>,
}

impl Tool for SubmitVerdictTool {
    fn name(&self) -> String {
        "submit_verdict".to_string()
    }

    fn definition(&self) -> rig_core::completion::ToolDefinition {
        rig_core::completion::ToolDefinition {
            name: "submit_verdict".to_string(),
            description: "Submit your final position for this round. \
                Set agree=true only when the opponent's latest position can be forwarded \
                unchanged: no corrections, caveats, unresolved blockers, or changed finding set. \
                An agreeing verdict is the forwardable position itself, not an audit narrative."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "description": "Your final position for this round"
                    },
                    "agree": {
                        "type": "boolean",
                        "description": "True only for literal, unchanged agreement; any correction or unresolved point requires false"
                    }
                },
                "required": ["verdict", "agree"],
                "additionalProperties": false
            }),
        }
    }

    fn call(
        &self,
        args: Value,
        _work_dir: PathBuf,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>> {
        let verdict_store = Arc::clone(&self.verdict);
        // a turn's tool calls run concurrently (agent.rs phase 2). When this future is polled it runs
        // straight to the store write with no await before it, and join_all polls futures in provider
        // order — so if a single (malformed) turn emits multiple submit_verdict calls, the provider-last
        // one deterministically wins. Keep this future await-free before the write or that breaks.
        Box::pin(async move {
            let text = args
                .get("verdict")
                .and_then(|v| v.as_str())
                .filter(|v| !v.trim().is_empty())
                .ok_or_else(|| eyre::eyre!("missing or blank verdict"))?
                .to_string();
            // accept both bool true and string "true" in case the model serializes it as a string
            let agree = match args.get("agree") {
                Some(Value::Bool(b)) => *b,
                Some(Value::String(s)) => s.eq_ignore_ascii_case("true"),
                _ => false,
            };
            *verdict_store.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(DebateVerdict { text, agree });
            Ok("ok".to_string())
        })
    }
}

async fn run_debate_turn(request: DebateTurnRequest<'_>) -> DebateTurnResult {
    let verdict_store: Arc<Mutex<Option<DebateVerdict>>> = Arc::new(Mutex::new(None));
    let submit_tool = Arc::new(SubmitVerdictTool {
        verdict: Arc::clone(&verdict_store),
    });

    let mut tools_map: HashMap<String, Arc<dyn Tool>> = all_tools();
    add_spawn_subagent_tool(&mut tools_map);
    tools_map.insert("submit_verdict".to_string(), submit_tool as Arc<dyn Tool>);
    let subagent_counter = Arc::new(AtomicUsize::new(0));
    let config = AgentConfig {
        name: format!("debate-{}", request.model),
        session_agent: request.session_agent,
        model: request.model.to_string(),
        max_turns: request.max_turns,
        max_tokens: request.max_tokens,
        compact_threshold: request.compact_threshold,
        system_prompt: request.system_prompt.to_string(),
        subagent_system_prompt: request.subagent_system_prompt,
        client: request.client,
        depth: AgentDepth::TopLevel,
        terminal_tools: vec!["submit_verdict".to_string()],
        empty_response_nudge: Some(
            "Please proceed with your analysis and call submit_verdict when you are done."
                .to_string(),
        ),
        max_empty_responses: 3,
        subagent_counter,
        llm_semaphore: request.llm_semaphore,
        progress: request.progress,
        project_context: request.project_context,
        session_writer: request.session_writer,
    };

    let outcome = run_agent(
        config,
        request.initial_message,
        &tools_map,
        request.work_dir,
    )
    .await
    .and_then(|result| {
        let stored = verdict_store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let verdict = stored.ok_or_else(|| {
            eyre::eyre!("debate agent completed without calling required submit_verdict tool")
        })?;
        Ok((result, verdict))
    });
    match outcome {
        Ok((result, verdict)) => DebateTurnResult {
            verdict,
            turns: result.turns,
            tool_calls: result.tool_calls,
            subagents_spawned: result.subagents_spawned,
            usage: result.usage,
            agent_failed: false,
        },
        Err(err) => {
            tracing::debug!(model = request.model, error = ?err, "debate agent failed");
            if claim_failure_warning(&request.failure_warning_emitted) {
                warn!("{}", debate_failure_warning(&err));
            }
            DebateTurnResult {
                verdict: DebateVerdict {
                    text: format!("*Agent failed: {err:#}*"),
                    agree: false,
                },
                turns: 0,
                tool_calls: 0,
                subagents_spawned: 0,
                usage: TokenUsage::default(),
                agent_failed: true,
            }
        }
    }
}

fn debate_failure_warning(err: &eyre::Report) -> &'static str {
    if is_operational_limit_error(err) {
        "A model reached its usage limit; continuing with the remaining debate where possible"
    } else {
        "A debate turn failed; continuing with the remaining debate where possible"
    }
}

fn claim_failure_warning(emitted: &AtomicBool) -> bool {
    !emitted.swap(true, Ordering::AcqRel)
}

/// One debater's fixed identity for the whole debate — the per-side inputs that would
/// otherwise be spelled out twice per round.
struct DebateSide<'a> {
    role: &'static str,
    client: Arc<dyn LLMClientDyn>,
    compact_threshold: Option<u64>,
    max_tokens: Option<u64>,
    model: &'a str,
    system_prompt: String,
    /// Trajectory filename stem; the round number is appended.
    session_stem: String,
}

/// Everything a lane's turns need that is identical for both sides. One instance per lane;
/// most fields borrow run-level state, while progress is owned by the lane.
struct RoundEnv<'a> {
    skin: &'a MadSkin,
    repo: &'a Path,
    topic: &'a str,
    project_context: &'a str,
    session_logger: Option<&'a SessionLogger>,
    max_turns: usize,
    verbose: bool,
    stdout_ok: bool,
    llm_semaphore: &'a Arc<tokio::sync::Semaphore>,
    failure_warning_emitted: &'a Arc<AtomicBool>,
    subagent_system_prompt: Option<String>,
    /// Print verdicts as turns finish (single lane only) — concurrent lanes buffer instead,
    /// since their interleaved output would be unattributable.
    live_output: bool,
    /// One bar owns the lane's row for its full lifetime. It is hidden by indicatif outside
    /// the interactive non-verbose path, so the lifecycle does not need a second code path.
    lane_progress: ProgressBar,
}

/// Run one side's turn, updating its lane-owned progress row before and after the agent.
async fn run_debate_side(
    side: &DebateSide<'_>,
    env: &RoundEnv<'_>,
    verdicts: &[LaneVerdict],
    round: usize,
) -> DebateTurnResult {
    let pb = env.lane_progress.clone();
    let role = colored_role_stderr(side.role);
    pb.set_message(crate::progress::bar_message(format!(
        "{role} · round {round} — debating…"
    )));
    let msg = build_turn_message(env.topic, verdicts, round, side.role);
    let start = std::time::Instant::now();
    let progress_pb = pb.clone();
    let progress_role = role.clone();
    let progress = (!env.verbose).then_some(Arc::new(move |progress: AgentProgress| {
        progress_pb.set_message(crate::progress::bar_message(format!(
            "{progress_role} · round {round} — debating… ({} turns, {} tool calls, {} subagents)",
            progress.turns, progress.tool_calls, progress.subagents_spawned
        )));
    }) as Arc<dyn Fn(AgentProgress) + Send + Sync>);

    let session_agent = format!("{}-{round}", side.session_stem);
    let turn_span = info_span!(
        "debate.turn",
        otel.status_code = Empty,
        nitpicker.role = %bounded(side.role),
        nitpicker.round = round as u64,
        gen_ai.request.model = %bounded(side.model),
        nitpicker.agree = Empty,
    );
    let result = run_debate_turn(DebateTurnRequest {
        client: Arc::clone(&side.client),
        compact_threshold: side.compact_threshold,
        max_tokens: side.max_tokens,
        model: side.model,
        system_prompt: &side.system_prompt,
        subagent_system_prompt: env.subagent_system_prompt.clone(),
        initial_message: &msg,
        max_turns: env.max_turns,
        work_dir: env.repo,
        llm_semaphore: Arc::clone(env.llm_semaphore),
        failure_warning_emitted: Arc::clone(env.failure_warning_emitted),
        progress,
        project_context: Some(env.project_context.to_string()),
        session_writer: env
            .session_logger
            .map(|logger| logger.child(format!("{session_agent}.jsonl"))),
        session_agent,
    })
    .instrument(turn_span.clone())
    .await;
    turn_span.record("nitpicker.agree", result.verdict.agree);
    if result.agent_failed {
        turn_span.record("otel.status_code", "ERROR");
    }

    let elapsed = start.elapsed().as_secs();
    pb.set_message(crate::progress::bar_message(format!(
        "{role} ✓ round {round} ({} turns, {} tool calls, {} subagents, {}, {} out, {elapsed}s)",
        result.turns,
        result.tool_calls,
        result.subagents_spawned,
        crate::progress::input_with_cache_share(
            result.usage.input_tokens,
            result.usage.cached_input_tokens
        ),
        crate::progress::compact_tokens(result.usage.output_tokens)
    )));
    if env.live_output && env.verbose && env.stdout_ok && stdout_is_terminal() {
        println!();
        env.skin.print_text(&result.verdict.text);
        println!();
    }
    result
}

fn build_turn_message(topic: &str, verdicts: &[LaneVerdict], round: usize, role: &str) -> String {
    let mut msg = format!("Topic: {topic}\n");
    if verdicts.is_empty() {
        msg.push_str("\nNo prior dialogue yet.\n");
    } else {
        msg.push_str("\nDialogue so far:\n");
        for verdict in verdicts {
            msg.push_str(&format!(
                "\n### {} (Round {})\n{}\n",
                verdict.role, verdict.round, verdict.text
            ));
        }
    }
    if round > 1 {
        msg.push_str(
            "\nThis is a follow-up round: respond to the latest opposing verdict and investigate \
             only its disputed or changed points; do not re-run discovery over the whole target.\n",
        );
    }
    msg.push_str(&format!(
        "\n---\nRound {round} — your turn as {role}. Explore the codebase as needed, then call submit_verdict."
    ));
    msg
}

fn role_color(role: &str) -> &'static str {
    match role {
        "Actor" | "Reviewer" => "\x1b[96m",   // bright cyan
        "Critic" | "Validator" => "\x1b[93m", // bright yellow
        "Meta-review" => "\x1b[92m",          // bright green
        _ => "",
    }
}

fn use_color() -> bool {
    stdout_is_terminal() && crate::progress::color_env_allows()
}

fn use_stderr_color() -> bool {
    crate::progress::stderr_supports_color()
}

fn stdout_is_terminal() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

fn colored_role(role: &str) -> String {
    if use_color() {
        format!("{}{role}\x1b[0m", role_color(role))
    } else {
        role.to_string()
    }
}

fn colored_role_stderr(role: &str) -> String {
    if use_stderr_color() {
        format!("{}{role}\x1b[0m", role_color(role))
    } else {
        role.to_string()
    }
}

fn print_cast_line(role: &str, info: &str) {
    let pad = " ".repeat(12usize.saturating_sub(role.len()));
    println!("  {}{pad} {info}", colored_role(role));
}

fn make_spinner(mp: &MultiProgress) -> ProgressBar {
    let spinner_style = ProgressStyle::with_template("{spinner:.cyan} {prefix:<12} {msg}")
        .unwrap()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", ""]);
    let pb = mp.add(ProgressBar::new_spinner());
    pb.set_style(spinner_style);
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

pub struct DebateOptions<'a> {
    pub max_rounds: usize,
    pub max_turns: usize,
    pub verbose: bool,
    pub task: RunTask<'a>,
    pub alloy: bool,
    pub fallback: bool,
    pub format: crate::output::OutputFormat,
}

pub struct DebateOutcome {
    pub report: String,
    pub transcript_path: std::path::PathBuf,
    pub usage: UsageReport,
    /// At least one turn failed; the report is synthesized from a partial dialogue. Surfaced as
    /// exit code 3 in the default-review/`ask`/`pr` CLI arms.
    pub degraded: bool,
    /// Per-preset lane counts in resolution order (attempted is always 1 — one lane per
    /// preset), matching the parallel path's per-job shape. `None` for Ask.
    pub coverage: Option<Vec<crate::output::PresetCoverage>>,
}

/// One lane's full result: an independent actor/critic debate over a single preset (or the
/// whole topic for `ask`). Convergence, degradation, and the dialogue are all lane-local;
/// the global meta-review synthesizes across surviving lanes.
struct DebateLaneOutcome {
    preset_name: Option<String>,
    verdicts: Vec<LaneVerdict>,
    converged: bool,
    final_round: usize,
    any_turn_succeeded: bool,
    degraded: bool,
    usage: UsageReport,
}

pub async fn run_debate(
    repo: &Path,
    prompt: &str,
    config: &Config,
    opts: DebateOptions<'_>,
) -> Result<DebateOutcome> {
    let span = telemetry::run_span(RunMode::Debate, &opts.task, config.reviewer.len());
    telemetry::record_run(span, run_debate_inner(repo, prompt, config, opts), |o| {
        (o.degraded, &o.usage)
    })
    .await
}

async fn run_debate_inner(
    repo: &Path,
    prompt: &str,
    config: &Config,
    opts: DebateOptions<'_>,
) -> Result<DebateOutcome> {
    let DebateOptions {
        max_rounds,
        max_turns,
        verbose,
        task,
        alloy,
        fallback,
        format,
    } = opts;
    let presets = task.presets();
    let lane_tasks = task.lanes();
    // in json mode stdout is reserved for the final envelope; the cast lines and
    // rendered verdicts below would otherwise corrupt it.
    let stdout_ok = matches!(format, crate::output::OutputFormat::Text);
    let _terminal_title = crate::progress::start_terminal_title(repo, format);
    if config.reviewer.len() < 2 {
        eyre::bail!(
            "debate requires at least 2 reviewers in config (actor = reviewer[0], critic = reviewer[1])"
        );
    }

    let agg_cfg = &config.aggregator;

    // proxy handle stays bound for the function so its local server outlives the debate;
    // only its base URL is threaded into the client builders.
    let gemini_proxy = crate::proxy::GeminiProxy::maybe_start(config).await;
    let proxy_url = gemini_proxy.url();

    let reviewer_pool = (alloy || fallback)
        .then(|| crate::routing::build_reviewer_pool(config, &gemini_proxy, proxy_url.as_deref()));
    let (actor, critic) = resolve_roles(
        config,
        &gemini_proxy,
        proxy_url.as_deref(),
        reviewer_pool.as_ref(),
        alloy,
        fallback,
    )?;
    let session_logger = crate::run::open_session(config).await?;

    let project_context = crate::context::build_context(repo).await;

    let agg_client: Arc<dyn LLMClientDyn> = match &reviewer_pool {
        Some(pool) if fallback => crate::routing::aggregator_client(
            config,
            &gemini_proxy,
            proxy_url.as_deref(),
            pool,
            true,
        )?,
        _ => gemini_proxy.annotate(build_aggregator_client(agg_cfg, proxy_url.as_deref()))?,
    };

    let actor_role = task.actor_role();
    let critic_role = task.critic_role();
    // one in-flight-LLM cap for the whole run: concurrent lanes and their subagents share
    // it, so lane count scales wall-clock breadth without multiplying provider pressure
    let llm_semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_LLM_CALLS));
    let failure_warning_emitted = Arc::new(AtomicBool::new(false));

    let done_style = ProgressStyle::with_template("  {prefix:<12} {msg}").unwrap();
    let skin = MadSkin::default();

    let (mp, _progress_guard) = crate::run::progress_display(verbose);

    // cast lines show which models are participating in interactive text mode, but piped/json
    // stdout stays machine-readable/final-report-only.
    if stdout_ok && stdout_is_terminal() {
        print_cast(task, config, &actor, &critic, alloy);
    }

    // One lane per preset for Review; Ask is the degenerate single unscoped lane and
    // keeps its exact pre-preset behavior, live verbose output included.
    let live_output = lane_tasks.len() == 1;

    let lane_futures = lane_tasks
        .iter()
        .enumerate()
        .map(|(lane_index, lane_task)| {
            let preset = lane_task.preset();
            // Sticky failover belongs to one logical agent: each lane forks the role templates, so
            // forks keep independent active indices while their cloned FallbackSlots share
            // run-wide route availability.
            let lane_client = |role: &DebateRole| match fallback && !alloy {
                true => crate::routing::independent_agent_client(&role.client),
                false => Arc::clone(&role.client),
            };
            let (actor_stem, critic_stem) = lane_session_stems(lane_index, preset);
            let actor_side = DebateSide {
                role: actor_role,
                client: lane_client(&actor),
                compact_threshold: actor.compact_threshold,
                max_tokens: actor.max_tokens,
                model: &actor.label.alias,
                system_prompt: lane_task.actor_system(),
                session_stem: actor_stem,
            };
            let critic_side = DebateSide {
                role: critic_role,
                client: lane_client(&critic),
                compact_threshold: critic.compact_threshold,
                max_tokens: critic.max_tokens,
                model: &critic.label.alias,
                system_prompt: lane_task.critic_system(),
                session_stem: critic_stem,
            };
            let lane_progress = make_spinner(&mp);
            lane_progress.set_prefix(preset.map_or("debate", |p| p.name.as_str()).to_string());
            lane_progress.set_message(crate::progress::bar_message("waiting…"));
            let env = RoundEnv {
                skin: &skin,
                repo,
                topic: prompt,
                project_context: &project_context,
                session_logger: session_logger.as_ref(),
                max_turns,
                verbose,
                stdout_ok,
                llm_semaphore: &llm_semaphore,
                failure_warning_emitted: &failure_warning_emitted,
                subagent_system_prompt: lane_task.subagent_prompt(),
                live_output,
                lane_progress,
            };
            let lane_span = info_span!(
                "debate.lane",
                nitpicker.lane = lane_index as u64,
                nitpicker.preset = preset.map_or("", |p| bounded(&p.name)),
                nitpicker.rounds = Empty,
                nitpicker.converged = Empty,
                nitpicker.degraded = Empty,
            );
            run_lane(
                preset.map(|p| p.name.clone()),
                actor_side,
                critic_side,
                env,
                max_rounds,
                &done_style,
            )
            .instrument(lane_span)
        });
    let lanes: Vec<DebateLaneOutcome> = futures::future::join_all(lane_futures).await;

    let mut usage = UsageReport::default();
    let mut degraded = false;
    for lane in &lanes {
        usage.merge(&lane.usage);
        degraded |= lane.degraded;
    }

    // Concurrent lanes buffered their dialogue (live printing would interleave); surface it
    // per lane now that ordering is deterministic.
    if !live_output && verbose && stdout_ok && stdout_is_terminal() {
        for lane in &lanes {
            if let Some(name) = &lane.preset_name {
                println!("\n─── {name} ───");
            }
            for verdict in &lane.verdicts {
                println!("\n{} (Round {}):", verdict.role, verdict.round);
                skin.print_text(&verdict.text);
            }
        }
    }

    // Lane metadata travels on every outcome: the scalars follow the single-lane derivation
    // whether the run fails or succeeds, so a failed `ask` run's record isn't shaped differently
    // from a successful one.
    let (rounds, converged) = single_lane_scalars(&lanes);
    let record = crate::run::RunRecord {
        logger: session_logger.as_ref(),
        base: AggregationRecord {
            kind: "aggregation".to_string(),
            model: agg_cfg.model.clone(),
            text: String::new(),
            error: None,
            rounds,
            converged,
            presets: presets.map(|ps| ps.iter().map(|p| p.name.clone()).collect()),
            lanes: presets.map(|_| lane_records(&lanes)),
            verdicts: debate_verdict_records(&lanes),
            jobs: None,
        },
    };

    // Every turn failed everywhere (provider down, bad config): the dialogue is nothing but
    // failure stubs, so synthesizing a meta-verdict would fabricate a confident review from
    // errors. Surface the failure instead — `run_pr` maps this to a `status: "error"`
    // envelope and posts no comment. A lane with no successful turn is likewise omitted from
    // synthesis below: its stubs are execution noise, not review evidence. The lane outcomes
    // are persisted first: a run where every turn failed is the one that most needs a durable
    // record of what was attempted.
    if surviving(&lanes).is_empty() {
        let err = eyre::eyre!("all debate turns failed; refusing to synthesize a verdict");
        record.persist_failure(&err, "all-turns-failed").await;
        return Err(err);
    }

    let coverage = presets.map(|ps| lane_coverage(ps, &lanes));

    // meta-review: non-agentic single completion over the surviving lanes' dialogue
    let survivors = surviving(&lanes);
    let meta_prompt = meta_prompt(task, prompt, &survivors, coverage.as_deref());
    let meta_completion = Completion {
        model: agg_cfg.model.clone(),
        prompt: Message::user(meta_prompt),
        preamble: Some(task.meta_preamble()),
        history: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        max_tokens: Some(config.aggregator_max_tokens()),
        additional_params: None,
    };
    let pb = make_spinner(&mp);
    pb.set_prefix(colored_role_stderr("Meta-review"));
    pb.set_message(crate::progress::bar_message("synthesizing…"));
    let synthesis = crate::run::synthesize(
        crate::run::Synthesis {
            client: Ok(agg_client),
            completion: meta_completion,
            role: "meta-review",
            failure_context: presets.map(|presets| {
                format!(
                    "meta-review failed over {} surviving lane(s) across {} preset(s)",
                    survivors.len(),
                    presets.len()
                )
            }),
            progress: pb,
            done_style,
            llm_semaphore: &llm_semaphore,
        },
        record,
    )
    .await?;
    usage.add(synthesis.usage, 0);
    let meta_text = synthesis.text;
    let meta_model = synthesis.model;
    // The transcript is a verbose-only debugging artifact. Keep both rendering and writing
    // behind the flag: multi-lane verdicts can be large, and non-verbose callers ignore the path.
    let transcript_path = if verbose {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut transcript = render_transcript(
            task,
            prompt,
            &actor.label.full,
            &critic.label.full,
            &meta_model,
            &lanes,
            max_rounds,
        );
        transcript.push_str(&format!("---\n\n## Meta-review\n\n{meta_text}\n"));
        write_transcript(&transcript_stem(task.label(), ts, presets), &transcript)?
    } else {
        PathBuf::new()
    };

    let report = match &coverage {
        Some(coverage) => crate::output::with_partial_review_note(
            meta_text,
            lanes.iter().filter(|lane| lane.degraded).count(),
            lanes.len(),
            "debate lanes",
            coverage,
        ),
        None => meta_text,
    };
    Ok(DebateOutcome {
        report,
        transcript_path,
        usage,
        degraded,
        coverage,
    })
}

/// One debate role's client and run-level settings, fixed for the whole run.
struct DebateRole {
    client: Arc<dyn LLMClientDyn>,
    label: ModelLabel,
    compact_threshold: Option<u64>,
    /// Output caps stay bound to the logical role; the routing clients replace them with the
    /// selected route's cap at the completion boundary.
    max_tokens: Option<u64>,
}

/// Actor (`reviewer[0]`) and critic (`reviewer[1]`) for the run's routing mode: a shared Alloy
/// pool, fallback priority rings, or each reviewer's own client.
fn resolve_roles(
    config: &Config,
    gemini_proxy: &crate::proxy::GeminiProxy,
    proxy_url: Option<&str>,
    reviewer_pool: Option<&crate::routing::ReviewerClientPool>,
    alloy: bool,
    fallback: bool,
) -> Result<(DebateRole, DebateRole)> {
    let shared = match alloy {
        true => Some(crate::routing::alloy_client(
            reviewer_pool.expect("alloy builds reviewer pool"),
            fallback,
        )?),
        false => None,
    };
    let pool_threshold =
        reviewer_pool.map(|pool| crate::routing::reviewer_pool_compact_threshold(config, pool));
    let role = |index: usize| -> Result<DebateRole> {
        let cfg = &config.reviewer[index];
        let client = match (&shared, reviewer_pool) {
            (Some(shared), _) => Arc::clone(shared),
            (None, Some(pool)) => crate::routing::debate_reviewer_client(pool, index, 1 - index)?,
            (None, None) => gemini_proxy.annotate(build_reviewer_client(cfg, proxy_url))?,
        };
        Ok(DebateRole {
            client,
            label: match alloy {
                true => ModelLabel::alloy(config.reviewer.iter().map(|r| r.model.as_str())),
                false => ModelLabel::plain(&cfg.model),
            },
            compact_threshold: match pool_threshold {
                Some(threshold) => threshold,
                None => config.reviewer_compact_threshold(cfg),
            },
            max_tokens: cfg.max_tokens,
        })
    };
    Ok((role(0)?, role(1)?))
}

fn print_cast(
    task: RunTask<'_>,
    config: &Config,
    actor: &DebateRole,
    critic: &DebateRole,
    alloy: bool,
) {
    let cast = [
        (task.actor_role(), &config.reviewer[0], actor),
        (task.critic_role(), &config.reviewer[1], critic),
    ];
    for (role, cfg, debate_role) in cast {
        match alloy {
            true => print_cast_line(role, &debate_role.label.full),
            false => print_cast_line(role, &format!("{} · {}", cfg.name, debate_role.label.full)),
        }
    }
    print_cast_line("Meta-review", &config.aggregator.model);
    if let Some(presets) = task.presets() {
        let names: Vec<&str> = presets.iter().map(|p| p.name.as_str()).collect();
        print_cast_line("Presets", &names.join(", "));
    }
    println!();
}

/// One lane's actor/critic rounds until the critic literally agrees or `max_rounds` runs out.
/// Turn failures fold into stubs, so a lane always produces an outcome.
async fn run_lane(
    preset_name: Option<String>,
    actor: DebateSide<'_>,
    critic: DebateSide<'_>,
    env: RoundEnv<'_>,
    max_rounds: usize,
    done_style: &ProgressStyle,
) -> DebateLaneOutcome {
    let started = std::time::Instant::now();
    let mut lane = DebateLaneOutcome {
        preset_name,
        verdicts: Vec::new(),
        converged: false,
        final_round: 0,
        any_turn_succeeded: false,
        degraded: false,
        usage: UsageReport::default(),
    };

    'debate: for round in 1..=max_rounds {
        lane.final_round = round;

        let actor_turn = run_debate_side(&actor, &env, &lane.verdicts, round).await;
        lane.usage
            .add(actor_turn.usage, actor_turn.subagents_spawned);
        lane.any_turn_succeeded |= !actor_turn.agent_failed;
        lane.degraded |= actor_turn.agent_failed;
        lane.verdicts.push(LaneVerdict {
            role: actor.role.to_string(),
            round,
            text: actor_turn.verdict.text,
            agent_failed: actor_turn.agent_failed,
        });

        let critic_turn = run_debate_side(&critic, &env, &lane.verdicts, round).await;
        lane.usage
            .add(critic_turn.usage, critic_turn.subagents_spawned);
        lane.any_turn_succeeded |= !critic_turn.agent_failed;
        lane.degraded |= critic_turn.agent_failed;
        // Convergence requires a real agreement: a critic that agrees with a failed
        // actor's `*Agent failed*` stub (or a failed critic, whose verdict defaults to
        // agree=false) must not end the debate early.
        let agreed =
            critic_turn.verdict.agree && !actor_turn.agent_failed && !critic_turn.agent_failed;
        lane.verdicts.push(LaneVerdict {
            role: critic.role.to_string(),
            round,
            text: critic_turn.verdict.text,
            agent_failed: critic_turn.agent_failed,
        });

        if agreed {
            lane.converged = true;
            break 'debate;
        }
    }

    env.lane_progress.set_style(done_style.clone());
    env.lane_progress
        .finish_with_message(crate::progress::bar_message(lane_progress_summary(
            &lane,
            started.elapsed().as_secs(),
        )));
    let span = tracing::Span::current();
    span.record("nitpicker.rounds", lane.final_round as u64);
    span.record("nitpicker.converged", lane.converged);
    span.record("nitpicker.degraded", lane.degraded);
    lane
}

/// The meta-review's user message: Topic keeps its single-lane shape; preset runs list the
/// surviving lanes only, with the roster and coverage gaps stated up front.
fn meta_prompt(
    task: RunTask<'_>,
    prompt: &str,
    survivors: &[&DebateLaneOutcome],
    coverage: Option<&[crate::output::PresetCoverage]>,
) -> String {
    match task.presets() {
        // Topic (`ask`): single lane, prompt shape unchanged
        None => {
            let note = match lane_pruned_to_final_round(survivors[0]) {
                true => {
                    "The debate converged; only the final round is shown — it supersedes the \
                 earlier dialogue.\n\n"
                }
                false => "",
            };
            format!(
                "The following is a debate about: {prompt}\n\n{note}{dialogue}\n\n---\n{instruction}",
                dialogue = lane_dialogue(survivors[0]),
                instruction = task.meta_instruction(),
            )
        }
        Some(presets) => {
            // roster covers surviving lanes only — a dead lane's rubric with no matching
            // section would read as an angle that was reviewed and found clean
            let surviving_presets: Vec<crate::presets::ReviewPreset> = presets
                .iter()
                .filter(|p| {
                    survivors
                        .iter()
                        .any(|lane| lane.preset_name.as_deref() == Some(p.name.as_str()))
                })
                .cloned()
                .collect();
            let gaps = crate::prompts::coverage_gaps(coverage.unwrap_or_default(), "lanes")
                .map(|gaps| format!("{gaps}\n\n"))
                .unwrap_or_default();
            format!(
                "The following are independent review debates about the same target, one per review angle.\n\
             Target: {prompt}\n\n{roster}\n\n{gaps}{sections}\n\n---\n\
             Notes:\n\
             - \"*Agent failed: …*\" markers are execution errors kept for chronology; they are \
             not review evidence and not agreement.\n\
             - A lane that ended without convergence carries unresolved disagreement — weigh it \
             by the evidence, do not read it as agreement.\n\
             - A lane marked degraded had a turn fail; its dialogue is partial.\n\
             {instruction}",
                roster = crate::prompts::preset_roster(&surviving_presets),
                sections = lane_sections(survivors),
                instruction = task.meta_instruction(),
            )
        }
    }
}

/// The verbose transcript up to (not including) the meta-review section.
fn render_transcript(
    task: RunTask<'_>,
    prompt: &str,
    actor_model: &str,
    critic_model: &str,
    meta_model: &str,
    lanes: &[DebateLaneOutcome],
    max_rounds: usize,
) -> String {
    let now = chrono::Local::now();
    let label = task.label();
    let actor_role = task.actor_role();
    let critic_role = task.critic_role();
    let presets = task.presets();
    match presets {
        // Ask: single lane, transcript shape unchanged
        None => {
            let lane = &lanes[0];
            let convergence_status = match lane.converged {
                true => format!("converged at round {}", lane.final_round),
                false => format!("max rounds ({max_rounds}) reached without convergence"),
            };
            let mut transcript = format!(
                "# Debate Transcript ({label})\n\n\
                    **Topic:** {prompt}\n\
                    **{actor_role} model:** {}\n\
                    **{critic_role} model:** {}\n\
                    **Meta-reviewer:** {}\n\
                    **Date:** {}\n\
                    **Convergence:** {convergence_status}\n\
                    **Rounds:** {}\n\n---\n\n",
                actor_model,
                critic_model,
                meta_model,
                now.format("%Y-%m-%d %H:%M:%S"),
                lane.final_round,
            );
            for verdict in &lane.verdicts {
                transcript.push_str(&format!(
                    "## {} — Round {}\n\n{}\n\n",
                    verdict.role, verdict.round, verdict.text
                ));
            }
            transcript
        }
        Some(_) => {
            let mut transcript = format!(
                "# Debate Transcript ({label})\n\n\
                    **Topic:** {prompt}\n\
                    **{actor_role} model:** {}\n\
                    **{critic_role} model:** {}\n\
                    **Meta-reviewer:** {}\n\
                    **Date:** {}\n\
                    **Lanes:** {}\n\n---\n\n",
                actor_model,
                critic_model,
                meta_model,
                now.format("%Y-%m-%d %H:%M:%S"),
                lanes.len(),
            );
            for lane in lanes {
                transcript.push_str(&render_lane_transcript_section(lane, max_rounds));
            }
            transcript
        }
    }
}

/// Per-preset lane coverage: attempted is always 1 (one lane per preset); a lane survived
/// iff at least one of its turns really ran.
fn lane_coverage(
    presets: &[crate::presets::ReviewPreset],
    lanes: &[DebateLaneOutcome],
) -> Vec<crate::output::PresetCoverage> {
    presets
        .iter()
        .map(|p| {
            let survived = lanes.iter().any(|lane| {
                lane.preset_name.as_deref() == Some(p.name.as_str()) && lane.any_turn_succeeded
            });
            crate::output::PresetCoverage {
                preset: p.name.clone(),
                attempted: 1,
                succeeded: usize::from(survived),
            }
        })
        .collect()
}

/// Trajectory stems for one lane's two sides. Topic keeps the pre-preset stems; preset
/// lanes prefix the lane index AND bounded preset slug — the index is load-bearing
/// because distinct preset names can sanitize (or truncate) to the same slug.
fn lane_session_stems(
    lane_index: usize,
    preset: Option<&crate::presets::ReviewPreset>,
) -> (String, String) {
    match preset {
        None => ("review".to_string(), "validate".to_string()),
        Some(preset) => {
            let prefix = format!(
                "lane-{}-{}",
                lane_index + 1,
                crate::presets::path_slug(&preset.name)
            );
            (format!("{prefix}-review"), format!("{prefix}-validate"))
        }
    }
}

/// The record's scalar `rounds`/`converged`, populated whenever there is exactly one lane
/// (Topic always; single-preset review) — pre-lanes `reflect` rendered only the scalars, so
/// clearing them for a one-lane run would lose metadata it had. Every persistence path uses
/// this, so a failed run's record keeps the same shape as a successful one's.
fn single_lane_scalars(lanes: &[DebateLaneOutcome]) -> (Option<usize>, Option<bool>) {
    match lanes {
        [lane] => (Some(lane.final_round), Some(lane.converged)),
        _ => (None, None),
    }
}

fn lane_records(lanes: &[DebateLaneOutcome]) -> Vec<LaneRecord> {
    lanes
        .iter()
        .map(|lane| LaneRecord {
            preset: lane.preset_name.clone().unwrap_or_default(),
            rounds: lane.final_round,
            converged: lane.converged,
            degraded: lane.degraded,
        })
        .collect()
}

fn debate_verdict_records(lanes: &[DebateLaneOutcome]) -> Vec<VerdictRecord> {
    lanes
        .iter()
        .flat_map(|lane| {
            lane.verdicts.iter().map(|verdict| VerdictRecord {
                lens: lane.preset_name.clone(),
                stage: format!("{} · round {}", verdict.role, verdict.round),
                text: verdict.text.clone(),
                ok: !verdict.agent_failed,
            })
        })
        .collect()
}

/// A lane survives — and reaches synthesis — iff at least one of its turns really ran.
fn surviving(lanes: &[DebateLaneOutcome]) -> Vec<&DebateLaneOutcome> {
    lanes
        .iter()
        .filter(|lane| lane.any_turn_succeeded)
        .collect()
}

fn lane_progress_summary(lane: &DebateLaneOutcome, elapsed: u64) -> String {
    let rounds = match lane.final_round {
        1 => "1 round".to_string(),
        count => format!("{count} rounds"),
    };
    let status = match (lane.any_turn_succeeded, lane.converged, lane.degraded) {
        (false, _, _) => format!("✗ failed after {rounds}"),
        (true, true, false) => format!("✓ converged at round {}", lane.final_round),
        (true, true, true) => format!("⚠ converged at round {} · degraded", lane.final_round),
        (true, false, false) => format!("✓ done after {rounds} · no convergence"),
        (true, false, true) => format!("⚠ done after {rounds} · no convergence · degraded"),
    };
    format!(
        "{status} ({}, {} out, {} subagents, {elapsed}s)",
        crate::progress::input_with_cache_share(
            lane.usage.input_tokens,
            lane.usage.cached_input_tokens
        ),
        crate::progress::compact_tokens(lane.usage.output_tokens),
        lane.usage.subagents_spawned,
    )
}

/// A cleanly converged lane is pruned to its final round in the meta input: verdicts are
/// self-contained by prompt contract and the agreeing critic restates every confirmed
/// finding, so earlier rounds are superseded chronology — exactly the material a
/// synthesizer misreads into withdrawn-claim narration. A *degraded* lane can converge after an
/// earlier failed turn, but the self-containment premise is then untrusted and its full trail stays
/// — as it does for contested lanes and the human transcript.
fn lane_pruned_to_final_round(lane: &DebateLaneOutcome) -> bool {
    lane.converged && !lane.degraded
}

/// Dialogue as the meta-review sees it (see `lane_pruned_to_final_round`).
fn lane_dialogue(lane: &DebateLaneOutcome) -> String {
    lane.verdicts
        .iter()
        .filter(|verdict| !lane_pruned_to_final_round(lane) || verdict.round == lane.final_round)
        .map(|verdict| {
            format!(
                "### {} (Round {})\n{}",
                verdict.role, verdict.round, verdict.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Meta-review input sections, surviving lanes only — convergence AND degradation state
/// are stated per lane, so unresolved disagreement arrives as disagreement and a partial
/// dialogue arrives flagged as partial.
fn lane_sections(survivors: &[&DebateLaneOutcome]) -> String {
    survivors
        .iter()
        .map(|lane| {
            let name = lane.preset_name.as_deref().unwrap_or("(unnamed)");
            let convergence = match (lane.converged, lane_pruned_to_final_round(lane)) {
                (true, true) => format!(
                    "converged at round {} (earlier rounds superseded and omitted)",
                    lane.final_round
                ),
                (true, false) => format!("converged at round {}", lane.final_round),
                (false, _) => format!("no convergence after {} round(s)", lane.final_round),
            };
            let degraded = match lane.degraded {
                true => " · degraded",
                false => "",
            };
            format!(
                "## Preset: {name} — {convergence}{degraded}\n\n{}",
                lane_dialogue(lane)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// `debate-{ts}.md` for Topic (unchanged); review runs append bounded sanitized preset
/// slugs so concurrent artifacts are attributable at a glance.
fn transcript_stem(
    label: &str,
    ts: u64,
    presets: Option<&[crate::presets::ReviewPreset]>,
) -> String {
    match presets {
        None => format!("{label}-{ts}"),
        Some(presets) => {
            let mut slugs = presets
                .iter()
                .map(|p| sanitize_path_component(&p.name))
                .collect::<Vec<_>>()
                .join("-");
            // sanitize output is ASCII, so a byte truncate cannot split a char
            const MAX_SLUG_LEN: usize = 80;
            slugs.truncate(MAX_SLUG_LEN);
            format!("{label}-{ts}-{slugs}")
        }
    }
}

/// The temp dir is shared and the stem predictable, so the file gets a random suffix and is
/// created exclusively, owner-only: a pre-planted symlink or file is never written through.
fn write_transcript(stem: &str, transcript: &str) -> Result<PathBuf> {
    let file = tempfile::Builder::new()
        .prefix(&format!("{stem}-"))
        .suffix(".md")
        .tempfile()?;
    std::io::Write::write_all(&mut file.as_file(), transcript.as_bytes())?;
    let (_, path) = file.keep()?;
    Ok(path)
}

/// Human transcript section for one lane — every lane appears (dead ones flagged), unlike
/// the meta input, because the transcript is the debugging artifact.
fn render_lane_transcript_section(lane: &DebateLaneOutcome, max_rounds: usize) -> String {
    let name = lane.preset_name.as_deref().unwrap_or("(unnamed)");
    let convergence = match lane.converged {
        true => format!("converged at round {}", lane.final_round),
        false => format!("max rounds ({max_rounds}) reached without convergence"),
    };
    let degraded = match lane.degraded {
        true => "yes",
        false => "no",
    };
    let mut section = format!(
        "## Preset: {name}\n\n**Convergence:** {convergence}\n**Degraded:** {degraded}\n\n"
    );
    if !lane.any_turn_succeeded {
        section.push_str("*Lane failed: no successful turns; omitted from synthesis.*\n\n");
    }
    for verdict in &lane.verdicts {
        section.push_str(&format!(
            "### {} — Round {}\n\n{}\n\n",
            verdict.role, verdict.round, verdict.text
        ));
    }
    section
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::ReviewPreset;

    fn preset(name: &str) -> ReviewPreset {
        ReviewPreset {
            name: name.to_string(),
            prompt: format!("rubric for {name}"),
        }
    }

    fn lane(preset_name: Option<&str>, verdicts: &[(&str, usize, &str)]) -> DebateLaneOutcome {
        DebateLaneOutcome {
            preset_name: preset_name.map(str::to_string),
            verdicts: verdicts
                .iter()
                .map(|(role, round, text)| LaneVerdict {
                    role: role.to_string(),
                    round: *round,
                    text: text.to_string(),
                    agent_failed: false,
                })
                .collect(),
            converged: false,
            final_round: 1,
            any_turn_succeeded: true,
            degraded: false,
            usage: UsageReport::default(),
        }
    }

    #[test]
    fn verdict_tool_defines_literal_agreement() {
        let tool = SubmitVerdictTool {
            verdict: Arc::new(Mutex::new(None)),
        };
        let definition = tool.definition();
        assert!(definition.description.contains("forwarded unchanged"));
        assert_eq!(
            definition.parameters["properties"]["agree"]["description"],
            "True only for literal, unchanged agreement; any correction or unresolved point requires false"
        );
    }

    #[tokio::test]
    async fn blank_verdict_is_rejected_without_populating_the_store() {
        let store = Arc::new(Mutex::new(None));
        let tool = SubmitVerdictTool {
            verdict: Arc::clone(&store),
        };
        for blank in ["", "  \n\t"] {
            let args = json!({ "verdict": blank, "agree": true });
            assert!(tool.call(args, PathBuf::new()).await.is_err());
            assert!(store.lock().unwrap().is_none());
        }
        let args = json!({ "verdict": "No findings.", "agree": true });
        tool.call(args, PathBuf::new()).await.unwrap();
        let stored = store.lock().unwrap().take().unwrap();
        assert_eq!((stored.text.as_str(), stored.agree), ("No findings.", true));
    }

    #[test]
    fn operational_failure_warning_is_concise_and_emitted_once() {
        let err = eyre::Report::new(rig_core::completion::CompletionError::from_http_response(
            reqwest::StatusCode::FORBIDDEN,
            r#"{"error":{"type":"access_terminated_error","message":"You've reached your usage limit for this billing cycle. Your quota will be refreshed in the next cycle"}}"#,
        ));
        assert_eq!(
            debate_failure_warning(&err),
            "A model reached its usage limit; continuing with the remaining debate where possible"
        );
        let emitted = AtomicBool::new(false);
        assert!(claim_failure_warning(&emitted));
        assert!(!claim_failure_warning(&emitted));
    }

    /// Distinct presets that sanitize to the same slug must still produce distinct
    /// trajectory stems — otherwise two lanes write the same `.jsonl` and their records
    /// interleave unattributably.
    #[test]
    fn lane_stems_stay_unique_when_preset_names_sanitize_identically() {
        let a = preset("a/b");
        let b = preset("a?b");
        let (a_review, a_validate) = lane_session_stems(0, Some(&a));
        let (b_review, b_validate) = lane_session_stems(1, Some(&b));
        assert_ne!(a_review, b_review);
        assert_ne!(a_validate, b_validate);
        assert_ne!(a_review, a_validate);
    }

    /// Topic (`ask`) keeps the pre-preset stems exactly — its trajectory layout is a
    /// compatibility surface for `reflect`.
    #[test]
    fn topic_lane_keeps_legacy_stems() {
        assert_eq!(
            lane_session_stems(0, None),
            ("review".to_string(), "validate".to_string())
        );
    }

    /// The meta input must state each surviving lane's preset and convergence status —
    /// unresolved disagreement has to arrive labelled as such.
    #[test]
    fn lane_sections_carry_preset_names_and_convergence_state() {
        let mut converged_lane = lane(Some("security"), &[("Reviewer", 1, "finding X")]);
        converged_lane.converged = true;
        let open_lane = lane(Some("tone"), &[("Validator", 1, "disputed Y")]);
        let sections = lane_sections(&[&converged_lane, &open_lane]);
        // every surviving lane's name and dialogue reach the synthesizer
        for needle in ["security", "tone", "finding X", "disputed Y"] {
            assert!(sections.contains(needle), "missing {needle}");
        }
        // convergence state must be represented, whatever its wording: the same lane
        // renders differently converged vs not
        let mut reopened = lane(Some("security"), &[("Reviewer", 1, "finding X")]);
        reopened.converged = false;
        assert_ne!(
            lane_sections(&[&converged_lane]),
            lane_sections(&[&reopened])
        );
    }

    /// A converged lane reaches the meta pruned to its final round — the closing verdicts
    /// are self-contained by prompt contract, so earlier rounds are superseded noise. A
    /// contested lane and the human transcript keep the full dialogue.
    #[test]
    fn converged_lanes_reach_meta_pruned_to_final_round() {
        let verdicts: &[(&str, usize, &str)] = &[
            ("Reviewer", 1, "ROUND-ONE-CLAIM"),
            ("Validator", 1, "ROUND-ONE-CHALLENGE"),
            ("Reviewer", 2, "ROUND-TWO-FINDINGS"),
            ("Validator", 2, "ROUND-TWO-CONFIRMATION"),
        ];
        let mut converged = lane(Some("security"), verdicts);
        converged.converged = true;
        converged.final_round = 2;
        let section = lane_sections(&[&converged]);
        assert!(section.contains("ROUND-TWO-FINDINGS"));
        assert!(section.contains("ROUND-TWO-CONFIRMATION"));
        assert!(!section.contains("ROUND-ONE-CLAIM"));

        let mut open = lane(Some("security"), verdicts);
        open.final_round = 2;
        assert!(lane_sections(&[&open]).contains("ROUND-ONE-CLAIM"));

        // a lane with a failed turn in round 1 that converges in round 2 still has partial
        // evidence — its full trail must reach the meta-reviewer
        let mut converged_degraded = lane(Some("security"), verdicts);
        converged_degraded.converged = true;
        converged_degraded.degraded = true;
        converged_degraded.final_round = 2;
        assert!(lane_sections(&[&converged_degraded]).contains("ROUND-ONE-CLAIM"));

        let transcript = render_lane_transcript_section(&converged, 2);
        assert!(transcript.contains("ROUND-ONE-CLAIM"));
    }

    /// Dead lanes appear in the human transcript (flagged) but never in the meta input.
    #[test]
    fn dead_lanes_are_flagged_in_transcript_and_absent_from_meta_sections() {
        let mut dead = lane(Some("security"), &[("Reviewer", 1, "*Agent failed: boom*")]);
        dead.any_turn_succeeded = false;
        let alive = lane(Some("tone"), &[("Reviewer", 1, "real finding")]);

        let lanes = [dead, alive];
        let survivors = surviving(&lanes);
        assert_eq!(survivors.len(), 1);
        assert!(!lane_sections(&survivors).contains("security"));

        let section = render_lane_transcript_section(&lanes[0], 5);
        assert!(section.contains("Lane failed"));
    }

    /// The plan's meta-input contract is {preset, transcript, converged, degraded}: a lane
    /// whose dialogue is partial must arrive flagged, not as an ordinary transcript.
    #[test]
    fn degraded_lanes_are_marked_in_meta_sections() {
        // degraded state must be represented per lane, whatever its wording: an identical
        // lane renders differently degraded vs clean, and only the degraded lane changes
        let clean = lane(Some("security"), &[("Reviewer", 1, "raw text")]);
        let mut degraded = lane(Some("security"), &[("Reviewer", 1, "raw text")]);
        degraded.degraded = true;
        assert_ne!(lane_sections(&[&clean]), lane_sections(&[&degraded]));
    }

    #[test]
    fn persisted_verdict_status_uses_turn_outcome_not_rendered_text() {
        let mut outcome = lane(
            Some("correctness"),
            &[("Reviewer", 1, "*Agent failed: finding text*")],
        );
        outcome.verdicts.push(LaneVerdict {
            role: "Validator".to_string(),
            round: 1,
            text: "ordinary failure stub wording".to_string(),
            agent_failed: true,
        });

        let records = debate_verdict_records(&[outcome]);
        assert!(records[0].ok);
        assert!(!records[1].ok);
    }

    /// Topic filenames are unchanged; preset filenames append bounded sanitized slugs.
    #[test]
    fn transcript_stems_are_mode_appropriate_and_bounded() {
        assert_eq!(transcript_stem("debate", 42, None), "debate-42");

        let ps = vec![preset("security"), preset("ml-rigor")];
        assert_eq!(
            transcript_stem("review-debate", 42, Some(&ps)),
            "review-debate-42-security-ml-rigor"
        );

        let long: Vec<ReviewPreset> = (0..30).map(|i| preset(&format!("angle-{i}"))).collect();
        let name = transcript_stem("review-debate", 42, Some(&long));
        assert!(name.len() < 120, "unbounded filename: {name}");
    }

    #[test]
    fn transcripts_with_the_same_stem_never_share_a_file() {
        let stem = format!("nitpicker-transcript-test-{}", std::process::id());
        let first = write_transcript(&stem, "first").unwrap();
        let second = write_transcript(&stem, "second").unwrap();
        assert_ne!(first, second);
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "first");
        assert_eq!(std::fs::read_to_string(&second).unwrap(), "second");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&first).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        std::fs::remove_file(first).unwrap();
        std::fs::remove_file(second).unwrap();
    }
}
