use crate::output::UsageReport;
use crate::prompts::RunTask;
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nitpicker_agent::agent::{
    AgentConfig, AgentDepth, AgentProgress, MAX_CONCURRENT_LLM_CALLS, add_spawn_subagent_tool,
    run_agent,
};
use nitpicker_agent::config::Config;
use nitpicker_agent::llm::{Completion, LLMClientDyn, TokenUsage, is_operational_limit_error};
use nitpicker_agent::provider::{build_aggregator_client, build_reviewer_client};
use nitpicker_agent::session::{
    AggregationRecord, LaneRecord, SessionLogger, SessionWriter, sanitize_path_component,
};
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
use tracing::info;
use tracing::warn;

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
                .ok_or_else(|| eyre::eyre!("missing verdict"))?
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

async fn run_debate_turn(request: DebateTurnRequest<'_>) -> Result<DebateTurnResult> {
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

    let result = match run_agent(
        config,
        request.initial_message,
        &tools_map,
        request.work_dir,
    )
    .await
    {
        Ok(r) => r,
        Err(err) => {
            tracing::debug!(model = request.model, error = ?err, "debate agent failed");
            if claim_failure_warning(&request.failure_warning_emitted) {
                warn!("{}", debate_failure_warning(&err));
            }
            return Ok(DebateTurnResult {
                verdict: DebateVerdict {
                    text: format!("*Agent failed: {err:#}*"),
                    agree: false,
                },
                turns: 0,
                tool_calls: 0,
                subagents_spawned: 0,
                usage: TokenUsage::default(),
                agent_failed: true,
            });
        }
    };
    let usage = result.usage;
    let stored = verdict_store
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    let verdict = stored.ok_or_else(|| {
        eyre::eyre!("debate agent completed without calling required submit_verdict tool")
    })?;
    Ok(DebateTurnResult {
        verdict,
        turns: result.turns,
        tool_calls: result.tool_calls,
        subagents_spawned: result.subagents_spawned,
        usage,
        agent_failed: false,
    })
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
    role: &'a str,
    client: Arc<dyn LLMClientDyn>,
    compact_threshold: Option<u64>,
    max_tokens: Option<u64>,
    model: &'a str,
    system_prompt: &'a str,
    /// Trajectory filename stem; the round number is appended.
    session_stem: &'a str,
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
    subagent_system_prompt: Option<&'a str>,
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
    verdicts: &[(String, usize, String)],
    round: usize,
) -> Result<DebateTurnResult> {
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
    let result = run_debate_turn(DebateTurnRequest {
        client: Arc::clone(&side.client),
        compact_threshold: side.compact_threshold,
        max_tokens: side.max_tokens,
        model: side.model,
        system_prompt: side.system_prompt,
        subagent_system_prompt: env.subagent_system_prompt.map(str::to_string),
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
    .await?;

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
    Ok(result)
}

fn build_turn_message(
    topic: &str,
    verdicts: &[(String, usize, String)],
    round: usize,
    role: &str,
) -> String {
    let mut msg = format!("Topic: {topic}\n");
    if verdicts.is_empty() {
        msg.push_str("\nNo prior dialogue yet.\n");
    } else {
        msg.push_str("\nDialogue so far:\n");
        for (label, rnd, text) in verdicts {
            msg.push_str(&format!("\n### {label} (Round {rnd})\n{text}\n"));
        }
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

fn make_spinner(mp: &MultiProgress) -> (ProgressBar, ProgressStyle) {
    let spinner_style = ProgressStyle::with_template("{spinner:.cyan} {prefix:<12} {msg}")
        .unwrap()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", ""]);
    let pb = mp.add(ProgressBar::new_spinner());
    pb.set_style(spinner_style.clone());
    pb.enable_steady_tick(Duration::from_millis(80));
    (pb, spinner_style)
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
    /// (role_label, round_number, verdict_text)
    verdicts: Vec<(String, usize, String)>,
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
    if config.reviewer.len() < 2 {
        eyre::bail!(
            "debate requires at least 2 reviewers in config (actor = reviewer[0], critic = reviewer[1])"
        );
    }

    let actor_cfg = &config.reviewer[0];
    let critic_cfg = &config.reviewer[1];
    let agg_cfg = &config.aggregator;

    // proxy handle stays bound for the function so its local server outlives the debate;
    // only its base URL is threaded into the client builders.
    let gemini_proxy = crate::proxy::GeminiProxy::maybe_start(config).await;
    let proxy_url = gemini_proxy.url();

    let actor_client: Arc<dyn LLMClientDyn>;
    let critic_client: Arc<dyn LLMClientDyn>;
    let actor_label: ModelLabel;
    let critic_label: ModelLabel;
    let actor_compact_threshold: Option<u64>;
    let critic_compact_threshold: Option<u64>;
    // Output caps stay bound to the two logical roles; the routing clients replace them with the
    // selected route's cap at the completion boundary.
    let actor_max_tokens = actor_cfg.max_tokens;
    let critic_max_tokens = critic_cfg.max_tokens;

    let reviewer_pool = (alloy || fallback)
        .then(|| crate::review::build_reviewer_pool(config, &gemini_proxy, proxy_url.as_deref()));

    if alloy {
        let pool = reviewer_pool.as_ref().expect("alloy builds reviewer pool");
        let shared = crate::review::alloy_client(pool, fallback)?;
        actor_client = Arc::clone(&shared);
        critic_client = shared;
        let label = ModelLabel::alloy(config.reviewer.iter().map(|r| r.model.as_str()));
        actor_label = ModelLabel {
            alias: label.alias.clone(),
            full: label.full.clone(),
        };
        critic_label = label;
        let threshold = crate::review::reviewer_pool_compact_threshold(config, pool);
        actor_compact_threshold = threshold;
        critic_compact_threshold = threshold;
    } else if fallback {
        let pool = reviewer_pool
            .as_ref()
            .expect("fallback builds reviewer pool");
        actor_client = crate::review::reviewer_client(pool, 0, true)?;
        critic_client = crate::review::reviewer_client(pool, 1, true)?;
        actor_label = ModelLabel::plain(&actor_cfg.model);
        critic_label = ModelLabel::plain(&critic_cfg.model);
        let threshold = crate::review::reviewer_pool_compact_threshold(config, pool);
        actor_compact_threshold = threshold;
        critic_compact_threshold = threshold;
    } else {
        actor_client =
            gemini_proxy.annotate(build_reviewer_client(actor_cfg, proxy_url.as_deref()))?;
        critic_client =
            gemini_proxy.annotate(build_reviewer_client(critic_cfg, proxy_url.as_deref()))?;
        actor_label = ModelLabel::plain(&actor_cfg.model);
        critic_label = ModelLabel::plain(&critic_cfg.model);
        actor_compact_threshold = config.reviewer_compact_threshold(actor_cfg);
        critic_compact_threshold = config.reviewer_compact_threshold(critic_cfg);
    }
    let session_logger = SessionLogger::maybe_new(config.log_trajectories())?;
    if let Some(logger) = &session_logger {
        info!(path = %logger.root().display(), "trajectory logging enabled");
    }

    let project_context = crate::context::build_context(repo).await;

    let agg_client: Arc<dyn LLMClientDyn> = match &reviewer_pool {
        Some(pool) if fallback => crate::review::aggregator_client(
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

    let mp = Arc::new(MultiProgress::new());
    if verbose {
        mp.set_draw_target(ProgressDrawTarget::hidden());
    }
    let _progress_guard = (!verbose && crate::progress::stderr_is_terminal())
        .then(|| crate::progress::set_active_progress(&mp));

    // cast lines show which models are participating in interactive text mode, but piped/json
    // stdout stays machine-readable/final-report-only.
    if stdout_ok && stdout_is_terminal() {
        if alloy {
            print_cast_line(actor_role, &actor_label.full);
            print_cast_line(critic_role, &critic_label.full);
        } else {
            print_cast_line(
                actor_role,
                &format!("{} · {}", actor_cfg.name, actor_label.full),
            );
            print_cast_line(
                critic_role,
                &format!("{} · {}", critic_cfg.name, critic_label.full),
            );
        }
        print_cast_line("Meta-review", &agg_cfg.model);
        if let Some(presets) = presets {
            let names: Vec<&str> = presets.iter().map(|p| p.name.as_str()).collect();
            print_cast_line("Presets", &names.join(", "));
        }
        println!();
    }

    // One lane per preset for Review; Ask is the degenerate single unscoped lane and
    // keeps its exact pre-preset behavior, live verbose output included.
    let live_output = lane_tasks.len() == 1;

    let lane_futures = lane_tasks
        .iter()
        .enumerate()
        .map(|(lane_index, lane_task)| {
            let actor_client = Arc::clone(&actor_client);
            let critic_client = Arc::clone(&critic_client);
            let llm_semaphore = &llm_semaphore;
            let failure_warning_emitted = &failure_warning_emitted;
            let mp = &mp;
            let done_style = &done_style;
            let skin = &skin;
            let project_context = &project_context;
            let session_logger = session_logger.as_ref();
            let actor_alias = &actor_label.alias;
            let critic_alias = &critic_label.alias;
            async move {
                let preset = lane_task.preset();
                let (lane_progress, _) = make_spinner(mp);
                lane_progress.set_prefix(preset.map_or("debate", |p| p.name.as_str()).to_string());
                lane_progress.set_message(crate::progress::bar_message("waiting…"));
                let actor_system = lane_task.actor_system();
                let critic_system = lane_task.critic_system();
                let subagent_prompt = lane_task.subagent_prompt();
                let (actor_stem, critic_stem) = lane_session_stems(lane_index, preset);
                let actor = DebateSide {
                    role: actor_role,
                    client: actor_client,
                    compact_threshold: actor_compact_threshold,
                    max_tokens: actor_max_tokens,
                    model: actor_alias,
                    system_prompt: &actor_system,
                    session_stem: &actor_stem,
                };
                let critic = DebateSide {
                    role: critic_role,
                    client: critic_client,
                    compact_threshold: critic_compact_threshold,
                    max_tokens: critic_max_tokens,
                    model: critic_alias,
                    system_prompt: &critic_system,
                    session_stem: &critic_stem,
                };
                let env = RoundEnv {
                    skin,
                    repo,
                    topic: prompt,
                    project_context,
                    session_logger,
                    max_turns,
                    verbose,
                    stdout_ok,
                    llm_semaphore,
                    failure_warning_emitted,
                    subagent_system_prompt: subagent_prompt.as_deref(),
                    live_output,
                    lane_progress,
                };

                let started = std::time::Instant::now();
                let mut lane = DebateLaneOutcome {
                    preset_name: preset.map(|p| p.name.clone()),
                    verdicts: Vec::new(),
                    converged: false,
                    final_round: 0,
                    any_turn_succeeded: false,
                    degraded: false,
                    usage: UsageReport::default(),
                };

                'debate: for round in 1..=max_rounds {
                    lane.final_round = round;

                    let actor_turn = run_debate_side(&actor, &env, &lane.verdicts, round).await?;
                    lane.usage
                        .add(actor_turn.usage, actor_turn.subagents_spawned);
                    lane.any_turn_succeeded |= !actor_turn.agent_failed;
                    lane.degraded |= actor_turn.agent_failed;
                    lane.verdicts
                        .push((actor_role.to_string(), round, actor_turn.verdict.text));

                    let critic_turn = run_debate_side(&critic, &env, &lane.verdicts, round).await?;
                    lane.usage
                        .add(critic_turn.usage, critic_turn.subagents_spawned);
                    lane.any_turn_succeeded |= !critic_turn.agent_failed;
                    lane.degraded |= critic_turn.agent_failed;
                    // Convergence requires a real agreement: a critic that agrees with a failed
                    // actor's `*Agent failed*` stub (or a failed critic, whose verdict defaults to
                    // agree=false) must not end the debate early.
                    let agreed = critic_turn.verdict.agree
                        && !actor_turn.agent_failed
                        && !critic_turn.agent_failed;
                    lane.verdicts
                        .push((critic_role.to_string(), round, critic_turn.verdict.text));

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
                Ok::<DebateLaneOutcome, eyre::Report>(lane)
            }
        });
    let lane_results = futures::future::join_all(lane_futures).await;

    let mut lanes: Vec<DebateLaneOutcome> = Vec::new();
    for (lane_index, result) in lane_results.into_iter().enumerate() {
        match result {
            Ok(lane) => lanes.push(lane),
            // a lane-level error (nothing inside the loop should produce one — turn failures
            // fold into stubs) counts as a dead, degraded lane rather than aborting siblings
            Err(err) => {
                warn!(lane = lane_index, error = ?err, "debate lane failed");
                lanes.push(DebateLaneOutcome {
                    preset_name: lane_tasks[lane_index].preset().map(|p| p.name.clone()),
                    verdicts: Vec::new(),
                    converged: false,
                    final_round: 0,
                    any_turn_succeeded: false,
                    degraded: true,
                    usage: UsageReport::default(),
                });
            }
        }
    }

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
            for (label, rnd, text) in &lane.verdicts {
                println!("\n{label} (Round {rnd}):");
                skin.print_text(text);
            }
        }
    }

    // Every turn failed everywhere (provider down, bad config): the dialogue is nothing but
    // failure stubs, so synthesizing a meta-verdict would fabricate a confident review from
    // errors. Surface the failure instead — `run_pr` maps this to a `status: "error"`
    // envelope and posts no comment. A lane with no successful turn is likewise omitted from
    // synthesis below: its stubs are execution noise, not review evidence.
    if surviving(&lanes).is_empty() {
        let err = eyre::eyre!("all debate turns failed; refusing to synthesize a verdict");
        // persist the lane outcomes before bailing: a run where every turn failed is the one
        // that most needs a durable record of what was attempted. The scalars follow the
        // same single-lane derivation as the success and meta-failure paths, so a failed
        // `ask` run's record isn't shaped differently from a successful one.
        if let Some(logger) = &session_logger {
            let (rounds, converged) = single_lane_scalars(&lanes);
            let record = AggregationRecord {
                kind: "aggregation".to_string(),
                model: agg_cfg.model.clone(),
                text: String::new(),
                error: Some(crate::review::bounded_error_string(&err)),
                rounds,
                converged,
                presets: presets.map(|ps| ps.iter().map(|p| p.name.clone()).collect()),
                lanes: presets.map(|_| {
                    lanes
                        .iter()
                        .map(|lane| LaneRecord {
                            preset: lane.preset_name.clone().unwrap_or_default(),
                            rounds: lane.final_round,
                            converged: lane.converged,
                            degraded: lane.degraded,
                        })
                        .collect()
                }),
                jobs: None,
            };
            match logger.write_aggregation(&record).await {
                Ok(()) => {}
                Err(write_err) => {
                    warn!(error = ?write_err, "failed to persist all-turns-failed record");
                }
            }
        }
        return Err(err);
    }

    // meta-review: non-agentic single completion over the surviving lanes' dialogue
    let survivors = surviving(&lanes);
    let meta_prompt = match presets {
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
            format!(
                "The following are independent review debates about the same target, one per review angle.\n\
                 Target: {prompt}\n\n{roster}\n\n{sections}\n\n---\n\
                 Notes:\n\
                 - \"*Agent failed: …*\" markers are execution errors kept for chronology; they are \
                 not review evidence and not agreement.\n\
                 - A lane that ended without convergence carries unresolved disagreement — weigh it \
                 by the evidence, do not read it as agreement.\n\
                 - A lane marked degraded had a turn fail; its dialogue is partial.\n\
                 {instruction}",
                roster = crate::prompts::preset_roster(&surviving_presets),
                sections = lane_sections(&survivors),
                instruction = task.meta_instruction(),
            )
        }
    };
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
    let (pb, _) = make_spinner(&mp);
    pb.set_prefix(colored_role_stderr("Meta-review"));
    pb.set_message(crate::progress::bar_message("synthesizing…"));
    // preset runs get the count/context wrapping; Topic propagates the provider error
    // untouched, as it did before lanes existed
    let meta_result: eyre::Result<nitpicker_agent::llm::CompletionResponse> = agg_client
        .completion(meta_completion)
        .await
        .and_then(|response| {
            crate::review::validate_synthesis_response(&response, "meta-review")?;
            Ok(response)
        })
        .map_err(|err| match presets {
            Some(presets) => crate::presets::synthesis_failure(
                err,
                format!(
                    "meta-review failed over {} surviving lane(s) across {} preset(s)",
                    survivors.len(),
                    presets.len()
                ),
            ),
            None => err,
        });
    pb.set_style(done_style);
    // Lane metadata travels on both outcomes: a meta failure still persists the per-lane
    // record (the durable trace of what ran), flagged with `error` and an empty `text` so
    // consumers (reflect) don't render it as a verdict.
    let (rounds, converged) = single_lane_scalars(&lanes);
    let preset_names: Option<Vec<String>> =
        presets.map(|ps| ps.iter().map(|p| p.name.clone()).collect());
    let lane_records: Option<Vec<LaneRecord>> = presets.map(|_| {
        lanes
            .iter()
            .map(|lane| LaneRecord {
                preset: lane.preset_name.clone().unwrap_or_default(),
                rounds: lane.final_round,
                converged: lane.converged,
                degraded: lane.degraded,
            })
            .collect()
    });
    let meta_response = match meta_result {
        Ok(response) => {
            pb.finish_with_message("✓ done");
            response
        }
        Err(err) => {
            pb.finish_with_message(crate::progress::bar_message("✗ synthesis failed"));
            if let Some(logger) = &session_logger {
                let record = AggregationRecord {
                    kind: "aggregation".to_string(),
                    model: agg_cfg.model.clone(),
                    text: String::new(),
                    error: Some(crate::review::bounded_error_string(&err)),
                    rounds,
                    converged,
                    presets: preset_names,
                    lanes: lane_records,
                    jobs: None,
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
    usage.add(meta_response.usage, 0);
    let meta_text = meta_response.text();
    let meta_model = crate::review::synthesis_model(&meta_response, &agg_cfg.model);
    if let Some(logger) = &session_logger {
        let record = AggregationRecord {
            kind: "aggregation".to_string(),
            model: meta_model.clone(),
            text: meta_text.clone(),
            error: None,
            rounds,
            converged,
            presets: preset_names,
            lanes: lane_records,
            jobs: None,
        };
        logger.write_aggregation(&record).await?;
    }
    // The transcript is a verbose-only debugging artifact. Keep both rendering and writing
    // behind the flag: multi-lane verdicts can be large, and non-verbose callers ignore the path.
    let transcript_path = if verbose {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let transcript_path =
            std::env::temp_dir().join(transcript_filename(task.label(), ts, presets));
        let now = chrono::Local::now();
        let label = task.label();
        let mut transcript = match presets {
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
                    actor_label.full,
                    critic_label.full,
                    meta_model,
                    now.format("%Y-%m-%d %H:%M:%S"),
                    lane.final_round,
                );
                for (label, rnd, text) in &lane.verdicts {
                    transcript.push_str(&format!("## {label} — Round {rnd}\n\n{text}\n\n"));
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
                    actor_label.full,
                    critic_label.full,
                    meta_model,
                    now.format("%Y-%m-%d %H:%M:%S"),
                    lanes.len(),
                );
                for lane in &lanes {
                    transcript.push_str(&render_lane_transcript_section(lane, max_rounds));
                }
                transcript
            }
        };
        transcript.push_str(&format!("---\n\n## Meta-review\n\n{meta_text}\n"));
        tokio::fs::write(&transcript_path, &transcript).await?;
        transcript_path
    } else {
        PathBuf::new()
    };

    Ok(DebateOutcome {
        report: meta_text,
        transcript_path,
        usage,
        degraded,
        coverage: presets.map(|ps| {
            ps.iter()
                .map(|p| {
                    let survived = lanes.iter().any(|lane| {
                        lane.preset_name.as_deref() == Some(p.name.as_str())
                            && lane.any_turn_succeeded
                    });
                    crate::output::PresetCoverage {
                        preset: p.name.clone(),
                        attempted: 1,
                        succeeded: usize::from(survived),
                    }
                })
                .collect()
        }),
    })
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
        .filter(|(_, rnd, _)| !lane_pruned_to_final_round(lane) || *rnd == lane.final_round)
        .map(|(label, rnd, text)| format!("### {label} (Round {rnd})\n{text}"))
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
fn transcript_filename(
    label: &str,
    ts: u64,
    presets: Option<&[crate::presets::ReviewPreset]>,
) -> String {
    match presets {
        None => format!("{label}-{ts}.md"),
        Some(presets) => {
            let mut slugs = presets
                .iter()
                .map(|p| sanitize_path_component(&p.name))
                .collect::<Vec<_>>()
                .join("-");
            // sanitize output is ASCII, so a byte truncate cannot split a char
            const MAX_SLUG_LEN: usize = 80;
            slugs.truncate(MAX_SLUG_LEN);
            format!("{label}-{ts}-{slugs}.md")
        }
    }
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
    for (label, rnd, text) in &lane.verdicts {
        section.push_str(&format!("### {label} — Round {rnd}\n\n{text}\n\n"));
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
                .map(|(l, r, t)| (l.to_string(), *r, t.to_string()))
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

    #[test]
    fn operational_failure_warning_is_concise_and_emitted_once() {
        let err = eyre::eyre!(
            "403 Forbidden: You've reached your usage limit for this billing cycle. Your quota \
             will be refreshed in the next cycle"
        );
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

    /// Topic filenames are unchanged; preset filenames append bounded sanitized slugs.
    #[test]
    fn transcript_filenames_are_mode_appropriate_and_bounded() {
        assert_eq!(transcript_filename("debate", 42, None), "debate-42.md");

        let ps = vec![preset("security"), preset("ml-rigor")];
        assert_eq!(
            transcript_filename("review-debate", 42, Some(&ps)),
            "review-debate-42-security-ml-rigor.md"
        );

        let long: Vec<ReviewPreset> = (0..30).map(|i| preset(&format!("angle-{i}"))).collect();
        let name = transcript_filename("review-debate", 42, Some(&long));
        assert!(name.len() < 120, "unbounded filename: {name}");
    }
}
