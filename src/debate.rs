use crate::output::UsageReport;
pub use crate::prompts::DebateMode;
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nitpicker_agent::agent::{
    AgentConfig, AgentDepth, AgentProgress, MAX_CONCURRENT_LLM_CALLS, add_spawn_subagent_tool,
    run_agent,
};
use nitpicker_agent::config::Config;
use nitpicker_agent::llm::{Completion, LLMClientDyn, TokenUsage};
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
use std::sync::atomic::AtomicUsize;
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
    /// The agent finished without calling `submit_verdict`; `verdict` is its raw final text.
    used_fallback: bool,
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
                Set agree=true if you fully agree with the opponent's latest position (convergence)."
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
                        "description": "Set to true if you fully agree with opponent (convergence)"
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
            warn!(model = request.model, error = ?err, "debate agent failed");
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
                used_fallback: false,
            });
        }
    };
    let usage = result.usage;
    let stored = verdict_store
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    let used_fallback = stored.is_none();
    let verdict = stored.unwrap_or(DebateVerdict {
        text: result.text,
        agree: false,
    });
    Ok(DebateTurnResult {
        verdict,
        turns: result.turns,
        tool_calls: result.tool_calls,
        subagents_spawned: result.subagents_spawned,
        usage,
        agent_failed: false,
        used_fallback,
    })
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

/// Everything a lane's turns need that is identical for both sides. One instance per lane:
/// most fields borrow run-level state, the last three are the lane's own.
struct RoundEnv<'a> {
    mp: &'a MultiProgress,
    done_style: &'a ProgressStyle,
    skin: &'a MadSkin,
    repo: &'a Path,
    topic: &'a str,
    project_context: &'a str,
    session_logger: Option<&'a SessionLogger>,
    max_turns: usize,
    verbose: bool,
    stdout_ok: bool,
    llm_semaphore: &'a Arc<tokio::sync::Semaphore>,
    /// Preset name shown in progress prefixes; `None` for Topic (and its single lane).
    lane_tag: Option<&'a str>,
    subagent_system_prompt: Option<&'a str>,
    /// Print verdicts as turns finish (single lane only) — concurrent lanes buffer instead,
    /// since their interleaved output would be unattributable.
    live_output: bool,
}

/// Run one side's turn for a round: spinner up, agent run, spinner down, optional render.
async fn run_debate_side(
    side: &DebateSide<'_>,
    env: &RoundEnv<'_>,
    verdicts: &[(String, usize, String)],
    round: usize,
) -> Result<DebateTurnResult> {
    let (pb, _) = make_spinner(env.mp);
    let prefix = match env.lane_tag {
        Some(tag) => format!("{tag} · {}", colored_role_stderr(side.role)),
        None => colored_role_stderr(side.role),
    };
    pb.set_prefix(prefix);
    pb.set_message(crate::progress::bar_message(format!(
        "round {round} — debating…"
    )));
    let sub_pb = make_sub_spinner(env.mp, &pb);
    let msg = build_turn_message(env.topic, verdicts, round, side.role);
    let start = std::time::Instant::now();
    let progress_pb = pb.clone();
    let progress_sub_pb = sub_pb.clone();
    let progress = (!env.verbose).then_some(Arc::new(move |progress: AgentProgress| {
        progress_pb.set_message(crate::progress::bar_message(format!(
            "round {round} — debating… ({} turns, {} tool calls, {} subagents)",
            progress.turns, progress.tool_calls, progress.subagents_spawned
        )));
        progress_sub_pb.set_message(crate::progress::detail_message(
            "    ↳ ",
            progress.last_subagent.as_deref(),
        ));
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
        progress,
        project_context: Some(env.project_context.to_string()),
        session_writer: env
            .session_logger
            .map(|logger| logger.child(format!("{session_agent}.jsonl"))),
        session_agent,
    })
    .await?;

    let elapsed = start.elapsed().as_secs();
    sub_pb.finish_and_clear();
    pb.set_style(env.done_style.clone());
    pb.finish_with_message(crate::progress::bar_message(format!(
        "✓ round {round} ({} turns, {} tool calls, {} subagents, {}, {} out, {elapsed}s)",
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

fn make_sub_spinner(mp: &MultiProgress, pb: &ProgressBar) -> ProgressBar {
    let sub = mp.insert_after(pb, ProgressBar::new_spinner());
    sub.set_style(ProgressStyle::with_template("{msg}").unwrap());
    sub
}

pub struct DebateOptions {
    pub max_rounds: usize,
    pub max_turns: usize,
    pub verbose: bool,
    pub mode: DebateMode,
    pub alloy: bool,
    pub format: crate::output::OutputFormat,
}

pub struct DebateOutcome {
    pub report: String,
    pub transcript_path: std::path::PathBuf,
    pub usage: UsageReport,
    /// At least one turn failed or fell back to raw text; the report is synthesized from a
    /// partial dialogue. Surfaced as exit code 3 in the default-review/`ask`/`pr` CLI arms.
    pub degraded: bool,
    /// Presets whose lane survived (≥1 turn ran) — the angles the meta synthesis actually
    /// covered, where the resolved list documents only the selection. `None` for Topic.
    pub covered_presets: Option<Vec<String>>,
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
    opts: DebateOptions,
    presets: Option<&[crate::presets::ReviewPreset]>,
) -> Result<DebateOutcome> {
    let DebateOptions {
        max_rounds,
        max_turns,
        verbose,
        mode,
        alloy,
        format,
    } = opts;
    match (&mode, presets) {
        (DebateMode::Review(_), Some(_)) | (DebateMode::Topic, None) => {}
        (DebateMode::Review(_), None) | (DebateMode::Topic, Some(_)) => {
            unreachable!("Review debates take the resolved presets; Topic takes none")
        }
    }
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
    let gemini_proxy = crate::proxy::GeminiProxy::maybe_start(config).await?;
    let proxy_url = gemini_proxy.url();

    let actor_client: Arc<dyn LLMClientDyn>;
    let critic_client: Arc<dyn LLMClientDyn>;
    let actor_label: ModelLabel;
    let critic_label: ModelLabel;
    let actor_compact_threshold: Option<u64>;
    let critic_compact_threshold: Option<u64>;
    // In alloy mode the client pools every reviewer, so these come from the same two slots the
    // roles are otherwise pinned to — the existing convention for per-side settings.
    let actor_max_tokens = actor_cfg.max_tokens;
    let critic_max_tokens = critic_cfg.max_tokens;

    if alloy {
        let mut slots = Vec::new();
        for r in &config.reviewer {
            slots.push(nitpicker_agent::llm::AlloySlot {
                client: build_reviewer_client(r, proxy_url.as_deref())?,
                model: r.model.clone(),
                max_tokens: r.max_tokens,
            });
        }
        let shared: Arc<dyn LLMClientDyn> =
            Arc::new(nitpicker_agent::llm::AlloyClient::new(slots)?);
        actor_client = Arc::clone(&shared);
        critic_client = shared;
        let label = ModelLabel::alloy(config.reviewer.iter().map(|r| r.model.as_str()));
        actor_label = ModelLabel {
            alias: label.alias.clone(),
            full: label.full.clone(),
        };
        critic_label = label;
        actor_compact_threshold = config.reviewer_compact_threshold(actor_cfg);
        critic_compact_threshold = config.reviewer_compact_threshold(critic_cfg);
    } else {
        actor_client = build_reviewer_client(actor_cfg, proxy_url.as_deref())?;
        critic_client = build_reviewer_client(critic_cfg, proxy_url.as_deref())?;
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

    let agg_client: Arc<dyn LLMClientDyn> = build_aggregator_client(agg_cfg, proxy_url.as_deref())?;

    let actor_role = mode.actor_role();
    let critic_role = mode.critic_role();
    // one in-flight-LLM cap for the whole run: concurrent lanes and their subagents share
    // it, so lane count scales wall-clock breadth without multiplying provider pressure
    let llm_semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_LLM_CALLS));

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

    // One lane per preset for Review; Topic (`ask`) is the degenerate single `None` lane and
    // keeps its exact pre-preset behavior, live verbose output included.
    let lane_presets: Vec<Option<&crate::presets::ReviewPreset>> = match presets {
        Some(presets) => presets.iter().map(Some).collect(),
        None => vec![None],
    };
    let live_output = lane_presets.len() == 1;

    let lane_futures = lane_presets.iter().enumerate().map(|(lane_index, preset)| {
        let actor_client = Arc::clone(&actor_client);
        let critic_client = Arc::clone(&critic_client);
        let llm_semaphore = &llm_semaphore;
        let mp = &mp;
        let done_style = &done_style;
        let skin = &skin;
        let project_context = &project_context;
        let session_logger = session_logger.as_ref();
        let actor_alias = &actor_label.alias;
        let critic_alias = &critic_label.alias;
        let mode = &mode;
        async move {
            let actor_system = mode.actor_system(*preset);
            let critic_system = mode.critic_system(*preset);
            let subagent_prompt = preset.map(crate::prompts::preset_subagent_prompt);
            let (actor_stem, critic_stem) = lane_session_stems(lane_index, *preset);
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
                mp,
                done_style,
                skin,
                repo,
                topic: prompt,
                project_context,
                session_logger,
                max_turns,
                verbose,
                stdout_ok,
                llm_semaphore,
                lane_tag: preset.map(|p| p.name.as_str()),
                subagent_system_prompt: subagent_prompt.as_deref(),
                live_output,
            };

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
                lane.degraded |= actor_turn.agent_failed || actor_turn.used_fallback;
                lane.verdicts
                    .push((actor_role.to_string(), round, actor_turn.verdict.text));

                let critic_turn = run_debate_side(&critic, &env, &lane.verdicts, round).await?;
                lane.usage
                    .add(critic_turn.usage, critic_turn.subagents_spawned);
                lane.any_turn_succeeded |= !critic_turn.agent_failed;
                lane.degraded |= critic_turn.agent_failed || critic_turn.used_fallback;
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
                    preset_name: lane_presets[lane_index].map(|p| p.name.clone()),
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
        eyre::bail!("all debate turns failed; refusing to synthesize a verdict");
    }

    // meta-review: non-agentic single completion over the surviving lanes' dialogue
    let survivors = surviving(&lanes);
    let meta_prompt = match presets {
        // Topic (`ask`): single lane, prompt shape unchanged
        None => {
            let note = match survivors[0].converged {
                true => {
                    "The debate converged; only the final round is shown — it supersedes the \
                     earlier dialogue.\n\n"
                }
                false => "",
            };
            format!(
                "The following is a debate about: {prompt}\n\n{note}{dialogue}\n\n---\n{instruction}",
                dialogue = lane_dialogue(survivors[0]),
                instruction = mode.meta_instruction(),
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
                 - A lane marked degraded had a turn fail or end without a verdict; its dialogue \
                 is partial.\n\
                 {instruction}",
                roster = crate::prompts::preset_roster(&surviving_presets),
                sections = lane_sections(&survivors),
                instruction = mode.meta_instruction(),
            )
        }
    };
    let meta_completion = Completion {
        model: agg_cfg.model.clone(),
        prompt: Message::user(meta_prompt),
        preamble: Some(mode.meta_preamble().to_string()),
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
    let meta_response: nitpicker_agent::llm::CompletionResponse = agg_client
        .completion(meta_completion)
        .await
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
        })?;
    usage.add(meta_response.usage, 0);
    let meta_text = meta_response.text();
    pb.set_style(done_style);
    pb.finish_with_message("✓ done");
    if let Some(logger) = &session_logger {
        // the scalar rounds/converged fields stay populated whenever there is exactly one
        // lane (Topic always; single-preset review) — `reflect` renders only the scalars,
        // so clearing them for a one-lane run would lose metadata it has today
        let (rounds, converged) = match lanes.len() {
            1 => (Some(lanes[0].final_round), Some(lanes[0].converged)),
            _ => (None, None),
        };
        let record = AggregationRecord {
            kind: "aggregation".to_string(),
            model: agg_cfg.model.clone(),
            text: meta_text.clone(),
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
        logger.write_aggregation(&record).await?;
    }
    // write transcript file
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let transcript_path = std::env::temp_dir().join(transcript_filename(mode.label(), ts, presets));
    let now = chrono::Local::now();

    let label = mode.label();
    let mut transcript = match presets {
        // Topic (`ask`): single lane, transcript shape unchanged
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
                agg_cfg.model,
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
                agg_cfg.model,
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

    // only the verbose path surfaces this file; skip the write otherwise so a
    // long-running server doesn't litter the temp dir on every review.
    if verbose {
        tokio::fs::write(&transcript_path, &transcript).await?;
    }

    Ok(DebateOutcome {
        report: meta_text,
        transcript_path,
        usage,
        degraded,
        covered_presets: presets.map(|_| {
            surviving(&lanes)
                .iter()
                .filter_map(|lane| lane.preset_name.clone())
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

/// A lane survives — and reaches synthesis — iff at least one of its turns really ran.
fn surviving(lanes: &[DebateLaneOutcome]) -> Vec<&DebateLaneOutcome> {
    lanes
        .iter()
        .filter(|lane| lane.any_turn_succeeded)
        .collect()
}

/// Dialogue as the meta-review sees it. A converged lane is pruned to its final round:
/// verdicts are self-contained by prompt contract and the agreeing critic restates every
/// confirmed finding, so earlier rounds are superseded chronology — exactly the material
/// a synthesizer misreads into withdrawn-claim narration. Contested (non-converged) and
/// degraded lanes keep their full trail, and the human transcript always does.
fn lane_dialogue(lane: &DebateLaneOutcome) -> String {
    lane.verdicts
        .iter()
        .filter(|(_, rnd, _)| !lane.converged || *rnd == lane.final_round)
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
            let convergence = match lane.converged {
                true => format!(
                    "converged at round {} (earlier rounds superseded and omitted)",
                    lane.final_round
                ),
                false => format!("no convergence after {} round(s)", lane.final_round),
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
