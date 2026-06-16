use nitpicker_agent::agent::{
    AgentConfig, AgentDepth, AgentProgress, MAX_CONCURRENT_LLM_CALLS, add_spawn_subagent_tool,
    run_agent,
};
use nitpicker_agent::config::{Config, ReviewerConfig};
use nitpicker_agent::llm::{Completion, LLMClientDyn, TokenUsage};
use crate::output::UsageReport;
pub use crate::prompts::DebateMode;
#[cfg(feature = "antigravity")]
use nitpicker_agent::provider::config_needs_gemini_proxy;
use nitpicker_agent::provider::{build_aggregator_client, build_reviewer_client};
use nitpicker_agent::session::{AggregationRecord, SessionLogger, SessionWriter};
use nitpicker_agent::tools::{Tool, all_tools};
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
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
    model: &'a str,
    system_prompt: &'a str,
    initial_message: &'a str,
    max_turns: usize,
    work_dir: &'a Path,
    progress: Option<Arc<dyn Fn(AgentProgress) + Send + Sync>>,
    project_context: Option<String>,
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
        session_agent: "root".to_string(),
        model: request.model.to_string(),
        max_turns: request.max_turns,
        compact_threshold: request.compact_threshold,
        system_prompt: request.system_prompt.to_string(),
        subagent_system_prompt: None,
        client: request.client,
        depth: AgentDepth::TopLevel,
        terminal_tools: vec!["submit_verdict".to_string()],
        empty_response_nudge: Some(
            "Please proceed with your analysis and call submit_verdict when you are done."
                .to_string(),
        ),
        max_empty_responses: 3,
        subagent_counter,
        // debate turns never overlap, so a per-turn cap is the same as a global one
        llm_semaphore: Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_LLM_CALLS)),
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
    let usage = result.usage();
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

fn build_client(
    reviewer: &ReviewerConfig,
    proxy_url: Option<&str>,
) -> Result<Arc<dyn LLMClientDyn>> {
    build_reviewer_client(reviewer, proxy_url)
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
    /// partial dialogue. Surfaced as exit code 3 in the default-review/`ask` CLI arms.
    pub degraded: bool,
}

pub async fn run_debate(
    repo: &Path,
    prompt: &str,
    config: &Config,
    opts: DebateOptions,
) -> Result<DebateOutcome> {
    let DebateOptions {
        max_rounds,
        max_turns,
        verbose,
        mode,
        alloy,
        format,
    } = opts;
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

    // proxy client stays bound for the function so its local server outlives the debate;
    // only its base URL is threaded into the client builders.
    #[cfg(feature = "antigravity")]
    let gemini_proxy = match config_needs_gemini_proxy(config) {
        true => {
            info!("Starting Gemini proxy (agy-keyring)");
            Some(crate::gemini_proxy::GeminiProxyClient::new().await?)
        }
        false => None,
    };
    #[cfg(feature = "antigravity")]
    let proxy_url: Option<String> = gemini_proxy.as_ref().map(|p| p.base_url());
    #[cfg(not(feature = "antigravity"))]
    let proxy_url: Option<String> = None;

    let actor_client: Arc<dyn LLMClientDyn>;
    let critic_client: Arc<dyn LLMClientDyn>;
    let actor_label: ModelLabel;
    let critic_label: ModelLabel;
    let actor_compact_threshold: Option<u64>;
    let critic_compact_threshold: Option<u64>;

    if alloy {
        let mut slots = Vec::new();
        for r in &config.reviewer {
            slots.push((build_client(r, proxy_url.as_deref())?, r.model.clone()));
        }
        let shared: Arc<dyn LLMClientDyn> = Arc::new(nitpicker_agent::llm::AlloyClient::new(slots)?);
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
        actor_client = build_client(actor_cfg, proxy_url.as_deref())?;
        critic_client = build_client(critic_cfg, proxy_url.as_deref())?;
        actor_label = ModelLabel::plain(&actor_cfg.model);
        critic_label = ModelLabel::plain(&critic_cfg.model);
        actor_compact_threshold = config.reviewer_compact_threshold(actor_cfg);
        critic_compact_threshold = config.reviewer_compact_threshold(critic_cfg);
    }
    let session_logger = SessionLogger::maybe_new(config.log_trajectories())?;
    if let Some(logger) = &session_logger {
        info!(path = %logger.root().display(), "trajectory logging enabled");
    }

    let project_context = crate::review::build_context(repo).await;

    let agg_client: Arc<dyn LLMClientDyn> = build_aggregator_client(agg_cfg, proxy_url.as_deref())?;

    let actor_role = mode.actor_role();
    let critic_role = mode.critic_role();
    let actor_system = mode.actor_system();
    let critic_system = mode.critic_system();

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
        println!();
    }

    // (role_label, round_number, verdict_text)
    let mut verdicts: Vec<(String, usize, String)> = Vec::new();
    let mut converged = false;
    let mut final_round = 0usize;
    let mut any_turn_succeeded = false;
    let mut degraded = false;
    let mut usage = UsageReport::default();

    'debate: for round in 1..=max_rounds {
        final_round = round;

        let (pb, _) = make_spinner(&mp);
        pb.set_prefix(colored_role_stderr(actor_role));
        pb.set_message(crate::progress::bar_message(format!(
            "round {round} — debating…"
        )));
        let sub_pb = make_sub_spinner(&mp, &pb);
        let msg = build_turn_message(prompt, &verdicts, round, actor_role);
        let start = std::time::Instant::now();
        let actor_pb = pb.clone();
        let actor_sub_pb = sub_pb.clone();
        let actor_progress = (!verbose).then_some(Arc::new(move |progress: AgentProgress| {
            actor_pb.set_message(crate::progress::bar_message(format!(
                "round {round} — debating… ({} turns, {} tool calls, {} subagents)",
                progress.turns, progress.tool_calls, progress.subagents_spawned
            )));
            actor_sub_pb.set_message(crate::progress::detail_message(
                "    ↳ ",
                progress.last_subagent.as_deref(),
            ));
        })
            as Arc<dyn Fn(AgentProgress) + Send + Sync>);
        let DebateTurnResult {
            verdict,
            turns,
            tool_calls,
            subagents_spawned,
            usage: turn_usage,
            agent_failed: actor_failed,
            used_fallback: actor_fallback,
        } = run_debate_turn(DebateTurnRequest {
            client: Arc::clone(&actor_client),
            compact_threshold: actor_compact_threshold,
            model: &actor_label.alias,
            system_prompt: &actor_system,
            initial_message: &msg,
            max_turns,
            work_dir: repo,
            progress: actor_progress,
            project_context: Some(project_context.clone()),
            session_writer: session_logger
                .as_ref()
                .map(|logger| logger.child(format!("review-{round}.jsonl"))),
        })
        .await?;
        usage.add(turn_usage, subagents_spawned);
        let elapsed = start.elapsed().as_secs();
        sub_pb.finish_and_clear();
        pb.set_style(done_style.clone());
        pb.finish_with_message(crate::progress::bar_message(format!(
            "✓ round {round} ({turns} turns, {tool_calls} tool calls, {subagents_spawned} subagents, {} in, {} out, {} total tokens, {elapsed}s)",
            turn_usage.input_tokens, turn_usage.output_tokens, turn_usage.total_tokens
        )));
        if verbose && stdout_ok && stdout_is_terminal() {
            println!();
            skin.print_text(&verdict.text);
            println!();
        }
        any_turn_succeeded |= !actor_failed;
        degraded |= actor_failed || actor_fallback;
        verdicts.push((actor_role.to_string(), round, verdict.text));

        let (pb, _) = make_spinner(&mp);
        pb.set_prefix(colored_role_stderr(critic_role));
        pb.set_message(crate::progress::bar_message(format!(
            "round {round} — debating…"
        )));
        let sub_pb = make_sub_spinner(&mp, &pb);
        let msg = build_turn_message(prompt, &verdicts, round, critic_role);
        let start = std::time::Instant::now();
        let critic_pb = pb.clone();
        let critic_sub_pb = sub_pb.clone();
        let critic_progress = (!verbose).then_some(Arc::new(move |progress: AgentProgress| {
            critic_pb.set_message(crate::progress::bar_message(format!(
                "round {round} — debating… ({} turns, {} tool calls, {} subagents)",
                progress.turns, progress.tool_calls, progress.subagents_spawned
            )));
            critic_sub_pb.set_message(crate::progress::detail_message(
                "    ↳ ",
                progress.last_subagent.as_deref(),
            ));
        })
            as Arc<dyn Fn(AgentProgress) + Send + Sync>);
        let DebateTurnResult {
            verdict,
            turns,
            tool_calls,
            subagents_spawned,
            usage: turn_usage,
            agent_failed: critic_failed,
            used_fallback: critic_fallback,
        } = run_debate_turn(DebateTurnRequest {
            client: Arc::clone(&critic_client),
            compact_threshold: critic_compact_threshold,
            model: &critic_label.alias,
            system_prompt: &critic_system,
            initial_message: &msg,
            max_turns,
            work_dir: repo,
            progress: critic_progress,
            project_context: Some(project_context.clone()),
            session_writer: session_logger
                .as_ref()
                .map(|logger| logger.child(format!("validate-{round}.jsonl"))),
        })
        .await?;
        usage.add(turn_usage, subagents_spawned);
        let elapsed = start.elapsed().as_secs();
        sub_pb.finish_and_clear();
        pb.set_style(done_style.clone());
        pb.finish_with_message(crate::progress::bar_message(format!(
            "✓ round {round} ({turns} turns, {tool_calls} tool calls, {subagents_spawned} subagents, {} in, {} out, {} total tokens, {elapsed}s)",
            turn_usage.input_tokens, turn_usage.output_tokens, turn_usage.total_tokens
        )));
        if verbose && stdout_ok && stdout_is_terminal() {
            println!();
            skin.print_text(&verdict.text);
            println!();
        }
        any_turn_succeeded |= !critic_failed;
        degraded |= critic_failed || critic_fallback;
        // Convergence requires a real agreement: a critic that agrees with a failed actor's
        // `*Agent failed*` stub (or a failed critic, whose verdict defaults to agree=false) must
        // not end the debate early.
        let agreed = verdict.agree && !actor_failed && !critic_failed;
        verdicts.push((critic_role.to_string(), round, verdict.text));

        if agreed {
            converged = true;
            break 'debate;
        }
    }

    // Every turn failed (provider down, bad config): the dialogue is nothing but failure stubs, so
    // synthesizing a meta-verdict would fabricate a confident review from errors. Surface the
    // failure instead — `run_pr` maps this to a `status: "error"` envelope and posts no comment.
    if !any_turn_succeeded {
        eyre::bail!("all debate turns failed; refusing to synthesize a verdict");
    }

    // meta-review: non-agentic single completion over the full dialogue
    let dialogue = verdicts
        .iter()
        .map(|(label, rnd, text)| format!("### {label} (Round {rnd})\n{text}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    let meta_prompt = format!(
        "The following is a debate about: {prompt}\n\n{dialogue}\n\n---\n{}",
        mode.meta_instruction()
    );
    let meta_completion = Completion {
        model: agg_cfg.model.clone(),
        prompt: Message::user(meta_prompt),
        preamble: Some(mode.meta_preamble().to_string()),
        history: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        max_tokens: agg_cfg.max_tokens.or(Some(8192)),
        additional_params: None,
    };
    let (pb, _) = make_spinner(&mp);
    pb.set_prefix(colored_role_stderr("Meta-review"));
    pb.set_message(crate::progress::bar_message("synthesizing…"));
    let meta_response: nitpicker_agent::llm::CompletionResponse =
        agg_client.completion(meta_completion).await?;
    usage.add(meta_response.usage, 0);
    let meta_text = meta_response.text();
    pb.set_style(done_style);
    pb.finish_with_message("✓ done");
    if let Some(logger) = &session_logger {
        logger
            .write_aggregation(&AggregationRecord {
                kind: "aggregation".to_string(),
                model: agg_cfg.model.clone(),
                text: meta_text.clone(),
                rounds: Some(final_round),
                converged: Some(converged),
            })
            .await?;
    }
    // write transcript file
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let transcript_path = std::env::temp_dir().join(format!("{}-{ts}.md", mode.label()));
    let now = chrono::Local::now();
    let convergence_status = if converged {
        format!("converged at round {final_round}")
    } else {
        format!("max rounds ({max_rounds}) reached without convergence")
    };

    let label = mode.label();
    let mut transcript = format!(
        "# Debate Transcript ({label})\n\n\
        **Topic:** {prompt}\n\
        **{actor_role} model:** {}\n\
        **{critic_role} model:** {}\n\
        **Meta-reviewer:** {}\n\
        **Date:** {}\n\
        **Convergence:** {convergence_status}\n\
        **Rounds:** {final_round}\n\n---\n\n",
        actor_label.full,
        critic_label.full,
        agg_cfg.model,
        now.format("%Y-%m-%d %H:%M:%S"),
    );
    for (label, rnd, text) in &verdicts {
        transcript.push_str(&format!("## {label} — Round {rnd}\n\n{text}\n\n"));
    }
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
    })
}
