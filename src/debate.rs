use crate::agent::{AgentConfig, AgentDepth, AgentProgress, add_spawn_subagent_tool, run_agent};
use crate::config::{Config, ReviewerConfig};
use crate::llm::{Completion, LLMClientDyn};
pub use crate::prompts::DebateMode;
use crate::provider::{
    aggregator_needs_gemini_oauth, build_aggregator_client, build_reviewer_client,
    reviewer_needs_gemini_oauth,
};
use crate::tools::{Tool, all_tools};
use eyre::Result;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rig::completion::Message;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use termimad::MadSkin;
use tracing::info;

struct DebateVerdict {
    text: String,
    agree: bool,
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
}

struct SubmitVerdictTool {
    verdict: Arc<Mutex<Option<DebateVerdict>>>,
}

impl Tool for SubmitVerdictTool {
    fn name(&self) -> String {
        "submit_verdict".to_string()
    }

    fn definition(&self) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
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

async fn run_debate_turn(
    request: DebateTurnRequest<'_>,
) -> Result<(DebateVerdict, usize, usize, usize, u64, u64, u64)> {
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
        model: request.model.to_string(),
        max_turns: request.max_turns,
        compact_threshold: request.compact_threshold,
        system_prompt: request.system_prompt.to_string(),
        client: request.client,
        depth: AgentDepth::TopLevel,
        terminal_tools: vec!["submit_verdict".to_string()],
        empty_response_nudge: Some(
            "Please proceed with your analysis and call submit_verdict when you are done."
                .to_string(),
        ),
        max_empty_responses: 3,
        subagent_counter,
        progress: request.progress,
        project_context: request.project_context,
    };

    let result = run_agent(
        config,
        request.initial_message,
        &tools_map,
        request.work_dir,
    )
    .await?;
    if let Some(verdict) = verdict_store
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
    {
        return Ok((
            verdict,
            result.turns,
            result.tool_calls,
            result.subagents_spawned,
            result.total_input_tokens,
            result.total_output_tokens,
            result.total_tokens,
        ));
    }

    Ok((
        DebateVerdict {
            text: result.text,
            agree: false,
        },
        result.turns,
        result.tool_calls,
        result.subagents_spawned,
        result.total_input_tokens,
        result.total_output_tokens,
        result.total_tokens,
    ))
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

fn colored_role(role: &str) -> String {
    format!("{}{role}\x1b[0m", role_color(role))
}

fn print_cast_line(role: &str, info: &str) {
    let pad = " ".repeat(12usize.saturating_sub(role.len()));
    println!("  {}{pad} {info}", colored_role(role));
}

fn make_spinner(verbose: bool) -> (ProgressBar, ProgressStyle) {
    let spinner_style = ProgressStyle::with_template("{spinner:.cyan} {prefix:<12} {msg}")
        .unwrap()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", ""]);
    let pb = ProgressBar::new_spinner();
    if verbose {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }
    pb.set_style(spinner_style.clone());
    pb.enable_steady_tick(Duration::from_millis(80));
    (pb, spinner_style)
}

fn build_client(
    reviewer: &ReviewerConfig,
    gemini_proxy: Option<&crate::gemini_proxy::GeminiProxyClient>,
) -> Result<Arc<dyn LLMClientDyn>> {
    build_reviewer_client(reviewer, gemini_proxy)
}

pub async fn run_debate(
    repo: &Path,
    prompt: &str,
    config: &Config,
    max_rounds: usize,
    max_turns: usize,
    verbose: bool,
    mode: DebateMode,
) -> Result<String> {
    if config.reviewer.len() < 2 {
        eyre::bail!(
            "debate requires at least 2 reviewers in config (actor = reviewer[0], critic = reviewer[1])"
        );
    }

    let actor_cfg = &config.reviewer[0];
    let critic_cfg = &config.reviewer[1];
    let agg_cfg = &config.aggregator;

    let needs_oauth = [actor_cfg, critic_cfg]
        .iter()
        .any(|reviewer| reviewer_needs_gemini_oauth(reviewer))
        || aggregator_needs_gemini_oauth(agg_cfg);
    let gemini_proxy = if needs_oauth {
        info!("Starting Gemini proxy for OAuth authentication...");
        Some(crate::gemini_proxy::GeminiProxyClient::new().await?)
    } else {
        None
    };

    let actor_client = build_client(actor_cfg, gemini_proxy.as_ref())?;
    let critic_client = build_client(critic_cfg, gemini_proxy.as_ref())?;
    let actor_compact_threshold = config.reviewer_compact_threshold(actor_cfg)?;
    let critic_compact_threshold = config.reviewer_compact_threshold(critic_cfg)?;

    let project_context = crate::review::build_context(repo).await;

    let agg_client: Arc<dyn LLMClientDyn> =
        build_aggregator_client(agg_cfg, gemini_proxy.as_ref())?;

    let actor_role = mode.actor_role();
    let critic_role = mode.critic_role();
    let actor_system = mode.actor_system();
    let critic_system = mode.critic_system();

    let done_style = ProgressStyle::with_template("  {prefix:<12} {msg}").unwrap();
    let skin = MadSkin::default();

    // print cast before debate starts
    print_cast_line(
        actor_role,
        &format!("{} · {}", actor_cfg.name, actor_cfg.model),
    );
    print_cast_line(
        critic_role,
        &format!("{} · {}", critic_cfg.name, critic_cfg.model),
    );
    print_cast_line("Meta-review", &agg_cfg.model);
    println!();

    // (role_label, round_number, verdict_text)
    let mut verdicts: Vec<(String, usize, String)> = Vec::new();
    let mut converged = false;
    let mut final_round = 0usize;

    'debate: for round in 1..=max_rounds {
        final_round = round;

        let (pb, _) = make_spinner(verbose);
        pb.set_prefix(colored_role(actor_role));
        pb.set_message(format!("round {round} — debating…"));
        let msg = build_turn_message(prompt, &verdicts, round, actor_role);
        let start = std::time::Instant::now();
        let actor_pb = pb.clone();
        let actor_progress = (!verbose).then_some(Arc::new(move |progress: AgentProgress| {
            let sub = progress.last_subagent.as_deref().map(|s| format!("\n  ↳ {s}")).unwrap_or_default();
            actor_pb.set_message(format!(
                "round {round} — debating… ({} turns, {} tool calls, {} subagents){sub}",
                progress.turns, progress.tool_calls, progress.subagents_spawned
            ));
        })
            as Arc<dyn Fn(AgentProgress) + Send + Sync>);
        let (
            verdict,
            turns,
            tool_calls,
            subagents_spawned,
            total_input_tokens,
            total_output_tokens,
            total_tokens,
        ) = run_debate_turn(DebateTurnRequest {
            client: Arc::clone(&actor_client),
            compact_threshold: actor_compact_threshold,
            model: &actor_cfg.model,
            system_prompt: &actor_system,
            initial_message: &msg,
            max_turns,
            work_dir: repo,
            progress: actor_progress,
            project_context: Some(project_context.clone()),
        })
        .await?;
        let elapsed = start.elapsed().as_secs();
        pb.set_style(done_style.clone());
        pb.finish_with_message(format!(
            "✓ round {round} ({turns} turns, {tool_calls} tool calls, {subagents_spawned} subagents, {total_input_tokens} in, {total_output_tokens} out, {total_tokens} total tokens, {elapsed}s)"
        ));
        println!();
        skin.print_text(&verdict.text);
        println!();
        verdicts.push((actor_role.to_string(), round, verdict.text));

        let (pb, _) = make_spinner(verbose);
        pb.set_prefix(colored_role(critic_role));
        pb.set_message(format!("round {round} — debating…"));
        let msg = build_turn_message(prompt, &verdicts, round, critic_role);
        let start = std::time::Instant::now();
        let critic_pb = pb.clone();
        let critic_progress = (!verbose).then_some(Arc::new(move |progress: AgentProgress| {
            let sub = progress.last_subagent.as_deref().map(|s| format!("\n  ↳ {s}")).unwrap_or_default();
            critic_pb.set_message(format!(
                "round {round} — debating… ({} turns, {} tool calls, {} subagents){sub}",
                progress.turns, progress.tool_calls, progress.subagents_spawned
            ));
        })
            as Arc<dyn Fn(AgentProgress) + Send + Sync>);
        let (
            verdict,
            turns,
            tool_calls,
            subagents_spawned,
            total_input_tokens,
            total_output_tokens,
            total_tokens,
        ) = run_debate_turn(DebateTurnRequest {
            client: Arc::clone(&critic_client),
            compact_threshold: critic_compact_threshold,
            model: &critic_cfg.model,
            system_prompt: &critic_system,
            initial_message: &msg,
            max_turns,
            work_dir: repo,
            progress: critic_progress,
            project_context: Some(project_context.clone()),
        })
        .await?;
        let elapsed = start.elapsed().as_secs();
        pb.set_style(done_style.clone());
        pb.finish_with_message(format!(
            "✓ round {round} ({turns} turns, {tool_calls} tool calls, {subagents_spawned} subagents, {total_input_tokens} in, {total_output_tokens} out, {total_tokens} total tokens, {elapsed}s)"
        ));
        println!();
        skin.print_text(&verdict.text);
        println!();
        let agreed = verdict.agree;
        verdicts.push((critic_role.to_string(), round, verdict.text));

        if agreed {
            converged = true;
            break 'debate;
        }
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
    let (pb, _) = make_spinner(verbose);
    pb.set_prefix(colored_role("Meta-review"));
    pb.set_message("synthesizing…");
    let meta_response: crate::llm::CompletionResponse =
        agg_client.completion(meta_completion).await?;
    let meta_text = meta_response.text();
    pb.set_style(done_style);
    pb.finish_with_message("✓ done");
    println!();
    skin.print_text(&meta_text);

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
        actor_cfg.model,
        critic_cfg.model,
        agg_cfg.model,
        now.format("%Y-%m-%d %H:%M:%S"),
    );
    for (label, rnd, text) in &verdicts {
        transcript.push_str(&format!("## {label} — Round {rnd}\n\n{text}\n\n"));
    }
    transcript.push_str(&format!("---\n\n## Meta-review\n\n{meta_text}\n"));

    tokio::fs::write(&transcript_path, &transcript).await?;
    eprintln!("\nTranscript saved to: {}", transcript_path.display());

    Ok(meta_text)
}
