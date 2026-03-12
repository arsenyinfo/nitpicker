use crate::config::{Config, ProviderType, ReviewerConfig};
use crate::llm::{Completion, LLMClient, LLMClientDyn, LLMProvider, WithRetryExt};
pub use crate::prompts::DebateMode;
use crate::tools::{Tool, all_tools, floor_char_boundary};
use eyre::Result;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rig::OneOrMany;
use rig::completion::Message;
use rig::completion::message::{ToolResult, ToolResultContent, UserContent};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use termimad::MadSkin;
use tracing::info;

struct DebateVerdict {
    text: String,
    agree: bool,
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

const MAX_TURNS_PER_ROUND: usize = 50;
const MAX_TOOL_RESULT_BYTES: usize = 50_000;
const AGENT_MAX_TOKENS: u64 = 8192;

async fn run_debate_turn(
    client: Arc<dyn LLMClientDyn>,
    model: &str,
    system_prompt: &str,
    initial_message: &str,
    work_dir: &Path,
) -> Result<(DebateVerdict, usize)> {
    let verdict_store: Arc<Mutex<Option<DebateVerdict>>> = Arc::new(Mutex::new(None));
    let submit_tool = Arc::new(SubmitVerdictTool {
        verdict: Arc::clone(&verdict_store),
    });

    let mut tools_map: HashMap<String, Arc<dyn Tool>> = all_tools();
    tools_map.insert("submit_verdict".to_string(), submit_tool as Arc<dyn Tool>);

    let mut history = Vec::new();
    let mut prompt = Message::user(initial_message.to_string());
    history.push(prompt.clone());
    let mut tool_call_count = 0usize;
    let mut total_output_tokens = 0u64;
    let mut empty_response_count = 0usize;

    for _turn in 0..MAX_TURNS_PER_ROUND {
        let completion = Completion {
            model: model.to_string(),
            prompt: prompt.clone(),
            preamble: Some(system_prompt.to_string()),
            history: history[..history.len().saturating_sub(1)].to_vec(),
            tools: crate::tools::tool_definitions(&tools_map),
            temperature: None,
            max_tokens: Some(AGENT_MAX_TOKENS),
            additional_params: None,
        };

        let response = client.completion(completion).await?;
        total_output_tokens += response.output_tokens;
        let assistant_message = response.message();
        history.push(assistant_message.clone());

        if let Some(tool_calls) = response.tool_calls() {
            tool_call_count += tool_calls.len();
            let mut results = Vec::new();
            for call in &tool_calls {
                let tool_name = call.function.name.clone();
                let args = call.function.arguments.clone();
                info!(tool = %tool_name, args = %args, "tool call");
                let output = match tools_map.get(&tool_name) {
                    Some(tool) => match tool.call(args, work_dir.to_path_buf()).await {
                        Ok(output) => output,
                        Err(err) => format!("Error: {err}"),
                    },
                    None => format!("Error: unknown tool '{tool_name}'"),
                };
                let mut output = output;
                if output.len() > MAX_TOOL_RESULT_BYTES {
                    let boundary = floor_char_boundary(&output, MAX_TOOL_RESULT_BYTES);
                    output.truncate(boundary);
                    output.push_str("\n... truncated (>50k bytes)");
                }
                results.push(ToolResult {
                    id: call.id.clone(),
                    call_id: call.call_id.clone(),
                    content: OneOrMany::one(ToolResultContent::text(output)),
                });
            }

            // all tools in the batch have been executed; if submit_verdict was among them,
            // the turn is complete — results don't need to be sent back since there's no
            // further LLM call for this turn
            if let Some(verdict) = verdict_store
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
            {
                info!(
                    tool_calls = tool_call_count,
                    output_tokens = total_output_tokens,
                    "turn finished via submit_verdict"
                );
                return Ok((verdict, tool_call_count));
            }

            empty_response_count = 0;
            let tool_message = Message::User {
                content: OneOrMany::many(results.into_iter().map(UserContent::ToolResult))
                    .expect("tool results must not be empty"),
            };
            history.push(tool_message.clone());
            prompt = tool_message;
        } else {
            // LLM returned text without calling submit_verdict — implicit non-agree verdict
            let text = response.text();
            if text.is_empty() {
                empty_response_count += 1;
                if empty_response_count >= 3 {
                    eyre::bail!("model returned empty response 3 times consecutively within this turn");
                }
                let nudge = Message::user(
                    "Please proceed with your analysis and call submit_verdict with your findings."
                        .to_string(),
                );
                history.push(nudge.clone());
                prompt = nudge;
                continue;
            }
            info!(
                tool_calls = tool_call_count,
                output_tokens = total_output_tokens,
                "turn finished via text response"
            );
            return Ok((DebateVerdict { text, agree: false }, tool_call_count));
        }
    }

    eyre::bail!("debate turn exceeded {MAX_TURNS_PER_ROUND} turns")
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
        "Actor" | "Reviewer" => "\x1b[96m",    // bright cyan
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
    if reviewer.provider.is_gemini() && reviewer.use_oauth() {
        let proxy_url = gemini_proxy
            .map(|p| p.base_url())
            .ok_or_else(|| eyre::eyre!("Gemini OAuth requires proxy"))?;
        return crate::llm::create_gemini_client_with_proxy(&proxy_url);
    }
    let provider = match &reviewer.provider {
        ProviderType::Anthropic => LLMProvider::Anthropic,
        ProviderType::Gemini => LLMProvider::Gemini,
        ProviderType::AnthropicCompatible => LLMProvider::AnthropicCompatible {
            base_url: reviewer
                .base_url
                .clone()
                .ok_or_else(|| eyre::eyre!("base_url required for anthropic_compatible"))?,
            api_key_env: reviewer
                .api_key_env
                .clone()
                .ok_or_else(|| eyre::eyre!("api_key_env required for anthropic_compatible"))?,
        },
        ProviderType::OpenAiCompatible => LLMProvider::OpenAICompatible {
            base_url: reviewer
                .base_url
                .clone()
                .ok_or_else(|| eyre::eyre!("base_url required for openai_compatible"))?,
            api_key_env: reviewer
                .api_key_env
                .clone()
                .ok_or_else(|| eyre::eyre!("api_key_env required for openai_compatible"))?,
        },
    };
    Ok(provider.client_from_env()?.with_retry().into_arc())
}

pub async fn run_debate(
    repo: &Path,
    prompt: &str,
    config: &Config,
    max_rounds: usize,
    verbose: bool,
    mode: DebateMode,
) -> Result<()> {
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
        .any(|r| r.provider.is_gemini() && r.use_oauth())
        || (agg_cfg.provider.is_gemini() && agg_cfg.use_oauth());
    let gemini_proxy = if needs_oauth {
        info!("Starting Gemini proxy for OAuth authentication...");
        Some(crate::gemini_proxy::GeminiProxyClient::new().await?)
    } else {
        None
    };

    let actor_client = build_client(actor_cfg, gemini_proxy.as_ref())?;
    let critic_client = build_client(critic_cfg, gemini_proxy.as_ref())?;

    let agg_client: Arc<dyn LLMClientDyn> = if agg_cfg.provider.is_gemini() && agg_cfg.use_oauth() {
        let proxy_url = gemini_proxy
            .as_ref()
            .map(|p| p.base_url())
            .ok_or_else(|| eyre::eyre!("Gemini OAuth requires proxy"))?;
        crate::llm::create_gemini_client_with_proxy(&proxy_url)?
    } else {
        let provider = match &agg_cfg.provider {
            ProviderType::Anthropic => LLMProvider::Anthropic,
            ProviderType::Gemini => LLMProvider::Gemini,
            ProviderType::AnthropicCompatible => LLMProvider::AnthropicCompatible {
                base_url: agg_cfg
                    .base_url
                    .clone()
                    .ok_or_else(|| eyre::eyre!("base_url required"))?,
                api_key_env: agg_cfg
                    .api_key_env
                    .clone()
                    .ok_or_else(|| eyre::eyre!("api_key_env required"))?,
            },
            ProviderType::OpenAiCompatible => LLMProvider::OpenAICompatible {
                base_url: agg_cfg
                    .base_url
                    .clone()
                    .ok_or_else(|| eyre::eyre!("base_url required"))?,
                api_key_env: agg_cfg
                    .api_key_env
                    .clone()
                    .ok_or_else(|| eyre::eyre!("api_key_env required"))?,
            },
        };
        provider.client_from_env()?.with_retry().into_arc()
    };

    let actor_role = mode.actor_role();
    let critic_role = mode.critic_role();
    let actor_system = mode.actor_system();
    let critic_system = mode.critic_system();

    let done_style = ProgressStyle::with_template("  {prefix:<12} {msg}").unwrap();
    let skin = MadSkin::default();

    // print cast before debate starts
    print_cast_line(actor_role, &format!("{} · {}", actor_cfg.name, actor_cfg.model));
    print_cast_line(critic_role, &format!("{} · {}", critic_cfg.name, critic_cfg.model));
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
        let (verdict, tool_calls) = run_debate_turn(
            Arc::clone(&actor_client),
            &actor_cfg.model,
            &actor_system,
            &msg,
            repo,
        )
        .await?;
        let elapsed = start.elapsed().as_secs();
        pb.set_style(done_style.clone());
        pb.finish_with_message(format!(
            "✓ round {round} ({tool_calls} tool calls, {elapsed}s)"
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
        let (verdict, tool_calls) = run_debate_turn(
            Arc::clone(&critic_client),
            &critic_cfg.model,
            &critic_system,
            &msg,
            repo,
        )
        .await?;
        let elapsed = start.elapsed().as_secs();
        pb.set_style(done_style.clone());
        pb.finish_with_message(format!(
            "✓ round {round} ({tool_calls} tool calls, {elapsed}s)"
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
        temperature: None,
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

    Ok(())
}
