use crate::agent::{
    AgentConfig, AgentDepth, AgentProgress, add_spawn_subagent_tool, run_agent,
};
use crate::config::{Config, ReviewerConfig};
use crate::llm::{Completion, FinishReason};
use crate::provider::{
    aggregator_needs_gemini_oauth, build_aggregator_client, build_reviewer_client,
    reviewer_needs_gemini_oauth,
};
pub use crate::prompts::TaskMode;
use crate::tools::{all_tools, floor_char_boundary, is_binary_file};
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rig::completion::Message;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tracing::info;

pub async fn run_review(
    repo: &Path,
    user_prompt: &str,
    config: &Config,
    max_turns: usize,
    verbose: bool,
    mode: TaskMode,
) -> Result<String> {
    let mut tools = all_tools();
    add_spawn_subagent_tool(&mut tools);
    let context = build_context(repo).await;
    let system_prompt = mode.system_prompt();
    let initial_message = mode.initial_message("", user_prompt);
    let mut handles = Vec::new();

    let mp = MultiProgress::new();
    if verbose {
        mp.set_draw_target(ProgressDrawTarget::hidden());
    }
    let spinner_style = ProgressStyle::with_template("{spinner:.cyan} {prefix:<12} {msg}")
        .unwrap()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", ""]);
    let done_style = ProgressStyle::with_template("  {prefix:<12} {msg}").unwrap();

    // Check if we need to start the Gemini proxy for any OAuth-enabled reviewer
    let gemini_proxy = if config
        .reviewer
        .iter()
        .any(reviewer_needs_gemini_oauth)
        || aggregator_needs_gemini_oauth(&config.aggregator)
    {
        info!("Starting Gemini proxy for OAuth authentication...");
        Some(crate::gemini_proxy::GeminiProxyClient::new().await?)
    } else {
        None
    };

    for reviewer in &config.reviewer {
        let tools_map = tools.clone();
        let repo = repo.to_path_buf();
        let name = reviewer.name.clone();
        let subagent_counter = Arc::new(AtomicUsize::new(0));
        let agent_config = build_agent_config(
            config,
            reviewer,
            &system_prompt,
            max_turns,
            gemini_proxy.as_ref(),
            Arc::clone(&subagent_counter),
        );
        info!(reviewer = %name, "spawning agent");

        let pb = mp.add(ProgressBar::new_spinner());
        pb.set_style(spinner_style.clone());
        pb.set_prefix(name.clone());
        pb.set_message("reviewing…");
        pb.enable_steady_tick(Duration::from_millis(80));

        let done = done_style.clone();
        let initial_message = initial_message.clone();
        let context = context.clone();
        let handle: JoinHandle<(String, Result<String>)> = tokio::spawn(async move {
            let mut config = match agent_config {
                Ok(config) => config,
                Err(err) => {
                    pb.set_style(done.clone());
                    pb.finish_with_message(format!("✗ error: {err}"));
                    return (name, Err(err));
                }
            };
            config.project_context = Some(context);
            if !verbose {
                let progress_pb = pb.clone();
                config.progress = Some(Arc::new(move |progress: AgentProgress| {
                    progress_pb.set_message(format!(
                        "reviewing… ({} turns, {} tool calls, {} subagents)",
                        progress.turns, progress.tool_calls, progress.subagents_spawned
                    ));
                }));
            }
            let start = Instant::now();
            let result = run_agent(config, &initial_message, &tools_map, &repo).await;
            let elapsed = start.elapsed().as_secs();
            pb.set_style(done);
            match &result {
                Ok(r) => pb.finish_with_message(format!(
                    "✓ done ({elapsed}s, {} turns, {} tool calls, {} subagents, {} in, {} out, {} total tokens)",
                    r.turns,
                    r.tool_calls,
                    r.subagents_spawned,
                    r.total_input_tokens,
                    r.total_output_tokens,
                    r.total_tokens
                )),
                Err(e) => pb.finish_with_message(format!("✗ failed: {e}")),
            }
            (name, result.map(|r| r.text))
        });
        handles.push(handle);
    }

    let mut rendered = Vec::new();
    for handle in handles {
        match handle.await {
            Ok((name, Ok(text))) => {
                rendered.push(format!("## {name} review\n\n{text}"));
                info!(reviewer = %name, "review completed");
            }
            Ok((name, Err(err))) => {
                rendered.push(format!("## {name} review\n\n*Failed: {err}*"));
                info!(reviewer = %name, error = %err, "review failed");
            }
            Err(err) => {
                rendered.push(format!("## reviewer failed\n\n*Failed: {err}*"));
                info!(error = %err, "reviewer task failed");
            }
        }
    }

    let combined = rendered.join("\n\n---\n\n");
    let reduce_prompt = mode.reduce_prompt(&combined);

    let pb_agg = mp.add(ProgressBar::new_spinner());
    pb_agg.set_style(spinner_style);
    pb_agg.set_prefix("aggregator");
    pb_agg.set_message("synthesizing…");
    pb_agg.enable_steady_tick(Duration::from_millis(80));

    let agg = &config.aggregator;
    let client = build_aggregator_client(agg, gemini_proxy.as_ref())?;
    let completion = Completion {
        model: agg.model.clone(),
        prompt: Message::user(reduce_prompt),
        preamble: Some(mode.aggregator_preamble().to_string()),
        history: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: agg.max_tokens.or(Some(8192)),
        additional_params: None,
    };
    let response = client.completion(completion).await?;
    pb_agg.set_style(done_style);
    if response.finish_reason == FinishReason::ToolUse {
        pb_agg.finish_with_message("✗ failed: unexpected tool call");
        return Err(eyre::eyre!("aggregator returned tool calls unexpectedly"));
    }
    pb_agg.finish_with_message("✓ done");
    Ok(response.text())
}

const MAX_CONTEXT_SIZE: usize = 50_000;

pub async fn build_context(repo: &Path) -> String {
    let mut context = String::new();

    let repo_canonical = match tokio::fs::canonicalize(repo).await {
        Ok(p) => p,
        Err(_) => {
            tracing::warn!("Failed to canonicalize repo path, skipping context files");
            return context;
        }
    };

    for filename in ["CLAUDE.md", "AGENTS.md"] {
        let path = repo_canonical.join(filename);

        if !path.starts_with(&repo_canonical) {
            tracing::warn!("Context file path escapes repo root: {}", filename);
            continue;
        }

        let metadata = match tokio::fs::metadata(&path).await {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                tracing::warn!("Cannot access context file {}: {}", filename, e);
                continue;
            }
        };

        if !metadata.is_file() {
            continue;
        }

        match is_binary_file(&path).await {
            Ok(true) => {
                tracing::warn!("Context file appears to be binary, skipping: {}", filename);
                continue;
            }
            Ok(false) => {}
            Err(e) => {
                tracing::warn!("Cannot check if context file is binary {}: {}", filename, e);
                continue;
            }
        }

        match tokio::fs::read_to_string(&path).await {
            Ok(content) => {
                let content = if content.len() > MAX_CONTEXT_SIZE {
                    let boundary = floor_char_boundary(&content, MAX_CONTEXT_SIZE);
                    format!(
                        "{}\n... truncated ({} chars)",
                        &content[..boundary],
                        content.len()
                    )
                } else {
                    content
                };
                context.push_str("## Project Context (from ");
                context.push_str(filename);
                context.push_str(")\n\n");
                context.push_str(&content);
                break;
            }
            Err(e) => {
                tracing::warn!("Failed to read context file {}: {}", filename, e);
            }
        }
    }

    context
}

fn build_agent_config(
    config: &Config,
    reviewer: &ReviewerConfig,
    system_prompt: &str,
    max_turns: usize,
    gemini_proxy: Option<&crate::gemini_proxy::GeminiProxyClient>,
    subagent_counter: Arc<AtomicUsize>,
) -> Result<AgentConfig> {
    let client = build_reviewer_client(reviewer, gemini_proxy)?;
    let compact_threshold = config.reviewer_compact_threshold(reviewer)?;

    Ok(AgentConfig {
        name: reviewer.name.clone(),
        model: reviewer.model.clone(),
        max_turns,
        compact_threshold,
        system_prompt: system_prompt.to_string(),
        client,
        depth: AgentDepth::TopLevel,
        terminal_tools: Vec::new(),
        empty_response_nudge: None,
        max_empty_responses: 0,
        subagent_counter,
        progress: None,
        project_context: None,
    })
}
