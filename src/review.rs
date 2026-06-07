use crate::agent::{
    AgentConfig, AgentDepth, AgentProgress, MAX_CONCURRENT_LLM_CALLS, add_spawn_subagent_tool,
    run_agent,
};
use crate::config::{Config, ReviewerConfig};
use crate::llm::{Completion, FinishReason};
pub use crate::prompts::TaskMode;
#[cfg(feature = "antigravity")]
use crate::provider::config_needs_gemini_proxy;
use crate::provider::{build_aggregator_client, build_reviewer_client};
use crate::session::{AggregationRecord, SessionLogger, sanitize_path_component};
use crate::tools::{all_tools, floor_char_boundary, is_binary_file};
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rig_core::completion::Message;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use tokio::sync::Semaphore;

const MAX_CONCURRENT_REVIEWERS: usize = 8;
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
    let session_logger = SessionLogger::maybe_new(config.log_trajectories())?;
    if let Some(logger) = &session_logger {
        info!(path = %logger.root().display(), "trajectory logging enabled");
    }
    let context = build_context(repo).await;
    let system_prompt = mode.system_prompt();
    let initial_message = mode.initial_message(user_prompt);
    let mut handles = Vec::new();
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REVIEWERS));
    // shared across every reviewer + their subagents to cap account-wide in-flight LLM calls
    let llm_semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_LLM_CALLS));

    let mp = MultiProgress::new();
    if verbose {
        mp.set_draw_target(ProgressDrawTarget::hidden());
    }
    let spinner_style = ProgressStyle::with_template("{spinner:.cyan} {prefix:<12} {msg}")
        .unwrap()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", ""]);
    let done_style = ProgressStyle::with_template("  {prefix:<12} {msg}").unwrap();

    // the proxy client stays bound for the whole function so its local server outlives the
    // reviewers; only its base URL is threaded downstream (see build_reviewer_client).
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

    for reviewer in &config.reviewer {
        let tools_map = tools.clone();
        let repo = repo.to_path_buf();
        let name = reviewer.name.clone();
        let subagent_counter = Arc::new(AtomicUsize::new(0));
        let session_writer = session_logger.as_ref().map(|logger| {
            logger.child(format!("reviewer-{}.jsonl", sanitize_path_component(&name)))
        });
        let agent_config = build_agent_config(
            config,
            reviewer,
            &system_prompt,
            max_turns,
            proxy_url.as_deref(),
            Arc::clone(&subagent_counter),
            Arc::clone(&llm_semaphore),
            session_writer,
        );
        info!(reviewer = %name, "spawning agent");

        let pb = mp.add(ProgressBar::new_spinner());
        pb.set_style(spinner_style.clone());
        pb.set_prefix(name.clone());
        pb.set_message("reviewing…");
        pb.enable_steady_tick(Duration::from_millis(80));

        let sub_pb = mp.insert_after(&pb, ProgressBar::new_spinner());
        sub_pb.set_style(ProgressStyle::with_template("{msg}").unwrap());

        let done = done_style.clone();
        let initial_message = initial_message.clone();
        let context = context.clone();
        let sem = Arc::clone(&semaphore);
        let handle: JoinHandle<(String, Result<String>)> = tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");
            let mut config = match agent_config {
                Ok(config) => config,
                Err(err) => {
                    pb.set_style(done.clone());
                    pb.finish_with_message(format!("✗ error: {err}"));
                    sub_pb.finish_and_clear();
                    return (name, Err(err));
                }
            };
            config.project_context = Some(context);
            if !verbose {
                let progress_pb = pb.clone();
                let progress_sub_pb = sub_pb.clone();
                config.progress = Some(Arc::new(move |progress: AgentProgress| {
                    progress_pb.set_message(format!(
                        "reviewing… ({} turns, {} tool calls, {} subagents)",
                        progress.turns, progress.tool_calls, progress.subagents_spawned
                    ));
                    progress_sub_pb.set_message(
                        progress
                            .last_subagent
                            .as_deref()
                            .map(|s| format!("    ↳ {s}"))
                            .unwrap_or_default(),
                    );
                }));
            }
            let start = Instant::now();
            let result = run_agent(config, &initial_message, &tools_map, &repo).await;
            let elapsed = start.elapsed().as_secs();
            sub_pb.finish_and_clear();
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
                rendered.push(format!("## {name} review\n\n*Failed: {err:#}*"));
                info!(reviewer = %name, error = ?err, "review failed");
            }
            Err(err) => {
                rendered.push(format!("## reviewer failed\n\n*Failed: {err:#}*"));
                info!(error = ?err, "reviewer task failed");
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
    let client = build_aggregator_client(agg, proxy_url.as_deref())?;
    let completion = Completion {
        model: agg.model.clone(),
        prompt: Message::user(reduce_prompt),
        preamble: Some(mode.aggregator_preamble().to_string()),
        history: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
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
    let text = response.text();
    if let Some(logger) = &session_logger {
        logger
            .write_aggregation(&AggregationRecord {
                kind: "aggregation".to_string(),
                model: agg.model.clone(),
                text: text.clone(),
                rounds: None,
                converged: None,
            })
            .await?;
    }
    Ok(text)
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

// internal single-call-site builder; the args are distinct per-reviewer handles, not worth a struct
#[allow(clippy::too_many_arguments)]
fn build_agent_config(
    config: &Config,
    reviewer: &ReviewerConfig,
    system_prompt: &str,
    max_turns: usize,
    proxy_url: Option<&str>,
    subagent_counter: Arc<AtomicUsize>,
    llm_semaphore: Arc<Semaphore>,
    session_writer: Option<crate::session::SessionWriter>,
) -> Result<AgentConfig> {
    let client = build_reviewer_client(reviewer, proxy_url)?;
    let compact_threshold = config.reviewer_compact_threshold(reviewer);

    Ok(AgentConfig {
        name: reviewer.name.clone(),
        session_agent: "root".to_string(),
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
        llm_semaphore,
        progress: None,
        project_context: None,
        session_writer,
    })
}
