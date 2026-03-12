use crate::agent::{AgentConfig, run_agent};
use crate::config::{Config, ProviderType, ReviewerConfig};
use crate::llm::{Completion, FinishReason, LLMClient, LLMProvider, WithRetryExt};
pub use crate::prompts::TaskMode;
use crate::tools::{all_tools, floor_char_boundary, is_binary_file};
use eyre::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rig::completion::Message;
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tracing::info;

pub async fn run_review(
    repo: &Path,
    user_prompt: &str,
    config: &Config,
    verbose: bool,
    mode: TaskMode,
) -> Result<String> {
    let tools = all_tools();
    let context = build_context(repo).await;
    let system_prompt = mode.system_prompt();
    let initial_message = mode.initial_message(&context, user_prompt);
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
        .any(|r| r.provider.is_gemini() && r.use_oauth())
        || (config.aggregator.provider.is_gemini() && config.aggregator.use_oauth())
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
        let agent_config =
            build_agent_config(reviewer, &system_prompt, gemini_proxy.as_ref()).await;
        info!(reviewer = %name, "spawning agent");

        let pb = mp.add(ProgressBar::new_spinner());
        pb.set_style(spinner_style.clone());
        pb.set_prefix(name.clone());
        pb.set_message("reviewing…");
        pb.enable_steady_tick(Duration::from_millis(80));

        let done = done_style.clone();
        let initial_message = initial_message.clone();
        let handle: JoinHandle<(String, Result<String>)> = tokio::spawn(async move {
            let config = match agent_config {
                Ok(config) => config,
                Err(err) => {
                    pb.set_style(done.clone());
                    pb.finish_with_message(format!("✗ error: {err}"));
                    return (name, Err(err));
                }
            };
            let start = Instant::now();
            let result = run_agent(
                config,
                &initial_message,
                &tools_map,
                &repo,
            )
            .await;
            let elapsed = start.elapsed().as_secs();
            pb.set_style(done);
            match &result {
                Ok(r) => pb.finish_with_message(format!("✓ done ({elapsed}s, {} turns)", r.turns)),
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
    let client = build_aggregator_client(agg, gemini_proxy.as_ref()).await?;
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

async fn build_context(repo: &Path) -> String {
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
                    format!("{}\n... truncated ({} chars)", &content[..boundary], content.len())
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

async fn build_agent_config(
    reviewer: &ReviewerConfig,
    system_prompt: &str,
    gemini_proxy: Option<&crate::gemini_proxy::GeminiProxyClient>,
) -> Result<AgentConfig> {
    let client: std::sync::Arc<dyn crate::llm::LLMClientDyn> =
        if reviewer.provider.is_gemini() && reviewer.use_oauth() {
            // Use OAuth via proxy
            let proxy_url = gemini_proxy
                .map(|p| p.base_url())
                .ok_or_else(|| eyre::eyre!("Gemini proxy required for OAuth but not available"))?;
            info!("Using Gemini OAuth via proxy at {}", proxy_url);
            crate::llm::create_gemini_client_with_proxy(&proxy_url)?
        } else {
            // Use API key auth
            provider_from_config(
                &reviewer.provider,
                reviewer.base_url.as_deref(),
                reviewer.api_key_env.as_deref(),
            )?
            .client_from_env()?
            .with_retry()
            .into_arc()
        };

    Ok(AgentConfig {
        name: reviewer.name.clone(),
        model: reviewer.model.clone(),
        system_prompt: system_prompt.to_string(),
        client,
    })
}

async fn build_aggregator_client(
    agg: &crate::config::AggregatorConfig,
    gemini_proxy: Option<&crate::gemini_proxy::GeminiProxyClient>,
) -> Result<std::sync::Arc<dyn crate::llm::LLMClientDyn>> {
    if agg.provider.is_gemini() && agg.use_oauth() {
        // Use OAuth via proxy
        let proxy_url = gemini_proxy
            .map(|p| p.base_url())
            .ok_or_else(|| eyre::eyre!("Gemini proxy required for OAuth but not available"))?;
        info!("Using Gemini OAuth via proxy at {}", proxy_url);
        Ok(crate::llm::create_gemini_client_with_proxy(&proxy_url)?)
    } else {
        // Use API key auth
        Ok(provider_from_config(
            &agg.provider,
            agg.base_url.as_deref(),
            agg.api_key_env.as_deref(),
        )?
        .client_from_env()?
        .with_retry()
        .into_arc())
    }
}

fn provider_from_config(
    provider: &ProviderType,
    base_url: Option<&str>,
    api_key_env: Option<&str>,
) -> Result<LLMProvider> {
    match provider {
        ProviderType::Anthropic => Ok(LLMProvider::Anthropic),
        ProviderType::Gemini => Ok(LLMProvider::Gemini),
        ProviderType::AnthropicCompatible => Ok(LLMProvider::AnthropicCompatible {
            base_url: require_field(base_url, "base_url", "anthropic_compatible")?,
            api_key_env: require_field(api_key_env, "api_key_env", "anthropic_compatible")?,
        }),
        ProviderType::OpenAiCompatible => Ok(LLMProvider::OpenAICompatible {
            base_url: require_field(base_url, "base_url", "openai_compatible")?,
            api_key_env: require_field(api_key_env, "api_key_env", "openai_compatible")?,
        }),
    }
}

fn require_field(value: Option<&str>, field: &str, provider: &str) -> Result<String> {
    value
        .map(str::to_string)
        .ok_or_else(|| eyre::eyre!("{provider} provider requires `{field}`"))
}
