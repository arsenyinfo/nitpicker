use crate::agent::{
    AgentConfig, AgentDepth, AgentProgress, add_spawn_subagent_tool, run_agent,
};
use crate::config::{
    AggregatorConfig, Config, FallbackConfig, ProviderType, ReviewerConfig,
};
use crate::llm::{Completion, FinishReason, LLMClient, LLMClientDyn, LLMProvider, WithRetryExt};
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
use tracing::{info, warn};

/// A single build recipe for an LLM client. Used to lazily construct
/// primary + fallback clients without materializing them until needed.
pub struct ClientAttempt {
    pub label: String,
    pub model: String,
    pub max_tokens: Option<u64>,
    provider: ProviderType,
    base_url: Option<String>,
    api_key_env: Option<String>,
    use_oauth: bool,
}

impl ClientAttempt {
    pub fn build(
        &self,
        gemini_proxy: Option<&crate::gemini_proxy::GeminiProxyClient>,
    ) -> Result<Arc<dyn LLMClientDyn>> {
        if self.provider.is_gemini() && self.use_oauth {
            let proxy_url = gemini_proxy
                .map(|p| p.base_url())
                .ok_or_else(|| eyre::eyre!("Gemini proxy required for OAuth but not available"))?;
            info!("Using Gemini OAuth via proxy at {}", proxy_url);
            return crate::llm::create_gemini_client_with_proxy(&proxy_url);
        }
        Ok(provider_from_fields(
            &self.provider,
            self.base_url.as_deref(),
            self.api_key_env.as_deref(),
        )?
        .client_from_env()?
        .with_retry()
        .into_arc())
    }
}

/// Produce primary + fallback client attempts for a reviewer, in order.
pub fn reviewer_attempts(reviewer: &ReviewerConfig) -> Vec<ClientAttempt> {
    let mut attempts = Vec::with_capacity(1 + reviewer.fallbacks.len());
    attempts.push(ClientAttempt {
        label: "primary".to_string(),
        model: reviewer.model.clone(),
        max_tokens: None,
        provider: reviewer.provider.clone(),
        base_url: reviewer.base_url.clone(),
        api_key_env: reviewer.api_key_env.clone(),
        use_oauth: reviewer.use_oauth(),
    });
    for (idx, fb) in reviewer.fallbacks.iter().enumerate() {
        attempts.push(fallback_attempt(idx, fb));
    }
    attempts
}

/// Produce primary + fallback client attempts for the aggregator, in order.
pub fn aggregator_attempts(agg: &AggregatorConfig) -> Vec<ClientAttempt> {
    let mut attempts = Vec::with_capacity(1 + agg.fallbacks.len());
    attempts.push(ClientAttempt {
        label: "primary".to_string(),
        model: agg.model.clone(),
        max_tokens: agg.max_tokens,
        provider: agg.provider.clone(),
        base_url: agg.base_url.clone(),
        api_key_env: agg.api_key_env.clone(),
        use_oauth: agg.use_oauth(),
    });
    for (idx, fb) in agg.fallbacks.iter().enumerate() {
        attempts.push(fallback_attempt(idx, fb));
    }
    attempts
}

fn fallback_attempt(idx: usize, fb: &FallbackConfig) -> ClientAttempt {
    ClientAttempt {
        label: format!("fallback-{}", idx + 1),
        model: fb.model.clone(),
        max_tokens: fb.max_tokens,
        provider: fb.provider.clone(),
        base_url: fb.base_url.clone(),
        api_key_env: fb.api_key_env.clone(),
        use_oauth: fb.use_oauth(),
    }
}

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

    // Check if we need to start the Gemini proxy for any OAuth-enabled reviewer or fallback
    let gemini_proxy: Option<Arc<crate::gemini_proxy::GeminiProxyClient>> = if config
        .reviewer
        .iter()
        .any(|r| r.needs_gemini_proxy())
        || config.aggregator.needs_gemini_proxy()
    {
        info!("Starting Gemini proxy for OAuth authentication...");
        Some(Arc::new(
            crate::gemini_proxy::GeminiProxyClient::new().await?,
        ))
    } else {
        None
    };

    for reviewer in &config.reviewer {
        let tools_map = tools.clone();
        let repo = repo.to_path_buf();
        let name = reviewer.name.clone();
        let attempts = reviewer_attempts(reviewer);
        let system_prompt = system_prompt.to_string();
        let initial_message = initial_message.clone();
        let proxy = gemini_proxy.clone();
        info!(reviewer = %name, "spawning agent");

        let pb = mp.add(ProgressBar::new_spinner());
        pb.set_style(spinner_style.clone());
        pb.set_prefix(name.clone());
        pb.set_message("reviewing…");
        pb.enable_steady_tick(Duration::from_millis(80));

        let done = done_style.clone();
        let handle: JoinHandle<(String, Result<String>)> = tokio::spawn(async move {
            let mut last_err: Option<eyre::Report> = None;
            for (idx, attempt) in attempts.iter().enumerate() {
                let subagent_counter = Arc::new(AtomicUsize::new(0));
                let attempt_label = attempt.label.clone();
                let attempt_model = attempt.model.clone();
                if idx > 0 {
                    pb.set_message(format!(
                        "reviewing… ({attempt_label}: {attempt_model})"
                    ));
                    info!(
                        reviewer = %name,
                        attempt = %attempt_label,
                        model = %attempt_model,
                        "falling back to next model"
                    );
                }
                let client = match attempt.build(proxy.as_deref()) {
                    Ok(c) => c,
                    Err(err) => {
                        warn!(
                            reviewer = %name,
                            attempt = %attempt_label,
                            error = %err,
                            "failed to build client"
                        );
                        last_err = Some(err);
                        continue;
                    }
                };
                let agent_config = AgentConfig {
                    name: name.clone(),
                    model: attempt.model.clone(),
                    max_turns,
                    system_prompt: system_prompt.clone(),
                    client,
                    depth: AgentDepth::TopLevel,
                    terminal_tools: Vec::new(),
                    empty_response_nudge: None,
                    max_empty_responses: 0,
                    subagent_counter,
                    progress: if verbose {
                        None
                    } else {
                        let progress_pb = pb.clone();
                        Some(Arc::new(move |progress: AgentProgress| {
                            progress_pb.set_message(format!(
                                "reviewing… ({} turns, {} tool calls, {} subagents)",
                                progress.turns, progress.tool_calls, progress.subagents_spawned
                            ));
                        }))
                    },
                };
                let start = Instant::now();
                let result = run_agent(agent_config, &initial_message, &tools_map, &repo).await;
                let elapsed = start.elapsed().as_secs();
                match result {
                    Ok(r) => {
                        pb.set_style(done.clone());
                        let model_note = if idx == 0 {
                            String::new()
                        } else {
                            format!(", via {attempt_label}: {attempt_model}")
                        };
                        pb.finish_with_message(format!(
                            "✓ done ({elapsed}s, {} turns, {} tool calls, {} subagents, {} output tokens{model_note})",
                            r.turns, r.tool_calls, r.subagents_spawned, r.total_output_tokens
                        ));
                        return (name, Ok(r.text));
                    }
                    Err(err) => {
                        warn!(
                            reviewer = %name,
                            attempt = %attempt_label,
                            model = %attempt_model,
                            error = %err,
                            "agent attempt failed"
                        );
                        last_err = Some(err);
                    }
                }
            }
            let err = last_err
                .unwrap_or_else(|| eyre::eyre!("no client attempts available for reviewer"));
            pb.set_style(done.clone());
            pb.finish_with_message(format!("✗ failed: {err}"));
            (name, Err(err))
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
    let agg_attempts = aggregator_attempts(agg);
    let aggregator_preamble = mode.aggregator_preamble().to_string();
    let (text, finish_reason) = run_aggregator_with_fallback(
        &agg_attempts,
        gemini_proxy.as_deref(),
        &aggregator_preamble,
        &reduce_prompt,
        &pb_agg,
    )
    .await?;

    pb_agg.set_style(done_style);
    if finish_reason == FinishReason::ToolUse {
        pb_agg.finish_with_message("✗ failed: unexpected tool call");
        return Err(eyre::eyre!("aggregator returned tool calls unexpectedly"));
    }
    pb_agg.finish_with_message("✓ done");
    Ok(text)
}

/// Run a single non-agentic completion through primary + fallback aggregator clients.
/// Returns `(text, finish_reason)` from the first successful attempt.
pub async fn run_aggregator_with_fallback(
    attempts: &[ClientAttempt],
    gemini_proxy: Option<&crate::gemini_proxy::GeminiProxyClient>,
    preamble: &str,
    user_prompt: &str,
    pb: &ProgressBar,
) -> Result<(String, FinishReason)> {
    let mut last_err: Option<eyre::Report> = None;
    for (idx, attempt) in attempts.iter().enumerate() {
        if idx > 0 {
            pb.set_message(format!(
                "synthesizing… ({}: {})",
                attempt.label, attempt.model
            ));
            info!(
                attempt = %attempt.label,
                model = %attempt.model,
                "aggregator falling back to next model"
            );
        }
        let client = match attempt.build(gemini_proxy) {
            Ok(c) => c,
            Err(err) => {
                warn!(
                    attempt = %attempt.label,
                    error = %err,
                    "failed to build aggregator client"
                );
                last_err = Some(err);
                continue;
            }
        };
        let completion = Completion {
            model: attempt.model.clone(),
            prompt: Message::user(user_prompt.to_string()),
            preamble: Some(preamble.to_string()),
            history: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: attempt.max_tokens.or(Some(8192)),
            additional_params: None,
        };
        match client.completion(completion).await {
            Ok(response) => return Ok((response.text(), response.finish_reason)),
            Err(err) => {
                warn!(
                    attempt = %attempt.label,
                    model = %attempt.model,
                    error = %err,
                    "aggregator attempt failed"
                );
                last_err = Some(err);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| eyre::eyre!("no aggregator attempts available")))
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

fn provider_from_fields(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml_str: &str) -> Config {
        toml::from_str(toml_str).expect("valid test config")
    }

    #[test]
    fn reviewer_attempts_primary_only_when_no_fallbacks() {
        let cfg = parse(
            r#"
[aggregator]
model = "m"
provider = "anthropic"

[[reviewer]]
name = "r"
model = "primary-model"
provider = "anthropic"
"#,
        );
        let attempts = reviewer_attempts(&cfg.reviewer[0]);
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].label, "primary");
        assert_eq!(attempts[0].model, "primary-model");
    }

    #[test]
    fn reviewer_attempts_preserves_fallback_order() {
        let cfg = parse(
            r#"
[aggregator]
model = "m"
provider = "anthropic"

[[reviewer]]
name = "r"
model = "primary-model"
provider = "anthropic"

[[reviewer.fallbacks]]
model = "first-fallback"
provider = "anthropic"

[[reviewer.fallbacks]]
model = "second-fallback"
provider = "anthropic"
"#,
        );
        let attempts = reviewer_attempts(&cfg.reviewer[0]);
        assert_eq!(attempts.len(), 3);
        assert_eq!(attempts[0].label, "primary");
        assert_eq!(attempts[0].model, "primary-model");
        assert_eq!(attempts[1].label, "fallback-1");
        assert_eq!(attempts[1].model, "first-fallback");
        assert_eq!(attempts[2].label, "fallback-2");
        assert_eq!(attempts[2].model, "second-fallback");
    }

    #[test]
    fn aggregator_attempts_preserves_fallback_order() {
        let cfg = parse(
            r#"
[aggregator]
model = "agg-primary"
provider = "anthropic"
max_tokens = 4096

[[aggregator.fallbacks]]
model = "agg-fallback"
provider = "anthropic"
max_tokens = 2048

[[reviewer]]
name = "r"
model = "m"
provider = "anthropic"
"#,
        );
        let attempts = aggregator_attempts(&cfg.aggregator);
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].label, "primary");
        assert_eq!(attempts[0].model, "agg-primary");
        assert_eq!(attempts[0].max_tokens, Some(4096));
        assert_eq!(attempts[1].label, "fallback-1");
        assert_eq!(attempts[1].model, "agg-fallback");
        assert_eq!(attempts[1].max_tokens, Some(2048));
    }
}
