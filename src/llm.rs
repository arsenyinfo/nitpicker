use eyre::Result;
use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::message::ToolCall;
use rig::completion::{AssistantContent, CompletionModel, Message};
use rig::providers::{anthropic, gemini, openai};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MAX_COMPLETION_ATTEMPTS: usize = 4;
const BASE_BACKOFF_MS: u64 = 250;
const MAX_BACKOFF_MS: u64 = 5_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Completion {
    pub model: String,
    pub prompt: Message,
    pub preamble: Option<String>,
    pub history: Vec<Message>,
    pub tools: Vec<rig::completion::ToolDefinition>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params: Option<Value>,
}

impl Completion {
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    pub fn tools(mut self, tools: Vec<rig::completion::ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn additional_params(mut self, additional_params: Value) -> Self {
        self.additional_params = Some(additional_params);
        self
    }
}

impl From<Completion> for rig::completion::CompletionRequest {
    fn from(value: Completion) -> Self {
        let chat_history = value
            .history
            .into_iter()
            .chain(std::iter::once(value.prompt))
            .collect::<Vec<_>>();
        rig::completion::CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(chat_history)
                .expect("completion request must include at least one message"),
            preamble: value.preamble,
            documents: Vec::new(),
            tools: value.tools,
            temperature: value.temperature,
            max_tokens: value.max_tokens,
            additional_params: value.additional_params,
            output_schema: None,
            tool_choice: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub enum FinishReason {
    None,
    Stop,
    MaxTokens,
    ToolUse,
    Other(String),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub choice: OneOrMany<AssistantContent>,
    pub finish_reason: FinishReason,
    pub output_tokens: u64,
}

impl CompletionResponse {
    pub fn message(&self) -> Message {
        Message::Assistant {
            id: None,
            content: self.choice.clone(),
        }
    }

    pub fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        if self.finish_reason != FinishReason::ToolUse {
            return None;
        }
        let calls = self
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if calls.is_empty() { None } else { Some(calls) }
    }

    pub fn text(&self) -> String {
        self.choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text().to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub trait LLMClient: Send + Sync {
    fn completion(
        &self,
        completion: Completion,
    ) -> impl Future<Output = Result<CompletionResponse>> + Send;

    fn into_arc(self) -> Arc<dyn LLMClientDyn>
    where
        Self: Sized + 'static,
    {
        Arc::new(self)
    }
}

pub trait LLMClientDyn: Send + Sync {
    fn completion(
        &self,
        completion: Completion,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse>> + Send + '_>>;
}

impl<T: LLMClient> LLMClientDyn for T {
    fn completion(
        &self,
        completion: Completion,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse>> + Send + '_>> {
        Box::pin(LLMClient::completion(self, completion))
    }
}

impl LLMClient for Box<dyn LLMClientDyn> {
    fn completion(
        &self,
        completion: Completion,
    ) -> impl Future<Output = Result<CompletionResponse>> + Send {
        async move { (**self).completion(completion).await }
    }
}

pub struct RetryingLLM<C> {
    inner: C,
}

pub trait WithRetryExt: Sized {
    fn with_retry(self) -> RetryingLLM<Self> {
        RetryingLLM { inner: self }
    }
}

impl<T: LLMClient> WithRetryExt for T {}

impl<C: LLMClient> LLMClient for RetryingLLM<C> {
    fn completion(
        &self,
        completion: Completion,
    ) -> impl Future<Output = Result<CompletionResponse>> + Send {
        async move {
            let mut attempt = 0usize;
            loop {
                attempt += 1;
                match self.inner.completion(completion.clone()).await {
                    Ok(response) => return Ok(response),
                    Err(err) => {
                        if is_client_error(&err) || attempt >= MAX_COMPLETION_ATTEMPTS {
                            return Err(err);
                        }
                        let backoff = jittered_backoff(attempt);
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }
    }
}

fn is_client_error(err: &eyre::Report) -> bool {
    let msg = err.to_string();
    msg.contains(" 400") || msg.contains(" 401") || msg.contains(" 403") || msg.contains(" 404")
}

fn jittered_backoff(attempt: usize) -> Duration {
    let exp = 2u64.saturating_pow((attempt - 1) as u32);
    let base = (BASE_BACKOFF_MS * exp).min(MAX_BACKOFF_MS);
    let jitter = jitter_factor();
    let jittered = (base as f64 * jitter).round() as u64;
    Duration::from_millis(jittered.max(1))
}

fn jitter_factor() -> f64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let scaled = (nanos % 1000) as f64 / 1000.0;
    0.5 + scaled
}

pub enum LLMProvider {
    Anthropic,
    Gemini,
    AnthropicCompatible {
        base_url: String,
        api_key_env: String,
    },
    OpenAICompatible {
        base_url: String,
        api_key_env: String,
    },
}

impl LLMProvider {
    pub fn client_from_env(&self) -> Result<Box<dyn LLMClientDyn>> {
        match self {
            LLMProvider::Anthropic => Ok(Box::new(anthropic::Client::from_env())),
            LLMProvider::Gemini => Ok(Box::new(gemini::Client::from_env())),
            LLMProvider::AnthropicCompatible {
                base_url,
                api_key_env,
            } => {
                let api_key = std::env::var(api_key_env)
                    .map_err(|_| eyre::eyre!("missing env var {api_key_env}"))?;
                let client = anthropic::Client::builder()
                    .api_key(api_key)
                    .base_url(base_url)
                    .build()?;
                Ok(Box::new(client))
            }
            LLMProvider::OpenAICompatible {
                base_url,
                api_key_env,
            } => {
                let api_key = std::env::var(api_key_env)
                    .map_err(|_| eyre::eyre!("missing env var {api_key_env}"))?;
                let client = openai::CompletionsClient::builder()
                    .api_key(&api_key)
                    .base_url(base_url)
                    .build()?;
                Ok(Box::new(client))
            }
        }
    }
}

/// Create a Gemini client that routes through the local OAuth proxy
pub fn create_gemini_client_with_proxy(
    proxy_url: &str,
) -> Result<std::sync::Arc<dyn LLMClientDyn>> {
    // Build a Gemini client with the proxy URL as the base URL
    // The API key doesn't matter for OAuth proxy, but is required by the builder
    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_else(|_| "oauth-proxy".to_string());
    let client = gemini::Client::builder()
        .api_key(api_key)
        .base_url(proxy_url)
        .build()?;
    Ok(client.with_retry().into_arc())
}

impl LLMClient for anthropic::Client {
    fn completion(
        &self,
        completion: Completion,
    ) -> impl Future<Output = Result<CompletionResponse>> + Send {
        async move {
            let model_name = completion.model.clone();
            let mut request: rig::completion::CompletionRequest = completion.into();
            request.model = Some(model_name.clone());
            let model = self.completion_model(model_name);
            let response = model.completion(request).await?;
            let mut finish_reason = response
                .raw_response
                .stop_reason
                .clone()
                .map(|reason| match reason.as_str() {
                    "end_turn" => FinishReason::Stop,
                    "max_tokens" => FinishReason::MaxTokens,
                    "tool_use" => FinishReason::ToolUse,
                    other => FinishReason::Other(other.to_string()),
                })
                .unwrap_or(FinishReason::None);
            if response
                .choice
                .iter()
                .any(|content| matches!(content, AssistantContent::ToolCall(_)))
            {
                finish_reason = FinishReason::ToolUse;
            }
            Ok(CompletionResponse {
                choice: response.choice,
                finish_reason,
                output_tokens: response.usage.output_tokens,
            })
        }
    }
}

impl LLMClient for gemini::Client {
    fn completion(
        &self,
        completion: Completion,
    ) -> impl Future<Output = Result<CompletionResponse>> + Send {
        async move {
            let model_name = completion.model.clone();
            let params = GeminiAdditionalParams::from_completion(&completion);
            let mut request: rig::completion::CompletionRequest = completion.into();
            request.model = Some(model_name.clone());
            request.additional_params = Some(serde_json::to_value(params)?);
            let model = self.completion_model(model_name);
            let response = model.completion(request).await?;
            let mut finish_reason = response
                .raw_response
                .candidates
                .first()
                .and_then(|candidate| candidate.finish_reason.clone())
                .map(map_gemini_finish_reason)
                .unwrap_or(FinishReason::None);
            if response
                .choice
                .iter()
                .any(|content| matches!(content, AssistantContent::ToolCall(_)))
            {
                finish_reason = FinishReason::ToolUse;
            }
            Ok(CompletionResponse {
                choice: response.choice,
                finish_reason,
                output_tokens: response.usage.output_tokens,
            })
        }
    }
}

impl LLMClient for openai::CompletionsClient {
    fn completion(
        &self,
        completion: Completion,
    ) -> impl Future<Output = Result<CompletionResponse>> + Send {
        async move {
            let model_name = completion.model.clone();
            let mut request: rig::completion::CompletionRequest = completion.into();
            request.model = Some(model_name.clone());
            let model = self.completion_model(model_name);
            let response = model.completion(request).await?;
            let mut finish_reason = response
                .raw_response
                .choices
                .first()
                .map(|choice| match choice.finish_reason.as_str() {
                    "stop" => FinishReason::Stop,
                    "length" => FinishReason::MaxTokens,
                    "tool_calls" => FinishReason::ToolUse,
                    other => FinishReason::Other(other.to_string()),
                })
                .unwrap_or(FinishReason::None);
            if response
                .choice
                .iter()
                .any(|content| matches!(content, AssistantContent::ToolCall(_)))
            {
                finish_reason = FinishReason::ToolUse;
            }
            Ok(CompletionResponse {
                choice: response.choice,
                finish_reason,
                output_tokens: response.usage.output_tokens,
            })
        }
    }
}

#[derive(Debug, Serialize)]
struct GeminiAdditionalParams {
    generation_config: Option<GenerationConfig>,
}

impl GeminiAdditionalParams {
    fn from_completion(completion: &Completion) -> Self {
        let config = GenerationConfig {
            temperature: completion.temperature,
            max_output_tokens: completion.max_tokens.map(|value| value as i32),
        };
        Self {
            generation_config: Some(config),
        }
    }
}

#[derive(Debug, Serialize, Default)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
}

fn map_gemini_finish_reason(
    reason: gemini::completion::gemini_api_types::FinishReason,
) -> FinishReason {
    use gemini::completion::gemini_api_types::FinishReason as GeminiFinishReason;
    match reason {
        GeminiFinishReason::Stop => FinishReason::Stop,
        GeminiFinishReason::MaxTokens => FinishReason::MaxTokens,
        GeminiFinishReason::FinishReasonUnspecified => FinishReason::None,
        other => FinishReason::Other(format!("{other:?}")),
    }
}
