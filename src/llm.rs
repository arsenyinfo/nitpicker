use eyre::{Result, WrapErr};
use rig_core::OneOrMany;
use rig_core::client::CompletionClient;
use rig_core::completion::CompletionError;
use rig_core::completion::message::ToolCall;
use rig_core::completion::message::ToolChoice;
use rig_core::completion::{AssistantContent, CompletionModel, Message};
use rig_core::providers::{anthropic, gemini, openai, openrouter};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;
use tracing::warn;

const MAX_COMPLETION_ATTEMPTS: usize = 4;
const RATE_LIMIT_MAX_COMPLETION_ATTEMPTS: usize = 8;
const BASE_BACKOFF_MS: u64 = 250;
const MAX_BACKOFF_MS: u64 = 5_000;
const RATE_LIMIT_BASE_BACKOFF_MS: u64 = 5_000;
const RATE_LIMIT_MAX_BACKOFF_MS: u64 = 60_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Completion {
    pub model: String,
    pub prompt: Message,
    pub preamble: Option<String>,
    pub history: Vec<Message>,
    pub tools: Vec<rig_core::completion::ToolDefinition>,
    pub tool_choice: Option<ToolChoice>,
    pub max_tokens: Option<u64>,
    pub additional_params: Option<Value>,
}

impl Completion {
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    pub fn tools(mut self, tools: Vec<rig_core::completion::ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
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

impl From<Completion> for rig_core::completion::CompletionRequest {
    fn from(value: Completion) -> Self {
        let chat_history = value
            .history
            .into_iter()
            .chain(std::iter::once(value.prompt))
            .collect::<Vec<_>>();
        rig_core::completion::CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(chat_history)
                .expect("completion request must include at least one message"),
            preamble: value.preamble,
            documents: Vec::new(),
            tools: value.tools,
            temperature: None,
            max_tokens: value.max_tokens,
            additional_params: value.additional_params,
            output_schema: None,
            tool_choice: value.tool_choice,
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
    pub usage: TokenUsage,
    pub selected_model: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

impl TokenUsage {
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens.saturating_add(output_tokens),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConversationUsageWindow {
    compact_threshold: Option<u64>,
    usage: TokenUsage,
}

impl ConversationUsageWindow {
    pub fn new(compact_threshold: Option<u64>) -> Self {
        Self {
            compact_threshold,
            usage: TokenUsage::default(),
        }
    }

    pub fn should_compact(&self) -> bool {
        self.compact_threshold
            .map(|threshold| self.usage.total_tokens >= threshold)
            .unwrap_or(false)
    }

    pub fn usage(&self) -> TokenUsage {
        self.usage
    }

    pub fn record(&mut self, usage: TokenUsage) {
        // input_tokens from each API response is the full context size for that call,
        // not just the new tokens — replace rather than accumulate so should_compact()
        // compares against the actual current context size
        self.usage = usage;
    }

    pub fn reset(&mut self) {
        self.usage = TokenUsage::default();
    }
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
        // join the raw text blocks first, then strip once: a think block that spans
        // (or a truncated one that runs to EOF) is judged against the whole text, and an
        // all-reasoning response collapses to "" so callers' is_empty() checks fire.
        let raw = self
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text().to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        strip_think_blocks(&raw)
    }
}

// some providers (notably MiniMax/GLM/DeepSeek via OpenRouter) emit chain-of-thought
// inline as <think>...</think> in the message content rather than in a structured
// reasoning field rig can drop. A depth-tracking scanner (not a single regex, which can't
// match balanced nesting) keeps text only at nesting depth 0, so:
//   - `<think...>` / `</think...>` are matched case-insensitively, tolerating padded tags
//     like `<think >` (a regex whose open tag lacked `\s*` would leak their bodies)
//   - nested tags can't leak: inner reasoning stays inside depth > 0
//   - an unterminated block is dropped through end-of-text rather than leaking its body
//   - a stray closing tag at depth 0 is dropped
// note: this is content-wide, so a review that legitimately quotes a <think> tag in a code
// snippet will have it stripped too — an accepted tradeoff for clean aggregation.
fn strip_think_blocks(text: &str) -> String {
    // match `<think` (or `</think` when `close`) + optional whitespace + `>` at the start of
    // `s`, case-insensitively; return the matched byte length. `<thinking>` is not a tag (the
    // char after `think` must be whitespace or `>`).
    fn match_think_tag(s: &str, close: bool) -> Option<usize> {
        let prefix = if close { "</think" } else { "<think" };
        if !s.get(..prefix.len())?.eq_ignore_ascii_case(prefix) {
            return None;
        }
        let after = &s[prefix.len()..];
        let ws = after.len() - after.trim_start().len();
        after[ws..]
            .starts_with('>')
            .then_some(prefix.len() + ws + 1)
    }

    let mut out = String::with_capacity(text.len());
    let mut depth: usize = 0;
    let mut rest = text;
    while !rest.is_empty() {
        if rest.starts_with('<') {
            if let Some(len) = match_think_tag(rest, false) {
                depth += 1;
                rest = &rest[len..];
                continue;
            }
            if let Some(len) = match_think_tag(rest, true) {
                depth = depth.saturating_sub(1);
                rest = &rest[len..];
                continue;
            }
        }
        let ch = rest.chars().next().expect("rest is non-empty");
        if depth == 0 {
            out.push(ch);
        }
        rest = &rest[ch.len_utf8()..];
    }
    out.trim().to_string()
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
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        (**self).completion(completion).await
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
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            match self.inner.completion(completion.clone()).await {
                Ok(response) => {
                    if response.text().is_empty() && response.tool_calls().is_none() {
                        if attempt >= MAX_COMPLETION_ATTEMPTS {
                            eyre::bail!("model returned empty response after {attempt} attempts");
                        }
                        let backoff = jittered_backoff(attempt, BASE_BACKOFF_MS, MAX_BACKOFF_MS);
                        warn!(
                            model = %completion.model,
                            attempt,
                            max_attempts = MAX_COMPLETION_ATTEMPTS,
                            backoff_ms = backoff.as_millis(),
                            "retrying after empty model response"
                        );
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    return Ok(response);
                }
                Err(err) => {
                    let policy = retry_policy(&err);
                    if !policy.retry || attempt >= policy.max_attempts {
                        return Err(err);
                    }
                    let backoff =
                        jittered_backoff(attempt, policy.base_backoff_ms, policy.max_backoff_ms);
                    warn!(
                        model = %completion.model,
                        attempt,
                        max_attempts = policy.max_attempts,
                        backoff_ms = backoff.as_millis(),
                        error = %err,
                        "retrying model completion after error"
                    );
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }
}

/// randomly alternates between models within a single agentic loop — see https://xbow.com/blog/alloy-agents
///
/// inner clients must already be wrapped with retry (e.g. via `.with_retry()`); AlloyClient does not add retry itself
pub struct AlloyClient {
    slots: Vec<(Arc<dyn LLMClientDyn>, String)>,
}

impl AlloyClient {
    pub fn new(slots: Vec<(Arc<dyn LLMClientDyn>, String)>) -> Result<Self> {
        if slots.is_empty() {
            eyre::bail!("AlloyClient requires at least one slot");
        }
        Ok(Self { slots })
    }

    fn pick_idx(&self) -> usize {
        use rand::RngExt;
        rand::rng().random_range(0..self.slots.len())
    }
}

impl LLMClientDyn for AlloyClient {
    fn completion(
        &self,
        mut completion: Completion,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse>> + Send + '_>> {
        let (client, model) = &self.slots[self.pick_idx()];
        completion.model = model.clone();
        let client = Arc::clone(client);
        let model = model.clone();
        Box::pin(async move {
            let mut response = client.completion(completion).await?;
            response.selected_model = Some(model);
            Ok(response)
        })
    }
}

/// Run a single completion under a concurrency permit. The permit is held only for the duration of
/// this one call and released immediately after — never across a subagent spawn — so callers may
/// block on acquire without risking deadlock. This is the single chokepoint that bounds account-wide
/// in-flight LLM calls; route every concurrent completion (agent turns, compaction) through it.
pub async fn throttled_completion(
    semaphore: &Semaphore,
    client: &Arc<dyn LLMClientDyn>,
    completion: Completion,
) -> Result<CompletionResponse> {
    let _permit = semaphore.acquire().await.expect("llm semaphore closed");
    client.completion(completion).await
}

struct RetryPolicy {
    retry: bool,
    max_attempts: usize,
    base_backoff_ms: u64,
    max_backoff_ms: u64,
}

fn retry_policy(err: &eyre::Report) -> RetryPolicy {
    if is_rate_limit_error(err) {
        return RetryPolicy {
            retry: true,
            max_attempts: RATE_LIMIT_MAX_COMPLETION_ATTEMPTS,
            base_backoff_ms: RATE_LIMIT_BASE_BACKOFF_MS,
            max_backoff_ms: RATE_LIMIT_MAX_BACKOFF_MS,
        };
    }

    if is_non_retryable_client_error(err) {
        return RetryPolicy {
            retry: false,
            max_attempts: 0,
            base_backoff_ms: 0,
            max_backoff_ms: 0,
        };
    }

    RetryPolicy {
        retry: true,
        max_attempts: MAX_COMPLETION_ATTEMPTS,
        base_backoff_ms: BASE_BACKOFF_MS,
        max_backoff_ms: MAX_BACKOFF_MS,
    }
}

/// HTTP status codes the retry/refresh classifiers key on, paired with their canonical reason
/// phrase. The phrase lets us recognize the plain-text `"<code> <reason>"` status-line form
/// (`401 Unauthorized`, `429 Too Many Requests`) that carries no JSON status key.
const STATUS_REASONS: &[(u16, &str)] = &[
    (400, "bad request"),
    (401, "unauthorized"),
    (402, "payment required"),
    (403, "forbidden"),
    (404, "not found"),
    (429, "too many requests"),
    (500, "internal server error"),
    (502, "bad gateway"),
    (503, "service unavailable"),
    (504, "gateway timeout"),
];

/// How far back (in bytes) to scan for a status key before a candidate number. Comfortably covers
/// `"statusCode": ` even with extra spacing, while staying local enough that an unrelated
/// `code`/`status` field elsewhere in the body doesn't bleed in.
const STATUS_KEY_WINDOW: usize = 24;

/// True if `status` (an HTTP status code) appears in `msg` as a genuine status reference: a
/// standalone number (not part of a longer digit run) that is *also* in an HTTP-status context —
/// either immediately followed by its canonical reason phrase (`401 Unauthorized`) or preceded
/// within [`STATUS_KEY_WINDOW`] by a `status`/`code` key (covering `"statusCode": 401`, `:401`,
/// `Invalid status code 401`, ...). The context requirement keeps incidental standalone numbers in
/// a raw provider body — `400 tokens`, `trace 404`, `req_402abc` — from being misread as the
/// response status. Provider errors surface the status only inside the raw body, whose punctuation
/// varies (spaced `"statusCode": 401` vs compact `"statusCode":401`), so we can't rely on fixed
/// delimiters; the trade-off is that a status carrying neither a nearby key nor a reason phrase is
/// not recognized (rare in practice, and recoverable — at worst a retried/failed request).
pub(crate) fn mentions_http_status(msg: &str, status: u16) -> bool {
    let lower = msg.to_ascii_lowercase();
    let needle = status.to_string();
    let reason = STATUS_REASONS
        .iter()
        .find(|(code, _)| *code == status)
        .map(|(_, phrase)| *phrase);
    let bytes = lower.as_bytes();
    let mut from = 0;
    while let Some(rel) = lower[from..].find(&needle) {
        let start = from + rel;
        let end = start + needle.len();
        let prev_digit = start > 0 && bytes[start - 1].is_ascii_digit();
        let next_digit = end < bytes.len() && bytes[end].is_ascii_digit();
        if !prev_digit && !next_digit && status_in_context(&lower, start, end, reason) {
            return true;
        }
        from = start + 1;
    }
    false
}

/// Whether the standalone number at `start..end` in `lower` (already lowercased) sits in an
/// HTTP-status context: followed by its reason phrase, or preceded within [`STATUS_KEY_WINDOW`] by a
/// `status`/`code` key. `start`/`end` are byte offsets on char boundaries (the needle is ASCII); the
/// preceding-window start is floored to a boundary so a multibyte char in the body can't panic the
/// slice.
fn status_in_context(lower: &str, start: usize, end: usize, reason: Option<&str>) -> bool {
    if let Some(reason) = reason {
        if lower[end..].trim_start().starts_with(reason) {
            return true;
        }
    }
    let mut window_start = start.saturating_sub(STATUS_KEY_WINDOW);
    while !lower.is_char_boundary(window_start) {
        window_start += 1;
    }
    // `statuscode` is the lowercased compact `statusCode` (no separator), where neither `status`
    // nor `code` is a whole word on its own, so it needs its own key.
    key_word_present(lower, window_start, start, "status")
        || key_word_present(lower, window_start, start, "code")
        || key_word_present(lower, window_start, start, "statuscode")
}

/// Whether `key` appears as a whole word within `lower[lo..hi]`. The left edge must be the string
/// start or a non-`[a-z0-9_]` byte; the right edge must be the string end or a non-`[a-z0-9]` byte
/// (`_` is allowed on the right so `status_code` still matches via the `status` key, while
/// `decode`/`encode`/`unicode`/`error_code`/`codec`/`statuslike` do NOT count as keys). Boundary
/// checks read `lower`'s absolute bytes, so a word split by the `[lo..hi]` window edge is judged
/// against its real neighbours. `key` is ASCII; `lo`/`hi` are on char boundaries.
fn key_word_present(lower: &str, lo: usize, hi: usize, key: &str) -> bool {
    let bytes = lower.as_bytes();
    let region = &lower[lo..hi];
    let mut from = 0;
    while let Some(rel) = region[from..].find(key) {
        let abs = lo + from + rel;
        let left_ok = abs == 0 || {
            let b = bytes[abs - 1];
            !b.is_ascii_alphanumeric() && b != b'_'
        };
        let after = abs + key.len();
        let right_ok = after >= bytes.len() || !bytes[after].is_ascii_alphanumeric();
        if left_ok && right_ok {
            return true;
        }
        from += rel + 1;
    }
    false
}

/// Permanent error *types* that direct Anthropic/OpenAI bodies carry instead of a numeric status.
/// rig flattens a non-2xx response to `ProviderError(<raw body>)` with the HTTP status dropped, so
/// for those providers the numeric-status matchers never fire — these strings are the only signal.
/// `insufficient_quota` (out of credits) is permanent despite arriving as HTTP 429: retrying never
/// helps. Auth/permission `403`-class types are deliberately included here, *not* in the rate-limit
/// set. All lowercase; matched as substrings on the lowercased chain.
const NON_RETRYABLE_ERROR_TYPES: &[&str] = &[
    "authentication_error",
    "invalid_api_key",
    "permission_error",
    "permission_denied",
    "invalid_request_error",
    "not_found_error",
    "context_length_exceeded",
    "insufficient_quota",
];

/// Transient error *types* (overload / throttling) that warrant the rate-limit backoff policy.
const RATE_LIMIT_ERROR_TYPES: &[&str] = &[
    "rate_limit_error",
    "rate_limit_exceeded",
    "overloaded_error",
];

fn is_non_retryable_client_error(err: &eyre::Report) -> bool {
    // Walk the whole chain: provider clients map non-2xx to a `ProviderError` carrying the raw
    // response body, then `.wrap_err_with(...)` adds a top-level context. `err.to_string()` renders
    // only that context, so the status code would be invisible; `{err:#}` joins the full chain.
    let msg = format!("{err:#}");
    // a 5xx response takes precedence: even when a 4xx is nested in the body (e.g. an upstream
    // `"code": 403` inside a 502 envelope), the response itself is a retryable server error, so we
    // must not classify it as a permanent client error. Cover the full registered 5xx range; the
    // JSON `status`/`code`-key form is matched even for codes without a reason phrase here.
    if [500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511]
        .iter()
        .any(|&status| mentions_http_status(&msg, status))
    {
        return false;
    }
    if [400, 401, 402, 403, 404]
        .iter()
        .any(|&status| mentions_http_status(&msg, status))
    {
        return true;
    }
    let lower = msg.to_ascii_lowercase();
    NON_RETRYABLE_ERROR_TYPES.iter().any(|t| lower.contains(t))
}

fn is_rate_limit_error(err: &eyre::Report) -> bool {
    // Same reasoning as `is_non_retryable_client_error`: walk the full chain so a 429 carried in a
    // wrapped `ProviderError` body still maps to the rate-limit backoff policy. `retry_policy`
    // consults this before `is_non_retryable_client_error`, so a transient type wins any overlap.
    let msg = format!("{err:#}").to_ascii_lowercase();
    mentions_http_status(&msg, 429)
        || msg.contains("rate limit")
        || msg.contains("too many requests")
        || msg.contains("tokens per minute")
        || msg.contains("requests per minute")
        || RATE_LIMIT_ERROR_TYPES.iter().any(|t| msg.contains(t))
}

fn jittered_backoff(attempt: usize, base_backoff_ms: u64, max_backoff_ms: u64) -> Duration {
    let exp = 2u64.saturating_pow((attempt - 1) as u32);
    let base = (base_backoff_ms * exp).min(max_backoff_ms);
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

#[derive(Debug, Deserialize)]
struct OpenRouterErrorEnvelope {
    error: OpenRouterErrorBody,
}

#[derive(Debug, Deserialize)]
struct OpenRouterErrorBody {
    message: String,
    code: Option<u16>,
}

fn normalize_openrouter_completion_error(err: &CompletionError) -> eyre::Report {
    match err {
        CompletionError::ResponseError(msg) => normalize_openrouter_response_error(msg),
        CompletionError::ProviderError(msg) => eyre::eyre!("ProviderError: {msg}"),
        _ => eyre::eyre!("{err}"),
    }
}

fn normalize_openrouter_response_error(msg: &str) -> eyre::Report {
    if msg.contains("no message or tool call") || msg.contains("no choices") {
        return eyre::eyre!("empty response from model (no message or tool call)");
    }

    if let Some(err) = parse_openrouter_error_envelope(msg) {
        return match err.code {
            Some(code) => eyre::eyre!(
                "HttpError: Invalid status code {code} OpenRouter provider error: {}",
                err.message
            ),
            None => eyre::eyre!("ProviderError: {}", err.message),
        };
    }

    eyre::eyre!("ResponseError: {msg}")
}

fn parse_openrouter_error_envelope(msg: &str) -> Option<OpenRouterErrorBody> {
    let body = msg.split_once("response body:")?.1.trim();
    serde_json::from_str::<OpenRouterErrorEnvelope>(body)
        .ok()
        .map(|envelope| envelope.error)
}

fn is_local_base_url(base_url: Option<&str>) -> bool {
    base_url
        .map(|u| u.starts_with("http://localhost") || u.starts_with("http://127.0.0.1"))
        .unwrap_or(false)
}

/// Resolve the API key when its env var is unset: a local base_url needs no real key, otherwise it
/// is a hard error naming the missing var.
fn missing_or_local(key_env: &str, base_url: Option<&str>) -> Result<String> {
    if is_local_base_url(base_url) {
        Ok("local".to_string())
    } else {
        Err(eyre::eyre!("missing env var {key_env}"))
    }
}

pub enum LLMProvider {
    Anthropic {
        base_url: Option<String>,
        api_key_env: Option<String>,
    },
    Gemini {
        base_url: Option<String>,
        api_key_env: Option<String>,
    },
    OpenAi {
        base_url: Option<String>,
        api_key_env: Option<String>,
    },
    OpenRouter {
        api_key_env: String,
    },
}

impl LLMProvider {
    pub fn client_from_env(&self) -> Result<Box<dyn LLMClientDyn>> {
        match self {
            LLMProvider::Anthropic {
                base_url,
                api_key_env,
            } => {
                let key_env = api_key_env.as_deref().unwrap_or("ANTHROPIC_API_KEY");
                let api_key = std::env::var(key_env)
                    .or_else(|_| missing_or_local(key_env, base_url.as_deref()))?;
                let mut builder = anthropic::Client::builder().api_key(api_key);
                if let Some(url) = base_url {
                    builder = builder.base_url(url);
                }
                Ok(Box::new(builder.build()?))
            }
            LLMProvider::Gemini {
                base_url,
                api_key_env,
            } => {
                // An explicit api_key_env overrides the GEMINI_API_KEY → GOOGLE_AI_API_KEY default
                // chain; a local base_url needs no key (mirrors the Anthropic/OpenAi arms).
                let api_key = match api_key_env {
                    Some(key_env) => std::env::var(key_env)
                        .or_else(|_| missing_or_local(key_env, base_url.as_deref()))?,
                    None => std::env::var("GEMINI_API_KEY")
                        .or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))
                        .or_else(|_| {
                            missing_or_local(
                                "GEMINI_API_KEY (or GOOGLE_AI_API_KEY)",
                                base_url.as_deref(),
                            )
                        })?,
                };
                let mut builder = gemini::Client::builder().api_key(api_key);
                if let Some(url) = base_url {
                    builder = builder.base_url(url);
                }
                let client = builder
                    .build()
                    .map_err(|e| eyre::eyre!("failed to create Gemini client: {e}"))?;
                Ok(Box::new(client))
            }
            LLMProvider::OpenAi {
                base_url,
                api_key_env,
            } => {
                let key_env = api_key_env.as_deref().unwrap_or("OPENAI_API_KEY");
                let api_key = std::env::var(key_env)
                    .or_else(|_| missing_or_local(key_env, base_url.as_deref()))?;
                let mut builder = openai::CompletionsClient::builder().api_key(&api_key);
                if let Some(url) = base_url {
                    builder = builder.base_url(url);
                }
                Ok(Box::new(builder.build()?))
            }
            LLMProvider::OpenRouter { api_key_env } => {
                let api_key = std::env::var(api_key_env)
                    .map_err(|_| eyre::eyre!("missing env var {api_key_env}"))?;
                let client = openrouter::Client::builder()
                    .api_key(&api_key)
                    .http_headers(openrouter_headers()?)
                    .build()?;
                Ok(Box::new(client))
            }
        }
    }
}

pub fn openrouter_headers() -> Result<reqwest::header::HeaderMap> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        "HTTP-Referer",
        "https://github.com/arsenyinfo/nitpicker".parse()?,
    );
    headers.insert("X-OpenRouter-Title", "nitpicker".parse()?);
    headers.insert(
        "X-OpenRouter-Categories",
        "cli-agent,programming-app".parse()?,
    );
    Ok(headers)
}

impl LLMClient for openrouter::Client {
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        let model_name = completion.model.clone();
        let mut request: rig_core::completion::CompletionRequest = completion.into();
        request.model = Some(model_name.clone());
        let model = self.completion_model(&model_name);
        let response = model
            .completion(request)
            .await
            .map_err(|e| normalize_openrouter_completion_error(&e))?;
        let mut finish_reason = response
            .raw_response
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
            .map(|reason| match reason {
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
        let usage = response
            .raw_response
            .usage
            .map(|u| TokenUsage::new(u.prompt_tokens as u64, u.completion_tokens as u64))
            .unwrap_or_default();
        Ok(CompletionResponse {
            choice: response.choice,
            finish_reason,
            usage,
            selected_model: Some(model_name),
        })
    }
}

/// Create a Gemini client that routes through the local OAuth proxy
#[cfg(feature = "antigravity")]
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
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        let model_name = completion.model.clone();
        let mut request: rig_core::completion::CompletionRequest = completion.into();
        request.model = Some(model_name.clone());
        let model = self.completion_model(model_name.clone());
        let response = model
            .completion(request)
            .await
            .wrap_err_with(|| format!("Anthropic completion failed for model '{model_name}'"))?;
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
            usage: TokenUsage::new(response.usage.input_tokens, response.usage.output_tokens),
            selected_model: Some(model_name),
        })
    }
}

impl LLMClient for gemini::Client {
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        let model_name = completion.model.clone();
        let params = GeminiAdditionalParams::from_completion(&completion);
        let mut request: rig_core::completion::CompletionRequest = completion.into();
        request.model = Some(model_name.clone());
        request.additional_params = Some(serde_json::to_value(params)?);
        let model = self.completion_model(model_name.clone());
        let response = model
            .completion(request)
            .await
            .wrap_err_with(|| format!("Gemini completion failed for model '{model_name}'"))?;
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
            usage: TokenUsage::new(response.usage.input_tokens, response.usage.output_tokens),
            selected_model: Some(model_name),
        })
    }
}

fn model_needs_max_completion_tokens(model: &str) -> bool {
    model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("gpt-4o")
        || model.starts_with("gpt-4.1")
        || model.starts_with("gpt-4.5")
        || model.starts_with("gpt-5")
}

pub(crate) fn merge_json(mut base: Value, extra: Value) -> Value {
    if let (Some(base_obj), Some(extra_obj)) = (base.as_object_mut(), extra.as_object()) {
        for (k, v) in extra_obj {
            base_obj.insert(k.clone(), v.clone());
        }
        base
    } else {
        extra
    }
}

impl LLMClient for openai::CompletionsClient {
    async fn completion(&self, mut completion: Completion) -> Result<CompletionResponse> {
        // Newer OpenAI models (o1, o3, o4-mini, gpt-4o, gpt-5, etc.) reject the legacy
        // `max_tokens` param and require `max_completion_tokens` instead. Since rig always
        // serializes `max_tokens`, we move the value into `additional_params` and clear it.
        if let Some(max) = completion.max_tokens {
            if model_needs_max_completion_tokens(&completion.model) {
                let extra = serde_json::json!({ "max_completion_tokens": max });
                completion.additional_params = Some(match completion.additional_params.take() {
                    Some(existing) => merge_json(existing, extra),
                    None => extra,
                });
                completion.max_tokens = None;
            }
        }
        let model_name = completion.model.clone();
        let mut request: rig_core::completion::CompletionRequest = completion.into();
        request.model = Some(model_name.clone());
        let model = self.completion_model(model_name.clone());
        let response = model
            .completion(request)
            .await
            .wrap_err_with(|| format!("OpenAI completion failed for model '{model_name}'"))?;
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
            usage: TokenUsage::new(response.usage.input_tokens, response.usage.output_tokens),
            selected_model: Some(model_name),
        })
    }
}

#[derive(Debug, Serialize)]
struct GeminiAdditionalParams {
    generation_config: Option<GenerationConfig>,
}

impl GeminiAdditionalParams {
    fn from_completion(completion: &Completion) -> Self {
        let config = GenerationConfig {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduce how a provider 401 actually reaches the retry layer: rig surfaces the raw
    /// response body as `ProviderError`, and the per-provider `completion` impls wrap it with
    /// `.wrap_err_with(...)`. The status only lives in the source, so the classifier must walk
    /// the chain rather than read `err.to_string()`.
    fn wrapped_provider_error(body: &str) -> eyre::Report {
        let inner = eyre::eyre!("ProviderError: {body}");
        Err::<(), _>(inner)
            .wrap_err("Anthropic completion failed for model 'claude'")
            .unwrap_err()
    }

    #[test]
    fn strips_full_think_block() {
        let raw = "<think>let me reason about this</think>\n\nThe bug is in foo().";
        assert_eq!(strip_think_blocks(raw), "The bug is in foo().");
    }

    #[test]
    fn strips_stray_empty_think_tags() {
        // OpenRouter hoists MiniMax's reasoning into a separate field but leaves the tags.
        let raw = "<think></think>\n## Findings\n- looks fine";
        assert_eq!(strip_think_blocks(raw), "## Findings\n- looks fine");
    }

    #[test]
    fn strips_unbalanced_and_multiline_think() {
        let raw = "intro\n<think>\nstep 1\nstep 2\n</think>verdict: ok</think>";
        assert_eq!(strip_think_blocks(raw), "intro\nverdict: ok");
    }

    #[test]
    fn leaves_non_think_text_untouched() {
        let raw = "no reasoning tags here, just review text";
        assert_eq!(strip_think_blocks(raw), raw);
    }

    #[test]
    fn strips_unterminated_think_to_eof() {
        // a streamed/truncated block with no closing tag must not leak its body.
        let raw = "answer first\n<think>reasoning that never closes\nstep 2";
        assert_eq!(strip_think_blocks(raw), "answer first");
    }

    #[test]
    fn all_reasoning_collapses_to_empty() {
        // an all-think response (incl. the multi-block join) must be empty so the
        // agent loop's is_empty() nudge path fires instead of returning "\n".
        let joined = format!("{}\n{}", "<think>round one</think>", "<think>round two");
        assert_eq!(strip_think_blocks(&joined), "");
    }

    #[test]
    fn strips_nested_think_blocks() {
        // a single non-greedy regex stops at the first </think> and leaks the tail; the
        // depth-tracking scanner must drop the whole balanced span.
        let raw = "<think>outer <think>inner</think> still hidden</think>answer";
        assert_eq!(strip_think_blocks(raw), "answer");
    }

    #[test]
    fn strips_whitespace_padded_think_tags() {
        // `<think >` must be recognized as an open tag, not leaked as content.
        let raw = "<think >hidden</think >visible";
        assert_eq!(strip_think_blocks(raw), "visible");
    }

    #[test]
    fn leaves_thinking_word_untouched() {
        // `<thinking>` is a different tag — the char after `think` must be ws or `>`.
        let raw = "see <thinking>kept</thinking> here";
        assert_eq!(
            strip_think_blocks(raw),
            "see <thinking>kept</thinking> here"
        );
    }

    #[test]
    fn non_retryable_detects_status_in_wrapped_source() {
        let err = wrapped_provider_error(r#"{"statusCode": 401, "message": "Unauthorized"}"#);
        // `to_string()` only renders the top-level context — proves we must look deeper.
        assert!(!err.to_string().contains("401"));
        assert!(is_non_retryable_client_error(&err));
    }

    #[test]
    fn non_retryable_false_for_server_error() {
        let err = wrapped_provider_error(r#"{"statusCode": 500, "message": "boom"}"#);
        assert!(!is_non_retryable_client_error(&err));
    }

    #[test]
    fn rate_limit_detects_429_in_wrapped_source() {
        let err = wrapped_provider_error(r#"{"statusCode": 429, "message": "Too Many Requests"}"#);
        assert!(is_rate_limit_error(&err));
    }

    #[test]
    fn classifiers_detect_compact_json_status() {
        // Compact bodies (no space after the colon) must still classify — `:401,` / `:429,` would
        // slip past a plain `" 401"` / `" 429"` substring check.
        let unauthorized = wrapped_provider_error(r#"{"statusCode":401,"message":"nope"}"#);
        assert!(is_non_retryable_client_error(&unauthorized));
        let throttled = wrapped_provider_error(r#"{"statusCode":429,"message":"slow down"}"#);
        assert!(is_rate_limit_error(&throttled));
    }

    #[test]
    fn mentions_http_status_requires_standalone_number() {
        assert!(mentions_http_status(r#"{"code":401}"#, 401)); // bounded by punctuation, `code` key
        assert!(mentions_http_status("got 401 unauthorized", 401)); // reason phrase follows
        assert!(!mentions_http_status("request id 4017 failed", 401)); // part of a longer run
        assert!(!mentions_http_status("token count 1401", 401)); // trailing digits
    }

    #[test]
    fn mentions_http_status_requires_status_context() {
        // A standalone status-valued number that is neither keyed nor followed by its reason phrase
        // is incidental, not the response status — don't misclassify the surrounding error.
        assert!(!mentions_http_status("max 400 tokens allowed", 400));
        assert!(!mentions_http_status("trace 404 emitted", 404));
        assert!(!mentions_http_status("req_402abc failed", 402)); // embedded in an identifier
        assert!(!mentions_http_status("retry after 429 seconds", 429)); // bare number, wrong meaning
        // Genuine status references in their usual shapings still match.
        assert!(mentions_http_status(
            r#"{"statusCode":429,"message":"slow"}"#,
            429
        ));
        assert!(mentions_http_status(
            "HttpError: Invalid status code 401",
            401
        ));
        assert!(mentions_http_status("429 Too Many Requests", 429)); // reason phrase
    }

    #[test]
    fn key_must_be_a_whole_word_not_a_substring() {
        // `code` embedded in another word is not a status key, so these transient errors must NOT
        // be classified as non-retryable client errors (they should keep their retries).
        assert!(!mentions_http_status("decode error 404", 404)); // decode contains "code"
        assert!(!mentions_http_status("unicode error 400", 400)); // unicode contains "code"
        assert!(!mentions_http_status("encode failure 403", 403)); // encode contains "code"
        assert!(!mentions_http_status("error_code 402 seen", 402)); // underscore is a left edge
        // right boundary: `code`/`status` as a prefix of a longer word is not a key either.
        assert!(!mentions_http_status("codec 404 negotiation failed", 404));
        assert!(!mentions_http_status("statuslike 401 marker", 401));
        // window edge: even if the 24-byte key window cuts `unicode` right before `code`, the real
        // preceding char ('i') is still consulted, so it stays a non-match.
        assert!(!mentions_http_status("xxxxxxxxxxxxxxxxxunicode 404", 404));
        // `status_code` is still a real key (matched via the `status` word).
        assert!(mentions_http_status(r#"{"status_code": 401}"#, 401));
    }

    #[test]
    fn classifies_provider_error_types_without_numeric_status() {
        // rig flattens direct Anthropic/OpenAI non-2xx to ProviderError(body) with no numeric status;
        // these bodies carry only string error types. Confirm the type matchers fire.
        let anthropic_auth = wrapped_provider_error(
            r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
        );
        assert!(!anthropic_auth.to_string().contains("401"));
        assert!(is_non_retryable_client_error(&anthropic_auth));
        assert!(!is_rate_limit_error(&anthropic_auth));

        let openai_key =
            wrapped_provider_error(r#"{"error":{"code":"invalid_api_key","message":"bad key"}}"#);
        assert!(is_non_retryable_client_error(&openai_key));

        let bad_request = wrapped_provider_error(
            r#"{"error":{"type":"invalid_request_error","message":"prompt is too long"}}"#,
        );
        assert!(is_non_retryable_client_error(&bad_request));

        let ctx_len = wrapped_provider_error(r#"{"error":{"code":"context_length_exceeded"}}"#);
        assert!(is_non_retryable_client_error(&ctx_len));

        // insufficient_quota is permanent (out of credits) despite arriving as HTTP 429.
        let quota = wrapped_provider_error(
            r#"{"error":{"type":"insufficient_quota","message":"exceeded your current quota"}}"#,
        );
        assert!(is_non_retryable_client_error(&quota));
        assert!(!is_rate_limit_error(&quota));

        // Transient overload/throttling types take the rate-limit policy and are not non-retryable.
        let overloaded =
            wrapped_provider_error(r#"{"type":"error","error":{"type":"overloaded_error"}}"#);
        assert!(is_rate_limit_error(&overloaded));
        assert!(!is_non_retryable_client_error(&overloaded));

        let throttled = wrapped_provider_error(r#"{"error":{"code":"rate_limit_exceeded"}}"#);
        assert!(is_rate_limit_error(&throttled));
    }

    #[test]
    fn server_error_takes_precedence_over_nested_4xx() {
        // a 5xx response whose body nests a 4xx (e.g. an upstream code) is a retryable server
        // error, not a permanent client error — so it must NOT be classified non-retryable.
        let err = wrapped_provider_error(r#"{"statusCode":502,"error":{"code":403}}"#);
        assert!(!is_non_retryable_client_error(&err));
        // 5xx codes without a reason phrase here (501) are still matched via the status key.
        let nested = wrapped_provider_error(r#"{"statusCode":501,"error":{"code":403}}"#);
        assert!(!is_non_retryable_client_error(&nested));
        // a genuine 4xx with no 5xx in the chain is still non-retryable.
        let pure_4xx = wrapped_provider_error(r#"{"statusCode":403,"message":"forbidden"}"#);
        assert!(is_non_retryable_client_error(&pure_4xx));
    }
}
