use eyre::{Result, WrapErr};
use rig_core::OneOrMany;
use rig_core::client::CompletionClient;
use rig_core::completion::CompletionError;
use rig_core::completion::message::ReasoningContent;
use rig_core::completion::message::ToolCall;
use rig_core::completion::message::ToolChoice;
use rig_core::completion::{AssistantContent, CompletionModel, Message};
use rig_core::providers::{anthropic, gemini, openai, openrouter};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;
use tracing::warn;

const MAX_COMPLETION_ATTEMPTS: usize = 4;
const RATE_LIMIT_MAX_COMPLETION_ATTEMPTS: usize = 8;
const BASE_BACKOFF_MS: u64 = 250;
const MAX_BACKOFF_MS: u64 = 5_000;
const RATE_LIMIT_BASE_BACKOFF_MS: u64 = 5_000;
const RATE_LIMIT_MAX_BACKOFF_MS: u64 = 60_000;
const ANTHROPIC_DEFAULT_MAX_TOKENS: u64 = 8_192;

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
            // local telemetry policy only (`#[serde(skip)]`), never sent to a provider
            record_telemetry_content: false,
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
    /// Every prompt token the provider processed for this call, cache reads included.
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Always `input_tokens + output_tokens`.
    pub total_tokens: u64,
    /// A breakdown of `input_tokens`, not an additional charge. Passed through verbatim.
    #[serde(default)]
    pub cached_input_tokens: u64,
    /// The part of `input_tokens` written into the cache on this call.
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
}

/// Whether a provider's reported prompt count already contains its cache reads. Stated by each
/// client, never inferred: provider totals can carry categories that are neither prompt nor output
/// (Gemini) or disagree with their own parts (OpenRouter), so arithmetic tests misclassify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAccounting {
    /// Anthropic: cache reads and writes are reported beside `input_tokens`, not within it.
    OutsidePrompt,
    /// OpenAI (chat and Responses), OpenRouter, Gemini: cache reads are already in the prompt.
    InsidePrompt,
}

impl TokenUsage {
    /// Normalize usage so `input_tokens` always means every prompt token processed, cache reads
    /// included — without it an Anthropic cache hit under-reports the prompt and
    /// `ConversationUsageWindow::should_compact` never fires.
    ///
    /// Numbers are re-bucketed, never invented: a provider contradicting itself is reported as it
    /// reported itself. `total_tokens` is `input + output`, so anything counted outside both
    /// (Gemini's thinking tokens) goes unmetered.
    pub fn from_provider(usage: &rig_core::completion::Usage, accounting: CacheAccounting) -> Self {
        let input_tokens = match accounting {
            CacheAccounting::OutsidePrompt => usage
                .input_tokens
                .saturating_add(usage.cached_input_tokens)
                .saturating_add(usage.cache_creation_input_tokens),
            CacheAccounting::InsidePrompt => usage.input_tokens,
        };
        Self {
            input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: input_tokens.saturating_add(usage.output_tokens),
            cached_input_tokens: usage.cached_input_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
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
        // Each response reports the whole prompt it was sent, not the new tokens, so replace
        // rather than accumulate. Note this is the size of the *last* request: tool results
        // appended after it are not counted until the next response comes back.
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
        let (stripped, unclosed_body) = scan_think_blocks(&raw);
        // only an *unclosed* block can hide an answer; a closed one that leaves nothing behind is
        // genuinely all-reasoning and must read as empty so the retry fires.
        let Some(start) = unclosed_body else {
            return stripped;
        };
        // Recovery runs even with text before the block: that text only makes the loss silent,
        // since a non-empty response never retries.
        let recovered = self.recover_unterminated_think(&raw[start..]);
        match (stripped.is_empty(), recovered.is_empty()) {
            (_, true) => stripped,
            (true, false) => recovered,
            (false, false) => format!("{stripped}\n\n{recovered}"),
        }
    }

    /// Salvage an answer that an unterminated `<think>` block swallowed.
    ///
    /// Where the reasoning ends is only knowable because the provider also reports it structurally
    /// and the block repeats it verbatim before the answer, so it is subtracted, never guessed. An
    /// unmatched body returns empty: serving it would pass chain-of-thought off as the verdict.
    fn recover_unterminated_think(&self, body: &str) -> String {
        let Some(reasoning) = self.reasoning_text() else {
            return String::new();
        };
        let body = body.trim_start();
        let Some(answer) = body.strip_prefix(reasoning.trim()) else {
            return String::new();
        };
        // Word boundary, not whitespace: reasoning ending mid-word ("Ver" against "Verdict:") is a
        // coincidental prefix, but real replies do run straight on from the final punctuation.
        let cuts_mid_word = reasoning
            .trim()
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
            && answer.starts_with(char::is_alphanumeric);
        match cuts_mid_word {
            true => String::new(),
            false => answer.trim().to_string(),
        }
    }

    /// The response's structured reasoning, as reported beside the message content.
    fn reasoning_text(&self) -> Option<String> {
        let text = self
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Reasoning(reasoning) => Some(&reasoning.content),
                _ => None,
            })
            .flatten()
            .filter_map(|block| match block {
                ReasoningContent::Text { text, .. } => Some(text.as_str()),
                // Summary paraphrases and Encrypted/Redacted are opaque; none appear verbatim in
                // the block, so they would corrupt the reconstruction rather than complete it.
                _ => None,
            })
            .collect::<String>();
        match text.trim().is_empty() {
            true => None,
            false => Some(text),
        }
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
/// The text outside think blocks, plus where the body of an unclosed block begins. An offset
/// rather than a flag so recovery never re-parses tags: with several blocks or a stray closer, the
/// first `<think>` is not the one that stayed open.
fn scan_think_blocks(text: &str) -> (String, Option<usize>) {
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
    let mut consumed = 0usize;
    let mut open_body_start = None;
    let mut rest = text;
    while !rest.is_empty() {
        if rest.starts_with('<') {
            if let Some(len) = match_think_tag(rest, false) {
                depth += 1;
                consumed += len;
                if depth == 1 {
                    open_body_start = Some(consumed);
                }
                rest = &rest[len..];
                continue;
            }
            if let Some(len) = match_think_tag(rest, true) {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    open_body_start = None;
                }
                consumed += len;
                rest = &rest[len..];
                continue;
            }
        }
        let ch = rest.chars().next().expect("rest is non-empty");
        if depth == 0 {
            out.push(ch);
        }
        consumed += ch.len_utf8();
        rest = &rest[ch.len_utf8()..];
    }
    let unclosed_body = match depth > 0 {
        true => open_body_start,
        false => None,
    };
    (out.trim().to_string(), unclosed_body)
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

/// What the provider said about a response carrying neither text nor a tool call. Without it an
/// exhausted output budget is indistinguishable from a model that had nothing to say.
fn empty_response_diagnosis(response: &CompletionResponse) -> String {
    let output_tokens = response.usage.output_tokens;
    match &response.finish_reason {
        FinishReason::MaxTokens => format!(
            "hit the output limit after {output_tokens} tokens without emitting content — \
             reasoning consumed the whole budget, so raise max_tokens or leave it unset to use \
             the provider's own limit"
        ),
        other => format!("finish_reason {other:?}, {output_tokens} output tokens"),
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
                        let diagnosis = empty_response_diagnosis(&response);
                        if attempt >= MAX_COMPLETION_ATTEMPTS {
                            eyre::bail!(
                                "model returned empty response after {attempt} attempts: {diagnosis}"
                            );
                        }
                        let backoff = jittered_backoff(attempt, BASE_BACKOFF_MS, MAX_BACKOFF_MS);
                        warn!(
                            model = %completion.model,
                            attempt,
                            max_attempts = MAX_COMPLETION_ATTEMPTS,
                            backoff_ms = backoff.as_millis(),
                            diagnosis,
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

/// One model route in an alloy/fallback pool, carrying the settings that belong to it rather than
/// to the logical reviewer role. Inner clients must already be retry-wrapped.
#[derive(Clone)]
pub struct AlloySlot {
    pub client: Arc<dyn LLMClientDyn>,
    pub model: String,
    pub max_tokens: Option<u64>,
}

/// A fallback-capable route. Unlike the legacy public [`AlloySlot`] shape, clones share their
/// run-local availability bit so independent reviewer/aggregator clients stop probing a quota-
/// exhausted subscription after the first conclusive failure.
#[derive(Clone)]
pub struct FallbackSlot {
    client: Arc<dyn LLMClientDyn>,
    model: String,
    max_tokens: Option<u64>,
    unavailable: Arc<AtomicBool>,
}

impl FallbackSlot {
    pub fn new(
        client: Arc<dyn LLMClientDyn>,
        model: impl Into<String>,
        max_tokens: Option<u64>,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            max_tokens,
            unavailable: Arc::new(AtomicBool::new(false)),
        }
    }

    fn is_available(&self) -> bool {
        !self.unavailable.load(Ordering::Acquire)
    }

    fn mark_unavailable(&self) {
        self.unavailable.store(true, Ordering::Release);
    }

    pub fn client(&self) -> Arc<dyn LLMClientDyn> {
        Arc::clone(&self.client)
    }
}

impl From<AlloySlot> for FallbackSlot {
    fn from(slot: AlloySlot) -> Self {
        Self::new(slot.client, slot.model, slot.max_tokens)
    }
}

impl From<&FallbackSlot> for AlloySlot {
    fn from(slot: &FallbackSlot) -> Self {
        Self {
            client: Arc::clone(&slot.client),
            model: slot.model.clone(),
            max_tokens: slot.max_tokens,
        }
    }
}

/// Deterministic failover for one logical reviewer. The caller puts the configured primary first;
/// after an exhausted client retry policy, the remaining reviewer routes are tried in declaration
/// order, wrapping at the end. The same completion (including its full history) is replayed, so the
/// agent does not lose completed investigation work.
pub struct PriorityClient {
    slots: Vec<FallbackSlot>,
    active_idx: AtomicUsize,
}

impl PriorityClient {
    pub fn new(slots: Vec<FallbackSlot>) -> Result<Self> {
        if slots.is_empty() {
            eyre::bail!("PriorityClient requires at least one slot");
        }
        Ok(Self {
            slots,
            active_idx: AtomicUsize::new(0),
        })
    }
}

impl LLMClientDyn for PriorityClient {
    fn completion(
        &self,
        completion: Completion,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse>> + Send + '_>> {
        Box::pin(complete_from_slots(
            &self.slots,
            self.active_idx.load(Ordering::Acquire),
            true,
            completion,
            Some(&self.active_idx),
        ))
    }
}

/// Randomly chooses the first model for every completion — see
/// https://xbow.com/blog/alloy-agents. With fallback enabled, a failed random pick falls through
/// the remaining configured reviewers in declaration order.
pub struct AlloyClient {
    slots: Vec<FallbackSlot>,
    fallback: bool,
}

impl AlloyClient {
    pub fn new(slots: Vec<AlloySlot>) -> Result<Self> {
        Self::build(slots.into_iter().map(FallbackSlot::from).collect(), false)
    }

    pub fn new_with_fallback_routes(slots: Vec<FallbackSlot>) -> Result<Self> {
        Self::build(slots, true)
    }

    fn build(slots: Vec<FallbackSlot>, fallback: bool) -> Result<Self> {
        if slots.is_empty() {
            eyre::bail!("AlloyClient requires at least one slot");
        }
        Ok(Self { slots, fallback })
    }

    fn available_indices(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.is_available().then_some(index))
            .collect()
    }

    fn pick_idx(&self) -> Option<usize> {
        use rand::RngExt;
        let available = self.available_indices();
        if available.is_empty() {
            return None;
        }
        Some(available[rand::rng().random_range(0..available.len())])
    }
}

impl LLMClientDyn for AlloyClient {
    fn completion(
        &self,
        completion: Completion,
    ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse>> + Send + '_>> {
        match self.pick_idx() {
            Some(start_idx) => Box::pin(complete_from_slots(
                &self.slots,
                start_idx,
                self.fallback,
                completion,
                None,
            )),
            None => Box::pin(async {
                eyre::bail!("all configured model routes are unavailable for this run")
            }),
        }
    }
}

fn requires_tool_call(tool_choice: &Option<ToolChoice>) -> bool {
    matches!(
        tool_choice.as_ref(),
        Some(ToolChoice::Required | ToolChoice::Specific { .. })
    )
}

fn tool_protocol_error(completion: &Completion, response: &CompletionResponse) -> Option<String> {
    let calls = response.tool_calls();
    if response.finish_reason == FinishReason::ToolUse {
        let Some(calls) = calls.as_ref() else {
            return Some("reported tool use without returning a tool call".to_string());
        };
        if let Some(call) = calls.iter().find(|call| {
            !completion
                .tools
                .iter()
                .any(|tool| tool.name == call.function.name)
        }) {
            return Some(format!("called undeclared tool '{}'", call.function.name));
        }
        if let Some(ToolChoice::Specific { function_names }) = &completion.tool_choice
            && let Some(call) = calls
                .iter()
                .find(|call| !function_names.contains(&call.function.name))
        {
            return Some(format!(
                "called tool '{}' instead of the specifically requested tool",
                call.function.name
            ));
        }
    } else if requires_tool_call(&completion.tool_choice) {
        return Some("returned text instead of the required tool call".to_string());
    }
    None
}

async fn complete_from_slots(
    slots: &[FallbackSlot],
    start_idx: usize,
    fallback: bool,
    completion: Completion,
    sticky_index: Option<&AtomicUsize>,
) -> Result<CompletionResponse> {
    if !fallback {
        let slot = &slots[start_idx];
        let mut request = completion;
        request.model = slot.model.clone();
        request.max_tokens = slot.max_tokens;
        let mut response = slot.client.completion(request).await?;
        response.selected_model = Some(slot.model.clone());
        return Ok(response);
    }

    let attempts = slots.len();
    let mut last_err = None;
    let mut attempted_models = Vec::with_capacity(attempts);

    for offset in 0..attempts {
        let idx = (start_idx + offset) % slots.len();
        let slot = &slots[idx];
        if !slot.is_available() {
            continue;
        }
        let mut request = completion.clone();
        request.model = slot.model.clone();
        // A cap belongs to the selected model route, not the logical role whose turn this is.
        request.max_tokens = slot.max_tokens;
        attempted_models.push(slot.model.clone());
        let err = match slot.client.completion(request).await {
            Ok(mut response) => {
                let protocol_error = match &response.finish_reason {
                    FinishReason::MaxTokens => Some(format!(
                        "model route '{}' reached its output token limit after {} tokens",
                        slot.model, response.usage.output_tokens
                    )),
                    _ => tool_protocol_error(&completion, &response)
                        .map(|reason| format!("model route '{}': {reason}", slot.model)),
                };
                if let Some(message) = protocol_error {
                    eyre::eyre!(message)
                } else {
                    response.selected_model = Some(slot.model.clone());
                    if let Some(index) = sticky_index {
                        index.store(idx, Ordering::Release);
                    }
                    return Ok(response);
                }
            }
            Err(err) => {
                if is_sticky_fallback_error(&err) {
                    slot.mark_unavailable();
                }
                err
            }
        };
        if let Some(next_idx) = (1..attempts - offset)
            .map(|step| (idx + step) % slots.len())
            .find(|&candidate| slots[candidate].is_available())
        {
            warn!(
                failed_model = %slot.model,
                next_model = %slots[next_idx].model,
                error = %err,
                "model failed; trying next configured reviewer"
            );
        }
        last_err = Some(err);
    }

    match last_err {
        Some(err) => {
            let attempted = attempted_models.join(" -> ");
            Err(err.wrap_err(format!("all model routes failed ({attempted})")))
        }
        None => eyre::bail!("all configured model routes are unavailable for this run"),
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
    // A rolling subscription allowance measured in hours cannot recover within this command.
    // Surface it immediately so fallback mode can move to the next configured reviewer instead
    // of spending several minutes in the ordinary 429 backoff loop.
    if is_long_window_quota_error(err) {
        return RetryPolicy {
            retry: false,
            max_attempts: 0,
            base_backoff_ms: 0,
            max_backoff_ms: 0,
        };
    }

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

/// Error types that often arrive with HTTP 429 but are not transient throttles.
const PERMANENT_QUOTA_ERROR_TYPES: &[&str] = &["insufficient_quota"];

/// Registered server-error statuses. When one is the outer response status, it takes precedence
/// over client/quota text nested in a gateway body because the request itself remains retryable.
const SERVER_ERROR_STATUSES: &[u16] = &[500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511];

fn mentions_server_error_status(msg: &str) -> bool {
    SERVER_ERROR_STATUSES
        .iter()
        .any(|&status| mentions_http_status(msg, status))
}

/// Whether an error chain reports a context-window overflow — the one synthesis failure
/// where "select fewer presets" is real remediation. Matches the OpenAI-style type token
/// (`context_length_exceeded`) and the Anthropic shape — an `invalid_request_error` whose
/// message says the prompt is too long; a generic `invalid_request_error` alone does NOT
/// qualify (those are malformed-request bugs, not size problems).
pub fn is_context_length_error(err: &eyre::Report) -> bool {
    let msg = format!("{err:#}").to_ascii_lowercase();
    msg.contains("context_length_exceeded")
        || (msg.contains("invalid_request_error") && msg.contains("prompt is too long"))
}

fn is_non_retryable_client_error(err: &eyre::Report) -> bool {
    // Walk the whole chain: provider clients map non-2xx to a `ProviderError` carrying the raw
    // response body, then `.wrap_err_with(...)` adds a top-level context. `err.to_string()` renders
    // only that context, so the status code would be invisible; `{err:#}` joins the full chain.
    let msg = format!("{err:#}");
    // a 5xx response takes precedence: even when a 4xx is nested in the body (e.g. an upstream
    // `"code": 403` inside a 502 envelope), the response itself is a retryable server error, so we
    // must not classify it as a permanent client error. Cover the full registered 5xx range; the
    // JSON `status`/`code`-key form is matched even for codes without a reason phrase here.
    if mentions_server_error_status(&msg) {
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
    // wrapped `ProviderError` body still maps to the rate-limit backoff policy. Permanent quota
    // types are excluded first because retrying them only burns the full rate-limit retry budget.
    let msg = format!("{err:#}");
    if mentions_server_error_status(&msg) {
        return false;
    }
    let msg = msg.to_ascii_lowercase();
    if PERMANENT_QUOTA_ERROR_TYPES.iter().any(|t| msg.contains(t)) {
        return false;
    }
    mentions_http_status(&msg, 429)
        || msg.contains("rate limit")
        || msg.contains("too many requests")
        || msg.contains("tokens per minute")
        || msg.contains("requests per minute")
        || RATE_LIMIT_ERROR_TYPES.iter().any(|t| msg.contains(t))
}

fn is_long_window_quota_error(err: &eyre::Report) -> bool {
    let msg = format!("{err:#}");
    // Gateways sometimes wrap an upstream quota payload in a transient 5xx response. The outer
    // status wins: retry this route instead of treating it as exhausted for the rest of the run.
    if mentions_server_error_status(&msg) {
        return false;
    }
    let msg = msg.to_ascii_lowercase();
    msg.contains("out of tokens")
        || msg.contains("usage_limit_reached")
        || msg.contains("hit your usage limit")
        || msg.contains("usage limit has been reached")
        || ((msg.contains("usage limit") || msg.contains("token limit"))
            && (msg.contains("window")
                || msg.contains("reset")
                || msg.contains("refresh")
                || msg.contains("billing cycle")
                || msg.contains("next cycle")
                || msg.contains("5h")
                || msg.contains("5 h")
                || msg.contains("5 hour")))
}

/// Whether an error represents an operational provider limit that cannot recover during this run.
/// CLI layers use this to present a concise message while retaining the full report at debug level.
pub fn is_operational_limit_error(err: &eyre::Report) -> bool {
    is_sticky_fallback_error(err)
}

fn is_sticky_fallback_error(err: &eyre::Report) -> bool {
    let msg = format!("{err:#}");
    // Do not blacklist a route based on quota/rate-limit text nested inside a transient gateway
    // response. The retry wrapper should get the same opportunity it would for a plain 5xx.
    if mentions_server_error_status(&msg) {
        return false;
    }
    let msg = msg.to_ascii_lowercase();
    is_long_window_quota_error(err)
        || is_rate_limit_error(err)
        || PERMANENT_QUOTA_ERROR_TYPES
            .iter()
            .any(|kind| msg.contains(kind))
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
        let finish_reason = response
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
        let finish_reason = resolve_finish_reason(&response.choice, finish_reason);
        Ok(CompletionResponse {
            choice: response.choice,
            finish_reason,
            usage: TokenUsage::from_provider(&response.usage, CacheAccounting::InsidePrompt),
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
        // Anthropic requires `max_tokens`, so "no cap" cannot be expressed here. rig's own default
        // covers only model names it recognizes and falls back to 2048 for every compatible
        // gateway — less than one review turn spends on reasoning.
        request
            .max_tokens
            .get_or_insert(ANTHROPIC_DEFAULT_MAX_TOKENS);
        let model = self.completion_model(model_name.clone());
        let response = model
            .completion(request)
            .await
            .wrap_err_with(|| format!("Anthropic completion failed for model '{model_name}'"))?;
        let finish_reason = response
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
        let finish_reason = resolve_finish_reason(&response.choice, finish_reason);
        Ok(CompletionResponse {
            choice: response.choice,
            finish_reason,
            usage: TokenUsage::from_provider(&response.usage, CacheAccounting::OutsidePrompt),
            selected_model: Some(model_name),
        })
    }
}

impl LLMClient for gemini::Client {
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        let model_name = completion.model.clone();
        let params = build_gemini_additional_params(&completion)?;
        let mut request: rig_core::completion::CompletionRequest = completion.into();
        request.model = Some(model_name.clone());
        request.additional_params = Some(params);
        let model = self.completion_model(model_name.clone());
        let response = model
            .completion(request)
            .await
            .wrap_err_with(|| format!("Gemini completion failed for model '{model_name}'"))?;
        let finish_reason = response
            .raw_response
            .candidates
            .first()
            .and_then(|candidate| candidate.finish_reason.clone())
            .map(map_gemini_finish_reason)
            .unwrap_or(FinishReason::None);
        let finish_reason = resolve_finish_reason(&response.choice, finish_reason);
        Ok(CompletionResponse {
            choice: response.choice,
            finish_reason,
            usage: TokenUsage::from_provider(&response.usage, CacheAccounting::InsidePrompt),
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

/// Deep-merge `extra` over `base`: nested objects are merged recursively; on a scalar/type
/// conflict `extra` wins. Used to overlay computed request params without discarding caller-set
/// ones (e.g. a Gemini `generation_config` the caller supplied alongside its own keys).
pub(crate) fn merge_json(base: Value, extra: Value) -> Value {
    match (base, extra) {
        (Value::Object(mut base_obj), Value::Object(extra_obj)) => {
            for (k, v) in extra_obj {
                let merged = match base_obj.remove(&k) {
                    Some(existing) => merge_json(existing, v),
                    None => v,
                };
                base_obj.insert(k, merged);
            }
            Value::Object(base_obj)
        }
        (_, extra) => extra,
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
        let finish_reason = response
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
        let finish_reason = resolve_finish_reason(&response.choice, finish_reason);
        Ok(CompletionResponse {
            choice: response.choice,
            finish_reason,
            usage: TokenUsage::from_provider(&response.usage, CacheAccounting::InsidePrompt),
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
            // saturating: `as i32` would turn a cap past i32::MAX into a negative field
            max_output_tokens: completion
                .max_tokens
                .map(|value| i32::try_from(value).unwrap_or(i32::MAX)),
        };
        Self {
            generation_config: Some(config),
        }
    }
}

/// Overlay the computed generation config onto any caller-supplied `additional_params` instead of
/// replacing them, mirroring the OpenAI path's `merge_json`. The computed value wins on conflict
/// (e.g. the typed `max_tokens`), while caller keys like safety settings or a nested
/// `generation_config.temperature` survive.
fn build_gemini_additional_params(completion: &Completion) -> Result<Value> {
    let computed = serde_json::to_value(GeminiAdditionalParams::from_completion(completion))?;
    Ok(match &completion.additional_params {
        Some(existing) => merge_json(existing.clone(), computed),
        None => computed,
    })
}

#[derive(Debug, Serialize, Default)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
}

/// A structured tool call in the choice overrides the wire finish reason: providers can pair
/// tool calls with `stop`-like reasons, and `CompletionResponse::tool_calls()` only surfaces
/// calls when the reason is `ToolUse`.
fn resolve_finish_reason(
    choice: &OneOrMany<AssistantContent>,
    wire_reason: FinishReason,
) -> FinishReason {
    match choice
        .iter()
        .any(|content| matches!(content, AssistantContent::ToolCall(_)))
    {
        true => FinishReason::ToolUse,
        false => wire_reason,
    }
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
    use rig_core::completion::Usage;
    use std::sync::Mutex as StdMutex;

    fn strip_think_blocks(text: &str) -> String {
        scan_think_blocks(text).0
    }
    use serde_json::json;

    fn response_with(choice: Vec<AssistantContent>) -> CompletionResponse {
        CompletionResponse {
            choice: OneOrMany::many(choice).expect("non-empty choice"),
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::default(),
            selected_model: None,
        }
    }

    struct RecordingClient {
        fail: bool,
        calls: Arc<StdMutex<Vec<Completion>>>,
    }

    impl LLMClient for RecordingClient {
        async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
            self.calls.lock().unwrap().push(completion);
            if self.fail {
                eyre::bail!("route failed");
            }
            Ok(response_with(vec![AssistantContent::text("ok")]))
        }
    }

    fn recording_slot(
        model: &str,
        max_tokens: Option<u64>,
        fail: bool,
    ) -> (FallbackSlot, Arc<StdMutex<Vec<Completion>>>) {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let client = RecordingClient {
            fail,
            calls: Arc::clone(&calls),
        }
        .into_arc();
        (FallbackSlot::new(client, model, max_tokens), calls)
    }

    fn test_completion() -> Completion {
        Completion {
            model: "logical-primary".to_string(),
            prompt: Message::user("continue"),
            preamble: Some("system".to_string()),
            history: vec![Message::user("prior work")],
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: Some(999),
            additional_params: None,
        }
    }

    #[tokio::test]
    async fn priority_client_uses_next_route_without_losing_the_completion() {
        let (first, first_calls) = recording_slot("first", Some(10), true);
        let (second, second_calls) = recording_slot("second", Some(20), false);
        let client = PriorityClient::new(vec![first, second]).unwrap();

        let response = client.completion(test_completion()).await.unwrap();

        assert_eq!(response.selected_model.as_deref(), Some("second"));
        {
            let first_guard = first_calls.lock().unwrap();
            let second_guard = second_calls.lock().unwrap();
            assert_eq!(first_guard.len(), 1);
            assert_eq!(second_guard.len(), 1);
            assert_eq!(first_guard[0].model, "first");
            assert_eq!(first_guard[0].max_tokens, Some(10));
            assert_eq!(second_guard[0].model, "second");
            assert_eq!(second_guard[0].max_tokens, Some(20));
            assert_eq!(second_guard[0].history.len(), 1);
            assert_eq!(second_guard[0].preamble.as_deref(), Some("system"));
        }

        // A successful failover becomes this logical agent's active route, so the next turn does
        // not probe the failed primary again.
        client.completion(test_completion()).await.unwrap();
        assert_eq!(first_calls.lock().unwrap().len(), 1);
        assert_eq!(second_calls.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn priority_client_wraps_through_declaration_order() {
        let (first, first_calls) = recording_slot("first", None, false);
        let (second, second_calls) = recording_slot("second", None, true);
        let (third, third_calls) = recording_slot("third", None, true);
        // Production callers rotate the configured ring so the logical primary is first.
        let client = PriorityClient::new(vec![second, third, first]).unwrap();

        let response = client.completion(test_completion()).await.unwrap();

        assert_eq!(response.selected_model.as_deref(), Some("first"));
        assert_eq!(second_calls.lock().unwrap().len(), 1);
        assert_eq!(third_calls.lock().unwrap().len(), 1);
        assert_eq!(first_calls.lock().unwrap().len(), 1);
    }

    struct MaxTokensClient {
        calls: Arc<AtomicUsize>,
    }

    impl LLMClient for MaxTokensClient {
        async fn completion(&self, _completion: Completion) -> Result<CompletionResponse> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut response = response_with(vec![AssistantContent::text("partial")]);
            response.finish_reason = FinishReason::MaxTokens;
            response.usage.output_tokens = 10;
            Ok(response)
        }
    }

    #[tokio::test]
    async fn priority_client_retries_truncated_response_on_the_next_route() {
        let truncated_calls = Arc::new(AtomicUsize::new(0));
        let truncated = FallbackSlot::new(
            MaxTokensClient {
                calls: Arc::clone(&truncated_calls),
            }
            .into_arc(),
            "truncated",
            Some(10),
        );
        let (healthy, healthy_calls) = recording_slot("healthy", Some(20), false);
        let client = PriorityClient::new(vec![truncated, healthy]).unwrap();

        let response = client.completion(test_completion()).await.unwrap();

        assert_eq!(response.selected_model.as_deref(), Some("healthy"));
        assert_eq!(truncated_calls.load(Ordering::Relaxed), 1);
        assert_eq!(healthy_calls.lock().unwrap().len(), 1);
    }

    struct ToolUseClient {
        calls: Arc<AtomicUsize>,
    }

    impl LLMClient for ToolUseClient {
        async fn completion(&self, _completion: Completion) -> Result<CompletionResponse> {
            use rig_core::completion::message::ToolFunction;
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut response = response_with(vec![AssistantContent::ToolCall(ToolCall::new(
                "call-1".to_string(),
                ToolFunction::new("allowed".to_string(), serde_json::json!({})),
            ))]);
            response.finish_reason = FinishReason::ToolUse;
            Ok(response)
        }
    }

    struct RequiredToolClient {
        calls: Arc<AtomicUsize>,
    }

    impl LLMClient for RequiredToolClient {
        async fn completion(&self, _completion: Completion) -> Result<CompletionResponse> {
            use rig_core::completion::message::ToolFunction;
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut response = response_with(vec![AssistantContent::ToolCall(ToolCall::new(
                "call-1".to_string(),
                ToolFunction::new("submit".to_string(), serde_json::json!({})),
            ))]);
            response.finish_reason = FinishReason::ToolUse;
            Ok(response)
        }
    }

    #[tokio::test]
    async fn priority_client_retries_unexpected_tool_use_for_no_tools_request() {
        let tool_use_calls = Arc::new(AtomicUsize::new(0));
        let tool_use = FallbackSlot::new(
            ToolUseClient {
                calls: Arc::clone(&tool_use_calls),
            }
            .into_arc(),
            "tool-use",
            None,
        );
        let (healthy, healthy_calls) = recording_slot("healthy", None, false);
        let client = PriorityClient::new(vec![tool_use, healthy]).unwrap();

        let response = client.completion(test_completion()).await.unwrap();

        assert_eq!(response.selected_model.as_deref(), Some("healthy"));
        assert_eq!(tool_use_calls.load(Ordering::Relaxed), 1);
        assert_eq!(healthy_calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn priority_client_accepts_tool_use_when_tools_were_declared() {
        let tool_use_calls = Arc::new(AtomicUsize::new(0));
        let tool_use = FallbackSlot::new(
            ToolUseClient {
                calls: Arc::clone(&tool_use_calls),
            }
            .into_arc(),
            "tool-use",
            None,
        );
        let (unused, unused_calls) = recording_slot("unused", None, false);
        let client = PriorityClient::new(vec![tool_use, unused]).unwrap();
        let mut completion = test_completion();
        completion.tools.push(rig_core::completion::ToolDefinition {
            name: "allowed".to_string(),
            description: "An allowed tool".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        });

        let response = client.completion(completion).await.unwrap();

        assert_eq!(response.finish_reason, FinishReason::ToolUse);
        assert_eq!(response.selected_model.as_deref(), Some("tool-use"));
        assert_eq!(tool_use_calls.load(Ordering::Relaxed), 1);
        assert!(unused_calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn required_tool_protocol_failure_tries_the_next_route() {
        let (plain_text, plain_text_calls) = recording_slot("plain-text", None, false);
        let required_tool_calls = Arc::new(AtomicUsize::new(0));
        let required_tool = FallbackSlot::new(
            RequiredToolClient {
                calls: Arc::clone(&required_tool_calls),
            }
            .into_arc(),
            "required-tool",
            None,
        );
        let client = PriorityClient::new(vec![plain_text, required_tool]).unwrap();
        let mut completion = test_completion();
        completion.tools.push(rig_core::completion::ToolDefinition {
            name: "submit".to_string(),
            description: "Submit the result".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        });
        completion.tool_choice = Some(ToolChoice::Required);

        let response = client.completion(completion).await.unwrap();

        assert_eq!(response.selected_model.as_deref(), Some("required-tool"));
        assert_eq!(plain_text_calls.lock().unwrap().len(), 1);
        assert_eq!(required_tool_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn undeclared_required_tool_call_tries_the_next_route() {
        let wrong_tool_calls = Arc::new(AtomicUsize::new(0));
        let wrong_tool = FallbackSlot::new(
            ToolUseClient {
                calls: Arc::clone(&wrong_tool_calls),
            }
            .into_arc(),
            "wrong-tool",
            None,
        );
        let required_tool_calls = Arc::new(AtomicUsize::new(0));
        let required_tool = FallbackSlot::new(
            RequiredToolClient {
                calls: Arc::clone(&required_tool_calls),
            }
            .into_arc(),
            "required-tool",
            None,
        );
        let client = PriorityClient::new(vec![wrong_tool, required_tool]).unwrap();
        let mut completion = test_completion();
        completion.tools.push(rig_core::completion::ToolDefinition {
            name: "submit".to_string(),
            description: "Submit the result".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        });
        completion.tool_choice = Some(ToolChoice::Required);

        let response = client.completion(completion).await.unwrap();

        assert_eq!(response.selected_model.as_deref(), Some("required-tool"));
        assert_eq!(wrong_tool_calls.load(Ordering::Relaxed), 1);
        assert_eq!(required_tool_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn specifically_disallowed_tool_call_tries_the_next_route() {
        let wrong_tool_calls = Arc::new(AtomicUsize::new(0));
        let wrong_tool = FallbackSlot::new(
            ToolUseClient {
                calls: Arc::clone(&wrong_tool_calls),
            }
            .into_arc(),
            "wrong-tool",
            None,
        );
        let required_tool_calls = Arc::new(AtomicUsize::new(0));
        let required_tool = FallbackSlot::new(
            RequiredToolClient {
                calls: Arc::clone(&required_tool_calls),
            }
            .into_arc(),
            "required-tool",
            None,
        );
        let client = PriorityClient::new(vec![wrong_tool, required_tool]).unwrap();
        let mut completion = test_completion();
        for name in ["allowed", "submit"] {
            completion.tools.push(rig_core::completion::ToolDefinition {
                name: name.to_string(),
                description: "A declared tool".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            });
        }
        completion.tool_choice = Some(ToolChoice::Specific {
            function_names: vec!["submit".to_string()],
        });

        let response = client.completion(completion).await.unwrap();

        assert_eq!(response.selected_model.as_deref(), Some("required-tool"));
        assert_eq!(wrong_tool_calls.load(Ordering::Relaxed), 1);
        assert_eq!(required_tool_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn priority_client_reports_when_every_route_is_unavailable() {
        let (slot, calls) = recording_slot("unavailable", None, false);
        slot.mark_unavailable();
        let client = PriorityClient::new(vec![slot]).unwrap();

        let err = client.completion(test_completion()).await.unwrap_err();

        assert_eq!(
            format!("{err:#}"),
            "all configured model routes are unavailable for this run"
        );
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn alloy_without_fallback_attempts_only_the_selected_route() {
        let (first, first_calls) = recording_slot("first", None, true);
        let (second, second_calls) = recording_slot("second", None, true);
        let client =
            AlloyClient::new(vec![AlloySlot::from(&first), AlloySlot::from(&second)]).unwrap();

        assert!(client.completion(test_completion()).await.is_err());
        assert_eq!(
            first_calls.lock().unwrap().len() + second_calls.lock().unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn alloy_fallback_succeeds_from_every_random_start() {
        let (failed_a, _) = recording_slot("failed-a", None, true);
        let (healthy, _) = recording_slot("healthy", None, false);
        let (failed_b, _) = recording_slot("failed-b", None, true);
        let client =
            AlloyClient::new_with_fallback_routes(vec![failed_a, healthy, failed_b]).unwrap();

        // Whichever slot randomness selects first, walking the ring once must reach `healthy`.
        for _ in 0..32 {
            let response = client.completion(test_completion()).await.unwrap();
            assert_eq!(response.selected_model.as_deref(), Some("healthy"));
        }
    }

    #[test]
    fn alloy_random_start_excludes_unavailable_routes() {
        let (unavailable, _) = recording_slot("unavailable", None, false);
        unavailable.mark_unavailable();
        let (first, _) = recording_slot("first", None, false);
        let (second, _) = recording_slot("second", None, false);
        let client =
            AlloyClient::new_with_fallback_routes(vec![unavailable, first, second]).unwrap();

        assert_eq!(client.available_indices(), vec![1, 2]);
        for _ in 0..32 {
            assert!(matches!(client.pick_idx(), Some(1 | 2)));
        }
    }

    struct QuotaClient {
        calls: Arc<AtomicUsize>,
    }

    impl LLMClient for QuotaClient {
        async fn completion(&self, _completion: Completion) -> Result<CompletionResponse> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            eyre::bail!("out of tokens per 5h window")
        }
    }

    #[tokio::test]
    async fn quota_failure_is_shared_across_priority_clients_for_the_run() {
        let quota_calls = Arc::new(AtomicUsize::new(0));
        let quota = FallbackSlot::new(
            QuotaClient {
                calls: Arc::clone(&quota_calls),
            }
            .into_arc(),
            "quota",
            None,
        );
        let (healthy, _) = recording_slot("healthy", None, false);
        let first = PriorityClient::new(vec![quota.clone(), healthy.clone()]).unwrap();
        let second = PriorityClient::new(vec![quota, healthy]).unwrap();

        first.completion(test_completion()).await.unwrap();
        second.completion(test_completion()).await.unwrap();

        assert_eq!(quota_calls.load(Ordering::Relaxed), 1);
    }

    fn reasoning_item(text: &str) -> AssistantContent {
        AssistantContent::Reasoning(rig_core::completion::message::Reasoning::new(text))
    }

    /// The block is never closed, so the scanner drops the verdict with the reasoning — but the
    /// reasoning is also reported structurally, so it can be subtracted exactly.
    #[test]
    fn unterminated_think_keeps_the_answer_and_drops_the_duplicated_reasoning() {
        let reasoning = "The debate is extremely brief. The Actor suggested `eyre`.";
        let response = response_with(vec![
            reasoning_item(reasoning),
            AssistantContent::text(format!(
                "<think>\n{reasoning}\n\n### Verdict\n**Agreed: use `eyre` for applications.**"
            )),
        ]);
        assert_eq!(
            response.text(),
            "### Verdict\n**Agreed: use `eyre` for applications.**"
        );
    }

    /// Leading text hides the answer just as thoroughly, but makes the response read as
    /// successful — so nothing retries and the verdict is lost silently.
    #[test]
    fn unterminated_think_after_a_prefix_still_recovers_the_answer() {
        let reasoning = "Weighing both sides.";
        let response = response_with(vec![
            reasoning_item(reasoning),
            AssistantContent::text(format!(
                "Here is my review.\n<think>\n{reasoning}\n\n### Verdict\nShip it."
            )),
        ]);
        assert_eq!(
            response.text(),
            "Here is my review.\n\n### Verdict\nShip it."
        );
    }

    /// OpenRouter splits `reasoning_details` mid-sentence, so a newline join stops the fragments
    /// matching their copy inside the block and the whole chain leaks into the answer.
    #[test]
    fn fragmented_reasoning_is_concatenated_without_separators() {
        let response = response_with(vec![
            reasoning_item("weighing the options"),
            reasoning_item(" carefully."),
            AssistantContent::text("<think>\nweighing the options carefully.\n\nVerdict: ship it."),
        ]);
        assert_eq!(response.text(), "Verdict: ship it.");
    }

    /// With no structured reasoning there is no way to tell reasoning from answer. Returning the
    /// body would both serve chain-of-thought as the verdict and make the response read as
    /// non-empty, silencing the retry that exists for a reply that never answered.
    #[test]
    fn unterminated_think_without_structured_reasoning_stays_empty() {
        let response = response_with(vec![AssistantContent::text(
            "<think>\nweighing it up\n\nVerdict: ship it.",
        )]);
        assert!(response.text().is_empty());
    }

    /// A summary is a paraphrase of the chain, not a slice of it, so it cannot be subtracted from
    /// the body — matching it as a fragment would leave a mangled prefix behind.
    #[test]
    fn summary_reasoning_is_not_used_to_split_the_body() {
        let response = response_with(vec![
            AssistantContent::Reasoning(rig_core::completion::message::Reasoning::summaries(vec![
                "brief".to_string(),
            ])),
            AssistantContent::text("<think>private chain"),
        ]);
        assert!(response.text().is_empty());
    }

    /// A reasoning string that happens to be a byte prefix of the answer must not chop it: "Ver"
    /// against "Verdict: ship it." would otherwise yield "dict: ship it.".
    #[test]
    fn a_prefix_that_cuts_mid_word_is_not_treated_as_reasoning() {
        let response = response_with(vec![
            reasoning_item("Ver"),
            AssistantContent::text("<think>Verdict: ship it."),
        ]);
        assert!(response.text().is_empty());
    }

    /// With several blocks, the one that stayed open is not the first tag in the text. Recovery
    /// keys off the scanner's offset, so an earlier *closed* block cannot misdirect it.
    #[test]
    fn recovery_targets_the_block_that_stayed_open() {
        let response = response_with(vec![
            reasoning_item("secret"),
            AssistantContent::text("<think>old</think>\n<think>secret\n\nFINAL"),
        ]);
        assert_eq!(response.text(), "FINAL");
    }

    /// A stray closing tag at depth 0 is dropped by the scanner and must not be mistaken for the
    /// opener when recovering.
    #[test]
    fn a_stray_closing_tag_does_not_misdirect_recovery() {
        let response = response_with(vec![
            reasoning_item("secret"),
            AssistantContent::text("</think><think>secret\n\nFINAL"),
        ]);
        assert_eq!(response.text(), "FINAL");
    }

    /// A response that is only reasoning still collapses to empty, so the empty-response retry
    /// keeps working for models that genuinely returned no answer.
    #[test]
    fn a_reasoning_only_response_is_still_empty() {
        let reasoning = "still thinking about it";
        let response = response_with(vec![
            reasoning_item(reasoning),
            AssistantContent::text(format!("<think>\n{reasoning}")),
        ]);
        assert!(response.text().is_empty());
        let closed = response_with(vec![AssistantContent::text(
            "<think>done deliberating</think>",
        )]);
        assert!(closed.text().is_empty());
    }

    /// A well-formed closed block is untouched by the recovery path.
    #[test]
    fn closed_think_block_still_yields_only_the_answer() {
        let response = response_with(vec![AssistantContent::text(
            "<think>weighing it up</think>Verdict: ship it.",
        )]);
        assert_eq!(response.text(), "Verdict: ship it.");
    }

    /// Mirrors rig's Anthropic mapping: `input_tokens` excludes the cache buckets and
    /// `total_tokens` is their sum plus output (`anthropic/completion.rs`).
    fn anthropic_usage(input: u64, output: u64, cached: u64, creation: u64) -> Usage {
        let mut usage = Usage::new();
        usage.input_tokens = input;
        usage.output_tokens = output;
        usage.cached_input_tokens = cached;
        usage.cache_creation_input_tokens = creation;
        usage.total_tokens = input + cached + creation + output;
        usage
    }

    fn anthropic(usage: &Usage) -> TokenUsage {
        TokenUsage::from_provider(usage, CacheAccounting::OutsidePrompt)
    }

    fn prompt_inclusive(usage: &Usage) -> TokenUsage {
        TokenUsage::from_provider(usage, CacheAccounting::InsidePrompt)
    }

    /// The same 3508-token prompt served from cache, as each provider reports it: Anthropic shows
    /// `input_tokens: 52` with the rest under `cache_read`, OpenAI shows the whole prompt. Both
    /// must meter identically, or a mixed-provider run sums numbers that mean different things —
    /// and the Anthropic shape read raw is what hid 98% of the prompt.
    #[test]
    fn provider_shapes_meter_a_cache_hit_identically() {
        let from_anthropic = anthropic(&anthropic_usage(52, 8, 3456, 0));
        let mut openai = Usage::new();
        openai.input_tokens = 3508;
        openai.output_tokens = 8;
        openai.cached_input_tokens = 3456;
        openai.total_tokens = 3516;
        let from_openai = prompt_inclusive(&openai);

        assert_eq!(from_anthropic.input_tokens, 3508);
        assert_eq!(from_anthropic.input_tokens, from_openai.input_tokens);
        assert_eq!(
            from_anthropic.cached_input_tokens,
            from_openai.cached_input_tokens
        );
        for usage in [from_anthropic, from_openai] {
            assert_eq!(usage.total_tokens, usage.input_tokens + usage.output_tokens);
        }
    }

    /// Gemini reports thinking and tool-use prompts beside both `input_tokens` and
    /// `output_tokens`, and its parts don't sum to its total — so no arithmetic can tell its shape
    /// apart from Anthropic's. Numbers are rig's own fixture (`gemini/completion.rs`) with the
    /// cache lowered to 8, the value that makes the buckets *look* like they fit outside the
    /// prompt. Charging them would report a 40-token prompt as 48.
    #[test]
    fn gemini_cache_reads_are_never_added_to_its_prompt() {
        let mut usage = Usage::new();
        usage.input_tokens = 40;
        usage.output_tokens = 30;
        usage.reasoning_tokens = 10;
        usage.tool_use_prompt_tokens = 12;
        usage.cached_input_tokens = 8;
        usage.total_tokens = 100;
        let usage = prompt_inclusive(&usage);
        assert_eq!(usage.input_tokens, 40);
        assert_eq!(usage.cached_input_tokens, 8);
    }

    /// An OpenRouter gateway may report a total larger than prompt + completion. That slack must
    /// not turn its cache reads — already inside `prompt_tokens` — into extra prompt tokens.
    #[test]
    fn openrouter_total_disagreement_does_not_inflate_the_prompt() {
        let mut usage = Usage::new();
        usage.input_tokens = 500;
        usage.output_tokens = 10;
        usage.cached_input_tokens = 5;
        usage.total_tokens = 515;
        let usage = prompt_inclusive(&usage);
        assert_eq!(usage.input_tokens, 500);
        assert_eq!(usage.total_tokens, 510);
    }

    /// A gateway that omits usage totals must still meter the prompt it reported.
    #[test]
    fn missing_provider_total_still_meters_reported_input() {
        let mut usage = Usage::new();
        usage.input_tokens = 1200;
        usage.output_tokens = 40;
        let usage = prompt_inclusive(&usage);
        assert_eq!(usage.input_tokens, 1200);
        assert_eq!(usage.total_tokens, 1240);
    }

    /// Anthropic charges a cache *write* too, and reports it outside `input_tokens` like a read.
    #[test]
    fn anthropic_cache_writes_count_toward_the_prompt() {
        let usage = anthropic(&anthropic_usage(90, 10, 0, 4100));
        assert_eq!(usage.cache_creation_input_tokens, 4100);
        assert_eq!(usage.input_tokens, 4190);
        assert_eq!(usage.total_tokens, 4200);
    }

    /// A provider reporting more cache than prompt is contradicting itself. Reporting the prompt
    /// as 5000 to make the breakdown fit would invent 4990 tokens of spend that were never billed,
    /// so both numbers are passed through as reported and the contradiction stays visible.
    #[test]
    fn contradictory_cache_metadata_is_not_papered_over() {
        let mut usage = Usage::new();
        usage.input_tokens = 10;
        usage.output_tokens = 5;
        usage.total_tokens = 15;
        usage.cached_input_tokens = 5_000;
        usage.cache_creation_input_tokens = 700;
        let usage = prompt_inclusive(&usage);
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(usage.cached_input_tokens, 5_000);
        assert_eq!(usage.cache_creation_input_tokens, 700);
    }

    /// The whole point of the normalization: a cache hit must still trip the compaction
    /// threshold, since the full prompt is what gets re-sent next turn.
    #[test]
    fn compaction_threshold_sees_the_cached_prompt() {
        let mut window = ConversationUsageWindow::new(Some(3_000));
        window.record(anthropic(&anthropic_usage(52, 8, 3456, 0)));
        assert!(window.should_compact());
    }

    /// The cache fields are `#[serde(default)]` so trajectories written before they existed
    /// still decode.
    #[test]
    fn usage_without_cache_fields_still_deserializes() {
        let usage: TokenUsage = serde_json::from_value(
            json!({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}),
        )
        .expect("legacy usage payload must decode");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.cached_input_tokens, 0);
    }

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

    fn gemini_completion(max_tokens: Option<u64>, additional_params: Option<Value>) -> Completion {
        Completion {
            model: "gemini-3-pro".to_string(),
            prompt: Message::user("hi".to_string()),
            preamble: None,
            history: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            max_tokens,
            additional_params,
        }
    }

    #[test]
    fn merge_json_is_deep_and_extra_wins() {
        // nested objects merge recursively; scalar conflicts resolve to `extra`.
        let merged = merge_json(
            json!({"generation_config": {"temperature": 0.2}, "a": 1}),
            json!({"generation_config": {"max_output_tokens": 50}, "a": 2}),
        );
        assert_eq!(
            merged,
            json!({"generation_config": {"temperature": 0.2, "max_output_tokens": 50}, "a": 2})
        );
    }

    #[test]
    fn gemini_params_preserve_caller_config() {
        // A caller's safety settings and nested generation_config must survive the computed overlay.
        let params = build_gemini_additional_params(&gemini_completion(
            Some(50),
            Some(json!({
                "generation_config": {"temperature": 0.2},
                "safetySettings": [{"category": "HARM", "threshold": "NONE"}]
            })),
        ))
        .unwrap();
        assert_eq!(
            params,
            json!({
                "generation_config": {"temperature": 0.2, "max_output_tokens": 50},
                "safetySettings": [{"category": "HARM", "threshold": "NONE"}]
            })
        );
    }

    #[test]
    fn gemini_params_none_yields_generation_config_only() {
        let params = build_gemini_additional_params(&gemini_completion(Some(50), None)).unwrap();
        assert_eq!(
            params,
            json!({"generation_config": {"max_output_tokens": 50}})
        );
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
    fn response_text_collapses_whitespace_only_content_to_empty() {
        let response = response_with(vec![AssistantContent::text(" \n\t ")]);
        assert!(response.text().is_empty());
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
    fn long_subscription_window_skips_same_route_retries() {
        for message in [
            "out of tokens per 5h window; reset in 2h",
            "You've hit your usage limit",
            "You've reached your usage limit for this billing cycle. Your quota will be refreshed in the next cycle.",
            r#"{"error":{"code":"usage_limit_reached"}}"#,
        ] {
            let err = eyre::eyre!(message);
            assert!(is_long_window_quota_error(&err), "{message}");
            assert!(!retry_policy(&err).retry, "{message}");
        }
        let short_throttle = eyre::eyre!("429 Too Many Requests");
        assert!(!is_long_window_quota_error(&short_throttle));
        assert!(retry_policy(&short_throttle).retry);
    }

    #[test]
    fn nested_long_window_quota_inside_5xx_stays_transient() {
        for status in [500, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511] {
            let body = format!(
                r#"{{"statusCode":{status},"error":{{"code":"usage_limit_reached","message":"out of tokens per 5h window"}}}}"#
            );
            let err = wrapped_provider_error(&body);
            assert!(!is_long_window_quota_error(&err), "status {status}");
            let policy = retry_policy(&err);
            assert!(policy.retry, "status {status}");
            assert_eq!(
                policy.max_attempts, MAX_COMPLETION_ATTEMPTS,
                "status {status}"
            );
            assert!(!is_sticky_fallback_error(&err), "status {status}");
        }

        for nested in [
            r#"{"error":{"code":"insufficient_quota"}}"#,
            r#"{"error":{"message":"rate limit exceeded"}}"#,
        ] {
            let body = format!(r#"{{"statusCode":502,"upstream":{nested}}}"#);
            let err = wrapped_provider_error(&body);
            assert!(!is_rate_limit_error(&err), "{nested}");
            let policy = retry_policy(&err);
            assert!(policy.retry, "{nested}");
            assert_eq!(policy.max_attempts, MAX_COMPLETION_ATTEMPTS, "{nested}");
            assert_eq!(policy.base_backoff_ms, BASE_BACKOFF_MS, "{nested}");
            assert!(!is_sticky_fallback_error(&err), "{nested}");
        }
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

        // Both provider shapes of a context overflow classify as such; a generic
        // invalid_request_error must NOT — the fewer-presets remediation would be
        // misdirection on a malformed-request bug.
        assert!(is_context_length_error(&ctx_len));
        assert!(is_context_length_error(&bad_request));
        let malformed = wrapped_provider_error(
            r#"{"error":{"type":"invalid_request_error","message":"tools[0] is invalid"}}"#,
        );
        assert!(!is_context_length_error(&malformed));

        // insufficient_quota is permanent (out of credits) despite arriving as HTTP 429.
        let quota = wrapped_provider_error(
            r#"{"error":{"type":"insufficient_quota","message":"exceeded your current quota"}}"#,
        );
        assert!(is_non_retryable_client_error(&quota));
        assert!(!is_rate_limit_error(&quota));

        let quota_with_429 = wrapped_provider_error(
            r#"{"statusCode":429,"error":{"type":"insufficient_quota","message":"Too Many Requests"}}"#,
        );
        assert!(is_non_retryable_client_error(&quota_with_429));
        assert!(!is_rate_limit_error(&quota_with_429));
        assert!(!retry_policy(&quota_with_429).retry);

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

    fn tool_call_content() -> AssistantContent {
        use rig_core::completion::message::ToolFunction;
        AssistantContent::ToolCall(ToolCall::new(
            "call-1".to_string(),
            ToolFunction::new("read_file".to_string(), serde_json::json!({ "path": "x" })),
        ))
    }

    /// Providers can pair tool calls with `stop`-like wire reasons; `tool_calls()` only
    /// surfaces calls under `ToolUse`, so the override losing would silently end the agent loop.
    #[test]
    fn a_structured_tool_call_overrides_any_wire_finish_reason() {
        let alone = OneOrMany::one(tool_call_content());
        let beside_text = OneOrMany::many(vec![
            AssistantContent::text("narration before the call"),
            tool_call_content(),
        ])
        .expect("two items");
        for choice in [alone, beside_text] {
            for wire in [
                FinishReason::Stop,
                FinishReason::MaxTokens,
                FinishReason::None,
                FinishReason::Other("length_capped".to_string()),
            ] {
                assert_eq!(resolve_finish_reason(&choice, wire), FinishReason::ToolUse);
            }
        }
    }

    #[test]
    fn without_tool_calls_the_wire_finish_reason_stands() {
        let choice = OneOrMany::one(AssistantContent::text("done"));
        for wire in [
            FinishReason::Stop,
            FinishReason::MaxTokens,
            FinishReason::None,
            FinishReason::Other("content_filter".to_string()),
        ] {
            assert_eq!(resolve_finish_reason(&choice, wire.clone()), wire);
        }
    }
}
