use crate::tools::floor_char_boundary;
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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;
use tracing::field::Empty;
use tracing::{Instrument, debug, info_span, warn};

use crate::telemetry::{bounded, finish_reason_label};

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

    /// Fork mutable routing state for an independently progressing agent while retaining shared
    /// provider clients and run-wide route availability. Stateless clients return `None` and may
    /// safely be shared by `Arc`.
    fn fork_for_agent(&self) -> Option<Arc<dyn LLMClientDyn>> {
        None
    }
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
            // one span per provider round-trip: the failure CLASS is recorded, never the error text
            let span = info_span!(
                "llm.attempt",
                otel.status_code = Empty,
                "error.type" = Empty,
                gen_ai.request.model = %bounded(&completion.model),
                nitpicker.attempt = attempt as u64,
            );
            match self
                .inner
                .completion(completion.clone())
                .instrument(span.clone())
                .await
            {
                Ok(response) => {
                    if response.text().is_empty() && response.tool_calls().is_none() {
                        span.record("otel.status_code", "ERROR");
                        span.record("error.type", "empty_response");
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
                    let class = classify_provider_failure(&err);
                    span.record("otel.status_code", "ERROR");
                    span.record("error.type", class.as_str());
                    let policy = retry_policy_for(class);
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
/// run-local availability and warning state so independent reviewer/aggregator clients stop
/// probing a quota-exhausted subscription and announce that one route failure only once.
#[derive(Clone)]
pub struct FallbackSlot {
    client: Arc<dyn LLMClientDyn>,
    model: String,
    max_tokens: Option<u64>,
    unavailable: Arc<AtomicBool>,
    failover_warning_emitted: Arc<AtomicBool>,
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
            failover_warning_emitted: Arc::new(AtomicBool::new(false)),
        }
    }

    fn is_available(&self) -> bool {
        !self.unavailable.load(Ordering::Acquire)
    }

    fn mark_unavailable(&self) {
        self.unavailable.store(true, Ordering::Release);
    }

    fn claim_failover_warning(&self) -> bool {
        !self.failover_warning_emitted.swap(true, Ordering::AcqRel)
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

    fn fork_for_agent(&self) -> Option<Arc<dyn LLMClientDyn>> {
        Some(Arc::new(Self {
            slots: self.slots.clone(),
            active_idx: AtomicUsize::new(self.active_idx.load(Ordering::Acquire)),
        }))
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
        let (err, sticky_failure) = match slot.client.completion(request).await {
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
                    (eyre::eyre!(message), false)
                } else {
                    response.selected_model = Some(slot.model.clone());
                    if let Some(index) = sticky_index {
                        index.store(idx, Ordering::Release);
                    }
                    return Ok(response);
                }
            }
            Err(err) => {
                let sticky_failure = is_sticky_fallback_error(&err);
                if sticky_failure {
                    slot.mark_unavailable();
                }
                (err, sticky_failure)
            }
        };
        if let Some(next_idx) = (1..attempts - offset)
            .map(|step| (idx + step) % slots.len())
            .find(|&candidate| slots[candidate].is_available())
        {
            if sticky_failure {
                if slot.claim_failover_warning() {
                    warn!(
                        failed_model = %slot.model,
                        next_model = %slots[next_idx].model,
                        "model unavailable; trying next configured reviewer"
                    );
                } else {
                    debug!(
                        failed_model = %slot.model,
                        next_model = %slots[next_idx].model,
                        error = ?err,
                        "duplicate unavailable-route failure; continuing with fallback"
                    );
                }
            } else {
                warn!(
                    failed_model = %slot.model,
                    next_model = %slots[next_idx].model,
                    error = %err,
                    "model failed; trying next configured reviewer"
                );
            }
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
    let queued = Instant::now();
    let _permit = semaphore.acquire().await.expect("llm semaphore closed");
    // the span starts after the permit so its duration is provider latency; the wait is an attribute
    let queue_wait_ms = queued.elapsed().as_millis() as u64;
    let requested_model = bounded(&completion.model).to_string();
    let span = info_span!(
        "chat",
        otel.name = %format!("chat {requested_model}"),
        otel.kind = "client",
        otel.status_code = Empty,
        gen_ai.operation.name = "chat",
        gen_ai.request.model = %requested_model,
        gen_ai.request.max_tokens = completion.max_tokens,
        gen_ai.response.model = Empty,
        gen_ai.response.finish_reasons = Empty,
        gen_ai.usage.input_tokens = Empty,
        gen_ai.usage.output_tokens = Empty,
        gen_ai.usage.cache_read.input_tokens = Empty,
        gen_ai.usage.cache_creation.input_tokens = Empty,
        nitpicker.queue_wait_ms = queue_wait_ms,
    );
    let result = client.completion(completion).instrument(span.clone()).await;
    match &result {
        Ok(response) => {
            let response_model = response
                .selected_model
                .as_deref()
                .map_or(requested_model.as_str(), bounded);
            span.record("gen_ai.response.model", response_model);
            span.record(
                "gen_ai.response.finish_reasons",
                finish_reason_label(&response.finish_reason),
            );
            span.record("gen_ai.usage.input_tokens", response.usage.input_tokens);
            span.record("gen_ai.usage.output_tokens", response.usage.output_tokens);
            span.record(
                "gen_ai.usage.cache_read.input_tokens",
                response.usage.cached_input_tokens,
            );
            span.record(
                "gen_ai.usage.cache_creation.input_tokens",
                response.usage.cache_creation_input_tokens,
            );
        }
        Err(_) => {
            span.record("otel.status_code", "ERROR");
        }
    }
    result
}

struct RetryPolicy {
    retry: bool,
    max_attempts: usize,
    base_backoff_ms: u64,
    max_backoff_ms: u64,
}

#[cfg(test)]
fn retry_policy(err: &eyre::Report) -> RetryPolicy {
    retry_policy_for(classify_provider_failure(err))
}

fn retry_policy_for(class: ProviderFailureClass) -> RetryPolicy {
    match class {
        // A rolling subscription allowance measured in hours cannot recover within this command.
        // Surface it immediately so fallback can move on instead of entering the 429 backoff loop.
        ProviderFailureClass::PermanentQuota => RetryPolicy {
            retry: false,
            max_attempts: 0,
            base_backoff_ms: 0,
            max_backoff_ms: 0,
        },
        ProviderFailureClass::RateLimit => RetryPolicy {
            retry: true,
            max_attempts: RATE_LIMIT_MAX_COMPLETION_ATTEMPTS,
            base_backoff_ms: RATE_LIMIT_BASE_BACKOFF_MS,
            max_backoff_ms: RATE_LIMIT_MAX_BACKOFF_MS,
        },
        ProviderFailureClass::ContextLength | ProviderFailureClass::NonRetryableClient => {
            RetryPolicy {
                retry: false,
                max_attempts: 0,
                base_backoff_ms: 0,
                max_backoff_ms: 0,
            }
        }
        ProviderFailureClass::Server | ProviderFailureClass::Unknown => RetryPolicy {
            retry: true,
            max_attempts: MAX_COMPLETION_ATTEMPTS,
            base_backoff_ms: BASE_BACKOFF_MS,
            max_backoff_ms: MAX_BACKOFF_MS,
        },
    }
}

/// Exact provider error codes which describe a request that cannot succeed unchanged. These are
/// read from structured `error.code` / `error.type` / `error.status` / `error_type` fields, never
/// searched for in a rendered error string.
const NON_RETRYABLE_ERROR_CODES: &[&str] = &[
    "authentication_error",
    "invalid_api_key",
    "permission_error",
    "permission_denied",
    "invalid_request_error",
    "not_found_error",
];

/// Exact transient overload/throttling codes which warrant the longer rate-limit backoff policy.
const RATE_LIMIT_ERROR_CODES: &[&str] = &[
    "rate_limit_error",
    "rate_limit_exceeded",
    "overloaded_error",
];

/// Exact provider codes for allowance exhaustion which cannot recover during this command.
const PERMANENT_QUOTA_ERROR_CODES: &[&str] = &["insufficient_quota", "usage_limit_reached"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderFailureClass {
    Server,
    PermanentQuota,
    RateLimit,
    ContextLength,
    NonRetryableClient,
    Unknown,
}

impl ProviderFailureClass {
    /// Stable `error.type` vocabulary for the `llm.attempt` span.
    fn as_str(self) -> &'static str {
        match self {
            Self::Server => "server",
            Self::PermanentQuota => "permanent_quota",
            Self::RateLimit => "rate_limit",
            Self::ContextLength => "context_length",
            Self::NonRetryableClient => "non_retryable_client",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Default)]
struct ProviderErrorFacts {
    status: Option<u16>,
    body_status: Option<u16>,
    body_code_status: Option<u16>,
    codes: Vec<String>,
    messages: Vec<String>,
}

impl ProviderErrorFacts {
    fn from_report(err: &eyre::Report) -> Option<Self> {
        let completion_error = err
            .chain()
            .find_map(|source| source.downcast_ref::<CompletionError>())?;
        let mut facts = Self {
            status: completion_error
                .provider_response_status()
                .map(|status| status.as_u16()),
            ..Self::default()
        };
        match completion_error.provider_response_json() {
            Ok(Some(body)) => {
                collect_provider_status_hints(&body, &mut facts);
                collect_provider_error_fields(&body, &mut facts);
            }
            Err(_) => {
                if let Some(body) = completion_error.provider_response_body() {
                    // Some compatible gateways return text/plain. This is the sole unstructured
                    // compatibility input: it is still the provider body, never the rendered
                    // eyre chain or surrounding diagnostic text.
                    facts.messages.push(body.to_string());
                }
            }
            Ok(None) => {}
        }
        Some(facts)
    }

    fn effective_status(&self) -> Option<u16> {
        match self.status {
            Some(status) if !(200..300).contains(&status) => Some(status),
            _ => self.body_status.or(self.body_code_status).or(self.status),
        }
    }

    fn has_code(&self, expected: &[&str]) -> bool {
        self.codes
            .iter()
            .any(|code| expected.contains(&code.as_str()))
    }
}

fn numeric_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<u16> {
    keys.iter().find_map(|key| {
        object
            .get(*key)
            .and_then(Value::as_u64)
            .and_then(|value| u16::try_from(value).ok())
    })
}

/// Rank body-carried status hints by their semantic position instead of whichever numeric field a
/// recursive traversal happens to encounter first. Top-level status fields describe an envelope;
/// a numeric provider `code` is only a fallback for 2xx/statusless error envelopes.
fn collect_provider_status_hints(value: &Value, facts: &mut ProviderErrorFacts) {
    let Some(root) = value.as_object() else {
        return;
    };
    facts.body_status = numeric_field(root, &["status", "statusCode", "status_code"]);
    facts.body_code_status = numeric_field(root, &["code"]);

    if let Some(error) = root.get("error").and_then(Value::as_object) {
        facts.body_status = facts
            .body_status
            .or_else(|| numeric_field(error, &["status", "statusCode", "status_code"]));
        facts.body_code_status = facts
            .body_code_status
            .or_else(|| numeric_field(error, &["code"]));
    }
}

/// Collect only semantically named scalar fields from a provider JSON error envelope. Recursing
/// covers gateway shapes such as `{ "upstream": { "error": ... } }`; the real outer HTTP status
/// is stored separately and always wins over nested provider facts.
fn collect_provider_error_fields(value: &Value, facts: &mut ProviderErrorFacts) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                match (key.as_str(), value) {
                    ("code" | "type" | "status" | "error_type", Value::String(value)) => {
                        facts.codes.push(value.to_ascii_lowercase());
                    }
                    ("message", Value::String(value)) => facts.messages.push(value.clone()),
                    _ => collect_provider_error_fields(value, facts),
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_provider_error_fields(value, facts);
            }
        }
        _ => {}
    }
}

/// Keep Codex's stored provider error bounded without turning valid JSON into an unparseable
/// prefix. Large structured bodies are projected to the exact facts used by the classifier;
/// malformed/text bodies retain the old bounded-prefix behavior.
pub(crate) fn compact_provider_error_body(body: &str, compact_above_bytes: usize) -> String {
    if body.len() <= compact_above_bytes {
        return body.to_string();
    }
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return truncate_utf8(body, compact_above_bytes);
    };
    let mut facts = ProviderErrorFacts::default();
    collect_provider_status_hints(&value, &mut facts);
    collect_provider_error_fields(&value, &mut facts);
    let codes = facts
        .codes
        .into_iter()
        .take(16)
        .map(|code| serde_json::json!({ "code": truncate_utf8(&code, 128) }))
        .collect::<Vec<_>>();
    let messages = facts
        .messages
        .into_iter()
        .take(4)
        .map(|message| serde_json::json!({ "message": truncate_utf8(&message, 512) }))
        .collect::<Vec<_>>();
    serde_json::json!({
        "status": facts.body_status,
        "code": facts.body_code_status,
        "details": codes,
        "messages": messages,
    })
    .to_string()
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let end = floor_char_boundary(value, max_bytes);
    format!("{}…", &value[..end])
}

fn classify_provider_failure(err: &eyre::Report) -> ProviderFailureClass {
    let Some(facts) = ProviderErrorFacts::from_report(err) else {
        return ProviderFailureClass::Unknown;
    };
    let status = facts.effective_status();

    // Anthropic defines a captured HTTP 529 carrying `overloaded_error` as provider overload. It
    // follows the longer overload/throttling policy; an arbitrary gateway 5xx with a nested
    // overload payload must still retain normal server-error precedence.
    if facts.status == Some(529) && facts.has_code(&["overloaded_error"]) {
        return ProviderFailureClass::RateLimit;
    }
    // The captured HTTP status is otherwise the outer response. A gateway 5xx therefore remains
    // transient even when its JSON body embeds an upstream quota or client-error payload.
    if status.is_some_and(|status| (500..600).contains(&status)) {
        return ProviderFailureClass::Server;
    }
    if facts.has_code(PERMANENT_QUOTA_ERROR_CODES)
        || facts
            .messages
            .iter()
            .any(|message| is_long_window_quota_message(message))
    {
        return ProviderFailureClass::PermanentQuota;
    }
    if facts.has_code(RATE_LIMIT_ERROR_CODES) || status == Some(429) {
        return ProviderFailureClass::RateLimit;
    }
    if facts.has_code(&["context_length_exceeded"])
        || (facts.has_code(&["invalid_request_error"])
            && facts
                .messages
                .iter()
                .any(|message| message.to_ascii_lowercase().contains("prompt is too long")))
    {
        return ProviderFailureClass::ContextLength;
    }
    if facts.has_code(NON_RETRYABLE_ERROR_CODES)
        || status.is_some_and(|status| (400..=404).contains(&status))
    {
        return ProviderFailureClass::NonRetryableClient;
    }
    ProviderFailureClass::Unknown
}

pub(crate) fn provider_http_status(err: &eyre::Report) -> Option<u16> {
    ProviderErrorFacts::from_report(err).and_then(|facts| facts.effective_status())
}

pub(crate) fn provider_error_has_code(err: &eyre::Report, expected: &[&str]) -> bool {
    ProviderErrorFacts::from_report(err).is_some_and(|facts| facts.has_code(expected))
}

/// Whether an error chain reports a context-window overflow — the one synthesis failure
/// where "select fewer presets" is real remediation. Matches the OpenAI-style type token
/// (`context_length_exceeded`) and the Anthropic shape — an `invalid_request_error` whose
/// message says the prompt is too long; a generic `invalid_request_error` alone does NOT
/// qualify (those are malformed-request bugs, not size problems).
pub fn is_context_length_error(err: &eyre::Report) -> bool {
    classify_provider_failure(err) == ProviderFailureClass::ContextLength
}

#[cfg(test)]
fn is_non_retryable_client_error(err: &eyre::Report) -> bool {
    matches!(
        classify_provider_failure(err),
        ProviderFailureClass::ContextLength | ProviderFailureClass::NonRetryableClient
    )
}

#[cfg(test)]
fn is_rate_limit_error(err: &eyre::Report) -> bool {
    classify_provider_failure(err) == ProviderFailureClass::RateLimit
}

#[cfg(test)]
fn is_long_window_quota_error(err: &eyre::Report) -> bool {
    classify_provider_failure(err) == ProviderFailureClass::PermanentQuota
}

fn is_long_window_quota_message(msg: &str) -> bool {
    let msg = msg.to_ascii_lowercase();
    msg.contains("out of tokens")
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
    matches!(
        classify_provider_failure(err),
        ProviderFailureClass::PermanentQuota | ProviderFailureClass::RateLimit
    )
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

fn normalize_openrouter_completion_error(err: CompletionError) -> eyre::Report {
    let empty_response = matches!(
        &err,
        CompletionError::ResponseError(msg)
            if msg.contains("no message or tool call") || msg.contains("no choices")
    );
    let report = eyre::Report::new(err);
    if empty_response {
        report.wrap_err("empty response from model (no message or tool call)")
    } else {
        report
    }
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
                // First-party Anthropic supports cache_control. An arbitrary compatible gateway
                // may reject that field, so custom base URLs keep the legacy request shape unless
                // they are constructed explicitly as an AnthropicPromptCachingClient.
                let enable_prompt_caching = base_url.is_none();
                let key_env = api_key_env.as_deref().unwrap_or("ANTHROPIC_API_KEY");
                let api_key = std::env::var(key_env)
                    .or_else(|_| missing_or_local(key_env, base_url.as_deref()))?;
                let mut builder = anthropic::Client::builder().api_key(api_key);
                if let Some(url) = base_url {
                    builder = builder.base_url(url);
                }
                let client = builder.build()?;
                match enable_prompt_caching {
                    true => Ok(Box::new(AnthropicPromptCachingClient::new(client))),
                    false => Ok(Box::new(client)),
                }
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
            .map_err(normalize_openrouter_completion_error)
            .wrap_err_with(|| format!("OpenRouter completion failed for model '{model_name}'"))?;
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

/// Anthropic client policy used by first-party and Foundry routes. Rig's typed caching modes add
/// stable tools/system breakpoints plus a moving top-level conversation breakpoint.
pub(crate) struct AnthropicPromptCachingClient {
    inner: anthropic::Client,
}

impl AnthropicPromptCachingClient {
    pub(crate) fn new(inner: anthropic::Client) -> Self {
        Self { inner }
    }
}

async fn complete_anthropic(
    client: &anthropic::Client,
    completion: Completion,
    prompt_caching: bool,
) -> Result<CompletionResponse> {
    let model_name = completion.model.clone();
    let mut request: rig_core::completion::CompletionRequest = completion.into();
    request.model = Some(model_name.clone());
    // Anthropic requires `max_tokens`, so "no cap" cannot be expressed here. rig's own default
    // covers only model names it recognizes and falls back to 2048 for every compatible gateway —
    // less than one review turn spends on reasoning.
    request
        .max_tokens
        .get_or_insert(ANTHROPIC_DEFAULT_MAX_TOKENS);
    let mut model = client.completion_model(model_name.clone());
    if prompt_caching {
        model = model.with_prompt_caching().with_automatic_caching();
    }
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

impl LLMClient for anthropic::Client {
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        complete_anthropic(self, completion, false).await
    }
}

impl LLMClient for AnthropicPromptCachingClient {
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        complete_anthropic(&self.inner, completion, true).await
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

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
    async fn anthropic_first_party_policy_emits_cache_control_on_the_wire() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let header_end = loop {
                let mut chunk = [0_u8; 4096];
                let read = socket.read(&mut chunk).await.unwrap();
                assert!(read > 0, "client closed before sending HTTP headers");
                request.extend_from_slice(&chunk[..read]);
                if let Some(index) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    break index + 4;
                }
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .expect("request must carry content-length");
            while request.len() < header_end + content_length {
                let mut chunk = [0_u8; 4096];
                let read = socket.read(&mut chunk).await.unwrap();
                assert!(read > 0, "client closed before sending the request body");
                request.extend_from_slice(&chunk[..read]);
            }

            let response_body = json!({
                "content": [{ "type": "text", "text": "ok" }],
                "id": "msg_test",
                "model": "claude-test",
                "role": "assistant",
                "type": "message",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": { "input_tokens": 1, "output_tokens": 1 }
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
                response_body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            serde_json::from_slice::<Value>(&request[header_end..header_end + content_length])
                .unwrap()
        });

        let client = anthropic::Client::builder()
            .api_key("test-key")
            .base_url(format!("http://{address}"))
            .build()
            .unwrap();
        let caching_client = AnthropicPromptCachingClient::new(client);
        let mut completion = test_completion();
        completion.model = "claude-test".to_string();
        completion.tools = vec![rig_core::completion::ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        }];

        LLMClient::completion(&caching_client, completion)
            .await
            .unwrap();
        let body = server.await.unwrap();
        assert_eq!(body["cache_control"]["type"], "ephemeral");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(
            body["tools"].as_array().unwrap().last().unwrap()["cache_control"]["type"],
            "ephemeral"
        );
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
    async fn priority_clients_share_slots_without_sharing_stickiness() {
        let (failed, failed_calls) = recording_slot("failed", None, true);
        let (healthy, healthy_calls) = recording_slot("healthy", None, false);
        let first = PriorityClient::new(vec![failed.clone(), healthy.clone()]).unwrap();
        let independent = PriorityClient::new(vec![failed, healthy]).unwrap();

        first.completion(test_completion()).await.unwrap();
        first.completion(test_completion()).await.unwrap();
        independent.completion(test_completion()).await.unwrap();

        // The first logical agent sticks to its successful fallback, while the independent agent
        // still starts from its own primary. Shared slot state is tested separately for quota.
        assert_eq!(failed_calls.lock().unwrap().len(), 2);
        assert_eq!(healthy_calls.lock().unwrap().len(), 3);
    }

    #[test]
    fn fallback_slot_clones_share_one_failover_warning_claim() {
        let (slot, _) = recording_slot("shared-route", None, false);
        let clone = slot.clone();

        assert!(slot.claim_failover_warning());
        assert!(!clone.claim_failover_warning());

        // Dedupe follows a concrete route, not its model string: distinct credentials using the
        // same model must each retain their own warning claim.
        let (same_model_distinct_route, _) = recording_slot("shared-route", None, false);
        assert!(same_model_distinct_route.claim_failover_warning());
    }

    struct FailOnCallClient {
        calls: Arc<AtomicUsize>,
        fail_on: usize,
    }

    impl LLMClient for FailOnCallClient {
        async fn completion(&self, _completion: Completion) -> Result<CompletionResponse> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if call == self.fail_on {
                eyre::bail!("request-specific route failure")
            }
            Ok(response_with(vec![AssistantContent::text("ok")]))
        }
    }

    #[tokio::test]
    async fn agent_fork_inherits_but_cannot_overwrite_parent_stickiness() {
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));
        let first = FallbackSlot::new(
            FailOnCallClient {
                calls: Arc::clone(&first_calls),
                fail_on: 1,
            }
            .into_arc(),
            "first",
            None,
        );
        let second = FallbackSlot::new(
            FailOnCallClient {
                calls: Arc::clone(&second_calls),
                fail_on: 2,
            }
            .into_arc(),
            "second",
            None,
        );
        let parent = PriorityClient::new(vec![first, second]).unwrap();

        // The parent fails over to `second`; the fork inherits that route, then independently
        // wraps back to `first` when its own request fails there.
        parent.completion(test_completion()).await.unwrap();
        let fork = parent.fork_for_agent().expect("priority clients can fork");
        fork.completion(test_completion()).await.unwrap();
        parent.completion(test_completion()).await.unwrap();

        // The parent's final turn still starts on `second`. A shared active index would have made
        // it start on `first`, while a fork reset to zero would have skipped `second` initially.
        assert_eq!(first_calls.load(Ordering::Relaxed), 2);
        assert_eq!(second_calls.load(Ordering::Relaxed), 3);
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
            Err(eyre::Report::new(CompletionError::from_http_response(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                r#"{"error":{"code":"usage_limit_reached"}}"#,
            )))
        }
    }

    #[tokio::test]
    async fn quota_failure_is_shared_across_agent_forks_for_the_run() {
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
        let first = PriorityClient::new(vec![quota, healthy]).unwrap();
        let second = first.fork_for_agent().expect("priority clients can fork");

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

    fn provider_error(status: Option<u16>, body: &str) -> eyre::Report {
        let error = match status {
            Some(status) => CompletionError::from_http_response(
                reqwest::StatusCode::from_u16(status).unwrap(),
                body,
            ),
            None => CompletionError::from_provider_body(body),
        };
        eyre::Report::new(error).wrap_err("completion failed for model 'test'")
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
    fn classifiers_use_captured_http_status() {
        let unauthorized = provider_error(Some(401), r#"{"message":"Unauthorized"}"#);
        assert!(!unauthorized.to_string().contains("401"));
        assert_eq!(provider_http_status(&unauthorized), Some(401));
        assert!(is_non_retryable_client_error(&unauthorized));

        let throttled = provider_error(Some(429), r#"{"message":"slow down"}"#);
        assert!(is_rate_limit_error(&throttled));
        assert_eq!(
            retry_policy(&throttled).max_attempts,
            RATE_LIMIT_MAX_COMPLETION_ATTEMPTS
        );

        let server = provider_error(Some(503), r#"{"message":"unavailable"}"#);
        assert_eq!(
            classify_provider_failure(&server),
            ProviderFailureClass::Server
        );
        assert_eq!(retry_policy(&server).max_attempts, MAX_COMPLETION_ATTEMPTS);
    }

    #[test]
    fn exact_quota_codes_skip_same_route_retries() {
        for code in PERMANENT_QUOTA_ERROR_CODES {
            let body = format!(r#"{{"error":{{"code":"{code}"}}}}"#);
            let err = provider_error(Some(429), &body);
            assert!(is_long_window_quota_error(&err), "{code}");
            assert!(!is_rate_limit_error(&err), "{code}");
            assert!(!retry_policy(&err).retry, "{code}");
            assert!(is_sticky_fallback_error(&err), "{code}");
        }
    }

    #[test]
    fn structured_kimi_quota_messages_are_the_narrow_compatibility_fallback() {
        for (status, message) in [
            (429, "out of tokens per 5h window; reset in 2h"),
            (429, "You've hit your usage limit"),
            (
                403,
                "You've reached your usage limit for this billing cycle. Your quota will be refreshed in the next cycle.",
            ),
        ] {
            let body = json!({"error": {"type": "access_terminated_error", "message": message}});
            let err = provider_error(Some(status), &body.to_string());
            assert!(is_long_window_quota_error(&err), "{message}");
            assert!(!retry_policy(&err).retry, "{message}");
        }

        // Diagnostic text outside a structured provider body is deliberately ignored.
        let untyped = eyre::eyre!("insufficient_quota: out of tokens per 5h window");
        assert_eq!(
            classify_provider_failure(&untyped),
            ProviderFailureClass::Unknown
        );
        assert_eq!(retry_policy(&untyped).max_attempts, MAX_COMPLETION_ATTEMPTS);
    }

    #[test]
    fn real_5xx_status_precedes_nested_quota_and_rate_codes() {
        for status in (500..600).filter(|&status| status != 529) {
            for body in [
                r#"{"upstream":{"error":{"code":"insufficient_quota","message":"out of tokens per 5h window"}}}"#,
                r#"{"upstream":{"error":{"type":"overloaded_error"}}}"#,
            ] {
                let err = provider_error(Some(status), body);
                assert_eq!(
                    classify_provider_failure(&err),
                    ProviderFailureClass::Server
                );
                let policy = retry_policy(&err);
                assert!(policy.retry);
                assert_eq!(policy.max_attempts, MAX_COMPLETION_ATTEMPTS);
                assert!(!is_sticky_fallback_error(&err));
            }
        }
    }

    #[test]
    fn anthropic_529_overload_uses_the_long_overload_policy() {
        let err = provider_error(
            Some(529),
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        );

        assert_eq!(
            classify_provider_failure(&err),
            ProviderFailureClass::RateLimit
        );
        let policy = retry_policy(&err);
        assert!(policy.retry);
        assert_eq!(policy.max_attempts, RATE_LIMIT_MAX_COMPLETION_ATTEMPTS);
        assert_eq!(policy.base_backoff_ms, RATE_LIMIT_BASE_BACKOFF_MS);
        assert!(is_sticky_fallback_error(&err));
    }

    #[test]
    fn exact_provider_codes_classify_without_http_status() {
        for code in [
            "authentication_error",
            "invalid_api_key",
            "permission_error",
        ] {
            let body = format!(r#"{{"error":{{"type":"{code}"}}}}"#);
            let err = provider_error(None, &body);
            assert!(is_non_retryable_client_error(&err), "{code}");
        }

        for code in RATE_LIMIT_ERROR_CODES {
            let body = format!(r#"{{"error":{{"code":"{code}"}}}}"#);
            let err = provider_error(None, &body);
            assert!(is_rate_limit_error(&err), "{code}");
        }
    }

    #[test]
    fn context_length_uses_exact_code_and_structured_message() {
        let openai = provider_error(Some(400), r#"{"error":{"code":"context_length_exceeded"}}"#);
        assert!(is_context_length_error(&openai));

        let anthropic = provider_error(
            Some(400),
            r#"{"error":{"type":"invalid_request_error","message":"prompt is too long"}}"#,
        );
        assert!(is_context_length_error(&anthropic));

        let malformed = provider_error(
            Some(400),
            r#"{"error":{"type":"invalid_request_error","message":"tools[0] is invalid"}}"#,
        );
        assert!(!is_context_length_error(&malformed));
        assert!(is_non_retryable_client_error(&malformed));
    }

    #[test]
    fn provider_json_status_hint_handles_success_error_envelopes() {
        let err = provider_error(
            Some(200),
            r#"{"error":{"code":429,"type":"rate_limit_error","message":"slow down"}}"#,
        );
        assert_eq!(provider_http_status(&err), Some(429));
        assert!(is_rate_limit_error(&err));
    }

    #[test]
    fn text_provider_body_only_uses_the_quota_message_fallback() {
        for status in [403, 429] {
            let err = provider_error(
                Some(status),
                "You've reached your usage limit for this billing cycle; refresh in the next cycle",
            );
            assert!(is_long_window_quota_error(&err));
            assert!(!retry_policy(&err).retry);
            assert!(is_sticky_fallback_error(&err));
        }

        let err = provider_error(Some(429), "insufficient_quota in a non-JSON response");
        // Arbitrary body text is not scanned for codes; the captured 429 still classifies.
        assert!(is_rate_limit_error(&err));

        let statusless = provider_error(None, "insufficient_quota in a non-JSON response");
        assert_eq!(
            classify_provider_failure(&statusless),
            ProviderFailureClass::Unknown
        );
        assert!(!is_sticky_fallback_error(&statusless));
    }

    #[test]
    fn top_level_status_hint_precedes_nested_numeric_codes() {
        let err = provider_error(
            Some(200),
            r#"{"statusCode":502,"upstream":{"error":{"code":403,"type":"insufficient_quota"}}}"#,
        );
        assert_eq!(provider_http_status(&err), Some(502));
        assert_eq!(
            classify_provider_failure(&err),
            ProviderFailureClass::Server
        );
        assert!(!is_sticky_fallback_error(&err));
    }

    #[test]
    fn compact_provider_error_body_keeps_large_json_classifiable() {
        let body = json!({
            "error": {
                "type": "usage_limit_reached",
                "message": "quota exhausted",
                "diagnostic": "x".repeat(4_000),
            }
        })
        .to_string();
        let compact = compact_provider_error_body(&body, 1_000);
        serde_json::from_str::<Value>(&compact).expect("compacted provider body stays valid JSON");
        let err = provider_error(Some(429), &compact);
        assert!(is_long_window_quota_error(&err));
    }

    #[test]
    fn provider_error_truncation_reuses_utf8_boundary_handling() {
        assert_eq!(truncate_utf8("aéz", 2), "a…");
    }

    #[test]
    fn openrouter_normalization_preserves_typed_provider_source() {
        let err = normalize_openrouter_completion_error(CompletionError::from_http_response(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"type":"insufficient_quota"}}"#,
        ));
        assert_eq!(provider_http_status(&err), Some(429));
        assert!(is_long_window_quota_error(&err));
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

    mod spans {
        use super::*;
        use crate::telemetry::capture::SpanCapture;

        struct UsageClient;

        impl LLMClient for UsageClient {
            async fn completion(&self, _completion: Completion) -> Result<CompletionResponse> {
                Ok(CompletionResponse {
                    choice: OneOrMany::one(AssistantContent::text("ok")),
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage {
                        input_tokens: 100,
                        output_tokens: 10,
                        total_tokens: 110,
                        cached_input_tokens: 80,
                        cache_creation_input_tokens: 20,
                    },
                    selected_model: Some("route-b".to_string()),
                })
            }
        }

        const SECRET: &str = "SENTINEL-4f2a";

        /// Fails the first `failures` calls with an error that quotes the request, like a
        /// provider echoing a rejected prompt would.
        struct EchoingFailClient {
            calls: Arc<AtomicUsize>,
            failures: usize,
        }

        impl LLMClient for EchoingFailClient {
            async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
                let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
                if call <= self.failures {
                    eyre::bail!("upstream rejected {SECRET}: {:?}", completion.prompt)
                }
                Ok(response_with(vec![AssistantContent::text("ok")]))
            }
        }

        fn secret_completion() -> Completion {
            let mut completion = test_completion();
            completion.prompt = Message::user(format!("prompt {SECRET}"));
            completion.preamble = Some(format!("system {SECRET}"));
            completion
        }

        #[tokio::test]
        async fn throttled_completion_records_one_chat_span_with_usage() {
            let capture = SpanCapture::default();
            let _active = capture.activate();
            let semaphore = Semaphore::new(1);
            let client = UsageClient.into_arc();
            throttled_completion(&semaphore, &client, test_completion())
                .await
                .expect("completion succeeds");

            let chats = capture.named("chat");
            assert_eq!(chats.len(), 1);
            let chat = &chats[0];
            assert_eq!(chat.parent, None);
            assert_eq!(chat.field("otel.name"), Some("chat logical-primary"));
            assert_eq!(chat.field("otel.kind"), Some("client"));
            assert_eq!(chat.field("gen_ai.operation.name"), Some("chat"));
            assert_eq!(chat.field("gen_ai.request.model"), Some("logical-primary"));
            assert_eq!(chat.field("gen_ai.request.max_tokens"), Some("999"));
            assert_eq!(chat.field("gen_ai.response.model"), Some("route-b"));
            assert_eq!(chat.field("gen_ai.response.finish_reasons"), Some("stop"));
            assert_eq!(chat.field("gen_ai.usage.input_tokens"), Some("100"));
            assert_eq!(chat.field("gen_ai.usage.output_tokens"), Some("10"));
            assert_eq!(
                chat.field("gen_ai.usage.cache_read.input_tokens"),
                Some("80")
            );
            assert_eq!(
                chat.field("gen_ai.usage.cache_creation.input_tokens"),
                Some("20")
            );
            assert!(chat.field("nitpicker.queue_wait_ms").is_some());
            assert_eq!(chat.field("otel.status_code"), None);
        }

        #[tokio::test(start_paused = true)]
        async fn retries_become_attempt_spans_carrying_only_the_failure_class() {
            let capture = SpanCapture::default();
            let _active = capture.activate();
            let semaphore = Semaphore::new(1);
            let client = EchoingFailClient {
                calls: Arc::new(AtomicUsize::new(0)),
                failures: 1,
            }
            .with_retry()
            .into_arc();
            throttled_completion(&semaphore, &client, secret_completion())
                .await
                .expect("second attempt succeeds");

            let chat = &capture.named("chat")[0];
            assert_eq!(chat.field("otel.status_code"), None);
            let attempts = capture.named("llm.attempt");
            assert_eq!(attempts.len(), 2);
            for attempt in &attempts {
                assert_eq!(attempt.parent, Some(chat.id), "attempts nest under chat");
                assert_eq!(
                    attempt.field("gen_ai.request.model"),
                    Some("logical-primary")
                );
            }
            assert_eq!(attempts[0].field("nitpicker.attempt"), Some("1"));
            assert_eq!(attempts[0].field("otel.status_code"), Some("ERROR"));
            assert_eq!(attempts[0].field("error.type"), Some("unknown"));
            assert_eq!(attempts[1].field("nitpicker.attempt"), Some("2"));
            assert_eq!(attempts[1].field("otel.status_code"), None);
            assert_eq!(attempts[1].field("error.type"), None);
            capture.assert_no_secret(SECRET);
        }

        #[tokio::test(start_paused = true)]
        async fn exhausted_retries_mark_the_chat_span_as_error() {
            let capture = SpanCapture::default();
            let _active = capture.activate();
            let semaphore = Semaphore::new(1);
            let client = EchoingFailClient {
                calls: Arc::new(AtomicUsize::new(0)),
                failures: usize::MAX,
            }
            .with_retry()
            .into_arc();
            let err = throttled_completion(&semaphore, &client, secret_completion())
                .await
                .expect_err("every attempt fails");
            assert!(
                err.to_string().contains(SECRET),
                "the caller still sees the real error"
            );

            let chat = &capture.named("chat")[0];
            assert_eq!(chat.field("otel.status_code"), Some("ERROR"));
            assert_eq!(chat.field("gen_ai.response.model"), None);
            assert_eq!(capture.named("llm.attempt").len(), MAX_COMPLETION_ATTEMPTS);
            capture.assert_no_secret(SECRET);
        }

        #[tokio::test(start_paused = true)]
        async fn empty_responses_are_classified_without_quoting_them() {
            struct EmptyThenTextClient(Arc<AtomicUsize>);
            impl LLMClient for EmptyThenTextClient {
                async fn completion(&self, _c: Completion) -> Result<CompletionResponse> {
                    let call = self.0.fetch_add(1, Ordering::Relaxed);
                    let text = if call == 0 { "" } else { "ok" };
                    Ok(response_with(vec![AssistantContent::text(text)]))
                }
            }
            let capture = SpanCapture::default();
            let _active = capture.activate();
            let semaphore = Semaphore::new(1);
            let client = EmptyThenTextClient(Arc::new(AtomicUsize::new(0)))
                .with_retry()
                .into_arc();
            throttled_completion(&semaphore, &client, test_completion())
                .await
                .expect("retry succeeds");
            let attempts = capture.named("llm.attempt");
            assert_eq!(attempts.len(), 2);
            assert_eq!(attempts[0].field("error.type"), Some("empty_response"));
            assert_eq!(attempts[0].field("otel.status_code"), Some("ERROR"));
        }
    }
}
