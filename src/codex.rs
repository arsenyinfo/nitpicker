//! ChatGPT/Codex subscription auth for OpenAI reviewers.
//!
//! Reuses the OAuth token the Codex CLI stores in `~/.codex/auth.json` (read-only; never written
//! back), refreshing the short-lived access token in-memory via the refresh token. Requests go to
//! the Codex subscription endpoint, which speaks the OpenAI Responses API with several
//! backend-specific quirks (see [`CodexClient::send`]).
//!
//! Research-only: third-party use of the Codex OAuth client is arguably against OpenAI's ToS,
//! mirroring the posture of the Antigravity Gemini path.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use eyre::{Result, WrapErr};
use rig_core::completion::message::AssistantContent;
use rig_core::providers::openai::responses_api::{
    CompletionRequest as ResponsesBody, CompletionResponse as ResponsesResp,
};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::llm::{
    Completion, CompletionResponse, FinishReason, LLMClient, LLMClientDyn, TokenUsage,
    WithRetryExt, mentions_http_status, merge_json,
};

/// Codex CLI's public OAuth client id (PKCE, no secret) — shared with the official CLI.
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
/// rig appends the `/responses` path; this is the Codex subscription base.
const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const ORIGINATOR: &str = "nitpicker";
/// Refresh this long before the token's stated expiry to avoid racing the deadline.
const EXPIRY_SKEW: Duration = Duration::from_secs(60);
/// Bound the token refresh so a stalled auth request can't pin the token mutex (and thus every
/// concurrent completion sharing the client) indefinitely.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

fn user_agent() -> String {
    format!(
        "nitpicker/{} ({}; {})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

// ============================================================================
// JWT claim parsing (no signature verification — we only read public claims)
// ============================================================================

fn decode_jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Mirror the Codex CLI / opencode account-id resolution order.
fn account_id_from_claims(claims: &Value) -> Option<String> {
    let direct = claims.get("chatgpt_account_id").and_then(Value::as_str);
    let nested = claims
        .get("https://api.openai.com/auth")
        .and_then(|a| a.get("chatgpt_account_id"))
        .and_then(Value::as_str);
    let org = claims
        .get("organizations")
        .and_then(Value::as_array)
        .and_then(|orgs| orgs.first())
        .and_then(|o| o.get("id"))
        .and_then(Value::as_str);
    direct.or(nested).or(org).map(str::to_string)
}

fn account_id_from_tokens(id_token: Option<&str>, access_token: &str) -> Option<String> {
    id_token
        .and_then(decode_jwt_claims)
        .as_ref()
        .and_then(account_id_from_claims)
        .or_else(|| {
            decode_jwt_claims(access_token)
                .as_ref()
                .and_then(account_id_from_claims)
        })
}

/// Expiry from the access token's `exp` claim. Missing/unparseable → `UNIX_EPOCH` (already expired),
/// forcing a refresh on first use; that's not a thrash because the refresh response carries an
/// authoritative `expires_in`.
fn expiry_from_access_token(access_token: &str) -> SystemTime {
    decode_jwt_claims(access_token)
        .and_then(|c| c.get("exp").and_then(Value::as_u64))
        .map(|exp| UNIX_EPOCH + Duration::from_secs(exp))
        .unwrap_or(UNIX_EPOCH)
}

// ============================================================================
// Token store (~/.codex/auth.json)
// ============================================================================

#[derive(Clone)]
struct CodexTokens {
    access_token: String,
    refresh_token: String,
    account_id: Option<String>,
    expires_at: SystemTime,
}

impl CodexTokens {
    fn is_expired(&self) -> bool {
        self.expires_at <= SystemTime::now() + EXPIRY_SKEW
    }
}

/// `$CODEX_HOME/auth.json` (when CODEX_HOME is set, non-empty, absolute) else `~/.codex/auth.json`.
fn auth_path() -> Result<PathBuf> {
    resolve_auth_path(
        std::env::var("CODEX_HOME").ok().as_deref(),
        dirs::home_dir(),
    )
}

/// Pure resolution split from env/dirs lookup so it can be unit-tested.
fn resolve_auth_path(codex_home: Option<&str>, home_dir: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(home) = codex_home {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            let path = Path::new(trimmed);
            if path.is_absolute() {
                return Ok(path.join("auth.json"));
            }
            eyre::bail!("CODEX_HOME must be an absolute path, got \"{home}\"");
        }
    }
    let home =
        home_dir.ok_or_else(|| eyre::eyre!("cannot resolve home directory (set CODEX_HOME)"))?;
    Ok(home.join(".codex").join("auth.json"))
}

fn load_tokens(path: &Path) -> Result<CodexTokens> {
    let raw = std::fs::read_to_string(path).wrap_err_with(|| {
        format!(
            "cannot read {} — run `codex login` with a ChatGPT subscription first",
            path.display()
        )
    })?;
    parse_tokens(&raw, &path.display().to_string())
}

/// Parse a Codex `auth.json` payload. Split from IO so it is unit-testable without a real file.
fn parse_tokens(raw: &str, source: &str) -> Result<CodexTokens> {
    let v: Value =
        serde_json::from_str(raw).wrap_err_with(|| format!("invalid JSON in {source}"))?;
    let tokens = v.get("tokens").ok_or_else(|| {
        eyre::eyre!(
            "no ChatGPT subscription tokens in {source} — run `codex login` (API-key mode is not supported here)"
        )
    })?;
    let access_token = tokens
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| eyre::eyre!("no access_token in {source}"))?
        .to_string();
    let refresh_token = tokens
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| eyre::eyre!("no refresh_token in {source}"))?
        .to_string();
    let account_id = tokens
        .get("account_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| {
            account_id_from_tokens(
                tokens.get("id_token").and_then(Value::as_str),
                &access_token,
            )
        });
    let expires_at = expiry_from_access_token(&access_token);
    Ok(CodexTokens {
        access_token,
        refresh_token,
        account_id,
        expires_at,
    })
}

#[derive(serde::Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_in: Option<u64>,
}

/// Exchange a refresh token for a fresh access token. Expiry comes from the response's
/// `expires_in` (authoritative — a token without an `exp` claim never thrashes the cache).
async fn refresh_tokens(
    http: &reqwest::Client,
    refresh_token: &str,
    prev_account_id: Option<&str>,
) -> Result<CodexTokens> {
    let resp = http
        .post(TOKEN_URL)
        .timeout(REFRESH_TIMEOUT)
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", CLIENT_ID),
        ])
        .send()
        .await
        .wrap_err("Codex token refresh request failed")?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        // surface the HTTP status (with reason phrase) so the 4xx/refresh-rotation logic can
        // classify it. The OAuth error body is `{"error": ...}` and doesn't echo the submitted
        // tokens, but truncate it anyway so nothing unbounded reaches logs.
        eyre::bail!(
            "Codex token refresh returned HTTP {status}: {}",
            truncate(&body, 1000)
        );
    }
    let parsed: RefreshResponse =
        serde_json::from_str(&body).wrap_err("parsing Codex token refresh response")?;
    // prefer the authoritative `expires_in`; if absent, derive from the new access token's JWT `exp`
    // rather than guessing a fixed lifetime.
    let expires_at = match parsed.expires_in {
        Some(secs) => SystemTime::now() + Duration::from_secs(secs),
        None => expiry_from_access_token(&parsed.access_token),
    };
    let account_id = prev_account_id
        .map(str::to_string)
        .or_else(|| account_id_from_tokens(parsed.id_token.as_deref(), &parsed.access_token));
    Ok(CodexTokens {
        access_token: parsed.access_token,
        refresh_token: parsed
            .refresh_token
            .unwrap_or_else(|| refresh_token.to_string()),
        account_id,
        expires_at,
    })
}

// ============================================================================
// Client
// ============================================================================

pub struct CodexClient {
    http: reqwest::Client,
    auth_path: PathBuf,
    tokens: Mutex<CodexTokens>,
}

impl CodexClient {
    /// Load tokens from `~/.codex/auth.json` (or `$CODEX_HOME`) and build the client.
    pub fn new() -> Result<Self> {
        let path = auth_path()?;
        let tokens = load_tokens(&path)?;
        Ok(Self {
            http: reqwest::Client::new(),
            auth_path: path,
            tokens: Mutex::new(tokens),
        })
    }

    /// Refresh under the held lock. If the refresh token was rejected (4xx) the Codex CLI may have
    /// rotated it on disk concurrently — reload `auth.json` once and retry before giving up.
    async fn refresh_locked(&self, tokens: &mut CodexTokens) -> Result<()> {
        match refresh_tokens(
            &self.http,
            &tokens.refresh_token,
            tokens.account_id.as_deref(),
        )
        .await
        {
            Ok(fresh) => {
                *tokens = fresh;
                Ok(())
            }
            Err(err) if is_rotation_4xx(&err) => {
                let disk = load_tokens(&self.auth_path)
                    .wrap_err("Codex refresh token rejected and reloading auth.json failed")?;
                let fresh = refresh_tokens(
                    &self.http,
                    &disk.refresh_token,
                    disk.account_id.as_deref(),
                )
                .await
                .wrap_err(
                    "Codex refresh failed even after reloading auth.json (run `codex login`)",
                )?;
                *tokens = fresh;
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    /// Current access token + account id, refreshing first if within the expiry skew. Double-checked
    /// under the lock so a concurrent wave of subagents triggers at most one refresh.
    async fn current_access(&self) -> Result<(String, Option<String>)> {
        let mut guard = self.tokens.lock().await;
        if guard.is_expired() {
            self.refresh_locked(&mut guard).await?;
        }
        Ok((guard.access_token.clone(), guard.account_id.clone()))
    }

    /// Force a refresh after a 401, gated on the rejected access token: if a concurrent 401 already
    /// refreshed (the cached token differs from the one we used), reuse that instead of refetching.
    async fn force_refresh(&self, used_access: &str) -> Result<(String, Option<String>)> {
        let mut guard = self.tokens.lock().await;
        if guard.access_token != used_access {
            return Ok((guard.access_token.clone(), guard.account_id.clone()));
        }
        self.refresh_locked(&mut guard).await?;
        Ok((guard.access_token.clone(), guard.account_id.clone()))
    }

    /// Build the Responses body, POST it over SSE, and parse the result. Reuses rig's request
    /// lowering and response parsing; the DIY HTTP exists only to satisfy the backend's quirks:
    /// top-level `instructions` (required; rig hardcodes it to None), `stream: true` (required),
    /// `store: false` + omitted `max_output_tokens` (required), and — because store:false is
    /// stateless — output items arrive via `response.output_item.done` rather than in the terminal
    /// `response.completed` event.
    async fn send(
        &self,
        completion: &Completion,
        access: &str,
        account_id: Option<&str>,
    ) -> Result<CompletionResponse> {
        let body = build_body(completion)?;
        let mut req = self
            .http
            .post(format!("{CODEX_BASE_URL}/responses"))
            .bearer_auth(access)
            .header("originator", ORIGINATOR)
            .header(reqwest::header::USER_AGENT, user_agent())
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .json(&body);
        if let Some(id) = account_id {
            req = req.header("ChatGPT-Account-Id", id);
        }
        let resp = req
            .send()
            .await
            .wrap_err("Codex completion request failed")?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .wrap_err("reading Codex completion response")?;
        if !status.is_success() {
            // `{status}` renders as e.g. "401 Unauthorized", which `mentions_http_status` recognizes.
            eyre::bail!("Codex API HTTP {status}: {}", truncate(&text, 1000));
        }
        let raw = parse_sse(&text)?;
        let model = completion.model.clone();
        to_completion_response(raw, model)
    }
}

/// Process-wide shared Codex client (retry-wrapped), built once on first use.
///
/// The token store is inherently process-global — there is a single `~/.codex/auth.json` and a
/// single live token chain — so every Codex reviewer/aggregator must share ONE [`CodexClient`].
/// Per-client caches would each hold their own in-memory refresh token; after a rotation, one
/// client's fresh token would be invisible to the others, and a concurrent refresh on the stale
/// disk token could fail. Sharing one instance gives a single token cache with working
/// concurrent-refresh dedup.
pub fn shared_client() -> Result<Arc<dyn LLMClientDyn>> {
    static CELL: OnceLock<StdMutex<Option<Arc<dyn LLMClientDyn>>>> = OnceLock::new();
    let cell = CELL.get_or_init(|| StdMutex::new(None));
    let mut guard = cell.lock().expect("codex shared-client mutex poisoned");
    if let Some(existing) = guard.as_ref() {
        return Ok(Arc::clone(existing));
    }
    let client = CodexClient::new()?.with_retry().into_arc();
    *guard = Some(Arc::clone(&client));
    Ok(client)
}

impl LLMClient for CodexClient {
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        let (access, account_id) = self.current_access().await?;
        match self.send(&completion, &access, account_id.as_deref()).await {
            Ok(response) => Ok(response),
            // The token can lapse/revoke between our expiry check and the call landing. Force a
            // single refresh-and-retry rather than surfacing a 401 (RetryingLLM treats it as fatal).
            Err(err) if mentions_http_status(&format!("{err:#}"), 401) => {
                let (access, account_id) = self.force_refresh(&access).await?;
                self.send(&completion, &access, account_id.as_deref()).await
            }
            Err(err) => Err(err),
        }
    }
}

// ============================================================================
// Request/response mapping
// ============================================================================

/// Lower a nitpicker [`Completion`] to the Codex Responses body, applying the backend quirks.
fn build_body(completion: &Completion) -> Result<ResponsesBody> {
    let mut c = completion.clone();
    // codex rejects max_output_tokens outright (matches the Codex CLI, which omits it).
    c.max_tokens = None;
    // store:false is mandatory; merge so an existing additional_params object is preserved.
    let store = json!({ "store": false });
    c.additional_params = Some(match c.additional_params.take() {
        Some(existing) => merge_json(existing, store),
        None => store,
    });
    // the system prompt must be sent as top-level `instructions`, not an input item, so take it out
    // of the rig request (None preamble => rig adds no system input item).
    let instructions = c.preamble.take();
    let instructions = match instructions {
        Some(s) if !s.trim().is_empty() => s,
        _ => eyre::bail!(
            "Codex auth requires a system prompt: this completion had no `instructions`"
        ),
    };
    let model = c.model.clone();
    let rig_req: rig_core::completion::CompletionRequest = c.into();
    let mut body = ResponsesBody::try_from((model, rig_req))
        .map_err(|e| eyre::eyre!("lowering completion to Codex Responses body failed: {e}"))?;
    body.instructions = Some(instructions);
    body.stream = Some(true);
    body.max_output_tokens = None;
    Ok(body)
}

/// Split an SSE stream into per-event payloads. Events are separated by a blank line, and a single
/// event may carry its payload across several `data:` fields that must be rejoined with `\n`
/// (per the SSE spec). `str::lines()` strips a trailing `\r`, so this handles `\n` and `\r\n`.
fn sse_event_payloads(text: &str) -> Vec<String> {
    let mut events = vec![];
    let mut current = String::new();
    for line in text.lines() {
        if line.is_empty() {
            if !current.is_empty() {
                events.push(std::mem::take(&mut current));
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            if !current.is_empty() {
                current.push('\n');
            }
            // an optional single leading space after the colon is part of the SSE framing, not data
            current.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
        // other field lines (`event:`, `id:`, comments) are not needed here
    }
    if !current.is_empty() {
        events.push(current);
    }
    events
}

/// Parse the Codex SSE stream into rig's response type. The terminal event is `response.completed`
/// or `response.incomplete` (the latter when the model stops early, e.g. hitting the output cap);
/// both carry the full `response` object. With `store:false` that object's `output` is empty, so the
/// real items are accumulated from the per-item `response.output_item.done` events and injected.
fn parse_sse(text: &str) -> Result<ResponsesResp> {
    let mut items: Vec<Value> = vec![];
    let mut last_parse_err: Option<String> = None;
    for data in sse_event_payloads(text) {
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let ev: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            // keep going (most events are deltas we ignore), but remember the failure so a missing
            // terminal event reports the real parse error instead of a generic "not found".
            Err(e) => {
                last_parse_err = Some(e.to_string());
                continue;
            }
        };
        match ev.get("type").and_then(Value::as_str) {
            Some("response.output_item.done") => {
                if let Some(item) = ev.get("item") {
                    items.push(item.clone());
                }
            }
            Some("response.completed") | Some("response.incomplete") => {
                let mut response = ev
                    .get("response")
                    .cloned()
                    .ok_or_else(|| eyre::eyre!("Codex terminal SSE event missing `response`"))?;
                // only inject accumulated items when the event didn't echo its own output
                // (store:false leaves it empty; a future stored mode might populate it).
                let echoed_output = response
                    .get("output")
                    .and_then(Value::as_array)
                    .is_some_and(|a| !a.is_empty());
                if !echoed_output {
                    response["output"] = Value::Array(std::mem::take(&mut items));
                }
                return serde_json::from_value::<ResponsesResp>(response)
                    .wrap_err("parsing Codex terminal response payload");
            }
            Some("response.failed") | Some("error") => {
                eyre::bail!(
                    "Codex stream error event: {}",
                    truncate(&ev.to_string(), 1000)
                );
            }
            _ => {}
        }
    }
    match last_parse_err {
        Some(e) => eyre::bail!("no terminal event in Codex SSE stream; last parse error: {e}"),
        None => eyre::bail!(
            "no terminal event in Codex SSE stream: {}",
            truncate(text, 1000)
        ),
    }
}

fn to_completion_response(raw: ResponsesResp, model: String) -> Result<CompletionResponse> {
    let incomplete_reason = raw.incomplete_details.as_ref().map(|d| d.reason.clone());
    let parsed: rig_core::completion::CompletionResponse<ResponsesResp> = raw
        .try_into()
        .map_err(|e| eyre::eyre!("parsing Codex response into completion: {e}"))?;
    let has_tool_call = parsed
        .choice
        .iter()
        .any(|c| matches!(c, AssistantContent::ToolCall(_)));
    let finish_reason = if has_tool_call {
        FinishReason::ToolUse
    } else {
        match incomplete_reason.as_deref() {
            Some("max_output_tokens") => FinishReason::MaxTokens,
            Some(other) => FinishReason::Other(other.to_string()),
            None => FinishReason::Stop,
        }
    };
    Ok(CompletionResponse {
        // rig's `TryFrom` above already errors on an empty response, so `choice` is non-empty here.
        choice: parsed.choice,
        finish_reason,
        usage: TokenUsage::new(parsed.usage.input_tokens, parsed.usage.output_tokens),
        selected_model: Some(model),
    })
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let end = (0..=max)
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0);
    format!("{}…", &s[..end])
}

/// A 4xx-other-than-429 status anywhere in the error chain — used to decide whether a failed token
/// refresh warrants reloading a rotated refresh token from disk. 429 is excluded: a rate-limited
/// refresh is transient and reloading `auth.json` can't fix it, so it should bubble up to the outer
/// retry/backoff instead of immediately firing a second refresh.
fn is_rotation_4xx(err: &eyre::Report) -> bool {
    let msg = format!("{err:#}");
    (400..500)
        .filter(|status| *status != 429)
        .any(|status| mentions_http_status(&msg, status))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::message::Message;

    /// Build an unsigned JWT (`alg: none`) with the given claims — enough to exercise our
    /// claims-reading code, which never verifies the signature.
    fn jwt(claims: Value) -> String {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"none"}"#);
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
        format!("{header}.{payload}.sig")
    }

    #[test]
    fn account_id_resolution_order() {
        assert_eq!(
            account_id_from_claims(&json!({ "chatgpt_account_id": "root" })).as_deref(),
            Some("root")
        );
        assert_eq!(
            account_id_from_claims(
                &json!({ "https://api.openai.com/auth": { "chatgpt_account_id": "nested" } })
            )
            .as_deref(),
            Some("nested")
        );
        // root preferred over nested
        assert_eq!(
            account_id_from_claims(&json!({
                "chatgpt_account_id": "root",
                "https://api.openai.com/auth": { "chatgpt_account_id": "nested" }
            }))
            .as_deref(),
            Some("root")
        );
        // organizations fallback
        assert_eq!(
            account_id_from_claims(
                &json!({ "organizations": [{ "id": "org-1" }, { "id": "org-2" }] })
            )
            .as_deref(),
            Some("org-1")
        );
        assert_eq!(account_id_from_claims(&json!({ "email": "x@y.z" })), None);
    }

    #[test]
    fn account_id_prefers_id_token_then_access_token() {
        let id_token = jwt(json!({ "chatgpt_account_id": "from-id" }));
        let access = jwt(json!({ "chatgpt_account_id": "from-access" }));
        assert_eq!(
            account_id_from_tokens(Some(&id_token), &access).as_deref(),
            Some("from-id")
        );
        let id_token_no_acct = jwt(json!({ "email": "x@y.z" }));
        assert_eq!(
            account_id_from_tokens(Some(&id_token_no_acct), &access).as_deref(),
            Some("from-access")
        );
    }

    #[test]
    fn expiry_from_exp_claim_and_missing() {
        let with_exp = jwt(json!({ "exp": 4_000_000_000u64 }));
        assert_eq!(
            expiry_from_access_token(&with_exp),
            UNIX_EPOCH + Duration::from_secs(4_000_000_000)
        );
        // missing exp => already expired (UNIX_EPOCH) so first use refreshes
        assert_eq!(expiry_from_access_token(&jwt(json!({}))), UNIX_EPOCH);
        assert_eq!(expiry_from_access_token("not-a-jwt"), UNIX_EPOCH);
    }

    #[test]
    fn parse_tokens_full_and_derived_account_id() {
        let access = jwt(json!({ "exp": 4_000_000_000u64 }));
        // explicit account_id wins
        let raw = json!({
            "tokens": { "access_token": access, "refresh_token": "rt", "account_id": "acct-explicit" }
        })
        .to_string();
        let t = parse_tokens(&raw, "test").unwrap();
        assert_eq!(t.account_id.as_deref(), Some("acct-explicit"));
        assert!(!t.is_expired());

        // derived from id_token when account_id absent
        let id_token = jwt(json!({ "chatgpt_account_id": "acct-derived" }));
        let raw = json!({
            "tokens": { "access_token": access, "refresh_token": "rt", "id_token": id_token }
        })
        .to_string();
        let t = parse_tokens(&raw, "test").unwrap();
        assert_eq!(t.account_id.as_deref(), Some("acct-derived"));
    }

    #[test]
    fn parse_tokens_rejects_api_key_mode_and_missing_fields() {
        // API-key mode: no `tokens` object
        let raw = json!({ "auth_mode": "apikey", "OPENAI_API_KEY": "sk-x" }).to_string();
        assert!(parse_tokens(&raw, "test").is_err());
        // tokens present but no access_token
        let raw = json!({ "tokens": { "refresh_token": "rt" } }).to_string();
        assert!(parse_tokens(&raw, "test").is_err());
    }

    #[test]
    fn build_body_applies_codex_quirks() {
        let completion = Completion {
            model: "gpt-5.4".to_string(),
            prompt: Message::user("hello"),
            preamble: Some("You are a reviewer.".to_string()),
            history: vec![],
            tools: vec![],
            tool_choice: None,
            max_tokens: Some(4096),
            additional_params: None,
        };
        let body = build_body(&completion).unwrap();
        assert_eq!(body.instructions.as_deref(), Some("You are a reviewer."));
        assert_eq!(body.stream, Some(true));
        assert_eq!(body.max_output_tokens, None);
        // store:false ends up in the flattened additional parameters
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json.get("store"), Some(&Value::Bool(false)));
        // instructions is top-level, not folded into an input system item
        let has_system_input = json
            .get("input")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .any(|i| i.get("role").and_then(Value::as_str) == Some("system"))
            })
            .unwrap_or(false);
        assert!(
            !has_system_input,
            "system must be in `instructions`, not `input`"
        );
    }

    #[test]
    fn build_body_requires_a_system_prompt() {
        let completion = Completion {
            model: "gpt-5.4".to_string(),
            prompt: Message::user("hello"),
            preamble: None,
            history: vec![],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            additional_params: None,
        };
        assert!(build_body(&completion).is_err());
    }

    #[test]
    fn build_body_preserves_existing_additional_params() {
        let completion = Completion {
            model: "gpt-5.4".to_string(),
            prompt: Message::user("hello"),
            preamble: Some("sys".to_string()),
            history: vec![],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            additional_params: Some(json!({ "reasoning": { "effort": "low" } })),
        };
        let body = build_body(&completion).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json.get("store"), Some(&Value::Bool(false)));
        assert!(json.get("reasoning").is_some());
    }

    #[test]
    fn resolve_auth_path_modes() {
        // absolute CODEX_HOME wins
        let p = resolve_auth_path(Some("/custom/codex"), Some(PathBuf::from("/home/u"))).unwrap();
        assert_eq!(p, PathBuf::from("/custom/codex/auth.json"));
        // relative CODEX_HOME errors
        assert!(resolve_auth_path(Some("relative/dir"), Some(PathBuf::from("/home/u"))).is_err());
        // empty CODEX_HOME falls back to home
        let p = resolve_auth_path(Some("  "), Some(PathBuf::from("/home/u"))).unwrap();
        assert_eq!(p, PathBuf::from("/home/u/.codex/auth.json"));
        // unset falls back to home
        let p = resolve_auth_path(None, Some(PathBuf::from("/home/u"))).unwrap();
        assert_eq!(p, PathBuf::from("/home/u/.codex/auth.json"));
        // no home and no CODEX_HOME errors
        assert!(resolve_auth_path(None, None).is_err());
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello", 3), "hel…");
        // multibyte: "é" is 2 bytes; truncating at 1 byte must not split it
        let s = "é".repeat(5); // 10 bytes
        let out = truncate(&s, 3);
        assert!(out.ends_with('…'));
        // the kept prefix is valid UTF-8 (no panic) and has whole chars only
        assert!(out.chars().filter(|&c| c == 'é').count() <= 1);
    }

    #[test]
    fn parse_sse_accumulates_output_items_with_empty_completed() {
        // mirrors the store:false stream: items arrive in output_item.done; completed has empty output
        let item = json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{ "type": "output_text", "text": "pong" }]
        });
        let completed = json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [],
            "usage": {
                "input_tokens": 5,
                "input_tokens_details": { "cached_tokens": 0 },
                "output_tokens": 1,
                "output_tokens_details": { "reasoning_tokens": 0 },
                "total_tokens": 6
            }
        });
        let sse = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n",
            json!({ "type": "response.output_item.done", "item": item }),
            json!({ "type": "response.completed", "response": completed })
        );
        let raw = parse_sse(&sse).unwrap();
        let resp = to_completion_response(raw, "gpt-5.4".to_string()).unwrap();
        assert_eq!(resp.text(), "pong");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.input_tokens, 5);
    }

    #[test]
    fn parse_sse_surfaces_error_events() {
        let sse = format!(
            "data: {}\n",
            json!({ "type": "response.failed", "error": { "message": "boom" } })
        );
        assert!(parse_sse(&sse).is_err());
    }

    fn usage_json() -> Value {
        json!({
            "input_tokens": 5,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": 1,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 6
        })
    }

    #[test]
    fn parse_sse_handles_incomplete_as_max_tokens() {
        // model stopped early on the output cap: terminal event is `response.incomplete`
        let item = json!({
            "type": "message", "id": "m", "role": "assistant", "status": "incomplete",
            "content": [{ "type": "output_text", "text": "partial" }]
        });
        let response = json!({
            "id": "r", "object": "response", "created_at": 1, "status": "incomplete",
            "model": "gpt-5.4", "output": [],
            "incomplete_details": { "reason": "max_output_tokens" },
            "usage": usage_json()
        });
        let sse = format!(
            "data: {}\n\ndata: {}\n",
            json!({ "type": "response.output_item.done", "item": item }),
            json!({ "type": "response.incomplete", "response": response })
        );
        let raw = parse_sse(&sse).unwrap();
        let resp = to_completion_response(raw, "gpt-5.4".to_string()).unwrap();
        assert_eq!(resp.text(), "partial");
        assert_eq!(resp.finish_reason, FinishReason::MaxTokens);
    }

    #[test]
    fn parse_sse_rejoins_multiline_data_fields() {
        // a single event whose JSON payload is split across two `data:` lines (valid SSE);
        // splitting after a `,` keeps each half from being valid alone but the `\n`-join valid.
        let completed = json!({
            "type": "response.completed",
            "response": {
                "id": "r", "object": "response", "created_at": 1, "status": "completed",
                "model": "gpt-5.4",
                "output": [{
                    "type": "message", "id": "m", "role": "assistant", "status": "completed",
                    "content": [{ "type": "output_text", "text": "hi" }]
                }],
                "usage": usage_json()
            }
        })
        .to_string();
        let comma = completed.find(',').unwrap();
        let (head, tail) = completed.split_at(comma);
        let sse = format!("data: {head}\ndata: {tail}\n\n");
        let raw = parse_sse(&sse).unwrap();
        let resp = to_completion_response(raw, "gpt-5.4".to_string()).unwrap();
        assert_eq!(resp.text(), "hi");
    }

    #[test]
    fn is_rotation_4xx_excludes_429_and_5xx() {
        let mk = |s: &str| eyre::eyre!("Codex token refresh returned HTTP {s}");
        assert!(is_rotation_4xx(&mk("400 Bad Request")));
        assert!(is_rotation_4xx(&mk("401 Unauthorized")));
        // 429 is transient — must NOT trigger the reload-from-disk rotation path
        assert!(!is_rotation_4xx(&mk("429 Too Many Requests")));
        assert!(!is_rotation_4xx(&mk("500 Internal Server Error")));
    }
}
