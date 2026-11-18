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
use rig_core::ProviderError;
use rig_core::completion::CompletionRequest;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::providers::openai::wire::CallerIdentity;
use rig_core::providers::{chatgpt, openai};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::llm::{
    Completion, CompletionResponse, LLMClient, LLMClientDyn, WithRetryExt,
    compact_provider_error_body, provider_http_status,
};

/// Codex CLI's public OAuth client id (PKCE, no secret) — shared with the official CLI.
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const ORIGINATOR: &str = "nitpicker";
/// Refresh this long before the token's stated expiry to avoid racing the deadline.
const EXPIRY_SKEW: Duration = Duration::from_secs(60);
/// Bound the token refresh so a stalled auth request can't pin the token mutex (and thus every
/// concurrent completion sharing the client) indefinitely.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(30);
/// Give up establishing a connection after this long (dead host / no route).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Abort a stalled response: if no bytes arrive for this long, the connection is hung. This is an
/// idle (per-read) timeout, not a total cap, so a legitimately long streaming turn is not aborted
/// as long as it keeps making progress — but a hung completion can no longer pin an
/// `llm_semaphore` permit indefinitely.
const READ_TIMEOUT: Duration = Duration::from_secs(120);

/// Shared reqwest client for the Codex endpoints, configured so a hung connection can't pin a
/// shared concurrency permit forever (`reqwest::Client::new()` sets no timeouts).
fn build_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .wrap_err("building Codex HTTP client")
}

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
        // checked_add: a garbage/huge `exp` would otherwise overflow SystemTime and panic; treat
        // overflow as already-expired (forces a refresh) rather than aborting the process.
        .and_then(|exp| UNIX_EPOCH.checked_add(Duration::from_secs(exp)))
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

/// True when a usable Codex CLI OAuth token file is present (CODEX_HOME or `~/.codex/auth.json`).
/// Used by `init` detection; reuses the same parse the client does, so API-key-mode files (which
/// the client rejects) don't surface as a codex provider.
pub fn auth_available() -> bool {
    auth_path().and_then(|p| load_tokens(&p)).is_ok()
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
        return Err(eyre::Report::new(ProviderError::from_http_response(
            status,
            compact_provider_error_body(&body, 1000),
        )))
        .wrap_err("Codex token refresh failed");
    }
    let parsed: RefreshResponse =
        serde_json::from_str(&body).wrap_err("parsing Codex token refresh response")?;
    // prefer the authoritative `expires_in`; if absent, derive from the new access token's JWT `exp`
    // rather than guessing a fixed lifetime.
    let expires_at = match parsed.expires_in {
        // checked_add guards against an absurd `expires_in` overflowing SystemTime (panic);
        // on overflow fall back to the access token's own `exp`.
        Some(secs) => SystemTime::now()
            .checked_add(Duration::from_secs(secs))
            .unwrap_or_else(|| expiry_from_access_token(&parsed.access_token)),
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
            http: build_http_client()?,
            auth_path: path,
            tokens: Mutex::new(tokens),
        })
    }

    /// Refresh under the held lock. If the refresh token was rejected (4xx) the Codex CLI may have
    /// rotated it on disk concurrently — reload `auth.json` once and retry before giving up.
    ///
    /// Read-only limitation: when the OAuth server rotates the refresh token, the new one is held
    /// only in memory (we never write back to `auth.json` — that's the Codex CLI's file). A later
    /// process therefore reads the pre-rotation token from disk; if the server invalidated it, the
    /// reload-retry below hits the same stale token and can't self-heal — so it surfaces a clear
    /// `run \`codex login\`` error rather than persisting a token behind the CLI's back.
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

    /// Use Rig's Codex dialect for stateless Responses framing and replay.
    async fn send(
        &self,
        completion: &Completion,
        access: &str,
        account_id: Option<&str>,
    ) -> Result<CompletionResponse> {
        let wire = codex_wire(&completion.model, access, account_id);
        let model = rig_core::Model::new(wire, rig_reqwest::ReqwestClient::from(self.http.clone()));
        let response = model
            .call(codex_request(completion)?)
            .await
            .wrap_err("Codex completion failed")?;
        Ok(CompletionResponse::from_rig(
            response,
            completion.model.clone(),
        ))
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
            Err(err) if provider_http_status(&err) == Some(401) => {
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

fn codex_wire(model: &str, access: &str, account_id: Option<&str>) -> Responses {
    let mut config = openai::OpenAIConfig::with_key(&chatgpt::DIALECT, access);
    config.instructions = None;
    config.identity = Some(CallerIdentity {
        originator: ORIGINATOR.into(),
        user_agent: user_agent(),
    });
    config.account_id = account_id.map(str::to_owned);
    Responses::new(config, model)
}

fn codex_request(completion: &Completion) -> Result<CompletionRequest> {
    eyre::ensure!(
        completion
            .preamble
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty()),
        "Codex auth requires a system prompt: this completion had no `instructions`"
    );
    let mut request = CompletionRequest::from(completion.clone());
    request.max_tokens = None;
    request.options.cache_key = Some(codex_prompt_cache_key(completion)?);
    Ok(request)
}

/// Stable, opaque cache-routing key for the prefix that remains fixed during one conversation.
/// OpenAI still matches the actual prompt prefix; this only keeps requests with the same
/// model/system/tool family routed together. Including the project-aware system prompt naturally
/// partitions unrelated reviews instead of concentrating all traffic on one hot key.
fn codex_prompt_cache_key(completion: &Completion) -> Result<String> {
    fn hash_part(hasher: &mut Sha256, bytes: &[u8]) {
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }

    let mut hasher = Sha256::new();
    hash_part(&mut hasher, completion.model.as_bytes());
    hash_part(
        &mut hasher,
        completion
            .preamble
            .as_deref()
            .unwrap_or_default()
            .as_bytes(),
    );
    let tools = serde_json::to_vec(&completion.tools)
        .wrap_err("serializing Codex tools for prompt cache routing")?;
    hash_part(&mut hasher, &tools);
    let digest = hasher.finalize();
    let suffix = digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("nitpicker-{suffix}"))
}

/// A 4xx-other-than-429 status anywhere in the error chain — used to decide whether a failed token
/// refresh warrants reloading a rotated refresh token from disk. 429 is excluded: a rate-limited
/// refresh is transient and reloading `auth.json` can't fix it, so it should bubble up to the outer
/// retry/backoff instead of immediately firing a second refresh.
fn is_rotation_4xx(err: &eyre::Report) -> bool {
    provider_http_status(err).is_some_and(|status| (400..500).contains(&status) && status != 429)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::message::{
        AssistantContent, AssistantMessage, CallId, Message, ToolCall, ToolFunction, ToolName,
        ToolResultContent, UserContent,
    };
    use serde_json::json;

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

    fn build_body_value(completion: &Completion) -> Result<Value> {
        use rig_core::wire::{Body, Mode, Wire};
        let wire = codex_wire(&completion.model, "unused", None);
        let encoded = wire.encode(codex_request(completion)?, Mode::Unary)?;
        let Body::Bytes(bytes) = encoded.request.into_body() else {
            panic!("expected a JSON body")
        };
        Ok(serde_json::from_slice(&bytes)?)
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
        let json = build_body_value(&completion).unwrap();
        assert_eq!(json["instructions"], "You are a reviewer.");
        assert_eq!(json["stream"], true);
        assert!(json.get("max_output_tokens").is_none());
        assert_eq!(json.get("store"), Some(&Value::Bool(false)));
        assert!(
            json.get("prompt_cache_key")
                .and_then(Value::as_str)
                .is_some_and(|key| key.starts_with("nitpicker-")),
            "Codex requests must carry a stable prompt cache routing key"
        );
        // encrypted reasoning content is requested so reasoning items survive store:false round-trips
        assert_eq!(
            json.get("include").and_then(Value::as_array),
            Some(&vec![Value::String(
                "reasoning.encrypted_content".to_string()
            )]),
            "must request reasoning.encrypted_content to avoid bare rs_ id 404s"
        );
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
        assert!(build_body_value(&completion).is_err());
    }

    #[test]
    fn prompt_cache_key_tracks_the_stable_prefix_not_the_conversation_tail() {
        let first = Completion {
            model: "gpt-5.4".to_string(),
            prompt: Message::user("initial request"),
            preamble: Some("stable reviewer and project context".to_string()),
            history: vec![],
            tools: vec![rig_core::completion::ToolDefinition {
                name: ToolName::new("read_file").unwrap(),
                annotations: Default::default(),
                description: "Read a file".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } }
                }),
            }],
            tool_choice: None,
            max_tokens: None,
            additional_params: None,
        };
        let mut later_turn = first.clone();
        later_turn.history = vec![
            Message::user("initial request"),
            Message::assistant("intermediate answer"),
        ];
        later_turn.prompt = Message::user("continue");

        let first_key = codex_prompt_cache_key(&first).unwrap();
        let later_key = codex_prompt_cache_key(&later_turn).unwrap();
        assert_eq!(first_key, later_key);

        let mut different_prefix = later_turn;
        different_prefix.preamble = Some("different reviewer or project context".to_string());
        let different_key = codex_prompt_cache_key(&different_prefix).unwrap();
        assert_ne!(first_key, different_key);
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
        let json = build_body_value(&completion).unwrap();
        assert_eq!(json.get("store"), Some(&Value::Bool(false)));
        assert!(json.get("reasoning").is_some());
    }

    #[test]
    fn body_value_preserves_assistant_text_history() {
        let completion = Completion {
            model: "gpt-5.5".to_string(),
            prompt: Message::user("continue"),
            preamble: Some("sys".to_string()),
            history: vec![Message::assistant("previous answer")],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            additional_params: None,
        };

        let body = build_body_value(&completion).unwrap();
        let assistant_message = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item.get("role").and_then(Value::as_str) == Some("assistant"))
            .unwrap();

        assert_eq!(assistant_message["content"][0]["text"], "previous answer");
    }

    /// A tool id from another provider (an Alloy history replayed into Codex) is not a native
    /// `fc_...` id, so it must be absent from `function_call` and pair by `call_id` alone.
    #[test]
    fn body_value_omits_foreign_function_call_item_id() {
        let tool_id = "tool_5Xt5WpWtZ6Whah4Z2uZcZO34";
        let call = ToolCall::new(
            CallId::from_wire(tool_id),
            ToolFunction::new(
                ToolName::new("read_file").unwrap(),
                json!({"path":"src/main.rs"}),
            ),
        );
        let result = call.result(vec![ToolResultContent::text("ok")]);
        let completion = Completion {
            model: "gpt-5.5".to_string(),
            prompt: Message::user("continue"),
            preamble: Some("sys".to_string()),
            history: vec![
                Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
                    call,
                )])),
                Message::User {
                    content: vec![UserContent::ToolResult(result)],
                },
            ],
            tools: vec![],
            tool_choice: None,
            max_tokens: None,
            additional_params: None,
        };

        let body = build_body_value(&completion).unwrap();
        let function_call = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .unwrap();

        assert!(function_call.get("id").is_none());
        assert_eq!(function_call["call_id"], tool_id);

        // The paired tool output must carry the same call_id, or the backend cannot match them.
        let function_call_output = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output"))
            .unwrap();
        assert_eq!(function_call_output["call_id"], tool_id);
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

    async fn codex_reply(sse: String) -> Result<CompletionResponse> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}", listener.local_addr()?);
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            loop {
                assert_ne!(socket.read_buf(&mut request).await.unwrap(), 0);
                if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                    let length: usize = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length: "))
                        .unwrap()
                        .parse()
                        .unwrap();
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                sse.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        let mut wire = codex_wire("gpt-5.4", "unused", None);
        wire.provider = wire.provider.with_base_url(base_url);
        let model =
            rig_core::Model::new(wire, rig_reqwest::ReqwestClient::from(build_http_client()?));
        let response = model
            .call(CompletionRequest::new("hello").preamble("sys"))
            .await;
        server.await?;
        Ok(CompletionResponse::from_rig(response?, "gpt-5.4".into()))
    }

    #[tokio::test]
    async fn parse_sse_accumulates_output_items_with_empty_completed() {
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
                "input_tokens": 5000,
                "input_tokens_details": { "cached_tokens": 4800 },
                "output_tokens": 1,
                "output_tokens_details": { "reasoning_tokens": 0 },
                "total_tokens": 5001
            }
        });
        let sse = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n",
            json!({ "type": "response.output_item.done", "item": item }),
            json!({ "type": "response.completed", "response": completed })
        );
        let resp = codex_reply(sse).await.unwrap();
        assert_eq!(resp.text(), "pong");
        assert_eq!(resp.finish_reason, crate::llm::FinishReason::Stop);
        // cache reads are inside `input_tokens` here, so the prompt must not be inflated by them
        assert_eq!(resp.usage.input_tokens, 5000);
        assert_eq!(resp.usage.cached_input_tokens, 4800);
        assert_eq!(resp.usage.total_tokens, 5001);
    }

    #[tokio::test]
    async fn parse_sse_surfaces_error_events() {
        let sse = format!(
            "data: {}\n\n",
            json!({ "type": "response.failed", "error": { "message": "boom" } })
        );
        assert!(codex_reply(sse).await.is_err());
    }

    #[tokio::test]
    async fn parse_sse_preserves_codes_from_large_error_events() {
        let sse = format!(
            "data: {}\n\n",
            json!({
                "type": "response.failed",
                "error": {
                    "type": "usage_limit_reached",
                    "message": "quota exhausted",
                    "diagnostic": "x".repeat(4_000),
                }
            })
        );
        let err = codex_reply(sse).await.unwrap_err();
        assert!(crate::llm::provider_error_has_code(
            &err,
            &["usage_limit_reached"]
        ));
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

    #[tokio::test]
    async fn parse_sse_handles_incomplete_as_max_tokens() {
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
            "data: {}\n\ndata: {}\n\n",
            json!({ "type": "response.output_item.done", "item": item }),
            json!({ "type": "response.incomplete", "response": response })
        );
        let resp = codex_reply(sse).await.unwrap();
        assert_eq!(resp.text(), "partial");
        assert_eq!(resp.finish_reason, crate::llm::FinishReason::MaxTokens);
    }

    #[tokio::test]
    async fn parse_sse_rejoins_multiline_data_fields() {
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
        let resp = codex_reply(sse).await.unwrap();
        assert_eq!(resp.text(), "hi");
    }

    #[test]
    fn is_rotation_4xx_excludes_429_and_5xx() {
        let mk = |status| {
            eyre::Report::new(ProviderError::from_http_response(
                reqwest::StatusCode::from_u16(status).unwrap(),
                r#"{"error":"test"}"#,
            ))
        };
        assert!(is_rotation_4xx(&mk(400)));
        assert!(is_rotation_4xx(&mk(401)));
        // 429 is transient — must NOT trigger the reload-from-disk rotation path
        assert!(!is_rotation_4xx(&mk(429)));
        assert!(!is_rotation_4xx(&mk(500)));
    }
}
