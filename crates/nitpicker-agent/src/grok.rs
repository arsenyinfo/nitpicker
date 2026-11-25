//! Grok / xAI subscription auth for OpenAI-compatible Responses reviewers.
//!
//! Reuses the OAuth token the Grok Build CLI stores in `~/.grok/auth.json` (read-only; never
//! written back), refreshing the access token in-memory via the OIDC issuer and client id stored
//! alongside the entry. Requests go to the Grok CLI chat proxy (`cli-chat-proxy.grok.com/v1`),
//! which speaks the OpenAI Responses API.
//!
//! Research-only: third-party use of the Grok CLI OAuth client is unsupported and may break if
//! xAI changes proxy headers or minimum CLI version checks.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chrono::DateTime;
use eyre::{Result, WrapErr};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, USER_AGENT};
use rig_core::ProviderError;
use rig_core::completion::CompletionRequest;
use rig_core::providers::openai;
use rig_core::providers::openai::responses_api::wire::Responses;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::llm::{
    Completion, CompletionResponse, LLMClient, LLMClientDyn, WithRetryExt,
    compact_provider_error_body, provider_http_status,
};

const GROK_PROXY_BASE_URL: &str = "https://cli-chat-proxy.grok.com/v1";
const DEFAULT_XAI_ISSUER: &str = "https://auth.x.ai";
/// Known-good Grok CLI version used when `grok --version` is unavailable and
/// `GROK_CLIENT_VERSION` is unset. The proxy rejects requests missing or below its minimum
/// `x-grok-client-version` with HTTP 426.
const DEFAULT_CLIENT_VERSION: &str = "1.0.92";
const CLIENT_IDENTIFIER: &str = "grok-shell";
const CLIENT_MODE: &str = "headless";
const TOKEN_AUTH_HEADER_VALUE: &str = "xai-grok-cli";
const AUTH_RESPONSE_HEADER_VALUE: &str = "authenticate-response";

/// Refresh this long before the token's stated expiry to avoid racing the deadline.
const EXPIRY_SKEW: Duration = Duration::from_secs(60);
/// Bound discovery and token refresh so a stalled auth request can't pin the token mutex
/// indefinitely.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(30);
/// Give up establishing a connection after this long (dead host / no route).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Idle (per-read) timeout so a hung completion can't pin an `llm_semaphore` permit forever.
const READ_TIMEOUT: Duration = Duration::from_secs(120);

fn parse_grok_version_output(output: &str) -> Option<String> {
    output
        .split_whitespace()
        .find(|tok| {
            let mut parts = tok.split('.');
            matches!(
                (parts.next(), parts.next(), parts.next(), parts.next()),
                (Some(a), Some(b), Some(c), None)
                    if !a.is_empty()
                        && !b.is_empty()
                        && !c.is_empty()
                        && a.bytes().all(|x| x.is_ascii_digit())
                        && b.bytes().all(|x| x.is_ascii_digit())
                        && c.bytes().all(|x| x.is_ascii_digit())
            )
        })
        .map(str::to_string)
}

fn detect_client_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        if let Ok(v) = std::env::var("GROK_CLIENT_VERSION") {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
        if let Ok(out) = std::process::Command::new("grok").arg("--version").output()
            && out.status.success()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Some(ver) = parse_grok_version_output(&text) {
                return ver;
            }
        }
        DEFAULT_CLIENT_VERSION.to_string()
    })
}

fn grok_proxy_headers(client_version: &str) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-xai-token-auth"),
        HeaderValue::from_static(TOKEN_AUTH_HEADER_VALUE),
    );
    headers.insert(
        HeaderName::from_static("x-authenticateresponse"),
        HeaderValue::from_static(AUTH_RESPONSE_HEADER_VALUE),
    );
    headers.insert(
        HeaderName::from_static("x-grok-client-version"),
        HeaderValue::from_str(client_version)
            .wrap_err("invalid x-grok-client-version header value")?,
    );
    headers.insert(
        HeaderName::from_static("x-grok-client-identifier"),
        HeaderValue::from_static(CLIENT_IDENTIFIER),
    );
    headers.insert(
        HeaderName::from_static("x-grok-client-mode"),
        HeaderValue::from_static(CLIENT_MODE),
    );
    let ua = format!(
        "{CLIENT_IDENTIFIER}/{client_version} ({}; {})",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    headers.insert(
        USER_AGENT,
        HeaderValue::from_str(&ua).wrap_err("invalid Grok User-Agent header value")?,
    );
    Ok(headers)
}

fn build_proxy_http_client() -> Result<reqwest::Client> {
    let version = detect_client_version();
    reqwest::Client::builder()
        .default_headers(grok_proxy_headers(version)?)
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .wrap_err("building Grok proxy HTTP client")
}

fn build_auth_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(REFRESH_TIMEOUT)
        .build()
        .wrap_err("building Grok auth HTTP client")
}

// ============================================================================
// JWT / expiry helpers (no signature verification — public claims only)
// ============================================================================

fn decode_jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn expiry_from_access_token(access_token: &str) -> SystemTime {
    decode_jwt_claims(access_token)
        .and_then(|c| c.get("exp").and_then(Value::as_u64))
        .and_then(|exp| UNIX_EPOCH.checked_add(Duration::from_secs(exp)))
        .unwrap_or(UNIX_EPOCH)
}

fn parse_rfc3339_system_time(raw: &str) -> Option<SystemTime> {
    let dt = DateTime::parse_from_rfc3339(raw).ok()?;
    let secs = u64::try_from(dt.timestamp()).ok()?;
    let nanos = dt.timestamp_subsec_nanos();
    UNIX_EPOCH.checked_add(Duration::new(secs, nanos))
}

// ============================================================================
// Token store (~/.grok/auth.json)
// ============================================================================

#[derive(Clone)]
struct GrokTokens {
    access_token: String,
    refresh_token: String,
    oidc_issuer: String,
    oidc_client_id: String,
    principal_type: Option<String>,
    principal_id: Option<String>,
    expires_at: SystemTime,
}

impl GrokTokens {
    fn is_expired(&self) -> bool {
        self.expires_at <= SystemTime::now() + EXPIRY_SKEW
    }
}

/// `$GROK_HOME/auth.json` (when `GROK_HOME` is set, non-empty, absolute) else `~/.grok/auth.json`.
fn auth_path() -> Result<PathBuf> {
    resolve_auth_path(std::env::var("GROK_HOME").ok().as_deref(), dirs::home_dir())
}

fn resolve_auth_path(grok_home: Option<&str>, home_dir: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(home) = grok_home {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            let path = Path::new(trimmed);
            if path.is_absolute() {
                return Ok(path.join("auth.json"));
            }
            eyre::bail!("GROK_HOME must be an absolute path, got \"{home}\"");
        }
    }
    let home =
        home_dir.ok_or_else(|| eyre::eyre!("cannot resolve home directory (set GROK_HOME)"))?;
    Ok(home.join(".grok").join("auth.json"))
}

fn load_tokens(path: &Path) -> Result<GrokTokens> {
    let raw = std::fs::read_to_string(path)
        .wrap_err_with(|| format!("cannot read {} — run `grok login` first", path.display()))?;
    parse_tokens(&raw, &path.display().to_string())
}

/// True when a usable Grok CLI OAuth token file is present (`$GROK_HOME` or `~/.grok/auth.json`).
/// Used by `init` detection; reuses the same parse the client does.
pub fn auth_available() -> bool {
    auth_path().and_then(|p| load_tokens(&p)).is_ok()
}

fn parse_entry(entry: &Value) -> Option<GrokTokens> {
    let access_token = entry
        .get("key")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?
        .to_string();
    let refresh_token = entry
        .get("refresh_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?
        .to_string();
    let oidc_issuer = entry
        .get("oidc_issuer")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?
        .trim_end_matches('/')
        .to_string();
    let oidc_client_id = entry
        .get("oidc_client_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?
        .to_string();
    let principal_type = entry
        .get("principal_type")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let principal_id = entry
        .get("principal_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let expires_at = entry
        .get("expires_at")
        .and_then(Value::as_str)
        .and_then(parse_rfc3339_system_time)
        .unwrap_or_else(|| expiry_from_access_token(&access_token));
    Some(GrokTokens {
        access_token,
        refresh_token,
        oidc_issuer,
        oidc_client_id,
        principal_type,
        principal_id,
        expires_at,
    })
}

/// Parse a Grok CLI `auth.json` payload (map of scope key → credential entry).
/// Prefers first-party `https://auth.x.ai` OIDC entries; falls back to any valid OIDC entry with
/// the latest `expires_at`.
fn parse_tokens(raw: &str, source: &str) -> Result<GrokTokens> {
    let v: Value =
        serde_json::from_str(raw).wrap_err_with(|| format!("invalid JSON in {source}"))?;
    let map = v
        .as_object()
        .ok_or_else(|| eyre::eyre!("expected a JSON object in {source}"))?;

    let mut best: Option<(bool, SystemTime, GrokTokens)> = None;
    for entry in map.values() {
        let Some(tokens) = parse_entry(entry) else {
            continue;
        };
        let is_default_issuer = tokens.oidc_issuer == DEFAULT_XAI_ISSUER;
        let dominated = match &best {
            None => true,
            Some((best_default, best_exp, _)) => {
                (is_default_issuer, tokens.expires_at) > (*best_default, *best_exp)
            }
        };
        if dominated {
            best = Some((is_default_issuer, tokens.expires_at, tokens));
        }
    }

    best.map(|(_, _, tokens)| tokens)
        .ok_or_else(|| eyre::eyre!("no Grok CLI OIDC tokens in {source} — run `grok login` first"))
}

#[derive(serde::Deserialize)]
struct OidcDiscovery {
    token_endpoint: String,
}

#[derive(serde::Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

async fn discover_token_endpoint(http: &reqwest::Client, issuer: &str) -> Result<String> {
    let url = format!(
        "{}/.well-known/openid-configuration",
        issuer.trim_end_matches('/')
    );
    let resp = http
        .get(&url)
        .timeout(REFRESH_TIMEOUT)
        .send()
        .await
        .wrap_err_with(|| format!("Grok OIDC discovery request to {url} failed"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(eyre::Report::new(ProviderError::from_http_response(
            status,
            compact_provider_error_body(&body, 1000),
        )))
        .wrap_err("Grok OIDC discovery failed");
    }
    let parsed: OidcDiscovery =
        serde_json::from_str(&body).wrap_err("parsing Grok OIDC discovery response")?;
    eyre::ensure!(
        !parsed.token_endpoint.trim().is_empty(),
        "Grok OIDC discovery returned empty token_endpoint"
    );
    Ok(parsed.token_endpoint)
}

async fn refresh_tokens(
    http: &reqwest::Client,
    token_endpoint: &str,
    current: &GrokTokens,
) -> Result<GrokTokens> {
    let mut params: Vec<(&str, &str)> = vec![
        ("grant_type", "refresh_token"),
        ("refresh_token", &current.refresh_token),
        ("client_id", &current.oidc_client_id),
    ];
    if let Some(pt) = current.principal_type.as_deref() {
        params.push(("principal_type", pt));
    }
    if let Some(pid) = current.principal_id.as_deref() {
        params.push(("principal_id", pid));
    }

    let resp = http
        .post(token_endpoint)
        .timeout(REFRESH_TIMEOUT)
        .form(&params)
        .send()
        .await
        .wrap_err("Grok token refresh request failed")?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(eyre::Report::new(ProviderError::from_http_response(
            status,
            compact_provider_error_body(&body, 1000),
        )))
        .wrap_err("Grok token refresh failed");
    }
    let parsed: RefreshResponse =
        serde_json::from_str(&body).wrap_err("parsing Grok token refresh response")?;
    let expires_at = match parsed.expires_in {
        Some(secs) => SystemTime::now()
            .checked_add(Duration::from_secs(secs))
            .unwrap_or_else(|| expiry_from_access_token(&parsed.access_token)),
        None => expiry_from_access_token(&parsed.access_token),
    };
    Ok(GrokTokens {
        access_token: parsed.access_token,
        refresh_token: parsed
            .refresh_token
            .unwrap_or_else(|| current.refresh_token.clone()),
        oidc_issuer: current.oidc_issuer.clone(),
        oidc_client_id: current.oidc_client_id.clone(),
        principal_type: current.principal_type.clone(),
        principal_id: current.principal_id.clone(),
        expires_at,
    })
}

// ============================================================================
// Client
// ============================================================================

struct TokenState {
    tokens: GrokTokens,
    /// Cached `(issuer, token_endpoint)` from OIDC discovery.
    token_endpoint: Option<(String, String)>,
}

pub struct GrokClient {
    proxy_http: reqwest::Client,
    auth_http: reqwest::Client,
    auth_path: PathBuf,
    state: Mutex<TokenState>,
}

impl GrokClient {
    /// Load tokens from `~/.grok/auth.json` (or `$GROK_HOME`) and build the client.
    pub fn new() -> Result<Self> {
        let path = auth_path()?;
        let tokens = load_tokens(&path)?;
        Ok(Self {
            proxy_http: build_proxy_http_client()?,
            auth_http: build_auth_http_client()?,
            auth_path: path,
            state: Mutex::new(TokenState {
                tokens,
                token_endpoint: None,
            }),
        })
    }

    async fn resolve_token_endpoint(&self, state: &mut TokenState, issuer: &str) -> Result<String> {
        if let Some((cached_issuer, cached_endpoint)) = &state.token_endpoint
            && cached_issuer == issuer
        {
            return Ok(cached_endpoint.clone());
        }
        let endpoint = discover_token_endpoint(&self.auth_http, issuer).await?;
        state.token_endpoint = Some((issuer.to_string(), endpoint.clone()));
        Ok(endpoint)
    }

    /// Refresh under the held lock. If the refresh token was rejected (4xx other than 429), the
    /// Grok CLI may have rotated it on disk concurrently — reload `auth.json` once and retry
    /// before giving up.
    async fn refresh_locked(&self, state: &mut TokenState) -> Result<()> {
        let issuer = state.tokens.oidc_issuer.clone();
        let endpoint = self.resolve_token_endpoint(state, &issuer).await?;
        match refresh_tokens(&self.auth_http, &endpoint, &state.tokens).await {
            Ok(fresh) => {
                state.tokens = fresh;
                Ok(())
            }
            Err(err) if is_rotation_4xx(&err) => {
                let disk = load_tokens(&self.auth_path)
                    .wrap_err("Grok refresh token rejected and reloading auth.json failed")?;
                let disk_endpoint = self
                    .resolve_token_endpoint(state, &disk.oidc_issuer)
                    .await?;
                let fresh = refresh_tokens(&self.auth_http, &disk_endpoint, &disk)
                    .await
                    .wrap_err(
                        "Grok refresh failed even after reloading auth.json (run `grok login`)",
                    )?;
                state.tokens = fresh;
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    async fn current_access(&self) -> Result<String> {
        let mut guard = self.state.lock().await;
        if guard.tokens.is_expired() {
            self.refresh_locked(&mut guard).await?;
        }
        Ok(guard.tokens.access_token.clone())
    }

    async fn force_refresh(&self, used_access: &str) -> Result<String> {
        let mut guard = self.state.lock().await;
        if guard.tokens.access_token != used_access {
            return Ok(guard.tokens.access_token.clone());
        }
        self.refresh_locked(&mut guard).await?;
        Ok(guard.tokens.access_token.clone())
    }

    async fn send(&self, completion: &Completion, access: &str) -> Result<CompletionResponse> {
        let wire = grok_wire(&completion.model, access);
        let model = rig_core::Model::new(
            wire,
            rig_reqwest::ReqwestClient::from(self.proxy_http.clone()),
        );
        let response = model
            .call(grok_request(completion))
            .await
            .wrap_err("Grok completion failed")?;
        Ok(CompletionResponse::from_rig(
            response,
            completion.model.clone(),
        ))
    }
}

/// Process-wide shared Grok client (retry-wrapped), built once on first use.
///
/// Mirrors `codex::shared_client`: `~/.grok/auth.json` holds a single live token chain, so every
/// Grok reviewer/aggregator shares one token cache and concurrent-refresh lock.
pub fn shared_client() -> Result<Arc<dyn LLMClientDyn>> {
    static CELL: OnceLock<StdMutex<Option<Arc<dyn LLMClientDyn>>>> = OnceLock::new();
    let cell = CELL.get_or_init(|| StdMutex::new(None));
    let mut guard = cell.lock().expect("grok shared-client mutex poisoned");
    if let Some(existing) = guard.as_ref() {
        return Ok(Arc::clone(existing));
    }
    let client = GrokClient::new()?.with_retry().into_arc();
    *guard = Some(Arc::clone(&client));
    Ok(client)
}

impl LLMClient for GrokClient {
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        let access = self.current_access().await?;
        match self.send(&completion, &access).await {
            Ok(response) => Ok(response),
            Err(err) if provider_http_status(&err) == Some(401) => {
                let access = self.force_refresh(&access).await?;
                self.send(&completion, &access).await
            }
            Err(err) => Err(err),
        }
    }
}

// ============================================================================
// Request/response mapping
// ============================================================================

fn grok_wire(model: &str, access: &str) -> Responses {
    let config = openai::OpenAIConfig::new(access).with_base_url(GROK_PROXY_BASE_URL);
    Responses::new(config, model)
}

fn grok_request(completion: &Completion) -> CompletionRequest {
    let mut request = CompletionRequest::from(completion.clone());
    // Keep server-side storage disabled so multi-turn replay stays stateless, matching Grok CLI.
    let store_false = serde_json::json!({ "store": false });
    request.additional_params = Some(match request.additional_params.take() {
        Some(serde_json::Value::Object(mut map)) => {
            map.entry("store".to_string())
                .or_insert(serde_json::Value::Bool(false));
            serde_json::Value::Object(map)
        }
        Some(other) => other,
        None => store_false,
    });
    request
}

fn is_rotation_4xx(err: &eyre::Report) -> bool {
    provider_http_status(err).is_some_and(|status| (400..500).contains(&status) && status != 429)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::message::Message;
    use serde_json::json;

    fn jwt(claims: Value) -> String {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"none"}"#);
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
        format!("{header}.{payload}.sig")
    }

    #[test]
    fn parse_version_extracts_semver_token() {
        assert_eq!(
            parse_grok_version_output("grok 1.0.92 (b06756fd6256) [stable]").as_deref(),
            Some("1.0.92")
        );
        assert_eq!(parse_grok_version_output("1.2.3").as_deref(), Some("1.2.3"));
        assert_eq!(parse_grok_version_output("grok dev build"), None);
    }

    #[test]
    fn resolve_auth_path_respects_grok_home_and_rejects_relative() {
        let p = resolve_auth_path(Some("/custom/grok"), Some(PathBuf::from("/home/u"))).unwrap();
        assert_eq!(p, PathBuf::from("/custom/grok/auth.json"));

        let p = resolve_auth_path(None, Some(PathBuf::from("/home/u"))).unwrap();
        assert_eq!(p, PathBuf::from("/home/u/.grok/auth.json"));

        let p = resolve_auth_path(Some("   "), Some(PathBuf::from("/home/u"))).unwrap();
        assert_eq!(p, PathBuf::from("/home/u/.grok/auth.json"));

        assert!(resolve_auth_path(Some("relative/grok"), Some(PathBuf::from("/home/u"))).is_err());
    }

    #[test]
    fn parse_tokens_reads_oidc_entry_and_prefers_default_issuer() {
        let access = jwt(json!({ "exp": 2_000_000_000u64 }));
        let raw = json!({
            "https://other.example::cid": {
                "key": access,
                "auth_mode": "oidc",
                "refresh_token": "rt-other",
                "expires_at": "2099-01-01T00:00:00Z",
                "oidc_issuer": "https://other.example",
                "oidc_client_id": "cid-other"
            },
            "https://auth.x.ai::cid": {
                "key": access,
                "auth_mode": "oidc",
                "refresh_token": "rt-xai",
                "expires_at": "2026-11-25T16:41:39.926269Z",
                "oidc_issuer": "https://auth.x.ai/",
                "oidc_client_id": "cid-xai",
                "principal_type": "User",
                "principal_id": "uid-1"
            }
        })
        .to_string();

        let tokens = parse_tokens(&raw, "test").unwrap();
        assert_eq!(tokens.refresh_token, "rt-xai");
        assert_eq!(tokens.oidc_issuer, "https://auth.x.ai");
        assert_eq!(tokens.oidc_client_id, "cid-xai");
        assert_eq!(tokens.principal_type.as_deref(), Some("User"));
        assert_eq!(tokens.principal_id.as_deref(), Some("uid-1"));
        assert!(tokens.expires_at > UNIX_EPOCH);
    }

    #[test]
    fn parse_tokens_rejects_empty_or_missing_refresh() {
        assert!(parse_tokens("{}", "test").is_err());
        let raw = json!({
            "legacy": {
                "key": "some-key",
                "auth_mode": "legacy"
            }
        })
        .to_string();
        assert!(parse_tokens(&raw, "test").is_err());
    }

    #[test]
    fn grok_proxy_headers_include_required_cli_headers() {
        let headers = grok_proxy_headers("1.0.92").unwrap();
        assert_eq!(headers.get("x-xai-token-auth").unwrap(), "xai-grok-cli");
        assert_eq!(
            headers.get("x-authenticateresponse").unwrap(),
            "authenticate-response"
        );
        assert_eq!(headers.get("x-grok-client-version").unwrap(), "1.0.92");
        assert_eq!(
            headers.get("x-grok-client-identifier").unwrap(),
            "grok-shell"
        );
        assert_eq!(headers.get("x-grok-client-mode").unwrap(), "headless");
        let ua = headers.get(USER_AGENT).unwrap().to_str().unwrap();
        assert!(ua.starts_with("grok-shell/1.0.92 ("));
    }

    #[test]
    fn grok_wire_encodes_responses_request_with_store_false() {
        use rig_core::wire::{Body, Mode, Wire};

        let completion = Completion {
            model: "grok-4.7".to_string(),
            prompt: Message::user("ping"),
            preamble: Some("You are terse.".to_string()),
            history: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            max_tokens: None,
            additional_params: None,
        };
        let wire = grok_wire(&completion.model, "unused");
        let encoded = wire
            .encode(grok_request(&completion), Mode::Unary)
            .expect("encoding Grok Responses request");
        assert_eq!(
            encoded.request.uri().to_string(),
            "https://cli-chat-proxy.grok.com/v1/responses"
        );
        let Body::Bytes(bytes) = encoded.request.into_body() else {
            panic!("expected a JSON body")
        };
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["model"], "grok-4.7");
        assert_eq!(body["store"], false);
        assert_eq!(body["instructions"], "You are terse.");
    }
}
