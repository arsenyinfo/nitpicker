//! Azure AD (Entra ID) token authentication for Foundry-hosted OpenAI/Anthropic models.
//!
//! Acquires a short-lived bearer token via the Azure SDK and transparently refreshes it,
//! rebuilding the underlying rig client when the token nears expiry. This is the analogue of
//! the Python SDK's `azure_ad_token_provider` callback.
//!
//! Gated behind the `azure` cargo feature (see Cargo.toml) — the whole module is only compiled
//! when that feature is enabled.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use azure_core::credentials::{Secret, TokenCredential};
use azure_identity::{ClientSecretCredential, DeveloperToolsCredential, ManagedIdentityCredential};
use eyre::{Result, WrapErr};
use rig_core::providers::{anthropic, openai};
use tokio::sync::Mutex;

use crate::config::ProviderType;
use crate::llm::{Completion, CompletionResponse, LLMClient, LLMClientDyn, mentions_http_status};

/// Default AAD scope for Azure AI Foundry / Cognitive Services.
pub const DEFAULT_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

/// Refresh slightly before the token actually expires to avoid using a token that lapses
/// mid-request. Mirrors the 60s skew used by the Gemini proxy token store.
const EXPIRY_SKEW_SECS: i64 = 60;

type RebuildFn = Box<dyn Fn(&str) -> Result<Arc<dyn LLMClientDyn>> + Send + Sync>;

struct Cached {
    client: Arc<dyn LLMClientDyn>,
    expires_at_unix: i64,
}

/// An [`LLMClient`] that authenticates via an Azure AD bearer token and rebuilds the inner rig
/// client whenever the cached token nears expiry. Each reviewer/aggregator owns its own instance.
pub struct AzureAdClient {
    credentials: Vec<Arc<dyn TokenCredential>>,
    scope: String,
    rebuild: RebuildFn,
    cache: Mutex<Option<Cached>>,
}

impl AzureAdClient {
    /// Try each credential in the chain in order, returning the first token obtained.
    async fn fetch_token(&self) -> Result<(String, i64)> {
        let scopes = [self.scope.as_str()];
        let mut errors = Vec::new();
        for cred in &self.credentials {
            match cred.get_token(&scopes, None).await {
                Ok(token) => {
                    return Ok((
                        token.token.secret().to_string(),
                        token.expires_on.unix_timestamp(),
                    ));
                }
                Err(err) => errors.push(format_error_chain(&err)),
            }
        }
        eyre::bail!(
            "no Azure credential could obtain a token for scope {}: {}",
            self.scope,
            errors.join("; ")
        );
    }

    /// Return a client backed by a non-expired token, refreshing (and rebuilding) if needed.
    ///
    /// Holds the cache mutex across the whole check-then-refresh so concurrent callers (e.g. a
    /// reviewer's parallel subagents, which share this client) don't each kick off a redundant
    /// token fetch — double-checked locking, with the lock also serializing the in-flight refresh.
    async fn ensure_client(&self) -> Result<Arc<dyn LLMClientDyn>> {
        let mut guard = self.cache.lock().await;
        if let Some(cached) = guard.as_ref() {
            if cached.expires_at_unix - now_unix() > EXPIRY_SKEW_SECS {
                return Ok(Arc::clone(&cached.client));
            }
        }
        let (client, cached) = self.build_fresh().await?;
        *guard = Some(cached);
        Ok(client)
    }

    /// Refresh after a 401 and replace the cache. Unlike `ensure_client`, this can't gate on
    /// expiry — the token was rejected *despite* our clock thinking it valid (revoked or lapsed
    /// early), so re-checking expiry would wrongly conclude "still fresh" and hand back the same
    /// bad client. Instead it gates on client identity: `stale` is the client that just 401'd, and
    /// if a concurrent 401 already rebuilt the cached client, this caller reuses that one rather
    /// than firing a redundant token fetch. The lock is held across `build_fresh`, so a burst of
    /// concurrent 401s triggers exactly one fetch (the rest short-circuit on the changed pointer).
    async fn refresh(&self, stale: &Arc<dyn LLMClientDyn>) -> Result<Arc<dyn LLMClientDyn>> {
        let mut guard = self.cache.lock().await;
        if let Some(cached) = guard.as_ref() {
            if !Arc::ptr_eq(&cached.client, stale) {
                // Another concurrent 401 already refreshed since `stale` was handed out — reuse it.
                return Ok(Arc::clone(&cached.client));
            }
        }
        let (client, cached) = self.build_fresh().await?;
        *guard = Some(cached);
        Ok(client)
    }

    /// Fetch a token and rebuild the client, returning the client plus the cache entry to store.
    /// Does not touch the cache mutex, so the caller controls locking.
    async fn build_fresh(&self) -> Result<(Arc<dyn LLMClientDyn>, Cached)> {
        let (token, expires_at_unix) = self.fetch_token().await?;
        let client = (self.rebuild)(&token)?;
        let cached = Cached {
            client: Arc::clone(&client),
            expires_at_unix,
        };
        Ok((client, cached))
    }
}

impl LLMClient for AzureAdClient {
    async fn completion(&self, completion: Completion) -> Result<CompletionResponse> {
        let client = self.ensure_client().await?;
        match client.completion(completion.clone()).await {
            Ok(response) => Ok(response),
            // A token can lapse or be revoked between our expiry check and the call landing.
            // Force a single refresh-and-retry rather than surfacing a 401 (which the outer
            // RetryingLLM treats as non-retryable).
            Err(err) if is_unauthorized(&err) => {
                let client = self.refresh(&client).await?;
                client.completion(completion).await
            }
            Err(err) => Err(err),
        }
    }
}

/// Build a refreshing Azure AD client for an OpenAI- or Anthropic-shaped Foundry endpoint.
///
/// `base_url` is the Foundry endpoint (e.g. `https://<resource>.services.ai.azure.com/openai/v1`
/// or `.../anthropic`). `scope` defaults to [`DEFAULT_SCOPE`]; `credentials` selects the
/// credential chain (`"dev"`, `"prod"`, or unset/`"auto"`).
pub fn build_azure_client(
    provider: &ProviderType,
    base_url: Option<&str>,
    scope: Option<&str>,
    credentials: Option<&str>,
) -> Result<AzureAdClient> {
    let base_url = resolve_base_url(base_url)?;
    let scope = resolve_scope(scope);
    let credentials = build_credential_chain(credentials)?;

    let rebuild: RebuildFn = match provider {
        ProviderType::OpenAi => {
            let base = base_url.clone();
            Box::new(move |token: &str| {
                // OpenAI's rig client uses Bearer auth, so the AAD token slots straight in.
                let client = openai::CompletionsClient::builder()
                    .api_key(token)
                    .base_url(&base)
                    .build()
                    .wrap_err("failed to build Azure OpenAI client")?;
                Ok(client.into_arc())
            })
        }
        ProviderType::Anthropic => {
            let base = base_url.clone();
            Box::new(move |token: &str| {
                // The Anthropic rig client hardcodes the `x-api-key` header, so inject the AAD
                // bearer via custom headers; `build()` preserves it alongside `anthropic-version`.
                // Foundry's Anthropic gateway authenticates on `Authorization: Bearer`, so pass a
                // placeholder (not the real token) to `.api_key(...)` to avoid leaking the AAD JWT
                // into the unused `x-api-key` header.
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {token}")
                        .parse()
                        .wrap_err("invalid Azure bearer token header")?,
                );
                let client = anthropic::Client::builder()
                    .api_key("azure-ad")
                    .base_url(&base)
                    .http_headers(headers)
                    .build()
                    .wrap_err("failed to build Azure Anthropic client")?;
                Ok(client.into_arc())
            })
        }
        _ => eyre::bail!(
            "auth = \"azure-ad\" is only supported with provider \"openai\" or \"anthropic\""
        ),
    };

    Ok(AzureAdClient {
        credentials,
        scope,
        rebuild,
        cache: Mutex::new(None),
    })
}

/// Resolve and validate the Foundry `base_url`, trimming surrounding whitespace. The config
/// validator (`validate_azure_fields`) trims before its emptiness check, so a whitespace-padded URL
/// passes validation; trimming here too keeps the two in step and stops the padded value from
/// reaching rig's `.base_url()` verbatim (which would build a malformed endpoint). Empty/whitespace
/// is rejected — unlike `azure_scope`, the endpoint has no default to fall back to.
fn resolve_base_url(base_url: Option<&str>) -> Result<String> {
    base_url
        .map(str::trim)
        .filter(|u| !u.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            eyre::eyre!("auth = \"azure-ad\" requires `base_url` (the Azure Foundry endpoint)")
        })
}

/// Resolve the AAD scope, treating an empty/whitespace `azure_scope` as unset so it falls back to
/// [`DEFAULT_SCOPE`]. A plain `unwrap_or` wouldn't substitute for `Some("")`/`Some("  ")`, so an
/// empty scope would otherwise reach `get_token` and fail only at the first LLM call. The field has
/// a documented default, so defaulting an empty value is more useful than rejecting it at config
/// time (cf. the mandatory `base_url`, which is rejected when empty).
fn resolve_scope(scope: Option<&str>) -> String {
    scope
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_SCOPE)
        .to_string()
}

/// Render an error together with its `source()` chain. The Azure SDK's error type's `Display` ignores
/// the alternate (`{:#}`) flag and prints only the top frame — unlike `eyre::Report`, whose `{:#}`
/// joins the chain — so a plain `to_string()` (or even `{err:#}`) would drop the underlying HTTP
/// status / AAD error code when a credential fails. Walk `source()` by hand to keep that detail in
/// the joined "no credential could obtain a token" message.
fn format_error_chain(err: &dyn std::error::Error) -> String {
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        out.push_str(": ");
        out.push_str(&cause.to_string());
        source = cause.source();
    }
    out
}

/// Build the ordered credential chain for the requested mode. Mirrors the Azure SDK's
/// `AZURE_TOKEN_CREDENTIALS` convention: explicit config wins, then the env var, else `"auto"`.
fn build_credential_chain(mode: Option<&str>) -> Result<Vec<Arc<dyn TokenCredential>>> {
    let mode = mode
        .map(str::to_string)
        .or_else(|| std::env::var("AZURE_TOKEN_CREDENTIALS").ok())
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "auto".to_string());

    let mut chain: Vec<Arc<dyn TokenCredential>> = Vec::new();
    match mode.as_str() {
        // Developer tools only (Azure CLI → Azure Developer CLI), excluding managed identity. Use
        // on a VM where you want `az login` instead of a system-assigned managed identity.
        // Construction failures are skipped (then caught by the `is_empty` guard below) rather than
        // fatal, matching the `auto` branch.
        "dev" => {
            if let Ok(dev) = DeveloperToolsCredential::new(None) {
                chain.push(dev);
            }
        }
        // Production identities only: env service principal, then managed identity.
        "prod" => {
            push_env_service_principal(&mut chain);
            if let Ok(mi) = ManagedIdentityCredential::new(None) {
                chain.push(mi);
            }
        }
        // Default chain: env service principal → managed identity → developer tools. Credentials
        // whose construction fails for this environment are skipped rather than fatal.
        "auto" => {
            push_env_service_principal(&mut chain);
            if let Ok(mi) = ManagedIdentityCredential::new(None) {
                chain.push(mi);
            }
            if let Ok(dev) = DeveloperToolsCredential::new(None) {
                chain.push(dev);
            }
        }
        other => {
            eyre::bail!(
                "unknown azure_credentials value \"{other}\" — expected \"dev\", \"prod\", or unset (\"auto\")"
            );
        }
    }

    if chain.is_empty() {
        eyre::bail!("no Azure credentials could be constructed for mode \"{mode}\"");
    }
    Ok(chain)
}

/// Add a service-principal credential built from `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` /
/// `AZURE_CLIENT_SECRET` if all three are set (the Azure SDK's "environment credential").
fn push_env_service_principal(chain: &mut Vec<Arc<dyn TokenCredential>>) {
    let (Ok(tenant), Ok(client_id), Ok(secret)) = (
        std::env::var("AZURE_TENANT_ID"),
        std::env::var("AZURE_CLIENT_ID"),
        std::env::var("AZURE_CLIENT_SECRET"),
    ) else {
        return;
    };
    if let Ok(cred) = ClientSecretCredential::new(&tenant, client_id, Secret::new(secret), None) {
        chain.push(cred);
    }
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn is_unauthorized(err: &eyre::Report) -> bool {
    // The status lives in the wrapped source (rig surfaces the raw response body as `ProviderError`,
    // and `llm.rs` wraps it with `.wrap_err_with(...)`), so we must walk the whole chain —
    // `err.to_string()` would only render the top-level context. `{err:#}` is the alternate Display
    // that joins the full chain.
    //
    // Key on the numeric 401 only, NOT the bare word "unauthorized": a 403 (e.g. an RBAC/deployment
    // denial whose body reads "Unauthorized to use this deployment") must not trigger a refresh —
    // re-minting the same identity's token can't fix a permissions error, it just wastes a token
    // fetch + a retried completion. `mentions_http_status` matches the status across the JSON/HTTP
    // shapings it can take in a raw body (`: 401`, `:401`, `401,`, `"401"`, ...), shared with the
    // `llm.rs` retry classifiers so the two paths agree on what a 401 looks like.
    let msg = format!("{err:#}");
    if mentions_http_status(&msg, 401) {
        return true;
    }
    // A Foundry route fronting OpenAI/Anthropic can return a provider-style auth body with no numeric
    // status (rig drops it). Match the auth-specific error *types* only — NOT the 403 permission
    // types, which a token refresh can't fix.
    let lower = msg.to_ascii_lowercase();
    lower.contains("authentication_error") || lower.contains("invalid_api_key")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror how a Foundry 401 reaches `completion`: rig returns the raw body as `ProviderError`,
    /// wrapped by `llm.rs` via `.wrap_err_with(...)`. The status text is only in the source.
    fn wrapped_provider_error(body: &str) -> eyre::Report {
        Err::<(), _>(eyre::eyre!("ProviderError: {body}"))
            .wrap_err("Anthropic completion failed for model 'claude'")
            .unwrap_err()
    }

    #[test]
    fn is_unauthorized_walks_the_chain() {
        let err = wrapped_provider_error(r#"{"statusCode": 401, "message": "Unauthorized"}"#);
        // Top-level Display hides the status; the chain-walk must still catch it.
        let top_level = err.to_string().to_ascii_lowercase();
        assert!(!top_level.contains("unauthorized"));
        assert!(is_unauthorized(&err));
    }

    #[test]
    fn is_unauthorized_false_for_unrelated_error() {
        let err = wrapped_provider_error(r#"{"statusCode": 500, "message": "boom"}"#);
        assert!(!is_unauthorized(&err));
    }

    #[test]
    fn is_unauthorized_detects_compact_json_status() {
        // A compact 401 body (no space after the colon) must still trigger the refresh-and-retry —
        // `:401,` slips past a plain `" 401"` substring check.
        let err = wrapped_provider_error(r#"{"statusCode":401,"message":"token expired"}"#);
        assert!(is_unauthorized(&err));
    }

    #[test]
    fn is_unauthorized_false_for_403_with_unauthorized_text() {
        // A 403 (deployment/RBAC denial) whose body literally says "unauthorized" must not be
        // mistaken for a 401 — refreshing the token can't fix a permissions error, so triggering a
        // refresh-and-retry would just waste a token fetch before the real 403 propagates.
        let err = wrapped_provider_error(
            r#"{"statusCode": 403, "message": "Unauthorized to use this deployment"}"#,
        );
        assert!(!is_unauthorized(&err));
    }

    #[test]
    fn is_unauthorized_detects_provider_auth_types_without_status() {
        // A Foundry route fronting OpenAI/Anthropic returns an auth body with no numeric status.
        let openai = wrapped_provider_error(r#"{"error":{"code":"invalid_api_key"}}"#);
        assert!(is_unauthorized(&openai));
        let anthropic =
            wrapped_provider_error(r#"{"type":"error","error":{"type":"authentication_error"}}"#);
        assert!(is_unauthorized(&anthropic));
        // A 403-class permission type is NOT an auth failure a refresh can fix.
        let permission = wrapped_provider_error(r#"{"error":{"type":"permission_error"}}"#);
        assert!(!is_unauthorized(&permission));
    }

    #[test]
    fn resolve_scope_falls_back_on_empty_or_whitespace() {
        // `azure_scope` is a defaulted field: empty/whitespace means "unset", so fall back to the
        // documented default rather than letting an empty scope reach `get_token` and fail late.
        assert_eq!(resolve_scope(None), DEFAULT_SCOPE);
        assert_eq!(resolve_scope(Some("")), DEFAULT_SCOPE);
        assert_eq!(resolve_scope(Some("   ")), DEFAULT_SCOPE);
        let custom = "https://example.com/.default";
        assert_eq!(resolve_scope(Some(custom)), custom);
        // surrounding whitespace on a real scope is trimmed off
        assert_eq!(
            resolve_scope(Some("  https://example.com/.default  ")),
            custom
        );
    }

    #[test]
    fn resolve_base_url_trims_and_rejects_empty() {
        // A padded URL passes config validation (which trims too); the runtime must trim it as well
        // so rig never receives the untrimmed string and builds a malformed endpoint.
        let url = "https://res.services.ai.azure.com/openai/v1";
        assert_eq!(resolve_base_url(Some(url)).unwrap(), url);
        assert_eq!(
            resolve_base_url(Some("  https://res.services.ai.azure.com/openai/v1  ")).unwrap(),
            url
        );
        // No default for the endpoint — empty/whitespace/unset is a hard error.
        assert!(resolve_base_url(None).is_err());
        assert!(resolve_base_url(Some("")).is_err());
        assert!(resolve_base_url(Some("   ")).is_err());
    }

    #[test]
    fn format_error_chain_joins_sources() {
        // The Azure SDK error's Display prints only the top frame; `format_error_chain` must append
        // each `source()` so a credential failure's underlying cause survives into the bail message.
        #[derive(Debug)]
        struct Inner;
        impl std::fmt::Display for Inner {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("az login required")
            }
        }
        impl std::error::Error for Inner {}

        #[derive(Debug)]
        struct Outer(Inner);
        impl std::fmt::Display for Outer {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("credential failed")
            }
        }
        impl std::error::Error for Outer {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }

        assert_eq!(
            format_error_chain(&Outer(Inner)),
            "credential failed: az login required"
        );
        assert_eq!(format_error_chain(&Inner), "az login required");
    }

    /// A no-op client used only to populate the cache for the refresh-dedup test; its `completion`
    /// is never invoked on the short-circuit path under test.
    struct DummyClient;
    impl LLMClient for DummyClient {
        async fn completion(&self, _completion: Completion) -> Result<CompletionResponse> {
            eyre::bail!("dummy client should not be called")
        }
    }

    #[tokio::test]
    async fn refresh_reuses_cache_when_another_call_already_refreshed() {
        // Simulate the concurrent-401 race: the cache already holds a freshly-rebuilt client, and a
        // slower caller arrives carrying the *stale* client it had 401'd on. Because the cached
        // client no longer matches `stale`, `refresh` must hand it back without a token fetch — so
        // `rebuild` (which would do network I/O) must never fire.
        let cached_client: Arc<dyn LLMClientDyn> = DummyClient.into_arc();
        let stale_client: Arc<dyn LLMClientDyn> = DummyClient.into_arc();
        let rebuild: RebuildFn =
            Box::new(|_: &str| panic!("rebuild must not run when another refresh already won"));
        let client = AzureAdClient {
            credentials: Vec::new(),
            scope: DEFAULT_SCOPE.to_string(),
            rebuild,
            cache: Mutex::new(Some(Cached {
                client: Arc::clone(&cached_client),
                expires_at_unix: now_unix() + 3600,
            })),
        };

        let returned = client.refresh(&stale_client).await.unwrap();
        assert!(Arc::ptr_eq(&returned, &cached_client));
    }
}
