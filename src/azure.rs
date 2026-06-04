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
use crate::llm::{Completion, CompletionResponse, LLMClient, LLMClientDyn};

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
                Err(err) => errors.push(err.to_string()),
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

    /// Acquire a fresh token, rebuild the rig client, and replace the cache. Takes the lock itself,
    /// so it's used on the 401 fallback path where no lock is already held.
    async fn refresh(&self) -> Result<Arc<dyn LLMClientDyn>> {
        let (client, cached) = self.build_fresh().await?;
        *self.cache.lock().await = Some(cached);
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
                let client = self.refresh().await?;
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
    let base_url = base_url
        .filter(|u| !u.is_empty())
        .ok_or_else(|| {
            eyre::eyre!("auth = \"azure-ad\" requires `base_url` (the Azure Foundry endpoint)")
        })?
        .to_string();
    let scope = scope.unwrap_or(DEFAULT_SCOPE).to_string();
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
    // The status/`unauthorized` text lives in the wrapped source (rig surfaces the raw response
    // body as `ProviderError`, and `llm.rs` wraps it with `.wrap_err_with(...)`), so we must walk
    // the whole chain — `err.to_string()` would only render the top-level context. `{err:#}` is the
    // alternate Display that joins the full chain.
    let msg = format!("{err:#}").to_ascii_lowercase();
    msg.contains(" 401") || msg.contains("401 ") || msg.contains("unauthorized")
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
}
