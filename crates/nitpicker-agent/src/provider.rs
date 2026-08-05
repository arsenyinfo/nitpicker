#[cfg(feature = "antigravity")]
use crate::config::Config;
use crate::config::{
    AggregatorConfig, ClientSettings, ProviderType, ReviewerConfig, is_azure_ad_auth, is_codex_auth,
};
use crate::llm::{LLMClient, LLMClientDyn, LLMProvider, WithRetryExt};
use eyre::Result;
use std::sync::Arc;

// The Antigravity Gemini proxy path is gated behind the `antigravity` feature; these
// predicates (and the proxy branches below) compile out entirely when it is off.
#[cfg(feature = "antigravity")]
pub fn needs_gemini_proxy(provider: &ProviderType, auth: Option<&str>) -> bool {
    provider.is_gemini() && matches!(auth, Some("agy-keyring"))
}

#[cfg(feature = "antigravity")]
pub fn reviewer_needs_gemini_proxy(reviewer: &ReviewerConfig) -> bool {
    needs_gemini_proxy(&reviewer.provider, reviewer.auth.as_deref())
}

#[cfg(feature = "antigravity")]
pub fn aggregator_needs_gemini_proxy(agg: &AggregatorConfig) -> bool {
    needs_gemini_proxy(&agg.provider, agg.auth.as_deref())
}

#[cfg(feature = "antigravity")]
pub fn config_needs_gemini_proxy(config: &Config) -> bool {
    aggregator_needs_gemini_proxy(&config.aggregator)
        || config.reviewer.iter().any(reviewer_needs_gemini_proxy)
}

/// Build a refreshing Azure AD client (feature `azure`). The config validator already rejects
/// `auth = "azure-ad"` when the feature is absent, so the disabled arm is defensive.
fn build_azure_ad_client(
    provider: &ProviderType,
    base_url: Option<&str>,
    scope: Option<&str>,
    credentials: Option<&str>,
) -> Result<Arc<dyn LLMClientDyn>> {
    #[cfg(feature = "azure")]
    {
        Ok(
            crate::azure::build_azure_client(provider, base_url, scope, credentials)?
                .with_retry()
                .into_arc(),
        )
    }
    #[cfg(not(feature = "azure"))]
    {
        let _ = (provider, base_url, scope, credentials);
        eyre::bail!("auth = \"azure-ad\" requires building nitpicker with `--features azure`")
    }
}

pub fn provider_from_config(
    provider: &ProviderType,
    base_url: Option<&str>,
    api_key_env: Option<&str>,
) -> Result<LLMProvider> {
    match provider {
        ProviderType::Anthropic => Ok(LLMProvider::Anthropic {
            base_url: base_url.map(str::to_string),
            api_key_env: api_key_env.map(str::to_string),
        }),
        ProviderType::Gemini => Ok(LLMProvider::Gemini {
            base_url: base_url.map(str::to_string),
            api_key_env: api_key_env.map(str::to_string),
        }),
        ProviderType::OpenAi => Ok(LLMProvider::OpenAi {
            base_url: base_url.map(str::to_string),
            api_key_env: api_key_env.map(str::to_string),
        }),
        ProviderType::OpenRouter => Ok(LLMProvider::OpenRouter {
            api_key_env: api_key_env.unwrap_or("OPENROUTER_API_KEY").to_string(),
        }),
    }
}

fn build_client_from_settings(
    settings: ClientSettings<'_>,
    proxy_url: Option<&str>,
) -> Result<Arc<dyn LLMClientDyn>> {
    #[cfg(feature = "antigravity")]
    if needs_gemini_proxy(settings.provider, settings.auth) {
        let url =
            proxy_url.ok_or_else(|| eyre::eyre!("Gemini proxy required but not available"))?;
        return crate::llm::create_gemini_client_with_proxy(url);
    }
    #[cfg(not(feature = "antigravity"))]
    let _ = proxy_url;

    if is_azure_ad_auth(settings.auth) {
        return build_azure_ad_client(
            settings.provider,
            settings.base_url,
            settings.azure_scope,
            settings.azure_credentials,
        );
    }

    if is_codex_auth(settings.auth) {
        return crate::codex::shared_client();
    }

    Ok(
        provider_from_config(settings.provider, settings.base_url, settings.api_key_env)?
            .client_from_env()?
            .with_retry()
            .into_arc(),
    )
}

pub fn build_reviewer_client(
    reviewer: &ReviewerConfig,
    proxy_url: Option<&str>,
) -> Result<Arc<dyn LLMClientDyn>> {
    build_client_from_settings(ClientSettings::from(reviewer), proxy_url)
}

pub fn build_aggregator_client(
    agg: &AggregatorConfig,
    proxy_url: Option<&str>,
) -> Result<Arc<dyn LLMClientDyn>> {
    build_client_from_settings(ClientSettings::from(agg), proxy_url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_carries_base_url_and_api_key_env() {
        let provider = provider_from_config(
            &ProviderType::Gemini,
            Some("http://localhost:8080"),
            Some("MY_GEMINI_KEY"),
        )
        .unwrap();
        match provider {
            LLMProvider::Gemini {
                base_url,
                api_key_env,
            } => {
                assert_eq!(base_url.as_deref(), Some("http://localhost:8080"));
                assert_eq!(api_key_env.as_deref(), Some("MY_GEMINI_KEY"));
            }
            _ => panic!("expected Gemini variant"),
        }
    }

    /// The OpenRouter arm is the only one with logic rather than field copies: a hardcoded
    /// env-var default, and `base_url` deliberately discarded.
    #[test]
    fn openrouter_defaults_its_key_env_and_ignores_base_url() {
        match provider_from_config(
            &ProviderType::OpenRouter,
            Some("https://ignored.example"),
            None,
        )
        .unwrap()
        {
            LLMProvider::OpenRouter { api_key_env } => {
                assert_eq!(api_key_env, "OPENROUTER_API_KEY");
            }
            _ => panic!("expected OpenRouter variant"),
        }
        match provider_from_config(&ProviderType::OpenRouter, None, Some("CUSTOM_KEY")).unwrap() {
            LLMProvider::OpenRouter { api_key_env } => assert_eq!(api_key_env, "CUSTOM_KEY"),
            _ => panic!("expected OpenRouter variant"),
        }
    }

    // used by the antigravity proxy test and the not(azure) hint test; the azure-only build
    // compiles both out
    #[cfg(any(feature = "antigravity", not(feature = "azure")))]
    fn gemini_proxy_reviewer() -> ReviewerConfig {
        ReviewerConfig {
            name: String::new(),
            model: String::new(),
            provider: ProviderType::Gemini,
            base_url: None,
            api_key_env: None,
            max_tokens: None,
            compact_threshold: None,
            auth: Some("agy-keyring".to_string()),
            azure_scope: None,
            azure_credentials: None,
        }
    }

    #[cfg(any(feature = "antigravity", not(feature = "azure")))]
    fn error_of(result: Result<Arc<dyn LLMClientDyn>>) -> String {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(err) => err.to_string(),
        }
    }

    /// The message is a contract: both role builders must fail identically when the proxy is
    /// configured but was not started.
    #[cfg(feature = "antigravity")]
    #[test]
    fn missing_proxy_is_a_hard_error_for_agy_keyring_on_both_roles() {
        let reviewer = gemini_proxy_reviewer();
        let err = error_of(build_reviewer_client(&reviewer, None));
        assert_eq!(err, "Gemini proxy required but not available");

        let agg = AggregatorConfig {
            model: String::new(),
            provider: ProviderType::Gemini,
            base_url: None,
            api_key_env: None,
            max_tokens: None,
            auth: Some("agy-keyring".to_string()),
            azure_scope: None,
            azure_credentials: None,
        };
        let err = error_of(build_aggregator_client(&agg, None));
        assert_eq!(err, "Gemini proxy required but not available");
    }

    /// Feature-off, `auth = "azure-ad"` reaching the builder (validation bypassed) must name
    /// the build flag rather than fail obscurely downstream.
    #[cfg(not(feature = "azure"))]
    #[test]
    fn azure_ad_without_the_feature_names_the_build_flag() {
        let mut reviewer = gemini_proxy_reviewer();
        reviewer.provider = ProviderType::OpenAi;
        reviewer.auth = Some("azure-ad".to_string());
        let err = error_of(build_reviewer_client(&reviewer, None));
        assert_eq!(
            err,
            "auth = \"azure-ad\" requires building nitpicker with `--features azure`"
        );
    }
}
