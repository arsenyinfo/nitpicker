use crate::config::{AggregatorConfig, Config, ProviderType, ReviewerConfig};
use crate::gemini_proxy::GeminiProxyClient;
use crate::llm::{LLMClient, LLMClientDyn, LLMProvider, WithRetryExt};
use eyre::Result;
use std::sync::Arc;

pub fn needs_gemini_proxy(provider: &ProviderType, auth: Option<&str>) -> bool {
    provider.is_gemini() && matches!(auth, Some("agy-keyring"))
}

pub fn reviewer_needs_gemini_proxy(reviewer: &ReviewerConfig) -> bool {
    needs_gemini_proxy(&reviewer.provider, reviewer.auth.as_deref())
}

pub fn aggregator_needs_gemini_proxy(agg: &AggregatorConfig) -> bool {
    needs_gemini_proxy(&agg.provider, agg.auth.as_deref())
}

pub fn config_needs_gemini_proxy(config: &Config) -> bool {
    aggregator_needs_gemini_proxy(&config.aggregator)
        || config.reviewer.iter().any(reviewer_needs_gemini_proxy)
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
        ProviderType::Gemini => Ok(LLMProvider::Gemini),
        ProviderType::OpenAi => Ok(LLMProvider::OpenAi {
            base_url: base_url.map(str::to_string),
            api_key_env: api_key_env.map(str::to_string),
        }),
        ProviderType::OpenRouter => Ok(LLMProvider::OpenRouter {
            api_key_env: api_key_env.unwrap_or("OPENROUTER_API_KEY").to_string(),
        }),
    }
}

pub fn build_reviewer_client(
    reviewer: &ReviewerConfig,
    gemini_proxy: Option<&GeminiProxyClient>,
) -> Result<Arc<dyn LLMClientDyn>> {
    if reviewer_needs_gemini_proxy(reviewer) {
        let proxy_url = gemini_proxy
            .map(|p| p.base_url())
            .ok_or_else(|| eyre::eyre!("Gemini proxy required but not available"))?;
        return crate::llm::create_gemini_client_with_proxy(&proxy_url);
    }

    Ok(provider_from_config(
        &reviewer.provider,
        reviewer.base_url.as_deref(),
        reviewer.api_key_env.as_deref(),
    )?
    .client_from_env()?
    .with_retry()
    .into_arc())
}

pub fn build_aggregator_client(
    agg: &AggregatorConfig,
    gemini_proxy: Option<&GeminiProxyClient>,
) -> Result<Arc<dyn LLMClientDyn>> {
    if aggregator_needs_gemini_proxy(agg) {
        let proxy_url = gemini_proxy
            .map(|p| p.base_url())
            .ok_or_else(|| eyre::eyre!("Gemini proxy required but not available"))?;
        return crate::llm::create_gemini_client_with_proxy(&proxy_url);
    }

    Ok(provider_from_config(
        &agg.provider,
        agg.base_url.as_deref(),
        agg.api_key_env.as_deref(),
    )?
    .client_from_env()?
    .with_retry()
    .into_arc())
}
