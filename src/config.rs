use eyre::Result;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_TURNS: usize = 100;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defaults: Option<DefaultsConfig>,
    pub aggregator: AggregatorConfig,
    pub reviewer: Vec<ReviewerConfig>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefaultsConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alloy: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_threshold: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_trajectories: Option<bool>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregatorConfig {
    #[serde(default)]
    pub model: String,
    pub provider: ProviderType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<String>,
    /// AAD scope for `auth = "azure-ad"` (defaults to the Cognitive Services scope).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azure_scope: Option<String>,
    /// Azure credential chain selector for `auth = "azure-ad"`: `"dev"`, `"prod"`, or unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azure_credentials: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewerConfig {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub model: String,
    pub provider: ProviderType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_threshold: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<String>,
    /// AAD scope for `auth = "azure-ad"` (defaults to the Cognitive Services scope).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azure_scope: Option<String>,
    /// Azure credential chain selector for `auth = "azure-ad"`: `"dev"`, `"prod"`, or unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azure_credentials: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub enum ProviderType {
    #[serde(rename = "anthropic", alias = "anthropic_compatible")]
    Anthropic,
    #[serde(rename = "gemini")]
    Gemini,
    #[serde(rename = "openai", alias = "openai_compatible")]
    OpenAi,
    #[serde(rename = "openrouter")]
    OpenRouter,
}

impl ProviderType {
    pub fn is_gemini(&self) -> bool {
        matches!(self, ProviderType::Gemini)
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.reviewer.is_empty() {
            eyre::bail!("no reviewers configured");
        }

        self.validate_alloy(self.default_alloy())?;

        validate_free_model(
            "[aggregator]",
            &self.aggregator.provider,
            &self.aggregator.model,
        )?;

        // Validate `auth` before the env-var check (matching the reviewer loop below): an unknown
        // auth value like the typo `azure_ad` should surface its own clear error rather than being
        // masked by a missing-API-key error when the provider key happens to be unset.
        validate_auth(
            "[aggregator]",
            &self.aggregator.provider,
            &self.aggregator.auth,
            self.aggregator.base_url.as_deref(),
            self.aggregator.azure_credentials.as_deref(),
        )?;

        if let Some(env) = required_env_var_aggregator(&self.aggregator) {
            check_env_var(env)
                .map_err(|_| eyre::eyre!("[aggregator]: env var {env} is not set"))?;
        }

        for reviewer in &self.reviewer {
            let reviewer_label = match reviewer.name.is_empty() {
                true => "reviewer <unnamed>".to_string(),
                false => format!("reviewer {}", reviewer.name),
            };
            validate_free_model(&reviewer_label, &reviewer.provider, &reviewer.model)?;
            validate_auth(
                &reviewer_label,
                &reviewer.provider,
                &reviewer.auth,
                reviewer.base_url.as_deref(),
                reviewer.azure_credentials.as_deref(),
            )?;
            if let Some(env) = required_env_var_reviewer(reviewer) {
                check_env_var(env).map_err(|_| {
                    eyre::eyre!("reviewer {}: env var {env} is not set", reviewer.name)
                })?;
            }
            if reviewer.compact_threshold == Some(0) {
                eyre::bail!(
                    "reviewer {}: compact_threshold must be greater than 0",
                    reviewer.name
                );
            }
        }

        if self.defaults.as_ref().and_then(|d| d.compact_threshold) == Some(0) {
            eyre::bail!("[defaults].compact_threshold must be greater than 0");
        }

        Ok(())
    }

    pub fn default_debate(&self) -> bool {
        self.defaults
            .as_ref()
            .and_then(|d| d.debate)
            .unwrap_or(true)
    }

    pub fn validate_alloy(&self, alloy: bool) -> Result<()> {
        if alloy && self.reviewer.len() < 2 {
            eyre::bail!("--alloy requires at least 2 reviewers, found {}", self.reviewer.len());
        }
        Ok(())
    }

    pub fn default_alloy(&self) -> bool {
        self.defaults
            .as_ref()
            .and_then(|d| d.alloy)
            .unwrap_or(false)
    }

    pub fn max_turns(&self, override_max_turns: Option<usize>) -> Result<usize> {
        match override_max_turns {
            Some(max_turns) => Ok(max_turns),
            None => self.default_max_turns(),
        }
    }

    pub fn default_max_turns(&self) -> Result<usize> {
        let max_turns = self
            .defaults
            .as_ref()
            .and_then(|d| d.max_turns)
            .unwrap_or(DEFAULT_MAX_TURNS);

        if max_turns == 0 {
            eyre::bail!("[defaults].max_turns must be greater than 0");
        }

        Ok(max_turns)
    }

    pub fn default_compact_threshold(&self) -> Option<u64> {
        self.defaults.as_ref().and_then(|d| d.compact_threshold)
    }

    pub fn log_trajectories(&self) -> bool {
        self.defaults
            .as_ref()
            .and_then(|d| d.log_trajectories)
            .unwrap_or(false)
    }

    pub fn reviewer_compact_threshold(&self, reviewer: &ReviewerConfig) -> Option<u64> {
        reviewer
            .compact_threshold
            .or(self.default_compact_threshold())
    }
}

fn validate_free_model(label: &str, provider: &ProviderType, model: &str) -> Result<()> {
    if model == "free" && !matches!(provider, ProviderType::OpenRouter) {
        eyre::bail!("{label}: model = \"free\" is only supported with provider = \"openrouter\"");
    }

    Ok(())
}

fn validate_auth(
    label: &str,
    provider: &ProviderType,
    auth: &Option<String>,
    base_url: Option<&str>,
    azure_credentials: Option<&str>,
) -> Result<()> {
    match (provider, auth.as_deref()) {
        // Unset auth is always fine — providers fall back to their default env-var key.
        (_, None) => Ok(()),
        (ProviderType::Gemini, Some("oauth")) => {
            eyre::bail!(
                "{label}: auth = \"oauth\" has been removed — use auth = \"agy-keyring\" (see README) or unset `auth` to use GEMINI_API_KEY"
            );
        }
        (ProviderType::Gemini, Some("agy-keyring")) => Ok(()),
        (ProviderType::Gemini, Some(other)) => {
            eyre::bail!(
                "{label}: unknown auth value \"{other}\" — expected \"agy-keyring\" or unset"
            );
        }
        // Azure AD is only meaningful for the OpenAI/Anthropic Foundry passthrough endpoints,
        // and only works when the `azure` feature was compiled in.
        (ProviderType::OpenAi | ProviderType::Anthropic, Some("azure-ad")) => {
            if !cfg!(feature = "azure") {
                eyre::bail!(
                    "{label}: auth = \"azure-ad\" requires building nitpicker with `--features azure`"
                );
            }
            validate_azure_fields(label, base_url, azure_credentials)
        }
        (_, Some("azure-ad")) => {
            eyre::bail!(
                "{label}: auth = \"azure-ad\" is only supported with provider \"openai\" or \"anthropic\""
            );
        }
        // Any other auth value on a non-Gemini provider is a typo or unsupported — reject it at
        // config time rather than failing cryptically at client construction.
        (_, Some(other)) => {
            eyre::bail!("{label}: unknown auth value \"{other}\"");
        }
    }
}

/// Validate the mandatory `auth = "azure-ad"` fields at config time so a typo fails fast here
/// instead of at the first LLM call. Mirrors the runtime checks in `azure::build_azure_client`
/// (base_url) and `azure::build_credential_chain` (azure_credentials).
fn validate_azure_fields(
    label: &str,
    base_url: Option<&str>,
    azure_credentials: Option<&str>,
) -> Result<()> {
    if base_url.map(str::trim).filter(|u| !u.is_empty()).is_none() {
        eyre::bail!(
            "{label}: auth = \"azure-ad\" requires a non-empty `base_url` (the Azure Foundry endpoint)"
        );
    }
    if let Some(mode) = azure_credentials {
        let normalized = mode.trim().to_ascii_lowercase();
        if !normalized.is_empty() && !matches!(normalized.as_str(), "dev" | "prod" | "auto") {
            eyre::bail!(
                "{label}: unknown azure_credentials value \"{mode}\" — expected \"dev\", \"prod\", or unset (\"auto\")"
            );
        }
    }
    Ok(())
}

fn check_env_var(name: &str) -> Result<(), std::env::VarError> {
    // gemini accepts either GEMINI_API_KEY or GOOGLE_AI_API_KEY
    if name == "GEMINI_API_KEY" {
        if std::env::var("GEMINI_API_KEY").is_ok() || std::env::var("GOOGLE_AI_API_KEY").is_ok() {
            return Ok(());
        }
        return Err(std::env::VarError::NotPresent);
    }
    std::env::var(name).map(|_| ())
}

fn is_local_server(base_url: Option<&str>) -> bool {
    base_url
        .map(|u| u.starts_with("http://localhost") || u.starts_with("http://127.0.0.1"))
        .unwrap_or(false)
}

fn required_env_var_reviewer(reviewer: &ReviewerConfig) -> Option<&str> {
    if matches!(reviewer.provider, ProviderType::Gemini) && is_gemini_proxy_auth(&reviewer.auth) {
        return None;
    }
    if is_azure_ad_auth(reviewer.auth.as_deref()) {
        return None;
    }
    if is_local_server(reviewer.base_url.as_deref()) {
        return None;
    }
    if let Some(env) = &reviewer.api_key_env {
        return Some(env.as_str());
    }
    default_env_var(&reviewer.provider)
}

fn required_env_var_aggregator(agg: &AggregatorConfig) -> Option<&str> {
    if matches!(agg.provider, ProviderType::Gemini) && is_gemini_proxy_auth(&agg.auth) {
        return None;
    }
    if is_azure_ad_auth(agg.auth.as_deref()) {
        return None;
    }
    if is_local_server(agg.base_url.as_deref()) {
        return None;
    }
    if let Some(env) = &agg.api_key_env {
        return Some(env.as_str());
    }
    default_env_var(&agg.provider)
}

fn is_gemini_proxy_auth(auth: &Option<String>) -> bool {
    matches!(auth.as_deref(), Some("agy-keyring"))
}

/// Canonical check shared with `provider.rs` (the client-build path), kept here next to the
/// config types so validation and construction can't drift apart.
pub fn is_azure_ad_auth(auth: Option<&str>) -> bool {
    matches!(auth, Some("azure-ad"))
}

fn default_env_var(provider: &ProviderType) -> Option<&'static str> {
    match provider {
        ProviderType::Anthropic => Some("ANTHROPIC_API_KEY"),
        ProviderType::Gemini => Some("GEMINI_API_KEY"),
        ProviderType::OpenAi => Some("OPENAI_API_KEY"),
        ProviderType::OpenRouter => Some("OPENROUTER_API_KEY"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOUNDRY_URL: &str = "https://res.services.ai.azure.com/openai/v1";

    #[test]
    fn azure_ad_auth_detection() {
        assert!(is_azure_ad_auth(Some("azure-ad")));
        assert!(!is_azure_ad_auth(Some("agy-keyring")));
        assert!(!is_azure_ad_auth(None));
    }

    #[test]
    fn validate_auth_rejects_azure_ad_on_unsupported_providers() {
        let auth = Some("azure-ad".to_string());
        assert!(validate_auth("[t]", &ProviderType::Gemini, &auth, None, None).is_err());
        assert!(validate_auth("[t]", &ProviderType::OpenRouter, &auth, None, None).is_err());
    }

    #[test]
    fn validate_auth_azure_ad_on_supported_providers() {
        let auth = Some("azure-ad".to_string());
        let openai = validate_auth("[t]", &ProviderType::OpenAi, &auth, Some(FOUNDRY_URL), None);
        let anthropic =
            validate_auth("[t]", &ProviderType::Anthropic, &auth, Some(FOUNDRY_URL), None);
        // Accepted only when compiled with the `azure` feature; otherwise validation fails fast
        // with a build hint.
        if cfg!(feature = "azure") {
            assert!(openai.is_ok());
            assert!(anthropic.is_ok());
        } else {
            assert!(openai.is_err());
            assert!(anthropic.is_err());
        }
    }

    #[test]
    fn validate_auth_allows_unset_and_known_values() {
        assert!(validate_auth("[t]", &ProviderType::OpenAi, &None, None, None).is_ok());
        assert!(
            validate_auth(
                "[t]",
                &ProviderType::Gemini,
                &Some("agy-keyring".to_string()),
                None,
                None
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_auth_rejects_unknown_value_on_non_gemini() {
        // Typos like `azure_ad`/`Azure-AD` must fail at config time, not at client construction.
        let typo = Some("azure_ad".to_string());
        assert!(validate_auth("[t]", &ProviderType::OpenAi, &typo, Some(FOUNDRY_URL), None).is_err());
        assert!(validate_auth("[t]", &ProviderType::Anthropic, &typo, None, None).is_err());
    }

    #[cfg(feature = "azure")]
    #[test]
    fn validate_auth_azure_ad_requires_base_url() {
        let auth = Some("azure-ad".to_string());
        assert!(validate_auth("[t]", &ProviderType::OpenAi, &auth, None, None).is_err());
        assert!(validate_auth("[t]", &ProviderType::OpenAi, &auth, Some(""), None).is_err());
        assert!(validate_auth("[t]", &ProviderType::OpenAi, &auth, Some(FOUNDRY_URL), None).is_ok());
    }

    #[cfg(feature = "azure")]
    #[test]
    fn validate_auth_azure_ad_validates_credentials() {
        let auth = Some("azure-ad".to_string());
        let ok = |creds| {
            validate_auth("[t]", &ProviderType::OpenAi, &auth, Some(FOUNDRY_URL), creds).is_ok()
        };
        assert!(ok(None));
        assert!(ok(Some("dev")));
        assert!(ok(Some("PROD"))); // case/whitespace normalized like the runtime chain builder
        assert!(ok(Some("auto")));
        assert!(
            validate_auth(
                "[t]",
                &ProviderType::OpenAi,
                &auth,
                Some(FOUNDRY_URL),
                Some("deve")
            )
            .is_err()
        );
    }
}
