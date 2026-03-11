use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub aggregator: AggregatorConfig,
    pub reviewer: Vec<ReviewerConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregatorConfig {
    pub model: String,
    pub provider: ProviderType,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub max_tokens: Option<u64>,
    /// Authentication method: "api_key" (default) or "oauth"
    pub auth: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewerConfig {
    pub name: String,
    pub model: String,
    pub provider: ProviderType,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    /// Authentication method: "api_key" (default) or "oauth"
    pub auth: Option<String>,
}

#[derive(Deserialize)]
pub enum ProviderType {
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "gemini")]
    Gemini,
    #[serde(rename = "anthropic_compatible")]
    AnthropicCompatible,
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

impl ProviderType {
    pub fn is_gemini(&self) -> bool {
        matches!(self, ProviderType::Gemini)
    }
}

impl AggregatorConfig {
    /// Returns true if OAuth should be used for authentication
    pub fn use_oauth(&self) -> bool {
        self.auth.as_ref().map(|a| a == "oauth").unwrap_or(false)
    }
}

impl ReviewerConfig {
    /// Returns true if OAuth should be used for authentication
    pub fn use_oauth(&self) -> bool {
        self.auth.as_ref().map(|a| a == "oauth").unwrap_or(false)
    }
}
