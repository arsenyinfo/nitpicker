use eyre::Result;
use serde::Deserialize;

pub const DEFAULT_MAX_TURNS: usize = 70;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub defaults: Option<DefaultsConfig>,
    pub aggregator: AggregatorConfig,
    pub reviewer: Vec<ReviewerConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefaultsConfig {
    pub debate: Option<bool>,
    pub max_turns: Option<usize>,
    pub compact_threshold: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregatorConfig {
    #[serde(default)]
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
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub model: String,
    pub provider: ProviderType,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub compact_threshold: Option<u64>,
    /// Authentication method: "api_key" (default) or "oauth"
    pub auth: Option<String>,
}

#[derive(Deserialize)]
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
    pub fn default_debate(&self) -> bool {
        self.defaults
            .as_ref()
            .and_then(|defaults| defaults.debate)
            .unwrap_or(true)
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
            .and_then(|defaults| defaults.max_turns)
            .unwrap_or(DEFAULT_MAX_TURNS);

        if max_turns == 0 {
            eyre::bail!("[defaults].max_turns must be greater than 0");
        }

        Ok(max_turns)
    }

    pub fn default_compact_threshold(&self) -> Result<Option<u64>> {
        let threshold = self
            .defaults
            .as_ref()
            .and_then(|defaults| defaults.compact_threshold);

        if threshold == Some(0) {
            eyre::bail!("[defaults].compact_threshold must be greater than 0")
        }

        Ok(threshold)
    }

    pub fn reviewer_compact_threshold(&self, reviewer: &ReviewerConfig) -> Result<Option<u64>> {
        if reviewer.compact_threshold == Some(0) {
            eyre::bail!(
                "reviewer {} compact_threshold must be greater than 0",
                reviewer.name
            );
        }

        Ok(reviewer
            .compact_threshold
            .or(self.default_compact_threshold()?))
    }
}
