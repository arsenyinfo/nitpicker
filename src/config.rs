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
    #[serde(default)]
    pub fallbacks: Vec<FallbackConfig>,
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
    #[serde(default)]
    pub fallbacks: Vec<FallbackConfig>,
}

/// A backup model configuration used when the primary (or a previous fallback) fails.
/// Shares the same shape as the primary reviewer/aggregator block, minus the `name` field.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct FallbackConfig {
    pub model: String,
    pub provider: ProviderType,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub auth: Option<String>,
    pub max_tokens: Option<u64>,
}

impl FallbackConfig {
    pub fn use_oauth(&self) -> bool {
        self.auth.as_ref().map(|a| a == "oauth").unwrap_or(false)
    }
}

#[derive(Deserialize, Clone)]
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

    /// True if the primary or any fallback needs the Gemini OAuth proxy.
    pub fn needs_gemini_proxy(&self) -> bool {
        (self.provider.is_gemini() && self.use_oauth())
            || self
                .fallbacks
                .iter()
                .any(|f| f.provider.is_gemini() && f.use_oauth())
    }
}

impl ReviewerConfig {
    /// Returns true if OAuth should be used for authentication
    pub fn use_oauth(&self) -> bool {
        self.auth.as_ref().map(|a| a == "oauth").unwrap_or(false)
    }

    /// True if the primary or any fallback needs the Gemini OAuth proxy.
    pub fn needs_gemini_proxy(&self) -> bool {
        (self.provider.is_gemini() && self.use_oauth())
            || self
                .fallbacks
                .iter()
                .any(|f| f.provider.is_gemini() && f.use_oauth())
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_config_without_fallbacks() {
        let toml_str = r#"
[aggregator]
model = "claude-sonnet-4-6"
provider = "anthropic"

[[reviewer]]
name = "claude"
model = "claude-sonnet-4-6"
provider = "anthropic"
"#;
        let cfg: Config = toml::from_str(toml_str).expect("should parse");
        assert_eq!(cfg.reviewer.len(), 1);
        assert!(cfg.reviewer[0].fallbacks.is_empty());
        assert!(cfg.aggregator.fallbacks.is_empty());
    }

    #[test]
    fn parses_reviewer_with_nested_fallbacks() {
        let toml_str = r#"
[aggregator]
model = "claude-sonnet-4-6"
provider = "anthropic"

[[reviewer]]
name = "gemini"
model = "gemini-3.1-pro-preview"
provider = "gemini"
auth = "oauth"

[[reviewer.fallbacks]]
model = "claude-opus-4-7"
provider = "anthropic"

[[reviewer.fallbacks]]
model = "gpt-5"
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
"#;
        let cfg: Config = toml::from_str(toml_str).expect("should parse");
        assert_eq!(cfg.reviewer.len(), 1);
        assert_eq!(cfg.reviewer[0].fallbacks.len(), 2);
        assert_eq!(cfg.reviewer[0].fallbacks[0].model, "claude-opus-4-7");
        assert!(matches!(
            cfg.reviewer[0].fallbacks[0].provider,
            ProviderType::Anthropic
        ));
        assert_eq!(cfg.reviewer[0].fallbacks[1].model, "gpt-5");
        assert_eq!(
            cfg.reviewer[0].fallbacks[1].api_key_env.as_deref(),
            Some("OPENAI_API_KEY")
        );
    }

    #[test]
    fn parses_aggregator_with_fallbacks() {
        let toml_str = r#"
[aggregator]
model = "gemini-3-flash-preview"
provider = "gemini"
auth = "oauth"

[[aggregator.fallbacks]]
model = "claude-opus-4-7"
provider = "anthropic"

[[reviewer]]
name = "r"
model = "m"
provider = "anthropic"
"#;
        let cfg: Config = toml::from_str(toml_str).expect("should parse");
        assert_eq!(cfg.aggregator.fallbacks.len(), 1);
        assert_eq!(cfg.aggregator.fallbacks[0].model, "claude-opus-4-7");
    }

    #[test]
    fn rejects_unknown_field_on_reviewer() {
        let toml_str = r#"
[aggregator]
model = "m"
provider = "anthropic"

[[reviewer]]
name = "r"
model = "m"
provider = "anthropic"
fallback = "typo"
"#;
        let result = toml::from_str::<Config>(toml_str);
        let err = match result {
            Ok(_) => panic!("typo should be rejected"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("fallback") || msg.contains("unknown field"),
            "error should mention unknown field: {msg}"
        );
    }

    #[test]
    fn needs_gemini_proxy_detects_oauth_on_primary() {
        let toml_str = r#"
[aggregator]
model = "m"
provider = "anthropic"

[[reviewer]]
name = "g"
model = "m"
provider = "gemini"
auth = "oauth"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.reviewer[0].needs_gemini_proxy());
        assert!(!cfg.aggregator.needs_gemini_proxy());
    }

    #[test]
    fn needs_gemini_proxy_detects_oauth_on_fallback() {
        let toml_str = r#"
[aggregator]
model = "m"
provider = "anthropic"

[[reviewer]]
name = "r"
model = "m"
provider = "anthropic"

[[reviewer.fallbacks]]
model = "g"
provider = "gemini"
auth = "oauth"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.reviewer[0].needs_gemini_proxy());
    }

    #[test]
    fn needs_gemini_proxy_false_when_no_oauth() {
        let toml_str = r#"
[aggregator]
model = "m"
provider = "anthropic"

[[reviewer]]
name = "r"
model = "m"
provider = "anthropic"
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(!cfg.reviewer[0].needs_gemini_proxy());
        assert!(!cfg.aggregator.needs_gemini_proxy());
    }
}
