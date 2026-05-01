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

        if self.default_debate() && self.reviewer.len() < 2 {
            eyre::bail!(
                "debate mode requires at least 2 reviewers, found {} — add another reviewer or set debate = false in [defaults]",
                self.reviewer.len()
            );
        }

        if let Some(env) = required_env_var_aggregator(&self.aggregator) {
            check_env_var(env)
                .map_err(|_| eyre::eyre!("[aggregator]: env var {env} is not set"))?;
        }

        for reviewer in &self.reviewer {
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
        reviewer.compact_threshold.or(self.default_compact_threshold())
    }
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
    if matches!(reviewer.provider, ProviderType::Gemini)
        && reviewer.auth.as_deref() == Some("oauth")
    {
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
    if matches!(agg.provider, ProviderType::Gemini) && agg.auth.as_deref() == Some("oauth") {
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

fn default_env_var(provider: &ProviderType) -> Option<&'static str> {
    match provider {
        ProviderType::Anthropic => Some("ANTHROPIC_API_KEY"),
        ProviderType::Gemini => Some("GEMINI_API_KEY"),
        ProviderType::OpenAi => Some("OPENAI_API_KEY"),
        ProviderType::OpenRouter => Some("OPENROUTER_API_KEY"),
    }
}
