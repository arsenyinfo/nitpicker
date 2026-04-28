use crate::config::{AggregatorConfig, Config, ProviderType, ReviewerConfig};
use chrono::{Duration, Utc};
use eyre::Result;
use regex::Regex;
use serde::Deserialize;

const MODELS_URL: &str = "https://openrouter.ai/api/v1/models\
    ?supported_parameters=tools%2Ctemperature\
    &categories=programming";
const DEFAULT_API_KEY_ENV: &str = "OPENROUTER_API_KEY";

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

// 200k+ context is a reasonable proxy for a capable free model
const MIN_CONTEXT_LENGTH: u64 = 200_000;

#[derive(Deserialize)]
struct ModelInfo {
    id: String,
    created: Option<u64>,
    context_length: Option<u64>,
    expiration_date: Option<String>,
    pricing: ModelPricing,
}

#[derive(Deserialize)]
struct ModelPricing {
    prompt: String,
    completion: String,
}

impl ModelPricing {
    fn is_free(&self) -> bool {
        self.prompt.parse::<f64>().unwrap_or(1.0) == 0.0
            && self.completion.parse::<f64>().unwrap_or(1.0) == 0.0
    }
}

fn api_key_env(reviewers: &[ReviewerConfig], aggregator: &AggregatorConfig) -> String {
    reviewers
        .iter()
        .find(|r| needs_auto_model(&r.provider, &r.model))
        .and_then(|r| r.api_key_env.clone())
        .or_else(|| {
            needs_auto_model(&aggregator.provider, &aggregator.model)
                .then(|| aggregator.api_key_env.clone())
                .flatten()
        })
        .unwrap_or_else(|| DEFAULT_API_KEY_ENV.to_string())
}

fn parse_total_params_b(model_id: &str) -> u64 {
    // extracts the largest param count from the model id, normalised to billions * 10
    // handles Xb (billions) and Xt (trillions), e.g.:
    //   "120b-a12b" → 1200, "1t" → 10000, "31b" → 310, "qwen3-coder" → 0
    let re = Regex::new(r"(\d+(?:\.\d+)?)(b|t)").unwrap();
    re.captures_iter(&model_id.to_lowercase())
        .filter_map(|c| {
            let n = c[1].parse::<f64>().ok()?;
            let multiplier = if &c[2] == "t" { 1000.0 } else { 1.0 };
            Some((n * multiplier * 10.0) as u64)
        })
        .max()
        .unwrap_or(0)
}

fn expires_within_24h(expiration_date: Option<&str>) -> bool {
    let Some(date) = expiration_date else {
        return false;
    };
    let Ok(expires) = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d") else {
        return false;
    };
    let expires_dt = expires.and_hms_opt(0, 0, 0).unwrap().and_utc();
    expires_dt <= Utc::now() + Duration::hours(24)
}

async fn fetch_models(api_key: &str, needed: usize) -> Result<Vec<String>> {
    let client = reqwest::Client::new();
    let response = client
        .get(MODELS_URL)
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await
        .map_err(|e| eyre::eyre!("failed to fetch OpenRouter free models: {e}"))?;

    if !response.status().is_success() {
        eyre::bail!("OpenRouter models API returned {}", response.status());
    }

    let body: ModelsResponse = response
        .json()
        .await
        .map_err(|e| eyre::eyre!("failed to parse OpenRouter models response: {e}"))?;

    let mut models: Vec<ModelInfo> = body
        .data
        .into_iter()
        .filter(|m| m.pricing.is_free())
        .filter(|m| m.context_length.unwrap_or(0) >= MIN_CONTEXT_LENGTH)
        .filter(|m| !expires_within_24h(m.expiration_date.as_deref()))
        .collect();
    models.sort_by_key(|m| {
        (
            std::cmp::Reverse(parse_total_params_b(&m.id)),
            std::cmp::Reverse(m.context_length.unwrap_or(0)),
            std::cmp::Reverse(m.created.unwrap_or(0)),
        )
    });
    let models: Vec<String> = models.into_iter().map(|m| m.id).collect();

    tracing::info!(candidates = %models.join(", "), "openrouter_free candidate models");

    if models.len() < needed {
        eyre::bail!(
            "OpenRouter returned only {} free models, needed {}",
            models.len(),
            needed
        );
    }

    Ok(models)
}

fn needs_auto_model(provider: &ProviderType, model: &str) -> bool {
    matches!(provider, ProviderType::OpenRouter) && (model.is_empty() || model == "free")
}

pub async fn resolve_free_models(config: &mut Config) -> Result<()> {
    for reviewer in &config.reviewer {
        if !needs_auto_model(&reviewer.provider, &reviewer.model) {
            if reviewer.name.is_empty() {
                eyre::bail!("reviewer must specify a name (omit model or set model = \"free\" on openrouter to auto-assign)");
            }
            if reviewer.model.is_empty() {
                eyre::bail!("reviewer '{}' must specify a model", reviewer.name);
            }
        }
    }
    if !needs_auto_model(&config.aggregator.provider, &config.aggregator.model)
        && config.aggregator.model.is_empty()
    {
        eyre::bail!("[aggregator] must specify a model");
    }

    let reviewer_indices: Vec<usize> = config
        .reviewer
        .iter()
        .enumerate()
        .filter(|(_, r)| needs_auto_model(&r.provider, &r.model))
        .map(|(i, _)| i)
        .collect();
    let agg_is_free = needs_auto_model(&config.aggregator.provider, &config.aggregator.model);

    let needed = reviewer_indices.len() + usize::from(agg_is_free);
    if needed == 0 {
        return Ok(());
    }

    let key_env = api_key_env(&config.reviewer, &config.aggregator);
    let api_key = std::env::var(&key_env)
        .map_err(|_| eyre::eyre!("missing env var {key_env} (required for openrouter auto-model)"))?;

    let models = fetch_models(&api_key, needed).await?;
    let mut iter = models.into_iter();

    for (slot, idx) in reviewer_indices.into_iter().enumerate() {
        let model = iter.next().expect("checked above");
        if config.reviewer[idx].name.is_empty() {
            config.reviewer[idx].name = format!("free_{}", (b'A' + slot as u8) as char);
        }
        tracing::info!(reviewer = %config.reviewer[idx].name, %model, "resolved openrouter auto model");
        config.reviewer[idx].model = model;
    }

    if agg_is_free {
        let model = iter.next().expect("checked above");
        tracing::info!(%model, "resolved openrouter auto aggregator model");
        config.aggregator.model = model;
    }

    Ok(())
}
