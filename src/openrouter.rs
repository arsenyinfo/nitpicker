use crate::config::{Config, ProviderType};
use crate::llm::{Completion, LLMClient};
use chrono::{Duration, Utc};
use eyre::Result;
use regex::Regex;
use rig::completion::{Message, ToolDefinition};
use rig::providers::openrouter;
use serde::Deserialize;
use serde_json::json;
use std::time::Instant;

const MODELS_URL: &str = "https://openrouter.ai/api/v1/models\
    ?supported_parameters=tools%2Ctemperature\
    &categories=programming";
const DEFAULT_API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const SMOKE_TEST_TIMEOUT_SECS: u64 = 15;
const SMOKE_TEST_CALLS_REQUIRED: usize = 2;
const FETCH_MODELS_MAX_ATTEMPTS: usize = 3;
const FETCH_MODELS_BACKOFF_MS: u64 = 1_000;

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
    let mut last_err: eyre::Report = eyre::eyre!("no attempts made");

    for attempt in 1..=FETCH_MODELS_MAX_ATTEMPTS {
        match try_fetch_models(&client, api_key, needed).await {
            Ok(models) => return Ok(models),
            Err(err) => {
                tracing::warn!(attempt, error = %err, "failed to fetch OpenRouter models, retrying");
                last_err = err;
                if attempt < FETCH_MODELS_MAX_ATTEMPTS {
                    tokio::time::sleep(std::time::Duration::from_millis(
                        FETCH_MODELS_BACKOFF_MS * attempt as u64,
                    ))
                    .await;
                }
            }
        }
    }

    Err(last_err)
}

async fn try_fetch_models(
    client: &reqwest::Client,
    api_key: &str,
    needed: usize,
) -> Result<Vec<String>> {
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

    tracing::info!(candidates = %models.join(", "), "experimental openrouter free candidate models");

    if models.len() < needed {
        eyre::bail!(
            "OpenRouter experimental free resolver found only {} candidate models, needed {}",
            models.len(),
            needed
        );
    }

    Ok(models)
}

fn smoke_test_tool() -> ToolDefinition {
    ToolDefinition {
        name: "smoke_test_tool".to_string(),
        description: "Sanity-check tool used to verify model tool calling.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "ok": {
                    "type": "boolean",
                    "description": "Set to true"
                }
            },
            "required": ["ok"],
            "additionalProperties": false
        }),
    }
}

async fn smoke_test_call(api_key: &str, model: &str, slot_label: &str, attempt: usize) -> bool {
    let client = match openrouter::Client::new(api_key) {
        Ok(client) => client,
        Err(err) => {
            tracing::warn!(slot = slot_label, %model, attempt, error = %err, "openrouter smoke test client init failed");
            return false;
        }
    };
    let started = Instant::now();
    tracing::info!(slot = slot_label, %model, attempt, "experimental openrouter free smoke test start");

    let completion = Completion {
        model: model.to_string(),
        prompt: Message::user(
            "Call the tool named smoke_test_tool with {\"ok\":true}. Do not answer with plain text.".to_string(),
        ),
        preamble: Some(
            "You are a tool-use smoke test. You must respond by calling the provided tool exactly once.".to_string(),
        ),
        history: Vec::new(),
        tools: vec![smoke_test_tool()],
        tool_choice: None,
        max_tokens: Some(32),
        additional_params: None,
    };

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(SMOKE_TEST_TIMEOUT_SECS),
        client.completion(completion),
    )
    .await;
    let latency_ms = started.elapsed().as_millis() as u64;

    match result {
        Ok(Ok(response)) if response.tool_calls().is_some() => {
            tracing::info!(slot = slot_label, %model, attempt, latency_ms, "experimental openrouter free smoke test passed");
            true
        }
        Ok(Ok(response)) => {
            tracing::warn!(
                slot = slot_label,
                %model,
                attempt,
                latency_ms,
                finish_reason = ?response.finish_reason,
                text = response.text(),
                "experimental openrouter free smoke test failed: no tool call"
            );
            false
        }
        Ok(Err(err)) => {
            tracing::warn!(slot = slot_label, %model, attempt, latency_ms, error = %err, "experimental openrouter free smoke test failed");
            false
        }
        Err(_) => {
            tracing::warn!(slot = slot_label, %model, attempt, latency_ms, timeout_secs = SMOKE_TEST_TIMEOUT_SECS, "experimental openrouter free smoke test timed out");
            false
        }
    }
}

async fn smoke_test_model(api_key: &str, model: &str, slot_label: &str) -> bool {
    for attempt in 1..=SMOKE_TEST_CALLS_REQUIRED {
        if !smoke_test_call(api_key, model, slot_label, attempt).await {
            return false;
        }
    }
    true
}

/// Runs smoke tests for all candidates concurrently under a single API key.
/// Returns a bool per candidate (same index) indicating whether it passed.
async fn probe_candidates_concurrent(
    api_key: &str,
    candidates: &[String],
    batch_label: &str,
) -> Vec<bool> {
    let mut set = tokio::task::JoinSet::new();
    for (idx, model) in candidates.iter().cloned().enumerate() {
        let key = api_key.to_string();
        let label = batch_label.to_string();
        set.spawn(async move {
            let passed = smoke_test_model(&key, &model, &label).await;
            (idx, passed)
        });
    }
    let mut results = vec![false; candidates.len()];
    while let Some(res) = set.join_next().await {
        if let Ok((idx, passed)) = res {
            results[idx] = passed;
        }
    }
    results
}

async fn select_reviewer_models(
    slot_api_keys: &[String],
    candidates: Vec<String>,
    reviewer_count: usize,
) -> Result<(Vec<String>, Vec<String>)> {
    if reviewer_count == 0 {
        return Ok((Vec::new(), candidates));
    }

    // probe all candidates concurrently for each distinct api key
    let unique_keys: std::collections::HashSet<&str> =
        slot_api_keys.iter().map(String::as_str).collect();
    let mut key_results: std::collections::HashMap<&str, Vec<bool>> = Default::default();
    for key in unique_keys {
        key_results.insert(
            key,
            probe_candidates_concurrent(key, &candidates, "reviewer_probe").await,
        );
    }

    // greedy assignment: for each slot pick the first passing candidate (by quality order) not yet taken
    let mut assigned = std::collections::HashSet::new();
    let mut selected = Vec::with_capacity(reviewer_count);

    for (slot, slot_key) in slot_api_keys.iter().enumerate().take(reviewer_count) {
        let slot_label = format!("free_slot_{}", slot + 1);
        let results = key_results.get(slot_key.as_str()).expect("key was probed");

        let winner = results
            .iter()
            .enumerate()
            .find(|&(ref idx, &passed)| passed && !assigned.contains(idx));

        let Some((winner_idx, _)) = winner else {
            eyre::bail!(
                "No usable OpenRouter experimental free model found for {} after concurrent smoke tests of {} candidates",
                slot_label,
                candidates.len()
            );
        };

        assigned.insert(winner_idx);
        selected.push(candidates[winner_idx].clone());
    }

    let remaining = candidates
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !assigned.contains(i))
        .map(|(_, m)| m)
        .collect();

    Ok((selected, remaining))
}

async fn select_aggregator_model(
    api_key: &str,
    candidates: &[String],
    reviewer_models: &[String],
) -> Result<String> {
    let results = probe_candidates_concurrent(api_key, candidates, "aggregator_probe").await;

    if let Some(model) = candidates
        .iter()
        .zip(results.iter())
        .find(|&(_, &passed)| passed)
        .map(|(m, _)| m.clone())
    {
        return Ok(model);
    }

    if let Some(model) = reviewer_models.first() {
        tracing::info!(
            model = %model,
            "reusing first reviewer model for experimental openrouter free aggregator"
        );
        return Ok(model.clone());
    }

    eyre::bail!(
        "No usable OpenRouter experimental free model found for aggregator after concurrent smoke tests of {} candidates",
        candidates.len()
    )
}

fn needs_auto_model(provider: &ProviderType, model: &str) -> bool {
    matches!(provider, ProviderType::OpenRouter) && (model.is_empty() || model == "free")
}

pub async fn resolve_free_models(config: &mut Config) -> Result<()> {
    for reviewer in &config.reviewer {
        if !needs_auto_model(&reviewer.provider, &reviewer.model) {
            if reviewer.name.is_empty() {
                eyre::bail!(
                    "reviewer must specify a name (omit model or set model = \"free\" on openrouter for experimental auto-assignment)"
                );
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

    let reviewer_count = reviewer_indices.len();
    if reviewer_count == 0 && !agg_is_free {
        return Ok(());
    }

    let slot_api_keys: Vec<String> = reviewer_indices
        .iter()
        .map(|&idx| {
            let key_env = config.reviewer[idx]
                .api_key_env
                .as_deref()
                .unwrap_or(DEFAULT_API_KEY_ENV);
            std::env::var(key_env).map_err(|_| eyre::eyre!("missing env var {key_env}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let agg_api_key = if agg_is_free {
        let key_env = config
            .aggregator
            .api_key_env
            .as_deref()
            .unwrap_or(DEFAULT_API_KEY_ENV);
        std::env::var(key_env).map_err(|_| {
            eyre::eyre!("missing env var {key_env} (required for experimental openrouter free auto-selection)")
        })?
    } else {
        String::new()
    };

    let fetch_key = slot_api_keys
        .first()
        .map(String::as_str)
        .unwrap_or(agg_api_key.as_str());

    let models = fetch_models(fetch_key, reviewer_count.max(1)).await?;
    let (reviewer_models, remaining_models) =
        select_reviewer_models(&slot_api_keys, models, reviewer_count).await?;
    let aggregator_model = if agg_is_free {
        Some(select_aggregator_model(&agg_api_key, &remaining_models, &reviewer_models).await?)
    } else {
        None
    };
    let mut iter = reviewer_models.into_iter();

    for (slot, idx) in reviewer_indices.into_iter().enumerate() {
        let model = iter.next().expect("checked above");
        if config.reviewer[idx].name.is_empty() {
            config.reviewer[idx].name = format!("free_{}", (b'A' + slot as u8) as char);
        }
        tracing::info!(reviewer = %config.reviewer[idx].name, %model, "resolved experimental openrouter free model");
        config.reviewer[idx].model = model;
    }

    if agg_is_free {
        let model = aggregator_model.expect("checked above");
        tracing::info!(%model, "resolved experimental openrouter free aggregator model");
        config.aggregator.model = model;
    }

    Ok(())
}
