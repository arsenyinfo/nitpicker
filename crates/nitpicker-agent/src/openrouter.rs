use crate::config::{Config, ProviderType};
use crate::llm::{Completion, LLMClient, openrouter_headers};
use chrono::{Duration, Utc};
use eyre::Result;
use rig_core::completion::{Message, ToolDefinition};
use rig_core::providers::openrouter;
use serde::Deserialize;
use serde_json::json;
use std::time::Instant;

const MODELS_URL: &str =
    "https://openrouter.ai/api/v1/models?supported_parameters=tools%2Ctemperature";
const RANKINGS_URL: &str =
    "https://openrouter.ai/api/frontend/models?order=top-weekly&category=programming";
// ranking API is not part of the official openrouter API and may be unreliable
const DEFAULT_API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const SMOKE_TEST_TIMEOUT_SECS: u64 = 15;
const SMOKE_TEST_CALLS_REQUIRED: usize = 2;
const SMOKE_TEST_BATCH_SIZE: usize = 3;
const FETCH_MODELS_MAX_ATTEMPTS: usize = 3;
const FETCH_MODELS_BACKOFF_MS: u64 = 1_000;

static PARAMS_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"(\d+(?:\.\d+)?)(b|t)").unwrap());

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

const MIN_CONTEXT_LENGTH: u64 = 128_000;
const COMPACT_HEADROOM: u64 = 25_000;

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

#[derive(Deserialize)]
struct RankingsResponse {
    data: Vec<RankedModel>,
}

#[derive(Deserialize)]
struct RankedModel {
    slug: String,
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
    PARAMS_RE
        .captures_iter(&model_id.to_lowercase())
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

async fn fetch_models(api_key: &str, needed: usize) -> Result<Vec<(String, u64)>> {
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

async fn fetch_model_list(
    client: &reqwest::Client,
    api_key: &str,
    url: &str,
) -> Result<ModelsResponse> {
    let response = client
        .get(url)
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await
        .map_err(|e| eyre::eyre!("failed to fetch OpenRouter models from {url}: {e}"))?;
    if !response.status().is_success() {
        eyre::bail!(
            "OpenRouter models API returned {} for {url}",
            response.status()
        );
    }
    response
        .json()
        .await
        .map_err(|e| eyre::eyre!("failed to parse OpenRouter models response: {e}"))
}

async fn fetch_programming_ranks(
    client: &reqwest::Client,
) -> std::collections::HashMap<String, usize> {
    let result = async {
        let response = client.get(RANKINGS_URL).send().await?;
        if !response.status().is_success() {
            eyre::bail!("rankings API returned {}", response.status());
        }
        let body: RankingsResponse = response.json().await?;
        Ok::<_, eyre::Report>(body)
    }
    .await;

    match result {
        Ok(body) => {
            let ranks: std::collections::HashMap<String, usize> = body
                .data
                .into_iter()
                .enumerate()
                .map(|(rank, m)| (m.slug, rank))
                .collect();
            tracing::info!(
                count = ranks.len(),
                "openrouter programming rankings fetched"
            );
            ranks
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to fetch programming rankings, falling back to context/params sort");
            Default::default()
        }
    }
}

async fn try_fetch_models(
    client: &reqwest::Client,
    api_key: &str,
    needed: usize,
) -> Result<Vec<(String, u64)>> {
    let (all_result, rank_map) = tokio::join!(
        fetch_model_list(client, api_key, MODELS_URL),
        fetch_programming_ranks(client),
    );

    let all_body = all_result?;
    let mut models: Vec<ModelInfo> = all_body
        .data
        .into_iter()
        .filter(|m| m.pricing.is_free())
        .filter(|m| m.context_length.unwrap_or(0) >= MIN_CONTEXT_LENGTH)
        .filter(|m| !expires_within_24h(m.expiration_date.as_deref()))
        .collect();

    models.sort_by_key(|m| {
        let base_id = m.id.strip_suffix(":free").unwrap_or(&m.id);
        let rank = rank_map.get(base_id).copied().unwrap_or(usize::MAX);
        (
            rank,
            std::cmp::Reverse(parse_total_params_b(&m.id)),
            std::cmp::Reverse(m.context_length.unwrap_or(0)),
            std::cmp::Reverse(m.created.unwrap_or(0)),
        )
    });
    let ranked_summary = models
        .iter()
        .map(|m| {
            let base_id = m.id.strip_suffix(":free").unwrap_or(&m.id);
            match rank_map.get(base_id) {
                Some(rank) => format!("{}(#{})", m.id, rank + 1),
                None => format!("{}(?)", m.id),
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    tracing::info!(candidates = %ranked_summary, "experimental openrouter free candidate models");

    let models: Vec<(String, u64)> = models
        .into_iter()
        .map(|m| (m.id, m.context_length.unwrap_or(MIN_CONTEXT_LENGTH)))
        .collect();

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

fn build_openrouter_client(api_key: &str) -> Result<openrouter::Client> {
    openrouter::Client::builder()
        .api_key(api_key)
        .http_headers(openrouter_headers()?)
        .build()
        .map_err(|e| eyre::eyre!("failed to build openrouter client: {e}"))
}

async fn smoke_test_call(api_key: &str, model: &str, slot_label: &str, attempt: usize) -> bool {
    let client = match build_openrouter_client(api_key) {
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

/// Probes candidates in batches of SMOKE_TEST_BATCH_SIZE, returning indices of the first
/// `needed` passing models. Stops as soon as enough are found.
async fn probe_candidates(
    api_key: &str,
    candidates: &[(String, u64)],
    slot_label: &str,
    needed: usize,
) -> Vec<usize> {
    let mut found = Vec::new();
    let indexed: Vec<(usize, &(String, u64))> = candidates.iter().enumerate().collect();
    for chunk in indexed.chunks(SMOKE_TEST_BATCH_SIZE) {
        let handles: Vec<_> = chunk
            .iter()
            .map(|&(idx, (model_id, _))| {
                let key = api_key.to_string();
                let model = model_id.to_string();
                let label = slot_label.to_string();
                tokio::spawn(async move {
                    let passed = smoke_test_model(&key, &model, &label).await;
                    (idx, passed)
                })
            })
            .collect();
        for handle in handles {
            match handle.await {
                Ok((idx, true)) => found.push(idx),
                Ok(_) => {}
                Err(e) => tracing::error!(error = %e, "smoke test task panicked"),
            }
        }
        if found.len() >= needed {
            break;
        }
    }
    found
}

async fn select_reviewer_models(
    reviewer_routes: &[(usize, String)],
    candidates: Vec<(String, u64)>,
    fallback: bool,
) -> Result<(Vec<Option<(String, u64)>>, Vec<(String, u64)>)> {
    let reviewer_count = reviewer_routes.len();
    if reviewer_count == 0 {
        return Ok((Vec::new(), candidates));
    }

    // probe in batches per unique key; probe enough for all reviewer slots in case of overlap
    let unique_keys: std::collections::HashSet<&str> = reviewer_routes
        .iter()
        .map(|(_, key)| key.as_str())
        .collect();
    let mut key_passing: std::collections::HashMap<&str, Vec<usize>> = Default::default();
    for &key in &unique_keys {
        key_passing.insert(
            key,
            probe_candidates(key, &candidates, "reviewer_probe", reviewer_count).await,
        );
    }

    // greedy assignment: for each slot pick the first passing candidate (by quality order) not yet taken
    let mut assigned = std::collections::HashSet::new();
    let mut selected = Vec::with_capacity(reviewer_count);

    for (slot, (reviewer_index, slot_key)) in reviewer_routes.iter().enumerate() {
        let slot_label = format!("free_slot_{}", slot + 1);
        let passing = key_passing.get(slot_key.as_str()).expect("key was probed");

        let winner_idx = passing.iter().find(|&&idx| !assigned.contains(&idx));
        let Some(&winner_idx) = winner_idx else {
            if fallback {
                tracing::warn!(
                    reviewer_index,
                    "experimental OpenRouter route failed smoke testing; leaving it for fallback to skip"
                );
                selected.push(None);
                continue;
            }
            eyre::bail!(
                "No usable OpenRouter experimental free model found for {} after smoke tests of {} candidates",
                slot_label,
                candidates.len()
            );
        };

        assigned.insert(winner_idx);
        selected.push(Some(candidates[winner_idx].clone()));
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
    candidates: &[(String, u64)],
    reviewer_models: &[(String, u64)],
) -> Result<String> {
    let passing = probe_candidates(api_key, candidates, "aggregator_probe", 1).await;

    if let Some(&idx) = passing.first() {
        return Ok(candidates[idx].0.clone());
    }

    // Reusing a reviewer model is safe only after proving that the aggregator's own key can call
    // it. This matters in fallback mode, where a present but revoked key must leave the route
    // unresolved instead of borrowing another route's successful probe.
    let reviewer_passing = probe_candidates(api_key, reviewer_models, "aggregator_probe", 1).await;
    if let Some(&idx) = reviewer_passing.first() {
        let model = &reviewer_models[idx].0;
        tracing::info!(
            model = %model,
            "reusing first reviewer model for experimental openrouter free aggregator"
        );
        return Ok(model.clone());
    }

    eyre::bail!(
        "No usable OpenRouter experimental free model found for aggregator after smoke tests of {} candidates",
        candidates.len()
    )
}

fn needs_auto_model(provider: &ProviderType, model: &str) -> bool {
    matches!(provider, ProviderType::OpenRouter) && (model.is_empty() || model == "free")
}

/// Whether experimental OpenRouter auto-selection has not produced a concrete route. Fallback
/// client pools use this after best-effort resolution to skip routes whose key/catalog/probe failed.
pub fn is_unresolved_free_route(provider: &ProviderType, model: &str) -> bool {
    needs_auto_model(provider, model)
}

pub async fn resolve_free_models(config: &mut Config) -> Result<()> {
    resolve_free_models_with_fallback(config, false).await
}

/// Resolve experimental free routes. Fallback execution is best-effort: missing credentials,
/// catalog failures, and failed smoke tests leave only the affected auto-routes unresolved, and
/// client-pool construction reports and skips them. Strict execution preserves fail-fast behavior.
pub async fn resolve_free_models_with_fallback(config: &mut Config, fallback: bool) -> Result<()> {
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

    let reviewer_routes: Vec<(usize, String)> = config
        .reviewer
        .iter()
        .enumerate()
        .filter(|(_, r)| needs_auto_model(&r.provider, &r.model))
        .filter_map(|(idx, reviewer)| {
            let key_env = reviewer
                .api_key_env
                .as_deref()
                .unwrap_or(DEFAULT_API_KEY_ENV);
            match std::env::var(key_env) {
                Ok(key) => Some(Ok((idx, key))),
                Err(_) if fallback => {
                    tracing::warn!(
                        reviewer_index = idx,
                        env = key_env,
                        "experimental OpenRouter route unavailable; leaving it for fallback to skip"
                    );
                    None
                }
                Err(_) => Some(Err(eyre::eyre!("missing env var {key_env}"))),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let agg_is_free = needs_auto_model(&config.aggregator.provider, &config.aggregator.model);
    let agg_api_key = if agg_is_free {
        let key_env = config
            .aggregator
            .api_key_env
            .as_deref()
            .unwrap_or(DEFAULT_API_KEY_ENV);
        match std::env::var(key_env) {
            Ok(key) => Some(key),
            Err(_) if fallback => {
                tracing::warn!(
                    env = key_env,
                    "experimental OpenRouter aggregator unavailable; leaving it for fallback to skip"
                );
                None
            }
            Err(_) => {
                return Err(eyre::eyre!(
                    "missing env var {key_env} (required for experimental openrouter free auto-selection)"
                ));
            }
        }
    } else {
        None
    };

    let reviewer_count = reviewer_routes.len();
    if reviewer_count == 0 && agg_api_key.is_none() {
        return Ok(());
    }

    // A revoked route must not prevent a healthy auto-route from supplying the shared catalog.
    // Strict mode preserves the historical first-key failure; fallback mode tries every distinct
    // configured key and leaves all auto-routes unresolved only if none can fetch candidates.
    let mut seen_fetch_keys = std::collections::HashSet::new();
    let fetch_keys = reviewer_routes
        .iter()
        .map(|(_, key)| key.as_str())
        .chain(agg_api_key.as_deref());
    let needed_models = if fallback { 1 } else { reviewer_count.max(1) };
    let mut models = None;
    let mut last_fetch_error = None;
    for fetch_key in fetch_keys {
        if !seen_fetch_keys.insert(fetch_key) {
            continue;
        }
        match fetch_models(fetch_key, needed_models).await {
            Ok(found) => {
                models = Some(found);
                break;
            }
            Err(err) if fallback => {
                tracing::warn!(
                    error = %err,
                    "experimental OpenRouter route could not fetch the model catalog; trying another fallback route"
                );
                last_fetch_error = Some(err);
            }
            Err(err) => return Err(err),
        }
    }
    let Some(models) = models else {
        if let Some(err) = last_fetch_error {
            tracing::warn!(
                error = %err,
                "no experimental OpenRouter route could resolve free models; leaving them for fallback to skip"
            );
        }
        return Ok(());
    };
    let (reviewer_models, remaining_models) =
        select_reviewer_models(&reviewer_routes, models, fallback).await?;
    let resolved_reviewer_models = reviewer_models
        .iter()
        .filter_map(Clone::clone)
        .collect::<Vec<_>>();
    let aggregator_model = match agg_api_key.as_deref() {
        Some(api_key) => {
            match select_aggregator_model(api_key, &remaining_models, &resolved_reviewer_models)
                .await
            {
                Ok(model) => Some(model),
                Err(err) if fallback => {
                    tracing::warn!(
                        error = %err,
                        "experimental OpenRouter aggregator unavailable; leaving it for fallback to skip"
                    );
                    None
                }
                Err(err) => return Err(err),
            }
        }
        None => None,
    };

    for (slot, ((idx, _), selected)) in reviewer_routes.into_iter().zip(reviewer_models).enumerate()
    {
        let Some((model, context_length)) = selected else {
            continue;
        };
        if config.reviewer[idx].name.is_empty() {
            config.reviewer[idx].name = format!("free_{slot}");
        }
        let model_threshold = context_length.saturating_sub(COMPACT_HEADROOM);
        let configured = config.reviewer[idx]
            .compact_threshold
            .or_else(|| config.default_compact_threshold());
        let threshold = match configured {
            Some(t) if t > model_threshold => {
                tracing::warn!(
                    reviewer = %config.reviewer[idx].name,
                    configured = t,
                    capped = model_threshold,
                    context_length,
                    "compact_threshold exceeds free model context window, capping"
                );
                model_threshold
            }
            Some(t) => t,
            None => model_threshold,
        };
        config.reviewer[idx].compact_threshold = Some(threshold);
        tracing::info!(reviewer = %config.reviewer[idx].name, %model, context_length, "resolved experimental openrouter free model");
        config.reviewer[idx].model = model;
    }

    if let Some(model) = aggregator_model {
        tracing::info!(%model, "resolved experimental openrouter free aggregator model");
        config.aggregator.model = model;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AggregatorConfig, ReviewerConfig};

    fn unkeyed_free_config() -> Config {
        let missing = "NITPICKER_TEST_MISSING_OPENROUTER_FALLBACK_KEY_9E42".to_string();
        Config {
            defaults: None,
            aggregator: AggregatorConfig {
                model: "free".to_string(),
                provider: ProviderType::OpenRouter,
                base_url: None,
                api_key_env: Some(missing.clone()),
                max_tokens: None,
                auth: None,
                azure_scope: None,
                azure_credentials: None,
            },
            reviewer: vec![ReviewerConfig {
                name: String::new(),
                model: "free".to_string(),
                provider: ProviderType::OpenRouter,
                base_url: None,
                api_key_env: Some(missing),
                max_tokens: None,
                compact_threshold: None,
                auth: None,
                azure_scope: None,
                azure_credentials: None,
            }],
            presets: None,
        }
    }

    #[tokio::test]
    async fn fallback_leaves_unkeyed_free_routes_for_client_construction_to_skip() {
        let mut fallback_config = unkeyed_free_config();
        resolve_free_models_with_fallback(&mut fallback_config, true)
            .await
            .unwrap();
        assert_eq!(fallback_config.aggregator.model, "free");
        assert_eq!(fallback_config.reviewer[0].model, "free");

        let err = resolve_free_models(&mut unkeyed_free_config())
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("missing env var"));
    }

    #[test]
    fn unresolved_free_route_requires_openrouter_auto_selection() {
        assert!(is_unresolved_free_route(&ProviderType::OpenRouter, "free"));
        assert!(is_unresolved_free_route(&ProviderType::OpenRouter, ""));
        assert!(!is_unresolved_free_route(
            &ProviderType::OpenRouter,
            "openai/gpt-oss-120b:free"
        ));
        assert!(!is_unresolved_free_route(&ProviderType::OpenAi, "free"));
    }
}
