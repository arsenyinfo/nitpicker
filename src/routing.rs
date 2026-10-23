//! Reviewer/aggregator client routing shared by parallel review and debate: per-reviewer
//! clients, fallback priority rings, the Alloy pool, and the aggregator's fallback chain.

use eyre::Result;
use nitpicker_agent::config::Config;
use nitpicker_agent::llm::{AlloyClient, AlloySlot, FallbackSlot, LLMClientDyn, PriorityClient};
use nitpicker_agent::provider::{build_aggregator_client, build_reviewer_client};
use std::sync::Arc;
use tracing::warn;

pub(crate) type ReviewerClientPool = Vec<std::result::Result<FallbackSlot, String>>;

pub(crate) fn build_reviewer_pool(
    config: &Config,
    gemini_proxy: &crate::proxy::GeminiProxy,
    proxy_url: Option<&str>,
) -> ReviewerClientPool {
    config
        .reviewer
        .iter()
        .map(|reviewer| {
            if nitpicker_agent::openrouter::is_unresolved_free_route(
                &reviewer.provider,
                &reviewer.model,
            ) {
                return Err("experimental OpenRouter free route could not be resolved".to_string());
            }
            gemini_proxy
                .annotate(build_reviewer_client(reviewer, proxy_url))
                .map(|client| {
                    FallbackSlot::new(client, reviewer.model.clone(), reviewer.max_tokens)
                })
                .map_err(|err| format!("{err:#}"))
        })
        .collect()
}

/// A fallback/alloy agent may run on any healthy reviewer route, while compaction happens before
/// route selection. Use the smallest configured threshold that any usable route needs.
pub(crate) fn reviewer_pool_compact_threshold(
    config: &Config,
    pool: &[std::result::Result<FallbackSlot, String>],
) -> Option<u64> {
    config
        .reviewer
        .iter()
        .zip(pool)
        .filter(|(_, route)| route.is_ok())
        .filter_map(|(reviewer, _)| config.reviewer_compact_threshold(reviewer))
        .min()
}

/// Resolve one logical reviewer to either its own client or an ordered priority ring. The ring
/// starts at that reviewer and wraps through the declaration order, so every role keeps its normal
/// primary while sharing the same simple fallback policy.
pub(crate) fn reviewer_client(
    pool: &ReviewerClientPool,
    primary_index: usize,
    fallback: bool,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    reviewer_client_with_deferred_route(pool, primary_index, None, fallback)
}

/// Give an independently progressing agent its own mutable routing state. Priority clients fork
/// their sticky index while retaining the shared route availability carried by `FallbackSlot`;
/// stateless provider and Alloy clients are safe to share directly.
pub(crate) fn independent_agent_client(template: &Arc<dyn LLMClientDyn>) -> Arc<dyn LLMClientDyn> {
    template
        .fork_for_agent()
        .unwrap_or_else(|| Arc::clone(template))
}

/// Build one side of a debate while preserving model diversity for as long as possible. The
/// counterpart's configured primary is still a valid last resort, but spare reviewers are tried
/// first so one failed side does not immediately collapse both roles onto the same model.
pub(crate) fn debate_reviewer_client(
    pool: &ReviewerClientPool,
    primary_index: usize,
    counterpart_primary_index: usize,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    reviewer_client_with_deferred_route(pool, primary_index, Some(counterpart_primary_index), true)
}

fn reviewer_client_with_deferred_route(
    pool: &ReviewerClientPool,
    primary_index: usize,
    deferred_index: Option<usize>,
    fallback: bool,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    if !fallback {
        return match &pool[primary_index] {
            Ok(slot) => Ok(slot.client()),
            Err(message) => Err(eyre::eyre!(message.clone())),
        };
    }

    let mut slots = Vec::with_capacity(pool.len());
    let mut build_errors = Vec::new();
    for index in reviewer_route_order(pool.len(), primary_index, deferred_index) {
        match &pool[index] {
            Ok(slot) => slots.push(slot.clone()),
            Err(message) => {
                warn!(
                    reviewer_index = index,
                    error = %message,
                    "reviewer route unavailable; trying next configured reviewer"
                );
                build_errors.push(message.as_str());
            }
        }
    }
    if slots.is_empty() {
        eyre::bail!(
            "no reviewer client could be built: {}",
            build_errors.join("; ")
        );
    }
    Ok(Arc::new(PriorityClient::new(slots)?))
}

fn reviewer_route_order(
    reviewer_count: usize,
    primary_index: usize,
    deferred_index: Option<usize>,
) -> Vec<usize> {
    let mut order = (0..reviewer_count)
        .map(|offset| (primary_index + offset) % reviewer_count)
        .collect::<Vec<_>>();
    if let Some(deferred_index) = deferred_index
        && deferred_index != primary_index
        && let Some(position) = order.iter().position(|&index| index == deferred_index)
    {
        order.remove(position);
        order.push(deferred_index);
    }
    order
}

pub(crate) fn alloy_client(
    pool: &ReviewerClientPool,
    fallback: bool,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    let mut slots = Vec::with_capacity(pool.len());
    for route in pool {
        match route {
            Ok(slot) => slots.push(slot.clone()),
            Err(message) if fallback => {
                warn!(error = %message, "alloy route unavailable; continuing with healthy reviewers")
            }
            Err(message) => return Err(eyre::eyre!(message.clone())),
        }
    }
    let client = match fallback {
        true => AlloyClient::new_with_fallback_routes(slots)?,
        false => AlloyClient::new(slots.iter().map(AlloySlot::from).collect())?,
    };
    Ok(Arc::new(client))
}

pub(crate) fn aggregator_client(
    config: &Config,
    gemini_proxy: &crate::proxy::GeminiProxy,
    proxy_url: Option<&str>,
    reviewer_pool: &ReviewerClientPool,
    fallback: bool,
) -> Result<Arc<dyn nitpicker_agent::llm::LLMClientDyn>> {
    let primary = if nitpicker_agent::openrouter::is_unresolved_free_route(
        &config.aggregator.provider,
        &config.aggregator.model,
    ) {
        Err(eyre::eyre!(
            "experimental OpenRouter free aggregator could not be resolved"
        ))
    } else {
        gemini_proxy.annotate(build_aggregator_client(&config.aggregator, proxy_url))
    };
    if !fallback {
        return primary;
    }

    let mut slots = Vec::with_capacity(1 + reviewer_pool.len());
    match primary {
        Ok(client) => slots.push(FallbackSlot::new(
            client,
            config.aggregator.model.clone(),
            Some(config.aggregator_max_tokens()),
        )),
        Err(err) => warn!(error = %err, "aggregator route unavailable; trying reviewer pool"),
    }
    for route in reviewer_pool {
        match route {
            Ok(slot) => slots.push(slot.clone()),
            Err(message) => warn!(
                error = %message,
                "reviewer route unavailable for aggregator fallback"
            ),
        }
    }
    if slots.is_empty() {
        eyre::bail!("no aggregator or reviewer fallback client could be built");
    }
    Ok(Arc::new(PriorityClient::new(slots)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nitpicker_agent::llm::{Completion, CompletionResponse, LLMClient};

    struct UnusedClient;

    impl LLMClient for UnusedClient {
        async fn completion(&self, _completion: Completion) -> Result<CompletionResponse> {
            unreachable!("threshold tests never call the model")
        }
    }

    fn usable_slot(model: &str) -> std::result::Result<FallbackSlot, String> {
        Ok(FallbackSlot::new(Arc::new(UnusedClient), model, None))
    }

    #[test]
    fn fallback_compaction_uses_smallest_usable_route_threshold() {
        let config: Config = toml::from_str(
            r#"
                [aggregator]
                provider = "openai"

                [[reviewer]]
                model = "large"
                provider = "openai"
                compact_threshold = 120000

                [[reviewer]]
                model = "small"
                provider = "openai"
                compact_threshold = 24000

                [[reviewer]]
                model = "unset"
                provider = "openai"
            "#,
        )
        .unwrap();

        let all_usable = vec![
            usable_slot("large"),
            usable_slot("small"),
            usable_slot("unset"),
        ];
        assert_eq!(
            reviewer_pool_compact_threshold(&config, &all_usable),
            Some(24_000)
        );

        let small_failed = vec![
            usable_slot("large"),
            Err("small failed to build".to_string()),
            usable_slot("unset"),
        ];
        assert_eq!(
            reviewer_pool_compact_threshold(&config, &small_failed),
            Some(120_000)
        );

        let only_unset_usable = vec![
            Err("large failed to build".to_string()),
            Err("small failed to build".to_string()),
            usable_slot("unset"),
        ];
        assert_eq!(
            reviewer_pool_compact_threshold(&config, &only_unset_usable),
            None
        );

        let defaulted: Config = toml::from_str(
            r#"
                [defaults]
                compact_threshold = 50000

                [aggregator]
                provider = "openai"

                [[reviewer]]
                model = "defaulted"
                provider = "openai"
            "#,
        )
        .unwrap();
        assert_eq!(
            reviewer_pool_compact_threshold(&defaulted, &[usable_slot("defaulted")]),
            Some(50_000)
        );
    }

    #[test]
    fn debate_fallback_defers_the_counterpart_primary() {
        assert_eq!(reviewer_route_order(4, 0, Some(1)), vec![0, 2, 3, 1]);
        assert_eq!(reviewer_route_order(4, 1, Some(0)), vec![1, 2, 3, 0]);

        // With no spare route, using the counterpart is still preferable to aborting the run.
        assert_eq!(reviewer_route_order(2, 0, Some(1)), vec![0, 1]);
        assert_eq!(reviewer_route_order(2, 1, Some(0)), vec![1, 0]);
    }

    #[test]
    fn ordinary_review_fallback_keeps_declaration_order() {
        assert_eq!(reviewer_route_order(4, 0, None), vec![0, 1, 2, 3]);
        assert_eq!(reviewer_route_order(4, 2, None), vec![2, 3, 0, 1]);
    }
}
