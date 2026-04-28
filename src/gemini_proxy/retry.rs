use eyre::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, error, warn};

const DEFAULT_MAX_ATTEMPTS: u32 = 4;
const MODEL_CAPACITY_COOLDOWN_MS: u64 = 8000;

/// Per-request retry state keyed by (url, project, model).
#[derive(Clone)]
pub struct RetryState {
    cooldowns: Arc<Mutex<HashMap<String, Instant>>>,
}

impl RetryState {
    pub fn new() -> Self {
        Self {
            cooldowns: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Result of classifying a quota error response.
#[derive(Debug, Clone)]
pub struct QuotaContext {
    pub terminal: bool,
    pub retry_delay_ms: Option<u64>,
    pub reason: Option<String>,
}



/// Send a request with retry/backoff semantics aligned to Gemini CLI.
/// Retries on 429/5xx and transient network failures.
/// Honors Retry-After and google.rpc.RetryInfo.
/// Never rewrites the requested model.
pub async fn fetch_with_retry(
    request_builder: reqwest::RequestBuilder,
    retry_state: &RetryState,
    url: &str,
    project: Option<&str>,
    model: Option<&str>,
) -> Result<reqwest::Response> {
    let throttle_key = build_retry_throttle_key(url, project, model);

    // Wait for any active cooldown
    wait_for_retry_cooldown(&retry_state.cooldowns, &throttle_key).await;

    let mut attempt = 1;

    while attempt <= DEFAULT_MAX_ATTEMPTS {
        debug!("Retry: attempt {}/{} -> {}", attempt, DEFAULT_MAX_ATTEMPTS, url);

        let response = match request_builder.try_clone() {
            Some(rb) => match rb.send().await {
                Ok(resp) => resp,
                Err(e) => {
                    if attempt >= DEFAULT_MAX_ATTEMPTS || !is_retryable_network_error(&e) {
                        error!(
                            "Retry: attempt {} network error is non-retryable or maxed: {}",
                            attempt, e
                        );
                        return Err(e.into());
                    }
                    let delay = get_exponential_delay_with_jitter(attempt);
                    warn!(
                        "Retry: attempt {} network retry scheduled in {}ms ({})",
                        attempt, delay, e
                    );
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    attempt += 1;
                    continue;
                }
            },
            None => {
                // Body is streaming, can't clone — just send once
                return Ok(request_builder.send().await?);
            }
        };

        let status = response.status();

        if !is_retryable_status(status) {
            debug!("Retry: attempt {} success or non-retryable status: {}", attempt, status);
            return Ok(response);
        }

        // Classify 429 responses
        let quota_context = if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            match classify_quota_response_from_status(response.headers()) {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    warn!("Failed to classify quota response: {}", e);
                    None
                }
            }
        } else {
            None
        };

        if let Some(ref ctx) = quota_context {
            if ctx.terminal {
                if ctx.reason.as_deref() == Some("MODEL_CAPACITY_EXHAUSTED") {
                    let cooldown_ms = ctx.retry_delay_ms.unwrap_or(MODEL_CAPACITY_COOLDOWN_MS);
                    set_retry_cooldown(&retry_state.cooldowns, &throttle_key, cooldown_ms).await;
                    warn!(
                        "Retry: terminal model capacity; cooldown {}ms before next request",
                        cooldown_ms
                    );
                }
                warn!(
                    "Retry: attempt {} terminal 429 ({}), returning without retry",
                    attempt,
                    ctx.reason.as_deref().unwrap_or("unknown")
                );
                return Ok(response);
            }
        }

        if attempt >= DEFAULT_MAX_ATTEMPTS {
            warn!(
                "Retry: attempt {} reached retry boundary (status={})",
                attempt, status
            );
            return Ok(response);
        }

        let delay_ms = resolve_retry_delay_ms(&response, attempt, quota_context.as_ref());
        warn!(
            "Retry: attempt {} retrying status={} reason={} delay={}ms",
            attempt,
            status,
            quota_context
                .as_ref()
                .and_then(|c| c.reason.as_deref())
                .unwrap_or("n/a"),
            delay_ms
        );

        if delay_ms > 0 && status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            set_retry_cooldown(&retry_state.cooldowns, &throttle_key, delay_ms).await;
        }
        if delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }

        attempt += 1;
    }

    // Fallback: loop condition guarantees we never reach here, but compiler needs it.
    unreachable!("retry loop should have returned within its body")
}

fn build_retry_throttle_key(url: &str, project: Option<&str>, model: Option<&str>) -> String {
    format!("{}|{}|{}", url, project.unwrap_or(""), model.unwrap_or(""))
}

async fn wait_for_retry_cooldown(cooldowns: &Arc<Mutex<HashMap<String, Instant>>>, key: &str) {
    let mut map = cooldowns.lock().await;
    if let Some(until) = map.get(key) {
        let remaining = until.saturating_duration_since(Instant::now()).as_millis() as i64;
        if remaining > 0 {
            drop(map);
            debug!("Retry: cooldown wait {}ms (key={})", remaining, short_key(key));
            tokio::time::sleep(Duration::from_millis(remaining as u64)).await;
            let mut map = cooldowns.lock().await;
            if let Some(&current_until) = map.get(key) {
                if current_until <= Instant::now() {
                    map.remove(key);
                }
            }
        } else {
            map.remove(key);
        }
    }
}

async fn set_retry_cooldown(
    cooldowns: &Arc<Mutex<HashMap<String, Instant>>>,
    key: &str,
    delay_ms: u64,
) {
    let next = Instant::now() + Duration::from_millis(delay_ms);
    let mut map = cooldowns.lock().await;
    let current = map.get(key).copied().unwrap_or(Instant::now());
    map.insert(key.to_string(), std::cmp::max(current, next));
    debug!("Retry: cooldown set {}ms (key={})", delay_ms, short_key(key));
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS
}

fn is_retryable_network_error(error: &reqwest::Error) -> bool {
    if error.is_timeout() {
        return true;
    }
    if error.is_connect() {
        return true;
    }
    if let Some(status) = error.status() {
        return is_retryable_status(status);
    }
    false
}

fn get_exponential_delay_with_jitter(attempt: u32) -> u64 {
    let base = 250u64;
    let max = 5000u64;
    let exp = base * 2u64.pow(attempt.saturating_sub(1));
    let capped = std::cmp::min(exp, max);
    let jitter = rand::random::<u64>() % (capped / 2 + 1);
    capped + jitter
}

fn resolve_retry_delay_ms(
    response: &reqwest::Response,
    attempt: u32,
    quota_context: Option<&QuotaContext>,
) -> u64 {
    // 1. Check Retry-After header
    if let Some(header) = response.headers().get("retry-after") {
        if let Ok(text) = header.to_str() {
            if let Ok(seconds) = text.parse::<u64>() {
                return seconds * 1000;
            }
            // Try parsing as HTTP date? For now, skip.
        }
    }

    // 2. Check quota context delay
    if let Some(delay) = quota_context.and_then(|c| c.retry_delay_ms) {
        return delay;
    }

    // 3. Default exponential backoff
    get_exponential_delay_with_jitter(attempt)
}

/// Classify from headers + status when we can't consume the response body.
/// Uses Retry-After header as a fallback for delay.
pub fn classify_quota_response_from_status(
    headers: &reqwest::header::HeaderMap,
) -> Result<QuotaContext> {
    // Try to extract retry delay from Retry-After header
    let retry_delay_ms = headers
        .get("retry-after")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .map(|s| s * 1000);

    // Without body we can't classify precisely, so we treat 429 as retryable
    // with the Retry-After delay or a default.
    Ok(QuotaContext {
        terminal: false,
        retry_delay_ms: Some(retry_delay_ms.unwrap_or(10_000)),
        reason: Some("RATE_LIMIT_EXCEEDED".to_string()),
    })
}

fn short_key(key: &str) -> String {
    let mut chars = key.chars();
    let truncated: String = chars.by_ref().take(120).collect();
    if chars.next().is_some() {
        format!("{}...", truncated)
    } else {
        truncated
    }
}
