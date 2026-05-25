use axum::{
    Router,
    body::{Body, Bytes},
    extract::{Path, State},
    http::{HeaderMap, HeaderValue, Response, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use eyre::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

use super::{
    antigravity_platform, code_assist_base_url,
    retry::{RetryState, fetch_with_retry},
    token::{TokenData, TokenStore},
    transform,
};

#[derive(Clone)]
pub struct ProxyState {
    pub token_store: TokenStore,
    pub http_client: reqwest::Client,
    pub project_id: Arc<RwLock<Option<String>>>,
    pub token_refresh_lock: Arc<tokio::sync::Mutex<()>>,
    pub retry_state: RetryState,
}

#[derive(Debug, Serialize)]
struct LoadCodeAssistRequest {
    metadata: ClientMetadata,
}

#[derive(Debug, Serialize)]
struct FetchAvailableModelsRequest {
    project: String,
    #[serde(rename = "requestId")]
    request_id: String,
}

#[derive(Debug, Serialize)]
struct ClientMetadata {
    #[serde(rename = "ideType")]
    ide_type: String,
    platform: String,
    #[serde(rename = "pluginType")]
    plugin_type: String,
}

#[derive(Debug, Deserialize)]
struct LoadCodeAssistResponse {
    #[serde(rename = "cloudaicompanionProject")]
    cloudaicompanion_project: Option<String>,
    #[serde(rename = "paidTier")]
    paid_tier: Option<PaidTier>,
}

#[derive(Debug, Deserialize)]
struct PaidTier {
    id: String,
    name: String,
}

pub async fn run_proxy_internal(
    listener: tokio::net::TcpListener,
    state: Arc<ProxyState>,
    shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) -> Result<()> {
    // try to initialize project by calling loadCodeAssist
    match get_valid_token(&state).await {
        Ok(token) => {
            if let Err(e) = init_project(&state, &token).await {
                error!("Failed to initialize project: {}", e);
            }
        }
        Err(e) => error!("Failed to get valid token for init: {}", e),
    }

    let app = Router::new()
        .route("/", get(root_handler))
        .route("/v1beta/{*path}", post(handle_v1beta))
        .route("/v1/{*path}", post(handle_v1))
        .route("/health", get(health_handler))
        .with_state(state);

    info!(
        "Starting Gemini proxy server on http://{}",
        listener.local_addr()?
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await?;

    Ok(())
}

async fn init_project(state: &ProxyState, token: &TokenData) -> Result<()> {
    info!("Initializing project via loadCodeAssist...");

    let request = LoadCodeAssistRequest {
        metadata: ClientMetadata {
            ide_type: "ANTIGRAVITY".to_string(),
            platform: antigravity_platform(),
            plugin_type: "GEMINI".to_string(),
        },
    };

    let url = format!("{}/v1internal:loadCodeAssist", code_assist_base_url());

    let response = state
        .http_client
        .post(&url)
        .header(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", token.access_token))
                .map(|mut value| {
                    value.set_sensitive(true);
                    value
                })
                .map_err(|_| eyre::eyre!("Token contains invalid header characters"))?,
        )
        .header(header::CONTENT_TYPE, "application/json")
        .header(
            header::HeaderName::from_static("x-goog-api-client"),
            "google-api-go-client/0.5",
        )
        .header(
            header::HeaderName::from_static("client-metadata"),
            // ideType=ANTIGRAVITY identifies as the AG2 native CLI; platform follows
            // the binary's MACOS/LINUX/WINDOWS convention.
            &format!(
                "ideType=ANTIGRAVITY,platform={},pluginType=GEMINI",
                antigravity_platform()
            ),
        )
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let text = response.text().await?;
        eyre::bail!("loadCodeAssist failed: {}", text);
    }

    let raw = response.text().await?;
    let load_response: LoadCodeAssistResponse = serde_json::from_str(&raw)
        .map_err(|e| eyre::eyre!("Failed to parse loadCodeAssist response: {e}\nbody: {raw}"))?;

    if let Some(project) = load_response.cloudaicompanion_project {
        info!("Got managed project: {}", project);
        if let Some(ref tier) = load_response.paid_tier {
            info!("Paid tier: {} ({})", tier.name, tier.id);
        }
        if let Err(e) = fetch_available_models(state, token, &project).await {
            error!("Failed to fetch available models: {:#}", e);
        }
        let mut project_lock = state.project_id.write().await;
        *project_lock = Some(project);
    } else {
        error!(
            "No cloudaicompanionProject in loadCodeAssist response. raw body: {}",
            raw
        );
    }

    Ok(())
}

async fn fetch_available_models(
    state: &ProxyState,
    token: &TokenData,
    project: &str,
) -> Result<()> {
    let request = FetchAvailableModelsRequest {
        project: project.to_string(),
        request_id: uuid::Uuid::new_v4().to_string(),
    };
    let url = format!("{}/v1internal:fetchAvailableModels", code_assist_base_url());

    let response = state
        .http_client
        .post(&url)
        .header(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", token.access_token))
                .map(|mut value| {
                    value.set_sensitive(true);
                    value
                })
                .map_err(|_| eyre::eyre!("Token contains invalid header characters"))?,
        )
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json")
        .header(
            header::USER_AGENT,
            super::build_gemini_user_agent("fetchAvailableModels"),
        )
        .header(
            header::HeaderName::from_static("x-goog-api-client"),
            "google-api-go-client/0.5",
        )
        .header(
            header::HeaderName::from_static("client-metadata"),
            &format!(
                "ideType=ANTIGRAVITY,platform={},pluginType=GEMINI",
                antigravity_platform()
            ),
        )
        .json(&request)
        .send()
        .await?;

    let status = response.status();
    let raw = response.text().await?;
    if !status.is_success() {
        eyre::bail!("fetchAvailableModels failed: status={status} body={raw}");
    }

    let json: serde_json::Value = serde_json::from_str(&raw)?;
    info!("Available models: {}", summarize_available_models(&json));
    Ok(())
}

fn summarize_available_models(json: &serde_json::Value) -> String {
    let default_agent = json
        .get("defaultAgentModelId")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("<none>");
    let tiered = json
        .get("tieredModelIds")
        .map(serde_json::Value::to_string)
        .unwrap_or_else(|| "<none>".to_string());
    let mut model_ids = json
        .get("models")
        .and_then(serde_json::Value::as_object)
        .map(|models| models.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    model_ids.sort();
    let total = model_ids.len();
    let preview = model_ids
        .into_iter()
        .take(30)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "defaultAgentModelId={default_agent}, tieredModelIds={tiered}, models({total})=[{preview}]"
    )
}

async fn root_handler() -> impl IntoResponse {
    "Gemini OAuth Proxy - Use /v1beta/models/{model}:generateContent"
}

async fn health_handler(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let project_guard = state.project_id.read().await;
    let project_status = if project_guard.is_some() {
        "Project initialized"
    } else {
        "Project not initialized"
    };
    drop(project_guard);

    match state.token_store.load() {
        Ok(Some(token)) => {
            if token.is_expired() {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Token expired - run `agy` to refresh - {}", project_status),
                )
            } else {
                (
                    StatusCode::OK,
                    format!("Healthy - Token valid - {}", project_status),
                )
            }
        }
        Ok(None) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "No token - login required".to_string(),
        ),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Error reading token".to_string(),
        ),
    }
}

async fn handle_v1beta(
    State(state): State<Arc<ProxyState>>,
    Path(path): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    handle_request(state, path, headers, body, "v1beta").await
}

async fn handle_v1(
    State(state): State<Arc<ProxyState>>,
    Path(path): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    handle_request(state, path, headers, body, "v1").await
}

async fn handle_request(
    state: Arc<ProxyState>,
    path: String,
    _headers: HeaderMap,
    body: Bytes,
    _version: &str,
) -> impl IntoResponse {
    debug!("Received request for path: {}", path);

    // get or refresh token
    let token = match get_valid_token(&state).await {
        Ok(token) => token,
        Err(e) => {
            error!("Failed to get valid token: {}", e);
            return (
                StatusCode::UNAUTHORIZED,
                format!("Authentication required: {}", e),
            )
                .into_response();
        }
    };

    // initialize project if not already done
    let project_id = {
        let project_guard = state.project_id.read().await;
        project_guard.clone()
    };

    let project_id = match project_id {
        Some(p) => p,
        None => {
            // try to initialize project
            if let Err(e) = init_project(&state, &token).await {
                error!("Failed to initialize project: {}", e);
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Failed to initialize project: {}", e),
                )
                    .into_response();
            }
            let project_guard = state.project_id.read().await;
            match project_guard.clone() {
                Some(p) => p,
                None => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Could not get project from loadCodeAssist".to_string(),
                    )
                        .into_response();
                }
            }
        }
    };

    // parse the incoming Gemini request
    let gemini_req: transform::GeminiRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => {
            error!("Failed to parse request body: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                format!("Invalid request body: {}", e),
            )
                .into_response();
        }
    };

    // extract model from path
    let model = transform::extract_model_from_path(&path);
    debug!("Using model: {}", model);

    // transform to Code Assist format
    let code_assist_req =
        transform::transform_request(gemini_req, model.clone(), Some(project_id.clone()));

    // build the Code Assist API URL. agy's real chat traffic uses SSE even for
    // print mode; the non-streaming endpoint returns 403 for AG2 tokens.
    let code_assist_url = format!(
        "{}/v1internal:streamGenerateContent?alt=sse",
        code_assist_base_url()
    );

    // prepare headers
    let mut request_headers = HeaderMap::new();
    let auth_value = match HeaderValue::from_str(&format!("Bearer {}", token.access_token)) {
        Ok(mut v) => {
            v.set_sensitive(true);
            v
        }
        Err(_) => {
            error!("Token contains invalid header characters");
            return (StatusCode::INTERNAL_SERVER_ERROR, "Invalid token format").into_response();
        }
    };
    request_headers.insert(header::AUTHORIZATION, auth_value);
    request_headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    request_headers.insert(
        header::HeaderName::from_static("x-goog-api-client"),
        HeaderValue::from_static("google-api-go-client/0.5"),
    );
    let client_metadata = format!(
        "ideType=ANTIGRAVITY,platform={},pluginType=GEMINI",
        antigravity_platform()
    );
    if let Ok(v) = HeaderValue::from_str(&client_metadata) {
        request_headers.insert(header::HeaderName::from_static("client-metadata"), v);
    }
    // set User-Agent with model to match gemini-cli format
    let user_agent = super::build_gemini_user_agent(&model);
    request_headers.insert(
        header::USER_AGENT,
        HeaderValue::from_str(&user_agent)
            .unwrap_or_else(|_| HeaderValue::from_static("GeminiCLI/0.0.0 (unknown; unknown)")),
    );

    // request-scoped identifier for backend tracing
    request_headers.insert(
        header::HeaderName::from_static("x-activity-request-id"),
        HeaderValue::from_str(&super::create_activity_request_id())
            .unwrap_or_else(|_| HeaderValue::from_static("00000000")),
    );

    request_headers.insert(
        header::ACCEPT,
        HeaderValue::from_static("text/event-stream"),
    );

    // log the actual request for debugging
    let request_body = serde_json::to_string(&code_assist_req).unwrap_or_default();
    debug!("Request URL: {}", code_assist_url);
    let mut logged_headers = request_headers.clone();
    if let Some(value) = logged_headers.get_mut(header::AUTHORIZATION) {
        *value = HeaderValue::from_static("[redacted]");
    }
    debug!("Request headers: {:?}", logged_headers);
    debug!("Request body: {}", request_body);

    debug!("Forwarding request to Code Assist API");

    // send request to Code Assist API with retry logic
    let request_builder = state
        .http_client
        .post(&code_assist_url)
        .headers(request_headers)
        .json(&code_assist_req);

    let response = match fetch_with_retry(
        request_builder,
        &state.retry_state,
        &code_assist_url,
        Some(&project_id),
        Some(&model),
    )
    .await
    {
        Ok(resp) => resp,
        Err(e) => {
            // fail-hard: surface the full eyre/reqwest error chain (TLS, DNS, reset, etc.).
            error!("Failed to forward request: {:#}", e);
            let detail = e
                .chain()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" -> ");
            return (
                StatusCode::BAD_GATEWAY,
                format!("Failed to connect to Code Assist API: {detail}"),
            )
                .into_response();
        }
    };

    let status = response.status();
    let response_headers = response.headers().clone();

    let body = match response.text().await {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to read response body: {:#}", e);
            return (
                StatusCode::BAD_GATEWAY,
                format!("Failed to read upstream response: {e}"),
            )
                .into_response();
        }
    };

    debug!("Received response from Code Assist API: status={}", status);

    // fail-hard: on any non-2xx, return upstream status + body verbatim so the
    // caller sees the exact Google error (code, status, message). No remapping,
    // no transform.
    if !status.is_success() {
        error!(
            "Code Assist API non-success: status={} body={}",
            status, body
        );
        let mut builder = Response::builder().status(status);
        for (key, value) in response_headers.iter() {
            if key.as_str().starts_with("content-") {
                builder = builder.header(key.as_str(), value.as_bytes());
            }
        }
        return builder
            .body(Body::from(body))
            .map(IntoResponse::into_response)
            .unwrap_or_else(|e| {
                error!("Failed to build error pass-through response: {}", e);
                (StatusCode::BAD_GATEWAY, "proxy build failure").into_response()
            });
    }

    let json: serde_json::Value = match transform::transform_stream_response(&body) {
        Ok(j) => j,
        Err(stream_err) => match serde_json::from_str(&body) {
            Ok(j) => j,
            Err(json_err) => {
                error!(
                    "Upstream 2xx body was neither SSE nor JSON: stream={:#} json={} body={}",
                    stream_err, json_err, body
                );
                return (StatusCode::BAD_GATEWAY, body).into_response();
            }
        },
    };

    let transformed = match transform::transform_response(json) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to transform response: {:#}", e);
            return (status, format!("transform failed: {e:#}")).into_response();
        }
    };

    let mut builder = Response::builder().status(status);
    for (key, value) in response_headers.iter() {
        if key.as_str().starts_with("content-") || key.as_str() == "x-goog" {
            builder = builder.header(key.as_str(), value.as_bytes());
        }
    }
    builder
        .body(Body::from(transformed.to_string()))
        .map(IntoResponse::into_response)
        .unwrap_or_else(|e| {
            error!("Failed to build response: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to build response",
            )
                .into_response()
        })
}

async fn get_valid_token(state: &ProxyState) -> Result<TokenData> {
    // serialize concurrent loads so we don't hammer the keyring on a burst of requests.
    let _guard = state.token_refresh_lock.lock().await;
    let token = state
        .token_store
        .load()?
        .ok_or_else(|| eyre::eyre!("Antigravity keyring token not found; run `agy` once"))?;

    if token.is_expired() {
        eyre::bail!("Antigravity keyring token is expired; run `agy` to refresh it");
    }

    Ok(token)
}
