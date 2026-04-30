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
    CODE_ASSIST_BASE_URL,
    oauth::refresh_access_token,
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
    // Try to initialize project by calling loadCodeAssist
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
            ide_type: "IDE_UNSPECIFIED".to_string(),
            platform: "PLATFORM_UNSPECIFIED".to_string(),
            plugin_type: "GEMINI".to_string(),
        },
    };

    let url = format!("{}/v1internal:loadCodeAssist", CODE_ASSIST_BASE_URL);

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
            "gl-node/22.17.0",
        )
        .header(
            header::HeaderName::from_static("client-metadata"),
            "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
        )
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let text = response.text().await?;
        eyre::bail!("loadCodeAssist failed: {}", text);
    }

    let load_response: LoadCodeAssistResponse = response.json().await?;

    if let Some(project) = load_response.cloudaicompanion_project {
        info!("Got managed project: {}", project);
        if let Some(ref tier) = load_response.paid_tier {
            info!("Paid tier: {} ({})", tier.name, tier.id);
        }
        let mut project_lock = state.project_id.write().await;
        *project_lock = Some(project);
    } else {
        error!("No cloudaicompanionProject in loadCodeAssist response");
    }

    Ok(())
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
                if token.is_refreshable() {
                    (
                        StatusCode::OK,
                        format!("Token expired but refreshable - {}", project_status),
                    )
                } else {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("Token expired - login required - {}", project_status),
                    )
                }
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
    headers: HeaderMap,
    body: Bytes,
    _version: &str,
) -> impl IntoResponse {
    debug!("Received request for path: {}", path);

    // Get or refresh token
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

    // Initialize project if not already done
    let project_id = {
        let project_guard = state.project_id.read().await;
        project_guard.clone()
    };

    let project_id = match project_id {
        Some(p) => p,
        None => {
            // Try to initialize project
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

    // Parse the incoming Gemini request
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

    // Extract model from path
    let model = transform::extract_model_from_path(&path);
    debug!("Using model: {}", model);

    // Transform to Code Assist format
    let code_assist_req =
        transform::transform_request(gemini_req, model.clone(), Some(project_id.clone()));

    // Build the Code Assist API URL
    let code_assist_url = format!("{}/v1internal:generateContent", CODE_ASSIST_BASE_URL);

    // Prepare headers
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
        HeaderValue::from_static("gl-node/22.17.0"),
    );
    request_headers.insert(
        header::HeaderName::from_static("client-metadata"),
        HeaderValue::from_static(
            "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
        ),
    );

    // Set User-Agent with model to match gemini-cli format
    let user_agent = super::build_gemini_user_agent(&model);
    request_headers.insert(
        header::USER_AGENT,
        HeaderValue::from_str(&user_agent)
            .unwrap_or_else(|_| HeaderValue::from_static("GeminiCLI/0.0.0 (unknown; unknown)")),
    );

    // Request-scoped identifier for backend tracing
    request_headers.insert(
        header::HeaderName::from_static("x-activity-request-id"),
        HeaderValue::from_str(&super::create_activity_request_id())
            .unwrap_or_else(|_| HeaderValue::from_static("00000000")),
    );

    // Copy accept header from original request if present
    if let Some(value) = headers.get("accept") {
        request_headers.insert(header::ACCEPT, value.clone());
    }

    // Log the actual request for debugging
    let request_body = serde_json::to_string(&code_assist_req).unwrap_or_default();
    debug!("Request URL: {}", code_assist_url);
    let mut logged_headers = request_headers.clone();
    if let Some(value) = logged_headers.get_mut(header::AUTHORIZATION) {
        *value = HeaderValue::from_static("[redacted]");
    }
    debug!("Request headers: {:?}", logged_headers);
    debug!("Request body: {}", request_body);

    debug!("Forwarding request to Code Assist API");

    // Send request to Code Assist API with retry logic
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
            error!("Failed to forward request after retries: {}", e);
            return (
                StatusCode::BAD_GATEWAY,
                format!("Failed to connect to Code Assist API: {}", e),
            )
                .into_response();
        }
    };

    let status = response.status();
    let response_headers = response.headers().clone();

    match response.text().await {
        Ok(body) => {
            debug!("Received response from Code Assist API: status={}", status);

            // Parse and transform response
            match serde_json::from_str::<serde_json::Value>(&body) {
                Ok(json) => {
                    // Code Assist sometimes returns HTTP 4xx with an internal error code of 5xx,
                    // which are transient server-side failures. Remap them to 500 so the caller's
                    // retry logic treats them correctly.
                    let effective_status = if status.is_client_error() {
                        let inner_code = json
                            .get("error")
                            .and_then(|e| e.get("code"))
                            .and_then(|c| c.as_str())
                            .unwrap_or("");
                        if inner_code.starts_with('5') {
                            error!(
                                "Code Assist returned {} with internal code {}, remapping to 500",
                                status, inner_code
                            );
                            StatusCode::INTERNAL_SERVER_ERROR
                        } else {
                            status
                        }
                    } else {
                        status
                    };

                    // Transform response if needed
                    match transform::transform_response(json) {
                        Ok(transformed) => {
                            let mut builder = Response::builder().status(effective_status);

                            // Copy relevant headers
                            for (key, value) in response_headers.iter() {
                                if key.as_str().starts_with("content-") || key.as_str() == "x-goog"
                                {
                                    builder = builder.header(key.as_str(), value.as_bytes());
                                }
                            }

                            match builder.body(Body::from(transformed.to_string())) {
                                Ok(response) => response.into_response(),
                                Err(e) => {
                                    error!("Failed to build response: {}", e);
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        "Failed to build response",
                                    )
                                        .into_response()
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to transform response: {}", e);
                            (status, "Upstream response could not be transformed").into_response()
                        }
                    }
                }
                Err(_) => (status, "Upstream response was not JSON").into_response(),
            }
        }
        Err(e) => {
            error!("Failed to read response body: {}", e);
            (StatusCode::BAD_GATEWAY, "Failed to read response").into_response()
        }
    }
}

async fn get_valid_token(state: &ProxyState) -> Result<TokenData> {
    let mut token = state
        .token_store
        .load()?
        .ok_or_else(|| eyre::eyre!("No token found"))?;

    if !token.is_expired() {
        return Ok(token);
    }

    let _guard = state.token_refresh_lock.lock().await;
    token = state
        .token_store
        .load()?
        .ok_or_else(|| eyre::eyre!("No token found"))?;

    if !token.is_expired() {
        return Ok(token);
    }

    let refresh_token = token
        .refresh_token
        .clone()
        .ok_or_else(|| eyre::eyre!("Token expired and no refresh token available"))?;
    info!("Token expired, refreshing...");
    let refreshed = refresh_access_token(&refresh_token).await?;

    token = TokenData {
        access_token: refreshed.access_token,
        refresh_token: refreshed
            .refresh_token
            .or_else(|| token.refresh_token.clone()),
        expires_at: chrono::Utc::now() + chrono::Duration::seconds(refreshed.expires_in),
        token_type: refreshed.token_type,
    };

    state.token_store.save(&token)?;
    info!("Token refreshed successfully");

    Ok(token)
}
