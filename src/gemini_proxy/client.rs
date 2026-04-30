use crate::gemini_proxy::{
    oauth::{build_authorization_url, exchange_code_for_token, generate_pkce_challenge},
    proxy::{ProxyState, run_proxy_internal},
    token::{TokenData, TokenStore},
};
use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    response::Html,
    routing::get,
};
use eyre::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

/// Type alias for the OAuth callback channel state to reduce complexity
type OAuthCallbackState = Arc<mpsc::Sender<(oneshot::Sender<String>, String, String)>>;

/// Client that manages OAuth authentication and runs a local proxy
pub struct GeminiProxyClient {
    port: u16,
    token_store: TokenStore,
    _shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl GeminiProxyClient {
    pub async fn new() -> Result<Self> {
        let token_store = TokenStore::new()?;

        // Check if we need to authenticate
        if let Ok(Some(token)) = token_store.load() {
            if token.is_expired() && !token.is_refreshable() {
                info!("Token expired and not refreshable, triggering authentication...");
                authenticate_interactive(&token_store).await?;
            }
        } else {
            info!("No token found, triggering authentication...");
            authenticate_interactive(&token_store).await?;
        }

        // Bind a listener now so the port is held; pass it to the server task
        let listener = find_available_port().await?;
        let port = listener.local_addr()?.port();
        info!("Starting Gemini proxy on port {}", port);

        let state = Arc::new(ProxyState {
            token_store: token_store.clone(),
            http_client: reqwest::Client::new(),
            project_id: Arc::new(tokio::sync::RwLock::new(None)),
            token_refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
            retry_state: super::retry::RetryState::new(),
        });

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            if let Err(e) = run_proxy_internal(listener, state, shutdown_rx).await {
                error!("Proxy server error: {}", e);
            }
        });

        Ok(Self {
            port,
            token_store,
            _shutdown_tx: shutdown_tx,
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn check_auth_status(&self) -> Result<AuthStatus> {
        match self.token_store.load()? {
            Some(token) => {
                if token.is_expired() {
                    if token.is_refreshable() {
                        Ok(AuthStatus::ExpiredButRefreshable)
                    } else {
                        Ok(AuthStatus::Expired)
                    }
                } else {
                    Ok(AuthStatus::Valid)
                }
            }
            None => Ok(AuthStatus::NotAuthenticated),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AuthStatus {
    Valid,
    ExpiredButRefreshable,
    Expired,
    NotAuthenticated,
}

async fn find_available_port() -> Result<tokio::net::TcpListener> {
    let base = 15000 + (std::process::id() % 10000) as u16;
    for port in base..=base + 1000 {
        match tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await {
            Ok(listener) => return Ok(listener),
            Err(_) => continue,
        }
    }
    eyre::bail!("Could not find an available port")
}

async fn authenticate_interactive(token_store: &TokenStore) -> Result<()> {
    info!("Starting OAuth login flow...");

    // Generate PKCE challenge
    let (code_verifier, code_challenge) = generate_pkce_challenge();
    let state = uuid::Uuid::new_v4().to_string();

    // Channel to receive the authorization code
    let (tx, mut rx) = mpsc::channel::<(oneshot::Sender<String>, String, String)>(1);

    // Bind first so we know the actual port before building the auth URL
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("Failed to bind OAuth callback server")?;
    let port = listener.local_addr()?.port();
    let redirect_uri = format!("http://localhost:{}/oauth2callback", port);
    info!(
        "OAuth callback server listening on http://127.0.0.1:{}",
        port
    );

    let auth_url = build_authorization_url(&state, &code_challenge, &redirect_uri)?;

    // Start local callback server
    let app = Router::new()
        .route("/oauth2callback", get(oauth_callback_handler))
        .with_state(Arc::new(tx));

    let server_handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            error!("OAuth callback server error: {}", e);
        }
    });

    // Print authorization URL and wait for callback
    println!("\n=== Gemini OAuth Authentication ===\n");
    println!("Opening browser to authorize the application...\n");
    println!("If the browser doesn't open automatically, visit:");
    println!("{}\n", auth_url);

    // Try to open browser
    if let Err(e) = webbrowser::open(&auth_url) {
        info!("Failed to open browser: {}", e);
        println!("Please manually open the URL above in your browser.");
    }

    // Wait for callback
    println!("Waiting for authorization...");

    let result = tokio::time::timeout(tokio::time::Duration::from_secs(120), rx.recv()).await;

    match result {
        Ok(Some((return_tx, code, received_state))) => {
            // Verify state
            if received_state != state {
                let _ = return_tx.send("Invalid state parameter".to_string());
                error!("State mismatch - possible CSRF attack");
                return Err(eyre::eyre!("Invalid state parameter"));
            }

            // Exchange code for token
            match exchange_code_for_token(&code, &code_verifier, &redirect_uri).await {
                Ok(token_response) => {
                    let token = TokenData {
                        access_token: token_response.access_token,
                        refresh_token: token_response.refresh_token,
                        expires_at: chrono::Utc::now()
                            + chrono::Duration::seconds(token_response.expires_in),
                        token_type: token_response.token_type,
                    };

                    token_store.save(&token)?;

                    let _ = return_tx.send("success".to_string());

                    println!("\n✓ Authentication successful!");
                    println!("Token saved and will be used for Gemini API calls.");
                    println!("Token expires at: {}\n", token.expires_at);
                }
                Err(e) => {
                    let _ = return_tx.send(format!("Token exchange failed: {}", e));
                    return Err(e);
                }
            }
        }
        Ok(None) => {
            return Err(eyre::eyre!("OAuth channel closed unexpectedly"));
        }
        Err(_) => {
            return Err(eyre::eyre!("OAuth timeout - took too long to authenticate"));
        }
    }

    // Clean up server
    drop(server_handle);

    Ok(())
}

async fn oauth_callback_handler(
    Query(params): Query<HashMap<String, String>>,
    State(tx): State<OAuthCallbackState>,
) -> impl axum::response::IntoResponse {
    let code = params.get("code").cloned().unwrap_or_default();
    let state = params.get("state").cloned().unwrap_or_default();
    let error = params.get("error").cloned();

    let (response_tx, mut response_rx) = oneshot::channel::<String>();

    if let Some(err) = error {
        let _ = tx.send((response_tx, String::new(), state.clone())).await;
        return (
            StatusCode::BAD_REQUEST,
            Html(format!(
                "<h1>Authorization Failed</h1><p>Error: {}</p>",
                err
            )),
        );
    }

    if code.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Html("<h1>Authorization Failed</h1><p>No authorization code received.</p>".to_string()),
        );
    }

    let _ = tx.send((response_tx, code, state)).await;

    // Wait for token exchange result
    match tokio::time::timeout(tokio::time::Duration::from_secs(30), &mut response_rx).await {
        Ok(Ok(result)) if result == "success" => (
            StatusCode::OK,
            Html(
                "<h1>Authorization Successful!</h1><p>You can close this window and return to the terminal.</p>"
                    .to_string(),
            ),
        ),
        Ok(Ok(error_msg)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Html(format!("<h1>Authorization Failed</h1><p>{}</p>", error_msg)),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Html("<h1>Timeout</h1><p>Authorization timed out.</p>".to_string()),
        ),
    }
}
