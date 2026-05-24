use crate::gemini_proxy::{
    proxy::{ProxyState, run_proxy_internal},
    token::TokenStore,
};
use eyre::Result;
use std::sync::Arc;
use tracing::{error, info};

/// Client that runs a local proxy backed by the Antigravity keyring token.
pub struct GeminiProxyClient {
    port: u16,
    _shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl GeminiProxyClient {
    pub async fn new() -> Result<Self> {
        let token_store = TokenStore::new_agy_keyring();

        match token_store.load()? {
            Some(token) if token.is_expired() => {
                eyre::bail!("Antigravity keyring token is expired; run `agy` to refresh it");
            }
            Some(_) => {}
            None => {
                eyre::bail!(
                    "Antigravity keyring token not found; run `agy` once to populate the system keyring"
                );
            }
        }

        let listener = find_available_port().await?;
        let port = listener.local_addr()?.port();
        info!("Starting Gemini proxy on port {}", port);

        let state = Arc::new(ProxyState {
            token_store,
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
            _shutdown_tx: shutdown_tx,
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
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
