use eyre::{Result, WrapErr};
use nitpicker_agent::config::Config;

/// Owns the optional Gemini AG2 proxy for a run. The inner client's drop shuts the local
/// server down, so the handle must stay bound for as long as any client may call through it —
/// only the base URL is threaded downstream.
pub struct GeminiProxy {
    #[cfg(feature = "antigravity")]
    client: Option<crate::gemini_proxy::GeminiProxyClient>,
    /// Rendered startup failure, kept so client-build errors for proxy-needing reviewers
    /// can carry the root cause. Always `None` when the feature is off.
    startup_error: Option<String>,
}

impl GeminiProxy {
    /// Starts the proxy when any configured reviewer/aggregator uses `auth = "agy-keyring"`
    /// (feature `antigravity`); otherwise returns an inert handle. A startup failure (e.g.
    /// missing/expired keyring token) also returns an inert handle rather than an error:
    /// clients that need the proxy fail individually at build time — with the cause attached
    /// via `startup_error` — instead of one bad AG2 slot aborting the whole run.
    pub async fn maybe_start(config: &Config) -> Self {
        #[cfg(feature = "antigravity")]
        {
            match nitpicker_agent::provider::config_needs_gemini_proxy(config) {
                true => {
                    tracing::info!("Starting Gemini proxy (agy-keyring)");
                    match crate::gemini_proxy::GeminiProxyClient::new().await {
                        Ok(client) => Self {
                            client: Some(client),
                            startup_error: None,
                        },
                        Err(err) => {
                            let rendered = format!("{err:#}");
                            tracing::warn!(
                                error = %rendered,
                                "gemini proxy failed to start; agy-keyring clients will fail individually"
                            );
                            Self {
                                client: None,
                                startup_error: Some(rendered),
                            }
                        }
                    }
                }
                false => Self {
                    client: None,
                    startup_error: None,
                },
            }
        }
        #[cfg(not(feature = "antigravity"))]
        {
            let _ = config;
            Self {
                startup_error: None,
            }
        }
    }

    pub fn url(&self) -> Option<String> {
        #[cfg(feature = "antigravity")]
        {
            self.client.as_ref().map(|client| client.base_url())
        }
        #[cfg(not(feature = "antigravity"))]
        {
            None
        }
    }

    pub fn startup_error(&self) -> Option<&str> {
        self.startup_error.as_deref()
    }

    /// Attaches the proxy's startup failure to a client-construction error. Without it a
    /// client that needed the dead proxy reports only "Gemini proxy required but not
    /// available", losing the actual cause (e.g. an expired keyring token).
    pub fn annotate<T>(&self, result: Result<T>) -> Result<T> {
        match self.startup_error() {
            Some(cause) => result.wrap_err_with(|| format!("gemini proxy startup failed: {cause}")),
            None => result,
        }
    }
}
