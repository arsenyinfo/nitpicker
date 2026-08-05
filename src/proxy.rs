use eyre::Result;
use nitpicker_agent::config::Config;

/// Owns the optional Gemini AG2 proxy for a run. The inner client's drop shuts the local
/// server down, so the handle must stay bound for as long as any client may call through it —
/// only the base URL is threaded downstream.
pub struct GeminiProxy {
    #[cfg(feature = "antigravity")]
    client: Option<crate::gemini_proxy::GeminiProxyClient>,
}

impl GeminiProxy {
    /// Starts the proxy when any configured reviewer/aggregator uses `auth = "agy-keyring"`
    /// (feature `antigravity`); otherwise returns an inert handle.
    pub async fn maybe_start(config: &Config) -> Result<Self> {
        #[cfg(feature = "antigravity")]
        {
            let client = match nitpicker_agent::provider::config_needs_gemini_proxy(config) {
                true => {
                    tracing::info!("Starting Gemini proxy (agy-keyring)");
                    Some(crate::gemini_proxy::GeminiProxyClient::new().await?)
                }
                false => None,
            };
            Ok(Self { client })
        }
        #[cfg(not(feature = "antigravity"))]
        {
            let _ = config;
            Ok(Self {})
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
}
