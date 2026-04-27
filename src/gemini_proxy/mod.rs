pub mod client;
pub mod oauth;
pub mod proxy;
pub mod retry;
pub mod token;
pub mod transform;

pub use client::{AuthStatus, GeminiProxyClient};

// Google OAuth endpoints for Gemini Code Assist
pub const CODE_ASSIST_BASE_URL: &str = "https://cloudcode-pa.googleapis.com";
pub const OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
pub const OAUTH_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";

// Public OAuth client credentials for Google Code Assist.
// These are intentionally public client IDs/secrets used by Google-installed apps.
// You can override them via environment variables if needed.
pub const CLIENT_ID: &str =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";
pub const CLIENT_SECRET: &str = "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";

pub const SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
];

/// Gemini CLI version synced from upstream.
/// Update with: check `.local/gemini-cli/packages/cli/package.json` or upstream plugin.
pub const GEMINI_CLI_VERSION: &str = "0.38.2";

/// Build a Gemini CLI-style User-Agent string.
/// Honors `NITPICKER_GEMINI_CLI_VERSION` env override.
pub fn build_gemini_user_agent(model: &str) -> String {
    let version = std::env::var("NITPICKER_GEMINI_CLI_VERSION")
        .ok()
        .and_then(|v| {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .unwrap_or_else(|| GEMINI_CLI_VERSION.to_string());
    let platform = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    format!("GeminiCLI/{}/{model} ({platform}; {arch})", version)
}

/// Create a short request-scoped activity id for backend tracing.
/// Mirrors Gemini CLI behavior (short random string, not full UUID).
pub fn create_activity_request_id() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let chars: String = (0..8)
        .map(|_| {
            const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
            CHARSET[rng.random_range(0..CHARSET.len())] as char
        })
        .collect();
    chars
}
