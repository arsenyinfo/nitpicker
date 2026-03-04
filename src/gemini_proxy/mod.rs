pub mod client;
pub mod oauth;
pub mod proxy;
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
