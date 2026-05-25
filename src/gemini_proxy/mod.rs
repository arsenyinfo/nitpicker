pub mod client;
pub mod proxy;
pub mod retry;
pub mod token;
pub mod transform;

pub use client::GeminiProxyClient;

// the AG2 native CLI ships pointing at `daily-cloudcode-pa.googleapis.com` and
// the stable host appears to reject the AG2 client_id at the edge, so daily
// is the default. override via `NITPICKER_CODE_ASSIST_HOST=stable` or a URL.
pub const CODE_ASSIST_BASE_URL_STABLE: &str = "https://cloudcode-pa.googleapis.com";
pub const CODE_ASSIST_BASE_URL_DAILY: &str = "https://daily-cloudcode-pa.googleapis.com";

/// resolve the Code Assist base URL, honoring `NITPICKER_CODE_ASSIST_HOST`
/// (accepts `stable`, `daily`, or a full https URL).
pub fn code_assist_base_url() -> String {
    match std::env::var("NITPICKER_CODE_ASSIST_HOST").ok().as_deref() {
        Some("stable") => CODE_ASSIST_BASE_URL_STABLE.to_string(),
        Some("daily") | None => CODE_ASSIST_BASE_URL_DAILY.to_string(),
        Some(other) => other.to_string(),
    }
}

/// antigravity native CLI version (matches `agy changelog`'s topmost entry).
/// override via `NITPICKER_ANTIGRAVITY_CLI_VERSION`.
pub const ANTIGRAVITY_CLI_VERSION: &str = "1.0.1";

/// build an antigravity-cli User-Agent. the CloudCode backend cross-checks
/// the UA prefix against the OAuth client_id, so this must stay paired with
/// the AG2 client id above.
pub fn build_gemini_user_agent(model: &str) -> String {
    let version = std::env::var("NITPICKER_ANTIGRAVITY_CLI_VERSION")
        .ok()
        .and_then(|v| {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .unwrap_or_else(|| ANTIGRAVITY_CLI_VERSION.to_string());
    let platform = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    format!("antigravity-cli/{version}/{model} ({platform}; {arch})")
}

/// antigravity platform enum. must be one of {DARWIN,LINUX,WINDOWS}_{AMD64,ARM64}.
pub fn antigravity_platform() -> String {
    match std::env::var("NITPICKER_ANTIGRAVITY_PLATFORM") {
        Ok(value) if !value.trim().is_empty() => value.trim().to_string(),
        _ => match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "x86_64") => "DARWIN_AMD64".to_string(),
            ("macos", "aarch64") => "DARWIN_ARM64".to_string(),
            ("linux", "x86_64") => "LINUX_AMD64".to_string(),
            ("linux", "aarch64") => "LINUX_ARM64".to_string(),
            ("windows", "x86_64") => "WINDOWS_AMD64".to_string(),
            ("windows", "aarch64") => "WINDOWS_ARM64".to_string(),
            _ => "DARWIN_ARM64".to_string(),
        },
    }
}

/// create a short request-scoped activity id for backend tracing.
/// mirrors Gemini CLI behavior (short random string, not full UUID).
pub fn create_activity_request_id() -> String {
    use rand::RngExt;
    let mut rng = rand::rng();
    let chars: String = (0..8)
        .map(|_| {
            const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
            CHARSET[rng.random_range(0..CHARSET.len())] as char
        })
        .collect();
    chars
}
