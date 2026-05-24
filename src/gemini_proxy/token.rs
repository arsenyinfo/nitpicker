use chrono::{DateTime, Utc};
use eyre::{Context, Result};
use serde::Deserialize;

const AGY_KEYRING_SERVICE: &str = "gemini";
const AGY_KEYRING_ACCOUNT: &str = "antigravity";

#[derive(Debug, Clone)]
pub struct TokenData {
    pub access_token: String,
    pub expires_at: DateTime<Utc>,
}

impl TokenData {
    pub fn is_expired(&self) -> bool {
        Utc::now() + chrono::Duration::seconds(60) >= self.expires_at
    }
}

#[derive(Clone)]
pub struct TokenStore;

impl TokenStore {
    pub fn new_agy_keyring() -> Self {
        Self
    }

    pub fn load(&self) -> Result<Option<TokenData>> {
        load_agy_keyring_token()
    }
}

#[derive(Debug, Deserialize)]
struct AgyKeyringPayload {
    token: AgyKeyringToken,
}

#[derive(Debug, Deserialize)]
struct AgyKeyringToken {
    access_token: String,
    expiry: DateTime<Utc>,
}

fn load_agy_keyring_token() -> Result<Option<TokenData>> {
    let entry = keyring::Entry::new(AGY_KEYRING_SERVICE, AGY_KEYRING_ACCOUNT)
        .wrap_err("failed to initialize Antigravity keyring entry")?;

    let raw = match entry.get_password() {
        Ok(password) => password,
        Err(keyring::Error::NoEntry) => return Ok(None),
        Err(err) => return Err(err).wrap_err("failed to read Antigravity keyring token"),
    };

    parse_agy_token_payload(&raw).map(Some)
}

fn parse_agy_token_payload(raw: &str) -> Result<TokenData> {
    let encoded = raw
        .strip_prefix("go-keyring-base64:")
        .map(str::as_bytes)
        .map(|encoded| base64::Engine::decode(&base64::engine::general_purpose::STANDARD, encoded))
        .transpose()?
        .unwrap_or_else(|| raw.as_bytes().to_vec());
    let payload: AgyKeyringPayload = serde_json::from_slice(&encoded)?;

    Ok(TokenData {
        access_token: payload.token.access_token,
        expires_at: payload.token.expiry,
    })
}
