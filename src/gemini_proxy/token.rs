use chrono::{DateTime, Utc};
use eyre::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const AGY_KEYRING_SERVICE: &str = "gemini";
const AGY_KEYRING_ACCOUNT: &str = "antigravity";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenData {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: DateTime<Utc>,
    pub token_type: String,
}

impl TokenData {
    pub fn is_expired(&self) -> bool {
        Utc::now() + chrono::Duration::seconds(60) >= self.expires_at
    }

    pub fn is_refreshable(&self) -> bool {
        self.refresh_token.is_some()
    }
}

#[derive(Clone)]
pub struct TokenStore {
    source: TokenSource,
}

#[derive(Clone)]
enum TokenSource {
    File(PathBuf),
    AgyKeyring,
}

impl TokenStore {
    pub fn new() -> Result<Self> {
        let home_dir =
            dirs::home_dir().ok_or_else(|| eyre::eyre!("Could not find home directory"))?;
        let app_dir = home_dir.join(".nitpicker");
        std::fs::create_dir_all(&app_dir)?;

        let path = app_dir.join("gemini-token.json");
        Ok(Self {
            source: TokenSource::File(path),
        })
    }

    pub fn new_agy_keyring() -> Self {
        Self {
            source: TokenSource::AgyKeyring,
        }
    }

    pub fn refresh_managed_externally(&self) -> bool {
        matches!(self.source, TokenSource::AgyKeyring)
    }

    pub fn source_name(&self) -> &'static str {
        match &self.source {
            TokenSource::File(_) => "nitpicker token file",
            TokenSource::AgyKeyring => "Antigravity keyring token",
        }
    }

    pub fn load(&self) -> Result<Option<TokenData>> {
        match &self.source {
            TokenSource::File(path) => {
                if !path.exists() {
                    return Ok(None);
                }

                let contents = std::fs::read_to_string(path)?;
                let token: TokenData = serde_json::from_str(&contents)?;
                Ok(Some(token))
            }
            TokenSource::AgyKeyring => load_agy_keyring_token(),
        }
    }

    pub fn save(&self, token: &TokenData) -> Result<()> {
        let path = match &self.source {
            TokenSource::File(path) => path,
            TokenSource::AgyKeyring => {
                eyre::bail!("cannot write Antigravity keyring token from nitpicker")
            }
        };
        let contents = serde_json::to_string_pretty(token)?;
        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(path)?;
            file.write_all(contents.as_bytes())?;
        }
        #[cfg(not(unix))]
        std::fs::write(path, &contents)?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct AgyKeyringPayload {
    token: AgyKeyringToken,
}

#[derive(Debug, Deserialize)]
struct AgyKeyringToken {
    access_token: String,
    refresh_token: Option<String>,
    token_type: String,
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
        refresh_token: payload.token.refresh_token,
        expires_at: payload.token.expiry,
        token_type: payload.token.token_type,
    })
}
