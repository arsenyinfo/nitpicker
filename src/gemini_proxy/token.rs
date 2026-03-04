use chrono::{DateTime, Utc};
use eyre::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    path: PathBuf,
}

impl TokenStore {
    pub fn new() -> Result<Self> {
        let home_dir =
            dirs::home_dir().ok_or_else(|| eyre::eyre!("Could not find home directory"))?;
        let app_dir = home_dir.join(".nitpicker");
        std::fs::create_dir_all(&app_dir)?;

        let path = app_dir.join("gemini-token.json");
        Ok(Self { path })
    }

    pub fn load(&self) -> Result<Option<TokenData>> {
        if !self.path.exists() {
            return Ok(None);
        }

        let contents = std::fs::read_to_string(&self.path)?;
        let token: TokenData = serde_json::from_str(&contents)?;
        Ok(Some(token))
    }

    pub fn save(&self, token: &TokenData) -> Result<()> {
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
                .open(&self.path)?;
            file.write_all(contents.as_bytes())?;
        }
        #[cfg(not(unix))]
        std::fs::write(&self.path, &contents)?;
        Ok(())
    }
}
