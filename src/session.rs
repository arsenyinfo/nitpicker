use eyre::{Result, WrapErr};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

#[derive(Clone)]
pub struct SessionLogger {
    root: Arc<PathBuf>,
}

#[derive(Serialize)]
pub struct AggregationRecord<'a> {
    pub kind: &'a str,
    pub model: &'a str,
    pub text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rounds: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub converged: Option<bool>,
}

#[derive(Serialize)]
pub struct ToolCallRecord<'a> {
    pub ts_unix_ms: u128,
    pub agent: &'a str,
    pub depth: usize,
    pub turn: usize,
    pub tool: &'a str,
    pub args: &'a Value,
    pub status: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spawned_agent: Option<&'a str>,
}

impl SessionLogger {
    pub fn maybe_new(enabled: bool) -> Result<Option<Self>> {
        if !enabled {
            return Ok(None);
        }

        let home = dirs::home_dir().ok_or_else(|| eyre::eyre!("failed to resolve home directory"))?;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let pid = std::process::id();
        let root = home
            .join(".nitpicker")
            .join("sessions")
            .join(format!("session-{ts}-{pid}"));
        std::fs::create_dir_all(&root)
            .wrap_err_with(|| format!("failed to create session dir {}", root.display()))?;
        Ok(Some(Self {
            root: Arc::new(root),
        }))
    }

    pub fn root(&self) -> &Path {
        self.root.as_ref()
    }

    pub fn child(&self, relative_path: impl AsRef<Path>) -> SessionWriter {
        SessionWriter {
            root: Arc::clone(&self.root),
            relative_path: relative_path.as_ref().to_path_buf(),
        }
    }

    pub async fn write_aggregation(&self, record: &AggregationRecord<'_>) -> Result<()> {
        let path = self.root.join("aggregation.json");
        let body = serde_json::to_vec_pretty(record)?;
        tokio::fs::write(&path, body)
            .await
            .wrap_err_with(|| format!("failed to write aggregation log {}", path.display()))?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct SessionWriter {
    root: Arc<PathBuf>,
    relative_path: PathBuf,
}

impl SessionWriter {
    pub async fn append_tool_call(&self, record: &ToolCallRecord<'_>) -> Result<()> {
        let path = self.root.join(&self.relative_path);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .wrap_err_with(|| format!("failed to create session log dir {}", parent.display()))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .wrap_err_with(|| format!("failed to open session log {}", path.display()))?;
        let line = serde_json::to_string(record)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        Ok(())
    }
}

pub fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

pub fn sanitize_path_component(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '-',
        })
        .collect();
    let trimmed = sanitized.trim_matches('-');
    if trimmed.is_empty() {
        "agent".to_string()
    } else {
        trimmed.to_string()
    }
}
