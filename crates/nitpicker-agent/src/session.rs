use eyre::{Result, WrapErr};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct SessionLogger {
    root: Arc<PathBuf>,
    // serializes appends so concurrent subagents sharing a writer don't interleave lines
    write_lock: Arc<Mutex<()>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AggregationRecord {
    pub kind: String,
    pub model: String,
    pub text: String,
    /// Present iff synthesis failed after the per-job/lane work completed (`text` is empty
    /// then): the record still carries `jobs`/`lanes`, which would otherwise die with the
    /// aggregator. Consumers must not render `text` as a verdict when this is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rounds: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub converged: Option<bool>,
    /// Resolved preset names for the run, in order. Absent on pre-preset records and on
    /// runs without preset fan-out (`ask`), so old sessions keep deserializing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presets: Option<Vec<String>>,
    /// Per-lane convergence metadata for preset debate runs; absent elsewhere.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lanes: Option<Vec<LaneRecord>>,
    /// Per-job outcomes for parallel review runs; absent elsewhere. The durable record of
    /// what actually ran — a failed job is otherwise only a transient log line, and a
    /// client-build failure writes no trajectory file at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jobs: Option<Vec<JobRecord>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LaneRecord {
    pub preset: String,
    pub rounds: usize,
    pub converged: bool,
    pub degraded: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JobRecord {
    pub label: String,
    /// Absent on Ask-path jobs, which have no preset dimension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    pub ok: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub ts_unix_ms: u128,
    pub agent: String,
    pub depth: usize,
    pub turn: usize,
    pub tool: String,
    pub args: Value,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spawned_agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Model that produced the turn issuing this call, when the client reports it — the
    /// only durable per-turn attribution on alloy runs, where each turn may pick a
    /// different model. Absent on compaction records and pre-0.3.0 sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl SessionLogger {
    pub fn maybe_new(enabled: bool) -> Result<Option<Self>> {
        if !enabled {
            return Ok(None);
        }

        let home =
            dirs::home_dir().ok_or_else(|| eyre::eyre!("failed to resolve home directory"))?;
        let ts = now_unix_ms();
        let pid = std::process::id();
        let root = home
            .join(".nitpicker")
            .join("sessions")
            .join(format!("session-{ts}-{pid}"));
        std::fs::create_dir_all(&root)
            .wrap_err_with(|| format!("failed to create session dir {}", root.display()))?;
        Ok(Some(Self {
            root: Arc::new(root),
            write_lock: Arc::new(Mutex::new(())),
        }))
    }

    pub fn root(&self) -> &Path {
        self.root.as_ref()
    }

    pub fn child(&self, relative_path: impl AsRef<Path>) -> SessionWriter {
        SessionWriter {
            root: Arc::clone(&self.root),
            relative_path: relative_path.as_ref().to_path_buf(),
            write_lock: Arc::clone(&self.write_lock),
        }
    }

    pub async fn write_aggregation(&self, record: &AggregationRecord) -> Result<()> {
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
    write_lock: Arc<Mutex<()>>,
}

impl SessionWriter {
    pub async fn append_tool_call(&self, record: &ToolCallRecord) -> Result<()> {
        let path = self.root.join(&self.relative_path);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.wrap_err_with(|| {
                format!("failed to create session log dir {}", parent.display())
            })?;
        }
        let mut buf = serde_json::to_vec(record)?;
        buf.push(b'\n');
        // hold the lock across open+write so concurrent appends emit whole lines, not interleaved bytes
        let _guard = self.write_lock.lock().await;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .wrap_err_with(|| format!("failed to open session log {}", path.display()))?;
        file.write_all(&buf).await?;
        // tokio's write_all returns once bytes are handed to the background blocking write;
        // without the flush a real I/O error is dropped with the file handle, and a
        // process::exit right after append can lose the final record
        file.flush().await?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The record must be on disk, whole, by the time `append_tool_call` returns — exit paths
    /// that skip teardown (`process::exit`) rely on it.
    #[tokio::test]
    async fn appended_record_is_durable_and_parseable_on_return() {
        let dir = tempfile::tempdir().unwrap();
        let writer = SessionWriter {
            root: Arc::new(dir.path().to_path_buf()),
            relative_path: PathBuf::from("agent.jsonl"),
            write_lock: Arc::new(Mutex::new(())),
        };
        let record = ToolCallRecord {
            ts_unix_ms: 1,
            agent: "reviewer-1-x".to_string(),
            depth: 0,
            turn: 1,
            tool: "read_file".to_string(),
            args: serde_json::json!({"path": "a.rs"}),
            status: "ok".to_string(),
            spawned_agent: None,
            result: None,
            model: Some("kimi-k2".to_string()),
        };
        writer.append_tool_call(&record).await.unwrap();
        writer.append_tool_call(&record).await.unwrap();

        let content = std::fs::read_to_string(dir.path().join("agent.jsonl")).unwrap();
        let lines: Vec<ToolCallRecord> = content
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].agent, "reviewer-1-x");
    }

    /// `reflect` deserializes historical `aggregation.json` files into this exact type and
    /// discards unparseable ones as incomplete sessions — a legacy record (no presets/lanes
    /// keys) and a new-shape record must both keep parsing.
    #[test]
    fn aggregation_records_parse_across_schema_generations() {
        let legacy = r#"{"kind":"aggregation","model":"m","text":"t","rounds":2,"converged":true}"#;
        let parsed: AggregationRecord = serde_json::from_str(legacy).unwrap();
        assert_eq!(parsed.rounds, Some(2));
        assert!(parsed.error.is_none());
        assert!(parsed.presets.is_none());
        assert!(parsed.lanes.is_none());
        assert!(parsed.jobs.is_none());

        let current = AggregationRecord {
            kind: "aggregation".to_string(),
            model: "m".to_string(),
            text: String::new(),
            error: Some("provider 500".to_string()),
            rounds: None,
            converged: None,
            presets: Some(vec!["security".to_string()]),
            lanes: Some(vec![LaneRecord {
                preset: "security".to_string(),
                rounds: 1,
                converged: true,
                degraded: false,
            }]),
            jobs: Some(vec![JobRecord {
                label: "security · r".to_string(),
                preset: Some("security".to_string()),
                ok: false,
            }]),
        };
        let round_tripped: AggregationRecord =
            serde_json::from_str(&serde_json::to_string(&current).unwrap()).unwrap();
        assert_eq!(round_tripped.error.as_deref(), Some("provider 500"));
        assert_eq!(round_tripped.lanes.unwrap()[0].preset, "security");
        assert!(!round_tripped.jobs.unwrap()[0].ok);

        // absent options serialize to absent keys, keeping old readers indifferent
        let legacy_shaped = AggregationRecord {
            kind: "aggregation".to_string(),
            model: "m".to_string(),
            text: "t".to_string(),
            error: None,
            rounds: Some(1),
            converged: Some(false),
            presets: None,
            lanes: None,
            jobs: None,
        };
        let json = serde_json::to_string(&legacy_shaped).unwrap();
        assert!(!json.contains("error"));
        assert!(!json.contains("presets"));
        assert!(!json.contains("lanes"));
        assert!(!json.contains("jobs"));
    }
}
