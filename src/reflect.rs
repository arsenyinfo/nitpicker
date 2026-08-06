use eyre::Result;
use nitpicker_agent::agent::{AgentConfig, AgentDepth, MAX_CONCURRENT_LLM_CALLS, run_agent};
use nitpicker_agent::config::Config;
use nitpicker_agent::llm::{Completion, LLMClientDyn, throttled_completion};
use nitpicker_agent::provider::build_reviewer_client;
use nitpicker_agent::session::{AggregationRecord, ToolCallRecord};
use nitpicker_agent::tools::{floor_char_boundary, reflect_tools};
use rig_core::completion::Message;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tracing::{info, warn};

const MAP_PROMPT: &str = "\
You are analyzing a nitpicker code review session — a multi-agent LLM-based code reviewer.
Each session records all tool calls (file reads, git diffs, greps, subagent spawns) made by reviewer agents.

Analyze this session and produce a concise report covering the friction points observed. Focus in the issues related to agents behavior such as confusing errors, ineffective tool usage and delegation patterns, and overall poor execution. Reference specific tool calls or patterns where possible. Ignore infrastucture level issues (e.g. rate limits, timeouts) and focus on aspect that could be improved through better prompt design, tool configuration, or agent coordination. The report should be actionable and specific, ideally with examples from the session. Keep it under 600 words.
---
";

const REDUCE_PROMPT: &str = "\
You are synthesizing analysis of multiple nitpicker review sessions — a multi-agent LLM-based code reviewer.

Based on the reports for individual sessions, synthesize an overall summary of common friction points and potential improvements across sessions. Identify patterns in agent behavior, tool usage, and execution that lead to issues.

Review the nitpicker code (tools, prompts) to cross-reference the identified issues and suggest specific improvements to the system. Focus on actionable changes that could be made to prompts, tool configurations, or agent coordination to address the observed friction points. The summary should be concise and focused on key insights and recommendations for improving nitpicker's performance as a code reviewer.
---
";

struct SessionData {
    name: String,
    records: Vec<ToolCallRecord>,
    aggregation: Option<AggregationRecord>,
}

impl SessionData {
    /// A session is complete when synthesis produced a verdict — an error-flagged
    /// aggregation record means the run reached synthesis and died there.
    fn is_complete(&self) -> bool {
        match &self.aggregation {
            Some(agg) => agg.error.is_none(),
            None => false,
        }
    }

    fn status(&self) -> &'static str {
        match &self.aggregation {
            Some(agg) => match agg.error {
                Some(_) => "synthesis failed",
                None => "complete",
            },
            None => "incomplete",
        }
    }

    fn error_count(&self) -> usize {
        self.records.iter().filter(|r| r.status == "error").count()
    }

    fn agent_names(&self) -> Vec<&str> {
        let mut seen: Vec<&str> = Vec::new();
        for r in &self.records {
            if !seen.contains(&r.agent.as_str()) {
                seen.push(&r.agent);
            }
        }
        seen
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..floor_char_boundary(s, max)])
    }
}

/// Icon per trajectory-log status (`agent.rs::ToolCallStatus::as_str`). Every status needs its
/// own glyph: rendering "started" (a spawn whose result lives in the subagent's own records) or
/// "blocked_cycle" as ✗ teaches the analysis model that those calls failed. Unknown vocabulary
/// gets its own glyph too — aliasing a future status to ✗ would silently recreate that bug.
fn status_icon(status: &str) -> &'static str {
    match status {
        "ok" => "✓",
        "error" => "✗",
        "started" => "▶",
        "blocked_cycle" => "⊘",
        _ => "?",
    }
}

fn format_session(session: &SessionData) -> String {
    let mut lines = Vec::new();
    lines.push(format!("# Session: {}", session.name));
    lines.push(format!("- Status: {}", session.status()));
    lines.push(format!("- Agents: {}", session.agent_names().join(", ")));
    // records, not calls: a failed spawn contributes a started/error lifecycle pair
    lines.push(format!(
        "- Records: {}, errors: {}",
        session.records.len(),
        session.error_count()
    ));
    if let Some(agg) = &session.aggregation {
        lines.push(format!("- Aggregation model: {}", agg.model));
        if let Some(rounds) = agg.rounds {
            lines.push(format!("- Debate rounds: {rounds}"));
        }
        if let Some(converged) = agg.converged {
            lines.push(format!("- Converged early: {converged}"));
        }
        if let Some(lanes) = &agg.lanes {
            for lane in lanes {
                lines.push(format!(
                    "- Lane {}: {} round(s), converged: {}, degraded: {}",
                    lane.preset, lane.rounds, lane.converged, lane.degraded
                ));
            }
        }
        if let Some(jobs) = &agg.jobs {
            let ok = jobs.iter().filter(|j| j.ok).count();
            lines.push(format!("- Jobs: {ok}/{} succeeded", jobs.len()));
            for job in jobs.iter().filter(|j| !j.ok) {
                lines.push(format!("  - failed: {}", job.label));
            }
        }
        lines.push(String::new());
        match &agg.error {
            Some(error) => {
                lines.push("## Synthesis failure".to_string());
                lines.push(truncate(error, 600));
            }
            None => {
                lines.push("## Verdict summary".to_string());
                lines.push(truncate(&agg.text, 600));
            }
        }
    }

    lines.push(String::new());
    lines.push("## Tool call trace".to_string());

    for r in &session.records {
        let args = truncate(&r.args.to_string(), 20000);
        let indent = "  ".repeat(r.depth);
        let icon = status_icon(&r.status);
        lines.push(format!(
            "{indent}{icon} [{}] turn {}: {}({args})",
            r.agent, r.turn, r.tool
        ));
        if let Some(sp) = &r.spawned_agent {
            lines.push(format!("{indent}  → spawned: {sp}"));
        }
        if let Some(result) = &r.result {
            lines.push(format!("{indent}  → result: {}", truncate(result, 2000)));
        }
    }

    let mut result = lines.join("\n");
    result = truncate(&result, 200_000);
    result
}

fn load_session(path: &Path) -> Result<SessionData> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let mut jsonl_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect();
    jsonl_files.sort();

    let mut records: Vec<ToolCallRecord> = Vec::new();
    for file in &jsonl_files {
        let content = std::fs::read_to_string(file)?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<ToolCallRecord>(line) {
                Ok(r) => records.push(r),
                Err(e) => warn!("skipping malformed line in {}: {e}", file.display()),
            }
        }
    }
    records.sort_by_key(|r| r.ts_unix_ms);

    let agg_path = path.join("aggregation.json");
    let aggregation = if agg_path.exists() {
        let content = std::fs::read_to_string(&agg_path)?;
        match serde_json::from_str::<AggregationRecord>(&content) {
            Ok(agg) => Some(agg),
            Err(e) => {
                warn!(
                    "failed to parse aggregation.json in {}: {e}",
                    path.display()
                );
                None
            }
        }
    } else {
        None
    };

    Ok(SessionData {
        name,
        records,
        aggregation,
    })
}

fn discover_sessions(dir: &Path, n: usize) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        eyre::bail!(
            "sessions directory not found: {} — run nitpicker with log_trajectories = true first",
            dir.display()
        );
    }
    let mut sessions: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("session-"))
                    .unwrap_or(false)
        })
        .collect();
    sessions.sort_by(|a, b| b.cmp(a)); // newest first
    sessions.truncate(n);
    Ok(sessions)
}

async fn analyze_session(
    session_md: String,
    model: String,
    client: Arc<dyn LLMClientDyn>,
    semaphore: Arc<Semaphore>,
) -> Result<String> {
    let completion = Completion {
        model,
        // MAP_PROMPT goes in the system preamble (not folded into the user message): codex auth
        // requires a top-level system prompt, and this matches every other call site's shape.
        prompt: Message::user(session_md),
        preamble: Some(MAP_PROMPT.to_string()),
        history: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        max_tokens: Some(1024),
        additional_params: None,
    };
    Ok(throttled_completion(&semaphore, &client, completion)
        .await?
        .text())
}

async fn synthesize(
    analyses: Vec<(String, String)>,
    model: String,
    client: Arc<dyn LLMClientDyn>,
    repo: &Path,
) -> Result<String> {
    let body = analyses
        .iter()
        .map(|(name, text)| format!("## {name}\n\n{text}"))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let tools_map = reflect_tools();
    let config = AgentConfig {
        name: "synthesizer".to_string(),
        session_agent: "synthesizer".to_string(),
        model,
        max_turns: 20,
        max_tokens: None,
        compact_threshold: None,
        system_prompt: REDUCE_PROMPT.to_string(),
        subagent_system_prompt: None,
        client,
        depth: AgentDepth::TopLevel,
        terminal_tools: vec![],
        empty_response_nudge: None,
        max_empty_responses: 3,
        subagent_counter: Arc::new(AtomicUsize::new(0)),
        llm_semaphore: Arc::new(tokio::sync::Semaphore::new(
            nitpicker_agent::agent::MAX_CONCURRENT_LLM_CALLS,
        )),
        progress: None,
        project_context: None,
        session_writer: None,
    };

    let result = run_agent(config, &body, &tools_map, repo).await?;

    Ok(result.text)
}

pub struct ReflectArgs {
    pub sessions_dir: Option<PathBuf>,
    pub n: usize,
    pub repo: PathBuf,
    pub config: Config,
}

pub async fn run_reflect(args: ReflectArgs) -> Result<()> {
    let dir = match args.sessions_dir {
        Some(d) => d,
        None => {
            let home =
                dirs::home_dir().ok_or_else(|| eyre::eyre!("failed to resolve home directory"))?;
            home.join(".nitpicker").join("sessions")
        }
    };
    let session_paths: Vec<PathBuf> = discover_sessions(&dir, args.n)?;

    if session_paths.is_empty() {
        eyre::bail!("no sessions found");
    }

    info!("loading {} sessions…", session_paths.len());

    let sessions: Vec<SessionData> = session_paths
        .iter()
        .filter_map(|p| match load_session(p) {
            Ok(s) => Some(s),
            Err(e) => {
                warn!("skipping {}: {e}", p.display());
                None
            }
        })
        .collect();

    let tmp_dir = PathBuf::from("/tmp/nitpicker-sessions");
    match std::fs::create_dir_all(&tmp_dir) {
        Ok(()) => {
            for session in &sessions {
                let md = format_session(session);
                let out = tmp_dir.join(format!("{}.md", session.name));
                match std::fs::write(&out, &md) {
                    Ok(()) => info!("saved formatted session to {}", out.display()),
                    Err(e) => warn!("failed to save formatted session to {}: {e}", out.display()),
                }
            }
        }
        Err(e) => warn!(
            "failed to create session dump directory {}: {e}",
            tmp_dir.display()
        ),
    }

    if sessions.is_empty() {
        eyre::bail!("no sessions could be loaded");
    }

    let complete = sessions.iter().filter(|s| s.is_complete()).count();
    info!(
        "{} sessions loaded ({complete} complete, {} incomplete)",
        sessions.len(),
        sessions.len() - complete
    );

    let cfg = &args.config;

    let first_reviewer = cfg
        .reviewer
        .first()
        .ok_or_else(|| eyre::eyre!("config must have at least one reviewer"))?;
    let map_model = first_reviewer.model.clone();
    let map_client = build_reviewer_client(first_reviewer, None)?;

    let (reduce_model, reduce_client) = if cfg.reviewer.len() >= 2 {
        let reviewer = &cfg.reviewer[1];
        let client = build_reviewer_client(reviewer, None)?;
        (reviewer.model.clone(), client)
    } else {
        let client = build_reviewer_client(first_reviewer, None)?;
        (first_reviewer.model.clone(), client)
    };

    info!("analyzing sessions with {}…", map_model);
    // bound concurrent in-flight session-analysis calls the same way the review path does
    let map_semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_LLM_CALLS));
    let mut handles: Vec<(String, JoinHandle<Result<String>>)> = Vec::new();
    for session in &sessions {
        let md = format_session(session);
        let name = session.name.clone();
        let model = map_model.clone();
        let client = Arc::clone(&map_client);
        let semaphore = Arc::clone(&map_semaphore);
        let handle =
            tokio::spawn(async move { analyze_session(md, model, client, semaphore).await });
        handles.push((name, handle));
    }

    let mut analyses: Vec<(String, String)> = Vec::with_capacity(handles.len());
    for (name, handle) in handles {
        match handle.await? {
            Ok(text) => analyses.push((name, text)),
            Err(e) => warn!("analysis failed for session: {e}"),
        }
    }

    if analyses.is_empty() {
        eyre::bail!("all session analyses failed");
    }

    info!("synthesizing with {}…", reduce_model);
    let report = synthesize(
        analyses,
        reduce_model.clone(),
        Arc::clone(&reduce_client),
        &args.repo,
    )
    .await?;

    println!("{report}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// load_session pools every file in the session by timestamp and discards filenames — the
    /// record's `agent` label must survive the merge verbatim, since it is the only thing left
    /// attributing interleaved records to their agents.
    #[test]
    fn load_session_merges_files_by_timestamp_preserving_agent_labels() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("reviewer-1-x.jsonl"),
            concat!(
                r#"{"ts_unix_ms":2,"agent":"reviewer-1-x","depth":0,"turn":1,"tool":"spawn_subagent","args":{},"status":"started","spawned_agent":"reviewer-1-x/subagent-1"}"#,
                "\n",
                r#"{"ts_unix_ms":3,"agent":"reviewer-1-x/subagent-1","depth":1,"turn":1,"tool":"grep","args":{},"status":"ok"}"#,
                "\n",
            ),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("reviewer-2-x.jsonl"),
            concat!(
                r#"{"ts_unix_ms":1,"agent":"reviewer-2-x","depth":0,"turn":1,"tool":"read_file","args":{},"status":"ok"}"#,
                "\n",
            ),
        )
        .unwrap();

        let session = load_session(dir.path()).unwrap();
        let agents: Vec<&str> = session.records.iter().map(|r| r.agent.as_str()).collect();
        assert_eq!(
            agents,
            ["reviewer-2-x", "reviewer-1-x", "reviewer-1-x/subagent-1"]
        );
        assert!(!session.is_complete());
    }

    /// Lane/job metadata and synthesis failures must reach the analysis model — and an
    /// error-flagged record renders its error, never its text slot as a verdict.
    #[test]
    fn format_session_renders_lanes_jobs_and_synthesis_failure() {
        use nitpicker_agent::session::{JobRecord, LaneRecord};
        let session = SessionData {
            name: "s".to_string(),
            records: Vec::new(),
            aggregation: Some(AggregationRecord {
                kind: "aggregation".to_string(),
                model: "m".to_string(),
                text: "SHOULD-NOT-RENDER".to_string(),
                error: Some("PROVIDER-DIED".to_string()),
                rounds: None,
                converged: None,
                presets: None,
                lanes: Some(vec![LaneRecord {
                    preset: "security".to_string(),
                    rounds: 2,
                    converged: false,
                    degraded: true,
                }]),
                jobs: Some(vec![
                    JobRecord {
                        label: "security · a".to_string(),
                        preset: Some("security".to_string()),
                        ok: true,
                    },
                    JobRecord {
                        label: "tone · b".to_string(),
                        preset: Some("tone".to_string()),
                        ok: false,
                    },
                ]),
            }),
        };
        let md = format_session(&session);
        assert!(md.contains("PROVIDER-DIED"));
        assert!(!md.contains("SHOULD-NOT-RENDER"));
        for needle in ["security", "tone · b", "1/2"] {
            assert!(md.contains(needle), "missing {needle}");
        }
        assert!(!session.is_complete());
    }

    #[test]
    fn each_trajectory_status_classifies_distinctly() {
        // the four known statuses plus the unknown fallback must all stay distinguishable —
        // in particular an unrecognized future status must not masquerade as a failure
        let statuses = ["ok", "started", "blocked_cycle", "error", "unknown"];
        for (i, a) in statuses.iter().enumerate() {
            for b in &statuses[i + 1..] {
                assert_ne!(status_icon(a), status_icon(b), "{a} vs {b}");
            }
        }
    }
}
