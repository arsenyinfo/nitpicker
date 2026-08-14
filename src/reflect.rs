use eyre::Result;
use nitpicker_agent::agent::{AgentConfig, AgentDepth, MAX_CONCURRENT_LLM_CALLS, run_agent};
use nitpicker_agent::config::Config;
use nitpicker_agent::llm::{Completion, LLMClientDyn, throttled_completion};
use nitpicker_agent::provider::build_reviewer_client;
use nitpicker_agent::session::{AggregationRecord, SessionAttribution, ToolCallRecord};
use nitpicker_agent::tools::{floor_char_boundary, reflect_tools};
use rig_core::completion::Message;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tracing::{info, warn};

const MAX_FORMATTED_SESSION_BYTES: usize = 200_000;
const MAX_STAGED_VERDICTS_BYTES: usize = 40_000;
const MAX_TOOL_TRACE_BYTES: usize = 120_000;

const MAP_PROMPT: &str = "\
You are analyzing one recorded nitpicker session. The input contains deterministic metrics,
staged verdicts, the final outcome, and the tool trace.

Assess whether the execution produced a useful outcome and identify avoidable friction in agent
behavior, tool usage, delegation, or coordination. Repetition is not automatically waste:
independent reviewers may intentionally verify the same evidence. Distinguish useful replication
from redundant rediscovery, and distinguish observations from hypotheses.

For every claimed friction point include:
- Evidence: cite concrete records as `[agent turn N: tool]` and relevant deterministic counts.
- Outcome effect: explain whether it affected correctness, convergence, cost, or only presentation.
- Confidence: high, medium, or low.
- Smallest experiment: the least invasive change that could test the hypothesis.

Do not infer a code-level root cause from the trace alone; the cross-session synthesizer can inspect
the implementation. Ignore provider outages, rate limits, and timeouts except when agent behavior
made them worse. Do not use words such as dominant, systemic, nearly all, or widespread in this
single-session report. Include a short `No material friction observed` conclusion when warranted.
Keep the report under 700 words.
---
";

const REDUCE_PROMPT: &str = "\
You are synthesizing evidence from multiple recorded nitpicker sessions. The user will use this
report to decide what to change, so calibration matters more than producing many recommendations.

Use repository tools to verify proposed code-level causes against the current implementation.
Do not convert repeated behavior into an architectural prescription without considering deliberate
reviewer independence, cache behavior, context cost, and false-negative risk. Prefer the smallest
reversible experiment over a broad redesign.

Output these sections:

## Overall synthesis
A short, calibrated assessment.

## Measured patterns
A table with Pattern, Support (exact `N/M sessions`), Representative evidence (session names plus
agent/tool references), Outcome effect, and Confidence. Never say dominant, systemic, nearly all,
or widespread without the corresponding count. Do not combine unlike observations to inflate N.

## Recommended experiments
For each recommendation give: observed behavior and support; current-code cause (`Verified` with
paths/symbols, or `Unverified`); smallest change; tradeoff or regression risk; and a metric that
would show whether it helped. Rank by evidence strength and expected value.

Infrastructure failures are out of scope unless nitpicker behavior amplified them. Separate
correctness problems from token/latency inefficiency and editorial friction. Be concise.
Use uncertain or rejected hypotheses only to calibrate the synthesis internally. Do not print them
or recommendations that lack enough support to act on.
---
";

struct SessionData {
    name: String,
    records: Vec<ToolCallRecord>,
    aggregation: Option<AggregationRecord>,
    attribution: Option<SessionAttribution>,
}

#[derive(Debug, PartialEq, Eq)]
struct RepeatedCall {
    tool: String,
    args: String,
    count: usize,
    agents: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
struct SessionMetrics {
    duration_ms: u128,
    invocations: usize,
    status_counts: BTreeMap<String, usize>,
    tool_counts: BTreeMap<String, usize>,
    model_turns: BTreeMap<String, usize>,
    max_depth: usize,
    repeated_calls: Vec<RepeatedCall>,
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

    fn run_kind(&self) -> &'static str {
        let Some(agg) = &self.aggregation else {
            return "unknown (incomplete)";
        };
        if agg.presets.is_some()
            || agg
                .jobs
                .as_ref()
                .is_some_and(|jobs| jobs.iter().any(|job| job.preset.is_some()))
            || agg.verdicts.iter().any(|verdict| verdict.lens.is_some())
        {
            "preset review"
        } else {
            "unscoped run"
        }
    }

    fn metrics(&self) -> SessionMetrics {
        // A failed spawn has a `started` then `error` lifecycle pair. Count the underlying
        // invocation once while retaining both statuses in status_counts. Every other record is
        // an invocation in its own right: one model turn may issue identical calls concurrently.
        let mut failed_spawn_keys = BTreeSet::new();
        let mut status_counts = BTreeMap::new();
        let mut model_turns = BTreeSet::new();
        let mut min_ts = None;
        let mut max_ts = None;
        let mut max_depth = 0;

        for record in &self.records {
            *status_counts.entry(record.status.clone()).or_insert(0) += 1;
            min_ts = Some(min_ts.map_or(record.ts_unix_ms, |ts: u128| ts.min(record.ts_unix_ms)));
            max_ts = Some(max_ts.map_or(record.ts_unix_ms, |ts: u128| ts.max(record.ts_unix_ms)));
            max_depth = max_depth.max(record.depth);
            if let Some(model) = &record.model {
                model_turns.insert((record.agent.clone(), record.turn, model.clone()));
            }

            if record.tool == "spawn_subagent"
                && record.status == "error"
                && record.spawned_agent.is_some()
            {
                failed_spawn_keys.insert(spawn_lifecycle_key(record));
            }
        }

        let invocations = self.records.iter().filter(|record| {
            !(record.tool == "spawn_subagent"
                && record.status == "started"
                && failed_spawn_keys.contains(&spawn_lifecycle_key(record)))
        });

        let mut tool_counts = BTreeMap::new();
        let mut repeated: BTreeMap<(String, String), (usize, BTreeSet<String>)> = BTreeMap::new();
        let mut invocation_count = 0;
        for record in invocations {
            invocation_count += 1;
            *tool_counts.entry(record.tool.clone()).or_insert(0) += 1;
            let entry = repeated
                .entry((record.tool.clone(), record.args.to_string()))
                .or_insert_with(|| (0, BTreeSet::new()));
            entry.0 += 1;
            entry.1.insert(record.agent.clone());
        }

        let mut repeated_calls: Vec<RepeatedCall> = repeated
            .into_iter()
            .filter(|(_, (count, _))| *count > 1)
            .map(|((tool, args), (count, agents))| RepeatedCall {
                tool,
                args,
                count,
                agents: agents.into_iter().collect(),
            })
            .collect();
        repeated_calls.sort_by(|a, b| {
            b.count
                .cmp(&a.count)
                .then_with(|| a.tool.cmp(&b.tool))
                .then_with(|| a.args.cmp(&b.args))
        });
        repeated_calls.truncate(12);

        let mut model_turn_counts = BTreeMap::new();
        for (_, _, model) in model_turns {
            *model_turn_counts.entry(model).or_insert(0) += 1;
        }

        SessionMetrics {
            duration_ms: max_ts.unwrap_or(0).saturating_sub(min_ts.unwrap_or(0)),
            invocations: invocation_count,
            status_counts,
            tool_counts,
            model_turns: model_turn_counts,
            max_depth,
            repeated_calls,
        }
    }
}

fn spawn_lifecycle_key(record: &ToolCallRecord) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
        record.agent,
        record.turn,
        record.tool,
        record.args,
        record.spawned_agent.as_deref().unwrap_or_default()
    )
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..floor_char_boundary(s, max)])
    }
}

fn truncate_middle(s: &str, max: usize) -> String {
    const MARKER: &str = "\n… middle omitted to preserve both early and late evidence …\n";
    if s.len() <= max {
        return s.to_string();
    }
    if max <= MARKER.len() {
        return truncate(s, max);
    }

    let content_budget = max - MARKER.len();
    let head_end = floor_char_boundary(s, content_budget / 2);
    let mut tail_start = s.len().saturating_sub(content_budget - head_end);
    while tail_start < s.len() && !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    format!("{}{MARKER}{}", &s[..head_end], &s[tail_start..])
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

fn format_counts(counts: &BTreeMap<String, usize>) -> String {
    match counts.is_empty() {
        true => "none".to_string(),
        false => counts
            .iter()
            .map(|(name, count)| format!("{name}={count}"))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

fn verdict_chunk(verdict: &nitpicker_agent::session::VerdictRecord, text_budget: usize) -> String {
    let lens = truncate(verdict.lens.as_deref().unwrap_or("unscoped"), 200);
    let stage = truncate(&verdict.stage, 200);
    let status = if verdict.ok { "ok" } else { "failed" };
    format!(
        "### {lens} · {stage} · {status}\n{}\n",
        truncate(&verdict.text, text_budget)
    )
}

fn final_lane_verdict_indices(agg: &AggregationRecord) -> Vec<usize> {
    let Some(lanes) = &agg.lanes else {
        return Vec::new();
    };

    let mut indices = Vec::new();
    let mut seen = BTreeSet::new();
    for lane in lanes {
        let matches = agg
            .verdicts
            .iter()
            .enumerate()
            .filter(|(_, verdict)| verdict.lens.as_deref() == Some(lane.preset.as_str()))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        for index in matches
            .into_iter()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            if seen.insert(index) {
                indices.push(index);
            }
        }
    }
    indices
}

fn format_staged_verdicts(agg: &AggregationRecord) -> String {
    if agg.verdicts.is_empty() {
        return String::new();
    }

    const HEADER: &str = "## Recorded staged verdicts\n";
    const OMISSION_RESERVE: usize = 128;
    let mut output = HEADER.to_string();
    let mandatory = final_lane_verdict_indices(agg);
    let mandatory_set = mandatory.iter().copied().collect::<BTreeSet<_>>();
    let mut rendered = BTreeSet::new();

    for (position, index) in mandatory.iter().enumerate() {
        let slots_left = mandatory.len() - position;
        let remaining = MAX_STAGED_VERDICTS_BYTES
            .saturating_sub(output.len())
            .saturating_sub(OMISSION_RESERVE);
        let slot_budget = remaining / slots_left.max(1);
        let header_budget = 500;
        let text_budget = slot_budget.saturating_sub(header_budget).min(4_000);
        let chunk = verdict_chunk(&agg.verdicts[*index], text_budget);
        if output.len() + chunk.len() + OMISSION_RESERVE <= MAX_STAGED_VERDICTS_BYTES {
            output.push_str(&chunk);
            rendered.insert(*index);
        }
    }

    for index in (0..agg.verdicts.len()).rev() {
        if mandatory_set.contains(&index) {
            continue;
        }
        let chunk = verdict_chunk(&agg.verdicts[index], 4_000);
        if output.len() + chunk.len() + OMISSION_RESERVE > MAX_STAGED_VERDICTS_BYTES {
            continue;
        }
        output.push_str(&chunk);
        rendered.insert(index);
    }

    let omitted = agg.verdicts.len().saturating_sub(rendered.len());
    if omitted > 0 {
        output.push_str(&format!(
            "… {omitted} additional staged verdict(s) omitted within the section budget\n"
        ));
    }
    truncate(&output, MAX_STAGED_VERDICTS_BYTES)
}

fn format_tool_trace(records: &[ToolCallRecord]) -> String {
    let mut lines = vec!["## Tool call trace".to_string()];
    for r in records {
        let args = truncate(&r.args.to_string(), 4_000);
        let indent = "  ".repeat(r.depth);
        let icon = status_icon(&r.status);
        let model = r
            .model
            .as_deref()
            .map(|model| format!(" [{model}]"))
            .unwrap_or_default();
        lines.push(format!(
            "{indent}{icon} [{}] turn {}{model}: {}({args})",
            r.agent, r.turn, r.tool,
        ));
        if let Some(sp) = &r.spawned_agent {
            lines.push(format!("{indent}  → spawned: {sp}"));
        }
        if let Some(result) = &r.result {
            lines.push(format!("{indent}  → result: {}", truncate(result, 2000)));
        }
    }
    truncate_middle(&lines.join("\n"), MAX_TOOL_TRACE_BYTES)
}

fn format_session(session: &SessionData) -> String {
    let metrics = session.metrics();
    let mut lines = Vec::new();
    lines.push(format!("# Session: {}", session.name));
    lines.push(format!("- Run type: {}", session.run_kind()));
    lines.push(format!("- Status: {}", session.status()));
    lines.push(format!("- Agents: {}", session.agent_names().join(", ")));
    lines.push(format!(
        "- Trace records: {}, error records: {}",
        session.records.len(),
        session.error_count()
    ));

    lines.push(String::new());
    lines.push("## Deterministic trace metrics".to_string());
    lines.push(format!(
        "- Observed tool-trace span: {} ms",
        metrics.duration_ms
    ));
    lines.push(format!(
        "- Unique tool invocations: {}",
        metrics.invocations
    ));
    lines.push(format!(
        "- Record statuses: {}",
        format_counts(&metrics.status_counts)
    ));
    lines.push(format!(
        "- Tool invocations: {}",
        format_counts(&metrics.tool_counts)
    ));
    lines.push(format!("- Maximum subagent depth: {}", metrics.max_depth));
    lines.push(format!(
        "- Model-attributed tool turns: {}",
        format_counts(&metrics.model_turns)
    ));
    if metrics.repeated_calls.is_empty() {
        lines.push("- Repeated exact tool requests: none".to_string());
    } else {
        lines.push("- Repeated exact tool requests (not automatically waste):".to_string());
        for repeated in &metrics.repeated_calls {
            lines.push(format!(
                "  - {}× {}({}); agents: {}",
                repeated.count,
                repeated.tool,
                truncate(&repeated.args, 500),
                repeated.agents.join(", ")
            ));
        }
    }

    lines.push(String::new());
    lines.push("## Experiment attribution".to_string());
    match &session.attribution {
        Some(attribution) => {
            let revision = attribution.binary_revision.as_deref().unwrap_or("unknown");
            lines.push(format!(
                "- Binary: nitpicker {} ({revision})",
                attribution.binary_version
            ));
            lines.push(format!(
                "- Protocol prompt SHA-256: {}",
                attribution.protocol_prompt_sha256
            ));
        }
        None => lines.push("- Unavailable (session predates attribution)".to_string()),
    }

    if let Some(agg) = &session.aggregation {
        lines.push(String::new());
        lines.push("## Run outcome".to_string());
        lines.push(format!("- Aggregation model: {}", agg.model));
        if let Some(presets) = &agg.presets {
            lines.push(format!("- Presets: {}", presets.join(", ")));
        }
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
                lines.push("### Synthesis failure".to_string());
                lines.push(truncate(error, 2_000));
            }
            None => {
                lines.push("### Final verdict".to_string());
                lines.push(truncate(&agg.text, 12_000));
            }
        }
    }

    if let Some(staged) = session
        .aggregation
        .as_ref()
        .map(format_staged_verdicts)
        .filter(|staged| !staged.is_empty())
    {
        lines.push(String::new());
        lines.push(staged);
    }

    lines.push(String::new());
    lines.push(format_tool_trace(&session.records));

    let mut result = lines.join("\n");
    result = truncate_middle(&result, MAX_FORMATTED_SESSION_BYTES);
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
        Some(
            serde_json::from_str::<AggregationRecord>(&content).map_err(|error| {
                eyre::eyre!(
                    "unsupported or malformed aggregation.json in {}: {error}",
                    path.display()
                )
            })?,
        )
    } else {
        None
    };

    let attribution_path = path.join("attribution.json");
    let attribution = if attribution_path.exists() {
        match std::fs::read_to_string(&attribution_path)
            .map_err(eyre::Report::from)
            .and_then(|content| serde_json::from_str(&content).map_err(eyre::Report::from))
        {
            Ok(attribution) => Some(attribution),
            Err(error) => {
                warn!(
                    path = %attribution_path.display(),
                    error = ?error,
                    "ignoring malformed session attribution"
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
        attribution,
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
        max_tokens: Some(1536),
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
    max_tokens: Option<u64>,
    repo: &Path,
) -> Result<String> {
    let session_count = analyses.len();
    let reports = analyses
        .iter()
        .map(|(name, text)| format!("## {name}\n\n{text}"))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    let body = format!(
        "Dataset: {session_count} successfully analyzed session(s). Use this denominator for every support count.\n\n{reports}"
    );

    let tools_map = reflect_tools();
    let config = AgentConfig {
        name: "synthesizer".to_string(),
        session_agent: "synthesizer".to_string(),
        model,
        max_turns: 20,
        max_tokens,
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

    // Reflection reduction is an investigative, tool-using agent rather than the ordinary
    // report-normalization step. Use the critic/reviewer model for it; aggregators are commonly
    // configured as cheaper models suited only to merging already-resolved findings.
    let reduce_reviewer = cfg.reviewer.get(1).unwrap_or(first_reviewer);
    let reduce_model = reduce_reviewer.model.clone();
    let reduce_client = build_reviewer_client(reduce_reviewer, None)?;
    let reduce_max_tokens = reduce_reviewer.max_tokens;

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
        reduce_max_tokens,
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
        std::fs::write(
            dir.path().join("attribution.json"),
            r#"{"binary_version":"0.9.3","binary_revision":"abc123","protocol_prompt_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}"#,
        )
        .unwrap();

        let session = load_session(dir.path()).unwrap();
        let agents: Vec<&str> = session.records.iter().map(|r| r.agent.as_str()).collect();
        assert_eq!(
            agents,
            ["reviewer-2-x", "reviewer-1-x", "reviewer-1-x/subagent-1"]
        );
        let attribution = session.attribution.as_ref().unwrap();
        assert_eq!(attribution.binary_revision.as_deref(), Some("abc123"));
        assert!(!session.is_complete());
    }

    #[test]
    fn load_session_rejects_outdated_aggregation_schema() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("aggregation.json"),
            r#"{"kind":"aggregation","model":"m","text":"old"}"#,
        )
        .unwrap();

        let error = load_session(dir.path())
            .err()
            .expect("outdated schema fails");
        assert!(error.to_string().contains("unsupported or malformed"));
    }

    #[test]
    fn load_session_tolerates_malformed_optional_attribution() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("attribution.json"), "not json").unwrap();

        let session = load_session(dir.path()).unwrap();

        assert!(session.attribution.is_none());
    }

    /// Lane/job metadata and synthesis failures must reach the analysis model — and an
    /// error-flagged record renders its error, never its text slot as a verdict.
    #[test]
    fn format_session_renders_lanes_jobs_and_synthesis_failure() {
        use nitpicker_agent::session::{JobRecord, LaneRecord, SessionAttribution, VerdictRecord};
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
                verdicts: vec![VerdictRecord {
                    lens: Some("security".to_string()),
                    stage: "Reviewer · round 1".to_string(),
                    text: "STAGED-VERDICT".to_string(),
                    ok: true,
                }],
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
            attribution: Some(SessionAttribution {
                binary_version: "0.9.3".to_string(),
                binary_revision: Some("abc123-dirty".to_string()),
                protocol_prompt_sha256: "f".repeat(64),
            }),
        };
        let md = format_session(&session);
        assert!(md.contains("PROVIDER-DIED"));
        assert!(!md.contains("SHOULD-NOT-RENDER"));
        for needle in [
            "security",
            "tone · b",
            "1/2",
            "STAGED-VERDICT",
            "nitpicker 0.9.3 (abc123-dirty)",
            "ffffffffffffffff",
        ] {
            assert!(md.contains(needle), "missing {needle}");
        }
        assert!(!session.is_complete());
    }

    #[test]
    fn metrics_deduplicate_spawn_lifecycle_and_measure_repeated_requests() {
        fn record(
            ts: u128,
            agent: &str,
            turn: usize,
            tool: &str,
            args: serde_json::Value,
            status: &str,
            model: Option<&str>,
        ) -> ToolCallRecord {
            ToolCallRecord {
                ts_unix_ms: ts,
                agent: agent.to_string(),
                depth: usize::from(agent.contains("subagent")),
                turn,
                tool: tool.to_string(),
                args,
                status: status.to_string(),
                spawned_agent: None,
                result: None,
                model: model.map(str::to_string),
            }
        }

        fn spawned(mut record: ToolCallRecord, agent: &str) -> ToolCallRecord {
            record.spawned_agent = Some(agent.to_string());
            record
        }

        let grep_args = serde_json::json!({"pattern": "needle", "path": "src"});
        let spawn_args = serde_json::json!({"task": "inspect"});
        let session = SessionData {
            name: "s".to_string(),
            records: vec![
                record(
                    10,
                    "reviewer",
                    1,
                    "grep",
                    grep_args.clone(),
                    "ok",
                    Some("m"),
                ),
                record(
                    20,
                    "reviewer/subagent-1",
                    1,
                    "grep",
                    grep_args,
                    "ok",
                    Some("m"),
                ),
                spawned(
                    record(
                        30,
                        "reviewer",
                        2,
                        "spawn_subagent",
                        spawn_args.clone(),
                        "started",
                        Some("m"),
                    ),
                    "reviewer/subagent-2",
                ),
                spawned(
                    record(
                        31,
                        "reviewer",
                        2,
                        "spawn_subagent",
                        spawn_args.clone(),
                        "error",
                        Some("m"),
                    ),
                    "reviewer/subagent-2",
                ),
                spawned(
                    record(
                        32,
                        "reviewer",
                        2,
                        "spawn_subagent",
                        spawn_args,
                        "started",
                        Some("m"),
                    ),
                    "reviewer/subagent-3",
                ),
                record(
                    40,
                    "reviewer",
                    3,
                    "read_file",
                    serde_json::json!({"path": "src/main.rs"}),
                    "ok",
                    Some("m"),
                ),
                record(
                    41,
                    "reviewer",
                    3,
                    "read_file",
                    serde_json::json!({"path": "src/main.rs"}),
                    "ok",
                    Some("m"),
                ),
            ],
            aggregation: None,
            attribution: None,
        };

        let metrics = session.metrics();
        assert_eq!(metrics.duration_ms, 31);
        assert_eq!(metrics.invocations, 6);
        assert_eq!(metrics.tool_counts.get("grep"), Some(&2));
        assert_eq!(metrics.tool_counts.get("spawn_subagent"), Some(&2));
        assert_eq!(metrics.tool_counts.get("read_file"), Some(&2));
        assert_eq!(metrics.status_counts.get("started"), Some(&2));
        assert_eq!(metrics.status_counts.get("error"), Some(&1));
        assert_eq!(metrics.model_turns.get("m"), Some(&4));
        assert_eq!(metrics.max_depth, 1);
        assert_eq!(metrics.repeated_calls.len(), 3);
        assert_eq!(metrics.repeated_calls[0].tool, "grep");
        assert_eq!(metrics.repeated_calls[0].count, 2);
        assert_eq!(metrics.repeated_calls[1].tool, "read_file");
        assert_eq!(metrics.repeated_calls[1].count, 2);
        assert_eq!(metrics.repeated_calls[2].tool, "spawn_subagent");
        assert_eq!(metrics.repeated_calls[2].count, 2);
    }

    #[test]
    fn staged_verdicts_and_late_trace_evidence_survive_independent_budgets() {
        use nitpicker_agent::session::{LaneRecord, VerdictRecord};

        let lenses = ["correctness", "security", "performance", "simplicity"];
        let mut verdicts = Vec::new();
        for lens in lenses {
            for round in 1..=5 {
                for role in ["review", "validate"] {
                    let marker = if round == 5 {
                        format!("FINAL-{lens}-{role}")
                    } else {
                        format!("OLDER-{lens}-{round}-{role}")
                    };
                    verdicts.push(VerdictRecord {
                        lens: Some(lens.to_string()),
                        stage: format!("{role} · round {round}"),
                        text: format!("{marker}-{}", "v".repeat(6_000)),
                        ok: true,
                    });
                }
            }
        }
        let lanes = lenses
            .iter()
            .map(|lens| LaneRecord {
                preset: (*lens).to_string(),
                rounds: 5,
                converged: false,
                degraded: false,
            })
            .collect();
        let records = (0..60)
            .map(|turn| ToolCallRecord {
                ts_unix_ms: turn as u128,
                agent: "reviewer".to_string(),
                depth: 0,
                turn,
                tool: "read_file".to_string(),
                args: serde_json::json!({"path": format!("{turn}-{}", "p".repeat(5_000))}),
                status: "ok".to_string(),
                spawned_agent: None,
                result: Some(if turn == 59 {
                    format!("LATE-TRACE-EVIDENCE-{}", "r".repeat(3_000))
                } else {
                    "r".repeat(3_000)
                }),
                model: Some("m".to_string()),
            })
            .collect();
        let session = SessionData {
            name: "large".to_string(),
            records,
            aggregation: Some(AggregationRecord {
                kind: "aggregation".to_string(),
                model: "m".to_string(),
                text: "final".to_string(),
                error: None,
                rounds: None,
                converged: None,
                presets: Some(lenses.iter().map(|lens| (*lens).to_string()).collect()),
                lanes: Some(lanes),
                verdicts,
                jobs: None,
            }),
            attribution: None,
        };

        let staged = format_staged_verdicts(session.aggregation.as_ref().unwrap());
        assert!(staged.len() <= MAX_STAGED_VERDICTS_BYTES);
        let formatted = format_session(&session);
        assert!(formatted.len() <= MAX_FORMATTED_SESSION_BYTES);
        for lens in lenses {
            assert!(formatted.contains(&format!("FINAL-{lens}-review")));
            assert!(formatted.contains(&format!("FINAL-{lens}-validate")));
        }
        assert!(formatted.contains("LATE-TRACE-EVIDENCE"));
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
