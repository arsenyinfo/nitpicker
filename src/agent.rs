use crate::compact::compact_history;
use crate::llm::{Completion, ConversationUsageWindow, LLMClientDyn};
use crate::prompts::subagent_system_prompt;
use crate::tools::{Tool, floor_char_boundary, tool_definitions};
use eyre::Result;
use rig::OneOrMany;
use rig::completion::Message;
use rig::completion::message::{ToolResult, ToolResultContent, UserContent};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

const MAX_TOOL_RESULT_BYTES: usize = 50_000;
const MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS: usize = 3;
const MAX_CYCLE_REPETITIONS: usize = 2;
const TOOL_CALL_HISTORY_WINDOW: usize = 8;
const MAX_SUBAGENT_DEPTH: usize = 2;
const DEFAULT_AGENT_TEMPERATURE: f64 = 0.2;
const REPEATED_TOOL_CALL_RECOVERY_TEMPERATURE: f64 = 0.6;

pub struct AgentResult {
    pub text: String,
    pub turns: usize,
    pub tool_calls: usize,
    pub subagents_spawned: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_tokens: u64,
}

pub struct AgentProgress {
    pub turns: usize,
    pub tool_calls: usize,
    pub subagents_spawned: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentDepth {
    TopLevel,
    Subagent { level: usize },
}

impl AgentDepth {
    fn level(self) -> usize {
        match self {
            AgentDepth::TopLevel => 0,
            AgentDepth::Subagent { level } => level,
        }
    }

    fn is_subagent(self) -> bool {
        matches!(self, AgentDepth::Subagent { .. })
    }

    fn can_spawn_subagent(self) -> bool {
        self.level() < MAX_SUBAGENT_DEPTH
    }
}

pub struct AgentConfig {
    pub name: String,
    pub model: String,
    pub max_turns: usize,
    pub compact_threshold: Option<u64>,
    pub system_prompt: String,
    pub client: Arc<dyn LLMClientDyn>,
    pub depth: AgentDepth,
    pub terminal_tools: Vec<String>,
    pub empty_response_nudge: Option<String>,
    pub max_empty_responses: usize,
    pub subagent_counter: Arc<AtomicUsize>,
    pub progress: Option<Arc<dyn Fn(AgentProgress) + Send + Sync>>,
}

struct FinishTool {
    result: Arc<Mutex<Option<String>>>,
}

struct ToolCallContext<'a> {
    config: &'a AgentConfig,
    runtime_tools: &'a HashMap<String, Arc<dyn Tool>>,
    tools_map: &'a HashMap<String, Arc<dyn Tool>>,
    work_dir: &'a Path,
    turn: usize,
    current_turns: usize,
    total_tool_calls: usize,
    initial_subagent_count: usize,
}

struct ToolCallOutcome {
    output: String,
    nested_tool_calls: usize,
    repeated_tool_call_blocked: bool,
}

impl Tool for FinishTool {
    fn name(&self) -> String {
        "finish".to_string()
    }

    fn definition(&self) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: "finish".to_string(),
            description:
                "Finish the assigned subtask and return the final result to the parent agent."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "Concise final result for the parent agent"
                    }
                },
                "required": ["result"],
                "additionalProperties": false
            }),
        }
    }

    fn call(
        &self,
        args: Value,
        _work_dir: PathBuf,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>> {
        let result_store = Arc::clone(&self.result);
        Box::pin(async move {
            let result = args
                .get("result")
                .and_then(|value| value.as_str())
                .ok_or_else(|| eyre::eyre!("missing result"))?
                .to_string();
            *result_store.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
            Ok("ok".to_string())
        })
    }
}

pub async fn run_agent(
    config: AgentConfig,
    initial_message: &str,
    tools_map: &HashMap<String, Arc<dyn Tool>>,
    work_dir: &Path,
) -> Result<AgentResult> {
    let finish_store = Arc::new(Mutex::new(None));
    let mut runtime_tools = tools_map.clone();
    if !config.depth.can_spawn_subagent() {
        runtime_tools.remove("spawn_subagent");
    }
    if config.depth.is_subagent() {
        let finish_tool = Arc::new(FinishTool {
            result: Arc::clone(&finish_store),
        });
        runtime_tools.insert("finish".to_string(), finish_tool as Arc<dyn Tool>);
    }

    let mut history = Vec::new();
    let mut prompt = Message::user(initial_message.to_string());
    history.push(prompt.clone());
    let mut total_input_tokens = 0u64;
    let mut total_output_tokens = 0u64;
    let mut total_tokens = 0u64;
    let mut conversation_usage = ConversationUsageWindow::new(config.compact_threshold);
    let mut total_tool_calls = 0usize;
    let mut empty_response_count = 0usize;
    let mut tool_call_history: VecDeque<String> = VecDeque::new();
    let mut last_tool_call_outputs: HashMap<String, String> = HashMap::new();
    let initial_subagent_count = config.subagent_counter.load(Ordering::Relaxed);
    let mut boost_next_completion_temperature = false;

    for turn in 0..config.max_turns {
        if conversation_usage.should_compact() {
            let usage_before_compaction = conversation_usage.usage();
            if let Some(compaction_usage) = compact_history(
                Arc::clone(&config.client),
                &config.model,
                &config.system_prompt,
                &mut history,
                &mut prompt,
                turn + 1,
                usage_before_compaction,
            )
            .await?
            {
                total_input_tokens = total_input_tokens.saturating_add(compaction_usage.input_tokens);
                total_output_tokens = total_output_tokens.saturating_add(compaction_usage.output_tokens);
                total_tokens = total_tokens.saturating_add(compaction_usage.total_tokens);
                conversation_usage.reset();
            }
        }

        let temperature = if boost_next_completion_temperature {
            info!(
                agent = %config.name,
                turn,
                temperature = REPEATED_TOOL_CALL_RECOVERY_TEMPERATURE,
                "temporarily increasing temperature after repeated tool-call block"
            );
            REPEATED_TOOL_CALL_RECOVERY_TEMPERATURE
        } else {
            DEFAULT_AGENT_TEMPERATURE
        };
        boost_next_completion_temperature = false;

        let completion = Completion {
            model: config.model.clone(),
            prompt: prompt.clone(),
            preamble: Some(config.system_prompt.clone()),
            history: history[..history.len().saturating_sub(1)].to_vec(),
            tools: tool_definitions(&runtime_tools),
            temperature: Some(temperature),
            max_tokens: Some(8192),
            additional_params: None,
        };

        let response = config.client.completion(completion).await?;
        total_input_tokens = total_input_tokens.saturating_add(response.usage.input_tokens);
        total_output_tokens = total_output_tokens.saturating_add(response.usage.output_tokens);
        total_tokens = total_tokens.saturating_add(response.usage.total_tokens);
        conversation_usage.record(response.usage);
        let assistant_message = response.message();
        history.push(assistant_message.clone());

        if let Some(tool_calls) = response.tool_calls() {
            empty_response_count = 0;
            let mut results = Vec::new();
            total_tool_calls += tool_calls.len();
            report_progress(&config, turn + 1, total_tool_calls, initial_subagent_count);
            let mut should_terminate = false;
            for call in tool_calls {
                let tool_name = call.function.name.clone();
                let args = call.function.arguments.clone();
                let tool_call_key = format!("{tool_name}:{args}");
                tool_call_history.push_back(tool_call_key.clone());
                if tool_call_history.len() > TOOL_CALL_HISTORY_WINDOW {
                    tool_call_history.pop_front();
                }
                let cycle_len = detect_tool_call_cycle(&tool_call_history);
                should_terminate |= config.terminal_tools.iter().any(|name| name == &tool_name);
                info!(agent = %config.name, tool = %tool_name, args = %args, turn, "tool call");
                let outcome = execute_tool_call(
                    ToolCallContext {
                        config: &config,
                        runtime_tools: &runtime_tools,
                        tools_map,
                        work_dir,
                        turn,
                        current_turns: turn + 1,
                        total_tool_calls,
                        initial_subagent_count,
                    },
                    &tool_name,
                    args,
                    cycle_len,
                )
                .await?;
                if outcome.repeated_tool_call_blocked {
                    boost_next_completion_temperature = true;
                }
                let ToolCallOutcome {
                    output,
                    nested_tool_calls,
                    repeated_tool_call_blocked: _,
                } = outcome;
                last_tool_call_outputs.insert(tool_call_key, output.clone());
                total_tool_calls += nested_tool_calls;
                report_progress(&config, turn + 1, total_tool_calls, initial_subagent_count);
                let mut output = output;
                if output.len() > MAX_TOOL_RESULT_BYTES {
                    let boundary = floor_char_boundary(&output, MAX_TOOL_RESULT_BYTES);
                    output.truncate(boundary);
                    output.push_str("\n... truncated (>50k bytes)");
                }
                results.push(ToolResult {
                    id: call.id.clone(),
                    call_id: call.call_id.clone(),
                    content: OneOrMany::one(ToolResultContent::text(output)),
                });
            }

            if config.depth.is_subagent() {
                if let Some(result) = finish_store
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                {
                    info!(
                        agent = %config.name,
                        turn,
                        input_tokens = response.usage.input_tokens,
                        output_tokens = response.usage.output_tokens,
                        total_tokens = response.usage.total_tokens,
                        total_input_tokens,
                        total_output_tokens,
                        total_tokens_so_far = total_tokens,
                        response_len = result.len(),
                        "subagent finished"
                    );
                    return Ok(AgentResult {
                        text: result,
                        turns: turn + 1,
                        tool_calls: total_tool_calls,
                        subagents_spawned: config.subagent_counter.load(Ordering::Relaxed)
                            - initial_subagent_count,
                        total_input_tokens,
                        total_output_tokens,
                        total_tokens,
                    });
                }
            }

            if should_terminate {
                info!(
                    agent = %config.name,
                    turn,
                    total_tool_calls,
                    total_input_tokens,
                    total_output_tokens,
                    total_tokens,
                    "terminal tool called"
                );
                return Ok(AgentResult {
                    text: String::new(),
                    turns: turn + 1,
                    tool_calls: total_tool_calls,
                    subagents_spawned: config.subagent_counter.load(Ordering::Relaxed)
                        - initial_subagent_count,
                    total_input_tokens,
                    total_output_tokens,
                    total_tokens,
                });
            }

            let tool_message = Message::User {
                content: OneOrMany::many(results.into_iter().map(UserContent::ToolResult))
                    .expect("tool results must not be empty"),
            };
            history.push(tool_message.clone());
            prompt = tool_message;
        } else {
            tool_call_history.clear();
            let text = response.text();
            report_progress(&config, turn + 1, total_tool_calls, initial_subagent_count);
            if text.is_empty() {
                if let Some(nudge) = &config.empty_response_nudge {
                    empty_response_count += 1;
                    if empty_response_count <= config.max_empty_responses {
                        let nudge = Message::user(nudge.clone());
                        history.push(nudge.clone());
                        prompt = nudge;
                        continue;
                    }
                }
                eyre::bail!("empty response from model (no text, no tool calls)");
            }
            if config.depth.is_subagent() {
                eyre::bail!("subagent returned text without calling finish")
            }
            info!(
                agent = %config.name,
                turn,
                input_tokens = response.usage.input_tokens,
                output_tokens = response.usage.output_tokens,
                total_tokens = response.usage.total_tokens,
                total_input_tokens,
                total_output_tokens,
                total_tokens_so_far = total_tokens,
                response_len = text.len(),
                "finished"
            );
            return Ok(AgentResult {
                text,
                turns: turn + 1,
                tool_calls: total_tool_calls,
                subagents_spawned: config.subagent_counter.load(Ordering::Relaxed)
                    - initial_subagent_count,
                total_input_tokens,
                total_output_tokens,
                total_tokens,
            });
        }
    }

    eyre::bail!("agent loop exceeded {} turns", config.max_turns)
}

fn report_progress(
    config: &AgentConfig,
    turns: usize,
    tool_calls: usize,
    initial_subagent_count: usize,
) {
    if let Some(progress) = &config.progress {
        progress(AgentProgress {
            turns,
            tool_calls,
            subagents_spawned: config.subagent_counter.load(Ordering::Relaxed)
                - initial_subagent_count,
        });
    }
}

pub fn add_spawn_subagent_tool(tools_map: &mut HashMap<String, Arc<dyn Tool>>) {
    tools_map.insert("spawn_subagent".to_string(), Arc::new(SpawnSubagentTool));
}

struct SpawnSubagentTool;

impl Tool for SpawnSubagentTool {
    fn name(&self) -> String {
        "spawn_subagent".to_string()
    }

    fn definition(&self) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: "spawn_subagent".to_string(),
            description: "Delegate a focused investigation to a subagent. Provide a small detailed task with all needed context."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A compact self-contained task for the subagent"
                    }
                },
                "required": ["task"],
                "additionalProperties": false
            }),
        }
    }

    fn call(
        &self,
        _args: Value,
        _work_dir: PathBuf,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>> {
        Box::pin(async { unreachable!("spawn_subagent is handled internally") })
    }
}

async fn run_subagent(
    parent_config: &AgentConfig,
    args: &Value,
    tools_map: &HashMap<String, Arc<dyn Tool>>,
    work_dir: &Path,
    parent_turns: usize,
    parent_tool_calls: usize,
    initial_subagent_count: usize,
) -> (String, usize) {
    let task = match args.get("task").and_then(|value| value.as_str()) {
        Some(task) if !task.trim().is_empty() => task,
        _ => return ("Error: missing task".to_string(), 0),
    };

    let subagent_id = parent_config
        .subagent_counter
        .fetch_add(1, Ordering::Relaxed)
        + 1;
    report_progress(
        parent_config,
        parent_turns,
        parent_tool_calls,
        initial_subagent_count,
    );
    let subagent_level = parent_config.depth.level() + 1;
    let subagent_config = AgentConfig {
        name: format!("{}/subagent-{subagent_id}", parent_config.name),
        model: parent_config.model.clone(),
        max_turns: parent_config.max_turns,
        compact_threshold: parent_config.compact_threshold,
        system_prompt: subagent_system_prompt().to_string(),
        client: Arc::clone(&parent_config.client),
        depth: AgentDepth::Subagent {
            level: subagent_level,
        },
        terminal_tools: Vec::new(),
        empty_response_nudge: None,
        max_empty_responses: 0,
        subagent_counter: Arc::clone(&parent_config.subagent_counter),
        progress: None,
    };

    match Box::pin(run_agent(subagent_config, task, tools_map, work_dir)).await {
        Ok(result) => (result.text, result.tool_calls),
        Err(err) => (format!("Error: {err}"), 0),
    }
}

/// Returns the cycle period (1, 2, or 3) if the tail of `history` forms a repeated cycle,
/// or 0 if no cycle is detected. Period-1 maps to the existing consecutive-identical check.
fn detect_tool_call_cycle(history: &VecDeque<String>) -> usize {
    let h: Vec<&str> = history.iter().map(String::as_str).collect();
    let n = h.len();
    for period in 1..=3 {
        let reps = if period == 1 {
            MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS
        } else {
            MAX_CYCLE_REPETITIONS
        };
        let needed = period * reps;
        if n < needed {
            continue;
        }
        let start = n - needed;
        let pattern = &h[start..start + period];
        if h[start..]
            .chunks(period)
            .take(reps)
            .all(|chunk| chunk == pattern)
        {
            return period;
        }
    }
    0
}

async fn execute_tool_call(
    ctx: ToolCallContext<'_>,
    tool_name: &str,
    args: Value,
    cycle_len: usize,
) -> Result<ToolCallOutcome> {
    if cycle_len > 0 {
        warn!(
            agent = %ctx.config.name,
            tool = %tool_name,
            args = %args,
            turn = ctx.turn,
            cycle_len,
            next_temperature = REPEATED_TOOL_CALL_RECOVERY_TEMPERATURE,
            "blocking cyclic tool call"
        );
        let msg = if cycle_len == 1 {
            format!("Warning: repeated identical tool call blocked for {tool_name}. Think twice; try changing the arguments or using a different tool.")
        } else {
            format!("Warning: tool call cycle of period {cycle_len} detected (e.g. A→B→A→B). You are looping without making progress. Try a different approach or tool.")
        };
        return Ok(ToolCallOutcome {
            output: msg,
            nested_tool_calls: 0,
            repeated_tool_call_blocked: true,
        });
    }

    if tool_name == "spawn_subagent" {
        if !ctx.config.depth.can_spawn_subagent() {
            return Ok(ToolCallOutcome {
                output: "Error: subagent depth limit reached; cannot spawn another subagent"
                    .to_string(),
                nested_tool_calls: 0,
                repeated_tool_call_blocked: false,
            });
        }
        info!(agent = %ctx.config.name, turn = ctx.turn, "spawning subagent");
        let (output, nested_tool_calls) = run_subagent(
            ctx.config,
            &args,
            ctx.tools_map,
            ctx.work_dir,
            ctx.current_turns,
            ctx.total_tool_calls,
            ctx.initial_subagent_count,
        )
        .await;
        return Ok(ToolCallOutcome {
            output,
            nested_tool_calls,
            repeated_tool_call_blocked: false,
        });
    }

    match ctx.runtime_tools.get(tool_name) {
        Some(tool) => match tool.call(args, ctx.work_dir.to_path_buf()).await {
            Ok(output) => Ok(ToolCallOutcome {
                output,
                nested_tool_calls: 0,
                repeated_tool_call_blocked: false,
            }),
            Err(err) => {
                warn!(agent = %ctx.config.name, tool = %tool_name, error = %err, "tool error");
                Ok(ToolCallOutcome {
                    output: format!("Error: {err}"),
                    nested_tool_calls: 0,
                    repeated_tool_call_blocked: false,
                })
            }
        },
        None => {
            let msg = format!("Error: unknown tool '{tool_name}'");
            warn!(agent = %ctx.config.name, tool = %tool_name, "unknown tool");
            Ok(ToolCallOutcome {
                output: msg,
                nested_tool_calls: 0,
                repeated_tool_call_blocked: false,
            })
        }
    }
}
