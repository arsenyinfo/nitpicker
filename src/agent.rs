use crate::llm::{Completion, LLMClientDyn};
use crate::tools::{Tool, tool_definitions};
use eyre::Result;
use rig::OneOrMany;
use rig::completion::Message;
use rig::completion::message::{ToolResult, ToolResultContent, UserContent};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

const MAX_TURNS: usize = 100;
const MAX_TOOL_RESULT_BYTES: usize = 50_000;

pub struct AgentResult {
    pub text: String,
    pub turns: usize,
}

pub struct AgentConfig {
    pub name: String,
    pub model: String,
    pub system_prompt: String,
    pub client: Arc<dyn LLMClientDyn>,
}

pub async fn run_agent(
    config: AgentConfig,
    initial_message: &str,
    tools_map: &HashMap<String, Arc<dyn Tool>>,
    work_dir: &Path,
) -> Result<AgentResult> {
    let mut history = Vec::new();
    let mut prompt = Message::user(initial_message.to_string());
    history.push(prompt.clone());
    let mut total_output_tokens = 0u64;

    for turn in 0..MAX_TURNS {
        let completion = Completion {
            model: config.model.clone(),
            prompt: prompt.clone(),
            preamble: Some(config.system_prompt.clone()),
            history: history[..history.len().saturating_sub(1)].to_vec(),
            tools: tool_definitions(tools_map),
            temperature: None,
            max_tokens: Some(8192),
            additional_params: None,
        };

        let response = config.client.completion(completion).await?;
        total_output_tokens += response.output_tokens;
        let assistant_message = response.message();
        history.push(assistant_message.clone());

        if let Some(tool_calls) = response.tool_calls() {
            let mut results = Vec::new();
            for call in tool_calls {
                let tool_name = call.function.name.clone();
                let args = call.function.arguments.clone();
                info!(agent = %config.name, tool = %tool_name, args = %args, turn, "tool call");
                let output = match tools_map.get(&tool_name) {
                    Some(tool) => match tool.call(args, work_dir.to_path_buf()).await {
                        Ok(output) => output,
                        Err(err) => {
                            warn!(agent = %config.name, tool = %tool_name, error = %err, "tool error");
                            format!("Error: {err}")
                        }
                    },
                    None => {
                        let msg = format!("Error: unknown tool '{tool_name}'");
                        warn!(agent = %config.name, tool = %tool_name, "unknown tool");
                        msg
                    }
                };
                let mut output = output;
                if output.len() > MAX_TOOL_RESULT_BYTES {
                    let boundary = output.floor_char_boundary(MAX_TOOL_RESULT_BYTES);
                    output.truncate(boundary);
                    output.push_str("\n... truncated (>50k bytes)");
                }
                results.push(ToolResult {
                    id: call.id.clone(),
                    call_id: call.call_id.clone(),
                    content: OneOrMany::one(ToolResultContent::text(output)),
                });
            }
            let tool_message = Message::User {
                content: OneOrMany::many(results.into_iter().map(UserContent::ToolResult))
                    .expect("tool results must not be empty"),
            };
            history.push(tool_message.clone());
            prompt = tool_message;
        } else {
            let text = response.text();
            if text.is_empty() {
                eyre::bail!("empty response from model (no text, no tool calls)");
            }
            info!(
                agent = %config.name,
                turn,
                output_tokens = response.output_tokens,
                total_output_tokens,
                response_len = text.len(),
                "finished"
            );
            return Ok(AgentResult {
                text,
                turns: turn + 1,
            });
        }
    }

    eyre::bail!("agent loop exceeded MAX_TURNS")
}
