use crate::llm::{Completion, CompletionResponse, LLMClientDyn, TokenUsage};
use eyre::Result;
use rig_core::completion::message::{ToolChoice, ToolResultContent, UserContent};
use rig_core::completion::{AssistantContent, Message};
use std::sync::Arc;
use tracing::{info, warn};

const COMPACTION_MAX_OUTPUT_TOKENS: u64 = 8_192;
const COMPACTION_MAX_CORRECTIONS: usize = 2;
const CONTINUE_MESSAGE: &str = "Continue from where you left off.";
const SUMMARY_TAG: &str = "summary";
const COMPACTION_SUMMARY_INSTRUCTIONS: &str = "Focus strictly on information vital for code analysis. Omit conversational filler, raw search/tool outputs, and large blocks of code.\n\
When constructing the summary, you MUST use the following exact markdown structure inside the tags:\n\
<summary>\n\
## Review Goal\n\
[1-2 sentences on the core objective of this code exploration/review. What are we looking for?]\n\
## Key Findings & Discoveries\n\
- [List major architectural insights, design patterns discovered, or critical issues/bugs identified so far.]\n\
- [Keep these concise but highly technical.]\n\
## Codebase Map (Relevant Files)\n\
- [List the critical files and directories that have been identified as relevant to the goal.]\n\
- [Briefly note why they are relevant (e.g., `src/auth/token.rs`: handles JWT validation and is where the bug likely resides).]\n\
## Explored Territory\n\
- [Briefly list what areas, files, or concepts have already been thoroughly investigated so we do not repeat work.]\n\
## Last Action & Immediate Context\n\
- [Describe the most recent tool calls and their results. What was the agent trying to find or verify? What did it just learn?]\n\
## Open Questions & Next Steps\n\
- [List any unresolved anomalies, pending review items, constraints to remember, or specific files that still need to be examined.]\n\
</summary>\n\
Return EXACTLY ONE tagged block starting with <summary> and ending with </summary>. Do not include any text, pleasantries, or explanations outside of these tags.";
const MISSING_SUMMARY_CORRECTION: &str = "Your response is missing the required <summary>...</summary> block. Reply with ONLY the tagged summary block and nothing else outside the tags.";
const WRAP_SUMMARY_CORRECTION: &str = "Wrap your entire response in <summary>...</summary> tags and output nothing outside them.";

pub struct CompactionOutcome {
    pub usage: TokenUsage,
    pub summary: String,
    pub trigger_usage: TokenUsage,
}

enum CompactionInput<'a> {
    Messages(&'a [Message]),
    Transcript(&'a str),
}

enum CorrectionMode {
    MissingSummary,
    WrapSummary,
}

pub async fn compact_history(
    client: Arc<dyn LLMClientDyn>,
    model: &str,
    system_prompt: &str,
    history: &mut Vec<Message>,
    prompt: &mut Message,
    turn: usize,
    usage: TokenUsage,
) -> Result<Option<CompactionOutcome>> {
    if history.is_empty() {
        return Ok(None);
    }

    let outcome = summarize_history(client, model, system_prompt, history, turn, usage).await?;
    let continue_msg = Message::user(CONTINUE_MESSAGE.to_string());
    *history = vec![
        Message::user(format!(
            "compacted conversation summary before turn {turn}:\n\n{}",
            outcome.summary
        )),
        continue_msg.clone(),
    ];
    *prompt = continue_msg;
    info!(
        turn,
        trigger_input_tokens = outcome.trigger_usage.input_tokens,
        trigger_output_tokens = outcome.trigger_usage.output_tokens,
        trigger_total_tokens = outcome.trigger_usage.total_tokens,
        input_tokens = outcome.usage.input_tokens,
        output_tokens = outcome.usage.output_tokens,
        total_tokens = outcome.usage.total_tokens,
        "compaction complete"
    );
    Ok(Some(outcome))
}

async fn summarize_history(
    client: Arc<dyn LLMClientDyn>,
    model: &str,
    system_prompt: &str,
    thread: &[Message],
    turn: usize,
    usage: TokenUsage,
) -> Result<CompactionOutcome> {
    match run_compaction(
        &client,
        model,
        system_prompt,
        CompactionInput::Messages(thread),
        turn,
        usage,
    )
    .await
    {
        Ok(outcome) => Ok(outcome),
        Err(err) => {
            warn!("in-context compaction failed ({err}), retrying with rendered transcript");
            let transcript = render_history_as_text(thread);
            let outcome = run_compaction(
                &client,
                model,
                system_prompt,
                CompactionInput::Transcript(&transcript),
                turn,
                usage,
            )
            .await?;
            info!("rendered transcript compaction succeeded");
            Ok(outcome)
        }
    }
}

async fn run_compaction(
    client: &Arc<dyn LLMClientDyn>,
    model: &str,
    system_prompt: &str,
    input: CompactionInput<'_>,
    turn: usize,
    usage: TokenUsage,
) -> Result<CompactionOutcome> {
    let prompt = build_compaction_prompt(&input, turn, usage);
    let response = client
        .completion(compaction_completion(
            model,
            system_prompt,
            history_for_initial_request(&input),
            Message::user(prompt.clone()),
        ))
        .await?;

    match extract_summary(&response) {
        Some(summary) => Ok(CompactionOutcome {
            usage: response.usage,
            summary,
            trigger_usage: usage,
        }),
        None => {
            let attempt = InitialAttempt {
                prompt,
                response,
                correction_mode: correction_mode(&input),
            };
            run_corrections(client, model, system_prompt, &input, usage, attempt).await
        }
    }
}

struct InitialAttempt {
    prompt: String,
    response: CompletionResponse,
    correction_mode: CorrectionMode,
}

async fn run_corrections(
    client: &Arc<dyn LLMClientDyn>,
    model: &str,
    system_prompt: &str,
    input: &CompactionInput<'_>,
    trigger_usage: TokenUsage,
    attempt: InitialAttempt,
) -> Result<CompactionOutcome> {
    let InitialAttempt {
        prompt: original_prompt,
        response: initial_response,
        correction_mode,
    } = attempt;
    let correction = correction_prompt(&correction_mode);
    let mut history = correction_history(input, original_prompt, initial_response.message());
    let mut previous = initial_response;

    for _ in 0..COMPACTION_MAX_CORRECTIONS {
        let next = client
            .completion(compaction_completion(
                model,
                system_prompt,
                history.clone(),
                Message::user(correction.to_string()),
            ))
            .await?;
        match extract_summary(&next) {
            Some(summary) => {
                return Ok(CompactionOutcome {
                    usage: next.usage,
                    summary,
                    trigger_usage,
                });
            }
            None => {
                history.push(next.message());
                history.push(Message::user(correction.to_string()));
                previous = next;
            }
        }
    }

    match correction_mode {
        CorrectionMode::MissingSummary => Err(eyre::eyre!(
            "compaction output missing <summary> block after {COMPACTION_MAX_CORRECTIONS} corrections: {}",
            previous.text()
        )),
        CorrectionMode::WrapSummary => Err(eyre::eyre!(
            "rendered compaction also failed to produce <summary> block: {}",
            previous.text()
        )),
    }
}

fn compaction_completion(
    model: &str,
    system_prompt: &str,
    history: Vec<Message>,
    prompt: Message,
) -> Completion {
    Completion {
        model: model.to_string(),
        prompt,
        preamble: Some(system_prompt.to_string()),
        history,
        tools: Vec::new(),
        tool_choice: Some(ToolChoice::None),
        max_tokens: Some(COMPACTION_MAX_OUTPUT_TOKENS),
        additional_params: None,
    }
}

fn history_for_initial_request(input: &CompactionInput<'_>) -> Vec<Message> {
    match input {
        CompactionInput::Messages(thread) => thread.to_vec(),
        CompactionInput::Transcript(_) => Vec::new(),
    }
}

fn correction_history(
    input: &CompactionInput<'_>,
    original_prompt: String,
    assistant_message: Message,
) -> Vec<Message> {
    let mut history = history_for_initial_request(input);
    history.push(Message::user(original_prompt));
    history.push(assistant_message);
    history
}

fn correction_mode(input: &CompactionInput<'_>) -> CorrectionMode {
    match input {
        CompactionInput::Messages(_) => CorrectionMode::MissingSummary,
        CompactionInput::Transcript(_) => CorrectionMode::WrapSummary,
    }
}

fn correction_prompt(mode: &CorrectionMode) -> &'static str {
    match mode {
        CorrectionMode::MissingSummary => MISSING_SUMMARY_CORRECTION,
        CorrectionMode::WrapSummary => WRAP_SUMMARY_CORRECTION,
    }
}

fn build_compaction_prompt(input: &CompactionInput<'_>, turn: usize, usage: TokenUsage) -> String {
    let intro = match input {
        CompactionInput::Messages(_) => {
            "This thread is getting out of hand and needs to be compacted. Stop previous work and summarize the conversation so far.".to_string()
        }
        CompactionInput::Transcript(transcript) => format!(
            "Summarize the following code exploration and review session transcript so the agent can continue seamlessly from where it left off.\n\n<transcript>\n{transcript}\n</transcript>"
        ),
    };

    format!(
        "{intro}\n\n{COMPACTION_SUMMARY_INSTRUCTIONS}\n\
This compaction is happening before turn {turn}. Context size at compaction: {} input, {} output, {} total tokens.",
        usage.input_tokens, usage.output_tokens, usage.total_tokens
    )
}

fn extract_summary(response: &CompletionResponse) -> Option<String> {
    extract_tag(&response.text(), SUMMARY_TAG)
}

fn render_history_as_text(history: &[Message]) -> String {
    let mut parts = Vec::new();
    for msg in history {
        match msg {
            Message::User { content } => {
                for item in content.iter() {
                    match item {
                        UserContent::Text(t) if !t.text.is_empty() => {
                            parts.push(format!("[user]: {}", t.text));
                        }
                        UserContent::ToolResult(tr) => {
                            let result = tr
                                .content
                                .iter()
                                .filter_map(|content| match content {
                                    ToolResultContent::Text(t) => Some(t.text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("");
                            parts.push(format!("[tool result {}]: {}", tr.id, result));
                        }
                        _ => {}
                    }
                }
            }
            Message::Assistant { content, .. } => {
                for item in content.iter() {
                    match item {
                        AssistantContent::Text(t) if !t.text().is_empty() => {
                            parts.push(format!("[assistant]: {}", t.text()));
                        }
                        AssistantContent::ToolCall(tc) => {
                            parts.push(format!(
                                "[called: {}({})]",
                                tc.function.name, tc.function.arguments
                            ));
                        }
                        _ => {}
                    }
                }
            }
            Message::System { .. } => {}
        }
    }
    parts.join("\n\n")
}

fn extract_tag(text: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = text.find(&start_tag)? + start_tag.len();
    let end = start + text[start..].find(&end_tag)?;
    if end < start {
        return None;
    }
    let content = text[start..end].trim();
    if content.is_empty() {
        None
    } else {
        Some(content.to_string())
    }
}
