use crate::llm::{Completion, LLMClientDyn, TokenUsage};
use eyre::Result;
use rig::completion::Message;
use std::sync::Arc;

const COMPACTION_MAX_OUTPUT_TOKENS: u64 = 8_192;
const COMPACTION_MAX_ATTEMPTS: usize = 2;

pub async fn compact_history(
    client: Arc<dyn LLMClientDyn>,
    model: &str,
    system_prompt: &str,
    history: &mut Vec<Message>,
    prompt: &mut Message,
    turn: usize,
    usage: TokenUsage,
) -> Result<Option<TokenUsage>> {
    if history.is_empty() {
        return Ok(None);
    }

    let (response, summary) = summarize_history(client, model, system_prompt, history.clone(), turn, usage).await?;

    let continue_msg = Message::user("Continue from where you left off.".to_string());
    *history = vec![
        Message::user(format!("compacted conversation summary before turn {turn}:\n\n{summary}")),
        continue_msg.clone(),
    ];
    *prompt = continue_msg;
    Ok(Some(response.usage))
}

async fn summarize_history(
    client: Arc<dyn LLMClientDyn>,
    model: &str,
    system_prompt: &str,
    thread: Vec<Message>,
    turn: usize,
    usage: TokenUsage,
) -> Result<(crate::llm::CompletionResponse, String)> {
    let mut last_text = None;

    for attempt in 1..=COMPACTION_MAX_ATTEMPTS {
        let response = client
            .completion(Completion {
                model: model.to_string(),
                prompt: Message::user(build_compaction_prompt(turn, usage, attempt)),
                preamble: Some(system_prompt.to_string()),
                history: thread.clone(),
                tools: Vec::new(),
                temperature: Some(0.2),
                max_tokens: Some(COMPACTION_MAX_OUTPUT_TOKENS),
                additional_params: None,
            })
            .await?;

        let text = response.text();
        if let Some(summary) = extract_tag(&text, "summary") {
            return Ok((response, summary));
        }
        last_text = Some(text);
    }

    Err(eyre::eyre!(
        "compaction output missing <summary> block after {COMPACTION_MAX_ATTEMPTS} attempts: {}",
        last_text.unwrap_or_default()
    ))
}

fn build_compaction_prompt(turn: usize, usage: TokenUsage, attempt: usize) -> String {
    format!(
        "You are an expert software engineer summarizing a code exploration and review session \
to free up context window space.\n\n\
The most recent messages will be preserved verbatim after this summary. Your goal is \
to capture the durable context from the older messages that is necessary to continue the \
exploration/review without losing the plot.\n\
Focus strictly on information vital for code analysis. Omit conversational filler, raw \
search/tool outputs, and large blocks of code.\n\
When constructing the summary, you MUST use the following exact markdown structure \
inside the tags:\n\
<summary>\n\
## Review Goal\n\
[1-2 sentences on the core objective of this code exploration/review. What are we \
looking for?]\n\
## Key Findings & Discoveries\n\
- [List major architectural insights, design patterns discovered, or critical \
issues/bugs identified so far.]\n\
- [Keep these concise but highly technical.]\n\
## Codebase Map (Relevant Files)\n\
- [List the critical files and directories that have been identified as relevant \
to the goal.]\n\
- [Briefly note why they are relevant (e.g., `src/auth/token.rs`: handles JWT \
validation and is where the bug likely resides).]\n\
## Explored Territory\n\
- [Briefly list what areas, files, or concepts have already been thoroughly \
investigated so we do not repeat work.]\n\
## Open Questions & Next Steps\n\
- [List any unresolved anomalies, pending review items, constraints to remember, \
or specific files that still need to be examined.]\n\
</summary>\n\
This compaction is happening before turn {turn}. Usage since the previous compaction: \
{} input, {} output, {} total tokens.\n\
CRITICAL: This is attempt {attempt} of {COMPACTION_MAX_ATTEMPTS}. Return EXACTLY ONE \
tagged block starting with <summary> and ending with </summary>. Do not include any \
text, pleasantries, or explanations outside of these tags. If a prior attempt failed, \
ensure you output ONLY the XML tags this time.",
        usage.input_tokens, usage.output_tokens, usage.total_tokens
    )
}

fn extract_tag(text: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = text.find(&start_tag)? + start_tag.len();
    let end = text.rfind(&end_tag)?;
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
