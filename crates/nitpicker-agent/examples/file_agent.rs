//! A file-reading agent that fans out subagents, built with the public prelude.
//!
//! Run against any directory (defaults to the current one):
//!
//! ```sh
//! ANTHROPIC_API_KEY=sk-... cargo run --example file_agent -p nitpicker-agent -- .
//! ```

use std::path::Path;

use nitpicker_agent::prelude::*;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let work_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());

    let client = client_from_env(LLMProvider::Anthropic {
        base_url: None,
        api_key_env: None,
    })?;

    let result = AgentBuilder::new(
        "explorer",
        "claude-sonnet-5",
        "You explore codebases. Build a quick map, then delegate disjoint threads to subagents.",
        client,
    )
    .subagent_system_prompt(
        "You are a focused file-reading worker. Inspect only what the task asks and report \
         findings concisely with file:line evidence. Call finish when done.",
    )
    .max_turns(20)
    .run(
        "Summarize this project's module layout and what each top-level source file does.",
        &file_agent_tools(),
        Path::new(&work_dir),
    )
    .await?;

    println!("{}\n", result.text);
    eprintln!(
        "[turns={} tool_calls={} subagents={} tokens={}]",
        result.turns, result.tool_calls, result.subagents_spawned, result.total_tokens
    );
    Ok(())
}
