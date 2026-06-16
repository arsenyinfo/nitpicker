//! Reusable agentic loop extracted from [nitpicker](https://github.com/arsenyinfo/nitpicker):
//! an LLM agent that reads files (read_file / glob / grep / git) and fans out parallel
//! subagents, with provider-agnostic LLM clients and optional config-file-driven setup.
//!
//! The fastest way in is the [`prelude`] plus [`AgentBuilder`]:
//!
//! ```no_run
//! use nitpicker_agent::prelude::*;
//! use std::path::Path;
//!
//! # async fn run() -> eyre::Result<()> {
//! let client = client_from_env(LLMProvider::Anthropic { base_url: None, api_key_env: None })?;
//! let result = AgentBuilder::new("explorer", "claude-sonnet-4-6", "You explore codebases.", client)
//!     .subagent_system_prompt("You are a focused file-reading worker. Report findings concisely.")
//!     .run("Map the module layout of this repo.", &file_agent_tools(), Path::new("."))
//!     .await?;
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod codex;
pub mod compact;
pub mod config;
pub mod llm;
pub mod openrouter;
pub mod prompts;
pub mod provider;
pub mod session;
pub mod tools;

#[cfg(feature = "azure")]
pub mod azure;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use eyre::Result;
use tokio::sync::Semaphore;

use agent::{AgentConfig, AgentDepth, AgentProgress, AgentResult, MAX_CONCURRENT_LLM_CALLS, run_agent};
use config::DEFAULT_MAX_TURNS;
use llm::{LLMClient, LLMClientDyn, LLMProvider, WithRetryExt};
use tools::Tool;

/// Curated entrypoints for driving a file-reading agent with subagents.
pub mod prelude {
    pub use crate::agent::{
        AgentConfig, AgentDepth, AgentProgress, AgentResult, MAX_CONCURRENT_LLM_CALLS,
        add_spawn_subagent_tool, run_agent,
    };
    pub use crate::llm::{LLMClientDyn, LLMProvider, WithRetryExt};
    pub use crate::tools::{Tool, all_tools};
    pub use crate::{AgentBuilder, client_from_env, file_agent_tools};
}

/// The file/git toolset plus `spawn_subagent` — the "reads files and spawns subagents" kit.
pub fn file_agent_tools() -> HashMap<String, Arc<dyn Tool>> {
    let mut tools = tools::all_tools();
    agent::add_spawn_subagent_tool(&mut tools);
    tools
}

/// Build a retry-wrapped client for a provider, reading its API key from the environment.
pub fn client_from_env(provider: LLMProvider) -> Result<Arc<dyn LLMClientDyn>> {
    Ok(provider.client_from_env()?.with_retry().into_arc())
}

/// Ergonomic constructor for [`AgentConfig`]. Required inputs are passed to [`AgentBuilder::new`];
/// every other field defaults to the same values the nitpicker review path uses, and is
/// overridable via the chainable setters. Use [`AgentBuilder::build`] for a raw config or
/// [`AgentBuilder::run`] to execute in one call.
///
/// Concurrency note: each builder defaults to its **own** `Semaphore::new(MAX_CONCURRENT_LLM_CALLS)`,
/// so it bounds in-flight LLM calls within one agent (and its subagents) but not *across*
/// independent builders — N agents can run N × `MAX_CONCURRENT_LLM_CALLS` calls at once. To cap
/// aggregate concurrency account-wide, share one `Arc<Semaphore>` via [`AgentBuilder::llm_semaphore`].
pub struct AgentBuilder {
    config: AgentConfig,
}

impl AgentBuilder {
    pub fn new(
        name: impl Into<String>,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
        client: Arc<dyn LLMClientDyn>,
    ) -> Self {
        let config = AgentConfig {
            name: name.into(),
            session_agent: "root".to_string(),
            model: model.into(),
            max_turns: DEFAULT_MAX_TURNS,
            compact_threshold: None,
            system_prompt: system_prompt.into(),
            subagent_system_prompt: None,
            client,
            depth: AgentDepth::TopLevel,
            terminal_tools: Vec::new(),
            empty_response_nudge: None,
            max_empty_responses: 0,
            subagent_counter: Arc::new(AtomicUsize::new(0)),
            llm_semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_LLM_CALLS)),
            progress: None,
            project_context: None,
            session_writer: None,
        };
        Self { config }
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.config.max_turns = max_turns;
        self
    }

    pub fn compact_threshold(mut self, threshold: u64) -> Self {
        self.config.compact_threshold = Some(threshold);
        self
    }

    /// Override the system prompt spawned subagents run with (default: the built-in generic
    /// prompt). The override propagates through nested subagent spawns.
    pub fn subagent_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.subagent_system_prompt = Some(prompt.into());
        self
    }

    /// Extra context appended to the system prompt (e.g. a project summary).
    pub fn project_context(mut self, context: impl Into<String>) -> Self {
        self.config.project_context = Some(context.into());
        self
    }

    /// Share a concurrency limiter across agents instead of the per-builder default
    /// (`Semaphore::new(MAX_CONCURRENT_LLM_CALLS)`), to cap aggregate in-flight LLM calls.
    pub fn llm_semaphore(mut self, semaphore: Arc<Semaphore>) -> Self {
        self.config.llm_semaphore = semaphore;
        self
    }

    pub fn progress(mut self, progress: Arc<dyn Fn(AgentProgress) + Send + Sync>) -> Self {
        self.config.progress = Some(progress);
        self
    }

    pub fn build(self) -> AgentConfig {
        self.config
    }

    /// Build and run the agent to completion.
    pub async fn run(
        self,
        initial_message: &str,
        tools: &HashMap<String, Arc<dyn Tool>>,
        work_dir: &std::path::Path,
    ) -> Result<AgentResult> {
        run_agent(self.config, initial_message, tools, work_dir).await
    }
}
