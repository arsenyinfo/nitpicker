# nitpicker-agent

The reusable agentic core extracted from [nitpicker](https://github.com/arsenyinfo/nitpicker):
an LLM agent that reads files (`read_file` / `glob` / `grep` / read-only `git`) and fans out
parallel subagents, with provider-agnostic LLM clients and optional config-file-driven setup.

It carries none of nitpicker's CLI, review/debate, or PR machinery — just the loop, the tools,
and the providers.

The caller owns task semantics through its system prompt, initial message, and optional subagent
prompt. The crate owns only the generic loop contracts it interprets itself—compaction, final-turn
handling, and the fallback subagent protocol—which remain auditable Markdown under `prompts/`.

## Quick start

```rust
use nitpicker_agent::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let client = client_from_env(LLMProvider::Anthropic { base_url: None, api_key_env: None })?;

    let result = AgentBuilder::new(
        "explorer",
        "claude-sonnet-5",
        "You explore codebases.",
        client,
    )
    // optional: customize how spawned subagents behave (defaults to a generic prompt)
    .subagent_system_prompt("You are a focused file-reading worker. Report findings concisely.")
    .run("Map the module layout of this repo.", &file_agent_tools(), Path::new("."))
    .await?;

    println!("{}", result.text);
    Ok(())
}
```

See `examples/file_agent.rs` for a runnable version.

## What you control

- **Top-level prompt + task:** `AgentBuilder::new(name, model, system_prompt, client)` and the
  `initial_message` passed to `.run(...)`.
- **Tools:** any `HashMap<String, Arc<dyn Tool>>`. `file_agent_tools()` gives the file/git
  toolset plus `spawn_subagent`; `all_tools()` omits subagents; add your own `Tool` impls.
- **Subagents:** `subagent_system_prompt(...)` overrides the prompt spawned subagents run with;
  the override propagates through nested spawns. Depth is capped at 2.
- **Clients:** `client_from_env(LLMProvider::…)` for Anthropic / OpenAI / Gemini / OpenRouter
  via env-var API keys, or build any `Arc<dyn LLMClientDyn>` yourself. Config-file-driven
  construction is available via the `config` and `provider` modules.

## Features

- `azure` — `auth = "azure-ad"` for Azure AI Foundry (pulls in the Azure SDK; raises MSRV to 1.88).
- `antigravity` — compiles in the Gemini-proxy client hook + config validation (the proxy
  server itself lives in the `nitpicker` binary).

## Breaking changes

**0.3.0**
- Added `presets: Option<BTreeMap<String, PresetConfig>>` to `Config` and `presets: Option<Vec<String>>` to `DefaultsConfig` — source-breaking for code constructing these structs with literals (add `presets: None`). New `PresetConfig` type.
- Added `presets`/`lanes`/`jobs`/`error` optional fields to `AggregationRecord` and `model` to `ToolCallRecord` (same literal-construction caveat), plus the new `LaneRecord`/`JobRecord` types. An `error`-flagged aggregation record means synthesis failed post-collection — consumers must not render its `text` as a verdict.

**0.2.0**
- Replaced `AgentResult` token fields (`total_input_tokens`, `total_output_tokens`, `total_tokens`) and `usage()` method with `usage: TokenUsage`, which includes `cached_input_tokens` and `cache_creation_input_tokens`.
- `TokenUsage::input_tokens` now includes provider cache reads across all providers.
- Added `max_tokens: Option<u64>` to `AgentConfig` (defaults to `None`, using provider per-model limits; Anthropic defaults to 8192). `AgentBuilder::max_tokens` takes `NonZeroU64`.
- `AlloyClient::new` now takes `Vec<AlloySlot>` (`client`, `model`, `max_tokens`) to scope output caps per pooled model.

## License

MIT
