# nitpicker-agent

The reusable agentic core extracted from [nitpicker](https://github.com/arsenyinfo/nitpicker):
an LLM agent that reads files (`read_file` / `glob` / `grep` / read-only `git`) and fans out
parallel subagents, with provider-agnostic LLM clients and optional config-file-driven setup.

It carries none of nitpicker's CLI, review/debate, or PR machinery — just the loop, the tools,
and the providers.

## Quick start

```rust
use nitpicker_agent::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let client = client_from_env(LLMProvider::Anthropic { base_url: None, api_key_env: None })?;

    let result = AgentBuilder::new(
        "explorer",
        "claude-sonnet-4-6",
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

## License

MIT
