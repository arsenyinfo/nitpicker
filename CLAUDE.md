# nitpicker — codebase guide

## Build & run

```bash
cargo check                          # fast type-check
cargo build                          # debug build
cargo build --release                # release build
cargo run -- --repo .                # run against current repo
cargo run -- --gemini-oauth          # trigger Gemini auth flow
RUST_LOG=nitpicker=debug cargo run -- --repo .
```

## Architecture

```
main.rs         CLI, config loading, wires everything together
config.rs       TOML config deserialization (Config, ReviewerConfig, AggregatorConfig)
review.rs       orchestrates parallel reviewers → aggregation
agent.rs        agentic tool-use loop for a single reviewer
llm.rs          LLM client trait, per-provider impls, retry wrapper
tools.rs        tool definitions: read_file, glob, grep, git
gemini_proxy/   local HTTP proxy that translates Gemini API calls to Google Code Assist
```

### Request flow

1. `review.rs` spawns one `tokio::task` per `[[reviewer]]`
2. Each task runs `agent.rs::run_agent` — an agentic loop: call LLM → execute tool calls → feed results back → repeat until the model returns text (max 100 turns)
3. All reviewer outputs are collected, concatenated, and sent to the aggregator model in a single completion call
4. The aggregator's response is printed to stdout

### LLM abstraction (`llm.rs`)

- `LLMClient` trait: one method, `completion(Completion) -> Result<CompletionResponse>`
- Per-provider impls: `anthropic::Client`, `gemini::Client`, `openai::CompletionsClient`
- `RetryingLLM<C>` wraps any client with jittered exponential backoff (4 attempts, 250ms–5s). Skips retry on 4xx errors.
- Always wrap clients with `.with_retry()` — the OAuth Gemini path is no exception

### Tools (`tools.rs`)

Tools return `String`, never `Err` — errors are returned as `"Error: ..."` strings so the LLM can self-correct. The exception is truly unrecoverable errors (e.g. missing required argument).

`GitTool` only allows a fixed allowlist of read-only subcommands. Commands are passed directly to `Command::new("git").args(tokens)` — no shell involved.

### Gemini OAuth proxy (`gemini_proxy/`)

When `auth = "oauth"` is set for a Gemini reviewer/aggregator, nitpicker:
1. Runs a local axum HTTP server on a random port
2. Translates incoming Gemini API requests to Google Code Assist API format
3. Attaches a valid OAuth Bearer token (refreshed automatically)

The OAuth client credentials in `mod.rs` are Google's public installed-app credentials — intentionally public, same pattern as `gcloud` CLI.

Token stored at `~/.nitpicker/gemini-token.json` with `0o600` permissions.

## Adding a new provider

1. Add a variant to `ProviderType` in `config.rs` with a `#[serde(rename = "...")]`
2. Add a new arm to `provider_from_config` in `review.rs`
3. Add a new variant to `LLMProvider` in `llm.rs` and implement `client_from_env`
4. Implement `LLMClient` for the provider's client type

## Key constraints

- Reviewers run concurrently — reviewer code must be `Send + Sync`
- Tool results are truncated to 50k bytes before being sent to the LLM
- Git tool output is truncated to 50k chars
- Agent loop is capped at 100 turns per reviewer
