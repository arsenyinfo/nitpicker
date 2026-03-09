# nitpicker

Multi-reviewer code review using LLMs. Spawns parallel agents with different models/prompts, aggregates their feedback into a final verdict.

## Quick start

```bash
# Review current PR/diff (map-reduce)
cargo run -- --repo .

# Debate the diff (actor-critic, requires ≥2 reviewers)
cargo run -- --repo . --debate
cargo run -- --repo . --debate --rounds 3

# Static analysis of existing code
cargo run -- --repo . --analyze
cargo run -- --repo . --analyze src/db/

# Custom focus
cargo run -- --repo . --prompt "focus on SQL injection"

# Ask a free-form question (map-reduce: parallel answers, aggregated)
cargo run -- ask "should we use eyre or thiserror?"

# Ask with debate (actor-critic dialogue, then meta-review)
cargo run -- ask --debate "should we use eyre or thiserror?"

# Gemini OAuth (first-time setup)
cargo run -- --gemini-oauth
```

## Architecture

```
main.rs         CLI, config loading, wires everything together
config.rs       TOML config deserialization (Config, ReviewerConfig, AggregatorConfig)
review.rs       orchestrates parallel reviewers → aggregation
debate.rs       sequential actor/critic debate loop → meta-review
agent.rs        agentic tool-use loop for a single reviewer
llm.rs          LLM client trait, per-provider impls, retry wrapper
tools.rs        tool definitions: read_file, glob, grep, git
gemini_proxy/   local HTTP proxy that translates Gemini API calls to Google Code Assist
```

### Review flow

1. `review.rs` spawns one `tokio::task` per `[[reviewer]]` in config
2. Each task runs `agent.rs::run_agent` — an agentic loop: call LLM → execute tool calls → feed results back → repeat until the model returns text (max 100 turns)
3. All reviewer outputs are collected, concatenated, and sent to the aggregator model in a single completion call
4. The aggregator's response is printed to stdout

### Debate flow (`--debate` / `ask --debate`)

1. `reviewer[0]` = Actor/Reviewer, `reviewer[1]` = Critic/Validator, `aggregator` = Meta-reviewer
2. Each round: Actor turn → Critic turn. Both have access to all file/git tools plus `submit_verdict(verdict, agree)`
3. `agree=true` from Critic → convergence, loop ends early
4. After all rounds: meta-reviewer synthesizes the full dialogue in a single non-agentic completion
5. Transcript saved to `debate-{ts}.md` (topic) or `review-debate-{ts}.md` (code review)
6. `DebateMode::Topic` (from `ask --debate`) uses Actor/Critic roles and general debate prompts
7. `DebateMode::Review` (from `--debate`) uses Reviewer/Validator roles and code-review-focused prompts

### LLM abstraction (`llm.rs`)

- `LLMClient` trait: one method, `completion(Completion) -> Result<CompletionResponse>`
- Per-provider impls: `anthropic::Client`, `gemini::Client`, `openai::CompletionsClient`
- `RetryingLLM<C>` wraps any client with jittered exponential backoff (4 attempts, 250ms–5s). Skips retry on 4xx errors.
- Always wrap clients with `.with_retry()` — the OAuth Gemini path is no exception

### Tools (`tools.rs`)

Tools return `String`, never `Err` — errors are returned as `"Error: ..."` strings so the LLM can self-correct. The exception is truly unrecoverable errors (e.g. missing required argument).

`GitTool` only allows a fixed allowlist of read-only subcommands. Commands are passed directly to `Command::new("git").args(tokens)` — no shell involved.

`GrepTool` recursively searches files, detecting binary files by checking for null bytes in the first 8 KiB.

### Gemini OAuth proxy (`gemini_proxy/`)

When `auth = "oauth"` is set for a Gemini reviewer/aggregator, nitpicker:
1. Runs a local axum HTTP server on a random port
2. Translates incoming Gemini API requests to Google Code Assist API format
3. Attaches a valid OAuth Bearer token (refreshed automatically)

The OAuth client credentials in `mod.rs` are Google's public installed-app credentials — intentionally public, same pattern as `gcloud` CLI.

Token stored at `~/.nitpicker/gemini-token.json` with `0o600` permissions.

## Configuration

Config hierarchy (first wins):
1. `--config <path>` (explicit)
2. `nitpicker.toml` in repo root
3. `~/.nitpicker/config.toml` (global)

Reviewers automatically load project context from `CLAUDE.md` or `AGENTS.md` if present in the repo root.

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
- Context files (`CLAUDE.md`, `AGENTS.md`) are limited to 50k chars