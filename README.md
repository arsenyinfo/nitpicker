# nitpicker

[![crates.io](https://img.shields.io/crates/v/nitpicker.svg)](https://crates.io/crates/nitpicker)

Multi-reviewer code review using LLMs. Spawns parallel agents with different models/prompts, aggregates their feedback into a final verdict. Supports two modes — parallel aggregation and actor-critic debate — across two task types: code review and free-form questions.

[**Free Web version**](https://arseny.info/nitpicker) is available for open source projects.

Each reviewer is an agentic loop that can call tools (read files, grep, glob, git commands) to explore the repo before writing its review. Review prompts now encourage a quick initial map, a short working plan, and early subagent delegation for disjoint investigations. Tool outputs include lightweight headers and clearer truncation/no-match messages so agents can reason about partial evidence more reliably. A separate aggregator model deduplicates and synthesizes the individual reviews into a final verdict.

## Requirements

- Rust toolchain
- A git repository to review
- At least one configured LLM (API key or Gemini OAuth)

## Installation

```bash
cargo install nitpicker
```

## Quick start

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Review

```bash
nitpicker
nitpicker --repo /path/to/repo
nitpicker --repo /path/to/repo --prompt "focus on src/api/"
nitpicker --analyze src/components/
nitpicker --analyze  # entire repo
```

### Parallel Mode

```bash
nitpicker --no-debate
nitpicker --no-debate --analyze src/
nitpicker --no-debate --max-turns 40
```

### PR review

```bash
nitpicker pr
nitpicker pr https://github.com/owner/repo/pull/42
nitpicker pr --no-comment
nitpicker pr https://github.com/owner/repo/pull/42 --no-comment
# force a fresh temp clone even when the URL points to your current repo
nitpicker pr https://github.com/owner/repo/pull/42 --clone
# machine-readable output for embedding (one JSON object on stdout)
nitpicker pr https://github.com/owner/repo/pull/42 --no-comment --json
```

### Ask

```bash
nitpicker ask "should we use eyre or thiserror for error handling?"
nitpicker ask --no-debate "is this authentication flow secure?"
nitpicker ask --rounds 3 "should we split this module?"
nitpicker ask --max-turns 40 "should we split this module?"
```

## Configuration

Configuration is loaded from (first match wins):

1. `--config <path>` (explicit flag)
2. `nitpicker.toml` in repo root
3. `~/.nitpicker/config.toml` (global config)

```bash
# create a config in current directory
nitpicker init

# prefer OpenRouter experimental free models when OPENROUTER_API_KEY is set
nitpicker init --free

# create a global config at ~/.nitpicker/config.toml
nitpicker init --global
```

Example `nitpicker.toml`:

```toml
[defaults]
debate = true          # optional, default: true
max_turns = 100        # optional, default: 100
log_trajectories = false # optional, default: false

[aggregator]
model = "claude-sonnet-5"
provider = "anthropic"
max_tokens = 16384       # optional, default: 16384

[[reviewer]]
name = "claude"          # used in output headers and logs
model = "claude-sonnet-5"
provider = "anthropic"
# max_tokens = 32768     # optional, default: unset (the provider's own per-model limit)

[[reviewer]]
name = "gpt"
model = "gpt-5.6-sol"
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
```

> **Tip:** Use providers that were not used for the initial building of your codebase to enforce diversity of thought.

Unknown config keys are rejected. For example, use `max_tokens` for output length; `token_limit` is not a supported field.

`max_tokens` caps a single response, and on a reasoning model it is a budget for reasoning *plus* the answer — set too low, the model spends it all thinking and returns empty content, which is indistinguishable from a model that said nothing. Reviewers therefore default to no cap (the provider applies its own per-model limit); set one only to bound spend. The aggregator writes one bounded synthesis and defaults to 16384. Two exceptions: Anthropic's API requires the field, so an unset reviewer cap becomes 8192 there — raise it explicitly if your model reasons past that; and `auth = "codex"` ignores the setting entirely, since that endpoint rejects `max_output_tokens`.

Debate mode is enabled by default for `nitpicker`, `nitpicker ask`, and `nitpicker pr`. Pass `--no-debate` to use parallel aggregation for a single run. Use `[defaults].max_turns` or `--max-turns` to control the per-agent tool-use loop limit.

Set `[defaults].log_trajectories = true` to save per-agent JSONL traces and a final `aggregation.json` under `~/.nitpicker/sessions/session-<timestamp>-<pid>/`.

### Provider types

| `provider` | Auth | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` env var (or `api_key_env`), or `auth = "azure-ad"` | `base_url` optional |
| `gemini` | `GEMINI_API_KEY`/`GOOGLE_AI_API_KEY` env var (or `api_key_env`), or `auth = "agy-keyring"` | `base_url` optional (e.g. a local Gemini-compatible server); `agy-keyring` reuses the Antigravity CLI OAuth token from the system keyring — research only, [see warning](#antigravity-keyring-research-only) |
| `openai` | `OPENAI_API_KEY` env var (or `api_key_env`), `auth = "azure-ad"`, or `auth = "codex"` | `codex` reuses your ChatGPT subscription via the Codex CLI token — research only, [see warning](#chatgptcodex-subscription-research-only) |
| `openrouter` | `OPENROUTER_API_KEY` env var (or `api_key_env`) | explicit model names are recommended; `model = "free"` is experimental |

`anthropic_compatible` and `openai_compatible` are accepted as aliases for backward compatibility.

`auth = "azure-ad"` authenticates with a refreshing Azure AD (Entra ID) token instead of a static key — for OpenAI and Anthropic models hosted on Azure AI Foundry. Requires a build with the `azure` feature, [see below](#azure-ad-azure-ai-foundry).

`auth = "codex"` authenticates with your ChatGPT Plus/Pro (Codex) subscription instead of a paid API key, reusing the token the Codex CLI stores on disk, [see below](#chatgptcodex-subscription-research-only).

### OpenRouter models

`openrouter` supports both explicit pinned models and an experimental free auto-selection mode.

Pinned models are the supported default and the recommended setup:

```toml
# recommended: explicit model
[[reviewer]]
name = "qwen"
model = "qwen/qwen3-30b-a3b"
provider = "openrouter"
```

Experimental best-effort free auto-selection is also available:

```toml
# experimental: auto-select a currently available free model
# omit `model` or set model = "free"
[[reviewer]]
provider = "openrouter"

# explicit experimental form
[[reviewer]]
model = "free"
provider = "openrouter"
```

When `model` is omitted or set to `"free"`, nitpicker tries to pick a currently working free model at startup.

This mode is convenient, but it is not production-stable and may fail due to upstream availability, routing differences, or timeouts.

If you want predictable behavior, pin explicit model names instead of relying on free auto-selection.

```bash
export OPENROUTER_API_KEY="your-key"
```

A free OpenRouter account is sufficient for the experimental free mode — no credit card required, just rate limits.

### Antigravity Keyring (research only)

> [!CAUTION]
> **Research only — do not use on a Google account you care about.**
> AG2's [Additional Terms of Service](https://antigravity.google/terms) Section 6 prohibits "using the Service in connection with products not provided by us", which directly covers reusing the `agy` OAuth token from a third-party client like nitpicker. Google has been actively enforcing this in 2026: paid AI Ultra subscribers have received account suspensions, often without warning, for using third-party AG2 OAuth bridges (OpenClaw, OpenCode, Pi Agent). Detection appears aggressive — even light testing has triggered bans. The earlier `gemini-cli` OAuth path was discouraged on similar grounds ([discussion](https://github.com/google-gemini/gemini-cli/discussions/22970)).
> If you want billed Gemini access without this risk, set `GEMINI_API_KEY` and drop the `auth` line.

AG2 is Google's current agentic IDE, succeeding both the older Gemini CLI OAuth path and the earlier AG1 preview. The `gemini-3.x` family ships only through AG2's CloudCode backend, so `auth = "agy-keyring"` exists purely as a research path to compare those models against the rest of the reviewer pool, with full awareness of the ToS posture above.

The proxy reads `agy`'s OAuth token from the system keyring (`service=gemini`, `account=antigravity`) via the `keyring` crate (Secret Service on Linux, Keychain on macOS, Credential Manager on Windows), relies on `agy` to refresh it, and routes chat through CloudCode's `v1internal:streamGenerateContent` SSE endpoint. Run `agy` and complete its login first. `NITPICKER_ANTIGRAVITY_PLATFORM` can override the auto-detected platform enum if needed.

This path requires a build with the `antigravity` feature (off by default, since it pulls in the local proxy stack — `axum` — and the `keyring` crate with its native backends):

```bash
cargo build --release --features antigravity
# or: cargo install --features antigravity ...
```

Without the feature, `auth = "agy-keyring"` is rejected at config validation with a build hint, and `nitpicker init` won't offer the keyring reviewer.

Tested AG2 models (current author config): `gemini-3.1-pro-low`, `gemini-3.5-flash-low`. Other IDs returned by `fetchAvailableModels` (e.g. `gemini-3-flash-agent`) likely work but have not been exercised.

```toml
[aggregator]
model = "gemini-3.5-flash-low"
provider = "gemini"
auth = "agy-keyring"

[[reviewer]]
name = "gemini"
model = "gemini-3.1-pro-low"
provider = "gemini"
auth = "agy-keyring"
```

### ChatGPT/Codex subscription (research only)

> [!CAUTION]
> **Research only.** This reuses the OpenAI Codex CLI's public OAuth client to call OpenAI through your ChatGPT Plus/Pro subscription. Third-party use of that client is arguably outside OpenAI's terms — same posture as the Antigravity path above. Use a paid `OPENAI_API_KEY` for anything you care about.

`auth = "codex"` (on an `openai` reviewer/aggregator) reuses the OAuth token the [Codex CLI](https://developers.openai.com/codex) stores in `~/.codex/auth.json`. Log in once with `codex login` (choosing your ChatGPT account, not an API key); nitpicker reads the token **read-only** and refreshes the short-lived access token in-memory via the refresh token — it never writes back to `auth.json`. Set `CODEX_HOME` to override the token directory.

Under the hood this talks to the Codex subscription endpoint (`chatgpt.com/backend-api/codex/responses`), which speaks the OpenAI Responses API with subscription-specific quirks (a required top-level system prompt, mandatory streaming, `store: false`, no `max_output_tokens`, and encrypted reasoning items round-tripped across turns since nothing is server-side persisted); nitpicker handles all of that transparently. No API-key env var is needed.

Models are your subscription's Codex models (e.g. `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`):

```toml
[aggregator]
model = "gpt-5.4"
provider = "openai"
auth = "codex"

[[reviewer]]
name = "codex"
model = "gpt-5.4"
provider = "openai"
auth = "codex"
```

### Azure AD (Azure AI Foundry)

Call OpenAI and Anthropic models hosted on [Azure AI Foundry](https://ai.azure.com) using a short-lived Azure AD (Entra ID) token instead of a static key. nitpicker acquires the token via the Azure SDK and transparently refreshes it (rebuilding the client before the token expires), so long reviews and debates don't die mid-run — the equivalent of the Python SDK's `azure_ad_token_provider`.

This path requires a build with the `azure` feature (off by default, since it pulls in the Azure SDK and needs Rust 1.88+):

```bash
cargo build --release --features azure
# or: cargo install --features azure ...
```

Set `auth = "azure-ad"` on an `openai` or `anthropic` reviewer/aggregator and point `base_url` at your Foundry endpoint:

```toml
[[reviewer]]
name = "gpt"
provider = "openai"                                                  # OpenAI models → /openai/v1
base_url = "https://<resource>.services.ai.azure.com/openai/v1"
model = "gpt-4o"                                                     # your Foundry deployment / model
auth = "azure-ad"

[[reviewer]]
name = "claude"
provider = "anthropic"                                               # Anthropic models → /anthropic
base_url = "https://<resource>.services.ai.azure.com/anthropic"
model = "claude-sonnet-4-5"
auth = "azure-ad"
azure_credentials = "dev"                                            # optional, see below
```

Optional per-reviewer/aggregator fields:

- `azure_scope` — AAD token scope. Defaults to `https://cognitiveservices.azure.com/.default`.
- `azure_credentials` — selects the credential chain, mirroring the Azure SDK's `AZURE_TOKEN_CREDENTIALS`:
  - `"dev"` — developer tools only (`az login`, Azure Developer CLI), excluding managed identity. Use on a VM where you want `az login` instead of a system-assigned managed identity.
  - `"prod"` — env service principal (`AZURE_TENANT_ID`/`AZURE_CLIENT_ID`/`AZURE_CLIENT_SECRET`), then managed identity.
  - unset / `"auto"` — env service principal → managed identity → developer tools, in that order.

  If unset, the `AZURE_TOKEN_CREDENTIALS` env var is honored as a fallback.

## CLI reference

```
nitpicker [OPTIONS]
nitpicker ask [--no-debate] [--rounds N] [--max-turns N] [OPTIONS] <topic>
nitpicker pr [URL] [--no-comment] [--no-debate] [--rounds N] [--max-turns N] [OPTIONS]
nitpicker init [--global] [--free]
```

### Review (default)

```
--repo <PATH>          git repository to review [default: .]
--config <PATH>        config file [default: <repo>/nitpicker.toml, then ~/.nitpicker/config.toml]
--prompt <TEXT>        review instructions (optional, has a sensible default)
--context-file <PATH>  inject a file's contents into the prompt; repeatable
--analyze [PATH]       analyze existing code instead of reviewing changes
--no-debate            use parallel aggregation instead of actor-critic debate
--rounds <N>           maximum debate rounds [default: 5]
--max-turns <N>        maximum tool-use turns per agent or debate turn [default: 100 via config]
-v, --verbose          show info-level logs (hidden by default)
```

#### `--context-file`

The agents' own tools are sandboxed to the repository, so they cannot open a design doc that lives
outside it. `--context-file` reads such a file directly into the prompt:

```bash
nitpicker --context-file ~/notes/migration-plan.md --context-file /tmp/rfc.md
```

Available on the default review mode, `ask`, and `pr`; the flag may be given before or after the
subcommand. Files are injected verbatim, in the order given, after the task and any `--prompt`
text. Total injected size — contents plus each block's wrapper — is capped at 256 KiB; a missing,
non-regular (FIFO, device node), binary, or non-UTF-8 file is an error, raised before any model is
called. Because the contents are placed in the task prompt rather than the system prompt, spawned
subagents do not inherit them.

### PR subcommand

```
nitpicker pr [URL] [--no-comment] [--no-debate] [--rounds N] [--max-turns N] [--prompt TEXT] [--context-file PATH] [--repo .] [--config PATH] [--json] [-v]
```

Reviews a GitHub PR using its title, description, and diff. Requires the `gh` CLI (`gh auth login` to authenticate).

- Without `URL`: reviews the current branch's open PR (must be run inside the repo)
- With `URL` (`https://github.com/owner/repo/pull/N`): clones the repo into a temp dir, checks out the PR branch, reviews it, then cleans up
- By default, posts the review as a PR comment. Pass `--no-comment` to skip posting.
- `--no-debate`, `--rounds`, and `--max-turns` work the same as in the default review mode
- `--json` emits a single machine-readable JSON object on stdout (status, PR metadata, models, `report_markdown`, `usage`, …) instead of the human report, with all logs/progress on stderr — handy for calling nitpicker as a subprocess. Exits non-zero on failure, with a `status: "error"` object on stdout. The `usage` block reports aggregate `input_tokens`/`output_tokens`/`total_tokens`, `cached_input_tokens`/`cache_creation_input_tokens`, and `subagents_spawned` for the run (best-effort: successful completions only). The cache fields are a breakdown of `input_tokens`, not an extra charge — a healthy multi-turn run shows most of its input served from cache. Cost formulas must discount them: `input_tokens` now counts cache reads on every provider (before 0.8.3 the Anthropic path omitted them), so a naive `input_tokens * input_price` over-states a cache-heavy run.

### Ask subcommand

```
nitpicker ask [--no-debate] [--rounds N] [--max-turns N] [--context-file PATH] [--repo .] [--config PATH] [-v] <topic>
```

 Runs agents on a free-form question instead of a code diff. By default, two agents take turns as Actor/Critic before a meta-reviewer concludes. Pass `--no-debate` to switch to the parallel reviewer plus aggregator flow.

### Debate mode (default)

Two LLM agents take turns exploring the codebase with file/git tools and submitting verdicts. The Critic can signal agreement (`agree=true`) to end early. A meta-reviewer synthesizes the dialogue.

- `reviewer[0]` in config → Actor (review: Reviewer)
- `reviewer[1]` in config → Critic (review: Validator)
- `aggregator` → Meta-reviewer

Interactive text runs show a compact cast/progress view while debating, then print the final synthesized result. In a terminal, `--verbose` also shows intermediate debate output and the saved transcript path; redirected stdout stays final-report-only.

Transcript saved to `{tempdir}/debate-{timestamp}.md` or `review-debate-{timestamp}.md`.

### Exit codes (default review and `ask`)

| code | meaning |
|------|---------|
| 0 | clean verdict |
| 1 | hard failure — no verdict (bad config, missing key, every reviewer/turn failed) |
| 2 | CLI usage error (clap's exit code for bad arguments) |
| 3 | degraded verdict — report printed, but a reviewer failed, or a debate turn failed or ended without calling `submit_verdict` |

Non-interactive, non-verbose stdout carries exactly the final report, so the binary can be driven as a subprocess: read stdout for the verdict, branch on the exit code. `pr` keeps its own contract (`--json` emits `status: ok|error` and exits 0/1).

## Using the agent as a library

The agentic core — the loop, the file/git tools, and the provider-agnostic LLM clients — lives in a separate crate, [`nitpicker-agent`](crates/nitpicker-agent), with none of the CLI/review/PR machinery. Use it to drive your own file-reading agent with subagents:

```rust
use nitpicker_agent::prelude::*;
use std::path::Path;

let client = client_from_env(LLMProvider::Anthropic { base_url: None, api_key_env: None })?;
let result = AgentBuilder::new("explorer", "claude-sonnet-5", "You explore codebases.", client)
    .subagent_system_prompt("You are a focused file-reading worker. Report findings concisely.")
    .run("Map the module layout of this repo.", &file_agent_tools(), Path::new("."))
    .await?;
println!("{}", result.text);
```

`file_agent_tools()` is the read-only file/git toolset plus `spawn_subagent`. You control the top-level prompt, the subagent prompt, the toolset, and the client; config-file-driven client construction is available via the `config`/`provider` modules. See `crates/nitpicker-agent/examples/file_agent.rs`.

## Changelog

**0.8.4** — 2026-08-10
- New `--context-file <PATH>` flag (repeatable) injects out-of-repo files (regular files only, capped at 256 KiB in total including block wrappers) into the task prompt for review, `ask`, and `pr`; accepted before or after the subcommand.
- `--repo`, `--config`, and `-v`/`--verbose` are now global flags: accepted before or after a subcommand, no longer silently ignored in the root position.

**0.8.3** — 2026-08-03 (`nitpicker-agent` 0.2.0)
- **Stable Tool Definitions**: Send tool definitions in name-sorted order to preserve request prefix determinism and maximize provider prompt-cache hits across turns.
- **Cache-Aware Token Metering**: Added `cached_input_tokens` and `cache_creation_input_tokens` to `TokenUsage` and `pr --json`, normalized `input_tokens` across provider cache shapes (fixing compaction triggers on auto-caching providers), and updated progress display (e.g. `1.0M in · 90% cached`).
- **Flexible Output Caps**: Removed hardcoded 8192-token output caps on agent turns and compaction (preventing reasoning models from exhausting token budgets on internal thinking before emitting output). Added `max_tokens` configuration per reviewer (`None` uses provider default; aggregator defaults to 16,384).
- **Improved Empty Response Diagnostics**: Surfaced `finish_reason` and token usage when retrying empty responses.
- **Unterminated Think Block Salvage**: Improved parsing to recover verdicts buried after unclosed `<think>` tags even when preceding content is present.
- **Resilient Compaction**: Made history compaction best-effort so a summarizer failure no longer aborts the review run.
- **Alloy Slot Cap Isolation**: Scoped `max_tokens` per model in `AlloyClient` pools.
- **Dependency & Cleanup**: Bumped `rig-core` to 0.41 and removed redundant `fc_` tool call ID normalization.
- **Breaking (`nitpicker-agent`)**: Replaced `AgentResult` token fields with `usage: TokenUsage`. `AlloyClient::new` now takes `Vec<AlloySlot>`.

**0.8.2** — 2026-07-14 (`nitpicker-agent` 0.1.2)
- Prevent debate subagents from inheriting `submit_verdict` tools to avoid premature convergence.
- Run history compaction under a dedicated summarizer system prompt.
- Wrap project context (`CLAUDE.md`/`AGENTS.md`) in `<context-only>` tags and pass task to parallel aggregator.
- Added `ReviewScope` (`Diff`/`Static`) and `--analyze` static-analysis mode.

**0.8.1** — 2026-07-03
- Bumped `rig-core` to 0.39; simplified Codex assistant text normalization.

**0.8.0** — 2026-06-16
- Extracted core agent logic into published [`nitpicker-agent`](crates/nitpicker-agent) crate.
- Added `subagent_system_prompt` customization to `AgentConfig`.

**0.7.1** — 2026-06-14
- Exit 3 for degraded review/ask verdicts; refined interactive progress rendering.
- Audit fixes: hardened git tool sandboxing, provider auth/retry error classification, and `pr` checkout safety.

**0.7.0** — 2026-06-07
- Added research `auth = "codex"` for OpenAI provider using local Codex CLI OAuth tokens.
- Fixed Codex multi-turn reasoning and gated keyring auth under `antigravity` feature flag.

**0.6.3** — 2026-06-06
- Security audit fixes: restricted git subcommands (`for-each-ref`/`show-ref`), panic-safe detached HEAD restore, retry classifier matching.

**0.6.2** — 2026-06-05
- `nitpicker pr --json` emits machine-readable output on stdout with logs/spinners on stderr.
- Fixed macOS symlink workspace canonicalization under `pr --clone`.

**0.6.1** — 2026-06-05
- Concurrent subagent execution: run `spawn_subagent` calls in parallel via `join_all` bounded by a global semaphore (default 16).

**0.6.0** — 2026-06-02
- Added `auth = "azure-ad"` support for OpenAI/Anthropic providers via Entra ID bearer tokens (gated under `azure` feature flag).

**0.5.1** — 2026-05-25
- PR checkout safety improvements, `--clone` flag, partial clones (`--filter=blob:none`).

**0.5.0** — 2026-05-24
- Added `auth = "agy-keyring"` for Gemini; removed legacy `auth = "oauth"`.

**0.4.0** — 2026-05-17
- Added Alloy mode (`--alloy`) to pool reviewer models into a shared random-selection pool.

**0.3.0 – 0.3.3** — 2026-05-06 – 2026-05-11
- OpenRouter free model auto-selection (`init --free`), trajectory logging, `reflect` analysis tool.

**0.1.2 – 0.2.3** — 2026-03-15 – 2026-05-01
- PR review subcommand, debate mode synthesis, proactive conversation compaction, bounded subagent runtime, atomic file locks.
