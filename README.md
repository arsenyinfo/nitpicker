# nitpicker

[![crates.io](https://img.shields.io/crates/v/nitpicker.svg)](https://crates.io/crates/nitpicker)

Multi-reviewer code review using LLMs. Spawns parallel agents with different models/prompts, aggregates their feedback into a final verdict. Supports two modes — parallel aggregation and actor-critic debate — across two task types: code review and free-form questions.

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
model = "claude-sonnet-4-6"
provider = "anthropic"
max_tokens = 8192        # optional, default: 8192

[[reviewer]]
name = "claude"          # used in output headers and logs
model = "claude-sonnet-4-6"
provider = "anthropic"

[[reviewer]]
name = "gpt"
model = "gpt-5.2-codex"
provider = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
```

> **Tip:** Use providers that were not used for the initial building of your codebase to enforce diversity of thought.

Unknown config keys are rejected. For example, use `max_tokens` for output length; `token_limit` is not a supported field.

Debate mode is enabled by default for `nitpicker`, `nitpicker ask`, and `nitpicker pr`. Pass `--no-debate` to use parallel aggregation for a single run. Use `[defaults].max_turns` or `--max-turns` to control the per-agent tool-use loop limit.

Set `[defaults].log_trajectories = true` to save per-agent JSONL traces and a final `aggregation.json` under `~/.nitpicker/sessions/session-<timestamp>-<pid>/`.

### Provider types

| `provider` | Auth | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` env var (or `api_key_env`), or `auth = "azure-ad"` | `base_url` optional |
| `gemini` | `GEMINI_API_KEY` env var, or `auth = "agy-keyring"` | `agy-keyring` reuses the Antigravity CLI OAuth token from the system keyring — research only, [see warning](#antigravity-keyring-research-only) |
| `openai` | `OPENAI_API_KEY` env var (or `api_key_env`), or `auth = "azure-ad"` | `base_url` optional |
| `openrouter` | `OPENROUTER_API_KEY` env var (or `api_key_env`) | explicit model names are recommended; `model = "free"` is experimental |

`anthropic_compatible` and `openai_compatible` are accepted as aliases for backward compatibility.

`auth = "azure-ad"` authenticates with a refreshing Azure AD (Entra ID) token instead of a static key — for OpenAI and Anthropic models hosted on Azure AI Foundry. Requires a build with the `azure` feature, [see below](#azure-ad-azure-ai-foundry).

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
--repo <PATH>      git repository to review [default: .]
--config <PATH>    config file [default: <repo>/nitpicker.toml, then ~/.nitpicker/config.toml]
--prompt <TEXT>    review instructions (optional, has a sensible default)
--analyze [PATH]   analyze existing code instead of reviewing changes
--no-debate        use parallel aggregation instead of actor-critic debate
--rounds <N>       maximum debate rounds [default: 5]
--max-turns <N>    maximum tool-use turns per agent or debate turn [default: 100 via config]
-v, --verbose      show info-level logs (hidden by default)
```

### PR subcommand

```
nitpicker pr [URL] [--no-comment] [--no-debate] [--rounds N] [--max-turns N] [--prompt TEXT] [--repo .] [--config PATH] [-v]
```

Reviews a GitHub PR using its title, description, and diff. Requires the `gh` CLI (`gh auth login` to authenticate).

- Without `URL`: reviews the current branch's open PR (must be run inside the repo)
- With `URL` (`https://github.com/owner/repo/pull/N`): clones the repo into a temp dir, checks out the PR branch, reviews it, then cleans up
- By default, posts the review as a PR comment. Pass `--no-comment` to skip posting.
- `--no-debate`, `--rounds`, and `--max-turns` work the same as in the default review mode

### Ask subcommand

```
nitpicker ask [--no-debate] [--rounds N] [--max-turns N] [--repo .] [--config PATH] [-v] <topic>
```

 Runs agents on a free-form question instead of a code diff. By default, two agents take turns as Actor/Critic before a meta-reviewer concludes. Pass `--no-debate` to switch to the parallel reviewer plus aggregator flow.

### Debate mode (default)

Two LLM agents take turns exploring the codebase with file/git tools and submitting verdicts. The Critic can signal agreement (`agree=true`) to end early. A meta-reviewer synthesizes the dialogue.

- `reviewer[0]` in config → Actor (review: Reviewer)
- `reviewer[1]` in config → Critic (review: Validator)
- `aggregator` → Meta-reviewer

By default, nitpicker prints only the final synthesized result. Use `--verbose` to show intermediate debate output and the saved transcript path.

Transcript saved to `{tempdir}/debate-{timestamp}.md` or `review-debate-{timestamp}.md`.

## Changelog

**0.6.0** — 2026-06-02
- Added `auth = "azure-ad"` for the `openai` and `anthropic` providers: authenticate to Azure AI Foundry-hosted models with a refreshing Azure AD (Entra ID) bearer token instead of a static key. Token acquisition uses the Azure SDK (`DefaultAzureCredential`-style chain) and is transparently refreshed before expiry. New optional config fields `azure_scope` and `azure_credentials` (the latter mirrors `AZURE_TOKEN_CREDENTIALS`: `dev`/`prod`/auto). Gated behind the off-by-default `azure` cargo feature (build with `--features azure`; requires Rust 1.88+). See [Azure AD section](#azure-ad-azure-ai-foundry).
- `auth = "azure-ad"` configs are validated up front: `Config::validate` now requires a non-empty `base_url`, rejects an unknown `azure_credentials`, and surfaces typo'd `auth` values on any non-Gemini provider instead of failing cryptically at the first LLM call. Retry/401 classification was also fixed to inspect the full error chain, so token-expiry 401s actually trigger a refresh-and-retry.

**0.5.1** — 2026-05-25
- `pr` checkout safety: skip checkout when already on PR head, new `--clone` flag to force a fresh temp clone, lock now acquired before any git mutation and works correctly on macOS
- Temp-clone PR review uses partial clone (`--filter=blob:none`) so the diff base is always reachable

**0.5.0** — 2026-05-24
- Added `auth = "agy-keyring"` for the Gemini provider: reads the Antigravity (`agy`) CLI OAuth token from the system keyring and routes through AG2's CloudCode SSE endpoint. Treat as research only — AG2 ToS Section 6 prohibits third-party OAuth clients and Google has been actively suspending paid accounts for this pattern in 2026. See README warning before using.
- **Breaking:** removed the legacy `auth = "oauth"` browser flow and the `nitpicker --gemini-oauth` CLI flag. The flow had been broken since the proxy was retargeted at AG2 (the matching AG2 client_secret is not public, so token exchange could not complete). The config validator now rejects `auth = "oauth"` with a migration hint to `agy-keyring` or `GEMINI_API_KEY`.

**0.4.0** — 2026-05-17
- Alloy mode (`--alloy` / `defaults.alloy = true`): pools all reviewer models into a shared random-selection pool so every debate turn can draw from any configured model (based on [XBOW technique](https://xbow.com/blog/alloy-agents))

**0.3.3** — 2026-05-11: 
- init --free flag for OpenRouter free model auto-selection

**0.3.2** — 2026-05-11
- More reliable support for free OpenRouter models
- Compaction mechanism improvements

**0.3.1** — 2026-05-07
- Better tool guidance and self-describing tool outputs
- Review prompts now encourage short plans, earlier subagent delegation, and stricter evidence checks

**0.3.0** — 2026-05-06
- Session trajectory logging plus internal `reflect` analysis tooling
- Minor fixes, including graceful shutdown once turns are exhausted and allowing `pr` to review a URL from a non-git cwd

**0.2.3** — 2026-05-01
- Better `nitpicker init` experience and minor bug fixes

**0.2.2** — 2026-04-30
- Session artifacts now capture tool trajectories and final aggregation output for debugging
- Default CLI output now stays focused on the final synthesized review unless `--verbose` is enabled

**0.2.1** — 2026-04-30
- First class support for OpenRouter: new provider type, and experimental free auto-selection mode
- Minor bug fixes (breaking the cycle, temperature params)

**0.2.0** — 2026-04-28
- PR comments are now included in the review prompt for full context
- Per-repo file lock prevents concurrent `nitpicker pr` runs on the same repository
- Atomic lock acquisition with stale lock detection (no TOCTOU race)

**0.1.4** — 2026-04-23
- Proactive conversation compaction to prevent context overflow mid-review
- Claude Opus 4.7 compatibility fix
- Base branch detection fix for repos with `master` as default

**0.1.3** — 2026-04-22
- Bounded subagent runtime for debate and review
- Configurable turn limits (`--max-turns`, `[defaults].max_turns`)
- Kimi repeating tool-call fix

**0.1.2** — 2026-03-15
- `pr` subcommand: review GitHub PRs and post result as comment
- Debate synthesis posted as PR comment
- Three-dot diff for accurate stale-branch diffs
- Global config (`nitpicker init --global`)
- Rate limit handling with backoff
