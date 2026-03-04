# nitpicker

Runs multiple LLM reviewers concurrently on a git repository, then aggregates their findings into a single prioritized report.

Each reviewer is an agentic loop that can call tools (read files, grep, glob, git commands) to explore the repo before writing its review. A separate aggregator model deduplicates and synthesizes the individual reviews into a final verdict.

## Requirements

- Rust toolchain
- A git repository to review
- At least one configured LLM (API key or Gemini OAuth)

## Installation

```bash
cargo install --git https://github.com/arsenyinfo/nitpicker
```

## Quick start

```bash
# review the last commit in the current repo (requires nitpicker.toml at repo root)
nitpicker

# review a specific repo
./target/release/nitpicker --repo /path/to/repo

# custom prompt
./target/release/nitpicker --repo /path/to/repo --prompt "Review only src/api/"
```

## Configuration

Place `nitpicker.toml` at the root of the repo being reviewed, or pass `--config /path/to/file.toml`.

```toml
[aggregator]
model = "claude-opus-4-6"
provider = "anthropic"
max_tokens = 8192        # optional, default: no limit

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

### Provider types

| `provider` | Auth | Required fields |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` env var | — |
| `gemini` | `GEMINI_API_KEY` env var, or `auth = "oauth"` | — |
| `anthropic_compatible` | env var named by `api_key_env` | `base_url`, `api_key_env` |
| `openai_compatible` | env var named by `api_key_env` | `base_url`, `api_key_env` |

### Gemini OAuth 

Gemini can be used via Google Code Assist OAuth (for free or with subscription, limits apply) — no API key needed, just a Google account.

```toml
[aggregator]
model = "gemini-2.5-pro"
provider = "gemini"
auth = "oauth"

[[reviewer]]
name = "gemini"
model = "gemini-2.5-pro"
provider = "gemini"
auth = "oauth"
```

Authenticate once before reviewing:

```bash
nitpicker --gemini-oauth
```

This opens a browser, completes the OAuth flow, and saves the token to `~/.nitpicker/gemini-token.json`. The token is refreshed automatically on subsequent runs.

## CLI reference

```
--repo <PATH>      git repository to review [default: .]
--config <PATH>    config file [default: <repo>/nitpicker.toml]
--prompt <TEXT>    review instructions (optional, has a sensible default)
--gemini-oauth     run Gemini OAuth authentication flow and exit
```
