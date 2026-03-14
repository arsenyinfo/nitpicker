# nitpicker

Multi-reviewer code review using LLMs. Spawns parallel agents with different models/prompts, aggregates their feedback into a final verdict. Supports two modes — parallel aggregation and actor-critic debate — across two task types: code review and free-form questions.

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
# review current PR/diff (requires nitpicker.toml at repo root)
nitpicker

# review a specific repo
nitpicker --repo /path/to/repo

# customized prompt
nitpicker --repo /path/to/repo --prompt "focus on src/api/"

# analyze existing code (no PR/diff required)
nitpicker --analyze src/components/
nitpicker --analyze src/main.rs
nitpicker --analyze  # analyze entire repo

# debate mode: two agents argue about the diff, meta-reviewer synthesizes (requires ≥2 reviewers)
nitpicker --debate
nitpicker --debate --analyze src/  # debate about existing code
nitpicker --debate --rounds 3

# review the current branch's open PR and post result as a comment
nitpicker pr

# review a specific PR by URL and post result as a comment
nitpicker pr https://github.com/owner/repo/pull/42

# review a PR without posting a comment
nitpicker pr --no-comment
nitpicker pr https://github.com/owner/repo/pull/42 --no-comment

# ask a free-form question (parallel: multiple opinions aggregated)
nitpicker ask "should we use eyre or thiserror for error handling?"

# ask with debate (actor-critic dialogue, then meta-review)
nitpicker ask --debate "is this authentication flow secure?"
nitpicker ask --debate --rounds 3 "should we split this module?"
```

## Configuration

Configuration is loaded from (first match wins):

1. `--config <path>` (explicit flag)
2. `nitpicker.toml` in repo root
3. `~/.nitpicker/config.toml` (global config)

```bash
# create a config in current directory
nitpicker init

# create a global config at ~/.nitpicker/config.toml
nitpicker init --global
```

Example `nitpicker.toml`:

```toml
[defaults]
debate = false         # optional, default: false

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

It is recommended to use providers that were not used for the initial building to enforce diversity of thought. 

Unknown config keys are rejected. For example, use `max_tokens` for output length; `token_limit` is not a supported field.

Set `[defaults].debate = true` to enable debate mode by default for both `nitpicker` and `nitpicker ask`. Passing `--debate` still works and explicitly enables debate for a single run.

### Provider types

| `provider` | Auth | Required fields |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` env var | — |
| `gemini` | `GEMINI_API_KEY` env var, or `auth = "oauth"` | — |
| `anthropic_compatible` | env var named by `api_key_env` | `base_url`, `api_key_env` |
| `openai_compatible` | env var named by `api_key_env` | `base_url`, `api_key_env` |

### Gemini OAuth 

Gemini can be used via Google Code Assist OAuth (for free or with subscription, limits apply) — no API key needed, just a Google account. This approach mimics the auth of [Gemini CLI](https://geminicli.com/), so no guarantees on reliability.

```toml
[aggregator]
model = "gemini-3-flash-preview"
provider = "gemini"
auth = "oauth"

[[reviewer]]
name = "gemini"
model = "gemini-3.1-pro-preview"
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
nitpicker [OPTIONS]
nitpicker ask [--debate] [--rounds N] [OPTIONS] <topic>
nitpicker pr [URL] [--no-comment] [--debate] [--rounds N] [OPTIONS]
nitpicker init [--global]
```

### Review (default)

```
--repo <PATH>      git repository to review [default: .]
--config <PATH>    config file [default: <repo>/nitpicker.toml, then ~/.nitpicker/config.toml]
--prompt <TEXT>    review instructions (optional, has a sensible default)
--analyze [PATH]   analyze existing code instead of reviewing changes
--debate           use actor-critic debate instead of parallel aggregation
--rounds <N>       maximum debate rounds (only with --debate) [default: 5]
--gemini-oauth     run Gemini OAuth authentication flow and exit
-v, --verbose      show info-level logs (hidden by default)
```

### PR subcommand

```
nitpicker pr [URL] [--no-comment] [--debate] [--rounds N] [--prompt TEXT] [--repo .] [--config PATH] [-v]
```

Reviews a GitHub PR using its title, description, and diff. Requires the `gh` CLI (`gh auth login` to authenticate).

- Without `URL`: reviews the current branch's open PR (must be run inside the repo)
- With `URL` (`https://github.com/owner/repo/pull/N`): clones the repo into a temp dir, checks out the PR branch, reviews it, then cleans up
- By default, posts the review as a PR comment. Pass `--no-comment` to skip posting.
- `--debate` and `--rounds` work the same as in the default review mode (debate output is not posted as a comment yet)

### Ask subcommand

```
nitpicker ask [--debate] [--rounds N] [--repo .] [--config PATH] [-v] <topic>
```

Runs agents on a free-form question instead of a code diff. Without `--debate`, agents answer in parallel and an aggregator synthesizes their responses. With `--debate`, two agents take turns as Actor/Critic before a meta-reviewer concludes.

### Debate mode (`--debate`)

Two LLM agents take turns exploring the codebase with file/git tools and submitting verdicts. The Critic can signal agreement (`agree=true`) to end early. A meta-reviewer synthesizes the dialogue.

- `reviewer[0]` in config → Actor (review: Reviewer)
- `reviewer[1]` in config → Critic (review: Validator)
- `aggregator` → Meta-reviewer

Transcript saved to `{tempdir}/debate-{timestamp}.md` or `review-debate-{timestamp}.md`.
