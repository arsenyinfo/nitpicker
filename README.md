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

# create a global config at ~/.nitpicker/config.toml
nitpicker init --global
```

Example `nitpicker.toml`:

```toml
[defaults]
debate = true          # optional, default: true
max_turns = 70         # optional, default: 70

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

### Provider types

| `provider` | Auth | Required fields |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` env var | — |
| `gemini` | `GEMINI_API_KEY` env var, or `auth = "oauth"` | — |
| `anthropic_compatible` | env var named by `api_key_env` | `base_url`, `api_key_env` |
| `openai_compatible` | env var named by `api_key_env` | `base_url`, `api_key_env` |

### Gemini OAuth

> [!WARNING]
> Google has stated that using Gemini CLI OAuth with third-party software is a
> policy-violating use case and may trigger abuse detection or account
> restrictions. It is unclear how aggressively this is enforced, but you should
> assume there is real risk and use this at your own discretion.
> See: https://github.com/google-gemini/gemini-cli/discussions/22970

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
nitpicker ask [--no-debate] [--rounds N] [--max-turns N] [OPTIONS] <topic>
nitpicker pr [URL] [--no-comment] [--no-debate] [--rounds N] [--max-turns N] [OPTIONS]
nitpicker init [--global]
```

### Review (default)

```
--repo <PATH>      git repository to review [default: .]
--config <PATH>    config file [default: <repo>/nitpicker.toml, then ~/.nitpicker/config.toml]
--prompt <TEXT>    review instructions (optional, has a sensible default)
--analyze [PATH]   analyze existing code instead of reviewing changes
--no-debate        use parallel aggregation instead of actor-critic debate
--rounds <N>       maximum debate rounds [default: 5]
--max-turns <N>    maximum tool-use turns per agent or debate turn [default: 70 via config]
--gemini-oauth     run Gemini OAuth authentication flow and exit
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

Transcript saved to `{tempdir}/debate-{timestamp}.md` or `review-debate-{timestamp}.md`.

## Changelog

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
