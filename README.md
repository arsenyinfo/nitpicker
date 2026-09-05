# nitpicker

[![crates.io](https://img.shields.io/crates/v/nitpicker.svg)](https://crates.io/crates/nitpicker)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Multi-model adversarial code review for your terminal and CI.

Supported by [Archestra](https://github.com/archestra-ai/archestra) · [**Free Web Version**](https://arseny.info/nitpicker) for open source projects

---

![Actor-Critic Architecture](assets/actor-critic.svg)

Most LLM code reviews suffer from two failure modes: **lazy rubber-stamping** (*"Looks good! 🎉"*) or **hallucinated pedantry** (inventing fake bugs because the model lacks repo context).

**nitpicker** fixes this by pitching models against each other in an adversarial debate:

* **Actor-Critic Dynamics**: The **Reviewer (Actor)** explores the codebase to maximize recall (finding plausible bugs and edge cases). The **Validator (Critic)** aggressively attempts to falsify those claims using actual repo evidence. Disputed claims are debated; only surviving findings make it to the report.
* **Deep Repo Exploration**: Reviewers don't just stare at a 40-line `git diff`. They run tools (`read_file`, `grep`, `glob`, `git`) and spawn concurrent subagents to trace call graphs, check types, and verify assumptions before claiming an issue exists.
* **Model Diversity & Economics**: Pit different model families against each other (Claude vs. GPT vs. Gemini), or pair frontier models with fast, inexpensive models via OpenRouter (Qwen, DeepSeek, Kimi). Burn 1M tokens reviewing a complex PR without burning your monthly budget.
* **Production & CI Ready**: Regularly used across several engineering orgs, including CI pipelines. Features deterministic exit codes (`0`, `1`, `3`), headless `--json` output, and OpenTelemetry tracing.

---

## Quick Start

```bash
cargo install nitpicker
```

Set an API key for your preferred provider:

```bash
export ANTHROPIC_API_KEY="your-api-key"
# or: export OPENROUTER_API_KEY="your-api-key"
# or: export OPENAI_API_KEY="your-api-key"
```

Run a review:

```bash
# Review your current branch / uncommitted changes
nitpicker

# Review an open GitHub pull request and post the verdict as a comment
nitpicker pr https://github.com/owner/repo/pull/42

# Ask a technical or architectural question about the repo
nitpicker ask "should we use eyre or thiserror for error handling?"
```

---

## Common Workflows

### 1. Review Local Diffs
By default, nitpicker compares your current branch against its base branch using an adversarial debate:

```bash
nitpicker                                    # review current diff
nitpicker --repo /path/to/repo               # review another repo
nitpicker --prompt "focus on SQL injection"  # add custom reviewer instructions
nitpicker --no-debate                        # use parallel reviewers instead of debate
nitpicker --fallback                         # auto-failover to next reviewer if a model hits rate limits
```

### 2. GitHub PR Reviews & CI
Review PRs locally or in automated CI workflows (requires GitHub CLI `gh`):

```bash
# Review current branch's PR and comment on GitHub
nitpicker pr

# Review any remote PR by URL (clones into temp dir, reviews, cleans up)
nitpicker pr https://github.com/owner/repo/pull/42

# Review without posting a comment to GitHub
nitpicker pr https://github.com/owner/repo/pull/42 --no-comment

# Machine-readable output for CI pipelines (single JSON object on stdout)
nitpicker pr https://github.com/owner/repo/pull/42 --no-comment --json
```

### 3. Ask Architecture & Design Questions
Debate design choices or sanity-check logic across your codebase:

```bash
nitpicker ask "is our token refresh flow thread-safe?"
nitpicker ask --rounds 3 "should we split this crate into a workspace?"
nitpicker ask --no-debate "how does configuration loading work?"
```

### 4. Audit Existing Code (Static Analysis)
Audit existing codebases without needing an active diff:

```bash
nitpicker --analyze src/auth/                # audit a specific directory
nitpicker --analyze                          # audit the entire repository
```

---

## Review Lenses (Presets)

Instead of generic feedback, review against targeted rubrics. Presets control **what** to investigate, while debate or parallel modes control **how**.

```bash
nitpicker --preset security                  # focus strictly on vulnerabilities
nitpicker --preset security,performance      # run both security and performance lanes
nitpicker --preset ai-systems                # audit agent prompts, tools, and context boundaries
nitpicker --preset ml-rigor                  # audit ML pipelines, data leakage, and eval hygiene
```

### Built-in Presets

| Preset | Focus Area |
|---|---|
| `correctness` | Logic bugs, race conditions, edge cases, error handling, off-by-one errors. *(Default)* |
| `security` | Injection, auth boundaries, input validation, secret leaks, SSRF. *(Default)* |
| `performance` | Unnecessary allocations, N+1 queries, async blocking, algorithmic complexity. *(Default)* |
| `simplicity` | Over-engineering, dead code, redundant abstractions, readability. *(Default)* |
| `ai-systems` | Agent tool schemas, prompt injection vectors, context truncation, deterministic fallbacks. *(Opt-in)* |
| `ml-rigor` | Training/validation leakage, metric gaming, distribution shift, silent failures. *(Opt-in)* |
| `tone` | Clarity, audience fit, terminology consistency, documentation voice. *(Opt-in)* |
| `general` | Broad standalone review for unconventional targets. Cannot be combined with other presets. |

You can also define custom presets in `nitpicker.toml`:

```toml
[presets.api-security]
prompt = """
Review API trust boundaries, JWT verification, and rate-limiting.
Require a concrete attacker-controlled call path for every finding.
"""
```

---

## Configuration

Initialize a configuration file:

```bash
nitpicker init             # creates ./nitpicker.toml
nitpicker init --global    # creates ~/.nitpicker/config.toml
nitpicker init --free      # configures free OpenRouter models
```

Configuration resolution order: `--config <path>` > `./nitpicker.toml` > `~/.nitpicker/config.toml`.

### Example `nitpicker.toml`

```toml
[defaults]
debate = true          # use Actor-Critic debate (default: true)
fallback = true        # failover to next reviewer if a provider rate-limits (default: false)
max_turns = 100        # tool-use loop limit per agent
# presets = ["correctness", "security"]  # default includes performance and simplicity

# Model that synthesizes the final report from surviving debate findings
[aggregator]
model = "claude-sonnet-5"
provider = "anthropic"
max_tokens = 16384

# Reviewer 1 (Actor / Discovery)
[[reviewer]]
name = "claude"
model = "claude-sonnet-5"
provider = "anthropic"

# Reviewer 2 (Critic / Validator)
[[reviewer]]
name = "gpt"
model = "gpt-5.6-sol"
provider = "openai"

# Reviewer 3 (Fast / Inexpensive via OpenRouter)
[[reviewer]]
name = "qwen"
model = "qwen/qwen3-30b-a3b"
provider = "openrouter"
```

> **Tip:** Pair different model families to avoid shared blind spots.

---

## Providers & Authentication

| Provider | Authentication | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Automatic 5-minute prompt caching enabled by default. |
| `openai` | `OPENAI_API_KEY` | Compatible with OpenAI models and custom OpenAI gateways. |
| `codex` *(OpenAI)* | Reuses `~/.codex/auth.json` | **Popular**: Uses your existing ChatGPT Plus/Pro subscription token from OpenAI Codex CLI. No API key needed. [Details below](#chatgpt--codex-subscription). |
| `openrouter` | `OPENROUTER_API_KEY` | Access open-weights & Chinese frontier models (Qwen, DeepSeek, Kimi, GLM). |
| `gemini` | `GEMINI_API_KEY` | Google Gemini models via official API. |
| `azure` *(Entra ID)* | `auth = "azure-ad"` | Azure AI Foundry models with auto-refreshing tokens (requires `--features azure`). |

### ChatGPT / Codex Subscription
If you have a ChatGPT Plus or Pro subscription and the [OpenAI Codex CLI](https://developers.openai.com/codex) installed, nitpicker can authenticate through your subscription without a separate paid API key:

```bash
# 1. Log in once via the official Codex CLI
codex login

# 2. Configure nitpicker to reuse the token (read-only)
```

```toml
[[reviewer]]
name = "codex"
model = "gpt-5.6-sol"
provider = "openai"
auth = "codex"
```

### OpenRouter & Model Economics
For high-volume reviews, OpenRouter gives you access to cost-effective models with large context windows:

```toml
[[reviewer]]
name = "deepseek"
model = "deepseek/deepseek-v4-pro"
provider = "openrouter"

# Or experimental auto-selection of currently available free models:
[[reviewer]]
model = "free"
provider = "openrouter"
```

---

## Production & CI Integration

### Deterministic Exit Codes

nitpicker uses standard exit codes designed for automated pipelines and scripts:

| Code | Meaning | CI Recommendation |
|:---:|---|---|
| `0` | Clean verdict | Pass pipeline |
| `1` | Hard failure (network error, bad config, missing keys) | Fail pipeline / Alert |
| `2` | CLI usage error (invalid flags) | Fix CI script |
| `3` | Degraded verdict (report printed, but a reviewer/turn failed) | Warning / Soft alert |

### Headless JSON Output
With `nitpicker pr <url> --json`, nitpicker emits exactly one JSON object on `stdout` with structured findings, token metrics, and coverage data. All progress bars, logs, and diagnostic traces are strictly sent to `stderr`:

```bash
OUTPUT=$(nitpicker pr https://github.com/org/repo/pull/42 --no-comment --json)
STATUS=$(echo "$OUTPUT" | jq -r '.status')
DEGRADED=$(echo "$OUTPUT" | jq -r '.degraded')
```

### OpenTelemetry Tracing
Export every run as an [OpenTelemetry](https://opentelemetry.io) trace (requires `--features otel`):

```bash
cargo install nitpicker --features otel

export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export OTEL_SERVICE_NAME="nitpicker"
nitpicker pr https://github.com/owner/repo/pull/42
```

Traces adhere to standard GenAI semantic conventions (`gen_ai.*`), recording token usage, model failovers, tool latencies, and retry attempts. Sensitive prompt text and proprietary repository source code are never exported.

---

## Advanced Features

### External Design Docs (`--context-file`)
Reviewers are sandboxed inside the git repository. If your review needs external context (e.g. an RFC, architecture doc, or migration plan located elsewhere on disk), inject it safely:

```bash
nitpicker --context-file ~/docs/rfc-42.md --context-file /tmp/spec.md
```

### Automatic Fallback Rings (`--fallback`)
Avoid losing 10 minutes of deep review to transient 429 rate-limits. When `--fallback` is enabled, any reviewer that exhausts retries automatically hands its history and completed tool outputs to the next configured model in the ring.

### Self-Reflection (`nitpicker reflect`)
Analyze past review sessions to diagnose recurring false positives, agent friction, and tool loops:

```bash
# Enable trajectory logging in your config:
# [defaults]
# log_trajectories = true

nitpicker reflect          # inspect recent sessions
nitpicker reflect --n 10   # inspect last 10 runs
```

### Rust Library (`nitpicker-agent`)
The underlying agentic loop and file/git tools are published as a standalone crate [`nitpicker-agent`](crates/nitpicker-agent):

```rust
use nitpicker_agent::prelude::*;
use std::path::Path;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let client = client_from_env(LLMProvider::Anthropic { base_url: None, api_key_env: None })?;
    let result = AgentBuilder::new("explorer", "claude-sonnet-5", "You explore codebases.", client)
        .run("Map the module layout of this repo.", &file_agent_tools(), Path::new("."))
        .await?;

    println!("{}", result.text);
    Ok(())
}
```

<details>
<summary>Experimental: Antigravity Keyring Auth (Research Only)</summary>

> [!CAUTION]
> **Research only.** AG2 is Google's internal agentic IDE backend. `auth = "agy-keyring"` reads the `agy` CLI OAuth token from the system keyring to evaluate `gemini-3.x` models. Google actively monitors and suspends accounts using third-party OAuth bridges. For standard Gemini access, use `GEMINI_API_KEY`.
</details>

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version release notes.
