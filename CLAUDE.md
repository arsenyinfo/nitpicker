# nitpicker

Multi-reviewer code review using LLMs. Spawns parallel agents with different models/prompts, aggregates their feedback into a final verdict.

## Contributor memo

- Before opening a PR, update `README.md` and `CLAUDE.md` for any user-facing or architecture-relevant changes.
- If you bump the version, add a short summary entry to the changelog in `README.md`.

## Quick start

```bash
# Review current PR/diff (debate by default)
cargo run -- --repo .

# Use parallel aggregation instead of debate
cargo run -- --repo . --no-debate
cargo run -- --repo . --no-debate --max-turns 40

# Static analysis of existing code
cargo run -- --repo . --analyze
cargo run -- --repo . --analyze src/db/

# Custom focus
cargo run -- --repo . --prompt "focus on SQL injection"

# Ask a free-form question (debate by default)
cargo run -- ask "should we use eyre or thiserror?"

# Ask with parallel aggregation instead
cargo run -- ask --no-debate "should we use eyre or thiserror?"

# Review current branch's open PR and post result as a comment (requires gh CLI)
cargo run -- pr

# Review a remote PR by URL
cargo run -- pr https://github.com/owner/repo/pull/42

# Machine-readable output for embedding (single JSON object on stdout)
cargo run -- pr https://github.com/owner/repo/pull/42 --no-comment --json

# Reflect on saved sessions
cargo run -- reflect
cargo run -- reflect --n 10

# Gemini OAuth (first-time setup)
cargo run -- --gemini-oauth

# Generate config preferring OpenRouter experimental free models
cargo run -- init --free

# Alloy mode: pool all reviewer models into one shared random-selection client
cargo run -- --alloy
cargo run -- ask --alloy "should we use eyre or thiserror?"
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
pr.rs           GitHub PR subcommand: fetch metadata via gh, review, post comment
output.rs       JSON output contract for `pr --json` (OutputFormat, PrReviewOutput envelope, emit_json)
reflect.rs      Reflect subcommand: analyze saved session trajectories and synthesize improvements
gemini_proxy/   local HTTP proxy that translates Gemini API calls to Google Code Assist
azure.rs        Azure AD token auth for Foundry-hosted OpenAI/Anthropic (feature `azure`, off by default)
codex.rs        ChatGPT/Codex subscription auth — reuses `~/.codex/auth.json`, talks the Codex Responses endpoint
```

### Review flow

1. `review.rs` spawns one `tokio::task` per `[[reviewer]]` in config
2. Each task runs `agent.rs::run_agent` — an agentic loop: call LLM → execute tool calls → feed results back → repeat until the model returns text (default max 100 turns, overrideable via config/CLI)
3. All reviewer outputs are collected, concatenated, and sent to the aggregator model in a single completion call
4. The aggregator's response is printed to stdout

### Debate flow (default review mode and `ask`)

1. `reviewer[0]` = Actor/Reviewer, `reviewer[1]` = Critic/Validator, `aggregator` = Meta-reviewer
2. Each round: Actor turn → Critic turn. Both have access to all file/git tools plus `submit_verdict(verdict, agree)`
3. `agree=true` from Critic → convergence, loop ends early
4. After all rounds: meta-reviewer synthesizes the full dialogue in a single non-agentic completion
5. Default stdout shows only the final synthesized result; `--verbose` also prints the intermediate debate text and transcript path
6. Transcript saved to the OS temp dir as `debate-{ts}.md` (topic) or `review-debate-{ts}.md` (code review)
7. `DebateMode::Topic` (from `ask`) uses Actor/Critic roles and general debate prompts
8. `DebateMode::Review` (from default review mode) uses Reviewer/Validator roles and code-review-focused prompts

**Alloy mode** (`--alloy` / `defaults.alloy = true`): instead of pinning actor and critic to `reviewer[0]`/`reviewer[1]`, builds an `AlloyClient` that randomly selects from all configured reviewer models each turn. Requires ≥ 2 reviewers.

### Agent execution (`agent.rs`)

- Each reviewer runs an agentic loop with file/git tools until it returns text or reaches the turn limit
- Review prompts encourage a quick local map, a short working plan, and fanning out **all** disjoint threads as one broad parallel wave of subagents, re-spawning only when a finding demands a follow-up (each extra serial wave adds wall-clock latency)
- Within a single turn, all tool calls run **concurrently** (`join_all`): a wave of `spawn_subagent` calls overlaps instead of running one-at-a-time, so subagent breadth no longer scales wall-clock. The turn is processed in three phases — ordered cycle/terminal bookkeeping (no awaits), concurrent execution, then results folded back in original index order (provider requires tool-result ordering)
- Concurrent in-flight LLM calls are bounded by a shared `llm_semaphore` (`MAX_CONCURRENT_LLM_CALLS`, default 16), acquired only around each `completion()` call — never held across a subagent spawn, so a blocking acquire bounds account-wide provider concurrency without deadlock. Shared across all reviewers + subagents in `review.rs`; per-turn in `debate.rs` (debate turns never overlap)
- Reviewers can delegate deeper investigations via `spawn_subagent`
- Subagent depth is capped at 2 to bound recursion and cost
- Subagents return results through a hidden `finish(result)` tool; debate agents use `submit_verdict(verdict, agree)` instead. A terminal tool only ends the loop when it **actually ran** (not cycle-blocked, not errored) — a blocked/malformed terminal call never populated the verdict/finish store, so terminating on it would return an empty result; instead the agent gets another turn to retry
- Repetitive tool-call cycles are blocked, and the agent can force a context reset to break out of loops
- Session-log appends are serialized by a shared mutex and written as a single buffer (`session.rs`), so a concurrent subagent wave sharing a writer can't interleave partial lines

### PR flow (`pr.rs`)

0. `run_pr` is a thin wrapper around `run_pr_inner`: it stamps a start `Instant` and, in `--json` mode, turns any `Err` into a `status: "error"` JSON object on stdout + `process::exit(1)` (text mode keeps the eyre-to-stderr path). Config loading happens inside `run_pr_inner` so its failures honor the JSON contract too. There is deliberately no JSON panic hook — reviewer work runs in `tokio::spawn` tasks whose panics are caught as `JoinError` and folded into the report (a process-wide hook would double-emit there); a genuine top-level panic aborts non-zero with a stderr message.
1. `check_gh()` verifies the `gh` CLI is available
2. `PrFlow` enum picks the path: `CurrentBranch` (no URL), `InPlace` (URL + origin matches + no `--clone`), or `TempClone`. `PrLock` is acquired BEFORE any git mutation for the first two; `TempClone` is lock-free (unique temp dir per process). Liveness uses `libc::kill(pid, 0)`. The PR number is carried out of the flow arms (it is not part of `PrMeta`) for the JSON envelope.
3. In-place: refresh remote-tracking branches, skip checkout if `HEAD == headRefOid`, otherwise require a clean working tree and `git switch -c` to a namespaced `nitpicker/pr-N` from `FETCH_HEAD`. The original HEAD is captured as `HeadState::{Branch,Detached}` and restored by a `BranchRestoreGuard` whose `Drop` runs on every exit path — clean return, early `?`, or panic — so the user can't be stranded on `nitpicker/pr-N`. A detached HEAD is restored with `git switch --detach <sha>` (plain `git switch -- <sha>` refuses a bare commit). The guard drops before `PrLock`, so restore happens while the lock is still held.
4. Temp clone: `git clone --filter=blob:none` (partial clone, so merge-base is reachable) then fetch + switch to the PR head; `TempDir` drops at the end.
5. `fetch_pr_meta` retrieves title, body, and `headRefOid` via `gh pr view --json`; `fetch_pr_comments` pulls issue-level comments separately.
6. `build_pr_prompt` assembles the review prompt from PR title + body + PR comments + diff context + optional `--prompt`.
7. Review runs via `debate::run_debate` by default, or `review::run_review` with `--no-debate`. Unless `--no-comment`, result is posted back via `gh pr comment`.
8. Output is governed by the `--json` flag (on `PrArgs`, scoped to `pr` only) which maps to the internal `OutputFormat` enum: `Text` keeps the legacy human stdout (report printed, then comment posted); `Json` posts the comment first (so its outcome is reflected in `comment_posted`), then writes one `PrReviewOutput` line to stdout via `output::emit_json` (which flushes before the caller's `process::exit`). In JSON mode, `debate.rs` suppresses its cast-line/verdict `println!`s and the `termimad` verdict rendering (threaded via `DebateOptions.format`), and tracing is always routed to stderr — so stdout stays a single clean JSON object. Subprocess caveats (for callers): `gh` auth/rate-limit is shared across processes, `--repo` must be an existing dir, kill via process-group on timeout (blocking `git`/`gh` children don't get the signal otherwise), and set `log_trajectories=false` to avoid per-run session writes.

### Reflect flow (`reflect.rs`)

1. Load recent session directories from `~/.nitpicker/sessions` or explicit `--session` paths
2. Parse per-agent JSONL tool traces and `aggregation.json` into typed session records
3. Format each session into a compact markdown summary of agents, tool activity, and final verdict
4. Run one analysis task per session using the first reviewer model
5. Synthesize the per-session analyses into a final report using the second reviewer model when available, otherwise reuse the first

### LLM abstraction (`llm.rs`)

- `LLMClient` trait: one method, `completion(Completion) -> Result<CompletionResponse>`
- Per-provider impls: `anthropic::Client`, `gemini::Client`, `openai::CompletionsClient`
- `AlloyClient` wraps a vec of `(Arc<dyn LLMClientDyn>, model_name)` slots and picks one at random per call (XBOW Alloy technique)
- `RetryingLLM<C>` wraps any client with jittered exponential backoff (4 attempts, 250ms–5s). Skips retry on 4xx errors. The 4xx/429 classifiers match against the full error chain (`format!("{err:#}")`), not `err.to_string()`, because provider impls surface the status only in the wrapped source (`ProviderError(body)` under a `.wrap_err_with(...)` context). The shared `mentions_http_status` matcher (also used by `azure::is_unauthorized`) requires the status number to be standalone (not in a longer digit run) **and** in an HTTP-status context — preceded by a nearby `status`/`code` key or followed by its reason phrase (`401 Unauthorized`) — so an incidental number in a body (`400 tokens`, `trace 404`) isn't misread as the response status. The key must be a **whole word** (left edge is string start or a non-`[a-z0-9_]` byte, checked against the full string so a word split by the scan window is still judged correctly), so `decode`/`encode`/`unicode`/`error_code` don't count as a `code` status key and a transient error like `error decoding response body … 404` keeps its retries. `is_non_retryable_client_error` also gives **5xx precedence**: if any of 500/502/503/504 appears in the chain it returns retryable even when a 4xx is nested in the body (e.g. an upstream `"code": 403` inside a 502 envelope), so a recoverable server error isn't dropped.
- Always wrap clients with `.with_retry()` — the OAuth Gemini path is no exception
- `AzureAdClient` (in `azure.rs`, feature `azure`) is a refreshing decorator: it acquires an AAD bearer token via the Azure SDK and rebuilds the inner rig client just before the token expires. Built in `provider.rs` when `auth = "azure-ad"`, then wrapped with `.with_retry()` like every other client. Since 401 is non-retryable, it also force-refreshes once on a 401 (detected via the same chain-walk as the retry classifiers). `ensure_client` uses double-checked locking so concurrent callers (e.g. parallel subagents sharing the client) don't each refresh; the 401-refresh path dedups the same way but gates on **client identity** rather than expiry (the token was rejected despite not being clock-expired, so an expiry re-check would wrongly skip the refresh) — a burst of concurrent 401s triggers exactly one token fetch.

### Azure AD auth (`azure.rs`)

- Gated behind the off-by-default `azure` cargo feature (the base crate's MSRV is 1.85; the `azure` feature raises it to 1.88 via `azure_core`). The whole module compiles out when the feature is off; `provider.rs` and the config validator bail with a `--features azure` hint if `auth = "azure-ad"` is configured without it.
- `Config::validate` fails fast on `auth = "azure-ad"`: it requires a non-empty `base_url` and rejects an unknown credential mode (anything other than `dev`/`prod`/`auto`/unset). The credential-mode check mirrors the runtime resolution order — when `azure_credentials` is unset it validates the `AZURE_TOKEN_CREDENTIALS` env-var fallback too, so a bogus env value fails here rather than at the first LLM call. Unknown `auth` values on any non-Gemini provider are also rejected rather than silently accepted.
- For Foundry, `provider = "openai"` (base_url `.../openai/v1`) sends the token via the OpenAI client's Bearer auth; `provider = "anthropic"` (base_url `.../anthropic`) injects `Authorization: Bearer` through rig's `.http_headers()` since that client otherwise hardcodes `x-api-key` — `.api_key(...)` gets a placeholder so the AAD token isn't leaked into the unused `x-api-key` header.
- Credential chain selected by `azure_credentials` (`dev`/`prod`/auto, falling back to the `AZURE_TOKEN_CREDENTIALS` env var); scope via `azure_scope` (default `https://cognitiveservices.azure.com/.default`; empty/whitespace is treated as unset and falls back to the default rather than failing at the first call). `base_url` is trimmed at both config validation and client construction, so a whitespace-padded endpoint normalizes identically instead of reaching rig verbatim. Credential construction is non-fatal for all modes — failures are skipped and an empty chain produces a clear "no Azure credentials could be constructed" error. Each reviewer/aggregator owns its own client and caches the token until ~60s before expiry.

### ChatGPT/Codex subscription auth (`codex.rs`)

- `auth = "codex"` (validated for `provider = "openai"` only; no env var required) reuses the OAuth token the Codex CLI writes to `~/.codex/auth.json` (or `$CODEX_HOME/auth.json` when set, non-empty, absolute — relative/unresolvable paths fail fast). The file is read **read-only**; nitpicker never writes back. API-key-mode files (no `tokens` object) are rejected with a `codex login` hint.
- Token lifecycle: initial expiry is decoded from the access token's JWT `exp` claim (missing/unparseable → already-expired, forcing one refresh). Refresh POSTs `grant_type=refresh_token` to `auth.openai.com/oauth/token` with the public Codex client id; expiry then comes from the response's `expires_in` (authoritative, so a token without `exp` never thrashes). Account id is `tokens.account_id`, else derived from `id_token`/`access_token` claims (`chatgpt_account_id` → nested `https://api.openai.com/auth` → `organizations[0].id`). A refresh rejected with a 4xx reloads `auth.json` once (the Codex CLI may have rotated the refresh token concurrently) before failing.
- Concurrency: token cache + reqwest client live in one `CodexClient` (the token is supplied per-request, so unlike `AzureAdClient` there's no inner-client rebuild). `current_access` double-checks expiry under the lock so a concurrent subagent wave refreshes once; a 401 forces a single refresh-and-retry gated on the rejected access token (a burst of 401s collapses to one fetch). Wrapped with `.with_retry()` like every client; the 401 path is handled internally since RetryingLLM treats 401 as fatal.
- Request path: the endpoint `chatgpt.com/backend-api/codex/responses` speaks the OpenAI **Responses** API but rig's high-level responses client is unusable here (it hardcodes `instructions: None`). So `CodexClient` reuses rig's public `responses_api::{CompletionRequest, CompletionResponse}` types for request **lowering** and response **parsing** but does the HTTP itself to satisfy the backend's quirks: top-level `instructions` = the system prompt (taken out of the rig request so it isn't also added as an input item; a completion with no system prompt is rejected up front), `stream: true` (mandatory), `store: false` (merged into `additional_params`), and `max_output_tokens` omitted (rejected outright). Because `store: false` is stateless, the terminal `response.completed` event carries an empty `output`, so items are accumulated from `response.output_item.done` events and injected before rig parses. Finish reason: tool calls → ToolUse; else `incomplete_details.reason == "max_output_tokens"` → MaxTokens; else Stop.
- **Multi-turn reasoning under `store: false`**: a reasoning item the model returns this turn is, by default, replayed next turn as a bare `rs_...` id — which the stateless backend can't resolve (`HTTP 404 — Items are not persisted when store is set to false`), so every loop past turn 1 died. `build_body` therefore merges `include: ["reasoning.encrypted_content"]` alongside `store: false`. rig round-trips that blob both ways (response `Output::Reasoning.encrypted_content` → core `ReasoningContent::Encrypted` → request reasoning input item), so reasoning replays **inline** rather than by id. rig only auto-adds that `include` when a `reasoning` config is present, which nitpicker doesn't set, hence the explicit merge.
- `init` detection (`detect.rs::detect_codex`) surfaces a logged-in Codex CLI as a commented `auth = "codex"` reviewer (`gpt-5.5`) via `codex::auth_available()` (reuses the same `auth.json` parse, so API-key-mode files don't qualify), gated on no `openai`-named provider already detected.
- Research-only framing in user-facing copy (third-party use of the Codex OAuth client is arguably against OpenAI ToS), mirroring the AG2 gemini path. Tokens are never logged.

### Tools (`tools.rs`)

Tools return `String`, never `Err` — errors are returned as `"Error: ..."` strings so the LLM can self-correct. The exception is truly unrecoverable errors (e.g. missing required argument).

`GitTool` only allows a fixed allowlist of read-only subcommands. Commands are passed directly to `Command::new("git").args(tokens)` — no shell involved. The subcommand allowlist (`ALLOWED_GIT_SUBCOMMANDS`) deliberately excludes the `branch`/`tag` porcelain, whose read and write modes can't be told apart by arguments (attempting to validate them as read-only repeatedly leaked ref-creation bypasses via abbreviations, `=`-glued values, the `--` marker, and value-taking flags absorbing the next token). Ref listing/queries are served instead by the read-only plumbing `for-each-ref`/`show-ref`, which have **no ref-creating/deleting mode** — they are safe by construction for any argument shape (`for-each-ref --contains <sha> refs/heads/` covers branch/tag queries). The remaining argument check, `ensure_readonly_git`, blocks the output-to-file flag (`--output`/`-o`, including long-option abbreviations like `--out`) — but only on `diff`/`log`/`show`, the subcommands that actually support it (elsewhere `-o` is read-only, e.g. `ls-files -o` = `--others`). All git commands also run with `GIT_OPTIONAL_LOCKS=0` so a nominally-read command (e.g. `git status`) can't rewrite `.git/index` stat caches or contend on `index.lock` against the user's repo in `pr` in-place mode.

`GrepTool` recursively searches files and skips binary files. Context loading for `CLAUDE.md` / `AGENTS.md` also skips binary files.

Tool outputs are intentionally a bit self-describing: `read_file` includes file/range headers, `glob`/`grep` return explicit no-match messages, and truncation messages say when output is partial.

### Session artifacts (`session.rs`)

- When `[defaults].log_trajectories = true`, nitpicker writes session artifacts under `~/.nitpicker/sessions/session-<timestamp>-<pid>/`
- Reviewer and debate-turn traces are stored as per-agent JSONL files
- Final synthesized output is saved as `aggregation.json`

### Gemini AG2 proxy (`gemini_proxy/`)

When `auth = "agy-keyring"` is set for a Gemini reviewer/aggregator, nitpicker:
1. Runs a local axum HTTP server on a random port
2. Translates incoming Gemini API requests to Google Code Assist API format
3. Attaches the Antigravity OAuth Bearer token read from the system keyring
4. Sends chat through `v1internal:streamGenerateContent?alt=sse` and folds SSE chunks back into Gemini-style JSON

The token is read via the `keyring` crate (Secret Service on Linux, Keychain on macOS, Credential Manager on Windows) at `service=gemini`, `account=antigravity`, decoding the optional `go-keyring-base64:` wrapper. Refresh is delegated to `agy` — if the token is expired the proxy bails with "run `agy` to refresh it". `fetchAvailableModels` is called on proxy startup to discover available model IDs; tested AG2 models are `gemini-3.1-pro-low` and `gemini-3.5-flash-low` (others like `gemini-3-flash-agent` should work but are untested).

This auth path is explicitly disallowed by AG2 ToS Section 6 ("using the Service in connection with products not provided by us") and Google is actively suspending paid accounts for third-party OAuth bridges — keep it framed as research only in any user-facing copy.

The legacy `auth = "oauth"` (browser PKCE flow with file-backed token storage) was removed in 0.5.0 — the proxy was retargeted at AG2 endpoints whose matching client_secret is not public, so the flow could not complete. The config validator now rejects `auth = "oauth"` with a migration hint to `agy-keyring` or `GEMINI_API_KEY`.

## Configuration

Config hierarchy (first wins):
1. `--config <path>` (explicit)
2. `nitpicker.toml` in repo root
3. `~/.nitpicker/config.toml` (global)

Reviewers automatically load project context from `CLAUDE.md` or `AGENTS.md` if present in the repo root.

`nitpicker init --free` prefers OpenRouter in the generated config and writes `model = "free"` for OpenRouter slots when `OPENROUTER_API_KEY` is set. When the generated config uses two reviewer slots, it emits two OpenRouter free reviewers so both slots get free-model auto-selection. If the key is missing, init warns and falls back to the normal provider order.

## Adding a new provider

1. Add a variant to `ProviderType` in `config.rs` with a `#[serde(rename = "...")]`
2. Add a new arm to `provider_from_config` in `review.rs`
3. Add a new variant to `LLMProvider` in `llm.rs` and implement `client_from_env`
4. Implement `LLMClient` for the provider's client type

## Key constraints

- Reviewers run concurrently — reviewer code must be `Send + Sync`
- Parallel review execution is capped at 8 concurrent reviewers; within an agent, a turn's tool calls run concurrently and all in-flight LLM calls share a global cap of `MAX_CONCURRENT_LLM_CALLS` (16)
- Tool results are truncated to 50k bytes before being sent to the LLM
- Git tool output is truncated to 50k chars
- Agent and debate turn loops default to 100 turns and can be overridden via config or CLI
- Context files (`CLAUDE.md`, `AGENTS.md`) are limited to 50k chars
- Prefer `match` over `if let` for better exhaustiveness checking, even if it requires a `_ => unreachable!()` arm
