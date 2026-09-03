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

# Ordered fallback: keep the current agent/history and try the next configured reviewer
cargo run -- --fallback
```

## Architecture

This is a two-crate Cargo workspace: a publishable library crate `nitpicker-agent`
(`crates/nitpicker-agent/`) holds the reusable agentic core, and the `nitpicker` binary at
the repo root (`src/`) holds the CLI/review/debate/PR layer and depends on the library via
`nitpicker_agent::`. The boundary is one-directional: the library never references a binary
module. The binary's `azure`/`antigravity` features forward to the library's same-named
features (see "Feature boundary" below).

```
crates/nitpicker-agent/  — published library crate `nitpicker-agent`
  lib.rs          public surface: pub mod re-exports, `prelude`, `AgentBuilder`,
                  `file_agent_tools()`, `client_from_env()`
  agent.rs        agentic tool-use loop for a single agent (run_agent, AgentConfig, subagents)
  compact.rs      conversation history compaction
  llm.rs          LLM client trait, per-provider impls, retry wrapper, AlloyClient
  tools.rs        tool definitions: read_file, glob, grep, git (Tool trait, all_tools)
  session.rs      session/trajectory JSONL writers
  config.rs       TOML config deserialization (Config, ReviewerConfig, AggregatorConfig)
  provider.rs     build LLM clients from config (build_reviewer_client, provider_from_config)
  codex.rs        ChatGPT/Codex subscription auth — reuses `~/.codex/auth.json`
  azure.rs        Azure AD token auth for Foundry-hosted models (feature `azure`, off by default)
  openrouter.rs   OpenRouter free-model resolution
  prompts.rs      `subagent_system_prompt()` (the default; overridable per AgentConfig)
  prompts/        generic loop contracts owned by the library: compaction, final-turn handling,
                  and the fallback subagent protocol

src/  — `nitpicker` binary (CLI)
  main.rs         CLI, config loading, wires everything together
  review.rs       orchestrates parallel reviewers → aggregation
  debate.rs       sequential actor/critic debate loop → meta-review
  pr.rs           GitHub PR subcommand: fetch metadata via gh, review, post comment
  presets.rs      review presets: built-in rubric registry, CLI/config resolution, synthesis-failure wrapping
  output.rs       JSON output contract for `pr --json` (OutputFormat, PrReviewOutput, emit_json)
  progress.rs     interactive progress formatting + tracing writer bridge for spinner-safe logs;
                  owns the text-mode, TTY-only animated OSC 0 terminal-title indicator used during runs
  reflect.rs      Reflect subcommand: analyze saved session trajectories and synthesize improvements
  detect.rs       provider auto-detection for `init`
  context.rs      `--context-file` loading + prompt injection (ContextFile, load_context_files, append_to_prompt) and project-context discovery (build_context: CLAUDE.md/AGENTS.md)
  prompts.rs      review/debate/ask prompts (RunTask, LaneTask, ReviewScope)
  proxy.rs        owning handle for the optional Gemini proxy (GeminiProxy::maybe_start/url); inert when `antigravity` is off
  gemini_proxy/   local HTTP proxy server translating Gemini API → Google Code Assist (feature `antigravity`, off by default)
```

### Feature boundary

- `azure` (library): owns the Azure SDK deps (`azure_identity`/`azure_core`) and the `azure.rs`
  module. The binary's `azure` feature is `["nitpicker-agent/azure"]` — pure forward.
- `antigravity` (library): code-gate only (no extra deps) — compiles in the config validation
  for `auth = "agy-keyring"` plus the proxy-URL client hook (`llm::create_gemini_client_with_proxy`).
  The binary's `antigravity` feature forwards to it **and** adds the proxy *server*
  (`gemini_proxy/`) with its `axum`/`uuid`/`keyring` deps. The two must be enabled together
  (forwarding ensures this): enabling it binary-side without the library gate would accept the
  auth value while the client hook compiled out.

### Customizable prompts (library)

The top-level agent's `system_prompt` and `initial_message` are caller-supplied and injected
verbatim. The subagent system prompt defaults to `prompts::subagent_system_prompt()` but is
overridable via `AgentConfig::subagent_system_prompt` (and `AgentBuilder::subagent_system_prompt`);
the override is inherited by nested subagents. `None` ⇒ the built-in generic prompt.

### Review presets (`presets.rs`)

A preset is one named review angle: `ReviewPreset { name, prompt }`. Presets decide **what** to
investigate; parallel/debate/alloy decide **how** — no preset changes provider behavior or grants
write capabilities. Resolution (`presets::resolve`): CLI `--preset` (repeatable, comma-split,
trimmed, first-seen dedup, case-sensitive, capped at 16 post-dedup) > `[defaults].presets` > the
built-in four in prompt order (correctness, security, performance, simplicity). Domain-specific
`ai-systems`, `ml-rigor`, and `tone` are opt-in; `general` is a standalone alternative. A
`[presets.<name>]` table overrides the same-named built-in. The library's
`Config` holds preset **data** only (blank-prompt/padded-name/control-byte validation); the registry and all
name resolution live in the binary. Resolution runs after `load_config`+validate but BEFORE
`resolve_free_models` (which fires live smoke completions) at all three review entry points, so an
unknown name fails before any network call. CLI Alloy/fallback modes are likewise resolved and
reviewer-count-validated before that boundary; in `pr` both checks happen inside `run_pr_inner` to
honor the JSON error contract. `--preset` follows the `--context-file` flatten/merge pattern (root + `pr`,
merged root-first, not clap-global); `ask`/`init`/`reflect` reject it — post-subcommand by clap,
root-position by an explicit check (root flags otherwise parse and get silently discarded). Prompt
composition stays in `prompts.rs`: the per-preset rubric is appended AFTER the shared protocol
block so same-model jobs keep a byte-identical tools+protocol prefix for provider caching (the
library appends shared project context after the system prompt, so cross-preset sharing covers
tools+protocol only). Subagents inherit a rubric-aware prompt via `AgentConfig::subagent_system_prompt`.
The one final synthesis (both modes) receives the preset roster — name and FULL rubric — in the
user message (preambles stay protocol-only), filtered to presets with surviving evidence: a
rubric with no matching report/lane would read as an angle reviewed clean (the session record
and `pr --json` keep the full resolved list — they document resolution; execution coverage travels
separately as the envelope's per-preset `coverage` counts/`degraded` and the session record's `jobs`/`lanes`). A preset
run's synthesis failure is wrapped with the run's preset/job counts, plus a "select fewer
presets" hint only when `llm::is_context_length_error` classifies it as a context-window
overflow; Ask/Topic failures propagate untouched.

### Review flow

0. Diff review entrypoints call `detect_diff_context` once before fan-out. It resolves full HEAD,
   base, and merge-base object ids; records the exact two-endpoint committed comparison; and embeds
   bounded committed `git diff --name-status -M` plus working-tree `git status --short` maps (the
   latter includes untracked files). The resulting frozen orientation is part of the user topic, so
   parallel jobs and every later debate turn receive identical revisions and paths without an
   LLM-generated summary. Agents still inspect hunks and code normally. Static `--analyze` and
   free-form `ask` runs do not receive this diff snapshot. The base ref (`resolve_base_branch`) is
   local `<branch>` unless `origin/<branch>` exists and yields a strictly newer merge-base with HEAD
   (`local_base_is_behind_remote`) — a branch cut from `origin/main` while local `main` lagged would
   otherwise diff against the stale merge-base and pull already-upstream commits into the review.
   Equal or diverged merge-bases keep the local ref, a switch is logged as a warning, and no fetch
   is performed (only remote-tracking refs are consulted)
1. `review.rs` spawns one `tokio::task` per {reviewer × preset} job (`plan_jobs`); the Ask path (`presets: None`, used by `ask --no-debate`) stays one job per reviewer with byte-identical legacy prompts, labels, and failure handling. Clients are built once per reviewer and Arc-cloned per job; normally a reviewer whose client build fails takes down only its own jobs. In fallback mode (`--fallback` / `defaults.fallback = true`), each logical reviewer instead gets a `PriorityClient` ring beginning at its own index and wrapping through declaration order; unavailable build routes are skipped. Job labels (`{preset} · {name}`, ` #{i+1}` on duplicate reviewer names, job-ordinal fallback for crafted collisions) head the report sections; session identities are `reviewer-{i}-{name}-{j}-{preset}` (both indices load-bearing under sanitization collisions)
2. Each task runs `agent.rs::run_agent` — an agentic loop: call LLM → execute tool calls → feed results back → repeat until the model returns text (default max 100 turns, overrideable via config/CLI). Discovery prompts default to one early bounded/disjoint subagent wave when the target has independently reviewable surfaces; validation prompts permit delegation only for targeted multi-step verification of submitted claims, never broad missed-finding discovery. All jobs spawn eagerly; the shared `llm_semaphore` (16) is the only concurrency bound (the old 8-reviewer semaphore is gone)
3. Successful outputs are collected, concatenated, and sent to the aggregator model in a single completion call, prefixed with the original task and the preset roster. The final finding schema requires a `Lens` field: workers use their exact preset name, and synthesis retains every evidence-contributing lens when it merges duplicates (lens attribution is distinct from forbidden reviewer/model attribution). Preset runs EXCLUDE failure stubs from the aggregator input (execution noise, not review evidence — logs/progress/degraded accounting carry them); the Ask path keeps its stubs. If **every** job failed, `run_review` bails before the aggregator — synthesizing a verdict from nothing but failure notes would fabricate a confident review; in `pr --json` this surfaces as `status: "error"` and no comment is posted
4. The aggregator's response is printed to stdout only when complete: `ToolUse` and `MaxTokens` finish reasons fail synthesis, so an unexpected tool call or truncated fallback-route verdict cannot be presented as the final report. Debate meta-review enforces the same boundary. `ReviewOutcome.degraded` (some but not all jobs failed) makes the default-review/`ask` arms exit 3 after printing — stdout is flushed first since `process::exit` skips teardown (contract: 0 clean / 1 hard failure / 3 degraded; 2 is clap's usage-error code, deliberately unused; `pr` joins the same contract — its degraded flag propagates out of `run_pr` so the exit happens after the checkout-restore guards drop)

### Debate flow (default review mode and `ask`)

1. `reviewer[0]` = Actor/Reviewer, `reviewer[1]` = Critic/Validator, `aggregator` = Meta-reviewer. Review mode runs **one independent lane per preset**, all lanes concurrent (`join_all` over borrowing futures — lane state borrows `run_debate` locals, so no `tokio::spawn`); Topic (`ask`) is the degenerate single `None` lane and byte-compatible with pre-preset behavior. The outer runner owns proxy, clients/alloy pool, repo context, session logger, one `MultiProgress`, a per-run `llm_semaphore` shared by every lane + subagent, and the meta step
2. Each round within a lane: Actor turn → Critic turn, strictly sequential. Both have all file/git tools plus `submit_verdict(verdict, agree)`; lane subagents inherit the preset's rubric-aware subagent prompt (Topic keeps the generic one)
3. `agree=true` from Critic → convergence, lane-local, ends that lane early — but agreement is literal: the opponent's latest verdict must be forwardable unchanged; any changed finding, field, uncertainty, caveat, or unresolved blocker requires `agree=false`. The tool definition repeats this invariant. Actor/Reviewer owns recall and discovery; Critic/Validator owns precision, evaluates only the submitted set, and reports all currently visible objections together instead of hunting for new findings/options or manufacturing disagreement. A critic that agrees with a failed actor's `*Agent failed*` stub (or a failed critic, whose verdict defaults to `agree=false`) does not converge. Review candidates receive the optional Uncertainty field; validators and both synthesis paths receive only the confirmed schema, which cannot represent unresolved uncertainty. Review verdicts end with a mandatory `## Coverage` block (`prompts/protocol/coverage-schema.md`: inspected files/hunks, checked risk classes, not-inspected areas); the validator checks it against the frozen snapshot and rejects with `agree=false` only a gap that could plausibly change the finding set — wording, grouping, and files it would merely have listed differently are explicitly not grounds, since nothing downstream reads the block and each rejection costs a full extra round — so a lane with no findings gets a bounded coverage check instead of an aimless validator turn. Rejecting every submitted finding is `agree=false` with the critique (the reviewer resubmits without them), so rejection rationale survives in the staged verdicts. An agreeing empty verdict is the `No findings. Great job! 🎉` line followed by the reviewer's coverage block unchanged, and the meta prompt is told coverage blocks are not findings. Nothing in the harness parses verdict prose. Round ≥2 user messages (`build_turn_message`) add a follow-up-round instruction — respond only to the disputed or changed points, no re-discovery — and the topic prompts carry the same "do not restart" clauses the review prompts already had
4. After all lanes: meta-reviewer synthesizes the surviving lanes (a lane survives iff ≥1 turn really ran; dead lanes are omitted from the meta input — roster included — but appear flagged in the transcript) in a single non-agentic completion. A **cleanly converged** lane's meta input is pruned to its final round (`lane_pruned_to_final_round`): verdicts are self-contained by prompt contract and the agreeing critic restates every confirmed finding, so earlier rounds are superseded chronology that a synthesizer misreads into withdrawn-claim narration. A degraded lane can converge too, but an earlier failed turn voids the self-containment premise, so converged AND degraded lanes keep their full trail, and the on-disk transcript always does. Per-lane sections state convergence AND degraded status; failed-turn markers are flagged as execution errors, non-convergence as unresolved disagreement. If **every** turn in every lane failed, `run_debate` bails before the meta step rather than fabricating a verdict from errors (→ `status: "error"` in `pr --json`)
5. One progress row owns each lane for its full lifetime: in interactive non-verbose mode the preset is the fixed prefix, while role/round/stats update in place until the row finishes with convergence/degradation and aggregate usage; indicatif hides the same rows on non-TTY/verbose paths. A Presets cast line lists the roster, and non-interactive stdout stays final-verdict-only for subprocess callers. In a terminal, `--verbose` prints intermediate debate text — live for a single lane, buffered per lane and printed post-completion for concurrent lanes (interleaving would be unattributable). `DebateOutcome.degraded` (any turn in any lane failed) → exit 3 in the default-review/`ask`/`pr` arms, same contract as the review flow
6. One combined transcript in the OS temp dir: `debate-{ts}.md` (topic, unchanged) or `review-debate-{ts}-{preset-slugs}.md` (review; slugs bounded), one section per lane with convergence/degraded metadata. Lane trajectory stems are `lane-{j}-{preset}-review/validate-{round}` (both index and slug load-bearing); `AggregationRecord` keeps scalar rounds/converged whenever there is exactly one lane (reflect renders only the scalars) and adds `presets` + per-lane `lanes` metadata on preset runs
7. `RunTask::Ask` uses Actor/Critic roles and general debate prompts with one unscoped `LaneTask::Ask`
8. `RunTask::Review { scope, presets }` owns the resolved preset slice and derives one `LaneTask::Review { scope, preset }` per angle. Parallel and debate entrypoints therefore cannot pair an Ask task with presets or a Review task without them. Review lanes use Reviewer/Validator roles and code-review-focused prompts composed with exactly one rubric. `ReviewScope` (`Diff` vs `Static`) keeps change-attribution rules for diff review ("post-change code", "fixes the diff landed") and swaps them for impact-based static-analysis framing under `--analyze`

**Alloy mode** (`--alloy` / `defaults.alloy = true`): instead of pinning actor and critic to `reviewer[0]`/`reviewer[1]`, builds an `AlloyClient` that randomly selects from all configured reviewer models each turn. Each `AlloySlot` carries its model's own `max_tokens`, applied together with the model at selection — a cap belongs to the model it was set for, not to the role whose turn it is. With fallback enabled, the random selection remains the first attempt; failure walks the remaining slots in declaration order, wrapping once. Requires ≥ 2 reviewers. Mixed-provider histories must stay provider-portable; the Codex boundary normalizes missing Responses `call_id`s from generic tool-call ids before lowering.

**Fallback mode** (`--fallback` / `defaults.fallback = true`): requires ≥ 2 reviewers and adds no per-reviewer configuration. A normal reviewer/actor/critic starts at its assigned reviewer index, then walks the configured reviewer ring; the aggregator starts at `[aggregator]`, then walks reviewers from index 0. `PriorityClient` replays only the failed `Completion`, preserving the full agent history and every completed tool result, and makes the successful route sticky for that logical agent. Successful failover is observable through the selected-model trajectory/log fields but is not degradation because no evidence was lost. Each route retains its own `max_tokens`; a `MaxTokens` response is incomplete, so it tries the next route without blacklisting the capped route. Because compaction runs before route selection, fallback-capable agents use the minimum resolved `compact_threshold` among successfully built reviewer routes; Alloy uses the same conservative pool minimum even without fallback. `FallbackSlot` clones share run-local availability: rate/quota failure removes that route from other priority clients and future Alloy random selection for the remainder of the run, while request-specific errors fall through for the current completion without globally blacklisting the model. `AlloySlot` retains its original public three-field shape for non-fallback Alloy; fallback Alloy accepts shared `FallbackSlot` routes so availability propagates across priority clients. Clearly long-lived subscription quota errors (`out of tokens`, `usage_limit_reached`, five-hour/reset-window wording) bypass the ordinary eight-attempt rate-limit backoff so the next route starts immediately; short 429s keep the existing retry policy before the route is marked unavailable. A transient 5xx envelope takes precedence over nested quota text, preserving retries and route availability.

### Agent execution (`agent.rs`)

- Each reviewer runs an agentic loop with file/git tools until it returns text or reaches the turn limit
- Review prompts encourage a quick local map, a short working plan, and fanning out **all** disjoint threads as one broad parallel wave of subagents, re-spawning only when a finding demands a follow-up (each extra serial wave adds wall-clock latency)
- Within a single turn, all tool calls run **concurrently** (`join_all`): a wave of `spawn_subagent` calls overlaps instead of running one-at-a-time, so subagent breadth no longer scales wall-clock. The turn is processed in three phases — ordered cycle/terminal bookkeeping (no awaits), concurrent execution, then results folded back in original index order (provider requires tool-result ordering)
- Concurrent in-flight LLM calls are bounded by a shared `llm_semaphore` (`MAX_CONCURRENT_LLM_CALLS`, default 16), acquired only around each `completion()` call — never held across a subagent spawn, so a blocking acquire bounds account-wide provider concurrency without deadlock. One per run in both `review.rs` (all jobs + subagents) and `debate.rs` (all lanes + subagents — hoisted out of the per-turn scope when lanes became concurrent)
- Reviewers can delegate deeper investigations via `spawn_subagent`. A subagent starts from an empty conversation — system prompt, project context, and the `task` string only, no topic, snapshot, or parent tool results — so the `task` schema says so and tells the parent to restate revision ids, paths, and claims verbatim; `subagent.md` tells depth-1 subagents to do the work themselves rather than re-delegating
- Subagent depth is capped at 2 to bound recursion and cost
- Subagents never inherit the parent's terminal tools (e.g. debate's `submit_verdict`, which writes into parent-owned verdict state and could falsely converge a debate) — they terminate via their own per-run `finish` tool
- Project context (`CLAUDE.md`/`AGENTS.md`) is appended to the system prompt wrapped in a `<context-only>` tag that marks it as repository-authored reference material, not instructions — it is target-controlled content in `pr` mode
- Compaction is best-effort (`compact_and_account`): a summarizer that fails after its own retries and corrections leaves the run uncompacted rather than aborting the agent, since continuing may still finish. The failure still has to stay visible in both directions the `Compaction` enum feeds. (1) The usage window is reset only by a compaction that actually shrank the history. This matters at the **cycle-break** site alone, which `continue`s past `conversation_usage.record(...)`: there the reset was the entire retry signal, so clearing it after a failure left the next turn carrying the un-shrunk history. At the threshold site the block falls through to its own completion, whose `record` overwrites the window with the real prompt size either way — the gating is a no-op there, kept so the invariant is stated rather than accidental. (2) The trajectory records a `compact` call with `status: "error"` carrying the (truncated) error, since logging `ok` with no summary makes `reflect` count and render it as a success
- Compaction runs under a dedicated summarizer system prompt; the agent's role prompt (which orders tool calls that are unavailable during summarization) is embedded in the compaction request as reference-only material
- Subagents return results through a hidden `finish(result)` tool; debate agents use `submit_verdict(verdict, agree)` instead. A terminal tool only ends the loop when it **actually ran** (not cycle-blocked, not errored) — a blocked/malformed terminal call never populated the verdict/finish store, so terminating on it would return an empty result; instead the agent gets another turn to retry
- Repetitive tool-call cycles are blocked, and the agent can force a context reset to break out of loops
- Session-log appends are serialized by a shared mutex and written as a single buffer (`session.rs`), so a concurrent subagent wave sharing a writer can't interleave partial lines

### PR flow (`pr.rs`)

0. `run_pr` is a thin wrapper around `run_pr_inner`: it stamps a start `Instant` and, in `--json` mode, turns any `Err` into a `status: "error"` JSON object on stdout + `process::exit(1)` (text mode keeps the eyre-to-stderr path). Config loading happens inside `run_pr_inner` so its failures honor the JSON contract too — and AFTER `prepare_pr`, because `pr` mode never reads `nitpicker.toml` from the working tree (the tree holds the PR head — target-controlled content that could redirect `base_url` or inject into the trusted rubric slot via `[presets.<name>]` overrides). Repo-level config comes from the PR base branch blob (`git show origin/<base>:nitpicker.toml`, base ref from `gh pr view`'s `baseRefName`; `load_pr_config`/`choose_repo_config`), falling back to `~/.nitpicker/config.toml`; a warning names the checked-out copy whenever it diverges from or shadows the base version. Two conditions make that anchor sound: (1) `origin` must be a **github.com** remote (`origin_is_github`/`remote_host`) — the in-place flow selects on the owner/repo slug alone, so a same-slug remote on another host would otherwise supply the "base" policy; (2) the head-vs-base comparison runs on **git object ids** (`head_config_state`), never a working-tree read — an unbounded read of a PR-supplied symlink (`/dev/zero`) would hang or exhaust memory before the review starts. An unresolvable `origin/<base>` warns (policy could not be consulted); a merely absent file is quiet. An explicit `--config` stays trusted and unchanged. A config/preset error therefore surfaces after checkout — the restore guards cover it. There is deliberately no JSON panic hook — reviewer work runs in `tokio::spawn` tasks whose panics are caught as `JoinError` and folded into the report (a process-wide hook would double-emit there); a genuine top-level panic aborts non-zero with a stderr message.
1. `check_gh()` verifies the `gh` CLI is available
2. `PrFlow` enum picks the path: `CurrentBranch` (no URL), `InPlace` (URL + origin matches + no `--clone`), or `TempClone`. `PrLock` is acquired BEFORE any git mutation for the first two; `TempClone` is lock-free (unique temp dir per process). The lock is an advisory `flock(LOCK_EX|LOCK_NB)` on a fixed per-repo lock file keyed by `git --git-common-dir`, so linked worktrees that share refs also share the lock. It is held for the process lifetime via an open fd — the kernel releases it on any exit including a crash, so there is no stale-pid detection and no check-then-create TOCTOU (the old pid-file scheme had both). The lock file is never unlinked (unlinking would let a racer lock a fresh inode). `flock` is unix-only; non-unix falls back to exclusive `create_new` (no crash-release). The PR number is carried out of the flow arms (it is not part of `PrMeta`) for the JSON envelope.
3. In-place: refresh remote-tracking branches, then skip the fetch+checkout only when `HEAD == headRefOid` **and** HEAD is on a real branch with a clean tree (a detached HEAD at the PR head, or a dirty tree, falls through to checkout — skipping would either break `detect_diff_context` or review uncommitted WIP and post it). Checkout fetches `+refs/pull/N/head` into a private `refs/nitpicker/pr-N-head` ref (not the shared `.git/FETCH_HEAD`, which a concurrent fetch could rewrite) and `git switch -c` a namespaced `nitpicker/pr-N` from it, requiring a clean working tree. The original HEAD is captured as `HeadState::{Branch,Detached}` and restored by a `BranchRestoreGuard` whose `Drop` runs on every exit path — clean return, early `?`, or panic — so the user can't be stranded on `nitpicker/pr-N`. A detached HEAD is restored with `git switch --detach <sha>` (plain `git switch -- <sha>` refuses a bare commit). The guard drops before `PrLock`, so restore happens while the lock is still held. The `--json` envelope's `head_sha` is the commit actually checked out (`rev-parse HEAD`), not the possibly-stale `headRefOid` from `gh pr view`.
4. Temp clone: `git clone --filter=blob:none` (partial clone, so merge-base is reachable) then fetch + switch to the PR head; `TempDir` drops at the end.
5. `fetch_pr_meta` retrieves title, body, and `headRefOid` via `gh pr view --json`; `fetch_pr_comments` pulls issue-level comments separately.
6. `build_pr_prompt` assembles the review prompt from PR title + body + PR comments + diff context + optional `--prompt`.
7. Review runs via `debate::run_debate` by default, or `review::run_review` with `--no-debate`. Unless `--no-comment`, result is posted back via `gh pr comment`.
8. Output is governed by the `--json` flag (on `PrArgs`, scoped to `pr` only) which maps to the internal `OutputFormat` enum: `Text` keeps the legacy human stdout (report printed, then comment posted); `Json` posts the comment first (so its outcome is reflected in `comment_posted`), then writes one `PrReviewOutput` line to stdout via `output::emit_json` (which flushes before the caller's `process::exit`). In JSON mode, `debate.rs` suppresses its cast-line/verdict `println!`s and the `termimad` verdict rendering (threaded via `DebateOptions.format`), and tracing is always routed to stderr — so stdout stays a single clean JSON object. The envelope carries the resolved ordered `presets` on success only — every error envelope omits the key, including failures after resolution (`PrReviewOutput::error` clears all optionals) — plus `degraded` (any job/turn failed; mirrored as exit code 3, emitted-then-exit so the JSON always lands) and `coverage` (per-preset attempted/succeeded counts in resolution order; entries with `succeeded > 0` identify evidence-producing angles, while exact counts retain partial coverage; envelope-only by design, never fed to the synthesis prompt). The envelope's `usage` block (`UsageReport`: `input_tokens`/`output_tokens`/`total_tokens`/`cached_input_tokens`/`cache_creation_input_tokens`/`subagents_spawned`) is aggregated from the run: `review::run_review` returns a `ReviewOutcome` and `debate::run_debate` a `DebateOutcome`, each folding every reviewer/debate-turn `AgentResult` (subagents + compaction already folded in) plus the aggregator/meta completion — whose usage was previously discarded — via `UsageReport::add`. It is **best-effort** metering: tokens are sourced only from successful `CompletionResponse`s, so a failed reviewer/subagent or a discarded retry contributes 0 (a lower bound on spend, not an exact meter). `usage` is `None` in the `status: "error"` envelope. The cache keys are additive — `SCHEMA_VERSION` stays at 1 since existing consumers that ignore unknown keys are unaffected — but note `input_tokens` itself changed meaning in 0.8.3 on the Anthropic path (it now includes cache reads), so a consumer computing cost must discount `cached_input_tokens` at the provider's cache-read rate rather than charging the whole prompt. Subprocess caveats (for callers): `gh` auth/rate-limit is shared across processes, `--repo` must be an existing dir, kill via process-group on timeout (blocking `git`/`gh` children don't get the signal otherwise), and set `log_trajectories=false` to avoid per-run session writes.

### Context files (`context.rs`)

`--context-file <PATH>` (repeatable) injects a file the agents' tools cannot reach — those are
canonicalize-sandboxed to the repo, so a design doc living outside it is otherwise unreadable. The
flag lives in `ContextFileArgs`, flattened into `Args` (root), `Command::Ask`, and `PrArgs`, and the
root and subcommand vectors are concatenated root-first (= command-line order) at each use site, so
the flag works before or after the subcommand. It is deliberately **not** a clap `global` arg:
global propagation keeps one winning occurrence list (the subcommand's), so a repeatable flag split
around the subcommand would silently drop the root's values. The scalar shared flags (`--repo`,
`--config`, `-v`) *are* global — declared once on `CommonArgs` at the root, readable from
`Args.common` regardless of position (`run_pr` receives them as a parameter). `init`/`reflect`
don't flatten `ContextFileArgs`, so `--context-file` after them is a parse error rather than a
silent no-op; before them it is ignored like every other root-only flag.

Loading is eager and total-budgeted (`MAX_TOTAL_BYTES`, 256 KiB): only regular files are accepted
(a FIFO or device node would block or stream without bound). The check is an fstat on the opened
fd, not a stat on the path — stat-then-open leaves a race where the path is swapped for a FIFO and
the open itself blocks; on unix the open uses `O_NONBLOCK` (harmless for regular-file reads) so
even a raced-in FIFO opens instantly and is then rejected. Each file is read through a bounded reader capped at the
remaining budget + 1 byte so an oversized file fails fast instead of being buffered whole, and the
budget meters the **serialized block** (escaped contents + fence + path + separator, via
`render_block`) rather than raw file bytes — so wrapper overhead is charged and empty files are not
free. Missing, non-regular, over-budget, binary (null-byte guard, mirroring the tool sandbox's),
and non-UTF-8 files are hard errors, so a bad path fails before any model call. Contents are
injected verbatim — no trimming; the only mutation is the closing-tag neutralization below.

`append_to_prompt` appends the blocks to the **task prompt**, not the
system prompt — these are per-run material with the same lifetime as `--prompt`, whereas
`project_context` (`CLAUDE.md`/`AGENTS.md`) is a single slot that propagates to subagents. The
consequence, deliberate: **subagents do not see context files**.

Each file is wrapped in `<context_file path="…">`, with the literal closing tag neutralized in the
contents and `"` escaped in the path attribute — the same breakout that `run_agent` guards against
for `<context-only>`. Injection happens at three call sites (`main.rs` review arm, `main.rs` ask arm,
`pr.rs::run_review_inner`) by wrapping an already-assembled prompt string, so no prompt builder had
to change shape. In `pr` mode the load happens at the injection site rather than at entry, so a bad
path surfaces after the clone/checkout but still before the first model call.

### Reflect flow (`reflect.rs`)

1. Load recent session directories from `~/.nitpicker/sessions` or an explicit `--sessions-dir`
2. Parse per-agent JSONL tool traces, `aggregation.json`, and the optional `attribution.json` sidecar into typed session records. Staged
   verdicts are required; outdated or malformed aggregation records reject that session instead of
   entering reflection with incomplete evidence. A missing or malformed attribution sidecar is
   rendered as unavailable rather than rejecting the session
3. Deterministically calculate unique invocation/status/tool/model counts, maximum delegation depth,
   duration, and repeated exact requests (spawn `started`/`error` lifecycle pairs count as one
   invocation). Format those metrics with agents, per-lane/per-job outcomes, staged verdicts, the
   final verdict, and the attributed tool trace. An error-flagged aggregation record renders its
   synthesis failure (status "synthesis failed"), never its empty text as a verdict
4. Run one analysis task per session using the first reviewer model. The map prompt requires trace
   citations, outcome effect, confidence, and the smallest experiment, and treats independent
   repeated verification as a tradeoff rather than automatic waste
5. Synthesize the per-session analyses with reviewer 2 (the critic), or reviewer 1 when only one is
   configured. This is deliberately not `[aggregator]`: reflection reduction is a tool-using
   investigation that verifies hypotheses in the repo, while ordinary aggregators are often cheap
   report-formatting models. The reduce prompt requires exact `N/M` support, representative
   sessions, verified/unverified code causes, regression risks, and measurable experiments; weak
   hypotheses calibrate the synthesis internally and are omitted from the user-facing report
6. Print the report only. `reflect` does not copy formatted traces into a shared temp directory

### LLM abstraction (`llm.rs`)

- `LLMClient` trait: one method, `completion(Completion) -> Result<CompletionResponse>`
- `TokenUsage::from_provider(usage, CacheAccounting)` is the single ingest point for provider usage, and **normalizes `input_tokens`** to mean all prompt tokens including cache reads. Providers disagree natively: Anthropic reports cache reads *outside* `input_tokens` (`OutsidePrompt`), while OpenAI/Responses/OpenRouter/Gemini fold them in (`InsidePrompt`). The accounting is stated by each client rather than inferred from the numbers — an earlier revision derived it from `total_tokens` and misclassified both Gemini (whose thinking and tool-use prompts are counted beside input *and* output, so its parts don't sum to its total) and OpenRouter gateways (whose totals can disagree with prompt+completion). `cached_input_tokens`/`cache_creation_input_tokens` are breakdowns of `input_tokens`, never second addends, and are passed through exactly as reported — a provider that contradicts itself stays visible rather than being clamped, since inflating the prompt to fit a cache figure would fabricate spend. `total_tokens == input_tokens + output_tokens` (barring saturation), which means provider-side categories counted outside both are not metered: Gemini's thinking and tool-use-prompt counts, and a gateway that omits usage entirely meters zero (rig's sentinel for "no metrics supplied"). Without the normalization, a cache hit on an auto-caching provider reports a fraction of the real prompt (measured: 52 for a 3508-token prompt) and `ConversationUsageWindow::should_compact` never fires
- `CompletionResponse::text()` strips inline `<think>` chain-of-thought, which some providers (MiniMax/GLM/DeepSeek via OpenRouter) emit in the message content instead of a structured field. A block the provider never closes swallows the answer along with the reasoning, so a complete verdict reads as an empty response and dies after four retries. `recover_unterminated_think` therefore subtracts the reasoning from the block body and keeps the remainder — whether or not text precedes the block, since leading text only makes the loss silent (the reply looks successful and nothing retries). The reasoning is subtracted, never guessed: it comes from `AssistantContent::Reasoning`, whose `Text` blocks are concatenated with **no separator** (OpenRouter splits them mid-sentence, so a newline join breaks the prefix match and leaks the whole chain), while `Summary`/`Encrypted`/`Redacted` are excluded since they don't appear verbatim in the body. `scan_think_blocks` reports where the unclosed block *starts*, so several blocks or a stray closer can't misdirect recovery, and the split must fall on a word boundary — real responses run the answer straight on from the reasoning's final punctuation. Anything unmatched returns `""`, as does a *closed* block that strips to nothing: serving the body would pass chain-of-thought off as the answer and silence the retry
- Per-provider impls: `anthropic::Client`, `gemini::Client`, `openai::CompletionsClient`, `openrouter::Client`, plus `CodexClient` (`codex.rs`) and `AzureAdClient` (`azure.rs`), which delegate to the rig clients above. First-party Anthropic and Foundry routes wrap the raw client in `AnthropicPromptCachingClient`, enabling Rig's manual tools/system breakpoints plus its top-level automatic moving breakpoint; arbitrary custom Anthropic `base_url` gateways retain the uncached compatibility shape unless explicitly wrapped.
- `AlloyClient` wraps model/client/token-cap slots and picks one at random per call (XBOW Alloy technique); `new_with_fallback_routes` keeps that random first choice and then tries the remaining shared routes in priority order. `PriorityClient` uses the same completion-boundary routing with a fixed first slot for non-Alloy fallback.
- `RetryingLLM<C>` wraps any client with jittered exponential backoff (4 attempts, 250ms–5s; rate limits get 8 attempts at 5s–60s). Provider adapters preserve Rig's typed `CompletionError`; the classifier downcasts it to read the captured outer HTTP status and parses only structured JSON `code`/`type`/`status` fields. Exact permanent quota codes (`insufficient_quota`, `usage_limit_reached`) bypass retries and make the fallback route unavailable, while exact transient codes (`rate_limit_error`, `rate_limit_exceeded`, `overloaded_error`) and otherwise-unknown 429s use the longer backoff. The real outer 5xx status always wins over nested upstream quota data. Kimi and similar subscription gateways do not consistently expose a distinct long-window code, so the sole compatibility heuristic examines structured provider `message` fields for five-hour/reset/billing-cycle wording; arbitrary rendered `eyre` text is never scanned. Codex and OpenRouter preserve their typed provider source too, and Azure/Codex token refresh decisions consume the same captured status/code facts.
- Anthropic is the one provider whose API **requires** `max_tokens`, so "no cap" can't be expressed there: `impl LLMClient for anthropic::Client` substitutes `ANTHROPIC_DEFAULT_MAX_TOKENS` (8192) when the caller sends none. Letting rig fill it instead would give 2048 for any model name rig doesn't recognize — every Anthropic-compatible gateway (z.ai, proxied models) — which is less than one review turn spends on reasoning alone
- An empty response (no text, no tool calls) is reported with `empty_response_diagnosis`: `finish_reason` plus the output tokens spent. `MaxTokens` there means the budget was consumed before any content, which is a configuration problem that otherwise presents as a broken model. A response with *nothing* in it — no content, no reasoning, no tool call — never reaches this: rig rejects it as a `CompletionError` first, so the metadata is lost. The cap-exhaustion case does reach it, since rig folds provider reasoning into the choice
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
- Request path: the endpoint `chatgpt.com/backend-api/codex/responses` speaks the OpenAI **Responses** API but rig's high-level responses client is unusable here (it hardcodes `instructions: None`). So `CodexClient` reuses rig's public `responses_api::{CompletionRequest, CompletionResponse}` types for request **lowering** and response **parsing** but does the HTTP itself to satisfy the backend's quirks: top-level `instructions` = the system prompt (taken out of the rig request so it isn't also added as an input item; a completion with no system prompt is rejected up front), `stream: true` (mandatory), `store: false` (merged into `additional_params`), `prompt_cache_key` = an opaque SHA-256-derived key over the stable model/system/tool family (unchanged as history grows, naturally partitioned by project/preset), and `max_output_tokens` omitted (rejected outright). Before lowering/sending, missing assistant/tool-result `call_id`s are filled from generic ids, so Alloy histories produced by non-Responses providers remain replayable by Codex. Nothing rewrites the serialized body: rig lowers assistant text into a valid Responses shape itself (a bare-string `AssistantInput` for the id-less messages nitpicker builds), and since 0.41 it also drops function-call item ids that aren't provider-native `fc_...` ids, pairing a call to its output by `call_id` alone. Because `store: false` is stateless, the terminal `response.completed` event carries an empty `output`, so items are accumulated from `response.output_item.done` events and injected before rig parses. Finish reason: tool calls → ToolUse; else `incomplete_details.reason == "max_output_tokens"` → MaxTokens; else Stop.
- **Multi-turn reasoning under `store: false`**: a reasoning item the model returns this turn is, by default, replayed next turn as a bare `rs_...` id — which the stateless backend can't resolve (`HTTP 404 — Items are not persisted when store is set to false`), so every loop past turn 1 died. `build_body` therefore merges `include: ["reasoning.encrypted_content"]` alongside `store: false`. rig round-trips that blob both ways (response `Output::Reasoning.encrypted_content` → core `ReasoningContent::Encrypted` → request reasoning input item), so reasoning replays **inline** rather than by id. rig only auto-adds that `include` when a `reasoning` config is present, which nitpicker doesn't set, hence the explicit merge.
- `init` detection (`detect.rs::detect_codex`) surfaces a logged-in Codex CLI as a commented `auth = "codex"` reviewer (`gpt-5.6-sol`) via `codex::auth_available()` (reuses the same `auth.json` parse, so API-key-mode files don't qualify), gated on no `openai`-named provider already detected.
- Research-only framing in user-facing copy (third-party use of the Codex OAuth client is arguably against OpenAI ToS), mirroring the AG2 gemini path. Tokens are never logged.

### Tools (`tools.rs`)

Tools return `Err` for any failure (bad arguments, sandbox rejections, a non-zero git exit). `run_agent` does not abort on it: `execute_tool_call` feeds the message back to the model as `"Error: ..."` text so it can self-correct, and records the call with `ToolCallStatus::Error` plus the truncated message as the trajectory `result`. A tool must never encode a failure as successful `Ok` text — the status would then depend on sniffing stdout, and a file that happens to start with `Error:` is not a failure.

`GitTool` only allows a fixed allowlist of read-only subcommands. Commands are passed directly to `Command::new("git").args(tokens)` — no shell involved. Because of that, `reject_shell_syntax` refuses exact operator tokens (`|`, `&&`, `>`, `2>&1`, …, plus `$(`/backtick prefixes) with an error naming the `blame -L`/`read_file` alternatives: models glue pipes onto git out of habit, and a `show rev:path | sed` that reaches git as literal argv either fails loudly or succeeds with the wrong output and gets "fixed" with a whole-file dump that the 50k cap then truncates. Matching is per whole token, so `--format=%h|%an` stays legal. The subcommand allowlist (`ALLOWED_GIT_SUBCOMMANDS`) deliberately excludes the `branch`/`tag` porcelain, whose read and write modes can't be told apart by arguments (attempting to validate them as read-only repeatedly leaked ref-creation bypasses via abbreviations, `=`-glued values, the `--` marker, and value-taking flags absorbing the next token). Ref listing/queries are served instead by the read-only plumbing `for-each-ref`/`show-ref`, which have **no ref-creating/deleting mode** — they are safe by construction for any argument shape (`for-each-ref --contains <sha> refs/heads/` covers branch/tag queries). The remaining argument check, `ensure_readonly_git`, has three layers: (1) across **all** subcommands, reject any argument value (the token, or the part after `=`) that is an absolute path or contains a `..` component — git flags like `diff --no-index <abs>` / `blame --contents <abs>` otherwise read straight from the filesystem, bypassing the canonicalize sandbox that `read_file`/`grep`/`glob` enforce; (2) deny those two filesystem-reading flags by name (incl. abbreviations via `long_flag_matches`, which also closes a relative-but-symlinked path); (3) block the output-to-file flag (`--output`/`-o`, including long-option abbreviations like `--out`) — but only on `diff`/`log`/`show`, the subcommands that actually support it (elsewhere `-o` is read-only, e.g. `ls-files -o` = `--others`). (A glued short-flag value with no `=`, e.g. `-O<orderfile>`/`-X<file>`, escapes layer 1's value extraction but those flags don't print file contents, so they aren't a content-leak vector; the content-leaking `--no-index`/`--contents` are long flags caught by layer 2.) All git commands also run with `GIT_OPTIONAL_LOCKS=0` so a nominally-read command (e.g. `git status`) can't rewrite `.git/index` stat caches or contend on `index.lock` against the user's repo in `pr` in-place mode.

`tool_definitions` sorts by tool name. The order is load-bearing for cost, not tidiness: the tool schemas open every request, so providers that cache on prefix identity re-prefill the whole conversation when they are reordered — and `HashMap` iteration order varies per map instance, with `debate.rs` building a fresh map every turn. Measured against `api.kimi.com`, a reshuffled array alone drops the cache read from 3456 tokens to 0. `run_agent` also separates stable advertised definitions from its per-turn execution allowlist: single-terminal and text-only terminal turns keep the complete schema array, constrain the provider with `ToolChoice::Specific`/`None`, and independently reject disallowed calls before invoking a tool. Multiple terminal tools fall back to advertising only that subset because the provider-neutral `ToolChoice::Required` cannot restrict execution to several named tools. A tool that is impossible for an agent's whole lifetime (for example `spawn_subagent` at maximum depth) is omitted consistently from every turn instead.

`GrepTool` recursively searches files and skips binary files. Context loading for `CLAUDE.md` / `AGENTS.md` also skips binary files.

Tool outputs are intentionally a bit self-describing: `read_file` includes file/range headers, `glob`/`grep` return explicit no-match messages, and truncation messages say when output is partial. `read_file` swaps an inverted `start_line`/`end_line` pair and says so in the header instead of erroring (models reverse them often enough that the error was a pure wasted turn) or clamping to one line. Because failures are `Err`, `reflect` counts failed git pipes and bad ranges as errors rather than successes.

### Session artifacts (`session.rs`)

- When `[defaults].log_trajectories = true`, nitpicker writes session artifacts under `~/.nitpicker/sessions/session-<timestamp>-<pid>/`
- Reviewer and debate-turn traces are stored as per-agent JSONL files; each record's `agent` field names the agent that wrote it — top-level agents use their file's stem (`reviewer-<i>-<name>` / `<side>-<round>`), subagents write into the same file under `<parent>/subagent-<id>` — because `reflect` merges a session's files by timestamp and the label is the only thing keeping interleaved agents attributable
- A failed `spawn_subagent` appends an error completion record in the parent's trace (a successful one deliberately doesn't — the subagent's own records, written under its own label to the shared session file, carry its result); `compact` records use the same 1-based turn numbering as tool records and name the turn they precede
- Appends are flushed before returning, so write errors propagate to the caller's warn and `process::exit` paths can't drop the tail record
- Final synthesized output is saved as `aggregation.json`; on parallel runs it carries a per-job outcome list (`jobs`: label, preset, ok) — the durable record of jobs whose failures are otherwise only transient logs (a client-build failure writes no trace file at all) — and preset debate runs carry the analogous `lanes`. A synthesis failure *after* job/lane collection still writes the record: `error` set, `text` empty, jobs/lanes intact (all three post-collection failure paths in `review.rs`, debate's meta failure, **and** both refuse-to-synthesize bails — total failure is when the record matters most; a failed record write warns rather than masking the original error). `ToolCallRecord` carries `model` when the client reports the turn's selected model — the only durable per-turn attribution on alloy runs (text-only turns write no record; compaction records stay unattributed)
- `attribution.json` (written by both `run_review` and `run_debate` once the logger exists; a write failure warns) records the nitpicker package version, the build-time git revision (`build.rs`, `-dirty` suffix on an unclean tree, absent outside a checkout), and a SHA-256 over the rendered protocol prompts (`prompts::protocol_prompt_sha256`: every role/scope system prompt and synthesis preamble with placeholder preset/rubric, plus the library's subagent/compaction/final-turn prompts). The fingerprint covers orchestration instructions only, never user tasks or configured rubrics, so `reflect` can compare cohorts that ran the same protocol over different targets. It is a sidecar rather than an `AggregationRecord` field so the library's public record literal stays source-compatible

### Gemini AG2 proxy (`gemini_proxy/`)

Gated behind the off-by-default `antigravity` cargo feature. The whole module compiles out when the feature is off, which drops `axum`, `keyring`, and `uuid` from the default build; `src/proxy.rs::GeminiProxy` owns the optional startup (an inert handle feature-off; a startup *failure* — e.g. missing/expired keyring token — also degrades to an inert handle rather than aborting the run, so only proxy-needing clients fail at build time, per-job in parallel mode with the startup cause attached) and `provider.rs`/`review.rs`/`debate.rs` thread only its base URL (`Option<&str>`/`Option<String>`) downstream so their signatures compile feature-off, and the proxy predicates (`*_needs_gemini_proxy`, `ProviderType::is_gemini`) plus `create_gemini_client_with_proxy` and `detect::detect_agy_keyring` are all `#[cfg(feature = "antigravity")]`. The config validator bails with a `--features antigravity` hint if `auth = "agy-keyring"` is configured without it (mirrors the azure gate). Combined with a size-tuned `[profile.release]` (`opt-level = "z"`, `lto = "thin"`, `strip = true`; `panic` left at `unwind`), the default release binary is ~8.7M (down from ~16M).

When `auth = "agy-keyring"` is set for a Gemini reviewer/aggregator (feature-on), nitpicker:
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
- Parallel review spawns every reviewer×preset job eagerly (no job-count cap); within an agent, a turn's tool calls run concurrently and all in-flight LLM calls share a global cap of `MAX_CONCURRENT_LLM_CALLS` (16), which is the only concurrency bound in both modes
- Tool results are truncated to 50k bytes before being sent to the LLM
- Git tool output is truncated to 50k chars
- Agent and debate turn loops default to 100 turns and can be overridden via config or CLI
- Agent turns and compaction (which inherits the agent's cap) send **no** `max_tokens` by default, so the provider's own per-model limit applies; `[[reviewer]].max_tokens` sets one explicitly and `[aggregator].max_tokens` defaults to `DEFAULT_AGGREGATOR_MAX_TOKENS` (16384). A cap budgets reasoning *plus* the answer, so one below what the model reasons through returns empty content rather than a shorter answer. `0` is rejected at config time; `auth = "codex"` ignores the setting (the endpoint rejects `max_output_tokens`)
- Context files (`CLAUDE.md`, `AGENTS.md`) are limited to 50k chars
- Prefer `match` over `if let` for better exhaustiveness checking, even if it requires a `_ => unreachable!()` arm
