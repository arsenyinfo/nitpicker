# Changelog

Notable user-visible changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.9.3] - 2026-08-13

`nitpicker-agent` 0.5.0

### Added

- Show an animated repository activity indicator in supported terminal tab/window titles during
  interactive review runs, without changing redirected or `pr --json` output.
- Persist staged reviewer and debate verdicts in session aggregation records so reflection can
  compare execution trajectories with their intermediate and final outcomes.

### Changed

- Calibrate `reflect` with deterministic trace metrics, evidence-counted cross-session claims,
  explicit confidence/tradeoffs, critic-model synthesis, omission of rejected hypotheses from the
  final report, and no shared temp dump.
- Separate debate responsibilities across review and `ask`: Actor/Reviewer owns discovery and
  recall, while Critic/Validator consolidates evidence-based objections to the submitted set and
  does not hunt for new findings/options or manufacture disagreement.
- Give discovery roles an early disjoint delegation default for multi-surface work while limiting
  validation-role subagents to targeted verification of submitted claims.
- Inject one deterministic, commit-pinned review snapshot before fan-out, including base/HEAD,
  merge base, exact comparison semantics, working-tree state, and committed/uncommitted file maps.
- **Breaking (`nitpicker-agent`)**: `AggregationRecord` adds the required
  `verdicts: Vec<VerdictRecord>` field; pre-change aggregation records are not accepted by
  `reflect`.

### Fixed

- Keep diff reviews available when the detected base and HEAD have no merge base by falling back
  to a direct tree comparison, and bound frozen snapshot file maps by rendered bytes.
- Reject token-limit-truncated agent answers instead of publishing partial reports as complete.
- Preserve final per-lane verdicts and both early and late tool evidence within independently
  bounded `reflect` sections.
- Count identical same-turn tool calls as distinct reflection invocations while still collapsing
  a failed subagent spawn's started/error lifecycle pair.
- Persist debate verdict success from the turn outcome rather than inferring it from display text.
- Restore `auth = "agy-keyring"`: the proxy's `loadCodeAssist` request now sends a
  `User-Agent`, without which Google's backend classified the client as the deprecated
  "Gemini Code Assist for individuals" tier and returned no project, 503-ing every request.

## [0.9.2] - 2026-08-10

`nitpicker-agent` 0.4.0

### Added

- Add opt-in ordered reviewer fallback with `fallback = true` or `--fallback`, including
  fallback-capable Alloy routing and aggregator synthesis.

### Changed

- Enable fallback in generated configs whenever `init` can provide at least two reviewer routes.
- **Breaking (`nitpicker-agent`)**: `DefaultsConfig` adds `fallback: Option<bool>`; downstream
  struct literals must set `fallback: None` when fallback is not configured.

### Fixed

- Improve fallback reliability across provider limits, unavailable routes, concurrent review jobs,
  debate lanes, and subagents.
- Require debate agents to submit structured verdicts before accepting a turn as successful.
- Resolve `--repo` through Git so linked worktrees and paths inside a worktree are supported.

## [0.9.1] - 2026-08-09

### Changed

- Debate progress uses one live TTY row per lens, and final findings identify their lens.
- Review prompts reject premature convergence and focus follow-up rounds on unresolved work.
- Security and simplicity checks better distinguish real authority gains, dead code, and live
  compatibility contracts.

### Fixed

- Reject comma-containing custom preset names, which are ambiguous in lens lists.

## [0.9.0] - 2026-08-09

`nitpicker-agent` 0.3.0

- **Review presets**: four universal defaults (`correctness`, `security`, `performance`,
  `simplicity`), opt-in domain angles (`ai-systems`, `ml-rigor`, `tone`), standalone `general`,
  and project-defined `[presets.<name>]` tables. Parallel mode fans out reviewers × presets;
  debate mode runs one concurrent Reviewer/Validator lane per preset with a global meta-review.
- `pr --json` adds resolved `presets`, `degraded`, and per-preset `coverage`; session and
  transcript artifacts identify their jobs and lanes.
- `pr` reads repository policy from the trusted PR base branch rather than the contributor's
  working tree, with explicit remote-host and object-id checks.
- Synthesis failures persist their job/lane outcomes for `reflect`; trajectories record the
  selected model per turn in alloy runs.
- Review and debate jobs share one 16-call provider concurrency limit. Preset selection and
  context-window failures now fail early with targeted diagnostics.
- Review/debate prompts drop withdrawn findings and chronology; cleanly converged lanes expose
  only their self-contained final round to synthesis.

## [0.8.5] - 2026-08-05

`nitpicker-agent` 0.2.1

- Session recorder fixes: unique per-agent record identities, failed subagent spawns logged,
  1-based `compact` turn numbers, and flushed appends.
- `reflect` no longer renders successful spawns or blocked cycles as failures.

## [0.8.4] - 2026-08-05

- Added repeatable `--context-file <PATH>` injection for review, `ask`, and `pr`, restricted to
  regular files and capped at 256 KiB including wrappers.
- `--repo`, `--config`, and `--verbose` are global flags accepted before or after subcommands.

## [0.8.3] - 2026-08-05

`nitpicker-agent` 0.2.0

- Stable, name-sorted tool definitions improve provider prompt-cache hits.
- Cache-aware token metering and JSON usage fields normalize provider cache shapes.
- Removed fixed agent output caps; added per-reviewer `max_tokens` and a 16,384-token aggregator
  default.
- Improved empty-response diagnostics and recovery from malformed `<think>` blocks.
- Made compaction best-effort and isolated output caps per model in alloy pools.
- Bumped `rig-core` to 0.41.
- **Breaking (`nitpicker-agent`)**: `AgentResult` now carries `usage: TokenUsage`, and
  `AlloyClient::new` accepts `Vec<AlloySlot>`.

## [0.8.2] - 2026-07-14

`nitpicker-agent` 0.1.2

- Prevented debate subagents from inheriting `submit_verdict`.
- Added dedicated compaction prompts and context-only project-instruction wrappers.
- Added `ReviewScope` and `--analyze` static-analysis mode.

## [0.8.1] - 2026-07-03

- Bumped `rig-core` to 0.39 and simplified Codex assistant-text normalization.

## [0.8.0] - 2026-06-16

- Extracted the reusable agent loop into the published `nitpicker-agent` crate.
- Added subagent-system-prompt customization.

## [0.7.1] - 2026-06-14

- Added exit code 3 for degraded verdicts and refined interactive progress.
- Hardened git sandboxing, provider error classification, and PR checkout safety.

## [0.7.0] - 2026-06-07

- Added experimental Codex subscription authentication.
- Fixed Codex multi-turn reasoning and gated keyring authentication under `antigravity`.

## [0.6.3] - 2026-06-06

- Restricted additional git plumbing, made detached-HEAD restoration panic-safe, and hardened
  retry classification.

## [0.6.2] - 2026-06-05

- Added machine-readable `nitpicker pr --json` output.
- Fixed macOS symlink workspace canonicalization under `pr --clone`.

## [0.6.1] - 2026-06-05

- Made subagent waves concurrent under a global semaphore.

## [0.6.0] - 2026-06-02

- Added Entra ID authentication for OpenAI and Anthropic providers behind `azure`.

## [0.5.1] - 2026-05-25

- Improved PR checkout safety and added partial-clone `--clone` behavior.

## [0.5.0] - 2026-05-24

- Added Gemini keyring authentication and removed the legacy browser OAuth flow.

## [0.4.0] - 2026-05-17

- Added Alloy mode for randomized reviewer-model pools.

## [0.3.0–0.3.3] - 2026-05-06–2026-05-11

- Added OpenRouter free-model selection, trajectory logging, and `reflect`.

## [0.1.2–0.2.3] - 2026-03-15–2026-05-01

- Added PR review, debate synthesis, proactive compaction, bounded subagents, and atomic locks.
