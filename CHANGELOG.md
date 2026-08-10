# Changelog

Notable user-visible changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

`nitpicker-agent` 0.4.0

### Added

- Add opt-in ordered reviewer fallback with `fallback = true` or `--fallback`, including
  fallback-capable Alloy routing and aggregator synthesis.

### Changed

- **Breaking (`nitpicker-agent`)**: `DefaultsConfig` adds `fallback: Option<bool>`; downstream
  struct literals must set `fallback: None` when fallback is not configured.

### Fixed

- Require debate agents to submit structured verdicts: a plain-text conclusion gets one
  terminal-only correction turn with forced tool use, then route fallback on noncompliance.
- Treat billing-cycle usage exhaustion as a run-long unavailable route and collapse repeated
  debate-turn failures into one concise operational warning.
- Keep fallback stickiness local to each independent debate lane while sharing run-wide route
  availability, so one lane's request-specific failover cannot reroute another lane.
- In fallback mode, skip experimental OpenRouter free routes whose catalog lookup or smoke test
  fails, while continuing with healthy fixed or auto-resolved routes.
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
