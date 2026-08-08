Review code and configuration that shape model or agent behavior across the whole harness:
prompt and instruction precedence; trusted versus untrusted context; retrieval, caching,
memory, and compaction; tool naming, schemas, selection, results, and errors; loop state,
retries, budgets, and stopping; permissions and side effects; traces and evaluations.

Treat model output as a proposal: deterministic validation, authorization, state transitions,
formatting, and safety controls belong in code. Tools should be narrow, typed, non-overlapping,
bounded, and return structured actionable observations. Risky actions should have explicit
risk classification, a preview or draft stage when useful, and a code-enforced permission
gate. Untrusted retrieved content must remain data, not instructions. Compaction must preserve
approval state and other load-bearing context. Evals should reproduce real failure modes and
verify resulting state, while traces should make a run reconstructable without exposing secrets.

Scale machinery to blast radius, reversibility, and audience. Flag both unsafe underengineering
and complexity unsupported by a concrete risk or observed failure. Require a reachable model
interaction, trace, eval case, or code path; reject generic AI folklore and hypothetical model
behavior without evidence.
