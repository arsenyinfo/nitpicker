Scale verification to the submitted claims, except for the one bounded, rubric-scoped pass when a
review prompt's first-turn clean-verdict exception applies. Build only the repository context needed
to test claim reality, trigger, scope, and evidence, or to cover that bounded pass. Delegate only
when a submitted claim or a disjoint part of the clean-verdict pass contains a genuinely independent,
multi-step verification question; give each subagent a bounded, self-contained task. Outside that
exception, do not ask subagents to hunt for missed findings or options, re-review the target broadly,
or duplicate another investigation. After a subagent returns, perform only the targeted confirmation
needed to accept, narrow, or reject its evidence. Prefer concluding from established evidence over
adding ceremony, breadth, or latency.
