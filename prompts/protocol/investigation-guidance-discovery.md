Scale the investigation to the target. First build a quick map of the relevant code: intent,
affected files, nearby modules, and important call paths. For a small or local target, inspect it
directly. On your first turn, when the target spans multiple independently reviewable modules,
call paths, or hypotheses, default to one early wave of bounded, disjoint subagents. Give each a
self-contained question and surface, keep one central investigation for your own tool calls, then
aggregate their evidence. Do not broadly recrawl a delegated surface; perform only the targeted
verification needed before submitting. Use another wave only when concrete evidence opens a
specific unanswered question. Prefer concluding from established evidence over adding ceremony,
breadth, or latency.
