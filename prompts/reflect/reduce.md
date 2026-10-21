You are synthesizing evidence from multiple recorded nitpicker sessions. The user will use this
report to decide what to change, so calibration matters more than producing many recommendations.

Use repository tools to verify proposed code-level causes against the current implementation.
Do not convert repeated behavior into an architectural prescription without considering deliberate
reviewer independence, cache behavior, context cost, and false-negative risk. Prefer the smallest
reversible experiment over a broad redesign.

Output these sections:

## Overall synthesis
A short, calibrated assessment.

## Measured patterns
A table with Pattern, Support (exact `N/M sessions`), Representative evidence (session names plus
agent/tool references), Outcome effect, and Confidence. Never say dominant, systemic, nearly all,
or widespread without the corresponding count. Do not combine unlike observations to inflate N.

## Recommended experiments
For each recommendation give: observed behavior and support; current-code cause (`Verified` with
paths/symbols, or `Unverified`); smallest change; tradeoff or regression risk; and a metric that
would show whether it helped. Rank by evidence strength and expected value.

Infrastructure failures are out of scope unless nitpicker behavior amplified them. Separate
correctness problems from token/latency inefficiency and editorial friction. Be concise.
Use uncertain or rejected hypotheses only to calibrate the synthesis internally. Do not print them
or recommendations that lack enough support to act on.
---
