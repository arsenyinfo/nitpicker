Focus strictly on information vital for continuing the current task. Omit conversational filler,
raw search/tool outputs, and large verbatim excerpts.

Use exactly this Markdown structure inside the tags:
<summary>
## Objective
[1-2 sentences stating the task and desired outcome.]
## Key Discoveries
- [Important facts, constraints, decisions, or issues established so far.]
## Relevant Artifacts and Locations
- [Files, directories, records, or other sources that matter, with a brief reason.]
## Explored Territory
- [Areas, sources, or hypotheses already investigated so work is not repeated.]
## Last Action and Immediate Context
- [The most recent tool calls, their results, and what the agent was verifying.]
## Open Questions and Next Steps
- [Unresolved anomalies, pending work, and the next concrete checks.]
</summary>

Return exactly one tagged block starting with <summary> and ending with </summary>. Include
nothing outside the tags.
