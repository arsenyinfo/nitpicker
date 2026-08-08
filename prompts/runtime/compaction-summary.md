Focus strictly on information vital for code analysis. Omit conversational filler, raw
search/tool outputs, and large blocks of code.

Use exactly this Markdown structure inside the tags:
<summary>
## Review Goal
[1-2 sentences on the core objective of this code exploration or review.]
## Key Findings & Discoveries
- [Major architectural insights, design patterns, or critical issues identified so far.]
## Codebase Map (Relevant Files)
- [Critical files and directories, with a brief note explaining their relevance.]
## Explored Territory
- [Areas, files, or concepts already investigated so work is not repeated.]
## Last Action & Immediate Context
- [The most recent tool calls, their results, and what the agent was verifying.]
## Open Questions & Next Steps
- [Unresolved anomalies, constraints, pending items, and specific next checks.]
</summary>

Return exactly one tagged block starting with <summary> and ending with </summary>. Include
nothing outside the tags.
