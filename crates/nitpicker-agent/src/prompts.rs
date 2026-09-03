/// Default system prompt for a spawned subagent. Callers can override it via
/// `AgentConfig::subagent_system_prompt`; this generic prompt is used when none is set.
pub fn subagent_system_prompt() -> &'static str {
    include_str!("../prompts/subagent.md").trim()
}

pub fn compaction_summary_prompt() -> &'static str {
    include_str!("../prompts/compaction-summary.md")
}

pub fn final_turn_wrap_up_prompt() -> &'static str {
    include_str!("../prompts/final-turn-wrap-up.md")
}
