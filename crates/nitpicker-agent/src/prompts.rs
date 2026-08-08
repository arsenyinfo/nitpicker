/// Default system prompt for a spawned subagent. Callers can override it via
/// `AgentConfig::subagent_system_prompt`; this generic prompt is used when none is set.
pub fn subagent_system_prompt() -> &'static str {
    include_str!("../../../prompts/runtime/subagent.md").trim()
}
