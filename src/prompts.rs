const INVESTIGATION_GUIDANCE: &str = include_str!("../prompts/protocol/investigation-guidance.md");
const FINDING_FIELDS: &str = include_str!("../prompts/protocol/finding-schema.md");
const OPTIONS_SCHEMA: &str = include_str!("../prompts/protocol/options-schema.md");
const OPTIONS_SCHEMA_WITH_NO_CONSENSUS: &str =
    include_str!("../prompts/protocol/options-schema-no-consensus.md");
const REVIEWER_TEMPLATE: &str = include_str!("../prompts/protocol/reviewer.md");
const ASK_TEMPLATE: &str = include_str!("../prompts/protocol/ask.md");
const REVIEW_AGGREGATOR_TEMPLATE: &str = include_str!("../prompts/protocol/review-aggregator.md");
const ASK_AGGREGATOR_TEMPLATE: &str = include_str!("../prompts/protocol/ask-aggregator.md");
const PRESET_SUBAGENT_TEMPLATE: &str = include_str!("../prompts/protocol/preset-subagent.md");
const DEBATE_ACTOR_TOPIC_TEMPLATE: &str = include_str!("../prompts/protocol/debate-actor-topic.md");
const DEBATE_ACTOR_REVIEW_TEMPLATE: &str =
    include_str!("../prompts/protocol/debate-actor-review.md");
const DEBATE_VALIDATOR_TOPIC_TEMPLATE: &str =
    include_str!("../prompts/protocol/debate-validator-topic.md");
const DEBATE_VALIDATOR_REVIEW_TEMPLATE: &str =
    include_str!("../prompts/protocol/debate-validator-review.md");
const DEBATE_META_TOPIC_TEMPLATE: &str = include_str!("../prompts/protocol/debate-meta-topic.md");
const DEBATE_META_REVIEW_TEMPLATE: &str = include_str!("../prompts/protocol/debate-meta-review.md");

const NO_FINDINGS: &str = "No findings. Great job! 🎉";

/// Render one compile-time prompt template. Template placeholders are deliberately tiny and
/// dependency-free: prompt authors can audit the Markdown directly, while Rust owns only the
/// values that vary by run. Validate the template before inserting values so a custom rubric
/// containing `{{...}}` remains ordinary prompt text rather than looking unresolved.
fn render(template: &str, values: &[(&str, &str)]) -> String {
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        let after_open = &rest[start + 2..];
        let end = after_open
            .find("}}")
            .unwrap_or_else(|| panic!("unterminated prompt placeholder in {template:?}"));
        let name = &after_open[..end];
        assert!(
            values.iter().any(|(candidate, _)| *candidate == name),
            "prompt template uses unknown placeholder {{{{{name}}}}}"
        );
        rest = &after_open[end + 2..];
    }

    let mut rendered = template.trim().to_string();
    for (name, value) in values {
        rendered = rendered.replace(&format!("{{{{{name}}}}}"), value);
    }
    rendered
}

/// Whether the review targets a change (diff/PR) or existing code (`--analyze`).
/// Change-attribution rules only make sense for the former.
#[derive(Clone, Copy)]
pub enum ReviewScope {
    Diff,
    Static,
}

impl ReviewScope {
    fn target_noun(&self) -> &'static str {
        match self {
            ReviewScope::Diff => "changes under review",
            ReviewScope::Static => "code under analysis",
        }
    }

    fn finding_scope_rule(&self) -> &'static str {
        match self {
            ReviewScope::Diff => {
                "- Only flag problems in the current (post-change) code. Do not narrate improvements the diff made — \"X now correctly does Y\" is not a finding.\n"
            }
            ReviewScope::Static => {
                "- You are reviewing existing code, not a change: prioritize by impact and severity, not by how recently the code was written.\n"
            }
        }
    }

    fn synthesis_drop_clause(&self) -> &'static str {
        match self {
            ReviewScope::Diff => {
                "items that narrate fixes the diff already landed or praise correct code"
            }
            ReviewScope::Static => "items that praise correct code",
        }
    }

    fn no_landed_fixes_clause(&self) -> &'static str {
        match self {
            ReviewScope::Diff => ", no recollection of fixes the diff already landed",
            ReviewScope::Static => "",
        }
    }

    fn critic_reality_check(&self) -> &'static str {
        match self {
            ReviewScope::Diff => {
                "1. Is this a real problem in the current (post-change) code, or is the reviewer narrating a fix the diff already made? Reject narration of landed improvements.\n"
            }
            ReviewScope::Static => {
                "1. Is this a real problem in the code as it exists, or a claim about code that is not actually there? Verify the premise.\n"
            }
        }
    }
}

pub enum TaskMode {
    Review(ReviewScope),
    Ask,
}

impl TaskMode {
    /// Review workers receive exactly one rubric. Its slots are last in the template, keeping
    /// the protocol prefix identical across same-model preset jobs for provider prompt caching.
    pub fn system_prompt(&self, preset: Option<&crate::presets::ReviewPreset>) -> String {
        match (self, preset) {
            (TaskMode::Review(scope), Some(preset)) => render(
                REVIEWER_TEMPLATE,
                &[
                    ("TARGET", scope.target_noun()),
                    ("SCOPE_RULE", scope.finding_scope_rule()),
                    ("INVESTIGATION_GUIDANCE", INVESTIGATION_GUIDANCE.trim()),
                    ("FINDING_SCHEMA", FINDING_FIELDS.trim()),
                    ("NO_FINDINGS", NO_FINDINGS),
                    ("PRESET_NAME", &preset.name),
                    ("RUBRIC", &preset.prompt),
                ],
            ),
            (TaskMode::Review(_), None) | (TaskMode::Ask, Some(_)) => {
                unreachable!("Review runs take exactly one preset per worker; Ask takes none")
            }
            (TaskMode::Ask, None) => {
                render(ASK_TEMPLATE, &[("OPTIONS_SCHEMA", OPTIONS_SCHEMA.trim())])
            }
        }
    }

    pub fn initial_message(&self, user_prompt: &str) -> String {
        if user_prompt.trim().is_empty() {
            return String::new();
        }
        match self {
            TaskMode::Review(_) => format!("Focus your review on: {user_prompt}\n\n"),
            TaskMode::Ask => format!("Question to answer: {user_prompt}\n\n"),
        }
    }

    /// Review synthesis carries every active preset's name and full rubric, including project
    /// overrides; Ask has no preset roster.
    pub fn reduce_prompt(
        &self,
        task: &str,
        combined: &str,
        presets: Option<&[crate::presets::ReviewPreset]>,
    ) -> String {
        let inputs = match self {
            TaskMode::Review(_) => "Individual reviews to synthesize",
            TaskMode::Ask => "Individual answers to synthesize",
        };
        let roster = match (self, presets) {
            (TaskMode::Review(_), Some(presets)) => format!("{}\n\n", preset_roster(presets)),
            (TaskMode::Ask, None) => String::new(),
            (TaskMode::Review(_), None) | (TaskMode::Ask, Some(_)) => {
                unreachable!("Review synthesis takes the active presets; Ask takes none")
            }
        };
        match task.trim().is_empty() {
            true => format!("{roster}{inputs}:\n\n{combined}"),
            false => {
                format!(
                    "Original task given to each agent:\n{task}\n\n{roster}{inputs}:\n\n{combined}"
                )
            }
        }
    }

    pub fn aggregator_preamble(&self) -> String {
        match self {
            TaskMode::Review(scope) => render(
                REVIEW_AGGREGATOR_TEMPLATE,
                &[
                    ("NO_LANDED_FIXES", scope.no_landed_fixes_clause()),
                    ("DROP_CLAUSE", scope.synthesis_drop_clause()),
                    ("FINDING_SCHEMA", FINDING_FIELDS.trim()),
                    ("NO_FINDINGS", NO_FINDINGS),
                ],
            ),
            TaskMode::Ask => render(
                ASK_AGGREGATOR_TEMPLATE,
                &[("OPTIONS_SCHEMA", OPTIONS_SCHEMA_WITH_NO_CONSENSUS.trim())],
            ),
        }
    }
}

/// Every active preset's name and full rubric for the final synthesizer.
pub(crate) fn preset_roster(presets: &[crate::presets::ReviewPreset]) -> String {
    let blocks: Vec<String> = presets
        .iter()
        .map(|p| format!("### {}\n{}", p.name, p.prompt))
        .collect();
    format!(
        "Active review presets — each input below investigated exactly one of these angles:\n\n{}",
        blocks.join("\n\n")
    )
}

/// Preset-aware subagent prompt: the library's generic contract plus the parent lane's rubric.
pub fn preset_subagent_prompt(preset: &crate::presets::ReviewPreset) -> String {
    render(
        PRESET_SUBAGENT_TEMPLATE,
        &[
            ("BASE", nitpicker_agent::prompts::subagent_system_prompt()),
            ("PRESET_NAME", &preset.name),
            ("RUBRIC", &preset.prompt),
        ],
    )
}

pub enum DebateMode {
    Topic,
    Review(ReviewScope),
}

impl DebateMode {
    pub fn actor_role(&self) -> &'static str {
        match self {
            DebateMode::Topic => "Actor",
            DebateMode::Review(_) => "Reviewer",
        }
    }

    pub fn critic_role(&self) -> &'static str {
        match self {
            DebateMode::Topic => "Critic",
            DebateMode::Review(_) => "Validator",
        }
    }

    pub(crate) fn actor_system(&self, preset: Option<&crate::presets::ReviewPreset>) -> String {
        match (self, preset) {
            (DebateMode::Review(_), None) | (DebateMode::Topic, Some(_)) => {
                unreachable!("Review lanes take exactly one preset; Topic takes none")
            }
            (DebateMode::Topic, None) => render(
                DEBATE_ACTOR_TOPIC_TEMPLATE,
                &[("INVESTIGATION_GUIDANCE", INVESTIGATION_GUIDANCE.trim())],
            ),
            (DebateMode::Review(scope), Some(preset)) => render(
                DEBATE_ACTOR_REVIEW_TEMPLATE,
                &[
                    ("TARGET", scope.target_noun()),
                    ("SCOPE_RULE", scope.finding_scope_rule()),
                    ("FINDING_SCHEMA", FINDING_FIELDS.trim()),
                    ("NO_FINDINGS", NO_FINDINGS),
                    ("INVESTIGATION_GUIDANCE", INVESTIGATION_GUIDANCE.trim()),
                    ("PRESET_NAME", &preset.name),
                    ("RUBRIC", &preset.prompt),
                ],
            ),
        }
    }

    pub(crate) fn critic_system(&self, preset: Option<&crate::presets::ReviewPreset>) -> String {
        match (self, preset) {
            (DebateMode::Review(_), None) | (DebateMode::Topic, Some(_)) => {
                unreachable!("Review lanes take exactly one preset; Topic takes none")
            }
            (DebateMode::Topic, None) => render(
                DEBATE_VALIDATOR_TOPIC_TEMPLATE,
                &[("INVESTIGATION_GUIDANCE", INVESTIGATION_GUIDANCE.trim())],
            ),
            (DebateMode::Review(scope), Some(preset)) => render(
                DEBATE_VALIDATOR_REVIEW_TEMPLATE,
                &[
                    ("REALITY_CHECK", scope.critic_reality_check()),
                    ("INVESTIGATION_GUIDANCE", INVESTIGATION_GUIDANCE.trim()),
                    ("PRESET_NAME", &preset.name),
                    ("RUBRIC", &preset.prompt),
                ],
            ),
        }
    }

    pub(crate) fn meta_instruction(&self) -> &'static str {
        match self {
            DebateMode::Topic => "Debate transcript to synthesize into the final answer.",
            DebateMode::Review(_) => "Debate transcript to synthesize into the final summary.",
        }
    }

    pub(crate) fn meta_preamble(&self) -> String {
        match self {
            DebateMode::Topic => render(
                DEBATE_META_TOPIC_TEMPLATE,
                &[("OPTIONS_SCHEMA", OPTIONS_SCHEMA_WITH_NO_CONSENSUS.trim())],
            ),
            DebateMode::Review(scope) => render(
                DEBATE_META_REVIEW_TEMPLATE,
                &[
                    ("NO_LANDED_FIXES", scope.no_landed_fixes_clause()),
                    ("DROP_CLAUSE", scope.synthesis_drop_clause()),
                    ("FINDING_SCHEMA", FINDING_FIELDS.trim()),
                    ("NO_FINDINGS", NO_FINDINGS),
                ],
            ),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            DebateMode::Topic => "debate",
            DebateMode::Review(_) => "review-debate",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::ReviewPreset;

    fn preset(name: &str, rubric: &str) -> ReviewPreset {
        ReviewPreset {
            name: name.to_string(),
            prompt: rubric.to_string(),
        }
    }

    /// A worker carries exactly its assigned rubric and same-scope prompts only diverge at
    /// the final preset slot, preserving cross-preset provider cache sharing.
    #[test]
    fn review_system_prompt_carries_exactly_its_preset() {
        let a = preset("angle-a", "RUBRIC-MARKER-A");
        let b = preset("angle-b", "RUBRIC-MARKER-B");
        for scope in [ReviewScope::Diff, ReviewScope::Static] {
            let prompt_a = TaskMode::Review(scope).system_prompt(Some(&a));
            let prompt_b = TaskMode::Review(scope).system_prompt(Some(&b));
            assert!(prompt_a.contains("RUBRIC-MARKER-A"));
            assert!(!prompt_a.contains("RUBRIC-MARKER-B"));
            assert!(prompt_b.contains("RUBRIC-MARKER-B"));

            let shared_prefix_len = prompt_a
                .bytes()
                .zip(prompt_b.bytes())
                .take_while(|(x, y)| x == y)
                .count();
            let rubric_a_at = prompt_a.find("angle-a").expect("angle name present");
            assert!(shared_prefix_len >= rubric_a_at);
        }
    }

    #[test]
    fn reduce_prompt_carries_every_active_preset_rubric() {
        let presets = [
            preset("angle-a", "RUBRIC-MARKER-A"),
            preset("angle-b", "RUBRIC-MARKER-B"),
        ];
        let out = TaskMode::Review(ReviewScope::Diff).reduce_prompt(
            "the task",
            "the reviews",
            Some(&presets),
        );
        for needle in [
            "angle-a",
            "RUBRIC-MARKER-A",
            "angle-b",
            "RUBRIC-MARKER-B",
            "the task",
            "the reviews",
        ] {
            assert!(out.contains(needle), "missing {needle}");
        }
    }

    #[test]
    fn ask_reduce_prompt_takes_no_roster() {
        let out = TaskMode::Ask.reduce_prompt("the question", "the answers", None);
        assert!(out.contains("the question"));
        assert!(out.contains("the answers"));
    }

    #[test]
    fn preset_subagent_prompt_extends_the_generic_contract_with_the_rubric() {
        let out = preset_subagent_prompt(&preset("angle-a", "RUBRIC-MARKER-A"));
        assert!(out.starts_with(nitpicker_agent::prompts::subagent_system_prompt()));
        assert!(out.contains("RUBRIC-MARKER-A"));
    }

    #[test]
    fn every_external_template_renders_all_of_its_placeholders() {
        let p = preset("angle", "rubric");
        let prompts = [
            TaskMode::Review(ReviewScope::Diff).system_prompt(Some(&p)),
            TaskMode::Review(ReviewScope::Static).system_prompt(Some(&p)),
            TaskMode::Ask.system_prompt(None),
            TaskMode::Review(ReviewScope::Diff).aggregator_preamble(),
            TaskMode::Ask.aggregator_preamble(),
            preset_subagent_prompt(&p),
            DebateMode::Topic.actor_system(None),
            DebateMode::Review(ReviewScope::Diff).actor_system(Some(&p)),
            DebateMode::Topic.critic_system(None),
            DebateMode::Review(ReviewScope::Static).critic_system(Some(&p)),
            DebateMode::Topic.meta_preamble(),
            DebateMode::Review(ReviewScope::Diff).meta_preamble(),
        ];
        assert!(prompts.iter().all(|prompt| !prompt.contains("{{")));
    }

    #[test]
    fn custom_rubric_may_contain_template_like_text() {
        let p = preset("custom", "Explain {{PROJECT_TOKEN}} exactly.");
        let out = TaskMode::Review(ReviewScope::Diff).system_prompt(Some(&p));
        assert!(out.contains("{{PROJECT_TOKEN}}"));
    }
}
