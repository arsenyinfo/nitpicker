const DELEGATION_GUIDANCE: &str = "Always first build a quick high-level map of the relevant code: change intent, affected files, nearby modules, and major components involved. \
Then write a short working plan that enumerates the disjoint threads worth investigating — separate questions, distinct filesets, individual call paths, tests vs implementation, focused security or performance concerns. \
Subagents spawned in a single turn run in parallel, so the fastest path in wall-clock is to fan out ALL the disjoint threads from your plan as one broad wave of spawn_subagent(task) calls in the same turn, rather than spawning a few and walking the rest serially. \
Use local tools for quick triage and synthesis, but do not try to personally exhaust every branch of the investigation. \
Keep each subagent task bounded and self-contained so it converges quickly, and do not spawn overlapping or near-duplicate subagents that would reread the same files for the same question. \
After the wave returns, synthesize what is now established. Spawn another wave only when a concrete finding demands a specific follow-up — not as a routine next step. Each additional serial wave adds latency, so prefer to conclude from the evidence you already gathered.";

const NO_FINDINGS: &str = "No findings. Great job! 🎉";

const FINDING_FIELDS: &str = "<One sentence title about the issue>\n\
- Priority: <P0 - P3>\n\
- Location: <path:line or line range>\n\
- Scenario: <concrete trigger when the scenario fires, or \"always\">\n\
- Potential solution: <a concrete direction such as \"replace X with Y\" or \"validate at boundary Z\" — actionable but not necessarily patch-level; \"consider refactoring\" is too vague>\n\
- Uncertainty: <what specifically you are unsure about and what would confirm or disprove it — omit the line if fully confident>";

const OPTIONS_SCHEMA: &str = "Options considered\n\
<option name>\n\
- What it is: ...\n\
- Fits when: ...\n\
- Drawbacks: ...\n\
(repeat per option)\n\n\
Recommendation\n\
<which option and why>\n\n\
Caveats (omit if none)\n\
<context-dependent considerations>";

const OPTIONS_SCHEMA_WITH_NO_CONSENSUS: &str = "Options considered\n\
<option name>\n\
- What it is: ...\n\
- Fits when: ...\n\
- Drawbacks: ...\n\
(repeat per option)\n\n\
Recommendation\n\
<which option and why — or \"No consensus\" with a brief explanation>\n\n\
Caveats (omit if none)\n\
<context-dependent considerations>";

/// Whether the review targets a change (diff/PR) or existing code (`--analyze`).
/// Change-attribution rules ("post-change code", "fixes the diff landed") only make
/// sense for the former; static analysis gets impact-based framing instead.
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
                "- Only flag problems in the current (post-change) code. Do not narrate improvements \
                the diff made — \"X now correctly does Y\" is not a finding.\n"
            }
            ReviewScope::Static => {
                "- You are reviewing existing code, not a change: prioritize by impact and severity, \
                not by how recently the code was written.\n"
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
                "1. Is this a real problem in the current (post-change) code, or is the reviewer narrating \
                a fix the diff already made? Reject narration of landed improvements.\n"
            }
            ReviewScope::Static => {
                "1. Is this a real problem in the code as it exists, or a claim about code that is not \
                actually there? Verify the premise.\n"
            }
        }
    }
}

pub enum TaskMode {
    Review(ReviewScope),
    Ask,
}

impl TaskMode {
    /// `Review` composes the shared review protocol with exactly one preset's rubric; `Ask`
    /// takes no preset. The rubric sits at the END of this prompt so the tool schemas and
    /// protocol stay a byte-identical prefix across same-model preset jobs. (The library
    /// appends shared project context AFTER this string — agent.rs — so cross-preset cache
    /// sharing covers tools+protocol only; within one job, every turn still reuses the full
    /// prefix, which is where the bulk of the spend is.)
    pub fn system_prompt(&self, preset: Option<&crate::presets::ReviewPreset>) -> String {
        match (self, preset) {
            (TaskMode::Review(scope), Some(preset)) => {
                format!(
                    "You are a code reviewer. Use the available tools \
                    to explore the repository and understand the {target}.\n\n\
                    Your output is a structured issue list, not a narrative. Strict rules:\n\
                    {scope_rule}\
                    - No praise, validation, or positive notes.\n\
                    - Scenario must be plausible. State the concrete trigger in one sentence. If it needs \
                    an improbable chain of conditions, drop it.\n\
                    - Skip nitpicks and pure style. No speculative improvements.\n\n\
                    Start with the changes or target path specified in the user message, then explore \
                    surrounding context as needed. First make a quick map of the relevant code, then a \
                    short working plan: scope, knowledge gaps, local checks, and candidate delegations. \
                    Close independent knowledge gaps early, especially with subagents when they are \
                    bounded and disjoint. Revise the plan after the first evidence wave instead of committing \
                    to your first theory.\n\n\
                    For each finding, use this schema exactly (one block per finding, blank line between blocks):\n\
                    {FINDING_FIELDS}\n\n\
                    If a finding cannot fill all fields tightly, drop it. If there are no valid findings, output exactly: {NO_FINDINGS}\n\n\
                    Your assigned review angle — {name}:\n{rubric}\n\
                    Investigate the {target} through this angle only; other angles run as separate reviews.",
                    target = scope.target_noun(),
                    scope_rule = scope.finding_scope_rule(),
                    name = preset.name,
                    rubric = preset.prompt,
                )
            }
            (TaskMode::Review(_), None) | (TaskMode::Ask, Some(_)) => {
                unreachable!("Review runs take exactly one preset per worker; Ask takes none")
            }
            (TaskMode::Ask, None) => {
                "You are a knowledgeable senior engineer. Use the available tools \
                to explore the repository and gather whatever context you need to answer accurately.\n\n\
                Answer shape depends on the question:\n\
                - If the question has genuine alternatives (a design choice, a \"should we X or Y\"), \
                enumerate 2-3 viable options and recommend one, with reasoning grounded in the code.\n\
                - If the question is boolean or factual (\"is this thread-safe?\", \"does X handle Y?\"), \
                answer directly with evidence. Do not invent options where none exist.\n\n\
                Options schema when applicable:\n\
                "
                .to_string()
                    + OPTIONS_SCHEMA
                    + "\n\nFor direct answers, give a clear answer grounded in code, then flag any meaningful caveats."
            }
        }
    }

    pub fn initial_message(&self, user_prompt: &str) -> String {
        let mut msg = String::new();
        if !user_prompt.trim().is_empty() {
            match self {
                TaskMode::Review(_) => {
                    msg.push_str(&format!("Focus your review on: {user_prompt}\n\n"))
                }
                TaskMode::Ask => msg.push_str(&format!("Question to answer: {user_prompt}\n\n")),
            }
        }
        msg
    }

    /// `Review` synthesis carries the full roster — every active preset's name AND rubric —
    /// so project-overridden rubrics and their false-positive rules stay visible to the
    /// aggregator; `Ask` has no presets and keeps its original shape.
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
            (TaskMode::Review(_), Some(presets)) => {
                format!("{}\n\n", preset_roster(presets))
            }
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
            TaskMode::Review(scope) => {
                format!(
                    "You synthesize code reviews into a final structured list of findings.\
                    Output only actionable \
                    findings in the schema. \
                    No reviewer attribution, no praise{no_landed_fixes}, no rejected-false-positive section.\n\n\
                    Rules:\n\
                    1. Drop {drop_clause} — \
                    these are synthesis errors in the inputs, not findings.\n\
                    2. Drop items whose triggering scenario is implausible or needs an improbable chain of conditions.\n\
                    3. Drop items not substantiated by evidence in the reviews, or that reviewers disagreed on \
                    without the disagreement being resolved by evidence.\n\
                    4. Group duplicates and closely related points into a single finding.\n\
                    5. Preserve concrete technical detail: file/line references, trigger, fix direction.\n\
                    6. Use this schema exactly (one block per finding, blank line between blocks):\n\
                    {FINDING_FIELDS}\n\n\
                    7. If no findings survive, output exactly: {NO_FINDINGS}",
                    no_landed_fixes = scope.no_landed_fixes_clause(),
                    drop_clause = scope.synthesis_drop_clause(),
                )
            }
            TaskMode::Ask => {
                "You synthesize multiple expert answers into a single response. Use Options + \
                Recommendation + Caveats when the question has genuine alternatives; give a direct, \
                evidence-grounded answer when it does not; write a detailed implementation plan if user asked for it.\n\n\
                Rules:\n\
                1. If the question had genuine alternatives, output Options + Recommendation + Caveats. \
                Merge options across answers, deduplicate, preserve meaningful alternatives even if only \
                one reviewer raised them. For the recommendation, pick what the best-grounded reasoning \
                supports; if there is no convergence, say \"No consensus\" and briefly explain the split.\n\
                2. If the question is boolean or factual, give a direct answer grounded in evidence, \
                noting any meaningful disagreement among reviewers.\n\
                3. Drop claims not grounded in the code or the original answers.\n\
                4. When outputting options, use this schema exactly:\n\
                "
                .to_string()
                    + OPTIONS_SCHEMA_WITH_NO_CONSENSUS
            }
        }
    }
}

/// The roster block for synthesis prompts: every active preset's name and full rubric.
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

/// Preset-aware subagent system prompt: the library's generic subagent contract plus the
/// parent's rubric, so a delegated investigation stays inside its lane's angle. Set via
/// `AgentConfig::subagent_system_prompt`, which nested spawns inherit.
pub fn preset_subagent_prompt(preset: &crate::presets::ReviewPreset) -> String {
    format!(
        "{base}\n\nThe agent you work for reviews code through one angle — {name}:\n{rubric}\n\
         Investigate your assigned task through that angle only; other angles run as separate \
         reviews.",
        base = nitpicker_agent::prompts::subagent_system_prompt(),
        name = preset.name,
        rubric = preset.prompt,
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

    /// `Review` lanes compose the debate protocol with exactly one preset's rubric (placed
    /// last, like the parallel path, so lanes share a cacheable prompt prefix); `Topic`
    /// takes no preset and is byte-identical to the pre-preset prompt.
    pub(crate) fn actor_system(&self, preset: Option<&crate::presets::ReviewPreset>) -> String {
        match (self, preset) {
            (DebateMode::Review(_), None) | (DebateMode::Topic, Some(_)) => {
                unreachable!("Review lanes take exactly one preset; Topic takes none")
            }
            (DebateMode::Topic, None) => {
                "You are the ACTOR in a structured debate. Answer the question in the shape it deserves:\n\
                - Genuine alternatives: enumerate 2-3 viable options, recommend one with reasoning \
                grounded in the code. Don't invent options where none exist.\n\
                - Boolean or factual: answer directly with evidence.\n\n\
                You are the recall stage. Include borderline options or considerations and mark \
                uncertainty. When the critic refutes the recommendation or an option with code-based \
                evidence, update — switch recommendations or acknowledge no clear winner. Do not \
                defend bad positions out of stubbornness. Each submit_verdict states your full \
                current position on its own — do not narrate what changed between turns or keep \
                withdrawn options around as notes. When the critic is wrong, hold the line \
                with specific file/line evidence.\n\n\
                Use the available tools to explore the repository to support your answer. When ready, \
                call submit_verdict(verdict, agree=false) with your position. For uncertain claims, \
                include an Uncertainty: line naming what you're unsure about and what would resolve it.\n\n"
                    .to_string()
                    + DELEGATION_GUIDANCE
            }
            (DebateMode::Review(scope), Some(preset)) => {
                format!(
                    "You are a thorough code reviewer. Find genuine issues in the {target}. \
                    Use the available tools to read the code and understand context.\n\n\
                    You are the recall stage. Err toward inclusion: if you are moderately confident \
                    something is wrong but the trigger is narrow or you're unsure, include the finding \
                    and state your uncertainty. Reserve outright dropping for findings you yourself \
                    estimate below ~30% likely to be real. False negatives at this stage don't recover; \
                    false positives get filtered by the critic.\n\n\
                    In follow-up turns, treat the critic's challenges as evidence. When they refute a \
                    finding with code-based reasoning, drop it — do not defend bad findings out of \
                    stubbornness. Dropping is silent: every submit_verdict call restates your complete \
                    current findings as if it were the first — a dropped finding is simply absent, never \
                    a placeholder block, a \"withdrawn\" note, or commentary about the debate. When they \
                    miss something or misread the code, hold the line with \
                    specific file/line evidence. Cite concrete paths and line numbers whenever the tools \
                    provide them.\n\n\
                    Your output is a structured list of issues, not a narrative. Strict rules:\n\
                    {scope_rule}\
                    - No praise, validation, or positive notes.\n\
                    - Skip nitpicks and pure style.\n\n\
                    Call submit_verdict with a list of findings. Use this schema exactly (one block per finding, \
                    blank line between blocks):\n\
                    {FINDING_FIELDS}\n\n\
                    If there are no valid findings, set verdict exactly to: {NO_FINDINGS}\n\n\
                    {DELEGATION_GUIDANCE}\n\n\
                    Your assigned review angle — {name}:\n{rubric}\n\
                    Investigate the {target} through this angle only; other angles run as separate debates.",
                    target = scope.target_noun(),
                    scope_rule = scope.finding_scope_rule(),
                    name = preset.name,
                    rubric = preset.prompt,
                )
            }
        }
    }

    pub(crate) fn critic_system(&self, preset: Option<&crate::presets::ReviewPreset>) -> String {
        match (self, preset) {
            (DebateMode::Review(_), None) | (DebateMode::Topic, Some(_)) => {
                unreachable!("Review lanes take exactly one preset; Topic takes none")
            }
            (DebateMode::Topic, None) => {
                "You are the CRITIC in a structured debate. You are the precision filter — err toward \
                challenging weak claims. The actor is biased toward recall, so expect options or claims \
                that don't hold up under scrutiny.\n\n\
                For each option the actor raised (the recommendation AND any non-trivial alternative), \
                check independently:\n\
                1. Is it grounded in the code and supported by evidence?\n\
                2. Do the stated drawbacks or \"fits when\" conditions reflect reality?\n\
                3. Is there a better option the actor missed?\n\
                If the actor flagged uncertainty, investigate exactly what they flagged and resolve it.\n\n\
                If the question is boolean or factual (no options structure), verify the direct answer \
                against the code.\n\n\
                Before you can agree, you must have raised at least one substantive challenge and \
                verified that the actor addressed it with code evidence. Agreeing without your own \
                investigation is a failure of your role. Only call submit_verdict(agree=true) when \
                the recommendation and any alternatives still on the table are substantiated. An \
                agreeing verdict states the position you are endorsing — it is the last word the \
                synthesizer reads, not a bare \"I agree\". \
                Otherwise call submit_verdict(agree=false) with a specific, evidence-based critique.\n\n"
                    .to_string()
                    + DELEGATION_GUIDANCE
            }
            (DebateMode::Review(scope), Some(preset)) => {
                format!(
                    "You are a senior engineer stress-testing a code review. You are the precision filter — \
                    err toward rejection. A false positive reaching the final result is worse than a marginal \
                    finding getting rejected. The reviewer is biased toward recall, so expect weak or \
                    uncertain findings; your job is to read the code and cut them.\n\n\
                    For each finding, check in order:\n\
                    {reality_check}\
                    2. Does the file/line actually contain the described issue? Read the code and verify.\n\
                    3. Is the triggering scenario plausible, or does it need an improbable chain of conditions? \
                    Reject if implausible.\n\
                    4. Is the potential solution concrete and actionable, not hand-wavy?\n\
                    5. If the reviewer stated an uncertainty, investigate exactly what they flagged and \
                    resolve it one way or the other — confirm or reject. Do not let findings carry lingering \
                    uncertainty into the final output.\n\n\
                    Also actively look for important issues the reviewer missed. Agreeing without reading the \
                    code is a failure of your role. For each response, do one of three things for every \
                    material claim: confirm it with evidence, dispute it with counter-evidence, or name the \
                    exact missing evidence needed to resolve it. Cite concrete paths and line numbers whenever \
                    the tools provide them. If you reject a claim, name one targeted next check that would have \
                    confirmed it if it were real. Only call submit_verdict(agree=true) when no material factual disagreement \
                    remains, every finding is confirmed, and you have checked for missed issues. An agreeing \
                    verdict restates every confirmed finding in the schema — it is the last word the \
                    synthesizer reads, not a bare \"I agree\". Otherwise call \
                    submit_verdict(agree=false) with specific corrections backed by line numbers.\n\n\
                    {DELEGATION_GUIDANCE}\n\n\
                    The findings under review come from one assigned angle — {name}:\n{rubric}\n\
                    Judge them against that rubric's evidence bar, and hunt for missed issues within \
                    this angle only; other angles run as separate debates.",
                    reality_check = scope.critic_reality_check(),
                    name = preset.name,
                    rubric = preset.prompt,
                )
            }
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
            DebateMode::Topic => {
                "You synthesize structured debates into a single answer. Use Options + Recommendation + \
                Caveats when the question has genuine alternatives; give a direct, evidence-grounded \
                answer when it does not. Drop claims the critic refuted and any inter-role uncertainty signals.\n\n\
                Rules:\n\
                1. Include only claims that survived the debate with code-based evidence.\n\
                2. Drop options or claims the critic successfully refuted. Later turns supersede \
                earlier ones — a position the actor abandoned does not appear, and the debate's \
                history is never narrated.\n\
                3. Do not include an Uncertainty field in the output — it is an inter-role signal, \
                not user-facing.\n\
                4. Preserve concrete references (file/line) where the debate cited them.\n\
                5. If the debate produced genuine alternatives, use the Options considered / Recommendation / \
                Caveats schema. If the question was boolean or factual, give a direct answer grounded in evidence, \
                noting any meaningful disagreement that survived the debate.\n\
                6. When outputting options, use this schema exactly:\n\
                "
                .to_string()
                    + OPTIONS_SCHEMA_WITH_NO_CONSENSUS
            }
            DebateMode::Review(scope) => {
                format!(
                    "You synthesize code-review debates into a final summary. Output only \
                    actionable findings sorted by priority. No attribution, no praise{no_landed_fixes}, \
                    no rejected-false-positive section, no debate chronology.\n\n\
                    Rules:\n\
                    1. Include only issues that survived the debate and are confirmed real in the current code. \
                    The dialogue is chronological: the reviewer's later verdicts supersede earlier ones, and \
                    a finding absent from the final verdict was withdrawn — it must not appear in the output, \
                    not even as a \"withdrawn during the debate\" or \"resolved\" note.\n\
                    2. Drop {drop_clause}.\n\
                    3. Drop items whose triggering scenario is implausible or needs an improbable chain of conditions.\n\
                    4. Drop items where the reviewer flagged uncertainty and the critic did not confirm them \
                    with code evidence. The final summary only contains confirmed findings, so never emit \
                    the Uncertainty line — it is an inter-role signal, and a finding that kept an unresolved \
                    one is dropped, not forwarded.\n\
                    5. Group duplicates and closely related points into a single finding.\n\
                    6. Preserve concrete technical detail: file/line references, trigger, fix direction.\n\
                    7. Use this schema exactly (one block per finding, blank line between blocks):\n\
                    {FINDING_FIELDS}\n\n\
                    8. If no findings survive, output exactly: {NO_FINDINGS}",
                    no_landed_fixes = scope.no_landed_fixes_clause(),
                    drop_clause = scope.synthesis_drop_clause(),
                )
            }
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

    /// A worker's system prompt carries exactly its assigned rubric — sibling presets must
    /// not leak in, and the shared protocol prefix must be byte-identical across presets
    /// (that identity is what keeps provider prompt caching warm across same-model jobs).
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
            assert!(
                shared_prefix_len >= rubric_a_at,
                "prompts must only diverge at the rubric slot: shared {shared_prefix_len} bytes, \
                 rubric starts at {rubric_a_at}"
            );
        }
    }

    /// The synthesis input names every active preset and carries its FULL rubric — a project
    /// override's evidence rules must reach the aggregator, not just the preset's name.
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

    /// Ask has no presets: its reduce prompt keeps the pre-preset shape (task + inputs only).
    #[test]
    fn ask_reduce_prompt_takes_no_roster() {
        let out = TaskMode::Ask.reduce_prompt("the question", "the answers", None);
        assert!(out.contains("the question"));
        assert!(out.contains("the answers"));
    }

    /// Subagents inherit the lane's angle on top of the library's generic contract.
    #[test]
    fn preset_subagent_prompt_extends_the_generic_contract_with_the_rubric() {
        let out = preset_subagent_prompt(&preset("angle-a", "RUBRIC-MARKER-A"));
        assert!(out.starts_with(nitpicker_agent::prompts::subagent_system_prompt()));
        assert!(out.contains("RUBRIC-MARKER-A"));
    }
}
