const VERIFY_WARNING: &str =
    "Your opponent may sound confident but still make factual errors or overlook edge cases. \
Independently verify every claim against the actual code before accepting it.";

const DELEGATION_GUIDANCE: &str = "You can delegate focused investigations with spawn_subagent(task). \
Each task should be a single concrete question or lookup — one symbol, one file, one concept. \
When you have multiple independent questions, spawn them all at once in a single turn rather than sequentially. \
Keep the parent agent focused on synthesis and final judgment; use subagents for bounded digging. \
Prefer many narrow parallel subagents over one broad multi-part task.";

pub enum TaskMode {
    Review,
    Ask,
}

impl TaskMode {
    pub fn system_prompt(&self) -> &'static str {
        match self {
            TaskMode::Review => {
                "You are a code reviewer. Use the available tools (git, read_file, glob, grep) \
                to explore the repository and understand the changes.\n\n\
                Review criteria:\n\
                - Correctness: logic bugs, edge cases, off-by-one errors\n\
                - Security: injection, auth issues, secrets in code, unsafe deserialization \
                (only flag a security issue if you can trace a concrete exploit path, not just recognize a pattern)\n\
                - Performance: unnecessary allocations, N+1 queries, blocking calls in async context\n\
                - ML rigor: data leakage, incorrect loss/metrics, numerical instability, non-reproducibility\n\
                - Maintainability: dead code, copy-paste, unused variables, missing error handling\n\n\
                Style: fail loudly, not silently. No swallowed exceptions, no magic fallbacks, \
                no unexplained constants. Anything that can go wrong at runtime must be explicitly \
                checked and logged.\n\n\
                Be concise. Skip nitpicks and purely stylistic issues. Do not suggest speculative improvements.\n\
                Do not modify the repository. If you need scratch space, use /tmp."
            }
            TaskMode::Ask => {
                "You are a knowledgeable senior engineer. Use the available tools (git, read_file, glob, grep) \
                to explore the repository and gather whatever context you need to give an accurate, \
                well-grounded answer.\n\n\
                Be concise and direct. Ground your answer in the actual code when relevant.\n\
                Do not modify the repository. If you need scratch space, use /tmp."
            }
        }
    }

    pub fn initial_message(&self, context: &str, user_prompt: &str) -> String {
        let mut msg = String::new();
        if !context.is_empty() {
            msg.push_str(context);
            msg.push_str("\n\n");
        }
        if !user_prompt.trim().is_empty() {
            match self {
                TaskMode::Review => {
                    msg.push_str(&format!("Focus your review on: {user_prompt}\n\n"))
                }
                TaskMode::Ask => msg.push_str(&format!("Question to answer: {user_prompt}\n\n")),
            }
        }
        match self {
            TaskMode::Review => msg.push_str("Begin your review. Start with the changes or target path specified in your instructions, then explore surrounding context as needed."),
            TaskMode::Ask => msg.push_str("Begin your investigation. Explore the codebase as needed to give an accurate, well-grounded answer to the question in your instructions."),
        }
        msg
    }

    pub fn reduce_prompt(&self, combined: &str) -> String {
        match self {
            TaskMode::Review => format!(
                "Your job is to synthesize multiple code reviews into a single self-contained final report.\n\
                Focus on fidelity, not compression. Deduplicate findings, resolve conflicts, preserve the \
                important technical detail behind each confirmed issue, and include which reviewers raised \
                or supported it.\n\n\
                Rules:\n\
                1. Include only confirmed findings. If reviewers disagree, resolve the disagreement based on \
                the evidence in the reviews and exclude findings that were not substantiated.\n\
                2. For each finding, preserve the key context: what is wrong, why it matters, any important \
                conditions or edge cases, relevant file/line references, and which reviewers support it.\n\
                3. Do not include discarded or weakly supported issues as findings. If useful for \
                transparency, mention them only in a short final \"Rejected false positives\" section.\n\
                4. Group duplicates and closely related points into a single finding.\n\
                5. Preserve severity: Critical, Important, Suggestions.\n\
                6. Start with a one-sentence overall verdict that begins with APPROVE or REJECT.\n\n\
                Format your response as plain text with these headings:\n\
                Overall verdict\n\
                Critical issues\n\
                Important issues\n\
                Suggestions\n\
                Rejected false positives (optional, brief)\n\n\
                If a category has no items, omit it. The report should be detailed enough that a reader can \
                understand and act on the findings without reading the original individual reviews.\n\n\
                Individual reviews:\n\n\
                {combined}"
            ),
            TaskMode::Ask => format!(
                "Multiple engineers have answered the same question. Synthesize their responses into a \
                single clear, accurate, and actionable answer. Highlight points of agreement and \
                note any meaningful disagreements. Be concise.\n\n\
                Individual answers:\n\n\
                {combined}"
            ),
        }
    }

    pub fn aggregator_preamble(&self) -> &'static str {
        match self {
            TaskMode::Review => {
                "You synthesize code reviews into detailed, self-contained final reports. \
                Preserve concrete technical detail, supporting evidence, reviewer attribution, and severity. \
                Exclude refuted or weakly supported findings from the main findings sections."
            }
            TaskMode::Ask => "You synthesize multiple expert answers into a single clear response.",
        }
    }
}

pub enum DebateMode {
    Topic,
    Review,
}

impl DebateMode {
    pub fn actor_role(&self) -> &'static str {
        match self {
            DebateMode::Topic => "Actor",
            DebateMode::Review => "Reviewer",
        }
    }

    pub fn critic_role(&self) -> &'static str {
        match self {
            DebateMode::Topic => "Critic",
            DebateMode::Review => "Validator",
        }
    }

    pub(crate) fn actor_system(&self) -> String {
        match self {
            DebateMode::Topic => {
                "You are the ACTOR in a structured debate. Propose and defend the best solution. \
                Use the available tools to explore the repository to support your arguments. \
                When ready, call submit_verdict(verdict, agree=false) with your final position.\n\n"
                    .to_string()
                    + DELEGATION_GUIDANCE
                    + "\n\n"
                    + VERIFY_WARNING
            }
            DebateMode::Review => {
                "You are a thorough code reviewer. Find genuine issues — bugs, security flaws, \
                performance problems, unclear logic — in the changes described. Use the available \
                tools to read the code and understand context. Call submit_verdict with a clear, \
                evidence-based list of findings. For each finding, include severity, file/line references, \
                the concrete failure mode or risk, why it matters, and any important triggering conditions \
                or constraints.\n\n"
                    .to_string()
                    + DELEGATION_GUIDANCE
                    + "\n\n"
                    + VERIFY_WARNING
            }
        }
    }

    pub(crate) fn critic_system(&self) -> String {
        match self {
            DebateMode::Topic => {
                "You are the CRITIC in a structured debate. Your job is to stress-test the actor's \
                position — find weaknesses, check claims against the actual code, and demand evidence. \
                Be genuinely skeptical: assume the actor is wrong or incomplete until you have verified \
                their claims yourself using the available tools. \
                Before you can agree, you must have raised at least one substantive challenge and \
                verified that the actor addressed it with evidence from the codebase. \
                Agreeing immediately or without doing your own investigation is a failure of your role. \
                Only call submit_verdict(agree=true) when you have exhausted your challenges and the \
                actor's position is demonstrably correct. Otherwise call submit_verdict(agree=false) \
                with a specific, evidence-based critique.\n\n"
                    .to_string()
                    + DELEGATION_GUIDANCE
                    + "\n\n"
                    + VERIFY_WARNING
            }
            DebateMode::Review => {
                "You are a senior engineer stress-testing a code review. Treat every finding as a \
                suspected false positive until you verify it yourself in the actual code. For each \
                finding: read the relevant file, check the surrounding context, and decide independently \
                whether the issue is real. Also actively look for important issues the reviewer missed. \
                Agreeing without reading the code is a failure of your role. \
                Classify each reviewed issue as confirmed, rejected false positive, or still uncertain, \
                and explain that classification with evidence. \
                Only call submit_verdict(agree=true) when every finding is verified correct AND you \
                have checked for missed issues. Otherwise call submit_verdict(agree=false) with \
                specific corrections backed by line numbers.\n\n"
                    .to_string()
                    + DELEGATION_GUIDANCE
                    + "\n\n"
                    + VERIFY_WARNING
            }
        }
    }

    pub(crate) fn meta_instruction(&self) -> &'static str {
        match self {
            DebateMode::Topic => {
                "Synthesize the key insights. What was agreed on? What trade-offs remain? Be concise. \
                What is the best actionable conclusion?"
            }
            DebateMode::Review => {
                "You are synthesizing a code-review debate into the final report. \
                Your goal is to produce a self-contained review that preserves the technical detail \
                needed by someone who has not read the debate.\n\n\
                Focus on fidelity, not compression. Preserve the important specifics: what the issue is, \
                why it matters, the conditions under which it occurs, and the exact file and line references \
                cited in the debate. Omit conversational filler and repeated arguments.\n\n\
                Rules:\n\
                1. Include only issues that survived the debate and are confirmed real.\n\
                2. Do not present discarded or refuted issues as findings.\n\
                3. If you mention false positives for transparency, put them only in a short final \
                \"Rejected false positives\" section.\n\
                4. For each confirmed finding, preserve the key reasoning: what is wrong, why it matters, \
                and any important constraints, edge cases, or exploit/runtime conditions established in the debate.\n\
                5. Keep the debate's severity distinctions: Critical, Important, Suggestions.\n\
                6. Group duplicates or closely related points into a single finding.\n\
                7. Start with a one-sentence overall verdict beginning with APPROVE or REJECT.\n\n\
                Format your response as:\n\
                Overall verdict\n\
                Critical issues\n\
                Important issues\n\
                Suggestions\n\
                Rejected false positives (optional, brief)\n\n\
                Each finding should be detailed enough that a reader can understand the problem and act on it \
                without reading the original debate."
            }
        }
    }

    pub(crate) fn meta_preamble(&self) -> &'static str {
        match self {
            DebateMode::Topic => {
                "You synthesize structured debates into concise, actionable conclusions."
            }
            DebateMode::Review => {
                "You synthesize code-review debates into detailed, self-contained final reports. \
                Preserve the concrete technical details behind each confirmed finding, including file paths, \
                line numbers, impact, and the reasoning established during the debate. \
                Exclude refuted findings from the main findings sections."
            }
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            DebateMode::Topic => "debate",
            DebateMode::Review => "review-debate",
        }
    }
}

pub fn subagent_system_prompt() -> &'static str {
    "You are a focused subagent working for another agent. Solve only the assigned task. \
    Use the available tools to inspect the repository as needed. \
    When you need multiple independent pieces of information, call all relevant tools simultaneously \
    in a single turn rather than sequentially — this is faster and avoids wasting context. \
    Keep your final result concise, evidence-based, and grounded in the code. \
    Include file paths and line numbers when relevant. \
    Do not ask follow-up questions. When you are done, call finish with your final result."
}
