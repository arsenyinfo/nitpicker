pub enum TaskMode {
    Review,
    Ask,
}

impl TaskMode {
    fn preamble(&self) -> &'static str {
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

    pub fn initial_message(&self) -> &'static str {
        match self {
            TaskMode::Review => "Begin your review. Start with the changes or target path specified in your instructions, then explore surrounding context as needed.",
            TaskMode::Ask => "Begin your investigation. Explore the codebase as needed to give an accurate, well-grounded answer to the question in your instructions.",
        }
    }

    pub fn system_prompt(&self, context: &str, user_prompt: &str) -> String {
        let user_instructions = if user_prompt.trim().is_empty() {
            String::new()
        } else {
            match self {
                TaskMode::Review => format!("\n\nFocus your review on: {user_prompt}"),
                TaskMode::Ask => format!("\n\nQuestion to answer: {user_prompt}"),
            }
        };
        format!("{}{context}{user_instructions}", self.preamble())
    }

    pub fn reduce_prompt(&self, combined: &str) -> String {
        match self {
            TaskMode::Review => format!(
                "Your job is to synthesize multiple code reviews into a single,\n\
                actionable summary. Deduplicate findings, resolve conflicts, and prioritize by severity. Include source of every item (refer reviewers).\n\n\
                Format your response as:\n\
                1. **Critical** - must fix before merge\n\
                2. **Important** - should fix, but not blocking\n\
                3. **Suggestions** - nice to have improvements\n\n\
                If there are no findings in a category, omit it.\n\n\
                Start your response with a one-sentence overall verdict on whether the code is ready to merge or not (must start with APPROVE or REJECT). Markdown is not supported.\n\n\
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
            TaskMode::Review => "You synthesize code reviews into a concise final verdict.",
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

    pub(crate) fn actor_system(&self) -> &'static str {
        match self {
            DebateMode::Topic => {
                "You are the ACTOR in a structured debate. Propose and defend the best solution. \
                Use the available tools to explore the repository to support your arguments. \
                When ready, call submit_verdict(verdict, agree=false) with your final position."
            }
            DebateMode::Review => {
                "You are a thorough code reviewer. Find genuine issues — bugs, security flaws, \
                performance problems, unclear logic — in the changes described. Use the available \
                tools to read the code and understand context. Call submit_verdict with a clear, \
                evidence-based list of findings."
            }
        }
    }

    pub(crate) fn critic_system(&self) -> &'static str {
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
                with a specific, evidence-based critique."
            }
            DebateMode::Review => {
                "You are a senior engineer stress-testing a code review. Treat every finding as a \
                suspected false positive until you verify it yourself in the actual code. For each \
                finding: read the relevant file, check the surrounding context, and decide independently \
                whether the issue is real. Also actively look for important issues the reviewer missed. \
                Agreeing without reading the code is a failure of your role. \
                Only call submit_verdict(agree=true) when every finding is verified correct AND you \
                have checked for missed issues. Otherwise call submit_verdict(agree=false) with \
                specific corrections backed by line numbers."
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
                "Synthesize the review findings. Which issues are confirmed real? Which were false positives? \
                What is the most important thing to fix? Be concise and actionable."
            }
        }
    }

    pub(crate) fn meta_preamble(&self) -> &'static str {
        match self {
            DebateMode::Topic | DebateMode::Review => {
                "You synthesize structured debates into concise, actionable conclusions."
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
