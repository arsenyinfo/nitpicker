use crate::pr;
use clap::{Args as ClapArgs, Parser, Subcommand};
use std::path::PathBuf;

/// Flags shared across the default review mode and the subcommands. Declared once here and
/// marked `global`, so they are accepted before or after a subcommand and always land in
/// `Args.common` — per-subcommand copies would be independent namespaces, and a flag parsed
/// into the copy an arm doesn't read would be silently dropped.
#[derive(Debug, ClapArgs)]
pub(crate) struct CommonArgs {
    #[arg(long, global = true, default_value = ".")]
    pub(crate) repo: PathBuf,

    #[arg(long, global = true)]
    pub(crate) config: Option<PathBuf>,

    #[arg(long, short, global = true)]
    pub(crate) verbose: bool,

    /// Try the next configured reviewer when the selected model fails
    #[arg(long, global = true)]
    pub(crate) fallback: bool,
}

/// `--context-file`, kept out of the global `CommonArgs` deliberately: clap propagates a global
/// arg by keeping one winning occurrence list (the subcommand's), so a repeatable flag split
/// around the subcommand would silently drop the root's values. Instead this struct is flattened
/// at the root and into `ask`/`pr`, and the two vectors are concatenated root-first (= the
/// command-line order) at each use site.
#[derive(Debug, ClapArgs)]
pub(crate) struct ContextFileArgs {
    /// Read a file into the prompt verbatim; repeatable. Unlike the agents' own tools, this is not
    /// confined to the repo, so it can carry design notes or working docs that live outside it.
    #[arg(long = "context-file", value_name = "PATH")]
    pub(crate) context_file: Vec<PathBuf>,
}

pub(crate) fn merged_context_files(root: &ContextFileArgs, sub: &ContextFileArgs) -> Vec<PathBuf> {
    root.context_file
        .iter()
        .chain(&sub.context_file)
        .cloned()
        .collect()
}

/// `--preset`, shaped exactly like `ContextFileArgs` and for the same reason: a repeatable
/// flag must not be `global` (clap would keep only the subcommand's occurrence list), so it
/// is flattened at the root and into `pr`, and merged root-first at each use site.
#[derive(Debug, ClapArgs)]
pub(crate) struct PresetArgs {
    /// Review preset(s) to run — repeatable and comma-separated (e.g. --preset security,ml-rigor).
    /// Replaces the configured `[defaults].presets` list for this run.
    #[arg(long = "preset", value_name = "NAME", value_delimiter = ',')]
    pub(crate) preset: Vec<String>,
}

pub(crate) fn merged_presets(root: &PresetArgs, sub: &PresetArgs) -> Vec<String> {
    root.preset.iter().chain(&sub.preset).cloned().collect()
}

#[derive(Debug, Parser)]
#[command(name = "nitpicker")]
pub(crate) struct Args {
    #[command(subcommand)]
    pub(crate) command: Option<Command>,

    #[command(flatten)]
    pub(crate) common: CommonArgs,

    #[command(flatten)]
    pub(crate) context: ContextFileArgs,

    #[command(flatten)]
    pub(crate) presets: PresetArgs,

    #[arg(
        long,
        help = "Additional review instructions appended to the diff context (use `ask` for fully custom prompts)"
    )]
    pub(crate) prompt: Option<String>,

    /// Analyze existing code instead of reviewing changes
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    pub(crate) analyze: Option<PathBuf>,

    /// Disable actor-critic debate and use parallel aggregation instead
    #[arg(long)]
    pub(crate) no_debate: bool,

    /// Mix all reviewer models into a shared pool; each LLM call picks one at random
    #[arg(long)]
    pub(crate) alloy: bool,

    /// Maximum debate rounds
    #[arg(long, default_value = "5")]
    pub(crate) rounds: usize,

    /// Maximum tool-use turns per agent or debate turn
    #[arg(long, value_parser = parse_positive_usize)]
    pub(crate) max_turns: Option<usize>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Generate a nitpicker config template
    Init {
        /// Write to ~/.nitpicker/config.toml instead of <repo>/nitpicker.toml
        #[arg(long)]
        global: bool,

        /// Prefer OpenRouter experimental free models in the generated config
        #[arg(long)]
        free: bool,
    },
    /// Ask multiple LLM agents a free-form question about the codebase
    Ask {
        #[command(flatten)]
        context: ContextFileArgs,
        /// Question or topic to discuss
        topic: String,
        /// Disable actor-critic debate and use parallel aggregation instead
        #[arg(long)]
        no_debate: bool,
        /// Mix all reviewer models into a shared pool; each LLM call picks one at random
        #[arg(long)]
        alloy: bool,
        /// Maximum debate rounds
        #[arg(long, default_value = "5")]
        rounds: usize,
        /// Maximum tool-use turns per agent or debate turn
        #[arg(long, value_parser = parse_positive_usize)]
        max_turns: Option<usize>,
    },
    /// Review a GitHub PR (current branch's PR or a remote PR by URL)
    Pr(pr::PrArgs),
    /// Reflect on past nitpicker sessions to identify patterns and friction points
    Reflect {
        /// Directory containing sessions (default: ~/.nitpicker/sessions)
        #[arg(long)]
        sessions_dir: Option<PathBuf>,
        /// Number of most recent sessions to analyze
        #[arg(long, default_value = "20")]
        n: usize,
    },
}

pub(crate) fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid positive integer: {value}"))?;

    if parsed == 0 {
        return Err("value must be greater than 0".to_string());
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fallback_allowed, presets_allowed};
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Args::command().debug_assert();
    }

    fn parse(argv: &[&str]) -> Args {
        Args::try_parse_from(argv).expect("argv parses")
    }

    fn ask_context(args: &Args) -> Vec<PathBuf> {
        match &args.command {
            Some(Command::Ask { context, .. }) => merged_context_files(&args.context, context),
            _ => panic!("expected ask subcommand"),
        }
    }

    fn pr_context(args: &Args) -> Vec<PathBuf> {
        match &args.command {
            Some(Command::Pr(pr_args)) => merged_context_files(&args.context, &pr_args.context),
            _ => panic!("expected pr subcommand"),
        }
    }

    #[test]
    fn context_file_before_the_subcommand_reaches_ask() {
        let args = parse(&["nitpicker", "--context-file", "/a", "ask", "topic"]);
        assert_eq!(ask_context(&args), [PathBuf::from("/a")]);
    }

    #[test]
    fn context_file_after_the_subcommand_reaches_ask() {
        let args = parse(&["nitpicker", "ask", "--context-file", "/a", "topic"]);
        assert_eq!(ask_context(&args), [PathBuf::from("/a")]);
    }

    #[test]
    fn context_files_split_around_the_subcommand_merge_in_cli_order() {
        let args = parse(&[
            "nitpicker",
            "--context-file",
            "/a",
            "ask",
            "--context-file",
            "/b",
            "topic",
        ]);
        assert_eq!(
            ask_context(&args),
            [PathBuf::from("/a"), PathBuf::from("/b")]
        );

        let args = parse(&[
            "nitpicker",
            "--context-file",
            "/a",
            "pr",
            "--context-file",
            "/b",
        ]);
        assert_eq!(
            pr_context(&args),
            [PathBuf::from("/a"), PathBuf::from("/b")]
        );
    }

    #[test]
    fn global_scalars_land_in_common_from_either_side_of_the_subcommand() {
        let args = parse(&["nitpicker", "-v", "--repo", "/x", "ask", "topic"]);
        assert!(args.common.verbose);
        assert_eq!(args.common.repo, PathBuf::from("/x"));

        let args = parse(&["nitpicker", "ask", "topic", "--repo", "/x", "-v"]);
        assert!(args.common.verbose);
        assert_eq!(args.common.repo, PathBuf::from("/x"));

        let args = parse(&["nitpicker", "pr", "--repo", "/x", "--config", "/c.toml"]);
        assert!(!args.common.verbose);
        assert_eq!(args.common.repo, PathBuf::from("/x"));
        assert_eq!(args.common.config, Some(PathBuf::from("/c.toml")));

        let args = parse(&["nitpicker", "--fallback", "ask", "topic"]);
        assert!(args.common.fallback);
        let args = parse(&["nitpicker", "pr", "--fallback"]);
        assert!(args.common.fallback);
    }

    #[test]
    fn subcommands_without_context_files_reject_the_flag() {
        for argv in [
            ["nitpicker", "reflect", "--context-file", "/a"],
            ["nitpicker", "init", "--context-file", "/a"],
        ] {
            assert!(Args::try_parse_from(argv).is_err());
        }
    }

    fn pr_presets(args: &Args) -> Vec<String> {
        match &args.command {
            Some(Command::Pr(pr_args)) => merged_presets(&args.presets, &pr_args.presets),
            _ => panic!("expected pr subcommand"),
        }
    }

    #[test]
    fn preset_reaches_pr_from_either_side_of_the_subcommand() {
        let args = parse(&["nitpicker", "--preset", "security", "pr"]);
        assert_eq!(pr_presets(&args), ["security"]);

        let args = parse(&["nitpicker", "pr", "--preset", "security"]);
        assert_eq!(pr_presets(&args), ["security"]);
    }

    /// Repeated flags append, commas split within one occurrence, and values split around
    /// the subcommand merge root-first (= command-line order) — same contract as
    /// `--context-file`, and the reason `--preset` is not a clap `global`.
    #[test]
    fn presets_split_around_the_subcommand_merge_in_cli_order_with_commas_expanded() {
        let args = parse(&[
            "nitpicker",
            "--preset",
            "security,ml-rigor",
            "pr",
            "--preset",
            "tone",
        ]);
        assert_eq!(pr_presets(&args), ["security", "ml-rigor", "tone"]);
    }

    #[test]
    fn repeated_preset_flags_append_on_the_root_review_path() {
        let args = parse(&["nitpicker", "--preset", "security", "--preset", "tone"]);
        assert!(args.command.is_none());
        assert_eq!(args.presets.preset, ["security", "tone"]);
    }

    #[test]
    fn subcommands_without_presets_reject_the_flag() {
        let cases: [&[&str]; 3] = [
            &["nitpicker", "ask", "topic", "--preset", "security"],
            &["nitpicker", "reflect", "--preset", "security"],
            &["nitpicker", "init", "--preset", "security"],
        ];
        for argv in cases {
            assert!(Args::try_parse_from(argv).is_err(), "argv: {argv:?}");
        }
    }

    /// Root-position `--preset` parses before any subcommand, so the non-review arms must
    /// reject it explicitly instead of silently discarding it.
    #[test]
    fn root_position_presets_are_rejected_for_non_review_subcommands() {
        let cases: [&[&str]; 3] = [
            &["nitpicker", "--preset", "security", "ask", "topic"],
            &["nitpicker", "--preset", "security", "init"],
            &["nitpicker", "--preset", "security", "reflect"],
        ];
        for argv in cases {
            let args = parse(argv);
            assert!(!presets_allowed(&args.command), "argv: {argv:?}");
        }

        let args = parse(&["nitpicker", "--preset", "security", "pr"]);
        assert!(presets_allowed(&args.command));
        let args = parse(&["nitpicker", "--preset", "security"]);
        assert!(presets_allowed(&args.command));
    }

    #[test]
    fn fallback_is_scoped_to_review_and_ask_commands() {
        for argv in [
            &["nitpicker", "--fallback"][..],
            &["nitpicker", "--fallback", "ask", "topic"][..],
            &["nitpicker", "pr", "--fallback"][..],
        ] {
            let args = parse(argv);
            assert!(fallback_allowed(&args.command), "argv: {argv:?}");
        }
        for argv in [
            &["nitpicker", "init", "--fallback"][..],
            &["nitpicker", "reflect", "--fallback"][..],
        ] {
            let args = parse(argv);
            assert!(!fallback_allowed(&args.command), "argv: {argv:?}");
        }
    }

    #[test]
    fn execution_flags_keep_root_and_subcommand_namespaces() {
        for subcommand in [vec!["ask", "topic"], vec!["pr"]] {
            let mut argv = vec![
                "nitpicker",
                "--no-debate",
                "--alloy",
                "--rounds",
                "9",
                "--max-turns",
                "7",
            ];
            argv.extend(subcommand.clone());
            let args = parse(&argv);
            assert!(args.no_debate && args.alloy);
            assert_eq!((args.rounds, args.max_turns), (9, Some(7)));
            let (no_debate, alloy, rounds, turns) = match args.command.unwrap() {
                Command::Ask {
                    no_debate,
                    alloy,
                    rounds,
                    max_turns,
                    ..
                } => (no_debate, alloy, rounds, max_turns),
                Command::Pr(pr) => (pr.no_debate, pr.alloy, pr.rounds, pr.max_turns),
                _ => unreachable!(),
            };
            assert_eq!((no_debate, alloy, rounds, turns), (false, false, 5, None));

            let mut argv = vec!["nitpicker"];
            argv.extend(subcommand);
            argv.extend([
                "--no-debate",
                "--alloy",
                "--rounds",
                "3",
                "--max-turns",
                "2",
            ]);
            let args = parse(&argv);
            assert_eq!(
                (args.no_debate, args.alloy, args.rounds, args.max_turns),
                (false, false, 5, None)
            );
            let values = match args.command.unwrap() {
                Command::Ask {
                    no_debate,
                    alloy,
                    rounds,
                    max_turns,
                    ..
                } => (no_debate, alloy, rounds, max_turns),
                Command::Pr(pr) => (pr.no_debate, pr.alloy, pr.rounds, pr.max_turns),
                _ => unreachable!(),
            };
            assert_eq!(values, (true, true, 3, Some(2)));
        }
        for prefix in [
            vec!["nitpicker"],
            vec!["nitpicker", "ask", "topic"],
            vec!["nitpicker", "pr"],
        ] {
            for value in ["0", "-1", "bad"] {
                let mut argv = prefix.clone();
                argv.extend(["--max-turns", value]);
                assert!(Args::try_parse_from(argv).is_err());
            }
        }
    }
}
