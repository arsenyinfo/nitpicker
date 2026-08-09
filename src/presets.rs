//! Preset registry and resolution. A preset is one named review angle — a rubric prompt
//! injected into a single worker's prompts. Presets decide *what* to investigate; the
//! execution mode (parallel/debate/alloy) decides *how*. Resolution must run before any
//! LLM call (including OpenRouter free-model probes), so a bad name fails the run cold.

use eyre::Result;
use nitpicker_agent::config::Config;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewPreset {
    pub name: String,
    pub prompt: String,
}

const BUILT_IN_PRESETS: &[(&str, &str)] = &[
    (
        "correctness",
        include_str!("../prompts/presets/correctness.md"),
    ),
    ("security", include_str!("../prompts/presets/security.md")),
    (
        "performance",
        include_str!("../prompts/presets/performance.md"),
    ),
    (
        "simplicity",
        include_str!("../prompts/presets/simplicity.md"),
    ),
    ("ml-rigor", include_str!("../prompts/presets/ml-rigor.md")),
    (
        "ai-systems",
        include_str!("../prompts/presets/ai-systems.md"),
    ),
    ("tone", include_str!("../prompts/presets/tone.md")),
    ("general", include_str!("../prompts/presets/general.md")),
];

/// Upper bound on selected presets per run. Every selection runs as its own review job or
/// debate lane, so an unbounded list turns a config mistake into unbounded spend — the
/// LLM semaphore bounds provider *rate*, not run size.
const MAX_SELECTED_PRESETS: usize = 16;

/// Universal default coverage. Domain-specific rubrics (`ml-rigor`, `ai-systems`, `tone`)
/// stay opt-in, while `general` is the standalone unscoped alternative.
const DEFAULT_PRESET_NAMES: &[&str] = &["correctness", "security", "performance", "simplicity"];

/// Resolve the run's ordered preset list: CLI `--preset` values replace `[defaults].presets`,
/// which replaces the built-in default. Names are trimmed, deduplicated first-seen, and
/// case-sensitive; a `[presets.<name>]` table overrides the same-named built-in.
pub fn resolve(cli_names: &[String], config: &Config) -> Result<Vec<ReviewPreset>> {
    let configured = config.defaults.as_ref().and_then(|d| d.presets.as_deref());
    let selection: Vec<&str> = match (cli_names.is_empty(), configured) {
        (false, _) => normalized(cli_names, "--preset")?,
        (true, Some(names)) => {
            if names.is_empty() {
                eyre::bail!("[defaults].presets must not be empty");
            }
            normalized(names, "[defaults].presets")?
        }
        (true, None) => DEFAULT_PRESET_NAMES.to_vec(),
    };

    let mut seen = std::collections::HashSet::new();
    let selected: Vec<&str> = selection
        .into_iter()
        .filter(|name| seen.insert(*name))
        .collect();
    if selected.len() > MAX_SELECTED_PRESETS {
        eyre::bail!(
            "{} presets selected; the maximum is {MAX_SELECTED_PRESETS} — every selected \
             preset runs as its own review job or debate lane",
            selected.len()
        );
    }
    let resolved = selected
        .into_iter()
        .map(|name| lookup(name, config))
        .collect::<Result<Vec<_>>>()?;
    if resolved.len() > 1 && resolved.iter().any(|preset| preset.name == "general") {
        eyre::bail!(
            "preset \"general\" is a standalone broad review and cannot be combined with other presets"
        );
    }
    Ok(resolved)
}

fn normalized<'a>(names: &'a [String], source: &str) -> Result<Vec<&'a str>> {
    names
        .iter()
        .map(|raw| {
            let name = raw.trim();
            if name.is_empty() {
                eyre::bail!("{source} contains an empty preset name");
            }
            Ok(name)
        })
        .collect()
}

fn lookup(name: &str, config: &Config) -> Result<ReviewPreset> {
    let project = config.presets.as_ref().and_then(|m| m.get(name));
    let prompt = match project {
        Some(preset) => preset.prompt.clone(),
        None => match BUILT_IN_PRESETS.iter().find(|(n, _)| *n == name) {
            Some((_, prompt)) => prompt.trim().to_string(),
            None => eyre::bail!(
                "unknown preset {name:?} — available presets: {}",
                available_names(config).join(", ")
            ),
        },
    };
    Ok(ReviewPreset {
        name: name.to_string(),
        prompt,
    })
}

/// Sanitized, length-bounded path component for a preset name. Session stems embed it, and
/// an unbounded name would push the trajectory filename past the filesystem limit — that
/// write failure is only warned, silently losing the trace.
pub fn path_slug(name: &str) -> String {
    let mut slug = nitpicker_agent::session::sanitize_path_component(name);
    // sanitize output is ASCII, so byte truncation cannot split a char
    slug.truncate(40);
    slug
}

/// Wrap a failed global-synthesis call with the run's shape, adding the fewer-presets
/// remediation only when the error actually is a context-window overflow — on an auth or
/// transport failure that hint would be misdirection.
pub fn synthesis_failure(err: eyre::Report, description: String) -> eyre::Report {
    let hint = match nitpicker_agent::llm::is_context_length_error(&err) {
        true => {
            " — the assembled synthesis input likely exceeds the aggregator's context window; \
             select fewer presets (--preset) or configure an aggregator model with a larger \
             context window"
        }
        false => "",
    };
    err.wrap_err(format!("{description}{hint}"))
}

fn available_names(config: &Config) -> Vec<String> {
    let mut names: Vec<String> = BUILT_IN_PRESETS
        .iter()
        .map(|(n, _)| n.to_string())
        .collect();
    let project = config.presets.iter().flat_map(|m| m.keys());
    for name in project {
        if !names.iter().any(|n| n == name) {
            names.push(name.clone());
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use nitpicker_agent::config::{
        AggregatorConfig, DefaultsConfig, PresetConfig, ProviderType, ReviewerConfig,
    };
    use std::collections::BTreeMap;

    fn config(defaults_presets: Option<Vec<&str>>, project: &[(&str, &str)]) -> Config {
        let presets = match project.is_empty() {
            true => None,
            false => Some(
                project
                    .iter()
                    .map(|(name, prompt)| {
                        (
                            name.to_string(),
                            PresetConfig {
                                prompt: prompt.to_string(),
                            },
                        )
                    })
                    .collect::<BTreeMap<_, _>>(),
            ),
        };
        Config {
            defaults: defaults_presets.map(|names| DefaultsConfig {
                debate: None,
                alloy: None,
                max_turns: None,
                compact_threshold: None,
                log_trajectories: None,
                presets: Some(names.iter().map(|n| n.to_string()).collect()),
            }),
            aggregator: AggregatorConfig {
                model: "m".to_string(),
                provider: ProviderType::OpenAi,
                base_url: None,
                api_key_env: None,
                max_tokens: None,
                auth: Some("codex".to_string()),
                azure_scope: None,
                azure_credentials: None,
            },
            reviewer: vec![ReviewerConfig {
                name: "r".to_string(),
                model: "m".to_string(),
                provider: ProviderType::OpenAi,
                base_url: None,
                api_key_env: None,
                max_tokens: None,
                compact_threshold: None,
                auth: Some("codex".to_string()),
                azure_scope: None,
                azure_credentials: None,
            }],
            presets,
        }
    }

    fn names(presets: &[ReviewPreset]) -> Vec<&str> {
        presets.iter().map(|p| p.name.as_str()).collect()
    }

    fn cli(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    /// With no CLI and no config selection, the run resolves to the four universal angles.
    #[test]
    fn absent_selection_resolves_to_builtin_default() {
        let resolved = resolve(&[], &config(None, &[])).expect("resolves");
        assert_eq!(
            names(&resolved),
            ["correctness", "security", "performance", "simplicity"]
        );
    }

    /// CLI presets replace — not extend — the configured default list.
    #[test]
    fn cli_replaces_configured_defaults() {
        let cfg = config(Some(vec!["correctness", "performance"]), &[]);
        let resolved = resolve(&cli(&["security"]), &cfg).expect("resolves");
        assert_eq!(names(&resolved), ["security"]);
    }

    #[test]
    fn configured_defaults_used_without_cli() {
        let cfg = config(Some(vec!["tone", "ai-systems"]), &[]);
        let resolved = resolve(&[], &cfg).expect("resolves");
        assert_eq!(names(&resolved), ["tone", "ai-systems"]);
    }

    /// Segments arrive comma-split from clap; surrounding whitespace must not change identity
    /// (`--preset "security, simplicity"` names two real presets).
    #[test]
    fn segments_are_trimmed() {
        let resolved =
            resolve(&cli(&[" security ", "simplicity"]), &config(None, &[])).expect("resolves");
        assert_eq!(names(&resolved), ["security", "simplicity"]);
    }

    #[test]
    fn duplicates_collapse_to_first_seen_order() {
        let resolved = resolve(
            &cli(&["security", "correctness", "security"]),
            &config(None, &[]),
        )
        .expect("resolves");
        assert_eq!(names(&resolved), ["security", "correctness"]);
    }

    /// An unknown name must fail before any LLM call and teach the available vocabulary.
    #[test]
    fn unknown_name_fails_listing_available() {
        let cfg = config(None, &[("api-security", "custom rubric")]);
        let err = resolve(&cli(&["secuirty"]), &cfg).expect_err("unknown preset");
        let msg = format!("{err:#}");
        assert!(msg.contains("secuirty"), "names the offender: {msg}");
        assert!(msg.contains("api-security"), "lists project presets: {msg}");
        assert!(msg.contains("correctness"), "lists built-ins: {msg}");
    }

    /// `--preset ''` and `--preset security,,performance` both produce an empty segment.
    #[test]
    fn empty_segments_are_rejected() {
        for argv in [vec![""], vec!["security", "", "performance"]] {
            let err = resolve(&cli(&argv), &config(None, &[])).expect_err("empty segment");
            assert!(format!("{err:#}").contains("empty preset name"));
        }
    }

    #[test]
    fn empty_configured_default_list_is_rejected() {
        let err = resolve(&[], &config(Some(vec![]), &[])).expect_err("empty defaults");
        assert!(format!("{err:#}").contains("[defaults].presets must not be empty"));
    }

    #[test]
    fn general_is_standalone_after_deduplication() {
        assert_eq!(
            names(&resolve(&cli(&["general", "general"]), &config(None, &[])).expect("resolves")),
            ["general"]
        );

        for selection in [cli(&["general", "security"]), cli(&["security", "general"])] {
            let err = resolve(&selection, &config(None, &[])).expect_err("mixed general");
            assert!(format!("{err:#}").contains("standalone broad review"));
        }

        let cfg = config(Some(vec!["general", "tone"]), &[]);
        assert!(resolve(&[], &cfg).is_err());
    }

    /// A same-named `[presets.<name>]` table replaces the built-in, so projects can
    /// customize an angle without inheritance machinery.
    #[test]
    fn project_definition_overrides_builtin() {
        let cfg = config(None, &[("security", "only look at the auth layer")]);
        let resolved = resolve(&cli(&["security"]), &cfg).expect("resolves");
        assert_eq!(resolved[0].prompt, "only look at the auth layer");
    }

    #[test]
    fn names_are_case_sensitive() {
        let err = resolve(&cli(&["Security"]), &config(None, &[])).expect_err("case mismatch");
        assert!(format!("{err:#}").contains("unknown preset"));
    }

    /// Every selection runs as its own job/lane, so the count is capped — but only after
    /// dedup, since duplicates collapse before any spend.
    #[test]
    fn selection_is_capped_after_dedup() {
        let defs: Vec<(String, String)> = (0..MAX_SELECTED_PRESETS + 1)
            .map(|i| (format!("p{i}"), format!("rubric {i}")))
            .collect();
        let refs: Vec<(&str, &str)> = defs.iter().map(|(n, p)| (n.as_str(), p.as_str())).collect();
        let cfg = config(None, &refs);
        let all: Vec<String> = defs.iter().map(|(n, _)| n.clone()).collect();

        let err = resolve(&all, &cfg).expect_err("over cap");
        assert!(format!("{err:#}").contains("maximum is 16"), "{err:#}");

        let at_cap = all[..MAX_SELECTED_PRESETS].to_vec();
        assert_eq!(
            resolve(&at_cap, &cfg).expect("at cap").len(),
            MAX_SELECTED_PRESETS
        );

        // dedup runs first: a long list of one repeated name is a 1-preset selection
        let repeated = vec!["security".to_string(); MAX_SELECTED_PRESETS + 1];
        assert_eq!(
            resolve(&repeated, &config(None, &[]))
                .expect("resolves")
                .len(),
            1
        );
    }

    /// Every built-in and the default list itself must resolve — a registry typo would
    /// otherwise only surface at run time.
    #[test]
    fn all_builtins_and_default_names_resolve() {
        let cfg = config(None, &[]);
        for (name, _) in BUILT_IN_PRESETS {
            resolve(&cli(&[name]), &cfg).expect("built-in resolves");
        }
        assert!(
            DEFAULT_PRESET_NAMES
                .iter()
                .all(|n| BUILT_IN_PRESETS.iter().any(|(b, _)| b == n))
        );
        for name in ["ml-rigor", "ai-systems", "tone", "general"] {
            assert!(!DEFAULT_PRESET_NAMES.contains(&name), "{name} stays opt-in");
        }
    }
}
