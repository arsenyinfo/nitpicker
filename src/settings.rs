use eyre::Result;
use nitpicker_agent::{config, openrouter};
use std::path::Path;

pub(crate) fn load_config(explicit_path: Option<&Path>, repo: &Path) -> Result<config::Config> {
    if let Some(path) = explicit_path {
        read_config_file(path)
    } else if repo.join("nitpicker.toml").exists() {
        read_config_file(&repo.join("nitpicker.toml"))
    } else {
        load_global_config()
    }
}

fn read_config_file(path: &Path) -> Result<config::Config> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| eyre::eyre!("failed to read config {:?}: {e}", path))?;
    let config: config::Config =
        toml::from_str(&content).map_err(|e| eyre::eyre!("invalid config: {e}"))?;
    config.validate_structure()?;
    Ok(config)
}

/// The `~/.nitpicker/config.toml` fallback alone — `pr` mode reaches for this directly,
/// since its repo-level config comes from the PR base branch blob, never the working tree.
pub(crate) fn load_global_config() -> Result<config::Config> {
    let path = dirs::home_dir()
        .map(|home| home.join(".nitpicker").join("config.toml"))
        .filter(|path| path.exists())
        .ok_or_else(|| {
            eyre::eyre!("no config found — run `nitpicker init [--global]` to generate one")
        })?;
    read_config_file(&path)
}

pub(crate) async fn load_resolved_config(
    explicit_path: Option<&Path>,
    repo: &Path,
) -> Result<config::Config> {
    let mut config = load_config(explicit_path, repo)?;
    finalize_routing_config(&mut config, false).await?;
    Ok(config)
}

/// Finish config validation and experimental route resolution after the caller has resolved the
/// effective CLI/config fallback mode. Strict execution requires every credential up front;
/// fallback execution lets route construction skip unusable entries.
pub(crate) async fn finalize_routing_config(
    config: &mut config::Config,
    fallback: bool,
) -> Result<()> {
    if !fallback {
        config.validate_credentials()?;
    }
    openrouter::resolve_free_models_with_fallback(config, fallback).await
}

pub(crate) fn resolve_routing_modes(
    config: &config::Config,
    cli_alloy: bool,
    cli_fallback: bool,
) -> Result<(bool, bool)> {
    let alloy = cli_alloy || config.default_alloy();
    config.validate_alloy(alloy)?;
    let fallback = cli_fallback || config.default_fallback();
    config.validate_fallback(fallback)?;
    Ok((alloy, fallback))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_modes_reject_cli_pooling_with_one_reviewer() {
        let config: config::Config = toml::from_str(
            r#"
                [aggregator]
                model = "m"
                provider = "openai"
                auth = "codex"

                [[reviewer]]
                model = "m"
                provider = "openai"
                auth = "codex"
            "#,
        )
        .unwrap();

        let err = resolve_routing_modes(&config, false, true).unwrap_err();
        assert!(format!("{err:#}").contains("requires at least 2 reviewers"));
        let err = resolve_routing_modes(&config, true, false).unwrap_err();
        assert!(format!("{err:#}").contains("--alloy requires at least 2 reviewers"));
        assert_eq!(
            resolve_routing_modes(&config, false, false).unwrap(),
            (false, false)
        );
    }

    #[test]
    fn routing_modes_combine_each_cli_and_config_setting_independently() {
        for configured_alloy in [false, true] {
            for configured_fallback in [false, true] {
                let config: config::Config = toml::from_str(&format!(
                    r#"
                    [defaults]
                    alloy = {configured_alloy}
                    fallback = {configured_fallback}
                    [aggregator]
                    provider = "openai"
                    [[reviewer]]
                    provider = "openai"
                    [[reviewer]]
                    provider = "openai"
                "#
                ))
                .unwrap();
                for cli_alloy in [false, true] {
                    for cli_fallback in [false, true] {
                        assert_eq!(
                            resolve_routing_modes(&config, cli_alloy, cli_fallback).unwrap(),
                            (
                                configured_alloy || cli_alloy,
                                configured_fallback || cli_fallback
                            )
                        );
                    }
                }
            }
        }
    }

    /// The config file shape for presets: `[presets.<name>]` tables and the
    /// `[defaults].presets` selection list round-trip through the library's Config.
    #[test]
    fn preset_config_tables_parse_and_validate() {
        let toml_str = r#"
            [defaults]
            presets = ["tone", "security"]

            [aggregator]
            model = "m"
            provider = "openai"
            auth = "codex"

            [[reviewer]]
            name = "r"
            model = "m"
            provider = "openai"
            auth = "codex"

            [presets.tone]
            prompt = "review the docs for tone"
        "#;
        let config: config::Config = toml::from_str(toml_str).expect("parses");
        config.validate().expect("validates");
        let defaults = config.defaults.as_ref().expect("defaults present");
        assert_eq!(
            defaults.presets.as_deref(),
            Some(&["tone".to_string(), "security".to_string()][..])
        );
        let presets = config.presets.as_ref().expect("presets present");
        assert_eq!(presets["tone"].prompt, "review the docs for tone");
    }

    #[test]
    fn unknown_fields_inside_a_preset_table_are_rejected() {
        let toml_str = r#"
            [aggregator]
            model = "m"
            provider = "openai"
            auth = "codex"

            [[reviewer]]
            model = "m"
            provider = "openai"
            auth = "codex"

            [presets.tone]
            prompt = "p"
            model = "sneaky-per-preset-model"
        "#;
        assert!(toml::from_str::<config::Config>(toml_str).is_err());
    }

    #[test]
    fn blank_preset_prompts_fail_validation() {
        let toml_str = r#"
            [aggregator]
            model = "m"
            provider = "openai"
            auth = "codex"

            [[reviewer]]
            model = "m"
            provider = "openai"
            auth = "codex"

            [presets.tone]
            prompt = "   "
        "#;
        let config: config::Config = toml::from_str(toml_str).expect("parses");
        let err = config.validate().expect_err("blank prompt");
        assert!(format!("{err:#}").contains("[presets.tone].prompt"));
    }
}
