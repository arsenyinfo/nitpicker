use crate::detect;
use eyre::Result;
use nitpicker_agent::config;
use std::path::{Path, PathBuf};

pub(crate) async fn run_init(path: PathBuf, prefer_free: bool) -> eyre::Result<()> {
    println!("Detecting available providers...\n");
    let detected = detect::detect_all().await;

    if detected.is_empty() {
        eyre::bail!(
            "no providers detected — set at least one of: \
             ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, \
             OPENROUTER_API_KEY, KIMI_API_KEY, ZAI_API_KEY, MINIMAX_API_KEY, MISTRAL_API_KEY, \
             DATABRICKS_TOKEN (with DATABRICKS_HOST or ~/.databrickscfg)"
        );
    }

    println!("Detected providers:");
    for d in &detected {
        let key_info = match d.api_key_env {
            Some(env) => env.to_string(),
            None => d.auth.unwrap_or("api_key").to_string(),
        };
        println!("  ✓ {} ({}) via {}", d.name, key_info, d.source);
    }

    let use_openrouter_free = should_prefer_openrouter_free(&detected, prefer_free);
    if prefer_free && !use_openrouter_free {
        println!(
            "\nWarning: `--free` prefers OpenRouter free models, but OPENROUTER_API_KEY is not set; using the normal provider order."
        );
    }

    let prioritized = prioritize_init_detected(&detected, use_openrouter_free);
    let config = build_init_config(&prioritized, use_openrouter_free);
    let mut toml_str = toml::to_string_pretty(&config)
        .map_err(|e| eyre::eyre!("failed to serialize config: {e}"))?;

    let active_names: std::collections::HashSet<&str> = config
        .reviewer
        .iter()
        .map(|r| r.name.as_str())
        .chain(std::iter::once(prioritized[0].name))
        .collect();
    let extras: Vec<&detect::Detected> = detected
        .iter()
        .filter(|d| !active_names.contains(d.name))
        .collect();
    if !extras.is_empty() {
        toml_str.push_str("\n# Other detected providers — uncomment to add as a reviewer:\n");
        for d in extras {
            toml_str.push('\n');
            toml_str.push_str(&format_commented_reviewer(d));
            toml_str.push('\n');
        }
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, &toml_str)?;
    println!("\nCreated {}", path.display());

    print_init_hints(&detected);
    Ok(())
}

fn format_commented_reviewer(d: &detect::Detected) -> String {
    let mut lines = vec![
        "# [[reviewer]]".to_string(),
        format!("# name = \"{}\"", d.name),
        format!("# model = \"{}\"", d.model),
        format!("# provider = \"{}\"", d.provider),
    ];
    if let Some(url) = &d.base_url {
        lines.push(format!("# base_url = \"{url}\""));
    }
    if let Some(env) = d.api_key_env {
        if d.local_server {
            lines.push(format!(
                "# api_key_env = \"{env}\"  # set to any non-empty value"
            ));
        } else {
            lines.push(format!("# api_key_env = \"{env}\""));
        }
    }
    if let Some(auth) = d.auth {
        lines.push(format!("# auth = \"{auth}\""));
    }
    lines.join("\n")
}

fn build_init_config(
    detected: &[&detect::Detected],
    prefer_openrouter_free: bool,
) -> config::Config {
    let non_local_count = detected.iter().filter(|d| !d.local_server).count();
    let debate = non_local_count >= 2;

    // aggregator: highest priority (list is already sorted)
    let agg = detected[0];
    let aggregator = config::AggregatorConfig {
        model: init_model_for_detected(agg, prefer_openrouter_free),
        provider: parse_provider_type(agg.provider),
        base_url: agg.base_url.clone(),
        api_key_env: agg.api_key_env.map(str::to_string),
        max_tokens: None,
        auth: agg.auth.map(str::to_string),
        azure_scope: None,
        azure_credentials: None,
    };

    // Fallback needs a second route even when debate is disabled. OpenRouter free selection can
    // produce two distinct model routes from its one detected credential.
    let reviewer_slots = if detected.len() >= 2 || prefer_openrouter_free {
        2
    } else {
        1
    };
    let reviewers = pick_reviewers(detected, reviewer_slots, prefer_openrouter_free);
    let fallback = reviewers.len() >= 2;

    config::Config {
        defaults: Some(config::DefaultsConfig {
            debate: Some(debate),
            alloy: None,
            fallback: Some(fallback),
            max_turns: Some(config::DEFAULT_MAX_TURNS),
            compact_threshold: Some(100_000),
            log_trajectories: Some(false),
            presets: None,
        }),
        aggregator,
        reviewer: reviewers,
        presets: None,
    }
}

fn pick_reviewers(
    detected: &[&detect::Detected],
    count: usize,
    prefer_openrouter_free: bool,
) -> Vec<config::ReviewerConfig> {
    if prefer_openrouter_free {
        return detected
            .first()
            .into_iter()
            .cycle()
            .take(count)
            .map(|d| make_reviewer(d, prefer_openrouter_free))
            .collect();
    }

    let mut result = Vec::new();
    let mut seen_names: std::collections::HashSet<&str> = Default::default();

    // first pass: diverse provider names
    for d in detected {
        if result.len() >= count {
            break;
        }
        if seen_names.insert(d.name) {
            result.push(make_reviewer(d, prefer_openrouter_free));
        }
    }

    result
}

fn make_reviewer(d: &detect::Detected, prefer_openrouter_free: bool) -> config::ReviewerConfig {
    config::ReviewerConfig {
        name: d.name.to_string(),
        model: init_model_for_detected(d, prefer_openrouter_free),
        provider: parse_provider_type(d.provider),
        base_url: d.base_url.clone(),
        api_key_env: d.api_key_env.map(str::to_string),
        max_tokens: None,
        compact_threshold: None,
        auth: d.auth.map(str::to_string),
        azure_scope: None,
        azure_credentials: None,
    }
}

fn should_prefer_openrouter_free(detected: &[detect::Detected], prefer_free: bool) -> bool {
    if !prefer_free {
        return false;
    }

    let has_openrouter = detected.iter().any(|d| d.name == "openrouter");
    has_openrouter && std::env::var("OPENROUTER_API_KEY").is_ok()
}

fn prioritize_init_detected(
    detected: &[detect::Detected],
    prefer_openrouter_free: bool,
) -> Vec<&detect::Detected> {
    let mut prioritized: Vec<&detect::Detected> = detected.iter().collect();
    if prefer_openrouter_free {
        prioritized.sort_by_key(|d| if d.name == "openrouter" { 0 } else { 1 });
    }
    prioritized
}

fn init_model_for_detected(d: &detect::Detected, prefer_openrouter_free: bool) -> String {
    if prefer_openrouter_free && d.name == "openrouter" {
        return "free".to_string();
    }

    d.model.clone()
}

fn parse_provider_type(s: &str) -> config::ProviderType {
    match s {
        "anthropic" => config::ProviderType::Anthropic,
        "gemini" => config::ProviderType::Gemini,
        "openrouter" => config::ProviderType::OpenRouter,
        _ => config::ProviderType::OpenAi,
    }
}

fn print_init_hints(detected: &[detect::Detected]) {
    let unset: Vec<&detect::Detected> = detected
        .iter()
        .filter(|d| {
            !d.local_server
                && d.api_key_env
                    .map(|env| std::env::var(env).is_err())
                    .unwrap_or(false)
        })
        .collect();

    if !unset.is_empty() {
        println!("\nProviders detected but env vars not yet set:");
        for d in unset {
            println!(
                "  export {}=...  # found via {}",
                d.api_key_env.unwrap(),
                d.source
            );
        }
    }

    let has_google_ai_key =
        std::env::var("GOOGLE_AI_API_KEY").is_ok() && std::env::var("GEMINI_API_KEY").is_err();
    if has_google_ai_key {
        println!("\n  Note: found GOOGLE_AI_API_KEY — the gemini client reads GEMINI_API_KEY;");
        println!("  add `export GEMINI_API_KEY=$GOOGLE_AI_API_KEY` to your shell profile.");
    }
}

pub(crate) fn init_config_path(global: bool, repo: &Path) -> Result<PathBuf> {
    if global {
        let home =
            dirs::home_dir().ok_or_else(|| eyre::eyre!("failed to resolve home directory"))?;
        Ok(home.join(".nitpicker").join("config.toml"))
    } else {
        // --repo is a global flag, so `init --repo <dir>` must target that repo's root
        // rather than silently writing into the cwd
        Ok(repo.join("nitpicker.toml"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_writes_into_the_repo_named_by_the_global_repo_flag() {
        let path = init_config_path(false, Path::new("/some/repo")).unwrap();
        assert_eq!(path, PathBuf::from("/some/repo/nitpicker.toml"));
    }

    fn detected_provider(name: &'static str, provider: &'static str) -> detect::Detected {
        detect::Detected {
            name,
            provider,
            model: format!("{name}-model"),
            base_url: None,
            api_key_env: None,
            auth: None,
            source: "test",
            local_server: false,
        }
    }

    #[test]
    fn init_enables_fallback_when_it_can_generate_two_routes() {
        let first = detected_provider("first", "openai");
        let second = detected_provider("second", "anthropic");
        let config = build_init_config(&[&first, &second], false);

        assert_eq!(config.reviewer.len(), 2);
        assert!(config.default_fallback());
        assert!(
            toml::to_string_pretty(&config)
                .unwrap()
                .contains("fallback = true")
        );
    }

    #[test]
    fn free_init_generates_two_fallback_routes_from_openrouter() {
        let openrouter = detected_provider("openrouter", "openrouter");
        let config = build_init_config(&[&openrouter], true);

        assert_eq!(config.reviewer.len(), 2);
        assert!(config.reviewer.iter().all(|route| route.model == "free"));
        assert!(config.default_fallback());
        assert!(
            toml::to_string_pretty(&config)
                .unwrap()
                .contains("fallback = true")
        );
    }

    #[test]
    fn init_does_not_enable_impossible_single_route_fallback() {
        let only = detected_provider("only", "openai");
        let config = build_init_config(&[&only], false);

        assert_eq!(config.reviewer.len(), 1);
        assert!(!config.default_fallback());
    }

    #[test]
    fn init_selection_preserves_order_and_deduplicates_names() {
        let mut local = detected_provider("local", "openai");
        local.local_server = true;
        let remote = detected_provider("remote", "anthropic");
        let mut duplicate = detected_provider("remote", "gemini");
        duplicate.model = "ignored-duplicate".into();
        for count in [0, 1, 2, 3, 10] {
            let reviewers = pick_reviewers(&[&local, &remote, &duplicate], count, false);
            let names: Vec<_> = reviewers.iter().map(|r| r.name.as_str()).collect();
            assert_eq!(names, ["local", "remote"][..count.min(2)]);
            if reviewers.len() == 2 {
                assert_eq!(reviewers[1].model, "remote-model");
            }
        }
        let config = build_init_config(&[&local, &remote], false);
        assert!(!config.default_debate());
        assert!(config.default_fallback());
        assert_eq!(config.aggregator.model, "local-model");
        let empty: [&detect::Detected; 0] = [];
        assert!(pick_reviewers(&empty, 2, false).is_empty());
        assert!(pick_reviewers(&empty, 2, true).is_empty());
    }

    #[test]
    fn init_free_prioritization_is_stable_and_only_changes_openrouter_models() {
        let detected = [
            detected_provider("first", "anthropic"),
            detected_provider("openrouter", "openrouter"),
            detected_provider("last", "gemini"),
        ];
        let names =
            |items: Vec<&detect::Detected>| items.iter().map(|d| d.name).collect::<Vec<_>>();
        assert_eq!(
            names(prioritize_init_detected(&detected, false)),
            ["first", "openrouter", "last"]
        );
        let prioritized = prioritize_init_detected(&detected, true);
        assert_eq!(names(prioritized.clone()), ["openrouter", "first", "last"]);
        let config = build_init_config(&prioritized, true);
        assert_eq!(config.aggregator.model, "free");
        assert_eq!(config.reviewer.len(), 2);
        assert!(
            config
                .reviewer
                .iter()
                .all(|r| r.name == "openrouter" && r.model == "free")
        );
        assert_eq!(init_model_for_detected(&detected[0], true), "first-model");
    }

    #[test]
    fn init_template_and_commented_provider_bytes_are_stable() {
        let mut provider = detected_provider("local", "openai");
        provider.local_server = true;
        provider.base_url = Some("http://localhost:1234/v1".into());
        provider.api_key_env = Some("LOCAL_KEY");
        provider.auth = Some("example-auth");
        let rendered = toml::to_string_pretty(&build_init_config(&[&provider], false)).unwrap();
        assert_eq!(rendered, include_str!("../tests/fixtures/init-config.toml"));
        assert_eq!(
            format_commented_reviewer(&provider),
            include_str!("../tests/fixtures/init-comment.txt").trim_end_matches('\n')
        );
        provider.base_url = None;
        provider.api_key_env = None;
        provider.auth = None;
        assert_eq!(
            format_commented_reviewer(&provider),
            "# [[reviewer]]\n# name = \"local\"\n# model = \"local-model\"\n# provider = \"openai\""
        );
    }
}
