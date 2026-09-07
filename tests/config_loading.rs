use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

struct Fixture {
    _temp: TempDir,
    home: PathBuf,
    repo: PathBuf,
    explicit: PathBuf,
    global: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        let repo = temp.path().join("repo");
        let explicit = temp.path().join("explicit.toml");
        let global = home.join(".nitpicker/config.toml");
        fs::create_dir_all(global.parent().unwrap()).unwrap();
        fs::create_dir(&repo).unwrap();
        let fixture = Self {
            _temp: temp,
            home,
            repo,
            explicit,
            global,
        };
        let output = fixture
            .command("git")
            .args(["init", "--quiet"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        fixture
    }

    fn command(&self, program: &str) -> Command {
        let mut command = Command::new(program);
        command
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", &self.home)
            .env("USERPROFILE", &self.home)
            .env("XDG_CONFIG_HOME", &self.home)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .current_dir(&self.repo);
        command
    }

    fn error(&self, explicit: Option<&Path>) -> String {
        let mut command = self.command(env!("CARGO_BIN_EXE_nitpicker"));
        command.arg("--repo").arg(&self.repo);
        if let Some(path) = explicit {
            command.arg("--config").arg(path);
        }
        let output = command.output().unwrap();
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        assert!(output.stdout.is_empty(), "{output:?}");
        String::from_utf8(output.stderr).unwrap()
    }
}

// Unknown presets identify the selected file after structural validation, before credentials
// or any provider request. Every route requires a synthetic key absent from the child environment.
fn config(source: &str) -> String {
    format!(
        r#"[defaults]
presets = ["source-{source}"]
[aggregator]
provider = "openai"
model = "unused"
api_key_env = "NITPICKER_CONFIG_TEST_KEY"
[[reviewer]]
name = "unused"
provider = "openai"
model = "unused"
api_key_env = "NITPICKER_CONFIG_TEST_KEY"
"#
    )
}

#[test]
fn config_precedence_is_explicit_then_repo_then_global_without_credentials() {
    let fixture = Fixture::new();
    let repo_config = fixture.repo.join("nitpicker.toml");
    fs::write(&fixture.explicit, config("explicit")).unwrap();
    fs::write(&repo_config, config("repo")).unwrap();
    fs::write(&fixture.global, config("global")).unwrap();

    for (explicit, source) in [
        (Some(fixture.explicit.as_path()), "explicit"),
        (None, "repo"),
    ] {
        let error = fixture.error(explicit);
        assert!(
            error.contains(&format!("unknown preset \"source-{source}\"")),
            "{error}"
        );
        assert!(!error.contains("NITPICKER_CONFIG_TEST_KEY"), "{error}");
    }
    fs::remove_file(repo_config).unwrap();
    let error = fixture.error(None);
    assert!(
        error.contains("unknown preset \"source-global\""),
        "{error}"
    );
}

#[test]
fn missing_or_malformed_explicit_config_does_not_fall_back() {
    let fixture = Fixture::new();
    fs::write(fixture.repo.join("nitpicker.toml"), config("repo")).unwrap();
    fs::write(&fixture.global, config("global")).unwrap();

    let error = fixture.error(Some(&fixture.explicit));
    assert!(error.contains("failed to read config"), "{error}");
    assert!(error.contains("explicit.toml"), "{error}");

    fs::write(&fixture.explicit, "[malformed").unwrap();
    let error = fixture.error(Some(&fixture.explicit));
    assert!(error.contains("invalid config"), "{error}");
    assert!(!error.contains("unknown preset"), "{error}");
}

#[test]
fn malformed_repo_config_does_not_fall_back_to_global() {
    let fixture = Fixture::new();
    fs::write(fixture.repo.join("nitpicker.toml"), "[malformed").unwrap();
    fs::write(&fixture.global, config("global")).unwrap();

    let error = fixture.error(None);
    assert!(error.contains("invalid config"), "{error}");
    assert!(!error.contains("unknown preset"), "{error}");
}

#[test]
fn absent_and_malformed_global_configs_report_distinct_failures() {
    let fixture = Fixture::new();
    let error = fixture.error(None);
    assert!(error.contains("no config found"), "{error}");

    fs::write(&fixture.global, "[malformed").unwrap();
    let error = fixture.error(None);
    assert!(error.contains("invalid config"), "{error}");
}

#[test]
fn every_config_source_checks_structure_before_presets_and_credentials() {
    let fixture = Fixture::new();
    let repo_config = fixture.repo.join("nitpicker.toml");
    let invalid = format!("{}max_tokens = 0\n", config("invalid-structure"));
    for path in [&fixture.explicit, &repo_config, &fixture.global] {
        fs::write(path, &invalid).unwrap();
        let explicit = (path == &fixture.explicit).then_some(path.as_path());
        let error = fixture.error(explicit);
        assert!(error.contains("max_tokens"), "{error}");
        assert!(!error.contains("unknown preset"), "{error}");
        assert!(!error.contains("NITPICKER_CONFIG_TEST_KEY"), "{error}");
        fs::remove_file(path).unwrap();
    }
}
