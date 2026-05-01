use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

pub struct Detected {
    pub name: &'static str,
    pub provider: &'static str,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key_env: Option<&'static str>,
    pub auth: Option<&'static str>,
    pub source: &'static str,
    /// true for local servers that accept any non-empty API key value
    pub local_server: bool,
}

struct ProviderDef {
    name: &'static str,
    provider: &'static str,
    model: &'static str,
    base_url: Option<&'static str>,
    api_key_env: &'static str,
}

// ordered highest to lowest priority
const PROVIDER_ORDER: &[&str] = &[
    "databricks",
    "openrouter",
    "anthropic",
    "gemini",
    "openai",
    "kimi",
    "mistral",
    "zai",
    "minimax",
    "ollama",
    "lmstudio",
];

fn priority_index(name: &str) -> usize {
    PROVIDER_ORDER.iter().position(|&n| n == name).unwrap_or(usize::MAX)
}

static ANTHROPIC: ProviderDef = ProviderDef {
    name: "anthropic",
    provider: "anthropic",
    model: "claude-sonnet-4-6",
    base_url: None,
    api_key_env: "ANTHROPIC_API_KEY",
};
static OPENAI: ProviderDef = ProviderDef {
    name: "openai",
    provider: "openai",
    model: "gpt-5.5",
    base_url: None,
    api_key_env: "OPENAI_API_KEY",
};
static GEMINI_API: ProviderDef = ProviderDef {
    name: "gemini",
    provider: "gemini",
    model: "gemini-3-flash-preview",
    base_url: None,
    api_key_env: "GEMINI_API_KEY",
};
static MISTRAL: ProviderDef = ProviderDef {
    name: "mistral",
    provider: "openai",
    model: "mistral-medium-3.5",
    base_url: Some("https://api.mistral.ai/v1"),
    api_key_env: "MISTRAL_API_KEY",
};
static OPENROUTER: ProviderDef = ProviderDef {
    name: "openrouter",
    provider: "openrouter",
    model: "deepseek/deepseek-v4-pro",
    base_url: None,
    api_key_env: "OPENROUTER_API_KEY",
};
static KIMI: ProviderDef = ProviderDef {
    name: "kimi",
    provider: "anthropic",
    model: "kimi-for-coding",
    base_url: Some("https://api.kimi.com/coding/"),
    api_key_env: "KIMI_API_KEY",
};
static ZAI: ProviderDef = ProviderDef {
    name: "zai",
    provider: "anthropic",
    model: "glm-5.1",
    base_url: Some("https://api.z.ai/api/anthropic"),
    api_key_env: "ZAI_API_KEY",
};
static MINIMAX: ProviderDef = ProviderDef {
    name: "minimax",
    provider: "anthropic",
    model: "MiniMax-M2.7",
    base_url: Some("https://api.minimax.io/anthropic"),
    api_key_env: "MINIMAX_API_KEY",
};

static ENV_PROVIDERS: &[&ProviderDef] = &[
    &ANTHROPIC, &OPENAI, &GEMINI_API, &MISTRAL, &OPENROUTER, &KIMI, &ZAI, &MINIMAX,
];

fn opencode_name_to_provider(name: &str) -> Option<&'static ProviderDef> {
    match name {
        "anthropic" => Some(&ANTHROPIC),
        "openai" => Some(&OPENAI),
        "minimax" => Some(&MINIMAX),
        "kimi-for-coding" | "moonshotai" | "kimi" => Some(&KIMI),
        "zai-coding-plan" | "zai" => Some(&ZAI),
        "mistral" => Some(&MISTRAL),
        "openrouter" => Some(&OPENROUTER),
        _ => None,
    }
}

fn from_def(def: &'static ProviderDef, source: &'static str) -> Detected {
    Detected {
        name: def.name,
        provider: def.provider,
        model: def.model.to_string(),
        base_url: def.base_url.map(str::to_string),
        api_key_env: Some(def.api_key_env),
        auth: None,
        source,
        local_server: false,
    }
}

#[derive(Deserialize)]
struct OpencodeEntry {
    #[serde(rename = "type")]
    auth_type: String,
    key: Option<String>,
}

pub async fn detect_all() -> Vec<Detected> {
    let mut detected: Vec<Detected> = Vec::new();

    // env vars for known providers (dedup by provider name)
    for &def in ENV_PROVIDERS {
        if std::env::var(def.api_key_env).is_ok()
            && !detected.iter().any(|d| d.name == def.name)
        {
            detected.push(from_def(def, "env var"));
        }
    }

    // GOOGLE_AI_API_KEY as alias for GEMINI_API_KEY
    if std::env::var("GOOGLE_AI_API_KEY").is_ok()
        && !detected.iter().any(|d| d.name == "gemini")
    {
        detected.push(Detected {
            name: "gemini",
            provider: "gemini",
            model: GEMINI_API.model.to_string(),
            base_url: None,
            // intentionally no api_key_env: the rig gemini client reads GEMINI_API_KEY;
            // we note in init output that the user should alias it
            api_key_env: None,
            auth: None,
            source: "env var (GOOGLE_AI_API_KEY — alias GEMINI_API_KEY)",
            local_server: false,
        });
    }

    // gemini oauth from opencode or gcloud
    if !detected.iter().any(|d| d.name == "gemini") {
        if let Some(oauth) = detect_gemini_oauth() {
            detected.push(oauth);
        }
    }

    // opencode api-key providers not already found via env vars
    for d in detect_opencode() {
        let already = detected
            .iter()
            .any(|e| e.api_key_env == d.api_key_env || e.name == d.name);
        if !already {
            detected.push(d);
        }
    }

    // local servers
    if let Some(d) = detect_ollama().await {
        if !detected.iter().any(|e| e.name == "ollama") {
            detected.push(d);
        }
    }
    if let Some(d) = detect_lmstudio().await {
        if !detected.iter().any(|e| e.name == "lmstudio") {
            detected.push(d);
        }
    }

    if let Some(d) = detect_databricks() {
        if !detected.iter().any(|e| e.name == "databricks") {
            detected.push(d);
        }
    }

    detected.sort_by_key(|d| priority_index(d.name));
    detected
}

fn detect_gemini_oauth() -> Option<Detected> {
    if let Some(path) = opencode_auth_path() {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(auth) = serde_json::from_str::<HashMap<String, OpencodeEntry>>(&content) {
                if matches!(auth.get("google"), Some(e) if e.auth_type == "oauth") {
                    return Some(gemini_oauth_detected("opencode (google oauth)"));
                }
            }
        }
    }

    if let Some(home) = dirs::home_dir() {
        if home
            .join(".config/gcloud/application_default_credentials.json")
            .exists()
        {
            return Some(gemini_oauth_detected("gcloud ADC"));
        }
    }

    None
}

fn gemini_oauth_detected(source: &'static str) -> Detected {
    Detected {
        name: "gemini",
        provider: "gemini",
        model: "gemini-3-flash-preview".to_string(),
        base_url: None,
        api_key_env: None,
        auth: Some("oauth"),
        source,
        local_server: false,
    }
}

fn opencode_auth_path() -> Option<PathBuf> {
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))?;
    let path = base.join("opencode/auth.json");
    path.exists().then_some(path)
}

fn detect_opencode() -> Vec<Detected> {
    let Some(path) = opencode_auth_path() else {
        return Vec::new();
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(auth) = serde_json::from_str::<HashMap<String, OpencodeEntry>>(&content) else {
        return Vec::new();
    };

    let mut results = Vec::new();
    for (name, entry) in &auth {
        if entry.auth_type != "api" {
            continue;
        }
        let key = entry.key.as_deref().unwrap_or("");
        if key.len() < 10 {
            continue;
        }
        if let Some(def) = opencode_name_to_provider(name) {
            results.push(from_def(def, "opencode"));
        }
    }
    results
}

async fn detect_ollama() -> Option<Detected> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1))
        .build()
        .ok()?;

    if let Ok(resp) = client.get("http://localhost:11434/api/tags").send().await {
        if resp.status().is_success() {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(model) = json["models"]
                    .as_array()
                    .and_then(|m| m.first())
                    .and_then(|m| m["name"].as_str())
                {
                    return Some(ollama_detected(model.to_string(), "ollama (running)"));
                }
            }
        }
    }

    // installed but not running
    let binary_found = std::process::Command::new("which")
        .arg("ollama")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if binary_found {
        let model = find_ollama_installed_model().unwrap_or_else(|| "llama3.2".to_string());
        return Some(ollama_detected(model, "ollama (installed, not running)"));
    }

    None
}

fn ollama_detected(model: String, source: &'static str) -> Detected {
    Detected {
        name: "ollama",
        provider: "openai",
        model,
        base_url: Some("http://localhost:11434/v1".to_string()),
        api_key_env: Some("OLLAMA_API_KEY"),
        auth: None,
        source,
        local_server: true,
    }
}

fn find_ollama_installed_model() -> Option<String> {
    let manifests = dirs::home_dir()?
        .join(".ollama/models/manifests/registry.ollama.ai/library");
    std::fs::read_dir(manifests)
        .ok()?
        .flatten()
        .find(|e| e.file_type().ok().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().into_owned())
}

async fn detect_lmstudio() -> Option<Detected> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1))
        .build()
        .ok()?;

    if let Ok(resp) = client.get("http://localhost:1234/v1/models").send().await {
        if resp.status().is_success() {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(model) = json["data"]
                    .as_array()
                    .and_then(|m| m.first())
                    .and_then(|m| m["id"].as_str())
                {
                    return Some(lmstudio_detected(model.to_string(), "lmstudio (running)"));
                }
            }
        }
    }

    if lmstudio_app_exists() {
        return Some(lmstudio_detected(
            "local-model".to_string(),
            "lmstudio (installed, not running — update model name after loading one)",
        ));
    }

    None
}

fn lmstudio_detected(model: String, source: &'static str) -> Detected {
    Detected {
        name: "lmstudio",
        provider: "openai",
        model,
        base_url: Some("http://localhost:1234/v1".to_string()),
        api_key_env: Some("LMSTUDIO_API_KEY"),
        auth: None,
        source,
        local_server: true,
    }
}

fn lmstudio_app_exists() -> bool {
    std::path::Path::new("/Applications/LM Studio.app").exists()
        || dirs::home_dir()
            .map(|h| {
                h.join(".local/share/applications/lm-studio.desktop")
                    .exists()
            })
            .unwrap_or(false)
}

fn detect_databricks() -> Option<Detected> {
    let has_token = std::env::var("DATABRICKS_TOKEN").is_ok();
    let has_cli = std::process::Command::new("which")
        .arg("databricks")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !has_token && !has_cli {
        return None;
    }

    let host = std::env::var("DATABRICKS_HOST")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(read_databricks_host_from_cfg)?;

    let base_url = format!("{}/serving-endpoints", host.trim_end_matches('/'));
    let source = if has_token { "env var" } else { "databricks CLI" };

    Some(Detected {
        name: "databricks",
        provider: "openai",
        model: "databricks-claude-sonnet-4-6".to_string(),
        base_url: Some(base_url),
        api_key_env: Some("DATABRICKS_TOKEN"),
        auth: None,
        source,
        local_server: false,
    })
}

fn read_databricks_host_from_cfg() -> Option<String> {
    let path = dirs::home_dir()?.join(".databrickscfg");
    let content = std::fs::read_to_string(path).ok()?;
    let mut in_default = false;
    for line in content.lines() {
        let line = line.trim();
        if line == "[DEFAULT]" {
            in_default = true;
            continue;
        }
        if line.starts_with('[') {
            in_default = false;
            continue;
        }
        if in_default {
            if let Some(rest) = line.strip_prefix("host") {
                let val = rest.trim_start_matches([' ', '=']).trim();
                if !val.is_empty() {
                    return Some(val.to_string());
                }
            }
        }
    }
    None
}
