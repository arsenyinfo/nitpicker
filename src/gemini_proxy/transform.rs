use eyre::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct GeminiRequest {
    #[serde(default)]
    pub contents: Vec<Content>,
    #[serde(default)]
    pub generation_config: Option<GenerationConfig>,
    #[serde(default)]
    pub system_instruction: Option<Content>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_config: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Content {
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

/// Flat struct covering all Gemini part types (text, thought, function call/response, inline data).
/// Using a flat struct rather than an untagged enum avoids ambiguity when thought parts
/// have both `text` and `thought` fields.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Present and true on thought (reasoning) parts from Gemini 2.x+
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    /// Opaque signature required by Code Assist when thought parts are echoed back
    #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    #[serde(rename = "inlineData", skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<InlineData>,
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
}

impl Part {
    fn is_thought(&self) -> bool {
        self.thought == Some(true)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionResponse {
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<i32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct CodeAssistRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub model: String,
    #[serde(rename = "userPromptId")]
    pub user_prompt_id: String,
    pub request: GeminiRequestPayload,
}

#[derive(Debug, Serialize)]
pub struct GeminiRequestPayload {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "generationConfig")]
    pub generation_config: Option<serde_json::Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "systemInstruction")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<Value>,
}

pub fn extract_model_from_path(path: &str) -> String {
    path.split('/')
        .find(|s| s.starts_with("gemini-"))
        .map(|s| s.split(':').next().unwrap_or(s).to_string())
        // default to gemini-2.5-flash when no model segment is present
        .unwrap_or_else(|| "gemini-2.5-flash".to_string())
}

pub fn transform_request(
    gemini_req: GeminiRequest,
    model: String,
    project_id: Option<String>,
) -> CodeAssistRequest {
    let generation_config = gemini_req.generation_config.map(|config| {
        let mut map = serde_json::Map::new();
        if let Some(temp) = config.temperature {
            map.insert("temperature".to_string(), serde_json::json!(temp));
        }
        if let Some(max_tokens) = config.max_output_tokens {
            map.insert("maxOutputTokens".to_string(), serde_json::json!(max_tokens));
        }
        if let Some(top_p) = config.top_p {
            map.insert("topP".to_string(), serde_json::json!(top_p));
        }
        if let Some(top_k) = config.top_k {
            map.insert("topK".to_string(), serde_json::json!(top_k));
        }
        map
    });

    // strip thought parts from model turns: Code Assist requires thought_signature when
    // thought parts are echoed back, but rig drops them from history, causing 400 errors.
    // removing them entirely is accepted fine.
    let contents = gemini_req
        .contents
        .into_iter()
        .map(|mut content| {
            if content.role.as_deref() == Some("model") {
                content.parts.retain(|p| !p.is_thought());
            }
            content
        })
        .collect();

    CodeAssistRequest {
        project: project_id,
        model,
        user_prompt_id: Uuid::new_v4().to_string(),
        request: GeminiRequestPayload {
            contents,
            generation_config,
            system_instruction: gemini_req.system_instruction,
            tools: gemini_req.tools,
            tool_config: gemini_req.tool_config,
        },
    }
}

pub fn transform_response(code_assist_response: Value) -> Result<Value> {
    let mut response = if let Some(inner) = code_assist_response.get("response") {
        inner.clone()
    } else {
        code_assist_response
    };

    // rig-core requires responseId to be present
    if response.get("responseId").is_none() {
        if let Some(obj) = response.as_object_mut() {
            obj.insert(
                "responseId".to_string(),
                serde_json::json!(Uuid::new_v4().to_string()),
            );
        }
    }

    Ok(response)
}
