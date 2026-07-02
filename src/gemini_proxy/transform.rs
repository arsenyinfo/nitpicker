use eyre::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
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
    #[serde(rename = "toolConfig")]
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

pub fn transform_stream_response(body: &str) -> Result<Value> {
    let events = parse_sse_events(body)?;
    if events.is_empty() {
        eyre::bail!("upstream SSE response did not contain JSON data events");
    }

    let mut response = serde_json::Map::new();
    let mut candidate = serde_json::Map::new();
    let mut content = serde_json::Map::new();
    let mut parts = Vec::new();

    for event in events {
        let Some(event_response) = event.get("response").unwrap_or(&event).as_object() else {
            continue;
        };

        for (key, value) in event_response {
            if key != "candidates" {
                response.insert(key.clone(), value.clone());
            }
        }

        let Some(event_candidate) = event_response
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
            .and_then(Value::as_object)
        else {
            continue;
        };

        for (key, value) in event_candidate {
            if key != "content" {
                candidate.insert(key.clone(), value.clone());
            }
        }

        let Some(event_content) = event_candidate.get("content").and_then(Value::as_object) else {
            continue;
        };

        for (key, value) in event_content {
            if key != "parts" {
                content.insert(key.clone(), value.clone());
            }
        }

        if let Some(event_parts) = event_content.get("parts").and_then(Value::as_array) {
            parts.extend(event_parts.iter().cloned());
        }
    }

    if !parts.is_empty() {
        content.insert("parts".to_string(), Value::Array(parts));
    }
    if !content.is_empty() {
        candidate.insert("content".to_string(), Value::Object(content));
    }
    if !candidate.is_empty() {
        response.insert(
            "candidates".to_string(),
            Value::Array(vec![Value::Object(candidate)]),
        );
    }

    Ok(Value::Object(response))
}

fn parse_sse_events(body: &str) -> Result<Vec<Value>> {
    let mut events = Vec::new();
    let mut data = String::new();

    for line in body.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            flush_sse_event(&mut events, &mut data)?;
            continue;
        }

        let Some(chunk) = line.strip_prefix("data:") else {
            continue;
        };
        if !data.is_empty() {
            data.push('\n');
        }
        data.push_str(chunk.trim_start());
    }

    flush_sse_event(&mut events, &mut data)?;
    Ok(events)
}

fn flush_sse_event(events: &mut Vec<Value>, data: &mut String) -> Result<()> {
    let trimmed = data.trim();
    if trimmed.is_empty() {
        data.clear();
        return Ok(());
    }
    if trimmed != "[DONE]" {
        events.push(serde_json::from_str(trimmed)?);
    }
    data.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_camelcase_request_fields() {
        // rig-core's Gemini client (the real producer hitting this proxy) serializes camelCase
        // keys. Without rename_all, these fields silently defaulted to None, dropping the system
        // prompt and generation params on every request.
        let body = serde_json::json!({
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
            "systemInstruction": { "role": "system", "parts": [{ "text": "be terse" }] },
            "generationConfig": { "maxOutputTokens": 42, "topP": 0.9, "topK": 5 },
            "toolConfig": { "functionCallingConfig": { "mode": "AUTO" } }
        });
        let req: GeminiRequest = serde_json::from_value(body).unwrap();
        assert!(req.system_instruction.is_some(), "systemInstruction dropped");
        assert!(req.tool_config.is_some(), "toolConfig dropped");
        let cfg = req.generation_config.expect("generationConfig dropped");
        assert_eq!(cfg.max_output_tokens, Some(42));
        assert_eq!(cfg.top_p, Some(0.9));
        assert_eq!(cfg.top_k, Some(5));
    }

    #[test]
    fn outbound_payload_renames_tool_config() {
        let payload = GeminiRequestPayload {
            contents: vec![],
            generation_config: None,
            system_instruction: None,
            tools: None,
            tool_config: Some(serde_json::json!({ "x": 1 })),
        };
        let value = serde_json::to_value(&payload).unwrap();
        assert!(value.get("toolConfig").is_some());
        assert!(value.get("tool_config").is_none());
    }
}
