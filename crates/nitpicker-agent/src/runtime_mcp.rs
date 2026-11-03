//! The runtime gateway is required before the first model turn. Only policy
//! controls are imported; repository review never gains general gateway tools.
use crate::tools::Tool;
use eyre::{Result, bail, eyre};
use reqwest::{Client, Url, header::HeaderMap};
use rig_core::completion::ToolDefinition;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    path::PathBuf,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

const POLICY_TOOLS: [&str; 2] = [
    "archestra__get_remedy_plans",
    "archestra__execute_remedy_plan",
];
const MAX_RESPONSE: usize = 2 * 1024 * 1024;
const PROTOCOL: &str = "2025-03-26";

pub(crate) fn is_policy_tool(name: &str) -> bool {
    POLICY_TOOLS.contains(&name)
}

pub(crate) async fn add_runtime_tools(tools: &mut HashMap<String, Arc<dyn Tool>>) -> Result<()> {
    if std::env::var("ARCHESTRA_AGENT_RUNTIME_TASK_ID").is_err() {
        return Ok(());
    }
    // Fail closed on a partially configured runtime instead of reviewing without policy controls.
    let url = std::env::var("ARCHESTRA_MCP_GATEWAY_URL")?;
    let token = std::env::var("ARCHESTRA_MCP_GATEWAY_TOKEN")?;
    let headers = crate::llm::runtime_proxy_headers(Some(&url))?;
    if !headers.contains_key("x-archestra-run-id") || !headers.contains_key("x-appa-session-id") {
        bail!("runtime MCP gateway must share the injected proxy origin and session identity");
    }
    let gateway = Gateway::connect(&url, &token, headers).await?;
    let definitions = gateway.discover().await?;
    for definition in definitions {
        let name = definition.name.clone();
        if tools.contains_key(&name) {
            bail!("runtime MCP tool collides with local tool: {name}");
        }
        tools.insert(
            name,
            Arc::new(GatewayTool {
                gateway: Arc::clone(&gateway),
                definition,
            }),
        );
    }
    tracing::info!(
        tools = POLICY_TOOLS.len(),
        "runtime MCP policy tools initialized"
    );
    Ok(())
}

struct Gateway {
    client: Client,
    url: Url,
    headers: HeaderMap,
    next_id: AtomicU64,
}

impl Gateway {
    async fn connect(url: &str, token: &str, mut headers: HeaderMap) -> Result<Arc<Self>> {
        if token.trim().is_empty() {
            bail!("runtime MCP gateway token is empty");
        }
        headers.insert("authorization", format!("Bearer {token}").parse()?);
        headers.insert("accept", "application/json, text/event-stream".parse()?);
        let mut gateway = Self {
            client: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(120))
                .build()?,
            url: Url::parse(url)?,
            headers,
            next_id: AtomicU64::new(1),
        };
        let id = gateway.next_id.fetch_add(1, Ordering::Relaxed);
        let mut response = gateway
            .client
            .post(gateway.url.clone())
            .headers(gateway.headers.clone())
            .json(
                &json!({"jsonrpc":"2.0","id":id,"method":"initialize","params":{
                    "protocolVersion":PROTOCOL,"capabilities":{},
                    "clientInfo":{"name":"nitpicker","version":env!("CARGO_PKG_VERSION")}
                }}),
            )
            .send()
            .await
            .map_err(|_| eyre!("runtime MCP initialize transport failed"))?;
        if !response.status().is_success() {
            bail!("runtime MCP initialize HTTP {}", response.status());
        }
        if let Some(session) = response.headers().get("mcp-session-id").cloned() {
            gateway.headers.insert("mcp-session-id", session);
        }
        let result = read_response(&mut response, id).await?;
        let version = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .ok_or_else(|| eyre!("runtime MCP initialize omitted protocolVersion"))?;
        if version != PROTOCOL {
            bail!("runtime MCP protocol version unsupported: {version}");
        }
        gateway
            .headers
            .insert("mcp-protocol-version", version.parse()?);
        let response = gateway
            .client
            .post(gateway.url.clone())
            .headers(gateway.headers.clone())
            .json(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
            .send()
            .await
            .map_err(|_| eyre!("runtime MCP initialized notification failed"))?;
        if !response.status().is_success() {
            bail!("runtime MCP initialized HTTP {}", response.status());
        }
        Ok(Arc::new(gateway))
    }

    async fn rpc(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        // No automatic retries: an uncertain remedy must not be executed twice.
        let mut response = self
            .client
            .post(self.url.clone())
            .headers(self.headers.clone())
            .json(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))
            .send()
            .await
            .map_err(|_| eyre!("runtime MCP {method} transport failed; outcome unknown"))?;
        if !response.status().is_success() {
            bail!("runtime MCP {method} HTTP {}", response.status());
        }
        read_response(&mut response, id).await
    }

    async fn discover(self: &Arc<Self>) -> Result<Vec<ToolDefinition>> {
        let mut definitions = HashMap::new();
        let mut cursor = None;
        for _ in 0..32 {
            let params = match &cursor {
                Some(cursor) => json!({"cursor":cursor}),
                None => json!({}),
            };
            let result = self.rpc("tools/list", params).await?;
            let rows = result
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| eyre!("runtime MCP tools/list omitted tools"))?;
            for row in rows {
                let name = row.get("name").and_then(Value::as_str).unwrap_or_default();
                if !is_policy_tool(name) {
                    continue;
                }
                let definition = ToolDefinition {
                    name: name.to_owned(),
                    description: row
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    parameters: row
                        .get("inputSchema")
                        .cloned()
                        .ok_or_else(|| eyre!("MCP tool schema missing"))?,
                };
                if definitions.insert(name.to_owned(), definition).is_some() {
                    bail!("duplicate runtime MCP tool: {name}");
                }
            }
            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if cursor.is_none() {
                for name in POLICY_TOOLS {
                    if !definitions.contains_key(name) {
                        bail!("required runtime MCP policy tool unavailable: {name}");
                    }
                }
                return Ok(definitions.into_values().collect());
            }
        }
        bail!("runtime MCP tool discovery exceeded 32 pages")
    }
}

// The gateway normally returns JSON. Support finite SSE responses as required
// by Streamable HTTP, with a total bound and without buffering an endless stream.
async fn read_response(response: &mut reqwest::Response, id: u64) -> Result<Value> {
    let sse = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| eyre!("runtime MCP response interrupted"))?
    {
        if bytes.len() + chunk.len() > MAX_RESPONSE {
            bail!("runtime MCP response exceeds 2 MiB");
        }
        bytes.extend_from_slice(&chunk);
        if sse {
            if let Some(result) = parse_sse(&bytes, id)? {
                return Ok(result);
            }
        }
    }
    if sse {
        bail!("runtime MCP stream ended without matching response");
    }
    parse_rpc(serde_json::from_slice(&bytes)?, id)?
        .ok_or_else(|| eyre!("runtime MCP response ID mismatch"))
}

fn parse_rpc(value: Value, id: u64) -> Result<Option<Value>> {
    if value.get("id").and_then(Value::as_u64) != Some(id) {
        return Ok(None);
    }
    if let Some(error) = value.get("error") {
        bail!(
            "runtime MCP protocol error {}",
            error.get("code").unwrap_or(&Value::Null)
        );
    }
    Ok(Some(value.get("result").cloned().ok_or_else(|| {
        eyre!("runtime MCP response omitted result")
    })?))
}

fn parse_sse(bytes: &[u8], id: u64) -> Result<Option<Value>> {
    let text = String::from_utf8_lossy(bytes).replace("\r\n", "\n");
    let mut events = text.split("\n\n").collect::<Vec<_>>();
    events.pop(); // never parse a partial event
    for event in events {
        let data = event
            .lines()
            .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() {
            continue;
        }
        if let Some(result) = parse_rpc(serde_json::from_str(&data)?, id)? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

struct GatewayTool {
    gateway: Arc<Gateway>,
    definition: ToolDefinition,
}
impl Tool for GatewayTool {
    fn name(&self) -> String {
        self.definition.name.clone()
    }
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }
    fn call(
        &self,
        args: Value,
        _work_dir: PathBuf,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>> {
        let gateway = Arc::clone(&self.gateway);
        let name = self.name();
        Box::pin(async move {
            let result = gateway
                .rpc("tools/call", json!({"name":name,"arguments":args}))
                .await?;
            // Keep the complete envelope: rulings/approval-required are observations,
            // never interpreted here as consent or as a successful local source read.
            let text = serde_json::to_string(&result)?;
            if result.get("isError").and_then(Value::as_bool) == Some(true) {
                bail!("{text}");
            }
            Ok(text)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn gateway_handshake_preserves_identity_and_returns_approval_without_faking_consent() {
        use std::io::{BufRead, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for step in 0..5 {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
                let mut lines = String::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    lines.push_str(&line);
                }
                let headers = lines.to_ascii_lowercase();
                assert!(headers.contains("authorization: bearer test-token"));
                assert!(headers.contains("x-appa-session-id: workspace"));
                assert!(headers.contains("x-archestra-run-id: run"));
                assert!(headers.contains("x-archestra-runtime-binding: binding"));
                if step > 0 {
                    assert!(headers.contains("mcp-session-id: transport-session"));
                }
                let size: usize = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length: "))
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap();
                let mut bytes = vec![0; size];
                reader.read_exact(&mut bytes).unwrap();
                let body: Value = serde_json::from_slice(&bytes).unwrap();
                let method = body["method"].as_str().unwrap();
                let result = match step {
                    0 => {
                        assert_eq!(method, "initialize");
                        json!({"protocolVersion": PROTOCOL,"capabilities":{"tools":{}},"serverInfo":{"name":"test","version":"1"}})
                    }
                    1 => {
                        assert_eq!(method, "notifications/initialized");
                        write!(stream,"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                        continue;
                    }
                    2 => {
                        assert_eq!(method, "tools/list");
                        json!({"tools":[{"name":POLICY_TOOLS[0],"inputSchema":{"type":"object"}},{"name":"archestra__run_tool","inputSchema":{"type":"object"}}],"nextCursor":"next"})
                    }
                    3 => {
                        assert_eq!(body["params"]["cursor"], "next");
                        json!({"tools":[{"name":POLICY_TOOLS[1],"inputSchema":{"type":"object"}}]})
                    }
                    _ => {
                        assert_eq!(method, "tools/call");
                        assert_eq!(body["params"]["name"], POLICY_TOOLS[1]);
                        assert_eq!(body["params"]["arguments"]["offer_id"], "offer");
                        json!({"isError":true,"content":[{"type":"text","text":"Approval required; remedy has not run"}]})
                    }
                };
                let text = json!({"jsonrpc":"2.0","id":body["id"],"result":result}).to_string();
                write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: transport-session\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",text.len(),text).unwrap();
            }
        });
        let mut headers = HeaderMap::new();
        for (key, value) in [
            ("x-appa-session-id", "workspace"),
            ("x-archestra-run-id", "run"),
            ("x-archestra-runtime-binding", "binding"),
        ] {
            headers.insert(key, value.parse().unwrap());
        }
        let gateway = Gateway::connect(&format!("http://{address}"), "test-token", headers)
            .await
            .unwrap();
        let definitions = gateway.discover().await.unwrap();
        assert_eq!(definitions.len(), 2);
        let definition = definitions
            .into_iter()
            .find(|tool| tool.name == POLICY_TOOLS[1])
            .unwrap();
        let tool = GatewayTool {
            gateway,
            definition,
        };
        let error = tool
            .call(json!({"offer_id":"offer"}), PathBuf::new())
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Approval required; remedy has not run")
        );
        server.join().unwrap();
    }

    #[test]
    fn policy_surface_is_narrow() {
        assert!(is_policy_tool("archestra__execute_remedy_plan"));
        assert!(!is_policy_tool("archestra__run_tool"));
        assert!(!is_policy_tool("archestra__update_guardrails_policy"));
    }
    #[test]
    fn sse_matches_id_and_waits_for_complete_event() {
        let data = b"event: message\r\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\"}\r\n\r\ndata: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"ok\":true}}\r\n\r\n";
        assert_eq!(parse_sse(data, 3).unwrap(), Some(json!({"ok":true})));
        assert_eq!(parse_sse(&data[..data.len() - 2], 3).unwrap(), None);
        assert_eq!(parse_sse(data, 4).unwrap(), None);
    }
    #[test]
    fn rpc_errors_are_not_success() {
        assert!(
            parse_rpc(
                json!({"id":1,"error":{"code":-32000,"message":"secret"}}),
                1
            )
            .unwrap_err()
            .to_string()
            .find("secret")
            .is_none()
        );
        assert!(parse_rpc(json!({"id":1}), 1).is_err());
    }
}
