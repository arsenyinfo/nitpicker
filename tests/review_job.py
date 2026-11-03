"""Subprocess contract tests against a synthetic OpenAI transport, with real Git/file tools."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BINARY = os.environ.get("NITPICKER_TEST_BIN", "target/debug/nitpicker")


class ReviewJobTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        (self.repo / "AGENTS.md").write_text("TRUSTED_BASE_MARKER")
        (self.repo / "src").mkdir()
        (self.repo / "src/lib.rs").write_text("pub fn value() -> u32 { 1 }\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD")
        (self.repo / "AGENTS.md").write_text("UNTRUSTED_HEAD_MARKER")
        (self.repo / "src/lib.rs").write_text("pub fn value() -> u32 { 2 }\n")
        self.git("add", ".")
        self.git("commit", "-qm", "change")
        self.head = self.git("rev-parse", "HEAD")
        self.request = self.root / "request.json"
        self.request.write_text(json.dumps({"schema_version": 1, "repository": "example/repo", "base_sha": self.base, "head_sha": self.head}))
        self.calls = []
        self.mcp_calls = []
        self.recover_block = False
        self.missing_policy_tools = False
        self.result = {"summary": "Reviewed", "findings": [], "previous": [], "reviewed": ["src/lib.rs"], "limitations": "Synthetic transport"}
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/mcp":
                    test.mcp_calls.append((data, dict(self.headers)))
                    method = data["method"]
                    if method == "notifications/initialized":
                        self.send_response(202)
                        self.end_headers()
                        return
                    if method == "initialize":
                        result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}}
                    elif method == "tools/list":
                        result = {"tools": [] if test.missing_policy_tools else [{"name": name, "description": "Policy fixture", "inputSchema": {"type": "object", "properties": {}}} for name in ["archestra__get_remedy_plans", "archestra__execute_remedy_plan"]]}
                    elif method == "tools/call":
                        result = {"content": [{"type": "text", "text": "Fixture policy observation; retry the read_file call"}]}
                    else:
                        raise AssertionError(method)
                    body = json.dumps({"jsonrpc": "2.0", "id": data["id"], "result": result}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                test.calls.append((data, dict(self.headers)))
                if self.path.endswith("/responses"):
                    if not data.get("tools"):
                        item = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": json.dumps(test.result)}]}
                    elif not any(x.get("type") == "function_call_output" for x in data["input"]):
                        item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": json.dumps({"path": "src/lib.rs"}), "status": "completed"}
                    else:
                        item = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "No findings. Great job! 🎉\n## Coverage\nsrc/lib.rs inspected"}]}
                    response = {"id": "resp_1", "object": "response", "created_at": 1, "status": "completed", "model": data["model"], "output": [item], "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}}
                    body = ("data: " + json.dumps({"type": "response.completed", "response": response}) + "\n\n").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if data["model"] == "fail":
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"synthetic invalid key","type":"authentication_error"}}')
                    return
                tools = {tool["function"]["name"] for tool in data.get("tools", [])}
                if not tools:
                    message = {"role": "assistant", "content": json.dumps(test.result)}
                    finish = "stop"
                elif test.recover_block and not any(m["role"] == "tool" for m in data["messages"]):
                    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "policy-1", "type": "function", "function": {"name": "archestra__get_remedy_plans", "arguments": "{}"}}]}
                    finish = "tool_calls"
                elif not any(m["role"] == "tool" and m.get("tool_call_id") == "read-1" for m in data["messages"]):
                    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "read-1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "src/lib.rs"})}}]}
                    finish = "tool_calls"
                elif "submit_verdict" in tools:
                    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "verdict-1", "type": "function", "function": {"name": "submit_verdict", "arguments": json.dumps({"verdict": "No findings. Great job! 🎉\n## Coverage\nsrc/lib.rs inspected", "agree": True})}}]}
                    finish = "tool_calls"
                else:
                    message = {"role": "assistant", "content": "No findings. Great job! 🎉\n## Coverage\nsrc/lib.rs inspected"}
                    finish = "stop"
                response = {"id": "fixture", "object": "chat.completion", "created": 1, "model": data["model"], "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
                body = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()

    def invoke(self, second="good", debate=False, managed=False):
        config = self.root / "config.toml"
        auth_config = 'auth = "codex"\n' if managed else ''
        config.write_text(f'''[defaults]
debate = {str(debate).lower()}
log_trajectories = false
presets = ["correctness"]
[aggregator]
model = "synthesis"
provider = "openai"
{auth_config}\
base_url = "{self.url}"
api_key_env = "NITPICKER_TEST_KEY"
[[reviewer]]
name = "first"
model = "good"
provider = "openai"
{auth_config}\
base_url = "{self.url}"
api_key_env = "NITPICKER_TEST_KEY"
[[reviewer]]
name = "second"
model = "{second}"
provider = "openai"
{auth_config}\
base_url = "{self.url}"
api_key_env = "NITPICKER_TEST_KEY"
''')
        env = {**os.environ, "NITPICKER_TEST_KEY": "synthetic", "OTEL_SDK_DISABLED": "true", "ARCHESTRA_MCP_GATEWAY_URL": self.url + "/mcp", "ARCHESTRA_MCP_GATEWAY_TOKEN": "mcp-synthetic", "ARCHESTRA_LLM_PROXY_URL": self.url + "/v1/openai/fixture", "ARCHESTRA_AGENT_RUNTIME_BINDING": "synthetic-binding", "ARCHESTRA_AGENT_RUNTIME_TASK_ID": "fixture-task", "ARCHESTRA_AGENT_RUNTIME_WORKSPACE_ID": "fixture-session"}
        return subprocess.run([BINARY, "--repo", str(self.repo), "--config", str(config), "--request", str(self.request), "--rounds", "1", "--max-turns", "4"], env=env, capture_output=True, text=True, timeout=20)

    def test_runtime_policy_notice_is_dispatched_before_source_read(self):
        self.recover_block = True
        process = self.invoke(debate=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertFalse(json.loads(process.stdout)["_nitpicker"]["degraded"])
        controls = [(body, headers) for body, headers in self.mcp_calls if body["method"] == "tools/call"]
        self.assertTrue(controls)
        for _, headers in self.mcp_calls:
            headers = {key.lower(): value for key, value in headers.items()}
            self.assertEqual(headers["authorization"], "Bearer mcp-synthetic")
            self.assertEqual(headers["x-appa-session-id"], "fixture-session")
            self.assertEqual(headers["x-archestra-runtime-binding"], "synthetic-binding")
        wire = json.dumps(self.calls)
        self.assertIn("pub fn value()", wire)
        self.assertNotIn("mcp-synthetic", wire)

    def test_missing_runtime_policy_tools_fails_before_model_calls(self):
        self.missing_policy_tools = True
        process = self.invoke()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(self.calls)
        self.assertIn("required runtime MCP policy tool unavailable", process.stderr)

    def test_parallel_json_uses_real_file_tools_and_pinned_base_instructions(self):
        process = self.invoke()
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["_nitpicker"]["head_sha"], self.head)
        self.assertFalse(result["_nitpicker"]["degraded"])
        self.assertEqual(result["findings"], [])
        self.assertEqual(self.git("status", "--porcelain"), "")
        wire = json.dumps([c[0] for c in self.calls])
        self.assertIn("TRUSTED_BASE_MARKER", wire)
        self.assertNotIn("UNTRUSTED_HEAD_MARKER", wire)
        self.assertTrue(any(m.get("role") == "tool" and "pub fn value" in json.dumps(m) for c, _ in self.calls for m in c["messages"]))
        for _, headers in self.calls:
            normalized = {k.lower(): v for k, v in headers.items()}
            self.assertEqual(normalized.get("x-archestra-runtime-binding"), "synthetic-binding")

    def test_debate_emits_the_same_result_contract(self):
        process = self.invoke(debate=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["summary"], "Reviewed")

    def test_managed_codex_uses_responses_and_replays_tool_results(self):
        process = self.invoke(managed=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["summary"], "Reviewed")
        self.assertTrue(any(x.get("type") == "function_call_output" for data, _ in self.calls for x in data["input"]))
        for data, headers in self.calls:
            self.assertTrue(data["stream"])
            self.assertFalse(data["store"])
            self.assertNotIn("max_output_tokens", data)
            self.assertEqual({k.lower(): v for k, v in headers.items()}["x-archestra-runtime-binding"], "synthetic-binding")

    def test_degraded_synthesis_stays_json_and_returns_three(self):
        process = self.invoke(second="fail")
        self.assertEqual(process.returncode, 3, process.stderr)
        result = json.loads(process.stdout)
        self.assertTrue(result["_nitpicker"]["degraded"])
        self.assertEqual(result["_nitpicker"]["coverage"][0]["succeeded"], 1)

    def test_malformed_synthesis_cannot_become_a_successful_empty_review(self):
        self.result = {"summary": "Missing findings"}
        process = self.invoke()
        self.assertEqual(process.returncode, 1)
        self.assertEqual(process.stdout, "")

    def test_stale_head_fails_before_model_access(self):
        data = json.loads(self.request.read_text())
        data["head_sha"] = self.base
        self.request.write_text(json.dumps(data))
        process = self.invoke()
        self.assertEqual(process.returncode, 1)
        self.assertEqual(self.calls, [])

    def test_previous_fixed_status_and_identity_reach_the_result(self):
        finding = {"file": "src/lib.rs", "line": 1, "severity": "medium", "summary": "Regression", "failure_scenario": "Bad input fails", "recommendation": "Restore validation"}
        data = json.loads(self.request.read_text())
        data["previous_findings"] = [{**finding, "id": "old-1", "status": "fixed"}]
        self.request.write_text(json.dumps(data))
        self.result = {"summary": "Regressed", "findings": [{**finding, "previous_id": "old-1"}], "previous": [{"id": "old-1", "status": "regressed"}]}
        process = self.invoke()
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["findings"][0]["previous_id"], "old-1")
        self.assertEqual(result["previous"][0]["status"], "regressed")
        wire = "\n".join(str(m.get("content", "")) for c, _ in self.calls for m in c["messages"])
        self.assertIn('"status":"fixed"', wire)
        self.assertIn("Automation finding lifecycle", wire)


if __name__ == "__main__":
    unittest.main()
