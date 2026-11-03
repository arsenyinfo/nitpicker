# Pinned automation reviews

`--request PATH` runs a read-only review of a fixed checkout and emits one JSON
object compatible with the Archestra code-review Action. The caller prepares
the checkout and owns GitHub publication. Nitpicker performs review/debate and
its existing final synthesis directly in JSON; no Markdown parser or additional
model call translates the result.

```sh
nitpicker --repo /path/to/checkout --config /trusted/nitpicker.toml \
  --request /path/to/request.json > result.json
```

This mode requires an explicit config and a clean checkout whose HEAD equals
`head_sha`. It works with `--no-debate`, `--preset`, `--fallback`, `--alloy`,
`--rounds`, and `--max-turns` using their normal meanings. It cannot be combined
with a subcommand, `--analyze`, `--prompt`, or `--context-file`.

The version-1 request is:

```json
{
  "schema_version": 1,
  "repository": "owner/repository",
  "base_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "head_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "previous_findings": [],
  "instructions": "Additional trusted review instructions",
  "brief": "Caller scope, including what changed since the previous review",
  "checks": []
}
```

The last four fields are optional. Previous findings use the result finding
fields below plus their durable `id`. The input file uses the existing bounded
context-file reader (regular file, valid UTF-8, at most 256 KiB including framing).
Repository instructions are read from `base_sha`: root AGENTS.md/CLAUDE.md and
instructions in directories covering changed files. The checked-out head's
instructions and config are not auto-loaded. Review scope is the merge-base of
the fixed base/head to head; re-review summaries follow the caller's brief.

The output contains `summary`, `findings`, `previous`, `reviewed`, `limitations`,
and additive `_nitpicker` diagnostics (schema version, base/head SHA, degraded,
execution coverage, usage). Each finding includes:

- `file`, `line`, optional `end_line`: repository-relative head location;
- `severity`: critical, high, medium, low, or nit;
- `summary`, `failure_scenario`, `recommendation`, optional `title`/`suggestion`;
- optional `previous_id`: the supplied identity of a persistent/regressed issue.

Each supplied previous ID must have one disposition in `previous`: `fixed`,
`still_present`, or `regressed`. Fixed requires evidence in `note` and must not
remain in findings; the other dispositions require a finding with that
`previous_id`. The Action assigns identities to new findings and manages threads.
Result types, field lengths, reference integrity, and line ranges are validated
before anything is emitted. Invalid/truncated synthesis is a hard failure.

Stdout is the single JSON object on success, including degraded success.
Progress and errors go to stderr. Exit codes remain 0 complete, 3 degraded
with a valid result, and 1 hard failure without a result; clap usage errors use
2. A runtime may accept 3 if it has a valid result. Findings themselves do not
change the process exit code. Use the findings' severities to derive a verdict.

## Managed Codex model access

Existing `provider = "openai"`, `auth = "codex"` with no endpoint/key override
continues to use the local Codex-format OAuth store, with no CLI subprocess.
Supply **both** `base_url` and `api_key_env` to use the same Codex Responses
protocol through a managed proxy instead:

```toml
[[reviewer]]
name = "codex"
provider = "openai"
auth = "codex"
model = "openai:gpt-6.1-sol"
base_url = "https://backend.archestra.dev/v1/model-router"
api_key_env = "ARCHESTRA_VIRTUAL_KEY"
```

This path never reads or refreshes local OAuth credentials. The gateway owns
account selection and refresh. It uses the existing Responses SSE/tool-history
handling, `store: false`, and encrypted reasoning replay. The endpoint must use
HTTPS, loopback HTTP, or the exact origin injected in `ARCHESTRA_LLM_PROXY_URL`
(which can be HTTP inside a cluster). Other HTTP endpoints are rejected. A
partial endpoint/key configuration is rejected.

When running inside Archestra, the OpenAI, Anthropic, Gemini, and managed Codex
clients forward injected run/session/binding headers only to the same parsed
origin as `ARCHESTRA_LLM_PROXY_URL`; those credentials are excluded from direct
third-party provider requests. Custom configs must select routes accessible to
their actual key. A ChatGPT subscription requires the user's own permitted
provider mapping; an API-only virtual key does not imply subscription access.

## Verification

```sh
cargo test --workspace
cargo build
NITPICKER_TEST_BIN="$PWD/target/debug/nitpicker" python3 tests/review_job.py -v
```

## Runtime policy channel and Kimi terminal calls

In an Archestra runtime, Nitpicker initializes the injected MCP gateway before
agent turns and exposes only `archestra__get_remedy_plans` and
`archestra__execute_remedy_plan`. Both channels use the same runtime/session
identity and binding. Discovery failure aborts the agent; policy refusals and
approval requirements remain tool observations. The bridge neither grants consent
nor imports general gateway tools. It uses bounded Streamable HTTP JSON/SSE,
disables redirects and does not retry remedy calls. Concurrent reviewers share the
runtime policy session, just as their LLM requests do; no synthetic child identities
or policy resets are created. The isolated live test currently exercises one
sequential debate lane; concurrent policy recovery still needs live validation.

Kimi named terminal calls use `required` with the permitted tool subset because
K3.1 thinking rejects a specific function choice. The harness still requires the
terminal tool to run successfully; plain text cannot substitute for a verdict.
Runtime terminal turns keep the policy controls available for proxy recovery.
