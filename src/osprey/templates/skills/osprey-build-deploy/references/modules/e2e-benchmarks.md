# Module: benchmarks

End-to-end agent benchmark suite that exercises the deployed assistant by running realistic multi-step queries through the Claude Agent SDK and grading the results with deterministic checks plus an LLM judge. Use it to catch regressions in agent behavior after deploys, after MCP server changes, after profile YAML edits, and after model bumps. The harness lives in `benchmarks/` in the facility profile repo and runs **inside** a web-terminal container — it cannot run on a developer's laptop because it needs the Agent SDK, OSPREY, and every MCP server reachable.

**Enabled when**: `modules.benchmarks.enabled: true` AND `modules.web_terminals.enabled: true` (the suite execs into a web terminal).

## Configuration

```yaml
modules:
  benchmarks:
    enabled: true
    suite_path: "data/benchmarks/e2e_workflow_benchmarks.json"   # path to suite JSON
    runs_in_container: "${facility.prefix}-web-${first_user}"     # which web terminal to exec into
    project_dir: "/app/${facility.prefix}-assistant/"             # path to the built project inside the container
    judge_model: null                                             # null = inherit from llm.model
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `suite_path` | path | yes | Path to the suite JSON file, relative to repo root. |
| `runs_in_container` | string | yes | Container name to `exec` into. Must be one of the `${config.facility.prefix}-web-${user}` containers — typically the first user from `modules.web_terminals.users`. |
| `project_dir` | absolute path | yes | Path to the built assistant project inside the container. The image bakes it at `/app/${config.facility.prefix}-assistant/` by default. |
| `judge_model` | string or null | no | Model used by the LLM judge. `null` = inherit from `${config.llm.model}`. Override only when the agent and the judge should use different models (sometimes useful for cost). |

## Why it runs inside the container

The benchmark harness depends on three things that don't all exist on a developer's laptop:

1. **Claude Agent SDK** (`claude_agent_sdk`) — installed in the web terminal image, not pip-publishable as a normal package.
2. **OSPREY** with the facility's profile already built and rendered — the SDK needs `project_dir` to be a real Claude Code project.
3. **Every MCP server reachable** by Docker DNS — the agent's tool calls go to the live HTTP MCP servers on the deploy network.

A web terminal container has all three. It also runs with `bypassPermissions` for the SDK, which is only safe in a sandboxed container — never on a real developer machine where unintended writes could touch real files.

## What scaffolding adds when this module is enabled

- compose: nothing new — uses the existing web terminal container. Adds a bind mount of `benchmarks/` into the container if it isn't already overlaid by the web-terminal Dockerfile.
- .gitlab-ci.yml: optionally a `benchmarks` job in a `nightly` stage that triggers via scheduled pipeline. Off by default — most facilities run benchmarks on demand, not every push.
- scripts/deploy.sh: nothing — benchmarks are operator-driven, not part of the deploy pipeline.
- scripts/verify.sh: nothing — verify is fast and advisory; benchmarks are slow and expensive.
- .env.template: nothing — uses the same LLM credentials as the rest of the deploy.

## Running benchmarks

### Full suite

```bash
ssh ${config.deploy.host} "${config.runtime.engine} exec ${config.modules.benchmarks.runs_in_container} \
  python -m benchmarks --suite ${config.modules.benchmarks.suite_path} --output results/"
```

Results land in `results/` inside the container — copy out with `${config.runtime.engine} cp` if needed.

### Specific queries by index

```bash
ssh ${config.deploy.host} "${config.runtime.engine} exec ${config.modules.benchmarks.runs_in_container} \
  python -m benchmarks --suite ${config.modules.benchmarks.suite_path} --indices 0,2,5"
```

### Custom model (e.g., a cheaper judge)

```bash
ssh ${config.deploy.host} "${config.runtime.engine} exec ${config.modules.benchmarks.runs_in_container} \
  python -m benchmarks --suite ${config.modules.benchmarks.suite_path} --model anthropic/claude-haiku"
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--suite` | (required) | Path to the suite JSON inside the container |
| `--model` | `${config.modules.benchmarks.judge_model}` or `${config.llm.model}` | Model for agent execution AND for the judge unless overridden |
| `--output` | `results/` | Output directory (per-run subdirectory created) |
| `--project-dir` | `${config.modules.benchmarks.project_dir}` | Path to the built Claude Code project |
| `--indices` | all | Comma-separated query indices to run (subset) |
| `--iterations` | 1 | Run each query N times (for flakey-query stability assessment) |

## Per-query pipeline

Each query in the suite executes the same four-stage pipeline:

1. **Pre-flight gating** (optional, per query): HTTP GET to the integration-tests `/checks` endpoint with the categories the query depends on. If any required check is in `ERROR`, the query is marked `SKIPPED` and the harness moves on. This catches "infrastructure was down so the query couldn't possibly work" and prevents false-failure noise.
2. **SDK execution**: the Claude Agent SDK (`claude_agent_sdk.query()`) drives the built project at `${config.modules.benchmarks.project_dir}` with `bypassPermissions=True`. The SDK records every tool call, every result, every text emission into a trace.
3. **Deterministic checks** against the trace:
   - `required_tools`: substring match on tool names that must appear in the trace.
   - `required_tool_success`: tools that must be called AND have at least one non-error invocation (tolerates retries).
   - `expected_artifacts`: tool results must contain `artifact_id` for the named tools (confirms the query actually produced something durable, not just chat).
4. **LLM judge**: the judge model receives the user query, the assistant's final response, and the `judge_expectations` text. It returns a structured verdict (PASS/FAIL with reasoning).

A query passes only if pre-flight didn't skip it AND deterministic checks all pass AND the judge returns PASS.

## Authoring queries

Each query in `${config.modules.benchmarks.suite_path}` is a JSON object:

```json
{
  "id": "archiver_dcct_overnight",
  "category": "archiver_analysis",
  "user_query": "What was the average beam current overnight?",
  "required_tools": ["mcp__archiver__"],
  "required_tool_success": ["mcp__archiver__get_data"],
  "expected_artifacts": ["mcp__archiver__get_data"],
  "preflight": {
    "categories": ["services", "epics_read"],
    "required_checks": ["services.archiver", "epics.beam_current"]
  },
  "judge_expectations": "Response should report a numeric average in mA over the requested time window, with units, and reference the actual archiver data fetched."
}
```

Field reference:

| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | Unique identifier, used in result filenames |
| `category` | yes | Free-form grouping label |
| `user_query` | yes | The prompt sent to the agent |
| `required_tools` | no | Substring matches against tool names; ALL must appear |
| `required_tool_success` | no | Tools that must succeed at least once (retries tolerated) |
| `expected_artifacts` | no | Tool names whose results must contain `artifact_id` |
| `preflight` | no | Infrastructure gating: `categories` (which integration-test categories to run) + `required_checks` (specific check names that must be OK) |
| `judge_expectations` | no | Plain text describing what the LLM judge should look for |

## Local unit tests

The evaluation logic itself (preflight HTTP calls, deterministic checks, judge invocation) has unit tests under `tests/benchmarks/`. These are mocked — they don't call the SDK or the judge — so they run on the developer's laptop without any deploy infrastructure:

```bash
pytest tests/benchmarks/ -v
```

Use these to validate that benchmark code changes are sound before running an end-to-end benchmark, which is slow and expensive.

## Result format

Each run writes a JSON file per query under `--output`, plus a top-level summary:

```json
{
  "run_id": "2025-04-22T18:30:00Z",
  "model": "anthropic/claude-sonnet-4-20250514",
  "judge_model": "anthropic/claude-sonnet-4-20250514",
  "queries_total": 12,
  "queries_passed": 9,
  "queries_failed": 2,
  "queries_skipped": 1,
  "per_query": [...]
}
```

Each `per_query` entry includes the trace, the deterministic check verdicts, and the full judge transcript (system prompt + judge response). Read the judge transcript for any FAIL — that's where the "why" lives.

## Operating the module

### When to add benchmark queries

After every new agent capability lands. If you add a new MCP tool, add at least one query that exercises a workflow involving that tool. If you change an agent's system prompt or its allowed tool list, add a query that catches the regression you'd most fear.

### Interpreting a failed judge call

The judge response is the source of truth. Read it. Common patterns:

- **Judge says the answer is correct but `required_tools` failed**: the agent solved the query without using the expected tool (maybe via a shortcut or memorized knowledge). Either widen the test (acceptable behavior), or tighten the user query so the tool is the only path.
- **Judge says the answer is wrong but deterministic checks all passed**: the agent went through the motions but produced a bad answer. Read the SDK trace — usually the tool returned bad data, or the agent misinterpreted the result.
- **Judge gives an ambiguous verdict**: the `judge_expectations` text is too vague. Sharpen it.

### Handling flakey queries

Queries that depend on live data (current beam state, real-time PV values) can flap PASS/FAIL across runs. Two responses:

1. Use `--iterations 5` and record pass-rate. If pass-rate is < 80%, treat the query as broken and rewrite it to be more deterministic (e.g., replace "what's the beam current right now" with "fetch beam current from <fixed timestamp>").
2. Add a `flakey: true` field (informally) and exclude these from CI runs; only run them manually for debugging.

## Cost notes

Each query consumes LLM tokens twice: once for the agent's multi-step execution, once for the judge's evaluation. For a suite of N queries running M iterations:

```
total_tokens ≈ N × M × (agent_tokens_per_query + judge_tokens_per_query)
```

Agent token usage is highly variable — a tool-heavy multi-step query can run 10–50× the tokens of a simple lookup. Judge token usage is more stable (a few thousand input + a few hundred output per call).

For ALS-scale suites (~10–20 queries, mostly run on demand), benchmark cost is a rounding error compared to interactive use. For larger suites or scheduled runs, budget explicitly:

- Estimate per-query cost from a single-iteration run: `(input_tokens × input_price + output_tokens × output_price)` from the SDK's per-call usage report.
- Multiply by N × M for the planned run.
- Compare against the facility's monthly LLM budget.
- Consider using a cheaper `judge_model` (e.g., Haiku) — judging is shorter-context than agent execution, so a smaller model is often sufficient.

## Common failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: claude_agent_sdk` | Trying to run benchmarks outside the web terminal container | Use `${config.runtime.engine} exec` into a web terminal |
| All queries fail at pre-flight | `integration-tests` MCP server is down | Restart it; verify with `verify.sh` |
| Judge always returns FAIL with vague reasoning | `judge_expectations` is too strict or too vague | Rewrite expectations as a checklist of observable properties |
| `bypassPermissions` errors despite being set | SDK version mismatch with Claude Code | Update Claude Code in the web terminal image |
| Suite runs but `results/` is empty | `--output` path not writable inside container | Use a path under `/tmp/` or a bind-mounted volume |
| One query consistently fails on a server that's healthy | MCP tool the query expects has been renamed/removed | Update the query's `required_tools` substring, or restore the tool |
| Cost much higher than expected | Agent is hitting tool retry loops | Inspect the SDK trace; tighten the query or add a `max_turns` setting |

## Cross-references

- If `modules.web_terminals.enabled` is false, this module cannot run — disable it or enable web terminals first.
- If `modules.integration_tests` is the always-on integration tests module, benchmark `preflight` blocks reference its check categories. Keep the category names in sync if the integration tests module evolves.
- If `modules.event_dispatcher.enabled` is true, you can wire a benchmark run as an EPICS-CA-triggered or scheduled webhook (e.g., nightly). The benchmark CLI is plain enough to invoke from any orchestration.

## Disabling

Set `modules.benchmarks.enabled: false` (or remove the block) and re-scaffold. Then:

- The web terminal containers no longer get the benchmarks bind mount.
- Any nightly CI job (if you opted into it) is removed.
- The suite JSON file (`${config.modules.benchmarks.suite_path}`) and the `benchmarks/` package stay in the repo — delete them manually if you really want them gone.
- No data needs to be cleaned up; results land in container-local `results/` directories that vanish with the container.
