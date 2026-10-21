"""Tier 3 agentic per-preset smoke tests.

One canonical workflow per preset. Each test:
  1. Builds a deployment repo from the preset (with --skip-deps for speed;
     Tier 1 covers the real-build path).
  2. Asks an agent (via the Claude Agent SDK) to perform the canonical task.
  3. Asserts the right MCP tool appears in the tool trace (the contract that
     would catch a "preset config drift means the agent never reaches for
     the documented tool" failure).
  4. Optionally has the LLM judge evaluate the response.

Budget per test: max_turns=4, max_budget_usd=0.25, target wall-clock <5 min.
Gating in CI (agentic-per-preset blocks the merge like every other e2e lane);
coverage of the "is the tool wired up at all?" contract comes from Tier 0+1.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import yaml

from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_credentials as creds
from tests.e2e._facility_select import bounded_setpoint, repo_view, seeded_readback
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.judge import LLMJudge, WorkflowResult
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    SDKWorkflowResult,
    enable_writes_in_project,
    init_project,
    is_claude_code_available,
    judge_result_preview,
    render_dir,
    run_sdk_query,
    run_sdk_query_with_hooks,
)

# ALS_APG_API_KEY is enforced via `requires_als_apg` — the root
# `tests/conftest.py` hook auto-skips when the key is missing. Every test
# in this file passes `provider="als-apg"` to `init_project` and `LLMJudge`.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available"),
]


def _to_workflow_result(query: str, sdk_result: SDKWorkflowResult) -> WorkflowResult:
    """Convert ``SDKWorkflowResult`` (rich tool traces) to the plain-text shape
    the LLM judge expects."""
    response = "\n".join(sdk_result.text_blocks).strip()
    trace_lines: list[str] = []
    for t in sdk_result.tool_traces:
        trace_lines.append(f"TOOL: {t.name}  input={t.input}")
        if t.result:
            preview = judge_result_preview(t.result)
            trace_lines.append(f"  result: {preview}")
    return WorkflowResult(
        query=query,
        response=response,
        execution_trace="\n".join(trace_lines),
        artifacts=[],
    )


def _channel_finder_server_name(repo: Path) -> str | None:
    """Return the channel-finder backend module short name (or None if absent).

    Takes the repo root and reads the render's ``.mcp.json`` — the file the
    agent session loads its MCP servers from.
    """
    cfg = json.loads((render_dir(repo) / ".mcp.json").read_text(encoding="utf-8"))
    for name, entry in cfg.get("mcpServers", {}).items():
        if "channel-finder" in name or "channel_finder" in name:
            args = entry.get("args") or []
            for arg in args:
                if isinstance(arg, str) and "channel_finder_" in arg:
                    return name
            return name
    return None


async def _assert_approval_hook_fires(
    repo: Path,
    query: str,
    expected_write_tool: str,
    *,
    max_turns: int = 6,
    max_budget_usd: float = 0.30,
) -> None:
    """Run a write-style query and verify the approval hook recorded an event.

    Uses ``run_sdk_query_with_hooks`` with ``permission_mode="default"`` and
    ``approval_policy="auto_approve"`` so hooks fire (vs. ``run_sdk_query``
    which uses ``bypassPermissions`` and silently elides them). Asserts:

    - at least one ``HookEvent`` was recorded (otherwise hooks are absent)
    - one of those events corresponds to ``expected_write_tool``
    - decision was ``"allow"`` under the auto-approve policy
    """
    result = await run_sdk_query_with_hooks(
        repo,
        query,
        approval_policy="auto_approve",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )
    assert result.hook_events, (
        "no hook events recorded — approval hook did not fire on a write query. "
        "Either the agent never reached for a write tool, or hooks are not "
        "wired into the render's .claude/settings.json."
    )
    matching = [e for e in result.hook_events if e.tool_name == expected_write_tool]
    assert matching, (
        f"approval hook fired but not for {expected_write_tool}. "
        f"Recorded: {[(e.tool_name, e.decision) for e in result.hook_events]}"
    )
    assert all(e.decision == "allow" for e in matching), (
        f"auto_approve policy should allow all writes, got: "
        f"{[(e.tool_name, e.decision, e.reason) for e in matching]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.harness_benchmark
@pytest.mark.asyncio
async def test_hello_world_canonical_flow(tmp_path: Path) -> None:
    """Hello-world preset: agent reads a seeded facility channel.

    Asserts the agent reached for ``mcp__controls__channel_read``. If the
    preset wiring breaks, this fires within 5 min instead of the nested
    Claude Code session hanging on missing tools.
    """
    repo = init_project(tmp_path, "hello_demo", template="hello_world", provider="als-apg")
    judge = LLMJudge(provider="als-apg")
    channel = seeded_readback(repo_view(repo))
    query = (
        f"Use the controls MCP server to read the channel named '{channel}' "
        "and report its current value."
    )
    result = await run_sdk_query(repo, query, max_turns=4, max_budget_usd=0.25)

    assert any(t.name == "mcp__controls__channel_read" for t in result.tool_traces), (
        f"agent did not call mcp__controls__channel_read. Tools called: {result.tool_names}"
    )

    eval = await judge.evaluate(
        _to_workflow_result(query, result),
        expectations=(
            "The agent reads a control-system channel using the controls MCP "
            "server and reports a numeric value back to the user. The response "
            "should not contain unhandled errors."
        ),
    )
    assert eval.passed, eval.reasoning


@pytest.mark.agentic_benchmark
@pytest.mark.asyncio
async def test_control_assistant_channel_finder_flow(tmp_path: Path) -> None:
    """Control-assistant preset: agent uses the channel-finder pipeline.

    Soft assertion (the channel-finder backend differs by preset config) —
    we only require that *some* ``mcp__channel-finder__*`` tool was used.
    """
    repo = init_project(tmp_path, "ca_demo", template="control_assistant", provider="als-apg")
    cf_server = _channel_finder_server_name(repo)
    if cf_server is None:
        pytest.skip("control-assistant preset has no channel-finder server")

    judge = LLMJudge(provider="als-apg")
    query = "Help me find the address of a beam-position-monitor channel for sector 5."
    result = await run_sdk_query(repo, query, max_turns=4, max_budget_usd=0.25)

    cf_tool_calls = [t for t in result.tool_traces if t.name.startswith("mcp__channel-finder__")]
    assert cf_tool_calls, (
        f"agent did not call any mcp__channel-finder__* tool. Tools called: {result.tool_names}"
    )

    eval = await judge.evaluate(
        _to_workflow_result(query, result),
        expectations=(
            "The agent uses the channel-finder MCP server to search for or "
            "narrow down beam-position-monitor channels. It should engage "
            "with the channel-finder pipeline and report progress back to "
            "the user — even if no exact match is found."
        ),
    )
    assert eval.passed, eval.reasoning


@pytest.mark.harness_benchmark
@pytest.mark.asyncio
async def test_hello_world_write_triggers_approval_hook(tmp_path: Path) -> None:
    """Hello-world write probe: agent sets a writable facility setpoint; approval hook fires.

    Flips ``writes_enabled: true`` so the writes-disabled kill switch
    (``osprey_writes_check.py``, safety_layer 1) doesn't deny before the
    approval hook (safety_layer 2) gets to return ``ask`` — which is what
    triggers the SDK's ``can_use_tool`` callback.
    """
    repo = init_project(tmp_path, "hw_write", template="hello_world", provider="als-apg")
    enable_writes_in_project(repo)
    channel, value = bounded_setpoint(repo_view(repo))
    await _assert_approval_hook_fires(
        repo,
        # Directive prompt: explicit tool name so the agent commits to the write
        # path on its first turn instead of dithering with reads or refusals.
        # The channel holds a limits record whose band contains the value, so the
        # limits hook (ahead of approval in the PreToolUse chain) passes it on.
        f"Use the mcp__controls__channel_write tool to write the value {value} "
        f"to the channel '{channel}'.",
        expected_write_tool="mcp__controls__channel_write",
    )


@pytest.mark.harness_benchmark
@pytest.mark.asyncio
async def test_control_assistant_write_triggers_approval_hook(tmp_path: Path) -> None:
    """Control-assistant write probe: same contract on the heavier preset."""
    repo = init_project(tmp_path, "ca_write", template="control_assistant", provider="als-apg")
    enable_writes_in_project(repo)
    channel, value = bounded_setpoint(repo_view(repo))
    await _assert_approval_hook_fires(
        repo,
        # Directive prompt: explicit tool name + concrete channel address so the
        # agent doesn't burn turns on channel-finder navigation. The channel MUST
        # be writable under the build's limits view: the limits hook sits AHEAD
        # of the approval hook in the PreToolUse chain, and its deny on a
        # read-only channel short-circuits the chain before approval ever runs
        # (correct safety behavior — but then this test observes no hook events).
        # The channel is a bounded writable setpoint and the value lies inside its
        # band, so the limits hook passes it on.
        f"Use the mcp__controls__channel_write tool to write the value {value} to "
        f"the channel '{channel}'.",
        expected_write_tool="mcp__controls__channel_write",
    )


# ---------------------------------------------------------------------------
# The hello-world tutorial's query, through a deployed store and gateway
# ---------------------------------------------------------------------------

#: The compose project (and repo directory) the tutorial test deploys under.
#: The `agentic-per-preset` lane's `if: always()` cleanup step removes this
#: project's containers and volumes by these exact names.
TUTORIAL_PROJECT_NAME = "osprey-e2e-hello-world-tutorial"
TUTORIAL_STORE_CONTAINER = f"{TUTORIAL_PROJECT_NAME}-telemetry-store"
TUTORIAL_GATEWAY_CONTAINER = f"{TUTORIAL_PROJECT_NAME}-telemetry-gateway"

#: The SQL the hello-world tutorial asks its reader to run, kept in a file of
#: its own so the page includes exactly what this test runs.
TUTORIAL_SQL = (
    Path(__file__).resolve().parents[2] / "docs/source/getting-started/hello-world-tutorial.sql"
)

#: The reader the hello-world preset declares.
TUTORIAL_READER = "operator"

_TUTORIAL_VERB_TIMEOUT_S = 600
_TUTORIAL_READY_TIMEOUT_S = 180.0
_TUTORIAL_ROW_TIMEOUT_S = 180.0

#: Variables a developer shell or an earlier test may carry that would make
#: `osprey chat` treat itself as a container, or file under another identity.
_TUTORIAL_SCRUBBED_ENV = (
    "CLAUDECODE",
    tc.GATEWAY_HOST_ENV,
    tc.GATEWAY_PORT_ENV,
    creds.LOGIN_ENV,
    creds.AUDIT_IDENTITY_ENV,
    "OSPREY_TERMINAL_SESSION",
    "OSPREY_TERMINAL_USER",
)


def _tutorial_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _TUTORIAL_SCRUBBED_ENV}


def _tutorial_verb(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run one `osprey` verb against *repo*; fail with both streams on a non-zero exit."""
    result = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=_TUTORIAL_VERB_TIMEOUT_S,
        env=_tutorial_env(),
        check=False,
    )
    assert result.returncode == 0, (
        f"osprey {' '.join(args)} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result


def _remove_tutorial_containers() -> None:
    for name in (TUTORIAL_GATEWAY_CONTAINER, TUTORIAL_STORE_CONTAINER):
        subprocess.run(
            ["docker", "rm", "-f", name], capture_output=True, text=True, timeout=120, check=False
        )


def _reader_password(repo: Path, reader: str) -> str:
    for line in (repo / creds.reader_file(reader)).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == creds.READER_KEY:
            return value.strip()
    raise AssertionError(f"{creds.reader_file(reader)} holds no {creds.READER_KEY}")


def _query_as_reader(store_port: int, password: str, sql: str) -> list[dict]:
    """POST *sql* to the loopback store as the tutorial's reader; rows as dicts."""
    response = httpx.post(
        f"http://127.0.0.1:{store_port}/",
        content=f"{sql.rstrip().rstrip(';')}\nFORMAT JSONEachRow",
        auth=(TUTORIAL_READER, password),
        timeout=30,
    )
    assert response.status_code == 200, (
        f"the store refused the reader's query (HTTP {response.status_code}): {response.text}"
    )
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _wait_tutorial_stack(config: dict, password: str) -> None:
    """Wait until the store answers the reader and the gateway answers its counters page."""
    deadline = time.monotonic() + _TUTORIAL_READY_TIMEOUT_S
    last = ""
    while time.monotonic() < deadline:
        try:
            if _query_as_reader(tc.store_port(config), password, "SELECT 1 AS one") == [{"one": 1}]:
                httpx.get(
                    f"http://127.0.0.1:{tc.metrics_port(config)}/metrics", timeout=3
                ).raise_for_status()
                return
        except (AssertionError, httpx.HTTPError) as exc:
            last = str(exc)
        time.sleep(1.0)
    raise AssertionError(f"store and gateway not ready: {last}")


@pytest.fixture
def tutorial_stack(tmp_path: Path) -> Iterator[Path]:
    """Hello-world built, its store and gateway up, one chat turn sent through the gateway.

    Yields the repo root. Teardown is `osprey down`, then this project's
    containers and volumes by exact name, in the same `finally` that holds the
    verdict, so a red run leaves nothing for the next one.
    """
    if shutil.which("docker") is None:
        pytest.fail("the hello-world tutorial test needs docker on PATH")
    _remove_tutorial_containers()
    remove_project_volumes(TUTORIAL_PROJECT_NAME)
    repo = tmp_path / TUTORIAL_PROJECT_NAME
    try:
        init_project(tmp_path, TUTORIAL_PROJECT_NAME, template="hello_world", provider="als-apg")
        config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
        from osprey.deployment.compose_generator import resolve_project_name

        assert resolve_project_name(config) == TUTORIAL_PROJECT_NAME, (
            "the deployment's compose project is not the name the CI cleanup step removes"
        )
        assert set(config.get("deployed_services") or []) == {
            tc.STORE_SERVICE,
            tc.GATEWAY_SERVICE,
        }, f"hello-world deploys more than the store and gateway: {config.get('deployed_services')}"
        # `osprey up` refuses a repo without a .env; the provider key comes from
        # the environment the lane exports.
        (repo / ".env").touch()
        _tutorial_verb(repo, "up", "-d")
        _wait_tutorial_stack(config, _reader_password(repo, TUTORIAL_READER))
        _tutorial_verb(
            repo,
            "chat",
            "--repo",
            str(repo),
            "--no-pin",
            "--print",
            "Use the controls MCP server to read the channel named 'example' "
            "and report its current value.",
        )
        yield repo
    finally:
        if repo.is_dir():
            down = subprocess.run(
                [sys.executable, "-m", "osprey.cli.main", "down"],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=300,
                env=_tutorial_env(),
                check=False,
            )
            if down.returncode != 0:
                print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
        _remove_tutorial_containers()
        remove_project_volumes(TUTORIAL_PROJECT_NAME)


@pytest.mark.harness_benchmark
def test_hello_world_tutorial_query_through_gateway(tutorial_stack: Path) -> None:
    """The tutorial's SQL, run as the preset's reader, shows the chat turn just sent.

    One `osprey chat` turn reaches the store two ways: the agent harness's own
    events through the gateway into `otel_logs`, and the tool call's ledger
    line, read by the gateway from `var/audit/`, into `audit_ledger`. Both are
    stamped with the host identity the build recorded.
    """
    repo = tutorial_stack
    config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8"))
    identity = tc.host_identity(config)
    assert identity, "the render records no deployment.host_identity"
    password = _reader_password(repo, TUTORIAL_READER)
    sql = TUTORIAL_SQL.read_text(encoding="utf-8")

    deadline = time.monotonic() + _TUTORIAL_ROW_TIMEOUT_S
    rows: list[dict] = []
    while time.monotonic() < deadline:
        rows = _query_as_reader(tc.store_port(config), password, sql)
        sources = {row["source"] for row in rows if row["User"] == identity}
        if {"otel_logs", "audit_ledger"} <= sources:
            break
        time.sleep(2.0)

    mine = {row["source"] for row in rows if row["User"] == identity}
    assert "otel_logs" in mine, f"no otel_logs row for {identity!r} in the tutorial query: {rows}"
    assert "audit_ledger" in mine, (
        f"no audit_ledger row for {identity!r} in the tutorial query: {rows}"
    )
