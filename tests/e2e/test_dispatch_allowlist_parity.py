"""E2E: dispatch trigger allowlist is the single authority (real CLI).

Pins the three defects behind the dispatch-allowlist-parity fix and proves
the fix end-to-end against a real built project, a real dispatch worker
subprocess, and the bundled Claude CLI:

* **Repro A (settings-allow bypass):** ``mcp__osprey_workspace__data_list``
  is in the provisioned ``settings.json`` ``permissions.allow``; per SDK
  semantics such calls never reach ``can_use_tool``, so pre-fix they executed
  even when the trigger's ``allowed_tools`` excluded them (observed on a
  deployed worker as ``Agent`` / ``mcp__controls__archiver_read`` running
  un-triggered). Post-fix the PreToolUse hook — which fires for every call —
  denies them.

* **Repro B (subagent starvation):** with the settings allow-rules stripped
  (so success cannot come from settings), a channel-finder delegation must
  work with a trigger list that names NONE of the subagent's tools — the
  hook grants each subagent exactly its declared ``tools:`` surface.
  Pre-fix, the flat allowlist callback denied every subagent call.

* **Repro C (approval-hook-allow bypass):** with approval disabled, the
  facility ``osprey_approval.py`` hook emits explicit ``allow`` for
  ``mcp__controls__*`` — and CLI hook aggregation is NOT deny-dominates
  (see osprey_approval.py's own aggregation note), so pre-fix that allow
  could override the worker's deny. Under ``OSPREY_DISPATCH_RUN=1`` the
  approval hook emits no decision, so the worker hook's deny stands.

Runs are direct ``POST /dispatch`` calls to the worker (bearer-token), no
dispatcher needed. Requires ALS-APG credentials; skips cleanly without.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.test_dispatch_tutorial import (
    HEALTH_TIMEOUT_SEC,
    _find_osprey_console_script,
    _free_port,
    _terminate,
    _wait_for_health,
)

TOKEN = "parity-e2e-token"
RUN_TIMEOUT_SEC = 320.0  # worker's own DISPATCH_TIMEOUT (300s) + polling slack

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.flaky(reruns=2, reruns_delay=5),  # agentic-e2e convention
]

# Deny messages emitted by the worker's tool policy (tool_policy.py).
HOOK_DENY_MARKERS = (
    "is not in this trigger's allowed_tools list",
    "is not in subagent",
    "dispatch server denylist",
)


def _denied_by_policy(result_text: str | None) -> bool:
    return any(marker in (result_text or "") for marker in HOOK_DENY_MARKERS)


# ---------------------------------------------------------------------------
# Project fixtures — one real build, mutated copies per scenario
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a real als-apg control-assistant project once per module."""
    base = tmp_path_factory.mktemp("parity_build")
    project_dir = base / "proj"
    proc = subprocess.run(
        [
            str(_find_osprey_console_script()),
            "build",
            "proj",
            "--preset",
            "control-assistant",
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(base),
            "--force",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "CLAUDECODE": ""},
    )
    if proc.returncode != 0:
        pytest.fail(f"osprey build failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return project_dir


def _copy_project(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, symlinks=True)
    return dst


@pytest.fixture(scope="module")
def stripped_project(built_project: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy of the built project with the Repro-B tools stripped from
    ``settings.json`` ``permissions.allow`` — success can then only come from
    the worker's hook, never from settings (discriminating fixture)."""
    project = _copy_project(built_project, tmp_path_factory.mktemp("parity_stripped") / "proj")
    settings_path = project / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text())
    before = settings.get("permissions", {}).get("allow", [])
    after = [
        entry
        for entry in before
        if "channel-finder" not in entry
        and "submit_response" not in entry
        and "data_list" not in entry
        and not entry.startswith(("Task(", "Agent("))
    ]
    assert len(after) < len(before), "fixture did not strip anything — check settings.json"
    settings["permissions"]["allow"] = after
    settings_path.write_text(json.dumps(settings, indent=2))
    return project


@pytest.fixture(scope="module")
def approval_off_project(built_project: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy of the built project with ``approval.enabled: false`` so the
    facility approval hook's explicit-allow path fires deterministically."""
    import yaml

    project = _copy_project(built_project, tmp_path_factory.mktemp("parity_approval_off") / "proj")
    config_path = project / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config.setdefault("approval", {})["enabled"] = False
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return project


# ---------------------------------------------------------------------------
# Worker harness
# ---------------------------------------------------------------------------


def _start_worker(project_dir: Path) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "osprey.mcp_server.dispatch_worker"],
        cwd=str(project_dir),
        env={
            **os.environ,
            "DISPATCH_WORKER_PORT": str(port),
            "DISPATCH_WORKER_TOKEN": TOKEN,
            "OSPREY_PROJECT_DIR": str(project_dir),
            "CONFIG_FILE": str(project_dir / "config.yml"),
            "CLAUDECODE": "",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    _wait_for_health(f"{url}/health", HEALTH_TIMEOUT_SEC, proc)
    return proc, url


@pytest.fixture
def worker(request) -> Iterator[str]:
    """Start a real worker on the project fixture named by the test param."""
    project_dir = request.getfixturevalue(request.param)
    proc, url = _start_worker(project_dir)
    try:
        yield url
    finally:
        _terminate(proc)


def _http_json(url: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(  # noqa: S310 - localhost only
        url,
        data=body,
        method="POST" if body else "GET",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def dispatch_and_wait(worker_url: str, prompt: str, allowed_tools: list[str]) -> dict:
    """POST /dispatch and poll until the run leaves 'running'."""
    accepted = _http_json(
        f"{worker_url}/dispatch",
        {"prompt": prompt, "allowed_tools": allowed_tools, "max_turns": 15},
    )
    run_id = accepted["run_id"]
    deadline = time.monotonic() + RUN_TIMEOUT_SEC
    while time.monotonic() < deadline:
        run = _http_json(f"{worker_url}/dispatch/{run_id}")
        if run.get("status") not in ("running", "pending"):
            return run
        time.sleep(3.0)
    pytest.fail(f"dispatch run {run_id} did not finish within {RUN_TIMEOUT_SEC}s")


def _calls(run: dict, prefix: str) -> list[dict]:
    return [tc for tc in run.get("tool_calls", []) if tc["name"].startswith(prefix)]


# ---------------------------------------------------------------------------
# Repro A — settings-allow bypass is closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("worker", ["built_project"], indirect=True)
def test_settings_allowed_tool_is_denied_when_trigger_excludes_it(worker):
    """Pre-fix (red): data_list executed because settings.json allow-rules are
    evaluated before can_use_tool (SDK never consults the callback for them).
    Post-fix: the PreToolUse hook denies it — trigger list is the authority."""
    run = dispatch_and_wait(
        worker,
        "Use the data_list tool to list the available data files, then summarize.",
        allowed_tools=["mcp__controls__channel_read"],
    )

    attempts = _calls(run, "mcp__osprey_workspace__data_list")
    assert attempts, f"agent never attempted data_list; text={run.get('text_output')!r}"
    for tc in attempts:
        assert _denied_by_policy(tc["result"]), (
            f"settings-allowed tool executed despite trigger exclusion: {tc}"
        )


# ---------------------------------------------------------------------------
# Repro B — declared subagents work without trigger changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("worker", ["stripped_project"], indirect=True)
def test_subagent_tools_work_in_settings_stripped_project(worker):
    """Pre-fix (red): every channel-finder subagent call was denied by the flat
    trigger allowlist. Post-fix: the context-aware hook grants the subagent its
    declared tools:, in a fixture where settings.json cannot be the reason."""
    run = dispatch_and_wait(
        worker,
        "Find the channel address for the storage ring beam current.",
        allowed_tools=["mcp__controls__channel_read", "mcp__osprey_workspace__data_list"],
    )

    sub_calls = _calls(run, "mcp__channel-finder__") + _calls(
        run, "mcp__osprey_workspace__submit_response"
    )
    assert sub_calls, (
        "agent never delegated to channel-finder; "
        f"tools={[tc['name'] for tc in run.get('tool_calls', [])]}, "
        f"text={run.get('text_output')!r}"
    )
    denied = [tc for tc in sub_calls if _denied_by_policy(tc["result"])]
    assert not denied, f"subagent tool calls denied (starvation persists): {denied}"

    # CF-2 leg: harness pass-through — if the agent waited for MCP cold-start,
    # that call must not have been denied by the policy.
    for tc in _calls(run, "WaitForMcpServers"):
        assert not _denied_by_policy(tc["result"])


# ---------------------------------------------------------------------------
# Repro C — approval-hook explicit allow cannot override the worker's deny
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("worker", ["approval_off_project"], indirect=True)
def test_approval_hook_allow_does_not_override_worker_deny(worker):
    """With approval disabled the facility hook would emit an explicit allow
    for mcp__controls__* (documented to override static permission lists, and
    CLI aggregation is not deny-dominates). Under OSPREY_DISPATCH_RUN=1 it
    emits no decision, so the worker hook's deny must stand for a
    trigger-excluded controls tool."""
    run = dispatch_and_wait(
        worker,
        # Fully specified so the agent has no reason to ask a clarifying
        # question instead of attempting the tool call (observed flake: with no
        # time range the agent asked "what period?" and never touched the tool,
        # so the deny this test exists to observe never happened).
        "Call the archiver_read tool NOW for channel SR:BEAM:CURRENT with "
        "start='2h ago' and end='now'. Do not ask any clarifying questions; "
        "attempt the tool call immediately and report its result.",
        allowed_tools=["mcp__controls__channel_read"],
    )

    attempts = _calls(run, "mcp__controls__archiver_read")
    assert attempts, f"agent never attempted archiver_read; text={run.get('text_output')!r}"
    for tc in attempts:
        assert _denied_by_policy(tc["result"]), (
            f"approval-hook allow overrode the worker deny: {tc}"
        )
