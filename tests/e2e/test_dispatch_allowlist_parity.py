"""E2E: dispatch trigger allowlist is the single authority (real CLI).

Pins the three defects behind the dispatch-allowlist-parity fix and proves
the fix end-to-end against a real deployment repo, a real dispatch worker
subprocess, and the bundled Claude CLI:

* **Repro A (settings-allow bypass):** the probe tool is in the provisioned
  ``settings.json`` ``permissions.allow``; per SDK semantics such calls never
  reach ``can_use_tool``, so pre-fix they executed even when the trigger's
  ``allowed_tools`` excluded them (observed on a deployed worker as ``Agent``
  / ``mcp__controls__archiver_read`` running un-triggered). Post-fix the
  PreToolUse hook — which fires for every call — denies them. The probe must
  be a tool NO declared subagent lists (asserted by
  ``test_probe_tool_is_discriminating``): a surface-declared tool such as
  ``artifact_list`` is legitimately allowed inside a delegated subagent (that is
  Repro B's feature), so with such a probe the assertion would hinge on
  whether the model happens to delegate — the exact nondeterminism that made
  this test flaky.

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

Each scenario gets its own COPY of the built repo, mutated in its ``build/``
zone — the render is what the worker reads, and a copy keeps one scenario's
edit from reaching another's worker.

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

# Repro A's probe: settings-allowed, but in NO subagent's declared tools:
# surface — so it is denied on every path (main thread by the trigger list,
# subagent context by the surface check) and the assertion cannot depend on
# whether the model delegates. test_probe_tool_is_discriminating enforces
# both properties against the real deployment repo.
PROBE_TOOL = "mcp__osprey_workspace__session_log"


def _denied_by_policy(result_text: str | None) -> bool:
    return any(marker in (result_text or "") for marker in HOOK_DENY_MARKERS)


# ---------------------------------------------------------------------------
# Repo fixtures — one real build, mutated copies per scenario
# ---------------------------------------------------------------------------


def _run_osprey(argv: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_find_osprey_console_script()), *argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Init + build a real als-apg control-assistant deployment repo once per module."""
    base = tmp_path_factory.mktemp("parity_build")
    repo = base / "proj"

    init = _run_osprey(
        [
            "init",
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
        ],
        cwd=base,
    )
    if init.returncode != 0:
        pytest.fail(f"osprey init failed (rc={init.returncode}):\n{init.stdout}\n{init.stderr}")

    build = _run_osprey(
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
        cwd=base,
    )
    if build.returncode != 0:
        pytest.fail(f"osprey build failed (rc={build.returncode}):\n{build.stdout}\n{build.stderr}")
    return repo


def _copy_repo(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, symlinks=True)
    return dst


@pytest.fixture(scope="module")
def stripped_repo(built_repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy of the built repo with the Repro-B tools stripped from the render's
    ``settings.json`` ``permissions.allow`` — success can then only come from
    the worker's hook, never from settings (discriminating fixture)."""
    repo = _copy_repo(built_repo, tmp_path_factory.mktemp("parity_stripped") / "proj")
    settings_path = repo / "build" / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text())
    before = settings.get("permissions", {}).get("allow", [])
    after = [
        entry
        for entry in before
        if "channel-finder" not in entry
        and "submit_response" not in entry
        and "artifact_list" not in entry
        and not entry.startswith(("Task(", "Agent("))
    ]
    assert len(after) < len(before), "fixture did not strip anything — check settings.json"
    settings["permissions"]["allow"] = after
    settings_path.write_text(json.dumps(settings, indent=2))
    return repo


@pytest.fixture(scope="module")
def approval_off_repo(built_repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy of the built repo with ``approval.enabled: false`` in the render's
    config so the facility approval hook's explicit-allow path fires
    deterministically."""
    import yaml

    repo = _copy_repo(built_repo, tmp_path_factory.mktemp("parity_approval_off") / "proj")
    config_path = repo / "build" / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config.setdefault("approval", {})["enabled"] = False
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return repo


# ---------------------------------------------------------------------------
# Worker harness
# ---------------------------------------------------------------------------


def _start_worker(repo: Path) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "osprey.mcp_server.dispatch_worker"],
        cwd=str(repo),
        env={
            **os.environ,
            "DISPATCH_WORKER_PORT": str(port),
            "DISPATCH_WORKER_TOKEN": TOKEN,
            # Repo root + the render's config one level down, exactly as the
            # dispatch_worker compose template wires the deployed worker.
            "OSPREY_PROJECT_DIR": str(repo),
            "CONFIG_FILE": str(repo / "build" / "config.yml"),
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
    """Start a real worker on the repo fixture named by the test param."""
    repo = request.getfixturevalue(request.param)
    proc, url = _start_worker(repo)
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


def test_probe_tool_is_discriminating(built_repo):
    """The probe stays valid only while (a) settings.json allows it — else it
    stops pinning the settings-allow bypass — and (b) no declared subagent
    lists it — else a delegated call is legitimately allowed (Repro B's
    feature) and the deny assertion becomes a bet on the model not delegating.
    If (b) ever trips, pick a new probe from the settings allow-list that no
    agent declares; do NOT weaken the deny assertion."""
    from osprey.mcp_server.dispatch_worker.agent_surfaces import parse_project_agents

    render = built_repo / "build"
    settings = json.loads((render / ".claude" / "settings.json").read_text())
    allow = settings.get("permissions", {}).get("allow", [])
    assert PROBE_TOOL in allow, (
        f"probe {PROBE_TOOL} is no longer settings-allowed — Repro A would no "
        f"longer exercise the settings-allow bypass; pick a settings-allowed probe"
    )

    # Repro C's probe needs the same no-surface property for the same reason.
    for probe in (PROBE_TOOL, "mcp__controls__archiver_read"):
        offenders = {
            name: sorted(surface)
            for name, surface in parse_project_agents(render).items()
            if surface is not None and probe in surface
        }
        assert not offenders, (
            f"probe {probe} is now declared by subagent(s) {sorted(offenders)} — "
            f"a delegated call would legitimately execute and its deny assertion "
            f"goes flaky; pick a probe that no agent declares"
        )


@pytest.mark.parametrize("worker", ["built_repo"], indirect=True)
def test_settings_allowed_tool_is_denied_when_trigger_excludes_it(worker):
    """Pre-fix (red): the probe executed because settings.json allow-rules are
    evaluated before can_use_tool (SDK never consults the callback for them).
    Post-fix: the PreToolUse hook denies it — trigger list is the authority.

    Fully specified, no-delegation prompt for the same reason as Repro C's:
    the deny this test observes requires the model to actually attempt the
    call. Delegated attempts are fine — the probe is in no agent's surface,
    so the subagent-context deny fires and its marker matches too."""
    run = dispatch_and_wait(
        worker,
        "Call the session_log tool NOW, yourself — do not delegate to a "
        "subagent and do not ask any clarifying questions. Attempt the tool "
        "call immediately and report its result.",
        allowed_tools=["mcp__controls__channel_read"],
    )

    attempts = _calls(run, PROBE_TOOL)
    assert attempts, f"agent never attempted session_log; text={run.get('text_output')!r}"
    for tc in attempts:
        assert _denied_by_policy(tc["result"]), (
            f"settings-allowed tool executed despite trigger exclusion: {tc}"
        )


# ---------------------------------------------------------------------------
# Repro B — declared subagents work without trigger changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("worker", ["stripped_repo"], indirect=True)
def test_subagent_tools_work_in_settings_stripped_repo(worker):
    """Pre-fix (red): every channel-finder subagent call was denied by the flat
    trigger allowlist. Post-fix: the context-aware hook grants the subagent its
    declared tools:, in a fixture where settings.json cannot be the reason."""
    run = dispatch_and_wait(
        worker,
        "Find the channel address for the storage ring beam current.",
        allowed_tools=["mcp__controls__channel_read", "mcp__osprey_workspace__artifact_list"],
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


@pytest.mark.parametrize("worker", ["approval_off_repo"], indirect=True)
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
