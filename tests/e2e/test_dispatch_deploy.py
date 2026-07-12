"""Full-stack Docker e2e for event dispatch (L2) — highest fidelity.

Unlike the subprocess sweep (``test_dispatch_tutorial.py``), this exercises the
REAL shipped artifacts: the compose templates, the bundled Dockerfile (which
installs Node + the Claude Code CLI the worker needs), the worker ``.env`` mount
that carries provider auth, and the in-network ``dispatch-worker-1:9190``
routing baked into the shipped ``tutorial_triggers.yml`` — none of which the
subprocess path touches.

It builds a control-assistant project, deploys the stack with
``osprey deploy up -d --dev``, fires all four tutorial webhooks at the dispatcher
(host-published on :8020), and asserts:

  * hello-dispatch / triage-event / save-report -> a run completes
  * denied-tool-demo -> rejected by the worker denylist (no completed run)

The worker receives its provider key via the project ``.env`` mounted at
``/app/<project>/.env`` (see the dispatch_worker compose template) — without that
wiring the agent run cannot authenticate, so this test also guards that mount.

Gating: needs Docker and ``ALS_APG_API_KEY``. We do NOT gate on a host
``claude`` binary — the CLI lives inside the built image, not on the runner.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

DISPATCHER_URL = "http://localhost:8020"
TOKEN = "dev-token"  # matches the .env tokens written below

# Container image build (Node + Claude CLI install) is slow on a cold cache.
DEPLOY_UP_TIMEOUT_SEC = 900
HEALTH_TIMEOUT_SEC = 180.0
RUN_TIMEOUT_SEC = 300.0

# hello-dispatch / triage-event / save-report should complete; denied-tool-demo
# must be rejected by the server-side denylist.
_COMPLETING_TRIGGERS = ("hello-dispatch", "triage-event", "save-report")
_DENIED_TRIGGER = "denied-tool-demo"

_DEMO_PAYLOAD = {
    "signal": "demo:vacuum:pressure",
    "value": 4.2,
    "threshold": 3.0,
    "severity": "warning",
}

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    pytest.mark.flaky(reruns=1),
]


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build + ``osprey deploy up`` a control-assistant stack; tear down after."""
    if not os.environ.get("ALS_APG_API_KEY"):
        pytest.skip("ALS_APG_API_KEY not set")

    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("dispatch_deploy_build")
    project_dir = base / "proj"

    build = _run(
        [
            str(osprey_bin),
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
        cwd=base,
        timeout=300,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    # The worker mounts this .env (compose template) so inject_provider_env can
    # resolve the provider key. The compose templates have no token default (they
    # fail closed), and `deploy up` would otherwise auto-generate a random token;
    # we write fixed tokens here so the bearer below is predictable, and pass the
    # provider secret through.
    (project_dir / ".env").write_text(
        "EVENT_DISPATCHER_TOKEN=dev-token\n"
        "DISPATCH_WORKER_TOKEN=dev-token\n"
        f"ALS_APG_API_KEY={os.environ['ALS_APG_API_KEY']}\n",
        encoding="utf-8",
    )

    # Force a fresh image build so the deployed services run CURRENT source. The
    # dispatcher runs osprey-dispatch:local; the worker now runs the unified
    # project image proj:local (built by `deploy up --dev` from the project root).
    # `osprey deploy up` does not pass --build to compose, so it would otherwise
    # reuse existing images and silently test stale code. The freshly-built dev
    # wheel invalidates the relevant build-cache layers on rebuild.
    for image in ("osprey-dispatch:local", "proj:local"):
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)

    try:
        up = _run(
            [str(osprey_bin), "deploy", "up", "-d", "--dev"],
            cwd=project_dir,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey deploy up failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{DISPATCHER_URL}/health", HEALTH_TIMEOUT_SEC)
        # The dispatcher being healthy does not mean the worker is: the worker runs
        # the heavier project image and boots later. Gate on its proxied feed so
        # tests never race a still-warming worker (and a worker that never comes up
        # fails here with container logs, not as a bare 502 mid-test).
        _wait_for_worker_feed(HEALTH_TIMEOUT_SEC)
        yield project_dir
    finally:
        down = _run([str(osprey_bin), "deploy", "down"], cwd=project_dir, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey deploy down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )


def _wait_for_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:  # noqa: S310 - localhost
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {url} (last: {last_err})")


def _dump_stack_diagnostics(context: str) -> str:
    """Return container state + recent worker/dispatcher logs for failure output.

    A full-Docker e2e that fails mid-run is nearly undiagnosable from the pytest
    traceback alone (the interesting state is inside the containers). Surfacing
    ``docker ps`` plus the tail of both service logs turns an opaque ``502`` into
    an actionable report (e.g. the worker crash-looping vs. merely slow to boot).
    """
    lines = [f"=== stack diagnostics ({context}) ==="]
    ps = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=osprey-", "--format", "{{.Names}}\t{{.Status}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines.append("containers:\n" + (ps.stdout.strip() or ps.stderr.strip() or "(none)"))
    for name in ("osprey-dispatch-worker-1", "osprey-event-dispatcher"):
        logs = subprocess.run(
            ["docker", "logs", "--tail", "40", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        tail = ((logs.stdout or "") + (logs.stderr or "")).strip() or "(no logs)"
        lines.append(f"--- {name} (last 40 log lines) ---\n{tail}")
    return "\n".join(lines)


def _wait_for_worker_feed(timeout: float) -> None:
    """Wait until the dispatcher can proxy the worker run feed (HTTP 200).

    The fixture's ``/health`` gate proves only the DISPATCHER is up, but every run
    assertion reads ``/dashboard/runs``, which the dispatcher proxies to the
    worker. The worker runs the full project image and can take appreciably longer
    to become ready than the dispatcher — until it does, the proxy returns ``502``.
    Gate on a ``200`` here so the test never races a still-warming worker, and a
    genuine worker startup failure surfaces here (with container logs) instead of
    as a bare ``502`` mid-test.
    """
    deadline = time.monotonic() + timeout
    last = "(no response yet)"
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{DISPATCHER_URL}/dashboard/runs",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
                if resp.status == 200:
                    return
                last = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(2.0)
    raise AssertionError(
        f"worker feed at {DISPATCHER_URL}/dashboard/runs not ready after "
        f"{timeout:.0f}s (last: {last})\n{_dump_stack_diagnostics('worker feed wait timeout')}"
    )


def _fire(trigger: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{DISPATCHER_URL}/webhook/{trigger}",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
        assert resp.status == 202, f"{trigger}: expected 202 from webhook, got {resp.status}"
        fired = json.loads(resp.read().decode("utf-8"))
    assert fired.get("dispatched") is True, f"{trigger}: {fired}"


def _worker_artifact_files() -> list[str]:
    """List artifact files the worker persisted to its workspace volume.

    Read directly from the worker container because the dispatcher run feed
    reports only status/tool counts, not whether a real artifact landed. This is
    what distinguishes a genuine ``save-report`` (which must persist via the
    ``mcp__osprey_workspace__`` artifact tool) from a hollow "completed" run that
    only claimed success — see the assertion in ``test_full_stack_dispatch``.
    """
    proc = subprocess.run(
        ["docker", "exec", "osprey-dispatch-worker-1", "ls", "/app/proj/_agent_data/artifacts"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.split() if line.endswith(".md")]


def _runs_by_trigger() -> dict[str, dict]:
    """Snapshot the dispatcher run feed keyed by trigger_name (latest wins).

    The dispatcher's /dashboard/runs proxies the worker feed and enriches each
    run with the trigger_name that produced it. It is a bearer-gated read endpoint,
    so the snapshot must send the same EVENT_DISPATCHER_TOKEN written to .env above.
    """
    req = urllib.request.Request(  # noqa: S310
        f"{DISPATCHER_URL}/dashboard/runs",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            runs = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # A transient gateway error means the dispatcher momentarily could not
        # reach the worker upstream (e.g. mid-restart). Treat it as "no feed yet"
        # so the caller's poll loop retries instead of aborting the whole run.
        if exc.code in (502, 503, 504):
            return {}
        raise
    except (urllib.error.URLError, ConnectionError):
        return {}
    by_trigger: dict[str, dict] = {}
    for run in runs:
        name = run.get("trigger_name")
        if name:
            by_trigger[name] = run
    return by_trigger


def test_full_stack_dispatch(deployed_stack: Path) -> None:
    """All four shipped triggers behave correctly through the real Docker stack."""
    # Fire the three completing triggers + the denied one.
    _fire("hello-dispatch", {})
    _fire("triage-event", _DEMO_PAYLOAD)
    _fire("save-report", _DEMO_PAYLOAD)
    _fire(_DENIED_TRIGGER, {})

    # Poll until each completing trigger has a terminal run.
    deadline = time.monotonic() + RUN_TIMEOUT_SEC
    by_trigger: dict[str, dict] = {}
    while time.monotonic() < deadline:
        by_trigger = _runs_by_trigger()
        if all(
            by_trigger.get(t, {}).get("status") in ("completed", "error")
            for t in _COMPLETING_TRIGGERS
        ):
            break
        time.sleep(3.0)

    # If the feed never converged, attach container state so a CI failure reads as
    # "worker crashed / never produced the run" rather than an opaque status miss.
    converged = all(
        by_trigger.get(t, {}).get("status") in ("completed", "error")
        for t in _COMPLETING_TRIGGERS
    )
    diag = "" if converged else "\n" + _dump_stack_diagnostics("run feed did not converge")

    for trigger in _COMPLETING_TRIGGERS:
        run = by_trigger.get(trigger)
        assert run is not None, (
            f"{trigger}: no run appeared in the dispatcher feed: {by_trigger}{diag}"
        )
        assert run.get("status") == "completed", (
            f"{trigger}: expected completed, got status={run.get('status')!r} "
            f"error={run.get('error')!r}{diag}"
        )

    # save-report must persist via the workspace artifact tool, not merely report
    # "completed". Without the worker's startup artifact provisioning, .mcp.json is
    # absent, mcp__osprey_workspace__artifact_save does not exist, the agent's
    # Write fallback is denied by the allowlist, and the run hollow-completes with
    # no artifact on disk. Asserting a real .md artifact landed guards that path.
    artifacts = _worker_artifact_files()
    assert artifacts, (
        "save-report completed but no .md artifact was persisted to the worker "
        "workspace — the osprey_workspace MCP server is likely not provisioned "
        "in-container (missing .mcp.json), so the agent could not actually save"
    )

    # The denylisted trigger is rejected at the worker /dispatch endpoint BEFORE
    # any run record is created, so it must never surface as a completed run.
    denied = by_trigger.get(_DENIED_TRIGGER)
    assert denied is None or denied.get("status") != "completed", (
        f"denied-tool-demo should be rejected by the denylist, not completed: {denied!r}"
    )
