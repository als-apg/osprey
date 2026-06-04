"""Real-token subprocess sweep over the shipped tutorial triggers (L1).

Proves the bundled ``tutorial_triggers.yml`` triggers actually work end to end
ACROSS PROCESSES with a real Claude Agent SDK run, using the als-apg provider
(the Bedrock proxy reachable from GitHub Actions runners). For each token
trigger this:

  1. Builds a real control-assistant project once (module-scoped fixture).
  2. Loads the REAL shipped ``tutorial_triggers.yml`` and overrides ONLY
     ``dispatcher.dispatch_target`` to the local worker (the shipped value is a
     compose service name that does not resolve outside the deploy network).
  3. Starts the real worker (``python -m osprey.mcp_server.dispatch_worker``)
     and dispatcher (``python -m osprey.dispatch``) as subprocesses on free
     ports.
  4. Fires the trigger's webhook and polls the dispatcher run feed for the run
     matching the returned dispatch_id, then asserts a trigger-specific outcome.

``denied-tool-demo`` is intentionally NOT exercised here — it is covered
strictly and token-free in the L0 floor
(``tests/unit/dispatch_worker/test_dispatch_api.py`` denylist 403).

This supersedes the old ``tests/integration/test_golden_path.py`` (which fired
its own inline trigger, used cborg, and never ran with tokens in CI): the
subprocess harness below is lifted from it, but now exercises the SHIPPED
trigger definitions and runs in the advisory ``e2e-tests`` job.

Gating: we gate on ``HAS_SDK`` (not a system ``claude --version`` check) — the
Claude Agent SDK ships a bundled ``claude`` binary, and the worker subprocess
resolves it through the SDK, so a PATH check would spuriously skip on CI.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

import pytest
import yaml

from tests.e2e.sdk_helpers import HAS_SDK

TOKEN = "tutorial-e2e-token"  # shared dispatcher<->worker bearer for the test

# Generous budget for one real Claude round-trip (CLI spawn + provider latency).
RUN_TIMEOUT_SEC = 150.0
HEALTH_TIMEOUT_SEC = 45.0

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.flaky(reruns=2, reruns_delay=5),
]


# ---------------------------------------------------------------------------
# Subprocess harness (lifted from the retired golden-path integration test)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Bind to :0, read the assigned port, release it (standard free-port trick)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_get_json(
    url: str, timeout: float = 5.0, headers: dict[str, str] | None = None
) -> dict | list:
    req = urllib.request.Request(url, method="GET", headers=headers or {})  # noqa: S310 - localhost only
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _drain_output(proc: subprocess.Popen) -> str:
    """Best-effort grab of a subprocess's combined output for failure messages."""
    if proc.stdout is None:
        return "(no captured output)"
    try:
        data = proc.stdout.read1(16384) if hasattr(proc.stdout, "read1") else b""
    except Exception:
        data = b""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    return f"--- subprocess output (partial) ---\n{text}" if text else "(no captured output)"


def _wait_for_health(url: str, timeout: float, proc: subprocess.Popen) -> None:
    """Poll ``url`` until it returns HTTP 200, or fail with captured stderr."""
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"subprocess for {url} exited early (rc={proc.returncode}).\n{_drain_output(proc)}"
            )
        try:
            req = urllib.request.Request(url, method="GET")  # noqa: S310 - localhost only
            with urllib.request.urlopen(req, timeout=3.0) as resp:  # noqa: S310
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(0.5)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {url} (last error: {last_err}).\n"
        f"{_drain_output(proc)}"
    )


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Process ignored SIGKILL within the grace window; teardown is
            # best-effort, so leave it for the OS to reap rather than hang the test.
            pass


def _find_osprey_console_script() -> Path:
    """Locate the ``osprey`` console script for the active interpreter."""
    import shutil

    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError(
        "Could not locate the 'osprey' console script. "
        f"Tried {Path(sys.executable).parent / 'osprey'} and PATH."
    )


# ---------------------------------------------------------------------------
# Build fixture + shipped-triggers loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a real als-apg control-assistant project once per module.

    ``--skip-deps`` keeps it fast (no project venv); the worker/dispatcher run
    with this repo's interpreter, so the project venv is not needed.
    """
    base = tmp_path_factory.mktemp("dispatch_tutorial_build")
    project_dir = base / "proj"
    osprey_bin = _find_osprey_console_script()

    proc = subprocess.run(
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
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "CLAUDECODE": ""},
    )
    if proc.returncode != 0:
        pytest.fail(
            f"osprey build (als-apg) failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    if not (project_dir / "config.yml").is_file():
        pytest.fail(f"build succeeded but config.yml missing under {project_dir}")
    return project_dir


def _shipped_triggers_doc() -> dict:
    """Load the REAL bundled tutorial_triggers.yml as a mutable dict."""
    path = resources.files("osprey.profiles.triggers") / "tutorial_triggers.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_local_triggers(dst: Path, worker_port: int) -> None:
    """Write the shipped triggers with dispatch_target repointed at the local worker.

    Only ``dispatcher.dispatch_target`` is changed — the trigger definitions are
    the shipped ones verbatim, so this exercises exactly what ships.
    """
    doc = _shipped_triggers_doc()
    doc.setdefault("dispatcher", {})["dispatch_target"] = f"http://127.0.0.1:{worker_port}"
    dst.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


@pytest.fixture
def dispatch_stack(built_project: Path, tmp_path: Path) -> Iterator[dict]:
    """Start a real worker + dispatcher as subprocesses; tear them down after."""
    worker_port = _free_port()
    dispatcher_port = _free_port()
    triggers_path = tmp_path / "triggers.yml"
    _write_local_triggers(triggers_path, worker_port)

    worker_proc: subprocess.Popen | None = None
    dispatcher_proc: subprocess.Popen | None = None
    try:
        worker_env = {
            **os.environ,
            "DISPATCH_WORKER_PORT": str(worker_port),
            "DISPATCH_WORKER_TOKEN": TOKEN,
            "OSPREY_PROJECT_DIR": str(built_project),
            # The spawned Claude CLI resolves OSPREY config from CWD / CONFIG_FILE.
            "CONFIG_FILE": str(built_project / "config.yml"),
            "CLAUDECODE": "",
        }
        worker_proc = subprocess.Popen(
            [sys.executable, "-m", "osprey.mcp_server.dispatch_worker"],
            cwd=str(built_project),
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _wait_for_health(f"http://127.0.0.1:{worker_port}/health", HEALTH_TIMEOUT_SEC, worker_proc)

        dispatcher_env = {
            **os.environ,
            "TRIGGERS_YML": str(triggers_path),
            "EVENT_DISPATCHER_TOKEN": TOKEN,
            "DISPATCH_WORKER_TOKEN": TOKEN,
            "FASTMCP_TRANSPORT": "http",
            "FASTMCP_PORT": str(dispatcher_port),
            "FASTMCP_HOST": "127.0.0.1",
            "MCP_TRANSPORT": "http",
            "MCP_PORT": str(dispatcher_port),
            "CLAUDECODE": "",
        }
        dispatcher_proc = subprocess.Popen(
            [sys.executable, "-m", "osprey.dispatch"],
            cwd=str(built_project),
            env=dispatcher_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _wait_for_health(
            f"http://127.0.0.1:{dispatcher_port}/health", HEALTH_TIMEOUT_SEC, dispatcher_proc
        )

        yield {
            "dispatcher_url": f"http://127.0.0.1:{dispatcher_port}",
            "worker_url": f"http://127.0.0.1:{worker_port}",
            "project_dir": built_project,
            "worker_proc": worker_proc,
        }
    finally:
        _terminate(dispatcher_proc)
        _terminate(worker_proc)


# ---------------------------------------------------------------------------
# Fire + poll
# ---------------------------------------------------------------------------


def _fire_webhook(dispatcher_url: str, trigger: str, payload: dict) -> str:
    """Fire a trigger's webhook; return the dispatch_id of the queued run."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{dispatcher_url}/webhook/{trigger}",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
        assert resp.status == 202, f"expected 202 from webhook, got {resp.status}"
        fired = json.loads(resp.read().decode("utf-8"))
    assert fired.get("dispatched") is True, fired
    dispatch_id = fired.get("dispatch_id")
    assert dispatch_id, f"no dispatch_id in webhook response: {fired}"
    return dispatch_id


def _wait_for_run(dispatcher_url: str, dispatch_id: str, worker_proc: subprocess.Popen) -> dict:
    """Poll the dispatcher run feed for the run with ``dispatch_id``; return it terminal.

    We correlate on the dispatch_id returned by firing rather than "any terminal
    run": the worker process is restarted per parametrized case while the project
    cwd is shared, and the Claude SDK resumes its per-cwd transcript on a fresh
    process, surfacing an extra phantom run with no dispatch_id. Matching the
    dispatch_id ignores that phantom and pins the assertion to the run we fired.
    The dispatcher's /dashboard/runs proxies the worker feed and enriches each run
    with its dispatch_id. It is a bearer-gated read endpoint, so the poll must send
    the same EVENT_DISPATCHER_TOKEN the dispatcher was started with.
    """
    deadline = time.monotonic() + RUN_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if worker_proc.poll() is not None:
            raise AssertionError(
                f"worker exited mid-run (rc={worker_proc.returncode}).\n"
                f"{_drain_output(worker_proc)}"
            )
        runs = _http_get_json(
            f"{dispatcher_url}/dashboard/runs",
            timeout=5.0,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert isinstance(runs, list)
        for run in runs:
            if run.get("dispatch_id") == dispatch_id and run.get("status") in (
                "completed",
                "error",
            ):
                return run
        time.sleep(2.0)
    raise AssertionError(
        f"run for dispatch_id={dispatch_id} did not reach a terminal state within "
        f"{RUN_TIMEOUT_SEC:.0f}s.\n{_drain_output(worker_proc)}"
    )


_DEMO_PAYLOAD = {
    "signal": "demo:vacuum:pressure",
    "value": 4.2,
    "threshold": 3.0,
    "severity": "warning",
}


@pytest.mark.parametrize(
    "trigger,payload",
    [
        ("hello-dispatch", {}),
        ("triage-event", _DEMO_PAYLOAD),
        ("save-report", _DEMO_PAYLOAD),
    ],
)
def test_tutorial_trigger_completes(dispatch_stack: dict, trigger: str, payload: dict) -> None:
    """Each token tutorial trigger runs end to end and produces its expected effect."""
    project_dir: Path = dispatch_stack["project_dir"]

    # save-report persists via the workspace artifact tool, which writes to
    # _agent_data/artifacts/ (the worker's mounted workspace volume). Snapshot
    # the existing artifacts before firing so the assertion proves THIS run
    # produced a new one (the build dir is shared module-scoped).
    artifacts_dir = project_dir / "_agent_data" / "artifacts"
    before = set(artifacts_dir.glob("*.md")) if artifacts_dir.is_dir() else set()

    dispatch_id = _fire_webhook(dispatch_stack["dispatcher_url"], trigger, payload)
    run = _wait_for_run(
        dispatch_stack["dispatcher_url"], dispatch_id, dispatch_stack["worker_proc"]
    )

    assert run["status"] == "completed", (
        f"{trigger}: run did not complete: status={run['status']!r} error={run.get('error')!r}"
    )
    text_output = (run.get("text_output") or "").strip()
    assert text_output, f"{trigger}: completed run had empty text_output: {run!r}"

    if trigger == "triage-event":
        # Proves the webhook payload reached the agent's context.
        lowered = text_output.lower()
        assert any(tok in lowered for tok in ("pressure", "vacuum", "4.2")), (
            f"triage-event output does not reference the payload: {text_output!r}"
        )
    elif trigger == "save-report":
        # Proves tool use + persistence: a new markdown artifact landed in the
        # workspace volume via the workspace MCP artifact tool.
        new_artifacts = (
            set(artifacts_dir.glob("*.md")) - before if artifacts_dir.is_dir() else set()
        )
        assert new_artifacts, (
            f"save-report did not persist a new artifact under {artifacts_dir} "
            f"(workspace MCP artifact tool). run={run!r}"
        )
