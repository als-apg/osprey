"""Real-token subprocess sweep over the shipped tutorial triggers (L1).

Proves the bundled ``tutorial_triggers.yml`` triggers actually work end to end
ACROSS PROCESSES with a real Claude Agent SDK run, using the provider the run
named in ``OSPREY_E2E_PROVIDER``; the lane skips when that provider's
credential is absent. For each token trigger this:

  1. Builds a real control-assistant deployment repo once (module-scoped fixture).
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
(``tests/dispatch_worker/test_dispatch_api.py`` denylist 403).

This supersedes the old ``tests/integration/test_golden_path.py`` (which fired
its own inline trigger, used cborg, and never ran with tokens in CI): the
subprocess harness below is lifted from it, but now exercises the SHIPPED
trigger definitions and runs in the advisory ``e2e-tests`` job.

Gating: we gate on ``HAS_SDK`` (not a system ``claude --version`` check) — the
worker subprocess refuses to start without the installed pinned ``claude``, and
a runner that lacks it must fail this test loudly, not skip it.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

import pytest
import yaml

from tests.e2e._dispatch_harness import build_repo, log_tail, running_dispatch_stack
from tests.e2e._worker_record import WorkerRecord, worker_record
from tests.e2e.provider import E2E_MODEL, e2e_provider
from tests.e2e.sdk_helpers import HAS_SDK

TOKEN = "tutorial-e2e-token"  # shared dispatcher<->worker bearer for the test

# Generous budget for one real Claude round-trip (CLI spawn + provider latency).
RUN_TIMEOUT_SEC = 150.0
HEALTH_TIMEOUT_SEC = 45.0

pytestmark = [
    # The provider comes from the run, so the credential gate has to follow it
    # rather than name one gateway: see tests/e2e/provider.py.
    pytest.mark.requires_e2e_provider,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.flaky(reruns=2, reruns_delay=5),
]


# ---------------------------------------------------------------------------
# Subprocess harness (lifted from the retired golden-path integration test)
# ---------------------------------------------------------------------------


def _http_get_json(
    url: str, timeout: float = 5.0, headers: dict[str, str] | None = None
) -> dict | list:
    req = urllib.request.Request(url, method="GET", headers=headers or {})  # localhost only
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Build fixture + shipped-triggers loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Init + build a real control-assistant deployment repo once per module."""
    return build_repo(
        tmp_path_factory.mktemp("dispatch_tutorial_build"),
        provider=e2e_provider(),
        model=E2E_MODEL,
    )


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
def agent_record(built_repo: Path, tmp_path: Path) -> Iterator[WorkerRecord]:
    """The record the worker records its runs in and reads them back from."""
    with worker_record(tmp_path / "agent_record", built_repo) as record:
        yield record


@pytest.fixture
def dispatch_stack(built_repo: Path, agent_record: WorkerRecord, tmp_path: Path) -> Iterator[dict]:
    """Start a real worker + dispatcher as subprocesses; tear them down after."""
    with running_dispatch_stack(
        built_repo,
        tmp_path / "triggers.yml",
        _write_local_triggers,
        token=TOKEN,
        health_timeout=HEALTH_TIMEOUT_SEC,
        # A run that is no longer in flight is read back from the record, as
        # the deployed worker reads it from the stack's.
        worker_env=agent_record.worker_env(),
    ) as stack:
        yield stack


# ---------------------------------------------------------------------------
# Fire + poll
# ---------------------------------------------------------------------------


def _fire_webhook(dispatcher_url: str, trigger: str, payload: dict) -> str:
    """Fire a trigger's webhook; return the dispatch_id of the queued run."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # localhost only
        f"{dispatcher_url}/webhook/{trigger}",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        assert resp.status == 202, f"expected 202 from webhook, got {resp.status}"
        fired = json.loads(resp.read().decode("utf-8"))
    assert fired.get("dispatched") is True, fired
    dispatch_id = fired.get("dispatch_id")
    assert dispatch_id, f"no dispatch_id in webhook response: {fired}"
    return dispatch_id


def _wait_for_run(
    dispatcher_url: str, dispatch_id: str, worker_proc: subprocess.Popen, worker_log: Path
) -> dict:
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
                f"worker exited mid-run (rc={worker_proc.returncode}).\n{log_tail(worker_log)}"
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
        f"{RUN_TIMEOUT_SEC:.0f}s.\n{log_tail(worker_log)}"
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
    repo: Path = dispatch_stack["repo"]

    # save-report persists via the workspace artifact tool, which writes to the
    # repo's durable state zone, var/agent_data/artifacts/ (the directory the
    # deployed worker's workspace volume mounts over). Snapshot the existing
    # artifacts before firing so the assertion proves THIS run produced a new one
    # (the repo is shared module-scoped).
    artifacts_dir = repo / "var" / "agent_data" / "artifacts"
    before = set(artifacts_dir.glob("*.md")) if artifacts_dir.is_dir() else set()

    dispatch_id = _fire_webhook(dispatch_stack["dispatcher_url"], trigger, payload)
    run = _wait_for_run(
        dispatch_stack["dispatcher_url"],
        dispatch_id,
        dispatch_stack["worker_proc"],
        dispatch_stack["worker_log"],
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
