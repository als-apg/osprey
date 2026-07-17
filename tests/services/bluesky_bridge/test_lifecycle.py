"""Unit tests for the bridge lifecycle core end-to-end over HTTP (FakePlanRunner, no bluesky).

Exercises the assembled FastAPI app (`bluesky_bridge/app.py`) rather than calling
`runs.py`/`security.py` directly, so these tests also stand as a contract test
for the routes' error semantics (FR5: 404/409/403/503/500) and the run
lifecycle state machine as seen by an HTTP client (e.g. the `scan` MCP tools).
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge.app import app, launch_run, set_runner_factory, stop_run
from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner
from osprey.services.bluesky_bridge.runs import registry

_ENV_VAR = "BLUESKY_LAUNCH_TOKEN"
_TOKEN = "s3cr3t"


@pytest.fixture(autouse=True)
def _isolated_bridge_state():
    """Every test gets an empty run registry and a fresh default runner factory.

    Both `registry` and the app's `_runner_factory` are process-wide
    singletons (as they are in a real bridge instance), so tests must clean
    up after themselves rather than relying on module reimport.
    """
    registry._runs.clear()
    set_runner_factory(FakePlanRunner)
    yield
    registry._runs.clear()
    set_runner_factory(FakePlanRunner)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _create_run(client: TestClient) -> str:
    resp = client.post("/runs", json={"plan_name": "grid_scan", "plan_args": {"num": 3}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    return body["id"]


# =========================================================================
# Token gate: 503 unarmed, 403 mismatch, 200 valid
# =========================================================================


def test_launch_without_token_returns_503_and_touches_no_scanner(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)
    runner = FakePlanRunner()
    set_runner_factory(lambda: runner)

    run_id = _create_run(client)
    resp = client.post(f"/runs/{run_id}/launch")

    assert resp.status_code == 503
    assert runner.reinitialize_calls == 0
    assert runner.start_calls == 0
    assert client.get(f"/runs/{run_id}").json()["status"] == "pending"


def test_launch_with_invalid_token_returns_403_and_touches_no_scanner(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    runner = FakePlanRunner()
    set_runner_factory(lambda: runner)

    run_id = _create_run(client)
    resp = client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": "wrong"})

    assert resp.status_code == 403
    assert runner.reinitialize_calls == 0
    assert runner.start_calls == 0
    assert client.get(f"/runs/{run_id}").json()["status"] == "pending"


def test_launch_with_valid_token_starts_the_scan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    runner = FakePlanRunner()
    set_runner_factory(lambda: runner)

    run_id = _create_run(client)
    resp = client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "running"
    assert body["launched_by"] == "agent"
    assert body["run_uid"]
    assert runner.reinitialize_calls == 1
    assert runner.start_calls == 1


# =========================================================================
# Concurrency / re-launch guards: 409
# =========================================================================


def test_concurrent_second_launch_returns_409_while_active(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    set_runner_factory(FakePlanRunner)

    run_id = _create_run(client)
    first = client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})
    assert first.status_code == 200

    second = client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})
    assert second.status_code == 409
    assert "already launched" in second.json()["detail"]


def test_launch_after_stop_returns_409(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stopping an intent that was never launched still permanently forecloses
    # launch — do_launch's `stopped` guard fires regardless.
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    set_runner_factory(FakePlanRunner)

    run_id = _create_run(client)
    stop_resp = client.post(f"/runs/{run_id}/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"

    launch_resp = client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})
    assert launch_resp.status_code == 409
    assert "stopped" in launch_resp.json()["detail"]


def test_launch_after_stopping_an_already_launched_run_returns_409(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    set_runner_factory(FakePlanRunner)

    run_id = _create_run(client)
    first_launch = client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})
    assert first_launch.status_code == 200

    stop_resp = client.post(f"/runs/{run_id}/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"

    launch_resp = client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})
    assert launch_resp.status_code == 409


# =========================================================================
# State machine: intent -> running -> completed | stopped | error
# =========================================================================


def test_state_machine_intent_before_launch(client: TestClient) -> None:
    run_id = _create_run(client)
    assert client.get(f"/runs/{run_id}").json()["status"] == "pending"


def test_state_machine_running_after_launch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    set_runner_factory(FakePlanRunner)

    run_id = _create_run(client)
    client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})
    assert client.get(f"/runs/{run_id}").json()["status"] == "running"


def test_state_machine_completed_once_scan_finishes_cleanly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    runner = FakePlanRunner()
    set_runner_factory(lambda: runner)

    run_id = _create_run(client)
    client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})
    runner.simulate_completion()

    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "completed"
    assert body["completion"] == 1.0


def test_state_machine_stopped_via_stop_route(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    runner = FakePlanRunner()
    set_runner_factory(lambda: runner)

    run_id = _create_run(client)
    client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})
    stop_resp = client.post(f"/runs/{run_id}/stop")

    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"
    assert runner.stop_calls == 1
    assert client.get(f"/runs/{run_id}").json()["status"] == "stopped"


def test_state_machine_error_when_scanner_ends_in_error_state(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    runner = FakePlanRunner()
    set_runner_factory(lambda: runner)

    run_id = _create_run(client)
    client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})
    runner.simulate_error("device timeout")

    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "error"
    assert "error" in body


def test_state_machine_error_when_reinitialize_fails_at_launch_time(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    set_runner_factory(lambda: FakePlanRunner(reinitialize_fails=True))

    run_id = _create_run(client)
    resp = client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": _TOKEN})

    assert resp.status_code == 500
    assert client.get(f"/runs/{run_id}").json()["status"] == "error"


# =========================================================================
# Unknown run: 404
# =========================================================================


def test_get_unknown_run_returns_404(client: TestClient) -> None:
    resp = client.get("/runs/does-not-exist")
    assert resp.status_code == 404


def test_launch_unknown_run_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_VAR, _TOKEN)
    resp = client.post("/runs/does-not-exist/launch", headers={"X-Launch-Token": _TOKEN})
    assert resp.status_code == 404


def test_stop_unknown_run_returns_404(client: TestClient) -> None:
    resp = client.post("/runs/does-not-exist/stop")
    assert resp.status_code == 404


# =========================================================================
# Listing
# =========================================================================


def test_list_runs_returns_newest_first(client: TestClient) -> None:
    first_id = _create_run(client)
    second_id = _create_run(client)

    body = client.get("/runs").json()
    assert [run["id"] for run in body] == [second_id, first_id]


# =========================================================================
# Race: stop() landing during do_launch's unlocked runner-build window
# =========================================================================


def test_stop_during_unlocked_build_window_never_leaves_a_live_untracked_scan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministically interleave launch and stop around the unlocked window.

    `do_launch` sets `launching=True` under the lock, then builds/starts the
    runner OUTSIDE the lock (it may be slow) before re-acquiring the lock to
    publish `runner`/`launched`. A `stop()` landing in that unlocked window
    used to see `launched=False`/`runner=None`, skip stopping anything, and
    just record `stopped=True` — after which the just-started runner got
    published anyway: a live running behind a run reporting "stopped".

    This test forces exactly that interleaving via a runner whose
    `reinitialize` blocks (simulating the slow build) until the stop call has
    actually landed, and asserts the runner is stopped exactly once and never
    left running.
    """
    monkeypatch.setenv(_ENV_VAR, _TOKEN)

    reinitialize_started = threading.Event()
    stop_landed = threading.Event()
    created: dict[str, FakePlanRunner] = {}

    class BlockingBuildScanner(FakePlanRunner):
        def reinitialize(self, exec_config: object) -> bool:
            # Signal that do_launch has reached the unlocked build phase
            # (past the `launching=True` lock section), then hold here —
            # simulating a slow runner build — until stop() has landed.
            reinitialize_started.set()
            assert stop_landed.wait(timeout=5), "stop() never landed during the build window"
            return super().reinitialize(exec_config)

    def factory() -> BlockingBuildScanner:
        runner = BlockingBuildScanner()
        created["runner"] = runner
        return runner

    set_runner_factory(factory)
    run_id = _create_run(client)

    launch_result: dict[str, object] = {}

    def run_launch() -> None:
        launch_result["response"] = launch_run(run_id, x_launch_token=_TOKEN)

    launchr = threading.Thread(target=run_launch)
    launchr.start()
    try:
        assert reinitialize_started.wait(timeout=5), "launch never reached the build phase"

        # The launch thread is now blocked inside the unlocked build window
        # (launching=True, runner/launched not yet published). Stop lands here.
        stop_run(run_id)
        stop_landed.set()

        launchr.join(timeout=5)
        assert not launchr.is_alive(), "launch thread did not finish"
    finally:
        # Always unblock the background thread even if an assertion above failed.
        stop_landed.set()

    runner = created["runner"]
    assert runner.stop_calls == 1, "runner must be stopped exactly once, not left running"
    assert runner.is_run_active() is False

    run = registry.get(run_id)
    assert run.stopped is True
    assert run.status == "stopped"
