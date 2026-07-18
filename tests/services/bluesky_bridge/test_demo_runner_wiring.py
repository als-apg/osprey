"""Tests for the opt-in demo-runner FastAPI startup wiring (task 2.14a).

`app.py`'s `_lifespan` hook wires a real bluesky-backed `BlueskyPlanRunner`
(against mock ophyd-async devices) ONLY when `BLUESKY_DEMO_RUNNER` is truthy
(`_is_demo_runner_enabled`) AND bluesky is importable — mirroring the
`/plans` route's guarded/lazy-import pattern so the Phase-1 "app.py
import-clean of bluesky" invariant holds whether the extra is absent or the
flag is unset. The truthy check deliberately accepts a few equivalent
spellings ("1", "true", "yes", "on", case/whitespace-insensitive) rather than
one exact string, so this hook can never silently drift out of sync with
whatever value the deploy template/generator actually sets (the template's
own house convention is `"true"`, matching `container_lifecycle.py`'s
`DEV_MODE="true"`, but this hook is not brittle to that one spelling).
Exercised here:

- `_is_demo_runner_enabled()` directly, across the full truthy/non-truthy
  value matrix (fast, no TestClient/lifespan needed).
- Flag absent, or a non-truthy value, or set but bluesky absent: app.py
  stays import-clean and falls back to `FakePlanRunner` — runs entirely in the
  main venv, no bluesky.
- Flag truthy (both "1" and "true", the two production-relevant spellings)
  AND bluesky present: the app wires `BlueskyPlanRunner`, and a full
  launch -> scan -> get_run_data round trip returns real buffered rows.
  Guarded with `pytest.importorskip` so these are skipped (not failed) when
  bluesky isn't installed, keeping `ci_check` green.
- Task 2.5: `BLUESKY_TILED_URI` wires a `TiledWriter` subscription onto the
  demo runner, without disturbing the mock-devices round trip above (the
  real e2e coverage for this path).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import live_rows
from osprey.services.bluesky_bridge.app import app, set_runner_factory
from osprey.services.bluesky_bridge.plan_runner import FakePlanRunner
from osprey.services.bluesky_bridge.runs import registry

_ENV_VAR = "BLUESKY_DEMO_RUNNER"
_TOKEN_VAR = "BLUESKY_LAUNCH_TOKEN"
_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean flag, registry, live-row buffer, and runner factory."""
    for var in (_ENV_VAR, _TILED_URI_ENV, _TILED_API_KEY_ENV):
        monkeypatch.delenv(var, raising=False)
    registry._runs.clear()
    live_rows._clear()
    set_runner_factory(FakePlanRunner)
    yield
    registry._runs.clear()
    live_rows._clear()
    set_runner_factory(FakePlanRunner)


# =========================================================================
# _is_demo_runner_enabled(): the full truthy/non-truthy value matrix
# =========================================================================


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " true ", "yes", "YES", "on", "On"])
def test_is_demo_runner_enabled_accepts_truthy_variants(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_ENV_VAR, value)
    assert app_module._is_demo_runner_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "bogus"])
def test_is_demo_runner_enabled_rejects_non_truthy_variants(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_ENV_VAR, value)
    assert app_module._is_demo_runner_enabled() is False


def test_is_demo_runner_enabled_false_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)
    assert app_module._is_demo_runner_enabled() is False


# =========================================================================
# Flag unset/off: app stays import-clean, FakePlanRunner default unchanged
# =========================================================================


def test_flag_unset_stays_import_clean_and_keeps_fake_scanner_default() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

    assert app_module._runner_factory is FakePlanRunner


def test_flag_false_is_treated_as_off_not_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """App-level confirmation (the value matrix above already covers the
    helper's own logic exhaustively) that an off value flows through the
    real lifespan hook, not just the boolean helper in isolation.
    """
    monkeypatch.setenv(_ENV_VAR, "false")

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

    assert app_module._runner_factory is FakePlanRunner


# =========================================================================
# Flag truthy but bluesky absent: guarded fallback, no crash
# =========================================================================


@pytest.mark.parametrize("value", ["1", "true"])
def test_flag_set_but_bluesky_absent_falls_back_to_fake_scanner(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    try:
        import bluesky  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("bluesky is installed in this environment; nothing to fall back from")

    monkeypatch.setenv(_ENV_VAR, value)

    with caplog.at_level("WARNING", logger="osprey.services.bluesky_bridge.app"):
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200

    assert app_module._runner_factory is FakePlanRunner
    assert any("falling back to FakePlanRunner" in rec.message for rec in caplog.records)


# =========================================================================
# Flag truthy AND bluesky present: real wiring, full round trip
# =========================================================================


@pytest.mark.parametrize("value", ["1", "true"])
def test_flag_set_with_bluesky_present_wires_and_completes_a_real_scan(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    monkeypatch.setenv(_ENV_VAR, value)
    monkeypatch.setenv(_TOKEN_VAR, "s3cr3t")

    with TestClient(app) as client:
        assert app_module._runner_factory is not FakePlanRunner

        create_resp = client.post(
            "/runs",
            json={
                "plan_name": "grid_scan",
                "plan_args": {
                    "detectors": ["det1"],
                    "axes": [{"setpoint": "motor1", "start": 0.0, "stop": 1.0, "num_points": 3}],
                },
            },
        )
        assert create_resp.status_code == 200, create_resp.text
        run_id = create_resp.json()["id"]

        launch_resp = client.post(f"/runs/{run_id}/launch", headers={"X-Launch-Token": "s3cr3t"})
        assert launch_resp.status_code == 200, launch_resp.text

        deadline = time.monotonic() + 15.0
        while client.get(f"/runs/{run_id}").json()["status"] == "running":
            if time.monotonic() > deadline:
                raise AssertionError("demo scan did not complete within the timeout")
            time.sleep(0.05)

        status_body = client.get(f"/runs/{run_id}").json()
        assert status_body["status"] == "completed"

        data_resp = client.get(f"/runs/{run_id}/data")
        assert data_resp.status_code == 200, data_resp.text
        data_body = data_resp.json()
        assert data_body["run_uid"] == status_body["run_uid"]
        assert data_body["row_count"] == 3
        assert len(data_body["rows"]) == 3
        assert "partial" not in data_body


# =========================================================================
# Task 2.5: BLUESKY_TILED_URI wires a TiledWriter subscription
# =========================================================================


def test_tiled_uri_set_subscribes_a_tiled_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors `test_epics_substrate_scanner_wiring.py`'s equivalent assertion
    for the demo runner factory — spying on `TiledWriter.from_uri` keeps this
    a fast unit test while still exercising the real wiring path end to end,
    independent of the mock-devices round trip above (the real e2e coverage
    for this path).
    """
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    from bluesky.callbacks.tiled_writer import TiledWriter

    calls: list[tuple[str, dict]] = []

    def fake_from_uri(uri, **kwargs):
        calls.append((uri, kwargs))
        return lambda name, doc: None

    monkeypatch.setattr(TiledWriter, "from_uri", fake_from_uri)
    monkeypatch.setenv(_ENV_VAR, "true")
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    monkeypatch.setenv(_TILED_API_KEY_ENV, "test-api-key")

    with TestClient(app):
        pass

    from osprey.services.bluesky_bridge.plan_runner_bluesky import BlueskyPlanRunner

    runner = app_module._runner_factory()
    assert isinstance(runner, BlueskyPlanRunner)
    assert calls == [("http://tiled:8000", {"api_key": "test-api-key"})]
    assert runner.tiled_degraded is False


def test_tiled_uri_unset_builds_scanner_with_no_tiled_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (no Tiled configured) behaves exactly like Phase 1: `_build_tiled_writer_factory`
    returns `None`, so `BlueskyPlanRunner.__init__` never imports or calls `TiledWriter` at all.
    """
    pytest.importorskip("bluesky")
    pytest.importorskip("ophyd_async")

    monkeypatch.setenv(_ENV_VAR, "true")

    with TestClient(app):
        pass

    assert app_module._build_tiled_writer_factory() is None

    from osprey.services.bluesky_bridge.plan_runner_bluesky import BlueskyPlanRunner

    runner = app_module._runner_factory()
    assert isinstance(runner, BlueskyPlanRunner)
