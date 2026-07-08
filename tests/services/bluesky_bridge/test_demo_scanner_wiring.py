"""Tests for the opt-in demo-scanner FastAPI startup wiring (task 2.14a).

`app.py`'s `_lifespan` hook wires a real bluesky-backed `BlueskyScanner`
(against mock ophyd-async devices) ONLY when `BLUESKY_DEMO_SCANNER` is truthy
(`_is_demo_scanner_enabled`) AND bluesky is importable — mirroring the
`/plans` route's guarded/lazy-import pattern so the Phase-1 "app.py
import-clean of bluesky" invariant holds whether the extra is absent or the
flag is unset. The truthy check deliberately accepts a few equivalent
spellings ("1", "true", "yes", "on", case/whitespace-insensitive) rather than
one exact string, so this hook can never silently drift out of sync with
whatever value the deploy template/generator actually sets (the template's
own house convention is `"true"`, matching `container_lifecycle.py`'s
`DEV_MODE="true"`, but this hook is not brittle to that one spelling).
Exercised here:

- `_is_demo_scanner_enabled()` directly, across the full truthy/non-truthy
  value matrix (fast, no TestClient/lifespan needed).
- Flag absent, or a non-truthy value, or set but bluesky absent: app.py
  stays import-clean and falls back to `FakeScanner` — runs entirely in the
  main venv, no bluesky.
- Flag truthy (both "1" and "true", the two production-relevant spellings)
  AND bluesky present: the app wires `BlueskyScanner`, and a full
  promote -> scan -> read_scan_data round trip returns real buffered rows.
  Guarded with `pytest.importorskip` so these are skipped (not failed) when
  bluesky isn't installed, keeping `ci_check` green.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import live_rows
from osprey.services.bluesky_bridge.app import app, set_scanner_factory
from osprey.services.bluesky_bridge.runs import registry
from osprey.services.bluesky_bridge.scanner import FakeScanner

_ENV_VAR = "BLUESKY_DEMO_SCANNER"
_TOKEN_VAR = "BLUESKY_PROMOTE_TOKEN"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a clean flag, registry, live-row buffer, and scanner factory."""
    monkeypatch.delenv(_ENV_VAR, raising=False)
    registry._runs.clear()
    live_rows._clear()
    set_scanner_factory(FakeScanner)
    yield
    registry._runs.clear()
    live_rows._clear()
    set_scanner_factory(FakeScanner)


# =========================================================================
# _is_demo_scanner_enabled(): the full truthy/non-truthy value matrix
# =========================================================================


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " true ", "yes", "YES", "on", "On"])
def test_is_demo_scanner_enabled_accepts_truthy_variants(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_ENV_VAR, value)
    assert app_module._is_demo_scanner_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "bogus"])
def test_is_demo_scanner_enabled_rejects_non_truthy_variants(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(_ENV_VAR, value)
    assert app_module._is_demo_scanner_enabled() is False


def test_is_demo_scanner_enabled_false_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)
    assert app_module._is_demo_scanner_enabled() is False


# =========================================================================
# Flag unset/off: app stays import-clean, FakeScanner default unchanged
# =========================================================================


def test_flag_unset_stays_import_clean_and_keeps_fake_scanner_default() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

    assert app_module._scanner_factory is FakeScanner


def test_flag_false_is_treated_as_off_not_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """App-level confirmation (the value matrix above already covers the
    helper's own logic exhaustively) that an off value flows through the
    real lifespan hook, not just the boolean helper in isolation.
    """
    monkeypatch.setenv(_ENV_VAR, "false")

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200

    assert app_module._scanner_factory is FakeScanner


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

    assert app_module._scanner_factory is FakeScanner
    assert any("falling back to FakeScanner" in rec.message for rec in caplog.records)


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
        assert app_module._scanner_factory is not FakeScanner

        create_resp = client.post(
            "/runs",
            json={"plan_name": "count", "plan_args": {"detectors": ["det1"], "num": 3}},
        )
        assert create_resp.status_code == 200, create_resp.text
        run_id = create_resp.json()["id"]

        promote_resp = client.post(f"/runs/{run_id}/promote", headers={"X-Promote-Token": "s3cr3t"})
        assert promote_resp.status_code == 200, promote_resp.text

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
