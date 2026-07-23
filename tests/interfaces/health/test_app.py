"""Tests for the System Health FastAPI app factory (task 2.4).

Route registration and constant-time liveness are checked with ``TestClient`` +
``app.openapi()["paths"]`` (repo convention). Behavior that depends on a
completed background refresh is driven with an ``httpx.AsyncClient`` over an ASGI
transport and an explicit drain of ``app.state.engine``'s in-flight task, so the
assertions never race the scheduler. The real (fast, network-free) core suite
runs against a temp ``config.yml`` — its per-check statuses vary by host, so only
structural facts (envelope keys, the ``model_chat`` skip row, unfiltered
serving) are asserted.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.health import app as app_mod
from osprey.interfaces.health.app import create_app

_ENVELOPE_KEYS = {
    "summary",
    "ok",
    "warnings",
    "errors",
    "skips",
    "total",
    "results",
    "elapsed_ms",
    "deadline_hit",
    "stale",
    "warming",
    "interval_s",
    "title",
}


def _write_config(
    tmp_path: Path, body: str = "facility_name: Test\nhealth:\n  suite_timeout_s: 15\n"
) -> Path:
    config = tmp_path / "config.yml"
    config.write_text(body)
    return config


async def _drain(app: object) -> None:
    """Await the engine's in-flight refresh task so its cache write lands."""
    engine = app.state.engine  # type: ignore[attr-defined]
    task = engine.current_refresh_task()
    if task is not None:
        await task


def _asgi_client(app: object) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    return httpx.AsyncClient(transport=transport, base_url="http://health.test")


# -- registration + liveness ---------------------------------------------------


def test_routes_registered() -> None:
    app = create_app()
    paths = set(app.openapi()["paths"])
    assert {"/", "/health", "/checks"} <= paths


def test_health_is_constant_time_and_executes_no_checks(tmp_path: Path) -> None:
    app = create_app(_write_config(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"status": "ok", "service": "system-health", "configured": False}
        # /health must not have kicked a refresh (no suite executed for liveness).
        assert app.state.engine.current_refresh_task() is None


def test_root_serves_dashboard_index(tmp_path: Path) -> None:
    app = create_app(_write_config(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    # The bundle ships an index.html; the factory serves it as HTML.
    assert "text/html" in resp.headers["content-type"]


# -- /checks behavior (real refresh) -------------------------------------------


async def test_checks_warming_then_real_report(tmp_path: Path) -> None:
    app = create_app(_write_config(tmp_path))
    async with _asgi_client(app) as client:
        first = (await client.get("/checks")).json()
        assert first["warming"] is True
        assert first["stale"] is True
        assert first["results"] == []

        await _drain(app)  # let the single background refresh complete

        second = (await client.get("/checks")).json()
        assert _ENVELOPE_KEYS <= set(second)
        assert second["warming"] is False
        assert second["total"] > 0
        # on_demand model_chat is emitted as a skip row (never executed, full=False).
        assert any(r["name"] == "model_chat" and r["status"] == "skip" for r in second["results"])


async def test_checks_categories_query_is_accepted_and_ignored(tmp_path: Path) -> None:
    app = create_app(_write_config(tmp_path))
    async with _asgi_client(app) as client:
        await client.get("/checks")  # kick
        await _drain(app)

        full = (await client.get("/checks")).json()
        filtered = (await client.get("/checks?categories=model_chat")).json()

    # The query param changes nothing: identical unfiltered report both ways.
    assert filtered["total"] == full["total"]
    assert [r["name"] for r in filtered["results"]] == [r["name"] for r in full["results"]]
    assert any(r["name"] == "model_chat" for r in filtered["results"])


async def test_checks_degraded_on_missing_config(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "config.yml"  # never created
    app = create_app(missing)
    async with _asgi_client(app) as client:
        await client.get("/checks")  # kick
        await _drain(app)
        env = (await client.get("/checks")).json()

    assert env["warming"] is False
    assert env["interval_s"] == 60.0  # degraded default cadence
    # /health stays 200 and reports the config as unusable.
    async with _asgi_client(app) as client:
        health = (await client.get("/health")).json()
    assert health["status"] == "ok"
    assert health["configured"] is False


# -- guarded construction ------------------------------------------------------


def test_guarded_app_when_engine_wiring_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the engine-less guarded path (okf invariant: factory never raises).
    monkeypatch.setattr(app_mod, "_build_engine", lambda config_path: (None, None))
    app = create_app()
    assert app.state.engine is None

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["configured"] is False

        checks = client.get("/checks")
        assert checks.status_code == 200
        env = checks.json()
        assert env["warming"] is False
        assert env["interval_s"] == 60.0
        assert env["results"] == []


# -- lifespan teardown ---------------------------------------------------------


def test_lifespan_registers_then_unregisters_atexit_hook(tmp_path: Path) -> None:
    app = create_app(_write_config(tmp_path))
    lifecycle = app.state.lifecycle
    assert lifecycle.atexit_registered is False

    with TestClient(app):
        # Startup armed the process-exit teardown hook.
        assert lifecycle.atexit_registered is True

    # Clean shutdown unregistered it, so stacked TestClients never leak hooks.
    assert lifecycle.atexit_registered is False
