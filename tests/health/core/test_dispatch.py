"""The ``dispatch`` health category reads the event dispatcher's own surfaces."""

from __future__ import annotations

import asyncio
import json

import httpx

from osprey.health.core.dispatch import dispatch
from osprey.health.models import Status

_CONFIG = {"services": {"event_dispatcher": {"port": 10010}}}
_NOW = 1_800_000_000.0


def _handler(*, pool=None, runs=None, worker_error=None, health_status=200, state_status=200):
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            body = {"status": "ok", "pool": pool or {"running": 1, "max": 4, "queued": 0}}
            return httpx.Response(health_status, json=body)
        if request.url.path == "/dashboard/state":
            body = {"pool": {}, "runs": runs or [], "worker_error": worker_error}
            return httpx.Response(state_status, json=body)
        return httpx.Response(404)

    return handle, seen


def _rows(config, handler, monkeypatch, token: str | None = "tok") -> dict:
    if token is None:
        monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    else:
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", token)
    run = dispatch(config, transport=httpx.MockTransport(handler), now=lambda: _NOW)
    return {row.name: row for row in asyncio.run(run())}


def test_no_rows_without_a_dispatcher(monkeypatch):
    handle, seen = _handler()
    assert _rows({}, handle, monkeypatch) == {}
    assert seen == []


def test_pool_occupancy(monkeypatch):
    handle, seen = _handler(pool={"running": 1, "max": 4, "queued": 0})
    row = _rows(_CONFIG, handle, monkeypatch)["dispatch.pool"]

    assert row.status is Status.OK
    assert row.value == "1 / 4 busy"
    assert row.probed and row.latency_ms > 0
    assert str(seen[0].url) == "http://127.0.0.1:10010/health"


def test_queued_work_is_shown(monkeypatch):
    handle, _ = _handler(pool={"running": 2, "max": 2, "queued": 3})
    assert _rows(_CONFIG, handle, monkeypatch)["dispatch.pool"].value == "2 / 2 busy · 3 queued"


def test_unreachable_dispatcher_warns_and_skips_the_run_count(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    rows = _rows(_CONFIG, refuse, monkeypatch)
    assert rows["dispatch.pool"].status is Status.WARNING
    assert rows["dispatch.pool"].value == "unreachable"
    assert "dispatch.failed_24h" not in rows


def test_failed_runs_in_the_last_day(monkeypatch):
    runs = [
        {"run_id": "7c1e0000aaaa", "status": "error", "created_at": _NOW - 3600},
        {"run_id": "old", "status": "error", "created_at": _NOW - 3 * 86400},
        {"run_id": "fine", "status": "completed", "created_at": _NOW - 60},
    ]
    handle, seen = _handler(runs=runs)
    row = _rows(_CONFIG, handle, monkeypatch)["dispatch.failed_24h"]

    assert row.status is Status.WARNING
    assert row.value == "1 · run 7c1e0000"
    state = next(r for r in seen if r.url.path == "/dashboard/state")
    assert state.headers["Authorization"] == "Bearer tok"
    assert json.loads(json.dumps(dict(state.url.params))) == {"timeline_hours": "24"}


def test_no_failed_runs(monkeypatch):
    handle, _ = _handler(runs=[{"run_id": "x", "status": "completed", "created_at": _NOW}])
    row = _rows(_CONFIG, handle, monkeypatch)["dispatch.failed_24h"]
    assert row.status is Status.OK
    assert row.value == "0"


def test_without_a_token_the_run_count_is_a_skip(monkeypatch):
    handle, seen = _handler()
    row = _rows(_CONFIG, handle, monkeypatch, token=None)["dispatch.failed_24h"]
    assert row.status is Status.SKIP
    assert row.value == "no dispatcher token here"
    assert all(r.url.path == "/health" for r in seen)


def test_worker_error_is_reported(monkeypatch):
    handle, _ = _handler(worker_error="worker down")
    row = _rows(_CONFIG, handle, monkeypatch)["dispatch.failed_24h"]
    assert row.status is Status.WARNING
    assert row.value == "worker unreachable"


def test_a_rejected_token_warns(monkeypatch):
    handle, _ = _handler(state_status=401)
    row = _rows(_CONFIG, handle, monkeypatch)["dispatch.failed_24h"]
    assert row.status is Status.WARNING
    assert row.value == "HTTP 401"


def test_default_port_from_the_layout(monkeypatch):
    handle, seen = _handler()
    _rows({"services": {"event_dispatcher": {}}}, handle, monkeypatch)
    assert seen[0].url.port == 10010
