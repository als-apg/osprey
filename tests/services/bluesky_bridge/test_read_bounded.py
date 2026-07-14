"""Tests for `GET /runs/{id}/data` — the bounded live-row read route (task 2.2).

Two layers:

1. Direct `FastAPI` `TestClient` tests against the bridge app, seeding the
   live-row buffer via `LiveRowRecorder` (no bluesky import, no Tiled server)
   to exercise the route's pagination/truncation/partial logic.
2. A MANDATORY non-patched integration test (Phase 1 handoff item): a real
   HTTP server hosting this app, hit by the *actual* `read_scan_data` MCP
   tool over a real socket — `_http_get_json` is never mocked here, so a
   missing/renamed bridge route can never hide behind a patched primitive
   again.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import uvicorn
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import live_rows
from osprey.services.bluesky_bridge.app import app, set_scanner_factory
from osprey.services.bluesky_bridge.live_rows import LiveRowRecorder
from osprey.services.bluesky_bridge.runs import Run, do_promote, registry
from osprey.services.bluesky_bridge.scanner import FakeScanner

_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Every test also gets Tiled unconfigured: several tests below (any run
    with no live buffer) fall through `read_run_data` to the real
    `_from_tiled`, not a mock. Left to ambient env, an exported
    `BLUESKY_TILED_URI` would make those tests attempt a real Tiled
    connection instead of exercising the "not configured" branch they claim
    to — hanging, or raising `KeyError` on the unset `BLUESKY_TILED_API_KEY`
    and surfacing as a 500 instead of the asserted 404.
    """
    registry._runs.clear()
    live_rows._clear()
    set_scanner_factory(FakeScanner)
    monkeypatch.delenv(_TILED_URI_ENV, raising=False)
    monkeypatch.delenv(_TILED_API_KEY_ENV, raising=False)
    yield
    registry._runs.clear()
    live_rows._clear()
    set_scanner_factory(FakeScanner)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _promoted_run_with_uid(run_uid: str) -> Run:
    """A run promoted with a FakeScanner pre-seeded with `run_uid`."""
    run = registry.add(request={"plan_name": "count"})
    do_promote(run, lambda: FakeScanner(run_uid=run_uid))
    return run


def _feed(run_uid: str, rows: list[dict], *, stop: bool = False) -> None:
    """Push synthetic start/event[/stop] documents into the live buffer."""
    recorder = LiveRowRecorder()
    recorder("start", {"uid": run_uid})
    for row in rows:
        recorder("event", {"data": row})
    if stop:
        recorder("stop", {"run_start": run_uid})


# =========================================================================
# 409: no run_uid yet
# =========================================================================


def test_read_data_before_promotion_returns_409(client: TestClient) -> None:
    run = registry.add(request={"plan_name": "count"})
    resp = client.get(f"/runs/{run.id}/data")
    assert resp.status_code == 409


def test_read_data_unknown_run_returns_404(client: TestClient) -> None:
    resp = client.get("/runs/does-not-exist/data")
    assert resp.status_code == 404


# =========================================================================
# Empty stream
# =========================================================================


def test_read_data_with_no_buffer_and_no_tiled_returns_404(client: TestClient) -> None:
    """Task 3.3: a promoted run with no live buffer falls back to `_from_tiled`,
    which returns `None` here (BLUESKY_TILED_URI unset in this test env) — so
    this is a 404, not the old Phase-1 200-empty shape. A 200-empty would make
    a run whose data genuinely can't be found look like a valid empty scan;
    see `test_dual_source_read.py` for the full dual-source branching matrix
    (evicted-buffer fallback, schema parity, registry-miss handling).
    """
    run = _promoted_run_with_uid("uid-empty")
    resp = client.get(f"/runs/{run.id}/data")
    assert resp.status_code == 404


def test_read_data_started_but_zero_events_reports_full_shape(client: TestClient) -> None:
    run = _promoted_run_with_uid("uid-2")
    _feed("uid-2", [])
    body = client.get(f"/runs/{run.id}/data").json()
    assert body["run_uid"] == "uid-2"
    assert body["columns"] == []
    assert body["rows"] == []
    assert body["row_count"] == 0
    assert body["truncated"] is False
    assert body["partial"] is True


# =========================================================================
# Pagination / truncation
# =========================================================================


def test_read_data_returns_all_rows_when_under_max_rows(client: TestClient) -> None:
    run = _promoted_run_with_uid("uid-3")
    _feed("uid-3", [{"x": 1.0}, {"x": 2.0}], stop=True)
    body = client.get(f"/runs/{run.id}/data").json()
    assert body["columns"] == ["x"]
    assert body["rows"] == [[1.0], [2.0]]
    assert body["row_count"] == 2
    assert body["truncated"] is False
    assert "partial" not in body


def test_read_data_caps_at_max_rows_and_flags_truncated(client: TestClient) -> None:
    run = _promoted_run_with_uid("uid-4")
    _feed("uid-4", [{"x": float(i)} for i in range(5)], stop=True)
    body = client.get(f"/runs/{run.id}/data?max_rows=2").json()
    assert body["rows"] == [[0.0], [1.0]]
    assert body["row_count"] == 5
    assert body["truncated"] is True


def test_read_data_offset_paginates_forward(client: TestClient) -> None:
    run = _promoted_run_with_uid("uid-5")
    _feed("uid-5", [{"x": float(i)} for i in range(5)], stop=True)
    body = client.get(f"/runs/{run.id}/data?max_rows=2&offset=2").json()
    assert body["rows"] == [[2.0], [3.0]]
    assert body["truncated"] is True


def test_read_data_tail_returns_most_recent_rows(client: TestClient) -> None:
    run = _promoted_run_with_uid("uid-6")
    _feed("uid-6", [{"x": float(i)} for i in range(5)], stop=True)
    body = client.get(f"/runs/{run.id}/data?max_rows=2&tail=true").json()
    assert body["rows"] == [[3.0], [4.0]]
    assert body["truncated"] is True


def test_read_data_tail_with_offset_skips_from_the_end(client: TestClient) -> None:
    run = _promoted_run_with_uid("uid-7")
    _feed("uid-7", [{"x": float(i)} for i in range(5)], stop=True)
    body = client.get(f"/runs/{run.id}/data?max_rows=2&tail=true&offset=1").json()
    # Skip the very last row (index 4), then take the 2 rows before that.
    assert body["rows"] == [[2.0], [3.0]]


# =========================================================================
# Partial mid-run
# =========================================================================


def test_read_data_mid_run_reports_partial_true(client: TestClient) -> None:
    run = _promoted_run_with_uid("uid-8")
    _feed("uid-8", [{"x": 1.0}], stop=False)
    body = client.get(f"/runs/{run.id}/data").json()
    assert body["partial"] is True


def test_read_data_after_stop_omits_partial(client: TestClient) -> None:
    run = _promoted_run_with_uid("uid-9")
    _feed("uid-9", [{"x": 1.0}], stop=True)
    body = client.get(f"/runs/{run.id}/data").json()
    assert "partial" not in body


# =========================================================================
# row_count reflects the true total even past the storage cap
# =========================================================================


def test_read_data_row_count_is_true_total_even_past_storage_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live_rows, "_MAX_ROWS_PER_RUN", 3)
    run = _promoted_run_with_uid("uid-10")
    _feed("uid-10", [{"x": float(i)} for i in range(5)], stop=True)
    body = client.get(f"/runs/{run.id}/data").json()
    assert len(body["rows"]) == 3
    assert body["row_count"] == 5
    assert body["truncated"] is True


# =========================================================================
# MANDATORY: non-patched, real-socket integration test through the MCP tool
# =========================================================================


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"bridge did not become ready on port {port} within {timeout}s")


@contextmanager
def _live_bridge() -> Iterator[str]:
    """Run the real bridge app on a real TCP port in a background thread.

    Same module-level `registry`/`live_rows` singletons as the test process
    (this is a thread, not a subprocess) — but the request/response still
    travels over a real socket through the real ASGI app, so a missing or
    misrouted endpoint fails exactly as it would in production.
    """
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    _wait_for_port(port)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(timeout=5)


async def test_read_scan_data_tool_end_to_end_over_real_http(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual MCP tool, unpatched, talking to the actual bridge route over a real socket.

    Guards against the exact gap flagged at the Phase 1 handoff: every other
    `read_scan_data` test patches `_http_get_json`, so a renamed/missing
    bridge route could pass every unit test while being unreachable in
    production. This test never touches `_http_get_json` — it runs the real
    bridge, seeds its real run registry + live-row buffer, points the scan
    server context at the real port, and calls the real MCP tool.
    """
    from osprey.mcp_server.scan import server_context
    from osprey.mcp_server.scan.tools import read_tools
    from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

    run = _promoted_run_with_uid("uid-e2e")
    _feed("uid-e2e", [{"x": float(i)} for i in range(5)], stop=True)

    monkeypatch.chdir(tmp_path)  # no config.yml in scope -> env var + defaults only

    with _live_bridge() as base_url:
        monkeypatch.setenv("BLUESKY_BRIDGE_URL", base_url)
        server_context.reset_server_context()
        server_context.initialize_server_context()
        try:
            result = await get_tool_fn(read_tools.read_scan_data)(run_id=run.id, max_rows=2)
        finally:
            server_context.reset_server_context()

    data = extract_response_dict(result)
    assert data["run_uid"] == "uid-e2e"
    assert data["rows"] == [[0.0], [1.0]]
    assert data["row_count"] == 5
    assert data["truncated"] is True
