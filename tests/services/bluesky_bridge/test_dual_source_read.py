"""Tests for `get_run_data`'s dual-source branching.

`GET /runs/{run_id}/data` has two data sources: the live-row buffer
(`live_rows.py`) and Tiled (`_from_tiled`). This file tests only the BRANCHING
between them — which source gets consulted, and in what order — by mocking
`app_module._from_tiled` directly rather than a real or faked Tiled client
(that boundary is already covered by `test_tiled_read_source.py`). `_window`'s
pagination/truncation math is already covered by `test_read_bounded.py`'s live-
path tests and `test_tiled_read_source.py`'s Tiled-path tests; this file does
not re-test it.

The route holds no run state of its own: `run_id` is looked up in the buffer
directly. That single lookup is what makes both live-row keyings work — a
queue-executed run is buffered under its OSPREY run id by the document plane,
while a run reaching a bare recorder is buffered under the RunEngine's own uid
— and both are pinned below.

Exercised here:

- Live buffer present (even empty-but-partial): served from live, `_from_tiled`
  never called. The fallback trigger is `buf is None`, never falsy rows — a
  present-but-empty in-flight buffer must NOT be diverted to Tiled.
- Both buffer keyings resolve through the same lookup.
- Live buffer evicted (`live_rows._MAX_RUNS` exceeded), and never created at
  all: both fall back to `_from_tiled(run_id, ...)`.
- Neither source has the run: 404, not a 200-empty (the MCP tool maps 404 to
  `unknown_run`; a 200-empty would make a nonexistent run look like a valid
  empty run).
- Schema parity: a completed live-sourced response and a Tiled-sourced
  response carry the identical key set.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import live_rows
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.live_rows import LiveRowRecorder

_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Every test in this file monkeypatches `app_module._from_tiled` directly
    rather than relying on the real one, so none is *currently* sensitive to
    ambient Tiled env — but clearing both vars here too means a future test
    that forgets to mock `_from_tiled` fails loudly (wrong branch exercised)
    rather than passing by accident against whatever happens to be unset in
    the ambient environment. See `test_read_bounded.py`'s matching fixture
    for the concrete failure mode this guards against.
    """
    live_rows._clear()
    monkeypatch.delenv(_TILED_URI_ENV, raising=False)
    monkeypatch.delenv(_TILED_API_KEY_ENV, raising=False)
    yield
    live_rows._clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _feed(key: str, rows: list[dict], *, stop: bool = False, keyed: bool = False) -> None:
    """Push synthetic start/event[/stop] documents into the live buffer.

    ``keyed`` picks the buffer keying: the document plane's (the OSPREY run id,
    passed explicitly) or a bare recorder's (the start document's own uid).
    """
    recorder = LiveRowRecorder(key=key) if keyed else LiveRowRecorder()
    recorder("start", {"uid": key if not keyed else "some-other-run-engine-uid"})
    for row in rows:
        recorder("event", {"data": row})
    if stop:
        recorder("stop", {})


def _refusing_from_tiled(*args: Any, **kwargs: Any) -> dict | None:
    raise AssertionError("_from_tiled must not be called when a live buffer is present")


# =========================================================================
# Live buffer present -> served live, Tiled never consulted
# =========================================================================


def test_live_buffer_present_serves_live_and_skips_tiled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_from_tiled", _refusing_from_tiled)
    _feed("run-live", [{"x": 1.0}, {"x": 2.0}], stop=True)

    resp = client.get("/runs/run-live/data")

    assert resp.status_code == 200
    assert resp.json()["rows"] == [[1.0], [2.0]]


def test_a_document_plane_keyed_buffer_is_found_by_the_osprey_run_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production keying: the worker's RunEngine minted a uid the bridge
    never chose, so the document plane buffers under the id the operator's
    item carried instead."""
    monkeypatch.setattr(app_module, "_from_tiled", _refusing_from_tiled)
    _feed("osprey-run-id", [{"x": 3.0}], stop=True, keyed=True)

    resp = client.get("/runs/osprey-run-id/data")

    assert resp.status_code == 200
    assert resp.json()["rows"] == [[3.0]]


def test_a_run_engine_uid_keyed_buffer_is_found_by_that_uid(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other keying, which coexists by design: a run carrying no OSPREY id
    is buffered under the only identity it has."""
    monkeypatch.setattr(app_module, "_from_tiled", _refusing_from_tiled)
    _feed("run-engine-uid", [{"x": 4.0}], stop=True)

    resp = client.get("/runs/run-engine-uid/data")

    assert resp.status_code == 200
    assert resp.json()["rows"] == [[4.0]]


def test_in_flight_empty_buffer_stays_on_live_path_not_tiled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRITICAL: a present-but-empty buffer (`partial: true`, zero rows) is an
    in-flight run, not a "nothing here" signal — the fallback trigger must be
    `buf is None`, never falsy rows. If this regresses to a falsy-rows check,
    every in-flight run with zero events so far gets incorrectly diverted to
    Tiled (which has nothing yet either, since TiledWriter only flushes at
    the stop doc) on every poll until its first event arrives.
    """
    monkeypatch.setattr(app_module, "_from_tiled", _refusing_from_tiled)
    _feed("run-empty-partial", [], stop=False)

    resp = client.get("/runs/run-empty-partial/data")

    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"] == []
    assert body["partial"] is True


def test_the_live_path_reports_run_uid_as_none_rather_than_inventing_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bridge holds the buffer but not the uid the worker's RunEngine
    minted. The key stays present so consumers see one response shape; its
    value is honestly unknown."""
    monkeypatch.setattr(app_module, "_from_tiled", _refusing_from_tiled)
    _feed("run-no-uid", [{"x": 1.0}], stop=True, keyed=True)

    body = client.get("/runs/run-no-uid/data").json()

    assert "run_uid" in body
    assert body["run_uid"] is None


# =========================================================================
# No live buffer -> falls back to Tiled, keyed by the run id
# =========================================================================


def test_evicted_live_buffer_falls_back_to_tiled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`live_rows._MAX_RUNS` eviction retires a buffer for a run whose data is
    still durable in Tiled."""
    monkeypatch.setattr(live_rows, "_MAX_RUNS", 1)
    _feed("run-evicted-a", [{"x": 1.0}], stop=True)
    # A second run's start doc evicts run A's buffer (_MAX_RUNS=1).
    _feed("run-evicted-b", [{"x": 9.0}], stop=True)
    assert live_rows.get("run-evicted-a") is None  # sanity: eviction happened

    tiled_body = {
        "run_uid": "bluesky-uid-a",
        "columns": ["x"],
        "rows": [[1.0]],
        "row_count": 1,
        "truncated": False,
    }
    calls: list[tuple] = []

    def fake_from_tiled(run_id: str, max_rows: int, offset: int | None, tail: bool) -> dict | None:
        calls.append((run_id, max_rows, offset, tail))
        return tiled_body

    monkeypatch.setattr(app_module, "_from_tiled", fake_from_tiled)

    resp = client.get("/runs/run-evicted-a/data?max_rows=50&offset=1&tail=true")

    assert resp.status_code == 200
    assert resp.json() == tiled_body
    # `_from_tiled` gets the caller's run id, and the pagination params flow
    # through unchanged.
    assert calls == [("run-evicted-a", 50, 1, True)]


def test_a_run_with_no_buffer_at_all_falls_back_to_tiled_by_run_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post-restart shape: every buffer is gone with the process, but Tiled
    still has the run under its durable `osprey_run_id` stamp. A run still
    executing at that moment lands here too, which is correct — Tiled is the
    only source that survived.
    """
    tiled_body = {
        "run_uid": "bluesky-uid-after-restart",
        "columns": ["motor"],
        "rows": [[1.0], [2.0]],
        "row_count": 2,
        "truncated": False,
    }
    calls: list[tuple] = []

    def fake_from_tiled(run_id: str, max_rows: int, offset: int | None, tail: bool) -> dict | None:
        calls.append((run_id, max_rows, offset, tail))
        return tiled_body

    monkeypatch.setattr(app_module, "_from_tiled", fake_from_tiled)

    resp = client.get("/runs/some-run-id-from-before-restart/data")

    assert resp.status_code == 200
    assert resp.json() == tiled_body
    assert calls == [("some-run-id-from-before-restart", 100, None, False)]


# =========================================================================
# Matched-but-empty Tiled run -> 200 with zero rows, NOT a 404
#
# `None` from `_from_tiled` means "no run matched the search" -> 404. A run
# that matched but never got a "primary" stream (e.g. it errored before its
# first point) is a different thing entirely: it genuinely exists in the
# catalog, so `_from_tiled` returns the empty-but-real shape, and this route
# must pass that straight through as a 200 — using `is None`, not falsiness,
# is exactly what keeps this branch from ever converting a real, if dataless,
# run into a bogus `unknown_run`.
# =========================================================================


def test_matched_but_empty_tiled_run_returns_200_not_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_but_real = {
        "run_uid": "bluesky-uid-errored-early",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
    }
    monkeypatch.setattr(app_module, "_from_tiled", lambda *a, **kw: empty_but_real)

    resp = client.get("/runs/errored-before-first-point/data")

    assert resp.status_code == 200
    assert resp.json() == empty_but_real


# =========================================================================
# Neither source has the run -> 404, never a 200-empty
# =========================================================================


def test_no_buffer_and_no_tiled_match_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_from_tiled", lambda *a, **kw: None)

    resp = client.get("/runs/truly-unknown-run/data")

    assert resp.status_code == 404


# =========================================================================
# Schema parity: live-sourced and Tiled-sourced completed-run responses
# carry the identical key set
# =========================================================================


def test_schema_parity_between_live_and_tiled_sourced_responses(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _feed("run-schema-live", [{"x": 1.0}], stop=True)
    live_body = client.get("/runs/run-schema-live/data").json()

    tiled_body_payload = {
        "run_uid": "uid-schema-tiled",
        "columns": ["x"],
        "rows": [[1.0]],
        "row_count": 1,
        "truncated": False,
    }
    monkeypatch.setattr(app_module, "_from_tiled", lambda *a, **kw: tiled_body_payload)
    tiled_body = client.get("/runs/some-other-run-id/data").json()

    assert set(live_body.keys()) == set(tiled_body.keys())
    assert set(live_body.keys()) == {"run_uid", "columns", "rows", "row_count", "truncated"}
