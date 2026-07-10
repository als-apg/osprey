"""Tests for `read_run_data`'s dual-source branching (task 3.3).

`GET /runs/{run_id}/data` now has two data sources: the live-row buffer
(`live_rows.py`, task 2.2) and Tiled (`_from_tiled`, task 3.2). This file
tests only the BRANCHING between them — which source gets consulted, and in
what order — by mocking `app_module._from_tiled` directly rather than a real
or faked Tiled client (that boundary is already covered by
`test_tiled_read_source.py`). `_window`'s pagination/truncation math is
already covered by `test_read_bounded.py`'s live-path tests and
`test_tiled_read_source.py`'s Tiled-path tests; this file does not re-test it.

Exercised here:

- Live buffer present (even empty-but-partial): served from live, `_from_tiled`
  never called. The fallback trigger is `buf is None`, never falsy rows — a
  present-but-empty in-flight buffer must NOT be diverted to Tiled.
- Live buffer evicted (`live_rows._MAX_RUNS` exceeded) but the run is still in
  the registry: falls back to `_from_tiled(run_id, ...)`.
- Registry miss (simulating a post-restart lookup, where the whole in-memory
  registry — including `run.run_uid` — is gone): also falls back to
  `_from_tiled(run_id, ...)`, called with the OSPREY `run_id`, never a
  `run_uid` (there is none to find).
- Neither source has the run: 404, not a 200-empty (the MCP tool maps 404 to
  `unknown_run`; a 200-empty would make a nonexistent run look like a valid
  empty scan).
- Schema parity: a completed live-sourced response and a Tiled-sourced
  response carry the identical key set.
- The pre-existing 409 (run known, never promoted) still short-circuits
  before Tiled is ever consulted — there's provably nothing to read from
  either source in that case.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge import live_rows
from osprey.services.bluesky_bridge.app import app, set_scanner_factory
from osprey.services.bluesky_bridge.live_rows import LiveRowRecorder
from osprey.services.bluesky_bridge.runs import Run, do_promote, registry
from osprey.services.bluesky_bridge.scanner import FakeScanner

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


def _refusing_from_tiled(*args: Any, **kwargs: Any) -> dict | None:
    raise AssertionError("_from_tiled must not be called when a live buffer is present")


# =========================================================================
# Live buffer present -> served live, Tiled never consulted
# =========================================================================


def test_live_buffer_present_serves_live_and_skips_tiled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_from_tiled", _refusing_from_tiled)
    run = _promoted_run_with_uid("uid-live")
    _feed("uid-live", [{"x": 1.0}, {"x": 2.0}], stop=True)

    resp = client.get(f"/runs/{run.id}/data")

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_uid"] == "uid-live"
    assert body["rows"] == [[1.0], [2.0]]


def test_in_flight_empty_buffer_stays_on_live_path_not_tiled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRITICAL: a present-but-empty buffer (`partial: true`, zero rows) is an
    in-flight run, not a "nothing here" signal — the fallback trigger must be
    `buf is None`, never falsy rows. If this regresses to a falsy-rows check,
    every in-flight scan with zero events so far gets incorrectly diverted to
    Tiled (which has nothing yet either, since TiledWriter only flushes at
    the stop doc) on every poll until its first event arrives.
    """
    monkeypatch.setattr(app_module, "_from_tiled", _refusing_from_tiled)
    run = _promoted_run_with_uid("uid-empty-partial")
    _feed("uid-empty-partial", [], stop=False)

    resp = client.get(f"/runs/{run.id}/data")

    assert resp.status_code == 200
    body = resp.json()
    assert body["rows"] == []
    assert body["partial"] is True


# =========================================================================
# Live buffer evicted (registry still has the run) -> falls back to Tiled
# =========================================================================


def test_evicted_live_buffer_falls_back_to_tiled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`live_rows._MAX_RUNS` eviction is independent of the registry (which
    never evicts) — a run can still be tracked in the registry while its
    buffer is long gone.
    """
    monkeypatch.setattr(live_rows, "_MAX_RUNS", 1)
    run_a = _promoted_run_with_uid("uid-evicted-a")
    _feed("uid-evicted-a", [{"x": 1.0}], stop=True)
    # A second run's start doc evicts run A's buffer (_MAX_RUNS=1).
    _feed("uid-evicted-b", [{"x": 9.0}], stop=True)
    assert live_rows.get("uid-evicted-a") is None  # sanity: eviction happened

    tiled_body = {
        "run_uid": "uid-evicted-a",
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

    resp = client.get(f"/runs/{run_a.id}/data?max_rows=50&offset=1&tail=true")

    assert resp.status_code == 200
    assert resp.json() == tiled_body
    # `_from_tiled` gets the OSPREY run_id (still known from the registry
    # hit here), and the pagination params flow through unchanged.
    assert calls == [(run_a.id, 50, 1, True)]


# =========================================================================
# Registry miss (post-restart) -> falls back to Tiled by run_id, not run_uid
# =========================================================================


def test_registry_miss_falls_back_to_tiled_by_run_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulates a bridge restart: the in-memory registry (and therefore
    `run.run_uid`) is gone, but Tiled still has the run under its durable
    `osprey_run_id` stamp. There is no `run_uid` to look anything up by here
    — `_from_tiled` must be called with the caller's `run_id` directly.
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

    # No registry.add() at all -> "post-restart" registry miss.
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
# catalog, so `_from_tiled` returns the empty-but-real shape (task 3.2), and
# this route must pass that straight through as a 200 — using `is None`, not
# falsiness, is exactly what keeps this branch from ever converting a real,
# if dataless, run into a bogus `unknown_run`.
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


def test_registry_miss_and_no_tiled_match_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_from_tiled", lambda *a, **kw: None)

    resp = client.get("/runs/truly-unknown-run/data")

    assert resp.status_code == 404


def test_registry_hit_never_had_a_buffer_and_no_tiled_match_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct from `test_evicted_live_buffer_falls_back_to_tiled` above: this
    run's buffer was never created at all (no `_feed` call), not evicted after
    existing — same `buf is None` branch, different one of the two reasons it
    can be `None`, both landing on 404 when Tiled has nothing either.
    """
    run = _promoted_run_with_uid("uid-gone")
    monkeypatch.setattr(app_module, "_from_tiled", lambda *a, **kw: None)

    resp = client.get(f"/runs/{run.id}/data")

    assert resp.status_code == 404


# =========================================================================
# 409 still short-circuits before Tiled is ever consulted
# =========================================================================


def test_unpromoted_run_returns_409_without_consulting_tiled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_from_tiled", _refusing_from_tiled)
    run = registry.add(request={"plan_name": "count"})

    resp = client.get(f"/runs/{run.id}/data")

    assert resp.status_code == 409


# =========================================================================
# Schema parity: live-sourced and Tiled-sourced completed-run responses
# carry the identical key set
# =========================================================================


def test_schema_parity_between_live_and_tiled_sourced_responses(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_run = _promoted_run_with_uid("uid-schema-live")
    _feed("uid-schema-live", [{"x": 1.0}], stop=True)
    live_body = client.get(f"/runs/{live_run.id}/data").json()

    tiled_body_payload = {
        "run_uid": "uid-schema-tiled",
        "columns": ["x"],
        "rows": [[1.0]],
        "row_count": 1,
        "truncated": False,
    }
    monkeypatch.setattr(app_module, "_from_tiled", lambda *a, **kw: tiled_body_payload)
    tiled_body = client.get("/runs/some-restart-era-run-id/data").json()

    assert set(live_body.keys()) == set(tiled_body.keys())
    assert set(live_body.keys()) == {"run_uid", "columns", "rows", "row_count", "truncated"}
