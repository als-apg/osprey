"""Tests for `GET /runs/{run_id}/figure` — the figure route.

The route is TOTAL over known runs: every run either the live buffer or Tiled
knows gets a 200 figure, every degradation is the default figure with its
machine-readable ``reason``, and only a run neither source knows 404s —
byte-for-byte `get_run_data`'s 404, pinned here by comparing the two routes'
raw response bodies. ``partial`` and ``source`` are asserted present on every
single response this file receives (see `_get_figure`).

Sources are faked at the same seams the sibling files use: the live buffer is
fed real documents through `LiveRowRecorder` / `RunDocumentRouter`, and Tiled
through a fake `tiled.client.from_uri` modeled on `test_tiled_read_source.py`'s
(flattened-column composite, table under ``.base["internal"]``), extended with
plan stamps, stop-document presence, multi-node search results, and call
counters — the counters are what make "a settled Tiled hit issues zero Tiled
calls" an assertion rather than a hope.

Plan resolution is faked by monkeypatching `plan_loader.get_facility_plans`
for the reason-path tests (so each tier/failure is constructed exactly), and
exercised against the REAL catalog for the shipped-`orm` resolution test and
the perf gate — those two are the ones that prove the route needs no manager,
no Tiled, and no test-injected catalog to serve a live run's figure.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from osprey.services.bluesky_bridge import figure_cache, live_rows
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.document_plane import RunDocumentRouter
from osprey.services.bluesky_bridge.figure import (
    DEFAULT_MAX_POINTS,
    REASON_NO_RENDER,
    REASON_PARAMS_MISMATCH,
    REASON_PLAN_IDENTITY_UNAVAILABLE,
    REASON_RENDER_FAILED,
    REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS,
    REASON_SOURCE_UNAVAILABLE,
    Figure,
    LinesMark,
    Panel,
    Point,
    Series,
)
from osprey.services.bluesky_bridge.figure import (
    BarsMark as FigBarsMark,
)
from osprey.services.bluesky_bridge.live_rows import LiveRowRecorder
from osprey.services.bluesky_bridge.plan_loader import FacilityPlans
from osprey.services.bluesky_bridge.plan_types import PlanSpec
from osprey.services.bluesky_bridge.queue_backend import PLAN_META_KEY, RUN_ID_META_KEY

_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    """Fresh buffers, fresh cache, no ambient Tiled config.

    Clearing the Tiled env means a test that forgets to fake the client fails
    loudly on the wrong branch instead of passing against whatever the ambient
    environment happens to hold (`test_dual_source_read.py`'s precedent).
    """
    live_rows._clear()
    figure_cache._clear()
    monkeypatch.delenv(_TILED_URI_ENV, raising=False)
    monkeypatch.delenv(_TILED_API_KEY_ENV, raising=False)
    yield
    live_rows._clear()
    figure_cache._clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _get_figure(client: TestClient, run_id: str) -> dict:
    """GET the figure, asserting the envelope invariants every response owes.

    ``partial`` and ``source`` are ALWAYS present — that is the route's
    contract, so it is asserted on every single fetch this file performs
    rather than in one dedicated test.
    """
    response = client.get(f"/runs/{run_id}/figure")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "partial" in body
    assert body["source"] in ("live", "tiled")
    return body


# =========================================================================
# Catalog fakes
# =========================================================================


class _EchoParams(BaseModel):
    """Permissive schema for fake plans whose kwargs don't matter."""

    model_config = {"extra": "allow"}


class _StrictParams(BaseModel):
    """Schema with a required field, for the schema-drift path."""

    required_gain: float = Field(...)


def _spec(
    name: str = "demo",
    *,
    provenance: str = "shipped",
    render: Any = None,
    schema: type[BaseModel] = _EchoParams,
) -> PlanSpec[Any]:
    return PlanSpec(
        name=name,
        plan=lambda devices, params: None,
        schema=schema,
        provenance=provenance,  # type: ignore[arg-type]
        render=render,
    )


def _install_catalog(monkeypatch: pytest.MonkeyPatch, *specs: PlanSpec[Any]) -> None:
    plans = FacilityPlans(plans={spec.name: spec for spec in specs})
    monkeypatch.setattr(
        "osprey.services.bluesky_bridge.plan_loader.get_facility_plans", lambda: plans
    )


def _echo_render(rows: list[dict[str, Any]], params: Any) -> Figure:
    """A deterministic render built row-by-row from what it was handed.

    One series per detector-ish column, y taken verbatim (None survives as a
    gap) — so any live/Tiled difference the adapter failed to normalize (NaN
    vs None, row order) would change the output and fail the parity test.
    ``partial``/``source`` are the team's placeholder convention: the route
    must overwrite both.
    """
    columns = sorted({key for row in rows for key in row})
    panels = [
        Panel(
            title=column,
            mark=LinesMark(
                series=[
                    Series(
                        label=column,
                        points=[Point(x=float(i), y=row.get(column)) for i, row in enumerate(rows)],
                        source_points=len(rows),
                    )
                ],
                source_points=len(rows),
            ),
        )
        for column in columns
    ]
    return Figure(panels=panels, partial=True, source="live")


# =========================================================================
# Live-buffer seeding
# =========================================================================


def _feed_live(
    run_id: str,
    rows: list[dict[str, Any]],
    *,
    plan: Any = None,
    stop: bool = False,
) -> None:
    """Push start/event[/stop] documents into the live buffer, stamped."""
    recorder = LiveRowRecorder(key=run_id, plan=plan)
    recorder("start", {"uid": f"engine-{run_id}"})
    for row in rows:
        recorder("event", {"data": row})
    if stop:
        recorder("stop", {})


# =========================================================================
# Tiled fakes (modeled on test_tiled_read_source.py, plus what the figure
# route needs: stamps, stop presence, start.time, multi-match, call counters)
# =========================================================================


class _FakeInternalTable:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def read(self) -> pd.DataFrame:
        return self._df


class _FakeBaseContainer:
    def __init__(self, tables: dict) -> None:
        self._tables = tables

    def __getitem__(self, key: str):
        return self._tables[key]


class _FakeCompositeClient:
    """The run's `"primary"` child: keys are flattened column names, the table
    hangs off `.base` — same fidelity rationale as `test_tiled_read_source.py`."""

    def __init__(self, columns: list[str], base: _FakeBaseContainer) -> None:
        self._columns = list(columns)
        self.base = base

    def __getitem__(self, key: str):
        if key in self._columns:
            return object()
        raise KeyError(f"Key '{key}' not found.")


class _FakeRunNode:
    def __init__(self, metadata: dict, streams: dict | None = None) -> None:
        self.metadata = metadata
        self._streams = streams or {}

    def __getitem__(self, key: str):
        return self._streams[key]

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True


class _FakeSearchResult:
    def __init__(self, nodes: list[_FakeRunNode]) -> None:
        self._nodes = nodes

    def values(self) -> list[_FakeRunNode]:
        return self._nodes


class _FakeTiledClient:
    """Counts every search; a settled cache hit must leave the count alone."""

    def __init__(self, nodes_by_run_id: dict[str, list[_FakeRunNode]]) -> None:
        self._nodes_by_run_id = nodes_by_run_id
        self.search_queries: list[object] = []

    def search(self, query) -> _FakeSearchResult:
        self.search_queries.append(query)
        return _FakeSearchResult(list(self._nodes_by_run_id.get(query.value, [])))


def _install_fake_tiled(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeTiledClient,
    *,
    api_key: str | None = "test-api-key",
) -> list[tuple]:
    """Configure the env and patch `from_uri`; returns the from_uri call log."""
    pytest.importorskip("tiled")
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")
    if api_key is not None:
        monkeypatch.setenv(_TILED_API_KEY_ENV, api_key)

    calls: list[tuple] = []

    def fake_from_uri(uri, **kwargs):
        calls.append((uri, kwargs))
        return client

    monkeypatch.setattr("tiled.client.from_uri", fake_from_uri)
    return calls


def _tiled_run_node(
    run_uid: str,
    run_id: str,
    rows: list[dict],
    *,
    plan: Any = None,
    stopped: bool = True,
    start_time: float = 1000.0,
    row_order: list[int] | None = None,
) -> _FakeRunNode:
    """A run container shaped like TiledWriter's output.

    ``row_order`` stores the table's rows in that physical order while
    ``seq_num`` still records emission order — the route must sort, so a
    shuffled table and an ordered one produce the same figure.
    """
    start: dict[str, Any] = {"uid": run_uid, "osprey_run_id": run_id, "time": start_time}
    if plan is not None:
        start[PLAN_META_KEY] = plan
    metadata: dict[str, Any] = {"start": start}
    if stopped:
        metadata["stop"] = {"exit_status": "success"}

    table_rows = []
    for i, row in enumerate(rows):
        table_row = {"seq_num": i + 1, "time": 1000.0 + i, **row}
        table_row.update({f"ts_{k}": 1000.0 + i for k in row})
        table_rows.append(table_row)
    if row_order is not None:
        table_rows = [table_rows[i] for i in row_order]
    if not table_rows:
        return _FakeRunNode(metadata=metadata, streams={})
    df = pd.DataFrame(table_rows)
    primary = _FakeCompositeClient(
        columns=list(df.columns),
        base=_FakeBaseContainer({"internal": _FakeInternalTable(df)}),
    )
    return _FakeRunNode(metadata=metadata, streams={"primary": primary})


# =========================================================================
# 404 parity with get_run_data
# =========================================================================


def test_unknown_run_404_matches_get_run_data_byte_for_byte(client: TestClient) -> None:
    """No live buffer AND no Tiled match: the figure route's 404 is
    `get_run_data`'s, byte for byte — the MCP tool maps both identically."""
    figure_response = client.get("/runs/no-such-run/figure")
    data_response = client.get("/runs/no-such-run/data")

    assert figure_response.status_code == 404
    assert data_response.status_code == 404
    assert figure_response.content == data_response.content


def test_unknown_run_with_tiled_configured_is_still_the_same_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_tiled(monkeypatch, _FakeTiledClient({}))

    figure_response = client.get("/runs/no-such-run/figure")
    data_response = client.get("/runs/no-such-run/data")

    assert figure_response.status_code == 404
    assert figure_response.content == data_response.content


# =========================================================================
# Reason paths (live branch, catalog faked per test)
# =========================================================================


def test_missing_stamp_is_plan_identity_unavailable(client: TestClient) -> None:
    _feed_live("run-1", [{"m": 1.0, "d": 10.0}], plan=None)

    body = _get_figure(client, "run-1")

    assert body["reason"] == REASON_PLAN_IDENTITY_UNAVAILABLE
    assert body["source"] == "live"
    assert body["partial"] is True
    # The default figure still draws the numeric columns.
    assert [panel["title"] for panel in body["panels"]] == ["m", "d"]


def test_stamp_with_null_name_is_plan_identity_unavailable(client: TestClient) -> None:
    _feed_live("run-1", [{"m": 1.0}], plan={"name": None, "kwargs": {}})

    assert _get_figure(client, "run-1")["reason"] == REASON_PLAN_IDENTITY_UNAVAILABLE


@pytest.mark.parametrize("bad_name", [["orm"], 42, {"nested": "orm"}, 1.5])
def test_non_string_stamp_name_is_plan_identity_unavailable(
    client: TestClient, bad_name: Any
) -> None:
    """A malformed stamp whose name is not a string must degrade like any
    other unusable identity — never reach a catalog lookup or a cache key,
    where an unhashable name would 500 a route that promises 200s."""
    _feed_live("run-1", [{"m": 1.0}], plan={"name": bad_name, "kwargs": {}}, stop=True)

    body = _get_figure(client, "run-1")

    assert body["reason"] == REASON_PLAN_IDENTITY_UNAVAILABLE


def test_a_raising_catalog_is_no_render_and_never_cached(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing plan catalog is this process's transient trouble, not a truth
    about the run: it serves the default figure with `no_render` but must NOT
    stick to a settled run — once the catalog recovers, the next poll serves
    the real figure."""
    _feed_live("run-1", [{"m": 1.0, "d": 2.0}], plan={"name": "echo", "kwargs": {}}, stop=True)

    def broken_catalog() -> FacilityPlans:
        raise OSError("session plan dir is read-only")

    monkeypatch.setattr(
        "osprey.services.bluesky_bridge.plan_loader.get_facility_plans", broken_catalog
    )
    degraded = _get_figure(client, "run-1")
    assert degraded["reason"] == REASON_NO_RENDER

    _install_catalog(monkeypatch, _spec("echo", render=_echo_render))
    recovered = _get_figure(client, "run-1")

    assert recovered["reason"] is None
    assert [panel["title"] for panel in recovered["panels"]] == ["d", "m"]


def test_stamp_name_with_no_current_owner_is_no_render(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan removed or renamed since the run: identity is known, but no plan
    code currently owns a view of it."""
    _install_catalog(monkeypatch)  # empty catalog
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "gone", "kwargs": {}})

    assert _get_figure(client, "run-1")["reason"] == REASON_NO_RENDER


def test_spec_without_render_is_no_render(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec("plain", render=None))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "plain", "kwargs": {}})

    assert _get_figure(client, "run-1")["reason"] == REASON_NO_RENDER


@pytest.mark.parametrize("provenance", ["session", "unreviewed"])
def test_session_tier_spec_is_the_session_reason_not_no_render(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, provenance: str
) -> None:
    """Session-tier specs always carry `render=None` (the loader strips it),
    so the tier must be decided on provenance — the authoring agent learns
    WHY there is no custom view, not just that there isn't one."""
    _install_catalog(monkeypatch, _spec("mine", provenance=provenance, render=None))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "mine", "kwargs": {}})

    assert _get_figure(client, "run-1")["reason"] == REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS


def test_schema_drift_is_params_mismatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec("strict", render=_echo_render, schema=_StrictParams))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "strict", "kwargs": {"old_field": 2.0}})

    assert _get_figure(client, "run-1")["reason"] == REASON_PARAMS_MISMATCH


def test_raising_render_is_render_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exploding(rows: list[dict[str, Any]], params: Any) -> Figure:
        raise RuntimeError("boom")

    _install_catalog(monkeypatch, _spec("bad", render=exploding))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "bad", "kwargs": {}})

    assert _get_figure(client, "run-1")["reason"] == REASON_RENDER_FAILED


def test_render_returning_a_non_figure_is_render_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec("liar", render=lambda rows, params: {"not": "a figure"}))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "liar", "kwargs": {}})

    assert _get_figure(client, "run-1")["reason"] == REASON_RENDER_FAILED


def test_malformed_source_snapshot_is_params_mismatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rows_from_columnar` refuses a window claiming more rows than its run
    (`total_seen < len(rows)`); the route degrades exactly like a schema
    mismatch — the stored shape no longer matches what the code expects."""
    monkeypatch.setattr(
        live_rows,
        "get",
        lambda run_id: {
            "columns": ["m"],
            "rows": [[1.0], [2.0]],
            "partial": True,
            "total_seen": 1,  # fewer than the rows physically present: impossible
            "plan": None,
        },
    )

    body = _get_figure(client, "run-1")

    assert body["reason"] == REASON_PARAMS_MISMATCH
    assert body["panels"] == []


def test_plan_render_serves_reason_none_and_the_sources_truth(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: the plan's own figure, with `partial`/`source` overwritten
    from the source — the render's own values are placeholders by convention
    (it deliberately claims partial=True/source="live" here; the buffer is
    settled, so the route must flip `partial`)."""
    _install_catalog(monkeypatch, _spec("echo", render=_echo_render))
    _feed_live(
        "run-1",
        [{"m": 1.0, "d": 10.0}, {"m": 2.0, "d": 20.0}],
        plan={"name": "echo", "kwargs": {}},
        stop=True,
    )

    body = _get_figure(client, "run-1")

    assert body["reason"] is None
    assert body["partial"] is False  # the source's truth, not the render's placeholder
    assert body["source"] == "live"
    assert [panel["title"] for panel in body["panels"]] == ["d", "m"]


# =========================================================================
# Live settled caching
# =========================================================================


def _counting_render(calls: list[list[dict[str, Any]]]):
    def render(rows: list[dict[str, Any]], params: Any) -> Figure:
        calls.append(rows)
        return _echo_render(rows, params)

    return render


def test_settled_live_run_renders_once_then_serves_the_cache(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[dict[str, Any]]] = []
    _install_catalog(monkeypatch, _spec("echo", render=_counting_render(calls)))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "echo", "kwargs": {}}, stop=True)

    first = _get_figure(client, "run-1")
    second = _get_figure(client, "run-1")

    assert len(calls) == 1
    assert first == second


def test_partial_live_run_recomputes_every_tick(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[dict[str, Any]]] = []
    _install_catalog(monkeypatch, _spec("echo", render=_counting_render(calls)))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "echo", "kwargs": {}}, stop=False)

    _get_figure(client, "run-1")
    _get_figure(client, "run-1")

    assert len(calls) == 2  # nothing cached while the run may still grow


def test_settled_hit_revalidates_total_seen_against_the_buffer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recycled run id whose eviction has not landed yet shows up as a
    row-count mismatch between the cached entry and the buffer in hand — the
    live branch's free staleness check."""
    calls: list[list[dict[str, Any]]] = []
    _install_catalog(monkeypatch, _spec("echo", render=_counting_render(calls)))

    def snapshot(total_seen: int, value: float) -> dict[str, Any]:
        return {
            "columns": ["m"],
            "rows": [[value]] * total_seen,
            "partial": False,
            "total_seen": total_seen,
            "plan": {"name": "echo", "kwargs": {}},
        }

    monkeypatch.setattr(live_rows, "get", lambda run_id: snapshot(2, 1.0))
    _get_figure(client, "run-1")
    monkeypatch.setattr(live_rows, "get", lambda run_id: snapshot(3, 9.0))
    body = _get_figure(client, "run-1")

    assert len(calls) == 2
    assert calls[1] == [{"m": 9.0}] * 3
    assert body["panels"][0]["mark"]["series"][0]["points"][0]["y"] == 9.0


def test_a_new_start_doc_for_a_cached_run_id_recomputes_from_the_new_rows(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalidate-on-write: the document plane's eviction hook (not any
    read-side validation) is what makes a re-run under the same OSPREY run id
    serve the re-run's figure."""
    calls: list[list[dict[str, Any]]] = []
    _install_catalog(monkeypatch, _spec("echo", render=_counting_render(calls)))

    def drive(engine_uid: str, value: float) -> None:
        router = RunDocumentRouter()
        router(
            "start",
            {
                "uid": engine_uid,
                RUN_ID_META_KEY: "run-1",
                PLAN_META_KEY: {"name": "echo", "kwargs": {}},
            },
        )
        router("descriptor", {"uid": f"desc-{engine_uid}", "run_start": engine_uid})
        router("event", {"descriptor": f"desc-{engine_uid}", "data": {"m": value}})
        router("stop", {"run_start": engine_uid})

    drive("engine-1", 1.0)
    first = _get_figure(client, "run-1")
    assert first["panels"][0]["mark"]["series"][0]["points"][0]["y"] == 1.0

    drive("engine-2", 42.0)
    second = _get_figure(client, "run-1")

    assert second["panels"][0]["mark"]["series"][0]["points"][0]["y"] == 42.0
    assert len(calls) == 2


# =========================================================================
# Tiled branch
# =========================================================================


def test_tiled_settled_run_serves_the_plan_figure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec("echo", render=_echo_render))
    node = _tiled_run_node(
        "uid-1", "run-1", [{"m": 1.0, "d": 10.0}], plan={"name": "echo", "kwargs": {}}
    )
    _install_fake_tiled(monkeypatch, _FakeTiledClient({"run-1": [node]}))

    body = _get_figure(client, "run-1")

    assert body["reason"] is None
    assert body["source"] == "tiled"
    assert body["partial"] is False  # stop document present


def test_tiled_stop_doc_absent_is_partial_and_never_cached(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec("echo", render=_echo_render))
    node = _tiled_run_node(
        "uid-1", "run-1", [{"m": 1.0}], plan={"name": "echo", "kwargs": {}}, stopped=False
    )
    tiled = _FakeTiledClient({"run-1": [node]})
    _install_fake_tiled(monkeypatch, tiled)

    first = _get_figure(client, "run-1")
    second = _get_figure(client, "run-1")

    assert first["partial"] is True
    assert second["partial"] is True
    assert len(tiled.search_queries) == 2  # in-flight figures are recomputed, not cached


def test_a_settled_tiled_hit_issues_zero_tiled_calls(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache's whole point: the second GET on a settled Tiled run touches
    neither `from_uri` nor `search` — asserted with call counters, not timing."""
    _install_catalog(monkeypatch, _spec("echo", render=_echo_render))
    node = _tiled_run_node(
        "uid-1", "run-1", [{"m": 1.0, "d": 2.0}], plan={"name": "echo", "kwargs": {}}
    )
    tiled = _FakeTiledClient({"run-1": [node]})
    from_uri_calls = _install_fake_tiled(monkeypatch, tiled)

    first = _get_figure(client, "run-1")
    searches_after_first = len(tiled.search_queries)
    clients_after_first = len(from_uri_calls)

    second = _get_figure(client, "run-1")

    assert first == second
    assert len(tiled.search_queries) == searches_after_first
    assert len(from_uri_calls) == clients_after_first


def test_tiled_configured_without_api_key_passes_none(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token-less catalog is a working configuration: `from_uri` gets
    `api_key=None`, never a `KeyError` from a bare env subscript."""
    node = _tiled_run_node("uid-1", "run-1", [{"m": 1.0}])
    from_uri_calls = _install_fake_tiled(
        monkeypatch, _FakeTiledClient({"run-1": [node]}), api_key=None
    )

    body = _get_figure(client, "run-1")

    assert from_uri_calls == [("http://tiled:8000", {"api_key": None})]
    assert body["source"] == "tiled"


def test_tiled_read_failure_is_source_unavailable_not_a_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tiled")
    monkeypatch.setenv(_TILED_URI_ENV, "http://tiled:8000")

    def broken_from_uri(uri, **kwargs):
        raise ConnectionError("catalog unreachable")

    monkeypatch.setattr("tiled.client.from_uri", broken_from_uri)

    body = _get_figure(client, "run-1")

    assert body["reason"] == REASON_SOURCE_UNAVAILABLE
    assert body["source"] == "tiled"  # the store the route WAS reading
    assert body["partial"] is True  # settledness unknowable: keep polling
    assert body["panels"] == []


def test_multiple_matches_resolve_to_the_newest_start_time(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run of an interrupted item leaves two catalog nodes under one
    OSPREY run id; the figure must come from the newest start, not from
    whatever order the server answered in."""
    _install_catalog(monkeypatch, _spec("echo", render=_echo_render))
    stale = _tiled_run_node(
        "uid-old", "run-1", [{"m": 1.0}], plan={"name": "echo", "kwargs": {}}, start_time=1000.0
    )
    fresh = _tiled_run_node(
        "uid-new", "run-1", [{"m": 77.0}], plan={"name": "echo", "kwargs": {}}, start_time=2000.0
    )
    # Stale one listed FIRST: `matches[0]` would pick it.
    _install_fake_tiled(monkeypatch, _FakeTiledClient({"run-1": [stale, fresh]}))

    body = _get_figure(client, "run-1")

    assert body["panels"][0]["mark"]["series"][0]["points"][0]["y"] == 77.0


def test_live_and_tiled_produce_the_same_figure_through_the_shared_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parity: Tiled hands back NaN for a missing reading and may return rows
    out of order; the live buffer stores None in emission order. Same run,
    same figure — differing only in the `source` stamp."""
    _install_catalog(monkeypatch, _spec("echo", render=_echo_render))
    stamp = {"name": "echo", "kwargs": {}}
    logical_rows: list[dict[str, Any]] = [
        {"m": 1.0, "d": 10.0},
        {"m": 2.0, "d": None},
        {"m": 3.0, "d": 30.0},
    ]

    client = TestClient(app)
    _feed_live("run-live", logical_rows, plan=stamp, stop=True)
    live_body = _get_figure(client, "run-live")

    live_rows._clear()
    figure_cache._clear()
    tiled_rows = [
        {key: (np.nan if value is None else value) for key, value in row.items()}
        for row in logical_rows
    ]
    node = _tiled_run_node("uid-1", "run-tiled", tiled_rows, plan=stamp, row_order=[2, 0, 1])
    _install_fake_tiled(monkeypatch, _FakeTiledClient({"run-tiled": [node]}))
    tiled_body = _get_figure(client, "run-tiled")

    assert live_body.pop("source") == "live"
    assert tiled_body.pop("source") == "tiled"
    assert live_body == tiled_body


def test_tiled_start_doc_without_a_stamp_is_plan_identity_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _tiled_run_node("uid-1", "run-1", [{"m": 1.0}], plan=None)
    _install_fake_tiled(monkeypatch, _FakeTiledClient({"run-1": [node]}))

    assert _get_figure(client, "run-1")["reason"] == REASON_PLAN_IDENTITY_UNAVAILABLE


# =========================================================================
# Decimation at the serving edge
# =========================================================================


def _long_lines_figure(n: int) -> Figure:
    return Figure(
        panels=[
            Panel(
                title="long",
                mark=LinesMark(
                    series=[
                        Series(
                            label="s",
                            points=[Point(x=float(i), y=float(i)) for i in range(n)],
                            source_points=n,
                        )
                    ],
                    source_points=n,
                ),
            )
        ],
        partial=True,
        source="live",
    )


def test_plan_lines_series_are_decimated_to_the_budget(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = 5000
    _install_catalog(monkeypatch, _spec("long", render=lambda rows, params: _long_lines_figure(n)))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "long", "kwargs": {}})

    series = _get_figure(client, "run-1")["panels"][0]["mark"]["series"][0]

    assert len(series["points"]) == DEFAULT_MAX_POINTS
    assert series["decimated"] is True
    assert series["source_points"] == n
    # Endpoints survive decimation.
    assert series["points"][0]["x"] == 0.0
    assert series["points"][-1]["x"] == float(n - 1)


def test_plan_bars_are_decimated_with_categories_kept_aligned(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = 3000

    def bars_render(rows: list[dict[str, Any]], params: Any) -> Figure:
        return Figure(
            panels=[
                Panel(
                    title="bars",
                    mark=FigBarsMark(
                        categories=[f"c{i}" for i in range(n)],
                        values=[float(i) for i in range(n)],
                        source_points=n,
                    ),
                )
            ],
            partial=True,
            source="live",
        )

    _install_catalog(monkeypatch, _spec("bars", render=bars_render))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "bars", "kwargs": {}})

    mark = _get_figure(client, "run-1")["panels"][0]["mark"]

    assert len(mark["values"]) == DEFAULT_MAX_POINTS
    assert len(mark["categories"]) == DEFAULT_MAX_POINTS
    assert mark["decimated"] is True
    assert mark["source_points"] == n
    # Category/value pairs survive selection together.
    for category, value in zip(mark["categories"], mark["values"], strict=True):
        assert category == f"c{int(value)}"


def test_default_figure_series_are_decimated_too(client: TestClient) -> None:
    """The budget is the route's, not the plan's: a stampless 2500-row run's
    default figure is bounded exactly like a plan-rendered one."""
    n = 2500
    _feed_live("run-1", [{"m": float(i)} for i in range(n)], plan=None)

    series = _get_figure(client, "run-1")["panels"][0]["mark"]["series"][0]

    assert len(series["points"]) == DEFAULT_MAX_POINTS
    assert series["decimated"] is True
    assert series["source_points"] == n


def test_inconsistent_plan_decimation_truth_is_render_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A series claiming fewer source points than it holds would make
    `decimate` report a series as larger after decimation than before — plan
    garbage, degraded like any other render failure."""

    def lying_render(rows: list[dict[str, Any]], params: Any) -> Figure:
        figure = _long_lines_figure(2001)
        figure.panels[0].mark.series[0].source_points = 5  # type: ignore[union-attr]
        return figure

    _install_catalog(monkeypatch, _spec("liar", render=lying_render))
    _feed_live("run-1", [{"m": 1.0}], plan={"name": "liar", "kwargs": {}})

    assert _get_figure(client, "run-1")["reason"] == REASON_RENDER_FAILED


# =========================================================================
# Real catalog: shipped-plan resolution and the perf gate
# =========================================================================


def _orm_rows(correctors: list[str], detectors: list[str], num: int) -> list[dict[str, float]]:
    """Rows a real `orm` run emits: corrector-major, every row carrying every
    corrector's current (idle 0.0) and every detector's reading."""
    rng = np.random.default_rng(11)
    truth = rng.normal(size=(len(detectors), len(correctors)))
    span = 2.0
    step = (2 * span) / (num - 1)
    currents = [-span + i * step for i in range(num)]
    rows: list[dict[str, float]] = []
    for j, corrector in enumerate(correctors):
        for current in currents:
            row = dict.fromkeys(correctors, 0.0)
            row[corrector] = current
            orbit = truth[:, j] * current
            for i, detector in enumerate(detectors):
                row[detector] = float(orbit[i])
            rows.append(row)
    return rows


def _orm_stamp(correctors: list[str], detectors: list[str], num: int) -> dict[str, Any]:
    return {
        "name": "orm",
        "kwargs": {
            "correctors": correctors,
            "detectors": detectors,
            "span_a": 2.0,
            "num": num,
        },
    }


def test_live_run_resolves_its_plan_with_no_manager_and_no_tiled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The whole live path needs nothing but this process: the in-process
    stamp resolves against the REAL shipped catalog — no manager RPC, no
    Tiled config, no test-injected catalog."""
    pytest.importorskip("bluesky")
    monkeypatch.setenv("BLUESKY_SESSION_PLAN_DIR", str(tmp_path))

    correctors, detectors, num = ["CH0", "CH1"], ["BPM0", "BPM1"], 3
    _feed_live(
        "run-orm",
        _orm_rows(correctors, detectors, num),
        plan=_orm_stamp(correctors, detectors, num),
        stop=False,
    )

    body = _get_figure(client, "run-orm")

    assert body["reason"] is None
    assert body["source"] == "live"
    assert body["partial"] is True
    titles = [panel["title"] for panel in body["panels"]]
    assert "CH0 sweep" in titles
    assert "Response matrix" in titles


def test_orm_figure_at_facility_scale_stays_off_the_row_scan_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The plan's perf gate: 70 correctors x 100 BPMs x 1470 rows through the
    WHOLE route — buffer read, adapter, real `orm.render`, decimation, dump,
    HTTP. The run is left partial, so every GET recomputes (the live-tick cost
    this bounds); median, never max, per the CI-timing house rule.

    The bound is a REGRESSION GUARD, not a fine-grained perf assertion. What
    it exists to catch is a return to the pre-vectorized row-scan fit, which
    measured 5-77 s at this scale; the vectorized path measures ~15 ms on an
    unloaded machine. The threshold sits far above that measurement on
    purpose: this suite runs under xdist on shared CI runners, where the same
    work has been observed at 405 ms purely from contention. A threshold tight
    enough to police the ~15 ms figure would fail on runner load rather than
    on a real slowdown — it would report the weather, not the code. Two
    seconds still leaves the row-scan path no room to come back unnoticed."""
    pytest.importorskip("bluesky")
    monkeypatch.setenv("BLUESKY_SESSION_PLAN_DIR", str(tmp_path))

    correctors = [f"CH{j}" for j in range(70)]
    detectors = [f"BPM{i}" for i in range(100)]
    num = 21
    rows = _orm_rows(correctors, detectors, num)
    assert len(rows) == 1470
    _feed_live("run-perf", rows, plan=_orm_stamp(correctors, detectors, num), stop=False)

    timings = []
    for _ in range(5):
        started = time.perf_counter()
        body = _get_figure(client, "run-perf")
        timings.append(time.perf_counter() - started)

    assert body["reason"] is None
    assert statistics.median(timings) < 2.0
