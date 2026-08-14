"""Unit tests for the scan MCP client tools' request/response mapping.

Complements ``test_read_run_tools.py`` (which covers per-tool bridge-status
translation in depth) by focusing on two things across every read/allow-listed
client tool: that each tool builds the right bridge request and maps a normal
response straight through, and that a genuinely unreachable bridge surfaces
the standard ``bluesky_bridge_unreachable`` error envelope raised by the
module-level ``_http_get_json``/``_http_post_json`` primitives — patched here
(phoebus pattern, no network) so tools never need their own try/except
around that failure mode.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.tools import read_tools
from osprey.mcp_server.bluesky.tools.read_tools import FIGURE_POINT_BUDGET
from osprey.mcp_server.errors import make_error
from osprey.services.bluesky_bridge.figure import (
    FIGURE_REASONS,
    REASON_SOURCE_UNAVAILABLE,
    BarsMark,
    Figure,
    HeatmapMark,
    LinesMark,
    Panel,
    Point,
    Series,
)
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.read_tools"


def _fn(name):
    return get_tool_fn(getattr(read_tools, name))


def _unreachable(*_args, **_kwargs):
    """Simulate the real _http_* primitive's behavior on a connection failure."""
    make_error(
        "bluesky_bridge_unreachable",
        "Could not reach the Bluesky bridge: connection refused",
        ["Confirm the facility Bluesky bridge process is running."],
    )


# ---------------------------------------------------------------------------
# get_run — GET /runs/{id}
# ---------------------------------------------------------------------------


async def test_get_run_request_response_mapping(monkeypatch):
    captured = {}

    record = {
        "id": "run-1",
        "status": "running",
        "plan_name": "orm",
        "plan_args": {"num_points": 5},
        "item_uid": "u1",
        "progress": {
            "rows_seen": 3,
            "expected_points": 10,
            "fraction": 0.3,
            "complete": False,
        },
    }

    def fake_get(path, **kwargs):
        captured["path"] = path
        return 200, record

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)

    result = await _fn("get_run")(run_id="run-1")

    assert captured["path"] == "/runs/run-1"
    # Relayed whole: the tool reshapes nothing, so a key the bridge adds
    # reaches the agent without a code change here.
    assert extract_response_dict(result) == record


async def test_get_run_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("get_run")(run_id="run-1")


def test_get_run_docstring_names_every_key_the_run_record_can_emit(monkeypatch):
    """The docstring IS the contract: it's the only description of the JSON
    run record an agent ever sees, and nothing in the bridge/MCP path is typed
    to catch drift between it and `runs.record_from_item` — this is exactly how
    `tiled_degraded` slipped through unseen originally. Exercises the record
    builder across every status it can report, in both the has-it and lacks-it
    direction for each optional key, and asserts every key any of them can emit
    is literally named (quoted) in `get_run`'s docstring, so a future key added
    to the record without a docstring update fails here rather than staying
    silently invisible.
    """
    from osprey.services.bluesky_bridge import document_plane, runs

    # `progress` is the one optional key sourced from outside the item, so the
    # document plane is stubbed rather than driven — this test is about which
    # keys the record can carry, not about how progress is computed.
    monkeypatch.setattr(
        document_plane,
        "progress",
        lambda run_id: {
            "rows_seen": 3,
            "expected_points": 10,
            "fraction": 0.3,
            "complete": False,
        },
    )

    pending = {
        "item_uid": "u1",
        "name": "orm",
        "kwargs": {"num_points": 5},
        "meta": {"osprey_run_id": "r1"},
    }
    finished = {
        **pending,
        "result": {"run_uids": ["abc-123"], "exit_status": "completed"},
    }
    failed = {**finished, "result": {"exit_status": "failed", "msg": "device timeout"}}
    # An item enqueued out of band: no osprey_run_id, so no progress lookup —
    # covers the absent-optional direction rather than only the present one.
    bare = {"name": "orm", "kwargs": {}}

    all_keys: set[str] = set()
    for item, status in (
        (pending, runs.STATUS_PENDING),
        (pending, runs.STATUS_RUNNING),
        (finished, runs.STATUS_COMPLETED),
        (finished, runs.STATUS_STOPPED),
        (failed, runs.STATUS_ERROR),
        (bare, runs.STATUS_PENDING),
    ):
        all_keys.update(runs.record_from_item(item, status).keys())

    # Guard against the assertion going vacuous if the record ever stops
    # carrying its optional keys: every documented key must actually appear.
    assert {"id", "status", "plan_name", "plan_args"} <= all_keys
    assert {"item_uid", "run_uid", "error", "progress"} <= all_keys

    doc = read_tools.get_run.__doc__ or ""
    missing = {key for key in all_keys if f'"{key}"' not in doc}
    assert not missing, f"get_run docstring is missing keys the run record emits: {missing}"


def test_get_run_docstring_does_not_still_promise_retired_keys():
    """The inverse drift: a key the record can no longer emit must not linger.

    ``completion``/``tiled_degraded``/``launched_by`` were on the old record.
    A docstring that still promises them sends the agent looking for fields
    that will never arrive — the same failure as a missing key, pointed the
    other way, and one the test above cannot catch.
    """
    doc = read_tools.get_run.__doc__ or ""
    for retired in ("completion", "tiled_degraded", "launched_by"):
        assert f'"{retired}"' not in doc, (
            f"get_run docstring still promises the retired key {retired!r}"
        )


# ---------------------------------------------------------------------------
# list_plans — GET /plans
# ---------------------------------------------------------------------------


async def test_list_plans_request_response_mapping(monkeypatch):
    captured = {}
    plans = [{"name": "count", "params": {}}, {"name": "grid_scan", "params": {"motors": []}}]

    def fake_get(path, **kwargs):
        captured["path"] = path
        return 200, plans

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)

    result = await _fn("list_plans")()

    assert captured["path"] == "/plans"
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["plans"] == plans


async def test_list_plans_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("list_plans")()


# ---------------------------------------------------------------------------
# list_devices — GET /devices
# ---------------------------------------------------------------------------


async def test_list_devices_request_response_mapping(monkeypatch):
    captured = {}
    devices = [{"name": "BPM1", "is_readable": True}, {"name": "COR1", "is_movable": True}]

    def fake_get(path, **kwargs):
        captured["path"] = path
        return 200, devices

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)

    result = await _fn("list_devices")()

    assert captured["path"] == "/devices"
    data = extract_response_dict(result)
    assert data["status"] == "success"
    # Relayed whole: a flag the bridge starts reporting reaches the agent
    # without a code change here.
    assert data["devices"] == devices


async def test_list_devices_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("list_devices")()


# ---------------------------------------------------------------------------
# list_runs — GET /runs
# ---------------------------------------------------------------------------


async def test_list_runs_request_response_mapping(monkeypatch):
    captured = {}
    runs = [{"id": "run-2", "status": "completed"}, {"id": "run-1", "status": "stopped"}]

    def fake_get(path, **kwargs):
        captured["path"] = path
        return 200, runs

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)

    result = await _fn("list_runs")(limit=5)

    assert captured["path"] == "/runs?limit=5"
    data = extract_response_dict(result)
    assert data["status"] == "success"
    assert data["runs"] == runs


async def test_list_runs_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("list_runs")()


# ---------------------------------------------------------------------------
# get_run_data — GET /runs/{id}/data (bounded stub)
# ---------------------------------------------------------------------------


async def test_get_run_data_request_response_mapping(monkeypatch):
    captured = {}
    body = {
        "run_uid": "uid-1",
        "columns": ["m1", "det1"],
        "rows": [[0.0, 1.1], [1.0, 2.2]],
        "row_count": 2,
        "truncated": False,
        "partial": True,
    }

    def fake_get(path, **kwargs):
        captured["path"] = path
        return 200, body

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)

    result = await _fn("get_run_data")(run_id="run-1", max_rows=25, offset=10, tail=False)

    assert captured["path"] == "/runs/run-1/data?max_rows=25&offset=10"
    data = extract_response_dict(result)
    assert data == body


async def test_get_run_data_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("get_run_data")(run_id="run-1")


# ---------------------------------------------------------------------------
# get_run_figure — GET /runs/{id}/figure (point-bounded projection)
#
# The tool is the one read that does NOT relay its body: the route serves
# heatmaps whole (a canvas can afford 7000 cells; an agent's context cannot),
# so these pin the projection's promise — one 2000-point budget for the whole
# figure, grids over 40x40 summarized, and every count still honest about what
# the run actually took.
# ---------------------------------------------------------------------------


def _serve(monkeypatch, body, status=200):
    """Point the tool's HTTP primitive at ``body``; return the captured path."""
    captured = {}

    def fake_get(path, **kwargs):
        captured["path"] = path
        return status, body

    monkeypatch.setattr(f"{_MOD}._http_get_json", fake_get)
    return captured


def _lines_panel(title="Sweep", *, series):
    return Panel(title=title, mark=LinesMark(**_lines_mark_kwargs(series)))


def _lines_mark_kwargs(series):
    return {
        "series": series,
        "decimated": any(s.decimated for s in series),
        "source_points": sum(s.source_points for s in series),
    }


def _series(label, n, *, source_points=None, decimated=False, gap_at=None):
    points = [Point(x=float(i), y=None if i == gap_at else float(i)) for i in range(n)]
    return Series(
        label=label,
        points=points,
        decimated=decimated,
        source_points=n if source_points is None else source_points,
    )


def _heatmap_panel(rows, columns, *, title="Grid", values=None):
    grid = values or [[float(j * columns + i) for i in range(columns)] for j in range(rows)]
    return Panel(
        title=title,
        mark=HeatmapMark(
            x_labels=[f"C{i}" for i in range(columns)],
            y_labels=[f"B{j}" for j in range(rows)],
            values=grid,
            value_label="Response slope",
            value_units="mm/A",
            source_points=rows * columns,
        ),
    )


def _figure(*panels, partial=False, source="live", reason=None):
    return Figure(panels=list(panels), partial=partial, source=source, reason=reason).model_dump()


def _total_points(projected):
    """Every value the projection actually returned, across all panels."""
    total = 0
    for panel in projected["panels"]:
        mark = panel["mark"]
        if mark["kind"] == "lines":
            total += sum(len(s["points"]) for s in mark["series"])
        elif mark["kind"] == "bars":
            total += len(mark["values"])
        elif mark["kind"] == "heatmap":
            total += sum(len(row) for row in mark["values"])
    return total


async def test_get_run_figure_request_response_mapping(monkeypatch):
    """A figure already under budget is relayed shape-for-shape, panels intact."""
    body = _figure(
        _lines_panel(series=[_series("det1", 3, gap_at=1)]),
        partial=True,
        source="live",
        reason=None,
    )
    captured = _serve(monkeypatch, body)

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    assert captured["path"] == "/runs/run-1/figure"
    assert (projected["partial"], projected["source"], projected["reason"]) == (True, "live", None)
    assert projected["panels"][0]["mark"]["series"][0]["points"] == [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": None},  # a gap stays a gap, never a zero
        {"x": 2.0, "y": 2.0},
    ]
    assert projected["point_budget"] == FIGURE_POINT_BUDGET
    assert projected["points_returned"] == 3


async def test_get_run_figure_unknown_run(monkeypatch):
    """404 maps to unknown_run exactly as get_run_data does."""
    _serve(monkeypatch, {"detail": "unknown run 'nope'"}, status=404)
    with assert_raises_error(error_type="unknown_run"):
        await _fn("get_run_figure")(run_id="nope")


async def test_get_run_figure_bridge_error(monkeypatch):
    _serve(monkeypatch, {"detail": "boom"}, status=503)
    with assert_raises_error(error_type="bluesky_bridge_error"):
        await _fn("get_run_figure")(run_id="run-1")


async def test_get_run_figure_unreachable(monkeypatch):
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with assert_raises_error(error_type="bluesky_bridge_unreachable"):
        await _fn("get_run_figure")(run_id="run-1")


async def test_get_run_figure_relays_partial_source_and_any_reason(monkeypatch):
    """A reason is a default view, not an error — including one this build has
    never heard of. The route deliberately leaves ``reason`` unvalidated so an
    unforeseen value still serves a figure; the tool must not narrow that."""
    body = _figure(
        _lines_panel(series=[_series("det1", 2)]),
        partial=True,
        source="tiled",
        reason="a_reason_from_a_newer_bridge",
    )
    _serve(monkeypatch, body)

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    assert projected["reason"] == "a_reason_from_a_newer_bridge"
    assert projected["partial"] is True
    assert projected["source"] == "tiled"


async def test_get_run_figure_source_unavailable_is_a_200_with_no_panels(monkeypatch):
    """An unreadable source is an empty figure with a reason, never a refusal."""
    _serve(monkeypatch, _figure(reason=REASON_SOURCE_UNAVAILABLE))

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    assert projected["panels"] == []
    assert projected["reason"] == REASON_SOURCE_UNAVAILABLE
    assert projected["points_returned"] == 0


# --- the budget ------------------------------------------------------------


async def test_budget_is_spent_across_the_whole_figure_not_per_series(monkeypatch):
    """Six 2000-point series in two panels come back as 2000 points TOTAL.

    The route's bound is per series; this tool's is per figure, which is the
    whole reason it exists — six panel-legal series are six times the context.
    """
    _serve(
        monkeypatch,
        _figure(
            _lines_panel(title="A", series=[_series(f"a{i}", 2000) for i in range(3)]),
            _lines_panel(title="B", series=[_series(f"b{i}", 2000) for i in range(3)]),
        ),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    kept = [len(s["points"]) for panel in projected["panels"] for s in panel["mark"]["series"]]
    assert sum(kept) == FIGURE_POINT_BUDGET
    assert projected["points_returned"] == FIGURE_POINT_BUDGET
    # Evenly split, to within the one point integer division leaves over.
    assert max(kept) - min(kept) <= 1


async def test_a_series_needing_less_than_its_share_hands_the_rest_back(monkeypatch):
    """Max-min fairness: a 5-point series does not reserve 1000 points.

    An even split alone would leave 995 points unspent while the long series
    is thinned to half of what the figure could have shown.
    """
    _serve(
        monkeypatch,
        _figure(_lines_panel(series=[_series("short", 5), _series("long", 50_000)])),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    short, long = projected["panels"][0]["mark"]["series"]
    assert len(short["points"]) == 5
    assert len(long["points"]) == FIGURE_POINT_BUDGET - 5
    assert short["decimated"] is False
    assert long["decimated"] is True


async def test_restriding_composes_with_the_routes_own_decimation(monkeypatch):
    """An already-thinned series still reports what the RUN took, not what the
    route handed us. Dropping decimate's prior-truth arguments would make this
    series claim 2000 source points — a figure saying it drew everything."""
    _serve(
        monkeypatch,
        _figure(
            _lines_panel(
                series=[_series("det1", 2000, source_points=200_000, decimated=True)],
            )
        ),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    series = projected["panels"][0]["mark"]["series"][0]
    assert series["source_points"] == 200_000
    assert series["decimated"] is True
    assert projected["panels"][0]["mark"]["source_points"] == 200_000


async def test_mark_level_truth_is_a_floor_the_projection_never_lowers(monkeypatch):
    """A projection may RAISE the loss a figure reports, never lower it.

    A plan can set mark-level truth its series do not individually carry —
    rows it dropped before building them — and recomputing the aggregates
    from the series alone would quietly erase that, turning a partial view
    into one that claims to have drawn everything.
    """
    body = _figure(_lines_panel(series=[_series("det1", 10)]))
    body["panels"][0]["mark"]["decimated"] = True
    body["panels"][0]["mark"]["source_points"] = 999_999

    _serve(monkeypatch, body)

    mark = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))["panels"][0]["mark"]

    assert mark["decimated"] is True
    assert mark["source_points"] == 999_999
    # The series' own truth is untouched — only the mark aggregates are floored.
    assert mark["series"][0]["source_points"] == 10


async def test_the_last_points_go_to_series_that_have_data(monkeypatch):
    """An empty series must not spend one of the scarce single points.

    Under starvation every unit is fighting for one point each; letting a
    series with nothing to show take one would cost a series that has data
    its only place in the figure.
    """
    empty = [_series(f"empty{i}", 0) for i in range(100)]
    measured = [_series(f"det{i}", 3) for i in range(400)]
    _serve(
        monkeypatch,
        _figure(
            _heatmap_panel(40, 40),  # 1600 cells, leaving exactly 400 points
            _lines_panel(series=[*empty, *measured]),
        ),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    lines_panel = projected["panels"][1]
    kept = {series["label"]: len(series["points"]) for series in lines_panel["mark"]["series"]}
    assert len(kept) == 500, "no series was dropped — the empty ones cost nothing"
    assert all(kept[f"det{i}"] == 1 for i in range(400)), "a measured series was starved"
    assert lines_panel["annotations"] == []
    assert _total_points(projected) == FIGURE_POINT_BUDGET


async def test_endpoints_and_gaps_survive_the_restride(monkeypatch):
    """The x range is never shortened and a dropout is never smoothed over."""
    _serve(
        monkeypatch,
        _figure(_lines_panel(series=[_series("det1", 10_000, gap_at=4_321)])),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    points = projected["panels"][0]["mark"]["series"][0]["points"]
    assert (points[0]["x"], points[-1]["x"]) == (0.0, 9_999.0)
    assert any(point["y"] is None for point in points)


async def test_series_past_the_budget_are_omitted_and_said_so(monkeypatch):
    """More series than points: the ones that fit are kept, the rest are named.

    Nothing in the mark shape could report a series that simply is not there,
    so the loss is a panel annotation — a sentence, per the spec's rule for
    what annotations are.
    """
    _serve(
        monkeypatch,
        _figure(
            _heatmap_panel(40, 40),  # 1600 cells, leaving 400 points
            _lines_panel(series=[_series(f"s{i}", 3) for i in range(500)]),
        ),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    lines_panel = projected["panels"][1]
    assert len(lines_panel["mark"]["series"]) == 400
    assert lines_panel["mark"]["decimated"] is True
    assert "100 of 500 series are not shown" in lines_panel["annotations"][0]
    assert _total_points(projected) <= FIGURE_POINT_BUDGET


# --- heatmaps: the projection's real payload problem -----------------------


async def test_a_facility_scale_heatmap_is_summarized_and_costs_nothing(monkeypatch):
    """A 70x100 response matrix is 7000 cells the agent cannot read anyway.

    Summarizing it is what actually bounds this projection (the route serves
    heatmaps whole, in spec), and because the summary is fixed-size it spends
    none of the budget: the line series still get all 2000 points.
    """
    grid = [[float(j - i) for i in range(100)] for j in range(70)]
    grid[3][7] = -999.0  # the extreme cell, by magnitude and by minimum
    _serve(
        monkeypatch,
        _figure(
            _heatmap_panel(70, 100, title="Response matrix", values=grid),
            _lines_panel(series=[_series("det1", 50_000)]),
        ),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    grid_panel, lines_panel = projected["panels"]
    summary = grid_panel["mark"]
    assert summary["kind"] == "heatmap_summary"
    assert (summary["rows"], summary["columns"], summary["cells"]) == (70, 100, 7000)
    assert summary["empty_cells"] == 0
    assert (summary["value_label"], summary["value_units"]) == ("Response slope", "mm/A")
    assert summary["x_labels"] == {"count": 100, "first": "C0", "last": "C99"}
    assert summary["min"] == {"value": -999.0, "x_label": "C7", "y_label": "B3"}
    assert summary["largest_magnitude"][0] == summary["min"]
    assert len(summary["largest_magnitude"]) == 5
    # Lossy, and says so in the same vocabulary every other mark uses.
    assert (summary["decimated"], summary["source_points"]) == (True, 7000)
    assert "larger than the 40x40" in grid_panel["annotations"][0]
    # Costs nothing: the series keep the entire budget.
    assert len(lines_panel["mark"]["series"][0]["points"]) == FIGURE_POINT_BUDGET


async def test_heatmap_summary_reports_cells_never_measured(monkeypatch):
    """A cell the run never took is counted, not folded in as a zero."""
    grid = [[None if i == 0 else float(i) for i in range(50)] for _ in range(50)]
    _serve(monkeypatch, _figure(_heatmap_panel(50, 50, values=grid)))

    summary = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))["panels"][0][
        "mark"
    ]

    assert summary["cells"] == 2500
    assert summary["empty_cells"] == 50
    assert summary["min"]["value"] == 1.0  # the Nones are absent, not minimal


async def test_a_grid_at_the_limit_passes_through_as_cells_and_is_charged(monkeypatch):
    """40x40 is returned cell by cell, and its 1600 cells come OUT of the budget.

    Passing a small grid through whole and then spending the budget again on
    series would put 3600 points in front of the agent under a 2000 promise.
    """
    _serve(
        monkeypatch,
        _figure(
            _heatmap_panel(40, 40),
            _lines_panel(series=[_series("det1", 50_000)]),
        ),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    assert projected["panels"][0]["mark"]["kind"] == "heatmap"
    assert projected["panels"][0]["annotations"] == []
    assert len(projected["panels"][1]["mark"]["series"][0]["points"]) == 400
    assert projected["points_returned"] == FIGURE_POINT_BUDGET
    assert _total_points(projected) == FIGURE_POINT_BUDGET


async def test_one_row_over_the_limit_summarizes(monkeypatch):
    """The limit is on either dimension: 41x40 is a summary, 40x40 is cells."""
    _serve(monkeypatch, _figure(_heatmap_panel(41, 40)))

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    assert projected["panels"][0]["mark"]["kind"] == "heatmap_summary"


async def test_a_small_grid_that_does_not_fit_is_summarized_for_that_reason(monkeypatch):
    """Two 40x40 grids are 3200 cells. The second is summarized because the
    budget ran out, and the annotation says budget rather than size — the
    agent should not read it as an oversized grid."""
    _serve(monkeypatch, _figure(_heatmap_panel(40, 40), _heatmap_panel(40, 40, title="Second")))

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    first, second = projected["panels"]
    assert first["mark"]["kind"] == "heatmap"
    assert second["mark"]["kind"] == "heatmap_summary"
    assert "do not fit what is left of the figure's point budget" in second["annotations"][0]
    assert _total_points(projected) == 1600


async def test_a_grid_that_measured_nothing_summarizes_to_no_extremes(monkeypatch):
    """A grid of nothing has no minimum, no maximum and no largest cell.

    Reporting 0.0 for any of them would put a reading in front of the
    operator that the run never took.
    """
    _serve(
        monkeypatch,
        _figure(_heatmap_panel(50, 50, values=[[None] * 50 for _ in range(50)])),
    )

    summary = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))["panels"][0][
        "mark"
    ]

    assert (summary["min"], summary["max"]) == (None, None)
    assert summary["largest_magnitude"] == []
    assert (summary["cells"], summary["empty_cells"]) == (2500, 2500)


async def test_a_ragged_grid_is_measured_by_what_is_actually_there(monkeypatch):
    """Rows of differing length, and more cells than labels, are wire-legal.

    ``columns`` takes the widest row (the size check errs toward summarizing)
    while ``cells`` counts what the grid actually holds. A cell past the end
    of the label list has no name, and says so rather than borrowing one.
    """
    values = [[1.0, 2.0, 90.0], [3.0], *([[4.0, 5.0]] * 39)]
    panel = _heatmap_panel(41, 3, values=values)
    panel.mark.x_labels = ["C0", "C1"]  # one column short of the widest row
    _serve(monkeypatch, _figure(panel))

    summary = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))["panels"][0][
        "mark"
    ]

    assert (summary["rows"], summary["columns"]) == (41, 3)
    assert summary["cells"] == 3 + 1 + 39 * 2
    assert summary["max"] == {"value": 90.0, "x_label": None, "y_label": "B0"}
    assert summary["x_labels"] == {"count": 2, "first": "C0", "last": "C1"}


async def test_a_tall_thin_grid_is_over_the_limit(monkeypatch):
    """41x1 is 41 cells and would fit the budget easily — the limit is on the
    dimension, not the total, because a 41-row grid is already past what an
    agent reads cell by cell."""
    _serve(monkeypatch, _figure(_heatmap_panel(41, 1)))

    mark = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))["panels"][0]["mark"]

    assert mark["kind"] == "heatmap_summary"
    assert (mark["rows"], mark["columns"]) == (41, 1)


async def test_a_grid_is_never_returned_half_measured(monkeypatch):
    """Striding cells out of a grid would read as missing measurements, so a
    heatmap is all cells or a summary — never a grid with holes in it."""
    _serve(monkeypatch, _figure(_heatmap_panel(30, 30)))

    mark = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))["panels"][0]["mark"]

    assert [len(row) for row in mark["values"]] == [30] * 30


# --- bars ------------------------------------------------------------------


async def test_bars_are_restrided_with_their_categories(monkeypatch):
    """Bars ride through the same stride as lines, and the surviving values
    stay aligned with the categories they belong to."""
    values = [float(i) for i in range(5_000)]
    _serve(
        monkeypatch,
        _figure(
            Panel(
                title="Corrector anomaly",
                mark=BarsMark(
                    label="anomaly",
                    categories=[f"C{i}" for i in range(5_000)],
                    values=values,
                    source_points=5_000,
                ),
            )
        ),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    mark = projected["panels"][0]["mark"]
    assert len(mark["values"]) == FIGURE_POINT_BUDGET
    assert mark["decimated"] is True
    assert mark["source_points"] == 5_000
    assert mark["categories"][0] == "C0" and mark["categories"][-1] == "C4999"
    assert all(
        category == f"C{int(value)}"
        for category, value in zip(mark["categories"], mark["values"], strict=True)
    )


async def test_bars_share_the_budget_with_series(monkeypatch):
    """A bars mark is one more claimant on the same budget, not a free panel."""
    _serve(
        monkeypatch,
        _figure(
            Panel(
                title="Anomaly",
                mark=BarsMark(
                    categories=[f"C{i}" for i in range(100)],
                    values=[float(i) for i in range(100)],
                    source_points=100,
                ),
            ),
            _lines_panel(series=[_series("det1", 50_000)]),
        ),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    assert len(projected["panels"][0]["mark"]["values"]) == 100
    assert len(projected["panels"][1]["mark"]["series"][0]["points"]) == 1_900
    assert projected["points_returned"] == FIGURE_POINT_BUDGET


# --- totality --------------------------------------------------------------


async def test_a_figure_this_tool_cannot_read_is_an_error_not_a_relay(monkeypatch):
    """The one thing this tool promises never to do is hand over an unbounded
    figure, so an unreadable body refuses rather than falling back to the raw
    payload — and says where to look instead."""
    _serve(monkeypatch, {"panels": [{"title": "t", "mark": {"kind": "spirograph"}}]})

    with assert_raises_error(error_type="bluesky_bridge_error"):
        await _fn("get_run_figure")(run_id="run-1")


async def test_the_refusal_message_is_bounded_like_the_figure_is(monkeypatch):
    """The error path is the last place the bound could leak.

    Pydantic reports a validation failure once per offending node, so a single
    extra field on Point across 12x2000 points is tens of thousands of errors
    and megabytes of message — handed to the agent as an error, which is
    exactly the payload this tool exists to keep out of its context. Only the
    count and the first few locations tell an operator anything.
    """
    body = _figure(
        _lines_panel(series=[_series(f"det{i}", 2_000) for i in range(12)]),
    )
    for series in body["panels"][0]["mark"]["series"]:
        for point in series["points"]:
            point["z"] = 1.0

    _serve(monkeypatch, body)

    with pytest.raises(ToolError) as exc_info:
        await _fn("get_run_figure")(run_id="run-1")
    envelope = json.loads(str(exc_info.value))
    assert len(envelope["error_message"]) < 600, "the refusal itself blew the context budget"
    assert "24000 validation errors" in envelope["error_message"]


async def test_a_mark_lying_about_its_decimation_refuses_rather_than_raising(monkeypatch):
    """A series claiming fewer source points than it carries is a bug in the
    figure, not something to repair silently: decimate rejects it, and the tool
    turns that into the standard envelope instead of an unhandled error."""
    body = _figure(_lines_panel(series=[_series("det1", 10)]))
    body["panels"][0]["mark"]["series"][0]["source_points"] = 3

    _serve(monkeypatch, body)

    with pytest.raises(ToolError) as exc_info:
        await _fn("get_run_figure")(run_id="run-1")
    envelope = json.loads(str(exc_info.value))
    assert envelope["error_type"] == "bluesky_bridge_error"
    assert "get_run_data" in " ".join(envelope["suggestions"])


async def test_a_figure_with_nothing_in_it_yet_keeps_its_empty_marks(monkeypatch):
    """A run that has not produced rows yet has empty series, not missing ones.

    Nothing to draw costs nothing, so an empty mark is never reported as one
    the budget could not reach — an operator reading "not shown" would go
    looking for a detector that is in fact simply not there yet.
    """
    _serve(
        monkeypatch,
        _figure(
            _lines_panel(series=[_series("det1", 0)]),
            Panel(title="Anomaly", mark=BarsMark(source_points=0)),
            partial=True,
        ),
    )

    projected = extract_response_dict(await _fn("get_run_figure")(run_id="run-1"))

    lines_panel, bars_panel = projected["panels"]
    assert lines_panel["mark"]["series"][0]["points"] == []
    assert lines_panel["annotations"] == []
    assert bars_panel["mark"]["values"] == []
    assert bars_panel["annotations"] == []
    assert projected["points_returned"] == 0


# --- the docstring is the contract -----------------------------------------


def test_docstring_narrates_every_reason_the_bridge_can_send():
    """The docstring is the only description of a reason the agent ever sees.

    A reason added to the bridge without a line here reaches the operator as a
    bare code, or worse as "the plan failed" — the default view is real data,
    and narrating it as an error is the failure mode this pins against.
    """
    doc = read_tools.get_run_figure.__doc__
    for reason in FIGURE_REASONS:
        assert f'"{reason}"' in doc, f"{reason} is not narrated in get_run_figure's docstring"
    assert "Point-bounded by design: this never returns an unbounded figure." in doc
    assert "DEFAULT VIEW" in doc
    assert "heatmap_summary" in doc


# ---------------------------------------------------------------------------
# get_tool_fn unwrapping sanity check
# ---------------------------------------------------------------------------


async def test_all_client_tools_are_registered_fastmcp_function_tools():
    """Every client tool is registered on the scan server as a FunctionTool."""
    for name in (
        "get_run",
        "list_plans",
        "list_devices",
        "list_runs",
        "get_run_data",
        "get_run_figure",
    ):
        tool = await mcp.get_tool(name)
        assert tool.fn is getattr(read_tools, name), f"{name} was not registered via @mcp.tool()"
        assert get_tool_fn(tool) is tool.fn


async def test_unreachable_envelope_is_a_tool_error_with_standard_shape(monkeypatch):
    """The propagated exception is a ToolError carrying the full standard envelope."""
    monkeypatch.setattr(f"{_MOD}._http_get_json", _unreachable)
    with pytest.raises(ToolError) as exc_info:
        await _fn("list_runs")()
    envelope = json.loads(str(exc_info.value))
    assert envelope["error"] is True
    assert envelope["error_type"] == "bluesky_bridge_unreachable"
    assert envelope["suggestions"]
