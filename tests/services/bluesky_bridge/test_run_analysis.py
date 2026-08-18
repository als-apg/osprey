"""Tests for a settled run's peak statistics — `analysis.py` and its wiring.

SC-8: a settled single-movable run's payload carries center/width/center-of-mass
per readable channel as plain JSON numbers, and a run that cannot be analyzed
carries an ABSENT result naming the reason — never an error, never a fabricated
number.

Two layers are exercised, deliberately apart:

- `analysis.analyze` on its own: what a run must declare before statistics mean
  anything (exactly one movable, at least one readable), how a channel's column
  is found (the shared `plan_fields.resolve_column`, so both the connector's
  ``"motor1"`` spelling and a child-signal ``"motor1-readback"`` work), and the
  pre-filter — a pair is kept only when BOTH its readings are finite numbers,
  because a substituted y would move the center of mass toward a reading that
  never happened.
- The `GET /runs/{id}/data` wiring: statistics ride on the run's payload, are
  computed over the whole run rather than the caller's window, appear only once
  the source says the run has settled, and are computed once per settled run.

The stock statistics module is imported lazily inside `analyze`; the
import-clean gate (`test_app_import_clean.py`) is what pins that, and one test
here pins the *behavior* when the import fails — an absence with a reason
rather than a 500 on a read route.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from osprey.services.bluesky_bridge import analysis, figure_cache, live_rows
from osprey.services.bluesky_bridge import app as app_module
from osprey.services.bluesky_bridge.app import app
from osprey.services.bluesky_bridge.plan_fields import (
    MovableChannel,
    ReadableChannels,
    channel_roles,
)
from osprey.services.bluesky_bridge.plan_loader import FacilityPlans
from osprey.services.bluesky_bridge.plan_types import PlanSpec

_TILED_URI_ENV = "BLUESKY_TILED_URI"


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch):
    live_rows._clear()
    figure_cache._clear()
    analysis._clear()
    monkeypatch.delenv(_TILED_URI_ENV, raising=False)
    yield
    live_rows._clear()
    figure_cache._clear()
    analysis._clear()


# =========================================================================
# Data helpers
# =========================================================================


def _gaussian_rows(count: int = 21, *, center: float = 0.0, sigma: float = 1.0) -> list[list[Any]]:
    """``[x, y]`` rows tracing a gaussian — a peak with a knowable center."""
    span = 4.0 * sigma
    return [
        [
            x := center - span + (2 * span) * index / (count - 1),
            math.exp(-(((x - center) / sigma) ** 2) / 2),
        ]
        for index in range(count)
    ]


def _analyze(
    columns: list[str],
    rows: list[list[Any]],
    movable: list[str],
    readable: list[str],
) -> dict[str, Any]:
    return analysis.analyze(columns, rows, movable, readable)


def _channel(payload: dict[str, Any], name: str) -> dict[str, Any]:
    """One channel's entry, asserting it is there at all."""
    entries = [entry for entry in payload["channels"] if entry["channel"] == name]
    assert entries, f"{name!r} not in {[entry['channel'] for entry in payload['channels']]}"
    return entries[0]


# =========================================================================
# --- What a run must declare before statistics mean anything ---
# =========================================================================


def test_a_declared_single_movable_run_gets_statistics() -> None:
    payload = _analyze(["motor", "det"], _gaussian_rows(), ["motor"], ["det"])

    assert payload["available"] is True
    assert payload["reason"] is None
    assert payload["x_channel"] == "motor"
    assert payload["x_column"] == "motor"
    assert payload["points"] == 21
    det = _channel(payload, "det")
    assert det["reason"] is None
    assert det["points"] == 21
    assert det["center"] == pytest.approx(0.0, abs=0.05)
    # FWHM of a unit gaussian is 2*sqrt(2 ln 2) ~ 2.3548.
    assert det["width"] == pytest.approx(2.3548, abs=0.1)
    assert det["center_of_mass"] == pytest.approx(0.0, abs=0.05)


def test_a_plan_that_declares_no_movable_has_no_independent_variable() -> None:
    payload = _analyze(["motor", "det"], _gaussian_rows(), [], ["det"])

    assert payload["available"] is False
    assert payload["reason"] == analysis.REASON_NO_DECLARED_MOVABLE
    assert payload["channels"] == []


def test_a_plan_that_declares_no_readable_has_nothing_to_fit() -> None:
    payload = _analyze(["motor", "det"], _gaussian_rows(), ["motor"], [])

    assert payload["available"] is False
    assert payload["reason"] == analysis.REASON_NO_DECLARED_READABLE


def test_several_movables_are_skipped_rather_than_fitted_against_one_of_them() -> None:
    """A serial sweep (the orbit response measurement) has no single x axis.

    Stricter than the default figure on purpose: the figure asks whether the
    declared movables RESOLVE to one column, this asks whether the plan drove
    one channel at all. A peak fitted against one of several driven channels is
    a number about nothing.
    """
    payload = _analyze(["cor1", "cor2", "bpm1"], _gaussian_rows(), ["cor1", "cor2"], ["bpm1"])

    assert payload["available"] is False
    assert payload["reason"] == analysis.REASON_MULTIPLE_MOVABLE_CHANNELS


def test_a_movable_that_never_reached_the_data_is_absent_with_its_own_reason() -> None:
    payload = _analyze(["det"], _gaussian_rows(), ["motor"], ["det"])

    assert payload["available"] is False
    assert payload["reason"] == analysis.REASON_MOVABLE_COLUMN_MISSING


def test_a_run_shorter_than_the_minimum_is_not_fitted() -> None:
    rows = _gaussian_rows()[: analysis.MIN_POINTS - 1]
    payload = _analyze(["motor", "det"], rows, ["motor"], ["det"])

    assert payload["available"] is False
    assert payload["reason"] == analysis.REASON_INSUFFICIENT_POINTS


def test_an_empty_run_is_absent_rather_than_a_zero_center() -> None:
    payload = _analyze(["motor", "det"], [], ["motor"], ["det"])

    assert payload["available"] is False
    assert payload["reason"] == analysis.REASON_INSUFFICIENT_POINTS


# =========================================================================
# --- Finding a channel's column ---
# =========================================================================


def test_a_child_signal_column_is_found_through_the_shared_resolver() -> None:
    """A mock device reads back under ``"<channel>-<signal>"``.

    The whole contract exists to kill name-guessing: this is
    `plan_fields.resolve_column`'s prefix rule, not a rule this module owns.
    """
    rows = _gaussian_rows()
    payload = _analyze(["motor1-readback", "bpm1-value"], rows, ["motor1"], ["bpm1"])

    assert payload["available"] is True
    assert payload["x_column"] == "motor1-readback"
    assert _channel(payload, "bpm1")["column"] == "bpm1-value"


def test_a_readable_with_no_column_is_reported_beside_the_ones_that_have_one() -> None:
    rows = [[x, y, y] for x, y in _gaussian_rows()]
    payload = _analyze(["motor", "det", "other"], rows, ["motor"], ["det", "missing"])

    assert payload["available"] is True
    assert _channel(payload, "det")["reason"] is None
    missing = _channel(payload, "missing")
    assert missing["reason"] == analysis.REASON_CHANNEL_COLUMN_MISSING
    assert missing["column"] is None
    assert missing["center"] is None


def test_a_channel_declared_both_movable_and_readable_is_its_own_axis() -> None:
    payload = _analyze(["motor", "det"], _gaussian_rows(), ["motor"], ["motor", "det"])

    assert payload["available"] is True
    assert _channel(payload, "motor")["reason"] == analysis.REASON_CHANNEL_IS_THE_X_AXIS
    assert _channel(payload, "det")["reason"] is None


# =========================================================================
# --- The pre-filter ---
# =========================================================================


def test_pairs_with_a_missing_or_nan_reading_are_dropped_before_the_fit() -> None:
    """One NaN reaching the stock code poisons every statistic it computes.

    The dropped points must not shift the answer either: the same gaussian with
    three readings knocked out still finds its center.
    """
    rows = _gaussian_rows()
    clean = _analyze(["motor", "det"], rows, ["motor"], ["det"])
    holed = [list(row) for row in rows]
    holed[3][1] = None
    holed[7][1] = float("nan")
    holed[11][0] = None

    payload = _analyze(["motor", "det"], holed, ["motor"], ["det"])

    det = _channel(payload, "det")
    assert det["points"] == len(rows) - 3
    assert payload["points"] == len(rows)  # rows read, not pairs kept
    assert det["center"] == pytest.approx(_channel(clean, "det")["center"], abs=0.05)
    assert math.isfinite(det["center_of_mass"])


def test_a_channel_that_reads_back_as_text_has_too_few_pairs() -> None:
    rows = [[x, "moving"] for x, _ in _gaussian_rows()]
    payload = _analyze(["motor", "det"], rows, ["motor"], ["det"])

    assert payload["available"] is True
    det = _channel(payload, "det")
    assert det["reason"] == analysis.REASON_INSUFFICIENT_POINTS
    assert det["points"] == 0


def test_a_boolean_reading_is_a_state_not_a_coordinate() -> None:
    rows = [[x, index % 2 == 0] for index, (x, _) in enumerate(_gaussian_rows())]
    payload = _analyze(["motor", "shutter"], rows, ["motor"], ["shutter"])

    assert _channel(payload, "shutter")["reason"] == analysis.REASON_INSUFFICIENT_POINTS


def test_a_ragged_row_costs_a_point_rather_than_raising() -> None:
    rows: list[list[Any]] = [list(row) for row in _gaussian_rows()]
    rows[5] = [rows[5][0]]

    payload = _analyze(["motor", "det"], rows, ["motor"], ["det"])

    assert payload["available"] is True
    assert _channel(payload, "det")["points"] == len(rows) - 1


# =========================================================================
# --- What the numbers are, and what a null one means ---
# =========================================================================


def test_a_channel_that_never_crosses_its_half_maximum_has_no_center_or_width() -> None:
    """A flat channel is analyzed, not skipped: the nulls are the answer.

    ``reason`` stays null — the fit ran and the data has no peak in it, which
    is a different fact from "this channel could not be analyzed".
    """
    rows = [[x, 4.2] for x, _ in _gaussian_rows()]
    payload = _analyze(["motor", "flat"], rows, ["motor"], ["flat"])

    flat = _channel(payload, "flat")
    assert flat["reason"] is None
    assert flat["center"] is None
    assert flat["width"] is None
    assert flat["center_of_mass"] == pytest.approx(0.0, abs=0.2)


def test_every_statistic_reaches_the_wire_as_a_plain_json_number() -> None:
    """The stock code answers in numpy scalars, which json refuses."""
    payload = _analyze(["motor", "det"], _gaussian_rows(), ["motor"], ["det"])
    det = _channel(payload, "det")

    for key in ("center", "width", "center_of_mass"):
        assert type(det[key]) is float, f"{key} is {type(det[key]).__name__}"
    assert json.loads(json.dumps(payload)) == payload


def test_the_absent_payload_carries_the_same_keys_as_an_available_one() -> None:
    available = _analyze(["motor", "det"], _gaussian_rows(), ["motor"], ["det"])

    assert set(analysis.absent(analysis.REASON_RUN_IN_PROGRESS)) == set(available)


# =========================================================================
# --- A library problem is an absence, never a failed read ---
# =========================================================================


def test_an_unimportable_statistics_module_is_an_absence_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "bluesky.callbacks.fitting", None)

    payload = _analyze(["motor", "det"], _gaussian_rows(), ["motor"], ["det"])

    assert payload["available"] is False
    assert payload["reason"] == analysis.REASON_STATISTICS_UNAVAILABLE


def test_a_channel_whose_fit_raises_is_absent_beside_the_ones_that_worked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = analysis._peak_stats

    def _explode(peak_stats_cls: Any, xs: Any, ys: Any) -> dict[str, Any]:
        if list(ys)[0] == 0.0:
            raise RuntimeError("no")
        return real(peak_stats_cls, xs, ys)

    monkeypatch.setattr(analysis, "_peak_stats", _explode)
    rows = [[x, 0.0, y] for x, y in _gaussian_rows()]

    payload = _analyze(["motor", "bad", "det"], rows, ["motor"], ["bad", "det"])

    assert payload["available"] is True
    assert _channel(payload, "bad")["reason"] == analysis.REASON_STATISTICS_UNAVAILABLE
    assert _channel(payload, "det")["reason"] is None


# =========================================================================
# --- The route wiring ---
# =========================================================================


class _ScanParams(BaseModel):
    """A one-movable plan, declared the way a plan author declares it."""

    axis: MovableChannel = Field(...)
    monitors: ReadableChannels = Field(default_factory=list)


class _SweepParams(BaseModel):
    """A serial sweep: several movables, so no single independent variable."""

    correctors: list[str] = Field(default_factory=list)
    monitors: ReadableChannels = Field(default_factory=list)


def _spec(name: str = "demo", schema: type[BaseModel] = _ScanParams) -> PlanSpec[Any]:
    return PlanSpec(
        name=name,
        plan=lambda devices, params: None,
        schema=schema,
        roles=tuple(channel_roles(schema)),
    )


def _install_catalog(monkeypatch: pytest.MonkeyPatch, *specs: PlanSpec[Any]) -> None:
    plans = FacilityPlans(plans={spec.name: spec for spec in specs})
    monkeypatch.setattr(
        "osprey.services.bluesky_bridge.plan_loader.get_facility_plans", lambda: plans
    )


def _stamp(name: str = "demo", **kwargs: Any) -> dict[str, Any]:
    return {"name": name, "kwargs": kwargs}


def _feed_live(
    run_id: str, rows: list[dict[str, Any]], *, plan: Any = None, stop: bool = True
) -> None:
    recorder = live_rows.LiveRowRecorder(key=run_id, plan=plan)
    recorder("start", {"uid": f"engine-{run_id}"})
    for row in rows:
        recorder("event", {"data": row})
    if stop:
        recorder("stop", {})


def _scan_events() -> list[dict[str, Any]]:
    return [{"motor": x, "det": y} for x, y in _gaussian_rows()]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _get_data(client: TestClient, run_id: str, **params: Any) -> dict[str, Any]:
    query = "&".join(f"{key}={value}" for key, value in params.items())
    response = client.get(f"/runs/{run_id}/data" + (f"?{query}" if query else ""))
    assert response.status_code == 200, response.text
    body = response.json()
    assert "analysis" in body, "every data response owes an analysis block"
    return body


def test_a_settled_live_run_serves_its_statistics_on_the_data_payload(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec())
    _feed_live("run-1", _scan_events(), plan=_stamp(axis="motor", monitors=["det"]))

    body = _get_data(client, "run-1")

    assert body["analysis"]["available"] is True
    assert body["analysis"]["x_channel"] == "motor"
    assert _channel(body["analysis"], "det")["center"] == pytest.approx(0.0, abs=0.05)


def test_statistics_are_computed_over_the_whole_run_not_the_window(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window is what the CALLER asked for; a peak over one page is a lie."""
    _install_catalog(monkeypatch, _spec())
    _feed_live("run-2", _scan_events(), plan=_stamp(axis="motor", monitors=["det"]))

    body = _get_data(client, "run-2", max_rows=2)

    assert len(body["rows"]) == 2
    assert body["analysis"]["points"] == 21
    assert _channel(body["analysis"], "det")["points"] == 21


def test_a_run_still_producing_rows_says_so_instead_of_fitting_half_a_scan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec())
    _feed_live("run-3", _scan_events(), plan=_stamp(axis="motor", monitors=["det"]), stop=False)

    body = _get_data(client, "run-3")

    assert body["partial"] is True
    assert body["analysis"]["available"] is False
    assert body["analysis"]["reason"] == analysis.REASON_RUN_IN_PROGRESS


def test_a_run_with_no_plan_stamp_has_no_declaration_to_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec())
    _feed_live("run-4", _scan_events(), plan=None)

    body = _get_data(client, "run-4")

    assert body["analysis"]["reason"] == analysis.REASON_PLAN_IDENTITY_UNAVAILABLE


def test_a_stamped_name_with_no_current_owner_has_no_declaration_to_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec("other"))
    _feed_live("run-5", _scan_events(), plan=_stamp(axis="motor", monitors=["det"]))

    body = _get_data(client, "run-5")

    assert body["analysis"]["reason"] == analysis.REASON_PLAN_IDENTITY_UNAVAILABLE


def test_a_serial_sweeps_payload_says_why_it_has_no_peak(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec(schema=_ScanParams))
    _feed_live("run-6", _scan_events(), plan=_stamp(axis="motor", monitors=["det"]))
    _install_catalog(monkeypatch, _spec(schema=_SweepParams))

    body = _get_data(client, "run-6")

    # `_SweepParams` declares no movable at all — a plan whose sweep field was
    # never role-typed reads as "nothing declared", not as a guess at `axis`.
    assert body["analysis"]["reason"] == analysis.REASON_NO_DECLARED_MOVABLE


def test_stamped_parameters_that_no_longer_validate_are_still_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema drift must not cost the statistics: the stamp is walked as stored."""

    class _Drifted(BaseModel):
        axis: MovableChannel = Field(...)
        monitors: ReadableChannels = Field(...)
        gain: float = Field(...)  # never stamped

    _install_catalog(monkeypatch, _spec(schema=_Drifted))
    _feed_live("run-7", _scan_events(), plan=_stamp(axis="motor", monitors=["det"]))

    body = _get_data(client, "run-7")

    assert body["analysis"]["available"] is True


# =========================================================================
# --- Computed once per settled run ---
# =========================================================================


def _counting_analyze(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    real = analysis.analyze

    def _counted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(analysis, "analyze", _counted)
    monkeypatch.setattr(app_module.analysis, "analyze", _counted)
    return calls


def test_a_settled_runs_statistics_are_computed_once_however_often_it_is_polled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_catalog(monkeypatch, _spec())
    _feed_live("run-8", _scan_events(), plan=_stamp(axis="motor", monitors=["det"]))
    calls = _counting_analyze(monkeypatch)

    first = _get_data(client, "run-8")
    second = _get_data(client, "run-8", max_rows=5)

    assert calls[0] == 1
    assert first["analysis"] == second["analysis"]


def test_re_running_a_run_id_never_serves_the_previous_occupants_peak(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The document plane evicts on the new start doc; the generation bump is
    what makes the held analysis unservable."""
    _install_catalog(monkeypatch, _spec())
    _feed_live("run-9", _scan_events(), plan=_stamp(axis="motor", monitors=["det"]))
    first = _get_data(client, "run-9")["analysis"]

    figure_cache.evict("run-9")
    shifted = [{"motor": x + 5.0, "det": y} for x, y in _gaussian_rows()]
    _feed_live("run-9", shifted, plan=_stamp(axis="motor", monitors=["det"]))

    second = _get_data(client, "run-9")["analysis"]

    assert _channel(first, "det")["center"] == pytest.approx(0.0, abs=0.05)
    assert _channel(second, "det")["center"] == pytest.approx(5.0, abs=0.05)


def test_a_catalog_that_raised_is_not_stuck_to_the_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing catalog is this process's trouble, not a truth about the run —
    the same distinction the figure route's ``cacheable`` draws."""

    def _broken() -> FacilityPlans:
        raise RuntimeError("catalog on fire")

    monkeypatch.setattr("osprey.services.bluesky_bridge.plan_loader.get_facility_plans", _broken)
    _feed_live("run-10", _scan_events(), plan=_stamp(axis="motor", monitors=["det"]))

    broken = _get_data(client, "run-10")
    assert broken["analysis"]["reason"] == analysis.REASON_STATISTICS_UNAVAILABLE

    _install_catalog(monkeypatch, _spec())
    recovered = _get_data(client, "run-10")

    assert recovered["analysis"]["available"] is True


# =========================================================================
# --- The stored-run path ---
# =========================================================================


def test_a_run_read_back_from_storage_carries_the_same_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_from_tiled` analyses the snapshot it already read, not a second read."""
    _install_catalog(monkeypatch, _spec())
    rows = [[x, y] for x, y in _gaussian_rows()]
    snapshot = {
        "columns": ["motor", "det"],
        "rows": rows,
        "total_seen": len(rows),
        "partial": False,
        "plan": _stamp(axis="motor", monitors=["det"]),
        "run_uid": "engine-stored",
    }
    monkeypatch.setattr(app_module, "_tiled_run_snapshot", lambda run_id: snapshot)

    result = app_module._from_tiled("run-stored", max_rows=2, offset=None, tail=False)

    assert result is not None
    assert len(result["rows"]) == 2
    assert result["analysis"]["available"] is True
    assert result["analysis"]["points"] == len(rows)
    assert _channel(result["analysis"], "det")["center"] == pytest.approx(0.0, abs=0.05)


def test_a_stored_run_still_in_flight_after_a_restart_is_not_fitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_catalog(monkeypatch, _spec())
    rows = [[x, y] for x, y in _gaussian_rows()]
    snapshot = {
        "columns": ["motor", "det"],
        "rows": rows,
        "total_seen": len(rows),
        "partial": True,  # no stop document in the catalog
        "plan": _stamp(axis="motor", monitors=["det"]),
        "run_uid": "engine-stored",
    }
    monkeypatch.setattr(app_module, "_tiled_run_snapshot", lambda run_id: snapshot)

    result = app_module._from_tiled("run-stored", max_rows=100, offset=None, tail=False)

    assert result is not None
    assert result["analysis"]["reason"] == analysis.REASON_RUN_IN_PROGRESS
