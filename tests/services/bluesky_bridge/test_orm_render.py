"""Unit tests for `plans_core/orm.py`'s `render()`.

What is actually load-bearing about an `orm` figure, and gets pinned here:

- it draws sweeps `build_response_matrix` cannot fit at all -- a
  `monodirectional` run renders a full matrix where the legacy path raises;
- it is honest mid-run: completed sweeps are fitted, the in-flight one is
  traced and says so, and correctors the sweep has not reached are absent
  rather than drawn as dead;
- it never raises, whatever it is handed, because a figure is a view and the
  run's own result does not depend on it;
- a row set the run outgrew (the live buffer's per-run cap) keeps its traces
  but loses the peer-comparison panels, which over an arbitrary subset of the
  machine would describe the cap rather than the machine.

Rows are hand-built to the shape `build_plan` emits -- one row per (corrector,
current) point, corrector-major, every row carrying every corrector's current.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from osprey.services.bluesky_bridge import live_rows, plan_loader
from osprey.services.bluesky_bridge.figure import BarsMark, Figure, HeatmapMark, LinesMark
from osprey.services.bluesky_bridge.orm_analysis import (
    DegenerateFitError,
    build_response_matrix,
)
from osprey.services.bluesky_bridge.plans_core import orm

# =========================================================================
# Fixtures: rows in the shape the real plan emits
# =========================================================================


def _currents(span_a: float, num: int, *, sweep: str) -> list[float]:
    """The currents `build_plan` sweeps, by the same arithmetic."""
    if sweep == "monodirectional":
        step = span_a / (num - 1)
        return [i * step for i in range(num)]
    step = (2 * span_a) / (num - 1)
    return [-span_a + i * step for i in range(num)]


def _rows(
    correctors: list[str],
    detectors: list[str],
    truth: np.ndarray,
    currents: list[float],
    *,
    stop_after: int | None = None,
) -> list[dict[str, float]]:
    """Rows an `orm` run emits for a machine whose response is *truth*.

    Corrector-major emission order, every row carrying EVERY corrector's
    current (idle ones read back 0.0) alongside every detector -- the document
    shape `build_plan` produces.
    """
    rows: list[dict[str, float]] = []
    for j, corrector in enumerate(correctors):
        for current in currents:
            row = dict.fromkeys(correctors, 0.0)
            row[corrector] = current
            for i, detector in enumerate(detectors):
                row[detector] = float(truth[i, j] * current)
            rows.append(row)
            if stop_after is not None and len(rows) >= stop_after:
                return rows
    return rows


def _truth(n_det: int, n_corr: int, seed: int = 7) -> np.ndarray:
    """A deterministic, non-degenerate `[n_det, n_corr]` response matrix."""
    return np.random.default_rng(seed).normal(size=(n_det, n_corr))


def _params(**overrides) -> orm.PARAMS:
    defaults = {
        "correctors": ["hcm1", "hcm2"],
        "detectors": ["bpm1", "bpm2", "bpm3"],
        "span_a": 1.0,
        "num": 7,
    }
    return orm.PARAMS(**{**defaults, **overrides})


def _panel(figure: Figure, title: str):
    return next(panel for panel in figure.panels if panel.title == title)


# =========================================================================
# The sweeps the legacy fit cannot take
# =========================================================================


def test_monodirectional_run_renders_a_full_response_matrix() -> None:
    """A one-sided sweep fits here and raises on the legacy row-scan path.

    That contrast is the whole reason `render` fits by acquisition index:
    `build_response_matrix` needs every corrector's samples to be symmetric
    about 0, which a `monodirectional` sweep is not.
    """
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2", "bpm3"]
    truth = _truth(len(detectors), len(correctors))
    params = _params(correctors=correctors, detectors=detectors, num=7, sweep="monodirectional")
    rows = _rows(correctors, detectors, truth, _currents(1.0, 7, sweep="monodirectional"))

    with pytest.raises(DegenerateFitError):
        build_response_matrix(rows, correctors, detectors)

    figure = orm.render(rows, params)

    heatmap = _panel(figure, "Response matrix").mark
    assert isinstance(heatmap, HeatmapMark)
    assert heatmap.x_labels == correctors
    assert heatmap.y_labels == detectors
    assert np.allclose(np.array(heatmap.values), truth, atol=1e-9)


def test_bidirectional_run_renders_the_same_matrix() -> None:
    """The default sweep is not a special case -- same panels, same slopes."""
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2", "bpm3"]
    truth = _truth(len(detectors), len(correctors))
    params = _params(correctors=correctors, detectors=detectors, num=7)
    rows = _rows(correctors, detectors, truth, _currents(1.0, 7, sweep="bidirectional"))

    heatmap = _panel(orm.render(rows, params), "Response matrix").mark
    assert isinstance(heatmap, HeatmapMark)
    assert np.allclose(np.array(heatmap.values), truth, atol=1e-9)


# =========================================================================
# Panel structure -- the contract the panel renderer and the agent consume
# =========================================================================


def test_panel_order_leads_with_the_fit_and_demotes_the_sweeps() -> None:
    """The finding leads; the raw evidence follows in collapsed sections.

    Order is stable, so a live figure grows instead of rearranging: the fitted
    panels appear above the sweep section as sweeps complete.
    """
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2", "bpm3"]
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(correctors, detectors, truth, _currents(1.0, 7, sweep="bidirectional"))

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))

    assert [panel.title for panel in figure.panels] == [
        "Response matrix",
        "Response by BPM",
        "Corrector anomaly score",
        "BPM anomaly score",
        "Singular values",
        "hcm1 sweep",
        "hcm2 sweep",
    ]
    # The fitted headline is inline; the sweeps and the spectrum are demoted.
    assert [panel.section for panel in figure.panels] == [
        None,
        None,
        None,
        None,
        "Singular values",
        "Per-corrector sweeps",
        "Per-corrector sweeps",
    ]

    traces = _panel(figure, "hcm1 sweep").mark
    assert isinstance(traces, LinesMark)
    assert [series.label for series in traces.series] == detectors
    assert traces.source_points == len(detectors) * 7
    assert traces.decimated is False

    correctors_bar = _panel(figure, "Corrector anomaly score").mark
    bpm_bar = _panel(figure, "BPM anomaly score").mark
    assert isinstance(correctors_bar, BarsMark)
    assert isinstance(bpm_bar, BarsMark)
    assert correctors_bar.categories == correctors
    assert bpm_bar.categories == detectors


def test_response_by_bpm_is_the_matrix_read_column_wise() -> None:
    """One line per fitted corrector, carrying that corrector's matrix column.

    The panel ships EVERY fitted corrector and marks itself selectable: the
    reader's choice of which lines to draw is a filter over data already on the
    wire, never a refetch.
    """
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2", "bpm3"]
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(correctors, detectors, truth, _currents(1.0, 7, sweep="bidirectional"))

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))
    panel = _panel(figure, "Response by BPM")

    assert panel.series_picker is True
    assert panel.section is None  # a headline panel, not demoted

    mark = panel.mark
    assert isinstance(mark, LinesMark)
    assert [series.label for series in mark.series] == correctors

    # Series j is column j of the matrix, against a 1-based BPM index.
    for j, series in enumerate(mark.series):
        assert [point.x for point in series.points] == [1.0, 2.0, 3.0]
        assert [point.y for point in series.points] == pytest.approx(truth[:, j], abs=1e-9)


def test_response_by_bpm_carries_only_fitted_correctors() -> None:
    """An in-flight corrector has no column, so it has no line -- the same
    subsequence the heatmap's x axis uses, never a flat zero line that would
    read as a dead corrector."""
    correctors = ["hcm1", "hcm2", "hcm3"]
    detectors = ["bpm1", "bpm2"]
    num = 7
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(
        correctors,
        detectors,
        truth,
        _currents(1.0, num, sweep="bidirectional"),
        stop_after=num + 3,  # hcm1 complete, hcm2 in flight, hcm3 untouched
    )

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors, num=num))
    mark = _panel(figure, "Response by BPM").mark

    assert isinstance(mark, LinesMark)
    assert [series.label for series in mark.series] == ["hcm1"]


def test_singular_values_panel_is_log10_and_sectioned() -> None:
    """The spectrum is a second opinion, so it is demoted into its own section.

    Values are log10 because the renderer draws linear axes only and a
    spectrum's meaning is its decades of fall-off; the axis label says so.
    """
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2", "bpm3"]
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(correctors, detectors, truth, _currents(1.0, 7, sweep="bidirectional"))

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))
    panel = _panel(figure, "Singular values")

    assert panel.section == "Singular values"
    assert panel.y_label == "log10 singular value"

    mark = panel.mark
    assert isinstance(mark, LinesMark)
    points = mark.series[0].points
    assert [point.x for point in points] == [1.0, 2.0]  # min(n_bpm, n_corr)

    expected = np.log10(np.linalg.svd(truth, compute_uv=False))
    assert [point.y for point in points] == pytest.approx(expected, abs=1e-9)


def test_a_numerically_zero_singular_value_is_a_gap_not_a_minus_sixteen() -> None:
    """A rank-deficient matrix leaves ~1e-16 residue, not an exact zero.

    Plotting log10 of that residue would stretch the axis over sixteen decades
    and squash every real mode flat, so anything at or below the numerical-rank
    tolerance is a gap.
    """
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2"]
    # Both correctors drive an identical BPM shape: a rank-1 matrix, so the
    # second singular value is zero.
    truth = np.array([[1.0, 1.0], [2.0, 2.0]])
    rows = _rows(correctors, detectors, truth, _currents(1.0, 7, sweep="bidirectional"))

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))
    mark = _panel(figure, "Singular values").mark

    assert isinstance(mark, LinesMark)
    points = mark.series[0].points
    assert points[0].y is not None
    assert points[1].y is None


def test_trace_points_are_the_recorded_current_and_reading() -> None:
    """A trace point is a claim about one row, not a smoothed curve."""
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2", "bpm3"]
    truth = _truth(len(detectors), len(correctors))
    currents = _currents(1.0, 7, sweep="bidirectional")
    rows = _rows(correctors, detectors, truth, currents)

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))
    traces = _panel(figure, "hcm2 sweep").mark
    assert isinstance(traces, LinesMark)

    series = traces.series[0]
    assert [point.x for point in series.points] == pytest.approx(currents)
    assert [point.y for point in series.points] == pytest.approx(
        [truth[0, 1] * current for current in currents]
    )


def test_figure_round_trips_through_model_validate() -> None:
    """What the route dumps to JSON validates back to the same marks."""
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2"]
    rows = _rows(
        correctors,
        detectors,
        _truth(len(detectors), len(correctors)),
        _currents(1.0, 7, sweep="bidirectional"),
    )

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))
    assert Figure.model_validate(figure.model_dump()) == figure


def test_partial_and_source_are_placeholders_the_route_overwrites() -> None:
    """The team-wide render convention: always `partial=True, source="live"`.

    A render sees rows, never their provenance, so neither field can be its
    answer to give -- the figure route stamps the truth onto both on every
    path. `True` is the honest direction for the placeholder: a forgotten
    overwrite costs a client extra polling, never a false "settled". Pinned
    across a finished run, a mid-flight one, and a degraded one so nobody
    later "improves" this into a data-derived value the route discards.
    """
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2"]
    truth = _truth(len(detectors), len(correctors))
    currents = _currents(1.0, 7, sweep="bidirectional")
    params = _params(correctors=correctors, detectors=detectors)

    finished = _rows(correctors, detectors, truth, currents)
    mid_flight = _rows(correctors, detectors, truth, currents, stop_after=9)
    degraded = [{"motor": 1.0}]

    for rows in (finished, mid_flight, degraded, []):
        figure = orm.render(rows, params)
        assert figure.partial is True
        assert figure.source == "live"


# =========================================================================
# Mid-run honesty
# =========================================================================


def test_mid_sweep_fits_completed_correctors_and_traces_the_in_flight_one() -> None:
    """Three sweeps done, one in flight, one not started."""
    correctors = ["hcm1", "hcm2", "hcm3", "hcm4", "hcm5"]
    detectors = ["bpm1", "bpm2", "bpm3"]
    truth = _truth(len(detectors), len(correctors))
    num = 7
    rows = _rows(
        correctors,
        detectors,
        truth,
        _currents(1.0, num, sweep="bidirectional"),
        stop_after=3 * num + 4,  # three complete sweeps, then four points of the fourth
    )

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors, num=num))

    titles = [panel.title for panel in figure.panels]
    sweeps = [title for title in titles if title.endswith(" sweep")]
    assert sweeps == ["hcm1 sweep", "hcm2 sweep", "hcm3 sweep", "hcm4 sweep"]
    assert "hcm5 sweep" not in titles  # not started: no panel, not an empty one

    heatmap = _panel(figure, "Response matrix").mark
    assert isinstance(heatmap, HeatmapMark)
    assert heatmap.x_labels == ["hcm1", "hcm2", "hcm3"]
    assert np.allclose(np.array(heatmap.values), truth[:, :3], atol=1e-9)

    in_flight = _panel(figure, "hcm4 sweep")
    traces = in_flight.mark
    assert isinstance(traces, LinesMark)
    assert len(traces.series[0].points) == 4
    assert any("still in flight" in note for note in in_flight.annotations)
    assert any("4 of 7 points" in note for note in in_flight.annotations)

    matrix_notes = _panel(figure, "Response matrix").annotations
    assert any("3 of 5 corrector sweeps" in note for note in matrix_notes)


def test_full_input_fits_every_corrector_and_a_windowed_input_does_not() -> None:
    """The property the route's full-row-set contract exists to protect.

    A 300-row run (5 correctors x 60 points): given all of it, every corrector
    -- including the three whose sweeps begin past row 100 -- gets a non-zero
    fitted slope. Given only the first 100 rows, they are simply absent; the
    figure never invents a column for a corrector it did not see finish.
    """
    correctors = [f"hcm{j}" for j in range(1, 6)]
    detectors = ["bpm1", "bpm2", "bpm3"]
    num = 60
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(correctors, detectors, truth, _currents(1.0, num, sweep="bidirectional"))
    assert len(rows) == 300

    params = _params(correctors=correctors, detectors=detectors, num=num)

    full = _panel(orm.render(rows, params), "Response matrix").mark
    assert isinstance(full, HeatmapMark)
    assert full.x_labels == correctors
    values = np.array(full.values)
    assert values.shape == (len(detectors), len(correctors))
    for j, corrector in enumerate(correctors):
        assert np.all(np.abs(values[:, j]) > 0.0), f"{corrector} fitted a zero column"
    assert np.allclose(values, truth, atol=1e-9)

    windowed = _panel(orm.render(rows[:100], params), "Response matrix").mark
    assert isinstance(windowed, HeatmapMark)
    assert windowed.x_labels == ["hcm1"]  # hcm2 is 40/60 in; hcm3-5 have no rows at all


def test_a_bpm_that_missed_a_reading_draws_a_gap_not_a_zero() -> None:
    """A missing reading is `None` on the wire -- a break in the line."""
    correctors, detectors = ["hcm1"], ["bpm1", "bpm2"]
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(correctors, detectors, truth, _currents(1.0, 7, sweep="bidirectional"))
    rows[3]["bpm1"] = None  # type: ignore[assignment]

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))
    traces = _panel(figure, "hcm1 sweep").mark
    assert isinstance(traces, LinesMark)
    assert traces.series[0].points[3].y is None
    assert traces.series[1].points[3].y is not None


# =========================================================================
# Caps: what a facility-scale run is allowed to put on the wire
# =========================================================================


def test_the_row_cap_mirror_matches_the_live_buffer() -> None:
    """`orm.py` duplicates the buffer's cap; drift is a wrong figure."""
    assert orm._LIVE_BUFFER_ROW_CAP == live_rows._MAX_ROWS_PER_RUN


def test_a_capped_row_set_keeps_its_traces_and_drops_the_peer_panels() -> None:
    """Exactly the cap's worth of rows for a sweep that wanted more.

    The traces still stand -- each point is a claim about one stored row -- but
    a peer-comparison score over whichever correctors fit under the cap would
    describe the cap, not the machine, so those panels go and an annotation
    says why.
    """
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2", "bpm3"]
    num = 6000  # 12,000 rows expected, 10,000 stored
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(
        correctors,
        detectors,
        truth,
        _currents(1.0, num, sweep="bidirectional"),
        stop_after=orm._LIVE_BUFFER_ROW_CAP,
    )
    assert len(rows) == orm._LIVE_BUFFER_ROW_CAP

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors, num=num))

    assert [panel.title for panel in figure.panels] == ["hcm1 sweep", "hcm2 sweep"]
    lead = figure.panels[0].annotations
    assert any("more rows than the bridge stores" in note for note in lead)
    assert any("10,000" in note for note in lead)

    traces = figure.panels[0].mark
    assert isinstance(traces, LinesMark)
    assert traces.decimated is True
    assert len(traces.series[0].points) == 2000
    assert traces.series[0].source_points == num


def test_an_uncapped_run_longer_than_the_cap_still_fits() -> None:
    """A settled Tiled read is not a capped buffer just because it is big.

    The cap is recognized by an exact row count against the parameters' own
    expected total, so a complete row set that merely exceeds the cap keeps
    every panel.
    """
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2"]
    num = 6000
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(correctors, detectors, truth, _currents(1.0, num, sweep="bidirectional"))
    assert len(rows) == 12_000

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors, num=num))

    assert "Response matrix" in [panel.title for panel in figure.panels]


def test_panels_and_series_are_capped_with_annotations_naming_the_excess() -> None:
    """A 20-corrector, 30-BPM run is not 20 panels of 30 lines."""
    correctors = [f"hcm{j}" for j in range(1, 21)]
    detectors = [f"bpm{i}" for i in range(1, 31)]
    num = 5
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(correctors, detectors, truth, _currents(1.0, num, sweep="bidirectional"))

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors, num=num))

    trace_titles = [panel.title for panel in figure.panels if panel.title.endswith(" sweep")]
    assert len(trace_titles) == orm._MAX_TRACE_PANELS
    assert trace_titles[-1] == "hcm20 sweep"  # the tail, where a live sweep is

    first = _panel(figure, trace_titles[0])
    assert any("8 most recently swept of 20 correctors" in n for n in first.annotations)
    assert any("first 12 of 30 BPMs" in n for n in first.annotations)

    traces = first.mark
    assert isinstance(traces, LinesMark)
    assert len(traces.series) == orm._MAX_TRACE_SERIES

    # The fit itself is never capped: every corrector and BPM keeps its cell.
    heatmap = _panel(figure, "Response matrix").mark
    assert isinstance(heatmap, HeatmapMark)
    assert len(heatmap.x_labels) == 20
    assert len(heatmap.y_labels) == 30


# =========================================================================
# Totality: a figure is a view, and its failure is never the run's
# =========================================================================


def test_no_rows_render_an_empty_figure() -> None:
    figure = orm.render([], _params())
    assert figure.panels == []


def test_non_numeric_rows_do_not_raise() -> None:
    """Rows the fit cannot even build an array from degrade to no panels."""
    rows = [{"hcm1": "not a number", "bpm1": "nor this"} for _ in range(7)]
    figure = orm.render(rows, _params())
    assert figure.panels == []


def test_rows_from_a_different_plan_do_not_raise() -> None:
    """None of the named devices appear: nothing to draw, still a figure."""
    rows = [{"motor": float(i), "det": float(i) ** 2} for i in range(20)]
    figure = orm.render(rows, _params())
    assert figure.panels == []


def test_a_stuck_corrector_is_traced_and_named_not_fitted() -> None:
    """Currents that never move have no slope -- say so, do not invent one."""
    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2"]
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(correctors, detectors, truth, _currents(1.0, 7, sweep="bidirectional"))
    for row in rows[7:]:  # hcm2's slice: the corrector never moves
        row["hcm2"] = 0.0

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))

    stuck = _panel(figure, "hcm2 sweep")
    assert any("never moved" in note for note in stuck.annotations)
    heatmap = _panel(figure, "Response matrix").mark
    assert isinstance(heatmap, HeatmapMark)
    assert heatmap.x_labels == ["hcm1"]


def test_a_failing_fit_panel_degrades_to_the_traces(monkeypatch) -> None:
    """The second stage of the fallback, exercised rather than assumed."""

    def _boom(matrix):
        raise RuntimeError("anomaly scoring blew up")

    monkeypatch.setattr(orm, "column_anomaly", _boom)

    correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2"]
    rows = _rows(
        correctors,
        detectors,
        _truth(len(detectors), len(correctors)),
        _currents(1.0, 7, sweep="bidirectional"),
    )

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))

    assert [panel.title for panel in figure.panels] == ["hcm1 sweep", "hcm2 sweep"]
    traces = figure.panels[0].mark
    assert isinstance(traces, LinesMark)
    assert len(traces.series) == 2


def test_every_trace_x_is_finite() -> None:
    """`Point.x` rejects a non-finite coordinate; render must not offer one."""
    correctors, detectors = ["hcm1"], ["bpm1"]
    truth = _truth(len(detectors), len(correctors))
    rows = _rows(correctors, detectors, truth, _currents(1.0, 7, sweep="bidirectional"))
    rows[2]["hcm1"] = float("nan")

    figure = orm.render(rows, _params(correctors=correctors, detectors=detectors))
    traces = figure.panels[0].mark
    assert isinstance(traces, LinesMark)
    assert len(traces.series[0].points) == 6
    assert all(math.isfinite(point.x) for point in traces.series[0].points)


# =========================================================================
# The loader picks it up
# =========================================================================


def test_the_shipped_orm_spec_carries_this_render() -> None:
    plan_loader.reset_facility_plans()
    try:
        spec = plan_loader.get_facility_plans().plans["orm"]
        assert spec.render is not None
        assert spec.render.__name__ == "render"

        # The loader execs the file under a synthetic module name, so identity
        # with `orm.render` does not hold -- behaviour is what matters.
        correctors, detectors = ["hcm1", "hcm2"], ["bpm1", "bpm2"]
        rows = _rows(
            correctors,
            detectors,
            _truth(len(detectors), len(correctors)),
            _currents(1.0, 7, sweep="bidirectional"),
        )
        params = spec.schema.model_validate(
            {
                "correctors": correctors,
                "detectors": detectors,
                "span_a": 1.0,
                "num": 7,
            }
        )
        figure = spec.render(rows, params)
        assert "Response matrix" in [panel.title for panel in figure.panels]
    finally:
        plan_loader.reset_facility_plans()
