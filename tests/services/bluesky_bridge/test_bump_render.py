"""Unit tests for `plans_core/orbit_bump_sweep.py`'s `render()`.

What is load-bearing about an `orbit_bump_sweep` figure, and gets pinned here:

- it says whether the bump was a *bump* -- displaced at the targets, back on
  the reference orbit at the closure BPMs -- and whether every step stayed
  inside the tolerance band it asked for;
- it attributes rows by position (``baseline_reads`` baseline rows, then one
  row per scale of the amplitude profile), so a short row set draws the prefix
  it has and says how much of the profile is missing rather than mislabelling
  what it drew;
- a `best_effort` run's misses are named -- which step, which BPM, reached
  against requested -- because that is the whole reason the run continued;
- it never raises, whatever it is handed, because a figure is a view and the
  run's own result does not depend on it;
- every series that reaches the wire is capped and decimated, the flat band
  lines included: they are as long as the sweep and would blow the point
  budget on their own.

Rows are hand-built to the shape `build_plan` emits -- ``baseline_reads``
baseline rows, then exactly one row per amplitude step in profile order, every
row carrying every corrector, BPM, and detector the parameters name.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.services.bluesky_bridge.figure import DEFAULT_MAX_POINTS, Figure, LinesMark
from osprey.services.bluesky_bridge.plans_core import orbit_bump_sweep as bump

# =========================================================================
# Fixtures: rows in the shape the real plan emits
# =========================================================================

_MONITOR_SHIFT = 0.05
"""How far a monitor BPM -- one nothing constrains -- moves per unit scale."""


def _params(**overrides: Any) -> bump.PARAMS:
    defaults: dict[str, Any] = {
        "correctors": ["c1", "c2", "c3"],
        "targets": [{"bpm": "bpm_t", "value": 0.3}],
        "closure_bpms": ["bpm_c1", "bpm_c2"],
        "bpms": ["bpm_m1"],
        "num": 4,
        "baseline_reads": 5,
        "probe_amplitude": 0.01,
        "tolerance": 0.02,
        "max_trim_iterations": 3,
        "settle_s": 0.0,
    }
    return bump.PARAMS(**{**defaults, **overrides})


def _bpm_names(params: bump.PARAMS) -> list[str]:
    """Every requested BPM, named once, in request order -- `_prepare`'s own order."""
    names: list[str] = []
    for name in [target.bpm for target in params.targets] + [
        *params.closure_bpms,
        *params.bpms,
    ]:
        if name not in names:
            names.append(name)
    return names


def _reference(params: bump.PARAMS) -> dict[str, float]:
    """The orbit the machine sits on before anything moves, one value per BPM."""
    return {name: 0.5 * (position + 1) for position, name in enumerate(_bpm_names(params))}


def _working(params: bump.PARAMS) -> dict[str, float]:
    """Each corrector's pre-scan working point -- nonzero, as a corrected ring's is."""
    return {name: 0.25 * (index + 1) for index, name in enumerate(params.correctors)}


def _rows(
    params: bump.PARAMS,
    *,
    limit: int | None = None,
    residual: Any = None,
    drop: tuple[str, ...] = (),
    key: Any = None,
    extra_rows: int = 0,
) -> list[dict[str, Any]]:
    """The rows an `orbit_bump_sweep` run emits for a machine that tracks its demand.

    Args:
        params: The parameters the run was launched with.
        limit: Keep only the first N rows -- a run still walking the profile.
        residual: ``(step_index, bpm_name) -> float``, added to that BPM's
            reading at that step. This is how a step is pushed outside the band.
        drop: Device names to omit from every row, as a run whose event never
            carried that key.
        key: ``name -> event key``, for the ``f"{name}-{signal}"`` spelling
            ophyd uses for a device with more than one readable child.
        extra_rows: Rows appended past the end of the profile.
    """
    key = key or (lambda name: name)
    scales = bump._profile_scales(params.num, params.sweep)
    reference = _reference(params)
    working = _working(params)
    # `absolute` targets name an orbit position, so the displacement the machine
    # actually makes is measured from the reference -- the same conversion
    # `build_plan` does once, before the profile walk.
    desired: dict[str, float] = {
        target.bpm: (
            target.value - reference[target.bpm] if params.mode == "absolute" else target.value
        )
        for target in params.targets
    }
    desired.update(dict.fromkeys(params.closure_bpms, 0.0))

    def _row(values: dict[str, float]) -> dict[str, Any]:
        return {key(name): value for name, value in values.items() if name not in drop}

    rows: list[dict[str, Any]] = []
    for _ in range(params.baseline_reads):
        baseline = dict(reference)
        baseline.update(working)
        baseline.update(dict.fromkeys(params.detectors, 0.0))
        rows.append(_row(baseline))

    for index, scale in enumerate(scales):
        values: dict[str, float] = {}
        for name, against in reference.items():
            shift = scale * desired.get(name, _MONITOR_SHIFT)
            if residual is not None:
                shift += residual(index, name)
            values[name] = against + shift
        for position, name in enumerate(params.correctors):
            values[name] = working[name] + scale * 0.1 * (position + 1)
        for position, name in enumerate(params.detectors):
            # `0.001 * index` makes the descent leg sit off the ascent at the
            # same amplitude: a hysteresis the per-leg lines have to keep apart.
            values[name] = scale + 0.001 * index + 0.01 * position
        rows.append(_row(values))

    rows.extend([dict(rows[-1]) for _ in range(extra_rows)])
    return rows if limit is None else rows[:limit]


def _panel(figure: Figure, title: str):
    return next(panel for panel in figure.panels if panel.title == title)


def _lines(figure: Figure, title: str) -> LinesMark:
    mark = _panel(figure, title).mark
    assert isinstance(mark, LinesMark)
    return mark


def _labels(figure: Figure, title: str) -> list[str]:
    return [series.label for series in _lines(figure, title).series]


# =========================================================================
# The four panel families, over a converged run
# =========================================================================


def test_a_converged_monodirectional_run_draws_the_four_panels() -> None:
    """Titles, axes, and units are the contract the panel and the agent read.

    The order is stable, so a live figure grows instead of rearranging: the
    detector panel appears last because it is the only one a bump run may
    legitimately not have.
    """
    params = _params(detectors=["det1"])
    figure = bump.render(_rows(params), params)

    assert [panel.title for panel in figure.panels] == [
        "Orbit shift across BPMs",
        "Band residual at constrained BPMs",
        "Corrector offsets",
        "Detector response",
    ]

    orbit = _panel(figure, "Orbit shift across BPMs")
    assert (orbit.x_label, orbit.y_label, orbit.y_units) == (
        "BPM index",
        "Orbit shift",
        "BPM units",
    )
    assert orbit.series_picker is True

    band = _panel(figure, "Band residual at constrained BPMs")
    assert (band.x_label, band.y_label, band.y_units) == (
        "Amplitude step",
        "Residual (reached minus requested)",
        "BPM units",
    )

    offsets = _panel(figure, "Corrector offsets")
    assert (offsets.x_label, offsets.y_label, offsets.y_units) == (
        "Amplitude step",
        "Offset from working point",
        "corrector units",
    )
    # Three or four correctors is the whole schema: nothing to pick between.
    assert offsets.series_picker is False

    detectors = _panel(figure, "Detector response")
    assert (detectors.x_label, detectors.y_label, detectors.y_units) == (
        "Bump amplitude (fraction of requested)",
        "Detector reading",
        None,
    )

    # Every panel is lines: there is no fit here to draw as a matrix.
    assert all(isinstance(panel.mark, LinesMark) for panel in figure.panels)


def test_orbit_shift_draws_one_line_per_step_across_the_requested_bpms() -> None:
    """The bump itself: displaced at the target, flat at the closure BPMs.

    x is the BPM's position in the requested order -- 1-based, targets then
    closure then monitors -- and y is that BPM's shift from the baseline
    reference orbit, which is what makes "closed" visible as a line coming back
    to zero.
    """
    params = _params()
    figure = bump.render(_rows(params), params)
    mark = _lines(figure, "Orbit shift across BPMs")

    scales = bump._profile_scales(params.num, params.sweep)
    assert [series.label for series in mark.series] == [
        "step 1, scale +0.25 (ascent)",
        "step 2, scale +0.5 (ascent)",
        "step 3, scale +0.75 (ascent)",
        "step 4, scale +1 (ascent)",
        "step 5, scale +0.75 (descent)",
        "step 6, scale +0.5 (descent)",
        "step 7, scale +0.25 (descent)",
        "step 8, scale +0 (descent)",
    ]

    for index, series in enumerate(mark.series):
        assert [point.x for point in series.points] == [1.0, 2.0, 3.0, 4.0]
        assert [point.y for point in series.points] == pytest.approx(
            [
                scales[index] * 0.3,  # bpm_t, the target
                0.0,  # bpm_c1, closure
                0.0,  # bpm_c2, closure
                scales[index] * _MONITOR_SHIFT,  # bpm_m1, recorded not constrained
            ],
            abs=1e-12,
        )

    annotations = _panel(figure, "Orbit shift across BPMs").annotations
    assert any("target BPMs (index 1)" in note for note in annotations)
    assert any("closure BPMs (index 2, 3)" in note for note in annotations)
    assert any("carries no lattice positions" in note for note in annotations)


def test_band_residual_traces_every_constrained_bpm_between_two_band_lines() -> None:
    """The residual, plus the band it has to stay in, drawn rather than described."""
    params = _params()
    figure = bump.render(_rows(params), params)
    mark = _lines(figure, "Band residual at constrained BPMs")

    assert [series.label for series in mark.series] == [
        "bpm_t",  # targets first
        "bpm_c1",
        "bpm_c2",  # then closure
        "+tolerance",
        "-tolerance",
    ]

    for series in mark.series[:3]:
        assert [point.x for point in series.points] == [float(n) for n in range(1, 9)]
        assert [point.y for point in series.points] == pytest.approx([0.0] * 8, abs=1e-12)

    for series, sign in ((mark.series[3], 1.0), (mark.series[4], -1.0)):
        assert [point.y for point in series.points] == [sign * params.tolerance] * 8

    annotations = _panel(figure, "Band residual at constrained BPMs").annotations
    assert any("±0.02 BPM units" in note for note in annotations)
    assert "Sweep legs: steps 1-4 ascent; steps 5-8 descent." in annotations
    assert any("terminal working-point verification" in note for note in annotations)


def test_corrector_offsets_are_measured_against_the_baseline_working_point() -> None:
    """Zero means "back where the run found it", which the last step must read."""
    params = _params()
    figure = bump.render(_rows(params), params)
    mark = _lines(figure, "Corrector offsets")

    scales = bump._profile_scales(params.num, params.sweep)
    assert [series.label for series in mark.series] == params.correctors
    for position, series in enumerate(mark.series):
        assert [point.x for point in series.points] == [float(n) for n in range(1, 9)]
        assert [point.y for point in series.points] == pytest.approx(
            [scale * 0.1 * (position + 1) for scale in scales], abs=1e-12
        )
        assert series.points[-1].y == pytest.approx(0.0, abs=1e-12)


def test_detector_response_is_drawn_per_leg_against_signed_amplitude() -> None:
    """One line per leg: plotted against amplitude the profile doubles back, and
    a single line through both legs would close the hysteresis into a scribble."""
    params = _params(detectors=["det1", "det2"])
    figure = bump.render(_rows(params), params)
    mark = _lines(figure, "Detector response")

    assert [series.label for series in mark.series] == [
        "det1 (ascent)",
        "det1 (descent)",
        "det2 (ascent)",
        "det2 (descent)",
    ]

    ascent, descent = mark.series[0], mark.series[1]
    assert [point.x for point in ascent.points] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert [point.x for point in descent.points] == pytest.approx([0.75, 0.5, 0.25, 0.0])

    # The two legs meet at scale 0.75 and do not lie on top of each other there.
    assert ascent.points[2].x == descent.points[0].x
    assert ascent.points[2].y != descent.points[0].y


def test_a_bidirectional_run_draws_all_four_legs() -> None:
    """The negative side is not a special case: same panels, four legs."""
    params = _params(num=3, sweep="bidirectional", detectors=["det1"])
    rows = _rows(params)
    assert len(rows) == params.baseline_reads + 4 * params.num

    figure = bump.render(rows, params)

    step_legs = [
        label.split("(")[-1].rstrip(")") for label in _labels(figure, "Orbit shift across BPMs")
    ]
    assert step_legs == (
        ["ascent"] * 3
        + ["descent"] * 3
        + ["ascent, negative side"] * 3
        + ["descent, negative side"] * 3
    )

    assert _labels(figure, "Detector response") == [
        "det1 (ascent)",
        "det1 (descent)",
        "det1 (ascent, negative side)",
        "det1 (descent, negative side)",
    ]

    band_notes = _panel(figure, "Band residual at constrained BPMs").annotations
    assert (
        "Sweep legs: steps 1-3 ascent; steps 4-6 descent; steps 7-9 ascent, "
        "negative side; steps 10-12 descent, negative side." in band_notes
    )

    # The negative legs really are negative, and the profile still lands on zero.
    orbit = _lines(figure, "Orbit shift across BPMs")
    assert orbit.series[8].label == "step 9, scale -1 (ascent, negative side)"
    assert orbit.series[8].points[0].y == pytest.approx(-0.3)
    assert orbit.series[-1].points[0].y == pytest.approx(0.0, abs=1e-12)


def test_absolute_targets_are_converted_against_the_baseline_reference() -> None:
    """`absolute` names an orbit position; the panels are all reference-relative,
    so the demand is converted once, exactly as `build_plan` converts it."""
    params = _params(mode="absolute", targets=[{"bpm": "bpm_t", "value": 0.8}])
    # bpm_t's reference is 0.5, so the run is asking for a 0.3 displacement --
    # the same bump the relative fixtures above ask for outright.
    figure = bump.render(_rows(params), params)

    band = _lines(figure, "Band residual at constrained BPMs")
    assert [point.y for point in band.series[0].points] == pytest.approx([0.0] * 8, abs=1e-12)


# =========================================================================
# Best effort: the misses are the finding
# =========================================================================


def test_an_unconverged_step_is_named_with_what_it_reached() -> None:
    """A `best_effort` run records the miss instead of stopping -- so the figure
    has to say which step, which BPM, and how far off, or the record is lost."""
    params = _params(best_effort=True)
    rows = _rows(
        params, residual=lambda index, name: 0.5 if (index == 2 and name == "bpm_c1") else 0.0
    )

    figure = bump.render(rows, params)
    annotations = _panel(figure, "Band residual at constrained BPMs").annotations

    assert (
        "Step 3 (scale +0.75) is outside the band at bpm_c1: "
        "reached +0.5 against +0 BPM units requested." in annotations
    )
    assert any("mean of two separate reads" in note for note in annotations)


def test_the_named_bpm_is_the_worst_one_at_that_step() -> None:
    """One note per offending step, naming the BPM that missed by the most: a
    note per BPM per step would bury the step that actually went wrong."""
    params = _params(best_effort=True)
    misses = {"bpm_c1": 0.3, "bpm_c2": 0.5}
    rows = _rows(params, residual=lambda index, name: misses.get(name, 0.0) if index == 1 else 0.0)

    annotations = _panel(bump.render(rows, params), "Band residual at constrained BPMs").annotations
    step_notes = [note for note in annotations if note.startswith("Step ")]

    assert len(step_notes) == 1
    assert step_notes[0].startswith("Step 2 (scale +0.5) is outside the band at bpm_c2:")


def test_a_bpm_the_panel_capped_away_is_still_checked_for_misses() -> None:
    """The band panel draws the first 12 constrained BPMs, but a miss anywhere
    is a miss: a capped panel that also capped its checking would report a
    converged sweep over a machine that missed at BPM 13."""
    closure = [f"bpm_c{index}" for index in range(1, 14)]
    params = _params(best_effort=True, closure_bpms=closure)
    hidden = closure[-1]  # constrained index 13, past `_MAX_BAND_SERIES`

    rows = _rows(
        params, residual=lambda index, name: 0.4 if (index == 0 and name == hidden) else 0.0
    )
    figure = bump.render(rows, params)
    panel = _panel(figure, "Band residual at constrained BPMs")

    drawn = [series.label for series in _lines(figure, "Band residual at constrained BPMs").series]
    assert hidden not in drawn
    assert len(drawn) == bump._MAX_BAND_SERIES + 2  # the band lines are not series it capped
    assert any("Showing the first 12 of 14 constrained BPMs" in note for note in panel.annotations)
    assert any(f"is outside the band at {hidden}:" in note for note in panel.annotations)


def test_past_six_misses_the_notes_stop_listing_and_start_counting() -> None:
    """A run that missed everywhere is one finding, not a wall of them."""
    params = _params(best_effort=True, num=6)
    rows = _rows(
        params, residual=lambda index, name: 0.4 if (index < 8 and name == "bpm_c1") else 0.0
    )

    annotations = _panel(bump.render(rows, params), "Band residual at constrained BPMs").annotations

    assert len([note for note in annotations if note.startswith("Step ")]) == bump._MAX_MISS_NOTES
    assert "2 further step(s) are outside the band." in annotations


def test_a_converged_run_carries_no_miss_notes() -> None:
    """The negative control: the format above only appears when something missed."""
    params = _params()
    annotations = _panel(
        bump.render(_rows(params), params), "Band residual at constrained BPMs"
    ).annotations

    assert not [note for note in annotations if note.startswith("Step ")]
    assert not [note for note in annotations if "outside the band" in note]


# =========================================================================
# Caps: what a facility-scale run is allowed to put on the wire
# =========================================================================


def test_a_long_profile_thins_its_step_series_and_keeps_both_ends() -> None:
    """Sixteen steps is not sixteen lines over four BPMs.

    The drawn set comes from the profile length -- fixed before the run starts
    -- so it does not swap between poll ticks, and it keeps both endpoints
    because the first and last steps are the ones with something to prove.
    """
    params = _params(num=8)  # 16 steps
    figure = bump.render(_rows(params), params)
    labels = _labels(figure, "Orbit shift across BPMs")

    assert len(labels) == bump._MAX_STEP_SERIES
    assert labels[0].startswith("step 1, ")
    assert labels[-1].startswith("step 16, ")
    assert any(
        "Showing 12 of 16 amplitude steps" in note
        for note in _panel(figure, "Orbit shift across BPMs").annotations
    )

    # The residual panel is per BPM, not per step: it keeps every step.
    band = _lines(figure, "Band residual at constrained BPMs")
    assert len(band.series[0].points) == 16


def test_the_detector_budget_is_spent_across_legs_not_per_leg() -> None:
    """Each detector is drawn once per leg, so twelve series is six detectors on
    a two-leg profile and three on a four-leg one."""
    detectors = [f"det{index}" for index in range(1, 11)]

    mono = _params(detectors=detectors)
    mono_labels = _labels(bump.render(_rows(mono), mono), "Detector response")
    assert len(mono_labels) == bump._MAX_DETECTOR_SERIES
    assert mono_labels[-1] == "det6 (descent)"

    bidi = _params(num=3, sweep="bidirectional", detectors=detectors)
    bidi_labels = _labels(bump.render(_rows(bidi), bidi), "Detector response")
    assert len(bidi_labels) == bump._MAX_DETECTOR_SERIES
    assert bidi_labels[-1] == "det3 (descent, negative side)"

    assert any(
        "Showing the first 6 of 10 detectors" in note
        for note in _panel(bump.render(_rows(mono), mono), "Detector response").annotations
    )


def test_the_correctors_are_never_capped() -> None:
    """`PARAMS` admits three or four and nothing else, so there is nothing to cap
    -- and dropping one would hide a corrector the run left off its working point."""
    params = _params(
        correctors=["c1", "c2", "c3", "c4"], closure_bpms=["bpm_c1", "bpm_c2", "bpm_c3"]
    )
    figure = bump.render(_rows(params), params)

    assert _labels(figure, "Corrector offsets") == ["c1", "c2", "c3", "c4"]


def test_every_series_is_decimated_including_the_flat_band_lines() -> None:
    """The band lines are as long as the sweep, so they cost what a BPM trace does.

    A flat line looks free and is not: two of them at full length is a fifth of
    the point budget spent on a number the annotation already states.
    """
    params = _params(num=1001, bpms=[])  # 2002 amplitude steps
    figure = bump.render(_rows(params), params)
    mark = _lines(figure, "Band residual at constrained BPMs")

    assert mark.decimated is True
    for series in mark.series:
        assert series.decimated is True
        assert len(series.points) == DEFAULT_MAX_POINTS
        assert series.source_points == 2002

    band_line = mark.series[-1]
    assert band_line.label == "-tolerance"
    assert {point.y for point in band_line.points} == {-params.tolerance}

    # The corrector traces are the same length and thinned the same way.
    offsets = _lines(figure, "Corrector offsets")
    assert offsets.decimated is True
    assert all(len(series.points) == DEFAULT_MAX_POINTS for series in offsets.series)


# =========================================================================
# Mid-run and degraded row sets
# =========================================================================


def test_a_run_still_walking_the_profile_draws_its_prefix_and_says_so() -> None:
    """Three steps of eight: the leading rows still mean what they mean."""
    params = _params()
    rows = _rows(params, limit=params.baseline_reads + 3)

    figure = bump.render(rows, params)
    lead = figure.panels[0].annotations

    assert lead[0] == (
        "3 of 8 amplitude steps recorded so far; the rest of the profile is not in this figure."
    )
    assert len(_labels(figure, "Orbit shift across BPMs")) == 3
    assert len(_lines(figure, "Corrector offsets").series[0].points) == 3

    band_notes = _panel(figure, "Band residual at constrained BPMs").annotations
    assert "Sweep legs: steps 1-3 ascent." in band_notes
    # The terminal verification has not happened yet, so nothing claims it did.
    assert not any("terminal working-point verification" in note for note in band_notes)


def test_rows_past_the_end_of_the_profile_are_annotated_not_drawn() -> None:
    """Never expected -- but drawing them would attribute them to a step."""
    params = _params()
    figure = bump.render(_rows(params, extra_rows=2), params)

    assert figure.panels[0].annotations[0] == (
        "This run produced 2 row(s) past the end of its amplitude profile, which are not drawn."
    )
    assert len(_labels(figure, "Orbit shift across BPMs")) == 8


def test_baseline_rows_alone_draw_nothing() -> None:
    """Every panel is a statement about an amplitude step. Before the first one
    there is no honest partial view, only an empty figure that keeps polling."""
    params = _params()
    figure = bump.render(_rows(params, limit=params.baseline_reads), params)

    assert figure.panels == []


def test_fewer_rows_than_the_baseline_draw_nothing() -> None:
    """The reference orbit is the baseline mean; an incomplete baseline is not one."""
    params = _params()
    figure = bump.render(_rows(params, limit=params.baseline_reads - 1), params)

    assert figure.panels == []


def test_a_missing_bpm_key_is_a_gap_that_keeps_the_axis_aligned() -> None:
    """The BPM axis is positional, so a BPM that never reported keeps its slot --
    dropping it would shift every BPM after it onto the wrong index."""
    params = _params()
    figure = bump.render(_rows(params, drop=("bpm_c2",)), params)

    orbit = _lines(figure, "Orbit shift across BPMs")
    for series in orbit.series:
        assert [point.x for point in series.points] == [1.0, 2.0, 3.0, 4.0]
        assert series.points[2].y is None
        assert series.points[3].y is not None

    # It keeps its line in the residual panel too, drawn as all gaps.
    band = _lines(figure, "Band residual at constrained BPMs")
    missing = next(series for series in band.series if series.label == "bpm_c2")
    assert all(point.y is None for point in missing.points)


def test_a_hyphenated_event_key_resolves_to_its_device() -> None:
    """ophyd keys a multi-signal device as `f"{name}-{signal}"`; render resolves
    exact-then-prefix, so both spellings draw the same figure."""
    params = _params()
    bare = bump.render(_rows(params), params)
    hyphenated = bump.render(_rows(params, key=lambda name: f"{name}-readback"), params)

    assert bare == hyphenated


def test_a_panel_with_nothing_finite_is_dropped_not_drawn_empty() -> None:
    """The correctors never reported: three flat gap lines say less than no panel."""
    params = _params()
    figure = bump.render(_rows(params, drop=("c1", "c2", "c3")), params)

    assert [panel.title for panel in figure.panels] == [
        "Orbit shift across BPMs",
        "Band residual at constrained BPMs",
    ]


def test_the_detector_panel_is_absent_when_no_detectors_were_requested() -> None:
    """Detectors are optional, and an empty panel is not a view of nothing."""
    params = _params()
    titles = [panel.title for panel in bump.render(_rows(params), params).panels]

    assert "Detector response" not in titles
    assert len(titles) == 3


def test_one_panel_failing_does_not_take_the_figure_with_it(monkeypatch) -> None:
    """`_panel_or_none` exercised rather than assumed: the other three still serve."""

    def _boom(sweep, params):
        raise RuntimeError("miss detection blew up")

    monkeypatch.setattr(bump, "_misses", _boom)

    params = _params(detectors=["det1"])
    figure = bump.render(_rows(params), params)

    assert [panel.title for panel in figure.panels] == [
        "Orbit shift across BPMs",
        "Corrector offsets",
        "Detector response",
    ]


# =========================================================================
# Totality: a figure is a view, and its failure is never the run's
# =========================================================================


def test_no_rows_render_an_empty_figure() -> None:
    assert bump.render([], _params()).panels == []


def test_rows_from_a_different_plan_do_not_raise() -> None:
    """None of the named devices appear: nothing to draw, still a figure."""
    rows = [{"motor": float(index), "det": float(index) ** 2} for index in range(20)]

    assert bump.render(rows, _params()).panels == []


def test_non_numeric_rows_do_not_raise() -> None:
    """Strings where readings were expected degrade to no panels."""
    params = _params()
    rows = [
        dict.fromkeys([*_bpm_names(params), *params.correctors], "not a number")
        for _ in range(params.baseline_reads + 8)
    ]

    assert bump.render(rows, params).panels == []


def test_a_non_finite_reading_is_a_gap_not_a_raise() -> None:
    """`Point.x` rejects a non-finite coordinate and `Point.y` nulls one, so a
    NaN that survived the row adapter has to become a gap before it gets there."""
    params = _params()
    rows = _rows(params)
    rows[params.baseline_reads + 2]["bpm_t"] = float("nan")
    rows[params.baseline_reads + 3]["bpm_t"] = float("inf")

    figure = bump.render(rows, params)
    band = _lines(figure, "Band residual at constrained BPMs")
    target = next(series for series in band.series if series.label == "bpm_t")

    assert target.points[2].y is None
    assert target.points[3].y is None
    assert target.points[4].y is not None


def test_a_failure_below_every_panel_serves_an_empty_figure(monkeypatch) -> None:
    """The outer catch: `render` is total even when the row resolution itself goes."""

    def _boom(rows, params):
        raise RuntimeError("row attribution blew up")

    monkeypatch.setattr(bump, "_prepare", _boom)

    figure = bump.render(_rows(_params()), _params())
    assert figure.panels == []
    assert figure.partial is True


def test_partial_and_source_are_placeholders_the_route_overwrites() -> None:
    """The team-wide render convention: always `partial=True, source="live"`.

    A render sees rows, never their provenance, so neither field can be its
    answer to give -- the figure route stamps the truth onto both on every
    path. `True` is the honest direction for the placeholder: a forgotten
    overwrite costs a client extra polling, never a false "settled".
    """
    params = _params(detectors=["det1"])
    finished = _rows(params)
    mid_flight = _rows(params, limit=params.baseline_reads + 2)
    degraded = [{"motor": 1.0}]

    for rows in (finished, mid_flight, degraded, []):
        figure = bump.render(rows, params)
        assert figure.partial is True
        assert figure.source == "live"
        assert figure.reason is None


def test_figure_round_trips_through_model_validate() -> None:
    """What the route dumps to JSON validates back to the same marks."""
    params = _params(detectors=["det1"])
    figure = bump.render(_rows(params), params)

    assert Figure.model_validate(figure.model_dump()) == figure


def test_render_conformance_binding_pins_the_plan_spec_signature() -> None:
    """The loader holds `PlanSpec[Any]`, so a module-level `render` is not
    type-checked by merely existing; this binding is what makes mypy check it."""
    assert bump._RENDER_CONFORMANCE is bump.render
