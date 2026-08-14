"""Shipped plan: ``orm``, an orbit-response-matrix sweep.

Discovered via the layered directory catalog's ``shipped`` tier (a folder
scan of this package's ``plans_core/`` dir — see ``plan_loader.py``): sweep
each corrector, one at a time, over a bounded current range, reading every
BPM detector at each point.

Device-agnostic: ``correctors``/``detectors`` are resolved by string name
against whatever ``devices`` dict the bridge passes in; nothing here names a
facility PV or a fixed device set.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, Literal

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from pydantic import BaseModel, Field, model_validator

# Absolute imports, not relative: the plan loader execs this file by path under
# a synthetic `sys.modules` name with no parent package (see
# `plan_loader._load_plan_module_from_path`), so a relative import raises at
# load time and quarantines the plan.
from osprey.services.bluesky_bridge.figure import (
    BarsMark,
    Figure,
    HeatmapMark,
    LinesMark,
    Panel,
    Point,
    Series,
    decimate,
)
from osprey.services.bluesky_bridge.orm_analysis import (
    SlicedResponseFit,
    column_anomaly,
    row_anomaly,
    sliced_response_matrix,
)

logger = logging.getLogger("osprey.services.bluesky_bridge.plans_core.orm")

PLAN_METADATA = {
    "name": "orm",
    "description": (
        "Sweep each corrector over a bounded current range, reading all BPM "
        "detectors at every point, to measure an orbit-response matrix."
    ),
    "category": "accelerator",
    "required_devices": ["correctors", "detectors"],
    "writes": True,
}


class PARAMS(BaseModel):
    """Parameters for ``orm``: correctors to sweep, BPMs to read.

    Sweeps each corrector in ``correctors``, one at a time, over ``num``
    evenly-spaced currents, reading every detector in ``detectors`` at each
    point. ``sweep`` selects the current range: ``bidirectional`` spans the
    symmetric ``[-span_a, +span_a]`` (kicks both ways, so a linear fit rejects
    a corrector's hysteresis/offset); ``monodirectional`` spans ``[0, span_a]``
    (one-sided, for a corrector that should never be driven negative).

    The ``x-widget`` schema hints steer the plan panel's parameter GUI —
    device lists render as scrollable channel columns, ``sweep`` as a two-way
    segmented control — without changing what this model validates.
    """

    correctors: list[str] = Field(
        ...,
        min_length=1,
        title="Correctors",
        description="Corrector device names to sweep, one at a time.",
        json_schema_extra={"x-widget": "channel-list"},
    )
    detectors: list[str] = Field(
        ...,
        min_length=1,
        title="BPMs",
        description="BPM detector device names to read at every point.",
        json_schema_extra={"x-widget": "channel-list"},
    )
    span_a: float = Field(
        ...,
        gt=0,
        le=10.0,
        title="Max kick (A)",
        description="Maximum corrector kick, in amps, at the far end of the sweep.",
    )
    num: int = Field(
        ...,
        ge=3,
        title="Number of steps",
        description="Number of evenly-spaced current points per corrector.",
    )
    sweep: Literal["bidirectional", "monodirectional"] = Field(
        default="bidirectional",
        title="Sweep direction",
        description=(
            "bidirectional sweeps [-span_a, +span_a]; monodirectional sweeps [0, +span_a]."
        ),
        json_schema_extra={"x-widget": "segmented"},
    )

    @model_validator(mode="after")
    def _correctors_and_detectors_disjoint(self) -> PARAMS:
        """Reject a device named as both a driven corrector and a read BPM."""
        overlap = set(self.correctors) & set(self.detectors)
        if overlap:
            raise ValueError(
                f"correctors and detectors must be disjoint (overlap: {sorted(overlap)})"
            )
        return self


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    """Build the orbit-response-matrix sweep generator.

    Mirrors the built-in ``orm`` plan's (``plans.py``) idiom: for each
    corrector, sweep ``num`` currents evenly spaced over the range ``sweep``
    selects — ``[-span_a, +span_a]`` (bidirectional) or ``[0, +span_a]``
    (monodirectional) — moving the corrector then ``trigger_and_read``-ing
    every corrector together with every BPM detector in one bundle, so each
    point emits exactly one event carrying the driven corrector's current
    alongside every BPM reading. Each corrector is restored to 0 A once its
    own sweep finishes, including on abort (the ``try``/``finally`` runs on
    ``GeneratorExit`` too) — a restore failure is caught and logged rather
    than allowed to replace an in-flight sweep exception.
    """
    correctors = [(name, devices[name]) for name in params.correctors]
    corrector_devices = [corrector for _, corrector in correctors]
    detector_devices = [devices[name] for name in params.detectors]
    if params.sweep == "monodirectional":
        step = params.span_a / (params.num - 1)
        currents = [i * step for i in range(params.num)]
    else:
        step = (2 * params.span_a) / (params.num - 1)
        currents = [-params.span_a + i * step for i in range(params.num)]

    all_devices = corrector_devices + detector_devices

    @bpp.stage_decorator(all_devices)
    @bpp.run_decorator()
    def _sweep():
        for name, corrector in correctors:
            try:
                for current in currents:
                    yield from bps.mv(corrector, current)
                    yield from bps.trigger_and_read(all_devices)
            finally:
                try:
                    yield from bps.mv(corrector, 0.0)
                except Exception:
                    logger.warning(
                        "orm plan: failed to restore corrector %s to 0 A "
                        "during cleanup; preserving the original error",
                        name,
                        exc_info=True,
                    )

    return _sweep()


# --- Live view ---------------------------------------------------------------
#
# `render()` runs in the bridge API process on every poll tick of every client
# watching an `orm` run, over the run's FULL stored row set. Everything below
# is sized for that: one vectorized fit (`sliced_response_matrix`), no per-row
# Python scan, and hard caps on how much of a facility-scale run reaches the
# wire.

_LIVE_BUFFER_ROW_CAP = 10_000
"""Mirror of the bridge live buffer's per-run row cap.

`render()` is handed rows with no word on where they came from or whether they
are all of them, so it recognizes a capped buffer arithmetically: the buffer
stores exactly this many rows and then stops (`live_rows._MAX_ROWS_PER_RUN`),
so a row set of exactly this length for a sweep that should have produced more
is a run whose stored rows are a strict subset of what it measured. This module
cannot import that private constant -- a facility-authored plan could not
either, and the loader execs plan files outside the package -- so the value is
duplicated here and pinned equal by `test_orm_render.py`, which fails on drift
rather than letting a stale copy silently mis-shape a figure.
"""

_MAX_TRACE_PANELS = 8
"""Most corrector sweeps drawn as trace panels. A 70-corrector run is 70 panels
of 100 lines each -- neither the browser nor the agent's context can take it."""

_MAX_TRACE_SERIES = 12
"""Most BPM traces per sweep panel. The first N in the requested order, not the
N largest: series identity has to stay stable from tick to tick, and a
"largest" selection would swap lines in and out as the sweep fills in."""


def _sweep_series(fit: SlicedResponseFit, index: int) -> tuple[list[Series], bool, int, int]:
    """Corrector *index*'s BPM traces: series, decimated, points, samples drawn.

    A sample whose corrector current is non-finite is dropped -- there is no
    place on the x axis for a point with no x (and `Point.x` rejects it) --
    while a non-finite BPM reading is kept and becomes a drawn gap. The last
    element is how many samples survived that, which is what tells a caller
    whether this corrector reported a current at all.
    """
    currents = fit.currents[index]
    readings = fit.readings[index]
    kept = [k for k, current in enumerate(currents) if math.isfinite(float(current))]

    series: list[Series] = []
    decimated = False
    source_points = 0
    for i, detector in enumerate(fit.detectors[:_MAX_TRACE_SERIES]):
        points = [Point(x=float(currents[k]), y=float(readings[k, i])) for k in kept]
        thinned = decimate(points)
        series.append(Series(label=detector, **thinned._asdict()))
        decimated = decimated or thinned.decimated
        source_points += thinned.source_points
    return series, decimated, source_points, len(kept)


def _trace_panels(fit: SlicedResponseFit, num: int, lead_annotations: list[str]) -> list[Panel]:
    """One panel per traced corrector: BPM readings against that corrector's current.

    A corrector that has recorded no current yet -- the sweep has not reached
    it, or the run's rows never carried that device -- gets no panel at all; an
    empty plot says less than no plot. Past `_MAX_TRACE_PANELS` the *most
    recently swept* correctors are kept, not the first: the plan sweeps in
    order, so the tail is where a live run's in-flight sweep is, and a settled
    run's tail is as arbitrary a choice as its head.
    """
    traced = [j for j in range(len(fit.correctors)) if fit.currents[j].size]
    hidden = max(0, len(traced) - _MAX_TRACE_PANELS)
    if hidden:
        traced = traced[-_MAX_TRACE_PANELS:]
    shown_detectors = min(len(fit.detectors), _MAX_TRACE_SERIES)

    panels: list[Panel] = []
    for j in traced:
        series, decimated, source_points, samples = _sweep_series(fit, j)
        if not samples:
            continue

        annotations: list[str] = []
        if not panels:
            annotations.extend(lead_annotations)
            if hidden:
                annotations.append(
                    f"Showing the {len(traced)} most recently swept of "
                    f"{len(fit.correctors)} correctors."
                )
        if shown_detectors < len(fit.detectors):
            annotations.append(f"Showing the first {shown_detectors} of {len(fit.detectors)} BPMs.")
        if not fit.complete[j]:
            recorded = int(fit.currents[j].shape[0])
            if recorded < num:
                annotations.append(
                    f"This sweep is still in flight: {recorded} of {num} points "
                    "recorded, so no response is fitted for it yet."
                )
            else:
                annotations.append(
                    "This corrector's recorded currents never moved, so no "
                    "response could be fitted for it."
                )
        panels.append(
            Panel(
                title=f"{fit.correctors[j]} sweep",
                x_label="Corrector current",
                x_units="A",
                y_label="BPM reading",
                annotations=annotations,
                mark=LinesMark(series=series, decimated=decimated, source_points=source_points),
            )
        )
    return panels


def _fit_panels(fit: SlicedResponseFit) -> list[Panel]:
    """The response matrix and its two model-free anomaly views.

    Only completed sweeps have columns, so the heatmap's x axis is
    `fitted_correctors` (a subsequence of the requested correctors) and says so
    when the two differ -- an in-flight corrector must not read as a dead one.
    """
    incomplete = len(fit.correctors) - len(fit.fitted_correctors)
    matrix_note = (
        [
            f"{len(fit.fitted_correctors)} of {len(fit.correctors)} corrector sweeps "
            "have completed; only those have a column here."
        ]
        if incomplete
        else []
    )

    return [
        Panel(
            title="Response matrix",
            x_label="Corrector",
            y_label="BPM",
            annotations=matrix_note,
            mark=HeatmapMark(
                x_labels=list(fit.fitted_correctors),
                y_labels=list(fit.detectors),
                values=[[float(cell) for cell in row] for row in fit.matrix],
                value_label="Response slope",
                value_units="BPM units / A",
                source_points=int(fit.matrix.size),
            ),
        ),
        Panel(
            title="Corrector anomaly score",
            x_label="Corrector",
            y_label="Anomaly score",
            annotations=[
                "Each corrector's column scored against the median of its peers "
                "-- higher is more unlike the rest of the machine."
            ],
            mark=BarsMark(
                label="Corrector anomaly",
                categories=list(fit.fitted_correctors),
                values=[float(score) for score in column_anomaly(fit.matrix)],
                source_points=len(fit.fitted_correctors),
            ),
        ),
        Panel(
            title="BPM anomaly score",
            x_label="BPM",
            y_label="Anomaly score",
            annotations=[
                "Each BPM's row scored against the median of its peers -- "
                "higher is more unlike the rest of the machine."
            ],
            mark=BarsMark(
                label="BPM anomaly",
                categories=list(fit.detectors),
                values=[float(score) for score in row_anomaly(fit.matrix)],
                source_points=len(fit.detectors),
            ),
        ),
    ]


def _build(rows: list[dict[str, Any]], params: PARAMS, *, with_fit: bool) -> Figure:
    """Assemble the figure. Raises freely -- `render` is what must not."""
    expected_rows = len(params.correctors) * params.num
    capped = len(rows) == _LIVE_BUFFER_ROW_CAP and expected_rows > _LIVE_BUFFER_ROW_CAP

    fit = sliced_response_matrix(rows, params.correctors, params.detectors, params.num)

    lead_annotations: list[str] = []
    if capped:
        lead_annotations.append(
            f"This run produced more rows than the bridge stores per run "
            f"({_LIVE_BUFFER_ROW_CAP:,} of about {expected_rows:,} expected), so the "
            "response matrix and anomaly scores are not shown -- they would be "
            "computed over whichever correctors happened to fit under the cap, and "
            "an anomaly score compares each corrector against its peers."
        )

    panels = _trace_panels(fit, params.num, lead_annotations)
    if with_fit and not capped and fit.fitted_correctors:
        panels.extend(_fit_panels(fit))

    # Placeholders: the figure route stamps the truth onto both on every path.
    # A render sees rows, never their provenance, so `partial=True` is the
    # honest direction for a forgotten overwrite -- extra polling, never a
    # false "settled".
    return Figure(panels=panels, partial=True, source="live")


def render(rows: list[dict[str, Any]], params: PARAMS) -> Figure:
    """The `orm` plan's own view of a run: sweep traces, then the fitted matrix.

    Panel order is stable, so a live figure grows rather than rearranging:

    1. One **sweep trace** panel per corrector that has recorded a point --
       every BPM reading against that corrector's own current. These always
       appear, for a `monodirectional` sweep as readily as a `bidirectional`
       one and for a run three points in as readily as a finished one.
    2. Once at least one corrector's sweep has completed, the **response
       matrix** heatmap (BPMs against correctors, labelled with the real device
       names), then the **corrector** and **BPM anomaly score** bars from
       `orm_analysis`'s peer-structure detectors.

    Rows are attributed to correctors by acquisition index -- corrector ``j``'s
    samples are ``rows[j * num : (j + 1) * num]``, which is what this plan's
    `build_plan` emits -- so the caller owes emission order with nothing dropped
    from the front. The one incompleteness this function can see for itself is
    a **capped row set**: exactly `_LIVE_BUFFER_ROW_CAP` rows for a sweep whose
    parameters call for more. There the traces still stand (each is a claim
    about individual rows) but the matrix and anomaly panels are suppressed,
    because a peer-comparison score computed over whichever correctors fit
    under the cap is a statement about the cap, not about the machine. An
    annotation on the first panel says so.

    **Never raises.** A figure is a view, not a result: any internal failure
    degrades to the traces alone, and a failure in those to a figure with no
    panels, so the route always has something to serve.

    **`partial` and `source` are placeholders the route overwrites.** This
    function sees rows, not their provenance, so it returns ``partial=True,
    source="live"`` unconditionally and the figure route stamps the truth onto
    both on every path. ``True`` is the honest direction for the placeholder: a
    forgotten overwrite costs a client extra polling, never a false "settled".

    Args:
        rows: The run's event `data` dicts in emission order -- the run's whole
            stored set, never a window of it.
        params: The parameters the run was launched with.

    Returns:
        A `Figure` whose panels are as complete as the rows allow.
    """
    try:
        return _build(rows, params, with_fit=True)
    except Exception:
        logger.warning("orm render failed; falling back to sweep traces only", exc_info=True)
    try:
        return _build(rows, params, with_fit=False)
    except Exception:
        logger.warning("orm render failed entirely; serving an empty figure", exc_info=True)
        return Figure(panels=[], partial=True, source="live")


_RENDER_CONFORMANCE: Callable[[list[dict[str, Any]], PARAMS], Figure] = render
"""Static proof that `render` matches `PlanSpec.render`'s signature.

The loader holds `PlanSpec[Any]`, so a module-level `render` is not type-checked
by merely existing (see `plan_loader._resolve_render`); this binding is what
makes mypy check it here, where the mistake would be made.
"""
