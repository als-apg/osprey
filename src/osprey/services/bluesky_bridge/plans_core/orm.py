"""Shipped plan: ``orm``, an orbit-response-matrix sweep.

Discovered via the layered directory catalog's ``shipped`` tier (a folder
scan of this package's ``plans_core/`` dir — see ``plan_loader.py``): kick
each corrector channel, one at a time, either side of the working point it
was already holding, reading every BPM channel at each point, then put it
back.

Device-agnostic: ``correctors``/``bpms`` are resolved by string name against
whatever ``devices`` dict the bridge passes in; nothing here names a facility
PV or a fixed device set. Both fields declare their channel role, so every
consumer — the load gate, the enqueue pre-check, the dry run, the pre-flight
trajectory — reads what this plan drives and what it reads rather than
guessing it from a field name.

This plan opens its own run, so it stamps its own run metadata through
``scan_metadata()``: the channels it moves and reads, and the total point
count (one point per corrector per swept current). No dimensionality hint —
the sweeps are serial and per-corrector, not one continuous traversal, so
there is no truthful value for one.
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
    RowWindow,
    Series,
    decimate,
)
from osprey.services.bluesky_bridge.orm_analysis import (
    SlicedResponseFit,
    column_anomaly,
    row_anomaly,
    singular_values,
    sliced_response_matrix,
)
from osprey.services.bluesky_bridge.plan_fields import (
    MovableChannels,
    ReadableChannels,
    scan_metadata,
)

logger = logging.getLogger("osprey.services.bluesky_bridge.plans_core.orm")

PLAN_METADATA = {
    "name": "orm",
    "description": (
        "Kick each corrector channel either side of its own pre-scan working "
        "point, reading every BPM channel at every point, to measure an "
        "orbit-response matrix. Every corrector is put back where it was."
    ),
    "writes": True,
}


class PARAMS(BaseModel):
    """Parameters for ``orm``: correctors to sweep, BPMs to read.

    Sweeps each corrector in ``correctors``, one at a time, over ``num``
    evenly-spaced kicks *away from that corrector's own pre-scan working
    point*, reading every channel in ``bpms`` at each point. ``sweep``
    selects the kick range: ``bidirectional`` spans the symmetric
    ``[-span_a, +span_a]`` (kicks both ways, so a linear fit rejects a
    corrector's hysteresis/offset); ``monodirectional`` spans ``[0, span_a]``
    (one-sided, for a corrector that should never be driven below where it
    already sits).

    ``span_a`` carries no upper bound of its own. It is an excursion, not an
    absolute setpoint, and how large an excursion a corrector tolerates is a
    property of the deployment, not of this schema — the connector's
    reference monitor checks every setpoint against that project's own
    ``channel_limits.json`` at write time and refuses (aborting the run) what
    the machine will not take. A literal ceiling here would only ever be one
    facility's number.

    The ``x-widget`` schema hints steer the plan panel's parameter GUI —
    device lists render as scrollable channel columns, ``sweep`` as a two-way
    segmented control — without changing what this model validates.
    """

    correctors: MovableChannels = Field(
        ...,
        min_length=1,
        title="Correctors",
        description="Corrector channel names to sweep, one at a time.",
        json_schema_extra={"x-widget": "channel-list"},
    )
    bpms: ReadableChannels = Field(
        ...,
        min_length=1,
        title="BPMs",
        description="BPM channel names to read at every point.",
        json_schema_extra={"x-widget": "channel-list"},
    )
    span_a: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        title="Max kick (A)",
        description=(
            "Largest kick, in amps, away from a corrector's own pre-scan "
            "working point at the far end of the sweep."
        ),
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
    def _correctors_and_bpms_disjoint(self) -> PARAMS:
        """Reject a channel named as both a driven corrector and a read BPM."""
        overlap = set(self.correctors) & set(self.bpms)
        if overlap:
            raise ValueError(f"correctors and bpms must be disjoint (overlap: {sorted(overlap)})")
        return self


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    """Build the orbit-response-matrix sweep generator.

    For each corrector in turn: read where it is already sitting, sweep it
    over ``num`` kicks evenly spaced across the range ``sweep`` selects —
    ``[-span_a, +span_a]`` (bidirectional) or ``[0, +span_a]``
    (monodirectional) — and put it back. Each kick is applied *relative to
    that corrector's own pre-scan working point*, and at each point the plan
    ``trigger_and_read``-s every corrector together with every BPM channel
    in one bundle, so each point emits exactly one event carrying the driven
    corrector's current alongside every BPM reading.

    **Relative, because a machine with beam in it is not at zero.** A ring
    running corrected orbit holds every corrector at a nonzero
    orbit-correction working point. Sweeping absolute currents about zero
    would measure the response about a point the machine is not at, and
    "restoring" to a literal 0 A would drop the entire correction the ring
    was holding — on a stored beam, that is the orbit gone. Reading each
    corrector's working point costs one extra read per corrector and makes
    the plan mean the same thing on a real ring as on a virtual accelerator
    whose correctors happen to idle at zero.

    The read is `bps.rd`, so the working point is the corrector's own
    readback — which is the right value to restore under this bridge's
    device contract, where a settable's ``set()`` does not complete until
    the readback agrees with the demand (see ``devices/connector.py``'s
    ``ConnectorSettable``). It happens BEFORE the ``try``, so a corrector
    whose read fails is never entered at all and the restore below can never
    run without a target.

    Each corrector is restored to its recorded working point once its own
    sweep finishes, including on abort (the ``try``/``finally`` runs on
    ``GeneratorExit`` too) — a restore failure is caught and logged rather
    than allowed to replace an in-flight sweep exception.

    The run this opens is stamped with ``scan_metadata()``: the correctors it
    moves, the BPMs it reads, and ``num`` points per corrector across every
    corrector. Sweeps are serial and per-corrector rather than one continuous
    traversal, so no dimensionality hint is declared.

    Raises:
        ValueError: A corrector read back a non-finite working point. Every
            setpoint derived from it would be non-finite too, and a NaN
            demand is not something a limits check can refuse (``nan <
            low`` and ``nan > high`` are both false), so it would reach the
            IOC and only then fail as a readback-settle timeout. Refusing
            here means no write is attempted at all.
    """
    correctors = [(name, devices[name]) for name in params.correctors]
    corrector_devices = [corrector for _, corrector in correctors]
    bpm_devices = [devices[name] for name in params.bpms]
    if params.sweep == "monodirectional":
        step = params.span_a / (params.num - 1)
        kicks = [i * step for i in range(params.num)]
    else:
        step = (2 * params.span_a) / (params.num - 1)
        kicks = [-params.span_a + i * step for i in range(params.num)]

    all_devices = corrector_devices + bpm_devices

    @bpp.stage_decorator(all_devices)
    @bpp.run_decorator(
        md=scan_metadata(
            movable=params.correctors,
            readable=params.bpms,
            points=params.num * len(params.correctors),
        )
    )
    def _sweep():
        for name, corrector in correctors:
            working_point = float((yield from bps.rd(corrector)))
            if not math.isfinite(working_point):
                raise ValueError(
                    f"orm plan: corrector {name!r} read back a non-finite working "
                    f"point ({working_point}); refusing to sweep relative to it"
                )
            try:
                for kick in kicks:
                    yield from bps.mv(corrector, working_point + kick)
                    yield from bps.trigger_and_read(all_devices)
            finally:
                try:
                    yield from bps.mv(corrector, working_point)
                except Exception:
                    logger.warning(
                        "orm plan: failed to restore corrector %s to its pre-scan "
                        "working point (%r) during cleanup; preserving the original "
                        "error",
                        name,
                        working_point,
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

_MAX_TRACE_PANELS = 8
"""Most corrector sweeps drawn as trace panels. A 70-corrector run is 70 panels
of 100 lines each -- neither the browser nor the agent's context can take it."""

_MAX_TRACE_SERIES = 12
"""Most BPM traces per sweep panel. The first N in the requested order, not the
N largest: series identity has to stay stable from tick to tick, and a
"largest" selection would swap lines in and out as the sweep fills in."""

_SWEEPS_SECTION = "Per-corrector sweeps"
"""Section holding the raw sweep traces. They are the run's evidence rather
than its finding, so once there is a finding to lead with they move below it
and the reader picks the corrector they want."""

_SPECTRUM_SECTION = "Singular values"
"""Section holding the singular value spectrum -- a second opinion on the
matrix above, consulted rather than scanned."""


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
    for i, bpm in enumerate(fit.bpms[:_MAX_TRACE_SERIES]):
        points = [Point(x=float(currents[k]), y=float(readings[k, i])) for k in kept]
        thinned = decimate(points)
        series.append(Series(label=bpm, **thinned._asdict()))
        decimated = decimated or thinned.decimated
        source_points += thinned.source_points
    return series, decimated, source_points, len(kept)


def _trace_panels(
    fit: SlicedResponseFit,
    num: int,
    lead_annotations: list[str],
    *,
    section: str | None,
) -> list[Panel]:
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
    shown_bpms = min(len(fit.bpms), _MAX_TRACE_SERIES)

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
        if shown_bpms < len(fit.bpms):
            annotations.append(f"Showing the first {shown_bpms} of {len(fit.bpms)} BPMs.")
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
                section=section,
                mark=LinesMark(series=series, decimated=decimated, source_points=source_points),
            )
        )
    return panels


def _response_by_bpm_panel(fit: SlicedResponseFit) -> Panel:
    """The matrix read column-wise: each corrector's response across the BPMs.

    The same numbers as the heatmap in the encoding that makes a *shape*
    legible -- the sign oscillation of betatron phase advance, a flat line for
    a dead corrector, a spike every trace shares for a dead BPM. Every fitted
    corrector ships as its own series and the reader picks which are drawn, so
    changing the comparison costs no fetch.

    The BPM axis is the requested BPM order, which is not `s` order: the
    plan is device-agnostic and carries no lattice positions, so the panel says
    so rather than implying a geometry it cannot know.
    """
    series = [
        Series(
            label=corrector,
            points=[
                Point(
                    x=float(i + 1),
                    y=(float(cell) if math.isfinite(float(cell)) else None),
                )
                for i, cell in enumerate(row[j] for row in fit.matrix)
            ],
            source_points=len(fit.bpms),
        )
        for j, corrector in enumerate(fit.fitted_correctors)
    ]

    return Panel(
        title="Response by BPM",
        x_label="BPM index",
        y_label="Response slope",
        y_units="BPM units / A",
        annotations=[
            "Each line is one corrector's column of the matrix above. The "
            "oscillation in sign is betatron phase advance; a flat line is a "
            "corrector with no measured response.",
            "BPMs are in the order they were requested, which is not their "
            "order around the ring -- this plan carries no lattice positions.",
        ],
        series_picker=True,
        mark=LinesMark(series=series, source_points=len(fit.bpms) * len(series)),
    )


def _spectrum_panel(fit: SlicedResponseFit) -> Panel | None:
    """The singular value spectrum, or `None` when there is nothing to decompose.

    Plotted as log10 because a spectrum's whole shape is its decades of
    fall-off, and the renderer draws linear axes only -- taking the log here is
    honest about the transform in the axis label, where a linear plot of the
    same values would simply hide every mode below the first.

    Values at or below the numerical-rank tolerance become gaps rather than
    points. A rank-deficient matrix does not decompose to exact zeros: it
    leaves floating-point residue around 1e-16 of the largest value, and
    `log10` of *that* is a -16 that stretches the axis over sixteen decades and
    squashes every real mode into a sliver. The threshold is the standard
    numerical-rank one (`numpy.linalg.matrix_rank`'s): below it a value is an
    artefact of the decomposition, not a measured mode.
    """
    values = singular_values(fit.matrix)
    if values.size == 0:
        return None

    # `math.ulp(1.0)` IS the machine epsilon `sys.float_info.epsilon` reports,
    # reached through the one module the plan validator's import allowlist
    # permits here -- a plan file may not import `sys`.
    rows, columns = fit.matrix.shape
    tolerance = float(values[0]) * max(rows, columns) * math.ulp(1.0)

    return Panel(
        title="Singular values",
        x_label="Singular value index",
        y_label="log10 singular value",
        annotations=[
            "How many independent correction modes the measured machine "
            "supports: where this curve flattens is the noise floor an orbit "
            "correction would truncate at.",
        ],
        section=_SPECTRUM_SECTION,
        mark=LinesMark(
            series=[
                Series(
                    label="Singular value",
                    points=[
                        Point(
                            x=float(index + 1),
                            y=(math.log10(value) if value > tolerance else None),
                        )
                        for index, value in enumerate(float(v) for v in values)
                    ],
                    source_points=int(values.size),
                )
            ],
            source_points=int(values.size),
        ),
    )


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
            annotations=[
                *matrix_note,
                "Colour is signed: one hue each side of zero, scaled "
                "symmetrically, so an unresponsive BPM reads as background "
                "rather than as a mid-scale response.",
            ],
            mark=HeatmapMark(
                x_labels=list(fit.fitted_correctors),
                y_labels=list(fit.bpms),
                values=[[float(cell) for cell in row] for row in fit.matrix],
                value_label="Response slope",
                value_units="BPM units / A",
                source_points=int(fit.matrix.size),
            ),
        ),
        _response_by_bpm_panel(fit),
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
                categories=list(fit.bpms),
                values=[float(score) for score in row_anomaly(fit.matrix)],
                source_points=len(fit.bpms),
            ),
        ),
    ]


def _build(window: RowWindow, params: PARAMS, *, with_fit: bool) -> Figure:
    """Assemble the figure. Raises freely -- `render` is what must not."""
    truncated = not window.rows_complete

    fit = sliced_response_matrix(window.rows, params.correctors, params.bpms, params.num)

    lead_annotations: list[str] = []
    if truncated:
        lead_annotations.append(
            f"This run produced more rows than are being drawn "
            f"({len(window.rows):,} of {window.total_seen:,}), so the response matrix "
            "and anomaly scores are not shown -- they would be computed over "
            "whichever correctors happened to fit, and an anomaly score compares "
            "each corrector against its peers."
        )

    # The finding leads and the evidence follows, collapsed. But a section is a
    # demotion, and there is nothing to demote beneath when the fit was skipped:
    # with no fitted panels the sweeps ARE the figure, so they stay inline --
    # which is also what keeps the capped-run warning above the fold, on the
    # first panel the reader actually sees.
    #
    # `lead_annotations` is only ever non-empty when `truncated`, and a
    # truncated run has no fitted panels -- so the warning always lands on an
    # inline sweep panel, never inside a collapsed section.
    fitted: list[Panel] = []
    if with_fit and not truncated and fit.fitted_correctors:
        fitted.extend(_fit_panels(fit))
        spectrum = _spectrum_panel(fit)
        if spectrum is not None:
            fitted.append(spectrum)

    sweeps = _trace_panels(
        fit,
        params.num,
        lead_annotations,
        section=_SWEEPS_SECTION if fitted else None,
    )

    panels = [*fitted, *sweeps]

    # Placeholders: the figure route stamps the truth onto both on every path.
    # A render sees rows, never their provenance, so `partial=True` is the
    # honest direction for a forgotten overwrite -- extra polling, never a
    # false "settled".
    return Figure(panels=panels, partial=True, source="live")


def render(window: RowWindow, params: PARAMS) -> Figure:
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
    from the front. ``window.rows_complete`` is what says whether that holds:
    ``False`` means the rows are a prefix of the run (the bridge's live buffer
    hit its per-run cap; no figure source ever hands this plan a middle or a
    tail). There the traces still stand -- a prefix keeps each corrector's
    slice where it belongs, and each trace is a claim about individual rows --
    but the matrix and anomaly panels are suppressed, because a peer-comparison
    score computed over whichever correctors fit is a statement about the cut,
    not about the machine. An annotation on the first panel says so, quoting
    ``window.total_seen`` for how much of the run went undrawn.

    **Never raises.** A figure is a view, not a result: any internal failure
    degrades to the traces alone, and a failure in those to a figure with no
    panels, so the route always has something to serve.

    **`partial` and `source` are placeholders the route overwrites.** This
    function sees rows, not their provenance, so it returns ``partial=True,
    source="live"`` unconditionally and the figure route stamps the truth onto
    both on every path. ``True`` is the honest direction for the placeholder: a
    forgotten overwrite costs a client extra polling, never a false "settled".

    Args:
        window: The run's event `data` dicts in emission order, and whether
            they are all of them (`RowWindow`).
        params: The parameters the run was launched with.

    Returns:
        A `Figure` whose panels are as complete as the rows allow.
    """
    try:
        return _build(window, params, with_fit=True)
    except Exception:
        logger.warning("orm render failed; falling back to sweep traces only", exc_info=True)
    try:
        return _build(window, params, with_fit=False)
    except Exception:
        logger.warning("orm render failed entirely; serving an empty figure", exc_info=True)
        return Figure(panels=[], partial=True, source="live")


_RENDER_CONFORMANCE: Callable[[RowWindow, PARAMS], Figure] = render
"""Static proof that `render` matches `PlanSpec.render`'s signature.

The loader holds `PlanSpec[Any]`, so a module-level `render` is not type-checked
by merely existing (see `plan_loader._resolve_render`); this binding is what
makes mypy check it here, where the mistake would be made.
"""
