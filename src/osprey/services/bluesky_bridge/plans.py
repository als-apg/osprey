"""The v1 built-in scan plan set, wrapping bluesky's ``bp.scan``/``bp.count``/
``bp.grid_scan`` with pydantic parameter schemas that `GET /plans` serves.

Device-agnostic by construction: every schema names detectors/motors by
string, resolved against whatever device mapping the caller passes in (the
mock factory, the EPICS factory, or a facility-injected ``get_devices()`` —
see ``plan_loader.py``), never against a fixed device set. This module imports
``bluesky`` and lives behind the optional ``osprey-framework[bluesky-bridge]``
extra — keep it OUT of the lifecycle core's import path (``app.py`` only
reaches it via a guarded import inside the `/plans` route).
"""

from __future__ import annotations

from typing import Any

from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from pydantic import BaseModel, Field, model_validator

from .plan_types import PlanSpec


class ScanAxis(BaseModel):
    """One motor's sweep range, shared by ``scan`` and ``grid_scan`` schemas."""

    motor: str
    start: float
    stop: float


class CountParams(BaseModel):
    """Parameters for the built-in ``count`` plan (wraps ``bp.count``)."""

    detectors: list[str] = Field(..., description="Device names to read at each count.")
    num: int = Field(1, ge=1, description="Number of readings to take.")
    delay: float | None = Field(
        default=None, ge=0, description="Delay in seconds between successive readings."
    )


class ScanParams(BaseModel):
    """Parameters for the built-in ``scan`` plan (wraps ``bp.scan``)."""

    detectors: list[str] = Field(..., description="Device names to read at each step.")
    axes: list[ScanAxis] = Field(
        ..., min_length=1, description="Motors to move together, each over its own start/stop."
    )
    num: int = Field(..., ge=2, description="Number of evenly-spaced points along the axes.")


class GridScanParams(BaseModel):
    """Parameters for the built-in ``grid_scan`` plan (wraps ``bp.grid_scan``)."""

    detectors: list[str] = Field(..., description="Device names to read at each grid point.")
    axes: list[ScanAxis] = Field(..., min_length=1, description="One entry per grid dimension.")
    num_points: list[int] = Field(
        ..., description="Points per axis; must have the same length as `axes`."
    )
    snake_axes: bool = Field(
        default=False, description="Snake back-and-forth across successive axes."
    )

    @model_validator(mode="after")
    def _num_points_matches_axes(self) -> GridScanParams:
        """Reject a params set whose ``num_points`` and ``axes`` lengths disagree.

        Enforced in the schema (not just at plan-build time) so the mismatch
        surfaces as an ordinary parameter-validation error the moment
        ``plan_args`` are validated in ``reinitialize()``, uniform with every
        other schema violation — rather than as a bare ValueError deeper in
        plan construction.
        """
        if len(self.num_points) != len(self.axes):
            raise ValueError(
                f"num_points must have one entry per axis "
                f"(got {len(self.num_points)} num_points, {len(self.axes)} axes)"
            )
        return self


class ORMParams(BaseModel):
    """Parameters for the built-in ``orm`` plan: an orbit-response-matrix sweep.

    Sweeps each corrector in ``correctors``, one at a time, over ``num``
    evenly-spaced currents spanning ``[-span_a, +span_a]``, reading every
    detector in ``detectors`` (the BPMs) at each point.

    ``span_a`` is double-bounded: within the corrector ``channel_limits``
    band (±12 A, see ``channel_limits.json``'s ``SR:MAG:{HCM,VCM}:*:CURRENT:SP``
    entries) *and* within the response model's "typical ±10 A range" that
    ``lattice/response.py``'s ``AMPS_PER_RADIAN_KICK`` comment documents as
    giving a plausible, still-linear kick — the tighter of the two. The
    ``DRVH``/``DRVL`` clamp newly set on the corrector ``:SP`` record (FR10)
    is a second, independent bound below this schema's own, applying to any
    writer, not just this plan.
    """

    correctors: list[str] = Field(
        ..., min_length=1, description="Corrector device names to sweep, one at a time."
    )
    detectors: list[str] = Field(
        ..., min_length=1, description="BPM detector device names to read at every point."
    )
    span_a: float = Field(
        ...,
        gt=0,
        le=10.0,
        description="Half-width, in amps, of the symmetric current sweep around zero.",
    )
    num: int = Field(..., ge=3, description="Number of evenly-spaced current points per corrector.")

    @model_validator(mode="after")
    def _correctors_and_detectors_disjoint(self) -> ORMParams:
        """Reject a device named as both a driven corrector and a read BPM.

        Sweeping a device while also reading it as a response detector is a
        configuration mistake — the point being driven can't simultaneously
        measure its own response — so this is caught here, as an ordinary
        schema-level error raised uniformly at ``reinitialize()``, rather
        than surfacing as ambiguous plan behavior once the RunEngine is
        already running.
        """
        overlap = set(self.correctors) & set(self.detectors)
        if overlap:
            raise ValueError(
                f"correctors and detectors must be disjoint (overlap: {sorted(overlap)})"
            )
        return self


def _count_plan(devices: dict[str, Any], params: CountParams) -> Any:
    detectors = [devices[name] for name in params.detectors]
    return bp.count(detectors, num=params.num, delay=params.delay)


def _scan_plan(devices: dict[str, Any], params: ScanParams) -> Any:
    detectors = [devices[name] for name in params.detectors]
    args: list[Any] = []
    for axis in params.axes:
        args.extend([devices[axis.motor], axis.start, axis.stop])
    return bp.scan(detectors, *args, num=params.num)


def _grid_scan_plan(devices: dict[str, Any], params: GridScanParams) -> Any:
    # GridScanParams' model_validator already guarantees equal lengths; the
    # strict=True zip below is a backstop, not the primary check.
    detectors = [devices[name] for name in params.detectors]
    args: list[Any] = []
    for axis, npts in zip(params.axes, params.num_points, strict=True):
        args.extend([devices[axis.motor], axis.start, axis.stop, npts])
    return bp.grid_scan(detectors, *args, snake_axes=params.snake_axes)


def _orm_plan(devices: dict[str, Any], params: ORMParams) -> Any:
    """Build the non-adaptive orbit-response-matrix sweep.

    For each corrector, sweep ``num`` currents evenly spaced over
    ``[-span_a, +span_a]`` (computed up front — the full trajectory is fixed
    before the RunEngine ever starts, so ``estimate_current_completion``
    stays a binary 0/1, not an adaptive step-by-step estimate). At each
    point, move the corrector then ``trigger_and_read`` every corrector
    together with every BPM detector in one bundle, so each point emits
    exactly ONE event whose ``data`` carries the driven corrector's current
    (its readback, which echoes the commanded setpoint — see FR10) alongside
    every BPM reading. The bundle always reads every corrector, not only the
    one currently being swept: bluesky's RunEngine fixes a stream's field set
    from its first ``create()``/``save()`` call and rejects a later call that
    reads a different object set, so a bundle that shrank to "only the
    currently-driven corrector" would raise once a second corrector's sweep
    began — every other corrector simply reads back its idle (0 A) value in
    those rows. Each corrector is restored to 0 A once its own sweep
    finishes, including on abort (the ``try``/``finally`` runs on
    ``GeneratorExit`` too).
    """
    correctors = [(name, devices[name]) for name in params.correctors]
    corrector_devices = [corrector for _, corrector in correctors]
    detector_devices = [devices[name] for name in params.detectors]
    step = (2 * params.span_a) / (params.num - 1)
    # Symmetric about 0 (±span_a), and each corrector is restored to 0 A
    # between sweeps (see the `finally` below) -- `orm_analysis.py`'s
    # build_response_matrix relies on both: they put every idle-corrector
    # row at the fit's x-mean, so it carries zero leverage on the fitted
    # slope regardless of what BPM reading that row actually carries.
    currents = [-params.span_a + i * step for i in range(params.num)]

    all_devices = corrector_devices + detector_devices

    @bpp.stage_decorator(all_devices)
    @bpp.run_decorator()
    def _sweep():
        for _, corrector in correctors:
            try:
                for current in currents:
                    yield from bps.mv(corrector, current)
                    yield from bps.trigger_and_read(all_devices)
            finally:
                yield from bps.mv(corrector, 0.0)

    return _sweep()


# The v1 built-in plan registry, keyed by the name a `RunRequest.plan_name`
# resolves against.
BUILTIN_PLANS: dict[str, PlanSpec[Any]] = {
    "count": PlanSpec(
        name="count",
        plan=_count_plan,
        schema=CountParams,
        description="Read detectors N times with no motor motion (bluesky bp.count).",
    ),
    "scan": PlanSpec(
        name="scan",
        plan=_scan_plan,
        schema=ScanParams,
        description="Scan one or more motors together over evenly-spaced points (bluesky bp.scan).",
    ),
    "grid_scan": PlanSpec(
        name="grid_scan",
        plan=_grid_scan_plan,
        schema=GridScanParams,
        description="Scan a rectangular grid of motors (bluesky bp.grid_scan).",
    ),
    "orm": PlanSpec(
        name="orm",
        plan=_orm_plan,
        schema=ORMParams,
        description=(
            "Sweep each corrector over a bounded current range, reading all BPM "
            "detectors at every point, to measure an orbit-response matrix."
        ),
    ),
}


def list_plan_specs() -> list[dict[str, Any]]:
    """Serialize `BUILTIN_PLANS` for `GET /plans`."""
    return [spec.to_dict() for spec in BUILTIN_PLANS.values()]
