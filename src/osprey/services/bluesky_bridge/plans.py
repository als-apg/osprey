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

from bluesky import plans as bp
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
}


def list_plan_specs() -> list[dict[str, Any]]:
    """Serialize `BUILTIN_PLANS` for `GET /plans`."""
    return [spec.to_dict() for spec in BUILTIN_PLANS.values()]
