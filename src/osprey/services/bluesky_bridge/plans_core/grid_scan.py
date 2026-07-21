"""Shipped plan: ``grid_scan``, an n-dimensional rectangular grid scan.

Discovered via the layered directory catalog's ``shipped`` tier (a folder
scan of this package's ``plans_core/`` dir — see ``plan_loader.py``): step a
set of setpoint devices over a rectangular grid, reading a set of detectors
at every grid point, wrapping ``bluesky.plans.grid_scan``.

Device-agnostic: ``setpoints``/``detectors`` are resolved by string name
against whatever ``devices`` dict the bridge passes in; nothing here names a
facility PV or a fixed device set.
"""

from __future__ import annotations

from typing import Any

from bluesky import plans as bp
from pydantic import BaseModel, Field, model_validator

PLAN_METADATA = {
    "name": "grid_scan",
    "description": (
        "Scan a rectangular grid of setpoint devices, reading all detectors at "
        "every grid point (bluesky bp.grid_scan)."
    ),
    "category": "accelerator",
    "required_devices": ["setpoints", "detectors"],
    "writes": True,
}


class GridAxis(BaseModel):
    """One setpoint device's sweep range and point count, for one grid dimension."""

    setpoint: str
    start: float
    stop: float
    num_points: int = Field(..., ge=2, description="Points along this axis.")


class PARAMS(BaseModel):
    """Parameters for ``grid_scan``: one `GridAxis` per grid dimension.

    Steps each axis's setpoint device over its own ``[start, stop]`` range in
    ``num_points`` evenly-spaced steps, reading every device in ``detectors``
    at each combination of axis positions (a rectangular grid of
    ``prod(num_points)`` total points).
    """

    detectors: list[str] = Field(
        ..., min_length=1, description="Device names to read at each grid point."
    )
    axes: list[GridAxis] = Field(..., min_length=1, description="One entry per grid dimension.")
    snake_axes: bool = Field(
        default=False, description="Snake back-and-forth across successive axes."
    )

    @model_validator(mode="after")
    def _setpoints_and_detectors_disjoint(self) -> PARAMS:
        """Reject a device named as both a driven setpoint and a read detector."""
        setpoints = {axis.setpoint for axis in self.axes}
        overlap = setpoints & set(self.detectors)
        if overlap:
            raise ValueError(
                f"setpoints and detectors must be disjoint (overlap: {sorted(overlap)})"
            )
        return self


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    """Build the n-dimensional grid-scan generator.

    Mirrors the built-in ``grid_scan`` plan's (``plans.py``) idiom: resolves
    each axis's setpoint/detector by string name against ``devices`` and
    hands the flattened ``(device, start, stop, num_points)`` triples straight
    to ``bp.grid_scan``.
    """
    detectors = [devices[name] for name in params.detectors]
    args: list[Any] = []
    for axis in params.axes:
        args.extend([devices[axis.setpoint], axis.start, axis.stop, axis.num_points])
    return bp.grid_scan(detectors, *args, snake_axes=params.snake_axes)
