"""File-based exemplar plan: ``response_matrix``, an orbit-response-matrix sweep.

Representative reference plan; not a physics-validated procedure. It exists to
show authors the layered directory catalog's file contract (``PLAN_METADATA``
+ ``PARAMS`` + ``build_plan``) end to end, using the same physics the built-in
``orm`` plan (``plans.py``) covers programmatically — sweep each corrector,
one at a time, over a bounded current range, reading every BPM detector at
each point.

Registered as ``response_matrix`` (not ``orm``) so it never shadows the
built-in of the same physics at the ``GET /plans`` merge — see the layered
loader's trust-collision rules in ``plan_loader.py``.

Device-agnostic: ``correctors``/``detectors`` are resolved by string name
against whatever ``devices`` dict the bridge passes in; nothing here names a
facility PV or a fixed device set.
"""

from __future__ import annotations

import logging
from typing import Any

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("osprey.services.bluesky_bridge.plans_core.response_matrix")

PLAN_METADATA = {
    "name": "response_matrix",
    "description": (
        "Sweep each corrector over a bounded current range, reading all BPM "
        "detectors at every point, to measure an orbit-response matrix."
    ),
    "category": "accelerator",
    "required_devices": ["correctors", "detectors"],
    "writes": True,
}


class PARAMS(BaseModel):
    """Parameters for ``response_matrix``: correctors to sweep, BPMs to read.

    Sweeps each corrector in ``correctors``, one at a time, over ``num``
    evenly-spaced currents spanning ``[-span_a, +span_a]``, reading every
    detector in ``detectors`` at each point.
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
    corrector, sweep ``num`` currents evenly spaced over ``[-span_a,
    +span_a]``, moving the corrector then ``trigger_and_read``-ing every
    corrector together with every BPM detector in one bundle, so each point
    emits exactly one event carrying the driven corrector's current alongside
    every BPM reading. Each corrector is restored to 0 A once its own sweep
    finishes, including on abort (the ``try``/``finally`` runs on
    ``GeneratorExit`` too) — a restore failure is caught and logged rather
    than allowed to replace an in-flight sweep exception.
    """
    correctors = [(name, devices[name]) for name in params.correctors]
    corrector_devices = [corrector for _, corrector in correctors]
    detector_devices = [devices[name] for name in params.detectors]
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
                        "response_matrix plan: failed to restore corrector %s to 0 A "
                        "during cleanup; preserving the original error",
                        name,
                        exc_info=True,
                    )

    return _sweep()
