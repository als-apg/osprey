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
from typing import Any, Literal

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from pydantic import BaseModel, Field, model_validator

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
