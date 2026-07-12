"""Cross-check: the ORM measured by driving `PhysicsBridge` against the
independent `lattice/response.py` model oracle (task 5.1).

Two physics code paths that never call each other are compared here:
`PhysicsBridge` (the live IOC-facing setpoint -> orbit -> BPM-reading path,
including the seeded-error read pipeline `bind()`/`_push_bpm_readbacks` wires
up) and `lattice.response.orbit_response` (an offline, from-scratch model of
the same corrector-kick -> closed-orbit response, built and cached
independently in `response.py`'s own module-global ring). Building a
response-slope matrix from each and comparing them proves the live bridge's
physics matches the model, not merely that each is internally consistent.

Corrector and BPM device ids are derived from the manifest's pyat-coupled
inventory (`lattice.inventory`), never hardcoded preset channel names, per
the epic's VA-safety-test convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from osprey.services.bluesky_bridge.orm_analysis import build_response_matrix
from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.lattice import inventory, orbit_response

# Symmetric sweep parameters, matching the real `orm` plan's contract
# (`plans.py::_orm_plan`): a zero-mean sweep is required by
# `build_response_matrix`'s degenerate-fit guard.
SPAN_A = 5.0
NUM_POINTS = 5

_AXES = ("X", "Y")


class _FakeRecord:
    """Minimal duck-typed stand-in for a softioc In record: just `.set()`.

    Mirrors `test_physics_bridge.py`'s `FakeRecord` -- reading through
    `bind()`'s bound records (rather than `bpm_positions()`, the physics-only
    truth) exercises the same seeded-error read pipeline
    (`_push_bpm_readbacks`/`errors.bpm_read`) a real `orm` plan reads BPM RB
    channels through.
    """

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


def _corrector_and_bpm_names() -> tuple[list[str], list[str]]:
    """Generic corrector/BPM device names from the manifest inventory.

    A handful of correctors (not the full HCM+VCM inventory) keeps the sweep
    fast while still exercising both planes; every BPM is read at every
    point, matching the real plan's "read all BPMs" contract.
    """
    inv = inventory.pyat_coupled_device_ids()
    correctors = [f"HCM{d}" for d in inv["HCM"][:3]] + [f"VCM{d}" for d in inv["VCM"][:3]]
    bpms = [f"BPM{d}" for d in inv["BPM"]]
    return correctors, bpms


def _sweep_currents(span: float, num: int) -> list[float]:
    """The same symmetric, zero-mean current sequence `_orm_plan` sweeps."""
    step = (2 * span) / (num - 1)
    return [-span + i * step for i in range(num)]


def _sp_address(corrector: str) -> str:
    family, device = corrector[:3], corrector[3:]
    return f"SR:MAG:{family}:{device}:CURRENT:SP"


def _bpm_axis_address(bpm: str, axis: str) -> str:
    device = bpm[len("BPM") :]
    return f"SR:DIAG:BPM:{device}:POSITION:{axis}"


def _detector_key(bpm: str, axis: str) -> str:
    return f"{bpm}:{axis}"


def _bind_bpm_records(bridge: PhysicsBridge, bpms: list[str]) -> dict[tuple[str, str], _FakeRecord]:
    records: dict[str, _FakeRecord] = {}
    by_axis: dict[tuple[str, str], _FakeRecord] = {}
    for bpm in bpms:
        for axis in _AXES:
            rec = _FakeRecord()
            records[_bpm_axis_address(bpm, axis)] = rec
            by_axis[(bpm, axis)] = rec
    bridge.bind(records)
    return by_axis


def _measure_rows(
    bridge: PhysicsBridge, correctors: list[str], bpms: list[str], span: float, num: int
) -> list[dict[str, float]]:
    """Drive `bridge` over a symmetric sweep of each corrector in turn.

    Builds rows in the same shape `_orm_plan` emits: every row carries every
    corrector's current (the just-swept one at its commanded value, every
    other one at its idle 0.0 -- FR10's SP->RB echo means a real bridge's
    corrector readback equals what we set here exactly) alongside every BPM
    axis reading, read through `bind()`'s bound records.
    """
    by_axis = _bind_bpm_records(bridge, bpms)
    currents = _sweep_currents(span, num)
    rows: list[dict[str, float]] = []

    for corrector in correctors:
        try:
            for current in currents:
                bridge.on_setpoint(_sp_address(corrector), current)
                row = {c: (current if c == corrector else 0.0) for c in correctors}
                for bpm in bpms:
                    for axis in _AXES:
                        row[_detector_key(bpm, axis)] = by_axis[(bpm, axis)].value
                rows.append(row)
        finally:
            bridge.on_setpoint(_sp_address(corrector), 0.0)

    return rows


def _model_matrix(correctors: list[str], bpms: list[str], span: float) -> np.ndarray:
    """The independent model-oracle response matrix from `lattice/response.py`.

    A symmetric two-point finite difference at +-span mirrors exactly what a
    degree-1 polyfit recovers over a symmetric sweep of an exactly-linear
    system (AT's `find_orbit4` closed-orbit solve is linear in kick angle at
    fixed optics) -- the same slope `build_response_matrix` fits.
    """
    n_axes = len(_AXES)
    detectors = [_detector_key(bpm, axis) for bpm in bpms for axis in _AXES]
    matrix = np.zeros((len(detectors), len(correctors)))
    for j, corrector in enumerate(correctors):
        plus = orbit_response(corrector, span)
        minus = orbit_response(corrector, -span)
        for i, bpm in enumerate(bpms):
            for a in range(n_axes):
                matrix[i * n_axes + a, j] = (plus[bpm][a] - minus[bpm][a]) / (2 * span)
    return matrix


def test_measured_orm_matches_model_oracle_to_1e9_relative():
    """SC: measured ORM == lattice/response.py model oracle to <=1e-9 relative."""
    correctors, bpms = _corrector_and_bpm_names()
    detectors = [_detector_key(bpm, axis) for bpm in bpms for axis in _AXES]

    bridge = PhysicsBridge()
    rows = _measure_rows(bridge, correctors, bpms, SPAN_A, NUM_POINTS)
    measured = build_response_matrix(rows, correctors, detectors)
    model = _model_matrix(correctors, bpms, SPAN_A)

    nonzero = np.abs(model) > 1e-15
    assert nonzero.any(), "model oracle produced an all-zero matrix -- test setup is broken"

    rel_err = np.abs(measured[nonzero] - model[nonzero]) / np.abs(model[nonzero])
    max_rel_err = float(rel_err.max())
    assert max_rel_err <= 1e-9, f"measured vs model relative error {max_rel_err:.3e} exceeds 1e-9"

    # Structurally-zero entries (a corrector's own out-of-plane response,
    # e.g. an HCM's effect on a BPM's Y reading) must measure as exactly zero
    # too, not merely "small relative to a zero reference".
    if (~nonzero).any():
        assert np.max(np.abs(measured[~nonzero])) < 1e-12


def test_seeded_bpm_offset_leaves_measured_orm_unchanged():
    """SC: a seeded BPM electrical offset leaves the measured ORM unchanged
    within the noise floor -- the ORM's structural blind spot (E5)."""
    correctors, bpms = _corrector_and_bpm_names()
    detectors = [_detector_key(bpm, axis) for bpm in bpms for axis in _AXES]
    offset_bpm = bpms[0]

    clean_bridge = PhysicsBridge()
    clean_rows = _measure_rows(clean_bridge, correctors, bpms, SPAN_A, NUM_POINTS)
    clean_matrix = build_response_matrix(clean_rows, correctors, detectors)

    offset_bridge = PhysicsBridge(bpm_errors={offset_bpm: {"offset_x": 50e-6, "offset_y": 30e-6}})
    offset_rows = _measure_rows(offset_bridge, correctors, bpms, SPAN_A, NUM_POINTS)
    offset_matrix = build_response_matrix(offset_rows, correctors, detectors)

    # Sanity: the offset actually perturbed the offset BPM's *readings* --
    # otherwise this test would vacuously pass by comparing two identical
    # matrices for the wrong reason.
    clean_first_reading = clean_rows[0][_detector_key(offset_bpm, "X")]
    offset_first_reading = offset_rows[0][_detector_key(offset_bpm, "X")]
    assert offset_first_reading == pytest.approx(clean_first_reading - 50e-6, abs=1e-12)

    delta = float(np.max(np.abs(offset_matrix - clean_matrix)))
    assert delta < 1e-9, f"BPM-offset-seeded ORM diverged from clean ORM by {delta:.3e}"
