"""A solved orbit bump is a real bump on the modelled machine.

``bump_analysis`` never touches an accelerator. It is handed a two-sided probe
of each corrector, fits the local corrector-to-monitor response from those
readings, and solves that fit for the offsets producing a requested orbit
change at the monitors an operator constrained. Everything it returns is
linear algebra over numbers a plan measured -- which is exactly why it can be
internally consistent and still wrong about the machine.

So the bump is solved here through the live path, applied through the live
path, and then compared against the independent prediction
:func:`~osprey.services.virtual_accelerator.lattice.response.orbit_response`
makes for the same offsets on its own model. The measured path -- probe,
solve, apply, read -- goes through ``PhysicsBridge`` and the records
``bind()`` wired up, so the seeded-error read pipeline a real plan's readings
travel is in the loop. The two share a served tree and nothing else.

**Why this comparison is a bound and the sibling ORM one is an equality.**
There, both sides evaluate one corrector's response about one orbit, and
anything but agreement to the last bits is a code-path bug. Here the oracle is
evaluated one corrector at a time, so its prediction for a three-corrector
bump is the linear *superposition* of three single-corrector responses.
Superposition is exact only on a linear ring, and a real lattice's
nonlinearities feed down once the orbit excursion is large enough to sample
them. Everything below is therefore held inside the small-signal window the
exported kick calibration is written for, where the residual that remains is a
small fraction of the bump -- and a sign error, a mis-ordered probe pair or a
wrong-row solve still misses by orders of magnitude more than the bound.

Correctors come from the served document by binding kind; which plane each
moves is measured, and the target bump is sized from the response that
measurement found rather than stated in metres. No family name, device count
or strength constant appears below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from osprey.services.bluesky_bridge.bump_analysis import fit_probe_response, solve_offsets
from osprey.services.virtual_accelerator.bindings import Binding, BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.lattice.calibration import to_physics
from osprey.services.virtual_accelerator.lattice.response import orbit_response
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

#: The half-width of the two-sided probe each corrector is dithered by, in the
#: hardware unit the facility states for it.
_PROBE = 2.0

#: How many correctors the bump is built from. Three is the smallest set that
#: can hit a target and close on both sides of it.
_CORRECTORS = 3

#: The bump's target, as a fraction of the largest orbit change one corrector
#: makes at the probe amplitude. Sized from the machine rather than stated in
#: metres, so the solved offsets land in the same small-signal window the
#: probe did on any ring.
_TARGET_FRACTION = 0.5

#: How far the measured bump may sit from the superposition prediction, as a
#: fraction of the bump's own peak. The gap is the ring's own nonlinearity
#: over three simultaneous kicks; see the module docstring.
_SUPERPOSITION_BOUND = 5.0e-2

#: How close the bump must come to the target it was solved for, as a fraction
#: of that target. Loose enough for the same nonlinearity, tight enough that a
#: bump landing at half its demand fails.
_TARGET_BOUND = 1.0e-1

#: How much of the bump's peak may appear outside the correctors' span before
#: it stops being a local bump.
_CLOSURE_BOUND = 1.0e-1


@pytest.fixture(scope="module")
def document() -> BindingsDocument:
    return load_bindings(PACKAGE_PATHS.va_bindings)


@pytest.fixture(scope="module")
def channels() -> list[dict]:
    return build_manifest()["channels"]


class FakeRecord:
    """The one thing the bridge asks of a record: that it can be ``set``."""

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


@dataclass(frozen=True)
class _Bump:
    """Everything one bump run produced, for the assertions to pick apart.

    ``measured`` and ``predicted`` are orbit *changes* away from the reference
    orbit, in physics units. The reference is measured, never assumed to be
    zero -- a ring's undisturbed closed orbit is not.
    """

    offsets: np.ndarray
    measured: np.ndarray
    predicted: np.ndarray
    target_row: int
    target: float
    outside_span: np.ndarray

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.measured)))


def _plane_of(binding: Binding) -> str:
    return binding.attribute


def _ring_order(model: PyATRingModel, bindings: list[Binding]) -> list[Binding]:
    """Bindings in the order their elements sit around the deck.

    By element name, looked up in the ring the model holds -- never by any
    position the export stated, which does not address the saved deck.
    """
    return sorted(bindings, key=lambda binding: model.element_index(binding.element))


def _dominant_plane(response: dict[str, tuple[float, float]]) -> str:
    """Which transverse plane a corrector actually moves, measured."""
    x = max(abs(entry[0]) for entry in response.values())
    y = max(abs(entry[1]) for entry in response.values())
    return "x" if x >= y else "y"


@pytest.fixture(scope="module")
def bump(channels: list[dict], document: BindingsDocument) -> _Bump:
    """One bump run: probe, solve, apply, read, and predict.

    Module scoped because the run is a few dozen closed-orbit solves and both
    paths are deterministic. Nothing below mutates it, and the bridge is left
    with every driven corrector back at its working point.
    """
    oracle_model = PyATRingModel(PACKAGE_PATHS.data_root, channels)
    monitors_all = [binding for binding in document.bindings if binding.kind == "monitor"]
    kicks = _ring_order(
        oracle_model, [binding for binding in document.bindings if binding.kind == "kick"]
    )

    # Which plane the first corrector moves decides the whole run: the bump is
    # built, constrained and read on that one plane, which is the shape
    # ``fit_probe_response`` takes -- one value per monitor.
    first = orbit_response(oracle_model, kicks[0], 2 * _PROBE, monitors=monitors_all)
    plane = _dominant_plane(first)
    actuators_all = [
        binding
        for binding in kicks
        if _dominant_plane(orbit_response(oracle_model, binding, 2 * _PROBE, monitors=monitors_all))
        == plane
    ]
    stride = len(actuators_all) // (_CORRECTORS + 1)
    actuators = [actuators_all[(index + 1) * stride] for index in range(_CORRECTORS)]

    monitors = _ring_order(
        oracle_model, [binding for binding in monitors_all if _plane_of(binding) == plane]
    )
    positions = [oracle_model.element_index(binding.element) for binding in monitors]
    span = (
        oracle_model.element_index(actuators[0].element),
        oracle_model.element_index(actuators[-1].element),
    )
    inside = [row for row, place in enumerate(positions) if span[0] <= place <= span[1]]
    outside = np.array(
        [row for row, place in enumerate(positions) if not span[0] <= place <= span[1]]
    )
    assert inside and outside.size, "the chosen correctors span every monitor or none"
    target_row = inside[len(inside) // 2]

    bridge = PhysicsBridge(PyATRingModel(PACKAGE_PATHS.data_root, channels))
    records = {binding.setpoint_address: FakeRecord() for binding in monitors}
    bridge.bind(records)

    def orbit() -> np.ndarray:
        """One read, in physics units, one value per monitor in ring order."""
        return np.array(
            [
                float(to_physics(binding.calibration, records[binding.setpoint_address].value))
                for binding in monitors
            ]
        )

    def raw() -> dict[str, float]:
        """The same read in the unit each monitor publishes, which is what a
        plan's event documents carry and what ``fit_probe_response`` fits."""
        return {
            binding.setpoint_address: float(records[binding.setpoint_address].value)
            for binding in monitors
        }

    idle = {binding.setpoint_address: float(binding.nominal) for binding in actuators}
    reference = orbit()

    # Strictly [corrector 0 high, corrector 0 low, corrector 1 high, ...]:
    # the pair for column j is read at rows 2j and 2j+1. Each corrector is
    # restored before the next is probed, so every pair dithers one orbit.
    probe_rows: list[dict[str, float]] = []
    for binding in actuators:
        address = binding.setpoint_address
        try:
            for sign in (+1.0, -1.0):
                bridge.on_setpoint(address, idle[address] + sign * _PROBE)
                probe_rows.append(raw())
        finally:
            bridge.on_setpoint(address, idle[address])

    names = [binding.setpoint_address for binding in actuators]
    response = fit_probe_response(
        probe_rows, names, [binding.setpoint_address for binding in monitors], _PROBE
    )

    # Sized from what the probe just measured, so the demand is reachable
    # without leaving the window the probe itself stayed in.
    target = _TARGET_FRACTION * float(np.max(np.abs(response))) * _PROBE
    rows = [target_row, *outside[:: max(1, outside.size // 2)][:2]]
    desired = np.zeros(len(rows))
    desired[0] = target
    offsets = solve_offsets(response[rows, :], desired)

    try:
        for binding, offset in zip(actuators, offsets, strict=True):
            bridge.on_setpoint(
                binding.setpoint_address, idle[binding.setpoint_address] + float(offset)
            )
        measured = orbit() - reference
    finally:
        for binding in actuators:
            bridge.on_setpoint(binding.setpoint_address, idle[binding.setpoint_address])

    # The oracle's prediction: each corrector's own response column, scaled by
    # how far that corrector moved in physics, summed. Superposition -- which
    # is the approximation the bound admits.
    predicted = np.zeros(len(monitors))
    axis = 0 if plane == "x" else 1
    for binding, offset in zip(actuators, offsets, strict=True):
        held = idle[binding.setpoint_address]
        column = orbit_response(oracle_model, binding, 2 * _PROBE, monitors=monitors_all)
        moved = float(to_physics(binding.calibration, held + float(offset))) - float(
            to_physics(binding.calibration, held)
        )
        predicted += moved * np.array([column[monitor.element][axis] for monitor in monitors])

    # The monitor readings are in physics; the fitted response and the target
    # were in the monitors' published unit, so the target is converted through
    # the same curve before it is compared with a measured change.
    scale = float(
        to_physics(monitors[target_row].calibration, target)
        - to_physics(monitors[target_row].calibration, 0.0)
    )

    return _Bump(
        offsets=offsets,
        measured=measured,
        predicted=predicted,
        target_row=target_row,
        target=scale,
        outside_span=outside,
    )


class TestTheBumpIsRealOnTheModelledMachine:
    def test_the_measured_bump_matches_the_oracles_superposition(self, bump: _Bump) -> None:
        """The cross-check itself: the orbit the live path produced is the one
        the model predicts for the offsets that were applied."""
        assert bump.peak > 0.0
        assert np.max(np.abs(bump.measured - bump.predicted)) <= _SUPERPOSITION_BOUND * bump.peak

    def test_the_bump_reaches_the_target_it_was_solved_for(self, bump: _Bump) -> None:
        """A solve that agreed with the oracle while missing its own demand
        would mean the constraint rows were not the rows that were read."""
        assert bump.measured[bump.target_row] == pytest.approx(
            bump.target, rel=_TARGET_BOUND, abs=_TARGET_BOUND * abs(bump.target)
        )

    def test_the_bump_closes_outside_the_correctors_span(self, bump: _Bump) -> None:
        """What makes it a bump rather than a global orbit distortion: past
        the last corrector there is nothing left of it."""
        assert np.max(np.abs(bump.measured[bump.outside_span])) <= _CLOSURE_BOUND * bump.peak

    def test_the_solve_moved_every_corrector(self, bump: _Bump) -> None:
        """Guards the agreement above from a degenerate solve: an all-zero
        offset vector produces no bump, and no bump matches no prediction
        perfectly."""
        assert bump.offsets.shape == (_CORRECTORS,)
        assert np.all(np.abs(bump.offsets) > 0.0)
