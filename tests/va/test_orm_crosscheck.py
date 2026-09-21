"""The orbit response a client measures is the orbit response the model states.

Two paths reach the same number and neither calls the other. The *measured*
path is the live one a plan takes: writes arrive at ``PhysicsBridge`` as
addresses and hardware values, readings come back out of the records
``bind()`` wired up, and the slope is fitted from those readings alone. The
*oracle* path is
:func:`~osprey.services.virtual_accelerator.lattice.response.orbit_response`,
the verify lane a facility's own exported response matrix is checked against:
it sweeps one binding on its own model and converts both ends through the
calibrations the bindings document carries.

Each is built on its own model instance, so they share a served tree and
nothing else -- no lattice, no variable catalog, no solved orbit. Agreement
therefore means the serving layer adds nothing between a client and the
physics: not a unit it converted twice, not a sign, not a value it cached past
a write. Either path alone would be internally consistent while wrong.

Two comparisons, in two tolerance regimes, and the difference between them is
the point:

* against the oracle's own estimator -- the same two-point secant over the same
  sweep -- any disagreement at all is a code-path bug, so the bound is machine
  precision;
* against the estimator a real ``orm`` plan run uses -- a degree-1 fit over a
  five-point sweep, in ``orm_analysis.build_response_matrix`` -- the two are
  genuinely different numbers on a ring whose response is not perfectly
  straight, so the bound admits that curvature and no more.

Correctors are chosen by binding *kind* and monitors by theirs; which plane
each moves is measured, never assumed. No family name or device count appears
below.
"""

from __future__ import annotations

import numpy as np
import pytest

from osprey.services.bluesky_bridge.orm_analysis import build_response_matrix
from osprey.services.virtual_accelerator.bindings import Binding, BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.lattice.calibration import to_physics
from osprey.services.virtual_accelerator.lattice.response import orbit_response
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

#: The full width of the sweep both paths drive, in the hardware unit the
#: facility states for a corrector. Wide enough to move an orbit far above the
#: solver's own repeatability, narrow enough to stay in the small-signal
#: regime the exported kick calibration is written for.
_SWEEP = 4.0

#: How many correctors the comparison drives. Three is enough for a column
#: each of the two planes and one to spare, and every one of them costs closed
#: orbit solves on both paths.
_ACTUATORS = 3

#: Points in the five-point sweep a real ``orm`` plan run emits.
_PLAN_POINTS = 5

#: Machine-precision agreement between two evaluations of one estimator. The
#: two paths compute the same quotient from the same two solves, so what is
#: left is the last bits of the arithmetic.
_IDENTICAL = 1.0e-9

#: How far the plan's five-point fit may sit from the oracle's two-point
#: secant. They are different estimators of a response that is not perfectly
#: linear in the kick, so they disagree by the ring's own curvature over the
#: sweep -- small, but many orders above the bound above. A sign error or a
#: transposed matrix blows through this by orders of magnitude.
_ESTIMATOR_SPREAD = 1.0e-3

#: A seeded readout offset, in the unit the monitor publishes.
_OFFSET = 0.5


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


def _actuators(document: BindingsDocument) -> list[Binding]:
    """The correctors driven, spread across the document rather than adjacent.

    Adjacent devices in one straight section answer almost the same column;
    spreading them means a transposed or mis-ordered matrix cannot pass by
    looking roughly right.
    """
    kicks = [binding for binding in document.bindings if binding.kind == "kick"]
    assert len(kicks) >= _ACTUATORS, "the tree binds too few correctors to compare"
    stride = len(kicks) // _ACTUATORS
    return [kicks[index * stride] for index in range(_ACTUATORS)]


def _monitors(document: BindingsDocument) -> list[Binding]:
    return [binding for binding in document.bindings if binding.kind == "monitor"]


def _plane(monitor: Binding) -> int:
    """Which half of an ``orbit_response`` entry a monitor's axis is."""
    return 0 if monitor.attribute == "x" else 1


def _oracle(model: PyATRingModel, document: BindingsDocument) -> dict[str, np.ndarray]:
    """The oracle's response column for each driven corrector.

    Keyed by the corrector's address, each column one physics-per-physics
    entry per monitor binding, in document order -- the shape the measured
    matrix below is built in, so the two can be compared row for row.
    """
    monitors = _monitors(document)
    columns = {}
    for binding in _actuators(document):
        response = orbit_response(model, binding, _SWEEP, monitors=monitors)
        columns[binding.setpoint_address] = np.array(
            [response[monitor.element][_plane(monitor)] for monitor in monitors]
        )
    return columns


def _bridge(
    channels: list[dict], document: BindingsDocument, **kwargs
) -> tuple[PhysicsBridge, dict]:
    """A bridge on its own model, reading through bound records.

    Any keyword is the model's: a fault is model state, and the bridge reads
    it back from there at the moment it serves.
    """
    bridge = PhysicsBridge(PyATRingModel(PACKAGE_PATHS.data_root, channels, **kwargs))
    records = {monitor.setpoint_address: FakeRecord() for monitor in _monitors(document)}
    bridge.bind(records)
    return bridge, records


def _read(records: dict, document: BindingsDocument) -> np.ndarray:
    """One orbit read off the records, converted to physics per monitor.

    The bridge publishes each reading in the unit its own monitor states; the
    binding's calibration is the one curve that takes that to physics, and the
    oracle used the same one on its side.
    """
    return np.array(
        [
            float(to_physics(monitor.calibration, records[monitor.setpoint_address].value))
            for monitor in _monitors(document)
        ]
    )


def _span(binding: Binding, held: float) -> float:
    """The actuator's sweep width in physics, through its own calibration."""
    return float(to_physics(binding.calibration, held + _SWEEP / 2)) - float(
        to_physics(binding.calibration, held - _SWEEP / 2)
    )


def _measured_secant(
    bridge: PhysicsBridge, records: dict, document: BindingsDocument
) -> dict[str, np.ndarray]:
    """The oracle's estimator, evaluated through the live path.

    Two arms either side of where each corrector idles, the corrector restored
    before the next is driven -- so every column is a dither about one orbit.
    """
    columns = {}
    for binding in _actuators(document):
        address = binding.setpoint_address
        held = _held(bridge, binding)
        try:
            bridge.on_setpoint(address, held + _SWEEP / 2)
            high = _read(records, document)
            bridge.on_setpoint(address, held - _SWEEP / 2)
            low = _read(records, document)
        finally:
            bridge.on_setpoint(address, held)
        columns[address] = (high - low) / _span(binding, held)
    return columns


def _held(bridge: PhysicsBridge, binding: Binding) -> float:
    """Where a corrector idles: the nominal the served tree declares for it.

    Not zero. Zero is where a machine with no orbit to correct happens to
    idle, and a facility that seeds its correctors elsewhere still has to
    produce the same response about its own working point.
    """
    return float(binding.nominal)


@pytest.fixture(scope="module")
def oracle(channels: list[dict], document: BindingsDocument) -> dict[str, np.ndarray]:
    """One oracle evaluation, on a model nothing else touches."""
    return _oracle(PyATRingModel(PACKAGE_PATHS.data_root, channels), document)


class TestTheLivePathAndTheOracleAreOnePhysics:
    def test_the_measured_secant_is_the_oracles_number(
        self, channels: list[dict], document: BindingsDocument, oracle: dict[str, np.ndarray]
    ) -> None:
        bridge, records = _bridge(channels, document)

        measured = _measured_secant(bridge, records, document)

        for address, column in measured.items():
            assert column == pytest.approx(oracle[address], rel=_IDENTICAL, abs=_IDENTICAL), address

    def test_the_response_is_not_trivially_zero(self, oracle: dict[str, np.ndarray]) -> None:
        """Guards the agreement above: two empty columns agree perfectly."""
        for address, column in oracle.items():
            assert np.max(np.abs(column)) > 0.0, address

    def test_each_corrector_answers_a_column_of_its_own(
        self, oracle: dict[str, np.ndarray]
    ) -> None:
        """Two correctors producing one column would make the comparison blind
        to a mis-ordered matrix."""
        columns = list(oracle.values())
        for index, column in enumerate(columns):
            for other in columns[index + 1 :]:
                assert not np.allclose(column, other)

    def test_a_seeded_readout_offset_leaves_the_measured_response_alone(
        self, channels: list[dict], document: BindingsDocument, oracle: dict[str, np.ndarray]
    ) -> None:
        """A constant cancels out of a two-sided difference, so a readout
        offset must not reach a response. One that did would make every
        measured matrix a function of the readout faults seeded that day."""
        monitor = _monitors(document)[0]
        bridge, records = _bridge(
            channels,
            document,
            bpm_errors={monitor.element: {"offset_x": _OFFSET, "offset_y": _OFFSET}},
        )

        measured = _measured_secant(bridge, records, document)

        for address, column in measured.items():
            assert column == pytest.approx(oracle[address], rel=_IDENTICAL, abs=_IDENTICAL), address


class TestThePlansEstimatorAgreesWithTheOracle:
    def test_the_fitted_matrix_matches_the_oracle_within_the_rings_curvature(
        self, channels: list[dict], document: BindingsDocument, oracle: dict[str, np.ndarray]
    ) -> None:
        """The shape a real ``orm`` run emits, fitted by the code that fits it.

        Every row carries every corrector's current, the sweep of each is
        centred on where that corrector idles, and the others sit at their own
        working points throughout -- the invariant
        ``build_response_matrix`` checks and fits against.
        """
        bridge, records = _bridge(channels, document)
        actuators = _actuators(document)
        monitors = _monitors(document)
        addresses = [binding.setpoint_address for binding in actuators]
        idle = {binding.setpoint_address: _held(bridge, binding) for binding in actuators}
        offsets = np.linspace(-_SWEEP / 2, _SWEEP / 2, _PLAN_POINTS)

        rows: list[dict[str, float]] = []
        for binding in actuators:
            address = binding.setpoint_address
            try:
                for offset in offsets:
                    bridge.on_setpoint(address, idle[address] + float(offset))
                    reading = _read(records, document)
                    rows.append(
                        {
                            **idle,
                            address: idle[address] + float(offset),
                            **{
                                monitor.setpoint_address: float(value)
                                for monitor, value in zip(monitors, reading, strict=True)
                            },
                        }
                    )
            finally:
                bridge.on_setpoint(address, idle[address])

        fitted = build_response_matrix(
            rows, addresses, [monitor.setpoint_address for monitor in monitors]
        )

        for column, binding in enumerate(actuators):
            address = binding.setpoint_address
            # The fit is physics-per-hardware: its readings went through the
            # monitors' curves above, its currents did not. One secant of the
            # actuator's own curve over the same sweep puts it in the oracle's
            # units -- exact for a straight calibration, and the honest local
            # linearization for a curved one.
            in_physics = fitted[:, column] * (_SWEEP / _span(binding, idle[address]))
            reference = np.max(np.abs(oracle[address]))
            assert np.max(np.abs(in_physics - oracle[address])) <= _ESTIMATOR_SPREAD * reference
