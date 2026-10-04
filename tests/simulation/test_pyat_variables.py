"""The pyat engine's variable classes, each built from one wiring record.

These run against a small purpose-built lattice: what is under test is the
conversion -- hardware units onto element attributes and back -- and a FODO
lattice carrying one element per wiring shape exercises every branch of it in
milliseconds. Every element name, calibration and slice layout comes from the
wiring records below.

The rigidity is written out longhand in :func:`brho` rather than imported, so a
failure cannot be explained away by the module under test agreeing with
itself.
"""

from __future__ import annotations

import inspect
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from lume.exceptions import ReadOnlyError  # noqa: E402
from lume_pyat.actions import PyATReadOnlyScalarVariable  # noqa: E402
from lume_pyat.exceptions import OrbitSolveError  # noqa: E402
from lume_pyat.model import LUMEPyATModel  # noqa: E402
from lume_pyat.simulator import PyATSimulator  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from osprey.simulation.engines import calibration as cal  # noqa: E402
from osprey.simulation.engines import pyat_variables  # noqa: E402
from osprey.simulation.engines.pyat_variables import (  # noqa: E402
    EV_PER_GEV,
    CalibratedSetpoint,
    EnergyVariable,
    KickVariable,
    MonitorVariable,
    RFVariable,
    StrengthVariable,
    variable_from_wiring,
)

N_CELLS = 8
QUAD_K = 1.0

#: The energy the synthetic lattice is built at, and the energy every
#: calibration below is stated at.
DECK_ENERGY_GEV = 3.0

QUADRUPOLE = "QUADRUPOLE"
SEXTUPOLE_SLICES = ("SEXTUPOLE_A", "SEXTUPOLE_B")
CORRECTOR_SLICES = ("CORRECTOR_A", "CORRECTOR_B")
BEND = "BEND"
MONITOR = "MONITOR"
CAVITIES = ("CAVITY_A", "CAVITY_B")

DEVICES = frozenset({QUADRUPOLE, BEND, MONITOR, *SEXTUPOLE_SLICES, *CORRECTOR_SLICES, *CAVITIES})

QUAD_GAIN = 0.002
SEXT_GAIN, SEXT_OFFSET = 0.5, 0.25
KICK_GAIN = 1.0e-5
RF_GAIN = 1.0e6

#: The bend's setpoint-to-energy curve, and the nominal setpoint at which the
#: lattice is at its deck energy. Only the curve's ratio is trusted.
ENERGY_GRID = [0.0, 400.0, 800.0]
ENERGY_VALUES = [0.0, 1.6, 3.2]
BEND_NOMINAL = 800.0
BEND_RAISED = 840.0
RAISED_ENERGY_GEV = 3.15

QUAD_CURRENT = 250.0
SEXT_CURRENT = 40.0
KICK_CURRENT = 3.5
RF_SETPOINT = 499.68

#: The committed demo's facility tree.
DEMO = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/facility"
)


def brho(energy_gev: float) -> float:
    """The beam rigidity at ``energy_gev``, in tesla-metres, written longhand."""
    rest_mass = 0.51099906e-3
    return (10.0 / 2.99792458) * math.sqrt((energy_gev + rest_mass) ** 2 - rest_mass**2)


def rigidity_factor(energy_gev: float) -> float:
    return brho(DECK_ENERGY_GEV) / brho(energy_gev)


def _named(cell: int, filler: str, device: str) -> str:
    return device if cell == 1 else f"{filler}_{cell:02d}"


def build_test_lattice() -> Any:
    """An eight-cell FODO lattice carrying one element per wiring shape."""
    elements: list[Any] = []
    for cell in range(1, N_CELLS + 1):
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(_named(cell, "QUAD_F", QUADRUPOLE), 0.3, QUAD_K),
            at.Drift("DRIFT", 0.4),
            at.Sextupole(_named(cell, "SEXT_F", SEXTUPOLE_SLICES[0]), 0.15, 1.0),
            at.Sextupole(_named(cell, "SEXT_F", SEXTUPOLE_SLICES[1]), 0.15, 1.0),
            at.Corrector(_named(cell, "COR", CORRECTOR_SLICES[0]), 0.0, [0.0, 0.0]),
            at.Corrector(_named(cell, "COR", CORRECTOR_SLICES[1]), 0.0, [0.0, 0.0]),
            at.Dipole(_named(cell, "DIPOLE", BEND), 1.0, 2 * np.pi / N_CELLS),
            at.Monitor(_named(cell, "BPM", MONITOR)),
            at.Drift("DRIFT", 0.4),
            at.Sextupole(_named(cell, "SEXT_D", "SEXT_D"), 0.15, -1.0),
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(_named(cell, "QUAD_D", "QUAD_D"), 0.3, -QUAD_K),
        ]
    elements += [
        at.RFCavity(name, 0.0, 1.0e6, 499.68e6, 328, DECK_ENERGY_GEV * EV_PER_GEV)
        for name in CAVITIES
    ]
    lattice = at.Lattice(elements, name="TEST", energy=DECK_ENERGY_GEV * EV_PER_GEV, periodicity=1)
    lattice.disable_6d()
    return lattice


@pytest.fixture
def lattice() -> Any:
    return build_test_lattice()


@pytest.fixture
def simulator(lattice: Any) -> PyATSimulator:
    return PyATSimulator(lattice)


def _linear(gain: float, offset: float = 0.0) -> dict[str, Any]:
    return {"linear": {"gain": gain, "offset": offset}}


def _table(grid: list[float], values: list[float]) -> dict[str, Any]:
    return {"table": {"grid": grid, "values": values}}


def _slices(names: tuple[str, ...], weight: float | None = None) -> list[dict[str, Any]]:
    return [{"element": name, **({} if weight is None else {"weight": weight})} for name in names]


def _record(address: str, **slots: Any) -> dict[str, Any]:
    return {"id": f"SR/{address}", "address": address, **slots}


def quadrupole_record(**slots: Any) -> dict[str, Any]:
    """A single-element, rigidity-scaled magnet setpoint."""
    return _record(
        "MAG:QUADRUPOLE:SP",
        **{
            "element": QUADRUPOLE,
            "engine": {"attribute": "PolynomB", "index": 1},
            "calibration": {"curve": _linear(QUAD_GAIN), "energy_scaling": "brho"},
            "default": 0.0,
            "unit": "A",
            **slots,
        },
    )


def sextupole_record(**slots: Any) -> dict[str, Any]:
    """A split magnet setpoint the control system does not scale."""
    return _record(
        "MAG:SEXTUPOLE:SP",
        **{
            "slices": _slices(SEXTUPOLE_SLICES),
            "engine": {"attribute": "PolynomB", "index": 2},
            "calibration": {"curve": _linear(SEXT_GAIN, SEXT_OFFSET), "energy_scaling": "none"},
            "default": 0.0,
            **slots,
        },
    )


def corrector_record(**slots: Any) -> dict[str, Any]:
    """A split corrector setpoint: half the kick on each piece."""
    return _record(
        "MAG:CORRECTOR:SP",
        **{
            "slices": _slices(CORRECTOR_SLICES, 0.5),
            "engine": {"attribute": "KickAngle", "index": 0},
            "calibration": {"curve": _linear(KICK_GAIN), "energy_scaling": "brho"},
            "default": 0.0,
            **slots,
        },
    )


def cavity_record(**slots: Any) -> dict[str, Any]:
    """The frequency over both cavities, set in MHz and held in Hz."""
    return _record(
        "RF:FREQUENCY:SP",
        **{
            "slices": _slices(CAVITIES),
            "engine": {"attribute": "Frequency"},
            "calibration": {"curve": _linear(RF_GAIN), "energy_scaling": "none"},
            "default": RF_SETPOINT,
            **slots,
        },
    )


def monitor_record(**slots: Any) -> dict[str, Any]:
    """One reading of the lattice's addressable monitor, published in mm."""
    return _record(
        "DIAG:MONITOR:X",
        **{
            "element": MONITOR,
            "engine": {"axis": "x"},
            "calibration": {"curve": _linear(1.0e-3)},
            **slots,
        },
    )


def energy_record(**slots: Any) -> dict[str, Any]:
    """The bend's setpoint, driving the deck energy through its curve."""
    return _record(
        "MAG:BEND:SP",
        **{
            "engine": {"attribute": "energy"},
            "calibration": {"curve": _table(ENERGY_GRID, ENERGY_VALUES)},
            "default": BEND_NOMINAL,
            **slots,
        },
    )


def build(record: dict[str, Any], deck_energy_gev: float = DECK_ENERGY_GEV, **scalar: Any) -> Any:
    variable = variable_from_wiring(record, deck_energy_gev=deck_energy_gev, **scalar)
    assert variable is not None
    return variable


def element(lattice: Any, name: str) -> Any:
    return next(candidate for candidate in lattice if candidate.FamName == name)


def lattice_state(lattice: Any) -> dict[str, float]:
    """Every number a write here can reach, flattened for comparison."""
    state = {"lattice.energy": float(lattice.energy)}
    for candidate in lattice:
        name = candidate.FamName
        if name not in DEVICES:
            continue
        for attribute in ("PolynomA", "PolynomB", "KickAngle"):
            held = getattr(candidate, attribute, None)
            if held is not None:
                state.update({f"{name}.{attribute}[{i}]": float(v) for i, v in enumerate(held)})
        for attribute in ("Frequency", "Energy"):
            held = getattr(candidate, attribute, None)
            if held is not None:
                state[f"{name}.{attribute}"] = float(held)
    return state


def build_model(lattice: Any) -> tuple[LUMEPyATModel, dict[str, str]]:
    """A model over ``lattice`` carrying one variable of every kind."""
    writables = [
        build(quadrupole_record()),
        build(sextupole_record()),
        build(corrector_record()),
        build(cavity_record()),
    ]
    knob = build(energy_record())
    knob.couple(writables)
    model = LUMEPyATModel(
        simulator=PyATSimulator(lattice),
        action_variables=[*writables, knob, build(monitor_record())],
    )
    names = {
        "quadrupole": writables[0].name,
        "energy": knob.name,
    }
    return model, names


class TestCopiedNotShared:
    def test_the_engine_classes_are_its_own(self) -> None:
        from osprey.services.virtual_accelerator.model import variables

        assert variables.CalibratedSetpoint is not pyat_variables.CalibratedSetpoint

    def test_the_module_imports_nothing_of_the_old_service(self) -> None:
        code = (
            "import sys\n"
            "import osprey.simulation.engines.pyat_variables\n"
            "loaded = [m for m in sys.modules if m.startswith("
            "'osprey.services.virtual_accelerator')]\n"
            "assert not loaded, loaded\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr

    def test_after_write_names_its_first_parameter_lattice(self) -> None:
        parameters = list(inspect.signature(EnergyVariable._after_write).parameters)
        assert parameters[1] == "lattice"


def test_the_energy_knob_s_attribute_is_a_deck_property_of_the_engine() -> None:
    from osprey.simulation.engines import pyat

    assert pyat_variables.ENERGY_ATTRIBUTE in pyat.DECK_PROPERTIES


class TestCalibrationModule:
    def test_the_inverse_wins_over_the_curve(self) -> None:
        curve, inverse = cal.Linear(2.0, 1.0), cal.Linear(10.0, 0.0)
        assert cal.to_hardware(curve, inverse, 3.0) == 30.0

    def test_a_linear_curve_inverts_algebraically(self) -> None:
        assert cal.to_hardware(cal.Linear(2.0, 1.0), None, 7.0) == 3.0

    def test_no_calibration_is_the_identity(self) -> None:
        assert cal.to_hardware(None, None, 7.0) == 7.0
        assert cal.to_physics(None, 7.0) == 7.0

    def test_a_table_without_an_inverse_has_no_way_back(self) -> None:
        with pytest.raises(cal.NoInverse) as caught:
            cal.to_hardware(cal.Table((0.0, 1.0), (0.0, 2.0)), None, 1.0)
        assert caught.value.reason == "table"

    def test_a_zero_gain_has_no_way_back(self) -> None:
        with pytest.raises(cal.NoInverse) as caught:
            cal.to_hardware(cal.Linear(0.0, 1.0), None, 1.0)
        assert caught.value.reason == "zero-gain"

    def test_the_rigidity_is_the_massive_form(self) -> None:
        assert cal.brho(3.0) == pytest.approx(brho(3.0), rel=1e-15)
        assert cal.energy_factor(RAISED_ENERGY_GEV, DECK_ENERGY_GEV) == pytest.approx(
            rigidity_factor(RAISED_ENERGY_GEV), rel=1e-15
        )


class TestFromWiring:
    @pytest.mark.parametrize(
        ("record", "kind"),
        [
            (quadrupole_record(), StrengthVariable),
            (sextupole_record(), StrengthVariable),
            (corrector_record(), KickVariable),
            (cavity_record(), RFVariable),
            (monitor_record(), MonitorVariable),
            (energy_record(), EnergyVariable),
        ],
    )
    def test_the_engine_block_picks_the_class(self, record: dict[str, Any], kind: type) -> None:
        assert type(build(record)) is kind

    @pytest.mark.parametrize("attribute", ["tune", "chromaticity"])
    def test_an_optics_output_is_not_an_element_variable(self, attribute: str) -> None:
        record = _record("DIAG:TUNE:X", engine={"attribute": attribute, "axis": "x"})
        assert variable_from_wiring(record, deck_energy_gev=DECK_ENERGY_GEV) is None

    def test_the_record_states_name_unit_and_default(self) -> None:
        variable = build(quadrupole_record(default=12.0))
        assert (variable.name, variable.unit, variable.default_value) == (
            "MAG:QUADRUPOLE:SP",
            "A",
            12.0,
        )

    def test_a_caller_field_wins_over_the_record(self) -> None:
        assert build(quadrupole_record(), default_value=5.0).default_value == 5.0

    def test_slices_become_weighted_bindings(self) -> None:
        bindings = build(corrector_record()).bindings
        assert [(b.element_name, b.attribute, b.index, b.weight) for b in bindings] == [
            (CORRECTOR_SLICES[0], "KickAngle", 0, 0.5),
            (CORRECTOR_SLICES[1], "KickAngle", 0, 0.5),
        ]

    def test_a_setpoint_with_a_table_and_no_inverse_is_refused(self) -> None:
        record = quadrupole_record(calibration={"curve": _table([0.0, 1.0], [0.0, 2.0])})
        with pytest.raises(ValidationError, match="no inverse"):
            build(record)

    def test_an_unknown_energy_scaling_is_refused(self) -> None:
        record = quadrupole_record(
            calibration={"curve": _linear(QUAD_GAIN), "energy_scaling": "gamma"}
        )
        with pytest.raises(ValueError, match="energy_scaling"):
            build(record)

    def test_the_energy_knob_takes_its_nominal_from_the_default(self) -> None:
        assert build(energy_record()).nominal == BEND_NOMINAL


class TestStrengthVariable:
    def test_writes_the_value_the_calibration_converts_it_to(self, simulator) -> None:
        build(sextupole_record())._set(simulator, SEXT_CURRENT)

        held = element(simulator.lattice, SEXTUPOLE_SLICES[0]).PolynomB[2]
        assert held == pytest.approx(SEXT_GAIN * SEXT_CURRENT + SEXT_OFFSET)

    def test_every_slice_of_a_split_magnet_carries_the_whole_strength(self, simulator) -> None:
        build(sextupole_record())._set(simulator, SEXT_CURRENT)

        held = [element(simulator.lattice, name).PolynomB[2] for name in SEXTUPOLE_SLICES]
        assert held[0] == pytest.approx(held[1])
        assert held[0] != 0.0

    def test_a_slice_takes_the_share_its_weight_states(self, simulator) -> None:
        build(sextupole_record(slices=_slices(SEXTUPOLE_SLICES, 0.5)))._set(simulator, SEXT_CURRENT)

        whole = SEXT_GAIN * SEXT_CURRENT + SEXT_OFFSET
        held = [element(simulator.lattice, name).PolynomB[2] for name in SEXTUPOLE_SLICES]
        assert held == pytest.approx([0.5 * whole, 0.5 * whole])

    def test_reads_back_through_the_algebraic_inverse_of_a_linear_curve(self, simulator) -> None:
        variable = build(sextupole_record())
        variable._set(simulator, SEXT_CURRENT)

        assert variable._get(simulator) == pytest.approx(SEXT_CURRENT, rel=1e-12)

    def test_reads_back_through_the_stated_inverse_not_the_curve(self, simulator) -> None:
        """The two directions are independent data: the stated inverse wins."""
        calibration = {
            "curve": _linear(QUAD_GAIN),
            "inverse": _linear(510.0),
            "energy_scaling": "brho",
        }
        variable = build(quadrupole_record(calibration=calibration))
        variable._set(simulator, QUAD_CURRENT)

        assert variable._get(simulator) == pytest.approx(1.02 * QUAD_CURRENT, rel=1e-9)

    def test_a_table_reads_back_through_its_inverse(self, simulator) -> None:
        calibration = {
            "curve": _table([0.0, 100.0, 300.0], [0.0, 0.2, 0.8]),
            "inverse": _table([0.0, 0.2, 0.8], [0.0, 100.0, 300.0]),
        }
        variable = build(quadrupole_record(calibration=calibration))
        variable._set(simulator, QUAD_CURRENT)

        held = element(simulator.lattice, QUADRUPOLE).PolynomB[1]
        assert held == pytest.approx(0.2 + 0.6 * (QUAD_CURRENT - 100.0) / 200.0)
        assert variable._get(simulator) == pytest.approx(QUAD_CURRENT, rel=1e-12)

    def test_no_calibration_writes_the_hardware_value(self, simulator) -> None:
        variable = build(quadrupole_record(calibration=None))
        variable._set(simulator, 0.7)

        assert element(simulator.lattice, QUADRUPOLE).PolynomB[1] == 0.7
        assert variable._get(simulator) == 0.7


class TestReadback:
    def test_is_the_written_value_mapped_back_through_the_inverse(self) -> None:
        calibration = {"curve": _linear(QUAD_GAIN), "inverse": _linear(510.0)}
        variable = build(quadrupole_record(calibration=calibration))

        assert variable.readback(QUAD_CURRENT) == pytest.approx(1.02 * QUAD_CURRENT, rel=1e-9)

    def test_is_the_written_value_where_no_inverse_is_stated(self) -> None:
        assert build(sextupole_record()).readback(SEXT_CURRENT) == SEXT_CURRENT


class TestRigidityScaling:
    def test_a_scaled_write_takes_the_factor_for_the_present_energy(self, simulator) -> None:
        simulator.lattice.energy = RAISED_ENERGY_GEV * EV_PER_GEV

        build(quadrupole_record())._set(simulator, QUAD_CURRENT)

        expected = QUAD_GAIN * QUAD_CURRENT * rigidity_factor(RAISED_ENERGY_GEV)
        assert element(simulator.lattice, QUADRUPOLE).PolynomB[1] == pytest.approx(expected)

    def test_an_unscaled_write_ignores_the_energy(self, simulator) -> None:
        simulator.lattice.energy = RAISED_ENERGY_GEV * EV_PER_GEV

        build(sextupole_record())._set(simulator, SEXT_CURRENT)

        expected = SEXT_GAIN * SEXT_CURRENT + SEXT_OFFSET
        held = element(simulator.lattice, SEXTUPOLE_SLICES[0]).PolynomB[2]
        assert held == pytest.approx(expected)

    def test_the_factor_is_removed_again_on_the_way_back(self, simulator) -> None:
        variable = build(quadrupole_record())
        simulator.lattice.energy = RAISED_ENERGY_GEV * EV_PER_GEV
        variable._set(simulator, QUAD_CURRENT)

        assert variable._get(simulator) == pytest.approx(QUAD_CURRENT, rel=1e-9)


class TestKickVariable:
    def test_divides_the_kick_over_its_slices(self, simulator) -> None:
        build(corrector_record())._set(simulator, KICK_CURRENT)

        held = [element(simulator.lattice, name).KickAngle[0] for name in CORRECTOR_SLICES]
        whole = KICK_GAIN * KICK_CURRENT
        assert held == pytest.approx([whole / 2.0, whole / 2.0])

    def test_reads_back_the_whole_kick(self, simulator) -> None:
        variable = build(corrector_record())
        variable._set(simulator, KICK_CURRENT)

        assert variable._get(simulator) == pytest.approx(KICK_CURRENT, rel=1e-9)


class TestRFVariable:
    def test_writes_the_frequency_to_every_cavity(self, simulator) -> None:
        build(cavity_record())._set(simulator, RF_SETPOINT)

        held = [element(simulator.lattice, name).Frequency for name in CAVITIES]
        assert held == pytest.approx([RF_SETPOINT * RF_GAIN] * len(CAVITIES))

    def test_reads_back_in_the_unit_it_is_set_in(self, simulator) -> None:
        variable = build(cavity_record())
        variable._set(simulator, RF_SETPOINT)

        assert variable._get(simulator) == pytest.approx(RF_SETPOINT, rel=1e-12)


def _metres(simulator: PyATSimulator) -> float:
    return PyATReadOnlyScalarVariable(
        name="raw", element_name=MONITOR, axis="x", read_only=True
    )._get(simulator)


class TestMonitorVariable:
    def test_serves_the_reading_through_the_algebraic_inverse(self, simulator) -> None:
        """Metres in, millimetres out: the curve's gain is 1e-3 mm to m."""
        build(corrector_record())._set(simulator, KICK_CURRENT)
        simulator.solve()
        metres = _metres(simulator)

        assert metres != 0.0
        assert build(monitor_record())._get(simulator) == pytest.approx(1.0e3 * metres)

    def test_serves_the_reading_through_a_stated_inverse(self, simulator) -> None:
        calibration = {
            "curve": _table([-1.0, 1.0], [-1.0e-3, 1.0e-3]),
            "inverse": _table([-1.0e-3, 1.0e-3], [-2.0, 2.0]),
        }
        build(corrector_record())._set(simulator, KICK_CURRENT)
        simulator.solve()
        metres = _metres(simulator)

        served = build(monitor_record(calibration=calibration))._get(simulator)
        assert served == pytest.approx(2.0e3 * metres)

    def test_a_unit_gain_serves_the_reading_unchanged(self, simulator) -> None:
        """The demo's BPM shape: linear gain 1.0, no inverse."""
        build(corrector_record())._set(simulator, KICK_CURRENT)
        simulator.solve()
        record = monitor_record(calibration={"curve": _linear(1.0), "energy_scaling": "none"})

        assert build(record)._get(simulator) == _metres(simulator)

    def test_a_table_without_an_inverse_is_refused(self) -> None:
        record = monitor_record(calibration={"curve": _table([-1.0, 1.0], [-1e-3, 1e-3])})
        with pytest.raises(ValidationError, match="no inverse"):
            build(record)

    def test_reads_the_axis_its_record_names(self, simulator) -> None:
        build(corrector_record())._set(simulator, KICK_CURRENT)
        simulator.solve()

        vertical = monitor_record(engine={"axis": "y"})
        assert build(monitor_record())._get(simulator) != 0.0
        assert build(vertical)._get(simulator) == 0.0

    def test_is_read_only(self, simulator) -> None:
        with pytest.raises(ReadOnlyError):
            build(monitor_record())._set(simulator, 1.0)


class TestEnergyVariable:
    def test_the_nominal_setpoint_leaves_the_deck_energy(self, simulator) -> None:
        build(energy_record())._set(simulator, BEND_NOMINAL)

        assert simulator.lattice.energy == pytest.approx(DECK_ENERGY_GEV * EV_PER_GEV, rel=1e-15)

    def test_the_energy_follows_the_curve_ratio(self, simulator) -> None:
        build(energy_record())._set(simulator, BEND_RAISED)

        assert simulator.lattice.energy == pytest.approx(RAISED_ENERGY_GEV * EV_PER_GEV)

    def test_the_cavities_move_with_the_lattice(self, simulator) -> None:
        build(energy_record())._set(simulator, BEND_RAISED)

        held = [element(simulator.lattice, name).Energy for name in CAVITIES]
        assert held == pytest.approx([RAISED_ENERGY_GEV * EV_PER_GEV] * len(CAVITIES))

    def test_rescales_every_rigidity_scaled_setpoint_it_adopted(self, simulator) -> None:
        magnet = build(quadrupole_record())
        magnet._set(simulator, QUAD_CURRENT)
        knob = build(energy_record())
        knob.couple([magnet])

        knob._set(simulator, BEND_RAISED)

        expected = QUAD_GAIN * QUAD_CURRENT * rigidity_factor(RAISED_ENERGY_GEV)
        assert element(simulator.lattice, QUADRUPOLE).PolynomB[1] == pytest.approx(expected)

    def test_leaves_the_unscaled_setpoints_where_they_are(self, simulator) -> None:
        magnet = build(sextupole_record())
        magnet._set(simulator, SEXT_CURRENT)
        before = element(simulator.lattice, SEXTUPOLE_SLICES[0]).PolynomB[2]
        knob = build(energy_record())
        knob.couple([magnet])

        knob._set(simulator, BEND_RAISED)

        assert element(simulator.lattice, SEXTUPOLE_SLICES[0]).PolynomB[2] == before

    def test_couple_adopts_only_the_rigidity_scaled_variables(self) -> None:
        knob = build(energy_record())
        writables = [build(r()) for r in (quadrupole_record, sextupole_record, corrector_record)]

        assert knob.couple(writables) == ("MAG:QUADRUPOLE:SP", "MAG:CORRECTOR:SP")

    def test_refuses_a_nominal_the_curve_maps_to_no_energy(self) -> None:
        with pytest.raises(ValidationError, match="maps the nominal setpoint"):
            build(energy_record(default=0.0))

    def test_refuses_a_setpoint_that_maps_to_no_usable_energy(self, simulator) -> None:
        with pytest.raises(ValueError, match="positive and finite"):
            build(energy_record())._set(simulator, -BEND_NOMINAL)

        assert simulator.lattice.energy == DECK_ENERGY_GEV * EV_PER_GEV


class TestWriteOrder:
    def test_an_energy_move_then_a_setpoint_equals_the_reverse(self) -> None:
        energy_first, names = build_model(build_test_lattice())
        energy_first.set({names["energy"]: BEND_RAISED})
        energy_first.set({names["quadrupole"]: QUAD_CURRENT})

        setpoint_first, _ = build_model(build_test_lattice())
        setpoint_first.set({names["quadrupole"]: QUAD_CURRENT})
        setpoint_first.set({names["energy"]: BEND_RAISED})

        assert lattice_state(setpoint_first.lattice) == pytest.approx(
            lattice_state(energy_first.lattice), rel=1e-12
        )

    def test_a_rejected_batch_leaves_the_rescaled_fields_as_they_were(self) -> None:
        model, names = build_model(build_test_lattice())
        model.set({names["quadrupole"]: QUAD_CURRENT})
        before = lattice_state(model.lattice)

        with pytest.raises(OrbitSolveError):
            model.set({names["energy"]: BEND_RAISED, names["quadrupole"]: 1.0e4})

        assert lattice_state(model.lattice) == before

    def test_repeating_an_energy_move_stays_exact(self) -> None:
        model, names = build_model(build_test_lattice())
        model.set({names["quadrupole"]: QUAD_CURRENT})
        expected = lattice_state(model.lattice)

        model.set({names["energy"]: BEND_RAISED})
        model.set({names["energy"]: BEND_NOMINAL})

        assert lattice_state(model.lattice) == pytest.approx(expected, rel=1e-12)


def demo_energy(simulator: PyATSimulator) -> float:
    """The demo deck's own energy, in GeV."""
    return float(simulator.lattice.energy) / EV_PER_GEV


class TestDemoWiring:
    """The committed demo's block words drive the deck elements they name."""

    @pytest.fixture(scope="class")
    def records(self) -> dict[str, dict[str, Any]]:
        models = yaml.safe_load((DEMO / "models.yaml").read_text(encoding="utf-8"))
        sr = next(model for model in models if model["name"] == "SR")
        return {record["address"]: record for record in sr["wiring"]}

    @pytest.fixture
    def demo(self) -> PyATSimulator:
        return PyATSimulator(at.load_lattice(str(DEMO / "decks" / "SR.json")))

    def test_a_bpm_reads_the_closed_orbit_on_its_axis(self, records, demo) -> None:
        record = records["SR:DIAG:BPM:01:POSITION:X"]
        assert record["engine"] == {"axis": "x"}
        variable = build(record, demo_energy(demo))
        demo.solve()

        assert variable.element_name == "BPM01"
        assert variable._get(demo) == float(demo.last_solution["BPM01"][0])

    def test_a_strength_writes_its_polynomial_coefficient(self, records, demo) -> None:
        record = records["SR:MAG:QF:01:CURRENT:SP"]
        gain = record["calibration"]["curve"]["linear"]["gain"]
        variable = build(record, demo_energy(demo), default_value=0.0)
        variable._set(demo, 300.0)

        assert element(demo.lattice, "QF01").PolynomB[1] == pytest.approx(gain * 300.0)
        assert variable._get(demo) == pytest.approx(300.0, rel=1e-12)

    def test_the_cavity_setpoint_writes_hertz(self, records, demo) -> None:
        variable = build(
            records["SR:RF:CAVITY:01:FREQUENCY:SP"], demo_energy(demo), default_value=500.0
        )
        assert type(variable) is RFVariable
        variable._set(demo, 500.417)

        assert element(demo.lattice, "CAV").Frequency == pytest.approx(500.417e6, rel=1e-15)
        assert variable._get(demo) == pytest.approx(500.417, rel=1e-12)

    @pytest.mark.parametrize("address", ["SR:DIAG:TUNE:X", "SR:DIAG:CHROM:Y"])
    def test_an_optics_readback_builds_no_variable(self, records, address: str) -> None:
        assert variable_from_wiring(records[address], deck_energy_gev=1.0) is None


def test_every_setpoint_class_is_a_calibrated_setpoint() -> None:
    assert all(
        issubclass(kind, CalibratedSetpoint)
        for kind in (StrengthVariable, KickVariable, RFVariable)
    )
