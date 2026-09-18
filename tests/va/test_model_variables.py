"""The five kinds of model variable a bindings document can name.

These run against a small purpose-built ring rather than a facility's real
one: what is under test is the *conversion* -- hardware units onto lattice
attributes and back -- and a generic FODO ring carrying one element per
binding kind exercises every branch of it in milliseconds. Nothing here is
named after a family: every element name, calibration and slice layout comes
from the fixtures below, exactly as a variable gets them from
``va_bindings.json``.

The load-bearing assertion is the order one: a client writes the energy knob
and a magnet setpoint independently, so the ring has to end in the same state
whichever order they arrive in -- in one batch or in two. Everything else here
guards one kind's slice convention, its way back to hardware units, or a
refusal.

The rigidity is written out longhand in :func:`brho` rather than imported, so
a test failure cannot be explained away by the module under test agreeing with
itself.
"""

from __future__ import annotations

import math

import at
import numpy as np
import pytest
from lume.exceptions import ReadOnlyError
from lume_pyat.actions import PyATReadOnlyScalarVariable
from lume_pyat.exceptions import OrbitSolveError
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator
from pydantic import ValidationError

from osprey.services.virtual_accelerator.bindings import Linear, Table
from osprey.services.virtual_accelerator.model.variables import (
    EV_PER_GEV,
    EnergyVariable,
    KickVariable,
    MonitorVariable,
    RFVariable,
    StrengthVariable,
)

N_CELLS = 8
QUAD_K = 1.0

#: The energy the synthetic ring's lattice is built at, and the energy every
#: calibration below is sampled at.
DECK_ENERGY_GEV = 3.0

#: The addressable elements of cell 1, one per binding kind. Every other cell
#: carries the same optics under filler names, so each of these is reached by
#: exactly one ``FamName`` -- which is how a binding names an element.
QUADRUPOLE = "QUADRUPOLE"
SEXTUPOLE_SLICES = ("SEXTUPOLE_A", "SEXTUPOLE_B")
CORRECTOR_SLICES = ("CORRECTOR_A", "CORRECTOR_B")
BEND = "BEND"
MONITOR = "MONITOR"
CAVITIES = ("CAVITY_A", "CAVITY_B")

DEVICES = frozenset({QUADRUPOLE, BEND, MONITOR, *SEXTUPOLE_SLICES, *CORRECTOR_SLICES, *CAVITIES})

#: A quadrupole family's exported pair: 2 mrad/m of gradient per amp one way,
#: and its exact reciprocal back. Exact, so a round trip that loses the value
#: shows up as a difference rather than as rounding.
QUAD_CALIBRATION = Linear(gain=0.002, offset=0.0)
QUAD_INVERSE = Linear(gain=500.0, offset=0.0)

#: A sextupole family, split over two elements and not rigidity-scaled (a
#: linear control-system conversion does not scale), with an offset so a write
#: that skipped the calibration could not pass by accident.
SEXT_CALIBRATION = Linear(gain=0.5, offset=0.25)

#: A corrector family: microradians of kick per amp, and back.
KICK_CALIBRATION = Linear(gain=1.0e-5, offset=0.0)
KICK_INVERSE = Linear(gain=1.0e5, offset=0.0)

#: The cavity frequency, set in MHz and held in Hz.
RF_CALIBRATION = Linear(gain=1.0e6, offset=0.0)
RF_INVERSE = Linear(gain=1.0e-6, offset=0.0)

#: A position monitor's inverse: metres of solved orbit to the millimetres the
#: control system publishes. The unit factor lives in the exported curve, so
#: nothing in the model multiplies by a thousand.
MONITOR_INVERSE = Linear(gain=1000.0, offset=0.0)

#: The bend's sampled setpoint-to-energy curve, and the nominal setpoint at
#: which the ring is at its deck energy. The curve's absolute values are
#: deliberately not the deck energy: only the ratio is trusted.
ENERGY_TABLE = Table(grid=(0.0, 400.0, 800.0), values=(0.0, 1.6, 3.2))
BEND_NOMINAL = 800.0

#: A setpoint 5 % above nominal, and the energy it implies: the table is
#: straight through its last segment, so 840 A reads 3.36 and the ring runs at
#: ``3.0 * 3.36 / 3.2``.
BEND_RAISED = 840.0
RAISED_ENERGY_GEV = 3.15

#: The currents the tests drive, away from any nominal so an unwritten
#: baseline could not be mistaken for a successful write.
QUAD_CURRENT = 250.0
SEXT_CURRENT = 40.0
KICK_CURRENT = 3.5
RF_SETPOINT = 499.68


def brho(energy_gev: float) -> float:
    """The beam rigidity at ``energy_gev``, in tesla-metres.

    The control system's own expression, spelled out here: ``(10/c) *
    sqrt((E + E0)^2 - E0^2)`` with the electron rest mass in GeV. Written
    longhand on purpose -- an expectation computed by the code under test
    would pass whatever that code did.
    """
    rest_mass = 0.51099906e-3
    return (10.0 / 2.99792458) * math.sqrt((energy_gev + rest_mass) ** 2 - rest_mass**2)


def rigidity_factor(energy_gev: float) -> float:
    """What a rigidity-scaled physics value is worth at ``energy_gev``."""
    return brho(DECK_ENERGY_GEV) / brho(energy_gev)


def _named(cell: int, filler: str, device: str) -> str:
    """Name an element of ``cell``: the addressable device in cell 1, filler elsewhere."""
    return device if cell == 1 else f"{filler}_{cell:02d}"


def build_test_ring() -> at.Lattice:
    """An eight-cell FODO ring carrying one element per binding kind.

    Stable in both planes with a comfortable margin, so a test may drive a
    device hard and still solve. Two cavities, so "every cavity" is a claim
    the tests can fail; they stay disabled, because what is under test is
    where a frequency and an energy land, not six-dimensional dynamics.
    """
    elements: list[at.Element] = []
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
    ring = at.Lattice(
        elements, name="TEST_RING", energy=DECK_ENERGY_GEV * EV_PER_GEV, periodicity=1
    )
    ring.disable_6d()
    return ring


@pytest.fixture
def ring() -> at.Lattice:
    """A fresh lattice per test -- every write here mutates it in place."""
    return build_test_ring()


@pytest.fixture
def simulator(ring: at.Lattice) -> PyATSimulator:
    return PyATSimulator(ring)


def slices(names: tuple[str, ...], attribute: str, index: int | None, weight: float) -> list[dict]:
    """One binding per named element, all on the same field at ``weight``."""
    return [
        {"element_name": name, "attribute": attribute, "index": index, "weight": weight}
        for name in names
    ]


def quadrupole(**overrides) -> StrengthVariable:
    """The single-slice, rigidity-scaled magnet setpoint most tests drive."""
    fields = {
        "name": "RING:MAG:QUADRUPOLE:SP",
        "bindings": slices((QUADRUPOLE,), "PolynomB", 1, 1.0),
        "calibration": QUAD_CALIBRATION,
        "monitor_inverse": QUAD_INVERSE,
        "energy_scaling": "brho",
        "deck_energy_gev": DECK_ENERGY_GEV,
        "default_value": 0.0,
        "unit": "A",
        "default_validation_config": "none",
    }
    return StrengthVariable(**{**fields, **overrides})


def sextupole(**overrides) -> StrengthVariable:
    """A split magnet setpoint that the control system does not scale."""
    fields = {
        "name": "RING:MAG:SEXTUPOLE:SP",
        "bindings": slices(SEXTUPOLE_SLICES, "PolynomB", 2, 1.0),
        "calibration": SEXT_CALIBRATION,
        "energy_scaling": "none",
        "deck_energy_gev": DECK_ENERGY_GEV,
        "default_value": 0.0,
        "unit": "A",
        "default_validation_config": "none",
    }
    return StrengthVariable(**{**fields, **overrides})


def corrector(**overrides) -> KickVariable:
    """A split corrector setpoint: half the kick on each piece."""
    fields = {
        "name": "RING:MAG:CORRECTOR:SP",
        "bindings": slices(CORRECTOR_SLICES, "KickAngle", 0, 0.5),
        "calibration": KICK_CALIBRATION,
        "monitor_inverse": KICK_INVERSE,
        "energy_scaling": "brho",
        "deck_energy_gev": DECK_ENERGY_GEV,
        "default_value": 0.0,
        "unit": "A",
        "default_validation_config": "none",
    }
    return KickVariable(**{**fields, **overrides})


def cavity(**overrides) -> RFVariable:
    """The ring frequency, over both cavities."""
    fields = {
        "name": "RING:RF:FREQUENCY:SP",
        "bindings": slices(CAVITIES, "Frequency", None, 1.0),
        "calibration": RF_CALIBRATION,
        "monitor_inverse": RF_INVERSE,
        "energy_scaling": "none",
        "deck_energy_gev": DECK_ENERGY_GEV,
        "default_value": RF_SETPOINT,
        "unit": "MHz",
        "default_validation_config": "none",
    }
    return RFVariable(**{**fields, **overrides})


def monitor(**overrides) -> MonitorVariable:
    """One transverse reading of the ring's single addressable monitor."""
    fields = {
        "name": "RING:DIAG:MONITOR:X",
        "element_name": MONITOR,
        "axis": "x",
        "monitor_inverse": MONITOR_INVERSE,
        "read_only": True,
        "unit": "mm",
        "default_validation_config": "none",
    }
    return MonitorVariable(**{**fields, **overrides})


def energy_knob(**overrides) -> EnergyVariable:
    """The bend's setpoint, driving the ring energy through its table."""
    fields = {
        "name": "RING:MAG:BEND:SP",
        "energy_table": ENERGY_TABLE,
        "nominal": BEND_NOMINAL,
        "deck_energy_gev": DECK_ENERGY_GEV,
        "default_value": BEND_NOMINAL,
        "unit": "A",
        "default_validation_config": "none",
    }
    return EnergyVariable(**{**fields, **overrides})


def element(ring: at.Lattice, name: str) -> at.Element:
    """The one element of ``ring`` named ``name``."""
    return next(candidate for candidate in ring if candidate.FamName == name)


def lattice_state(ring: at.Lattice) -> dict[str, float]:
    """Every number a write in this suite can reach, flattened for comparison.

    The ring energy, and each addressable element's polynomial coefficients,
    kick components, frequency and energy. Flat floats rather than arrays so
    one assertion covers the whole state and names the field that moved.
    """
    state = {"ring.energy": float(ring.energy)}
    for candidate in ring:
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


def build_model(ring: at.Lattice) -> tuple[LUMEPyATModel, dict[str, str]]:
    """A model over ``ring`` carrying one variable of every kind.

    Returns the model and the address of each kind, so a test names a write
    by kind rather than by repeating an address.
    """
    writables = [quadrupole(), sextupole(), corrector(), cavity()]
    knob = energy_knob()
    knob.couple(writables)
    model = LUMEPyATModel(
        simulator=PyATSimulator(ring), action_variables=[*writables, knob, monitor()]
    )
    names = {
        "quadrupole": writables[0].name,
        "sextupole": writables[1].name,
        "corrector": writables[2].name,
        "cavity": writables[3].name,
        "energy": knob.name,
        "monitor": "RING:DIAG:MONITOR:X",
    }
    return model, names


class TestStrengthVariable:
    """A hardware setpoint onto a polynomial coefficient, slice by slice."""

    def test_writes_the_value_the_calibration_converts_it_to(self, simulator) -> None:
        """The element holds the physics value, never the hardware one."""
        sextupole()._set(simulator, SEXT_CURRENT)

        held = element(simulator.lattice, SEXTUPOLE_SLICES[0]).PolynomB[2]
        assert held == pytest.approx(SEXT_CALIBRATION.gain * SEXT_CURRENT + SEXT_CALIBRATION.offset)

    def test_every_slice_of_a_split_magnet_carries_the_whole_strength(self, simulator) -> None:
        """A split magnet's pieces each take the family's full strength."""
        sextupole()._set(simulator, SEXT_CURRENT)

        held = [element(simulator.lattice, name).PolynomB[2] for name in SEXTUPOLE_SLICES]
        assert held[0] == pytest.approx(held[1])
        assert held[0] != 0.0

    def test_the_first_slice_reads_back_the_whole_strength(self, simulator) -> None:
        """Read is the inverse of write, to the precision of the exported pair."""
        variable = quadrupole()
        variable._set(simulator, QUAD_CURRENT)

        assert variable._get(simulator) == pytest.approx(QUAD_CURRENT, rel=1e-9)

    def test_refuses_slices_that_share_the_strength_out(self) -> None:
        """Halved weights would leave each piece at half the strength."""
        with pytest.raises(ValidationError, match="written to every slice in full"):
            sextupole(bindings=slices(SEXTUPOLE_SLICES, "PolynomB", 2, 0.5))

    def test_reads_back_through_the_exported_inverse_not_the_calibration(self, simulator) -> None:
        """The two directions are independent data: a readback follows the
        exported inverse even where it disagrees with the calibration."""
        variable = quadrupole(monitor_inverse=Linear(gain=510.0, offset=0.0))
        variable._set(simulator, QUAD_CURRENT)

        assert variable._get(simulator) == pytest.approx(1.02 * QUAD_CURRENT, rel=1e-9)

    def test_a_binding_with_no_inverse_has_no_way_back_to_hardware(self, simulator) -> None:
        """An identity readback carries no inverse, and none is fabricated."""
        variable = sextupole()
        variable._set(simulator, SEXT_CURRENT)

        with pytest.raises(NotImplementedError, match="no monitor_inverse"):
            variable._get(simulator)


class TestReadback:
    """What the control system reads back once a setpoint has been written."""

    def test_is_the_written_value_mapped_back_through_the_inverse(self) -> None:
        """``monitor_inverse(calibration(I))`` -- no inversion anywhere."""
        variable = quadrupole(monitor_inverse=Linear(gain=510.0, offset=0.0))

        assert variable.readback(QUAD_CURRENT) == pytest.approx(1.02 * QUAD_CURRENT, rel=1e-9)

    def test_collapses_to_the_written_value_where_no_inverse_was_exported(self) -> None:
        """Which is the whole of an identity readback."""
        assert sextupole().readback(SEXT_CURRENT) == SEXT_CURRENT

    def test_does_not_move_with_the_ring_energy(self, simulator) -> None:
        """Both directions scale with the rigidity, so the factors cancel and
        one readback answers at every energy."""
        variable = quadrupole()
        simulator.lattice.energy = RAISED_ENERGY_GEV * EV_PER_GEV

        assert variable.readback(QUAD_CURRENT) == pytest.approx(QUAD_CURRENT, rel=1e-9)


class TestRigidityScaling:
    """What a family the control system scales with the rigidity is worth."""

    def test_a_scaled_write_takes_the_factor_for_the_energy_the_ring_is_at(self, simulator) -> None:
        """Not the deck energy: the setpoint lands where the last energy move
        left the family."""
        simulator.lattice.energy = RAISED_ENERGY_GEV * EV_PER_GEV

        quadrupole()._set(simulator, QUAD_CURRENT)

        expected = QUAD_CALIBRATION.gain * QUAD_CURRENT * rigidity_factor(RAISED_ENERGY_GEV)
        assert element(simulator.lattice, QUADRUPOLE).PolynomB[1] == pytest.approx(expected)

    def test_an_unscaled_write_ignores_the_ring_energy(self, simulator) -> None:
        """A linear control-system conversion does not scale, so the energy
        knob leaves the family exactly where it was."""
        simulator.lattice.energy = RAISED_ENERGY_GEV * EV_PER_GEV

        sextupole()._set(simulator, SEXT_CURRENT)

        expected = SEXT_CALIBRATION.gain * SEXT_CURRENT + SEXT_CALIBRATION.offset
        assert element(simulator.lattice, SEXTUPOLE_SLICES[0]).PolynomB[2] == pytest.approx(
            expected
        )

    def test_the_factor_is_removed_again_on_the_way_back(self, simulator) -> None:
        """A read at a moved energy still answers with the setpoint written."""
        variable = quadrupole()
        simulator.lattice.energy = RAISED_ENERGY_GEV * EV_PER_GEV
        variable._set(simulator, QUAD_CURRENT)

        assert variable._get(simulator) == pytest.approx(QUAD_CURRENT, rel=1e-9)


class TestKickVariable:
    """A corrector setpoint, divided over the slices it is bound to."""

    def test_divides_the_kick_over_its_slices(self, simulator) -> None:
        """Two pieces, half the kick each: together they bend the beam by the
        whole of it."""
        corrector()._set(simulator, KICK_CURRENT)

        held = [element(simulator.lattice, name).KickAngle[0] for name in CORRECTOR_SLICES]
        whole = KICK_CALIBRATION.gain * KICK_CURRENT * rigidity_factor(DECK_ENERGY_GEV)
        assert held == pytest.approx([whole / 2.0, whole / 2.0])

    def test_reads_back_the_whole_kick(self, simulator) -> None:
        """Slice one times the slice count, which is the value the control
        system reads."""
        variable = corrector()
        variable._set(simulator, KICK_CURRENT)

        assert variable._get(simulator) == pytest.approx(KICK_CURRENT, rel=1e-9)

    def test_refuses_slices_that_each_carry_the_whole_kick(self) -> None:
        """Unit weights would bend the beam by twice what was asked for."""
        with pytest.raises(ValidationError, match="divided over the 2 slices"):
            corrector(bindings=slices(CORRECTOR_SLICES, "KickAngle", 0, 1.0))

    def test_refuses_slices_with_unequal_shares(self) -> None:
        """Shares that are not 1/n leave the first slice reading back wrong."""
        uneven = slices(CORRECTOR_SLICES, "KickAngle", 0, 0.5)
        uneven[1]["weight"] = 0.6

        with pytest.raises(ValidationError, match="divided over the 2 slices"):
            corrector(bindings=uneven)


class TestRFVariable:
    """The cavity frequency, over every cavity in the ring."""

    def test_writes_the_frequency_to_every_cavity(self, simulator) -> None:
        cavity()._set(simulator, RF_SETPOINT)

        held = [element(simulator.lattice, name).Frequency for name in CAVITIES]
        assert held == pytest.approx([RF_SETPOINT * RF_CALIBRATION.gain] * len(CAVITIES))

    def test_reads_back_the_frequency_in_the_unit_it_is_set_in(self, simulator) -> None:
        variable = cavity()
        variable._set(simulator, RF_SETPOINT)

        assert variable._get(simulator) == pytest.approx(RF_SETPOINT, rel=1e-9)

    def test_refuses_slices_that_share_the_frequency_out(self) -> None:
        """A ring's cavities all run at the same frequency."""
        with pytest.raises(ValidationError, match="written to every slice in full"):
            cavity(bindings=slices(CAVITIES, "Frequency", None, 0.5))


class TestMonitorVariable:
    """One solved orbit reading, in the units the facility publishes."""

    def test_serves_the_reading_through_the_exported_inverse(self, simulator) -> None:
        """The inverse carries the metre-to-millimetre factor, so the reading
        goes through it unscaled."""
        corrector()._set(simulator, KICK_CURRENT)
        simulator.solve()
        metres = PyATReadOnlyScalarVariable(
            name="raw", element_name=MONITOR, axis="x", read_only=True
        )._get(simulator)

        served = monitor()._get(simulator)

        assert metres != 0.0, "the kick must move the orbit for this to test anything"
        assert served == pytest.approx(MONITOR_INVERSE.gain * metres)

    def test_reads_the_axis_its_binding_names(self, simulator) -> None:
        """A horizontal kick moves the horizontal reading and not the other."""
        corrector()._set(simulator, KICK_CURRENT)
        simulator.solve()

        assert monitor()._get(simulator) != 0.0
        assert monitor(name="RING:DIAG:MONITOR:Y", axis="y")._get(simulator) == 0.0

    def test_is_read_only(self, simulator) -> None:
        with pytest.raises(ReadOnlyError):
            monitor()._set(simulator, 1.0)

    def test_refuses_a_monitor_with_no_exported_inverse(self) -> None:
        """A reading has no written value to fall back on, so the inverse is
        what makes it publishable at all."""
        with pytest.raises(ValidationError, match="monitor_inverse"):
            MonitorVariable(name="RING:DIAG:MONITOR:X", element_name=MONITOR, axis="x")


class TestEnergyVariable:
    """The ring energy, driven by the bend's own hardware setpoint."""

    def test_the_nominal_setpoint_leaves_the_ring_at_its_deck_energy(self, simulator) -> None:
        """Whatever the table's absolute scale says: only its ratio is trusted.

        To the last bits rather than exactly: pyAT re-derives the energy it is
        handed from the particle's momentum, so a set-then-read round trip
        moves the value by an ulp of its own accord.
        """
        energy_knob()._set(simulator, BEND_NOMINAL)

        assert simulator.lattice.energy == pytest.approx(DECK_ENERGY_GEV * EV_PER_GEV, rel=1e-15)

    def test_the_energy_follows_the_table_ratio(self, simulator) -> None:
        energy_knob()._set(simulator, BEND_RAISED)

        assert simulator.lattice.energy == pytest.approx(RAISED_ENERGY_GEV * EV_PER_GEV)

    def test_the_cavities_move_with_the_ring(self, simulator) -> None:
        """pyAT holds the energy in two places and a solve reads both."""
        energy_knob()._set(simulator, BEND_RAISED)

        held = [element(simulator.lattice, name).Energy for name in CAVITIES]
        assert held == pytest.approx([RAISED_ENERGY_GEV * EV_PER_GEV] * len(CAVITIES))

    def test_rescales_every_rigidity_scaled_binding_it_adopted(self, simulator) -> None:
        """The setpoint has not moved, so the strength moves with the rigidity."""
        magnet = quadrupole()
        magnet._set(simulator, QUAD_CURRENT)
        knob = energy_knob()
        knob.couple([magnet])

        knob._set(simulator, BEND_RAISED)

        expected = QUAD_CALIBRATION.gain * QUAD_CURRENT * rigidity_factor(RAISED_ENERGY_GEV)
        assert element(simulator.lattice, QUADRUPOLE).PolynomB[1] == pytest.approx(expected)

    def test_shares_the_rescale_out_over_a_kick_s_slices(self, simulator) -> None:
        """A factor on what each piece already holds keeps the sharing intact."""
        kick = corrector()
        kick._set(simulator, KICK_CURRENT)
        knob = energy_knob()
        knob.couple([kick])

        knob._set(simulator, BEND_RAISED)

        whole = KICK_CALIBRATION.gain * KICK_CURRENT * rigidity_factor(RAISED_ENERGY_GEV)
        held = [element(simulator.lattice, name).KickAngle[0] for name in CORRECTOR_SLICES]
        assert held == pytest.approx([whole / 2.0, whole / 2.0])

    def test_leaves_the_unscaled_bindings_where_they_are(self, simulator) -> None:
        """Which is a known physics gap, and the control system's own behaviour."""
        magnet = sextupole()
        magnet._set(simulator, SEXT_CURRENT)
        before = element(simulator.lattice, SEXTUPOLE_SLICES[0]).PolynomB[2]
        knob = energy_knob()
        knob.couple([magnet])

        knob._set(simulator, BEND_RAISED)

        assert element(simulator.lattice, SEXTUPOLE_SLICES[0]).PolynomB[2] == before

    def test_couple_adopts_only_the_rigidity_scaled_variables(self) -> None:
        """A caller hands over every writable and does not have to know which
        families move with the energy."""
        knob = energy_knob()

        adopted = knob.couple([quadrupole(), sextupole(), corrector(), cavity()])

        assert adopted == (quadrupole().name, corrector().name)

    def test_declares_every_field_its_write_touches(self, simulator) -> None:
        """What the model does not know about, it cannot roll back."""
        magnet = quadrupole()
        knob = energy_knob()
        knob.couple([magnet])

        targets = knob.snapshot_targets(simulator)

        assert (None, "energy") in targets
        assert [(index, attribute) for index, attribute in targets if attribute == "Energy"] == [
            (simulator.element_index(name), "Energy") for name in CAVITIES
        ]
        assert (simulator.element_index(QUADRUPOLE), "PolynomB") in targets

    def test_refuses_a_nominal_the_table_maps_to_no_energy(self) -> None:
        """The ratio would be undefined for every setpoint, not just this one."""
        with pytest.raises(ValidationError, match="maps the nominal setpoint"):
            energy_knob(nominal=0.0)

    def test_refuses_a_deck_energy_a_ring_cannot_run_at(self) -> None:
        with pytest.raises(ValidationError, match="positive number of GeV"):
            energy_knob(deck_energy_gev=0.0)

    def test_refuses_a_setpoint_that_maps_to_no_usable_energy(self, simulator) -> None:
        """And writes nothing: a ring at a negative energy is not a state to
        roll back from."""
        with pytest.raises(ValueError, match="positive and finite"):
            energy_knob()._set(simulator, -BEND_NOMINAL)

        assert simulator.lattice.energy == DECK_ENERGY_GEV * EV_PER_GEV

    def test_the_present_energy_does_not_say_what_setpoint_made_it(self, simulator) -> None:
        """The table is exported in one direction only, and is not inverted."""
        with pytest.raises(NotImplementedError, match="one direction only"):
            energy_knob()._get(simulator)


class TestWriteOrder:
    """Two clients, two writes, no ordering between them."""

    def test_an_energy_move_then_a_setpoint_equals_the_setpoint_then_the_move(
        self,
    ) -> None:
        """The property the serving path needs: the ring ends in one state."""
        energy_first, names = build_model(build_test_ring())
        energy_first.set({names["energy"]: BEND_RAISED})
        energy_first.set({names["quadrupole"]: QUAD_CURRENT})

        setpoint_first, _ = build_model(build_test_ring())
        setpoint_first.set({names["quadrupole"]: QUAD_CURRENT})
        setpoint_first.set({names["energy"]: BEND_RAISED})

        assert lattice_state(setpoint_first.lattice) == pytest.approx(
            lattice_state(energy_first.lattice), rel=1e-12
        )

    def test_one_batch_lands_the_same_whichever_order_it_lists(self) -> None:
        """A batch is a dict, and a client's insertion order is its own."""
        energy_first, names = build_model(build_test_ring())
        energy_first.set({names["energy"]: BEND_RAISED, names["quadrupole"]: QUAD_CURRENT})

        setpoint_first, _ = build_model(build_test_ring())
        setpoint_first.set({names["quadrupole"]: QUAD_CURRENT, names["energy"]: BEND_RAISED})

        assert lattice_state(setpoint_first.lattice) == pytest.approx(
            lattice_state(energy_first.lattice), rel=1e-12
        )

    def test_the_whole_move_is_one_batch(self) -> None:
        """The rescale runs inside the energy write, so a client that moves the
        energy and a setpoint together pays for one orbit solve."""
        model, names = build_model(build_test_ring())
        solves = 0
        solve = model.simulator.solve

        def counted() -> dict[str, tuple[float, float]]:
            nonlocal solves
            solves += 1
            return solve()

        model.simulator.solve = counted  # type: ignore[method-assign]
        model.set({names["energy"]: BEND_RAISED, names["quadrupole"]: QUAD_CURRENT})

        assert solves == 1

    def test_a_rejected_batch_leaves_the_rescaled_fields_as_they_were(self) -> None:
        """The energy write reaches every adopted binding, so the rollback has
        to as well -- a stale strength would be scaled twice by the next move."""
        model, names = build_model(build_test_ring())
        model.set({names["quadrupole"]: QUAD_CURRENT})
        before = lattice_state(model.lattice)

        with pytest.raises(OrbitSolveError):
            model.set({names["energy"]: BEND_RAISED, names["quadrupole"]: 1.0e4})

        assert lattice_state(model.lattice) == before

    def test_repeating_an_energy_move_stays_exact_for_the_setpoint_on_the_machine(
        self,
    ) -> None:
        """Nothing retains a hardware value, so a round trip through two energies
        has to come back to the strength the setpoint asks for."""
        model, names = build_model(build_test_ring())
        model.set({names["quadrupole"]: QUAD_CURRENT})
        expected = lattice_state(model.lattice)

        model.set({names["energy"]: BEND_RAISED})
        model.set({names["energy"]: BEND_NOMINAL})

        assert lattice_state(model.lattice) == pytest.approx(expected, rel=1e-12)
