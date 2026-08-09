"""Tests for the ALS-U current->strength transformer variable.

These run against a small purpose-built lattice rather than the real AR
ring: what is under test is the *conversion*, and a generic FODO ring
carrying one device per formula family exercises every branch of it in
milliseconds. The element names are real ones (``QF01``, ``DIPOLE01``, ...)
because ``StrengthMap`` reads each device's nominal current from
``machine.json`` by address -- the currents are the facility's, only the
lattice around them is synthetic.

The load-bearing assertions are the equivalence ones: writing through
:class:`CurrentSetpointVariable` must leave the lattice bit-identical to
what ``StrengthMap.apply`` -- the call the model has always made -- leaves
behind. Everything else here guards a binding or an error path.
"""

from __future__ import annotations

import at
import numpy as np
import pytest
from lume_pyat.exceptions import UnknownElementError
from lume_pyat.simulator import PyATSimulator
from pydantic import ValidationError

from osprey.services.virtual_accelerator.lattice.calibration import AMPS_PER_RADIAN_KICK
from osprey.services.virtual_accelerator.lattice.strengths import (
    CORRECTOR_FAMILIES,
    DIPOLE_FAMILY,
    QUADRUPOLE_FAMILIES,
    SEXTUPOLE_FAMILIES,
    StrengthMap,
)
from osprey.services.virtual_accelerator.model.variables import (
    DECLARED_ATTRIBUTE,
    CurrentSetpointVariable,
)

N_CELLS = 8
QUAD_K = 1.0

# One device per formula family, and the current each is driven to in the
# equivalence tests -- deliberately away from the machine.json nominal so a
# write that silently did nothing would show up as a match against the
# unwritten baseline.
DEVICES = {
    "HCM": 3.5,
    "VCM": -2.25,
    "QF": 400.0,
    "QD": 250.0,
    "SF": 100.0,
    "SD": 80.0,
    "DIPOLE": 950.0,
}

# The magnet half of DEVICES: everything whose formula scales a baked
# strength by the fraction of nominal current, rather than a kick.
MAGNETS = [family for family in sorted(DEVICES) if family not in CORRECTOR_FAMILIES]


def _element_name(cell: int, generic: str, device: str) -> str:
    """Name an element of ``cell``: the addressable device in cell 1, filler elsewhere.

    A ``FamName`` is how both ``StrengthMap`` and ``PyATSimulator`` reach an
    element, so each real device name has to belong to exactly one element --
    but the ring needs its other cells to stay optically stable.
    """
    return device if cell == 1 else f"{generic}_{cell:02d}"


def build_test_ring() -> at.Lattice:
    """An eight-cell FODO ring carrying one addressable device per family.

    Stable in both planes with a comfortable margin, so a caller may perturb
    a device and still solve. The lattice has no relationship to the real AR
    ring beyond the names of the seven elements addressed here.
    """
    elements: list[at.Element] = []
    for cell in range(1, N_CELLS + 1):
        elements += [
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(_element_name(cell, "QUAD_F", "QF01"), 0.3, QUAD_K),
            at.Drift("DRIFT", 0.4),
            at.Sextupole(_element_name(cell, "SEXT_F", "SF01"), 0.15, 1.0),
            at.Drift("DRIFT", 0.4),
            at.Corrector(_element_name(cell, "COR_H", "HCM01"), 0.0, [0.0, 0.0]),
            at.Dipole(_element_name(cell, "BEND", "DIPOLE01"), 1.0, 2 * np.pi / N_CELLS),
            at.Corrector(_element_name(cell, "COR_V", "VCM01"), 0.0, [0.0, 0.0]),
            at.Drift("DRIFT", 0.4),
            at.Sextupole(_element_name(cell, "SEXT_D", "SD01"), 0.15, -1.0),
            at.Drift("DRIFT", 0.4),
            at.Quadrupole(_element_name(cell, "QUAD_D", "QD01"), 0.3, -QUAD_K),
        ]
    ring = at.Lattice(elements, name="TEST_RING", energy=1.0e9, periodicity=1)
    ring.disable_6d()  # 4D optics: the ring carries no RF cavity
    return ring


@pytest.fixture
def ring() -> at.Lattice:
    """A fresh lattice per test -- every write here mutates it in place."""
    return build_test_ring()


@pytest.fixture
def strength_map(ring) -> StrengthMap:
    """A map baked from the same ring the simulator will drive."""
    return StrengthMap(ring)


@pytest.fixture
def simulator(ring) -> PyATSimulator:
    return PyATSimulator(ring)


def make_variable(family: str, strength_map: StrengthMap, **overrides) -> CurrentSetpointVariable:
    """A setpoint variable for ``family`` device ``01``, catalog-shaped."""
    fields = {
        "name": f"SR:MAG:{family}:01:CURRENT:SP",
        "family": family,
        "device_id": "01",
        "default_value": 0.0,
        "unit": "A",
        "default_validation_config": "none",
    }
    return CurrentSetpointVariable(strength_map=strength_map, **{**fields, **overrides})


def element_state(ring: at.Lattice, fam_name: str) -> tuple[list[float] | None, list[float] | None]:
    """``(PolynomB, KickAngle)`` of one element, as plain lists for comparison."""
    element = next(el for el in ring if el.FamName == fam_name)
    return (
        [float(v) for v in element.PolynomB] if hasattr(element, "PolynomB") else None,
        [float(v) for v in element.KickAngle] if hasattr(element, "KickAngle") else None,
    )


class TestDeclaredAttribute:
    def test_covers_exactly_the_families_strengthmap_writes(self):
        """A family StrengthMap.apply accepts but this map misses would bind
        no attribute, and one it rejects would bind a dead one."""
        assert set(DECLARED_ATTRIBUTE) == (
            set(CORRECTOR_FAMILIES)
            | set(QUADRUPOLE_FAMILIES)
            | set(SEXTUPOLE_FAMILIES)
            | {DIPOLE_FAMILY}
        )

    def test_correctors_declare_kick_angle(self):
        assert {DECLARED_ATTRIBUTE[family] for family in CORRECTOR_FAMILIES} == {"KickAngle"}

    def test_every_magnet_declares_the_whole_polynom_b(self):
        """Not ``K``: the quadrupole write goes through the alias, but the
        dipole's index-0 write has no alias at all, and rollback snapshots
        whatever is declared."""
        magnets = set(QUADRUPOLE_FAMILIES) | set(SEXTUPOLE_FAMILIES) | {DIPOLE_FAMILY}
        assert {DECLARED_ATTRIBUTE[family] for family in magnets} == {"PolynomB"}


class TestBinding:
    def test_element_name_and_attribute_are_derived(self, strength_map):
        variable = make_variable("QF", strength_map)
        assert variable.element_name == "QF01"
        assert variable.attribute == "PolynomB"
        assert variable.index is None  # the declared attribute is the whole array

    def test_matching_explicit_binding_is_accepted(self, strength_map):
        """The bindings layer derives the element name for its monitors too,
        so it has one to hand and may pass it."""
        variable = make_variable("HCM", strength_map, element_name="HCM01", attribute="KickAngle")
        assert variable.element_name == "HCM01"

    @pytest.mark.parametrize(
        ("field", "value"),
        [("element_name", "QF02"), ("attribute", "K")],
    )
    def test_contradicting_explicit_binding_is_rejected(self, strength_map, field, value):
        """A binding that disagrees would write one element and snapshot
        another -- rollback would restore the wrong thing."""
        with pytest.raises(ValidationError, match="contradicts"):
            make_variable("QF", strength_map, **{field: value})

    def test_unknown_family_fails_at_definition(self, strength_map):
        with pytest.raises(ValidationError, match="not pyat-coupled"):
            make_variable("ZZ", strength_map)

    def test_strength_map_is_shared_not_copied(self, strength_map):
        """All 348 setpoints of one ring read the same baked baseline."""
        first = make_variable("QF", strength_map)
        second = make_variable("SF", strength_map)
        assert first._strength_map is strength_map
        assert second._strength_map is strength_map

    def test_strength_map_is_not_a_declared_field(self, strength_map):
        """It is infrastructure, not part of the variable's shape -- a
        ring-sized object has no business in a catalog dump."""
        assert "strength_map" not in make_variable("QF", strength_map).model_dump()

    def test_default_value_is_still_required(self, strength_map):
        """Inherited: reset() writes the default back to the lattice."""
        with pytest.raises(ValidationError, match="default_value"):
            make_variable("QF", strength_map, default_value=None)


class TestSet:
    @pytest.mark.parametrize("family", sorted(DEVICES))
    def test_write_matches_strength_map_exactly(self, family, ring, simulator, strength_map):
        """The equivalence property: the variable is a call site for
        ``StrengthMap.apply``, not a second implementation of it."""
        current = DEVICES[family]
        oracle_ring = build_test_ring()
        StrengthMap(oracle_ring).apply(oracle_ring, family, "01", current)

        make_variable(family, strength_map)._set(simulator, current)

        fam_name = f"{family}01"
        assert element_state(ring, fam_name) == element_state(oracle_ring, fam_name)

    @pytest.mark.parametrize("family", sorted(DEVICES))
    def test_write_actually_moves_the_element(self, family, ring, simulator, strength_map):
        """Guards the test above against a no-op write matching a no-op oracle."""
        before = element_state(ring, f"{family}01")
        make_variable(family, strength_map)._set(simulator, DEVICES[family])
        assert element_state(ring, f"{family}01") != before

    def test_corrector_writes_only_its_own_plane(self, ring, simulator, strength_map):
        make_variable("VCM", strength_map)._set(simulator, 4.0)
        _polynom_b, kick_angle = element_state(ring, "VCM01")
        assert kick_angle == [0.0, 4.0 / AMPS_PER_RADIAN_KICK]

    def test_missing_element_raises_unknown_element(self, simulator, strength_map):
        """StrengthMap's bare ValueError, named for what it is -- the class
        ``model.pyat`` exports as ``UnknownDeviceError``."""
        variable = make_variable("QF", strength_map, device_id="99", name="QF99")
        with pytest.raises(UnknownElementError, match="not pyat-coupled") as excinfo:
            variable._set(simulator, 400.0)
        assert "'QF99'" in str(excinfo.value)
        assert isinstance(excinfo.value, ValueError)

    def test_write_is_absolute_not_cumulative(self, ring, simulator, strength_map):
        variable = make_variable("HCM", strength_map)
        variable._set(simulator, 3.0)
        variable._set(simulator, 3.0)
        _polynom_b, kick_angle = element_state(ring, "HCM01")
        assert kick_angle == [3.0 / AMPS_PER_RADIAN_KICK, 0.0]


class TestGet:
    @pytest.mark.parametrize("family", sorted(DEVICES))
    def test_round_trips_the_written_current(self, family, simulator, strength_map):
        variable = make_variable(family, strength_map)
        variable._set(simulator, DEVICES[family])
        assert variable._get(simulator) == pytest.approx(DEVICES[family])

    @pytest.mark.parametrize("family", MAGNETS)
    def test_unwritten_magnet_reads_back_its_nominal(self, family, simulator, strength_map):
        """The baked strength *is* the nominal current's strength, so a
        lattice nobody has written sits at its machine.json nominal."""
        variable = make_variable(family, strength_map)
        nominal = strength_map.i_nom(f"SR:MAG:{family}:01:CURRENT:SP")
        assert variable._get(simulator) == pytest.approx(nominal)

    def test_unwritten_corrector_reads_back_zero(self, simulator, strength_map):
        assert make_variable("HCM", strength_map)._get(simulator) == 0.0

    def test_missing_element_raises_unknown_element(self, simulator, strength_map):
        variable = make_variable("QF", strength_map, device_id="99", name="QF99")
        with pytest.raises(UnknownElementError):
            variable._get(simulator)

    def test_family_that_lost_its_formula_refuses_to_guess(self, simulator, strength_map):
        """Fields stay mutable after construction; a family with no formula
        must refuse rather than fall through to another family's inverse."""
        variable = make_variable("QF", strength_map)
        variable.family = "ZZ"
        with pytest.raises(UnknownElementError, match="not pyat-coupled"):
            variable._get(simulator)
