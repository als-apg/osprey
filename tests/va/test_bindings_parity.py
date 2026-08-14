"""The bound catalog addresses the ring it was baked against.

The model is served from the bound catalog alone. There is no field-for-field
parity check against ``build_variable_catalog``: with no second derivation to
compare against, comparing the bound catalog to a plain one rebuilt beside it
would only assert that one function calls another. What this module pins are
the properties that hold on their own terms -- the catalog's shape, and the
baselines below.

**Baked baselines (standing).** Every setpoint's element must be the one
``StrengthMap`` snapshotted its baked strength from, bit-for-bit. This is
the standing check: the ring, the map and these
variables are one set, and a change that rebuilt the ring after baking --
or reordered construction so the map baked a lattice the variables do not
write -- would leave every conversion scaled by a stale baseline while
every value still looked plausible. Bit-identity, not tolerance, because
these are copies of one number, not the results of two computations.

The uniqueness assertions guard the other construction-time foot-gun:
``lume_pyat`` refuses to bind a name that addresses more than one element,
so a binding scheme that ever produced a duplicate would fail at model
construction rather than here. The ALS-U AR flat-name convention gives
every addressable device its own name; this proves it, against the real
ring, for all 492 bindings.
"""

from __future__ import annotations

import at
import pytest
from lume.variables import ScalarVariable
from lume_pyat.actions import PyATReadOnlyScalarVariable
from lume_pyat.simulator import PyATSimulator

from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.lattice.calibration import AMPS_PER_RADIAN_KICK
from osprey.services.virtual_accelerator.lattice.strengths import (
    _CORRECTOR_PLANE,
    CORRECTOR_FAMILIES,
    DIPOLE_FAMILY,
    QUADRUPOLE_FAMILIES,
    SEXTUPOLE_FAMILIES,
    StrengthMap,
)
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    build_manifest,
)
from osprey.services.virtual_accelerator.model.bindings import build_action_variables
from osprey.services.virtual_accelerator.model.variables import (
    DECLARED_ATTRIBUTE,
    CurrentSetpointVariable,
)

# The contract, pinned as literals for the same reason test_pyat_ring_model
# pins them: a silent drift in either count is what this suite exists to catch.
EXPECTED_INPUTS = 348
EXPECTED_OUTPUTS = 144
EXPECTED_MONITORS = 72

# Amps for the corrector conversion witness. AMPS_PER_RADIAN_KICK is 1e6,
# whose reciprocal is not exact in binary, so `2.5 / A` and `2.5 * (1 / A)`
# differ in the last bit -- which is what makes this current able to tell
# the two formulations apart at all.
WITNESS_CURRENT = 2.5


@pytest.fixture(scope="module")
def channels() -> list[dict]:
    """The manifest channel list, built once and shared by both catalogs."""
    return build_manifest()["channels"]


@pytest.fixture(scope="module")
def ring() -> at.Lattice:
    """The real ALS-U AR ring -- the lattice the bindings must address."""
    return build_ring()


@pytest.fixture(scope="module")
def strength_map(ring) -> StrengthMap:
    """Baked from that same ring, before anything has been written to it."""
    return StrengthMap(ring)


@pytest.fixture(scope="module")
def bound(channels, strength_map) -> dict[str, ScalarVariable]:
    return build_action_variables(channels, strength_map=strength_map)


@pytest.fixture(scope="module")
def setpoints(bound) -> dict[str, CurrentSetpointVariable]:
    return {name: var for name, var in bound.items() if not var.read_only}


@pytest.fixture(scope="module")
def monitors(bound) -> dict[str, PyATReadOnlyScalarVariable]:
    return {name: var for name, var in bound.items() if var.read_only}


@pytest.fixture(scope="module")
def channel_by_address(channels) -> dict[str, dict]:
    return {channel["address"]: channel for channel in channels}


def live_baseline(element: at.Element, family: str) -> float:
    """Read back, from the element itself, the strength ``StrengthMap`` baked.

    The same reads ``StrengthMap.__init__`` performs -- restated here rather
    than shared, because a helper both sides called would agree with itself
    even if it read the wrong field.
    """
    if family in QUADRUPOLE_FAMILIES:
        # PolynomB[1], not K: pyAT aliases the two, and this is the storage
        # the declared attribute snapshots for rollback.
        return float(element.PolynomB[1])
    if family in SEXTUPOLE_FAMILIES:
        return float(element.PolynomB[2])
    if family == DIPOLE_FAMILY:
        return float(element.PolynomB[0])
    return float(element.KickAngle[_CORRECTOR_PLANE[family]])


class TestCatalogShape:
    """What the bound catalog contains, independent of how it was derived."""

    def test_counts(self, setpoints, monitors):
        assert len(setpoints) == EXPECTED_INPUTS
        assert len(monitors) == EXPECTED_OUTPUTS

    def test_address_keys_are_full_six_level_addresses(self, bound):
        """No address translation anywhere: the key is the channel address."""
        assert all(len(address.split(":")) == 6 for address in bound)

    def test_setpoint_echo_is_excluded(self, bound):
        """``:RB`` is the IOC mirroring a write back; not model state."""
        assert not [address for address in bound if address.endswith(":RB")]

    def test_variables_are_bound_action_classes(self, setpoints, monitors):
        assert {type(v) for v in setpoints.values()} == {CurrentSetpointVariable}
        assert {type(v) for v in monitors.values()} == {PyATReadOnlyScalarVariable}


class TestBinding:
    def test_element_name_is_family_plus_device(self, bound, channel_by_address):
        """The flat-name concatenation, for setpoints and monitors alike."""
        assert all(
            variable.element_name
            == f"{channel_by_address[address]['family']}{channel_by_address[address]['device']}"
            for address, variable in bound.items()
        )

    def test_setpoints_declare_their_family_attribute(self, setpoints):
        assert all(
            variable.attribute == DECLARED_ATTRIBUTE[variable.family]
            for variable in setpoints.values()
        )

    def test_setpoints_declare_the_whole_array(self, setpoints):
        """``index is None``: rollback snapshots the storage, not one slot."""
        assert all(variable.index is None for variable in setpoints.values())

    def test_setpoints_carry_their_family_and_device(self, setpoints, channel_by_address):
        assert all(
            (variable.family, variable.device_id)
            == (channel_by_address[address]["family"], channel_by_address[address]["device"])
            for address, variable in setpoints.items()
        )

    def test_every_setpoint_shares_the_one_strength_map(self, setpoints, strength_map):
        assert all(variable._strength_map is strength_map for variable in setpoints.values())

    def test_monitor_axis_comes_from_the_subfield(self, monitors, channel_by_address):
        expected = {"X": "x", "Y": "y"}
        assert all(
            variable.axis == expected[channel_by_address[address]["subfield"]]
            for address, variable in monitors.items()
        )

    def test_both_axes_of_every_monitor_are_present(self, monitors):
        axes_by_element: dict[str, set[str]] = {}
        for variable in monitors.values():
            axes_by_element.setdefault(variable.element_name, set()).add(variable.axis)
        assert len(axes_by_element) == EXPECTED_MONITORS
        assert all(axes == {"x", "y"} for axes in axes_by_element.values())


class TestAddressability:
    """No binding may name something the lattice cannot tell apart."""

    def test_setpoint_element_names_are_distinct(self, setpoints):
        """348 setpoints, 348 elements: two setpoints sharing an element
        would each overwrite the other's strength."""
        assert len({variable.element_name for variable in setpoints.values()}) == EXPECTED_INPUTS

    def test_monitor_bindings_are_distinct(self, monitors):
        """72 monitors read twice each -- the *binding* (element and axis) is
        what has to be unique, not the element name."""
        bindings = {(variable.element_name, variable.axis) for variable in monitors.values()}
        assert len(bindings) == EXPECTED_OUTPUTS
        assert len({element for element, _axis in bindings}) == EXPECTED_MONITORS

    def test_every_bound_name_addresses_exactly_one_ring_element(self, bound, ring):
        """What ``PyATSimulator.unique_element_index`` enforces at model
        construction. The ring does repeat names (drifts, mostly); none of
        them may be one a variable binds."""
        counts: dict[str, int] = {}
        for element in ring:
            counts[element.FamName] = counts.get(element.FamName, 0) + 1
        ambiguous = sorted(
            {
                variable.element_name
                for variable in bound.values()
                if counts.get(variable.element_name, 0) != 1
            }
        )
        assert ambiguous == []

    def test_every_monitor_binds_an_actual_monitor(self, monitors, ring):
        """A BPM reading comes off the solved orbit, which is keyed by
        monitor ``FamName`` -- binding a non-monitor would read nothing."""
        monitor_names = {element.FamName for element in ring if isinstance(element, at.Monitor)}
        assert {variable.element_name for variable in monitors.values()} <= monitor_names


class TestBakedBaselines:
    """Standing: the map's baked baseline is the live ring's, bit-for-bit."""

    def test_every_setpoint_baseline_matches_the_live_element(self, setpoints, strength_map, ring):
        element_by_name = {element.FamName: element for element in ring}
        divergent = []
        for variable in setpoints.values():
            element = element_by_name[variable.element_name]
            baked = strength_map.baked(variable.element_name)
            live = live_baseline(element, variable.family)
            if baked != live:
                divergent.append((variable.element_name, baked, live))
        assert divergent == []
        assert len(setpoints) == EXPECTED_INPUTS  # all 348 were actually checked

    def test_quadrupole_baselines_are_the_polynom_b_gradient(self, setpoints, ring):
        """``K`` and ``PolynomB[1]`` are one number under pyAT aliasing --
        which is why declaring ``PolynomB`` covers the quadrupole write."""
        element_by_name = {element.FamName: element for element in ring}
        quads = [v for v in setpoints.values() if v.family in QUADRUPOLE_FAMILIES]
        assert quads  # the ring has quadrupole setpoints to check
        assert all(
            float(element_by_name[v.element_name].K)
            == float(element_by_name[v.element_name].PolynomB[1])
            for v in quads
        )

    def test_scaled_baselines_are_nonzero(self, setpoints, strength_map):
        """Guards the identity above from holding trivially: a ring whose
        strengths were all zero would match any baseline. Only the families
        whose formula *scales* a baked strength are covered -- the two whose
        baseline is legitimately zero are pinned below."""
        scaled = QUADRUPOLE_FAMILIES | SEXTUPOLE_FAMILIES
        assert all(
            strength_map.baked(v.element_name) != 0.0
            for v in setpoints.values()
            if v.family in scaled
        )

    def test_zero_baselines_are_the_two_that_should_be(self, setpoints, strength_map, ring):
        """A dipole's baked value is its *field error*, exactly zero at
        nominal current, and a freshly built ring carries no corrector kick.
        Both are meant to be zero -- so what has to be nonzero instead is the
        bending geometry the dipole formula scales."""
        element_by_name = {element.FamName: element for element in ring}
        zeroed = {v.family for v in setpoints.values() if strength_map.baked(v.element_name) == 0.0}
        assert zeroed == CORRECTOR_FAMILIES | {DIPOLE_FAMILY}
        assert all(
            float(element_by_name[v.element_name].BendingAngle) != 0.0
            and float(element_by_name[v.element_name].Length) != 0.0
            for v in setpoints.values()
            if v.family == DIPOLE_FAMILY
        )

    def test_corrector_conversion_divides_and_never_multiplies_by_the_reciprocal(self, setpoints):
        """The kick is ``I / AMPS_PER_RADIAN_KICK``. Multiplying by a
        precomputed reciprocal would be a different number in the last bit,
        and the ORM cross-check compares against this at 1e-9.

        Driven against a throwaway one-corrector lattice: only the
        arithmetic is under test, and the real ring must stay at the
        baselines the checks above just read.
        """
        variable = next(v for v in setpoints.values() if v.family in CORRECTOR_FAMILIES)
        plane = _CORRECTOR_PLANE[variable.family]
        witness = at.Lattice(
            [at.Corrector(variable.element_name, 0.0, [0.0, 0.0]), at.Drift("DRIFT", 1.0)],
            name="WITNESS",
            energy=1.0e9,
            periodicity=1,
        )
        variable._set(PyATSimulator(witness), WITNESS_CURRENT)

        kick = float(witness[0].KickAngle[plane])
        assert kick == WITNESS_CURRENT / AMPS_PER_RADIAN_KICK
        assert kick != WITNESS_CURRENT * (1.0 / AMPS_PER_RADIAN_KICK)


def test_partition_is_the_pyat_coupled_one(bound, channel_by_address):
    """Every bound variable comes from the partition this model serves."""
    assert all(
        channel_by_address[address]["partition"] == PARTITION_PYAT_COUPLED for address in bound
    )
