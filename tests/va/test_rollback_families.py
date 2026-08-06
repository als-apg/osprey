"""Per-family write-rollback tests for the virtual accelerator's model layer.

``StrengthMap.apply`` has four formula families, and each writes into a
different piece of an element's storage: a corrector's ``KickAngle``, a
quadrupole's ``PolynomB[1]`` (reached through the ``K`` alias), a dipole's
``PolynomB[0]``, a sextupole's ``PolynomB[2]``. ``DECLARED_ATTRIBUTE``
collapses all four onto two declarations -- ``KickAngle`` and the *whole*
``PolynomB`` array -- and rollback snapshots exactly what is declared. So
the question each test here answers is the same one per family: after a
batch this family took part in is rejected, is the storage the write landed
in back to the bit it started at?

``tests/va/test_pyat_ring_model.py`` already pins that for ``QF``. These
tests extend it to every formula family, and add the index-level evidence
that makes the "declare the whole array" rule checkable: each test asserts
both that rollback restores the entire array and that the family's write,
once committed, moves the one index its formula names and leaves its
neighbours alone.

Everything is driven through ``PyATRingModel.set``. The element-level
rollback primitives are ``lume_pyat``'s own objects; calling them directly
would test that package rather than what this repo's adapter leaves behind
after a rejected write, which is the thing that can actually regress here.
"""

from __future__ import annotations

import numpy as np
import pytest

from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.lattice.strengths import (
    # Which KickAngle index each corrector family writes. Imported for the
    # same reason model/variables.py imports it: it is the fact
    # StrengthMap.apply writes through, and restating it here would be a
    # second copy to keep in step.
    _CORRECTOR_PLANE,
    CORRECTOR_FAMILIES,
    DIPOLE_FAMILY,
    QUADRUPOLE_FAMILIES,
    SEXTUPOLE_FAMILIES,
)
from osprey.services.virtual_accelerator.model import (
    PyATRingModel,
    build_variable_catalog,
)
from osprey.services.virtual_accelerator.model.variables import DECLARED_ATTRIBUTE

# The batch that loses the closed orbit: one quadrupole family at 1.2x its
# own nominal current. Measured against the real ring -- QF and QFA both
# push the one-turn trace past |2| at this factor, QD does not, so the two
# that do are the only ones used and the third is never a fallback.
DESTABILIZING_FACTOR = 1.2
DESTABILIZING_FAMILY = "QF"
DESTABILIZING_ALTERNATE = "QFA"

# A benign move: halfway from the device's nominal to its own upper limit.
# Taken from the band rather than as a fraction of the nominal because the
# two are not related -- the dipoles' band is tighter than one percent of
# their nominal, and correctors sit at a zero nominal no fraction moves off
# at all. Halfway to the limit is in band and a real move for every family.
BENIGN_BAND_FRACTION = 0.5


@pytest.fixture(scope="module")
def catalog():
    return build_variable_catalog()


@pytest.fixture(scope="module")
def setpoints_by_family(catalog) -> dict[str, list[str]]:
    """Every writable ``:SP`` address the catalog carries, grouped by family.

    Addresses come from the catalog rather than being formatted here: these
    tests write real channels, and a family whose devices the catalog stops
    exposing should surface as a missing key, not as a write to an address
    that no longer exists.
    """
    grouped: dict[str, list[str]] = {}
    for address, variable in catalog.items():
        if not variable.read_only:
            grouped.setdefault(address.split(":")[2], []).append(address)
    return {family: sorted(addresses) for family, addresses in grouped.items()}


@pytest.fixture(scope="module")
def bpm_readings(catalog) -> list[str]:
    """Every BPM POSITION output address, derived from the catalog."""
    return [address for address, variable in catalog.items() if variable.read_only]


@pytest.fixture
def model() -> PyATRingModel:
    """A fresh, unfaulted model. Function-scoped: these tests mutate it."""
    return PyATRingModel()


def declared_storage(model: PyATRingModel, address: str) -> np.ndarray:
    """A copy of the array a rejected batch has to put back for ``address``.

    Reaches into the lattice on purpose, through the model's own public ring
    accessors: a rolled-back write leaves no trace in the value API by
    design, so the element is the only place the evidence lives.
    """
    _ring, _system, family, device, _field, _subfield = address.split(":")
    element = model.lattice[model.element_index(f"{family}{device}")]
    return np.array(getattr(element, DECLARED_ATTRIBUTE[family]), copy=True)


def benign_value(model: PyATRingModel, catalog, address: str) -> float:
    """An in-band setpoint for ``address`` that is a real move off nominal."""
    nominal = model.get(address)
    low, high = catalog[address].value_range
    value = nominal + BENIGN_BAND_FRACTION * (high - nominal)
    assert low <= value <= high, f"{address} probe {value} outside its band ({low}, {high})"
    assert value != nominal, f"{address} probe is no move at all"
    return value


def destabilizing_batch(
    model: PyATRingModel, setpoints_by_family: dict[str, list[str]], avoid: str
) -> dict[str, float]:
    """A whole quadrupole family at 1.2x nominal -- never family ``avoid``.

    The fault has to be a family the device under test is *not* in. Sharing
    one would make the device's own write the destabilizing write, which is
    the QF-only case ``test_pyat_ring_model.py`` already covers; here the
    point is that an innocent bystander in the batch is restored too.
    """
    family = DESTABILIZING_ALTERNATE if avoid == DESTABILIZING_FAMILY else DESTABILIZING_FAMILY
    return {
        address: model.get(address) * DESTABILIZING_FACTOR
        for address in setpoints_by_family[family]
    }


def reject_then_commit(
    model: PyATRingModel,
    catalog,
    setpoints_by_family: dict[str, list[str]],
    bpm_readings: list[str],
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject a batch containing this family's first device, then commit it alone.

    Asserts the contract every family shares: bit-exact storage restoration,
    the retained input untouched, and the rejected solve leaving both the
    simulator's cached reading and the model's cached BPM outputs describing
    the pre-batch orbit. Returns the device's declared storage array
    before the rejected batch and after the same write is committed on its
    own. The committed write is what keeps the restoration assertion honest:
    an array that never moves is trivially restored, and this shows the
    value was a real mutation of that storage all along.
    """
    address = setpoints_by_family[family][0]
    value = benign_value(model, catalog, address)

    before_storage = declared_storage(model, address)
    before_input = model.get(address)
    before_solution = dict(model.simulator.last_solution)
    before_readings = model.get(bpm_readings)

    fault = destabilizing_batch(model, setpoints_by_family, avoid=family)
    assert address not in fault, "the device under test must be a bystander, not the fault"
    with pytest.raises(OrbitSolveError):
        model.set({address: value, **fault})

    assert np.array_equal(declared_storage(model, address), before_storage)
    assert model.get(address) == before_input
    assert dict(model.simulator.last_solution) == before_solution
    assert model.get(bpm_readings) == before_readings

    model.set({address: value})
    return before_storage, declared_storage(model, address)


@pytest.mark.parametrize("family", sorted(CORRECTOR_FAMILIES))
def test_corrector_rollback_restores_the_whole_kick_angle(
    model, catalog, setpoints_by_family, bpm_readings, family
):
    """``KickAngle[plane] = I / AMPS_PER_RADIAN_KICK``.

    The write is to one plane; the snapshot is the two-element sequence, so
    that is what has to come back.
    """
    before, after = reject_then_commit(model, catalog, setpoints_by_family, bpm_readings, family)
    plane = _CORRECTOR_PLANE[family]
    assert after[plane] != before[plane]
    assert after[1 - plane] == before[1 - plane]


@pytest.mark.parametrize("family", sorted(QUADRUPOLE_FAMILIES))
def test_quadrupole_rollback_restores_the_whole_polynom_b(
    model, catalog, setpoints_by_family, bpm_readings, family
):
    """``K = K_baked * I / I_nom``.

    ``K`` is an alias pyAT routes onto ``PolynomB[1]``, so the write lands in
    storage the variable never names -- which is exactly why the declared
    attribute is the array and not the alias.
    """
    before, after = reject_then_commit(model, catalog, setpoints_by_family, bpm_readings, family)
    assert after[1] != before[1]
    assert after[0] == before[0]


def test_dipole_rollback_restores_the_whole_polynom_b(
    model, catalog, setpoints_by_family, bpm_readings
):
    """``PolynomB[0] = (I / I_nom - 1) * BendingAngle / Length``.

    The trim-coil term is written in place into an array that already holds
    a baked combined-function gradient at index 1, and ``PolynomB[0]`` has
    no scalar alias at all. Both the written field error and the untouched
    baked gradient have to survive the rollback.
    """
    before, after = reject_then_commit(
        model, catalog, setpoints_by_family, bpm_readings, DIPOLE_FAMILY
    )
    assert after[0] != before[0]
    assert after[1] == before[1]


@pytest.mark.parametrize("family", sorted(SEXTUPOLE_FAMILIES))
def test_sextupole_rollback_restores_the_whole_polynom_b(
    model, catalog, setpoints_by_family, bpm_readings, family
):
    """``PolynomB[2] = h_baked * I / I_nom``.

    Written in place into the same array the dipole and quadrupole formulas
    write different indices of.
    """
    before, after = reject_then_commit(model, catalog, setpoints_by_family, bpm_readings, family)
    assert after[2] != before[2]
    assert after[0] == before[0]
    assert after[1] == before[1]


def test_every_declared_write_footprint_has_a_formula_family_here():
    """No family may be pyat-coupled without a rollback test above.

    ``DECLARED_ATTRIBUTE`` is the authoritative write-footprint map; adding a
    family to ``StrengthMap`` without covering it here would leave its
    rollback unpinned, and this is what says so.
    """
    covered = CORRECTOR_FAMILIES | QUADRUPOLE_FAMILIES | SEXTUPOLE_FAMILIES | {DIPOLE_FAMILY}
    assert covered == set(DECLARED_ATTRIBUTE)
