"""Six-dimensional behaviour of the served ring, and what a readback answers.

The other VA test modules pin seams: which file is read, which refusal names
which key, which class implements which kind. This one pins *physics through
those seams* -- the four statements the proposal's SC3 and SC8 make about a
ring served from an emitted tree:

* An **rf frequency step** moves the orbit where the ring is dispersive, and
  leaves the plane that carries no dispersion where it was. That only happens
  at all because the served ring solves a 6D closed orbit through its RF
  bucket; a 4D ring would take the frequency write and answer the same orbit.
* A **dipole setpoint step** moves the tunes, because the energy knob rescales
  every rigidity-scaled field it has adopted -- and the size of that move is
  fixed: the bound strength ends exactly ``K * brho(E_deck) / brho(E)``, so the
  field it lost is ``K * (1 - brho(E_deck)/brho(E))`` and the tune shift is the
  one *that* field change implies and no other.
* A **readback** is the facility's own reverse curve applied to the setpoint's
  physics value, so a family whose exported inverse is not the reciprocal of
  its calibration reads back a *different* number than was written, a family
  whose inverse is a sampled table reads back that table's value, and a family
  exported with no inverse at all reads back exactly what was written.
* A **sliced kick** is divided over its slices and read back whole: each of
  ``n`` slices carries ``1/n`` of the physics kick, and the reading is slice
  one multiplied by the slice count again.

The ring is synthetic -- a small stable FODO lattice with eight cells, one
cavity, dipoles that give it horizontal dispersion and no vertical, and unique
names on every bound element. It is not any facility's deck, and no address
here carries a facility's vocabulary: what pairs an address with an element,
a calibration and an inverse is the bindings document and only it. The
rigidity is restated from first principles at the bottom of this module rather
than imported, so the identity the energy write is held to is stated
independently of the code that implements it.

Two facility-specific claims of SC3 and SC8 cannot be made against a
synthetic tree -- that NSLS-II's rf couples through a cavity found *by class*
because its ``ATIndex`` came back empty, that its dipole is latched by rule on
a flat ``bend2gev``, and that its ``SQ`` family collapses to an identity
readback. Those are asserted against the re-exported 2.0 fixtures, and skip
until task 1.8 produces them (see :data:`_FIXTURE_ROOT`).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import at
import pytest

from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    SETPOINT_SUBFIELD,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

C_LIGHT = 299792458.0

#: The deck energy this fixture tree was exported at, in GeV.
DECK_ENERGY_GEV = 3.0

#: The focusing strength the ring is built with, and the cavity harmonic.
QUAD_K = 1.1
HARMONIC = 88

#: Hardware-to-physics gains, one per bound family. The first family is the
#: one the energy knob adopts, so its gain is the one the rigidity identity is
#: checked against.
SCALED_GAIN = 0.01
PLAIN_GAIN = 0.02
KICK_GAIN = 1.0e-6

#: Millimetres per metre: the monitor inverse a facility publishing
#: millimetres exports.
MM_PER_M = 1.0e3

#: How far the reverse curve of the differing-inverse family sits from the
#: reciprocal of its own calibration -- the two are sampled along independent
#: paths, so they agree only to the precision of the samples, and here they
#: disagree by five per cent.
INVERSE_MISMATCH = 0.95

#: The addresses the fixture document binds. Nothing relates them to the
#: element names below; the bindings document does that and only it.
SCALED_SP = "R1:PWR:MAG_A:07:CUR:SP"
SCALED_RB = "R1:PWR:MAG_A:07:CUR:RB"
DIFFERING_SP = "R1:PWR:MAG_B:07:CUR:SP"
DIFFERING_RB = "R1:PWR:MAG_B:07:CUR:RB"
TABLE_SP = "R1:PWR:MAG_C:07:CUR:SP"
TABLE_RB = "R1:PWR:MAG_C:07:CUR:RB"
IDENTITY_SP = "R1:PWR:MAG_D:07:CUR:SP"
IDENTITY_RB = "R1:PWR:MAG_D:07:CUR:RB"
KICK_SP = "R1:PWR:COR_A:03:CUR:SP"
KICK_RB = "R1:PWR:COR_A:03:CUR:RB"
CAVITY_SP = "R1:RF:CAV_A:01:FREQ:SP"
BPM_X = "R1:DIA:MON_A:12:POS:X"
BPM_Y = "R1:DIA:MON_A:12:POS:Y"
BEND_SP = "R1:PWR:BND_A:01:CUR:SP"
BEND_RB = "R1:PWR:BND_A:01:CUR:RB"

# The deck's own names for the elements each address drives.
SCALED_ELEMENT = "QF1"
DIFFERING_ELEMENT = "QD1"
TABLE_ELEMENT = "QF2"
IDENTITY_ELEMENT = "QD2"
KICK_ELEMENTS = ("HC1", "HC2", "HC3")
MONITOR_ELEMENT = "BPM1"
CAVITY_ELEMENT = "RFC"

#: The bend setpoints the energy tests write. The large one is a ten per cent
#: move, big enough that the tune shift is unmistakable; the small one is a
#: tenth of a per cent, where the ring's own 6D response to the energy move is
#: three orders of magnitude below the shift the rescaled field produces and
#: the two can be compared to 1e-6.
BEND_NOMINAL = 300.0
BEND_LARGE_STEP = 330.0
BEND_SMALL_STEP = 300.3

#: How far the rf frequency is stepped, as a fraction of the nominal. Ten
#: parts per million moves this ring's horizontal orbit by some 80 microns.
RF_STEP = 1.0e-5


# -- the synthetic emitted tree ----------------------------------------------


def _ring(cells: int = 8, kq: float = QUAD_K) -> at.Lattice:
    """A small stable ring with one cavity and uniquely named magnets.

    Eight FODO cells, each carrying a monitor and a corrector, closed by
    sixteen dipoles -- so the ring bends, and a monitor sitting after a dipole
    reads a horizontal dispersion of about a metre and a half while its
    vertical dispersion is zero. Returned 4D, the way a ring saved out of a
    facility's simulator model arrives; the served tree's own loader is what
    switches longitudinal motion on.
    """
    angle = 2 * math.pi / (2 * cells)
    elements: list[Any] = []
    for cell in range(1, cells + 1):
        elements += [
            at.Quadrupole(f"QF{cell}", 0.3, kq),
            at.Drift("DR", 1.0),
            at.Dipole(f"BD{cell}A", 1.0, angle),
            at.Monitor(f"BPM{cell}"),
            at.Drift("DR", 1.0),
            at.Quadrupole(f"QD{cell}", 0.3, -kq),
            at.Drift("DR", 1.0),
            at.Corrector(f"HC{cell}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BD{cell}B", 1.0, angle),
            at.Drift("DR", 1.0),
        ]
    ring = at.Lattice(elements, name="fixture", energy=DECK_ENERGY_GEV * 1.0e9, periodicity=1)
    frequency = HARMONIC * C_LIGHT / ring.circumference
    ring.append(at.RFCavity(CAVITY_ELEMENT, 0.0, 1.0e6, frequency, HARMONIC, ring.energy))
    ring.disable_6d()
    return ring


def _served_ring(data_dir: Path) -> at.Lattice:
    """The ring in the state a served tree boots it in, for reference optics.

    The tree's own lattice file, loaded and switched to 6D the way
    ``build_ring`` does it, but with no model and no bindings over it -- so the
    tunes it is compared against come from pyAT alone, off the same bytes the
    model was served.
    """
    ring = at.load_lattice(ManifestPaths(data_root=data_dir).lattice_json)
    ring.enable_6d(at.RFCavity)
    return ring


#: The cavity frequency the fixture ring is built at, in MHz -- the hardware
#: unit the rf calibration below states.
CAVITY_NOMINAL_MHZ = HARMONIC * C_LIGHT / _ring().circumference / 1.0e6

#: The nominal hardware setpoint of each bound magnet: the current that puts
#: the element at the strength the ring was built with.
SCALED_NOMINAL = QUAD_K / SCALED_GAIN
DIFFERING_NOMINAL = -QUAD_K / PLAIN_GAIN
TABLE_NOMINAL = QUAD_K / PLAIN_GAIN
IDENTITY_NOMINAL = -QUAD_K / PLAIN_GAIN

#: The table inverse: physics strength to hardware current, sampled at three
#: points and deliberately not a straight line, so reading it back is visibly
#: neither the written value nor any single gain times it.
TABLE_INVERSE_GRID = (0.0, 1.0, 2.0)
TABLE_INVERSE_VALUES = (0.0, 50.0, 96.0)


def _table_inverse(physics: float) -> float:
    """The table inverse, restated: piecewise linear through its own points."""
    for left in range(len(TABLE_INVERSE_GRID) - 1):
        lower, upper = TABLE_INVERSE_GRID[left], TABLE_INVERSE_GRID[left + 1]
        if lower <= physics <= upper:
            slope = (TABLE_INVERSE_VALUES[left + 1] - TABLE_INVERSE_VALUES[left]) / (upper - lower)
            return TABLE_INVERSE_VALUES[left] + slope * (physics - lower)
    raise AssertionError(f"physics value {physics} is outside the sampled grid")


def _linear(gain: float, offset: float = 0.0) -> dict:
    return {"kind": "linear", "gain": gain, "offset": offset}


def _table(grid: tuple[float, ...], values: tuple[float, ...]) -> dict:
    return {"kind": "table", "grid": list(grid), "values": list(values)}


def _strength(**overrides: Any) -> dict:
    """A quadrupole setpoint, onto one element's ``PolynomB[1]``.

    The default is the family the energy knob adopts: its physics value moves
    with the beam rigidity, and its exported inverse is the exact reciprocal
    of its calibration, so its readback is the value that was written and
    nothing in the energy tests is confused by a readback mismatch.
    """
    body = {
        "kind": "strength",
        "family": "mag_a",
        "setpoint_address": SCALED_SP,
        "readback_address": SCALED_RB,
        "readback": "inverse",
        "element": SCALED_ELEMENT,
        "attribute": "PolynomB",
        "index": 1,
        "slices": [{"element": SCALED_ELEMENT, "weight": 1.0}],
        "owner": "mag_a",
        "calibration": _linear(SCALED_GAIN),
        "monitor_inverse": _linear(1.0 / SCALED_GAIN),
        "nominal": SCALED_NOMINAL,
        "energy_scaling": "brho",
        "energy_table": None,
    }
    body.update(overrides)
    return body


def _differing_inverse() -> dict:
    """A family whose reverse curve is not the reciprocal of its calibration.

    The control system sampled hardware-to-physics and physics-to-hardware
    along two independent paths, and here they disagree by five per cent. Its
    physics value is one the control system does not rescale with the energy,
    so the energy knob leaves it alone.
    """
    return _strength(
        family="mag_b",
        setpoint_address=DIFFERING_SP,
        readback_address=DIFFERING_RB,
        element=DIFFERING_ELEMENT,
        slices=[{"element": DIFFERING_ELEMENT, "weight": 1.0}],
        owner="mag_b",
        calibration=_linear(PLAIN_GAIN),
        monitor_inverse=_linear(INVERSE_MISMATCH / PLAIN_GAIN),
        nominal=DIFFERING_NOMINAL,
        energy_scaling="none",
    )


def _table_inverse_family() -> dict:
    """A family whose reverse curve is a sampled table, not a line."""
    return _strength(
        family="mag_c",
        setpoint_address=TABLE_SP,
        readback_address=TABLE_RB,
        element=TABLE_ELEMENT,
        slices=[{"element": TABLE_ELEMENT, "weight": 1.0}],
        owner="mag_c",
        calibration=_linear(PLAIN_GAIN),
        monitor_inverse=_table(TABLE_INVERSE_GRID, TABLE_INVERSE_VALUES),
        nominal=TABLE_NOMINAL,
        energy_scaling="none",
    )


def _identity_family() -> dict:
    """A family exported with no inverse at all: an identity readback.

    What the emit lane writes when the facility's reverse path gives back the
    value that was written -- the readback is the setpoint, and there is no
    curve to apply.
    """
    return _strength(
        family="mag_d",
        setpoint_address=IDENTITY_SP,
        readback_address=IDENTITY_RB,
        readback="identity",
        element=IDENTITY_ELEMENT,
        slices=[{"element": IDENTITY_ELEMENT, "weight": 1.0}],
        owner="mag_d",
        calibration=_linear(PLAIN_GAIN),
        monitor_inverse=None,
        nominal=IDENTITY_NOMINAL,
        energy_scaling="none",
    )


def _kick() -> dict:
    """A corrector setpoint shared equally over three lattice pieces."""
    return _strength(
        kind="kick",
        family="cor_a",
        setpoint_address=KICK_SP,
        readback_address=KICK_RB,
        element=KICK_ELEMENTS[0],
        attribute="KickAngle",
        index=0,
        slices=[{"element": name, "weight": 1.0 / len(KICK_ELEMENTS)} for name in KICK_ELEMENTS],
        owner="cor_a",
        calibration=_linear(KICK_GAIN),
        monitor_inverse=_linear(1.0 / KICK_GAIN),
        nominal=0.0,
    )


def _monitor(address: str, axis: str, family: str) -> dict:
    """One orbit reading: metres on the ring, millimetres on the wire."""
    return _strength(
        kind="monitor",
        family=family,
        setpoint_address=address,
        readback_address=None,
        element=MONITOR_ELEMENT,
        attribute=axis,
        index=None,
        slices=[{"element": MONITOR_ELEMENT, "weight": 1.0}],
        owner="mon_a",
        calibration=_linear(1.0 / MM_PER_M),
        monitor_inverse=_linear(MM_PER_M),
        nominal=None,
        energy_scaling="none",
    )


def _rf() -> dict:
    """The cavity frequency, written in MHz."""
    return _strength(
        kind="rf",
        family="cav_a",
        setpoint_address=CAVITY_SP,
        readback_address=None,
        readback="same_as_setpoint",
        element=CAVITY_ELEMENT,
        attribute="Frequency",
        index=None,
        slices=[{"element": CAVITY_ELEMENT, "weight": 1.0}],
        owner="cav_a",
        calibration=_linear(1.0e6),
        monitor_inverse=None,
        nominal=CAVITY_NOMINAL_MHZ,
        energy_scaling="none",
    )


def _energy() -> dict:
    """The ring's energy knob: the one binding that drives no element."""
    return _strength(
        kind="energy",
        family="bnd_a",
        setpoint_address=BEND_SP,
        readback_address=BEND_RB,
        readback="identity",
        element=None,
        attribute=None,
        index=None,
        slices=[],
        owner=None,
        calibration=None,
        monitor_inverse=None,
        nominal=BEND_NOMINAL,
        energy_scaling="none",
        energy_table=_table((270.0, 300.0, 330.0), (2.7, DECK_ENERGY_GEV, 3.3)),
    )


def _bindings() -> list[dict]:
    """Every binding the fixture tree carries, in document order."""
    return [
        _strength(),
        _differing_inverse(),
        _table_inverse_family(),
        _identity_family(),
        _kick(),
        _monitor(BPM_X, "x", "mon_x"),
        _monitor(BPM_Y, "y", "mon_y"),
        _rf(),
        _energy(),
    ]


def _channel(address: str, **overrides: Any) -> dict:
    """One manifest channel, in the per-channel schema the manifest carries."""
    ring, system, family, device, field, subfield = address.split(":")
    channel = {
        "address": address,
        "ring": ring,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": PARTITION_PYAT_COUPLED,
        "record_type": "ai",
        "noise": False,
    }
    channel.update(overrides)
    return channel


def _manifest() -> list[dict]:
    """The channel list a deployment of this tree resolved."""
    return [
        _channel(address)
        for address in (
            SCALED_SP,
            SCALED_RB,
            DIFFERING_SP,
            TABLE_SP,
            IDENTITY_SP,
            KICK_SP,
            BPM_X,
            BPM_Y,
            CAVITY_SP,
            BEND_SP,
        )
    ]


def _machine() -> dict:
    """Every nominal and unit the served ``machine.json`` declares."""
    return {
        SCALED_SP: {"value": SCALED_NOMINAL, "units": "A"},
        DIFFERING_SP: {"value": DIFFERING_NOMINAL, "units": "A"},
        TABLE_SP: {"value": TABLE_NOMINAL, "units": "A"},
        IDENTITY_SP: {"value": IDENTITY_NOMINAL, "units": "A"},
        KICK_SP: {"value": 0.0, "units": "A"},
        BPM_X: {"value": 0.0, "units": "mm"},
        BPM_Y: {"value": 0.0, "units": "mm"},
        CAVITY_SP: {"value": CAVITY_NOMINAL_MHZ, "units": "MHz"},
        BEND_SP: {"value": BEND_NOMINAL, "units": "A"},
    }


def _limits() -> dict:
    """The write bands the served ``channel_limits.json`` ships."""
    magnet = {"min_value": -200.0, "max_value": 200.0}
    return {
        "_version": "1.0",
        "defaults": {"writable": True, "confirm": True},
        SCALED_SP: dict(magnet),
        DIFFERING_SP: dict(magnet),
        TABLE_SP: dict(magnet),
        IDENTITY_SP: dict(magnet),
        KICK_SP: {"min_value": -10.0, "max_value": 10.0},
        CAVITY_SP: {"min_value": 400.0, "max_value": 600.0},
        BEND_SP: {"min_value": 250.0, "max_value": 350.0},
    }


def _tree(root: Path) -> Path:
    """Write a served tree and return the data directory addressing it."""
    paths = ManifestPaths(data_root=root)
    paths.lattice_json.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(_ring(), paths.lattice_json)
    document = {
        "system": "StorageRing",
        "energy_gev": DECK_ENERGY_GEV,
        "lattice_sha256": hashlib.sha256(paths.lattice_json.read_bytes()).hexdigest(),
        "bindings": _bindings(),
    }
    paths.va_bindings.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    paths.machine_json.write_text(json.dumps({"name": "fixture", "channels": _machine()}))
    paths.channel_limits.write_text(json.dumps(_limits()))
    return root


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A served tree whose files all agree -- the boot case."""
    return _tree(tmp_path_factory.mktemp("served") / "data")


@pytest.fixture(scope="module")
def booted(data_dir: Path) -> PyATRingModel:
    """One boot, shared by every test that only reads the model."""
    return PyATRingModel(data_dir, _manifest())


@pytest.fixture
def model(data_dir: Path) -> PyATRingModel:
    """A fresh model, for the tests that write to it."""
    return PyATRingModel(data_dir, _manifest())


# -- the rigidity, restated ---------------------------------------------------

#: Electron rest mass in GeV, and tesla-metres per GeV/c of momentum. Stated
#: here so the identity below is an independent statement of the physics
#: rather than a re-use of the conversion under test.
REST_MASS_GEV = 0.510998950e-3


def _rigidity(energy_gev: float) -> float:
    """Beam rigidity in tesla-metres at a kinetic energy of ``energy_gev``."""
    total = energy_gev + REST_MASS_GEV
    return math.sqrt(total * total - REST_MASS_GEV * REST_MASS_GEV) * 1.0e9 / C_LIGHT


def _element(model: PyATRingModel, name: str) -> Any:
    """The ring element a bound name addresses, the way a binding does."""
    return model.lattice[model.element_index(name)]


def _named(ring: at.Lattice, name: str) -> Any:
    """The one element of a bare ring carrying ``name``."""
    found = [element for element in ring if element.FamName == name]
    assert len(found) == 1, f"{name} names {len(found)} elements of this ring"
    return found[0]


def _transverse_tunes(ring: at.Lattice) -> tuple[float, float]:
    """The ring's horizontal and vertical tunes."""
    tunes = ring.get_tune()
    return float(tunes[0]), float(tunes[1])


# -- SC3: the rf step, through a 6D closed orbit ------------------------------


class TestAnRFStepMovesTheDispersiveOrbit:
    """The frequency reaches the orbit, which is the whole point of the 6D solve."""

    def test_the_served_ring_solves_a_six_dimensional_orbit(self, booted: PyATRingModel) -> None:
        """Everything in this class rests on it: a 4D ring would take the
        frequency write and answer the same orbit forever."""
        assert booted.lattice.is_6d

    def test_an_unperturbed_ring_sits_on_axis_in_both_planes(self, booted: PyATRingModel) -> None:
        assert booted.get(BPM_X) == pytest.approx(0.0, abs=1e-9)
        assert booted.get(BPM_Y) == pytest.approx(0.0, abs=1e-9)

    def test_a_frequency_step_moves_the_reading_where_the_ring_is_dispersive(
        self, model: PyATRingModel
    ) -> None:
        """Ten parts per million of frequency is tens of microns of orbit at a
        monitor whose dispersion is of order a metre: the frequency sets the
        momentum the closed orbit is found at, and dispersion turns that into
        a position."""
        model.set({CAVITY_SP: CAVITY_NOMINAL_MHZ * (1.0 + RF_STEP)})

        assert abs(model.get(BPM_X)) > 1.0e-2, "the horizontal reading moved, in millimetres"

    def test_the_plane_that_carries_no_dispersion_does_not_move(self, model: PyATRingModel) -> None:
        """A flat ring has no vertical dispersion, so the same write that moves
        the horizontal reading leaves the vertical one where it was. The
        reading follows the optics rather than the write."""
        model.set({CAVITY_SP: CAVITY_NOMINAL_MHZ * (1.0 + RF_STEP)})

        assert model.get(BPM_Y) == pytest.approx(0.0, abs=1e-9)

    def test_reversing_the_step_reverses_the_shift(self, data_dir: Path) -> None:
        """Equal and opposite frequency errors put the orbit equally far either
        side of the axis -- the linear response a momentum offset has, and a
        stronger statement than any single reading's magnitude."""
        up = PyATRingModel(data_dir, _manifest())
        up.set({CAVITY_SP: CAVITY_NOMINAL_MHZ * (1.0 + RF_STEP)})
        down = PyATRingModel(data_dir, _manifest())
        down.set({CAVITY_SP: CAVITY_NOMINAL_MHZ * (1.0 - RF_STEP)})

        assert down.get(BPM_X) == pytest.approx(-up.get(BPM_X), rel=1e-3)

    def test_writing_the_nominal_frequency_back_restores_the_orbit(
        self, model: PyATRingModel
    ) -> None:
        """The frequency is state on the lattice, not an increment: the ring
        ends where its own nominal puts it however it got there."""
        model.set({CAVITY_SP: CAVITY_NOMINAL_MHZ * (1.0 + RF_STEP)})
        model.set({CAVITY_SP: CAVITY_NOMINAL_MHZ})

        assert model.get(BPM_X) == pytest.approx(0.0, abs=1e-9)


# -- SC3: the dipole step, through the energy knob ----------------------------


class TestADipoleStepMovesTheTunes:
    """An energy write is a strength write to every family that follows brho."""

    def test_a_dipole_setpoint_step_changes_both_transverse_tunes(
        self, model: PyATRingModel
    ) -> None:
        before = _transverse_tunes(model.lattice)

        model.set({BEND_SP: BEND_LARGE_STEP})

        after = _transverse_tunes(model.lattice)
        assert abs(after[0] - before[0]) > 1.0e-3, "the horizontal tune moved"
        assert abs(after[1] - before[1]) > 1.0e-3, "the vertical tune moved"

    def test_the_energy_write_leaves_the_adopted_field_at_the_rigidity_identity(
        self, model: PyATRingModel
    ) -> None:
        """``K(E) = K * brho(E_deck)/brho(E)``, so the field the write took
        away is ``K * (1 - brho(E_deck)/brho(E))``.

        The rigidity ratio is taken from the energy the ring reports rather
        than the energy the table implies: pyAT re-derives the number it is
        handed by an ulp, and the rescale reads the ring back.
        """
        before = float(_element(model, SCALED_ELEMENT).PolynomB[1])

        model.set({BEND_SP: BEND_LARGE_STEP})

        ratio = _rigidity(DECK_ENERGY_GEV) / _rigidity(float(model.lattice.energy) / 1.0e9)
        after = float(_element(model, SCALED_ELEMENT).PolynomB[1])
        assert before - after == pytest.approx(before * (1.0 - ratio), rel=1e-6)

    def test_the_tune_shift_is_the_one_that_identity_implies(
        self, model: PyATRingModel, data_dir: Path
    ) -> None:
        """And nothing else moves the tunes: a ring left at the deck energy
        whose bound quadrupole alone gives up ``K * (1 - brho(E_deck)/brho(E))``
        answers the same tunes as the ring the energy write left behind.

        The step is a tenth of a per cent, where the residual -- the 6D closed
        orbit's own response to being asked for a different energy through
        unchanged dipoles -- is some 2e-7, against a tune shift of 2e-4. The
        residual grows with the step: at ten per cent it is 2e-5, and this
        equality holds only to four digits.
        """
        reference = _served_ring(data_dir)
        assert _transverse_tunes(reference) == pytest.approx(
            _transverse_tunes(model.lattice), abs=1e-12
        ), "the reference ring is the one the tree boots"
        strength = float(_element(model, SCALED_ELEMENT).PolynomB[1])

        model.set({BEND_SP: BEND_SMALL_STEP})

        ratio = _rigidity(DECK_ENERGY_GEV) / _rigidity(float(model.lattice.energy) / 1.0e9)
        _named(reference, SCALED_ELEMENT).PolynomB[1] = strength * ratio
        assert _transverse_tunes(model.lattice) == pytest.approx(
            _transverse_tunes(reference), abs=1e-6
        )

    def test_a_family_the_control_system_does_not_rescale_stays_put(
        self, model: PyATRingModel
    ) -> None:
        """``energy_scaling: none`` is the statement that the facility's own
        conversion leaves this family's physics value alone, so the energy
        knob must not touch it."""
        before = float(_element(model, DIFFERING_ELEMENT).PolynomB[1])

        model.set({BEND_SP: BEND_LARGE_STEP})

        assert float(_element(model, DIFFERING_ELEMENT).PolynomB[1]) == before

    def test_the_two_writes_compose_in_either_order(self, data_dir: Path) -> None:
        """A client writes the energy and a setpoint independently and nothing
        orders them, so the ring has to end in the same state either way: the
        setpoint write applies the factor for the energy the ring is at, and
        the energy write rescales what is already on the lattice."""
        energy_first = PyATRingModel(data_dir, _manifest())
        energy_first.set({BEND_SP: BEND_LARGE_STEP})
        energy_first.set({SCALED_SP: 120.0})

        setpoint_first = PyATRingModel(data_dir, _manifest())
        setpoint_first.set({SCALED_SP: 120.0})
        setpoint_first.set({BEND_SP: BEND_LARGE_STEP})

        assert float(_element(energy_first, SCALED_ELEMENT).PolynomB[1]) == pytest.approx(
            float(_element(setpoint_first, SCALED_ELEMENT).PolynomB[1]), rel=1e-12
        )


# -- SC8: what a readback answers ---------------------------------------------


class TestTheReadbackIsTheExportedInverse:
    """A readback is ``monitor_inverse(calibration(written))``, never an
    inversion of the calibration and never an echo unless the export says so."""

    def test_a_differing_linear_inverse_reads_back_a_different_number(
        self, booted: PyATRingModel
    ) -> None:
        """The two curves were sampled independently, so the round trip is not
        the identity -- and a readback that happened to echo the setpoint would
        hide exactly that."""
        variable = booted.supported_variables[DIFFERING_SP]

        reading = variable.readback(-40.0)

        assert reading == pytest.approx(INVERSE_MISMATCH * -40.0)
        assert reading != pytest.approx(-40.0)

    def test_the_differing_inverse_reads_the_same_off_the_lattice(
        self, model: PyATRingModel
    ) -> None:
        """The readback is not bookkeeping beside the write: reading the bound
        element back through the same inverse gives the same number."""
        variable = model.supported_variables[DIFFERING_SP]
        setpoint = 0.98 * DIFFERING_NOMINAL

        model.set({DIFFERING_SP: setpoint})

        assert variable._get(model.simulator) == pytest.approx(variable.readback(setpoint))

    def test_a_table_inverse_reads_back_the_curve_at_the_physics_value(
        self, booted: PyATRingModel
    ) -> None:
        """A sampled reverse curve is read at the setpoint's physics value, so
        two setpoints on different segments of the table read back with
        different slopes."""
        variable = booted.supported_variables[TABLE_SP]

        for setpoint in (TABLE_NOMINAL, 90.0):
            physics = PLAIN_GAIN * setpoint
            assert variable.readback(setpoint) == pytest.approx(_table_inverse(physics))

    def test_the_table_inverse_is_not_a_single_gain(self, booted: PyATRingModel) -> None:
        """Otherwise the test above would pass against a line, and a table
        export would be indistinguishable from a linear one."""
        variable = booted.supported_variables[TABLE_SP]

        assert variable.readback(2 * TABLE_NOMINAL) != pytest.approx(
            2 * variable.readback(TABLE_NOMINAL)
        )

    def test_a_family_with_no_exported_inverse_reads_back_what_was_written(
        self, booted: PyATRingModel
    ) -> None:
        """An identity readback: the facility's reverse path gives back the
        value that was written, so the emit lane exported no curve and there is
        nothing to apply."""
        variable = booted.supported_variables[IDENTITY_SP]

        assert variable.readback(-37.5) == -37.5

    def test_an_identity_family_has_no_path_from_physics_back_to_hardware(
        self, booted: PyATRingModel
    ) -> None:
        """And it does not fabricate one by inverting the calibration: asked
        for the hardware value the lattice is holding, it says there is no
        exported curve to answer with."""
        variable = booted.supported_variables[IDENTITY_SP]

        with pytest.raises(NotImplementedError, match="monitor_inverse"):
            variable._get(booted.simulator)

    def test_a_calibrated_family_with_an_exact_inverse_round_trips(
        self, model: PyATRingModel
    ) -> None:
        """The control case for the three above: where the facility's two
        samplings do agree, the readback is the setpoint."""
        variable = model.supported_variables[SCALED_SP]

        assert variable.readback(95.0) == pytest.approx(95.0)


# -- SC8: the sliced kick -----------------------------------------------------


class TestASlicedKick:
    """A kick is divisible: ``n`` pieces bend the beam by the sum of what they do."""

    def test_each_slice_carries_one_nth_of_the_physics_kick(self, model: PyATRingModel) -> None:
        elements = {element.FamName: element for element in model.lattice}

        model.set({KICK_SP: 3.0})

        share = 3.0 * KICK_GAIN / len(KICK_ELEMENTS)
        for name in KICK_ELEMENTS:
            assert elements[name].KickAngle[0] == pytest.approx(share)

    def test_the_reading_is_slice_one_times_the_slice_count(self, model: PyATRingModel) -> None:
        """Which is the whole kick again, and the value the control system
        reads: the first slice is the one the binding reads back, and its
        weight is what divides it."""
        variable = model.supported_variables[KICK_SP]

        model.set({KICK_SP: 3.0})

        slice_one = float(_element(model, KICK_ELEMENTS[0]).KickAngle[0])
        assert slice_one * len(KICK_ELEMENTS) == pytest.approx(3.0 * KICK_GAIN)
        assert variable._get(model.simulator) == pytest.approx(3.0)

    def test_the_kick_moves_the_orbit_it_is_summed_over(self, model: PyATRingModel) -> None:
        """The slices are not bookkeeping either: three pieces at a third each
        bend the beam, so the monitor moves.

        Three amps is three microradians at this fixture's kick calibration --
        deliberately small, so the corrector stays in the small-signal regime
        -- which is a sub-micron orbit shift here. The reading is in
        millimetres, so the bar is set below that and three orders of
        magnitude above the solve's own numerical floor.
        """
        model.set({KICK_SP: 3.0})

        assert abs(model.get(BPM_X)) > 1.0e-4, "the horizontal reading moved, in millimetres"


# -- SC3 and SC8 on the re-exported facility trees ----------------------------

#: Where the re-exported 2.0 fixtures live. A served tree is any directory
#: under a facility's fixture folder that carries a ``va_bindings.json`` where
#: :class:`ManifestPaths` resolves one, so these tests bind to whatever layout
#: the re-export and the emit lane settle on rather than to a path guessed
#: here.
_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "mml"


def _served_tree(facility: str) -> Path | None:
    """The data root of a served tree under ``facility``'s fixtures, if any.

    Only the four files this model reads are required, not everything a
    deployment's ``data/`` carries: a fixture tree is emitted for the VA lane
    and has no channel databases of its own.
    """
    folder = _FIXTURE_ROOT / facility
    if not folder.is_dir():
        return None
    for bindings in sorted(folder.rglob("simulation/va_bindings.json")):
        paths = ManifestPaths(data_root=bindings.parent.parent)
        served = (paths.lattice_json, paths.va_bindings, paths.machine_json, paths.channel_limits)
        if all(path.is_file() for path in served):
            return paths.data_root
    return None


def _skip_reason(facility: str) -> str:
    """Why a facility's tests are skipped, naming the task that lands them."""
    return (
        f"task 1.8 (reexport-real-fixtures) has not produced a served 2.0 tree for "
        f"{facility}: no data root under {_FIXTURE_ROOT / facility} carries "
        f"simulation/va_bindings.json with the rest of its required sources"
    )


SPEAR3_TREE = _served_tree("spear3")
NSLS2_TREE = _served_tree("nsls2")


def _facility_channels(data_dir: Path) -> list[dict]:
    """A channel list for a served tree, derived from the tree's own bindings.

    An emitted tree carries no channel manifest -- a deployment's channel set
    comes from the facility's databases at build time -- so the addresses the
    document binds are the namespace to serve here, and nothing parses the
    address text: what says a channel is written is the binding's kind.
    """
    document = load_bindings(ManifestPaths(data_root=data_dir).va_bindings)
    return [
        {
            "address": binding.setpoint_address,
            "subfield": SETPOINT_SUBFIELD if binding.is_writable else "READ",
            "partition": PARTITION_PYAT_COUPLED,
        }
        for binding in document.bindings
    ]


def _facility_model(data_dir: Path) -> PyATRingModel:
    return PyATRingModel(data_dir, _facility_channels(data_dir))


def _one_kind(data_dir: Path, kind: str) -> list:
    document = load_bindings(ManifestPaths(data_root=data_dir).va_bindings)
    return [binding for binding in document.bindings if binding.kind == kind]


@pytest.mark.skipif(SPEAR3_TREE is None, reason=_skip_reason("spear3"))
class TestTheReExportedMeasuredTree:
    """The facility whose bend is a control knob and whose rf is coupled."""

    def test_an_rf_frequency_step_changes_a_monitor_reading(self) -> None:
        assert SPEAR3_TREE is not None
        model = _facility_model(SPEAR3_TREE)
        (frequency,) = _one_kind(SPEAR3_TREE, "rf")
        monitors = [binding.setpoint_address for binding in _one_kind(SPEAR3_TREE, "monitor")]
        before = {address: model.get(address) for address in monitors}

        model.set({frequency.setpoint_address: frequency.nominal * (1.0 + RF_STEP)})

        moved = [
            address for address in monitors if abs(model.get(address) - before[address]) > 1.0e-3
        ]
        assert moved, "no monitor reading moved: the ring is dispersive somewhere"

    def test_a_dipole_setpoint_step_changes_the_tunes(self) -> None:
        assert SPEAR3_TREE is not None
        model = _facility_model(SPEAR3_TREE)
        (knob,) = _one_kind(SPEAR3_TREE, "energy")
        before = _transverse_tunes(model.lattice)

        model.set({knob.setpoint_address: knob.nominal * 1.02})

        after = _transverse_tunes(model.lattice)
        assert abs(after[0] - before[0]) > 1.0e-4
        assert abs(after[1] - before[1]) > 1.0e-4

    def test_the_energy_write_leaves_every_adopted_field_at_the_identity(self) -> None:
        """The same rigidity identity as the synthetic tree's, on a real deck:
        every field the knob adopted is worth ``brho(E_deck)/brho(E)`` of what
        it held."""
        assert SPEAR3_TREE is not None
        model = _facility_model(SPEAR3_TREE)
        (knob,) = _one_kind(SPEAR3_TREE, "energy")
        document = load_bindings(ManifestPaths(data_root=SPEAR3_TREE).va_bindings)
        scaled = [
            binding
            for binding in document.bindings
            if binding.energy_scaling == "brho" and binding.slices
        ]
        before = {
            slice_.element: float(_held(model, slice_.element, binding))
            for binding in scaled
            for slice_ in binding.slices
        }

        model.set({knob.setpoint_address: knob.nominal * 1.02})

        ratio = _rigidity(document.energy_gev) / _rigidity(float(model.lattice.energy) / 1.0e9)
        for binding in scaled:
            for slice_ in binding.slices:
                held = float(_held(model, slice_.element, binding))
                assert held == pytest.approx(before[slice_.element] * ratio, rel=1e-6)


@pytest.mark.skipif(NSLS2_TREE is None, reason=_skip_reason("nsls2"))
class TestTheReExportedModelDerivedTree:
    """The facility whose cavity came back with an empty ``ATIndex`` and whose
    ``bend2gev`` is flat, so the bend is latched rather than coupled."""

    def test_the_rf_binding_names_an_element_the_ring_carries_as_a_cavity(self) -> None:
        """The export found no ``ATIndex`` for this family and coupled it by
        class instead, so what the binding names has to be a cavity of the
        served ring -- not merely a name the ring happens to carry."""
        assert NSLS2_TREE is not None
        model = _facility_model(NSLS2_TREE)
        (frequency,) = _one_kind(NSLS2_TREE, "rf")

        for slice_ in frequency.slices:
            assert isinstance(_element(model, slice_.element), at.RFCavity)

    def test_an_rf_frequency_step_changes_a_monitor_reading(self) -> None:
        assert NSLS2_TREE is not None
        model = _facility_model(NSLS2_TREE)
        (frequency,) = _one_kind(NSLS2_TREE, "rf")
        monitors = [binding.setpoint_address for binding in _one_kind(NSLS2_TREE, "monitor")]
        before = {address: model.get(address) for address in monitors}

        model.set({frequency.setpoint_address: frequency.nominal * (1.0 + RF_STEP)})

        moved = [
            address for address in monitors if abs(model.get(address) - before[address]) > 1.0e-3
        ]
        assert moved, "no monitor reading moved: the ring is dispersive somewhere"

    def test_the_dipole_is_latched_rather_than_bound_as_the_energy_knob(self) -> None:
        """A flat ``bend2gev`` says nothing about what current means what
        energy, so the family is latched by rule and the document carries no
        energy binding at all -- which is a facility whose bends are not a
        control channel, not a defect."""
        assert NSLS2_TREE is not None

        assert _one_kind(NSLS2_TREE, "energy") == []

    def test_the_skew_quadrupole_family_collapses_to_an_identity_readback(self) -> None:
        """Its exported inverse was the reciprocal of its own calibration, so
        the emit lane wrote no curve: the readback is the setpoint."""
        assert NSLS2_TREE is not None
        document = load_bindings(ManifestPaths(data_root=NSLS2_TREE).va_bindings)

        skew = [binding for binding in document.bindings if binding.family.upper() == "SQ"]

        assert skew, "the re-exported tree binds the skew quadrupole family"
        for binding in skew:
            assert binding.readback == "identity"
            assert binding.monitor_inverse is None


def _held(model: PyATRingModel, element: str, binding: Any) -> float:
    """What one element holds in the field a binding writes."""
    value = getattr(_element(model, element), str(binding.attribute))
    return value if binding.index is None else value[binding.index]
