"""Check A in hardware units: the model file's hardware answers, through the build's calibrations.

The Middle Layer answers every model question a second time in hardware units,
converting through its own calibrations. ``test_check_a`` holds the physics
answers to the engine plug-in on the deck the build wrote; these tests hold
each hardware answer to its physics value taken through the calibration the
build's wiring states for the device it reads or drives. The orbit response,
the closed orbit, the dispersion and the chromaticity find every device
through the model's wiring, and a listed device the wiring does not carry
fails naming its family and ``DeviceList`` row.

* **The orbit response in mm/A.** ``measbpmresp('Model', 'Hardware')`` is the
  physics matrix taken through ``physics2hw``: each row times its monitor's
  hardware-per-physics slope, each column over its corrector's, the
  corrector's a secant over the hardware block's width from its start value.
* **The closed orbit** at every monitor, its hardware reading through the
  monitor's calibration.
* **The dispersion and the chromaticity**, per unit of RF frequency in
  hardware: the physics value over ``-f_RF * mcf``, through the RF setpoint's
  calibration and, for the dispersion, each monitor's.
* **The tune and chromaticity response**, each device's hardware copy of the
  family value by the scale its Middle Layer function applies --
  ``k_per_amp * Leff / mean(Leff)`` for the tune response, ``k_per_amp /
  (-RF0 * MCF)`` for the chromaticity response. ``k_per_amp`` is the model
  file's own record of each device's conversion over its step, and each
  device's write entry in the build, a string supply's slice included, meets
  it as a secant from the build's start value over that step.
  ``meastuneresp`` answers the sum of its per-device columns, and its physics
  answer carries ``measrespmat``'s units quirk (:func:`_units_quirk`).

A transport line's monitors read nothing in its wiring, so its physics values
are the plug-in's own, read where the export's ``AT.ATIndex`` places each
monitor on the deck the build wrote, and its monitor conversion is the one the
export states.

No step or solver stands between the two sides, so they agree to the digits
the file writes (``CONVERSION_RTOL``); the chromaticity response alone is held
to the band its two unit runs leave (``CHROMATICITY_RTOL``), and a response
device's build calibration to the band the import's table sampling of the
facility's conversion leaves (``TABLE_SAMPLING_RTOL``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from tests.facility._mml_built import FIXTURES, BuiltModel
from tests.facility._model_reference import (
    CONVERSION_ATOL_M,
    CONVERSION_RTOL,
    MATLAB_LINES,
    MATLAB_RINGS,
    TRANSPORT_DISPERSION_ATOL_M,
    model_reference,
    refusal,
    section,
)

pytest.importorskip("at")

from tests.facility import test_check_a as check_a
from tests.services.mml import _mml_recipes as recipes

# xdist_group("mml_built"): the session ``mml_built`` fixture builds each fixture
# tree once, and every module reading it shares the group so that one build
# serves them all on one worker.
pytestmark = pytest.mark.xdist_group("mml_built")

RINGS = pytest.mark.parametrize(
    ("tree", "stem"), MATLAB_RINGS, ids=[stem for _tree, stem in MATLAB_RINGS]
)
LINES = pytest.mark.parametrize(
    ("tree", "stem"), MATLAB_LINES, ids=[stem for _tree, stem in MATLAB_LINES]
)
MODELS = pytest.mark.parametrize(
    ("tree", "stem"),
    (*MATLAB_RINGS, *MATLAB_LINES),
    ids=[stem for _tree, stem in (*MATLAB_RINGS, *MATLAB_LINES)],
)
RESPONSES = pytest.mark.parametrize("name", ["tune_response", "chromaticity_response"])

Built = Callable[[str, str], BuiltModel]

#: How far the two unit systems' chromaticity responses agree. Each run steps
#: the family through a different unit round trip, so the stepped
#: chromaticities differ by the model's own chromaticity noise. SPEAR3 has a
#: cavity and ``modelchro`` differences tunes over a 1 Hz RF step, which leaves
#: 3-7e-9. NSLS-II has none and ``tunechrom`` differences tunes 1e-8 apart in
#: momentum, which scales tune roundoff by 1e8: a 1e-14 change in the sextupole
#: strengths moves its horizontal chromaticity by 1e-7.
CHROMATICITY_RTOL = {"spear3.storagering": 2e-8, "nsls2.storagering": 2e-7}

#: How far a tune or chromaticity corrector's secant through the build's
#: calibration may sit from the model file's ``k_per_amp``. The import samples
#: the facility's nonlinear conversion (``amp2k``) into a piecewise-linear
#: table, so a secant over the small ``DeltaRespMat`` step reads the slope of
#: the table segment it falls in, not the facility function's own slope there;
#: on the coarsest grids that differs by up to 2.8e-2.
TABLE_SAMPLING_RTOL = 3e-2

#: How near an energy knob's start value keeps to the nominal currents the
#: export states. No model file records a hardware answer for the knob, so
#: nothing holds it to ``CONVERSION_RTOL``.
ENERGY_NOMINAL_RTOL = 1e-4

#: Every model-file section whose hardware answer a test here converts.
CHECKED = frozenset(
    {
        "state.orbit",
        "dispersion",
        "chromaticity",
        "orbit_response",
        "tune_response",
        "chromaticity_response",
    }
)


# ---------------------------------------------------------------------------
# The build's calibrations
# ---------------------------------------------------------------------------


class Calibrations:
    """One model's wiring entries, found by family and ``DeviceList`` row.

    A family's entry is the one whose engine words are the family's and that
    binds the listed device: an entry with no slices binds the device its
    address sits on, an entry over slices binds each device a slice names, so
    a supply driving a whole string binds every device of the string.
    """

    def __init__(self, built: BuiltModel) -> None:
        self.built = built
        self.wiring = check_a.Wiring(built)

    def binding(
        self, family: str, direction: str, device: Sequence[float]
    ) -> tuple[dict[str, Any], float]:
        """The one ``direction`` entry of ``family`` binding the listed device, and its weight.

        The weight is the one the entry's first slice on the device states,
        the share of the entry's physics value the device takes; an entry with
        no slices, or a slice stating none, weighs 1.
        """
        row = check_a._row(device)
        words = self.wiring.engine(family)
        members = self.wiring.groups[self.wiring.mapping.mapped(family)]
        owners = [name for name in sorted(members) if self.wiring.rows.get(name) == row]
        assert len(owners) == 1, f"{self.built.name} {family} lists {len(owners)} devices {row}"
        (owner,) = owners
        found: list[tuple[dict[str, Any], float]] = []
        for entry in self.built.wiring:
            if entry.get("direction") != direction or dict(entry.get("engine") or {}) != words:
                continue
            own = self.wiring.on_device.get(str(entry["address"]))
            pieces = [
                piece for piece in entry.get("slices") or [{}] if piece.get("device", own) == owner
            ]
            if pieces:
                weight = pieces[0].get("weight")
                found.append((entry, 1.0 if weight is None else float(weight)))
        assert len(found) == 1, (
            f"{self.built.name} wires {len(found)} {direction} entries for {family} {row} ({owner})"
        )
        return found[0]

    def entry(self, family: str, direction: str, device: Sequence[float]) -> dict[str, Any]:
        """The one ``direction`` entry of ``family`` binding the listed device."""
        return self.binding(family, direction, device)[0]

    def curve(self, family: str, direction: str, device: Sequence[float]) -> Any:
        """The hardware-to-physics curve of one device's entry."""
        from osprey.simulation.engines.calibration import curve_from_record, field

        entry = self.entry(family, direction, device)
        curve = curve_from_record(field(field(entry, "calibration"), "curve"))
        assert curve is not None, f"{entry['address']} carries no calibration"
        return curve

    def start(self, family: str, device: Sequence[float]) -> float:
        """The hardware value the build starts one setpoint at."""
        from osprey.simulation.engines import pyat as engine

        entry = self.entry(family, "write", device)
        values = engine.start_values(self.built.deck, [entry], self.built.settings)
        value = values[str(entry["address"])]
        assert isinstance(value, float), f"{entry['address']} starts at {value!r}"
        return value


_CALIBRATIONS: dict[tuple[str, str], Calibrations] = {}


def _calibrations(built: BuiltModel) -> Calibrations:
    key = (str(built.facility), built.stem)
    if key not in _CALIBRATIONS:
        _CALIBRATIONS[key] = Calibrations(built)
    return _CALIBRATIONS[key]


def _secant(curve: Any, start: float, width: float) -> float:
    """The physics change per hardware unit of ``curve`` over one step."""
    from osprey.simulation.engines.calibration import to_physics

    return (to_physics(curve, start + width) - to_physics(curve, start)) / width


def _monitor_per_metre(calibrations: Calibrations, family: str, device: Sequence[float]) -> float:
    """A monitor's hardware reading per metre of orbit, from its calibration's slope."""
    return 1.0 / _secant(calibrations.curve(family, "read", device), 0.0, 1.0)


def _rf_per_hertz(calibrations: Calibrations, step: float) -> float:
    """The RF setpoint's hardware unit per hertz, over a step from its start value."""
    families = [
        family
        for family, wired in calibrations.wiring.model.wiring.items()
        if wired.engine is not None and wired.engine.attribute == "Frequency"
    ]
    assert len(families) == 1, f"{calibrations.built.name} wires RF families {families}"
    (family,) = families
    rows = sorted(
        row
        for name, row in calibrations.wiring.rows.items()
        if name in calibrations.wiring.groups[calibrations.wiring.mapping.mapped(family)]
    )
    assert len(rows) == 1, f"{family} lists devices {rows}"
    start = calibrations.start(family, rows[0])
    return 1.0 / _secant(calibrations.curve(family, "write", rows[0]), start, step)


def _offenders(expected: np.ndarray, stated: np.ndarray, where: Sequence[str]) -> list[str]:
    """The entries further than ``CONVERSION_RTOL`` of themselves plus that of the block's rms."""
    floor = CONVERSION_RTOL * float(np.sqrt(np.mean(stated**2)))
    bad = ~(np.abs(expected - stated) <= CONVERSION_RTOL * np.abs(stated) + floor)
    return [
        f"{where[int(i)]}: {expected.flat[i]!r} vs {stated.flat[i]!r}" for i in np.flatnonzero(bad)
    ]


def _hardware_answers(reference: dict[str, Any]) -> set[str]:
    """Every section of a model file that answers in hardware units, refusals left out."""
    found: set[str] = set()
    for name, block in reference.items():
        if name == "_export" or refusal(block) is not None:
            continue
        if name == "state":
            if any(refusal(entry.get("hardware")) is None for entry in block["orbit"].values()):
                found.add("state.orbit")
        elif "hardware" in block and refusal(block["hardware"]) is None:
            found.add(name)
    return found


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@MODELS
def test_every_hardware_answer_is_converted_here(tree: str, stem: str) -> None:
    """No section of the model file answers in hardware units without a test here."""
    unchecked = _hardware_answers(model_reference(tree, stem)) - CHECKED
    assert not unchecked, f"{stem} answers in hardware units with no conversion check: {unchecked}"


# ---------------------------------------------------------------------------
# The orbit response and the closed orbit
# ---------------------------------------------------------------------------


@MODELS
def test_hardware_orbit_response_is_the_physics_one_through_the_calibrations(
    tree: str, stem: str, mml_built: Built
) -> None:
    """Each mm/A entry is its m/rad entry times the monitor's slope over the corrector's.

    ``physics2hw`` on a response scales row ``i`` by monitor ``i``'s hardware
    per metre and divides column ``j`` by ``dI_j / dtheta_j``, ``dI_j`` the
    hardware block's width and ``dtheta_j`` the kick that width makes from the
    corrector's start value -- a secant of the corrector's calibration. The
    blocks list the same devices in the same order in both unit sets.
    """
    reference = model_reference(tree, stem)
    response = section(reference, "orbit_response")
    message = refusal(response["hardware"])
    if message is not None:
        pytest.skip(f"the model file refused the hardware orbit response: {message}")
    calibrations = _calibrations(mml_built(tree, stem))
    offenders: list[str] = []
    for physics, hardware in zip(response["physics"], response["hardware"], strict=True):
        for side in ("monitor", "actuator"):
            assert physics[side]["family"] == hardware[side]["family"]
            assert physics[side]["device_list"] == hardware[side]["device_list"]
        monitor = physics["monitor"]["family"]
        corrector = physics["actuator"]["family"]
        monitors = physics["monitor"]["device_list"]
        correctors = physics["actuator"]["device_list"]
        widths = np.broadcast_to(
            np.asarray(hardware["actuator_delta"], dtype=float), (len(correctors),)
        )
        rows = np.array([_monitor_per_metre(calibrations, monitor, row) for row in monitors])
        columns = np.array(
            [
                _secant(
                    calibrations.curve(corrector, "write", row),
                    calibrations.start(corrector, row),
                    float(width),
                )
                for row, width in zip(correctors, widths, strict=True)
            ]
        )
        expected = check_a._entries(physics) * rows[:, None] * columns[None, :]
        where = [f"{monitor}{m} / {corrector}{c}" for m in monitors for c in correctors]
        offenders += _offenders(expected, check_a._entries(hardware), where)
    assert not offenders, (
        f"{len(offenders)} hardware entries differ from physics through the calibrations: "
        f"{offenders[:10]}"
    )


@RINGS
def test_closed_orbit_hardware_is_the_physics_one_through_the_monitor_calibration(
    tree: str, stem: str, mml_built: Built
) -> None:
    """``state.orbit``'s hardware reading, through each monitor's calibration, is its physics."""
    from osprey.simulation.engines.calibration import to_physics

    reference = model_reference(tree, stem)
    state = section(reference, "state")
    calibrations = _calibrations(mml_built(tree, stem))
    for family, block in state["orbit"].items():
        rows = block["device_list"]
        converted = [
            to_physics(calibrations.curve(family, "read", row), float(hardware))
            for row, hardware in zip(rows, block["hardware"], strict=True)
        ]
        np.testing.assert_allclose(
            converted,
            np.asarray(block["physics"], dtype=float),
            rtol=CONVERSION_RTOL,
            atol=CONVERSION_ATOL_M,
            err_msg=f"closed orbit at {family}: hardware through the calibration vs physics",
        )


def _exported_monitors(
    built: BuiltModel, family: str, device_list: Sequence[Sequence[float]]
) -> tuple[list[int], np.ndarray]:
    """Each listed line monitor's deck position and hardware reading per metre, as exported.

    The position is the export's ``AT.ATIndex``; the reading per metre the
    export's ``Monitor`` ``Physics2HWParams``, one number or one per device.
    """
    ao = json.loads((FIXTURES / built.tree / f"{built.stem}.ao.json").read_text(encoding="utf-8"))
    body = ao[family]
    rows = [check_a._row(row) for row in np.atleast_2d(body["DeviceList"])]
    per_metre = np.ravel(np.asarray(body["Monitor"]["Physics2HWParams"], dtype=float))
    if per_metre.size == 1:
        per_metre = np.repeat(per_metre, len(rows))
    assert per_metre.size == len(rows), f"{family} states {per_metre.size} conversions"
    positions = check_a._exported_positions(built, family, device_list)
    scale = np.array([per_metre[rows.index(check_a._row(row))] for row in device_list])
    return positions, scale


@LINES
def test_line_orbit_hardware_is_the_plug_in_orbit_through_the_export_conversion(
    tree: str, stem: str, mml_built: Built
) -> None:
    """Each monitor's hardware history is the plug-in's one-pass orbit there, in the export's unit.

    ``getpvmodel`` on a line answers a reading history per monitor; every
    reading in it is the orbit the launch twiss tracks to the monitor.
    """
    reference = model_reference(tree, stem)
    state = section(reference, "state")
    built = mml_built(tree, stem)
    deck = check_a._deck(built, reference)
    start, _twiss = check_a._launch(deck)
    for family, row in check_a._planes(reference).items():
        block = state["orbit"][family]
        positions, per_metre = _exported_monitors(built, family, block["device_list"])
        orbit = recipes.track(deck.lattice, start, positions)[row, 0, :]
        hardware = np.asarray(block["hardware"], dtype=float)
        history = hardware.reshape(len(positions), -1)
        expected = np.repeat((orbit * per_metre)[:, None], history.shape[1], axis=1)
        np.testing.assert_allclose(
            history,
            expected,
            rtol=CONVERSION_RTOL,
            atol=CONVERSION_ATOL_M * float(np.max(per_metre)),
            err_msg=f"line orbit at {family}: the plug-in's orbit in hardware units",
        )


# ---------------------------------------------------------------------------
# The dispersion and the chromaticity
# ---------------------------------------------------------------------------


@RINGS
def test_dispersion_hardware_is_the_physics_one_per_unit_of_rf(
    tree: str, stem: str, mml_built: Built
) -> None:
    """``modeldisp('Hardware')`` is each monitor's reading per unit of the RF setpoint.

    A physics dispersion is metres per unit of momentum; an RF change ``df``
    moves the momentum by ``-df / (f_cavity * mcf)``, both recorded beside the
    answer. Each monitor's calibration takes the metres to its reading, the RF
    setpoint's the hertz to its unit.
    """
    reference = model_reference(tree, stem)
    block = section(reference, "dispersion")
    calibrations = _calibrations(mml_built(tree, stem))
    per_rf = 1.0 / (
        -float(block["f_cavity"])
        * float(block["mcf"])
        * _rf_per_hertz(calibrations, float(block["delta_rf_hw"]))
    )
    offenders: list[str] = []
    for plane in ("x", "y"):
        monitors = block["monitors"][plane]
        family = monitors["family"]
        rows = monitors["device_list"]
        scale = np.array([_monitor_per_metre(calibrations, family, row) for row in rows])
        expected = np.asarray(block["physics"][plane], dtype=float) * scale * per_rf
        stated = np.asarray(block["hardware"][plane], dtype=float)
        offenders += _offenders(expected, stated, [f"{family}{row}" for row in rows])
    assert not offenders, f"{len(offenders)} hardware dispersions differ: {offenders[:10]}"


@LINES
def test_line_dispersion_hardware_is_the_plug_in_twiss_dispersion(
    tree: str, stem: str, mml_built: Built
) -> None:
    """``modeldisp`` on a line answers ``twissline``'s dispersion, metres, in both unit sets.

    Its transport branch reads the twiss dispersion and converts nothing, so
    the hardware answer is the plug-in's dispersion at each monitor's exported
    position as it stands; the band adds the one-pass rounding of that replay.
    """
    reference = model_reference(tree, stem)
    block = section(reference, "dispersion")
    assert block["method"] == "twiss", block["method"]
    built = mml_built(tree, stem)
    deck = check_a._deck(built, reference)
    _start, twiss = check_a._launch(deck)
    for plane, row in (("x", 0), ("y", 2)):
        monitors = block["monitors"][plane]
        positions, _per_metre = _exported_monitors(
            built, monitors["family"], monitors["device_list"]
        )
        eta = recipes.twissline_dispersion(deck.lattice, twiss, positions)
        np.testing.assert_allclose(
            np.asarray(block["hardware"][plane], dtype=float),
            eta[row],
            rtol=CONVERSION_RTOL,
            atol=TRANSPORT_DISPERSION_ATOL_M,
            err_msg=f"line dispersion {plane} in hardware units",
        )


@RINGS
def test_chromaticity_hardware_is_the_physics_one_per_unit_of_rf(
    tree: str, stem: str, mml_built: Built
) -> None:
    """``modelchro('Hardware')`` is the tune change per unit of the RF setpoint.

    A physics chromaticity is tune per unit of momentum; an RF change ``df``
    moves the momentum by ``-df / (f_RF * mcf)``, the dispersion's recorded
    frequency and compaction. The RF setpoint's calibration takes the hertz to
    its unit over the step the answer was taken at.
    """
    reference = model_reference(tree, stem)
    chrom = section(reference, "chromaticity")
    message = refusal(chrom["hardware"])
    if message is not None:
        pytest.skip(f"the model file refused the hardware chromaticity: {message}")
    dispersion = section(reference, "dispersion")
    calibrations = _calibrations(mml_built(tree, stem))
    per_rf = 1.0 / (
        -float(dispersion["f_cavity"])
        * float(dispersion["mcf"])
        * _rf_per_hertz(calibrations, float(chrom["delta_rf_hw"]))
    )
    expected = np.asarray(chrom["physics"], dtype=float) * per_rf
    np.testing.assert_allclose(
        np.asarray(chrom["hardware"], dtype=float),
        expected,
        rtol=CONVERSION_RTOL,
        atol=0,
        err_msg="chromaticity: hardware vs physics per unit of RF",
    )


# ---------------------------------------------------------------------------
# The family responses
# ---------------------------------------------------------------------------


def _family_value(value: Any, members: int, where: str, name: str) -> np.ndarray:
    """The one ``(x, y)`` family value per unit of the family's step.

    ``meastuneresp`` defaults to its matrix output, which sums the per-device
    columns ``measrespmat`` expands the family value into (mml/meastuneresp.m
    ~80, ~426): its two numbers are the family value times the device count.
    ``measchroresp`` returns the per-device columns unsummed, each a copy of
    the family value.

    Args:
        value: The Middle Layer's answer: two numbers, or two rows or two
            columns of ``members`` equal numbers.
        members: How many devices the family has.
        where: What the value is, for the failure message.
        name: The response section the value answers.
    """
    array = np.asarray(value, dtype=float)
    if name == "tune_response":
        if array.shape != (2,):
            raise AssertionError(f"{where}: shape {array.shape} is not the summed pair")
        return array / members
    if array.shape == (2,):
        return array
    if array.shape == (2, members):
        columns = array
    elif array.shape == (members, 2):
        columns = array.T
    else:
        raise AssertionError(f"{where}: shape {array.shape} is neither 2 nor 2 x {members}")
    np.testing.assert_allclose(
        columns, np.repeat(columns[:, :1], members, axis=1), rtol=1e-12, err_msg=where
    )
    return columns[:, 0]


def _per_device(value: Any, members: int, where: str) -> np.ndarray:
    """A family answer as ``2 x members``: one ``(x, y)`` column per device.

    Args:
        value: The Middle Layer's answer: two numbers, or two rows or two
            columns of ``members`` numbers.
        members: How many devices the family has.
        where: What the value is, for the failure message.
    """
    array = np.asarray(value, dtype=float)
    if array.shape == (2,):
        return np.repeat(array[:, None], members, axis=1)
    if array.shape == (2, members):
        return array
    if array.shape == (members, 2):
        return array.T
    raise AssertionError(f"{where}: shape {array.shape} is neither 2 nor 2 x {members}")


def _chromaticity_per_hardware(reference: dict[str, Any]) -> float:
    """The factor ``measchro`` turns a physics chromaticity into its hardware unit by.

    In Model mode ``measchro`` answers ``modelchro('Physics')`` and, asked for
    hardware, divides it by ``-RF0 * MCF`` (mml/measchro.m ~311-372): ``RF0``
    is ``getrf('Model', 'Hardware')``, recorded as ``rf0_hw``, and ``MCF`` is
    ``getmcf('Model')`` on the saved deck, recorded beside the dispersion. On a
    ring with no cavity ``getrf`` answers the Middle Layer's own constant, not
    the frequency the deck's circumference gives, so only the recorded value
    holds.
    """
    dispersion = section(reference, "dispersion")
    response = section(reference, "chromaticity_response")
    missing = [key for key, held in (("mcf", dispersion), ("rf0_hw", response)) if key not in held]
    if missing:
        pytest.fail(f"the model file records no {missing}; the hardware chromaticity has no unit")
    return 1.0 / (-float(response["rf0_hw"]) * float(dispersion["mcf"]))


def _mcf_reread(
    reference: dict[str, Any], facts: dict[str, Any], physics: np.ndarray
) -> np.ndarray:
    """Each plane's hardware chromaticity response over the one a fixed compaction gives.

    ``measchro`` reads the compaction again at every point (mml/measchro.m
    ~314), and the Middle Layer's compaction is a one-sided difference that
    moves when the family is stepped. So the hardware response is
    ``(C2 / M2 - C1 / M1) / dC`` against ``(C2 - C1) / (M1 dC)``: ``C1`` the
    chromaticity the step starts from (``chromaticity_start``), ``C2 = C1 +
    dC`` with ``dC = physics * sum(k_per_amp * width)``, ``M1`` the deck's
    compaction and ``M2`` the stepped one (``mcf_stepped``). The factor is
    ``((C1 + dC) M1 / M2 - C1) / dC``.
    """
    response = section(reference, "chromaticity_response")
    held_in = (("chromaticity_start", response), ("mcf_stepped", facts))
    missing = [key for key, held in held_in if key not in held]
    if missing:
        pytest.fail(f"the model file records no {missing}; the compaction re-read has no replay")
    start = np.asarray(response["chromaticity_start"], dtype=float).ravel()
    deck = float(section(reference, "dispersion")["mcf"])
    stepped = float(facts["mcf_stepped"])
    change = physics * float(np.sum(check_a._step(facts)))
    return np.asarray(((start + change) * deck / stepped - start) / change, dtype=float)


def _hardware_per_device(reference: dict[str, Any], name: str, facts: dict[str, Any]) -> np.ndarray:
    """Each device's hardware answer per unit of the family's physics value.

    ``measrespmat`` scales device ``n``'s column by ``k_n * Leff_n / mean(Leff)``
    (mml/measrespmat.m ~688-719); ``measchroresp`` by ``k_n`` alone
    (mml/measchroresp.m ~509-515), on top of ``measchro``'s hardware unit.
    """
    k_per_amp = check_a._flat(facts, "k_per_amp")
    if name == "tune_response":
        return np.asarray(k_per_amp * check_a._leff_weights(facts), dtype=float)
    return k_per_amp * _chromaticity_per_hardware(reference)


def _units_quirk(facts: dict[str, Any]) -> np.ndarray:
    """Each device's physics tune-response column over the one a correct normalisation gives.

    ``measrespmat`` picks its step conversion by the units the family STORES
    (``family2units``, mml/measrespmat.m ~690, ~714), not the units the caller
    asked for. A tune corrector stores hardware units, so a physics measurement
    steps the family by the true ``dK_n = k_per_amp_n * width_n`` and then reads
    those K values as amperes: it normalises by ``DPb_n = hw2physics(K0_n +
    dK_n) - hw2physics(K0_n)`` (the model file's ``physics_step_read_as_amps``)
    and scales column ``n`` by ``DPb_n * w_n / dK_n``, ``w = Leff / mean(Leff)``.
    Against the columns ``dQ / sum(dK * w)`` that is the factor ``f_n =
    (DPb_n * w_n / dK_n) * sum(dK * w) / sum(DPb * w)``.
    """
    step = check_a._step(facts)
    weights = check_a._leff_weights(facts)
    read_as_amps = np.asarray(facts["physics_step_read_as_amps"], dtype=float).ravel()
    assert read_as_amps.shape == step.shape, (
        f"physics_step_read_as_amps has {read_as_amps.size} entries for {step.size} devices"
    )
    factors = (
        (read_as_amps * weights / step)
        * float(np.sum(step * weights))
        / float(np.sum(read_as_amps * weights))
    )
    return np.asarray(factors, dtype=float)


@RINGS
@RESPONSES
def test_response_hardware_is_the_physics_one_through_k_per_amp(
    tree: str, stem: str, name: str
) -> None:
    """Every device's hardware value is the family's physics value times its recorded scale.

    One stepped change answers both calls: the family value is the same in
    both, and the hardware call scales each device's copy by the conversion
    the Middle Layer applies to it. The tune response sums those copies, so
    its hardware pair is the family value times the sum of the scales.
    """
    reference = model_reference(tree, stem)
    block = section(reference, name)
    for family, facts in check_a._answered(block).items():
        members = len(facts["device_list"])
        physics = _family_value(block["physics"][family], members, f"{name} {family} physics", name)
        scale = _hardware_per_device(reference, name, facts)
        if name == "tune_response":
            hardware = np.asarray(block["hardware"][family], dtype=float)
            expected = physics * float(np.sum(scale)) / float(np.mean(_units_quirk(facts)))
        else:
            hardware = _per_device(block["hardware"][family], members, f"{name} {family} hardware")
            reread = _mcf_reread(reference, facts, physics)
            expected = (physics * reread)[:, None] * scale[None, :]
        np.testing.assert_allclose(
            hardware,
            expected,
            rtol=CONVERSION_RTOL if name == "tune_response" else CHROMATICITY_RTOL[stem],
            atol=0,
            err_msg=f"{name} {family}: hardware vs physics through k_per_amp",
        )


@RINGS
@RESPONSES
def test_response_k_per_amp_is_the_build_calibration_over_the_step(
    tree: str, stem: str, name: str, mml_built: Built
) -> None:
    """Each answered device's write entry, as a secant over its step, is its ``k_per_amp``.

    A device's change per ampere in the build is its entry's calibration
    taken as a secant from the build's start value over the family's
    ``delta_resp_mat``, times the share its slice takes of a string supply.
    A device no write entry binds fails naming its family and row. The band
    is the table sampling of the facility's conversion (``TABLE_SAMPLING_RTOL``).
    """
    reference = model_reference(tree, stem)
    block = section(reference, name)
    calibrations = _calibrations(mml_built(tree, stem))
    offenders: list[str] = []
    for family, facts in check_a._answered(block).items():
        recorded = check_a._flat(facts, "k_per_amp")
        widths = check_a._flat(facts, "delta_resp_mat")
        for row, k_per_amp, width in zip(facts["device_list"], recorded, widths, strict=True):
            entry, weight = calibrations.binding(family, "write", row)
            curve = calibrations.curve(family, "write", row)
            start = calibrations.start(family, row)
            built = weight * _secant(curve, start, float(width))
            if not abs(built - k_per_amp) <= TABLE_SAMPLING_RTOL * abs(k_per_amp):
                offenders.append(
                    f"{family}{check_a._row(row)} ({entry['address']}): "
                    f"{built!r} vs k_per_amp {float(k_per_amp)!r}"
                )
    assert not offenders, (
        f"{name}: {len(offenders)} devices' build calibration differs from k_per_amp: "
        f"{offenders[:10]}"
    )


# ---------------------------------------------------------------------------
# The energy knob
# ---------------------------------------------------------------------------


@MODELS
def test_energy_knob_starts_at_the_nominal_currents_the_export_states(
    tree: str, stem: str, mml_built: Built
) -> None:
    """A wired energy knob's start value lies within ``ENERGY_NOMINAL_RTOL`` of each nominal.

    The model files hold no reference answer for the knob, so its start value,
    the export's energy table read back at the deck energy, is held only near
    every ``Setpoint`` nominal the export states for the knob's family.
    """
    from osprey.simulation.engines import pyat as engine

    built = mml_built(tree, stem)
    calibrations = _calibrations(built)
    knobs = sorted(
        family
        for family, wired in calibrations.wiring.model.wiring.items()
        if wired.engine is not None and wired.engine.attribute == "energy"
    )
    if not knobs:
        pytest.skip(f"{built.name} wires no energy knob")
    va = json.loads((FIXTURES / tree / f"{stem}.va.json").read_text(encoding="utf-8"))
    for family in knobs:
        words = calibrations.wiring.engine(family)
        entries = [
            entry
            for entry in built.wiring
            if entry.get("direction") == "write" and dict(entry.get("engine") or {}) == words
        ]
        assert len(entries) == 1, f"{built.name} wires {len(entries)} {family} energy knobs"
        (entry,) = entries
        start = engine.start_values(built.deck, [entry], built.settings)[str(entry["address"])]
        nominals = np.asarray(va["families"][family]["nominals"]["Setpoint"]["values"], dtype=float)
        assert nominals.size, f"the export states no {family} Setpoint nominal"
        np.testing.assert_allclose(
            np.full(nominals.shape, float(start)),
            nominals,
            rtol=ENERGY_NOMINAL_RTOL,
            atol=0,
            err_msg=f"{family} ({entry['address']}): the build's start value vs the nominals",
        )


# ---------------------------------------------------------------------------
# Addressing and the helpers, on hand-built inputs
# ---------------------------------------------------------------------------


def test_a_device_the_wiring_does_not_carry_fails_naming_it(mml_built: Built) -> None:
    """A listed row no device of the family holds fails naming the family and the row."""
    calibrations = _calibrations(mml_built(*MATLAB_RINGS[0]))
    with pytest.raises(AssertionError, match=r"BPMx lists 0 devices \(99, 99\)"):
        calibrations.entry("BPMx", "read", [99, 99])


def test_a_device_of_a_string_binds_through_its_slice_of_the_supply(mml_built: Built) -> None:
    """Every device a string supply's slices name binds that one supply, at its slice's weight."""
    calibrations = _calibrations(mml_built(*MATLAB_RINGS[0]))
    supplies = {
        (str(entry["address"]), weight)
        for row in ([3, 1], [3, 2], [4, 1])
        for entry, weight in [calibrations.binding("SF", "write", row)]
    }
    assert len(supplies) == 1, supplies
    ((address, weight),) = supplies
    (entry,) = [item for item in calibrations.built.wiring if item["address"] == address]
    assert len(entry["slices"]) > 1, f"{address} drives no string"
    assert weight == 1.0


def test_hardware_answers_name_every_answered_section() -> None:
    """A refused hardware answer is left out; an answered one is named by its section."""
    reference = {
        "_export": {"hardware": [1.0]},
        "state": {"orbit": {"BPMx": {"hardware": [0.1]}, "BPMy": {"hardware": [0.2]}}},
        "tune": {"refused": "transport line"},
        "chromaticity": {"hardware": {"refused": "no cavity"}},
        "dispersion": {"hardware": {"x": [1.0], "y": [0.0]}},
    }
    assert _hardware_answers(reference) == {"state.orbit", "dispersion"}


def test_family_value_reads_every_layout() -> None:
    """A chromaticity copy reads as one pair; a summed tune pair is divided by the members."""
    chromaticity = "chromaticity_response"
    for value in ([0.1, -0.2], [[0.1, 0.1, 0.1], [-0.2, -0.2, -0.2]], [[0.1, -0.2]] * 3):
        np.testing.assert_array_equal(_family_value(value, 3, "layout", chromaticity), [0.1, -0.2])
    with pytest.raises(AssertionError, match="unequal"):
        _family_value([[0.1, 0.2, 0.1], [-0.2, -0.2, -0.2]], 3, "unequal", chromaticity)
    np.testing.assert_allclose(
        _family_value([0.3, -0.6], 3, "summed", "tune_response"), [0.1, -0.2]
    )
    with pytest.raises(AssertionError, match="summed pair"):
        _family_value([[0.1, 0.1, 0.1], [-0.2, -0.2, -0.2]], 3, "columns", "tune_response")


def test_hardware_scales_follow_each_middle_layer_function() -> None:
    """TRM columns scale by ``k_n Leff_n / mean(Leff)``; CRM columns by ``k_n / (-RF0 * MCF)``."""
    facts = {
        "device_list": [[1, 1], [2, 1]],
        "delta_resp_mat": [2.0, 2.0],
        "leff": [0.2, 0.4],
        "k_per_amp": [0.01, 0.03],
    }
    reference = {
        "dispersion": {"mcf": 1.7e-3},
        "chromaticity_response": {"rf0_hw": 476.3},
    }
    np.testing.assert_allclose(
        _hardware_per_device(reference, "tune_response", facts),
        [0.01 * 0.2 / 0.3, 0.03 * 0.4 / 0.3],
    )
    unit = 1.0 / (-476.3 * 1.7e-3)
    np.testing.assert_allclose(
        _hardware_per_device(reference, "chromaticity_response", facts),
        [0.01 * unit, 0.03 * unit],
    )


def test_per_device_hardware_reads_members_that_differ() -> None:
    """A hardware answer whose devices differ is read per device, never as one replicated value."""
    np.testing.assert_array_equal(
        _per_device([[1.0, 2.0], [3.0, 4.0]], 2, "rows"), [[1.0, 2.0], [3.0, 4.0]]
    )
    np.testing.assert_array_equal(_per_device([1.0, 3.0], 2, "pair"), [[1.0, 1.0], [3.0, 3.0]])
    with pytest.raises(AssertionError, match="shape"):
        _per_device([1.0, 2.0, 3.0], 2, "shape")


def test_units_quirk_is_the_leff_weights_where_amps_and_k_coincide() -> None:
    """Where K read as amperes steps as K does, the factors are ``Leff / mean(Leff)``.

    Their mean is then one, so the tune response's two unit sets relate by
    ``k_per_amp * Leff / mean(Leff)`` alone; any other reading moves the mean.
    """
    facts = {
        "device_list": [[1, 1], [2, 1]],
        "delta_resp_mat": [2.0, 2.0],
        "leff": [0.2, 0.4],
        "k_per_amp": [0.01, 0.03],
        "physics_step_read_as_amps": [0.02, 0.06],
    }
    np.testing.assert_allclose(_units_quirk(facts), [2.0 / 3.0, 4.0 / 3.0], rtol=1e-15)
    doubled = {**facts, "physics_step_read_as_amps": [0.04, 0.06]}
    assert abs(float(np.mean(_units_quirk(doubled))) - 1.0) > 1e-3
