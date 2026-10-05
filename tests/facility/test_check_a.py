"""Check A, import fidelity: the built deck replays the Middle Layer's own model answers.

A 2.1 export records, beside the deck it saves, what the Middle Layer's model
mode answered on that deck: tunes, chromaticity, dispersion, the closed orbit,
the orbit response and the tune and chromaticity response of every corrector
family. Model mode answers through shortcuts of its own -- a Linear orbit
response at fixed path length, a dispersion about half the deck's RF offset,
whole-family unipolar steps, a one-sided chromaticity step with the momentum
compaction read again at every call -- and every test here repeats them: pyAT
runs the Middle Layer's recipe, ported step for step in ``_mml_recipes``, on
the deck the facility build wrote for the model (its cavities put back the way
the saved deck held them), each device found through the model's wiring, and
meets the model file within the replay's numerical floor
(``_model_reference``). A pass proves the import and the build carried the
deck, its element addressing and its family facts across without loss; the
shortcuts live here and nowhere else, so no measurement is ever held to them.

The recipe is chosen by what the file recorded (a ``method``, a calculator, a
cavity state), never by which tree it came from. The two facility storage rings
and the NSLS-II transport line are checked, the line to the same strict floor
as the rings; the engine builds each model on the deck as the build wrote it,
in 6D on the rings. A shifted quadrupole shows the tune replay can fail.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from tests.facility._mml_built import BuiltModel
from tests.facility._model_reference import (
    CIRCUMFERENCE_RTOL,
    DISPERSION_ORBITS,
    MATLAB_LINES,
    MATLAB_RINGS,
    MCF_RTOL,
    ORBIT_ATOL_M,
    ORM_RMS_FRACTION,
    TRANSPORT_DISPERSION_ATOL_M,
    TUNE_ATOL,
    TUNE_DIFFERENCE_ATOL,
    model_reference,
    refusal,
    section,
)

at = pytest.importorskip("at")

from tests.services.mml import _mml_recipes as recipes  # noqa: E402

# xdist_group("mml_built"): the session ``mml_built`` fixture builds each fixture
# tree once, and every module reading it shares the group so that one build
# serves them all on one worker.
pytestmark = pytest.mark.xdist_group("mml_built")

#: How far the non-vacuity quadrupole's gradient is moved, in 1/m**2.
QUAD_SHIFT = 1e-3

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


# ---------------------------------------------------------------------------
# The built deck and its wiring
# ---------------------------------------------------------------------------


def _row(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


class Wiring:
    """One model's wiring, read as the elements each ``DeviceList`` row of a family binds.

    A family is the mapping's: its group of devices in the model, and the
    engine words the model wires it through. An entry with exactly those words binds its
    element to the device its address sits on; an entry over slices binds each
    slice to the device the slice names, so a supply driving a whole string
    binds every device of the string to its own elements.
    """

    def __init__(self, built: BuiltModel) -> None:
        from osprey.facility.layers.mml.mapping import MAPPING_FILE, read_mapping

        self.built = built
        self.mapping = read_mapping(built.facility / MAPPING_FILE)
        (self.model,) = [
            model for model in self.mapping.models.values() if model.name == built.name
        ]
        self.groups = {
            str(group["id"]): set(group.get("members", [])) for group in built.document["groups"]
        }
        self.rows = {
            str(device["id"]): _row(device["attributes"]["DeviceList"])
            for device in built.document["devices"]
            if device.get("model") == built.name
            and "DeviceList" in (device.get("attributes") or {})
        }
        self.on_device = {
            str(channel["id"]): (channel.get("on") or {}).get("device")
            for channel in built.document["channels"]
        }

    def engine(self, family: str) -> dict[str, Any]:
        """The engine words the model wires ``family`` through."""
        wired = self.model.wiring.get(family)
        assert wired is not None and wired.engine is not None, (
            f"{self.built.name} wires no engine words for family {family}"
        )
        return {
            key: getattr(wired.engine, key)
            for key in ("attribute", "index", "axis")
            if getattr(wired.engine, key) is not None
        }

    def elements(self, family: str, direction: str) -> dict[tuple[int, ...], list[str]]:
        """The elements each row of ``family`` binds in one direction, by ``DeviceList`` row."""
        members = self.groups[self.mapping.mapped(family)]
        words = self.engine(family)
        devices = {self.rows[device]: device for device in sorted(members) if device in self.rows}
        assert len(devices) == len([device for device in members if device in self.rows]), (
            f"{family} devices share a DeviceList row"
        )
        found: dict[str, list[str]] = {}
        for entry in self.built.wiring:
            if entry.get("direction") != direction or dict(entry.get("engine") or {}) != words:
                continue
            own = self.on_device.get(str(entry["address"]))
            pieces = [
                (str(piece["element"]), piece.get("device", own))
                for piece in entry.get("slices") or [{"element": entry["element"]}]
            ]
            for element, device in pieces:
                if device in members and element not in found.setdefault(device, []):
                    found[device].append(element)
        return {row: found[device] for row, device in devices.items() if device in found}


class Deck:
    """The built deck as the Middle Layer's saved deck holds it, addressed through the wiring.

    The build writes every cavity through ``RFCavityPass`` and builds a
    zero-length one into a deck that has none. The Middle Layer's model mode
    starts from the saved deck, so the replay puts the cavities back as
    ``state.cavity`` records them: a cavity the file calls ``Off`` passes
    through ``IdentityPass`` (``setcavity('Off')``), one it calls ``On`` through
    ``CavityPass``, and a deck with ``none`` loses the cavity the import built.
    A transport line holds no cavity and is replayed as written.
    """

    def __init__(self, built: BuiltModel, cavity: str) -> None:
        lattice = at.load_lattice(str(built.deck))
        if cavity == "none":
            added = recipes.cavities(lattice)
            assert all(float(lattice[index].Length) == 0 for index in added), (
                "the import built a cavity with length into a deck the Middle Layer held none in"
            )
            lattice = at.Lattice(
                [element for index, element in enumerate(lattice) if index not in added],
                energy=lattice.energy,
                periodicity=1,
            )
        else:
            recipes.set_cavity(lattice, cavity)
        self.lattice = lattice
        self.wiring = Wiring(built)
        self._where: dict[str, list[int]] = {}
        for index, element in enumerate(lattice):
            self._where.setdefault(str(element.FamName), []).append(index)

    def copy(self) -> Any:
        """A private copy of the replay deck, safe to step."""
        return copy.deepcopy(self.lattice)

    def where(self, element: str) -> int:
        """The replay-deck position of an element the wiring names."""
        found = self._where.get(element, [])
        assert len(found) == 1, f"the deck holds element {element} {len(found)} times"
        return found[0]

    def _bound(
        self, family: str, direction: str, device_list: Sequence[Sequence[float]]
    ) -> list[list[str]]:
        bound = self.wiring.elements(family, direction)
        rows = [_row(row) for row in device_list]
        missing = [row for row in rows if row not in bound]
        assert not missing, f"the wiring binds no {direction} element for {family} {missing}"
        return [bound[row] for row in rows]

    def positions(self, family: str, device_list: Sequence[Sequence[float]]) -> list[int]:
        """The replay-deck position of each listed monitor of ``family``, in listed order."""
        positions = []
        bound = self._bound(family, "read", device_list)
        for row, elements in zip(device_list, bound, strict=True):
            assert len(elements) == 1, f"{family} {row} reads {elements}"
            positions.append(self.where(elements[0]))
        return positions

    def kick_plane(self, family: str) -> int:
        """The ``KickAngle`` component a corrector family kicks, by its engine words."""
        engine = self.wiring.engine(family)
        assert engine.get("attribute") == "KickAngle", f"{family} is no corrector: {engine}"
        return int(engine["index"])

    def corrector(self, family: str, device: Sequence[float]) -> tuple[int, int]:
        """The replay-deck position a corrector device kicks, and its ``KickAngle`` plane.

        The Middle Layer's calculator kicks one element per device, so a device
        split over several elements has no replay.
        """
        (elements,) = self._bound(family, "write", [device])
        assert len(elements) == 1, f"{family} {_row(device)} kicks {elements}"
        return self.where(elements[0]), self.kick_plane(family)

    def stepped(self, family: str, facts: dict[str, Any]) -> Any:
        """A copy of the deck with every listed device of ``family`` stepped by its ``dK``.

        ``dK = k_per_amp * delta_resp_mat`` is the step the Middle Layer's
        family measurement lands on the model (``hw2physics`` of the nominal
        plus the width, less that of the nominal); ``setsp`` writes a device's
        strength to every element it drives.
        """
        lattice = self.copy()
        engine = self.wiring.engine(family)
        attribute, index = str(engine["attribute"]), int(engine["index"])
        bound = self._bound(family, "write", facts["device_list"])
        for elements, step in zip(bound, _step(facts), strict=True):
            for element in elements:
                target = lattice[self.where(element)]
                values = np.array(getattr(target, attribute), dtype=float)
                values[index] += step
                setattr(target, attribute, values)
        return lattice


def _deck(built: BuiltModel, reference: dict[str, Any]) -> Deck:
    return Deck(built, section(reference, "state")["cavity"])


def _step(facts: dict[str, Any]) -> np.ndarray:
    """``dK_n = k_per_amp_n * delta_resp_mat_n``, one per device."""
    return _flat(facts, "k_per_amp") * _flat(facts, "delta_resp_mat")


def _flat(facts: dict[str, Any], key: str) -> np.ndarray:
    """One per-device fact as a flat array, a scalar spread over the family."""
    members = len(facts["device_list"])
    values = np.asarray(facts[key], dtype=float).ravel()
    if values.size == 1:
        values = np.repeat(values, members)
    assert values.shape == (members,), f"{key} has {values.size} entries for {members} devices"
    return values


def _leff_weights(facts: dict[str, Any]) -> np.ndarray:
    """``Leff / mean(Leff)`` as ``measrespmat`` takes it, a zero length read as ``1/n``."""
    lengths = _flat(facts, "leff").copy()
    lengths[lengths == 0] = 1.0 / lengths.size
    return lengths / np.mean(lengths)


def _planes(reference: dict[str, Any]) -> dict[str, int]:
    """The orbit row (0 for ``x``, 2 for ``y``) each monitor family of the file reads."""
    monitors = section(reference, "dispersion")["monitors"]
    planes = {monitors["x"]["family"]: 0, monitors["y"]["family"]: 2}
    assert len(planes) == 2, f"the model file names one monitor family for both planes: {planes}"
    return planes


def _entries(block: dict[str, Any]) -> np.ndarray:
    return np.asarray(block["data"], dtype=float).reshape(
        len(block["monitor"]["device_list"]), len(block["actuator"]["device_list"])
    )


def _assert_matrix(computed: Sequence[np.ndarray], blocks: Sequence[dict[str, Any]]) -> None:
    """Every entry within ``ORM_RMS_FRACTION`` of the rms of every block's entries."""
    stated = np.concatenate([_entries(block).ravel() for block in blocks])
    rms = float(np.sqrt(np.mean(stated[np.isfinite(stated)] ** 2)))
    band = ORM_RMS_FRACTION * rms
    offenders = []
    for matrix, block in zip(computed, blocks, strict=True):
        expected = _entries(block)
        error = np.abs(matrix - expected)
        for row, column in zip(*np.nonzero(~(error <= band)), strict=True):
            offenders.append(
                f"{block['monitor']['family']}{block['monitor']['device_list'][row]} / "
                f"{block['actuator']['family']}{block['actuator']['device_list'][column]}: "
                f"{matrix[row, column]!r} vs {expected[row, column]!r}"
            )
    assert not offenders, (
        f"{len(offenders)} orbit-response entries more than {band:.3g} "
        f"({ORM_RMS_FRACTION:g} of the rms) off: {offenders[:10]}"
    )


# ---------------------------------------------------------------------------
# The engine on the built deck
# ---------------------------------------------------------------------------


@MODELS
def test_the_engine_builds_the_model_on_the_deck_the_build_wrote(
    tree: str, stem: str, mml_built: Built
) -> None:
    """The model constructs from its wiring: a periodic deck solves in 6D, a line in one pass."""
    from lume_pyat.simulator import PyATSimulator

    from osprey.simulation.engines import pyat as engine
    from osprey.simulation.engines.pyat_single_pass import SinglePassSimulator

    built = mml_built(tree, stem)
    model = engine.build(built.name, built.wiring, built.deck, built.settings)

    assert isinstance(model.simulator, PyATSimulator)
    if (tree, stem) in MATLAB_LINES:
        assert model.solve == "single_pass"
        assert isinstance(model.simulator, SinglePassSimulator)
    else:
        assert model.solve == "periodic"
        assert not isinstance(model.simulator, SinglePassSimulator)
        assert model.simulator.lattice.is_6d


# ---------------------------------------------------------------------------
# Storage rings
# ---------------------------------------------------------------------------


def _harmonic_orbit(lattice: Any, circumference: float, refpts: list[int]) -> np.ndarray:
    """``getpvmodel``'s closed orbit with the cavity off: the synchronous orbit at the
    path-length change the deck's RF offset from ``h * c / L`` implies."""
    cavity = lattice[recipes.cavities(lattice)[0]]
    frequency, harmonic = float(cavity.Frequency), float(cavity.HarmNumber)
    offset = frequency - recipes.C_LIGHT * harmonic / circumference
    dct = -recipes.C_LIGHT * offset * harmonic / frequency**2
    return recipes.findsyncorbit(lattice, dct, refpts)[1]


@RINGS
def test_ring_state_replays_the_model_file(tree: str, stem: str, mml_built: Built) -> None:
    """Circumference, harmonic number, ``mcf`` and the closed orbit at every monitor.

    The orbit is ``getpvmodel``'s: with no cavity the synchronous orbit at zero
    path-length change, with the cavity off the synchronous orbit at the deck's
    RF offset, with it on (or radiation on) the 6D orbit.
    """
    reference = model_reference(tree, stem)
    state = section(reference, "state")
    deck = _deck(mml_built(tree, stem), reference)
    lattice = deck.lattice

    assert recipes.circumference(lattice) == pytest.approx(
        float(state["circumference_m"]), rel=CIRCUMFERENCE_RTOL
    )
    assert recipes.mcf(lattice) == pytest.approx(float(state["mcf"]), rel=MCF_RTOL)
    cavities = recipes.cavities(lattice)
    if cavities:
        assert float(lattice[cavities[0]].HarmNumber) == float(state["harmonic_number"])

    everywhere = list(range(len(lattice) + 1))
    if state["cavity"] == "none":
        orbit = recipes.findsyncorbit(lattice, 0.0, everywhere)[1]
    elif state["cavity"] == "Off":
        orbit = _harmonic_orbit(lattice, float(state["circumference_m"]), everywhere)
    else:
        orbit = recipes.findorbit6(lattice, everywhere)[1]
    for family, row in _planes(reference).items():
        block = state["orbit"][family]
        where = deck.positions(family, block["device_list"])
        np.testing.assert_allclose(
            orbit[row, where],
            np.asarray(block["physics"], dtype=float),
            rtol=0,
            atol=ORBIT_ATOL_M,
            err_msg=f"closed orbit at {family}",
        )


def _assert_tunes(lattice: Any, reference: dict[str, Any]) -> None:
    """``tunechrom(THERING, 0)`` and ``modeltune``, each against its recorded pair."""
    tune = section(reference, "tune")
    np.testing.assert_allclose(
        recipes.tunechrom(copy.deepcopy(lattice), 0.0),
        tune["fixed_momentum"],
        rtol=0,
        atol=TUNE_ATOL,
        err_msg="fixed-momentum tunes",
    )
    np.testing.assert_allclose(
        recipes.modeltune(copy.deepcopy(lattice)),
        tune["cavity_on"],
        rtol=0,
        atol=TUNE_ATOL,
        err_msg=f"cavity-on tunes ({tune['method']})",
    )


@RINGS
def test_tunes_replay_the_model_file(tree: str, stem: str, mml_built: Built) -> None:
    """Fixed-momentum tunes by ``tunechrom``; the cavity-on pair by ``modeltune``.

    ``modeltune`` switches the cavities on and reads ``getnusympmat`` of
    ``findm66``, and falls back to ``twissring`` on a deck with none.
    """
    reference = model_reference(tree, stem)
    deck = _deck(mml_built(tree, stem), reference)
    expected = {"findm66": bool(recipes.cavities(deck.lattice)), "twissring": True}
    method = section(reference, "tune")["method"]
    assert method in expected, f"the model file records tune method {method!r}"
    _assert_tunes(deck.lattice, reference)


def _chromaticity_step(lattice: Any, delta_rf_hz: float, compaction: float) -> float:
    """The momentum step ``modelchro`` differentiates the tunes over.

    With a cavity it steps the RF by ``delta_rf_hz``, which is ``dp = delta_rf_hz
    / (f_RF * mcf)``; without one ``tunechrom`` steps the momentum itself.
    """
    cavities = recipes.cavities(lattice)
    if not cavities:
        return recipes.TUNECHROM_STEP
    return abs(delta_rf_hz) / (float(lattice[cavities[0]].Frequency) * compaction)


@RINGS
def test_chromaticity_replays_the_model_file(tree: str, stem: str, mml_built: Built) -> None:
    """``modelchro`` over the recorded RF step, in physics units and, with a cavity, hardware.

    With a cavity: ``getnusympmat(findm66)`` at the cavity frequency and one
    ``delta_rf_hz`` above it, the change scaled by ``-mcf * RF0`` with ``mcf``
    re-read on the lattice as ``modelchro`` holds it; in hardware the change per
    MHz of RF. Without one: ``tunechrom``'s 4D chromaticity, physics only.
    """
    reference = model_reference(tree, stem)
    chrom = section(reference, "chromaticity")
    state = section(reference, "state")
    deck = _deck(mml_built(tree, stem), reference)
    method = chrom["method"]
    has_cavity = bool(recipes.cavities(deck.lattice))
    assert method == ("findm66" if has_cavity else "tunechrom"), method

    delta_rf = float(chrom["delta_rf_hz"])
    physics = recipes.modelchro(deck.copy(), delta_rf)
    dp = _chromaticity_step(deck.lattice, delta_rf, float(state["mcf"]))
    band = 2.0 * TUNE_DIFFERENCE_ATOL / dp
    np.testing.assert_allclose(
        physics, chrom["physics"], rtol=0, atol=band, err_msg=f"physics chromaticity ({method})"
    )
    if refusal(chrom["hardware"]) is None:
        step_hw = float(chrom["delta_rf_hw"])
        hardware = recipes.modelchro(deck.copy(), delta_rf, hardware_step=step_hw)
        np.testing.assert_allclose(
            hardware,
            chrom["hardware"],
            rtol=0,
            atol=2.0 * TUNE_DIFFERENCE_ATOL / abs(step_hw),
            err_msg=f"hardware chromaticity ({method})",
        )


def _replayed_dispersion(deck: Deck, block: dict[str, Any], state: dict[str, Any]) -> np.ndarray:
    """``modeldisp``'s dispersion at every deck position, by the recorded method.

    ``findsyncorbit`` (cavity off) solves the synchronous orbit at ``dct =
    -C * (+-delta_rf + rf_change) * h / f**2 / 2`` -- the deck's RF offset
    halved with the step; ``findorbit6`` (cavity on) the 6D orbit at the cavity
    frequency +- half the step; ``findorbit4`` (no cavity) the 4D orbit at
    ``-+dp/2``, ``dp = delta_rf / mcf / (h * C / L)``. The orbit difference over
    ``delta_rf`` is scaled by ``-f * mcf``.

    Returns:
        The dispersion, ``(rows, positions)``.
    """
    lattice = deck.copy()
    everywhere = list(range(len(lattice) + 1))
    delta_rf = float(block["delta_rf_hz"])
    frequency = float(block["f_cavity"])
    compaction = float(block["mcf"])
    harmonic = float(state["harmonic_number"])
    method = block["method"]
    if method == "findsyncorbit":
        offset = float(block["rf_change_hz"])
        c = recipes.C_LIGHT

        def orbit(sign: float) -> np.ndarray:
            dct = (-c * (sign * delta_rf + offset) * harmonic / frequency**2) / 2.0
            return recipes.findsyncorbit(lattice, dct, everywhere)[1]

        plus, minus = orbit(+1.0), orbit(-1.0)
    elif method == "findorbit6":
        index = recipes.cavities(lattice)
        held = [float(lattice[i].Frequency) for i in index]

        def orbit(sign: float) -> np.ndarray:
            for i, value in zip(index, held, strict=True):
                lattice[i].Frequency = value + sign * delta_rf / 2.0
            return recipes.findorbit6(lattice, everywhere)[1]

        plus, minus = orbit(+1.0), orbit(-1.0)
    elif method == "findorbit4":
        dp = delta_rf / compaction / (harmonic * recipes.C_LIGHT / float(state["circumference_m"]))
        plus = recipes.findorbit4(lattice, -dp / 2.0, everywhere)[1]
        minus = recipes.findorbit4(lattice, dp / 2.0, everywhere)[1]
    else:
        pytest.fail(f"the model file records dispersion method {method!r}, which no recipe replays")
    return (plus - minus) / delta_rf * (-frequency * compaction)


@RINGS
def test_dispersion_replays_the_model_file(tree: str, stem: str, mml_built: Built) -> None:
    """``modeldisp`` at every monitor, by the recorded method, the deck's RF offset halved."""
    reference = model_reference(tree, stem)
    block = section(reference, "dispersion")
    state = section(reference, "state")
    deck = _deck(mml_built(tree, stem), reference)
    span = _replayed_dispersion(deck, block, state)
    dp_span = abs(float(block["delta_rf_hz"])) / (float(block["f_cavity"]) * float(block["mcf"]))
    band = DISPERSION_ORBITS * recipes.ORBIT_CONVERGENCE / dp_span
    for plane, row in (("x", 0), ("y", 2)):
        monitors = block["monitors"][plane]
        where = deck.positions(monitors["family"], monitors["device_list"])
        np.testing.assert_allclose(
            span[row, where],
            np.asarray(block["physics"][plane], dtype=float),
            rtol=0,
            atol=band,
            err_msg=f"dispersion {plane} ({block['method']})",
        )


@RINGS
def test_orbit_response_replays_the_model_file(tree: str, stem: str, mml_built: Built) -> None:
    """``measbpmresp('Model')``: the Linear calculator at fixed path length, per radian."""
    reference = model_reference(tree, stem)
    response = section(reference, "orbit_response")
    assert (response["calculator"], response["closed_orbit_type"]) == (
        "Linear",
        "FixedPathLength",
    ), "a periodic deck's model matrix is the Linear calculator at fixed path length"
    deck = _deck(mml_built(tree, stem), reference)
    planes = _planes(reference)
    compaction = recipes.mcf(deck.lattice)
    blocks = response["physics"]
    computed = []
    for block in blocks:
        monitors = deck.positions(block["monitor"]["family"], block["monitor"]["device_list"])
        correctors = [
            deck.corrector(block["actuator"]["family"], device)
            for device in block["actuator"]["device_list"]
        ]
        # measbpmresp folds each monitor's GCR (gain, crunch, roll) and each
        # corrector's Roll into the matrix; the port folds neither, so it holds
        # only a deck whose values leave the matrix untouched.
        for position in monitors:
            gcr = np.ravel(getattr(deck.lattice[position], "GCR", [1, 1, 0, 0])).astype(float)
            assert np.array_equal(gcr, [1.0, 1.0, 0.0, 0.0]), f"monitor {position} GCR {gcr}"
        for position, _plane in correctors:
            roll = np.ravel(getattr(deck.lattice[position], "Roll", [0, 0])).astype(float)
            assert not roll.any(), f"corrector {position} Roll {roll}"
        replay = recipes.loco_linear(deck.copy(), correctors, monitors, compaction)
        computed.append(replay[:, :, planes[block["monitor"]["family"]] // 2].T)
    _assert_matrix(computed, blocks)


def _answered(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The families of a response section the Middle Layer answered, or skip naming the refusals."""
    families = block["families"]
    answered = {name: facts for name, facts in families.items() if refusal(facts) is None}
    if not answered:
        refused = {name: refusal(facts) for name, facts in families.items()}
        pytest.skip(f"the model file refused every family: {refused}")
    return answered


def _replayed_tune_response(
    deck: Deck, facts: dict[str, Any], family: str
) -> dict[str, tuple[np.ndarray, float]]:
    """``meastuneresp(family, [], [], 'Model', units)``, summed over the family.

    The whole family steps once by its ``dK``, unipolar, and ``modeltune``
    reads the tunes before and after; ``meastuneresp`` sums the per-device
    columns. In hardware units column ``n`` is ``dQ / sum(dK * w) * k_n *
    w_n`` (``w = Leff / mean(Leff)``). In physics units ``measrespmat`` picks
    its step conversion by the units the family stores, not the units asked
    for: a tune corrector stores hardware units, so it reads the K values it
    stepped as amperes, normalises by ``DPb = physics_step_read_as_amps`` and
    scales column ``n`` by ``DPb_n * w_n / dK_n``.

    Returns:
        Per unit set, the answer and the factor the tune change is multiplied
        by, which carries the tune rounding into the answer.
    """
    before = recipes.modeltune(deck.copy())
    after = recipes.modeltune(deck.stepped(family, facts))
    change = (after - before + 0.5) % 1.0 - 0.5
    step = _step(facts)
    weights = _leff_weights(facts)
    read_as_amps = _flat(facts, "physics_step_read_as_amps")
    physics = float(np.sum(read_as_amps * weights / step)) / float(np.sum(read_as_amps * weights))
    hardware = float(np.sum(_flat(facts, "k_per_amp") * weights)) / float(np.sum(step * weights))
    return {
        "physics": (change * physics, abs(physics)),
        "hardware": (change * hardware, abs(hardware)),
    }


def _replayed_chromaticity_response(
    deck: Deck, facts: dict[str, Any], family: str, reference: dict[str, Any]
) -> dict[str, tuple[np.ndarray, float]]:
    """``measchroresp(family, [], [], [], 'Model', units)``: the family value per device.

    The whole family steps once by its ``dK``, unipolar, and ``measchro``
    reads the chromaticity before and after. In model mode ``measchro``
    answers ``modelchro('Physics')`` over its own 1 Hz step, which re-reads the
    momentum compaction on the lattice as it stands; in hardware units it divides
    that by ``-RF0 * getmcf('Model')`` -- the RF frequency in the RF family's
    unit and ``mcf`` re-read on the stepped lattice as the deck holds its cavity
    -- and ``measchroresp`` scales device ``n`` by ``k_n``. The change is
    divided by ``sum(dK)``. A deck with no cavity takes its ``RF0`` from a
    Middle Layer constant the model file does not record, so its hardware
    answer is not replayed.

    Returns:
        Per unit set, the ``(2, members)`` answer and the factor a
        chromaticity's rounding is multiplied by on its way into the answer.
    """
    lattices = (deck.copy(), deck.stepped(family, facts))
    total = float(np.sum(_step(facts)))
    members = len(facts["device_list"])
    physics = [recipes.modelchro(lattice) for lattice in lattices]
    answers = {
        "physics": (
            np.repeat(((physics[1] - physics[0]) / total)[:, None], members, axis=1),
            1.0 / abs(total),
        )
    }
    cavities = recipes.cavities(deck.lattice)
    if cavities:
        chrom = section(reference, "chromaticity")
        rf0 = (
            float(deck.lattice[cavities[0]].Frequency)
            * float(chrom["delta_rf_hw"])
            / float(chrom["delta_rf_hz"])
        )
        hardware = [
            value / (-rf0 * recipes.mcf(lattice))
            for value, lattice in zip(physics, lattices, strict=True)
        ]
        scale = _flat(facts, "k_per_amp") / total
        answers["hardware"] = (
            (hardware[1] - hardware[0])[:, None] * scale[None, :],
            float(np.max(np.abs(scale))) / abs(rf0 * float(section(reference, "state")["mcf"])),
        )
    return answers


def _stated(block: dict[str, Any], units: str, family: str, members: int) -> np.ndarray:
    """A recorded family answer as the replay lays it out.

    ``meastuneresp`` answers one summed pair; ``measchroresp`` one ``(x, y)``
    column per device.
    """
    array = np.asarray(block[units][family], dtype=float)
    if array.shape == (2,):
        return array
    columns = array if array.shape == (2, members) else array.T
    assert columns.shape == (2, members), f"{units} {family}: shape {array.shape}"
    return columns


@RINGS
@RESPONSES
def test_family_response_replays_the_model_file(
    tree: str, stem: str, name: str, mml_built: Built
) -> None:
    """Each answered family stepped whole, as ``meastuneresp``/``measchroresp`` step it.

    Both unit sets are replayed. The band is the rounding of the tune
    difference behind the answer -- two tunes for a tune response, two
    chromaticities, each two tunes over ``measchro``'s 1 Hz momentum step, for
    a chromaticity response -- carried through the factor the replay
    multiplies it by.
    """
    reference = model_reference(tree, stem)
    block = section(reference, name)
    deck = _deck(mml_built(tree, stem), reference)
    chromaticity = (
        2.0
        * TUNE_DIFFERENCE_ATOL
        / _chromaticity_step(deck.lattice, 1.0, float(section(reference, "state")["mcf"]))
    )
    failures = []
    for family, facts in _answered(block).items():
        members = len(facts["device_list"])
        if name == "tune_response":
            answers = _replayed_tune_response(deck, facts, family)
            rounding = 2.0 * TUNE_DIFFERENCE_ATOL
        else:
            answers = _replayed_chromaticity_response(deck, facts, family, reference)
            rounding = 2.0 * chromaticity
        for units, (computed, factor) in answers.items():
            expected = _stated(block, units, family, members)
            band = rounding * factor
            error = np.abs(computed - expected)
            if not np.all(error <= band):
                failures.append(
                    f"{family} {units}: off by {error.max():.3g} > {band:.3g} "
                    f"(largest stated {np.max(np.abs(expected)):.6g})"
                )
    assert not failures, f"{name}: " + "; ".join(failures)


# ---------------------------------------------------------------------------
# Transport lines
# ---------------------------------------------------------------------------


def _exported_positions(
    built: BuiltModel, family: str, device_list: Sequence[Sequence[float]]
) -> list[int]:
    """The deck position of each listed device of a line family, by the export's ``ATIndex``.

    The replay places a line family's listed devices where the export's
    ``AT.ATIndex`` (one-based) puts them on the deck the build wrote, which
    keeps the export's element order.
    """
    from tests.facility._mml_built import FIXTURES

    ao = json.loads((FIXTURES / built.tree / f"{built.stem}.ao.json").read_text(encoding="utf-8"))
    body = ao[family]
    rows = [_row(row) for row in np.atleast_2d(body["DeviceList"])]
    index = np.ravel(body["AT"]["ATIndex"])
    return [int(index[rows.index(_row(row))]) - 1 for row in device_list]


def _launch(deck: Deck) -> tuple[np.ndarray, dict[str, Any]]:
    """The launch orbit and twiss the line's first element carries, as ``TwissData``."""
    twiss = deck.lattice[0].TwissData
    start = np.zeros(6)
    start[:4] = np.ravel(twiss["ClosedOrbit"]).astype(float)
    start[4] = float(twiss["dP"])
    start[5] = float(twiss["dL"])
    return start, twiss


@LINES
def test_transport_dispersion_replays_the_model_file(
    tree: str, stem: str, mml_built: Built
) -> None:
    """``modeltwiss('Eta')``: ``twissline``'s dispersion from the first element's twiss."""
    reference = model_reference(tree, stem)
    block = section(reference, "dispersion")
    assert block["method"] == "twiss", block["method"]
    built = mml_built(tree, stem)
    deck = _deck(built, reference)
    _start, twiss = _launch(deck)
    for plane, row in (("x", 0), ("y", 2)):
        monitors = block["monitors"][plane]
        where = _exported_positions(built, monitors["family"], monitors["device_list"])
        eta = recipes.twissline_dispersion(deck.lattice, twiss, where)
        np.testing.assert_allclose(
            eta[row],
            np.asarray(block["physics"][plane], dtype=float),
            rtol=0,
            atol=TRANSPORT_DISPERSION_ATOL_M,
            err_msg=f"transport dispersion {plane}",
        )


@LINES
def test_transport_orbit_response_replays_the_model_file(
    tree: str, stem: str, mml_built: Built
) -> None:
    """``measbpmresp('Model')`` on a line: the Full calculator at fixed momentum, bipolar."""
    reference = model_reference(tree, stem)
    response = section(reference, "orbit_response")
    assert (response["calculator"], response["closed_orbit_type"]) == (
        "Full",
        "FixedMomentum",
    ), "a transport line's model matrix is the Full calculator at fixed momentum"
    built = mml_built(tree, stem)
    deck = _deck(built, reference)
    start, _twiss = _launch(deck)
    planes = _planes(reference)
    computed = []
    for block in response["physics"]:
        family = block["actuator"]["family"]
        plane = deck.kick_plane(family)
        monitors = _exported_positions(
            built, block["monitor"]["family"], block["monitor"]["device_list"]
        )
        positions = [
            deck.corrector(family, device)[0] for device in block["actuator"]["device_list"]
        ]
        kicks = np.broadcast_to(np.asarray(block["actuator_delta"], dtype=float), (len(positions),))
        replay = recipes.loco_transport_full(
            deck.lattice, start, [(p, plane) for p in positions], kicks, monitors
        )
        computed.append(replay[:, :, planes[block["monitor"]["family"]] // 2].T)
    _assert_matrix(computed, response["physics"])


# ---------------------------------------------------------------------------
# Non-vacuity and addressing
# ---------------------------------------------------------------------------


@RINGS
def test_a_shifted_quadrupole_fails_the_tune_replay(tree: str, stem: str, mml_built: Built) -> None:
    """One quadrupole's gradient moved by 1e-3 is far outside the tune band."""
    reference = model_reference(tree, stem)
    deck = _deck(mml_built(tree, stem), reference)
    lattice = deck.copy()
    quadrupoles = [
        index for index, element in enumerate(lattice) if isinstance(element, at.Quadrupole)
    ]
    assert quadrupoles, "the built deck carries no quadrupole"
    element = lattice[quadrupoles[0]]
    element.PolynomB = np.array(element.PolynomB, dtype=float)
    element.PolynomB[1] += QUAD_SHIFT
    with pytest.raises(AssertionError, match="tunes"):
        _assert_tunes(lattice, reference)


def test_monitors_listed_out_of_ring_order_read_in_listed_order(mml_built: Built) -> None:
    """A device list that does not start at the deck's first monitor still reads row by row."""
    reference = model_reference(*MATLAB_RINGS[0])
    deck = _deck(mml_built(*MATLAB_RINGS[0]), reference)
    monitors = section(reference, "dispersion")["monitors"]["x"]
    listed = monitors["device_list"]
    forward = deck.positions(monitors["family"], listed)
    rotated = deck.positions(monitors["family"], listed[1:] + listed[:1])
    assert rotated == forward[1:] + forward[:1]
