"""The pyat engine's single-pass solve, over a three-cell synthetic line and an imported one.

A ``single_pass`` model tracks one particle once through its line from the
``twiss_in`` its settings state: its monitors read that pass, its beta
functions are propagated from ``twiss_in``, and it serves no tunes or
chromaticity. The NSLS-II transport line, imported and built from its Middle
Layer export, answers each corrector's step as the Middle Layer's transport
calculator does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

at = pytest.importorskip("at")

import numpy as np  # noqa: E402
from lume_pyat.exceptions import OrbitSolveError  # noqa: E402

from osprey.facility.errors import FacilityBuildError  # noqa: E402
from osprey.simulation.engines import pyat as engine  # noqa: E402
from osprey.simulation.engines import pyat_single_pass  # noqa: E402
from osprey.simulation.engines.pyat_model import (  # noqa: E402
    BETA_AT_MONITORS,
    CHROMATICITY,
    ORBIT_AT_MONITORS,
    TUNES,
)
from osprey.simulation.engines.pyat_single_pass import SinglePassSimulator  # noqa: E402
from tests.facility._mml_built import FIXTURES, BuiltModel, mml_built  # noqa: E402, F401
from tests.facility._model_reference import (  # noqa: E402
    ORM_RMS_FRACTION,
    model_reference,
    section,
)
from tests.services.mml import _mml_recipes as recipes  # noqa: E402

MODEL = "LINE"
CELLS = (1, 2, 3)
TWISS = {
    "beta": [5.0, 3.0],
    "alpha": [0.1, -0.2],
    "closed_orbit": [1.0e-4, 2.0e-5, -5.0e-5, 1.0e-5],
}
SETTINGS = {"pyat": {"solve": "single_pass", "twiss_in": TWISS}}
KICK = "COR_1:X:SP"
QUAD = "QF_2:SP"


def _elements() -> list[Any]:
    elements: list[Any] = []
    for cell in CELLS:
        elements += [
            at.Drift(f"DA_{cell}", 0.5),
            at.Quadrupole(f"QF_{cell}", 0.3, 1.2),
            at.Drift(f"DB_{cell}", 0.5),
            at.Corrector(f"COR_{cell}", 0.0, [0.0, 0.0]),
            at.Monitor(f"BPM_{cell}"),
            at.Drift(f"DC_{cell}", 0.5),
            at.Quadrupole(f"QD_{cell}", 0.3, -1.1),
        ]
    return elements


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    """Three FODO cells, a corrector and a monitor in each, saved as a line."""
    lattice = at.Lattice(_elements(), energy=3e9, particle="electron", periodicity=1)
    path = tmp_path / "line.json"
    at.save_lattice(lattice, str(path))
    return path


def _monitor(cell: int, axis: str) -> dict[str, Any]:
    address = f"BPM_{cell}:{axis.upper()}"
    return {
        "id": f"{MODEL}/{address}",
        "address": address,
        "direction": "read",
        "element": f"BPM_{cell}",
        "engine": {"axis": axis},
        "unit": "m",
    }


WIRING: list[dict[str, Any]] = [
    *(_monitor(cell, axis) for cell in CELLS for axis in ("x", "y")),
    {
        "id": f"{MODEL}/{KICK}",
        "address": KICK,
        "direction": "write",
        "element": "COR_1",
        "engine": {"attribute": "KickAngle", "index": 0},
        "default": 0.0,
        "unit": "rad",
    },
    {
        "id": f"{MODEL}/{QUAD}",
        "address": QUAD,
        "direction": "write",
        "element": "QF_2",
        "engine": {"attribute": "PolynomB", "index": 1},
        "default": 1.2,
    },
]

MONITORS = [record["address"] for record in WIRING if record["direction"] == "read"]


def _build(deck: Path, settings: Any = SETTINGS, wiring: Any = WIRING) -> Any:
    return engine.build(MODEL, wiring, deck, settings)


def _tracked(deck: Path, kick: float = 0.0) -> np.ndarray:
    """The pass pyAT itself tracks: one ``(x, y)`` row per monitor."""
    lattice = at.load_lattice(str(deck))
    lattice[lattice.get_uint32_index("COR_1")[0]].KickAngle = [kick, 0.0]
    monitors = lattice.get_uint32_index(at.Monitor)
    prepared = engine.prepare(deck, SETTINGS)
    r_out = at.lattice_track(
        lattice, prepared.twiss_in["closed_orbit"].reshape(6, 1), refpts=monitors
    )[0]
    return np.array([[r_out[0, 0, row, 0], r_out[2, 0, row, 0]] for row in range(len(monitors))])


def _readings(model: Any) -> np.ndarray:
    values = model.get(MONITORS)
    return np.array([[values[f"BPM_{cell}:X"], values[f"BPM_{cell}:Y"]] for cell in CELLS])


class TestBuild:
    def test_a_three_cell_line_builds_on_the_single_pass_simulator(self, deck: Path):
        model = _build(deck)
        assert isinstance(model.simulator, SinglePassSimulator)
        assert model.solve == "single_pass"

    def test_it_has_no_tunes_or_chromaticity_variable(self, deck: Path):
        model = _build(deck)
        assert TUNES not in model.supported_variables
        assert CHROMATICITY not in model.supported_variables
        with pytest.raises(ValueError, match="'tunes' is not supported"):
            model.get([TUNES])

    def test_its_monitors_read_the_tracked_pass(self, deck: Path):
        model = _build(deck)
        np.testing.assert_allclose(_readings(model), _tracked(deck), rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            model.get([ORBIT_AT_MONITORS])[ORBIT_AT_MONITORS], _tracked(deck), rtol=0, atol=1e-15
        )

    def test_a_kick_moves_the_downstream_monitors_as_pyat_tracks_them(self, deck: Path):
        model = _build(deck)
        model.set({KICK: 1.0e-4})
        np.testing.assert_allclose(_readings(model), _tracked(deck, 1.0e-4), rtol=0, atol=1e-15)
        assert _readings(model)[1, 0] != pytest.approx(_tracked(deck)[1, 0], abs=1e-9)

    def test_beta_is_propagated_from_twiss_in(self, deck: Path):
        model = _build(deck)
        lattice = at.load_lattice(str(deck))
        twiss_in = engine.prepare(deck, SETTINGS).twiss_in
        expected = at.get_optics(
            lattice, refpts=lattice.get_uint32_index(at.Monitor), twiss_in=twiss_in
        )[2].beta
        np.testing.assert_allclose(
            model.get([BETA_AT_MONITORS])[BETA_AT_MONITORS], expected, rtol=0, atol=1e-12
        )

    def test_a_tune_record_on_a_single_pass_model_is_refused(self, deck: Path):
        tune = {
            "id": f"{MODEL}/TUNE:X",
            "address": "TUNE:X",
            "direction": "read",
            "engine": {"attribute": "tune", "axis": "x"},
        }
        with pytest.raises(ValueError, match="serves no tunes or chromaticity"):
            _build(deck, wiring=[*WIRING, tune])

    def test_twiss_in_on_a_periodic_model_is_refused(self, deck: Path):
        with pytest.raises(FacilityBuildError) as stopped:
            _build(deck, settings={"pyat": {"twiss_in": TWISS}})
        assert stopped.value.kind == "engine-invalid"
        assert "pyat.twiss_in is set on a periodic model" in str(stopped.value)


class TestFailedSolve:
    def test_a_raising_get_optics_leaves_inputs_unchanged(
        self, deck: Path, monkeypatch: pytest.MonkeyPatch
    ):
        model = _build(deck)
        before = model.get([KICK, QUAD, *MONITORS])
        solution = model.simulator.last_solution

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("no optics here")

        monkeypatch.setattr(pyat_single_pass.at, "get_optics", refuse)
        with pytest.raises(
            OrbitSolveError, match=r"^optics solve raised ValueError: no optics here$"
        ):
            model.set({KICK: 1.0e-4})

        assert model.get([KICK, QUAD, *MONITORS]) == before
        assert model.simulator.last_solution is solution
        assert model.simulator.element("COR_1").KickAngle[0] == 0.0

    def test_a_lost_particle_names_where_pyat_lost_it(self, deck: Path):
        model = _build(deck)
        before = model.get([KICK, *MONITORS])
        with pytest.raises(OrbitSolveError) as lost:
            model.set({KICK: 5.0})
        text = engine.error_text(lost.value)
        assert re.fullmatch(
            r"lost at element \d+ \([A-Z]+_\d\) turn 0: \[(-?[0-9.e+-]+, ){5}-?[0-9.e+-]+\]", text
        ), text
        assert model.get([KICK, *MONITORS]) == before

    def test_the_loss_text_renders_the_loss_map(self, deck: Path):
        lattice = at.load_lattice(str(deck))
        index = lattice.get_uint32_index("COR_1")[0]
        lattice[index].KickAngle = [5.0, 0.0]
        loss_map = at.lattice_track(lattice, np.zeros((6, 1)), losses=True)[2]["loss_map"]
        element = int(loss_map.elem[0])
        coord = ", ".join(f"{value:.6g}" for value in loss_map.coord[0])
        assert pyat_single_pass.loss_text(lattice, loss_map) == (
            f"lost at element {element} ({lattice[element].FamName}) turn 0: [{coord}]"
        )

    def test_a_surviving_particle_has_no_loss_text(self, deck: Path):
        lattice = at.load_lattice(str(deck))
        loss_map = at.lattice_track(lattice, np.zeros((6, 1)), losses=True)[2]["loss_map"]
        assert pyat_single_pass.loss_text(lattice, loss_map) is None


# ---------------------------------------------------------------------------
# The imported transport line
# ---------------------------------------------------------------------------

LTB = ("nsls2", "nsls2.ltb")

#: The orbit row (0 for ``x``, 2 for ``y``) each monitor family of the line reads.
LTB_PLANES = {"BPMx": 0, "BPMy": 2}


def _ltb_row(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _ltb_monitor_positions(
    built: BuiltModel, family: str, device_list: Sequence[Sequence[float]]
) -> list[int]:
    """The deck position of each listed monitor, by the export's one-based ``AT.ATIndex``."""
    ao = json.loads((FIXTURES / built.tree / f"{built.stem}.ao.json").read_text(encoding="utf-8"))
    body = ao[family]
    rows = [_ltb_row(row) for row in np.atleast_2d(body["DeviceList"])]
    index = np.ravel(body["AT"]["ATIndex"])
    return [int(index[rows.index(_ltb_row(row))]) - 1 for row in device_list]


def _ltb_setpoints(built: BuiltModel, family: str) -> dict[tuple[int, ...], dict[str, Any]]:
    """The setpoint each ``DeviceList`` row of a corrector family writes, from the wiring."""
    from osprey.facility.layers.mml.mapping import MAPPING_FILE, read_mapping

    mapping = read_mapping(built.facility / MAPPING_FILE)
    (model,) = [model for model in mapping.models.values() if model.name == built.name]
    words = model.wiring[family].engine
    assert words.attribute == "KickAngle", f"{family} is no corrector"
    engine_words = {"attribute": "KickAngle", "index": int(words.index)}
    (members,) = [
        set(group.get("members", []))
        for group in built.document["groups"]
        if str(group["id"]) == mapping.mapped(family)
    ]
    rows = {
        str(device["id"]): _ltb_row(device["attributes"]["DeviceList"])
        for device in built.document["devices"]
        if device.get("model") == built.name and str(device["id"]) in members
    }
    on_device = {
        str(channel["id"]): (channel.get("on") or {}).get("device")
        for channel in built.document["channels"]
    }
    found: dict[tuple[int, ...], dict[str, Any]] = {}
    for entry in built.wiring:
        device = on_device.get(str(entry["address"]))
        if (
            entry.get("direction") == "write"
            and dict(entry.get("engine") or {}) == engine_words
            and device in rows
        ):
            assert rows[device] not in found, f"{family} {rows[device]} has two setpoints"
            found[rows[device]] = entry
    return found


def _ltb_hardware(entry: dict[str, Any], physics: float) -> float:
    """The setpoint value the wiring's inverse calibration gives for a physics value."""
    from osprey.simulation.engines.calibration import curve_from_record, to_hardware

    calibration = entry["calibration"]
    return to_hardware(
        curve_from_record(calibration["curve"]),
        curve_from_record(calibration["inverse"]),
        physics,
    )


@pytest.fixture
def ltb_line(mml_built: Callable[[str, str], BuiltModel]) -> BuiltModel:  # noqa: F811
    """The LTB model of the session's one nsls2 build."""
    return mml_built(*LTB)


@pytest.mark.xdist_group("mml_built")
class TestImportedLtbLine:
    """The imported NSLS-II LTB line at ``solve: single_pass``, built through ``mml_built``.

    The line's monitors are unwired in this fixture, so the plug-in's deck
    holds no monitor element and serves no orbit reading. Each corrector is
    stepped through the plug-in's own setpoint, and the plug-in's lattice is
    tracked once from its normalised ``twiss_in`` launch orbit to the deck
    positions the export's ``AT.ATIndex`` gives for the listed ``BPMx`` and
    ``BPMy`` devices.
    """

    def test_the_ltb_line_builds_single_pass_from_the_imported_twiss_in(self, ltb_line: BuiltModel):
        built = ltb_line
        model = engine.build(built.name, built.wiring, built.deck, built.settings)
        assert isinstance(model.simulator, SinglePassSimulator)
        stated = built.settings["pyat"]["twiss_in"]
        launch = model.simulator.twiss_in["closed_orbit"]
        assert launch.shape == (6,)
        np.testing.assert_array_equal(launch[:4], np.asarray(stated["closed_orbit"], dtype=float))
        np.testing.assert_array_equal(launch[4:], np.zeros(2))

    def test_the_ltb_corrector_columns_match_the_transport_recipe(self, ltb_line: BuiltModel):
        built = ltb_line
        model = engine.build(built.name, built.wiring, built.deck, built.settings)
        lattice = model.simulator.lattice
        start = np.asarray(model.simulator.twiss_in["closed_orbit"], dtype=float)
        reference_deck = at.load_lattice(str(built.deck))
        where = {str(element.FamName): [] for element in reference_deck}
        for index, element in enumerate(reference_deck):
            where[str(element.FamName)].append(index)

        blocks = section(model_reference(*LTB), "orbit_response")["physics"]
        computed, expected = [], []
        for block in blocks:
            family = block["actuator"]["family"]
            setpoints = _ltb_setpoints(built, family)
            monitors = _ltb_monitor_positions(
                built, block["monitor"]["family"], block["monitor"]["device_list"]
            )
            row = LTB_PLANES[block["monitor"]["family"]]
            kicks = np.broadcast_to(
                np.asarray(block["actuator_delta"], dtype=float),
                (len(block["actuator"]["device_list"]),),
            )
            correctors, columns = [], []
            for device, kick in zip(block["actuator"]["device_list"], kicks, strict=True):
                entry = setpoints[_ltb_row(device)]
                (position,) = where[str(entry["element"])]
                plane = int(entry["engine"]["index"])
                correctors.append((position, plane))
                element = lattice[position]
                held = float(element.KickAngle[plane])
                arms, applied = [], []
                for sign in (1.0, -1.0):
                    model.set({entry["address"]: _ltb_hardware(entry, held + sign * kick / 2.0)})
                    applied.append(float(element.KickAngle[plane]))
                    tracked = at.lattice_track(lattice, start.reshape(6, 1), refpts=monitors)[0]
                    arms.append(tracked[row, 0, :, 0])
                model.set({entry["address"]: _ltb_hardware(entry, held)})
                assert applied[0] - applied[1] == pytest.approx(kick, rel=0, abs=1e-15)
                columns.append((arms[0] - arms[1]) / kick)
            computed.append(np.array(columns).T)
            replay = recipes.loco_transport_full(reference_deck, start, correctors, kicks, monitors)
            expected.append(replay[:, :, row // 2].T)

        stated = np.concatenate([matrix.ravel() for matrix in expected])
        band = ORM_RMS_FRACTION * float(np.sqrt(np.mean(stated**2)))
        assert band > 0, "the recipe's columns are all zero"
        for matrix, replay, block in zip(computed, expected, blocks, strict=True):
            np.testing.assert_allclose(
                matrix,
                replay,
                rtol=0,
                atol=band,
                err_msg=f"{block['monitor']['family']} / {block['actuator']['family']}",
            )
