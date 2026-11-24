"""The pyat engine's single-pass solve over the imported NSLS-II transport line.

The line is imported and built from its Middle Layer export through the
session's ``mml_built``, and each corrector's step is answered as the Middle
Layer's transport calculator does. The synthetic and demo lines' single-pass
cases live in ``tests/simulation/test_single_pass_plugin.py``; this one lives
beside the other readers of ``mml_built`` so the one nsls2 build serves them all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import at
import numpy as np
import pytest

from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines.pyat_single_pass import SinglePassSimulator
from tests.facility._mml_built import BuiltModel
from tests.facility._model_reference import (
    ORM_RMS_FRACTION,
    model_reference,
    section,
)
from tests.services.mml import _mml_recipes as recipes

# xdist_group("mml_built"): the session ``mml_built`` fixture builds each fixture
# tree once, and every module reading it shares the group so that one build
# serves them all on one worker.
pytestmark = pytest.mark.xdist_group("mml_built")

LTB = ("nsls2", "nsls2.ltb")

#: The orbit plane (0 for ``x``, 1 for ``y``) each monitor family of the line reads.
LTB_PLANES = {"BPMx": 0, "BPMy": 1}


def _ltb_row(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _ltb_wired(
    built: BuiltModel, family: str, direction: str
) -> dict[tuple[int, ...], dict[str, Any]]:
    """The wiring entry each ``DeviceList`` row of a family carries in ``direction``.

    A family's entries are those whose engine block is the one the mapping
    wires the family through, on a device the import's row table records for
    one of the family's export rows.
    """
    from osprey_facility.layers.mml.mapping import MAPPING_FILE, read_mapping
    from osprey_facility.layers.mml.rows import read_rows

    mapping = read_mapping(built.facility / MAPPING_FILE)
    (model,) = [model for model in mapping.models.values() if model.name == built.name]
    words = model.wiring[family].engine
    if words.axis is not None:
        engine_words: dict[str, Any] = {"axis": words.axis}
    else:
        engine_words = {"attribute": words.attribute, "index": int(words.index)}
    rows = {
        device: _ltb_row(row)
        for row, device in read_rows(built.facility).get((built.name, family), {}).items()
    }
    on_device = {
        str(channel["id"]): (channel.get("on") or {}).get("device")
        for channel in built.document["channels"]
    }
    found: dict[tuple[int, ...], dict[str, Any]] = {}
    for entry in built.wiring:
        device = on_device.get(str(entry["address"]))
        if (
            entry.get("direction") == direction
            and dict(entry.get("engine") or {}) == engine_words
            and device in rows
        ):
            assert rows[device] not in found, f"{family} {rows[device]} has two {direction}s"
            found[rows[device]] = entry
    return found


def _ltb_setpoints(built: BuiltModel, family: str) -> dict[tuple[int, ...], dict[str, Any]]:
    """The setpoint each ``DeviceList`` row of a corrector family writes, from the wiring."""
    found = _ltb_wired(built, family, "write")
    assert {entry["engine"].get("attribute") for entry in found.values()} == {"KickAngle"}, (
        f"{family} is no corrector"
    )
    return found


def _ltb_readings(model: Any, entries: Sequence[dict[str, Any]]) -> np.ndarray:
    """What the plug-in serves at each monitor entry, in the physics units of its wiring."""
    from osprey_connectors.simulation.engines import curve_from_record, to_physics

    served = model.get([entry["address"] for entry in entries])
    return np.array(
        [
            to_physics(curve_from_record(entry["calibration"]["curve"]), served[entry["address"]])
            for entry in entries
        ]
    )


def _ltb_hardware(entry: dict[str, Any], physics: float) -> float:
    """The setpoint value the wiring's inverse calibration gives for a physics value."""
    from osprey_connectors.simulation.engines import curve_from_record, to_hardware

    calibration = entry["calibration"]
    return to_hardware(
        curve_from_record(calibration["curve"]),
        curve_from_record(calibration["inverse"]),
        physics,
    )


@pytest.fixture
def ltb_line(mml_built: Callable[[str, str], BuiltModel]) -> BuiltModel:
    """The LTB model of the session's one nsls2 build."""
    return mml_built(*LTB)


class TestImportedLtbLine:
    """The imported NSLS-II LTB line at ``solve: single_pass``, built through ``mml_built``.

    Each corrector is stepped through the plug-in's own setpoint, and each
    listed ``BPMx`` and ``BPMy`` device is read through the reading the
    plug-in serves on its wired address. The replay tracks the deck the build
    wrote from the same launch orbit to the elements those addresses read.
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
            readbacks = _ltb_wired(built, block["monitor"]["family"], "read")
            entries = [readbacks[_ltb_row(device)] for device in block["monitor"]["device_list"]]
            monitors = [where[str(entry["element"])][0] for entry in entries]
            plane = LTB_PLANES[block["monitor"]["family"]]
            kicks = np.broadcast_to(
                np.asarray(block["actuator_delta"], dtype=float),
                (len(block["actuator"]["device_list"]),),
            )
            correctors, columns = [], []
            for device, kick in zip(block["actuator"]["device_list"], kicks, strict=True):
                entry = setpoints[_ltb_row(device)]
                (position,) = where[str(entry["element"])]
                kicked = int(entry["engine"]["index"])
                correctors.append((position, kicked))
                element = lattice[position]
                held = float(element.KickAngle[kicked])
                arms, applied = [], []
                for sign in (1.0, -1.0):
                    model.set({entry["address"]: _ltb_hardware(entry, held + sign * kick / 2.0)})
                    applied.append(float(element.KickAngle[kicked]))
                    arms.append(_ltb_readings(model, entries))
                model.set({entry["address"]: _ltb_hardware(entry, held)})
                assert applied[0] - applied[1] == pytest.approx(kick, rel=0, abs=1e-15)
                columns.append((arms[0] - arms[1]) / kick)
            computed.append(np.array(columns).T)
            replay = recipes.loco_transport_full(reference_deck, start, correctors, kicks, monitors)
            expected.append(replay[:, :, plane].T)

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
