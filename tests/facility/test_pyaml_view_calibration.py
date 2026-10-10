"""The pyAML magnet model each setpoint is converted by.

pyAML's strength is the integrated field over the rigidity ``E / c``. The deck
holds ``curve(hardware) * weight`` on each slice, so a linear curve becomes a
``linear_model`` of factor ``gain * scale * brho`` (``scale`` the sum of weight
times length for a multipole, of the weights for a kick) and offset
``-offset / gain``; a table becomes an ``inline_curve`` extended past its ends
and past the limits band. A setpoint whose curve pyAML cannot represent has no
magnet, is in no array, and is named in a note on stderr.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from osprey.facility.views.pyaml import CONFIGURATION_FILE
from tests.facility._pyaml_trees import measured_tree, write_view

pytest.importorskip("pyaml")

#: The rigidity pyAML derives from the synthetic decks' 3 GeV.
BRHO = 3.0e9 / 299792458.0


def _wiring(tree: dict[str, Any], model: str, address: str) -> dict[str, Any]:
    (record,) = [m for m in tree["models.yaml"] if m["name"] == model]
    (entry,) = [w for w in record["wiring"] if w["address"] == address]
    return entry


def _configuration(tmp_path: Path, tree: dict[str, Any], model: str) -> dict[str, Any]:
    directory, _ = write_view(tmp_path, tree)
    loaded: dict[str, Any] = yaml.safe_load((directory / model / CONFIGURATION_FILE).read_text())
    return loaded


def _model(tmp_path: Path, tree: dict[str, Any], model: str, name: str) -> dict[str, Any] | None:
    devices = _configuration(tmp_path, tree, model)["devices"]
    found = [device["model"] for device in devices if device["name"] == name]
    return found[0] if found else None


def test_a_linear_multipole_scales_by_its_slice_lengths(tmp_path: Path) -> None:
    model = _model(tmp_path, measured_tree(), "SR", "QF:SP")
    assert model is not None
    assert model["type"] == "pyaml.magnet.linear_model"
    assert model["unit"] == "1/m"
    # Two slices of 0.25 m, weight 1, gain 0.01.
    assert model["calibration_factor"] == pytest.approx(0.01 * 0.5 * BRHO)
    assert "calibration_offset" not in model


def test_a_kick_scales_by_its_weights_alone(tmp_path: Path) -> None:
    model = _model(tmp_path, measured_tree(), "LINE", "LCOR:H:SP")
    assert model is not None
    assert model["unit"] == "rad"
    assert model["hardware_unit"] == "A"
    assert model["calibration_factor"] == pytest.approx(0.01 * BRHO)


def test_an_offset_becomes_the_hardware_value_of_zero_strength(tmp_path: Path) -> None:
    tree = measured_tree()
    _wiring(tree, "SR", "QD:SP")["calibration"] = {
        "curve": {"linear": {"gain": 0.02, "offset": 0.1}}
    }
    model = _model(tmp_path, tree, "SR", "QD:SP")
    assert model is not None
    assert model["calibration_offset"] == pytest.approx(-0.1 / 0.02)
    assert model["calibration_factor"] == pytest.approx(0.02 * 0.5 * BRHO)


def test_no_calibration_is_the_identity_curve(tmp_path: Path) -> None:
    tree = measured_tree()
    del _wiring(tree, "SR", "QD:SP")["calibration"]
    model = _model(tmp_path, tree, "SR", "QD:SP")
    assert model is not None
    assert model["calibration_factor"] == pytest.approx(0.5 * BRHO)


def test_pyaml_inverts_the_model_onto_the_deck_strength(tmp_path: Path) -> None:
    """The hardware value the model gives the deck's strength is the calibration's inverse."""
    from pyaml.accelerator import Accelerator

    tree = measured_tree()
    directory, _ = write_view(tmp_path, tree)
    sr = Accelerator.load(str(directory / "SR" / CONFIGURATION_FILE))
    magnet = sr.design.magnet.get("QD:SP")
    strength = magnet.strength.get()
    assert strength == pytest.approx(-1.0 * 0.5)
    hardware = magnet.model.compute_hardware_values(np.array([strength]))[0]
    assert hardware == pytest.approx(-1.0 / 0.01)


def test_a_table_is_an_inline_curve_reaching_past_its_ends_and_the_band(
    tmp_path: Path,
) -> None:
    tree = measured_tree()
    _wiring(tree, "SR", "QD:SP")["calibration"] = {
        "curve": {"table": {"grid": [0.0, 10.0, 20.0], "values": [0.0, 1.0, 2.5]}},
        "inverse": {"table": {"grid": [0.0, 1.0, 2.5], "values": [0.0, 10.0, 20.0]}},
    }
    tree["limits.yaml"] = {"records": [{"address": "QD:SP", "min_value": -15.0, "max_value": 40.0}]}
    model = _model(tmp_path, tree, "SR", "QD:SP")
    assert model is not None
    assert "calibration_factor" not in model
    scale = 0.5 * BRHO
    mat = np.asarray(model["curve"]["mat"])
    assert model["curve"]["type"] == "pyaml.magnet.inline_curve"
    assert list(mat[:, 0]) == pytest.approx([-25.0, 0.0, 10.0, 20.0, 50.0])
    assert list(mat[:, 1] / scale) == pytest.approx([-2.5, 0.0, 1.0, 2.5, 7.0])


def test_a_table_that_is_not_monotone_has_no_model(tmp_path: Path) -> None:
    tree = measured_tree()
    _wiring(tree, "SR", "QD:SP")["calibration"] = {
        "curve": {"table": {"grid": [0.0, 10.0, 20.0], "values": [0.0, 1.0, 0.5]}},
        "inverse": {"table": {"grid": [0.0, 0.5, 1.0], "values": [0.0, 20.0, 10.0]}},
    }
    assert _model(tmp_path, tree, "SR", "QD:SP") is None


def test_a_curve_per_slice_leaves_the_setpoint_out_with_a_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from pyaml.accelerator import Accelerator

    tree = measured_tree()
    entry = _wiring(tree, "SR", "QF:SP")
    entry["slices"] = [
        {"element": "QFA"},
        {"element": "QFB", "curve": {"linear": {"gain": 0.02, "offset": 0.0}}},
    ]
    configuration = _configuration(tmp_path, tree, "SR")
    assert "QF:SP" not in [device["name"] for device in configuration["devices"]]
    (quads,) = [array for array in configuration["arrays"] if array["name"] == "SR_Q"]
    assert quads["elements"] == ["QD:SP"]
    assert "view pyaml: SR leaves out 1 setpoint pyAML has no magnet model for: QF:SP" in (
        capsys.readouterr().err
    )
    assert Accelerator.load(str(tmp_path / "render/data/pyaml/SR" / CONFIGURATION_FILE))
