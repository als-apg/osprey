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

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from osprey.facility.views.pyaml import CONFIGURATION_FILE, TRM_FILE
from tests.facility._pyaml_trees import (
    _sr_deck,
    measured_tree,
    with_correctors,
    with_rf,
    write_view,
)
from tests.facility._synthetic_trees import Deck

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


def test_a_magnet_split_across_thin_and_thick_elements_is_left_out_with_a_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """pyAML writes one polynomial into every slice, which a thin and a thick slice read apart."""
    from pyaml.accelerator import Accelerator

    def elements(at: Any) -> list[Any]:
        found = _sr_deck().elements(at)
        (after,) = [i + 1 for i, element in enumerate(found) if element.FamName == "SCOR"]
        thick = at.Corrector("SCORT", 0.1, [0.0, 0.0])
        return [*found[:after], thick, *found[after:]]

    tree = with_correctors(with_rf(measured_tree()))
    tree["measurement/SR.yaml"] |= {"kinds": ["orm"], "corrector_delta": 1.0e-5}
    tree["decks/sr.json"] = Deck(elements)
    entry = _wiring(tree, "SR", "SCOR:H:SP")
    del entry["element"]
    entry["slices"] = [{"element": "SCOR"}, {"element": "SCORT"}]

    configuration = _configuration(tmp_path, tree, "SR")

    err = capsys.readouterr().err
    assert (
        "view pyaml: SR leaves out 1 setpoint whose slices mix zero-length and thick elements, "
        "which pyAML drives with the wrong strength: SCOR:H:SP"
    ) in err
    assert "pyAML has no magnet model" not in err
    assert "SCOR:H:SP" not in [device["name"] for device in configuration["devices"]]
    for array in configuration["arrays"]:
        assert "SCOR:H:SP" not in array["elements"]
    assert Accelerator.load(str(tmp_path / "render/data/pyaml/SR" / CONFIGURATION_FILE))


def _thin_qd_tree() -> dict[str, Any]:
    """The measured tree with QD a zero-length thin multipole between two drifts."""

    def elements(at: Any) -> list[Any]:
        found = _sr_deck().elements(at)
        (index,) = [i for i, element in enumerate(found) if element.FamName == "QD"]
        thin = at.ThinMultipole("QD", [0.0, 0.0], [0.0, -0.5])
        return [
            *found[:index],
            at.Drift("DQ", 0.25),
            thin,
            at.Drift("DQ", 0.25),
            *found[index + 1 :],
        ]

    tree = measured_tree()
    tree["decks/sr.json"] = Deck(elements)
    return tree


def test_a_thin_quadrupole_steps_the_tune_the_deck_s_strength_change_does(
    tmp_path: Path,
) -> None:
    """A magnet whose every slice is zero-length is scaled at unit length, as pyAML reads it."""
    import at
    from pyaml.accelerator import Accelerator

    tree = _thin_qd_tree()
    configuration = _configuration(tmp_path, tree, "SR")
    (device,) = [device for device in configuration["devices"] if device["name"] == "QD:SP"]
    assert device["model"]["calibration_factor"] == pytest.approx(0.01 * BRHO)

    step = 1.0e-3
    design = Accelerator.load(str(tmp_path / "render/data/pyaml/SR" / CONFIGURATION_FILE)).design
    magnet = design.magnet.get("QD:SP")
    held = magnet.strength.get()
    assert held == pytest.approx(-0.5)
    before = design.get_lattice().get_tune()
    magnet.strength.set(held + step)
    after = design.get_lattice().get_tune()

    deck = at.load_lattice(str(tmp_path / "data" / "facility" / "decks" / "sr.json"))
    (quad,) = [element for element in deck if element.FamName == "QD"]
    expected_before = deck.get_tune()
    quad.PolynomB[1] += step
    expected = deck.get_tune() - expected_before
    assert np.max(np.abs(expected)) > 1.0e-5
    assert after - before == pytest.approx(expected, rel=1.0e-4)


def test_a_quad_group_holding_a_thin_quadrupole_keeps_its_tune_response(
    tmp_path: Path,
) -> None:
    """A zero-length slice weighs in at unit length, so the thin quadrupole's column is its beta."""
    import at

    directory, written = write_view(tmp_path, _thin_qd_tree())
    assert directory / "SR" / TRM_FILE in written
    trm = json.loads((directory / "SR" / TRM_FILE).read_text())
    assert trm["variable_names"] == ["QD:SP", "QF:SP"]

    lattice = at.load_lattice(str(directory / "SR" / "lattice.json")).disable_6d(copy=True)
    _, beta, *_ = at.avlinopt(lattice, refpts=np.arange(len(lattice)))
    (qd,) = [i for i, element in enumerate(lattice) if element.FamName == "QD"]
    matrix = np.asarray(trm["matrix"]) * 4 * math.pi
    assert matrix[:, 0] == pytest.approx((beta[qd, 0], -beta[qd, 1]), rel=1e-8)
