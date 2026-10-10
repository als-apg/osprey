"""The pyAML view: which models get one, what it references and how pyAML loads it.

A served model that names a deck and has a measurement file gets
``data/pyaml/<model>/configuration.yaml``; a periodic one also gets the design
simulator's lattice beside it as ``lattice.json``, the engine's copy of its deck
with the correctors' kicks as polynomials, and a ``single_pass`` one has no
design simulator. Every other served model is named
in a note on stderr.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.views import VIEWS
from osprey.facility.views.pyaml import (
    CONFIGURATION_FILE,
    LATTICE_FILE,
    measured_models,
    pyaml_view_wanted,
    view_names,
)
from tests.facility._pyaml_trees import measured_tree, view_inputs, write_view

pytest.importorskip("pyaml")


def _measurement(doc: dict[str, Any], model: str) -> dict[str, Any]:
    (record,) = [record for record in doc["models"] if record["name"] == model]
    measurement: dict[str, Any] = record["measurement"]
    return measurement


def _configuration(directory: Path, model: str) -> dict[str, Any]:
    return yaml.safe_load((directory / model / CONFIGURATION_FILE).read_text(encoding="utf-8"))


def test_the_view_is_written_after_the_simulator_and_before_the_facts() -> None:
    names = [view.name for view in VIEWS]
    assert names.index("simulator") < names.index("pyaml") < names.index("facts")
    (view,) = [view for view in VIEWS if view.name == "pyaml"]
    assert view.path == "pyaml"


def test_each_served_measured_model_gets_a_configuration(tmp_path: Path) -> None:
    directory, written = write_view(tmp_path, measured_tree())
    relative = sorted(path.relative_to(directory).as_posix() for path in written)
    assert relative == [
        "LINE/configuration.yaml",
        "SR/configuration.yaml",
        "SR/lattice.json",
        "SR/trm.json",
    ]


def test_a_periodic_model_references_its_lattice_beside_the_configuration(
    tmp_path: Path,
) -> None:
    from osprey.simulation.engines.pyat import polynomial_kicks

    directory, _ = write_view(tmp_path, measured_tree())
    (simulator,) = _configuration(directory, "SR")["simulators"]
    assert simulator == {
        "type": "pyaml.lattice.simulator",
        "name": "design",
        "lattice": f"${{path:{LATTICE_FILE}}}",
    }
    deck = tmp_path / "data" / "facility" / "decks" / "sr.json"
    lattice = (directory / "SR" / LATTICE_FILE).read_text(encoding="utf-8")
    assert lattice == polynomial_kicks(deck, []).text


def test_a_single_pass_model_has_no_design_simulator(tmp_path: Path) -> None:
    directory, _ = write_view(tmp_path, measured_tree())
    assert _configuration(directory, "LINE")["simulators"] == []
    assert not (directory / "LINE" / LATTICE_FILE).exists()


def test_pyaml_loads_the_periodic_model_with_a_design_holder_and_the_line_without(
    tmp_path: Path,
) -> None:
    from pyaml.accelerator import Accelerator

    directory, _ = write_view(tmp_path, measured_tree())
    sr = Accelerator.load(str(directory / "SR" / CONFIGURATION_FILE))
    line = Accelerator.load(str(directory / "LINE" / CONFIGURATION_FILE))
    assert sr.design is not None
    # The split quadrupole's integrated strength: K = 1.0 on two slices of 0.25 m.
    assert sr.design.magnet.get("QF:SP").strength.get() == pytest.approx(0.5)
    assert line.design is None
    assert line.live is not None


def test_no_file_is_included_and_every_path_reference_is_written(tmp_path: Path) -> None:
    directory, _ = write_view(tmp_path, measured_tree())
    for model in ("SR", "LINE"):
        text = (directory / model / CONFIGURATION_FILE).read_text(encoding="utf-8")
        assert "${file:" not in text
        for target in re.findall(r"\$\{path:([^}]+)\}", text):
            assert (directory / model / target).is_file(), target


def test_the_view_is_the_same_bytes_every_time(tmp_path: Path) -> None:
    first, written = write_view(tmp_path / "a", measured_tree())
    second, _ = write_view(tmp_path / "b", measured_tree())
    for path in written:
        relative = path.relative_to(first)
        assert (second / relative).read_bytes() == path.read_bytes(), relative


@pytest.mark.parametrize("vcor", ["LINE/VCM", "LINE/HCM"], ids=["two-groups", "one-group"])
def test_every_name_in_the_configuration_resolves_back_through_the_mapping(
    tmp_path: Path, vcor: str
) -> None:
    from pyaml_cs_osprey.catalog import parse_reference
    from pyaml_cs_osprey.names import UnmappedName

    tree = measured_tree()
    tree["measurement/LINE.yaml"]["groups"]["vcor"] = vcor
    inputs = view_inputs(tmp_path, tree)
    directory, _ = write_view(tmp_path / "view", tree)
    for model in ("SR", "LINE"):
        names = view_names(inputs.doc, model)
        configuration = _configuration(directory, model)
        arrays = configuration["arrays"]
        groups_named = _measurement(inputs.doc, model)["groups"]
        named = {names.array_name(role, group): group for role, group in groups_named.items()}
        assert {array["name"]: names.array_group(array["name"]) for array in arrays} == named
        for device in configuration["devices"]:
            name = device["name"]
            if "model" in device:
                address = names.magnet_address(name)
                assert parse_reference(device["model"]["powerconverter"]).address == address
            elif device["type"] == "pyaml.bpm.bpm":
                assert names.bpm_name(names.bpm_device(name)) == name
            elif device["type"] == "pyaml.rf.rf_plant":
                assert names.rf_plant_name(names.rf_address(name)) == name
            else:
                with pytest.raises(UnmappedName):
                    names.magnet_address(name)


def test_an_unserved_model_has_no_view_and_no_note(tmp_path: Path) -> None:
    inputs = view_inputs(tmp_path, measured_tree(), served=["LINE"])
    assert measured_models(inputs) == (["LINE"], {})


def test_a_served_model_without_a_measurement_file_is_named_in_a_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = measured_tree()
    del tree["measurement/LINE.yaml"]
    directory, written = write_view(tmp_path, tree)
    assert not (directory / "LINE").exists()
    assert {path.parent.name for path in written} == {"SR"}
    captured = capsys.readouterr()
    assert "pyAML view omitted for LINE: no measurement/LINE.yaml" in captured.err
    assert captured.out == ""


def test_a_render_with_no_measured_model_does_not_carry_the_view(tmp_path: Path) -> None:
    tree = measured_tree()
    del tree["measurement/LINE.yaml"]
    del tree["measurement/SR.yaml"]
    wanted, reason = pyaml_view_wanted(view_inputs(tmp_path, tree))
    assert wanted is False
    assert reason == (
        "pyAML view omitted for LINE: no measurement/LINE.yaml; "
        "pyAML view omitted for SR: no measurement/SR.yaml"
    )


def test_a_render_serving_no_deck_model_does_not_carry_the_view(tmp_path: Path) -> None:
    wanted, reason = pyaml_view_wanted(view_inputs(tmp_path, measured_tree(), served=[]))
    assert (wanted, reason) == (False, "no served model names a deck")


def test_the_profile_mirror_may_not_carry_a_pyaml_view(tmp_path: Path) -> None:
    from osprey.cli.profile_conventions import (
        RESERVED_MIRROR_PATTERNS,
        facility_mirror_violation,
    )

    assert "data/pyaml/**" in RESERVED_MIRROR_PATTERNS
    mirrored = tmp_path / "data" / "pyaml" / "SR" / CONFIGURATION_FILE
    mirrored.parent.mkdir(parents=True)
    mirrored.write_text("type: pyaml.accelerator\n", encoding="utf-8")
    refusal = facility_mirror_violation(tmp_path)
    assert refusal is not None
    assert "data/pyaml/SR/configuration.yaml" in str(refusal)
