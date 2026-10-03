"""The mml layer's importer writing record sources under ``imported/mml/``.

Each case copies a fixture tree's mapping to
``data/facility/imported/mml/mapping.yaml`` and imports the tree's exports
into a fresh ``data/facility/``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers.mml.importer import LAYER_DIR, MappingProblems, import_mml
from osprey.facility.layers.mml.mapping import MAPPING_FILE, ImportStop
from osprey.facility.validate import run_stages
from tests.facility.test_word_ratchet import OUTSIDE_FORMAT_FILES

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

#: Each tree's exports, by the stem every file of one export is named after.
TREES: dict[str, tuple[str, ...]] = {
    "spear3": ("spear3.storagering",),
    "nsls2": ("nsls2.storagering", "nsls2.ltb"),
}

#: The authored files the spear3 import seeds beside the layer's records.
SEEDED = (
    "classes.yaml",
    "identity.yaml",
    "limits.yaml",
    "measurement/StorageRing.yaml",
    "seeds.yaml",
)

_LTB_TWISS = {
    "beta": [4.5, 4.8],
    "alpha": [-0.5, -0.6],
    "dispersion": [0.0, 0.0, 0.0, 0.0],
    "closed_orbit": [0.0, 0.0, 0.0, 0.0],
}


def _facility(tmp_path: Path, tree: str) -> Path:
    facility = tmp_path / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / tree / MAPPING_FILE, target)
    return facility


def _import(tmp_path: Path, tree: str, stems: tuple[str, ...] | None = None) -> Path:
    facility = _facility(tmp_path, tree)
    exports = [FIXTURES / tree / f"{stem}.ao.json" for stem in stems or TREES[tree]]
    import_mml(exports, facility)
    return facility


def _rows(facility: Path, name: str) -> list[dict[str, Any]]:
    return yaml.safe_load((facility / LAYER_DIR / name).read_text(encoding="utf-8"))


def _by_id(rows: list[dict[str, Any]], key: str = "id") -> dict[str, dict[str, Any]]:
    return {row[key]: row for row in rows}


@pytest.fixture(scope="module")
def spear3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _import(tmp_path_factory.mktemp("spear3"), "spear3")


def test_writes_record_sources_never_a_view(spear3: Path) -> None:
    layer = spear3 / LAYER_DIR
    assert sorted(p.name for p in layer.iterdir()) == [
        "StorageRing.response.json",
        "channels.yaml",
        "decks",
        "devices.yaml",
        "groups.yaml",
        "mapping.yaml",
        "models.yaml",
    ]
    assert sorted(p.name for p in (layer / "decks").iterdir()) == ["StorageRing.json"]
    files = [p.relative_to(spear3).as_posix() for p in layer.rglob("*") if p.is_file()]
    assert sorted(p.relative_to(spear3).as_posix() for p in spear3.rglob("*") if p.is_file()) == (
        sorted([*files, *SEEDED])
    )


def test_each_record_file_is_sorted_by_id(spear3: Path) -> None:
    for name in ("devices.yaml", "channels.yaml", "groups.yaml"):
        ids = [row["id"] for row in _rows(spear3, name)]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))


def test_devices_carry_device_list_and_element_list(spear3: Path) -> None:
    ao = json.loads((FIXTURES / "spear3" / "spear3.storagering.ao.json").read_text())
    devices = _by_id(_rows(spear3, "devices.yaml"))
    for name, row, element in zip(
        ao["BPMx"]["CommonNames"], ao["BPMx"]["DeviceList"], ao["BPMx"]["ElementList"], strict=True
    ):
        device = devices[f"StorageRing/{name}"]
        assert device["class"] == "BeamPositionMonitor"
        assert device["attributes"] == {"DeviceList": row, "ElementList": element}


def test_each_family_is_a_group_of_its_devices(spear3: Path) -> None:
    ao = json.loads((FIXTURES / "spear3" / "spear3.storagering.ao.json").read_text())
    devices = _rows(spear3, "devices.yaml")
    groups = _by_id(_rows(spear3, "groups.yaml"))
    members = {member for group in groups.values() for member in group["members"]}
    assert members == {device["id"] for device in devices}
    hcm = groups["HCM"]
    assert hcm["members"] == sorted(f"StorageRing/{name}" for name in ao["HCM"]["CommonNames"])
    assert hcm["description"]


def test_a_setpoint_pairs_with_its_family_monitor(spear3: Path) -> None:
    ao = json.loads((FIXTURES / "spear3" / "spear3.storagering.ao.json").read_text())
    channels = _by_id(_rows(spear3, "channels.yaml"))
    setpoint = ao["HCM"]["Setpoint"]["ChannelNames"][0].strip()
    monitor = ao["HCM"]["Monitor"]["ChannelNames"][0].strip()
    assert channels[setpoint]["role"] == "setpoint"
    assert channels[setpoint]["pair"] == monitor
    assert channels[setpoint]["on"] == {"device": f"StorageRing/{ao['HCM']['CommonNames'][0]}"}
    assert channels[monitor]["role"] == "readback"
    assert "pair" not in channels[monitor]


def test_an_address_a_write_field_names_is_a_setpoint_whichever_field_named_it_first(
    spear3: Path,
) -> None:
    ao = json.loads((FIXTURES / "spear3" / "spear3.storagering.ao.json").read_text())
    family = ao["RF"]
    address = family["Setpoint"]["ChannelNames"].strip()
    assert family["Monitor"]["ChannelNames"].strip() == address
    assert list(family).index("Monitor") < list(family).index("Setpoint")
    channel = _by_id(_rows(spear3, "channels.yaml"))[address]
    assert channel["role"] == "setpoint"
    assert "pair" not in channel
    assert list(channel) == ["id", "on", "role", "unit", "description"]
    described = yaml.safe_load((spear3 / MAPPING_FILE).read_text(encoding="utf-8"))
    fields = described["families"]["RF"]["fields"]
    assert fields["Setpoint"]["description"] != fields["Monitor"]["description"]
    assert channel["description"] == fields["Setpoint"]["description"]
    assert channel["unit"] == family["Setpoint"]["HWUnits"]


def test_the_imported_tree_loads_with_its_mapping_beside_the_records(spear3: Path) -> None:
    report = run_stages(spear3, project_name="demo")
    assert [e.format_message() for e in report.errors if "is not a layer file" in e.detail] == []
    assert report.failed != "load"


def test_response_export_is_copied_byte_for_byte(spear3: Path) -> None:
    copied = spear3 / LAYER_DIR / "StorageRing.response.json"
    source = FIXTURES / "spear3" / "spear3.storagering.response.json"
    assert copied.read_bytes() == source.read_bytes()


def test_a_second_import_writes_the_same_bytes(tmp_path: Path, spear3: Path) -> None:
    again = _import(tmp_path, "spear3")
    for name in ("devices.yaml", "channels.yaml", "groups.yaml", "models.yaml"):
        assert (again / LAYER_DIR / name).read_bytes() == (spear3 / LAYER_DIR / name).read_bytes()


def _stated(model: dict[str, Any]) -> dict[str, Any]:
    """A model entry without its wiring records."""
    return {key: value for key, value in model.items() if key != "wiring"}


def test_a_periodic_model_states_no_settings(spear3: Path) -> None:
    (model,) = _rows(spear3, "models.yaml")
    assert _stated(model) == {
        "name": "StorageRing",
        "engine": "pyat",
        "deck": "imported/mml/decks/StorageRing.json",
    }
    assert model["wiring"]


def test_nsls2_imports_with_the_demo_ontology_blocked(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "nsls2")
    exports = [str(FIXTURES / "nsls2" / f"{stem}.ao.json") for stem in TREES["nsls2"]]
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.modules['osprey.services.facility_knowledge.ttl_generator.ontology_map'] = None\n"
        "from osprey.facility.layers.mml.importer import import_mml\n"
        "import_mml([Path(p) for p in sys.argv[2:]], Path(sys.argv[1]))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(facility), *exports],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    groups = _by_id(_rows(facility, "groups.yaml"))
    bpms = groups["BPMx"]["members"]
    assert any(m.startswith("LTB/") for m in bpms)
    assert any(m.startswith("StorageRing/") for m in bpms)
    assert bpms == sorted(set(bpms))
    for model in ("StorageRing", "LTB"):
        copied = facility / LAYER_DIR / f"{model}.response.json"
        source = FIXTURES / "nsls2" / f"nsls2.{model.lower()}.response.json"
        assert copied.read_bytes() == source.read_bytes()


def test_no_facility_word_under_the_layer() -> None:
    """The layer's own code names no facility; the exporter it ships is an outside format."""
    tracked = subprocess.run(
        ["git", "ls-files", "src/osprey/facility/layers/mml/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    files = [path for path in tracked if path not in OUTSIDE_FORMAT_FILES]
    assert "src/osprey/facility/layers/mml/importer.py" in files
    word = re.compile(r"\b(als|spear3?|nsls-?(ii|2)?|ltb|gtl|gtb|bts|quokka)\b", re.IGNORECASE)
    found = [
        f"{path}:{number}"
        for path in files
        for number, line in enumerate(
            (REPO_ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines(), 1
        )
        if word.search(line)
    ]
    assert found == []


def test_an_unnamed_system_stops_the_import(tmp_path: Path) -> None:
    facility = _facility(tmp_path, "spear3")
    path = facility / MAPPING_FILE
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["models"]["Other"] = document["models"].pop("StorageRing")
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(MappingProblems) as stop:
        import_mml([FIXTURES / "spear3" / "spear3.storagering.ao.json"], facility)
    assert [str(problem) for problem in stop.value.problems] == [
        "models.Other: Other is no exported system",
        "models: leaves out the exported system StorageRing",
    ]


def _nsls2_copy(tmp_path: Path, *, model_json: bool = True, deck: bool = True) -> list[Path]:
    """Both nsls2 exports copied, the LTB one with or without its model file and deck."""
    exports = tmp_path / "exports"
    exports.mkdir()
    for stem in TREES["nsls2"]:
        for source in sorted((FIXTURES / "nsls2").glob(f"{stem}.*")):
            if stem == "nsls2.ltb" and source.name.endswith(".model.json") and not model_json:
                continue
            if stem == "nsls2.ltb" and source.name.endswith(".lattice.mat") and not deck:
                continue
            shutil.copyfile(source, exports / source.name)
    return [exports / f"{stem}.ao.json" for stem in TREES["nsls2"]]


def test_transport_from_the_model_file_runs_single_pass_from_the_deck_twiss(
    tmp_path: Path,
) -> None:
    facility = _import(tmp_path, "nsls2")
    models = _by_id(_rows(facility, "models.yaml"), "name")
    assert _stated(models["LTB"]) == {
        "name": "LTB",
        "engine": "pyat",
        "deck": "imported/mml/decks/LTB.json",
        "settings": {"pyat": {"solve": "single_pass", "twiss_in": _LTB_TWISS}},
    }
    assert list(models["LTB"]) == ["name", "engine", "deck", "settings", "wiring"]
    assert _stated(models["StorageRing"]) == {
        "name": "StorageRing",
        "engine": "pyat",
        "deck": "imported/mml/decks/StorageRing.json",
    }


def test_transport_from_the_machine_type_without_a_model_file(tmp_path: Path) -> None:
    exports = _nsls2_copy(tmp_path, model_json=False)
    facility = _facility(tmp_path, "nsls2")
    import_mml(exports, facility)
    model = _by_id(_rows(facility, "models.yaml"), "name")["LTB"]
    assert model["settings"] == {"pyat": {"solve": "single_pass", "twiss_in": _LTB_TWISS}}


def test_the_model_file_decides_over_the_machine_type(tmp_path: Path) -> None:
    exports = _nsls2_copy(tmp_path)
    model_file = exports[0].with_name("nsls2.ltb.model.json")
    document = json.loads(model_file.read_text(encoding="utf-8"))
    document["state"]["is_transport"] = 0
    model_file.write_text(json.dumps(document), encoding="utf-8")
    facility = _facility(tmp_path, "nsls2")
    import_mml(exports, facility)
    model = _by_id(_rows(facility, "models.yaml"), "name")["LTB"]
    assert _stated(model) == {
        "name": "LTB",
        "engine": "pyat",
        "deck": "imported/mml/decks/LTB.json",
    }


def test_a_transport_line_without_initial_twiss_stops(tmp_path: Path) -> None:
    exports = _nsls2_copy(tmp_path, deck=False)
    facility = _facility(tmp_path, "nsls2")
    with pytest.raises(ImportStop) as stop:
        import_mml(exports, facility)
    assert stop.value.format_message() == (
        "import mml: mapping-undecided: LTB: transport line without initial twiss"
    )
    assert not (facility / LAYER_DIR / "models.yaml").exists()


def test_a_mapping_that_fails_its_check_stops_the_import_before_anything_is_written(
    tmp_path: Path,
) -> None:
    facility = _facility(tmp_path, "synthetic")
    mapping = facility / MAPPING_FILE
    text = mapping.read_text(encoding="utf-8")
    assert text.count("    name: SR\n") == 1
    mapping.write_text(text.replace("    name: SR\n", "    name: S R\n"), encoding="utf-8")

    with pytest.raises(MappingProblems) as stop:
        import_mml([FIXTURES / "synthetic" / "quokka.sr.ao.json"], facility)

    assert stop.value.exit_code == 1
    lines = stop.value.format_message().splitlines()
    assert lines[0] == "models.SR.name: 'S R' is not PN_LOCAL"
    assert lines[1:-1] == [str(problem) for problem in stop.value.problems[1:]]
    assert lines[-1] == f"{len(lines) - 1} problems in {mapping}; fix each and check again."
    assert sorted(path.name for path in (facility / LAYER_DIR).iterdir()) == ["mapping.yaml"]
    assert sorted(path.name for path in facility.iterdir()) == ["imported"]
