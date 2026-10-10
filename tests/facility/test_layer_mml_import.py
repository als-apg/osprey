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
        "rows.json",
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


def test_devices_carry_their_common_name_as_label_and_no_attributes(spear3: Path) -> None:
    ao = json.loads((FIXTURES / "spear3" / "spear3.storagering.ao.json").read_text())
    devices = _by_id(_rows(spear3, "devices.yaml"))
    for name in ao["BPMx"]["CommonNames"]:
        device = devices[f"StorageRing/{name}"]
        assert device["class"] == "BeamPositionMonitor"
        assert device["label"] == name
    assert [d["id"] for d in devices.values() if "attributes" in d or "names" in d] == []


def test_rows_json_maps_every_export_row_to_its_device(spear3: Path, tmp_path: Path) -> None:
    from osprey.facility.layers.mml.rows import ROWS_FILE, ROWS_SCHEMA, read_rows

    ao = json.loads((FIXTURES / "spear3" / "spear3.storagering.ao.json").read_text())
    raw = (spear3 / LAYER_DIR / ROWS_FILE).read_bytes()
    document = json.loads(raw)
    rows = document["rows"]
    keys = [(row["model"], row["family"], row["device_list"]) for row in rows]

    assert document["schema"] == ROWS_SCHEMA
    assert keys == sorted(keys)
    assert read_rows(spear3)["StorageRing", "BPMx"] == {
        tuple(row): f"StorageRing/{name}"
        for name, row in zip(ao["BPMx"]["CommonNames"], ao["BPMx"]["DeviceList"], strict=True)
    }
    again = _import(tmp_path, "spear3")
    assert (again / LAYER_DIR / ROWS_FILE).read_bytes() == raw


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
    assert list(channel) == ["id", "on", "role", "tolerance", "unit", "description"]
    described = yaml.safe_load((spear3 / MAPPING_FILE).read_text(encoding="utf-8"))
    fields = described["families"]["RF"]["fields"]
    assert fields["Setpoint"]["description"] != fields["Monitor"]["description"]
    owner = _by_id(_rows(spear3, "devices.yaml"))[channel["on"]["device"]]
    label = owner.get("label", owner["id"])
    assert channel["description"] == f"{label}: {fields['Setpoint']['description']}"
    assert channel["unit"] == family["Setpoint"]["HWUnits"]


def _setpoint(tree: str, family: str, index: int = 0) -> str:
    stem = TREES[tree][0]
    ao = json.loads((FIXTURES / tree / f"{stem}.ao.json").read_text())
    names = ao[family]["Setpoint"]["ChannelNames"]
    return (names[index] if isinstance(names, list) else names).strip()


def test_a_setpoint_takes_its_exports_tolerance(spear3: Path, tmp_path: Path) -> None:
    nsls2 = _import(tmp_path, "nsls2")

    assert _by_id(_rows(spear3, "channels.yaml"))[_setpoint("spear3", "HCM")]["tolerance"] == {
        "absolute": 0.101
    }
    assert _by_id(_rows(nsls2, "channels.yaml"))[_setpoint("nsls2", "HCM")]["tolerance"] == {
        "absolute": 0.01
    }


def test_a_per_element_tolerance_is_sliced(tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    for source in sorted((FIXTURES / "spear3").glob("spear3.storagering.*")):
        shutil.copyfile(source, exports / source.name)
    ao_path = exports / "spear3.storagering.ao.json"
    ao = json.loads(ao_path.read_text())
    count = len(ao["HCM"]["Setpoint"]["Tolerance"])
    ao["HCM"]["Setpoint"]["Tolerance"] = [0.1 + i / 100 for i in range(count)]
    ao_path.write_text(json.dumps(ao), encoding="utf-8")
    facility = _facility(tmp_path, "spear3")

    import_mml([ao_path], facility)

    channels = _by_id(_rows(facility, "channels.yaml"))
    assert channels[_setpoint("spear3", "HCM", 3)]["tolerance"] == {"absolute": 0.1 + 3 / 100}


def test_an_eps_or_infinite_tolerance_writes_none(
    spear3: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    nsls2 = _import(tmp_path, "nsls2")
    ao = json.loads((FIXTURES / "spear3" / "spear3.storagering.ao.json").read_text())
    voltage = ao["RF"]["VoltageCtrl"]["ChannelNames"].strip()

    assert "tolerance" not in _by_id(_rows(nsls2, "channels.yaml"))[_setpoint("nsls2", "RF")]
    assert "tolerance" not in _by_id(_rows(spear3, "channels.yaml"))[voltage]
    assert re.search(
        r"^\d+ setpoint devices export no usable `Setpoint.Tolerance`$",
        capsys.readouterr().out,
        re.MULTILINE,
    )


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


def test_nsls2_imports_both_trees_into_one_facility(tmp_path: Path) -> None:
    facility = _import(tmp_path, "nsls2")

    groups = _by_id(_rows(facility, "groups.yaml"))
    bpms = groups["BPM"]["members"]
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


def test_a_field_s_signal_role_lands_on_its_channels(spear3: Path) -> None:
    channels = _by_id(_rows(spear3, "channels.yaml"))
    groups = _by_id(_rows(spear3, "groups.yaml"))
    assert channels["01G-BPM1:U"]["signal"] == "position_x_readback"
    assert channels["MS1-BD:CurrSetpt"]["signal"] == "current_setpoint"
    assert all("signals" not in group for group in groups.values())


def test_plane_twins_are_one_group(spear3: Path) -> None:
    groups = _by_id(_rows(spear3, "groups.yaml"))
    mapping = yaml.safe_load((spear3 / MAPPING_FILE).read_text(encoding="utf-8"))["families"]
    twins = ["BPMx", "BPMy", "BTSBPMx", "BTSBPMy", "KickerAmp", "KickerDelay"]
    twins += ["BLErr", "BLOpen", "BLSum"]

    assert {"BPM", "BTSBPM", "Kicker", "BL"} <= set(groups)
    assert set(twins).isdisjoint(groups)
    assert {"BPMx", "BPMy"} <= set(groups["BPM"]["names"])
    for twin in ("BPMx", "BPMy"):
        assert mapping[twin]["description"] in groups["BPM"]["description"]
    assert len(groups) == 38


def test_two_folds_with_one_stem_keep_both_groups() -> None:
    from osprey.facility.layers.mml.importer import _physical_groups

    def group(token: str, members: list[str]) -> dict[str, Any]:
        return {"id": token, "members": members}

    groups = {
        "BPMa": group("BPMa", ["A1", "A2"]),
        "BPMb": group("BPMb", ["A1", "A2"]),
        "BPMx": group("BPMx", ["X1", "X2"]),
        "BPMy": group("BPMy", ["X1", "X2"]),
    }

    folded = _physical_groups(groups)

    assert sorted(folded) == ["BPMa", "BPMx"]
    assert folded["BPMa"]["members"] == ["A1", "A2"]
    assert folded["BPMx"]["members"] == ["X1", "X2"]


def test_a_channel_is_described_by_its_device_and_its_field_sentence(spear3: Path) -> None:
    channel = _by_id(_rows(spear3, "channels.yaml"))["01G-BPM1:U"]
    device = _by_id(_rows(spear3, "devices.yaml"))[channel["on"]["device"]]
    mapping = yaml.safe_load((spear3 / MAPPING_FILE).read_text(encoding="utf-8"))["families"]
    sentence = mapping["BPMx"]["fields"]["Monitor"]["description"]

    assert channel["description"] == f"{device['label']}: {sentence}"


def test_a_readback_several_setpoints_share_pairs_none_of_them(tmp_path: Path) -> None:
    facility = tmp_path / "data" / "facility"
    (facility / MAPPING_FILE).parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "paired" / MAPPING_FILE, facility / MAPPING_FILE)
    import_mml([FIXTURES / "paired" / "quokka.ring.ao.json"], facility)
    channels = _by_id(_rows(facility, "channels.yaml"))
    assert channels["QK:R12:HCM:RB"]["endpoint_of"] == ["RING/hcm_1", "RING/hcm_2"]
    for setpoint in ("QK:R1:HCM1:SP", "QK:R2:HCM1:SP"):
        assert channels[setpoint]["role"] == "setpoint"
        assert "pair" not in channels[setpoint], setpoint
    assert not run_stages(facility, project_name="demo").errors
