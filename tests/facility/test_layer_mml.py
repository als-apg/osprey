"""The mml layer's rules, each on the smallest input that shows it.

The role and pair rule, the stop on an undecided direction, shared endpoints
and the dedup of a device two systems list run on a small invented export
written straight to records. The rules that need a deck (``slices``, the
planes of the orbit correctors) run on the fixture exports, and the merge
rules run the imported sources through the build's load and combine stages.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.combine import CombineResult, combine
from osprey.facility.layers.mml.importer import (
    LAYER_DIR,
    Exports,
    MappingProblems,
    import_mml,
    write_records,
)
from osprey.facility.layers.mml.mapping import (
    MAPPING_FILE,
    FieldRole,
    ImportStop,
    field_roles,
    parse_mapping,
)
from osprey.facility.sources import load_sources

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

TREES: dict[str, tuple[str, ...]] = {
    "spear3": ("spear3.storagering",),
    "nsls2": ("nsls2.storagering", "nsls2.ltb"),
}


# -- a small invented export ---------------------------------------------------


def _export() -> dict[str, Any]:
    """Two systems: ``Q2`` and ``Q3`` share a supply, ``UP`` lists a ``DOWN`` device."""
    return {
        "_import_order": ["UP", "DOWN"],
        "UP": {
            "Q": {
                "CommonNames": ["Q1", "Q2", "Q3", "DOWN_Q1"],
                "Setpoint": {
                    "ChannelNames": ["UP:Q1:SP", "UP:Q2:SP", "UP:Q2:SP", "DOWN:Q1:SP"],
                    "HWUnits": "A",
                },
                "Monitor": {
                    "ChannelNames": ["UP:Q1:RB", "UP:Q2:RB", "UP:Q2:RB", "DOWN:Q1:RB"],
                },
                "Trim": {"ChannelNames": ["UP:Q1:TRIM", "", "", ""]},
            },
            "K": {
                "CommonNames": ["K1"],
                "Setpoint": {"ChannelNames": ["UP:K1:SP"]},
                "Monitor": {"ChannelNames": ["UP:K1:CMD"]},
            },
            "S": {"CommonNames": ["S1"], "Setpoint": {"ChannelNames": ["UP:S1:SP"]}},
        },
        "DOWN": {
            "Q": {
                "CommonNames": ["DOWN_Q1", "DOWN_Q2"],
                "Setpoint": {"ChannelNames": ["DOWN:Q1:SP", "DOWN:Q2:SP"]},
                "Monitor": {"ChannelNames": ["DOWN:Q1:RB", "DOWN:Q2:RB"]},
            },
        },
    }


def _family(klass: str, channels: int, *fields: str) -> dict[str, Any]:
    return {
        "class": klass,
        "devices": "names",
        "aliases": [],
        "description": f"The {klass} family.",
        "provenance": "stated",
        "channels": channels,
        "fields": {name: {"description": f"{name}.", "provenance": "stated"} for name in fields},
    }


def _direction(direction: str | None) -> dict[str, Any]:
    return {"direction": direction, "provenance": "stated", "override": False}


def _document() -> dict[str, Any]:
    return {
        "models": {
            "UP": {"name": "First", "description": "The first line.", "provenance": "stated"},
            "DOWN": {"name": "Second", "description": "The second line.", "provenance": "stated"},
        },
        "section_order": ["First", "Second"],
        "families": {
            "Q": _family("Quadrupole", 13, "Setpoint", "Monitor", "Trim"),
            "K": _family("Corrector", 2, "Setpoint", "Monitor"),
            "S": _family("Sextupole", 1, "Setpoint"),
        },
        "directions": {
            "Q.Setpoint": _direction("write"),
            "Q.Monitor": _direction("read"),
            "Q.Trim": _direction("write"),
            "K.Setpoint": _direction("write"),
            "K.Monitor": _direction("write"),
            "S.Setpoint": _direction("write"),
        },
        "judgments": {"Q": {"shared_pvs": "keep_all"}},
    }


def _rows(facility: Path, name: str) -> list[dict[str, Any]]:
    return yaml.safe_load((facility / LAYER_DIR / name).read_text(encoding="utf-8"))


def _by(rows: list[dict[str, Any]], key: str = "id") -> dict[str, dict[str, Any]]:
    return {row[key]: row for row in rows}


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> Path:
    facility = tmp_path_factory.mktemp("small") / "data" / "facility"
    write_records(Exports(ao=_export()), parse_mapping(_document()), facility)
    return facility


# -- the role rule and pair derivation -----------------------------------------


def test_a_direction_is_a_role_and_a_setpoint_pairs_with_its_family_monitor() -> None:
    assert field_roles(parse_mapping(_document())) == {
        "Q.Setpoint": FieldRole("setpoint", pair="Monitor"),
        "Q.Monitor": FieldRole("readback"),
        "Q.Trim": FieldRole("setpoint"),
        "K.Setpoint": FieldRole("setpoint"),
        "K.Monitor": FieldRole("setpoint"),
        "S.Setpoint": FieldRole("setpoint"),
    }


def test_a_setpoint_names_the_monitor_address_of_its_own_slot(small: Path) -> None:
    channels = _by(_rows(small, "channels.yaml"))
    assert {address: row.get("pair") for address, row in channels.items()} == {
        "UP:Q1:SP": "UP:Q1:RB",
        "UP:Q2:SP": "UP:Q2:RB",
        "DOWN:Q1:SP": "DOWN:Q1:RB",
        "DOWN:Q2:SP": "DOWN:Q2:RB",
        "UP:Q1:RB": None,
        "UP:Q2:RB": None,
        "DOWN:Q1:RB": None,
        "DOWN:Q2:RB": None,
        "UP:Q1:TRIM": None,
        "UP:K1:SP": None,
        "UP:K1:CMD": None,
        "UP:S1:SP": None,
    }


def test_every_channel_takes_the_role_of_its_field(small: Path) -> None:
    channels = _by(_rows(small, "channels.yaml"))
    readbacks = {address for address, row in channels.items() if row["role"] == "readback"}
    assert readbacks == {"UP:Q1:RB", "UP:Q2:RB", "DOWN:Q1:RB", "DOWN:Q2:RB"}
    assert {row["role"] for row in channels.values()} == {"setpoint", "readback"}


def test_no_readback_names_a_pair(small: Path) -> None:
    for row in _rows(small, "channels.yaml"):
        assert row["role"] == "setpoint" or "pair" not in row


# -- a null direction ------------------------------------------------------------


def test_a_null_direction_has_no_role() -> None:
    document = _document()
    document["directions"]["Q.Trim"] = _direction(None)
    with pytest.raises(ImportStop) as stop:
        field_roles(parse_mapping(document))
    assert stop.value.problem == "mapping-undecided"
    assert stop.value.format_message() == (
        "import mml: mapping-undecided: directions.Q.Trim.direction: write read or write"
    )


def test_a_null_direction_writes_no_record(tmp_path: Path) -> None:
    document = _document()
    document["directions"]["Q.Trim"] = _direction(None)
    facility = tmp_path / "data" / "facility"
    with pytest.raises(ImportStop, match="mapping-undecided: directions.Q.Trim.direction"):
        write_records(Exports(ao=_export()), parse_mapping(document), facility)
    assert not facility.exists()


def test_a_null_direction_stops_the_import_of_a_fixture_tree(tmp_path: Path) -> None:
    facility = tmp_path / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    document = yaml.safe_load((FIXTURES / "spear3" / MAPPING_FILE).read_text(encoding="utf-8"))
    document["directions"]["HCM.Setpoint"]["direction"] = None
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ImportStop) as stop:
        import_mml([FIXTURES / "spear3" / "spear3.storagering.ao.json"], facility)
    assert stop.value.format_message() == (
        "import mml: mapping-undecided: directions.HCM.Setpoint.direction: write read or write"
    )
    assert [p.relative_to(facility).as_posix() for p in facility.rglob("*") if p.is_file()] == [
        MAPPING_FILE
    ]


# -- shared endpoints ------------------------------------------------------------


def test_an_address_several_devices_bind_is_an_endpoint_of_each(small: Path) -> None:
    channels = _by(_rows(small, "channels.yaml"))
    for address in ("UP:Q2:SP", "UP:Q2:RB"):
        assert channels[address]["endpoint_of"] == ["First/Q2", "First/Q3"]
        assert "on" not in channels[address]
    assert channels["UP:Q1:SP"]["on"] == {"device": "First/Q1"}
    assert "endpoint_of" not in channels["UP:Q1:SP"]


def test_a_shared_endpoint_is_written_once(small: Path) -> None:
    ids = [row["id"] for row in _rows(small, "channels.yaml")]
    assert ids == sorted(set(ids))
    assert ids.count("UP:Q2:SP") == 1


def test_a_channel_record_never_carries_slices(small: Path) -> None:
    assert [row["id"] for row in _rows(small, "channels.yaml") if "slices" in row] == []


# -- segment dedup ---------------------------------------------------------------


def test_a_device_two_systems_list_is_one_record_of_its_own_system(small: Path) -> None:
    devices = [row["id"] for row in _rows(small, "devices.yaml")]
    assert devices == [
        "First/K1",
        "First/Q1",
        "First/Q2",
        "First/Q3",
        "First/S1",
        "Second/Q1",
        "Second/Q2",
    ]
    channels = _by(_rows(small, "channels.yaml"))
    assert channels["DOWN:Q1:SP"]["on"] == {"device": "Second/Q1"}


def test_same_named_families_of_two_systems_are_one_group(small: Path) -> None:
    groups = _by(_rows(small, "groups.yaml"))
    assert groups["Q"]["members"] == ["First/Q1", "First/Q2", "First/Q3", "Second/Q1", "Second/Q2"]


def test_the_small_import_states_no_id_twice_within_the_layer(small: Path) -> None:
    loaded = load_sources(small)
    assert [error.format_message() for error in loaded.errors] == []
    assert [error.format_message() for error in combine(loaded.sources).errors] == []


# -- the fixture exports: decks, slices and planes -------------------------------

at = pytest.importorskip("at")


def _import(root: Path, tree: str) -> Path:
    facility = root / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / tree / MAPPING_FILE, target)
    import_mml([FIXTURES / tree / f"{stem}.ao.json" for stem in TREES[tree]], facility)
    return facility


@pytest.fixture(scope="module")
def spear3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _import(tmp_path_factory.mktemp("spear3"), "spear3")


@pytest.fixture(scope="module")
def nsls2(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _import(tmp_path_factory.mktemp("nsls2"), "nsls2")


def _combined(facility: Path) -> CombineResult:
    loaded = load_sources(facility)
    assert [error.format_message() for error in loaded.errors] == []
    return combine(loaded.sources)


@pytest.mark.parametrize("tree", sorted(TREES))
def test_a_wired_shared_endpoint_names_each_device_in_a_slice(
    tree: str, request: pytest.FixtureRequest
) -> None:
    facility: Path = request.getfixturevalue(tree)
    shared = {
        row["id"]: row["endpoint_of"]
        for row in _rows(facility, "channels.yaml")
        if "endpoint_of" in row
    }
    wired = [
        record
        for model in _rows(facility, "models.yaml")
        for record in model["wiring"]
        if record["address"] in shared
    ]
    assert wired
    for record in wired:
        assert "element" not in record
        assert sorted({piece["device"] for piece in record["slices"]}) == shared[record["address"]]
        elements = [piece["element"] for piece in record["slices"]]
        assert len(elements) == len(set(elements))


def test_a_record_wiring_one_element_is_an_address_one_device_binds(spear3: Path) -> None:
    owned = {row["id"] for row in _rows(spear3, "channels.yaml") if "on" in row}
    (model,) = _rows(spear3, "models.yaml")
    single = [record["address"] for record in model["wiring"] if "element" in record]
    assert single
    assert [address for address in single if address not in owned] == []
    assert [
        record["address"]
        for record in model["wiring"]
        if "slices" in record and "element" in record
    ] == []


def test_two_exports_merged_state_no_id_twice(nsls2: Path) -> None:
    assert [error.format_message() for error in _combined(nsls2).errors] == []
    for name in ("devices.yaml", "channels.yaml", "groups.yaml"):
        ids = [row["id"] for row in _rows(nsls2, name)]
        assert len(ids) == len(set(ids)), name


def test_the_storage_model_gives_disjoint_corrector_candidates_per_plane(nsls2: Path) -> None:
    from osprey.simulation.engines.pyat import plane

    ao = json.loads((FIXTURES / "nsls2" / "nsls2.storagering.ao.json").read_text(encoding="utf-8"))
    document = _combined(nsls2).document
    setpoints = {row["id"] for row in document["channels"] if row.get("role") == "setpoint"}
    model = _by(document["models"], "name")["StorageRing"]
    candidates: dict[str, set[str]] = {"x": set(), "y": set()}
    for record in model["wiring"]:
        found = plane(record)
        if found is not None and record["address"] in setpoints:
            candidates[found].add(record["address"])

    def stated(family: str) -> set[str]:
        return {name.strip() for name in ao[family]["Setpoint"]["ChannelNames"] if name.strip()}

    assert candidates["x"] == stated("HCM")
    assert candidates["y"] == stated("VCM")
    assert len(candidates["x"]) == len(candidates["y"]) == 180
    assert candidates["x"].isdisjoint(candidates["y"])


# -- merging with an authored model ----------------------------------------------


def test_an_imported_model_merges_per_field_with_an_authored_one(
    spear3: Path, tmp_path: Path
) -> None:
    facility = tmp_path / "facility"
    shutil.copytree(spear3, facility)
    (imported,) = _rows(facility, "models.yaml")
    settings = {"pyat": {"solve": "closed_orbit"}}
    authored = [{"name": "StorageRing", "settings": settings}]
    (facility / "models.yaml").write_text(yaml.safe_dump(authored), encoding="utf-8")

    result = _combined(facility)

    assert [error.format_message() for error in result.errors] == []
    (model,) = result.document["models"]
    assert model["name"] == "StorageRing"
    assert model["engine"] == imported["engine"] == "pyat"
    assert model["deck"] == imported["deck"]
    assert model["settings"] == settings
    assert [record["address"] for record in model["wiring"]] == [
        record["address"] for record in imported["wiring"]
    ]


def test_a_field_both_layers_state_alike_merges(spear3: Path, tmp_path: Path) -> None:
    facility = tmp_path / "facility"
    shutil.copytree(spear3, facility)
    authored = [{"name": "StorageRing", "engine": "pyat"}]
    (facility / "models.yaml").write_text(yaml.safe_dump(authored), encoding="utf-8")
    assert [error.format_message() for error in _combined(facility).errors] == []


def test_settings_the_two_layers_state_in_part_conflict_whole(tmp_path: Path) -> None:
    twiss = {"beta": [4.5, 4.8], "alpha": [-0.5, -0.6]}
    imported = [{"name": "line", "engine": "pyat", "settings": {"pyat": {"solve": "single_pass"}}}]
    authored = [{"name": "line", "settings": {"pyat": {"twiss_in": twiss}}}]
    layer = tmp_path / LAYER_DIR
    layer.mkdir(parents=True)
    (layer / "models.yaml").write_text(yaml.safe_dump(imported), encoding="utf-8")
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(authored), encoding="utf-8")

    (error,) = _combined(tmp_path).errors

    assert (error.kind, error.record_kind, error.record_id) == ("layer-conflict", "model", "line")
    assert error.detail.startswith("`settings` differs:")
    assert "authored=" in error.detail and "mml=" in error.detail
    assert set(error.sources) == {"models.yaml", "imported/mml/models.yaml"}


def test_the_imported_transfer_line_conflicts_with_authored_twiss(
    nsls2: Path, tmp_path: Path
) -> None:
    facility = tmp_path / "facility"
    shutil.copytree(nsls2, facility)
    imported = _by(_rows(facility, "models.yaml"), "name")["LTB"]
    assert imported["settings"]["pyat"]["solve"] == "single_pass"
    authored = [{"name": "LTB", "settings": {"pyat": {"twiss_in": {"beta": [1.0, 1.0]}}}}]
    (facility / "models.yaml").write_text(yaml.safe_dump(authored), encoding="utf-8")

    (error,) = _combined(facility).errors

    assert (error.kind, error.record_kind, error.record_id) == ("layer-conflict", "model", "LTB")
    assert error.detail.startswith("`settings` differs:")


def test_an_exported_family_the_mapping_leaves_out_writes_no_record(tmp_path: Path) -> None:
    document = _document()
    del document["families"]["S"]
    del document["directions"]["S.Setpoint"]
    facility = tmp_path / "data" / "facility"
    with pytest.raises(MappingProblems, match="families: leaves out the exported family S"):
        write_records(Exports(ao=_export()), parse_mapping(document), facility)
    assert not facility.exists()
