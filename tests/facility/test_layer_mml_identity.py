"""The mml layer's identity model on the nsls2 and spear3 fixture exports.

A device is the export's ``CommonNames`` entry at its slot's position, or
what the mapping's ``devices`` answer says where the export names none; an
address bound by several devices is one channel naming each of them in
``endpoint_of``, and slots naming one device, within a family or across the
export's systems, are one record.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from functools import reduce
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers.mml.identity import (
    axis_twins,
    common_class,
    device_ids,
    endpoints,
)
from osprey.facility.layers.mml.importer import LAYER_DIR, import_mml
from osprey.facility.layers.mml.mapping import MAPPING_FILE, ImportStop, SameAs
from osprey.facility.validate import run_stages
from osprey.services.mml.family import FamilyView

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

TREES: dict[str, tuple[str, ...]] = {
    "spear3": ("spear3.storagering",),
    "nsls2": ("nsls2.storagering", "nsls2.ltb"),
}


def _import(root: Path, tree: str) -> Path:
    facility = root / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / tree / MAPPING_FILE, target)
    import_mml([FIXTURES / tree / f"{stem}.ao.json" for stem in TREES[tree]], facility)
    return facility


def _rows(facility: Path, name: str) -> list[dict[str, Any]]:
    return yaml.safe_load((facility / LAYER_DIR / name).read_text(encoding="utf-8"))


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in rows}


def _ao(stem: str) -> dict[str, Any]:
    tree = stem.split(".", 1)[0]
    return json.loads((FIXTURES / tree / f"{stem}.ao.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", params=sorted(TREES))
def imported(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _import(tmp_path_factory.mktemp(request.param), request.param)


@pytest.fixture(scope="module")
def spear3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _import(tmp_path_factory.mktemp("spear3-only"), "spear3")


@pytest.fixture(scope="module")
def nsls2(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _import(tmp_path_factory.mktemp("nsls2-only"), "nsls2")


# -- CommonName by position ---------------------------------------------------


def test_no_device_id_is_a_family_ordinal(imported: Path) -> None:
    devices = [row["id"] for row in _rows(imported, "devices.yaml")]
    tokens = {group["id"] for group in _rows(imported, "groups.yaml")}
    ordinal = re.compile(rf"/(?:{'|'.join(map(re.escape, sorted(tokens)))})_\d+$")
    assert [device for device in devices if ordinal.search(device)] == []
    assert len(devices) == len(set(devices))


def test_each_device_is_its_common_name_at_its_slot(spear3: Path) -> None:
    ao = _ao("spear3.storagering")
    channels = _by_id(_rows(spear3, "channels.yaml"))
    devices = _by_id(_rows(spear3, "devices.yaml"))
    for name, address in zip(
        ao["BPMx"]["CommonNames"], ao["BPMx"]["Monitor"]["ChannelNames"], strict=True
    ):
        device = f"StorageRing/{name}"
        assert devices[device]["names"] == [name]
        assert channels[address.strip()]["on"] == {"device": device}


def test_a_family_answered_address_is_named_by_its_addresses(spear3: Path) -> None:
    ao = _ao("spear3.storagering")
    devices = _by_id(_rows(spear3, "devices.yaml"))
    channels = _by_id(_rows(spear3, "channels.yaml"))
    gauges = [address.strip() for address in ao["IonGauge"]["Monitor"]["ChannelNames"]]
    owners = [channels[address]["on"]["device"] for address in gauges]
    assert len(set(owners)) == len(gauges)
    assert owners[0] == "StorageRing/VG01_AM1"
    assert all(owner in devices and "names" not in devices[owner] for owner in owners)


def test_families_naming_one_device_share_it(nsls2: Path) -> None:
    """Both corrector planes of one CommonName are one combined device."""
    ao = _ao("nsls2.storagering")
    devices = _by_id(_rows(nsls2, "devices.yaml"))
    channels = _by_id(_rows(nsls2, "channels.yaml"))
    groups = _by_id(_rows(nsls2, "groups.yaml"))
    name = ao["HCM"]["CommonNames"][0]
    device = f"StorageRing/{name}"
    for family in ("HCM", "VCM"):
        setpoint = ao[family]["Setpoint"]["ChannelNames"][0].strip()
        assert channels[setpoint]["on"] == {"device": device}
        assert device in groups[family]["members"]
    assert devices[device]["class"] == "Corrector"


# -- shared endpoints ---------------------------------------------------------


def test_each_address_is_written_once(imported: Path) -> None:
    text = (imported / LAYER_DIR / "channels.yaml").read_text(encoding="utf-8")
    ids = Counter(row["id"] for row in yaml.safe_load(text))
    assert [address for address, count in ids.items() if count > 1] == []


def test_a_shared_address_names_every_owner(imported: Path) -> None:
    devices = _by_id(_rows(imported, "devices.yaml"))
    shared = [row for row in _rows(imported, "channels.yaml") if "endpoint_of" in row]
    assert shared
    for row in shared:
        assert "on" not in row
        assert "slices" not in row
        assert len(row["endpoint_of"]) > 1
        assert row["endpoint_of"] == sorted(set(row["endpoint_of"]))
        assert all(device in devices for device in row["endpoint_of"])


def test_one_supply_is_an_endpoint_of_every_magnet_on_it(spear3: Path) -> None:
    ao = _ao("spear3.storagering")
    channels = _by_id(_rows(spear3, "channels.yaml"))
    bends = sorted({f"StorageRing/{name}" for name in ao["BEND"]["CommonNames"]})
    address = ao["BEND"]["Monitor"]["ChannelNames"][0].strip()
    assert channels[address]["endpoint_of"] == bends


def test_a_family_folded_into_another_binds_that_family_s_devices(nsls2: Path) -> None:
    """Each position monitor is one device carrying both axes' channels."""
    channels = _by_id(_rows(nsls2, "channels.yaml"))
    groups = _by_id(_rows(nsls2, "groups.yaml"))
    for stem, model, count in (("nsls2.storagering", "StorageRing", 180), ("nsls2.ltb", "LTB", 7)):
        ao = _ao(stem)
        devices = [f"{model}/{name}" for name in ao["BPMx"]["CommonNames"]]
        assert len(devices) == count
        for family in ("BPMx", "BPMy"):
            addresses = ao[family]["Monitor"]["ChannelNames"]
            assert len(addresses) == count
            bound = [(a.strip(), d) for a, d in zip(addresses, devices, strict=True) if a.strip()]
            assert bound
            assert [channels[address]["on"]["device"] for address, _ in bound] == [
                device for _, device in bound
            ]
    assert groups["BPMy"]["members"] == groups["BPMx"]["members"]
    assert len(groups["BPMx"]["members"]) == 187


def test_an_address_both_axes_of_one_device_bind_is_on_that_device(nsls2: Path) -> None:
    ao = _ao("nsls2.ltb")
    channels = _by_id(_rows(nsls2, "channels.yaml"))
    address = ao["BPMx"]["Sum"]["ChannelNames"][0].strip()
    assert address == ao["BPMy"]["Sum"]["ChannelNames"][0].strip()
    assert channels[address]["on"] == {"device": f"LTB/{ao['BPMx']['CommonNames'][0]}"}


# -- dedup --------------------------------------------------------------------


def test_slots_repeating_a_common_name_are_one_device(nsls2: Path) -> None:
    ao = _ao("nsls2.storagering")
    groups = _by_id(_rows(nsls2, "groups.yaml"))
    channels = _by_id(_rows(nsls2, "channels.yaml"))
    names = ao["SQ"]["CommonNames"]
    assert len(names) > len(set(names))
    assert groups["SQ"]["members"] == sorted({f"StorageRing/{name}" for name in names})
    address = ao["SQ"]["Setpoint"]["ChannelNames"][0].strip()
    assert channels[address]["on"] == {"device": f"StorageRing/{names[0]}"}


def test_the_imported_tree_states_no_id_twice(imported: Path) -> None:
    report = run_stages(imported, project_name="demo")
    assert [e.format_message() for e in report.errors if e.kind == "layer-duplicate"] == []
    assert report.failed != "load"


def _view(system: str, family: str, names: list[str], addresses: list[str]) -> FamilyView:
    body = {"CommonNames": names, "Setpoint": {"ChannelNames": addresses}}
    return FamilyView(system, family, body)


def test_a_device_listed_under_two_systems_is_one_record() -> None:
    """A slot whose address and name carry another system's token is that system's device."""
    views = [
        _view("Up", "Q", ["Up_Q1", "Down_Q1"], ["Up:Q1:SP", "Down:Q1:SP"]),
        _view("Down", "Q", ["Down_Q1", "Down Q2"], ["Down:Q1:SP", "Down:Q2:SP"]),
    ]
    ids = device_ids(views, {"Up": "First", "Down": "Second"})
    assert ids == [["First/Q1", "Second/Q1"], ["Second/Q1", "Second/Q2"]]
    assert endpoints(views, ids) == {
        "Up:Q1:SP": ["First/Q1"],
        "Down:Q1:SP": ["Second/Q1"],
        "Down:Q2:SP": ["Second/Q2"],
    }


def test_an_address_two_devices_bind_names_both() -> None:
    views = [
        _view("Up", "A", ["A1", "A2"], ["Up:S:SP", "Up:S:SP"]),
        _view("Up", "B", ["B1"], ["Up:S:SP"]),
    ]
    ids = device_ids(views, {"Up": "First"})
    assert endpoints(views, ids) == {"Up:S:SP": ["First/A1", "First/A2", "First/B1"]}


# -- what the mapping decides -------------------------------------------------


def _stop(views: list[FamilyView], devices: dict[str, Any] | None = None) -> ImportStop:
    with pytest.raises(ImportStop) as stop:
        device_ids(views, {"Up": "First"}, devices)
    return stop.value


def test_a_family_naming_no_device_stops_until_the_mapping_decides() -> None:
    views = [
        _view("Up", "G", [], ["Up:VG1/AM1", "Up:VG2/AM1"]),
        _view("Up", "Q", ["Q1"], ["Up:Q1:SP"]),
        _view("Up", "H", ["H1", ""], ["Up:H1:SP", "Up:H2:SP"]),
    ]
    stop = _stop(views)
    assert stop.problem == "mapping-undecided"
    assert stop.format_message().splitlines() == [
        "import mml: mapping-undecided: families.G.devices: "
        "write address, a list of names or {same_as: <family>}",
        "import mml: mapping-undecided: families.H.devices: "
        "write address, a list of names or {same_as: <family>}",
    ]


def test_names_on_a_family_naming_no_device_is_invalid() -> None:
    stop = _stop([_view("Up", "H", ["H1", ""], ["Up:H1:SP", "Up:H2:SP"])], {"H": "names"})
    assert stop.format_message() == (
        "import mml: mapping-invalid: families.H.devices: H in Up names no device 2; "
        "write address, a list of names or {same_as: <family>}"
    )


def test_address_names_each_slot_by_its_address_token() -> None:
    views = [_view("Up", "G", [], ["Up:VG1/AM1", "Up:VG1/AM1:B"])]
    ids = device_ids(views, {"Up": "First"}, {"G": "address"})
    assert ids == [["First/VG1_AM1", "First/VG1_AM1_1"]]


def test_address_overrides_the_names_the_export_states() -> None:
    views = [_view("Up", "Q", ["Q1", "Q2"], ["Up:MA:SP", "Up:MB:SP"])]
    assert device_ids(views, {"Up": "First"}, {"Q": "address"}) == [["First/MA", "First/MB"]]


def test_an_address_id_is_never_handed_to_two_slots() -> None:
    views = [
        _view("Up", "A", [], ["Up:X:A", "Up:X:B"]),
        _view("Up", "C", [], ["Up:W:A", "Up:X:D"]),
    ]
    ids = device_ids(views, {"Up": "First"}, {"A": "address", "C": "address"})
    assert ids == [["First/X", "First/X_1"], ["First/W", "First/X_2"]]


def test_a_braced_address_names_its_device_through_the_brace() -> None:
    views = [
        _view("SR", "B", [], ["SR:C30-BI{BPM:1}Pos:Y-I", "SR:C30-BI{BPM:5}Pos:Y-I"]),
        _view("SR", "L", [], ["LTB-BI{BPM:1}Pos:X-I", "LTB-BI{BPM:2}Pos:X-I"]),
    ]
    ids = device_ids(views, {"SR": "First"}, {"B": "address", "L": "address"})
    assert ids == [
        ["First/C30_BI_BPM_1", "First/C30_BI_BPM_5"],
        ["First/LTB_BI_BPM_1", "First/LTB_BI_BPM_2"],
    ]


def test_a_list_names_each_slot_in_order() -> None:
    views = [_view("Up", "G", [], ["Up:VG1/AM1", "Up:VG2/AM1"])]
    ids = device_ids(views, {"Up": "First"}, {"G": ("Gauge_A", "Gauge_B")})
    assert ids == [["First/Gauge_A", "First/Gauge_B"]]


def test_a_list_of_another_length_than_the_devices_is_invalid() -> None:
    stop = _stop([_view("Up", "G", [], ["Up:VG1/AM1", "Up:VG2/AM1"])], {"G": ("Gauge_A",)})
    assert stop.format_message() == (
        "import mml: mapping-invalid: families.G.devices: lists 1 names and G has 2 devices in Up"
    )


def _axes() -> list[FamilyView]:
    return [
        _view("Up", "Py", [], ["Up{P:1}Pos:Y", "Up{P:2}Pos:Y"]),
        _view("Up", "Px", ["P1", "P2"], ["Up{P:1}Pos:X", "Up{P:2}Pos:X"]),
    ]


def test_same_as_gives_each_slot_the_named_family_s_device() -> None:
    views = _axes()
    ids = device_ids(views, {"Up": "First"}, {"Py": SameAs("Px")})
    assert ids == [["First/P1", "First/P2"], ["First/P1", "First/P2"]]
    assert endpoints(views, ids)["Up{P:1}Pos:Y"] == ["First/P1"]


def test_same_as_follows_a_list_too() -> None:
    views = [
        _view("Up", "Py", [], ["Up{P:1}Pos:Y"]),
        _view("Up", "Px", [], ["Up{P:1}Pos:X"]),
    ]
    ids = device_ids(views, {"Up": "First"}, {"Py": SameAs("Px"), "Px": ("Monitor_1",)})
    assert ids == [["First/Monitor_1"], ["First/Monitor_1"]]


@pytest.mark.parametrize(
    ("answers", "line"),
    [
        (
            {"Py": SameAs("Px"), "Px": "address"},
            "Px is not identified by names or a list; name a family that is",
        ),
        (
            {"Py": SameAs("Pz"), "Pz": SameAs("Px")},
            "Pz is not identified by names or a list; name a family that is",
        ),
        ({"Py": SameAs("Other")}, "Up carries no family Other"),
        ({"Py": SameAs("Q")}, "Py has 2 devices in Up and Q has 1"),
    ],
)
def test_same_as_a_family_that_cannot_lend_its_devices_is_invalid(
    answers: dict[str, Any], line: str
) -> None:
    views = [*_axes(), _view("Up", "Q", ["Q1"], ["Up:Q1:SP"])]
    stop = _stop(views, answers)
    assert stop.format_message().splitlines() == [
        f"import mml: mapping-invalid: families.Py.devices: {line}"
    ]


def test_families_binding_the_two_axes_of_one_address_are_twins() -> None:
    py, px = _axes()
    assert axis_twins(py, px)
    assert not axis_twins(py, py)
    assert not axis_twins(py, _view("Up", "Q", ["Q1", "Q2"], ["Up{Q:1}Pos:X", "Up{P:2}Pos:X"]))
    assert not axis_twins(py, _view("Up", "Q", ["Q1"], ["Up{P:1}Pos:X"]))


def test_a_nameless_family_the_mapping_leaves_out_stops_the_import(tmp_path: Path) -> None:
    facility = tmp_path / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    document = yaml.safe_load((FIXTURES / "nsls2" / MAPPING_FILE).read_text(encoding="utf-8"))
    del document["families"]["BPMy"]["devices"]
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ImportStop) as stop:
        import_mml(
            [FIXTURES / "nsls2" / f"nsls2.{stem}.ao.json" for stem in ("storagering", "ltb")],
            facility,
        )
    assert stop.value.format_message() == (
        "import mml: mapping-undecided: families.BPMy.devices: "
        "write address, a list of names or {same_as: <family>}"
    )
    assert not (facility / LAYER_DIR / "devices.yaml").exists()


def test_a_device_class_does_not_depend_on_family_order() -> None:
    branches = {"Left": "LeftRoot", "Right": "RightRoot", "Under": "LeftRoot"}
    folded = {
        reduce(lambda a, b: common_class(a, b, branches), order)
        for order in permutations(("Left", "Right", "Under"))
    }
    assert folded == {None}
