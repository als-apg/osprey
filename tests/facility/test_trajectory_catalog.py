"""The trajectory catalog: the devices a model's first-turn response matrix covers.

``trajectory_catalog`` reads a model's trajectory response matrix against the
facility file and answers ``{bpms, hcms, vcms, groups}``: one entry per matrix
name, in matrix order — the ``x`` rows for the monitors, the horizontal then the
vertical columns for the correctors — each ``{name, device, label, places, s,
plane, groups}``; and every facility group listing a matrix device, its members
given as matrix names in matrix order. A matrix name with no device stops the
build, naming it. The facts carry the catalog per model under
``trajectory_catalog``, read from the pyAML view the render wrote.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.facility.views import pyaml
from osprey.facility.views.facts import (
    FACTS_KEYS,
    facts_document,
    trajectory_matrices,
    zero_source_facts,
)
from osprey.facility.views.pyaml import (
    PYAML_DIR,
    TRAJECTORY_RM_FILE,
    trajectory_catalog,
)

if TYPE_CHECKING:
    from tests._builds import BuiltProject

#: Every key of an entry, in order.
ENTRY_KEYS = ("name", "device", "label", "places", "s", "plane", "groups")

#: Every key of a catalog, in order.
CATALOG_KEYS = ("bpms", "hcms", "vcms", "groups")


def _doc() -> dict[str, Any]:
    """A small facility: two monitors, one corrector per plane, one combined corrector.

    The words are deliberately not any shipped facility's, so nothing in the
    view can lean on them.
    """
    return {
        "identity": {"code": "test"},
        "places": [
            {"id": "RING", "level": "machine"},
            {"id": "RING/ARC2", "level": "arc", "names": ["second arc"]},
            {"id": "RING/ARC1", "level": "arc"},
        ],
        "devices": [
            {"id": "MON-B", "label": "Monitor B", "place": "RING/ARC2", "s": 7.5},
            {"id": "MON-A", "names": ["mon a", "aa"], "place": "RING/ARC1", "s": 1.25},
            {"id": "KICK-H", "place": "RING/ARC1", "s": 2},
            {"id": "KICK-V", "label": "Vertical kicker"},
            {"id": "KICK-HV", "label": "Both", "place": "RING/ARC2", "s": None},
            {"id": "OTHER", "place": "RING/ARC1", "s": 3.0},
        ],
        "channels": [
            {"id": "MON-A:X", "on": {"device": "MON-A"}},
            {"id": "MON-A:Y", "on": {"device": "MON-A"}},
            {"id": "MON-B:X", "on": {"device": "MON-B"}},
            {"id": "MON-B:Y", "on": {"device": "MON-B"}},
            {"id": "KICK-H:SP", "on": {"device": "KICK-H"}},
            {"id": "KICK-V:SP", "on": {"device": "KICK-V"}},
            {"id": "KICK-HV:H:SP", "on": {"device": "KICK-HV"}},
            {"id": "KICK-HV:V:SP", "on": {"device": "KICK-HV"}},
            {"id": "OTHER:SP", "on": {"device": "OTHER"}},
        ],
        "groups": [
            {"id": "MONS", "label": "The monitors", "members": ["MON-B", "MON-A"]},
            {"id": "KICKS", "names": ["kickers"], "members": ["KICK-HV", "KICK-H", "KICK-V"]},
            {"id": "ALL", "members": ["OTHER", "KICK-V", "MON-A"]},
            {"id": "NONE", "members": ["OTHER"]},
        ],
    }


def _matrix() -> dict[str, Any]:
    """The matrix: monitor B before A along the lattice, columns H then V."""
    return {
        "observable_names": ["MON-B:X", "MON-A:X", "MON-B:Y", "MON-A:Y"],
        "observable_planes": ["H", "H", "V", "V"],
        "variable_names": ["KICK-HV:H:SP", "KICK-H:SP", "KICK-V:SP", "KICK-HV:V:SP"],
        "variable_planes": ["H", "H", "V", "V"],
    }


def _by_name(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for role in ("bpms", "hcms", "vcms") for entry in catalog[role]}


# --- the entries ----------------------------------------------------------------------


def test_the_catalog_keeps_the_key_order() -> None:
    catalog = trajectory_catalog(_doc(), "M", _matrix())
    assert tuple(catalog) == CATALOG_KEYS
    for entry in _by_name(catalog).values():
        assert tuple(entry) == ENTRY_KEYS


def test_the_monitors_are_the_x_rows_in_matrix_order() -> None:
    catalog = trajectory_catalog(_doc(), "M", _matrix())
    assert [entry["name"] for entry in catalog["bpms"]] == ["MON-B:X", "MON-A:X"]
    assert [entry["device"] for entry in catalog["bpms"]] == ["MON-B", "MON-A"]
    assert all(entry["plane"] is None for entry in catalog["bpms"])


def test_the_correctors_are_the_columns_by_plane_in_matrix_order() -> None:
    catalog = trajectory_catalog(_doc(), "M", _matrix())
    assert [entry["name"] for entry in catalog["hcms"]] == ["KICK-HV:H:SP", "KICK-H:SP"]
    assert [entry["name"] for entry in catalog["vcms"]] == ["KICK-V:SP", "KICK-HV:V:SP"]
    assert {entry["plane"] for entry in catalog["hcms"]} == {"x"}
    assert {entry["plane"] for entry in catalog["vcms"]} == {"y"}


def test_a_combined_corrector_is_one_entry_per_plane_on_one_device() -> None:
    entries = _by_name(trajectory_catalog(_doc(), "M", _matrix()))
    assert entries["KICK-HV:H:SP"]["device"] == entries["KICK-HV:V:SP"]["device"] == "KICK-HV"


def test_the_label_is_the_device_label_else_its_first_name_else_its_id() -> None:
    entries = _by_name(trajectory_catalog(_doc(), "M", _matrix()))
    assert entries["MON-B:X"]["label"] == "Monitor B"
    assert entries["MON-A:X"]["label"] == "aa"
    assert entries["KICK-H:SP"]["label"] == "KICK-H"


def test_the_places_are_the_chain_root_to_leaf_in_the_facilitys_words() -> None:
    entries = _by_name(trajectory_catalog(_doc(), "M", _matrix()))
    assert entries["MON-B:X"]["places"] == [
        {"id": "RING", "level": "machine"},
        {"id": "RING/ARC2", "level": "arc"},
    ]
    assert entries["KICK-V:SP"]["places"] == []


def test_a_place_without_a_level_has_none() -> None:
    doc = _doc()
    del doc["places"][0]["level"]
    entries = _by_name(trajectory_catalog(doc, "M", _matrix()))
    assert entries["MON-A:X"]["places"][0] == {"id": "RING", "level": None}


def test_s_is_the_devices_position_in_metres_or_none() -> None:
    entries = _by_name(trajectory_catalog(_doc(), "M", _matrix()))
    assert entries["MON-B:X"]["s"] == 7.5
    assert entries["KICK-H:SP"]["s"] == 2.0
    assert entries["KICK-V:SP"]["s"] is None
    assert entries["KICK-HV:V:SP"]["s"] is None


def test_each_entry_names_the_groups_listing_its_device_sorted() -> None:
    entries = _by_name(trajectory_catalog(_doc(), "M", _matrix()))
    assert entries["MON-A:X"]["groups"] == ["ALL", "MONS"]
    assert entries["KICK-H:SP"]["groups"] == ["KICKS"]


# --- the groups -----------------------------------------------------------------------


def test_the_groups_are_those_listing_a_matrix_device_members_as_matrix_names() -> None:
    groups = trajectory_catalog(_doc(), "M", _matrix())["groups"]
    assert groups == [
        {"id": "ALL", "label": "ALL", "members": ["MON-A:X", "KICK-V:SP"]},
        {
            "id": "KICKS",
            "label": "kickers",
            "members": ["KICK-HV:H:SP", "KICK-H:SP", "KICK-V:SP", "KICK-HV:V:SP"],
        },
        {"id": "MONS", "label": "The monitors", "members": ["MON-B:X", "MON-A:X"]},
    ]


# --- the stop -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        lambda doc: doc["channels"].remove({"id": "KICK-V:SP", "on": {"device": "KICK-V"}}),
        lambda doc: doc["channels"][5].update(on={}),
        lambda doc: doc["devices"].pop(3),
    ],
    ids=["no-channel", "channel-on-no-device", "device-not-a-record"],
)
def test_a_matrix_name_with_no_device_stops_the_build_naming_it(change: Any) -> None:
    doc = _doc()
    change(doc)
    with pytest.raises(FacilityBuildError) as raised:
        trajectory_catalog(doc, "M", _matrix())
    assert raised.value.kind == "reference-missing"
    assert raised.value.record_id == "KICK-V:SP"
    assert "KICK-V:SP" in str(raised.value)
    assert "M" in raised.value.detail


def test_the_catalog_names_no_facility_word() -> None:
    source = "".join(
        inspect.getsource(function)
        for function in (pyaml.trajectory_catalog, pyaml._catalog_entry, pyaml._place_chain)
    )
    for word in ("SR:", "SR/", "BPM", "SECT", "CLEANING", "ring", "sector"):
        assert word not in source, word


# --- the facts ------------------------------------------------------------------------


def test_the_facts_carry_the_catalog_of_each_model_with_a_matrix() -> None:
    doc = _doc()
    facts = facts_document(doc, [], None, None, {"M": _matrix()})
    assert facts["trajectory_catalog"] == {"M": trajectory_catalog(doc, "M", _matrix())}


def test_the_facts_without_a_matrix_carry_an_empty_catalog() -> None:
    assert facts_document(_doc(), [])["trajectory_catalog"] == {}


def test_the_zero_source_facts_carry_the_catalog_key() -> None:
    assert zero_source_facts({})["trajectory_catalog"] == {}
    assert "trajectory_catalog" in FACTS_KEYS


def test_the_matrices_are_read_from_the_pyaml_view_on_disk(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for model in ("B", "A"):
        directory = data / PYAML_DIR / model
        directory.mkdir(parents=True)
        (directory / TRAJECTORY_RM_FILE).write_text(json.dumps(_matrix()), encoding="utf-8")
    (data / PYAML_DIR / "C").mkdir()
    matrices = trajectory_matrices(data)
    assert list(matrices) == ["A", "B"]
    assert matrices["A"] == _matrix()


def test_no_pyaml_view_reads_no_matrix(tmp_path: Path) -> None:
    assert trajectory_matrices(tmp_path / "data") == {}


def test_the_catalog_does_not_change_the_facility_file() -> None:
    doc = _doc()
    before = copy.deepcopy(doc)
    trajectory_catalog(doc, "M", _matrix())
    assert doc == before


# --- the example facility -------------------------------------------------------------


@pytest.mark.slow
def test_the_example_catalog_is_the_matrix_devices_in_matrix_order(
    built_control_assistant: BuiltProject,
) -> None:
    from osprey.facility.sources import group_families

    build = built_control_assistant.build_dir
    facts = json.loads((build / "data/facility_facts.json").read_text(encoding="utf-8"))
    matrix = json.loads(
        (build / "data" / PYAML_DIR / "SR" / TRAJECTORY_RM_FILE).read_text(encoding="utf-8")
    )
    catalog = facts["trajectory_catalog"]["SR"]
    rows, planes = matrix["observable_names"], matrix["observable_planes"]
    assert [entry["name"] for entry in catalog["bpms"]] == [
        row for row, plane in zip(rows, planes, strict=True) if plane == "H"
    ]
    assert [entry["name"] for entry in [*catalog["hcms"], *catalog["vcms"]]] == matrix[
        "variable_names"
    ]
    facility = built_control_assistant.facility
    devices = {device["id"]: device for device in facility["devices"]}
    places = {place["id"]: place for place in facility.get("places", [])}
    for entry in _by_name(catalog).values():
        device = devices[entry["device"]]
        assert entry["label"]
        assert entry["places"], entry["name"]
        assert entry["places"][-1]["id"] == device["place"]
        for place in entry["places"]:
            assert place["level"] == places[place["id"]].get("level")
    names = set(_by_name(catalog))
    assert catalog["groups"]
    for group in catalog["groups"]:
        assert set(group["members"]) <= names
    (record,) = [record for record in facility["models"] if record["name"] == "SR"]
    (bpm_group,) = group_families(record["measurement"]["groups"]["bpm"])
    (listed,) = [group for group in catalog["groups"] if group["id"] == bpm_group]
    assert listed["members"] == [entry["name"] for entry in catalog["bpms"]]
