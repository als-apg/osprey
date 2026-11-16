"""The facility index a relations import resolves against."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers import LayerSourceInvalid
from osprey.facility.layers.relations.facility import FacilityIndex


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, document in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = document if isinstance(document, str) else yaml.safe_dump(document)
        path.write_text(text, encoding="utf-8")
    return root


def _tree(tmp_path: Path, extra: dict[str, Any] | None = None) -> Path:
    files: dict[str, Any] = {
        "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}],
        "records/devices.yaml": [
            {"id": "QF1", "class": "Quadrupole", "place": "SR/S01"},
            {"id": "BPM1", "place": "SR/S01"},
        ],
        "records/groups.yaml": [{"id": "QUADS", "members": ["QF1"]}],
        "imported/controls/devices.yaml": [
            {"id": "PS1", "class": "PowerSupply"},
            {"id": "BPM1", "class": "BPM"},
        ],
        "imported/controls/places.yaml": [{"id": "BR"}, {"id": "BR/S01"}],
        "imported/cables/places.yaml": [{"id": "SR/RACK1"}],
        "imported/cables/devices.yaml": [{"id": "OLD1", "class": "Quadrupole"}],
        "imported/cables/groups.yaml": [{"id": "OLDGROUP", "members": ["OLD1"]}],
    }
    files.update(extra or {})
    return _write(tmp_path, files)


class TestReplacedLayer:
    def test_the_replaced_layers_records_are_absent(self, tmp_path: Path) -> None:
        index = FacilityIndex.load(_tree(tmp_path), "cables")
        assert "OLD1" not in index.devices
        assert "SR/RACK1" not in index.places
        assert "OLDGROUP" not in index.groups

    def test_devices_of_another_imported_layer_are_present(self, tmp_path: Path) -> None:
        index = FacilityIndex.load(_tree(tmp_path), "cables")
        assert index.devices["PS1"] == "PowerSupply"
        assert index.devices["QF1"] == "Quadrupole"
        assert "BR/S01" in index.places

    def test_a_class_stated_only_by_another_layer_fills_in(self, tmp_path: Path) -> None:
        index = FacilityIndex.load(_tree(tmp_path), "cables")
        assert index.devices["BPM1"] == "BPM"

    def test_a_layer_not_on_disk_drops_nothing(self, tmp_path: Path) -> None:
        index = FacilityIndex.load(_tree(tmp_path), "new")
        assert "OLD1" in index.devices
        assert index.places == {"SR", "SR/S01", "BR", "BR/S01", "SR/RACK1"}

    def test_groups_carry_their_members(self, tmp_path: Path) -> None:
        index = FacilityIndex.load(_tree(tmp_path), "cables")
        assert index.groups == {"QUADS": {"QF1"}}

    def test_authored_places_hold_only_authored_records(self, tmp_path: Path) -> None:
        index = FacilityIndex.load(_tree(tmp_path), "cables")
        assert index.authored_places == {"SR", "SR/S01"}

    def test_classes_are_the_known_classes(self, tmp_path: Path) -> None:
        tree = _tree(tmp_path, {"classes.yaml": [{"class": "Undulator2", "parent": "Quadrupole"}]})
        index = FacilityIndex.load(tree, "cables")
        assert "Undulator2" in index.classes
        assert "Quadrupole" in index.classes

    def test_a_missing_facility_is_empty(self, tmp_path: Path) -> None:
        index = FacilityIndex.load(tmp_path / "absent", "cables")
        assert index.devices == {} and index.places == set() and index.groups == {}


class TestLoadErrors:
    def test_errors_in_the_replaced_layers_files_are_not_refusals(self, tmp_path: Path) -> None:
        tree = _tree(
            tmp_path,
            {
                "imported/cables/devices.yaml": "- id: [unclosed\n",
                "imported/cables/places.yaml": [{"id": "X", "bogus": 1}],
                "imported/cables/stray.yaml": [],
            },
        )
        index = FacilityIndex.load(tree, "cables")
        assert "PS1" in index.devices

    def test_another_load_error_is_one_refusal_naming_the_first(self, tmp_path: Path) -> None:
        tree = _tree(
            tmp_path,
            {
                "imported/controls/devices.yaml": "- id: [unclosed\n",
                "records/places.yaml": [{"id": "SR", "bogus": 1}],
            },
        )
        with pytest.raises(LayerSourceInvalid) as caught:
            FacilityIndex.load(tree, "cables")
        errors = caught.value.errors
        assert len(errors) == 1
        assert errors[0].format_message() == (
            "facility: source-invalid: place SR — the facility does not load:"
            " unknown key `bogus`; fix: remove `bogus` from records/places.yaml"
        )

    def test_an_error_of_a_layer_sharing_a_name_prefix_is_a_refusal(self, tmp_path: Path) -> None:
        tree = _tree(tmp_path, {"imported/cables2/devices.yaml": "- id: [unclosed\n"})
        with pytest.raises(LayerSourceInvalid):
            FacilityIndex.load(tree, "cables")


class TestResolvePlace:
    @pytest.fixture
    def index(self) -> FacilityIndex:
        return FacilityIndex(
            places={"SR", "SR/S01", "BR", "BR/S01", "SR/S01/RACK1"},
            authored_places={"SR", "SR/S01"},
        )

    def test_a_flat_name_with_one_match_returns_its_id(self, index: FacilityIndex) -> None:
        assert index.resolve_place("RACK1", set()) == "SR/S01/RACK1"
        assert index.resolve_place("SR", set()) == "SR"

    def test_a_flat_name_with_two_matches_returns_the_candidates(
        self, index: FacilityIndex
    ) -> None:
        assert index.resolve_place("S01", set()) == ["BR/S01", "SR/S01"]

    def test_a_flat_name_matches_whole_segments_only(self, index: FacilityIndex) -> None:
        assert index.resolve_place("01", set()) is None
        assert index.resolve_place("ACK1", set()) is None

    def test_a_path_must_match_exactly(self, index: FacilityIndex) -> None:
        assert index.resolve_place("SR/S01", set()) == "SR/S01"
        assert index.resolve_place("S01/RACK1", set()) is None
        assert index.resolve_place("SR/S02", set()) is None

    def test_no_match_is_none(self, index: FacilityIndex) -> None:
        assert index.resolve_place("TUNNEL", set()) is None

    def test_extra_places_resolve(self, index: FacilityIndex) -> None:
        assert index.resolve_place("SR/S01/RACK9", {"SR/S01/RACK9"}) == "SR/S01/RACK9"
        assert index.resolve_place("RACK9", {"SR/S01/RACK9"}) == "SR/S01/RACK9"

    def test_extra_places_join_the_candidates(self, index: FacilityIndex) -> None:
        assert index.resolve_place("SR", {"LTB/SR"}) == ["LTB/SR", "SR"]

    def test_base_authored_places_ignores_another_layers_places(self, index: FacilityIndex) -> None:
        base = index.authored_places
        assert index.resolve_place("S01", set(), base=base) == "SR/S01"
        assert index.resolve_place("BR", set(), base=base) is None
        assert index.resolve_place("BR/S01", set(), base=base) is None
        assert index.resolve_place("RACK1", set(), base=base) is None

    def test_base_still_admits_extra(self, index: FacilityIndex) -> None:
        base = index.authored_places
        assert index.resolve_place("BR", {"BR"}, base=base) == "BR"

    def test_resolution_does_not_change_the_index(self, index: FacilityIndex) -> None:
        before = set(index.places)
        index.resolve_place("X", {"X"})
        assert index.places == before

    def test_the_loaded_index_resolves_with_authored_base(self, tmp_path: Path) -> None:
        index = FacilityIndex.load(_tree(tmp_path), "cables")
        assert index.resolve_place("S01", set()) == ["BR/S01", "SR/S01"]
        assert index.resolve_place("S01", set(), base=index.authored_places) == "SR/S01"
