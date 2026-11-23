"""The ``--names`` table of a relations import: rows into ``scheme_names`` (p5-r4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.layers.relations.facility import FacilityIndex
from osprey.facility.layers.relations.names import names
from osprey.facility.layers.relations.tables import NAMES, Refusals, Row, read_table
from osprey.facility.sources import keyed_clashes

HEADER = "thing,name,scheme,basis,rule,document,version,locator\n"


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, document in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return root


def _index(tmp_path: Path) -> FacilityIndex:
    tree = _write(
        tmp_path / "facility",
        {
            "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}],
            "records/devices.yaml": [{"id": "QF1", "class": "Quadrupole"}],
            "records/groups.yaml": [{"id": "QUADS", "members": ["QF1"]}],
            "imported/mml/devices.yaml": [{"id": "BPM1", "class": "BPM"}],
            "imported/mml/places.yaml": [{"id": "SR/S02"}],
            "imported/cables/devices.yaml": [{"id": "OLD", "class": "Device"}],
        },
    )
    return FacilityIndex.load(tree, "cables")


def _rows(tmp_path: Path, body: str) -> list[Row]:
    path = tmp_path / "names.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return read_table(path, NAMES)


def _messages(refusals: Refusals) -> list[str]:
    return [error.format_message() for error in refusals.errors()]


def _layer() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stubs = {"PS1": {"id": "PS1", "class": "PowerSupply"}}
    places = [{"id": "SR/S01/RACK1"}, {"id": "SR/S02"}, {"id": "TUNNEL"}, {"id": "TUNNEL/S01"}]
    return stubs, places


def _run(tmp_path: Path, body: str, groups: set[str] | None = None):
    stubs, places = _layer()
    refusals = Refusals()
    names(
        _rows(tmp_path, body),
        _index(tmp_path),
        refusals,
        stubs=stubs,
        places=places,
        groups=groups or set(),
    )
    return stubs, {p["id"]: p for p in places}, refusals


def _run_named(tmp_path: Path, body: str):
    """The records the layer states only to name a thing another source has."""
    stubs, places = _layer()
    refusals = Refusals()
    devices, named_places = names(
        _rows(tmp_path, body), _index(tmp_path), refusals, stubs=stubs, places=places
    )
    return devices, named_places, {p["id"]: p for p in places}, refusals


class TestNamesLand:
    def test_a_name_lands_on_a_stub_device(self, tmp_path: Path) -> None:
        stubs, _, refusals = _run(tmp_path, "PS1,PS-Q-0301,cable-schedule,,,cs,B,p. 4\n")
        assert not refusals
        assert stubs["PS1"]["scheme_names"] == [
            {
                "name": "PS-Q-0301",
                "scheme": "cable-schedule",
                "source": {"document": "cs", "version": "B", "locator": "p. 4"},
            }
        ]

    @pytest.mark.parametrize(
        ("row", "place_id", "name"),
        [
            pytest.param(
                "RACK1,Rack 1,drawings,,,,,\n",
                "SR/S01/RACK1",
                "Rack 1",
                id="a_name_lands_on_a_created_place",
            ),
            pytest.param(
                "TUNNEL,T,drawings,,,,,\n", "TUNNEL", "T", id="a_path_names_a_created_place"
            ),
        ],
    )
    def test_a_created_place_is_named(
        self, tmp_path: Path, row: str, place_id: str, name: str
    ) -> None:
        _, places, refusals = _run(tmp_path, row)
        assert not refusals
        assert places[place_id]["scheme_names"] == [{"name": name, "scheme": "drawings"}]

    def test_stated_basis_is_not_written_and_inferred_keeps_its_rule(self, tmp_path: Path) -> None:
        stubs, _, refusals = _run(
            tmp_path,
            "PS1,A,s1,stated,,,,\nPS1,B,s2,inferred,label-match,,,\n",
        )
        assert not refusals
        assert stubs["PS1"]["scheme_names"] == [
            {"name": "A", "scheme": "s1"},
            {"name": "B", "scheme": "s2", "basis": "inferred", "rule": "label-match"},
        ]

    def test_names_are_in_the_builds_keyed_order(self, tmp_path: Path) -> None:
        stubs, _, _ = _run(tmp_path, "PS1,Z,s,,,,,\nPS1,A,t,,,,,\nPS1,A,s,,,,,\n")
        assert [(e["name"], e["scheme"]) for e in stubs["PS1"]["scheme_names"]] == [
            ("A", "s"),
            ("A", "t"),
            ("Z", "s"),
        ]

    def test_a_record_without_rows_gets_no_slot(self, tmp_path: Path) -> None:
        _, places, _ = _run(tmp_path, "PS1,A,s,,,,,\n")
        assert "scheme_names" not in places["TUNNEL"]


class TestKeyedSet:
    def test_equal_entries_are_one_entry(self, tmp_path: Path) -> None:
        stubs, _, refusals = _run(tmp_path, "PS1,A,s,,,d,1,\nPS1,A,s,stated,,d,1,\n")
        assert not refusals
        assert stubs["PS1"]["scheme_names"] == [
            {"name": "A", "scheme": "s", "source": {"document": "d", "version": "1"}}
        ]

    def test_one_key_with_two_contents_is_refused_on_every_row(self, tmp_path: Path) -> None:
        stubs, _, refusals = _run(tmp_path, "PS1,A,s,,,d,1,\nPS1,A,s,,,d,2,\nPS1,B,s,,,,,\n")
        (message,) = _messages(refusals)
        assert "PS1 s A" in message
        assert "different content" in message
        assert "2" in message and "3" in message
        assert stubs["PS1"]["scheme_names"] == [{"name": "B", "scheme": "s"}]
        assert keyed_clashes("scheme_names", stubs["PS1"]["scheme_names"]) == []

    def test_one_name_in_two_schemes_is_two_entries(self, tmp_path: Path) -> None:
        stubs, _, refusals = _run(tmp_path, "PS1,A,s,,,,,\nPS1,A,t,,,,,\n")
        assert not refusals
        assert len(stubs["PS1"]["scheme_names"]) == 2


class TestAnotherSourcesRecord:
    def test_an_authored_device_gets_a_record_carrying_only_the_name(self, tmp_path: Path) -> None:
        devices, places, _, refusals = _run_named(tmp_path, "QF1,Q-0301,cables,,,cs,B,\n")
        assert not refusals
        assert devices == [
            {
                "id": "QF1",
                "scheme_names": [
                    {
                        "name": "Q-0301",
                        "scheme": "cables",
                        "source": {"document": "cs", "version": "B"},
                    }
                ],
            }
        ]
        assert places == []

    def test_another_layers_device_is_named_too(self, tmp_path: Path) -> None:
        devices, _, _, refusals = _run_named(tmp_path, "BPM1,B,s,,,,,\nBPM1,A,s,,,,,\n")
        assert not refusals
        assert devices == [
            {
                "id": "BPM1",
                "scheme_names": [{"name": "A", "scheme": "s"}, {"name": "B", "scheme": "s"}],
            }
        ]

    def test_an_authored_place_gets_a_record_carrying_only_the_name(self, tmp_path: Path) -> None:
        devices, places, created, refusals = _run_named(tmp_path, "SR/S01,X,s,,,,,\n")
        assert not refusals
        assert devices == []
        assert places == [{"id": "SR/S01", "scheme_names": [{"name": "X", "scheme": "s"}]}]
        assert "SR/S01" not in created

    def test_a_created_place_another_layer_has_is_named_on_the_created_record(
        self, tmp_path: Path
    ) -> None:
        devices, places, created, refusals = _run_named(tmp_path, "SR/S02,X,s,,,,,\n")
        assert not refusals
        assert (devices, places) == ([], [])
        assert created["SR/S02"]["scheme_names"] == [{"name": "X", "scheme": "s"}]

    def test_records_come_sorted_by_id(self, tmp_path: Path) -> None:
        devices, _, _, _ = _run_named(tmp_path, "QF1,Q,s,,,,,\nBPM1,B,s,,,,,\n")
        assert [d["id"] for d in devices] == ["BPM1", "QF1"]

    def test_a_refused_row_states_no_record(self, tmp_path: Path) -> None:
        devices, _, _, refusals = _run_named(tmp_path, "QF1,Q,s,guessed,,,,\n")
        assert refusals
        assert devices == []


class TestRefused:
    @pytest.mark.parametrize(
        ("row", "fragment"),
        [
            pytest.param(
                "OLD,O,s,,,,,\n", "unknown thing", id="a_device_of_the_replaced_layer_is_unknown"
            ),
            pytest.param(
                "S01,X,s,,,,,\n",
                "matches several places: SR/S01, TUNNEL/S01",
                id="a_flat_name_matching_several_places",
            ),
        ],
    )
    def test_a_row_is_refused(self, tmp_path: Path, row: str, fragment: str) -> None:
        _, _, refusals = _run(tmp_path, row)
        (message,) = _messages(refusals)
        assert fragment in message

    def test_a_group_is_refused(self, tmp_path: Path) -> None:
        _, _, refusals = _run(tmp_path, "QUADS,Q,s,,,,,\nRACKED,R,s,,,,,\n", groups={"RACKED"})
        messages = _messages(refusals)
        assert len(messages) == 2
        assert all("a group has no `scheme_names`" in m for m in messages)

    def test_an_unknown_thing_is_refused(self, tmp_path: Path) -> None:
        stubs, places, refusals = _run(tmp_path, "NOBODY,N,s,,,,,\n")
        (message,) = _messages(refusals)
        assert "NOBODY" in message and "unknown thing" in message
        assert "scheme_names" not in stubs["PS1"]

    @pytest.mark.parametrize(
        ("row", "fragment"),
        [
            pytest.param("PS1,A,s,guessed,,,,\n", "not a basis", id="a_bad_basis"),
            pytest.param("PS1,A,s,inferred,,,,\n", "`rule` is missing", id="inferred_without_rule"),
        ],
    )
    def test_a_refused_row_names_nothing(self, tmp_path: Path, row: str, fragment: str) -> None:
        stubs, _, refusals = _run(tmp_path, row)
        (message,) = _messages(refusals)
        assert fragment in message
        assert "scheme_names" not in stubs["PS1"]
