"""The ``--members`` table of a relations import: rows into group records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.facility.layers.relations.facility import FacilityIndex
from osprey.facility.layers.relations.members import group_ids, groups
from osprey.facility.layers.relations.tables import MEMBERS, Refusals, Row, read_table

HEADER = "group,member,member_kind,document,version,locator\n"


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
            "records/places.yaml": [{"id": "SR"}],
            "records/devices.yaml": [
                {"id": "QF1", "class": "Quadrupole"},
                {"id": "QF2", "class": "Quadrupole"},
            ],
            "records/groups.yaml": [{"id": "QUADS", "members": ["QF1"]}],
            "imported/controls/devices.yaml": [{"id": "PS1", "class": "PowerSupply"}],
            "imported/controls/groups.yaml": [{"id": "SUPPLIES", "members": ["PS1"]}],
            "imported/cables/groups.yaml": [{"id": "OLDGROUP", "members": ["QF2"]}],
        },
    )
    return FacilityIndex.load(tree, "cables")


def _rows(tmp_path: Path, body: str) -> list[Row]:
    path = tmp_path / "members.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return read_table(path, MEMBERS)


def _messages(refusals: Refusals) -> list[str]:
    return [error.format_message() for error in refusals.errors()]


class TestGroupIds:
    def test_names_only_the_groups_no_other_source_has(self, tmp_path: Path) -> None:
        rows = _rows(
            tmp_path,
            "RACKED,QF1,,,,\nQUADS,QF1,,,,\nSUPPLIES,PS1,,,,\nOLDGROUP,QF2,,,,\nRACKED,QF2,,,,\n",
        )
        # OLDGROUP is the replaced layer's own previous output, so it is created anew.
        assert group_ids(rows, _index(tmp_path)) == {"RACKED", "OLDGROUP"}

    def test_matches_what_groups_writes(self, tmp_path: Path) -> None:
        rows = _rows(tmp_path, "RACKED,QF1,,,,\nQUADS,QF1,,,,\n")
        index = _index(tmp_path)
        written = groups(rows, index, {}, Refusals())
        assert {record["id"] for record in written} == group_ids(rows, index)


class TestNewGroup:
    def test_a_new_group_is_written_with_its_members_sorted(self, tmp_path: Path) -> None:
        rows = _rows(tmp_path, "RACKED,QF2,device,,,\nRACKED,QF1,,,,\nRACKED,QF1,,,,\n")
        refusals = Refusals()
        assert groups(rows, _index(tmp_path), {}, refusals) == [
            {"id": "RACKED", "members": ["QF1", "QF2"]}
        ]
        assert not refusals

    def test_a_group_record_carries_no_citation(self, tmp_path: Path) -> None:
        rows = _rows(tmp_path, "RACKED,QF1,,cable-schedule,B,p. 4\n")
        (record,) = groups(rows, _index(tmp_path), {}, Refusals())
        assert set(record) == {"id", "members"}

    def test_records_are_sorted_by_id_whatever_the_row_order(self, tmp_path: Path) -> None:
        rows = _rows(tmp_path, "ZETA,QF1,,,,\nALPHA,QF2,,,,\n")
        written = groups(rows, _index(tmp_path), {}, Refusals())
        assert [record["id"] for record in written] == ["ALPHA", "ZETA"]

    def test_a_stub_member_is_accepted_and_takes_the_row_citation(self, tmp_path: Path) -> None:
        stub: dict[str, Any] = {"id": "RACK1-PDU", "class": "Unidentified"}
        rows = _rows(
            tmp_path,
            "RACKED,RACK1-PDU,,schedule,B,p. 9\n"
            "RACKED,RACK1-PDU,,schedule,A,p. 2\n"
            "RACKED,RACK1-PDU,,schedule,A,p. 2\n",
        )
        written = groups(rows, _index(tmp_path), {"RACK1-PDU": stub}, Refusals())
        assert written == [{"id": "RACKED", "members": ["RACK1-PDU"]}]
        assert stub["citations"] == [
            {"document": "schedule", "version": "A", "locator": "p. 2"},
            {"document": "schedule", "version": "B", "locator": "p. 9"},
        ]

    def test_a_stub_keeps_its_own_citations_and_gains_no_duplicate(self, tmp_path: Path) -> None:
        own = {"document": "schedule", "version": "A", "locator": "p. 2"}
        stub: dict[str, Any] = {"id": "PDU", "class": "Unidentified", "citations": [dict(own)]}
        rows = _rows(tmp_path, "RACKED,PDU,,schedule,A,p. 2\n")
        groups(rows, _index(tmp_path), {"PDU": stub}, Refusals())
        assert stub["citations"] == [own]

    def test_a_facility_member_gains_no_citation(self, tmp_path: Path) -> None:
        stub: dict[str, Any] = {"id": "PDU"}
        rows = _rows(tmp_path, "RACKED,QF1,,schedule,A,p. 2\nRACKED,PDU,,,,\n")
        groups(rows, _index(tmp_path), {"PDU": stub}, Refusals())
        assert "citations" not in stub


class TestGroupAnotherSourceHas:
    def test_a_held_member_of_another_layers_group_is_accepted_and_unwritten(
        self, tmp_path: Path
    ) -> None:
        rows = _rows(tmp_path, "SUPPLIES,PS1,,schedule,A,p. 1\nQUADS,QF1,device,,,\n")
        refusals = Refusals()
        assert groups(rows, _index(tmp_path), {}, refusals) == []
        assert not refusals

    def test_an_unheld_member_is_added_in_this_layer(self, tmp_path: Path) -> None:
        rows = _rows(tmp_path, "SUPPLIES,QF1,,,,\nSUPPLIES,QF1,,,,\nRACKED,QF2,,,,\n")
        refusals = Refusals()
        written = groups(rows, _index(tmp_path), {}, refusals)
        assert written == [
            {"id": "RACKED", "members": ["QF2"]},
            {"id": "SUPPLIES", "members": ["QF1"]},
        ]
        assert not refusals

    def test_the_layer_states_every_member_its_rows_name_once_one_is_new(
        self, tmp_path: Path
    ) -> None:
        rows = _rows(tmp_path, "SUPPLIES,PS1,,,,\nSUPPLIES,QF1,,,,\n")
        assert groups(rows, _index(tmp_path), {}, Refusals()) == [
            {"id": "SUPPLIES", "members": ["PS1", "QF1"]}
        ]

    def test_an_added_stub_member_takes_the_row_citation(self, tmp_path: Path) -> None:
        stub: dict[str, Any] = {"id": "PDU"}
        rows = _rows(tmp_path, "SUPPLIES,PDU,,schedule,A,p. 1\n")
        written = groups(rows, _index(tmp_path), {"PDU": stub}, Refusals())
        assert written == [{"id": "SUPPLIES", "members": ["PDU"]}]
        assert stub["citations"] == [{"document": "schedule", "version": "A", "locator": "p. 1"}]

    def test_an_unknown_member_of_another_sources_group_is_refused(self, tmp_path: Path) -> None:
        rows = _rows(tmp_path, "SUPPLIES,NOPE,,,,\n")
        refusals = Refusals()
        assert groups(rows, _index(tmp_path), {}, refusals) == []
        (error,) = refusals.errors()
        assert error.record_id == "NOPE"
        assert error.detail == "unknown device (members.csv row 2)"

    def test_a_held_stub_row_adds_no_citation(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        index.groups["SUPPLIES"].add("PDU")
        stub: dict[str, Any] = {"id": "PDU"}
        rows = _rows(tmp_path, "SUPPLIES,PDU,,schedule,A,p. 1\n")
        assert groups(rows, index, {"PDU": stub}, Refusals()) == []
        assert "citations" not in stub


class TestRefusedRows:
    def test_an_unknown_member_is_refused(self, tmp_path: Path) -> None:
        rows = _rows(tmp_path, "RACKED,NOPE,,,,\n")
        refusals = Refusals()
        assert groups(rows, _index(tmp_path), {}, refusals) == []
        (error,) = refusals.errors()
        assert error.record_id == "NOPE"
        assert error.detail == "unknown device (members.csv row 2)"

    def test_a_member_kind_other_than_device_is_refused(self, tmp_path: Path) -> None:
        rows = _rows(tmp_path, "RACKED,SR,place,,,\nRACKED,QF1,,,,\n")
        refusals = Refusals()
        assert groups(rows, _index(tmp_path), {}, refusals) == [
            {"id": "RACKED", "members": ["QF1"]}
        ]
        (error,) = refusals.errors()
        assert error.record_id == "place"
        assert error.record_kind == "member_kind"
        assert "members.csv row 2" in error.detail

    def test_the_table_name_is_the_one_passed(self, tmp_path: Path) -> None:
        rows = _rows(tmp_path, "RACKED,NOPE,,,,\n")
        refusals = Refusals()
        groups(rows, _index(tmp_path), {}, refusals, table="racks.csv")
        (error,) = refusals.errors()
        assert error.sources == ("racks.csv",)
