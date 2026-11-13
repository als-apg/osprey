"""The relations table of a relations import: rows into claims and stub devices (p5-r2, p5-r5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.facility.layers.relations.claims import claims
from osprey.facility.layers.relations.facility import FacilityIndex
from osprey.facility.layers.relations.tables import RELATIONS, Refusals, Row, read_table

HEADER = (
    "type,subject,object,subject_kind,object_kind,subject_port,object_port,object_unit,"
    "attrs,basis,rule,document,version,locator,subject_class,object_class\n"
)


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
            "records/places.yaml": [
                {"id": "SR"},
                {"id": "SR/S01"},
                {"id": "BR"},
                {"id": "BR/S01"},
                {"id": "SR/S02"},
            ],
            "records/devices.yaml": [
                {"id": "QF1", "class": "Quadrupole"},
                {"id": "QF2", "class": "Quadrupole"},
                {"id": "PS1", "class": "PowerSupply"},
            ],
            "records/groups.yaml": [{"id": "QUADS", "members": ["QF1", "QF2"]}],
            "imported/cables/devices.yaml": [{"id": "OLDSTUB", "class": "Unidentified"}],
        },
    )
    return FacilityIndex.load(tree, "cables")


def _row(**cells: str) -> str:
    columns = HEADER.strip().split(",")
    return ",".join(cells.get(column, "") for column in columns) + "\n"


def _rows(tmp_path: Path, *lines: str) -> list[Row]:
    path = tmp_path / "relations.csv"
    path.write_text(HEADER + "".join(lines), encoding="utf-8")
    return read_table(path, RELATIONS)


def _run(
    tmp_path: Path,
    *lines: str,
    stub_unknown: bool = False,
    places: set[str] | None = None,
    groups: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    refusals = Refusals()
    out, stubs = claims(
        _rows(tmp_path, *lines),
        _index(tmp_path),
        refusals,
        layer="cables",
        places=places or set(),
        groups=groups or set(),
        stub_unknown=stub_unknown,
    )
    return out, stubs, [error.format_message() for error in refusals.errors()]


class TestClaimShape:
    def test_row_becomes_a_claim_in_record_shape(self, tmp_path: Path) -> None:
        out, stubs, messages = _run(
            tmp_path,
            _row(
                type="cabled_to",
                subject="QF1",
                object="PS1",
                subject_port="A",
                object_port="3",
                attrs="cable=C-17; routed_from=tray 4",
                document="CS-1",
                version="B",
                locator="p. 4",
            ),
        )
        assert messages == []
        assert stubs == {}
        assert len(out) == 1
        claim = out[0]
        assert claim["id"].startswith("cables/cabled_to/")
        assert claim["type"] == "cabled_to"
        assert claim["subject"] == {"device": "QF1", "port": "A"}
        assert claim["object"] == {"device": "PS1", "port": "3"}
        assert claim["attributes"] == {"cable": "C-17", "routed_from": "tray 4"}
        assert claim["source"] == {"document": "CS-1", "version": "B", "locator": "p. 4"}
        assert "basis" not in claim

    def test_inferred_keeps_basis_and_rule(self, tmp_path: Path) -> None:
        out, _, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QF1", object="PS1", basis="inferred", rule="R1"),
        )
        assert messages == []
        assert out[0]["basis"] == "inferred"
        assert out[0]["rule"] == "R1"

    def test_place_end_with_unit_and_created_place(self, tmp_path: Path) -> None:
        out, _, messages = _run(
            tmp_path,
            _row(
                type="located_in",
                subject="QF1",
                object="S02",
                object_kind="place",
                object_unit="22",
            ),
            _row(type="located_in", subject="QF2", object="RACK9", object_kind="place"),
            places={"SR/S01/RACK9"},
        )
        assert messages == []
        objects = sorted((c["object"]["place"], c["object"].get("unit")) for c in out)
        assert objects == [("SR/S01/RACK9", None), ("SR/S02", 22)]

    def test_group_end_resolves_facility_and_created_groups(self, tmp_path: Path) -> None:
        out, _, messages = _run(
            tmp_path,
            _row(type="bounded_by", subject="QUADS", subject_kind="group", object="PS1"),
            _row(type="bounded_by", subject="RACKED", subject_kind="group", object="PS1"),
            groups={"RACKED"},
        )
        assert messages == []
        assert sorted(c["subject"]["group"] for c in out) == ["QUADS", "RACKED"]

    def test_output_sorted_by_id(self, tmp_path: Path) -> None:
        out, _, _ = _run(
            tmp_path,
            _row(type="powered_by", subject="QF1", object="PS1"),
            _row(type="powered_by", subject="QF2", object="PS1"),
            _row(type="cabled_to", subject="QF1", object="QF2", attrs="cable=C1"),
        )
        ids = [c["id"] for c in out]
        assert ids == sorted(ids) and len(ids) == 3


class TestStubs:
    """p5-r2: stub only with a class column or the explicit flag."""

    def test_no_flag_and_no_class_refused(self, tmp_path: Path) -> None:
        out, stubs, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QX9", object="PS1"),
        )
        assert out == [] and stubs == {}
        assert len(messages) == 1
        assert "QX9" in messages[0]
        assert "unknown device (relations.csv row 2)" in messages[0]
        assert "--stub-unknown" in messages[0]

    def test_flag_stubs_unidentified(self, tmp_path: Path) -> None:
        out, stubs, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QX9", object="PS1", document="CS-1"),
            stub_unknown=True,
        )
        assert messages == []
        assert stubs == {
            "QX9": {"id": "QX9", "class": "Unidentified", "citations": [{"document": "CS-1"}]}
        }
        assert out[0]["subject"] == {"device": "QX9"}

    def test_replaced_layers_own_stub_is_not_in_the_facility(self, tmp_path: Path) -> None:
        _, stubs, messages = _run(
            tmp_path, _row(type="powered_by", subject="OLDSTUB", object="PS1"), stub_unknown=True
        )
        assert messages == []
        assert stubs["OLDSTUB"]["class"] == "Unidentified"

    def test_class_column_stubs_without_flag(self, tmp_path: Path) -> None:
        out, stubs, messages = _run(
            tmp_path,
            _row(
                type="powered_by",
                subject="QX9",
                object="PS9",
                subject_class="Quadrupole",
                object_class="PowerSupply",
            ),
            _row(type="powered_by", subject="QX9", object="PS1"),
        )
        assert messages == []
        assert stubs == {
            "PS9": {"id": "PS9", "class": "PowerSupply"},
            "QX9": {"id": "QX9", "class": "Quadrupole"},
        }
        assert len(out) == 2

    def test_class_column_wins_over_flag_default(self, tmp_path: Path) -> None:
        _, stubs, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QX9", object="PS1"),
            _row(
                type="cabled_to",
                subject="QX9",
                object="QF1",
                attrs="cable=C1",
                subject_class="Quadrupole",
            ),
            stub_unknown=True,
        )
        assert messages == []
        assert stubs["QX9"]["class"] == "Quadrupole"

    def test_unknown_class_refused(self, tmp_path: Path) -> None:
        out, stubs, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QX9", object="PS1", subject_class="Quadropole"),
            _row(type="powered_by", subject="QX8", object="PS1", subject_class="Quadropole"),
        )
        assert out == [] and stubs == {}
        assert len(messages) == 1
        assert "Quadropole" in messages[0]
        assert "unknown class (relations.csv rows 2, 3)" in messages[0]

    def test_two_classes_refused(self, tmp_path: Path) -> None:
        out, stubs, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QX9", object="PS1", subject_class="Quadrupole"),
            _row(
                type="cabled_to",
                subject="QF1",
                object="QX9",
                attrs="cable=C1",
                object_class="PowerSupply",
            ),
            _row(type="cabled_to", subject="QF2", object="QX9", attrs="cable=C2"),
            stub_unknown=True,
        )
        assert out == [] or all("QX9" not in str(c) for c in out)
        assert "QX9" not in stubs
        assert len(messages) == 1
        assert "stubbed with two classes: PowerSupply, Quadrupole" in messages[0]
        assert "relations.csv rows 2, 3" in messages[0]

    def test_differing_class_refused(self, tmp_path: Path) -> None:
        out, stubs, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QF1", object="PS1", subject_class="Sextupole"),
        )
        assert out == [] and stubs == {}
        assert len(messages) == 1
        assert "`subject_class` Sextupole differs from its class Quadrupole" in messages[0]

    def test_equal_class_accepted_and_writes_nothing(self, tmp_path: Path) -> None:
        out, stubs, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QF1", object="PS1", subject_class="Quadrupole"),
        )
        assert messages == [] and stubs == {}
        assert out[0]["subject"] == {"device": "QF1"}

    def test_place_and_group_ends_never_stubbed(self, tmp_path: Path) -> None:
        out, stubs, messages = _run(
            tmp_path,
            _row(type="located_in", subject="QF1", object="NOWHERE", object_kind="place"),
            _row(type="bounded_by", subject="NOGROUP", subject_kind="group", object="PS1"),
            stub_unknown=True,
        )
        assert out == [] and stubs == {}
        joined = "\n".join(messages)
        assert "unknown place" in joined and "unknown group" in joined

    def test_class_on_place_end_refused(self, tmp_path: Path) -> None:
        _, _, messages = _run(
            tmp_path,
            _row(
                type="located_in",
                subject="QF1",
                object="SR",
                object_kind="place",
                object_class="Quadrupole",
            ),
        )
        assert len(messages) == 1
        assert "a class on an end that names a place" in messages[0]


class TestFolded:
    def test_unknown_type_folded(self, tmp_path: Path) -> None:
        _, _, messages = _run(
            tmp_path,
            _row(type="wired_to", subject="QF1", object="PS1"),
            _row(type="wired_to", subject="QF2", object="PS1"),
        )
        assert len(messages) == 1
        assert "unknown relation type (relations.csv rows 2, 3)" in messages[0]
        assert "wired_to" in messages[0]

    def test_unknown_place_folded(self, tmp_path: Path) -> None:
        _, _, messages = _run(
            tmp_path,
            _row(type="located_in", subject="QF1", object="S99", object_kind="place"),
            _row(type="located_in", subject="QF2", object="S99", object_kind="place"),
        )
        assert len(messages) == 1
        assert "unknown place (relations.csv rows 2, 3)" in messages[0]

    def test_unknown_group_folded(self, tmp_path: Path) -> None:
        _, _, messages = _run(
            tmp_path,
            _row(type="bounded_by", subject="NOGROUP", subject_kind="group", object="PS1"),
            _row(
                type="bounded_by",
                subject="NOGROUP",
                subject_kind="group",
                object="PS1",
                document="X",
            ),
        )
        assert len(messages) == 1
        assert "unknown group (relations.csv rows 2, 3)" in messages[0]

    def test_ambiguous_flat_place_refused(self, tmp_path: Path) -> None:
        _, _, messages = _run(
            tmp_path,
            _row(type="located_in", subject="QF1", object="S01", object_kind="place"),
        )
        assert len(messages) == 1
        assert "matches several places: BR/S01, SR/S01" in messages[0]
        assert "full path" in messages[0]


class TestRowChecks:
    def test_bad_kind_refused(self, tmp_path: Path) -> None:
        _, _, messages = _run(
            tmp_path, _row(type="powered_by", subject="QF1", subject_kind="rack", object="PS1")
        )
        assert len(messages) == 1 and "not an end kind" in messages[0]

    def test_bad_basis_refused(self, tmp_path: Path) -> None:
        _, _, messages = _run(
            tmp_path, _row(type="powered_by", subject="QF1", object="PS1", basis="guessed")
        )
        assert len(messages) == 1 and "not a basis" in messages[0]

    def test_inferred_without_rule_refused(self, tmp_path: Path) -> None:
        out, _, messages = _run(
            tmp_path, _row(type="powered_by", subject="QF1", object="PS1", basis="inferred")
        )
        assert out == []
        assert len(messages) == 1 and "`basis` is inferred and `rule` is missing" in messages[0]

    def test_malformed_attrs_refused(self, tmp_path: Path) -> None:
        out, _, messages = _run(
            tmp_path, _row(type="cabled_to", subject="QF1", object="PS1", attrs="cable")
        )
        assert out == []
        assert len(messages) == 1 and "no `=`" in messages[0]

    def test_unit_not_integer_refused(self, tmp_path: Path) -> None:
        _, _, messages = _run(
            tmp_path,
            _row(
                type="located_in",
                subject="QF1",
                object="SR",
                object_kind="place",
                object_unit="top",
            ),
        )
        assert len(messages) == 1 and "not an integer" in messages[0]

    def test_kind_the_type_does_not_allow_refused(self, tmp_path: Path) -> None:
        _, _, messages = _run(tmp_path, _row(type="located_in", subject="QF1", object="PS1"))
        assert len(messages) == 1
        assert "located_in allows place there" in messages[0]

    def test_required_attribute_refused(self, tmp_path: Path) -> None:
        _, _, messages = _run(tmp_path, _row(type="supplied_by", subject="QF1", object="PS1"))
        assert len(messages) == 1 and "attributes.medium" in messages[0]


class TestDuplicates:
    """p5-r5: equal content under one id is one claim; unequal is refused."""

    def test_exact_duplicate_kept_once(self, tmp_path: Path) -> None:
        line = _row(type="powered_by", subject="QF1", object="PS1", document="CS-1", locator="p1")
        out, _, messages = _run(tmp_path, line, line)
        assert messages == []
        assert len(out) == 1

    def test_reversed_symmetric_row_kept_once(self, tmp_path: Path) -> None:
        out, _, messages = _run(
            tmp_path,
            _row(type="cabled_to", subject="QF1", object="PS1", attrs="cable=C1"),
            _row(type="cabled_to", subject="PS1", object="QF1", attrs="cable=C1"),
        )
        assert messages == []
        assert len(out) == 1
        assert out[0]["subject"] == {"device": "QF1"}

    def test_duplicate_id_with_different_content_refused(self, tmp_path: Path) -> None:
        out, _, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QF1", object="PS1", document="CS-1", locator="p1"),
            _row(type="powered_by", subject="QF1", object="PS1", document="CS-1", locator="p2"),
            _row(type="powered_by", subject="QF2", object="PS1"),
        )
        assert [c["subject"] for c in out] == [{"device": "QF2"}]
        assert len(messages) == 1
        assert (
            "rows with one claim id state different content (relations.csv rows 2, 3)"
            in (messages[0])
        )

    def test_two_documents_are_two_claims(self, tmp_path: Path) -> None:
        out, _, messages = _run(
            tmp_path,
            _row(type="powered_by", subject="QF1", object="PS1", document="CS-1"),
            _row(type="powered_by", subject="QF1", object="PS1", document="CS-2"),
        )
        assert messages == []
        assert len(out) == 2

    def test_result_independent_of_row_order(self, tmp_path: Path) -> None:
        a = _row(
            type="powered_by", subject="QX9", object="PS1", document="B", subject_class="Quadrupole"
        )
        b = _row(type="cabled_to", subject="QX9", object="QF1", attrs="cable=C1", document="A")
        first = _run(tmp_path, a, b)
        second = _run(tmp_path, b, a)
        assert first == second
        assert first[1]["QX9"]["citations"] == [{"document": "A"}, {"document": "B"}]
