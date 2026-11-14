"""The relations layer's tables: CSV read into rows, refusals folded per value."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.facility.layers import LayerSourceInvalid
from osprey.facility.layers.relations.tables import (
    CITATION_COLUMNS,
    FOLD_LIMIT,
    MEMBERS,
    NAMES,
    PLACES,
    RELATIONS,
    Refusals,
    Row,
    citation,
    parse_attrs,
    read_table,
)


def _csv(tmp_path: Path, text: str, name: str = "relations.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _refused(path: Path, spec=RELATIONS) -> list[str]:
    with pytest.raises(LayerSourceInvalid) as stop:
        read_table(path, spec)
    return [error.format_message() for error in stop.value.errors]


class TestSpecs:
    def test_the_relations_table_holds_the_design_columns(self) -> None:
        assert RELATIONS.required == ("type", "subject", "object")
        for column in (
            "subject_kind",
            "object_kind",
            "subject_port",
            "object_port",
            "object_unit",
            "attrs",
            "basis",
            "rule",
            "subject_class",
            "object_class",
        ):
            assert column in RELATIONS.optional

    def test_the_other_tables_hold_their_columns(self) -> None:
        assert NAMES.required == ("thing", "name", "scheme")
        assert {"basis", "rule"} <= set(NAMES.optional)
        assert PLACES.required == ("place",)
        assert {"within", "basis", "rule"} <= set(PLACES.optional)
        assert MEMBERS.required == ("group", "member")
        assert "member_kind" in MEMBERS.optional

    @pytest.mark.parametrize("spec", [RELATIONS, NAMES, PLACES, MEMBERS])
    def test_every_table_takes_the_citation_columns_with_imported(self, spec) -> None:
        assert CITATION_COLUMNS == ("document", "version", "state", "locator", "imported")
        assert set(CITATION_COLUMNS) <= set(spec.optional)


class TestRead:
    def test_rows_carry_line_numbers_and_only_filled_stripped_cells(self, tmp_path: Path) -> None:
        path = _csv(
            tmp_path,
            "type, subject ,object,subject_port\n"
            "\n"
            " part_of , SR/QF1 , SR/S01 ,\n"
            ",,,\n"
            "powered_by,SR/QF1,SR/PS1, J1 \n",
        )
        assert read_table(path, RELATIONS) == [
            Row(3, {"type": "part_of", "subject": "SR/QF1", "object": "SR/S01"}),
            Row(
                5,
                {
                    "type": "powered_by",
                    "subject": "SR/QF1",
                    "object": "SR/PS1",
                    "subject_port": "J1",
                },
            ),
        ]

    def test_a_byte_order_mark_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "places.csv"
        path.write_bytes("﻿place,within\nS01,SR\n".encode())
        assert read_table(path, PLACES) == [Row(2, {"place": "S01", "within": "SR"})]

    def test_a_row_may_hold_fewer_cells_than_the_header(self, tmp_path: Path) -> None:
        path = _csv(tmp_path, "place,within,document\nSR\n", "places.csv")
        assert read_table(path, PLACES) == [Row(2, {"place": "SR"})]

    def test_a_row_unpacks_as_line_and_cells(self, tmp_path: Path) -> None:
        path = _csv(tmp_path, "group,member\nG,D\n", "members.csv")
        [(line, cells)] = read_table(path, MEMBERS)
        assert (line, cells) == (2, {"group": "G", "member": "D"})


class TestRefusedTables:
    def test_an_unknown_column_is_refused(self, tmp_path: Path) -> None:
        lines = _refused(_csv(tmp_path, "type,subject,object,colour\nx,a,b,red\n"))
        assert len(lines) == 1
        assert lines[0].startswith(
            "facility: source-invalid: path relations.csv — unknown column `colour`; "
            "fix: remove the column; the columns are type, subject, object,"
        )

    def test_a_missing_required_column_is_refused(self, tmp_path: Path) -> None:
        assert _refused(_csv(tmp_path, "type,subject\nx,a\n")) == [
            "facility: source-invalid: path relations.csv — has no `object` column; "
            "fix: add the column `object`"
        ]

    def test_a_repeated_column_is_refused(self, tmp_path: Path) -> None:
        assert _refused(_csv(tmp_path, "type,subject,object,subject\nx,a,b,c\n")) == [
            "facility: source-invalid: path relations.csv — names column `subject` twice; "
            "fix: keep one"
        ]

    def test_a_row_with_extra_cells_is_refused(self, tmp_path: Path) -> None:
        path = _csv(tmp_path, "type,subject,object\npart_of,a,b,c\npart_of,a,b\n")
        assert _refused(path) == [
            "facility: source-invalid: path relations.csv — holds more cells than the header "
            "names (3) (relations.csv row 2); fix: fill one cell per header column"
        ]

    def test_non_utf8_input_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "relations.csv"
        path.write_bytes("type,subject,object\npart_of,Pr\xfcf,b\n".encode("latin-1"))
        [line] = _refused(path)
        assert line.startswith("facility: source-invalid: path relations.csv — is not UTF-8: ")
        assert line.endswith("; fix: save it as UTF-8")

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        assert _refused(_csv(tmp_path, "\n\n")) == [
            "facility: source-invalid: path relations.csv — holds no header row; "
            "fix: add a header row naming `type`, `subject`, `object`"
        ]

    def test_an_empty_required_cell_refuses_the_row_only(self, tmp_path: Path) -> None:
        refusals = Refusals()
        path = _csv(tmp_path, "type,subject,object\npart_of,,b\npart_of,a,b\n")
        assert read_table(path, RELATIONS, refusals) == [
            Row(3, {"type": "part_of", "subject": "a", "object": "b"})
        ]
        assert [e.format_message() for e in refusals.errors()] == [
            "facility: source-invalid: column subject — empty required cell "
            "(relations.csv row 2); fix: fill `subject` or remove the row"
        ]

    def test_a_citation_with_no_document_is_refused(self, tmp_path: Path) -> None:
        path = _csv(tmp_path, "place,locator\nSR,p. 4\n", "places.csv")
        assert _refused(path, PLACES) == [
            "facility: source-invalid: column document — a citation with no document "
            "(`locator` filled) (places.csv row 2); "
            "fix: fill `document`, or empty the citation columns"
        ]

    def test_an_imported_that_is_no_iso_date_is_refused(self, tmp_path: Path) -> None:
        path = _csv(tmp_path, "place,document,imported\nSR,D1,13/11/2026\n", "places.csv")
        assert _refused(path, PLACES) == [
            "facility: source-invalid: imported 13/11/2026 — not an ISO date "
            "(places.csv row 2); fix: write `imported` as YYYY-MM-DD"
        ]

    def test_refusals_given_are_appended_to_and_nothing_raises(self, tmp_path: Path) -> None:
        refusals = Refusals()
        assert read_table(_csv(tmp_path, "type,colour\n"), RELATIONS, refusals) == []
        assert len(refusals) == 3
        assert all(e.kind == "source-invalid" for e in refusals.errors())


class TestFold:
    def _unknown_type(self, refusals: Refusals, value: str, line: int) -> None:
        refusals.refuse(
            "relations.csv",
            line,
            record_kind="type",
            value=value,
            problem="unknown relation type",
            remedy="use a relation type of the core",
        )

    def test_one_error_per_value_with_three_rows_and_a_count(self) -> None:
        refusals = Refusals()
        for line in (9, 2, 5, 14, 7):
            self._unknown_type(refusals, "poweredby", line)
        self._unknown_type(refusals, "cable", 3)
        assert [e.format_message() for e in refusals.errors()] == [
            "facility: source-invalid: type poweredby — unknown relation type "
            "(relations.csv rows 2, 5, 7 and 2 more); fix: use a relation type of the core",
            "facility: source-invalid: type cable — unknown relation type "
            "(relations.csv row 3); fix: use a relation type of the core",
        ]

    def test_three_rows_take_no_count(self) -> None:
        refusals = Refusals()
        for line in (4, 2, 3, 3):
            self._unknown_type(refusals, "x", line)
        [error] = refusals.errors()
        assert error.detail == "unknown relation type (relations.csv rows 2, 3, 4)"

    def test_one_value_under_two_problems_is_two_errors(self) -> None:
        refusals = Refusals()
        self._unknown_type(refusals, "x", 2)
        refusals.refuse(
            "relations.csv", 3, record_kind="type", value="x", problem="other", remedy="r"
        )
        assert len(refusals.errors()) == 2

    def test_past_the_limit_one_closing_line_counts_the_rest(self) -> None:
        refusals = Refusals()
        for number in range(FOLD_LIMIT + 5):
            self._unknown_type(refusals, f"t{number}", number + 2)
        errors = refusals.errors()
        assert len(errors) == FOLD_LIMIT + 1
        assert errors[FOLD_LIMIT - 1].record_id == f"t{FOLD_LIMIT - 1}"
        assert errors[-1].format_message() == (
            "facility: source-invalid: import relations — 5 more refused values; "
            "fix: clear the refusals above and import again to see the rest"
        )

    def test_at_the_limit_there_is_no_closing_line(self) -> None:
        refusals = Refusals()
        for number in range(FOLD_LIMIT):
            self._unknown_type(refusals, f"t{number}", 2)
        assert len(refusals.errors()) == FOLD_LIMIT

    def test_no_refusal_is_falsy(self) -> None:
        refusals = Refusals()
        assert not refusals
        assert refusals.errors() == []


class TestCitation:
    def test_absent_fields_are_omitted(self) -> None:
        assert citation({"document": "D1", "locator": "p. 4", "place": "SR"}) == {
            "document": "D1",
            "locator": "p. 4",
        }

    def test_every_field_with_imported_kept_as_written(self) -> None:
        cells = {
            "document": "D1",
            "version": "B",
            "state": "released",
            "locator": "sheet 3",
            "imported": "2026-11-13",
        }
        assert citation(cells) == cells

    def test_no_document_is_no_citation(self) -> None:
        assert citation({"place": "SR"}) is None


class TestAttrs:
    def test_pairs_are_read_in_order_and_stripped(self) -> None:
        refusals = Refusals()
        assert parse_attrs(" cable = C-12 ; medium=power;", refusals, "relations.csv", 2) == {
            "cable": "C-12",
            "medium": "power",
        }
        assert not refusals

    def test_a_value_may_hold_an_equals_sign(self) -> None:
        assert parse_attrs("note=a=b", Refusals(), "relations.csv", 2) == {"note": "a=b"}

    def test_an_empty_cell_is_no_attributes(self) -> None:
        assert parse_attrs("", Refusals(), "relations.csv", 2) == {}

    @pytest.mark.parametrize(
        ("text", "value", "problem"),
        [
            ("cable", "cable", "an attribute pair with no `=`"),
            ("=C-12", "=C-12", "an attribute with no key"),
            ("cable=a;cable=b", "cable=b", "attribute `cable` given twice"),
        ],
    )
    def test_a_malformed_pair_is_a_refusal(self, text: str, value: str, problem: str) -> None:
        refusals = Refusals()
        assert parse_attrs(text, refusals, "relations.csv", 4) is None
        [error] = refusals.errors()
        assert (error.kind, error.record_kind, error.record_id) == (
            "source-invalid",
            "attrs",
            value,
        )
        assert error.detail == f"{problem} (relations.csv row 4)"
