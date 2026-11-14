"""The ``--places`` table: rows into the place records a relations layer creates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.facility.layers.relations.facility import FacilityIndex
from osprey.facility.layers.relations.places import places
from osprey.facility.layers.relations.tables import PLACES, Refusals, read_table


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, document in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return root


def _index(tmp_path: Path, extra: dict[str, Any] | None = None) -> FacilityIndex:
    files: dict[str, Any] = {
        "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}, {"id": "BR"}],
    }
    files.update(extra or {})
    return FacilityIndex.load(_write(tmp_path / "facility", files), "cables")


def _run(
    tmp_path: Path, text: str, index: FacilityIndex | None = None
) -> tuple[list[dict], list[str]]:
    path = tmp_path / "places.csv"
    path.write_text(text, encoding="utf-8")
    refusals = Refusals()
    rows = read_table(path, PLACES, refusals)
    assert not refusals, [e.format_message() for e in refusals.errors()]
    out = places(rows, index or _index(tmp_path), refusals, path.name)
    return out, [e.format_message() for e in refusals.errors()]


def _ok(tmp_path: Path, text: str, index: FacilityIndex | None = None) -> list[dict]:
    out, refused = _run(tmp_path, text, index)
    assert refused == []
    return out


def _cite(document: str, **rest: str) -> dict[str, str]:
    return {"document": document, **rest}


class TestPlacing:
    def test_within_an_authored_place(self, tmp_path: Path) -> None:
        assert _ok(tmp_path, "place,within\nRACK1,S01\n") == [{"id": "SR/S01/RACK1"}]

    def test_within_a_path(self, tmp_path: Path) -> None:
        assert _ok(tmp_path, "place,within\nRACK1,SR/S01\n") == [{"id": "SR/S01/RACK1"}]

    def test_a_path_cell_is_the_id(self, tmp_path: Path) -> None:
        assert _ok(tmp_path, "place\nSR/S01/RACK1\n") == [{"id": "SR/S01/RACK1"}]

    def test_a_path_with_its_parent_beside_it(self, tmp_path: Path) -> None:
        assert _ok(tmp_path, "place,within\nSR/S01/RACK1,S01\n") == [{"id": "SR/S01/RACK1"}]

    def test_within_a_place_of_the_same_table(self, tmp_path: Path) -> None:
        out = _ok(tmp_path, "place,within\nSHELF2,RACK1\nRACK1,S01\n")
        assert out == [{"id": "SR/S01/RACK1"}, {"id": "SR/S01/RACK1/SHELF2"}]

    def test_the_citation_is_kept_and_basis_rule_are_not(self, tmp_path: Path) -> None:
        out = _ok(
            tmp_path,
            "place,within,basis,rule,document,locator\nRACK1,S01,inferred,r7,DWG-1,p. 3\n",
        )
        assert out == [{"id": "SR/S01/RACK1", "citations": [_cite("DWG-1", locator="p. 3")]}]

    def test_output_is_sorted_and_independent_of_row_order(self, tmp_path: Path) -> None:
        rows = ["B,SR", "A,BR", "C,A", "B,SR/S01", "A"]
        forward = _ok(tmp_path, "place,within\n" + "\n".join(rows) + "\n")
        backward = _ok(tmp_path, "place,within\n" + "\n".join(reversed(rows)) + "\n")
        assert forward == backward
        assert [r["id"] for r in forward] == sorted(r["id"] for r in forward)


class TestChainRule:
    def test_parents_on_one_chain_give_the_deepest(self, tmp_path: Path) -> None:
        out = _ok(
            tmp_path,
            "place,within,document\nRACK1,SR,D-A\nRACK1,SR/S01,D-B\nRACK1,,D-C\n",
        )
        assert out == [
            {"id": "SR/S01/RACK1", "citations": [_cite("D-A"), _cite("D-B"), _cite("D-C")]}
        ]

    def test_parents_off_one_chain_give_one_record_each(self, tmp_path: Path) -> None:
        out = _ok(tmp_path, "place,within,document\nRACK1,SR,D-A\nRACK1,BR,D-B\nRACK1,,\n")
        assert out == [
            {"id": "BR/RACK1", "citations": [_cite("D-B")]},
            {"id": "SR/RACK1", "citations": [_cite("D-A")]},
        ]

    def test_off_chain_a_cited_unplaced_row_is_its_own_root(self, tmp_path: Path) -> None:
        out = _ok(tmp_path, "place,within,document\nRACK1,SR,D-A\nRACK1,BR,\nRACK1,,D-C\n")
        assert [r["id"] for r in out] == ["BR/RACK1", "RACK1", "SR/RACK1"]
        assert out[1] == {"id": "RACK1", "citations": [_cite("D-C")]}

    def test_equal_citations_are_one(self, tmp_path: Path) -> None:
        out = _ok(tmp_path, "place,within,document\nRACK1,SR,D-A\nRACK1,SR,D-A\n")
        assert out == [{"id": "SR/RACK1", "citations": [_cite("D-A")]}]


class TestRoots:
    def test_unplaced_and_uncited_is_top_level_without_citations(self, tmp_path: Path) -> None:
        assert _ok(tmp_path, "place,within\nHALL\n") == [{"id": "HALL"}]

    def test_unplaced_and_cited_is_top_level_cited(self, tmp_path: Path) -> None:
        out = _ok(tmp_path, "place,document\nHALL,SITE-PLAN\n")
        assert out == [{"id": "HALL", "citations": [_cite("SITE-PLAN")]}]


class TestOtherSources:
    def test_an_authored_place_is_not_recreated(self, tmp_path: Path) -> None:
        assert _ok(tmp_path, "place,within,document\nS01,SR,D-A\nSR,,D-B\n") == []

    def test_another_imported_layers_place_is_written_all_the_same(self, tmp_path: Path) -> None:
        text = "place,within,document\nRACK1,S01,D-A\nSHELF,RACK1,D-B\n"
        alone = _ok(tmp_path, text, _index(tmp_path / "a"))
        other = {
            "imported/controls/places.yaml": [
                {"id": "SR/S01/RACK1"},
                {"id": "BR/RACK1"},
                {"id": "BR/S01"},
            ]
        }
        beside = _ok(tmp_path, text, _index(tmp_path / "b", other))
        assert (
            alone
            == beside
            == [
                {"id": "SR/S01/RACK1", "citations": [_cite("D-A")]},
                {"id": "SR/S01/RACK1/SHELF", "citations": [_cite("D-B")]},
            ]
        )

    def test_within_never_resolves_to_another_imported_layers_place(self, tmp_path: Path) -> None:
        index = _index(tmp_path, {"imported/controls/places.yaml": [{"id": "LINAC"}]})
        out, refused = _run(tmp_path, "place,within\nGUN,LINAC\n", index)
        assert out == []
        assert len(refused) == 1 and "unknown place" in refused[0] and "LINAC" in refused[0]


class TestRefusals:
    def test_unknown_within(self, tmp_path: Path) -> None:
        out, refused = _run(tmp_path, "place,within\nRACK1,NOWHERE\nRACK2,NOWHERE\n")
        assert out == []
        assert len(refused) == 1
        assert "NOWHERE" in refused[0] and "unknown place (places.csv rows 2, 3)" in refused[0]

    def test_ambiguous_within_lists_the_candidates(self, tmp_path: Path) -> None:
        index = _index(
            tmp_path,
            {
                "records/places.yaml": [
                    {"id": "SR"},
                    {"id": "SR/S01"},
                    {"id": "BR"},
                    {"id": "BR/S01"},
                ]
            },
        )
        out, refused = _run(tmp_path, "place,within\nRACK1,S01\n", index)
        assert out == []
        assert len(refused) == 1 and "matches several places: BR/S01, SR/S01" in refused[0]

    def test_a_path_whose_within_names_another_parent(self, tmp_path: Path) -> None:
        out, refused = _run(tmp_path, "place,within\nSR/S01/RACK1,BR\n")
        assert out == []
        assert len(refused) == 1 and "is not the path's parent" in refused[0]

    def test_a_path_with_an_unknown_parent(self, tmp_path: Path) -> None:
        out, refused = _run(tmp_path, "place\nXX/RACK1\n")
        assert out == []
        assert len(refused) == 1 and "unknown place" in refused[0]

    def test_a_bad_basis(self, tmp_path: Path) -> None:
        out, refused = _run(tmp_path, "place,within,basis\nRACK1,SR,guessed\n")
        assert out == []
        assert len(refused) == 1 and "not a basis" in refused[0]

    def test_inferred_without_rule(self, tmp_path: Path) -> None:
        out, refused = _run(tmp_path, "place,within,basis\nRACK1,SR,inferred\n")
        assert out == []
        assert len(refused) == 1 and "`basis` is inferred and `rule` is missing" in refused[0]

    def test_a_refused_row_places_nothing_below_it(self, tmp_path: Path) -> None:
        out, refused = _run(tmp_path, "place,within,basis\nRACK1,SR,guessed\nSHELF,RACK1,\n")
        assert out == []
        assert len(refused) == 2
