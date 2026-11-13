"""Relation claims through stage S1 (load), S2 (layer merge, fixes) and the schema stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from osprey.facility.combine import FIXES_HEADER, CombineResult, combine
from osprey.facility.errors import FacilityBuildError
from osprey.facility.provenance import build_provenance
from osprey.facility.sources import load_sources
from osprey.facility.validate import run_stages


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
        path.write_text(text, encoding="utf-8")
    return root


def _load_errors(tmp_path: Path, files: dict[str, Any]) -> list[FacilityBuildError]:
    return load_sources(_write(tmp_path, files)).errors


def _build(tmp_path: Path, files: dict[str, Any]) -> CombineResult:
    loaded = load_sources(_write(tmp_path, files))
    assert [e.format_message() for e in loaded.errors] == []
    return combine(loaded.sources)


def _one(errors: list[FacilityBuildError]) -> FacilityBuildError:
    assert len(errors) == 1, [e.format_message() for e in errors]
    return errors[0]


def _relation(result: CombineResult, rid: str) -> dict[str, Any]:
    (record,) = [r for r in result.document["relations"] if r["id"] == rid]
    return record


def _claim(rid: str, supply: str = "Q1-PS", **extra: Any) -> dict[str, Any]:
    return {
        "id": rid,
        "type": "powered_by",
        "subject": {"device": "Q1"},
        "object": {"device": supply, "port": "OUT1"},
        **extra,
    }


def _fixes(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"schema": FIXES_HEADER, "fixes": list(entries)}


#: Records a claim can name, beside one imported claim per end kind.
_TREE: dict[str, Any] = {
    "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}],
    "records/devices.yaml": [{"id": "Q1-PS", "class": "PowerSupply"}],
    "imported/mml/devices.yaml": [{"id": "Q1", "class": "Quadrupole", "place": "SR/S01"}],
    "imported/mml/groups.yaml": [{"id": "SR/QUAD", "members": ["Q1"]}],
    "imported/mml/relations.yaml": [
        _claim("q1-power"),
        {
            "id": "q1-in-s01",
            "type": "located_in",
            "subject": {"device": "Q1"},
            "object": {"place": "SR/S01"},
        },
        {
            "id": "quads-power",
            "type": "powered_by",
            "subject": {"group": "SR/QUAD"},
            "object": {"device": "Q1-PS"},
        },
    ],
}


class TestEmit:
    def test_authored_and_imported_claims_carry_layer_and_file(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/relations.yaml": [_claim("q1-power-a")],
                "imported/mml/relations.yaml": [_claim("q1-power-b", "Q1-PS2")],
            },
        )
        assert result.errors == []
        assert [r["id"] for r in result.document["relations"]] == ["q1-power-a", "q1-power-b"]
        authored = _relation(result, "q1-power-a")
        assert authored["provenance"] == build_provenance(
            sources=[("authored", "records/relations.yaml", ["object", "subject", "type"])]
        )
        imported = _relation(result, "q1-power-b")
        assert imported["provenance"] == build_provenance(
            sources=[("mml", "imported/mml/relations.yaml", ["object", "subject", "type"])]
        )
        assert imported["object"] == {"device": "Q1-PS2", "port": "OUT1"}

    def test_two_layers_two_supplies_for_one_magnet_both_survive(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "imported/mml/relations.yaml": [_claim("q1-power-mml", "Q1-PS")],
                "imported/csv/relations.yaml": [_claim("q1-power-csv", "Q1-PSB")],
            },
        )
        assert result.errors == []
        supplies = {r["id"]: r["object"]["device"] for r in result.document["relations"]}
        assert supplies == {"q1-power-csv": "Q1-PSB", "q1-power-mml": "Q1-PS"}

    def test_empty_relations_file_emits_no_key(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/devices.yaml": [{"id": "Q1", "class": "Quad"}],
                "records/relations.yaml": [],
            },
        )
        assert result.errors == []
        assert "relations" not in result.document

    def test_no_relations_file_emits_no_key(self, tmp_path: Path) -> None:
        result = _build(tmp_path, {"records/devices.yaml": [{"id": "Q1", "class": "Quad"}]})
        assert result.errors == []
        assert "relations" not in result.document


class TestLayers:
    def test_same_id_equal_content_across_layers_is_one_claim(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "imported/mml/relations.yaml": [_claim("q1-power")],
                "imported/csv/relations.yaml": [_claim("q1-power")],
            },
        )
        assert result.errors == []
        (claim,) = result.document["relations"]
        assert [s["layer"] for s in claim["provenance"]["sources"]] == ["csv", "mml"]

    def test_same_id_unequal_content_across_layers_is_layer_conflict(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "imported/mml/relations.yaml": [_claim("q1-power", "Q1-PS")],
                "imported/csv/relations.yaml": [_claim("q1-power", "Q1-PSB")],
            },
        )
        err = _one(result.errors)
        assert (err.kind, err.record_kind, err.record_id) == (
            "layer-conflict",
            "relation",
            "q1-power",
        )
        assert "object" in err.detail
        assert sorted(err.sources) == [
            "imported/csv/relations.yaml",
            "imported/mml/relations.yaml",
        ]

    def test_repeated_id_in_one_layer_is_layer_duplicate(self, tmp_path: Path) -> None:
        errors = _load_errors(
            tmp_path,
            {"imported/mml/relations.yaml": [_claim("q1-power"), _claim("q1-power", "Q1-PSB")]},
        )
        err = _one(errors)
        assert (err.kind, err.record_kind, err.record_id) == (
            "layer-duplicate",
            "relation",
            "q1-power",
        )


class TestLoadRefusals:
    def test_standing_in_input_is_refused(self, tmp_path: Path) -> None:
        err = _one(
            _load_errors(
                tmp_path, {"records/relations.yaml": [_claim("q1-power", standing="current")]}
            )
        )
        assert (err.kind, err.record_kind, err.record_id) == (
            "source-invalid",
            "relation",
            "q1-power",
        )
        assert list(err.sources) == ["records/relations.yaml"]
        assert "`standing` is computed by the build" in err.detail

    def test_contests_in_input_is_refused(self, tmp_path: Path) -> None:
        err = _one(
            _load_errors(
                tmp_path, {"imported/mml/relations.yaml": [_claim("q1-power", contests=["x"])]}
            )
        )
        assert (err.kind, err.record_kind) == ("source-invalid", "relation")
        assert list(err.sources) == ["imported/mml/relations.yaml"]
        assert "`contests` is computed by the build" in err.detail


class TestSchema:
    def test_unknown_type_is_refused_naming_relations_file(self, tmp_path: Path) -> None:
        root = _write(
            tmp_path / "facility",
            {
                "records/relations.yaml": [
                    {**_claim("q1-power"), "type": "not_a_type"},
                ]
            },
        )
        report = run_stages(root, project_name="p")
        err = _one(report.errors)
        assert (err.kind, err.record_kind, err.record_id) == (
            "source-invalid",
            "path",
            "relations.0.type",
        )
        assert list(err.sources) == ["records/relations.yaml"]
        assert err.detail.startswith("records/relations.yaml: ")

    def test_unknown_type_in_an_imported_layer_names_its_file(self, tmp_path: Path) -> None:
        root = _write(
            tmp_path / "facility",
            {"imported/mml/relations.yaml": [{**_claim("q1-power"), "type": "not_a_type"}]},
        )
        err = _one(run_stages(root, project_name="p").errors)
        assert err.record_id == "relations.0.type"
        assert list(err.sources) == ["imported/mml/relations.yaml"]


class TestFixes:
    def test_set_addresses_a_claim_by_id(self, tmp_path: Path) -> None:
        fix = {
            "op": "set",
            "kind": "relation",
            "id": "q1-power",
            "fields": {"object": {"device": "Q1-PS", "port": "OUT2"}},
            "was": {"object": {"mml": {"device": "Q1-PS", "port": "OUT1"}}},
            "why": "The export names the wrong output.",
        }
        result = _build(tmp_path, {**_TREE, "fixes.yaml": _fixes(fix)})
        assert [e.format_message() for e in result.errors] == []
        claim = _relation(result, "q1-power")
        assert claim["object"] == {"device": "Q1-PS", "port": "OUT2"}
        assert claim["provenance"]["fixes"]

    def test_add_of_kind_relation_adds_a_claim(self, tmp_path: Path) -> None:
        fix = {
            "op": "add",
            "kind": "relation",
            "id": "q1-power-spare",
            "record": {
                "type": "powered_by",
                "subject": {"device": "Q1"},
                "object": {"device": "Q1-PS", "port": "OUT2"},
            },
            "why": "The export omits the spare feed.",
        }
        result = _build(tmp_path, {**_TREE, "fixes.yaml": _fixes(fix)})
        assert [e.format_message() for e in result.errors] == []
        claim = _relation(result, "q1-power-spare")
        assert (claim["type"], claim["subject"], claim["object"]) == (
            "powered_by",
            {"device": "Q1"},
            {"device": "Q1-PS", "port": "OUT2"},
        )
        assert claim["provenance"]["fixes"]

    def test_drop_addresses_a_claim_by_id(self, tmp_path: Path) -> None:
        fix = {"op": "drop", "kind": "relation", "id": "q1-power", "why": "Not wired."}
        result = _build(tmp_path, {**_TREE, "fixes.yaml": _fixes(fix)})
        assert [e.format_message() for e in result.errors] == []
        assert [r["id"] for r in result.document["relations"]] == ["q1-in-s01", "quads-power"]
        assert ("relation", "q1-power") in result.dropped

    def test_dropping_every_claim_emits_no_key(self, tmp_path: Path) -> None:
        tree = {**_TREE, "imported/mml/relations.yaml": [_claim("q1-power")]}
        fix = {"op": "drop", "kind": "relation", "id": "q1-power", "why": "x"}
        result = _build(tmp_path, {**tree, "fixes.yaml": _fixes(fix)})
        assert result.errors == []
        assert "relations" not in result.document

    def test_dropping_a_device_a_claim_names_is_reported(self, tmp_path: Path) -> None:
        fix = {"op": "drop", "kind": "device", "id": "Q1-PS", "why": "x"}
        tree = {
            **_TREE,
            "records/devices.yaml": [],
            "imported/mml/devices.yaml": [
                {"id": "Q1", "class": "Quad", "place": "SR/S01"},
                {"id": "Q1-PS", "class": "PowerSupply"},
            ],
        }
        result = _build(tmp_path, {**tree, "fixes.yaml": _fixes(fix)})
        err = _one(result.errors)
        assert (err.kind, err.record_kind, err.record_id) == ("fix-referenced", "device", "Q1-PS")
        assert err.detail == (
            "`drop` leaves 2 referrer(s): relation q1-power (object); relation quads-power (object)"
        )

    def test_dropping_a_place_a_claim_names_is_reported(self, tmp_path: Path) -> None:
        places = [{"id": "SR/S02"}]
        tree = {
            **_TREE,
            "imported/mml/places.yaml": places,
            "imported/mml/relations.yaml": [
                {
                    "id": "ps-in-s02",
                    "type": "located_in",
                    "subject": {"device": "Q1"},
                    "object": {"place": "SR/S02"},
                }
            ],
        }
        fix = {"op": "drop", "kind": "place", "id": "SR/S02", "why": "x"}
        result = _build(tmp_path, {**tree, "fixes.yaml": _fixes(fix)})
        err = _one(result.errors)
        assert (err.kind, err.record_kind, err.record_id) == ("fix-referenced", "place", "SR/S02")
        assert err.detail == "`drop` leaves 1 referrer(s): relation ps-in-s02 (object)"

    def test_dropping_a_group_a_claim_names_is_reported(self, tmp_path: Path) -> None:
        fix = {"op": "drop", "kind": "group", "id": "SR/QUAD", "why": "x"}
        result = _build(tmp_path, {**_TREE, "fixes.yaml": _fixes(fix)})
        err = _one(result.errors)
        assert (err.kind, err.record_kind, err.record_id) == ("fix-referenced", "group", "SR/QUAD")
        assert err.detail == "`drop` leaves 1 referrer(s): relation quads-power (subject)"

    def test_dropping_the_claim_with_its_end_builds(self, tmp_path: Path) -> None:
        fixes = _fixes(
            {"op": "drop", "kind": "group", "id": "SR/QUAD", "why": "x"},
            {"op": "drop", "kind": "relation", "id": "quads-power", "why": "x"},
        )
        result = _build(tmp_path, {**_TREE, "fixes.yaml": fixes})
        assert [e.format_message() for e in result.errors] == []


class TestGroupMembership:
    def test_two_layers_with_different_members_stop_with_layer_conflict(
        self, tmp_path: Path
    ) -> None:
        # p3-r2: membership is not a claim kind; one group stated twice with
        # different members is a layer conflict, never a union.
        tree = {
            **_TREE,
            "imported/mml/devices.yaml": [
                {"id": "Q1", "class": "Quad", "place": "SR/S01"},
                {"id": "Q2", "class": "Quad", "place": "SR/S01"},
            ],
            "imported/csv/groups.yaml": [{"id": "SR/QUAD", "members": ["Q1", "Q2"]}],
        }
        result = _build(tmp_path, tree)
        err = _one(result.errors)
        assert (err.kind, err.record_kind, err.record_id) == ("layer-conflict", "group", "SR/QUAD")
        assert "members" in err.format_message()


class TestThroughTheStages:
    def test_a_group_subject_on_powered_by_stops_at_records(self, tmp_path: Path) -> None:
        # B-8: powered_by allows a device or place subject, so `quads-power`
        # resolves at S4 and stops at S5 with relation-invalid.
        report = run_stages(_write(tmp_path, _TREE), project_name="p")
        assert report.failed == "records", [e.format_message() for e in report.errors]
        err = _one(report.errors)
        assert (err.kind, err.record_kind, err.record_id) == (
            "relation-invalid",
            "relation",
            "quads-power",
        )
        assert err.sources == ("imported/mml/relations.yaml",)
        assert err.format_message() == (
            "facility: relation-invalid: relation quads-power — `subject` names a group; "
            "powered_by allows device or place there; fix: name a device or place as the "
            "subject of powered_by"
        )
