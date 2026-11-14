"""Relation claims and citations through stage S4 (references)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.combine import combine
from osprey.facility.errors import FacilityBuildError
from osprey.facility.sources import load_sources
from osprey.facility.validate import StageReport, check_references, run_stages


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
        path.write_text(text, encoding="utf-8")
    return root


def _run(tmp_path: Path, files: dict[str, Any]) -> StageReport:
    return run_stages(_write(tmp_path, files), project_name="p")


def _one(report: StageReport) -> FacilityBuildError:
    assert report.failed == "references", [e.format_message() for e in report.errors]
    assert len(report.errors) == 1, [e.format_message() for e in report.errors]
    return report.errors[0]


def _claim(rid: str, supply: str = "Q1-PS", **extra: Any) -> dict[str, Any]:
    return {
        "id": rid,
        "type": "powered_by",
        "subject": {"device": "Q1"},
        "object": {"device": supply, "port": "OUT1"},
        **extra,
    }


def _tree(**extra: Any) -> dict[str, Any]:
    """A facility every claim below can resolve against, plus ``extra`` files."""
    files: dict[str, Any] = {
        "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}],
        "records/devices.yaml": [
            {"id": "Q1-PS", "class": "PowerSupply"},
            {"id": "Q1", "class": "Quadrupole", "place": "SR/S01"},
        ],
    }
    files.update(extra)
    return files


_DOCS = [{"id": "spec-1", "title": "Magnet list"}]


class TestResolves:
    def test_resolving_claims_and_citations_pass(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path,
            _tree(
                **{
                    "documents.yaml": _DOCS,
                    "records/relations.yaml": [
                        _claim("q1-power", source={"document": "spec-1"}),
                        {
                            "id": "q1-in-s01",
                            "type": "located_in",
                            "subject": {"device": "Q1"},
                            "object": {"place": "SR/S01"},
                        },
                    ],
                }
            ),
        )
        assert report.failed is None, [e.format_message() for e in report.errors]

    def test_no_documents_file_checks_no_citation(self, tmp_path: Path) -> None:
        files = _tree(
            **{"records/relations.yaml": [_claim("q1-power", source={"document": "nowhere"})]}
        )
        files["records/places.yaml"] = [
            {"id": "SR", "citations": [{"document": "nowhere"}]},
            {"id": "SR/S01"},
        ]
        files["records/devices.yaml"][1]["scheme_names"] = [
            {"name": "QF1", "scheme": "lattice", "source": {"document": "nowhere"}}
        ]
        report = _run(tmp_path, files)
        assert report.failed is None, [e.format_message() for e in report.errors]


class TestClaimEnds:
    def test_subject_end_names_no_device(self, tmp_path: Path) -> None:
        claim = _claim("q9-power")
        claim["subject"] = {"device": "Q9"}
        error = _one(_run(tmp_path, _tree(**{"records/relations.yaml": [claim]})))
        assert (error.kind, error.record_kind, error.record_id) == (
            "reference-missing",
            "relation",
            "q9-power",
        )
        assert error.sources == ("records/relations.yaml",)
        assert error.detail == (
            "records/relations.yaml `subject.device` names device Q9, which does not exist"
        )
        assert error.remedy == "add device Q9 or correct `subject.device`"

    def test_object_end_names_no_place(self, tmp_path: Path) -> None:
        claim = {
            "id": "q1-in-s09",
            "type": "located_in",
            "subject": {"device": "Q1"},
            "object": {"place": "SR/S09"},
        }
        error = _one(_run(tmp_path, _tree(**{"imported/mml/relations.yaml": [claim]})))
        assert (error.record_kind, error.record_id) == ("relation", "q1-in-s09")
        assert error.sources == ("imported/mml/relations.yaml",)
        assert error.detail == (
            "imported/mml/relations.yaml `object.place` names place SR/S09, which does not exist"
        )

    def test_end_names_no_group(self, tmp_path: Path) -> None:
        claim = {
            "id": "quads-power",
            "type": "powered_by",
            "subject": {"group": "SR/QUAD"},
            "object": {"device": "Q1-PS"},
        }
        error = _one(_run(tmp_path, _tree(**{"records/relations.yaml": [claim]})))
        assert error.detail == (
            "records/relations.yaml `subject.group` names group SR/QUAD, which does not exist"
        )

    def test_end_naming_a_dropped_device_reads_dropped_by_fix(self, tmp_path: Path) -> None:
        # Combine refuses a drop a claim end still names (fix-referenced), so the
        # dropped record is planted in the combine result S4 reads.
        loaded = load_sources(
            _write(tmp_path, _tree(**{"records/relations.yaml": [_claim("q1-power")]}))
        )
        assert loaded.errors == []
        combined = combine(loaded.sources)
        combined.document["devices"] = [
            d for d in combined.document["devices"] if d["id"] != "Q1-PS"
        ]
        combined.dropped[("device", "Q1-PS")] = {
            "op": "drop",
            "kind": "device",
            "id": "Q1-PS",
            "why": "decommissioned",
        }
        (error,) = check_references(loaded.sources, combined)
        assert (error.record_kind, error.record_id) == ("relation", "q1-power")
        assert error.sources == ("records/relations.yaml", "fixes.yaml")
        assert error.detail == (
            "records/relations.yaml `object.device` names device Q1-PS, "
            "dropped by fix drop device Q1-PS: decommissioned"
        )
        assert error.remedy == "remove the entry or drop the fix"

    def test_end_stating_two_kinds_resolves_each(self, tmp_path: Path) -> None:
        claim = _claim("q1-power")
        claim["object"] = {"device": "Q1-PS", "place": "SR/S09"}
        error = _one(_run(tmp_path, _tree(**{"records/relations.yaml": [claim]})))
        assert error.detail == (
            "records/relations.yaml `object.place` names place SR/S09, which does not exist"
        )

    def test_dangling_end_with_bad_shape_prints_only_reference_missing(
        self, tmp_path: Path
    ) -> None:
        claim = _claim("q1-power")
        claim["subject"] = {"device": "Q9", "group": "SR/QUAD"}
        files = _tree(
            **{
                "records/relations.yaml": [claim],
                "records/groups.yaml": [{"id": "SR/QUAD", "members": ["Q1"]}],
            }
        )
        report = _run(tmp_path, files)
        assert report.failed == "references"
        assert {e.kind for e in report.errors} == {"reference-missing"}
        assert [e.detail for e in report.errors] == [
            "records/relations.yaml `subject.device` names device Q9, which does not exist"
        ]


class TestCitations:
    def test_claim_source_names_no_document(self, tmp_path: Path) -> None:
        files = _tree(
            **{
                "documents.yaml": _DOCS,
                "records/relations.yaml": [_claim("q1-power", source={"document": "spec-9"})],
            }
        )
        error = _one(_run(tmp_path, files))
        assert (error.kind, error.record_kind, error.record_id) == (
            "reference-missing",
            "relation",
            "q1-power",
        )
        assert error.sources == ("records/relations.yaml",)
        assert error.detail == (
            "records/relations.yaml `source.document` names document spec-9, which does not exist"
        )
        assert error.remedy == "add document spec-9 or correct `source.document`"

    def test_device_citation_names_no_document(self, tmp_path: Path) -> None:
        files = _tree(**{"documents.yaml": _DOCS})
        files["records/devices.yaml"][1]["citations"] = [
            {"document": "spec-1"},
            {"document": "dwg-9", "locator": "sheet 3"},
        ]
        error = _one(_run(tmp_path, files))
        assert (error.record_kind, error.record_id) == ("device", "Q1")
        assert error.sources == ("records/devices.yaml",)
        assert error.detail == (
            "records/devices.yaml `citations.document` names document dwg-9, which does not exist"
        )

    def test_place_citation_names_no_document(self, tmp_path: Path) -> None:
        files = _tree(**{"documents.yaml": _DOCS})
        files["records/places.yaml"] = [
            {"id": "SR", "citations": [{"document": "dwg-9"}]},
            {"id": "SR/S01"},
        ]
        error = _one(_run(tmp_path, files))
        assert (error.record_kind, error.record_id) == ("place", "SR")
        assert error.sources == ("records/places.yaml",)
        assert error.detail == (
            "records/places.yaml `citations.document` names document dwg-9, which does not exist"
        )

    def test_device_scheme_name_source_names_no_document(self, tmp_path: Path) -> None:
        files = _tree(**{"documents.yaml": _DOCS})
        files["records/devices.yaml"][1]["scheme_names"] = [
            {"name": "QF1", "scheme": "lattice", "source": {"document": "spec-1"}},
            {"name": "SR01C:QF1", "scheme": "controls", "source": {"document": "dwg-9"}},
        ]
        error = _one(_run(tmp_path, files))
        assert (error.record_kind, error.record_id) == ("device", "Q1")
        assert error.detail == (
            "records/devices.yaml `scheme_names.source.document` names document dwg-9, "
            "which does not exist"
        )

    def test_place_scheme_name_source_names_no_document(self, tmp_path: Path) -> None:
        files = _tree(**{"documents.yaml": _DOCS})
        files["records/places.yaml"] = [
            {
                "id": "SR",
                "scheme_names": [
                    {"name": "storage ring", "scheme": "lattice", "source": {"document": "dwg-9"}}
                ],
            },
            {"id": "SR/S01"},
        ]
        error = _one(_run(tmp_path, files))
        assert (error.record_kind, error.record_id) == ("place", "SR")
        assert error.detail == (
            "records/places.yaml `scheme_names.source.document` names document dwg-9, "
            "which does not exist"
        )


# --- S5: claim shape -------------------------------------------------------------------


def _records(report: StageReport) -> list[str]:
    assert report.failed == "records", [e.format_message() for e in report.errors]
    return [e.format_message() for e in report.errors]


#: One claim per ``relation-invalid`` condition, with the detail and remedy it prints.
_SHAPES: dict[str, tuple[dict[str, Any], str]] = {
    "end_no_kind": (
        _claim("a", subject={"port": "X"}),
        "`subject` names no device, place or group; fix: name one of `device`, `place`, "
        "`group` in `subject`",
    ),
    "end_two_kinds": (
        _claim("a", subject={"device": "Q1", "place": "SR"}),
        "`subject` names more than one of device, place, group; fix: keep one of `device`, "
        "`place`, `group` in `subject`",
    ),
    "unit_off_place": (
        _claim("a", object={"device": "Q1-PS", "unit": 2}),
        "`object` gives `unit` on an end that names no place; fix: drop `unit` from `object`, "
        "or name the place it counts in",
    ),
    "kind_not_allowed": (
        _claim("a", type="located_in", object={"device": "Q1-PS"}),
        "`object` names a device; located_in allows place there; fix: name a place as the "
        "object of located_in",
    ),
    "inferred_no_rule": (
        _claim("a", basis="inferred"),
        "`basis` is inferred and `rule` is missing; fix: name the rule that derived the claim "
        "in `rule`, or drop `basis: inferred`",
    ),
    "missing_medium": (
        _claim("a", type="supplied_by"),
        "supplied_by gives no `attributes.medium`; fix: state the medium in `attributes.medium`",
    ),
    "same_end": (
        _claim("a", object={"device": "Q1"}),
        "`subject` and `object` name the same end; fix: name two different ends, or tell them "
        "apart by `port`",
    ),
}


class TestShape:
    @pytest.mark.parametrize("case", sorted(_SHAPES))
    def test_each_condition_stops_with_one_line(self, tmp_path: Path, case: str) -> None:
        claim, text = _SHAPES[case]
        lines = _records(_run(tmp_path, _tree(**{"records/relations.yaml": [claim]})))
        assert lines == [f"facility: relation-invalid: relation a — {text}"]

    def test_sound_claims_pass_every_stage(self, tmp_path: Path) -> None:
        claims = [
            _claim("q1-power"),
            _claim("q1-inferred", basis="inferred", rule="naming"),
            {
                "id": "q1-in-sr",
                "type": "located_in",
                "subject": {"device": "Q1"},
                "object": {"place": "SR", "unit": 3},
            },
        ]
        report = _run(tmp_path, _tree(**{"records/relations.yaml": claims}))
        assert report.failed is None, [e.format_message() for e in report.errors]

    def test_a_claim_breaking_two_rules_prints_both(self, tmp_path: Path) -> None:
        claim = _claim("a", basis="inferred", object={"device": "Q1"})
        lines = _records(_run(tmp_path, _tree(**{"records/relations.yaml": [claim]})))
        assert [line.split(" — ")[1].split(";")[0] for line in lines] == [
            "`basis` is inferred and `rule` is missing",
            "`subject` and `object` name the same end",
        ]

    def test_the_line_names_the_file_stating_the_claim(self, tmp_path: Path) -> None:
        tree = _tree(**{"imported/mml/relations.yaml": [_claim("a", basis="inferred")]})
        report = _run(tmp_path, tree)
        assert report.failed == "records", [e.format_message() for e in report.errors]
        (error,) = report.errors
        assert (error.kind, error.record_kind, error.record_id) == (
            "relation-invalid",
            "relation",
            "a",
        )
        assert error.sources == ("imported/mml/relations.yaml",)

    def test_a_fixed_slot_names_fixes_yaml(self, tmp_path: Path) -> None:
        fix = {
            "op": "set",
            "kind": "relation",
            "id": "a",
            "fields": {"basis": "inferred"},
            "why": "x",
        }
        tree = _tree(
            **{
                "imported/mml/relations.yaml": [_claim("a")],
                "fixes.yaml": {"schema": "osprey.facility.fixes/1", "fixes": [fix]},
            }
        )
        report = _run(tmp_path, tree)
        assert report.failed == "records", [e.format_message() for e in report.errors]
        (error,) = report.errors
        assert error.sources == ("fixes.yaml",)
