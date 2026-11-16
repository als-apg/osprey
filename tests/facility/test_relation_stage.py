"""The relation stage: standing and contests in the facility file, and its two warnings."""

from __future__ import annotations

import io
import itertools
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import LATER_STAGES, build_facility_with_warnings
from osprey.facility.relations import fill_relation_slots
from osprey.facility.render import facility_bytes
from osprey.facility.validate import StageReport, run_stages, validate


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
        path.write_text(text, encoding="utf-8")
    return root


def _run(root: Path, files: dict[str, Any]) -> StageReport:
    report = run_stages(_write(root, files), project_name="p", later=LATER_STAGES)
    assert report.failed is None, [e.format_message() for e in report.errors]
    return report


def _claims(report: StageReport) -> dict[str, dict[str, Any]]:
    document = report.validated.document
    assert document is not None
    return {claim["id"]: claim for claim in document.get("relations", [])}


def _lines(report: StageReport) -> list[str]:
    return [warning.line for warning in report.validated.warnings]


def _power(rid: str, supply: str, *, subject: str = "Q1", **extra: Any) -> dict[str, Any]:
    return {
        "id": rid,
        "type": "powered_by",
        "subject": {"device": subject},
        "object": {"device": supply},
        **extra,
    }


def _cited(document: str) -> dict[str, Any]:
    return {"source": {"document": document}}


_DOCS = [
    {"id": "spec-1", "title": "Magnet list", "kind": "spec"},
    {"id": "dwg-1", "title": "Cabling drawing", "kind": "drawing"},
]

_POLICY = {
    "schema": "osprey.facility.policy/1",
    "relations": {"precedence": {"default": ["spec", "drawing"]}},
}


def _tree(**extra: Any) -> dict[str, Any]:
    files: dict[str, Any] = {
        "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}],
        "records/devices.yaml": [
            {"id": "PS1", "class": "PowerSupply"},
            {"id": "PS2", "class": "PowerSupply"},
            {"id": "PS3", "class": "PowerSupply"},
            {"id": "Q1", "class": "Quadrupole", "place": "SR/S01"},
            {"id": "Q2", "class": "Quadrupole", "place": "SR/S01"},
        ],
        "documents.yaml": _DOCS,
    }
    files.update(extra)
    return files


class TestRegistration:
    def test_relations_runs_first_after_s5(self) -> None:
        assert LATER_STAGES[0] == ("relations", fill_relation_slots)
        assert LATER_STAGES[-1][0] == "knowledge"


class TestSlots:
    def test_undisputed_claim_is_accepted_without_contests(self, tmp_path: Path) -> None:
        report = _run(tmp_path, _tree(**{"records/relations.yaml": [_power("a", "PS1")]}))
        claim = _claims(report)["a"]
        assert claim["standing"] == "accepted"
        assert "contests" not in claim
        assert "standing" in claim["provenance"]["defaults"]
        assert "contests" not in claim["provenance"]["defaults"]
        assert _lines(report) == []

    def test_slots_land_in_facility_json(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path,
            _tree(**{"records/relations.yaml": [_power("a", "PS1"), _power("b", "PS2")]}),
        )
        document = report.validated.document
        assert document is not None
        rendered = json.loads(facility_bytes(document))
        claims = {claim["id"]: claim for claim in rendered["relations"]}
        assert claims["a"]["standing"] == "contested"
        assert claims["a"]["contests"] == ["b"]
        assert claims["b"]["contests"] == ["a"]
        assert {"contests", "standing"} <= set(claims["a"]["provenance"]["defaults"])

    def test_build_returns_the_slots(self, tmp_path: Path) -> None:
        root = _write(tmp_path, _tree(**{"records/relations.yaml": [_power("a", "PS1")]}))
        document, _warnings = build_facility_with_warnings(root, project_name="p")
        assert document["relations"][0]["standing"] == "accepted"

    def test_no_claims_leaves_the_file_byte_identical(self, tmp_path: Path) -> None:
        files = _tree()
        with_stage = _run(tmp_path / "a", files)
        later = [stage for stage in LATER_STAGES if stage[0] != "relations"]
        without = run_stages(_write(tmp_path / "b", files), project_name="p", later=later)
        assert with_stage.validated.document is not None
        assert without.validated.document is not None
        assert "relations" not in with_stage.validated.document
        assert facility_bytes(with_stage.validated.document) == facility_bytes(
            without.validated.document
        )
        assert _lines(with_stage) == []


class TestSettled:
    def test_ranked_dispute_supersedes_and_prints_no_warning(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path,
            _tree(
                **{
                    "policy.yaml": _POLICY,
                    "records/relations.yaml": [
                        _power("a", "PS1", **_cited("spec-1")),
                        _power("b", "PS2", **_cited("dwg-1")),
                    ],
                }
            ),
        )
        claims = _claims(report)
        assert claims["a"]["standing"] == "accepted"
        assert claims["a"]["contests"] == ["b"]
        assert claims["b"]["standing"] == "superseded"
        assert claims["b"]["contests"] == ["a"]
        assert _lines(report) == []

    def test_tied_dispute_warns_once_per_key(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path,
            _tree(
                **{
                    "policy.yaml": _POLICY,
                    "records/relations.yaml": [
                        _power("a", "PS1", **_cited("spec-1")),
                        _power("b", "PS2", **_cited("spec-1")),
                        _power("c", "PS3", **_cited("dwg-1")),
                    ],
                }
            ),
        )
        claims = _claims(report)
        assert [claims[rid]["standing"] for rid in "abc"] == [
            "contested",
            "contested",
            "superseded",
        ]
        assert _lines(report) == [
            "facility: relation-contested: powered_by device Q1, no port — claims a, b state "
            "different ends and none ranks first; fix: rank the kinds of their documents in "
            "policy.yaml, or correct the wrong claim"
        ]


class TestWarnings:
    def _files(self) -> dict[str, Any]:
        return _tree(
            **{
                "records/relations.yaml": [
                    _power("q1-a", "PS1"),
                    _power("q1-b", "PS2"),
                    _power("q2-a", "PS1", subject="Q2"),
                    _power("q2-b", "PS3", subject="Q2"),
                    {
                        "id": "q1-at-place",
                        "type": "powered_by",
                        "subject": {"device": "Q1"},
                        "object": {"place": "SR/S01"},
                    },
                ]
            }
        )

    def test_one_contested_line_per_key_and_one_coarse_line_per_type(self, tmp_path: Path) -> None:
        report = _run(tmp_path, self._files())
        summaries = [warning.summary for warning in report.validated.warnings]
        assert summaries == [
            "facility: relation-coarse: relation type powered_by — 1 claim states less than "
            "the facility knows (1 place end)",
            "facility: relation-contested: powered_by device Q1, no port — claims q1-a, q1-b "
            "state different ends and none ranks first",
            "facility: relation-contested: powered_by device Q2, no port — claims q2-a, q2-b "
            "state different ends and none ranks first",
        ]
        coarse = _claims(report)["q1-at-place"]
        assert coarse["standing"] == "accepted"
        assert "contests" not in coarse

    def test_warnings_print_after_a_clean_build_and_exit_zero(self, tmp_path: Path) -> None:
        root = _write(tmp_path, self._files())
        out = io.StringIO()
        assert validate(root, project_name="p", file=out) == 0
        printed = [line for line in out.getvalue().splitlines() if line.startswith("facility: ")]
        assert [line.split(":")[1].strip() for line in printed] == [
            "relation-coarse",
            "relation-contested",
            "relation-contested",
        ]

    def test_coarse_counts_by_reason(self, tmp_path: Path) -> None:
        port = {
            "id": "with-port",
            "type": "powered_by",
            "subject": {"device": "Q2"},
            "object": {"device": "PS1", "port": "OUT1"},
        }
        report = _run(
            tmp_path,
            _tree(
                **{
                    "records/relations.yaml": [
                        port,
                        _power("no-port", "PS1", subject="Q2"),
                        {
                            "id": "at-place",
                            "type": "powered_by",
                            "subject": {"place": "SR/S01"},
                            "object": {"device": "PS2"},
                        },
                    ]
                }
            ),
        )
        assert [warning.summary for warning in report.validated.warnings] == [
            "facility: relation-coarse: relation type powered_by — 2 claims state less than "
            "the facility knows (1 place end, 1 missing port)"
        ]


#: Claims split over three imported layers: two disputes, one coarse claim.
_LAYER_CLAIMS: list[list[dict[str, Any]]] = [
    [_power("one-a", "PS1"), _power("one-b", "PS1", subject="Q2")],
    [_power("two-a", "PS2"), _power("two-b", "PS3", subject="Q2")],
    [
        _power("three-a", "PS3"),
        {
            "id": "three-place",
            "type": "powered_by",
            "subject": {"device": "Q2"},
            "object": {"place": "SR/S01"},
        },
    ],
]


class TestLayerPermutation:
    @pytest.mark.parametrize("policy", [None, _POLICY])
    def test_warning_lines_do_not_depend_on_layer_order(
        self, tmp_path: Path, policy: dict[str, Any] | None
    ) -> None:
        seen: list[tuple[list[str], dict[str, tuple[Any, Any]]]] = []
        for index, order in enumerate(itertools.permutations(range(3))):
            files = _tree()
            if policy is not None:
                files["policy.yaml"] = policy
            for layer, claims in zip(("a", "b", "c"), order, strict=True):
                files[f"imported/{layer}/relations.yaml"] = _LAYER_CLAIMS[claims]
            report = _run(tmp_path / str(index), files)
            slots = {
                rid: (claim["standing"], claim.get("contests"))
                for rid, claim in _claims(report).items()
            }
            seen.append((_lines(report), slots))
        lines, slots = seen[0]
        assert any("relation-contested" in line for line in lines)
        assert any("relation-coarse" in line for line in lines)
        assert all(other == (lines, slots) for other in seen)
