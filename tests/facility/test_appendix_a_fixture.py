"""The Appendix A fixture builds, and its claims stand as the design reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests._preset_data import packaged_facility_dir
from tests.facility._appendix_a import (
    ADDED_IDS,
    CLAIM_IDS,
    READOUT_CONTROL_CLAIM,
    SPAN_STAND_IN,
    STUB_IDS,
    VARIANTS,
    appendix_a_tree,
    build_appendix_a,
    example_without_appendix_a,
    with_readout_control,
)

_KIND_FILES = {
    "documents": "documents.yaml",
    "places": "records/places.yaml",
    "devices": "records/devices.yaml",
    "groups": "records/groups.yaml",
    "claims": "records/relations.yaml",
}


def _load(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def _claims(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {claim["id"]: claim for claim in document.get("relations", [])}


def _basis(claim: dict[str, Any]) -> str:
    return claim.get("basis", "stated")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    return {
        variant: build_appendix_a(tmp_path_factory.mktemp(variant) / "facility", variant)
        for variant in VARIANTS
    }


class TestNoPolicy:
    def test_counts_match_the_design_summary(self, built: dict[str, dict[str, Any]]) -> None:
        claims = built["no_policy"]["relations"]
        assert len(claims) == 18
        assert Counter(_basis(claim) for claim in claims) == {"stated": 16, "inferred": 2}
        assert Counter(claim["standing"] for claim in claims) == {
            "accepted": 16,
            "contested": 2,
        }

    def test_the_two_racks_contest_each_other(self, built: dict[str, dict[str, Any]]) -> None:
        claims = _claims(built["no_policy"])
        assert claims["qf05-ps-rack-a"]["standing"] == "contested"
        assert claims["qf05-ps-rack-b"]["standing"] == "contested"
        assert claims["qf05-ps-rack-a"]["contests"] == ["qf05-ps-rack-b"]
        assert claims["qf05-ps-rack-b"]["contests"] == ["qf05-ps-rack-a"]

    def test_inferred_claims_are_readout_and_identity(
        self, built: dict[str, dict[str, Any]]
    ) -> None:
        inferred = {c["id"] for c in built["no_policy"]["relations"] if _basis(c) == "inferred"}
        assert inferred == {"bpm13-readout", "qf05-identity"}

    def test_every_claim_id_is_built(self, built: dict[str, dict[str, Any]]) -> None:
        assert set(_claims(built["no_policy"])) == set(CLAIM_IDS)
        assert len(CLAIM_IDS) == 18


class TestPrecedence:
    def test_rack_layout_wins_over_cable_schedule(self, built: dict[str, dict[str, Any]]) -> None:
        claims = _claims(built["located_in_precedence"])
        assert claims["qf05-ps-rack-a"]["standing"] == "accepted"
        assert claims["qf05-ps-rack-b"]["standing"] == "superseded"

    def test_counts_with_policy(self, built: dict[str, dict[str, Any]]) -> None:
        claims = built["located_in_precedence"]["relations"]
        assert len(claims) == 18
        assert Counter(claim["standing"] for claim in claims) == {
            "accepted": 17,
            "superseded": 1,
        }

    def test_policy_file_has_the_loader_shape(self) -> None:
        policy = appendix_a_tree("located_in_precedence")["policy.yaml"]
        assert policy["relations"] == {
            "precedence": {"located_in": ["rack-layout", "cable-schedule"]}
        }
        assert "policy.yaml" not in appendix_a_tree("no_policy")


class TestProjectedPlaces:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_rack_claims_project_as_the_design_reports(
        self, built: dict[str, dict[str, Any]], variant: str
    ) -> None:
        devices = {device["id"]: device for device in built[variant]["devices"]}
        expected = {
            "SR/BPM13-EL": "HALL/ROW3/RK08",
            "SR/QF05-PS": "HALL/ROW3/RK07" if variant == "located_in_precedence" else None,
        }
        assert {rid: devices[rid].get("place") for rid in expected} == expected
        for rid, place in expected.items():
            place_from = devices[rid]["provenance"].get("place_from")
            assert place_from == ("relation" if place else None), rid


class TestRecords:
    def test_places_devices_groups_documents(self, built: dict[str, dict[str, Any]]) -> None:
        document = built["no_policy"]
        devices = {device["id"]: device for device in document["devices"]}
        assert devices["CTRL/srioc03"]["class"] == "Host"
        assert devices["SR/MAG-Q-0301"]["class"] == "Quadrupole"
        assert [s["layer"] for s in devices["SR/MAG-Q-0301"]["provenance"]["sources"]] == [
            "authored"
        ]
        assert [n["name"] for n in devices["SR/QF05-PS"]["scheme_names"]] == [
            "PS-Q-0301",
            "QF05 supply",
        ]
        places = {place["id"] for place in document["places"]}
        assert {"HALL/ROW3/RK07", "HALL/ROW3/RK08", "SR/SECT3/G2"} <= places
        groups = {group["id"]: group for group in document["groups"]}
        assert set(groups["SR/VS03"]["members"]) == {"SR/ION-PUMP13", "SR/BPM13", "SR/QF05"}
        assert {d["id"] for d in document["documents"]} == {
            "PS-LIST",
            "RACK-ROW3",
            "CS-RING",
            "NET-PLAN",
        }

    def test_tree_is_a_fresh_copy(self) -> None:
        tree = appendix_a_tree()
        tree["records/relations.yaml"].clear()
        assert len(appendix_a_tree()["records/relations.yaml"]) == 18

    def test_unknown_variant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown Appendix A variant"):
            appendix_a_tree("bogus")  # type: ignore[arg-type]


class TestReadoutControl:
    def test_adds_the_missing_controlled_by_claim(self, tmp_path: Path) -> None:
        tree = appendix_a_tree()
        extended = with_readout_control(tree)
        assert len(tree["records/relations.yaml"]) == 18
        document = build_appendix_a(tmp_path / "facility", tree=extended)
        claims = _claims(document)
        assert len(claims) == 19
        added = claims[READOUT_CONTROL_CLAIM["id"]]
        assert added["type"] == "controlled_by"
        assert added["subject"] == {"device": "SR/BPM13-EL"}
        assert added["object"] == {"device": "CTRL/sr-mag-03"}
        assert added["standing"] == "accepted"


class TestReadsTheExample:
    """The fixture's Appendix A records are the packaged example's, and the pre variant drops them."""

    def test_the_fixture_holds_the_examples_appendix_a_records(self) -> None:
        tree = appendix_a_tree()
        example = packaged_facility_dir()
        for kind, relpath in _KIND_FILES.items():
            shipped = {r["id"]: r for r in _load(example / relpath)}
            added = [shipped[rid] for rid in ADDED_IDS[kind]]
            fixture = [r for r in tree[relpath] if r["id"] in set(ADDED_IDS[kind])]
            assert fixture == added, kind
            assert not set(ADDED_IDS[kind]) & set(STUB_IDS.get(kind, ())), kind
        assert tree["documents.yaml"] == _load(example / "documents.yaml")
        assert tree["records/relations.yaml"] == _load(example / "records/relations.yaml")
        assert ADDED_IDS["claims"] == CLAIM_IDS
        assert "SR/MAG-Q-0301" in ADDED_IDS["devices"]
        assert not any(relpath.startswith("imported/") for relpath in tree)

    def test_the_stubs_sit_where_the_example_puts_them(self) -> None:
        tree = appendix_a_tree()
        example = packaged_facility_dir()
        for kind, field in (("places", "level"), ("devices", "class")):
            relpath = _KIND_FILES[kind]
            shipped = {r["id"]: r for r in _load(example / relpath)}
            stubs = {r["id"]: r for r in tree[relpath] if r["id"] in set(STUB_IDS[kind])}
            assert set(stubs) == set(STUB_IDS[kind]), kind
            for rid, stub in stubs.items():
                assert stub[field] == shipped[rid][field], rid
                # SR/QF05 and SR/BPM13 are span-placed in the example; their stubs author
                # the span's sector instead.
                assert stub.get("place") == shipped[rid].get("place", SPAN_STAND_IN.get(rid)), rid
        devices = {d["id"]: d for d in tree["records/devices.yaml"]}
        assert devices["SR/VALVE04"]["place"] == "SR/SECT4"

    def test_the_example_without_appendix_a_builds_with_no_claim(self, tmp_path: Path) -> None:
        from osprey.facility.build import LATER_STAGES
        from osprey.facility.validate import run_stages

        facility = example_without_appendix_a(tmp_path / "facility")
        assert facility == tmp_path / "facility"
        assert not (facility / "documents.yaml").exists()
        assert not (facility / "records" / "relations.yaml").exists()
        report = run_stages(facility, project_name="p", later=LATER_STAGES)
        assert report.failed is None, [e.format_message() for e in report.errors]
        document = report.validated.document
        assert not document.get("relations")
        assert not document.get("documents")
        for kind in ("places", "devices", "groups"):
            ids = {r["id"] for r in document.get(kind) or []}
            assert not ids & set(ADDED_IDS[kind]), kind
            shipped = {r["id"] for r in _load(packaged_facility_dir() / _KIND_FILES[kind])}
            assert ids == shipped - set(ADDED_IDS[kind]), kind
