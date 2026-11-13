"""The packaged example facility carries the design's Appendix A, and it builds.

The example ships ``documents.yaml``, ``records/relations.yaml`` and the
Appendix A places, devices and group, but no ``policy.yaml``. Built as it
ships, it holds 18 claims (16 stated, 2 inferred), 16 accepted and 2
contested: the two rack claims on ``SR/QF05-PS`` disagree, so that supply
stays unplaced and the build warns ``relation-contested``; the cable
``w-03-0240`` ends at a rack, so the build warns ``relation-coarse``.
``SR/BPM13-EL`` takes its place ``HALL/ROW3/RK08`` from its rack claim.
With the design's policy added (``rack-layout`` over ``cable-schedule`` for
``located_in``) the cable-schedule claim is superseded, 17 claims are
accepted and ``SR/QF05-PS`` lands in ``HALL/ROW3/RK07``. The example renders
in every channel-finder pipeline mode, and a fresh ``osprey init`` of the
demo preset is still wholly bundled.
"""

from __future__ import annotations

import copy
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import LATER_STAGES
from osprey.facility.bundled import classify
from osprey.facility.render import render_facility_outputs
from osprey.facility.validate import run_stages
from tests._builds import init_project
from tests.facility._appendix_a import CLAIM_IDS, POLICY
from tests.facility.served_tree import IN_PROCESS_RENDER_CONFIG

TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"
EXAMPLE = TEMPLATE_ROOT / "facilities" / "example"

MODES = ("in_context", "hierarchical", "middle_layer")


def _build(facility_dir: Path) -> tuple[dict[str, Any], list[Any]]:
    report = run_stages(facility_dir, project_name="demo", later=LATER_STAGES)
    if report.failed is not None:
        raise AssertionError([error.format_message() for error in report.errors])
    document = report.validated.document
    assert document is not None
    return document, list(report.validated.warnings)


@pytest.fixture(scope="module")
def shipped() -> tuple[dict[str, Any], list[Any]]:
    return _build(EXAMPLE)


@pytest.fixture(scope="module")
def with_policy(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    facility = tmp_path_factory.mktemp("policy") / "facility"
    shutil.copytree(EXAMPLE, facility)
    (facility / "policy.yaml").write_text(yaml.safe_dump(POLICY, sort_keys=False))
    document, _ = _build(facility)
    return document


def _claims(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {claim["id"]: claim for claim in document["relations"]}


def _device(document: dict[str, Any], device_id: str) -> dict[str, Any]:
    [device] = [d for d in document["devices"] if d["id"] == device_id]
    return device


def test_the_example_ships_appendix_a_without_a_policy() -> None:
    assert (EXAMPLE / "documents.yaml").is_file()
    assert (EXAMPLE / "records" / "relations.yaml").is_file()
    assert not (EXAMPLE / "policy.yaml").exists()
    assert not (EXAMPLE / "imported").exists()


def test_shipped_claims_counts_and_standings(shipped: tuple[dict[str, Any], list[Any]]) -> None:
    document, _ = shipped
    claims = _claims(document)
    assert sorted(claims) == sorted(CLAIM_IDS)
    assert len(claims) == 18
    assert Counter(c.get("basis", "stated") for c in claims.values()) == {
        "stated": 16,
        "inferred": 2,
    }
    assert Counter(c["standing"] for c in claims.values()) == {"accepted": 16, "contested": 2}
    contested = sorted(cid for cid, c in claims.items() if c["standing"] == "contested")
    assert contested == ["qf05-ps-rack-a", "qf05-ps-rack-b"]


def test_shipped_warns_contested_and_coarse(shipped: tuple[dict[str, Any], list[Any]]) -> None:
    _, warnings = shipped
    relation = [w for w in warnings if w.kind.startswith("relation-")]
    assert [(w.kind, w.record_kind, w.record_id) for w in relation] == [
        ("relation-coarse", "relation type", "cabled_to"),
        ("relation-contested", "located_in", "device SR/QF05-PS"),
    ]
    [contested] = [w for w in relation if w.kind == "relation-contested"]
    assert "qf05-ps-rack-a" in contested.detail
    assert "qf05-ps-rack-b" in contested.detail


def test_shipped_projects_bpm_electronics_and_leaves_supply_unplaced(
    shipped: tuple[dict[str, Any], list[Any]],
) -> None:
    document, _ = shipped
    electronics = _device(document, "SR/BPM13-EL")
    assert electronics["place"] == "HALL/ROW3/RK08"
    assert electronics["provenance"]["place_from"] == "relation"
    assert _device(document, "SR/QF05-PS").get("place") is None


def test_with_policy_one_claim_is_superseded(with_policy: dict[str, Any]) -> None:
    claims = _claims(with_policy)
    assert Counter(c["standing"] for c in claims.values()) == {"accepted": 17, "superseded": 1}
    assert claims["qf05-ps-rack-b"]["standing"] == "superseded"
    supply = _device(with_policy, "SR/QF05-PS")
    assert supply["place"] == "HALL/ROW3/RK07"
    assert supply["provenance"]["place_from"] == "relation"


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
def test_every_channel_finder_mode_builds(
    shipped: tuple[dict[str, Any], list[Any]], tmp_path: Path, mode: str
) -> None:
    document, _ = shipped
    config = copy.deepcopy(dict(IN_PROCESS_RENDER_CONFIG))
    config["channel_finder"] = {"pipeline_mode": mode}
    written = render_facility_outputs(
        tmp_path, copy.deepcopy(document), config, EXAMPLE, omitted_reported=set()
    )
    assert written


def test_a_fresh_init_stays_wholly_bundled(tmp_path: Path) -> None:
    project = init_project(tmp_path, "demo", "demo")
    facility = project / "data" / "facility"
    assert (facility / "documents.yaml").is_file()
    assert (facility / "records" / "relations.yaml").is_file()
    kinds = classify(facility, TEMPLATE_ROOT)
    assert kinds.wholly_bundled
    assert kinds.name == "example"
