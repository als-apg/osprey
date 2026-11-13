"""A group states its kind: ``family`` (the default) or ``section``.

Only a family supplies a device's purpose text and files into the middle-layer
index. A section is a set of things that belong together (a vacuum section, a
zone): it keeps its members and its node in the graph, and is never a family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from rdflib import OWL, RDF, Graph, Literal, URIRef

from osprey.facility.build import LATER_STAGES
from osprey.facility.records import FacilityRecords, is_family
from osprey.facility.validate import run_stages
from osprey.facility.views.channel_finder import middle_layer_families
from osprey.facility.views.graph import graph_text
from osprey.facility.views.graph_iri import iri
from tests._preset_data import packaged_facility_dir
from tests.facility._appendix_a import appendix_a_tree, build_appendix_a
from tests.facility._graph_relations import NARAD_P
from tests.facility._synthetic_trees import write_tree

CODE = "p"
VS03 = "SR/VS03"
TEXT = "Vacuum section 3"
MEMBERS = ("SR/ION-PUMP13", "SR/BPM13", "SR/QF05")
GROUP_KIND = URIRef(f"{NARAD_P}groupKind")
FAMILY_DESCRIPTION = URIRef(f"{NARAD_P}familyDescription")


def _tree(kind: str | None) -> dict[str, Any]:
    """The Appendix A tree, its one group described and of ``kind`` (``None``: unstated).

    Each member holds one channel, so that it counts in the middle-layer index.
    """
    tree = appendix_a_tree()
    tree["records/channels.yaml"] = [
        {"id": f"{member}:STATE", "on": {"device": member}, "description": "State"}
        for member in MEMBERS
    ]
    [group] = tree["records/groups.yaml"]
    group.pop("kind", None)
    group["description"] = TEXT
    if kind is not None:
        group["kind"] = kind
    return tree


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str | None, dict[str, Any]]:
    return {
        kind: build_appendix_a(tmp_path_factory.mktemp(str(kind)) / "facility", tree=_tree(kind))
        for kind in (None, "family", "section")
    }


def _graph(document: dict[str, Any]) -> Graph:
    return Graph().parse(data=graph_text(FacilityRecords(document)), format="turtle")


def _group(document: dict[str, Any]) -> dict[str, Any]:
    [group] = [row for row in document["groups"] if row["id"] == VS03]
    return group


def _family_descriptions(document: dict[str, Any]) -> dict[str, str]:
    graph = _graph(document)
    found = {}
    for member in MEMBERS:
        texts = list(graph.objects(URIRef(iri(CODE, "device", member)), FAMILY_DESCRIPTION))
        if texts:
            found[member] = str(texts[0])
    return found


class TestTheSlot:
    @pytest.mark.parametrize("kind", ["family", "section"])
    def test_a_stated_kind_is_kept(
        self, built: dict[str | None, dict[str, Any]], kind: str
    ) -> None:
        assert _group(built[kind])["kind"] == kind

    def test_an_unstated_kind_is_not_written(self, built: dict[str | None, dict[str, Any]]) -> None:
        group = _group(built[None])
        assert "kind" not in group
        assert "kind" not in (group.get("provenance") or {}).get("defaults", [])

    def test_an_unstated_kind_is_a_family(self) -> None:
        assert is_family({"id": "G"})
        assert is_family({"id": "G", "kind": "family"})
        assert not is_family({"id": "G", "kind": "section"})

    def test_another_kind_stops_the_build(self, tmp_path: Path) -> None:
        root = tmp_path / "facility"
        write_tree(root, _tree("zone"))
        report = run_stages(root, project_name="p", later=LATER_STAGES)
        assert report.failed is not None
        messages = [error.format_message() for error in report.errors]
        assert any("groups.0.kind" in m and "'family' or 'section'" in m for m in messages), (
            messages
        )


class TestPurposeText:
    @pytest.mark.parametrize("kind", [None, "family"])
    def test_a_family_describes_its_members(
        self, built: dict[str | None, dict[str, Any]], kind: str | None
    ) -> None:
        assert _family_descriptions(built[kind]) == dict.fromkeys(MEMBERS, TEXT)

    def test_a_section_describes_no_member(self, built: dict[str | None, dict[str, Any]]) -> None:
        assert _family_descriptions(built["section"]) == {}


class TestMiddleLayerFamilies:
    @staticmethod
    def _pairs(document: dict[str, Any]) -> set[tuple[str, str]]:
        families = middle_layer_families(FacilityRecords(document))
        return {pair for pairs in families.values() for pair in pairs}

    @pytest.mark.parametrize("kind", [None, "family"])
    def test_a_family_is_filed(
        self, built: dict[str | None, dict[str, Any]], kind: str | None
    ) -> None:
        assert ("SR", "VS03") in self._pairs(built[kind])

    def test_a_section_is_not_filed(self, built: dict[str | None, dict[str, Any]]) -> None:
        pairs = self._pairs(built["section"])
        assert ("SR", "VS03") not in pairs
        # Its members file into the derived Families of their classes instead.
        assert {("SR", "Quadrupole"), ("SR", "BeamPositionMonitor"), ("SR", "Pump")} <= pairs


class TestGroupNode:
    @pytest.mark.parametrize(
        ("kind", "written"), [(None, "family"), ("family", "family"), ("section", "section")]
    )
    def test_the_node_states_its_kind(
        self, built: dict[str | None, dict[str, Any]], kind: str | None, written: str
    ) -> None:
        graph = _graph(built[kind])
        node = URIRef(iri(CODE, "group", VS03))
        assert list(graph.objects(node, GROUP_KIND)) == [Literal(written)]
        assert (GROUP_KIND, RDF.type, OWL.DatatypeProperty) in graph

    def test_a_section_keeps_its_members(self, built: dict[str | None, dict[str, Any]]) -> None:
        graph = _graph(built["section"])
        node = URIRef(iri(CODE, "group", VS03))
        members = {str(member) for member in graph.objects(node, URIRef(f"{NARAD_P}hasMember"))}
        assert members == {iri(CODE, "device", member) for member in MEMBERS}


class TestTheExample:
    def test_the_vacuum_section_is_a_described_section(self) -> None:
        groups = yaml.safe_load(
            (packaged_facility_dir() / "records" / "groups.yaml").read_text(encoding="utf-8")
        )
        [section] = [group for group in groups if group.get("kind") == "section"]
        assert section["id"] == VS03
        assert section["description"] == TEXT
        assert all("kind" not in group for group in groups if group["id"] != VS03)
