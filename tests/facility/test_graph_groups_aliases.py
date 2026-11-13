"""The graph view writes group nodes for claim ends and scheme names as alt labels."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
from rdflib import OWL, RDF, SKOS, Graph, Literal, URIRef

from osprey.facility.records import FacilityRecords
from osprey.facility.views.graph import NARAD_SEM, GraphSettings, graph_text
from osprey.facility.views.graph_iri import iri
from tests.facility._appendix_a import VARIANTS, appendix_a_tree, build_appendix_a
from tests.facility._graph_relations import NARAD_P

CODE = "p"
GROUP = URIRef(f"{NARAD_SEM}Group")
GROUP_ID = URIRef(f"{NARAD_P}groupId")
HAS_MEMBER = URIRef(f"{NARAD_P}hasMember")
VS03 = "SR/VS03"
SETTINGS = [
    GraphSettings(claims, basis)
    for claims, basis in itertools.product(("all", "contested", "none"), ("all", "stated"))
]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    return {
        variant: build_appendix_a(tmp_path_factory.mktemp(variant) / "facility", variant)
        for variant in VARIANTS
    }


def _graph(document: dict[str, Any], settings: GraphSettings = GraphSettings()) -> Graph:
    text = graph_text(FacilityRecords(document), settings)
    return Graph().parse(data=text, format="turtle")


def _group(document: dict[str, Any], group_id: str) -> dict[str, Any]:
    return next(row for row in document["groups"] if row["id"] == group_id)


def _alt_labels(graph: Graph, node: str) -> set[str]:
    return {str(label) for label in graph.objects(URIRef(node), SKOS.altLabel)}


def _record_name_match(graph: Graph, name: str) -> set[str]:
    """The record-name match rule: a non-class node with an altLabel equal to ``name``, any case."""
    classes = set(graph.subjects(RDF.type, OWL.Class))
    return {
        str(node)
        for node, label in graph.subject_objects(SKOS.altLabel)
        if node not in classes and str(label).lower() == name.lower()
    }


class TestGroupNodes:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_bounded_by_group_is_a_node_with_its_members(
        self, built: dict[str, dict[str, Any]], variant: str
    ) -> None:
        document = built[variant]
        graph = _graph(document)
        node = URIRef(iri(CODE, "group", VS03))
        assert (node, RDF.type, GROUP) in graph
        assert set(graph.objects(node, GROUP_ID)) == {Literal(VS03)}
        members = {str(member) for member in graph.objects(node, HAS_MEMBER)}
        assert members == {iri(CODE, "device", m) for m in _group(document, VS03)["members"]}
        # The label and description of a group are not written on its node.
        assert not list(graph.objects(node, URIRef(f"{NARAD_P}description")))

    def test_group_type_is_not_a_class(self, built: dict[str, dict[str, Any]]) -> None:
        graph = _graph(built["no_policy"])
        assert (GROUP, RDF.type, OWL.Class) not in graph
        assert not list(graph.predicate_objects(GROUP))

    def test_group_properties_are_declared(self, built: dict[str, dict[str, Any]]) -> None:
        graph = _graph(built["no_policy"])
        assert (GROUP_ID, RDF.type, OWL.DatatypeProperty) in graph
        assert (HAS_MEMBER, RDF.type, OWL.ObjectProperty) in graph

    @pytest.mark.parametrize("settings", SETTINGS, ids=str)
    def test_group_node_stays_under_every_setting(
        self, built: dict[str, dict[str, Any]], settings: GraphSettings
    ) -> None:
        # B-2: the vs03 claims are stated and uncontested, so under ``claims:
        # contested`` or ``none`` they have no claim node; their group keeps its node.
        document = built["no_policy"]
        graph = _graph(document, settings)
        node = URIRef(iri(CODE, "group", VS03))
        assert (node, RDF.type, GROUP) in graph
        members = {str(member) for member in graph.objects(node, HAS_MEMBER)}
        assert members == {iri(CODE, "device", m) for m in _group(document, VS03)["members"]}

    def test_groups_not_claim_ends_get_no_node(self, tmp_path: Path) -> None:
        tree = appendix_a_tree()
        tree["records/relations.yaml"] = [
            row
            for row in tree["records/relations.yaml"]
            if "group" not in row["subject"] and "group" not in row["object"]
        ]
        document = build_appendix_a(tmp_path / "facility", tree=tree)
        assert document["groups"]
        graph = _graph(document)
        assert not list(graph.subjects(RDF.type, GROUP))
        for predicate in (GROUP_ID, HAS_MEMBER):
            assert not list(graph.subject_objects(predicate))
            assert not list(graph.predicate_objects(predicate))

    def test_group_names_are_alt_labels(self, tmp_path: Path) -> None:
        tree = appendix_a_tree()
        tree["records/groups.yaml"][0]["names"] = ["VS3", "Sector 3 vacuum"]
        graph = _graph(build_appendix_a(tmp_path / "facility", tree=tree))
        assert _alt_labels(graph, iri(CODE, "group", VS03)) == {"VS3", "Sector 3 vacuum"}


class TestAliases:
    def test_alias_found_by_record_name_match(self, built: dict[str, dict[str, Any]]) -> None:
        graph = _graph(built["no_policy"])
        assert _record_name_match(graph, "ps-q-0301") == {iri(CODE, "device", "SR/QF05-PS")}
        assert _record_name_match(graph, "QF05 SUPPLY") == {iri(CODE, "device", "SR/QF05-PS")}
        assert _record_name_match(graph, "mag-q-0301") == {iri(CODE, "device", "SR/MAG-Q-0301")}

    def test_scheme_is_not_written(self, built: dict[str, dict[str, Any]]) -> None:
        text = graph_text(FacilityRecords(built["no_policy"]))
        for scheme in ("cable-schedule", "drawings", "label-match"):
            assert scheme not in text

    @pytest.mark.parametrize("claims", ("all", "contested", "none"))
    def test_stated_basis_drops_inferred_alias(
        self, built: dict[str, dict[str, Any]], claims: str
    ) -> None:
        graph = _graph(built["no_policy"], GraphSettings(claims, "stated"))
        labels = _alt_labels(graph, iri(CODE, "device", "SR/QF05-PS"))
        assert "PS-Q-0301" in labels
        assert "QF05 supply" not in labels
        assert _record_name_match(graph, "QF05 supply") == set()

    def test_aliases_merge_with_names(self, tmp_path: Path) -> None:
        tree = appendix_a_tree()
        device = next(r for r in tree["records/devices.yaml"] if r["id"] == "SR/QF05-PS")
        device["names"] = ["QF05 PS", "PS-Q-0301"]
        tree["records/places.yaml"][-1]["names"] = ["Arc 3"]
        tree["records/places.yaml"][-1]["scheme_names"] = [{"name": "S03", "scheme": "survey"}]
        place_id = tree["records/places.yaml"][-1]["id"]
        graph = _graph(build_appendix_a(tmp_path / "facility", tree=tree))
        node = URIRef(iri(CODE, "device", "SR/QF05-PS"))
        assert _alt_labels(graph, str(node)) == {"QF05 PS", "PS-Q-0301", "QF05 supply"}
        assert len(list(graph.objects(node, SKOS.altLabel))) == 3
        assert _alt_labels(graph, iri(CODE, "place", place_id)) == {"Arc 3", "S03"}
