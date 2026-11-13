"""The graph view writes each relation claim as one edge between the nodes its ends name."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import pytest
from rdflib import OWL, RDF, Graph, URIRef

from osprey.facility.records import FacilityRecords
from osprey.facility.validate import relation_types
from osprey.facility.views.graph import (
    PROPERTY_NAMES,
    RELATION_PREDICATES,
    GraphSettings,
    graph_text,
)
from osprey.facility.views.graph_iri import iri
from tests.facility._appendix_a import VARIANTS, appendix_a_tree, build_appendix_a
from tests.facility._graph_relations import (
    LOCATED_IN,
    NARAD_P,
    PART_OF,
    expected_located_in,
    expected_part_of,
    expected_relation_edges,
    relation_predicate,
)

CODE = "p"

#: Every relation predicate and its contested twin, as full IRIs.
RELATION_IRIS = frozenset(
    f"{NARAD_P}{name}{suffix}"
    for name in RELATION_PREDICATES.values()
    for suffix in ("", "Contested")
)

ALL_SETTINGS = [
    GraphSettings(claims, basis)
    for claims, basis in itertools.product(("all", "contested", "none"), ("all", "stated"))
]


def _device(name: str) -> str:
    return iri(CODE, "device", name)


def _place(name: str) -> str:
    return iri(CODE, "place", name)


def _p(name: str) -> str:
    return f"{NARAD_P}{name}"


#: The edges the design's Appendix A states without a policy.
NO_POLICY_EDGES = frozenset(
    {
        (_device("SR/TIM-EVR3:OUT4"), _p("partOf"), _device("SR/TIM-EVR3")),
        (_device("SR/BPM13-EL"), _p("locatedIn"), _place("HALL/ROW3/RK08")),
        (_device("SR/QF05-PS"), _p("locatedInContested"), _place("HALL/ROW3/RK07")),
        (_device("SR/QF05-PS"), _p("locatedInContested"), _place("HALL/ROW3/RK08")),
        (_device("SR/QF05"), _p("poweredBy"), _device("SR/QF05-PS")),
        (_device("SR/QF05-PS"), _p("poweredBy"), _device("HALL/PANEL-3A")),
        (_device("SR/QF05"), _p("cabledTo"), _device("SR/QF05-PS")),
        (_device("SR/BPM13"), _p("cabledTo"), _place("HALL/ROW3/RK08")),
        (_device("SR/BPM13"), _p("readOutBy"), _device("SR/BPM13-EL")),
        (_device("SR/QF05-PS"), _p("controlledBy"), _device("CTRL/sr-mag-03")),
        (_device("CTRL/sr-mag-03"), _p("runsOn"), _device("CTRL/srioc03")),
        (_device("SR/BPM13-EL"), _p("triggeredBy"), _device("SR/TIM-EVR3:OUT4")),
        (_device("SR/QF05-PS"), _p("interlockedBy"), _device("SR/MPS-C3")),
        (iri(CODE, "group", "SR/VS03"), _p("boundedBy"), _device("SR/VALVE03")),
        (iri(CODE, "group", "SR/VS03"), _p("boundedBy"), _device("SR/VALVE04")),
        (_device("CTRL/srioc03"), _p("networkedTo"), _device("CTRL/SW-3A")),
        (_device("SR/QF05"), _p("suppliedBy"), _device("SR/LCW-3")),
        (_device("SR/QF05"), _p("sameAs"), _device("SR/MAG-Q-0301")),
    }
)

#: Under the policy ranking rack-layout first, rack-a wins and rack-b is superseded.
POLICY_EDGES = (
    NO_POLICY_EDGES
    - {
        (_device("SR/QF05-PS"), _p("locatedInContested"), _place("HALL/ROW3/RK07")),
        (_device("SR/QF05-PS"), _p("locatedInContested"), _place("HALL/ROW3/RK08")),
    }
) | {(_device("SR/QF05-PS"), _p("locatedIn"), _place("HALL/ROW3/RK07"))}

#: The two inferred claims' edges, which ``basis: stated`` drops.
INFERRED_EDGES = frozenset(
    {
        (_device("SR/BPM13"), _p("readOutBy"), _device("SR/BPM13-EL")),
        (_device("SR/QF05"), _p("sameAs"), _device("SR/MAG-Q-0301")),
    }
)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    return {
        variant: build_appendix_a(tmp_path_factory.mktemp(variant) / "facility", variant)
        for variant in VARIANTS
    }


def _graph(text: str) -> Graph:
    return Graph().parse(data=text, format="turtle")


def _relation_edges(graph: Graph) -> frozenset[tuple[str, str, str]]:
    return frozenset((str(s), str(p), str(o)) for s, p, o in graph if str(p) in RELATION_IRIS)


def _base_edges(document: dict[str, Any]) -> frozenset[tuple[str, str, str]]:
    """The ``partOf`` and ``locatedIn`` edges places and devices carry without claims."""
    return frozenset(
        {(s, PART_OF, o) for s, o in expected_part_of(document)}
        | {(s, LOCATED_IN, o) for s, o in expected_located_in(document)}
    )


def _text(document: dict[str, Any], settings: GraphSettings = GraphSettings()) -> str:
    return graph_text(FacilityRecords(document), settings)


def _build(tmp_path: Path, tree: dict[str, Any]) -> dict[str, Any]:
    return build_appendix_a(tmp_path / "facility", tree=tree)


def _device_row(tree: dict[str, Any], device_id: str) -> dict[str, Any]:
    return next(row for row in tree["records/devices.yaml"] if row["id"] == device_id)


def _objects(graph: Graph, subject: str, predicate: str) -> set[str]:
    return {str(o) for o in graph.objects(URIRef(subject), URIRef(predicate))}


class TestPredicates:
    def test_keys_are_the_relation_types(self) -> None:
        assert set(RELATION_PREDICATES) == set(relation_types())

    def test_each_name_is_its_type_in_camel_case(self) -> None:
        assert {
            relation_type: relation_predicate(relation_type) for relation_type in relation_types()
        } == dict(RELATION_PREDICATES)

    def test_same_as_is_not_owl_same_as(self) -> None:
        assert RELATION_PREDICATES["same_as"] == "sameAs"


class TestAppendixAEdges:
    def test_the_expectation_matches_the_design_without_a_policy(
        self, built: dict[str, dict[str, Any]]
    ) -> None:
        assert expected_relation_edges(built["no_policy"]) == NO_POLICY_EDGES

    def test_the_expectation_matches_the_design_under_the_policy(
        self, built: dict[str, dict[str, Any]]
    ) -> None:
        assert expected_relation_edges(built["located_in_precedence"]) == POLICY_EDGES

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("settings", ALL_SETTINGS, ids=lambda s: f"{s.claims}-{s.basis}")
    def test_the_view_writes_exactly_the_expected_edges(
        self, built: dict[str, dict[str, Any]], variant: str, settings: GraphSettings
    ) -> None:
        document = built[variant]
        graph = _graph(_text(document, settings))
        assert _relation_edges(graph) == (
            expected_relation_edges(document, settings) | _base_edges(document)
        )

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_basis_stated_drops_exactly_the_inferred_edges(
        self, built: dict[str, dict[str, Any]], variant: str
    ) -> None:
        document = built[variant]
        every = _relation_edges(_graph(_text(document)))
        stated = _relation_edges(_graph(_text(document, GraphSettings(basis="stated"))))
        assert every - stated == INFERRED_EDGES

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_the_claims_setting_never_changes_the_edges(
        self, built: dict[str, dict[str, Any]], variant: str
    ) -> None:
        document = built[variant]
        for basis in ("all", "stated"):
            edges = {
                _relation_edges(_graph(_text(document, GraphSettings(claims, basis))))
                for claims in ("all", "contested", "none")
            }
            assert len(edges) == 1

    def test_a_superseded_claim_writes_no_edge(self, built: dict[str, dict[str, Any]]) -> None:
        graph = _graph(_text(built["located_in_precedence"]))
        qf05_ps = _device("SR/QF05-PS")
        assert _objects(graph, qf05_ps, LOCATED_IN) == {_place("HALL/ROW3/RK07")}
        assert not _objects(graph, qf05_ps, _p("locatedInContested"))

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_no_owl_same_as(self, built: dict[str, dict[str, Any]], variant: str) -> None:
        graph = _graph(_text(built[variant]))
        assert not list(graph.triples((None, OWL.sameAs, None)))
        assert _objects(graph, _device("SR/QF05"), _p("sameAs")) == {_device("SR/MAG-Q-0301")}


class TestDeclarations:
    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("basis", ["all", "stated"])
    def test_each_written_predicate_and_only_it_is_declared(
        self, built: dict[str, dict[str, Any]], variant: str, basis: str
    ) -> None:
        document = built[variant]
        graph = _graph(_text(document, GraphSettings(basis=basis)))
        written = {p for _s, p, _o in _relation_edges(graph)}
        declared = {
            str(s) for s in graph.subjects(RDF.type, OWL.ObjectProperty) if str(s) in RELATION_IRIS
        }
        fixed = {_p(name) for name in PROPERTY_NAMES}
        assert declared - fixed == written - fixed
        assert written <= declared

    def test_a_document_without_claims_declares_no_relation_predicate(
        self, built: dict[str, dict[str, Any]]
    ) -> None:
        document = copy.deepcopy(built["no_policy"])
        document["relations"] = []
        text = _text(document)
        graph = _graph(text)
        declared = {str(s) for s in graph.subjects(RDF.type, None) if str(s).startswith(NARAD_P)}
        assert declared == {_p(name) for name in PROPERTY_NAMES}
        assert iri(CODE, "group", "SR/VS03") not in text


class TestOrder:
    def test_claim_order_does_not_change_the_turtle(self, built: dict[str, dict[str, Any]]) -> None:
        document = built["no_policy"]
        reversed_document = copy.deepcopy(document)
        reversed_document["relations"] = list(reversed(reversed_document["relations"]))
        # The header line hashes the facility file's bytes, which the order changes.
        body = _text(document).split("\n", 1)[1]
        assert _text(reversed_document).split("\n", 1)[1] == body


class TestLocatedInBesideThePlace:
    def test_an_agreeing_claim_collapses_into_the_place_edge(self, tmp_path: Path) -> None:
        tree = appendix_a_tree()
        _device_row(tree, "SR/BPM13-EL")["place"] = "HALL/ROW3/RK08"
        graph = _graph(_text(_build(tmp_path, tree)))
        assert _objects(graph, _device("SR/BPM13-EL"), LOCATED_IN) == {_place("HALL/ROW3/RK08")}

    def test_a_disagreeing_claim_stays_a_second_edge(self, tmp_path: Path) -> None:
        tree = appendix_a_tree()
        _device_row(tree, "SR/BPM13-EL")["place"] = "HALL/ROW3/RK07"
        graph = _graph(_text(_build(tmp_path, tree)))
        assert _objects(graph, _device("SR/BPM13-EL"), LOCATED_IN) == {
            _place("HALL/ROW3/RK07"),
            _place("HALL/ROW3/RK08"),
        }


def _reverse_cable(claim_id: str, cable: str) -> dict[str, Any]:
    """A ``cabled_to`` claim stating QF05's power cable from the supply's side."""
    return {
        "id": claim_id,
        "type": "cabled_to",
        "subject": {"device": "SR/QF05-PS", "port": "OUT1"},
        "object": {"device": "SR/QF05", "port": "T1"},
        "attributes": {"cable": cable},
    }


class TestSymmetricDirection:
    @pytest.mark.parametrize(
        ("claim_id", "cable", "forward"),
        [
            ("w-rev", "W-03-0117", True),
            ("a-rev", "W-03-0117", False),
            ("w-rev", "W-03-0999", True),
            ("a-rev", "W-03-0999", False),
        ],
        ids=[
            "same-cable-later-id",
            "same-cable-earlier-id",
            "other-cable-later-id",
            "other-cable-earlier-id",
        ],
    )
    def test_opposite_claims_give_one_edge_in_the_smallest_ids_direction(
        self, tmp_path: Path, claim_id: str, cable: str, forward: bool
    ) -> None:
        tree = appendix_a_tree()
        tree["records/relations.yaml"].append(_reverse_cable(claim_id, cable))
        document = _build(tmp_path, tree)
        standings = {c["id"]: c["standing"] for c in document["relations"]}
        assert standings[claim_id] == standings["w-03-0117"] == "accepted"

        graph = _graph(_text(document))
        qf05, supply = _device("SR/QF05"), _device("SR/QF05-PS")
        cabled = {
            (s, o)
            for s, p, o in _relation_edges(graph)
            if p == _p("cabledTo") and {s, o} == {qf05, supply}
        }
        assert cabled == ({(qf05, supply)} if forward else {(supply, qf05)})
        assert _relation_edges(graph) == (expected_relation_edges(document) | _base_edges(document))
