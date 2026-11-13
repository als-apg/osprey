"""The graph view writes one claim node per relation claim the ``claims`` setting selects."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import pytest
from rdflib import OWL, RDF, Graph, Literal, URIRef

from osprey.facility.records import FacilityRecords
from osprey.facility.views.graph import (
    NARAD_SEM,
    PROPERTY_NAMES,
    RELATION_PREDICATES,
    GraphSettings,
    graph_text,
)
from osprey.facility.views.graph_iri import iri
from tests.facility._appendix_a import CLAIM_IDS, VARIANTS, appendix_a_tree, build_appendix_a
from tests.facility._graph_relations import NARAD_P

CODE = "p"
CLAIM = URIRef(f"{NARAD_SEM}Claim")

#: Every literal predicate a claim node may carry, beside its attributes.
CLAIM_LITERALS = (
    "claimId",
    "relationType",
    "basis",
    "rule",
    "standing",
    "document",
    "documentVersion",
    "documentState",
    "locator",
    "subjectPort",
    "objectPort",
    "subjectUnit",
    "objectUnit",
)
CLAIM_LINKS = ("claimSubject", "claimObject", "contests")

#: The Appendix A attributes and the predicates they are written under.
ATTRIBUTE_NAMES = {"cable", "routed_from", "length_m", "signal", "side", "medium"}

#: Every relation predicate and its contested twin, as full IRIs.
RELATION_IRIS = frozenset(
    f"{NARAD_P}{name}{suffix}"
    for name in RELATION_PREDICATES.values()
    for suffix in ("", "Contested")
)

INFERRED = {"bpm13-readout", "qf05-identity"}


def _p(name: str) -> str:
    return f"{NARAD_P}{name}"


def _node(claim_id: str) -> str:
    return iri(CODE, "relation", claim_id)


def _end(end: dict[str, Any]) -> str:
    for kind in ("device", "place", "group"):
        if kind in end:
            return iri(CODE, kind, end[kind])
    raise AssertionError(end)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    return {
        variant: build_appendix_a(tmp_path_factory.mktemp(variant) / "facility", variant)
        for variant in VARIANTS
    }


def _graph(text: str) -> Graph:
    return Graph().parse(data=text, format="turtle")


def _text(document: dict[str, Any], settings: GraphSettings = GraphSettings()) -> str:
    return graph_text(FacilityRecords(document), settings)


def _claims(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {claim["id"]: claim for claim in document["relations"]}


def _claim_nodes(graph: Graph) -> set[str]:
    return {str(s) for s in graph.subjects(RDF.type, CLAIM)}


def _fields(graph: Graph, node: str) -> dict[str, set[Any]]:
    """Every predicate of ``node`` but ``rdf:type`` -> its objects (literals as Python values)."""
    fields: dict[str, set[Any]] = {}
    for predicate, obj in graph.predicate_objects(URIRef(node)):
        if predicate == RDF.type:
            continue
        value = obj.toPython() if isinstance(obj, Literal) else str(obj)
        fields.setdefault(str(predicate), set()).add(value)
    return fields


def _expected(claim: dict[str, Any], written: set[str]) -> dict[str, set[Any]]:
    """The fields a claim node carries, from the claim alone."""
    source = claim.get("source") or {}
    subject, obj = claim["subject"], claim["object"]
    literals = {
        "claimId": claim["id"],
        "relationType": claim["type"],
        "basis": claim.get("basis", "stated"),
        "rule": claim.get("rule"),
        "standing": claim["standing"],
        "document": source.get("document"),
        "documentVersion": source.get("version"),
        "documentState": source.get("state"),
        "locator": source.get("locator"),
        "subjectPort": subject.get("port"),
        "objectPort": obj.get("port"),
        "subjectUnit": subject.get("unit"),
        "objectUnit": obj.get("unit"),
        **(claim.get("attributes") or {}),
    }
    fields = {_p(name): {value} for name, value in literals.items() if value is not None}
    fields[_p("claimSubject")] = {_end(subject)}
    fields[_p("claimObject")] = {_end(obj)}
    rivals = {_node(rival) for rival in claim.get("contests") or [] if rival in written}
    if rivals:
        fields[_p("contests")] = rivals
    return fields


class TestFields:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_every_claim_node_matches_its_claim(
        self, built: dict[str, dict[str, Any]], variant: str
    ) -> None:
        document = built[variant]
        graph = _graph(_text(document))
        claims = _claims(document)
        assert set(claims) == set(CLAIM_IDS)
        assert _claim_nodes(graph) == {_node(claim_id) for claim_id in CLAIM_IDS}
        for claim_id, claim in claims.items():
            assert _fields(graph, _node(claim_id)) == _expected(claim, set(claims)), claim_id

    def test_typed_values(self, built: dict[str, dict[str, Any]]) -> None:
        graph = _graph(_text(built["no_policy"]))
        node = URIRef(_node("bpm13-el-rack"))
        (unit,) = graph.objects(node, URIRef(_p("objectUnit")))
        assert unit.datatype == URIRef("http://www.w3.org/2001/XMLSchema#integer")
        (length,) = graph.objects(URIRef(_node("w-03-0117")), URIRef(_p("length_m")))
        assert length.toPython() == 42 and isinstance(length.toPython(), int)

    def test_group_end_and_inferred_fields(self, built: dict[str, dict[str, Any]]) -> None:
        graph = _graph(_text(built["no_policy"]))
        upstream = _fields(graph, _node("vs03-upstream"))
        assert upstream[_p("claimSubject")] == {iri(CODE, "group", "SR/VS03")}
        assert upstream[_p("side")] == {"upstream"}
        readout = _fields(graph, _node("bpm13-readout"))
        assert readout[_p("basis")] == {"inferred"}
        assert readout[_p("rule")] == {"bpm-electronics-in-cable-destination-rack"}
        assert readout[_p("objectPort")] == {"CH1"}
        stated = _fields(graph, _node("qf05-control"))
        assert stated[_p("basis")] == {"stated"}
        assert _p("document") not in stated and _p("rule") not in stated

    def test_no_dispute_without_policy_contests_both_ways(
        self, built: dict[str, dict[str, Any]]
    ) -> None:
        graph = _graph(_text(built["no_policy"]))
        a, b = _node("qf05-ps-rack-a"), _node("qf05-ps-rack-b")
        assert _fields(graph, a)[_p("contests")] == {b}
        assert _fields(graph, b)[_p("contests")] == {a}
        assert _fields(graph, a)[_p("standing")] == {"contested"}

    def test_policy_winner_and_loser(self, built: dict[str, dict[str, Any]]) -> None:
        graph = _graph(_text(built["located_in_precedence"]))
        a, b = _node("qf05-ps-rack-a"), _node("qf05-ps-rack-b")
        assert _fields(graph, a)[_p("standing")] == {"accepted"}
        assert _fields(graph, b)[_p("standing")] == {"superseded"}
        assert _fields(graph, a)[_p("contests")] == {b}
        assert _fields(graph, b)[_p("contests")] == {a}

    def test_claim_is_not_declared_a_class(self, built: dict[str, dict[str, Any]]) -> None:
        graph = _graph(_text(built["no_policy"]))
        assert (CLAIM, RDF.type, OWL.Class) not in graph
        assert not list(graph.triples((CLAIM, None, None)))


class TestClaimsSetting:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_contested_keeps_the_disputed_pair(
        self, built: dict[str, dict[str, Any]], variant: str
    ) -> None:
        document = built[variant]
        graph = _graph(_text(document, GraphSettings(claims="contested")))
        pair = {_node("qf05-ps-rack-a"), _node("qf05-ps-rack-b")}
        assert _claim_nodes(graph) == pair
        claims = _claims(document)
        for claim_id in ("qf05-ps-rack-a", "qf05-ps-rack-b"):
            expected = _expected(claims[claim_id], {"qf05-ps-rack-a", "qf05-ps-rack-b"})
            assert _fields(graph, _node(claim_id)) == expected

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_none_writes_no_claim_node(
        self, built: dict[str, dict[str, Any]], variant: str
    ) -> None:
        text = _text(built[variant], GraphSettings(claims="none"))
        graph = _graph(text)
        assert not _claim_nodes(graph)
        assert iri(CODE, "relation", "") not in text
        declared = {str(s) for s in graph.subjects(RDF.type, None) if str(s).startswith(NARAD_P)}
        for name in (*CLAIM_LITERALS, *CLAIM_LINKS, *ATTRIBUTE_NAMES):
            assert _p(name) not in declared

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("basis", ["all", "stated"])
    def test_the_setting_never_changes_the_edges(
        self, built: dict[str, dict[str, Any]], variant: str, basis: str
    ) -> None:
        def edges(claims: str) -> set[tuple[str, str, str]]:
            graph = _graph(_text(built[variant], GraphSettings(claims, basis)))
            return {(str(s), str(p), str(o)) for s, p, o in graph if str(p) in RELATION_IRIS}

        assert edges("all") == edges("contested") == edges("none")

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_stated_drops_inferred_nodes_with_no_dangling_contests(
        self, built: dict[str, dict[str, Any]], variant: str
    ) -> None:
        document = copy.deepcopy(built[variant])
        # Make an inferred claim a rival, so its node's absence must drop the link too.
        claims = _claims(document)
        claims["qf05-ps-rack-a"]["contests"] = [
            *claims["qf05-ps-rack-a"]["contests"],
            "qf05-identity",
        ]
        claims["qf05-identity"]["contests"] = ["qf05-ps-rack-a"]
        for setting in ("all", "contested"):
            graph = _graph(_text(document, GraphSettings(setting, "stated")))
            nodes = _claim_nodes(graph)
            assert not nodes & {_node(claim_id) for claim_id in INFERRED}
            for rival in graph.objects(None, URIRef(_p("contests"))):
                assert str(rival) in nodes
            assert _fields(graph, _node("qf05-ps-rack-a"))[_p("contests")] == {
                _node("qf05-ps-rack-b")
            }
        graph = _graph(_text(document, GraphSettings("all", "stated")))
        assert _claim_nodes(graph) == {
            _node(claim_id) for claim_id in CLAIM_IDS if claim_id not in INFERRED
        }


class TestAttributes:
    def _tree(self, attributes: dict[str, Any]) -> dict[str, Any]:
        tree = appendix_a_tree("no_policy")
        claim = next(
            row for row in tree["records/relations.yaml"] if row["id"] == "qf05-ps-interlock"
        )
        claim["attributes"] = attributes
        return tree

    def test_colliding_and_non_identifier_keys(self, tmp_path: Path) -> None:
        document = build_appendix_a(
            tmp_path / "facility",
            tree=self._tree(
                {
                    "standing": "x",
                    "sourceName": "y",
                    "cable-id": "z",
                    "flag": True,
                    "gain": 1.5,
                    "nested": {"a": 1},
                }
            ),
        )
        graph = _graph(_text(document))
        fields = _fields(graph, _node("qf05-ps-interlock"))
        assert fields[_p("attribute_standing")] == {"x"}
        assert fields[_p("attribute_sourceName")] == {"y"}
        assert fields[_p("attribute_cable_x2D_id")] == {"z"}
        assert fields[_p("flag")] == {"true"}
        assert fields[_p("gain")] == {1.5}
        # The claim's own standing and no device-name predicate are touched.
        assert fields[_p("standing")] == {"accepted"}
        assert _p("sourceName") not in fields
        assert not any("nested" in predicate for predicate in fields)
        declared = set(graph.subjects(RDF.type, OWL.DatatypeProperty))
        for name in ("attribute_standing", "attribute_sourceName", "attribute_cable_x2D_id"):
            assert URIRef(_p(name)) in declared


class TestDeclarations:
    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize(
        "settings",
        [
            GraphSettings(c, b)
            for c, b in itertools.product(("all", "contested"), ("all", "stated"))
        ],
    )
    def test_each_written_predicate_and_only_it_is_declared(
        self, built: dict[str, dict[str, Any]], variant: str, settings: GraphSettings
    ) -> None:
        graph = _graph(_text(built[variant], settings))
        nodes = {URIRef(node) for node in _claim_nodes(graph)}
        used = {p for s, p, _o in graph if s in nodes and p != RDF.type}
        fixed = {URIRef(_p(name)) for name in PROPERTY_NAMES}
        candidates = {
            URIRef(_p(name)) for name in (*CLAIM_LITERALS, *CLAIM_LINKS, *ATTRIBUTE_NAMES)
        }
        datatype = set(graph.subjects(RDF.type, OWL.DatatypeProperty))
        objects = set(graph.subjects(RDF.type, OWL.ObjectProperty))
        assert (datatype | objects) & candidates - fixed == used - fixed
        for name in CLAIM_LINKS:
            if URIRef(_p(name)) in used:
                assert URIRef(_p(name)) in objects
        for name in (*CLAIM_LITERALS, *ATTRIBUTE_NAMES):
            if URIRef(_p(name)) in used:
                assert URIRef(_p(name)) in datatype

    def test_contested_declares_no_unused_predicate(self, built: dict[str, dict[str, Any]]) -> None:
        graph = _graph(_text(built["no_policy"], GraphSettings(claims="contested")))
        declared = set(graph.subjects(RDF.type, None))
        for name in ("rule", "subjectPort", "objectPort", "cable", "side", "medium"):
            assert URIRef(_p(name)) not in declared
        assert URIRef(_p("objectUnit")) in declared

    def test_claim_order_does_not_change_the_turtle(self, built: dict[str, dict[str, Any]]) -> None:
        document = built["no_policy"]
        reversed_document = copy.deepcopy(document)
        reversed_document["relations"] = list(reversed(reversed_document["relations"]))
        body = _text(document).split("\n", 1)[1]
        assert _node("w-03-0117") in body
        assert _text(reversed_document).split("\n", 1)[1] == body
