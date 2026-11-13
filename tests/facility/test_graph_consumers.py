"""The consumers of the facility graph read relation claims and groups as the view means them.

The graph view writes each relation claim twice: as an edge between the nodes
its ends name, and as a claim node (``narad_sem:Claim``) carrying the claim's
fields. A group is written as a group node (``narad_sem:Group``). Neither type
is declared ``owl:Class``, because every consumer reads the declared class set
as the device taxonomy. These tests build the design's Appendix A facility and
check that three consumers keep to that:

- the Knowledge panel's payload draws the relation edges between places and
  devices, and no claim or group node, nor any edge to one;
- the channel-finder graph-index parse lists no claim or group among its
  devices or its class nodes;
- the knowledge stub seeder writes one stub per device and none for a claim or
  a group.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from rdflib import OWL, RDF, Graph, URIRef

from osprey.facility.records import FacilityRecords
from osprey.facility.relations import end_identity
from osprey.facility.views.graph import (
    CLAIM,
    CLAIM_OBJECT_PROPERTY_NAMES,
    GROUP,
    GROUP_PROPERTY_NAMES,
    SEMANTIC_SIGNAL,
    GraphSettings,
    graph_text,
)
from osprey.facility.views.graph_iri import iri, namespace
from osprey.interfaces.okf_panel.graph import graph_payload
from osprey.services.channel_finder.graph_index.builder import parse_corpus
from osprey.services.facility_knowledge.seeder.ttl_seeder import seed_from_ttl
from tests.facility._appendix_a import (
    VARIANTS,
    appendix_a_tree,
    build_appendix_a,
    with_readout_control,
)
from tests.facility._graph_relations import NARAD_P, expected_relation_edges

CODE = "p"

#: The view settings, every combination of ``claims`` and ``basis``.
ALL_SETTINGS = [
    GraphSettings(claims, basis)
    for claims in ("all", "contested", "none")
    for basis in ("all", "stated")
]

#: Channels added to the fragment, which has none, so the graph index binds devices.
_CHANNELS = [
    {"id": "SR:QF05:RB", "on": {"device": "SR/QF05"}},
    {"id": "SR:QF05-PS:I", "on": {"device": "SR/QF05-PS"}},
    {"id": "SR:BPM13:X", "on": {"device": "SR/BPM13"}},
    {"id": "SR:ION-PUMP13:P", "on": {"device": "SR/ION-PUMP13"}},
]

_EXCLUDED_NAMESPACES = (namespace("relation"), namespace("group"))


def _tree(variant: str) -> dict[str, Any]:
    tree = with_readout_control(appendix_a_tree(variant))  # type: ignore[arg-type]
    tree["records/channels.yaml"] = copy.deepcopy(_CHANNELS)
    return tree


@pytest.fixture(scope="module", params=VARIANTS)
def built(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp(f"appendix-a-{request.param}") / "data" / "facility"
    document = build_appendix_a(root, tree=_tree(request.param))
    return FacilityRecords(document)


def _graph(text: str) -> Graph:
    return Graph().parse(data=text, format="turtle")


def _typed(graph: Graph, type_iri: str) -> set[str]:
    return {str(subject) for subject in graph.subjects(RDF.type, URIRef(type_iri))}


def _is_excluded(value: str) -> bool:
    return value.startswith(_EXCLUDED_NAMESPACES) or value.rsplit("/", 1)[-1] in {
        "Claim",
        "Group",
    }


class TestTheFixture:
    """The graph these tests read has claims, groups and channels, so no check is vacuous."""

    @pytest.mark.parametrize("settings", ALL_SETTINGS, ids=str)
    def test_the_graph_holds_group_nodes_and_claim_nodes_unless_claims_are_off(
        self, built: FacilityRecords, settings: GraphSettings
    ) -> None:
        graph = _graph(graph_text(built, settings))
        assert _typed(graph, GROUP)
        if settings.claims != "none":
            assert _typed(graph, CLAIM)

    def test_the_document_has_channels_and_relation_claims(self, built: FacilityRecords) -> None:
        assert built.document["channels"]
        assert built.document["relations"]


class TestThePanelPayload:
    """``graph_payload`` draws relation edges and leaves claims and groups out."""

    @pytest.mark.parametrize("settings", ALL_SETTINGS, ids=str)
    def test_relation_edges_between_places_and_devices_are_drawn(
        self, built: FacilityRecords, settings: GraphSettings
    ) -> None:
        payload = graph_payload(_graph(graph_text(built, settings)))
        nodes = payload["nodes"]
        drawn = {
            (
                (nodes[source]["kind"], nodes[source].get("record")),
                relation,
                (nodes[target]["kind"], nodes[target].get("record")),
            )
            for source, target, relation in payload["edges"]
        }
        expected = set()
        for subject, predicate, obj in expected_relation_edges(built, settings):
            ends = [_end_of(built, subject), _end_of(built, obj)]
            if all(end[0] in {"place", "device"} for end in ends):
                expected.add((ends[0], predicate.removeprefix(NARAD_P), ends[1]))
        assert expected
        assert expected <= drawn

    @pytest.mark.parametrize("settings", ALL_SETTINGS, ids=str)
    def test_no_claim_or_group_is_a_node(
        self, built: FacilityRecords, settings: GraphSettings
    ) -> None:
        graph = _graph(graph_text(built, settings))
        payload = graph_payload(graph)
        kinds = Counter(node["kind"] for node in payload["nodes"])
        document = built.document
        assert kinds["place"] == len(document["places"])
        assert kinds["device"] == len(document["devices"])
        assert kinds["channel"] == len(document["channels"])
        assert kinds["signal"] == len(_typed(graph, SEMANTIC_SIGNAL))
        assert kinds["class"] == len(set(graph.subjects(RDF.type, OWL.Class)))
        assert sum(kinds.values()) == len(payload["nodes"])
        devices = {node["record"] for node in payload["nodes"] if node["kind"] == "device"}
        assert devices == {device["id"] for device in document["devices"]}
        for node in payload["nodes"]:
            assert node["label"] not in {"Claim", "Group"}
            assert node["type"] not in {"Claim", "Group"}

    @pytest.mark.parametrize("settings", ALL_SETTINGS, ids=str)
    def test_no_edge_touches_a_claim_or_a_group(
        self, built: FacilityRecords, settings: GraphSettings
    ) -> None:
        payload = graph_payload(_graph(graph_text(built, settings)))
        relations = {relation for _source, _target, relation in payload["edges"]}
        # The claim and group node-valued properties never reach the payload as edges.
        assert not relations & (CLAIM_OBJECT_PROPERTY_NAMES | set(GROUP_PROPERTY_NAMES))
        assert "boundedBy" not in relations  # its subject is the group SR/VS03


class TestTheGraphIndex:
    """``parse_corpus`` lists no claim or group among the devices or the class nodes."""

    @pytest.mark.parametrize("settings", ALL_SETTINGS, ids=str)
    def test_no_claim_or_group_is_a_device_or_a_class(
        self, built: FacilityRecords, settings: GraphSettings
    ) -> None:
        corpus = parse_corpus(graph_text(built, settings))
        assert corpus.binding_rows
        bound = {row.device_uri for row in corpus.binding_rows}
        assert bound == {
            iri(CODE, "device", channel["on"]["device"]) for channel in built.document["channels"]
        }
        for row in corpus.binding_rows:
            assert not _is_excluded(row.device_uri)
            assert not any(_is_excluded(uri) for uri in row.class_uris)
        assert corpus.class_rows
        for row in corpus.class_rows:
            assert not _is_excluded(row.uri)
            assert row.name not in {"Claim", "Group"}


class TestTheStubSeeder:
    """``seed_from_ttl`` writes one stub per device and none for a claim or a group."""

    @pytest.mark.parametrize("settings", ALL_SETTINGS, ids=str)
    def test_one_stub_per_device(
        self, built: FacilityRecords, settings: GraphSettings, tmp_path: Path
    ) -> None:
        ttl = tmp_path / "facility.ttl"
        ttl.write_text(graph_text(built, settings), encoding="utf-8")
        stubs = seed_from_ttl(ttl)
        resources = {stub.resource for stub in stubs}
        assert len(resources) == len(stubs)
        assert resources == {
            iri(CODE, "device", device["id"]) for device in built.document["devices"]
        }
        for stub in stubs:
            assert not _is_excluded(stub.resource)
            assert stub.device_class not in {"Claim", "Group"}


def _end_of(records: FacilityRecords, value: str) -> tuple[str, str | None]:
    """The ``(kind, record id)`` an IRI written for a relation end names."""
    for claim in records.document["relations"]:
        for end in (claim["subject"], claim["object"]):
            identity = end_identity(end)
            if identity is not None and iri(CODE, *identity) == value:
                return identity
    raise AssertionError(f"no relation end mints {value}")
