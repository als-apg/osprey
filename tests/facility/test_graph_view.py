"""The graph view: the facility file written as the knowledge graph's Turtle.

``data/graph/facility.ttl`` opens with the facility file's sha256, types every
channel a ``narad_sem:ChannelBinding`` joined to its signal by its role, joins
each place to its parent by ``partOf`` and each placed device to its place by
``locatedIn``, writes the names of places, devices and channels as
``skos:altLabel``, and carries exactly the ``narad_p:`` predicates of
:data:`PREDICATES` when no relation is claimed. The demo, which ships the
Appendix A claims, adds :data:`DEMO_RELATION_PREDICATES`: each accepted claim
is one plain edge, each contested claim one ``…Contested`` edge, each claim one
``narad_sem:Claim`` node, and a group a claim names one ``narad_sem:Group``
node with ``groupId`` and ``hasMember``.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from rdflib import RDF, Graph, Literal, Namespace, URIRef

from osprey.facility.records import FacilityRecords
from osprey.facility.relations import end_identity
from osprey.facility.render import facility_bytes
from osprey.facility.validate import FACILITY_HEADER, relation_types
from osprey.facility.views import VIEWS, ViewInputs
from osprey.facility.views.graph import GRAPH_FILE, HEADER_PREFIX, graph_text, write_graph_view
from osprey.facility.views.graph_iri import iri
from tests.facility._graph_relations import (
    expected_located_in,
    expected_part_of,
    relation_predicate,
)

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

P = Namespace("https://narad.example.org/property/")
SEM = Namespace("https://narad.example.org/schema/shared_semantics/")
GRAPH_VIEW = "data/graph/facility.ttl"

#: Every ``narad_p:`` predicate the view writes.
PREDICATES = {
    "bindingId",
    "description",
    "deviceId",
    "facility",
    "familyDescription",
    "fullPv",
    "hasBinding",
    "rawType",
    "readsSignal",
    "writesSignal",
    "sPositionM",
    "sectionCode",
    "sourceName",
    "system",
    "systemDescription",
    "ordinalInPlace",
    "ordinalInModel",
    "placePath",
    "lengthM",
    "partOf",
    "locatedIn",
}

#: The ``narad_p:`` predicates the demo's relation claims add to :data:`PREDICATES`.
DEMO_RELATION_PREDICATES = {
    # relation types the demo claims, besides partOf and locatedIn
    "boundedBy",
    "cabledTo",
    "controlledBy",
    "interlockedBy",
    "networkedTo",
    "poweredBy",
    "readOutBy",
    "runsOn",
    "sameAs",
    "suppliedBy",
    "triggeredBy",
    # the two contested located_in claims
    "locatedInContested",
    # the claim nodes
    "basis",
    "claimId",
    "claimObject",
    "claimSubject",
    "contests",
    "document",
    "documentState",
    "documentVersion",
    "locator",
    "objectPort",
    "objectUnit",
    "relationType",
    "rule",
    "standing",
    "subjectPort",
    # scalar claim attributes, written under their key
    "cable",
    "length_m",
    "medium",
    "routed_from",
    "side",
    "signal",
    # the group node of SR/VS03
    "groupId",
    "groupKind",
    "hasMember",
}

#: The claims the demo ships, and how they stand without a policy.
DEMO_CLAIMS = 18
DEMO_ACCEPTED = 16
DEMO_CONTESTED = 2


def _doc() -> dict[str, Any]:
    return {
        "schema": FACILITY_HEADER,
        "identity": {"code": "fx"},
        "places": [
            {"id": "M", "level": "machine", "description": "The machine."},
            {"id": "M/S1", "level": "sector"},
            {"id": "M/S2", "level": "sector", "description": "Sector two."},
            {"id": "M/S1/C1", "level": "cell", "names": ["C1", "Cell one"]},
        ],
        "devices": [
            {
                "id": "M/BPM1",
                "class": "BeamPositionMonitor",
                "place": "M/S1",
                "names": ["BPM1"],
                "label": "BPM 1",
                "model": "M",
                "s": 1.25,
                "length": 0.0,
                "ordinalInPlace": 1,
                "ordinalInModel": 1,
            },
            {"id": "M/Q1", "class": "Quadrupole", "place": "M/S2"},
            {"id": "M/LOOSE", "class": "Quadrupole"},
            {"id": "M/S1/C1/BOX", "place": "M/S1/C1"},
        ],
        "channels": [
            {
                "id": "M:BPM1:X",
                "on": {"device": "M/BPM1"},
                "signal": "position_x_readback",
                "description": "Horizontal position.",
                "names": ["BPM1 X"],
            },
            {
                "id": "M:Q1:SP",
                "role": "setpoint",
                "pair": "M:Q1:RB",
                "on": {"device": "M/Q1"},
                "signal": "current_setpoint",
            },
            {
                "id": "M:Q1:RB",
                "on": {"device": "M/Q1"},
                "signal": "current_readback",
                "endpoint_of": ["M/BPM1"],
            },
            {"id": "M:Q1:NOTE", "role": "none", "on": {"device": "M/Q1"}, "signal": "status"},
            {"id": "M:TUNE", "on": {"place": "M"}},
        ],
        "groups": [
            {"id": "M/DIAG", "description": "Diagnostics.", "members": ["M/BPM1", "M/Q1"]},
            {"id": "M/BPM", "description": "Monitors.", "members": ["M/BPM1"]},
        ],
    }


SKOS_ALT_LABEL = URIRef("http://www.w3.org/2004/02/skos/core#altLabel")
SKOS_LABEL = URIRef("http://www.w3.org/2004/02/skos/core#label")


def _node(kind: str, raw: str) -> URIRef:
    return URIRef(iri("fx", kind, raw))


def _graph(text: str) -> Graph:
    return Graph().parse(data=text, format="turtle")


def _narad_predicates(graph: Graph) -> set[str]:
    return {
        str(predicate).removeprefix(str(P))
        for predicate in set(graph.predicates())
        if str(predicate).startswith(str(P))
    }


def _value(graph: Graph, kind: str, raw: str, name: str) -> Any:
    values = list(graph.objects(URIRef(iri("fx", kind, raw)), P[name]))
    assert len(values) <= 1, values
    return values[0].toPython() if values else None


def test_the_first_line_is_the_facility_file_sha256() -> None:
    doc = _doc()
    first = graph_text(FacilityRecords(doc)).split("\n", 1)[0]

    assert first == HEADER_PREFIX + hashlib.sha256(facility_bytes(doc)).hexdigest()


def test_equal_facility_files_give_equal_bytes() -> None:
    assert graph_text(FacilityRecords(_doc())) == graph_text(FacilityRecords(_doc()))


def test_a_device_carries_its_identity_place_and_position() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))
    device = URIRef(iri("fx", "device", "M/BPM1"))

    assert set(graph.objects(device, RDF.type)) == {SEM.BeamPositionMonitor}
    expected = {
        "deviceId": "M/BPM1",
        "facility": "fx",
        "rawType": "BeamPositionMonitor",
        "system": "M",
        "placePath": "M/S1",
        "sectionCode": "S1",
        "sourceName": "M/BPM1",
        "familyDescription": "Monitors.",
        "systemDescription": "The machine.",
        "sPositionM": 1.25,
        "lengthM": 0.0,
        "ordinalInPlace": 1,
        "ordinalInModel": 1,
    }
    for name, value in expected.items():
        assert _value(graph, "device", "M/BPM1", name) == value, name


def test_a_device_carries_its_top_place_and_family_descriptions() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    assert _value(graph, "device", "M/Q1", "systemDescription") == "The machine."
    assert _value(graph, "device", "M/Q1", "familyDescription") == "Diagnostics."


def test_a_device_in_no_described_group_has_no_family_description() -> None:
    doc = _doc()
    doc["groups"] = [
        {"id": "M/BPM", "description": "Monitors.", "members": ["M/BPM1"]},
        {"id": "M/QUADS", "members": ["M/Q1"]},
    ]
    graph = _graph(graph_text(FacilityRecords(doc)))

    assert _value(graph, "device", "M/BPM1", "familyDescription") == "Monitors."
    assert _value(graph, "device", "M/Q1", "familyDescription") is None


def test_a_device_without_s_has_no_position_or_ordinals() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    for name in ("sPositionM", "lengthM", "ordinalInPlace", "ordinalInModel"):
        assert _value(graph, "device", "M/Q1", name) is None, name
    assert _value(graph, "device", "M/Q1", "sourceName") == "M/Q1"


def test_a_place_carries_its_path_and_last_segment() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    assert _value(graph, "place", "M/S2", "placePath") == "M/S2"
    assert _value(graph, "place", "M/S2", "sectionCode") == "S2"
    assert _value(graph, "place", "M", "sectionCode") == "M"


def test_a_channel_is_a_binding_joined_to_its_signal_by_its_role() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    def channel(address: str) -> URIRef:
        return URIRef(iri("fx", "channel", address))

    assert set(graph.objects(channel("M:BPM1:X"), RDF.type)) == {SEM.ChannelBinding}
    assert _value(graph, "channel", "M:BPM1:X", "bindingId") == "M:BPM1:X"
    assert _value(graph, "channel", "M:BPM1:X", "fullPv") == "M:BPM1:X"
    assert _value(graph, "channel", "M:BPM1:X", "description") == "Horizontal position."
    assert list(graph.objects(channel("M:BPM1:X"), P.readsSignal)) == [SEM.position_x_readback]
    assert list(graph.objects(channel("M:Q1:SP"), P.writesSignal)) == [SEM.current_setpoint]
    assert list(graph.objects(channel("M:Q1:SP"), P.readsSignal)) == []
    note = channel("M:Q1:NOTE")
    assert (
        list(graph.objects(note, P.readsSignal)) + list(graph.objects(note, P.writesSignal)) == []
    )
    assert (SEM.current_setpoint, RDF.type, SEM.SemanticSignal) in graph
    assert (SEM.status, RDF.type, SEM.SemanticSignal) not in graph


def test_has_binding_joins_the_owner_and_every_endpoint_device() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))
    readback = URIRef(iri("fx", "channel", "M:Q1:RB"))

    owners = set(graph.subjects(P.hasBinding, readback))
    assert owners == {
        URIRef(iri("fx", "device", "M/Q1")),
        URIRef(iri("fx", "device", "M/BPM1")),
    }
    assert set(graph.subjects(P.hasBinding, URIRef(iri("fx", "channel", "M:TUNE")))) == {
        URIRef(iri("fx", "place", "M"))
    }


def test_the_device_class_and_those_above_it_are_declared() -> None:
    from rdflib import OWL, RDFS

    graph = _graph(graph_text(FacilityRecords(_doc())))

    assert (SEM.BeamPositionMonitor, RDF.type, OWL.Class) in graph
    parents = set(graph.transitive_objects(SEM.BeamPositionMonitor, RDFS.subClassOf))
    assert SEM.AcceleratorDevice in parents


def test_the_view_writes_one_file(tmp_path: Path) -> None:
    doc = _doc()
    inputs = ViewInputs(
        records=FacilityRecords(doc), rendered_config={}, facility_dir=tmp_path, served=[]
    )

    written = write_graph_view(tmp_path / "graph", inputs)

    assert written == [tmp_path / "graph" / GRAPH_FILE]
    assert written[0].read_text(encoding="utf-8") == graph_text(FacilityRecords(doc))


def test_the_graph_view_is_always_written() -> None:
    [view] = [view for view in VIEWS if view.name == "graph"]

    assert view.path == "graph"
    assert view.written_when(
        ViewInputs(records=FacilityRecords({}), rendered_config={}, facility_dir=Path(), served=[])
    ) == (True, "always written")


def test_a_place_below_the_top_is_part_of_its_parent() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    assert list(graph.objects(_node("place", "M/S1"), P.partOf)) == [_node("place", "M")]
    assert list(graph.objects(_node("place", "M/S2"), P.partOf)) == [_node("place", "M")]


def test_a_three_level_place_is_part_of_its_parent_not_the_top() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    assert list(graph.objects(_node("place", "M/S1/C1"), P.partOf)) == [_node("place", "M/S1")]
    chain = list(graph.transitive_objects(_node("place", "M/S1/C1"), P.partOf))
    assert chain == [_node("place", "M/S1/C1"), _node("place", "M/S1"), _node("place", "M")]


def test_a_top_place_is_part_of_nothing() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    assert list(graph.objects(_node("place", "M"), P.partOf)) == []
    assert len(list(graph.subject_objects(P.partOf))) == 3


def test_a_placed_device_is_located_in_its_place() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    assert list(graph.objects(_node("device", "M/BPM1"), P.locatedIn)) == [_node("place", "M/S1")]
    assert list(graph.objects(_node("device", "M/Q1"), P.locatedIn)) == [_node("place", "M/S2")]


def test_a_device_without_a_place_is_located_nowhere() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    assert list(graph.objects(_node("device", "M/LOOSE"), P.locatedIn)) == []
    assert _value(graph, "device", "M/LOOSE", "deviceId") == "M/LOOSE"


def test_a_device_without_a_class_is_still_located_in_its_place() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))
    box = _node("device", "M/S1/C1/BOX")

    assert list(graph.objects(box, RDF.type)) == []
    assert list(graph.objects(box, P.locatedIn)) == [_node("place", "M/S1/C1")]
    assert len(list(graph.subject_objects(P.locatedIn))) == 3


def test_places_devices_and_channels_carry_their_names_once_each() -> None:
    graph = _graph(graph_text(FacilityRecords(_doc())))

    def names(kind: str, raw: str) -> list[str]:
        return sorted(str(name) for name in graph.objects(_node(kind, raw), SKOS_ALT_LABEL))

    assert names("place", "M/S1/C1") == ["C1", "Cell one"]
    assert names("device", "M/BPM1") == ["BPM1"]
    assert names("channel", "M:BPM1:X") == ["BPM1 X"]
    assert names("place", "M") == []
    assert names("device", "M/Q1") == []
    assert names("channel", "M:Q1:SP") == []


def test_a_name_is_written_once_however_often_it_is_listed() -> None:
    doc = _doc()
    doc["devices"][0]["names"] = ["BPM1", "BPM1"]
    text = graph_text(FacilityRecords(doc))

    assert len(list(_graph(text).objects(_node("device", "M/BPM1"), SKOS_ALT_LABEL))) == 1


def test_no_record_label_is_written() -> None:
    from rdflib import RDFS

    graph = _graph(graph_text(FacilityRecords(_doc())))
    device = _node("device", "M/BPM1")

    assert list(graph.objects(device, RDFS.label)) == []
    assert list(graph.objects(device, SKOS_LABEL)) == []
    assert Literal("BPM 1") not in set(graph.objects())


def test_a_document_without_names_gives_no_record_alt_label() -> None:
    from rdflib import OWL

    doc = _doc()
    for kind in ("places", "devices", "channels"):
        for record in doc[kind]:
            record.pop("names", None)
    graph = _graph(graph_text(FacilityRecords(doc)))

    labelled = set(graph.subjects(SKOS_ALT_LABEL, None))
    assert all((subject, RDF.type, OWL.Class) in graph for subject in labelled), labelled


def test_part_of_and_located_in_are_object_properties() -> None:
    from rdflib import OWL

    graph = _graph(graph_text(FacilityRecords(_doc())))

    for predicate in (P.partOf, P.locatedIn):
        assert set(graph.objects(predicate, RDF.type)) == {OWL.ObjectProperty}, predicate


@pytest.mark.slow
class TestTheBuiltGraph:
    @pytest.fixture(scope="class")
    def text(self, built_demo: BuiltProject) -> str:
        return (built_demo.build_dir / GRAPH_VIEW).read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def graph(self, text: str) -> Graph:
        return _graph(text)

    def test_the_build_writes_the_view_under_its_header(
        self, built_demo: BuiltProject, text: str
    ) -> None:
        digest = hashlib.sha256(built_demo.facility_raw).hexdigest()

        assert text.split("\n", 1)[0] == HEADER_PREFIX + digest

    def test_the_predicates_are_exactly_the_kept_and_new_ones(self, graph: Graph) -> None:
        assert _narad_predicates(graph) == PREDICATES | DEMO_RELATION_PREDICATES

    def test_every_setpoint_and_readback_with_a_signal_has_one_signal_edge(
        self, built_demo: BuiltProject, graph: Graph
    ) -> None:
        facility = built_demo.facility
        code = facility["identity"]["code"]
        edges = Counter(
            subject
            for predicate in (P.readsSignal, P.writesSignal)
            for subject, _ in graph.subject_objects(predicate)
        )
        signalled = [
            channel
            for channel in facility["channels"]
            if channel.get("signal") and channel.get("role", "readback") != "none"
        ]

        assert signalled
        for channel in signalled:
            node = URIRef(iri(code, "channel", channel["id"]))
            predicate = P.writesSignal if channel.get("role") == "setpoint" else P.readsSignal
            assert edges[node] == 1, channel["id"]
            assert list(graph.objects(node, predicate)) == [SEM[channel["signal"]]]
        assert sum(edges.values()) == len(signalled)

    def test_the_septum_waveform_is_a_binding_of_a_septum(
        self, built_demo: BuiltProject, graph: Graph
    ) -> None:
        from rdflib import OWL

        code = built_demo.facility["identity"]["code"]
        wave = URIRef(iri(code, "channel", "SR:MAG:SEPTUM:01:CURRENT:WAVEFORM"))
        sek = URIRef(iri(code, "device", "SR/SEPTUM01"))

        assert set(graph.objects(wave, RDF.type)) == {SEM.ChannelBinding}
        assert list(graph.objects(wave, P.fullPv)) == [Literal("SR:MAG:SEPTUM:01:CURRENT:WAVEFORM")]
        assert list(graph.objects(wave, P.readsSignal)) == [SEM.pulse_waveform]
        assert "500 samples" in str(next(graph.objects(wave, P.description)))
        assert (sek, P.hasBinding, wave) in graph
        assert (sek, RDF.type, SEM.Septum) in graph
        assert (SEM.Septum, RDF.type, OWL.Class) in graph

    def test_every_device_carries_the_facility_code(
        self, built_demo: BuiltProject, graph: Graph
    ) -> None:
        facility = built_demo.facility
        code = facility["identity"]["code"]

        assert set(graph.subjects(P.facility, Literal(code))) == {
            URIRef(iri(code, "device", device["id"])) for device in facility["devices"]
        }

    def test_every_place_but_a_top_one_is_part_of_its_parent(
        self, built_demo: BuiltProject, graph: Graph
    ) -> None:
        expected = expected_part_of(built_demo.facility)

        assert expected
        assert {
            (str(subject), str(obj)) for subject, obj in graph.subject_objects(P.partOf)
        } == expected

    def test_every_placed_device_is_located_in_its_own_place(
        self, built_demo: BuiltProject, graph: Graph
    ) -> None:
        expected = expected_located_in(built_demo.facility)
        located = Counter(str(subject) for subject, _ in graph.subject_objects(P.locatedIn))

        assert expected
        assert {
            (str(subject), str(obj)) for subject, obj in graph.subject_objects(P.locatedIn)
        } == expected
        assert set(located.values()) == {1}, "a device is located in more than one place"

    def test_each_claim_is_one_edge_under_its_standing(
        self, built_demo: BuiltProject, graph: Graph
    ) -> None:
        facility = built_demo.facility
        code = facility["identity"]["code"]
        claims = facility["relations"]
        symmetric = {name for name, row in relation_types().items() if row.get("symmetric")}
        standings = Counter(claim["standing"] for claim in claims)

        assert len(claims) == DEMO_CLAIMS
        assert standings == {"accepted": DEMO_ACCEPTED, "contested": DEMO_CONTESTED}
        for claim in claims:
            subject_end = end_identity(claim["subject"])
            object_end = end_identity(claim["object"])
            assert subject_end is not None and object_end is not None
            subject = URIRef(iri(code, *subject_end))
            obj = URIRef(iri(code, *object_end))
            name = relation_predicate(claim["type"])
            plain, contested = P[name], P[name + "Contested"]
            edge, other = (
                (plain, contested)
                if claim["standing"] == "accepted"
                else (
                    contested,
                    plain,
                )
            )
            ends = (
                {(subject, obj), (obj, subject)} if claim["type"] in symmetric else {(subject, obj)}
            )
            assert any((s, edge, o) in graph for s, o in ends), claim["id"]
            assert not any((s, other, o) in graph for s, o in ends), claim["id"]

    def test_the_contested_claims_place_the_power_supply_in_two_racks(
        self, built_demo: BuiltProject, graph: Graph
    ) -> None:
        code = built_demo.facility["identity"]["code"]

        assert set(graph.subject_objects(P.locatedInContested)) == {
            (URIRef(iri(code, "device", "SR/QF05-PS")), URIRef(iri(code, "place", rack)))
            for rack in ("HALL/ROW3/RK07", "HALL/ROW3/RK08")
        }

    def test_every_claim_is_one_claim_node(self, built_demo: BuiltProject, graph: Graph) -> None:
        nodes = set(graph.subjects(RDF.type, SEM.Claim))

        assert len(nodes) == DEMO_CLAIMS
        assert {str(next(graph.objects(node, P.claimId))) for node in nodes} == {
            claim["id"] for claim in built_demo.facility["relations"]
        }

    def test_the_bounded_group_is_one_group_node_with_its_members(
        self, built_demo: BuiltProject, graph: Graph
    ) -> None:
        code = built_demo.facility["identity"]["code"]
        group = URIRef(iri(code, "group", "SR/VS03"))

        assert set(graph.subjects(RDF.type, SEM.Group)) == {group}
        assert list(graph.objects(group, P.groupId)) == [Literal("SR/VS03")]
        assert set(graph.objects(group, P.hasMember)) == {
            URIRef(iri(code, "device", member))
            for member in ("SR/ION-PUMP13", "SR/BPM13", "SR/QF05")
        }

    def test_two_independent_builds_render_equal_bytes(self) -> None:
        from osprey.facility.build import build_facility
        from tests._preset_data import packaged_facility_dir

        # Two builds and two read models: graph_text memoizes on the records
        # object, so one read model rendered twice would compare a string to itself.
        first, second = (
            graph_text(
                FacilityRecords(build_facility(packaged_facility_dir("demo"), project_name="demo"))
            )
            for _ in range(2)
        )

        assert first is not second
        assert first.encode("utf-8") == second.encode("utf-8")
