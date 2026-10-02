"""The graph view: the facility file written as the knowledge graph's Turtle.

``data/graph/facility.ttl`` opens with the facility file's sha256, types every
channel a ``narad_sem:ChannelBinding`` joined to its signal by its role, and
carries exactly the ``narad_p:`` predicates of :data:`PREDICATES`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rdflib import RDF, Graph, Namespace, URIRef

from osprey.facility.render import facility_bytes
from osprey.facility.views import ViewInputs
from osprey.facility.views.graph import GRAPH_FILE, HEADER_PREFIX, graph_text, write_graph_view
from osprey.facility.views.graph_iri import iri

P = Namespace("https://narad.example.org/property/")
SEM = Namespace("https://narad.example.org/schema/shared_semantics/")

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
    "ringDescription",
    "ordinalInPlace",
    "ordinalInModel",
    "placePath",
    "lengthM",
}


def _doc() -> dict[str, Any]:
    return {
        "schema": "osprey.facility.facility/1",
        "identity": {"code": "fx"},
        "places": [
            {"id": "M", "level": "machine", "description": "The machine."},
            {"id": "M/S1", "level": "sector"},
            {"id": "M/S2", "level": "sector", "description": "Sector two."},
        ],
        "devices": [
            {
                "id": "M/BPM1",
                "class": "BeamPositionMonitor",
                "place": "M/S1",
                "names": ["BPM1", "BPM 1"],
                "model": "M",
                "s": 1.25,
                "length": 0.0,
                "ordinalInPlace": 1,
                "ordinalInModel": 1,
            },
            {"id": "M/Q1", "class": "Quadrupole", "place": "M/S2"},
        ],
        "channels": [
            {
                "id": "M:BPM1:X",
                "on": {"device": "M/BPM1"},
                "signal": "position_x_readback",
                "description": "Horizontal position.",
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
    first = graph_text(doc).split("\n", 1)[0]

    assert first == HEADER_PREFIX + hashlib.sha256(facility_bytes(doc)).hexdigest()


def test_equal_facility_files_give_equal_bytes() -> None:
    assert graph_text(_doc()) == graph_text(_doc())


def test_a_device_carries_its_identity_place_and_position() -> None:
    graph = _graph(graph_text(_doc()))
    device = URIRef(iri("fx", "device", "M/BPM1"))

    assert set(graph.objects(device, RDF.type)) == {SEM.BeamPositionMonitor}
    expected = {
        "deviceId": "M/BPM1",
        "facility": "fx",
        "rawType": "BeamPositionMonitor",
        "system": "M",
        "placePath": "M/S1",
        "sectionCode": "S1",
        "sourceName": "BPM1",
        "familyDescription": "Monitors.",
        "systemDescription": "The machine.",
        "ringDescription": "The machine.",
        "sPositionM": 1.25,
        "lengthM": 0.0,
        "ordinalInPlace": 1,
        "ordinalInModel": 1,
    }
    for name, value in expected.items():
        assert _value(graph, "device", "M/BPM1", name) == value, name


def test_the_place_description_is_the_device_place_before_the_one_above() -> None:
    graph = _graph(graph_text(_doc()))

    assert _value(graph, "device", "M/Q1", "ringDescription") == "Sector two."
    assert _value(graph, "device", "M/Q1", "systemDescription") == "The machine."
    assert _value(graph, "device", "M/Q1", "familyDescription") == "Diagnostics."


def test_a_device_without_s_has_no_position_or_ordinals() -> None:
    graph = _graph(graph_text(_doc()))

    for name in ("sPositionM", "lengthM", "ordinalInPlace", "ordinalInModel", "sourceName"):
        assert _value(graph, "device", "M/Q1", name) is None, name


def test_a_place_carries_its_path_and_last_segment() -> None:
    graph = _graph(graph_text(_doc()))

    assert _value(graph, "place", "M/S2", "placePath") == "M/S2"
    assert _value(graph, "place", "M/S2", "sectionCode") == "S2"
    assert _value(graph, "place", "M", "sectionCode") == "M"


def test_a_channel_is_a_binding_joined_to_its_signal_by_its_role() -> None:
    graph = _graph(graph_text(_doc()))

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
    graph = _graph(graph_text(_doc()))
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

    graph = _graph(graph_text(_doc()))

    assert (SEM.BeamPositionMonitor, RDF.type, OWL.Class) in graph
    parents = set(graph.transitive_objects(SEM.BeamPositionMonitor, RDFS.subClassOf))
    assert SEM.AcceleratorDevice in parents


def test_the_view_writes_one_file(tmp_path: Path) -> None:
    doc = _doc()
    inputs = ViewInputs(doc=doc, rendered_config={}, facility_dir=tmp_path, served=[])

    written = write_graph_view(tmp_path / "graph", inputs)

    assert written == [tmp_path / "graph" / GRAPH_FILE]
    assert written[0].read_text(encoding="utf-8") == graph_text(doc)
