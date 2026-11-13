"""The Knowledge panel's graph view: the facility graph as nodes and edges.

``osprey build`` writes the facility graph as Turtle
(:mod:`osprey.facility.views.graph`); this module reads that file and flattens
it to the one payload the panel's Graph view draws. Nothing here talks to the
graph store: the file is the corpus the store is seeded from, so the view
works whether or not the store runs.

A node is a place, a device, a channel, a signal or a class. An edge is any
triple whose subject and object are both nodes, named by its predicate's local
name (``rdf:type`` is ``type``): a device is joined to its place by
``locatedIn`` and a place to its parent by ``partOf``, as the graph holds them.
A record's kind is read from its IRI's namespace when its types do not settle
it, so a device without a class is still a device.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.facility.views.graph import CHANNEL_BINDING, SEMANTIC_SIGNAL
from osprey.facility.views.graph_iri import decode, namespace

__all__ = [
    "NODE_KINDS",
    "attach_pages",
    "graph_payload",
    "page_references",
    "read_graph",
]

#: The node kinds of the payload, in the order the view lists them.
NODE_KINDS: tuple[str, ...] = ("place", "device", "channel", "signal", "class")

#: The node kinds that are records of the facility file.
RECORD_KINDS: frozenset[str] = frozenset({"place", "device", "channel"})

_OWL = "http://www.w3.org/2002/07/owl#"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_SCHEMA_TYPES = frozenset({f"{_OWL}DatatypeProperty", f"{_OWL}ObjectProperty"})


def _local(iri: str) -> str:
    """The part of an IRI after its last ``/`` or ``#``."""
    return iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def read_graph(ttl_path: Path) -> dict[str, Any]:
    """Read the facility graph at ``ttl_path`` as the Graph view's payload.

    Args:
        ttl_path: The Turtle file ``osprey build`` wrote.

    Returns:
        The payload of :func:`graph_payload`.

    Raises:
        OSError: The file cannot be read.
        Exception: rdflib refuses the file as Turtle.
    """
    import rdflib

    graph = rdflib.Graph()
    graph.parse(ttl_path, format="turtle")
    return graph_payload(graph)


def graph_payload(graph: Any) -> dict[str, Any]:
    """Flatten an RDF graph in the facility-graph shape to nodes and edges.

    Args:
        graph: An :class:`rdflib.Graph` holding the facility graph.

    Returns:
        ``{"nodes": [...], "edges": [...]}``. A node is
        ``{id, kind, label, type, props}``: ``id`` is the IRI's local name,
        ``kind`` one of :data:`NODE_KINDS`, ``type`` a device's class name
        (else ``None``), ``props`` its literals by predicate local name. A
        place, device or channel also carries ``record``, its id in the
        facility file. An edge is ``[source, target, relation]`` with the two
        nodes by index. Nodes are ordered by IRI
        and edges sorted, so one file always gives one payload.
    """
    import rdflib
    from rdflib import OWL, RDF

    channel_type = rdflib.URIRef(CHANNEL_BINDING)
    signal_type = rdflib.URIRef(SEMANTIC_SIGNAL)
    device_namespace = namespace("device")
    place_namespace = namespace("place")
    classes = set(graph.subjects(RDF.type, OWL.Class))
    schema = {
        subject
        for schema_type in _SCHEMA_TYPES
        for subject in graph.subjects(RDF.type, rdflib.URIRef(schema_type))
    }

    nodes: list[dict[str, Any]] = []
    index: dict[Any, int] = {}
    for subject in sorted(set(graph.subjects()), key=str):
        if subject in schema or not isinstance(subject, rdflib.URIRef):
            continue
        types = set(graph.objects(subject, RDF.type))
        props: dict[str, str] = {}
        for predicate, value in sorted(graph.predicate_objects(subject), key=str):
            if isinstance(value, rdflib.Literal):
                key = _local(str(predicate))
                props[key] = f"{props[key]}, {value}" if key in props else str(value)
        name = _local(str(subject))
        device_classes = sorted(types & classes, key=str)
        node_type: str | None = None
        if subject in classes:
            kind, label = "class", props.get("label", name)
        elif channel_type in types:
            kind, label = "channel", props.get("fullPv", name)
        elif signal_type in types:
            kind, label = "signal", props.get("label", name)
        elif device_classes:
            kind, label = "device", props.get("deviceId", name)
            node_type = _local(str(device_classes[0]))
        elif str(subject).startswith(device_namespace):
            kind, label = "device", props.get("deviceId", name)
        elif str(subject).startswith(place_namespace):
            kind, label = "place", props.get("placePath", name)
        else:
            continue
        index[subject] = len(nodes)
        nodes.append({"id": name, "kind": kind, "label": label, "type": node_type, "props": props})

    _attach_record_ids(nodes)

    edges: set[tuple[int, int, str]] = set()
    for subject, predicate, value in graph:
        if subject in index and value in index:
            relation = "type" if str(predicate) == _RDF_TYPE else _local(str(predicate))
            edges.add((index[subject], index[value], relation))

    return {"nodes": nodes, "edges": [list(edge) for edge in sorted(edges)]}


def _attach_record_ids(nodes: list[dict[str, Any]]) -> None:
    """Give each place, device and channel node its facility-file id as ``record``.

    The graph's IRIs are minted from the file's identity code, which every
    device carries as its ``facility`` literal. A graph with no device, or an
    IRI this code did not mint, leaves the node without ``record``.
    """
    code = next(
        (node["props"]["facility"] for node in nodes if "facility" in node["props"]),
        None,
    )
    if code is None:
        return
    for node in nodes:
        if node["kind"] not in RECORD_KINDS:
            continue
        try:
            node["record"] = decode(node["id"], code)
        except ValueError:
            continue


def page_references(bundle: Any, records: Any) -> dict[tuple[str, str], list[str]]:
    """Which knowledge pages name which facility records.

    Args:
        bundle: The served :class:`OKFBundle`.
        records: The render's :class:`FacilityRecords`.

    Returns:
        ``{(kind, id): [concept id, ...]}`` for every record a page's facility
        link resolves to, the concept ids sorted. A page that cannot be read
        names nothing.
    """
    from osprey.facility.references import find_facility_links

    pages: dict[tuple[str, str], set[str]] = {}
    for entry in bundle.list_concepts():
        try:
            body = bundle.read_concept(entry.concept_id).body or ""
        except Exception:  # an unreadable page names nothing; the reader reports it.
            continue
        for link in find_facility_links(body, records):
            for record in records.describe(link.target) or ():
                pages.setdefault((str(record["kind"]), str(record["id"])), set()).add(
                    entry.concept_id
                )
    return {key: sorted(found) for key, found in pages.items()}


def attach_pages(
    payload: dict[str, Any], pages: Mapping[tuple[str, str], Iterable[str]]
) -> dict[str, Any]:
    """Add ``pages`` to each node a knowledge page names.

    Args:
        payload: A payload of :func:`graph_payload`; changed in place.
        pages: The result of :func:`page_references`.

    Returns:
        ``payload``. A node no page names gets no ``pages`` key.
    """
    for node in payload["nodes"]:
        named = pages.get((node["kind"], node.get("record", "")))
        if named:
            node["pages"] = list(named)
    return payload
