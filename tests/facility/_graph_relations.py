"""What the graph view should say about places, devices and their names, from the records.

Each expectation is derived from the facility records by the rule the view
states, never from the view's own writer, so a test comparing the two catches
a writer that drifts from the rule. Every IRI is returned as a plain string, so
a Cypher ``n.uri`` result compares directly.

- :func:`expected_part_of` — each place whose id has a ``/`` is ``partOf`` the
  place whose id is its own without the last segment;
- :func:`expected_located_in` — each device with a place is ``locatedIn`` that
  place;
- :func:`expected_name_holders` — the places, devices and channels one of whose
  ``names`` equals a name, compared lower-cased as whole entries;
- :func:`strip` — a graph without what the view added beside the base writer:
  ``partOf``, ``locatedIn``, their two ``owl:ObjectProperty`` declarations and
  the ``skos:altLabel`` of places, devices and channels (a class keeps its
  aliases).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from osprey.facility.records import FacilityRecords
from osprey.facility.views.graph_iri import iri, namespace

if TYPE_CHECKING:
    from rdflib import Graph

NARAD_P = "https://narad.example.org/property/"
PART_OF = f"{NARAD_P}partOf"
LOCATED_IN = f"{NARAD_P}locatedIn"
SKOS_ALT_LABEL = "http://www.w3.org/2004/02/skos/core#altLabel"

#: The kinds whose ``names`` the view writes as ``skos:altLabel``.
NAMED_KINDS = ("place", "device", "channel")


def _records(records: FacilityRecords | Mapping[str, Any]) -> FacilityRecords:
    return records if isinstance(records, FacilityRecords) else FacilityRecords(records)


def _code(records: FacilityRecords) -> str:
    return str(records.document["identity"]["code"])


def expected_part_of(records: FacilityRecords | Mapping[str, Any]) -> frozenset[tuple[str, str]]:
    """Each ``(place, parent place)`` IRI pair the view should join by ``partOf``."""
    read = _records(records)
    code = _code(read)
    return frozenset(
        (iri(code, "place", place_id), iri(code, "place", place_id.rsplit("/", 1)[0]))
        for place_id in (str(place["id"]) for place in read.of_kind("place"))
        if "/" in place_id
    )


def expected_located_in(
    records: FacilityRecords | Mapping[str, Any],
) -> frozenset[tuple[str, str]]:
    """Each ``(device, its place)`` IRI pair the view should join by ``locatedIn``."""
    read = _records(records)
    code = _code(read)
    return frozenset(
        (iri(code, "device", str(device["id"])), iri(code, "place", str(device["place"])))
        for device in read.of_kind("device")
        if device.get("place")
    )


def expected_name_holders(
    records: FacilityRecords | Mapping[str, Any], name: str
) -> frozenset[str]:
    """The IRIs of the places, devices and channels with ``name`` among their ``names``.

    A name matches an entry when the two are equal lower-cased, as whole
    entries: no substring and no token match.
    """
    read = _records(records)
    code = _code(read)
    wanted = name.lower()
    return frozenset(
        iri(code, kind, str(record["id"]))
        for kind in NAMED_KINDS
        for record in read.of_kind(kind)
        if any(str(entry).lower() == wanted for entry in record.get("names") or ())
    )


def strip(graph: Graph | str) -> Graph:
    """A copy of ``graph`` (or of the Turtle text) without what the base writer lacked."""
    from rdflib import OWL, RDF, Graph, URIRef

    source = graph if isinstance(graph, Graph) else Graph().parse(data=graph, format="turtle")
    record_namespaces = tuple(namespace(kind) for kind in NAMED_KINDS)
    added = {URIRef(PART_OF), URIRef(LOCATED_IN)}
    alt_label = URIRef(SKOS_ALT_LABEL)
    stripped = Graph()
    for subject, predicate, obj in source:
        if predicate in added:
            continue
        if subject in added and predicate == RDF.type and obj == OWL.ObjectProperty:
            continue
        if predicate == alt_label and str(subject).startswith(record_namespaces):
            continue
        stripped.add((subject, predicate, obj))
    return stripped
