"""The graph view: the facility file as the knowledge graph's Turtle.

Written to ``<render>/data/graph/facility.ttl``. The first line is
``# osprey:facility-sha256 <hex>``, the sha256 of the facility file's bytes, so
the file changes whenever the facility file or this writer does. Every node's
IRI comes from :mod:`osprey.facility.views.graph_iri`.

Nodes and their ``narad_p:`` predicates:

- a place: ``placePath`` (its id) and ``sectionCode`` (the last segment of its
  id);
- a device, typed by its class: ``deviceId``, ``facility`` (the identity
  ``code``), ``rawType`` (the class name), ``system`` (the top place of its
  place path), ``placePath`` and ``sectionCode`` of its place, ``sourceName``
  (its first name), ``familyDescription`` (the description of the smallest
  described group naming it), ``systemDescription`` (the top place's
  description), ``sPositionM``, ``lengthM``, ``ordinalInPlace`` and
  ``ordinalInModel`` (each only when the facility file carries it), and one
  ``hasBinding`` per channel it is ``on`` or an ``endpoint_of``;
- a channel, a ``narad_sem:ChannelBinding``: ``bindingId`` and ``fullPv`` (its
  address), ``description``, and one ``readsSignal`` (a readback) or
  ``writesSignal`` (a setpoint) to ``narad_sem:<signal>`` when it names a
  signal.

A place carries ``hasBinding`` to the channels that are ``on`` it. Each signal
named is a ``narad_sem:SemanticSignal`` labelled with its name; each device
class and every class above it is an ``owl:Class`` under its parent, with its
aliases as ``skos:altLabel``. Subjects follow the facility file's order,
predicates are sorted with ``rdf:type`` first and objects are sorted, so equal
facility files give equal bytes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from osprey.facility import PN_LOCAL
from osprey.facility.build import FacilityDocument
from osprey.facility.views import ViewInputs
from osprey.facility.views.graph_iri import iri

__all__ = [
    "GRAPH_FILE",
    "HEADER_PREFIX",
    "NARAD_P",
    "NARAD_SEM",
    "PREFIXES",
    "PROPERTY_NAMES",
    "graph_text",
    "write_graph_view",
]

GRAPH_FILE = "facility.ttl"

#: The first line's prefix; the facility file's sha256 hex digest follows it.
HEADER_PREFIX = "# osprey:facility-sha256 "

NARAD_P = "https://narad.example.org/property/"
NARAD_SEM = "https://narad.example.org/schema/shared_semantics/"
OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
SKOS = "http://www.w3.org/2004/02/skos/core#"
XSD = "http://www.w3.org/2001/XMLSchema#"

#: The prefixes the file declares, in the order it declares them.
PREFIXES: dict[str, str] = {
    "narad_p": NARAD_P,
    "narad_sem": NARAD_SEM,
    "owl": OWL,
    "rdfs": RDFS,
    "skos": SKOS,
    "xsd": XSD,
}

P_BINDING_ID = "bindingId"
P_DESCRIPTION = "description"
P_DEVICE_ID = "deviceId"
P_FACILITY = "facility"
P_FAMILY_DESCRIPTION = "familyDescription"
P_FULL_PV = "fullPv"
P_HAS_BINDING = "hasBinding"
P_LENGTH_M = "lengthM"
P_ORDINAL_IN_MODEL = "ordinalInModel"
P_ORDINAL_IN_PLACE = "ordinalInPlace"
P_PLACE_PATH = "placePath"
P_RAW_TYPE = "rawType"
P_READS_SIGNAL = "readsSignal"
P_S_POSITION_M = "sPositionM"
P_SECTION_CODE = "sectionCode"
P_SOURCE_NAME = "sourceName"
P_SYSTEM = "system"
P_SYSTEM_DESCRIPTION = "systemDescription"
P_WRITES_SIGNAL = "writesSignal"

#: Every ``narad_p:`` predicate the view writes, sorted; each is declared in
#: the file's vocabulary block.
PROPERTY_NAMES: tuple[str, ...] = tuple(
    sorted(
        (
            P_BINDING_ID,
            P_DESCRIPTION,
            P_DEVICE_ID,
            P_FACILITY,
            P_FAMILY_DESCRIPTION,
            P_FULL_PV,
            P_HAS_BINDING,
            P_LENGTH_M,
            P_ORDINAL_IN_MODEL,
            P_ORDINAL_IN_PLACE,
            P_PLACE_PATH,
            P_RAW_TYPE,
            P_READS_SIGNAL,
            P_S_POSITION_M,
            P_SECTION_CODE,
            P_SOURCE_NAME,
            P_SYSTEM,
            P_SYSTEM_DESCRIPTION,
            P_WRITES_SIGNAL,
        )
    )
)

#: The predicates whose objects are nodes.
OBJECT_PROPERTY_NAMES: frozenset[str] = frozenset({P_HAS_BINDING, P_READS_SIGNAL, P_WRITES_SIGNAL})

#: A channel role -> the predicate joining it to its signal.
_SIGNAL_PREDICATE: dict[str, str] = {"readback": P_READS_SIGNAL, "setpoint": P_WRITES_SIGNAL}

RDF_TYPE = f"{RDF}type"
CHANNEL_BINDING = f"{NARAD_SEM}ChannelBinding"
SEMANTIC_SIGNAL = f"{NARAD_SEM}SemanticSignal"

_STRING_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}

#: An object: ``(kind, lexical form)`` with kind ``iri``, ``str``, ``int`` or ``dec``.
_Term = tuple[str, str]
_Subjects = dict[str, dict[str, set[_Term]]]


def _p(name: str) -> str:
    return f"{NARAD_P}{name}"


def _iri(value: str) -> _Term:
    return ("iri", value)


def _text(value: Any) -> _Term:
    return ("str", str(value))


def _integer(value: Any) -> _Term:
    return ("int", str(int(value)))


def _decimal(value: Any) -> _Term:
    lexical = repr(float(value))
    if "e" in lexical or "E" in lexical:
        lexical = format(Decimal(lexical), "f")
    if "." not in lexical:
        lexical = f"{lexical}.0"
    return ("dec", lexical)


def _add(subjects: _Subjects, subject: str, predicate: str, term: _Term) -> None:
    subjects.setdefault(subject, {}).setdefault(predicate, set()).add(term)


def _top(place_id: str) -> str:
    return place_id.split("/", 1)[0]


def _last_segment(place_id: str) -> str:
    return place_id.rsplit("/", 1)[-1]


def _family_descriptions(groups: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Each device's family description: that of the smallest described group naming it.

    Ties between groups of one size go to the earlier group in the facility file.
    """
    chosen: dict[str, tuple[int, int, str]] = {}
    for position, group in enumerate(groups):
        description = group.get("description")
        if not description:
            continue
        members = group.get("members") or []
        rank = (len(members), position, str(description))
        for member in members:
            if member not in chosen or rank < chosen[member]:
                chosen[member] = rank
    return {member: rank[2] for member, rank in chosen.items()}


def _class_rows(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every class a device may carry: the vocabulary's and the facility-added ones."""
    from osprey.facility.validate import vocabulary

    rows: dict[str, dict[str, Any]] = {
        row["name"]: {
            "iri": row.get("iri") or f"{NARAD_SEM}{row['name']}",
            "parent": row.get("parent"),
            "aliases": list(row.get("aliases") or []),
        }
        for row in vocabulary()["classes"]
    }
    for row in doc.get("classes") or []:
        name = str(row["class"])
        rows[name] = {
            "iri": f"{NARAD_SEM}{name}",
            "parent": row.get("parent"),
            "aliases": list(row.get("aliases") or []),
        }
    return rows


def _place_triples(subjects: _Subjects, code: str, places: Iterable[Mapping[str, Any]]) -> None:
    for place in places:
        place_id = str(place["id"])
        subject = iri(code, "place", place_id)
        _add(subjects, subject, _p(P_PLACE_PATH), _text(place_id))
        _add(subjects, subject, _p(P_SECTION_CODE), _text(_last_segment(place_id)))


def _device_triples(
    subjects: _Subjects,
    code: str,
    doc: Mapping[str, Any],
    classes: Mapping[str, Mapping[str, Any]],
    places: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Add every device's triples; return the classes the devices carry."""
    families = _family_descriptions(doc.get("groups") or [])
    used: set[str] = set()
    for device in doc.get("devices") or []:
        device_id = str(device["id"])
        subject = iri(code, "device", device_id)
        _add(subjects, subject, _p(P_DEVICE_ID), _text(device_id))
        _add(subjects, subject, _p(P_FACILITY), _text(code))
        klass = device.get("class")
        if klass:
            used.add(str(klass))
            row = classes.get(str(klass))
            _add(subjects, subject, RDF_TYPE, _iri(row["iri"] if row else f"{NARAD_SEM}{klass}"))
            _add(subjects, subject, _p(P_RAW_TYPE), _text(klass))
        place = device.get("place")
        if place:
            top = _top(str(place))
            _add(subjects, subject, _p(P_SYSTEM), _text(top))
            _add(subjects, subject, _p(P_PLACE_PATH), _text(place))
            _add(subjects, subject, _p(P_SECTION_CODE), _text(_last_segment(str(place))))
            top_description = places.get(top, {}).get("description")
            if top_description:
                _add(subjects, subject, _p(P_SYSTEM_DESCRIPTION), _text(top_description))
        names = device.get("names") or []
        if names:
            _add(subjects, subject, _p(P_SOURCE_NAME), _text(names[0]))
        if device_id in families:
            _add(subjects, subject, _p(P_FAMILY_DESCRIPTION), _text(families[device_id]))
        if "s" in device:
            _add(subjects, subject, _p(P_S_POSITION_M), _decimal(device["s"]))
        if "length" in device:
            _add(subjects, subject, _p(P_LENGTH_M), _decimal(device["length"]))
        if "ordinalInPlace" in device:
            _add(subjects, subject, _p(P_ORDINAL_IN_PLACE), _integer(device["ordinalInPlace"]))
        if "ordinalInModel" in device:
            _add(subjects, subject, _p(P_ORDINAL_IN_MODEL), _integer(device["ordinalInModel"]))
    return used


def _channel_triples(subjects: _Subjects, code: str, doc: Mapping[str, Any]) -> set[str]:
    """Add every channel's triples and its owners' edges; return the signals named."""
    signals: set[str] = set()
    for channel in doc.get("channels") or []:
        address = str(channel["id"])
        subject = iri(code, "channel", address)
        _add(subjects, subject, RDF_TYPE, _iri(CHANNEL_BINDING))
        _add(subjects, subject, _p(P_BINDING_ID), _text(address))
        _add(subjects, subject, _p(P_FULL_PV), _text(address))
        if channel.get("description"):
            _add(subjects, subject, _p(P_DESCRIPTION), _text(channel["description"]))
        predicate = _SIGNAL_PREDICATE.get(str(channel.get("role", "readback")))
        signal = channel.get("signal")
        if predicate and signal:
            signals.add(str(signal))
            _add(subjects, subject, _p(predicate), _iri(f"{NARAD_SEM}{signal}"))
        owner = channel.get("on") or {}
        owners = [iri(code, "device", str(device)) for device in channel.get("endpoint_of") or []]
        if owner.get("device"):
            owners.append(iri(code, "device", str(owner["device"])))
        if owner.get("place"):
            owners.append(iri(code, "place", str(owner["place"])))
        for node in owners:
            _add(subjects, node, _p(P_HAS_BINDING), _iri(subject))
    return signals


def _class_triples(
    subjects: _Subjects, used: Iterable[str], classes: Mapping[str, Mapping[str, Any]]
) -> None:
    """Declare each class the devices carry and every class above it."""
    declared: set[str] = set()
    pending = sorted(used)
    while pending:
        name = pending.pop()
        row = classes.get(name)
        if name in declared or row is None:
            continue
        declared.add(name)
        if row["parent"]:
            pending.append(str(row["parent"]))
    for name in sorted(declared):
        row = classes[name]
        subject = row["iri"]
        _add(subjects, subject, RDF_TYPE, _iri(f"{OWL}Class"))
        parent = classes.get(str(row["parent"])) if row["parent"] else None
        if parent is not None:
            _add(subjects, subject, f"{RDFS}subClassOf", _iri(parent["iri"]))
        for alias in row["aliases"]:
            _add(subjects, subject, f"{SKOS}altLabel", _text(alias))


def _vocabulary_triples(subjects: _Subjects) -> None:
    _add(subjects, SEMANTIC_SIGNAL, RDF_TYPE, _iri(f"{OWL}Class"))
    _add(subjects, CHANNEL_BINDING, RDF_TYPE, _iri(f"{OWL}Class"))
    _add(subjects, CHANNEL_BINDING, f"{RDFS}subClassOf", _iri(f"{OWL}Thing"))
    for name in PROPERTY_NAMES:
        kind = "ObjectProperty" if name in OBJECT_PROPERTY_NAMES else "DatatypeProperty"
        _add(subjects, _p(name), RDF_TYPE, _iri(f"{OWL}{kind}"))


def _render_iri(value: str) -> str:
    for prefix, namespace in PREFIXES.items():
        if value.startswith(namespace) and PN_LOCAL.fullmatch(value[len(namespace) :]):
            return f"{prefix}:{value[len(namespace) :]}"
    return f"<{value}>"


def _render_term(term: _Term) -> str:
    kind, lexical = term
    if kind == "iri":
        return _render_iri(lexical)
    if kind == "str":
        return '"' + "".join(_STRING_ESCAPES.get(char, char) for char in lexical) + '"'
    return lexical


def _term_order(term: _Term) -> tuple[str, Decimal | str]:
    kind, lexical = term
    return (kind, Decimal(lexical) if kind in ("int", "dec") else lexical)


def _subject_block(subject: str, predicates: Mapping[str, set[_Term]]) -> list[str]:
    ordered = [RDF_TYPE] if RDF_TYPE in predicates else []
    ordered.extend(sorted(name for name in predicates if name != RDF_TYPE))
    clauses = []
    for predicate in ordered:
        verb = "a" if predicate == RDF_TYPE else _render_iri(predicate)
        objects = ",\n        ".join(
            _render_term(term) for term in sorted(predicates[predicate], key=_term_order)
        )
        clauses.append(f"{verb} {objects}")
    lines = [f"{_render_iri(subject)} {clauses[0]}"]
    lines.extend(f"    {clause}" for clause in clauses[1:])
    return [line + " ;" for line in lines[:-1]] + [lines[-1] + " ."]


def graph_text(doc: FacilityDocument) -> str:
    """The graph view's Turtle for one facility file.

    Args:
        doc: The facility file as ``build_facility`` returned it.

    Returns:
        The Turtle text, its first line the facility file's sha256 header,
        ending in one newline.
    """
    from osprey.facility.render import facility_bytes

    code = str(doc["identity"]["code"])
    places = {str(place["id"]): place for place in doc.get("places") or []}
    classes = _class_rows(doc)

    subjects: _Subjects = {}
    _place_triples(subjects, code, doc.get("places") or [])
    used = _device_triples(subjects, code, doc, classes, places)
    signals = _channel_triples(subjects, code, doc)
    for signal in sorted(signals):
        subject = f"{NARAD_SEM}{signal}"
        _add(subjects, subject, RDF_TYPE, _iri(SEMANTIC_SIGNAL))
        _add(subjects, subject, f"{RDFS}label", _text(signal))
    _class_triples(subjects, used, classes)
    _vocabulary_triples(subjects)

    digest = hashlib.sha256(facility_bytes(doc)).hexdigest()
    lines = [f"{HEADER_PREFIX}{digest}"]
    lines.extend(f"@prefix {prefix}: <{namespace}> ." for prefix, namespace in PREFIXES.items())
    for subject, predicates in subjects.items():
        lines.append("")
        lines.extend(_subject_block(subject, predicates))
    return "\n".join(lines) + "\n"


def write_graph_view(root: Path, inputs: ViewInputs) -> list[Path]:
    """Write the graph view into ``root``.

    Args:
        root: The render's ``data/graph`` directory.
        inputs: The render's view inputs.

    Returns:
        The file written.
    """
    root.mkdir(parents=True, exist_ok=True)
    target = root / GRAPH_FILE
    target.write_bytes(graph_text(inputs.doc).encode("utf-8"))
    return [target]
