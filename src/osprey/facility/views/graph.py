"""The graph view: the facility file as the knowledge graph's Turtle.

Written to ``<render>/data/graph/facility.ttl``. The first line is
``# osprey:facility-sha256 <hex>``, the sha256 of the facility file's bytes.
The file changes with the facility file, the ``claims`` and ``basis`` settings
and this writer. Every node's IRI comes from
:mod:`osprey.facility.views.graph_iri`.

Nodes and their ``narad_p:`` predicates:

- a place: ``placePath`` (its id), ``sectionCode`` (the last segment of its
  id) and, unless it is a top place, one ``partOf`` to the place whose id is
  its own without the last segment;
- a device, typed by its class when it has one: ``deviceId``, ``facility``
  (the identity ``code``), ``rawType`` (the class name, only with a class),
  ``sourceName`` (its id), ``familyDescription`` (the description of the
  smallest described group naming it), ``sPositionM``, ``lengthM``,
  ``ordinalInPlace`` and ``ordinalInModel`` (each only when the facility file
  carries it), and one ``hasBinding`` per channel it is ``on`` or an
  ``endpoint_of``; a device with a place also carries one ``locatedIn`` to
  that place, ``placePath`` and ``sectionCode`` of the place, ``system`` (the
  top place of its place path) and ``systemDescription`` (the top place's
  description), except that ``basis: stated`` drops the ``locatedIn`` of a
  place projected from ``located_in`` claims (``place_from: relation``) when
  no accepted claim from the device to that place is stated;
- a channel, a ``narad_sem:ChannelBinding``: ``bindingId`` and ``fullPv`` (its
  address), ``description``, and one ``readsSignal`` (a readback) or
  ``writesSignal`` (a setpoint) to ``narad_sem:<signal>`` when it names a
  signal.

A place carries ``hasBinding`` to the channels that are ``on`` it. A place,
device or channel carries one ``skos:altLabel`` per entry of its ``names``,
and a place or device one more per ``name`` of its ``scheme_names`` (the
scheme is not written; ``basis: stated`` drops the ``inferred`` ones); its
``label`` is not written. Each signal named is a
``narad_sem:SemanticSignal`` labelled with its name; each device class and
every class above it is an ``owl:Class`` under its parent, with its aliases as
``skos:altLabel``. The predicates whose objects are nodes (``hasBinding``,
``readsSignal``, ``writesSignal``, ``partOf``, ``locatedIn``) are declared
``owl:ObjectProperty``, the rest ``owl:DatatypeProperty``.

Each relation claim the ``basis`` setting keeps (``stated`` drops the
``inferred`` ones) is one edge from its subject end to its object end, each
end the device, place or group it names (ports and units do not change the
edge): an ``accepted`` claim under its type's predicate in
:data:`RELATION_PREDICATES`, a ``contested`` one under that predicate's
``Contested`` twin, a ``superseded`` one not at all. Equal edges are one edge,
so a ``located_in`` claim naming a device's own place adds nothing to its
``locatedIn``; a claim naming another place adds a second. A symmetric type's
edges are one per unordered pair of ends, in the direction of the
smallest-id claim stating it. Each relation predicate written, and only those,
is declared ``owl:ObjectProperty``. The ``claims`` setting does not change the
edges.

The ``claims`` setting chooses the claims that also get a claim node, a
``narad_sem:Claim`` (not declared ``owl:Class``) at the ``relation`` IRI of the
claim's id: ``all`` every claim the ``basis`` setting keeps, ``contested`` those
with a non-empty ``contests``, ``none`` no claim. A claim node carries
``claimSubject`` and ``claimObject`` to its ends' nodes, ``contests`` to each
rival that has a claim node too, and the literals ``claimId``,
``relationType``, ``basis`` (``stated`` when the claim has none), ``rule``,
``standing``, ``document``, ``documentVersion``, ``documentState``,
``locator`` (its ``source``), ``subjectPort``, ``objectPort``, ``subjectUnit``
and ``objectUnit``, each only when the claim has it. Each scalar attribute is
one literal (a bool as ``"true"`` or ``"false"``) under its key when the key
is an identifier the view does not otherwise write, else under
``attribute_`` and the key with every UTF-8 byte outside ``[A-Za-z0-9_]``
written ``_xHH_``. Each claim-node predicate written, and only those, is
declared.

Each group that is an end of a claim the ``basis`` setting keeps, whatever the
claim's standing and whatever the ``claims`` setting, is a ``narad_sem:Group``
(not declared ``owl:Class``): ``groupId`` (its id), one ``skos:altLabel`` per
entry of its ``names`` and one ``hasMember`` to each member device; its
``label`` and ``description`` are not written. ``groupId`` and ``hasMember``
are declared only when written, so a group no claim names is no node.

Subjects follow the facility file's order, predicates are sorted with
``rdf:type`` first and objects are sorted, so equal facility files give equal
bytes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

from osprey.facility import DEFAULT_ROLE, PN_LOCAL
from osprey.facility.records import FacilityRecords
from osprey.facility.views import GRAPH_BASIS_KEY, GRAPH_CLAIMS_KEY, ViewInputs
from osprey.facility.views.graph_iri import iri

if TYPE_CHECKING:
    from osprey.facility.errors import FacilityBuildError

__all__ = [
    "ATTRIBUTE_PREFIX",
    "CLAIM_PROPERTY_NAMES",
    "GRAPH_FILE",
    "HEADER_PREFIX",
    "NARAD_P",
    "NARAD_SEM",
    "PREFIXES",
    "PROPERTY_NAMES",
    "RELATION_PREDICATES",
    "RESERVED_NAMES",
    "GraphSettings",
    "graph_settings",
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
P_LOCATED_IN = "locatedIn"
P_ORDINAL_IN_MODEL = "ordinalInModel"
P_ORDINAL_IN_PLACE = "ordinalInPlace"
P_PART_OF = "partOf"
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
            P_LOCATED_IN,
            P_ORDINAL_IN_MODEL,
            P_ORDINAL_IN_PLACE,
            P_PART_OF,
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
OBJECT_PROPERTY_NAMES: frozenset[str] = frozenset(
    {P_HAS_BINDING, P_LOCATED_IN, P_PART_OF, P_READS_SIGNAL, P_WRITES_SIGNAL}
)

#: A relation type -> the ``narad_p:`` predicate of its accepted claims' edges.
#: A contested claim's edge takes the predicate's ``Contested`` twin.
RELATION_PREDICATES: Mapping[str, str] = MappingProxyType(
    {
        "part_of": P_PART_OF,
        "located_in": P_LOCATED_IN,
        "powered_by": "poweredBy",
        "cabled_to": "cabledTo",
        "read_out_by": "readOutBy",
        "controlled_by": "controlledBy",
        "triggered_by": "triggeredBy",
        "interlocked_by": "interlockedBy",
        "bounded_by": "boundedBy",
        "networked_to": "networkedTo",
        "supplied_by": "suppliedBy",
        "same_as": "sameAs",
        "runs_on": "runsOn",
    }
)

#: The suffix of a relation predicate's twin for contested claims.
CONTESTED_SUFFIX = "Contested"

P_BASIS = "basis"
P_CLAIM_ID = "claimId"
P_CLAIM_OBJECT = "claimObject"
P_CLAIM_SUBJECT = "claimSubject"
P_CONTESTS = "contests"
P_DOCUMENT = "document"
P_DOCUMENT_STATE = "documentState"
P_DOCUMENT_VERSION = "documentVersion"
P_LOCATOR = "locator"
P_OBJECT_PORT = "objectPort"
P_OBJECT_UNIT = "objectUnit"
P_RELATION_TYPE = "relationType"
P_RULE = "rule"
P_STANDING = "standing"
P_SUBJECT_PORT = "subjectPort"
P_SUBJECT_UNIT = "subjectUnit"

#: The ``narad_p:`` predicates of a claim node, sorted; each is declared only
#: when a claim node carries it.
CLAIM_PROPERTY_NAMES: tuple[str, ...] = tuple(
    sorted(
        (
            P_BASIS,
            P_CLAIM_ID,
            P_CLAIM_OBJECT,
            P_CLAIM_SUBJECT,
            P_CONTESTS,
            P_DOCUMENT,
            P_DOCUMENT_STATE,
            P_DOCUMENT_VERSION,
            P_LOCATOR,
            P_OBJECT_PORT,
            P_OBJECT_UNIT,
            P_RELATION_TYPE,
            P_RULE,
            P_STANDING,
            P_SUBJECT_PORT,
            P_SUBJECT_UNIT,
        )
    )
)

#: The claim-node predicates whose objects are nodes.
CLAIM_OBJECT_PROPERTY_NAMES: frozenset[str] = frozenset(
    {P_CLAIM_OBJECT, P_CLAIM_SUBJECT, P_CONTESTS}
)

P_GROUP_ID = "groupId"
P_HAS_MEMBER = "hasMember"

#: The ``narad_p:`` predicates of a group node; each is declared only when written.
GROUP_PROPERTY_NAMES: tuple[str, ...] = (P_GROUP_ID, P_HAS_MEMBER)

#: Every ``narad_p:`` local name the view writes other than an attribute's;
#: an attribute key equal to one is written under ``attribute_<key>``.
RESERVED_NAMES: frozenset[str] = frozenset(
    {
        *PROPERTY_NAMES,
        *RELATION_PREDICATES.values(),
        *(f"{name}{CONTESTED_SUFFIX}" for name in RELATION_PREDICATES.values()),
        *CLAIM_PROPERTY_NAMES,
        *GROUP_PROPERTY_NAMES,
    }
)

#: The prefix of an attribute's predicate when its key is not written as is.
ATTRIBUTE_PREFIX = "attribute_"

_IDENTIFIER_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")

#: A claim's ``source`` field -> the claim-node predicate carrying it.
_SOURCE_FIELDS: tuple[tuple[str, str], ...] = (
    (P_DOCUMENT, "document"),
    (P_DOCUMENT_VERSION, "version"),
    (P_DOCUMENT_STATE, "state"),
    (P_LOCATOR, "locator"),
)

#: A claim end -> its link, port and unit predicates.
_END_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("subject", P_CLAIM_SUBJECT, P_SUBJECT_PORT, P_SUBJECT_UNIT),
    ("object", P_CLAIM_OBJECT, P_OBJECT_PORT, P_OBJECT_UNIT),
)

#: A channel role -> the predicate joining it to its signal.
_SIGNAL_PREDICATE: dict[str, str] = {"readback": P_READS_SIGNAL, "setpoint": P_WRITES_SIGNAL}

RDF_TYPE = f"{RDF}type"
CHANNEL_BINDING = f"{NARAD_SEM}ChannelBinding"
SEMANTIC_SIGNAL = f"{NARAD_SEM}SemanticSignal"
#: A claim node's type; not declared ``owl:Class`` (the class set is the device taxonomy).
CLAIM = f"{NARAD_SEM}Claim"
#: A group node's type; not declared ``owl:Class``, for the same reason.
GROUP = f"{NARAD_SEM}Group"

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


def _names(
    subjects: _Subjects, subject: str, record: Mapping[str, Any], basis: str = "all"
) -> None:
    """Write each of a record's ``names`` and ``scheme_names`` as one ``skos:altLabel``.

    Of a scheme name only its ``name`` is written; under ``basis: stated`` an
    ``inferred`` one is not written at all.
    """
    for name in record.get("names") or []:
        _add(subjects, subject, f"{SKOS}altLabel", _text(name))
    for alias in record.get("scheme_names") or []:
        if not isinstance(alias, Mapping) or alias.get("name") is None:
            continue
        if basis == "stated" and alias.get("basis", "stated") == "inferred":
            continue
        _add(subjects, subject, f"{SKOS}altLabel", _text(alias["name"]))


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


def _class_rows(records: FacilityRecords) -> dict[str, dict[str, Any]]:
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
    for row in records.document.get("classes") or []:
        name = str(row["class"])
        rows[name] = {
            "iri": f"{NARAD_SEM}{name}",
            "parent": row.get("parent"),
            "aliases": list(row.get("aliases") or []),
        }
    return rows


def _place_triples(
    subjects: _Subjects, code: str, places: Iterable[Mapping[str, Any]], basis: str
) -> None:
    for place in places:
        place_id = str(place["id"])
        subject = iri(code, "place", place_id)
        _add(subjects, subject, _p(P_PLACE_PATH), _text(place_id))
        _add(subjects, subject, _p(P_SECTION_CODE), _text(_last_segment(place_id)))
        if "/" in place_id:
            _add(
                subjects,
                subject,
                _p(P_PART_OF),
                _iri(iri(code, "place", place_id.rsplit("/", 1)[0])),
            )
        _names(subjects, subject, place, basis)


def _inferred_projections(records: FacilityRecords) -> frozenset[str]:
    """The devices whose place was projected (``place_from: relation``) from inferred claims only.

    A projecting claim is an accepted ``located_in`` claim from the device to
    its place; a device keeps its place edge under ``basis: stated`` when any
    of them is stated.
    """
    from osprey.facility.relations import ACCEPTED, end_identity

    projected = {
        str(device["id"]): str(device["place"])
        for device in records.of_kind("device")
        if device.get("place") and (device.get("provenance") or {}).get("place_from") == "relation"
    }
    if not projected:
        return frozenset()
    stated: set[str] = set()
    for claim in records.document.get("relations") or []:
        if not isinstance(claim, Mapping) or claim.get("type") != "located_in":
            continue
        if claim.get("standing") != ACCEPTED or claim.get("basis", "stated") == "inferred":
            continue
        subject = end_identity(claim.get("subject") or {})
        obj = end_identity(claim.get("object") or {})
        if subject is None or obj is None or subject[0] != "device" or obj[0] != "place":
            continue
        if projected.get(subject[1]) == obj[1]:
            stated.add(subject[1])
    return frozenset(projected.keys() - stated)


def _device_triples(
    subjects: _Subjects,
    code: str,
    records: FacilityRecords,
    classes: Mapping[str, Mapping[str, Any]],
    basis: str,
) -> set[str]:
    """Add every device's triples; return the classes the devices carry."""
    families = _family_descriptions(records.of_kind("group"))
    unstated = _inferred_projections(records) if basis == "stated" else frozenset()
    used: set[str] = set()
    for device in records.of_kind("device"):
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
            if basis != "stated" or device_id not in unstated:
                _add(subjects, subject, _p(P_LOCATED_IN), _iri(iri(code, "place", str(place))))
            _add(subjects, subject, _p(P_SYSTEM), _text(top))
            _add(subjects, subject, _p(P_PLACE_PATH), _text(place))
            _add(subjects, subject, _p(P_SECTION_CODE), _text(_last_segment(str(place))))
            top_place = records.record("place", top)
            top_description = top_place.get("description") if top_place else None
            if top_description:
                _add(subjects, subject, _p(P_SYSTEM_DESCRIPTION), _text(top_description))
        _add(subjects, subject, _p(P_SOURCE_NAME), _text(device_id))
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
        _names(subjects, subject, device, basis)
    return used


def _channel_triples(subjects: _Subjects, code: str, records: FacilityRecords) -> set[str]:
    """Add every channel's triples and its owners' edges; return the signals named."""
    signals: set[str] = set()
    for channel in records.of_kind("channel"):
        address = str(channel["id"])
        subject = iri(code, "channel", address)
        _add(subjects, subject, RDF_TYPE, _iri(CHANNEL_BINDING))
        _add(subjects, subject, _p(P_BINDING_ID), _text(address))
        _add(subjects, subject, _p(P_FULL_PV), _text(address))
        if channel.get("description"):
            _add(subjects, subject, _p(P_DESCRIPTION), _text(channel["description"]))
        _names(subjects, subject, channel)
        predicate = _SIGNAL_PREDICATE.get(str(channel.get("role", DEFAULT_ROLE)))
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


def _declare(subjects: _Subjects, name: str, object_property: bool) -> None:
    """Declare a ``narad_p:`` predicate the view writes beyond the fixed vocabulary.

    A name of :data:`PROPERTY_NAMES` is left to :func:`_vocabulary_triples`,
    which always declares it, so its declaration keeps its place in the file.
    """
    if name in PROPERTY_NAMES:
        return
    kind = "ObjectProperty" if object_property else "DatatypeProperty"
    _add(subjects, _p(name), RDF_TYPE, _iri(f"{OWL}{kind}"))


def _written_claims(records: FacilityRecords, settings: GraphSettings) -> list[Mapping[str, Any]]:
    """The relation claims the view writes from: all, or the stated ones under ``basis: stated``."""
    claims = [
        claim for claim in records.document.get("relations") or [] if isinstance(claim, Mapping)
    ]
    if settings.basis == "stated":
        claims = [claim for claim in claims if claim.get("basis", "stated") != "inferred"]
    return claims


def _relation_triples(subjects: _Subjects, code: str, claims: Iterable[Mapping[str, Any]]) -> None:
    """Add one edge per distinct (subject end, predicate, object end) of the claims."""
    from osprey.facility.relations import ACCEPTED, CONTESTED, end_identity
    from osprey.facility.validate import relation_types

    types = relation_types()
    edges: set[tuple[str, str, str]] = set()
    # (predicate, unordered ends) -> (smallest claim id, subject, object).
    symmetric: dict[tuple[str, frozenset[str]], tuple[str, str, str]] = {}
    for claim in claims:
        relation_type = str(claim.get("type"))
        standing = claim.get("standing")
        if standing == ACCEPTED:
            name = RELATION_PREDICATES[relation_type]
        elif standing == CONTESTED:
            name = f"{RELATION_PREDICATES[relation_type]}{CONTESTED_SUFFIX}"
        else:
            continue
        ends = (end_identity(claim.get("subject") or {}), end_identity(claim.get("object") or {}))
        if ends[0] is None or ends[1] is None:
            continue
        subject = iri(code, *ends[0])
        obj = iri(code, *ends[1])
        if types[relation_type].get("symmetric"):
            key = (name, frozenset((subject, obj)))
            claim_id = str(claim["id"])
            if key not in symmetric or claim_id < symmetric[key][0]:
                symmetric[key] = (claim_id, subject, obj)
        else:
            edges.add((subject, name, obj))
    edges.update((subject, key[0], obj) for key, (_, subject, obj) in symmetric.items())
    for subject, name, obj in sorted(edges):
        _add(subjects, subject, _p(name), _iri(obj))
    for name in sorted({name for _, name, _ in edges}):
        _declare(subjects, name, object_property=True)


def _attribute_name(key: str) -> str:
    """The ``narad_p:`` local name an attribute key is written under.

    A key that is an identifier and not a name the view writes otherwise is
    its own name; any other key is ``attribute_`` plus the key, each UTF-8
    byte outside ``[A-Za-z0-9_]`` written as ``_xHH_``.
    """
    if PN_LOCAL.fullmatch(key) and key not in RESERVED_NAMES:
        return key
    escaped = "".join(
        chr(byte) if byte in _IDENTIFIER_BYTES else f"_x{byte:02X}_" for byte in key.encode("utf-8")
    )
    return f"{ATTRIBUTE_PREFIX}{escaped}"


def _attribute_term(value: Any) -> _Term | None:
    """An attribute's literal, or ``None`` for a value that is not a scalar."""
    if isinstance(value, bool):
        return _text("true" if value else "false")
    if isinstance(value, int):
        return _integer(value)
    if isinstance(value, float):
        return _decimal(value)
    if isinstance(value, str):
        return _text(value)
    return None


def _claim_node_triples(
    subjects: _Subjects, code: str, claims: Iterable[Mapping[str, Any]], settings: GraphSettings
) -> None:
    """Add one claim node per claim the ``claims`` setting selects.

    ``all`` selects every claim, ``contested`` the claims with a non-empty
    ``contests``, ``none`` no claim. A ``contests`` link is written only to a
    claim node that is itself written.
    """
    from osprey.facility.relations import end_identity

    if settings.claims == "none":
        return
    selected = [claim for claim in claims if settings.claims == "all" or claim.get("contests")]
    written = {str(claim["id"]) for claim in selected}
    used: set[str] = set()

    def add(subject: str, name: str, term: _Term) -> None:
        _add(subjects, subject, _p(name), term)
        used.add(name)

    for claim in sorted(selected, key=lambda claim: str(claim["id"])):
        claim_id = str(claim["id"])
        node = iri(code, "relation", claim_id)
        _add(subjects, node, RDF_TYPE, _iri(CLAIM))
        add(node, P_CLAIM_ID, _text(claim_id))
        add(node, P_RELATION_TYPE, _text(claim.get("type")))
        add(node, P_BASIS, _text(claim.get("basis", "stated")))
        for name, key in ((P_RULE, "rule"), (P_STANDING, "standing")):
            if claim.get(key) is not None:
                add(node, name, _text(claim[key]))
        source = claim.get("source")
        if isinstance(source, Mapping):
            for name, key in _SOURCE_FIELDS:
                if source.get(key) is not None:
                    add(node, name, _text(source[key]))
        for side, link, port, unit in _END_FIELDS:
            end = claim.get(side)
            if not isinstance(end, Mapping):
                continue
            identity = end_identity(end)
            if identity is not None:
                add(node, link, _iri(iri(code, *identity)))
            if end.get("port") is not None:
                add(node, port, _text(end["port"]))
            if end.get("unit") is not None:
                add(node, unit, _integer(end["unit"]))
        for rival in claim.get("contests") or []:
            if str(rival) in written:
                add(node, P_CONTESTS, _iri(iri(code, "relation", str(rival))))
        attributes = claim.get("attributes")
        if isinstance(attributes, Mapping):
            for key, value in attributes.items():
                term = _attribute_term(value)
                if term is not None:
                    add(node, _attribute_name(str(key)), term)
    for name in sorted(used):
        _declare(subjects, name, object_property=name in CLAIM_OBJECT_PROPERTY_NAMES)


def _group_triples(
    subjects: _Subjects, code: str, records: FacilityRecords, claims: Iterable[Mapping[str, Any]]
) -> None:
    """Add a node for each group that is an end of a written claim.

    Every claim counts, whatever its standing and whatever the ``claims``
    setting, so a group keeps its node and members when its claims have no
    claim node or no edge.
    """
    from osprey.facility.relations import end_identity

    group_ids: set[str] = set()
    for claim in claims:
        for side in ("subject", "object"):
            end = claim.get(side)
            identity = end_identity(end) if isinstance(end, Mapping) else None
            if identity is not None and identity[0] == "group":
                group_ids.add(identity[1])
    used: set[str] = set()
    for group_id in sorted(group_ids):
        group = records.record("group", group_id)
        if group is None:
            continue
        node = iri(code, "group", group_id)
        _add(subjects, node, RDF_TYPE, _iri(GROUP))
        _add(subjects, node, _p(P_GROUP_ID), _text(group_id))
        used.add(P_GROUP_ID)
        _names(subjects, node, group)
        for member in group.get("members") or []:
            _add(subjects, node, _p(P_HAS_MEMBER), _iri(iri(code, "device", str(member))))
            used.add(P_HAS_MEMBER)
    for name in sorted(used):
        _declare(subjects, name, object_property=name == P_HAS_MEMBER)


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


class GraphSettings(NamedTuple):
    """What the graph view writes of the facility file's relation claims.

    Attributes:
        claims: ``all`` writes every claim, ``contested`` only claims in a
            dispute, ``none`` no claim nodes.
        basis: ``all`` writes stated and inferred relations, ``stated`` only
            stated ones.
    """

    claims: str = "all"
    basis: str = "all"


#: The values each setting allows, keyed by its dotted config key.
_ALLOWED: dict[str, tuple[str, ...]] = {
    GRAPH_CLAIMS_KEY: ("all", "contested", "none"),
    GRAPH_BASIS_KEY: ("all", "stated"),
}


def _refusal(key: str, detail: str, remedy: str) -> FacilityBuildError:
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.served import CONFIG_SOURCE

    return FacilityBuildError(
        "profile-invalid", key, [CONFIG_SOURCE], remedy, record_kind="path", detail=detail
    )


def graph_settings(config: Mapping[str, Any]) -> GraphSettings:
    """The graph view's settings, read from a rendered config.

    Args:
        config: The rendered config, or the part of it at the graph view's reads.

    Returns:
        The settings; a key the config does not state, or states empty, is ``all``.

    Raises:
        FacilityBuildError: ``profile-invalid`` naming the dotted key and its
            allowed values when a key holds anything else.
    """
    root, _, _ = GRAPH_CLAIMS_KEY.partition(".")
    block = config.get(root)
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise _refusal(
            root,
            f"`{block}` is not a mapping",
            f"make `{root}` a mapping of `claims` and `basis`",
        )
    values: list[str] = []
    for key, allowed in _ALLOWED.items():
        field = key.rpartition(".")[2]
        value = block.get(field)
        if value is None:
            value = GraphSettings._field_defaults[field]
        if not isinstance(value, str) or value not in allowed:
            listed = ", ".join(f"`{name}`" for name in allowed)
            raise _refusal(
                key,
                f"`{value}` is not one of {listed}",
                f"set `{key}` to one of {listed}",
            )
        values.append(value)
    return GraphSettings(*values)


#: The records and settings last rendered and their Turtle: a render's views
#: share one read model, so its graph is rendered once per setting.
_rendered: tuple[FacilityRecords, GraphSettings, str] | None = None


def graph_text(records: FacilityRecords, settings: GraphSettings = GraphSettings()) -> str:
    """The graph view's Turtle for one facility file.

    Args:
        records: The records of the facility file as ``build_facility`` returned it.
        settings: What the view writes of the relation claims.

    Returns:
        The Turtle text, its first line the facility file's sha256 header,
        ending in one newline.
    """
    global _rendered
    if _rendered is None or _rendered[0] is not records or _rendered[1] != settings:
        _rendered = (records, settings, _turtle(records, settings))
    return _rendered[2]


def _turtle(records: FacilityRecords, settings: GraphSettings) -> str:
    """Compute :func:`graph_text` without the per-build memo."""
    code = str(records.document["identity"]["code"])
    classes = _class_rows(records)

    subjects: _Subjects = {}
    _place_triples(subjects, code, records.of_kind("place"), settings.basis)
    used = _device_triples(subjects, code, records, classes, settings.basis)
    signals = _channel_triples(subjects, code, records)
    for signal in sorted(signals):
        subject = f"{NARAD_SEM}{signal}"
        _add(subjects, subject, RDF_TYPE, _iri(SEMANTIC_SIGNAL))
        _add(subjects, subject, f"{RDFS}label", _text(signal))
    claims = _written_claims(records, settings)
    _relation_triples(subjects, code, claims)
    _claim_node_triples(subjects, code, claims, settings)
    _group_triples(subjects, code, records, claims)
    _class_triples(subjects, used, classes)
    _vocabulary_triples(subjects)

    digest = hashlib.sha256(records.facility_bytes()).hexdigest()
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

    Raises:
        FacilityBuildError: ``profile-invalid`` when a graph setting holds a
            value it does not allow, before anything is written.
    """
    settings = graph_settings(inputs.rendered_config)
    root.mkdir(parents=True, exist_ok=True)
    target = root / GRAPH_FILE
    target.write_bytes(graph_text(inputs.records, settings).encode("utf-8"))
    return [target]
