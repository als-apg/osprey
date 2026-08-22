"""Turtle serialisation of the generated demo-machine knowledge graph.

Turns a :class:`~osprey.services.facility_knowledge.ttl_generator.model.GraphModel`
plus an :class:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.OntologyMap`
into a NARAD-convention Turtle file that ``osprey knowledge seed-graph`` can
import into the graph store exactly like the shipped ALS corpus
(``src/osprey/templates/services/graphdb/als_gtb.ttl``).

The output mirrors that corpus triple-for-triple in *shape*: the same prefixes,
the same predicate spellings, devices carrying the identity and position
properties, ``narad_sem:ChannelBinding`` nodes carrying ``bindingId`` /
``confidence`` / ``fullPv`` / ``protocol`` and one of
``narad_p:readsSignal`` / ``narad_p:writesSignal``, ``narad_sem:SemanticSignal``
individuals, and an ``owl:Class`` hierarchy joined by ``rdfs:subClassOf``.

**System.** Every device also carries ``narad_p:system`` — the raw system
token of its address (``MAG``, ``DIAG``, ``VAC``).  It is the coarsest grouping
the address grammar offers, so a question like "how many vacuum devices are
there" is one property filter rather than a family-by-family enumeration.  It is
the token itself, not the prose: ``narad_p:systemDescription`` carries the text.

**Prose.** When the model carries description text, bindings also get
``narad_p:description`` / ``fieldDescription`` / ``subfieldDescription`` and
devices get ``familyDescription`` / ``systemDescription`` / ``ringDescription``.
Each is a distinct predicate carrying at most one value per node — neosemantics
collapses repeated values of one predicate, so two texts under one name would
lose data on import.  ``narad_sem:SemanticSignal`` carries no description text
by design: a signal is keyed ``(FAMILY, FIELD, SUBFIELD)`` without a ring, while
the prose is ring-qualified, so only the binding's six-token address resolves it
unambiguously.  Description fields left ``None`` emit nothing at all.

Those are the *Turtle-side* spellings.  neosemantics maps them to uppercase
relationship types (``HASBINDING``, ``READSSIGNAL``, …) when it imports the
file; those uppercase names belong to the Cypher projection only and must never
appear here.

**Determinism.** The same model always serialises to the same bytes.  Nothing is
left to a library's iteration order: the triples are assembled in an explicit
order, subjects are emitted in that order, predicates are sorted by IRI with
``rdf:type`` pinned first, and multiple objects of one predicate are sorted by
their lexical value.  That matters because the seed marker in the graph store is
a checksum over the Turtle text, so unstable output would make every deploy look
like a corpus change.

**Lazy rdflib.** ``rdflib`` is imported inside the functions that need it, never
at module top level — :func:`serialize_turtle` and :func:`write_turtle` do not
need it at all.  ``tests/services/facility_knowledge/test_import_isolation.py``
guards the discipline for the whole package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from .direction import DIRECTION_READ, DIRECTION_WRITE
from .model import (
    FACILITY,
    NARAD_PROPERTY_NS,
    NARAD_SEM_NS,
    ChannelBinding,
    Device,
    GraphModel,
    SignalGroup,
)
from .ontology_map import ClassDef, OntologyMap

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import rdflib

__all__ = [
    "CHANNEL_BINDING_CLASS_IRI",
    "OBJECT_PROPERTY_NAMES",
    "PREFIXES",
    "PROPERTY_NAMES",
    "SEMANTIC_SIGNAL_CLASS_IRI",
    "UndirectedSignalError",
    "build_graph",
    "serialize_turtle",
    "write_turtle",
]


# ---------------------------------------------------------------------------
# Vocabulary — the TTL-side spellings, matching als_gtb.ttl exactly
# ---------------------------------------------------------------------------

OWL_NS = "http://www.w3.org/2002/07/owl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"

#: Prefixes declared in the generated file — the same six ``als_gtb.ttl`` binds.
PREFIXES: dict[str, str] = {
    "narad_p": NARAD_PROPERTY_NS,
    "narad_sem": NARAD_SEM_NS,
    "owl": OWL_NS,
    "rdfs": RDFS_NS,
    "skos": SKOS_NS,
    "xsd": XSD_NS,
}

RDF_TYPE = f"{RDF_NS}type"
RDFS_LABEL = f"{RDFS_NS}label"
RDFS_SUBCLASS_OF = f"{RDFS_NS}subClassOf"
SKOS_ALT_LABEL = f"{SKOS_NS}altLabel"
OWL_CLASS = f"{OWL_NS}Class"
OWL_THING = f"{OWL_NS}Thing"
OWL_DATATYPE_PROPERTY = f"{OWL_NS}DatatypeProperty"
OWL_OBJECT_PROPERTY = f"{OWL_NS}ObjectProperty"

#: ``narad_sem:ChannelBinding`` — the class every binding node is typed as.
CHANNEL_BINDING_CLASS_IRI = f"{NARAD_SEM_NS}ChannelBinding"
#: ``narad_sem:SemanticSignal`` — the class every signal individual is typed as.
SEMANTIC_SIGNAL_CLASS_IRI = f"{NARAD_SEM_NS}SemanticSignal"

# Property local names, in the narad_p namespace.
P_BINDING_ID = "bindingId"
P_CONFIDENCE = "confidence"
P_DESCRIPTION = "description"
P_DEVICE_ID = "deviceId"
P_FACILITY = "facility"
P_FAMILY_DESCRIPTION = "familyDescription"
P_FIELD_DESCRIPTION = "fieldDescription"
P_FULL_PV = "fullPv"
P_HAS_BINDING = "hasBinding"
P_ORDINAL_IN_FACILITY = "ordinalInFacility"
P_ORDINAL_IN_SECTION = "ordinalInSection"
P_PROTOCOL = "protocol"
P_RAW_TYPE = "rawType"
P_READS_SIGNAL = "readsSignal"
P_RING_DESCRIPTION = "ringDescription"
P_S_POSITION_M = "sPositionM"
P_SECTION_CODE = "sectionCode"
P_SOURCE_NAME = "sourceName"
P_SOURCE_SECTION_ID = "sourceSectionId"
P_SUBFIELD_DESCRIPTION = "subfieldDescription"
P_SYSTEM = "system"
P_SYSTEM_DESCRIPTION = "systemDescription"
P_WRITES_SIGNAL = "writesSignal"

#: Prose predicates on a ``narad_sem:ChannelBinding``, paired with the
#: :class:`~osprey.services.facility_knowledge.ttl_generator.model.ChannelBinding`
#: attribute each reads.  Emission follows this order.
_BINDING_DESCRIPTION_FIELDS: tuple[tuple[str, str], ...] = (
    (P_DESCRIPTION, "description"),
    (P_FIELD_DESCRIPTION, "field_description"),
    (P_SUBFIELD_DESCRIPTION, "subfield_description"),
)

#: Prose predicates on a device node, paired with the
#: :class:`~osprey.services.facility_knowledge.ttl_generator.model.Device`
#: attribute each reads.  Emission follows this order.
_DEVICE_DESCRIPTION_FIELDS: tuple[tuple[str, str], ...] = (
    (P_FAMILY_DESCRIPTION, "family_description"),
    (P_SYSTEM_DESCRIPTION, "system_description"),
    (P_RING_DESCRIPTION, "ring_description"),
)

#: Properties declared ``a owl:ObjectProperty`` — they point at another node.
OBJECT_PROPERTY_NAMES: tuple[str, ...] = (P_HAS_BINDING, P_READS_SIGNAL, P_WRITES_SIGNAL)

#: Every ``narad_p:`` property the generated corpus uses, sorted.
PROPERTY_NAMES: tuple[str, ...] = tuple(
    sorted(
        (
            P_BINDING_ID,
            P_CONFIDENCE,
            P_DESCRIPTION,
            P_DEVICE_ID,
            P_FACILITY,
            P_FAMILY_DESCRIPTION,
            P_FIELD_DESCRIPTION,
            P_FULL_PV,
            P_HAS_BINDING,
            P_ORDINAL_IN_FACILITY,
            P_ORDINAL_IN_SECTION,
            P_PROTOCOL,
            P_RAW_TYPE,
            P_READS_SIGNAL,
            P_RING_DESCRIPTION,
            P_S_POSITION_M,
            P_SECTION_CODE,
            P_SOURCE_NAME,
            P_SOURCE_SECTION_ID,
            P_SUBFIELD_DESCRIPTION,
            P_SYSTEM,
            P_SYSTEM_DESCRIPTION,
            P_WRITES_SIGNAL,
        )
    )
)

#: Direction -> the ``narad_p:`` predicate that carries it on a binding.
_DIRECTION_PREDICATE: dict[str, str] = {
    DIRECTION_READ: P_READS_SIGNAL,
    DIRECTION_WRITE: P_WRITES_SIGNAL,
}

_INDENT = "    "
_CONTINUATION = "        "
_PN_LOCAL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRING_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


class UndirectedSignalError(ValueError):
    """Raised when a signal group reaches the emitter without a direction.

    A ``narad_sem:ChannelBinding`` must carry exactly one of
    ``narad_p:readsSignal`` / ``narad_p:writesSignal``, so the direction pass
    (:mod:`osprey.services.facility_knowledge.ttl_generator.direction`) has to
    run before serialisation.
    """

    def __init__(self, group: SignalGroup) -> None:
        super().__init__(group.key)
        self.group_key = group.key
        self._message = (
            f"Signal group {':'.join(group.key)} has no read/write direction, so its "
            f"{len(group.members)} binding(s) cannot be serialised. Run "
            "direction.assign_directions() (or resolve_and_assign) on the model first."
        )

    def __str__(self) -> str:
        return self._message


class _UnknownDirectionError(ValueError):
    """Raised when a signal group carries a direction the vocabulary has no word for."""


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Term:
    """One RDF object, kept independent of rdflib.

    Both renderers read this: :func:`_render_term` writes Turtle text and
    :func:`_rdflib_term` builds the rdflib object.  Keeping a single canonical
    description of every object is what makes the text and the graph agree —
    parsing the text back yields exactly the graph :func:`build_graph` builds.

    Args:
        kind: ``"iri"``, ``"str"``, ``"int"`` or ``"dec"``.
        value: Lexical form.  Integers and decimals are stored as the exact
            string that is written out, so text and graph cannot drift.
    """

    kind: str
    value: str


def _iri(value: str) -> _Term:
    """An IRI object."""
    return _Term("iri", value)


def _text(value: str) -> _Term:
    """A plain string literal."""
    return _Term("str", value)


def _integer(value: int) -> _Term:
    """An ``xsd:integer`` literal, written bare (``3``)."""
    return _Term("int", str(int(value)))


def _decimal(value: float) -> _Term:
    """An ``xsd:decimal`` literal, written bare (``1.0``) as ``als_gtb.ttl`` does."""
    lexical = repr(float(value))
    if "e" in lexical or "E" in lexical or "n" in lexical:
        # Infinities, NaN and exponent forms are not Turtle decimals; fall back
        # to a plain positional spelling.  The demo machine never gets here (its
        # positions are small ordinals), but a facility model might.
        lexical = format(Decimal(value), "f")
    if "." not in lexical:
        lexical = f"{lexical}.0"
    return _Term("dec", lexical)


def _escape(value: str) -> str:
    """Escape a string for a Turtle short quoted literal."""
    return "".join(_STRING_ESCAPES.get(char, char) for char in value)


def _render_iri(iri: str) -> str:
    """Render an IRI as a prefixed name when one of :data:`PREFIXES` covers it."""
    for prefix, namespace in PREFIXES.items():
        if iri.startswith(namespace):
            local = iri[len(namespace) :]
            if _PN_LOCAL.fullmatch(local):
                return f"{prefix}:{local}"
    return f"<{iri}>"


def _render_term(term: _Term) -> str:
    """Render one object term as Turtle."""
    if term.kind == "iri":
        return _render_iri(term.value)
    if term.kind == "str":
        return f'"{_escape(term.value)}"'
    # Integers and decimals use Turtle's bare numeric forms, like als_gtb.ttl.
    return term.value


def _object_sort_key(term: _Term) -> tuple[str, str]:
    """Order the objects of one predicate: IRIs first, then by lexical value."""
    return (term.kind, term.value)


def _rdflib_term(term: _Term) -> object:
    """Build the rdflib term for one object.

    ``rdflib`` is imported here rather than at module scope; see the module
    docstring.
    """
    from rdflib import Literal, URIRef

    if term.kind == "iri":
        return URIRef(term.value)
    if term.kind == "str":
        return Literal(term.value)
    if term.kind == "int":
        return Literal(int(term.value))
    return Literal(Decimal(term.value))


# ---------------------------------------------------------------------------
# Triple assembly
# ---------------------------------------------------------------------------


def _prop(name: str) -> str:
    """Full IRI of a ``narad_p:`` property."""
    return f"{NARAD_PROPERTY_NS}{name}"


def _description_triples(
    subject: str, source: object, fields: tuple[tuple[str, str], ...]
) -> list[tuple[str, str, _Term]]:
    """Prose triples for one node, one per described level.

    Args:
        subject: IRI of the node the text hangs on.
        source: The model object holding the description attributes.
        fields: ``(property local name, attribute name)`` pairs, in emission
            order — :data:`_BINDING_DESCRIPTION_FIELDS` or
            :data:`_DEVICE_DESCRIPTION_FIELDS`.

    Returns:
        One triple per attribute that holds text.  An attribute that is ``None``
        contributes nothing, so a model built without prose serialises exactly as
        it did before any description reached it.  Every name appears at most
        once, which is what keeps neosemantics from collapsing two texts into
        one on import.
    """
    triples: list[tuple[str, str, _Term]] = []
    for name, attribute in fields:
        text = getattr(source, attribute)
        if text is not None:
            triples.append((subject, _prop(name), _text(text)))
    return triples


def _device_triples(device: Device, ontology: OntologyMap) -> list[tuple[str, str, _Term]]:
    """Triples for one device node, in ``als_gtb.ttl``'s shape.

    Args:
        device: The device to describe.
        ontology: Table that says which ``narad_sem`` class its family is.

    Returns:
        ``(subject, predicate, object)`` triples; the subject is the device IRI.

    Raises:
        UnknownFamilyError: If the ontology table has no class for the family.
    """
    klass = ontology.class_for(device.family)
    subject = device.iri
    triples: list[tuple[str, str, _Term]] = [
        (subject, RDF_TYPE, _iri(klass.iri)),
        (subject, _prop(P_DEVICE_ID), _text(device.device_id)),
        (subject, _prop(P_FACILITY), _text(FACILITY)),
        (subject, _prop(P_ORDINAL_IN_FACILITY), _integer(device.ordinal_in_facility)),
        (subject, _prop(P_ORDINAL_IN_SECTION), _integer(device.ordinal_in_section)),
        (subject, _prop(P_RAW_TYPE), _text(device.raw_type)),
        (subject, _prop(P_S_POSITION_M), _decimal(device.s_position_m)),
        (subject, _prop(P_SECTION_CODE), _text(device.section_code)),
        (subject, _prop(P_SOURCE_NAME), _text(device.source_name)),
        (subject, _prop(P_SOURCE_SECTION_ID), _text(device.source_section_id)),
        (subject, _prop(P_SYSTEM), _text(device.system)),
    ]
    triples.extend(_description_triples(subject, device, _DEVICE_DESCRIPTION_FIELDS))
    triples.extend(
        (subject, _prop(P_HAS_BINDING), _iri(binding_iri)) for binding_iri in device.binding_iris
    )
    return triples


def _binding_triples(binding: ChannelBinding, predicate: str) -> list[tuple[str, str, _Term]]:
    """Triples for one ChannelBinding node.

    Args:
        binding: The binding to describe.
        predicate: ``narad_p:readsSignal`` or ``narad_p:writesSignal`` — the
            local name, chosen from its signal group's direction.

    Returns:
        The six triples ``als_gtb.ttl`` gives a binding, plus one prose triple
        for each of :data:`_BINDING_DESCRIPTION_FIELDS` the binding carries.
    """
    subject = binding.iri
    triples: list[tuple[str, str, _Term]] = [
        (subject, RDF_TYPE, _iri(CHANNEL_BINDING_CLASS_IRI)),
        (subject, _prop(P_BINDING_ID), _text(binding.binding_id)),
        (subject, _prop(P_CONFIDENCE), _text(binding.confidence)),
        (subject, _prop(P_FULL_PV), _text(binding.full_pv)),
        (subject, _prop(P_PROTOCOL), _text(binding.protocol)),
        (subject, _prop(predicate), _iri(binding.signal_iri)),
    ]
    triples.extend(_description_triples(subject, binding, _BINDING_DESCRIPTION_FIELDS))
    return triples


def _signal_triples(group: SignalGroup) -> list[tuple[str, str, _Term]]:
    """Triples for one ``narad_sem:SemanticSignal`` individual.

    A type and a label, and deliberately no prose: a signal is keyed
    ``(FAMILY, FIELD, SUBFIELD)`` while the hierarchy's field and subfield texts
    are ring-qualified, so several rings' texts would land on one signal and
    neosemantics would keep only one of them.  The prose goes on the bindings,
    whose six-token address resolves every text exactly.
    """
    return [
        (group.iri, RDF_TYPE, _iri(SEMANTIC_SIGNAL_CLASS_IRI)),
        (group.iri, RDFS_LABEL, _text(group.name)),
    ]


def _class_triples(klass: ClassDef) -> list[tuple[str, str, _Term]]:
    """Triples declaring one device class, with its parent and synonyms."""
    triples: list[tuple[str, str, _Term]] = [(klass.iri, RDF_TYPE, _iri(OWL_CLASS))]
    parent_iri = klass.parent_iri
    if parent_iri is not None:
        triples.append((klass.iri, RDFS_SUBCLASS_OF, _iri(parent_iri)))
    triples.extend(
        (klass.iri, SKOS_ALT_LABEL, _text(label)) for label in sorted(set(klass.alt_labels))
    )
    return triples


def _vocabulary_triples() -> list[tuple[str, str, _Term]]:
    """Triples for the fixed classes and the property declarations.

    ``narad_sem:SemanticSignal`` is a bare ``owl:Class`` and
    ``narad_sem:ChannelBinding`` is an ``owl:Class`` under ``owl:Thing``, exactly
    as ``als_gtb.ttl`` declares them.  Every ``narad_p:`` property the corpus
    uses is declared too — :data:`PROPERTY_NAMES` is the single list, so the six
    description properties are declared here as ``owl:DatatypeProperty`` by
    being in it.  ``narad_p:hasBinding`` is declared as both a datatype and an
    object property, again mirroring the shipped corpus.
    """
    triples: list[tuple[str, str, _Term]] = [
        (SEMANTIC_SIGNAL_CLASS_IRI, RDF_TYPE, _iri(OWL_CLASS)),
        (CHANNEL_BINDING_CLASS_IRI, RDF_TYPE, _iri(OWL_CLASS)),
        (CHANNEL_BINDING_CLASS_IRI, RDFS_SUBCLASS_OF, _iri(OWL_THING)),
    ]
    for name in PROPERTY_NAMES:
        iri = _prop(name)
        if name == P_HAS_BINDING:
            triples.append((iri, RDF_TYPE, _iri(OWL_DATATYPE_PROPERTY)))
            triples.append((iri, RDF_TYPE, _iri(OWL_OBJECT_PROPERTY)))
        elif name in OBJECT_PROPERTY_NAMES:
            triples.append((iri, RDF_TYPE, _iri(OWL_OBJECT_PROPERTY)))
        else:
            triples.append((iri, RDF_TYPE, _iri(OWL_DATATYPE_PROPERTY)))
    return triples


def _direction_predicate(group: SignalGroup) -> str:
    """Return the signal predicate a group's direction calls for.

    Raises:
        UndirectedSignalError: If the group has no direction yet.
        ValueError: If the direction is not one of ``direction.DIRECTION_READ``
            / ``direction.DIRECTION_WRITE``.
    """
    if group.direction is None:
        raise UndirectedSignalError(group)
    try:
        return _DIRECTION_PREDICATE[group.direction]
    except KeyError:
        known = ", ".join(repr(value) for value in sorted(_DIRECTION_PREDICATE))
        raise _UnknownDirectionError(
            f"Signal group {':'.join(group.key)} carries direction "
            f"{group.direction!r}; expected one of {known}"
        ) from None


def _model_triples(model: GraphModel, ontology: OntologyMap) -> list[tuple[str, str, _Term]]:
    """Assemble every triple of the corpus, in emission order.

    The order is devices, bindings, signal individuals, device classes, then the
    fixed classes and property declarations — the model's own deterministic
    ordering carries through, so the result is stable for a given input.

    Args:
        model: The derived graph, with directions already assigned.
        ontology: The FAMILY-to-class table.

    Returns:
        Triples as ``(subject IRI, predicate IRI, object term)``.

    Raises:
        UndirectedSignalError: If any signal group is still undirected.
        UnknownFamilyError: If a device family has no class in the table.
    """
    predicate_by_group = {group.key: _direction_predicate(group) for group in model.signal_groups}

    triples: list[tuple[str, str, _Term]] = []
    for device in model.devices:
        triples.extend(_device_triples(device, ontology))
    for binding in model.bindings:
        triples.extend(_binding_triples(binding, predicate_by_group[binding.signal_key]))
    for group in model.signal_groups:
        triples.extend(_signal_triples(group))

    families = {device.family for device in model.devices}
    for klass in ontology.used_classes(sorted(families)):
        triples.extend(_class_triples(klass))
    triples.extend(_vocabulary_triples())
    return triples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_graph(model: GraphModel, ontology: OntologyMap) -> rdflib.Graph:
    """Build the corpus as an in-memory rdflib graph.

    Mostly useful for tests and for callers that want to reason over the graph
    rather than over its text; :func:`serialize_turtle` does not go through it.

    Args:
        model: The derived graph, with directions already assigned.
        ontology: The FAMILY-to-class table.

    Returns:
        An ``rdflib.Graph`` with :data:`PREFIXES` bound, holding every triple of
        the corpus.

    Raises:
        UndirectedSignalError: If any signal group is still undirected.
        UnknownFamilyError: If a device family has no class in the table.
        ImportError: If ``rdflib`` is not installed.
    """
    from rdflib import Graph, URIRef

    graph = Graph()
    for prefix, namespace in PREFIXES.items():
        graph.bind(prefix, namespace, override=True)
    for subject, predicate, obj in _model_triples(model, ontology):
        graph.add((URIRef(subject), URIRef(predicate), _rdflib_term(obj)))
    return graph


def serialize_turtle(model: GraphModel, ontology: OntologyMap) -> str:
    """Serialise the corpus as deterministic Turtle text.

    Pure text assembly — no ``rdflib`` involved, so this works in an
    installation without the ``knowledge`` extra.  Identical input always yields
    identical output, byte for byte.

    Args:
        model: The derived graph, with directions already assigned.
        ontology: The FAMILY-to-class table.

    Returns:
        The Turtle document, ending in a newline.

    Raises:
        UndirectedSignalError: If any signal group is still undirected.
        UnknownFamilyError: If a device family has no class in the table.
    """
    by_subject: dict[str, dict[str, dict[tuple[str, str], _Term]]] = {}
    for subject, predicate, obj in _model_triples(model, ontology):
        objects = by_subject.setdefault(subject, {}).setdefault(predicate, {})
        objects[(obj.kind, obj.value)] = obj

    lines = [f"@prefix {prefix}: <{namespace}> ." for prefix, namespace in PREFIXES.items()]
    lines.append("")
    for subject, predicates in by_subject.items():
        lines.extend(_subject_block(subject, predicates))
        lines.append("")
    return "\n".join(lines[:-1]) + "\n"


def _subject_block(subject: str, predicates: dict[str, dict[tuple[str, str], _Term]]) -> list[str]:
    """Render one subject block.

    ``rdf:type`` comes first and is written as ``a``; the remaining predicates
    are sorted by IRI, which reproduces ``als_gtb.ttl``'s ordering (``rdfs:`` and
    ``skos:`` before the ``https://`` ``narad_p:`` properties, and ``rawType``
    before ``sPositionM`` before ``sectionCode``).  Multiple objects of one
    predicate are sorted by their lexical value — again as ``als_gtb.ttl`` does,
    so ``"yag"`` precedes ``"yag screen"``.
    """
    ordered = [RDF_TYPE] if RDF_TYPE in predicates else []
    ordered.extend(sorted(name for name in predicates if name != RDF_TYPE))

    clauses: list[str] = []
    for predicate in ordered:
        verb = "a" if predicate == RDF_TYPE else _render_iri(predicate)
        objects = [
            _render_term(term)
            for term in sorted(predicates[predicate].values(), key=_object_sort_key)
        ]
        joined = f",\n{_CONTINUATION}".join(objects)
        clauses.append(f"{verb} {joined}")

    lines = [f"{_render_iri(subject)} {clauses[0]}"]
    lines[0] += " ;" if len(clauses) > 1 else " ."
    for index, clause in enumerate(clauses[1:], start=2):
        terminator = " ;" if index < len(clauses) else " ."
        lines.append(f"{_INDENT}{clause}{terminator}")
    return lines


def write_turtle(model: GraphModel, ontology: OntologyMap, path: Path) -> Path:
    """Serialise the corpus and write it to *path* as UTF-8.

    Args:
        model: The derived graph, with directions already assigned.
        ontology: The FAMILY-to-class table.
        path: File to write.  Missing parent directories are created.

    Returns:
        *path*, so a caller can report where the corpus landed.

    Raises:
        UndirectedSignalError: If any signal group is still undirected.
        UnknownFamilyError: If a device family has no class in the table.
        OSError: If the file cannot be written.
    """
    text = serialize_turtle(model, ontology)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
