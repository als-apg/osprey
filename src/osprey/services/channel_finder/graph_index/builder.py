"""Deriving the search index's rows from a NARAD Turtle corpus.

The rows are defined by what the n10s-seeded store answers, not by a reading of
the NARAD vocabulary: a parity lane compares them against the search and
ontology queries kept as the reference implementation in
``tests/integration/_graph_oracles.py``, so every rule below copies a clause of
those queries and says which.

How n10s shapes the store (``handleVocabUris='MAP'`` with no mappings, so
local names; ``handleRDFTypes='LABELS_AND_NODES'``; ``applyNeo4jNaming``):

* every ``rdf:type`` becomes both a node label and a ``TYPE`` relationship to
  the type's node. A node is ``:ChannelBinding``, ``:SemanticSignal`` or
  ``:Class`` because the corpus types it with a URI whose local name is that
  word (``narad_sem:ChannelBinding``, ``owl:Class``); nothing is labelled
  ``:Class`` for merely being the target of a ``TYPE`` edge;
* an object property becomes an upper-cased relationship: ``HASBINDING``,
  ``READSSIGNAL``, ``WRITESSIGNAL``, ``SUBCLASSOF``;
* a literal becomes a scalar property under its local name, except
  ``skos:altLabel``, which is the one list-valued property.

Only ``rdflib`` is imported here, inside :func:`parse_corpus`, and ``duckdb``
and ``pandas`` only inside :func:`build_from_rows`, so importing this module
stays cheap: the roster and the health check reach this package on paths where
any of those dependencies appearing in ``sys.modules`` would be the regression.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import astuple, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.exceptions import GraphIndexBuildError
from .schema import META_KEYS, SCHEMA_VERSION, create_tables
from .taxonomy import class_name, prune_device_taxonomy

if TYPE_CHECKING:  # pragma: no cover - typing only; the roster stays a lazy import
    from osprey.channel_roster.records import RosterSource

logger = logging.getLogger(__name__)

#: The ``narad_p:`` property namespace, spelled here for the same reason
#: :mod:`osprey.channel_roster.graph` spells it: importing the facility
#: knowledge package would pull ``osprey.services.qmd`` into the build.
NARAD_P = "https://narad.example.org/property/"

#: Local names n10s turns into the labels the Cypher matches on. Any namespace
#: works: with local-name vocabulary handling, ``owl:Class`` and ``rdfs:Class``
#: both label a node ``:Class``.
CLASS_LABEL = "Class"
CHANNEL_BINDING_LABEL = "ChannelBinding"
SEMANTIC_SIGNAL_LABEL = "SemanticSignal"

#: The edge types as the store spells them and as ``edges`` carries them —
#: ``type(e)`` of the ``READSSIGNAL|WRITESSIGNAL`` relationships. Listed in the
#: fixed ascending order a row's ``edges`` list uses.
EDGE_READS = "READSSIGNAL"
EDGE_WRITES = "WRITESSIGNAL"

#: How the ``channels`` table spells a direction. These are the two values of
#: :data:`osprey.channel_roster.records.ChannelDirection`, repeated here rather
#: than imported for the reason the module docstring gives for every other
#: spelled constant: the roster's record module is stdlib-only, but reaching it
#: from a table-shaping helper would tie this module's import graph to it for
#: two strings. A record built with anything else raises in ``ChannelRecord``,
#: and the demo-corpus pin below compares these rows against real records.
DIRECTION_READ = "read"
DIRECTION_WRITE = "write"

#: Bound on the ``SUBCLASSOF`` walk, copied from ``[:SUBCLASSOF*0..10]``: a
#: corpus whose class edges form a cycle must not walk forever, and no real
#: ontology nests device classes ten deep.
MAX_SUBCLASS_HOPS = 10


@dataclass(slots=True)
class BindingRow:
    """One row of the ``bindings`` table, fields in column order.

    One row per ``(device, binding)`` pair the store's
    ``(d:Resource)-[:HASBINDING]->(b:ChannelBinding)`` match yields: a binding
    node hung under two devices is two rows, and two binding nodes carrying one
    ``fullPv`` are two rows too.
    """

    binding_uri: str
    full_pv: str
    description: str | None
    device_uri: str
    device_name: str | None
    section: str | None
    system: str | None
    edges: list[str]
    signal_uris: list[str]
    signal_names: list[str]
    class_uris: list[str]
    haystack: str


@dataclass(slots=True)
class ClassRow:
    """One row of the ``classes`` table, fields in column order.

    Only classes of the pruned device taxonomy become rows; the counts are over
    bound devices — subjects with at least one ``hasBinding`` target that is
    typed ``ChannelBinding`` — exactly as the ontology query in
    ``tests/integration/_graph_oracles.py`` counts them.
    """

    uri: str
    name: str
    alt_labels: list[str]
    parents: list[str]
    direct_devices: int
    rollup_devices: int


@dataclass(slots=True)
class ChannelRow:
    """One row of the ``channels`` table, fields in column order.

    The channel roster, one row per address: a ``full_pv`` bound under two
    devices is one channel however many bindings carry it. ``direction`` is
    ``"read"``, ``"write"`` or ``None`` when the corpus cannot say, and
    ``readback`` is ``None`` -- never the empty string -- for an address the
    corpus pairs nothing with.
    """

    address: str
    direction: str | None
    readback: str | None


@dataclass(slots=True)
class ParsedCorpus:
    """Everything one rdflib parse of the corpus yields.

    ``binding_rows`` and ``class_rows`` are the index tables. ``graph``,
    ``writes``, ``reads`` and ``bindings`` are the roster's raw material, shaped
    exactly as :func:`osprey.channel_roster.graph.read_graph_roster` builds them
    so that ``_records`` and ``_corpus_readbacks`` from that module can be
    applied without a second parse: ``writes``/``reads`` are the sets of
    binding nodes carrying ``writesSignal``/``readsSignal`` (whatever the edge
    points at), and ``bindings`` is ``(address, binding_node)`` for every
    non-empty ``fullPv`` literal in the corpus, typed or not.
    """

    graph: Any
    binding_rows: list[BindingRow]
    class_rows: list[ClassRow]
    writes: set[Any]
    reads: set[Any]
    bindings: list[tuple[str, Any]]
    #: Distinct ``sectionCode`` literal values over every subject, which is how
    #: the store's section census counts them (bound or not).
    section_codes: frozenset[str] = field(default_factory=frozenset)
    #: Subjects typed ``SemanticSignal``, the store's signal census.
    signal_count: int = 0


def parse_corpus(text: str) -> ParsedCorpus:
    """Parse Turtle text and derive the index rows the store would answer.

    Args:
        text: The corpus, as ``ttl_path.read_text(encoding="utf-8")`` returns
            it — the same string the seeders hash.

    Returns:
        The rows and the roster's raw material; see :class:`ParsedCorpus`.
        A corpus binding no channel is not an error: both row lists are empty
        and the caller decides what to say about it.

    Raises:
        GraphIndexBuildError: When the text is not valid Turtle. The message
            carries the parser's own.
    """
    from rdflib import RDF, RDFS, SKOS, BNode, Graph, Literal, URIRef

    graph = Graph()
    try:
        graph.parse(data=text, format="turtle")
    except Exception as e:
        raise GraphIndexBuildError(f"The corpus is not valid Turtle: {e}") from e

    p_has_binding = URIRef(NARAD_P + "hasBinding")
    p_full_pv = URIRef(NARAD_P + "fullPv")
    p_reads = URIRef(NARAD_P + "readsSignal")
    p_writes = URIRef(NARAD_P + "writesSignal")
    p_description = URIRef(NARAD_P + "description")
    p_source_name = URIRef(NARAD_P + "sourceName")
    p_section = URIRef(NARAD_P + "sectionCode")
    p_system = URIRef(NARAD_P + "system")

    def is_node(term: Any) -> bool:
        return isinstance(term, URIRef | BNode)

    def literal(subject: Any, predicate: Any) -> str | None:
        """The one scalar the store holds for ``subject.predicate``.

        n10s keeps a single value for every property outside its multi-value
        list; a corpus repeating one is drift, and the smallest lexical form
        is taken so the answer does not depend on parse order.
        """
        values = sorted(str(o) for o in graph.objects(subject, predicate) if isinstance(o, Literal))
        return values[0] if values else None

    # -- the labels n10s derives from rdf:type ------------------------------
    labelled: dict[str, set[Any]] = {
        CLASS_LABEL: set(),
        CHANNEL_BINDING_LABEL: set(),
        SEMANTIC_SIGNAL_LABEL: set(),
    }
    for subject, type_uri in graph.subject_objects(RDF.type):
        if not is_node(type_uri):
            continue
        members = labelled.get(class_name(str(type_uri)))
        if members is not None:
            members.add(subject)
    class_nodes = labelled[CLASS_LABEL]
    binding_nodes = labelled[CHANNEL_BINDING_LABEL]
    signal_nodes = labelled[SEMANTIC_SIGNAL_LABEL]

    # -- the roster's raw material, shaped as channel_roster.graph shapes it --
    writes = set(graph.subjects(p_writes, None))
    reads = set(graph.subjects(p_reads, None))
    bindings = [
        (str(address), binding)
        for binding, address in graph.subject_objects(p_full_pv)
        if isinstance(address, Literal) and str(address)
    ]

    # -- SUBCLASSOF ancestry, the store's [:SUBCLASSOF*0..10] to a :Class ----
    parents_of: dict[Any, list[Any]] = {}
    for sub, parent in graph.subject_objects(RDFS.subClassOf):
        if is_node(parent):
            parents_of.setdefault(sub, []).append(parent)

    ancestors_cache: dict[Any, frozenset[Any]] = {}

    def class_ancestors(start: Any) -> frozenset[Any]:
        """Every ``:Class`` node within ten ``SUBCLASSOF`` hops of ``start``.

        ``start`` itself is included when it is a class (the ``*0`` hop). The
        walk passes through unlabelled nodes but collects only classes, as the
        pattern's unconstrained interior and ``(anc:Class)`` end do; the
        visited set is what a bounded variable-length match amounts to on a
        cycle.
        """
        cached = ancestors_cache.get(start)
        if cached is not None:
            return cached
        found: set[Any] = set()
        seen = {start}
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in class_nodes:
                found.add(node)
            if depth == MAX_SUBCLASS_HOPS:
                continue
            for parent in parents_of.get(node, ()):
                if parent not in seen:
                    seen.add(parent)
                    queue.append((parent, depth + 1))
        result = frozenset(found)
        ancestors_cache[start] = result
        return result

    # -- devices: subjects bound to at least one :ChannelBinding ------------
    bound_targets: dict[Any, set[Any]] = {}
    for device, binding in graph.subject_objects(p_has_binding):
        if binding in binding_nodes:
            bound_targets.setdefault(device, set()).add(binding)

    device_classes: dict[Any, frozenset[Any]] = {}
    device_ancestors: dict[Any, frozenset[Any]] = {}
    for device in bound_targets:
        own = frozenset(t for t in graph.objects(device, RDF.type) if t in class_nodes)
        device_classes[device] = own
        device_ancestors[device] = frozenset().union(*(class_ancestors(t) for t in own))

    # -- binding rows -------------------------------------------------------
    def signal_name(signal: Any) -> str:
        """``coalesce(s.label, last(split(s.uri, '/')))`` — no ``#`` split here."""
        label = literal(signal, RDFS.label)
        return label if label is not None else str(signal).rsplit("/", 1)[-1]

    signal_name_cache: dict[Any, str] = {}
    alt_labels_cache: dict[Any, list[str]] = {}

    def alt_labels(cls: Any) -> list[str]:
        cached = alt_labels_cache.get(cls)
        if cached is None:
            cached = sorted(
                str(o) for o in graph.objects(cls, SKOS.altLabel) if isinstance(o, Literal)
            )
            alt_labels_cache[cls] = cached
        return cached

    class_names_cache: dict[frozenset[Any], list[str]] = {}

    def class_names(ancestors: frozenset[Any]) -> list[str]:
        """The lowercase names the token filter matches for a device's classes.

        Copies ``classNames``: the trailing fragment of every ancestor URI plus
        every ``altLabel`` of every ancestor, all lower-cased.
        """
        cached = class_names_cache.get(ancestors)
        if cached is None:
            names = [class_name(str(a)).lower() for a in sorted(ancestors, key=str)]
            for a in sorted(ancestors, key=str):
                names.extend(x.lower() for x in alt_labels(a))
            cached = names
            class_names_cache[ancestors] = cached
        return cached

    binding_rows: list[BindingRow] = []
    for device, targets in bound_targets.items():
        device_uri = str(device)
        device_name = literal(device, p_source_name)
        section = literal(device, p_section)
        system = literal(device, p_system)
        ancestors = device_ancestors[device]
        class_uris = sorted(str(a) for a in ancestors)
        names_of_classes = class_names(ancestors)
        for binding in targets:
            full_pv = literal(binding, p_full_pv)
            if not full_pv:
                # The store would answer this binding with a null fullPv; the
                # table's NOT NULL column cannot, and the roster skips it too.
                continue
            description = literal(binding, p_description)
            edges: set[str] = set()
            signals: set[tuple[str, str]] = set()
            for predicate, edge in ((p_reads, EDGE_READS), (p_writes, EDGE_WRITES)):
                for target in graph.objects(binding, predicate):
                    # ``(b)-[e]->(s:SemanticSignal)``: an edge to anything not
                    # typed SemanticSignal binds neither ``e`` nor ``s``.
                    if target not in signal_nodes:
                        continue
                    edges.add(edge)
                    name = signal_name_cache.get(target)
                    if name is None:
                        name = signal_name(target)
                        signal_name_cache[target] = name
                    signals.add((name, str(target)))
            ordered_signals = sorted(signals)
            signal_names = [name for name, _ in ordered_signals]
            haystack_parts = [full_pv]
            if description is not None:
                haystack_parts.append(description)
            if device_name is not None:
                haystack_parts.append(device_name)
            haystack_parts.extend(signal_names)
            haystack_parts.extend(names_of_classes)
            binding_rows.append(
                BindingRow(
                    binding_uri=str(binding),
                    full_pv=full_pv,
                    description=description,
                    device_uri=device_uri,
                    device_name=device_name,
                    section=section,
                    system=system,
                    edges=sorted(edges),
                    signal_uris=[uri for _, uri in ordered_signals],
                    signal_names=signal_names,
                    class_uris=class_uris,
                    haystack=" ".join(haystack_parts).lower(),
                )
            )
    binding_rows.sort(key=lambda r: (r.full_pv, r.device_uri, r.binding_uri))

    # -- class rows: the ontology query, then the explorer's pruning --------
    rollup: dict[Any, set[Any]] = {c: set() for c in class_nodes}
    direct: dict[Any, set[Any]] = {c: set() for c in class_nodes}
    for device, own in device_classes.items():
        for cls in own:
            direct[cls].add(device)
        for anc in device_ancestors[device]:
            rollup[anc].add(device)

    raw_rows = [
        {
            "uri": str(cls),
            "altLabel": alt_labels(cls),
            # ``(c)-[:SUBCLASSOF]->(p:Class)``: direct parents that are classes.
            "parents": sorted({str(p) for p in parents_of.get(cls, ()) if p in class_nodes}),
            "rollup": len(rollup[cls]),
            "direct": len(direct[cls]),
        }
        for cls in class_nodes
    ]
    direct_by_uri = {row["uri"]: row["direct"] for row in raw_rows}
    class_rows = [
        ClassRow(
            uri=kept["uri"],
            name=kept["name"],
            alt_labels=list(kept["altLabel"]),
            parents=list(kept["parents"]),
            direct_devices=direct_by_uri[kept["uri"]],
            rollup_devices=kept["rollup"],
        )
        for kept in prune_device_taxonomy(raw_rows)
    ]

    section_codes = frozenset(
        str(o) for o in graph.objects(None, p_section) if isinstance(o, Literal)
    )

    return ParsedCorpus(
        graph=graph,
        binding_rows=binding_rows,
        class_rows=class_rows,
        writes=writes,
        reads=reads,
        bindings=bindings,
        section_codes=section_codes,
        signal_count=len(signal_nodes),
    )


def _row_direction(row: BindingRow) -> str | None:
    """Which way one binding row points, or ``None`` when it cannot say.

    The same rule :func:`osprey.channel_roster.graph._direction` applies to a
    binding node, read off the row's ``edges`` instead: a row carrying only
    ``WRITESSIGNAL`` is settable, one carrying only ``READSSIGNAL`` is
    readable, and one carrying both or neither abstains. Claiming both is
    drift in the corpus, and calling such a channel writable on a guess is the
    one error that reaches hardware.
    """
    is_write = EDGE_WRITES in row.edges
    is_read = EDGE_READS in row.edges
    if is_write and not is_read:
        return DIRECTION_WRITE
    if is_read and not is_write:
        return DIRECTION_READ
    return None


def channels_from_rows(rows: Iterable[BindingRow]) -> list[ChannelRow]:
    """Collapse binding rows into the roster, one row per address.

    The vote is the roster's own: an address's direction is the one its
    bindings agree on, a binding that states none abstains, and bindings that
    disagree leave the address undirected. Every readback is ``None``: a
    corpus states a pair between two *bindings* through their device grouping
    and their ``bindingId`` fields, neither of which survives into a
    :class:`BindingRow`, so the rows cannot answer that half.

    Use this when the rows are all that is at hand -- a rebuild from a stored
    index. When the parse is at hand, :func:`channels_from_corpus` is the
    answer of record: it derives the same table from the roster reader itself,
    readbacks included, and over an untyped ``hasBinding`` target or an edge to
    an untyped signal the two legitimately differ (see that function).

    Args:
        rows: The binding rows, in any order.

    Returns:
        One :class:`ChannelRow` per distinct ``full_pv``, sorted by address.
    """
    votes: dict[str, set[str]] = {}
    for row in rows:
        directions = votes.setdefault(row.full_pv, set())
        direction = _row_direction(row)
        if direction is not None:
            directions.add(direction)
    return [
        ChannelRow(
            address=address,
            direction=next(iter(votes[address])) if len(votes[address]) == 1 else None,
            readback=None,
        )
        for address in sorted(votes)
    ]


def channels_from_corpus(parsed: ParsedCorpus, source: RosterSource) -> list[ChannelRow]:
    """Derive the ``channels`` table from the parse, as the roster reader would.

    The roster is not re-implemented here. :attr:`ParsedCorpus.writes`,
    ``reads``, ``bindings`` and ``graph`` are shaped exactly as
    :func:`osprey.channel_roster.graph.read_graph_roster` shapes them, so that
    module's own ``_records`` and ``_corpus_readbacks`` are applied to them and
    the index carries the records that reader answers -- byte for byte, with
    the readbacks the corpus states. Reaching for its private names is
    deliberate: a second spelling of the address vote or of the
    ``Setpoint``/``Monitor`` pairing is a second answer to drift away, and the
    index exists to serve the roster's answer without a second parse. The
    import runs inside the function so importing this module stays cheap and
    the dependency points one way only, from the index to the roster.

    This differs from :func:`channels_from_rows` on two corpus shapes, both
    honest: the roster is untyped-agnostic, so a ``hasBinding`` target that is
    never typed ``ChannelBinding`` is a channel here and no row there, and it
    reads direction off the ``readsSignal``/``writesSignal`` predicates
    themselves, so a binding pointing at a node that is not typed
    ``SemanticSignal`` is directed here while the row it produced carries no
    edge. The shipped demo corpus has neither shape, and the two agree on it.

    Args:
        parsed: The corpus parse, from :func:`parse_corpus`.
        source: The resolved corpus the roster records name as their
            provenance. Only carried through the records; nothing is read from
            disk here.

    Returns:
        One :class:`ChannelRow` per address, sorted by address, carrying the
        direction the corpus states and the readback it pairs.
    """
    from osprey.channel_roster.graph import _corpus_readbacks, _records

    readbacks = _corpus_readbacks(parsed.graph, parsed.writes, parsed.reads, parsed.bindings)
    records = _records(parsed.bindings, source, parsed.writes, parsed.reads, readbacks)
    return [
        ChannelRow(address=record.address, direction=record.direction, readback=record.readback)
        for record in records
    ]


# -- writing the rows into a DuckDB file ------------------------------------

#: The columns of each bulk-loaded table, in the order the schema declares them
#: and the order the row dataclasses' fields are in. They are named rather than
#: taken as ``*`` so the ``INSERT`` states which value goes where: a frame whose
#: columns had drifted out of table order would otherwise load silently.
BINDING_COLUMNS = (
    "binding_uri",
    "full_pv",
    "description",
    "device_uri",
    "device_name",
    "section",
    "system",
    "edges",
    "signal_uris",
    "signal_names",
    "class_uris",
    "haystack",
)
CLASS_COLUMNS = ("uri", "name", "alt_labels", "parents", "direct_devices", "rollup_devices")
CHANNEL_COLUMNS = ("address", "direction", "readback")

#: The name the bulk loader registers its frame under. Nothing else in the
#: build names a view, and it is unregistered again before the next table, so
#: one name is enough.
BULK_VIEW = "osprey_index_rows"

#: The ``meta`` columns the caller states. ``schema_version`` is not among them:
#: the writer stamps it from :data:`~.schema.SCHEMA_VERSION`, because a caller
#: that could choose it could write a file the reader then refuses.
CALLER_META_KEYS = tuple(key for key in META_KEYS if key != "schema_version")


@dataclass(slots=True)
class IndexBuildReport:
    """What one index build wrote, returned so the caller can report it.

    ``path`` is the index that now exists. ``corpus_sha256`` and the five
    census counts are the ``meta`` row as written — the same numbers the
    explorer's badges and the health check read back. ``channel_count`` is the
    number of ``channels`` rows written; ``meta`` has no column for it, since
    the roster's size is not a badge.
    """

    path: Path
    corpus_sha256: str
    binding_count: int
    device_count: int
    class_count: int
    signal_count: int
    section_count: int
    channel_count: int


def _insert(con: Any, table: str, columns: tuple[str, ...], rows: Iterable[Any]) -> int:
    """Bulk-load ``rows`` into ``table``, returning how many were written.

    The rows are pivoted into one column per table column and handed to DuckDB
    as a single registered ``pandas`` frame, which the ``INSERT ... SELECT``
    then reads columnwise. The obvious spelling — one ``executemany`` over the
    row tuples — costs about a quarter of a millisecond per row whatever the
    batch size, because it walks DuckDB's prepared-statement path once per row;
    a hundred thousand bindings took nearly forty seconds that way against
    under a second here. ``pandas`` is a runtime dependency already (the
    archiver connectors return frames), so this buys the order of magnitude
    without adding one.

    DuckDB reads each object column's Python values directly, so a list stays a
    list, ``None`` stays SQL ``NULL`` and an empty list stays an empty list —
    the three shapes the reader tests for. A column of nothing but ``None``
    arrives typed as the target column by the ``INSERT``'s own cast.

    Raises:
        GraphIndexBuildError: When a row does not have one value per column.
            The columnar path would otherwise pad the short row out with nulls
            and load it, and a row the caller built wrong must not reach a
            file the reader trusts.
    """
    import pandas as pd

    tuples = [tuple(row) for row in rows]
    if not tuples:
        return 0
    for position, row in enumerate(tuples):
        if len(row) != len(columns):
            raise GraphIndexBuildError(
                f"Row {position} of {table} carries {len(row)} values, but the table "
                f"has {len(columns)} columns: {', '.join(columns)}."
            )

    frame = pd.DataFrame(
        {
            name: list(values)
            for name, values in zip(columns, zip(*tuples, strict=True), strict=True)
        },
        columns=list(columns),
        dtype=object,
    )
    con.register(BULK_VIEW, frame)
    try:
        con.execute(f"INSERT INTO {table} SELECT {', '.join(columns)} FROM {BULK_VIEW}")
    finally:
        con.unregister(BULK_VIEW)
    return len(tuples)


def _channel_tuple(row: ChannelRow) -> tuple[Any, ...]:
    """A ``channels`` row with an unstated readback as SQL NULL, never ``''``.

    :class:`ChannelRow` already promises this, and the roster builds it that
    way; the normalisation is here because a caller assembling rows by hand is
    the one path that could slip an empty string into a column the reader tests
    with ``IS NULL``.
    """
    address, direction, readback = astuple(row)
    return (address, direction, readback or None)


def _meta_values(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    """The ``meta`` row in column order, or raise if the caller's keys are wrong.

    Raises:
        GraphIndexBuildError: When ``meta`` misses one of
            :data:`CALLER_META_KEYS` or carries a key that is not one of them —
            ``schema_version`` included, which the writer stamps itself.
    """
    supplied = set(meta)
    expected = set(CALLER_META_KEYS)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing or unknown:
        faults = []
        if missing:
            faults.append(f"missing {', '.join(missing)}")
        if unknown:
            faults.append(f"unknown {', '.join(unknown)}")
        raise GraphIndexBuildError(
            f"The index meta row is wrong: {'; '.join(faults)}. "
            f"Supply exactly: {', '.join(CALLER_META_KEYS)} "
            "(schema_version is the writer's to stamp)."
        )
    return (SCHEMA_VERSION, *(meta[key] for key in CALLER_META_KEYS))


def _remove(path: Path) -> None:
    """Delete ``path`` and the write-ahead log DuckDB may have left beside it."""
    path.unlink(missing_ok=True)
    Path(f"{path}.wal").unlink(missing_ok=True)


def build_from_rows(
    rows: Iterable[BindingRow],
    classes: Iterable[ClassRow],
    channels: Iterable[ChannelRow],
    index_path: Path,
    meta: Mapping[str, Any],
) -> IndexBuildReport:
    """Write the derived rows into a DuckDB index file, atomically.

    The file is built beside its destination as ``<name>.tmp-<pid>`` and moved
    over it with :func:`os.replace` only once every row and the ``meta`` row are
    written and the connection is closed. A reader holding the old index
    therefore never sees a half-built one, and a build that fails leaves the
    previous index exactly as it was.

    ``duckdb`` and ``pandas`` are imported inside this function: importing this
    module must stay cheap for the roster and the health check, which reach it
    on paths where pulling a database engine into the process would be the
    regression. The tables are loaded columnwise through a registered frame
    rather than row by row — see :func:`_insert` for why.

    Args:
        rows: The ``bindings`` rows, written in the order given.
        classes: The ``classes`` rows.
        channels: The ``channels`` rows — the roster, one per address.
        index_path: Where the index goes. Its parent directory must already
            exist; creating it is the caller's job, because only the caller
            knows whether a missing directory is a build step not yet run or a
            path typed wrong.
        meta: The census the ``meta`` row carries, keyed by
            :data:`CALLER_META_KEYS`. ``binding_count`` and ``class_count`` must
            agree with the rows handed in; the other three counts and the digest
            are the caller's to state, since no row here carries them.

    Returns:
        An :class:`IndexBuildReport` naming the file and what it holds.

    Raises:
        GraphIndexBuildError: When ``meta`` has the wrong keys, when its counts
            disagree with the rows written, when a row does not carry one value
            per table column, or when ``index_path``'s directory does not
            exist. Anything the database itself raises — a NOT NULL column left
            null, a value of the wrong type, a disk that fills — comes through
            unchanged, after the temporary file has been removed.
    """
    import duckdb

    meta_values = _meta_values(meta)
    index_path = Path(index_path)
    if not index_path.parent.is_dir():
        raise GraphIndexBuildError(
            f"Cannot write the search index at {index_path}: its directory "
            f"{index_path.parent} does not exist. Create it before building."
        )

    # A fresh file every time: create_tables issues plain CREATE TABLE, so a
    # leftover temp file from a killed build would fail the build rather than be
    # overwritten.
    tmp_path = index_path.with_name(f"{index_path.name}.tmp-{os.getpid()}")
    _remove(tmp_path)

    try:
        con = duckdb.connect(str(tmp_path))
        try:
            create_tables(con)
            binding_count = _insert(
                con, "bindings", BINDING_COLUMNS, (astuple(row) for row in rows)
            )
            class_count = _insert(con, "classes", CLASS_COLUMNS, (astuple(row) for row in classes))
            channel_count = _insert(
                con, "channels", CHANNEL_COLUMNS, (_channel_tuple(row) for row in channels)
            )
            if binding_count == 0:
                logger.warning(
                    "The corpus bound no channels: writing an empty search index at %s. "
                    "Search will answer nothing until the corpus names a channel binding.",
                    index_path,
                )
            _check_counts(meta, binding_count=binding_count, class_count=class_count)
            con.execute(
                f"INSERT INTO meta VALUES ({', '.join('?' * len(META_KEYS))})",
                list(meta_values),
            )
        finally:
            con.close()
        os.replace(tmp_path, index_path)
    except BaseException:
        _remove(tmp_path)
        raise

    return IndexBuildReport(
        path=index_path,
        corpus_sha256=meta["corpus_sha256"],
        binding_count=binding_count,
        device_count=meta["device_count"],
        class_count=class_count,
        signal_count=meta["signal_count"],
        section_count=meta["section_count"],
        channel_count=channel_count,
    )


def _check_counts(meta: Mapping[str, Any], *, binding_count: int, class_count: int) -> None:
    """Refuse a ``meta`` row that miscounts the rows written beside it.

    ``binding_count`` is what the reader calls an index empty by and what the
    explorer's badge shows, and ``class_count`` is the size of the taxonomy just
    written: both are the same numbers the writer has just counted, so a
    disagreement is the caller's census drifting from its own rows, not a
    difference of opinion worth carrying into the file.
    """
    for key, written in (("binding_count", binding_count), ("class_count", class_count)):
        stated = meta[key]
        if stated != written:
            raise GraphIndexBuildError(
                f"The index meta row states {key}={stated} but {written} rows were "
                "written. The counts must come from the same rows as the tables."
            )


# -- the whole build, corpus to index ---------------------------------------


def build_graph_index(ttl_path: Path, index_path: Path) -> IndexBuildReport:
    """Derive a search index from a Turtle corpus, in one pass over the file.

    The corpus is read once, and that one string is what everything downstream
    is computed from: the digest the ``meta`` row carries, and the parse the
    rows come from. Reading it twice would let a corpus rewritten between the
    two reads produce an index whose digest names a file it does not hold, and
    the digest is exactly what the build hook memoises on.

    :func:`~osprey.services.facility_knowledge.seeder.graph_seeder.ttl_sha256`
    is that digest, not a second spelling of it: the seed marker in the store
    and the ``corpus_sha256`` column here have to be the same number, or a
    deployment whose store and index were filled from one corpus would look to
    every consumer like two. Reading through :meth:`~pathlib.Path.read_text` is
    what makes them comparable at all -- it normalises line endings, so a
    corpus checked out with CRLF hashes the same as the LF file it came from.

    The five census counts are the populations the device, signal and section
    count queries in ``tests/integration/_graph_oracles.py`` count in the seeded
    store, derived here from the same parse rather than asked of a store the
    build may not be running.

    Args:
        ttl_path: The corpus to read.
        index_path: Where the index goes. Its parent directory must already
            exist (see :func:`build_from_rows`).

    Returns:
        An :class:`IndexBuildReport` naming the file written and what it holds.

    Raises:
        OSError: When the corpus cannot be read -- :exc:`FileNotFoundError` for
            a path that is not there. Deliberately not wrapped: a corpus a
            build has not staged yet and a corpus that is there and malformed
            send an operator to different fixes, and a caller can only tell
            them apart if they arrive as different exceptions.
        GraphIndexBuildError: When the corpus is not valid Turtle, or when the
            index cannot be written (see :func:`build_from_rows`).
    """
    from osprey.channel_roster.records import RosterSource, RosterSourceKind
    from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

    started = time.perf_counter()
    ttl_path = Path(ttl_path)
    text = ttl_path.read_text(encoding="utf-8")
    digest = ttl_sha256(text)
    parsed = parse_corpus(text)

    # Provenance only: the records carry this source, and nothing is read from
    # disk through it. The corpus is the honest answer to "where did these
    # channels come from" here, whatever file the roster later opens to get
    # them back.
    source = RosterSource(kind=RosterSourceKind.GRAPH, path=ttl_path)
    channels = channels_from_corpus(parsed, source)

    meta = {
        "corpus_sha256": digest,
        "corpus_filename": ttl_path.name,
        "binding_count": len(parsed.binding_rows),
        # A device is a subject with at least one binding row, which is what the
        # store's device census counts: a bound :Resource, DISTINCT because a
        # device with several channels binds several times.
        "device_count": len({row.device_uri for row in parsed.binding_rows}),
        # Already the pruned device taxonomy: parse_corpus builds class_rows
        # through prune_device_taxonomy, so this is its length by construction.
        "class_count": len(parsed.class_rows),
        "signal_count": parsed.signal_count,
        "section_count": len(parsed.section_codes),
    }

    report = build_from_rows(parsed.binding_rows, parsed.class_rows, channels, index_path, meta)
    # DEBUG: the callers own the operator-facing line (the build's progress
    # line names the corpus, the ``build-index`` verb prints its own summary),
    # and a build keeps absolute paths out of its INFO view.
    logger.debug(
        "Built the channel search index at %s in %.2f s: %d bindings, %d devices, "
        "%d classes, %d signals, %d sections, %d channels (corpus %s).",
        report.path,
        time.perf_counter() - started,
        report.binding_count,
        report.device_count,
        report.class_count,
        report.signal_count,
        report.section_count,
        report.channel_count,
        report.corpus_sha256[:12],
    )
    return report
