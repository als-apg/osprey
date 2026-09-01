"""Enumerate a facility's channels from its knowledge-graph corpus.

The graph paradigm's roster is the Turtle corpus the build stages for the graph
store, not the store itself: the store is a disposable mirror seeded from that
file, and it is not dialed here (nor is it reachable at build time on a host
that has not started it yet).

Direction comes free on this source. Every ``ChannelBinding`` in the corpus
carries its address as ``narad_p:fullPv`` and points at exactly one of
``narad_p:writesSignal`` / ``narad_p:readsSignal``, so the corpus *states* which
channels are settable rather than leaving it to be inferred from address
grammar. That is the whole reason the graph is preferred over the write-limits
projection: the projection gates a subset, the corpus enumerates the machine.

The NARAD IRIs are spelled as literal strings and ``rdflib`` is imported inside
the reader, exactly as :mod:`osprey.deployment.channel_snapshot` does and for
the same two reasons: importing anything from
``osprey.services.facility_knowledge`` executes that package's ``__init__`` and
pulls ``osprey.services.qmd`` into the build's import graph, and a deployment
that never reads a corpus should not pay for rdflib at import time.

Failure is data, never an exception: a corpus that cannot be read comes back as
a :class:`~osprey.channel_roster.records.RosterAbsence` naming the path and the
underlying error. Which absence it is says whether the corpus is *absent* or
*broken* -- ``MISSING_SOURCE`` for a path that is not there, ``CORRUPT_SOURCE``
for one that is there and will not parse (an unimportable rdflib included) --
because consumers apply opposite rules to those two, and a consumer that had to
re-``stat`` the file to tell them apart would be reading a filesystem that has
moved on since. Whether either is fatal stays the consumer's call: the build's
three-way rule and the web routes apply their own.
"""

from __future__ import annotations

from osprey.channel_roster.records import (
    ChannelDirection,
    ChannelRecord,
    RosterAbsence,
    RosterAbsenceReason,
    RosterResult,
    RosterSource,
)
from osprey.utils.logger import get_logger

logger = get_logger("channel_roster.graph")

#: The ``narad_p:`` property namespace, spelled here rather than imported --
#: see the module docstring. ``facility_knowledge/seeder/ttl_seeder.py`` spells
#: the same prefix, and ``seeder/graph_seeder.py``'s ``NARAD_PREFIXES`` is the
#: prefix table of record.
_NARAD_P = "https://narad.example.org/property/"

#: The control-system address of a binding, and the roster's membership: one
#: record per ``fullPv`` literal.
_FULL_PV_IRI = _NARAD_P + "fullPv"

#: A binding that drives a signal -- a settable channel.
_WRITES_SIGNAL_IRI = _NARAD_P + "writesSignal"

#: A binding that observes a signal -- a readable channel.
_READS_SIGNAL_IRI = _NARAD_P + "readsSignal"

#: Said to an operator when rdflib is missing. rdflib is a core dependency, so
#: an environment without it is incomplete rather than differently configured.
RDFLIB_MISSING_DETAIL = (
    "rdflib is not importable, so the corpus cannot be parsed; rdflib is a core "
    "dependency, so this environment is incomplete -- reinstall it with: "
    "pip install --upgrade osprey-framework"
)


def read_graph_roster(source: RosterSource) -> RosterResult:
    """Read every channel the knowledge-graph corpus declares.

    Args:
        source: The resolved corpus to read, as
            :func:`~osprey.channel_roster.sources.resolve_roster_source`
            settled it.

    Returns:
        A :class:`~osprey.channel_roster.records.RosterResult` holding one
        record per ``narad_p:fullPv`` literal, sorted by address and carrying
        the direction the corpus states; or one carrying a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.MISSING_SOURCE`
        absence when the configured corpus is not there, a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.CORRUPT_SOURCE`
        one when it is there and cannot be parsed, or an
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.EMPTY_SOURCE`
        one when it reads cleanly and binds no channel -- an unseeded corpus is
        a staging gap, and serving it as an empty facility would be a lie with
        a source attached. Readbacks are not
        assigned here -- pairing is one heuristic applied to both sources, and
        lives in :mod:`osprey.channel_roster.pairing`.
    """
    if not source.path.exists():
        # Not there is not the same as unreadable: a corpus a build has not
        # staged yet is an absent facility, and the consumer that refuses on a
        # broken source stays browse-only on this one. Told apart here rather
        # than by every consumer re-``stat``ing the file behind the answer.
        # ``exists`` rather than ``is_file``: a path that names a directory IS
        # there and cannot be parsed, which is the other half of the pair.
        from osprey.channel_roster.sources import GRAPH_CORPUS_CONFIG_KEYS

        logger.warning(
            f"The knowledge-graph corpus {source.for_display()} is not there, so this "
            "build enumerates no channels from the graph."
        )
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.MISSING_SOURCE,
                path=source.path,
                spelled=source.spelled,
                config_keys=GRAPH_CORPUS_CONFIG_KEYS,
            )
        )

    try:
        from rdflib import Graph, Literal, URIRef
    except ImportError:
        logger.warning(
            f"The knowledge-graph corpus at {source.path} cannot be read: {RDFLIB_MISSING_DETAIL}."
        )
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.CORRUPT_SOURCE,
                path=source.path,
                spelled=source.spelled,
                detail=RDFLIB_MISSING_DETAIL,
            )
        )

    try:
        graph = Graph()
        # The format is forced rather than guessed from the extension, as every
        # other reader of this key forces it -- the deploy's n10s import, the
        # TTL seeder, the channel snapshot -- and rdflib would otherwise hand a
        # corpus named ``.rdf`` to its XML parser.
        graph.parse(str(source.path), format="turtle")
        writes = set(graph.subjects(URIRef(_WRITES_SIGNAL_IRI), None))
        reads = set(graph.subjects(URIRef(_READS_SIGNAL_IRI), None))
        bindings = [
            (str(address), binding)
            for binding, address in graph.subject_objects(URIRef(_FULL_PV_IRI))
            if isinstance(address, Literal) and str(address)
        ]
    except Exception as e:
        logger.warning(f"The knowledge-graph corpus at {source.path} could not be read ({e}).")
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.CORRUPT_SOURCE,
                path=source.path,
                spelled=source.spelled,
                detail=str(e),
            )
        )

    records = tuple(
        ChannelRecord(
            address=address,
            source=source,
            direction=_direction(binding, writes, reads),
        )
        for address, binding in sorted(bindings, key=lambda binding: binding[0])
    )
    if not records:
        logger.warning(
            f"The knowledge-graph corpus at {source.path} parsed cleanly and declares no "
            "channel bindings."
        )
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.EMPTY_SOURCE,
                path=source.path,
                spelled=source.spelled,
            )
        )
    return RosterResult(records=records, source=source)


def _direction(binding: object, writes: set, reads: set) -> ChannelDirection | None:
    """Return which way ``binding`` points, or ``None`` when the corpus cannot say.

    A binding that claims both directions is reported as unknown rather than as
    settable: the vocabulary allows exactly one, so a corpus asserting both has
    drifted, and calling such a channel writable on a guess is the one error
    that reaches hardware.
    """
    is_write = binding in writes
    is_read = binding in reads
    if is_write and not is_read:
        return "write"
    if is_read and not is_write:
        return "read"
    return None
