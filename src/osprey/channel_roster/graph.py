"""Enumerate a facility's channels from the search index the build wrote.

The graph paradigm's roster is the flat search index a build derives from the
Turtle corpus it stages for the graph store -- neither the store nor the
corpus. The store is a disposable mirror seeded from that file, and it is not
dialed here (nor is it reachable at build time on a host that has not started
it yet). The corpus is parsed once, when the index is written, rather than on
every read: a roster read is one ``SELECT`` over the index's ``channels``
table, and this path never imports rdflib.

Direction comes free on this source. Every ``ChannelBinding`` in the corpus
carries its address as ``narad_p:fullPv`` and points at exactly one of
``narad_p:writesSignal`` / ``narad_p:readsSignal``, so the corpus *states*
which channels are settable rather than leaving it to be inferred from address
grammar, and the build carries that statement into the index's ``direction``
column. That is the whole reason the graph is preferred over the write-limits
projection: the projection gates a subset, the corpus enumerates the machine.

The rules that turn a corpus into those channels still live in this module --
:func:`_records`, :func:`_corpus_readbacks`, :func:`_binding_field` and
:func:`_direction`. The builder applies them when it writes the index
(:func:`osprey.services.channel_finder.graph_index.builder.channels_from_corpus`),
so the rules are stated once rather than twice where the two copies could
drift, and the rows this reader returns are the records it used to build
itself. They import ``rdflib`` inside the function that needs it, exactly as
:mod:`osprey.deployment.channel_snapshot` does and for the same two reasons:
importing anything from ``osprey.services.facility_knowledge`` executes that
package's ``__init__`` and pulls ``osprey.services.qmd`` into the build's
import graph, and a deployment that never reads a corpus should not pay for
rdflib at import time. ``duckdb`` is lazy for the same reason, inside
:func:`~osprey.services.channel_finder.graph_index.reader.open_graph_index`.

Failure is data, never an exception: an index that cannot be read comes back as
a :class:`~osprey.channel_roster.records.RosterAbsence` naming the path and the
underlying error. Which absence it is says whether the index is *absent* or
*broken* -- ``MISSING_SOURCE`` for one no build has written, ``CORRUPT_SOURCE``
for one that is there and unreadable, a stale schema version included --
because consumers apply opposite rules to those two, and a consumer that had to
re-``stat`` the file to tell them apart would be reading a filesystem that has
moved on since. Both name the command that writes the index, which is the only
remedy either has. Whether either is fatal stays the consumer's call: the
build's three-way rule and the web routes apply their own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osprey.channel_roster.records import (
    ChannelDirection,
    ChannelRecord,
    RosterAbsence,
    RosterAbsenceReason,
    RosterResult,
    RosterSource,
)
from osprey.deployment.graphdb_service import GRAPHDB_BUILD_INDEX_COMMAND
from osprey.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only; the reader stays a lazy import
    from osprey.services.channel_finder.graph_index.reader import GraphIndexAbsence

logger = get_logger("channel_roster.graph")

#: The roster as the build wrote it: one row per address, already collapsed
#: and voted on by :func:`_records`, carrying the readback the corpus stated.
#: Read in address order, so the reader hands back the order it always handed
#: back without sorting a few thousand rows a second time.
_CHANNELS_SELECT = "SELECT address, direction, readback FROM channels ORDER BY address"

#: Said to an operator whose index is not there or was written for another
#: schema version. Both have one remedy -- write the index -- and ``osprey
#: build`` writes it as part of a render, so an operator who is rendering
#: anyway does not need the standalone command.
BUILD_INDEX_REMEDY = f"Build it with `{GRAPHDB_BUILD_INDEX_COMMAND}`, or re-run `osprey build`"

#: The ``narad_p:`` property namespace, spelled here rather than imported --
#: see the module docstring. ``facility_knowledge/seeder/ttl_seeder.py`` spells
#: the same prefix, and ``seeder/graph_seeder.py``'s ``NARAD_PREFIXES`` is the
#: prefix table of record.
_NARAD_P = "https://narad.example.org/property/"

#: A device's bindings. The corpus groups a device's channels under its
#: device node, and that grouping is what pairs a setpoint with the readback
#: reporting it (see :func:`_corpus_readbacks`).
_HAS_BINDING_IRI = _NARAD_P + "hasBinding"

#: A binding's identifier, ``narad:binding:<facility>:<system>:<family>:
#: <index>:<Field>:val`` -- the ``<Field>`` token is the binding's field name.
_BINDING_ID_IRI = _NARAD_P + "bindingId"

#: The NARAD field vocabulary's setpoint/readback pair: a device's
#: ``<stem>Setpoint`` binding is reported by its ``<stem>Monitor`` binding
#: (``Setpoint``/``Monitor`` for a magnet's current, ``GapSetpoint``/
#: ``GapMonitor`` for an insertion device's gap). The same two field names the
#: facility-knowledge seeder reads off a binding
#: (``seeder/ttl_seeder._binding_channel_name``). This is the corpus stating
#: a pair; the ``:SP``/``:RB`` address grammar in :mod:`.pairing` is the
#: fallback for the records a corpus groups nothing with.
SETPOINT_FIELD_SUFFIX = "Setpoint"
MONITOR_FIELD_SUFFIX = "Monitor"

#: The trailing token of a binding id names its value slot, not its field.
_BINDING_ID_VALUE_TOKEN = "val"


def read_graph_roster(source: RosterSource) -> RosterResult:
    """Read every channel the facility's search index enumerates.

    Args:
        source: The resolved index to read, as
            :func:`~osprey.channel_roster.sources.resolve_roster_source`
            settled it.

    Returns:
        A :class:`~osprey.channel_roster.records.RosterResult` holding one
        record per address, in address order, carrying the direction the
        corpus stated; or one carrying a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.MISSING_SOURCE`
        absence when no build has written the index, a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.CORRUPT_SOURCE`
        one when it is there and cannot be read -- a file the driver refuses,
        one written for another schema version, a row that is not a channel --
        or an
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.EMPTY_SOURCE`
        one when it reads cleanly and enumerates no channel: an index built
        from an unseeded corpus is a staging gap, and serving it as an empty
        facility would be a lie with a source attached. A settable record
        carries the readback the corpus itself stated for it -- the
        ``<stem>Monitor`` binding beside its ``<stem>Setpoint`` binding on the
        same device (:func:`_corpus_readbacks`); the records the corpus pairs
        nothing with are left to the address-grammar pass in
        :mod:`osprey.channel_roster.pairing`, which never overrides a readback
        the source stated.
    """
    from osprey.services.channel_finder.graph_index.reader import (
        GraphIndexAbsence,
        open_graph_index,
    )

    opened = open_graph_index(source.path)
    if isinstance(opened, GraphIndexAbsence):
        return _absent_index(source, opened)

    try:
        cursor = opened.cursor()
        try:
            rows = cursor.execute(_CHANNELS_SELECT).fetchall()
        finally:
            cursor.close()
        records = tuple(
            ChannelRecord(address=address, source=source, direction=direction, readback=readback)
            for address, direction, readback in rows
        )
    except Exception as e:  # noqa: BLE001 - an index we cannot read is data, not a crash
        # An index carrying our meta row can still carry a row that is not a
        # channel -- a readback on a readable address, an empty address. The
        # file is there and this build cannot use it, which is the corrupt
        # half of the pair, not a traceback out of the middle of a build.
        logger.warning(f"The channel search index at {source.path} could not be read ({e}).")
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.CORRUPT_SOURCE,
                path=source.path,
                spelled=source.spelled,
                detail=str(e),
            )
        )
    finally:
        opened.close()

    if not records:
        logger.warning(
            f"The channel search index at {source.path} was read and enumerates no channels."
        )
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.EMPTY_SOURCE,
                path=source.path,
                spelled=source.spelled,
            )
        )
    return RosterResult(records=records, source=source)


def _absent_index(source: RosterSource, absence: GraphIndexAbsence) -> RosterResult:
    """Turn the index reader's absence into the roster's, with the remedy on it.

    Not there is not the same as unreadable: an index no build has written yet
    is an absent facility, and the consumer that refuses on a broken source
    stays browse-only on this one. Told apart here rather than by every
    consumer re-``stat``ing the file behind the answer. A stale schema version
    is the broken half -- the file is there and this build cannot read it --
    and it carries the same remedy as an absent index, because writing the
    index again is what fixes either.

    Args:
        source: The index that was opened.
        absence: What
            :func:`~osprey.services.channel_finder.graph_index.reader.open_graph_index`
            said about it; its ``detail`` is carried verbatim.
    """
    from osprey.channel_roster.sources import GRAPH_CORPUS_CONFIG_KEYS

    if absence.reason == "missing":
        logger.warning(
            f"The channel search index {source.for_display()} is not there, so this "
            "build enumerates no channels from the graph."
        )
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.MISSING_SOURCE,
                path=source.path,
                spelled=source.spelled,
                config_keys=GRAPH_CORPUS_CONFIG_KEYS,
                detail=f"{BUILD_INDEX_REMEDY}.",
            )
        )

    # The absence's own sentence, then this reader's remedy: the roster's
    # template ends the sentence, so the driver's own terminator is dropped
    # rather than doubled. DuckDB ends some of its messages with "!", which
    # would otherwise render as "file!.".
    detail = absence.detail.rstrip(".!?")
    logger.warning(f"The channel search index at {source.path} could not be read ({detail}).")
    return RosterResult(
        absence=RosterAbsence(
            reason=RosterAbsenceReason.CORRUPT_SOURCE,
            path=source.path,
            spelled=source.spelled,
            detail=f"{detail}. {BUILD_INDEX_REMEDY}"
            if absence.reason == "schema_mismatch"
            else detail,
        )
    )


def _records(
    bindings: list[tuple[str, object]],
    source: RosterSource,
    writes: set,
    reads: set,
    readbacks: dict[str, str],
) -> tuple[ChannelRecord, ...]:
    """One record per address, in address order.

    The roster is a namespace: two bindings carrying one ``fullPv`` are one
    channel, however many devices the corpus hangs it under (a timing
    system's delay generator is bound once per device it serves; a facility
    corpus can carry a fifth of its bindings this way). Every consumer keys
    on the address -- the plan-device document names each device for it and
    the build refuses a name claimed twice, the manifest serves it once -- so
    the collapse happens here, once, rather than in each of them.

    Direction is the one the address's bindings agree on. A binding that
    states none abstains; bindings that disagree leave the address with no
    direction, the same honest unknown a single binding claiming both ways
    gets. A settable address carries the readback the corpus stated for it,
    provided the readback ADDRESS resolves readable too -- a pair is stated
    between bindings, and the binding it names as the monitor may share its
    address with a write binding elsewhere in the corpus.
    """
    votes: dict[str, set[ChannelDirection]] = {}
    for address, binding in bindings:
        directions = votes.setdefault(address, set())
        direction = _direction(binding, writes, reads)
        if direction is not None:
            directions.add(direction)
    resolved: dict[str, ChannelDirection | None] = {
        address: next(iter(directions)) if len(directions) == 1 else None
        for address, directions in votes.items()
    }

    records: list[ChannelRecord] = []
    for address in sorted(resolved):
        direction = resolved[address]
        readback = readbacks.get(address) if direction == "write" else None
        if readback is not None and resolved.get(readback) != "read":
            readback = None
        records.append(
            ChannelRecord(address=address, source=source, direction=direction, readback=readback)
        )
    return tuple(records)


def _binding_field(binding_id: str) -> str | None:
    """The field name a binding id carries, or ``None`` when it carries none.

    ``narad:binding:als:SR:BEND:0:Setpoint:val`` names ``Setpoint``: the
    tokens are colon-separated, and the trailing ``val`` is the binding's
    value slot rather than its field, so it is dropped when present.
    """
    tokens = binding_id.split(":")
    if tokens and tokens[-1] == _BINDING_ID_VALUE_TOKEN:
        tokens = tokens[:-1]
    if len(tokens) < 2 or not tokens[-1]:
        return None
    return tokens[-1]


def _corpus_readbacks(
    graph: object, writes: set, reads: set, bindings: list[tuple[str, object]]
) -> dict[str, str]:
    """Return ``setpoint address -> readback address`` for every pair the corpus states.

    A pair is stated by the device grouping: a device (``narad_p:hasBinding``)
    carrying a write binding whose field is ``<stem>Setpoint`` and a read
    binding whose field is ``<stem>Monitor``. Both halves are checked against
    the direction sets the corpus declares -- a ``Monitor`` the corpus calls
    settable is drift, not a readback -- and a binding without a device, an
    id or an address states no pair. Where two devices state different
    readbacks for one setpoint address, the first in sorted device order
    wins, so the answer does not depend on parse order.

    Args:
        graph: The parsed corpus.
        writes: Bindings carrying ``writesSignal``.
        reads: Bindings carrying ``readsSignal``.
        bindings: ``(address, binding)`` for every binding with an address.
    """
    from rdflib import URIRef

    address_of = {binding: address for address, binding in bindings}
    fields_by_device: dict[object, dict[str, object]] = {}
    for device, binding in graph.subject_objects(URIRef(_HAS_BINDING_IRI)):
        binding_id = graph.value(binding, URIRef(_BINDING_ID_IRI))
        field = _binding_field(str(binding_id)) if binding_id is not None else None
        if field is None or binding not in address_of:
            continue
        fields_by_device.setdefault(device, {})[field] = binding

    readbacks: dict[str, str] = {}
    for device in sorted(fields_by_device, key=str):
        fields = fields_by_device[device]
        for field, setpoint in fields.items():
            if not field.endswith(SETPOINT_FIELD_SUFFIX):
                continue
            if setpoint not in writes or setpoint in reads:
                continue
            stem = field[: -len(SETPOINT_FIELD_SUFFIX)]
            monitor = fields.get(stem + MONITOR_FIELD_SUFFIX)
            if monitor is None or monitor not in reads or monitor in writes:
                continue
            setpoint_address = address_of[setpoint]
            readback_address = address_of[monitor]
            if setpoint_address == readback_address:
                continue
            readbacks.setdefault(setpoint_address, readback_address)
    return readbacks


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
