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
        a source attached. A settable record carries the readback the corpus
        itself states for it -- the ``<stem>Monitor`` binding beside its
        ``<stem>Setpoint`` binding on the same device
        (:func:`_corpus_readbacks`); the records the corpus pairs nothing
        with are left to the address-grammar pass in
        :mod:`osprey.channel_roster.pairing`, which never overrides a
        readback the source stated.
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
        readbacks = _corpus_readbacks(graph, writes, reads, bindings)
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

    records = _records(bindings, source, writes, reads, readbacks)
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
