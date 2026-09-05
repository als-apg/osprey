"""The demo-shaped corpus every graph-paradigm test is written against.

The graph paradigm has no database file to point a test at, so what stands in
for one is this module: a synthesized corpus, the search index built from it
the way ``osprey build`` builds one, and a fake store context that answers the
device read by looking at the Cypher it was handed.

The corpus has exactly one source. :func:`synth_rows` lays a grid of sections,
device families and ordinals out as channel rows, and everything else here is
derived from those rows: the ontology and its rollups, the census the badge
shows, the pages the finder slices, the facets the rail draws, and the device
the card is built from. Nothing states a count that the rows do not already
say, so a change to the grid moves every number with it rather than leaving a
stale literal behind.

There is one copy of that corpus on purpose. The route tests, the launcher in
``tests/interfaces/conftest.py`` that boots a real graph-mode server, and the
browser and visual lanes that photograph it all draw from here, so a number
asserted in one lane means the same thing in every other.

The fake is keyed by the query text itself rather than by call order, and the
match is exact against the imported constants, so a query the fake has no
answer for fails loudly instead of being served somebody else's rows — a route
that grows a store read is a failing assertion here, not a silently wrong page.
It also records the thread each call arrived on, so the "store reads never run
on the event loop" contract can be asserted rather than assumed, and the
parameters each read was given, so a route's request contract can be asserted
from the store side.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from math import ceil
from pathlib import Path
from typing import Any, NamedTuple

from osprey.interfaces.channel_finder.database_api import _GRAPH_EXPLORE_MAX_ROWS
from osprey.mcp_server.graph.server_context import QueryResult
from osprey.services.channel_finder.graph_index.reader import (
    GraphIndex,
    GraphIndexAbsence,
    open_graph_index,
)
from osprey.services.channel_finder.graph_index.taxonomy import prune_device_taxonomy
from osprey.services.channel_finder.graph_queries import GRAPH_DEVICE_CYPHER

__all__ = [
    "AGENT_FACET_CAP",
    "BINDING_CLASS_URI",
    "DEMO_CLASS_COUNT",
    "DEMO_CLASS_ROWS",
    "DEMO_CORPUS_FILENAME",
    "DEMO_DEVICES",
    "DEMO_DEVICE_CLASSES",
    "DEMO_DEVICE_ROW",
    "DEMO_FACETS",
    "DEMO_FAMILIES",
    "DEMO_INDEX_FILENAME",
    "DEMO_PAGE_COUNT",
    "DEMO_ROWS",
    "DEMO_SECTIONS",
    "DEMO_STATISTICS",
    "DEMO_STORE_URI",
    "DEV_NAMESPACE",
    "ORDINALS_PER_FAMILY",
    "SEARCH_FACET_CAP",
    "SEARCH_PAGE_SIZE",
    "SEM_NAMESPACE",
    "SIGNAL_CLASS_URI",
    "DeviceSpec",
    "FakeGraphContext",
    "Family",
    "binding_uri",
    "build_demo_index",
    "class_uri",
    "demo_class_rows",
    "demo_context",
    "demo_facets",
    "demo_index_path",
    "demo_page",
    "demo_turtle",
    "device_card",
    "device_uri",
    "install_graph_paradigm",
    "open_demo_index",
    "signal_uri",
    "synth_rows",
]

#: Namespace the demo corpus mints its class URIs in.
SEM_NAMESPACE = "https://narad.example.org/schema/shared_semantics/"

#: Namespace the demo corpus mints device URIs in: the same host as the schema,
#: under a data path, which is how an n10s import keeps the individuals it
#: loads apart from the classes they are typed by.
DEV_NAMESPACE = "https://narad.example.org/data/shared_semantics/"

#: The bolt URI the demo store is reachable at, matching the ``services.graphdb``
#: block the test launcher writes.
DEMO_STORE_URI = "bolt://localhost:7687"

#: Rows one page of the finder holds, as the index slices it.
SEARCH_PAGE_SIZE = 50

#: Entries a facet list carries before the explorer calls it clipped. Read from
#: the explorer's own cap rather than restated.
SEARCH_FACET_CAP = _GRAPH_EXPLORE_MAX_ROWS

#: Entries a facet list carries before the *agent* tool clips it, which is a far
#: tighter budget than a browser rail's. Spelled rather than imported: reaching
#: into ``osprey.mcp_server`` would pull fastmcp and the agent server into the
#: interface lane. The tool's own tests pin this against its ``FACET_CAP``.
AGENT_FACET_CAP = 10


def class_uri(name: str) -> str:
    """Return the class URI the demo corpus would hold for *name*.

    Args:
        name: Bare class name, e.g. ``"Quadrupole"``.

    Returns:
        The fully qualified URI under :data:`SEM_NAMESPACE`.
    """
    return f"{SEM_NAMESPACE}{name}"


def device_uri(source_name: str) -> str:
    """Return the URI the demo corpus holds for the device named *source_name*.

    Args:
        source_name: The device's control-system name, e.g. ``"SR01C___QFA1___"``.

    Returns:
        The fully qualified device URI under :data:`DEV_NAMESPACE`.
    """
    return f"{DEV_NAMESPACE}{source_name}"


def signal_uri(name: str) -> str:
    """Return the URI the demo corpus holds for the semantic signal *name*.

    Args:
        name: Bare signal name, e.g. ``"current"``.

    Returns:
        The fully qualified signal URI under :data:`SEM_NAMESPACE`.
    """
    return f"{SEM_NAMESPACE}{name}"


def binding_uri(full_pv: str) -> str:
    """Return the URI the demo corpus holds for the binding addressing *full_pv*.

    Args:
        full_pv: The channel address, e.g. ``"SR01C___QFA1___AM00"``.

    Returns:
        The fully qualified binding URI under :data:`DEV_NAMESPACE`.
    """
    return f"{DEV_NAMESPACE}binding/{full_pv}"


# ---------------------------------------------------------------------------
# The grid the corpus is synthesized from
# ---------------------------------------------------------------------------


class Family(NamedTuple):
    """One device family, repeated in every section of the demo corpus."""

    #: Address stem an ALS-style device name is built on, before its ordinal.
    stem: str
    #: Bare name of the class every device of the family is typed by.
    class_name: str
    #: The facility system the family belongs to.
    system: str
    #: Bare name of the signal the family's principal channel carries.
    signal: str
    #: Whether an operator sets the family as well as reads it, which is what
    #: gives its devices a second channel with a ``WRITESSIGNAL`` edge.
    settable: bool = False
    #: A second readback on its own signal, or ``None`` for a family with one.
    second_signal: str | None = None
    #: What the device card calls the family's field.
    field: str = ""
    #: What the device card says the family is.
    description: str = ""


#: The nine families the grid repeats. Six are set as well as read, which is
#: what puts a setpoint beside their readback; one reads two signals through
#: two addresses; the rest read one.
DEMO_FAMILIES: tuple[Family, ...] = (
    Family("QFA", "Quadrupole", "MG", "current", True, None, "Current", "Quadrupole, family A"),
    Family("QDA", "Quadrupole", "MG", "current", True, None, "Current", "Quadrupole, family D"),
    Family("SF", "Sextupole", "MG", "current", True, None, "Current", "Focusing sextupole"),
    Family("BEND", "Dipole", "MG", "current", True, None, "Current", "Storage ring bend magnet"),
    Family(
        "HCM",
        "HorizontalCorrector",
        "MG",
        "current",
        True,
        None,
        "Current",
        "Horizontal corrector",
    ),
    Family(
        "VCM", "VerticalCorrector", "MG", "current", True, None, "Current", "Vertical corrector"
    ),
    Family(
        "BPM",
        "BeamPositionMonitor",
        "DI",
        "beamPositionX",
        False,
        "beamPositionY",
        "Position",
        "Beam position monitor",
    ),
    Family("IPUMP", "IonPump", "VA", "pressure", False, None, "Pressure", "Ion pump"),
    Family("GAUGE", "VacuumGauge", "VA", "pressure", False, None, "Pressure", "Vacuum gauge"),
)

#: The one family off the grid: a single cavity, whose one address is both read
#: and set, which is the only way the ``RW`` direction gets exercised.
_CAVITY = Family("RFCAV", "RFCavity", "RF", "cavityVoltage", False, None, "Voltage", "RF cavity")

#: The sections the grid covers.
DEMO_SECTIONS = ("SR01C", "SR02C", "SR03C")

#: Devices each family has per section. Four is what carries the corpus past
#: one page of :data:`SEARCH_PAGE_SIZE`, so a finder reading this fixture
#: really paginates instead of only claiming it would.
ORDINALS_PER_FAMILY = 4

#: What every system code stands for, as the device card spells it.
_SYSTEM_DESCRIPTIONS = {
    "MG": "Storage ring magnets",
    "DI": "Storage ring diagnostics",
    "VA": "Storage ring vacuum",
    "RF": "Storage ring RF",
}

#: What the device card calls the machine the corpus describes.
_RING_DESCRIPTION = "Storage ring"

#: The ontology the corpus declares: bare class name to its parents and its
#: alternative labels. It is wider than the grid on purpose — ``Undulator``,
#: ``RFAmplifier`` and ``CurrentMonitor`` bind no device, so the build prunes
#: them the way a real ontology's unbound leaves are pruned, while the branch
#: above a pruned leaf survives. The last two are real classes about signals and
#: bindings rather than devices, which is where n10s reads its labels from.
_ONTOLOGY: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "AcceleratorDevice": ((), ()),
    "Magnet": (("AcceleratorDevice",), ()),
    "Dipole": (("Magnet",), ("BEND",)),
    "Quadrupole": (("Magnet",), ("QUAD",)),
    "Sextupole": (("Magnet",), ("SEXT",)),
    "Corrector": (("Magnet",), ()),
    "HorizontalCorrector": (("Corrector",), ()),
    "VerticalCorrector": (("Corrector",), ()),
    "Diagnostic": (("AcceleratorDevice",), ()),
    "BeamPositionMonitor": (("Diagnostic",), ("BPM",)),
    "CurrentMonitor": (("Diagnostic",), ()),
    "VacuumDevice": (("AcceleratorDevice",), ()),
    "IonPump": (("VacuumDevice",), ()),
    "VacuumGauge": (("VacuumDevice",), ()),
    "RFDevice": (("AcceleratorDevice",), ()),
    "RFCavity": (("RFDevice",), ()),
    "RFAmplifier": (("RFDevice",), ()),
    "InsertionDevice": (("AcceleratorDevice",), ()),
    "Undulator": (("InsertionDevice",), ()),
    "SemanticSignal": ((), ()),
    "ChannelBinding": ((), ()),
}

#: The edges a binding carries, spelled as the corpus and the index spell them.
_EDGE_READ = "READSSIGNAL"
_EDGE_WRITE = "WRITESSIGNAL"


class DeviceSpec(NamedTuple):
    """One synthesized device: where it sits and what family it belongs to."""

    name: str
    section: str
    ordinal: int
    family: Family


def _device_name(section: str, stem: str, ordinal: int) -> str:
    """Return the ALS-style name the *ordinal*-th *stem* takes in *section*."""
    return f"{section}___{f'{stem}{ordinal}':_<7}"


def _row(
    device: str,
    section: str,
    system: str,
    suffix: str,
    signal: str | None,
    edges: list[str],
    description: str,
) -> dict[str, Any]:
    """Build one channel row shaped as the index's search projects it."""
    return {
        "fullPv": f"{device}{suffix}",
        "description": description,
        "device": device,
        "device_uri": device_uri(device),
        "section": section,
        "system": system,
        "edges": list(edges),
        "signals": [{"uri": signal_uri(signal), "name": signal}] if signal else [],
    }


def _synthesize() -> tuple[list[dict[str, Any]], dict[str, DeviceSpec]]:
    """Lay the grid out as channel rows and the devices they address.

    Returns:
        The rows, ordered by ``fullPv`` as the index returns them, and one
        :class:`DeviceSpec` per device the rows name.
    """
    rows: list[dict[str, Any]] = []
    devices: dict[str, DeviceSpec] = {}

    def add(spec: DeviceSpec, suffix: str, signal: str | None, edges: list[str], what: str) -> None:
        devices[spec.name] = spec
        rows.append(_row(spec.name, spec.section, spec.family.system, suffix, signal, edges, what))

    for section in DEMO_SECTIONS:
        for family in DEMO_FAMILIES:
            for ordinal in range(1, ORDINALS_PER_FAMILY + 1):
                spec = DeviceSpec(
                    _device_name(section, family.stem, ordinal), section, ordinal, family
                )
                add(spec, "AM00", family.signal, [_EDGE_READ], f"{family.signal} readback")
                if family.settable:
                    add(spec, "SP00", family.signal, [_EDGE_WRITE], f"{family.signal} setpoint")
                if family.second_signal:
                    add(
                        spec,
                        "AM01",
                        family.second_signal,
                        [_EDGE_READ],
                        f"{family.second_signal} readback",
                    )

    # One address that is read and set through the same channel, and one that
    # carries no semantic signal at all: the two rows a facet on direction has
    # nothing to show unless the corpus actually holds them.
    cavity = DeviceSpec(
        _device_name(DEMO_SECTIONS[0], _CAVITY.stem, 1), DEMO_SECTIONS[0], 1, _CAVITY
    )
    add(
        cavity,
        "AM00",
        _CAVITY.signal,
        [_EDGE_READ, _EDGE_WRITE],
        f"{_CAVITY.signal}, read and set on one address",
    )
    add(
        devices[_device_name(DEMO_SECTIONS[-1], "GAUGE", ORDINALS_PER_FAMILY)],
        "ST00",
        None,
        [],
        "controller status word, no semantic signal",
    )

    rows.sort(key=lambda row: row["fullPv"])
    return rows, devices


def synth_rows() -> list[dict[str, Any]]:
    """Return the corpus as channel rows, freshly synthesized.

    Every other number this module publishes is derived from these rows, so a
    test that wants an expected value derives it from here too rather than
    restating it.

    Returns:
        One row per ``(device, binding)`` pair, ordered by ``fullPv``.
    """
    return _synthesize()[0]


#: The synthesized corpus, built once: the rows, and the device each addresses.
DEMO_ROWS, DEMO_DEVICES = _synthesize()

#: What class each device of the corpus is typed by, by bare class name.
DEMO_DEVICE_CLASSES: dict[str, str] = {
    name: spec.family.class_name for name, spec in DEMO_DEVICES.items()
}

#: Pages the finder slices the whole corpus into. Above one by construction:
#: a finder that never paginates in its own fixture cannot show that it does.
DEMO_PAGE_COUNT = ceil(len(DEMO_ROWS) / SEARCH_PAGE_SIZE)


def demo_page(page: int = 1) -> list[dict[str, Any]]:
    """Return the rows an unfiltered search puts on *page*.

    Args:
        page: One-based page number, as the route counts them.

    Returns:
        That page's slice of the corpus, empty past the last page.
    """
    return DEMO_ROWS[(page - 1) * SEARCH_PAGE_SIZE : page * SEARCH_PAGE_SIZE]


def _class_ancestors(name: str) -> list[str]:
    """Return the bare names of *name* and every class it rolls up to."""
    seen: list[str] = []
    pending = [name]
    while pending:
        current = pending.pop()
        if current in seen or current not in _ONTOLOGY:
            continue
        seen.append(current)
        pending.extend(_ONTOLOGY[current][0])
    return seen


def demo_class_rows() -> list[dict[str, Any]]:
    """Return the ontology, with the counts the corpus's devices give it.

    ``direct`` counts the devices typed as the class itself and ``rollup``
    counts those plus everything under it, both from :data:`DEMO_ROWS` — so a
    branch nothing is typed as carries a number only its subclasses can explain,
    and a leaf the grid never binds carries zero and gets pruned at build time.

    Returns:
        One row per declared class, shaped as an ontology read returns them:
        its ``uri``, ``altLabel``, ``parents``, ``rollup`` and ``direct``.
    """
    direct: dict[str, int] = {}
    for class_name in DEMO_DEVICE_CLASSES.values():
        direct[class_name] = direct.get(class_name, 0) + 1

    rollup: dict[str, int] = {}
    for class_name, count in direct.items():
        for ancestor in _class_ancestors(class_name):
            rollup[ancestor] = rollup.get(ancestor, 0) + count

    return [
        {
            "uri": class_uri(name),
            "altLabel": list(alt_labels),
            "parents": [class_uri(parent) for parent in parents],
            "rollup": rollup.get(name, 0),
            "direct": direct.get(name, 0),
        }
        for name, (parents, alt_labels) in _ONTOLOGY.items()
    ]


#: The ontology as the corpus declares it, counts and all.
DEMO_CLASS_ROWS: list[dict[str, Any]] = demo_class_rows()

#: The device classes that survive pruning — the taxonomy an operator came for,
#: which is what the build keeps and what the badge counts. Run through the
#: pruner itself rather than counted by hand, so the two cannot disagree.
DEMO_CLASS_COUNT = len(prune_device_taxonomy(DEMO_CLASS_ROWS))

#: The five numbers the graph statistics answer carries for this corpus, each
#: read off the rows above.
DEMO_STATISTICS: dict[str, int] = {
    "devices": len(DEMO_DEVICES),
    "channels": len(DEMO_ROWS),
    "classes": DEMO_CLASS_COUNT,
    "signals": len({entry["name"] for row in DEMO_ROWS for entry in row["signals"]}),
    "sections": len({row["section"] for row in DEMO_ROWS}),
}


def _ordered(counts: dict[str, int]) -> list[dict[str, Any]]:
    """Order one facet's counts as the index does: count descending, value ascending."""
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
    ]


def demo_facets(rows: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Count the five facets over *rows* the way the index counts them.

    Args:
        rows: The matches to count over. Defaults to the whole corpus, which is
            what an unfiltered search matches.

    Returns:
        The five facet lists, in the order the rail draws them.
    """
    matched = DEMO_ROWS if rows is None else rows
    section: dict[str, int] = {}
    system: dict[str, int] = {}
    signal: dict[str, int] = {}
    direction: dict[str, int] = {}
    klass: dict[str, int] = {}

    for row in matched:
        section[row["section"]] = section.get(row["section"], 0) + 1
        system[row["system"]] = system.get(row["system"], 0) + 1
        for entry in row["signals"]:
            signal[entry["name"]] = signal.get(entry["name"], 0) + 1
        edges = row["edges"]
        reads, writes = _EDGE_READ in edges, _EDGE_WRITE in edges
        for value in (
            (["R"] if reads else [])
            + (["W"] if writes else [])
            + (["RW"] if reads and writes else [])
            + (["none"] if not edges else [])
        ):
            direction[value] = direction.get(value, 0) + 1

    # The class facet counts devices rather than channels, and counts one under
    # every class it rolls up to — which is what puts an abstract branch like
    # Magnet in the list with a number no device is typed by directly.
    for device in sorted({row["device"] for row in matched}):
        for name in _class_ancestors(DEMO_DEVICE_CLASSES[device]):
            uri = class_uri(name)
            klass[uri] = klass.get(uri, 0) + 1

    return {
        "section": _ordered(section),
        "system": _ordered(system),
        "class": _ordered(klass),
        "signal": _ordered(signal),
        "dir": _ordered(direction),
    }


#: The facets an unfiltered search answers with.
DEMO_FACETS: dict[str, list[dict[str, Any]]] = demo_facets()


# ---------------------------------------------------------------------------
# The device a card is built from
# ---------------------------------------------------------------------------


def _subfield(full_pv: str) -> str | None:
    """Return what the card calls the address's role, or ``None`` when it has none."""
    if full_pv.endswith("SP00"):
        return "Setpoint"
    return "Readback" if "AM" in full_pv[-4:] else None


def device_card(device: str) -> dict[str, Any]:
    """Build the store-shaped device row for *device* out of the corpus rows.

    The store answers a device with its bindings grouped under the signals they
    carry, plus the descriptive fields the card prints. Everything that names an
    address, a signal or a class is read back off :data:`DEMO_ROWS`, so the card
    and the search page can only ever describe the same device.

    Args:
        device: The device's control-system name.

    Returns:
        One row shaped as ``GRAPH_DEVICE_CYPHER`` returns it.

    Raises:
        KeyError: When the corpus holds no such device.
    """
    spec = DEMO_DEVICES[device]
    rows = [row for row in DEMO_ROWS if row["device"] == device]

    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        for entry in row["signals"]:
            group = groups.setdefault(
                entry["name"], {"uri": entry["uri"], "name": entry["name"], "bindings": []}
            )
            group["bindings"].append(
                {
                    "fullPv": row["fullPv"],
                    "description": row["description"],
                    "fieldDescription": spec.family.field,
                    "subfieldDescription": _subfield(row["fullPv"]),
                    "protocol": "ca",
                    "confidence": 0.94 if _EDGE_WRITE in row["edges"] else 0.98,
                    "edges": list(row["edges"]),
                }
            )
    for group in groups.values():
        group["bindings"].sort(key=lambda binding: binding["fullPv"])

    return {
        "uri": device_uri(device),
        "device": device,
        "class": class_uri(spec.family.class_name),
        "classes": [class_uri(spec.family.class_name)],
        "rawType": spec.family.stem,
        "section": spec.section,
        "system": spec.family.system,
        # A plausible position along the ring: the section sets the arc, the
        # ordinal the step within it.
        "sPositionM": round(DEMO_SECTIONS.index(spec.section) * 30.0 + spec.ordinal * 4.271, 3),
        "ordinalInSection": spec.ordinal,
        "systemDescription": _SYSTEM_DESCRIPTIONS[spec.family.system],
        "familyDescription": spec.family.description,
        "ringDescription": _RING_DESCRIPTION,
        "signals": list(groups.values()),
    }


#: The device the card is asserted against: the corpus's first quadrupole, whose
#: readback and setpoint are grouped under the one signal they both carry.
DEMO_DEVICE_ROW: dict[str, Any] = device_card(
    _device_name(DEMO_SECTIONS[0], DEMO_FAMILIES[0].stem, 1)
)


# ---------------------------------------------------------------------------
# The fake store, for the one read that still goes to it
# ---------------------------------------------------------------------------

#: Marks a constructor argument nobody passed, so an explicit ``None`` — a
#: device read that must answer no rows — is told apart from the default.
_UNSET: Any = object()


def _as_result(value: Any, default_rows: list[dict[str, Any]]) -> QueryResult:
    """Read a constructor argument as the result its read answers with.

    Args:
        value: A :class:`QueryResult`, one row, a list of rows, ``None`` for a
            read that answers nothing, or :data:`_UNSET` for the default.
        default_rows: Rows the read answers with when nothing was passed.

    Returns:
        The result the fake will hand back for that read.
    """
    if value is _UNSET:
        return QueryResult(rows=list(default_rows), truncated=False)
    if value is None:
        return QueryResult(rows=[], truncated=False)
    if isinstance(value, QueryResult):
        return value
    if isinstance(value, dict):
        return QueryResult(rows=[value], truncated=False)
    return QueryResult(rows=list(value), truncated=False)


class FakeGraphContext:
    """Stand-in for ``GraphContext``, answering per query and recording calls.

    Each read is matched exactly against the query constant the endpoint
    sends — the device read, which is the one the routes still send to the
    store — which keeps the fake keyed to real query text rather than to call
    order or to a substring that a later edit could break. Cypher with no
    entry raises: a read this fake has never been taught is a test that has
    drifted from the routes, and answering it with somebody else's rows would
    hide that.
    """

    def __init__(
        self,
        *,
        device_result: Any = _UNSET,
        raises: BaseException | None = None,
        empty: bool = False,
        empty_raises: BaseException | None = None,
        uri: str | None = DEMO_STORE_URI,
    ) -> None:
        """Build a fake store.

        Args:
            device_result: What the device read answers with — a
                ``QueryResult``, the one row, or a list of rows.
                Defaults to :data:`DEMO_DEVICE_ROW`; ``None`` answers no rows,
                which is how a store reports a device it does not hold.
            raises: Raised by every :meth:`run_read` when given.
            empty: What :meth:`is_empty` reports.
            empty_raises: Raised by :meth:`is_empty` when given, which is how a
                store that is down rather than unseeded behaves.
            uri: The store URI, or ``None`` for an unconfigured context.
        """
        self._results: dict[str, QueryResult] = {
            GRAPH_DEVICE_CYPHER: _as_result(device_result, [DEMO_DEVICE_ROW]),
        }
        self._raises = raises
        self._empty = empty
        self._empty_raises = empty_raises
        self._uri = uri
        #: Every read, as the query it sent and the row cap it asked for.
        self.calls: list[tuple[str, int | None]] = []
        #: The parameters each of those reads carried, one entry per call.
        self.params: list[Any] = []
        self.empty_checks = 0
        self.shutdowns = 0
        self.initializations = 0
        #: The thread every store call arrived on, and whether that thread was
        #: running an event loop when it did.
        self.thread_ids: list[int] = []
        self.saw_running_loop: list[bool] = []

    @property
    def uri(self) -> str | None:
        """The store URI, as ``GraphContext.uri`` reports it."""
        return self._uri

    @property
    def configured(self) -> bool:
        """Whether a store connection was resolved, as ``GraphContext`` reports."""
        return self._uri is not None

    def _record(self) -> None:
        """Note the thread this call arrived on and whether a loop runs there."""
        self.thread_ids.append(threading.get_ident())
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.saw_running_loop.append(False)
        else:
            self.saw_running_loop.append(True)

    def initialize(self) -> None:
        """Match ``GraphContext.initialize``, which dials nothing and is idempotent."""
        self.initializations += 1

    def run_read(
        self,
        cypher: str,
        params: Any = None,
        *,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Answer *cypher* from the preset rows, recording the call.

        Args:
            cypher: The query the endpoint sent, matched exactly.
            params: Query parameters, recorded for assertion.
            max_rows: Row cap the endpoint asked for, recorded for assertion.

        Returns:
            The preset result for *cypher*.

        Raises:
            AssertionError: When *cypher* is not one of the queries this fake
                answers.
            BaseException: Whatever ``raises`` was constructed with.
        """
        self._record()
        self.calls.append((cypher, max_rows))
        self.params.append(params)
        if self._raises is not None:
            raise self._raises
        try:
            return self._results[cypher]
        except KeyError:
            raise AssertionError(f"unkeyed cypher: {cypher[:80]}") from None

    def reads_of(self, cypher: str) -> list[tuple[Any, int | None]]:
        """Return the parameters and row cap of every read that sent *cypher*.

        Args:
            cypher: One of the query constants this fake answers.

        Returns:
            One ``(params, max_rows)`` pair per matching call, in call order.
        """
        return [
            (params, max_rows)
            for (sent, max_rows), params in zip(self.calls, self.params, strict=True)
            if sent == cypher
        ]

    def last_params(self, cypher: str) -> Any:
        """Return the parameters the most recent read of *cypher* carried.

        Args:
            cypher: One of the query constants this fake answers.

        Returns:
            The parameters of that read.

        Raises:
            AssertionError: When no read sent *cypher*.
        """
        reads = self.reads_of(cypher)
        if not reads:
            raise AssertionError(f"no read sent: {cypher[:80]}")
        return reads[-1][0]

    def is_empty(self) -> bool:
        """Report whether the store holds a corpus, recording the probe.

        Returns:
            What ``empty`` was constructed with.

        Raises:
            BaseException: Whatever ``empty_raises`` was constructed with,
                which is how an unreachable store answers this probe.
        """
        self._record()
        self.empty_checks += 1
        if self._empty_raises is not None:
            raise self._empty_raises
        return self._empty

    def shutdown(self) -> None:
        """Close the store seam, as the app's lifespan does on teardown."""
        self.shutdowns += 1


def install_graph_paradigm(
    client: Any,
    ctx: FakeGraphContext | None = None,
    index: Any = _UNSET,
) -> None:
    """Put *client*'s app into graph mode, with *ctx* as its store seam if given.

    Written directly onto ``app.state`` rather than by restarting the lifespan,
    which is how the route tests in this package reach a paradigm the fixture
    app was not built for.

    A context that could not be built leaves no attribute behind at all — which
    is the state the routes must answer 503 from — so the default is to remove
    the attribute rather than to set it to ``None``.

    The search index is staged the way the lifespan stages it: a real handle on
    a real file by default, or the absence a route answers 503 from. The app
    closes whatever is installed here on teardown, so each call gets its own
    handle on the one file this process built — and a call that replaces an
    earlier handle closes that one first, since the lifespan only ever closes
    the handle present at teardown.

    Args:
        client: The FastAPI test client whose app state is rewritten.
        ctx: The store seam to install, or ``None`` for an app whose context
            could not be built.
        index: The search index to install — a :class:`GraphIndex`, a
            :class:`GraphIndexAbsence`, or ``None`` for an app that holds no
            index handle at all. Omitted, a fresh handle on this process's demo
            index is opened.
    """
    app_state = client.app.state
    app_state.pipeline_type = "graph"
    app_state.available_pipelines = ["graph"]
    app_state.databases = {}
    app_state.graph_backed = True
    if ctx is None:
        if hasattr(app_state, "graph_context"):
            delattr(app_state, "graph_context")
    else:
        app_state.graph_context = ctx

    close_previous = getattr(getattr(app_state, "graph_index", None), "close", None)
    if close_previous is not None:
        close_previous()

    resolved = open_demo_index() if index is _UNSET else index
    if resolved is None:
        if hasattr(app_state, "graph_index"):
            delattr(app_state, "graph_index")
    else:
        app_state.graph_index = resolved


def demo_context(**overrides: Any) -> FakeGraphContext:
    """Return a fake holding the demo corpus, with per-test overrides applied.

    Takes no positional arguments, so it can be installed directly as the app's
    ``_make_graph_context`` seam.

    Args:
        **overrides: Any :class:`FakeGraphContext` keyword, replacing the demo
            default for that argument.

    Returns:
        A fake store answering — from the constructor's own defaults — the
        demo device.
    """
    return FakeGraphContext(**overrides)


# ---------------------------------------------------------------------------
# The real search index the graph paradigm reads
# ---------------------------------------------------------------------------
#
# The store is faked above because no Neo4j runs in a test. The search index is
# not faked, because it is a file: the corpus this module already describes is
# emitted as Turtle and run through the same parse -> derive -> write path
# ``osprey build`` runs, so a route test reads rows a builder actually wrote.

#: URIs of the two classes n10s derives the binding and signal labels from. Both
#: are ordinary classes in the corpus, which is why the ontology declares each
#: and the taxonomy prunes them back out.
BINDING_CLASS_URI = class_uri("ChannelBinding")
SIGNAL_CLASS_URI = class_uri("SemanticSignal")

#: What the built index's ``meta`` row names as the corpus it came from, and
#: the file name the index itself is written under.
DEMO_CORPUS_FILENAME = "demo_corpus.ttl"
DEMO_INDEX_FILENAME = "graph.duckdb"

#: Separator between the predicates of one Turtle subject.
_TRIPLE_SEP = " ;\n    "

#: The corpus's prefixes. Every subject below is written as a full IRI instead:
#: an ALS-style device name is full of underscores, and a prefixed name would
#: have to escape them.
_TURTLE_PREFIXES = "\n".join(
    (
        "@prefix narad_p: <https://narad.example.org/property/> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "",
    )
)


def _turtle_literal(value: str) -> str:
    """Quote *value* as a Turtle string literal."""
    return json.dumps(value)


def _subject(uri: str, triples: list[str]) -> str:
    """Render one subject and its predicates as a Turtle statement."""
    return f"<{uri}> {_TRIPLE_SEP.join(triples)} ."


def demo_turtle() -> str:
    """Emit the synthesized corpus as Turtle, exactly as this module describes it.

    The classes come from :data:`DEMO_CLASS_ROWS` — every one of them, so the
    branches no device is typed by are present and get pruned out of the
    taxonomy the way a real ontology's unbound classes are. The devices,
    bindings and signals come from :data:`DEMO_ROWS`, so the corpus, the index
    built from it and the fake store describe one facility rather than three.

    Returns:
        The corpus, ready for
        :func:`~osprey.services.channel_finder.graph_index.builder.parse_corpus`.
    """
    lines = [_TURTLE_PREFIXES]

    for row in DEMO_CLASS_ROWS:
        triples = ["a owl:Class"]
        triples += [f"rdfs:subClassOf <{parent}>" for parent in row["parents"]]
        triples += [f"skos:altLabel {_turtle_literal(label)}" for label in row["altLabel"]]
        lines.append(_subject(row["uri"], triples))

    signals = {entry["name"]: entry["uri"] for row in DEMO_ROWS for entry in row["signals"]}
    for name, uri in sorted(signals.items()):
        lines.append(
            _subject(uri, [f"a <{SIGNAL_CLASS_URI}>", f"rdfs:label {_turtle_literal(name)}"])
        )

    # One block per device carrying every binding that addresses it: the corpus
    # is subject-oriented, and a device with a readback and a setpoint is one
    # node with two ``hasBinding`` edges.
    by_device: dict[str, list[dict[str, Any]]] = {}
    for row in DEMO_ROWS:
        by_device.setdefault(row["device"], []).append(row)

    for device, rows in sorted(by_device.items()):
        first = rows[0]
        triples = [
            f"a <{class_uri(DEMO_DEVICE_CLASSES[device])}>",
            f"narad_p:sourceName {_turtle_literal(device)}",
            f"narad_p:sectionCode {_turtle_literal(first['section'])}",
            f"narad_p:system {_turtle_literal(first['system'])}",
        ]
        triples += [f"narad_p:hasBinding <{binding_uri(row['fullPv'])}>" for row in rows]
        lines.append(_subject(first["device_uri"], triples))

        for row in rows:
            binding = [
                f"a <{BINDING_CLASS_URI}>",
                f"narad_p:fullPv {_turtle_literal(row['fullPv'])}",
                f"narad_p:description {_turtle_literal(row['description'])}",
            ]
            for edge, predicate in (
                (_EDGE_READ, "narad_p:readsSignal"),
                (_EDGE_WRITE, "narad_p:writesSignal"),
            ):
                if edge in row["edges"]:
                    binding += [f"{predicate} <{entry['uri']}>" for entry in row["signals"]]
            lines.append(_subject(binding_uri(row["fullPv"]), binding))

    return "\n".join(lines) + "\n"


def build_demo_index(directory: Path | str) -> Path:
    """Build the demo corpus's search index into *directory* and return its path.

    Runs the path ``osprey build`` runs — parse the corpus, derive the channel
    rows from the binding rows, write one DuckDB file — so nothing here
    hand-writes a row the reader will later be asserted against.

    The builder keeps ``rdflib`` and ``duckdb`` inside its own functions, and
    this one keeps the builder inside itself for the same reason: importing this
    fixture module must stay cheap for the tests that only want the fake store.

    Args:
        directory: Where the index goes. Must already exist.

    Returns:
        Path to the written index file.
    """
    import hashlib

    from osprey.services.channel_finder.graph_index.builder import (
        build_from_rows,
        channels_from_rows,
        parse_corpus,
    )

    text = demo_turtle()
    parsed = parse_corpus(text)
    index_path = Path(directory) / DEMO_INDEX_FILENAME
    build_from_rows(
        parsed.binding_rows,
        parsed.class_rows,
        channels_from_rows(parsed.binding_rows),
        index_path,
        {
            "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "corpus_filename": DEMO_CORPUS_FILENAME,
            "binding_count": len(parsed.binding_rows),
            "device_count": len({row.device_uri for row in parsed.binding_rows}),
            "class_count": len(parsed.class_rows),
            "signal_count": parsed.signal_count,
            "section_count": len(parsed.section_codes),
        },
    )
    return index_path


#: This process's index, and the directory holding it. Building costs a parse
#: and three inserts; opening one read-only costs a ``meta`` read — so the file
#: is built once and every caller opens its own handle on it. The directory is
#: held on the module so it outlives the call that made it.
_INDEX_DIR: Any = None
_INDEX_PATH: Path | None = None


def demo_index_path() -> Path:
    """Return the path of this process's demo index, building it on first use.

    Returns:
        Path to the built index file.
    """
    global _INDEX_DIR, _INDEX_PATH
    if _INDEX_PATH is None:
        _INDEX_DIR = tempfile.TemporaryDirectory(prefix="osprey-demo-index-")
        _INDEX_PATH = build_demo_index(_INDEX_DIR.name)
    return _INDEX_PATH


def open_demo_index() -> GraphIndex:
    """Open a fresh read-only handle on this process's demo index.

    Each caller owns its handle and closes it — which is what the app's lifespan
    does on teardown for the one it was handed.

    Returns:
        The open index.

    Raises:
        AssertionError: When the freshly built index cannot be opened, so a
            broken build fails as itself rather than as an unavailable index on
            every route test downstream.
    """
    opened = open_graph_index(demo_index_path())
    assert not isinstance(opened, GraphIndexAbsence), opened.detail
    return opened
