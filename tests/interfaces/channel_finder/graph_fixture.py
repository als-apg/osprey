"""The demo-shaped graph store every graph-paradigm test is written against.

The graph paradigm has no database file to point a test at, so what stands in
for one is this module: a corpus shaped like the demo ontology, the five
statistics answers that corpus implies, the store's relationship vocabulary,
and a fake context that answers each of those reads by looking at the Cypher it
was handed.

There is one copy of that corpus on purpose. The route tests, the launcher in
``tests/interfaces/conftest.py`` that boots a real graph-mode server, and the
browser and visual lanes that photograph it all draw from here, so a number
asserted in one lane means the same thing in every other.

The fake is keyed by the query text itself rather than by call order: one
request makes several reads with different Cypher, and dispatching on the text
keeps the fake tied to what the endpoint actually sends. The match is exact
against the imported constants, so a query the fake has no answer for fails
loudly instead of being served somebody else's rows — a route that grows a
sixth read is a failing assertion here, not a silently wrong page. It also
records the thread each call arrived on, so the "store reads never run on the
event loop" contract can be asserted rather than assumed, and the parameters
each read was given, so a route's request contract can be asserted from the
store side.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from osprey.interfaces.channel_finder.database_api import (
    _GRAPH_EXPLORE_MAX_ROWS,
    GRAPH_CHANNEL_COUNT_CYPHER,
    GRAPH_DEVICE_COUNT_CYPHER,
    GRAPH_ONTOLOGY_CYPHER,
    GRAPH_SECTION_COUNT_CYPHER,
    GRAPH_SIGNAL_COUNT_CYPHER,
)
from osprey.mcp_server.graph.server_context import QueryResult

# Imported from the module that owns it rather than through the explorer's
# re-export: the vocabulary query belongs to the seeder's schema snapshot, and
# importing it from there is what makes a move show up as a failure here.
from osprey.services.channel_finder.graph_queries import (
    GRAPH_DEVICE_CYPHER,
    GRAPH_SEARCH_CYPHER,
)
from osprey.services.facility_knowledge.seeder.prompt_snapshot import RELATIONSHIP_TYPES_CYPHER

__all__ = [
    "DEMO_CLASS_COUNT",
    "DEMO_CLASS_ROWS",
    "DEMO_COUNTS",
    "DEMO_DEVICE_ROW",
    "DEMO_RELATIONSHIP_ROWS",
    "DEMO_RELATIONSHIP_TYPES",
    "DEMO_SEARCH_FACET_OVERFLOW",
    "DEMO_SEARCH_ROW",
    "DEMO_SECTIONS",
    "DEMO_STATISTICS",
    "DEMO_STORE_URI",
    "DEV_NAMESPACE",
    "SEARCH_FACET_CAP",
    "SEARCH_PAGE_SIZE",
    "SEED_COMMAND",
    "SEM_NAMESPACE",
    "FakeGraphContext",
    "class_row",
    "class_uri",
    "demo_context",
    "device_uri",
    "install_graph_paradigm",
    "signal_uri",
]

#: Namespace the demo corpus mints its class URIs in.
SEM_NAMESPACE = "https://narad.example.org/schema/shared_semantics/"

#: The command an empty-store answer must name.
SEED_COMMAND = "osprey knowledge seed-graph"

#: The bolt URI the demo store is reachable at, matching the ``services.graphdb``
#: block the test launcher writes.
DEMO_STORE_URI = "bolt://localhost:7687"


def class_uri(name: str) -> str:
    """Return the class URI the demo corpus would hold for *name*.

    Args:
        name: Bare class name, e.g. ``"Quadrupole"``.

    Returns:
        The fully qualified URI under :data:`SEM_NAMESPACE`.
    """
    return f"{SEM_NAMESPACE}{name}"


def class_row(
    name: str,
    rollup: int,
    parents: list[str] | None = None,
    alt_labels: list[str] | None = None,
    direct: int | None = None,
) -> dict[str, Any]:
    """Build one ontology row shaped as ``GRAPH_ONTOLOGY_CYPHER`` returns it.

    Args:
        name: Bare class name.
        rollup: Devices under the class *including* its subclasses.
        parents: Bare names of the classes this one is a subclass of.
        alt_labels: Alternative labels the corpus carries for the class.
        direct: Devices typed as the class itself. Defaults to the rollup,
            which is what the store answers for a class with no subclasses; a
            branch states its own.

    Returns:
        A row with the ``uri``, ``altLabel``, ``parents``, ``rollup`` and
        ``direct`` columns the ontology query projects.
    """
    return {
        "uri": class_uri(name),
        "altLabel": list(alt_labels or []),
        "parents": [class_uri(parent) for parent in (parents or [])],
        "rollup": rollup,
        "direct": rollup if direct is None else direct,
    }


#: A demo-shaped ontology: 19 device classes plus two non-device leaves that
#: pruning must drop. The rollups nest exactly — every branch sums to its
#: parent and the whole tree to the root's 512 — so a test asserting one number
#: is asserting the shape of the tree under it. That is also why every branch
#: states ``direct=0``: nothing is typed as the grouping itself.
DEMO_CLASS_ROWS: list[dict[str, Any]] = [
    class_row("AcceleratorDevice", 512, direct=0),
    class_row("Magnet", 382, ["AcceleratorDevice"], direct=0),
    class_row("Dipole", 44, ["Magnet"], ["BEND"]),
    class_row("Quadrupole", 86, ["Magnet"], ["QUAD"]),
    class_row("Sextupole", 96, ["Magnet"], ["SEXT"]),
    class_row("Corrector", 156, ["Magnet"], direct=0),
    class_row("HorizontalCorrector", 80, ["Corrector"]),
    class_row("VerticalCorrector", 76, ["Corrector"]),
    class_row("Diagnostic", 70, ["AcceleratorDevice"], direct=0),
    class_row("BeamPositionMonitor", 60, ["Diagnostic"], ["BPM"]),
    class_row("CurrentMonitor", 10, ["Diagnostic"]),
    class_row("VacuumDevice", 40, ["AcceleratorDevice"], direct=0),
    class_row("IonPump", 25, ["VacuumDevice"]),
    class_row("VacuumGauge", 15, ["VacuumDevice"]),
    class_row("RFDevice", 12, ["AcceleratorDevice"], direct=0),
    class_row("RFCavity", 4, ["RFDevice"]),
    class_row("RFAmplifier", 8, ["RFDevice"]),
    class_row("InsertionDevice", 8, ["AcceleratorDevice"], direct=0),
    class_row("Undulator", 8, ["InsertionDevice"]),
    # Real classes in the store, but about signals and bindings rather than
    # devices: no devices roll up to them and nothing calls them a parent.
    class_row("SemanticSignal", 0),
    class_row("ChannelBinding", 0),
]

#: The device classes that survive pruning — the taxonomy an operator came for,
#: which is the 21 stored classes less the two non-device leaves.
DEMO_CLASS_COUNT = 19

#: The store's relationship vocabulary, in the order ``db.relationshipTypes()``
#: reports it (alphabetical, as the query orders by name).
DEMO_RELATIONSHIP_TYPES = [
    "HASBINDING",
    "READSSIGNAL",
    "SUBCLASSOF",
    "TYPE",
    "WRITESSIGNAL",
]

#: That vocabulary as rows, shaped as the schema query returns them.
DEMO_RELATIONSHIP_ROWS: list[dict[str, Any]] = [
    {"relationshipType": name} for name in DEMO_RELATIONSHIP_TYPES
]

#: What each of the store's ``count(...) AS n`` queries answers for the demo
#: corpus. ``devices`` is the root class's rollup by construction: both are
#: "a ``:Resource`` with at least one channel binding", counted once.
DEMO_COUNTS: dict[str, int] = {
    "devices": 512,
    "channels": 2908,
    "signals": 113,
    "sections": 3,
}

#: The five numbers the graph statistics answer carries for this corpus. Four
#: come from store counts; ``classes`` is the pruned taxonomy above rather than
#: a count query, which is why it is stated here beside them.
DEMO_STATISTICS: dict[str, int] = {
    "devices": DEMO_COUNTS["devices"],
    "channels": DEMO_COUNTS["channels"],
    "classes": DEMO_CLASS_COUNT,
    "signals": DEMO_COUNTS["signals"],
    "sections": DEMO_COUNTS["sections"],
}

#: Each count query paired with the :data:`DEMO_COUNTS` entry that answers it.
_COUNT_CYPHERS: tuple[tuple[str, str], ...] = (
    (GRAPH_DEVICE_COUNT_CYPHER, "devices"),
    (GRAPH_CHANNEL_COUNT_CYPHER, "channels"),
    (GRAPH_SIGNAL_COUNT_CYPHER, "signals"),
    (GRAPH_SECTION_COUNT_CYPHER, "sections"),
)


# ---------------------------------------------------------------------------
# The page and the device a finder search reads out of the demo corpus
# ---------------------------------------------------------------------------

#: Namespace the demo corpus mints device URIs in: the same host as the schema,
#: under a data path, which is how an n10s import keeps the individuals it
#: loads apart from the classes they are typed by.
DEV_NAMESPACE = "https://narad.example.org/data/shared_semantics/"

#: Rows one page of the finder holds, as ``GRAPH_SEARCH_CYPHER`` slices it.
SEARCH_PAGE_SIZE = 50

#: Entries a facet list carries before the explorer calls it clipped. Read from
#: the explorer's own cap rather than restated, so :data:`DEMO_SEARCH_FACET_OVERFLOW`
#: stays exactly one entry over whatever the route asks the store for.
SEARCH_FACET_CAP = _GRAPH_EXPLORE_MAX_ROWS

#: The sections the demo corpus covers — as many as ``DEMO_COUNTS["sections"]``.
DEMO_SECTIONS = ("SR01C", "SR02C", "SR03C")


def device_uri(source_name: str) -> str:
    """Return the URI the demo corpus holds for the device named *source_name*.

    Args:
        source_name: The device's control-system name, e.g. ``"SR01C___QFA____"``.

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


#: The device families the demo page is drawn from: the address stem an
#: ALS-style name is built on, the class the device is typed by, its system
#: code, and the signal its principal channel carries.
_SEARCH_FAMILIES: tuple[tuple[str, str, str, str], ...] = (
    ("QFA", "Quadrupole", "MG", "current"),
    ("QDA", "Quadrupole", "MG", "current"),
    ("SF", "Sextupole", "MG", "current"),
    ("BEND", "Dipole", "MG", "current"),
    ("HCM", "HorizontalCorrector", "MG", "current"),
    ("VCM", "VerticalCorrector", "MG", "current"),
    ("BPM", "BeamPositionMonitor", "DI", "beamPositionX"),
    ("IPUMP", "IonPump", "VA", "pressure"),
    ("GAUGE", "VacuumGauge", "VA", "pressure"),
)

#: The families an operator sets as well as reads, which is what gives their
#: devices a second channel with a ``WRITESSIGNAL`` edge.
_SETTABLE_FAMILIES = frozenset({"QFA", "QDA", "SF", "BEND", "HCM", "VCM"})


def _device_name(section: str, stem: str) -> str:
    """Return the ALS-style device name *stem* takes in *section*."""
    return f"{section}___{stem:_<7}"


def _search_row(
    device: str,
    section: str,
    system: str,
    suffix: str,
    signal: str | None,
    edges: list[str],
    description: str,
) -> dict[str, Any]:
    """Build one page row shaped as ``GRAPH_SEARCH_CYPHER`` projects it."""
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


def _build_search_page() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build the demo page and the class each of its devices is typed by.

    Returns:
        The fifty rows, ordered by ``fullPv`` as the store returns them, and a
        map from device name to the bare name of its class.
    """
    rows: list[dict[str, Any]] = []
    classes: dict[str, str] = {}
    for section in DEMO_SECTIONS:
        for stem, class_name, system, signal in _SEARCH_FAMILIES:
            device = _device_name(section, stem)
            classes[device] = class_name
            rows.append(
                _search_row(
                    device, section, system, "AM00", signal, ["READSSIGNAL"], f"{signal} readback"
                )
            )
            if stem in _SETTABLE_FAMILIES:
                rows.append(
                    _search_row(
                        device,
                        section,
                        system,
                        "SP00",
                        signal,
                        ["WRITESSIGNAL"],
                        f"{signal} setpoint",
                    )
                )
            if stem == "BPM":
                rows.append(
                    _search_row(
                        device,
                        section,
                        system,
                        "AM01",
                        "beamPositionY",
                        ["READSSIGNAL"],
                        "beamPositionY readback",
                    )
                )

    # One address that is read and set through the same channel, and one that
    # carries no semantic signal at all: the two rows a facet on direction has
    # nothing to show unless the corpus actually holds them.
    cavity = _device_name("SR01C", "RFCAV")
    classes[cavity] = "RFCavity"
    rows.append(
        _search_row(
            cavity,
            "SR01C",
            "RF",
            "AM00",
            "cavityVoltage",
            ["READSSIGNAL", "WRITESSIGNAL"],
            "cavityVoltage, read and set on one address",
        )
    )
    rows.append(
        _search_row(
            _device_name("SR03C", "GAUGE"),
            "SR03C",
            "VA",
            "ST00",
            None,
            [],
            "controller status word, no semantic signal",
        )
    )

    rows.sort(key=lambda row: row["fullPv"])
    return rows, classes


def _class_ancestors(name: str) -> list[str]:
    """Return the URIs of *name* and every class it rolls up to."""
    parents = {row["uri"]: row["parents"] for row in DEMO_CLASS_ROWS}
    seen: list[str] = []
    pending = [class_uri(name)]
    while pending:
        uri = pending.pop()
        if uri in seen:
            continue
        seen.append(uri)
        pending.extend(parents.get(uri, []))
    return seen


def _facet(counts: dict[str, int]) -> list[dict[str, Any]]:
    """Order one facet's counts as the query does: count descending, value ascending."""
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
    ]


def _build_search_facets(
    rows: list[dict[str, Any]], classes: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    """Count the five facets over *rows* the way the search query counts them."""
    section: dict[str, int] = {}
    system: dict[str, int] = {}
    signal: dict[str, int] = {}
    direction: dict[str, int] = {}
    klass: dict[str, int] = {}

    for row in rows:
        section[row["section"]] = section.get(row["section"], 0) + 1
        system[row["system"]] = system.get(row["system"], 0) + 1
        for entry in row["signals"]:
            signal[entry["name"]] = signal.get(entry["name"], 0) + 1
        edges = row["edges"]
        for value in (
            (["R"] if "READSSIGNAL" in edges else [])
            + (["W"] if "WRITESSIGNAL" in edges else [])
            + (["RW"] if "READSSIGNAL" in edges and "WRITESSIGNAL" in edges else [])
            + (["none"] if not edges else [])
        ):
            direction[value] = direction.get(value, 0) + 1

    # The class facet counts devices rather than channels, and counts one under
    # every class it rolls up to — which is what puts an abstract branch like
    # Magnet in the list with a number no device is typed by directly.
    for device in sorted({row["device"] for row in rows}):
        for uri in _class_ancestors(classes[device]):
            klass[uri] = klass.get(uri, 0) + 1

    return {
        "section": _facet(section),
        "system": _facet(system),
        "class": _facet(klass),
        "signal": _facet(signal),
        "dir": _facet(direction),
    }


_SEARCH_ROWS, _SEARCH_DEVICE_CLASSES = _build_search_page()

#: The one row a demo-corpus search answers with, shaped as
#: ``GRAPH_SEARCH_CYPHER`` returns it.
#:
#: ``total`` sits above one page on purpose: a finder that never paginates in
#: its own fixture cannot show that it paginates at all. The facets are counted
#: over the page rather than over all ``total`` matches — the store would count
#: them over every hit — so nothing here asserts that a facet's counts sum to
#: ``total``.
DEMO_SEARCH_ROW: dict[str, Any] = {
    "total": 128,
    "devices": 64,
    "rows": _SEARCH_ROWS,
    "facets": _build_search_facets(_SEARCH_ROWS, _SEARCH_DEVICE_CLASSES),
}

#: The same facets with one list run past the cap, which is how a store tells
#: the explorer that a facet was clipped: the query asks for ``cap`` entries and
#: gets ``cap + 1`` back only when there were more values than it can show.
DEMO_SEARCH_FACET_OVERFLOW: dict[str, list[dict[str, Any]]] = {
    **DEMO_SEARCH_ROW["facets"],
    "signal": [
        {"value": f"signal{index:04d}", "count": 1} for index in range(SEARCH_FACET_CAP + 1)
    ],
}

#: One device of the demo corpus, shaped as ``GRAPH_DEVICE_CYPHER`` returns it:
#: the first quadrupole of the page, with its readback and its setpoint grouped
#: under the one signal they both carry.
DEMO_DEVICE_ROW: dict[str, Any] = {
    "uri": device_uri(_device_name("SR01C", "QFA")),
    "device": _device_name("SR01C", "QFA"),
    "class": class_uri("Quadrupole"),
    "classes": [class_uri("Quadrupole")],
    "rawType": "QFA",
    "section": "SR01C",
    "system": "MG",
    "sPositionM": 12.734,
    "ordinalInSection": 1,
    "systemDescription": "Storage ring magnets",
    "familyDescription": "Focusing quadrupole, family A",
    "ringDescription": "Storage ring",
    "signals": [
        {
            "uri": signal_uri("current"),
            "name": "current",
            "bindings": [
                {
                    "fullPv": f"{_device_name('SR01C', 'QFA')}AM00",
                    "description": "current readback",
                    "fieldDescription": "Current",
                    "subfieldDescription": "Readback",
                    "protocol": "ca",
                    "confidence": 0.98,
                    "edges": ["READSSIGNAL"],
                },
                {
                    "fullPv": f"{_device_name('SR01C', 'QFA')}SP00",
                    "description": "current setpoint",
                    "fieldDescription": "Current",
                    "subfieldDescription": "Setpoint",
                    "protocol": "ca",
                    "confidence": 0.94,
                    "edges": ["WRITESSIGNAL"],
                },
            ],
        }
    ],
}

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

    One request makes several reads with different Cypher, so a single preset
    result cannot serve them all. Each read is matched exactly against the
    query constant the endpoint sends — the ontology tree, the four censuses,
    the relationship vocabulary, the faceted search and the device read — which
    keeps the fake keyed to real query text rather than to call order or to a
    substring that a later edit could break. Cypher with no entry raises: a
    read this fake has never been taught is a test that has drifted from the
    routes, and answering it with somebody else's rows would hide that.
    """

    def __init__(
        self,
        *,
        class_rows: list[dict[str, Any]] | None = None,
        class_truncated: bool = False,
        relationship_rows: list[dict[str, Any]] | None = None,
        relationship_truncated: bool = False,
        counts: dict[str, int] | None = None,
        search_result: Any = _UNSET,
        device_result: Any = _UNSET,
        raises: BaseException | None = None,
        empty: bool = False,
        empty_raises: BaseException | None = None,
        uri: str | None = DEMO_STORE_URI,
    ) -> None:
        """Build a fake store.

        Args:
            class_rows: Rows the ontology query answers with.
            class_truncated: Whether that read hit the row cap.
            relationship_rows: Rows the vocabulary query answers with.
            relationship_truncated: Whether that read hit the row cap.
            counts: Answers for the ``count(...) AS n`` queries, keyed as
                :data:`DEMO_COUNTS` is. An unlisted count answers no rows.
            search_result: What the faceted search answers with — a
                ``QueryResult``, the one row, or a list of rows. Defaults to
                :data:`DEMO_SEARCH_ROW`; ``None`` answers no rows.
            device_result: What the device read answers with, same forms.
                Defaults to :data:`DEMO_DEVICE_ROW`; ``None`` answers no rows,
                which is how a store reports a device it does not hold.
            raises: Raised by every :meth:`run_read` when given.
            empty: What :meth:`is_empty` reports.
            empty_raises: Raised by :meth:`is_empty` when given, which is how a
                store that is down rather than unseeded behaves.
            uri: The store URI, or ``None`` for an unconfigured context.
        """
        self._results: dict[str, QueryResult] = {
            GRAPH_ONTOLOGY_CYPHER: QueryResult(
                rows=list(class_rows or []), truncated=class_truncated
            ),
            RELATIONSHIP_TYPES_CYPHER: QueryResult(
                rows=list(relationship_rows or []), truncated=relationship_truncated
            ),
            GRAPH_SEARCH_CYPHER: _as_result(search_result, [DEMO_SEARCH_ROW]),
            GRAPH_DEVICE_CYPHER: _as_result(device_result, [DEMO_DEVICE_ROW]),
        }
        resolved = dict(counts or {})
        for cypher, name in _COUNT_CYPHERS:
            rows = [{"n": resolved[name]}] if name in resolved else []
            self._results[cypher] = QueryResult(rows=rows, truncated=False)
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


def install_graph_paradigm(client: Any, ctx: FakeGraphContext | None = None) -> None:
    """Put *client*'s app into graph mode, with *ctx* as its store seam if given.

    Written directly onto ``app.state`` rather than by restarting the lifespan,
    which is how the route tests in this package reach a paradigm the fixture
    app was not built for.

    A context that could not be built leaves no attribute behind at all — which
    is the state the routes must answer 503 from — so the default is to remove
    the attribute rather than to set it to ``None``.

    Args:
        client: The FastAPI test client whose app state is rewritten.
        ctx: The store seam to install, or ``None`` for an app whose context
            could not be built.
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


def demo_context(**overrides: Any) -> FakeGraphContext:
    """Return a fake holding the demo corpus, with per-test overrides applied.

    Takes no positional arguments, so it can be installed directly as the app's
    ``_make_graph_context`` seam.

    Args:
        **overrides: Any :class:`FakeGraphContext` keyword, replacing the demo
            default for that argument.

    Returns:
        A fake store answering the demo ontology, vocabulary and counts, and —
        from the constructor's own defaults — the demo search page and device.
    """
    kwargs: dict[str, Any] = {
        "class_rows": DEMO_CLASS_ROWS,
        "relationship_rows": DEMO_RELATIONSHIP_ROWS,
        "counts": DEMO_COUNTS,
    }
    kwargs.update(overrides)
    return FakeGraphContext(**kwargs)
