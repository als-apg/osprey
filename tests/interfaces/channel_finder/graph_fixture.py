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

The fake is keyed by query text rather than by call order: one request makes
several reads with different Cypher, and keying on the text keeps the fake tied
to what the endpoint actually sends. It also records the thread each call
arrived on, so the "store reads never run on the event loop" contract can be
asserted rather than assumed.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from osprey.mcp_server.graph.server_context import QueryResult

__all__ = [
    "DEMO_CLASS_COUNT",
    "DEMO_CLASS_ROWS",
    "DEMO_COUNTS",
    "DEMO_RELATIONSHIP_ROWS",
    "DEMO_RELATIONSHIP_TYPES",
    "DEMO_STATISTICS",
    "DEMO_STORE_URI",
    "SEED_COMMAND",
    "SEM_NAMESPACE",
    "FakeGraphContext",
    "class_row",
    "class_uri",
    "demo_context",
    "install_graph_paradigm",
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
) -> dict[str, Any]:
    """Build one ontology row shaped as ``GRAPH_ONTOLOGY_CYPHER`` returns it.

    Args:
        name: Bare class name.
        rollup: Devices under the class *including* its subclasses.
        parents: Bare names of the classes this one is a subclass of.
        alt_labels: Alternative labels the corpus carries for the class.

    Returns:
        A row with the ``uri``, ``altLabel``, ``parents`` and ``rollup``
        columns the ontology query projects.
    """
    return {
        "uri": class_uri(name),
        "altLabel": list(alt_labels or []),
        "parents": [class_uri(parent) for parent in (parents or [])],
        "rollup": rollup,
    }


#: A demo-shaped ontology: 19 device classes plus two non-device leaves that
#: pruning must drop. The rollups nest exactly — every branch sums to its
#: parent and the whole tree to the root's 512 — so a test asserting one number
#: is asserting the shape of the tree under it.
DEMO_CLASS_ROWS: list[dict[str, Any]] = [
    class_row("AcceleratorDevice", 512),
    class_row("Magnet", 382, ["AcceleratorDevice"]),
    class_row("Dipole", 44, ["Magnet"], ["BEND"]),
    class_row("Quadrupole", 86, ["Magnet"], ["QUAD"]),
    class_row("Sextupole", 96, ["Magnet"], ["SEXT"]),
    class_row("Corrector", 156, ["Magnet"]),
    class_row("HorizontalCorrector", 80, ["Corrector"]),
    class_row("VerticalCorrector", 76, ["Corrector"]),
    class_row("Diagnostic", 70, ["AcceleratorDevice"]),
    class_row("BeamPositionMonitor", 60, ["Diagnostic"], ["BPM"]),
    class_row("CurrentMonitor", 10, ["Diagnostic"]),
    class_row("VacuumDevice", 40, ["AcceleratorDevice"]),
    class_row("IonPump", 25, ["VacuumDevice"]),
    class_row("VacuumGauge", 15, ["VacuumDevice"]),
    class_row("RFDevice", 12, ["AcceleratorDevice"]),
    class_row("RFCavity", 4, ["RFDevice"]),
    class_row("RFAmplifier", 8, ["RFDevice"]),
    class_row("InsertionDevice", 8, ["AcceleratorDevice"]),
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

#: Substrings that tell one count query from another, paired with the
#: :data:`DEMO_COUNTS` entry that answers it. Each substring is unique to its
#: own query: the ontology query also counts devices, but projects the count as
#: ``rollup`` rather than ``n``, so it never matches here.
_COUNT_QUERY_KEYS: tuple[tuple[str, str], ...] = (
    ("SemanticSignal", "signals"),
    ("sectionCode", "sections"),
    ("count(DISTINCT d) AS n", "devices"),
    ("count(b) AS n", "channels"),
)

#: The substring that identifies the relationship-vocabulary query.
_RELATIONSHIP_QUERY_KEY = "relationshipTypes"


class FakeGraphContext:
    """Stand-in for ``GraphContext``, answering per query and recording calls.

    One request makes several reads with different Cypher, so a single preset
    result cannot serve them all. Queries are told apart by a substring of
    their own text — the class query names ``SUBCLASSOF``, the vocabulary query
    names ``relationshipTypes``, each count query names its own projection —
    which keeps the fake keyed to what the endpoint actually sends rather than
    to call order. Anything unrecognized falls through to the class rows, so a
    test that cares about one read does not have to enumerate the others.
    """

    def __init__(
        self,
        *,
        class_rows: list[dict[str, Any]] | None = None,
        class_truncated: bool = False,
        relationship_rows: list[dict[str, Any]] | None = None,
        relationship_truncated: bool = False,
        counts: dict[str, int] | None = None,
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
            raises: Raised by every :meth:`run_read` when given.
            empty: What :meth:`is_empty` reports.
            empty_raises: Raised by :meth:`is_empty` when given, which is how a
                store that is down rather than unseeded behaves.
            uri: The store URI, or ``None`` for an unconfigured context.
        """
        self._class_result = QueryResult(rows=list(class_rows or []), truncated=class_truncated)
        self._relationship_result = QueryResult(
            rows=list(relationship_rows or []), truncated=relationship_truncated
        )
        self._counts = dict(counts or {})
        self._raises = raises
        self._empty = empty
        self._empty_raises = empty_raises
        self._uri = uri
        self.calls: list[tuple[str, int | None]] = []
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
            cypher: The query the endpoint sent.
            params: Query parameters, accepted and ignored.
            max_rows: Row cap the endpoint asked for, recorded for assertion.

        Returns:
            The preset result matching *cypher*.

        Raises:
            BaseException: Whatever ``raises`` was constructed with.
        """
        self._record()
        self.calls.append((cypher, max_rows))
        if self._raises is not None:
            raise self._raises
        if _RELATIONSHIP_QUERY_KEY in cypher:
            return self._relationship_result
        for key, name in _COUNT_QUERY_KEYS:
            if key in cypher:
                if name not in self._counts:
                    return QueryResult(rows=[], truncated=False)
                return QueryResult(rows=[{"n": self._counts[name]}], truncated=False)
        return self._class_result

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
        A fake store answering the demo ontology, vocabulary and counts.
    """
    kwargs: dict[str, Any] = {
        "class_rows": DEMO_CLASS_ROWS,
        "relationship_rows": DEMO_RELATIONSHIP_ROWS,
        "counts": DEMO_COUNTS,
    }
    kwargs.update(overrides)
    return FakeGraphContext(**kwargs)
