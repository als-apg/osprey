"""Tree-read guards for the shared throwaway-graph-store recipe.

The properties that hold the arrangement in ``tests/_graphdb_container.py``
together belong to no one lane: a graph container is built in exactly one
module, no module reaches for the superseded import path, and the plugin recipe
has exactly one implementation. A store that runs is the only other thing that
would prove them, which is to say they would go unproved on every host without
a container engine or without the neo4j extra, since each of the lanes
concerned skips there. So they are read off the tree instead, which works
wherever the suite is collected. This file exempts itself from the scan,
because the guards below name what they forbid.

The module also pins, against a fake driver session, what a store read that
gets no answer does: it waits for the store to answer again and then reads
once more, and fails as :class:`tests._graphdb_container.GraphStoreUnavailable`,
naming the store, only when the store does not come back. The fast lane has no
container, so a fake stands in for the driver there.
"""

from __future__ import annotations

import ast
import functools
import socket
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from neo4j.exceptions import ClientError, ServiceUnavailable

from osprey.mcp_server.graph.server_context import GraphUnreachable
from tests._container_support import ContainerExitedError
from tests._graphdb_container import (
    GRAPHDB_TEST_DATABASE,
    STORE_PROBE_INTERVAL_S,
    STORE_PROBE_TIMEOUT_S,
    STORE_RECOVERY_S,
    GraphStoreUnavailable,
    WatchedSession,
    WatchedStore,
    _store_answers,
    watched_graphdb_store,
)
from tests._tree_scan import python_sources

#: Construction of a graph container. One call site, in the recipe. The
#: trailing ``(`` is what makes this a construction rather than a mention: the
#: recipe also names the class in an import and a return annotation, and
#: neither of those is a container.
CONTAINER_CONSTRUCTOR = "Neo4jContainer("

#: The module the recipe imports that class from.
CONTAINER_MODULE = "testcontainers.community.neo4j"

#: The superseded path, which is a shim that warns and forwards. It is not a
#: substring of :data:`CONTAINER_MODULE`, so the guard below does not fire on
#: the recipe's own import.
SUPERSEDED_MODULE = "testcontainers.neo4j"

#: The one module allowed to build a graph container, relative to ``tests/``.
RECIPE = "_graphdb_container.py"

#: Every lane that reads a real graph store, relative to ``tests/``. Each one
#: reads it only through :class:`tests._graphdb_container.WatchedSession` or
#: inside :meth:`tests._graphdb_container.WatchedStore.reading`.
REAL_STORE_LANES = (
    "integration/test_graph_index_parity_ties.py",
    "integration/test_graphdb_store.py",
    "integration/test_graph_mcp.py",
)

#: The lanes that hand the seeder a session opened in the lane itself, so they
#: open every such session beside ``store.reading()`` in one ``with``. The
#: tie-parity module is not one: its seeding does that through its own
#: ``_session`` helper, and its module-long session is read only through the
#: watched one.
SEEDER_SESSION_LANES = (
    "integration/test_graphdb_store.py",
    "integration/test_graph_mcp.py",
)

#: A raw driver read. The watched session's methods are ``single`` and
#: ``records``, so a module that reads only through it never spells this.
RAW_DRIVER_READ = ".run("

#: The private halves of the plugin recipe. A lane that assembles its own
#: plugin directory has to call one of these, so naming them names every way
#: of re-implementing :func:`tests._graphdb_container.resolve_plugin_dir`.
#: Matched without a trailing ``(`` so that importing one is caught too:
#: reaching across the module boundary for a private helper is the thing.
PLUGIN_RECIPE_HELPERS = ("_fetch_n10s_jar", "_copy_bundled_apoc")


def test_only_the_shared_recipe_builds_a_graph_container() -> None:
    """A store built anywhere else is a store built to its own recipe."""
    builders = [
        rel for rel, text in python_sources(Path(__file__)) if CONTAINER_CONSTRUCTOR in text
    ]

    assert builders == [RECIPE], (
        f"a graph container is built outside {RECIPE}: {builders}. A lane that "
        f"builds its own container re-states the image pin, the throwaway password "
        f"and the procedure allowlist — call tests._graphdb_container.graphdb_store "
        f"or graphdb_store_published_port instead."
    )


def test_no_module_imports_the_superseded_graph_path() -> None:
    """The short path is a shim that warns and forwards."""
    offenders = [rel for rel, text in python_sources(Path(__file__)) if SUPERSEDED_MODULE in text]

    assert not offenders, (
        f"{SUPERSEDED_MODULE} is imported in: {offenders}. Neo4jContainer comes "
        f"from {CONTAINER_MODULE}."
    )


def test_only_the_shared_recipe_resolves_the_plugins() -> None:
    """A lane that resolves its own plugins resolves them to its own order."""
    offenders = sorted(
        {
            rel
            for rel, text in python_sources(Path(__file__))
            for helper in PLUGIN_RECIPE_HELPERS
            if helper in text
        }
    )

    assert offenders == [RECIPE], (
        f"the plugin recipe is re-implemented outside {RECIPE}: {offenders}. A "
        f"lane that resolves its own plugins re-states the probe order, and the "
        f"copy has already dropped a probe by the time it is noticed — depend on "
        f"the session fixture ``graphdb_plugin_dir`` instead."
    )


def _lane_text(lane: str) -> str:
    """The source of *lane*, a path relative to ``tests/``."""
    texts = [text for rel, text in python_sources(Path(__file__)) if rel == lane]

    assert len(texts) == 1, f"{lane} is not in the tree"
    return texts[0]


def _is_call_to(node: ast.AST, attribute: str) -> bool:
    """Whether *node* calls a method or module attribute named *attribute*."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    )


def _seeder_sessions(text: str) -> list[tuple[int, bool]]:
    """Every ``open_session(...)`` call in *text*, as ``(line, watched)``.

    A call is watched when it is an item of a ``with`` that also enters a
    ``.reading()``, so everything done on the session, its close included,
    runs inside the store's watch.
    """
    tree = ast.parse(text)
    watched = {
        id(item.context_expr)
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(_is_call_to(item.context_expr, "reading") for item in node.items)
        for item in node.items
    }
    return sorted(
        (node.lineno, id(node) in watched)
        for node in ast.walk(tree)
        if _is_call_to(node, "open_session")
    )


@pytest.mark.parametrize("lane", REAL_STORE_LANES)
def test_a_real_store_lane_reads_only_through_the_watch(lane: str) -> None:
    """A raw read there would fail a stalled store as the test's own result."""
    assert RAW_DRIVER_READ not in _lane_text(lane), (
        f"a store read in {lane} bypasses WatchedSession, so a store that stops "
        f"answering would fail it as the test's own result. Read through "
        f"WatchedSession.single / .records."
    )


@pytest.mark.parametrize("lane", SEEDER_SESSION_LANES)
def test_a_lane_opens_its_seeder_sessions_inside_the_watch(lane: str) -> None:
    """The seeder reads a raw session, so the session is opened inside the watch."""
    sessions = _seeder_sessions(_lane_text(lane))

    assert sessions, f"{lane} opens no seeder session, so this guard proves nothing there"
    unwatched = [line for line, watched in sessions if not watched]
    assert not unwatched, (
        f"{lane} opens a seeder session outside the store's watch at line(s) "
        f"{unwatched}, so a stalled store would fail a seeder call there as a driver "
        f"traceback. Open it as ``with store.reading(), "
        f"graph_seeder.open_session(...) as session:``."
    )


# ---------------------------------------------------------------------------
# A store read that gets no answer fails as the store
# ---------------------------------------------------------------------------

#: The store's name in the failures below.
LABEL = "graphdb (neo4j + n10s)"

#: The store's address in the failures below.
URI = "bolt://localhost:50769"


class _FakeResult:
    """A driver result over scripted rows; a row that is an exception is raised."""

    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def single(self) -> object:
        return self.rows[0] if self.rows else None

    def __iter__(self) -> Iterator[object]:
        for row in self.rows:
            if isinstance(row, BaseException):
                raise row
            yield row


class _FakeSession:
    """A driver session that answers each ``run`` with the next scripted outcome."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []

    def run(self, cypher: str, params: dict) -> _FakeResult:
        self.calls.append((cypher, params))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeResult(outcome)


def _watched(outcomes: list[object]) -> tuple[WatchedStore, _FakeSession, WatchedSession]:
    """A watched session on a fake driver session scripted with *outcomes*."""
    store = WatchedStore(URI, label=LABEL)
    fake = _FakeSession(outcomes)
    return store, fake, WatchedSession(fake, store)


def test_a_read_that_gets_no_answer_fails_naming_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure names the store, its address and the test that was reading."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/x.py::test_a[shape] (call)")
    lost = ServiceUnavailable("Failed to read from defunct connection")
    _store, _fake, session = _watched([[{"n": 1}], lost])

    assert session.single("RETURN 1 AS n") == {"n": 1}
    with pytest.raises(GraphStoreUnavailable) as raised:
        session.single("RETURN 1 AS n")

    assert isinstance(raised.value, AssertionError)
    assert raised.value.__cause__ is lost
    message = str(raised.value)
    for fragment in (
        LABEL,
        URI,
        "tests/x.py::test_a[shape] (call)",
        "ServiceUnavailable",
        "Failed to read from defunct connection",
        "not a parity result",
    ):
        assert fragment in message, f"{fragment!r} is missing from: {message}"
    assert "stopped answering 2 times" not in message


def test_a_store_that_stopped_answering_is_asked_again() -> None:
    """A stalled store can come back, so a loss does not fail the reads after it."""
    _store, fake, session = _watched([ServiceUnavailable("defunct"), [{"n": 2}]])

    with pytest.raises(GraphStoreUnavailable):
        session.single("RETURN 2 AS n")
    assert session.single("RETURN 2 AS n") == {"n": 2}

    assert len(fake.calls) == 2


def test_a_second_loss_names_the_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated loss says how often the store stopped answering, and when first."""
    _store, _fake, session = _watched([ServiceUnavailable("defunct"), ServiceUnavailable("gone")])

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/x.py::test_a (call)")
    with pytest.raises(GraphStoreUnavailable):
        session.single("RETURN 1 AS n")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/x.py::test_b (call)")
    with pytest.raises(GraphStoreUnavailable) as raised:
        session.single("RETURN 1 AS n")

    message = str(raised.value)
    assert "test_b (call)" in message.splitlines()[0]
    assert "stopped answering 2 times" in message
    assert "the first was during tests/x.py::test_a (call)" in message


def test_a_result_that_breaks_while_it_is_read_fails_the_same_way() -> None:
    """The records are fetched inside the guard, so a mid-stream loss is caught too."""
    _store, _fake, session = _watched([[{"n": 1}, ServiceUnavailable("defunct mid-stream")]])

    with pytest.raises(GraphStoreUnavailable) as raised:
        session.records("MATCH (n) RETURN n")

    assert "defunct mid-stream" in str(raised.value)


@pytest.mark.parametrize(
    "error",
    [
        ClientError("Invalid input 'MATCH'"),
        AssertionError("a test's own check"),
        GraphUnreachable("The graph store did not answer: no route"),
    ],
    ids=["client-error", "assertion", "context-error-without-a-lost-store"],
)
def test_any_other_read_failure_is_left_as_it_is(error: Exception) -> None:
    """Only a store that stops answering is renamed; every other failure is its own."""
    wait = _Wait()
    store = WatchedStore(URI, label=LABEL, wait_for_store=wait)
    session = WatchedSession(_FakeSession([error]), store)

    with pytest.raises(type(error)) as raised:
        session.single("MATCH (n) RETURN n")

    assert raised.value is error
    assert not isinstance(raised.value, GraphStoreUnavailable)
    assert store.losses == []
    assert wait.calls == 0


class _Inspector:
    """Says a fixed container state and counts how often it was asked."""

    def __init__(self, state: str) -> None:
        self.state = state
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return self.state


class _Wait:
    """Stands in for the wait on a lost store: counts calls, then raises *error* or returns."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


def test_a_read_that_loses_the_store_waits_for_it_and_reads_again() -> None:
    """Once the store answers again, the same read runs once more and its answer is the result."""
    wait = _Wait()
    store = WatchedStore(URI, label=LABEL, wait_for_store=wait)
    fake = _FakeSession([ServiceUnavailable("defunct"), [{"n": 2}]])
    session = WatchedSession(fake, store)

    assert session.single("RETURN 2 AS n") == {"n": 2}

    assert wait.calls == 1
    assert len(fake.calls) == 2
    assert len(store.losses) == 1


@pytest.mark.parametrize(
    "waited",
    [
        AssertionError(
            f"{LABEL} at {URI}: started, but never answered — it went 120s without visible progress"
        ),
        ContainerExitedError(
            f"{LABEL} at {URI}: the container reached state 'exited' during the wait\n"
            "exit code: 137"
        ),
    ],
    ids=["no-answer", "exited"],
)
def test_a_store_that_does_not_answer_again_fails_naming_the_wait(
    waited: AssertionError,
) -> None:
    """A store that does not come back fails as the store, saying what the wait saw."""
    lost = ServiceUnavailable("defunct")
    wait = _Wait(waited)
    store = WatchedStore(URI, label=LABEL, wait_for_store=wait)
    fake = _FakeSession([lost])
    session = WatchedSession(fake, store)

    with pytest.raises(GraphStoreUnavailable) as raised:
        session.single("RETURN 1 AS n")

    assert raised.value.__cause__ is lost
    message = str(raised.value)
    for fragment in (LABEL, URI, "did not answer again", str(waited)):
        assert fragment in message, f"{fragment!r} is missing from: {message}"
    assert wait.calls == 1
    assert len(fake.calls) == 1


def test_a_reread_that_loses_the_store_again_fails() -> None:
    """One wait and one re-read per call: a second loss is the failure."""
    gone = ServiceUnavailable("gone")
    wait = _Wait()
    store = WatchedStore(URI, label=LABEL, wait_for_store=wait)
    session = WatchedSession(_FakeSession([ServiceUnavailable("defunct"), gone]), store)

    with pytest.raises(GraphStoreUnavailable) as raised:
        session.single("RETURN 1 AS n")

    assert raised.value.__cause__ is gone
    assert wait.calls == 1
    message = str(raised.value)
    assert "answered again" in message
    assert "stopped answering 2 times" in message


def test_a_lost_store_behind_the_graph_context_error_is_a_loss() -> None:
    """The graph context raises its own error from the driver's, so the cause decides."""
    wait = _Wait()
    store = WatchedStore(URI, label=LABEL, wait_for_store=wait)
    outcomes: list[Callable[[], int]] = []

    def lost() -> int:
        try:
            raise ServiceUnavailable("defunct")
        except ServiceUnavailable as exc:
            raise GraphUnreachable("The graph store did not answer: defunct") from exc

    outcomes.extend([lost, lambda: 7])

    assert store.read(lambda: outcomes.pop(0)()) == 7
    assert wait.calls == 1


def test_seeding_inside_reading_never_waits() -> None:
    """A block that writes cannot be run a second time, so it fails at once."""
    wait = _Wait()
    store = WatchedStore(URI, label=LABEL, wait_for_store=wait)

    with pytest.raises(GraphStoreUnavailable):
        with store.reading():
            raise ServiceUnavailable("defunct")

    assert wait.calls == 0


def test_a_loss_already_reported_inside_reading_passes_through() -> None:
    """A watched read inside ``reading()`` reports its loss once; the block adds none."""
    gone = ServiceUnavailable("gone")
    store = WatchedStore(URI, label=LABEL, wait_for_store=_Wait())
    session = WatchedSession(_FakeSession([ServiceUnavailable("defunct"), gone]), store)

    with pytest.raises(GraphStoreUnavailable) as raised:
        with store.reading():
            session.single("RETURN 1 AS n")

    assert raised.value.__cause__ is gone
    assert len(store.losses) == 2
    assert "stopped answering 2 times" in str(raised.value)


def test_the_failure_says_what_state_the_container_was_in() -> None:
    """A lost read adds its container's state, read once, after the first line."""
    inspector = _Inspector("exited (code 137, out of memory)")
    store = WatchedStore(URI, label=LABEL, inspect=inspector)
    session = WatchedSession(_FakeSession([ServiceUnavailable("defunct"), [{"n": 1}]]), store)

    with pytest.raises(GraphStoreUnavailable) as raised:
        session.single("RETURN 1 AS n")
    assert session.single("RETURN 1 AS n") == {"n": 1}

    lines = str(raised.value).splitlines()
    assert lines[0].startswith(f"{LABEL} at {URI} stopped answering during ")
    assert "Container state when the read failed: exited (code 137, out of memory)" in lines
    assert inspector.calls == 1


def test_without_an_inspector_the_failure_says_nothing_about_the_container() -> None:
    """A store built from a URI and a label alone keeps the message it had."""
    _store, _fake, session = _watched([ServiceUnavailable("defunct")])

    with pytest.raises(GraphStoreUnavailable) as raised:
        session.single("RETURN 1 AS n")

    message = str(raised.value)
    assert "Container state" not in message
    assert message.endswith("not at the code under test.")


def test_an_inspector_that_fails_leaves_the_lost_read_as_the_failure() -> None:
    """An inspector that raises is reported in the line, never over the lost read."""

    def inspect() -> str:
        raise RuntimeError("daemon went away")

    lost = ServiceUnavailable("defunct")
    store = WatchedStore(URI, label=LABEL, inspect=inspect)
    session = WatchedSession(_FakeSession([lost]), store)

    with pytest.raises(GraphStoreUnavailable) as raised:
        session.single("RETURN 1 AS n")

    assert raised.value.__cause__ is lost
    assert (
        "Container state when the read failed: could not be read (RuntimeError: daemon went away)"
        in str(raised.value).splitlines()
    )


def test_the_watched_store_inspects_its_own_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """The store it yields reads the state of the container it started, then stops it."""

    class _StartedContainer:
        def __init__(self) -> None:
            self.stopped = False

        def get_connection_url(self) -> str:
            return URI

        def stop(self) -> None:
            self.stopped = True

    started = _StartedContainer()
    inspected: list[object] = []

    def container_state(container: object) -> str:
        inspected.append(container)
        return "paused"

    waited_on: list[object] = []

    def wait_until_ready(*_args: object, **kwargs: object) -> None:
        waited_on.append(kwargs["container"])
        raise AssertionError("store did not answer")

    monkeypatch.setattr("tests._graphdb_container.start_or_skip", lambda factory, *, label: started)
    monkeypatch.setattr("tests._graphdb_container.container_state", container_state)
    monkeypatch.setattr("tests._graphdb_container.wait_until_ready", wait_until_ready)

    with watched_graphdb_store(Path("plugins"), label=LABEL) as store:
        session = WatchedSession(_FakeSession([ServiceUnavailable("defunct")]), store)
        with pytest.raises(GraphStoreUnavailable) as raised:
            session.single("RETURN 1 AS n")

    assert store.uri == URI
    assert store.label == LABEL
    assert waited_on == [started]
    assert inspected == [started]
    assert "Container state when the read failed: paused" in str(raised.value).splitlines()
    assert started.stopped


def test_the_watched_store_waits_on_its_own_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lost read waits on the container the store started, probing the store's URI."""

    class _StartedContainer:
        def get_connection_url(self) -> str:
            return URI

        def stop(self) -> None:
            pass

    started = _StartedContainer()
    waits: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def wait_until_ready(*args: object, **kwargs: object) -> None:
        waits.append((args, kwargs))

    monkeypatch.setattr("tests._graphdb_container.start_or_skip", lambda factory, *, label: started)
    monkeypatch.setattr("tests._graphdb_container.container_state", lambda container: "running")
    monkeypatch.setattr("tests._graphdb_container.wait_until_ready", wait_until_ready)

    with watched_graphdb_store(Path("plugins"), label=LABEL) as store:
        session = WatchedSession(_FakeSession([ServiceUnavailable("defunct"), [{"n": 1}]]), store)
        assert session.single("RETURN 1 AS n") == {"n": 1}

    assert len(waits) == 1
    (probe, *_rest), kwargs = waits[0]
    assert kwargs["container"] is started
    assert kwargs["timeout"] == STORE_RECOVERY_S
    assert kwargs["ceiling"] == STORE_RECOVERY_S
    assert kwargs["interval"] == STORE_PROBE_INTERVAL_S
    assert isinstance(probe, functools.partial)
    assert probe.func is _store_answers
    assert probe.args == (URI,)


class _FakeDriver:
    """A driver that records how it was built, what it was asked, and whether it closed."""

    def __init__(self, uri: str, error: BaseException | None = None, **kwargs: object) -> None:
        self.uri = uri
        self.kwargs = kwargs
        self.error = error
        self.queries: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    def execute_query(self, query: str, **kwargs: object) -> object:
        self.queries.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return object()

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("error", [None, ServiceUnavailable("defunct")], ids=["answers", "lost"])
def test_the_store_probe_asks_the_test_database_with_no_driver_retry(
    monkeypatch: pytest.MonkeyPatch, error: BaseException | None
) -> None:
    """One short query on the test database, no retry inside the driver, driver always closed."""
    import neo4j

    drivers: list[_FakeDriver] = []

    def driver(uri: str, **kwargs: object) -> _FakeDriver:
        drivers.append(_FakeDriver(uri, error, **kwargs))
        return drivers[-1]

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", driver)

    if error is None:
        _store_answers(URI)
    else:
        with pytest.raises(ServiceUnavailable) as raised:
            _store_answers(URI)
        assert raised.value is error

    (built,) = drivers
    assert built.uri == URI
    assert built.queries == [("RETURN 1", {"database_": GRAPHDB_TEST_DATABASE})]
    assert built.kwargs["connection_timeout"] == STORE_PROBE_TIMEOUT_S
    assert built.kwargs["connection_acquisition_timeout"] == STORE_PROBE_TIMEOUT_S
    assert built.kwargs["max_transaction_retry_time"] == 0.0
    assert built.closed


def test_the_store_probe_fails_fast_on_a_port_nothing_answers() -> None:
    """On the real driver the probe raises at once: it hides no retry loop."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    started = time.monotonic()
    with pytest.raises(ServiceUnavailable):
        _store_answers(f"bolt://127.0.0.1:{port}")

    assert time.monotonic() - started < STORE_PROBE_TIMEOUT_S
