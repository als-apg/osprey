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

The module also pins, against a fake driver session, how a store read that
gets no answer fails: as :class:`tests._graphdb_container.GraphStoreUnavailable`,
naming the store, with the next read still asking it. The fast lane has no
container, so a fake stands in for the driver there.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from neo4j.exceptions import ClientError, ServiceUnavailable

from tests._graphdb_container import GraphStoreUnavailable, WatchedSession, WatchedStore
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

#: The tie-parity module, relative to ``tests/``, whose store reads must all go
#: through :class:`tests._graphdb_container.WatchedSession`.
TIE_PARITY_MODULE = "integration/test_graph_index_parity_ties.py"

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


def test_the_tie_parity_module_reads_its_store_only_through_the_watch() -> None:
    """A raw read there would fail a stalled store as a parity result."""
    texts = [text for rel, text in python_sources(Path(__file__)) if rel == TIE_PARITY_MODULE]

    assert len(texts) == 1, f"{TIE_PARITY_MODULE} is not in the tree"
    assert RAW_DRIVER_READ not in texts[0], (
        "a store read in the tie-parity module bypasses WatchedSession, so a store "
        "that stops answering would fail it as a parity result. Read through "
        "ties_session.single / .records."
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
    [ClientError("Invalid input 'MATCH'"), AssertionError("a test's own check")],
    ids=["client-error", "assertion"],
)
def test_any_other_read_failure_is_left_as_it_is(error: Exception) -> None:
    """Only a store that stops answering is renamed; every other failure is its own."""
    store, _fake, session = _watched([error])

    with pytest.raises(type(error)) as raised:
        session.single("MATCH (n) RETURN n")

    assert raised.value is error
    assert not isinstance(raised.value, GraphStoreUnavailable)
    assert store.losses == []
