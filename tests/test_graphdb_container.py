"""Tree-read guards for the shared throwaway-graph-store recipe.

The properties that hold the arrangement in ``tests/_graphdb_container.py``
together belong to no one lane: a graph container is built in exactly one
module, no module reaches for the superseded import path, the plugin recipe
has exactly one implementation, and every lane that reads a real store reads
it only through the store's watch. A store that runs is the only other thing
that would prove them, which is to say they would go unproved on every host
without a container engine or without the neo4j extra, since each of the lanes
concerned skips there. So they are read off the tree instead, which works
wherever the suite is collected. This file exempts itself from the scan,
because the guards below name what they forbid.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
#: reads it only through :class:`tests._graphdb_container.WatchedSession`,
#: inside :meth:`tests._graphdb_container.WatchedStore.reading`, or through
#: :meth:`tests._graphdb_container.WatchedStore.read`.
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

#: The graph context's read; a lane that calls it directly calls it only
#: inside :meth:`tests._graphdb_container.WatchedStore.read`.
CONTEXT_READ = "run_read"

#: The lanes that read the store through the graph context directly, relative
#: to ``tests/``.
CONTEXT_READ_LANES = ("integration/test_graph_mcp.py",)

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


def _unwatched_context_reads(text: str) -> list[int]:
    """The lines of every ``run_read(...)`` call in *text* not inside a ``.read(...)``."""
    tree = ast.parse(text)
    watched = {
        id(inner)
        for node in ast.walk(tree)
        if _is_call_to(node, "read")
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]
        for inner in ast.walk(argument)
    }
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if _is_call_to(node, CONTEXT_READ) and id(node) not in watched
    )


@pytest.mark.parametrize("lane", CONTEXT_READ_LANES)
def test_a_lane_reads_the_graph_context_only_through_the_watch(lane: str) -> None:
    """A direct context read outside the watch would fail a stalled store as the product's error."""
    text = _lane_text(lane)

    assert f"{CONTEXT_READ}(" in text, (
        f"{lane} calls no {CONTEXT_READ}(, so this guard proves nothing there"
    )
    unwatched = _unwatched_context_reads(text)
    assert unwatched == [], (
        f"{lane} reads the graph context outside the store's watch at line(s) "
        f"{unwatched}, so a stalled store would fail such a read as the product's "
        f"GraphUnreachable. Read through ``store.read(lambda: context.{CONTEXT_READ}(...))``."
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
