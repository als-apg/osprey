"""Tree-read guards for the shared throwaway-graph-store recipe.

Two properties hold the arrangement in ``tests/_graphdb_container.py``
together, and neither belongs to any one lane: a graph container is built in
exactly one module, and no module reaches for the superseded import path. A
store that runs is the only other thing that would prove them, which is to say
they would go unproved on every host without a container engine or without the
neo4j extra, since each of the lanes concerned skips there. So they are read
off the tree instead, which works wherever the suite is collected. This file
exempts itself from the scan, because both guards below name what they forbid.
"""

from __future__ import annotations

from pathlib import Path

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
