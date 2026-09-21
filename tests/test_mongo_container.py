"""Tree-read guards for the shared throwaway-MongoDB recipe.

Two properties hold the arrangement in ``tests/_mongo_container.py`` together,
and neither belongs to any one suite: a Mongo container is built in exactly one
module, and no module reaches for the superseded import path. A store that runs
is the only other thing that would prove them, which is to say they would go
unproved on every host without a container engine, since each of the fixtures
concerned skips there. So they are read off the tree instead, which works
wherever the suite is collected. This file exempts itself from the scan,
because both guards below name what they forbid.
"""

from __future__ import annotations

from pathlib import Path

from tests._tree_scan import python_sources

#: Construction of a Mongo container. One call site, in the recipe.
CONTAINER_CONSTRUCTOR = "MongoDbContainer("

#: The module the recipe imports that class from.
CONTAINER_MODULE = "testcontainers.community.mongodb"

#: The superseded path, which is a shim that warns and forwards.
SUPERSEDED_MODULE = "testcontainers.mongodb"

#: The one module allowed to build a Mongo container, relative to ``tests/``.
RECIPE = "_mongo_container.py"


def test_only_the_shared_recipe_builds_a_mongo_container() -> None:
    """A store built anywhere else is a store nothing waited for."""
    builders = [
        rel for rel, text in python_sources(Path(__file__)) if CONTAINER_CONSTRUCTOR in text
    ]

    assert builders == [RECIPE], (
        f"a Mongo container is built outside {RECIPE}: {builders}. A fixture that "
        f"builds its own container skips the readiness wait and can hand its tests a "
        f"port that belongs to a server on its way out — call "
        f"tests._mongo_container.started_mongo instead."
    )


def test_no_module_imports_the_superseded_mongo_path() -> None:
    """The short path is a shim that warns and forwards."""
    offenders = [rel for rel, text in python_sources(Path(__file__)) if SUPERSEDED_MODULE in text]

    assert not offenders, (
        f"{SUPERSEDED_MODULE} is imported in: {offenders}. MongoDbContainer comes "
        f"from {CONTAINER_MODULE}."
    )
