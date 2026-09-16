"""Tree-read guards for the shared throwaway-MongoDB recipe.

Two properties hold the arrangement in ``tests/_mongo_container.py`` together,
and neither belongs to any one suite: a Mongo container is built in exactly one
module, and no module reaches for the superseded import path. A store that runs
is the only other thing that would prove them, which is to say they would go
unproved on every host without a container engine, since each of the fixtures
concerned skips there. So they are read off the tree instead, which works
wherever the suite is collected.
"""

from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent

#: Construction of a Mongo container. One call site, in the recipe.
CONTAINER_CONSTRUCTOR = "MongoDbContainer("

#: The module the recipe imports that class from.
CONTAINER_MODULE = "testcontainers.community.mongodb"

#: The superseded path, which is a shim that warns and forwards.
SUPERSEDED_MODULE = "testcontainers.mongodb"

#: The one module allowed to build a Mongo container, relative to ``tests/``.
RECIPE = "_mongo_container.py"


def _scan() -> list[tuple[str, str]]:
    """Every Python file under ``tests/`` with its text, as ``(relative path, text)``.

    This file is left out: both guards below name what they forbid, so a scan
    that included it would report itself. ``test_no_legacy_symbols.py`` exempts
    itself from its own scan for the same reason.
    """
    self_path = Path(__file__).resolve()
    found: list[tuple[str, str]] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.resolve() == self_path:
            continue
        found.append((str(path.relative_to(TESTS_ROOT)), path.read_text(encoding="utf-8")))
    return found


def test_only_the_shared_recipe_builds_a_mongo_container() -> None:
    """A store built anywhere else is a store nothing waited for."""
    builders = [rel for rel, text in _scan() if CONTAINER_CONSTRUCTOR in text]

    assert builders == [RECIPE], (
        f"a Mongo container is built outside {RECIPE}: {builders}. A fixture that "
        f"builds its own container skips the readiness wait and can hand its tests a "
        f"port that belongs to a server on its way out — call "
        f"tests._mongo_container.started_mongo instead."
    )


def test_no_module_imports_the_superseded_mongo_path() -> None:
    """The short path is a shim that warns and forwards."""
    offenders = [rel for rel, text in _scan() if SUPERSEDED_MODULE in text]

    assert not offenders, (
        f"{SUPERSEDED_MODULE} is imported in: {offenders}. MongoDbContainer comes "
        f"from {CONTAINER_MODULE}."
    )
