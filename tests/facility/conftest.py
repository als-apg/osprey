"""Builds the facility tests make of their own trees.

``build_project`` builds a small project whose ``data/facility/`` a test writes.
The shared demo build, ``built_demo``, lives in
``tests/_builds.py``, beside :class:`BuiltProject`, which the facility modules
import from here. ``mml_built``, each MML fixture tree built once per session,
lives in ``tests/facility/_mml_built.py``.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import Result

from tests._builds import BuiltProject, init_project, run_build
from tests.facility._fixture_kind import read_entry_points_afresh
from tests.facility._mml_built import mml_built  # noqa: F401 - the session MML builds
from tests.facility._synthetic_trees import write_tree

__all__ = ["BuiltProject"]


@pytest.fixture
def entry_points_read_afresh() -> Iterator[None]:
    """The test reads the entry points of its own path, and leaves none read behind.

    For a module that installs a fixture distribution: the session's one reading
    of the entry points is dropped before the test and again after it, so a
    reading made while the package was on the path (by the code, or by a module
    first imported then, which keeps the name it bound) does not outlive it.
    """
    read_entry_points_afresh()
    yield
    read_entry_points_afresh()


@pytest.fixture
def build_project(
    tmp_path: Path,
) -> Callable[..., tuple[BuiltProject, Result]]:
    """Build a hello-world repo whose ``data/facility/`` holds the given tree.

    The returned callable takes the tree (``None`` for no ``data/facility/`` at
    all), the repo name and a ``profile`` mapping deep-merged into the repo's
    ``profile.yml`` before the build, and returns the project and the build's
    result.
    """

    def build(
        tree: Mapping[str, Any] | None,
        name: str = "demo",
        profile: Mapping[str, Any] | None = None,
    ) -> tuple[BuiltProject, Result]:
        repo = init_project(tmp_path, "hello-world", name)
        facility_dir = repo / "data" / "facility"
        shutil.rmtree(facility_dir)
        if tree is not None:
            write_tree(facility_dir, tree)
        if profile is not None:
            profile_file = repo / "profile.yml"
            merged = _deep_merge(yaml.safe_load(profile_file.read_text(encoding="utf-8")), profile)
            profile_file.write_text(yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")
        return BuiltProject(repo), run_build(repo)

    return build


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``over`` merged in: mappings merge key by key, anything else replaces."""
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
