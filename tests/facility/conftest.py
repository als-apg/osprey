"""Builds the facility tests make of their own trees.

``build_project`` builds a small project whose ``data/facility/`` a test writes.
The shared control-assistant build, ``built_control_assistant``, lives in
``tests/_builds.py``, beside :class:`BuiltProject`, which the facility modules
import from here.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import Result

from tests._builds import BuiltProject, init_project, run_build
from tests.facility._synthetic_trees import write_tree

__all__ = ["BuiltProject"]


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
