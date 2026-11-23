"""The merged export of each MML fixture tree, for tests that read a tree's export.

The trees stay under ``tests/fixtures/mml/``; this helper loads one tree's
exports through the mml layer's JSON reader and merges them, so no test has to
import another test module to get at a tree's export.
"""

from __future__ import annotations

from typing import Any

from osprey_facility.layers.mml.loaders.json_any import VA_SUFFIX, load_json, load_sibling
from osprey_facility.layers.mml.systems import merge_inputs
from osprey_facility_tests._harness import REPO_ROOT
from osprey_facility_tests._mml_trees import SUPPORTED

__all__ = ["FIXTURES", "TREES", "fixture_export"]

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

#: Each tree's exports, by the stem every file of one export is named after.
TREES = {tree.name: tree.exports for tree in SUPPORTED}


def fixture_export(tree: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The tree's merged export: ``(ao, ad, va)``, each keyed by system."""
    loaded = [load_json(FIXTURES / tree / f"{stem}.ao.json") for stem in TREES[tree]]
    ao, ad = merge_inputs([(one, None) for one in loaded])
    va = {
        one.export["submachine"]: load_sibling(FIXTURES / tree / f"{stem}{VA_SUFFIX}")
        for stem, one in zip(TREES[tree], loaded, strict=True)
        if one.export is not None
    }
    return ao, ad, va
