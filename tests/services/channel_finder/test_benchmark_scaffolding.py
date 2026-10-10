"""The build copies the benchmark query file that matches ``channel_finder_mode``.

The control-assistant tree ships two query sources under
``data/benchmarks/cross_paradigm/queries/``: ``in_context_queries.json`` for the
in_context pipeline and ``tree_queries.json`` for every other mode.
:func:`materialize_benchmark_queries` copies the matching one to
``data/benchmarks/queries.json`` and prunes the staging subtree.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from osprey.build.modes import VALID_CHANNEL_FINDER_MODES
from osprey.cli.templates.scaffolding import materialize_benchmark_queries

REPO_ROOT = Path(__file__).resolve().parents[3]
QUERIES_DIR = (
    REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/benchmarks/cross_paradigm/queries"
)

EXPECTED_SOURCE = {
    "in_context": "in_context_queries.json",
    "hierarchical": "tree_queries.json",
    "middle_layer": "tree_queries.json",
    "graph": "tree_queries.json",
}


def test_shipped_query_files_are_named_by_pipeline() -> None:
    assert sorted(p.name for p in QUERIES_DIR.glob("*.json")) == [
        "in_context_queries.json",
        "tree_queries.json",
    ]


def test_every_mode_has_an_expected_source() -> None:
    assert set(EXPECTED_SOURCE) == set(VALID_CHANNEL_FINDER_MODES)


@pytest.mark.parametrize("mode", sorted(EXPECTED_SOURCE))
def test_render_carries_the_mode_matching_queries(tmp_path: Path, mode: str) -> None:
    project_dir = tmp_path / "project"
    staged = project_dir / "data" / "benchmarks" / "cross_paradigm" / "queries"
    shutil.copytree(QUERIES_DIR, staged)

    materialize_benchmark_queries(project_dir, mode)

    queries = project_dir / "data" / "benchmarks" / "queries.json"
    expected = QUERIES_DIR / EXPECTED_SOURCE[mode]
    assert queries.read_bytes() == expected.read_bytes()
    assert not (project_dir / "data" / "benchmarks" / "cross_paradigm").exists()


def test_missing_mode_source_stops_before_anything_is_removed(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    staged = project_dir / "data" / "benchmarks" / "cross_paradigm" / "queries"
    staged.mkdir(parents=True)
    (staged / "tree_queries.json").write_text("[]\n")

    with pytest.raises(FileNotFoundError, match="in_context_queries.json"):
        materialize_benchmark_queries(project_dir, "in_context")

    assert (staged / "tree_queries.json").exists()
    assert not (project_dir / "data" / "benchmarks" / "queries.json").exists()
