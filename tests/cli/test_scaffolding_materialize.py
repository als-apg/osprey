"""Tests for :func:`osprey.cli.templates.scaffolding.materialize_tier_artifacts`.

Covers the build-time step that flattens tier-routed channel databases AND
the matching benchmark query file:

* single-paradigm modes (``in_context`` / ``hierarchical`` / ``middle_layer``)
* tier 3 selection materializes the right source DB (byte-exact)
* queries file is materialized to ``data/benchmarks/queries.json`` (byte-exact)
* missing source DB raises :class:`FileNotFoundError`
* ``tiers/`` and ``benchmarks/cross_paradigm/`` subtrees are pruned after a
  successful run
* invalid/legacy modes (``"all"``, ``None``, ``"bogus"``) are rejected
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from osprey.cli.templates.scaffolding import (
    materialize_tier_artifacts,
    prune_csv_build_artifacts,
)

# Absolute paths to the preset's source-of-truth artifacts.
_PRESET_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
)
_PRESET_CDB_DIR = _PRESET_DATA_DIR / "channel_databases"
_PRESET_QUERIES_DIR = _PRESET_DATA_DIR / "benchmarks" / "cross_paradigm" / "queries"

_PARADIGMS: tuple[str, ...] = ("in_context", "hierarchical", "middle_layer")


def _make_rendered_project(tmp_path: Path) -> Path:
    """Build a fake post-``osprey build`` tree.

    Mirrors what :func:`copy_template_data` produces: the entire
    ``channel_databases/`` tree (including ``tiers/tier{1,2,3}/``) and the
    ``benchmarks/cross_paradigm/queries/`` tree both live under
    ``<project>/data/``.
    """
    project_dir = tmp_path / "proj"
    cdb_dst = project_dir / "data" / "channel_databases"
    cdb_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_PRESET_CDB_DIR, cdb_dst)

    queries_dst = project_dir / "data" / "benchmarks" / "cross_paradigm" / "queries"
    queries_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_PRESET_QUERIES_DIR, queries_dst)
    return project_dir


@pytest.fixture()
def rendered_project(tmp_path: Path) -> Path:
    """Tmp-path project with the preset channel_databases + queries trees copied in."""
    return _make_rendered_project(tmp_path)


class TestSingleParadigmModes:
    """Single-paradigm modes materialize only that paradigm."""

    @pytest.mark.parametrize("paradigm", _PARADIGMS)
    def test_single_mode_only_materializes_that_paradigm(
        self, rendered_project: Path, paradigm: str
    ):
        materialize_tier_artifacts(rendered_project, tier=1, channel_finder_mode=paradigm)

        cdb = rendered_project / "data" / "channel_databases"

        # The requested paradigm got copied with matching content.
        flat = cdb / f"{paradigm}.json"
        assert flat.is_file()
        expected = (_PRESET_CDB_DIR / "tiers" / "tier1" / f"{paradigm}.json").read_bytes()
        assert flat.read_bytes() == expected

        # The other paradigms were NOT created at the flat root.
        for other in _PARADIGMS:
            if other == paradigm:
                continue
            assert not (cdb / f"{other}.json").exists(), (
                f"{other}.json should not have been materialized for mode={paradigm!r}"
            )

        # Queries file landed at the flat path, byte-equal to the tier source.
        queries_flat = rendered_project / "data" / "benchmarks" / "queries.json"
        assert queries_flat.is_file()
        expected_queries = (_PRESET_QUERIES_DIR / "tier1_queries.json").read_bytes()
        assert queries_flat.read_bytes() == expected_queries

        # Both staging subtrees are pruned regardless of which paradigm was picked.
        assert not (cdb / "tiers").exists()
        assert not (rendered_project / "data" / "benchmarks" / "cross_paradigm").exists()


class TestTier3Selection:
    """Tier 3 must pull from `tiers/tier3/<paradigm>.json` and
    `cross_paradigm/queries/tier3_queries.json`, byte-exact."""

    @pytest.mark.parametrize("paradigm", _PARADIGMS)
    def test_tier3_flat_file_byte_equals_preset_tier3(self, rendered_project: Path, paradigm: str):
        materialize_tier_artifacts(rendered_project, tier=3, channel_finder_mode=paradigm)

        flat = rendered_project / "data" / "channel_databases" / f"{paradigm}.json"
        expected = (_PRESET_CDB_DIR / "tiers" / "tier3" / f"{paradigm}.json").read_bytes()
        assert flat.read_bytes() == expected

        # Queries file for tier 3 lands at the flat path.
        queries_flat = rendered_project / "data" / "benchmarks" / "queries.json"
        expected_queries = (_PRESET_QUERIES_DIR / "tier3_queries.json").read_bytes()
        assert queries_flat.read_bytes() == expected_queries


class TestMissingSourceDb:
    """Missing source DB raises FileNotFoundError and leaves the tree intact."""

    def test_missing_source_raises_file_not_found(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        tier_dir = project_dir / "data" / "channel_databases" / "tiers" / "tier1"
        tier_dir.mkdir(parents=True)
        # Put two of three paradigm files in place; middle_layer.json is missing.
        (tier_dir / "in_context.json").write_text("{}")
        (tier_dir / "hierarchical.json").write_text("{}")

        with pytest.raises(FileNotFoundError, match="middle_layer"):
            materialize_tier_artifacts(project_dir, tier=1, channel_finder_mode="middle_layer")

        # On failure the tiers/ tree must still be present (no partial rmtree).
        assert tier_dir.exists()
        # No flat files should have been created (failure happens before any copy).
        cdb = project_dir / "data" / "channel_databases"
        for paradigm in _PARADIGMS:
            assert not (cdb / f"{paradigm}.json").exists()

    def test_missing_source_for_single_mode_raises(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        # tier2 dir exists but is empty — single-paradigm request should fail.
        (project_dir / "data" / "channel_databases" / "tiers" / "tier2").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="hierarchical"):
            materialize_tier_artifacts(project_dir, tier=2, channel_finder_mode="hierarchical")


class TestMissingQueriesFile:
    """Missing queries file is also a hard error and leaves the tree intact."""

    def test_missing_queries_raises_file_not_found(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        # Stage tier DB but NOT the queries subtree.
        tier_dir = project_dir / "data" / "channel_databases" / "tiers" / "tier1"
        tier_dir.mkdir(parents=True)
        (tier_dir / "in_context.json").write_text("{}")

        with pytest.raises(FileNotFoundError, match="queries"):
            materialize_tier_artifacts(project_dir, tier=1, channel_finder_mode="in_context")

        # Failure happens before any copy/rmtree — tiers/ still in place,
        # no flat artifacts created.
        assert tier_dir.exists()
        assert not (project_dir / "data" / "channel_databases" / "in_context.json").exists()
        assert not (project_dir / "data" / "benchmarks" / "queries.json").exists()


class TestSubtreePruning:
    """Both staging subtrees must be absent after a successful materialization."""

    def test_post_rmtree_absence(self, rendered_project: Path):
        cdb = rendered_project / "data" / "channel_databases"
        cross_paradigm = rendered_project / "data" / "benchmarks" / "cross_paradigm"
        # Sanity-check the fixture before we run.
        assert (cdb / "tiers").is_dir()
        assert (cdb / "tiers" / "tier1").is_dir()
        assert (cross_paradigm / "queries").is_dir()

        materialize_tier_artifacts(rendered_project, tier=1, channel_finder_mode="hierarchical")

        assert not (cdb / "tiers").exists()
        assert not cross_paradigm.exists()


class TestUnknownMode:
    """Unrecognized/legacy channel_finder_mode values are hard errors."""

    @pytest.mark.parametrize("mode", ["bogus", "all", None])
    def test_unknown_mode_raises_value_error(self, rendered_project: Path, mode):
        with pytest.raises(ValueError, match="Unknown channel_finder_mode"):
            materialize_tier_artifacts(rendered_project, tier=1, channel_finder_mode=mode)


class TestNoOpOnMissingTiersSubtree:
    """Bundles without channel-finder DBs have no `tiers/` — must be a no-op."""

    def test_no_op_when_tiers_subtree_absent(self, tmp_path: Path):
        # Mimic a hello_world-style bundle: data/ exists but no channel DBs.
        project_dir = tmp_path / "proj"
        (project_dir / "data").mkdir(parents=True)
        # Must NOT raise, even with a fully-valid tier/mode combination.
        materialize_tier_artifacts(project_dir, tier=1, channel_finder_mode="hierarchical")
        # Nothing got created — flat root doesn't exist either.
        assert not (project_dir / "data" / "channel_databases").exists()


class TestPruneCsvBuildArtifacts:
    """``data/raw/`` is removed for paradigms without a CSV → DB build path."""

    def _make_raw_dir(self, project_dir: Path) -> Path:
        raw = project_dir / "data" / "raw"
        raw.mkdir(parents=True)
        (raw / "address_list.csv").write_text("channel,address,description\n")
        (raw / "CSV_EXAMPLE.csv").write_text("channel,address,description\n")
        return raw

    def test_in_context_keeps_raw_dir(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        raw = self._make_raw_dir(project_dir)
        prune_csv_build_artifacts(project_dir, channel_finder_mode="in_context")
        assert raw.is_dir()
        assert (raw / "address_list.csv").exists()

    @pytest.mark.parametrize("mode", ["hierarchical", "middle_layer"])
    def test_non_in_context_removes_raw_dir(self, tmp_path: Path, mode: str):
        project_dir = tmp_path / "proj"
        raw = self._make_raw_dir(project_dir)
        prune_csv_build_artifacts(project_dir, channel_finder_mode=mode)
        assert not raw.exists()

    def test_no_op_when_raw_dir_absent(self, tmp_path: Path):
        # Bundles that don't ship a raw/ directory (e.g. hello_world) must not
        # raise. Run with a non-in_context mode to exercise the second guard.
        project_dir = tmp_path / "proj"
        (project_dir / "data").mkdir(parents=True)
        prune_csv_build_artifacts(project_dir, channel_finder_mode="hierarchical")
        # data/ still exists; nothing else was created.
        assert (project_dir / "data").is_dir()
        assert not (project_dir / "data" / "raw").exists()
