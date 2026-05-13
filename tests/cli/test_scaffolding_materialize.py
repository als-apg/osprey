"""Tests for :func:`osprey.cli.templates.scaffolding.materialize_tier_dbs`.

Covers the build-time step that flattens tier-routed channel databases:

* happy path with every paradigm (``channel_finder_mode='all'`` / ``None``)
* single-paradigm modes (``in_context`` / ``hierarchical`` / ``middle_layer``)
* tier 3 selection materializes the right source DB (byte-exact)
* missing source DB raises :class:`FileNotFoundError`
* ``tiers/`` subtree is pruned after a successful run
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from osprey.cli.templates.scaffolding import materialize_tier_dbs

# Absolute path to the preset's channel_databases dir — the source of truth
# for the byte-exact and content checks below.
_PRESET_CDB_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "channel_databases"
)

_PARADIGMS: tuple[str, ...] = ("in_context", "hierarchical", "middle_layer")


def _make_rendered_project(tmp_path: Path) -> Path:
    """Build a fake post-``osprey build`` tree by copying the preset CDB dir.

    Mirrors what :func:`copy_template_data` produces: the entire
    ``channel_databases/`` tree (including ``tiers/tier{1,2,3}/``) lives
    under ``<project>/data/channel_databases/``.
    """
    project_dir = tmp_path / "proj"
    dst = project_dir / "data" / "channel_databases"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_PRESET_CDB_DIR, dst)
    return project_dir


@pytest.fixture()
def rendered_project(tmp_path: Path) -> Path:
    """Tmp-path project with the preset channel_databases tree copied in."""
    return _make_rendered_project(tmp_path)


class TestMaterializeAllParadigms:
    """`channel_finder_mode='all'` / `None` materialize every paradigm."""

    @pytest.mark.parametrize("mode", ["all", None])
    def test_happy_path_tier1_all_paradigms(self, rendered_project: Path, mode):
        materialize_tier_dbs(rendered_project, tier=1, channel_finder_mode=mode)

        cdb = rendered_project / "data" / "channel_databases"
        for paradigm in _PARADIGMS:
            flat = cdb / f"{paradigm}.json"
            assert flat.is_file(), f"{paradigm}.json should be at flat root"
            # Content matches the preset tier1 source.
            expected = (_PRESET_CDB_DIR / "tiers" / "tier1" / f"{paradigm}.json").read_bytes()
            assert flat.read_bytes() == expected
            # Content is still valid JSON.
            json.loads(flat.read_text())

        # tiers/ subtree is pruned.
        assert not (cdb / "tiers").exists()


class TestSingleParadigmModes:
    """Single-paradigm modes materialize only that paradigm."""

    @pytest.mark.parametrize("paradigm", _PARADIGMS)
    def test_single_mode_only_materializes_that_paradigm(
        self, rendered_project: Path, paradigm: str
    ):
        materialize_tier_dbs(
            rendered_project, tier=1, channel_finder_mode=paradigm
        )

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

        # tiers/ subtree is pruned regardless of which paradigm was picked.
        assert not (cdb / "tiers").exists()


class TestTier3Selection:
    """Tier 3 must pull from `tiers/tier3/<paradigm>.json`, byte-exact."""

    @pytest.mark.parametrize("paradigm", _PARADIGMS)
    def test_tier3_flat_file_byte_equals_preset_tier3(
        self, rendered_project: Path, paradigm: str
    ):
        materialize_tier_dbs(
            rendered_project, tier=3, channel_finder_mode=paradigm
        )

        flat = rendered_project / "data" / "channel_databases" / f"{paradigm}.json"
        expected = (_PRESET_CDB_DIR / "tiers" / "tier3" / f"{paradigm}.json").read_bytes()
        assert flat.read_bytes() == expected

    def test_tier3_all_paradigms(self, rendered_project: Path):
        materialize_tier_dbs(rendered_project, tier=3, channel_finder_mode="all")

        cdb = rendered_project / "data" / "channel_databases"
        for paradigm in _PARADIGMS:
            flat = cdb / f"{paradigm}.json"
            expected = (
                _PRESET_CDB_DIR / "tiers" / "tier3" / f"{paradigm}.json"
            ).read_bytes()
            assert flat.read_bytes() == expected
        assert not (cdb / "tiers").exists()


class TestMissingSourceDb:
    """Missing source DB raises FileNotFoundError and leaves the tree intact."""

    def test_missing_source_raises_file_not_found(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        tier_dir = project_dir / "data" / "channel_databases" / "tiers" / "tier1"
        tier_dir.mkdir(parents=True)
        # Only put two of three paradigm files in place.
        (tier_dir / "in_context.json").write_text("{}")
        (tier_dir / "hierarchical.json").write_text("{}")
        # middle_layer.json is intentionally missing.

        with pytest.raises(FileNotFoundError, match="middle_layer"):
            materialize_tier_dbs(project_dir, tier=1, channel_finder_mode="all")

        # On failure the tiers/ tree must still be present (no partial rmtree).
        assert tier_dir.exists()
        # And no flat files should have been created (failure happens before
        # any copy because we resolve all pairs first).
        cdb = project_dir / "data" / "channel_databases"
        for paradigm in _PARADIGMS:
            assert not (cdb / f"{paradigm}.json").exists()

    def test_missing_source_for_single_mode_raises(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        # tier2 dir exists but is empty — single-paradigm request should fail.
        (project_dir / "data" / "channel_databases" / "tiers" / "tier2").mkdir(
            parents=True
        )

        with pytest.raises(FileNotFoundError, match="hierarchical"):
            materialize_tier_dbs(
                project_dir, tier=2, channel_finder_mode="hierarchical"
            )


class TestTiersDirectoryRemoval:
    """`tiers/` subtree must be absent after a successful materialization."""

    def test_post_rmtree_absence(self, rendered_project: Path):
        cdb = rendered_project / "data" / "channel_databases"
        # Sanity-check the fixture before we run.
        assert (cdb / "tiers").is_dir()
        assert (cdb / "tiers" / "tier1").is_dir()

        materialize_tier_dbs(rendered_project, tier=1, channel_finder_mode="all")

        assert not (cdb / "tiers").exists()


class TestUnknownMode:
    """An unrecognized channel_finder_mode is a hard error, not silent."""

    def test_unknown_mode_raises_value_error(self, rendered_project: Path):
        with pytest.raises(ValueError, match="Unknown channel_finder_mode"):
            materialize_tier_dbs(
                rendered_project, tier=1, channel_finder_mode="bogus"
            )


class TestNoOpOnMissingTiersSubtree:
    """Bundles without channel-finder DBs have no `tiers/` — must be a no-op."""

    def test_no_op_when_tiers_subtree_absent(self, tmp_path: Path):
        # Mimic a hello_world-style bundle: data/ exists but no channel DBs.
        project_dir = tmp_path / "proj"
        (project_dir / "data").mkdir(parents=True)
        # Must NOT raise, even with a fully-valid tier/mode combination.
        materialize_tier_dbs(project_dir, tier=1, channel_finder_mode="all")
        # Nothing got created — flat root doesn't exist either.
        assert not (project_dir / "data" / "channel_databases").exists()
