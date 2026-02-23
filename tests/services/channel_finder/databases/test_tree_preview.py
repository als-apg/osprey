"""Tests for HierarchicalChannelDatabase tree preview methods."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.databases.hierarchical import (
    HierarchicalChannelDatabase,
)


@pytest.fixture()
def hier_db(tmp_path: Path) -> HierarchicalChannelDatabase:
    """Create a simple 3-level hierarchical database (all tree-type levels)."""
    db_path = tmp_path / "hier.json"
    data = {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "device", "type": "tree"},
                {"name": "signal", "type": "tree"},
            ],
            "naming_pattern": "{system}:{device}:{signal}",
        },
        "tree": {
            "SR": {
                "_description": "Storage Ring",
                "BPM": {
                    "_description": "Beam Position Monitor",
                    "X": {},
                    "Y": {},
                },
                "QUAD": {
                    "_description": "Quadrupole",
                    "I": {},
                },
            },
            "BR": {
                "_description": "Booster Ring",
                "MAG": {
                    "_description": "Magnet",
                    "I": {},
                    "V": {},
                },
            },
        },
    }
    db_path.write_text(json.dumps(data, indent=2))
    return HierarchicalChannelDatabase(str(db_path))


@pytest.fixture()
def hier_db_instances(tmp_path: Path) -> HierarchicalChannelDatabase:
    """Create a 3-level database with an instance-type middle level."""
    db_path = tmp_path / "hier_inst.json"
    data = {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "device", "type": "instances"},
                {"name": "signal", "type": "tree"},
            ],
            "naming_pattern": "{system}-{device}:{signal}",
        },
        "tree": {
            "MAG": {
                "_description": "Magnets",
                "DEVICE": {
                    "_expansion": {
                        "_type": "range",
                        "_pattern": "{:02d}",
                        "_range": [1, 3],
                    },
                    "I": {"_description": "Current"},
                    "V": {"_description": "Voltage"},
                },
            },
        },
    }
    db_path.write_text(json.dumps(data, indent=2))
    return HierarchicalChannelDatabase(str(db_path))


# ---------------------------------------------------------------------------
# Full tree preview
# ---------------------------------------------------------------------------


class TestGenerateTreePreview:
    def test_header_contains_channel_count(
        self, hier_db: HierarchicalChannelDatabase
    ):
        preview = hier_db.generate_tree_preview()
        first_line = preview.splitlines()[0]
        assert "Database Structure" in first_line
        assert "total channels" in first_line

    def test_hierarchy_line(self, hier_db: HierarchicalChannelDatabase):
        preview = hier_db.generate_tree_preview()
        assert "Hierarchy: system" in preview
        assert "device" in preview
        assert "signal" in preview

    def test_naming_pattern_line(self, hier_db: HierarchicalChannelDatabase):
        preview = hier_db.generate_tree_preview()
        assert "Naming: {system}:{device}:{signal}" in preview

    def test_node_names_with_channel_counts(
        self, hier_db: HierarchicalChannelDatabase
    ):
        preview = hier_db.generate_tree_preview()
        # Top-level systems should appear with channel counts
        assert "SR" in preview
        assert "BR" in preview
        # Channel count format: "NodeName (N ch)"
        assert " ch)" in preview

    def test_descriptions_included(self, hier_db: HierarchicalChannelDatabase):
        preview = hier_db.generate_tree_preview()
        assert "Storage Ring" in preview
        assert "Beam Position Monitor" in preview
        assert "Quadrupole" in preview
        assert "Booster Ring" in preview

    def test_max_depth_limits_rendering(
        self, hier_db: HierarchicalChannelDatabase
    ):
        # depth=1 should show only system-level nodes, not their children
        shallow = hier_db.generate_tree_preview(max_depth=1)
        assert "SR" in shallow
        assert "BR" in shallow
        # BPM and QUAD are at depth 1 within the tree render, so should not appear
        assert "BPM" not in shallow
        assert "QUAD" not in shallow

    def test_max_children_limits_shown(self, tmp_path: Path):
        """Build a database with many top-level nodes to test truncation."""
        db_path = tmp_path / "many.json"
        tree = {}
        for i in range(10):
            tree[f"SYS{i:02d}"] = {"SIG": {}}
        data = {
            "hierarchy": {
                "levels": [
                    {"name": "system", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                ],
                "naming_pattern": "{system}:{signal}",
            },
            "tree": tree,
        }
        db_path.write_text(json.dumps(data, indent=2))
        db = HierarchicalChannelDatabase(str(db_path))

        preview = db.generate_tree_preview(max_children=3)
        # Only 3 systems shown, 7 truncated
        assert "... and 7 more" in preview

    def test_caching_returns_same_object(
        self, hier_db: HierarchicalChannelDatabase
    ):
        first = hier_db.generate_tree_preview(max_depth=2, max_children=3)
        second = hier_db.generate_tree_preview(max_depth=2, max_children=3)
        assert first is second

    def test_different_params_not_cached_together(
        self, hier_db: HierarchicalChannelDatabase
    ):
        a = hier_db.generate_tree_preview(max_depth=1, max_children=5)
        b = hier_db.generate_tree_preview(max_depth=3, max_children=5)
        # Different depth should produce different content
        assert a != b


# ---------------------------------------------------------------------------
# Instance levels in tree preview
# ---------------------------------------------------------------------------


class TestInstanceLevelPreview:
    def test_instance_level_renders_count_and_range(
        self, hier_db_instances: HierarchicalChannelDatabase
    ):
        preview = hier_db_instances.generate_tree_preview()
        # Instance levels render as "[level] N instances (first..last)"
        assert "[device]" in preview
        assert "3 instances" in preview
        assert "01..03" in preview

    def test_instance_level_children_rendered(
        self, hier_db_instances: HierarchicalChannelDatabase
    ):
        preview = hier_db_instances.generate_tree_preview()
        # Children of the instance container (I, V) should appear
        assert "I" in preview
        assert "V" in preview


# ---------------------------------------------------------------------------
# Subtree preview
# ---------------------------------------------------------------------------


class TestGenerateSubtreePreview:
    def test_subtree_scopes_to_position(
        self, hier_db: HierarchicalChannelDatabase
    ):
        preview = hier_db.generate_subtree_preview(
            previous_selections={"system": "SR"}
        )
        assert "Subtree at:" in preview
        assert "system=SR" in preview
        # Should show children of SR (BPM, QUAD) but not BR
        assert "BPM" in preview
        assert "QUAD" in preview
        assert "BR" not in preview

    def test_subtree_shows_channel_count(
        self, hier_db: HierarchicalChannelDatabase
    ):
        preview = hier_db.generate_subtree_preview(
            previous_selections={"system": "SR"}
        )
        assert "channels below this point" in preview

    def test_subtree_invalid_position_returns_empty(
        self, hier_db: HierarchicalChannelDatabase
    ):
        result = hier_db.generate_subtree_preview(
            previous_selections={"system": "NONEXISTENT"}
        )
        assert result == ""

    def test_subtree_all_levels_selected_returns_empty(
        self, hier_db: HierarchicalChannelDatabase
    ):
        result = hier_db.generate_subtree_preview(
            previous_selections={
                "system": "SR",
                "device": "BPM",
                "signal": "X",
            }
        )
        assert result == ""

    def test_subtree_deeper_navigation(
        self, hier_db: HierarchicalChannelDatabase
    ):
        preview = hier_db.generate_subtree_preview(
            previous_selections={"system": "SR", "device": "BPM"}
        )
        assert "system=SR" in preview
        assert "device=BPM" in preview
        # Should show signals under BPM
        assert "X" in preview
        assert "Y" in preview
