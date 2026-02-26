"""
Unit tests for the preview_database tool.

Tests all parameters and their combinations using the native module
(no Jinja2 rendering or importlib hacks needed).
"""

import io
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from osprey.services.channel_finder.databases import HierarchicalChannelDatabase
from osprey.services.channel_finder.tools import preview_database as preview_mod
from osprey.services.channel_finder.tools.preview_database import preview_database

EXAMPLES_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "channel_databases"
    / "examples"
)


def _get_osprey_theme():
    try:
        from osprey.cli.styles import osprey_theme

        return osprey_theme
    except ImportError:
        return None


def _capture_preview(db_path, **kwargs):
    """Run preview_database with a capturing console and return the output string."""
    output = io.StringIO()
    test_console = Console(
        file=output,
        width=120,
        legacy_windows=False,
        force_terminal=False,
        no_color=True,
        theme=_get_osprey_theme(),
    )

    original_console = preview_mod.console
    preview_mod.console = test_console

    try:
        preview_database(db_path=db_path, **kwargs)
        return output.getvalue()
    finally:
        preview_mod.console = original_console


@pytest.fixture
def consecutive_db_path():
    """Path to consecutive_instances.json."""
    return str(EXAMPLES_DIR / "consecutive_instances.json")


@pytest.fixture
def instance_first_db_path():
    """Path to instance_first.json."""
    return str(EXAMPLES_DIR / "instance_first.json")


@pytest.fixture
def optional_levels_db_path():
    """Path to optional_levels.json."""
    return str(EXAMPLES_DIR / "optional_levels.json")


class TestDepthParameter:
    """Test --depth parameter."""

    def test_depth_3_limits_tree_levels(self, consecutive_db_path):
        """Test that depth=3 actually limits the tree to 3 levels."""
        result = _capture_preview(consecutive_db_path, depth=3, max_items=10, sections="tree")

        assert re.search(r"Display Depth\s+3", result), (
            "Output should show 'Display Depth' with value 3"
        )
        assert "Hierarchy Tree" in result, "Tree section header should be present"

        tree_section = result[result.find("Hierarchy Tree") : result.find("\U0001f4a1 Tip:")]
        branch_lines = [line for line in tree_section.split("\n") if "\u2501\u2501" in line]

        assert len(branch_lines) > 0, "Tree should have branches"

        max_indentation = max(line.count("\u2503") for line in branch_lines) if branch_lines else 0
        assert max_indentation <= 3, (
            f"Tree depth should not exceed 3, but found {max_indentation} levels"
        )

        assert "--depth -1" in result, "Should show tip about --depth -1"

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, -1])
    def test_depth_values(self, consecutive_db_path, depth):
        """Test different depth values."""
        with patch.object(preview_mod, "console"):
            try:
                preview_database(
                    depth=depth, max_items=5, sections="tree", db_path=consecutive_db_path
                )
            except Exception as e:
                pytest.fail(f"depth={depth} raised exception: {e}")


class TestMaxItemsParameter:
    """Test --max-items parameter."""

    def test_max_items_limits_branches(self, consecutive_db_path):
        """Test that max_items=3 actually limits branches to 3 per level."""
        result = _capture_preview(consecutive_db_path, depth=5, max_items=3, sections="tree")

        assert re.search(r"Max Items/Level\s+3", result), (
            "Output should show 'Max Items/Level' with value 3"
        )

        tree_section = result[result.find("Hierarchy Tree") : result.find("\U0001f4a1 Tip:")]
        assert "... " in tree_section and " more " in tree_section, (
            "Tree should show truncation message"
        )

        top_level_pattern = re.compile(r"\u2502\s\s[\u2523\u2517]\u2501\u2501\s+[A-Z]")
        top_level_branches = [
            line for line in tree_section.split("\n") if top_level_pattern.search(line)
        ]
        assert len(top_level_branches) == 3, (
            f"Should show exactly 3 top-level items, found {len(top_level_branches)}"
        )

    @pytest.mark.parametrize("max_items", [1, 3, 5, 10, 20, -1])
    def test_max_items_values(self, consecutive_db_path, max_items):
        """Test different max_items values."""
        with patch.object(preview_mod, "console"):
            try:
                preview_database(
                    depth=3, max_items=max_items, sections="tree", db_path=consecutive_db_path
                )
            except Exception as e:
                pytest.fail(f"max_items={max_items} raised exception: {e}")


class TestSectionsParameter:
    """Test --sections parameter."""

    def test_sections_tree_only(self, consecutive_db_path):
        """Test sections='tree' shows only tree, no stats/breakdown."""
        result = _capture_preview(consecutive_db_path, depth=3, max_items=5, sections="tree")
        assert "Hierarchy Tree" in result
        assert "Hierarchy Level Statistics" not in result
        assert "Channel Count Breakdown" not in result
        assert "Sample Channels" not in result

    def test_sections_stats_only(self, consecutive_db_path):
        """Test sections='stats' shows only stats, no tree."""
        result = _capture_preview(consecutive_db_path, depth=3, max_items=5, sections="stats")
        assert "Hierarchy Level Statistics" in result
        assert "Hierarchy Tree" not in result
        assert "Channel Count Breakdown" not in result

    def test_sections_all(self, consecutive_db_path):
        """Test sections='all' includes all sections."""
        result = _capture_preview(consecutive_db_path, depth=3, max_items=5, sections="all")
        assert "Hierarchy Tree" in result
        assert "Hierarchy Level Statistics" in result
        assert "Channel Count Breakdown" in result
        assert "Sample Channels" in result

    @pytest.mark.parametrize(
        "sections",
        [
            "tree",
            "stats",
            "breakdown",
            "samples",
            "tree,stats",
            "tree,breakdown",
            "tree,stats,breakdown",
            "tree,stats,breakdown,samples",
            "all",
        ],
    )
    def test_sections_combinations(self, consecutive_db_path, sections):
        """Test different section combinations."""
        with patch.object(preview_mod, "console"):
            try:
                preview_database(
                    depth=3, max_items=5, sections=sections, db_path=consecutive_db_path
                )
            except Exception as e:
                pytest.fail(f"sections={sections} raised exception: {e}")


class TestDepthMaxItemsCombinations:
    """Test combinations of depth and max_items."""

    @pytest.mark.parametrize(
        "depth,max_items",
        [
            (1, 1),
            (1, 5),
            (2, 1),
            (2, 5),
            (3, 1),
            (3, 5),
            (3, 10),
            (4, 10),
            (5, 5),
            (-1, -1),
            (-1, 5),
            (3, -1),
        ],
    )
    def test_depth_max_items_combinations(self, instance_first_db_path, depth, max_items):
        """Test all combinations of depth and max_items."""
        with patch.object(preview_mod, "console"):
            try:
                preview_database(
                    depth=depth,
                    max_items=max_items,
                    sections="tree",
                    db_path=instance_first_db_path,
                )
            except Exception as e:
                pytest.fail(f"depth={depth}, max_items={max_items} raised exception: {e}")


class TestFocusParameter:
    """Test --focus parameter."""

    def test_focus_filters_tree(self, consecutive_db_path):
        """Test that focus='M' only shows M subtree, not V or D."""
        result = _capture_preview(
            consecutive_db_path, depth=3, max_items=10, sections="tree", focus="M"
        )

        assert re.search(r"Focus Path\s+M", result), "Should show Focus Path = M"

        assert "QB" in result or "DP" in result or "CM" in result, (
            "Should show M subsystems (QB, DP, or CM)"
        )

        tree_section = result[result.find("Hierarchy Tree") :]
        assert "\u2501\u2501 V " not in tree_section, "Should NOT show V system"
        assert "\u2501\u2501 D " not in tree_section, "Should NOT show D system"

    def test_focus_valid_path(self, consecutive_db_path):
        """Test focusing on a valid path."""
        with patch.object(preview_mod, "console"):
            try:
                preview_database(
                    depth=3, max_items=5, sections="tree", focus="M", db_path=consecutive_db_path
                )
            except Exception as e:
                pytest.fail(f"focus='M' raised exception: {e}")

    def test_focus_invalid_path(self, consecutive_db_path):
        """Test focusing on an invalid path."""
        with patch.object(preview_mod, "console"):
            preview_database(
                depth=3,
                max_items=5,
                sections="tree",
                focus="NONEXISTENT",
                db_path=consecutive_db_path,
            )


class TestPathParameter:
    """Test --path parameter with different databases."""

    def test_path_consecutive_instances(self, consecutive_db_path):
        """Test with consecutive_instances.json."""
        with patch.object(preview_mod, "console"):
            preview_database(
                depth=3, max_items=5, sections="tree,stats", db_path=consecutive_db_path
            )

    def test_path_instance_first(self, instance_first_db_path):
        """Test with instance_first.json."""
        with patch.object(preview_mod, "console"):
            preview_database(
                depth=3, max_items=5, sections="tree,stats", db_path=instance_first_db_path
            )

    def test_path_optional_levels(self, optional_levels_db_path):
        """Test with optional_levels.json."""
        with patch.object(preview_mod, "console"):
            preview_database(depth=5, max_items=5, sections="all", db_path=optional_levels_db_path)


class TestBackwardsCompatibility:
    """Test --full flag backwards compatibility."""

    def test_full_flag(self, consecutive_db_path):
        """Test that --full sets depth and max_items to -1."""
        with patch.object(preview_mod, "preview_hierarchical") as mock_preview:
            preview_database(
                depth=3,
                max_items=10,
                sections="tree",
                show_full=True,
                db_path=consecutive_db_path,
            )

            assert mock_preview.called
            call_kwargs = mock_preview.call_args[1]
            assert call_kwargs["depth"] == -1
            assert call_kwargs["max_items"] == -1


class TestStatisticsCalculation:
    """Test statistics calculation functions."""

    def test_level_statistics(self, consecutive_db_path):
        """Test level statistics are calculated correctly."""
        db = HierarchicalChannelDatabase(consecutive_db_path)

        stats = preview_mod._calculate_level_statistics(db, db.hierarchy_levels)

        assert isinstance(stats, list)
        assert len(stats) == len(db.hierarchy_levels)

        for level_name, count in stats:
            assert level_name in db.hierarchy_levels
            assert isinstance(count, int)
            assert count > 0

    def test_breakdown_calculation(self, consecutive_db_path):
        """Test breakdown calculation."""
        db = HierarchicalChannelDatabase(consecutive_db_path)

        breakdown = preview_mod._calculate_breakdown(db, db.hierarchy_levels, focus=None)

        assert isinstance(breakdown, list)
        assert len(breakdown) > 0

        for path, count in breakdown:
            assert isinstance(path, str)
            assert isinstance(count, int)
            assert count > 0

        counts = [count for _, count in breakdown]
        assert counts == sorted(counts, reverse=True)


class TestCrossDatabaseCompatibility:
    """Test that features work across all database formats."""

    @pytest.mark.parametrize(
        "db_fixture_name",
        ["consecutive_db_path", "instance_first_db_path", "optional_levels_db_path"],
    )
    def test_depth_works_on_all_databases(self, db_fixture_name, request):
        """Test that depth parameter works on all database formats."""
        db_path = request.getfixturevalue(db_fixture_name)
        result = _capture_preview(db_path, depth=2, max_items=5, sections="tree")

        assert "Hierarchy Tree" in result, f"Tree should render on {db_fixture_name}"
        assert re.search(r"Display Depth\s+2", result), (
            f"Depth parameter should show on {db_fixture_name}"
        )
        assert "channels" in result.lower(), f"Should show channel count on {db_fixture_name}"


class TestEdgeCases:
    """Test edge cases."""

    def test_depth_zero(self, consecutive_db_path):
        """Test depth=0."""
        with patch.object(preview_mod, "console"):
            preview_database(depth=0, max_items=10, sections="tree", db_path=consecutive_db_path)

    def test_max_items_zero(self, consecutive_db_path):
        """Test max_items=0."""
        with patch.object(preview_mod, "console"):
            preview_database(depth=3, max_items=0, sections="tree", db_path=consecutive_db_path)

    def test_empty_sections(self, consecutive_db_path):
        """Test with empty sections string."""
        with patch.object(preview_mod, "console"):
            preview_database(depth=3, max_items=10, sections="", db_path=consecutive_db_path)


DB_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "channel_databases"
)


@pytest.fixture
def in_context_db_path():
    """Path to in_context.json."""
    path = str(DB_DIR / "in_context.json")
    if not Path(path).exists():
        pytest.skip("In-context database not found")
    return path


@pytest.fixture
def middle_layer_db_path():
    """Path to middle_layer.json."""
    path = str(DB_DIR / "middle_layer.json")
    if not Path(path).exists():
        pytest.skip("Middle layer database not found")
    return path


class TestInContextPreview:
    """Test in-context database preview path."""

    def test_in_context_renders_without_error(self, in_context_db_path):
        """preview_in_context renders without error."""
        with patch.object(preview_mod, "console"):
            preview_mod.preview_in_context(
                in_context_db_path, presentation_mode="template", show_full=False
            )

    def test_in_context_shows_channel_count(self, in_context_db_path):
        """In-context preview shows channel count and statistics."""
        result = _capture_preview(in_context_db_path)
        assert "channels" in result.lower()
        assert "Database Statistics" in result or "Preview" in result

    def test_in_context_show_full(self, in_context_db_path):
        """show_full=False limits display, show_full=True shows all."""
        limited = _capture_preview(in_context_db_path, show_full=False)
        full = _capture_preview(in_context_db_path, show_full=True)
        # Full output should be at least as long as limited
        assert len(full) >= len(limited)


class TestMiddleLayerPreview:
    """Test middle layer database preview path."""

    def test_middle_layer_renders_without_error(self, middle_layer_db_path):
        """preview_middle_layer renders without error."""
        with patch.object(preview_mod, "console"):
            preview_mod.preview_middle_layer(
                middle_layer_db_path, depth=2, max_items=2, sections=["tree"]
            )

    def test_middle_layer_shows_stats(self, middle_layer_db_path):
        """Middle layer preview shows system/family/field stats."""
        result = _capture_preview(middle_layer_db_path, sections="tree")
        assert "channels" in result.lower()

    def test_middle_layer_focus_filter(self, middle_layer_db_path):
        """Focus parameter filters the middle layer tree."""
        with patch.object(preview_mod, "console"):
            preview_mod.preview_middle_layer(
                middle_layer_db_path, depth=3, max_items=3, sections=["tree"], focus="SR"
            )


class TestAutoDetectDatabaseType:
    """Test auto-detection of database type from JSON structure."""

    def test_hierarchical_auto_detected(self, consecutive_db_path):
        """Hierarchical database auto-detected from JSON structure."""
        result = _capture_preview(consecutive_db_path, depth=2, max_items=2, sections="tree")
        assert "Hierarchical" in result

    def test_middle_layer_auto_detected(self, middle_layer_db_path):
        """Middle layer database auto-detected from JSON keys (SR, BR, etc.)."""
        result = _capture_preview(middle_layer_db_path, depth=2, max_items=2, sections="tree")
        assert "Middle Layer" in result

    def test_in_context_auto_detected(self, in_context_db_path):
        """In-context database auto-detected as fallback."""
        result = _capture_preview(in_context_db_path)
        assert "In-Context" in result


class TestPreviewDetectPipelineConfig:
    """Test detect_pipeline_config from preview_database module."""

    def test_detects_middle_layer_when_mode_set(self):
        """Detects middle_layer when pipeline_mode: middle_layer."""
        config = {
            "channel_finder": {
                "pipeline_mode": "middle_layer",
                "pipelines": {
                    "middle_layer": {"database": {"path": "/ml/path.json"}},
                    "hierarchical": {"database": {"path": "/h/path.json"}},
                },
            }
        }
        ptype, db_config = preview_mod.detect_pipeline_config(config)
        assert ptype == "middle_layer"

    def test_detects_hierarchical_when_path_present(self):
        """Detects hierarchical when path present (no mode set)."""
        config = {
            "channel_finder": {
                "pipelines": {
                    "hierarchical": {"database": {"path": "/h/path.json"}},
                }
            }
        }
        ptype, db_config = preview_mod.detect_pipeline_config(config)
        assert ptype == "hierarchical"

    def test_fallback_priority_order(self):
        """Falls back through priorities: middle_layer -> hierarchical -> in_context."""
        config = {
            "channel_finder": {
                "pipelines": {
                    "middle_layer": {"database": {"path": "/ml/path.json"}},
                    "hierarchical": {"database": {"path": "/h/path.json"}},
                    "in_context": {"database": {"path": "/ic/path.json"}},
                }
            }
        }
        ptype, db_config = preview_mod.detect_pipeline_config(config)
        assert ptype == "middle_layer"

    def test_returns_none_with_empty_config(self):
        """Returns (None, None) with empty config."""
        config = {"channel_finder": {"pipelines": {}}}
        ptype, db_config = preview_mod.detect_pipeline_config(config)
        assert ptype is None
        assert db_config is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
