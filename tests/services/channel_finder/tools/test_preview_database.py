"""Tests for the database preview presentation tool.

Focuses on the behavioral core: path resolution, the pure tree/statistics
helpers, and the three pipeline preview renderers plus the dispatch function.
Rich output is captured through an in-memory themed console and asserted on
plain-text substrings. Config loading is patched; no real config.yml needed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from osprey.cli.styles import osprey_theme
from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase
from osprey.services.channel_finder.tools import preview_database as mod
from osprey.services.channel_finder.tools.preview_database import (
    _build_middle_layer_tree,
    _calculate_breakdown,
    _calculate_level_statistics,
    _count_channels_at_path,
    _count_channels_matching_focus,
    _get_children_at_level,
    _navigate_middle_layer_focus,
    _navigate_to_focus,
    _resolve_path,
    preview_database,
    preview_hierarchical,
    preview_in_context,
    preview_middle_layer,
)


def _capture_console() -> Console:
    return Console(file=io.StringIO(), width=200, force_terminal=False, theme=osprey_theme)


def _text(console: Console) -> str:
    return console.file.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hier_file(tmp_path: Path) -> Path:
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
                "BPM": {"X": {}, "Y": {}},
                "QUAD": {"I": {}},
            },
            "BR": {
                "_description": "Booster Ring",
                "MAG": {"I": {}, "V": {}},
            },
        },
    }
    p = tmp_path / "hier.json"
    p.write_text(json.dumps(data, indent=2))
    return p


@pytest.fixture()
def hier_db(hier_file: Path) -> HierarchicalChannelDatabase:
    return HierarchicalChannelDatabase(str(hier_file))


@pytest.fixture()
def ml_file(tmp_path: Path) -> Path:
    data = {
        "SR": {
            "_description": "Storage Ring",
            "BPM": {
                "_description": "BPM family",
                "Monitor": {
                    "ChannelNames": ["SR01:BPM:X", "SR01:BPM:Y"],
                    "DataType": "double",
                },
                "Setpoint": {"X": {"ChannelNames": ["SR01:BPM:XSet"]}},
                "setup": {"DeviceList": [[1, 1]]},
            },
        },
    }
    p = tmp_path / "ml.json"
    p.write_text(json.dumps(data, indent=2))
    return p


@pytest.fixture()
def ml_db(ml_file: Path) -> MiddleLayerDatabase:
    return MiddleLayerDatabase(str(ml_file))


@pytest.fixture()
def in_context_file(tmp_path: Path) -> Path:
    data = {
        "facility_name": "Test Facility",
        "channels": [
            {
                "template": True,
                "base_name": "Dipole",
                "instances": [1, 3],
                "sub_channels": ["SP", "RB"],
                "description": "Dipole {instance:02d}",
            },
            {
                "channel": "STANDALONE:CH",
                "address": "STANDALONE:CH",
                "description": "A standalone channel",
            },
        ],
    }
    p = tmp_path / "in_context.json"
    p.write_text(json.dumps(data, indent=2))
    return p


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_absolute_path_returned_unchanged(self, tmp_path: Path):
        abs_path = tmp_path / "db.json"
        assert _resolve_path(str(abs_path)) == abs_path

    def test_relative_path_uses_workspace_resolver(self, monkeypatch, tmp_path):
        import osprey.utils.workspace as ws

        target = tmp_path / "resolved.json"
        monkeypatch.setattr(ws, "resolve_path", lambda s: target)
        assert _resolve_path("relative/db.json") == target

    def test_relative_path_falls_back_to_cwd_on_error(self, monkeypatch):
        import osprey.utils.workspace as ws

        def boom(_s):
            raise RuntimeError("no workspace")

        monkeypatch.setattr(ws, "resolve_path", boom)
        result = _resolve_path("relative/db.json")
        assert result == Path.cwd() / "relative/db.json"


# ---------------------------------------------------------------------------
# Pure hierarchy helpers
# ---------------------------------------------------------------------------


class TestNavigateToFocus:
    def test_navigates_one_level(self, hier_db: HierarchicalChannelDatabase):
        subtree, level = _navigate_to_focus(hier_db.tree, ["SR"], hier_db.hierarchy_levels, hier_db)
        assert level == 1
        assert "BPM" in subtree

    def test_navigates_two_levels(self, hier_db: HierarchicalChannelDatabase):
        subtree, level = _navigate_to_focus(
            hier_db.tree, ["SR", "BPM"], hier_db.hierarchy_levels, hier_db
        )
        assert level == 2
        assert "X" in subtree and "Y" in subtree

    def test_missing_focus_returns_none(self, hier_db: HierarchicalChannelDatabase):
        subtree, level = _navigate_to_focus(
            hier_db.tree, ["NOPE"], hier_db.hierarchy_levels, hier_db
        )
        assert subtree is None and level is None


class TestCountChannelsMatchingFocus:
    def test_counts_matching_system(self, hier_db: HierarchicalChannelDatabase):
        matches = _count_channels_matching_focus(hier_db, ["SR"], hier_db.hierarchy_levels)
        assert len(matches) == 3  # SR:BPM:X, SR:BPM:Y, SR:QUAD:I

    def test_counts_matching_device(self, hier_db: HierarchicalChannelDatabase):
        matches = _count_channels_matching_focus(hier_db, ["SR", "BPM"], hier_db.hierarchy_levels)
        assert len(matches) == 2


class TestLevelStatistics:
    def test_unique_counts_per_level(self, hier_db: HierarchicalChannelDatabase):
        stats = _calculate_level_statistics(hier_db, hier_db.hierarchy_levels)
        as_dict = dict(stats)
        assert as_dict["system"] == 2  # SR, BR
        assert as_dict["device"] == 3  # BPM, QUAD, MAG
        assert as_dict["signal"] == 4  # X, Y, I, V


class TestCalculateBreakdown:
    def test_breakdown_sorted_by_count_desc(self, hier_db: HierarchicalChannelDatabase):
        breakdown = _calculate_breakdown(hier_db, hier_db.hierarchy_levels, focus=None)
        as_dict = dict(breakdown)
        # Top-level SR has 3 channels, BR has 2.
        assert as_dict["SR"] == 3
        assert as_dict["BR"] == 2
        assert as_dict["SR:BPM"] == 2
        # Sorted descending by count.
        counts = [c for _, c in breakdown]
        assert counts == sorted(counts, reverse=True)


class TestCountChannelsAtPath:
    def test_counts_full_path(self, hier_db: HierarchicalChannelDatabase):
        count = _count_channels_at_path(
            hier_db, hier_db.hierarchy_levels, ["SR", "BPM"], current_level_idx=1
        )
        assert count == 2

    def test_counts_single_level(self, hier_db: HierarchicalChannelDatabase):
        count = _count_channels_at_path(
            hier_db, hier_db.hierarchy_levels, ["BR"], current_level_idx=0
        )
        assert count == 2


class TestGetChildrenAtLevel:
    def test_plain_dict_children_skip_underscore_keys(self):
        data = {"_description": "x", "A": {}, "B": {}, "leaf": "notdict"}
        children = _get_children_at_level(data, "device", ["system", "device"], 1)
        assert set(children.keys()) == {"A", "B"}

    def test_range_expansion(self):
        data = {"_expansion": {"_type": "range", "_pattern": "{:02d}", "_range": [1, 3]}}
        children = _get_children_at_level(data, "device", ["system", "device"], 1)
        assert set(children.keys()) == {"01", "02", "03"}

    def test_list_expansion(self):
        data = {"_expansion": {"_type": "list", "_instances": ["A1", "A2"]}}
        children = _get_children_at_level(data, "device", ["system", "device"], 1)
        assert set(children.keys()) == {"A1", "A2"}

    def test_non_dict_returns_empty(self):
        assert _get_children_at_level("scalar", "device", ["device"], 0) == {}


# ---------------------------------------------------------------------------
# Middle-layer pure helpers
# ---------------------------------------------------------------------------


class TestBuildMiddleLayerTree:
    def test_tree_counts_and_descriptions(self, ml_db: MiddleLayerDatabase):
        tree = _build_middle_layer_tree(ml_db)
        assert "SR" in tree
        assert tree["SR"]["_channels"] == 3
        assert tree["SR"]["_description"] == "Storage Ring"
        assert "BPM" in tree["SR"]["_families"]
        assert tree["SR"]["_families"]["BPM"]["_channels"] == 3


class TestNavigateMiddleLayerFocus:
    def test_navigate_system(self, ml_db: MiddleLayerDatabase):
        tree = _build_middle_layer_tree(ml_db)
        node, title = _navigate_middle_layer_focus(tree, ["SR"])
        assert node is not None
        assert "_families" in node
        assert "SR" in title

    def test_navigate_family(self, ml_db: MiddleLayerDatabase):
        tree = _build_middle_layer_tree(ml_db)
        node, title = _navigate_middle_layer_focus(tree, ["SR", "BPM"])
        assert node is not None
        assert "_fields" in node

    def test_missing_system_returns_none(self, ml_db: MiddleLayerDatabase):
        tree = _build_middle_layer_tree(ml_db)
        node, title = _navigate_middle_layer_focus(tree, ["NOPE"])
        assert node is None and title is None

    def test_too_deep_returns_none(self, ml_db: MiddleLayerDatabase):
        tree = _build_middle_layer_tree(ml_db)
        node, title = _navigate_middle_layer_focus(tree, ["SR", "BPM", "Monitor"])
        assert node is None and title is None


# ---------------------------------------------------------------------------
# preview_hierarchical
# ---------------------------------------------------------------------------


class TestPreviewHierarchical:
    def test_default_tree_section(self, hier_file: Path):
        console = _capture_console()
        preview_hierarchical(str(hier_file), console=console)
        out = _text(console)
        assert "Hierarchical Database Preview" in out
        assert "Successfully loaded" in out
        assert "SR" in out and "BR" in out
        assert "Preview complete" in out

    def test_all_sections(self, hier_file: Path):
        console = _capture_console()
        preview_hierarchical(
            str(hier_file),
            sections=["tree", "stats", "breakdown", "samples"],
            console=console,
        )
        out = _text(console)
        assert "Hierarchy Level Statistics" in out
        assert "Channel Count Breakdown" in out
        assert "Sample Channels" in out

    def test_focus_scopes_output(self, hier_file: Path):
        console = _capture_console()
        preview_hierarchical(str(hier_file), focus="SR", console=console)
        out = _text(console)
        assert "BPM" in out
        assert "QUAD" in out

    def test_focus_not_found(self, hier_file: Path):
        console = _capture_console()
        preview_hierarchical(str(hier_file), focus="NONEXISTENT", console=console)
        out = _text(console)
        assert "not found" in out

    def test_unlimited_depth_and_items(self, hier_file: Path):
        console = _capture_console()
        preview_hierarchical(str(hier_file), depth=-1, max_items=-1, console=console)
        out = _text(console)
        # Full expansion reaches leaf signals.
        assert "X" in out and "Y" in out


# ---------------------------------------------------------------------------
# preview_middle_layer
# ---------------------------------------------------------------------------


class TestPreviewMiddleLayer:
    def test_default_render(self, ml_file: Path):
        console = _capture_console()
        preview_middle_layer(str(ml_file), console=console)
        out = _text(console)
        assert "Middle Layer Database Preview" in out
        assert "Successfully loaded" in out
        assert "Preview complete" in out

    def test_stats_and_samples_sections(self, ml_file: Path):
        console = _capture_console()
        preview_middle_layer(str(ml_file), sections=["stats", "tree", "samples"], console=console)
        out = _text(console)
        assert "Database Statistics" in out
        assert "Total Channels" in out
        assert "Sample Channels" in out

    def test_focus_not_found(self, ml_file: Path):
        console = _capture_console()
        preview_middle_layer(str(ml_file), focus="NONEXISTENT", console=console)
        out = _text(console)
        assert "not found" in out

    def test_show_full_expands(self, ml_file: Path):
        console = _capture_console()
        preview_middle_layer(str(ml_file), show_full=True, console=console)
        out = _text(console)
        assert "Preview complete" in out


# ---------------------------------------------------------------------------
# preview_in_context
# ---------------------------------------------------------------------------


class TestPreviewInContext:
    def test_template_mode_render(self, in_context_file: Path):
        console = _capture_console()
        preview_in_context(str(in_context_file), presentation_mode="template", console=console)
        out = _text(console)
        assert "In-Context Database Preview" in out
        assert "Successfully loaded" in out
        assert "LLM Presentation" in out
        assert "Preview complete" in out

    def test_explicit_mode_lists_channel_names(self, in_context_file: Path):
        console = _capture_console()
        preview_in_context(str(in_context_file), presentation_mode="explicit", console=console)
        out = _text(console)
        assert "STANDALONE:CH" in out

    def test_show_full_flag(self, in_context_file: Path):
        console = _capture_console()
        preview_in_context(
            str(in_context_file),
            presentation_mode="explicit",
            show_full=True,
            console=console,
        )
        out = _text(console)
        assert "all" in out  # "(all N channels)" title


# ---------------------------------------------------------------------------
# preview_database dispatch
# ---------------------------------------------------------------------------


class TestPreviewDatabaseDispatch:
    def test_dispatch_hierarchical_by_path(self, hier_file: Path):
        console = _capture_console()
        preview_database(db_path=str(hier_file), console=console)
        assert "Hierarchical Database Preview" in _text(console)

    def test_dispatch_middle_layer_by_path(self, ml_file: Path):
        console = _capture_console()
        preview_database(db_path=str(ml_file), console=console)
        assert "Middle Layer Database Preview" in _text(console)

    def test_dispatch_in_context_by_path(self, in_context_file: Path):
        console = _capture_console()
        preview_database(db_path=str(in_context_file), console=console)
        assert "In-Context Database Preview" in _text(console)

    def test_load_error_reports_and_returns(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json")
        console = _capture_console()
        preview_database(db_path=str(bad), console=console)
        assert "Error loading database" in _text(console)

    def test_all_sections_keyword_expands(self, hier_file: Path):
        console = _capture_console()
        preview_database(db_path=str(hier_file), sections="all", console=console)
        out = _text(console)
        assert "Hierarchy Level Statistics" in out
        assert "Channel Count Breakdown" in out
        assert "Sample Channels" in out

    def test_no_config_reports_error(self, monkeypatch):
        import osprey.utils.config as config_mod

        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: {})
        monkeypatch.setattr(mod, "detect_pipeline_config", lambda config: (None, None))
        console = _capture_console()
        preview_database(console=console)
        assert "No database configured" in _text(console)

    def test_config_path_dispatches(self, monkeypatch, in_context_file: Path):
        import osprey.utils.config as config_mod

        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: {})
        monkeypatch.setattr(
            mod,
            "detect_pipeline_config",
            lambda config: (
                "in_context",
                {"path": str(in_context_file), "presentation_mode": "template"},
            ),
        )
        console = _capture_console()
        preview_database(console=console)
        assert "In-Context Database Preview" in _text(console)
