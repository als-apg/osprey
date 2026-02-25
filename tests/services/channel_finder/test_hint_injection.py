"""Tests for tree preview and hint injection into hierarchical prompts."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_database():
    """Create a mock database with hierarchy config."""
    db = MagicMock()
    db.hierarchy_levels = ["system", "family", "signal"]
    db.hierarchy_config = {
        "levels": {
            "system": {"type": "tree"},
            "family": {"type": "tree"},
            "signal": {"type": "tree"},
        }
    }
    db.get_hierarchy_definition.return_value = ["system", "family", "signal"]
    db.generate_tree_preview.return_value = (
        "Database Structure (100 total channels)\n"
        "Hierarchy: system → family → signal\n"
        "MAG (50 ch)\nRF (50 ch)"
    )
    db.generate_subtree_preview.return_value = (
        "Subtree at: system=MAG\n(50 channels below this point)\nBPM (25 ch)\nCOR (25 ch)"
    )
    return db


@pytest.fixture()
def mock_feedback_store():
    """Create a mock feedback store with hints."""
    store = MagicMock()
    store.get_hints.return_value = [
        {"selections": {"system": "MAG", "family": "BPM", "signal": "X"}, "channel_count": 10},
    ]
    return store


def _make_pipeline(database, feedback_store=None, tree_preview_config=None):
    """Create a HierarchicalPipeline with mocked dependencies."""
    from osprey.services.channel_finder.pipelines.hierarchical.pipeline import (
        HierarchicalPipeline,
    )

    # Patch out the config/prompts loading in __init__ and the BasePipeline
    # config lookup (get_config_value called in super().__init__).
    with (
        patch(
            "osprey.services.channel_finder.pipelines.hierarchical.pipeline.get_config_builder"
        ) as mock_cb,
        patch(
            "osprey.services.channel_finder.pipelines.hierarchical.pipeline.load_prompts"
        ) as mock_lp,
        patch(
            "osprey.utils.config.get_config_value",
            return_value="lenient",
        ),
    ):
        mock_cb.return_value.raw_config = {}
        mock_lp.return_value = MagicMock(spec=[])  # No attributes → no hierarchical_context

        pipeline = HierarchicalPipeline(
            database=database,
            model_config={"provider": "test", "model_id": "test"},
            facility_name="test facility",
            query_splitting=False,
            feedback_store=feedback_store,
            tree_preview_config=tree_preview_config,
        )
    return pipeline


def test_tree_preview_at_root_level(mock_database):
    """Full tree preview is injected at the first hierarchy level."""
    pipeline = _make_pipeline(mock_database, tree_preview_config={"enabled": True})

    context = pipeline._build_look_ahead_context(
        level="system", previous_selections={}, query="find magnets"
    )

    assert "DATABASE OVERVIEW" in context
    assert "Database Structure (100 total channels)" in context
    mock_database.generate_tree_preview.assert_called_once()


def test_subtree_preview_at_deeper_level(mock_database):
    """Subtree preview is injected at deeper levels."""
    pipeline = _make_pipeline(mock_database, tree_preview_config={"enabled": True})

    context = pipeline._build_look_ahead_context(
        level="family",
        previous_selections={"system": "MAG"},
        query="find BPMs",
    )

    assert "SUBTREE PREVIEW" in context
    assert "Subtree at: system=MAG" in context
    mock_database.generate_subtree_preview.assert_called_once()


def test_feedback_hints_injected(mock_database, mock_feedback_store):
    """Feedback hints are injected when store has matches."""
    pipeline = _make_pipeline(
        mock_database,
        feedback_store=mock_feedback_store,
        tree_preview_config={"enabled": False},
    )

    context = pipeline._build_look_ahead_context(
        level="system", previous_selections={}, query="find magnets"
    )

    assert "PRIOR SUCCESSFUL PATHS" in context
    assert "10 channels" in context
    mock_feedback_store.get_hints.assert_called_once()


def test_no_context_when_disabled(mock_database):
    """No context when tree preview is disabled and no feedback store."""
    pipeline = _make_pipeline(mock_database, tree_preview_config={"enabled": False})

    context = pipeline._build_look_ahead_context(
        level="system", previous_selections={}, query="test"
    )

    assert context == ""


def test_no_hints_when_store_empty(mock_database):
    """No hints section when feedback store has no matches."""
    store = MagicMock()
    store.get_hints.return_value = []
    pipeline = _make_pipeline(
        mock_database,
        feedback_store=store,
        tree_preview_config={"enabled": False},
    )

    context = pipeline._build_look_ahead_context(
        level="system", previous_selections={}, query="test"
    )

    assert context == ""


def test_look_ahead_in_full_prompt(mock_database, mock_feedback_store):
    """Tree preview and hints appear in the full _build_level_prompt output."""
    pipeline = _make_pipeline(
        mock_database,
        feedback_store=mock_feedback_store,
        tree_preview_config={"enabled": True},
    )

    options = [{"name": "MAG", "description": "Magnets"}, {"name": "RF", "description": "RF"}]
    prompt = pipeline._build_level_prompt("find magnets", "system", options, {})

    assert "DATABASE OVERVIEW" in prompt
    assert "PRIOR SUCCESSFUL PATHS" in prompt
    assert "User Query:" in prompt


def test_tree_preview_config_defaults(mock_database):
    """Default tree preview config works when None passed."""
    pipeline = _make_pipeline(mock_database, tree_preview_config=None)

    # Defaults: enabled=True, max_depth=3, max_children=5
    assert pipeline._tree_preview_enabled is True
    assert pipeline._tree_preview_max_depth == 3
    assert pipeline._tree_preview_max_children == 5
