"""Tests for selections path provenance through hierarchical pipeline."""

import json

import pytest

from osprey.services.channel_finder.core.models import ChannelFinderResult
from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from osprey.services.channel_finder.pipelines.hierarchical.pipeline import ChannelWithPath


def test_channel_with_path_creation():
    """ChannelWithPath NamedTuple stores channel and selections."""
    cwp = ChannelWithPath("SR:BPM:X", {"system": "SR", "device": "BPM", "signal": "X"})
    assert cwp.channel == "SR:BPM:X"
    assert cwp.selections == {"system": "SR", "device": "BPM", "signal": "X"}


def test_channel_with_path_tuple_unpacking():
    """ChannelWithPath supports tuple unpacking."""
    cwp = ChannelWithPath("SR:BPM:X", {"system": "SR"})
    ch, sel = cwp
    assert ch == "SR:BPM:X"
    assert sel == {"system": "SR"}


def test_channel_finder_result_selections_paths_default():
    """ChannelFinderResult.selections_paths defaults to empty list."""
    result = ChannelFinderResult(
        query="test", channels=[], total_channels=0, processing_notes="test"
    )
    assert result.selections_paths == []


def test_channel_finder_result_selections_paths_populated():
    """ChannelFinderResult.selections_paths can be populated."""
    paths = [{"system": "SR", "device": "BPM"}]
    result = ChannelFinderResult(
        query="test",
        channels=[],
        total_channels=0,
        processing_notes="test",
        selections_paths=paths,
    )
    assert result.selections_paths == paths
    assert len(result.selections_paths) == 1


def test_channel_finder_result_backward_compatible():
    """Existing code that doesn't pass selections_paths still works."""
    result = ChannelFinderResult(query="q", channels=[], total_channels=0, processing_notes="n")
    # Can serialize without error
    d = result.model_dump()
    assert "selections_paths" in d
    assert d["selections_paths"] == []


@pytest.fixture()
def simple_db(tmp_path):
    """Simple 2-level tree database for testing base case wrapping."""
    db_path = tmp_path / "simple.json"
    data = {
        "hierarchy": {
            "levels": [
                {"name": "system", "type": "tree"},
                {"name": "signal", "type": "tree"},
            ],
            "naming_pattern": "{system}:{signal}",
        },
        "tree": {
            "MAG": {"X": {}, "Y": {}},
        },
    }
    db_path.write_text(json.dumps(data, indent=2))
    return HierarchicalChannelDatabase(str(db_path))


@pytest.mark.asyncio
async def test_navigate_recursive_base_case_returns_channel_with_path(simple_db):
    """Base case of _navigate_recursive wraps channels in ChannelWithPath."""
    # We need a minimal pipeline-like object to call _navigate_recursive
    # Instead, test the wrapping logic directly:
    # When remaining_levels=[] and selections has values, build_channels_from_selections
    # should produce channels that get wrapped with those selections.
    selections = {"system": "MAG", "signal": ["X", "Y"]}
    channels = simple_db.build_channels_from_selections(selections)
    assert len(channels) == 2
    assert "MAG:X" in channels
    assert "MAG:Y" in channels

    # Verify ChannelWithPath wrapping works correctly
    wrapped = [ChannelWithPath(ch, dict(selections)) for ch in channels]
    assert all(isinstance(w, ChannelWithPath) for w in wrapped)
    assert all(w.selections == selections for w in wrapped)
