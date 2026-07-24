"""Tests for channel-finder core Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from osprey.services.channel_finder.core.models import ChannelFinderResult, ChannelInfo


class TestChannelInfo:
    def test_description_defaults_to_none(self):
        ch = ChannelInfo(channel="SR:BPM:1", address="SR:BPM:1")
        assert ch.description is None

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ChannelInfo(channel="SR:BPM:1")  # address missing


class TestChannelFinderResult:
    def test_selections_paths_default_is_independent_per_instance(self):
        # default_factory must give each result its own list, not a shared one.
        a = ChannelFinderResult(query="q", channels=[], total_channels=0, processing_notes="")
        b = ChannelFinderResult(query="q", channels=[], total_channels=0, processing_notes="")
        a.selections_paths.append({"system": "SR"})
        assert b.selections_paths == []

    def test_nested_channel_coercion(self):
        result = ChannelFinderResult(
            query="dipoles",
            channels=[{"channel": "SR:MAG:1", "address": "SR:MAG:1"}],
            total_channels=1,
            processing_notes="ok",
        )
        assert isinstance(result.channels[0], ChannelInfo)
        assert result.channels[0].channel == "SR:MAG:1"

    def test_missing_required_fields_raise(self):
        with pytest.raises(ValidationError):
            ChannelFinderResult(query="q", channels=[])  # totals/notes missing
