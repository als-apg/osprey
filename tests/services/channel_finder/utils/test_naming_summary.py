"""Tests for naming summary generation."""

import pytest

from osprey.services.channel_finder.utils.naming_summary import generate_naming_summary


class TestGenerateNamingSummary:
    def test_fallback_when_no_pipeline(self):
        config = {}
        summary = generate_naming_summary(config)
        assert "PV Naming Conventions" in summary
        assert "search_pvs" in summary

    def test_fallback_with_empty_channel_finder(self):
        config = {"channel_finder": {}}
        summary = generate_naming_summary(config)
        assert "search_pvs" in summary

    def test_returns_string(self):
        config = {"channel_finder": {"pipeline_mode": "hierarchical"}}
        summary = generate_naming_summary(config)
        assert isinstance(summary, str)
        assert len(summary) > 0
