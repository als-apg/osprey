"""Tests for get_naming_patterns MCP tool."""

import json
import os

import pytest
import yaml

from osprey.mcp_server.direct_channel_finder.registry import initialize_dcf_registry


@pytest.fixture
def mock_config(tmp_path):
    config = {"channel_finder": {"direct": {"backend": "mock"}}}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(config))
    old = os.environ.get("OSPREY_CONFIG")
    os.environ["OSPREY_CONFIG"] = str(config_path)
    initialize_dcf_registry()
    yield
    if old is None:
        del os.environ["OSPREY_CONFIG"]
    else:
        os.environ["OSPREY_CONFIG"] = old


@pytest.fixture
def get_naming_patterns_fn():
    from osprey.mcp_server.direct_channel_finder.tools.get_naming_patterns import (
        get_naming_patterns,
    )

    from .conftest import get_tool_fn

    return get_tool_fn(get_naming_patterns)


class TestGetNamingPatterns:
    def test_returns_summary(self, mock_config, get_naming_patterns_fn):
        result = json.loads(get_naming_patterns_fn())
        assert "naming_summary" in result
        assert "facility_name" in result
        assert isinstance(result["naming_summary"], str)
        assert len(result["naming_summary"]) > 0

    def test_fallback_without_curated_db(self, mock_config, get_naming_patterns_fn):
        # Config has no pipeline databases, should get fallback
        result = json.loads(get_naming_patterns_fn())
        assert "naming_summary" in result
        # Fallback summary should mention search_pvs
        assert "search_pvs" in result["naming_summary"] or "PV" in result["naming_summary"]

    def test_error_without_registry(self, get_naming_patterns_fn):
        result = json.loads(get_naming_patterns_fn())
        assert result["error"] is True
        assert result["error_type"] == "naming_patterns_error"
