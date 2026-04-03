"""Tests for search_pvs MCP tool."""

import json
import os

import pytest
import yaml

from osprey.mcp_server.direct_channel_finder.server_context import initialize_dcf_context


@pytest.fixture
def mock_config(tmp_path):
    config = {"channel_finder": {"direct": {"backend": "mock"}}}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(config))
    old = os.environ.get("OSPREY_CONFIG")
    os.environ["OSPREY_CONFIG"] = str(config_path)
    initialize_dcf_context()
    yield
    if old is None:
        del os.environ["OSPREY_CONFIG"]
    else:
        os.environ["OSPREY_CONFIG"] = old


@pytest.fixture
def search_pvs_fn():
    from osprey.mcp_server.direct_channel_finder.tools.search_pvs import search_pvs

    from .conftest import get_tool_fn

    return get_tool_fn(search_pvs)


class TestSearchPvs:
    def test_basic_search(self, mock_config, search_pvs_fn):
        result = json.loads(search_pvs_fn("SR:*:BPM:*"))
        assert "records" in result
        assert result["total_count"] > 0
        assert result["pattern"] == "SR:*:BPM:*"

    def test_empty_search(self, mock_config, search_pvs_fn):
        result = json.loads(search_pvs_fn("NONEXISTENT:*"))
        assert result["total_count"] == 0
        assert result["records"] == []

    def test_pagination(self, mock_config, search_pvs_fn):
        result = json.loads(search_pvs_fn("*", page=1, page_size=5))
        assert len(result["records"]) == 5
        assert result["has_more"] is True
        assert result["page"] == 1

    def test_filter_record_type(self, mock_config, search_pvs_fn):
        result = json.loads(search_pvs_fn("*", record_type="ao"))
        assert result["total_count"] > 0
        for rec in result["records"]:
            assert rec["record_type"] == "ao"
        assert result["filters"]["record_type"] == "ao"

    def test_filter_description(self, mock_config, search_pvs_fn):
        result = json.loads(search_pvs_fn("*", description_contains="beam current"))
        assert result["total_count"] > 0

    def test_page_size_clamped(self, mock_config, search_pvs_fn):
        result = json.loads(search_pvs_fn("*", page_size=999))
        assert result["page_size"] == 200

    def test_error_without_registry(self, search_pvs_fn):
        result = json.loads(search_pvs_fn("*"))
        assert result["error"] is True
        assert result["error_type"] == "backend_not_available"
