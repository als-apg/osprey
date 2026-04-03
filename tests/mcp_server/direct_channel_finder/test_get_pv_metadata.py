"""Tests for get_pv_metadata MCP tool."""

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
def get_pv_metadata_fn():
    from osprey.mcp_server.direct_channel_finder.tools.get_pv_metadata import (
        get_pv_metadata,
    )

    from .conftest import get_tool_fn

    return get_tool_fn(get_pv_metadata)


@pytest.fixture
def search_pvs_fn():
    from osprey.mcp_server.direct_channel_finder.tools.search_pvs import search_pvs

    from .conftest import get_tool_fn

    return get_tool_fn(search_pvs)


class TestGetPvMetadata:
    def test_basic_metadata(self, mock_config, get_pv_metadata_fn, search_pvs_fn):
        # Find a PV name first
        search = json.loads(search_pvs_fn("SR:DIAG:BPM:*", page_size=1))
        pv_name = search["records"][0]["name"]

        result = json.loads(get_pv_metadata_fn([pv_name]))
        assert result["found_count"] == 1
        assert result["not_found_count"] == 0
        assert result["records"][0]["name"] == pv_name

    def test_missing_pv(self, mock_config, get_pv_metadata_fn):
        result = json.loads(get_pv_metadata_fn(["DOES:NOT:EXIST"]))
        assert result["found_count"] == 0
        assert result["not_found_count"] == 1
        assert result["not_found"] == ["DOES:NOT:EXIST"]

    def test_mixed_found_and_missing(self, mock_config, get_pv_metadata_fn, search_pvs_fn):
        search = json.loads(search_pvs_fn("SR:DIAG:BPM:*", page_size=1))
        pv_name = search["records"][0]["name"]

        result = json.loads(get_pv_metadata_fn([pv_name, "DOES:NOT:EXIST"]))
        assert result["found_count"] == 1
        assert result["not_found_count"] == 1

    def test_too_many_pvs(self, mock_config, get_pv_metadata_fn):
        pv_names = [f"PV:{i}" for i in range(101)]
        result = json.loads(get_pv_metadata_fn(pv_names))
        assert result["error"] is True
        assert result["error_type"] == "too_many_pvs"

    def test_error_without_registry(self, get_pv_metadata_fn):
        result = json.loads(get_pv_metadata_fn(["SR:DIAG:BPM:01:POSITION:X"]))
        assert result["error"] is True
        assert result["error_type"] == "backend_not_available"
