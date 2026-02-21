"""Tests for the OSPREY type registry — single source of truth for type metadata."""

import json
import re

import pytest

from osprey.mcp_server.type_registry import (
    ARTIFACT_TYPES,
    DATA_TYPES,
    TOOL_TYPES,
    get_artifact_types,
    get_data_types,
    get_tool_types,
    registry_to_api_dict,
    valid_data_type_keys,
)

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class TestTypeDefs:
    """Every TypeDef must have a non-empty label and a valid 6-digit hex colour."""

    @pytest.mark.parametrize("key,td", list(ARTIFACT_TYPES.items()), ids=list(ARTIFACT_TYPES))
    def test_artifact_type_fields(self, key, td):
        assert td.key == key
        assert td.label, f"artifact type {key!r} has empty label"
        assert HEX_RE.match(td.color), f"artifact type {key!r} has invalid colour {td.color!r}"

    @pytest.mark.parametrize("key,td", list(DATA_TYPES.items()), ids=list(DATA_TYPES))
    def test_data_type_fields(self, key, td):
        assert td.key == key
        assert td.label, f"data type {key!r} has empty label"
        assert HEX_RE.match(td.color), f"data type {key!r} has invalid colour {td.color!r}"

    @pytest.mark.parametrize("key,td", list(TOOL_TYPES.items()), ids=list(TOOL_TYPES))
    def test_tool_type_fields(self, key, td):
        assert td.key == key
        assert td.label, f"tool type {key!r} has empty label"
        assert HEX_RE.match(td.color), f"tool type {key!r} has invalid colour {td.color!r}"


class TestPublicAPI:
    def test_get_artifact_types_returns_copy(self):
        a = get_artifact_types()
        a["bogus"] = None  # mutate
        assert "bogus" not in ARTIFACT_TYPES

    def test_get_data_types_returns_copy(self):
        d = get_data_types()
        d["bogus"] = None
        assert "bogus" not in DATA_TYPES

    def test_get_tool_types_returns_copy(self):
        t = get_tool_types()
        t["bogus"] = None
        assert "bogus" not in TOOL_TYPES

    def test_valid_data_type_keys_matches_data_types(self):
        assert valid_data_type_keys() == set(DATA_TYPES)


class TestRegistryToAPIDict:
    def test_json_serialisable(self):
        d = registry_to_api_dict()
        # Should not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_structure(self):
        d = registry_to_api_dict()
        assert set(d.keys()) == {"artifact_types", "data_types", "tool_types"}
        for domain in d.values():
            for _key, info in domain.items():
                assert "label" in info
                assert "color" in info

    def test_all_artifact_types_present(self):
        d = registry_to_api_dict()
        assert set(d["artifact_types"]) == set(ARTIFACT_TYPES)

    def test_all_data_types_present(self):
        d = registry_to_api_dict()
        assert set(d["data_types"]) == set(DATA_TYPES)

    def test_all_tool_types_present(self):
        d = registry_to_api_dict()
        assert set(d["tool_types"]) == set(TOOL_TYPES)


class TestKnownDataTypes:
    """Verify that every data_type string produced by Python tools is in the registry."""

    TOOL_PRODUCED_TYPES = [
        "timeseries",       # archiver_read
        "channel_values",   # channel_read
        "write_results",    # channel_write
        "code_output",      # execute
        "visualization",    # create_static_plot, create_interactive_plot
        "dashboard",        # create_dashboard
        "document",         # create_document
        "memory",           # memory_save
        "screenshot",       # screen_capture
        "graph_extraction", # graph_extract
        "graph_comparison", # graph_compare
        "graph_reference",  # graph_save_reference
        "agent_response",   # submit_response (default)
        "channel_addresses",  # agents via submit_response
        "logbook_research",   # agents via submit_response
        "search_results",     # agents via submit_response
    ]

    @pytest.mark.parametrize("dt", TOOL_PRODUCED_TYPES)
    def test_tool_data_type_registered(self, dt):
        assert dt in valid_data_type_keys(), f"{dt!r} not in registry"


class TestBugFixes:
    """Regression: the bugs identified in the plan must stay fixed."""

    def test_code_result_not_in_registry(self):
        """JS had 'code_result' but Python uses 'code_output'."""
        assert "code_result" not in DATA_TYPES

    def test_code_output_in_registry(self):
        assert "code_output" in DATA_TYPES

    def test_channel_list_not_in_registry(self):
        """Phantom type — no producer ever existed."""
        assert "channel_list" not in DATA_TYPES
