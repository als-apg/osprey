"""Tests for all 7 MATLAB MML MCP tools."""

import json

import pytest

from osprey.mcp_server.matlab.tools.browse import mml_browse
from osprey.mcp_server.matlab.tools.dependencies import mml_dependencies
from osprey.mcp_server.matlab.tools.get_function import mml_get
from osprey.mcp_server.matlab.tools.list_groups import mml_list_groups
from osprey.mcp_server.matlab.tools.path import mml_path
from osprey.mcp_server.matlab.tools.search import mml_search
from osprey.mcp_server.matlab.tools.stats import mml_stats
from tests.matlab.conftest import get_tool_fn

# --- mml_search ---------------------------------------------------------------


class TestMmlSearch:
    @pytest.mark.asyncio
    async def test_basic_search(self, patch_db):
        fn = get_tool_fn(mml_search)
        result = json.loads(await fn(query="orbit correction"))
        assert "error" not in result
        assert result["results_found"] >= 1
        names = [f["function_name"] for f in result["functions"]]
        assert "orbitcorrection" in names

    @pytest.mark.asyncio
    async def test_search_with_group_filter(self, patch_db):
        fn = get_tool_fn(mml_search)
        result = json.loads(await fn(query="BPM", group="StorageRing"))
        assert "error" not in result
        for func in result["functions"]:
            assert func["group"] == "StorageRing"

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self, patch_db):
        fn = get_tool_fn(mml_search)
        result = json.loads(await fn(query="init config", type="script"))
        assert "error" not in result
        for func in result["functions"]:
            assert func["type"] == "script"

    @pytest.mark.asyncio
    async def test_search_empty_query(self, patch_db):
        fn = get_tool_fn(mml_search)
        result = json.loads(await fn(query=""))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_search_no_results(self, patch_db):
        fn = get_tool_fn(mml_search)
        result = json.loads(await fn(query="xyznonexistentterm123"))
        assert "error" not in result
        assert result["results_found"] == 0

    @pytest.mark.asyncio
    async def test_search_limit(self, patch_db):
        fn = get_tool_fn(mml_search)
        result = json.loads(await fn(query="BPM orbit", limit=2))
        assert result["results_found"] <= 2


# --- mml_get ------------------------------------------------------------------


class TestMmlGet:
    @pytest.mark.asyncio
    async def test_get_existing_function(self, patch_db):
        fn = get_tool_fn(mml_get)
        result = json.loads(await fn(function_name="getbpm"))
        assert "error" not in result
        assert result["function_name"] == "getbpm"
        assert result["group"] == "StorageRing"
        assert result["type"] == "defined"
        assert result["in_degree"] == 25

    @pytest.mark.asyncio
    async def test_get_includes_callers_and_callees(self, patch_db):
        fn = get_tool_fn(mml_get)
        result = json.loads(await fn(function_name="getbpm"))
        assert "callers" in result
        assert "callees" in result
        # getbpm is called by orbitcorrection and bts_init
        assert "orbitcorrection" in result["callers"]
        assert "bts_init" in result["callers"]
        # getbpm calls family2channel
        assert "family2channel" in result["callees"]

    @pytest.mark.asyncio
    async def test_get_without_source(self, patch_db):
        fn = get_tool_fn(mml_get)
        result = json.loads(await fn(function_name="getbpm", include_source=False))
        assert "source_code" not in result

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, patch_db):
        fn = get_tool_fn(mml_get)
        result = json.loads(await fn(function_name="nonexistent_func"))
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    @pytest.mark.asyncio
    async def test_get_empty_name(self, patch_db):
        fn = get_tool_fn(mml_get)
        result = json.loads(await fn(function_name=""))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"


# --- mml_browse ---------------------------------------------------------------


class TestMmlBrowse:
    @pytest.mark.asyncio
    async def test_browse_by_group(self, patch_db):
        fn = get_tool_fn(mml_browse)
        result = json.loads(await fn(group="StorageRing"))
        assert "error" not in result
        assert result["total_matching"] == 3
        for func in result["functions"]:
            assert func["group"] == "StorageRing"

    @pytest.mark.asyncio
    async def test_browse_by_type(self, patch_db):
        fn = get_tool_fn(mml_browse)
        result = json.loads(await fn(type="script"))
        assert "error" not in result
        for func in result["functions"]:
            assert func["type"] == "script"

    @pytest.mark.asyncio
    async def test_browse_no_filter_error(self, patch_db):
        fn = get_tool_fn(mml_browse)
        result = json.loads(await fn())
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_browse_sort_by_in_degree(self, patch_db):
        fn = get_tool_fn(mml_browse)
        result = json.loads(
            await fn(group="StorageRing", sort_by="in_degree", order="desc")
        )
        assert "error" not in result
        degrees = [f["in_degree"] for f in result["functions"]]
        assert degrees == sorted(degrees, reverse=True)

    @pytest.mark.asyncio
    async def test_browse_pagination(self, patch_db):
        fn = get_tool_fn(mml_browse)
        result = json.loads(await fn(group="StorageRing", limit=2, offset=0))
        assert result["results_returned"] <= 2
        assert result["offset"] == 0


# --- mml_dependencies ---------------------------------------------------------


class TestMmlDependencies:
    @pytest.mark.asyncio
    async def test_callees_depth_1(self, patch_db):
        fn = get_tool_fn(mml_dependencies)
        result = json.loads(
            await fn(function_name="orbitcorrection", direction="callees", depth=1)
        )
        assert "error" not in result
        callees = [c["function_name"] for c in result["callees"]]
        assert "getbpm" in callees
        assert "setsp" in callees
        assert "getgolden" in callees

    @pytest.mark.asyncio
    async def test_callers_depth_1(self, patch_db):
        fn = get_tool_fn(mml_dependencies)
        result = json.loads(
            await fn(function_name="family2channel", direction="callers", depth=1)
        )
        assert "error" not in result
        callers = [c["function_name"] for c in result["callers"]]
        assert "getbpm" in callers
        assert "setsp" in callers
        assert "orbitcorrection" in callers

    @pytest.mark.asyncio
    async def test_both_directions(self, patch_db):
        fn = get_tool_fn(mml_dependencies)
        result = json.loads(
            await fn(function_name="setsp", direction="both", depth=1)
        )
        assert "error" not in result
        assert "callers" in result
        assert "callees" in result

    @pytest.mark.asyncio
    async def test_deeper_traversal(self, patch_db):
        fn = get_tool_fn(mml_dependencies)
        result = json.loads(
            await fn(function_name="orbitcorrection", direction="callees", depth=2)
        )
        assert "error" not in result
        # Depth 2 should include family2channel (called by getbpm which is called by orbitcorrection)
        callees = [c["function_name"] for c in result["callees"]]
        assert "family2channel" in callees

    @pytest.mark.asyncio
    async def test_nonexistent_function(self, patch_db):
        fn = get_tool_fn(mml_dependencies)
        result = json.loads(await fn(function_name="nonexistent"))
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    @pytest.mark.asyncio
    async def test_invalid_direction(self, patch_db):
        fn = get_tool_fn(mml_dependencies)
        result = json.loads(
            await fn(function_name="getbpm", direction="invalid")
        )
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_empty_function_name(self, patch_db):
        fn = get_tool_fn(mml_dependencies)
        result = json.loads(await fn(function_name=""))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"


# --- mml_path -----------------------------------------------------------------


class TestMmlPath:
    @pytest.mark.asyncio
    async def test_direct_path(self, patch_db):
        fn = get_tool_fn(mml_path)
        result = json.loads(await fn(source="orbitcorrection", target="getbpm"))
        assert "error" not in result
        assert result["path"] is not None
        assert result["path"][0] == "orbitcorrection"
        assert result["path"][-1] == "getbpm"
        assert result["length"] == 1

    @pytest.mark.asyncio
    async def test_indirect_path(self, patch_db):
        fn = get_tool_fn(mml_path)
        result = json.loads(
            await fn(source="bts_init", target="family2channel")
        )
        assert "error" not in result
        assert result["path"] is not None
        assert result["path"][0] == "bts_init"
        assert result["path"][-1] == "family2channel"
        assert result["length"] >= 1

    @pytest.mark.asyncio
    async def test_same_source_target(self, patch_db):
        fn = get_tool_fn(mml_path)
        result = json.loads(await fn(source="getbpm", target="getbpm"))
        assert result["path"] == ["getbpm"]
        assert result["length"] == 0

    @pytest.mark.asyncio
    async def test_no_path(self, patch_db):
        fn = get_tool_fn(mml_path)
        # family2channel doesn't call anything in our sample
        result = json.loads(
            await fn(source="family2channel", target="orbitcorrection")
        )
        assert result["path"] is None

    @pytest.mark.asyncio
    async def test_nonexistent_source(self, patch_db):
        fn = get_tool_fn(mml_path)
        result = json.loads(await fn(source="nonexistent", target="getbpm"))
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    @pytest.mark.asyncio
    async def test_empty_source(self, patch_db):
        fn = get_tool_fn(mml_path)
        result = json.loads(await fn(source="", target="getbpm"))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"


# --- mml_list_groups ----------------------------------------------------------


class TestMmlListGroups:
    @pytest.mark.asyncio
    async def test_list_all_groups(self, patch_db):
        fn = get_tool_fn(mml_list_groups)
        result = json.loads(await fn())
        assert "error" not in result
        assert result["groups_found"] == 6  # StorageRing, Common, MML, BTS, GTB, Booster
        names = [g["group"] for g in result["groups"]]
        assert "StorageRing" in names
        assert "Common" in names
        assert "MML" in names

    @pytest.mark.asyncio
    async def test_groups_sorted_by_count(self, patch_db):
        fn = get_tool_fn(mml_list_groups)
        result = json.loads(await fn())
        counts = [g["function_count"] for g in result["groups"]]
        assert counts == sorted(counts, reverse=True)


# --- mml_stats ----------------------------------------------------------------


class TestMmlStats:
    @pytest.mark.asyncio
    async def test_stats(self, patch_db):
        fn = get_tool_fn(mml_stats)
        result = json.loads(await fn())
        assert "error" not in result
        stats = result["statistics"]
        assert stats["total_functions"] == 8
        assert stats["total_edges"] == 9  # 10 - 1 self-loop
        assert "groups" in stats
        assert "types" in stats

    @pytest.mark.asyncio
    async def test_stats_groups_is_dict(self, patch_db):
        fn = get_tool_fn(mml_stats)
        result = json.loads(await fn())
        stats = result["statistics"]
        assert isinstance(stats["groups"], dict)
        assert "StorageRing" in stats["groups"]

    @pytest.mark.asyncio
    async def test_stats_top_called(self, patch_db):
        fn = get_tool_fn(mml_stats)
        result = json.loads(await fn())
        stats = result["statistics"]
        assert isinstance(stats["top_called"], dict)
        # family2channel has highest in_degree (40)
        assert "family2channel" in stats["top_called"]
