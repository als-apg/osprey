"""Tests for the graph_compare MCP tool."""

import json

import pytest

from osprey.mcp_server.data_context import get_data_context, initialize_data_context


class TestGraphCompare:
    """Tests for graph_compare tool."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        """Set up workspace and seed data context with test entries."""
        ws = tmp_path / "osprey-workspace"
        ws.mkdir()
        (ws / "data").mkdir()
        initialize_data_context(workspace_root=ws)

        # Seed current data
        ctx = get_data_context()
        ctx.save(
            tool="graph_extract",
            data={
                "columns": ["x", "y"],
                "data": [[0, 1.0], [1, 2.0], [2, 3.0], [3, 4.0], [4, 5.0]],
            },
            description="Current measurement",
            summary={"num_points": 5},
            access_details={"format": "tabular"},
            data_type="graph_extraction",
        )

        # Seed reference data
        ctx.save(
            tool="graph_save_reference",
            data={
                "columns": ["x", "y"],
                "data": [[0, 1.1], [1, 2.1], [2, 3.1], [3, 4.1], [4, 5.1]],
                "reference_title": "Nominal profile",
            },
            description="Nominal profile",
            summary={"num_points": 5},
            access_details={"format": "tabular"},
            data_type="graph_reference",
        )

    @pytest.fixture
    def tool_fn(self):
        from tests.mcp_server.conftest import get_tool_fn

        from osprey.mcp_server.workspace.tools.graph_tools import graph_compare

        return get_tool_fn(graph_compare)

    async def test_compare_by_entry_ids(self, tool_fn):
        result = json.loads(
            await tool_fn(current_entry_id=1, reference_entry_id=2)
        )
        assert result["status"] == "success"
        assert "comparison" in result
        metrics = result["comparison"]["metrics"]
        assert "rmse" in metrics
        assert "correlation" in metrics
        assert metrics["rmse"] >= 0
        assert abs(metrics["correlation"]) <= 1.0

    async def test_compare_by_query(self, tool_fn):
        result = json.loads(
            await tool_fn(current_entry_id=1, reference_query="Nominal")
        )
        assert result["status"] == "success"
        assert result["comparison"]["reference_entry_id"] == 2

    async def test_missing_current_entry(self, tool_fn):
        result = json.loads(
            await tool_fn(current_entry_id=999, reference_entry_id=2)
        )
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    async def test_missing_reference_entry(self, tool_fn):
        result = json.loads(
            await tool_fn(current_entry_id=1, reference_entry_id=999)
        )
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    async def test_no_reference_provided(self, tool_fn):
        result = json.loads(
            await tool_fn(current_entry_id=1)
        )
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    async def test_query_no_match(self, tool_fn):
        result = json.loads(
            await tool_fn(current_entry_id=1, reference_query="nonexistent")
        )
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    async def test_selected_metrics(self, tool_fn):
        result = json.loads(
            await tool_fn(
                current_entry_id=1,
                reference_entry_id=2,
                metrics=["rmse"],
            )
        )
        assert result["status"] == "success"
        metrics = result["comparison"]["metrics"]
        assert "rmse" in metrics
        assert "correlation" not in metrics
