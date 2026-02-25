"""Tests for the graph_compare MCP tool."""

import json

import pytest

from osprey.mcp_server.artifact_store import get_artifact_store, initialize_artifact_store


class TestGraphCompare:
    """Tests for graph_compare tool."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        """Set up workspace and seed artifact store with test entries."""
        ws = tmp_path / "osprey-workspace"
        ws.mkdir()
        (ws / "artifacts").mkdir()
        initialize_artifact_store(workspace_root=ws)

        store = get_artifact_store()

        # Seed current data
        current = store.save_data(
            tool="graph_extract",
            data={
                "columns": ["x", "y"],
                "data": [[0, 1.0], [1, 2.0], [2, 3.0], [3, 4.0], [4, 5.0]],
            },
            title="Current measurement",
            description="Current measurement",
            summary={"num_points": 5},
            access_details={"format": "tabular"},
            category="graph_extraction",
        )

        # Seed reference data
        reference = store.save_data(
            tool="graph_save_reference",
            data={
                "columns": ["x", "y"],
                "data": [[0, 1.1], [1, 2.1], [2, 3.1], [3, 4.1], [4, 5.1]],
                "reference_title": "Nominal profile",
            },
            title="Nominal profile",
            description="Nominal profile",
            summary={"num_points": 5},
            access_details={"format": "tabular"},
            category="graph_reference",
        )

        self._current_id = current.id
        self._reference_id = reference.id

    @pytest.fixture
    def tool_fn(self):
        from osprey.mcp_server.workspace.tools.graph_tools import graph_compare
        from tests.mcp_server.conftest import get_tool_fn

        return get_tool_fn(graph_compare)

    async def test_compare_by_entry_ids(self, tool_fn):
        result = json.loads(
            await tool_fn(
                current_entry_id=self._current_id,
                reference_entry_id=self._reference_id,
            )
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
            await tool_fn(
                current_entry_id=self._current_id,
                reference_query="Nominal",
            )
        )
        assert result["status"] == "success"
        assert result["comparison"]["reference_entry_id"] == self._reference_id

    async def test_missing_current_entry(self, tool_fn):
        result = json.loads(
            await tool_fn(
                current_entry_id="nonexistent_id",
                reference_entry_id=self._reference_id,
            )
        )
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    async def test_missing_reference_entry(self, tool_fn):
        result = json.loads(
            await tool_fn(
                current_entry_id=self._current_id,
                reference_entry_id="nonexistent_id",
            )
        )
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    async def test_no_reference_provided(self, tool_fn):
        result = json.loads(await tool_fn(current_entry_id=self._current_id))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    async def test_query_no_match(self, tool_fn):
        result = json.loads(
            await tool_fn(
                current_entry_id=self._current_id,
                reference_query="nonexistent",
            )
        )
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    async def test_selected_metrics(self, tool_fn):
        result = json.loads(
            await tool_fn(
                current_entry_id=self._current_id,
                reference_entry_id=self._reference_id,
                metrics=["rmse"],
            )
        )
        assert result["status"] == "success"
        metrics = result["comparison"]["metrics"]
        assert "rmse" in metrics
        assert "correlation" not in metrics
