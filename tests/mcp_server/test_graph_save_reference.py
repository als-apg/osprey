"""Tests for the graph_save_reference MCP tool."""

import json

import pytest

from osprey.mcp_server.data_context import get_data_context, initialize_data_context


class TestGraphSaveReference:
    """Tests for graph_save_reference tool."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        """Set up workspace and seed a source entry."""
        ws = tmp_path / "osprey-workspace"
        ws.mkdir()
        (ws / "data").mkdir()
        initialize_data_context(workspace_root=ws)

        ctx = get_data_context()
        ctx.save(
            tool="graph_extract",
            data={
                "columns": ["time", "value"],
                "data": [[0, 10.0], [1, 20.0], [2, 15.0]],
                "title": "Extracted chart",
            },
            description="Extracted chart",
            summary={"num_points": 3},
            access_details={"format": "tabular"},
            data_type="graph_extraction",
        )

    @pytest.fixture
    def tool_fn(self):
        from tests.mcp_server.conftest import get_tool_fn

        from osprey.mcp_server.workspace.tools.graph_tools import graph_save_reference

        return get_tool_fn(graph_save_reference)

    async def test_save_reference_success(self, tool_fn):
        result = json.loads(
            await tool_fn(
                source_entry_id=1,
                title="Nominal beam profile",
                description="Baseline measurement from Jan 2024",
            )
        )
        assert result["status"] == "success"
        assert result["context_entry_id"] == 2
        assert result["summary"]["title"] == "Nominal beam profile"
        assert result["summary"]["source_entry_id"] == 1
        assert result["summary"]["num_points"] == 3

    async def test_save_reference_default_title(self, tool_fn):
        result = json.loads(await tool_fn(source_entry_id=1))
        assert result["status"] == "success"
        assert "Reference:" in result["description"]

    async def test_save_reference_not_found(self, tool_fn):
        result = json.loads(await tool_fn(source_entry_id=999))
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    async def test_saved_reference_is_discoverable(self, tool_fn):
        """Verify saved references can be found via data_type_filter."""
        await tool_fn(source_entry_id=1, title="My Reference")

        ctx = get_data_context()
        refs = ctx.list_entries(data_type_filter="graph_reference")
        assert len(refs) == 1
        assert refs[0].description == "My Reference"

    async def test_saved_reference_preserves_data(self, tool_fn, tmp_path):
        """Verify the reference contains the original data."""
        await tool_fn(source_entry_id=1, title="Preserved Data")

        ctx = get_data_context()
        ref_entry = ctx.get_entry(2)
        assert ref_entry is not None

        # Read the data file
        data_file = ctx.get_file_path(2)
        assert data_file is not None

        import json as json_mod

        with open(data_file) as f:
            payload = json_mod.load(f)

        ref_data = payload["data"]
        assert ref_data["reference_title"] == "Preserved Data"
        assert ref_data["source_entry_id"] == 1
