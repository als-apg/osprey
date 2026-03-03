"""Tests for the graph_save_reference MCP tool."""

import json

import pytest

from osprey.stores.artifact_store import get_artifact_store, initialize_artifact_store


class TestGraphSaveReference:
    """Tests for graph_save_reference tool."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        """Set up workspace and seed a source entry."""
        ws = tmp_path / "osprey-workspace"
        ws.mkdir()
        (ws / "artifacts").mkdir()
        initialize_artifact_store(workspace_root=ws)

        store = get_artifact_store()
        source = store.save_data(
            tool="graph_extract",
            data={
                "columns": ["time", "value"],
                "data": [[0, 10.0], [1, 20.0], [2, 15.0]],
                "title": "Extracted chart",
            },
            title="Extracted chart",
            description="Extracted chart",
            summary={"num_points": 3},
            access_details={"format": "tabular"},
            category="graph_extraction",
        )
        self._source_id = source.id

    @pytest.fixture
    def tool_fn(self):
        from osprey.mcp_server.workspace.tools.graph_tools import graph_save_reference
        from tests.mcp_server.conftest import get_tool_fn

        return get_tool_fn(graph_save_reference)

    async def test_save_reference_success(self, tool_fn):
        result = json.loads(
            await tool_fn(
                source_entry_id=self._source_id,
                title="Nominal beam profile",
                description="Baseline measurement from Jan 2024",
            )
        )
        assert result["status"] == "success"
        assert isinstance(result["artifact_id"], str)
        assert result["summary"]["title"] == "Nominal beam profile"
        assert result["summary"]["source_entry_id"] == self._source_id
        assert result["summary"]["num_points"] == 3

    async def test_save_reference_default_title(self, tool_fn):
        result = json.loads(await tool_fn(source_entry_id=self._source_id))
        assert result["status"] == "success"
        assert "Reference:" in result["title"]

    async def test_save_reference_not_found(self, tool_fn):
        result = json.loads(await tool_fn(source_entry_id="nonexistent_id"))
        assert result["error"] is True
        assert result["error_type"] == "not_found"

    async def test_saved_reference_is_discoverable(self, tool_fn):
        """Verify saved references can be found via category_filter."""
        await tool_fn(source_entry_id=self._source_id, title="My Reference")

        store = get_artifact_store()
        refs = store.list_entries(category_filter="graph_reference")
        assert len(refs) == 1
        assert refs[0].title == "My Reference"

    async def test_saved_reference_preserves_data(self, tool_fn, tmp_path):
        """Verify the reference contains the original data."""
        raw = await tool_fn(source_entry_id=self._source_id, title="Preserved Data")
        ref_result = json.loads(raw)
        ref_id = ref_result["artifact_id"]

        store = get_artifact_store()
        ref_entry = store.get_entry(ref_id)
        assert ref_entry is not None

        # Read the data file (raw JSON, no envelope)
        data_file = store.get_file_path(ref_id)
        assert data_file is not None

        import json as json_mod

        with open(data_file) as f:
            payload = json_mod.load(f)

        assert payload["reference_title"] == "Preserved Data"
        assert payload["source_entry_id"] == self._source_id
