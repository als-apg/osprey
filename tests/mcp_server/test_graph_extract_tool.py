"""Tests for the graph_extract MCP tool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.stores.artifact_store import initialize_artifact_store


class TestGraphExtract:
    """Tests for graph_extract tool."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        """Set up workspace and artifact store."""
        ws = tmp_path / "osprey-workspace"
        ws.mkdir()
        (ws / "artifacts").mkdir()
        initialize_artifact_store(workspace_root=ws)

    @pytest.fixture
    def tool_fn(self):
        """Get the raw tool function."""
        from osprey.mcp_server.workspace.tools.graph_tools import graph_extract
        from tests.mcp_server.conftest import get_tool_fn

        return get_tool_fn(graph_extract)

    async def test_file_not_found(self, tool_fn):
        result = json.loads(await tool_fn(image_path="/nonexistent/chart.png"))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"

    async def test_unsupported_format(self, tool_fn, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("not an image")
        result = json.loads(await tool_fn(image_path=str(txt_file)))
        assert result["error"] is True
        assert "Unsupported image format" in result["error_message"]

    async def test_service_unavailable(self, tool_fn, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(b"fake-png")

        with patch("osprey.mcp_server.workspace.tools.graph_client.DePlotClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.is_available.return_value = False
            MockClient.return_value = mock_instance

            result = json.loads(await tool_fn(image_path=str(img)))

        assert result["error"] is True
        assert result["error_type"] == "service_unavailable"
        assert "uv run python -m osprey.services.deplot" in result["suggestions"][0]

    async def test_successful_extraction(self, tool_fn, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(b"fake-png")

        mock_result = {
            "columns": ["time", "current"],
            "data": [[0, 100], [1, 200], [2, 150]],
            "raw_table": "time | current\n0 | 100\n1 | 200\n2 | 150",
            "title": "Beam Current",
        }

        with patch("osprey.mcp_server.workspace.tools.graph_client.DePlotClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.is_available.return_value = True
            mock_instance.extract.return_value = mock_result
            MockClient.return_value = mock_instance

            result = json.loads(await tool_fn(image_path=str(img), title="Test Chart"))

        assert result["status"] == "success"
        assert result["artifact_id"] is not None
        assert result["summary"]["num_points"] == 3
        assert result["summary"]["columns"] == ["time", "current"]

    async def test_extraction_error(self, tool_fn, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(b"fake-png")

        with patch("osprey.mcp_server.workspace.tools.graph_client.DePlotClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.is_available.return_value = True
            mock_instance.extract.side_effect = RuntimeError("Model crashed")
            MockClient.return_value = mock_instance

            result = json.loads(await tool_fn(image_path=str(img)))

        assert result["error"] is True
        assert result["error_type"] == "extraction_error"
