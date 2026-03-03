"""Tests for the create_interactive_plot MCP tool."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from osprey.stores.artifact_store import get_artifact_store
from osprey.mcp_server.workspace.execution.sandbox_executor import SandboxExecutionResult
from osprey.mcp_server.workspace.tools._viz_common import resolve_data_source

# Mock target for the sandbox executor used by create_interactive_plot
_SANDBOX_EXEC_TARGET = "osprey.mcp_server.workspace.execution.sandbox_executor.execute_sandbox_code"
_SANDBOX_FOLDER_TARGET = (
    "osprey.mcp_server.workspace.execution.sandbox_executor.create_sandbox_execution_folder"
)


class TestCreateInteractivePlot:
    """Tests for create_interactive_plot tool."""

    @pytest.fixture
    def tool_fn(self):
        """Get the raw tool function."""
        from osprey.mcp_server.workspace.tools.create_interactive_plot import (
            create_interactive_plot,
        )
        from tests.mcp_server.conftest import get_tool_fn

        return get_tool_fn(create_interactive_plot)

    @pytest.fixture
    def mock_execution_folder(self, tmp_path):
        folder = tmp_path / "exec"
        folder.mkdir()
        return folder

    async def test_empty_code_returns_error(self, tool_fn):
        result = json.loads(await tool_fn(code="", title="test"))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"
        assert "No plotting code" in result["error_message"]

    async def test_whitespace_code_returns_error(self, tool_fn):
        result = json.loads(await tool_fn(code="   ", title="test"))
        assert result["error"] is True

    async def test_execution_failure(self, tool_fn, mock_execution_folder):
        mock_result = SandboxExecutionResult(
            success=False,
            stdout="",
            stderr="ImportError: No module named 'plotly'",
            error_message="ImportError: No module named 'plotly'",
        )

        with (
            patch(_SANDBOX_EXEC_TARGET, new_callable=AsyncMock, return_value=mock_result),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            result = json.loads(await tool_fn(code="import plotly", title="Bad Plot"))

        assert result["error"] is True
        assert result["error_type"] == "execution_error"

    async def test_successful_plot_no_artifacts(self, tool_fn, mock_execution_folder):
        mock_result = SandboxExecutionResult(
            success=True,
            stdout="",
            stderr="",
            artifacts=[],
        )

        with (
            patch(_SANDBOX_EXEC_TARGET, new_callable=AsyncMock, return_value=mock_result),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            result = json.loads(await tool_fn(code="x = 1", title="No-op Plot"))

        assert result["status"] == "success"
        assert result["artifact_ids"] == []

    async def test_successful_plotly_artifact(self, tool_fn, tmp_path, mock_execution_folder):
        """Verify Plotly HTML artifacts from save_artifact() are saved."""
        art_path = tmp_path / "abc123_interactive_chart.html"
        art_path.write_text("<html><body>plotly chart</body></html>")

        mock_result = SandboxExecutionResult(
            success=True,
            stdout="Artifact saved: Interactive Chart (plot_html, 42 bytes)",
            stderr="",
            artifacts=[
                {
                    "path": art_path,
                    "title": "Interactive Chart",
                    "description": "A Plotly chart",
                    "artifact_type": "plot_html",
                    "mime_type": "text/html",
                }
            ],
        )

        with (
            patch(_SANDBOX_EXEC_TARGET, new_callable=AsyncMock, return_value=mock_result),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            result = json.loads(
                await tool_fn(
                    code="import plotly.graph_objects as go\nfig = go.Figure()\nsave_artifact(fig, 'Interactive Chart')",
                    title="Interactive Chart",
                    description="A Plotly chart",
                )
            )

        assert result["status"] == "success"
        assert len(result["artifact_ids"]) == 1
        assert result["artifact_id"] is not None

    async def test_preamble_has_no_matplotlib(self, tool_fn, mock_execution_folder):
        """Verify no matplotlib imports in the interactive plot preamble."""
        captured_code = None

        async def mock_execute(code, execution_folder):
            nonlocal captured_code
            captured_code = code
            return SandboxExecutionResult(success=True, stdout="", stderr="", artifacts=[])

        with (
            patch(_SANDBOX_EXEC_TARGET, side_effect=mock_execute),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            await tool_fn(code="x = 1", title="Test")

        assert "import numpy as np" in captured_code
        assert "import pandas as pd" in captured_code
        # No matplotlib in the preamble
        assert "matplotlib" not in captured_code.split("x = 1")[0]

    async def test_data_source_file_path(self, tool_fn, mock_execution_folder):
        """Verify data_source generates loading code."""
        captured_code = None

        async def mock_execute(code, execution_folder):
            nonlocal captured_code
            captured_code = code
            return SandboxExecutionResult(success=True, stdout="", stderr="", artifacts=[])

        with (
            patch(_SANDBOX_EXEC_TARGET, side_effect=mock_execute),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            await tool_fn(
                code="pass",
                title="Test",
                data_source="/path/to/data.csv",
            )

        assert captured_code is not None
        assert "_data_path" in captured_code
        assert "pd.read_csv" in captured_code

    async def test_data_source_artifact_id(self, tool_fn, tmp_path, mock_execution_folder):
        """Verify artifact ID data_source resolves via ArtifactStore."""
        captured_code = None
        art_file = tmp_path / "a1b2c3d4e5f6_data.csv"
        art_file.write_text("x,y\n1,2\n")

        async def mock_execute(code, execution_folder):
            nonlocal captured_code
            captured_code = code
            return SandboxExecutionResult(success=True, stdout="", stderr="", artifacts=[])

        with (
            patch(_SANDBOX_EXEC_TARGET, side_effect=mock_execute),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
            patch("osprey.stores.artifact_store.get_artifact_store") as mock_store,
        ):
            mock_store.return_value.get_file_path.return_value = art_file
            await tool_fn(
                code="pass",
                title="Test",
                data_source="a1b2c3d4e5f6",
            )

        assert captured_code is not None
        # Resolved path should appear in the generated code
        assert str(art_file) in captured_code

    async def test_data_source_artifact_store_entry(self, tool_fn, tmp_path, mock_execution_folder):
        """Verify hex artifact ID data_source resolves via ArtifactStore."""
        # Save data to ArtifactStore so a hex artifact ID exists
        store = get_artifact_store()
        entry = store.save_data(
            tool="archiver_read",
            data={"dataframe": {}},
            title="test data",
            description="test data",
            summary={"channels_queried": 1},
            access_details={"format": "timeseries"},
            category="timeseries",
        )

        captured_code = None

        async def mock_execute(code, execution_folder):
            nonlocal captured_code
            captured_code = code
            return SandboxExecutionResult(success=True, stdout="", stderr="", artifacts=[])

        with (
            patch(_SANDBOX_EXEC_TARGET, side_effect=mock_execute),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            await tool_fn(
                code="pass",
                title="Test",
                data_source=entry.id,
            )

        assert captured_code is not None
        # Should contain the resolved absolute path
        assert "_data_path" in captured_code
        assert entry.filename in captured_code


class TestResolveDataSource:
    """Tests for the resolve_data_source helper."""

    def test_file_path_returned_as_is(self):
        result = resolve_data_source("/some/path/to/data.csv")
        assert result == "/some/path/to/data.csv"

    def test_relative_path_returned_as_is(self):
        result = resolve_data_source("data/002_archiver_read.json")
        assert result == "data/002_archiver_read.json"

    def test_artifact_id_resolves_via_store(self):
        with patch("osprey.stores.artifact_store.get_artifact_store") as mock_store:
            mock_store.return_value.get_file_path.return_value = Path("/art/file.csv")
            result = resolve_data_source("a1b2c3d4e5f6")
        assert result == "/art/file.csv"

    def test_artifact_id_not_found_raises(self):
        with patch("osprey.stores.artifact_store.get_artifact_store") as mock_store:
            mock_store.return_value.get_file_path.return_value = None
            with pytest.raises(FileNotFoundError, match="Artifact.*not found"):
                resolve_data_source("a1b2c3d4e5f6")

    def test_artifact_hex_id_resolves_via_artifact_store(self):
        """Hex artifact ID should resolve via ArtifactStore.get_file_path()."""
        store = get_artifact_store()
        entry = store.save_data(
            tool="test",
            data={"value": 42},
            title="test artifact",
            description="test",
        )

        result = resolve_data_source(entry.id)
        assert result.endswith(entry.filename)
        assert Path(result).exists()

    def test_artifact_hex_id_not_found_raises(self):
        """Non-existent hex artifact ID should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Artifact.*not found"):
            resolve_data_source("deadbeef0123")

    def test_numeric_string_treated_as_file_path(self):
        """Numeric strings are no longer context entry IDs; treated as file paths."""
        # Numeric strings don't match the 12-char hex pattern, so they
        # fall through to the "assume file path" branch and are returned as-is.
        result = resolve_data_source("42")
        assert result == "42"
