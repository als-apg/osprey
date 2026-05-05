"""Tests for the create_static_plot MCP tool."""

import json
from unittest.mock import AsyncMock, patch
from tests.mcp_server.conftest import extract_response_dict

import pytest

from osprey.mcp_server.workspace.execution.sandbox_executor import SandboxExecutionResult

# Mock target for the sandbox executor used by create_static_plot
_SANDBOX_EXEC_TARGET = "osprey.mcp_server.workspace.execution.sandbox_executor.execute_sandbox_code"
_SANDBOX_FOLDER_TARGET = (
    "osprey.mcp_server.workspace.execution.sandbox_executor.create_sandbox_execution_folder"
)


class TestCreateStaticPlot:
    """Tests for create_static_plot tool."""

    @pytest.fixture
    def tool_fn(self):
        """Get the raw tool function."""
        from osprey.mcp_server.workspace.tools.create_static_plot import create_static_plot
        from tests.mcp_server.conftest import get_tool_fn

        return get_tool_fn(create_static_plot)

    @pytest.fixture
    def mock_execution_folder(self, tmp_path):
        folder = tmp_path / "exec"
        folder.mkdir()
        return folder

    async def test_empty_code_returns_error(self, tool_fn):
        result = extract_response_dict(await tool_fn(code="", title="test"))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"
        assert "No plotting code" in result["error_message"]

    async def test_whitespace_code_returns_error(self, tool_fn):
        result = extract_response_dict(await tool_fn(code="   ", title="test"))
        assert result["error"] is True

    async def test_execution_failure(self, tool_fn, mock_execution_folder):
        mock_result = SandboxExecutionResult(
            success=False,
            stdout="",
            stderr="NameError: name 'undefined_var' is not defined",
            error_message="NameError: name 'undefined_var' is not defined",
        )

        with (
            patch(_SANDBOX_EXEC_TARGET, new_callable=AsyncMock, return_value=mock_result),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            result = extract_response_dict(await tool_fn(code="plt.plot(undefined_var)", title="Bad Plot"))

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
            result = extract_response_dict(await tool_fn(code="x = 1", title="No-op Plot"))

        assert result["status"] == "success"
        assert result["artifact_ids"] == []

    async def test_successful_plot_with_artifacts(self, tool_fn, tmp_path, mock_execution_folder):
        """Verify artifacts from save_artifact() calls are saved."""
        art_path = tmp_path / "abc123_sine_wave.png"
        art_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_result = SandboxExecutionResult(
            success=True,
            stdout="Artifact saved: Sine Wave (plot_png, 108 bytes)",
            stderr="",
            artifacts=[
                {
                    "path": art_path,
                    "title": "Sine Wave",
                    "description": "A simple sine wave plot",
                    "artifact_type": "plot_png",
                    "mime_type": "image/png",
                }
            ],
        )

        with (
            patch(_SANDBOX_EXEC_TARGET, new_callable=AsyncMock, return_value=mock_result),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            result = extract_response_dict(
                await tool_fn(
                    code="fig, ax = plt.subplots()\nax.plot([1,2,3])\nsave_artifact(fig, 'Sine Wave')",
                    title="Sine Wave",
                    description="A simple sine wave plot",
                )
            )

        assert result["status"] == "success"
        assert len(result["artifact_ids"]) == 1
        assert result["artifact_id"] is not None

    async def test_data_source_file_path(self, tool_fn, mock_execution_folder):
        """Verify data_source generates loading code in executed code."""
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
                code="plt.plot(data)",
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
                code="plt.plot(data)",
                title="Test",
                data_source="a1b2c3d4e5f6",
            )

        assert captured_code is not None
        # Resolved path should appear in the generated code
        assert str(art_file) in captured_code

    async def test_preamble_includes_matplotlib_imports(self, tool_fn, mock_execution_folder):
        """Verify matplotlib imports and styling are prepended."""
        captured_code = None

        async def mock_execute(code, execution_folder):
            nonlocal captured_code
            captured_code = code
            return SandboxExecutionResult(success=True, stdout="", stderr="", artifacts=[])

        with (
            patch(_SANDBOX_EXEC_TARGET, side_effect=mock_execute),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            await tool_fn(code="plt.plot([1,2])", title="Test")

        assert "import numpy as np" in captured_code
        assert "import pandas as pd" in captured_code
        assert "matplotlib.use('Agg')" in captured_code
        assert "plt.rcParams.update" in captured_code

    async def test_no_tight_layout_epilogue(self, tool_fn, mock_execution_folder):
        """Verify no plt.tight_layout() is auto-appended (must be explicit)."""
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

        # The preamble sets figure.autolayout but no explicit tight_layout call appended
        lines = captured_code.strip().split("\n")
        # Last line should be from user code, not tight_layout
        assert not lines[-1].strip().startswith("plt.tight_layout")
