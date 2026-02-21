"""Tests for the create_dashboard MCP tool."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.workspace.execution.sandbox_executor import SandboxExecutionResult

# Mock targets for the sandbox executor used by create_dashboard
_SANDBOX_EXEC_TARGET = (
    "osprey.mcp_server.workspace.execution.sandbox_executor.execute_sandbox_code"
)
_SANDBOX_FOLDER_TARGET = (
    "osprey.mcp_server.workspace.execution.sandbox_executor.create_sandbox_execution_folder"
)
# Mock target for the bokeh availability check
_BOKEH_CHECK_TARGET = (
    "osprey.mcp_server.workspace.tools.create_dashboard.importlib"
)


class TestCreateDashboard:
    """Tests for create_dashboard tool."""

    @pytest.fixture
    def tool_fn(self):
        """Get the raw tool function."""
        from osprey.mcp_server.workspace.tools.create_dashboard import create_dashboard
        from tests.mcp_server.conftest import get_tool_fn

        return get_tool_fn(create_dashboard)

    @pytest.fixture
    def mock_execution_folder(self, tmp_path):
        folder = tmp_path / "exec"
        folder.mkdir()
        return folder

    @pytest.fixture(autouse=True)
    def mock_bokeh_available(self):
        """Mock bokeh as available for all tests (overridden in specific test)."""
        mock_importlib = MagicMock()
        mock_importlib.import_module.return_value = MagicMock()
        with patch(_BOKEH_CHECK_TARGET, mock_importlib):
            yield

    async def test_empty_code_returns_error(self, tool_fn):
        result = json.loads(await tool_fn(code="", title="test"))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"
        assert "No dashboard code" in result["error_message"]

    async def test_bokeh_not_installed(self, tool_fn):
        """Verify helpful error when bokeh is not installed."""
        # Override the autouse fixture
        mock_importlib = MagicMock()
        mock_importlib.import_module.side_effect = ImportError("No module named 'bokeh'")
        with patch(_BOKEH_CHECK_TARGET, mock_importlib):
            result = json.loads(
                await tool_fn(code="p = figure()", title="No Bokeh")
            )

        assert result["error"] is True
        assert result["error_type"] == "dependency_missing"
        assert "Bokeh" in result["error_message"]
        assert any("uv sync" in s for s in result["suggestions"])
        assert any("Plotly" in s for s in result["suggestions"])

    async def test_execution_failure(self, tool_fn, mock_execution_folder):
        mock_result = SandboxExecutionResult(
            success=False,
            stdout="",
            stderr="ImportError: No module named 'bokeh'",
            error_message="ImportError: No module named 'bokeh'",
        )

        with (
            patch(_SANDBOX_EXEC_TARGET, new_callable=AsyncMock, return_value=mock_result),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            result = json.loads(
                await tool_fn(code="p = figure()", title="Bad Dashboard")
            )

        assert result["error"] is True
        assert result["error_type"] == "execution_error"

    async def test_no_artifacts_produced(self, tool_fn, mock_execution_folder):
        """Dashboard code ran but didn't produce any artifacts via save_artifact()."""
        mock_result = SandboxExecutionResult(
            success=True,
            stdout="some output",
            stderr="",
            artifacts=[],
        )

        with (
            patch(_SANDBOX_EXEC_TARGET, new_callable=AsyncMock, return_value=mock_result),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            result = json.loads(
                await tool_fn(code="x = 1", title="Empty Dashboard")
            )

        assert result["error"] is True
        assert result["error_type"] == "execution_error"
        assert "no artifacts" in result["error_message"].lower()

    async def test_successful_dashboard(self, tool_fn, tmp_path, mock_execution_folder):
        """Dashboard with save_artifact() producing HTML."""
        html_path = tmp_path / "abc123_dashboard.html"
        html_content = "<html><body><h1>Dashboard</h1></body></html>"
        html_path.write_text(html_content)

        mock_result = SandboxExecutionResult(
            success=True,
            stdout="Artifact saved: Test Dashboard (dashboard_html, 47 bytes)",
            stderr="",
            artifacts=[
                {
                    "path": html_path,
                    "title": "Test Dashboard",
                    "description": "A test dashboard",
                    "artifact_type": "dashboard_html",
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
                    code="p = figure(title='Test')\nsave_artifact(p, 'Test Dashboard')",
                    title="Test Dashboard",
                    description="A test dashboard",
                )
            )

        assert result["status"] == "success"
        assert len(result["artifact_ids"]) == 1
        assert result["context_entry_id"] is not None

    async def test_preamble_includes_bokeh_imports(
        self, tool_fn, mock_execution_folder
    ):
        """Verify Bokeh imports are prepended to the code."""
        captured_code = None

        async def mock_execute(code, execution_folder):
            nonlocal captured_code
            captured_code = code
            return SandboxExecutionResult(
                success=True,
                stdout="",
                stderr="",
                artifacts=[],
            )

        with (
            patch(_SANDBOX_EXEC_TARGET, side_effect=mock_execute),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            # Will return error (no artifacts) but we just want to capture code
            await tool_fn(code="pass", title="Test")

        assert captured_code is not None
        assert "from bokeh.plotting import figure" in captured_code
        assert "from bokeh.layouts import column, row" in captured_code
        assert "import numpy as np" in captured_code

    async def test_no_epilogue_or_stdout_marker(self, tool_fn, mock_execution_folder):
        """Verify no auto-detect epilogue or __DASHBOARD_OUTPUT__ marker."""
        captured_code = None

        async def mock_execute(code, execution_folder):
            nonlocal captured_code
            captured_code = code
            return SandboxExecutionResult(
                success=True,
                stdout="",
                stderr="",
                artifacts=[],
            )

        with (
            patch(_SANDBOX_EXEC_TARGET, side_effect=mock_execute),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            await tool_fn(code="pass", title="Test")

        assert "__DASHBOARD_OUTPUT__" not in captured_code
        assert "_dashboard_output_path" not in captured_code
        assert "tempfile" not in captured_code

    async def test_multiple_artifacts(self, tool_fn, tmp_path, mock_execution_folder):
        """Multiple save_artifact() calls produce multiple artifacts."""
        html1 = tmp_path / "abc_panel1.html"
        html1.write_text("<html>panel 1</html>")
        html2 = tmp_path / "def_panel2.html"
        html2.write_text("<html>panel 2</html>")

        mock_result = SandboxExecutionResult(
            success=True,
            stdout="",
            stderr="",
            artifacts=[
                {
                    "path": html1,
                    "title": "Panel 1",
                    "description": "",
                    "artifact_type": "dashboard_html",
                    "mime_type": "text/html",
                },
                {
                    "path": html2,
                    "title": "Panel 2",
                    "description": "",
                    "artifact_type": "dashboard_html",
                    "mime_type": "text/html",
                },
            ],
        )

        with (
            patch(_SANDBOX_EXEC_TARGET, new_callable=AsyncMock, return_value=mock_result),
            patch(_SANDBOX_FOLDER_TARGET, return_value=mock_execution_folder),
        ):
            result = json.loads(
                await tool_fn(code="pass", title="Multi-panel Dashboard")
            )

        assert result["status"] == "success"
        assert len(result["artifact_ids"]) == 2
