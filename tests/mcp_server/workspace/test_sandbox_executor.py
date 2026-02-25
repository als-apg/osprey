"""Tests for the sandboxed execution engine."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.mcp_server.workspace.execution.sandbox_executor import (
    SandboxExecutionResult,
    _collect_artifacts,
    create_sandbox_execution_folder,
    execute_sandbox_code,
    validate_sandbox_code,
)


# ---------------------------------------------------------------------------
# validate_sandbox_code tests
# ---------------------------------------------------------------------------
class TestValidateSandboxCode:
    """Tests for the AST-level code validation."""

    def test_validate_safe_matplotlib_code(self):
        code = textwrap.dedent("""\
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            x = np.linspace(0, 10, 100)
            plt.plot(x, np.sin(x))
        """)
        is_safe, violations = validate_sandbox_code(code)
        assert is_safe
        assert violations == []

    def test_validate_blocks_subprocess(self):
        code = "import subprocess\nsubprocess.run(['ls'])"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("subprocess" in v for v in violations)

    def test_validate_blocks_eval(self):
        code = "result = eval('1+1')"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("eval" in v for v in violations)

    def test_validate_blocks_exec(self):
        code = "exec('import os')"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("exec" in v for v in violations)

    def test_validate_blocks_network(self):
        for mod in ["requests", "urllib", "socket"]:
            code = f"import {mod}"
            is_safe, violations = validate_sandbox_code(code)
            assert not is_safe, f"{mod} should be blocked"
            assert any(mod in v for v in violations)

    def test_validate_blocks_epics(self):
        code = "import epics\nepics.caput('PV:NAME', 42)"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("epics" in v for v in violations)

    def test_validate_blocks_dunder_import(self):
        code = "__import__('os').system('ls')"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("__import__" in v for v in violations)

    def test_validate_syntax_error(self):
        code = "def foo(\n  # missing closing paren"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("Syntax error" in v for v in violations)

    def test_validate_allowed_stdlib(self):
        code = textwrap.dedent("""\
            import json
            import datetime
            import math
            import re
            import collections
            from pathlib import Path
            import itertools
        """)
        is_safe, violations = validate_sandbox_code(code)
        assert is_safe
        assert violations == []

    def test_validate_blocks_os_system(self):
        code = "import os\nos.system('rm -rf /')"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("os.system" in v for v in violations)

    def test_validate_blocks_compile(self):
        code = "compile('print(1)', '<string>', 'exec')"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("compile" in v for v in violations)

    def test_validate_blocks_ctypes(self):
        code = "import ctypes"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("ctypes" in v for v in violations)

    def test_validate_blocks_shutil_rmtree(self):
        code = "import shutil\nshutil.rmtree('/tmp/important')"
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert any("shutil.rmtree" in v for v in violations)

    def test_validate_mpl_toolkits_3d_import_allowed(self):
        """mpl_toolkits.mplot3d must be whitelisted for 3D matplotlib plots."""
        code = textwrap.dedent("""\
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        """)
        is_safe, violations = validate_sandbox_code(code)
        assert is_safe, f"mpl_toolkits.mplot3d should be allowed but got: {violations}"
        assert violations == []

    def test_validate_mpl_toolkits_import_styles(self):
        """All common mpl_toolkits import patterns should pass validation."""
        patterns = [
            "import mpl_toolkits",
            "from mpl_toolkits.mplot3d import Axes3D",
            "from mpl_toolkits.mplot3d import art3d",
            "import mpl_toolkits.mplot3d",
        ]
        for code in patterns:
            is_safe, violations = validate_sandbox_code(code)
            assert is_safe, f"Pattern '{code}' should be allowed but got: {violations}"

    def test_validate_plotly_import_allowed(self):
        code = "import plotly.graph_objects as go\nfig = go.Figure()"
        is_safe, violations = validate_sandbox_code(code)
        assert is_safe

    def test_validate_plotly_3d_scatter_allowed(self):
        """Plotly 3D scatter imports should be allowed."""
        code = textwrap.dedent("""\
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Scatter3d(x=[1,2], y=[3,4], z=[5,6])])
        """)
        is_safe, violations = validate_sandbox_code(code)
        assert is_safe

    def test_validate_bokeh_import_allowed(self):
        code = "from bokeh.plotting import figure\np = figure()"
        is_safe, violations = validate_sandbox_code(code)
        assert is_safe

    def test_validate_seaborn_import_allowed(self):
        code = "import seaborn as sns"
        is_safe, violations = validate_sandbox_code(code)
        assert is_safe

    def test_validate_scipy_import_allowed(self):
        code = "from scipy import stats"
        is_safe, violations = validate_sandbox_code(code)
        assert is_safe

    def test_validate_multiple_violations(self):
        code = textwrap.dedent("""\
            import subprocess
            import socket
            eval('1+1')
        """)
        is_safe, violations = validate_sandbox_code(code)
        assert not is_safe
        assert len(violations) >= 3


# ---------------------------------------------------------------------------
# execute_sandbox_code tests
# ---------------------------------------------------------------------------
class TestExecuteSandboxCode:
    """Tests for end-to-end subprocess execution."""

    @pytest.fixture
    def execution_folder(self, tmp_path):
        """Create a temp execution folder."""
        folder = tmp_path / "test_execution"
        folder.mkdir()
        return folder

    @pytest.fixture
    def workspace_root(self, tmp_path):
        """Create a temp workspace root."""
        ws = tmp_path / "osprey-workspace"
        ws.mkdir()
        (ws / "data").mkdir()
        return ws

    async def test_execute_simple_code(self, execution_folder, workspace_root):
        """Simple code executes successfully."""
        code = textwrap.dedent("""\
            x = [1, 2, 3]
            print(sum(x))
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success
        assert result.execution_time_seconds is not None
        assert "6" in result.stdout

    async def test_no_auto_capture_of_matplotlib_figures(self, execution_folder, workspace_root):
        """Matplotlib code without save_artifact() produces no artifacts."""
        code = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot([1, 2, 3], [4, 5, 6])
            plt.title('Test Plot')
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success
        # No auto-capture: artifacts should be empty
        assert result.artifacts == []

    async def test_save_artifact_produces_artifacts(self, execution_folder, workspace_root):
        """save_artifact() call produces artifacts in the manifest."""
        code = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [4, 5, 6])
            save_artifact(fig, "Test Plot")
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success
        assert len(result.artifacts) == 1
        assert result.artifacts[0]["title"] == "Test Plot"
        assert result.artifacts[0]["artifact_type"] == "plot_png"
        assert result.artifacts[0]["path"].exists()

    async def test_execute_timeout(self, execution_folder, workspace_root):
        code = textwrap.dedent("""\
            import time
            time.sleep(60)
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=2,
            )

        assert not result.success
        assert "timed out" in result.error_message

    async def test_execute_validation_failure_returns_error(self, execution_folder, workspace_root):
        code = "import subprocess\nsubprocess.run(['ls'])"

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(code=code, execution_folder=execution_folder)

        assert not result.success
        assert "validation failed" in result.error_message.lower()
        # No script should have been written
        assert not (execution_folder / "wrapped_script.py").exists()

    async def test_execute_3d_matplotlib_plot(self, execution_folder, workspace_root):
        """3D matplotlib scatter plot executes and produces an artifact.

        Regression test: mpl_toolkits.mplot3d must be importable inside the
        sandbox so the data-visualizer agent can create 3D static plots.
        """
        code = textwrap.dedent("""\
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = np.random.rand(50)
            y = np.random.rand(50)
            z = np.random.rand(50)
            ax.scatter(x, y, z)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            save_artifact(fig, "3D Scatter Plot")
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success, f"3D matplotlib plot failed: {result.error_message}\n{result.stderr}"
        assert len(result.artifacts) == 1
        assert result.artifacts[0]["title"] == "3D Scatter Plot"
        assert result.artifacts[0]["artifact_type"] == "plot_png"
        assert result.artifacts[0]["path"].exists()

    async def test_execute_runtime_error(self, execution_folder, workspace_root):
        code = "x = 1 / 0"

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(code=code, execution_folder=execution_folder)

        assert not result.success
        assert result.error_message is not None

    async def test_stdout_captured(self, execution_folder, workspace_root):
        code = "print('hello from sandbox')"

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(code=code, execution_folder=execution_folder)

        assert result.success
        assert "hello from sandbox" in result.stdout


# ---------------------------------------------------------------------------
# Sandbox tests
# ---------------------------------------------------------------------------
class TestSandboxedOpen:
    """Tests for the open() filesystem sandbox."""

    @pytest.fixture
    def execution_folder(self, tmp_path):
        folder = tmp_path / "test_execution"
        folder.mkdir()
        return folder

    @pytest.fixture
    def workspace_root(self, tmp_path):
        ws = tmp_path / "osprey-workspace"
        ws.mkdir()
        (ws / "data").mkdir()
        return ws

    async def test_sandboxed_open_workspace_allowed(self, execution_folder, workspace_root):
        # Create a test file in the workspace
        test_file = workspace_root / "data" / "test.csv"
        test_file.write_text("a,b\n1,2\n")

        code = f"""\
data_path = r"{test_file}"
with open(data_path) as f:
    content = f.read()
print(content)
"""

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(code=code, execution_folder=execution_folder)

        assert result.success
        assert "a,b" in result.stdout

    async def test_sandboxed_open_outside_blocked(self, execution_folder, workspace_root):
        code = """\
try:
    with open('/etc/passwd') as f:
        content = f.read()
    print('SHOULD NOT REACH HERE')
except PermissionError:
    print('BLOCKED')
"""

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(code=code, execution_folder=execution_folder)

        assert result.success
        assert "BLOCKED" in result.stdout


# ---------------------------------------------------------------------------
# Artifact collection tests
# ---------------------------------------------------------------------------
class TestCollection:
    """Tests for artifact collection helpers."""

    def test_artifact_collection(self, tmp_path):
        folder = tmp_path / "exec"
        folder.mkdir()
        (folder / "artifacts").mkdir()

        # Write manifest and artifact file
        manifest = [
            {
                "filename": "abc_chart.html",
                "title": "Chart",
                "description": "A chart",
                "artifact_type": "html",
                "mime_type": "text/html",
            }
        ]
        (folder / "artifacts" / "manifest.json").write_text(json.dumps(manifest))
        (folder / "artifacts" / "abc_chart.html").write_text("<html></html>")

        artifacts = _collect_artifacts(folder)
        assert len(artifacts) == 1
        assert artifacts[0]["title"] == "Chart"
        assert artifacts[0]["path"].exists()

    def test_artifact_collection_missing_file(self, tmp_path):
        folder = tmp_path / "exec"
        folder.mkdir()
        (folder / "artifacts").mkdir()

        manifest = [{"filename": "missing.txt", "title": "Missing"}]
        (folder / "artifacts" / "manifest.json").write_text(json.dumps(manifest))

        artifacts = _collect_artifacts(folder)
        assert artifacts == []

    def test_artifact_collection_no_manifest(self, tmp_path):
        folder = tmp_path / "exec"
        folder.mkdir()

        artifacts = _collect_artifacts(folder)
        assert artifacts == []


class TestCreateExecutionFolder:
    """Tests for create_sandbox_execution_folder."""

    def test_create_execution_folder(self, tmp_path):
        ws = tmp_path / "osprey-workspace"
        ws.mkdir()

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=ws,
        ):
            folder = create_sandbox_execution_folder()

        assert folder.exists()
        assert "sandbox_executions" in str(folder)
        # No figures/ subdirectory should be created
        assert not (folder / "figures").exists()


class TestSandboxExecutionResult:
    """Tests for the SandboxExecutionResult dataclass."""

    def test_no_figures_field(self):
        """Verify SandboxExecutionResult has no figures field."""
        result = SandboxExecutionResult(
            success=True,
            stdout="",
            stderr="",
        )
        assert not hasattr(result, "figures")
        assert result.artifacts == []

    def test_result_with_artifacts(self):
        result = SandboxExecutionResult(
            success=True,
            stdout="output",
            stderr="",
            artifacts=[{"path": Path("/tmp/test.html"), "title": "Test"}],
            execution_time_seconds=1.5,
        )
        assert result.success
        assert len(result.artifacts) == 1
        assert result.execution_time_seconds == 1.5


class TestSaveArtifactBokehSupport:
    """Tests for Bokeh detection in save_artifact() within the sandbox wrapper."""

    @pytest.fixture
    def execution_folder(self, tmp_path):
        folder = tmp_path / "test_execution"
        folder.mkdir()
        return folder

    @pytest.fixture
    def workspace_root(self, tmp_path):
        ws = tmp_path / "osprey-workspace"
        ws.mkdir()
        (ws / "data").mkdir()
        return ws

    async def test_bokeh_import_allowed_in_sandbox(self, execution_folder, workspace_root):
        """Verify Bokeh can import in sandbox (needs venv/share data files)."""
        code = textwrap.dedent("""\
            from bokeh.plotting import figure
            from bokeh.layouts import column
            p = figure(title="test", width=400, height=300)
            p.line([1, 2, 3], [4, 5, 6])
            save_artifact(column(p), "Bokeh Test")
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success, f"Bokeh import failed: {result.error_message}\n{result.stderr}"
        assert len(result.artifacts) == 1
        assert result.artifacts[0]["artifact_type"] == "dashboard_html"

    async def test_save_artifact_plotly_figure(self, execution_folder, workspace_root):
        """Verify save_artifact() handles Plotly figures."""
        code = textwrap.dedent("""\
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Scatter(x=[1,2,3], y=[4,5,6]))
            save_artifact(fig, "Plotly Test")
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success
        assert len(result.artifacts) == 1
        assert result.artifacts[0]["artifact_type"] == "plot_html"
        assert result.artifacts[0]["path"].suffix == ".html"

    async def test_save_artifact_string(self, execution_folder, workspace_root):
        """Verify save_artifact() handles string content."""
        code = textwrap.dedent("""\
            save_artifact("# Hello World\\nSome markdown content", "String Test")
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success
        assert len(result.artifacts) == 1
        assert result.artifacts[0]["artifact_type"] == "markdown"

    async def test_save_artifact_dict(self, execution_folder, workspace_root):
        """Verify save_artifact() handles dict content."""
        code = textwrap.dedent("""\
            save_artifact({"key": "value", "count": 42}, "Dict Test")
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success
        assert len(result.artifacts) == 1
        assert result.artifacts[0]["artifact_type"] == "json"

    async def test_save_artifact_pandas_dataframe(self, execution_folder, workspace_root):
        """Verify save_artifact() handles pandas DataFrames."""
        code = textwrap.dedent("""\
            import pandas as pd
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            save_artifact(df, "DataFrame Test")
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success
        assert len(result.artifacts) == 1
        assert result.artifacts[0]["artifact_type"] == "table_html"

    async def test_save_artifact_bytes(self, execution_folder, workspace_root):
        """Verify save_artifact() handles bytes content."""
        code = textwrap.dedent("""\
            save_artifact(b"binary content here", "Bytes Test")
        """)

        with patch(
            "osprey.mcp_server.common.resolve_workspace_root",
            return_value=workspace_root,
        ):
            result = await execute_sandbox_code(
                code=code,
                execution_folder=execution_folder,
                timeout=30,
            )

        assert result.success
        assert len(result.artifacts) == 1
        assert result.artifacts[0]["artifact_type"] == "binary"
