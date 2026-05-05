"""Tests for the create_document MCP tool."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from tests.mcp_server.conftest import extract_response_dict

import pytest

from osprey.stores.artifact_store import get_artifact_store


class TestCreateDocument:
    """Tests for create_document tool."""

    @pytest.fixture
    def tool_fn(self):
        """Get the raw tool function."""
        from osprey.mcp_server.workspace.tools.create_document import create_document
        from tests.mcp_server.conftest import get_tool_fn

        return get_tool_fn(create_document)

    @pytest.fixture
    def simple_latex(self):
        """Minimal valid LaTeX source."""
        return r"""\documentclass{article}
\begin{document}
Hello, World!
\end{document}
"""

    @pytest.fixture
    def beamer_latex(self):
        """Minimal Beamer slide deck."""
        return r"""\documentclass{beamer}
\begin{document}
\begin{frame}{Title Slide}
Welcome to the presentation.
\end{frame}
\end{document}
"""

    async def test_empty_source_returns_error(self, tool_fn):
        result = extract_response_dict(await tool_fn(latex_source="", title="test"))
        assert result["error"] is True
        assert result["error_type"] == "validation_error"
        assert "No LaTeX source" in result["error_message"]

    async def test_pdflatex_not_installed(self, tool_fn, simple_latex):
        with patch("shutil.which", return_value=None):
            result = extract_response_dict(await tool_fn(latex_source=simple_latex, title="Test Doc"))

        assert result["error"] is True
        assert result["error_type"] == "dependency_missing"
        assert "pdflatex" in result["error_message"]
        assert any("brew" in s or "apt" in s for s in result["suggestions"])

    async def test_compilation_failure(self, tool_fn):
        bad_latex = r"\documentclass{article}\begin{document}\badcommand\end{document}"

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = "! Undefined control sequence.\n\\badcommand"
            result.stderr = ""
            return result

        with (
            patch("shutil.which", return_value="/usr/bin/pdflatex"),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = extract_response_dict(await tool_fn(latex_source=bad_latex, title="Bad Doc"))

        assert result["error"] is True
        assert result["error_type"] == "compilation_error"

    async def test_compilation_timeout(self, tool_fn, simple_latex):
        import subprocess

        with (
            patch("shutil.which", return_value="/usr/bin/pdflatex"),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("pdflatex", 60)),
        ):
            result = extract_response_dict(await tool_fn(latex_source=simple_latex, title="Timeout Doc"))

        assert result["error"] is True
        assert result["error_type"] == "execution_error"
        assert "timed out" in result["error_message"]

    async def test_successful_compilation(self, tool_fn, simple_latex, tmp_path):
        """Test successful PDF generation with mocked pdflatex."""
        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            # Create a fake PDF on the first pass
            cwd = Path(kwargs.get("cwd", "."))
            pdf_path = cwd / "document.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with (
            patch("shutil.which", return_value="/usr/bin/pdflatex"),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = extract_response_dict(
                await tool_fn(
                    latex_source=simple_latex,
                    title="Test Report",
                    description="A test report",
                )
            )

        assert result["status"] == "success"
        # Should have 2 artifacts: PDF + .tex source
        assert len(result["artifact_ids"]) == 2
        assert result["pdf_artifact_id"] is not None
        # pdflatex called 2 times (2 passes)
        assert call_count == 2

    async def test_pdf_and_source_artifacts(self, tool_fn, simple_latex):
        """Verify both PDF and .tex source are saved as artifacts."""

        def mock_run(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", "."))
            (cwd / "document.pdf").write_bytes(b"%PDF-1.4 content")
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with (
            patch("shutil.which", return_value="/usr/bin/pdflatex"),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = extract_response_dict(await tool_fn(latex_source=simple_latex, title="Artifacts Test"))

        assert result["status"] == "success"
        store = get_artifact_store()

        # PDF artifact
        pdf_id = result["artifact_ids"][0]
        pdf_entry = store.get_entry(pdf_id)
        assert pdf_entry is not None
        assert pdf_entry.mime_type == "application/pdf"
        assert pdf_entry.artifact_type == "file"

        # Source artifact
        tex_id = result["artifact_ids"][1]
        tex_entry = store.get_entry(tex_id)
        assert tex_entry is not None
        assert tex_entry.mime_type == "application/x-tex"
        assert tex_entry.artifact_type == "text"

    async def test_artifact_ids_resolved_to_build_dir(self, tool_fn, simple_latex):
        """Verify referenced artifact figures are copied to the build directory."""
        # Create a test artifact
        store = get_artifact_store()
        art_entry = store.save_file(
            file_content=b"fake-png-data",
            filename="beam_plot.png",
            artifact_type="image",
            title="Beam Plot",
            description="A beam plot",
            mime_type="image/png",
            tool_source="test",
        )

        captured_cwd = None

        def mock_run(cmd, **kwargs):
            nonlocal captured_cwd
            captured_cwd = kwargs.get("cwd")
            cwd = Path(captured_cwd)
            (cwd / "document.pdf").write_bytes(b"%PDF-1.4 content")
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        latex_with_fig = (
            r"\documentclass{article}"
            r"\usepackage{graphicx}"
            r"\begin{document}"
            r"\includegraphics{beam_plot.png}"
            r"\end{document}"
        )

        with (
            patch("shutil.which", return_value="/usr/bin/pdflatex"),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = extract_response_dict(
                await tool_fn(
                    latex_source=latex_with_fig,
                    title="Fig Test",
                    artifact_ids=[art_entry.id],
                )
            )

        assert result["status"] == "success"
        # The figure should have been copied to build dir
        assert result["figures_included"] == 1

    async def test_missing_artifact_id_skipped(self, tool_fn, simple_latex):
        """Non-existent artifact IDs are skipped without failing."""

        def mock_run(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", "."))
            (cwd / "document.pdf").write_bytes(b"%PDF-1.4 content")
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with (
            patch("shutil.which", return_value="/usr/bin/pdflatex"),
            patch("subprocess.run", side_effect=mock_run),
        ):
            result = extract_response_dict(
                await tool_fn(
                    latex_source=simple_latex,
                    title="Missing Art",
                    artifact_ids=["nonexistent123"],
                )
            )

        # Should still succeed — missing artifact is just skipped
        assert result["status"] == "success"
        assert result["figures_included"] == 0
