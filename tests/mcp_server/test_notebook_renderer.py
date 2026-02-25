"""Tests for the notebook renderer utility.

Covers:
  - create_notebook_from_code: valid notebook structure
  - render_notebook_to_html: produces HTML output
  - get_or_render_html: caching behavior
"""

import time

import nbformat
import pytest

from osprey.mcp_server.notebook_renderer import (
    create_notebook_from_code,
    get_or_render_html,
    render_notebook_to_html,
)


class TestCreateNotebookFromCode:
    """Tests for create_notebook_from_code()."""

    @pytest.mark.unit
    def test_creates_valid_notebook(self):
        """Notebook has correct nbformat version and is valid."""
        nb = create_notebook_from_code(
            code="print('hello')",
            description="Test execution",
        )
        assert nb.nbformat == 4
        nbformat.validate(nb)

    @pytest.mark.unit
    def test_includes_header_code_cells(self):
        """Notebook includes a markdown header and a code cell."""
        nb = create_notebook_from_code(
            code="x = 1 + 2\nprint(x)",
            description="Addition test",
        )
        assert len(nb.cells) >= 2
        assert nb.cells[0].cell_type == "markdown"
        assert "Addition test" in nb.cells[0].source
        assert nb.cells[1].cell_type == "code"
        assert "x = 1 + 2" in nb.cells[1].source

    @pytest.mark.unit
    def test_includes_output_cell_with_stdout(self):
        """When stdout is provided, a results cell is added."""
        nb = create_notebook_from_code(
            code="print(42)",
            description="Print test",
            stdout="42\n",
        )
        assert len(nb.cells) == 3
        results_cell = nb.cells[2]
        assert results_cell.cell_type == "markdown"
        assert "42" in results_cell.source
        assert "Output" in results_cell.source

    @pytest.mark.unit
    def test_includes_error_in_output_cell(self):
        """When stderr is provided, errors section appears in results cell."""
        nb = create_notebook_from_code(
            code="1/0",
            description="Error test",
            stderr="ZeroDivisionError: division by zero",
        )
        results_cell = nb.cells[2]
        assert "Errors" in results_cell.source
        assert "ZeroDivisionError" in results_cell.source

    @pytest.mark.unit
    def test_header_shows_status(self):
        """Header cell shows Success when no stderr, Error when stderr present."""
        nb_success = create_notebook_from_code(code="pass", description="ok")
        assert "Success" in nb_success.cells[0].source

        nb_error = create_notebook_from_code(code="pass", description="fail", stderr="err")
        assert "Error" in nb_error.cells[0].source

    @pytest.mark.unit
    def test_no_output_cell_when_empty(self):
        """No results cell when both stdout and stderr are empty."""
        nb = create_notebook_from_code(code="x = 1", description="silent")
        assert len(nb.cells) == 2  # header + code only


class TestRenderNotebookToHtml:
    """Tests for render_notebook_to_html()."""

    @pytest.mark.unit
    def test_renders_html_containing_code(self, tmp_path):
        """Rendered HTML contains the original code."""
        nb = create_notebook_from_code(
            code="print('HELLO_WORLD_UNIQUE_MARKER')",
            description="Render test",
        )
        nb_path = tmp_path / "test.ipynb"
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        html = render_notebook_to_html(nb_path)
        assert "HELLO_WORLD_UNIQUE_MARKER" in html
        assert "<html" in html.lower()

    @pytest.mark.unit
    def test_renders_html_with_output(self, tmp_path):
        """Rendered HTML includes stdout content."""
        nb = create_notebook_from_code(
            code="print('UNIQUE_OUTPUT_MARKER')",
            description="Output render",
            stdout="UNIQUE_OUTPUT_MARKER\n",
        )
        nb_path = tmp_path / "test_output.ipynb"
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        html = render_notebook_to_html(nb_path)
        assert "UNIQUE_OUTPUT_MARKER" in html


class TestGetOrRenderHtml:
    """Tests for get_or_render_html() caching behavior."""

    @pytest.mark.unit
    def test_creates_cache_file(self, tmp_path):
        """First call creates the cached HTML file."""
        nb = create_notebook_from_code(code="CACHE_TEST_MARKER_ABC123", description="Cache test")
        nb_path = tmp_path / "cached.ipynb"
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        cache_dir = tmp_path / "cache"
        html, html_path = get_or_render_html(nb_path, cache_dir=cache_dir)

        assert html_path.exists()
        assert "CACHE_TEST_MARKER_ABC123" in html
        assert html_path.name == "cached_rendered.html"

    @pytest.mark.unit
    def test_uses_cache_on_second_call(self, tmp_path):
        """Second call returns cached HTML without re-rendering."""
        nb = create_notebook_from_code(code="y = 2", description="Cache hit test")
        nb_path = tmp_path / "cached2.ipynb"
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        cache_dir = tmp_path / "cache2"
        _, html_path = get_or_render_html(nb_path, cache_dir=cache_dir)
        first_mtime = html_path.stat().st_mtime

        # Small delay to ensure mtime would differ if re-rendered
        time.sleep(0.05)

        _, html_path2 = get_or_render_html(nb_path, cache_dir=cache_dir)
        assert html_path2.stat().st_mtime == first_mtime

    @pytest.mark.unit
    def test_invalidates_stale_cache(self, tmp_path):
        """Cache is regenerated when notebook is newer than cached HTML."""
        nb = create_notebook_from_code(code="STALE_ORIGINAL_MARKER", description="Stale test")
        nb_path = tmp_path / "stale.ipynb"
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

        cache_dir = tmp_path / "cache3"
        _, html_path = get_or_render_html(nb_path, cache_dir=cache_dir)
        first_mtime = html_path.stat().st_mtime

        # Wait, then update the notebook (newer mtime)
        time.sleep(0.05)
        nb2 = create_notebook_from_code(code="STALE_UPDATED_MARKER", description="Updated")
        with open(nb_path, "w") as f:
            nbformat.write(nb2, f)

        html, html_path2 = get_or_render_html(nb_path, cache_dir=cache_dir)
        assert html_path2.stat().st_mtime > first_mtime
        assert "STALE_UPDATED_MARKER" in html
