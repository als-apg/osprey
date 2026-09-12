"""Tests for the notebook renderer utility.

Covers:
  - create_notebook_from_code: valid notebook structure
  - render_notebook_to_html: produces HTML output, offline-aware
  - get_or_render_html: caching behavior, one cache file per mode
"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import nbformat
import pytest

from osprey.stores.notebook_renderer import (
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


#: A URL the default nbconvert template links and an offline render cannot use.
#: MathJax is the one that costs the most when it is missing — inline LaTeX in
#: a notebook's markdown simply does not render.
MATHJAX_URL = "cdnjs.cloudflare.com/ajax/libs/mathjax"


class TestOfflineAwareRendering:
    """Which render each deployment posture gets.

    A render whose assets are fetched at view time depends on the viewer's
    network; a render with them blanked loses MathJax, interactive widgets and
    Mermaid. Neither is right everywhere, so the deployment's own ``offline``
    posture decides.
    """

    @staticmethod
    def _notebook(tmp_path, name: str = "assets"):
        nb = create_notebook_from_code(code="x = 1", description="Asset test")
        path = tmp_path / f"{name}.ipynb"
        with open(path, "w") as handle:
            nbformat.write(nb, handle)
        return path

    @pytest.mark.unit
    def test_an_offline_render_references_no_remote_asset(self, tmp_path):
        """Nothing to fetch: the isolated deployment's render is self-contained."""
        html = render_notebook_to_html(self._notebook(tmp_path), offline=True)

        assert MATHJAX_URL not in html

    @pytest.mark.unit
    def test_a_connected_render_keeps_the_exporters_own_assets(self, tmp_path):
        """A deployment that can reach them gets LaTeX, widgets and Mermaid."""
        html = render_notebook_to_html(self._notebook(tmp_path), offline=False)

        assert MATHJAX_URL in html

    @pytest.mark.unit
    def test_the_mode_defaults_to_the_deployments_posture(self, tmp_path, monkeypatch):
        """Callers that pass nothing get the posture the deployment declares."""
        notebook = self._notebook(tmp_path)

        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        assert MATHJAX_URL not in render_notebook_to_html(notebook)

        monkeypatch.setenv("OSPREY_OFFLINE", "0")
        assert MATHJAX_URL in render_notebook_to_html(notebook)


class TestGetOrRenderHtml:
    """Tests for get_or_render_html() caching behavior."""

    @pytest.mark.unit
    def test_creates_cache_file(self, tmp_path, monkeypatch):
        """First call creates the cached HTML file."""
        # The cache name carries the render mode, so the mode is pinned rather
        # than inherited from whatever posture the ambient shell declares.
        monkeypatch.setenv("OSPREY_OFFLINE", "0")
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

    @pytest.mark.unit
    def test_each_mode_caches_under_its_own_name(self, tmp_path, monkeypatch):
        """Flipping the posture re-renders instead of serving the other document.

        One cache filename for two different renders is a deployment that
        turns ``offline`` on, restarts, and keeps serving the render that
        fetches from the internet — for as long as the notebook is untouched.
        """
        nb = create_notebook_from_code(code="z = 3", description="Mode test")
        nb_path = tmp_path / "modes.ipynb"
        with open(nb_path, "w") as handle:
            nbformat.write(nb, handle)
        cache_dir = tmp_path / "cache_modes"

        monkeypatch.setenv("OSPREY_OFFLINE", "0")
        online_html, online_path = get_or_render_html(nb_path, cache_dir=cache_dir)
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        offline_html, offline_path = get_or_render_html(nb_path, cache_dir=cache_dir)

        assert online_path != offline_path
        assert online_path.name == "modes_rendered.html"
        assert offline_path.name == "modes_rendered.offline.html"
        assert MATHJAX_URL in online_html
        assert MATHJAX_URL not in offline_html

    @pytest.mark.unit
    def test_a_stale_cache_is_regenerated_in_the_offline_mode_too(self, tmp_path, monkeypatch):
        """The staleness rule is per mode, not only for the connected one."""
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        nb = create_notebook_from_code(code="OFFLINE_ORIGINAL_MARKER", description="Stale")
        nb_path = tmp_path / "offline_stale.ipynb"
        with open(nb_path, "w") as handle:
            nbformat.write(nb, handle)
        cache_dir = tmp_path / "cache_offline_stale"

        _, html_path = get_or_render_html(nb_path, cache_dir=cache_dir)
        first_mtime = html_path.stat().st_mtime

        time.sleep(0.05)
        nb2 = create_notebook_from_code(code="OFFLINE_UPDATED_MARKER", description="Updated")
        with open(nb_path, "w") as handle:
            nbformat.write(nb2, handle)

        html, html_path2 = get_or_render_html(nb_path, cache_dir=cache_dir)

        assert html_path2 == html_path
        assert html_path2.stat().st_mtime > first_mtime
        assert "OFFLINE_UPDATED_MARKER" in html


TOKYO = ZoneInfo("Asia/Tokyo")  # UTC+9, no DST


@pytest.mark.unit
def test_header_timestamp_is_in_the_facility_zone(monkeypatch):
    """The header an operator opens carries the facility offset, not a UTC literal."""
    monkeypatch.setattr(
        "osprey.utils.config.get_facility_timezone",
        lambda: TOKYO,
    )
    nb = create_notebook_from_code(code="print(1)", description="Zone test")
    line = next(ln for ln in nb.cells[0].source.splitlines() if ln.startswith("**Timestamp:**"))
    stamp = line.removeprefix("**Timestamp:**").strip()

    parsed = datetime.fromisoformat(stamp)
    assert parsed.utcoffset().total_seconds() == 9 * 3600
    assert "UTC" not in stamp
