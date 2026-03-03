"""Notebook rendering utilities for OSPREY MCP tools.

Creates Jupyter notebooks from execute tool code and results,
and renders them to HTML for the Artifact Gallery.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import nbformat

logger = logging.getLogger("osprey.stores.notebook_renderer")


def create_notebook_from_code(
    code: str,
    description: str,
    stdout: str = "",
    stderr: str = "",
) -> nbformat.NotebookNode:
    """Create a notebook from executed Python code and its output.

    Args:
        code: The Python source code that was executed.
        description: Human-readable description of the code.
        stdout: Captured stdout from execution.
        stderr: Captured stderr from execution.

    Returns:
        A NotebookNode ready for serialization.
    """
    cells = []

    # Header cell
    status = "Error" if stderr else "Success"
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = (
        f"# {description}\n\n"
        f"**Status:** {status}  \n"
        f"**Timestamp:** {timestamp}  \n"
        f"**Source:** `execute`\n"
    )
    cells.append(nbformat.v4.new_markdown_cell(header))

    # Code cell
    cells.append(nbformat.v4.new_code_cell(code))

    # Results cell (if there's output)
    if stdout or stderr:
        parts = []
        if stdout:
            parts.append(f"## Output\n\n```\n{stdout}\n```")
        if stderr:
            parts.append(f"## Errors\n\n```\n{stderr}\n```")
        cells.append(nbformat.v4.new_markdown_cell("\n\n".join(parts)))

    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells
    return notebook


def render_notebook_to_html(ipynb_path: Path) -> str:
    """Render a .ipynb file to HTML using nbconvert.

    Args:
        ipynb_path: Path to the .ipynb file.

    Returns:
        HTML string of the rendered notebook.
    """
    from nbconvert import HTMLExporter

    with open(ipynb_path) as f:
        nb = nbformat.read(f, as_version=4)

    exporter = HTMLExporter()
    html, _ = exporter.from_notebook_node(nb)
    return html


def get_or_render_html(ipynb_path: Path, cache_dir: Path | None = None) -> tuple[str, Path]:
    """Render notebook to HTML with filesystem caching.

    If a cached ``{stem}_rendered.html`` exists and is newer than the
    ``.ipynb`` file, the cache is returned.  Otherwise the notebook is
    re-rendered and the cache is updated.

    Args:
        ipynb_path: Path to the .ipynb file.
        cache_dir: Directory for cached HTML files. Defaults to the
            same directory as the notebook.

    Returns:
        (html_string, html_file_path)
    """
    cache_dir = cache_dir or ipynb_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    html_path = cache_dir / f"{ipynb_path.stem}_rendered.html"

    # Use cache if it exists and is newer than the notebook
    if html_path.exists():
        nb_mtime = ipynb_path.stat().st_mtime
        html_mtime = html_path.stat().st_mtime
        if html_mtime >= nb_mtime:
            return html_path.read_text(), html_path

    # Render and cache
    html = render_notebook_to_html(ipynb_path)
    html_path.write_text(html)
    return html, html_path
