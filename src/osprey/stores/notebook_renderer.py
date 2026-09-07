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


# nbconvert's default HTML template links these assets from public CDNs. The
# gallery serves a rendered notebook inside a sandboxed iframe, and a render
# whose assets are fetched at view time is a render whose appearance depends on
# the viewer's network: on a site that blocks external hosts those loads fail
# and the notebook shows broken. So this renderer is deliberately
# self-contained — every one of these is an nbconvert HTMLExporter traitlet,
# and blanking them yields HTML that depends only on the template's inlined
# CSS.
#
# The policy is a choice, not a statement about where OSPREY runs: it is
# applied unconditionally, on a connected deployment as much as an isolated
# one. The cost it pays everywhere is that inline LaTeX (MathJax), interactive
# widgets and Mermaid diagrams do not render. Making it conditional would mean
# reading the deployment's offline posture here AND carrying the mode in the
# cache key below, since the two renders are different documents.
_EXTERNAL_ASSET_TRAITS = (
    "mathjax_url",
    "require_js_url",
    "jquery_url",
    "jupyter_widgets_base_url",
    "widget_renderer_url",
    "mermaid_js_url",
    "mermaid_layout_elk_js_url",
)


def render_notebook_to_html(ipynb_path: Path) -> str:
    """Render a .ipynb file to self-contained HTML using nbconvert.

    The output references no external resources, so what it looks like does not
    depend on what the viewer's network can reach — see
    :data:`_EXTERNAL_ASSET_TRAITS` for what that costs and why it is
    unconditional.

    Args:
        ipynb_path: Path to the .ipynb file.

    Returns:
        HTML string of the rendered notebook.
    """
    from nbconvert import HTMLExporter

    with open(ipynb_path) as f:
        nb = nbformat.read(f, as_version=4)

    exporter = HTMLExporter(embed_images=True)
    for trait in _EXTERNAL_ASSET_TRAITS:
        if exporter.has_trait(trait):
            setattr(exporter, trait, "")

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
