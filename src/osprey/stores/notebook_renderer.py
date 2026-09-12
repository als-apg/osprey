"""Notebook rendering utilities for OSPREY MCP tools.

Creates Jupyter notebooks from execute tool code and results,
and renders them to HTML for the Artifact Gallery.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import nbformat

from osprey.utils.config import to_facility_iso

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
    timestamp = to_facility_iso(datetime.now(UTC))
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


# nbconvert's default HTML template links these assets from public CDNs, and
# each is an HTMLExporter traitlet that blanking removes from the output.
#
# Which of the two renders is right is the deployment's ``offline`` posture,
# not a property of the gallery. An OFFLINE render depends on the template's
# inlined CSS alone: nothing is fetched when a viewer opens it, which is the
# only render that works where external hosts are unreachable, and it costs
# inline LaTeX (MathJax), interactive widgets and Mermaid diagrams. A CONNECTED
# render keeps all three. The two are different documents, so the cache below
# names them differently.
_EXTERNAL_ASSET_TRAITS = (
    "mathjax_url",
    "require_js_url",
    "jquery_url",
    "jupyter_widgets_base_url",
    "widget_renderer_url",
    "mermaid_js_url",
    "mermaid_layout_elk_js_url",
)


def _resolve_offline(offline: bool | None) -> bool:
    """The render mode, from the caller or from the deployment's posture."""
    if offline is not None:
        return offline

    from osprey.interfaces.vendor import is_offline

    return is_offline()


def render_notebook_to_html(ipynb_path: Path, *, offline: bool | None = None) -> str:
    """Render a .ipynb file to HTML using nbconvert.

    Args:
        ipynb_path: Path to the .ipynb file.
        offline: Whether to produce a self-contained render. ``None`` reads the
            deployment's own posture, which is what every caller but a test
            wants — see :data:`_EXTERNAL_ASSET_TRAITS` for what each mode
            gives up.

    Returns:
        HTML string of the rendered notebook.
    """
    from nbconvert import HTMLExporter

    with open(ipynb_path) as f:
        nb = nbformat.read(f, as_version=4)

    exporter = HTMLExporter(embed_images=True)
    if _resolve_offline(offline):
        for trait in _EXTERNAL_ASSET_TRAITS:
            if exporter.has_trait(trait):
                setattr(exporter, trait, "")

    html, _ = exporter.from_notebook_node(nb)
    return html


def get_or_render_html(ipynb_path: Path, cache_dir: Path | None = None) -> tuple[str, Path]:
    """Render notebook to HTML with filesystem caching.

    If a cached render for the deployment's current mode exists and is newer
    than the ``.ipynb`` file, the cache is returned. Otherwise the notebook is
    re-rendered and the cache is updated.

    The mode is part of the cache filename because the two modes are different
    documents: without it, flipping the deployment's posture would keep serving
    the render the other mode produced for as long as the notebook is untouched.

    Args:
        ipynb_path: Path to the .ipynb file.
        cache_dir: Directory for cached HTML files. Defaults to the
            same directory as the notebook.

    Returns:
        (html_string, html_file_path)
    """
    cache_dir = cache_dir or ipynb_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    offline = _resolve_offline(None)
    suffix = ".offline.html" if offline else ".html"
    html_path = cache_dir / f"{ipynb_path.stem}_rendered{suffix}"

    # Use cache if it exists and is newer than the notebook
    if html_path.exists():
        nb_mtime = ipynb_path.stat().st_mtime
        html_mtime = html_path.stat().st_mtime
        if html_mtime >= nb_mtime:
            return html_path.read_text(), html_path

    # Render and cache
    html = render_notebook_to_html(ipynb_path, offline=offline)
    html_path.write_text(html)
    return html, html_path
