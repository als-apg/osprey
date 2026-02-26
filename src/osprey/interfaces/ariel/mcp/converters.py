"""Artifact-to-attachment converters for ARIEL logbook entries.

Maps MIME types to async converter functions that produce logbook-friendly
formats (PNG for rendered content, passthrough for images/PDFs).

Each converter is ``async (source_path, output_dir) -> Path``:
  - Passthrough converters return *source_path* unchanged.
  - Rendering converters write a PNG into *output_dir* and return its path.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

logger = logging.getLogger("osprey.interfaces.ariel.mcp.converters")

ConverterFn = Callable[[Path, Path], Coroutine[Any, Any, Path]]


# ---------------------------------------------------------------------------
# HTML template wrapper
# ---------------------------------------------------------------------------


def _wrap_in_html(title: str, body_html: str) -> str:
    """Wrap content in a self-contained styled HTML page for rendering to PNG."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    margin: 40px;
    line-height: 1.6;
    color: #1a1a1a;
    background: #ffffff;
    max-width: 1100px;
  }}
  h1, h2, h3 {{ color: #2c3e50; }}
  pre {{
    background: #f4f4f4;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.45;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  }}
  code {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 13px;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
  }}
  th, td {{
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
  }}
  th {{ background: #f0f0f0; }}
  img {{ max-width: 100%; }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Converter functions
# ---------------------------------------------------------------------------


async def passthrough(source: Path, output_dir: Path) -> Path:
    """Return the source path unchanged (images, PDFs, binaries)."""
    return source


async def html_to_png(source: Path, output_dir: Path) -> Path:
    """Render an HTML file to PNG via Playwright."""
    from osprey.mcp_server.export.converter import convert_html_to_image

    png_path = output_dir / f"{source.stem}.png"
    await convert_html_to_image(source, png_path)
    return png_path


async def markdown_to_png(source: Path, output_dir: Path) -> Path:
    """Render a Markdown file to PNG: markdown -> HTML -> Playwright screenshot."""
    import markdown as md

    from osprey.mcp_server.export.converter import convert_html_to_image

    text = source.read_text(encoding="utf-8")
    body = md.markdown(text, extensions=["tables", "fenced_code"])
    html = _wrap_in_html(source.stem, body)

    html_path = output_dir / f"{source.stem}.html"
    html_path.write_text(html, encoding="utf-8")

    png_path = output_dir / f"{source.stem}.png"
    await convert_html_to_image(html_path, png_path)
    return png_path


async def notebook_to_png(source: Path, output_dir: Path) -> Path:
    """Render a Jupyter notebook to PNG: nbconvert -> HTML -> Playwright screenshot."""
    from osprey.mcp_server.export.converter import convert_html_to_image
    from osprey.mcp_server.notebook_renderer import render_notebook_to_html

    html = render_notebook_to_html(source)
    html_path = output_dir / f"{source.stem}.html"
    html_path.write_text(html, encoding="utf-8")

    png_path = output_dir / f"{source.stem}.png"
    await convert_html_to_image(html_path, png_path)
    return png_path


async def json_to_png(source: Path, output_dir: Path) -> Path:
    """Render a JSON file to PNG: pretty-print -> styled HTML -> Playwright screenshot."""
    from osprey.mcp_server.export.converter import convert_html_to_image

    raw = source.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        pretty = raw

    body = f"<h2>{source.name}</h2>\n<pre>{pretty}</pre>"
    html = _wrap_in_html(source.stem, body)

    html_path = output_dir / f"{source.stem}.html"
    html_path.write_text(html, encoding="utf-8")

    png_path = output_dir / f"{source.stem}.png"
    await convert_html_to_image(html_path, png_path)
    return png_path


async def text_to_png(source: Path, output_dir: Path) -> Path:
    """Render a text file to PNG: wrap in styled <pre> -> HTML -> Playwright screenshot."""
    from osprey.mcp_server.export.converter import convert_html_to_image

    text = source.read_text(encoding="utf-8")
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body = f"<h2>{source.name}</h2>\n<pre>{escaped}</pre>"
    html = _wrap_in_html(source.stem, body)

    html_path = output_dir / f"{source.stem}.html"
    html_path.write_text(html, encoding="utf-8")

    png_path = output_dir / f"{source.stem}.png"
    await convert_html_to_image(html_path, png_path)
    return png_path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONVERTER_REGISTRY: dict[str, ConverterFn] = {
    # Images — passthrough
    "image/png": passthrough,
    "image/jpeg": passthrough,
    "image/gif": passthrough,
    "image/svg+xml": passthrough,
    "image/webp": passthrough,
    # Documents — passthrough
    "application/pdf": passthrough,
    "application/octet-stream": passthrough,
    # HTML — render to PNG
    "text/html": html_to_png,
    # Markdown — render to PNG
    "text/markdown": markdown_to_png,
    "text/x-markdown": markdown_to_png,
    # Notebooks — render to PNG
    "application/x-ipynb+json": notebook_to_png,
    # JSON — render to PNG
    "application/json": json_to_png,
    # Text variants — render to PNG
    "text/plain": text_to_png,
    "text/x-python": text_to_png,
    "application/x-tex": text_to_png,
}


def get_converter(mime_type: str) -> ConverterFn:
    """Look up the converter for a MIME type, falling back to text_to_png."""
    if mime_type in CONVERTER_REGISTRY:
        return CONVERTER_REGISTRY[mime_type]

    # Fallback: any image/* type passes through
    if mime_type.startswith("image/"):
        return passthrough

    logger.debug("No converter for MIME type '%s', falling back to text_to_png", mime_type)
    return text_to_png
