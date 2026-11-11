"""The injected snippets behind the content strip and the table font.

The gallery draws a content strip over a ``plot_html`` or ``table_html``
frame. The snippet the server injects into every plot page
(``_RESPONSIVE_PLOTLY``) runs the strip's commands through
``static/js/plot-commands.js`` and hides Plotly's modebar while the page is
framed. Both the plot and the table snippet tell the gallery while the
pointer is over the frame (``static/js/frame-pointer.js``), since a sandboxed
frame never gives its owner ``:hover``. The modules' behaviour is pinned by
``plot-commands.test.mjs`` and ``frame-pointer.test.mjs``. The table snippet
also sets the body font token, so a pandas table no longer falls back to the
browser's serif.
"""

from __future__ import annotations

import re

import pytest

from osprey.interfaces.artifacts import inline_assets
from osprey.interfaces.artifacts.app import _RESPONSIVE_PLOTLY, _RESPONSIVE_TABLE_HTML

PLOT_COMMANDS_MODULE = "/static/js/plot-commands.js"
FRAME_POINTER_MODULE = "/static/js/frame-pointer.js"


def _style_outside_print(snippet: str) -> str:
    """The snippet's stylesheet with its ``@media print`` block taken out."""
    style = re.search(r"<style>(.*?)</style>", snippet, re.DOTALL)
    assert style is not None
    return re.sub(r"@media print\s*\{.*?\}\s*\}", "", style.group(1), flags=re.DOTALL)


def _save(store, artifact_type: str, content: bytes):
    return store.save_file(
        file_content=content,
        filename=f"{artifact_type}.html",
        artifact_type=artifact_type,
        title=artifact_type,
        description="",
        mime_type="text/html",
        tool_source="test",
    )


def test_the_plot_snippet_hides_the_modebar_only_in_a_frame():
    css = _style_outside_print(_RESPONSIVE_PLOTLY)
    assert re.search(r"html\[data-osprey-framed\] \.modebar-container", css)
    assert re.search(r"html\[data-osprey-framed\] \.modebar \{ display: none !important; \}", css)
    # Unframed (a plot opened on its own), nothing hides the bar outside print.
    unscoped = re.findall(r"(?:^|[,}\n])\s*\.modebar(?:-container)?\b", css)
    assert unscoped == []


def test_the_plot_snippet_takes_the_strip_commands():
    assert f"from '{PLOT_COMMANDS_MODULE}'" in _RESPONSIVE_PLOTLY
    assert "followPlotCommands(window, plots);" in _RESPONSIVE_PLOTLY
    # The opened view is recorded once the plots are drawn, for Reset axes.
    init = _RESPONSIVE_PLOTLY.split("function initAll()", 1)[1].split("}", 1)[0]
    assert "rememberViews(plots());" in init


@pytest.mark.parametrize("snippet", [_RESPONSIVE_PLOTLY, _RESPONSIVE_TABLE_HTML])
def test_plot_and_table_pages_report_the_pointer(snippet):
    assert f"import {{ reportPointer }} from '{FRAME_POINTER_MODULE}';" in snippet
    assert "reportPointer(window);" in snippet


@pytest.mark.parametrize(
    ("module", "names"),
    [
        (
            "plot-commands.js",
            [
                "osprey-artifact-command",
                "box-zoom",
                "pan",
                "zoom-in",
                "zoom-out",
                "autoscale",
                "reset-axes",
                "export-png",
            ],
        ),
        ("frame-pointer.js", ["osprey-artifact-pointer"]),
    ],
)
def test_the_page_modules_name_their_messages_and_import_nothing(module, names):
    source = (inline_assets.ARTIFACTS_STATIC / "js" / module).read_text()
    for name in names:
        assert f"'{name}'" in source
    # Inlined as a data: module, a page module can resolve no import of its own.
    assert not re.search(r"^\s*import\b", source, re.MULTILINE)


@pytest.mark.parametrize(
    ("artifact_type", "modules"),
    [
        ("plot_html", [PLOT_COMMANDS_MODULE, FRAME_POINTER_MODULE]),
        ("table_html", [FRAME_POINTER_MODULE]),
    ],
)
def test_a_served_page_carries_its_modules_inline(app_client, artifact_type, modules):
    page = b'<html><head></head><body><div class="plotly-graph-div"></div></body></html>'
    entry = _save(app_client.app.state.artifact_store, artifact_type, page)
    body = app_client.get(f"/files/{entry.id}/{entry.filename}").text
    for module in modules:
        assert inline_assets.module_data_url(module) in body
        assert module not in body


def test_a_served_plot_hides_the_modebar_in_a_frame(app_client):
    page = b'<html><head></head><body><div class="plotly-graph-div"></div></body></html>'
    entry = _save(app_client.app.state.artifact_store, "plot_html", page)
    body = app_client.get(f"/files/{entry.id}/{entry.filename}").text
    assert "html[data-osprey-framed] .modebar-container" in body


def test_the_table_snippet_sets_the_body_font_at_zero_specificity():
    assert ":where(body) { color: var(--text-primary); font-family: var(--font-display); }" in (
        _RESPONSIVE_TABLE_HTML
    )
