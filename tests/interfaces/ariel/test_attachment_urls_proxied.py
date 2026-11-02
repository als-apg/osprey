"""A stored picture resolves under the panel prefix when the web terminal proxies ARIEL.

The web terminal serves the ARIEL page at ``<outer>/panel/ariel/`` and rewrites
root-absolute ``/api/...`` literals in the page's HTML, JS and CSS
(``routes/proxy.py::_rewrite_content``), but never inside a JSON response. So a
URL the page builds from a JSON value must be joined to the page's API base,
the one literal the proxy rewrites, or it resolves at the origin root and 404s.

These tests take the ``display_url`` the ARIEL API sends for a viewable
picture, put the shipped ``api.js`` through the proxy's own rewrite, and run it
under Node to show the picture lands under the panel prefix. ``safeHref``
joining attachment routes through ``apiUrl`` is pinned in
``attachments-render.test.mjs``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.ariel.api import routes
from osprey.interfaces.web_terminal.routes.proxy import _rewrite_content
from tests.interfaces.ariel._attachments import (
    PNG_URL,
    attachment_entry,
    attachment_row,
    viewable_row,
)
from tests.interfaces.ariel._routes_app import routes_app

_STATIC_JS = Path(__file__).resolve().parents[3] / "src/osprey/interfaces/ariel/static/js"
_OUTER = "/u/alice"
_PANEL = f"{_OUTER}/panel/ariel"

_needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def _viewable_display_url() -> tuple[str, str]:
    """The (attachment_id, display_url) the entry routes send for a viewable picture."""
    item = {"url": PNG_URL, "filename": "pic.png", "type": "image/png"}
    row = attachment_row("e-att", item, mime_type="image/png", viewable=True)
    response = routes._entry_to_response(
        attachment_entry([item]), attachment_rows=[row], model_id=None, file_source=False
    )
    att = response.attachments[0]
    assert att.viewable is True and att.display_url
    return row["attachment_id"], att.display_url


def _api_url(tmp_path: Path, *, proxied: bool, path: str) -> str:
    """Run ``apiUrl(path)`` from the shipped api.js, rewritten as the proxy serves it."""
    source = (_STATIC_JS / "api.js").read_text(encoding="utf-8")
    if proxied:
        source = _rewrite_content(source, "ariel", _OUTER)
    module = tmp_path / ("api.proxied.mjs" if proxied else "api.mjs")
    module.write_text(source, encoding="utf-8")
    script = (
        f"import {{ apiUrl }} from {json.dumps(module.as_uri())};"
        f"process.stdout.write(apiUrl({json.dumps(path)}));"
    )
    done = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


@_needs_node
def test_proxied_page_requests_the_rendition_under_the_panel_prefix(tmp_path):
    attachment_id, display_url = _viewable_display_url()

    assert _api_url(tmp_path, proxied=True, path=display_url) == (
        f"{_PANEL}/api/attachments/{attachment_id}/rendition"
    )


@_needs_node
def test_standalone_page_requests_a_rendition_the_server_serves(tmp_path):
    attachment_id, display_url = _viewable_display_url()

    src = _api_url(tmp_path, proxied=False, path=display_url)

    assert src == f"/api/attachments/{attachment_id}/rendition"
    service = AsyncMock()
    service.repository = AsyncMock()
    service.repository.get_rendition = AsyncMock(
        return_value=viewable_row(attachment_id=attachment_id, entry_id="e-att")
    )
    app = routes_app(service)
    served = TestClient(app).get(src)
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/jpeg"


def test_display_url_is_not_a_root_absolute_api_path():
    """A ``/api/...`` value in JSON is exactly what the proxy leaves unprefixed."""
    _, display_url = _viewable_display_url()

    assert not display_url.startswith("/api/")


def test_draft_attachment_preview_path_is_rewritten_into_the_panel():
    """The draft preview builds its URL from a JS literal, which the proxy prefixes."""
    form = (_STATIC_JS / "entries-form.js").read_text(encoding="utf-8")
    assert "`/api/drafts/${draftId}/attachments/" in form

    rewritten = _rewrite_content(form, "ariel", _OUTER)

    assert f"`{_PANEL}/api/drafts/${{draftId}}/attachments/" in rewritten
