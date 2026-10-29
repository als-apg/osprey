"""Browser proof of C-CAP: opaque panels load from their capability path.

A real hub runs with the auth gate on, and a real Chromium holds the operator's
session cookie. Two agent-writable panels are framed inside the hub page at the
cap URLs the hub mints for that browser:

* a **discovered** multi-file bundle (``/panel-static/<id>/~c/<cap>/``): an HTML
  entry, a module script it imports by relative URL, and a JSON sibling the
  module fetches;
* a **runtime** panel registered at a real loopback HTTP server, proxied through
  the hub's panel proxy (``/panel/<id>/~c/<cap>/``) with the same three files.

Each must load and render — module imported, JSON fetched, text painted — while
running at an opaque origin (``window.origin === "null"``); and each one's
script tries to POST to an operator route (``/api/panel-focus``) with
``credentials: "include"``, which must not change the hub's state. The same
POST from the hub's own page succeeds, so the refusal is the cap's doing and
not a broken route.

The hub's own frame factory (``panel-iframe-factory.js``) is proven on the same
two panels: opened from the rail, each is framed at its cap URL under the final
sandbox ``allow-scripts allow-popups allow-modals allow-downloads`` (no
``allow-same-origin``, no ``allow-forms``) while an OSPREY catalog panel keeps today's attributes; the
panel imports the first-party theme module through C-ASSETS and a live theme
change in the hub reaches it — both as the raw ``osprey-theme-change`` message
and as the ``data-theme`` its own follower applies. On their plain, cookie-gated
paths the same panels' documents still carry the sandbox CSP.

Skips cleanly when the chromium binary is not installed.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER, PANEL_CAP_SEGMENT
from osprey.interfaces.web_auth import get_web_credentials
from osprey.interfaces.web_terminal.panel_discovery import DiscoveredPanel
from tests.interfaces._hub import hub_server

try:
    from playwright.sync_api import Frame, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: The operator route each panel's script attacks. Observable through its GET.
_OPERATOR_ROUTE = "/api/panel-focus"

_INDEX = """<!doctype html>
<html><head><meta charset="utf-8">
<script type="module" src="./app.js"></script></head>
<body>
<div id="origin">pending</div>
<div id="out">pending</div>
<div id="attack">pending</div>
<div id="theme">pending</div>
<div id="assets">pending</div>
</body></html>
"""

#: The module: proves it ran at an opaque origin, fetches its JSON sibling by a
#: relative URL, then tries the operator route with the operator's cookie — once
#: as a CORS request and once as a no-cors simple POST, which needs no preflight.
_APP_JS = """
document.getElementById('origin').textContent = String(window.origin);
window.addEventListener('message', (e) => {
  if (e.source === window.parent && e.data && e.data.type === 'osprey-theme-change') {
    document.getElementById('theme').textContent = String(e.data.theme);
  }
});
// A first-party module imported by absolute path (C-ASSETS): the theme
// follower, which then applies the hub's broadcasts to this document.
import('/design-system/js/theme-manager.js')
  .then((m) => { m.initTheme(); document.getElementById('assets').textContent = 'loaded'; })
  .catch((e) => { document.getElementById('assets').textContent = 'import failed: ' + e; });
fetch('./data.json')
  .then((r) => r.json())
  .then((d) => { document.getElementById('out').textContent = d.word; })
  .catch((e) => { document.getElementById('out').textContent = 'fetch failed: ' + e; });

async function attack() {
  const outcomes = [];
  try {
    const r = await fetch('%(route)s', {
      method: 'POST', credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({panel: '%(target)s'}),
    });
    outcomes.push('cors:' + r.status);
  } catch (e) { outcomes.push('cors:blocked'); }
  try {
    await fetch('%(route)s', {
      method: 'POST', credentials: 'include', mode: 'no-cors',
      headers: {'Content-Type': 'text/plain'},
      body: JSON.stringify({panel: '%(target)s'}),
    });
    outcomes.push('nocors:sent');
  } catch (e) { outcomes.push('nocors:blocked'); }
  document.getElementById('attack').textContent = outcomes.join(' ');
}
attack();
"""

#: The real panel each panel's attack tries to focus — the OTHER opaque panel,
#: so an accepted attack would change the hub's state rather than be refused as
#: an unknown id, and nothing in the hub's own page ever focuses it here.
_ATTACK_TARGET = {"agent-bundle": "agent-dash", "agent-dash": "agent-bundle"}


def _write_bundle(directory: Path, word: str, target: str) -> None:
    directory.mkdir(parents=True)
    (directory / "index.html").write_text(_INDEX)
    (directory / "app.js").write_text(_APP_JS % {"route": _OPERATOR_ROUTE, "target": target})
    (directory / "data.json").write_text(json.dumps({"word": word}))


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request stderr logging
        pass


@contextmanager
def _runtime_backend(directory: Path) -> Iterator[str]:
    """Serve ``directory`` over real HTTP on loopback; yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _hub(tmp_path: Path) -> Iterator[tuple[str, object]]:
    """A real hub with one discovered and one runtime panel, auth gate on."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    bundle = tmp_path / "panels" / "agent-bundle"
    _write_bundle(bundle, "bundle-data", _ATTACK_TARGET["agent-bundle"])
    runtime_root = tmp_path / "runtime"
    _write_bundle(runtime_root, "runtime-data", _ATTACK_TARGET["agent-dash"])

    with _runtime_backend(runtime_root) as backend_url:
        custom = [
            {
                "id": "agent-bundle",
                "label": "BUNDLE",
                "url": "/panel-static/agent-bundle/",
                "healthEndpoint": None,
                "path": "/",
                "discovered": True,
            },
            # A runtime registration: no configDefined marker, so it earns no
            # credential upstream and is opaque.
            {"id": "agent-dash", "label": "RUNTIME", "url": backend_url, "path": "/"},
        ]
        with hub_server(workspace, custom_panels=custom) as (base_url, app):
            app.state.discovered_panel_dirs = {
                "agent-bundle": DiscoveredPanel(
                    id="agent-bundle",
                    label="BUNDLE",
                    entry="index.html",
                    directory=bundle,
                )
            }
            yield base_url, app


def _operator_get(base_url: str, path: str) -> dict:
    secret = get_web_credentials().operator_secret
    resp = requests.get(f"{base_url}{path}", headers={OPERATOR_SECRET_HEADER: secret}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _cap_url(page: Page, panel_id: str) -> str:
    """The cap URL the hub mints for THIS browser's login."""
    return page.evaluate(
        "async (id) => (await (await fetch(`/api/panels/${id}/cap`)).json()).cap_url", panel_id
    )


def _frame_panel(page: Page, panel_id: str, url: str) -> Frame:
    """Frame ``url`` inside the hub page, as the dock frames an opaque panel."""
    page.evaluate(
        """([id, src]) => {
            const f = document.createElement('iframe');
            f.id = 'cap-frame-' + id;
            f.src = src;
            f.style.cssText = 'position:fixed;left:0;top:0;width:400px;height:300px;z-index:99999';
            document.body.appendChild(f);
        }""",
        [panel_id, url],
    )
    page.wait_for_function(
        "(id) => document.getElementById('cap-frame-' + id).contentWindow !== null", arg=panel_id
    )
    handle = page.locator(f"#cap-frame-{panel_id}").element_handle()
    frame = handle.content_frame()
    assert frame is not None
    return frame


@pytest.mark.parametrize(
    ("panel_id", "root", "word"),
    [
        ("agent-bundle", "panel-static", "bundle-data"),
        ("agent-dash", "panel", "runtime-data"),
    ],
    ids=["discovered", "runtime"],
)
def test_an_opaque_panel_renders_from_its_cap_and_cannot_post_as_the_operator(
    chromium_browser, tmp_path, panel_id, root, word
):
    with _hub(tmp_path) as (base_url, _app):
        page = chromium_browser.new_page()
        try:
            target = _ATTACK_TARGET[panel_id]
            frame_posts: list[int] = []
            page.on(
                "response",
                lambda r: (
                    frame_posts.append(r.status)
                    if r.request.method == "POST"
                    and r.url.endswith(_OPERATOR_ROUTE)
                    and PANEL_CAP_SEGMENT in (r.frame.url or "")
                    else None
                ),
            )
            page.goto(base_url, wait_until="domcontentloaded")
            assert _operator_get(base_url, _OPERATOR_ROUTE)["active_panel"] != target

            url = _cap_url(page, panel_id)
            assert url.startswith(f"/{root}/{panel_id}/{PANEL_CAP_SEGMENT}/")

            frame = _frame_panel(page, panel_id, url)

            # It loads and renders: module imported, JSON sibling fetched.
            expect(frame.locator("#out")).to_have_text(word, timeout=15_000)
            # ...at an opaque origin.
            expect(frame.locator("#origin")).to_have_text("null")
            # Its attack ran to completion and reached nothing.
            expect(frame.locator("#attack")).not_to_have_text("pending", timeout=15_000)
            outcome = frame.locator("#attack").inner_text()
            assert "cors:200" not in outcome, outcome
            assert not any(200 <= status < 300 for status in frame_posts), frame_posts
            assert _operator_get(base_url, _OPERATOR_ROUTE)["active_panel"] != target

            # Control: the hub's own page, same cookie, same route — accepted.
            status = page.evaluate(
                """async ([route, target]) => (await fetch(route, {
                    method: 'POST', credentials: 'include',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({panel: target}),
                })).status""",
                [_OPERATOR_ROUTE, "artifacts"],
            )
            assert status == 200
            assert _operator_get(base_url, _OPERATOR_ROUTE)["active_panel"] == "artifacts"
        finally:
            page.close()


def test_the_panel_without_its_cap_does_not_load_in_an_opaque_frame(chromium_browser, tmp_path):
    """The plain path stays behind the cookie gate: no cap, no bundle files."""
    with _hub(tmp_path) as (base_url, _app):
        page = chromium_browser.new_page()
        try:
            page.goto(base_url, wait_until="domcontentloaded")
            status = page.evaluate(
                """async () => {
                    const r = await fetch('/panel-static/agent-bundle/data.json', {
                        credentials: 'omit',
                    });
                    return r.status;
                }"""
            )
            assert status == 401
        finally:
            page.close()


#: The sandbox the hub's factory frames an agent-writable panel under.
_OPAQUE_SANDBOX = "allow-scripts allow-popups allow-modals allow-downloads"
#: Today's attributes, kept by OSPREY-shipped panels.
_FIRST_PARTY_SANDBOX = (
    "allow-scripts allow-same-origin allow-popups allow-forms allow-modals allow-downloads"
)


def _open_from_rail(page: Page, panel_id: str) -> Frame:
    """Open ``panel_id`` from the hub's rail; return its frame once it is built."""
    page.locator(f'.panel-rail-button[data-panel-id="{panel_id}"]').click()
    selector = f'iframe[data-panel-id="{panel_id}"][data-panel-frame="opaque"]'
    page.wait_for_selector(selector, state="attached", timeout=15_000)
    handle = page.locator(selector).element_handle()
    assert handle is not None
    frame = handle.content_frame()
    assert frame is not None
    return frame


@pytest.mark.parametrize(
    ("panel_id", "root", "word"),
    [
        ("agent-bundle", "panel-static", "bundle-data"),
        ("agent-dash", "panel", "runtime-data"),
    ],
    ids=["discovered", "runtime"],
)
def test_the_factory_frames_an_agent_panel_opaque_and_the_theme_reaches_it(
    chromium_browser, tmp_path, panel_id, root, word
):
    with _hub(tmp_path) as (base_url, _app):
        page = chromium_browser.new_page()
        try:
            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_selector(
                f'.panel-rail-button[data-panel-id="{panel_id}"]', timeout=15_000
            )
            frame = _open_from_rail(page, panel_id)
            iframe = page.locator(f'iframe[data-panel-id="{panel_id}"]')

            # The factory's final sandbox: no same-origin, no forms.
            assert iframe.get_attribute("sandbox") == _OPAQUE_SANDBOX
            src = iframe.get_attribute("src") or ""
            assert f"/{root}/{panel_id}/{PANEL_CAP_SEGMENT}/" in src, src

            # It renders from its cap, at an opaque origin, and its first-party
            # module import (C-ASSETS) resolved.
            expect(frame.locator("#out")).to_have_text(word, timeout=15_000)
            expect(frame.locator("#origin")).to_have_text("null")
            expect(frame.locator("#assets")).to_have_text("loaded", timeout=15_000)

            # A live theme change in the hub reaches the opaque panel.
            before = frame.evaluate("document.documentElement.getAttribute('data-theme')")
            new_theme = page.evaluate(
                """async () => {
                    const m = await import('/design-system/js/theme-manager.js');
                    m.toggleTheme();
                    return m.getTheme();
                }"""
            )
            assert new_theme != before
            expect(frame.locator("#theme")).to_have_text(new_theme, timeout=10_000)
            # ...and the panel's own follower applied it.
            expect(frame.locator("html")).to_have_attribute("data-theme", new_theme)
        finally:
            page.close()


def test_an_osprey_catalog_panel_keeps_todays_sandbox(chromium_browser, tmp_path):
    with _hub(tmp_path) as (base_url, _app):
        page = chromium_browser.new_page()
        try:
            page.goto(base_url, wait_until="domcontentloaded")
            page.wait_for_selector('.panel-rail-button[data-panel-id="artifacts"]', timeout=15_000)
            page.locator('.panel-rail-button[data-panel-id="artifacts"]').click()
            frame = page.locator('iframe[data-panel-id="artifacts"]')
            frame.wait_for(state="attached", timeout=15_000)
            assert frame.get_attribute("sandbox") == _FIRST_PARTY_SANDBOX
            assert PANEL_CAP_SEGMENT not in (frame.get_attribute("src") or "")
        finally:
            page.close()


@pytest.mark.parametrize(
    "path",
    ["/panel-static/agent-bundle/", "/panel/agent-dash/index.html"],
    ids=["discovered", "runtime"],
)
def test_the_plain_path_serves_the_panel_document_sandboxed(tmp_path, path):
    """Opened at its cookie-gated URL, an agent panel's document is still opaque."""
    with _hub(tmp_path) as (base_url, _app):
        secret = get_web_credentials().operator_secret
        resp = requests.get(
            f"{base_url}{path}", headers={OPERATOR_SECRET_HEADER: secret}, timeout=10
        )
        assert resp.status_code == 200
        csp = resp.headers.get("content-security-policy", "")
        assert "sandbox" in csp and "allow-same-origin" not in csp, csp
