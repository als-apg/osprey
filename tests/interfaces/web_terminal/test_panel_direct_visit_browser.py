"""Browser e2e: a direct visit to a panel page opens the web terminal with that panel.

On the hermetic hub, where a session has a real stored layout, with a page that
calls the real ``applyEmbedded`` behind the real panel proxy. Pins:

- The visit leaves the panel's address for the hub's, and ``?open``/``?at`` leave
  the hub's address.
- The session's layout applies first and the panel opens after it, on the visit's
  page: when the layout does not hold the panel, and when it holds it on another
  page. Either way the apply would undo an open made too early.
- A panel that answers only later still opens.
- An unknown id leaves the hub as it is, and says so.
- Framed and ``embedded=true`` pages stay where they are.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import pytest
from docs.screenshots.contact_sheet import SEEDED_SESSIONS, hermetic_hub

pytestmark = [pytest.mark.browser, pytest.mark.slow]

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

if not _PLAYWRIGHT_AVAILABLE:  # pragma: no cover
    pytest.skip("playwright not installed", allow_module_level=True)

_TIMEOUT_MS = 20_000
_TREE = "#sidebar-session-tree"
_NOTICE = ".undo-notice-text"

#: Longer than the session layout's settle, so an apply that was going to undo
#: the open has done so by the time a test looks again.
_PAST_SETTLE_MS = 1_000

_ACTIVE = re.compile(r"\bactive\b")
_EMBEDDED = re.compile(r"\bembedded\b")

#: A first-party panel page's one duty here: every panel page calls this on load.
_PANEL_HTML = b"""<!doctype html><html><head><title>notes</title></head><body>
<script type="module">
  import { applyEmbedded } from '/design-system/js/frame-params.js';
  applyEmbedded();
</script></body></html>"""


@contextmanager
def _panel_backend(healthy: Event):
    """Serve the panel page at every path, with a health route that follows *healthy*."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/healthz"):
                self.send_response(200 if healthy.is_set() else 503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_PANEL_HTML)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _panel(panel_id: str, url: str, health: str | None = None) -> dict:
    return {
        "id": panel_id,
        "label": panel_id.title(),
        "url": url,
        "healthEndpoint": health,
        "path": "/",
        "configDefined": True,
    }


@pytest.fixture
def healthy() -> Event:
    return Event()


@pytest.fixture
def hub(healthy):
    with _panel_backend(healthy) as url:
        panels = [_panel("notes", url), _panel("late", url, health="/healthz")]
        with hermetic_hub(e2e=True, custom_panels=panels) as running:
            yield running


@pytest.fixture
def page(chromium_browser):
    ctx = chromium_browser.new_context(viewport={"width": 1400, "height": 900})
    yield ctx.new_page()
    ctx.close()


def _tile(page: Page, panel_id: str):
    return page.locator(f'.tile-tab[data-panel-id="iframe:{panel_id}"]')


def _frame_location(page: Page, panel_id: str) -> dict:
    """Where the panel's frame is, once it has loaded a document of this panel."""
    frame = page.locator(f'iframe.panel-iframe[data-panel-id="{panel_id}"]')
    expect(frame).to_be_attached(timeout=_TIMEOUT_MS)
    return (
        frame.element_handle()
        .content_frame()
        .evaluate("() => ({path: location.pathname, hash: location.hash, query: location.search})")
    )


def _expect_on_screen_at(page: Page, panel_id: str, path: str, hash_: str) -> None:
    """The tile is up on *path*, and still is once the session's apply is long over."""
    expect(_tile(page, panel_id)).to_have_count(1, timeout=_TIMEOUT_MS)
    for _ in range(2):
        for _ in range(100):
            where = _frame_location(page, panel_id)
            if (where["path"], where["hash"]) == (path, hash_):
                break
            page.wait_for_timeout(100)
        assert (where["path"], where["hash"]) == (path, hash_), where
        assert "embedded=true" in where["query"], where
        page.wait_for_timeout(_PAST_SETTLE_MS)
    expect(_tile(page, panel_id)).to_have_count(1)


def _in_a_session(page: Page, hub) -> str:
    """Boot the hub on a seeded session, with its layout stored; its transcript id."""
    tid = SEEDED_SESSIONS[0].transcript_id
    page.goto(hub.base_url, wait_until="domcontentloaded")
    row = page.locator(f'{_TREE} .osprey-tree-session[data-session-id="{tid}"]')
    expect(row).to_be_visible(timeout=_TIMEOUT_MS)
    row.locator(".osprey-tree-title").click()
    expect(row).to_have_class(_ACTIVE, timeout=_TIMEOUT_MS)
    _stored_layout(page, hub, tid)
    return tid


def _stored_layout(page: Page, hub, tid: str, *, holding: str | None = None) -> dict:
    """The session's stored layout, once it has one (and it holds *holding*)."""
    url = f"{hub.base_url}/api/sessions/{tid}/layout"
    for _ in range(100):
        layout = page.request.get(url).json()["layout"]
        if layout and (holding is None or holding in (layout.get("pages") or {})):
            return layout
        page.wait_for_timeout(100)
    raise AssertionError(f"the session never stored a layout holding {holding!r}")


def test_a_visit_opens_a_panel_the_session_layout_does_not_hold(page, hub):
    _in_a_session(page, hub)
    expect(_tile(page, "notes")).to_have_count(0)
    page.goto("about:blank")

    page.goto(f"{hub.base_url}/panel/notes/#entry?id=X", wait_until="domcontentloaded")

    expect(page).to_have_url(f"{hub.base_url}/", timeout=_TIMEOUT_MS)
    _expect_on_screen_at(page, "notes", "/panel/notes/", "#entry?id=X")


def test_a_visit_lands_on_its_page_when_the_layout_holds_the_panel_on_another(page, hub):
    tid = _in_a_session(page, hub)
    page.locator('[data-panel-id="notes"]').first.click()
    expect(_tile(page, "notes")).to_have_count(1, timeout=_TIMEOUT_MS)
    layout = _stored_layout(page, hub, tid, holding="notes")
    page.goto("about:blank")
    layout["pages"]["notes"] = "/panel/notes/other"
    stored = page.request.put(
        f"{hub.base_url}/api/sessions/{tid}/layout",
        data={"layout": layout},
        headers={"Origin": hub.base_url},
    )
    assert stored.ok, stored.text()

    page.goto(f"{hub.base_url}/panel/notes/deep?x=1#entry?id=X", wait_until="domcontentloaded")

    _expect_on_screen_at(page, "notes", "/panel/notes/deep", "#entry?id=X")
    assert "x=1" in _frame_location(page, "notes")["query"]


def test_a_panel_that_answers_after_settle_still_opens(page, hub, healthy):
    _in_a_session(page, hub)
    page.goto("about:blank")

    page.goto(f"{hub.base_url}/panel/late/#entry", wait_until="domcontentloaded")
    expect(page).to_have_url(f"{hub.base_url}/", timeout=_TIMEOUT_MS)
    page.wait_for_timeout(_PAST_SETTLE_MS * 2)
    expect(_tile(page, "late")).to_have_count(0)

    healthy.set()

    _expect_on_screen_at(page, "late", "/panel/late/", "#entry")


def test_an_unknown_id_says_so_and_leaves_the_hub_as_it_is(page, hub):
    page.goto(f"{hub.base_url}/?open=nope&at=%2Fpanel%2Fnope%2F", wait_until="domcontentloaded")

    expect(page.locator(_NOTICE)).to_have_text("nope is not a panel here.", timeout=_TIMEOUT_MS)
    expect(page).to_have_url(f"{hub.base_url}/")
    expect(_tile(page, "nope")).to_have_count(0)


def test_a_first_party_panel_page_goes_through_the_same_guard(page, hub):
    """The artifacts gallery is a real panel shell behind the real proxy."""
    page.goto(f"{hub.base_url}/panel/artifacts/", wait_until="domcontentloaded")

    expect(page).to_have_url(f"{hub.base_url}/", timeout=_TIMEOUT_MS)
    expect(_tile(page, "artifacts")).to_have_count(1, timeout=_TIMEOUT_MS)


def test_an_embedded_visit_stays_on_the_panel_page(page, hub):
    page.goto(f"{hub.base_url}/panel/notes/?embedded=true", wait_until="load")

    page.wait_for_timeout(_PAST_SETTLE_MS)
    assert page.url == f"{hub.base_url}/panel/notes/?embedded=true"
    expect(page.locator("body")).to_have_class(_EMBEDDED)
