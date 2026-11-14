"""A minimal hub for panel browser tests: a page that frames one panel.

A panel page has no chrome of its own. Its search box, buttons and menus are
contributions to the web terminal's tile bar (``header-contrib.js``), so a
browser test needs something that frames the panel with ``embedded=true`` and
renders those contributions. :func:`open_panel_host` does that without booting
the web terminal: it serves a small fixture page on the panel's own origin and
returns a :class:`PanelHost`.

The fixture page renders today's ``canopy-host-header-contribution`` kinds as
plain controls that post ``canopy-host-header-action`` back into the frame:

==========  =================================================================
``text``    a ``<span>``
``icon``    a button named by its ``title`` (``disabled``), the icon name in
            ``data-icon``
``nav``     one button per entry (``aria-current`` on the active one)
``button``  a button (``disabled``, ``title``, ``data-tone``)
``search``  an ``<input type="search">``; ``value`` seeds it at creation only,
            and a re-contribution with the same id leaves it alone, as the
            hub does
``menu``    one always-visible button per entry (``aria-checked``, ``disabled``)
==========  =================================================================

Every control carries ``data-host-item="<item id>"``; ``nav`` and ``menu``
entries also carry ``data-host-value="<entry id>"``. Reach them through
:meth:`PanelHost.bar_item`. Unknown kinds are ignored, as the hub ignores them.

Everything else on :class:`PanelHost` resolves first on the panel's
:class:`~playwright.sync_api.Frame`, then on the host
:class:`~playwright.sync_api.Page`, so a test drives the panel as it would a
page (``host.locator(...)``, ``host.evaluate(...)``, ``host.keyboard``,
``host.mouse``, ``host.close()``). Two page verbs a frame lacks are defined here:
``goto`` (keeps ``embedded=true``) and ``reload`` (reloads the frame, keeping its
hash and the host page). Both, like :func:`open_panel_host`, return once the panel
has set its page-ready mark.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tests.interfaces._browser import wait_for_page_ready

if TYPE_CHECKING:
    from playwright.sync_api import Frame, Locator, Page

#: Path of the fixture page on the panel's origin. It exists only as a Playwright route.
HOST_PATH = "/__panel_host__"

_BAR_HEIGHT_PX = 36

_PAGE_EMITTER_VERBS = frozenset({"on", "once", "remove_listener"})

_HOST_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; height: 100%%; overflow: hidden; }
  #host-bar { display: flex; gap: 4px; align-items: center; height: %(bar)dpx; }
  #panel-frame { display: block; border: 0; width: 100%%; height: calc(100%% - %(bar)dpx); }
</style>
</head>
<body>
<div id="host-bar"></div>
<iframe id="panel-frame"></iframe>
<script>
const bar = document.getElementById('host-bar');
const iframe = document.getElementById('panel-frame');
window.__contributions = [];
window.__panelMessages = [];

function act(id, value) {
  iframe.contentWindow.postMessage(
    { type: 'canopy-host-header-action', id, value },
    location.origin
  );
}

function entryButton(item, entry) {
  const button = document.createElement('button');
  button.type = 'button';
  button.dataset.hostItem = item.id;
  button.dataset.hostValue = entry.id;
  button.textContent = entry.label;
  if (item.kind === 'nav' && entry.active) button.setAttribute('aria-current', 'true');
  if (item.kind === 'menu' && 'checked' in entry) {
    button.setAttribute('aria-checked', String(Boolean(entry.checked)));
  }
  button.disabled = Boolean(entry.disabled);
  button.addEventListener('click', () => act(item.id, entry.id));
  return button;
}

/** @returns {HTMLElement[]} the controls of one contributed item */
function render(item, live) {
  if (item.kind === 'text') {
    const span = document.createElement('span');
    span.dataset.hostItem = item.id;
    span.textContent = item.text ?? '';
    return [span];
  }
  if (item.kind === 'button') {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.hostItem = item.id;
    button.dataset.tone = item.tone ?? 'default';
    button.textContent = item.label ?? '';
    if (item.title) button.title = item.title;
    button.disabled = Boolean(item.disabled);
    button.addEventListener('click', () => act(item.id));
    return [button];
  }
  if (item.kind === 'icon') {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.hostItem = item.id;
    button.dataset.icon = item.icon ?? '';
    button.textContent = item.title ?? item.id;
    button.disabled = Boolean(item.disabled);
    button.addEventListener('click', () => act(item.id));
    return [button];
  }
  if (item.kind === 'search') {
    const existing = live.get(item.id);
    if (existing) return [existing];
    const input = document.createElement('input');
    input.type = 'search';
    input.dataset.hostItem = item.id;
    input.placeholder = item.placeholder ?? '';
    input.value = item.value ?? '';
    input.addEventListener('input', () => act(item.id, input.value));
    return [input];
  }
  if ((item.kind === 'nav' || item.kind === 'menu') && Array.isArray(item.items)) {
    return item.items.map((entry) => entryButton(item, entry));
  }
  return [];
}

window.addEventListener('message', (event) => {
  if (event.origin !== location.origin || event.source !== iframe.contentWindow) return;
  const data = event.data;
  if (data?.type !== 'canopy-host-header-contribution') {
    window.__panelMessages.push(data);
    return;
  }
  window.__contributions.push(data);
  const live = new Map(
    [...bar.querySelectorAll('input[data-host-item]')].map((el) => [el.dataset.hostItem, el])
  );
  bar.replaceChildren(...data.items.flatMap((item) => render(item, live)));
});

iframe.src = %(src)s;
</script>
</body>
</html>
"""


def framed_url(url: str) -> str:
    """*url* with ``embedded=true`` in its query, ahead of any hash."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "embedded"]
    query.append(("embedded", "true"))
    return urlunsplit(parts._replace(query=urlencode(query)))


class PanelHost:
    """One panel framed by the fixture page.

    Attributes resolve on :attr:`frame` first and on :attr:`page` second.
    """

    def __init__(self, page: Page, frame: Frame) -> None:
        self.page = page
        self.frame = frame

    def __getattr__(self, name: str) -> Any:
        frame = self.__dict__["frame"]
        # A Frame inherits the emitter verbs but no page event ever reaches it.
        if name not in _PAGE_EMITTER_VERBS and hasattr(frame, name):
            return getattr(frame, name)
        return getattr(self.__dict__["page"], name)

    def bar_item(self, item_id: str, value: str | None = None) -> Locator:
        """The bar control of contributed *item_id* (and of its entry *value*)."""
        selector = f'#host-bar [data-host-item="{item_id}"]'
        if value is not None:
            selector += f'[data-host-value="{value}"]'
        return self.page.locator(selector)

    def search_box(self) -> Locator:
        """The bar's search field, whatever its current id."""
        return self.page.locator('#host-bar input[type="search"]')

    def contributions(self) -> list[dict[str, Any]]:
        """Every contribution the panel has sent, oldest first."""
        return self.page.evaluate("window.__contributions")

    def items(self) -> list[dict[str, Any]]:
        """The items of the panel's latest contribution, waiting for a first one."""
        self.page.wait_for_function("window.__contributions.length > 0")
        return self.page.evaluate("window.__contributions.at(-1).items")

    def panel_messages(self, message_type: str | None = None) -> list[Any]:
        """Other messages the panel posted to its parent, optionally of one ``type``."""
        messages = self.page.evaluate("window.__panelMessages")
        if message_type is None:
            return messages
        return [m for m in messages if isinstance(m, dict) and m.get("type") == message_type]

    def focus_panel(self) -> None:
        """Give the frame the keyboard, so ``host.keyboard`` reaches the panel."""
        self.page.evaluate("document.getElementById('panel-frame').focus()")

    def goto(self, url: str, **kwargs: Any) -> Any:
        """Navigate the frame to *url*, keeping ``embedded=true``; wait for its ready mark."""
        response = self.frame.goto(framed_url(url), **kwargs)
        wait_for_page_ready(self.frame)
        self.focus_panel()
        return response

    def reload(self, **kwargs: Any) -> None:
        """Reload the frame in place, hash included, leaving the host page up.

        Returns once the reloaded panel has set its ready mark.
        """
        frame = self.frame
        with self.page.expect_event("framenavigated", lambda f: f == frame):
            frame.evaluate("location.reload()")
        frame.wait_for_load_state(kwargs.get("wait_until", "load"))
        wait_for_page_ready(frame)
        self.focus_panel()


def open_panel_host(page: Page, url: str, *, wait_until: str = "load") -> PanelHost:
    """Frame the panel at *url* (with ``embedded=true``) in a fixture page on its origin.

    Returns once the panel has set its page-ready mark (``wait_for_page_ready``).

    Install any network fakes on *page* first: the frame's first requests go out here.
    """
    parts = urlsplit(url)
    host_url = urlunsplit((parts.scheme, parts.netloc, HOST_PATH, "", ""))
    html = _HOST_HTML % {"bar": _BAR_HEIGHT_PX, "src": json.dumps(framed_url(url))}

    page.route(
        host_url,
        lambda route: route.fulfill(status=200, content_type="text/html", body=html),
    )
    page.goto(host_url, wait_until="commit")  # page-ready: the fixture host, not an OSPREY page
    handle = page.wait_for_selector("#panel-frame")
    frame = handle.content_frame()
    assert frame is not None, "the fixture page has no panel frame"
    frame.wait_for_url(lambda u: "embedded=true" in u)
    wait_for_page_ready(frame)
    frame.wait_for_load_state(wait_until)
    host = PanelHost(page, frame)
    host.focus_panel()
    return host
