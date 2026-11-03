"""Browser tests: the collaborative-panels contract between dock and server.

``test_panels_browser.py`` proves the docked workspace itself (tiles, tabs,
drags, persistence, the echo guard). This suite proves the *two-party* contract
layered on top of it — the one where the agent sees, and can rearrange, what is
actually on screen:

  * the layout REPORT (``POST /api/panel-layout`` with a dock layout ``layout`` —
    every tile with its tabs, showing tab and size, the terminal's included —
    → ``GET /api/panels``'s ``layout`` / ``rev`` / ``layout_age_s`` /
    ``layout_dock``),
  * the agent's OPS (``POST /api/panel-op/<op>``, relayed as a ``panel_op``
    frame to the one current dock client and answered through
    ``/api/panel-op-result``): a plain ``open`` follows rule 7 and becomes a
    tab of the last service tile; ``open-set`` / ``open-arrangement`` arrange
    exactly, and a preset click is the same ``open-arrangement`` route;
  * the terminal invariant on those ops: no agent verb moves, closes or
    arranges the terminal.

Every assertion here needs a real dock, a real SSE stream and real
client-side fetches, so none of it is reachable from the unit suites:

  * the reported layout is the layout holder the live dock shows — a Python
    test can post any layout it likes, but only a browser can prove the
    layout the dock produces.
  * the "no echo loop" criteria are about POSTs a client does **not** make.
    They are asserted by intercepting the page's own requests; a mocked fetch
    proves only that the mock was not called.
  * the vertical axis (a column reported as a column, read top-to-bottom
    before what sits beside it) can only be pinned by making a real vertical
    split and reading the result back.

Run:
    uv run pytest tests/interfaces/web_terminal/test_panels_collab_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager

import pytest
import requests

from osprey.interfaces._serving import run_app_server
from tests.interfaces._browser import rail_entry, wait_for_hub_ready
from tests.interfaces._hub import hub_app
from tests.interfaces._panel_launch import unserved_url

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Panels used across the suite
# ---------------------------------------------------------------------------
#
# ``healthEndpoint: None`` marks a custom panel healthy on load with no polling
# and no backend (panel-manager.js), which is what most tests here want: the
# subject is tile placement, not health. The one test that needs an UNHEALTHY
# panel builds it separately, pointing a real healthEndpoint at a dead port.

_DATA_VIZ: dict = {
    "id": "data-viz",
    "label": "DATA VIZ",
    "url": "http://data-viz.internal:8080",
    "healthEndpoint": None,
    "path": "/",
}

_SCOPE: dict = {
    "id": "scope",
    "label": "SCOPE",
    "url": "http://scope.internal:8080",
    "healthEndpoint": None,
    "path": "/",
}

#: rail id → dock tab accessible name, for the tab locator below.
_LABELS = {"artifacts": "Artifacts", "data-viz": "DATA VIZ", "scope": "SCOPE"}


def _unhealthy_panel(panel_id: str = "beam-cam") -> dict:
    """A custom panel that polls a port nothing listens on — permanently unhealthy.

    A ``healthEndpoint`` is what makes a panel poll at all; pointing it at a
    closed port makes every poll fail, so the panel stays unhealthy for the
    lifetime of the test instead of racing a backend into readiness.
    """
    return {
        "id": panel_id,
        "label": panel_id.upper(),
        "url": unserved_url(),
        "healthEndpoint": "/health",
        "path": "/",
    }


# ---------------------------------------------------------------------------
# Live-server context manager
# ---------------------------------------------------------------------------


@contextmanager
def _live_server(
    workspace_dir,
    enabled_panels,
    custom_panels=None,
    presets=None,
    project_cwd: str | None = None,
    panels_delay: float = 0.0,
):
    """Launch a real web terminal server on a free port in a background thread.

    Mirrors the launcher in ``test_panels_browser.py`` (companion backends
    patched out, artifacts reporting the standard fallback URL) with the two
    knobs this suite needs:

    ``presets`` seeds ``app.state.panel_presets`` — the ``web.presets`` entries
    (``{name, panels, layout}``) the "+" popover's "Layouts" section renders and
    ``/api/panel-op/open-arrangement`` resolves by name — before the first page
    load, so both preset paths see it.

    ``panels_delay`` holds every ``GET /api/panels`` open for that many seconds.
    That is the boot window the layout reporter must not report inside: the
    stored-layout restore only happens after this round trip, so a reporter
    armed on a timer would publish the pre-restore arrangement as fact.

    Yields:
        (base_url: str, app: FastAPI) — live server address and the app object.
    """
    with hub_app(
        workspace_dir, enabled_panels=enabled_panels, custom_panels=custom_panels or []
    ) as app:
        if panels_delay:

            @app.middleware("http")
            async def _delay_panels(request, call_next):
                if request.url.path == "/api/panels" and request.method == "GET":
                    await asyncio.sleep(panels_delay)
                return await call_next(request)

        with run_app_server(app) as base_url:
            # Set AFTER startup: the lifespan assigns project_cwd and the preset
            # list from config, so a pre-startup value would be clobbered.
            if project_cwd is not None:
                app.state.project_cwd = project_cwd
            if presets is not None:
                app.state.panel_presets = presets
            yield base_url, app


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------


# Passive in-flight counter around window.fetch, installed before the app
# boots. The panel write helpers (panel-commands.js) are fire-and-forget by
# design: a click resolves in the DOM via the server's SSE echo, so the browser
# can still have focus/visibility POSTs in flight after every DOM- and
# server-side wait has converged. A test that then issues its own write from
# the pytest process races those stragglers on unrelated connections — this
# counter is what lets it wait for the browser's writes to actually settle.
# `.then(settle, settle)` fires when headers arrive, which is after the server
# has processed the POST — exactly the ordering the barrier needs.
_FETCH_COUNTER_JS = r"""(() => {
  window.__ospreyFetchInflight = 0;
  const orig = window.fetch.bind(window);
  window.fetch = (...args) => {
    window.__ospreyFetchInflight += 1;
    const settle = () => { window.__ospreyFetchInflight -= 1; };
    const p = orig(...args);
    p.then(settle, settle);
    return p;
  };
})();"""


def _wait_for_fetch_quiescence(page: Page, *, timeout: float = 10.0) -> None:
    """Block until the page has had no fetch in flight for 3 consecutive reads.

    A single zero-read is not enough: the SSE echo of a write can itself
    trigger the next write (a dock change schedules an open-tiles report), so
    the counter can pass through zero between links of a chain. Three reads
    50ms apart require a genuine quiet period, not an instant between requests.
    """
    deadline = time.monotonic() + timeout
    streak = 0
    inflight = None
    while time.monotonic() < deadline:
        inflight = page.evaluate("window.__ospreyFetchInflight ?? 0")
        streak = streak + 1 if inflight == 0 else 0
        if streak >= 3:
            return
        page.wait_for_timeout(50)
    raise AssertionError(f"browser fetches never went quiet ({inflight} still in flight)")


def _open_page(browser, base_url: str) -> Page:
    """Open a page with the fetch counter installed and wait until the hub is ready."""
    page = browser.new_page()
    page.add_init_script(_FETCH_COUNTER_JS)
    page.goto(base_url, wait_until="domcontentloaded")
    wait_for_hub_ready(page)
    return page


# ---------------------------------------------------------------------------
# Dock DOM / module helpers
# ---------------------------------------------------------------------------

# Every dockview group with its tab keys (in bar order), its showing tab and its
# pixel rectangle. The showing panel's header (dock-tab.js) carries the tile's
# tabs: its strip items when it holds several, its own ``data-panel-key`` when
# one (the terminal's is ``terminal``); reported under their operator labels.
_GROUPS_JS = r"""() => [...document.querySelectorAll('#dock-root .dv-groupview')].map(g => {
  const bar = [...g.querySelectorAll('.tile-tab')]
    .find((b) => !b.parentElement?.classList.contains('dv-inactive-tab')) ?? null;
  const items = bar ? [...bar.querySelectorAll('.tile-tab-item')] : [];
  const own = bar?.getAttribute('data-panel-key') ?? null;
  const tabs = items.length ? items.map((t) => t.getAttribute('data-panel-key')) : (own ? [own] : []);
  const showing = items.length
    ? (bar.querySelector('.tile-tab-item--showing')?.getAttribute('data-panel-key') ?? null)
    : own;
  const r = g.getBoundingClientRect();
  return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width),
           h: Math.round(r.height), tabs, showing };
})"""

# The client's OWN layout (dock layout), straight from the layout holder the
# reporter posts from. Asserting on this (rather than only on the server
# read-back) separates "the dock is arranged wrongly" from "the report never
# landed".
_LAYOUT_JS = r"""async () => {
  const { getLayoutHolder } = await import('/static/js/dock-apply.js');
  return getLayoutHolder()?.layout ?? null;
}"""

#: Operator label → tab key.
_TAB_KEYS = {"Agent": "terminal", **{label: key for key, label in _LABELS.items()}}


def _dock_groups(page: Page) -> list[dict]:
    groups = page.evaluate(_GROUPS_JS)
    names = {"terminal": "Agent", **_LABELS}
    for g in groups:
        g["tabs"] = [names.get(k, k) for k in g["tabs"]]
        g["showing"] = names.get(g["showing"], g["showing"])
    return groups


def _tiles(node: dict | None) -> list[dict]:
    """Every tile of a dock layout, depth first (reading order)."""
    if node is None:
        return []
    if "tile" in node:
        return [node]
    return [t for item in node["items"] for t in _tiles(item)]


def _shape(node: dict | None) -> str | None:
    """A layout as a readable shape: ``row[artifacts,[*data-viz|scope],terminal]``.

    A one-tab tile is its key; a tile with several tabs lists them with ``*``
    before the showing one (the reading test_dock_tiles_browser uses).
    """
    if node is None:
        return None
    if "tile" in node:
        tabs = node["tabs"]
        if len(tabs) == 1:
            return tabs[0]
        return "[" + "|".join(("*" if k == node["showing"] else "") + k for k in tabs) + "]"
    return f"{node['split']}[{','.join(_shape(item) for item in node['items'])}]"


def _service_keys(layout: dict | None) -> list[str] | None:
    """The service tabs of a layout: every tab in reading order, no terminal.

    Applied to the client's layout and the server read-back alike, so the two
    compare.
    """
    if layout is None:
        return None
    return [k for t in _tiles(layout) for k in t["tabs"] if k != "terminal"]


def _tile_of(layout: dict | None, key: str) -> str | None:
    return next((t["tile"] for t in _tiles(layout) if key in t["tabs"]), None)


def _client_layout(page: Page) -> dict | None:
    return page.evaluate(_LAYOUT_JS)


def _client_service_tabs(page: Page) -> list[str] | None:
    return _service_keys(_client_layout(page))


def _wait_for_client_tiles(page: Page, expected: list[str], timeout: float = 10.0) -> dict:
    """Block until the page's own layout holds exactly ``expected`` service tabs.

    Polls the holder rather than a tab locator: the assertion subject is the
    layout the reporter would post, and a tab appearing is not the same instant
    as the holder settling on its final shape. Returns that layout.
    """
    deadline = time.monotonic() + timeout
    layout = None
    while time.monotonic() < deadline:
        layout = _client_layout(page)
        if _service_keys(layout) == expected:
            return layout
        page.wait_for_timeout(100)
    raise AssertionError(f"the client never held {expected!r}; last layout: {layout!r}")


def _service_tab(page: Page, label: str):
    """The showing header's handle for the panel labelled ``label``: its strip
    item when its tile holds several tabs, the lone tab's header otherwise."""
    key = _TAB_KEYS.get(label, label)
    return page.locator(
        f'#dock-root .dv-tab:not(.dv-inactive-tab) .tile-tab-item[data-panel-key="{key}"], '
        f"#dock-root .dv-tab:not(.dv-inactive-tab) "
        f'.tile-tab[data-panel-key="{key}"]:not(:has(.tile-tab-item))'
    )


def _rail_ids(page: Page) -> list[str]:
    """Every rail entry's panel id, in rail order (the terminal anchor leads)."""
    return page.eval_on_selector_all(".panel-rail-button", "els => els.map(e => e.dataset.panelId)")


def _active_rail_id(page: Page) -> str | None:
    return page.evaluate(
        "() => document.querySelector('.panel-rail-button.active')?.dataset.panelId ?? null"
    )


def _open_beside(page: Page, panel_id: str) -> None:
    """Open ``panel_id`` as a NEW tile beside the active one from its menu.

    A plain rail click makes the panel a TAB of the last service tile (rule
    7), so every test here that needs side-by-side tiles grows the layout
    through the entry's right-click menu: "Open in a new tile" is the human
    open-beside verb (panel-context-menu.js).
    """
    entry = page.locator(f'button.panel-rail-button[data-panel-id="{panel_id}"]')
    entry.click(button="right")
    menu = page.locator(".rail-context-menu")
    expect(menu).to_have_count(1, timeout=5_000)
    menu.locator(".rail-context-item", has_text="Open in a new tile").click()
    expect(_service_tab(page, _LABELS[panel_id])).to_have_count(1, timeout=5_000)


def _close_tile(page: Page, panel_id: str) -> None:
    """Close a panel's tab through its own "×" — the human close gesture.

    Close is tab first: on a tile whose only tab this is, the tile closes too.
    """
    label = _LABELS[panel_id]
    tab = _service_tab(page, label)
    tab.click()
    bar = page.locator(
        f'#dock-root .dv-tab:not(.dv-inactive-tab) .tile-tab[data-tile="{_tile_of(_client_layout(page), panel_id)}"]'
    )
    bar.hover()
    bar.locator(".tile-tab-close").click()
    expect(_service_tab(page, label)).to_have_count(0, timeout=5_000)


#: One rail drag into the dock, dispatched in the page with a real DataTransfer
#: from the rail entry's own dragstart (which raises the iframe shield — the
#: test never does), then dragenter/dragover/drop on whatever is under the
#: point and dragend on the entry.
_RAIL_DRAG_JS = r"""async ({ panelId, x, y }) => {
  const frame = () => new Promise((r) => requestAnimationFrame(() => r(null)));
  const entry = document.querySelector(`button.panel-rail-button[data-panel-id="${panelId}"]`);
  if (!entry) return { error: 'no rail entry' };
  const dataTransfer = new DataTransfer();
  const init = (px, py) => ({ bubbles: true, cancelable: true, composed: true,
                              dataTransfer, clientX: px, clientY: py });
  const s = entry.getBoundingClientRect();
  entry.dispatchEvent(new DragEvent('dragstart', init(s.x + s.width / 2, s.y + s.height / 2)));
  const shieldRose = !!document.querySelector('.main-container.dock-dragging');
  let last = null;
  let preview = null;
  for (const dx of [-24, -12, -4, 0]) {
    const target = document.elementFromPoint(x + dx, y);
    if (!target) continue;
    if (target !== last) {
      last?.dispatchEvent(new DragEvent('dragleave', init(x + dx, y)));
      target.dispatchEvent(new DragEvent('dragenter', init(x + dx, y)));
      last = target;
    }
    target.dispatchEvent(new DragEvent('dragover', init(x + dx, y)));
    await frame();
    target.dispatchEvent(new DragEvent('dragover', init(x + dx, y)));
    await frame();
    const box = document.querySelector('.dock-drop-preview__rect');
    preview = box ? ([...box.classList].find((c) => c.startsWith('dock-drop-preview__rect--'))?.slice(25) ?? null) : null;
  }
  last?.dispatchEvent(new DragEvent('drop', init(x, y)));
  entry.dispatchEvent(new DragEvent('dragend', init(x, y)));
  return { shieldRose, preview };
}"""

#: A viewport point inside the content of the tile holding ``key``.
_CONTENT_POINT_JS = r"""({ key, fx, fy, dy }) => {
  const tab = [...document.querySelectorAll(`#dock-root .tile-tab-item[data-panel-key="${key}"], #dock-root .tile-tab[data-panel-key="${key}"]`)]
    .find((n) => !n.closest('.dv-tab')?.classList.contains('dv-inactive-tab')) ?? null;
  const content = tab?.closest('.dv-groupview')?.querySelector('.dv-content-container');
  if (!content) return null;
  const r = content.getBoundingClientRect();
  return { x: r.x + r.width * fx, y: r.y + r.height * fy + (dy ?? 0) };
}"""


# ---------------------------------------------------------------------------
# Request interception
# ---------------------------------------------------------------------------


def _track_panel_posts(page: Page) -> list[tuple[float, str, dict | None]]:
    """Record ``(monotonic_time, endpoint, parsed_body)`` for every panel POST.

    The no-loop and no-echo criteria of this feature are all statements about
    requests the client must NOT make, so the request stream is the instrument.
    Bodies are parsed because a layout report's *content* is the assertion (a
    ``{"layout": <dock layout>, "dock": true, "client_id": ...}`` body that holds
    no service tab during boot is the specific failure).
    """
    posts: list[tuple[float, str, dict | None]] = []

    def _record(req):
        if req.method != "POST" or "/api/panel" not in req.url:
            return
        body: dict | None
        try:
            body = json.loads(req.post_data or "null")
        except (ValueError, TypeError):  # pragma: no cover - defensive
            body = None
        posts.append((time.monotonic(), req.url.rsplit("/api/", 1)[-1], body))

    page.on("request", _record)
    return posts


def _endpoints(posts: list[tuple[float, str, dict | None]]) -> list[str]:
    return [endpoint for _, endpoint, _ in posts]


def _layout_reports(posts: list[tuple[float, str, dict | None]]) -> list[dict | None]:
    return [body for _, endpoint, body in posts if endpoint == "panel-layout"]


# ---------------------------------------------------------------------------
# Server read-back helpers
# ---------------------------------------------------------------------------


def _panels_state(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/api/panels", timeout=10)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _wait_for_service_tabs(base_url: str, expected, timeout: float = 15.0) -> dict:
    """Poll ``GET /api/panels`` until its ``layout`` holds exactly ``expected``.

    ``expected`` is the reported dock layout's service tabs in reading order, the
    terminal excluded. Returns the whole payload so the caller can assert the
    freshness fields and the layout off the same read that observed the
    occupancy.
    """
    deadline = time.monotonic() + timeout
    state: dict = {}
    while time.monotonic() < deadline:
        state = _panels_state(base_url)
        if _service_keys(state["layout"]) == expected:
            return state
        time.sleep(0.2)
    raise AssertionError(f"service tabs never became {expected!r}; last read: {state!r}")


def _wait_for_active(base_url: str, expected: str, timeout: float = 10.0) -> None:
    """Poll ``GET /api/panels`` until the server's recorded ``active`` matches."""
    deadline = time.monotonic() + timeout
    state: dict = {}
    while time.monotonic() < deadline:
        state = _panels_state(base_url)
        if state["active"] == expected:
            return
        time.sleep(0.2)
    raise AssertionError(f"active never became {expected!r}; last read: {state!r}")


# ===========================================================================
# (a) The human half: a tile gesture reaches the server
# ===========================================================================


def test_human_tile_close_updates_layout_and_freshness(tmp_path, chromium_browser):
    """Closing a tile by hand updates the reported layout, with honest freshness.

    Tile occupancy used to live entirely in the browser: the operator could close
    a panel and the agent would keep reading it off the rail as "visible". Here
    two tiles are opened, the server is checked to see both in reading order, the
    data-viz tab — its tile's only tab — is closed through its own "×", which
    closes the tile with it, and the server must converge on the remaining
    tile: a dock layout report (``layout`` with every tile, the terminal's
    included, and a fresh ``rev``) from a client that has a dock shell
    (``dock: true``), with an age that reflects the close that just happened,
    not the boot.

    Rail MEMBERSHIP is deliberately asserted unchanged: a close is occupancy, and
    the two must not be conflated in either direction.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_DATA_VIZ],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=15_000)
        _open_beside(page, "data-viz")

        # Both tiles reach the server, in the order they read off the screen.
        before = _wait_for_service_tabs(base_url, ["artifacts", "data-viz"])
        assert before["layout_dock"] is True, before
        assert _shape(before["layout"]) == "row[artifacts,data-viz,terminal]", before
        rail_before = _rail_ids(page)

        # Act — the operator closes the data-viz tab, and with it its tile.
        _close_tile(page, "data-viz")

        # Assert — occupancy converges, freshness describes THIS change.
        after = _wait_for_service_tabs(base_url, ["artifacts"])
        assert after["layout_dock"] is True, after
        assert _shape(after["layout"]) == "row[artifacts,terminal]", after
        assert after["rev"] > before["rev"], (before["rev"], after["rev"])
        assert after["layout"] == _client_layout(page), (after["layout"], _client_layout(page))
        assert isinstance(after["layout_age_s"], float), after
        assert 0.0 <= after["layout_age_s"] < 10.0, (
            f"age should describe the close that just landed: {after}"
        )

        # …and membership is untouched: a closed tile is still one rail click away.
        assert _rail_ids(page) == rail_before, _rail_ids(page)
        assert set(after["visible"]) == set(before["visible"]), (before, after)

        page.close()


# ===========================================================================
# (b) The agent's plain open follows rule 7
# ===========================================================================


def test_agent_open_follows_rule_7_and_posts_nothing_back(tmp_path, chromium_browser):
    """An agent open of a closed rail member becomes a tab, silently (rule 7).

    The polite agent SWITCH that opened a tile beside the operator's is gone. A
    plain agent open (``POST /api/panel-op/open``, relayed to the dock client as
    a ``panel_op`` frame) follows the same rule a person's rail click does: the
    panel becomes a tab of the last service tile used and is shown; nothing
    resizes. Two halves, both only observable in a browser:

    * **Non-destructive.** The operator has two tiles open; the agent opens a
      third panel that is a rail member but holds no tab. It joins the last
      service tile (data-viz's, the one the operator just opened) as its showing
      tab; both tiles keep their ids, their other tabs and their rectangles,
      and no tile is added.
    * **No echo.** Applying the op drives a placeholder add and a programmatic
      active-panel change, both of which look exactly like human dock gestures
      to dockview. If either leaked back out as a ``POST /api/panel-focus`` the
      client would be talking to itself; only the op's answer
      (``/api/panel-op-result``) and the occupancy report may follow.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_DATA_VIZ, _SCOPE],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=15_000)
        _open_beside(page, "data-viz")
        # scope is a rail member from config but holds no tab.
        assert "scope" in _rail_ids(page)
        expect(_service_tab(page, "SCOPE")).to_have_count(0)

        # Drain the open-beside gesture's own POSTs before counting.
        before = _wait_for_client_tiles(page, ["artifacts", "data-viz"])
        _wait_for_service_tabs(base_url, ["artifacts", "data-viz"])
        _wait_for_fetch_quiescence(page)
        page.wait_for_timeout(500)
        rects_before = sorted((g["x"], g["y"], g["w"], g["h"]) for g in _dock_groups(page))
        posts = _track_panel_posts(page)

        # Act — the agent's open_panel, as the MCP tool issues it.
        r = requests.post(
            f"{base_url}/api/panel-op/open",
            json={"request_id": "agent-open-scope", "member": "scope"},
            timeout=10,
        )
        assert r.status_code == 200, r.text
        result = r.json()
        assert result["status"] == "ok", result

        # The panel is on screen, as the showing tab of data-viz's tile…
        expect(_service_tab(page, "SCOPE")).to_have_count(1, timeout=10_000)
        expect(
            page.locator('button.panel-rail-button[data-panel-id="scope"].active')
        ).to_have_count(1, timeout=10_000)
        after = _wait_for_client_tiles(page, ["artifacts", "data-viz", "scope"])
        dv_tile = _tile_of(before, "data-viz")
        assert _tile_of(after, "scope") == dv_tile, _shape(after)
        assert _shape(after) == "row[artifacts,[data-viz|*scope],terminal]", _shape(after)

        # …and neither prior tile was disturbed: same tiles, nothing resized.
        assert [t["tile"] for t in _tiles(after)] == [t["tile"] for t in _tiles(before)]
        assert _tile_of(after, "artifacts") == _tile_of(before, "artifacts")
        rects_after = sorted((g["x"], g["y"], g["w"], g["h"]) for g in _dock_groups(page))
        assert rects_after == rects_before, (rects_before, rects_after)

        # The op returned the layout the dock applied, and the server holds it.
        assert _shape(result["layout"]) == _shape(after), (result["layout"], after)
        state = _wait_for_service_tabs(base_url, ["artifacts", "data-viz", "scope"])
        assert state["rev"] >= result["rev"], (state, result)

        # The client answered the op and reported; it said nothing else.
        page.wait_for_timeout(800)
        assert "panel-op-result" in _endpoints(posts), _endpoints(posts)
        assert set(_endpoints(posts)) <= {"panel-layout", "panel-op-result"}, (
            f"an applied agent open echoed back out: {_endpoints(posts)}"
        )

        page.close()


# ===========================================================================
# (c) An agent arrangement reaches the dock client, without a report loop
# ===========================================================================

#: data-viz | artifacts, as an open_set layout over member indexes.
_SIDE_BY_SIDE = {"split": "row", "items": [{"tabs": [0]}, {"tabs": [1]}]}


def test_arrange_converges_from_two_layouts_without_a_report_loop(tmp_path, chromium_browser):
    """One arrangement lands on THE dock client exactly; nobody loops reports.

    An agent arrangement is an op addressed to one dock client — the current
    one, the latest ``dock: true`` reporter — not a broadcast every client
    applies. Two pages are given deliberately different layouts (A: artifacts
    alone; B: three tiles, opened last, so B is the dock client), then one
    ``POST /api/panel-op/open-set`` with ``as: "layout"`` names data-viz |
    artifacts. B lands on exactly that arrangement — scope's tile closed,
    left-to-right on screen, beside the terminal — and the server's layout is
    B's; A, which the op never addressed, keeps its own workspace untouched.

    The second half is the convergence criterion. Applying an arrangement makes
    the dock emit layout changes, each of which the reporter would post — and
    every report the server *stored* would be a fact a hook could read. If the
    dedupe on either end were missing, the reports would ping-pong
    indefinitely; the contract is at most ONE report per client after the
    arrangement settles (the op's own answer records B's layout). That is a
    statement about requests, so it is measured by intercepting them.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_DATA_VIZ, _SCOPE],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        # Client A: the boot layout — artifacts alone.
        page_a = _open_page(chromium_browser, base_url)
        expect(_service_tab(page_a, "Artifacts")).to_have_count(1, timeout=15_000)
        _wait_for_client_tiles(page_a, ["artifacts"])

        # Client B: three tiles, deliberately in a different order.
        page_b = _open_page(chromium_browser, base_url)
        expect(_service_tab(page_b, "Artifacts")).to_have_count(1, timeout=15_000)
        _open_beside(page_b, "scope")
        _open_beside(page_b, "data-viz")
        _wait_for_client_tiles(page_b, ["artifacts", "scope", "data-viz"])
        # B's gestures report but broadcast nothing — human focus stays local.
        _wait_for_client_tiles(page_a, ["artifacts"])
        # B reported last, so B is the dock client the op will address.
        _wait_for_service_tabs(base_url, ["artifacts", "scope", "data-viz"])
        a_before = _client_layout(page_a)

        # Let both clients' own reports drain, then watch what they say next.
        _wait_for_fetch_quiescence(page_a)
        _wait_for_fetch_quiescence(page_b)
        page_b.wait_for_timeout(800)
        posts_a = _track_panel_posts(page_a)
        posts_b = _track_panel_posts(page_b)

        # Act — one declarative arrangement.
        r = requests.post(
            f"{base_url}/api/panel-op/open-set",
            json={
                "request_id": "agent-arrange-1",
                "members": ["data-viz", "artifacts"],
                "as": "layout",
                "layout": _SIDE_BY_SIDE,
            },
            timeout=10,
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "ok", r.json()

        # Assert — B, the dock client, holds exactly the arrangement.
        layout_b = _wait_for_client_tiles(page_b, ["data-viz", "artifacts"])
        expect(_service_tab(page_b, "SCOPE")).to_have_count(0, timeout=10_000)
        assert _shape(layout_b) == "row[data-viz,artifacts,terminal]", _shape(layout_b)
        # Left-to-right on screen, not merely in the layout.
        service = [g for g in _dock_groups(page_b) if "Agent" not in g["tabs"]]
        order = [g["tabs"][0] for g in sorted(service, key=lambda g: g["x"])]
        assert order == ["DATA VIZ", "Artifacts"], order

        # The arrangement reached the server as an observation, not just a command.
        state = _wait_for_service_tabs(base_url, ["data-viz", "artifacts"])
        assert _shape(state["layout"]) == _shape(layout_b), state

        # A was never addressed: its workspace is its own, unchanged.
        assert _client_layout(page_a) == a_before, (a_before, _client_layout(page_a))

        # Now the no-loop criterion: give any ping-pong a generous window to
        # show itself (a loop would be unbounded, so a few seconds is decisive).
        page_a.wait_for_timeout(2_500)
        page_b.wait_for_timeout(500)
        for name, posts in (("A", posts_a), ("B", posts_b)):
            reports = _layout_reports(posts)
            assert len(reports) <= 1, f"client {name} looped layout reports: {reports}"
        assert set(_endpoints(posts_b)) <= {"panel-layout", "panel-op-result"}, (
            f"client B echoed an applied arrangement back out: {_endpoints(posts_b)}"
        )
        assert _endpoints(posts_a) == [], f"client A acted on an op it was not sent: {posts_a}"

        page_a.close()
        page_b.close()


# ===========================================================================
# (d) Health is a client-side fact: focus fallback, and no empty tiles
# ===========================================================================


def test_arrange_with_unhealthy_member_focuses_elsewhere_and_docks_no_empty_tile(
    tmp_path, chromium_browser
):
    """An arrangement naming an unhealthy panel focuses elsewhere and opens no ghost.

    Panel health only exists in the browser, so the server relays the
    arrangement and the dock client applies it: the healthy members get their
    tiles, and focus lands on the first healthy listed tile. A tile is only
    ever docked for a panel that can fill it — an unhealthy panel's activation
    refuses, so docking a tile for it would leave a permanently empty rectangle
    on the operator's screen, worse than not opening it at all. Its rail entry
    survives either way, so nothing is lost: the panel is one click away the
    moment it comes back.

    The starting focus is deliberately on a panel the arrangement does NOT list
    (data-viz, opened beside). From the boot default, focus would already be
    sitting on ``artifacts`` and the focus assertion would be confirming a state
    that was true before the arrangement ran.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    broken = _unhealthy_panel()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_DATA_VIZ, broken],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=15_000)
        # Let the broken panel's first health polls fail so it is decidedly
        # unhealthy before the arrangement is applied.
        page.wait_for_timeout(1_500)
        assert (
            page.locator(f'button.panel-rail-button[data-panel-id="{broken["id"]}"]').count() == 1
        )

        # Move focus onto data-viz, which the arrangement below does not list.
        _open_beside(page, "data-viz")
        _wait_for_client_tiles(page, ["artifacts", "data-viz"])
        expect(
            page.locator('button.panel-rail-button[data-panel-id="data-viz"].active')
        ).to_have_count(1, timeout=10_000)
        _wait_for_service_tabs(base_url, ["artifacts", "data-viz"])
        _wait_for_fetch_quiescence(page)

        # Act — arrange artifacts | the unhealthy panel.
        r = requests.post(
            f"{base_url}/api/panel-op/open-set",
            json={
                "request_id": "agent-arrange-broken",
                "members": ["artifacts", broken["id"]],
                "as": "layout",
                "layout": _SIDE_BY_SIDE,
            },
            timeout=10,
        )
        assert r.status_code == 200, r.text
        result = r.json()
        assert result["status"] == "ok", result

        # Focus falls back to the first healthy listed tile.
        expect(
            page.locator('button.panel-rail-button[data-panel-id="artifacts"].active')
        ).to_have_count(1, timeout=10_000)
        page.wait_for_timeout(1_000)

        # No ghost tile for the panel that cannot fill one — and its rail entry
        # is still there, so the arrangement cost the operator nothing.
        assert _client_service_tabs(page) == ["artifacts"], (
            f"an unhealthy panel was given an empty tile: {_client_layout(page)}"
        )
        assert _service_keys(result["layout"]) == ["artifacts"], result
        assert broken["id"] in _rail_ids(page), _rail_ids(page)
        expect(_service_tab(page, "Artifacts")).to_have_count(1)
        assert _service_tab(page, broken["id"]).count() == 0
        assert _wait_for_service_tabs(base_url, ["artifacts"])["layout_dock"] is True

        page.close()


# ===========================================================================
# (e) Preset parity: the human click and the agent call are one operation
# ===========================================================================

#: web.presets in the loader's entry shape: a name, its members and its layout —
#: here two tiles side by side.
_PRESETS = [
    {
        "name": "Injection",
        "panels": ["artifacts", "data-viz"],
        "layout": {
            "split": "row",
            "items": [
                {"tabs": ["artifacts"], "showing": "artifacts", "size": 0.5},
                {"tabs": ["data-viz"], "showing": "data-viz", "size": 0.5},
            ],
        },
    }
]


def _workspace_signature(page: Page) -> dict:
    """What an operator would see: the layout's shape, rail membership, focus."""
    return {
        "shape": _shape(_client_layout(page)),
        "rail": _rail_ids(page),
        "active": _active_rail_id(page),
    }


def _apply_preset_by_click(page: Page) -> None:
    """Apply the "Injection" layout the way a human does — the "+" popover."""
    page.locator("#panel-add-btn").click()
    page.locator(".panel-add-item", has_text="Injection").click()


def test_preset_click_and_agent_preset_call_are_the_same_operation(tmp_path, chromium_browser):
    """A "Layouts" click and the agent's open-arrangement produce one workspace.

    Both paths are literally one route now: the human click POSTs
    ``/api/panel-op/open-arrangement`` (panel-presets.js through
    requestPanelOp), the agent POSTs the same route, and the server resolves
    the preset and relays it to the dock client as one ``panel_op``. The claim
    to prove is equality of the RESULT: the same arranged layout, the same
    focus, and the same pruned rail — a preset is exclusive, so the panel it
    omits leaves the launcher rail in both paths.

    Each path runs against its own freshly-booted server so neither inherits the
    other's state; the two signatures are then compared directly.

    The signature also carries the server's own ``active``/``visible``, because a
    preset names no focus target and a focus-less arrangement is exactly where
    the recorded focus can go stale: leaving ``active_panel`` where it was would
    have it name a panel the very same ``GET /api/panels`` response no longer
    lists as ``visible``, which is the pair every consumer of that response reads
    together. Only a real click can prove the human path lands the same way.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    def _boot(page_action) -> dict:
        with _live_server(
            workspace,
            enabled_panels={"artifacts"},
            custom_panels=[_DATA_VIZ, _SCOPE],
            presets=_PRESETS,
            project_cwd=str(workspace),
        ) as (base_url, _app):
            page = _open_page(chromium_browser, base_url)
            expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=15_000)
            # scope starts in the rail; the preset omits it, so its removal is
            # what makes the pruning observable.
            assert "scope" in _rail_ids(page)

            # Focus scope before applying the preset, as a new tab of the
            # artifacts tile, so the recorded focus is on a panel the
            # arrangement is about to close. Starting focused on a preset
            # MEMBER instead would let a stale active_panel still name a
            # visible panel, and the active-in-visible check below would pass
            # for the wrong reason.
            rail_entry(page, "scope", enabled_only=True).click()
            _wait_for_client_tiles(page, ["artifacts", "scope"])
            _wait_for_active(base_url, "scope")
            # _wait_for_active proves A focus write landed, not the LAST one:
            # the click's POSTs are fire-and-forget, so a straggler can arrive
            # AFTER the arrangement below, hit the membership-adding branch
            # (the arrange just pruned scope), and resurrect the pruned panel —
            # diverging layout, rail, and active all at once. Quiesce first.
            _wait_for_fetch_quiescence(page)
            _wait_for_service_tabs(base_url, ["artifacts", "scope"])
            posts = _track_panel_posts(page)

            page_action(page, base_url)

            _wait_for_client_tiles(page, ["artifacts", "data-viz"])
            expect(page.locator('button.panel-rail-button[data-panel-id="scope"]')).to_have_count(
                0, timeout=10_000
            )
            _wait_for_fetch_quiescence(page)
            signature = _workspace_signature(page)
            server = _wait_for_service_tabs(base_url, ["artifacts", "data-viz"])
            signature["server_shape"] = _shape(server["layout"])
            signature["server_active"] = server["active"]
            # Order is not part of the claim — `visible` keeps rail order — so
            # membership is compared as a set.
            signature["server_visible"] = sorted(server["visible"])
            signature["commands"] = sorted(
                {e for e in _endpoints(posts) if e not in ("panel-layout", "panel-op-result")}
            )
            page.close()
            return signature

    human = _boot(lambda page, _base: _apply_preset_by_click(page))
    agent = _boot(
        lambda _page, base: requests.post(
            f"{base}/api/panel-op/open-arrangement",
            json={"request_id": "agent-preset-1", "name": "Injection"},
            timeout=10,
        ).raise_for_status()
    )

    # The human click went through the very route the agent calls.
    assert "panel-op/open-arrangement" in human["commands"], human
    human_result = {k: v for k, v in human.items() if k != "commands"}
    agent_result = {k: v for k, v in agent.items() if k != "commands"}
    assert human_result == agent_result, f"preset paths diverged\nhuman: {human}\nagent: {agent}"
    # And the shared result is the preset's own layout: its two tiles side by
    # side, left of the terminal, the rail pruned to them (plus the terminal
    # anchor).
    assert human["shape"] == "row[artifacts,data-viz,terminal]", human
    assert human["server_shape"] == human["shape"], human
    assert "scope" not in human["rail"], human
    assert set(human["rail"]) == {"terminal", "data-viz", "artifacts"}, human
    assert human["server_visible"] == ["artifacts", "data-viz"], human

    # The recorded focus moved with the arrangement rather than being stranded
    # on the panel the preset just closed. Asserted as membership of the SAME
    # response's visible list, since that is the inconsistency a consumer would
    # actually hit — not merely that active changed to some particular id.
    assert human["server_active"] in human["server_visible"], (
        f"a preset click left the server's active panel outside visible: {human}"
    )


# ===========================================================================
# (f) The vertical axis of the reading order
# ===========================================================================


def test_vertical_split_reports_column_before_its_neighbour(tmp_path, chromium_browser):
    """A stacked column is reported as a column, read top-to-bottom first.

    The report is the dock layout tree itself now, so a genuinely 2D arrangement
    is reported with its orientation: with ``scope`` stacked under
    ``artifacts`` and ``data-viz`` full height to their right, the layout is
    ``row[column[artifacts,scope],data-viz,terminal]``, and its reading order
    walks down the column before going across —
    ``[artifacts, scope, data-viz]``, not the row-major ``[artifacts,
    data-viz, scope]`` a raster scan of the screen would produce.

    Only a real vertical split, made through the rail's own drag-and-drop,
    proves the dock builds that tree. The geometry is asserted first, so a drop
    that landed against the dock edge instead of inside the tile fails as the
    wrong *arrangement* rather than silently as the wrong *order*.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_DATA_VIZ, _SCOPE],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=15_000)
        # Three columns: artifacts | data-viz | terminal.
        _open_beside(page, "data-viz")
        _wait_for_client_tiles(page, ["artifacts", "data-viz"])

        # Act — drop scope into the bottom edge band of the artifacts tile (80
        # px above its content's bottom: inside the tile's edge band, 31 % of
        # its height, clear of the dock's 48 px bottom strip, which is an outer
        # drop), splitting that tile vertically rather than the dock.
        point = page.evaluate(
            _CONTENT_POINT_JS, {"key": "artifacts", "fx": 0.5, "fy": 1.0, "dy": -80}
        )
        assert point is not None
        drag = page.evaluate(_RAIL_DRAG_JS, {"panelId": "scope", **point})
        assert "error" not in drag, drag
        assert drag["shieldRose"] is True, f"the rail entry's dragstart raised no shield: {drag}"
        assert drag["preview"] == "split", drag
        expect(_service_tab(page, "SCOPE")).to_have_count(1, timeout=10_000)
        layout = _wait_for_client_tiles(page, ["artifacts", "scope", "data-viz"])
        page.wait_for_timeout(300)

        # Assert the arrangement really is a column beside a full-height tile.
        groups = {g["tabs"][0]: g for g in _dock_groups(page) if len(g["tabs"]) == 1}
        artifacts, scope, data_viz = groups["Artifacts"], groups["SCOPE"], groups["DATA VIZ"]
        assert abs(scope["x"] - artifacts["x"]) <= 2, groups
        assert abs(scope["w"] - artifacts["w"]) <= 2, groups
        assert scope["y"] > artifacts["y"], groups
        assert data_viz["x"] > artifacts["x"] + artifacts["w"] - 2, groups
        assert data_viz["h"] > artifacts["h"] + 2, (
            f"data-viz should span both rows of the column: {groups}"
        )

        # Assert the reported tree and its reading order: down, then across.
        assert _shape(layout) == "row[column[artifacts,scope],data-viz,terminal]", _shape(layout)
        state = _wait_for_service_tabs(base_url, ["artifacts", "scope", "data-viz"])
        assert _shape(state["layout"]) == _shape(layout), state

        page.close()


# ===========================================================================
# (g) The boot window: never publish a layout the restore is about to replace
# ===========================================================================


def test_slow_restore_reports_nothing_before_the_layout_is_final(tmp_path, chromium_browser):
    """No occupancy is reported until the stored layout has actually been applied.

    The dock boots into a default arrangement and only restores the operator's
    stored layout after an ``/api/panels`` round trip. A reporter armed on a
    timer would, on a slow response, publish that pre-restore arrangement — a
    dock layout with no service tab, an empty workspace — and a hook reading the
    layout would take it as fact.
    The reporter therefore arms on the restore itself, not on a deadline.

    That gap is a network latency, so the test makes it one: ``/api/panels`` is
    held open far longer than the reporter's settle window, and every layout
    POST the page makes is timestamped against the moment that response landed.
    A report before it is the bug; a report after it is the design.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        project_cwd=str(workspace),
        panels_delay=2.0,
    ) as (base_url, _app):
        page = chromium_browser.new_page()
        posts = _track_panel_posts(page)
        panels_response: list[float] = []

        def _record_response(resp):
            if resp.request.method == "GET" and resp.url.rsplit("?", 1)[0].endswith("/api/panels"):
                panels_response.append(time.monotonic())

        page.on("response", _record_response)

        page.goto(base_url, wait_until="domcontentloaded")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=15_000)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=20_000)
        # Outlast the settle window plus a generous margin so a late report is
        # still counted rather than raced past.
        page.wait_for_timeout(2_000)

        assert panels_response, "the page never fetched /api/panels"
        restore_seam = min(panels_response)
        early = [
            (endpoint, body)
            for when, endpoint, body in posts
            if endpoint == "panel-layout" and when < restore_seam
        ]
        assert not early, f"occupancy was reported before the layout was restored: {early}"

        # Reporting is armed, not merely silent: the settled layout does land,
        # and it is the real one rather than the pre-restore empty workspace.
        reports = _layout_reports(posts)
        assert reports, "no occupancy report was ever sent after the restore"
        assert all(r and r["dock"] is True for r in reports), reports
        assert all(_service_keys(r["layout"]) for r in reports), (
            f"an empty pre-restore workspace was reported: {reports}"
        )
        last = reports[-1]
        assert _shape(last["layout"]) == "row[artifacts,terminal]", reports
        assert isinstance(last["client_id"], str) and last["client_id"], last
        state = _panels_state(base_url)
        assert _service_keys(state["layout"]) == ["artifacts"], state
        assert _shape(state["layout"]) == "row[artifacts,terminal]", state

        page.close()


# ===========================================================================
# (h) Three-state occupancy: never reported / unknown / known
# ===========================================================================


def test_occupancy_read_back_distinguishes_unknown_from_empty(tmp_path, chromium_browser):
    """``layout`` tells "nothing is open" apart from "nobody can say".

    Three states share one field, and collapsing any two of them would let a
    hook assert something nobody observed:

    * **never reported** — no client has checked in; all three fields null.
    * **known empty** — a dock client reports that the operator closed every
      service tile: a layout holding only the terminal, with ``dock: true``.
    * **unknown** — a client without a dock shell is watching but cannot see
      tile order. It sends a presence report whose ``layout`` the server must
      record as *null*, not as an empty workspace, or a fallback client would
      overwrite a real list with a confident lie.

    The dock-less client is real, not simulated: the dockview bundle is blocked
    for that page, so the shell genuinely never arrives and the client reaches
    its own give-up branch.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        # --- never reported: nobody has looked at the screen yet.
        fresh = _panels_state(base_url)
        assert fresh["layout"] is None, fresh
        assert fresh["layout_age_s"] is None, fresh
        assert fresh["layout_dock"] is None, fresh

        # --- known empty: a dock client closes the last service tile.
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=15_000)
        _wait_for_service_tabs(base_url, ["artifacts"])
        _close_tile(page, "artifacts")
        known_empty = _wait_for_service_tabs(base_url, [])
        assert known_empty["layout_dock"] is True, known_empty
        assert _shape(known_empty["layout"]) == "terminal", known_empty
        assert isinstance(known_empty["layout_age_s"], float), known_empty
        page.close()

        # --- unknown: a client with no dock shell at all.
        dockless = chromium_browser.new_page()
        dockless.route("**/dockview-core*", lambda route: route.abort())
        dockless.goto(base_url, wait_until="domcontentloaded")
        expect(
            dockless.locator('button.panel-rail-button[data-panel-id="artifacts"]')
        ).to_be_attached(timeout=15_000)
        assert (
            dockless.evaluate(
                "async () => { const m = await import('/static/js/dock-workspace.js');"
                " return m.getDockApi(); }"
            )
            is None
        ), "the dock shell was supposed to be unavailable on this page"

        # The give-up branch waits out its fast poll budget (~4.5s) before it
        # concludes there is no dock and says so.
        deadline = time.monotonic() + 25.0
        state: dict = {}
        while time.monotonic() < deadline:
            state = _panels_state(base_url)
            if state["layout_dock"] is False:
                break
            dockless.wait_for_timeout(300)
        assert state.get("layout_dock") is False, f"no dock-less report arrived: {state}"
        assert state["layout"] is None, (
            f"a dock-less client's presence report was recorded as an empty workspace: {state}"
        )
        assert isinstance(state["layout_age_s"], float), state

        dockless.close()


# ===========================================================================
# (e) Human gestures stay local: no focus command leaks, no cross-client echo
# ===========================================================================


def test_tile_close_with_two_tiles_commands_nothing(tmp_path, chromium_browser):
    """A tile "×" with a SECOND service tile open still commands nothing.

    The single-tile variant is pinned in the sibling suite; with two tiles the
    close (of data-viz's only tab, which closes its tile) makes dockview
    auto-activate the surviving service tile, and that
    activation runs outside any human focus gesture — it must stay inside the
    echo guard exactly like retireTile's removal does, or the close leaks a
    ``setPanelFocus`` command the design says it must not send. The assertion
    window is generous because the leaked POST arrives asynchronously, well
    after the tab is gone.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(workspace, enabled_panels={"artifacts"}, custom_panels=[_DATA_VIZ]) as (
        base_url,
        _app,
    ):
        page = _open_page(chromium_browser, base_url)
        _open_beside(page, "data-viz")
        _wait_for_client_tiles(page, ["artifacts", "data-viz"])
        page.wait_for_timeout(800)  # drain boot/open-beside traffic

        posts = _track_panel_posts(page)
        _close_tile(page, "data-viz")
        page.wait_for_timeout(1500)  # the leak arrives asynchronously

        commands = [e for e in _endpoints(posts) if e != "panel-layout"]
        assert _client_service_tabs(page) == ["artifacts"], _client_service_tabs(page)
        assert commands == [], f"a human tile close must not command, got {commands}"

        page.close()


def test_human_focus_stays_local_to_the_gesturing_client(tmp_path, chromium_browser):
    """One operator's focus gestures never move another client's workspace.

    Client B opens a second tile (its activation tail reports the focus via
    ``setPanelFocus`` — a human gesture). Client A must keep its own active
    panel and its own tile set: human focus is a REPORT the server mirrors for
    the agent's benefit, never a command broadcast back to other clients
    (the contract stated in panel-commands.js and the collaborative-panels
    design). The server-side mirror is asserted off the same gesture, so this
    cannot pass by the report silently not landing.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(workspace, enabled_panels={"artifacts"}, custom_panels=[_DATA_VIZ]) as (
        base_url,
        _app,
    ):
        page_a = _open_page(chromium_browser, base_url)
        _wait_for_client_tiles(page_a, ["artifacts"])
        page_b = _open_page(chromium_browser, base_url)
        _wait_for_client_tiles(page_b, ["artifacts"])
        assert _active_rail_id(page_a) == "artifacts"

        # B's human gesture: open data-viz beside (activation reports focus).
        _open_beside(page_b, "data-viz")
        _wait_for_client_tiles(page_b, ["artifacts", "data-viz"])

        # The server mirrors the gesture for the agent's gaze...
        _wait_for_active(base_url, "data-viz")
        page_a.wait_for_timeout(1000)  # ...but no echo may reach client A:

        assert _active_rail_id(page_a) == "artifacts", (
            "client B's human focus gesture moved client A's active panel"
        )
        assert _client_service_tabs(page_a) == ["artifacts"], (
            "client B's human focus gesture changed client A's tiles"
        )

        page_a.close()
        page_b.close()


# ===========================================================================
# (i) The terminal is never an agent's to move: every verb refuses it
# ===========================================================================


def test_agent_ops_on_the_terminal_are_refused_and_change_nothing(tmp_path, chromium_browser):
    """Every agent verb that would move, close or arrange the terminal is refused.

    The terminal-refusal contract, carried onto the new verbs: closing its tab
    (``close-tab``), closing its tile (``close-tile``), dragging its tab or its
    tile (``drop``) and opening it (``open``) are each refused
    ``terminal_not_agent_movable``; a set or arrangement naming it
    (``open-set``) is ``terminal_in_layout``. Each refusal carries the current
    ``rev`` and changes nothing — the server's layout and rev, and the live
    client's layout, are exactly as before. The person's paths to the same
    terminal (its rail entry's close and reopen, the "Close terminal tile" row)
    stay, and are proven in the sibling suite.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_DATA_VIZ],
        project_cwd=str(workspace),
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "Artifacts")).to_have_count(1, timeout=15_000)
        state = _wait_for_service_tabs(base_url, ["artifacts"])
        _wait_for_fetch_quiescence(page)
        layout, rev = state["layout"], state["rev"]
        client_before = _client_layout(page)
        term_tile = _tile_of(layout, "terminal")
        service_tile = _tile_of(layout, "artifacts")
        assert term_tile and service_tile, layout

        ops = [
            ("close-tab", {"key": "terminal"}, "terminal_not_agent_movable"),
            ("close-tile", {"base_rev": rev, "tile": term_tile}, "terminal_not_agent_movable"),
            (
                "drop",
                {
                    "base_rev": rev,
                    "source": {"tab": "terminal"},
                    "target": {"tile": service_tile, "zone": "centre"},
                },
                "terminal_not_agent_movable",
            ),
            (
                "drop",
                {
                    "base_rev": rev,
                    "source": {"tile": term_tile},
                    "target": {"tile": service_tile, "edge": "left"},
                },
                "terminal_not_agent_movable",
            ),
            ("open", {"member": "terminal"}, "terminal_not_agent_movable"),
            (
                "open-set",
                {"members": ["terminal", "artifacts"], "as": "tabs"},
                "terminal_in_layout",
            ),
        ]
        for n, (route, body, reason) in enumerate(ops):
            r = requests.post(
                f"{base_url}/api/panel-op/{route}",
                json={"request_id": f"terminal-op-{n}", **body},
                timeout=10,
            )
            assert r.status_code in (409, 422), (route, r.status_code, r.text)
            refused = r.json()
            assert refused["status"] == "refused", (route, refused)
            assert refused["reason"] == reason, (route, refused)
            assert refused["rev"] == rev, (route, refused)

        # Nothing changed, on the server or on screen.
        page.wait_for_timeout(800)
        after = _panels_state(base_url)
        assert after["layout"] == layout and after["rev"] == rev, (state, after)
        assert _client_layout(page) == client_before, (client_before, _client_layout(page))
        expect(_service_tab(page, "Agent")).to_have_count(1)

        page.close()
