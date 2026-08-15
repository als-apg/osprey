"""Browser tests: the collaborative-panels contract between dock and server.

``test_panels_browser.py`` proves the docked workspace itself (tiles, drags,
persistence, the echo guard). This suite proves the *two-party* contract layered
on top of it — the one where the agent finally sees, and can rearrange, what is
actually on screen:

  * the occupancy REPORT (``POST /api/panel-layout`` → ``GET /api/panels``'s
    ``open_tiles`` / ``open_tiles_age_s`` / ``open_tiles_dock``),
  * the polite agent SWITCH (``panel-focus`` with ``source: agent`` opens a tile
    *beside* the operator's instead of evicting it),
  * the declarative ARRANGE (``POST /api/panel-arrange``, shared by the agent's
    ``arrange_workspace`` and the human "Layouts" click).

Every assertion here needs a real dockview grid, a real SSE stream and real
client-side fetches, so none of it is reachable from the unit suites:

  * ``open_tiles`` is serialized from the live grid tree — a Python test can post
    any list it likes, but only a browser can prove the list the dock produces.
  * the "no echo loop" criteria are about POSTs a client does **not** make.
    They are asserted by intercepting the page's own requests; a mocked fetch
    proves only that the mock was not called.
  * the vertical axis of the reading order (a column reported top-to-bottom
    *before* what sits beside it) has no serialized representation to assert
    against — a persisted branch carries no orientation, so the ordering can
    only be pinned by making a real vertical split and reading the result back.

Run:
    uv run pytest tests/interfaces/web_terminal/test_panels_collab_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import requests

from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _free_port, _run_app_server

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
_LABELS = {"artifacts": "WORKSPACE", "data-viz": "DATA VIZ", "scope": "SCOPE"}


def _unhealthy_panel(panel_id: str = "beam-cam") -> dict:
    """A custom panel that polls a port nothing listens on — permanently unhealthy.

    A ``healthEndpoint`` is what makes a panel poll at all; pointing it at a
    closed port makes every poll fail, so the panel stays unhealthy for the
    lifetime of the test instead of racing a backend into readiness.
    """
    return {
        "id": panel_id,
        "label": panel_id.upper(),
        "url": f"http://127.0.0.1:{_free_port()}",
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

    ``presets`` seeds ``app.state.panel_presets`` — the ``web.presets`` layouts
    the "+" popover's "Layouts" section renders and ``/api/panel-arrange``
    resolves by name — before the first page load, so both preset paths see it.

    ``panels_delay`` holds every ``GET /api/panels`` open for that many seconds.
    That is the boot window the layout reporter must not report inside: the
    stored-layout restore only happens after this round trip, so a reporter
    armed on a timer would publish the pre-restore arrangement as fact.

    Yields:
        (base_url: str, app: FastAPI) — live server address and the app object.
    """
    if custom_panels is None:
        custom_panels = []

    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled_panels, custom_panels, None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(),
        ),
    ):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])

        if panels_delay:

            @app.middleware("http")
            async def _delay_panels(request, call_next):
                if request.url.path == "/api/panels" and request.method == "GET":
                    await asyncio.sleep(panels_delay)
                return await call_next(request)

        with _run_app_server(app) as base_url:
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

_DISMISS_RAIL_HINT = (
    "try { localStorage.setItem('osprey-rail-hint-dismissed-v1', '1') } catch (e) {}"
)


def _new_page(browser) -> Page:
    """A page with the one-time rail hint pre-dismissed (it would eat clicks)."""
    page = browser.new_page()
    page.add_init_script(_DISMISS_RAIL_HINT)
    return page


def _open_page(browser, base_url: str) -> Page:
    """Open a page and wait for the rail and the dock grid to be up."""
    page = _new_page(browser)
    page.goto(base_url, wait_until="domcontentloaded")
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=15_000
    )
    expect(page.locator(".dv-groupview").first).to_be_visible(timeout=15_000)
    page.evaluate("document.getElementById('welcome-overlay')?.remove()")
    return page


# ---------------------------------------------------------------------------
# Dock DOM / module helpers
# ---------------------------------------------------------------------------

# Every dockview group with its tab labels and pixel rectangle. The terminal
# tile's tab carries no accessible name (it adopts the live .terminal-header),
# so it is reported under the synthetic "SESSION" handle, as in the sibling
# suite.
_GROUPS_JS = r"""() => [...document.querySelectorAll('.dv-groupview')].map(g => {
  const tabs = [...g.querySelectorAll('.dv-tab')].map(t =>
    t.querySelector('.terminal-header')
      ? 'SESSION'
      : (t.querySelector('.tile-tab')?.getAttribute('aria-label') ?? ''));
  const r = g.getBoundingClientRect();
  return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width),
           h: Math.round(r.height), tabs };
})"""

# The client's OWN view of tile occupancy, straight from the serializer the
# reporter posts. Asserting on this (rather than only on the server read-back)
# separates "the dock is arranged wrongly" from "the report never landed".
_OPEN_TILES_JS = r"""async () => {
  const sync = await import('/static/js/dock-sync.js');
  const dock = await import('/static/js/dock-workspace.js');
  return sync.serializeOpenTiles(dock.getDockApi());
}"""

# Stamp / read a marker on the dockview panel OBJECT behind a service tile.
# dockPanelAt MOVES an already-docked placeholder by remove+re-add, so a
# surviving stamp is proof the tile was left strictly alone — a stronger claim
# than "a tab with that label is still on screen".
_STAMP_TILES_JS = r"""async (ids) => {
  const dock = await import('/static/js/dock-workspace.js');
  const api = dock.getDockApi();
  const out = {};
  for (const id of ids) {
    const panel = api && api.getPanel('iframe:' + id);
    if (panel) { panel.__collabStamp = 'stamped:' + id; out[id] = true; } else out[id] = false;
  }
  return out;
}"""

_READ_STAMPS_JS = r"""async (ids) => {
  const dock = await import('/static/js/dock-workspace.js');
  const api = dock.getDockApi();
  const out = {};
  for (const id of ids) {
    const panel = api && api.getPanel('iframe:' + id);
    out[id] = panel ? (panel.__collabStamp ?? null) : null;
  }
  return out;
}"""


def _dock_groups(page: Page) -> list[dict]:
    return page.evaluate(_GROUPS_JS)


def _client_open_tiles(page: Page) -> list[str] | None:
    return page.evaluate(_OPEN_TILES_JS)


def _wait_for_client_tiles(page: Page, expected: list[str], timeout: float = 10_000) -> None:
    """Block until the page's own serializer reports exactly ``expected``.

    Auto-waiting on the serializer rather than on a tab locator: the assertion
    subject is the tile list the reporter would post, and a tab appearing is not
    the same instant as the grid settling on its final shape.
    """
    page.wait_for_function(
        """async (want) => {
            const sync = await import('/static/js/dock-sync.js');
            const dock = await import('/static/js/dock-workspace.js');
            return JSON.stringify(sync.serializeOpenTiles(dock.getDockApi())) === want;
        }""",
        arg=json.dumps(expected),
        timeout=timeout,
    )


def _service_tab(page: Page, label: str):
    """The dockview tile strip whose accessible name is exactly ``label``."""
    strip = page.locator(f'.tile-tab[aria-label="{label}"]')
    return page.locator(".dv-tab").filter(has=strip)


def _rail_ids(page: Page) -> list[str]:
    """Every rail entry's panel id, in rail order (the terminal anchor leads)."""
    return page.eval_on_selector_all(".panel-rail-button", "els => els.map(e => e.dataset.panelId)")


def _active_rail_id(page: Page) -> str | None:
    return page.evaluate(
        "() => document.querySelector('.panel-rail-button.active')?.dataset.panelId ?? null"
    )


def _open_beside(page: Page, panel_id: str) -> None:
    """Open ``panel_id`` as a NEW tile beside the active one via the rail's ⊞.

    A plain rail click REPLACES the focused tile's panel (one panel per tile),
    so every test here that needs side-by-side tiles grows the layout through
    the entry's hover ⊞ corner — the human open-beside verb.
    """
    entry = page.locator(f'button.panel-rail-button[data-panel-id="{panel_id}"]')
    entry.hover()
    entry.locator(".panel-rail-beside").click()
    expect(_service_tab(page, _LABELS[panel_id])).to_have_count(1, timeout=5_000)


def _close_tile(page: Page, panel_id: str) -> None:
    """Close a service tile through its header "×" — the human close gesture."""
    label = _LABELS[panel_id]
    _service_tab(page, label).locator(".tile-tab-close").click()
    expect(_service_tab(page, label)).to_have_count(0, timeout=5_000)


# ---------------------------------------------------------------------------
# Request interception
# ---------------------------------------------------------------------------


def _track_panel_posts(page: Page) -> list[tuple[float, str, dict | None]]:
    """Record ``(monotonic_time, endpoint, parsed_body)`` for every panel POST.

    The no-loop and no-echo criteria of this feature are all statements about
    requests the client must NOT make, so the request stream is the instrument.
    Bodies are parsed because a layout report's *content* is the assertion (an
    ``{"tiles": [], "dock": true}`` during boot is the specific failure).
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


def _wait_for_open_tiles(base_url: str, expected, timeout: float = 15.0) -> dict:
    """Poll ``GET /api/panels`` until ``open_tiles`` equals ``expected``.

    Returns the whole payload so the caller can assert the freshness fields off
    the same read that observed the occupancy.
    """
    deadline = time.monotonic() + timeout
    state: dict = {}
    while time.monotonic() < deadline:
        state = _panels_state(base_url)
        if state["open_tiles"] == expected:
            return state
        time.sleep(0.2)
    raise AssertionError(f"open_tiles never became {expected!r}; last read: {state!r}")


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


def test_human_tile_close_updates_open_tiles_and_freshness(tmp_path, chromium_browser):
    """Closing a tile by hand updates ``open_tiles``, with honest freshness fields.

    Tile occupancy used to live entirely in the browser: the operator could close
    a panel and the agent would keep reading it off the rail as "visible". Here
    two tiles are opened, the server is checked to see both in reading order, one
    is closed through its header "×", and the server must converge on the
    remaining tile — reported by a client that has a dock shell (``dock: true``)
    and with an age that reflects the close that just happened, not the boot.

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
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=15_000)
        _open_beside(page, "data-viz")

        # Both tiles reach the server, in the order they read off the screen.
        before = _wait_for_open_tiles(base_url, ["artifacts", "data-viz"])
        assert before["open_tiles_dock"] is True, before
        rail_before = _rail_ids(page)

        # Act — the operator closes the data-viz tile.
        _close_tile(page, "data-viz")

        # Assert — occupancy converges, freshness describes THIS change.
        after = _wait_for_open_tiles(base_url, ["artifacts"])
        assert after["open_tiles_dock"] is True, after
        assert isinstance(after["open_tiles_age_s"], float), after
        assert 0.0 <= after["open_tiles_age_s"] < 10.0, (
            f"age should describe the close that just landed: {after}"
        )

        # …and membership is untouched: a closed tile is still one rail click away.
        assert _rail_ids(page) == rail_before, _rail_ids(page)
        assert set(after["visible"]) == set(before["visible"]), (before, after)

        page.close()


# ===========================================================================
# (b) The polite agent switch
# ===========================================================================


def test_agent_switch_opens_beside_and_posts_nothing_back(tmp_path, chromium_browser):
    """An agent switch onto an undocked rail member opens BESIDE, silently.

    Two halves, both only observable in a browser:

    * **Non-destructive.** The operator has two tiles open; the agent switches to
      a third panel that is a rail member but holds no tile. The new tile appears
      and the two existing tiles survive *as the same dockview panels* — proven
      with a marker stamped on the panel objects beforehand, so a remove-and-
      re-add (which would discard the operator's tile) cannot pass as survival.
    * **No echo.** Applying the switch drives a placeholder add and a
      programmatic active-panel change, both of which look exactly like human
      dock gestures to dockview. If either leaked back out as a
      ``POST /api/panel-focus`` the client would be talking to itself; only the
      occupancy report is allowed to follow an applied switch.
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
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=15_000)
        _open_beside(page, "data-viz")
        # scope is a rail member from config but holds no tile — the exact state
        # open_panel's "open beside" branch exists for.
        assert "scope" in _rail_ids(page)
        expect(_service_tab(page, "SCOPE")).to_have_count(0)

        stamped = page.evaluate(_STAMP_TILES_JS, ["artifacts", "data-viz"])
        assert stamped == {"artifacts": True, "data-viz": True}, stamped

        # Drain the open-beside gesture's own POSTs before counting.
        _wait_for_open_tiles(base_url, ["artifacts", "data-viz"])
        page.wait_for_timeout(800)
        posts = _track_panel_posts(page)

        # Act — the agent's open_panel, as the MCP tool issues it.
        r = requests.post(
            f"{base_url}/api/panel-focus",
            json={"panel": "scope", "source": "agent"},
            timeout=10,
        )
        assert r.status_code == 200, r.text

        # The panel is on screen, in a tile of its own…
        expect(_service_tab(page, "SCOPE")).to_have_count(1, timeout=10_000)
        expect(
            page.locator('button.panel-rail-button[data-panel-id="scope"].active')
        ).to_have_count(1, timeout=10_000)

        # …and neither prior tile was disturbed: same tabs, same panel objects.
        survived = page.evaluate(_READ_STAMPS_JS, ["artifacts", "data-viz"])
        assert survived == {
            "artifacts": "stamped:artifacts",
            "data-viz": "stamped:data-viz",
        }, f"an agent switch replaced an operator's tile: {survived}"
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1)
        expect(_service_tab(page, "DATA VIZ")).to_have_count(1)

        # The client reported the new occupancy and said nothing else.
        _wait_for_open_tiles(base_url, ["artifacts", "data-viz", "scope"])
        page.wait_for_timeout(800)
        assert set(_endpoints(posts)) <= {"panel-layout"}, (
            f"an applied agent switch echoed back out: {_endpoints(posts)}"
        )

        page.close()


# ===========================================================================
# (c) Arrange convergence, and the absence of a report loop
# ===========================================================================


def test_arrange_converges_from_two_layouts_without_a_report_loop(tmp_path, chromium_browser):
    """One arrange, two differently-arranged clients, one converged workspace.

    The arrangement is a declarative end state, so the workspace a client starts
    from must not change where it ends up. Two pages are given deliberately
    different layouts (one tile vs three in the opposite order), then a single
    ``POST /api/panel-arrange`` names two tiles and a focus target; both pages
    must land on exactly those tiles, in that order, focused the same way.

    The second half is the convergence criterion. Applying an arrangement makes
    the dock emit layout changes, each of which the reporter would post — and
    every report the server *stored* would be a fact a hook could read. If the
    dedupe on either end were missing, the two clients would ping-pong reports
    indefinitely; the contract is at most ONE report per client after the
    arrangement settles. That is a statement about requests, so it is measured
    by intercepting them.
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
        expect(_service_tab(page_a, "WORKSPACE")).to_have_count(1, timeout=15_000)

        # Client B: three tiles, deliberately in a different order.
        page_b = _open_page(chromium_browser, base_url)
        expect(_service_tab(page_b, "WORKSPACE")).to_have_count(1, timeout=15_000)
        _open_beside(page_b, "scope")
        _open_beside(page_b, "data-viz")

        # B's open-beside gestures report their focus but broadcast nothing —
        # human focus stays local — so A's boot layout (artifacts alone) is
        # untouched by B's setup and needs no re-establishing.
        _wait_for_client_tiles(page_a, ["artifacts"])
        assert _client_open_tiles(page_b) == ["artifacts", "scope", "data-viz"], _client_open_tiles(
            page_b
        )

        # Let both clients' own reports drain, then watch what they say next.
        page_a.wait_for_timeout(1_200)
        page_b.wait_for_timeout(1_200)
        posts_a = _track_panel_posts(page_a)
        posts_b = _track_panel_posts(page_b)

        # Act — one declarative arrangement for everyone.
        r = requests.post(
            f"{base_url}/api/panel-arrange",
            json={"tiles": ["data-viz", "artifacts"], "focus": "artifacts", "source": "agent"},
            timeout=10,
        )
        assert r.status_code == 200, r.text

        # Assert — identical end state on both clients, whatever they started from.
        for page in (page_a, page_b):
            _wait_for_client_tiles(page, ["data-viz", "artifacts"])
            expect(
                page.locator('button.panel-rail-button[data-panel-id="artifacts"].active')
            ).to_have_count(1, timeout=10_000)
            expect(_service_tab(page, "SCOPE")).to_have_count(0, timeout=10_000)
            # Left-to-right on screen, not merely in the serialization.
            service = [g for g in _dock_groups(page) if "SESSION" not in g["tabs"]]
            order = [g["tabs"][0] for g in sorted(service, key=lambda g: g["x"])]
            assert order == ["DATA VIZ", "WORKSPACE"], order

        # The arrangement reached the server as an observation, not just a command.
        _wait_for_open_tiles(base_url, ["data-viz", "artifacts"])

        # Now the no-loop criterion: give any ping-pong a generous window to
        # show itself (a loop would be unbounded, so a few seconds is decisive).
        page_a.wait_for_timeout(2_500)
        page_b.wait_for_timeout(2_500)
        for name, posts in (("A", posts_a), ("B", posts_b)):
            reports = _layout_reports(posts)
            assert len(reports) <= 1, f"client {name} looped layout reports: {reports}"
            assert set(_endpoints(posts)) <= {"panel-layout"}, (
                f"client {name} echoed an applied arrange back out: {_endpoints(posts)}"
            )

        page_a.close()
        page_b.close()


# ===========================================================================
# (d) Health is a client-side fact: focus fallback, and no empty tiles
# ===========================================================================


def test_arrange_with_unhealthy_focus_falls_back_and_docks_no_empty_tile(
    tmp_path, chromium_browser
):
    """An arrangement naming an unhealthy panel focuses elsewhere and opens no ghost.

    Panel health only exists in the browser, so the server records the requested
    focus verbatim and the client applies the fallback rule: the requested panel
    if it is healthy, else the first healthy tile listed. Here the requested
    focus target is a panel whose health endpoint is a dead port, so focus must
    land on the other listed tile instead.

    The starting focus is deliberately on a panel the arrangement does NOT list.
    Left on the boot default, focus would already be sitting on ``artifacts``
    (the default panel auto-activates on its first healthy settle), and the
    fallback assertion would be confirming a state that was true before the
    arrangement ran — it would still pass with the healthy-fallback rule deleted.
    From a start focused on an unlisted panel, losing that rule instead clears
    the active state altogether, and the assertion fails as it should.

    The second assertion is the one that keeps the workspace honest: a tile is
    only ever docked for a panel that can fill it. An unhealthy panel's
    activation refuses, so docking a tile for it would leave a permanently empty
    rectangle on the operator's screen — worse than not opening it at all. Its
    rail entry survives either way, so nothing is lost: the panel is one click
    away the moment it comes back.
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
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=15_000)
        # Let the broken panel's first health polls fail so it is decidedly
        # unhealthy before the arrangement is applied.
        page.wait_for_timeout(1_500)
        assert (
            page.locator(f'button.panel-rail-button[data-panel-id="{broken["id"]}"]').count() == 1
        )

        # Move focus onto data-viz, which the arrangement below does not list —
        # see the docstring: from the boot default the fallback assertion would
        # be confirmatory.
        _open_beside(page, "data-viz")
        _wait_for_client_tiles(page, ["artifacts", "data-viz"])
        expect(
            page.locator('button.panel-rail-button[data-panel-id="data-viz"].active')
        ).to_have_count(1, timeout=10_000)

        # Act — arrange onto the unhealthy panel, asking for it as the focus.
        r = requests.post(
            f"{base_url}/api/panel-arrange",
            json={"tiles": ["artifacts", broken["id"]], "focus": broken["id"], "source": "agent"},
            timeout=10,
        )
        assert r.status_code == 200, r.text

        # Focus falls back to the first healthy listed tile.
        expect(
            page.locator('button.panel-rail-button[data-panel-id="artifacts"].active')
        ).to_have_count(1, timeout=10_000)
        page.wait_for_timeout(1_000)

        # No ghost tile for the panel that cannot fill one — and its rail entry
        # is still there, so the arrangement cost the operator nothing.
        assert _client_open_tiles(page) == ["artifacts"], (
            f"an unhealthy panel was given an empty tile: {_client_open_tiles(page)}"
        )
        assert broken["id"] in _rail_ids(page), _rail_ids(page)
        expect(_service_tab(page, broken["label"])).to_have_count(0)
        assert _wait_for_open_tiles(base_url, ["artifacts"])["open_tiles_dock"] is True

        page.close()


# ===========================================================================
# (e) Preset parity: the human click and the agent call are one operation
# ===========================================================================

_PRESETS = [{"name": "Injection", "panels": ["data-viz", "artifacts"]}]


def _workspace_signature(page: Page) -> dict:
    """What an operator would see: tiles in order, rail membership, focus."""
    return {
        "tiles": _client_open_tiles(page),
        "rail": _rail_ids(page),
        "active": _active_rail_id(page),
    }


def _apply_preset_by_click(page: Page) -> None:
    """Apply the "Injection" layout the way a human does — the "+" popover."""
    page.locator("#panel-add-btn").click()
    expect(page.locator(".panel-add-menu.open")).to_be_visible(timeout=5_000)
    page.locator(".panel-add-item", has_text="Injection").click()


def test_preset_click_and_agent_preset_call_are_the_same_operation(tmp_path, chromium_browser):
    """A "Layouts" click and ``arrange_workspace(preset=…)`` produce one workspace.

    Both paths resolve the same preset server-side and broadcast the same
    arrangement, rather than the browser applying it panel by panel, so the
    claim to prove is equality of the
    RESULT, not of the code path: same tiles in the same order, the same focus,
    and the same pruned rail — a preset is exclusive, so the panel it omits
    leaves the launcher rail in both paths.

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
            expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=15_000)
            # scope starts in the rail; the preset omits it, so its removal is
            # what makes the pruning observable.
            assert "scope" in _rail_ids(page)

            # Focus scope before applying the preset, so the recorded focus is
            # on a panel the arrangement is about to close. Starting focused on
            # a preset MEMBER instead would let a stale active_panel still name
            # a visible panel, and the active-in-visible check below would pass
            # for the wrong reason.
            page.locator('button.panel-rail-button[data-panel-id="scope"]').click()
            _wait_for_client_tiles(page, ["scope"])
            _wait_for_active(base_url, "scope")

            page_action(page, base_url)

            _wait_for_client_tiles(page, ["data-viz", "artifacts"])
            page.wait_for_timeout(800)
            signature = _workspace_signature(page)
            server = _wait_for_open_tiles(base_url, ["data-viz", "artifacts"])
            signature["server_open_tiles"] = server["open_tiles"]
            signature["server_active"] = server["active"]
            # Order is not part of the claim — `visible` keeps rail order while
            # `tiles` carries the preset's config order, and the two need not
            # agree — so membership is compared as a set.
            signature["server_visible"] = sorted(server["visible"])
            page.close()
            return signature

    human = _boot(lambda page, _base: _apply_preset_by_click(page))
    agent = _boot(
        lambda _page, base: requests.post(
            f"{base}/api/panel-arrange",
            json={"preset": "Injection", "source": "agent"},
            timeout=10,
        ).raise_for_status()
    )

    assert human == agent, f"preset paths diverged\nhuman: {human}\nagent: {agent}"
    # And the shared result is the preset's own semantics: its members open in
    # config order, the rail pruned to them (plus the terminal anchor).
    assert human["tiles"] == ["data-viz", "artifacts"], human
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

# Drag a rail entry INTO a specific tile's content at a relative position — the
# rail's own HTML5 drag, dispatched in-page because Playwright's mouse-based
# drag_to cannot produce HTML5 dnd events. dragstart runs panel-rail's real
# handler (payload + iframe shields); the repeated dragover with settles between
# is required because dockview resolves its drop overlay asynchronously and a
# drop immediately after a single dragover is silently ignored.
_RAIL_DRAG_INTO_TILE_JS = r"""async ({ panelId, targetPanel, fx, fy }) => {
    const dock = await import('/static/js/dock-workspace.js');
    const api = dock.getDockApi();
    const target = api.getPanel('iframe:' + targetPanel);
    const content = target.group.element.querySelector('.dv-content-container');
    const r = content.getBoundingClientRect();
    const x = r.left + r.width * fx;
    const y = r.top + r.height * fy;
    const entry = document.querySelector(
        `button.panel-rail-button[data-panel-id="${panelId}"]`);
    const dt = new DataTransfer();
    entry.dispatchEvent(new DragEvent('dragstart',
        { bubbles: true, cancelable: true, dataTransfer: dt }));
    const opts = { bubbles: true, cancelable: true, dataTransfer: dt,
                   clientX: x, clientY: y };
    const el = document.elementFromPoint(x, y) ?? content;
    const settle = () => new Promise((res) => setTimeout(res, 120));
    el.dispatchEvent(new DragEvent('dragenter', opts));
    el.dispatchEvent(new DragEvent('dragover', opts));
    await settle();
    el.dispatchEvent(new DragEvent('dragover', opts));
    await settle();
    el.dispatchEvent(new DragEvent('drop', opts));
    entry.dispatchEvent(new DragEvent('dragend', { bubbles: true, dataTransfer: dt }));
}"""


def test_vertical_split_reports_column_before_its_neighbour(tmp_path, chromium_browser):
    """A stacked column is reported top-to-bottom BEFORE the tile beside it.

    The reading order is a depth-first walk of the grid tree, so a genuinely 2D
    arrangement flattens column-first: with ``scope`` stacked under
    ``artifacts`` and ``data-viz`` full height to their right, the report is
    ``[artifacts, scope, data-viz]`` — not the row-major ``[artifacts, data-viz,
    scope]`` a raster scan of the screen would produce.

    This is the one ordering claim no unit test can pin. A serialized branch
    carries its children in axis order but records no orientation, so a
    hand-built fixture cannot state which axis it means; only a real vertical
    split, made through the rail's own drag-and-drop, produces a tree whose
    vertical branch is unambiguous. The geometry is asserted first, so a drop
    that landed against the grid edge instead of inside the tile fails as the
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
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=15_000)
        # Three columns: artifacts | data-viz | terminal.
        _open_beside(page, "data-viz")
        assert _client_open_tiles(page) == ["artifacts", "data-viz"], _client_open_tiles(page)

        # Act — drop scope into the LOWER part of the artifacts tile, splitting
        # that tile vertically rather than the grid.
        page.evaluate(
            _RAIL_DRAG_INTO_TILE_JS,
            {"panelId": "scope", "targetPanel": "artifacts", "fx": 0.5, "fy": 0.85},
        )
        expect(_service_tab(page, "SCOPE")).to_have_count(1, timeout=10_000)
        page.wait_for_timeout(500)

        # Assert the arrangement really is a column beside a full-height tile.
        groups = {g["tabs"][0]: g for g in _dock_groups(page) if len(g["tabs"]) == 1}
        artifacts, scope, data_viz = groups["WORKSPACE"], groups["SCOPE"], groups["DATA VIZ"]
        assert abs(scope["x"] - artifacts["x"]) <= 2, groups
        assert abs(scope["w"] - artifacts["w"]) <= 2, groups
        assert scope["y"] > artifacts["y"], groups
        assert data_viz["x"] > artifacts["x"] + artifacts["w"] - 2, groups
        assert data_viz["h"] > artifacts["h"] + 2, (
            f"data-viz should span both rows of the column: {groups}"
        )

        # Assert the reading order: down the column, then across.
        assert _client_open_tiles(page) == ["artifacts", "scope", "data-viz"], _client_open_tiles(
            page
        )
        _wait_for_open_tiles(base_url, ["artifacts", "scope", "data-viz"])

        page.close()


# ===========================================================================
# (g) The boot window: never publish a layout the restore is about to replace
# ===========================================================================


def test_slow_restore_reports_nothing_before_the_layout_is_final(tmp_path, chromium_browser):
    """No occupancy is reported until the stored layout has actually been applied.

    The dock boots into a default arrangement and only restores the operator's
    stored layout after an ``/api/panels`` round trip. A reporter armed on a
    timer would, on a slow response, publish that pre-restore arrangement — an
    empty workspace — and a hook reading ``open_tiles`` would take it as fact.
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
        page = _new_page(chromium_browser)
        posts = _track_panel_posts(page)
        panels_response: list[float] = []

        def _record_response(resp):
            if resp.request.method == "GET" and resp.url.rsplit("?", 1)[0].endswith("/api/panels"):
                panels_response.append(time.monotonic())

        page.on("response", _record_response)

        page.goto(base_url, wait_until="domcontentloaded")
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=15_000)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=20_000)
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
        assert {"tiles": [], "dock": True} not in reports, reports
        assert reports[-1] == {"tiles": ["artifacts"], "dock": True}, reports
        assert _panels_state(base_url)["open_tiles"] == ["artifacts"]

        page.close()


# ===========================================================================
# (h) Three-state occupancy: never reported / unknown / known
# ===========================================================================


def test_occupancy_read_back_distinguishes_unknown_from_empty(tmp_path, chromium_browser):
    """``open_tiles`` tells "nothing is open" apart from "nobody can say".

    Three states share one field, and collapsing any two of them would let a
    hook assert something nobody observed:

    * **never reported** — no client has checked in; all three fields null.
    * **known empty** — a dock client reports that the operator closed every
      service tile: ``[]`` with ``dock: true``.
    * **unknown** — a client without a dock shell is watching but cannot see
      tile order. It sends a presence report whose ``tiles`` the server must
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
        assert fresh["open_tiles"] is None, fresh
        assert fresh["open_tiles_age_s"] is None, fresh
        assert fresh["open_tiles_dock"] is None, fresh

        # --- known empty: a dock client closes the last service tile.
        page = _open_page(chromium_browser, base_url)
        expect(_service_tab(page, "WORKSPACE")).to_have_count(1, timeout=15_000)
        _wait_for_open_tiles(base_url, ["artifacts"])
        _close_tile(page, "artifacts")
        known_empty = _wait_for_open_tiles(base_url, [])
        assert known_empty["open_tiles_dock"] is True, known_empty
        assert isinstance(known_empty["open_tiles_age_s"], float), known_empty
        page.close()

        # --- unknown: a client with no dock shell at all.
        dockless = _new_page(chromium_browser)
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
            if state["open_tiles_dock"] is False:
                break
            dockless.wait_for_timeout(300)
        assert state.get("open_tiles_dock") is False, f"no dock-less report arrived: {state}"
        assert state["open_tiles"] is None, (
            f"a dock-less client's presence report was recorded as an empty workspace: {state}"
        )
        assert isinstance(state["open_tiles_age_s"], float), state

        dockless.close()


# ===========================================================================
# (e) Human gestures stay local: no focus command leaks, no cross-client echo
# ===========================================================================


def test_tile_close_with_two_tiles_commands_nothing(tmp_path, chromium_browser):
    """A tile "×" with a SECOND service tile open still commands nothing.

    The single-tile variant is pinned in the sibling suite; with two tiles the
    close makes dockview auto-activate the surviving service tile, and that
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
        assert _client_open_tiles(page_a) == ["artifacts"], (
            "client B's human focus gesture changed client A's tiles"
        )

        page_a.close()
        page_b.close()
