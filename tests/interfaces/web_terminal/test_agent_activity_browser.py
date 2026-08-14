"""Browser tests: agent-activity highlighting, end to end.

``POST /api/agent-activity`` broadcasts an ``agent_activity`` frame over the
``/api/files/events`` SSE stream; the hub front-end turns it into either a
persistent rail badge (kind ``panel`` with a rail entry) or a transient
activity-strip entry (every other kind). None of that is reachable from the
FastAPI TestClient — it only exists once a real browser runs panel-manager.js
and activity-strip.js against a live SSE stream.

Coverage:

  (1) kind ``panel`` targeting a real rail panel → its rail button gains the
      persistent ``agent-attention`` badge; activating the panel (a rail
      click) clears it.
  (2) kind ``channel`` → an ``.activity-strip-entry`` naming the channel
      appears in ``#activity-strip`` and auto-clears after ACTIVITY_CLEAR_MS.
  (3) malformed bodies → 422 AND no DOM change (no strip entry, no badge),
      proven by ordering: a valid frame POSTed *after* the malformed ones
      lands (its badge appears) while the strip stays empty and no other
      badge exists.
  (4) the Plans view's guarded auto-switch, against a real in-process
      bluesky bridge + panels sidecar: a PATCH /draft with client_id
      ``mcp-agent`` while the panel is unbound on another plan switches the
      panel to the drafted plan and flashes the applied arg fields
      (``agent-flash``); an operator-origin PATCH does neither.
  (5) an agent switch glows the panel's TILE BODY, not only its rail tab —
      with a human rail click as the control that must glow nothing.
  (6) an agent hide reports itself on the activity strip, worded and
      labelled, and glows nothing (the rail entry is already gone); an agent
      show right after is the positive control proving the glow probe armed.
  (7) the history popover: five rapid frames render as five rows, newest
      first, and the same rows — including a hide mirrored into the server
      ring by the panel routes and worded "agent closed …" — come back after
      a reload, served from ``GET /api/agent-activity/recent``.
  (8) badge acknowledgment across a reload: a badge the operator cleared by
      surfacing the panel stays cleared, while a newer unseen event restores
      one. Each half is the other's discriminator.
  (9) the logbook's sender-local ``osprey:navigate`` postMessage: the sending
      client navigates and activates with NO agent attribution, a second
      browser context is untouched, and a ``javascript:`` or protocol-relative
      url is ignored by the host.
  (10) reduced motion: the same flash resolves to the held-ring keyframes
      (constant box-shadow, no background fill) instead of the decay, and
      still self-cleans.

Transient ``agent-flash`` classes self-clean on animationend (~900ms), so
they are observed through a document-start MutationObserver log rather than
racing Playwright's polling against the animation. SSE readiness is likewise
observed directly (an EventSource wrapper records opened stream URLs) so no
test POSTs before its page is actually subscribed.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_agent_activity_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import requests

from tests.interfaces.conftest import _apply_all, _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Browser, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Document-start probes
# ---------------------------------------------------------------------------
#
# Installed via add_init_script so they run before any page script:
#
#   window.__sseOpenUrls — URL of every EventSource stream that reached 'open'
#     (the readiness barrier: no POST until the page is really subscribed).
#   window.__sseFrames   — every parsed SSE 'message' frame, in order (lets a
#     test wait for a specific frame to have ARRIVED before asserting that
#     nothing changed — the only non-racy way to prove a negative).
#   window.__flashLog    — className of every element the moment it gains the
#     transient `agent-flash` class (which self-cleans on animationend, too
#     fast to assert via polling).
#   window.__navMsgLog   — every `osprey:navigate` postMessage the page
#     received. The barrier for the logbook NEGATIVE cases: a rejected url
#     leaves no DOM trace at all, so "the host ignored it" can only be
#     asserted once the message is known to have been DELIVERED. Registered
#     at document start, hence ahead of app.js's own listener — but a
#     wait_for_function poll runs in a later task, by which time every
#     listener in that dispatch (app.js's included) has finished.
#   window.__fetchDone   — URL of every fetch that settled. The barrier for
#     restoreAgentBadges: "no badge came back" is only meaningful after the
#     history read that would have brought one back has actually completed.
_PROBES_INIT_SCRIPT = """
(function () {
  window.__sseOpenUrls = [];
  window.__sseFrames = [];
  window.__flashLog = [];
  window.__navMsgLog = [];
  window.__fetchDone = [];

  window.addEventListener('message', function (ev) {
    if (ev.data && ev.data.type === 'osprey:navigate') { window.__navMsgLog.push(ev.data); }
  });

  const origFetch = window.fetch;
  if (origFetch) {
    window.fetch = function (input, init) {
      const url = String(typeof input === 'string' ? input : (input && input.url) || '');
      const done = function () { window.__fetchDone.push(url); };
      return origFetch.call(this, input, init).then(
        function (resp) { done(); return resp; },
        function (err) { done(); throw err; },
      );
    };
  }

  const OrigES = window.EventSource;
  if (OrigES) {
    window.EventSource = new Proxy(OrigES, {
      construct(target, args) {
        const es = new target(...args);
        es.addEventListener('open', function () { window.__sseOpenUrls.push(es.url); });
        es.addEventListener('message', function (ev) {
          try { window.__sseFrames.push(JSON.parse(ev.data)); } catch (e) {}
        });
        return es;
      },
    });
  }

  function attach(el) {
    new MutationObserver(function (muts) {
      for (const m of muts) {
        const t = m.target;
        if (t && t.classList && t.classList.contains('agent-flash')) {
          window.__flashLog.push(String(t.className));
        }
      }
    }).observe(el, { attributes: true, attributeFilter: ['class'], subtree: true });
  }
  if (document.documentElement) {
    attach(document.documentElement);
  } else {
    new MutationObserver(function (_m, obs) {
      if (document.documentElement) { obs.disconnect(); attach(document.documentElement); }
    }).observe(document, { childList: true, subtree: true });
  }
})();
"""


def _wait_for_sse_open(page: Page, url_fragment: str) -> None:
    """Block until an EventSource whose URL contains ``url_fragment`` is open."""
    page.wait_for_function(
        "(frag) => (window.__sseOpenUrls || []).some((u) => u.includes(frag))",
        arg=url_fragment,
        timeout=10_000,
    )


# ---------------------------------------------------------------------------
# Hub launcher (scenarios 1-3) — mirrors test_panels_browser._live_server
# ---------------------------------------------------------------------------

# Custom panel with healthEndpoint=None → healthy immediately, so its rail
# button exists and can carry the attention badge; artifacts (the default
# panel) auto-activates, leaving data-viz inactive — the badge target.
_CUSTOM_DATA_VIZ: dict = {
    "id": "data-viz",
    "label": "DATA VIZ",
    "url": "http://data-viz.internal:8080",
    "healthEndpoint": None,
    "path": "/",
}


@contextmanager
def _hub_server(workspace_dir: Path) -> Iterator[str]:
    """Launch a real web-terminal hub with artifacts + the data-viz panel.

    The companion backends are bypassed via the same three patches every hub
    browser suite uses; the artifacts panel reports the standard fallback URL
    so it loads-and-activates without a real backend process.

    Yields:
        The hub's base URL.
    """
    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [_CUSTOM_DATA_VIZ], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_artifact_server",
            side_effect=lambda a: setattr(a.state, "artifact_server_url", "http://127.0.0.1:8086"),
        ),
    ]
    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        with _run_app_server(app) as base_url:
            yield base_url


def _open_hub_page(browser: Browser, base_url: str) -> Page:
    """Open a hub page with the probes armed, ready to receive SSE broadcasts.

    Waits for both rail entries (async panel init done), the dock grid, and —
    critically — the /api/files/events EventSource to be OPEN, so a POST made
    right after this helper is guaranteed to reach the page.
    """
    page = browser.new_page()
    page.add_init_script(_PROBES_INIT_SCRIPT)
    page.goto(base_url, wait_until="domcontentloaded")
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=10_000
    )
    expect(page.locator('button.panel-rail-button[data-panel-id="data-viz"]')).to_be_attached(
        timeout=10_000
    )
    expect(page.locator(".dv-groupview").first).to_be_visible(timeout=10_000)
    page.evaluate("document.getElementById('welcome-overlay')?.remove()")
    _wait_for_sse_open(page, "/api/files/events")
    return page


def _post_activity(base_url: str, body: dict) -> requests.Response:
    return requests.post(f"{base_url}/api/agent-activity", json=body, timeout=10)


def _post_panel(base_url: str, path: str, body: dict) -> requests.Response:
    """POST a panel-route command (focus / visibility) and require a 200."""
    resp = requests.post(f"{base_url}{path}", json=body, timeout=10)
    assert resp.status_code == 200, resp.text
    return resp


def _reload_hub_page(page: Page) -> None:
    """Reload an open hub page and re-establish its readiness barriers.

    The barriers that matter after a reload are the rail (panel-manager has
    initialised and rendered membership) and an OPEN SSE stream (the restore
    pass hangs off its onOpen hook, and any POST made after this returns is
    guaranteed to reach the page).

    Deliberately waits for neither the data-viz rail entry — the reload tests
    reload across an agent hide, after which that entry is legitimately gone —
    nor a dock group: the reload tests read the rail and the strip, and the
    restored dock layout varies with what the page did before reloading.
    """
    page.reload(wait_until="domcontentloaded")
    expect(page.locator(_ARTIFACTS_RAIL)).to_be_attached(timeout=10_000)
    page.evaluate("document.getElementById('welcome-overlay')?.remove()")
    _wait_for_sse_open(page, "/api/files/events")


def _wait_for_history_read(page: Page) -> None:
    """Block until the page has finished reading the server's history ring.

    ``restoreAgentBadges`` runs off the SSE open hook, so both the "badge came
    back" and the "badge stayed cleared" assertions need the read itself to
    have settled first — otherwise the negative one merely observes that the
    fetch had not landed yet, which is true whether or not restore works.

    Lower bound only: ``__fetchDone`` is pushed at response settle, before the
    body is parsed and before the badge loop runs. A caller asserting that
    something did NOT happen must add its own bounded settle on top of this.
    """
    page.wait_for_function(
        "() => (window.__fetchDone || []).some((u) => u.includes('/api/agent-activity/recent'))",
        timeout=10_000,
    )


def _open_history_popover(page: Page):
    """Open the strip's history popover the way a keyboard operator does.

    Focus + Enter rather than a click: the strip is a one-row flex region in
    the footer whose live slot is empty between frames, and driving it through
    focus keeps the test off Playwright's hit-testing entirely while staying a
    real user gesture (the mount takes tabindex=0 and Enter/Space).

    Returns:
        The popover locator (a child of ``<body>``, absent while closed).
    """
    page.locator("#activity-strip").focus()
    page.keyboard.press("Enter")
    popover = page.locator("body > .activity-history-popover")
    expect(popover).to_be_visible(timeout=5_000)
    return popover


_ATTENTION_RE = re.compile(r"\bagent-attention\b")
_ACTIVE_RE = re.compile(r"\bactive\b")

_DATA_VIZ_RAIL = 'button.panel-rail-button[data-panel-id="data-viz"]'
_ARTIFACTS_RAIL = 'button.panel-rail-button[data-panel-id="artifacts"]'


# ---------------------------------------------------------------------------
# (1) kind 'panel' → rail badge; activation clears it
# ---------------------------------------------------------------------------


def test_panel_activity_badges_rail_entry_and_activation_clears_it(tmp_path, chromium_browser):
    """A panel-kind frame badges the target's rail button; activating it clears.

    data-viz is healthy but inactive (artifacts holds the slot), so the badge
    persists until the operator actually surfaces the panel — the transient
    agent-flash may have self-cleaned already and is deliberately not raced.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        rail_btn = page.locator('button.panel-rail-button[data-panel-id="data-viz"]')
        expect(rail_btn).not_to_have_class(_ATTENTION_RE)

        r = _post_activity(
            base_url, {"tool": "switch_panel", "target": {"kind": "panel", "panel": "data-viz"}}
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True}

        # The persistent badge lands on exactly the targeted rail entry.
        expect(rail_btn).to_have_class(_ATTENTION_RE, timeout=5_000)
        expect(
            page.locator('button.panel-rail-button[data-panel-id="artifacts"]')
        ).not_to_have_class(_ATTENTION_RE)

        # Activating the panel (a rail click) serves — and clears — the badge.
        rail_btn.click()
        expect(rail_btn).not_to_have_class(_ATTENTION_RE, timeout=5_000)
        expect(rail_btn).to_have_class(re.compile(r"\bactive\b"), timeout=5_000)

        page.close()


# ---------------------------------------------------------------------------
# (2) kind 'channel' → activity-strip entry, then auto-clear
# ---------------------------------------------------------------------------


def test_channel_activity_shows_strip_entry_then_auto_clears(tmp_path, chromium_browser):
    """A channel-kind frame lands in #activity-strip and auto-clears (~6s).

    Channel frames are never suppressed (no panel self-signals a channel
    write), so the entry must show regardless of which panel is active. The
    auto-clear is asserted as an eventual condition with a generous timeout —
    ACTIVITY_CLEAR_MS is 6000ms — rather than a fixed sleep.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        entry = page.locator("#activity-strip .activity-strip-entry")
        expect(entry).to_have_count(0)

        r = _post_activity(
            base_url,
            {"tool": "write_channel", "target": {"kind": "channel", "detail": "SR01:HCM1:SP"}},
        )
        assert r.status_code == 200, r.text

        expect(entry).to_be_visible(timeout=5_000)
        expect(entry).to_contain_text("SR01:HCM1:SP")

        # Auto-clear: the entry retires on its own after ACTIVITY_CLEAR_MS.
        expect(entry).to_have_count(0, timeout=12_000)

        page.close()


# ---------------------------------------------------------------------------
# (3) malformed POST → 422 and no DOM change
# ---------------------------------------------------------------------------


def test_malformed_post_422_and_no_dom_change(tmp_path, chromium_browser):
    """Unknown kinds / missing fields 422 and leave the DOM untouched.

    Proving "nothing happened" without an arbitrary sleep uses SSE ordering:
    a VALID panel-kind frame POSTed after the malformed ones must land (its
    badge appears), and since broadcasts are delivered in order on the one
    stream, anything the malformed POSTs had (wrongly) broadcast would have
    arrived first. The strip staying empty and the badge count staying at
    exactly one proves the 422s broadcast nothing.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)

        # Unknown target kind.
        r = _post_activity(base_url, {"tool": "switch_panel", "target": {"kind": "widget"}})
        assert r.status_code == 422, r.text
        # Missing tool.
        r = _post_activity(base_url, {"target": {"kind": "panel", "panel": "data-viz"}})
        assert r.status_code == 422, r.text

        # Ordering sentinel: a valid frame POSTed after the malformed ones.
        r = _post_activity(
            base_url, {"tool": "switch_panel", "target": {"kind": "panel", "panel": "data-viz"}}
        )
        assert r.status_code == 200, r.text
        expect(page.locator('button.panel-rail-button[data-panel-id="data-viz"]')).to_have_class(
            _ATTENTION_RE, timeout=5_000
        )

        # The sentinel arrived, so the malformed frames (had they broadcast)
        # would already be visible — and they are not: no strip entry, and the
        # sentinel's badge is the only attention badge in the document.
        expect(page.locator("#activity-strip .activity-strip-entry")).to_have_count(0)
        expect(page.locator(".agent-attention")).to_have_count(1)

        page.close()


# ---------------------------------------------------------------------------
# (4) Plans view guarded auto-switch — real bridge + sidecar, live SSE relay
# ---------------------------------------------------------------------------
#
# No hub involved: the guard lives entirely inside the bluesky panel bundle
# (draft-client.js), which the bluesky panels sidecar serves at /bluesky/ with
# the shared design-system assets mounted — so the page is loaded directly
# from the sidecar, exactly as the visual-regression suite loads panels. The
# sidecar's /draft relay and /draft/events SSE hop run against the REAL
# bridge app (uvicorn on a real port), not a mock.

# Valid draft args per shipped plan (the catalog ships exactly orm +
# grid_scan). The test drafts whichever plan is NOT auto-selected, so both
# must be expressible.
_PLAN_ARGS: dict[str, dict] = {
    "grid_scan": {
        "detectors": ["BPM1"],
        "axes": [{"setpoint": "COR1", "start": 0.0, "stop": 1.0, "num_points": 3}],
    },
    "orm": {"correctors": ["COR1"], "detectors": ["BPM1"], "span_a": 1.0, "num": 3},
}


@contextmanager
def _plan_panel_stack(tmp_path: Path) -> Iterator[str]:
    """Launch the real bridge + panels sidecar on live ports; yield sidecar URL.

    Env isolation mirrors tests/services/test_draft_roundtrip.py: shipped-tier
    plans only (no facility dirs/module, no config file), plus the bridge
    draft singleton cleared on entry AND exit so state never leaks between
    tests or into other suites in the same process. The sidecar resolves its
    bridge via BLUESKY_BRIDGE_URL, set here to the live bridge's port before
    the sidecar's lifespan runs.
    """
    from osprey.interfaces.bluesky_panels.app import app as sidecar_app
    from osprey.services.bluesky_bridge import draft as bridge_draft
    from osprey.services.bluesky_bridge import plan_loader
    from osprey.services.bluesky_bridge.app import app as bridge_app

    managed = [
        "BLUESKY_SESSION_PLAN_DIR",
        "BLUESKY_PLAN_DIRS",
        "BLUESKY_PLAN_MODULE",
        "BLUESKY_BRIDGE_URL",
        "OSPREY_CONFIG",
    ]
    saved = {k: os.environ.get(k) for k in managed}
    try:
        os.environ["BLUESKY_SESSION_PLAN_DIR"] = str(tmp_path / "plans_session")
        os.environ["OSPREY_CONFIG"] = str(tmp_path / "does-not-exist.yml")
        os.environ.pop("BLUESKY_PLAN_DIRS", None)
        os.environ.pop("BLUESKY_PLAN_MODULE", None)
        plan_loader.reset_facility_plans()
        bridge_draft._clear()

        with _run_app_server(bridge_app) as bridge_url:
            os.environ["BLUESKY_BRIDGE_URL"] = bridge_url
            with _run_app_server(sidecar_app) as sidecar_url:
                yield sidecar_url
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        plan_loader.reset_facility_plans()
        bridge_draft._clear()


def _open_plans_view(browser: Browser, sidecar_url: str) -> tuple[Page, str, str]:
    """Open /bluesky/ directly, wait for boot + SSE, return (page, selected, other).

    Boot auto-selects the first catalog plan; the draft targets the OTHER one
    so the agent PATCH is a genuine cross-plan switch, not a same-plan bind.
    """
    page = browser.new_page()
    page.add_init_script(_PROBES_INIT_SCRIPT)
    page.goto(f"{sidecar_url}/bluesky/", wait_until="domcontentloaded")

    selected_row = page.locator(".plan-row.selected")
    expect(selected_row).to_have_count(1, timeout=10_000)
    _wait_for_sse_open(page, "/draft/events")

    selected = selected_row.get_attribute("data-plan-name")
    assert selected in _PLAN_ARGS, f"auto-selected an unexpected plan: {selected!r}"
    other = next(name for name in _PLAN_ARGS if name != selected)
    # Both shipped plans must be on offer, or "switch to the other" is vacuous.
    expect(page.locator(f'.plan-row[data-plan-name="{other}"]')).to_have_count(1, timeout=5_000)
    return page, selected, other


def test_agent_draft_patch_auto_switches_plan_and_flashes_fields(tmp_path, chromium_browser):
    """A PATCH /draft as client_id 'mcp-agent' auto-switches the unbound panel.

    The panel sits unbound on the auto-selected plan with no local edits —
    exactly the guarded state in which an agent draft for ANOTHER plan may
    take the screen. The panel must re-select the drafted plan and flash the
    applied arg fields (the transient agent-flash is caught by the
    document-start MutationObserver, not by racing the 900ms animation).
    """
    with _plan_panel_stack(tmp_path) as sidecar_url:
        page, _selected, other = _open_plans_view(chromium_browser, sidecar_url)

        r = requests.patch(
            f"{sidecar_url}/draft",
            json={
                "plan_name": other,
                "plan_args_patch": _PLAN_ARGS[other],
                "client_id": "mcp-agent",
            },
            timeout=10,
        )
        assert r.status_code == 200, r.text

        # The panel switched itself onto the agent's drafted plan.
        expect(page.locator(".plan-row.selected")).to_have_attribute(
            "data-plan-name", other, timeout=10_000
        )

        # The applied arg fields carried the agent-flash glow.
        page.wait_for_function("() => (window.__flashLog || []).length > 0", timeout=5_000)
        flash_log = page.evaluate("window.__flashLog")
        assert any("param-row" in cls for cls in flash_log), (
            f"agent-flash never landed on a form field row: {flash_log!r}"
        )

        page.close()


def test_operator_draft_patch_does_not_switch_or_flash(tmp_path, chromium_browser):
    """An operator-origin PATCH never auto-switches and applies no agent styling.

    Same unbound starting state, but the PATCH carries a non-agent client_id.
    The probe log proves the frame for this PATCH ARRIVED at the page (matched
    by origin); arrival alone does not prove the async frame handler finished
    (the unbound apply path awaits ensureDraftPlanKnown() — a fetch — before
    any would-be switch), so a bounded settle after the arrival barrier covers
    that async handling before the negative assertions run.
    """
    with _plan_panel_stack(tmp_path) as sidecar_url:
        page, selected, other = _open_plans_view(chromium_browser, sidecar_url)

        r = requests.patch(
            f"{sidecar_url}/draft",
            json={
                "plan_name": other,
                "plan_args_patch": _PLAN_ARGS[other],
                "client_id": "operator-console",
            },
            timeout=10,
        )
        assert r.status_code == 200, r.text

        # Wait until the frame for this PATCH has actually reached the page.
        page.wait_for_function(
            "() => (window.__sseFrames || []).some((f) => f.origin === 'operator-console')",
            timeout=10_000,
        )
        # Bounded settle: the frame handler is async (it awaits a fetch before
        # any would-be switch), so give a wrongly-switching regression time to
        # manifest before asserting it didn't.
        page.wait_for_timeout(750)

        # No switch: the panel still shows the originally-selected plan.
        expect(page.locator(".plan-row.selected")).to_have_attribute("data-plan-name", selected)
        # No agent styling: nothing ever gained agent-flash.
        assert page.evaluate("(window.__flashLog || []).length") == 0
        expect(page.locator(".agent-flash")).to_have_count(0)

        page.close()


# ---------------------------------------------------------------------------
# (5) agent switch → tile-body glow (and a human click that glows nothing)
# ---------------------------------------------------------------------------


def test_agent_switch_glows_the_tile_body_not_only_the_rail(tmp_path, chromium_browser):
    """An agent switch glows the panel's TILE, on top of the rail-tab flash.

    A rail tab is ~20px of chrome; the claim under test is that the operator
    also sees the panel body the agent touched. So the assertion is not "some
    element flashed" — every attribution path flashes the rail — but that a
    `.tile-glow` overlay element flashed, which only glowPanel() produces.

    The human rail click that precedes it is the discriminator: it surfaces
    exactly the same panel through the same activation path, so a glow that
    fired for it would mean the glow tracks activation rather than agent
    attribution. Its `.active` class is the barrier proving it completed.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        rail_btn = page.locator(_DATA_VIZ_RAIL)

        # Control: a human surfaces the panel himself. Same activation, no
        # attribution — nothing may flash, on the rail or over the tile.
        rail_btn.click()
        expect(rail_btn).to_have_class(_ACTIVE_RE, timeout=5_000)
        # Bounded settle: glowPanel defers its flash into a requestAnimationFrame,
        # so a wrongly-firing glow lands a frame after `.active`. Without this the
        # control passes on the very regression it exists to catch.
        page.wait_for_timeout(250)
        assert page.evaluate("(window.__flashLog || []).length") == 0
        expect(page.locator(".dock-iframe-overlay .tile-glow")).to_have_count(0)

        # The agent switches to the very same panel.
        _post_panel(base_url, "/api/panel-focus", {"panel": "data-viz", "source": "agent"})

        # The tile body glowed — the class is caught by the document-start
        # MutationObserver, since it self-cleans well before a poll could see it.
        page.wait_for_function(
            "() => (window.__flashLog || []).some((c) => c.includes('tile-glow'))",
            timeout=10_000,
        )
        flash_log = page.evaluate("window.__flashLog")
        # ...and the rail tab flashed too: the two halves of the same signal.
        assert any("panel-rail-button" in cls for cls in flash_log), (
            f"the rail entry never flashed alongside the tile glow: {flash_log!r}"
        )
        # The glow overlay is a sibling of the dock's iframes, never a class on
        # the iframe itself (which would clip the embedded app to the ring).
        expect(page.locator(".dock-iframe-overlay .tile-glow")).to_have_count(1)
        # Read it off the log, not the live DOM: `agent-flash` self-cleans on
        # animationend, so a DOM query would pass on a regression it merely missed.
        assert not any("panel-iframe" in cls for cls in flash_log), (
            f"the iframe itself was flashed instead of the overlay: {flash_log!r}"
        )

        page.close()


# ---------------------------------------------------------------------------
# (6) agent hide → labelled strip entry, no glow
# ---------------------------------------------------------------------------


def test_agent_hide_reports_on_the_strip_and_glows_nothing(tmp_path, chromium_browser):
    """An agent hide is reported in words; there is nothing left to glow.

    The rail entry is removed by the same operation, so a glow would have
    nowhere to land and the strip line is the whole feedback. The line is
    asserted verbatim — verb AND catalog label, not the raw panel id — because
    "some entry appeared" is equally consistent with the unlabelled fallback.

    The agent SHOW at the end is the positive control: it proves the flash
    probe was armed and capable of recording a glow throughout, so the
    preceding empty `__flashLog` is evidence rather than an artefact.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        rail_btn = page.locator(_DATA_VIZ_RAIL)
        entry = page.locator("#activity-strip .activity-strip-entry")

        _post_panel(
            base_url,
            "/api/panel-visibility",
            {"panel": "data-viz", "visible": False, "source": "agent"},
        )

        expect(entry).to_have_text("agent closedDATA VIZ", timeout=5_000)
        # The hide really happened: membership is gone, not just narrated.
        expect(rail_btn).to_have_count(0, timeout=5_000)
        # Nothing flashed anywhere — no rail glow (the entry is gone) and no
        # tile glow (the visibility path never calls glowPanel).
        assert page.evaluate("(window.__flashLog || []).length") == 0
        expect(page.locator(".dock-iframe-overlay .tile-glow")).to_have_count(0)

        # Positive control: the same route, the other direction, does glow.
        _post_panel(
            base_url,
            "/api/panel-visibility",
            {"panel": "data-viz", "visible": True, "source": "agent"},
        )
        expect(rail_btn).to_have_count(1, timeout=5_000)
        expect(entry).to_have_text("agent openedDATA VIZ", timeout=5_000)
        page.wait_for_function(
            "() => (window.__flashLog || []).some((c) => c.includes('panel-rail-button'))",
            timeout=10_000,
        )

        page.close()


# ---------------------------------------------------------------------------
# (7) history popover — five rapid frames, and the same rows after a reload
# ---------------------------------------------------------------------------

_BURST_CHANNELS = [
    "SR01:HCM1:SP",
    "SR02:HCM1:SP",
    "SR03:HCM1:SP",
    "SR04:HCM1:SP",
    "SR05:HCM1:SP",
]


def test_history_popover_lists_every_frame_of_a_rapid_burst(tmp_path, chromium_browser):
    """Five frames in quick succession leave one live line but five rows.

    The live strip is single-slot latest-wins, so a burst is exactly the case
    where the popover has to earn its keep. Asserting a count of five alone
    would pass on five copies of the newest frame, so the subjects are matched
    against the posted channels in reverse: that pins the count, the
    newest-first ordering the endpoint promises, and the absence of both
    duplication and truncation in one comparison.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)

        for channel in _BURST_CHANNELS:
            r = _post_activity(
                base_url,
                {"tool": "write_channel", "target": {"kind": "channel", "detail": channel}},
            )
            assert r.status_code == 200, r.text

        # The live line coalesced the burst down to the newest frame.
        entry = page.locator("#activity-strip .activity-strip-entry")
        expect(entry).to_have_text(f"agent wrote{_BURST_CHANNELS[-1]}", timeout=5_000)

        popover = _open_history_popover(page)
        expect(popover.locator(".activity-history-row")).to_have_count(5, timeout=5_000)
        assert popover.locator(".activity-history-subject").all_text_contents() == list(
            reversed(_BURST_CHANNELS)
        )
        assert popover.locator(".activity-history-verb").all_text_contents() == ["agent wrote"] * 5

        page.close()


def test_history_popover_rows_survive_a_reload_including_a_hide(tmp_path, chromium_browser):
    """After a reload the popover shows the same history, read from the server.

    The history is the server's ring, not page state, and this is what makes
    that observable: nothing in the reloaded page ever saw these frames live.
    The hide is the discriminating row — it reaches the ring only because the
    panel ROUTE mirrors an agent-origin visibility change as ``hide_panel``
    (an SSE frame alone would leave no trace to re-read), and the popover must
    word it exactly as the live strip did before the reload.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)

        for channel in _BURST_CHANNELS[:2]:
            r = _post_activity(
                base_url,
                {"tool": "write_channel", "target": {"kind": "channel", "detail": channel}},
            )
            assert r.status_code == 200, r.text
        _post_panel(
            base_url,
            "/api/panel-visibility",
            {"panel": "data-viz", "visible": False, "source": "agent"},
        )

        # Pre-reload wording, from the live client-synthesised frame.
        expect(page.locator("#activity-strip .activity-strip-entry")).to_have_text(
            "agent closedDATA VIZ", timeout=5_000
        )

        _reload_hub_page(page)

        # The reloaded page has seen nothing; every row below came off the ring.
        expect(page.locator("#activity-strip .activity-strip-entry")).to_have_count(0)
        popover = _open_history_popover(page)
        expect(popover.locator(".activity-history-row")).to_have_count(3, timeout=5_000)
        assert popover.locator(".activity-history-verb").all_text_contents() == [
            "agent closed",
            "agent wrote",
            "agent wrote",
        ]
        assert popover.locator(".activity-history-subject").all_text_contents() == [
            "DATA VIZ",
            _BURST_CHANNELS[1],
            _BURST_CHANNELS[0],
        ]

        page.close()


# ---------------------------------------------------------------------------
# (8) badge acknowledgment across a reload
# ---------------------------------------------------------------------------


def test_acknowledged_badge_stays_cleared_across_reload_and_a_newer_one_returns(
    tmp_path, chromium_browser
):
    """A badge the operator served stays served; a later one he never saw returns.

    The two halves discriminate each other, which is the point of running them
    against one page: "stays cleared" on its own is equally consistent with
    restore being broken outright, and "comes back" on its own is equally
    consistent with the acknowledgment never being written. Only both, in this
    order, distinguish a working acknowledgment from a dead one.

    The acknowledgment is per-panel and keyed on the SERVER timestamp, so the
    reload compares ring rows against what the operator actually cleared —
    never against a browser clock.

    Focus is deliberately handed back to artifacts before each reload: a badge
    restored onto the panel that is already surfaced needs two rail clicks to
    clear (the first retires the tile), a known and accepted limit that this
    test stays clear of rather than fights.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        rail_btn = page.locator(_DATA_VIZ_RAIL)

        # --- The operator sees an agent action and serves it -----------------
        r = _post_activity(
            base_url, {"tool": "switch_panel", "target": {"kind": "panel", "panel": "data-viz"}}
        )
        assert r.status_code == 200, r.text
        expect(rail_btn).to_have_class(_ATTENTION_RE, timeout=5_000)
        rail_btn.click()
        expect(rail_btn).not_to_have_class(_ATTENTION_RE, timeout=5_000)

        # Surfacing the panel wrote the acknowledgment: the server ts of the
        # badge that was cleared, as a string — never a client clock.
        ack = page.evaluate("localStorage.getItem('agent-ack:data-viz')")
        assert ack is not None, "surfacing the panel did not record an acknowledgment"
        assert float(ack) > 0

        # Hand the workspace back to artifacts so the reload does not land with
        # data-viz surfaced (see the docstring's note on the two-click limit).
        page.locator(_ARTIFACTS_RAIL).click()
        expect(page.locator(_ARTIFACTS_RAIL)).to_have_class(_ACTIVE_RE, timeout=5_000)

        # --- Reload: the served badge must not come back ---------------------
        _reload_hub_page(page)
        _wait_for_history_read(page)
        # The restore pass is async past its fetch; give a regression that
        # re-badges everything time to manifest before asserting it did not.
        page.wait_for_timeout(500)
        assert page.evaluate("localStorage.getItem('agent-ack:data-viz')") == ack
        expect(page.locator(_DATA_VIZ_RAIL)).not_to_have_class(_ATTENTION_RE)
        expect(page.locator(".agent-attention")).to_have_count(0)

        # --- A newer action the operator never served ------------------------
        r = _post_activity(
            base_url, {"tool": "switch_panel", "target": {"kind": "panel", "panel": "data-viz"}}
        )
        assert r.status_code == 200, r.text
        expect(page.locator(_DATA_VIZ_RAIL)).to_have_class(_ATTENTION_RE, timeout=5_000)

        # --- Reload: THIS one must come back ---------------------------------
        _reload_hub_page(page)
        _wait_for_history_read(page)
        expect(page.locator(_DATA_VIZ_RAIL)).to_have_class(_ATTENTION_RE, timeout=5_000)

        page.close()


# ---------------------------------------------------------------------------
# (9) logbook send → sender-local navigation, unattributed, scheme-guarded
# ---------------------------------------------------------------------------
#
# The ARIEL logbook panel reports a successful submit to its host with
# `window.parent.postMessage({type:'osprey:navigate', panel, url}, origin)` —
# no `source` key, because the absence of agent attribution is the point: a
# human clicked "send to logbook", so the host must navigate that ONE client
# and tell nobody else. The tests post the same message from the page itself,
# which is the same same-origin delivery the iframe performs, and exercises
# the host listener in app.js exactly as the panel does.

_POST_NAVIGATE_JS = """
({ panel, url }) => window.postMessage(
  { type: 'osprey:navigate', panel: panel, url: url }, window.location.origin,
)
"""


def _data_viz_iframe(page: Page):
    return page.locator('iframe.panel-iframe[data-panel-id="data-viz"]')


def test_logbook_navigate_is_local_to_the_sending_client_and_unattributed(
    tmp_path, chromium_browser
):
    """A logbook send moves the sender's workspace only, with no agent styling.

    Two browser contexts against one hub. The sending client must navigate and
    surface the panel; the other must not move at all — a human's click in one
    control-room browser cannot be allowed to reach for someone else's screen.

    The second context's "nothing happened" is proven by ORDERING rather than
    by a sleep: a real agent frame POSTed afterwards lands in its strip, and
    since both travel the same one stream in order, anything the navigate had
    (wrongly) broadcast would already be visible there.

    And on the sender: navigating is not an agent action, so no flash, no
    badge and no strip line may appear even on the client that did move.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        sender = _open_hub_page(chromium_browser, base_url)
        bystander = _open_hub_page(chromium_browser, base_url)
        draft_url = "/ariel/logbook/draft/42"

        sender.evaluate(_POST_NAVIGATE_JS, {"panel": "data-viz", "url": draft_url})

        # The sender navigated the panel and surfaced it.
        expect(sender.locator(_DATA_VIZ_RAIL)).to_have_class(_ACTIVE_RE, timeout=5_000)
        expect(_data_viz_iframe(sender)).to_have_attribute(
            "src", re.compile(re.escape(draft_url)), timeout=5_000
        )
        # ...with no agent attribution of any kind.
        assert sender.evaluate("(window.__flashLog || []).length") == 0
        expect(sender.locator(".agent-attention")).to_have_count(0)
        expect(sender.locator("#activity-strip .activity-strip-entry")).to_have_count(0)

        # Ordering sentinel: a genuine broadcast reaches the second context.
        r = _post_activity(
            base_url,
            {"tool": "write_channel", "target": {"kind": "channel", "detail": "SR09:SENTINEL:SP"}},
        )
        assert r.status_code == 200, r.text
        expect(bystander.locator("#activity-strip .activity-strip-entry")).to_have_text(
            "agent wroteSR09:SENTINEL:SP", timeout=5_000
        )

        # The sentinel arrived, so a broadcast navigate would have too — and
        # the second context's workspace is exactly where it was.
        expect(bystander.locator(_DATA_VIZ_RAIL)).not_to_have_class(_ACTIVE_RE)
        expect(_data_viz_iframe(bystander)).to_have_count(0)
        assert bystander.evaluate("(window.__flashLog || []).length") == 0

        bystander.close()
        sender.close()


def test_logbook_navigate_ignores_javascript_and_protocol_relative_urls(tmp_path, chromium_browser):
    """The host takes only root-relative urls, whatever the origin check said.

    Same-origin is necessary but not sufficient: the sender may be an
    agent-authored artifact rendered in a sandboxed panel, and this url ends
    up as an iframe src. ``javascript:alert(1)`` survives the embed-src
    builder intact and would run in the HOST origin, and ``//evil.example/x``
    resolves to a cross-origin document — so a leading-slash test alone would
    be a hole, and both must be dropped.

    Rejection leaves no DOM trace, so the negative is anchored twice: the
    message log proves both messages were DELIVERED before anything is
    asserted about them, and the root-relative message sent afterwards proves
    the listener was alive and willing the whole time.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)

        for bad_url in ("javascript:alert(1)", "//evil.example/x"):
            page.evaluate(_POST_NAVIGATE_JS, {"panel": "data-viz", "url": bad_url})
        page.wait_for_function("() => (window.__navMsgLog || []).length === 2", timeout=5_000)
        # Delivery is not handling: activation runs past health guards, so
        # give a wrongly-accepting regression time to manifest.
        page.wait_for_timeout(500)

        expect(page.locator(_DATA_VIZ_RAIL)).not_to_have_class(_ACTIVE_RE)
        expect(_data_viz_iframe(page)).to_have_count(0)

        # The very next message, root-relative, DOES land — so the two above
        # were refused on their urls and not on a dead listener.
        good_url = "/ariel/logbook/draft/7"
        page.evaluate(_POST_NAVIGATE_JS, {"panel": "data-viz", "url": good_url})
        expect(page.locator(_DATA_VIZ_RAIL)).to_have_class(_ACTIVE_RE, timeout=5_000)
        iframe = _data_viz_iframe(page)
        expect(iframe).to_have_attribute("src", re.compile(re.escape(good_url)), timeout=5_000)
        src = iframe.get_attribute("src") or ""
        assert "evil.example" not in src and "javascript:" not in src, src

        page.close()


# ---------------------------------------------------------------------------
# (10) reduced motion → the same flash, held still
# ---------------------------------------------------------------------------
#
# The stylesheet's reduced-motion branch re-declares @keyframes under the SAME
# name rather than switching the animation off, because `animation: none`
# never fires animationend and would strand `.agent-flash` on the element
# forever. What a static assertion over the CSS text cannot show is that the
# override actually WINS at cascade time in a real engine — that is this
# test's whole job, so it drives the shipped flashElement against the shipped
# stylesheet and reads back what the engine resolved.

_FLASH_AND_PROBE_JS = """
async ({ selector, tokens }) => {
  const el = document.querySelector(selector);
  const mod = await import('/design-system/js/highlight.js');

  // Canonicalise the token values through the engine, so they are comparable
  // with computed colors (which re-serialise `0.20` as `0.2`).
  const probe = document.createElement('div');
  document.body.appendChild(probe);
  const resolved = {};
  for (const name of tokens) {
    probe.style.backgroundColor =
      getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    resolved[name] = getComputedStyle(probe).backgroundColor;
  }
  probe.remove();

  const resting = getComputedStyle(el).backgroundColor;
  mod.flashElement(el);
  const cs = getComputedStyle(el);
  return {
    tokens: resolved,
    resting: resting,
    animationName: cs.animationName,
    animationDuration: cs.animationDuration,
    background: cs.backgroundColor,
    shadow: cs.boxShadow,
    flashing: el.classList.contains('agent-flash'),
  };
}
"""

_READ_FLASH_JS = """
(selector) => {
  const el = document.querySelector(selector);
  const cs = getComputedStyle(el);
  return {
    background: cs.backgroundColor,
    shadow: cs.boxShadow,
    flashing: el.classList.contains('agent-flash'),
  };
}
"""

_TINT_20 = "--accent-tint-20"
_TINT_25 = "--accent-tint-25"


def _flash(page: Page, selector: str) -> dict:
    return page.evaluate(
        _FLASH_AND_PROBE_JS, {"selector": selector, "tokens": [_TINT_20, _TINT_25]}
    )


def _await_flash_cleanup(page: Page, selector: str) -> None:
    """Block until the flash class has self-cleaned (the animationend guard)."""
    page.wait_for_function(
        "(s) => !document.querySelector(s).classList.contains('agent-flash')",
        arg=selector,
        timeout=5_000,
    )


def test_reduced_motion_holds_the_glow_ring_still_instead_of_decaying(tmp_path, chromium_browser):
    """Under reduced motion the glow is a held ring, not a decay — and it ends.

    Both halves run against the same element in the same page, so the only
    variable is the media preference. Each assertion is chosen to come out
    DIFFERENTLY under the two branches:

      * duration — 0.9s decay vs the 1.2s hold (a longer dwell, because a
        still ring has no motion to catch the eye);
      * box-shadow sampled at two offsets — decaying vs identical;
      * background-color — the discriminating probe for the override winning.
        The base keyframes fill the element with --accent-tint-20 and fade it
        out; the reduced-motion keyframes deliberately omit background-color,
        so the element keeps its own. Same animation NAME either way, which is
        exactly why the name alone proves nothing here;
      * the class self-cleans — the property `animation: none` would have
        broken, since it never fires animationend.

    One theme is enough: the branch under test is the media query, and every
    theme reaches it through the same two accent tokens.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        # An inactive rail entry: a resting background of its own, so a base
        # fill painted over it is unambiguous.
        selector = _DATA_VIZ_RAIL

        # --- Normal motion: the decay -----------------------------------------
        page.emulate_media(reduced_motion="no-preference")
        normal = _flash(page, selector)
        assert normal["flashing"] is True
        assert normal["animationName"] == "agent-flash-glow"
        assert normal["animationDuration"] == "0.9s"
        # The base keyframes paint their own fill over the element.
        assert normal["background"] == normal["tokens"][_TINT_20], normal
        page.wait_for_timeout(300)
        normal_later = page.evaluate(_READ_FLASH_JS, selector)
        assert normal_later["shadow"] != normal["shadow"], "the ring did not decay"
        _await_flash_cleanup(page, selector)

        # --- Reduced motion: the hold ----------------------------------------
        page.emulate_media(reduced_motion="reduce")
        held = _flash(page, selector)
        assert held["flashing"] is True
        assert held["animationName"] == "agent-flash-glow"
        assert held["animationDuration"] == "1.2s"
        # The override omits the fill, so the element keeps its own background.
        assert held["background"] == held["resting"], held
        assert held["background"] != held["tokens"][_TINT_20], held
        # A real ring is painted, from the accent token, and held constant.
        assert "0px 0px 0px 3px" in held["shadow"], held["shadow"]
        assert held["tokens"][_TINT_25] in held["shadow"], held
        page.wait_for_timeout(300)
        held_later = page.evaluate(_READ_FLASH_JS, selector)
        assert held_later["shadow"] == held["shadow"], "the held ring moved"
        # It still ENDS: the stepped animation fires animationend, so the
        # class self-cleans exactly as the decay does.
        _await_flash_cleanup(page, selector)

        page.close()
