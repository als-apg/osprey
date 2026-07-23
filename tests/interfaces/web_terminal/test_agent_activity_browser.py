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
  (4) the plan panel's guarded auto-switch, against a real in-process
      bluesky bridge + panels sidecar: a PATCH /draft with client_id
      ``mcp-agent`` while the panel is unbound on another plan switches the
      panel to the drafted plan and flashes the applied arg fields
      (``agent-flash``); an operator-origin PATCH does neither.

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
_PROBES_INIT_SCRIPT = """
(function () {
  window.__sseOpenUrls = [];
  window.__sseFrames = [];
  window.__flashLog = [];

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


_ATTENTION_RE = re.compile(r"\bagent-attention\b")


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
# (4) plan panel guarded auto-switch — real bridge + sidecar, live SSE relay
# ---------------------------------------------------------------------------
#
# No hub involved: the guard lives entirely inside the plan panel bundle
# (draft-client.js), which the bluesky panels sidecar serves at /plan/ with
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
    from osprey.services.bluesky_bridge import draft as bridge_draft
    from osprey.services.bluesky_bridge import plan_loader
    from osprey.services.bluesky_bridge.app import app as bridge_app
    from osprey.services.bluesky_panels.app import app as sidecar_app

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


def _open_plan_panel(browser: Browser, sidecar_url: str) -> tuple[Page, str, str]:
    """Open /plan/ directly, wait for boot + SSE, return (page, selected, other).

    Boot auto-selects the first catalog plan; the draft targets the OTHER one
    so the agent PATCH is a genuine cross-plan switch, not a same-plan bind.
    """
    page = browser.new_page()
    page.add_init_script(_PROBES_INIT_SCRIPT)
    page.goto(f"{sidecar_url}/plan/", wait_until="domcontentloaded")

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
        page, _selected, other = _open_plan_panel(chromium_browser, sidecar_url)

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
        page, selected, other = _open_plan_panel(chromium_browser, sidecar_url)

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
