"""Browser suite: the dispatch dashboard's Simple mode subtracts what it claims to.

The EVENTS panel used to answer the ``data-ui-mode`` axis by carrying a second,
hand-written DOM: an operator-facing ``.simple-view`` rendered by its own
``renderSimple()``, shown while the expert layout was hidden wholesale. That is
*substitution*, and substitution is self-guarding — the simple page could only
ever show what it was written to show.

It now answers the axis by *subtraction*: one markup tree, with everything an
operator should not see carrying ``.expert-only``, and a single stylesheet rule
hiding that class under ``html[data-ui-mode="simple"]``. Subtraction is faster
to maintain and cannot drift out of sync with the expert view — but it fails
open. A new field added without the class, or a specificity mistake in the one
rule, silently leaks run ids, durations and firing controls onto an operator
screen with nothing failing.

This suite is that missing failure. It asserts against *computed visibility* in
a real browser rather than the presence of class names in markup, because the
whole risk here is that the class is present and the CSS nonetheless does not
hide it.

Placement rationale: matches ``test_visual.py``, which already stands the
dispatch dashboard up from this directory (it has no standalone FastAPI app of
its own in production), and the axis under test is the design system's
``mode-boot.js`` contract.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_dashboard_modes.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1100, "height": 900}

DESIGN_SYSTEM_STATIC_DIR = (
    __import__("pathlib").Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "design_system"
    / "static"
)

#: Surfaces Simple mode must remove that are *visible in Expert on load*, so a
#: passing assertion here can only mean the mode rule did the hiding. Keyed by
#: selector, valued by what leaks if the rule stops working.
EXPERT_ONLY_SURFACES = {
    "#tab-triggers": "the Triggers tab (firing and enable/disable)",
    "#timeline-window": "the timeline window control",
    ".filters": "the run status filters and search box",
}

#: The two routing screens. These are hidden on load in BOTH modes — the
#: router only lights the active view — so asserting them hidden as-is would
#: pass whether or not Simple mode does anything. They are checked separately,
#: after being forced active, so the assertion has to be earned.
EXPERT_ONLY_SCREENS = {
    "#view-triggers": "the trigger list",
    "#view-trigger": "the trigger detail, including the payload editor",
}


def _create_dispatch_dashboard_app() -> FastAPI:
    """Serve the real ``render_dashboard_html()`` output beside the real design system.

    Mirrors ``test_visual.py``'s helper of the same name: the dashboard is
    rendered HTML mounted into the event-dispatcher server in production, so a
    minimal real app is the honest way to drive it.
    """
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return str(render_dashboard_html(facility_name="Test Facility", pv_strip_prefix="SR:"))

    return app


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    with _run_app_server(_create_dispatch_dashboard_app()) as base_url:
        yield base_url


def _open(page: Page, base_url: str, mode: str) -> None:
    """Load the dashboard in one mode.

    ``?mode=`` is the top rung of mode-boot.js's resolution ladder, so it wins
    over localStorage and any server-rendered attribute — and it is applied
    pre-paint, which is what the mode-gated layout depends on.
    """
    page.goto(f"{base_url}/?mode={mode}", wait_until="domcontentloaded")
    page.wait_for_function(
        "mode => document.documentElement.getAttribute('data-ui-mode') === mode",
        arg=mode,
        timeout=10_000,
    )


def _visible(page: Page, selector: str) -> bool:
    """Computed visibility, not markup presence.

    ``offsetParent === null`` catches a display:none ancestor as well as the
    element's own rule, which is exactly the leak path a markup-only check
    would miss.
    """
    return page.evaluate(
        "sel => { const e = document.querySelector(sel);"
        "  return e !== null && e.offsetParent !== null; }",
        selector,
    )


@pytest.mark.parametrize("selector,leak", EXPERT_ONLY_SURFACES.items(), ids=lambda v: str(v)[:40])
def test_simple_mode_hides_every_expert_surface(
    chromium_browser: Browser, dashboard_url: str, selector: str, leak: str
) -> None:
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open(page, dashboard_url, "simple")
    assert not _visible(page, selector), (
        f"Simple mode still shows {leak} ({selector}) -- an operator screen must not carry it"
    )
    page.close()


@pytest.mark.parametrize("selector,leak", EXPERT_ONLY_SCREENS.items(), ids=lambda v: str(v)[:40])
def test_simple_mode_hides_the_trigger_screens_even_when_routed_to(
    chromium_browser: Browser, dashboard_url: str, selector: str, leak: str
) -> None:
    """Force the screen active, then assert the mode rule still wins.

    Both screens are hidden on load by the router regardless of mode, so this
    first stamps the router's own ``.on`` class — the exact state a deep link
    or a stale URL would produce — and only then asserts invisibility. Without
    the forcing step the assertion would be true for free.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open(page, dashboard_url, "simple")

    page.evaluate("sel => document.querySelector(sel).classList.add('on')", selector)
    assert not _visible(page, selector), (
        f"Simple mode leaks {leak} ({selector}) when the router activates it -- "
        "the .expert-only rule must beat the .view.on rule"
    )
    page.close()


def test_expert_mode_still_shows_them(chromium_browser: Browser, dashboard_url: str) -> None:
    """The other half of the pin: subtraction must not leak into Expert.

    Without this, deleting a surface outright would satisfy every assertion
    above — the suite would pass hardest on a dashboard that had lost the
    features entirely.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open(page, dashboard_url, "expert")

    for selector in EXPERT_ONLY_SURFACES:
        assert _visible(page, selector), (
            f"Expert mode should show {selector} -- the simple-mode rule is over-reaching"
        )

    # And the routing screens must be reachable in Expert, which is the mirror
    # of the forced-active check above.
    for selector in EXPERT_ONLY_SCREENS:
        page.evaluate("sel => document.querySelector(sel).classList.add('on')", selector)
        assert _visible(page, selector), f"Expert mode should be able to route to {selector}"
    page.close()


def test_simple_mode_speaks_plain_language(chromium_browser: Browser, dashboard_url: str) -> None:
    """Wording is chosen in JS off ``isSimple()``, not by CSS, so it needs its own pin.

    With no reachable dispatcher behind this app the state poll fails, which is
    itself the case worth pinning: an operator must still get a sentence rather
    than a bare counter row.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open(page, dashboard_url, "simple")

    runs_title = page.locator("#runs-title").inner_text()
    assert runs_title == "Recent activity", (
        f"Simple mode should title the list in plain words, got {runs_title!r}"
    )

    status = page.locator("#status-text").inner_text()
    assert status and not any(token in status for token in ("pool", "q:", "err")), (
        f"Simple mode's status line must not carry expert jargon, got {status!r}"
    )
    page.close()
