"""Browser suite: a 401 from the dispatcher must never be reported as an empty stack.

The dashboard's read path polls ``/dashboard/state``, which is bearer-gated. When
that call is rejected the page keeps its initial ``state.triggers = []``, and the
lane list and trigger list both branch on *length*. Unguarded, those branches
turn "we were not allowed to look" into the sentence **"No triggers are
registered."** — an assertion about the dispatcher's configuration that the page
has no evidence for, and which sends an operator to debug a trigger file that is
in fact perfectly healthy.

That is not a cosmetic defect. It is the panel reporting a fact it does not know,
and it is indistinguishable to the reader from the true empty case. So a 401
replaces every view with one centred not-authorised state, and the toolbar's
status chip says "Not authorised".

This suite drives a real browser against a dispatcher stub that answers 401 to
everything, because the bug lives entirely in client-side render branching —
asserting on the Python route or on markup presence would not reach it.

The second axis is *which* advice is honest. A page opened standalone should be
told to open it from the terminal's EVENTS tab. A page ALREADY inside that tab
must not be: there the 401 means the terminal has no token to inject, which is a
deployment fact. That fact names an env var, so it sits behind the state's Info
icon as a tooltip, not on the page.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_dashboard_unauthorized.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from osprey.interfaces._serving import run_app_server
from tests.interfaces.design_system._dispatch_dashboard import mount_design_system

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1100, "height": 900}


#: The claim under test. Present anywhere on an unauthorized page, the panel is
#: asserting the dispatcher has no triggers while holding no evidence either way.
FALSE_EMPTY_CLAIM = "No triggers are registered."

#: The panel-wide state host: a 401 paints the not-authorised state here.
PANEL_STATE = "#panel-state"


def _create_unauthorized_dashboard_app() -> FastAPI:
    """Serve the real dashboard on both entry paths, over a 401-ing dispatcher.

    The same rendered HTML is mounted at the standalone root and under the web
    terminal's panel-proxy prefix, because the page distinguishes the two by its
    own URL — serving one and simulating the other would test the stub.
    """
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    mount_design_system(app)

    def _page() -> str:
        return str(render_dashboard_html(facility_name="Test Facility", channel_strip_prefix="SR:"))

    @app.get("/", response_class=HTMLResponse)
    async def standalone() -> str:
        return _page()

    @app.get("/panel/events/dashboard", response_class=HTMLResponse)
    async def proxied() -> str:
        return _page()

    # Every data endpoint rejects, exactly as the real dispatcher does for a
    # request that carries no bearer.
    # The route declares ``{path:path}``, so FastAPI binds this parameter by name.
    @app.get("/{path:path}")
    async def unauthorized(path: str) -> JSONResponse:  # noqa: ARG001
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    return app


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    with run_app_server(_create_unauthorized_dashboard_app()) as base_url:
        yield base_url


def _open_and_settle(page: Page, url: str) -> str:
    """Load ``url`` and return the panel state's visible text once the 401 is handled.

    Before the first poll answers, the panel state reads "Connecting to the
    dispatcher…"; the poll's 401 repaints it as the not-authorised state, with
    the design-system state component (``.osprey-state``) and its Retry. A page
    that never gets there is reported with what it shows instead of a bare
    timeout.
    """
    page.goto(url, wait_until="domcontentloaded")
    try:
        page.wait_for_function(
            "sel => { const s = document.querySelector(sel + ' .osprey-state');"
            "  return s !== null && /authori[sz]ed/i.test(s.innerText); }",
            arg=PANEL_STATE,
            timeout=15_000,
        )
    except PlaywrightTimeoutError:
        text = page.evaluate("() => document.querySelector('main')?.innerText ?? ''")
        raise AssertionError(
            f"the panel never showed the not-authorised state after the 401: {text!r}"
        ) from None
    return page.inner_text(PANEL_STATE)


@pytest.mark.parametrize("path", ["/", "/?view=triggers"], ids=["activity", "triggers"])
def test_unauthorized_never_claims_no_triggers_are_registered(
    chromium_browser: Browser, dashboard_url: str, path: str
) -> None:
    """A 401 replaces every view with the not-authorised state, never an empty dispatcher."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_and_settle(page, f"{dashboard_url}{path}")
    visible = page.inner_text("main")
    views_hidden = page.evaluate(
        "() => [...document.querySelectorAll('main > .view')]"
        ".every(v => getComputedStyle(v).display === 'none')"
    )
    page.close()

    assert FALSE_EMPTY_CLAIM not in visible, (
        f"the page reported {FALSE_EMPTY_CLAIM!r} after a 401 -- the panel is "
        "asserting the dispatcher has no triggers when it was never allowed to read it"
    )
    assert views_hidden, "a view still shows beside the not-authorised state"
    assert "Not authorised" in visible


def test_not_authorised_state_is_centred_with_retry(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """The state sits in the middle of the pane and offers one Retry."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_and_settle(page, f"{dashboard_url}/")
    geometry = page.evaluate(
        "() => { const m = document.querySelector('main').getBoundingClientRect();"
        "  const s = document.querySelector('#panel-state .osprey-state-message')"
        "    .getBoundingClientRect();"
        "  return { main: (m.top + m.bottom) / 2, msg: (s.top + s.bottom) / 2,"
        "    mainX: (m.left + m.right) / 2, msgX: (s.left + s.right) / 2 }; }"
    )
    buttons = page.locator("#panel-state button.osprey-btn").all_inner_texts()
    page.close()

    assert abs(geometry["main"] - geometry["msg"]) < 60, geometry
    assert abs(geometry["mainX"] - geometry["msgX"]) < 40, geometry
    assert buttons == ["Retry"]


def test_standalone_page_is_told_to_open_the_events_tab(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """Opened directly, the actionable advice is to use the tab that injects a token."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    text = _open_and_settle(page, f"{dashboard_url}/")
    page.close()

    assert "EVENTS tab" in text


def test_in_terminal_page_keeps_the_token_name_behind_the_info_icon(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """Inside the EVENTS tab a 401 means the terminal holds no token. The page
    must not repeat navigation advice already followed, and the env var name is
    a technical detail: it is the Info icon's tooltip, not printed on the page."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    text = _open_and_settle(page, f"{dashboard_url}/panel/events/dashboard?embedded=true")
    tip = page.get_attribute(f"{PANEL_STATE} .osprey-state-message .state-tip", "data-tip")
    page.close()

    assert "EVENT_DISPATCHER_TOKEN" not in text
    assert tip is not None
    assert "EVENT_DISPATCHER_TOKEN" in tip
    assert "EVENTS tab" not in text, (
        "the in-terminal page told the operator to open the EVENTS tab they are already looking at"
    )
