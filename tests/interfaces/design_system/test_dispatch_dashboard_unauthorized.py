"""Browser suite: a 401 from the dispatcher must never be reported as an empty stack.

The dashboard's read path polls ``/dashboard/state``, which is bearer-gated. When
that call is rejected the page keeps its initial ``state.triggers = []``, and the
lane list and trigger list both branch on *length*. Unguarded, those branches
turn "we were not allowed to look" into the sentence **"No triggers are
registered."** — an assertion about the dispatcher's configuration that the page
has no evidence for, and which sends an operator to debug a trigger file that is
in fact perfectly healthy.

That is not a cosmetic defect. It is the panel reporting a fact it does not know,
and it is indistinguishable to the reader from the true empty case. The status
line and run list already carried an ``authBlocked`` branch; the two trigger
surfaces did not.

This suite drives a real browser against a dispatcher stub that answers 401 to
everything, because the bug lives entirely in client-side render branching —
asserting on the Python route or on markup presence would not reach it.

The second axis is *which* advice is honest. A page opened standalone should be
told to open it from the terminal's EVENTS tab. A page ALREADY inside that tab
must not be: there the 401 means the terminal has no token to inject, which is a
deployment fact, and repeating the navigation advice is the failure mode that
made this bug hard to read in the first place.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_dashboard_unauthorized.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1100, "height": 900}

DESIGN_SYSTEM_STATIC_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "design_system"
    / "static"
)

#: The claim under test. Present anywhere on an unauthorized page, the panel is
#: asserting the dispatcher has no triggers while holding no evidence either way.
FALSE_EMPTY_CLAIM = "No triggers are registered."

#: The two trigger surfaces that branch on `state.triggers.length`. `#lane-list`
#: renders on the default view; `#trigger-list` needs the Triggers screen, which
#: is expert-only — hence the explicit mode and view in the URLs below.
LANE_LIST = "#lane-list"
TRIGGER_LIST = "#trigger-list"


def _create_unauthorized_dashboard_app() -> FastAPI:
    """Serve the real dashboard on both entry paths, over a 401-ing dispatcher.

    The same rendered HTML is mounted at the standalone root and under the web
    terminal's panel-proxy prefix, because the page distinguishes the two by its
    own URL — serving one and simulating the other would test the stub.
    """
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )

    def _page() -> str:
        return str(render_dashboard_html(facility_name="Test Facility", pv_strip_prefix="SR:"))

    @app.get("/", response_class=HTMLResponse)
    async def standalone() -> str:
        return _page()

    @app.get("/panel/events/dashboard", response_class=HTMLResponse)
    async def proxied() -> str:
        return _page()

    # Every data endpoint rejects, exactly as the real dispatcher does for a
    # request that carries no bearer.
    @app.get("/{path:path}")
    async def unauthorized(path: str) -> JSONResponse:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    return app


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    with _run_app_server(_create_unauthorized_dashboard_app()) as base_url:
        yield base_url


def _open_and_settle(page: Page, url: str, selector: str) -> str:
    """Load ``url`` and return ``selector``'s text once the 401 has been handled.

    The page paints twice. ``init()`` renders synchronously from its empty
    initial state before the first poll is sent, and that paint is an
    ``.empty-state`` reading ``FALSE_EMPTY_CLAIM`` with nothing read yet; the
    poll's 401 then repaints it as the authorisation explanation. Waiting for an
    empty-state to *exist* is satisfied by the first paint, so the condition is
    an empty-state that is not the pre-poll placeholder. A page that never moves
    past that placeholder is reported as the false-empty claim it is, not as a
    bare timeout.
    """
    page.goto(url, wait_until="domcontentloaded")
    try:
        page.wait_for_function(
            "([sel, placeholder]) => { const e = document.querySelector(sel);"
            "  const s = e && e.querySelector('.empty-state');"
            "  return s !== null && s.innerText.trim() !== placeholder; }",
            arg=[selector, FALSE_EMPTY_CLAIM],
            timeout=15_000,
        )
    except PlaywrightTimeoutError:
        text = page.evaluate("sel => document.querySelector(sel)?.innerText ?? ''", selector)
        raise AssertionError(
            f"{selector} never repainted after the 401 -- still showing the pre-poll "
            f"placeholder: {text!r}"
        ) from None
    return page.evaluate("sel => document.querySelector(sel).innerText", selector)


@pytest.mark.parametrize(
    "path,selector",
    [
        ("/?mode=expert", LANE_LIST),
        ("/?mode=expert&view=triggers", TRIGGER_LIST),
    ],
    ids=["timeline-lanes", "trigger-list"],
)
def test_unauthorized_never_claims_no_triggers_are_registered(
    chromium_browser: Browser, dashboard_url: str, path: str, selector: str
) -> None:
    """Neither trigger surface may report an empty dispatcher when it got a 401."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    text = _open_and_settle(page, f"{dashboard_url}{path}", selector)
    page.close()

    assert FALSE_EMPTY_CLAIM not in text, (
        f"{selector} reported {FALSE_EMPTY_CLAIM!r} after a 401 -- the panel is "
        "asserting the dispatcher has no triggers when it was never allowed to read it"
    )
    assert "authorised" in text.lower() or "authorized" in text.lower(), (
        f"{selector} showed neither the false-empty claim nor an authorization "
        f"explanation; an operator is left with no account of the blank panel: {text!r}"
    )


def test_standalone_page_is_told_to_open_the_events_tab(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """Opened directly, the actionable advice is to use the tab that injects a token."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    text = _open_and_settle(page, f"{dashboard_url}/?mode=expert", LANE_LIST)
    page.close()

    assert "EVENTS tab" in text


def test_in_terminal_page_is_not_told_to_open_the_tab_it_is_already_in(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """Inside the EVENTS tab a 401 means the terminal holds no token, so the page
    must name that instead of repeating navigation advice already followed."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    text = _open_and_settle(page, f"{dashboard_url}/panel/events/dashboard?mode=expert", LANE_LIST)
    page.close()

    assert "EVENT_DISPATCHER_TOKEN" in text
    assert "EVENTS tab" not in text, (
        "the in-terminal page told the operator to open the EVENTS tab they are already looking at"
    )
