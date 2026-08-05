"""Browser suite: a run detail links to its trigger and to its own telemetry.

The run detail screen is the end of a drill-in (activity lane -> run), and
without cross-links it is a dead end: the operator can see *that* a run failed
but has to walk back out and hunt to see what configured it, or leave the
dashboard entirely to read what the agent actually did.

Two links close that. Both are *conditional*, and the conditions are the whole
risk. Each depends on data the dashboard does not control:

  * ``trigger_name`` is reverse-mapped from the dispatcher's in-memory pool, so
    it is absent for runs that outlived a dispatcher restart, and the trigger it
    names may since have been deleted from ``triggers.yml``;
  * the telemetry link needs a deployed store (a deploy-time config value) *and*
    the run's forced session id (supplied by the worker).

A link rendered when its destination is not there is worse than no link: it
promises a place to go and lands on "no longer registered", or on a telemetry
page showing every record in the window rather than this run's. So these tests
drive the real rendered dashboard in a browser and assert on the *rendered*
links for each row of that degradation table, rather than on markup the
renderer might never reach.

The telemetry href gets its own assertion because its failure mode is silent:
the store's logs UI drops a query it cannot parse and renders a perfectly normal
page scoped to the whole time window instead. Nothing errors -- the link just
quietly stops meaning "this run", which no smoke test would catch.

Placement rationale: matches ``test_dispatch_dashboard_modes.py`` and
``test_visual.py``, which already stand the dispatch dashboard up from this
directory (it has no standalone FastAPI app of its own in production).

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_run_crosslinks.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import base64
import pathlib
import urllib.parse
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
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "design_system"
    / "static"
)

#: Browser-facing telemetry base the dashboard is configured with. The port is
#: deliberately not the deployed default, so an assertion on a URL built from it
#: cannot pass against a hardcoded fallback.
TELEMETRY_BASE = "http://localhost:5099"

SESSION_ID = "3f8a1c02-0000-4000-8000-abcdefabcdef"
RUN_ID = "run-with-everything"
TRIGGER_NAME = "hello-dispatch"

#: A session id whose query base64-encodes to a PADDED value.
#:
#: The store's UI writes that padding as '.' rather than '='. With OSPREY's own
#: ids the rewrite never fires -- a 36-char UUID happens to make the SQL exactly
#: 81 characters, and base64 only pads when the input length is not a multiple
#: of 3 -- so a test using only a UUID cannot tell the rewrite from its absence.
#: The harness-supplied fallback id (``CLAUDE_CODE_SESSION_ID``, used when OSPREY
#: did not force one) carries no such length guarantee, so the padded case is
#: reachable in production and needs its own pin.
PADDED_SESSION_ID = "01HPROBE-SESSION-AAA"
PADDED_RUN_ID = "run-padded-session"

#: One armed trigger, and runs covering each row of the degradation table.
STATE = {
    "triggers": [
        {"name": TRIGGER_NAME, "status": "armed", "source": "manual", "on_error": "drop"},
    ],
    "runs": [
        {
            "run_id": RUN_ID,
            "trigger_name": TRIGGER_NAME,
            "session_id": SESSION_ID,
            "status": "completed",
            "created_at": 1785744790,
            "duration_sec": 18.4,
            "age_sec": 30,
            "num_turns": 3,
            "tool_count": 2,
            "text_output": "done",
            "tool_calls": [],
        },
        {
            "run_id": "run-no-trigger",
            "trigger_name": None,
            "session_id": SESSION_ID,
            "status": "completed",
            "created_at": 1785744790,
            "age_sec": 30,
            "text_output": "done",
            "tool_calls": [],
        },
        {
            "run_id": "run-deleted-trigger",
            "trigger_name": "trigger-since-deleted",
            "session_id": SESSION_ID,
            "status": "completed",
            "created_at": 1785744790,
            "age_sec": 30,
            "text_output": "done",
            "tool_calls": [],
        },
        {
            "run_id": PADDED_RUN_ID,
            "trigger_name": TRIGGER_NAME,
            "session_id": PADDED_SESSION_ID,
            "status": "completed",
            "created_at": 1785744790,
            "age_sec": 30,
            "text_output": "done",
            "tool_calls": [],
        },
        {
            "run_id": "run-no-session",
            "trigger_name": TRIGGER_NAME,
            "session_id": None,
            "status": "completed",
            "created_at": 1785744790,
            "age_sec": 30,
            "text_output": "done",
            "tool_calls": [],
        },
    ],
}


def _create_dispatch_dashboard_app(telemetry_url: str) -> FastAPI:
    """Serve the real ``render_dashboard_html()`` output beside the real design system."""
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return str(
            render_dashboard_html(
                facility_name="Test Facility",
                pv_strip_prefix="SR:",
                telemetry_url=telemetry_url,
            )
        )

    return app


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    """A dashboard configured WITH a telemetry store."""
    with _run_app_server(_create_dispatch_dashboard_app(TELEMETRY_BASE)) as base_url:
        yield base_url


@pytest.fixture
def dashboard_url_no_telemetry() -> Iterator[str]:
    """A dashboard deployed without a telemetry store (the env var is unset)."""
    with _run_app_server(_create_dispatch_dashboard_app("")) as base_url:
        yield base_url


def _open_run(page: Page, base_url: str, run_id: str) -> None:
    """Load the dashboard, install a known state, and route to one run's detail.

    The state poll has no dispatcher behind it here, so ``state`` is assigned
    directly and the router is driven the same way a click does it -- through
    ``navigate()``, which is what repaints the screen. Expert mode is forced:
    the actions row is ``.expert-only``, so asserting in Simple mode would find
    nothing regardless of whether the links were built.
    """
    page.goto(f"{base_url}/?mode=expert", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.documentElement.getAttribute('data-ui-mode') === 'expert'",
        timeout=10_000,
    )
    page.wait_for_function("() => typeof navigate === 'function'", timeout=10_000)
    page.evaluate(
        """([nextState, runId]) => {
            state.triggers = nextState.triggers;
            state.runs = nextState.runs;
            navigate({ name: 'run', runId });
        }""",
        [STATE, run_id],
    )


def _action_labels(page: Page) -> list[str]:
    """Visible text of every control in the run detail's actions row."""
    return page.evaluate(
        """() => Array.from(
             document.querySelectorAll('#run-detail .detail-actions > *')
           ).map(e => e.textContent.trim())"""
    )


def _telemetry_href(page: Page) -> str | None:
    return page.evaluate(
        """() => {
            const a = document.querySelector('#run-detail .detail-actions a[href]');
            return a ? a.href : null;
        }"""
    )


def test_run_detail_offers_both_cross_links(chromium_browser: Browser, dashboard_url: str) -> None:
    """With a live trigger and a session id, the run links to both."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, RUN_ID)

    labels = _action_labels(page)
    assert any("View trigger" in label for label in labels), (
        f"the run detail should link to the trigger that started it, got {labels!r}"
    )
    assert any("Logs" in label for label in labels), (
        f"the run detail should link to its own telemetry, got {labels!r}"
    )
    page.close()


def test_trigger_link_absent_when_the_run_has_no_trigger(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """A run whose trigger_name did not survive a dispatcher restart gets no link."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, "run-no-trigger")

    labels = _action_labels(page)
    assert not any("View trigger" in label for label in labels), (
        f"a run with no known trigger must not offer a trigger link, got {labels!r}"
    )
    page.close()


def test_trigger_link_absent_when_the_trigger_was_deleted(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """A trigger_name that is no longer registered must not be linked.

    This is the case a presence-only check would miss: the run carries a name,
    so a naive ``if (run.trigger_name)`` renders a link that lands on "that
    trigger is no longer registered".
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, "run-deleted-trigger")

    labels = _action_labels(page)
    assert not any("View trigger" in label for label in labels), (
        f"a deleted trigger must not be linked, got {labels!r}"
    )
    page.close()


def test_telemetry_link_absent_without_a_session_id(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """No session id means no way to scope the query -- so no link."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, "run-no-session")

    assert _telemetry_href(page) is None, (
        "a run with no session id must not offer a telemetry link -- "
        "the query would return every record in the window, not this run's"
    )
    page.close()


def test_telemetry_link_absent_when_no_store_is_deployed(
    chromium_browser: Browser, dashboard_url_no_telemetry: str
) -> None:
    """An unset telemetry URL is the deploy-time "no store" signal."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url_no_telemetry, RUN_ID)

    assert _telemetry_href(page) is None, (
        "with no telemetry store deployed the dashboard must not offer a link to one"
    )
    # The trigger link is independent and must still be there, so this test
    # cannot pass by the actions row having failed to render at all.
    labels = _action_labels(page)
    assert any("View trigger" in label for label in labels), (
        f"the trigger link must not depend on telemetry config, got {labels!r}"
    )
    page.close()


def test_telemetry_link_actually_selects_this_run(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """The href must carry a query scoped to THIS run's session.

    The failure this guards is silent rather than loud: the logs UI drops a
    query parameter it cannot parse and renders a normal-looking page covering
    the whole time window. So the assertion decodes the query the way the store
    does -- base64 with '=' padding rewritten to '.' -- and checks the session id
    is really in it, rather than checking the link merely exists.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, RUN_ID)

    href = _telemetry_href(page)
    assert href is not None, "expected a telemetry link for a run with a session id"
    assert href.startswith(TELEMETRY_BASE), (
        f"the link must point at the configured store, got {href!r}"
    )

    params = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
    assert params.get("sql_mode") == ["true"], (
        f"the query is only applied in SQL mode, got {params.get('sql_mode')!r}"
    )

    encoded = params["query"][0]
    decoded = base64.b64decode(encoded.replace(".", "=")).decode()
    assert SESSION_ID in decoded, (
        f"the query must select this run's session, decoded to {decoded!r}"
    )

    # Pin the encoding exactly, not just that it decodes -- b64decode accepts
    # several spellings of the same payload, so a decode-only check would wave
    # through a format drift.
    expected_sql = f"SELECT * FROM \"default\" WHERE session_id = '{SESSION_ID}'"
    expected_query = base64.b64encode(expected_sql.encode()).decode().replace("=", ".")
    assert encoded == expected_query, (
        f"query encoding drifted from the store's own format: {encoded!r}"
    )

    # The window must bracket the run itself, or the correct query returns
    # nothing.
    started_us = 1785744790 * 1_000_000
    assert int(params["from"][0]) <= started_us, "search window starts after the run did"
    assert int(params["to"][0]) >= started_us + int(18.4 * 1_000_000), (
        "search window ends before the run did"
    )
    page.close()


def test_telemetry_query_rewrites_base64_padding(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """Padded queries must spell their padding '.', the way the store's UI does.

    Uses a session id chosen so the encoding actually pads (see
    ``PADDED_SESSION_ID``) -- with a standard UUID this assertion would be
    vacuously true, since that payload never pads and '=' could never appear
    whether or not the rewrite exists.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, PADDED_RUN_ID)

    href = _telemetry_href(page)
    assert href is not None, "expected a telemetry link for the padded-session run"
    encoded = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)["query"][0]

    # Guard the guard: if this payload stopped padding, the test below would
    # silently stop testing anything.
    unpadded_would_pad = (
        base64.b64encode(
            f"SELECT * FROM \"default\" WHERE session_id = '{PADDED_SESSION_ID}'".encode()
        )
        .decode()
        .endswith("=")
    )
    assert unpadded_would_pad, (
        "PADDED_SESSION_ID no longer produces a padded encoding -- pick another, "
        "or this test proves nothing"
    )

    assert "=" not in encoded, f"base64 padding must be rewritten to '.', got {encoded!r}"
    assert encoded.endswith("."), f"expected the rewritten padding character, got {encoded!r}"
    assert (
        base64.b64decode(encoded.replace(".", "="))
        .decode()
        .endswith(f"session_id = '{PADDED_SESSION_ID}'")
    )
    page.close()


def test_telemetry_link_opens_outside_the_embedded_panel(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """The store is another application, and this dashboard is embedded in an iframe.

    Navigating in place would strand the operator inside the terminal's panel
    with no way back, so the link must open a new context -- and carry
    ``noopener``, since the target is not this app.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, RUN_ID)

    attrs = page.evaluate(
        """() => {
            const a = document.querySelector('#run-detail .detail-actions a[href]');
            return { target: a.getAttribute('target'), rel: a.getAttribute('rel') };
        }"""
    )
    assert attrs["target"] == "_blank", (
        f"the telemetry link must open outside the embedded panel, got {attrs!r}"
    )
    assert "noopener" in (attrs["rel"] or ""), (
        f"a link out of the app must carry noopener, got {attrs!r}"
    )
    page.close()


def test_view_trigger_navigates_to_that_trigger(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """Clicking through lands on the trigger's own screen, not merely a tab switch."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, RUN_ID)

    page.evaluate(
        """() => Array.from(
             document.querySelectorAll('#run-detail .detail-actions button')
           ).find(b => b.textContent.includes('View trigger')).click()"""
    )

    landed = page.evaluate("() => ({ name: view.name, trigger: view.triggerName })")
    assert landed == {"name": "trigger", "trigger": TRIGGER_NAME}, (
        f"expected to land on the trigger detail for {TRIGGER_NAME}, got {landed!r}"
    )
    # And the screen really painted that trigger, not just the router state.
    title = page.locator("#view-trigger .detail-title").inner_text()
    assert title == TRIGGER_NAME, f"trigger detail shows {title!r}"
    page.close()
