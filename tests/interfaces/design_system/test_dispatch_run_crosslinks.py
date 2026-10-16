"""Browser suite: a run detail links to its trigger and to its session's history.

The run detail screen is the end of a drill-in (activity lane -> run), and
without a cross-link it is a dead end: the operator can see *that* a run failed
but has to walk back out and hunt to see what configured it.

The link is *conditional*: ``trigger_name`` is reverse-mapped from the
dispatcher's in-memory pool, so it is absent for runs that outlived a dispatcher
restart, and the trigger it names may since have been deleted from
``triggers.yml``. A link rendered when its destination is not there promises a
place to go and lands on "no longer registered", so these tests drive the real
rendered dashboard in a browser and assert on the *rendered* link for each row
of that degradation table.

The history link needs the history page (a deploy-time ``OSPREY_HISTORY_URL``,
set only when the deployment serves the page) and a session id the link can
carry; without either the run offers no link.

Placement rationale: matches ``test_dispatch_dashboard_modes.py`` and
``test_visual.py``, which already stand the dispatch dashboard up from this
directory (it has no standalone FastAPI app of its own in production).

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_run_crosslinks.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pathlib
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


#: The history page's URL as the dispatcher is handed it: the readers' origin.
HISTORY_URL = "https://readers.example.org:10002/history/"


def _create_dispatch_dashboard_app(history_url: str = "") -> FastAPI:
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
                channel_strip_prefix="SR:",
                history_url=history_url,
            )
        )

    return app


@pytest.fixture
def dashboard_url() -> Iterator[str]:
    """The dashboard as the dispatcher serves it."""
    with _run_app_server(_create_dispatch_dashboard_app()) as base_url:
        yield base_url


@pytest.fixture
def dashboard_url_with_history() -> Iterator[str]:
    """The dashboard of a deployment that serves the history page."""
    with _run_app_server(_create_dispatch_dashboard_app(HISTORY_URL)) as base_url:
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


def test_run_detail_links_to_its_trigger(chromium_browser: Browser, dashboard_url: str) -> None:
    """With a live trigger, the run links to it, and to nothing outside the app."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, RUN_ID)

    labels = _action_labels(page)
    assert any("View trigger" in label for label in labels), (
        f"the run detail should link to the trigger that started it, got {labels!r}"
    )
    assert not any("Logs" in label for label in labels), (
        f"the run detail links to no telemetry store, got {labels!r}"
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


def _history_link(page: Page) -> dict | None:
    """The run detail's history link: its href, target, rel and label, or None."""
    return page.evaluate(
        """() => {
             const a = Array.from(document.querySelectorAll('#run-detail .detail-actions a'))
               .find(e => e.textContent.includes('history'));
             return a ? { href: a.getAttribute('href'), target: a.target, rel: a.rel,
                          label: a.textContent.trim() } : null;
           }"""
    )


def test_the_run_links_to_its_session_on_the_history_page(
    chromium_browser: Browser, dashboard_url_with_history: str
) -> None:
    """With the page deployed, the run opens its session there, in a new tab."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url_with_history, RUN_ID)

    link = _history_link(page)
    assert link == {
        "href": f"{HISTORY_URL}session/{SESSION_ID}",
        "target": "_blank",
        "rel": "noopener noreferrer",
        "label": "history (history: all and readers)",
    }, link
    page.close()


def test_no_history_link_without_the_history_page(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """An unset OSPREY_HISTORY_URL is the deploy-time "no page" signal."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url, RUN_ID)

    assert _history_link(page) is None
    labels = _action_labels(page)
    assert any("View trigger" in label for label in labels), labels
    page.close()


def test_no_history_link_without_a_session_id(
    chromium_browser: Browser, dashboard_url_with_history: str
) -> None:
    """A run with no session id has nothing to open on the history page."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_run(page, dashboard_url_with_history, "run-no-session")

    assert _history_link(page) is None
    page.close()
