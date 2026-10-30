"""Browser suite: a dispatch run's thinking is reached through its history link.

The dashboard's run detail shows no thinking of its own. Its history link opens
the run's session on the history page, and the page shows that session's
thinking, folded, only to the person who asked for the run and to a
``history: all`` viewer. Anyone else following the same link sees nothing of
the session. A deployment without a history page offers no link at all.

Everything behind the link is real: a running agent-record service over an
embedded backend, holding one dispatch run whose ``run:<id>:accept`` entry
names its requester, read by the history page with the ``history`` login the
deploy mints, behind a stand-in for nginx and the auth sidecar that answers for
the one sign-in this browser holds. The dashboard is the real
``render_dashboard_html()`` output handed that page's URL, and the link is
followed by clicking it.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from osprey.agent_record import contract as c
from osprey.agent_record import credentials as cr
from osprey.agent_record import identities as ids
from osprey.agent_record import service
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.client import RecordClient
from osprey.history import viewer
from osprey.history.app import create_app
from osprey.interfaces._serving import run_app_server
from osprey_connectors.dotenv import parse_dotenv_file
from tests.agent_record.test_identities import _config
from tests.interfaces.design_system.test_dispatch_run_crosslinks import HISTORY_LINK_LABEL

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

WORKER = "dispatch-worker-1"
SESSION = "d15ba7c0-0000-4000-8000-000000000001"
RUN_ID = "run-asked-by-alice"
ASKER = "alice"
THINKING = "Compare BPM 3 against its neighbours before blaming the orbit."
ANSWER = "BPM 3 alone drifted."

#: The sign-in this browser holds, as the sidecar would answer for it.
SIGN_INS: dict[str, dict[str, str]] = {
    "alice": {"person": "alice", "scope": "own", "view": "own:alice"},
    "bob": {"person": "bob", "scope": "own", "view": "own:bob"},
    "ops": {"person": "ops", "scope": "all", "view": "all"},
}

_HISTORY_HEADER_PREFIX = b"x-osprey-history-"

NOW = datetime.now(UTC)


def _entry(key: str, kind: str, seconds: int, body: dict[str, Any], **fields: Any) -> c.Entry:
    at = NOW - timedelta(minutes=5) + timedelta(seconds=seconds)
    return c.Entry(
        key=key,
        kind=kind,
        ts=at,
        identity=WORKER,
        body=body,
        session_id=SESSION,
        inserted_at=at,
        **fields,
    )


#: One dispatch run asked for by alice: its accept entry, its thinking, its answer.
ENTRIES = [
    _entry(
        c.run_key(RUN_ID, c.RUN_ACCEPT),
        c.KIND_RUN,
        0,
        {"phase": c.RUN_ACCEPT, "run_id": RUN_ID, "session_id": SESSION, "requester": ASKER},
        run_id=RUN_ID,
    ),
    _entry(f"cc:{WORKER}:{SESSION}:a1:0", c.KIND_THINKING, 2, {"text": THINKING}),
    _entry(f"cc:{WORKER}:{SESSION}:a2:0", c.KIND_ANSWER, 4, {"text": ANSWER}),
]

#: The dashboard's state: one finished run of that session.
STATE = {
    "triggers": [],
    "runs": [
        {
            "run_id": RUN_ID,
            "trigger_name": None,
            "session_id": SESSION,
            "status": "completed",
            "created_at": 1785744790,
            "age_sec": 30,
            "text_output": ANSWER,
            "tool_calls": [],
        }
    ],
}


# ---------------------------------------------------------------------------
# The record, the history page behind nginx and the sidecar, the dashboard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record(tmp_path_factory) -> Iterator[dict[str, str]]:
    """A running agent-record service holding the run, and the history login."""
    repo = tmp_path_factory.mktemp("repo")
    config = _config(workers=1)
    config["project_root"] = str(repo)
    cr.ensure_agent_record_credentials(repo, config, reader_keys=False)
    token = cr.read_login(repo / cr.login_file(ids.HISTORY_IDENTITY), ids.HISTORY_IDENTITY)
    assert token, "the deploy minted no history login"

    service_env = parse_dotenv_file(repo / cr.SERVICE_FILE)
    root = tmp_path_factory.mktemp("service")
    env = {
        cr.VERIFIER_ENV: service_env[cr.VERIFIER_ENV],
        cr.SECRETS_ENV: service_env[cr.SECRETS_ENV],
        service.SPOOL_ROOT_ENV: str(root / "spool"),
        service.RECORD_DIR_ENV: str(root / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(root / "transcripts"),
    }
    with EmbeddedBackend(root / "record.duckdb") as backend:
        backend.append(ENTRIES)
        app = service.create_app(config=config, backend=backend, env=env)
        with run_app_server(app) as url:
            yield {"url": url, "token": token}


class _SignedIn:
    """Which sign-in the stand-in sidecar answers for; each test sets it."""

    entry = "alice"


def _sidecar_answer(entry: str) -> list[tuple[bytes, bytes]]:
    answer = SIGN_INS[entry]
    values = {
        viewer.PERSON_HEADER: answer["person"],
        viewer.SCOPE_HEADER: answer["scope"],
        viewer.VIEW_HEADER: answer["view"],
        viewer.ENTRY_HEADER: entry,
        viewer.CHOICES_HEADER: f"{entry}={answer['person']}",
    }
    return [(name.lower().encode(), value.encode()) for name, value in values.items()]


def _nginx_stand_in(app, signed_in: type[_SignedIn]):
    """Strip ``/history``; replace the browser's history headers with the sidecar's answer."""
    prefix = viewer.ROOT_PATH

    async def asgi(scope, receive, send):
        if scope["type"] == "http" and (
            scope["path"] == prefix or scope["path"].startswith(f"{prefix}/")
        ):
            headers = [
                (name, value)
                for name, value in scope["headers"]
                if not name.lower().startswith(_HISTORY_HEADER_PREFIX)
            ]
            scope = dict(
                scope,
                path=scope["path"][len(prefix) :] or "/",
                headers=headers + _sidecar_answer(signed_in.entry),
            )
        await app(scope, receive, send)

    return asgi


@pytest.fixture
def history_origin(record: dict[str, str]) -> Iterator[str]:
    """The history page, reading the record with the minted ``history`` login.

    Built per test: the page's sign-in gate caches the process credential
    holder when the app is built, and that holder is replaced before every test.
    """

    def client_for(record_viewer: str) -> RecordClient:
        return RecordClient(
            record["url"], ids.HISTORY_IDENTITY, record["token"], viewer=record_viewer
        )

    app = create_app(env={}, max_days=30, client_for=client_for)
    with run_app_server(_nginx_stand_in(app, _SignedIn)) as base_url:
        yield base_url


def _dashboard_app(history_url: str) -> FastAPI:
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
def dashboard_url(history_origin: str) -> Iterator[str]:
    """The dashboard of a deployment that serves the history page."""
    with run_app_server(_dashboard_app(f"{history_origin}/history/")) as base_url:
        yield base_url


@pytest.fixture
def dashboard_url_without_history() -> Iterator[str]:
    """The dashboard of a deployment with no history page."""
    with run_app_server(_dashboard_app("")) as base_url:
        yield base_url


@pytest.fixture
def signed_in() -> Iterator[type[_SignedIn]]:
    yield _SignedIn
    _SignedIn.entry = "alice"


def _open_run(page: Page, base_url: str) -> None:
    """Load the dashboard in Expert mode, install the state, route to the run's detail."""
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
        [STATE, RUN_ID],
    )


def _history_link(page: Page):
    return page.locator("#run-detail .detail-actions a", has_text="history")


def _follow_history_link(browser: Browser, base_url: str) -> Page:
    """Open the run on the dashboard, click its history link, return the tab it opened."""
    context = browser.new_context(viewport=VIEWPORT)
    page = context.new_page()
    page.set_default_timeout(10_000)
    _open_run(page, base_url)
    link = _history_link(page)
    assert link.count() == 1, "the run detail offers no history link"
    assert link.inner_text().strip() == HISTORY_LINK_LABEL
    assert "thinking" not in page.locator("#run-detail").inner_text().replace(
        HISTORY_LINK_LABEL, ""
    ), "the run detail shows thinking of its own"
    with context.expect_page() as opened:
        link.click()
    history = opened.value
    history.wait_for_load_state("domcontentloaded")
    assert f"/history/session/{SESSION}" in history.url, history.url
    person = SIGN_INS[_SignedIn.entry]["person"]
    assert history.locator(".viewing-as .person").inner_text() == person, (
        "the history page did not render for the signed-in viewer"
    )
    return history


def _thinking(history: Page):
    return history.locator("section.transcript li.entry.thinking")


# ---------------------------------------------------------------------------
# Who sees the thinking behind the link
# ---------------------------------------------------------------------------


def _assert_thinking_folded_then_shown(history: Page) -> None:
    thinking = _thinking(history)
    assert thinking.count() == 1, history.content()
    fold = thinking.locator("details.thinking")
    assert fold.get_attribute("open") is None, "the thinking is not folded"
    assert not thinking.locator("pre.body").is_visible()
    thinking.locator("summary").click()
    assert THINKING in thinking.locator("pre.body").inner_text()


def test_the_asker_follows_the_link_and_sees_the_runs_thinking_folded(
    chromium_browser: Browser, dashboard_url: str, signed_in: type[_SignedIn]
) -> None:
    signed_in.entry = "alice"
    history = _follow_history_link(chromium_browser, dashboard_url)
    _assert_thinking_folded_then_shown(history)
    assert ANSWER in history.locator("section.transcript").inner_text()
    history.context.close()


def test_an_all_users_viewer_follows_the_link_and_sees_the_runs_thinking(
    chromium_browser: Browser, dashboard_url: str, signed_in: type[_SignedIn]
) -> None:
    signed_in.entry = "ops"
    history = _follow_history_link(chromium_browser, dashboard_url)
    _assert_thinking_folded_then_shown(history)
    history.context.close()


def test_another_user_follows_the_link_and_sees_nothing_of_the_run(
    chromium_browser: Browser, dashboard_url: str, signed_in: type[_SignedIn]
) -> None:
    signed_in.entry = "bob"
    history = _follow_history_link(chromium_browser, dashboard_url)
    assert _thinking(history).count() == 0
    content = history.content()
    assert THINKING not in content, "another user saw the run's thinking"
    assert ANSWER not in content, "another user saw the run's answer"
    history.context.close()


def test_no_history_page_means_no_link(
    chromium_browser: Browser, dashboard_url_without_history: str
) -> None:
    context = chromium_browser.new_context(viewport=VIEWPORT)
    page = context.new_page()
    _open_run(page, dashboard_url_without_history)
    assert page.locator("#run-detail .detail-title").count() == 1, "the run detail did not paint"
    assert _history_link(page).count() == 0
    assert page.locator("#run-detail a[href*='history']").count() == 0
    context.close()
