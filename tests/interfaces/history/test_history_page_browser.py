"""Browser suite: the history page as a person sees it, on canned store rows.

The in-process tests (``tests/history/test_history_app.py``) pin the HTML the
page writes. This suite proves what a real browser makes of it: hostile content
stays text and runs nothing, the "viewing as" switcher changes who the page is
viewed as and every other link keeps that choice, the preflight band differs
between an own view and an ``all`` view, a table outside the viewer's grants
reads "not in your tables", a ``/clear``-resumed terminal is one transcript, and
no subresource the page asks for under ``/history/`` is answered 404.

The page runs in process behind a stand-in for nginx and the auth sidecar: it
strips the ``/history`` prefix, drops any ``X-Osprey-History-*`` header the
browser sent, and adds the sidecar's answer for the sign-in the ``as`` parameter
selects. The store is an :class:`httpx.MockTransport` answering each
view-contract query from canned rows chosen by the view user the query runs as.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from osprey.history import queries, viewer
from osprey.history.app import create_app
from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page, Response

pytestmark = [pytest.mark.browser, pytest.mark.slow, pytest.mark.usefixtures("watch")]

ALL_TABLES = "otel_logs,otel_traces,audit_ledger"

#: The sign-ins this browser holds, as the sidecar would answer for each: a
#: roster person (own view), a shared card carrying ``history: all``, and a
#: reader whose tables leave out the conversation.
SIGN_INS: dict[str, dict[str, str]] = {
    "alice": {"person": "alice", "scope": "own", "view": "view_alice", "tables": ALL_TABLES},
    "ops": {"person": "ops", "scope": "all", "view": "view__all", "tables": ALL_TABLES},
    "reader:carol": {
        "person": "carol",
        "scope": "reader",
        "view": "view__reader_carol",
        "tables": "audit_ledger",
    },
}
#: The last sign-in, which the sidecar picks when the URL names none.
DEFAULT_ENTRY = "alice"

SESSION = "3f8a1c02-0000-4000-8000-abcdefabcdef"
RESUMED = "3f8a1c02-0000-4000-8000-000000000002"
BOB_SESSION = "b0b00000-0000-4000-8000-000000000001"

#: Content that would run, or load something, if the page rendered it as markup.
HOSTILE = '<script>window.__pwned = 1</script><img src="pwned.png" onerror="window.__pwned = 1">'

_EXECUTE_AS = re.compile(r"EXECUTE AS `([^`]+)`")
_HISTORY_HEADER_PREFIX = b"x-osprey-history-"


# ---------------------------------------------------------------------------
# The canned store
# ---------------------------------------------------------------------------


def _session_list(view_user: str) -> list[dict[str, Any]]:
    """Alice's terminal ran two sessions (a ``/clear`` resume); ``all`` also sees bob's."""
    rows = [
        {
            "User": "alice",
            "SessionId": SESSION,
            "terminal_session": "term-1",
            "first_ts": "2026-10-11 09:00:00.000000000",
            "last_ts": "2026-10-11 09:30:00.000000000",
            "prompts": 2,
        },
        {
            "User": "alice",
            "SessionId": RESUMED,
            "terminal_session": "term-1",
            "first_ts": "2026-10-11 09:31:00.000000000",
            "last_ts": "2026-10-11 10:00:00.000000000",
            "prompts": 1,
        },
    ]
    if view_user == "view__all":
        rows.append(
            {
                "User": "bob",
                "SessionId": BOB_SESSION,
                "terminal_session": "term-9",
                "first_ts": "2026-10-11 08:00:00.000000000",
                "last_ts": "2026-10-11 08:10:00.000000000",
                "prompts": 1,
            }
        )
    return rows


def _rows(name: str, view_user: str, params: dict[str, str]) -> list[dict[str, Any]]:
    session = params.get("param_session", SESSION)
    if name == "session_list":
        return _session_list(view_user)
    if name == "session_users":
        if session == BOB_SESSION:
            # Bob's session is in no view but ``all``: the store's row policy.
            user = "bob" if view_user == "view__all" else None
        else:
            user = "alice"
        if user is None:
            return []
        return [
            {
                "User": user,
                "terminal_session": "term-1",
                "first_ts": "2026-10-11 10:00:00.000000000",
                "last_ts": "2026-10-11 10:30:00.000000000",
            }
        ]
    if name == "conversation":
        return [
            {
                "User": params.get("param_user", "alice"),
                "SessionId": session,
                "TerminalSession": "term-1",
                "Timestamp": "2026-10-11 10:00:00.000000000",
                "Event": "user_prompt",
                "body_hash": "123",
                "captured": 1,
                "content": HOSTILE,
                "content_length": len(HOSTILE),
            }
        ]
    if name == "tool_log":
        return [
            {
                "User": "alice",
                "SessionId": session,
                "ToolUseId": "toolu_1",
                "Surface": "tool_call",
                "Timestamp": "2026-10-11 10:01:05.000000000",
                "started_at": "2026-10-11 10:01:00.000000000",
                "decision": "allowed",
                "body_hash": "77",
                "body": json.dumps({"subject": "mcp__bluesky__queue_add"}),
                "body_length": queries.PREVIEW_CHARS + 1,
                "arguments_ref": "",
                "result_ref": "",
            }
        ]
    if name == "preflight":
        return [
            {
                "User": "queueserver",
                "OnBehalfOf": "alice",
                "Surface": "preflight",
                "Timestamp": "2026-10-11 10:01:30.000000000",
                "decision": "allowed",
                "body_hash": "1",
                "body": "matched-preflight",
                "body_length": 17,
            },
            {
                "User": "queueserver",
                "OnBehalfOf": "alice",
                "Surface": "preflight",
                "Timestamp": "2026-10-11 10:20:00.000000000",
                "decision": "allowed",
                "body_hash": "2",
                "body": "forged-preflight",
                "body_length": 16,
            },
        ]
    return []


def _query_name(body: str) -> str:
    for name in queries.PAGE_QUERIES:
        if queries.load(name).strip().rstrip(";") in body:
            return name
    raise AssertionError(f"not a contract query: {body[:200]}")


def _store_handler(request: httpx.Request) -> httpx.Response:
    body = request.content.decode()
    match = _EXECUTE_AS.search(body)
    assert match, f"a store call that names no view user: {body[:200]}"
    params = {key: values[-1] for key, values in parse_qs(request.url.query.decode()).items()}
    rows = _rows(_query_name(body), match.group(1), params)
    return httpx.Response(200, text="\n".join(json.dumps(row) for row in rows))


# ---------------------------------------------------------------------------
# nginx and the sidecar, stood in
# ---------------------------------------------------------------------------


def _sidecar_answer(entry: str) -> list[tuple[bytes, bytes]]:
    answer = SIGN_INS[entry]
    choices = ",".join(f"{key}={sign_in['person']}" for key, sign_in in SIGN_INS.items())
    values = {
        viewer.PERSON_HEADER: answer["person"],
        viewer.SCOPE_HEADER: answer["scope"],
        viewer.VIEW_HEADER: answer["view"],
        viewer.TABLES_HEADER: answer["tables"],
        viewer.ENTRY_HEADER: entry,
        viewer.CHOICES_HEADER: choices,
    }
    return [(name.lower().encode(), value.encode()) for name, value in values.items()]


def _nginx_stand_in(app):
    """Strip ``/history``, replace the browser's history headers with the sidecar's answer.

    The ``as`` parameter selects one of :data:`SIGN_INS`; a URL without one is
    viewed as :data:`DEFAULT_ENTRY`. The ``lifespan`` scope passes through so the
    app's startup still runs.
    """
    prefix = viewer.ROOT_PATH

    async def asgi(scope, receive, send):
        if scope["type"] == "http" and (
            scope["path"] == prefix or scope["path"].startswith(f"{prefix}/")
        ):
            query = parse_qs(scope.get("query_string", b"").decode())
            entry = (query.get(viewer.SELECTOR_PARAM) or [DEFAULT_ENTRY])[-1]
            entry = entry if entry in SIGN_INS else DEFAULT_ENTRY
            headers = [
                (name, value)
                for name, value in scope["headers"]
                if not name.lower().startswith(_HISTORY_HEADER_PREFIX)
            ]
            scope = dict(
                scope,
                path=scope["path"][len(prefix) :] or "/",
                headers=headers + _sidecar_answer(entry),
            )
        await app(scope, receive, send)

    return asgi


@contextmanager
def _history_page() -> Iterator[str]:
    """The page behind the stand-in; yields the origin the browser opens ``/history/`` on."""
    app = create_app(
        store_url="http://store.test:8123",
        store_password="history-pw",
        queueservers=("queueserver",),
        max_days=30,
        transport=httpx.MockTransport(_store_handler),
    )
    with _run_app_server(_nginx_stand_in(app)) as base_url:
        yield base_url


@pytest.fixture
def history_url() -> Iterator[str]:
    with _history_page() as base_url:
        yield base_url


class _Watch:
    """Everything one page answered and every dialog it tried to open."""

    def __init__(self, page: Page) -> None:
        self.responses: list[tuple[int, str]] = []
        self.dialogs: list[str] = []
        page.on("response", self._response)
        page.on("dialog", self._dialog)

    def _response(self, response: Response) -> None:
        self.responses.append((response.status, response.url))

    def _dialog(self, dialog) -> None:
        self.dialogs.append(dialog.message)
        dialog.dismiss()

    def missing_under_history(self) -> list[str]:
        return [
            url
            for status, url in self.responses
            if status == 404 and urlsplit(url).path.startswith(viewer.ROOT_PATH + "/")
        ]

    def assets(self) -> list[tuple[int, str]]:
        return [
            (status, url)
            for status, url in self.responses
            if re.search(r"\.(css|js)$", urlsplit(url).path)
        ]


@pytest.fixture
def page(chromium_browser: Browser) -> Iterator[Page]:
    context = chromium_browser.new_context()
    page = context.new_page()
    page.set_default_timeout(10_000)
    yield page
    context.close()


@pytest.fixture
def watch(page: Page) -> Iterator[_Watch]:
    """Watches the page; at the end of every test, nothing under ``/history/`` was 404."""
    seen = _Watch(page)
    yield seen
    assert seen.missing_under_history() == [], "a subresource under /history/ answered 404"
    assert seen.dialogs == [], f"the page opened a dialog: {seen.dialogs}"


def _hrefs(page: Page, selector: str = "a[href]") -> list[str]:
    return page.eval_on_selector_all(selector, "els => els.map(e => e.getAttribute('href'))")


def _selection(href: str) -> list[str]:
    return parse_qs(urlsplit(href).query).get(viewer.SELECTOR_PARAM, [])


def _assert_links_keep(page: Page, entry: str) -> None:
    """Every link outside the switcher names exactly *entry* as its selection."""
    hrefs = _hrefs(page, "a[href]:not(nav.switcher a)")
    assert hrefs, "the page rendered no links"
    for href in hrefs:
        assert urlsplit(href).path.startswith(viewer.ROOT_PATH + "/"), href
        assert _selection(href) == [entry], f"{href} does not keep as={entry}"


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


def test_hostile_content_is_shown_as_text_and_runs_nothing(
    page: Page, watch: _Watch, history_url: str
) -> None:
    page.goto(f"{history_url}/history/session/{SESSION}?as=alice")
    prompt = page.locator("li.entry.event pre.body").first
    assert prompt.inner_text() == HOSTILE
    assert page.locator("main img").count() == 0, "the content became an element"
    assert page.locator("main script").count() == 0
    assert page.evaluate("typeof window.__pwned") == "undefined"
    assert not any("pwned" in url for _status, url in watch.responses)


# ---------------------------------------------------------------------------
# The "viewing as" switcher
# ---------------------------------------------------------------------------


def test_the_switcher_changes_who_the_page_is_viewed_as_and_every_link_keeps_it(
    page: Page, history_url: str
) -> None:
    page.goto(f"{history_url}/history/")
    assert page.locator(".viewing-as .person").inner_text() == "alice"
    assert page.locator(".viewing-as .scope").inner_text() == "own"
    current = page.locator("nav.switcher .choice.current")
    assert current.inner_text() == "alice"
    assert current.get_attribute("aria-current") == "true"
    switch = dict(
        zip(
            page.locator("nav.switcher a.choice").all_inner_texts(),
            _hrefs(page, "nav.switcher a.choice"),
            strict=True,
        )
    )
    assert {person: _selection(href) for person, href in switch.items()} == {
        "ops": ["ops"],
        "carol": ["reader:carol"],
    }
    _assert_links_keep(page, "alice")

    page.locator("nav.switcher a.choice", has_text="ops").click()
    page.wait_for_url(re.compile(r"[?&]as=ops(&|$)"))
    assert page.locator(".viewing-as .person").inner_text() == "ops"
    assert page.locator(".viewing-as .scope").inner_text() == "all"
    assert page.locator("nav.switcher .choice.current").inner_text() == "ops"
    _assert_links_keep(page, "ops")

    page.locator("table.sessions td.session > a", has_text=BOB_SESSION).click()
    page.wait_for_url(re.compile(rf"/history/session/{BOB_SESSION}\?"))
    assert _selection(page.url) == ["ops"]
    assert page.locator(".viewing-as .person").inner_text() == "ops"
    _assert_links_keep(page, "ops")

    page.locator("header h1 a").click()
    page.wait_for_url(re.compile(r"/history/\?as=ops$"))
    assert page.locator(".viewing-as .scope").inner_text() == "all"


# ---------------------------------------------------------------------------
# The preflight band
# ---------------------------------------------------------------------------


def test_an_own_view_shows_only_the_preflight_row_matched_to_a_submit(
    page: Page, history_url: str
) -> None:
    page.goto(f"{history_url}/history/session/{SESSION}?as=alice")
    band = page.locator("li.entry.preflight")
    assert band.count() == 1
    assert "matched-preflight" in band.inner_text()
    assert page.locator("li.entry.preflight.unmatched").count() == 0
    assert "forged-preflight" not in page.locator("main").inner_text()


def test_an_all_view_shows_every_preflight_row_and_labels_the_unmatched(
    page: Page, history_url: str
) -> None:
    page.goto(f"{history_url}/history/session/{SESSION}?as=ops")
    assert page.locator("li.entry.preflight").count() == 2
    unmatched = page.locator("li.entry.preflight.unmatched")
    assert unmatched.count() == 1
    assert "forged-preflight" in unmatched.inner_text()
    assert (
        unmatched.locator(".unmatched-label").inner_text() == "no matching queue submit within 60 s"
    )


# ---------------------------------------------------------------------------
# Not in your tables
# ---------------------------------------------------------------------------


def test_a_reader_without_the_conversation_table_is_told_so(page: Page, history_url: str) -> None:
    page.goto(f"{history_url}/history/?as=reader:carol")
    assert page.locator(".viewing-as .person").inner_text() == "carol"
    assert page.locator(".viewing-as .scope").inner_text() == "reader"
    notice = page.locator("p.notice.not-in-tables")
    assert notice.inner_text() == ("Sessions are read from otel_logs, which is not in your tables.")
    assert page.locator("table.sessions").count() == 0

    page.goto(f"{history_url}/history/session/{SESSION}?as=reader:carol")
    assert "otel_logs" in page.locator("p.notice.not-in-tables").inner_text()
    assert page.locator("section.transcript").count() == 0
    _assert_links_keep(page, "reader:carol")


# ---------------------------------------------------------------------------
# One transcript after a /clear resume
# ---------------------------------------------------------------------------


def test_a_resumed_terminal_is_one_transcript(page: Page, history_url: str) -> None:
    page.goto(f"{history_url}/history/?as=alice")
    rows = page.locator("table.sessions tr.transcript")
    assert rows.count() == 1, "a /clear resume split one terminal into two rows"
    assert page.locator("td.user").all_inner_texts() == ["alice"]
    continued = rows.locator("div.continued")
    assert continued.count() == 1
    assert continued.inner_text() == f"continued as {RESUMED}"

    continued.locator("a").click()
    page.wait_for_url(re.compile(rf"/history/session/{RESUMED}\?"))
    assert page.locator("section.transcript").count() == 1
    assert _selection(page.url) == ["alice"]


# ---------------------------------------------------------------------------
# Every subresource resolves under the prefix
# ---------------------------------------------------------------------------


def test_every_page_loads_its_assets_under_the_prefix(
    page: Page, watch: _Watch, history_url: str
) -> None:
    for path in (
        "/history/?as=alice",
        f"/history/session/{SESSION}?as=alice",
        "/history/?as=reader:carol",
        "/history/?as=ops&older=1",
    ):
        response = page.goto(f"{history_url}{path}")
        assert response is not None and response.status == 200, path
    assets = watch.assets()
    loaded = {urlsplit(url).path for status, url in assets if status == 200}
    assert {
        "/history/design-system/js/theme-boot.js",
        "/history/static/fonts/fonts.css",
        "/history/design-system/css/tokens.css",
        "/history/design-system/css/base.css",
        "/history/static/history.css",
    } <= loaded
    assert [entry for entry in assets if entry[0] != 200] == []
