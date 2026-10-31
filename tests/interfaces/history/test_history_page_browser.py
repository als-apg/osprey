"""Browser suite: the history page as a person sees it, on canned record answers.

The in-process tests (``tests/history/test_history_app.py``) pin the HTML the
page writes. This suite proves what a real browser makes of it, checked
against the approved one-column layout: the session list on top, newest first,
with an all-sources filter and dispatch runs shown as "(no prompt)"; the chosen
session's timeline below it, thinking folded to one line, every tool call with
its decision and result, subagent steps nested and folded under the call that
started them; paging through ``next`` at the bottom and a re-read offered when
newer entries arrived. Hostile content stays text and runs nothing, the
"viewing as" switcher changes who the page is viewed as and every other link
keeps that choice, and no subresource under ``/history/`` is answered 404.

The page runs in process behind a stand-in for nginx and the auth sidecar: it
strips the ``/history`` prefix, drops any ``X-Osprey-History-*`` header the
browser sent, and adds the sidecar's answer for the sign-in the ``as``
parameter selects. The record is a
:class:`~tests.agent_record.fakes.FakeRecordClient` per record viewer; there
is no store behind it.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import pytest

from osprey.agent_record.client import ClientAnswer
from osprey.history import viewer
from osprey.history.app import create_app
from osprey.interfaces._serving import run_app_server
from tests.agent_record.fakes import FakeRecordClient

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page, Response

pytestmark = [pytest.mark.browser, pytest.mark.slow, pytest.mark.usefixtures("watch")]

#: The sign-ins this browser holds, as the sidecar would answer for each: a
#: roster person (own view), a shared card carrying ``history: all``, and a reader.
SIGN_INS: dict[str, dict[str, str]] = {
    "alice": {"person": "alice", "scope": "own", "view": "own:alice"},
    "ops": {"person": "ops", "scope": "all", "view": "all"},
    "reader:carol": {"person": "carol", "scope": "reader", "view": "reader:carol"},
}
#: The last sign-in, which the sidecar picks when the URL names none.
DEFAULT_ENTRY = "alice"

SESSION = "3f8a1c02-0000-4000-8000-abcdefabcdef"
DISPATCH = "d15ba7c0-0000-4000-8000-000000000001"
CRON = "c0c0c0c0-0000-4000-8000-000000000001"
BOB_SESSION = "b0b00000-0000-4000-8000-000000000001"
CURSOR = "CURSOR-1"

#: Content that would run, or load something, if the page rendered it as markup.
HOSTILE = '<script>window.__pwned = 1</script><img src="pwned.png" onerror="window.__pwned = 1">'
THINKING = "Compare BPM 3 against its neighbours before blaming the orbit. " * 70

_HISTORY_HEADER_PREFIX = b"x-osprey-history-"


# ---------------------------------------------------------------------------
# The canned record
# ---------------------------------------------------------------------------


def _row(session_id: str, ts: str, identity: str, person: str, prompt: str | None) -> dict:
    return {
        "session_id": session_id,
        "persons": [person],
        "identities": [identity],
        "first_ts": ts,
        "last_ts": ts,
        "entries": 6,
        "prompt": prompt,
        "run_ids": [] if prompt else ["run-1"],
        "terminal_session": "",
    }


def _sessions(view: str) -> list[dict[str, Any]]:
    """Newest first, as the record lists them; ``all`` also sees bob's session."""
    rows = [
        _row(SESSION, "2026-10-24T10:42:00+00:00", "alice", "alice", HOSTILE),
        _row(DISPATCH, "2026-10-24T10:15:10+00:00", "dispatch-worker-1", "alice", None),
        _row(CRON, "2026-10-24T06:00:00+00:00", "dispatch-worker-2", "alice", None),
    ]
    if view == "all":
        rows.append(_row(BOB_SESSION, "2026-10-23T08:10:00+00:00", "bob", "bob", "bob's question"))
    return rows


def _users(view: str) -> list[dict[str, Any]]:
    users = [{"person": "alice", "sessions": 3, "last_ts": "2026-10-24T10:42:00+00:00"}]
    if view == "all":
        users.append({"person": "bob", "sessions": 1, "last_ts": "2026-10-23T08:10:00+00:00"})
    return users


def _node(kind: str, ts: str, body: Any, **extra: Any) -> dict[str, Any]:
    return {
        "key": f"k-{kind}-{ts}",
        "kind": kind,
        "ts": ts,
        "person": "alice",
        "identity": "alice",
        "session_id": SESSION,
        "body": body,
        "incomplete": False,
        "decision": None,
        "result": None,
        "subagent": None,
        **extra,
    }


def _call(
    ts: str, name: str, value: str, decision: str, source: str, result: str, **extra: Any
) -> dict[str, Any]:
    return _node(
        "tool_call",
        ts,
        {"id": f"toolu-{ts}", "name": name, "input": {"channel": value}},
        decision=_node("decision", ts, {"decision": decision, "source": source}),
        result=_node(
            "tool_result",
            ts,
            {"tool_use_id": f"toolu-{ts}", "content": [{"type": "text", "text": result}]},
        ),
        **extra,
    )


def _first_page() -> list[dict[str, Any]]:
    """The mock-up's session: thinking, a read, a subagent of three steps, a refused write."""
    return [
        _node("prompt", "2026-10-24T10:15:00+00:00", {"text": HOSTILE}),
        _node("thinking", "2026-10-24T10:15:02+00:00", {"text": THINKING}),
        _call(
            "2026-10-24T10:15:04+00:00", "read_channel", "SR:BPM3:X", "accept", "config", "0.41 mm"
        ),
        _node(
            "tool_call",
            "2026-10-24T10:15:05+00:00",
            {"id": "toolu-agent", "name": "Agent", "input": {"description": "check neighbours"}},
            decision=_node(
                "decision", "2026-10-24T10:15:05+00:00", {"decision": "accept", "source": "config"}
            ),
            result=_node(
                "subagent", "2026-10-24T10:15:09+00:00", {"agent_id": "a-1", "status": "completed"}
            ),
            subagent={
                "agent_id": "a-1",
                "timeline": [
                    _node("thinking", "2026-10-24T10:15:06+00:00", {"text": "neighbours " * 100}),
                    _call(
                        "2026-10-24T10:15:07+00:00",
                        "read_channel",
                        "SR:BPM2:X",
                        "accept",
                        "config",
                        "0.02 mm",
                    ),
                    _call(
                        "2026-10-24T10:15:08+00:00",
                        "read_channel",
                        "SR:BPM4:X",
                        "accept",
                        "user_temporary",
                        "0.03 mm",
                    ),
                ],
            },
        ),
        _call(
            "2026-10-24T10:15:09+00:00",
            "write_channel",
            "SR:COR3",
            "reject",
            "config",
            "limit exceeded",
        ),
    ]


def _second_page() -> list[dict[str, Any]]:
    return [
        _node(
            "answer",
            "2026-10-24T10:15:10+00:00",
            {"text": "BPM 3 alone drifted; corrector write refused"},
        )
    ]


def _timeline(session_id: str, after: str | None = None, **_: Any) -> ClientAnswer:
    if session_id != SESSION:
        return ClientAnswer(
            data=[_node("prompt", "2026-10-23T08:00:00+00:00", {"text": "bob's question"})],
            complete=True,
        )
    if after == CURSOR:
        return ClientAnswer(data=_second_page(), complete=True, newer=True)
    return ClientAnswer(data=_first_page(), complete=True, next=CURSOR, more=True)


class Record:
    """One fake client per record viewer, answering per viewer."""

    def __init__(self) -> None:
        self.clients: list[FakeRecordClient] = []

    def client_for(self, record_viewer: str) -> FakeRecordClient:
        client = FakeRecordClient(
            {
                "list_sessions": ClientAnswer(data=_sessions(record_viewer), complete=True),
                "session_users": ClientAnswer(data=_users(record_viewer), complete=True),
                "timeline": _timeline,
            },
            login="history",
            viewer=record_viewer,
        )
        self.clients.append(client)
        return client

    def timeline_calls(self) -> list[dict[str, Any]]:
        return [dict(kwargs) for c in self.clients for kwargs in c.calls_to("timeline")]


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


@pytest.fixture
def record() -> Record:
    return Record()


@contextmanager
def _history_page(record: Record) -> Iterator[str]:
    """The page behind the stand-in; yields the origin the browser opens ``/history/`` on."""
    app = create_app(env={}, max_days=30, client_for=record.client_for)
    with run_app_server(_nginx_stand_in(app)) as base_url:
        yield base_url


@pytest.fixture
def history_url(record: Record) -> Iterator[str]:
    with _history_page(record) as base_url:
        yield base_url


class _Watch:
    """Everything one page answered, every dialog it tried to open, every script error."""

    def __init__(self, page: Page) -> None:
        self.responses: list[tuple[int, str]] = []
        self.dialogs: list[str] = []
        self.errors: list[str] = []
        page.on("response", self._response)
        page.on("dialog", self._dialog)
        page.on("pageerror", lambda error: self.errors.append(str(error)))

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
    """Watches the page; at the end of every test nothing under ``/history/`` was
    404, no dialog opened and no script threw."""
    seen = _Watch(page)
    yield seen
    assert seen.missing_under_history() == [], "a subresource under /history/ answered 404"
    assert seen.dialogs == [], f"the page opened a dialog: {seen.dialogs}"
    assert seen.errors == [], f"a script on the page threw: {seen.errors}"


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


def _line(page: Page, selector: str) -> str:
    return " ".join(page.locator(selector).first.inner_text().split())


# ---------------------------------------------------------------------------
# The session list: newest first, all sources, dispatch runs without a prompt
# ---------------------------------------------------------------------------


def test_the_session_list_is_one_column_newest_first(page: Page, history_url: str) -> None:
    page.goto(f"{history_url}/history/?as=alice")
    rows = page.locator("ol.sessions li.session")
    assert rows.count() == 3
    texts = [" ".join(t.split()) for t in rows.all_inner_texts()]
    assert texts[0].startswith("▸ 10:42 web · alice")
    assert texts[1] == f"▸ 10:15 dispatch · alice (no prompt) {DISPATCH}"
    assert texts[2] == f"▸ 06:00 dispatch · alice (no prompt) {CRON}"
    assert page.locator("ol.sessions li.session .source").nth(1).get_attribute("title") == (
        "dispatch-worker-1"
    )
    assert page.locator("ol.sessions li.day").all_inner_texts() == ["2026-10-24"]


def test_the_source_filter_starts_at_all_sources_and_narrows_the_list(
    page: Page, history_url: str
) -> None:
    page.goto(f"{history_url}/history/?as=alice")
    select = page.locator(".list-head select")
    assert select.is_visible(), "the all-sources filter is not shown"
    assert select.locator("option:checked").inner_text() == "all sources"
    assert select.locator("option").all_inner_texts() == ["all sources", "dispatch", "web"]
    assert page.locator("ol.sessions li.session:visible").count() == 3

    select.select_option("dispatch")
    visible = page.locator("ol.sessions li.session:visible")
    assert visible.count() == 2
    assert all("(no prompt)" in t for t in visible.all_inner_texts())

    select.select_option("")
    assert page.locator("ol.sessions li.session:visible").count() == 3


# ---------------------------------------------------------------------------
# The chosen session's timeline
# ---------------------------------------------------------------------------


def test_the_timeline_sits_below_the_chosen_session(page: Page, history_url: str) -> None:
    page.goto(f"{history_url}/history/?as=alice")
    page.locator("ol.sessions li.session a.row").first.click()
    page.wait_for_url(re.compile(rf"/history/session/{SESSION}\?"))
    current = page.locator(".session-list .session.current")
    assert " ".join(current.inner_text().split()).startswith("▾ 10:15 web · alice")
    order = page.evaluate(
        """() => {
            const list = document.querySelector('.session-list');
            const transcript = document.querySelector('section.transcript');
            return list.compareDocumentPosition(transcript) & Node.DOCUMENT_POSITION_FOLLOWING;
        }"""
    )
    assert order, "the timeline is not below the session list"


def test_thinking_is_folded_to_one_line_until_shown(page: Page, history_url: str) -> None:
    page.goto(f"{history_url}/history/session/{SESSION}?as=alice")
    thinking = page.locator("section.transcript > ol > li.entry.thinking").first
    fold = thinking.locator("details.thinking")
    assert fold.get_attribute("open") is None
    summary = " ".join(thinking.locator("summary").inner_text().split())
    assert re.fullmatch(r"10:15:02 ⋯ thinking \(~1\.1k tokens\)", summary), summary
    body = thinking.locator("pre.body")
    assert not body.is_visible()
    thinking.locator("summary").click()
    assert body.is_visible()
    assert body.inner_text().startswith("Compare BPM 3")


def test_every_tool_call_shows_its_decision_and_result(page: Page, history_url: str) -> None:
    page.goto(f"{history_url}/history/session/{SESSION}?as=alice")
    calls = page.locator("section.transcript > ol > li.entry.tool_call > details.call > summary")
    lines = [" ".join(t.split()) for t in calls.all_inner_texts()]
    assert lines == [
        "10:15:04 read_channel SR:BPM3:X allowed → 0.41 mm",
        "10:15:09 write_channel SR:COR3 denied → limit exceeded",
    ]
    denied = page.locator("li.entry.tool_call .verdict.denied .decision")
    assert denied.inner_text() == "denied"
    first = page.locator("section.transcript > ol > li.entry.tool_call").first
    detail = first.locator(".detail")
    assert not detail.is_visible()
    first.locator("summary").click()
    assert detail.is_visible()
    assert "SR:BPM3:X" in detail.locator(".input").inner_text()
    assert "0.41 mm" in detail.locator(".result").inner_text()


def test_subagent_steps_nest_folded_under_the_call_that_started_them(
    page: Page, history_url: str
) -> None:
    page.goto(f"{history_url}/history/session/{SESSION}?as=alice")
    agent = page.locator("section.transcript > ol > li.entry.tool_call:has(details.agent)")
    assert agent.count() == 1
    summary = agent.locator("details.agent > summary")
    assert " ".join(summary.inner_text().split()) == (
        "10:15:05 Agent “check neighbours” allowed [3 steps]"
    )
    steps = agent.locator("details.agent > ol.entries.subagent > li.entry")
    assert steps.count() == 3
    assert not steps.first.is_visible(), "subagent steps are not folded"

    summary.click()
    assert steps.first.is_visible()
    assert [s.split()[0] for s in steps.all_inner_texts()] == ["10:15:06", "10:15:07", "10:15:08"]
    nested = [
        " ".join(t.split())
        for t in agent.locator("ol.entries.subagent > li.entry.tool_call summary").all_inner_texts()
    ]
    assert nested == [
        "10:15:07 read_channel SR:BPM2:X allowed → 0.02 mm",
        "10:15:08 read_channel SR:BPM4:X asked → 0.03 mm",
    ]
    nested_thinking = agent.locator("ol.entries.subagent > li.entry.thinking details.thinking")
    assert nested_thinking.get_attribute("open") is None


def test_a_long_session_pages_through_next_and_offers_a_reread_when_newer(
    page: Page, history_url: str, record: Record
) -> None:
    page.goto(f"{history_url}/history/session/{SESSION}?as=alice")
    assert page.locator("li.entry.answer").count() == 0
    nxt = page.locator("nav.paging a.next")
    assert nxt.inner_text() == "later entries ↓"
    nxt.click()
    page.wait_for_url(re.compile(r"[?&]after=CURSOR-1(&|$)"))
    assert record.timeline_calls()[-1] == {"session_id": SESSION, "after": CURSOR}
    assert "BPM 3 alone drifted" in page.locator("li.entry.answer pre.body").inner_text()
    assert page.locator("nav.paging a.next").count() == 0

    refresh = page.locator("p.notice.newer a.refresh")
    assert refresh.inner_text() == "Read from the start"
    refresh.click()
    page.wait_for_url(re.compile(rf"/history/session/{SESSION}\?as=alice$"))
    assert record.timeline_calls()[-1] == {"session_id": SESSION, "after": None}
    _assert_links_keep(page, "alice")


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


def test_hostile_content_is_shown_as_text_and_runs_nothing(
    page: Page, watch: _Watch, history_url: str
) -> None:
    page.goto(f"{history_url}/history/session/{SESSION}?as=alice")
    prompt = page.locator("li.entry.prompt pre.body").first
    assert prompt.inner_text() == HOSTILE
    page.goto(f"{history_url}/history/?as=alice")
    assert HOSTILE[:20] in page.locator("ol.sessions li.session .prompt").first.inner_text()
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
    assert page.locator("ol.sessions li.session", has_text="bob").count() == 0

    page.locator("nav.switcher a.choice", has_text="ops").click()
    page.wait_for_url(re.compile(r"[?&]as=ops(&|$)"))
    assert page.locator(".viewing-as .person").inner_text() == "ops"
    assert page.locator(".viewing-as .scope").inner_text() == "all"
    assert page.locator("nav.switcher .choice.current").inner_text() == "ops"
    _assert_links_keep(page, "ops")

    page.locator("ol.sessions li.session a.row", has_text="bob's question").click()
    page.wait_for_url(re.compile(rf"/history/session/{BOB_SESSION}\?"))
    assert _selection(page.url) == ["ops"]
    assert page.locator(".viewing-as .person").inner_text() == "ops"
    _assert_links_keep(page, "ops")

    page.locator("header h1 a").click()
    page.wait_for_url(re.compile(r"/history/\?as=ops$"))
    assert page.locator(".viewing-as .scope").inner_text() == "all"


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
        "/history/static/history.js",
    } <= loaded
    assert [entry for entry in assets if entry[0] != 200] == []
