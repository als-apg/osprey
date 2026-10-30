"""The history page in process: its gate, its record reads, and what it renders.

The record is a :class:`~tests.agent_record.fakes.FakeRecordClient` that
records every read and answers canned :class:`ClientAnswer` values, so each
test can say both what the page asked the record and what it showed. The gate
tests build the app from the environment the compose file renders and run
without the suite's auth seam, which would otherwise present the app's own
secret on every request.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from osprey.agent_record.client import ClientAnswer, RecordClient
from osprey.history import app as history_app
from osprey.history import viewer
from osprey.history.app import PREVIEW_CHARS, SESSION_WINDOW_DAYS, create_app
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import reset_web_credentials
from tests.agent_record.fakes import FakeRecordClient

SESSION = "3f8a1c02-0000-4000-8000-abcdefabcdef"
ENTRY = "alice"


def _headers(
    *,
    person: str = "alice",
    scope: str = "own",
    view: str = "own:alice",
    entry: str = ENTRY,
    choices: str = "alice=alice,reader:carol=carol",
) -> dict[str, str]:
    return {
        viewer.PERSON_HEADER: person,
        viewer.SCOPE_HEADER: scope,
        viewer.VIEW_HEADER: view,
        viewer.ENTRY_HEADER: entry,
        viewer.CHOICES_HEADER: choices,
    }


class Record:
    """One fake client per record viewer the page asks for, all sharing answers."""

    def __init__(self) -> None:
        self.answers: dict[str, Any] = {}
        self.unavailable: str | None = None
        self.clients: list[FakeRecordClient] = []

    def client_for(self, record_viewer: str) -> FakeRecordClient:
        client = FakeRecordClient(
            self.answers, unavailable=self.unavailable, login="history", viewer=record_viewer
        )
        self.clients.append(client)
        return client

    @property
    def calls(self) -> list[tuple[str, str, dict[str, Any]]]:
        return [
            (c.viewer or "", call.name, dict(call.kwargs)) for c in self.clients for call in c.calls
        ]

    def names(self) -> list[str]:
        return [name for _, name, _ in self.calls]


@pytest.fixture
def record() -> Record:
    return Record()


@pytest.fixture
def client(record: Record) -> TestClient:
    return TestClient(create_app(env={}, max_days=30, client_for=record.client_for))


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.hrefs.append(values["href"] or "")
        if tag == "link" and values.get("href"):
            self.assets.append(values["href"] or "")
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"] or "")


def _links(html: str) -> _Links:
    parser = _Links()
    parser.feed(html)
    return parser


def _assert_every_link_keeps_as(html: str) -> list[str]:
    links = _links(html)
    assert links.hrefs, "the page rendered no links"
    for href in links.hrefs:
        parts = urlsplit(href)
        assert parts.path.startswith(viewer.ROOT_PATH + "/"), href
        selected = parse_qs(parts.query).get(viewer.SELECTOR_PARAM)
        assert selected and len(selected) == 1, f"{href} drops the selection"
    return links.hrefs


def _ok(data: Any, **fields: Any) -> ClientAnswer:
    return ClientAnswer(data=data, complete=fields.pop("complete", True), **fields)


# ---------------------------------------------------------------------------
# The gate: nginx's secret passes, its absence does not
# ---------------------------------------------------------------------------


@pytest.fixture
def gated_app(monkeypatch: pytest.MonkeyPatch, record: Record) -> Callable[[], TestClient]:
    """The app built from exactly the environment its compose file renders."""
    for name in ("OSPREY_TERMINAL_SECRET", "OSPREY_TERMINAL_BIND_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS", "1")
    monkeypatch.setenv("OSPREY_TERMINAL_ROSTER_OWNERS", "HISTORY=history")
    monkeypatch.setenv("OSPREY_TERMINAL_SECRET_HISTORY", "the-gate-secret")
    reset_web_credentials()

    def build() -> TestClient:
        return TestClient(create_app(client_for=record.client_for))

    yield build
    reset_web_credentials()


@pytest.mark.no_auth_seam
def test_a_request_carrying_nginxs_gate_secret_passes(gated_app: Callable[[], TestClient]) -> None:
    response = gated_app().get(
        "/", headers={OPERATOR_SECRET_HEADER: "the-gate-secret", **_headers()}
    )
    assert response.status_code == 200, response.text


@pytest.mark.no_auth_seam
def test_a_request_without_the_gate_secret_is_refused(
    gated_app: Callable[[], TestClient], record: Record
) -> None:
    client = gated_app()
    assert client.get("/", headers=_headers()).status_code == 401
    assert (
        client.get("/", headers={OPERATOR_SECRET_HEADER: "a-guess", **_headers()}).status_code
        == 401
    )
    assert record.calls == []


def test_a_request_without_the_sidecars_answer_has_no_scope(
    client: TestClient, record: Record
) -> None:
    response = client.get("/")
    assert response.status_code == 403
    assert record.calls == []


@pytest.mark.parametrize(
    ("scope", "view"),
    [
        ("own", "view_alice"),
        ("own", "own:"),
        ("own", "own:alice` x"),
        ("all", "own:alice"),
        ("own", "all"),
        ("reader", "reader:"),
        ("own", "someone:alice"),
    ],
)
def test_a_viewer_the_record_does_not_read_is_refused(
    client: TestClient, record: Record, scope: str, view: str
) -> None:
    response = client.get("/", headers=_headers(scope=scope, view=view))
    assert response.status_code == 403
    assert record.calls == []


def test_the_header_names_are_the_sidecars() -> None:
    from osprey.services.auth_sidecar.routes import history as sidecar

    for name in ("PERSON", "SCOPE", "VIEW", "ROLE", "ENTRY", "CHOICES"):
        assert getattr(viewer, f"{name}_HEADER") == getattr(sidecar, f"{name}_HEADER")
    assert viewer.SELECTOR_PARAM == sidecar.SELECTOR_PARAM
    assert not hasattr(viewer, "TABLES_HEADER")
    assert not hasattr(sidecar, "TABLES_HEADER")


def test_the_viewer_spellings_are_the_records() -> None:
    from osprey.agent_record import service
    from osprey.services.auth_sidecar.routes import history as sidecar

    assert viewer.VIEWER_ALL == sidecar.VIEW_ALL == service.VIEWER_ALL
    assert viewer.VIEWER_OWN_PREFIX == sidecar.VIEW_OWN_PREFIX == service.VIEWER_OWN_PREFIX
    assert viewer.VIEWER_READER_PREFIX == sidecar.VIEW_READER_PREFIX == service.VIEWER_READER_PREFIX


@pytest.mark.parametrize(
    ("scope", "view", "person"),
    [("own", "own:alice", "alice"), ("all", "all", "alice"), ("reader", "reader:carol", "carol")],
)
def test_every_read_names_the_sidecars_viewer(
    client: TestClient, record: Record, scope: str, view: str, person: str
) -> None:
    client.get("/", headers=_headers(scope=scope, view=view, person=person))
    assert record.calls
    assert {who for who, _, _ in record.calls} == {view}


def test_the_default_client_reads_with_the_history_login() -> None:
    app = create_app(
        env={
            "OSPREY_AGENT_RECORD_URL": "http://agent-record:8000",
            "OSPREY_AGENT_RECORD_LOGIN": "history:the-token",
        }
    )
    built = app.state.client_for("own:alice")
    assert isinstance(built, RecordClient)
    assert (built.url, built.login, built.viewer) == (
        "http://agent-record:8000",
        "history",
        "own:alice",
    )


def test_an_unconfigured_record_renders_record_unavailable() -> None:
    client = TestClient(create_app(env={}))
    response = client.get("/", headers=_headers())
    assert response.status_code == 503
    assert "record unavailable" in response.text
    assert "OSPREY_AGENT_RECORD_URL" in response.text


# ---------------------------------------------------------------------------
# The session list
# ---------------------------------------------------------------------------


def _session_rows() -> list[dict[str, Any]]:
    return [
        {
            "session_id": "s-web",
            "persons": ["alice"],
            "identities": ["alice"],
            "first_ts": "2026-10-24T10:40:00+00:00",
            "last_ts": "2026-10-24T10:42:00+00:00",
            "entries": 12,
            "prompt": "why did BPM 3 drift overnight?",
            "run_ids": [],
            "terminal_session": "term-1",
        },
        {
            "session_id": "s-dispatch",
            "persons": ["bob"],
            "identities": ["dispatch-worker-1"],
            "first_ts": "2026-10-24T10:15:00+00:00",
            "last_ts": "2026-10-24T10:15:10+00:00",
            "entries": 6,
            "prompt": None,
            "run_ids": ["run-1"],
            "terminal_session": "",
        },
    ]


def _users() -> list[dict[str, Any]]:
    return [
        {"person": "alice", "sessions": 1, "last_ts": "2026-10-24T10:42:00+00:00"},
        {"person": "bob", "sessions": 1, "last_ts": "2026-10-24T10:15:10+00:00"},
    ]


def test_the_session_list_shows_every_session_newest_first(
    client: TestClient, record: Record
) -> None:
    record.answers = {"list_sessions": _ok(_session_rows()), "session_users": _ok(_users())}
    response = client.get("/", headers=_headers(scope="all", view="all"))
    assert response.status_code == 200, response.text
    text = response.text
    assert "why did BPM 3 drift overnight?" in text
    assert "(no prompt) s-dispatch" in text
    assert "dispatch-worker-1" in text
    assert text.index("s-web") < text.index("s-dispatch")
    assert record.names() == ["list_sessions", "session_users"]


def test_the_session_list_reads_the_window_and_older_widens_it(
    client: TestClient, record: Record, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.datetime(2026, 10, 24, 12, 0, tzinfo=datetime.UTC)
    monkeypatch.setattr(history_app, "_now", lambda: now)
    client.get("/", headers=_headers())
    client.get("/", params={"older": 2}, headers=_headers())
    client.get("/", params={"older": 999}, headers=_headers())
    sinces = [kwargs["since"] for _, name, kwargs in record.calls if name == "list_sessions"]
    assert sinces == [
        now - datetime.timedelta(days=SESSION_WINDOW_DAYS),
        now - datetime.timedelta(days=SESSION_WINDOW_DAYS + 2),
        now - datetime.timedelta(days=30),
    ]


def test_older_stops_at_the_retention(client: TestClient) -> None:
    response = client.get("/", params={"older": 999}, headers=_headers())
    assert 'class="older"' not in response.text
    assert 'class="newer"' in response.text


def test_the_person_filter_keeps_that_persons_sessions(client: TestClient, record: Record) -> None:
    record.answers = {"list_sessions": _ok(_session_rows()), "session_users": _ok(_users())}
    response = client.get("/", params={"person": "bob"}, headers=_headers(scope="all", view="all"))
    assert "s-dispatch" in response.text
    assert "why did BPM 3 drift overnight?" not in response.text


def test_an_unavailable_record_renders_record_unavailable(
    client: TestClient, record: Record
) -> None:
    record.unavailable = "agent-record is unreachable: ConnectError"
    response = client.get("/", headers=_headers())
    assert response.status_code == 503
    assert "record unavailable" in response.text
    assert "agent-record is unreachable" in response.text
    assert "No sessions" not in response.text
    _assert_every_link_keeps_as(response.text)


def test_an_incomplete_list_says_so(client: TestClient, record: Record) -> None:
    record.answers = {
        "list_sessions": _ok(_session_rows(), complete=False),
        "session_users": _ok(_users()),
    }
    response = client.get("/", headers=_headers())
    assert response.status_code == 200
    assert "incomplete" in response.text


def test_an_empty_list_that_is_incomplete_is_never_shown_as_complete(
    client: TestClient, record: Record
) -> None:
    record.answers = {"list_sessions": _ok([], complete=False), "session_users": _ok([])}
    response = client.get("/", headers=_headers())
    assert "incomplete" in response.text


def test_every_link_on_the_session_list_keeps_as(client: TestClient, record: Record) -> None:
    record.answers = {"list_sessions": _ok(_session_rows()), "session_users": _ok(_users())}
    response = client.get("/", params={"older": 1}, headers=_headers(entry="reader:carol"))
    hrefs = _assert_every_link_keeps_as(response.text)
    selections = {parse_qs(urlsplit(h).query)["as"][0] for h in hrefs}
    assert selections == {"reader:carol", "alice"}, "the switcher selects, every other link keeps"
    assert any("older=2" in h for h in hrefs)
    assert any(h.startswith("/history/session/s-web?") for h in hrefs)
    assert any("person=bob" in h for h in hrefs)


# ---------------------------------------------------------------------------
# One session's timeline
# ---------------------------------------------------------------------------


def _node(kind: str, ts: str, body: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
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


def _timeline() -> list[dict[str, Any]]:
    return [
        _node("prompt", "2026-10-24T10:15:00+00:00", {"text": "<script>alert(1)</script>"}),
        _node(
            "tool_call",
            "2026-10-24T10:15:04+00:00",
            {"name": "read_channel", "input": {"channel": "SR:BPM3:X"}},
            decision=_node("decision", "2026-10-24T10:15:04+00:00", {"decision": "allowed"}),
            result=_node("tool_result", "2026-10-24T10:15:04+00:00", {"text": "0.41 mm"}),
        ),
        _node(
            "tool_call",
            "2026-10-24T10:15:05+00:00",
            {"name": "Agent"},
            subagent={
                "agent_id": "a-1",
                "timeline": [
                    _node("tool_call", "2026-10-24T10:15:06+00:00", {"name": "nested_read"})
                ],
            },
        ),
        _node("answer", "2026-10-24T10:15:10+00:00", {"text": "x" * (PREVIEW_CHARS + 5)}),
    ]


def test_a_session_renders_its_timeline(client: TestClient, record: Record) -> None:
    record.answers = {"timeline": _ok(_timeline())}
    response = client.get(f"/session/{SESSION}", headers=_headers())
    assert response.status_code == 200, response.text
    text = response.text
    assert "read_channel" in text
    assert '<span class="decision">allowed</span>' in text
    assert "0.41 mm" in text
    assert re.search(r'<ol class="entries subagent">\s*<li class="entry tool_call">', text)
    assert "nested_read" in text
    assert record.calls == [("own:alice", "timeline", {"session_id": SESSION, "after": None})]


def test_content_is_escaped(client: TestClient, record: Record) -> None:
    record.answers = {"timeline": _ok(_timeline())}
    response = client.get(f"/session/{SESSION}", headers=_headers())
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_a_long_body_links_to_its_full_entry(client: TestClient, record: Record) -> None:
    record.answers = {"timeline": _ok(_timeline())}
    response = client.get(f"/session/{SESSION}", headers=_headers())
    full = [h for h in _links(response.text).hrefs if "/history/full/entry?" in h]
    assert len(full) == 1
    query = parse_qs(urlsplit(full[0]).query)
    assert query["key"] == ["k-answer-2026-10-24T10:15:10+00:00"]
    assert query["session"] == [SESSION]


def test_the_next_page_is_read_through_the_cursor(client: TestClient, record: Record) -> None:
    record.answers = {"timeline": _ok(_timeline(), next="CURSOR-1", more=True)}
    response = client.get(f"/session/{SESSION}", headers=_headers())
    hrefs = _assert_every_link_keeps_as(response.text)
    (nxt,) = [h for h in hrefs if "after=" in h]
    assert parse_qs(urlsplit(nxt).query)["after"] == ["CURSOR-1"]
    client.get(f"/session/{SESSION}", params={"after": "CURSOR-1"}, headers=_headers())
    assert record.calls[-1][2] == {"session_id": SESSION, "after": "CURSOR-1"}


def test_newer_entries_offer_a_reread_from_the_start(client: TestClient, record: Record) -> None:
    record.answers = {"timeline": _ok(_timeline(), newer=True)}
    response = client.get(f"/session/{SESSION}", params={"after": "C"}, headers=_headers())
    assert "Read from the start" in response.text


def test_an_incomplete_session_says_so(client: TestClient, record: Record) -> None:
    record.answers = {"timeline": _ok(_timeline(), complete=False)}
    response = client.get(f"/session/{SESSION}", headers=_headers())
    assert "incomplete" in response.text


def test_a_session_the_viewer_sees_nothing_of_is_404(client: TestClient, record: Record) -> None:
    record.answers = {"timeline": _ok([])}
    response = client.get(f"/session/{SESSION}", headers=_headers())
    assert response.status_code == 404
    assert f"no session {SESSION} in what you may see" in response.text


def test_an_unavailable_timeline_is_never_a_404(client: TestClient, record: Record) -> None:
    record.unavailable = "agent-record did not answer within 25 s"
    response = client.get(f"/session/{SESSION}", headers=_headers())
    assert response.status_code == 503
    assert "record unavailable" in response.text


def test_a_session_id_that_is_not_one_is_404_without_a_read(
    client: TestClient, record: Record
) -> None:
    response = client.get("/session/x%20y", headers=_headers())
    assert response.status_code == 404
    assert record.calls == []


# ---------------------------------------------------------------------------
# The full entry
# ---------------------------------------------------------------------------


def test_the_full_entry_shows_the_whole_body(client: TestClient, record: Record) -> None:
    body = {"text": "y" * (PREVIEW_CHARS * 2)}
    record.answers = {"full_entry": _ok(_node("answer", "2026-10-24T10:15:10+00:00", body))}
    response = client.get(
        "/full/entry", params={"key": "k-1", "session": SESSION}, headers=_headers()
    )
    assert response.status_code == 200
    assert "y" * (PREVIEW_CHARS * 2) in response.text
    assert record.calls == [("own:alice", "full_entry", {"key": "k-1"})]
    hrefs = _assert_every_link_keeps_as(response.text)
    assert any(h.startswith(f"/history/session/{SESSION}?") for h in hrefs)


def test_a_full_entry_the_viewer_cannot_see_is_404(client: TestClient, record: Record) -> None:
    record.answers = {"full_entry": _ok(None)}
    response = client.get("/full/entry", params={"key": "k-1"}, headers=_headers())
    assert response.status_code == 404
    _assert_every_link_keeps_as(response.text)


def test_an_unavailable_full_entry_renders_record_unavailable(
    client: TestClient, record: Record
) -> None:
    record.unavailable = "entry 'k-1' is incomplete: a chunk is missing"
    response = client.get("/full/entry", params={"key": "k-1"}, headers=_headers())
    assert response.status_code == 503
    assert "record unavailable" in response.text


# ---------------------------------------------------------------------------
# Assets and the page shell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [f"/session/{SESSION}", "/full/entry?key=k"])
def test_every_link_on_a_not_found_page_keeps_as(
    client: TestClient, record: Record, path: str
) -> None:
    record.answers = {"timeline": _ok([]), "full_entry": _ok(None)}
    _assert_every_link_keeps_as(client.get(path, headers=_headers()).text)


def test_assets_are_served_under_the_prefix_the_page_writes(client: TestClient) -> None:
    """nginx strips ``/history``; every asset URL the page writes resolves once it is stripped."""
    response = client.get("/", headers=_headers())
    assets = _links(response.text).assets
    assert assets
    for url in assets:
        assert url.startswith(viewer.ROOT_PATH + "/"), url
        stripped = url[len(viewer.ROOT_PATH) :]
        assert client.get(stripped).status_code == 200, stripped


def test_health_needs_no_credential(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_no_rendered_page_carries_an_inline_script(client: TestClient, record: Record) -> None:
    record.answers = {"timeline": _ok(_timeline())}
    html = client.get(f"/session/{SESSION}", headers=_headers()).text
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html)
