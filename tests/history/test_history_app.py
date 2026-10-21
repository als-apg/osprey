"""The history page in process: its gate, its store calls, and what it renders.

The store is an :class:`httpx.MockTransport` that records every call and
answers from canned rows, so each test can say both what the page asked and
what it showed. The gate tests build the app from the environment the compose
file renders and run without the suite's auth seam, which would otherwise
present the app's own secret on every request.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.deployment.telemetry_config import HISTORY_CONCURRENCY
from osprey.history import queries, viewer
from osprey.history.app import create_app
from osprey.history.store import HistoryStore, StoreRefusal
from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import reset_web_credentials

SESSION = "3f8a1c02-0000-4000-8000-abcdefabcdef"
ENTRY = "alice"

ALL_TABLES = "otel_logs,otel_traces,audit_ledger"


def _headers(
    *,
    person: str = "alice",
    scope: str = "own",
    view: str = "view_alice",
    tables: str = ALL_TABLES,
    entry: str = ENTRY,
    choices: str = "alice=alice,reader:carol=carol",
) -> dict[str, str]:
    return {
        viewer.PERSON_HEADER: person,
        viewer.SCOPE_HEADER: scope,
        viewer.VIEW_HEADER: view,
        viewer.TABLES_HEADER: tables,
        viewer.ENTRY_HEADER: entry,
        viewer.CHOICES_HEADER: choices,
    }


class Store:
    """A canned store: rows per query name, every call recorded."""

    def __init__(self, rows: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.rows = rows or {}
        self.calls: list[httpx.Request] = []
        self.fail: tuple[int, str, str] | None = None

    def name_of(self, request: httpx.Request) -> str:
        body = request.content.decode()
        for name in queries.PAGE_QUERIES:
            if queries.load(name).strip().rstrip(";") in body:
                return name
        raise AssertionError(f"not a contract query: {body[:200]}")

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.fail is not None:
            status, code, text = self.fail
            return httpx.Response(status, headers={"X-ClickHouse-Exception-Code": code}, text=text)
        lines = [json.dumps(row) for row in self.rows.get(self.name_of(request), [])]
        return httpx.Response(200, text="\n".join(lines))

    def names(self) -> list[str]:
        return [self.name_of(call) for call in self.calls]


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def client(store: Store) -> TestClient:
    app = create_app(
        store_url="http://store.test:8123",
        store_password="history-pw",
        queueservers=("queueserver",),
        max_days=30,
        transport=httpx.MockTransport(store.handler),
    )
    return TestClient(app)


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


# ---------------------------------------------------------------------------
# The gate: nginx's secret passes, its absence does not
# ---------------------------------------------------------------------------


@pytest.fixture
def gated_app(monkeypatch: pytest.MonkeyPatch, store: Store) -> Callable[[], TestClient]:
    """The app built from exactly the environment its compose file renders."""
    for name in ("OSPREY_TERMINAL_SECRET", "OSPREY_TERMINAL_BIND_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS", "1")
    monkeypatch.setenv("OSPREY_TERMINAL_ROSTER_OWNERS", "HISTORY=history")
    monkeypatch.setenv("OSPREY_TERMINAL_SECRET_HISTORY", "the-gate-secret")
    reset_web_credentials()

    def build() -> TestClient:
        return TestClient(create_app(transport=httpx.MockTransport(store.handler)))

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
    gated_app: Callable[[], TestClient], store: Store
) -> None:
    client = gated_app()
    assert client.get("/", headers=_headers()).status_code == 401
    assert (
        client.get("/", headers={OPERATOR_SECRET_HEADER: "a-guess", **_headers()}).status_code
        == 401
    )
    assert store.calls == []


def test_a_request_without_the_sidecars_answer_has_no_scope(
    client: TestClient, store: Store
) -> None:
    response = client.get("/")
    assert response.status_code == 403
    assert store.calls == []


def test_a_view_user_that_is_not_a_store_name_is_refused(client: TestClient, store: Store) -> None:
    response = client.get("/", headers=_headers(view="view_alice` SELECT 1 --"))
    assert response.status_code == 403
    assert store.calls == []


def test_the_header_names_are_the_sidecars() -> None:
    from osprey.services.auth_sidecar.routes import history as sidecar

    for name in ("PERSON", "SCOPE", "VIEW", "ROLE", "TABLES", "ENTRY", "CHOICES"):
        assert getattr(viewer, f"{name}_HEADER") == getattr(sidecar, f"{name}_HEADER")
    assert viewer.SELECTOR_PARAM == sidecar.SELECTOR_PARAM


# ---------------------------------------------------------------------------
# The query layer
# ---------------------------------------------------------------------------


def _session_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "session_users": [
            {
                "User": "alice",
                "terminal_session": "term-1",
                "first_ts": "2026-10-11 10:00:00.000000000",
                "last_ts": "2026-10-11 10:30:00.000000000",
            }
        ],
        "conversation": [
            {
                "User": "alice",
                "SessionId": SESSION,
                "TerminalSession": "term-1",
                "Timestamp": "2026-10-11 10:00:00.000000000",
                "Event": "user_prompt",
                "body_hash": "123",
                "captured": 1,
                "content": "<script>alert(1)</script>",
                "content_length": queries.PREVIEW_CHARS + 10,
            }
        ],
        "tool_log": [
            {
                "User": "alice",
                "SessionId": SESSION,
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
        ],
        "preflight": [
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
        ],
    }


def test_every_store_call_is_a_post_that_waits_for_the_end_of_the_query(
    client: TestClient, store: Store
) -> None:
    store.rows = _session_rows()
    response = client.get(f"/session/{SESSION}", headers=_headers())
    assert response.status_code == 200, response.text

    assert set(store.names()) >= {"session_users", "conversation", "tool_log", "preflight"}
    for call in store.calls:
        assert call.method == "POST"
        params = parse_qs(call.url.query.decode())
        assert params["http_wait_end_of_query"] == ["1"]
        assert params["log_comment"] == ["alice:view_alice"]
        assert params["quota_key"] == ["alice"]
        assert call.headers["X-ClickHouse-User"] == "history"
        assert call.headers["X-ClickHouse-Key"] == "history-pw"
        body = call.content.decode()
        name = store.name_of(call)
        assert body == (f"EXECUTE AS `view_alice` {queries.load(name).strip()}\nFORMAT JSONEachRow")


def test_values_travel_as_typed_parameters_never_in_the_query(
    client: TestClient, store: Store
) -> None:
    store.rows = _session_rows()
    hostile = "x'); DROP TABLE t; --"
    client.get(f"/session/{SESSION}", params={"user": hostile}, headers=_headers())
    (call,) = store.calls  # no user matches: nothing past the lookup
    assert hostile not in call.content.decode()
    assert parse_qs(call.url.query.decode())["param_session"] == [SESSION]


def test_the_semaphore_bounds_concurrent_store_calls() -> None:
    running = 0
    peak = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return httpx.Response(200, text="")

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            store = HistoryStore("http://store.test:8123", "pw", client=client)
            who = viewer.viewer_from_headers(_headers())
            await asyncio.gather(
                *(store.rows(who, "session_users", {"session": "s"}) for _ in range(8))
            )

    asyncio.run(main())
    assert peak == HISTORY_CONCURRENCY


def test_a_store_error_is_shown_with_its_name(client: TestClient, store: Store) -> None:
    store.fail = (
        500,
        "396",
        "Code: 396. DB::Exception: Limit for result exceeded. (TOO_MANY_ROWS_OR_BYTES) "
        "(version 26.8.18.2)",
    )
    response = client.get("/", headers=_headers())
    assert response.status_code == 502
    assert "TOO_MANY_ROWS_OR_BYTES (396)" in response.text


def test_an_unreachable_store_is_a_refusal_not_a_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = TestClient(create_app(transport=httpx.MockTransport(handler)))
    response = client.get("/", headers=_headers())
    assert response.status_code == 502
    assert "STORE_UNREACHABLE" in response.text


def test_a_store_refusal_names_the_error_from_the_body_when_no_header_does() -> None:
    from osprey.history.store import _refusal

    refusal = _refusal(
        httpx.Response(
            403,
            text="Code: 497. DB::Exception: alice: Not enough privileges. "
            "(ACCESS_DENIED) (version 26.8.18.2)",
        )
    )
    assert isinstance(refusal, StoreRefusal)
    assert (refusal.name, refusal.code) == ("ACCESS_DENIED", "497")


# ---------------------------------------------------------------------------
# Table-aware views
# ---------------------------------------------------------------------------


def test_a_view_whose_table_is_not_readable_is_never_queried(
    client: TestClient, store: Store
) -> None:
    response = client.get("/", headers=_headers(tables="audit_ledger"))
    assert response.status_code == 200
    assert "not in your tables" in response.text
    assert store.calls == []


def test_a_transcript_skips_the_ledger_when_it_is_not_readable(
    client: TestClient, store: Store
) -> None:
    store.rows = _session_rows()
    response = client.get(f"/session/{SESSION}", headers=_headers(tables="otel_logs"))
    assert response.status_code == 200
    assert set(store.names()) == {"session_users", "conversation"}
    assert "<code>audit_ledger</code> is not in your tables" in response.text


# ---------------------------------------------------------------------------
# Renderers, deep link and the preflight band
# ---------------------------------------------------------------------------


def test_content_is_escaped(client: TestClient, store: Store) -> None:
    store.rows = _session_rows()
    response = client.get(f"/session/{SESSION}", headers=_headers())
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_a_deep_link_with_no_match_is_404(client: TestClient, store: Store) -> None:
    response = client.get(f"/session/{SESSION}", headers=_headers())
    assert response.status_code == 404
    assert f"no session {SESSION} in what you may see" in response.text
    assert store.names() == ["session_users"]


def test_a_deep_link_shows_one_transcript_per_user(client: TestClient, store: Store) -> None:
    rows = _session_rows()
    rows["session_users"].append({**rows["session_users"][0], "User": "bob"})
    store.rows = rows
    response = client.get(f"/session/{SESSION}", headers=_headers(scope="all", view="view__all"))
    assert response.text.count('<section class="transcript">') == 2
    users = [
        parse_qs(call.url.query.decode()).get("param_user")
        for call in store.calls
        if store.name_of(call) == "conversation"
    ]
    assert users == [["alice"], ["bob"]]


def test_an_own_view_shows_only_preflight_rows_matched_to_a_submit(
    client: TestClient, store: Store
) -> None:
    store.rows = _session_rows()
    response = client.get(f"/session/{SESSION}", headers=_headers())
    assert "matched-preflight" in response.text
    assert "forged-preflight" not in response.text


def test_an_all_view_shows_every_preflight_row_and_labels_the_unmatched(
    client: TestClient, store: Store
) -> None:
    store.rows = _session_rows()
    response = client.get(f"/session/{SESSION}", headers=_headers(scope="all", view="view__all"))
    assert "matched-preflight" in response.text
    assert "forged-preflight" in response.text
    assert response.text.count("no matching queue submit within 60 s") == 1


def test_the_preflight_query_names_the_rendered_queueservers(
    client: TestClient, store: Store
) -> None:
    store.rows = _session_rows()
    client.get(f"/session/{SESSION}", headers=_headers())
    (call,) = [c for c in store.calls if store.name_of(c) == "preflight"]
    assert parse_qs(call.url.query.decode())["param_queueservers"] == ["['queueserver']"]


# ---------------------------------------------------------------------------
# The selection survives navigation
# ---------------------------------------------------------------------------


def _session_list_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "User": "alice",
            "SessionId": f"s-{i:04d}",
            "terminal_session": f"term-{i // 2}",
            "first_ts": f"2026-10-11 09:{i % 60:02d}:00.000000000",
            "last_ts": f"2026-10-11 10:{i % 60:02d}:00.000000000",
            "prompts": 1,
        }
        for i in range(count)
    ]


def test_every_link_on_the_session_list_keeps_as(client: TestClient, store: Store) -> None:
    store.rows = {"session_list": _session_list_rows(queries.SESSION_LIST_LIMIT)}
    response = client.get("/", params={"older": 1}, headers=_headers(entry="reader:carol"))
    hrefs = _assert_every_link_keeps_as(response.text)

    selections = {parse_qs(urlsplit(h).query)["as"][0] for h in hrefs}
    assert selections == {"reader:carol", "alice"}, "the switcher selects, every other link keeps"
    assert any("cursor_ts=" in h for h in hrefs), "the paging cursor is a link"
    assert any("older=2" in h for h in hrefs)
    assert any(h.startswith("/history/session/") for h in hrefs)


def test_every_link_on_a_transcript_keeps_as(client: TestClient, store: Store) -> None:
    rows = _session_rows()
    rows["conversation"] = rows["conversation"] * queries.PAGE_LIMIT
    store.rows = rows
    response = client.get(f"/session/{SESSION}", headers=_headers())
    hrefs = _assert_every_link_keeps_as(response.text)
    assert any("/history/full/log?" in h for h in hrefs)
    assert any("/history/full/ledger?" in h for h in hrefs)
    assert any("after_hash=" in h for h in hrefs), "the next-page cursor is a link"


@pytest.mark.parametrize(
    "path", [f"/session/{SESSION}", "/full/log?user=a&session=s&ts=t&body_hash=1"]
)
def test_every_link_on_a_not_found_page_keeps_as(client: TestClient, path: str) -> None:
    _assert_every_link_keeps_as(client.get(path, headers=_headers()).text)


def test_every_link_on_a_refusal_keeps_as(client: TestClient, store: Store) -> None:
    store.fail = (500, "159", "Code: 159. Timeout exceeded (TIMEOUT_EXCEEDED)")
    _assert_every_link_keeps_as(client.get("/", headers=_headers()).text)


def test_the_list_groups_a_resumed_session_under_its_first_terminal(
    client: TestClient, store: Store
) -> None:
    store.rows = {"session_list": _session_list_rows(2)}
    response = client.get("/", headers=_headers())
    assert response.text.count('<tr class="transcript">') == 1
    assert "continued as" in response.text


def test_assets_are_served_under_the_prefix_the_page_writes(client: TestClient) -> None:
    """nginx strips ``/history``; every asset URL the page writes resolves once it is stripped."""
    response = client.get("/", headers=_headers(tables=""))
    assets = _links(response.text).assets
    assert assets
    for url in assets:
        assert url.startswith(viewer.ROOT_PATH + "/"), url
        stripped = url[len(viewer.ROOT_PATH) :]
        assert client.get(stripped).status_code == 200, stripped


def test_health_needs_no_credential(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_no_rendered_page_carries_an_inline_script(client: TestClient, store: Store) -> None:
    store.rows = _session_rows()
    html = client.get(f"/session/{SESSION}", headers=_headers()).text
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html)
