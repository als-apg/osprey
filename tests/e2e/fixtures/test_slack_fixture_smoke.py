"""Smoke proofs for the Slack bridge's hermetic fixtures.

These tests assert nothing about the bridge's behaviour — they assert that the
fixtures its e2e lane stands on actually work: each fake binds a free loopback
port, answers the calls the adapter makes, and records what it was sent. The
fakes are proved **independently** with plain :mod:`httpx` and
:mod:`websockets` clients, so a failure in the lane points at the bridge rather
than at the scaffolding under it.

**Nothing here may skip.** No container runtime, no credential, no network and
nothing from ``slack_sdk``: "cannot boot" is a failure.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from .slack_fakes import (
    APP_TOKEN,
    BOT_ID,
    BOT_TOKEN,
    BOT_USER_ID,
    CHANNEL_ID,
    DM_CHANNEL_ID,
    FOREIGN_TEAM_ID,
    MPIM_CHANNEL_ID,
    SENDER_ID,
    TEAM_ID,
    FakeSlackWeb,
    FakeSocketServer,
    bot_message,
    channel_mention,
    dm_message,
    envelope,
    file_share,
    mpim_message,
    next_ts,
    shared_channel_mention,
    thread_reply,
)

pytestmark = [pytest.mark.e2e]

TIMEOUT = 10.0
API = "https://slack.com/api"


@pytest.fixture
def socket_server() -> Iterator[FakeSocketServer]:
    with FakeSocketServer() as server:
        yield server


@pytest.fixture
def web(socket_server: FakeSocketServer) -> Iterator[FakeSlackWeb]:
    with FakeSlackWeb(socket=socket_server) as server:
        yield server


@pytest.fixture
def client(web: FakeSlackWeb) -> Iterator[httpx.Client]:
    with web.http_client() as c:
        yield c


def _bot(token: str = BOT_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _call(client: httpx.Client, method: str, token: str = BOT_TOKEN, **args) -> httpx.Response:
    return client.post(f"{API}/{method}", json=args, headers=_bot(token))


# ---------------------------------------------------------------------------
# FakeSlackWeb
# ---------------------------------------------------------------------------


class TestFakeSlackWeb:
    def test_boots_on_a_free_loopback_port(self, web: FakeSlackWeb) -> None:
        assert web.base_url.startswith("http://127.0.0.1:")
        with FakeSlackWeb() as other:
            assert other.port != web.port

    def test_auth_test_reports_the_bot_identity(self, client: httpx.Client) -> None:
        body = _call(client, "auth.test").json()
        assert body["ok"] is True
        assert (body["user_id"], body["bot_id"], body["team_id"]) == (BOT_USER_ID, BOT_ID, TEAM_ID)

    def test_tokens_are_not_interchangeable(self, client: httpx.Client) -> None:
        assert _call(client, "auth.test", token=APP_TOKEN).json()["error"] == "invalid_auth"
        assert _call(client, "apps.connections.open", token=BOT_TOKEN).json()["error"] == (
            "invalid_auth"
        )
        no_auth = client.post(f"{API}/auth.test", json={})
        assert no_auth.json() == {"ok": False, "error": "not_authed"}

    def test_post_message_is_recorded_and_visible_in_the_thread(
        self, web: FakeSlackWeb, client: httpx.Client
    ) -> None:
        parent = channel_mention("hi")["event"]
        web.add_message(CHANNEL_ID, parent)
        posted = _call(
            client, "chat.postMessage", channel=CHANNEL_ID, text="answer", thread_ts=parent["ts"]
        ).json()
        assert posted["ok"] is True and posted["ts"]
        [record] = web.wait_for_posted(1, timeout=TIMEOUT)
        assert (record.channel, record.text, record.thread_ts) == (
            CHANNEL_ID,
            "answer",
            parent["ts"],
        )
        replies = client.get(
            f"{API}/conversations.replies",
            params={"channel": CHANNEL_ID, "ts": parent["ts"]},
            headers=_bot(),
        ).json()
        assert [m["ts"] for m in replies["messages"]] == [parent["ts"], posted["ts"]]
        assert replies["messages"][1]["bot_id"] == BOT_ID

    def test_form_encoded_arguments_are_accepted(
        self, web: FakeSlackWeb, client: httpx.Client
    ) -> None:
        resp = client.post(
            f"{API}/chat.postMessage", data={"channel": DM_CHANNEL_ID, "text": "x"}, headers=_bot()
        )
        assert resp.json()["ok"] is True
        assert web.posted[0].channel == DM_CHANNEL_ID

    def test_wait_for_posted_names_what_arrived(self, web: FakeSlackWeb) -> None:
        with pytest.raises(AssertionError, match="got 0"):
            web.wait_for_posted(1, timeout=0.1)

    def test_members_are_cursor_paged(self, web: FakeSlackWeb, client: httpx.Client) -> None:
        web.set_members(CHANNEL_ID, [f"U{i}" for i in range(5)])
        first = _call(client, "conversations.members", channel=CHANNEL_ID, limit=3).json()
        assert first["members"] == ["U0", "U1", "U2"]
        cursor = first["response_metadata"]["next_cursor"]
        second = _call(
            client, "conversations.members", channel=CHANNEL_ID, limit=3, cursor=cursor
        ).json()
        assert second["members"] == ["U3", "U4"]
        assert second["response_metadata"]["next_cursor"] == ""
        assert _call(client, "conversations.members", channel="C0NONE").json()["error"] == (
            "channel_not_found"
        )

    def test_users_info(self, web: FakeSlackWeb, client: httpx.Client) -> None:
        web.add_user("U0CAROL", real_name="Carol", display_name="carol", email="c@example.org")
        user = _call(client, "users.info", user="U0CAROL").json()["user"]
        assert user["profile"]["display_name"] == "carol"
        assert user["profile"]["email"] == "c@example.org"
        assert _call(client, "users.info", user=SENDER_ID).json()["ok"] is True
        assert _call(client, "users.info", user="U0NOBODY").json()["error"] == "user_not_found"

    def test_scripted_errors_answer_once_then_recover(
        self, web: FakeSlackWeb, client: httpx.Client
    ) -> None:
        web.fail_method("chat.postMessage", "not_in_channel")
        web.fail_method("chat.postMessage", "is_archived")
        args = {"channel": CHANNEL_ID, "text": "x"}
        first = _call(client, "chat.postMessage", **args)
        assert first.status_code == 200
        assert first.json() == {"ok": False, "error": "not_in_channel"}
        assert _call(client, "chat.postMessage", **args).json()["error"] == "is_archived"
        assert _call(client, "chat.postMessage", **args).json()["ok"] is True
        assert [c.error for c in web.calls_to("chat.postMessage")] == [
            "not_in_channel",
            "is_archived",
            "",
        ]
        assert len(web.posted) == 1

    def test_rate_limit_answers_429_with_retry_after(
        self, web: FakeSlackWeb, client: httpx.Client
    ) -> None:
        web.rate_limit("chat.postMessage", retry_after=2)
        limited = _call(client, "chat.postMessage", channel=CHANNEL_ID, text="x")
        assert limited.status_code == 429
        assert limited.headers["Retry-After"] == "2"
        assert limited.json()["error"] == "ratelimited"
        assert _call(client, "chat.postMessage", channel=CHANNEL_ID, text="x").status_code == 200

    def test_upload_trio_records_the_file(self, web: FakeSlackWeb, client: httpx.Client) -> None:
        content = b"col1,col2\n1,2\n"
        got = client.post(
            f"{API}/files.getUploadURLExternal",
            data={"filename": "data.csv", "length": str(len(content))},
            headers=_bot(),
        ).json()
        assert got["ok"] is True
        assert got["upload_url"].startswith("https://files.slack.com/upload/v1/")
        assert client.post(got["upload_url"], content=content).status_code == 200
        done = _call(
            client,
            "files.completeUploadExternal",
            files=[{"id": got["file_id"], "title": "Data"}],
            channel_id=CHANNEL_ID,
            thread_ts="1700000000.000100",
        ).json()
        assert done["ok"] is True
        [upload] = web.uploads
        assert (upload.content, upload.title, upload.channel_id, upload.thread_ts) == (
            content,
            "Data",
            CHANNEL_ID,
            "1700000000.000100",
        )
        assert upload.upload_had_authorization is False

    def test_complete_without_bytes_is_refused(self, client: httpx.Client) -> None:
        got = _call(client, "files.getUploadURLExternal", filename="a.txt", length=3).json()
        done = _call(client, "files.completeUploadExternal", files=[{"id": got["file_id"]}])
        assert done.json()["error"] == "file_not_found"

    def test_file_route_needs_the_bot_token(self, web: FakeSlackWeb, client: httpx.Client) -> None:
        ref = web.add_file("plot.png", b"\x89PNG-bytes", mimetype="image/png")
        assert ref["url_private_download"].startswith("https://files.slack.com/files-pri/")
        good = client.get(ref["url_private_download"], headers=_bot())
        assert good.status_code == 200
        assert good.content == b"\x89PNG-bytes"
        assert good.headers["Content-Type"] == "image/png"
        anon = client.get(ref["url_private"])
        assert anon.status_code == 200
        assert anon.headers["Content-Type"].startswith("text/html")
        assert web.downloads == [(ref["id"], True), (ref["id"], False)]

    def test_redirect_records_attempts_and_refuses_foreign_hosts(
        self, web: FakeSlackWeb, client: httpx.Client
    ) -> None:
        _call(client, "auth.test")
        with pytest.raises(httpx.ConnectError):
            client.get("https://evil.example.org/file.png")
        slack, foreign = web.attempts
        assert slack.url == f"{API}/auth.test"
        assert slack.authorization == f"Bearer {BOT_TOKEN}"
        assert slack.redirected is True
        assert foreign.url == "https://evil.example.org/file.png"
        assert foreign.authorization is None
        assert foreign.redirected is False

    def test_reset_forgets_traffic(self, web: FakeSlackWeb, client: httpx.Client) -> None:
        _call(client, "chat.postMessage", channel=CHANNEL_ID, text="x")
        web.reset()
        assert web.posted == [] and web.calls == [] and web.attempts == []


# ---------------------------------------------------------------------------
# FakeSocketServer
# ---------------------------------------------------------------------------


def _open_socket(client: httpx.Client) -> str:
    body = _call(client, "apps.connections.open", token=APP_TOKEN).json()
    assert body["ok"] is True
    return body["url"]


class TestFakeSocketServer:
    def test_connections_open_mints_a_loopback_ws_url(
        self, socket_server: FakeSocketServer, client: httpx.Client
    ) -> None:
        first, second = _open_socket(client), _open_socket(client)
        assert first.startswith(f"ws://127.0.0.1:{socket_server.port}/link/?ticket=")
        assert first != second

    def test_no_socket_configured(self) -> None:
        with FakeSlackWeb() as web, web.http_client() as c:
            body = _call(c, "apps.connections.open", token=APP_TOKEN).json()
        assert body == {"ok": False, "error": "no_socket_configured"}

    def test_hello_push_and_ack(
        self, socket_server: FakeSocketServer, client: httpx.Client
    ) -> None:
        url = _open_socket(client)
        with connect(url, open_timeout=TIMEOUT) as ws:
            hello = json.loads(ws.recv(timeout=TIMEOUT))
            assert hello["type"] == "hello"
            assert hello["connection_info"]["app_id"]
            socket_server.wait_for_connections(1, timeout=TIMEOUT)
            env = envelope(dm_message("hi"))
            socket_server.push(env)
            got = json.loads(ws.recv(timeout=TIMEOUT))
            assert got == env
            ws.send(json.dumps({"envelope_id": got["envelope_id"]}))
            assert socket_server.wait_for_acks(1, timeout=TIMEOUT) == [env["envelope_id"]]
        assert socket_server.paths[0].startswith("/link/?ticket=")

    @pytest.mark.parametrize("reason", ["warning", "refresh_requested", "link_disabled"])
    def test_disconnect_reasons(
        self, socket_server: FakeSocketServer, client: httpx.Client, reason: str
    ) -> None:
        with connect(_open_socket(client), open_timeout=TIMEOUT) as ws:
            ws.recv(timeout=TIMEOUT)
            socket_server.wait_for_connections(1, timeout=TIMEOUT)
            socket_server.send_disconnect(reason)
            frame = json.loads(ws.recv(timeout=TIMEOUT))
        assert frame["type"] == "disconnect" and frame["reason"] == reason

    def test_push_goes_to_the_newest_connection_and_drop_closes(
        self, socket_server: FakeSocketServer, client: httpx.Client
    ) -> None:
        with (
            connect(_open_socket(client), open_timeout=TIMEOUT) as old,
            connect(_open_socket(client), open_timeout=TIMEOUT) as new,
        ):
            old.recv(timeout=TIMEOUT)
            new.recv(timeout=TIMEOUT)
            socket_server.wait_for_connections(2, timeout=TIMEOUT)
            socket_server.push({"type": "events_api", "envelope_id": "e1", "payload": {}})
            assert json.loads(new.recv(timeout=TIMEOUT))["envelope_id"] == "e1"
            socket_server.drop_connections()
            with pytest.raises(ConnectionClosed):
                old.recv(timeout=TIMEOUT)
            with pytest.raises(ConnectionClosed):
                new.recv(timeout=TIMEOUT)

    def test_wait_for_acks_names_what_arrived(self, socket_server: FakeSocketServer) -> None:
        with pytest.raises(AssertionError, match="ack"):
            socket_server.wait_for_acks(1, timeout=0.1)


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


class TestEventBuilders:
    def test_payload_is_an_event_callback(self) -> None:
        payload = dm_message("hello")
        assert payload["type"] == "event_callback"
        assert payload["team_id"] == TEAM_ID
        assert payload["is_ext_shared_channel"] is False
        assert payload["event_id"].startswith("Ev")
        assert payload["authorizations"][0]["user_id"] == BOT_USER_ID

    def test_dm(self) -> None:
        event = dm_message("hello")["event"]
        assert (event["type"], event["channel_type"], event["channel"]) == (
            "message",
            "im",
            DM_CHANNEL_ID,
        )
        assert event["user"] == SENDER_ID and "thread_ts" not in event

    def test_channel_mention_has_no_channel_type(self) -> None:
        event = channel_mention("status?")["event"]
        assert event["type"] == "app_mention"
        assert "channel_type" not in event
        assert event["text"] == f"<@{BOT_USER_ID}> status?"
        assert channel_mention("")["event"]["text"] == f"<@{BOT_USER_ID}>"

    def test_mpim_arrives_twice_with_one_ts(self) -> None:
        as_message, as_mention = mpim_message("q")
        m, a = as_message["event"], as_mention["event"]
        assert (m["type"], m["channel_type"]) == ("message", "mpim")
        assert a["type"] == "app_mention" and "channel_type" not in a
        assert (m["channel"], m["ts"]) == (a["channel"], a["ts"]) == (m["channel"], m["ts"])
        assert m["channel"] == MPIM_CHANNEL_ID
        assert as_message["event_id"] != as_mention["event_id"]

    def test_thread_reply(self) -> None:
        parent = next_ts()
        channel = thread_reply(parent)["event"]
        assert channel["type"] == "app_mention" and channel["thread_ts"] == parent
        assert channel["ts"] != parent
        dm = thread_reply(parent, channel=DM_CHANNEL_ID)["event"]
        assert dm["type"] == "message" and dm["thread_ts"] == parent

    def test_file_share(self, web: FakeSlackWeb) -> None:
        ref = web.add_file("a.txt", b"abc")
        event = file_share([ref])["event"]
        assert event["subtype"] == "file_share"
        assert event["files"][0]["url_private_download"] == ref["url_private_download"]
        assert event["files"][0]["size"] == 3

    def test_bot_message(self) -> None:
        event = bot_message()["event"]
        assert event["subtype"] == "bot_message" and event["bot_id"]
        assert "user" not in event

    def test_shared_channel(self) -> None:
        payload = shared_channel_mention()
        assert payload["is_ext_shared_channel"] is True
        assert payload["event"]["user_team"] == FOREIGN_TEAM_ID

    def test_ts_strictly_increases(self) -> None:
        a, b = next_ts(), next_ts()
        assert float(b) > float(a)

    def test_envelope_retry_fields_are_optional(self) -> None:
        plain = envelope(dm_message())
        assert plain["type"] == "events_api" and "retry_attempt" not in plain
        retried = envelope(dm_message(), retry_attempt=1, retry_reason="timeout")
        assert (retried["retry_attempt"], retried["retry_reason"]) == (1, "timeout")
