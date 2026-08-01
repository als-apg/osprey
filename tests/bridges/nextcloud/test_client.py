"""TalkClient's OCS chat/room surface: routes, headers, the 304 idle path, the
long-poll timeout profile, and error propagation.

Every test drives a :class:`httpx.MockTransport` — nothing here opens a socket.
Two properties get more attention than the rest because they are the ones that
break a live bridge silently:

* a long poll's **read timeout must outlast the server's hold**, or every idle
  room reports a transport error instead of "nothing new";
* a ``304`` is Talk's normal idle answer and must never raise.

The suite also pins that the client adds **no retry** — a failure makes exactly
one request and propagates — because the room-type cache and the poll loop each
own their own capped backoff over these calls.
"""

import base64

import httpx
import pytest

from osprey.bridges.nextcloud_talk import NextcloudBridgeConfig, RoomInfo, TalkApiError, TalkClient
from osprey.bridges.nextcloud_talk.client import (
    CONNECT_TIMEOUT,
    DEFAULT_POLL_HOLD,
    LAST_GIVEN_HEADER,
    POLL_READ_MARGIN,
    REQUEST_TIMEOUT,
    ROOM_TYPE_ONE_TO_ONE,
)

CFG = NextcloudBridgeConfig(
    base_url="https://cloud.example.org",
    bot_account="osprey-bot",
    app_password="app-pw",
    rooms=("roomA",),
)

EXPECTED_AUTH = "Basic " + base64.b64encode(b"osprey-bot:app-pw").decode()

CHAT_PATH = "/ocs/v2.php/apps/spreed/api/v1/chat/roomA"
ROOM_PATH = "/ocs/v2.php/apps/spreed/api/v4/room/roomA"

# A group room with one message in it — the shape most tests want back.
ROOM_PAYLOAD = {"type": 2, "displayName": "Ops", "lastMessage": {"id": 7}}


def _ocs(data, *, status=200, headers=None):
    """An OCS v2 envelope response carrying ``data``."""
    return httpx.Response(
        status,
        json={"ocs": {"meta": {"status": "ok", "statuscode": status}, "data": data}},
        headers=headers or {},
    )


def _talk(request):
    """A minimal Talk that answers all three calls with valid payloads."""
    if "/api/v4/room/" in request.url.path:
        return _ocs(ROOM_PAYLOAD)
    if request.method == "POST":
        return _ocs({"id": 99})
    return _ocs([], headers={LAST_GIVEN_HEADER: "41"})


def _client(handler=_talk, **client_kwargs):
    """A TalkClient over a MockTransport running ``handler``."""
    transport = httpx.MockTransport(handler)
    return TalkClient(CFG, client=httpx.Client(transport=transport, **client_kwargs))


def _recording(handler=_talk, **client_kwargs):
    """``(client, requests)`` — as :func:`_client`, recording every request."""
    seen: list[httpx.Request] = []

    def recorder(request):
        seen.append(request)
        return handler(request)

    return _client(recorder, **client_kwargs), seen


# Each entry drives one public call, so a property that must hold for the whole
# surface (auth, headers, no retry) is asserted once per method.
ALL_CALLS = [
    pytest.param(lambda c: c.poll_messages("roomA", 0), id="poll_messages"),
    pytest.param(lambda c: c.post_message("roomA", "hi"), id="post_message"),
    pytest.param(lambda c: c.room_info("roomA"), id="room_info"),
]


# ==========================================================================
# Route and query construction — chat is v1, room is v4 (deliberately)
# ==========================================================================


def test_poll_targets_the_v1_chat_route_with_the_future_cursor():
    client, seen = _recording()
    client.poll_messages("roomA", 41)
    request = seen[0]
    assert request.method == "GET"
    assert str(request.url).startswith(f"https://cloud.example.org{CHAT_PATH}?")
    assert request.url.params["lookIntoFuture"] == "1"
    assert request.url.params["lastKnownMessageId"] == "41"
    # The anchor message itself stays out of the batch, so an idle poll cannot
    # re-deliver the message the cursor points at.
    assert request.url.params["includeLastKnown"] == "0"
    assert request.url.params["timeout"] == str(int(DEFAULT_POLL_HOLD))


def test_poll_sends_the_hold_truncated_to_whole_seconds():
    client, seen = _recording()
    client.poll_messages("roomA", 0, timeout=2.5)
    assert seen[0].url.params["timeout"] == "2"


def test_post_targets_the_v1_chat_route():
    client, seen = _recording()
    client.post_message("roomA", "hi")
    assert seen[0].method == "POST"
    assert str(seen[0].url) == f"https://cloud.example.org{CHAT_PATH}"


def test_room_info_targets_the_v4_room_route():
    client, seen = _recording()
    client.room_info("roomA")
    assert seen[0].method == "GET"
    assert str(seen[0].url) == f"https://cloud.example.org{ROOM_PATH}"


@pytest.mark.parametrize(
    ("token", "encoded"),
    [("od/dtoken", b"od%2Fdtoken"), ("tok?x=1", b"tok%3Fx%3D1"), ("tok en", b"tok%20en")],
)
def test_room_token_goes_on_the_wire_as_one_encoded_path_segment(token, encoded):
    # Room tokens come from operator-set config; encoding confines a token with
    # a delimiter in it to its own path segment, so it can only 404 on the
    # intended route instead of addressing a different one (or, for "?",
    # silently becoming a query string).
    client, seen = _recording()
    client.room_info(token)
    assert seen[0].url.raw_path == b"/ocs/v2.php/apps/spreed/api/v4/room/" + encoded
    assert seen[0].url.query == b""


# ==========================================================================
# The 304 idle path — Talk's "nothing new within the hold window"
# ==========================================================================


def test_poll_treats_304_as_an_empty_batch_and_not_an_error():
    client, seen = _recording(lambda request: httpx.Response(304))
    assert client.poll_messages("roomA", 41) == ([], None)
    assert len(seen) == 1


def test_poll_ignores_a_cursor_header_on_a_304():
    # A 304 means nothing was given, so no cursor can be advanced from it even
    # if the server echoes the header.
    client = _client(lambda request: httpx.Response(304, headers={LAST_GIVEN_HEADER: "77"}))
    assert client.poll_messages("roomA", 41) == ([], None)


# ==========================================================================
# X-Chat-Last-Given — the cursor for the next poll
# ==========================================================================


def test_poll_returns_messages_and_the_cursor_header():
    messages = [{"id": 42, "message": "hello"}, {"id": 43, "message": "again"}]
    client = _client(lambda request: _ocs(messages, headers={LAST_GIVEN_HEADER: "43"}))
    assert client.poll_messages("roomA", 41) == (messages, 43)


def test_poll_returns_no_cursor_when_the_header_is_absent():
    client = _client(lambda request: _ocs([{"id": 42}]))
    batch, last_given = client.poll_messages("roomA", 41)
    assert batch == [{"id": 42}]
    assert last_given is None


def test_poll_keeps_the_batch_when_the_cursor_header_is_unparseable():
    # The cursor is advisory — the caller can fall back to the batch's highest
    # id — so a junk header must not throw away messages that did arrive.
    client = _client(lambda request: _ocs([{"id": 42}], headers={LAST_GIVEN_HEADER: "not-an-id"}))
    assert client.poll_messages("roomA", 41) == ([{"id": 42}], None)


# ==========================================================================
# Auth and the headers OCS requires, on every call
# ==========================================================================


@pytest.mark.parametrize("call", ALL_CALLS)
def test_every_call_carries_basic_auth_and_the_ocs_headers(call):
    client, seen = _recording()
    call(client)
    headers = seen[0].headers
    assert headers["authorization"] == EXPECTED_AUTH
    # Nextcloud rejects OCS calls without OCS-APIRequest, and answers XML
    # without the JSON Accept header.
    assert headers["ocs-apirequest"] == "true"
    assert headers["accept"] == "application/json"


def test_credentials_come_from_the_config():
    cfg = NextcloudBridgeConfig(
        base_url="https://nc.example.org", bot_account="other-bot", app_password="other-pw"
    )
    seen: list[httpx.Request] = []

    def recorder(request):
        seen.append(request)
        return _ocs(ROOM_PAYLOAD)

    client = TalkClient(cfg, client=httpx.Client(transport=httpx.MockTransport(recorder)))
    client.room_info("roomA")
    expected = "Basic " + base64.b64encode(b"other-bot:other-pw").decode()
    assert seen[0].headers["authorization"] == expected
    assert str(seen[0].url).startswith("https://nc.example.org/ocs/v2.php/")


# ==========================================================================
# The timeout profile — a long poll must outlast the server's hold
# ==========================================================================


@pytest.mark.parametrize("hold", [0.0, 5.0, DEFAULT_POLL_HOLD, 60.0])
def test_long_poll_read_timeout_outlasts_the_requested_hold(hold):
    # THE detail: Talk holds an idle poll open for `hold` seconds before
    # answering 304. A read timeout at or below that would end every idle poll
    # locally as a ReadTimeout, making an idle room look like a broken network.
    client, seen = _recording()
    client.poll_messages("roomA", 0, timeout=hold)
    timeout = seen[0].extensions["timeout"]
    assert timeout["read"] > hold
    assert timeout["read"] == hold + POLL_READ_MARGIN


def test_long_poll_overrides_only_the_read_leg():
    client, seen = _recording()
    client.poll_messages("roomA", 0, timeout=DEFAULT_POLL_HOLD)
    timeout = seen[0].extensions["timeout"]
    assert timeout["connect"] == CONNECT_TIMEOUT
    assert timeout["write"] == REQUEST_TIMEOUT
    assert timeout["pool"] == CONNECT_TIMEOUT


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda c: c.post_message("roomA", "hi"), id="post_message"),
        pytest.param(lambda c: c.room_info("roomA"), id="room_info"),
    ],
)
def test_ordinary_calls_inherit_the_client_wide_timeout(call):
    # No per-request override on the short calls: they use the client's profile,
    # here a sentinel value proving nothing local re-specifies it.
    client, seen = _recording(timeout=httpx.Timeout(7.0))
    call(client)
    assert seen[0].extensions["timeout"] == {
        "connect": 7.0,
        "read": 7.0,
        "write": 7.0,
        "pool": 7.0,
    }


def test_default_client_uses_the_documented_timeout_profile():
    client = TalkClient(CFG)
    try:
        assert client._client.timeout == httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=REQUEST_TIMEOUT,
            write=REQUEST_TIMEOUT,
            pool=CONNECT_TIMEOUT,
        )
    finally:
        client.close()


def test_poll_rejects_a_negative_hold():
    client = _client()
    with pytest.raises(ValueError, match="hold must be >= 0"):
        client.poll_messages("roomA", 0, timeout=-1.0)


# ==========================================================================
# post_message
# ==========================================================================


def test_post_sends_the_text_form_encoded():
    client, seen = _recording()
    client.post_message("roomA", "hello world")
    assert seen[0].headers["content-type"] == "application/x-www-form-urlencoded"
    assert seen[0].content == b"message=hello+world"


def test_post_omits_reply_to_by_default():
    client, seen = _recording()
    client.post_message("roomA", "hi")
    assert b"replyTo" not in seen[0].content


def test_post_includes_reply_to_when_given():
    client, seen = _recording()
    client.post_message("roomA", "hi", reply_to=41)
    assert seen[0].content == b"message=hi&replyTo=41"


def test_post_returns_the_new_message_id():
    client = _client(lambda request: _ocs({"id": 99, "message": "hi"}))
    assert client.post_message("roomA", "hi") == 99


@pytest.mark.parametrize("data", [{}, {"id": "not-an-int"}, []])
def test_post_returns_none_when_the_response_carries_no_id(data):
    client = _client(lambda request: _ocs(data))
    assert client.post_message("roomA", "hi") is None


# ==========================================================================
# room_info parsing
# ==========================================================================


def test_room_info_parses_type_name_and_last_message():
    client = _client(lambda request: _ocs(ROOM_PAYLOAD))
    assert client.room_info("roomA") == RoomInfo(
        token="roomA", type=2, display_name="Ops", last_message_id=7
    )


def test_room_info_reports_a_one_to_one_room():
    client = _client(lambda request: _ocs({"type": ROOM_TYPE_ONE_TO_ONE, "displayName": "bob"}))
    assert client.room_info("roomA").type == ROOM_TYPE_ONE_TO_ONE


def test_room_info_reads_no_last_message_from_talks_empty_array():
    # Talk serializes an absent lastMessage as an empty *list* (PHP's empty
    # array), not null and not an empty object.
    client = _client(lambda request: _ocs({"type": 2, "displayName": "Ops", "lastMessage": []}))
    assert client.room_info("roomA").last_message_id is None


@pytest.mark.parametrize("payload", [{"type": 2}, {"type": 2, "displayName": None}])
def test_room_info_display_name_falls_back_to_empty_string(payload):
    client = _client(lambda request: _ocs(payload))
    assert client.room_info("roomA").display_name == ""


@pytest.mark.parametrize("payload", [{}, {"type": None}, {"type": "2"}])
def test_room_info_raises_when_the_type_is_not_an_integer(payload):
    # The bridge must never guess a room type: guessing "group" drops every
    # message in a one-to-one, guessing "one-to-one" answers every unaddressed
    # message in a group room.
    client = _client(lambda request: _ocs(payload))
    with pytest.raises(TalkApiError, match="no integer type"):
        client.room_info("roomA")


def test_room_info_raises_when_data_is_not_an_object():
    client = _client(lambda request: _ocs(["not", "a", "room"]))
    with pytest.raises(TalkApiError, match="not an object"):
        client.room_info("roomA")


# ==========================================================================
# Error propagation — raise to the caller; backoff is the caller's
# ==========================================================================


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 503])
@pytest.mark.parametrize("call", ALL_CALLS)
def test_error_statuses_raise_to_the_caller(call, status):
    client = _client(lambda request: httpx.Response(status, text="boom"))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        call(client)
    assert excinfo.value.response.status_code == status


@pytest.mark.parametrize("call", ALL_CALLS)
def test_transport_errors_propagate(call):
    def broken(request):
        raise httpx.ConnectError("no route to host", request=request)

    client = _client(broken)
    with pytest.raises(httpx.ConnectError):
        call(client)


@pytest.mark.parametrize("call", ALL_CALLS)
def test_a_failure_is_attempted_exactly_once(call):
    # No retry lives here: the room-type cache and the poll loop each own a
    # capped backoff, and a hidden retry would stretch one logical attempt past
    # the budget its caller thinks it has.
    client, seen = _recording(lambda request: httpx.Response(503, text="down"))
    with pytest.raises(httpx.HTTPStatusError):
        call(client)
    assert len(seen) == 1


@pytest.mark.parametrize("call", ALL_CALLS)
def test_a_non_json_body_raises_talk_api_error(call):
    client = _client(lambda request: httpx.Response(200, text="<html>login</html>"))
    with pytest.raises(TalkApiError, match="non-JSON body"):
        call(client)


@pytest.mark.parametrize("call", ALL_CALLS)
def test_a_body_without_an_ocs_envelope_raises_talk_api_error(call):
    client = _client(lambda request: httpx.Response(200, json={"data": []}))
    with pytest.raises(TalkApiError, match="no OCS envelope"):
        call(client)


@pytest.mark.parametrize("call", ALL_CALLS)
def test_an_envelope_without_data_raises_talk_api_error(call):
    client = _client(lambda request: httpx.Response(200, json={"ocs": {"meta": {}}}))
    with pytest.raises(TalkApiError, match="without data"):
        call(client)


@pytest.mark.parametrize("data", [{"id": 42}, "nope", [{"id": 42}, "nope"]])
def test_poll_raises_when_the_chat_payload_is_not_a_message_list(data):
    client = _client(lambda request: _ocs(data))
    with pytest.raises(TalkApiError, match="not a list of messages"):
        client.poll_messages("roomA", 41)


# ==========================================================================
# Client ownership: config-driven proxy trust, injection, teardown
# ==========================================================================


@pytest.mark.parametrize("trust_env", [False, True])
def test_default_client_honors_core_trust_env(trust_env):
    from osprey.bridges.core import CoreConfig

    client = TalkClient(NextcloudBridgeConfig(core=CoreConfig(trust_env=trust_env)))
    try:
        assert client._client.trust_env is trust_env
    finally:
        client.close()


def test_injected_client_is_used_as_given():
    injected = httpx.Client(transport=httpx.MockTransport(lambda request: _ocs([])))
    client = TalkClient(CFG, client=injected)
    assert client._client is injected
    client.close()


def test_close_closes_the_shared_client():
    client = _client()
    client.close()
    assert client._client.is_closed


def test_package_exports_the_client_surface():
    import osprey.bridges.nextcloud_talk as pkg

    assert (pkg.TalkClient, pkg.RoomInfo, pkg.TalkApiError) == (TalkClient, RoomInfo, TalkApiError)
