"""Tests for the Teams bridge's HTTP clients.

Every test drives the real code over an :class:`httpx.MockTransport`, so the
suite needs no network and no Azure tenant: the transport handler *is* the AAD
token endpoint (or, for the connector half, the Bot Connector), and it records
what was asked of it.

The clock is injected for the same reason. Expiry is the one behaviour here that
cannot be observed without controlling time, and sleeping through a real
``expires_in`` would make the suite either slow or a liar.
"""

from __future__ import annotations

import json
import threading
import time
from urllib.parse import parse_qs

import httpx
import pytest

from osprey.bridges.teams.client import (
    CONNECTOR_TIMEOUT_SEC,
    EXPIRY_MARGIN_SEC,
    ConnectorClient,
    ConnectorError,
    MessageSizeTooBig,
    TokenError,
    TokenSource,
    token_url,
)
from osprey.bridges.teams.config import TeamsBridgeConfig

APP_ID = "11111111-2222-3333-4444-555555555555"
APP_SECRET = "a-client-secret"
TENANT = "66666666-7777-8888-9999-000000000000"
EXPIRES_IN = 3600


def make_config(cloud: str = "commercial") -> TeamsBridgeConfig:
    """A config carrying only what the token exchange reads."""
    return TeamsBridgeConfig(
        app_id=APP_ID,
        app_secret=APP_SECRET,
        tenant_id=TENANT,
        cloud=cloud,
    )


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TokenEndpoint:
    """A recording stand-in for ``POST /{tenant}/oauth2/v2.0/token``.

    Answers a fresh ``access_token`` per call so a test can tell a cache hit from
    a refetch by the value alone, and keeps every request for assertion.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        body: dict | None = None,
        raises: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.status = status
        self.body = body
        self.raises = raises
        self.delay = delay
        self.requests: list[httpx.Request] = []
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
            count = len(self.requests)
        if self.delay:
            time.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        body = self.body
        if body is None:
            body = {"access_token": f"token-{count}", "expires_in": EXPIRES_IN}
        return httpx.Response(self.status, json=body)

    @property
    def calls(self) -> int:
        return len(self.requests)

    def form(self, index: int = 0) -> dict[str, str]:
        """The indexed request's form body, one value per field."""
        parsed = parse_qs(self.requests[index].content.decode())
        return {key: values[0] for key, values in parsed.items()}


def make_source(
    endpoint: TokenEndpoint,
    *,
    cfg: TeamsBridgeConfig | None = None,
    clock: FakeClock | None = None,
) -> TokenSource:
    """A :class:`TokenSource` wired to ``endpoint`` over a mock transport."""
    http = httpx.Client(transport=httpx.MockTransport(endpoint))
    return TokenSource(cfg or make_config(), http, clock=clock or FakeClock())


# --- the exchange ----------------------------------------------------------


def test_the_token_url_is_the_clouds_login_host_and_the_configured_tenant():
    assert token_url(make_config()) == (
        f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
    )
    assert token_url(make_config("gcchigh")) == (
        f"https://login.microsoftonline.us/{TENANT}/oauth2/v2.0/token"
    )


def test_a_token_is_fetched_with_the_client_credentials_grant():
    endpoint = TokenEndpoint()
    source = make_source(endpoint)

    assert source.token() == "token-1"

    request = endpoint.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
    assert endpoint.form() == {
        "grant_type": "client_credentials",
        "client_id": APP_ID,
        "client_secret": APP_SECRET,
        "scope": "https://api.botframework.com/.default",
    }


def test_a_gcchigh_token_is_fetched_from_the_us_host_with_the_us_scope():
    endpoint = TokenEndpoint()
    source = make_source(endpoint, cfg=make_config("gcchigh"))

    assert source.token() == "token-1"

    assert str(endpoint.requests[0].url).startswith("https://login.microsoftonline.us/")
    assert endpoint.form()["scope"] == "https://api.botframework.us/.default"


# --- caching ---------------------------------------------------------------


def test_a_cached_token_is_reused_without_a_second_request():
    endpoint = TokenEndpoint()
    clock = FakeClock()
    source = make_source(endpoint, clock=clock)

    first = source.token()
    clock.advance(EXPIRES_IN - EXPIRY_MARGIN_SEC - 1)
    second = source.token()

    assert first == second == "token-1"
    assert endpoint.calls == 1


def test_an_expired_token_is_refetched():
    endpoint = TokenEndpoint()
    clock = FakeClock()
    source = make_source(endpoint, clock=clock)

    assert source.token() == "token-1"
    clock.advance(EXPIRES_IN)

    assert source.token() == "token-2"
    assert endpoint.calls == 2


def test_a_token_is_refetched_a_margin_before_it_actually_expires():
    # The margin is the whole point: a token handed out one second before the
    # service stops honouring it is an answer that fails in flight.
    endpoint = TokenEndpoint()
    clock = FakeClock()
    source = make_source(endpoint, clock=clock)

    assert source.token() == "token-1"
    clock.advance(EXPIRES_IN - EXPIRY_MARGIN_SEC)

    assert source.token() == "token-2"
    assert endpoint.calls == 2


def test_a_token_response_without_an_expiry_is_used_once_and_not_cached():
    endpoint = TokenEndpoint(body={"access_token": "no-expiry"})
    source = make_source(endpoint)

    assert source.token() == "no-expiry"
    assert source.token() == "no-expiry"
    assert endpoint.calls == 2


def test_a_token_that_expires_inside_the_margin_is_not_cached():
    endpoint = TokenEndpoint(body={"access_token": "brief", "expires_in": 30})
    source = make_source(endpoint)

    assert source.token() == "brief"
    assert source.token() == "brief"
    assert endpoint.calls == 2


def test_concurrent_callers_share_one_token_fetch():
    # The drain thread and the receive loop both post, and a refresh per caller
    # would mean one AAD round trip per reply under load.
    endpoint = TokenEndpoint(delay=0.02)
    source = make_source(endpoint)
    seen: list[str] = []
    guard = threading.Lock()

    def call() -> None:
        value = source.token()
        with guard:
            seen.append(value)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert endpoint.calls == 1
    assert seen == ["token-1"] * 8


# --- failures --------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 500, 503])
def test_a_non_2xx_token_response_raises(status):
    endpoint = TokenEndpoint(status=status, body={"error": "invalid_client"})
    source = make_source(endpoint)

    with pytest.raises(TokenError, match=str(status)):
        source.token()


def test_a_failed_token_response_is_not_cached_as_a_token():
    endpoint = TokenEndpoint(status=401, body={"error": "invalid_client"})
    source = make_source(endpoint)

    with pytest.raises(TokenError):
        source.token()
    with pytest.raises(TokenError):
        source.token()

    assert endpoint.calls == 2


def test_a_token_transport_failure_raises_token_error():
    endpoint = TokenEndpoint(raises=httpx.ConnectError("no route to host"))
    source = make_source(endpoint)

    with pytest.raises(TokenError, match="no route to host"):
        source.token()


def test_a_token_response_that_is_not_json_raises():
    http = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>"))
    )
    source = TokenSource(make_config(), http, clock=FakeClock())

    with pytest.raises(TokenError):
        source.token()


def test_a_token_response_without_an_access_token_raises():
    endpoint = TokenEndpoint(body={"expires_in": EXPIRES_IN})
    source = make_source(endpoint)

    with pytest.raises(TokenError, match="access_token"):
        source.token()


def test_a_token_error_never_repeats_the_client_secret():
    # The error travels into a container log; the request body it describes
    # carries the bot's secret.
    endpoint = TokenEndpoint(status=401, body={"error_description": "AADSTS7000215"})
    source = make_source(endpoint)

    with pytest.raises(TokenError) as excinfo:
        source.token()

    assert APP_SECRET not in str(excinfo.value)


def test_the_token_source_defaults_to_a_real_monotonic_clock():
    endpoint = TokenEndpoint()
    http = httpx.Client(transport=httpx.MockTransport(endpoint))
    source = TokenSource(make_config(), http)

    assert source.token() == "token-1"
    assert source.token() == "token-1"
    assert endpoint.calls == 1


# --- the Bot Connector -----------------------------------------------------

SERVICE_URL = "https://smba.trafficmanager.net/amer/"
"""A ``serviceUrl`` shaped the way Teams actually sends one — with the trailing
slash that would otherwise double up in the posted path."""

CHANNEL_CONVERSATION = "19:aaaabbbbcccc@thread.tacv2;messageid=1700000000000"
"""A channel reply's conversation id: a thread id carrying the ``;messageid=``
suffix that roots the reply under the original message. It is posted exactly as
the activity sent it — dropping or re-encoding the suffix moves the answer out of
the thread it belongs to."""

ACTIVITY_ID = "1700000000001"
ACTIVITY = {"type": "message", "text": "hello"}


class ConnectorEndpoint:
    """A recording stand-in for ``POST /v3/conversations/{id}/activities/{id}``.

    Tracks how many calls are in flight at once as well as what was asked, so the
    lock discipline is observable rather than assumed.
    """

    def __init__(
        self,
        *,
        status: int = 201,
        body: dict | None = None,
        raises: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.status = status
        self.body = body
        self.raises = raises
        self.delay = delay
        self.requests: list[httpx.Request] = []
        self.peak_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
            count = len(self.requests)
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.raises is not None:
                raise self.raises
            body = self.body if self.body is not None else {"id": f"posted-{count}"}
            return httpx.Response(self.status, json=body)
        finally:
            with self._lock:
                self._in_flight -= 1

    @property
    def calls(self) -> int:
        return len(self.requests)

    def bearer(self, index: int = 0) -> str:
        return self.requests[index].headers["authorization"]

    def sent(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content.decode())


def make_connector(
    endpoint: ConnectorEndpoint,
    *,
    source: TokenSource | None = None,
    cfg: TeamsBridgeConfig | None = None,
) -> ConnectorClient:
    """A :class:`ConnectorClient` wired to ``endpoint`` over a mock transport."""
    cfg = cfg or make_config()
    http = httpx.Client(transport=httpx.MockTransport(endpoint))
    return ConnectorClient(cfg, source or make_source(TokenEndpoint(), cfg=cfg), http)


def test_a_reply_posts_the_activity_to_the_conversations_activity_url():
    endpoint = ConnectorEndpoint()
    client = make_connector(endpoint)

    assert client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY) is None

    request = endpoint.requests[0]
    assert request.method == "POST"
    assert str(request.url) == (
        "https://smba.trafficmanager.net/amer"
        f"/v3/conversations/{CHANNEL_CONVERSATION}/activities/{ACTIVITY_ID}"
    )
    assert endpoint.sent() == ACTIVITY
    assert request.headers["content-type"] == "application/json"


def test_a_reply_carries_the_bots_bearer_credential():
    endpoint = ConnectorEndpoint()
    client = make_connector(endpoint)

    client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)

    assert endpoint.bearer() == "Bearer token-1"


def test_a_reply_sends_the_conversation_id_exactly_as_the_activity_carried_it():
    # A channel reply's id carries ';messageid=<root>'; re-encoding or trimming it
    # posts the answer somewhere other than the thread that asked.
    endpoint = ConnectorEndpoint()
    client = make_connector(endpoint)

    client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)

    path = endpoint.requests[0].url.raw_path.decode()
    assert f"/v3/conversations/{CHANNEL_CONVERSATION}/activities/" in path


def test_a_service_url_without_a_trailing_slash_posts_to_the_same_path():
    endpoint = ConnectorEndpoint()
    client = make_connector(endpoint)

    client.reply(SERVICE_URL.rstrip("/"), CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)

    assert str(endpoint.requests[0].url) == (
        "https://smba.trafficmanager.net/amer"
        f"/v3/conversations/{CHANNEL_CONVERSATION}/activities/{ACTIVITY_ID}"
    )


def test_each_reply_asks_for_the_current_bearer_credential():
    # The credential outlives a single post but not the process: a client that
    # captured one at construction would post with a dead one an hour in.
    credentials = TokenEndpoint()
    clock = FakeClock()
    endpoint = ConnectorEndpoint()
    client = make_connector(endpoint, source=make_source(credentials, clock=clock))

    client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)
    clock.advance(EXPIRES_IN)
    client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)

    assert endpoint.bearer(0) == "Bearer token-1"
    assert endpoint.bearer(1) == "Bearer token-2"


def test_concurrent_replies_are_serialized_behind_one_lock():
    # The drain thread and the receive loop post at the same time over one client.
    endpoint = ConnectorEndpoint(delay=0.02)
    client = make_connector(endpoint)

    def call() -> None:
        client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert endpoint.calls == 8
    assert endpoint.peak_in_flight == 1


# --- connector failures ----------------------------------------------------


def test_an_oversized_activity_raises_message_size_too_big():
    endpoint = ConnectorEndpoint(status=413, body={"error": {"code": "MessageSizeTooBig"}})
    client = make_connector(endpoint)

    with pytest.raises(MessageSizeTooBig):
        client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)


def test_an_oversized_activity_is_also_a_connector_failure():
    # ops re-splits on the size failure and gives up on everything else, so the
    # narrow type must be catchable first and still answer to the general one.
    assert issubclass(MessageSizeTooBig, ConnectorError)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 502, 503])
def test_any_other_non_2xx_reply_raises_a_connector_error(status):
    endpoint = ConnectorEndpoint(status=status, body={"error": {"code": "ServiceError"}})
    client = make_connector(endpoint)

    with pytest.raises(ConnectorError, match=str(status)) as excinfo:
        client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)

    assert not isinstance(excinfo.value, MessageSizeTooBig)


def test_a_connector_error_names_the_activity_it_could_not_post():
    endpoint = ConnectorEndpoint(status=500, body={"error": {"code": "ServiceError"}})
    client = make_connector(endpoint)

    with pytest.raises(ConnectorError) as excinfo:
        client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)

    message = str(excinfo.value)
    assert CHANNEL_CONVERSATION in message
    assert "ServiceError" in message


def test_a_connector_transport_failure_raises_a_connector_error():
    endpoint = ConnectorEndpoint(raises=httpx.ConnectError("no route to host"))
    client = make_connector(endpoint)

    with pytest.raises(ConnectorError, match="no route to host"):
        client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)


def test_a_timed_out_reply_raises_a_connector_error():
    endpoint = ConnectorEndpoint(raises=httpx.ReadTimeout("timed out"))
    client = make_connector(endpoint)

    with pytest.raises(ConnectorError, match="timed out"):
        client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)


def test_a_credential_failure_reaches_the_caller_and_posts_nothing():
    # TokenError stays its own type on purpose: it is a configuration problem,
    # not a conversation one, and no amount of re-splitting an activity fixes it.
    endpoint = ConnectorEndpoint()
    failing = make_source(TokenEndpoint(status=401, body={"error": "invalid_client"}))
    client = make_connector(endpoint, source=failing)

    with pytest.raises(TokenError):
        client.reply(SERVICE_URL, CHANNEL_CONVERSATION, ACTIVITY_ID, ACTIVITY)

    assert endpoint.calls == 0


def test_the_connector_client_builds_its_own_http_client_by_default():
    client = ConnectorClient(make_config(), make_source(TokenEndpoint()))

    assert isinstance(client._http, httpx.Client)
    assert client._http.timeout.read == CONNECTOR_TIMEOUT_SEC
