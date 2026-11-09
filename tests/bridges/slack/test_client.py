"""Tests for the Slack bridge's Web API client.

Every test drives the real :class:`SlackWebClient` over an
:class:`httpx.MockTransport` whose handler records what was asked of it and
answers from a per-test script, so the suite needs no network and no workspace.
Sleep is injected: the 429 path is observed through what it was asked to wait,
never by waiting.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from osprey.bridges.core import CoreConfig, UndeliverableError
from osprey.bridges.slack import client as client_module
from osprey.bridges.slack.client import (
    API_BASE,
    FATAL_CODES,
    MEMBERS_PAGE_MAX,
    RATE_LIMIT_RETRIES,
    REFUSAL_CODES,
    RETRY_AFTER_MAX_SEC,
    SlackApiError,
    SlackAuthError,
    SlackWebClient,
)
from osprey.bridges.slack.config import SlackBridgeConfig

BOT_TOKEN = "xoxb-1111-2222-bot-secret"
APP_TOKEN = "xapp-1-A111-3333-app-secret"  # gitleaks:allow - a fake test token


def _cfg(*, trust_env: bool = False) -> SlackBridgeConfig:
    return SlackBridgeConfig(
        bot_token=BOT_TOKEN, app_token=APP_TOKEN, core=CoreConfig(trust_env=trust_env)
    )


Reply = httpx.Response | Callable[[httpx.Request], httpx.Response]


class Endpoint:
    """A recording Slack Web API: answers each request from a FIFO script.

    When the script runs dry every further call answers ``{"ok": true}``.
    """

    def __init__(self, *replies: Reply) -> None:
        self.replies: list[Reply] = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.replies:
            return httpx.Response(200, json={"ok": True})
        reply = self.replies.pop(0)
        return reply(request) if callable(reply) else reply

    def form(self, index: int = -1) -> dict[str, str]:
        """The form-encoded parameters of one recorded request."""
        body = self.requests[index].content.decode()
        return {k: v[0] for k, v in parse_qs(body, keep_blank_values=True).items()}


def ok(**body: Any) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, **body})


def err(code: str, **extra: Any) -> httpx.Response:
    return httpx.Response(200, json={"ok": False, "error": code, **extra})


def make(
    *replies: Reply, fatal: threading.Event | None = None
) -> tuple[SlackWebClient, Endpoint, list[float]]:
    endpoint = Endpoint(*replies)
    slept: list[float] = []
    http = httpx.Client(transport=httpx.MockTransport(endpoint))
    client = SlackWebClient(_cfg(), http=http, fatal=fatal, sleep=slept.append)
    return client, endpoint, slept


# --- call: wire shape -------------------------------------------------------------


def test_call_posts_form_to_method_url_with_bot_bearer_and_returns_body() -> None:
    client, endpoint, _ = make(ok(channel="C1", ts="1.2"))

    body = client.call("chat.postMessage", channel="C1", text="hi")

    assert body == {"ok": True, "channel": "C1", "ts": "1.2"}
    (request,) = endpoint.requests
    assert request.method == "POST"
    assert str(request.url) == f"{API_BASE}/chat.postMessage"
    assert API_BASE == "https://slack.com/api"
    assert request.headers["Authorization"] == f"Bearer {BOT_TOKEN}"
    assert endpoint.form() == {"channel": "C1", "text": "hi"}


def test_token_override_replaces_the_bearer_for_that_call_only() -> None:
    client, endpoint, _ = make(ok(url="wss://x"), ok())

    client.call("apps.connections.open", token=APP_TOKEN)
    client.call("auth.test")

    assert endpoint.requests[0].headers["Authorization"] == f"Bearer {APP_TOKEN}"
    assert endpoint.requests[1].headers["Authorization"] == f"Bearer {BOT_TOKEN}"


def test_none_params_are_omitted_and_structured_values_are_json_encoded() -> None:
    client, endpoint, _ = make()

    client.call("x.y", a=None, b=True, c=False, d=3, e=["U1", "U2"], f={"k": 1})

    form = endpoint.form()
    assert "a" not in form
    assert form["b"] == "true"
    assert form["c"] == "false"
    assert form["d"] == "3"
    assert json.loads(form["e"]) == ["U1", "U2"]
    assert json.loads(form["f"]) == {"k": 1}


# --- trust_env ---------------------------------------------------------------------


@pytest.mark.parametrize("trust_env", [True, False])
def test_default_http_client_follows_core_trust_env(
    monkeypatch: pytest.MonkeyPatch, trust_env: bool
) -> None:
    seen: dict[str, Any] = {}
    real = httpx.Client

    def spy(*args: Any, **kwargs: Any) -> httpx.Client:
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(client_module.httpx, "Client", spy)
    SlackWebClient(_cfg(trust_env=trust_env))

    assert seen["trust_env"] is trust_env
    assert seen.get("timeout") is not None


# --- 429 ---------------------------------------------------------------------------


def test_429_waits_retry_after_then_retries_and_returns() -> None:
    client, endpoint, slept = make(httpx.Response(429, headers={"Retry-After": "7"}), ok(ts="9.9"))

    assert client.call("chat.postMessage", channel="C1", text="x")["ts"] == "9.9"
    assert slept == [7.0]
    assert len(endpoint.requests) == 2
    assert endpoint.form(0) == endpoint.form(1)


def test_ratelimited_error_body_is_treated_like_429() -> None:
    client, _, slept = make(err("ratelimited"), ok())

    client.call("users.info", user="U1")

    assert len(slept) == 1


def test_missing_or_garbage_retry_after_uses_a_positive_default() -> None:
    client, _, slept = make(
        httpx.Response(429),
        httpx.Response(429, headers={"Retry-After": "soon"}),
        ok(),
    )

    client.call("auth.test")

    assert len(slept) == 2
    assert all(s > 0 for s in slept)


def test_huge_retry_after_is_capped() -> None:
    client, _, slept = make(httpx.Response(429, headers={"Retry-After": "86400"}), ok())

    client.call("auth.test")

    assert slept == [RETRY_AFTER_MAX_SEC]


def test_429_retries_are_bounded_then_raise_slack_api_error() -> None:
    replies = [httpx.Response(429, headers={"Retry-After": "1"})] * (RATE_LIMIT_RETRIES + 5)
    client, endpoint, slept = make(*replies)

    with pytest.raises(SlackApiError) as info:
        client.call("chat.postMessage", channel="C1", text="x")

    assert info.value.error == "ratelimited"
    assert len(endpoint.requests) == RATE_LIMIT_RETRIES + 1
    assert len(slept) == RATE_LIMIT_RETRIES


# --- error classification ------------------------------------------------------


def test_refusal_codes_are_exactly_decision_8() -> None:
    assert REFUSAL_CODES == {
        "channel_not_found",
        "not_in_channel",
        "is_archived",
        "restricted_action",
        "team_access_not_granted",
    }


def test_fatal_codes_are_exactly_decision_6_and_exclude_missing_scope() -> None:
    assert FATAL_CODES == {"invalid_auth", "not_authed", "token_revoked", "account_inactive"}
    assert "missing_scope" not in FATAL_CODES


@pytest.mark.parametrize("code", sorted(REFUSAL_CODES))
def test_refusal_code_raises_undeliverable_and_does_not_set_fatal(code: str) -> None:
    fatal = threading.Event()
    client, endpoint, slept = make(err(code), fatal=fatal)

    with pytest.raises(UndeliverableError) as info:
        client.post_message("C1", "hello")

    assert code in str(info.value)
    assert not fatal.is_set()
    assert len(endpoint.requests) == 1
    assert slept == []


@pytest.mark.parametrize("code", sorted(FATAL_CODES))
def test_fatal_code_sets_shared_event_and_raises_auth_error(code: str) -> None:
    fatal = threading.Event()
    client, endpoint, _ = make(err(code), fatal=fatal)

    with pytest.raises(SlackAuthError) as info:
        client.auth_test()

    assert fatal.is_set()
    assert info.value.error == code
    assert code in str(info.value)
    assert len(endpoint.requests) == 1


def test_fatal_code_on_app_token_call_also_sets_fatal() -> None:
    fatal = threading.Event()
    client, _, _ = make(err("invalid_auth"), fatal=fatal)

    with pytest.raises(SlackAuthError):
        client.call("apps.connections.open", token=APP_TOKEN)

    assert fatal.is_set()


def test_auth_error_is_not_a_slack_api_error() -> None:
    """An ``except SlackApiError`` in an ops member must never swallow a dead token."""
    assert not issubclass(SlackAuthError, SlackApiError)
    assert not issubclass(SlackApiError, SlackAuthError)


def test_client_without_fatal_event_still_owns_one() -> None:
    client, _, _ = make(err("token_revoked"))

    with pytest.raises(SlackAuthError):
        client.auth_test()

    assert client.fatal.is_set()


def test_injected_fatal_event_is_exposed() -> None:
    fatal = threading.Event()
    client, _, _ = make(fatal=fatal)
    assert client.fatal is fatal


@pytest.mark.parametrize("code", ["missing_scope", "user_not_found", "msg_too_long"])
def test_other_codes_raise_slack_api_error_carrying_code(code: str) -> None:
    fatal = threading.Event()
    client, _, _ = make(err(code, needed="chat:write"), fatal=fatal)

    with pytest.raises(SlackApiError) as info:
        client.call("chat.postMessage", channel="C1", text="x")

    assert info.value.error == code
    assert info.value.method == "chat.postMessage"
    assert code in str(info.value)
    assert not fatal.is_set()


def test_non_2xx_status_raises_slack_api_error() -> None:
    client, _, _ = make(httpx.Response(503, text="upstream unavailable"))

    with pytest.raises(SlackApiError) as info:
        client.auth_test()

    assert "503" in str(info.value)


def test_non_json_body_raises_slack_api_error() -> None:
    client, _, _ = make(httpx.Response(200, text="<html>login</html>"))

    with pytest.raises(SlackApiError):
        client.auth_test()


def test_non_object_json_body_raises_slack_api_error() -> None:
    client, _, _ = make(httpx.Response(200, json=["ok"]))

    with pytest.raises(SlackApiError):
        client.auth_test()


def test_transport_error_is_wrapped_in_slack_api_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, _, _ = make(boom)

    with pytest.raises(SlackApiError) as info:
        client.auth_test()

    assert isinstance(info.value.__cause__, httpx.HTTPError)
    assert "auth.test" in str(info.value)


# --- token-free, capped messages --------------------------------------------------


def _all_errors_text(replies: list[Reply]) -> list[str]:
    texts: list[str] = []
    for reply in replies:
        client, _, _ = make(reply)
        with pytest.raises(Exception) as info:
            client.call("apps.connections.open", token=APP_TOKEN)
        texts.append(str(info.value))
        texts.append(repr(info.value))
    return texts


def test_error_messages_never_contain_a_token() -> None:
    leaky = f"echo {BOT_TOKEN} {APP_TOKEN}"

    def leaky_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(leaky, request=request)

    texts = _all_errors_text(
        [
            httpx.Response(500, text=leaky),
            httpx.Response(200, text=leaky),
            err(leaky),
            err("invalid_auth", detail=leaky),
            err("not_in_channel", detail=leaky),
            leaky_transport,
        ]
    )
    for text in texts:
        assert BOT_TOKEN not in text
        assert APP_TOKEN not in text
    assert repr(SlackWebClient(_cfg(), http=httpx.Client())).count("secret") == 0


def test_error_messages_are_capped() -> None:
    texts = _all_errors_text([httpx.Response(500, text="x" * 10_000), err("y" * 10_000)])
    for text in texts:
        assert len(text) < 600


# --- typed helpers ----------------------------------------------------------------


def test_auth_test_returns_the_body() -> None:
    client, endpoint, _ = make(ok(user_id="UBOT", bot_id="BBOT", team_id="T1"))

    body = client.auth_test()

    assert body["user_id"] == "UBOT"
    assert str(endpoint.requests[0].url).endswith("/auth.test")


def test_post_message_top_level_omits_thread_ts() -> None:
    client, endpoint, _ = make(ok(ts="1.1"))

    assert client.post_message("C1", "hello")["ts"] == "1.1"

    assert str(endpoint.requests[0].url).endswith("/chat.postMessage")
    assert endpoint.form() == {"channel": "C1", "text": "hello"}


def test_post_message_in_thread_carries_thread_ts() -> None:
    client, endpoint, _ = make(ok(ts="1.2"))

    client.post_message("C1", "hello", thread_ts="1.0")

    assert endpoint.form() == {"channel": "C1", "text": "hello", "thread_ts": "1.0"}


def test_conversations_replies_returns_messages() -> None:
    client, endpoint, _ = make(ok(messages=[{"ts": "1.0", "text": "parent"}, {"ts": "1.1"}]))

    messages = client.conversations_replies("C1", "1.0", limit=2)

    assert [m["ts"] for m in messages] == ["1.0", "1.1"]
    assert str(endpoint.requests[0].url).endswith("/conversations.replies")
    assert endpoint.form() == {"channel": "C1", "ts": "1.0", "limit": "2"}


def test_conversations_replies_tolerates_missing_or_bad_messages() -> None:
    client, _, _ = make(ok(), ok(messages=[{"ts": "1"}, "junk", 3]))

    assert client.conversations_replies("C1", "1.0") == []
    assert client.conversations_replies("C1", "1.0") == [{"ts": "1"}]


def _page(members: list[str], cursor: str = "") -> httpx.Response:
    return ok(members=members, response_metadata={"next_cursor": cursor})


def test_conversations_members_reads_every_page_without_limit() -> None:
    client, endpoint, _ = make(_page(["U1", "U2"], "c2"), _page(["U3"], ""))

    members, more = client.conversations_members("C1", limit=None)

    assert members == ["U1", "U2", "U3"]
    assert more is False
    assert endpoint.form(0) == {"channel": "C1", "limit": str(MEMBERS_PAGE_MAX)}
    assert endpoint.form(1)["cursor"] == "c2"
    assert all(str(r.url).endswith("/conversations.members") for r in endpoint.requests)


def test_conversations_members_stops_at_limit_and_reports_more() -> None:
    client, endpoint, _ = make(_page(["U1", "U2", "U3"], "c2"))

    members, more = client.conversations_members("C1", limit=2)

    assert members == ["U1", "U2"]
    assert more is True
    assert len(endpoint.requests) == 1
    assert endpoint.form(0)["limit"] == "2"


def test_conversations_members_limit_exact_on_last_page_is_not_more() -> None:
    client, _, _ = make(_page(["U1", "U2"], ""))

    assert client.conversations_members("C1", limit=2) == (["U1", "U2"], False)


def test_conversations_members_limit_exact_with_cursor_left_is_more() -> None:
    client, _, _ = make(_page(["U1", "U2"], "c2"))

    assert client.conversations_members("C1", limit=2) == (["U1", "U2"], True)


def test_conversations_members_page_size_is_clamped() -> None:
    client, endpoint, _ = make(_page(["U1"]))

    client.conversations_members("C1", limit=5000)

    assert endpoint.form()["limit"] == str(MEMBERS_PAGE_MAX)


def test_conversations_members_stops_on_repeated_cursor() -> None:
    client, endpoint, _ = make(_page(["U1"], "loop"), _page(["U2"], "loop"), _page(["U3"], "loop"))

    members, more = client.conversations_members("C1", limit=None)

    assert members == ["U1", "U2"]
    assert more is False
    assert len(endpoint.requests) == 2


def test_conversations_members_skips_non_string_ids() -> None:
    client, _, _ = make(ok(members=["U1", 7, None, "U2"]))

    assert client.conversations_members("C1", limit=None) == (["U1", "U2"], False)


def test_conversations_members_refusal_propagates() -> None:
    client, _, _ = make(_page(["U1"], "c2"), err("channel_not_found"))

    with pytest.raises(UndeliverableError):
        client.conversations_members("C1", limit=None)


def test_users_info_returns_the_user_object() -> None:
    client, endpoint, _ = make(ok(user={"id": "U1", "name": "ada", "is_bot": False}))

    assert client.users_info("U1") == {"id": "U1", "name": "ada", "is_bot": False}
    assert str(endpoint.requests[0].url).endswith("/users.info")
    assert endpoint.form() == {"user": "U1"}


def test_users_info_without_user_object_raises() -> None:
    client, _, _ = make(ok())

    with pytest.raises(SlackApiError):
        client.users_info("U1")


def test_users_info_not_found_is_slack_api_error() -> None:
    client, _, _ = make(err("user_not_found"))

    with pytest.raises(SlackApiError) as info:
        client.users_info("U404")

    assert info.value.error == "user_not_found"
