"""Smoke proofs for the Microsoft Teams bridge's hermetic fixtures.

These tests assert nothing about the bridge's behaviour — they assert that the
fixtures its e2e lane stands on actually work: each fake binds a free loopback
port or an in-process queue, answers the calls the adapter makes, and records
what it was sent. The three fakes are proved **independently**, so a failure in
the lane points at the bridge rather than at the scaffolding under it.

**Nothing here may skip.** No container runtime, no credential, no network and
nothing from ``azure``: "cannot boot" is a failure. A fixture suite that skipped
its way to green would leave the e2e lane resting on fakes nobody had run.

Three of these tests reach into product code on purpose — ``TokenSource`` for the
seam, ``decode_body`` for the body shape, ``parse_event`` for the activity
builders. Each is the fixture's contract with the module it stands in for, and
the fixture is only useful if that contract holds; asserting it here is what
keeps the e2e lane from discovering it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import httpx
import pytest

from osprey.bridges.teams.client import ACTIVITY_URL_TEMPLATE, TokenSource, token_url
from osprey.bridges.teams.config import TeamsBridgeConfig
from osprey.bridges.teams.events import parse_event
from osprey.bridges.teams.receiver import QueueReceiver, decode_body

from .teams_fakes import (
    ACCESS_TOKEN,
    APP_ID,
    CHANNEL_ID,
    SENDER_ID,
    TENANT_ID,
    TOKEN_EXPIRES_IN,
    FakeConnectorServer,
    FakeQueueReceiver,
    FakeTokenServer,
    channel_activity,
    group_chat_activity,
    personal_activity,
)

pytestmark = [pytest.mark.e2e]

TIMEOUT = 10.0
CONVERSATION = f"{CHANNEL_ID};messageid=1700000000001"
"""A channel reply's conversation id: every character a path segment can carry."""


@pytest.fixture
def token() -> Iterator[FakeTokenServer]:
    with FakeTokenServer() as server:
        yield server


@pytest.fixture
def connector() -> Iterator[FakeConnectorServer]:
    with FakeConnectorServer() as server:
        yield server


@pytest.fixture
def cfg() -> TeamsBridgeConfig:
    return TeamsBridgeConfig(
        app_id=APP_ID,
        app_secret="fake-secret",
        tenant_id=TENANT_ID,
        servicebus_connection_string="Endpoint=sb://fake/;SharedAccessKeyName=k;SharedAccessKey=v",
        servicebus_queue="teams-activities",
    )


def _post_activity(
    connector: FakeConnectorServer,
    *,
    conversation_id: str = CONVERSATION,
    reply_to: str = "1700000000002",
    body: dict[str, object] | None = None,
    auth: str = f"Bearer {ACCESS_TOKEN}",
) -> httpx.Response:
    """Post one reply the way the Connector client does — same URL template."""
    url = ACTIVITY_URL_TEMPLATE.format(
        service_url=connector.base_url, conversation_id=conversation_id, activity_id=reply_to
    )
    return httpx.post(
        url,
        json=body if body is not None else {"type": "message", "text": "an answer"},
        headers={"Authorization": auth},
        timeout=TIMEOUT,
    )


# ---------------------------------------------------------------------------
# The fake AAD token endpoint
# ---------------------------------------------------------------------------


class TestFakeTokenServer:
    def test_boots_on_a_free_loopback_port(self, token: FakeTokenServer) -> None:
        assert token.base_url.startswith("http://127.0.0.1:")
        assert token.port > 0
        with FakeTokenServer() as other:
            assert other.port != token.port

    def test_the_token_route_answers_a_bearer_and_records_the_exchange(
        self, token: FakeTokenServer
    ) -> None:
        response = httpx.post(
            f"{token.base_url}/{TENANT_ID}/oauth2/v2.0/token",
            data={
                "grant_type": "client_credentials",
                "client_id": APP_ID,
                "client_secret": "fake-secret",
                "scope": "https://api.botframework.com/.default",
            },
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["access_token"] == ACCESS_TOKEN
        assert body["expires_in"] == TOKEN_EXPIRES_IN

        recorded = token.requests
        assert len(recorded) == 1
        assert recorded[0].tenant == TENANT_ID
        assert recorded[0].grant_type == "client_credentials"
        assert recorded[0].client_id == APP_ID
        assert recorded[0].client_secret == "fake-secret"
        assert recorded[0].scope == "https://api.botframework.com/.default"

    def test_an_unknown_route_is_an_aad_error_envelope(self, token: FakeTokenServer) -> None:
        response = httpx.post(f"{token.base_url}/nope", timeout=TIMEOUT)

        assert response.status_code == 404
        assert response.json()["error"] == "invalid_request"

    def test_the_http_client_seam_redirects_the_real_aad_url_onto_the_fake(
        self, token: FakeTokenServer, cfg: TeamsBridgeConfig
    ) -> None:
        """The seam's whole contract: the product aims at AAD, the fake answers.

        ``TokenSource`` builds the URL from the config's closed cloud table, so a
        fixture that handed it a loopback URL would be testing a URL no deployment
        ever uses. This asserts the attempted URL is the real one.
        """
        with token.http_client() as http:
            source = TokenSource(cfg, http)

            assert source.token() == ACCESS_TOKEN
            # Cached: the second call mints nothing, so the fake saw one exchange.
            assert source.token() == ACCESS_TOKEN

        assert token.attempted_urls == [token_url(cfg)]
        assert token.attempted_urls[0].startswith(f"https://{cfg.login_host}/")
        assert [r.scope for r in token.requests] == [cfg.token_scope]

    def test_reset_clears_what_was_recorded(self, token: FakeTokenServer) -> None:
        httpx.post(f"{token.base_url}/{TENANT_ID}/oauth2/v2.0/token", timeout=TIMEOUT)
        assert token.requests

        token.reset()

        assert token.requests == []
        assert token.attempted_urls == []


# ---------------------------------------------------------------------------
# The fake Bot Connector
# ---------------------------------------------------------------------------


class TestFakeConnectorServer:
    def test_boots_on_a_free_loopback_port(self, connector: FakeConnectorServer) -> None:
        assert connector.base_url.startswith("http://127.0.0.1:")
        with FakeConnectorServer() as other:
            assert other.port != connector.port

    def test_a_posted_reply_is_recorded_whole(self, connector: FakeConnectorServer) -> None:
        attachments = [{"contentType": "image/png", "contentUrl": "data:image/png;base64,AAA"}]

        response = _post_activity(
            connector,
            body={
                "type": "message",
                "text": "the beam current is 500 mA",
                "attachments": attachments,
            },
        )

        assert response.status_code == 200
        assert response.json()["id"] == "posted-1"

        posted = connector.posted
        assert len(posted) == 1
        assert posted[0].conversation_id == CONVERSATION
        assert posted[0].reply_to == "1700000000002"
        assert posted[0].text == "the beam current is 500 mA"
        assert posted[0].attachments == attachments
        assert posted[0].has_attachments is True
        assert posted[0].auth == f"Bearer {ACCESS_TOKEN}"
        assert posted[0].body["type"] == "message"
        assert connector.posted_text == ["the beam current is 500 mA"]

    def test_a_post_with_no_reply_target_records_an_empty_reply_to(
        self, connector: FakeConnectorServer
    ) -> None:
        response = httpx.post(
            f"{connector.base_url}/v3/conversations/{CONVERSATION}/activities",
            json={"type": "message", "text": "no reply target"},
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        assert connector.posted[0].reply_to == ""
        assert connector.posted[0].conversation_id == CONVERSATION

    def test_an_injected_413_refuses_one_post_and_keeps_the_attempt(
        self, connector: FakeConnectorServer
    ) -> None:
        """The re-split path: the oversized attempt is visible, and only it fails."""
        connector.reject_post(2)

        first = _post_activity(connector, body={"type": "message", "text": "one"})
        refused = _post_activity(connector, body={"type": "message", "text": "far too long"})
        third = _post_activity(connector, body={"type": "message", "text": "three"})

        assert [first.status_code, refused.status_code, third.status_code] == [200, 413, 200]
        assert refused.json()["error"]["code"] == "MessageSizeTooBig"

        assert [a.status for a in connector.attempts] == [200, 413, 200]
        assert [a.number for a in connector.attempts] == [1, 2, 3]
        assert connector.posted_text == ["one", "three"]
        assert [a.accepted for a in connector.attempts] == [True, False, True]

    def test_a_rejection_applies_to_its_post_only(self, connector: FakeConnectorServer) -> None:
        connector.reject_post(1)

        _post_activity(connector, body={"type": "message", "text": "rejected"})
        _post_activity(connector, body={"type": "message", "text": "accepted"})
        _post_activity(connector, body={"type": "message", "text": "also accepted"})

        assert connector.posted_text == ["accepted", "also accepted"]

    def test_post_numbers_are_one_based(self, connector: FakeConnectorServer) -> None:
        with pytest.raises(ValueError, match="1-based"):
            connector.reject_post(0)

    def test_an_unknown_route_is_a_connector_error_envelope(
        self, connector: FakeConnectorServer
    ) -> None:
        response = httpx.post(f"{connector.base_url}/v3/nope", json={}, timeout=TIMEOUT)

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "ResourceNotFound"

    def test_wait_for_posted_returns_once_the_post_lands(
        self, connector: FakeConnectorServer
    ) -> None:
        def post_soon() -> None:
            time.sleep(0.05)
            _post_activity(connector, body={"type": "message", "text": "late"})

        thread = threading.Thread(target=post_soon, daemon=True)
        thread.start()
        try:
            assert [a.text for a in connector.wait_for_posted(1, timeout=TIMEOUT)] == ["late"]
        finally:
            thread.join(timeout=TIMEOUT)

    def test_wait_for_posted_names_what_arrived_when_it_times_out(
        self, connector: FakeConnectorServer
    ) -> None:
        _post_activity(connector, body={"type": "message", "text": "only one"})

        with pytest.raises(AssertionError, match="only one"):
            connector.wait_for_posted(2, timeout=0.2)

    def test_reset_clears_posts_and_pending_rejections(
        self, connector: FakeConnectorServer
    ) -> None:
        connector.reject_post(1)
        _post_activity(connector, body={"type": "message", "text": "before"})

        connector.reset()
        _post_activity(connector, body={"type": "message", "text": "after"})

        assert connector.posted_text == ["after"]
        assert [a.number for a in connector.attempts] == [1]


# ---------------------------------------------------------------------------
# The fake Service Bus receiver
# ---------------------------------------------------------------------------


class TestFakeQueueReceiver:
    def test_it_satisfies_the_queue_receiver_protocol(self) -> None:
        assert isinstance(FakeQueueReceiver(), QueueReceiver)

    def test_an_enqueued_activity_arrives_as_a_decodable_body(
        self, connector: FakeConnectorServer
    ) -> None:
        receiver = FakeQueueReceiver()
        sent = personal_activity("what is the beam current?", service_url=connector.base_url)
        receiver.enqueue(sent)

        batch = receiver.receive(1, 5.0)

        assert len(batch) == 1
        # The body arrives as byte sections, the shape the SDK delivers.
        assert decode_body(batch[0]) == sent
        assert receiver.receive_calls == [(1, 5.0)]

    def test_a_body_can_be_read_twice(self, connector: FakeConnectorServer) -> None:
        """A redelivered message is decoded again; a spent iterator would look empty."""
        receiver = FakeQueueReceiver()
        sent = personal_activity("read me twice", service_url=connector.base_url)
        message = receiver.enqueue(sent)

        assert decode_body(message) == decode_body(message) == sent

    def test_an_empty_pull_blocks_briefly_and_returns_nothing(self) -> None:
        receiver = FakeQueueReceiver()

        started = time.monotonic()
        batch = receiver.receive(1, 5.0)
        elapsed = time.monotonic() - started

        assert batch == []
        # Blocked on the queue rather than spinning, but nowhere near the 5 s the
        # loop asked for — which the recorded call still shows it asked for.
        assert elapsed >= receiver.idle_wait
        assert elapsed < 1.0
        assert receiver.receive_calls == [(1, 5.0)]

    def test_a_pull_returns_a_message_enqueued_from_another_thread(
        self, connector: FakeConnectorServer
    ) -> None:
        receiver = FakeQueueReceiver(idle_wait=5.0)
        sent = personal_activity("from another thread", service_url=connector.base_url)

        def enqueue_soon() -> None:
            time.sleep(0.05)
            receiver.enqueue(sent)

        thread = threading.Thread(target=enqueue_soon, daemon=True)
        thread.start()
        try:
            batch = receiver.receive(1, 5.0)
        finally:
            thread.join(timeout=TIMEOUT)

        assert [decode_body(m) for m in batch] == [sent]

    def test_one_pull_never_returns_more_than_it_was_asked_for(
        self, connector: FakeConnectorServer
    ) -> None:
        receiver = FakeQueueReceiver()
        for i in range(3):
            receiver.enqueue(personal_activity(f"q{i}", service_url=connector.base_url))

        assert len(receiver.receive(1, 5.0)) == 1
        assert len(receiver.receive(2, 5.0)) == 2
        assert receiver.pending == 0

    def test_the_call_log_keeps_the_order_of_every_settlement(
        self, connector: FakeConnectorServer
    ) -> None:
        receiver = FakeQueueReceiver()
        first = receiver.enqueue(personal_activity("one", service_url=connector.base_url))
        second = receiver.enqueue(personal_activity("two", service_url=connector.base_url))

        for message in (first, second):
            receiver.register(message)
            receiver.complete(message)

        assert receiver.calls == [
            ("register", first.id),
            ("complete", first.id),
            ("register", second.id),
            ("complete", second.id),
        ]
        assert receiver.calls_for(first) == ["register", "complete"]
        assert receiver.calls_for(second.id) == ["register", "complete"]
        assert receiver.registered == [first.id, second.id]
        assert receiver.completed == [first.id, second.id]

    def test_dead_letter_records_its_reason(self) -> None:
        receiver = FakeQueueReceiver()
        poison = receiver.enqueue(b"not json at all")

        receiver.dead_letter(poison, "undecodable body")

        assert receiver.dead_lettered == [(poison.id, "undecodable body")]
        assert receiver.calls_for(poison) == ["dead_letter"]
        assert receiver.completed == []

    def test_a_poison_body_reaches_the_decoder_undecodable(self) -> None:
        receiver = FakeQueueReceiver()
        poison = receiver.enqueue(b"not json at all")

        with pytest.raises(ValueError, match="not JSON"):
            decode_body(poison)

    def test_redeliver_hands_the_same_activity_back_under_a_new_id(
        self, connector: FakeConnectorServer
    ) -> None:
        receiver = FakeQueueReceiver()
        sent = personal_activity("asked once", service_url=connector.base_url)
        first = receiver.enqueue(sent)
        receiver.receive(1, 5.0)

        again = receiver.redeliver(sent)
        batch = receiver.receive(1, 5.0)

        assert again.id != first.id
        assert "redelivery" in again.id
        assert decode_body(batch[0]) == sent

    def test_wait_for_settled_returns_once_the_settlements_land(self) -> None:
        receiver = FakeQueueReceiver()
        message = receiver.enqueue({"type": "message"})

        def settle_soon() -> None:
            time.sleep(0.05)
            receiver.complete(message)

        thread = threading.Thread(target=settle_soon, daemon=True)
        thread.start()
        try:
            assert receiver.wait_for_settled(1, timeout=TIMEOUT) == [("complete", message.id)]
        finally:
            thread.join(timeout=TIMEOUT)

    def test_wait_for_settled_names_the_call_log_when_it_times_out(self) -> None:
        receiver = FakeQueueReceiver()
        message = receiver.enqueue({"type": "message"})
        receiver.register(message)

        with pytest.raises(AssertionError, match="register"):
            receiver.wait_for_settled(1, timeout=0.2)


# ---------------------------------------------------------------------------
# The activity builders
# ---------------------------------------------------------------------------


class TestActivityBuilders:
    """Each builder is proved against the real parser — the only opinion that counts."""

    def test_a_mentioned_channel_post_parses_into_an_event(
        self, connector: FakeConnectorServer, cfg: TeamsBridgeConfig
    ) -> None:
        raw = channel_activity("what is the beam current?", service_url=connector.base_url)

        event = parse_event(raw, cfg)

        assert event is not None
        # The bot's own mention span is addressing, not question: it is stripped.
        assert event.text == "what is the beam current?"
        assert event.sender_id == SENDER_ID
        assert event.message_id == f"{CHANNEL_ID}:{raw['id']}"
        assert raw["serviceUrl"] == connector.base_url

    def test_a_channel_post_without_a_mention_is_ignored(
        self, connector: FakeConnectorServer, cfg: TeamsBridgeConfig
    ) -> None:
        raw = channel_activity(
            "talking among ourselves", service_url=connector.base_url, mention=False
        )

        assert parse_event(raw, cfg) is None

    def test_a_channel_reply_keys_on_the_thread_root(
        self, connector: FakeConnectorServer, cfg: TeamsBridgeConfig
    ) -> None:
        root = channel_activity("the first turn", service_url=connector.base_url)
        reply = channel_activity(
            "the second turn", service_url=connector.base_url, root_id=str(root["id"])
        )

        root_event = parse_event(root, cfg)
        reply_event = parse_event(reply, cfg)

        assert root_event is not None
        assert reply_event is not None
        # A root post carries the bare channel id and a reply the thread suffix;
        # both have to normalize to one history key or turn two would not see turn one.
        assert root_event.history_key == reply_event.history_key

    def test_a_personal_message_needs_no_mention(
        self, connector: FakeConnectorServer, cfg: TeamsBridgeConfig
    ) -> None:
        raw = personal_activity("just asking", service_url=connector.base_url)

        event = parse_event(raw, cfg)

        assert event is not None
        assert event.text == "just asking"
        assert "<at>" not in event.text

    def test_a_mentioned_group_chat_message_parses_into_an_event(
        self, connector: FakeConnectorServer, cfg: TeamsBridgeConfig
    ) -> None:
        raw = group_chat_activity("anyone know the current?", service_url=connector.base_url)

        event = parse_event(raw, cfg)

        assert event is not None
        assert event.text == "anyone know the current?"

    def test_a_group_chat_message_without_a_mention_is_ignored(
        self, connector: FakeConnectorServer, cfg: TeamsBridgeConfig
    ) -> None:
        raw = group_chat_activity("chatting", service_url=connector.base_url, mention=False)

        assert parse_event(raw, cfg) is None

    def test_every_activity_carries_the_tenant_and_a_unique_id(
        self, connector: FakeConnectorServer
    ) -> None:
        one = channel_activity("one", service_url=connector.base_url)
        two = channel_activity("two", service_url=connector.base_url)

        assert one["id"] != two["id"]
        assert one["channelData"]["tenant"]["id"] == TENANT_ID
        assert one["recipient"]["id"] == f"28:{APP_ID}"
