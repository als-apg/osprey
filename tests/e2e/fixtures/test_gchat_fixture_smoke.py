"""Smoke proofs for the Google Chat bridge's hermetic fixtures.

These tests assert nothing about the bridge — they assert that the fixtures the
bridge's e2e lane stands on actually work: each fake binds a free loopback port,
answers the routes the adapter calls, and records what it was sent.

**The HTTP fakes never skip.** They need no container runtime, no credentials and
no network, so "cannot boot" is a failure, not a skip — a suite that skips its way
to green would prove nothing. Only the Pub/Sub emulator may skip, and only when
the container runtime or its image is genuinely unavailable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from .gchat_chat_fake import REPLY_FALLBACK, FakeChatApiError, FakeChatServer
from .gchat_gcs_fake import PUBLIC_HOST, FakeGcsServer
from .gchat_pubsub_emulator import (
    PUBSUB_CONTAINER,
    RESOURCE_PREFIX,
    PubSubEmulator,
    decode_pubsub_message,
)

pytestmark = [pytest.mark.e2e]

SPACE = "spaces/AAAAsmoke"
TIMEOUT = 10.0


@pytest.fixture
def chat() -> Iterator[FakeChatServer]:
    with FakeChatServer() as server:
        yield server


@pytest.fixture
def gcs() -> Iterator[FakeGcsServer]:
    with FakeGcsServer() as server:
        yield server


# ---------------------------------------------------------------------------
# The fake Chat REST server
# ---------------------------------------------------------------------------


class TestFakeChatServer:
    def test_boots_on_a_free_loopback_port(self, chat: FakeChatServer) -> None:
        assert chat.base_url.startswith("http://127.0.0.1:")
        assert chat.port > 0
        with FakeChatServer() as other:
            assert other.port != chat.port

    def test_create_records_the_posted_message(self, chat: FakeChatServer) -> None:
        response = httpx.post(
            f"{chat.base_url}/v1/{SPACE}/messages",
            json={"text": "hello room"},
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        created = response.json()
        assert created["name"].startswith(f"{SPACE}/messages/")
        assert created["thread"]["name"].startswith(f"{SPACE}/threads/")

        posted = chat.posted
        assert len(posted) == 1
        assert posted[0].text == "hello room"
        assert posted[0].space == SPACE
        assert posted[0].name == created["name"]
        assert posted[0].has_cards is False

    def test_get_and_list_round_trip(self, chat: FakeChatServer) -> None:
        first = httpx.post(
            f"{chat.base_url}/v1/{SPACE}/messages", json={"text": "one"}, timeout=TIMEOUT
        ).json()
        httpx.post(f"{chat.base_url}/v1/{SPACE}/messages", json={"text": "two"}, timeout=TIMEOUT)

        fetched = httpx.get(f"{chat.base_url}/v1/{first['name']}", timeout=TIMEOUT)
        assert fetched.status_code == 200
        assert fetched.json()["text"] == "one"

        listed = httpx.get(f"{chat.base_url}/v1/{SPACE}/messages", timeout=TIMEOUT).json()
        assert [m["text"] for m in listed["messages"]] == ["one", "two"]

        paged = httpx.get(
            f"{chat.base_url}/v1/{SPACE}/messages", params={"pageSize": 1}, timeout=TIMEOUT
        ).json()
        assert [m["text"] for m in paged["messages"]] == ["one"]

    def test_unknown_message_is_a_google_error_envelope(self, chat: FakeChatServer) -> None:
        response = httpx.get(f"{chat.base_url}/v1/{SPACE}/messages/nope", timeout=TIMEOUT)

        assert response.status_code == 404
        assert response.json()["error"]["status"] == "NOT_FOUND"

    def test_cards_ride_along_and_are_recorded(self, chat: FakeChatServer) -> None:
        cards = [{"cardId": "c1", "card": {"sections": [{"widgets": []}]}}]

        httpx.post(
            f"{chat.base_url}/v1/{SPACE}/messages",
            json={"text": "with a card", "cardsV2": cards},
            timeout=TIMEOUT,
        )

        posted = chat.posted[0]
        assert posted.has_cards is True
        assert posted.cards == cards

    def test_media_download_serves_registered_bytes(self, chat: FakeChatServer) -> None:
        resource = chat.add_media("spaces/AAAAsmoke/attachments/plot", b"\x89PNG-bytes")

        response = httpx.get(chat.media_url(resource), timeout=TIMEOUT)

        assert response.status_code == 200
        assert response.content == b"\x89PNG-bytes"
        assert response.headers["content-type"] == "image/png"
        assert chat.media_requests == [resource]

    def test_media_download_requires_alt_media(self, chat: FakeChatServer) -> None:
        chat.add_media("attachments/x", b"data")

        response = httpx.get(f"{chat.base_url}/v1/media/attachments/x", timeout=TIMEOUT)

        assert response.status_code == 400
        assert response.json()["error"]["status"] == "INVALID_ARGUMENT"

    def test_unknown_media_is_404(self, chat: FakeChatServer) -> None:
        response = httpx.get(chat.media_url("attachments/missing"), timeout=TIMEOUT)

        assert response.status_code == 404


class TestFakeChatDiscoveryStandIn:
    """The ``service()`` stand-in — what ``ChatClient``'s injected seam receives."""

    def test_create_get_and_list_over_the_discovery_shape(self, chat: FakeChatServer) -> None:
        service = chat.service()

        created = (
            service.spaces()
            .messages()
            .create(
                parent=SPACE,
                body={"text": "posted through the stand-in"},
                messageReplyOption=REPLY_FALLBACK,
            )
            .execute()
        )
        fetched = service.spaces().messages().get(name=created["name"]).execute()
        listed = service.spaces().messages().list(parent=SPACE).execute()

        assert fetched["text"] == "posted through the stand-in"
        assert [m["name"] for m in listed["messages"]] == [created["name"]]

        posted = chat.posted[0]
        assert posted.params["messageReplyOption"] == REPLY_FALLBACK
        assert posted.body["text"] == "posted through the stand-in"

    def test_thread_name_is_honored_across_posts(self, chat: FakeChatServer) -> None:
        messages = chat.service().spaces().messages()

        first = messages.create(
            parent=SPACE,
            body={"text": "turn one", "thread": {"name": f"{SPACE}/threads/known"}},
            messageReplyOption=REPLY_FALLBACK,
        ).execute()
        second = messages.create(
            parent=SPACE,
            body={"text": "turn two", "thread": {"name": first["thread"]["name"]}},
            messageReplyOption=REPLY_FALLBACK,
        ).execute()

        # The first post's thread was unknown, so the fallback minted a new one;
        # the second names that thread and therefore lands in it.
        assert second["thread"]["name"] == first["thread"]["name"]
        assert [m.thread for m in chat.posted] == [first["thread"]["name"]] * 2

    def test_unknown_thread_without_the_fallback_option_raises(self, chat: FakeChatServer) -> None:
        messages = chat.service().spaces().messages()

        with pytest.raises(Exception) as excinfo:  # HttpError, or our google-free stand-in
            messages.create(
                parent=SPACE,
                body={"text": "no fallback", "thread": {"name": "spaces/X/threads/no"}},
            ).execute()

        error = excinfo.value
        status = getattr(error, "status_code", None) or getattr(
            getattr(error, "resp", None), "status", None
        )
        assert status == 404, f"expected a 404 from the fake, got {error!r}"
        assert isinstance(error, FakeChatApiError) or type(error).__name__ == "HttpError"
        assert chat.posted == []


# ---------------------------------------------------------------------------
# The fake GCS server
# ---------------------------------------------------------------------------


class TestFakeGcsServer:
    def test_boots_on_a_free_loopback_port(self, gcs: FakeGcsServer) -> None:
        assert gcs.base_url.startswith("http://127.0.0.1:")
        with FakeGcsServer() as other:
            assert other.port != gcs.port

    def test_put_then_public_get(self, gcs: FakeGcsServer) -> None:
        url = gcs.public_url("osprey-bucket", "plots/abc.png")

        put = httpx.put(
            url, content=b"png-bytes", headers={"Content-Type": "image/png"}, timeout=TIMEOUT
        )
        got = httpx.get(url, timeout=TIMEOUT)

        assert put.status_code == 200
        assert put.json()["name"] == "plots/abc.png"
        assert got.status_code == 200
        assert got.content == b"png-bytes"
        assert got.headers["content-type"] == "image/png"
        assert [u.name for u in gcs.uploads] == ["plots/abc.png"]

    def test_json_api_media_upload(self, gcs: FakeGcsServer) -> None:
        response = httpx.post(
            f"{gcs.base_url}/upload/storage/v1/b/osprey-bucket/o",
            params={"uploadType": "media", "name": "docs/report.pdf"},
            content=b"%PDF-1.7",
            headers={"Content-Type": "application/pdf"},
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        stored = gcs.get_object("osprey-bucket", "docs/report.pdf")
        assert stored is not None
        assert stored.data == b"%PDF-1.7"
        assert stored.content_type == "application/pdf"

    def test_head_returns_headers_without_a_body(self, gcs: FakeGcsServer) -> None:
        gcs.put_object("b", "o.txt", b"body", content_type="text/plain")

        response = httpx.head(gcs.public_url("b", "o.txt"), timeout=TIMEOUT)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain"
        assert response.content == b""

    def test_missing_object_is_404(self, gcs: FakeGcsServer) -> None:
        response = httpx.get(gcs.public_url("b", "missing.png"), timeout=TIMEOUT)

        assert response.status_code == 404

    def test_local_url_maps_a_real_google_url_onto_the_fake(self, gcs: FakeGcsServer) -> None:
        gcs.put_object("osprey-bucket", "plots/x.png", b"bytes", content_type="image/png")
        product_url = f"{PUBLIC_HOST}/osprey-bucket/plots/x.png"

        local = gcs.local_url(product_url)

        assert local == f"{gcs.base_url}/osprey-bucket/plots/x.png"
        assert httpx.get(local, timeout=TIMEOUT).content == b"bytes"

    def test_client_stand_in_uploads_downloads_and_stays_local(self, gcs: FakeGcsServer) -> None:
        blob = gcs.client().bucket("osprey-bucket").blob("plots/from-client.png")

        blob.upload_from_string(b"client-bytes", content_type="image/png")
        blob.make_public()

        assert blob.public_url == gcs.public_url("osprey-bucket", "plots/from-client.png")
        assert blob.public_url.startswith(gcs.base_url)
        assert blob.download_as_bytes() == b"client-bytes"
        stored = gcs.get_object("osprey-bucket", "plots/from-client.png")
        assert stored is not None
        assert stored.content_type == "image/png"


# ---------------------------------------------------------------------------
# The Pub/Sub emulator — the ONE fixture allowed to skip (missing runtime/image)
# ---------------------------------------------------------------------------


class TestPubSubEmulator:
    def test_container_name_is_exact_and_prefixed(self) -> None:
        """No runtime needed: the naming contract the safety rules rest on."""
        assert PUBSUB_CONTAINER == f"{RESOURCE_PREFIX}-pubsub"
        assert RESOURCE_PREFIX == "osprey-e2e-gchat"

    @pytest.mark.slow
    def test_emulator_answers_and_round_trips_a_published_message(
        self, pubsub_emulator: PubSubEmulator
    ) -> None:
        emulator = pubsub_emulator
        emulator.create_topic("smoke-topic")
        emulator.create_subscription("smoke-sub", "smoke-topic")
        payload = json.dumps({"message": {"sender": {"name": "users/1"}, "text": "hi"}})

        message_id = emulator.publish("smoke-topic", payload, attributes={"source": "smoke"})
        received = emulator.pull("smoke-sub")

        assert message_id
        assert len(received) == 1
        message = received[0]["message"]
        assert message["attributes"]["source"] == "smoke"
        assert decode_pubsub_message(message)["message"]["text"] == "hi"
        assert emulator.env["PUBSUB_EMULATOR_HOST"] == emulator.host
