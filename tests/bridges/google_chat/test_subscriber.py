"""Pub/Sub ingestion: decoding, the ack contract, and the serve loop.

Everything runs over a hand-written subscriber, future, message and runtime. No
Google library is imported at collection or at run time — the lazy imports in the
default subscriber factory are reached by exactly two tests, which supply fake
modules for them, so the suite passes in a dev environment that has never
installed ``google-cloud-pubsub``.

Three properties are worth stating outright, because a regression in any of them
is silent in production:

*  **plain JSON is decoded before base64 is attempted.** A raw JSON payload can
   itself be accepted by ``b64decode`` and yield garbage, so the reverse order
   loses events depending on their length — pinned by
   :func:`test_a_json_payload_that_base64_would_also_accept_is_decoded_as_json`,
   which first demonstrates that the hazard is real for its payload.
*  **every message is acked exactly once**, on every path including a raising
   handler and an undecodable payload, and never before the handler returns.
*  **the subscriber is built from the service-account key**, never from
   Application Default Credentials.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import sys
import threading
import types
from collections.abc import Callable
from typing import Any

import pytest

import osprey.bridges.google_chat.subscriber as subscriber_module
from osprey.bridges.core import CoreConfig
from osprey.bridges.google_chat.config import GoogleChatBridgeConfig
from osprey.bridges.google_chat.subscriber import (
    WAKE_INTERVAL,
    _default_subscriber,
    build_callback,
    decode_pubsub_data,
    serve,
)

SUBSCRIPTION = "projects/proj/subscriptions/gchat-events"
SA_KEY = "/nonexistent/sa.json"
TEST_WAKE = 0.01

EVENT: dict[str, Any] = {
    "type": "MESSAGE",
    "message": {"name": "spaces/AAA/messages/1", "text": "@osprey status?"},
    "space": {"name": "spaces/AAA", "type": "ROOM"},
}


@pytest.fixture
def cfg() -> GoogleChatBridgeConfig:
    return GoogleChatBridgeConfig(
        sa_key=SA_KEY,
        subscription=SUBSCRIPTION,
        app_id="users/999",
        core=CoreConfig(),
    )


# --- the stand-ins --------------------------------------------------------


class FakeMessage:
    """One delivered Pub/Sub message. Counts its acks, so a double-ack is visible."""

    def __init__(self, data: Any):
        self.data = data
        self.acks = 0

    def ack(self) -> None:
        self.acks += 1


class UnreadableMessage:
    """A message whose payload cannot even be read off it."""

    def __init__(self) -> None:
        self.acks = 0

    @property
    def data(self) -> Any:
        raise AttributeError("no data on this message")

    def ack(self) -> None:
        self.acks += 1


class SpyBase64:
    """The :mod:`base64` module, counting decode attempts.

    Rebound onto the subscriber module (never onto :mod:`base64` itself) so the
    decoder's internal ORDER is observable without any other code in the process
    seeing a patched stdlib.
    """

    def __init__(self) -> None:
        self.calls = 0

    def b64decode(self, data: Any, validate: bool = False) -> bytes:
        self.calls += 1
        return base64.b64decode(data, validate=validate)


class FakeFuture:
    """The streaming-pull future.

    ``result(timeout=)`` replays a scripted sequence of outcomes: an exception
    instance is raised, anything else is returned. Running off the end fails the
    test, so "the loop waited more times than the test scripted" surfaces as a
    failure rather than a hang.
    """

    def __init__(self, outcomes: list[Any] | None = None):
        self.outcomes = list(outcomes or [])
        self.timeouts: list[float | None] = []
        self.cancels = 0

    def result(self, timeout: float | None = None) -> Any:
        self.timeouts.append(timeout)
        if not self.outcomes:
            raise AssertionError("serve() waited more times than the test scripted")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def cancel(self) -> None:
        self.cancels += 1


class FakeSubscriber:
    """The subscriber client. Records the one ``subscribe`` call it is given."""

    def __init__(self, future: FakeFuture):
        self.future = future
        self.subscriptions: list[str] = []
        self.callbacks: list[Any] = []

    def subscribe(self, subscription: str, callback: Any) -> FakeFuture:
        self.subscriptions.append(subscription)
        self.callbacks.append(callback)
        return self.future


class FakeRuntime:
    """Stands in for :class:`~osprey.bridges.core.BridgeRuntime`.

    Only the two members the subscriber is allowed to drive exist, so a call to
    anything else fails loudly instead of being silently absorbed. The two hooks
    let a test act at the exact moment the engine is running — the only way to
    observe that the ack has NOT happened yet, or to signal shutdown from inside
    a wake.
    """

    def __init__(self, status: str = "handled", raises: BaseException | None = None):
        self.status = status
        self.raises = raises
        self.events: list[Any] = []
        self.supervised = 0
        self.on_handle: Callable[[], None] | None = None
        self.on_supervise: Callable[[], None] | None = None

    def handle_event(self, event: Any) -> str:
        self.events.append(event)
        if self.on_handle is not None:
            self.on_handle()
        if self.raises is not None:
            raise self.raises
        return self.status

    def supervise(self) -> None:
        self.supervised += 1
        if self.on_supervise is not None:
            self.on_supervise()


def run_serve(
    cfg: GoogleChatBridgeConfig,
    runtime: FakeRuntime,
    subscriber: FakeSubscriber,
    *,
    stop: threading.Event | None = None,
) -> None:
    """Call :func:`serve` over the fakes, with a wake interval that never waits."""
    serve(
        cfg,
        runtime,
        stop=stop if stop is not None else threading.Event(),
        subscriber_factory=lambda _cfg: subscriber,
        wake_interval=TEST_WAKE,
    )


def stop_after(stop: threading.Event, wakes: int) -> Callable[[], None]:
    """A supervise hook that sets ``stop`` on the ``wakes``-th wake, as a signal
    handler would — from outside the blocking wait, mid-loop."""
    seen = 0

    def hook() -> None:
        nonlocal seen
        seen += 1
        if seen >= wakes:
            stop.set()

    return hook


# --- decoding -------------------------------------------------------------


def test_a_payload_that_is_already_a_mapping_is_returned_unchanged() -> None:
    assert decode_pubsub_data(EVENT) is EVENT


def test_plain_json_bytes_decode_to_the_event() -> None:
    assert decode_pubsub_data(json.dumps(EVENT).encode()) == EVENT


def test_a_plain_json_string_decodes_to_the_event() -> None:
    assert decode_pubsub_data(json.dumps(EVENT)) == EVENT


def test_a_base64_payload_falls_back_to_the_base64_decoder() -> None:
    encoded = base64.b64encode(json.dumps(EVENT).encode())
    assert decode_pubsub_data(encoded) == EVENT
    assert decode_pubsub_data(encoded.decode()) == EVENT


def test_a_plain_json_payload_is_decoded_without_base64_being_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The order pin: JSON is tried FIRST, so base64 never sees the live payload.

    Asserted as "the base64 decoder was not called" rather than by comparing
    results, because the two orders agree on their *output* for almost every
    input — which is exactly what let this bug survive in the first place. The
    hazard: ``b64decode(validate=False)`` discards every character outside the
    base64 alphabet, so a JSON payload whose remaining characters happen to
    number a multiple of four is accepted and yields garbage. Whether that
    garbage then fails loudly or is quietly mistaken for an event depends on the
    bytes, so a base64-first decoder loses events intermittently, in proportion
    to how the payload happens to be shaped.

    ``base64`` is rebound on the module rather than patched globally, so nothing
    else in the process sees the spy.
    """
    payload = '{"type": "MESSAGE", "space": {"name": "spaces/A"}}'
    # The hazard is real for this payload rather than hypothetical: base64
    # accepts it, and what it returns is not the event.
    assert base64.b64decode(payload, validate=False) != payload.encode()

    spy = SpyBase64()
    monkeypatch.setattr(subscriber_module, "base64", spy)

    assert decode_pubsub_data(payload) == json.loads(payload)
    assert spy.calls == 0


def test_a_base64_payload_does_reach_the_base64_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the order pin: the fallback is still wired up, so the
    test above cannot be satisfied by deleting base64 support altogether."""
    spy = SpyBase64()
    monkeypatch.setattr(subscriber_module, "base64", spy)

    assert decode_pubsub_data(base64.b64encode(json.dumps(EVENT).encode())) == EVENT
    assert spy.calls == 1


def test_a_payload_that_is_neither_bytes_nor_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot decode pubsub data"):
        decode_pubsub_data(17)


def test_a_payload_that_is_neither_json_nor_base64_is_rejected() -> None:
    with pytest.raises(ValueError):
        decode_pubsub_data("not json, and not base64 either !!!")


def test_a_json_payload_that_is_not_an_event_object_is_rejected() -> None:
    with pytest.raises(ValueError, match="not an event object"):
        decode_pubsub_data('["a", "list"]')


# --- the callback's ack contract -----------------------------------------


def test_a_handled_message_is_acked_once() -> None:
    runtime = FakeRuntime()
    message = FakeMessage(json.dumps(EVENT).encode())

    build_callback(runtime)(message)

    assert runtime.events == [EVENT]
    assert message.acks == 1


def test_the_ack_happens_only_after_the_handler_returns() -> None:
    """The lease is held for the whole run, not just for its first instant."""
    runtime = FakeRuntime()
    message = FakeMessage(json.dumps(EVENT).encode())
    acks_during_handling: list[int] = []
    runtime.on_handle = lambda: acks_during_handling.append(message.acks)

    build_callback(runtime)(message)

    assert acks_during_handling == [0]
    assert message.acks == 1


def test_a_raising_handler_still_acks_and_does_not_raise_into_the_library() -> None:
    """An exception escaping into the client library would leave the message to
    time out its lease and be redelivered — an unbounded retry of a failure that
    the engine has already persisted and owns."""
    runtime = FakeRuntime(raises=RuntimeError("dispatch exploded"))
    message = FakeMessage(json.dumps(EVENT).encode())

    build_callback(runtime)(message)

    assert runtime.events == [EVENT]
    assert message.acks == 1


def test_an_undecodable_message_is_acked_and_never_reaches_the_handler() -> None:
    """Poison drop: redelivering it would reproduce the identical failure forever."""
    runtime = FakeRuntime()
    message = FakeMessage(b"\xff\xfe not json and not base64 \xff")

    build_callback(runtime)(message)

    assert runtime.events == []
    assert message.acks == 1


def test_a_message_whose_payload_cannot_even_be_read_is_acked() -> None:
    """Nothing the library hands over can escape the callback as an exception."""
    runtime = FakeRuntime()
    message = UnreadableMessage()

    build_callback(runtime)(message)

    assert runtime.events == []
    assert message.acks == 1


@pytest.mark.parametrize("status", ["handled", "ignored", "duplicate"])
def test_every_engine_outcome_is_acked(status: str) -> None:
    runtime = FakeRuntime(status=status)
    message = FakeMessage(json.dumps(EVENT).encode())

    build_callback(runtime)(message)

    assert message.acks == 1


# --- the serve loop -------------------------------------------------------


def test_the_configured_subscription_is_subscribed_with_the_engine_callback(
    cfg: GoogleChatBridgeConfig,
) -> None:
    runtime = FakeRuntime()
    stop = threading.Event()
    stop.set()
    subscriber = FakeSubscriber(FakeFuture())

    run_serve(cfg, runtime, subscriber, stop=stop)

    assert subscriber.subscriptions == [SUBSCRIPTION]
    # The callback handed to the library is the engine-bound one, not a stub:
    # driving it has to reach handle_event.
    message = FakeMessage(json.dumps(EVENT).encode())
    subscriber.callbacks[0](message)
    assert runtime.events == [EVENT]
    assert message.acks == 1


def test_a_stop_event_set_before_serving_never_waits_on_the_future(
    cfg: GoogleChatBridgeConfig,
) -> None:
    stop = threading.Event()
    stop.set()
    future = FakeFuture()

    run_serve(cfg, FakeRuntime(), FakeSubscriber(future), stop=stop)

    assert future.timeouts == []
    assert future.cancels == 1


def test_each_periodic_wake_supervises_the_drain_thread(cfg: GoogleChatBridgeConfig) -> None:
    """A timeout is the healthy path here, not an error: it is the only thing
    that notices a drain thread which died, and the interval is what bounds how
    long queued work can silently stall."""
    stop = threading.Event()
    runtime = FakeRuntime()
    runtime.on_supervise = stop_after(stop, 3)
    future = FakeFuture([concurrent.futures.TimeoutError() for _ in range(3)])

    run_serve(cfg, runtime, FakeSubscriber(future), stop=stop)

    assert runtime.supervised == 3
    assert future.timeouts == [TEST_WAKE] * 3
    assert future.cancels == 1


def test_a_shutdown_signalled_mid_loop_stops_and_cancels_the_future(
    cfg: GoogleChatBridgeConfig,
) -> None:
    stop = threading.Event()
    runtime = FakeRuntime()
    runtime.on_supervise = stop_after(stop, 1)
    future = FakeFuture([concurrent.futures.TimeoutError()])

    run_serve(cfg, runtime, FakeSubscriber(future), stop=stop)

    assert runtime.supervised == 1
    assert future.cancels == 1


def test_a_stream_that_ends_on_its_own_stops_ingestion(cfg: GoogleChatBridgeConfig) -> None:
    """``result()`` returning means the subscription itself ended. Nothing is
    arriving any more, so the loop must stop rather than spin on a dead pull."""
    runtime = FakeRuntime()
    future = FakeFuture([None])

    run_serve(cfg, runtime, FakeSubscriber(future))

    assert future.timeouts == [TEST_WAKE]
    assert runtime.supervised == 0
    assert future.cancels == 1


def test_a_stream_error_propagates_after_the_future_is_cancelled(
    cfg: GoogleChatBridgeConfig,
) -> None:
    """A bridge that has gone deaf must fail loudly and be restarted, not sit in
    a loop looking alive — and it must not leave the pull running on the way out."""
    future = FakeFuture([RuntimeError("stream died")])

    with pytest.raises(RuntimeError, match="stream died"):
        run_serve(cfg, FakeRuntime(), FakeSubscriber(future))

    assert future.cancels == 1


def test_the_wake_interval_defaults_to_the_documented_sixty_seconds(
    cfg: GoogleChatBridgeConfig,
) -> None:
    stop = threading.Event()
    runtime = FakeRuntime()
    runtime.on_supervise = stop_after(stop, 1)
    future = FakeFuture([concurrent.futures.TimeoutError()])

    serve(cfg, runtime, stop=stop, subscriber_factory=lambda _cfg: FakeSubscriber(future))

    assert WAKE_INTERVAL == 60.0
    assert future.timeouts == [60.0]


# --- the default subscriber factory --------------------------------------


class FakeCredentials:
    """What ``from_service_account_file`` returns, so identity can be asserted."""


class FakeServiceAccountCredentials:
    """Stands in for ``google.oauth2.service_account.Credentials``."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.credentials = FakeCredentials()

    def from_service_account_file(self, *args: Any, **kwargs: Any) -> FakeCredentials:
        self.calls.append((args, kwargs))
        return self.credentials


class FakeSubscriberClient:
    """Stands in for ``google.cloud.pubsub_v1.SubscriberClient``, recording exactly
    how it was constructed — the point of the two tests below."""

    instances: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any):
        FakeSubscriberClient.instances.append((args, kwargs))


@pytest.fixture
def fake_google(monkeypatch: pytest.MonkeyPatch) -> FakeServiceAccountCredentials:
    """Install fake ``google.cloud.pubsub_v1`` and ``google.oauth2.service_account``.

    Every module along both import paths is faked, so these tests neither depend
    on a Google package being installed nor touch a real one that is.
    ``monkeypatch`` restores ``sys.modules`` afterwards.
    """
    credentials = FakeServiceAccountCredentials()
    FakeSubscriberClient.instances = []

    pubsub_v1 = types.ModuleType("google.cloud.pubsub_v1")
    pubsub_v1.SubscriberClient = FakeSubscriberClient
    service_account = types.ModuleType("google.oauth2.service_account")
    service_account.Credentials = credentials

    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    oauth2 = types.ModuleType("google.oauth2")
    google.cloud = cloud
    google.oauth2 = oauth2
    cloud.pubsub_v1 = pubsub_v1
    oauth2.service_account = service_account

    for name, module in (
        ("google", google),
        ("google.cloud", cloud),
        ("google.cloud.pubsub_v1", pubsub_v1),
        ("google.oauth2", oauth2),
        ("google.oauth2.service_account", service_account),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return credentials


def test_the_default_factory_authenticates_from_the_configured_service_account_key(
    cfg: GoogleChatBridgeConfig, fake_google: FakeServiceAccountCredentials
) -> None:
    _default_subscriber(cfg)

    assert fake_google.calls == [((SA_KEY,), {})]


def test_the_default_factory_never_falls_back_to_application_default_credentials(
    cfg: GoogleChatBridgeConfig, fake_google: FakeServiceAccountCredentials
) -> None:
    """The construction pin.

    The deploy hosts are off-GCP: no metadata server, no
    ``GOOGLE_APPLICATION_CREDENTIALS``. A client built without explicit
    credentials would construct happily and then fail on the first pull, so the
    subscriber must be handed the service-account credentials and nothing else —
    no scopes either, which the Pub/Sub client supplies itself.
    """
    _default_subscriber(cfg)

    assert FakeSubscriberClient.instances == [((), {"credentials": fake_google.credentials})]


def test_no_google_name_is_bound_at_module_import(
    cfg: GoogleChatBridgeConfig, fake_google: FakeServiceAccountCredentials
) -> None:
    """The lazy-import seam, stated as an assertion rather than left to the fact
    that this file imported at all: the module must carry no Google binding, so
    importing it costs nothing on a host without the libraries — and the fakes
    prove the imports really do happen, inside the factory, when it runs."""
    import osprey.bridges.google_chat.subscriber as module

    assert not [name for name in vars(module) if name.startswith("google")]

    _default_subscriber(cfg)

    assert FakeSubscriberClient.instances  # the lazy imports resolved when called
    assert not [name for name in vars(module) if name.startswith("google")]
