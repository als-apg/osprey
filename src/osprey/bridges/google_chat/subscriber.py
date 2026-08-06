"""Pub/Sub ingestion: pull Chat events off a subscription and feed them to the engine.

This is the Google Chat adapter's half of the arrival model. Chat delivers events
to a Pub/Sub topic; the bridge holds one **streaming pull** on a subscription of
that topic, decodes each message into the raw Chat event, and hands it to
:meth:`~osprey.bridges.core.BridgeRuntime.handle_event`. Everything after that —
the dedup claim, dispatch, retry parking, crash recovery — is the engine's, which
is why nothing in this module knows what an event *means*.

**One subscriber per subscription.** A second process pulling the same
subscription would split delivery between them, and each would answer a different
half of the questions from a dedup store the other cannot see. The engine's dedup
defeats *redelivery*, not *concurrent* delivery to separate stores.

Two properties here are the whole reliability story:

*  **Decoding tries plain JSON before base64**, because the client library hands
   ``message.data`` over already decoded. :func:`decode_pubsub_data` documents why
   the reverse order silently drops events.
*  **Every message is acknowledged exactly once, in a ``finally``** — after the
   handler returns, never before. An ack is the bridge saying "this event is
   settled", and it is settled once ``handle_event`` returns, whether that was an
   answer, an ignore, a duplicate, or a raise: the pipeline has already persisted
   whatever it claimed, and the drain and startup reconcile own the rest. The only
   messages Pub/Sub redelivers are therefore the ones this process died holding,
   which is precisely the set crash recovery is built to finish.

Google's libraries are imported **lazily, inside the default factory only**, so
this module imports on a machine with no Google packages installed at all, and the
tests run over a fake subscriber with none of them present.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

from osprey.bridges.core import BridgeRuntime

from .config import GoogleChatBridgeConfig

logger = logging.getLogger(__name__)

WAKE_INTERVAL = 60.0
"""Seconds the serve loop blocks before waking to supervise the drain thread.

The streaming pull runs on the library's own threads, so this interval does not
pace ingestion at all — it only bounds how long a dead drain thread stays dead,
and how long a shutdown takes to be noticed (see :func:`serve`)."""

SubscriberFactory = Callable[["GoogleChatBridgeConfig"], Any]
"""How the Pub/Sub subscriber client is obtained. The default builds a real one
from the service-account key; tests inject a fake, which is what keeps
``google-cloud-pubsub`` out of the test path entirely."""


def decode_pubsub_data(data: Any) -> dict[str, Any]:
    """Decode a Pub/Sub message ``data`` payload into the raw Chat event.

    **Plain JSON is tried FIRST and base64 only as a fallback, and that order is
    load-bearing.** The Pub/Sub client library hands ``message.data`` over as
    ALREADY-decoded raw bytes (the JSON event), so plain JSON is the live path.
    Base64 covers transports that deliver the wire encoding verbatim instead (a
    REST push body, a replayed capture). Trying base64 first is subtly wrong:
    ``b64decode`` of a raw JSON string only *sometimes* raises — it depends on the
    payload's length and characters — and when it does not raise it yields garbage
    that fails to parse, so events disappear intermittently rather than reliably.

    Args:
        data: The message payload: the event mapping itself (already decoded by a
            caller), or JSON as ``bytes``/``bytearray``/``str``, optionally
            base64-encoded.

    Returns:
        The raw Chat event mapping, exactly as sent. No field is read here.

    Raises:
        ValueError: If ``data`` is not a type this can decode, if neither
            encoding yields JSON, or if it yields something other than an object.
            Every one of those is a poison message to the caller, which drops it
            (see :func:`build_callback`).
    """
    if isinstance(data, dict):
        return data
    if not isinstance(data, bytes | bytearray | str):
        raise ValueError(f"cannot decode pubsub data of type {type(data)!r}")
    try:
        raw = data.decode("utf-8") if isinstance(data, bytes | bytearray) else data
        return _as_event(json.loads(raw))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return _as_event(json.loads(base64.b64decode(data, validate=False).decode("utf-8")))


def _as_event(decoded: Any) -> dict[str, Any]:
    """Return ``decoded`` if it is a JSON object, else raise.

    A Chat event is always an object. Anything else — a bare number, a list — is a
    payload this subscription should never have carried, and saying so here keeps
    :func:`decode_pubsub_data` honest about what it returns rather than passing a
    surprise down to the parser.
    """
    if not isinstance(decoded, dict):
        raise ValueError(f"pubsub payload is a JSON {type(decoded).__name__}, not an event object")
    return decoded


def _default_subscriber(cfg: GoogleChatBridgeConfig) -> Any:
    """Build a Pub/Sub subscriber client from the bridge's own service-account key.

    Authenticated from ``cfg.sa_key`` — the SAME key that posts to Chat — and
    **never from Application Default Credentials**. ADC would read
    ``GOOGLE_APPLICATION_CREDENTIALS`` or a GCP metadata server, neither of which
    exists on the off-GCP hosts this bridge deploys to, so the client would build
    fine and then fail on the first pull. That service account additionally needs
    ``roles/pubsub.subscriber`` on the subscription.

    No ``scopes`` are requested, deliberately: the Pub/Sub client applies its own
    default scope, and naming one here would be a second place to keep correct.
    (Contrast :mod:`osprey.bridges.google_chat.gcs`, which must ask for the bucket
    write scope explicitly.)

    Both Google imports are lazy — see the module docstring.

    Args:
        cfg: Supplies the service-account key path.

    Returns:
        A ``google.cloud.pubsub_v1.SubscriberClient``.

    Raises:
        Exception: Whatever the Google libraries raise when they are absent or the
            key is unusable. Ingestion cannot degrade, so this propagates and the
            bridge fails to start rather than running deaf.
    """
    from google.cloud import pubsub_v1
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(cfg.sa_key)
    return pubsub_v1.SubscriberClient(credentials=credentials)


def build_callback(runtime: BridgeRuntime) -> Callable[[Any], None]:
    """Bind the per-message callback the streaming pull invokes.

    The returned callable runs on the client library's own threads, one call per
    delivered message, and **never raises**: an exception escaping into the
    library would be logged by it and leave the message to time out its lease and
    be redelivered, which for a message that already failed once is an unbounded
    retry of the same failure.

    It acknowledges every message it is given, from a single ``finally`` so no
    path can miss it:

    *  an undecodable payload is logged and **dropped** — redelivering it would
       produce the identical decode failure forever, so it is poison and the ack
       is what removes it from the subscription;
    *  ``handle_event`` raising is logged and acked all the same. The pipeline
       persists its claim before doing anything that can fail, so the entry is
       already recorded and owned by the drain and by startup reconcile; a
       redelivery would only be re-suppressed by the dedup claim.

    The ack happens AFTER the handler returns, which is what holds the lease for
    the whole run rather than the first instant of it.

    Args:
        runtime: The started engine, from
            :func:`~osprey.bridges.core.runtime.start`.

    Returns:
        The callback to hand to ``subscribe(..., callback=)``.
    """

    def callback(message: Any) -> None:
        try:
            try:
                event = decode_pubsub_data(message.data)
            except Exception:
                logger.exception("failed to decode pubsub message; acking to drop it")
                return
            try:
                status = runtime.handle_event(event)
            except Exception:
                logger.exception("handle_event failed; acking anyway")
            else:
                logger.debug("pubsub message %s", status)
        finally:
            message.ack()

    return callback


def serve(
    cfg: GoogleChatBridgeConfig,
    runtime: BridgeRuntime,
    *,
    stop: threading.Event,
    subscriber_factory: SubscriberFactory = _default_subscriber,
    wake_interval: float = WAKE_INTERVAL,
) -> None:
    """Hold the streaming pull until shutdown. The adapter's blocking ingestion loop.

    Passed to :func:`~osprey.bridges.core.runtime.run_forever`, which calls it only
    once startup crash recovery has finished and the drain thread is running — so
    the first new event is accepted after the previous process's in-flight work is
    settled.

    Messages are delivered on the library's threads, so this loop does no
    ingestion work itself. It blocks on the streaming pull's future and exists to
    do three things:

    *  **wake periodically to supervise the drain** — while the stream is healthy
       ``future.result(timeout=)`` raises :exc:`TimeoutError` (the ``concurrent.futures``
       spelling is an alias of it), which is the normal path here, not an error;
    *  **notice the stream ending** — a return or a raise from ``result()`` means
       the subscription itself stopped. A raise propagates: a bridge that has gone
       deaf must fail loudly and be restarted, not sit in a loop looking alive;
    *  **stop when told to**, cancelling the future so the pull's threads are torn
       down. ``stop`` is observed at the wake boundary, so a shutdown signalled
       mid-wait takes effect within ``wake_interval``; that is safe rather than
       merely tolerable, because an ungraceful kill loses only unacked messages,
       which Pub/Sub redelivers and the dedup claim de-duplicates.

    Args:
        cfg: Bridge config; supplies the subscription and (via the default
            factory) the credentials.
        runtime: The started engine. Every decoded event goes to
            :meth:`~osprey.bridges.core.BridgeRuntime.handle_event`, and
            :meth:`~osprey.bridges.core.BridgeRuntime.supervise` is called on each
            wake.
        stop: Shutdown signal. MUST be the same event the process's signal handler
            sets and the one handed to ``run_forever``, so one ``set()`` stops
            ingestion and the drain together.
        subscriber_factory: How to obtain the subscriber client. Tests inject a
            fake, so no credentials, no network, and no Google libraries.
        wake_interval: Seconds between supervise wakes.
    """
    subscriber = subscriber_factory(cfg)
    future = subscriber.subscribe(cfg.subscription, callback=build_callback(runtime))
    logger.info("google chat bridge listening on %s", cfg.subscription)
    try:
        while not stop.is_set():
            try:
                future.result(timeout=wake_interval)
            except TimeoutError:
                runtime.supervise()
                continue
            # result() returned a value: the streaming pull ended on its own
            # without raising. Nothing is arriving any more, so stop.
            logger.error("pub/sub streaming pull ended; stopping ingestion")
            break
    finally:
        # Reached on every exit — stop event, ended stream, or an exception on its
        # way out — because leaving the pull running would keep the process's
        # threads alive and keep leasing messages this bridge will never handle.
        future.cancel()
    logger.info("google chat bridge stopped listening on %s", cfg.subscription)
