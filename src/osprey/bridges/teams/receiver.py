"""Service Bus ingestion: pull bot activities off a queue and feed them to the engine.

This is the Teams adapter's half of the arrival model. The relay function
validates each activity Teams delivers and enqueues its body verbatim on a
Service Bus queue; the bridge holds one **peek-lock receiver** on that queue,
decodes each message into the raw activity, and hands it to
:meth:`~osprey.bridges.core.BridgeRuntime.handle_event`. Everything after that —
the dedup claim, dispatch, retry parking, crash recovery — is the engine's, which
is why nothing in this module knows what an activity *means*.

**One bridge per queue.** A second process pulling the same queue would split
delivery between them, and each would answer a different half of the questions
from a dedup store the other cannot see. The engine's dedup defeats
*redelivery*, not *concurrent* delivery to separate stores.

Unlike Pub/Sub's streaming pull, Service Bus is a **pull** model with a
single-threaded client, and that shapes everything here:

*  **One lock around every receiver call** — the pull, the lock-renewal
   registration and both settlements. The SDK's receiver is not thread-safe,
   and settlements happen on handler threads while the loop thread is pulling.
*  **Every pull is gated on a free handler slot, never on a count.** A
   :class:`threading.Semaphore` of :data:`POOL_SIZE` is acquired *before* each
   pull and released when the handler settles, so the receiver is never asked
   for a message nobody can take. A message pulled with no handler free would sit
   locked until its lock expired, then be redelivered — a self-inflicted delay
   every time the pool is busy.
*  **Every message is settled exactly once, in a ``finally``** — after the
   handler returns, never before. Completion is the bridge saying "this activity
   is settled", and it is settled once ``handle_event`` returns, whether that
   was an answer, an ignore, a duplicate, or a raise: the pipeline has already
   persisted whatever it claimed, and the drain and startup reconcile own the
   rest. A lost message lock at settlement is logged, not raised: the broker
   redelivers, and the engine's claim de-duplicates the redelivery.
*  **Undecodable bodies are dead-lettered**, not completed and not dropped:
   redelivering one would produce the identical decode failure forever, and the
   dead-letter queue is where an operator can see what the relay let through.

``azure.servicebus`` is imported nowhere at module level: the one import in the
whole package is function-local, inside :func:`make_receiver` at the foot of
this module. The loop above it never names an SDK type — the receiver arrives
through the :class:`QueueReceiver` seam, and the two SDK exception classes the
settlement path recognises are matched by *name* — so this module imports on a
machine with no Azure packages installed and its tests run over a fake receiver
with none of them present.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Protocol, runtime_checkable

from osprey.bridges.core import BridgeRuntime

from .config import TeamsBridgeConfig

logger = logging.getLogger(__name__)

POOL_SIZE = 4
"""Handler threads, and therefore the most messages ever locked at once.

The same number sizes the slot semaphore that gates every pull, so it bounds the
messages held under peek-lock — not merely the ones being handled."""

SETTLE_MARGIN_SEC = 120
"""Seconds of lock renewal kept beyond the engine's poll budget.

A handler can be holding a message for the whole dispatch wait plus the reply
posts; renewing the lock for ``poll_budget + SETTLE_MARGIN_SEC`` keeps the
settlement inside the lock rather than racing its expiry."""

WAKE_INTERVAL = 60.0
"""Seconds between wakes of the serve loop for :meth:`~osprey.bridges.core.BridgeRuntime.supervise`.

Ingestion is not paced by this: it bounds how long a dead drain thread stays
dead (see :func:`serve`)."""

RECEIVE_WAIT_SEC = 5.0
"""Seconds one pull waits for a message before returning an empty batch.

Every pull holds the receiver lock for up to this long, so it also bounds how
long a handler's settlement can queue behind an idle pull, and how long a
shutdown signalled mid-pull takes to be noticed."""

SLOT_TIMEOUT_SEC = 1.0
"""Seconds the loop waits for a free handler slot before re-checking ``stop``.

With every slot taken there is nothing to pull; this is how often the loop
looks up from that wait to notice a shutdown or a wake that is due."""

_SETTLEMENT_ERROR_NAMES = frozenset({"ServiceBusError", "MessageLockLostError"})
"""The SDK exception classes a failed settlement is *expected* to raise.

Recognised by walking the exception's MRO for these names, so the settlement
path imports nothing from the SDK. ``MessageLockLostError`` subclasses ``ServiceBusError`` in the
SDK, and so do the other settlement-time errors (``MessageAlreadySettled``,
``SessionLockLostError``), which is why the base name is in the set."""


@runtime_checkable
class QueueReceiver(Protocol):
    """What the serve loop needs from a Service Bus receiver.

    The real implementation wraps a peek-lock ``ServiceBusReceiver`` and an
    ``AutoLockRenewer``; tests and the end-to-end lane supply fakes. Every
    method is called under the loop's one lock, never concurrently, so an
    implementation need not be thread-safe.

    Messages are opaque to the loop except for their body (see
    :func:`decode_body`): the object ``receive`` returns is the object handed
    back to ``register``, ``complete`` and ``dead_letter``.
    """

    def receive(self, max_messages: int, max_wait: float) -> list[Any]:
        """Pull up to ``max_messages`` messages, waiting at most ``max_wait`` seconds.

        Returns the empty list when nothing arrived in time. ``max_messages`` is
        always at least 1. Raising means the receiver has failed; the loop
        propagates it.
        """
        ...

    def complete(self, msg: Any) -> None:
        """Settle ``msg`` as done. Called after the handler returns, from a ``finally``."""
        ...

    def dead_letter(self, msg: Any, reason: str) -> None:
        """Move ``msg`` to the dead-letter queue with ``reason``. Called for poison bodies."""
        ...

    def register(self, msg: Any) -> None:
        """Start lock renewal for ``msg``. Called before the handler is submitted."""
        ...


def decode_body(msg: Any) -> dict[str, Any]:
    """Decode a received message's body into the raw bot activity.

    The SDK exposes a data body as an iterator of byte sections; the relay
    enqueues exactly one JSON document per message, so the sections are joined
    and parsed as one. Fakes and replays may carry the body as a single
    ``bytes``/``str``, or as the already-decoded mapping.

    Args:
        msg: A received message with a ``body`` attribute.

    Returns:
        The raw activity mapping, exactly as sent. No field is read here.

    Raises:
        ValueError: If the message has no readable body, the body is not a
            type this can decode, it is not valid UTF-8 JSON, or it decodes to
            something other than an object. Every one of those is a poison
            message to the caller, which dead-letters it (see :func:`serve`).
    """
    try:
        body = msg.body
    except Exception as exc:
        raise ValueError(f"cannot read message body: {exc}") from exc
    if isinstance(body, dict):
        return body
    if isinstance(body, bytes | bytearray | str):
        raw: bytes | str = bytes(body) if isinstance(body, bytearray) else body
    elif isinstance(body, Iterable):
        raw = _join_sections(body)
    else:
        raise ValueError(f"cannot decode message body of type {type(body)!r}")
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"message body is not JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"message body is a JSON {type(decoded).__name__}, not an activity object")
    return decoded


def _join_sections(sections: Iterable[Any]) -> bytes:
    """Join a body delivered as byte (or text) sections into one payload."""
    parts: list[bytes] = []
    for section in sections:
        if isinstance(section, bytes | bytearray):
            parts.append(bytes(section))
        elif isinstance(section, str):
            parts.append(section.encode("utf-8"))
        else:
            raise ValueError(f"message body section of type {type(section)!r}")
    return b"".join(parts)


def _is_settlement_error(exc: BaseException) -> bool:
    """Whether ``exc`` is one of the SDK's settlement errors, matched by class name."""
    return any(cls.__name__ in _SETTLEMENT_ERROR_NAMES for cls in type(exc).__mro__)


def _settle(lock: threading.Lock, settle: Callable[[], None], what: str) -> None:
    """Run one settlement under ``lock``, logging rather than raising on failure.

    Completion and dead-lettering share this one failure policy. A settlement
    runs where nothing would see a raise — a handler thread for a completion,
    the pull loop for a poison message — and nothing is owed on failure either
    way: the broker redelivers a message whose lock was lost, and the redelivery
    is suppressed by the engine's dedup claim (a handled message) or
    dead-lettered again by the next pull (a poison one). So a lost lock is only
    a warning, while anything else is logged with its traceback, since a
    settlement that fails for a reason the SDK does not name is a bug worth
    seeing.

    Args:
        lock: The receiver lock; every receiver call is made under it.
        settle: The receiver call to make.
        what: What was attempted, as a gerund phrase, for the log line.
    """
    try:
        with lock:
            settle()
    except Exception as exc:
        if _is_settlement_error(exc):
            logger.warning(
                "%s failed (%s: %s); the broker will redeliver it",
                what,
                type(exc).__name__,
                exc,
            )
        else:
            logger.exception("%s failed", what)


def serve(
    receiver: QueueReceiver,
    runtime: BridgeRuntime,
    *,
    stop: threading.Event,
    wake_interval: float = WAKE_INTERVAL,
    slot_timeout: float = SLOT_TIMEOUT_SEC,
    clock: Callable[[], float] = time.monotonic,
    lock: threading.Lock | None = None,
) -> None:
    """Pull messages until shutdown. The adapter's blocking ingestion loop.

    Passed to :func:`~osprey.bridges.core.runtime.run_forever`, which calls it
    only once startup crash recovery has finished and the drain thread is
    running — so the first new activity is accepted after the previous process's
    in-flight work is settled.

    Each pass of the loop: wake the drain supervisor if an interval has
    elapsed; wait up to ``slot_timeout`` for a free handler slot, going round
    again on a miss so ``stop`` is re-checked; pull one message under the lock;
    give the slot back on an empty batch; otherwise decode the body
    (dead-lettering a poison message and giving the slot back), register the
    message for lock renewal, and submit it to the handler pool. The handler
    calls :meth:`~osprey.bridges.core.BridgeRuntime.handle_event` and, from a
    ``finally``, completes the message under the lock and releases its slot.

    A raise from the receiver's pull or registration propagates: a bridge that
    has gone deaf must fail loudly and be restarted, not sit in a loop looking
    alive. A raise from a settlement never propagates — it happens on a handler
    thread, where nothing would see it — and a lost message lock in particular
    is only a warning, because the broker redelivers and the engine's claim
    de-duplicates. On every exit the pool is shut down *waiting* for in-flight
    handlers, so their settlements happen before the receiver is closed by the
    caller.

    Args:
        receiver: The queue receiver; every call on it is made under ``lock``.
        runtime: The started engine. Every decoded activity goes to
            :meth:`~osprey.bridges.core.BridgeRuntime.handle_event`, and
            :meth:`~osprey.bridges.core.BridgeRuntime.supervise` is called on
            each wake.
        stop: Shutdown signal. MUST be the same event the process's signal
            handler sets and the one handed to ``run_forever``, so one ``set()``
            stops ingestion and the drain together.
        wake_interval: Seconds between supervise wakes.
        slot_timeout: Seconds to wait for a free slot before re-checking ``stop``.
        clock: Monotonic clock the wake schedule is read from. Tests inject one
            they advance.
        lock: The receiver lock. Tests inject one to assert it is held on
            every receiver call; production leaves it to the loop.
    """
    receiver_lock = lock if lock is not None else threading.Lock()
    slots = threading.Semaphore(POOL_SIZE)
    pool = ThreadPoolExecutor(max_workers=POOL_SIZE, thread_name_prefix="teams-handler")

    def handle(msg: Any, event: dict[str, Any]) -> None:
        try:
            status = runtime.handle_event(event)
        except Exception:
            logger.exception("handle_event failed; completing the message anyway")
        else:
            logger.debug("service bus message %s", status)
        finally:
            try:
                _settle(
                    receiver_lock, partial(receiver.complete, msg), "completing a handled message"
                )
            finally:
                slots.release()

    logger.info("teams bridge pulling from the service bus queue")
    next_wake = clock() + wake_interval
    try:
        while not stop.is_set():
            now = clock()
            if now >= next_wake:
                runtime.supervise()
                next_wake = clock() + wake_interval
            if not slots.acquire(timeout=slot_timeout):
                continue
            try:
                with receiver_lock:
                    batch = receiver.receive(1, RECEIVE_WAIT_SEC)
                if not batch:
                    slots.release()
                    continue
                msg = batch[0]
                try:
                    event = decode_body(msg)
                except ValueError as exc:
                    logger.error("dead-lettering an undecodable service bus message: %s", exc)
                    _settle(
                        receiver_lock,
                        partial(receiver.dead_letter, msg, f"undecodable body: {exc}"),
                        "dead-lettering a poison message",
                    )
                    slots.release()
                    continue
                with receiver_lock:
                    receiver.register(msg)
            except BaseException:
                # The pull or the registration raised: the receiver is broken.
                # Give the slot back so the pool drains cleanly, then fail loudly.
                slots.release()
                raise
            pool.submit(handle, msg, event)
    finally:
        # Reached on every exit — stop event, a dead receiver, or an exception on
        # its way out. Waiting is what lets in-flight handlers settle their
        # messages before the caller closes the receiver under them.
        pool.shutdown(wait=True)
    logger.info("teams bridge stopped pulling from the service bus queue")


ReceiverFactory = Callable[[TeamsBridgeConfig], QueueReceiver]
"""How the process obtains its receiver, mirroring Google Chat's ``SubscriberFactory``.

The seam the entrypoint's wiring injects: :func:`make_receiver` in production, a
fake queue in the tests and in the end-to-end lane. It lives here, beside the
Protocol it produces and the default that satisfies it."""


class ServiceBusQueueReceiver:
    """The production :class:`QueueReceiver`: a peek-lock receiver and its lock renewer.

    Built by :func:`make_receiver`. Nothing in this class body names an SDK type
    — the three objects arrive already built — which is what keeps the one
    ``azure.servicebus`` import inside that single function.

    The three are kept as plain attributes rather than hidden behind the
    Protocol: they are what :meth:`close` shuts down, and reading them is how a
    test sees what the factory built without reaching into SDK internals.

    Thread safety is the loop's, not this class's: :func:`serve` makes every
    call below under its one receiver lock, which is exactly what the SDK's
    single-threaded receiver requires.
    """

    def __init__(self, client: Any, receiver: Any, renewer: Any) -> None:
        """Hold the three SDK objects the Protocol members delegate to.

        Args:
            client: The ``ServiceBusClient`` the receiver was opened from. Kept
                so :meth:`close` can shut it down — the connection lives on the
                client, not on the receiver.
            receiver: The peek-lock ``ServiceBusReceiver`` on the queue.
            renewer: The ``AutoLockRenewer`` that renews the locks of registered
                messages.
        """
        self.client = client
        self.receiver = receiver
        self.renewer = renewer

    def receive(self, max_messages: int, max_wait: float) -> list[Any]:
        """Pull up to ``max_messages`` messages, waiting at most ``max_wait`` seconds.

        Args:
            max_messages: Most messages to take; the loop asks for one.
            max_wait: Seconds to wait for the first message before giving up.

        Returns:
            The messages pulled, empty when none arrived in time. Copied into a
            list of its own so the Protocol's return type holds whatever
            sequence the SDK hands back.
        """
        return list(
            self.receiver.receive_messages(max_message_count=max_messages, max_wait_time=max_wait)
        )

    def complete(self, msg: Any) -> None:
        """Settle ``msg`` as done.

        SDK errors propagate unwrapped: :func:`serve` recognises a lost lock by
        the exception's class name, and anything of this module's own wrapped
        around one would turn a one-line warning into an unexplained traceback.
        """
        self.receiver.complete_message(msg)

    def dead_letter(self, msg: Any, reason: str) -> None:
        """Move ``msg`` to the dead-letter queue with ``reason``.

        ``reason`` goes to the broker rather than only to the log: it is all an
        operator reading the dead-letter queue has to go on. SDK errors
        propagate unwrapped, for the reason :meth:`complete` gives.
        """
        self.receiver.dead_letter_message(msg, reason=reason)

    def register(self, msg: Any) -> None:
        """Start lock renewal for ``msg``.

        The renewer renews a lock by asking the receiver that holds it, so it is
        handed both.
        """
        self.renewer.register(self.receiver, msg)

    def close(self) -> None:
        """Shut down the renewer, the receiver and the client, in that order.

        Called by the entrypoint once :func:`serve` has returned and its pool
        has settled every in-flight message. The renewer goes first: renewal
        threads still running against a closed receiver have nothing left to
        renew and something to raise about. Each close is attempted even when an
        earlier one raised, so one failure cannot leak the other two.
        """
        try:
            self.renewer.close()
        finally:
            try:
                self.receiver.close()
            finally:
                self.client.close()


def make_receiver(cfg: TeamsBridgeConfig) -> QueueReceiver:
    """Open the configured Service Bus queue in peek-lock mode.

    The default :data:`ReceiverFactory`, and the only place in the package
    ``azure.servicebus`` is imported. The import is function-local for the
    reason the module docstring gives: the package has to import on a machine
    without the ``teams`` extra, which is every dev checkout, every ``osprey
    build`` run rendering the compose template, and most of CI.

    Nothing here dials anything. The client, the receiver and the renewer are
    all lazy in the SDK, so an unreachable namespace surfaces on the first pull
    rather than at wiring time — which is also what makes a construct-only test
    possible without a broker.

    Args:
        cfg: Bridge config supplying the connection string, the queue name, and
            — through ``cfg.core.poll_budget`` — how long a held message's lock
            must keep being renewed. That the credentials are present at all is
            :meth:`TeamsBridgeConfig.require_startup`'s check, not this
            function's.

    Returns:
        A :class:`ServiceBusQueueReceiver` over a peek-lock receiver and an
        ``AutoLockRenewer`` covering ``poll_budget`` plus
        :data:`SETTLE_MARGIN_SEC` — the whole dispatch wait plus the time it
        takes to post the answer — with one renewal thread per handler and one
        to spare, so a renewal never queues behind a full pool.

    Raises:
        ImportError: If the ``teams`` extra is not installed.
        ValueError: If the connection string is not a well-formed Service Bus
            connection string (raised by the SDK).
    """
    try:
        from azure.servicebus import AutoLockRenewer, ServiceBusClient, ServiceBusReceiveMode
    except ImportError as exc:
        raise ImportError(
            "the Microsoft Teams bridge needs azure-servicebus to open its queue; "
            "install the 'teams' extra (osprey-framework[teams])"
        ) from exc

    client = ServiceBusClient.from_connection_string(cfg.servicebus_connection_string)
    receiver = client.get_queue_receiver(
        queue_name=cfg.servicebus_queue,
        receive_mode=ServiceBusReceiveMode.PEEK_LOCK,
    )
    renewer = AutoLockRenewer(
        max_lock_renewal_duration=cfg.core.poll_budget + SETTLE_MARGIN_SEC,
        max_workers=POOL_SIZE + 1,
    )
    return ServiceBusQueueReceiver(client, receiver, renewer)
