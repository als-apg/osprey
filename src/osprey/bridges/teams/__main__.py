"""The bridge process: ``python -m osprey.bridges.teams``.

The whole of the adapter's ``__main__``: read the environment, refuse to start on an
incomplete one, build the collaborators, and hand
:func:`~osprey.bridges.core.runtime.run_forever` an ingestion loop holding one Service
Bus receiver. Nothing here decides message policy — every ordering, dedup, retry and
recovery rule lives in :mod:`osprey.bridges.core`, and every Teams wire detail in this
package's other modules. What this module owns is the *wiring*, and four of its
decisions are load-bearing enough that getting them wrong produces a bridge that looks
healthy and answers nobody.

**One :class:`threading.Event` stops everything.** It is handed to
:func:`~osprey.bridges.teams.receiver.serve` and to ``run_forever`` as the engine's
own, so one ``set()`` from the SIGTERM handler ends the pull loop and wakes the drain
thread. Two events here would be a container that acknowledges SIGTERM in the log and
then keeps leasing messages until it is killed.

**One :class:`~osprey.bridges.teams.client.TokenSource` and one
:class:`~osprey.bridges.teams.client.ConnectorClient`, shared by every posting
member.** The token source caches a bearer behind its own lock precisely because the
drain thread posts alongside live ingestion; a second instance would mint a second
token on every expiry, and a second connector client would be a second HTTP lock,
which is the shape those locks exist to prevent. :func:`build_wiring` is the single
place both are constructed.

**The receiver is built where it is consumed, and closed when the pull ends.**
:func:`~osprey.bridges.teams.receiver.serve` takes a receiver rather than making one,
and the SDK objects behind the production receiver — the client, the peek-lock
receiver and the lock renewer — are this process's to shut down. So
:func:`serve_events` calls the factory, serves, and closes in a ``finally``, after
``serve``'s pool has waited for every in-flight handler to settle its message. The
close is guarded: ``close`` is not part of the
:class:`~osprey.bridges.teams.receiver.QueueReceiver` Protocol, and the in-process
fakes the end-to-end tier injects do not have one.

**Injected seams, and why there are exactly four.** Every collaborator that would
otherwise reach Microsoft is injectable here: the queue receiver, the AAD token leg,
the Bot Connector leg, and the HTTP client used for the worker's byte route. An
end-to-end tier supplies all four (an in-process queue, a loopback token endpoint, a
loopback connector server) and thereby exercises this module's real wiring — the same
``build_wiring`` and ``run`` production calls — with no tenant, no credentials and no
broker.

**What is deliberately *not* wired.** Unlike Google Chat, this adapter does not
override the engine's ``fetch_prior_artifact``. Teams carries images inline as
attachment bytes and stamps no ``public_url`` on a delivered descriptor, so the
worker's copy is the only copy a follow-up question could be given and the engine's
own worker-route default is already the right one. A replacement here would be a
second implementation of the same fetch.

``azure.servicebus`` and Pillow are imported lazily in this package, so importing this
module — and the package root — needs neither installed.

Boot order: configure logging (before anything that logs) → read and validate the
environment → build the wiring → install the signal handlers → ``run_forever``, which
runs startup crash recovery to completion and starts the drain thread *before* the
queue is pulled.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType

import httpx

from osprey.bridges.core import BridgeRuntime, PipelineDeps, build_deps, run_forever

from .client import ConnectorClient, TokenSource
from .config import TeamsBridgeConfig, require_boot
from .ops import TeamsOps
from .receiver import QueueReceiver, ReceiverFactory, make_receiver, serve

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
"""Timestamp, level, logger, message — the same fields the framework's MCP-server
entrypoints show, through plain stdlib handlers: a container log is read with
``docker logs`` and wants no ANSI, and the bridge process deliberately stays clear of
the framework's rich logging stack. Records go to stderr (``basicConfig``'s default),
which Python line-buffers even when it is a pipe, so a log line is visible as it
happens rather than whenever an 8 KiB block fills."""

QUIET_LIBRARIES = ("httpx", "httpcore", "azure", "uamqp")
"""Loggers demoted to WARNING, each because it logs per-request or per-frame at INFO
and this process makes a request per posted chunk and per token refresh, and holds an
AMQP link that renews a lock every few seconds — at INFO they would bury every line
that says something. ``azure`` is named at the root on purpose: the Service Bus client
logs under several of its children (the receiver, the auto lock renewer, the AMQP
transport), and demoting the parent is what covers the ones this file would otherwise
have to keep in step with the SDK."""

CANNOT_START = "microsoft teams bridge cannot start: {reason}"
"""Message of the :class:`SystemExit` raised for an incomplete environment. One
template, so the abort an operator reads always names the missing variables."""


def configure_logging() -> None:
    """Send INFO and above to stderr in :data:`LOG_FORMAT`.

    A no-op when the root logger already has handlers (``basicConfig``'s own
    behaviour), so a host that configured logging before importing this module keeps
    its configuration.
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    for name in QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def config_from_env() -> TeamsBridgeConfig:
    """Read the environment into a validated config.

    Returns:
        The config, already checked by
        :func:`~osprey.bridges.teams.config.require_boot`.

    Raises:
        ValueError: If anything required is unset, naming the missing **environment
            variables**, or if ``TEAMS_CLOUD`` names a cloud that does not exist.
    """
    cfg = TeamsBridgeConfig.from_env()
    require_boot(cfg)
    return cfg


@dataclass(frozen=True)
class Wiring:
    """Everything Teams-specific one bridge process owns, built once at boot.

    Built by :func:`build_wiring` only — the invariants that make the pieces work
    together (one stop event, one token source behind one connector client, a deps
    bundle over that same ops instance) are properties of how they are constructed,
    not of the fields, so assembling one by hand can satisfy the types and still be
    wrong.
    """

    cfg: TeamsBridgeConfig
    stop: threading.Event
    """The single shutdown signal: the receive loop's, the engine's, and the signal
    handler's."""

    tokens: TokenSource
    """The one bearer cache. Shared so an expiry costs one AAD exchange rather than
    one per posting thread."""

    client: ConnectorClient
    ops: TeamsOps
    deps: PipelineDeps
    """The engine's collaborators, over this wiring's ops instance."""

    receiver_factory: ReceiverFactory
    """How the queue receiver is obtained, already resolved — :func:`make_receiver`
    unless a caller injected one. Resolved rather than left ``None`` because
    :func:`~osprey.bridges.teams.receiver.serve` takes a receiver and not a factory,
    so this module calls it either way and a ``None`` would only be a second spelling
    of the same default."""


def build_wiring(
    cfg: TeamsBridgeConfig,
    *,
    receiver_factory: ReceiverFactory = make_receiver,
    token_http: httpx.Client | None = None,
    connector_http: httpx.Client | None = None,
    worker_http: httpx.Client | None = None,
) -> Wiring:
    """Construct the adapter's collaborators: one stop event, one credential source,
    one connector client, one deps bundle.

    Nothing here dials anything, the default ``receiver_factory`` included: the
    Service Bus client, receiver and renewer are all lazy, so an unreachable namespace
    surfaces on the first pull rather than at wiring time.

    Args:
        cfg: The validated config.
        receiver_factory: How to obtain the queue receiver, instead of
            :func:`~osprey.bridges.teams.receiver.make_receiver`. An end-to-end tier
            injects a factory returning an in-process queue, which is what keeps
            ``azure.servicebus`` and the broker out of the path entirely. Called once
            per :func:`serve_events`, not here.
        token_http: HTTP client for the AAD token leg, instead of the one
            :class:`~osprey.bridges.teams.client.TokenSource` builds from ``cfg``.
        connector_http: HTTP client for the Bot Connector leg, instead of the one
            :class:`~osprey.bridges.teams.client.ConnectorClient` builds from ``cfg``.
            Separate from the token leg's: a different host, auth scheme and timeout
            budget.
        worker_http: HTTP client for the WORKER's artifact byte route — an internal
            service, reached with the dispatch token, and again not the Connector's.

    Returns:
        The wiring, ready for :func:`run`.
    """
    stop = threading.Event()
    tokens = TokenSource(cfg, token_http)
    client = ConnectorClient(cfg, tokens, connector_http)
    ops = TeamsOps(cfg, client=client, worker_http=worker_http)
    # No `fetch_prior_artifact` override: Teams posts image bytes inline and stamps no
    # public_url, so the engine's worker-route default already reads the only copy
    # that exists. See the module docstring.
    deps = build_deps(cfg.core, ops)
    return Wiring(
        cfg=cfg,
        stop=stop,
        tokens=tokens,
        client=client,
        ops=ops,
        deps=deps,
        receiver_factory=receiver_factory,
    )


def _close_receiver(receiver: QueueReceiver) -> None:
    """Shut the receiver down, if it is the kind that has to be shut down.

    ``close`` is deliberately absent from the
    :class:`~osprey.bridges.teams.receiver.QueueReceiver` Protocol: it is the
    production receiver's way of releasing an AMQP link and its renewal threads, and
    an in-process fake has nothing to release. So it is looked up rather than called,
    and a failure to close is logged rather than raised — this runs on the way out of
    the process, where a raise would replace the reason the bridge is stopping with
    the reason it could not tidy up.
    """
    close = getattr(receiver, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        logger.warning("closing the service bus receiver failed", exc_info=True)


def serve_events(wiring: Wiring, runtime: BridgeRuntime) -> None:
    """The adapter's blocking ingestion loop: pull the queue until shutdown.

    Passed to :func:`~osprey.bridges.core.runtime.run_forever`, which calls it only
    after startup crash recovery has finished and the drain thread is running — so the
    first new activity is accepted after the previous process's in-flight work is
    settled.

    Returning is what ends the process, so the wait belongs to
    :func:`~osprey.bridges.teams.receiver.serve` and the stop event it is given is
    ``wiring``'s — the same one the signal handler sets and ``run_forever`` holds.
    ``serve`` returns only once its handler pool has settled every message it took, so
    closing the receiver here cannot cut a settlement short.
    """
    receiver = wiring.receiver_factory(wiring.cfg)
    try:
        serve(receiver, runtime, stop=wiring.stop)
    finally:
        _close_receiver(receiver)


def run(
    wiring: Wiring,
    *,
    deps: PipelineDeps | None = None,
    gate: Callable[[], bool] | None = None,
) -> None:
    """Boot the engine on ``wiring`` and pull the queue until the stop event is set.

    Validates again: this is the entry point an in-process caller (an end-to-end test,
    an embedding host) reaches directly, and it must not be possible to boot a
    half-configured bridge by skipping :func:`config_from_env`.

    The stop event is handed to ``run_forever`` as the engine's own, so
    :meth:`BridgeRuntime.shutdown` and the ingestion shutdown are the same ``set()`` —
    the drain stops when ingestion does, in either direction.

    Args:
        wiring: The collaborators, from :func:`build_wiring`.
        deps: Engine collaborator bundle **replacing** ``wiring.deps`` outright.
        gate: The drain's health gate, instead of the dispatcher/worker ``/health``
            probe.

    Raises:
        ValueError: If ``wiring.cfg`` is incomplete.
    """
    require_boot(wiring.cfg)
    cfg = wiring.cfg
    # Neither the client secret nor the connection string appears here: a startup line
    # is the most-copied line in any incident, and one of the two is enough to post as
    # the bot.
    logger.info(
        "microsoft teams bridge starting: app=%s cloud=%s queue=%s trigger=%s "
        "dispatcher=%s worker=%s version=%s",
        cfg.app_id,
        cfg.cloud,
        cfg.servicebus_queue,
        cfg.core.trigger,
        cfg.core.dispatcher_url,
        cfg.core.worker_url,
        cfg.version_tag or "(none: the ack names no release)",
    )
    run_forever(
        cfg.core,
        wiring.ops,
        lambda runtime: serve_events(wiring, runtime),
        deps=deps if deps is not None else wiring.deps,
        gate=gate,
        stop=wiring.stop,
    )


def install_signal_handlers(stop: threading.Event) -> None:
    """Make SIGTERM and SIGINT set ``stop``.

    Compose stops the container with SIGTERM, and setting that one event is the entire
    shutdown: :func:`serve_events` returns once the loop sees it, and ``run_forever``'s
    ``finally`` calls :meth:`BridgeRuntime.shutdown` — the same event, so the drain
    thread wakes too.

    Main thread only — :func:`signal.signal` refuses to run anywhere else — which is
    why :func:`main` installs these and :func:`run` does not.
    """

    def _handler(signum: int, frame: FrameType | None) -> None:
        logger.info("received %s; shutting down", signal.Signals(signum).name)
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)


def main() -> None:
    """Run the bridge from the environment. The container's command.

    Raises:
        SystemExit: If the environment is incomplete. The message names the missing
            variables; the config error is not allowed to escape as a traceback,
            because the one thing an operator needs from a failed boot is that list.
    """
    configure_logging()
    try:
        cfg = config_from_env()
    except ValueError as exc:
        logger.error("cannot start: %s", exc)
        raise SystemExit(CANNOT_START.format(reason=exc)) from exc
    wiring = build_wiring(cfg)
    install_signal_handlers(wiring.stop)
    run(wiring)


if __name__ == "__main__":
    main()
