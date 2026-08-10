"""The bridge process: ``python -m osprey.bridges.google_chat``.

The whole of the adapter's ``__main__``: read the environment, refuse to start on an
incomplete one, build the collaborators, and hand
:func:`~osprey.bridges.core.runtime.run_forever` an ingestion loop holding one Pub/Sub
streaming pull. Nothing here decides message policy — every ordering, dedup, retry and
recovery rule lives in :mod:`osprey.bridges.core`, and every Chat/GCS wire detail in
this package's other modules. What this module owns is the *wiring*, and three of its
decisions are load-bearing enough that getting them wrong produces a bridge that looks
healthy and answers nobody.

**One :class:`threading.Event` stops everything.** It is handed to
:func:`~osprey.bridges.google_chat.subscriber.serve` and to ``run_forever`` as the
engine's own, so one ``set()`` from the SIGTERM handler cancels the streaming pull and
wakes the drain thread. Two events here would be a container that acknowledges SIGTERM
in the log and then keeps leasing messages until it is killed.

**One :class:`~osprey.bridges.google_chat.client.ChatClient`, shared by the posting
members and the reply-context reads.** The client serializes its HTTP leg behind its
own lock precisely because the drain thread posts alongside live ingestion; a second
instance would be a second connection with a second lock, which is the shape that lock
exists to prevent. :func:`build_wiring` is the single place it is constructed.

**The engine's prior-artifact fetcher is replaced, not extended.** The default reads
the worker's byte route and stops, and for this channel that throws away the better
copy: the worker's file is swept once the run is old, while the GCS object the space
is already linking to is permanent. :func:`build_wiring` overrides
:attr:`~osprey.bridges.core.pipeline.PipelineDeps.fetch_prior_artifact` with
:func:`~osprey.bridges.google_chat.ops.build_prior_artifact_fetcher`, which tries both
— so a follow-up question about yesterday's plot still gets the plot.

**Injected seams, and why there are exactly four.** Every collaborator that would
otherwise reach Google is injectable here: the Pub/Sub subscriber, the Chat discovery
service, the GCS storage client, and the HTTP client used for the worker's byte route.
An end-to-end tier supplies all four (an emulator subscriber, a fake Chat service, a
fake storage client) and thereby exercises this module's real wiring — the same
``build_wiring`` and ``run`` production calls — with no credentials and no Google
packages installed. The Chat seam in particular *bypasses* rather than replaces:
``ChatClient`` builds a real discovery service the moment it is given none, which
imports ``googleapiclient`` and reads the service-account key file, so a seam that
merely overwrote the attribute afterwards would already have failed.

Google's libraries are imported lazily throughout this package, so importing this
module — and the package root — needs none of them installed.

Boot order: configure logging (before the config is built, so its malformed-subscription
warning is not swallowed) → read and validate the environment → build the wiring →
install the signal handlers → ``run_forever``, which runs startup crash recovery to
completion and starts the drain thread *before* the subscription is pulled.
"""

from __future__ import annotations

import functools
import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import FrameType
from typing import Any

import httpx

from osprey.bridges.core import BridgeRuntime, PipelineDeps, build_deps, run_forever

from .client import ChatClient
from .config import GoogleChatBridgeConfig, require_boot
from .gcs import publish_artifacts, publish_documents
from .ops import DocumentPublisher, GoogleChatOps, ImagePublisher, build_prior_artifact_fetcher
from .subscriber import SubscriberFactory, serve

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
"""Timestamp, level, logger, message — the same fields the framework's MCP-server
entrypoints show, through plain stdlib handlers: a container log is read with
``docker logs`` and wants no ANSI, and the bridge process deliberately stays clear of
the framework's rich logging stack. Records go to stderr (``basicConfig``'s default),
which Python line-buffers even when it is a pipe, so a log line is visible as it
happens rather than whenever an 8 KiB block fills."""

QUIET_LIBRARIES = ("httpx", "httpcore", "googleapiclient.discovery", "google.auth")
"""Loggers demoted to WARNING, each because it logs per-request at INFO and this
process makes a request per posted chunk, per published artifact and per token
refresh — at INFO they would bury every line that says something."""

CANNOT_START = "google chat bridge cannot start: {reason}"
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


def config_from_env() -> GoogleChatBridgeConfig:
    """Read the environment into a validated config.

    Building the config is what emits the malformed-subscription warning (a name Google
    would accept is never rejected here, so it is a warning and never an abort), hence
    :func:`configure_logging` runs before this.

    Returns:
        The config, already checked by
        :func:`~osprey.bridges.google_chat.config.require_boot`.

    Raises:
        ValueError: If anything required is unset, naming the missing **environment
            variables**.
    """
    cfg = GoogleChatBridgeConfig.from_env()
    require_boot(cfg)
    return cfg


@dataclass(frozen=True)
class Wiring:
    """Everything Google-specific one bridge process owns, built once at boot.

    Built by :func:`build_wiring` only — the invariants that make the pieces work
    together (one stop event, one Chat client, a deps bundle carrying this channel's
    prior-artifact fetcher) are properties of how they are constructed, not of the
    fields, so assembling one by hand can satisfy the types and still be wrong.
    """

    cfg: GoogleChatBridgeConfig
    stop: threading.Event
    """The single shutdown signal: the subscriber's, the engine's, and the signal
    handler's."""

    client: ChatClient
    ops: GoogleChatOps
    deps: PipelineDeps
    """The engine's collaborators, with ``fetch_prior_artifact`` already overridden."""

    subscriber_factory: SubscriberFactory | None = None
    """How the Pub/Sub client is obtained, or ``None`` for
    :func:`~osprey.bridges.google_chat.subscriber.serve`'s own default (a real client
    built from the service-account key). Left unresolved rather than defaulted here so
    that default lives in exactly one place."""


def _publishers(
    gcs_client: Any | None, worker_http: httpx.Client | None
) -> tuple[ImagePublisher, DocumentPublisher]:
    """The two artifact publishers, with the injected clients bound into them.

    Both publishers take their storage client and their worker HTTP client as
    keyword-only seams, and :class:`~osprey.bridges.google_chat.ops.GoogleChatOps` takes
    the publishers as plain callables — so binding here is what carries an injection
    through ops without ops knowing anything about it.

    With neither injected this returns the module functions **themselves**, unwrapped:
    production's publishers are the ones the ops conformance suite pins, and a partial
    with no bound arguments would be a difference that only shows up in a test.
    """
    if gcs_client is None and worker_http is None:
        return publish_artifacts, publish_documents
    bound: dict[str, Any] = {}
    if worker_http is not None:
        bound["http"] = worker_http
    if gcs_client is not None:
        bound["client_factory"] = lambda _cfg: gcs_client
    return (
        functools.partial(publish_artifacts, **bound),
        functools.partial(publish_documents, **bound),
    )


def build_wiring(
    cfg: GoogleChatBridgeConfig,
    *,
    subscriber_factory: SubscriberFactory | None = None,
    chat_service: Any | None = None,
    gcs_client: Any | None = None,
    worker_http: httpx.Client | None = None,
) -> Wiring:
    """Construct the adapter's collaborators: one stop event, one Chat client, one
    deps bundle.

    Args:
        cfg: The validated config.
        subscriber_factory: How to obtain the Pub/Sub subscriber client, instead of the
            one built from ``cfg.sa_key``. An end-to-end tier injects a factory pointed
            at a Pub/Sub emulator.
        chat_service: Discovery service for the Chat REST calls, instead of the one
            :class:`ChatClient` builds from ``cfg``. Injecting one is what keeps
            ``googleapiclient`` and the key file out of the path entirely — see the
            module docstring on why this seam bypasses rather than replaces.
        gcs_client: Storage client for artifact publication, instead of the one built
            from ``cfg.sa_key``. Bound into both publishers.
        worker_http: HTTP client for the WORKER's artifact byte route, shared by the
            publishers and by the prior-artifact fetcher's worker leg — one client, one
            connection pool, for the one internal service both of them read.

    Returns:
        The wiring, ready for :func:`run`.
    """
    stop = threading.Event()
    # ChatClient builds its own discovery service when given none, which imports the
    # Google libraries and reads the key file. Passing the seam straight to the
    # constructor is what keeps that from happening at all, rather than happening and
    # then being discarded.
    client = ChatClient(cfg, chat_service)
    publisher, doc_publisher = _publishers(gcs_client, worker_http)
    ops = GoogleChatOps(cfg, client=client, publisher=publisher, doc_publisher=doc_publisher)
    deps = replace(
        build_deps(cfg.core, ops),
        # Not an embellishment: the engine's default reads only the worker's copy, which
        # is swept while the GCS object the answer already links to is permanent. See
        # build_prior_artifact_fetcher.
        fetch_prior_artifact=build_prior_artifact_fetcher(worker_http=worker_http),
    )
    return Wiring(
        cfg=cfg,
        stop=stop,
        client=client,
        ops=ops,
        deps=deps,
        subscriber_factory=subscriber_factory,
    )


def serve_events(wiring: Wiring, runtime: BridgeRuntime) -> None:
    """The adapter's blocking ingestion loop: hold the streaming pull until shutdown.

    Passed to :func:`~osprey.bridges.core.runtime.run_forever`, which calls it only
    after startup crash recovery has finished and the drain thread is running — so the
    first new event is accepted after the previous process's in-flight work is settled.

    Returning is what ends the process, so the wait belongs to
    :func:`~osprey.bridges.google_chat.subscriber.serve` and the stop event it is given
    is ``wiring``'s — the same one the signal handler sets and ``run_forever`` holds.
    """
    if wiring.subscriber_factory is None:
        serve(wiring.cfg, runtime, stop=wiring.stop)
    else:
        serve(
            wiring.cfg,
            runtime,
            stop=wiring.stop,
            subscriber_factory=wiring.subscriber_factory,
        )


def run(
    wiring: Wiring,
    *,
    deps: PipelineDeps | None = None,
    gate: Callable[[], bool] | None = None,
) -> None:
    """Boot the engine on ``wiring`` and pull the subscription until the stop event is
    set.

    Validates again: this is the entry point an in-process caller (an end-to-end test,
    an embedding host) reaches directly, and it must not be possible to boot a
    half-configured bridge by skipping :func:`config_from_env`.

    The stop event is handed to ``run_forever`` as the engine's own, so
    :meth:`BridgeRuntime.shutdown` and the ingestion shutdown are the same ``set()`` —
    the drain stops when ingestion does, in either direction.

    Args:
        wiring: The collaborators, from :func:`build_wiring`.
        deps: Engine collaborator bundle **replacing** ``wiring.deps`` outright. A
            caller passing one takes over ``fetch_prior_artifact`` with it, so a bundle
            that wants this channel's two-route fetcher must carry it (the simplest
            spelling is :func:`dataclasses.replace` on ``wiring.deps``).
        gate: The drain's health gate, instead of the dispatcher/worker ``/health``
            probe.

    Raises:
        ValueError: If ``wiring.cfg`` is incomplete.
    """
    require_boot(wiring.cfg)
    cfg = wiring.cfg
    logger.info(
        "google chat bridge starting: app=%s subscription=%s bucket=%s trigger=%s "
        "dispatcher=%s worker=%s",
        cfg.app_id,
        cfg.subscription,
        cfg.gcs_bucket or "(none: text-only delivery)",
        cfg.core.trigger,
        cfg.core.dispatcher_url,
        cfg.core.worker_url,
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
    shutdown: :func:`serve_events` returns once the pull is cancelled, and
    ``run_forever``'s ``finally`` calls :meth:`BridgeRuntime.shutdown` — the same event,
    so the drain thread wakes too.

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
