"""The bridge process: ``python -m osprey.bridges.nextcloud_talk``.

The whole of the adapter's ``__main__``: read the environment, refuse to start on an
incomplete one, build the collaborators, and hand
:func:`~osprey.bridges.core.runtime.run_forever` an ingestion loop of one long-poll
thread per configured room. Nothing here decides message policy — every ordering,
dedup, retry and recovery rule lives in :mod:`osprey.bridges.core`, and every Talk
wire detail in this package's other modules. What this module owns is the *wiring*,
and three of its decisions are load-bearing enough that getting them wrong produces a
bridge that looks healthy and answers nobody.

**Validation is two calls, not one.** :meth:`NextcloudBridgeConfig.require_startup`
covers the credentials, the rooms, the trigger and both tokens; it deliberately does
not cover the dispatcher/worker URLs, which have code defaults. Those defaults are a
trap under compose: an unset **bare** ``${VAR}`` renders as an *empty string*, not an
absent key, so ``CoreConfig.from_env``'s ``localhost`` fallbacks never fire and the
bridge boots with ``dispatcher_url == ""``. It would then POST every question to a
protocol-less URL and raise — and a raising ``handle_event`` leaves the poll offset
deliberately unadvanced, so the room re-fetches the same batch forever. That is an
unbounded retry loop where a startup abort belongs, which is what
:func:`require_boot`'s second check restores. (The rendered compose template guards
the same two with ``${VAR:?...}``; this check is what covers a hand-written or
otherwise un-rendered deployment.)

**One :class:`threading.Event` stops everything.** It is passed to every
:class:`~osprey.bridges.nextcloud_talk.poller.RoomPoller`, to the
:class:`~osprey.bridges.nextcloud_talk.rooms.RoomDirectory` as its backoff sleep, and
to ``run_forever`` as the engine's own stop, so one ``set()`` from the SIGTERM handler
ends the room threads, the drain thread, and any backoff wait already in progress.
Each of those wirings is a separate ``sleep``/``stop`` injection made at a *different*
constructor, and a missed one costs up to
:data:`~osprey.bridges.nextcloud_talk.rooms.BACKOFF_CAP` seconds of a container
ignoring SIGTERM.

**One :class:`RoomDirectory`, shared by the pollers and the ops object.** Only
:meth:`RoomDirectory.resolve_once` — which is poller-side — populates a directory, so
an ops object left to build its own would hold a permanently empty cache: every
``parse_event`` would raise
:class:`~osprey.bridges.nextcloud_talk.rooms.RoomTypeUnresolved`, every poll cycle
would leave its offset unadvanced, and the bridge would defer every message forever
without ever losing one or logging an error. :func:`build_wiring` is the single place
that instance is created, for exactly that reason.

Boot order: configure logging (before the config is built, so its non-HTTPS warning is
not swallowed) → read and validate the environment → build the wiring → install the
signal handlers → ``run_forever``, which runs startup crash recovery to completion and
starts the drain thread *before* :func:`serve_rooms` is called and the first room is
polled.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType

import httpx

from osprey.bridges.core import BridgeRuntime, PipelineDeps, run_forever

from .client import TalkClient
from .config import NextcloudBridgeConfig
from .offsets import OffsetStore
from .ops import NextcloudTalkOps
from .poller import RoomPoller
from .rooms import RoomDirectory

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
"""Timestamp, level, logger, message — the same fields the framework's MCP-server
entrypoints show, through plain stdlib handlers: a container log is read with
``docker logs`` and wants no ANSI, and the bridge process deliberately stays clear of
the framework's rich logging stack. Records go to stderr (``basicConfig``'s default),
which Python line-buffers even when it is a pipe, so a log line is visible as it
happens rather than whenever an 8 KiB block fills."""

QUIET_LIBRARIES = ("httpx", "httpcore")
"""Loggers demoted to WARNING. ``httpx`` logs one INFO line per request, and a bridge
makes one long-poll request per room per cycle forever — at INFO it would bury every
line that says something."""

JOIN_TIMEOUT = 10.0
"""Seconds to wait for a room thread after signalling shutdown. Generous: the threads
are daemons and cannot block process exit, so this only bounds how long a tidy
shutdown waits for one that is mid-request."""

CORE_URL_ENV = {"dispatcher_url": "DISPATCHER_URL", "worker_url": "WORKER_URL"}
"""The two ``CoreConfig`` URL fields :meth:`require_startup` does not cover, mapped to
the environment variables an operator actually sets. One mapping, so the check below
and its error message cannot name different things."""


def configure_logging() -> None:
    """Send INFO and above to stderr in :data:`LOG_FORMAT`.

    A no-op when the root logger already has handlers (``basicConfig``'s own
    behaviour), so a host that configured logging before importing this module keeps
    its configuration.
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    for name in QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def require_boot(cfg: NextcloudBridgeConfig) -> None:
    """Raise unless the environment carries everything the bridge cannot run without.

    Two checks. The second is not redundant with the first: see the module docstring
    for why an unset ``DISPATCHER_URL``/``WORKER_URL`` reaches this process as an empty
    string that no code default will ever replace, and what the bridge does with one.

    Args:
        cfg: The config to check.

    Raises:
        ValueError: If anything required is unset, naming the missing **environment
            variables** — the abort has to be diagnosable from the container log alone.
    """
    cfg.require_startup()
    try:
        cfg.core.require(*CORE_URL_ENV)
    except ValueError as exc:
        # The engine's check reports FIELD names; an operator sets env vars. Re-raise in
        # the spelling a deployment uses, keeping the original as the cause.
        missing = [env for field, env in CORE_URL_ENV.items() if not getattr(cfg.core, field)]
        raise ValueError(f"missing required config: {', '.join(missing)}") from exc


def config_from_env() -> NextcloudBridgeConfig:
    """Read the environment into a validated config.

    Building the config is what emits the non-HTTPS warning (a plain-http Nextcloud is
    a legitimate internal deployment, so it is a warning and never an abort), hence
    :func:`configure_logging` runs before this.

    Returns:
        The config, already checked by :func:`require_boot`.

    Raises:
        ValueError: If anything required is unset.
    """
    cfg = NextcloudBridgeConfig.from_env()
    require_boot(cfg)
    return cfg


@dataclass(frozen=True)
class Wiring:
    """Everything Nextcloud-specific one bridge process owns, built once at boot.

    Built by :func:`build_wiring` only — the invariants that make the pieces work
    together (one stop event, one room directory) are properties of how they are
    constructed, not of the fields, so assembling one by hand can satisfy the types and
    still be wrong.
    """

    cfg: NextcloudBridgeConfig
    stop: threading.Event
    """The single shutdown signal: every poller's, the directory's backoff sleep, and
    the engine's own."""

    client: TalkClient
    rooms: RoomDirectory
    offsets: OffsetStore
    ops: NextcloudTalkOps


def build_wiring(
    cfg: NextcloudBridgeConfig,
    *,
    talk: httpx.Client | None = None,
    worker_http: httpx.Client | None = None,
) -> Wiring:
    """Construct the adapter's collaborators: one stop event, one room directory.

    Args:
        cfg: The validated config.
        talk: HTTP client for the Talk/DAV surface, instead of the one
            :class:`TalkClient` builds from ``cfg``. Tests inject an
            :class:`httpx.MockTransport`-backed client here.
        worker_http: HTTP client for the WORKER's artifact byte route — a different
            host, auth scheme and timeout budget from Talk's, which is why it is a
            second client and not the Talk one.

    Returns:
        The wiring, ready for :func:`run`.
    """
    stop = threading.Event()
    client = TalkClient(cfg, client=talk)
    # The directory's backoff sleep is injected HERE, at its own construction, because
    # nothing later can: passing the stop event to the pollers does not reach it. Miss
    # this and a SIGTERM arriving during a room-type fetch backoff waits out up to
    # BACKOFF_CAP seconds before the process exits.
    rooms = RoomDirectory(client, sleep=stop.wait)
    # THIS directory instance goes to the ops object AND (in build_pollers) to every
    # room poller. Only RoomDirectory.resolve_once populates one and only the pollers
    # call it, so an ops object holding a private directory would read a permanently
    # empty cache: every parse_event would raise RoomTypeUnresolved, every poll cycle
    # would leave its offset unadvanced, and the bridge would defer every message
    # forever while reporting itself healthy. Two instances here is not a tidier
    # spelling of one — it is that failure.
    ops = NextcloudTalkOps(cfg, client=client, http=worker_http, rooms=rooms)
    return Wiring(
        cfg=cfg,
        stop=stop,
        client=client,
        rooms=rooms,
        offsets=OffsetStore(cfg.offsets_path),
        ops=ops,
    )


def build_pollers(wiring: Wiring, runtime: BridgeRuntime) -> list[RoomPoller]:
    """One :class:`RoomPoller` per configured room, all sharing ``wiring``'s
    collaborators.

    The shared pieces are all thread-safe and each poller keeps only its own room's
    cursor privately, so sharing is the intended shape rather than an economy: the
    directory in particular MUST be the one the ops object holds (see
    :func:`build_wiring`), and the stop event MUST be the one the signal handler sets.
    """
    return [
        RoomPoller(
            room,
            client=wiring.client,
            rooms=wiring.rooms,
            offsets=wiring.offsets,
            runtime=runtime,
            stop=wiring.stop,
        )
        for room in wiring.cfg.rooms
    ]


def serve_rooms(wiring: Wiring, runtime: BridgeRuntime) -> None:
    """The adapter's blocking ingestion loop: start every room thread, wait for
    shutdown, stop them.

    Passed to :func:`~osprey.bridges.core.runtime.run_forever`, which calls it only
    after startup crash recovery has finished and the drain thread is running — so the
    first message is polled after the previous process's in-flight work is settled.

    Returning is what ends the process, so the wait is on the stop event and nothing
    else. The ``finally`` matters for the one path that is not a signal (an exception
    escaping the wait, e.g. a ``KeyboardInterrupt`` on a host without the handlers
    installed): the rooms are still told to stop rather than left polling.
    """
    pollers = build_pollers(wiring, runtime)
    logger.info("starting %d room poller(s): %s", len(pollers), ", ".join(p.room for p in pollers))
    for poller in pollers:
        poller.start()
    try:
        wiring.stop.wait()
    finally:
        for poller in pollers:
            poller.stop()
        for poller in pollers:
            poller.join(JOIN_TIMEOUT)


def run(
    wiring: Wiring,
    *,
    deps: PipelineDeps | None = None,
    gate: Callable[[], bool] | None = None,
) -> None:
    """Boot the engine on ``wiring`` and serve its rooms until the stop event is set.

    Validates again: this is the entry point an in-process caller (an end-to-end test,
    an embedding host) reaches directly, and it must not be possible to boot a
    half-configured bridge by skipping :func:`config_from_env`.

    The stop event is handed to ``run_forever`` as the engine's own, so
    :meth:`BridgeRuntime.shutdown` and the room shutdown are the same ``set()`` — the
    drain stops when ingestion does, in either direction.

    Args:
        wiring: The collaborators, from :func:`build_wiring`.
        deps: Engine collaborator bundle, instead of the one
            :func:`~osprey.bridges.core.runtime.build_deps` derives from
            ``wiring.cfg.core`` and ``wiring.ops``. Tests pass a bundle carrying a
            :class:`~osprey.bridges.core.DispatchClient` over a mock transport.
        gate: The drain's health gate, instead of the dispatcher/worker ``/health``
            probe.

    Raises:
        ValueError: If ``wiring.cfg`` is incomplete.
    """
    require_boot(wiring.cfg)
    cfg = wiring.cfg
    logger.info(
        "nextcloud talk bridge starting: instance=%s bot=%s rooms=%s trigger=%s "
        "dispatcher=%s worker=%s",
        cfg.base_url,
        cfg.bot_account,
        ",".join(cfg.rooms),
        cfg.core.trigger,
        cfg.core.dispatcher_url,
        cfg.core.worker_url,
    )
    run_forever(
        cfg.core,
        wiring.ops,
        lambda runtime: serve_rooms(wiring, runtime),
        deps=deps,
        gate=gate,
        stop=wiring.stop,
    )


def install_signal_handlers(stop: threading.Event) -> None:
    """Make SIGTERM and SIGINT set ``stop``.

    Compose stops the container with SIGTERM, and setting that one event is the entire
    shutdown: :func:`serve_rooms` returns, ``run_forever``'s ``finally`` calls
    :meth:`BridgeRuntime.shutdown` (the same event, so the drain thread wakes too), and
    any room or directory backoff wait in progress returns immediately instead of being
    waited out.

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
        raise SystemExit(f"nextcloud talk bridge cannot start: {exc}") from exc
    wiring = build_wiring(cfg)
    install_signal_handlers(wiring.stop)
    run(wiring)


if __name__ == "__main__":
    main()
