"""The bridge process: ``python -m osprey.bridges.slack``.

The whole of the adapter's ``__main__``: read the environment, refuse to start on an
incomplete one, build the collaborators, and hand
:func:`~osprey.bridges.core.runtime.run_forever` an ingestion loop fed by one Socket
Mode connection. Nothing here decides message policy — every ordering, dedup, retry and
recovery rule lives in :mod:`osprey.bridges.core`, and every Slack wire detail in this
package's other modules. What this module owns is the *wiring*, and its load-bearing
decisions are these.

**One stop event and one fatal event.** The stop :class:`threading.Event` is the
signal handler's, the ingest thread's, the serve loop's and ``run_forever``'s own, so
one ``set()`` from SIGTERM ends ingestion and wakes the drain thread. The fatal event
is the :class:`~osprey.bridges.slack.client.SlackWebClient`'s: the client sets it on a
dead credential, the socket sets it on ``link_disabled``, and the ingest thread and the
serve loop read it — two of either would be a bridge that logs a dead token and keeps
running as if it had none.

**Who the bot is comes from ``auth.test``, at wiring time.** :func:`build_wiring` is the
one place this process talks to Slack before it runs: the answer becomes the
:class:`~osprey.bridges.slack.events.BotIdentity` every parse and every roster lookup
reads. A credential Slack refuses there is a bridge that cannot start, and
:func:`main` says so with the cause rather than a traceback.

**Proxy split (Decision 2).** The Slack legs — Web API and the Socket Mode WebSocket —
honour the environment's proxy settings exactly when ``BRIDGE_TRUST_ENV`` is set. The
legs to the dispatcher and the worker never do: the engine's deps are built from a copy
of the core config with ``trust_env=False``, and the adapter's own worker byte route
reads the same proxy-free copy. A site proxy in front of an internal service is a
bridge whose every dispatch times out.

**Ingest starts before the engine (Decision 4).** :func:`run` opens the socket and
starts the ingest thread, then calls ``run_forever``, whose startup reconcile runs
before ``serve`` is called. Events that arrive meanwhile are spooled to the durable
inbox and acknowledged, so Slack does not redeliver them behind a slow recovery; serve
picks them up once the engine is ready. Whatever ends ``run_forever``, the ingest thread
is stopped on the way out, so no daemon thread outlives it.

**Injected seams.** Every collaborator that would otherwise reach Slack is injectable:
the WebSocket ``connect`` and the Web API HTTP client. The inbox directory is too, so a
test never writes beside a real deployment's state. An end-to-end tier supplies them and
thereby exercises this module's real wiring — the same ``build_wiring`` and ``run``
production calls — with no workspace and no token.

Boot order: configure logging (before anything that logs) → read and validate the
environment → build the wiring (``auth.test``) → install the signal handlers → ``run``.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from websockets.sync.client import connect as ws_connect

from osprey.bridges.core import (
    PipelineDeps,
    build_deps,
    configure_bridge_logging,
    install_signal_handlers,
    run_forever,
)

from .client import SlackAuthError, SlackWebClient
from .config import SlackBridgeConfig, require_boot
from .events import BotIdentity
from .inbox import Inbox
from .ops import SlackOps
from .receiver import Ingest, serve
from .socket import SocketModeConnection

logger = logging.getLogger(__name__)

QUIET_LIBRARIES = ("httpx", "httpcore", "websockets")
"""Loggers demoted to WARNING: each logs per request or per frame at INFO, and this
process makes a request per posted chunk and holds a WebSocket that pings every few
seconds — at INFO they would bury every line that says something."""

CANNOT_START = "slack bridge cannot start: {reason}"
"""Message of the :class:`SystemExit` raised for an incomplete environment or a
credential Slack refuses. One template, so the abort an operator reads always names
the cause."""

INBOX_DIRNAME = "inbox"
"""The inbox's directory name, beside ``DEDUP_PATH`` on the bridge's state volume, so
a spooled event survives a restart exactly as the dedup record does."""

INGEST_JOIN_SEC = 5.0
"""How long :func:`run` waits for the ingest thread on the way out. The socket reads
with a one-second timeout, so a healthy thread is gone well inside this; one that is
not is a daemon and cannot hold the process up."""


def config_from_env() -> SlackBridgeConfig:
    """Read the environment into a validated config.

    Returns:
        The config, already checked by :func:`~osprey.bridges.slack.config.require_boot`.

    Raises:
        ValueError: If anything required is unset, naming the missing **environment
            variables**.
    """
    cfg = SlackBridgeConfig.from_env()
    require_boot(cfg)
    return cfg


@dataclass(frozen=True)
class Wiring:
    """Everything Slack-specific one bridge process owns, built once at boot.

    Built by :func:`build_wiring` only — the invariants that make the pieces work
    together (one stop event, one fatal event behind one Web API client, a deps bundle
    over that same ops instance) are properties of how they are constructed, not of
    the fields.
    """

    cfg: SlackBridgeConfig
    stop: threading.Event
    """The single shutdown signal: the ingest thread's, the serve loop's, the
    engine's, and the signal handler's."""

    fatal: threading.Event
    """The single dead-credential flag, :attr:`SlackWebClient.fatal`."""

    ctx: BotIdentity
    """Who the bot is, from ``auth.test``."""

    client: SlackWebClient
    connection: SocketModeConnection
    """The Socket Mode link; nothing is opened until the ingest thread iterates it."""

    inbox: Inbox
    ops: SlackOps
    deps: PipelineDeps
    """The engine's collaborators, over this wiring's ops instance and a proxy-free
    copy of the core config."""


def default_inbox_dir(cfg: SlackBridgeConfig) -> Path:
    """The inbox directory a deployment uses: :data:`INBOX_DIRNAME` beside
    ``DEDUP_PATH``, on the same state volume."""
    return Path(cfg.core.dedup_path).parent / INBOX_DIRNAME


def build_wiring(
    cfg: SlackBridgeConfig,
    *,
    connect: Callable[..., Any] | None = None,
    slack_http: httpx.Client | None = None,
    inbox_dir: str | Path | None = None,
) -> Wiring:
    """Construct the adapter's collaborators and learn who the bot is.

    The one network call is ``auth.test`` over the Web API client; the socket is
    built but not opened.

    Args:
        cfg: The validated config.
        connect: The WebSocket ``connect`` for the Socket Mode link, instead of
            ``websockets.sync.client.connect``.
        slack_http: HTTP client for the Web API and file hosts, instead of the one
            :class:`SlackWebClient` builds honouring ``BRIDGE_TRUST_ENV``.
        inbox_dir: Where the durable inbox lives, instead of
            :func:`default_inbox_dir`.

    Returns:
        The wiring, ready for :func:`run`.

    Raises:
        SlackAuthError: If Slack refuses the bot token (the fatal flag is set).
        SlackApiError: If ``auth.test`` fails for any other reason.
    """
    stop = threading.Event()
    fatal = threading.Event()
    client = SlackWebClient(cfg, slack_http, fatal)
    answer = client.auth_test()
    ctx = BotIdentity(
        bot_user_id=str(answer.get("user_id") or ""),
        bot_id=str(answer.get("bot_id") or ""),
        team_id=str(answer.get("team_id") or ""),
    )
    # Decision 2: the dispatcher, the worker and the adapter's worker byte route are
    # internal services and never go through a proxy; only the Slack legs follow
    # BRIDGE_TRUST_ENV, and they read the unmodified cfg.
    internal_core = dataclasses.replace(cfg.core, trust_env=False)
    ops = SlackOps(dataclasses.replace(cfg, core=internal_core), client, ctx)
    deps = build_deps(internal_core, ops)
    connection = SocketModeConnection(
        client,
        cfg.app_token,
        connect=connect if connect is not None else ws_connect,
        trust_env=cfg.core.trust_env,
    )
    inbox = Inbox(inbox_dir if inbox_dir is not None else default_inbox_dir(cfg))
    return Wiring(
        cfg=cfg,
        stop=stop,
        fatal=fatal,
        ctx=ctx,
        client=client,
        connection=connection,
        inbox=inbox,
        ops=ops,
        deps=deps,
    )


def run(
    wiring: Wiring,
    *,
    deps: PipelineDeps | None = None,
    gate: Callable[[], bool] | None = None,
) -> None:
    """Start ingestion, boot the engine on ``wiring``, and serve until stopped.

    Validates again: this is the entry point an in-process caller (an end-to-end test,
    an embedding host) reaches directly, and it must not be possible to boot a
    half-configured bridge by skipping :func:`config_from_env`.

    Args:
        wiring: The collaborators, from :func:`build_wiring`.
        deps: Engine collaborator bundle **replacing** ``wiring.deps`` outright.
        gate: The drain's health gate, instead of the dispatcher/worker ``/health``
            probe.

    Raises:
        ValueError: If ``wiring.cfg`` is incomplete.
        SlackAuthError: If the credential dies while running.
    """
    require_boot(wiring.cfg)
    cfg = wiring.cfg
    # Neither token appears here: a startup line is the most-copied line in any
    # incident, and either token is enough to act as the app.
    logger.info(
        "slack bridge starting: team=%s bot=%s trigger=%s dispatcher=%s worker=%s "
        "version=%s mentions=%s slack_proxy=%s",
        wiring.ctx.team_id,
        wiring.ctx.bot_user_id,
        cfg.core.trigger,
        cfg.core.dispatcher_url,
        cfg.core.worker_url,
        cfg.version_tag or "(none: the ack names no release)",
        "on" if cfg.mentions else "off",
        "environment" if cfg.core.trust_env else "none",
    )
    ingest = Ingest(wiring.connection, wiring.inbox, stop=wiring.stop, fatal=wiring.fatal)
    ingest.start()
    try:
        run_forever(
            wiring.deps.cfg,
            wiring.ops,
            lambda runtime: serve(
                wiring.inbox, runtime, stop=wiring.stop, fatal=wiring.fatal, ingest=ingest
            ),
            deps=deps if deps is not None else wiring.deps,
            gate=gate,
            stop=wiring.stop,
        )
    finally:
        wiring.stop.set()
        wiring.connection.close()
        ingest.join(timeout=INGEST_JOIN_SEC)


def _cannot_start(exc: BaseException) -> SystemExit:
    logger.error("cannot start: %s", exc)
    return SystemExit(CANNOT_START.format(reason=exc))


def main() -> None:
    """Run the bridge from the environment. The container's command.

    Raises:
        SystemExit: If the environment is incomplete (naming the missing variables)
            or Slack refuses the credential (naming Slack's error code). Neither is
            allowed to escape as a traceback, because the one thing an operator needs
            from a failed boot is the cause.
    """
    configure_bridge_logging(QUIET_LIBRARIES)
    try:
        cfg = config_from_env()
    except ValueError as exc:
        raise _cannot_start(exc) from exc
    try:
        wiring = build_wiring(cfg)
        install_signal_handlers(wiring.stop)
        run(wiring)
    except SlackAuthError as exc:
        raise _cannot_start(exc) from exc


if __name__ == "__main__":
    main()
