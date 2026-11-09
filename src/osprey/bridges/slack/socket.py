"""Slack Socket Mode: one outbound websocket that carries the app's events.

:class:`SocketModeConnection` asks ``apps.connections.open`` (with the app-level
``xapp-`` token) for a ``wss://`` URL, connects, waits for Slack's ``hello``, and
then iterates. Each item is either an :class:`Envelope` — ``(envelope_id, type,
payload)`` — or ``None``, the **idle tick**. :meth:`~SocketModeConnection.ack`
answers an envelope; :meth:`~SocketModeConnection.close` ends the iteration.

Idle ticks
----------
Every receive waits at most :data:`RECV_TIMEOUT_SEC`, and a quiet second yields
``None`` (Decision 7). Waits inside the connection — for ``hello``, or between
reconnect attempts — are sliced the same way, so a caller servicing a stop flag,
the fatal flag or a supervise timer on each tick is never parked for longer than
about a second.

Disconnects
-----------
* ``refresh_requested`` and ``warning`` (and any reason Slack adds later): a new
  connection is opened and has said ``hello`` **before** the old one is closed,
  so the app is never without a link. If the new one cannot be opened, the old
  keeps serving until it dies, which is then an ordinary abnormal close.
* ``link_disabled``: Socket Mode was turned off for the app. The shared fatal
  flag is set and :class:`~.client.SlackAuthError` raised — reconnecting would
  only loop (Decision 6).
* An abnormal close, or a failure to open or to reach ``hello``: reconnect after
  a capped, jittered exponential backoff. The streak restarts once a connection
  has stayed up for :data:`BACKOFF_RESET_SEC`, so one blip after hours of service
  waits the short delay, while a link that drops right after ``hello`` backs off.

An auth-fatal answer from ``apps.connections.open`` propagates as the client
raised it (it has already set the fatal flag).

Proxy and secrets
-----------------
The websocket honours the environment's proxy exactly when ``trust_env`` is set
(``proxy=True``; else ``proxy=None``) — Decision 2. Neither the app token nor the
connection URL (its ``ticket`` is a credential) is ever logged; failures are
logged by exception type only.

Threads
-------
Iterate from one thread. :meth:`ack` and :meth:`close` may be called from any
thread; a lock guards the current connection and serialises sends.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Callable, Generator, Iterator
from typing import Any, NamedTuple

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

from .client import SlackAuthError

logger = logging.getLogger(__name__)

__all__ = [
    "BACKOFF_CAP_SEC",
    "BACKOFF_FACTOR",
    "BACKOFF_RESET_SEC",
    "BACKOFF_START_SEC",
    "HELLO_TIMEOUT_SEC",
    "OPEN_TIMEOUT_SEC",
    "PING_INTERVAL_SEC",
    "PING_TIMEOUT_SEC",
    "RECV_TIMEOUT_SEC",
    "Envelope",
    "SocketModeConnection",
]

RECV_TIMEOUT_SEC = 1.0
"""Longest single wait on the socket, and so the longest gap between idle ticks."""

HELLO_TIMEOUT_SEC = 10.0
"""How long a fresh connection may take to send ``hello`` before it counts as failed."""

OPEN_TIMEOUT_SEC = 10.0
"""Timeout for the websocket opening handshake (TCP, proxy CONNECT, TLS, upgrade)."""

PING_INTERVAL_SEC = 10.0
"""Websocket keepalive ping period — short, because corporate proxies drop idle
CONNECT tunnels and a dead link must be noticed well before Slack's retry window."""

PING_TIMEOUT_SEC = 10.0
"""How long a ping may go unanswered before the link is treated as dead."""

BACKOFF_START_SEC = 1.0
"""Delay before the first reconnect attempt of a failure streak."""

BACKOFF_FACTOR = 2.0
"""Each further consecutive failure multiplies the delay by this."""

BACKOFF_CAP_SEC = 60.0
"""The longest delay between reconnect attempts."""

BACKOFF_RESET_SEC = 60.0
"""A connection that stayed up this long ends the failure streak."""

_LINK_DISABLED = "link_disabled"
_HELLO = "hello"
_DISCONNECT = "disconnect"


class Envelope(NamedTuple):
    """One Socket Mode delivery: ack ``envelope_id``; ``payload`` is a mapping
    (``{}`` when Slack sent something else)."""

    envelope_id: str
    type: str
    payload: dict[str, Any]


class SocketModeConnection:
    """The app's Socket Mode link, reconnecting itself until closed."""

    def __init__(
        self,
        web_client: Any,
        app_token: str,
        *,
        connect: Callable[..., Any] = ws_connect,
        trust_env: bool,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        """Wire the connection; nothing is opened until iteration starts.

        Args:
            web_client: The :class:`~.client.SlackWebClient`; its ``call`` opens the
                link and its ``fatal`` flag is set on ``link_disabled``.
            app_token: The ``xapp-`` token ``apps.connections.open`` needs.
            connect: ``websockets.sync.client.connect`` or a test fake.
            trust_env: Honour the environment's proxy settings (Decision 2).
            clock: Monotonic seconds, for the hello deadline, backoff and ticks.
            sleep: How a backoff slice is waited out.
            rng: ``[0, 1)`` source for the backoff jitter.
        """
        self._web = web_client
        self._app_token = app_token
        self._connect = connect
        self._proxy: bool | None = True if trust_env else None
        self._clock = clock
        self._sleep = sleep
        self._rng = rng
        self._lock = threading.Lock()
        self._conn: Any = None
        self._closed = False

    # -- public surface ----------------------------------------------------------

    def __iter__(self) -> Iterator[Envelope | None]:
        return self._run()

    def ack(self, envelope_id: str) -> bool:
        """Acknowledge one envelope on the current connection.

        Returns ``False`` (and logs) when there is no live connection to send on;
        the envelope is then redelivered by Slack, which the inbox dedups.
        """
        frame = json.dumps({"envelope_id": envelope_id})
        with self._lock:
            conn = self._conn
            if conn is None:
                logger.warning("slack socket: no connection to ack an envelope on")
                return False
            try:
                conn.send(frame)
            except (ConnectionClosed, OSError) as exc:
                logger.warning("slack socket: ack failed (%s)", type(exc).__name__)
                return False
        return True

    def close(self) -> None:
        """End the iteration and close the current connection. Idempotent."""
        with self._lock:
            self._closed = True
            conn, self._conn = self._conn, None
        _close_quietly(conn)

    # -- the loop --------------------------------------------------------------

    def _run(self) -> Generator[Envelope | None, None, None]:
        failures = 0
        connected_at = 0.0
        last_tick = self._clock()
        while not self._closed:
            conn = self._conn
            if conn is None:
                if failures:
                    yield from self._pause(self._backoff(failures))
                    if self._closed:
                        return
                fresh = yield from self._open()
                if fresh is None:
                    failures += 1
                    continue
                if not self._install(fresh):
                    return
                connected_at = last_tick = self._clock()
                continue

            try:
                raw = conn.recv(timeout=RECV_TIMEOUT_SEC)
            except TimeoutError:
                last_tick = self._clock()
                yield None
                continue
            except (ConnectionClosed, OSError) as exc:
                if self._closed:
                    return
                logger.warning("slack socket: connection lost (%s)", type(exc).__name__)
                self._drop(conn)
                if self._clock() - connected_at >= BACKOFF_RESET_SEC:
                    failures = 0
                failures += 1
                continue

            frame = _decode(raw)
            kind = frame.get("type") if frame is not None else None
            if kind == _DISCONNECT:
                reason = frame.get("reason") if frame is not None else None
                if reason == _LINK_DISABLED:
                    self._fatal_link_disabled()
                logger.info("slack socket: disconnect (%s); refreshing", _short(reason))
                fresh = yield from self._open()
                if fresh is None:
                    logger.warning("slack socket: refresh failed; staying on the old link")
                elif not self._install(fresh):
                    return
                else:
                    connected_at = last_tick = self._clock()
                    _close_quietly(conn)
                continue

            env = _envelope(frame)
            if env is not None:
                last_tick = self._clock()
                yield env
            elif self._clock() - last_tick >= RECV_TIMEOUT_SEC:
                last_tick = self._clock()
                yield None

    def _open(self) -> Generator[None, None, Any]:
        """Open one connection and wait for ``hello``; ``None`` on any failure."""
        try:
            answer = self._web.call("apps.connections.open", token=self._app_token)
        except SlackAuthError:
            raise
        except Exception as exc:  # any other failure is worth a retry
            logger.warning("slack socket: apps.connections.open failed (%s)", type(exc).__name__)
            return None
        url = answer.get("url") if isinstance(answer, dict) else None
        if not isinstance(url, str) or not url:
            logger.warning("slack socket: apps.connections.open returned no url")
            return None
        try:
            conn = self._connect(
                url,
                proxy=self._proxy,
                open_timeout=OPEN_TIMEOUT_SEC,
                ping_interval=PING_INTERVAL_SEC,
                ping_timeout=PING_TIMEOUT_SEC,
            )
        except Exception as exc:  # handshake, proxy, DNS, TLS: retry later
            logger.warning("slack socket: connect failed (%s)", type(exc).__name__)
            return None

        deadline = self._clock() + HELLO_TIMEOUT_SEC
        while True:
            if self._closed:
                _close_quietly(conn)
                return None
            remaining = deadline - self._clock()
            if remaining <= 0:
                logger.warning("slack socket: no hello within %.0f s", HELLO_TIMEOUT_SEC)
                _close_quietly(conn)
                return None
            try:
                raw = conn.recv(timeout=min(RECV_TIMEOUT_SEC, remaining))
            except TimeoutError:
                yield None
                continue
            except (ConnectionClosed, OSError) as exc:
                logger.warning("slack socket: closed before hello (%s)", type(exc).__name__)
                _close_quietly(conn)
                return None
            frame = _decode(raw)
            if frame is None:
                continue
            if frame.get("type") == _HELLO:
                return conn
            if frame.get("type") == _DISCONNECT and frame.get("reason") == _LINK_DISABLED:
                _close_quietly(conn)
                self._fatal_link_disabled()

    def _pause(self, seconds: float) -> Generator[None, None, None]:
        """Wait ``seconds`` in idle-tick slices, ending early on :meth:`close`."""
        deadline = self._clock() + seconds
        while not self._closed:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            self._sleep(min(RECV_TIMEOUT_SEC, remaining))
            yield None

    def _backoff(self, failures: int) -> float:
        base = min(BACKOFF_CAP_SEC, BACKOFF_START_SEC * BACKOFF_FACTOR ** (failures - 1))
        return base * (0.5 + 0.5 * self._rng())

    def _install(self, conn: Any) -> bool:
        """Make ``conn`` current; ``False`` (and close it) if we were closed meanwhile."""
        with self._lock:
            if self._closed:
                stale = conn
            else:
                self._conn, stale = conn, None
        _close_quietly(stale)
        return stale is None

    def _drop(self, conn: Any) -> None:
        with self._lock:
            if self._conn is conn:
                self._conn = None
        _close_quietly(conn)

    def _fatal_link_disabled(self) -> None:
        fatal = getattr(self._web, "fatal", None)
        if fatal is not None:
            fatal.set()
        self.close()
        raise SlackAuthError(
            "Slack disabled the Socket Mode link for this app (link_disabled)",
            error=_LINK_DISABLED,
        )


def _decode(raw: Any) -> dict[str, Any] | None:
    try:
        frame = json.loads(raw)
    except (TypeError, ValueError):
        logger.debug("slack socket: skipped a frame that is not JSON")
        return None
    return frame if isinstance(frame, dict) else None


def _envelope(frame: dict[str, Any] | None) -> Envelope | None:
    if frame is None:
        return None
    envelope_id, kind = frame.get("envelope_id"), frame.get("type")
    if not isinstance(envelope_id, str) or not envelope_id or not isinstance(kind, str):
        return None
    payload = frame.get("payload")
    return Envelope(envelope_id, kind, payload if isinstance(payload, dict) else {})


def _short(value: Any) -> str:
    return str(value)[:40] if value is not None else "no reason"


def _close_quietly(conn: Any) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except Exception:  # already closing or dead: that is the state we want
        logger.debug("slack socket: close raised", exc_info=True)
