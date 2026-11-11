"""The worker's end of the bus: a ``DEALER`` that connects to the dispatcher.

The link opens no listening socket. It connects, and ZeroMQ reconnects on its
own whenever the dispatcher goes away and comes back, so neither side has to
start first. Every :data:`~osprey.dispatch.bus.protocol.HEARTBEAT_SEC` it sends
``HELLO`` with the worker's whole state (the caller's ``hello`` function says
what that is). It answers each request in a task of its own through the
handler registered for that request's kind, and :meth:`WorkerLink.emit` pushes
run events without ever blocking: an event that cannot be sent at once (the
dispatcher is away, or the queue is full) is dropped, because the stream is a
live view and the run's outcome is in the agent record.

Generic on purpose: the worker's own handlers are registered from outside, so
this module knows nothing of the worker and the dispatcher image can import the
package.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import zmq
import zmq.asyncio

from osprey.dispatch.bus import protocol
from osprey.dispatch.bus.broker import is_local_endpoint
from osprey.dispatch.bus.keys import BusKeyError, ClientKeys

logger = logging.getLogger("osprey.dispatch.bus.link")

#: A request handler: the request body and blob in, the answer and an optional
#: blob out. Raise :class:`~osprey.dispatch.bus.protocol.BusError` to answer
#: ``ERR``; any other exception answers ``ERR 500`` without its text.
Handler = Callable[[dict[str, Any], bytes | None], Awaitable[tuple[Any, bytes | None]]]

#: Messages that may wait in the socket while the dispatcher is slow.
_SEND_HWM = 10_000

#: Seconds between attempts to send a heartbeat the connection could not take yet.
_JOIN_RETRY_SEC = 0.1

#: The request id at the head of a request body, as the broker writes it.
_LEADING_RID = re.compile(rb'\{"rid":"([0-9a-f]{32})"')


class WorkerLink:
    """One worker's connection to the dispatcher.

    Args:
        connect: The dispatcher's bus endpoint, e.g. ``tcp://event-dispatcher:10050``.
        name: This worker's name. Under CURVE it must be the stem of the
            certificate ``keys`` came from: the dispatcher drops a worker whose
            name and key disagree.
        handlers: The handler for each request kind this worker serves.
        hello: Returns the state each heartbeat carries (``boot_nonce``,
            ``capabilities``, ``in_flight``, ``capacity``). Called on every tick,
            which also makes it the place to prove the loop is alive.
        keys: The CURVE material, or ``None`` for an unauthenticated link, which
            is allowed only to a local endpoint.
        heartbeat_sec: Seconds between heartbeats.
        max_request_bytes: A request larger than this is answered ``ERR 413``
            without being decoded. ``None`` sets no bound.

    Raises:
        BusKeyError: ``keys`` is ``None`` and ``connect`` is not local.
    """

    def __init__(
        self,
        connect: str,
        name: str,
        handlers: Mapping[bytes, Handler],
        *,
        hello: Callable[[], dict[str, Any]],
        keys: ClientKeys | None,
        heartbeat_sec: float = protocol.HEARTBEAT_SEC,
        max_request_bytes: int | None = None,
    ) -> None:
        if keys is None and not is_local_endpoint(connect):
            raise BusKeyError(
                f"refusing to connect to {connect} without CURVE keys: an "
                "unauthenticated link reaches a local endpoint only"
            )
        self._connect = connect
        self._name = name
        self._handlers = dict(handlers)
        self._hello = hello
        self._keys = keys
        self._heartbeat_sec = heartbeat_sec
        self._max_request_bytes = max_request_bytes
        self._sock: zmq.asyncio.Socket | None = None
        self._seq: dict[str, int] = {}
        self._serving: set[asyncio.Task[None]] = set()
        #: Events dropped because they could not be sent at once.
        self.dropped_events = 0

    async def run(self) -> None:
        """Connect and serve until cancelled."""
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.ROUTING_ID, self._name.encode())
        # Queue only on a completed connection: while the dispatcher is away a
        # non-blocking send fails at once instead of piling up stale messages.
        sock.setsockopt(zmq.IMMEDIATE, 1)
        sock.setsockopt(zmq.SNDHWM, _SEND_HWM)
        if self._keys is not None:
            sock.curve_secretkey = self._keys.secret
            sock.curve_publickey = self._keys.public
            sock.curve_serverkey = self._keys.server_public
        sock.connect(self._connect)
        self._sock = sock
        logger.info("dispatch bus: %s connecting to %s", self._name, self._connect)
        heartbeat = asyncio.create_task(self._heartbeat_loop(), name="bus-link-heartbeat")
        try:
            await self._receive_loop()
        finally:
            heartbeat.cancel()
            for task in list(self._serving):
                task.cancel()
            self._sock = None
            sock.close(linger=0)
            ctx.term()

    def emit(self, run_id: str, event: dict[str, Any]) -> None:
        """Push one event of ``run_id`` to the dispatcher, never blocking.

        Safe to call from synchronous code on the link's event loop. Dropped,
        and counted in :attr:`dropped_events`, when it cannot be sent at once.
        """
        sock = self._sock
        if sock is None:
            self.dropped_events += 1
            return
        seq = self._seq.get(run_id, 0)
        if event.get("type") in protocol.TERMINAL_EVENTS:
            self._seq.pop(run_id, None)
        else:
            self._seq[run_id] = seq + 1
        frames = protocol.encode(protocol.EVENT, {"run_id": run_id, "seq": seq, "event": event})
        future = asyncio.ensure_future(sock.send_multipart(frames, flags=zmq.NOBLOCK))
        future.add_done_callback(self._count_drop)

    def _count_drop(self, future: asyncio.Future[Any]) -> None:
        if future.cancelled() or future.exception() is not None:
            self.dropped_events += 1

    async def _heartbeat_loop(self) -> None:
        assert self._sock is not None
        while True:
            try:
                state = self._hello()
            except Exception:
                logger.exception("dispatch bus: the heartbeat state could not be read")
                state = {}
            try:
                await self._sock.send_multipart(
                    protocol.encode(protocol.HELLO, {"worker": self._name, **state}),
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                # The connection is not complete yet — at start, or while the
                # dispatcher is away. Try again soon rather than a whole
                # heartbeat later, so a worker joins as soon as it can.
                await asyncio.sleep(min(self._heartbeat_sec, _JOIN_RETRY_SEC))
                continue
            await asyncio.sleep(self._heartbeat_sec)

    async def _receive_loop(self) -> None:
        assert self._sock is not None
        while True:
            frames = await self._sock.recv_multipart()
            if (
                self._max_request_bytes is not None
                and sum(len(frame) for frame in frames[1:]) > self._max_request_bytes
            ):
                rid = self._peek_rid(frames)
                if rid is not None:
                    await self._send(protocol.err(rid, 413, "Request body too large"))
                continue
            try:
                message = protocol.decode(frames)
            except protocol.ProtocolError as exc:
                logger.warning("dispatch bus: dropped a malformed message: %s", exc)
                continue
            if message.kind == protocol.PING:
                continue
            if message.kind not in protocol.REQUESTS:
                logger.warning("dispatch bus: unexpected %r from the dispatcher", message.kind)
                continue
            task = asyncio.create_task(self._serve(message))
            self._serving.add(task)
            task.add_done_callback(self._serving.discard)

    @staticmethod
    def _peek_rid(frames: list[bytes]) -> str | None:
        """The request id of an over-size request, read without decoding its body.

        The broker writes ``rid`` as the body's first key, so it is in the
        first bytes of the frame.
        """
        if len(frames) < 2:
            return None
        match = _LEADING_RID.match(frames[1])
        return match.group(1).decode() if match else None

    async def _serve(self, message: protocol.Message) -> None:
        rid = message.body.get("rid")
        if not isinstance(rid, str):
            return
        handler = self._handlers.get(message.kind)
        if handler is None:
            await self._send(protocol.err(rid, 501, f"{message.kind.decode()} not served"))
            return
        try:
            answer, blob = await handler(message.body, message.blob)
        except protocol.BusError as exc:
            await self._send(protocol.err(rid, exc.status, exc.detail))
        except Exception:
            logger.exception("dispatch bus: %s failed", message.kind.decode())
            await self._send(protocol.err(rid, 500, "internal error"))
        else:
            await self._send(protocol.ok(rid, answer, blob))

    async def _send(self, frames: list[bytes]) -> None:
        if self._sock is not None:
            await self._sock.send_multipart(frames)
