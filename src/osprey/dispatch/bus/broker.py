"""The dispatcher's end of the bus: one ``ROUTER`` that brokers runs across workers.

Workers connect; the broker never dials anyone. What it knows of a worker it
learns from that worker's heartbeats (``HELLO``), each of which carries the
worker's whole state, so a broker that restarted rebuilds its view from the
next round with no handshake of its own.

Three tables:

- **workers** — every worker heard from within the liveness window, with the
  runs it holds in flight. A worker silent for longer is expired: its pending
  requests fail with :class:`WorkerGoneError` and the streams of its unfinished
  runs end with a synthetic ``worker lost`` error.
- **owners** — which worker owns which run, learned from the reply to
  ``DISPATCH`` and from heartbeats. Kept after a run ends, because in the
  isolated workspace mode only the owner can serve its artifacts.
- **streams** — a bounded buffer of each run's events, which the worker
  pushes as they happen. Any number of viewers subscribe, each with a replay
  from the first buffered event.

Delivery is at most once. The broker never re-sends a dispatch on its own: an
agent run has side effects, and whether to try again is the trigger's
``on_error`` policy, as it is over HTTP.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import zmq
import zmq.asyncio

from osprey.dispatch.bus import protocol
from osprey.dispatch.bus.keys import BusKeyError, NamedCurveAuthenticator, ServerKeys

logger = logging.getLogger("osprey.dispatch.bus.broker")

#: Events kept per run for replay to a viewer who subscribes late.
STREAM_BUFFER = 1000
#: Seconds a finished run's stream stays subscribable.
STREAM_TTL_SEC = 60.0
#: Most run owners remembered. The oldest are forgotten first; a call for a
#: forgotten run is scattered to every live worker instead.
MAX_OWNERS = 10_000

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class WorkerGoneError(Exception):
    """No worker could take the request: none with room connected within the
    wait, or the one asked expired or could not be addressed before it replied."""


class StreamGoneError(Exception):
    """The run is known but its stream has been discarded."""


def is_local_endpoint(endpoint: str) -> bool:
    """Whether ``endpoint`` cannot be reached from another host.

    ``ipc://`` and ``inproc://`` are always local; ``tcp://`` is local when its
    host is a loopback address.
    """
    scheme, _, rest = endpoint.partition("://")
    if scheme in ("ipc", "inproc"):
        return True
    if scheme != "tcp":
        return False
    host = urlsplit(f"//{rest}").hostname or ""
    return host in _LOOPBACK_HOSTS


@dataclass
class WorkerState:
    """What the broker knows of one worker."""

    name: str
    boot_nonce: float | None
    capabilities: tuple[str, ...]
    capacity: int | None
    in_flight: set[str]
    last_seen: float
    last_dispatch: float = 0.0
    #: Dispatches sent to this worker that have not been answered yet.
    reserved: int = 0
    #: The routing id of the connection this worker last spoke on, which is
    #: where requests for it go. Under CURVE it is one the socket assigned, so
    #: no peer can claim it by name.
    route: bytes = b""

    @property
    def load(self) -> int:
        return len(self.in_flight) + self.reserved

    def has_room(self) -> bool:
        return self.capacity is None or self.load < self.capacity

    def describe(self, now: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "boot_nonce": self.boot_nonce,
            "capabilities": list(self.capabilities),
            "capacity": self.capacity,
            "in_flight": len(self.in_flight),
            "last_seen_sec": round(now - self.last_seen, 1),
        }


@dataclass
class _RunStream:
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=STREAM_BUFFER))
    subscribers: set[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=set)
    ended_at: float | None = None

    def push(self, event: dict[str, Any], now: float) -> None:
        if self.ended_at is not None:
            return
        self.events.append(event)
        for queue in self.subscribers:
            queue.put_nowait(event)
        if event.get("type") in protocol.TERMINAL_EVENTS:
            self.ended_at = now
            for queue in self.subscribers:
                queue.put_nowait(None)


class WorkerBus:
    """The broker. Create it, ``await start()`` on the serving loop, ``await stop()`` at exit.

    Args:
        bind: The endpoint to bind, e.g. ``tcp://0.0.0.0:10050``. A port of
            ``*`` binds an ephemeral one; :attr:`endpoint` then names it.
        keys: The CURVE material. ``None`` runs the bus without authentication,
            which is allowed only on a local endpoint (see
            :func:`is_local_endpoint`) and is meant for tests and a developer's
            own machine.
        heartbeat_sec: Seconds between the workers' heartbeats; a worker silent
            for :data:`~osprey.dispatch.bus.protocol.LIVENESS` of them is expired.

    Raises:
        BusKeyError: ``keys`` is ``None`` and ``bind`` is not local.
    """

    def __init__(
        self,
        bind: str,
        *,
        keys: ServerKeys | None,
        heartbeat_sec: float = protocol.HEARTBEAT_SEC,
    ) -> None:
        if keys is None and not is_local_endpoint(bind):
            raise BusKeyError(
                f"refusing to bind {bind} without CURVE keys: an unauthenticated "
                "bus binds a loopback, ipc:// or inproc:// endpoint only"
            )
        self._bind = bind
        self._keys = keys
        self._heartbeat_sec = heartbeat_sec
        self._expiry_sec = heartbeat_sec * protocol.LIVENESS

        self._ctx: zmq.asyncio.Context | None = None
        self._sock: zmq.asyncio.Socket | None = None
        self._auth: NamedCurveAuthenticator | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self.endpoint: str = bind

        self._workers: dict[str, WorkerState] = {}
        self._owners: OrderedDict[str, str] = OrderedDict()
        self._streams: OrderedDict[str, _RunStream] = OrderedDict()
        self._pending: dict[str, tuple[str, asyncio.Future[protocol.Message]]] = {}
        self._changed = asyncio.Event()

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        """Bind the socket and start the receive and expiry loops."""
        ctx = zmq.asyncio.Context()
        self._ctx = ctx
        sock = ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        # A send to a peer that is not connected raises instead of vanishing,
        # so a request to a worker that just left fails at once.
        sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        # Without keys a worker's routing id is its name; one that reconnects
        # takes it over at once instead of waiting for its old connection to
        # time out. Under CURVE the socket assigns routing ids, so this never
        # applies there.
        sock.setsockopt(zmq.ROUTER_HANDOVER, 1)
        if self._keys is not None:
            auth = NamedCurveAuthenticator(ctx, self._keys)
            auth.start()
            self._auth = auth
            sock.curve_secretkey = self._keys.secret
            sock.curve_publickey = self._keys.public
            sock.curve_server = True
        sock.bind(self._bind)
        self.endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self._sock = sock
        self._tasks = [
            asyncio.create_task(self._receive_loop(), name="bus-receive"),
            asyncio.create_task(self._expiry_loop(), name="bus-expiry"),
        ]
        logger.info(
            "dispatch bus bound at %s (%s)",
            self.endpoint,
            "CURVE" if self._keys is not None else "unauthenticated, local only",
        )

    async def stop(self) -> None:
        """Stop the loops, fail every pending request and close the socket."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        for name in list(self._workers):
            self._expire(name, "dispatcher stopping")
        if self._auth is not None:
            self._auth.stop()
            self._auth = None
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

    # -- what the broker knows -------------------------------------------------

    def live(self) -> list[str]:
        """Names of the workers heard from within the liveness window."""
        return sorted(self._workers)

    def describe(self) -> list[dict[str, Any]]:
        """One row per live worker, for ``/health``."""
        now = time.monotonic()
        return [self._workers[name].describe(now) for name in self.live()]

    def capabilities(self) -> list[str]:
        """The capabilities every live worker advertises; empty when none is live."""
        sets = [set(state.capabilities) for state in self._workers.values()]
        return sorted(set.intersection(*sets)) if sets else []

    def owner_of(self, run_id: str) -> str | None:
        """The worker that owns ``run_id``, when the broker knows it."""
        return self._owners.get(run_id)

    # -- requests --------------------------------------------------------------

    async def request(
        self,
        worker: str,
        kind: bytes,
        body: dict[str, Any],
        *,
        timeout: float,
    ) -> protocol.Message:
        """Send one request to ``worker`` and wait for its reply.

        Returns:
            The ``OK`` reply. Its ``body["body"]`` is the answer, its ``blob``
            any bytes that came with it.

        Raises:
            protocol.BusError: The worker answered ``ERR``.
            WorkerGoneError: The worker is not connected, or expired before it
                answered.
            TimeoutError: No answer within ``timeout``.
        """
        state = self._workers.get(worker)
        if state is None or self._sock is None:
            raise WorkerGoneError(f"worker {worker!r} is not connected")
        rid = protocol.new_rid()
        future: asyncio.Future[protocol.Message] = asyncio.get_running_loop().create_future()
        self._pending[rid] = (worker, future)
        try:
            try:
                await self._sock.send_multipart(
                    # ``rid`` leads the body, so a worker can refuse an
                    # over-size request without parsing it.
                    [state.route, *protocol.encode(kind, {"rid": rid, **body})]
                )
            except zmq.ZMQError as exc:
                raise WorkerGoneError(f"worker {worker!r} cannot be reached: {exc}") from exc
            reply = await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(rid, None)
        return reply

    async def dispatch(
        self, request: dict[str, Any], *, wait: float, timeout: float
    ) -> tuple[str, dict[str, Any]]:
        """Send a dispatch to the least-loaded worker with room for it.

        Waits up to ``wait`` seconds for such a worker to connect, so a trigger
        that fires before any worker is up waits instead of failing.

        Returns:
            The worker's name and its answer (``{"status", "run_id"}``).

        Raises:
            WorkerGoneError: No worker with room connected within ``wait``, or
                as :meth:`request`.
            protocol.BusError, TimeoutError: As :meth:`request`.
        """
        worker = await self._pick(wait)
        state = self._workers[worker]
        state.reserved += 1
        state.last_dispatch = time.monotonic()
        try:
            reply = await self.request(
                worker, protocol.DISPATCH, {"request": request}, timeout=timeout
            )
        finally:
            state.reserved -= 1
        answer = reply.body.get("body")
        if not isinstance(answer, dict) or not isinstance(answer.get("run_id"), str):
            raise protocol.BusError(502, "dispatch reply carries no run_id")
        run_id = answer["run_id"]
        self._own(run_id, worker)
        state.in_flight.add(run_id)
        self._streams.setdefault(run_id, _RunStream())
        return worker, answer

    async def _pick(self, wait: float) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while True:
            candidates = [s for s in self._workers.values() if s.has_room()]
            if candidates:
                return min(candidates, key=lambda s: (s.load, s.last_dispatch, s.name)).name
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise WorkerGoneError(
                    "no dispatch worker with room connected"
                    if self._workers
                    else "no dispatch worker is connected"
                )
            self._changed.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._changed.wait(), remaining)

    # -- streams ---------------------------------------------------------------

    async def stream(
        self, run_id: str, *, idle: float = 1.0
    ) -> AsyncIterator[dict[str, Any] | None]:
        """The events of ``run_id``: the buffered ones first, then each as it arrives.

        Yields ``None`` after every ``idle`` seconds without an event, so a
        caller can send a keepalive. Ends after the run's ``done`` or ``error``.

        Raises:
            KeyError: The broker knows nothing of the run.
            StreamGoneError: The run is known but its stream was discarded.
        """
        run_stream = self._streams.get(run_id)
        if run_stream is None:
            if run_id in self._owners:
                raise StreamGoneError(run_id)
            raise KeyError(run_id)
        replay = list(run_stream.events)
        if run_stream.ended_at is not None:
            for event in replay:
                yield event
            return
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        run_stream.subscribers.add(queue)
        try:
            for event in replay:
                yield event
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), idle)
                except TimeoutError:
                    yield None
                    continue
                if item is None:
                    return
                yield item
        finally:
            run_stream.subscribers.discard(queue)

    # -- loops -----------------------------------------------------------------

    async def _receive_loop(self) -> None:
        assert self._sock is not None
        while True:
            frames = await self._sock.recv_multipart(copy=False)
            try:
                self._handle(frames)
            except Exception:
                logger.exception("dispatch bus: a message could not be handled")

    def _handle(self, frames: list[zmq.Frame]) -> None:
        if len(frames) < 3:
            logger.warning("dispatch bus: dropped a message of %d frame(s)", len(frames))
            return
        route = frames[0].bytes
        if self._keys is not None:
            # The key decides who a peer is, never the routing id: a peer can
            # pick its own routing id, and with ROUTER_HANDOVER even take one
            # over from a live connection. pyzmq reads a metadata property by
            # name; its stub types only the integer frame options.
            user_id = frames[1].get("User-Id")  # type: ignore[arg-type]
            if not isinstance(user_id, str) or not user_id:
                logger.warning("dispatch bus: dropped a message with no authenticated name")
                return
            name = user_id
        else:
            name = route.decode(errors="replace")
        try:
            message = protocol.decode([frame.bytes for frame in frames[1:]])
        except protocol.ProtocolError as exc:
            logger.warning("dispatch bus: dropped a malformed message from %r: %s", name, exc)
            return
        now = time.monotonic()
        if message.kind == protocol.HELLO:
            self._hello(name, route, message.body, now)
            return
        state = self._workers.get(name)
        if state is not None:
            state.last_seen = now
            state.route = route
        if message.kind in (protocol.OK, protocol.ERR):
            self._reply(name, message)
        elif message.kind == protocol.EVENT:
            self._event(name, message.body, now)
        else:
            logger.warning("dispatch bus: %r sent an unexpected %r", name, message.kind)

    def _hello(self, name: str, route: bytes, body: dict[str, Any], now: float) -> None:
        nonce = body.get("boot_nonce")
        nonce = float(nonce) if isinstance(nonce, int | float) else None
        state = self._workers.get(name)
        if state is not None and nonce is not None and state.boot_nonce != nonce:
            self._expire(name, "restarted")
            state = None
        in_flight = {r for r in body.get("in_flight") or [] if isinstance(r, str)}
        capabilities = tuple(c for c in body.get("capabilities") or [] if isinstance(c, str))
        capacity = body.get("capacity")
        capacity = capacity if isinstance(capacity, int) and capacity > 0 else None
        if state is None:
            state = WorkerState(name, nonce, capabilities, capacity, in_flight, now, route=route)
            self._workers[name] = state
            logger.info(
                "dispatch bus: worker %r joined (%d run(s) in flight)", name, len(in_flight)
            )
        else:
            state.capabilities = capabilities
            state.capacity = capacity
            state.in_flight = in_flight
            state.last_seen = now
            state.route = route
        for run_id in in_flight:
            self._own(run_id, name)
            self._streams.setdefault(run_id, _RunStream())
        self._changed.set()

    def _reply(self, name: str, message: protocol.Message) -> None:
        rid = message.body.get("rid")
        entry = self._pending.get(rid) if isinstance(rid, str) else None
        if entry is None:
            logger.debug("dispatch bus: a reply from %r matches no request", name)
            return
        worker, future = entry
        if worker != name:
            logger.warning("dispatch bus: %r answered a request sent to %r; dropped", name, worker)
            return
        if future.done():
            return
        if message.kind == protocol.OK:
            future.set_result(message)
        else:
            status = message.body.get("status")
            detail = message.body.get("detail")
            future.set_exception(
                protocol.BusError(
                    status if isinstance(status, int) else 500,
                    detail if isinstance(detail, str) else "",
                )
            )

    def _event(self, name: str, body: dict[str, Any], now: float) -> None:
        run_id = body.get("run_id")
        event = body.get("event")
        if not isinstance(run_id, str) or not isinstance(event, dict):
            return
        self._own(run_id, name)
        self._streams.setdefault(run_id, _RunStream()).push(event, now)
        if event.get("type") in protocol.TERMINAL_EVENTS:
            state = self._workers.get(name)
            if state is not None:
                state.in_flight.discard(run_id)
                self._changed.set()

    async def _expiry_loop(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_sec)
            now = time.monotonic()
            for name, state in list(self._workers.items()):
                if now - state.last_seen > self._expiry_sec:
                    self._expire(name, f"silent for {now - state.last_seen:.1f}s")
            self._trim_streams(now)

    # -- bookkeeping -----------------------------------------------------------

    def _expire(self, name: str, reason: str) -> None:
        state = self._workers.pop(name, None)
        if state is None:
            return
        logger.warning(
            "dispatch bus: worker %r expired (%s); %d run(s) lost",
            name,
            reason,
            len(state.in_flight),
        )
        for worker, future in self._pending.values():
            if worker == name and not future.done():
                future.set_exception(WorkerGoneError(f"worker {name!r} {reason}"))
        now = time.monotonic()
        lost = {"type": "error", "message": f"worker lost: {name} {reason}"}
        for run_id in state.in_flight:
            run_stream = self._streams.get(run_id)
            if run_stream is not None:
                run_stream.push(lost, now)
        self._changed.set()

    def _own(self, run_id: str, worker: str) -> None:
        self._owners[run_id] = worker
        self._owners.move_to_end(run_id)
        while len(self._owners) > MAX_OWNERS:
            self._owners.popitem(last=False)

    def _trim_streams(self, now: float) -> None:
        for run_id, run_stream in list(self._streams.items()):
            ended = run_stream.ended_at
            if ended is not None and now - ended > STREAM_TTL_SEC and not run_stream.subscribers:
                del self._streams[run_id]
        while len(self._streams) > MAX_OWNERS:
            self._streams.popitem(last=False)
