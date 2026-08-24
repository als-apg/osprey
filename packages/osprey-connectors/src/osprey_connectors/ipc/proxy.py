"""Parent-side proxy for a connector living in the connector-host child.

A tool that reads or writes a channel does not care where the control-system
client library lives. :class:`ConnectorHostProxy` is what makes that true: it
offers the connector call surface tools actually use, and serves each call by
writing a request frame down a pipe and awaiting the matching reply from the
child that owns the real connector. The parent process therefore never imports
``epics``/``p4p`` — which is the entire reason the child exists, since those
libraries cannot be unloaded once loaded and pin the parent to one gateway for
its lifetime. That property is pinned by a test which imports this module in a
fresh interpreter and asserts none of them appeared in ``sys.modules``.

Why this is not a ``ControlSystemConnector`` subclass
----------------------------------------------------
It is a duck type with *exactly* the base class's signatures, not a subclass,
and deliberately so.
:class:`~osprey_connectors.control_system.base.ControlSystemConnector` uses
``__init_subclass__`` to wrap every subclass's ``write_channel`` /
``write_multiple_channels`` in the ``writes_enabled`` pre-check. That guard
belongs *at the connector*, and the connector is in the child: the child
already applies it, in the process that will do the write, against the config
that process was launched with. Inheriting it here would run the check a second
time in the parent — turning a refusal into a fabricated failure result
manufactured on this side of the boundary rather than the real one from the
child, and silently coupling the answer to whichever config the parent happens
to hold. Subclassing would also drag in ``connect()``, ``subscribe()``,
``unsubscribe()``, ``get_metadata()`` and ``validate_channel()`` as abstract
members the proxy has no wire method for. Mirroring the signatures gives tools
substitutability; not inheriting keeps enforcement in exactly one place. A test
compares the signatures parameter-by-parameter so the mirror cannot drift.

``subscribe`` is absent on purpose: nothing in the codebase calls it, and a
proxy method that could not deliver callbacks across the boundary would be a
lie about the surface.

Concurrency
-----------
One background reader task owns the read side of the pipe and demultiplexes
replies to per-request futures by ``request_id``, so several calls may be in
flight at once and replies may arrive in any order. Nothing blocks the pipe
while a call waits.

Failure is fail-closed
----------------------
EOF on the pipe, a stream that will not decode, or the reader task dying for
any other reason ends the proxy: every outstanding call completes with a
:class:`ConnectionError` naming the connector-host child as the cause, and
every later call raises the same. A dead proxy never revives — starting a
fresh child and a fresh proxy is the supervisor's job, not this object's.

Draining, for the target switch
-------------------------------
:meth:`refuse_new_requests` closes the door to *new* calls while leaving the
in-flight ones alone, and :meth:`drain` waits for those to finish with a
deadline. Together they are the handoff surface the runtime target switch uses:
refuse, drain, then disconnect the old child once the last call has landed (or
once the deadline says it never will).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult
from osprey_connectors.ipc import frames

__all__ = ["CHILD", "ConnectorHostProxy"]

logger = logging.getLogger("osprey_connectors.ipc.proxy")

#: How the child is named in every failure message it causes. Operators read
#: these; "connection lost" without a subject is not an actionable sentence.
CHILD = "connector-host child"

#: Read size for the pipe. A frame is reassembled by ``frames.FrameReader``, so
#: this only trades syscalls against buffer size.
_READ_CHUNK = 65536


class ConnectorHostProxy:
    """A connector-shaped handle on the connector running in the child process.

    Args:
        reader: Anything with ``await read(n) -> bytes`` returning ``b""`` at
            EOF — an :class:`asyncio.StreamReader`, and therefore an
            ``asyncio.subprocess.Process.stdout``.
        writer: Anything with ``write(bytes)``, ``await drain()`` and
            ``close()`` — an :class:`asyncio.StreamWriter`, and therefore an
            ``asyncio.subprocess.Process.stdin``. ``wait_closed()`` is used
            when present.
        timeout_grace_s: Extra time the proxy waits beyond a call's own
            ``timeout`` before giving up locally. The child applies the timeout
            to the control-system call itself and reports its own
            ``TimeoutError``, which is the more informative one; the local
            deadline exists only so a wedged child cannot hang a caller
            forever. Set it to ``0`` to make the local deadline the only one.

    The transport is duck-typed rather than a subprocess handle so tests can
    drive the proxy over a socketpair, and so the supervisor can hand it a
    process's pipes without this class knowing anything about process
    lifetime. :meth:`from_process` is the convenience for the latter.
    """

    def __init__(
        self,
        reader: Any,
        writer: Any,
        *,
        timeout_grace_s: float = 1.0,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._timeout_grace_s = timeout_grace_s
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        #: Set once the pipe is gone; every later call repeats it.
        self._dead_reason: str | None = None
        #: Set by refuse_new_requests()/disconnect(); in-flight calls continue.
        self._refusal: str | None = None
        self._disconnected = False

    @classmethod
    def from_process(cls, process: Any, **kwargs: Any) -> ConnectorHostProxy:
        """Wrap an ``asyncio.subprocess.Process``'s stdin/stdout pipes.

        The proxy does not own the process: it neither starts it nor reaps it.
        Killing and restarting the child belongs to the supervisor.
        """
        if process.stdout is None or process.stdin is None:
            raise ValueError(
                f"the {CHILD} must be spawned with stdin and stdout pipes for the proxy to use"
            )
        return cls(process.stdout, process.stdin, **kwargs)

    # ------------------------------------------------------------------
    # Connector call surface (signatures mirror ControlSystemConnector)
    # ------------------------------------------------------------------

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """Read one channel. Returns the ``ChannelValue`` the child produced."""
        return await self._call(
            "read_channel",
            {"channel_address": channel_address, "timeout": timeout},
            timeout=timeout,
        )

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        """Read many channels in **one** request frame.

        The batch crosses the boundary whole rather than as N round trips: the
        child's connector decides how to fan it out, exactly as it would
        in-process, and one pipe round trip covers the lot.
        """
        return await self._call(
            "read_multiple_channels",
            {"channel_addresses": list(channel_addresses), "timeout": timeout},
            timeout=timeout,
        )

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        verification_level: str | None = None,
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
        """Write one channel. Returns the ``ChannelWriteResult`` from the child.

        A refusal (limits, writes-disabled) arrives as a typed exception frame
        and is re-raised here as the real exception class with its fields
        intact, so a caller cannot tell the refusal came from another process.
        """
        kwargs: dict[str, Any] = {
            "channel_address": channel_address,
            "value": value,
            "timeout": timeout,
        }
        return await self._call(
            "write_channel", _with_verification(kwargs, verification_level, tolerance), timeout
        )

    async def write_multiple_channels(
        self,
        operations: list[tuple[str, Any]],
        timeout: float | None = None,
        verification_level: str | None = None,
        tolerance: float | None = None,
    ) -> list[ChannelWriteResult]:
        """Write many channels in one request frame.

        The whole batch goes over as a single request so the child's connector
        keeps whatever transactional behaviour it implements — a simulator that
        suppresses recalculation between writes, for instance, cannot do that
        if the parent has already split the batch into separate calls.
        """
        kwargs: dict[str, Any] = {
            "operations": [list(operation) for operation in operations],
            "timeout": timeout,
        }
        return await self._call(
            "write_multiple_channels",
            _with_verification(kwargs, verification_level, tolerance),
            timeout,
        )

    async def disconnect(self, *, ack_timeout: float = 2.0) -> None:
        """Ask the child to release its connector, then close the pipe.

        Idempotent and never raises: it is called from teardown paths and from
        the switch's failure branch, where a second exception would only bury
        the first. The acknowledgement is best-effort — a child that has
        already died, or that never answers, still leaves this method returning
        with the pipe closed and every outstanding call failed.
        """
        if self._disconnected:
            return
        self._disconnected = True
        if self._refusal is None:
            self._refusal = f"the {CHILD} has been disconnected"

        if self._dead_reason is None:
            request_id = frames.new_request_id()
            # The acknowledgement is advisory: everything here is best-effort,
            # and the pipe closes below whatever happened.
            with contextlib.suppress(Exception):
                future = self._register(request_id)
                self._ensure_reader_task()
                await self._send(frames.encode_request(request_id, "disconnect", {}))
                await asyncio.wait_for(asyncio.shield(future), ack_timeout)
            self._pending.pop(request_id, None)

        await self._shutdown(f"the {CHILD} was disconnected before this request completed")

    # ------------------------------------------------------------------
    # Handoff surface consumed by the runtime target switch
    # ------------------------------------------------------------------

    def refuse_new_requests(self, reason: str) -> None:
        """Stop accepting new calls; let the in-flight ones finish.

        ``reason`` is what a caller sees in the :class:`ConnectionError` raised
        by any later call, so it should say what is happening ("target switch
        in progress"), not merely that something is closed. Calling this twice
        keeps the first reason — the first one explains the decision.
        """
        if self._refusal is None:
            self._refusal = reason

    async def drain(self, timeout: float) -> bool:
        """Wait for outstanding requests to finish.

        Returns ``True`` when nothing is left in flight, ``False`` when the
        deadline expired with calls still pending. It never raises and never
        cancels anything: a caller that gets ``False`` decides for itself
        whether to wait longer or to abandon the child.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout, 0.0)
        while True:
            outstanding = [future for future in self._pending.values() if not future.done()]
            if not outstanding:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            _, still_pending = await asyncio.wait(outstanding, timeout=remaining)
            if still_pending:
                return False

    # ------------------------------------------------------------------
    # Request/response plumbing
    # ------------------------------------------------------------------

    async def _call(self, method: str, kwargs: dict[str, Any], timeout: float | None) -> Any:
        """Send one request and await its reply."""
        self._raise_if_unusable()
        self._ensure_reader_task()

        request_id = frames.new_request_id()
        future = self._register(request_id)
        try:
            await self._send(frames.encode_request(request_id, method, kwargs))
            return await self._await_reply(future, method, timeout)
        finally:
            self._pending.pop(request_id, None)

    async def _await_reply(
        self, future: asyncio.Future[Any], method: str, timeout: float | None
    ) -> Any:
        if timeout is None:
            return await future
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout + max(self._timeout_grace_s, 0.0)
            )
        except TimeoutError:
            # A TimeoutError with the future still pending is *our* deadline; one
            # with the future resolved is the child's own, which is the better
            # error and is re-raised untouched.
            if future.done():
                raise
            raise TimeoutError(f"the {CHILD} did not answer {method!r} within {timeout}s") from None

    def _register(self, request_id: str) -> asyncio.Future[Any]:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        return future

    async def _send(self, payload: bytes) -> None:
        try:
            self._writer.write(payload)
            await self._writer.drain()
        except (OSError, RuntimeError, AttributeError, ConnectionError) as exc:
            reason = f"writing to the {CHILD} failed: {exc}"
            await self._shutdown(reason)
            raise ConnectionError(reason) from exc

    def _raise_if_unusable(self) -> None:
        if self._dead_reason is not None:
            raise ConnectionError(self._dead_reason)
        if self._refusal is not None:
            raise ConnectionError(self._refusal)

    def _ensure_reader_task(self) -> None:
        if self._reader_task is None or self._reader_task.done():
            self._reader_task = asyncio.get_running_loop().create_task(self._read_loop())

    async def _read_loop(self) -> None:
        """Demultiplex replies until the pipe or the frame stream gives out."""
        stream = frames.FrameReader()
        try:
            while True:
                chunk = await self._reader.read(_READ_CHUNK)
                if not chunk:
                    await self._fail_all(f"the {CHILD} closed its output stream")
                    return
                for frame in stream.feed(chunk):
                    self._dispatch(frame)
        except asyncio.CancelledError:
            raise
        except ConnectionError as exc:
            # A transport that raises ConnectionError is telling this proxy why
            # its stream ended — the supervisor's reader does exactly that when
            # it kills a child for a target switch. That sentence is better than
            # anything this layer could write about it, so it is passed through
            # verbatim rather than wrapped in a description of the stream.
            await self._fail_all(str(exc) or f"the {CHILD} closed its output stream")
        except Exception as exc:
            await self._fail_all(f"the {CHILD} sent an unreadable reply stream: {exc}")

    def _dispatch(self, frame: Any) -> None:
        if isinstance(frame, frames.RequestFrame):
            # The parent issues requests; it never serves them. A request
            # arriving here means the stream is not what it claims to be.
            raise frames.FrameDecodeError(
                f"the {CHILD} sent a request frame for {frame.method!r} on the reply stream"
            )
        future = self._pending.get(frame.request_id)
        if future is None or future.done():
            # A reply to a call that already timed out or was abandoned. Dropping
            # it is correct — there is no one left to hand it to.
            logger.debug("discarding unmatched reply for request %s", frame.request_id)
            return
        if isinstance(frame, frames.ErrorFrame):
            # decode_frame already ran the typed-exception registry, so this is
            # the real class with its fields, or a ConnectionError standing in
            # for a class this build does not know.
            future.set_exception(frame.exception)
        else:
            future.set_result(frame.value)

    async def _fail_all(self, reason: str) -> None:
        """End the proxy: nobody waits forever, and nobody gets a fresh chance."""
        if self._dead_reason is None:
            self._dead_reason = reason
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(ConnectionError(self._dead_reason))

    async def _shutdown(self, reason: str) -> None:
        await self._fail_all(reason)
        task, self._reader_task = self._reader_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        with contextlib.suppress(Exception):
            self._writer.close()
        wait_closed = getattr(self._writer, "wait_closed", None)
        if wait_closed is not None:
            with contextlib.suppress(Exception):
                await wait_closed()


def _with_verification(
    kwargs: dict[str, Any], verification_level: str | None, tolerance: float | None
) -> dict[str, Any]:
    """Add the verification keywords only when the caller actually supplied them.

    ``None`` is the connector contract's *omission* sentinel, not a fourth
    verification level: a caller with no opinion leaves the keyword off so the
    connector resolves the level and tolerance for that specific channel.
    Forwarding ``None`` over the wire would hand the child an explicit
    ``verification_level=None`` and override a connector whose own default is
    something else, so the keys are left out instead.
    """
    if verification_level is not None:
        kwargs["verification_level"] = verification_level
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    return kwargs
