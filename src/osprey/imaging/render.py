"""Parent side of the isolated picture render worker.

:func:`render_isolated` turns picture bytes into a rendition by handing them to
one worker subprocess per process (:mod:`osprey.imaging.render_worker`). The
worker is spawned on the first call -- never at import -- with an empty
environment, ``cwd='/'`` and pipes of its own, and is closed after
:data:`RENDER_WORKER_IDLE_S` seconds with no task or after
:data:`WORKER_MAX_TASKS` tasks.

Submissions are serialised by an :class:`asyncio.Lock`; the per-task clock
(:data:`RENDER_TASK_TIMEOUT_S`) starts once the lock is held and the worker is
ready. Every reply is checked against fixed types and ranges and the rendition
bytes are re-sniffed; a reply that fails the check, a timeout, or a worker that
ends mid-task is a *worker failure*: the worker is killed and the task retried
once in a freshly spawned worker. The outcomes are:

* a reply that passes the check -- returned as is (a rendition, or a content
  refusal from :data:`~osprey.imaging.formats.CONTENT_SKIP_REASONS`);
* the same id failing in the fresh worker too -- ``decoder_failed``: the fresh
  worker answered ``ready``, so the picture, not the worker, is at fault;
* a spawn with no ``ready`` line within :data:`RENDER_READY_TIMEOUT_S`, or two
  consecutive worker failures on two different ids -- :class:`RenderUnavailable`,
  an infrastructure failure for which no skip reason is recorded.

The cached worker, its lock and its idle timer belong to one event loop; a call
from another loop kills that worker and starts afresh.

The module attributes :data:`WORKER_ARGV`, :data:`RENDER_TASK_TIMEOUT_S`,
:data:`RENDER_WORKER_IDLE_S`, :data:`RENDER_READY_TIMEOUT_S` and
:data:`WORKER_MAX_TASKS` are read at call time.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import signal
import struct
import sys
from dataclasses import dataclass
from typing import Any

from osprey.imaging.formats import (
    ACCEPTED,
    CONTENT_SKIP_REASONS,
    RENDITION_MAX_BYTES,
    RENDITION_MAX_SIDE,
    RENDITION_MODES,
    sniff,
)

WORKER_ARGV: tuple[str, ...] = (sys.executable, "-I", "-m", "osprey.imaging.render_worker")
"""Command line of the worker; ``-I`` with ``env={}`` blocks any path injection."""

RENDER_TASK_TIMEOUT_S: float = 30.0
"""Wall-clock bound of one task, counted from lock acquisition on a ready worker."""

RENDER_WORKER_IDLE_S: float = 120.0
"""Seconds with no task after which the worker is closed."""

RENDER_READY_TIMEOUT_S: float = 10.0
"""Seconds a freshly spawned worker has to write its ``ready`` line."""

WORKER_MAX_TASKS: int = 100
"""Tasks after which the worker is closed and the next call spawns a fresh one."""

CLOSE_GRACE_S: float = 2.0
"""Seconds a worker has to exit on end of input before it is killed."""

RENDITION_MIMES: frozenset[str] = frozenset({"image/png", "image/jpeg"})

_LENGTH = struct.Struct(">I")


class RenderUnavailable(RuntimeError):
    """The render worker cannot be run; no content verdict was reached.

    Attributes:
        exit_code: The worker's exit status when known (negative for a signal),
            else ``None``.
    """

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class _WorkerFailure(Exception):
    """A reply that breaks the protocol, or a worker that ended mid-exchange."""


@dataclass(frozen=True)
class Rendition:
    """A checked rendition as returned by the worker."""

    data: bytes
    mime: str
    format: str
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class RenderOutcome:
    """Result of :func:`render_isolated`: a rendition, or a content skip reason."""

    rendition: Rendition | None
    reason: str | None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_reply(header: Any, body: bytes) -> RenderOutcome:
    """Validate one reply against the protocol; raise :class:`_WorkerFailure` if it breaks it."""
    if not isinstance(header, dict):
        raise _WorkerFailure("reply header is not an object")
    ok = header.get("ok")
    if not isinstance(ok, bool):
        raise _WorkerFailure("reply 'ok' is not a boolean")
    reason = header.get("reason")
    if not ok:
        if not isinstance(reason, str) or reason not in CONTENT_SKIP_REASONS:
            raise _WorkerFailure(f"reply reason {reason!r} is not a content skip reason")
        if body:
            raise _WorkerFailure("refusal carries rendition bytes")
        return RenderOutcome(None, reason)

    if reason is not None:
        raise _WorkerFailure("successful reply carries a reason")
    fmt, mode, mime = header.get("format"), header.get("mode"), header.get("mime")
    width, height = header.get("w"), header.get("h")
    if not isinstance(fmt, str) or fmt not in ACCEPTED:
        raise _WorkerFailure(f"reply format {fmt!r} is not accepted")
    if not isinstance(mode, str) or mode not in RENDITION_MODES:
        raise _WorkerFailure(f"reply mode {mode!r} is not a rendition mode")
    if not isinstance(mime, str) or mime not in RENDITION_MIMES:
        raise _WorkerFailure(f"reply mime {mime!r} is not a rendition type")
    if not (_is_int(width) and _is_int(height)):
        raise _WorkerFailure(f"reply size {width!r}x{height!r} is not integral")
    assert isinstance(width, int) and isinstance(height, int)
    if not (1 <= width <= RENDITION_MAX_SIDE and 1 <= height <= RENDITION_MAX_SIDE):
        raise _WorkerFailure(f"reply size {width}x{height} is out of range")
    if not 0 < len(body) <= RENDITION_MAX_BYTES:
        raise _WorkerFailure(f"rendition of {len(body)} bytes is out of range")
    if sniff(body).mime != mime:
        raise _WorkerFailure(f"rendition bytes do not match {mime}")
    return RenderOutcome(Rendition(body, mime, fmt, width, height, mode), None)


# -- one worker process ---------------------------------------------------------------


@dataclass
class _Worker:
    process: asyncio.subprocess.Process
    tasks: int = 0

    @property
    def alive(self) -> bool:
        return self.process.returncode is None

    async def exchange(self, data: bytes) -> RenderOutcome:
        stdin, stdout = self.process.stdin, self.process.stdout
        if stdin is None or stdout is None:  # pragma: no cover - spawned with pipes
            raise _WorkerFailure("worker has no pipes")
        stdin.write(_LENGTH.pack(len(data)) + data)
        await stdin.drain()
        line = await stdout.readline()
        if not line.endswith(b"\n"):
            raise _WorkerFailure("worker ended before replying")
        header = json.loads(line)
        (length,) = _LENGTH.unpack(await stdout.readexactly(_LENGTH.size))
        if length > RENDITION_MAX_BYTES:
            raise _WorkerFailure(f"rendition of {length} bytes is out of range")
        body = await stdout.readexactly(length)
        return _check_reply(header, body)

    def kill(self) -> None:
        # A reaped worker's pid may already belong to another process.
        if self.process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(self.process.pid, signal.SIGKILL)

    def drop(self) -> None:
        """Kill the worker and close its stdin without awaiting (cancellation path)."""
        self.kill()
        if self.process.stdin is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self.process.stdin.close()

    async def close(self, *, kill: bool) -> int | None:
        """End the worker -- killed, or by end of input with a grace period -- and reap it."""
        if kill:
            self.kill()
        # Closing stdin is the end of input, and lets the transport finish once
        # the process has exited.
        if self.process.stdin is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), CLOSE_GRACE_S)
        except TimeoutError:
            self.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.process.wait(), CLOSE_GRACE_S)
        # Let the pipe transports run their close callbacks.
        await asyncio.sleep(0)
        return self.process.returncode


async def _spawn() -> _Worker:
    """Start a worker and wait for its ``ready`` line; raise :class:`RenderUnavailable` if none."""
    try:
        process = await asyncio.create_subprocess_exec(
            *WORKER_ARGV,
            env={},
            cwd="/",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        raise RenderUnavailable(f"render worker could not be started: {exc}") from exc

    worker = _Worker(process)
    problem = "wrote no ready line"
    try:
        assert process.stdout is not None
        line = await asyncio.wait_for(process.stdout.readline(), RENDER_READY_TIMEOUT_S)
        if line:
            handshake = json.loads(line)
            if isinstance(handshake, dict) and handshake.get("ready") is True:
                return worker
            problem = "wrote an invalid ready line"
        else:
            problem = "exited before its ready line"
    except TimeoutError:
        problem = f"wrote no ready line within {RENDER_READY_TIMEOUT_S:g} s"
    except (ValueError, OSError):
        problem = "wrote an invalid ready line"
    except BaseException:
        # Cancelled while waiting: the fresh worker is owned by nobody, so end it.
        worker.drop()
        raise
    exit_code = await worker.close(kill=True if problem.startswith("wrote") else False)
    raise RenderUnavailable(f"render worker {problem} (exit code {exit_code})", exit_code)


# -- the per-loop client ----------------------------------------------------------------


class _Client:
    """The cached worker, its lock and its idle timer, owned by one event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.lock = asyncio.Lock()
        self.worker: _Worker | None = None
        self.last_failure_id: str | None = None
        self.generation = 0
        self._idle_handle: asyncio.TimerHandle | None = None
        self._closers: set[asyncio.Task[None]] = set()

    # idle timer

    def cancel_idle(self) -> None:
        if self._idle_handle is not None:
            self._idle_handle.cancel()
            self._idle_handle = None

    def arm_idle(self) -> None:
        self.cancel_idle()
        if self.worker is not None:
            self._idle_handle = self.loop.call_later(
                RENDER_WORKER_IDLE_S, self._on_idle, self.generation
            )

    def _on_idle(self, generation: int) -> None:
        self._idle_handle = None
        task = self.loop.create_task(self._close_idle(generation))
        self._closers.add(task)
        task.add_done_callback(self._closers.discard)

    async def _close_idle(self, generation: int) -> None:
        async with self.lock:
            if generation == self.generation and self.worker is not None:
                await self._discard(kill=False)

    # worker lifetime

    async def _discard(self, *, kill: bool) -> int | None:
        worker, self.worker = self.worker, None
        if worker is None:
            return None
        return await worker.close(kill=kill)

    async def _ready_worker(self) -> _Worker:
        if self.worker is not None and not self.worker.alive:
            await self._discard(kill=True)
        if self.worker is None:
            try:
                self.worker = await _spawn()
            except RenderUnavailable:
                self.last_failure_id = None
                raise
        return self.worker

    def abandon(self) -> None:
        """Drop the worker without the (possibly closed) loop: kill it by pid."""
        self.cancel_idle()
        if self.worker is not None:
            self.worker.kill()
            self.worker = None

    async def render(self, data: bytes, task_id: str) -> RenderOutcome:
        self.generation += 1
        for _attempt in range(2):
            worker = await self._ready_worker()
            try:
                outcome = await asyncio.wait_for(worker.exchange(data), RENDER_TASK_TIMEOUT_S)
            except asyncio.CancelledError:
                # The worker may be mid-reply or hold a half-written frame; a
                # later call must never read this picture's reply as its own.
                # A cancel is not a failure of the picture.
                self.worker = None
                worker.drop()
                raise
            except (TimeoutError, _WorkerFailure, OSError, EOFError, ValueError):
                exit_code = await self._discard(kill=True)
                if self.last_failure_id is not None and self.last_failure_id != task_id:
                    self.last_failure_id = None
                    raise RenderUnavailable(
                        "render worker failed on two different pictures in a row "
                        f"(exit code {exit_code})",
                        exit_code,
                    ) from None
                self.last_failure_id = task_id
                continue
            self.last_failure_id = None
            worker.tasks += 1
            if worker.tasks >= WORKER_MAX_TASKS:
                await self._discard(kill=False)
            return outcome
        return RenderOutcome(None, "decoder_failed")


_CLIENT: _Client | None = None


def _client() -> _Client:
    global _CLIENT
    loop = asyncio.get_running_loop()
    if _CLIENT is None or _CLIENT.loop is not loop:
        if _CLIENT is not None:
            _CLIENT.abandon()
        _CLIENT = _Client(loop)
    return _CLIENT


def _forget_worker() -> None:
    """Kill any cached worker and forget the client (process teardown and tests)."""
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.abandon()
        _CLIENT = None


async def close_render_worker() -> None:
    """Close the cached worker by end of input and forget it (shutdown and tests).

    A worker owned by another (possibly closed) loop is killed by pid instead.
    """
    global _CLIENT
    client = _CLIENT
    if client is None:
        return
    _CLIENT = None
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if client.loop is not running:
        client.abandon()
        return
    async with client.lock:
        client.cancel_idle()
        await client._discard(kill=False)


def worker_pid() -> int | None:
    """Pid of the cached worker, or ``None`` when none is running."""
    if _CLIENT is None or _CLIENT.worker is None:
        return None
    return _CLIENT.worker.process.pid


async def render_isolated(data: bytes, *, task_id: str | None = None) -> RenderOutcome:
    """Render picture bytes in the isolated worker.

    Args:
        data: The picture bytes.
        task_id: Identity of the picture for the failure rules; defaults to the
            sha256 of ``data``.

    Returns:
        A :class:`RenderOutcome` with the rendition, or with a content skip
        reason (``decoder_failed`` when the picture failed in two workers).

    Raises:
        RenderUnavailable: No worker could be started, or workers failed on two
            different pictures in a row.
    """
    picture_id = task_id if task_id is not None else hashlib.sha256(data).hexdigest()
    client = _client()
    async with client.lock:
        client.cancel_idle()
        try:
            return await client.render(data, picture_id)
        finally:
            client.arm_idle()


async def probe_render_worker() -> bool:
    """Spawn a worker exactly as :func:`render_isolated` does and close it again.

    The cached worker is left untouched. This is the probe behind
    ``attachments.render`` in ``osprey ariel status``.

    Returns:
        ``True`` when the worker answered ``ready``, else ``False``.
    """
    try:
        worker = await _spawn()
    except RenderUnavailable:
        return False
    await worker.close(kill=False)
    return True
