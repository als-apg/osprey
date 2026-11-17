"""One connector-host child, as a supervisor holds it.

Every supervisor of :mod:`osprey_connectors.ipc._host` children needs the same
things — how the child is launched, what environment it is handed, how its
``init`` frame is answered, how it is asked whether it is alive, how it is put
down, and how the requests still in flight on a child that was put down learn
why — and :class:`ConnectorHost` is where they are spelled once. The
library's multi-target :class:`~osprey_connectors.ipc.ConnectorHostPool` and
the controls MCP server's single-child manager both hold their children as
one. Ending the process itself is :mod:`osprey_connectors.process`'s, which
spells it for every child a supervisor owns; this module wraps it with what a
connector-host child's proxy needs around it.

What stays with each supervisor is what it decides: whether the child's
report is what it derived, whether a write verdict moved while the child
connected, and the sentences an operator reads when a start fails. A failed
start is a :class:`ConnectorHostStartError` whose ``stage``, ``cause``,
``returncode`` and ``phases`` say what happened, so a supervisor words it its
own way; an error the child raised answering its ``init`` frame is raised
as-is, in its own class, with a note naming the child.

Nothing here imports a control-system client library, and nothing here sets an
``EPICS_*`` variable: a supervisor is by definition a process that talks to the
control system only through its children.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import ValidationError

from osprey_connectors.connector_types import ControlTarget
from osprey_connectors.control_system._base import ChannelValue
from osprey_connectors.dotenv import ENV_CHAIN_APPLIED_ENV
from osprey_connectors.errors import (
    ConnectorInternalError,
    ConnectorUnavailableError,
    RecoveryRecord,
)
from osprey_connectors.epics_env import without_epics_env
from osprey_connectors.execution_mode import ExecutionMode
from osprey_connectors.ipc._host import (
    START_MARKS_FD_ENV,
    START_PHASES,
    HostInit,
    HostReport,
    StartPhase,
)
from osprey_connectors.ipc._proxy import (
    CHILD,
    ChildLostError,
    ChildUnresponsiveError,
    ConnectorHostProxy,
)
from osprey_connectors.model import FrozenModel
from osprey_connectors.process import DEFAULT_TERMINATE_GRACE_S, reap_exit_code, terminate

__all__ = [
    "ConnectorHost",
    "ConnectorHostError",
    "ConnectorHostStartError",
    "HostLossCause",
    "HostStartStage",
    "StartMark",
]

#: The child is always this module, run with ``-m``. No arguments: everything
#: the child needs arrives on the wire, so nothing about a deployment shows up
#: in ``ps``.
_CHILD_MODULE = "osprey_connectors.ipc._host"

#: How long, after a child is killed, the proxy's reader gets to turn the dead
#: pipe into failures on the calls that were in flight.
_SETTLE_TIMEOUT_S = 2.0

#: Where a start failed: the ``stage`` of :class:`ConnectorHostStartError`.
HostStartStage = Literal["config", "spawn", "init", "verify"]

#: Why a child went away: the ``cause`` of
#: :class:`~osprey_connectors.ipc.ConnectorHostLostError`, and of a
#: :class:`ConnectorHostStartError` whose child went silent or exited.
HostLossCause = Literal["exited", "unresponsive", "stopped", "write_timeout", "unavailable"]


def _host_env() -> dict[str, str]:
    """The environment a child is launched with: :data:`os.environ`, minus EPICS.

    The child scrubs ``EPICS_CA_*``/``EPICS_PVA_*`` again on its own first
    line, and that is the scrub the design depends on. This one is the
    defense-in-depth half: an ambient gateway never reaches the process that
    could act on it, so no window exists between exec and scrub.

    The child is stamped with
    :data:`~osprey_connectors.dotenv.ENV_CHAIN_APPLIED_ENV`. Reading its config
    file would otherwise load the project ``.env`` from its working directory
    after the scrub, and an ``EPICS_*`` line there would put back what both
    scrubs took out.

    :data:`~osprey_connectors.ipc._host.START_MARKS_FD_ENV` is dropped too: a
    supervisor that is itself running under one never passes on a descriptor
    number that names something else in its child.
    """
    env = without_epics_env(os.environ)
    env.pop(START_MARKS_FD_ENV, None)
    env[ENV_CHAIN_APPLIED_ENV] = "1"
    return env


async def _spawn(
    python: str, env: Mapping[str, str], *, start_marks_fd: int | None = None
) -> asyncio.subprocess.Process:
    """Launch one connector-host child with its frame pipes attached.

    Args:
        python: The interpreter to run the child under.
        env: The child's environment, normally :func:`_host_env`.
        start_marks_fd: The write end of a pipe the child writes its start
            marks to (see :mod:`osprey_connectors.ipc._host`), or ``None`` for
            a child that writes none. The child inherits the descriptor under
            the same number; the caller closes its own copy after the spawn.

    Returns:
        The ``asyncio.subprocess.Process``. stderr is inherited, so the child's
        diagnostics land wherever the supervisor's own do.

    Raises:
        OSError: The interpreter could not be executed.
    """
    child_env = dict(env)
    pass_fds: tuple[int, ...] = ()
    if start_marks_fd is not None:
        child_env[START_MARKS_FD_ENV] = str(start_marks_fd)
        pass_fds = (start_marks_fd,)
    return await asyncio.create_subprocess_exec(
        python,
        "-m",
        _CHILD_MODULE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        env=child_env,
        pass_fds=pass_fds,
    )


class _AttributedReader:
    """A child's stdout, able to say why reading it stopped.

    The proxy fails every outstanding request with whatever ended its read
    stream. Left to itself that is an anonymous end-of-pipe, which tells an
    operator nothing about the decision that caused it — so the supervisor
    names the reason here before it kills the child, and the proxy's
    :class:`~osprey_connectors.ipc.ChildLostError` carries it.
    """

    def __init__(self, stream: asyncio.StreamReader) -> None:
        self._stream = stream
        self._reason: str | None = None

    def retire(self, reason: str) -> None:
        """Name the reason the stream is about to end. Call before the kill."""
        self._reason = reason

    async def read(self, count: int) -> bytes:
        if self._reason is not None:
            raise ConnectionError(self._reason)
        chunk: bytes = await self._stream.read(count)
        # The usual path: the reader was already blocked here when the child was
        # retired, and the kill it was told about is what ends the read.
        if not chunk and self._reason is not None:
            raise ConnectionError(self._reason)
        return chunk


async def _put_down(
    process: asyncio.subprocess.Process,
    proxy: ConnectorHostProxy,
    name_stream_end: Callable[[str], None],
    *,
    reason: str | None,
    grace_s: float,
) -> None:
    """Put down a child, naming *reason* first, and let its proxy settle. Never raises."""
    if reason is not None:
        name_stream_end(reason)
    await terminate(process, grace_s)
    await proxy.drain(timeout_s=_SETTLE_TIMEOUT_S)
    await proxy.disconnect(ack_timeout_s=0.0)


#: Teardowns of children whose start failed, held until they finish: a caller
#: cancelled while one runs does not cut it short, and nothing collects it early.
_FAILED_START_TEARDOWNS: set[asyncio.Task[None]] = set()


# ---------------------------------------------------------------------------
# Start marks
# ---------------------------------------------------------------------------


class StartMark(FrozenModel):
    """One start mark as a failed start reports it: the phase, and when after the spawn."""

    phase: StartPhase
    after_s: float


class _StartMarkPipe:
    """The pipe a starting child writes its start marks to.

    Opened only where a child can inherit a pipe by descriptor; elsewhere there
    is no descriptor to pass and nothing is ever read.
    """

    def __init__(self) -> None:
        self._read_fd: int | None = None
        self.write_fd: int | None = None
        self._written = b""
        if os.name == "posix":
            self._read_fd, self.write_fd = os.pipe()
            os.set_blocking(self._read_fd, False)

    def close_write(self) -> None:
        """Close this process's copy of the write end."""
        if self.write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self.write_fd)
            self.write_fd = None

    def close(self) -> None:
        """Close both ends."""
        self.close_write()
        if self._read_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._read_fd)
            self._read_fd = None

    def read(self, spawned_at: float) -> tuple[StartMark, ...]:
        """Every whole mark written so far, timed from *spawned_at*."""
        while self._read_fd is not None:
            try:
                chunk = os.read(self._read_fd, 4096)
            except OSError:
                # Nothing more is written yet (BlockingIOError), or the pipe is unusable.
                break
            if not chunk:
                break
            self._written += chunk
        whole = self._written[: self._written.rfind(b"\n") + 1]
        marks: list[StartMark] = []
        for line in whole.decode("ascii", "replace").splitlines():
            name, _, at = line.partition(" ")
            phase = next((known for known in START_PHASES if known == name), None)
            if phase is None:
                continue
            try:
                marks.append(StartMark(phase=phase, after_s=float(at) - spawned_at))
            except ValueError:
                continue
        return tuple(marks)


def _describe_start(phases: tuple[StartMark, ...]) -> str:
    """One sentence on how far a child got in starting, from the marks it wrote."""
    if not phases:
        return "It had not finished starting its interpreter and loading the connector host."
    last, seconds = phases[-1].phase, phases[-1].after_s
    reached = {
        "main": f"It was up after {seconds:.1f}s and had not read its init frame.",
        "init": (
            f"It read its init frame after {seconds:.1f}s and was importing the connector modules."
        ),
        "imports": (
            f"It had imported the connector modules after {seconds:.1f}s and was building "
            "its connector (connect())."
        ),
        "connected": f"It had built its connector after {seconds:.1f}s and had not answered.",
    }[last]
    listed = ", ".join(f"{mark.phase} +{mark.after_s:.1f}s" for mark in phases)
    return f"{reached} Start marks: {listed}"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConnectorHostError(ConnectorUnavailableError):
    """A connector-host child could not serve a call; the pool's failures say how.

    See :class:`~osprey_connectors.ipc.ConnectorHostPool` for each of them.

    Attributes:
        target: The control target the child serves. Always a resolved one:
            :meth:`~osprey_connectors.ipc.ConnectorHostPool.connector` refuses a
            target that does not resolve with a ``ValueError`` before any child
            is started.
        execution_mode: ``"readonly"`` or ``None``.
        pid: The child's process id, or ``None`` when none was started.
    """

    def __init__(
        self,
        message: str,
        *,
        target: ControlTarget,
        execution_mode: ExecutionMode | None,
        pid: int | None,
        recovery: tuple[RecoveryRecord, ...] = (),
    ) -> None:
        super().__init__(message, recovery=recovery)
        self.target = target
        self.execution_mode = execution_mode
        self.pid = pid


class ConnectorHostStartError(ConnectorHostError):
    """A child could not be brought up. Nothing was left running.

    Attributes:
        stage: ``"config"`` (an unresolved placeholder, a Channel Access
            type with no gateway to pin it to, or a simulated or stand-in
            target that would dial an endpoint ``live`` derives), ``"spawn"`` (the
            interpreter could not be started), ``"init"`` (no answer within the
            start timeout, an exit before answering, or an unusable answer) or
            ``"verify"`` (the report did not match the derivation).
        phases: The start marks the child had written when an ``init``-stage
            failure was raised, as :class:`StartMark` records in the order of
            :data:`~osprey_connectors.ipc._host.START_PHASES`; empty for every
            other stage, and for a child that wrote none.
        cause: For an ``init``-stage failure, ``"unresponsive"`` (no answer
            within the start timeout) or ``"exited"`` (the child closed its
            output stream before answering); ``None`` for an answer that was
            not a post-connect report, and for every other stage.
        returncode: The exit status of a child that exited before answering,
            when it was reaped; otherwise ``None``.

    ``__cause__`` is the failure under the start's: the ``OSError`` of a spawn,
    the link error of a silent or exited child, and for an unusable answer a
    :class:`~osprey_connectors.errors.ConnectorInternalError` whose
    ``class_tag`` names the answer's type.
    """

    def __init__(
        self,
        message: str,
        *,
        target: ControlTarget,
        execution_mode: ExecutionMode | None,
        pid: int | None,
        stage: HostStartStage,
        phases: tuple[StartMark, ...] = (),
        cause: HostLossCause | None = None,
        returncode: int | None = None,
    ) -> None:
        super().__init__(message, target=target, execution_mode=execution_mode, pid=pid)
        self.stage = stage
        self.phases = phases
        self.cause = cause
        self.returncode = returncode


def _default_label(init: HostInit) -> str:
    """How a failure names a child started on *init*: its target and execution mode."""
    mode = f" ({init.execution_mode})" if init.execution_mode else ""
    return f"target {init.target!r}{mode}"


def _post_connect_report(reply: object) -> HostReport:
    """*reply* to an ``init`` frame as the :class:`HostReport` it must be.

    Raises:
        ConnectorInternalError: *reply* is not a post-connect report; its
            ``class_tag`` names the reply's type.
    """
    try:
        return HostReport.model_validate(reply)
    except ValidationError as exc:
        raise ConnectorInternalError(
            f"the {CHILD} answered 'init' with a {type(reply).__name__} that is not "
            f"the post-connect report: {exc}",
            class_tag=type(reply).__name__,
        ) from exc


# ---------------------------------------------------------------------------
# The handle
# ---------------------------------------------------------------------------


class ConnectorHost:
    """One connector-host child, started and answering: its process, proxy and report.

    Built by :meth:`start`, which returns only a child that answered its
    ``init`` frame with a post-connect report. What the report must say is the
    supervisor's to judge: a supervisor that refuses it calls :meth:`retire`.
    Calls go through :attr:`proxy`; a supervisor that wants a child gone calls
    :meth:`stop` for a healthy one and :meth:`retire` for one it gives up on.
    Neither raises.

    Args:
        process: The child's ``asyncio.subprocess.Process``.
        proxy: The proxy over its pipes.
        report: Its post-connect report.
        name_stream_end: Names, before a kill, why the child's output stream
            is about to end, so the proxy fails what is in flight with it.
        terminate_grace_s: Time between ``SIGTERM`` and ``SIGKILL``.
    """

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        proxy: ConnectorHostProxy,
        report: HostReport,
        *,
        name_stream_end: Callable[[str], None],
        terminate_grace_s: float = DEFAULT_TERMINATE_GRACE_S,
    ) -> None:
        self._process = process
        self._proxy = proxy
        self._report = report
        self._name_stream_end = name_stream_end
        self._terminate_grace_s = terminate_grace_s

    @classmethod
    async def start(
        cls,
        init: HostInit,
        *,
        python: str = sys.executable,
        env: Mapping[str, str] | None = None,
        start_timeout_s: float,
        call_timeout_s: float | None = None,
        timeout_grace_s: float = 1.0,
        terminate_grace_s: float = DEFAULT_TERMINATE_GRACE_S,
        start_marks: bool = False,
        label: str | None = None,
    ) -> ConnectorHost:
        """Spawn a child, send it *init*, and return it once it has answered.

        Args:
            init: What the child is started for; sent as its ``init`` frame.
            python: The interpreter the child runs under.
            env: The child's environment; ``None`` for this process's own,
                minus every ``EPICS_CA_*``/``EPICS_PVA_*`` variable.
            start_timeout_s: How long the child gets to answer its ``init`` frame.
            call_timeout_s: The proxy's local deadline for a call that names no
                timeout of its own; ``None`` waits as long as the child takes.
            timeout_grace_s: How long past a call's own ``timeout_s`` the proxy
                waits before it gives up locally.
            terminate_grace_s: Time between ``SIGTERM`` and ``SIGKILL`` when the
                child is put down, and how long a child that closed its output
                stream gets to exit before it is.
            start_marks: Whether the child writes its start marks, so an
                ``init``-stage failure says how far it got (``phases``).
            label: How a failure's message names the child; by default its
                target and execution mode (``target 'live' (readonly)``).

        Returns:
            The child, with its post-connect report.

        Raises:
            ConnectorHostStartError: The child could not be spawned
                (``stage="spawn"``), or did not answer its ``init`` frame with a
                post-connect report (``stage="init"``). Nothing is left running.
            Exception: The child's own error answering its ``init`` frame, in
                the class it raised, with a note naming the child. Nothing is
                left running.
        """
        named = _default_label(init) if label is None else label
        target, mode = init.target, init.execution_mode
        marks = _StartMarkPipe() if start_marks else None
        spawned_at = time.monotonic()
        process = None
        try:
            process = await _spawn(
                python,
                _host_env() if env is None else env,
                start_marks_fd=marks.write_fd if marks is not None else None,
            )
        except OSError as exc:
            raise ConnectorHostStartError(
                f"Could not spawn a connector-host child for {named}: {exc}",
                target=target,
                execution_mode=mode,
                pid=None,
                stage="spawn",
            ) from exc
        finally:
            if marks is not None:
                # The child holds the only write end from here on, so the pipe
                # ends when the child closes it or exits.
                marks.close_write()
                if process is None:
                    marks.close()

        # _spawn attaches both pipes.
        assert process.stdout is not None and process.stdin is not None
        reader = _AttributedReader(process.stdout)
        proxy = ConnectorHostProxy(
            reader,
            process.stdin,
            timeout_grace_s=timeout_grace_s,
            call_timeout_s=call_timeout_s,
        )
        pid = int(process.pid)

        def failure(
            detail: str, *, cause: HostLossCause | None = None, returncode: int | None = None
        ) -> ConnectorHostStartError:
            phases = marks.read(spawned_at) if marks is not None else ()
            reached = f" {_describe_start(phases)}" if marks is not None else ""
            return ConnectorHostStartError(
                f"The connector-host child (pid {pid}) for {named} {detail}{reached}",
                target=target,
                execution_mode=mode,
                pid=pid,
                stage="init",
                phases=phases,
                cause=cause,
                returncode=returncode,
            )

        async def answer() -> object:
            """The child's reply to its init frame; a silent or lost child is a start failure."""
            try:
                return await proxy._request(
                    "init", init.model_dump(exclude_none=True), timeout_s=start_timeout_s
                )
            except ChildUnresponsiveError as exc:
                raise failure(
                    f"did not answer its init frame within {start_timeout_s}s.",
                    cause="unresponsive",
                ) from exc
            except ChildLostError as exc:
                # The pipe closed, so the child is exiting: its status is
                # reaped before anything signals it. A signal sent to an
                # exited, unreaped child reaps it out from under the event
                # loop's watcher, which then reports exit code 255. The
                # pipe is known closed here, so this wait runs the full
                # grace on purpose, longer than terminate's own window.
                await reap_exit_code(process, terminate_grace_s)
                await terminate(process, terminate_grace_s)
                raise failure(
                    f"exited before answering its init frame (exit code "
                    f"{process.returncode}): {exc}",
                    cause="exited",
                    returncode=process.returncode,
                ) from exc
            except Exception as exc:
                # Anything else is the child's own answer to its init frame.
                exc.add_note(f"raised by the connector-host child (pid {pid}) for {named}")
                raise

        try:
            reply = await answer()
            try:
                report = _post_connect_report(reply)
            except ConnectorInternalError as exc:
                raise failure(
                    f"answered its init frame with {type(reply).__name__}, "
                    "not the post-connect report."
                ) from exc
        except BaseException:
            teardown = asyncio.get_running_loop().create_task(
                _put_down(
                    process,
                    proxy,
                    reader.retire,
                    reason=f"The connector-host child for {named} failed to start.",
                    grace_s=terminate_grace_s,
                )
            )
            _FAILED_START_TEARDOWNS.add(teardown)
            teardown.add_done_callback(_FAILED_START_TEARDOWNS.discard)
            await asyncio.shield(teardown)
            raise
        finally:
            if marks is not None:
                marks.close()
        return cls(
            process,
            proxy,
            report,
            name_stream_end=reader.retire,
            terminate_grace_s=terminate_grace_s,
        )

    @property
    def pid(self) -> int:
        """The child's process id."""
        return int(self._process.pid)

    @property
    def proxy(self) -> ConnectorHostProxy:
        """The connector-shaped handle every call to the child goes through."""
        return self._proxy

    @property
    def report(self) -> HostReport:
        """The child's post-connect report, as it answered its ``init`` frame."""
        return self._report

    @property
    def returncode(self) -> int | None:
        """The child's exit status once it has exited and been reaped, else ``None``."""
        return self._process.returncode

    def usable(self) -> bool:
        """Whether the child is running and its pipe is still open."""
        return self._process.returncode is None and self._proxy.dead_reason is None

    async def ping(self, *, timeout_s: float) -> bool:
        """Whether the child answers a ping within *timeout_s*. Never raises.

        A child that missed a call's deadline but answers is alive and merely
        slow; one that does not answer may be wedged. The ping is served
        alongside any number of slow calls, so only a stuck event loop in the
        child leaves it unanswered.
        """
        try:
            await self._proxy._request("ping", {}, timeout_s=timeout_s)
        except Exception:
            return False
        return True

    async def probe(self, address: str, *, timeout_s: float) -> ChannelValue:
        """Read *address* in the child under a hard bound: its readiness probe.

        Returns:
            The channel's reading.

        Raises:
            ChildUnresponsiveError: The child did not answer within *timeout_s*
                plus the proxy's grace.
            ChildLostError: The child closed its output stream; it is given
                the terminate grace to exit first, so :attr:`returncode` names
                its exit status when it did.
            ConnectorInternalError: The child answered with something other
                than a reading.
            Exception: The child's own error for the read, in its own class.
        """
        try:
            reply = await self._proxy._request(
                "spawn_probe", {"address": address, "timeout_s": timeout_s}, timeout_s=timeout_s
            )
        except ChildLostError:
            await reap_exit_code(self._process, self._terminate_grace_s)
            raise
        if not isinstance(reply, ChannelValue):
            raise ConnectorInternalError(
                f"the {CHILD} answered 'spawn_probe' with a {type(reply).__name__}",
                class_tag=type(reply).__name__,
            )
        return reply

    async def retire(self, reason: str | None) -> None:
        """Put down a child given up on, and let its proxy settle. Never raises.

        Args:
            reason: Why, named on the child's output stream before the kill so
                every request in flight fails with it; ``None`` names nothing.
        """
        await _put_down(
            self._process,
            self._proxy,
            self._name_stream_end,
            reason=reason,
            grace_s=self._terminate_grace_s,
        )

    async def stop(self) -> None:
        """Stop a healthy child in order: acknowledge, then make sure it is gone. Never raises."""
        await self._proxy.disconnect()
        await terminate(self._process, self._terminate_grace_s)
