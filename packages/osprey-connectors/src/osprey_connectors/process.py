"""Starting and putting down the child processes a supervisor owns.

Every child a supervisor starts, whether a long-lived connector host or a
short job worker, is put down the same way, and this module is where that way
is spelled once. :func:`terminate` puts down one child; :class:`JobSlots`
keeps one current :class:`ChildJob` per name and classifies how each ended as
a :class:`ChildExit`. Four invariants hold throughout:

1. Nothing signals a child before the event loop has had the chance to reap
   its exit.
2. Every exit code reported comes from a reap.
3. A cause is recorded before the signal that causes it.
4. Only the current job's completion is applied: a caller checks
   :meth:`JobSlots.is_current` before acting on a job's exit, and a job that
   is no longer current never spawns.

The connector-host pool reports its own cause words for a child it lost.
They map onto :class:`ExitCause` as ``stopped`` → ``cancelled``,
``unresponsive`` and ``write_timeout`` → ``timed_out``, and ``exited`` →
``lost``.

It imports only the standard library and knows nothing about the control
system its callers talk to.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "DEFAULT_TERMINATE_GRACE_S",
    "REAP_WINDOW_S",
    "ChildExit",
    "ChildJob",
    "ExitCause",
    "JobSlots",
    "reap_exit_code",
    "terminate",
]

#: How long a child gets between ``SIGTERM`` and ``SIGKILL``.
DEFAULT_TERMINATE_GRACE_S = 2.0

#: Longest a supervisor waits for the event loop to collect a child that may
#: already have exited, before it treats the child as running. Collection
#: follows the exit within milliseconds; the window only bounds the cost to a
#: child that is in fact still running.
REAP_WINDOW_S = 0.5


async def reap_exit_code(process: Any, window_s: float) -> int | None:
    """Wait up to *window_s* for the event loop to collect *process*'s exit.

    Returns the exit code, or ``None`` if the child is still running when the
    window closes. Never signals the child and never raises.
    """
    if process.returncode is None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), window_s)
    returncode: int | None = process.returncode
    return returncode


async def terminate(process: Any, grace_s: float) -> int | None:
    """``SIGTERM``, then ``SIGKILL`` after the grace period. Never raises.

    A child that already exited is collected by the event loop before anything
    signals it: a signal sent to an exited, uncollected child reaps it out from
    under the loop's watcher, which then loses the child's own exit code and
    reports 255. The wait for that collection is bounded by
    ``min(grace_s, REAP_WINDOW_S)``, so it only delays putting down a child
    that is in fact still running.

    Returns:
        The reaped exit code, or ``None`` if the child outlived ``SIGKILL``'s
        grace period.
    """
    if await reap_exit_code(process, min(grace_s, REAP_WINDOW_S)) is not None:
        return _returncode(process)
    with contextlib.suppress(OSError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), grace_s)
        return _returncode(process)
    except TimeoutError:
        pass
    with contextlib.suppress(OSError):
        process.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait(), grace_s)
    return _returncode(process)


def _returncode(process: Any) -> int | None:
    returncode: int | None = process.returncode
    return returncode


class ExitCause(StrEnum):
    """Why a child ended."""

    #: Exited 0 on its own.
    COMPLETED = "completed"
    #: Exited non-zero on its own.
    FAILED = "failed"
    #: Put down because its supervisor no longer wanted it.
    CANCELLED = "cancelled"
    #: Put down because it ran past its deadline or stopped answering.
    TIMED_OUT = "timed_out"
    #: Exited while its supervisor still depended on it.
    LOST = "lost"


@dataclass(frozen=True)
class ChildExit:
    """How one job ended."""

    #: The job's id, minted by :meth:`JobSlots.reserve`.
    job: int
    cause: ExitCause
    #: The reaped exit code, or ``None`` if the job never spawned or its child
    #: outlived ``SIGKILL``'s grace period.
    returncode: int | None
    #: The last bytes the child wrote to stderr, when the job collected them.
    stderr_tail: str


#: One id sequence for every slot set in the process, so a job id names one
#: job however many :class:`JobSlots` exist.
_JOB_IDS = itertools.count(1)


class ChildJob:
    """One child process run as a job in a :class:`JobSlots` slot.

    Never constructed by a caller: :meth:`JobSlots.reserve` makes it, and the
    id it carries exists before the child does, so a caller can hand the id to
    the child on its command line.
    """

    def __init__(self, slots: JobSlots, name: str, job: int) -> None:
        self._slots = slots
        self._name = name
        self._job = job
        #: The ``asyncio.subprocess.Process``, ``None`` until started.
        self.process: Any = None
        #: The child's pid, ``None`` until started.
        self.pid: int | None = None
        self._asked: ExitCause | None = None
        self._starting: asyncio.Event | None = None
        self._tail = bytearray()
        self._drain: asyncio.Task[None] | None = None
        self._exit: ChildExit | None = None
        self._concluding = asyncio.Lock()

    @property
    def job(self) -> int:
        """The job's id, unique in this process."""
        return self._job

    @property
    def name(self) -> str:
        """The slot this job runs in."""
        return self._name

    def asked(self, cause: ExitCause) -> None:
        """Record why the job is about to be put down. The first call wins."""
        if self._asked is None:
            self._asked = cause

    async def start(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        stdin: Any = asyncio.subprocess.DEVNULL,
        stdout: Any = asyncio.subprocess.DEVNULL,
        collect_stderr: int = 0,
    ) -> None:
        """Spawn the job's child once every earlier job in its slot is reaped.

        A job that is no longer current once those are reaped, or that was
        asked to stop meanwhile, spawns nothing and ends as
        :attr:`ExitCause.CANCELLED`.

        Args:
            argv: The command line, program first.
            env: The child's environment; the supervisor's own when ``None``.
            cwd: The child's working directory.
            stdin: As for :func:`asyncio.create_subprocess_exec`.
            stdout: As for :func:`asyncio.create_subprocess_exec`.
            collect_stderr: When positive, stderr is drained from spawn to end
                of file and its last *collect_stderr* bytes become the exit's
                ``stderr_tail``, so a full pipe never blocks the child.
                Otherwise stderr is discarded.

        Raises:
            RuntimeError: The job was already started.
            OSError: The program could not be executed.
        """
        if self._starting is not None:
            raise RuntimeError(f"job {self._job} was already started")
        self._starting = asyncio.Event()
        try:
            for earlier in self._slots._earlier_jobs(self):
                await earlier.stop(ExitCause.CANCELLED)
            if (
                not self._slots.is_current(self)
                or self._asked is not None
                or self._exit is not None
            ):
                self.asked(ExitCause.CANCELLED)
                return
            self.process = await asyncio.create_subprocess_exec(
                *argv,
                env=None if env is None else dict(env),
                cwd=cwd,
                stdin=stdin,
                stdout=stdout,
                stderr=asyncio.subprocess.PIPE
                if collect_stderr > 0
                else asyncio.subprocess.DEVNULL,
            )
            self.pid = self.process.pid
            if collect_stderr > 0:
                self._drain = asyncio.get_running_loop().create_task(
                    self._collect(self.process.stderr, collect_stderr)
                )
        finally:
            self._starting.set()

    async def stop(
        self,
        cause: ExitCause = ExitCause.CANCELLED,
        grace_s: float = DEFAULT_TERMINATE_GRACE_S,
    ) -> ChildExit:
        """Put the job down for *cause* and return its one exit."""
        self.asked(cause)
        await self._started()
        return await self._conclude(grace_s)

    async def wait(
        self,
        deadline_s: float | None = None,
        grace_s: float = DEFAULT_TERMINATE_GRACE_S,
    ) -> ChildExit:
        """Wait for the job to end and return its one exit.

        Past *deadline_s* the job is put down as :attr:`ExitCause.TIMED_OUT`.
        An exit with no recorded cause is classified by its code: 0 is
        :attr:`ExitCause.COMPLETED`, anything else :attr:`ExitCause.FAILED`.
        A job that never spawned ends as :attr:`ExitCause.CANCELLED`.
        """
        await self._started()
        if self._exit is None and self.process is not None:
            try:
                await asyncio.wait_for(self.process.wait(), deadline_s)
            except TimeoutError:
                self.asked(ExitCause.TIMED_OUT)
        return await self._conclude(grace_s)

    async def _started(self) -> None:
        if self._starting is not None:
            await self._starting.wait()

    async def _conclude(self, grace_s: float) -> ChildExit:
        async with self._concluding:
            if self._exit is not None:
                return self._exit
            returncode: int | None = None
            if self.process is not None:
                returncode = await terminate(self.process, grace_s)
            if self._drain is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(self._drain), REAP_WINDOW_S)
                self._drain.cancel()
            if self.process is None:
                cause = ExitCause.CANCELLED
            elif self._asked is not None:
                cause = self._asked
            else:
                cause = ExitCause.COMPLETED if returncode == 0 else ExitCause.FAILED
            self._exit = ChildExit(
                self._job, cause, returncode, self._tail.decode("utf-8", errors="replace")
            )
            self._slots._forget(self)
            return self._exit

    async def _collect(self, stream: asyncio.StreamReader, keep: int) -> None:
        while chunk := await stream.read(65536):
            self._tail += chunk
            del self._tail[:-keep]


class JobSlots:
    """One current job per name.

    Reserving a name makes the new job current at once; the job it replaces is
    put down before the new one spawns.
    """

    def __init__(self) -> None:
        self._current: dict[str, ChildJob] = {}
        self._live: dict[str, list[ChildJob]] = {}

    def reserve(self, name: str) -> ChildJob:
        """Mint a job for *name* and make it the current one."""
        job = ChildJob(self, name, next(_JOB_IDS))
        self._current[name] = job
        self._live.setdefault(name, []).append(job)
        return job

    def is_current(self, job: ChildJob) -> bool:
        """Whether *job* is still the current job of its name."""
        return self._current.get(job.name) is job

    def current(self, name: str) -> ChildJob | None:
        """The current job of *name*, if any."""
        return self._current.get(name)

    async def stop_all(self, cause: ExitCause = ExitCause.CANCELLED) -> list[ChildExit]:
        """Make no job current and put every unfinished job down concurrently.

        Returns once every one of them is reaped, with their exits.
        """
        self._current.clear()
        jobs = [job for jobs in self._live.values() for job in jobs]
        return list(await asyncio.gather(*(job.stop(cause) for job in jobs)))

    def _earlier_jobs(self, job: ChildJob) -> list[ChildJob]:
        return [other for other in self._live.get(job.name, []) if other.job < job.job]

    def _forget(self, job: ChildJob) -> None:
        jobs = self._live.get(job.name, [])
        if job in jobs:
            jobs.remove(job)
        if not jobs:
            self._live.pop(job.name, None)
