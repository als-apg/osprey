"""The server fleet service: supervise one refresher per named target, serve the snapshots (C5).

The app process runs one refresher process
(:mod:`osprey.services.channel_directory.fleet.refresher`) per refreshed target and
serves what they write to the state directory. Nothing passes between the two but
those files and the refresher's exit status (D2).

Supervision (D3), per target:

* A refresher that exits is started again: at once after connector loss
  (:data:`~.refresher.EXIT_CONNECTOR_LOST`), with exponential backoff after any
  other failure.
* A refresher that cannot build its connector
  (:data:`~.refresher.EXIT_CANNOT_BUILD`) is retried with exponential backoff; its
  ``not_refreshed`` file says why meanwhile.
* A refresher whose probe (or, with no probe, every IOC it remembers) has been
  unanswered for :data:`RECYCLE_AFTER` consecutive cycles is recycled, with
  backoff: ``wait_stale`` cannot see a transport wholly lost to one-shot reads,
  and only a fresh process cures one.
* A refresher that exits :data:`~.refresher.EXIT_NOT_REFRESHED` is not restarted.

Any answered cycle resets every backoff.

Routes, all read-only (no route accepts a write method):

* ``GET /v1/snapshot?target=<t>`` -- the target's C5 snapshot. A named target is
  read from its file (``no_snapshot`` when none is written yet); a named target
  the fleet does not refresh (D12) and a target the fleet does not name answer
  ``not_refreshed`` with the reason.
* ``GET /health`` -- 200 while the app is alive, each named target's refresher
  status in the body.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, Query

from osprey.services.channel_directory.fleet.refresher import (
    EXIT_CANNOT_BUILD,
    EXIT_CONNECTOR_LOST,
    EXIT_NOT_REFRESHED,
)
from osprey.services.channel_directory.fleet.settings import FLEET_KEY, FleetSettings, resolve_fleet
from osprey.services.channel_directory.fleet.snapshot import Outcome, Snapshot, read_snapshot
from osprey.services.channel_directory.fleet.states import ReadPath, State
from osprey.services.channel_directory.resolve import SectionError

__all__ = [
    "BACKOFF_CAP_S",
    "BACKOFF_INITIAL_S",
    "RECYCLE_AFTER",
    "RESTART_DELAY_S",
    "Backoff",
    "RefresherProcess",
    "Spawn",
    "Supervisor",
    "SupervisorStatus",
    "create_app",
    "fleet_settings",
    "refresher_command",
    "unanswered",
]

logger = logging.getLogger(__name__)

#: Consecutive unanswered cycles after which a refresher is recycled.
RECYCLE_AFTER = 3
#: The first backoff delay; each further failure doubles it.
BACKOFF_INITIAL_S = 2.0
#: The longest backoff delay.
BACKOFF_CAP_S = 300.0
#: The pause before restarting a refresher that lost its connector.
RESTART_DELAY_S = 1.0
#: Seconds a refresher is given to exit after ``terminate`` before it is killed.
TERMINATE_GRACE_S = 10.0

#: States in which an IOC's liveness PV answered.
_ANSWERED = frozenset({State.ALIVE, State.STALLED})


class SupervisorStatus(StrEnum):
    """A named target's refresher, as ``/health`` states it."""

    STARTING = "starting"
    RUNNING = "running"
    #: Exited (connector loss or a crash); waiting to start again.
    RESTARTING = "restarting"
    #: Could not build its connector; waiting, with backoff, to try again.
    CANNOT_BUILD = "cannot_build"
    #: Unanswered for several cycles; stopped, waiting, with backoff, to start again.
    RECYCLING = "recycling"
    #: The fleet does not refresh this target; no refresher runs.
    NOT_REFRESHED = "not_refreshed"
    STOPPED = "stopped"


class RefresherProcess(Protocol):
    """The part of :class:`asyncio.subprocess.Process` the supervisor uses."""

    @property
    def pid(self) -> int: ...

    @property
    def returncode(self) -> int | None: ...

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


Spawn = Callable[[str], Awaitable[RefresherProcess]]
"""``spawn(target)``: start the refresher process of *target*."""


def refresher_command(config_path: Path, target: str, state_dir: Path) -> list[str]:
    """The command line of *target*'s refresher process."""
    return [
        sys.executable,
        "-m",
        "osprey.services.channel_directory.fleet.refresher",
        "--config",
        str(config_path),
        "--target",
        target,
        "--state-dir",
        str(state_dir),
    ]


def _subprocess_spawn(config_path: Path, state_dir: Path) -> Spawn:
    async def spawn(target: str) -> RefresherProcess:
        return await asyncio.create_subprocess_exec(
            *refresher_command(config_path, target, state_dir)
        )

    return spawn


class Backoff:
    """Exponential delays: *initial*, doubled per call, never above *cap*."""

    def __init__(self, initial: float = BACKOFF_INITIAL_S, cap: float = BACKOFF_CAP_S) -> None:
        self.initial = initial
        self.cap = cap
        self._next = initial

    def next(self) -> float:
        """The delay to wait now; the following one is twice it."""
        delay = self._next
        self._next = min(self.cap, self._next * 2)
        return delay

    def reset(self) -> None:
        self._next = self.initial


def unanswered(snapshot: Snapshot) -> bool:
    """Whether *snapshot*'s cycle went unanswered (D3).

    With a probe, the probe did not answer. Without one, the target remembers
    IOCs and not one of their liveness PVs answered. An ``ok`` probe, or an
    unverified read path with no IOC to judge by, is answered.
    """
    if snapshot.outcome is not Outcome.OK or snapshot.read_path is None:
        return False
    status = snapshot.read_path.status
    if status is ReadPath.DOWN:
        return True
    if status is ReadPath.UNVERIFIED:
        return bool(snapshot.servers) and not any(
            row.state in _ANSWERED for row in snapshot.servers
        )
    return False


@dataclass(frozen=True)
class _Exited:
    code: int


@dataclass(frozen=True)
class _Recycle:
    cycles: int


class Supervisor:
    """Keeps one target's refresher running (D3).

    Args:
        target: The target refreshed.
        spawn: Starts the refresher process.
        state_dir: Where the refresher writes its snapshot.
        poll_s: Seconds between two looks at the snapshot file.
        recycle_after: Consecutive unanswered cycles before a recycle.
        backoff_initial_s: The first backoff delay.
        backoff_cap_s: The longest backoff delay.
        restart_delay_s: The pause before a restart after connector loss.
        grace_s: Seconds between ``terminate`` and ``kill`` on a recycle.
        sleep: Waits out a backoff; tests record the delays through it.
        clock: The wall clock the refresher stamps its cycles with; a cycle
            stamped before the current process started is not judged.
    """

    def __init__(
        self,
        target: str,
        *,
        spawn: Spawn,
        state_dir: Path,
        poll_s: float,
        recycle_after: int = RECYCLE_AFTER,
        backoff_initial_s: float = BACKOFF_INITIAL_S,
        backoff_cap_s: float = BACKOFF_CAP_S,
        restart_delay_s: float = RESTART_DELAY_S,
        grace_s: float = TERMINATE_GRACE_S,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.target = target
        self.state_dir = Path(state_dir)
        self.status = SupervisorStatus.STARTING
        self.reason: str | None = None
        self.starts = 0
        self.unanswered_cycles = 0
        self.last_exit: int | None = None
        self.process: RefresherProcess | None = None
        self._spawn = spawn
        self._poll_s = poll_s
        self._recycle_after = recycle_after
        self._restart_delay_s = restart_delay_s
        self._grace_s = grace_s
        self._sleep = sleep
        self._clock = clock
        self._started_at = 0.0
        self._build = Backoff(backoff_initial_s, backoff_cap_s)
        self._recycle = Backoff(backoff_initial_s, backoff_cap_s)
        self._crash = Backoff(backoff_initial_s, backoff_cap_s)
        self._last_seen: float | None = None

    def health(self) -> dict[str, Any]:
        """This target's entry in ``/health``."""
        process = self.process
        return {
            "status": self.status.value,
            "reason": self.reason,
            "pid": process.pid if process is not None and process.returncode is None else None,
            "starts": self.starts,
            "last_exit": self.last_exit,
            "unanswered_cycles": self.unanswered_cycles,
        }

    async def run(self) -> None:
        """Supervise until cancelled or the refresher says the target is not refreshed."""
        try:
            while True:
                delay = await self._run_once()
                if delay is None:
                    return
                await self._sleep(delay)
        finally:
            await self._stop_process()
            if self.status is not SupervisorStatus.NOT_REFRESHED:
                self.status = SupervisorStatus.STOPPED

    async def _run_once(self) -> float | None:
        """Start the refresher, watch it to its end; the delay before the next start."""
        self.status = SupervisorStatus.STARTING
        self.unanswered_cycles = 0
        self._started_at = self._clock()
        try:
            self.process = await self._spawn(self.target)
        except Exception as exc:
            self.process = None
            self.status = SupervisorStatus.RESTARTING
            self.reason = f"the refresher could not be started: {type(exc).__name__}: {exc}"
            delay = self._crash.next()
            logger.error("%s: %s; retrying in %gs", self.target, self.reason, delay)
            return delay
        self.starts += 1
        self.status = SupervisorStatus.RUNNING
        self.reason = None
        ended = await self._watch(self.process)
        if isinstance(ended, _Recycle):
            await self._stop_process()
            self.status = SupervisorStatus.RECYCLING
            self.reason = f"unanswered for {ended.cycles} consecutive cycles"
            logger.warning("%s: refresher recycled: %s", self.target, self.reason)
            return self._recycle.next()
        code = ended.code
        self.last_exit = code
        if code == EXIT_NOT_REFRESHED:
            self.status = SupervisorStatus.NOT_REFRESHED
            self.reason = self._file_reason() or "the refresher does not refresh this target"
            logger.info("%s: not refreshed: %s", self.target, self.reason)
            return None
        if code == EXIT_CANNOT_BUILD:
            self.status = SupervisorStatus.CANNOT_BUILD
            self.reason = self._file_reason() or "the refresher could not build its connector"
            delay = self._build.next()
            logger.warning("%s: %s; retrying in %gs", self.target, self.reason, delay)
            return delay
        self._build.reset()
        self.status = SupervisorStatus.RESTARTING
        if code == EXIT_CONNECTOR_LOST:
            self.reason = "the refresher lost its connector"
            delay = self._restart_delay_s
        else:
            self.reason = f"the refresher exited with status {code}"
            delay = self._crash.next()
        logger.warning("%s: %s; restarting in %gs", self.target, self.reason, delay)
        return delay

    async def _watch(self, process: RefresherProcess) -> _Exited | _Recycle:
        """Wait for *process* to exit, judging each new cycle it publishes."""
        waiting = asyncio.ensure_future(process.wait())
        try:
            while True:
                done, _ = await asyncio.wait({waiting}, timeout=self._poll_s)
                if done:
                    return _Exited(waiting.result())
                if self._judge_new_cycle():
                    return _Recycle(self.unanswered_cycles)
        finally:
            if not waiting.done():
                waiting.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await waiting

    def _judge_new_cycle(self) -> bool:
        """Count a newly published cycle; whether the refresher is due for a recycle."""
        snapshot = read_snapshot(self.state_dir, self.target)
        if snapshot.outcome is not Outcome.OK or snapshot.generated_at is None:
            return False
        if snapshot.generated_at < self._started_at:
            return False
        if self._last_seen is not None and snapshot.generated_at <= self._last_seen:
            return False
        self._last_seen = snapshot.generated_at
        if unanswered(snapshot):
            self.unanswered_cycles += 1
            return self.unanswered_cycles >= self._recycle_after
        self.unanswered_cycles = 0
        self._build.reset()
        self._recycle.reset()
        self._crash.reset()
        return False

    def _file_reason(self) -> str | None:
        snapshot = read_snapshot(self.state_dir, self.target)
        return snapshot.reason if snapshot.outcome is Outcome.NOT_REFRESHED else None

    async def _stop_process(self) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=self._grace_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()


def fleet_settings(config: Mapping[str, Any]) -> FleetSettings:
    """The resolved ``channel_directory.fleet`` sub-block of *config*.

    Raises:
        SectionError: The sub-block is absent or malformed.
    """
    section = config.get("channel_directory")
    if not isinstance(section, Mapping) or section.get("fleet") is None:
        raise SectionError(FLEET_KEY, "is not configured: the server fleet is off")
    return resolve_fleet(config, section["fleet"])


def create_app(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    state_dir: Path,
    spawn: Spawn | None = None,
    supervise: bool = True,
    poll_s: float | None = None,
    supervisor_options: Mapping[str, Any] | None = None,
) -> FastAPI:
    """The fleet service's app.

    Args:
        config: The rendered config.
        config_path: The rendered config's file, handed to each refresher.
        state_dir: Where the refreshers write and the routes read.
        spawn: Starts a refresher process; a subprocess by default.
        supervise: Whether the lifespan starts the refreshers.
        poll_s: Seconds between two looks at a snapshot file; half the period,
            at least one second, by default.
        supervisor_options: Further :class:`Supervisor` keyword arguments.

    Raises:
        SectionError: ``channel_directory.fleet`` is absent or malformed.
    """
    settings = fleet_settings(config)
    state_dir = Path(state_dir)
    spawner = spawn or _subprocess_spawn(Path(config_path), state_dir)
    poll = poll_s if poll_s is not None else max(1.0, settings.period_s / 2)
    supervisors = {
        target.name: Supervisor(
            target.name,
            spawn=spawner,
            state_dir=state_dir,
            poll_s=poll,
            **dict(supervisor_options or {}),
        )
        for target in settings.refreshed_targets()
    }
    control_system = config.get("control_system")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        tasks: list[asyncio.Task[None]] = []
        if supervise:
            tasks = [
                asyncio.create_task(supervisor.run(), name=f"server-fleet-{name}")
                for name, supervisor in supervisors.items()
            ]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    app = FastAPI(title="OSPREY server fleet", lifespan=lifespan)
    app.state.settings = settings
    app.state.supervisors = supervisors
    app.state.state_dir = state_dir

    def snapshot_of(target: str) -> dict[str, Any]:
        named = settings.target(target)
        if named is None:
            names = ", ".join(t.name for t in settings.targets)
            if target in _configured_targets(control_system):
                reason = f"the fleet does not refresh {target!r}; {FLEET_KEY}.targets names {names}"
            else:
                reason = f"{target!r} is not a target of this deployment"
            return Snapshot.not_ok(Outcome.NOT_REFRESHED, target=target, reason=reason).to_json()
        if not named.refreshable:
            reason = named.reason or f"the fleet does not refresh {target!r}"
            return Snapshot.not_ok(Outcome.NOT_REFRESHED, target=target, reason=reason).to_json()
        return read_snapshot(state_dir, target).to_json()

    @app.get("/v1/snapshot")
    def get_snapshot(target: str = Query(..., min_length=1)) -> dict[str, Any]:
        return snapshot_of(target)

    @app.get("/health")
    def get_health() -> dict[str, Any]:
        targets: dict[str, Any] = {}
        for named in settings.targets:
            supervisor = supervisors.get(named.name)
            if supervisor is None:
                targets[named.name] = {
                    "status": SupervisorStatus.NOT_REFRESHED.value,
                    "reason": named.reason,
                }
            else:
                targets[named.name] = supervisor.health()
        return {"status": "ok", "time": time.time(), "targets": targets}

    return app


def _configured_targets(control_system: Any) -> list[str]:
    from osprey_connectors.types import configured_targets

    try:
        return list(configured_targets(control_system))
    except Exception:  # a section the build already refused
        return []
