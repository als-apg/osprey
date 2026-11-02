"""The watchdog runner: one prompt, every enabled watchdog, one read of the record.

:meth:`WatchdogRunner.on_prompt` reads the session's record nodes once and hands
them to :meth:`WatchdogRunner.on_nodes`, which runs each enabled watchdog as its
own asyncio task, so a hung judge costs only itself. Both methods schedule and
return a task at once: a request handler drops it, while the record entry and
replay await it for the verdicts. Each run ends the same way: a flag when the
verdict is flagged and the caller can hold, one recorded verdict, and a
change-notice frame for a flag or an unavailable verdict.

Deadlines are ``time.monotonic`` values. The prompt deadline bounds the record
read and the decision engine. An agent run may wait behind the in-flight bound,
so its time starts when it starts and lasts its own ``timeout_s``.

Nothing here logs the prompt or a judge's prose. The session hold record lives in
``osprey_connectors`` and is written only under the explicit ``hold_root``; the
environment variable other processes read is never consulted here.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import inspect
import logging
import os
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from osprey.services.watchdog.config import (
    ENGINE_AGENT,
    ENGINE_DECISION,
    WatchdogDefinition,
)
from osprey.services.watchdog.history import SessionNodes, read_session_nodes
from osprey.services.watchdog.verdict import ErrorType, Verdict, record_verdict
from osprey_connectors.posture_store import state_dir
from osprey_connectors.safety_file import atomic_replace
from osprey_connectors.session_hold import open_flag

logger = logging.getLogger(__name__)

#: Under the control-state directory: one lock file and at most one pending
#: marker per (watchdog, record session).
RUNS_DIR_NAME = "watchdog_runs"

#: How often a pending run looks at its marker and the lock.
_POLL_S = 0.05

#: How long shutdown waits for cancelled runs to record before it stops waiting.
_SHUTDOWN_GRACE_S = 10.0

#: ``watchdog_status`` values: the verdict could not be reached, or it flagged
#: and its flag could not be written.
STATUS_UNAVAILABLE = "unavailable"
STATUS_FLAG_WRITE_FAILED = "flag_write_failed"

#: Receives one frame: a ``type`` key plus the frame's fields. May be async.
Notifier = Callable[[dict[str, Any]], Any]


def prompt_budget_s(config: Mapping[str, Any]) -> float:
    """How long one prompt's decision run has: the live prompt budget."""
    from osprey.models.jev.config import JevSettings

    try:
        settings = JevSettings.from_config(config)
    except Exception as exc:
        logger.warning("Watchdog read the Jev defaults: %s", type(exc).__name__)
        settings = JevSettings()
    return settings.budgets.interactive.total_s


class Engine(Protocol):
    """A judge. Implementations answer a verdict instead of raising."""

    async def judge(
        self,
        defn: WatchdogDefinition,
        nodes: SessionNodes,
        prompt: str,
        *,
        deadline: float,
        session_key: str | None,
        record_session_id: str,
        submission_id: str,
    ) -> Verdict: ...


class RunBound(Protocol):
    """Admission of agent runs, per (watchdog, record session).

    A process that shares its bound with other processes supplies its own.
    """

    async def acquire(self, watchdog: str, record_session_id: str, submission_id: str) -> bool:
        """Wait for this run's turn; ``False`` when a newer prompt replaced it."""
        ...

    async def release(self, watchdog: str, record_session_id: str, submission_id: str) -> None:
        """Give up the turn an :meth:`acquire` granted."""
        ...


class InProcessBound:
    """At most one run in flight and one pending; a newer prompt replaces the pending one."""

    def __init__(self) -> None:
        # A key is present while a run is in flight; its value is the pending
        # run's turn, resolved True when it may start and False when replaced.
        self._slots: dict[tuple[str, str], asyncio.Future[bool] | None] = {}

    async def acquire(self, watchdog: str, record_session_id: str, _submission_id: str) -> bool:
        key = (watchdog, record_session_id)
        if key not in self._slots:
            self._slots[key] = None
            return True
        replaced = self._slots[key]
        if replaced is not None and not replaced.done():
            replaced.set_result(False)
        turn: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._slots[key] = turn
        try:
            return await turn
        except asyncio.CancelledError:
            if turn.done() and not turn.cancelled() and turn.result():
                self._pass_on(key)
            elif self._slots.get(key) is turn:
                self._slots[key] = None
            raise

    async def release(self, watchdog: str, record_session_id: str, _submission_id: str) -> None:
        self._pass_on((watchdog, record_session_id))

    def _pass_on(self, key: tuple[str, str]) -> None:
        pending = self._slots.get(key)
        if pending is not None and not pending.done():
            self._slots[key] = None
            pending.set_result(True)
        else:
            self._slots.pop(key, None)


class LockFileBound:
    """At most one run in flight and one pending, across processes.

    Whoever holds the ``flock`` is in flight; the kernel releases it when the
    process dies, so a crashed run blocks nobody. A run that finds the lock taken
    overwrites the pending marker with its submission id and polls. A waiter whose
    marker now names another submission was replaced. An absent marker replaces
    nobody: only a newer waiter ever writes a different id, and the marker's
    removal by a run that was just admitted may race that write.

    The lock file is never deleted, because deleting a ``flock`` file lets two
    processes hold "the" lock through different inodes.
    """

    def __init__(self) -> None:
        self._held: dict[tuple[str, str, str], int] = {}

    async def acquire(self, watchdog: str, record_session_id: str, submission_id: str) -> bool:
        lock_path, marker = self._paths(watchdog, record_session_id)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        turn = (watchdog, record_session_id, submission_id)
        try:
            if _try_lock(descriptor):
                self._held[turn] = descriptor
                return True
            _write_marker(marker, submission_id)
            while True:
                if _replaced(marker, submission_id):
                    os.close(descriptor)
                    return False
                if _try_lock(descriptor):
                    if _replaced(marker, submission_id):
                        os.close(descriptor)
                        return False
                    _clear_marker(marker, submission_id)
                    self._held[turn] = descriptor
                    return True
                await asyncio.sleep(_POLL_S)
        except BaseException:
            os.close(descriptor)
            _clear_marker(marker, submission_id)
            raise

    async def release(self, watchdog: str, record_session_id: str, submission_id: str) -> None:
        descriptor = self._held.pop((watchdog, record_session_id, submission_id), None)
        if descriptor is not None:
            os.close(descriptor)

    def _paths(self, watchdog: str, record_session_id: str) -> tuple[Path, Path]:
        """The lock and marker paths; a session id from a payload never names a path."""
        base = state_dir()
        if base is None:
            raise RuntimeError("no control-state directory for the in-flight bound")
        directory = base / RUNS_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(f"{watchdog}\0{record_session_id}".encode()).hexdigest()[:32]
        return directory / f"{name}.lock", directory / f"{name}.pending"


def _try_lock(descriptor: int) -> bool:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _read_marker(marker: Path) -> str | None:
    try:
        return marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _replaced(marker: Path, submission_id: str) -> bool:
    return _read_marker(marker) not in (None, submission_id)


def _write_marker(marker: Path, submission_id: str) -> None:
    # Best effort: the marker only outlives the waiters polling it by accident,
    # so a lost flush costs nothing a crash had not already cost.
    atomic_replace(marker, submission_id.encode("utf-8"), mode=0o600, sync="best_effort")


def _clear_marker(marker: Path, submission_id: str) -> None:
    try:
        if _read_marker(marker) == submission_id:
            marker.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Watchdog pending marker not cleared: %s", type(exc).__name__)


class WatchdogRunner:
    """Runs a build's enabled watchdogs over each prompt.

    :param definitions: The watchdogs; only the enabled ones run.
    :param recorder: Takes every verdict; ``record_verdict`` writes the audit
        entry, a no-op records nothing. Runs in a worker thread.
    :param hold_root: Where flags are written; ``None`` opens no flag.
    :param notify: Receives ``watchdog_flag`` and ``watchdog_status`` frames.
    :param engines: Engines by engine name. A missing one is built on first use
        from *config* or *build_dir*, and closed by :meth:`shutdown`.
    :param config: The full config mapping (the Jev settings and interactive budget).
    :param build_dir: The built project an agent watchdog runs in.
    :param bound: The agent-run bound; in-process by default.
    """

    def __init__(
        self,
        definitions: Iterable[WatchdogDefinition],
        *,
        recorder: Callable[[Verdict], Any] = record_verdict,
        hold_root: Path | None,
        notify: Notifier,
        engines: Mapping[str, Engine] | None = None,
        config: Mapping[str, Any] | None = None,
        build_dir: Path | None = None,
        bound: RunBound | None = None,
    ) -> None:
        self._definitions = tuple(d for d in definitions if d.enabled)
        self._recorder = recorder
        self._hold_root = hold_root
        self._notify = notify
        self._engines: dict[str, Engine] = dict(engines or {})
        self._built: list[Engine] = []
        self._build_lock = asyncio.Lock()
        self._config: Mapping[str, Any] = config or {}
        self._build_dir = build_dir
        self._bound: RunBound = bound or InProcessBound()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._cancellable: set[asyncio.Task[Any]] = set()
        self._closing = False

    def on_prompt(
        self,
        *,
        session_key: str | None,
        record_session_id: str,
        submission_id: str,
        prompt: str,
        received_at: float,
        can_hold: bool,
    ) -> asyncio.Task[list[Verdict]]:
        """Schedule the runs for one prompt; the task answers their verdicts.

        *received_at* is a ``time.monotonic`` value.
        """
        ids = _Submission(session_key, record_session_id, submission_id, prompt, can_hold)
        return self._spawn(self._read_then_run(ids, received_at))

    def on_nodes(
        self,
        *,
        session_key: str | None,
        record_session_id: str,
        submission_id: str,
        nodes: SessionNodes,
        prompt: str,
        deadline: float,
        can_hold: bool,
    ) -> asyncio.Task[list[Verdict]]:
        """Schedule the runs over nodes already read; the task answers their verdicts."""
        ids = _Submission(session_key, record_session_id, submission_id, prompt, can_hold)
        return self._spawn(self._run_all(ids, nodes, deadline))

    async def drain(self) -> None:
        """Wait until every scheduled run has ended."""
        while self._tasks:
            await asyncio.wait(set(self._tasks))

    async def shutdown(self) -> None:
        """End every run: those not yet decided record ``intake_lost``.

        A run past its judge finishes writing its flag and verdict. Later prompts
        record ``intake_lost`` as well.
        """
        self._closing = True
        for task in list(self._cancellable):
            task.cancel()
        if self._tasks:
            _, stuck = await asyncio.wait(set(self._tasks), timeout=_SHUTDOWN_GRACE_S)
            if stuck:
                logger.warning("Watchdog shutdown left %d run(s) unfinished", len(stuck))
        built, self._built = self._built, []
        for engine in built:
            close = getattr(engine, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:
                logger.warning("Watchdog engine close failed: %s", type(exc).__name__)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._forget)
        return task

    def _forget(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        self._cancellable.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Watchdog run failed: %s", type(task.exception()).__name__)

    async def _read_then_run(self, ids: _Submission, received_at: float) -> list[Verdict]:
        if not self._definitions:
            return []
        deadline = received_at + prompt_budget_s(self._config)
        if self._closing:
            return await self._run_all(ids, SessionNodes(), deadline, lost=True)
        task = asyncio.current_task()
        assert task is not None
        self._cancellable.add(task)
        try:
            nodes = await asyncio.to_thread(
                read_session_nodes, ids.record_session_id, deadline=deadline
            )
        except asyncio.CancelledError:
            return await self._run_all(ids, SessionNodes(), deadline, lost=True)
        except Exception as exc:
            logger.warning("Watchdog record read failed: %s", type(exc).__name__)
            nodes = SessionNodes(record_available=False, reason=type(exc).__name__)
        finally:
            self._cancellable.discard(task)
        return await self._run_all(ids, nodes, deadline)

    async def _run_all(
        self, ids: _Submission, nodes: SessionNodes, deadline: float, *, lost: bool = False
    ) -> list[Verdict]:
        runs = [
            self._spawn(self._run_one(defn, ids, nodes, deadline, lost=lost))
            for defn in self._definitions
        ]
        if not runs:
            return []
        answered = await asyncio.gather(*runs, return_exceptions=True)
        return [v for v in answered if isinstance(v, Verdict)]

    async def _run_one(
        self,
        defn: WatchdogDefinition,
        ids: _Submission,
        nodes: SessionNodes,
        deadline: float,
        *,
        lost: bool,
    ) -> Verdict:
        started = time.monotonic()
        if lost or self._closing:
            verdict = _unavailable(defn, ids, "intake_lost", started)
        else:
            task = asyncio.current_task()
            assert task is not None
            self._cancellable.add(task)
            try:
                verdict = await self._admit_and_judge(defn, ids, nodes, deadline, started)
            except asyncio.CancelledError:
                verdict = _unavailable(defn, ids, "intake_lost", started)
            finally:
                self._cancellable.discard(task)
        await self._finish(defn, ids, verdict)
        return verdict

    async def _admit_and_judge(
        self,
        defn: WatchdogDefinition,
        ids: _Submission,
        nodes: SessionNodes,
        deadline: float,
        started: float,
    ) -> Verdict:
        if defn.engine != ENGINE_AGENT:
            return await self._judge(defn, ids, nodes, deadline, started)
        turn = (defn.name, ids.record_session_id, ids.submission_id)
        try:
            admitted = await self._bound.acquire(*turn)
        except Exception as exc:
            logger.warning("Watchdog %r bound failed: %s", defn.name, type(exc).__name__)
            return _unavailable(defn, ids, "run_error", started)
        if not admitted:
            return _unavailable(defn, ids, "in_flight_bound", started)
        try:
            return await self._judge(defn, ids, nodes, time.monotonic() + defn.timeout_s, started)
        finally:
            try:
                await self._bound.release(*turn)
            except Exception as exc:
                logger.warning("Watchdog %r bound release: %s", defn.name, type(exc).__name__)

    async def _judge(
        self,
        defn: WatchdogDefinition,
        ids: _Submission,
        nodes: SessionNodes,
        deadline: float,
        started: float,
    ) -> Verdict:
        try:
            engine = await self._engine_for(defn.engine)
            return await engine.judge(
                defn,
                nodes,
                ids.prompt,
                deadline=deadline,
                session_key=ids.session_key,
                record_session_id=ids.record_session_id,
                submission_id=ids.submission_id,
            )
        except Exception as exc:
            logger.warning("Watchdog %r judge failed: %s", defn.name, type(exc).__name__)
            return _unavailable(defn, ids, "run_error", started)

    async def _engine_for(self, name: str) -> Engine:
        async with self._build_lock:
            engine = self._engines.get(name)
            if engine is None:
                engine = await asyncio.to_thread(self._build_engine, name)
                self._engines[name] = engine
                self._built.append(engine)
            return engine

    def _build_engine(self, name: str) -> Engine:
        if name == ENGINE_DECISION:
            from osprey.services.watchdog.engines.decision import DecisionEngine

            return DecisionEngine(self._config)
        if name == ENGINE_AGENT and self._build_dir is not None:
            from osprey.services.watchdog.engines.agent import AgentEngine

            return AgentEngine(self._build_dir)
        raise RuntimeError(f"no {name} engine available")

    async def _finish(self, defn: WatchdogDefinition, ids: _Submission, verdict: Verdict) -> None:
        """Flag, record, announce. Each step's failure costs only that step."""
        flag_id: str | None = None
        flag_failed = False
        if verdict.flagged and ids.can_hold:
            try:
                flag_id = await self._open_flag(defn, ids, verdict)
            except Exception as exc:
                flag_failed = True
                logger.error("Watchdog %r flag write failed: %s", defn.name, type(exc).__name__)
        try:
            await asyncio.to_thread(self._recorder, verdict)
        except Exception as exc:
            logger.error("Watchdog %r verdict not recorded: %s", defn.name, type(exc).__name__)
        if flag_id is not None:
            await self._send(
                {
                    "type": "watchdog_flag",
                    "session_key": ids.session_key,
                    "flag_id": flag_id,
                    "watchdog": defn.name,
                }
            )
        status = (
            STATUS_FLAG_WRITE_FAILED
            if flag_failed
            else STATUS_UNAVAILABLE
            if verdict.status == "unavailable"
            else None
        )
        if status is not None:
            await self._send(
                {
                    "type": "watchdog_status",
                    "session_key": ids.session_key,
                    "watchdog": defn.name,
                    "status": status,
                }
            )

    async def _open_flag(
        self, defn: WatchdogDefinition, ids: _Submission, verdict: Verdict
    ) -> str | None:
        """Write the flag; its id when this call opened it, else ``None``."""
        if ids.session_key is None or self._hold_root is None:
            return None
        flag_id = str(uuid.uuid4())
        opened = await asyncio.to_thread(
            open_flag,
            ids.session_key,
            flag_id=flag_id,
            source=defn.name,
            reason=verdict.reason or "",
            prompt_ref=verdict.prompt_ref,
            prompt_excerpt=ids.prompt,
            root=self._hold_root,
        )
        return flag_id if opened else None

    async def _send(self, frame: dict[str, Any]) -> None:
        try:
            sent = self._notify(frame)
            if inspect.isawaitable(sent):
                await sent
        except Exception as exc:
            logger.warning("Watchdog notifier failed: %s", type(exc).__name__)


@dataclass(frozen=True, slots=True)
class _Submission:
    """The prompt one set of runs belongs to."""

    session_key: str | None
    record_session_id: str
    submission_id: str
    prompt: str
    can_hold: bool


def unavailable_verdict(
    defn: WatchdogDefinition,
    *,
    session_key: str | None,
    record_session_id: str,
    submission_id: str,
    error_type: ErrorType,
    latency_s: float = 0.0,
) -> Verdict:
    """The verdict for a run that reached no judge's answer."""
    return Verdict(
        watchdog=defn.name,
        engine=defn.engine,  # type: ignore[arg-type]
        session_key=session_key,
        record_session_id=record_session_id,
        submission_id=submission_id,
        status="unavailable",
        error_type=error_type,
        threshold=defn.flag_threshold if defn.engine == ENGINE_DECISION else None,
        latency_s=latency_s,
    )


def _unavailable(
    defn: WatchdogDefinition, ids: _Submission, error_type: ErrorType, started: float
) -> Verdict:
    return unavailable_verdict(
        defn,
        session_key=ids.session_key,
        record_session_id=ids.record_session_id,
        submission_id=ids.submission_id,
        error_type=error_type,
        latency_s=time.monotonic() - started,
    )
