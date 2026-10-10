"""Compute manager — the figure workers' supervision.

Each figure has one job slot. A launch reserves the slot, which makes the new
job current at once, writes the job's immutable input file, and schedules one
task on the event loop that starts the worker, waits for it and applies its
result. Only the current job's result is applied: a job superseded by a later
launch of the same figure ends as ``cancelled`` and changes nothing, since its
successor has already made the figure computing. Every launch returns before
any worker is started or reaped, and every broadcast happens on the loop.

A figure's status is derived, never stored: ``ready`` when the store holds
the figure for the key of the inputs on screen, ``computing`` while the
figure's current job computes that key, ``failed`` when the last job for that
key failed, ``stale`` when the store holds the figure of the selected deck
only under other keys, and ``not_computed`` otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.interfaces.lattice_dashboard.state import (
    ALL_FIGURES,
    VERIFICATION_FIGURES,
    LatticeState,
)
from osprey_connectors.process import ChildExit, ChildJob, ExitCause, JobSlots

logger = logging.getLogger("osprey.lattice_dashboard.compute")

#: Longest a figure worker may run before it is put down as timed out.
WORKER_DEADLINE_S = 300.0

#: How much of a failed worker's stderr its error keeps.
_STDERR_TAIL = 500

#: The package each figure's worker module lives in.
_WORKERS = "osprey.interfaces.lattice_dashboard.workers"


def worker_argv(name: str, job_path: Path, output_path: Path) -> list[str]:
    """Return the command line that runs figure *name*'s worker on one job file."""
    return [sys.executable, "-m", f"{_WORKERS}.{name}", str(job_path), str(output_path)]


class ComputeManager:
    """Supervises the figure workers and derives each figure's status.

    Args:
        state: The selection, the what-if inputs and the figure store.
        broadcaster: Object with a ``broadcast(data)`` method for SSE push.
        slots: The job slots, one per figure; a fresh set when omitted.
    """

    def __init__(
        self, state: LatticeState, broadcaster: Any, slots: JobSlots | None = None
    ) -> None:
        self._state = state
        self._broadcaster = broadcaster
        self._slots = slots if slots is not None else JobSlots()
        self._tasks: set[asyncio.Task[None]] = set()
        #: The key each launched job computes, by job id.
        self._keys: dict[int, str] = {}
        #: The jobs not yet ended, by job id.
        self._running: set[int] = set()
        #: The last failure of each figure: the key it computed and its error.
        self._failures: dict[str, tuple[str, str]] = {}

    def refresh_fast(self) -> list[str]:
        """Recompute the selected model's fast figures; returns the ones launched."""
        names = list(self._state.selection.capabilities.fast_figures)
        return [name for name in names if self._launch(name)]

    def refresh_verification(self) -> list[str]:
        """Launch the verification figures; returns the ones launched."""
        return [name for name in VERIFICATION_FIGURES if self._launch(name)]

    def refresh_one(self, name: str) -> bool:
        """Launch figure *name*'s worker; False when no model is ready."""
        if name not in ALL_FIGURES:
            raise ValueError(f"Unknown figure: {name}. Must be one of {ALL_FIGURES}")
        return self._launch(name)

    def figure_status(self, name: str) -> dict[str, Any]:
        """Return figure *name*'s ``{status, key, updated, error}`` for the inputs on screen."""
        key = self._state.figure_key(name)
        if key is None:
            return {"status": "not_computed", "key": None, "updated": None, "error": None}
        path = self._state.figure_path(name, key)
        if path.is_file():
            updated = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            return {"status": "ready", "key": key, "updated": updated, "error": None}
        job = self._slots.current(name)
        if job is not None and job.job in self._running and self._keys.get(job.job) == key:
            return {"status": "computing", "key": key, "updated": None, "error": None}
        failure = self._failures.get(name)
        if failure is not None and failure[0] == key:
            return {"status": "failed", "key": key, "updated": None, "error": failure[1]}
        status = "stale" if self._state.has_other_key(name, key) else "not_computed"
        return {"status": status, "key": key, "updated": None, "error": None}

    async def stop_all(self) -> None:
        """Put every worker down, then cancel the supervision tasks.

        Returns once every worker is reaped and every task has ended.
        """
        await self._slots.stop_all(ExitCause.CANCELLED)
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _launch(self, name: str) -> bool:
        """Make a new job current for *name* and schedule its supervision."""
        spec = self._state.job_spec(name)
        if spec is None:
            return False
        job = self._slots.reserve(name)
        key = spec["key"]
        job_path = self._state.write_job(spec, job.job)
        output_path = self._state.figure_path(name, key)
        argv = worker_argv(name, job_path, output_path)
        self._keys[job.job] = key
        self._running.add(job.job)
        logger.info("Launching %s worker (job %d)", name, job.job)
        self._broadcaster.broadcast({"type": "figure_status", "name": name, "status": "computing"})
        task = asyncio.get_running_loop().create_task(
            self._supervise(job, argv, key, job_path, output_path)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _supervise(
        self, job: ChildJob, argv: list[str], key: str, job_path: Path, output_path: Path
    ) -> None:
        try:
            try:
                await job.start(argv, collect_stderr=_STDERR_TAIL)
            except OSError as exc:
                await job.stop(ExitCause.CANCELLED)
                if self._slots.is_current(job):
                    self._fail(job.name, key, f"Failed to launch worker: {exc}")
                return
            exit_ = await job.wait(deadline_s=WORKER_DEADLINE_S)
            if not self._slots.is_current(job):
                logger.debug("%s job %d is no longer current; result dropped", job.name, job.job)
                return
            self._apply(job.name, key, exit_, output_path)
        finally:
            self._running.discard(job.job)
            self._keys.pop(job.job, None)
            job_path.unlink(missing_ok=True)

    def _apply(self, name: str, key: str, exit_: ChildExit, output_path: Path) -> None:
        """Apply the current job's exit to its figure."""
        if exit_.cause is ExitCause.CANCELLED:
            return
        if exit_.cause is ExitCause.TIMED_OUT:
            self._fail(name, key, f"Worker timed out after {WORKER_DEADLINE_S:.0f} s")
            return
        if exit_.cause is not ExitCause.COMPLETED:
            self._fail(
                name, key, f"Worker exited with code {exit_.returncode}: {exit_.stderr_tail}"
            )
            return
        if not output_path.is_file():
            self._fail(name, key, "Worker completed but no output file produced")
            return

        logger.info("%s worker completed successfully", name)
        self._failures.pop(name, None)
        self._state.prune(name)
        self._broadcaster.broadcast({"type": "figure_ready", "name": name})
        if name == "optics":
            # The summary chips come from the optics figure through
            # /api/state, so ask for a state re-read.
            self._broadcaster.broadcast({"type": "state_updated"})

    def _fail(self, name: str, key: str, error: str) -> None:
        logger.warning("%s worker failed: %s", name, error)
        self._failures[name] = (key, error)
        self._broadcaster.broadcast({"type": "figure_error", "name": name, "error": error})
