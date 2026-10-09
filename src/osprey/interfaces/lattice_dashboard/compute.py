"""Compute manager — the figure workers' supervision.

Each figure has one job slot. A launch reserves the slot, which makes the new
job current at once, and schedules one task on the event loop that starts the
worker, waits for it and applies its result. Only the current job's result is
applied: a job superseded by a later launch of the same figure ends as
``cancelled`` and changes nothing, since its successor has already marked the
figure computing. Every launch returns before any worker is started or reaped,
and every state write and broadcast happens on the loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from osprey.interfaces.lattice_dashboard.state import (
    ALL_FIGURES,
    VERIFICATION_FIGURES,
    LatticeState,
    fast_figures,
)
from osprey_connectors.process import ChildExit, ChildJob, ExitCause, JobSlots

logger = logging.getLogger("osprey.lattice_dashboard.compute")

#: Longest a figure worker may run before it is put down as timed out.
WORKER_DEADLINE_S = 300.0

#: How much of a failed worker's stderr its error keeps.
_STDERR_TAIL = 500

#: The package each figure's worker module lives in.
_WORKERS = "osprey.interfaces.lattice_dashboard.workers"


def worker_argv(name: str, state_path: Path, output_path: Path) -> list[str]:
    """Return the command line that runs figure *name*'s worker."""
    return [sys.executable, "-m", f"{_WORKERS}.{name}", str(state_path), str(output_path)]


def _read_summary_updates(output_path: Path, name: str) -> dict[str, Any] | None:
    """Extract a worker's optional ``summary_updates`` block from its output.

    A worker that recomputes header-summary quantities on the ring it actually
    tracked (currently the optics worker: tunes, chromaticity, beta_max)
    publishes them under this top-level key; the figure adapters ignore it.

    Args:
        output_path: The worker's raw-data JSON file.
        name: Figure name, for log context.

    Returns:
        The block, or None when the worker published none or the file could
        not be parsed — a summary refresh is worth losing, a figure is not.
    """
    try:
        payload = json.loads(output_path.read_text())
    except (OSError, ValueError):
        logger.warning("%s: could not read worker output for summary updates", name)
        return None
    if not isinstance(payload, dict):
        return None
    updates = payload.get("summary_updates")
    return updates if isinstance(updates, dict) else None


class ComputeManager:
    """Supervises the figure workers.

    Args:
        state: LatticeState instance for reading/writing state.
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

    def refresh_fast(self) -> list[str]:
        """Recompute the loaded model's fast figures.

        A ``single_pass`` model draws optics only, so only that worker runs.
        """
        names = list(fast_figures(self._state.load().get("solve")))
        for name in names:
            self._launch(name)
        return names

    def refresh_verification(self) -> list[str]:
        """Launch DA + LMA verification workers."""
        names = list(VERIFICATION_FIGURES)
        for name in names:
            self._launch(name)
        return names

    def refresh_one(self, name: str) -> None:
        """Launch a single figure worker."""
        if name not in ALL_FIGURES:
            raise ValueError(f"Unknown figure: {name}. Must be one of {ALL_FIGURES}")
        self._launch(name)

    async def stop_all(self) -> None:
        """Put every worker down, then cancel the supervision tasks.

        Returns once every worker is reaped and every task has ended.
        """
        await self._slots.stop_all(ExitCause.CANCELLED)
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _launch(self, name: str) -> None:
        """Make a new job current for *name* and schedule its supervision."""
        job = self._slots.reserve(name)
        output_path = self._state.figures_dir / f"{name}.json"
        argv = worker_argv(name, self._state.state_path, output_path)
        logger.info("Launching %s worker (job %d)", name, job.job)
        self._state.mark_computing(name)
        self._broadcaster.broadcast({"type": "figure_status", "name": name, "status": "computing"})
        task = asyncio.get_running_loop().create_task(self._supervise(job, argv, output_path))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _supervise(self, job: ChildJob, argv: list[str], output_path: Path) -> None:
        try:
            await job.start(argv, collect_stderr=_STDERR_TAIL)
        except OSError as exc:
            await job.stop(ExitCause.CANCELLED)
            if self._slots.is_current(job):
                self._fail(job.name, f"Failed to launch worker: {exc}")
            return
        exit_ = await job.wait(deadline_s=WORKER_DEADLINE_S)
        if not self._slots.is_current(job):
            logger.debug("%s job %d is no longer current; its result is dropped", job.name, job.job)
            return
        self._apply(job.name, exit_, output_path)

    def _apply(self, name: str, exit_: ChildExit, output_path: Path) -> None:
        """Apply the current job's exit to its figure."""
        if exit_.cause is ExitCause.CANCELLED:
            return
        if exit_.cause is ExitCause.TIMED_OUT:
            self._fail(name, f"Worker timed out after {WORKER_DEADLINE_S:.0f} s")
            return
        if exit_.cause is not ExitCause.COMPLETED:
            self._fail(name, f"Worker exited with code {exit_.returncode}: {exit_.stderr_tail}")
            return
        if not output_path.exists():
            self._fail(name, "Worker completed but no output file produced")
            return

        logger.info("%s worker completed successfully", name)
        summary_updates = _read_summary_updates(output_path, name)
        self._state.mark_ready(name, summary_updates)
        self._broadcaster.broadcast({"type": "figure_ready", "name": name})
        if summary_updates:
            # figure_ready only makes the client fetch that one figure. The
            # summary chips come from /api/state, so ask for a state re-read.
            self._broadcaster.broadcast({"type": "state_updated"})

    def _fail(self, name: str, error: str) -> None:
        logger.warning("%s worker failed: %s", name, error)
        self._state.mark_error(name, error)
        self._broadcaster.broadcast({"type": "figure_error", "name": name, "error": error})
