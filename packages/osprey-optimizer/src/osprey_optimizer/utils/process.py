"""Process-management helpers shared by the API routers."""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import signal
from types import FrameType

from osprey_optimizer.utils.cancellation import cancel_on_signal

logger = logging.getLogger(__name__)

# How long a service shutdown waits for a cancelled job's worker to stop on its
# own (record CANCELLED, save partial results) before terminating it. Docker
# gives a container 10 s between SIGTERM and SIGKILL by default; 6 s leaves the
# rest for the terminate and kill steps after it (2 s, see
# job_control.stop_launched_jobs) and for uvicorn's own request drain. A
# deployment with a longer stop grace raises both together.
SHUTDOWN_GRACE_ENV = "OPTIMIZER_SHUTDOWN_GRACE_S"
DEFAULT_SHUTDOWN_GRACE_SECONDS = 6.0


def reap_finished_children() -> None:
    """Reap finished child processes to prevent zombie accumulation."""
    multiprocessing.active_children()


def shutdown_grace_seconds() -> float:
    """The shutdown grace from ``OPTIMIZER_SHUTDOWN_GRACE_S``, or the default.

    A value that is not a non-negative number is reported and ignored: a typo
    must not turn a bounded shutdown into an unbounded one.
    """
    raw = os.environ.get(SHUTDOWN_GRACE_ENV, "")
    if not raw.strip():
        return DEFAULT_SHUTDOWN_GRACE_SECONDS
    try:
        grace = float(raw)
    except ValueError:
        grace = math.nan
    if not math.isfinite(grace) or grace < 0:
        logger.warning(
            "Ignoring %s=%r (not a non-negative number); using %.1f s",
            SHUTDOWN_GRACE_ENV,
            raw,
            DEFAULT_SHUTDOWN_GRACE_SECONDS,
        )
        return DEFAULT_SHUTDOWN_GRACE_SECONDS
    return grace


def install_worker_signal_handlers() -> None:
    """Make SIGTERM/SIGINT to a job worker cancel its run, with a hard deadline.

    Called on the worker's main thread before the job starts. A worker is
    signalled directly by a kill, by a supervisor that signals the whole group
    (systemd's default, a terminal's Ctrl-C), or by ``multiprocessing``'s exit
    hook, which terminates daemon children and then joins them without a
    timeout. Without this, the signal ends the worker wherever it is, possibly
    mid-write, with no cancel record and no partial results.

    The first signal stops the run through the normal cancel path (the token
    reports it, the run saves its partial results, the lifecycle records
    CANCELLED; nothing is restored) and arms ``SIGALRM`` at its default action,
    so the kernel ends the worker if it has not stopped within the shutdown
    grace, keeping that join bounded. A second signal ends it at once.

    A no-op outside a ``multiprocessing`` child, so a process that runs a job
    in-process (a test) keeps its own handlers.
    """
    if multiprocessing.parent_process() is None:
        return
    deadline = max(1, math.ceil(shutdown_grace_seconds()))

    def _on_signal(_signum: int, _frame: FrameType | None) -> None:
        # No logging here: the main thread may hold a logging lock when the
        # handler runs on it. The run's cancel path logs the outcome.
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_DFL)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(deadline)
        cancel_on_signal()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)
