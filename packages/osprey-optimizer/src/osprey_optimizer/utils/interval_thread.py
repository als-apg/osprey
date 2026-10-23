"""A daemon thread that runs a callback on a fixed interval until stopped.

Shared by the job heartbeat (``api.job_tracker``) and the script log flusher
(``scripting.context``): both tick on an interval, both stop via an event so
shutdown is immediate rather than waiting out the current interval.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class IntervalThread:
    """Call ``tick`` every ``interval`` seconds on a daemon thread.

    Args:
        tick: called once per interval, on the worker thread.
        interval: seconds between ticks (the first tick happens after one
            interval, not immediately).
        name: thread name, also used in the swallowed-error log line.
        swallow_errors: log a failing tick and keep beating instead of letting
            the exception kill the thread. Required by any ticker whose death
            would go unnoticed (the heartbeat: it alone keeps a running job's
            keys alive, so one failed refresh must not stop it).
        final_tick: run one last tick after :meth:`stop`, so work buffered
            since the previous tick is not lost.
    """

    def __init__(
        self,
        tick: Callable[[], None],
        *,
        interval: float,
        name: str,
        swallow_errors: bool = False,
        final_tick: bool = False,
    ) -> None:
        self._tick = tick
        self._interval = interval
        self._name = name
        self._swallow_errors = swallow_errors
        self._final_tick = final_tick
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the daemon thread and begin ticking."""
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float) -> None:
        """Signal the loop to exit and join it. A no-op if never started."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        """True while the thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            self._run_tick()
        if self._final_tick:
            self._run_tick()

    def _run_tick(self) -> None:
        if not self._swallow_errors:
            self._tick()
            return
        try:
            self._tick()
        except Exception as e:
            logger.warning("%s tick failed: %s", self._name, e)
