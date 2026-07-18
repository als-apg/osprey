"""The injected runner seam: keeps the bridge lifecycle core import-clean of bluesky.

``PlanRunner`` is the boundary the lifecycle core (``runs.py``'s ``do_launch``) is
written against. The real implementation (a bluesky ``RunEngine`` in a daemon
thread, wired to ophyd-async devices and a ``TiledWriter``) lives in
``plan_runner_bluesky.py``. The bluesky stack (bluesky/ophyd-async/tiled) is a
core dependency, so that module always imports; this seam is an import-hygiene
boundary, not an install-size one. Everything in this module — the Protocol and
``FakePlanRunner`` — has no bluesky/ophyd/tiled dependency, so the lifecycle core
can be built, imported, and unit-tested without loading the RunEngine.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PlanRunner(Protocol):
    """One run's hardware/engine handle, as seen by the bridge lifecycle core.

    A fresh instance is built per launched run (see ``do_launch``): the caller
    calls ``reinitialize`` then ``start_run_thread``, polls ``is_run_active``/
    ``estimate_current_completion``/``current_state`` for status, and may call
    ``stop_run_thread`` to abort. ``last_run_uid`` surfaces the underlying
    run identifier (e.g. a bluesky start-doc uid) once the run has begun.
    ``error_message`` is the explicit terminal-error signal: non-None means the
    run ended in an unrecoverable error (``runs.py``'s ``Run.status`` reads
    this directly, rather than string-matching ``current_state``).
    """

    current_state: Any
    last_run_uid: str | None
    error_message: str | None

    def reinitialize(self, exec_config: Any) -> bool:
        """Prepare the run from ``exec_config``. Returns False on setup failure."""
        ...

    def start_run_thread(self) -> None:
        """Start the running in a background thread. Non-blocking."""
        ...

    def stop_run_thread(self) -> None:
        """Request the running plan stop. Safe to call even if not active."""
        ...

    def is_run_active(self) -> bool:
        """True while the run thread is running."""
        ...

    def estimate_current_completion(self) -> float:
        """A 0.0-1.0 progress estimate."""
        ...


class FakePlanRunner:
    """Deterministic ``PlanRunner`` test double — no bluesky dependency.

    Tracks call counts and lets a test script the run's progress and outcome
    directly, instead of driving a real RunEngine thread:

    - ``reinitialize_fails``: make ``reinitialize`` return False (setup failure).
    - ``simulate_progress(fraction)``: set the completion estimate mid-run.
    - ``simulate_completion()`` / ``simulate_error(message)``: end the run.

    ``start_run_thread`` mints a ``last_run_uid`` (if one wasn't pre-seeded via
    the constructor) so callers can exercise "run_uid becomes available once the
    run starts" behavior without a real document stream.
    """

    def __init__(self, run_uid: str | None = None, reinitialize_fails: bool = False) -> None:
        self.current_state: str = "idle"
        self.last_run_uid: str | None = run_uid
        self.reinitialize_fails = reinitialize_fails
        self.reinitialize_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        self._active = False
        self._completion = 0.0
        self.error_message: str | None = None

    def reinitialize(self, exec_config: Any) -> bool:
        self.reinitialize_calls += 1
        if self.reinitialize_fails:
            self.current_state = "error"
            self.error_message = "reinitialize() was configured to fail"
            return False
        self.current_state = "armed"
        return True

    def start_run_thread(self) -> None:
        self.start_calls += 1
        self._active = True
        self.current_state = "running"
        if self.last_run_uid is None:
            self.last_run_uid = uuid.uuid4().hex

    def stop_run_thread(self) -> None:
        self.stop_calls += 1
        self._active = False
        self.current_state = "stopped"

    def is_run_active(self) -> bool:
        return self._active

    def estimate_current_completion(self) -> float:
        return self._completion

    def simulate_progress(self, fraction: float) -> None:
        """Set the completion estimate (clamped to 0.0-1.0) without ending the run."""
        self._completion = max(0.0, min(1.0, fraction))

    def simulate_completion(self) -> None:
        """End the run successfully, as if the thread ran to completion."""
        self._active = False
        self._completion = 1.0
        self.current_state = "completed"

    def simulate_error(self, message: str = "run failed") -> None:
        """End the run in a failed state, as if the thread raised."""
        self._active = False
        self.current_state = "error"
        self.error_message = message
