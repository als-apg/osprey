"""The injected scanner seam: keeps the bridge lifecycle core import-clean of bluesky.

``Scanner`` is the boundary the lifecycle core (``runs.py``'s ``do_promote``) is
written against. The real implementation (a bluesky ``RunEngine`` in a daemon
thread, wired to ophyd-async devices and a ``TiledWriter``) arrives in Phase 2 as
``scanner_bluesky.py`` and lives behind the ``osprey-framework[scan-bridge]``
extra. Everything in this module — the Protocol and ``FakeScanner`` — has no
bluesky/ophyd/tiled dependency, so the lifecycle core can be built, imported, and
unit-tested before that extra ever needs to be installed.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Scanner(Protocol):
    """One scan's hardware/engine handle, as seen by the bridge lifecycle core.

    A fresh instance is built per promoted run (see ``do_promote``): the caller
    calls ``reinitialize`` then ``start_scan_thread``, polls ``is_scanning_active``/
    ``estimate_current_completion``/``current_state`` for status, and may call
    ``stop_scanning_thread`` to abort. ``last_run_uid`` surfaces the underlying
    run identifier (e.g. a bluesky start-doc uid) once the scan has begun.
    """

    current_state: Any
    last_run_uid: str | None

    def reinitialize(self, exec_config: Any) -> bool:
        """Prepare the scan from ``exec_config``. Returns False on setup failure."""
        ...

    def start_scan_thread(self) -> None:
        """Start the scan running in a background thread. Non-blocking."""
        ...

    def stop_scanning_thread(self) -> None:
        """Request the running scan stop. Safe to call even if not active."""
        ...

    def is_scanning_active(self) -> bool:
        """True while the scan thread is running."""
        ...

    def estimate_current_completion(self) -> float:
        """A 0.0-1.0 progress estimate."""
        ...


class FakeScanner:
    """Deterministic ``Scanner`` test double — no bluesky dependency.

    Tracks call counts and lets a test script the scan's progress and outcome
    directly, instead of driving a real RunEngine thread:

    - ``reinitialize_fails``: make ``reinitialize`` return False (setup failure).
    - ``simulate_progress(fraction)``: set the completion estimate mid-run.
    - ``simulate_completion()`` / ``simulate_error(message)``: end the scan.

    ``start_scan_thread`` mints a ``last_run_uid`` (if one wasn't pre-seeded via
    the constructor) so callers can exercise "run_uid becomes available once the
    scan starts" behavior without a real document stream.
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
            return False
        self.current_state = "armed"
        return True

    def start_scan_thread(self) -> None:
        self.start_calls += 1
        self._active = True
        self.current_state = "running"
        if self.last_run_uid is None:
            self.last_run_uid = uuid.uuid4().hex

    def stop_scanning_thread(self) -> None:
        self.stop_calls += 1
        self._active = False
        self.current_state = "stopped"

    def is_scanning_active(self) -> bool:
        return self._active

    def estimate_current_completion(self) -> float:
        return self._completion

    def simulate_progress(self, fraction: float) -> None:
        """Set the completion estimate (clamped to 0.0-1.0) without ending the scan."""
        self._completion = max(0.0, min(1.0, fraction))

    def simulate_completion(self) -> None:
        """End the scan successfully, as if the thread ran to completion."""
        self._active = False
        self._completion = 1.0
        self.current_state = "completed"

    def simulate_error(self, message: str = "scan failed") -> None:
        """End the scan in a failed state, as if the thread raised."""
        self._active = False
        self.current_state = "error"
        self.error_message = message
