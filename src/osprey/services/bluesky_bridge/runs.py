"""The bridge's in-memory run registry and lifecycle state machine.

``Run`` tracks one scan from intent through completion. Creating a run never
touches the ``Scanner`` seam (see ``scanner.py``) — that only happens once a
run is *promoted* via ``do_promote``, the single choke point that starts a
real scan. This module is deliberately free of bluesky/ophyd/tiled imports so
the lifecycle core stays importable — and unit testable with a
``FakeScanner`` — before those dependencies are ever installed. It does use
``fastapi.HTTPException`` for its error semantics (mirroring ``security.py``),
since FastAPI is a core bridge dependency, not one of the deferred ones.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

if TYPE_CHECKING:
    from .scanner import Scanner

# `Run.status` treats a scanner whose (lowercased, stringified) `current_state`
# equals one of these as the one terminal-failure signal from the scanner seam
# not already captured by `run.error` directly (e.g. a background scan thread
# failing asynchronously). Equality, not substring match, so a state like
# "no_error" can't false-positive. Only `FakeScanner` exists today and it sets
# exactly "error" (see `simulate_error`); Phase 2's real bluesky-backed
# `Scanner` should add an explicit terminal-error signal to the protocol
# instead of leaning on `current_state` string matching.
_ERROR_STATES = {"error"}


@dataclass
class Run:
    """One scan-run's lifecycle state.

    ``request`` is deliberately untyped here (see Task 1.5's pydantic
    ``RunRequest``) so this module never has to import FastAPI/pydantic.
    """

    id: str
    request: Any
    created_at: float = field(default_factory=time.time)
    promoted: bool = False
    promoting: bool = False  # guards the promote critical section (see do_promote)
    scanner: Scanner | None = None
    stopped: bool = False
    error: str | None = None
    # Which sanctioned launch path promoted this run, e.g. "agent" (the
    # token-gated `launch_scan` MCP tool route). Set at intent creation or,
    # at latest, by whatever promotes the run.
    launched_by: str | None = None

    @property
    def run_uid(self) -> str | None:
        """The underlying scan's run identifier, once the scan has started."""
        return self.scanner.last_run_uid if self.scanner is not None else None

    @property
    def status(self) -> str:
        """Derive the run's lifecycle state: intent -> running -> completed | stopped | error.

        ``stopped`` is checked before ``promoted`` so an intent stopped before
        it was ever promoted correctly reports "stopped" rather than a stale
        "intent" — `do_promote` refuses to (re-)promote a stopped run either
        way, so "intent" would be misleading once that door is permanently
        closed.
        """
        if self.error:
            return "error"
        if self.stopped:
            return "stopped"
        if not self.promoted:
            return "intent"
        if self.scanner is not None:
            if self.scanner.is_scanning_active():
                return "running"
            if str(self.scanner.current_state).lower() in _ERROR_STATES:
                return "error"
        return "completed"

    def to_dict(self) -> dict:
        status = self.status
        out: dict[str, Any] = {"id": self.id, "status": status}
        if self.promoted and self.scanner is not None:
            out["completion"] = self.scanner.estimate_current_completion()
        if self.launched_by:
            out["launched_by"] = self.launched_by
        run_uid = self.run_uid
        if run_uid:
            out["run_uid"] = run_uid
        if self.error:
            out["error"] = self.error
        elif status == "error" and self.scanner is not None:
            out["error"] = f"scan ended in state {self.scanner.current_state!r}"
        return out


class RunRegistry:
    """Thread-safe in-memory store of `Run` objects, keyed by id.

    ``lock`` also guards `do_promote`'s promoting/promoted/stopped
    check-and-set below — the registry is in-memory only, so a single lock
    per bridge process is enough to admit exactly one promotion per run.
    """

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self.lock = Lock()

    def add(self, request: Any, launched_by: str | None = None) -> Run:
        """Record a launch intent and return the new `Run`. Never touches the scanner seam."""
        run = Run(id=uuid.uuid4().hex, request=request, launched_by=launched_by)
        with self.lock:
            self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> Run:
        """Look up a run by id, raising 404 if it has no record.

        The registry is in-memory only, so this is also the honest answer for
        a run id from before a process restart — there is no persisted
        history to fall back to.
        """
        with self.lock:
            run = self._runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return run

    def list(self, limit: int = 20) -> list[Run]:
        """This process's tracked runs, newest first."""
        with self.lock:
            records = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        return records[: max(1, min(limit, 100))]


# Module-level singleton: the bridge process's one run registry.
registry = RunRegistry()


def do_promote(run: Run, scanner_factory: Callable[[], Scanner]) -> Run:
    """The single choke point that starts a real scan.

    Callers (the token-gated ``POST /runs/{id}/promote`` route, task 1.5) must
    already have enforced their sanctioned human decision — the promote token
    (``security.verify_promote_token``) plus `launch_scan`'s in-tool
    ``writes_enabled`` re-check — before calling this.

    ``scanner_factory`` builds a fresh `Scanner` OUTSIDE the lock (it may be
    slow, e.g. constructing a real bluesky `RunEngine`); the lock only guards
    the promoting/promoted/stopped check-and-set, so two concurrent promotes
    of the same run can't both start a scan, and a stopped run can never be
    promoted. Publishes ``promoted=True`` before clearing ``promoting`` so a
    concurrent promote always observes either "not yet promoting" or
    "promoted" — never a window where both are false.

    A stop can race the unlocked build/start window above: ``stop_run``
    (``app.py``) may run while this scanner is still being built, see
    ``promoted=False``/``scanner=None`` (nothing to stop yet), and merely
    record ``run.stopped = True``. Left unhandled, this scanner would then
    finish starting and get published anyway — a live, untracked scan behind
    a run that reports "stopped". So publishing re-acquires the lock and
    re-checks ``run.stopped`` in the same critical section that sets
    ``scanner``/``promoted``; if a stop landed during the build, the
    just-started scanner is stopped immediately after releasing the lock.
    """
    with registry.lock:
        if run.promoted:
            raise HTTPException(status_code=409, detail=f"run {run.id!r} already promoted")
        if run.promoting:
            raise HTTPException(
                status_code=409, detail=f"run {run.id!r} promotion already in progress"
            )
        if run.stopped:
            raise HTTPException(
                status_code=409, detail=f"run {run.id!r} was stopped; cannot promote"
            )
        run.promoting = True
        run.error = None  # clear any prior failed-promote error before this (re)try

    try:
        scanner = scanner_factory()
        if not scanner.reinitialize(run.request):
            raise RuntimeError("scanner.reinitialize() returned False")
        scanner.start_scan_thread()
    except Exception as exc:  # surface to the run rather than raising 500 blind
        with registry.lock:
            run.error = str(exc)
            run.promoting = False
        raise HTTPException(status_code=500, detail=f"promotion failed: {exc}") from exc

    with registry.lock:
        run.scanner = scanner
        run.promoted = True
        run.promoting = False
        stopped_during_build = run.stopped

    if stopped_during_build:
        scanner.stop_scanning_thread()

    return run
