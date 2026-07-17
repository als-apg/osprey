"""The bridge's in-memory run registry and lifecycle state machine.

``Run`` tracks one scan from intent through completion. Creating a run never
touches the ``PlanRunner`` seam (see ``plan_runner.py``) — that only happens once a
run is *launched* via ``do_launch``, the single choke point that starts a
real scan. This module is deliberately free of bluesky/ophyd/tiled imports so
the lifecycle core stays importable — and unit testable with a
``FakePlanRunner`` — before those dependencies are ever installed. It does use
``fastapi.HTTPException`` for its error semantics (mirroring ``security.py``),
since FastAPI is a core bridge dependency, not one of the deferred ones.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

if TYPE_CHECKING:
    from .plan_runner import PlanRunner

logger = logging.getLogger("osprey.services.bluesky_bridge.runs")


@dataclass
class Run:
    """One scan-run's lifecycle state.

    ``request`` is deliberately untyped here (see Task 1.5's pydantic
    ``RunRequest``) so this module never has to import FastAPI/pydantic.
    """

    id: str
    request: Any
    created_at: float = field(default_factory=time.time)
    launched: bool = False
    launching: bool = False  # guards the launch critical section (see do_launch)
    runner: PlanRunner | None = None
    stopped: bool = False
    error: str | None = None
    # Which sanctioned launch path launched this run, e.g. "agent" (the
    # token-gated `launch_run` MCP tool route). Set when the pending run is
    # created or, at latest, by whatever launches the run.
    launched_by: str | None = None

    @property
    def run_uid(self) -> str | None:
        """The underlying scan's run identifier, once the scan has started."""
        return self.runner.last_run_uid if self.runner is not None else None

    @property
    def status(self) -> str:
        """Derive the run's lifecycle state: pending -> running -> completed | stopped | error.

        ``stopped`` is checked before ``launched`` so a pending run stopped
        before it was ever launched correctly reports "stopped" rather than a
        stale "pending" — `do_launch` refuses to (re-)launch a stopped run
        either way, so "pending" would be misleading once that door is
        permanently closed.
        """
        if self.error:
            return "error"
        if self.stopped:
            return "stopped"
        if not self.launched:
            return "pending"
        if self.runner is not None:
            if self.runner.is_run_active():
                return "running"
            if self.runner.error_message is not None:
                return "error"
        return "completed"

    def to_dict(self) -> dict:
        status = self.status
        out: dict[str, Any] = {"id": self.id, "status": status}
        # Always present, never absent: `False` both before launch (no
        # runner yet) and for a runner that doesn't expose the attribute
        # (`FakePlanRunner`) — a missing key would read as "unknown", not the
        # same thing as "Tiled persistence is fine" (FR5). `bool(...)` makes
        # the field's JSON type structural rather than a property of whatever
        # duck-typed runner happens to be attached.
        out["tiled_degraded"] = bool(getattr(self.runner, "tiled_degraded", False))
        if self.launched and self.runner is not None:
            out["completion"] = self.runner.estimate_current_completion()
        if self.launched_by:
            out["launched_by"] = self.launched_by
        run_uid = self.run_uid
        if run_uid:
            out["run_uid"] = run_uid
        if self.error:
            out["error"] = self.error
        elif status == "error" and self.runner is not None:
            out["error"] = self.runner.error_message or (
                f"scan ended in state {self.runner.current_state!r}"
            )
        return out


class RunRegistry:
    """Thread-safe in-memory store of `Run` objects, keyed by id.

    ``lock`` also guards `do_launch`'s launching/launched/stopped
    check-and-set below — the registry is in-memory only, so a single lock
    per bridge process is enough to admit exactly one launch per run.
    """

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self.lock = Lock()

    def add(self, request: Any, launched_by: str | None = None) -> Run:
        """Record a launch intent and return the new `Run`. Never touches the runner seam."""
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


def do_launch(
    run: Run,
    runner_factory: Callable[[], PlanRunner],
    *,
    validator: Callable[[Run], None] | None = None,
) -> Run:
    """The single choke point that starts a real scan.

    Callers (the token-gated ``POST /runs/{id}/launch`` route, task 1.5) must
    already have enforced their sanctioned human decision — the launch token
    (``security.verify_launch_token``) plus `launch_run`'s in-tool
    ``writes_enabled`` re-check — before calling this.

    ``validator``, if given, is called with ``run`` BEFORE ``registry.lock``
    is acquired at all — deliberately outside the lock below, not folded into
    its check-and-set. That lock exists to guard a fast in-memory
    check-and-set only (see the next paragraph); a validator may do
    unbounded author-controlled work (task 2.5's session-plan validation gate
    re-scans and re-``exec_module``s every validated session plan via
    ``get_facility_plans()``), and running that under the same lock
    ``stop_run`` (``app.py``) shares would let a concurrent launch's
    validator delay an emergency stop of an unrelated, already-running plan —
    exactly the kind of latency this lock must never carry. It must raise
    ``fastapi.HTTPException`` to refuse the launch or return `None` to
    allow it. Dependency-injected (mirroring ``runner_factory``) so this
    module never has to import bluesky/``plan_loader`` itself — `None` (the
    default) skips the call entirely, preserving every existing caller that
    doesn't pass one. Running before the lock means a validator also runs
    ahead of the ``launched``/``launching``/``stopped`` checks below — a run
    that is both stopped and unvalidated gets the validator's 409 rather than
    the ``stopped`` 409, which is an immaterial precedence swap (both are
    409 rejections). The validator's own correctness never depends on the
    lock: it re-reads/re-hashes the plan file fresh off disk, and
    ``runner_factory().reinitialize()`` below re-resolves the plan registry
    again regardless, which is the actual TOCTOU-safe barrier against a plan
    that changes between the validator call and the scan actually starting.

    ``runner_factory`` builds a fresh `PlanRunner` OUTSIDE the lock (it may be
    slow, e.g. constructing a real bluesky `RunEngine`); the lock only guards
    the launching/launched/stopped check-and-set, so two concurrent launches
    of the same run can't both start a scan, and a stopped run can never be
    launched. Publishes ``launched=True`` before clearing ``launching`` so a
    concurrent launch always observes either "not yet launching" or
    "launched" — never a window where both are false.

    A stop can race the unlocked build/start window above: ``stop_run``
    (``app.py``) may run while this runner is still being built, see
    ``launched=False``/``runner=None`` (nothing to stop yet), and merely
    record ``run.stopped = True``. Left unhandled, this runner would then
    finish starting and get published anyway — a live, untracked scan behind
    a run that reports "stopped". So publishing re-acquires the lock and
    re-checks ``run.stopped`` in the same critical section that sets
    ``runner``/``launched``; if a stop landed during the build, the
    just-started runner is stopped immediately after releasing the lock.

    A *different* failure mode: ``runner.start_run_thread()`` itself raises
    after partially starting something (e.g. a real bluesky ``PlanRunner``
    spawned its daemon thread, which then failed before the thread became
    observably "active"). Without a guard, the except branch below records
    ``run.error`` and returns 500 — but ``run.runner`` is never published
    (the run reports "error", not "running"), so nothing else could ever call
    ``stop_run_thread()`` on that runner: a live, untracked, unstoppable
    scan. So the except branch stops whatever `runner_factory()` managed to
    build, unconditionally, before surfacing the error — safe even if nothing
    actually started, since every `PlanRunner.stop_run_thread()` must
    already tolerate being called on an inactive runner.
    """
    if validator is not None:
        validator(run)

    with registry.lock:
        if run.launched:
            raise HTTPException(status_code=409, detail=f"run {run.id!r} already launched")
        if run.launching:
            raise HTTPException(
                status_code=409, detail=f"run {run.id!r} launch already in progress"
            )
        if run.stopped:
            raise HTTPException(
                status_code=409, detail=f"run {run.id!r} was stopped; cannot launch"
            )
        run.launching = True
        run.error = None  # clear any prior failed-launch error before this (re)try

    runner: PlanRunner | None = None
    try:
        runner = runner_factory()
        if not runner.reinitialize(run.request):
            reason = getattr(runner, "error_message", None) or "no error_message set"
            raise RuntimeError(f"runner.reinitialize() returned False: {reason}")
        # Threads the registry's durable run id into the RunEngine start doc
        # (see `BlueskyPlanRunner._run`), so a Tiled-persisted run can still be
        # found after the in-memory registry — and `run_uid` with it — is
        # gone. Not part of the `PlanRunner` Protocol: `FakePlanRunner` ignores it.
        runner.osprey_run_id = run.id  # type: ignore[attr-defined]
        runner.start_run_thread()
    except Exception as exc:  # surface to the run rather than raising 500 blind
        if runner is not None:
            try:
                runner.stop_run_thread()
            except Exception:
                logger.warning(
                    "do_launch: stop_run_thread() on a failed-start runner for run %r also raised",
                    run.id,
                    exc_info=True,
                )
        with registry.lock:
            run.error = str(exc)
            run.launching = False
        raise HTTPException(status_code=500, detail=f"launch failed: {exc}") from exc

    with registry.lock:
        run.runner = runner
        run.launched = True
        run.launching = False
        stopped_during_build = run.stopped

    if stopped_during_build:
        runner.stop_run_thread()

    return run
