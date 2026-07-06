"""The Bluesky bridge's FastAPI app: wires the run registry, promote gate, and
scanner seam (``runs.py``, ``security.py``, ``scanner.py``) into HTTP routes.

Two processes, one machine (see PLAN.md's Technical Architecture): this app
runs in a separate container from OSPREY's own venv, reachable only over
HTTP plus the ``X-Promote-Token`` header. It stays import-clean of bluesky/
ophyd/tiled in Phase 1 — ``_scanner_factory`` defaults to the no-op
``FakeScanner`` so this app is runnable and manually smoke-testable
(``GET /health``, even a real ``promote``) before the bluesky-backed
``Scanner`` (Phase 2) exists. Phase 2's wiring swaps the factory via
``set_scanner_factory`` once that implementation lands.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

from .runs import do_promote, registry
from .scanner import FakeScanner, Scanner
from .security import verify_promote_token

app = FastAPI(title="OSPREY Bluesky Bridge")

# The `Scanner` implementation `do_promote` builds for every promotion.
_scanner_factory: Callable[[], Scanner] = FakeScanner


def set_scanner_factory(factory: Callable[[], Scanner]) -> None:
    """Override the `Scanner` implementation `do_promote` builds.

    The Phase 2 real bluesky-backed factory (and tests) call this instead of
    reaching into the private module global directly.
    """
    global _scanner_factory
    _scanner_factory = factory


class RunRequest(BaseModel):
    """A scan launch intent (`POST /runs`).

    Phase 1 has no plan registry yet (`plans.py` arrives in task 2.3), so this
    is intentionally generic: `plan_name` names a plan the eventual registry
    will resolve, and `plan_args` is forwarded to the scanner unmodified via
    `do_promote` -> `Scanner.reinitialize(run.request)`.
    """

    plan_name: str
    plan_args: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/runs")
def create_run(request: RunRequest) -> dict:
    """Record a launch *intent*. Never touches the scanner seam."""
    return registry.add(request).to_dict()


@app.get("/runs")
def list_runs(limit: int = 20) -> list[dict]:
    """This bridge process's tracked runs, newest first (in-memory only)."""
    return [run.to_dict() for run in registry.list(limit=limit)]


@app.get("/runs/{run_id}")
def get_run_status(run_id: str) -> dict:
    return registry.get(run_id).to_dict()


@app.post("/runs/{run_id}/promote")
def promote_run(run_id: str, x_promote_token: str = Header(default="")) -> dict:
    """Promote an intent to a real scan launch. Token-gated (see `security.py`).

    Callable only by holders of `BLUESKY_PROMOTE_TOKEN` — in practice, the
    `launch_scan` MCP tool, whose own invocation already required a human
    approval prompt (PreToolUse) plus an in-tool `writes_enabled` re-check.
    """
    verify_promote_token(x_promote_token)
    run = registry.get(run_id)
    promoted_run = do_promote(run, _scanner_factory)
    # Only recorded once do_promote actually succeeds (it raises 409/500
    # otherwise) — a rejected promote attempt must not mark the run as
    # launched by anything.
    if promoted_run.launched_by is None:
        promoted_run.launched_by = "agent"
    return promoted_run.to_dict()


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    """Abort a running scan. Not token-gated — halting is always allowed.

    Coordinates with `do_promote`'s unlocked scanner-build window under
    `registry.lock`: if a promote is concurrently mid-build (``promoting``
    set, ``scanner``/``promoted`` not yet published), this just records
    ``stopped`` and `do_promote` itself stops the just-started scanner once
    it re-checks `stopped` at publish time — see `runs.py`.
    """
    run = registry.get(run_id)
    with registry.lock:
        scanner_to_stop = run.scanner if run.promoted else None
        run.stopped = True
    if scanner_to_stop is not None:
        scanner_to_stop.stop_scanning_thread()
    return run.to_dict()


@app.get("/plans")
def list_plans() -> list:
    """Registered scan plans. Stub returning `[]` until task 2.3's plan registry lands."""
    return []
