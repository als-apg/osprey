"""Drive one catalog plan to completion on a real RunEngine, in-process.

Shared by the plan-execution tests (`test_exemplar_plans.py`,
`test_orm_plan_integration.py`), whose subject is the shipped plan bodies
themselves — does `orm` actually run, does `grid_scan` actually emit the points
it promises — not any bridge machinery.

Deliberately built on `qserver_startup.build_plan_functions`, the same wrapper
the queueserver worker's namespace is assembled from, so these tests exercise
the production call shape: a ``**kwargs``-only generator function whose kwargs
ARE the plan's PARAMS fields, validated by the plan's own pydantic schema on
first iteration. A hand-rolled `spec.plan(devices, params)` call here would
test a path nothing in production takes.

Not a test module (no `test_` prefix, so pytest never collects it): it is a
helper the plan tests import.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Mapping
from typing import Any

from bluesky import RunEngine

from osprey.services.bluesky_bridge import qserver_startup
from osprey.services.bluesky_bridge.live_rows import LiveRowRecorder

# How long to wait for an async device factory to connect. Generous — these
# are in-process mock devices, so this only ever fires on a genuine hang.
CONNECT_TIMEOUT = 30.0


def run_plan(
    plan_name: str,
    plan_args: dict[str, Any],
    *,
    devices: Any,
    plans: Mapping[str, Any],
    timeout: float = 15.0,
) -> str:
    """Run *plan_name* to completion and return the RunEngine's run uid.

    Rows land in `live_rows` under that uid (a `LiveRowRecorder` with no
    explicit key), which is what lets the caller assert on the data the plan
    produced.

    Args:
        plan_name: Catalog name, as it appears in *plans*.
        plan_args: The plan's PARAMS fields, unwrapped — the same shape a queue
            item's ``kwargs`` carry.
        devices: A device mapping, or a callable returning one (sync) or an
            awaitable of one (`devices/mock.py`'s `build_devices` is
            `async def`, and the devices it builds must be connected on the
            RunEngine's own loop, which is where the run will drive them).
        plans: The `{name: PlanSpec}` catalog to expose.
        timeout: Seconds to let the plan run before declaring it hung.

    Raises:
        TimeoutError: the plan did not finish within *timeout*.
        Exception: whatever the plan itself raised, re-raised unchanged — a
            failing plan must surface its own error, not a wrapper's summary.
    """
    # `context_managers=[]` disables the RunEngine's default SIGINT handling,
    # which only works on the main thread; the run is driven from the worker
    # thread below so the timeout above can actually be enforced.
    run_engine = RunEngine(context_managers=[])

    started: dict[str, str] = {}
    recorder = LiveRowRecorder()

    def _observe(name: str, doc: dict[str, Any]) -> None:
        if name == "start":
            started["uid"] = doc["uid"]
        recorder(name, doc)

    run_engine.subscribe(_observe)

    resolved = devices() if callable(devices) else devices
    if inspect.isawaitable(resolved):
        resolved = asyncio.run_coroutine_threadsafe(resolved, run_engine.loop).result(
            timeout=CONNECT_TIMEOUT
        )

    plan_functions = qserver_startup.build_plan_functions(dict(resolved), plans=plans)
    plan_function = plan_functions[plan_name]

    failure: dict[str, BaseException] = {}

    def _drive() -> None:
        try:
            run_engine(plan_function(**plan_args))
        except Exception as exc:  # re-raised on the calling thread below
            failure["exc"] = exc

    thread = threading.Thread(target=_drive, name=f"drive-{plan_name}", daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"plan {plan_name!r} did not complete within {timeout}s")
    if "exc" in failure:
        raise failure["exc"]

    return started["uid"]
