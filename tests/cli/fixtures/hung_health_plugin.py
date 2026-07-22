"""Hung health-plugin fixture — a check that never returns.

Used only by the subprocess no-hang contract test in ``test_health_cmd.py``. It
exercises the daemon-thread / ``os._exit`` guarantee that a wedged synchronous
health check can never block process exit.

``get_health_categories`` maps the ``hung`` category to a *plain sync function*
so the runner off-loads it onto a daemon thread (:func:`osprey.health.offload.run_sync`).
The function blocks forever, so on the suite deadline the thread is abandoned;
``abandoned_count() > 0`` then drives the CLI down its ``os._exit`` fallback,
which must still terminate the process.

This module is imported by the spawned CLI as a top-level module: the test puts
this directory on ``PYTHONPATH`` and lists ``hung_health_plugin`` under
``health.plugins`` in the project config.
"""

from __future__ import annotations

import threading


def _never_returns() -> list:
    """Block forever without busy-waiting; abandoned as a daemon thread on timeout."""
    threading.Event().wait()
    return []  # never reached


def get_health_categories() -> dict:
    """Expose the single never-returning ``hung`` category."""
    return {"hung": _never_returns}
