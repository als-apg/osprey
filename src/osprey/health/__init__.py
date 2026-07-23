"""Configurable health-check framework.

Exposes the result models eagerly. ``run_health_suite`` (the async runner
entry point) and ``HealthRuntime`` (the connector lifecycle context manager)
are resolved lazily via a module-level ``__getattr__`` so that importing this
package never pulls in the runner or runtime modules — and their heavier,
optional dependencies — until they are actually used.
"""

from __future__ import annotations

from typing import Any

from .models import STATUS_ICONS, CheckReport, CheckResult, Status

__all__ = [
    "STATUS_ICONS",
    "CheckReport",
    "CheckResult",
    "HealthRuntime",
    "Status",
    "run_health_suite",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve the runner and runtime symbols (PEP 562)."""
    if name == "run_health_suite":
        from .runner import run_health_suite

        return run_health_suite
    if name == "HealthRuntime":
        from .runtime import HealthRuntime

        return HealthRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
