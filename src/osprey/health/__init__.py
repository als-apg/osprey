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

#: Public name -> the submodule of this package that defines it. Entries are
#: resolved on first attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {
    "run_health_suite": ".runner",
    "HealthRuntime": ".runtime",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
