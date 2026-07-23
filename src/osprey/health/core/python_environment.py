"""Core ``python_environment`` health category.

Reports the running Python version (warn below 3.11), whether a virtual
environment is active, and whether the core runtime dependencies import.
Emits three rows:

* ``python_version`` — ``warning`` below 3.11, else ``ok``;
* ``virtual_environment`` — ``warning`` when not running inside a venv, else ``ok``;
* ``core_dependencies`` — ``error`` listing any of ``click``/``rich``/``yaml``/
  ``jinja2``/``litellm`` that fail to import, else ``ok``.

This category inspects only the interpreter, so its factory ignores both the
loaded config and the health runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "python_environment"

#: Runtime dependencies whose absence makes the environment unusable.
CORE_DEPENDENCIES: tuple[str, ...] = ("click", "rich", "yaml", "jinja2", "litellm")


def python_environment(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
) -> CategoryCallable:
    """Build the ``python_environment`` category callable.

    Args:
        config: Loaded config mapping. Unused — the checks inspect only the
            interpreter.
        context: Health runtime. Unused — no control-system connector is needed.

    Returns:
        A no-argument callable returning the category's check results.
    """

    def _run() -> list[CheckResult]:
        return _check_python_environment()

    return _run


def _check_python_environment() -> list[CheckResult]:
    """Run the three interpreter checks and return their results."""
    return [_check_python_version(), _check_virtual_environment(), _check_core_dependencies()]


def _check_python_version() -> CheckResult:
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if version < (3, 11):
        return CheckResult(
            "python_version",
            CATEGORY,
            Status.WARNING,
            f"Python {version_str} (recommended: 3.11+)",
        )
    return CheckResult("python_version", CATEGORY, Status.OK, f"Python {version_str}")


def _check_virtual_environment() -> CheckResult:
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        return CheckResult("virtual_environment", CATEGORY, Status.OK, "Virtual environment active")
    return CheckResult(
        "virtual_environment", CATEGORY, Status.WARNING, "Not in a virtual environment"
    )


def _check_core_dependencies() -> CheckResult:
    missing = [dep for dep in CORE_DEPENDENCIES if importlib.util.find_spec(dep) is None]
    if missing:
        return CheckResult(
            "core_dependencies",
            CATEGORY,
            Status.ERROR,
            f"Missing dependencies: {', '.join(missing)}",
        )
    return CheckResult(
        "core_dependencies",
        CATEGORY,
        Status.OK,
        f"All {len(CORE_DEPENDENCIES)} core dependencies installed",
    )
