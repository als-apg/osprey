"""Which backend runs agent Python code, and for how long.

The executor's vocabulary, read from ``execution.execution_method`` and
``python_executor.execution_timeout_seconds``. Spelled here, below both the CLI
(which reports the backend at build time) and the executor (which runs it), so
the two cannot disagree.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from osprey_connectors.workspace import resolve_config_path

__all__ = [
    "DEFAULT_EXECUTION_TIMEOUT_SECONDS",
    "EXECUTION_METHOD_SUBPROCESS",
    "resolve_execution_method",
]

logger = logging.getLogger("osprey.utils.execution_method")

# OSPREY runs agent Python code in exactly one backend: a subprocess on the host.
# ``local`` is an accepted alias for that same backend; ``container`` names a
# Jupyter kernel gateway OSPREY does not ship.
EXECUTION_METHOD_SUBPROCESS = "subprocess"

#: Wall-clock budget one agent Python execution gets when the deployment does not
#: set ``python_executor.execution_timeout_seconds``. Defined once here and read
#: by the executor so both ends of the timeout agree.
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 600

# Module-level latch so the ``container`` deprecation is logged once per process
# rather than on every config read (the executor resolves per tool call).
# Tests reset it via ``osprey.utils.execution_method._container_method_warned = False``.
_container_method_warned = False


def _describe_config_source(source: str | None) -> str:
    """Describe where an execution method came from, for log messages.

    Args:
        source: Caller-supplied description (usually a config file path). When
            ``None``, the active config path is resolved on a best-effort basis.

    Returns:
        str: A human-readable config source, never empty.
    """
    if source:
        return str(source)
    try:
        return str(resolve_config_path())
    except Exception:  # pragma: no cover - defensive: never fail a config read
        return "config.yml"


def resolve_execution_method(config: Mapping[str, Any] | None, *, source: str | None = None) -> str:
    """Normalize ``execution.execution_method`` to the backend that actually runs.

    This is the single choke point for the execution vocabulary. Every reader of
    ``execution.execution_method`` goes through it so legacy configs keep loading
    while the config file stops claiming a backend OSPREY does not ship:

    - ``subprocess`` — the honest name; returned as-is.
    - ``local`` — an alias for the same subprocess backend; mapped silently.
    - ``container`` — names a Jupyter kernel gateway OSPREY does not ship, so it
      fails outright at execution time. Mapped to ``subprocess`` with a one-time
      deprecation warning, because that value's behavior genuinely changes.
    - unset (missing, ``None``, or blank) — defaults to ``subprocess``.

    Args:
        config: Full configuration mapping (the one containing the ``execution``
            section), or ``None``. Anything without a usable ``execution`` section
            resolves to the default.
        source: Optional description of where the config came from (typically the
            config file path), used in the deprecation warning. Defaults to the
            active config path.

    Returns:
        str: Always ``"subprocess"`` — the only execution backend OSPREY ships.

    Raises:
        ValueError: If ``execution.execution_method`` is set to any other value.

    Example:
        >>> resolve_execution_method({"execution": {"execution_method": "local"}})
        'subprocess'
    """
    global _container_method_warned

    execution = config.get("execution") if isinstance(config, dict) else None
    raw = execution.get("execution_method") if isinstance(execution, dict) else None

    if raw is None:
        return EXECUTION_METHOD_SUBPROCESS

    if not isinstance(raw, str):
        raise ValueError(
            f"Invalid execution.execution_method: {raw!r} "
            f"(in {_describe_config_source(source)}). Expected 'subprocess'."
        )

    method = raw.strip().lower()

    if method in ("", EXECUTION_METHOD_SUBPROCESS, "local"):
        return EXECUTION_METHOD_SUBPROCESS

    if method == "container":
        if not _container_method_warned:
            _container_method_warned = True
            logger.warning(
                "execution.execution_method: 'container' is deprecated (in %s). "
                "OSPREY ships no containerized Python backend, so this value never "
                "worked; agent Python code now runs via the subprocess backend. "
                "Set execution.execution_method: subprocess.",
                _describe_config_source(source),
            )
        return EXECUTION_METHOD_SUBPROCESS

    raise ValueError(
        f"Invalid execution.execution_method: {raw!r} "
        f"(in {_describe_config_source(source)}). Expected 'subprocess' "
        f"(legacy 'local' and 'container' are also accepted)."
    )
