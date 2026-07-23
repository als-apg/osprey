"""Core ``file_system`` health category.

Reports on the project's on-disk layout and headroom: that the configured
project root and agent-data directory are present and writable, that a ``.env``
file exists, that a configured registry file is present, and that the working
filesystem has adequate free space.

Row names, statuses and messages are pinned by the CLI contract tests.
Config-dependent rows degrade gracefully when the loaded config is absent: a
missing ``project_root`` yields a single ``project_paths`` warning rather than
an error.

Rows emitted:

* ``project_paths`` — warning when no ``project_root`` is configured; error on
  the exception path (broad guard around project-path resolution).
* ``project_root_path`` — ok when the resolved project root exists, warning
  otherwise.
* ``agent_data_dir`` — ok when the agent-data directory is writable or can be
  created; warning when it is not writable or cannot be created.
* ``env_file`` — ok when a ``.env`` file is present in the working directory,
  warning otherwise.
* ``registry_file`` — emitted only when ``config.yml`` declares
  ``registry_path``: ok when the file exists, error when configured but missing.
* ``disk_space`` — warning when free space is below 1 GB or the filesystem is at
  least 90% full, ok otherwise; warning when disk usage cannot be read.

Unlike the ``configuration`` category (which reports on config loading and so
consumes a pre-built state), this category performs genuine live disk checks and
therefore needs the working directory. The factory conforms to the
``core/__init__`` contract — ``file_system(config, context=None)`` — and accepts
an optional keyword ``cwd`` (defaulting to :func:`Path.cwd`) so the CLI can
thread a ``--project-path`` override without changing the uniform call the
runner makes.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from osprey.health.models import CheckResult, Status

_CATEGORY = "file_system"


def file_system(
    config: dict[str, Any] | None,
    context: Any = None,
    *,
    cwd: Path | None = None,
) -> Callable[[], list[CheckResult]]:
    """Build the ``file_system`` category callable.

    Args:
        config: Pre-loaded config mapping (``config.yml`` contents), or ``None``
            when config is unavailable. Config-dependent rows degrade gracefully
            when it is empty/absent.
        context: Unused; the file-system checks need no control-system connector.
        cwd: Working directory used to locate ``.env`` and ``config.yml`` and to
            sample disk usage. Defaults to :func:`Path.cwd`, resolved when the
            callable runs so the CLI may pass a project path.

    Returns:
        A no-argument callable producing the category's :class:`CheckResult` rows.
    """
    resolved_config = config or {}
    base_dir = cwd

    def _run() -> list[CheckResult]:
        return _check_file_system(resolved_config, base_dir or Path.cwd())

    return _run


def _check_file_system(config: dict[str, Any], cwd: Path) -> list[CheckResult]:
    """Produce all ``file_system`` category rows for the given config and cwd."""
    results: list[CheckResult] = []

    # Check project paths from config.
    results.extend(_check_project_paths(config, cwd))

    # Check .env file.
    env_file = cwd / ".env"
    if env_file.exists():
        results.append(CheckResult("env_file", _CATEGORY, Status.OK, ".env file found"))
    else:
        results.append(CheckResult("env_file", _CATEGORY, Status.WARNING, ".env file not found"))

    # Check registry file (if specified in config).
    try:
        config_path = cwd / "config.yml"
        if config_path.exists():
            import yaml  # type: ignore[import-untyped]

            with open(config_path) as f:
                file_config = yaml.safe_load(f)

            registry_path_str = file_config.get("registry_path")
            if registry_path_str:
                # Resolve environment variables in path.
                registry_path_str = os.path.expandvars(registry_path_str)
                registry_path = cwd / registry_path_str

                if registry_path.exists():
                    results.append(
                        CheckResult(
                            "registry_file",
                            _CATEGORY,
                            Status.OK,
                            f"Registry file found: {registry_path}",
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            "registry_file",
                            _CATEGORY,
                            Status.ERROR,
                            f"Registry file not found: {registry_path}",
                        )
                    )
    except Exception:  # noqa: BLE001 - don't fail if we can't check registry
        pass

    # Check disk space. Container volumes (incl. the OpenObserve telemetry
    # store, which has no hard size cap) grow into this filesystem, so report
    # the percentage used and warn as it fills — the honest "how full" signal.
    try:
        stat = shutil.disk_usage(cwd)
        free_gb = stat.free / (1024**3)
        pct_used = (stat.used / stat.total * 100) if stat.total else 0.0

        if free_gb < 1.0 or pct_used >= 90.0:
            results.append(
                CheckResult(
                    "disk_space",
                    _CATEGORY,
                    Status.WARNING,
                    f"Disk {pct_used:.0f}% full ({free_gb:.1f} GB free)",
                    details="Container volumes (incl. the OpenObserve store) grow into this disk.",
                )
            )
        else:
            results.append(
                CheckResult(
                    "disk_space",
                    _CATEGORY,
                    Status.OK,
                    f"Disk {pct_used:.0f}% full ({free_gb:.1f} GB free)",
                )
            )
    except Exception as e:  # noqa: BLE001 - disk sampling is best-effort
        results.append(
            CheckResult("disk_space", _CATEGORY, Status.WARNING, f"Could not check disk space: {e}")
        )

    return results


def _check_project_paths(config: dict[str, Any], cwd: Path) -> list[CheckResult]:
    """Check that project_root and the agent-data directory are valid/accessible.

    A missing ``project_root`` yields a single ``project_paths`` warning; any
    exception during resolution yields a single ``project_paths`` error.
    """
    results: list[CheckResult] = []
    try:
        # Get project_root from config (could be hardcoded or env var).
        project_root = config.get("project_root")
        if not project_root:
            results.append(
                CheckResult(
                    "project_paths", _CATEGORY, Status.WARNING, "No project_root configured"
                )
            )
            return results

        # Resolve project_root (handles ${PROJECT_ROOT} expansion).
        project_root_resolved = os.path.expandvars(str(project_root))
        project_root_path = Path(project_root_resolved)

        # Check if project_root exists.
        if project_root_path.exists():
            results.append(
                CheckResult(
                    "project_root_path",
                    _CATEGORY,
                    Status.OK,
                    f"Project root exists: {project_root_path}",
                )
            )
        else:
            results.append(
                CheckResult(
                    "project_root_path",
                    _CATEGORY,
                    Status.WARNING,
                    f"Project root does not exist: {project_root_path}",
                )
            )
            # Don't return - we can still check if it could be created.

        # Check agent data directory.
        file_paths = config.get("file_paths", {})
        agent_data_dir = file_paths.get("agent_data_dir", "_agent_data")
        agent_data_path = project_root_path / agent_data_dir

        if agent_data_path.exists():
            # Check if it's writable.
            if os.access(agent_data_path, os.W_OK):
                results.append(
                    CheckResult(
                        "agent_data_dir",
                        _CATEGORY,
                        Status.OK,
                        f"Agent data directory writable: {agent_data_path}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        "agent_data_dir",
                        _CATEGORY,
                        Status.WARNING,
                        f"Agent data directory not writable: {agent_data_path}",
                    )
                )
        else:
            # Check if parent directory exists and is writable (can we create it?).
            parent_dir = agent_data_path.parent
            if parent_dir.exists() and os.access(parent_dir, os.W_OK):
                results.append(
                    CheckResult(
                        "agent_data_dir",
                        _CATEGORY,
                        Status.OK,
                        f"Agent data directory can be created: {agent_data_path}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        "agent_data_dir",
                        _CATEGORY,
                        Status.WARNING,
                        f"Cannot create agent data directory: {agent_data_path}",
                    )
                )

    except Exception as e:  # noqa: BLE001 - any resolution failure becomes a single error row
        results.append(
            CheckResult(
                "project_paths", _CATEGORY, Status.ERROR, f"Error checking project paths: {e}"
            )
        )
    return results


__all__ = ["file_system"]
