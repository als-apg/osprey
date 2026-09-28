"""A benchmarked project's secrets and providers.

A benchmark acts on a project directory it is handed, so it reads that
project's ``.env`` and ``api.providers`` rather than the ambient
configuration of whatever directory it happens to run from.

Public API:
    project_dotenv        — the project's ``.env`` as a mapping
    project_env           — the process environment over the project's ``.env``
    expand_api_providers  — ``api.providers`` with ``${VAR}`` references expanded
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey_connectors.config import resolve_env_vars


def project_dotenv(project_dir: Path) -> dict[str, str]:
    """Read the project's ``.env``, so a benchmark run needs no exported shell vars.

    Returns an empty mapping when there is no file, or when ``python-dotenv``
    is not installed — the caller falls back to the process environment.
    """
    env_file = project_dir / ".env"
    if not env_file.is_file():
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    return {key: value for key, value in dotenv_values(env_file).items() if value is not None}


def project_env(project_dir: Path) -> dict[str, str]:
    """Overlay the process environment on the project's ``.env``.

    ``os.environ`` wins over the project ``.env``, so a sweep can redirect a
    provider for one run without editing the deployment's file.

    Args:
        project_dir: The benchmarked project's directory (holding ``config.yml``).

    Returns:
        The merged mapping of variable names to values.
    """
    return {**project_dotenv(project_dir), **os.environ}


def expand_api_providers(config: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Return a config's ``api.providers`` with ``${VAR}`` references expanded.

    A reference whose variable ``env`` does not define is kept verbatim, so a
    caller can tell an unset reference from an empty value.

    Args:
        config: A parsed ``config.yml``.
        env: The variables references expand against, typically :func:`project_env`.

    Returns:
        The expanded providers mapping; ``{}`` when the config names no providers.
    """
    providers = (config.get("api") or {}).get("providers") or {}
    expanded: dict[str, Any] = resolve_env_vars(providers, environ=env)
    return expanded
