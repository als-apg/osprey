"""Live package inventory for the visualization tool descriptions.

The visualization tools used to name a fixed package set in their descriptions
whatever the deployment had actually installed, so the agent planned imports
against a set that need not exist.

This module renders that line from the environment the *visualization sandbox*
actually uses, which is deliberately not the one
:mod:`osprey.mcp_server.python_executor.tools._package_inventory` describes:

* the Python executor runs agent code under
  :func:`~osprey.mcp_server.python_executor.executor.resolve_agent_interpreter`
  — the project's own venv whenever the project ships one;
* :func:`~osprey.mcp_server.workspace.execution.sandbox_executor.execute_sandbox_code`
  runs it under ``sys.executable`` — the MCP server's own interpreter.

A viz-tool description must therefore describe ``sys.executable``, and it must
also respect the sandbox's import allowlist: a package can be installed and
still be unimportable here, because
:func:`~osprey.mcp_server.workspace.execution.sandbox_executor.validate_sandbox_code`
rejects imports outside the allowlist before the code ever runs. The honest
inventory is the intersection of the two gates.

Rendering, the placeholder token, the names-nothing fallback, and the docstring
substitution itself are reused from the Python-executor inventory so both tool
surfaces speak one vocabulary; only the environment being enumerated differs.
"""

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any, TypeVar

from osprey.mcp_server.python_executor.tools._package_inventory import (
    FALLBACK_DESCRIPTION,
    render_package_line,
    substitute_packages,
)

logger = logging.getLogger("osprey.mcp_server.workspace.tools.sandbox_packages")

F = TypeVar("F", bound=Callable[..., Any])


@lru_cache(maxsize=1)
def describe_sandbox_packages() -> str:
    """Describe the packages visualization code can import, from the live environment.

    Enumerates the MCP server's own interpreter — the one the viz sandbox
    spawns — and keeps only names the sandbox allowlist admits. Cached for the
    process lifetime: the answer cannot change while the server runs.

    Stdlib modules are absent by construction, since
    :func:`importlib.metadata.packages_distributions` reports only
    distribution-installed packages. That is intentional; the allowlisted stdlib
    subset is described in prose by the callers instead.

    Returns:
        str: A rendered package line, or
        :data:`~osprey.mcp_server.python_executor.tools._package_inventory.FALLBACK_DESCRIPTION`
        when the environment cannot be enumerated. Never a hardcoded package list.
    """
    try:
        from importlib.metadata import packages_distributions

        from osprey.mcp_server.workspace.execution.sandbox_executor import _ALLOWED_TOP_LEVEL

        importable = sorted(_ALLOWED_TOP_LEVEL & set(packages_distributions()))
    except Exception:  # noqa: BLE001 - description generation must never break startup
        logger.warning(
            "Could not enumerate the visualization sandbox environment; "
            "tool description will not name packages",
            exc_info=True,
        )
        return FALLBACK_DESCRIPTION

    return render_package_line(importable)


def with_sandbox_packages(fn: F) -> F:
    """Substitute the live sandbox inventory into a tool's docstring.

    Apply *below* ``@mcp.tool()`` so the docstring is rewritten before FastMCP
    reads it into the tool description.

    Args:
        fn: Tool function whose docstring contains
            :data:`~osprey.mcp_server.python_executor.tools._package_inventory.PACKAGES_PLACEHOLDER`.

    Returns:
        The same function, with the placeholder replaced in-place.
    """
    return substitute_packages(fn, describe_sandbox_packages)
