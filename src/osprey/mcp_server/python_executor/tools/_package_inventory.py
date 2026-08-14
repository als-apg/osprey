"""Live package inventory for the Python-executor tool descriptions.

The ``execute`` tool description must never name a fixed set of packages
(``numpy``, ``pandas``, ``scipy``, ``at``, ``matplotlib``, ``plotly``): a fixed
list makes the agent reason about an import set that may not exist — a package
present on the host but missing inside the deployed container would be
described as available either way.

This module enumerates the environment that
:func:`~osprey.mcp_server.python_executor.executor.resolve_agent_interpreter`
selects — the interpreter that runs agent-authored code — and renders a short
curated line for the tool description. Enumeration happens once per server
process, at tool-module import time, because it spawns a subprocess and the
description is static for the process lifetime.

When enumeration fails the line falls back to a generic sentence that names
nothing: a vague-but-true description is strictly better than a
confident-but-wrong one.
"""

import json
import logging
import subprocess
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from osprey.mcp_server.python_executor.executor import resolve_agent_interpreter

logger = logging.getLogger("osprey.mcp_server.python_executor.package_inventory")

#: Token that :func:`with_live_packages` replaces in a tool docstring.
PACKAGES_PLACEHOLDER = "<<AVAILABLE_PACKAGES>>"

#: Used whenever the environment cannot be enumerated.  It must never name a
#: package: a stale list is the exact failure this module removes.
FALLBACK_DESCRIPTION = (
    "The set of installed packages could not be determined; import what the task "
    "needs and handle ImportError if the import is unavailable."
)

#: Upper bound on names rendered into the description before ``+N more``.
MAX_LISTED_PACKAGES = 15

#: Scientific-stack import names, most relevant first.  Present names are listed
#: ahead of the alphabetical remainder so the cap spends its budget on the
#: packages analysis code actually reaches for.
PRIORITY_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "plotly",
    "at",
    "epics",
    "lmfit",
    "h5py",
    "sympy",
    "sklearn",
    "xarray",
    "numba",
    "statsmodels",
    "seaborn",
)

_ENUMERATION_TIMEOUT_SECONDS = 20.0

# Reports top-level *import* names rather than distribution names: the agent
# writes `import at`, not `import accelerator_toolbox`.  Non-identifiers and
# private names are dropped — compiled shims such as `<hash>__mypyc` are listed
# as top-level modules but cannot be imported by name.
_ENUMERATION_SNIPPET = (
    "import json, sys\n"
    "from importlib.metadata import packages_distributions\n"
    "names = sorted(\n"
    "    n for n in packages_distributions() if n.isidentifier() and not n.startswith('_')\n"
    ")\n"
    "json.dump(names, sys.stdout)\n"
)

F = TypeVar("F", bound=Callable[..., Any])


def _enumerate_top_level_packages(interpreter: Path) -> list[str]:
    """Enumerate top-level import names installed for ``interpreter``.

    Args:
        interpreter: Python interpreter to enumerate. Runs out-of-process so the
            answer describes that environment, not the MCP server's own.

    Returns:
        list[str]: Sorted top-level import names.

    Raises:
        OSError: The interpreter could not be executed.
        subprocess.SubprocessError: The probe failed or timed out.
        ValueError: The probe emitted output that is not a JSON list of names.
    """
    completed = subprocess.run(
        [str(interpreter), "-c", _ENUMERATION_SNIPPET],
        capture_output=True,
        text=True,
        timeout=_ENUMERATION_TIMEOUT_SECONDS,
        check=True,
    )
    names = json.loads(completed.stdout)
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Package enumeration returned a non-list payload")
    return names


def render_package_line(names: Sequence[str]) -> str:
    """Render the description line for a set of installed import names.

    Args:
        names: Top-level import names available to agent code.

    Returns:
        str: A sentence naming at most :data:`MAX_LISTED_PACKAGES` packages —
        scientific stack first, then alphabetical — with a ``+N more`` tail when
        the environment holds more. Returns :data:`FALLBACK_DESCRIPTION` for an
        empty inventory, which means enumeration told us nothing usable.
    """
    unique = sorted(set(names), key=lambda name: (name.lower(), name))
    if not unique:
        return FALLBACK_DESCRIPTION

    available = set(unique)
    prioritised = [name for name in PRIORITY_PACKAGES if name in available]
    already_listed = set(prioritised)
    remainder = [name for name in unique if name not in already_listed]

    listed = (prioritised + remainder)[:MAX_LISTED_PACKAGES]
    hidden = len(unique) - len(listed)

    line = ", ".join(listed)
    if hidden > 0:
        line += f", +{hidden} more"
    return f"Packages importable in the execution environment include: {line}."


@lru_cache(maxsize=1)
def describe_available_packages() -> str:
    """Describe the packages agent code can import, from the live environment.

    Cached for the process lifetime: enumeration spawns a subprocess and the
    tool description never changes while the server runs.

    Returns:
        str: A rendered package line, or :data:`FALLBACK_DESCRIPTION` when the
        environment cannot be enumerated. Never a hardcoded package list.
    """
    try:
        interpreter = resolve_agent_interpreter()
        names = _enumerate_top_level_packages(interpreter)
    except Exception:  # noqa: BLE001 - description generation must never break startup
        logger.warning(
            "Could not enumerate the agent execution environment; "
            "tool description will not name packages",
            exc_info=True,
        )
        return FALLBACK_DESCRIPTION

    return render_package_line(names)


def substitute_packages(fn: F, describe: Callable[[], str]) -> F:
    """Replace :data:`PACKAGES_PLACEHOLDER` in ``fn``'s docstring with ``describe()``.

    The substitution mechanism shared by every tool surface that advertises a
    live inventory; only the environment ``describe`` enumerates differs. Call
    ``describe`` lazily, here, so the (cached) enumeration happens when the tool
    module is imported rather than when this module is.

    Args:
        fn: Tool function whose docstring contains :data:`PACKAGES_PLACEHOLDER`.
        describe: Returns the rendered package line for the relevant environment.

    Returns:
        The same function, with the placeholder replaced in-place.
    """
    doc = fn.__doc__
    if doc and PACKAGES_PLACEHOLDER in doc:
        fn.__doc__ = doc.replace(PACKAGES_PLACEHOLDER, describe())
    return fn


def with_live_packages(fn: F) -> F:
    """Substitute the live package inventory into a tool's docstring.

    Apply *below* ``@mcp.tool()`` so the docstring is rewritten before FastMCP
    reads it into the tool description.

    Args:
        fn: Tool function whose docstring contains :data:`PACKAGES_PLACEHOLDER`.

    Returns:
        The same function, with the placeholder replaced in-place.
    """
    return substitute_packages(fn, describe_available_packages)
