"""OSPREY PV directory MCP server.

FastMCP server over the control system's own directory of channel addresses,
for the session's current control target. Every tool is a read.

The unconditional tools register with ``@mcp.tool()`` when their module is
imported. A gated tool module declares ``PREDICATE`` and exposes
``register(server, ctx)``; :func:`create_server` imports every module under
``tools/`` and calls ``register`` on each gated module whose predicate holds, so
adding a tool module never edits this file. Tools reach the services through
:func:`get_tool_context` at call time.

Usage:
    python -m osprey.mcp_server.pv_directory
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from fastmcp import FastMCP

from osprey.mcp_server.errors import make_error
from osprey.services.pv_directory.facts import Facts
from osprey.services.pv_directory.facts import panel_link as panel_link
from osprey.services.pv_directory.resolve import SectionError, predicates, resolve_section
from osprey.services.pv_directory.service import DirectoryService

logger = logging.getLogger("osprey.mcp_server.pv_directory")

mcp = FastMCP(
    "pv-directory",
    instructions=(
        "Read-only search of the control system's own directory of channel addresses.\n"
        "Names come from the control system, not from the curated channel list: a name "
        "that matches a word does not mean the channel measures what the word says. "
        "Each row says whether the curated list declares it (origin) and carries the "
        "panel link that shows it."
    ),
)

#: The error words a whole-call refusal uses (Contracts, Outcome).
NOT_SUPPORTED = "not_supported"
SERVICE_UNAVAILABLE = "service_unavailable"

_SUGGESTIONS = {
    NOT_SUPPORTED: [
        "The current control target has no channel directory; nothing here can be "
        "searched until the deployment configures one or the session switches target.",
    ],
    SERVICE_UNAVAILABLE: [
        "The directory service did not answer; retry later, or check the service's health.",
    ],
}


@dataclass(frozen=True)
class ToolContext:
    """What every tool reads: built once at server start from the rendered config.

    Attributes:
        predicates: The ``pv_directory`` context keys from ``resolve.py``.
        service: The process's one directory service.
        pv_facts: The PV and IOC facts over that service.
    """

    predicates: dict[str, bool]
    service: DirectoryService
    pv_facts: Facts


_context: ToolContext | None = None


def get_tool_context() -> ToolContext:
    """The context :func:`create_server` built.

    Raises:
        RuntimeError: The server was not created.
    """
    if _context is None:
        raise RuntimeError("PV directory tool context not initialized; call create_server().")
    return _context


def set_tool_context(context: ToolContext | None) -> None:
    """Install *context*, or drop it with ``None`` (tests)."""
    global _context
    _context = context


def build_tool_context(config: Mapping[str, Any], *, project_dir: Path | None) -> ToolContext:
    """The tool context for *config*, the rendered config read from *project_dir*.

    A malformed section logs a warning and holds no predicate, so no gated tool
    registers; the build refuses such a section before a server ever starts.
    """
    try:
        resolved = resolve_section(config, project_dir=project_dir)
    except SectionError as error:
        logger.warning("pv_directory section is malformed (%s); gated tools withheld", error)
        resolved = None
    service = DirectoryService(config, project_dir=project_dir)
    return ToolContext(
        predicates=predicates(resolved), service=service, pv_facts=Facts(service, config)
    )


def register_tools(server: FastMCP, context: ToolContext) -> list[str]:
    """Import every tool module; register each gated one whose predicate holds.

    Returns:
        The gated modules registered, by name.
    """
    from osprey.mcp_server.pv_directory import tools

    registered = []
    for found in sorted(pkgutil.iter_modules(tools.__path__), key=lambda found: found.name):
        module = importlib.import_module(f"{tools.__name__}.{found.name}")
        predicate = getattr(module, "PREDICATE", None)
        if predicate is not None and context.predicates[predicate]:
            module.register(server, context)
            registered.append(found.name)
    return registered


def _rendered_config() -> tuple[dict[str, Any], Path]:
    from osprey_connectors.workspace import load_osprey_config, resolve_config_path

    return load_osprey_config(), resolve_config_path().parent


def create_server() -> FastMCP:
    """Build the tool context, register the tools, and return the server."""
    from osprey.mcp_server.startup import prime_config_builder, startup_timer

    prime_config_builder()

    with startup_timer("tool_context"):
        config, project_dir = _rendered_config()
        context = build_tool_context(config, project_dir=project_dir)
        set_tool_context(context)

    with startup_timer("tool_imports"):
        gated = register_tools(mcp, context)

    logger.info("PV directory MCP server initialised (gated tools: %s)", gated or "none")
    return mcp


def refuse_unavailable(tool: str, answer: Mapping[str, Any]) -> NoReturn:
    """Refuse a call whose whole answer is ``unavailable``, with *answer* under ``details``.

    ``not_supported`` when no backend serves the current target (no source, the
    installed connectors lack the directory modules, or the backend cannot be
    built: the answer has no source, or the no-source form's ``service: "none"``);
    ``service_unavailable`` when a backend was asked and did not answer.
    """
    source = answer.get("source")
    no_backend = not isinstance(source, Mapping) or source.get("service") == "none"
    error_type = NOT_SUPPORTED if no_backend else SERVICE_UNAVAILABLE
    reason = answer.get("reason") or "no answer"
    make_error(error_type, f"{tool}: {reason}", list(_SUGGESTIONS[error_type]), dict(answer))
