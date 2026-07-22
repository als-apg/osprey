"""Shared scaffolding for the channel-finder MCP server variants.

The three channel-finder pipelines
(``channel_finder_hierarchical``, ``channel_finder_middle_layer``,
``channel_finder_in_context``) each ship an identical server bootstrap: an
entry point, a ``create_server()`` startup sequence, and per-context config
loading and path resolution. Only the FastMCP instance, the pipeline-specific
server context, and the tool modules differ between variants — those stay in
each package. This module holds the common bootstrap so the variants become
thin parameterizations of it.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from fastmcp import FastMCP


def run_cf_main(server_module: str) -> None:
    """Entry point for ``python -m osprey.mcp_server.channel_finder_<variant>``.

    Args:
        server_module: Dotted path of the variant's ``server`` module, which
            must expose a ``create_server()`` factory (e.g.
            ``"osprey.mcp_server.channel_finder_hierarchical.server"``).
    """
    from osprey.mcp_env import load_dotenv_from_project
    from osprey.mcp_server.startup import redirect_logging_to_stderr

    load_dotenv_from_project()
    redirect_logging_to_stderr()

    server = importlib.import_module(server_module).create_server()
    server.run()


def build_cf_server(
    *,
    mcp: FastMCP,
    logger: logging.Logger,
    initialize_context: Callable[[], object],
    import_tools: Callable[[], None],
    ready_message: str,
) -> FastMCP:
    """Run the shared channel-finder startup sequence and return ``mcp``.

    Primes the config builder, initializes the pipeline-specific server
    context, wires up the workspace singletons, then imports the variant's
    tool modules (each self-registers via ``@mcp.tool()``).

    Args:
        mcp: The variant's module-level ``FastMCP`` instance.
        logger: The variant's module logger, used for the ready message.
        initialize_context: Zero-arg callable that lazily imports and
            initializes the variant's server context.
        import_tools: Zero-arg callable that lazily imports the variant's
            tool modules so their ``@mcp.tool()`` decorators run.
        ready_message: Message logged once all tools are registered.
    """
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
    )
    from osprey.utils.workspace import resolve_workspace_root

    prime_config_builder()

    initialize_context()

    workspace_root = resolve_workspace_root()
    initialize_workspace_singletons(workspace_root)

    import_tools()

    logger.info(ready_message)
    return mcp


def load_cf_config(logger: logging.Logger) -> dict[str, Any]:
    """Load ``config.yml`` from the ``OSPREY_CONFIG`` env var or cwd.

    Returns an empty dict when the file is missing.
    """
    config_path = Path(
        os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml")))
    )
    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        logger.info("config loaded from %s", config_path)
    else:
        logger.warning("Config file not found: %s", config_path)

    return raw


def resolve_cf_path(path_str: str) -> str:
    """Resolve a path relative to the ``OSPREY_CONFIG`` file's directory."""
    config_path = Path(
        os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml")))
    )
    p = Path(path_str)
    if not p.is_absolute():
        p = config_path.parent / p
    return str(p.resolve())
