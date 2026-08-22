"""Shared scaffolding for the channel-finder MCP server variants.

Every channel-finder pipeline package (``channel_finder_hierarchical``,
``channel_finder_middle_layer``, ``channel_finder_in_context``,
``channel_finder_graph``) reaches its ``create_server()`` factory through the
same entry point, :func:`run_cf_main`. The three database-backed pipelines
share the whole startup sequence as well — config-builder priming, server
context, workspace singletons, tool imports — which :func:`build_cf_server`
holds so those variants become thin parameterizations of it; the graph
pipeline runs a leaner startup of its own (no workspace singletons) and takes
only the entry point. Only the FastMCP instance, the pipeline-specific server
context, and the tool modules differ between variants — those stay in each
package.
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
    from osprey.utils.logger import configure_logging

    load_dotenv_from_project()
    # stdout carries the JSON-RPC stream; configure_logging() routes records to stderr.
    configure_logging()

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

    prime_config_builder()

    initialize_context()

    initialize_workspace_singletons()

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


def _config_path() -> Path:
    return Path(os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml"))))


def resolve_cf_path(path_str: str) -> str:
    """Resolve a BUILD-OWNED relative path against the rendered config's directory.

    Use this for inputs the build produces — channel databases under ``data/``,
    which are rendered into ``<repo>/build/data/`` from the profile. Anchoring
    them on the render is right: the copy beside the config is the copy this
    deployment was built with.

    Do NOT use it for anything WRITTEN at run time. A relative path in a
    deployment repo resolves against one of two roots, and which one is not a
    detail:

    * ``data/…`` is build-owned and exists under both anchors, so either
      answer finds a real file — this function's anchor is the correct one.
    * ``var/…`` exists ONLY at the repo root. Resolved here it would land in
      ``<repo>/build/var/…``, inside the zone every ``osprey build`` wipes, and
      the loss would be silent: the store would be re-created empty on the next
      run with nothing to say it had ever held anything.

    :func:`resolve_cf_state_path` is the anchor for the second kind.
    """
    p = Path(path_str)
    if not p.is_absolute():
        p = _config_path().parent / p
    return str(p.resolve())


def resolve_cf_state_path(path_str: str) -> str:
    """Resolve a RUNTIME-WRITTEN relative path against the deployment REPO root.

    The counterpart to :func:`resolve_cf_path`, and the one to reach for
    whenever the path names something this process writes rather than reads —
    the feedback store, pending reviews, anything under ``var/``. The repo root
    is the only anchor under which durable state survives a rebuild.

    Falls back to the config's own directory when the config is not in a
    ``build/`` zone, which is what a standalone or already-flat deployment
    looks like; :func:`~osprey.utils.workspace.repo_root_for_config` makes that
    same judgement everywhere else.
    """
    from osprey.utils.workspace import repo_root_for_config

    p = Path(path_str)
    if not p.is_absolute():
        p = repo_root_for_config(_config_path()) / p
    return str(p.resolve())
