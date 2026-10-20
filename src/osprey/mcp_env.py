"""Dotenv loader for OSPREY MCP servers.

Claude Code does not read ``.env`` files — it only passes through shell
environment variables via ``${VAR:-}`` references in ``.mcp.json``.

This module bridges the gap so that the settings in the deployment repo's
env chain (``.env.shared`` defaults plus the host-local ``.env``) are
available to every MCP server at startup, without requiring users to
manually ``source .env`` or export keys in their shell profile.

Usage (in each ``__main__.py``)::

    from osprey.mcp_env import load_dotenv_from_project

    def main() -> None:
        load_dotenv_from_project()
        ...
"""

import os
from pathlib import Path


def load_dotenv_from_project() -> None:
    """Load the env chain from the repo derived from ``OSPREY_CONFIG``.

    There is one env chain per deployment and it lives at the repo root, beside
    ``profile.yml`` — never in the render, which every ``osprey build``
    re-creates from scratch. ``OSPREY_CONFIG`` points at the render
    (``<repo>/build/config.yml``), so the repo root is one level above it.

    Resolution order for finding the chain:
      1. The repo root above ``OSPREY_CONFIG``, when that config sits in the
         build zone
      2. The directory holding ``OSPREY_CONFIG`` — a container project
         directory, whose root *is* the render
      3. Current working directory, when ``OSPREY_CONFIG`` is unset

    The first of those holding any chain file wins outright; the chain is never
    stitched together across two of them.

    Shell exports outrank both files, ``.env`` outranks ``.env.shared``, and
    the shell-only launch variables are never applied
    (:func:`osprey_connectors.config.apply_env_chain` with ``override=False``).

    That a variable the shell already exports outranks both files is the
    property the empty-string clearing below protects:
    empty-string env vars (set by Claude Code's ``${VAR:-}`` when the shell
    variable is unset) are cleared first, so they read as absent rather than as
    a shell value that shadows the whole chain.
    """
    from osprey_connectors.config import apply_env_chain
    from osprey_connectors.dotenv import chain_files
    from osprey_connectors.workspace import repo_root_for_config

    config_path = os.environ.get("OSPREY_CONFIG", "")
    if config_path:
        config_dir = Path(config_path).parent
        roots = [repo_root_for_config(config_path), config_dir]
    else:
        roots = [Path.cwd()]

    chain = []
    for root in roots:
        chain = chain_files(root)
        if chain:
            break
    if not chain:
        return

    # Claude Code passes ${VAR:-} as VAR="" when the shell variable is
    # unset.  Clear these so dotenv can fill them from the chain.
    empty_keys = [k for k, v in os.environ.items() if v == ""]
    for key in empty_keys:
        os.environ.pop(key)

    apply_env_chain(os.environ, chain, override=False)
