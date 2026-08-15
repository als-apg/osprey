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

    **Why this loader reads the chain backwards.** The chain's precedence is
    ascending everywhere in OSPREY — ``.env.shared`` defaults below the
    host-local ``.env``, so a key set in both resolves to the ``.env`` value.
    Every other delivery mechanism realizes that by putting the *lower*
    precedence file first: a compose ``env_file:`` list is
    ``[.env.shared, .env]`` and the ``override=True`` loaders load shared then
    local, because in both the last write to a key wins. This loader is
    ``override=False`` — an already-set variable is left alone, so the *first*
    file to supply a key is the one that sticks. Under first-wins semantics the
    same precedence requires the opposite file order, so we load ``.env``
    first and ``.env.shared`` second. Same contract, inverted mechanics; the
    file order here is not a bug to be "fixed" into agreement with the compose
    lists.

    ``override=False`` also means a variable the shell already exports outranks
    both files, which is the property the empty-string clearing below protects:
    empty-string env vars (set by Claude Code's ``${VAR:-}`` when the shell
    variable is unset) are cleared first, so they read as absent rather than as
    a shell value that shadows the whole chain.
    """
    from osprey.utils.dotenv import chain_files
    from osprey.utils.workspace import repo_root_for_config

    config_path = os.environ.get("OSPREY_CONFIG", "")
    if config_path:
        config_dir = Path(config_path).parent
        roots = [repo_root_for_config(config_path), config_dir]
    else:
        roots = [Path.cwd()]

    # chain_files() returns ascending precedence (.env.shared, .env); this
    # loader consumes it reversed — see the docstring.
    chain = []
    for root in roots:
        chain = chain_files(root)
        if chain:
            break
    if not chain:
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    # Claude Code passes ${VAR:-} as VAR="" when the shell variable is
    # unset.  Clear these so dotenv can fill them from the chain.
    empty_keys = [k for k, v in os.environ.items() if v == ""]
    for key in empty_keys:
        os.environ.pop(key)

    for env_file in reversed(chain):
        load_dotenv(env_file, override=False)
