"""Dotenv loader for OSPREY MCP servers.

Claude Code does not read ``.env`` files — it only passes through shell
environment variables via ``${VAR:-}`` references in ``.mcp.json``.

This module bridges the gap so that the API keys in the deployment repo's
``.env`` are available to every MCP server at startup, without requiring
users to manually ``source .env`` or export keys in their shell profile.

Usage (in each ``__main__.py``)::

    from osprey.mcp_env import load_dotenv_from_project

    def main() -> None:
        load_dotenv_from_project()
        ...
"""

import os
from pathlib import Path


def load_dotenv_from_project() -> None:
    """Load ``.env`` from the repo derived from ``OSPREY_CONFIG``.

    There is one ``.env`` per deployment and it lives at the repo root, beside
    ``profile.yml`` — never in the render, which every ``osprey build``
    re-creates from scratch. ``OSPREY_CONFIG`` points at the render
    (``<repo>/build/config.yml``), so the repo root is one level above it.

    Resolution order for finding ``.env``:
      1. The repo root above ``OSPREY_CONFIG``, when that config sits in the
         build zone
      2. The directory holding ``OSPREY_CONFIG`` — a container project
         directory, whose root *is* the render
      3. Current working directory, when ``OSPREY_CONFIG`` is unset

    Empty-string env vars (set by Claude Code's ``${VAR:-}`` when the
    shell variable is unset) are cleared first so that ``python-dotenv``
    can fill them from ``.env`` with ``override=False``.
    """
    from osprey.utils.workspace import repo_root_for_config

    config_path = os.environ.get("OSPREY_CONFIG", "")
    if config_path:
        config_dir = Path(config_path).parent
        candidates = [repo_root_for_config(config_path) / ".env", config_dir / ".env"]
    else:
        candidates = [Path.cwd() / ".env"]

    env_file = next((candidate for candidate in candidates if candidate.is_file()), None)
    if env_file is None:
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    # Claude Code passes ${VAR:-} as VAR="" when the shell variable is
    # unset.  Clear these so dotenv can fill them from .env.
    empty_keys = [k for k, v in os.environ.items() if v == ""]
    for key in empty_keys:
        os.environ.pop(key)

    load_dotenv(env_file, override=False)
