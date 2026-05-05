"""Dotenv loader for OSPREY MCP servers.

Claude Code does not read ``.env`` files — it only passes through shell
environment variables via ``${VAR:-}`` references in ``.mcp.json``.

This module bridges the gap so that API keys written to ``.env`` by
``osprey build`` are available to every MCP server at startup, without
requiring users to manually ``source .env`` or export keys in their
shell profile.

Usage (in each ``__main__.py``)::

    from osprey.mcp_env import load_dotenv_from_project

    def main() -> None:
        load_dotenv_from_project()
        ...
"""

import os
from pathlib import Path


def load_dotenv_from_project() -> None:
    """Load ``.env`` from the project directory derived from ``OSPREY_CONFIG``.

    Resolution order for finding ``.env``:
      1. Parent directory of the path in ``OSPREY_CONFIG``
      2. Current working directory

    Empty-string env vars (set by Claude Code's ``${VAR:-}`` when the
    shell variable is unset) are cleared first so that ``python-dotenv``
    can fill them from ``.env`` with ``override=False``.
    """
    config_path = os.environ.get("OSPREY_CONFIG", "")
    if config_path:
        env_file = Path(config_path).parent / ".env"
    else:
        env_file = Path.cwd() / ".env"

    if not env_file.is_file():
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
