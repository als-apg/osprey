"""MCP protocol introspection — discover server tools via ``tools/list``.

Spawns each MCP server from ``.mcp.json`` using ``stdio_client()``,
calls ``session.list_tools()``, and returns enriched metadata including
tool names and descriptions.  Results are cached keyed on ``.mcp.json``
mtime so subsequent calls are instant.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

# Module-level cache: {"mtime": float, "servers": list[dict]}
_cache: dict[str, object] = {}

# Bash-style ${VAR:-default} pattern
_ENV_VAR_RE = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def _resolve_env_value(value: str) -> str:
    """Resolve ``${VAR:-default}`` patterns against the current environment."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)

    return _ENV_VAR_RE.sub(_replace, value)


def _resolve_env(env: dict[str, str] | None) -> dict[str, str]:
    """Resolve all env values and merge with inherited environment.

    MCP servers need the parent process environment (PATH, etc.) plus
    the server-specific overrides from .mcp.json.
    """
    merged = dict(os.environ)
    if env:
        for key, value in env.items():
            merged[key] = _resolve_env_value(str(value))
    return merged


def _categorize_server(args: list[str]) -> str:
    """Categorize server as 'osprey' or 'external' based on module path."""
    for arg in args:
        if isinstance(arg, str) and arg.startswith("osprey."):
            return "osprey"
    return "external"


async def introspect_server(
    name: str,
    server_config: dict,
    cwd: str,
    *,
    timeout: float = 10.0,
) -> dict:
    """Spawn an MCP server, call ``tools/list``, return metadata.

    Parameters
    ----------
    name
        Server key from ``.mcp.json`` (e.g. "controls", "workspace").
    server_config
        Server spec dict with ``command``, ``args``, ``env``.
    cwd
        Working directory for spawning the server.
    timeout
        Maximum seconds to wait for introspection.

    Returns
    -------
    dict
        Server metadata with tool list.  ``tools`` is ``None`` on failure.
    """
    command = server_config.get("command", "")
    args = server_config.get("args", [])
    env = server_config.get("env")
    category = _categorize_server(args)

    base_info = {
        "name": name,
        "category": category,
        "command": command,
        "args": args,
        "env": server_config.get("env", {}),
        "description": "",
        "tools": None,
        "tool_count": None,
    }

    try:
        resolved_env = _resolve_env(env)
        params = StdioServerParameters(
            command=command,
            args=args,
            env=resolved_env,
            cwd=cwd,
        )

        async with asyncio.timeout(timeout):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    init_result = await session.initialize()
                    base_info["description"] = getattr(
                        init_result, "instructions", None
                    ) or ""
                    result = await session.list_tools()

                    tools = [
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                        }
                        for tool in result.tools
                    ]

                    base_info["tools"] = tools
                    base_info["tool_count"] = len(tools)
                    logger.debug("introspect %s: %d tools discovered", name, len(tools))

    except TimeoutError:
        logger.warning("introspect %s: timed out after %.1fs", name, timeout)
    except Exception:
        logger.warning("introspect %s: failed", name, exc_info=True)

    return base_info


async def introspect_all_servers(
    mcp_json: dict,
    project_dir: str,
    *,
    timeout: float = 10.0,
) -> list[dict]:
    """Introspect all servers from ``.mcp.json`` in parallel.

    Parameters
    ----------
    mcp_json
        Parsed contents of ``.mcp.json``.
    project_dir
        Project directory (used as cwd for spawning servers).
    timeout
        Per-server timeout in seconds.

    Returns
    -------
    list[dict]
        Server metadata for each server, with tools where available.
    """
    servers = mcp_json.get("mcpServers", {})
    if not servers:
        return []

    tasks = [
        introspect_server(name, config, project_dir, timeout=timeout)
        for name, config in servers.items()
    ]

    return list(await asyncio.gather(*tasks))


async def get_mcp_servers_cached(
    mcp_json_path: str | Path,
    project_dir: str,
    *,
    timeout: float = 10.0,
) -> list[dict]:
    """Return cached MCP server metadata, re-introspecting on file change.

    Checks ``.mcp.json`` mtime and returns cached results if unchanged.

    Parameters
    ----------
    mcp_json_path
        Path to ``.mcp.json``.
    project_dir
        Project directory for spawning servers.
    timeout
        Per-server timeout in seconds.
    """
    import json

    path = Path(mcp_json_path)
    if not path.exists():
        return []

    current_mtime = path.stat().st_mtime

    cached_mtime = _cache.get("mtime")
    if cached_mtime == current_mtime and "servers" in _cache:
        logger.debug("mcp introspection cache hit (mtime=%.3f)", current_mtime)
        return _cache["servers"]  # type: ignore[return-value]

    mcp_json = json.loads(path.read_text())
    servers = await introspect_all_servers(mcp_json, project_dir, timeout=timeout)

    _cache["mtime"] = current_mtime
    _cache["servers"] = servers
    logger.info(
        "mcp introspection complete: %d servers, %d with tools",
        len(servers),
        sum(1 for s in servers if s["tools"] is not None),
    )

    return servers


def clear_cache() -> None:
    """Clear the introspection cache (useful for testing)."""
    _cache.clear()
