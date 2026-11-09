"""MCP protocol introspection — discover server tools via ``tools/list``.

Spawns each stdio MCP server from ``.mcp.json`` using ``stdio_client()``,
calls ``initialize`` and ``tools/list``, and returns what the server said
about itself: its name, version, protocol and instructions, the two round
trip times, and per tool its name, title, description, input schema and
annotations. An ``http`` entry is not spawned; it is listed with its URL and
``tools: None``. Results are cached keyed on ``.mcp.json`` mtime so later calls
are instant; :func:`get_mcp_servers_cached` re-introspects one server or all
on request, and :func:`peek_mcp_servers` reads the cache without spawning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

# Module-level cache: {"mtime": float, "servers": list[dict], "probed_at": float}
_cache: dict[str, object] = {}

# One introspection at a time per event loop: a drawer opened twice, or a
# re-probe while the first spawn is still running, waits for it instead of
# spawning every server again.
_locks: dict[int, asyncio.Lock] = {}

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
    transport = str(server_config.get("type") or ("http" if server_config.get("url") else "stdio"))

    base_info: dict[str, Any] = {
        "name": name,
        "category": category,
        "transport": transport,
        "command": command,
        "args": args,
        "env": server_config.get("env", {}),
        "description": "",
        "server_name": None,
        "server_version": None,
        "protocol": None,
        "instructions": "",
        "init_ms": None,
        "list_ms": None,
        "tools": None,
        "tool_count": None,
        "error": None,
    }
    if transport != "stdio":
        # Not spawned: an http server is a running service, not a child process.
        base_info["url"] = server_config.get("url")
        return base_info

    try:
        resolved_env = _resolve_env(env)
        params = StdioServerParameters(
            command=command,
            args=args,
            env=resolved_env,
            cwd=cwd,
        )

        t_spawn = time.perf_counter()
        async with (
            asyncio.timeout(timeout),
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            init_result = await session.initialize()
            t1 = time.perf_counter()
            result = await session.list_tools()
            t2 = time.perf_counter()

            instructions = getattr(init_result, "instructions", None) or ""
            info = getattr(init_result, "serverInfo", None)
            base_info["description"] = instructions
            base_info["instructions"] = instructions
            base_info["server_name"] = getattr(info, "name", None)
            base_info["server_version"] = getattr(info, "version", None)
            base_info["protocol"] = getattr(init_result, "protocolVersion", None)
            # Process spawn + MCP initialize: the cold start a session pays.
            base_info["init_ms"] = round((t1 - t_spawn) * 1000)
            base_info["list_ms"] = round((t2 - t1) * 1000)

            tools = [
                {
                    "name": tool.name,
                    "title": getattr(tool, "title", None),
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema or {},
                    "annotations": (
                        tool.annotations.model_dump(exclude_none=True)
                        if tool.annotations is not None
                        else None
                    ),
                }
                for tool in result.tools
            ]

            base_info["tools"] = tools
            base_info["tool_count"] = len(tools)
            logger.debug("introspect %s: %d tools discovered", name, len(tools))

    except TimeoutError:
        logger.warning("introspect %s: timed out after %.1fs", name, timeout)
        base_info["error"] = f"no answer within {timeout:g} s"
    except Exception as exc:
        logger.warning("introspect %s: failed", name, exc_info=True)
        base_info["error"] = f"{type(exc).__name__}: {exc}"

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


def _lock() -> asyncio.Lock:
    loop_id = id(asyncio.get_running_loop())
    lock = _locks.get(loop_id)
    if lock is None:
        lock = _locks[loop_id] = asyncio.Lock()
    return lock


async def get_mcp_servers_cached(
    mcp_json_path: str | Path,
    project_dir: str,
    *,
    timeout: float = 10.0,
    refresh: bool = False,
    server: str | None = None,
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
    refresh
        Re-introspect even when the cache is current: every server, or only
        *server* when it names one that is cached.
    server
        With *refresh*, the one server to re-introspect.
    """
    import json

    path = Path(mcp_json_path)
    if not path.exists():
        return []

    async with _lock():
        current_mtime = path.stat().st_mtime
        cached = _cache.get("servers") if _cache.get("mtime") == current_mtime else None
        if cached is not None and not refresh:
            logger.debug("mcp introspection cache hit (mtime=%.3f)", current_mtime)
            return cached  # type: ignore[return-value]

        mcp_json = json.loads(path.read_text())
        entries = mcp_json.get("mcpServers", {}) or {}
        if cached is not None and server is not None and server in entries:
            fresh = await introspect_server(server, entries[server], project_dir, timeout=timeout)
            servers = [fresh if s["name"] == server else s for s in cached]  # type: ignore[attr-defined]
        else:
            servers = await introspect_all_servers(mcp_json, project_dir, timeout=timeout)

        _cache["mtime"] = current_mtime
        _cache["servers"] = servers
        _cache["probed_at"] = time.time()
        logger.info(
            "mcp introspection complete: %d servers, %d with tools",
            len(servers),
            sum(1 for s in servers if s["tools"] is not None),
        )
        return servers


def peek_mcp_servers(mcp_json_path: str | Path) -> tuple[list[dict] | None, float | None]:
    """The cached introspection for *mcp_json_path* and when it was taken — never spawns.

    Returns ``(None, None)`` when nothing current is cached (no introspection
    yet, or ``.mcp.json`` changed since).
    """
    path = Path(mcp_json_path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, None
    if _cache.get("mtime") != mtime or "servers" not in _cache:
        return None, None
    return _cache["servers"], _cache.get("probed_at")  # type: ignore[return-value]


def clear_cache() -> None:
    """Clear the introspection cache (useful for testing)."""
    _cache.clear()
