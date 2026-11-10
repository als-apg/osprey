"""MCP tools: read the optimizer's environments.

==========================  ===================================================
Tool                        Optimizer route
==========================  ===================================================
list_environments           GET  /environments
get_environment             GET  /environments/{name}
environment_status          GET  /environments/{name}/status          (target)
read_observables            GET  /environments/{name}/observables     (target)
list_presets                GET  /environments/{name}/presets
initial_bounds              GET  /environments/{name}/initial-bounds  (target)
==========================  ===================================================

A route marked ``target`` reads a machine, and the service requires it to be
named; the tool names the session's control target, the one machine every
read of the session describes.
"""

import json
from typing import Any
from urllib.parse import quote

from osprey.mcp_server.optimizer.server import mcp
from osprey.mcp_server.optimizer.server_context import control_target, request


def _env(name: str) -> str:
    return f"/environments/{quote(name, safe='')}"


def _machine() -> dict[str, Any]:
    return {"target": control_target()}


@mcp.tool()
async def list_environments() -> str:
    """List the optimization environments the optimizer offers (name and summary of each)."""
    return json.dumps(await request("GET", "/environments"))


@mcp.tool()
async def get_environment(name: str) -> str:
    """Read one environment's definition: its metadata and its variable and observable definitions (safety ranges, step sizes). Reads no machine.

    Args:
        name: Environment name, from list_environments.
    """
    return json.dumps(await request("GET", _env(name)))


@mcp.tool()
async def environment_status(name: str) -> str:
    """Check whether an environment is reachable and healthy on the session's control target.

    Args:
        name: Environment name.
    """
    return json.dumps(await request("GET", f"{_env(name)}/status", params=_machine()))


@mcp.tool()
async def read_observables(name: str, names: list[str]) -> str:
    """Read the current values of an environment's observables (objective measurements) on the session's control target.

    Args:
        name: Environment name.
        names: Observable names, as get_environment lists them.
    """
    params = {**_machine(), "names": names}
    return json.dumps(await request("GET", f"{_env(name)}/observables", params=params))


@mcp.tool()
async def list_presets(name: str) -> str:
    """List the run presets an environment offers, by name (variables, overrides, settings); empty when it offers none.

    Args:
        name: Environment name.
    """
    return json.dumps(await request("GET", f"{_env(name)}/presets"))


@mcp.tool()
async def initial_bounds(name: str, preset: str | None = None) -> str:
    """Resolve the starting bounds for each variable from its current setpoint on the session's control target.

    Without a preset every variable gets its safety range; with one, only the preset's variables, with its overrides applied around the current setpoints. Returns ``bounds`` and ``current_values`` (each variable's current setpoint reading).

    Args:
        name: Environment name.
        preset: Preset name, from list_presets.
    """
    params = _machine()
    if preset is not None:
        params["preset"] = preset
    return json.dumps(await request("GET", f"{_env(name)}/initial-bounds", params=params))
