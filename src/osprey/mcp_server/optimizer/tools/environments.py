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
named; a tool given no ``target`` names the deployment's current one.
"""

import json
from typing import Any, Literal
from urllib.parse import quote

from osprey.mcp_server.optimizer.server import mcp
from osprey.mcp_server.optimizer.server_context import default_target, request

Target = Literal["va", "standin", "live"]


def _env(name: str) -> str:
    return f"/environments/{quote(name, safe='')}"


def _machine(target: Target | None) -> dict[str, Any]:
    return {"target": target or default_target()}


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
async def environment_status(name: str, target: Target | None = None) -> str:
    """Check whether an environment is reachable and healthy on a machine.

    Args:
        name: Environment name.
        target: Machine to check (va, standin or live). Default: the session's current control target.
    """
    return json.dumps(await request("GET", f"{_env(name)}/status", params=_machine(target)))


@mcp.tool()
async def read_observables(name: str, names: list[str], target: Target | None = None) -> str:
    """Read the current values of an environment's observables (objective measurements) on a machine.

    Args:
        name: Environment name.
        names: Observable names, as get_environment lists them.
        target: Machine to read (va, standin or live). Default: the session's current control target.
    """
    params = {**_machine(target), "names": names}
    return json.dumps(await request("GET", f"{_env(name)}/observables", params=params))


@mcp.tool()
async def list_presets(name: str) -> str:
    """List the run presets an environment offers, by name (variables, overrides, settings); empty when it offers none.

    Args:
        name: Environment name.
    """
    return json.dumps(await request("GET", f"{_env(name)}/presets"))


@mcp.tool()
async def initial_bounds(name: str, preset: str | None = None, target: Target | None = None) -> str:
    """Resolve the starting bounds for each variable from its current setpoint on a machine.

    Without a preset every variable gets its safety range; with one, only the preset's variables, with its overrides applied around the current setpoints. Returns ``bounds`` and ``current_values`` (each variable's current setpoint reading).

    Args:
        name: Environment name.
        preset: Preset name, from list_presets.
        target: Machine to read setpoints from (va, standin or live). Default: the session's current control target.
    """
    params = _machine(target)
    if preset is not None:
        params["preset"] = preset
    return json.dumps(await request("GET", f"{_env(name)}/initial-bounds", params=params))
