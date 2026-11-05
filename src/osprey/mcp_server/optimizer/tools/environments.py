"""MCP tools: read the optimizer's environments.

==========================  ===================================================
Tool                        Optimizer route
==========================  ===================================================
list_environments           GET  /environments
get_environment             GET  /environments/{name}
environment_status          GET  /environments/{name}/status          (target)
read_variables              GET  /environments/{name}/variables       (target)
read_observables            GET  /environments/{name}/observables     (target)
list_presets                GET  /environments/{name}/presets
initial_bounds              GET  /environments/{name}/initial-bounds  (target)
resolve_bounds              POST /environments/{name}/resolve-bounds  (computes only)
==========================  ===================================================

A route marked ``target`` reads a machine, and the service requires it to be
named; a tool given no ``target`` names the deployment's current one.
"""

import json
from typing import Any, Literal
from urllib.parse import quote

from osprey.mcp_server.optimizer.server import mcp
from osprey.mcp_server.optimizer.server_context import default_target, get_json, request

Target = Literal["va", "standin", "live"]


def _env(name: str) -> str:
    return f"/environments/{quote(name, safe='')}"


def _machine(target: Target | None) -> dict[str, Any]:
    return {"target": target or default_target()}


@mcp.tool()
async def list_environments() -> str:
    """List the optimization environments the optimizer offers (name and summary of each)."""
    return json.dumps(await get_json("/environments"))


@mcp.tool()
async def get_environment(name: str) -> str:
    """Read one environment's definition: its metadata and its variable and observable definitions (safety ranges, step sizes). Reads no machine.

    Args:
        name: Environment name, from list_environments.
    """
    return json.dumps(await get_json(_env(name)))


@mcp.tool()
async def environment_status(name: str, target: Target | None = None) -> str:
    """Check whether an environment is reachable and healthy on a machine.

    Args:
        name: Environment name.
        target: Machine to check (va, standin or live). Default: the session's current control target.
    """
    return json.dumps(await get_json(f"{_env(name)}/status", params=_machine(target)))


@mcp.tool()
async def read_variables(name: str, names: list[str], target: Target | None = None) -> str:
    """Read the current readbacks of an environment's variables on a machine.

    Each reading carries the value plus connected/timestamp metadata; an unreachable channel comes back with ``connected: false``.

    Args:
        name: Environment name.
        names: Variable names, as get_environment lists them.
        target: Machine to read (va, standin or live). Default: the session's current control target.
    """
    params = {**_machine(target), "names": names}
    return json.dumps(await get_json(f"{_env(name)}/variables", params=params))


@mcp.tool()
async def read_observables(name: str, names: list[str], target: Target | None = None) -> str:
    """Read the current values of an environment's observables (objective measurements) on a machine.

    Args:
        name: Environment name.
        names: Observable names, as get_environment lists them.
        target: Machine to read (va, standin or live). Default: the session's current control target.
    """
    params = {**_machine(target), "names": names}
    return json.dumps(await get_json(f"{_env(name)}/observables", params=params))


@mcp.tool()
async def list_presets(name: str) -> str:
    """List the run presets an environment offers, by name (variables, overrides, settings); empty when it offers none.

    Args:
        name: Environment name.
    """
    return json.dumps(await get_json(f"{_env(name)}/presets"))


@mcp.tool()
async def initial_bounds(name: str, preset: str | None = None, target: Target | None = None) -> str:
    """Resolve the starting bounds for each variable from its current setpoint on a machine.

    Without a preset every variable gets its safety range; with one, only the preset's variables, with its overrides applied around the current setpoints. Returns ``bounds`` and ``current_values``.

    Args:
        name: Environment name.
        preset: Preset name, from list_presets.
        target: Machine to read setpoints from (va, standin or live). Default: the session's current control target.
    """
    params = _machine(target)
    if preset is not None:
        params["preset"] = preset
    return json.dumps(await get_json(f"{_env(name)}/initial-bounds", params=params))


@mcp.tool()
async def resolve_bounds(
    name: str,
    current_values: dict[str, float],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Compute the bounds the optimizer would use for given current values and overrides, clamped to each variable's safety range. Computes only: reads and writes no machine.

    Args:
        name: Environment name.
        current_values: Variable name -> current value, for every variable to resolve.
        overrides: Variable name -> override: ``min``/``max``, or one of ``window_fraction`` (fraction of the safety range) / ``window_abs`` (absolute half-width) centered on the current value; optional ``step_size``. Every overridden variable needs a current value.
    """
    body = {"current_values": current_values, "overrides": overrides or {}}
    resp = await request("POST", f"{_env(name)}/resolve-bounds", json=body)
    return json.dumps(resp.json())
