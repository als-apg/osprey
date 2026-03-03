"""MCP tools: lattice dashboard — init, state, params, refresh, baseline.

Five tools for interacting with the lattice dashboard server:
  - ``lattice_init``: Load a lattice file into the dashboard.
  - ``lattice_state``: Get the current lattice state.
  - ``lattice_set_param``: Set a magnet family parameter.
  - ``lattice_refresh``: Trigger figure recomputation.
  - ``lattice_set_baseline``: Snapshot current state as baseline.
"""

import json
import logging

import httpx

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.utils.workspace import load_osprey_config

logger = logging.getLogger("osprey.mcp_server.tools.lattice_tools")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8097
_TIMEOUT = 30.0


def _get_dashboard_url() -> str:
    config = load_osprey_config()
    ld = config.get("lattice_dashboard", {})
    host = ld.get("host", _DEFAULT_HOST)
    port = ld.get("port", _DEFAULT_PORT)
    return f"http://{host}:{port}"


async def _dashboard_request(
    method: str, path: str, json_body: dict | None = None, timeout: float = _TIMEOUT
) -> dict:
    """Make an async HTTP request to the lattice dashboard server."""
    url = f"{_get_dashboard_url()}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        if method == "GET":
            resp = await client.get(url)
        elif method == "POST":
            resp = await client.post(url, json=json_body)
        elif method == "DELETE":
            resp = await client.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def lattice_init(lattice_path: str) -> str:
    """Load a lattice file into the dashboard.

    Initializes the dashboard with the given .m lattice file path.
    Computes optics summary, discovers magnet families, auto-sets baseline,
    and triggers computation of the 4 fast figures (optics, resonance,
    chromaticity, tune footprint).

    Args:
        lattice_path: Path to a MATLAB .m lattice file (e.g. "machine_data/als.m").

    Returns:
        JSON with lattice summary including energy, tunes, chromaticity,
        magnet families, and figure computation status.
    """
    try:
        result = await _dashboard_request(
            "POST",
            "/api/state/init",
            json_body={"lattice_path": lattice_path},
            timeout=60.0,
        )
        return json.dumps(
            {
                "status": "ok",
                "summary": result.get("summary", {}),
                "families": list(result.get("families", {}).keys()),
                "message": "Lattice loaded. Fast figures are computing.",
            },
            default=str,
        )
    except httpx.ConnectError:
        return json.dumps(
            make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
                ["The dashboard starts automatically with 'osprey web'."],
            )
        )
    except httpx.HTTPStatusError as exc:
        return json.dumps(
            make_error(
                "lattice_error",
                f"Failed to load lattice: {exc.response.text}",
                ["Check that the lattice file path is correct and readable."],
            )
        )
    except Exception as exc:
        logger.exception("lattice_init failed")
        return json.dumps(make_error("lattice_error", str(exc)))


@mcp.tool()
async def lattice_state() -> str:
    """Get the current lattice state including summary, families, figure status, and baseline.

    Returns:
        JSON with full lattice state: base_lattice path, parameter overrides,
        optics summary (energy, tunes, chromaticity), magnet families,
        figure computation status, and baseline comparison data.
    """
    try:
        result = await _dashboard_request("GET", "/api/state")
        return json.dumps(result, default=str)
    except httpx.ConnectError:
        return json.dumps(
            make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
                ["The dashboard starts automatically with 'osprey web'."],
            )
        )
    except Exception as exc:
        logger.exception("lattice_state failed")
        return json.dumps(make_error("lattice_error", str(exc)))


@mcp.tool()
async def lattice_set_param(family: str, value: float) -> str:
    """Set a magnet family parameter override.

    Updates the K-value (quadrupoles) or H-value (sextupoles) for all
    elements in the named family. Marks fast figures as stale.

    Args:
        family: Magnet family name (e.g. "QF", "SD").
        value: New parameter value.

    Returns:
        JSON confirming the update with the new state summary.
    """
    try:
        await _dashboard_request(
            "POST",
            "/api/state/param",
            json_body={"family": family, "value": value},
        )
        return json.dumps(
            {
                "status": "ok",
                "family": family,
                "value": value,
                "message": "Parameter set. Fast figures marked stale — call lattice_refresh to update.",
            },
            default=str,
        )
    except httpx.HTTPStatusError as exc:
        return json.dumps(
            make_error(
                "lattice_error",
                f"Failed to set parameter: {exc.response.text}",
                ["Check that the family name exists in the lattice."],
            )
        )
    except httpx.ConnectError:
        return json.dumps(
            make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
        )
    except Exception as exc:
        logger.exception("lattice_set_param failed")
        return json.dumps(make_error("lattice_error", str(exc)))


@mcp.tool()
async def lattice_refresh(figure: str | None = None) -> str:
    """Trigger recomputation of lattice figures.

    With no arguments, refreshes all 4 fast figures (optics, resonance,
    chromaticity, footprint). Pass "da" or "fma" to run verification
    figures, or any figure name to refresh just that one.

    Args:
        figure: Optional figure name. None = all fast figures.
            "da" or "fma" for verification. Or any specific figure name.

    Returns:
        JSON confirming which figures were launched for computation.
    """
    try:
        if figure is None:
            result = await _dashboard_request("POST", "/api/refresh")
        elif figure in ("da", "fma"):
            result = await _dashboard_request("POST", "/api/verify")
        else:
            result = await _dashboard_request("POST", f"/api/refresh/{figure}")
        return json.dumps(result, default=str)
    except httpx.ConnectError:
        return json.dumps(
            make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
        )
    except Exception as exc:
        logger.exception("lattice_refresh failed")
        return json.dumps(make_error("lattice_error", str(exc)))


@mcp.tool()
async def lattice_set_baseline() -> str:
    """Snapshot the current state as the comparison baseline.

    After setting a baseline, all figures will overlay the baseline
    (dashed lines) with the current state (solid lines) for visual
    comparison of lattice modifications.

    Returns:
        JSON with the baseline summary (tunes, chromaticity, etc.).
    """
    try:
        result = await _dashboard_request("POST", "/api/baseline")
        return json.dumps(
            {
                "status": "ok",
                "baseline": result,
                "message": "Baseline set. Figures will show overlay comparison on next refresh.",
            },
            default=str,
        )
    except httpx.ConnectError:
        return json.dumps(
            make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
        )
    except Exception as exc:
        logger.exception("lattice_set_baseline failed")
        return json.dumps(make_error("lattice_error", str(exc)))
