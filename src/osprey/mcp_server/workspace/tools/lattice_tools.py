"""MCP tools: lattice dashboard — init, state, params, figures, baseline, settings.

Ten tools wrapping the lattice dashboard HTTP surface:
  - ``lattice_init``: Load a lattice file into the dashboard.
  - ``lattice_state``: Get the current lattice state.
  - ``lattice_set_param``: Set a magnet family parameter.
  - ``lattice_refresh``: Trigger figure recomputation.
  - ``lattice_get_figure``: Fetch rendered Plotly figure JSON.
  - ``lattice_get_data``: Fetch raw numerical data behind a figure.
  - ``lattice_set_baseline``: Snapshot current state as baseline.
  - ``lattice_clear_baseline``: Remove the baseline snapshot.
  - ``lattice_get_settings``: Fetch computation/display settings.
  - ``lattice_update_settings``: Deep-merge new settings (partial update).
"""

import json
import logging
import os

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
    port = int(os.environ.get("OSPREY_LATTICE_DASHBOARD_PORT", ld.get("port", _DEFAULT_PORT)))
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
        elif method == "PUT":
            resp = await client.put(url, json=json_body)
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
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
                ["The dashboard starts automatically with 'osprey web'."],
            )
    except httpx.HTTPStatusError as exc:
        return make_error(
                "lattice_error",
                f"Failed to load lattice: {exc.response.text}",
                ["Check that the lattice file path is correct and readable."],
            )
    except Exception as exc:
        logger.exception("lattice_init failed")
        return make_error("lattice_error", str(exc))


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
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
                ["The dashboard starts automatically with 'osprey web'."],
            )
    except Exception as exc:
        logger.exception("lattice_state failed")
        return make_error("lattice_error", str(exc))


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
        return make_error(
                "lattice_error",
                f"Failed to set parameter: {exc.response.text}",
                ["Check that the family name exists in the lattice."],
            )
    except httpx.ConnectError:
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
    except Exception as exc:
        logger.exception("lattice_set_param failed")
        return make_error("lattice_error", str(exc))


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
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
    except Exception as exc:
        logger.exception("lattice_refresh failed")
        return make_error("lattice_error", str(exc))


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
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
    except Exception as exc:
        logger.exception("lattice_set_baseline failed")
        return make_error("lattice_error", str(exc))


@mcp.tool()
async def lattice_get_figure(name: str) -> str:
    """Fetch the rendered Plotly figure JSON for a named figure.

    Valid figure names: ``optics``, ``resonance``, ``chromaticity``,
    ``footprint``, ``da``, ``lma``. Returns a ``lattice_error`` if the
    figure hasn't been computed yet (call ``lattice_refresh`` first) or
    the name is unknown.

    Args:
        name: Figure name.

    Returns:
        JSON-serialized Plotly figure dict (data + layout), suitable
        for rendering or inspection.
    """
    try:
        result = await _dashboard_request("GET", f"/api/figures/{name}")
        return json.dumps(result, default=str)
    except httpx.HTTPStatusError as exc:
        return make_error(
                "lattice_error",
                f"Failed to fetch figure '{name}': {exc.response.text}",
                [
                    "Valid names: optics, resonance, chromaticity, footprint, da, lma.",
                    "If the figure status is 'idle' or 'stale', call lattice_refresh first.",
                ],
            )
    except httpx.ConnectError:
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
    except Exception as exc:
        logger.exception("lattice_get_figure failed")
        return make_error("lattice_error", str(exc))


@mcp.tool()
async def lattice_get_data(name: str) -> str:
    """Fetch the raw numerical arrays behind a figure.

    Returns the unprocessed JSON the figure builder consumes — useful
    when you need the underlying arrays (e.g. s-positions, beta values,
    survival mask) rather than the Plotly rendering. Valid names match
    ``lattice_get_figure``.

    Args:
        name: Figure name.

    Returns:
        JSON-serialized raw data dict.
    """
    try:
        result = await _dashboard_request("GET", f"/api/data/{name}")
        return json.dumps(result, default=str)
    except httpx.HTTPStatusError as exc:
        return make_error(
                "lattice_error",
                f"Failed to fetch data for '{name}': {exc.response.text}",
                [
                    "Valid names: optics, resonance, chromaticity, footprint, da, lma.",
                    "If the figure status is 'idle' or 'stale', call lattice_refresh first.",
                ],
            )
    except httpx.ConnectError:
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
    except Exception as exc:
        logger.exception("lattice_get_data failed")
        return make_error("lattice_error", str(exc))


@mcp.tool()
async def lattice_get_settings() -> str:
    """Fetch the current computation and display settings.

    Returns the merged settings dict — defaults overlaid with any
    updates made via ``lattice_update_settings``. Groups include ``da``
    (dynamic aperture resolution, bisect count, etc.) and ``lma``
    (momentum aperture scan parameters).

    Returns:
        JSON-serialized settings dict.
    """
    try:
        result = await _dashboard_request("GET", "/api/settings")
        return json.dumps(result, default=str)
    except httpx.ConnectError:
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
    except Exception as exc:
        logger.exception("lattice_get_settings failed")
        return make_error("lattice_error", str(exc))


@mcp.tool()
async def lattice_update_settings(settings: dict) -> str:
    """Deep-merge new settings into the dashboard state (partial update).

    Only the keys you pass are changed — missing keys keep their
    current value. The server validates and clamps values, so the
    response echoes the actually applied settings. Any figure touched
    by a changed key flips to ``stale`` and must be refreshed.

    Args:
        settings: Nested dict keyed by group (e.g. ``da``, ``lma``),
            each holding the keys to update. Example:
            ``{"da": {"n_angles": 25}}``.

    Returns:
        JSON with the applied (post-validation) settings dict.
    """
    try:
        result = await _dashboard_request("PUT", "/api/settings", json_body={"settings": settings})
        return json.dumps(result, default=str)
    except httpx.HTTPStatusError as exc:
        return make_error(
                "lattice_error",
                f"Failed to update settings: {exc.response.text}",
                ["Check the settings shape — nested dict keyed by group name."],
            )
    except httpx.ConnectError:
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
    except Exception as exc:
        logger.exception("lattice_update_settings failed")
        return make_error("lattice_error", str(exc))


@mcp.tool()
async def lattice_clear_baseline() -> str:
    """Remove the comparison baseline.

    After clearing, figure refreshes stop rendering baseline overlays.
    No-op if no baseline is set.

    Returns:
        JSON confirming the clear.
    """
    try:
        result = await _dashboard_request("DELETE", "/api/baseline")
        return json.dumps(result, default=str)
    except httpx.ConnectError:
        return make_error(
                "service_unavailable",
                "Lattice dashboard server is not running.",
            )
    except Exception as exc:
        logger.exception("lattice_clear_baseline failed")
        return make_error("lattice_error", str(exc))
