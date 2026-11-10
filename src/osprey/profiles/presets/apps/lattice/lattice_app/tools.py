"""The application's tool server: runs beside the agent, reaches the backend by the panel door.

The tools give the agent what the panel gives the operator, from the same cached results:
``lattice_state`` reads the models, the analyses and every ready scalar;
``lattice_result`` reads one analysis's tables, computing it if nobody has yet;
``lattice_export`` reads a table as the CSV or SDDS file the panel downloads. None of
them writes anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import httpx

# The routes live in the package's ``__init__``, so the tool server imports neither the
# backend nor its numerics.
from lattice_app import APP_NAME, EXPORT_ROUTE, MAX_WAIT_S, RESULT_ROUTE, STATE_ROUTE
from osprey.apps.sdk import agent_panel_headers, make_error, serve_tools, web_terminal_url

if TYPE_CHECKING:
    from fastmcp import FastMCP

#: A panel call's bound beyond what the backend itself waits, in seconds.
MARGIN_S = 10.0
#: The rows of a table a result read shows unless asked for more.
DEFAULT_MAX_ROWS = 200
#: The characters of a file an export read shows unless asked for more.
DEFAULT_MAX_CHARS = 60_000

Analysis = Literal["optics", "chromaticity", "da", "footprint", "lma"]


async def _panel_get(route: str, params: dict[str, Any], *, wait_s: float = 0.0) -> httpx.Response:
    """Read one backend route through the panel door."""
    url = f"{web_terminal_url()}/panel/{APP_NAME}{route}"
    query = {key: value for key, value in params.items() if value is not None}
    try:
        async with httpx.AsyncClient(timeout=wait_s + MARGIN_S) as client:
            response = await client.get(url, params=query, headers=agent_panel_headers())
    except httpx.HTTPError as exc:
        make_error("unreachable", f"{route} could not be reached: {exc!r}")
    if response.status_code >= 500:
        make_error("unreachable", f"{route} answered {response.status_code}: {response.text}")
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        make_error("refused", str(detail))
    return response


def thin(table: dict[str, Any], max_rows: int) -> dict[str, Any]:
    """*table* with at most *max_rows* evenly spaced rows, its first and last among them."""
    columns = table.get("columns") or {}
    total = len(next(iter(columns.values()), []))
    if max_rows < 2 or total <= max_rows:
        return {**table, "rows_total": total, "rows_shown": total}
    step = (total - 1) / (max_rows - 1)
    kept = sorted({round(index * step) for index in range(max_rows)})
    return {
        **table,
        "columns": {name: [values[index] for index in kept] for name, values in columns.items()},
        "rows_total": total,
        "rows_shown": len(kept),
    }


def register(server: FastMCP) -> None:
    """Register the application's tools on *server*."""

    @server.tool()
    async def lattice_state(model: str | None = None, energy_gev: float | None = None) -> dict:
        """Read what the Lattice panel shows: the models, and each analysis's status and scalars.

        Answers the lattice models the deployment offers (deck, engine, energies), and for
        one model every analysis it runs with its settings, deadline and status. A ready
        analysis carries its scalars (tunes, chromaticities, aperture area), so this one
        call answers most questions. Nothing is computed by this call.

        Args:
            model: The model to read; the first one when omitted.
            energy_gev: One of the model's energies; its first when omitted.
        """
        response = await _panel_get(STATE_ROUTE, {"model": model, "energy_gev": energy_gev})
        answer: dict = response.json()
        return answer

    @server.tool()
    async def lattice_result(
        analysis: Analysis,
        model: str | None = None,
        energy_gev: float | None = None,
        wait_s: float = MAX_WAIT_S,
        retry: bool = False,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> dict:
        """Read one analysis's result: its scalars and tables, as the panel plots them.

        The result is shared: computed once for the deployment and read by every terminal.
        If nobody has asked for it yet this call starts it and waits. A ``status`` of
        ``running`` or ``queued`` means it is still computing: call again. ``failed``
        carries the reason and stays failed until called with ``retry``.

        Args:
            analysis: ``optics``, ``chromaticity``, ``da`` (dynamic aperture),
                ``footprint`` (tune footprint) or ``lma`` (local momentum aperture).
            model: The model; the first one when omitted.
            energy_gev: One of the model's energies; its first when omitted.
            wait_s: Seconds to wait for a running computation, at most 45.
            retry: Start a failed analysis again.
            max_rows: Rows shown per table, evenly thinned beyond that; ``rows_total``
                says how many there are. Use ``lattice_export`` for every row.
        """
        wait = min(max(wait_s, 0.0), MAX_WAIT_S)
        params = {
            "analysis": analysis,
            "model": model,
            "energy_gev": energy_gev,
            "wait_s": wait,
            "retry": "true" if retry else None,
        }
        answer: dict = (await _panel_get(RESULT_ROUTE, params, wait_s=wait)).json()
        if isinstance(answer.get("tables"), dict):
            answer["tables"] = {
                name: thin(table, max_rows) for name, table in answer["tables"].items()
            }
        return answer

    @server.tool()
    async def lattice_export(
        analysis: Analysis,
        format: Literal["csv", "sdds"] = "csv",
        model: str | None = None,
        energy_gev: float | None = None,
        table: str | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> dict:
        """Read a ready result's table as the CSV or SDDS file the panel downloads.

        The file holds every row, with the result's scalars and provenance in its header
        (CSV comment lines, SDDS parameters). The analysis must be ready: read it with
        ``lattice_result`` first.

        Args:
            analysis: The analysis whose table to export.
            format: ``csv`` or ``sdds`` (ASCII SDDS, for the elegant toolchain).
            model: The model; the first one when omitted.
            energy_gev: One of the model's energies; its first when omitted.
            table: The table; the analysis's main one when omitted.
            max_chars: Characters of the file shown; ``truncated`` says if it was cut.
        """
        params = {
            "analysis": analysis,
            "format": format,
            "model": model,
            "energy_gev": energy_gev,
            "table": table,
        }
        response = await _panel_get(EXPORT_ROUTE, params)
        disposition = response.headers.get("content-disposition", "")
        filename = disposition.partition('filename="')[2].rstrip('"') or f"{analysis}.{format}"
        content = response.text
        return {
            "filename": filename,
            "format": format,
            "chars": len(content),
            "truncated": len(content) > max_chars,
            "content": content[:max_chars],
        }


if __name__ == "__main__":
    serve_tools(register)
