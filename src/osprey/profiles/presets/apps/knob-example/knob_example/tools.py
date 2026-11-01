"""The application's tool server: runs beside the agent, reaches the backend by the panel door.

``set_corrector`` is a write because the application's ``app.yml`` declares it under
``tools.approval``, so its server lists it in ``writes:``. It reaches the backend only with
the approval it was called under, whose fields the approval hook filled in.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastmcp import FastMCP

from knob_example import APP_NAME, STATE_ROUTE, WRITE_ROUTE
from osprey.apps.sdk import (
    ApprovalRefused,
    agent_panel_headers,
    check_approved_call,
    make_error,
    serve_tools,
    web_terminal_url,
)

#: A panel call's bound, in seconds.
TIMEOUT_S = 10.0


async def _panel_call(method: str, route: str, **kwargs: Any) -> dict[str, Any]:
    """Call one backend route through the panel door and answer its JSON object."""
    url = f"{web_terminal_url()}/panel/{APP_NAME}{route}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.request(method, url, headers=agent_panel_headers(), **kwargs)
    except httpx.HTTPError as exc:
        make_error("unreachable", f"{route} could not be reached: {exc}")
    if response.status_code >= 500:
        make_error("unreachable", f"{route} answered {response.status_code}: {response.text}")
    if response.is_error:
        raise ApprovalRefused("backend_refused", f"{route} refused: {response.text}")
    answer: dict[str, Any] = response.json()
    return answer


def build_server() -> FastMCP:
    """The application's tools."""
    server = FastMCP(APP_NAME)

    @server.tool()
    async def read_corrector() -> dict[str, Any]:
        """Read the corrector this application shows, on the session's control target."""
        return await _panel_call("GET", STATE_ROUTE)

    @server.tool()
    async def set_corrector(
        value: float,
        approved_target: str | None = None,
        approved_readback: float | None = None,
    ) -> dict[str, Any]:
        """Set the corrector this application shows on the session's control target.

        Args:
            value: The setpoint to write.
            approved_target: Filled in by the approval hook; leave unset.
            approved_readback: Filled in by the approval hook; leave unset.
        """
        bound = check_approved_call(
            "set_corrector",
            {
                "value": value,
                "approved_target": approved_target,
                "approved_readback": approved_readback,
            },
        )
        if not bound.bound or "approved_readback" not in bound.state_fields:
            raise ApprovalRefused(
                "approval_missing", "not run: the approval bound no readback to write over."
            )
        body = {
            "value": value,
            "target": bound.approved_target,
            "readback": bound.state_fields["approved_readback"],
        }
        return await _panel_call("POST", WRITE_ROUTE, json=body)

    return server


if __name__ == "__main__":
    serve_tools(build_server)
