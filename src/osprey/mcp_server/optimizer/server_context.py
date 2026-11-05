"""The optimizer MCP server's HTTP boundary.

Every tool calls :func:`request`, so every tool maps an unreachable service to
``optimizer_unreachable`` and a non-2xx answer to ``optimizer_error`` (carrying
the status and the service's ``detail``) the same way. The base URL comes from
the shared leaf :mod:`osprey.optimizer_connection`, resolved once per process.

:func:`default_target` names the machine a machine-reading route is asked
about when the caller names none: the deployment's current control target, read
the way every other holder reads it.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from osprey.mcp_server.control_system.target_banner import resolve_target_situation
from osprey.mcp_server.errors import make_error
from osprey.optimizer_connection import OPTIMIZER_URL_ENV_VAR, resolve_optimizer_url

logger = logging.getLogger("osprey.mcp_server.optimizer.server_context")

_TIMEOUT = 15.0  # seconds

_UNREACHABLE_HINTS = [
    "Confirm the optimizer service is running (`osprey status`).",
    f"Check the {OPTIMIZER_URL_ENV_VAR} env var: it overrides the configured URL.",
    "Without it, the URL is services.optimizer.port on loopback; a profile with no "
    "`optimizer:` block deploys no optimizer.",
]

_base_url: str | None = None


def base_url() -> str:
    """The optimizer base URL, resolved on first use and kept for the process."""
    global _base_url
    if _base_url is None:
        _base_url = resolve_optimizer_url()
    return _base_url


def reset_server_context() -> None:
    """Forget the resolved URL (for testing)."""
    global _base_url
    _base_url = None


def default_target() -> str:
    """The deployment's current control target (va, standin or live)."""
    return resolve_target_situation().control_target


def _body(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text


async def request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: Any = None,
) -> httpx.Response:
    """Send one request to the optimizer and return its 2xx response.

    Raises ``ToolError`` through ``make_error``: ``optimizer_unreachable`` when
    no answer came, ``optimizer_error`` for any non-2xx, with ``details``
    holding the HTTP ``status`` and the service's ``detail``.
    """
    url = f"{base_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.request(method, url, params=params, json=json)
    except httpx.HTTPError as exc:
        make_error(
            "optimizer_unreachable",
            f"Could not reach the optimizer at {url}: {exc}",
            _UNREACHABLE_HINTS,
        )
    if not resp.is_success:
        body = _body(resp)
        detail = body.get("detail") if isinstance(body, dict) else body
        make_error(
            "optimizer_error",
            f"The optimizer answered {method} {path} with HTTP {resp.status_code}: "
            f"{detail or resp.reason_phrase}",
            details={"status": resp.status_code, "detail": detail},
        )
    return resp


async def get_json(path: str, **kwargs: Any) -> Any:
    """GET *path* and return its parsed body (see :func:`request`)."""
    return _body(await request("GET", path, **kwargs))
