"""The optimizer MCP server's HTTP boundary: the one :func:`request` every tool sends through."""

from __future__ import annotations

from typing import Any, NoReturn

import httpx

from osprey.mcp_server.control_system.target_banner import resolve_target_situation
from osprey.mcp_server.errors import make_error
from osprey.optimizer_connection import OPTIMIZER_URL_ENV_VAR, optimizer_url
from osprey_connectors.workspace import load_osprey_config

#: Seconds to wait for an answer. Longer than the service's own deadlines on a
#: machine read (the connector pool's 60 s, the environment's 75 s backstop), so
#: the service's typed error arrives before this client gives up.
TIMEOUT_S = 80.0

_UNREACHABLE_HINTS = [
    "Confirm the optimizer service is running (`osprey status`).",
    f"Check the {OPTIMIZER_URL_ENV_VAR} env var: it overrides the configured URL.",
    "Without it, the URL is services.optimizer.port on loopback; a profile with no "
    "`optimizer:` block deploys no optimizer.",
]


def control_target() -> str:
    """The session's control target (va, standin or live): the machine every read describes."""
    return resolve_target_situation().control_target


def _detail(resp: httpx.Response) -> Any:
    try:
        body = resp.json()
    except ValueError:
        return resp.text
    return body.get("detail") if isinstance(body, dict) else body


def _refuse(method: str, path: str, resp: httpx.Response) -> NoReturn:
    """Raise the error envelope for a non-2xx answer, typed by what the caller can do."""
    status = resp.status_code
    detail = _detail(resp)
    reason = detail or resp.reason_phrase
    message = f"The optimizer answered {method} {path} with HTTP {status}: {reason}"
    details = {"status": status, "detail": detail}
    if status == 404:
        make_error("not_found", message, details=details)
    if status == 422:
        make_error("validation_error", message, details=details)
    make_error("optimizer_error", message, details=details)


async def request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    header: str | None = None,
) -> Any:
    """Send one request to the optimizer and return its parsed JSON body.

    With *header*, return ``(body, value of that response header or None)``.

    Raises ``ToolError`` through ``make_error``: ``optimizer_unreachable`` when
    no answer came (a timeout says the service may still be working);
    ``not_found`` (404), ``validation_error`` (422) or ``optimizer_error`` (any
    other non-2xx, or a 2xx whose body is not JSON), with ``details`` holding
    the HTTP ``status`` and the service's ``detail``.
    """
    url = f"{optimizer_url(load_osprey_config())}{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            resp = await client.request(method, url, params=params)
    except httpx.TimeoutException as exc:
        if isinstance(exc, httpx.ConnectTimeout):
            make_error(
                "optimizer_unreachable",
                f"Could not reach the optimizer at {url}: {exc}",
                _UNREACHABLE_HINTS,
            )
        make_error(
            "optimizer_unreachable",
            f"The optimizer at {url} took the request but gave no answer within "
            f"{TIMEOUT_S:.0f} s. It may still be working on it (a machine read that "
            "hangs is cut off by the service itself).",
            [
                "Retry once; a machine read that keeps timing out points at the machine, "
                "not the optimizer.",
                "Check the target's channels with the control-system tools.",
            ],
            details={"timeout_s": TIMEOUT_S},
        )
    except httpx.HTTPError as exc:
        make_error(
            "optimizer_unreachable",
            f"Could not reach the optimizer at {url}: {exc}",
            _UNREACHABLE_HINTS,
        )
    if not resp.is_success:
        _refuse(method, path, resp)
    try:
        body = resp.json()
    except ValueError:
        make_error(
            "optimizer_error",
            f"The optimizer answered {method} {path} with HTTP {resp.status_code} "
            f"but a body that is not JSON: {resp.text[:200]!r}",
            details={"status": resp.status_code, "detail": resp.text[:200]},
        )
    if header is not None:
        return body, resp.headers.get(header)
    return body
