"""Async HTTP client for dispatching prompts to dispatch worker services."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import httpx


class DispatchError(Exception):
    """Raised on connection errors, timeouts, or a retryable (5xx) worker error."""


class AuthError(Exception):
    """Raised when the worker returns HTTP 401 Unauthorized."""


class FatalDispatchError(Exception):
    """Raised on a deterministic 4xx rejection from the worker.

    A 4xx means the request itself is malformed/oversized (e.g. an input-files
    batch over cap, or a body past the size limit): an identical redispatch is
    rejected identically, so this is NOT retryable. ``error_code`` carries the
    machine-readable ``detail`` whitelisted from the worker's 4xx JSON body (one
    of :data:`KNOWN_REJECTION_CODES`), or ``None`` when the body carried no
    recognised code. The caller surfaces this as a pool error carrying
    ``error_code`` rather than routing it through the retry/drop policy.
    """

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


# Machine-readable 4xx ``detail`` codes the client is willing to propagate. Kept
# in lock-step with ``dispatch_worker.input_files_policy`` (DETAIL_CAP_EXCEEDED /
# DETAIL_INVALID). Whitelisting — rather than echoing whatever ``detail`` the
# body carries — keeps arbitrary worker-internal text out of the dispatcher.
KNOWN_REJECTION_CODES: frozenset[str] = frozenset(
    {"input_files_cap_exceeded", "input_files_invalid"}
)


def _extract_rejection_code(response: httpx.Response) -> str | None:
    """Whitelist the machine-readable ``detail`` code from a worker 4xx body.

    Returns the code only when it is one of :data:`KNOWN_REJECTION_CODES`; any
    other, absent, or unparseable ``detail`` yields ``None`` (generic). Never
    returns arbitrary body text — a 4xx body may still carry internal detail.
    """
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    return detail if detail in KNOWN_REJECTION_CODES else None


async def dispatch_to_worker(
    url: str,
    prompt: str,
    allowed_tools: list[str],
    token: str,
    timeout: float = 30.0,
    surface_prompt: str | None = None,
    surface_tools: list[str] | None = None,
    input_files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """POST a prompt to a dispatch worker's /dispatch endpoint.

    Args:
        url: Base URL of the dispatch worker service (e.g. "http://dispatch-worker-1:9190").
        prompt: The prompt text to dispatch.
        allowed_tools: List of tool names the agent is allowed to use.
        token: Bearer token for authentication.
        timeout: Base request timeout (seconds). Used as the read/pool timeout; the
            connect and write timeouts are widened independently (see below) so a
            large ``input_files`` upload is not killed by a modest read timeout.
        surface_prompt: Optional per-surface system-prompt fragment. Omitted from the
            request payload entirely when ``None`` or empty, so a worker predating this
            field sees an unchanged request.
        surface_tools: Optional per-surface tool-scope narrowing. Omitted from the
            request payload entirely when ``None`` or empty, same as ``surface_prompt``.
        input_files: Optional caller-supplied file batch (already validated upstream),
            each item ``{"filename", "mime", "content_b64", "ingest"}``. Omitted from
            the payload when ``None`` or empty, so a worker predating the field sees an
            unchanged request.

    Returns:
        Response JSON dict (typically contains ``run_id`` and ``status``).

    Raises:
        AuthError: If the server returns HTTP 401.
        FatalDispatchError: On a non-401 4xx (deterministic, non-retryable rejection),
            carrying a whitelisted ``error_code`` when the body supplied one.
        DispatchError: On connection errors, timeout, or a retryable 5xx.
    """
    dispatch_url = url.rstrip("/") + "/dispatch"
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict[str, Any] = {"prompt": prompt, "allowed_tools": allowed_tools}
    # Additive fields: only included when actually set, so an absent-surface
    # trigger (the overwhelming majority today) produces byte-identical
    # requests to what a pre-surface caller would send.
    if surface_prompt:
        payload["surface_prompt"] = surface_prompt
    if surface_tools:
        payload["surface_tools"] = surface_tools
    if input_files:
        payload["input_files"] = input_files

    # A dispatch body carrying input_files can reach ~24 MB. httpx's single-float
    # timeout would apply that same short window to the write phase and abort the
    # upload; give connect/write generous windows while keeping the read/pool
    # timeout at ``timeout`` (the worker returns 202 as soon as it enqueues).
    client_timeout = httpx.Timeout(timeout, connect=10.0, write=120.0)
    try:
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            response = await client.post(dispatch_url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise DispatchError(f"Timeout dispatching to {dispatch_url}: {exc}") from exc
    except httpx.ConnectError as exc:
        raise DispatchError(f"Connection error dispatching to {dispatch_url}: {exc}") from exc
    except httpx.RequestError as exc:
        raise DispatchError(f"Request error dispatching to {dispatch_url}: {exc}") from exc

    if response.status_code == 401:
        raise AuthError(f"Unauthorized (401) from {dispatch_url}")

    # A non-401 4xx is a deterministic rejection of THIS request — never retry it,
    # and never drop it to a silent None. Surface a typed FatalDispatchError
    # carrying a whitelisted machine-readable code (or None) so the caller records
    # it as a non-retryable pool error. 413 (body too large) is included here.
    if 400 <= response.status_code < 500:
        raise FatalDispatchError(
            f"HTTP {response.status_code} from worker",
            error_code=_extract_rejection_code(response),
        )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Surface only the status code — never echo the worker's response body
        # into the dispatcher's registry history; it can carry a stack trace or
        # other internal detail. Raise a typed DispatchError so the on_error
        # policy treats it like any other (retryable) dispatch failure.
        raise DispatchError(f"HTTP {response.status_code} from worker") from exc
    return cast(dict[str, Any], response.json())


async def fetch_worker_runs(url: str, token: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Fetch recent runs from a worker's /dashboard/runs endpoint.

    The /dashboard/runs endpoint is bearer-gated like the other worker endpoints,
    so the dispatcher's worker token is forwarded.

    Args:
        url: Base URL of the worker service (e.g. "http://dispatch-worker-1:9190").
        token: Bearer token for the worker (DISPATCH_WORKER_TOKEN).
        timeout: Request timeout in seconds.

    Returns:
        List of run dicts (run_id, status, created_at, etc.).

    Raises:
        DispatchError: On connection errors or timeout.
    """
    runs_url = url.rstrip("/") + "/dashboard/runs"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(runs_url, headers=headers)
    except httpx.TimeoutException as exc:
        raise DispatchError(f"Timeout fetching runs from {runs_url}: {exc}") from exc
    except httpx.ConnectError as exc:
        raise DispatchError(f"Connection error fetching runs from {runs_url}: {exc}") from exc
    except httpx.RequestError as exc:
        raise DispatchError(f"Request error fetching runs from {runs_url}: {exc}") from exc

    response.raise_for_status()
    return cast(list[dict[str, Any]], response.json())


async def cancel_worker_run(
    url: str, token: str, run_id: str, timeout: float = 10.0
) -> dict[str, Any]:
    """DELETE /dispatch/{run_id} on the worker to request cancellation.

    Args:
        url: Base URL of the worker service (e.g. "http://dispatch-worker-1:9190").
        token: Bearer token for authentication.
        run_id: The run ID to cancel.
        timeout: Request timeout in seconds.

    Returns:
        Worker's response dict (typically ``{"cancelled": bool, "run_id": str}``).

    Raises:
        AuthError: If the worker returns HTTP 401.
        DispatchError: On connection errors, timeouts, or non-2xx responses.
    """
    cancel_url = url.rstrip("/") + f"/dispatch/{run_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.delete(cancel_url, headers=headers)
    except httpx.TimeoutException as exc:
        raise DispatchError(f"Timeout cancelling at {cancel_url}: {exc}") from exc
    except httpx.ConnectError as exc:
        raise DispatchError(f"Connection error cancelling at {cancel_url}: {exc}") from exc
    except httpx.RequestError as exc:
        raise DispatchError(f"Request error cancelling at {cancel_url}: {exc}") from exc

    if response.status_code == 401:
        raise AuthError(f"Unauthorized (401) from {cancel_url}")
    if response.status_code == 404:
        raise DispatchError(f"run_id {run_id!r} not found on worker")
    response.raise_for_status()
    return cast(dict[str, Any], response.json())


async def proxy_worker_stream(url: str, token: str, run_id: str) -> AsyncIterator[bytes]:
    """Proxy an SSE stream from a worker's /dispatch/{run_id}/stream endpoint.

    Yields raw byte chunks from the upstream SSE stream. The browser's
    EventSource handles reassembly from arbitrary chunk boundaries.

    Args:
        url: Base URL of the worker service.
        token: Bearer token for authentication.
        run_id: The run ID to stream.

    Yields:
        Raw byte chunks from the SSE stream.

    Raises:
        AuthError: If the worker returns HTTP 401.
        DispatchError: On connection errors or request errors.
    """
    stream_url = url.rstrip("/") + f"/dispatch/{run_id}/stream"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", stream_url, headers=headers) as response:
                if response.status_code == 401:
                    raise AuthError(f"Unauthorized (401) from {stream_url}")
                if response.status_code != 200:
                    raise DispatchError(f"HTTP {response.status_code} from {stream_url}")
                async for chunk in response.aiter_bytes():
                    yield chunk
    except httpx.ConnectError as exc:
        raise DispatchError(f"Connection error streaming from {stream_url}: {exc}") from exc
    except httpx.RequestError as exc:
        raise DispatchError(f"Request error streaming from {stream_url}: {exc}") from exc
