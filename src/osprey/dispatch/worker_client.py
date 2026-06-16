"""Async HTTP client for dispatching prompts to dispatch worker services."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import httpx


class DispatchError(Exception):
    """Raised on connection errors or timeouts when dispatching to a worker."""


class AuthError(Exception):
    """Raised when the worker returns HTTP 401 Unauthorized."""


async def dispatch_to_worker(
    url: str,
    prompt: str,
    allowed_tools: list[str],
    token: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST a prompt to a dispatch worker's /dispatch endpoint.

    Args:
        url: Base URL of the dispatch worker service (e.g. "http://dispatch-worker-1:9190").
        prompt: The prompt text to dispatch.
        allowed_tools: List of tool names the agent is allowed to use.
        token: Bearer token for authentication.
        timeout: Request timeout in seconds.

    Returns:
        Response JSON dict (typically contains ``run_id`` and ``status``).

    Raises:
        AuthError: If the server returns HTTP 401.
        DispatchError: On connection errors or timeout.
    """
    dispatch_url = url.rstrip("/") + "/dispatch"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"prompt": prompt, "allowed_tools": allowed_tools}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(dispatch_url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise DispatchError(f"Timeout dispatching to {dispatch_url}: {exc}") from exc
    except httpx.ConnectError as exc:
        raise DispatchError(f"Connection error dispatching to {dispatch_url}: {exc}") from exc
    except httpx.RequestError as exc:
        raise DispatchError(f"Request error dispatching to {dispatch_url}: {exc}") from exc

    if response.status_code == 401:
        raise AuthError(f"Unauthorized (401) from {dispatch_url}")

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Surface only the status code — never echo the worker's response body
        # into the dispatcher's registry history; it can carry a stack trace or
        # other internal detail. Raise a typed DispatchError so the on_error
        # policy treats it like any other dispatch failure.
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
