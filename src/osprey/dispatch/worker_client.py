"""Async HTTP client for dispatching prompts to dispatch worker services."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import httpx


class WorkerRequestError(Exception):
    """Base class for every failure this client reports on a call to a worker.

    The one decision every caller makes about these is whether to try the same
    request again, so that decision is *data* on the base rather than something
    to infer from a subclass name: ``retryable`` is ``True`` when an identical
    redispatch could plausibly succeed (the worker was unreachable, or answered
    with something transient) and ``False`` when the worker answered and refused
    *this* request deterministically. ``reason`` is the stable, machine-readable
    code; the message is the human half. ``error_code`` is the machine-readable
    code lifted from the worker's own body when it supplied a recognised one
    (see :data:`KNOWN_REJECTION_CODES`), ``None`` otherwise.

    Because every failure is a :class:`WorkerRequestError`, ``except
    WorkerRequestError`` is complete — no failure mode escapes it, including a
    rejected bearer token.
    """

    reason = "worker_request_failed"
    retryable = True

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class WorkerUnreachableError(WorkerRequestError):
    """The worker could not be reached at all (connection error or timeout)."""

    reason = "worker_unreachable"
    retryable = True


class WorkerUnavailableError(WorkerRequestError):
    """The worker was reached but answered 5xx — transient, so worth retrying."""

    reason = "worker_unavailable"
    retryable = True


class WorkerAuthRejectedError(WorkerRequestError):
    """The worker rejected the dispatcher's bearer token with HTTP 401.

    Retryable only in the sense that the on-error policy owns it: a rotated
    token can make the identical request succeed, so this is not a deterministic
    rejection of the request's *content*.
    """

    reason = "worker_auth_rejected"
    retryable = True


class WorkerRejectedRequestError(WorkerRequestError):
    """The worker answered and refused this request deterministically (non-401 4xx).

    A 4xx means the request itself is malformed/oversized (e.g. an input-files
    batch over cap, or a body past the size limit) or names something that does
    not exist: an identical redispatch is rejected identically, so this is NOT
    retryable. ``error_code`` carries the machine-readable ``detail``
    whitelisted from the worker's 4xx JSON body (one of
    :data:`KNOWN_REJECTION_CODES`), or ``None`` when the body carried no
    recognised code. The caller surfaces this as a pool error carrying
    ``error_code`` rather than routing it through the retry/drop policy.
    """

    reason = "worker_rejected_request"
    retryable = False


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
        url: Base URL of the dispatch worker service (e.g.
            ``http://dispatch-worker-1:10011`` — worker 1's slot at the default
            port base).
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
        WorkerAuthRejectedError: If the server returns HTTP 401.
        WorkerRejectedRequestError: On a non-401 4xx (deterministic, non-retryable
            rejection), carrying a whitelisted ``error_code`` when the body supplied one.
        WorkerUnreachableError: On connection errors or timeout.
        WorkerUnavailableError: On a retryable 5xx.
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
        raise WorkerUnreachableError(f"Timeout dispatching to {dispatch_url}: {exc}") from exc
    except httpx.ConnectError as exc:
        raise WorkerUnreachableError(
            f"Connection error dispatching to {dispatch_url}: {exc}"
        ) from exc
    except httpx.RequestError as exc:
        raise WorkerUnreachableError(f"Request error dispatching to {dispatch_url}: {exc}") from exc

    if response.status_code == 401:
        raise WorkerAuthRejectedError(f"Unauthorized (401) from {dispatch_url}")

    # A non-401 4xx is a deterministic rejection of THIS request — never retry it,
    # and never drop it to a silent None. Surface a typed WorkerRejectedRequestError
    # carrying a whitelisted machine-readable code (or None) so the caller records
    # it as a non-retryable pool error. 413 (body too large) is included here.
    if 400 <= response.status_code < 500:
        raise WorkerRejectedRequestError(
            f"HTTP {response.status_code} from worker",
            error_code=_extract_rejection_code(response),
        )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Surface only the status code — never echo the worker's response body
        # into the dispatcher's registry history; it can carry a stack trace or
        # other internal detail. Raise a typed WorkerUnavailableError so the
        # on_error policy treats it like any other retryable dispatch failure.
        raise WorkerUnavailableError(f"HTTP {response.status_code} from worker") from exc
    return cast(dict[str, Any], response.json())


async def fetch_worker_runs(url: str, token: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Fetch recent runs from a worker's /dashboard/runs endpoint.

    The /dashboard/runs endpoint is bearer-gated like the other worker endpoints,
    so the dispatcher's worker token is forwarded.

    Args:
        url: Base URL of the worker service (e.g.
            ``http://dispatch-worker-1:10011`` — worker 1's slot at the default
            port base).
        token: Bearer token for the worker (DISPATCH_WORKER_TOKEN).
        timeout: Request timeout in seconds.

    Returns:
        List of run dicts (run_id, status, created_at, etc.).

    Raises:
        WorkerUnreachableError: On connection errors or timeout.
    """
    runs_url = url.rstrip("/") + "/dashboard/runs"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(runs_url, headers=headers)
    except httpx.TimeoutException as exc:
        raise WorkerUnreachableError(f"Timeout fetching runs from {runs_url}: {exc}") from exc
    except httpx.ConnectError as exc:
        raise WorkerUnreachableError(
            f"Connection error fetching runs from {runs_url}: {exc}"
        ) from exc
    except httpx.RequestError as exc:
        raise WorkerUnreachableError(f"Request error fetching runs from {runs_url}: {exc}") from exc

    response.raise_for_status()
    return cast(list[dict[str, Any]], response.json())


async def cancel_worker_run(
    url: str, token: str, run_id: str, timeout: float = 10.0
) -> dict[str, Any]:
    """DELETE /dispatch/{run_id} on the worker to request cancellation.

    Args:
        url: Base URL of the worker service (e.g.
            ``http://dispatch-worker-1:10011`` — worker 1's slot at the default
            port base).
        token: Bearer token for authentication.
        run_id: The run ID to cancel.
        timeout: Request timeout in seconds.

    Returns:
        Worker's response dict (typically ``{"cancelled": bool, "run_id": str}``).

    Raises:
        WorkerAuthRejectedError: If the worker returns HTTP 401.
        WorkerRejectedRequestError: If the worker does not know ``run_id`` (404).
        WorkerUnreachableError: On connection errors or timeouts.
    """
    cancel_url = url.rstrip("/") + f"/dispatch/{run_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.delete(cancel_url, headers=headers)
    except httpx.TimeoutException as exc:
        raise WorkerUnreachableError(f"Timeout cancelling at {cancel_url}: {exc}") from exc
    except httpx.ConnectError as exc:
        raise WorkerUnreachableError(f"Connection error cancelling at {cancel_url}: {exc}") from exc
    except httpx.RequestError as exc:
        raise WorkerUnreachableError(f"Request error cancelling at {cancel_url}: {exc}") from exc

    if response.status_code == 401:
        raise WorkerAuthRejectedError(f"Unauthorized (401) from {cancel_url}")
    if response.status_code == 404:
        raise WorkerRejectedRequestError(f"run_id {run_id!r} not found on worker")
    response.raise_for_status()
    return cast(dict[str, Any], response.json())


async def clear_worker_history(
    url: str, token: str, older_than_days: int = 0, timeout: float = 30.0
) -> dict[str, Any]:
    """DELETE /dispatch/runs on the worker to drop finished run records.

    Args:
        url: Base URL of the worker service (e.g.
            ``http://dispatch-worker-1:10011`` — worker 1's slot at the default
            port base).
        token: Bearer token for authentication.
        older_than_days: Age floor in days. ``0`` (the default) clears every
            finished run; a positive value clears only those older than it,
            which is the retention sweep's horizon applied on demand.
        timeout: Request timeout in seconds. Longer than the cancel path's: this
            unlinks up to a full history's worth of records in one call.

    Returns:
        Worker's response dict (``{"cleared": int, "records_deleted": int,
        "older_than_days": int}``).

    Raises:
        WorkerAuthRejectedError: If the worker returns HTTP 401.
        WorkerUnreachableError: On connection errors or timeouts.
    """
    clear_url = url.rstrip("/") + "/dispatch/runs"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"older_than_days": older_than_days}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # ``client.delete`` takes no body, so go through ``request``: the age
            # floor rides in the body rather than the URL, keeping it out of
            # access logs alongside the rest of the worker's write surface.
            response = await client.request("DELETE", clear_url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise WorkerUnreachableError(f"Timeout clearing history at {clear_url}: {exc}") from exc
    except httpx.ConnectError as exc:
        raise WorkerUnreachableError(
            f"Connection error clearing history at {clear_url}: {exc}"
        ) from exc
    except httpx.RequestError as exc:
        raise WorkerUnreachableError(
            f"Request error clearing history at {clear_url}: {exc}"
        ) from exc

    if response.status_code == 401:
        raise WorkerAuthRejectedError(f"Unauthorized (401) from {clear_url}")
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
        WorkerAuthRejectedError: If the worker returns HTTP 401.
        WorkerUnavailableError: If the worker answers with a non-200 status.
        WorkerUnreachableError: On connection errors or request errors.
    """
    stream_url = url.rstrip("/") + f"/dispatch/{run_id}/stream"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", stream_url, headers=headers) as response:
                if response.status_code == 401:
                    raise WorkerAuthRejectedError(f"Unauthorized (401) from {stream_url}")
                if response.status_code != 200:
                    raise WorkerUnavailableError(f"HTTP {response.status_code} from {stream_url}")
                async for chunk in response.aiter_bytes():
                    yield chunk
    except httpx.ConnectError as exc:
        raise WorkerUnreachableError(
            f"Connection error streaming from {stream_url}: {exc}"
        ) from exc
    except httpx.RequestError as exc:
        raise WorkerUnreachableError(f"Request error streaming from {stream_url}: {exc}") from exc
