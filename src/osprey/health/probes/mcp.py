"""MCP health probe — streamable-HTTP handshake and tool-list assertion.

Opens a streamable-HTTP session to a configured MCP endpoint via the ``mcp``
SDK, performs the protocol handshake (``initialize`` → ``notifications/
initialized``) and lists the server's tools, mirroring the client pattern in
``src/osprey/mcp_server/introspect.py``.

Grading:

* a completed handshake with the tool expectation satisfied is ``ok``;
* when ``expect_tools`` is configured, any named tool missing from the server —
  or a server exposing no tools at all — is an ``error`` whose ``value`` reports
  ``"{have}/{expect} tools"``;
* with no ``expect_tools``, a server that returns zero tools is an ``error`` (a
  healthy MCP server exposes at least one tool);
* the handshake timeout (``timeout_s``) maps to the configured ``timeout_status``
  (``error`` by default, ``warning`` opt-in); any other connection or protocol
  failure is an ``error`` carrying ``str(exc)`` in ``details``.

``latency_ms`` is measured from a single ``perf_counter()`` and read again in
every branch, so the wire result always reports how long the attempt took.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.probes import ProbeContext


async def run(spec: Mapping[str, Any], ctx: ProbeContext) -> CheckResult:
    """Probe an MCP endpoint by handshaking and listing its tools.

    Args:
        spec: Parsed check parameters. Recognized keys:

            * ``url`` (str, required): the streamable-HTTP MCP endpoint.
            * ``name`` (str): result name (default ``"mcp"``).
            * ``category`` (str): result category (default ``"mcp"``).
            * ``expect_tools`` (Sequence[str] | None): tool names that must be
              present; when omitted, only a non-empty tool list is required.
            * ``timeout_s`` (float): handshake timeout in seconds (default ``10``).
            * ``timeout_status`` (``"error"`` | ``"warning"``): status for a
              handshake timeout (default ``"error"``).
        ctx: Shared per-run context. Unused by this probe (MCP needs no
            control-system connector) but part of the uniform probe interface.

    Returns:
        A :class:`~osprey.health.models.CheckResult` with ``latency_ms`` set on
        every outcome and, when tools were listed, ``value`` reporting the tool
        count.
    """
    name = str(spec.get("name", "mcp"))
    category = str(spec.get("category", "mcp"))
    url = str(spec.get("url", ""))
    timeout_s = float(spec.get("timeout_s", 10.0))
    timeout_status = Status(spec.get("timeout_status", "error"))
    expect_raw = spec.get("expect_tools")
    expect_tools: list[str] = (
        [str(t) for t in expect_raw]
        if isinstance(expect_raw, Sequence) and not isinstance(expect_raw, str | bytes)
        else []
    )

    t0 = perf_counter()
    try:
        # asyncio.timeout bounds the whole handshake; nested context managers are
        # cancelled and cleaned up on expiry.
        async with asyncio.timeout(timeout_s):
            async with streamablehttp_client(url) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
    except TimeoutError:
        latency_ms = (perf_counter() - t0) * 1000.0
        return CheckResult(
            name,
            category,
            timeout_status,
            f"MCP handshake to {url} timed out after {timeout_s:g}s",
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 - any connection/protocol failure is an error
        latency_ms = (perf_counter() - t0) * 1000.0
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"MCP handshake to {url} failed",
            latency_ms=latency_ms,
            details=str(exc),
        )

    latency_ms = (perf_counter() - t0) * 1000.0
    tool_names = [tool.name for tool in result.tools]
    have = len(tool_names)

    if expect_tools:
        expect = len(expect_tools)
        missing = [t for t in expect_tools if t not in tool_names]
        value = f"{have}/{expect} tools"
        if missing:
            return CheckResult(
                name,
                category,
                Status.ERROR,
                f"MCP {url}: missing expected tool(s): {', '.join(missing)}",
                value=value,
                latency_ms=latency_ms,
                details=f"available: {', '.join(tool_names) or '(none)'}",
            )
        return CheckResult(
            name,
            category,
            Status.OK,
            f"MCP {url} → {have} tools, all {expect} expected present in {latency_ms:.0f}ms",
            value=value,
            latency_ms=latency_ms,
        )

    if have == 0:
        return CheckResult(
            name,
            category,
            Status.ERROR,
            f"MCP {url} handshake ok but server exposed no tools",
            value="0 tools",
            latency_ms=latency_ms,
        )

    return CheckResult(
        name,
        category,
        Status.OK,
        f"MCP {url} → {have} tools in {latency_ms:.0f}ms",
        value=f"{have} tools",
        latency_ms=latency_ms,
    )
