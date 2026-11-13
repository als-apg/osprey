"""MCP tool: directory_properties — the values one directory property holds."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.pv_directory.server import get_tool_context, mcp, refuse_unavailable
from osprey.services.pv_directory._connectors import as_payload
from osprey.services.pv_directory.vocabulary import service_failed

logger = logging.getLogger("osprey.mcp_server.pv_directory.tools.properties")

_TOOL = "directory_properties"
#: The directory deadline when ``pv_directory.settings.timeout_s`` sets none (D10).
_DEFAULT_TIMEOUT_S = 6.0


@mcp.tool(
    title="List a directory property's values",
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def directory_properties(property: str, prefix: str | None = None) -> str:
    """List the values one property of the control system's channel directory holds.

    Directory properties are the control system's own records about each
    channel address (its IOC, its host, a facility's own tags); the values are
    spelled as the directory spells them. Pass a value verbatim in
    directory_search ``properties`` to list the channels carrying it. A value
    that matches a word does not say what the channels carrying it measure.

    Args:
        property: The property name, as the directory spells it (for example
            ``iocName`` or ``recordType``).
        prefix: Only the values starting with this text.

    Returns:
        JSON object: ``property``, ``prefix``, ``source``, ``values`` and ``count``.
    """
    try:
        backend, source, reason = await get_tool_context().service.backend("pv_directory")
        if backend is None:
            refuse_unavailable(_TOOL, {"outcome": "unavailable", "reason": reason})
        deadline = _timeout(backend.spec.settings)
        try:
            answer = await asyncio.wait_for(
                backend.property_values(property, prefix, deadline), deadline
            )
        except TimeoutError:
            answer = {"reason": f"no answer within {deadline:g} s"}
        except Exception as failure:
            logger.warning("directory backend failed", exc_info=True)
            answer = {"reason": service_failed(failure)}
        payload = as_payload(answer)
        if isinstance(payload, Mapping):
            refuse_unavailable(
                _TOOL,
                {"outcome": "unavailable", "source": as_payload(source), **payload},
            )
        values = list(payload)
        result: dict[str, Any] = {
            "property": property,
            "prefix": prefix,
            "source": as_payload(source),
            "values": values,
            "count": len(values),
        }
        return json.dumps(result)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("%s failed", _TOOL)
        make_error(
            "internal_error",
            f"{_TOOL} failed: {exc}",
            ["Check the MCP server logs for details."],
        )


def _timeout(settings: Mapping[str, Any]) -> float:
    """The section's ``settings.timeout_s``, which every backend spec carries, else 6 s."""
    value = settings.get("timeout_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return _DEFAULT_TIMEOUT_S
    return float(value)
