"""MCP tool: provenance_locator.

Returns generic telemetry *coordinates* for the current agent session — a
pointer a consumer can render into a filed issue so a maintainer can retrieve
the turn's full provenance from the OTEL store (tool calls, subagents,
model/tokens/cost, user prompts). The telemetry store is the harness-agnostic
source of provenance truth; this tool is the seam that hands out the locator so
consumers never scrape harness-specific environment themselves.

Design (harness-agnostic seam):
    * ``session_id`` is OSPREY-owned. OSPREY forces a known session UUID at
      launch and injects it as ``OSPREY_TELEMETRY_SESSION_ID`` into the MCP
      subprocess env; this tool reads that. It falls back to the harness's own
      ``CLAUDE_CODE_SESSION_ID`` when OSPREY did not force one (e.g. an
      interactive surface that exports it to child processes).
    * The id returned is exactly the value the OTEL emitter tags records with as
      ``session.id`` — forcing guarantees they match, so the locator resolves.

Contract:
    * Returns JSON ``{session_id, service_name, org, stream, since, emitted_at}``.
    * ``emitted_at`` is stamped server-side at call time (never agent-guessed).
    * Never raises. When no id resolves, or telemetry is disabled/degraded for
      this run (so any id would resolve to nothing), returns ``session_id: null``
      with a ``note`` — honest "unavailable" rather than a dangling pointer.
    * Facility-agnostic: no facility strings. The "how to pull it" recipe is
      rendered by the consumer (e.g. als-profiles), not here.
"""

import json
import logging
import os
from datetime import UTC, datetime

from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.provenance_locator")

# Env var OSPREY injects at launch carrying the forced session UUID. Kept in
# sync with the injection sites (dispatch_worker.sdk_runner, web_terminal
# operator_session / PTY launch), which set the same name. A dedicated var —
# NOT OSPREY_SESSION_ID, which has unrelated side effects (it relocates
# session-scoped agent data and tags saved artifacts).
OSPREY_TELEMETRY_SESSION_ID_ENV = "OSPREY_TELEMETRY_SESSION_ID"
# Optional ISO-8601 session-start OSPREY may inject to bound the lookback query.
OSPREY_TELEMETRY_SESSION_START_ENV = "OSPREY_TELEMETRY_SESSION_START"


def _telemetry_enabled() -> bool:
    """Whether Claude Code OTEL export is actually on for this run.

    If telemetry is off (or has no exporter endpoint), records for this session
    never reach the store, so any ``session_id`` we hand out would resolve to
    nothing. Gate on the same env the emitter itself consumes.
    """
    if os.environ.get("CLAUDE_CODE_ENABLE_TELEMETRY") != "1":
        return False
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def _resolve_session_id() -> str | None:
    """The OSPREY-forced id, else the harness's own, else None."""
    return (
        os.environ.get(OSPREY_TELEMETRY_SESSION_ID_ENV)
        or os.environ.get("CLAUDE_CODE_SESSION_ID")
        or None
    )


def _service_name() -> str:
    """OTEL ``service.name`` from resource attributes, else the CC default."""
    attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    for pair in attrs.split(","):
        key, _sep, val = pair.partition("=")
        if key.strip() == "service.name" and val.strip():
            return val.strip()
    return "claude-code"


def _org() -> str:
    """OpenObserve org, parsed from the OTLP endpoint ``.../api/<org>``."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    marker = "/api/"
    if marker in endpoint:
        tail = endpoint.split(marker, 1)[1].strip("/")
        if tail:
            return tail.split("/", 1)[0]
    return "default"


@mcp.tool()
async def provenance_locator() -> str:
    """Return telemetry coordinates locating this session in the OTEL store.

    A consumer renders these into a filed issue so a maintainer can pull the
    session's full provenance from telemetry (the harness-agnostic source of
    truth) instead of trusting a reconstructed narration. No parameters.

    Returns:
        JSON ``{session_id, service_name, org, stream, since, emitted_at}``.
        ``session_id`` is ``null`` (with a ``note``) when no id resolves or
        telemetry is unavailable/degraded for this run. Never raises.
    """
    emitted_at = datetime.now(UTC).isoformat()
    try:
        session_id = _resolve_session_id()
        if not _telemetry_enabled():
            return json.dumps(
                {
                    "session_id": None,
                    "emitted_at": emitted_at,
                    "note": "telemetry unavailable for this run — no provenance locator",
                },
                indent=2,
            )
        if not session_id:
            return json.dumps(
                {
                    "session_id": None,
                    "emitted_at": emitted_at,
                    "note": "session id could not be resolved — no provenance locator",
                },
                indent=2,
            )
        return json.dumps(
            {
                "session_id": session_id,
                "service_name": _service_name(),
                "org": _org(),
                "stream": "default",
                "since": os.environ.get(OSPREY_TELEMETRY_SESSION_START_ENV) or None,
                "emitted_at": emitted_at,
            },
            indent=2,
        )
    except Exception as e:  # never raise: filing must not be blocked
        logger.warning("provenance_locator degraded: %s", e, exc_info=True)
        return json.dumps(
            {
                "session_id": None,
                "emitted_at": emitted_at,
                "note": f"provenance locator unavailable: {e}",
            },
            indent=2,
        )
