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


#: Backend whose records are addressed by an organization and a stream. Any
#: other backend is addressed some other way, and OpenObserve's coordinates
#: would be meaningless in it.
_OPENOBSERVE_BACKEND = "openobserve"

#: The stream OSPREY's OTLP records land in. One value, spelled once — it is
#: not derived from anything yet, and inventing a key for it would be a knob
#: nothing turns.
_DEFAULT_STREAM = "default"


def _store_coordinates() -> dict[str, str]:
    """Store-specific coordinates for the configured telemetry backend.

    Read from ``claude_code.telemetry`` rather than re-parsed out of the OTLP
    endpoint URL: the org is a configured value, and deriving it a second time
    from a URL the exporter built out of it is a producer that can disagree with
    the store it points at. The org itself comes from
    :func:`osprey.deployment.openobserve_provision.store_org`, the one resolver
    the provisioner and the exporter already share — a hand-rolled read here
    would be a third producer, and one that gets the empty-org case wrong: the
    default belongs to an ABSENT key, and an explicit ``org: ""`` is a value.

    Only ``openobserve`` is addressed by an org and a stream. Every other
    backend gets neither key at all — not a null, and not a placeholder: a
    consumer templating these into a filed issue must not be able to render an
    empty coordinate that reads as a real one.

    Returns:
        ``{"org": …, "stream": …}`` for the OpenObserve backend, else ``{}``.
    """
    try:
        from osprey.deployment.openobserve_provision import store_org
        from osprey.utils.workspace import load_osprey_config

        config = load_osprey_config()
        telemetry = ((config.get("claude_code") or {}).get("telemetry")) or {}
        if str(telemetry.get("backend") or "").strip().lower() != _OPENOBSERVE_BACKEND:
            return {}
        return {"org": store_org(config), "stream": _DEFAULT_STREAM}
    except Exception:
        return {}


@mcp.tool()
async def provenance_locator() -> str:
    """Return telemetry coordinates locating this session in the OTEL store.

    A consumer renders these into a filed issue so a maintainer can pull the
    session's full provenance from telemetry (the harness-agnostic source of
    truth) instead of trusting a reconstructed narration. No parameters.

    Returns:
        JSON ``{session_id, service_name, since, emitted_at}``, plus ``org`` and
        ``stream`` when the configured telemetry backend is addressed by them
        (:func:`_store_coordinates`). ``session_id`` is ``null`` (with a
        ``note``) when no id resolves or telemetry is unavailable/degraded for
        this run. Never raises.
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
                **_store_coordinates(),
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
