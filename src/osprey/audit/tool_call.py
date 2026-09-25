"""The opt-in full record of every osprey tool call.

The default audit surfaces hold identifiers only (:mod:`osprey.audit.envelope`).
This surface is the one place values are recorded: when
``audit.tool_call.enabled`` is on, the MCP audit middleware files one record per
tool call — reads included — with the full arguments, the full result, the
control target it ran against, the approval answer that let it through, and the
conversation and tool-use ids that join it to the default records and to the
agent harness's own telemetry. It lands in
``var/audit/<identity>/tool_call.jsonl`` and is sent to the telemetry store as
one log line (:mod:`osprey.audit.otlp`).

It holds values by design and does not use the envelope. A payload over
``audit.tool_call.max_inline_bytes`` is not dropped: its bytes are saved as an
artifact and the record keeps its size, sha256 and artifact id. The default
surfaces are unchanged whether this one is on or off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_ORIGIN",
    "DEFAULT_MAX_INLINE_BYTES",
    "ENABLED_KEY",
    "MAX_INLINE_KEY",
    "SURFACE_TOOL_CALL",
    "build_record",
    "capped",
    "serialize_result",
    "settings",
]

#: The ledger stem this surface files under.
SURFACE_TOOL_CALL = "tool_call"

#: Turns the surface on. Off by default.
ENABLED_KEY = "audit.tool_call.enabled"

#: The largest payload (arguments or result, each on its own) kept inline.
MAX_INLINE_KEY = "audit.tool_call.max_inline_bytes"

#: Default for :data:`MAX_INLINE_KEY`: 256 KiB.
DEFAULT_MAX_INLINE_BYTES = 262144

#: The ``origin`` a payload artifact is saved with, so a listing can tell it
#: apart from what a tool produced.
ARTIFACT_ORIGIN = "tool_call"

#: The rendered config this process was launched with.
_CONFIG_ENV = "OSPREY_CONFIG"

_warned_max = False


def settings() -> tuple[bool, int]:
    """``(enabled, max_inline_bytes)`` for this process.

    Read from the config named by ``OSPREY_CONFIG`` — the same anchor the audit
    middleware reads its hook config from. Unset or relative ``OSPREY_CONFIG``,
    or any error, is ``(False, default)``. The config reader caches per path
    for the life of the process, so turning the surface on or off takes a
    server restart. A max that is not a positive integer falls back to the
    default with one warning.
    """
    global _warned_max
    configured = (os.environ.get(_CONFIG_ENV) or "").strip()
    if not configured or not os.path.isabs(configured):
        return False, DEFAULT_MAX_INLINE_BYTES
    try:
        from osprey_connectors.config import get_config_value

        enabled = get_config_value(ENABLED_KEY, False, config_path=configured) is True
        raw_max = get_config_value(MAX_INLINE_KEY, DEFAULT_MAX_INLINE_BYTES, config_path=configured)
    except Exception:
        logger.debug("Could not read the tool_call settings", exc_info=True)
        return False, DEFAULT_MAX_INLINE_BYTES
    if isinstance(raw_max, bool) or not isinstance(raw_max, int) or raw_max <= 0:
        if not _warned_max:
            _warned_max = True
            logger.warning(
                "%s is %r, not a positive integer; using %d",
                MAX_INLINE_KEY,
                raw_max,
                DEFAULT_MAX_INLINE_BYTES,
            )
        return enabled, DEFAULT_MAX_INLINE_BYTES
    return enabled, raw_max


def _encode(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), default=str).encode("utf-8", "replace")


def capped(
    value: Any,
    *,
    label: str,
    subject: str,
    tool_use_id: str | None,
    max_inline: int,
) -> tuple[Any, dict[str, Any] | None]:
    """*value* inline, or a reference to its saved bytes when it is too large.

    Returns ``(value, None)`` when its compact JSON is at most *max_inline*
    bytes, else ``(None, {"size", "sha256", "artifact_id"})`` with the bytes
    saved as a JSON artifact. A failed save keeps the size and hash, sets
    ``artifact_id`` to ``None`` and names the error type in ``artifact_error``.
    """
    encoded = _encode(value)
    if len(encoded) <= max_inline:
        return value, None
    digest = hashlib.sha256(encoded).hexdigest()
    reference: dict[str, Any] = {"size": len(encoded), "sha256": digest, "artifact_id": None}
    try:
        from osprey.stores.artifact_store import get_artifact_store

        entry = get_artifact_store().save_file(
            file_content=encoded,
            filename=f"{tool_use_id or uuid.uuid4().hex}-{label}.json",
            artifact_type="json",
            title=f"{subject} {label}",
            mime_type="application/json",
            tool_source="audit.tool_call",
            origin=ARTIFACT_ORIGIN,
            metadata={"tool_use_id": tool_use_id, "sha256": digest},
        )
        reference["artifact_id"] = entry.id
    except Exception as exc:
        logger.debug("Could not save the %s payload as an artifact", label, exc_info=True)
        reference["artifact_error"] = type(exc).__name__
    return None, reference


def serialize_result(result: Any) -> Any:
    """A tool result as JSON-ready data.

    A FastMCP ``ToolResult`` becomes ``{"content": [...], "structured_content":
    ...}`` with every content block dumped as its wire form; anything else is
    round-tripped through JSON, with ``str`` for what JSON cannot hold.
    """
    content = getattr(result, "content", None)
    if isinstance(content, list) and hasattr(result, "structured_content"):
        blocks = []
        for block in content:
            dump = getattr(block, "model_dump", None)
            blocks.append(
                dump(mode="json", by_alias=True, exclude_none=True) if callable(dump) else block
            )
        return json.loads(
            json.dumps(
                {"content": blocks, "structured_content": result.structured_content},
                default=str,
            )
        )
    return json.loads(json.dumps(result, default=str))


def build_record(
    *,
    ts: str,
    actor: str,
    posture: str,
    posture_source: str,
    session: str | None,
    session_id: str | None,
    tool_use_id: str | None,
    server: str,
    subject: str,
    decision: str,
    reason: str,
    approval: dict[str, Any] | None,
    target: str | None,
    generation: int | None,
    arguments: Any,
    arguments_ref: dict[str, Any] | None,
    result: Any,
    result_ref: dict[str, Any] | None,
    error: str | None,
    is_error: bool,
    facts: dict[str, Any],
    duration_ms: float,
) -> dict[str, Any]:
    """One full record, in its documented key order.

    A payload that was stored as an artifact appears as ``arguments_ref`` /
    ``result_ref`` in place of ``arguments`` / ``result``.
    """
    record: dict[str, Any] = {
        "ts": ts,
        "surface": SURFACE_TOOL_CALL,
        "actor": actor,
        "posture": posture,
        "posture_source": posture_source,
        "session": session,
        "session_id": session_id,
        "tool_use_id": tool_use_id,
        "server": server,
        "subject": subject,
        "decision": decision,
        "reason": reason,
        "approval": approval,
        "target": target,
        "generation": generation,
    }
    if arguments_ref is not None:
        record["arguments_ref"] = arguments_ref
    else:
        record["arguments"] = arguments
    if result_ref is not None:
        record["result_ref"] = result_ref
    else:
        record["result"] = result
    record["error"] = error
    record["is_error"] = is_error
    record["facts"] = facts
    record["duration_ms"] = duration_ms
    return record
