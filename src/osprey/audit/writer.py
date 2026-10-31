"""The audit writer — every safety decision becomes one ``audit`` entry in the record.

:mod:`osprey.audit.envelope` says what a record *is*; this module hands it to
the agent record. Every safety surface — the MCP middleware, the HTTP layers,
the framework writers — passes an envelope to :func:`record_envelope` (or the
fields to :func:`record`) and gets back the entry's id, or ``None``.

**One entry per decision, through the record writer.** A record is written with
:func:`osprey.agent_record.writer.write` as one :data:`~osprey.agent_record.contract.KIND_AUDIT`
line in the acting identity's spool. Its body is the envelope's
:meth:`~osprey.audit.envelope.AuditEnvelope.to_dict` exactly, so the surface is
in the body's ``surface`` field (:data:`~osprey.agent_record.contract.SURFACE_FIELD`)
and every other field keeps the name and order it has always had. The
envelope's ``ts``, ``tool_use_id`` and ``on_behalf_of`` also travel as the
line's ids, the conversation (the tool call in scope's, else the process's) as
its ``session_id``, and the process's terminal session as its
``terminal_session``, so the record can join an audit entry to the tool call
and the session it belongs to without opening the body.

**Identity is the process's, never the envelope's.** The spool directory is the
acting identity's own (:func:`~osprey.agent_record.writer.spool_dir`, on the
shared ladder ``OSPREY_TERMINAL_USER`` → ``OSPREY_AUDIT_IDENTITY`` → local
account → ``unknown``), and the record takes the identity from that directory,
never from the line. :func:`record` fills the envelope's ``actor`` from the same
ladder, so the two agree by default. An envelope whose ``actor`` differs from
its directory is the normal shape for a service acting for someone; the record
keeps ``on_behalf_of`` only for the identities its trust list names.

Segments are per process, so two writers in one container — the root
maintenance phase and the dropped app user — never share a file, and nothing
here routes on which of them is writing.

**A bounded body.** An envelope is identifiers plus two documented free-text
fields, and :data:`MAX_RECORD_BYTES` (2 KB) bounds the encoded body so an
ordinary decision costs the record one small entry. The exceptions, in order:

* An oversize ``detail`` is replaced by :data:`DETAIL_DROPPED`. ``detail`` is
  documented as supplementary context, so it is the field that gives way.
* On :data:`~osprey.audit.envelope.SURFACE_EXECUTOR` the ``source`` is kept
  whole (up to the envelope's 8000-char bound). There the refused code *is* the
  artifact under audit, and a record without it is an alert rather than an
  audit trail.
* An identifier-only body that is still over budget is written whole and
  logged. Trimming identifiers would leave a record that names the wrong
  thing, which is worse than a large one.

The ``tool_call`` record (:func:`append_record`) is not an envelope and is
not bounded: it carries values by design.

**Nothing here may cost the operation it records.** Every entry point swallows
its errors and returns ``None``; the event is logged *before* the write, so a
spool that cannot be written downgrades the audit trail instead of erasing it.
An invalid envelope is included in that: construction validation lives in the
envelope, and :func:`record` calls it from inside this boundary so an emitter
never needs a ``try`` of its own. The record writer reports its own failures
to the spool's health file, and with the record off switch set it stores
nothing and answers ``None``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from osprey.agent_record import writer as record_writer
from osprey.agent_record.contract import KIND_AUDIT, SURFACE_FIELD, WriteResult
from osprey.audit.call import current_call, current_tool_use_id, harness_session_id
from osprey.audit.envelope import (
    DECISION_REFUSED,
    SURFACE_EXECUTOR,
    AuditEnvelope,
    terminal_session,
)
from osprey_connectors.identity import acting_identity
from osprey_connectors.logger import get_logger

logger = get_logger("audit_writer")

#: The size an envelope's encoded body stays inside, but for the documented
#: exceptions in the module docstring.
MAX_RECORD_BYTES: int = 2048

#: What an oversize ``detail`` is replaced by. A fixed machine-ish marker, not
#: a truncation: a record that says its context was dropped is honest, while a
#: silently shortened one reads like the whole context.
DETAIL_DROPPED: str = "<dropped: record over the append bound>"


def _encode(record: Mapping[str, Any]) -> bytes:
    """Serialize one record to the bytes of one strict-JSON line.

    Compact separators because the byte budget is the point. ``NaN`` and
    ``Infinity`` are not JSON, so a record that carries one is spelled through
    :func:`_finite` first.
    """
    try:
        text = json.dumps(record, separators=(",", ":"), allow_nan=False)
    except ValueError:
        text = json.dumps(_finite(record), separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8", "replace")


def _finite(value: Any) -> Any:
    """*value* with every non-finite float replaced by its name as a string.

    ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, through dicts (keys
    included), lists and tuples. Everything else is returned as it is, so the
    encoder's own rules still decide what it accepts.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {_finite(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _strict(record: dict[str, Any]) -> dict[str, Any]:
    """*record*, spelled as strict JSON when it carries a non-finite float."""
    try:
        json.dumps(record, allow_nan=False)
    except ValueError:
        return {key: _finite(item) for key, item in record.items()}
    return record


def _body_for(envelope: AuditEnvelope) -> dict[str, Any]:
    """The body to record for *envelope*, degraded to fit where it can be.

    The order is the module docstring's: supplementary context gives way
    first, the executor's evidence never gives way, and identifiers never give
    way — an over-budget record that is true beats an in-budget one that names
    the wrong thing.
    """
    body = envelope.to_dict()
    if len(_encode(body)) <= MAX_RECORD_BYTES:
        return body

    # Supplementary context gives way FIRST, executor records included: an
    # executor record with a small source and a full detail fits once the
    # detail goes. The executor exception below is for the source that is
    # genuinely too big, not for every executor record.
    dropped = None
    if body.get("detail") is not None:
        dropped = {**body, "detail": DETAIL_DROPPED}
        if len(_encode(dropped)) <= MAX_RECORD_BYTES:
            return dropped

    if body.get("source") is not None and body.get("surface") == SURFACE_EXECUTOR:
        # The documented exception: the refused code is why the record exists
        # at all. The detail rides along whole — dropping it did not buy the
        # bound back, so it buys nothing.
        return body

    logger.warning(
        "Audit record for %s/%s exceeds %d bytes with identifiers alone; writing it whole",
        body.get("surface"),
        body.get("subject"),
        MAX_RECORD_BYTES,
    )
    return dropped if dropped is not None else body


def _log_event(envelope: AuditEnvelope) -> None:
    """Put the event on the server log before the write is attempted.

    Refusals warn — a spool that cannot be written then downgrades the trail
    instead of erasing it. Admitted calls only debug: the MCP middleware
    records every tools/call, and a warning apiece would bury the refusals the
    log exists to surface.

    Every field a caller supplies is ``%r``-formatted, so a subject carrying a
    newline cannot forge a second log line.
    """
    message = "Audit %s: surface=%r actor=%r subject=%r reason=%r (posture=%r/%r session=%r)"
    args = (
        envelope.decision,
        envelope.surface,
        envelope.actor,
        envelope.subject,
        envelope.reason,
        envelope.posture,
        envelope.posture_source,
        envelope.session,
    )
    if envelope.decision == DECISION_REFUSED:
        logger.warning(message, *args)
    else:
        logger.debug(message, *args)


def _session_id(given: Any = None) -> str | None:
    """The conversation the entry belongs to: *given*, else the call in scope's, else the process's."""
    if isinstance(given, str) and given:
        return given
    call = current_call()
    if call is not None and call.session_id:
        return call.session_id
    return harness_session_id()


def _write(
    body: Mapping[str, Any],
    *,
    ts: Any,
    tool_use_id: Any,
    on_behalf_of: Any,
    session_id: Any = None,
) -> WriteResult:
    """Write one ``audit`` entry with *body* and the ids it can be joined on."""
    ids: dict[str, Any] = {}
    conversation = _session_id(session_id)
    if conversation:
        ids["session_id"] = conversation
    if isinstance(ts, str) and ts:
        ids["ts"] = ts
    if isinstance(tool_use_id, str) and tool_use_id:
        ids["tool_use_id"] = tool_use_id
    if isinstance(on_behalf_of, str) and on_behalf_of:
        ids["on_behalf_of"] = on_behalf_of
    session = terminal_session()
    if session:
        ids["terminal_session"] = session
    return record_writer.write(KIND_AUDIT, body, **ids)


def record_envelope(envelope: AuditEnvelope) -> WriteResult:
    """Write *envelope* as one ``audit`` entry; its ``entry_id``, or ``None``.

    ``None`` means the entry was not stored — an unwritable spool, the record
    off switch, anything at all. The caller does not branch on it (a refusal is
    enforced by the caller, not by this entry landing), but a test can tell
    "wrote" from "gave up quietly".

    Never raises: see the module docstring.
    """
    try:
        _log_event(envelope)
        return _write(
            _body_for(envelope),
            ts=envelope.ts,
            tool_use_id=envelope.tool_use_id,
            on_behalf_of=envelope.on_behalf_of,
        )
    except Exception:
        logger.warning("Could not write an audit entry", exc_info=True)
        return None


def append_record(surface: str, record: dict[str, Any]) -> WriteResult:
    """Write one prebuilt *record* as an ``audit`` entry on *surface*; its id or ``None``.

    The entry point for the ``tool_call`` surface (:mod:`osprey.audit.tool_call`),
    whose records are not envelopes and carry values by design, so they are
    not bounded by :data:`MAX_RECORD_BYTES`. The body is *record* as given,
    with its ``surface`` field set to *surface*.

    Never raises: see the module docstring.
    """
    try:
        body = _strict({**record, SURFACE_FIELD: surface})
        return _write(
            body,
            ts=record.get("ts"),
            tool_use_id=record.get("tool_use_id"),
            on_behalf_of=record.get("on_behalf_of"),
            session_id=record.get("session_id"),
        )
    except Exception:
        logger.warning("Could not write a %s audit entry", surface, exc_info=True)
        return None


def record(**fields: Any) -> WriteResult:
    """Build an envelope from *fields* and write it; its ``entry_id`` or ``None``.

    The entry point for emitters that have facts rather than an envelope.
    Three things it does that a bare constructor call cannot:

    * ``actor`` defaults to :func:`~osprey_connectors.identity.acting_identity`, so
      no emitter re-implements the ladder or accidentally names the process
      account where the container has a service identity.
    * ``tool_use_id`` defaults to the id of the tool call in scope
      (:func:`osprey.audit.call.current_tool_use_id`), so every recorder
      running inside the middleware's call scope names the call it belongs to
      without a call-site edit. Outside a scope the key is omitted.
    * Construction happens *inside* the never-raises boundary, so an emitter
      does not need a ``try`` of its own around a schema that validates. An
      invalid envelope degrades to a warning and ``None`` — the audit trail
      loses a record, the operation it describes is untouched.

    See :class:`~osprey.audit.envelope.AuditEnvelope` for the fields.
    """
    try:
        fields.setdefault("actor", acting_identity())
        tool_use_id = current_tool_use_id()
        if tool_use_id is not None:
            fields.setdefault("tool_use_id", tool_use_id)
        envelope = AuditEnvelope(**fields)
    except Exception:
        logger.warning(
            "Could not build an audit envelope for surface=%r subject=%r; record dropped",
            fields.get("surface"),
            fields.get("subject"),
            exc_info=True,
        )
        return None
    return record_envelope(envelope)
