"""Audit read route — ``GET /api/audit/recent``.

:mod:`osprey.audit.writer` writes every safety decision as an ``audit`` entry in
the agent record; this route serves the newest of them back to the Config panel,
read through :class:`~osprey.agent_record.client.RecordClient` under this
process's own record login.

Three things about it are deliberate.

**It is behind the Config panel tier, using that tier's own gate.** The audit
trail names every safety decision the deployment made — the subjects, the
sessions, the actors — so it is at least as sensitive as the ``config.yml`` and
``.claude/`` surfaces the panel edits. Rather than spell a second refusal, the
handler calls :func:`~osprey.interfaces.web_terminal.routes.config._require_config_panel`
directly: one switch (``web.config_panel.enabled``), one wording, nothing to
drift. The gate runs FIRST, ahead of the record read and ahead of every
parameter check — which is why ``limit`` arrives as a **string** and is parsed
and clamped inside the handler instead of being annotated ``int`` or bounded by
``Query(ge=..., le=...)``. Either of those hands FastAPI a coercion or
validation step that answers 422 before the handler runs, telling a caller who
may not open the Config panel that the route exists and what it takes. Doing it
by hand also buys the kinder semantics: an operator who asks for more than the
ceiling gets the ceiling, not a refusal.

**Only what this identity wrote.** The read asks for ``written_by="self"``, so
the record answers only the entries whose trusted identity is this login's own
source identity — the same identity the writer's spool directory is named for.
No request field names an identity; the deployment-wide view over every
identity is the history page's, by design. The one client-supplied string, the
``surface`` filter, is validated as an identifier and passed to the record as a
value to compare, never as anything that selects storage.

**Nothing here may take the web terminal down to report an audit trail, and
nothing here passes a partial answer off as a whole one.** A record that is
unconfigured, unreachable or refuses the read answers 200 in the same shape with
no records, a log line, and ``"record": "unavailable"`` plus the ``reason`` — the
way the writer swallows its own errors rather than costing the operation it
records, without ever making "the record could not answer" look like "there was
nothing to answer". A served answer the record marks incomplete (a scan, byte
or time limit) is ``"record": "incomplete"``; only a whole answer is
``"record": "complete"``.

The full ``tool_call`` surface is never served: its entries carry payloads the
recent-activity list is not for, so ``surface=tool_call`` answers nothing. An
unfiltered read leaves those entries out; when they crowd the newest page, the
route reads once more at the record's own row ceiling so decisions behind them
are still served, and when even that page holds fewer than ``limit`` decisions
while the record has more, the answer says ``"incomplete"``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from osprey.agent_record.client import ClientAnswer, RecordClient
from osprey.agent_record.contract import SURFACE_FIELD
from osprey.audit.tool_call import SURFACE_TOOL_CALL
from osprey.interfaces.web_terminal.routes.config import _require_config_panel
from osprey_connectors.identity import acting_identity

logger = logging.getLogger(__name__)

router = APIRouter()

#: Records returned when the caller names no ``limit``. Sized for a panel that
#: shows a scrollable recent-activity list, not for bulk export — the history
#: page is the tool for reading across identities.
DEFAULT_LIMIT: int = 50

#: Hard ceiling on ``limit``. Clamped to, never refused with: see the module
#: docstring on why nothing may answer before the gate.
MAX_LIMIT: int = 200

#: Floor on ``limit``. Zero and negative values clamp here rather than meaning
#: "everything" or "nothing" — a request that asks for records gets records.
MIN_LIMIT: int = 1

#: A surface the ``surface`` filter may name: an identifier, ASCII, at most 128
#: bytes. An ALLOWLIST rather than a list of forbidden characters — the set of
#: legitimate surface names is small and known. A surface outside this shape is
#: still served by the unfiltered read; it just cannot be named alone.
_SAFE_SURFACE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

#: The ``record`` field: the record answered whole, answered only part, or did
#: not answer at all. A reader never shows an empty answer as complete.
COMPLETE: str = "complete"
INCOMPLETE: str = "incomplete"
UNAVAILABLE: str = "unavailable"

#: Names that pass a character check but are not a surface.
_RESERVED_STEMS = frozenset({".", ".."})


def record_client() -> RecordClient:
    """The record client this process reads with, configured from its environment.

    Resolved per request and reached through the module, so a test replaces it
    with ``monkeypatch.setattr(audit, "record_client", ...)``. Never raises: an
    unconfigured client answers every read unavailable.
    """
    return RecordClient.from_env(os.environ)


def _clamp(limit: int) -> int:
    """Bring *limit* inside ``[MIN_LIMIT, MAX_LIMIT]``."""
    return max(MIN_LIMIT, min(MAX_LIMIT, limit))


def _parsed_limit(limit: str) -> int:
    """*limit* as a clamped integer, or a 400-shaped refusal.

    Taken off the request as a **string** and parsed here rather than annotated
    ``int``: an ``int`` annotation hands FastAPI a coercion step that answers
    422 before the handler — and therefore before the tier gate — so a caller
    who may not open the Config panel could still learn the route exists and
    what it takes. Everything this route decides, it decides after the gate.

    Raises:
        HTTPException: 400 when the value is not an integer. Out of range is
            not an error (it clamps); not a number is.
    """
    try:
        return _clamp(int(limit))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"limit must be a whole number; values outside "
                f"[{MIN_LIMIT}, {MAX_LIMIT}] are clamped, not refused"
            ),
        ) from None


def _validated_surface(surface: str | None) -> str | None:
    """*surface* if it is an identifier-shaped surface name, else a 400-shaped refusal.

    Raises:
        HTTPException: 400 when the value is not identifier-shaped.
    """
    if surface is None:
        return None
    if surface in _RESERVED_STEMS or not _SAFE_SURFACE.match(surface):
        raise HTTPException(
            status_code=400,
            detail=(
                "surface must name a single audit surface: letters, digits, dot, "
                "dash and underscore only, and no path separators"
            ),
        )
    return surface


@router.get("/api/audit/recent")
def recent_audit_records(
    request: Request,
    limit: str = str(DEFAULT_LIMIT),
    surface: str | None = None,
) -> dict[str, Any]:
    """Serve the newest audit records this identity wrote, newest first.

    Args:
        request: Incoming request, for ``app.state`` and the tier gate.
        limit: How many records to return, taken as text and parsed after the
            gate; clamped into ``[MIN_LIMIT, MAX_LIMIT]`` rather than
            validated, so the gate stays the first thing that answers.
        surface: Optional surface to read alone, compared with each entry's
            ``surface``.

    Returns:
        ``{"identity", "ledgers", "limit", "records", "record"}`` — the
        identity whose entries were read, the surfaces the records came from,
        the effective limit, the records themselves (each the envelope exactly
        as the writer recorded it, newest first), and whether the record's
        answer was ``complete``, ``incomplete`` or ``unavailable``; the last
        two add a ``reason``.

    Raises:
        HTTPException: 403 when the Config panel tier is disabled, 400 when
            *surface* is not a usable ledger name or *limit* is not a number.
    """
    _require_config_panel(request)

    wanted = _validated_surface(surface)
    effective = _parsed_limit(limit)
    identity = acting_identity()
    if wanted == SURFACE_TOOL_CALL:
        return _answer(identity, wanted, effective, [])

    client = record_client()
    answer = client.recent_audit(limit=effective, surface=wanted, written_by="self")
    records, displaced = _decisions(answer)
    if displaced and answer.more and len(records) < effective:
        # tool_call entries filled the page ahead of decisions: read the
        # record's whole row ceiling once, then keep the newest ``effective``.
        answer = client.recent_audit(surface=wanted, written_by="self")
        records, displaced = _decisions(answer)
    if not answer.available or not isinstance(answer.data, list):
        reason = answer.unavailable or "the record answered no list"
        logger.warning("Could not read the audit record; serving no records: %s", reason)
        return _answer(identity, wanted, effective, [], state=UNAVAILABLE, reason=reason)

    records = records[:effective]
    if not answer.complete:
        return _answer(
            identity,
            wanted,
            effective,
            records,
            state=INCOMPLETE,
            reason=answer.reason or "the record answered an incomplete page",
        )
    if displaced and answer.more and len(records) < effective:
        return _answer(
            identity,
            wanted,
            effective,
            records,
            state=INCOMPLETE,
            reason="newer tool_call entries filled the record's page ahead of older decisions",
        )
    return _answer(identity, wanted, effective, records)


def _decisions(answer: ClientAnswer) -> tuple[list[dict[str, Any]], bool]:
    """The served bodies of *answer*'s entries, and whether a tool_call one was dropped."""
    if not answer.available or not isinstance(answer.data, list):
        return [], False
    records: list[dict[str, Any]] = []
    displaced = False
    for item in answer.data:
        if not isinstance(item, dict) or not isinstance(item.get("body"), dict):
            continue
        if item["body"].get(SURFACE_FIELD) == SURFACE_TOOL_CALL:
            displaced = True
            continue
        records.append(dict(item["body"]))
    return records, displaced


def _answer(
    identity: str,
    surface: str | None,
    limit: int,
    records: list[dict[str, Any]],
    *,
    state: str = COMPLETE,
    reason: str = "",
) -> dict[str, Any]:
    """The response body: who read, which surfaces answered, the limit, the
    records, and whether the record's answer was whole."""
    if surface is not None:
        surfaces = [surface] if records else []
    else:
        surfaces = sorted({str(r.get(SURFACE_FIELD)) for r in records if r.get(SURFACE_FIELD)})
    body: dict[str, Any] = {
        "identity": identity,
        "ledgers": surfaces,
        "limit": limit,
        "records": records,
        "record": state,
    }
    if state != COMPLETE:
        body["reason"] = reason
    return body
