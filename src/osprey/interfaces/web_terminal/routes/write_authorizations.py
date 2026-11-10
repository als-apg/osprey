"""The web terminal's approve card: a person approving, seeing and stopping a program's writes.

A program that writes through the connector with no person at a prompt — an
optimizer run — asks for a bounded **write authorization** by writing a
request into its ledger directory (:func:`osprey_connectors.write_authorization.write_request`).
This terminal reads the holders' requests from a read-only bind
(``OSPREY_WRITE_REQUESTS_DIR``) and shows the signed-in login the ones that
name it as owner; approving one issues the grant into this login's own
``authorizations`` directory, beside its control-context record.

Routes, every one answering for the login this terminal acts as
(:func:`~osprey_connectors.identity.acting_identity`), and every refusal in the
posture route's envelope, ``{detail: {reason, phrase?, ceiling?}}``:

* ``GET /api/terminal/write-authorizations`` — the pending requests for this
  login (not expired, not yet approved) and this login's authorizations, each
  marked ``in_force``, ``self_issued`` and ``revocable``. Grants that ended
  more than an hour ago are pruned here. A program's self-issued grant at
  ``allowed`` naming this login is listed from the holders' ledger directory
  this terminal sees read-only, not ``revocable``: moving the switch off
  ``allowed`` ends it.
* ``GET /api/terminal/write-authorizations/requests/{id}`` — one request as the
  approve card shows it (:func:`osprey.write_authorization.card`): channels,
  ranges, the limits bound each range sits inside, current values read through
  this terminal's own connector, steps, expiry, lease, holder, label, the
  completion the run will perform, and that a write the limits refuse still
  costs its step; plus the locks that would refuse the approval.
* ``POST /api/terminal/write-authorizations`` ``{request_id, session_id?}`` —
  approve. Shaped like ``POST /api/terminal/posture``:
  the login is the acting identity, the login's ceiling is checked (``tier_caps`` for ``never`` or
  ``unknown``), then the locks no gesture lifts (``tier_never``,
  ``readonly_run``), then the request against the limits database (422
  ``not_in_limits`` / ``not_writable`` / ``not_numeric`` / ``outside_limits``,
  503 ``limits_unavailable``), then the grant is issued under the record lock,
  refusing 409 ``auth_overlap`` against another program's live grant on the
  same channel of the same machine: one in this login's directory, or a
  self-issued one in the holders' ledger this terminal sees.
* ``POST /api/terminal/write-authorizations/{id}/revoke`` ``{session_id?}`` —
  mark one of this login's grants revoked. The chip's off control for the
  machine does the same for every grant on it, through the switch's own writer.

Every gesture files one audit line, as the posture route's do.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from osprey import write_authorization as approval
from osprey.interfaces.web_terminal.config_view import (
    TIER_UNKNOWN,
    UNREADABLE,
    login_tier,
    rendered_config,
    section_of,
)
from osprey.interfaces.web_terminal.routes import websocket as posture_routes
from osprey_connectors import control_context, posture_store
from osprey_connectors import write_authorization as wa
from osprey_connectors.identity import acting_identity
from osprey_connectors.refusals import refusal_phrase

logger = logging.getLogger(__name__)

router = APIRouter()

#: The audit subjects of the two gestures.
AUDIT_SUBJECT_APPROVE = "write_authorization_approve"
AUDIT_SUBJECT_REVOKE = "write_authorization_revoke"

#: Route outcomes that are not write-verdict reasons: answers about the request.
REASON_INVALID_ID = "invalid_request_id"
REASON_REQUEST_MISSING = "request_missing"
REASON_REQUEST_UNREADABLE = "request_unreadable"
REASON_OWNER_MISMATCH = "owner_mismatch"
REASON_ALREADY_APPROVED = "already_approved"
REASON_NOT_TAKEN = "not_taken"
REASON_GRANT_MISSING = "authorization_missing"

#: How long reading the current values for a card or an approval may take, in
#: all: a machine that does not answer leaves them without values, not waiting.
VALUES_TIMEOUT_S: Final = 3.0
#: How long one channel's read may take.
VALUE_READ_TIMEOUT_S: Final = 2.0

#: Value reads given up on, kept referenced until their pools have closed.
_closing: set[asyncio.Future[Any]] = set()

_ID = re.compile(r"[0-9a-f]{32}")

_ENVELOPE_FIELDS = frozenset({"phrase", "ceiling"})


class ApproveRequest(BaseModel):
    """``POST /api/terminal/write-authorizations``: which request, from which session."""

    request_id: str
    session_id: str | None = None


class RevokeRequest(BaseModel):
    """``POST /api/terminal/write-authorizations/{id}/revoke``."""

    session_id: str | None = None


# -- reading ----------------------------------------------------------------------


async def read_current_values(
    config_path: Path | None, target: str, addresses: list[str]
) -> dict[str, Any]:
    """The current value of each address on *target*, read through this terminal's connector.

    A connector-host pool for the read alone, closed afterwards; the channels
    are read together. A channel that does not answer is ``None``, and so is
    everything when the read as a whole fails or exceeds
    :data:`VALUES_TIMEOUT_S`: a value not read is shown as not read, never
    guessed. The caller waits :data:`VALUES_TIMEOUT_S` at most: a read still
    going then is cancelled, and its pool closes in the background.
    """
    section = section_of(rendered_config(config_path))
    if not isinstance(section, dict) or not addresses:
        return dict.fromkeys(addresses)
    values: dict[str, Any] = dict.fromkeys(addresses)

    async def read_one(connector: Any, address: str) -> None:
        try:
            reading = await connector.read_channel(address, timeout=VALUE_READ_TIMEOUT_S)
            values[address] = reading.value
        except Exception:  # one channel unread is shown as unread
            logger.debug("Could not read %s for the approve card", address, exc_info=True)

    async def read_all() -> None:
        from osprey_connectors.ipc import ConnectorHostPool

        async with ConnectorHostPool(section, config_file=str(config_path)) as pool:
            connector = await pool.connector(target)
            await asyncio.gather(*(read_one(connector, address) for address in addresses))

    task = asyncio.ensure_future(read_all())
    done, _pending = await asyncio.wait({task}, timeout=VALUES_TIMEOUT_S)
    if task in done and task.exception() is None:
        return values
    if task in done:
        logger.warning("Could not read current values on %s for the approve card", target)
    else:
        # Not awaited: the card is answered now, and the pool's close runs on.
        _closing.add(task)
        task.add_done_callback(_closing.discard)
        task.cancel()
        logger.warning("Current values on %s took over %ss; the card shows none", target,
                       VALUES_TIMEOUT_S)  # fmt: skip
    return dict.fromkeys(addresses)


def _json_value(value: Any) -> Any:
    """*value* as JSON can carry it: a number or a string, else its ``repr``."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class _Facts:
    """What an approval or a card decides on, read in one worker-thread hop."""

    login: str
    configured: tuple[str, ...]
    ceiling: str
    lock: posture_store.WriteVerdict | None = None
    request: wa.WriteRequest | None = None
    request_problem: str | None = None
    limits: dict[str, Any] | None = None
    limits_refused: approval.ApprovalRefused | None = None
    approved: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _unique(paths: list[Path | None]) -> list[Path]:
    kept: dict[str, Path] = {}
    for path in paths:
        if path is not None:
            kept.setdefault(os.path.normpath(path), path)
    return list(kept.values())


def _request_directories() -> list[Path]:
    """Where a request naming this login is filed: the holders' directory, then this login's own.

    A program run on the host with no ledger bound files its request in its
    owner's grants directory, and keeps its ledger beside the grant.
    """
    own = wa.authorizations_dir()
    return _unique([wa.requests_dir(), None if own is None else own / wa.REQUESTS_DIR_NAME])


def _ledger_directories() -> list[Path]:
    """Where the holders' ledgers are, as this terminal sees them: theirs, then this login's own."""
    return _unique([*wa.self_issued_directories(), wa.authorizations_dir()])


def _read_request(request_id: str) -> tuple[wa.WriteRequest | None, str | None]:
    """The request *request_id*, or why there is none to show."""
    places = [directory / f"{request_id}.json" for directory in _request_directories()]
    if not places:
        return None, REASON_REQUEST_MISSING
    path = next((place for place in places if place.exists()), places[0])
    try:
        return wa.read_request(path), None
    except FileNotFoundError:
        return None, REASON_REQUEST_MISSING
    except Exception:  # every failure on a present request refuses it
        logger.warning("The write request %s cannot be read", request_id, exc_info=True)
        return None, REASON_REQUEST_UNREADABLE


def _own_grant_exists(request_id: str) -> bool:
    directory = wa.authorizations_dir()
    return directory is not None and (directory / f"{request_id}.json").exists()


def _facts(config_path: Path | None, request_id: str) -> _Facts:
    """Render, ceiling, locks, the request and the limits database. BLOCKING."""
    login = acting_identity()
    config = rendered_config(config_path)
    section = section_of(config)
    facts = _Facts(
        login=login,
        configured=posture_routes._configured_target_names(section),
        ceiling=login_tier(section, login),
    )
    facts.request, facts.request_problem = _read_request(request_id)
    if facts.request is None:
        return facts
    target = facts.request.target
    if target in facts.configured:
        verdict = posture_store.write_verdict(
            None if section is UNREADABLE else section,
            target,
            owner=login,
            ignore_launch_pin=True,
        )
        if verdict.reason in posture_routes._TURN_ON_LOCKS:
            facts.lock = verdict
    facts.approved = _own_grant_exists(request_id)
    if config is UNREADABLE or not isinstance(config, dict):
        facts.limits_refused = approval.ApprovalRefused(
            approval.LIMITS_UNAVAILABLE, "No approval without a readable render."
        )
        return facts
    try:
        facts.limits = approval.load_limits(config, config_path)
    except approval.ApprovalRefused as refused:
        facts.limits_refused = refused
    return facts


def _locks(facts: _Facts) -> dict[str, Any] | None:
    """The first lock that refuses an approval by this login, as the envelope's fields."""
    if facts.ceiling in ("never", TIER_UNKNOWN):
        return {"status": 403, "reason": posture_routes.POSTURE_REASON_TIER_CAPS,
                "ceiling": facts.ceiling}  # fmt: skip
    if facts.lock is not None:
        reason = str(facts.lock.reason)
        return {"status": 403, "reason": reason,
                "phrase": refusal_phrase(reason, facts.lock.kind)}  # fmt: skip
    return None


# -- refusing -----------------------------------------------------------------------


def _gesture(login: str, what: str, request_id: str) -> str:
    return f"login={login} {what}={request_id}"


def _refuse(
    session_id: str | None,
    *,
    subject: str,
    gesture: str,
    status_code: int,
    reason: str,
    **fields: Any,
) -> HTTPException:
    """File one refusal line and build the posture-shaped answer."""
    unknown = set(fields) - _ENVELOPE_FIELDS
    if unknown:
        raise TypeError(f"Not an approval refusal field: {sorted(unknown)}")
    from osprey.audit.envelope import DECISION_REFUSED

    posture_routes._record_control_gesture(
        session_id,
        subject=subject,
        decision=DECISION_REFUSED,
        reason=reason,
        detail=f"{gesture} outcome={reason}",
    )
    return HTTPException(status_code=status_code, detail={"reason": reason, **fields})


def _check_ids(session_id: str | None, request_id: str) -> None:
    try:
        posture_routes._require_session_uuid(session_id)
    except HTTPException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"reason": posture_routes.POSTURE_REASON_INVALID_SESSION},
        ) from exc
    if _ID.fullmatch(request_id) is None:
        raise HTTPException(status_code=400, detail={"reason": REASON_INVALID_ID})


# -- the routes ----------------------------------------------------------------------


def _listing() -> dict[str, Any]:
    """Pending requests for this login, and this login's grants. BLOCKING."""
    login = acting_identity()
    now = time.time_ns()
    record, _status = control_context.read_record_status()
    own = wa.authorizations_dir()
    if own is not None:
        try:
            wa.prune_grants(own, now_ns=now)
        except Exception:  # a listing that could not prune still lists
            logger.warning("Could not prune the authorizations in %s", own, exc_info=True)
    # This login's own grants, and the holders' self-issued grants naming it,
    # seen read-only: those end by moving the switch off ``allowed``, never by
    # a revoke from here. The ledgers are the holders', in the same view, and
    # a program on the host keeps its ledger beside the grant.
    found, unlisted = wa.listed_grants([] if own is None else [own], wa.self_issued_directories())
    ledgers = _ledger_directories()
    unreadable = len(unlisted)
    grants: list[dict[str, Any]] = []
    for grant, found_in in found:
        if grant.owner != login:
            continue
        standing = wa.grant_standing(grant, record, ledgers, now_ns=now)
        row = approval.grant_summary(grant, used=standing.used, now_ns=now)
        row["in_force"] = standing.in_force
        row["self_issued"] = grant.approver.surface == wa.SELF_ISSUE_SURFACE
        row["revocable"] = found_in == own
        grants.append(row)
    pending: list[dict[str, Any]] = []
    approved = {row["id"] for row in grants}
    for directory in _request_directories():
        try:
            requests, broken = wa.list_requests(directory)
        except OSError:
            requests, broken = (), (directory,)
        unreadable += len(broken)
        for request in requests:
            if request.owner != login or request.id in approved or now >= request.expires_at_ns:
                continue
            approved.add(request.id)
            pending.append(approval.card(request, [], now_ns=now) | {"channels": [
                {"address": address, "min": bound.min, "max": bound.max}
                for address, bound in request.channels.items()
            ]})  # fmt: skip
    return {
        "login": login,
        "requests": pending,
        "authorizations": grants,
        "unreadable": unreadable,
        "limits_note": approval.LIMITS_COST_SENTENCE,
    }


@router.get("/api/terminal/write-authorizations")
async def list_write_authorizations():
    """The pending requests that name this login, and this login's authorizations."""
    return await run_in_threadpool(_listing)


@router.get("/api/terminal/write-authorizations/requests/{request_id}")
async def get_write_request(request_id: str, request: Request):
    """One request as the approve card shows it, with current values and the locks."""
    _check_ids(None, request_id)
    config_path = request.app.state.config_path
    facts: _Facts = await run_in_threadpool(_facts, config_path, request_id)
    if facts.request is None:
        status = 404 if facts.request_problem == REASON_REQUEST_MISSING else 422
        raise HTTPException(status_code=status, detail={"reason": facts.request_problem})
    if facts.request.owner != facts.login:
        raise HTTPException(status_code=403, detail={"reason": REASON_OWNER_MISMATCH})
    addresses = list(facts.request.channels)
    current = {
        address: _json_value(value)
        for address, value in (
            await read_current_values(config_path, facts.request.target, addresses)
        ).items()
    }
    checks = approval.check_channels(facts.request, facts.limits or {}, current)
    body = approval.card(facts.request, checks, current=current)
    refusal = None
    if facts.request.target not in facts.configured:
        refusal = {"reason": posture_routes.POSTURE_REASON_UNKNOWN_TARGET}
    elif (lock := _locks(facts)) is not None:
        refusal = {key: value for key, value in lock.items() if key != "status"}
    elif facts.limits_refused is not None:
        refusal = {"reason": facts.limits_refused.reason, "phrase": facts.limits_refused.detail}
    elif facts.approved:
        refusal = {"reason": REASON_ALREADY_APPROVED}
    if refusal is not None:
        body["approvable"] = False
        body["refusal"] = refusal
    return body


@router.post("/api/terminal/write-authorizations")
async def approve_write_authorization(body: ApproveRequest, request: Request):
    """Approve one request for this login: the grant is issued into this login's directory.

    The ladder, in order (every non-200 is ``{detail: {reason, ...}}``):

    * **400** ``invalid_session_id``, ``invalid_request_id``, ``unknown_target``.
    * **404** ``request_missing``; **422** ``request_unreadable``.
    * **403** ``owner_mismatch`` — the request names another login.
    * **403** ``tier_caps`` with ``ceiling`` — the login's ceiling is
      ``never`` or ``unknown``; ``tier_never`` / ``readonly_run`` with
      ``phrase`` — a lock no gesture lifts.
    * **503** ``limits_unavailable``; **422** ``not_in_limits`` /
      ``not_writable`` / ``not_numeric`` / ``outside_limits`` with ``phrase``.
    * **409** ``auth_overlap`` (another program holds a channel of it),
      ``already_approved``; **410** ``auth_expired``.
    * **503** ``record_unavailable``, ``not_taken`` (nothing was written).
    * **200** ``{authorization: {...}}``.
    """
    from osprey.audit.envelope import DECISION_ALLOWED

    app = request.app
    session_id = body.session_id
    _check_ids(session_id, body.request_id)
    login = acting_identity()
    gesture = _gesture(login, "request", body.request_id)

    def refuse(status_code: int, reason: str, **fields: Any) -> HTTPException:
        return _refuse(
            session_id,
            subject=AUDIT_SUBJECT_APPROVE,
            gesture=gesture,
            status_code=status_code,
            reason=reason,
            **fields,
        )

    config_path = app.state.config_path
    facts: _Facts = await run_in_threadpool(_facts, config_path, body.request_id)
    if facts.request is None:
        missing = facts.request_problem == REASON_REQUEST_MISSING
        raise refuse(404 if missing else 422, str(facts.request_problem))
    wanted = facts.request
    if wanted.owner != login:
        raise refuse(403, REASON_OWNER_MISMATCH)
    # Judged on the one parsed request the grant is then issued from, so
    # nothing the holder writes after this read reaches the grant.
    if wanted.target not in facts.configured:
        raise refuse(400, posture_routes.POSTURE_REASON_UNKNOWN_TARGET)
    lock = _locks(facts)
    if lock is not None:
        status = lock.pop("status")
        reason = lock.pop("reason")
        raise refuse(status, reason, **lock)
    if facts.limits_refused is not None or facts.limits is None:
        refused = facts.limits_refused or approval.ApprovalRefused(
            approval.LIMITS_UNAVAILABLE, "No approval without the limits database."
        )
        raise refuse(503, refused.reason, phrase=refused.detail)
    current = await read_current_values(config_path, wanted.target, list(wanted.channels))
    problem = approval.refusal(approval.check_channels(wanted, facts.limits, current))
    if problem is not None:
        raise refuse(422, problem.reason, phrase=problem.detail)

    try:
        grant = await run_in_threadpool(lambda: wa.approve_request(wanted, surface="chip"))
    except wa.AuthorizationRefused as exc:
        status = {"auth_overlap": 409, "auth_expired": 410}.get(exc.reason, 503)
        phrase = refusal_phrase(exc.reason)
        raise refuse(
            status, exc.reason, phrase=f"{phrase} {exc.detail}" if exc.detail else phrase
        ) from exc
    except FileExistsError as exc:
        raise refuse(409, REASON_ALREADY_APPROVED) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("The approval %s was not taken", gesture, exc_info=True)
        raise refuse(503, REASON_NOT_TAKEN) from exc

    posture_routes._push_control_context(app)
    posture_routes._record_control_gesture(
        session_id,
        subject=AUDIT_SUBJECT_APPROVE,
        decision=DECISION_ALLOWED,
        reason="authorization_issued",
        detail=(
            f"{gesture} outcome=authorization_issued holder={grant.holder} "
            f"target={grant.target} steps={grant.steps} channels={','.join(grant.channels)}"
        ),
    )
    logger.info("Write authorization %s approved by %s for %s", grant.id, login, grant.holder)
    summary = approval.grant_summary(grant)
    summary["in_force"] = True
    return {"authorization": summary}


def _revoke(authorization: str, login: str) -> bool:
    directory = wa.authorizations_dir()
    if directory is None:
        raise RuntimeError("No authorizations directory is reachable")
    return bool(wa.revoke_grants(by=login, directory=directory, ids=[authorization]))


@router.post("/api/terminal/write-authorizations/{authorization}/revoke")
async def revoke_write_authorization(authorization: str, body: RevokeRequest, request: Request):
    """Mark one of this login's authorizations revoked: the holder's next write refuses.

    **200** ``{id, revoked}`` (``revoked`` false when it was already stopped or
    is not this login's); **400** for a malformed id; **503** ``not_taken``.
    A stop is never refused for tier, lock or a run in flight.
    """
    from osprey.audit.envelope import DECISION_ALLOWED

    session_id = body.session_id
    _check_ids(session_id, authorization)
    login = acting_identity()
    gesture = _gesture(login, "authorization", authorization)
    try:
        revoked = await run_in_threadpool(_revoke, authorization, login)
    except (OSError, RuntimeError) as exc:
        logger.warning("The revocation %s was not taken", gesture, exc_info=True)
        raise _refuse(
            session_id,
            subject=AUDIT_SUBJECT_REVOKE,
            gesture=gesture,
            status_code=503,
            reason=REASON_NOT_TAKEN,
            phrase=posture_routes.STOP_NOT_TAKEN_PHRASE,
        ) from exc
    posture_routes._push_control_context(request.app)
    posture_routes._record_control_gesture(
        session_id,
        subject=AUDIT_SUBJECT_REVOKE,
        decision=DECISION_ALLOWED,
        reason="authorization_revoked" if revoked else REASON_GRANT_MISSING,
        detail=f"{gesture} outcome={'authorization_revoked' if revoked else REASON_GRANT_MISSING}",
    )
    return {"id": authorization, "revoked": revoked}
