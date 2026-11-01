"""The watchdog routes: prompt intake, the session's open flags, the operator's answer.

``POST /api/watchdog/prompt`` is the trigger hook's door and carries only the
panel token, so what it accepts is narrow: three canonical UUIDs, of which the
session key names a session this process runs. It answers 202 before anything is
scheduled, and a repeated ``prompt_ref`` is answered 202 too without a second run.
No answer from intake drops a hold, because the hook falls back to its record
entry on any refusal and ``open_flag`` skips an existing ``(source, prompt_ref)``.

``GET /api/watchdog/flags`` and ``POST /api/watchdog/flags/answer`` are operator
tier by not being listed in ``PANEL_TIER_ROUTES``: an agent holding the panel
token can neither read its flags nor release its own hold. They accept any key
that is one safe path segment, since a session the hook held through the record
entry need not have a UUID key and its operator must still be able to answer.

A Stop ends the turn before it releases anything. Finding no turn says nothing
about what the agent started, so that is read on every Stop: a turn seen to end, or
none at all, with nothing still running closes the named flags; otherwise they stay
open and marked ``stop_unconfirmed``, naming what still runs. The session is never
terminated, and a flag another answer closed meanwhile is neither released nor
marked again.

The hold record is the sole truth. Frames are change notices that carry ids and
never prompt text, and a missed one costs a poll, never the gesture. The root is
the one ``app.state.session_hold_root`` resolved at startup and is always passed
explicitly: with no watchdog enabled it is ``None`` and nothing here touches
``session_hold``, whatever the environment says.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel, Field

from osprey.interfaces.web_terminal.operator_session import session_hold_root_of
from osprey.interfaces.web_terminal.process_tree import ProcessTableUnreadable
from osprey.interfaces.web_terminal.session_handoff import (
    INTERRUPT_GRACE_S,
    interrupt_pty_turn,
)
from osprey.interfaces.web_terminal.session_key import is_posture_key
from osprey.services.watchdog.verdict import (
    UNATTRIBUTED_OPERATOR,
    make_prompt_ref,
    record_answer,
    record_stop_unconfirmed,
)
from osprey_connectors.session_hold import (
    HoldFlag,
    answer_flags,
    is_safe_session_key,
    mark_stop_unconfirmed,
    open_flags,
)

logger = logging.getLogger(__name__)

router = APIRouter()

#: Length bound on identifiers: generous for a UUID or a flag id.
_MAX_ID_LEN = 256

#: Longest prompt intake takes. A longer one is refused and the hook's record
#: entry judges it instead.
_MAX_PROMPT_CHARS = 1_000_000

#: Most flag ids one answer names.
_MAX_FLAG_IDS = 256

#: How many ``prompt_ref`` values intake remembers. The dedupe only has to outlast
#: the hook's retries; past it ``open_flag`` still yields one flag per submission.
_SEEN_PROMPT_REFS = 4096

#: The source of the one flag a store reports for an unreadable record.
_RECORD_SOURCE = "session_hold"

#: What a Stop names when the process table could not say what the agent started.
_STARTED_UNREADABLE = "(running commands could not be read)"


def session_is_live(app: Any, session_key: str) -> bool:
    """Whether *session_key* names a session this process runs now.

    The terminal pool, the chat pool (a creation still starting counts) and the
    operator websocket's sessions: every surface that stamps the key on a child.
    """
    state = app.state
    pty_registry = getattr(state, "pty_registry", None)
    if pty_registry is not None and pty_registry.get_session(session_key) is not None:
        return True
    registry = getattr(state, "operator_registry", None)
    if registry is None:
        return False
    return registry.has_chat_key(session_key) or registry.get_session(session_key) is not None


def make_notifier(app: Any) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """The runner's and the answer route's notifier: the broadcaster and the chat feed."""

    async def notify(frame: dict[str, Any]) -> None:
        state = app.state
        broadcaster = getattr(state, "broadcaster", None)
        if broadcaster is not None:
            try:
                broadcaster.broadcast(dict(frame))
            except Exception as exc:  # a missed notice costs one poll, not the gesture
                logger.warning("Watchdog broadcast failed: %s", type(exc).__name__)
        registry = getattr(state, "operator_registry", None)
        key = frame.get("session_key")
        session = registry.get_chat_session(key) if registry is not None and key else None
        if session is not None and session.is_active:
            try:
                await session.post_notice(frame)
            except Exception as exc:
                logger.warning("Watchdog chat notice failed: %s", type(exc).__name__)

    return notify


def _first_sight(app: Any, prompt_ref: str) -> bool:
    """Remember *prompt_ref*; whether this is the first time it was seen."""
    seen: OrderedDict[str, None] | None = getattr(app.state, "watchdog_seen_prompts", None)
    if seen is None:
        seen = OrderedDict()
        app.state.watchdog_seen_prompts = seen
    if prompt_ref in seen:
        return False
    seen[prompt_ref] = None
    while len(seen) > _SEEN_PROMPT_REFS:
        seen.popitem(last=False)
    return True


class PromptSubmission(BaseModel):
    """Body of ``POST /api/watchdog/prompt``."""

    session_key: str = Field(max_length=_MAX_ID_LEN)
    record_session_id: str = Field(max_length=_MAX_ID_LEN)
    submission_id: str = Field(max_length=_MAX_ID_LEN)
    prompt: str = Field(max_length=_MAX_PROMPT_CHARS)


@router.post("/api/watchdog/prompt", status_code=202)
async def post_prompt(
    body: PromptSubmission, request: Request, background: BackgroundTasks
) -> dict[str, bool]:
    """Take one operator prompt from the trigger hook and judge it after answering."""
    if not all(
        is_posture_key(value)
        for value in (body.session_key, body.record_session_id, body.submission_id)
    ):
        raise HTTPException(status_code=400, detail="invalid_session_id")
    app = request.app
    runner = getattr(app.state, "watchdog_runner", None)
    if runner is None or session_hold_root_of(app) is None:
        raise HTTPException(status_code=404, detail="watchdog_disabled")
    if not session_is_live(app, body.session_key):
        raise HTTPException(status_code=404, detail="unknown_session")
    received_at = time.monotonic()
    if not _first_sight(app, make_prompt_ref(body.record_session_id, body.submission_id)):
        return {"accepted": True, "duplicate": True}

    # An async function, so it runs on the loop that owns the runner's tasks.
    async def schedule() -> None:
        runner.on_prompt(
            session_key=body.session_key,
            record_session_id=body.record_session_id,
            submission_id=body.submission_id,
            prompt=body.prompt,
            received_at=received_at,
            can_hold=True,
        )

    background.add_task(schedule)
    return {"accepted": True, "duplicate": False}


def _checked_key(session_key: str) -> str:
    if not is_safe_session_key(session_key):
        raise HTTPException(status_code=400, detail="invalid_session_id")
    return session_key


@router.get("/api/watchdog/flags")
async def get_flags(
    request: Request,
    session_key: Annotated[str, Query(max_length=_MAX_ID_LEN)] = "",
) -> dict[str, Any]:
    """Report whether a watchdog runs here and the session's open flags."""
    root = session_hold_root_of(request.app)
    if root is None:
        return {"enabled": False, "flags": []}
    key = _checked_key(session_key)
    flags = await asyncio.to_thread(open_flags, key, root=root)
    return {
        "enabled": True,
        "flags": [{**asdict(flag), "watchdog": flag.source} for flag in flags],
    }


class FlagAnswer(BaseModel):
    """Body of ``POST /api/watchdog/flags/answer``."""

    session_key: str = Field(max_length=_MAX_ID_LEN)
    flag_ids: list[Annotated[str, Field(max_length=_MAX_ID_LEN)]] = Field(max_length=_MAX_FLAG_IDS)
    answer: Literal["stop", "continue"]


def _open_by_id(key: str, root: Path) -> dict[str, HoldFlag]:
    """The session's open flags by id. BLOCKING.

    Read before closing or marking: a closed flag is no longer reported, and the
    audit entry needs what the flag said while it was open.
    """
    return {flag.flag_id: flag for flag in open_flags(key, root=root)}


class _AuditIds(TypedDict):
    """What names a flag in its audit entry."""

    watchdog: str
    prompt_ref: str
    flag_id: str


def _audit_ids(flag: HoldFlag | None, flag_id: str) -> _AuditIds:
    """A flag's audit names; one not read open answers to the record."""
    return {
        "watchdog": flag.source if flag is not None else _RECORD_SOURCE,
        "prompt_ref": flag.prompt_ref if flag is not None else "",
        "flag_id": flag_id,
    }


def _release(
    key: str,
    flag_ids: list[str],
    answer: Literal["stop", "continue"],
    operator: str,
    root: Path,
    *,
    denied: tuple[str, ...] = (),
    stopped: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Close exactly the named flags and audit each one released. BLOCKING."""
    named = _open_by_id(key, root)
    released = answer_flags(key, flag_ids, answer=answer, operator=operator, root=root)
    for flag_id in released:
        flag = named.get(flag_id)
        unconfirmed = flag is not None and flag.stop_unconfirmed
        record_answer(
            **_audit_ids(flag, flag_id),
            session_key=key,
            answer=answer,
            operator=operator,
            after_stop_unconfirmed=unconfirmed,
            started=flag.started
            if flag is not None and unconfirmed and answer == "continue"
            else (),
            denied=denied,
            stopped=stopped,
        )
    return released


@dataclass(frozen=True)
class _StopReport:
    """What stopping one session's turn found."""

    confirmed: bool
    denied: tuple[str, ...] = ()
    stopped: tuple[str, ...] = ()
    started: tuple[str, ...] = ()


def _started_labels(session: Any) -> tuple[str, ...]:
    """The commands *session*'s agent started and still running. BLOCKING."""
    try:
        return tuple(group.label for group in session.started_commands())
    except ProcessTableUnreadable:
        return (_STARTED_UNREADABLE,)


async def _stop_chat(session: Any) -> _StopReport:
    stop = await session.stop_turn()
    started = await asyncio.to_thread(_started_labels, session)
    return _StopReport(
        stop.outcome != "abandoned" and not started, stop.denied, stop.stopped_tasks, started
    )


async def _stop_pty(app: Any, key: str, session: Any) -> _StopReport:
    # The interrupt ends the turn, and a pending approval with it.
    outcome = await interrupt_pty_turn(app, key, grace=INTERRUPT_GRACE_S)
    started = await asyncio.to_thread(_started_labels, session)
    return _StopReport(outcome != "unconfirmed" and not started, started=started)


async def _stop_session(app: Any, key: str) -> _StopReport:
    """Stop every surface running under *key*; confirmed only if all of them are."""
    state = app.state
    stops: list[Awaitable[_StopReport]] = []
    registry = getattr(state, "operator_registry", None)
    if registry is not None:
        chats = {registry.get_chat_session(key), registry.get_session(key)} - {None}
        stops.extend(_stop_chat(session) for session in chats)
    pty_registry = getattr(state, "pty_registry", None)
    pty = pty_registry.get_session(key) if pty_registry is not None else None
    if pty is not None:
        stops.append(_stop_pty(app, key, pty))
    reports = await asyncio.gather(*stops, return_exceptions=True)
    results = _StopReport(True)
    for report in reports:
        if isinstance(report, Exception):
            # Not knowing whether the turn stopped is not a stop.
            logger.error("Watchdog stop failed: %s", type(report).__name__)
            report = _StopReport(False)
        elif isinstance(report, BaseException):
            raise report
        results = _StopReport(
            results.confirmed and report.confirmed,
            results.denied + report.denied,
            results.stopped + report.stopped,
            results.started + report.started,
        )
    return results


def _mark_unconfirmed(
    key: str,
    flag_ids: list[str],
    answered: list[HoldFlag],
    operator: str,
    report: _StopReport,
    root: Path,
) -> list[tuple[str, str]]:
    """Mark the named open flags and audit each one marked; ``(id, source)`` pairs. BLOCKING."""
    named = _open_by_id(key, root)
    marked = mark_stop_unconfirmed(
        key, flag_ids, operator=operator, started=report.started, root=root
    )
    pairs: list[tuple[str, str]] = []
    for flag_id in marked:
        ids = _audit_ids(named.get(flag_id), flag_id)
        record_stop_unconfirmed(
            **ids,
            session_key=key,
            operator=operator,
            started=report.started,
            denied=report.denied,
            stopped=report.stopped,
        )
        pairs.append((flag_id, ids["watchdog"]))
    if not pairs:
        _audit_unattached(key, answered, operator, report)
    return pairs


def _audit_unattached(
    key: str, answered: list[HoldFlag], operator: str, report: _StopReport
) -> None:
    """Audit a Stop that acted on *answered* but found them no longer open. BLOCKING."""
    for flag in answered:
        ids = _audit_ids(flag, flag.flag_id)
        if report.confirmed:
            record_answer(
                **ids,
                session_key=key,
                answer="stop",
                operator=operator,
                denied=report.denied,
                stopped=report.stopped,
            )
        else:
            record_stop_unconfirmed(
                **ids,
                session_key=key,
                operator=operator,
                started=report.started,
                denied=report.denied,
                stopped=report.stopped,
            )


def _named_open(key: str, flag_ids: list[str], root: Path) -> list[HoldFlag]:
    """The named flags still open. BLOCKING."""
    wanted = set(flag_ids)
    return [flag for flag in open_flags(key, root=root) if flag.flag_id in wanted]


async def _answer_stop(
    app: Any, key: str, flag_ids: list[str], operator: str, root: Path
) -> dict[str, list[str]]:
    # A stale dialog answering flags another tab already closed stops nothing.
    named = await asyncio.to_thread(_named_open, key, flag_ids, root)
    if not named:
        return {"released": [], "stop_unconfirmed": []}
    report = await _stop_session(app, key)
    if report.confirmed:
        released = await asyncio.to_thread(
            _release,
            key,
            flag_ids,
            "stop",
            operator,
            root,
            denied=report.denied,
            stopped=report.stopped,
        )
        if not released:
            await asyncio.to_thread(_audit_unattached, key, named, operator, report)
        await _notify_released(app, key, released, "stop")
        return {"released": list(released), "stop_unconfirmed": []}
    pairs = await asyncio.to_thread(_mark_unconfirmed, key, flag_ids, named, operator, report, root)
    notify = make_notifier(app)
    for flag_id, source in pairs:
        await notify(
            {
                "type": "watchdog_flag",
                "session_key": key,
                "flag_id": flag_id,
                "watchdog": source,
            }
        )
    return {"released": [], "stop_unconfirmed": [flag_id for flag_id, _ in pairs]}


async def _notify_released(
    app: Any, key: str, released: tuple[str, ...], answer: Literal["stop", "continue"]
) -> None:
    if released:
        await make_notifier(app)(
            {
                "type": "watchdog_resolved",
                "session_key": key,
                "flag_ids": list(released),
                "answer": answer,
            }
        )


@router.post("/api/watchdog/flags/answer")
async def post_answer(body: FlagAnswer, request: Request) -> dict[str, list[str]]:
    """Release exactly the named flags. Any other open flag keeps the session held."""
    app = request.app
    root = session_hold_root_of(app)
    if root is None:
        return {"released": [], "stop_unconfirmed": []}
    key = _checked_key(body.session_key)
    operator = getattr(request.state, "osprey_owner", None) or UNATTRIBUTED_OPERATOR
    try:
        if body.answer == "stop":
            return await _answer_stop(app, key, body.flag_ids, operator, root)
        released = await asyncio.to_thread(
            _release, key, body.flag_ids, body.answer, operator, root
        )
    except (OSError, RuntimeError) as exc:
        logger.error("Watchdog answer not written: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="hold_record_unavailable") from exc
    await _notify_released(app, key, released, body.answer)
    return {"released": list(released), "stop_unconfirmed": []}
