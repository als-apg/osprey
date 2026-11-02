"""What a watchdog decided about one prompt, and the audit entry each decision leaves.

A watchdog reads the session a prompt arrived in and answers with a
:class:`Verdict`: flagged or not, or unavailable when it could not decide. Every
verdict, and every operator answer to a flag it raised, is filed as one entry
on the ``safety_watchdog`` audit surface through :func:`osprey.audit.writer.record`:

* :func:`record_verdict` — decision ``flagged``, ``clear`` or ``unavailable``;
* :func:`record_answer` — decision ``stop`` or ``continue``, with the operator
  who answered (``'<unattributed>'`` when the answering surface knows none),
  what a Stop denied and stopped, and on a Continue after an unconfirmed Stop
  the commands still running;
* :func:`record_stop_unconfirmed` — decision ``stop_unconfirmed``, for a Stop
  whose turn did not end within the bound: the flag stays open and the entry
  names the started commands.

Each is written inside ``call_scope(None, record_session_id)``, so the entry is
joined on the Claude record session the prompt belongs to, whichever process
writes it — a detached hook child, the web terminal, a CLI run — and whatever
conversation ids that process's environment does or does not carry.

The vocabulary is deliberate: a watchdog that found nothing reports ``clear``,
never "safe". Nothing a watchdog says is a statement that an action is safe —
the thresholds are uncalibrated and the judges see only the prompt-time
session — so no decision word, field name or rendered detail says so. (The
surface name says ``safety``: the feature's name, not a judgement.)

``detail`` carries identifiers only (``prompt_ref``, ids, numbers, the closed
vocabulary): never the prompt, the judge's prose, or an agent message.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "ANSWER_CONTINUE",
    "ANSWER_STOP",
    "DECISION_CLEAR",
    "DECISION_CONTINUE",
    "DECISION_FLAGGED",
    "DECISION_STOP",
    "DECISION_STOP_UNCONFIRMED",
    "DECISION_UNAVAILABLE",
    "REASON_AFTER_STOP_UNCONFIRMED",
    "SURFACE_SAFETY_WATCHDOG",
    "UNATTRIBUTED_OPERATOR",
    "UNBOUND_ID",
    "Engine",
    "ErrorType",
    "InputDigest",
    "RefusedToolCall",
    "Verdict",
    "make_prompt_ref",
    "record_answer",
    "record_stop_unconfirmed",
    "record_verdict",
    "verdict_decision",
]

#: The audit surface every watchdog entry is filed on.
SURFACE_SAFETY_WATCHDOG = "safety_watchdog"

#: Decision words, one per kind of entry. ``clear`` is what a watchdog that
#: found nothing reports; there is deliberately no word claiming safety.
DECISION_FLAGGED = "flagged"
DECISION_CLEAR = "clear"
DECISION_UNAVAILABLE = "unavailable"
DECISION_STOP = "stop"
DECISION_CONTINUE = "continue"
DECISION_STOP_UNCONFIRMED = "stop_unconfirmed"

#: The two operator answers to a flag, as the answer route spells them.
ANSWER_STOP = DECISION_STOP
ANSWER_CONTINUE = DECISION_CONTINUE

#: The ``reason`` of a Continue on a flag an earlier Stop could not confirm.
REASON_AFTER_STOP_UNCONFIRMED = "after_stop_unconfirmed"

#: Who answered when the answering surface could name nobody.
UNATTRIBUTED_OPERATOR = "<unattributed>"

#: Stands in for the ids of a judgement not tied to a live prompt, such as a replay.
UNBOUND_ID = "unbound"

#: Which engine produced a verdict.
Engine = Literal["decision", "agent"]

#: Why a verdict is ``unavailable``. Closed: the audit ``reason`` of an
#: unavailable verdict is one of these.
ErrorType = Literal[
    "timeout",
    "budget",
    "in_flight_bound",
    "intake_lost",
    "no_probabilities",
    "decision_error",
    "missing_key",
    "parse_error",
    "run_error",
    "intake_failed",
]

#: The shape a judge's ``reason`` must have to be carried into the audit
#: ``reason`` field: an identifier (an option id, say), not prose. Anything
#: else is replaced there by the decision word.
_IDENTIFIER_REASON = re.compile(r"\A[A-Za-z0-9_.:-]{1,64}\Z")

#: The word no audit value may carry (see the module docstring): ``safe`` not
#: continued by another letter, so ``safe``, ``unsafe`` and ``safe_mode`` match
#: while the feature's own name, ``safety``, does not.
_FORBIDDEN_WORD = re.compile(r"safe(?![a-z])", re.IGNORECASE)


class InputDigest(BaseModel):
    """What the judge was shown, without showing it.

    :param sha256: Hex SHA-256 of the session view the judge read.
    :param entries_kept: Session entries in the view.
    :param entries_cut: Session entries left out to fit the judge's budget.
    :param tokens_estimate: The view's estimated size in tokens.
    :param drain_timed_out: The record read stopped at the deadline.
    :param record_available: The record could be read at all (``False`` →
        the judge saw the prompt only).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = ""
    entries_kept: int = 0
    entries_cut: int = 0
    tokens_estimate: int = 0
    drain_timed_out: bool = False
    record_available: bool = True


class RefusedToolCall(BaseModel):
    """A tool call an agent-engine judge attempted and was refused.

    :param tool: The tool name.
    :param reason: The refusal reason, an identifier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str
    reason: str


class Verdict(BaseModel):
    """One watchdog's answer about one prompt.

    :param watchdog: The watchdog's name (its profile map key).
    :param engine: ``decision`` or ``agent``.
    :param session_key: The web terminal pool key, ``None`` for a session with
        no web terminal (a bare ``claude``: record-only, never held).
    :param record_session_id: The Claude record session the prompt belongs to.
    :param submission_id: The id the trigger minted for this prompt.
    :param prompt_ref: ``<record_session_id>:<submission_id>``; joins verdict,
        flag and answer records. Derived when omitted.
    :param status: ``ok`` when the judge decided, ``unavailable`` when it could
        not (then *error_type* says why and *flagged* is ``False``).
    :param flagged: The judge found the prompt risky.
    :param reason: The judge's short reason — an option id where it has one.
    :param score: The judge's probability for the flag, where it has one.
    :param threshold: The score at or above which a decision watchdog flags;
        required for the decision engine.
    :param uncalibrated: The threshold has not been calibrated against
        labelled traffic. ``True`` until a calibration says otherwise.
    :param input_digest: What the judge was shown (see :class:`InputDigest`).
    :param refused_tool_calls: Tool calls an agent judge was refused.
    :param latency_s: Wall time the judge took.
    :param cost_usd: What the judge cost, where known.
    :param error_type: Why the verdict is unavailable; ``None`` when ``ok``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    watchdog: str = Field(min_length=1)
    engine: Engine
    session_key: str | None = None
    record_session_id: str = Field(min_length=1)
    submission_id: str = Field(min_length=1)
    prompt_ref: str = ""
    status: Literal["ok", "unavailable"]
    flagged: bool = False
    reason: str | None = None
    score: float | None = None
    threshold: float | None = None
    uncalibrated: bool = True
    input_digest: InputDigest = Field(default_factory=InputDigest)
    refused_tool_calls: list[RefusedToolCall] = Field(default_factory=list)
    latency_s: float = 0.0
    cost_usd: float | None = None
    error_type: ErrorType | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_prompt_ref(cls, data: object) -> object:
        """Fill ``prompt_ref`` from the two ids when the caller left it out."""
        if isinstance(data, dict) and not data.get("prompt_ref"):
            rid, sid = data.get("record_session_id"), data.get("submission_id")
            if isinstance(rid, str) and isinstance(sid, str):
                data = {**data, "prompt_ref": make_prompt_ref(rid, sid)}
        return data

    @model_validator(mode="after")
    def _consistent(self) -> Verdict:
        """Refuse a verdict whose fields contradict each other."""
        if self.status == "unavailable":
            if self.error_type is None:
                raise ValueError("an unavailable verdict names its error_type")
            if self.flagged:
                raise ValueError("an unavailable verdict is not flagged")
        elif self.error_type is not None:
            raise ValueError("an ok verdict carries no error_type")
        if self.engine == "decision" and self.threshold is None:
            raise ValueError("a decision verdict carries its threshold")
        if self.prompt_ref != make_prompt_ref(self.record_session_id, self.submission_id):
            raise ValueError("prompt_ref must be <record_session_id>:<submission_id>")
        return self


def make_prompt_ref(record_session_id: str, submission_id: str) -> str:
    """``<record_session_id>:<submission_id>`` — the key that joins verdict, flag and answer."""
    return f"{record_session_id}:{submission_id}"


def verdict_decision(verdict: Verdict) -> str:
    """The audit decision word for *verdict*: ``unavailable``, ``flagged`` or ``clear``."""
    if verdict.status == "unavailable":
        return DECISION_UNAVAILABLE
    return DECISION_FLAGGED if verdict.flagged else DECISION_CLEAR


# --------------------------------------------------------------------------
# Rendering the detail
# --------------------------------------------------------------------------


def _ids(values: Sequence[str]) -> str:
    """A list of identifiers as compact JSON, so a comma or space inside one stays unambiguous."""
    return json.dumps([str(value) for value in values], separators=(",", ":"))


def _number(value: float) -> str:
    return f"{value:.4g}"


def _scrub(text: str) -> str:
    """*text* with the forbidden word masked — the last guard on what is rendered.

    Identifiers are not ours to choose (a watchdog name, a command), so a value
    that happens to contain the word is masked rather than dropped: the record
    keeps the entry, and the entry never says "safe".
    """
    return _FORBIDDEN_WORD.sub("s*fe", text)


def _verdict_detail(verdict: Verdict) -> str:
    """Identifier-only detail for one verdict, ``key=value`` tokens separated by spaces."""
    digest = verdict.input_digest
    tokens = [
        f"prompt_ref={verdict.prompt_ref}",
        f"engine={verdict.engine}",
        f"status={verdict.status}",
    ]
    if verdict.score is not None:
        tokens.append(f"score={_number(verdict.score)}")
    if verdict.threshold is not None:
        tokens.append(f"threshold={_number(verdict.threshold)}")
    tokens.append(f"uncalibrated={'true' if verdict.uncalibrated else 'false'}")
    if verdict.error_type is not None:
        tokens.append(f"error_type={verdict.error_type}")
    tokens.extend(
        [
            f"latency_s={_number(verdict.latency_s)}",
            f"entries_kept={digest.entries_kept}",
            f"entries_cut={digest.entries_cut}",
            f"tokens_estimate={digest.tokens_estimate}",
            f"drain_timed_out={'true' if digest.drain_timed_out else 'false'}",
            f"record_available={'true' if digest.record_available else 'false'}",
        ]
    )
    if digest.sha256:
        tokens.append(f"input_sha256={digest.sha256}")
    if verdict.cost_usd is not None:
        tokens.append(f"cost_usd={_number(verdict.cost_usd)}")
    if verdict.refused_tool_calls:
        tokens.append(f"refused_tools={_ids([call.tool for call in verdict.refused_tool_calls])}")
    return _scrub(" ".join(tokens))


def _verdict_reason(verdict: Verdict) -> str:
    """``error_type`` when unavailable; else the judge's reason when identifier-shaped, else the decision."""
    if verdict.error_type is not None:
        return verdict.error_type
    reason = verdict.reason
    if (
        isinstance(reason, str)
        and _IDENTIFIER_REASON.match(reason)
        and not _FORBIDDEN_WORD.search(reason)
    ):
        return reason
    return verdict_decision(verdict)


def _operator(operator: str | None) -> str:
    return operator if isinstance(operator, str) and operator.strip() else UNATTRIBUTED_OPERATOR


def _answer_detail(
    *,
    prompt_ref: str,
    flag_id: str,
    operator: str,
    denied: Sequence[str],
    stopped: Sequence[str],
    started: Sequence[str],
    after_stop_unconfirmed: bool,
) -> str:
    tokens = [f"prompt_ref={prompt_ref}", f"flag_id={flag_id}", f"operator={operator}"]
    if denied:
        tokens.append(f"denied={_ids(denied)}")
    if stopped:
        tokens.append(f"stopped={_ids(stopped)}")
    if after_stop_unconfirmed:
        tokens.append("after_stop_unconfirmed=true")
    if started:
        tokens.append(f"started={_ids(started)}")
    return _scrub(" ".join(tokens))


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _write(
    *,
    record_session_id: str,
    session_key: str | None,
    watchdog: str,
    decision: str,
    reason: str,
    detail: str,
    on_behalf_of: str | None = None,
    posture_source: str | None = None,
) -> str | None:
    """One ``safety_watchdog`` envelope, joined on *record_session_id*; its entry id or ``None``.

    Never raises: an audit entry that cannot be written costs the record one
    line, never the watchdog run or the operator's answer.
    """
    try:
        from osprey.audit import posture, writer
        from osprey.audit.call import call_scope

        with call_scope(None, record_session_id or None):
            return writer.record(
                surface=SURFACE_SAFETY_WATCHDOG,
                posture=posture.posture(),
                posture_source=posture.posture_source(posture_source),
                session=session_key or record_session_id or None,
                subject=_scrub(watchdog),
                decision=decision,
                reason=_scrub(reason),
                detail=detail,
                on_behalf_of=on_behalf_of,
            )
    except Exception:
        logger.warning("Could not record a %s watchdog entry", decision, exc_info=True)
        return None


def record_verdict(verdict: Verdict, *, posture_source: str | None = None) -> str | None:
    """File *verdict* as one audit entry; its entry id, or ``None`` when not stored.

    Decision ``flagged``/``clear``/``unavailable``; subject the watchdog;
    ``reason`` the error type or the verdict; ``detail`` the ``prompt_ref``,
    threshold, ``uncalibrated``, score and input digest. *posture_source* is the
    writing surface's own answer where it has one (see
    :func:`osprey.audit.posture.posture_source`). Never raises.
    """
    try:
        return _write(
            record_session_id=verdict.record_session_id,
            session_key=verdict.session_key,
            watchdog=verdict.watchdog,
            decision=verdict_decision(verdict),
            reason=_verdict_reason(verdict),
            detail=_verdict_detail(verdict),
            posture_source=posture_source,
        )
    except Exception:
        logger.warning("Could not record a watchdog verdict", exc_info=True)
        return None


def _record_session_from(prompt_ref: str) -> str:
    return prompt_ref.split(":", 1)[0] if isinstance(prompt_ref, str) else ""


def _write_operator_entry(
    *,
    decision: str,
    reason: str,
    watchdog: str,
    session_key: str | None,
    prompt_ref: str,
    flag_id: str,
    operator: str | None,
    record_session_id: str | None,
    denied: Sequence[str],
    stopped: Sequence[str],
    started: Sequence[str],
    after_stop_unconfirmed: bool,
    posture_source: str | None,
) -> str | None:
    """One entry for what an operator did about one flag; empty *operator* is unattributed."""
    who = _operator(operator)
    return _write(
        record_session_id=record_session_id or _record_session_from(prompt_ref),
        session_key=session_key,
        watchdog=watchdog,
        decision=decision,
        reason=reason,
        detail=_answer_detail(
            prompt_ref=prompt_ref,
            flag_id=flag_id,
            operator=who,
            denied=denied,
            stopped=stopped,
            started=started,
            after_stop_unconfirmed=after_stop_unconfirmed,
        ),
        on_behalf_of=None if who == UNATTRIBUTED_OPERATOR else who,
        posture_source=posture_source,
    )


def record_answer(
    *,
    watchdog: str,
    session_key: str | None,
    prompt_ref: str,
    flag_id: str,
    answer: Literal["stop", "continue"],
    operator: str | None,
    record_session_id: str | None = None,
    denied: Sequence[str] = (),
    stopped: Sequence[str] = (),
    after_stop_unconfirmed: bool = False,
    started: Sequence[str] = (),
    posture_source: str | None = None,
) -> str | None:
    """File one operator answer to one flag; its entry id, or ``None`` when not stored.

    Decision ``stop`` or ``continue``; ``reason`` the answer, or
    ``after_stop_unconfirmed`` for a Continue on a flag an earlier Stop could
    not confirm, whose *started* commands the detail then lists. A Stop names
    what it *denied* (pending approvals) and *stopped* (background agents).
    *operator* is who answered; empty or ``None`` is recorded as
    ``'<unattributed>'``. *record_session_id* defaults to the first half of
    *prompt_ref*. Never raises.
    """
    try:
        if answer not in (ANSWER_STOP, ANSWER_CONTINUE):
            raise ValueError(f"answer must be stop or continue, got {answer!r}")
        late = bool(after_stop_unconfirmed) and answer == ANSWER_CONTINUE
        return _write_operator_entry(
            decision=answer,
            reason=REASON_AFTER_STOP_UNCONFIRMED if late else answer,
            watchdog=watchdog,
            session_key=session_key,
            prompt_ref=prompt_ref,
            flag_id=flag_id,
            operator=operator,
            record_session_id=record_session_id,
            denied=denied,
            stopped=stopped,
            started=started,
            after_stop_unconfirmed=late,
            posture_source=posture_source,
        )
    except Exception:
        logger.warning("Could not record a watchdog answer", exc_info=True)
        return None


def record_stop_unconfirmed(
    *,
    watchdog: str,
    session_key: str | None,
    prompt_ref: str,
    flag_id: str,
    operator: str | None,
    started: Sequence[str] = (),
    record_session_id: str | None = None,
    denied: Sequence[str] = (),
    stopped: Sequence[str] = (),
    posture_source: str | None = None,
) -> str | None:
    """File a Stop whose turn did not end within the bound; its entry id, or ``None``.

    Decision and reason ``stop_unconfirmed``; the flag stays open. The detail
    names the *started* commands still running, and what the Stop *denied* and
    *stopped* before giving up. Never raises.
    """
    try:
        return _write_operator_entry(
            decision=DECISION_STOP_UNCONFIRMED,
            reason=DECISION_STOP_UNCONFIRMED,
            watchdog=watchdog,
            session_key=session_key,
            prompt_ref=prompt_ref,
            flag_id=flag_id,
            operator=operator,
            record_session_id=record_session_id,
            denied=denied,
            stopped=stopped,
            started=started,
            after_stop_unconfirmed=False,
            posture_source=posture_source,
        )
    except Exception:
        logger.warning("Could not record an unconfirmed watchdog stop", exc_info=True)
        return None
