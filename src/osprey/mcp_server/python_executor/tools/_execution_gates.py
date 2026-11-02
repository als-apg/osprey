"""Shared execution-mode gates for the ``execute`` and ``execute_file`` tools.

Both tools take an ``execution_mode`` string and guard control-system writes
with two independent checks: a per-call readonly gate (pattern detection) and
the write verdict for the control target the run will launch on
(:func:`enforce_posture_clamp`). Each gate only recognises one canonical
spelling, so any *other* string falls through both — not "readonly", so write
patterns are not blocked; not "readwrite", so the verdict is never asked.
Rejecting unknown modes here closes that hole for every caller at once. The
verdict is :func:`osprey_connectors.posture_store.execution_verdict`, the write
verdict the connector's reference monitor asks at the moment of every write plus
the session's hold, and its
refusal is spoken in that verdict's words
(:func:`osprey_connectors.refusals.refusal_phrase`).

This module also owns what happens *around* a refusal, which is three things
the tools must not each spell for themselves: the durable audit record, the
operator alert, and the error handed back to the agent. Keeping them together
is what makes "blocked" mean the same thing at every layer — a write stopped by
the import denylist before launch and one stopped by the runtime guard mid-run
produce the same audit record and the same alert, differing only in the layer
that refused — which the record carries as its ``reason``.

Every refusal here files into the **unified** ledger
(:mod:`osprey.audit.writer`), on the ``executor`` surface, and every one of
them is an *inner* recorder: they run inside the MCP audit middleware, which
wraps every ``tools/call``. So each marks the decision as its own
(:mod:`osprey.audit.dedup`) and the middleware defers instead of filing a
second record for the same refusal. That matters most for the runtime guard,
which reports a refusal the subprocess made and then hands back a *successful*
result: without the marker the middleware would see a clean call and stamp
``allowed`` over a write it stopped. The mark is only visible to a layer that
awaited this call on the same task, which is why both recorders below are
reached inline from the ``async def`` tool bodies and never through a thread or
a spawned task.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn

from osprey.audit import posture
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async
from osprey.mcp_server.python_executor import executor
from osprey.utils.session_hold_api import execution_verdict
from osprey_connectors.posture_store import WriteVerdict
from osprey_connectors.types import baseline_target

if TYPE_CHECKING:
    from osprey.mcp_server.python_executor.executor import ExecutionResult
    from osprey.services.code_review.contract import ReviewRequest
    from osprey.services.code_review.review import ReviewOutcome

logger = logging.getLogger("osprey.mcp_server.tools.execution_gates")

#: The closed set of recognised execution modes. Downstream gates may test
#: equality against either member only because this set is enforced first.
VALID_EXECUTION_MODES = frozenset({"readonly", "readwrite"})

#: The posture, its provenance and its audit session id —
#: :mod:`osprey.audit.posture`'s spellings, re-exported under this module's
#: names because the tests and the gates' callers address them here.
POSTURE_ENV_VAR = posture.POSTURE_ENV_VAR
SANDBOX_POSTURE = posture.SANDBOX_MODE
POSTURE_SOURCE_ENV_VAR = posture.POSTURE_SOURCE_ENV_VAR
POSTURE_SESSION_ENV_VAR = posture.POSTURE_SESSION_ENV_VAR

#: This server's rendered name, for composing the ``mcp__<name>__<tool>``
#: subject the MCP middleware would have used for the same call — so the two
#: layers name the same thing and a reader can join them.
TOOL_PREFIX_ENV_VAR = "OSPREY_MCP_TOOL_PREFIX"

# `TOOL_PREFIX_ENV_VAR` above is re-spelled rather than imported from
# `osprey.mcp_server.audit_middleware`: this module is the *inner* recorder,
# and importing the outer layer would invert the dependency and pull fastmcp's
# middleware machinery into every executor tool.
# `tests/audit/test_dedup_contract.py` pins it against the middleware's, the
# way the middleware pins the registry's markers it cannot import.

#: How the ledger spells the two postures — see :mod:`osprey.audit.posture`.
POSTURE_SANDBOX = posture.POSTURE_SANDBOX
POSTURE_WRITES = posture.POSTURE_WRITES

#: The reason a posture refusal is filed under. The same spelling the
#: MCP middleware and the client-side ``osprey_writes_check.py`` hook use for
#: the same refusal: one refusal, three possible layers, and an operator
#: querying the ledger for it should not have to know which. A cross-layer test
#: pins all three (the hook by AST, since it imports nothing from ``osprey``).
REASON_POSTURE = "posture"

#: The layer that refused a control-system write, filed as the record's
#: ``reason``. Static layers refuse before the script is launched;
#: ``runtime_guard`` is the one that fires inside the subprocess, so it is the
#: layer that catches the spellings the static ones cannot see.
#: ``raw_client_write`` also fires inside the subprocess, in readwrite runs: the
#: armed raw-put block refusing a direct client-library write that bypassed the
#: connector's reference monitor. Spelled here,
#: beside the recorder, because this module is where every one of them converges
#: -- they used to live in the retired P1 ledger module, which was their only
#: other shared home.
LAYER_IMPORT_DENYLIST = "import_denylist"
LAYER_PATTERN_DETECTION = "pattern_detection"
LAYER_PATH_POLICY = "path_policy"
LAYER_RUNTIME_GUARD = "runtime_guard"
LAYER_RAW_CLIENT_WRITE = "raw_client_write"

#: Characters of the matched trigger, and of the agent's own description, kept
#: inside the record's ``detail``. Bounded here rather than left to the
#: envelope's silent ``detail`` truncation so that the trigger -- what actually
#: matched -- cannot be pushed out by a long description. The refused code
#: itself rides in ``source``, whole up to the envelope's own bound: on this
#: surface the code *is* the artifact under audit.
MAX_TRIGGER_CHARS = 400
MAX_DESCRIPTION_CHARS = 200


def require_known_execution_mode(execution_mode: str) -> None:
    """Raise ``ToolError`` (validation_error) unless the mode is recognised.

    Must run before any write gate: the gates branch on string equality, and
    an unrecognised value would otherwise satisfy neither branch and execute
    with no write protection at all.
    """
    if execution_mode in VALID_EXECUTION_MODES:
        return
    make_error(
        "validation_error",
        f"Unknown execution_mode {execution_mode!r}.",
        ['Use "readonly" (default) to block control-system writes, or "readwrite" to allow them.'],
    )


def write_activity_detail(has_writes: bool, execution_mode: str) -> str | None:
    """The Web Terminal write-activity detail for a launched run, or ``None``.

    A detected write pattern is reported in any mode but readonly. A
    read-write run is reported even when the scan found no write: a library
    call such as a pyAML ``strengths.set(...)`` writes through the runtime
    without matching any write pattern.

    Args:
        has_writes: Whether pattern detection found a control-system write.
        execution_mode: The run's execution mode.

    Returns:
        The detail to emit, or ``None`` when the run reports no write activity.
    """
    if execution_mode == "readonly":
        return None
    if has_writes:
        return "ran a script with control-system writes"
    if execution_mode == "readwrite":
        return "ran a script in read-write mode"
    return None


def _launch_target(section: Any) -> str | None:
    """The control target a run launched now would be about, or ``None``.

    The deployment's control-context record names it — read through the
    executor's own reader, so this gate and the launch's stamp
    (:func:`~osprey.mcp_server.python_executor.executor._apply_target_stamp`)
    cannot read two records. A stamp this process inherited is not consulted:
    the launch drops it, so a gate that honoured it would answer for a machine
    the run is never pointed at. With no record the run is unstamped and builds
    the deployment's baseline connector, so the question is about
    :func:`~osprey_connectors.types.baseline_target` — and with no record, every
    switch is off.

    A record naming a target this deployment cannot build is answered for that
    target, which the verdict refuses as ``target_unknown``: the launch would
    pin such a run sandboxed everywhere, and the refusal says why.

    ``None`` (the verdict's ``target_unknown``) when not even the baseline can
    be named: a ``control_system`` section that names no serving the
    deployment knows. The record's own reader never raises.
    """
    record = executor._control_context_record()
    if record is not None:
        return str(record.target)
    try:
        return baseline_target(section)
    except ValueError:  # an unnameable target is refused by the verdict, never granted
        logger.warning("Could not name the run's control target; refusing writes", exc_info=True)
        return None


def _launch_verdict() -> WriteVerdict:
    """The write verdict for the target a run launched now would write to.

    :func:`osprey_connectors.posture_store.execution_verdict` over the deployment's
    ``control_system`` section (:func:`osprey.audit.posture.control_system_section`)
    and :func:`_launch_target`, with the owner left to the verdict's own
    ladder. Never raises: a verdict that cannot be reached refuses as
    ``record_unavailable``.
    """
    try:
        section = posture.control_system_section()
        return execution_verdict(section, _launch_target(section))
    except Exception:  # execution_verdict never raises by contract; refuse if it does
        logger.warning("Write verdict could not be reached; refusing", exc_info=True)
        return WriteVerdict(allowed=False, reason="record_unavailable", target=None, kind=None)


def _tool_subject(tool: str) -> str:
    """The ``mcp__<prefix>__<tool>`` name this refusal is about.

    Composed exactly as the MCP audit middleware composes it, so the record the
    clamp files and the records the middleware files for neighbouring calls
    name the same subject. With no prefix — a server not launched by ``osprey``
    — the bare tool name, which is what the middleware falls back to too.
    """
    prefix = (os.environ.get(TOOL_PREFIX_ENV_VAR) or "").strip()
    return f"mcp__{prefix}__{tool}" if prefix else tool


def _refusal_detail(
    tool: str,
    execution_mode: str,
    trigger: Any,
    description: str | None,
    ca_user: str | None = None,
) -> str:
    """The supplementary context one refused write carries.

    ``tool`` and ``mode`` first: they are the two an operator scans a ledger
    for. ``ca_user``, when stamped, follows: it is an identifier, so it goes
    ahead of the bounded fields rather than risk being cut. The trigger and the
    agent's description follow, each bounded, because both are as long as
    whatever produced them.
    """
    parts = [f"tool={tool}", f"mode={execution_mode}"]
    if ca_user:
        parts.append(f"ca_user={ca_user}")
    parts.append(f"trigger={trigger}"[:MAX_TRIGGER_CHARS])
    if description:
        parts.append(f"description={description}"[:MAX_DESCRIPTION_CHARS])
    return " ".join(parts)


def _verdict_detail(verdict: WriteVerdict) -> str:
    """``reason=<r> target=<t>`` for one verdict refusal; no ``target`` when unnamed."""
    parts = [f"reason={verdict.reason}"]
    if verdict.target:
        parts.append(f"target={verdict.target}")
    return " ".join(parts)


def _record_posture_clamp(tool: str, verdict: WriteVerdict) -> None:
    """File the verdict's refusal, and claim it for this layer.

    Filed under :data:`REASON_POSTURE` with detail ``reason=<r> target=<t>`` —
    the verdict's reason and the target it was asked about, the same words
    the audit middleware's backstop files — so a ledger query for one refusal
    reads alike at every layer. The target is left off when none was named.

    Goes through :func:`~osprey.audit.dedup.record_and_mark` rather than the
    writer directly: this gate runs *inside* the MCP audit middleware, which
    would otherwise file a second record for the same refusal when the
    ``ToolError`` below reaches it. The marker carries the decision, so the
    middleware defers to a specific answer instead of merely staying quiet.

    No ``source``: the clamp fires before the code is read, so there is no
    offending artifact yet — only a session that may not write at all. Never
    raises; the writer swallows its own errors and the import is the only thing
    left that could, so it is guarded too. A refusal that could not be recorded
    is still a refusal.
    """
    try:
        from osprey.audit.dedup import record_and_mark
        from osprey.audit.envelope import DECISION_REFUSED, SURFACE_EXECUTOR

        record_and_mark(
            decision=DECISION_REFUSED,
            reason=REASON_POSTURE,
            surface=SURFACE_EXECUTOR,
            posture=POSTURE_SANDBOX,
            posture_source=posture.posture_source(),
            session=posture.posture_session(),
            subject=_tool_subject(tool),
            detail=_verdict_detail(verdict),
        )
    except Exception:  # the audit trail degrades; the refusal does not
        logger.warning("Could not record the posture refusal", exc_info=True)


def enforce_posture_clamp(execution_mode: str, *, tool: str) -> None:
    """Raise ``ToolError`` (safety_error) on readwrite runs the write verdict refuses.

    The verdict is asked about the control target the run would launch on
    (:func:`_launch_target`): the record's target, or the deployment's baseline
    when there is no record. Its terms decide in their one order — a read-only
    run, a tier that allows nothing, an unknown target or owner, an unreadable
    record, a switch that is off — and the refusal is that reason's phrase
    (:func:`osprey_connectors.refusals.refusal_phrase`), naming the machine's
    kind. The phrase says who can lift the refusal and names no config key.

    This gate is admission only. The run's launch pin is the same verdict with
    the pin ignored (:func:`~osprey.mcp_server.python_executor.executor._launch_posture`),
    and the connector's reference monitor asks it again on every write, so a
    switch turned off reaches a run that is already going at its next write.

    A refusal is recorded in the unified ledger on the ``executor`` surface and
    marked as this layer's own — see :func:`_record_posture_clamp`. Nothing is
    recorded when the gate does not fire: a readonly run and an allowed one are
    both ordinary, and a ledger that logged them would bury the refusals it
    exists for.

    Args:
        execution_mode: The mode this call asked for.
        tool: The tool name as the agent knows it, for the recorded subject.
            Required and keyword-only: a default would let a new call site
            record a refusal under another tool's name, and a positional would
            let it swap the two strings silently.
    """
    if execution_mode != "readwrite":
        return
    verdict = _launch_verdict()
    if verdict.allowed:
        return

    _record_posture_clamp(tool, verdict)

    from osprey_connectors.refusals import refusal_text

    reason = verdict.reason or "record_unavailable"
    make_error(
        "safety_error",
        refusal_text(reason, verdict.kind, detail=verdict.detail),
        ['Re-run with execution_mode="readonly" — reads are unaffected.'],
        details={"reason": reason, "target": verdict.target},
    )


def _record_write_refusal(
    *,
    tool: str,
    layer: str,
    trigger: Any,
    code: str,
    description: str | None,
    execution_mode: str,
    ca_user: str | None = None,
) -> None:
    """File one refused control-system write, and claim it for this layer.

    Goes through :func:`~osprey.audit.dedup.record_and_mark` for the reason the
    module docstring gives: this runs inside the MCP audit middleware, and the
    runtime-guard case hands back a *successful* result, so without the mark
    the middleware would file ``allowed`` on top of a refusal.

    The refused code goes in ``source``, which the ledger keeps whole on this
    surface -- a refusal whose source is not kept is an alert, not an audit
    trail. The ``layer`` becomes the record's ``reason``, so the four layers
    stay one query apart, exactly as they were when they shared a ``layer``
    field in the retired ledger.

    Never raises: the writer swallows its own errors, and the lazy import is
    the only thing left that could, so it is guarded too. A refusal that could
    not be recorded is still a refusal.
    """
    try:
        from osprey.audit.dedup import record_and_mark
        from osprey.audit.envelope import DECISION_REFUSED, SURFACE_EXECUTOR

        record_and_mark(
            decision=DECISION_REFUSED,
            reason=layer,
            surface=SURFACE_EXECUTOR,
            posture=posture.posture(),
            posture_source=posture.posture_source(),
            session=posture.posture_session(),
            subject=_tool_subject(tool),
            source=code,
            detail=_refusal_detail(tool, execution_mode, trigger, description, ca_user),
        )
    except Exception:  # the audit trail degrades; the refusal does not
        logger.warning("Could not record the refusal for audit", exc_info=True)


async def record_and_alert_refusal(
    *,
    tool: str,
    layer: str,
    trigger: Any,
    code: str,
    description: str | None = None,
    execution_mode: str,
    ca_user: str | None = None,
) -> None:
    """Write the audit record and alert the operator for one refused write.

    Both halves of the issue's "the operator should see it, and it should be
    auditable" requirement, in the order that matters: the durable record is
    written first, so a Web Terminal that is not running (CLI-only mode, where
    the alert is a no-op) still leaves the refusal on disk.

    The record is filed as this layer's own (:func:`_record_write_refusal`), so
    the MCP audit middleware defers to it. Every caller reaches here awaited
    inline from an ``async def`` tool body, which is what makes the mark
    visible to the middleware at all.

    ``execution_mode`` is required rather than defaulted: it is the mode the
    record states and the mode the operator alert names, and a default would
    make both an accident of which caller forgot to pass one. Every layer here
    refuses in *both* modes, so "readonly" is never a safe guess.

    Never raises. The recorder swallows its own errors and
    ``notify_agent_activity_async`` is fire-and-forget by contract, so a
    refusal is never turned into a traceback by the act of reporting it.

    ``ca_user``, when given, is stamped into the record's ``detail``: the
    account the refused write would have gone out as.
    """
    _record_write_refusal(
        tool=tool,
        layer=layer,
        trigger=trigger,
        code=code,
        description=description,
        execution_mode=execution_mode,
        ca_user=ca_user,
    )

    await notify_agent_activity_async(
        tool,
        "channel",
        detail=f"BLOCKED a control-system write in {execution_mode} mode ({layer})",
    )


async def refuse_readonly_write(
    *,
    tool: str,
    layer: str,
    trigger: Any,
    code: str,
    description: str | None,
    message: str,
    suggestions: list[str],
    execution_mode: str = "readonly",
) -> NoReturn:
    """Record, alert, then raise the ``safety_error`` the agent sees.

    The raise is last on purpose: :func:`make_error` raises rather than
    returning (it is the only path fastmcp turns into a clean error on the
    wire), so anything that has to happen for a refused write has to happen
    before it.

    ``execution_mode`` is recorded verbatim in the audit record — layers that
    refuse in every mode (the path policy) pass the run's real mode so a
    readwrite refusal is never logged as a readonly one.
    """
    await record_and_alert_refusal(
        tool=tool,
        layer=layer,
        trigger=trigger,
        code=code,
        description=description,
        execution_mode=execution_mode,
    )
    make_error("safety_error", message, suggestions)


async def enforce_path_policy(
    *,
    tool: str,
    code: str,
    description: str | None,
    execution_mode: str,
    project_root: Path | None = None,
) -> None:
    """Refuse *code* if it statically writes into the protected set.

    Deliberately *not* under the readonly branch. The render zone, the profile
    sources and the audit ledger are off limits to executed code in every mode,
    because the boundary here is the agent rewriting the configuration it is
    itself constrained by, which the write posture has nothing to say about.
    The roots are resolved parent-side and handed in; the walker never
    re-derives them.

    Both executor tools ask this question here rather than each spelling out
    the walk, the refusal and its suggestions, so the two cannot drift on what
    the protected set is or on what an agent is told when it hits one.

    Args:
        tool: Tool name as the agent knows it, recorded in the audit record.
        code: Source to walk.
        description: Caller's description, recorded with the refusal.
        execution_mode: The run's real mode. Recorded verbatim, and it selects
            the wording of one suggestion — both modes refuse, so the readwrite
            message must not read as "you are in readonly", or the agent would
            resubmit as readwrite and hit exactly the same refusal.
        project_root: Project root when the caller has already resolved one;
            ``None`` lets the resolvers derive it.

    Raises:
        Whatever :func:`make_error` raises for ``safety_error`` — the refusal
        the agent sees.
    """
    try:
        from osprey.mcp_server.python_executor.executor import (
            resolve_permitted_roots,
            resolve_protected_roots,
        )
        from osprey.services.python_executor.execution.path_policy import path_policy_issues

        path_issues = path_policy_issues(
            code,
            protected_roots=resolve_protected_roots(project_root),
            permitted_roots=resolve_permitted_roots(project_root),
        )
    except ImportError:
        logger.warning("Path policy module unavailable — skipping protected-path check")
        path_issues = []
    if not path_issues:
        return

    await refuse_readonly_write(
        tool=tool,
        layer=LAYER_PATH_POLICY,
        trigger=path_issues,
        code=code,
        description=description,
        execution_mode=execution_mode,
        message=(
            "This code writes into a location the deployment protects "
            "(the render zone, the profile sources, or the audit ledger). "
            "The path policy applies in every execution mode."
        ),
        suggestions=[
            *path_issues,
            "Write analysis output under the agent data zone instead.",
            (
                "Re-running as readwrite will not lift this: the protected set is "
                "independent of execution mode."
                if execution_mode == "readonly"
                else "The write posture permits control-system writes; it does not "
                "permit edits to OSPREY's own configuration."
            ),
            "Edit the profile sources yourself and re-run 'osprey build' to change them.",
        ],
    )


async def report_runtime_refusal(
    *,
    tool: str,
    stderr: str,
    code: str,
    description: str | None,
    execution_mode: str,
) -> bool:
    """Report a write refused *inside* the run, mid-run. Returns whether it did.

    The static layers refuse by raising, so they control their own reporting.
    The runtime ones cannot: they run inside the subprocess, and all that
    reaches here is a traceback on stderr. Without this, the layers that catch
    the evasive spellings — aliased imports, ``getattr``, ``importlib``, a
    shelled-out ``caput`` — would be the only ones that never alerted the
    operator or left an audit record.

    Two in-run layers refuse, one per mode, and each is recognised by its own
    marker rather than by its full message:

    * readonly — the runtime guard, matched on
      :data:`~osprey.services.python_executor.execution.wrapper.READONLY_REFUSAL_MARKER`,
      so the connector reference monitor's own refusal (a write that took the
      *approved* ``write_channel`` path in a readonly run) is reported too.
      Filed as :data:`LAYER_RUNTIME_GUARD`.
    * readwrite — the armed raw-put block, matched on
      :data:`~osprey_connectors.errors.RAW_CLIENT_WRITE_MARKER`: a direct
      client-library write that bypassed the reference monitor. Filed as
      :data:`LAYER_RAW_CLIENT_WRITE`, stamped with the ``ca_user`` this process
      runs as — the sandbox is its child, under the same account, so that is
      the account the refused write would have gone out as.

    A readwrite run still matches the readonly marker as well: a guard that
    emits it mid-``readwrite`` must be reported, and must not have the ledger
    and the operator alert call that run readonly.

    The script's own result is left alone: its stderr already names the mode
    and the way forward, and converting it into a tool error here would
    discard whatever the run legitimately produced before the refusal.
    """
    from osprey.audit.call import write_stamps
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER
    from osprey_connectors.errors import RAW_CLIENT_WRITE_MARKER

    stderr = stderr or ""
    ca_user: str | None = None
    if execution_mode == "readwrite" and RAW_CLIENT_WRITE_MARKER in stderr:
        layer, marker = LAYER_RAW_CLIENT_WRITE, RAW_CLIENT_WRITE_MARKER
        ca_user = write_stamps().get("ca_user")
    elif READONLY_REFUSAL_MARKER in stderr:
        layer, marker = LAYER_RUNTIME_GUARD, READONLY_REFUSAL_MARKER
    else:
        return False

    await record_and_alert_refusal(
        tool=tool,
        layer=layer,
        trigger=_refusal_lines(stderr, marker),
        code=code,
        description=description,
        execution_mode=execution_mode,
        ca_user=ca_user,
    )
    return True


def _refusal_lines(stderr: str, marker: str) -> list[str]:
    """The stderr lines naming the refusal, for the audit record's ``trigger``.

    The whole traceback would bury the fact in noise and the bare marker would
    drop the channel name the refusal message carries; the lines carrying
    *marker* keep what an auditor actually reads.
    """
    return [line.strip() for line in stderr.splitlines() if marker in line]


#: The subject of the pre-execution safety refusal, per tool. ``execute_file``
#: refuses a file the operator can open; every other tool refuses code it was
#: handed.
_SAFETY_SUBJECT = {"execute_file": "File"}


def _record_review(outcome: ReviewOutcome, request: ReviewRequest) -> None:
    """File one review's value-free record.

    A refusal goes through :func:`~osprey.audit.dedup.record_and_mark`, so the MCP audit
    middleware defers to it. A proceed goes through the writer, and the middleware still files
    the call. Never raises: the audit trail degrades, the outcome does not.
    """
    try:
        from osprey.audit import writer
        from osprey.audit.dedup import record_and_mark
        from osprey.audit.envelope import SURFACE_EXECUTOR
        from osprey.services.code_review.review import ACTION_REFUSE, audit_fields

        fields = audit_fields(outcome, request)
        file_record = record_and_mark if outcome.action == ACTION_REFUSE else writer.record
        file_record(
            decision=fields.decision,
            reason=fields.reason,
            surface=SURFACE_EXECUTOR,
            posture=posture.posture(),
            posture_source=posture.posture_source(),
            session=posture.posture_session(),
            subject=fields.subject,
            detail=fields.detail,
        )
    except Exception:
        logger.warning("Could not record the code review", exc_info=True)


async def _send_review_notice(tool: str, outcome: ReviewOutcome, request: ReviewRequest) -> None:
    """Tell the operator's activity feed, when the outcome has anything to tell it."""
    from osprey.services.code_review.review import notice_text

    text = notice_text(outcome, request)
    if text is None:
        return
    try:
        await notify_agent_activity_async(tool, "review", detail=text)
    except Exception:  # a notice is best-effort and never changes the outcome
        logger.warning("Could not send the code review notice", exc_info=True)


async def _conclude_review(tool: str, outcome: ReviewOutcome, request: ReviewRequest) -> str | None:
    """Record an outcome, send its notice, and refuse the run if the outcome stops it.

    Returns:
        What the review has to say to the agent about a run that proceeds, if anything.

    Raises:
        Whatever :func:`make_error` raises for ``review_blocked`` or ``review_unavailable``.
    """
    from osprey.services.code_review.contract import ReviewFailure
    from osprey.services.code_review.review import ACTION_REFUSE, agent_message

    _record_review(outcome, request)
    await _send_review_notice(tool, outcome, request)
    message = agent_message(outcome)
    if outcome.action != ACTION_REFUSE:
        return message
    result = outcome.result
    if isinstance(result, ReviewFailure):
        make_error("review_unavailable", str(message), details={"kind": result.kind})
    return make_error(
        "review_blocked",
        str(message),
        details={
            "verdict": result.verdict,
            "reasons": list(result.reasons),
            "suggested_revision": result.suggested_revision,
        },
    )


def _stamped_review(tool: str, gated: str) -> tuple[bool, str | None]:
    """Whether the approval prompt reviewed *gated* for this call, and what it told the agent.

    The stamp is the one :func:`~osprey.mcp_server.approval_record.require_approval_record`
    reads, for the call in scope, and it must name *tool*. Never raises: a stamp that cannot be
    read is no review, and the review here runs.
    """
    try:
        from osprey.audit.call import current_tool_use_id
        from osprey.mcp_server.approval_record import approval_ask_stamp
        from osprey.services.code_review.stamp import stamped_review

        stamp = approval_ask_stamp(current_tool_use_id())
        if stamp is None or not str(stamp.get("tool") or "").endswith(tool):
            return False, None
        return stamped_review(stamp, gated)
    except Exception:
        logger.warning("Could not read the prompt's review from the ask stamp", exc_info=True)
        return False, None


class _PreLaunchReview(NamedTuple):
    """What the review before the launch came to.

    Attributes:
        message: What the review has to say to the agent about a run that proceeds, if anything.
        spared: The request whose review was skipped because the executor was about to decline
            the run; ``None`` when there was nothing to spare.
        level: The level *spared* was skipped under.
    """

    message: str | None
    spared: ReviewRequest | None = None
    level: str | None = None


async def _review_before_launch(
    *,
    tool: str,
    code: str,
    gated: str,
    description: str,
    execution_mode: str,
    project_root: Path | None,
    script_args: Sequence[str],
    approved_target: str | None,
    approved_view_sha256: str | None,
) -> _PreLaunchReview:
    """Review the code when this deployment asks for it, and refuse what the review stops.

    Nothing is reviewed when the feature is off, when the mode is not one it reviews, or when
    the executor would decline the run anyway: that decline stays ``execute_code``'s to produce.
    That last skip is a prediction the launch re-decides, so it comes back as ``spared`` and the
    launch is held to it (:func:`run_gated_execution`).
    A section that cannot be read refuses in every mode, since the set to review is unknown.

    A call the approval prompt already reviewed is not reviewed again: its ask stamp carries the
    digest of the code the reviewer read and what the agent is told
    (:mod:`osprey.services.code_review.stamp`), and a digest that is this code's answers for it.
    That makes this gate the only review of a run no prompt saw: a governed call with the write
    switch at ``allowed``, a read-only run, a hook that could not review.

    Args:
        tool: Tool name as the agent knows it.
        code: The source the launch receives, which the executor's own checks read.
        gated: The source as the operator reads it, which is what gets reviewed.
        description: Caller's description.
        execution_mode: ``"readonly"`` or ``"readwrite"``.
        project_root: Project root when the caller resolved one.
        script_args: The script's arguments, for ``execute_file``.
        approved_target: The control target the approval hook put into the call.
        approved_view_sha256: The view digest the approval hook put into the call.

    Returns:
        The review's message, and the request it spared when it skipped on a predicted decline.

    Raises:
        Whatever :func:`make_error` raises for ``review_blocked`` or ``review_unavailable``.
    """
    from osprey.services.code_review.contract import (
        SURFACE_EXECUTE,
        ReviewFailure,
        ReviewRequest,
    )
    from osprey.services.code_review.settings import ReviewConfigError, load_review_settings

    subject = _tool_subject(tool)
    try:
        settings = load_review_settings(None)
    except ReviewConfigError as exc:
        from osprey.services.code_review.review import decide

        # Before any level or any question about the run: the record names neither.
        request = ReviewRequest(
            surface=SURFACE_EXECUTE, code=gated, subject=subject, description=description
        )
        return _PreLaunchReview(
            await _conclude_review(tool, decide(ReviewFailure("config", str(exc)), None), request)
        )
    if not settings.enabled or execution_mode not in settings.modes:
        return _PreLaunchReview(None)

    from osprey.mcp_server.python_executor.executor import (
        _resolve_project_root,
        admission_refusal,
    )
    from osprey.services.code_review.review import review_code
    from osprey_connectors.workspace import resolve_config_path

    approved = {
        "approved_target": approved_target,
        "approved_view_sha256": approved_view_sha256,
    }
    request = ReviewRequest(
        surface=SURFACE_EXECUTE,
        code=gated,
        subject=subject,
        description=description,
        mode=execution_mode,
        script_args=tuple(script_args),
    )
    if admission_refusal(code, execution_mode, tool, approved):
        return _PreLaunchReview(None, request, settings.level)

    reviewed, stamped_message = _stamped_review(tool, gated)
    if reviewed:
        return _PreLaunchReview(stamped_message)

    outcome = await review_code(
        request,
        settings,
        repo_root=project_root if project_root is not None else _resolve_project_root(),
        render_dir=resolve_config_path().parent,
    )
    return _PreLaunchReview(await _conclude_review(tool, outcome, request))


async def _conclude_spared_review(
    tool: str, review: _PreLaunchReview, exec_result: ExecutionResult
) -> str | None:
    """Answer for a review spared on a decline the launch did not make.

    Only ``flag`` answers here: under ``block`` the launch was held to the decline and refused a
    run it admitted (:data:`~osprey.mcp_server.python_executor.executor.FAILURE_KIND_UNREVIEWED`).
    A run that started is recorded and told as a review that did not run, which is what ``flag``
    does with one.
    """
    if (
        review.spared is None
        or review.level != "flag"
        or exec_result.failure_kind in executor.NOT_STARTED_FAILURE_KINDS
    ):
        return review.message
    from osprey.services.code_review.contract import ReviewFailure
    from osprey.services.code_review.review import decide

    failure = ReviewFailure(
        "skipped", "the run was about to be declined, and by the launch it no longer was"
    )
    return await _conclude_review(tool, decide(failure, review.level), review.spared)


async def run_gated_execution(
    *,
    tool: str,
    code: str,
    description: str,
    execution_mode: str,
    project_root: Path | None = None,
    record_code: str | None = None,
    notify: Callable[..., Awaitable[Any]] | None = None,
    approved_target: str | None = None,
    approved_view_sha256: str | None = None,
    code_review: bool = True,
    script_args: Sequence[str] = (),
) -> tuple[ExecutionResult, dict]:
    """Run *code* through every executor gate, launch it, and report the run.

    The one gate sequence every executor tool runs, in this order: the posture
    clamp, ``quick_safety_check``, the path policy, the readonly import
    denylist, pattern detection, the readonly pattern refusal, the code review, the launch, the
    write-activity report and the runtime-refusal report. A call that trips several gates is told about the
    first one only, so the order is part of what every tool says, and keeping
    it here is what stops two tools from drifting apart on it.

    The posture clamp goes first on purpose. Whether this *session* may write
    at all is not a question the pattern detector or the path policy get a say
    in, and a sandboxed caller should be told that rather
    than whatever a later gate happens to object to first. It is also what
    makes the executor's ``sandbox_env["OSPREY_EXECUTION_MODE"] =
    execution_mode`` overwrite safe: under the sandbox posture the only mode
    that ever reaches the spawn is readonly, so the overwrite can only ever
    re-assert the posture it found.

    The write-activity report fires once the script has been handed to the
    subprocess — writes it performed are already on the machine and a mid-run
    error does not undo them. ``execution_time_seconds`` is the launch
    discriminator: only the subprocess path sets it, while every "never
    launched" outcome comes back from ``execute_code``'s setup handler with it
    still ``None``. :func:`write_activity_detail` decides whether the run
    reports and with which words.

    The timeout is not a parameter: ``execute_code`` reads it from config.

    Args:
        tool: Tool name as the agent knows it — recorded with every refusal and
            reported with the run.
        code: The source handed to the subprocess backend.
        description: Caller's description, recorded with any refusal.
        execution_mode: ``"readonly"`` or ``"readwrite"``; anything else is
            refused before any gate.
        project_root: Project root when the caller has already resolved one;
            ``None`` lets the path-policy resolvers derive it.
        record_code: The source the gates inspect and the audit records, when
            it differs from what is launched — ``execute_file`` launches its
            file behind an argv preamble but gates and records the file as the
            operator would read it. ``None`` means *code*.
        notify: The write-activity reporter. A tool passes its own module-level
            reference so the seam a caller patches is the tool's; ``None``
            reports through this module's.
        approved_target: The control target the approval hook put into the
            call, handed to the launch.
        approved_view_sha256: The pyAML view digest the approval hook put into
            the call, handed to the launch.
        code_review: Whether this tool's code is reviewed at all. ``False`` skips the settings
            load too, so a malformed ``code_review`` section never refuses such a tool.
        script_args: The script's arguments, which only the review request uses.

    Returns:
        ``(exec_result, patterns)`` — the backend's result and the pattern
        detector's findings, for the tool's own response builder.

    Raises:
        Whatever :func:`make_error` raises for the refusing gate.
    """
    gated = code if record_code is None else record_code

    require_known_execution_mode(execution_mode)
    enforce_posture_clamp(execution_mode, tool=tool)

    try:
        from osprey.services.python_executor.analysis.safety_checks import quick_safety_check

        passed, safety_issues = quick_safety_check(gated)
        if not passed:
            make_error(
                "safety_error",
                f"{_SAFETY_SUBJECT.get(tool, 'Code')} failed pre-execution safety checks.",
                safety_issues,
            )
    except ImportError:
        logger.warning("Safety check module unavailable — executing without pre-checks")

    await enforce_path_policy(
        tool=tool,
        code=gated,
        description=description,
        execution_mode=execution_mode,
        project_root=project_root,
    )

    # The pre-execution half of the readonly contract; the runtime half (the
    # wrapper's readonly guard and the connector refusal) catches what no
    # static check can.
    if execution_mode == "readonly":
        try:
            from osprey.services.python_executor.analysis.safety_checks import (
                check_readonly_imports,
            )

            import_issues = check_readonly_imports(gated)
        except ImportError:
            import_issues = []
        if import_issues:
            await refuse_readonly_write(
                tool=tool,
                layer=LAYER_IMPORT_DENYLIST,
                trigger=import_issues,
                code=gated,
                description=description,
                message="Control-system client libraries cannot be imported in readonly mode.",
                suggestions=[
                    *import_issues,
                    "Use read_channel() from osprey.runtime for reads.",
                    (
                        "Set execution_mode to 'readwrite' if writes are intentional, "
                        "and write through osprey.runtime.write_channel(address, value)."
                    ),
                ],
            )

    try:
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
        )

        patterns = detect_control_system_operations(gated)
    except ImportError:
        logger.warning("Pattern detection module unavailable — skipping write detection")
        patterns = {"has_writes": False, "has_reads": False, "detected_patterns": {}}

    if patterns.get("has_writes") and execution_mode == "readonly":
        await refuse_readonly_write(
            tool=tool,
            layer=LAYER_PATTERN_DETECTION,
            trigger=patterns.get("detected_patterns", {}),
            code=gated,
            description=description,
            message="Control-system write patterns detected in readonly mode.",
            suggestions=[
                (
                    "Set execution_mode to 'readwrite' if writes are intentional, "
                    "and write through osprey.runtime.write_channel(address, value)."
                ),
                "Detected patterns: " + json.dumps(patterns.get("detected_patterns", {})),
            ],
        )

    review = _PreLaunchReview(None)
    if code_review:
        review = await _review_before_launch(
            tool=tool,
            code=code,
            gated=gated,
            description=description,
            execution_mode=execution_mode,
            project_root=project_root,
            script_args=script_args,
            approved_target=approved_target,
            approved_view_sha256=approved_view_sha256,
        )

    from osprey.mcp_server.python_executor.executor import execute_code

    exec_result = await execute_code(
        code=code,
        execution_mode=execution_mode,
        description=description,
        tool=tool,
        approved_target=approved_target,
        approved_view_sha256=approved_view_sha256,
        # A review spared on a predicted decline holds the launch to that decline, unless the
        # deployment only flags: there an unreviewed run goes ahead and the agent is told.
        expect_declined=review.spared is not None and review.level != "flag",
    )
    review_message = await _conclude_spared_review(tool, review, exec_result)
    if review_message is not None:
        exec_result.code_review = review_message

    detail = write_activity_detail(bool(patterns.get("has_writes")), execution_mode)
    if detail is not None and exec_result.execution_time_seconds is not None:
        reporter = notify if notify is not None else notify_agent_activity_async
        await reporter(tool, "channel", detail=detail)

    # A write the runtime guard refused mid-run reaches here only as a
    # traceback on stderr; report it like the pre-execution refusals.
    await report_runtime_refusal(
        tool=tool,
        stderr=exec_result.stderr,
        code=gated,
        description=description,
        execution_mode=execution_mode,
    )

    return exec_result, patterns
