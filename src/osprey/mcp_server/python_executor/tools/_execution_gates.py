"""Shared execution-mode gates for the ``execute`` and ``execute_file`` tools.

Both tools take an ``execution_mode`` string and guard control-system writes
with two independent checks: a per-call readonly gate (pattern detection) and a
deployment-level kill switch (the write posture of the control target this
session is on). Each gate only recognises one canonical spelling, so any *other*
string falls through both — not "readonly", so write patterns are not blocked;
not "readwrite", so the kill switch never fires. Rejecting unknown modes here
closes that hole for every caller at once, and gives the kill switch a single
implementation instead of one copy per tool. A third gate clamps a run to the
*session* posture inherited from the Web Terminal, which is about this session
rather than this deployment and so refuses in its own vocabulary.

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

import logging
import os
from pathlib import Path
from typing import Any, NoReturn

from osprey.audit import posture
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async

logger = logging.getLogger("osprey.mcp_server.tools.execution_gates")

#: The closed set of recognised execution modes. Downstream gates may test
#: equality against either member only because this set is enforced first.
VALID_EXECUTION_MODES = frozenset({"readonly", "readwrite"})

#: The session posture, its provenance and its posture-store key —
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

#: The reason a session-posture refusal is filed under. The same spelling the
#: MCP middleware and the client-side ``osprey_writes_check.py`` hook use for
#: the same refusal: one refusal, three possible layers, and an operator
#: querying the ledger for it should not have to know which. A cross-layer test
#: pins all three (the hook by AST, since it imports nothing from ``osprey``).
REASON_POSTURE = "posture"

#: The layer that refused a control-system write, filed as the record's
#: ``reason``. Static layers refuse before the script is launched;
#: ``runtime_guard`` is the one that fires inside the subprocess, so it is the
#: layer that catches the spellings the static ones cannot see. Spelled here,
#: beside the recorder, because this module is where every one of them converges
#: -- they used to live in the retired P1 ledger module, which was their only
#: other shared home.
LAYER_IMPORT_DENYLIST = "import_denylist"
LAYER_PATTERN_DETECTION = "pattern_detection"
LAYER_PATH_POLICY = "path_policy"
LAYER_RUNTIME_GUARD = "runtime_guard"

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


def session_control_target() -> str | None:
    """The control target this session is on, or ``None`` when there is none to read.

    ``None`` is not a failure: it is the honest answer for a session that never
    selected a target, and the posture lookup reads it as the deployment
    baseline — the connector an unstamped run provably builds.

    Every way of not knowing lands there too, which is why the read lives here
    and not inside :func:`enforce_deployment_writes_gate`'s import guard.
    Failing to learn the target must *narrow* the write question to the
    baseline; joining a guard whose failure path is ``return`` would drop the
    write check altogether on a state directory that happened to be unreadable.
    """
    try:
        from osprey.mcp_server.python_executor.executor import _session_target_record

        record = _session_target_record()
    except Exception:
        logger.warning(
            "Session control target unavailable — the deployment writes gate "
            "answers for the baseline target",
            exc_info=True,
        )
        return None

    if record is None:
        return None
    target = record.get("target")
    return str(target) if target else None


def enforce_deployment_writes_gate(execution_mode: str, target: str | None) -> None:
    """Raise ``ToolError`` (safety_error) on readwrite runs the target's posture does not arm.

    Fires whenever the caller asks for write mode, regardless of whether the
    pattern detector recognises specific write syntax — the deployment-level
    kill switch must not depend on detection accuracy.

    Write posture is per control target, so the question is asked about the one
    this session is on: a deployment whose baseline is a live machine can have
    writes armed on its virtual accelerator and refused on the machine, and a
    single deployment-wide answer would be wrong for one of them. ``None``
    means no target was readable, which the posture lookup answers for the
    baseline rather than by skipping.

    Two postures are consulted, because they refuse for different reasons and
    an operator reading either message must be told the right way forward: this
    SESSION's per-target entry in the posture store
    (:func:`_enforce_session_store_term`, changed from the header chip) and the
    deployment's own key (config, changed by editing the project and
    rebuilding). The session term goes first because the config read's
    ``ImportError`` path returns, and an unimportable config module must not
    take the session posture down with it.

    Args:
        execution_mode: The run's mode; only ``"readwrite"`` is gated here.
        target: The session's control target, from
            :func:`session_control_target`, or ``None`` for the baseline.
    """
    if execution_mode != "readwrite":
        return

    # Ahead of the config read, whose ImportError path RETURNS: a config module
    # that cannot be imported must not take the session's own posture down with
    # it. The two terms are independent — one reads the project config, the
    # other a JSON file under the agent-data root — so neither may be gated on
    # the other's availability.
    _enforce_session_store_term(target)

    try:
        from osprey.services.python_executor.execution.control import (
            get_execution_control_config,
        )

        exec_control_config = get_execution_control_config(target=target)
    except ImportError:
        logger.warning(
            "Execution control config unavailable — skipping deployment-level writes check"
        )
        return

    if (
        exec_control_config is not None
        and exec_control_config.control_system_writes_enabled is False
    ):
        key = exec_control_config.writes_enabled_key
        active_target = exec_control_config.active_target
        scope = f"control target '{active_target}'" if active_target else "this deployment"
        make_error(
            "safety_error",
            f"Control-system writes are disabled for {scope} ({key}=false in project config).",
            [
                f"Set {key}=true in the project config to enable writes for {scope}.",
            ],
            details={"active_target": active_target, "writes_enabled_key": key},
        )


def _enforce_session_store_term(target: str | None) -> None:
    """Raise ``ToolError`` (safety_error) when the SESSION narrowed this target.

    :func:`enforce_posture_clamp` above already refuses a sandboxed session, and
    for a session whose control target is knowable it is the gate that fires —
    :func:`osprey.audit.posture.posture` resolves that target and reads the same
    store. This term exists for the case it cannot cover: when nothing can say
    which target the session is on (no controls-server record — the MCP servers
    are not installed, or a process sits between them and this one),
    ``posture()`` degrades to the ENVIRONMENT answer, which for this feature is
    always the writes posture because no spawn site stamps a per-target
    narrowing into it. Without this the run would be granted writes that the
    header chip took away.

    The rule for that case is the store's own: with no resolvable target, the
    MOST RESTRICTIVE entry recorded for the session decides — which is exactly
    what :func:`~osprey_connectors.session_store.store_permits` does when it is
    handed ``None``, so it is asked rather than restated here.

    Degrades like the deployment check above: a store that cannot be read at all
    is logged and skipped rather than turned into a refusal on every run. The
    barrier this backs up is not this gate but the connector's own reference
    monitor inside the sandbox, which asks the same question of the same file at
    the moment of the write.
    """
    try:
        from osprey_connectors.session_store import store_permits

        permitted = store_permits(posture.posture_session(), target)
    except Exception:  # noqa: BLE001 - the store degrades; the run does not fail here
        logger.warning(
            "Session write posture unavailable — skipping the session-level writes check",
            exc_info=True,
        )
        return

    if permitted:
        return

    # The wording is enforce_posture_clamp's, deliberately: this is the same
    # refusal met one layer down, and an operator who meets both gates in one
    # session should not have to work out that they are the same answer.
    if target is None:
        make_error(
            "safety_error",
            "Writes are off for at least one control target in this session (the "
            "run's target could not be identified, so the most restrictive "
            "decides) — turned off from the control-target chip in the header.",
            [
                'Re-run with execution_mode="readonly" — reads are unaffected by the posture.',
                "Turn writes back on from the control-target chip in the header if "
                "the write is intended; the deployment config is not the gate here.",
            ],
            details={"active_target": target},
        )
    make_error(
        "safety_error",
        f"Writes are off for the '{target}' control target in this session — "
        "turned off from the control-target chip in the header, and in force "
        "for this session only.",
        [
            'Re-run with execution_mode="readonly" — reads are unaffected by the posture.',
            f"Turn writes back on for '{target}' from the control-target chip in "
            "the header if the write is intended; the deployment config is not "
            "the gate here.",
        ],
        details={"active_target": target},
    )


def _tool_subject(tool: str) -> str:
    """The ``mcp__<prefix>__<tool>`` name this refusal is about.

    Composed exactly as the MCP audit middleware composes it, so the record the
    clamp files and the records the middleware files for neighbouring calls
    name the same subject. With no prefix — a server not launched by ``osprey``
    — the bare tool name, which is what the middleware falls back to too.
    """
    prefix = (os.environ.get(TOOL_PREFIX_ENV_VAR) or "").strip()
    return f"mcp__{prefix}__{tool}" if prefix else tool


def _refusal_detail(tool: str, execution_mode: str, trigger: Any, description: str | None) -> str:
    """The supplementary context one refused write carries.

    ``tool`` and ``mode`` first: they are the two an operator scans a ledger
    for. The trigger and the agent's description follow, each bounded, because
    both are as long as whatever produced them.
    """
    parts = [f"tool={tool}", f"mode={execution_mode}", f"trigger={trigger}"[:MAX_TRIGGER_CHARS]]
    if description:
        parts.append(f"description={description}"[:MAX_DESCRIPTION_CHARS])
    return " ".join(parts)


def _record_posture_clamp(tool: str) -> None:
    """File the session-posture refusal, and claim it for this layer.

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
            detail=f"tool={tool}",
        )
    except Exception:  # noqa: BLE001 - the audit trail degrades; the refusal does not
        logger.warning("Could not record the session-posture refusal", exc_info=True)


def enforce_posture_clamp(execution_mode: str, *, tool: str) -> None:
    """Raise ``ToolError`` (safety_error) on readwrite runs the posture refuses.

    :func:`osprey.audit.posture.posture` answers ``sandbox`` for two different
    reasons, and the operator's next move is different for each — so the
    refusal forks on the source:

    * the **deployment** is running in readonly execution mode, i.e.
      ``OSPREY_EXECUTION_MODE`` is set to ``readonly`` on this very process.
      ``posture()`` short-circuits to that ENVIRONMENT answer before the store
      is ever consulted. Nothing about this session can lift it: the run has to
      be started without the variable, so the message must not send the
      operator to the chip, which already reads writes.
    * this **session's posture for ONE control target** is read-only, resolved
      from the session store. That is the operator's own narrowing, made from
      the control-target chip in the header, and the chip is where it lifts. It
      is per target, so the message names the target — this gate runs before
      every readwrite tool call, and "this session is sandboxed" would read as
      a session-wide block on a session that is working normally on every other
      machine. The target is resolved through
      :func:`~osprey.audit.posture.session_control_target`, the same resolver
      :func:`~osprey.audit.posture.posture` used to decide the clamp fires at
      all, so the name in the refusal is the machine the clamp fired for.
      Where that resolver cannot name a target the refusal says so and names no
      machine, rather than inventing one: the store's rule with no resolvable
      target is that the most restrictive entry decides, and which entry that
      was is not something this gate can honestly report.

    Either way this gate is what makes the executor obey the answer: without
    it, an agent could ask for ``readwrite`` and get it, because the deployment
    kill switch above only knows about ``writes_enabled`` and has nothing to
    say about either of these.

    The environment test is a **value** comparison, deliberately mirroring
    ``osprey_connectors``' ``is_readonly_run``: only the exact string
    ``"readonly"`` names a read-only run. A presence check would claim one for
    every process whose environment carries the variable for any other reason
    — including the executor's own per-run ``"readwrite"``.

    A refusal is recorded in the unified ledger on the ``executor`` surface and
    marked as this layer's own — see :func:`_record_posture_clamp`. Nothing is
    recorded when the gate does not fire: a readonly run and a session that was
    never sandboxed are both ordinary, and a ledger that logged them would bury
    the refusals it exists for.

    Args:
        execution_mode: The mode this call asked for.
        tool: The tool name as the agent knows it, for the recorded subject.
            Required and keyword-only: a default would let a new call site
            record a refusal under another tool's name, and a positional would
            let it swap the two strings silently.
    """
    if execution_mode != "readwrite":
        return
    if posture.posture() != POSTURE_SANDBOX:
        return

    _record_posture_clamp(tool)

    if os.environ.get(posture.POSTURE_ENV_VAR) == posture.SANDBOX_MODE:
        make_error(
            "safety_error",
            "This deployment is running in readonly execution mode, which refuses "
            "control-system writes regardless of what the run asks for.",
            [
                'Re-run with execution_mode="readonly" — reads are unaffected.',
                "Writes need the deployment started without "
                "OSPREY_EXECUTION_MODE=readonly; the control-target chip in the "
                "header cannot lift a deployment-wide read-only run.",
            ],
        )

    # Degrades to the target-less wording rather than to a crash. The resolver
    # is documented never to raise, but this runs on the refusal path of every
    # readwrite tool call: a surprise here would turn a refusal into a 500 and
    # lose the safety answer the clamp already reached.
    try:
        target = posture.session_control_target()
    except Exception:  # noqa: BLE001 - the name degrades; the refusal does not
        logger.warning(
            "Could not name the session's control target for the posture refusal",
            exc_info=True,
        )
        target = None

    if target is None:
        make_error(
            "safety_error",
            "Writes are off for at least one control target in this session (the "
            "run's target could not be identified, so the most restrictive "
            "decides) — turned off from the control-target chip in the header.",
            [
                'Re-run with execution_mode="readonly" — reads are unaffected by the posture.',
                "Turn writes back on from the control-target chip in the header if "
                "the write is intended; the deployment config is not the gate here.",
            ],
        )

    make_error(
        "safety_error",
        f"Writes are off for the '{target}' control target in this session — "
        "turned off from the control-target chip in the header, and in force "
        "for this session only.",
        [
            'Re-run with execution_mode="readonly" — reads are unaffected by the posture.',
            f"Turn writes back on for '{target}' from the control-target chip in "
            "the header if the write is intended; the deployment config is not "
            "the gate here.",
        ],
    )


def _record_write_refusal(
    *,
    tool: str,
    layer: str,
    trigger: Any,
    code: str,
    description: str | None,
    execution_mode: str,
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
            detail=_refusal_detail(tool, execution_mode, trigger, description),
        )
    except Exception:  # noqa: BLE001 - the audit trail degrades; the refusal does not
        logger.warning("Could not record the refusal for audit", exc_info=True)


async def record_and_alert_refusal(
    *,
    tool: str,
    layer: str,
    trigger: Any,
    code: str,
    description: str | None = None,
    execution_mode: str,
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
    """
    _record_write_refusal(
        tool=tool,
        layer=layer,
        trigger=trigger,
        code=code,
        description=description,
        execution_mode=execution_mode,
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
    """Report a write the *runtime* guard refused, mid-run. Returns whether it did.

    The static layers refuse by raising, so they control their own reporting.
    The runtime ones cannot: they run inside the subprocess, and all that
    reaches here is a traceback on stderr. Without this, the layers that catch
    the evasive spellings — aliased imports, ``getattr``, ``importlib``, a
    shelled-out ``caput`` — would be the only ones that never alerted the
    operator or left an audit record.

    Matching is on
    :data:`~osprey.services.python_executor.execution.wrapper.READONLY_REFUSAL_MARKER`
    rather than on the guard's full message, so the connector reference
    monitor's own refusal — a write that took the *approved* ``write_channel``
    path in a readonly run — is reported too.

    The script's own result is left alone: its stderr already names the mode
    and the way forward, and converting it into a tool error here would
    discard whatever the run legitimately produced before the refusal.

    ``execution_mode`` comes from the run rather than from a default. Today the
    marker only reaches here out of a readonly run — the readwrite filesystem
    refusal in
    :mod:`~osprey.services.python_executor.execution.wrapper` deliberately does
    not carry it — but that is a property of another module's message strings,
    and a guard that ever emits the marker mid-``readwrite`` (the connector
    reference monitor is the candidate) must not have the ledger and the
    operator alert call that run readonly.
    """
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

    if READONLY_REFUSAL_MARKER not in (stderr or ""):
        return False

    await record_and_alert_refusal(
        tool=tool,
        layer=LAYER_RUNTIME_GUARD,
        trigger=_refusal_lines(stderr),
        code=code,
        description=description,
        execution_mode=execution_mode,
    )
    return True


def _refusal_lines(stderr: str) -> list[str]:
    """The stderr lines naming the refusal, for the audit record's ``trigger``.

    The whole traceback would bury the fact in noise and the bare marker would
    drop the channel name the connector's message carries; the matching lines
    keep what an auditor actually reads.
    """
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

    return [line.strip() for line in stderr.splitlines() if READONLY_REFUSAL_MARKER in line]
