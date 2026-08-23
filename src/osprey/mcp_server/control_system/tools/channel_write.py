"""MCP tool: channel_write — write values to control-system channels.

Safety: PreToolUse hooks enforce human approval before this tool runs.
The tool docstring is the static prompt the agent sees.

Binding a write to the target it was approved on
------------------------------------------------
A value is approved for a *machine*, not for a channel name. Between the moment
an operator approves a write and the moment it reaches the control system, the
session's control-system target can change — a switch is a process lifecycle
operation running in another task, and it does not wait for a write that is
already in flight. Applying a value approved for the simulator to the real
machine because it arrived a second late is the failure this module exists to
prevent.

The window is closed with three observations of one quantity,
``(target, generation)``, each compared against the one before it:

* **at the approval prompt** — the binding the PreToolUse approval hook rendered
  the operator's ``Target:`` line from. Approval is enforced *outside* this
  server, so the tool cannot see the prompt; the hook therefore writes what it
  rendered into a stamp file beside the target state, keyed by the write payload
  itself (see :func:`_approval_stamp_key`), and this module reads it back. That
  is the only way the render-to-click window — where the human is thinking and a
  switch can land — is visible here at all. A call with no stamp is not
  compared: an older render, a deployment without the hook, and a write the
  policy allowed without asking must all keep working.
* **at entry** — the first statement of the tool, before anything it does can
  yield. This is the first instant the server itself exists in, and it closes
  the click-to-execution window.
* **immediately before the connector call** — after the connector has been
  resolved, before a single value is sent.

Any difference refuses the whole call, naming both pairs. The appearance or
disappearance of the record counts as a difference: "no record" and "a record"
are different claims about the session, and a server that started or stopped
publishing mid-call has changed something the operator was never shown.

The pair is read from the state file
:mod:`~osprey.mcp_server.control_system.target_state` publishes, and not from
the manager's in-memory accessors: that file is what the approval hook renders
its ``Target:`` line from, so binding to it binds to exactly what the operator
saw. The two are cross-checked rather than ranked — see below.

A server publishes its baseline record unconditionally at start
(``server._reset_target_state``), so the ordinary in-process deployment reads a
stable ``(baseline, 0)`` at every observation and never refuses. Reading
``None`` is the degraded case — an unwritable data root, a state directory that
cannot be resolved — and it is stable too, so that deployment also proceeds,
paying one cheap read.

Two writers of one truth
------------------------
The state file says what the operator was shown; the connector-host manager's
:meth:`~osprey.mcp_server.control_system.connector_host_manager.ConnectorHostManager.active_binding`
says what is actually being served. They are the same quantity published by one
writer, so any disagreement between them means the publish failed or has not
landed — and a write is exactly the wrong thing to let through while the
session's identity is in doubt. Once the manager has started a child, the
pre-write check therefore refuses on ANY disagreement between the two instead of
choosing a winner. A manager that never started has nothing to say and is not
consulted.

Why no locking is needed for the remaining window
-------------------------------------------------
A switch can still complete between the pre-write read and the connector call.
Closing that window is a contract this module *relies on* rather than one it
implements, and it belongs to the seam that serves the connector: a switch
retires the child it is replacing — refusing new requests on that child's proxy,
draining it, and naming the switch as the reason its stream ended — so a write
that arrives after the retirement fails with a :class:`ConnectionError`
attributed to the switch, which is the same refusal in a different voice. That
is the retirement behaviour
:meth:`~osprey.mcp_server.control_system.connector_host_manager.ConnectorHostManager._retire`
implements today; it becomes this module's guarantee once the tools are served
from the child, and it is the reason taking the switch's lock here would
serialise every write behind the supervisor for no additional guarantee.
"""

import hashlib
import json
import logging
import os

from osprey.errors import ChannelWriteBlockedError
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async

logger = logging.getLogger("osprey.mcp_server.tools.channel_write")

#: Error type of a write refused because the session's control-system target
#: moved after the write was approved. Deliberately its own word rather than the
#: ``write_refused`` the reference monitor raises: that one says a *channel* was
#: refused on policy grounds, this one says the *session* is no longer the one
#: the write was approved for, and an agent told to stop for the first reason
#: would give an operator the wrong account of the second.
TARGET_CHANGED_ERROR = "target_changed"

#: Which of the three comparisons refused a call, reported in ``details.window``
#: so an operator (and a test) can tell "the session moved while I was deciding"
#: from "the session moved while the write was being prepared" from "the two
#: sources of the session's identity disagree".
WINDOW_APPROVAL = "approval_prompt_to_entry"
WINDOW_EXECUTION = "entry_to_write"
WINDOW_SERVING = "published_vs_serving"

#: Name of the stamp the approval hook leaves beside the target state. The
#: prefix is deliberately not ``target_state_``: a stamp must never be picked up
#: by the state-file glob, whose files name a server, not an approval.
APPROVAL_STAMP_PREFIX = "write_approval_"
APPROVAL_STAMP_SUFFIX = ".json"

#: Severity at or above which a *verified* readback is still reported as an
#: alarm state. EPICS grades MINOR=1, MAJOR=2, INVALID=3; a MINOR alarm on a
#: channel that read back the value it was given is routine, MAJOR is not.
VERIFIED_ALARM_SEVERITY = 2

#: The closed set of ``write_state`` words, in the order the predicates below
#: test them. The generated safety rules name this key, so the spelling here and
#: the spelling there have to stay one string.
WRITE_STATE_KEY = "write_state"


def _alarm_severity(verification: object) -> int | None:
    """Readback alarm severity as an int, or None when it was not reported.

    Read through ``getattr`` and type-checked: a custom connector may return any
    duck-typed verification object, and a missing or non-numeric severity must
    degrade to "not reported" rather than raise inside the result projection.
    """
    severity = getattr(verification, "readback_alarm_severity", None)
    if isinstance(severity, bool) or not isinstance(severity, int):
        return None
    return severity


def _derive_write_state(result: object) -> str:
    """Classify one write result into a single closed-set word.

    Derived only from structured fields — never from ``notes``, which is
    display-only text — so the state an agent keys on cannot drift with wording.
    The predicates are ordered: the first one that matches wins, which is what
    keeps a refused write from also being reported as unverified.
    """
    if getattr(result, "blocked", None):
        return "blocked"
    if not getattr(result, "success", False):
        return "write_failed"

    verification = getattr(result, "verification", None)
    if verification is None:
        return "verification_not_reported"
    if getattr(verification, "level", None) == "none":
        return "verification_not_requested"

    severity = _alarm_severity(verification)
    if getattr(verification, "verified", False):
        if severity is not None and severity >= VERIFIED_ALARM_SEVERITY:
            return "verified_with_alarm"
        return "verified"

    if getattr(verification, "failure_kind", None) == "readback_failed":
        return "readback_failed"
    if severity is not None and severity > 0:
        return "unverified_alarm"
    return "verification_failed"


def _read_target_binding() -> tuple[str, int] | None:
    """The ``(target, generation)`` this server publishes, or ``None``.

    ``None`` is the single answer for every "there is no usable record" case —
    no state file, an unreadable one, a half-written one, a record whose target
    or generation is not what it should be. The state file's readers are
    fail-closed by contract precisely so that all of those arrive as one value,
    and collapsing them here is what makes an unpublished deployment stable
    across the call rather than intermittently "changed".
    """
    try:
        record = target_state.read()
    except Exception:  # pragma: no cover - defensive: reading must not fail a write
        logger.debug("Could not read the control-system target state", exc_info=True)
        return None
    return _binding_from_record(record)


def _binding_from_record(record: object) -> tuple[str, int] | None:
    """Normalize one raw record into a binding, or ``None``.

    The hook side normalizes the same record the same way (``selected_target``
    plus an integer generation, whitespace stripped), because the two have to
    agree on which records count as unpublished. A record that is half-readable
    is unpublished as a whole here: there is no safe guess at the missing half.
    """
    if not isinstance(record, dict):
        return None
    target = record.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    try:
        return target.strip(), int(record.get("generation"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _approval_stamp_key(operations: list[dict], verification_level: str | None) -> str | None:
    """The name the approval hook filed this write's stamp under, or ``None``.

    A SHA-256 over the canonical JSON of the write's own arguments. The hook and
    this server share no call identifier — the hook is handed a tool-call
    payload, the tool is handed its arguments — so the payload is the only thing
    that provably crosses the gap between them, and it is what both sides key
    on. The hook restates this derivation in stdlib-only Python (it runs outside
    this venv and cannot import this module); a test pins the two spellings
    against each other.

    ``None`` when no key can be formed, which the caller reads as "do not
    compare": no comparison is a better failure than a wrong one.
    """
    if not operations:
        return None
    try:
        payload = json.dumps(
            {"operations": operations, "verification_level": verification_level},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):  # pragma: no cover - arguments arrive as JSON
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _read_approval_stamp(
    operations: list[dict], verification_level: str | None
) -> tuple[bool, tuple[str, int] | None]:
    """``(found, binding)`` for the prompt that approved this write.

    ``found`` is ``False`` for every case in which no comparison may be made:
    no stamp, an unreadable one, or one another server on this checkout wrote —
    two sessions sharing a directory would otherwise cross-check each other's
    approvals, and a stamp whose ``server_pid`` is not this process is somebody
    else's prompt. ``found`` with a ``None`` binding is a real answer: the
    prompt was rendered while nothing published a target.
    """
    key = _approval_stamp_key(operations, verification_level)
    if key is None:
        return False, None
    try:
        path = target_state.state_dir() / f"{APPROVAL_STAMP_PREFIX}{key}{APPROVAL_STAMP_SUFFIX}"
        stamp = target_state.read_file(path)
    except Exception:  # pragma: no cover - defensive: reading must not fail a write
        logger.debug("Could not read the write-approval stamp", exc_info=True)
        return False, None
    if not isinstance(stamp, dict):
        return False, None
    server_pid = stamp.get("server_pid")
    if server_pid is not None and server_pid != os.getpid():
        logger.debug("Ignoring a write-approval stamp written for server pid %s", server_pid)
        return False, None
    return True, _binding_from_record(stamp)


def _serving_binding() -> tuple[str, int] | None:
    """What the connector-host manager is serving, or ``None`` if it serves nothing.

    ``None`` covers both "no manager has started a child" and any failure to
    ask, which is the same fail-open the rest of this module uses: a deployment
    running the in-process connector has no second opinion to cross-check
    against, and inventing one would refuse writes that are perfectly sound.
    """
    try:
        from osprey.mcp_server.control_system.server_context import get_server_context

        hosts = get_server_context().connector_hosts
        if not hosts.is_started():
            return None
        return hosts.active_binding()
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not read the connector host's active binding", exc_info=True)
        return None


def _describe_binding(binding: tuple[str, int] | None) -> str:
    """One binding, rendered for an operator reading the refusal."""
    if binding is None:
        return "an unpublished target (no target state record)"
    target, generation = binding
    return f"target {target!r} (generation {generation})"


def _refuse_target_changed(
    approved: tuple[str, int] | None,
    current: tuple[str, int] | None,
    *,
    window: str,
    summary: str,
) -> None:
    """Refuse this call, naming both bindings and which window they came from.

    Raises:
        fastmcp.ToolError: Carrying the standard envelope. Nothing has been sent
            to the control system at this point, in any of the three windows.
    """
    make_error(
        TARGET_CHANGED_ERROR,
        f"{summary}: approved on {_describe_binding(approved)}; "
        f"now {_describe_binding(current)} — re-run the write.",
        [
            "Nothing was written: the refusal happened before the control system was touched.",
            "Report the change to the operator — the value was approved for a different target.",
            "Do NOT re-issue the write without a fresh approval on the target now active.",
        ],
        details={
            "window": window,
            "approved_target": approved[0] if approved else None,
            "approved_generation": approved[1] if approved else None,
            "current_target": current[0] if current else None,
            "current_generation": current[1] if current else None,
        },
    )


def _check_approval_window(
    operations: list[dict], verification_level: str | None, entry: tuple[str, int] | None
) -> None:
    """Compare what the approval prompt showed against what the tool entered on."""
    found, approved = _read_approval_stamp(operations, verification_level)
    if not found or approved == entry:
        return
    _refuse_target_changed(
        approved,
        entry,
        window=WINDOW_APPROVAL,
        summary="The control-system target changed between the approval prompt and this call",
    )


def _check_execution_window(entry: tuple[str, int] | None) -> None:
    """Compare the entry capture against the state immediately before the write.

    Also cross-checks the published record against what the connector host is
    actually serving: they are one quantity with one writer, so a disagreement
    means the publish failed or has not landed, and the session's identity is in
    doubt at the exact moment a value would go out.
    """
    current = _read_target_binding()
    if current != entry:
        _refuse_target_changed(
            entry,
            current,
            window=WINDOW_EXECUTION,
            summary="The control-system target changed between approval and execution",
        )
    serving = _serving_binding()
    if serving is not None and serving != current:
        _refuse_target_changed(
            current,
            serving,
            window=WINDOW_SERVING,
            summary=(
                "The control-system target cannot be confirmed — the published record and "
                "the running connector host disagree"
            ),
        )


@mcp.tool()
async def channel_write(
    operations: list[dict],
    verification_level: str | None = None,
) -> str:
    """Write values to one or more control-system channels.

    Each operation is a dict with keys: channel (str), value (any), notes (str, optional).
    PreToolUse hooks handle human approval BEFORE this code runs.

    Every result carries one `write_state` word. Report the outcome its
    `write_state` names and nothing stronger — an unverified write is not a
    successful write, and you must not describe it as one. The states, in the
    order they are decided:

    - `blocked` — refused on policy or limits grounds; never sent to the machine.
    - `write_failed` — sent to the machine and failed.
    - `verification_not_reported` — the connector reported no verification at all.
    - `verification_not_requested` — level `none`: the write was sent, nothing
      confirmed it took effect.
    - `verified_with_alarm` — the readback matched, but the channel is in a MAJOR
      or worse alarm.
    - `verified` — the write was confirmed.
    - `readback_failed` — the readback itself could not be read, so the write is
      unconfirmed.
    - `unverified_alarm` — unconfirmed, and the channel reports an alarm.
    - `verification_failed` — unconfirmed: the readback disagreed with the setpoint.

    `summary.verification_failed` counts executed writes that asked for
    verification and did not get it — never a refused or unrequested one.

    Leave `verification_level` unset unless the operator names one. It is then
    resolved by the deployment, per write: the channel's limits entry, then the
    limits database `defaults.verification`, then the connector config, then
    `callback`. This tool does not raise on an unverified write — it reports it in
    `write_state`. (`osprey.runtime.write_channel`, on the Python path, raises.)

    Args:
        operations: List of write operations, each with "channel", "value", and optional "notes".
        verification_level: Optional override — "none", "callback", or "readback".
            Omit it to let the deployment resolve the level per write.

    Returns:
        JSON with per-operation results. Each `summary.results[]` entry carries
        `write_state`; its `verification` carries the readback value, the alarm
        name and severity, and `failure_kind` when the connector reported them.
        `summary.verification_failed` counts executed writes that asked for
        verification and did not get it.
    """
    # The first statement of the tool, before anything that can yield: the
    # earliest instant the server itself exists in. It is compared against what
    # the approval prompt was rendered on (below) and against the state
    # immediately before the write (further down).
    entry_binding = _read_target_binding()

    if not operations:
        return make_error(
            "validation_error",
            "No write operations provided.",
            ["Provide at least one operation with 'channel' and 'value'."],
        )

    # The render-to-click window: what the operator was shown against what this
    # call entered on. Checked before any other work, because a write approved
    # for a different session should cost nothing at all.
    _check_approval_window(operations, verification_level, entry_binding)

    # Limits validation (additional safety layer inside the tool)
    try:
        from osprey.connectors.control_system.limits_validator import LimitsValidator
    except ImportError:
        LimitsValidator = None  # type: ignore[assignment,misc]

    validator = None
    if LimitsValidator is not None:
        validator = LimitsValidator.from_config()

    violations: list[dict] = []
    for op in operations:
        channel = op.get("channel")
        value = op.get("value")
        if not channel:
            return make_error(
                "validation_error",
                "Each operation must include a 'channel' key.",
                ["Ensure every entry in operations has 'channel' and 'value'."],
            )
        if validator:
            try:
                validator.validate(channel, value)
            except Exception as exc:
                violation = {
                    "channel": channel,
                    "attempted_value": value,
                    "violation_type": getattr(exc, "violation_type", "unknown"),
                    "reason": getattr(exc, "violation_reason", str(exc)),
                }
                if getattr(exc, "min_value", None) is not None:
                    violation["min_value"] = exc.min_value
                if getattr(exc, "max_value", None) is not None:
                    violation["max_value"] = exc.max_value
                if getattr(exc, "max_step", None) is not None:
                    violation["max_step"] = exc.max_step
                if getattr(exc, "current_value", None) is not None:
                    violation["current_value"] = exc.current_value
                violations.append(violation)

    if violations:
        # Build a clear message with limits info for each violation
        parts = []
        for v in violations:
            part = f"{v['channel']}={v['attempted_value']}: {v['reason']}"
            if "min_value" in v or "max_value" in v:
                part += f" (allowed range: [{v.get('min_value')}, {v.get('max_value')}])"
            if "max_step" in v:
                part += f" (max step: {v['max_step']})"
            parts.append(part)

        return make_error(
            "limits_violation",
            f"Channel limits violated: {'; '.join(parts)}",
            [
                "Do NOT attempt to work around this limit.",
                "Report the violation to the operator with the allowed range.",
                "The operator may adjust the limits database if the value is appropriate.",
            ],
            details=violations,
        )

    # Execute writes
    async with connector_error_handler("channel_write"):
        from osprey.mcp_server.control_system.server_context import get_server_context

        registry = get_server_context()
        connector = await registry.control_system()

        # The last thing before a value goes out: resolving the connector was
        # the final await, so this read is as close to the write as the server
        # can get. A switch completing after it is refused by the switch itself
        # — see the module docstring — and not papered over here.
        _check_execution_window(entry_binding)

        # Determine per-channel verification level and tolerance
        connector_results = []  # Raw connector results for bridge
        results_serialised = []  # Serialised dicts for the data file

        if len(operations) == 1:
            op = operations[0]
            channel, value = op["channel"], op["value"]
            # An explicit level is the caller's decision and is forwarded
            # unchanged; the limits database only fills in a level the caller
            # left out. The tolerance is independent of that: a per-channel
            # tolerance applies whether or not the level was named, or an
            # explicit "readback" would silently fall back to the connector's
            # absolute default.
            level = verification_level
            tolerance = None
            if validator:
                cfg_level, cfg_tol = validator.get_verification_config(channel, value)
                if level is None and cfg_level:
                    level = cfg_level
                if cfg_tol is not None:
                    tolerance = cfg_tol
            # Omission is a sentinel, not a value: forwarding None would override
            # a legacy custom connector's own declared default.
            write_kwargs: dict = {}
            if level is not None:
                write_kwargs["verification_level"] = level
            if tolerance is not None:
                write_kwargs["tolerance"] = tolerance
            wr = await connector.write_channel(channel, value, **write_kwargs)
            connector_results.append(wr)
        else:
            # A batch carries one scalar level for every channel in it, so an
            # omitted level is not resolved here: leaving the keyword off lets
            # each channel resolve its own entry exactly as a single write does.
            write_ops = [(op["channel"], op["value"]) for op in operations]
            batch_kwargs: dict = {}
            if verification_level is not None:
                batch_kwargs["verification_level"] = verification_level
            connector_results = await connector.write_multiple_channels(write_ops, **batch_kwargs)

        for op, wr in zip(operations, connector_results, strict=True):
            result_entry = {
                "channel": wr.channel_address,
                "value_written": wr.value_written,
                "success": wr.success,
                "error_message": wr.error_message,
                "blocked": wr.blocked,
                "refusal_reason": wr.refusal_reason,
            }
            result_entry[WRITE_STATE_KEY] = _derive_write_state(wr)
            # "is not None", matching _derive_write_state: absence of a
            # verification object is what "verification_not_reported" means, and
            # the two must not disagree about which results have one.
            verification = getattr(wr, "verification", None)
            if verification is not None:
                result_entry["verification"] = {
                    "level": getattr(verification, "level", None),
                    "verified": getattr(verification, "verified", None),
                    "readback_value": getattr(verification, "readback_value", None),
                    "tolerance_used": getattr(verification, "tolerance_used", None),
                    # Machine-readable half of the verification. Absent fields
                    # stay null: "not reported" is deliberately distinct from a
                    # reported healthy severity of 0.
                    "readback_alarm_status": getattr(verification, "readback_alarm_status", None),
                    "readback_alarm_severity": getattr(
                        verification, "readback_alarm_severity", None
                    ),
                    "failure_kind": getattr(verification, "failure_kind", None),
                    "notes": getattr(verification, "notes", None),
                }
            if op.get("notes"):
                result_entry["notes"] = op["notes"]
            results_serialised.append(result_entry)

        # Build compact summary inline
        successful = sum(1 for r in results_serialised if r["success"])
        refused = sum(1 for r in results_serialised if r.get("blocked"))
        # A write that never executed is not a verification failure: only count
        # writes that succeeded, asked for verification, and did not get it.
        verification_failed = sum(
            1
            for r in results_serialised
            if r["success"]
            and (r.get("verification") or {}).get("level") in ("callback", "readback")
            and not (r.get("verification") or {}).get("verified")
        )
        summary = {
            "total_writes": len(results_serialised),
            "successful": successful,
            "failed": len(results_serialised) - successful,
            "refused": refused,
            "verification_failed": verification_failed,
            "results": [
                {
                    "channel": r["channel"],
                    "value": r["value_written"],
                    "success": r["success"],
                    WRITE_STATE_KEY: r[WRITE_STATE_KEY],
                    "error": r.get("error_message"),
                    "blocked": r.get("blocked"),
                    "refusal_reason": r.get("refusal_reason"),
                    "verification": r.get("verification"),
                }
                for r in results_serialised
            ],
        }
        # The caller's value, or null when they left the level to the deployment.
        access_details = {"verification_level": verification_level}

        if results_serialised and successful == 0:
            failures = [wr for wr in connector_results if not wr.success]
            if failures and all(wr.blocked for wr in failures):
                # Every failed op was a policy refusal (never sent to the control
                # system): surface a typed write-refusal envelope, not internal_error.
                refused_channels = [wr.channel_address for wr in failures]
                first = failures[0]
                raise ChannelWriteBlockedError(
                    first.channel_address,
                    first.refusal_reason or "WRITES_DISABLED",
                    message=(
                        f"All {len(failures)} write(s) refused by the reference monitor: "
                        f"{', '.join(refused_channels)}"
                    ),
                )
            # At least one failure was an attempted caput that failed (I/O failure,
            # not a policy refusal): preserve the internal_error classification.
            errors = sorted(
                {r["error_message"] for r in results_serialised if r.get("error_message")}
            )
            raise RuntimeError(
                f"All {len(results_serialised)} write(s) rejected: {'; '.join(errors)}"
            )

        # Agent-activity highlight (purely additive, after the fact): name only
        # the channels whose writes actually executed. Every refusal path —
        # limits_violation validation, all-blocked, all-failed — returns or
        # raises before this point, so those emit nothing. notify_agent_activity
        # never raises; the blocking call runs off the event loop.
        executed_channels = [r["channel"] for r in results_serialised if r["success"]]
        if executed_channels:
            await notify_agent_activity_async(
                "channel_write", "channel", detail=", ".join(executed_channels)
            )

        # Return ephemeral result (no persistent storage for channel writes)
        return json.dumps(
            {
                "status": "success",
                "description": f"Wrote {len(results_serialised)} channel(s)",
                "summary": summary,
                "access_details": access_details,
            },
            default=str,
        )
