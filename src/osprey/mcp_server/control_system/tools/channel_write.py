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

#: The key each result carries its outcome word under. The generated safety
#: rules name this key, so the spelling here and the spelling there have to stay
#: one string.
OUTCOME_KEY = "outcome"

#: The outcomes on which no value reached the channel. Everything else did put a
#: value on the wire, whether or not it was confirmed afterwards, and is
#: reported as an executed write.
UNEXECUTED_OUTCOMES = frozenset({"refused", "failed"})


def _project_observed_value(value: object) -> object:
    """One observed reading, bounded by the same budget a read is bounded by.

    A confirming re-read of a waveform channel returns the whole waveform, and
    an agent that asked to set one number must not be handed ten thousand back.
    Anything over ``control_system.read_inline_max_elements`` is described
    instead of shown, in the key shape ``channel_read`` already uses for a
    withheld value, so an agent meets one shape for "too big to show you".
    ``str``/``bytes`` are one channel value however long they are, and are
    always inline: summarising a string would lose the only thing it says.
    """
    if value is None or isinstance(value, str | bytes):
        return value

    from osprey.mcp_server.control_system.tools.channel_read import (
        ARTIFACT_REASON_PER_VALUE,
        _array_summary,
        _numpy,
        get_read_inline_max_elements,
    )

    np = _numpy()
    if np is None:
        return value
    try:
        array = np.asarray(value)
        if array.ndim == 0 or array.size <= get_read_inline_max_elements():
            return array.tolist() if isinstance(value, np.ndarray) else value
        return _array_summary(array, np, ARTIFACT_REASON_PER_VALUE)
    except Exception:  # noqa: BLE001 - a reading that cannot be measured is reported as it came
        logger.debug("Could not measure an observed value for projection", exc_info=True)
        return value


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


def _approval_stamp_key(operations: list[dict], confirm: bool | None) -> str | None:
    """The name the approval hook filed this write's stamp under, or ``None``.

    A SHA-256 over the canonical JSON of the write's own arguments —
    ``operations`` and ``confirm``, which are every parameter the tool accepts
    and exactly what the hook finds in the ``tool_input`` it is handed. The hook
    and this server share no call identifier — the hook is handed a tool-call
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
            {"operations": operations, "confirm": confirm},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):  # pragma: no cover - arguments arrive as JSON
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _stamp_is_ours(record: object) -> bool:
    """Whether *record* is an approval stamp this server's own prompt filed.

    A missing or ``None`` ``server_pid`` counts as ours: a project rendered
    before a target was published stamps without one, and there is nothing in
    such a stamp to attribute it elsewhere. Both callers ask this one question,
    so a stamp the lookup accepts as this server's cannot be invisible to the
    warning that has to explain a miss — which is exactly the gap that would
    let a never-published project's stale key go unreported.
    """
    if not isinstance(record, dict):
        return False
    server_pid = record.get("server_pid")
    return server_pid is None or server_pid == os.getpid()


def _warn_if_this_server_has_other_stamps() -> None:
    """Warn once per miss when stamps this server rendered exist under other keys.

    Both sides of the key derivation have to spell the payload the same way, and
    a project rendered before the hashed arguments last changed files its stamps
    under the old spelling. Every lookup then misses, and the approval-window
    check goes quiet without a single failure — the one failure mode of a
    two-party hash that nothing else would surface.

    Stamps left by *this* process's prompts are the evidence that separates the
    two explanations of a miss: if the hook is stamping for this server and none
    of its stamps is the one this call asked for, the derivations disagree. A
    stamp from another pid belongs to a second session sharing the state
    directory and says nothing about this one, so :func:`_stamp_is_ours` filters
    it out first — without that filter, an ordinary two-session checkout would
    warn on every unstamped write.
    """
    try:
        stamps = list(
            target_state.state_dir().glob(f"{APPROVAL_STAMP_PREFIX}*{APPROVAL_STAMP_SUFFIX}")
        )
    except Exception:  # pragma: no cover - defensive: reading must not fail a write
        logger.debug("Could not list the write-approval stamps", exc_info=True)
        return
    ours = []
    for path in stamps:
        try:
            record = target_state.read_file(path)
        except Exception:  # pragma: no cover - defensive
            continue
        if _stamp_is_ours(record):
            ours.append(path.name)
    if not ours:
        return
    logger.warning(
        "No approval stamp matches this write, but this server has %d other stamp(s) (%s). "
        "The approval hook and this server are deriving different keys — most likely the "
        "project was rendered before the stamp key changed. Re-run 'osprey build' in the "
        "project, or the target-binding check on the approval window is skipped.",
        len(ours),
        ", ".join(sorted(ours)[:3]),
    )


def _read_approval_stamp(
    operations: list[dict], confirm: bool | None
) -> tuple[bool, tuple[str, int] | None]:
    """``(found, binding)`` for the prompt that approved this write.

    ``found`` is ``False`` for every case in which no comparison may be made:
    no stamp, an unreadable one, or one another server on this checkout wrote —
    two sessions sharing a directory would otherwise cross-check each other's
    approvals, and a stamp whose ``server_pid`` is not this process is somebody
    else's prompt. ``found`` with a ``None`` binding is a real answer: the
    prompt was rendered while nothing published a target.
    """
    key = _approval_stamp_key(operations, confirm)
    if key is None:
        return False, None
    try:
        path = target_state.state_dir() / f"{APPROVAL_STAMP_PREFIX}{key}{APPROVAL_STAMP_SUFFIX}"
        stamp = target_state.read_file(path)
    except Exception:  # pragma: no cover - defensive: reading must not fail a write
        logger.debug("Could not read the write-approval stamp", exc_info=True)
        return False, None
    if not isinstance(stamp, dict):
        _warn_if_this_server_has_other_stamps()
        return False, None
    if not _stamp_is_ours(stamp):
        logger.debug(
            "Ignoring a write-approval stamp written for server pid %s", stamp.get("server_pid")
        )
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
    operations: list[dict], confirm: bool | None, entry: tuple[str, int] | None
) -> None:
    """Compare what the approval prompt showed against what the tool entered on."""
    found, approved = _read_approval_stamp(operations, confirm)
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
    confirm: bool | None = None,
) -> str:
    """Write values to one or more control-system channels.

    Each operation is a dict with keys: channel (str), value (any), notes (str, optional).
    PreToolUse hooks handle human approval BEFORE this code runs.

    Every result carries one `outcome` word, decided by the connector that
    performed the write. Report the outcome that word names and nothing
    stronger — a write that was not confirmed is not a confirmed write, and you
    must not describe it as one. The six words:

    - `refused` — nothing was written. Either OSPREY refused it on policy or
      limits grounds and never sent it to the machine, or the control system
      itself denied the write (`refusal_reason` `CONTROL_SYSTEM_REFUSED`).
    - `failed` — sent to the machine, which did not take it.
    - `confirmed` — a re-read of the channel holds the value sent. An alarm on
      the channel does not change that: report `alarm_status` and
      `alarm_severity` beside the confirmation, because a confirmed value in a
      MAJOR alarm is worth telling the operator about.
    - `mismatch` — the re-read holds a different value. `observed_value` is what
      the channel actually holds; a clamped or rounded setpoint appears here.
    - `unconfirmed` — sent, but the re-read itself failed, so what the channel
      holds now is unknown.
    - `unrequested` — confirmation is switched off for this channel; the value
      was sent and nothing was checked.

    `summary.outcomes` counts the words this call produced.

    Leave `confirm` unset unless the operator asks for it: the deployment
    resolves it per channel from the limits database, which confirms by default.
    This tool does not raise on a write that was not confirmed — it reports the
    word. (`osprey.runtime.write_channel`, on the Python path, raises.) It
    raises only when every write in the call came back `refused` or `failed`,
    which is the one case where no value reached any channel at all. A call
    mixing those with any other word returns normally, and the per-channel
    `outcome` is where you read what happened to each one.

    Args:
        operations: List of write operations, each with "channel", "value", and optional "notes".
        confirm: Optional override — re-read each channel and compare (true), or
            send without checking (false). Omit it to let the deployment decide.

    Returns:
        JSON with per-operation results. Each `summary.results[]` entry carries
        `outcome`, the `observed_value` the confirming re-read returned, the
        alarm name and severity when the connector reported them, and
        `refusal_reason` / `error` on the words that carry one. An
        `observed_value` too large to inline is reported as a bounded summary
        instead, exactly as `channel_read` reports one.
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
    # `confirm` is part of the stamp's identity because it is part of the write:
    # the hook reads it out of the same `tool_input` the operator was shown, so
    # both sides hash the same two arguments and a write approved with one
    # confirmation setting cannot be vouched for by a prompt that showed another.
    _check_approval_window(operations, confirm, entry_binding)

    # Limits validation (additional safety layer inside the tool)
    try:
        from osprey.connectors.control_system.limits_validator import LimitsValidator
    except ImportError:
        LimitsValidator = None  # type: ignore[assignment,misc]

    validator = None
    if LimitsValidator is not None:
        # The posture the session's target runs under, not the deployment's. A
        # deployment may relax unlisted channels for its simulator alone, and
        # the binding captured at entry is already the answer to "which machine
        # is this write for" — reading the state file a second time here could
        # only disagree with the binding every other check in this tool uses.
        # No binding means no published target, which resolves the
        # deployment-wide block, exactly as this call did before.
        validator = LimitsValidator.from_config(target=entry_binding[0] if entry_binding else None)

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

        # Omission is a sentinel, not a value: forwarding None would override a
        # deployment's own per-channel setting, and an `if confirm:` guard would
        # swallow an explicit False and confirm a write nobody asked to confirm.
        write_kwargs: dict = {}
        if confirm is not None:
            write_kwargs["confirm"] = confirm

        if len(operations) == 1:
            op = operations[0]
            wr = await connector.write_channel(op["channel"], op["value"], **write_kwargs)
            connector_results = [wr]
        else:
            write_ops = [(op["channel"], op["value"]) for op in operations]
            connector_results = await connector.write_multiple_channels(write_ops, **write_kwargs)

        # One result per operation, checked before a single row is projected.
        # The envelope reports exactly the rows the connector handed back, so a
        # connector that dropped one would produce a complete-looking report in
        # which a channel the operator approved simply is not mentioned — and on
        # the hardware-write surface, a write whose fate goes unreported is the
        # one thing worse than a write that failed. Fail loudly, naming the
        # connector, instead of shipping a plausible envelope.
        if len(connector_results) != len(operations):
            raise RuntimeError(
                f"{type(connector).__name__} returned {len(connector_results)} write result(s) "
                f"for {len(operations)} operation(s): the fate of at least one write is "
                f"unreported, so no outcome from this call can be trusted."
            )

        # One list, built once: what the agent reads is what this tool keeps, so
        # a field cannot be present in the bookkeeping and missing from the
        # response.
        results = [
            {
                "channel": wr.channel_address,
                "value": wr.value_written,
                OUTCOME_KEY: str(wr.outcome),
                "refusal_reason": wr.refusal_reason,
                "error": wr.error_message,
                "observed_value": _project_observed_value(wr.observed_value),
                # Absent alarm fields stay null: "not reported" is deliberately
                # distinct from a reported healthy severity of 0.
                "alarm_status": wr.alarm_status,
                "alarm_severity": wr.alarm_severity,
                "notes": wr.notes,
            }
            for wr in connector_results
        ]

        outcomes: dict[str, int] = {}
        for entry in results:
            outcomes[entry[OUTCOME_KEY]] = outcomes.get(entry[OUTCOME_KEY], 0) + 1

        summary = {
            "total_writes": len(results),
            "outcomes": outcomes,
            "results": results,
        }
        # The caller's value, or null when they left the decision to the deployment.
        access_details = {"confirm": confirm}

        if results and all(entry[OUTCOME_KEY] in UNEXECUTED_OUTCOMES for entry in results):
            # Nothing reached a channel. Anything else — a value that landed
            # without being confirmed, or one that came back different — is
            # reported rather than raised: the agent needs it to tell the
            # operator what the machine now holds.
            failed = [entry for entry in results if entry[OUTCOME_KEY] == "failed"]
            if not failed:
                # Every op was a refusal that put no value on a channel: surface
                # a typed write-refusal envelope, not internal_error. Who refused
                # decides the wording — a control-system denial was sent and
                # turned down, so calling it a reference-monitor policy refusal
                # would send the operator to the wrong place.
                first = results[0]
                reason = first["refusal_reason"] or "WRITES_DISABLED"
                refuser = (
                    "the control system"
                    if reason == "CONTROL_SYSTEM_REFUSED"
                    else "the reference monitor"
                )
                headline = (
                    f"All {len(results)} write(s) refused by {refuser}: "
                    f"{', '.join(entry['channel'] for entry in results)}"
                )
                # The per-result `error` is the only place the refusal says
                # WHICH posture refused and where to lift it — the session's
                # read-only setting for this control target and the header chip
                # that set it, or the config key for a deployment refusal. A
                # raised envelope replaces the results the agent would have
                # read, so the headline alone would drop that sentence on the
                # floor for exactly the common case: one channel, one refusal.
                detail = first["error"]
                raise ChannelWriteBlockedError(
                    first["channel"],
                    reason,
                    message=f"{headline} — {detail}" if detail else headline,
                )
            # At least one write was attempted and failed (an I/O failure, not a
            # policy refusal): preserve the internal_error classification.
            errors = sorted({entry["error"] for entry in results if entry["error"]})
            raise RuntimeError(f"All {len(results)} write(s) rejected: {'; '.join(errors)}")

        # Agent-activity highlight (purely additive, after the fact): name only
        # the channels a value actually reached. Every refusal path — limits
        # violation, all-refused, all-failed — returns or raises before this
        # point, so those emit nothing. notify_agent_activity never raises; the
        # blocking call runs off the event loop.
        executed_channels = [
            entry["channel"] for entry in results if entry[OUTCOME_KEY] not in UNEXECUTED_OUTCOMES
        ]
        if executed_channels:
            await notify_agent_activity_async(
                "channel_write", "channel", detail=", ".join(executed_channels)
            )

        # Return ephemeral result (no persistent storage for channel writes)
        return json.dumps(
            {
                "status": "success",
                "description": f"Wrote {len(results)} channel(s)",
                "summary": summary,
                "access_details": access_details,
            },
            default=str,
        )
