"""Shared wording for holders that stay pinned to the deployment baseline.

Most of the system follows the session's control-system target. A few holders
cannot, because they are bound to something the switch does not move:

* the **Phoebus bridge** talks to one running Phoebus product, whose PV context
  was established when that product started — a session-level target switch
  does not re-address it;
* the **health runtime** reports on the deployment as configured, not on
  whatever a session has selected for itself.

A holder in that position must never be silent about it. Two agent-facing
strings come out of this module, and both are rendered from the same computed
facts so a refusal and a label can never disagree:

* :func:`baseline_pinned_line` — the informational line a *read* tool prepends
  to its normal output while the session is switched away from the baseline;
* :func:`baseline_refusal` — the message + suggestions an *action* tool refuses
  with, so a write is never quietly applied to the target the session left.

Both render nothing (``None``) while the session is on the baseline, which is
what keeps unswitched output byte-identical to what it was before this module
existed.

Why this module lives here
--------------------------
:mod:`osprey.mcp_server.control_system.target_state` owns the state-file
contract, and this module is the one place that turns that record plus the
deployment config into the sentence a user reads. Keeping the two together
means there is exactly one in-venv answer to "which target is this session on,
and which one is this deployment's baseline". The phoebus MCP server and the
health runtime are each a *different process* from the controls server that
writes the state file; both import this module rather than restating the rule.
Claude Code hooks, which run outside the venv and cannot import any of this,
restate it stdlib-only — see the ``target_state`` docstring for that contract.

Reuse
-----
The API is deliberately holder-agnostic so the HealthRuntime row can be built
from it without a second implementation:

* :func:`resolve_target_situation` returns the three facts
  (``session_target`` / ``baseline_target`` / ``switched``) and never raises;
* every renderer takes a *subject* (``"Phoebus"``, ``"HealthRuntime"``, …) and
  an optional pre-computed :class:`TargetSituation`, so a caller that needs
  several strings resolves the state once and renders many.

A holder that wants different phrasing than :func:`baseline_pinned_line` should
still take its facts from :func:`resolve_target_situation` — re-deriving the
session target from config is how two holders start telling a user two
different things.

One caller is outside the session entirely: the web terminal's posture badge
asks which target a PTY it spawned is on. :func:`session_target_for_pid`
answers that, over the same :func:`_live_records` liveness filter, by walking
the ancestors of each record's ``owner_ppid`` instead of the caller's own — see
its docstring for why equality is not enough.
:func:`session_target_meta_for_pid` answers the same question with the record's
display metadata attached, off the same match, so the badge names a target with
the label its writer minted rather than deriving a second one from config.
:func:`session_record_for_pid` is the widest of the three: it hands back the
whole matched record, so a surface that needs several of its published facts at
once — the target, its label, the switch outcome, the reachability sweep — takes
them from ONE match rather than resolving the process table once per fact.

Failure posture
---------------
Every failure mode collapses to "on the baseline": absent, unreadable, or
corrupt state; a state file owned by another session; two ambiguous candidates;
an unreadable config. That means a broken read produces no refusal and no
label rather than a wrong one — the same fail-closed direction the state file's
own readers take.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.target_state import TARGET_LIVE
from osprey_connectors import types as connector_types
from osprey_connectors.workspace import load_osprey_config

logger = logging.getLogger("osprey.mcp_server.control_system.target_banner")

#: Subject string for the phoebus holder. Spelled once so the refusal and the
#: read-tool label cannot drift apart across two modules.
PHOEBUS_SUBJECT = "Phoebus"

#: Error type carried by a baseline-pinned refusal envelope. Machine-readable
#: category shared by every holder that refuses for this reason, so a caller can
#: recognise "you are switched away" without matching on prose.
BASELINE_REFUSAL_ERROR_TYPE = "target_switched"

#: Bound on the ancestor walk :func:`session_target_for_pid` runs. A process
#: tree deeper than this, or one with a cycle, is pathological; stopping early
#: yields no match, which is the fail-closed answer.
MAX_ANCESTOR_HOPS = 32

#: Seconds to wait for the ``ps`` fallback. ``ps -o ppid= -p <pid>`` is a table
#: lookup that returns in milliseconds on a healthy machine, and the badge polls
#: the route that calls this every few seconds per open card — so the budget is
#: sized for "the process table is wedged, give up and report the baseline"
#: rather than for a slow answer worth waiting on. Deliberately shorter than the
#: hook reader's five seconds: a hook blocks one agent turn that has already
#: decided to do work, this blocks a worker thread serving a status badge.
PS_TIMEOUT_S = 1

__all__ = [
    "BASELINE_REFUSAL_ERROR_TYPE",
    "MAX_ANCESTOR_HOPS",
    "PHOEBUS_SUBJECT",
    "TargetSituation",
    "baseline_pinned_line",
    "baseline_refusal",
    "prepend_line",
    "resolve_baseline_target",
    "resolve_session_target",
    "resolve_target_situation",
    "session_record_for_pid",
    "session_target_for_pid",
    "session_target_meta_for_pid",
]


@dataclass(frozen=True)
class TargetSituation:
    """The two targets a baseline-pinned holder has to talk about.

    Attributes:
        session_target: The target this session has selected (``live`` / ``va``).
        baseline_target: The target the deployment config declares.
    """

    session_target: str
    baseline_target: str

    @property
    def switched(self) -> bool:
        """Whether the session has moved off the deployment baseline."""
        return self.session_target != self.baseline_target


# -- resolution ------------------------------------------------------------


def resolve_baseline_target() -> str:
    """The deployment baseline: ``va`` for a virtual accelerator, else ``live``.

    The mapping comes from :func:`osprey_connectors.types.baseline_target` —
    the same predicate the connector-host supervisor and the switch-capability
    check read. Re-implementing it here would be a second opinion about what the
    deployment is, which is exactly the bug this module exists to prevent.
    """
    config = load_osprey_config()
    section = config.get("control_system") if isinstance(config, dict) else None
    return connector_types.baseline_target(section)


def _int_or_none(value: object) -> int | None:
    """Coerce a record field to ``int``, or ``None`` when it is not a number."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _live_records() -> list[dict]:
    """Every state record whose owning server process is still running."""
    try:
        entries = sorted(target_state.state_dir().glob(target_state.STATE_FILE_GLOB))
    except OSError:
        return []

    records: list[dict] = []
    for entry in entries:
        record = target_state.read_file(entry)
        if record is None:
            continue
        server_pid = _int_or_none(record.get("server_pid"))
        if server_pid is None or not target_state.is_process_alive(server_pid):
            continue
        records.append(record)
    return records


def resolve_session_target(baseline_target: str) -> str:
    """The target this session is on, or *baseline_target* when it is unknowable.

    A checkout can host several sessions at once, each with its own controls
    server and so its own state file. This process is not that server — the
    phoebus server and the health runtime are separate MCP processes — but it
    *is* a child of the same Claude Code process, so the record to trust is the
    one whose ``owner_ppid`` is this process's parent.

    Zero matches (no session has switched, or the switch happened under a
    different parent) and more than one match (ambiguous ownership) both mean
    the same thing: no answer. Both fall back to the baseline, so an unknown
    state produces no refusal and no label rather than a guess.
    """
    own_parent = os.getppid()
    matches = [r for r in _live_records() if _int_or_none(r.get("owner_ppid")) == own_parent]
    if len(matches) != 1:
        if matches:
            logger.debug("Ambiguous target state: %d records own ppid %s", len(matches), own_parent)
        return baseline_target
    target = matches[0].get("target")
    if target not in target_state.TARGET_NAMES:
        logger.debug("Target state names an unknown target %r; using baseline", target)
        return baseline_target
    return str(target)


# -- resolution for a process we are NOT inside ----------------------------


def _ppid_from_proc(pid: int) -> int | None:
    """Parent PID from ``/proc/<pid>/stat`` (Linux), or ``None``.

    The ``comm`` field is parenthesized and may itself contain spaces and
    parentheses, so the fields are taken after the LAST ``)``: they begin with
    ``state`` and ``ppid``.
    """
    try:
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8", errors="replace") as handle:
            data = handle.read()
    except (OSError, TypeError, ValueError):
        return None
    try:
        fields = data[data.rindex(")") + 1 :].split()
        return int(fields[1])
    except (ValueError, IndexError):
        return None


def _ppid_from_ps(pid: int) -> int | None:
    """Parent PID from ``ps -o ppid= -p <pid>`` (macOS and any POSIX), or ``None``.

    ``ps`` is the only portable answer where ``/proc`` does not exist. Every
    failure — missing binary, non-zero exit, timeout, unparseable output — is
    simply "no parent".
    """
    try:
        completed = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(int(pid))],
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT_S,
            check=False,
        )
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return int((completed.stdout or "").strip())
    except (AttributeError, ValueError):
        return None


def _parent_pid(pid: int) -> int | None:
    """Parent of *pid*, or ``None`` when the chain cannot be walked further.

    ``/proc`` first because it is a file read rather than a process spawn; the
    ``ps`` fallback carries macOS, where ``/proc`` does not exist and the first
    attempt fails immediately and cheaply. No ``psutil``: the framework does not
    depend on it, and this is the whole of what it would have been used for.

    The rule is deliberately the same one the stdlib-only hook reader states
    (``templates/claude_code/claude/hooks/osprey_target_state``) — hooks cannot
    import this module, so the two restatements are the drift the hook's
    docstring warns about, and they must stay word-for-word the same rule.
    """
    ppid = _ppid_from_proc(pid)
    if ppid is None:
        ppid = _ppid_from_ps(pid)
    return ppid


def _ancestor_pids(
    start_pid: int,
    max_hops: int = MAX_ANCESTOR_HOPS,
    *,
    stop_at: int | None = None,
    parent_of: Callable[[int], int | None] | None = None,
) -> list[int]:
    """The ancestor chain of *start_pid*, nearest first, INCLUDING itself.

    Including *start_pid* is what makes the common case fall out for free: when
    the PTY child is ``claude`` itself, the controls server's ``owner_ppid``
    *is* the PTY pid and needs no walk at all.

    The walk stops at PID 1, at *max_hops*, on a repeat (a cycle can only come
    from a lying process table), or the moment a parent cannot be determined.
    Any failure mid-walk simply ends the chain — a short chain yields no match,
    which is the fail-closed answer.

    Args:
        start_pid: Where to start. Appears first in the result.
        max_hops: Hard bound on the walk.
        stop_at: Stop the moment this pid is reached, without asking for its
            parent. The caller is asking "is *stop_at* an ancestor of
            *start_pid*", and every hop past the answer is a syscall — a fork of
            ``ps`` on a platform without ``/proc`` — spent learning nothing.
        parent_of: Parent lookup to use instead of :func:`_parent_pid`. The
            seam a caller walking several chains uses to share one memo across
            them; they converge on the same upper ancestors.
    """
    try:
        current = int(start_pid)
    except (TypeError, ValueError):
        return []
    lookup = _parent_pid if parent_of is None else parent_of

    chain: list[int] = []
    for _ in range(max(0, int(max_hops))):
        if current <= 1:
            break
        chain.append(current)
        if stop_at is not None and current == stop_at:
            break
        parent = lookup(current)
        if parent is None or parent <= 1 or parent in chain:
            break
        current = parent
    return chain


def _matched_record(pty_pid: object) -> dict | None:
    """The one live state record published from inside process *pty_pid*.

    The shared half of every pid-side accessor — :func:`session_record_for_pid`
    is its public face, and :func:`session_target_for_pid` and
    :func:`session_target_meta_for_pid` are narrowings of what it returns.
    The match, the cost discipline and the fail-closed rules are spelled once
    here so the name a badge shows and the metadata it shows beside it can never
    come from two different records. See
    :func:`session_target_for_pid` for why the match runs from each record's
    ``owner_ppid`` outwards rather than by equality, and why an ambiguous answer
    is no answer.

    Returns:
        The matched record, or ``None`` — for a pid that is not a pid, zero
        matches, more than one match, or a record naming a target this build
        does not know. Never raises.
    """
    pid = _int_or_none(pty_pid)
    if pid is None or pid <= 0:
        return None

    try:
        records = _live_records()
    except Exception:  # pragma: no cover - _live_records is defensive already
        logger.debug("Could not list live target-state records", exc_info=True)
        return None

    memo: dict[int, int | None] = {}

    def parent_of(child: int) -> int | None:
        if child not in memo:
            memo[child] = _parent_pid(child)
        return memo[child]

    matches: list[dict] = []
    for record in records:
        owner_ppid = _int_or_none(record.get("owner_ppid"))
        if owner_ppid is None or owner_ppid <= 0:
            continue
        try:
            chain = _ancestor_pids(owner_ppid, stop_at=pid, parent_of=parent_of)
        except Exception:  # pragma: no cover - process table oddity
            logger.debug("Could not walk the ancestors of pid %s", owner_ppid, exc_info=True)
            continue
        if pid in chain:
            matches.append(record)

    if len(matches) != 1:
        if matches:
            logger.debug("Ambiguous target state: %d records run inside pid %s", len(matches), pid)
        return None

    target = matches[0].get("target")
    if target not in target_state.TARGET_NAMES:
        logger.debug("Target state names an unknown target %r; no answer", target)
        return None
    return matches[0]


def session_record_for_pid(pty_pid: object) -> dict[str, Any] | None:
    """The WHOLE live state record published from inside process *pty_pid*.

    The public face of :func:`_matched_record`, and the widest of the three
    pid-side accessors. :func:`session_target_for_pid` answers *which* target
    and :func:`session_target_meta_for_pid` answers *how to name it*; this hands
    back everything its writer published — ``target``, the per-target
    ``targets`` metadata block, ``server_pid``, ``owner_ppid``, ``generation``,
    and whichever of ``last_switch``, ``reachability`` and
    ``last_posture_realign`` that writer has recorded so far.

    Those last three are published on the writer's own schedule, so a reader
    must treat every one of them as optional (``record.get(...)``): a record
    written by a server that has not switched, probed or realigned yet simply
    does not carry them, and neither does one written by an older build. An
    absent key means "not recorded", never "no".

    Why a caller wants the record rather than one derived answer: resolving is
    the expensive half — a scan of the state directory plus an ancestor walk
    that, on a platform without ``/proc``, forks ``ps`` — and a surface that
    renders several published facts at once would otherwise pay for it once per
    fact. Worse, it would open a window in which those facts came from
    different records: a switch landing mid-render would put one target's name
    beside another target's reachability. One match, one record, one story.

    The returned dict is the freshly-parsed record and is not shared with any
    other caller, so a caller may keep it; a caller that *caches* it owns
    deciding when it has gone stale (see the web terminal's per-session memo).

    Args:
        pty_pid: The pid of the PTY process the session runs in, exactly as for
            :func:`session_target_for_pid`.

    Returns:
        The matched record, or ``None`` — no pid, zero matches, more than one
        match, or a record naming a target this build does not know. Never
        raises.
    """
    return _matched_record(pty_pid)


def session_target_for_pid(pty_pid: object) -> str | None:
    """The target of the session running inside process *pty_pid*, or ``None``.

    :func:`resolve_session_target` answers for the session the CALLER is in, by
    matching ``owner_ppid`` against ``os.getppid()``. The web terminal is not in
    that session at all: it is the server that spawned the PTY, and the only
    handle it has is the PTY's own pid. So the match runs the other way round —
    walk the ancestors of each record's ``owner_ppid`` and ask whether the PTY
    pid is on that chain.

    Equality alone would not do. ``owner_ppid`` is the Claude Code process that
    spawned the controls server, which is the PTY child itself only when the
    shell command is plain ``claude``; with ``claude_code.cli_version`` pinned
    the PTY child is ``npx`` and Claude Code is its child, and with a custom
    ``shell`` it can be deeper still. Equality-only matching would report "no
    session target" on every one of those deployments — silently, and exactly
    where the operator most needs the badge to be right.

    Zero matches and more than one match both mean the same thing: no answer.
    The caller (not this function) decides what to show instead, which for the
    badge is the deployment baseline — the same fail-closed direction the hook
    reader and :func:`resolve_session_target` take.

    Args:
        pty_pid: The pid of the PTY process the session runs in. Anything that
            is not a positive integer — ``None`` from a session that has not
            started, most of all — is simply "no answer".

    Cost. Each hop is a syscall, and on a platform without ``/proc`` a fork of
    ``ps``, so the walk is kept as short as the question allows: it stops the
    moment *pty_pid* is reached rather than continuing to init, and one memo of
    parent lookups is shared across every record's walk — several sessions in
    one checkout have different ``owner_ppid`` leaves but converge on the same
    upper ancestors within a hop or two. The caller must still keep this off the
    event loop; ``routes/websocket.py`` runs it in a worker thread.

    Returns:
        ``live`` or ``va``, or ``None``. Never raises.
    """
    record = session_record_for_pid(pty_pid)
    if record is None:
        return None
    return str(record.get("target"))


def session_target_meta_for_pid(pty_pid: object) -> dict[str, Any] | None:
    """The display metadata of the target process *pty_pid* is on, or ``None``.

    :func:`session_target_for_pid` answers *which* target; this answers *how to
    name it*. The state file's per-target metadata — ``label``, ``endpoint``,
    ``real_machine`` and, where one is configured, ``probe_channel`` — is
    rendered once by the controls server that wrote the record, and every
    reader shows what it is handed. That is the whole point of asking here
    rather than deriving a name from config: a deployment whose live target is a
    stand-in is labelled as one by its writer, and a badge that re-derived the
    label from ``config.yml`` would be a second opinion about identity.

    The match is :func:`session_record_for_pid`, so this,
    :func:`session_target_for_pid` and any caller reading the whole record agree
    by construction: the same records, the same ancestor walk, and the same
    "zero or two-plus matches means no answer". This function is the narrowing
    of that record down to the metadata block; a caller that also needs the
    record's other published facts should take the record itself rather than
    resolve a second time.

    Args:
        pty_pid: The pid of the PTY process the session runs in, exactly as for
            :func:`session_target_for_pid`.

    Returns:
        The matched record's metadata for its own target, under a ``target`` key
        naming that target — or ``None`` when there is no answer. A record whose
        metadata block is missing or malformed still yields the target name: the
        caller gets one dict shape, and an absent key reads as "not recorded"
        rather than as a crash. Never raises.
    """
    record = session_record_for_pid(pty_pid)
    if record is None:
        return None

    target = str(record.get("target"))
    targets = record.get("targets")
    meta = targets.get(target) if isinstance(targets, dict) else None
    # ``target`` last: it is the validated name this record was matched on, and
    # a stray ``target`` key inside the metadata block must not be able to
    # rename the machine an operator is told they are pointed at.
    return {**(meta if isinstance(meta, dict) else {}), "target": target}


def resolve_target_situation() -> TargetSituation:
    """Resolve both targets. Never raises; every failure reads as "on baseline"."""
    try:
        baseline = resolve_baseline_target()
    except Exception:  # pragma: no cover - config layer is defensive already
        logger.debug("Could not resolve the deployment baseline target", exc_info=True)
        return TargetSituation(session_target=TARGET_LIVE, baseline_target=TARGET_LIVE)

    try:
        session = resolve_session_target(baseline)
    except Exception:
        logger.debug("Could not resolve the session target; assuming baseline", exc_info=True)
        session = baseline

    return TargetSituation(session_target=session, baseline_target=baseline)


# -- rendering -------------------------------------------------------------


def baseline_pinned_line(subject: str, situation: TargetSituation | None = None) -> str | None:
    """The informational line a baseline-pinned read tool prepends, or ``None``.

    ``None`` — not an empty string — while the session is on the baseline, so a
    caller cannot accidentally prepend a blank line to unswitched output.

    Args:
        subject: The holder speaking, e.g. ``"Phoebus"``.
        situation: Pre-resolved facts; resolved here when omitted.
    """
    situation = resolve_target_situation() if situation is None else situation
    if not situation.switched:
        return None
    return (
        f"{subject} is pinned to the deployment baseline "
        f"({situation.baseline_target}); the session target is {situation.session_target}"
    )


def baseline_refusal(
    subject: str,
    action: str,
    situation: TargetSituation | None = None,
) -> tuple[str, list[str]] | None:
    """Refusal message + suggestions for an action tool, or ``None`` on baseline.

    The message opens with the same sentence :func:`baseline_pinned_line`
    renders, so a user who has already seen the label on a read tool recognises
    the refusal as the same fact rather than a new one.

    Args:
        subject: The holder speaking, e.g. ``"Phoebus"``.
        action: What was refused, as a capitalised noun phrase — e.g.
            ``"Driving a Phoebus widget"``.
        situation: Pre-resolved facts; resolved here when omitted.

    Returns:
        ``(message, suggestions)``, or ``None`` when nothing is refused.
    """
    situation = resolve_target_situation() if situation is None else situation
    line = baseline_pinned_line(subject, situation)
    if line is None:
        return None
    message = (
        f"{line}. {action} would act on the '{situation.baseline_target}' target, "
        f"not the '{situation.session_target}' target this session is on."
    )
    suggestions = [
        f"Switch the session back to the deployment baseline: "
        f"control_target_set(target='{situation.baseline_target}').",
        f"Or act on the '{situation.session_target}' target through the control-system "
        f"tools, which follow the session target.",
    ]
    return message, suggestions


def prepend_line(line: str | None, payload: str) -> str:
    """Put *line* above *payload*, or return *payload* untouched when there is none.

    The untouched branch is the contract that keeps an unswitched tool's output
    byte-identical to what it produced before the holder was labelled.
    """
    return payload if not line else f"{line}\n{payload}"
