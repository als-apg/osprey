#!/usr/bin/env python3
# ruff: noqa: T201 -- stdout is this hook's reply to Claude Code
"""
---
name: Control Context
description: Puts the deployment's control context — the active target and each target's write state — in front of the agent at session start and whenever it moved since the agent's last turn
summary: The agent knows which machine the session is pointed at without asking, and learns of a switch or a posture change made under it before its next turn starts
event: SessionStart, UserPromptSubmit
---

## Flow

```
hook fires ──► read stdin JSON ──► session_id (absent/malformed ─► None)
                    │
                    ▼
               read the control record (osprey_target_state.read_record)
                    │
                    ├── no record ──► exit 0, silent (nothing to describe)
                    │
                    ▼
               state = {target, generation, writes per reachable target}
                    │
                    ▼
               SessionStart ─────────────────────────► emit the block, remember state
                    │
               UserPromptSubmit
                    │
                    ├── same as remembered ──► exit 0, silent
                    │
                    └── differs / nothing remembered ──► emit the block marked
                                                          (changed), with the
                                                          previous values, and
                                                          remember the new state
```

## Details

What the agent sees is harness state, not a message. It is the shape of a
lock file — fixed keys, one value each — and it is emitted only when there is
something the agent's own context does not already hold:

```
--- osprey control-context ---
target: va  gen: 2
writes: live=off va=asked
positions: off="the next write is refused until the chip moves the switch" asked="a person approves the next write" allowed="the login's switch position approves the next write; limits still apply"
machines: live="Real machine — writes move hardware" va="Simulator — nothing moves"
```

At a session start (``startup``, ``resume``, ``clear`` and ``compact`` alike:
a rebuilt context is a context that may have lost it) the block describes the
current state. On a prompt it is emitted only if the target, a write state or
a machine's words moved since the last emission for this session, marked
``(changed)`` and carrying the previous value beside each key that moved:

```
--- osprey control-context (changed) ---
target: live  gen: 3  (was va)
writes: live=allowed (was off) va=asked
positions: off="the next write is refused until the chip moves the switch" asked="a person approves the next write" allowed="the login's switch position approves the next write; limits still apply"
machines: live="Real machine — writes move hardware" va="Simulator — nothing moves"
```

An unchanged prompt injects nothing. Nothing in the block asks the agent to do
anything: the point is that a switch made from the terminal's chip, or a
position moved there, reaches the agent as a fact before its next
turn, so what it holds about the machine is never stale — and so it has no
reason to re-derive from the record what the harness already told it.

## The write states

``writes`` names every target a session here can reach
(``osprey_target_state.session_types``) by its raw key, each with the twin's
write verdict for this login (``osprey_target_state.write_verdict``):

* ``off`` — this login's switch for the machine is off (``switch_off``); the
  next write is refused until the chip moves the switch.
* ``asked`` — the verdict allows the write at this position: a person approves
  the next write.
* ``allowed`` — the verdict allows the write at this position: the login's
  switch position approves the next write, and limits still apply.
* ``refused:<reason>`` — any other refusal, named by its verdict reason
  (``tier_never``, ``readonly_run``, ``launch_pinned``, ...). In particular
  ``refused:record_unavailable`` when the switch record cannot be read: the
  block then says writes are refused, never that a machine is off, because an
  unreadable switch is not a switch someone turned off.

``asked`` and ``allowed`` are the verdict's own position, so a stored ``allowed``
under a tier capped at ``asked`` reads ``asked``, and a stored legacy ``on`` reads
``asked``. The ``positions`` line defines the three position words every time the
block is emitted, so the block says what its words mean on its own; it is fixed
text, not part of the remembered state, and never marks a change.

A record file that exists but cannot be read at all still emits — with
``target: unknown  gen: unknown`` and every machine ``refused:record_unavailable``
— because that is exactly the state the agent must not assume away. No record
file at all stays silent: there is nothing to describe.

## The machine words

``machines`` names each target by the words the operator sees for it, keyed by
the same raw key: ``"Real machine — writes move hardware"``, ``"Rehearsal —
nothing moves"`` or ``"Simulator — nothing moves"``. The kind comes from the
controls server's published report (``osprey_target_state.published_machine_kind``), so the
stand-in reads ``Rehearsal — nothing moves`` only while this deployment has it
stood up; a ``standin`` that is not reads as the real machine.

## Remembering what was emitted

The last emitted state is kept per Claude Code session id in the temp
directory (``osprey-control-context-<session_id>.json``), the way the panels
hooks keep their workspace snapshot. It is keyed by the session id from the
hook payload rather than by the deployment's session key, because a ``/clear``
gives the agent a new context and a new id together: the next prompt in it
finds nothing remembered and gets the full block. A payload without a usable
session id emits the block on every start and stays silent on prompts: with
nothing to remember against, "changed" cannot be answered, and repeating the
block on every prompt is the cost this hook exists to avoid.

## Fail-open

Every failure of the hook itself — an unwritable temp directory, a payload
that is not JSON, an exception anywhere — exits 0 with no output. (An
unreadable switch record is not a hook failure: the verdict answers it, and
the block says so.) This hook only describes; a turn
must never be held up by its absence.

## Saying which of those happened

Silence is this hook's normal outcome and it has five different meanings, so
every run ends with one ``log_hook`` record naming which: ``emit`` when a block
went out, ``skip:not-our-event``, ``skip:no-record``, ``skip:no-session`` and
``skip:unchanged`` for the four quiet ends, and ``error`` for a failure the
fail-open rule above swallowed. Debug-gated like every ``log_hook`` call, so it
costs a run nothing until someone turns it on.
"""

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import load_osprey_config, log_hook

try:
    import osprey_target_state as _target_state
except Exception:  # pragma: no cover - older render without the reader
    _target_state = None

# The stdlib-only file helper the memo is written through; without it no memo
# is kept, which costs one redundant context block on the next prompt.
try:
    import osprey_safety_file as _safety_file
except Exception:  # pragma: no cover - render without the helper
    _safety_file = None

#: The event that always emits: a context that has just been built or rebuilt.
_SESSION_START = "SessionStart"

#: The event that emits on change only.
_USER_PROMPT = "UserPromptSubmit"

#: What ``writes`` says about one target: the switch is off, the verdict allows
#: at its position (``asked`` or ``allowed``), or another verdict reason refuses
#: (``refused:<reason>``).
WRITES_OFF = "off"
WRITES_ASKED = "asked"
WRITES_ALLOWED = "allowed"
WRITES_REFUSED = "refused"

#: What each position means for the next write, as the ``positions`` line says it.
POSITION_MEANINGS = {
    WRITES_OFF: "the next write is refused until the chip moves the switch",
    WRITES_ASKED: "a person approves the next write",
    WRITES_ALLOWED: "the login's switch position approves the next write; limits still apply",
}

#: The verdict reason that reads as ``off`` rather than as a refusal.
_SWITCH_OFF_REASON = "switch_off"

#: The reason an unreadable record file refuses with.
_RECORD_UNAVAILABLE = "record_unavailable"

#: The owner-facing words for each machine kind (C-WORDS, D4). The stdlib copy
#: of ``osprey_connectors.write_target.KIND_BANNERS``: hooks import nothing from
#: the package, so a test pins the two.
MACHINE_WORDS = {
    "real": "Real machine — writes move hardware",
    "rehearsal": "Rehearsal — nothing moves",
    "simulator": "Simulator — nothing moves",
}

#: What the target line says when the record exists but cannot be read.
UNKNOWN = "unknown"

#: Session ids are Claude Code's own; only this alphabet reaches a filename.
_SESSION_ID_CHARS = frozenset("0123456789abcdefABCDEF-_")

#: The first line of the block, with and without the change mark.
HEADER = "--- osprey control-context ---"
HEADER_CHANGED = "--- osprey control-context (changed) ---"


def _read_payload(stream=None):
    """The hook's stdin JSON as a dict, or ``{}`` for anything else."""
    try:
        raw = (stream if stream is not None else sys.stdin).read()
        payload = json.loads(raw) if raw and raw.strip() else {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _session_id(payload):
    """The payload's ``session_id`` when it is safe to put in a filename."""
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    if not set(session_id) <= _SESSION_ID_CHARS:
        return None
    return session_id


def memo_path(session_id):
    """Where this session's last emitted state is kept."""
    return os.path.join(tempfile.gettempdir(), f"osprey-control-context-{session_id}.json")


def _writes_word(verdict):
    """The ``writes`` word for one twin verdict: its position when it allows."""
    if verdict.allowed:
        return verdict.position
    if verdict.reason == _SWITCH_OFF_REASON:
        return WRITES_OFF
    return f"{WRITES_REFUSED}:{verdict.reason}"


def _machine_words(hook_input, target, kind):
    """The owner-facing words for *target*, from its published kind."""
    if kind not in MACHINE_WORDS:
        kind = _target_state.published_machine_kind(target, hook_input=hook_input)
    return MACHINE_WORDS.get(kind, MACHINE_WORDS["real"])


def _record_unreadable(hook_input):
    """Whether a record file exists here but cannot be read."""
    path = _target_state.record_path(hook_input)
    if not path or not os.path.exists(path):
        return False
    _record, status = _target_state.read_record_status(hook_input)
    return status == "unreadable"


def current_state(hook_input):
    """``{target, generation, writes, machines}``, or ``None`` with no record.

    ``writes`` and ``machines`` are keyed by the raw target key. Each write
    word is the twin's verdict for this login; a record file that exists but
    cannot be read gives ``unknown`` identity fields and the verdict's
    ``record_unavailable`` — never ``off``.
    """
    if _target_state is None:
        return None
    record = _target_state.read_record(hook_input)
    if record is None and not _record_unreadable(hook_input):
        return None
    config = load_osprey_config(hook_input)
    section = config.get("control_system") if isinstance(config, dict) else None
    login = _target_state.acting_identity()
    writes = {}
    machines = {}
    for target in sorted(_target_state.session_types(section)):
        verdict = _target_state.write_verdict(section, target, owner=login, hook_input=hook_input)
        if record is None and verdict.reason == _SWITCH_OFF_REASON:
            # A record the reader rejected is not a record with the switch off.
            writes[target] = f"{WRITES_REFUSED}:{_RECORD_UNAVAILABLE}"
        else:
            writes[target] = _writes_word(verdict)
        machines[target] = _machine_words(hook_input, target, verdict.kind)
    return {
        "target": UNKNOWN if record is None else record.get("target"),
        "generation": UNKNOWN if record is None else record.get("generation"),
        "writes": writes,
        "machines": machines,
    }


def _remembered(session_id):
    """The state last emitted for *session_id*, or ``None``."""
    if session_id is None:
        return None
    try:
        with open(memo_path(session_id), encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _remember(session_id, state):
    """Keep *state* as this session's memo, replaced whole. Never raises."""
    if session_id is None or _safety_file is None:
        return
    with contextlib.suppress(Exception):
        _safety_file.atomic_replace(
            Path(memo_path(session_id)),
            json.dumps(state).encode("utf-8"),
            mode=0o600,
            sync="best_effort",
        )


def render_block(state, previous=None):
    """The lock-file block for *state*; with *previous*, marked and annotated."""
    changed = previous is not None
    lines = [HEADER_CHANGED if changed else HEADER]
    target_line = f"target: {state['target']}  gen: {state['generation']}"
    if changed and previous.get("target") != state["target"]:
        target_line += f"  (was {previous.get('target')})"
    lines.append(target_line)
    words = []
    old_writes = previous.get("writes", {}) if changed else {}
    for target, word in state["writes"].items():
        entry = f"{target}={word}"
        if changed and target in old_writes and old_writes[target] != word:
            entry += f" (was {old_writes[target]})"
        words.append(entry)
    lines.append("writes: " + " ".join(words))
    meanings = " ".join(f'{word}="{meaning}"' for word, meaning in POSITION_MEANINGS.items())
    lines.append(f"positions: {meanings}")
    names = []
    old_machines = previous.get("machines", {}) if changed else {}
    for target, label in state["machines"].items():
        entry = f'{target}="{label}"'
        if changed and target in old_machines and old_machines[target] != label:
            entry += f' (was "{old_machines[target]}")'
        names.append(entry)
    lines.append("machines: " + " ".join(names))
    return "\n".join(lines)


def _emit(event, block):
    """Print the block in the envelope Claude Code accepts for *event*.

    ``hookEventName`` is not decoration: the CLI validates the envelope against
    the schema for the named event and DROPS an output that omits it, reporting
    a non-blocking hook error into the transcript rather than into the hook's
    own stderr. An envelope without it costs the agent the whole block while
    the hook still exits 0 and still looks like it ran.
    """
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": block}}))


#: The status word for each way :func:`main` can end. Six outcomes: the one that
#: put a block in front of the agent, the four ordinary reasons there was nothing
#: to say, and the failure.
STATUS_EMIT = "emit"
STATUS_NOT_OUR_EVENT = "skip:not-our-event"
STATUS_NO_RECORD = "skip:no-record"
STATUS_NO_SESSION = "skip:no-session"
STATUS_UNCHANGED = "skip:unchanged"
STATUS_ERROR = "error"


def _log(payload, status, detail=""):
    """Record how this run ended, when hook debug is on. Never raises.

    This hook is otherwise invisible. It is registered with ``2>/dev/null``, it
    writes its memo into ``TMPDIR``, and a silent exit is its normal outcome for
    four different reasons — so "the agent got no block" and "the hook never ran"
    look identical from outside. :func:`osprey_hook_log.log_hook` is the channel
    that separates them: off by default, and when on it records one hook
    ``debug`` entry per run on the agent record, which is also what the web
    terminal's hook-activity feed reads.

    Wrapped the way every other hook wraps it (``osprey_writes_check``'s deny
    path is the precedent): a diagnostic may never cost the block it describes.
    """
    # logging must never cost the emit
    with contextlib.suppress(Exception):
        log_hook("control_context", payload, status=status, detail=detail)


def main():
    payload = {}
    try:
        payload = _read_payload()
        event = payload.get("hook_event_name")
        if event not in (_SESSION_START, _USER_PROMPT):
            _log(payload, STATUS_NOT_OUR_EVENT)
            return
        state = current_state(payload)
        if state is None:
            _log(payload, STATUS_NO_RECORD)
            return
        session_id = _session_id(payload)
        if event == _SESSION_START:
            _emit(event, render_block(state))
            _remember(session_id, state)
            _log(payload, STATUS_EMIT)
            return
        if session_id is None:
            _log(payload, STATUS_NO_SESSION)
            return
        previous = _remembered(session_id)
        if previous is None:
            _emit(event, render_block(state))
        elif previous != state:
            _emit(event, render_block(state, previous))
        else:
            _log(payload, STATUS_UNCHANGED)
            return
        _remember(session_id, state)
        _log(payload, STATUS_EMIT)
    except Exception as exc:
        _log(payload, STATUS_ERROR, detail=f"exception={type(exc).__name__}")
        return


if __name__ == "__main__":
    main()
    sys.exit(0)
