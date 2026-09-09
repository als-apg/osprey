#!/usr/bin/env python3
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
writes: live=sandbox va=armed
```

At a session start (``startup``, ``resume``, ``clear`` and ``compact`` alike:
a rebuilt context is a context that may have lost it) the block describes the
current state. On a prompt it is emitted only if the target or a write state
moved since the last emission for this session, marked ``(changed)`` and
carrying the previous value beside each key that moved:

```
--- osprey control-context (changed) ---
target: live  gen: 3  (was va)
writes: live=armed (was sandbox) va=armed
```

An unchanged prompt injects nothing. Nothing in the block asks the agent to do
anything: the point is that a switch made from the terminal's chip, or a
narrowing made in the header, reaches the agent as a fact before its next turn,
so what it holds about the machine is never stale — and so it has no reason to
re-derive from the record what the harness already told it.

## The three write states

``writes`` names every target a session here can reach
(``osprey_target_state.session_types``), each with one of:

* ``armed`` — a write proceeds: the deployment arms the target, the process is
  not a read-only run, and the operator has not narrowed it
  (``effective_writes_for``).
* ``sandbox`` — the operator narrowed this target on the chip, or the process
  is a read-only run. The deployment would arm it; the operator said not now.
* ``off`` — the deployment's own posture does not arm it (``writes_enabled``
  false or unstated for its connector type). No operator action lifts it.

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

Every failure — an unreadable record, an unwritable temp directory, a payload
that is not JSON — exits 0 with no output. This hook only describes; a turn
must never be held up by its absence.

## Saying which of those happened

Silence is this hook's normal outcome and it has five different meanings, so
every run ends with one ``log_hook`` record naming which: ``emit`` when a block
went out, ``skip:not-our-event``, ``skip:no-record``, ``skip:no-session`` and
``skip:unchanged`` for the four quiet ends, and ``error`` for a failure the
fail-open rule above swallowed. Debug-gated like every ``log_hook`` call, so it
costs a run nothing until someone turns it on.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import load_osprey_config, log_hook

try:
    import osprey_target_state as _target_state
except Exception:  # pragma: no cover - older render without the reader
    _target_state = None

#: The event that always emits: a context that has just been built or rebuilt.
_SESSION_START = "SessionStart"

#: The event that emits on change only.
_USER_PROMPT = "UserPromptSubmit"

#: What ``writes`` says about one target.
WRITES_ARMED = "armed"
WRITES_SANDBOX = "sandbox"
WRITES_OFF = "off"

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


def _writes_state(hook_input, section, target):
    """One of the three ``writes`` words for *target*."""
    if _target_state.effective_writes_for(hook_input, section, target):
        return WRITES_ARMED
    if _target_state.writes_posture(section, target) is True:
        return WRITES_SANDBOX
    return WRITES_OFF


def current_state(hook_input):
    """``{target, generation, writes: {target: word}}``, or ``None`` with no record."""
    if _target_state is None:
        return None
    record = _target_state.read_record(hook_input)
    if record is None:
        return None
    config = load_osprey_config(hook_input)
    section = config.get("control_system") if isinstance(config, dict) else None
    writes = {}
    for target in sorted(_target_state.session_types(section)):
        writes[target] = _writes_state(hook_input, section, target)
    return {
        "target": record.get("target"),
        "generation": record.get("generation"),
        "writes": writes,
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
    if session_id is None:
        return
    try:
        path = memo_path(session_id)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp, path)
    except Exception:
        pass


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
    that separates them: off by default, and when on it appends one record per
    run to ``<project>/.claude/hooks/hook_debug.jsonl``, which is also what the
    web terminal's hook-activity feed reads.

    Wrapped the way every other hook wraps it (``osprey_writes_check``'s deny
    path is the precedent): a diagnostic may never cost the block it describes.
    """
    try:
        log_hook("control_context", payload, status=status, detail=detail)
    except Exception:
        pass  # logging must never cost the emit


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
