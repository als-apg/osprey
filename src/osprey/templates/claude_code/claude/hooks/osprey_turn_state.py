#!/usr/bin/env python3
"""
---
name: Turn State
description: Reports the terminal agent's turn edges — busy on a submitted prompt, idle when the turn ends — to the web terminal
summary: The web terminal learns when its own agent is between turns from the process that runs it, instead of guessing from terminal output
event: UserPromptSubmit, Stop, StopFailure, SessionStart
---

## Flow

```
hook fires ──► OSPREY_SESSION_ID unset, host-prefixed OSPREY_TERMINAL_SESSION
                    │    and OSPREY_CLI_LAUNCH set? ──YES──► CLI pointer (below) ──► exit 0
                   NO
                    │
                    ▼
               OSPREY_WEB_PORT and OSPREY_SESSION_ID both set? ──NO──► exit 0, silent
                    │
                   YES
                    │
                    ▼
               read stdin JSON
                    │
                    ▼
               hook_event_name ──► busy | idle ──► unrecognised ──► exit 0, silent
                    │
                    ▼
               SessionStart after a compaction? ──YES──► exit 0, silent
                    │
                    ▼
               POST /api/agent-turn {session_id, pool_key, state, surface, ts, source}
               (2 s, fail-open)
                    │
                    ▼
               exit 0
```

## Details

The web terminal can show the same conversation in either of two surfaces, and
only one process may hold it at a time. Moving it means tearing the outgoing
process down, which is safe *between* turns and destructive in the middle of
one. This hook is how the server knows which it is: the process reports its own
turn edges rather than having the server infer them from terminal output.

Four events carry an edge:

* ``UserPromptSubmit`` — the operator submitted a prompt, so a turn starts:
  **busy**.
* ``Stop`` — the turn finished: **idle**.
* ``StopFailure`` — the turn ended in a failure rather than a completion, which
  is still the end of it: **idle**. Builds that do not emit this event simply
  never send the report, and the next edge is whatever arrives after it.
* ``SessionStart`` — a session that has just opened is between turns: **idle**.
  It also names the transcript the session opened with, which is the other half
  of what this endpoint records.

## Compaction is not a turn edge

``SessionStart`` fires for four sources — ``startup``, ``resume``, ``clear`` and
``compact`` — and the last one is not an edge at all. Auto-compaction happens
*inside* a running turn, so an idle report for it would tell the server the
process is free at the moment it is busiest, and a view flip would tear down a
turn in flight. The registered ``matcher`` in ``settings.json.j2`` therefore
lists only the first three, and this file refuses a ``compact`` payload as well:
the matcher is the contract, and the check here is what still holds if a
deployment's ``settings.json`` is edited by hand.

## Two identities

The body carries both, and they are not the same thing:

* ``pool_key`` is ``OSPREY_SESSION_ID`` — the session key the browser, the
  terminal pool and the audit ledger all share. It never changes.
* ``session_id`` is Claude Code's own session id, read from the hook payload. It
  names the transcript file, and a ``/clear`` moves it while the key stays put.

A payload with no session id is dropped rather than reported with a blank one:
the server writes the id it receives as the key's current transcript, so an
empty value would erase a mapping instead of updating it.

``surface`` is ``OSPREY_WEB_UX`` (``terminal`` | ``chat``). Only the terminal
surface is recorded — the chat knows its own turn boundaries — but the
report is sent either way and the server drops what it does not want. An
unlaunched-by-the-web-terminal session has neither variable and never sends
anything.

## Terminal-API authorization

The POST carries ``Authorization: Bearer <OSPREY_PANEL_TOKEN>`` whenever that
variable holds a non-blank value in the environment the hook inherits — the web
terminal exports it into the agent it launches. When it is unset, empty or
whitespace-only the header is omitted entirely rather than sent blank, matching
how the server-side ``mcp_server.http._panel_auth_headers`` reads the same
carrier.

Unlike the hooks that fall back to a default port, this one has no fallback:
without ``OSPREY_WEB_PORT`` there is no web terminal to report to, and a report
sent to a guessed port would be a report about a session that server has never
heard of.

## The CLI transcript pointer

An ``osprey chat`` or ``osprey query`` launch has no web terminal and no hub
ledger, yet "this session" must still follow a ``/clear`` there: the MCP
servers it started keep the pre-clear ``CLAUDE_CODE_SESSION_ID`` for their
whole life. So the hook keeps a pointer for it instead, the CLI mirror of the
web ledger moving before any tool runs. A process with no
``OSPREY_SESSION_ID``, an ``OSPREY_TERMINAL_SESSION`` carrying the host prefix
(``chat-``) and an ``OSPREY_CLI_LAUNCH`` writes the payload's ``session_id`` to
``<agent-data root>/cli-transcripts/<OSPREY_CLI_LAUNCH>`` on every
``SessionStart`` (startup, resume, clear — never ``compact``) and every
``UserPromptSubmit``. The root is ``osprey_hook_log.repo_agent_data_root()``,
the one ``osprey.stores.artifact_scope.current_scope`` reads from.

The file holds the transcript id on one line and is replaced atomically
(temporary file plus rename), so a reader never sees half of one. A launch id
that is not a safe single path component writes nothing. Each write also
prunes *other* launches' pointer files older than seven days — never the
writer's own. Each launcher exports a fresh launch id, never an inherited one,
so a launch nested inside another one's Bash keeps a pointer of its own.

## Import contract

The web branch is stdlib-only and must NOT import osprey, yaml, requests, or
any third-party lib: ``json``, ``os``, ``sys``, ``time``, ``urllib.request``.
The CLI branch additionally imports the siblings ``osprey_hook_log`` (for
``repo_agent_data_root``) and ``osprey_safety_file`` (the pointer is replaced
whole through ``atomic_replace``), through the same ``sys.path`` shim as the
other hooks, and only when that branch runs — the build catalog declares the
dependencies (``requires=("hooks/hook-log", "hooks/safety-file")``). This file itself stays
3.9-safe, since the hook runs under ``python3`` (possibly a system Python, not
the venv interpreter); an interpreter the sibling cannot load costs the
pointer, never the web report or the turn.

Fails open on any error — stays silent, writes nothing to stdout, and exits 0.
Every one of these events would otherwise block the operator's prompt or the end
of their turn, and a missed report costs a flip its fast path, never the turn.
"""

import contextlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

#: What each hook event says about the process. An event outside this table is
#: not a turn edge and is not reported.
_STATE_BY_EVENT = {
    "UserPromptSubmit": "busy",
    "Stop": "idle",
    "StopFailure": "idle",
    "SessionStart": "idle",
}

#: The one event whose payload names a source, and so the only one the
#: compaction backstop below can ask about.
_SESSION_START = "SessionStart"

#: The ``SessionStart`` source that must never be reported — see the module
#: docstring. The registered matcher excludes it; this is the backstop.
_COMPACT_SOURCE = "compact"

#: The events that move the CLI transcript pointer: a session opening (the
#: compaction backstop still applies) and a prompt being submitted.
_CLI_POINTER_EVENTS = frozenset({"SessionStart", "UserPromptSubmit"})

#: Directory under the agent-data root holding one pointer file per CLI launch.
#: Must equal ``osprey.stores.artifact_scope.CLI_POINTER_DIR``.
CLI_POINTER_DIR = "cli-transcripts"

#: The terminal-session prefix a host launch carries. Must equal
#: ``osprey.build.claude_code_telemetry.HOST_SESSION_PREFIX``.
HOST_SESSION_PREFIX = "chat-"

#: Age past which another launch's pointer file is pruned, in seconds (C10).
POINTER_MAX_AGE_S = 7 * 24 * 60 * 60

#: Seconds the POST may take. Long enough for a loopback call to a busy server,
#: short enough that a dead one never holds up a prompt or a turn's end.
_POST_TIMEOUT = 2


def _read_payload(stream=None):
    """Return the hook's stdin payload as a dict, or ``{}``.

    Args:
        stream: Text stream to read the payload from. Defaults to ``sys.stdin``.

    Returns:
        The decoded payload when it is a JSON object, and an empty dict for
        anything else — unreadable stdin, empty input, or a document that is
        not an object. The caller then finds no event name and reports nothing.
    """
    try:
        raw = (stream if stream is not None else sys.stdin).read()
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _turn_request(body, port):
    """Build the ``POST /api/agent-turn`` request carrying *body*.

    Args:
        body: The report, already reduced to the endpoint's fixed contract.
        port: The web terminal's port, from ``OSPREY_WEB_PORT``.

    Returns:
        A ``urllib.request.Request`` with the JSON body and, when
        ``OSPREY_PANEL_TOKEN`` holds a non-blank value, a bearer header. A blank
        one is omitted rather than sent: it would be a credential claim this
        hook cannot back.
    """
    headers = {"Content-Type": "application/json"}
    token = (os.environ.get("OSPREY_PANEL_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(
        f"http://127.0.0.1:{port}/api/agent-turn",
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )


def report_turn_state(body, port):
    """Send one turn report, swallowing every failure.

    A web terminal that is down, refusing the call or slow to answer costs the
    server one edge, which it recovers from its own evidence. It must never
    cost the operator the prompt or the turn that triggered the report.
    """
    with contextlib.suppress(Exception):
        urllib.request.urlopen(_turn_request(body, port), timeout=_POST_TIMEOUT).close()


def _safe_launch(launch):
    """Whether *launch* is a single, safe path component for a pointer file."""
    return (
        isinstance(launch, str)
        and bool(launch)
        and launch not in {".", ".."}
        and not any(char in launch for char in ("/", "\\", "\0"))
    )


def _siblings_on_path():
    """Put this hook's directory on ``sys.path`` for the sibling imports.

    Siblings (``osprey_hook_log``, ``osprey_safety_file``) are imported on the
    CLI branch only, never at module scope, so the web branch stays stdlib-only
    and an in-process import of this module stays inert.
    """
    hooks_dir = os.path.dirname(os.path.abspath(__file__))
    if hooks_dir not in sys.path:
        sys.path.insert(0, hooks_dir)


def _pointer_root(payload):
    """The agent-data root the pointer lives under (``repo_agent_data_root``)."""
    _siblings_on_path()
    from osprey_hook_log import repo_agent_data_root

    return repo_agent_data_root(payload)


def _prune_pointers(directory, own, now):
    """Remove other launches' pointer files older than :data:`POINTER_MAX_AGE_S`."""
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if name == own:
            continue
        path = os.path.join(directory, name)
        with contextlib.suppress(OSError):
            if os.path.isfile(path) and now - os.path.getmtime(path) > POINTER_MAX_AGE_S:
                os.unlink(path)


#: Read by the host launch's own MCP processes (``artifact_scope``), under the same account.
_POINTER_FILE_MODE = 0o600


def write_cli_pointer(root, launch, transcript):
    """Atomically write *transcript* as *launch*'s pointer under *root*.

    Returns the pointer path, or ``None`` when *launch* is not a safe name or
    *transcript* is not a one-line id. Raises on a filesystem failure; the
    caller fails open.
    """
    if not _safe_launch(launch):
        return None
    if not isinstance(transcript, str) or not transcript or "\n" in transcript:
        return None
    directory = os.path.join(str(root), CLI_POINTER_DIR)
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, launch)
    _siblings_on_path()
    from osprey_safety_file import atomic_replace

    atomic_replace(
        Path(target),
        (transcript + "\n").encode("utf-8"),
        mode=_POINTER_FILE_MODE,
        sync="best_effort",
    )
    _prune_pointers(directory, launch, time.time())
    return target


def _cli_launch():
    """The launch id when this process is a CLI launch's agent, else ``None``."""
    if (os.environ.get("OSPREY_SESSION_ID") or "").strip():
        return None
    if not (os.environ.get("OSPREY_TERMINAL_SESSION") or "").startswith(HOST_SESSION_PREFIX):
        return None
    return os.environ.get("OSPREY_CLI_LAUNCH") or None


def record_cli_pointer(launch):
    """The CLI branch: move *launch*'s pointer to the payload's transcript."""
    payload = _read_payload()
    event = payload.get("hook_event_name")
    if event not in _CLI_POINTER_EVENTS:
        return
    if event == _SESSION_START and payload.get("source") == _COMPACT_SOURCE:
        # Compaction keeps the transcript; it is not a move.
        return
    if not _safe_launch(launch):
        return
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return
    write_cli_pointer(_pointer_root(payload), launch, session_id)


def main():
    try:
        launch = _cli_launch()
        if launch is not None:
            # A host launch: no web terminal to report to, only the pointer.
            with contextlib.suppress(Exception):
                record_cli_pointer(launch)
            return 0

        port = (os.environ.get("OSPREY_WEB_PORT") or "").strip()
        pool_key = (os.environ.get("OSPREY_SESSION_ID") or "").strip()
        if not port or not pool_key:
            # Not a web-terminal session: nothing to report to.
            return 0

        payload = _read_payload()
        event = payload.get("hook_event_name")
        state = _STATE_BY_EVENT.get(event)
        if state is None:
            return 0
        if event == _SESSION_START and payload.get("source") == _COMPACT_SOURCE:
            # A compaction is not a turn edge, whatever the matcher allowed.
            # Only SessionStart carries a source, so the check asks about the
            # one event where the word means what this backstop reads it as.
            return 0

        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            # Without a transcript id the report would blank the one on record.
            return 0

        report_turn_state(
            {
                "session_id": session_id,
                "pool_key": pool_key,
                "state": state,
                "surface": (os.environ.get("OSPREY_WEB_UX") or "").strip(),
                "ts": time.time(),
                "source": payload.get("hook_event_name"),
            },
            port,
        )
        return 0
    except Exception:
        # Last-resort fail-open: never block a prompt, a turn or a session start.
        return 0


if __name__ == "__main__":
    sys.exit(main())
