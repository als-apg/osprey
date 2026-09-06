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
hook fires ──► OSPREY_WEB_PORT and OSPREY_SESSION_ID both set? ──NO──► exit 0, silent
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

``surface`` is ``OSPREY_WEB_UX`` (``expert`` | ``simple``). Only the expert
surface is recorded — the simple view knows its own turn boundaries — but the
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

stdlib-only — must NOT import osprey, yaml, requests, or any third-party lib.
Uses only ``json``, ``os``, ``sys``, ``time``, ``urllib.request``. The hook runs
under ``python3`` (possibly a system Python 3.9, not the venv interpreter), so
this file must be 3.9-safe.

Fails open on any error — stays silent, writes nothing to stdout, and exits 0.
Every one of these events would otherwise block the operator's prompt or the end
of their turn, and a missed report costs a flip its fast path, never the turn.
"""

import json
import os
import sys
import time
import urllib.request

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
    try:
        urllib.request.urlopen(_turn_request(body, port), timeout=_POST_TIMEOUT).close()
    except Exception:
        pass


def main():
    try:
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
