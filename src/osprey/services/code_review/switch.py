"""Whether an admin switched code review off in the web terminal this session runs under.

Code review is a runtime safety switch of the web terminal
(:mod:`osprey.interfaces.web_terminal.safety_switches`), but every review runs in
the agent's processes: the approval hook, the notebook hook and the executor.
Each asks :func:`switched_off` right before it would review, so a move counts from
the next review. The terminal answers on a UNIX socket in a directory only its
account can enter, and names it in :data:`SWITCH_SOCKET_ENV` for the sessions it
spawns (:mod:`osprey.interfaces.web_terminal.code_review_switch`).

The read fails closed. Only an answer that says ``"on": false`` in so many words,
served by this process's own ancestor (the terminal that spawned the session),
skips a review; no socket named, no answer, a timeout, an answer from any other
process or anything malformed is answered with ``None``, and the review runs.
The kernel names the serving process (``SO_PEERCRED`` on Linux, ``LOCAL_PEERPID``
on macOS; elsewhere nothing is believed), and it must run as this account and be
an ancestor. The agent runs as the same account and can bind a socket where the
terminal's was, but nothing it starts is an ancestor of the hook or executor
asking, so it cannot answer in the terminal's place, and the socket moves
nothing. Every skip is told to the agent, the operator's activity feed and the
audit trail, so a skip nobody meant shows up.
"""

from __future__ import annotations

from osprey.services.code_review.contract import ReviewSwitchedOff

#: The terminal's switch socket, published by the terminal for the sessions it spawns.
SWITCH_SOCKET_ENV = "OSPREY_TERMINAL_CODE_REVIEW_SOCKET"

#: The most of a switch answer read; the real one is a few dozen bytes.
SWITCH_ANSWER_LIMIT = 4096


def switched_off() -> ReviewSwitchedOff | None:
    """Who switched code review off, when the terminal says it is off; otherwise ``None``.

    Blocking for at most the read's timeout; async callers run it in a thread. Never raises.
    """
    try:
        from osprey.mcp_server.http import fetch_code_review_switch

        state = fetch_code_review_switch()
    except Exception:
        return None
    if not isinstance(state, dict) or state.get("on") is not False:
        return None
    by = state.get("off_by")
    at = state.get("off_at")
    return ReviewSwitchedOff(
        off_by=by if isinstance(by, str) and by.strip() else None,
        off_at=float(at) if isinstance(at, int | float) and not isinstance(at, bool) else None,
    )
