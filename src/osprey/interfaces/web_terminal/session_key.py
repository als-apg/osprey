"""The closed grammar of a session key.

A session key is the one identity a browser tab keeps for its whole life: the
same string names the PTY pool entry, the chat pool entry, the transcript map
entry and the audit session. It is written to stores
on disk and later spliced into a ``claude --resume`` argv and a transcript
filename, so the grammar is closed rather than length-bounded: a canonical
lowercase UUID — eight-four-four-four-twelve hex, no prefix, no suffix — and
nothing else.

Every key the web terminal legitimately handles is minted that way — a Claude
session-file stem, or ``crypto.randomUUID()`` in the browser — so the closed
form costs no reach while keeping decorated keys (the ``/ws/operator`` pool's
``operator-<hex8>``) and near-miss strings out of every store that decides a
child process's execution mode.

The posture routes, the chat route, the hand-off route and the agent-turn
route all ask :func:`is_posture_key` rather than carrying a pattern of their
own, so the surfaces cannot drift on what a key is. The pattern itself stays
private so there is one place to change it.
"""

from __future__ import annotations

import re

_SESSION_KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def is_posture_key(session_id: str | None) -> bool:
    """Whether *session_id* is a canonical, bare session UUID.

    Named for the surface that first closed the grammar: a key this answers
    ``False`` for is a key no store on disk will ever answer for.

    **``None`` is not a key**, and answers ``False``. That is what the routes
    which REQUIRE one want — the chat, hand-off and agent-turn routes all read
    ``if not is_posture_key(...)`` and refuse — and it is deliberately not the
    whole story for the three control-gesture routes. There the id is
    ``str | None`` and **validated only when present**: the control context is
    the deployment's, not a session's, and the Lab page's bar posts the same
    bodies with no terminal session to name. Those routes decide the absent
    case before asking, so that "not sent" and "sent malformed" stay different
    answers — a 200 or 202, against a 400 ``invalid_session_id``.
    """
    return session_id is not None and bool(_SESSION_KEY_RE.match(session_id))
