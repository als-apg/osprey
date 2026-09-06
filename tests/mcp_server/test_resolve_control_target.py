"""Tests for ``target_banner.resolve_control_target``, the record reader.

The banner used to answer "which target is this session on" by matching a
per-server state file against the caller's own parent PID. There is one control
context per deployment now, so the question has no session in it and the answer
is a field of one file: :func:`osprey_connectors.control_context.read_record`.

What this file pins is the whole of that contract — the reader's one success
case and its two ways of declining to answer. Every decline returns the
*baseline the caller passed in*, never a guess and never an exception: a
holder that cannot read the record must produce no refusal and no label rather
than a wrong one, and the callers (the Phoebus guard, the Bluesky lanes, the
health row, ``mcp_server.http.resolve_activity_target``) all rest on that.

``resolve_activity_target`` is the reason the baseline is a parameter rather
than something resolved here: it passes a sentinel no target can equal, which
is how it tells "nothing recorded" apart from "recorded as live".

Every case takes ``control_context_root``, which stamps
``OSPREY_AGENT_DATA_ROOT`` and drops the reader's cache on the way in and out.
Without it the reader falls back to ``resolve_shared_data_root()`` and would
answer from whatever record the developer's own deployment left in the checkout.
"""

from __future__ import annotations

from osprey.mcp_server.control_system import target_banner
from osprey_connectors import control_context

#: A baseline no record can name, so a test cannot pass by accident when the
#: reader answers with a real target that happens to match.
SENTINEL = "no-answer"


def test_no_record_answers_the_baseline(control_context_root):
    """An empty root is the ordinary state of a deployment nobody has switched."""
    assert target_banner.resolve_control_target(SENTINEL) == SENTINEL


def test_a_record_answers_its_target(control_context_root, write_control_context):
    """The one success case: the reader hands back the record's own target."""
    write_control_context(control_context_root, target="va", generation=3)

    assert target_banner.resolve_control_target("live") == "va"


def test_a_corrupt_record_answers_the_baseline(control_context_root):
    """Half-written bytes are "no answer", not an exception and not a guess.

    The record is replaced atomically, so this is not a state a reader should
    meet — but a truncated or hand-edited file must degrade the same way an
    absent one does, because a holder that raised here would fail a read tool
    for a reason that has nothing to do with the read.
    """
    path = control_context.record_path_under(control_context_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema": 1, "target": "va"', encoding="utf-8")
    control_context.invalidate_cache()

    assert target_banner.resolve_control_target(SENTINEL) == SENTINEL
