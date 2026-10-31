"""Hooks that fail open on stdin they cannot use.

A closed pipe, a truncated write and a non-object payload — falsy (``[]``) or
truthy (``[1,2,3]``) — give a hook nothing to judge. Each hook below must exit
0 with no decision on stdout and no traceback: anything else turns a bad
payload into a blocked, denied or stalled tool call. The truthy payload is the
one an emptiness check lets through, so it has to be rejected on shape.

Hooks whose fail-open contract carries an extra check (no pending review, an
untouched tree, a dead web port) keep their own copy beside their other tests.
"""

import pytest


@pytest.mark.parametrize(
    "stdin",
    ["", "{nope", "[]", "[1,2,3]"],
    ids=["empty", "invalid-json", "wrong-shape", "wrong-shape-truthy"],
)
@pytest.mark.parametrize(
    "hook_name",
    [
        # An "ask" here would stall every tool call behind a prompt nobody can answer.
        "osprey_approval.py",
        # A PostToolUse failure would surface on a tool call that already succeeded.
        "osprey_error_guidance.py",
        # No tool_name to match is no opinion, not a deny: Write is left to the
        # rest of the permission chain.
        "osprey_memory_guard.py",
        # Fail-closed once it has a channel and a value, but it never gets that far.
        "osprey_limits.py",
        # A PreToolUse hook fails open by exiting 0 and printing no decision.
        "osprey_writes_check.py",
    ],
)
def test_malformed_stdin_fails_open(tmp_path, hook_runner_raw, hook_name, stdin):
    """Unusable stdin passes the tool call through untouched."""
    returncode, stdout, stderr = hook_runner_raw(
        hook_name,
        tool_name=None,
        tool_input=None,
        cwd=tmp_path,
        stdin_override=stdin,
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "Traceback" not in stderr
