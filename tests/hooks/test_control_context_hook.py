"""``osprey_control_context``: the lock-file block, when it appears and what it says.

Run as Claude Code runs it — a subprocess fed the hook payload on stdin — with
the deployment's record written where the hook looks for it and the config
the hook reads the write posture from. What is pinned is the contract in the
hook's own docstring: a session start always describes; a prompt describes
only a change, marks it, and carries the previous value; nothing is said when
nothing moved, and nothing at all without a record.
"""

from __future__ import annotations

import json

import pytest

from tests._control_context_fixtures import write_control_context

pytestmark = pytest.mark.unit

SESSION_ID = "4f1c2a7e-0000-4000-8000-0000000000aa"

#: A switch-capable deployment armed for both of its targets.
ARMED_BOTH = {
    "control_system": {
        "type": "epics",
        "writes_enabled": True,
        "connector": {
            "epics": {"prefix": "RING:"},
            "virtual_accelerator": {"prefix": "VA:"},
        },
    }
}

#: The same deployment with its ring disarmed in config.
DISARMED_LIVE = {
    "control_system": {
        "type": "epics",
        "writes_enabled": True,
        "connector": {
            "epics": {"prefix": "RING:", "writes_enabled": False},
            "virtual_accelerator": {"prefix": "VA:"},
        },
    }
}


def _record(repo_root, target, generation, posture=None):
    write_control_context(
        repo_root / "var" / "agent_data", target=target, generation=generation, posture=posture
    )


def _context(result):
    return None if result is None else result["hookSpecificOutput"]["additionalContext"]


@pytest.fixture
def envelope(tmp_path, hook_runner, make_config, monkeypatch):
    """Run the hook for one event and return the WHOLE stdout envelope."""
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()

    def _run(event, config=ARMED_BOTH, session_id=SESSION_ID, **extra):
        payload = {"hook_event_name": event, **extra}
        if session_id is not None:
            payload["session_id"] = session_id
        return hook_runner(
            "osprey_control_context.py",
            "",
            {},
            config_path=make_config(config),
            cwd=tmp_path,
            hook_input_extra=payload,
        )

    return _run


@pytest.fixture
def run(envelope):
    """The block the hook emitted for one event; the memo is private to the test."""

    def _run(event, **kwargs):
        return _context(envelope(event, **kwargs))

    return _run


def test_a_session_start_describes_the_context(tmp_path, run):
    _record(tmp_path, "va", 2)

    assert run("SessionStart", source="startup") == (
        "--- osprey control-context ---\ntarget: va  gen: 2\nwrites: live=armed va=armed"
    )


def test_a_compaction_describes_it_again(tmp_path, run):
    """A rebuilt context may have lost the block; a start of any source restates it."""
    _record(tmp_path, "va", 2)
    run("SessionStart", source="startup")

    assert run("SessionStart", source="compact") is not None


def test_a_narrowing_shows_as_sandbox(tmp_path, run):
    _record(tmp_path, "va", 2, posture={"live": "sandbox"})

    assert run("SessionStart").endswith("writes: live=sandbox va=armed")


def test_a_target_the_deployment_does_not_arm_shows_as_off(tmp_path, run):
    _record(tmp_path, "va", 2)

    assert run("SessionStart", config=DISARMED_LIVE).endswith("writes: live=off va=armed")


def test_an_unchanged_prompt_says_nothing(tmp_path, run):
    _record(tmp_path, "va", 2)
    run("SessionStart")

    assert run("UserPromptSubmit") is None
    assert run("UserPromptSubmit") is None


def test_a_switch_under_the_session_is_marked_with_where_it_was(tmp_path, run):
    _record(tmp_path, "va", 2)
    run("SessionStart")
    _record(tmp_path, "live", 3)

    assert run("UserPromptSubmit") == (
        "--- osprey control-context (changed) ---\n"
        "target: live  gen: 3  (was va)\n"
        "writes: live=armed va=armed"
    )
    assert run("UserPromptSubmit") is None, "reported once, not on every prompt after it"


def test_a_narrowing_under_the_session_is_marked_on_its_target(tmp_path, run):
    _record(tmp_path, "live", 3)
    run("SessionStart")
    _record(tmp_path, "live", 3, posture={"live": "sandbox"})

    assert run("UserPromptSubmit") == (
        "--- osprey control-context (changed) ---\n"
        "target: live  gen: 3\n"
        "writes: live=sandbox (was armed) va=armed"
    )


def test_a_prompt_with_nothing_remembered_describes_in_full(tmp_path, run):
    """The memo is gone (or the hook is new to this session): the plain block, unmarked."""
    _record(tmp_path, "va", 2)

    assert run("UserPromptSubmit").startswith("--- osprey control-context ---\n")


def test_without_a_record_the_hook_is_silent(run):
    assert run("SessionStart") is None
    assert run("UserPromptSubmit") is None


def test_without_a_session_id_a_start_describes_and_a_prompt_is_silent(tmp_path, run):
    _record(tmp_path, "va", 2)

    assert run("SessionStart", session_id=None) is not None
    assert run("UserPromptSubmit", session_id=None) is None


def test_an_unrelated_event_is_ignored(tmp_path, run):
    _record(tmp_path, "va", 2)

    assert run("PreToolUse") is None


def test_the_block_is_plain_text_with_no_instruction(tmp_path, run):
    """Harness state, not a message: fixed keys, and not a sentence in it."""
    _record(tmp_path, "va", 2)
    block = run("SessionStart")

    assert block.count("\n") == 2
    assert not any(word in block.lower() for word in ("you", "please", "must", "should"))
    json.dumps(block)  # what the hook emitted round-trips as the JSON it came in


def test_the_envelope_names_the_event_it_was_emitted_for(tmp_path, envelope):
    """Without ``hookEventName`` the CLI drops the block and the agent sees nothing.

    Claude Code validates ``hookSpecificOutput`` against the schema for the
    event named INSIDE it. An envelope that omits the name fails that check:
    the additional context is discarded, and the failure is reported as a
    non-blocking hook error in the transcript rather than as a hook that
    crashed — so a hook missing this field goes on exiting 0 and emitting
    well-formed JSON that never reaches a turn.
    """
    _record(tmp_path, "va", 2)

    start = envelope("SessionStart", source="startup")
    assert start["hookSpecificOutput"]["hookEventName"] == "SessionStart"

    _record(tmp_path, "live", 3)
    prompt = envelope("UserPromptSubmit")
    assert prompt["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
