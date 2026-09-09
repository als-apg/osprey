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


# ---------------------------------------------------------------------------
# The debug trace
# ---------------------------------------------------------------------------


def _debug_records(project_dir):
    """Every ``control_context`` record in the run's ``hook_debug.jsonl``."""
    log_path = project_dir / ".claude" / "hooks" / "hook_debug.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    return [record for record in records if record["hook"] == "control_context"]


def test_every_way_the_hook_ends_leaves_one_debug_record(
    tmp_path, hook_runner_raw, make_config, monkeypatch
):
    """Silence has five meanings; the JSONL channel is what tells them apart.

    The hook is registered with ``2>/dev/null`` and its memo lives in ``TMPDIR``,
    so a run that said nothing is indistinguishable from a run that never
    happened — which is exactly the state the envelope bug shipped in. Drive all
    seven exits and assert the file, not stderr: the file is what the web
    terminal's hook-activity feed reads and what survives the shell redirect.
    """
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    # log_hook appends; it does not create the directory it appends into.
    (tmp_path / ".claude" / "hooks").mkdir(parents=True)
    config_path = make_config(ARMED_BOTH)

    session = SESSION_ID
    other = SESSION_ID[:-2] + "bb"
    corrupt = SESSION_ID[:-2] + "cc"

    def _run(event, session_id=session, **extra):
        payload = {"hook_event_name": event, **extra}
        if session_id is not None:
            payload["session_id"] = session_id
        returncode, _stdout, _stderr = hook_runner_raw(
            "osprey_control_context.py",
            "",
            {},
            config_path=config_path,
            cwd=tmp_path,
            hook_input_extra=payload,
        )
        assert returncode == 0, "the hook describes; it never holds up a turn"

    _run("SessionStart")  # skip:no-record — nothing written yet
    _record(tmp_path, "va", 2)
    _run("PreToolUse")  # skip:not-our-event
    _run("SessionStart")  # emit, and the memo now holds va/2
    _run("UserPromptSubmit", session_id=None)  # skip:no-session
    _run("UserPromptSubmit")  # skip:unchanged
    _record(tmp_path, "live", 3)
    _run("UserPromptSubmit")  # emit — the switch, marked
    _run("UserPromptSubmit", session_id=other)  # emit — nothing remembered
    # A memo whose `writes` is not a mapping: render_block walks straight into it.
    (tmp_path / "tmp" / f"osprey-control-context-{corrupt}.json").write_text(
        json.dumps({"target": "va", "generation": 1, "writes": 0})
    )
    _run("UserPromptSubmit", session_id=corrupt)  # error

    statuses = [record["status"] for record in _debug_records(tmp_path)]

    assert statuses == [
        "skip:no-record",
        "skip:not-our-event",
        "emit",
        "skip:no-session",
        "skip:unchanged",
        "emit",
        "emit",
        "error",
    ]


def test_the_error_record_names_what_went_wrong(
    tmp_path, hook_runner_raw, make_config, monkeypatch
):
    """An ``error`` with no exception in it is a dead end for whoever reads it."""
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    (tmp_path / ".claude" / "hooks").mkdir(parents=True)
    _record(tmp_path, "va", 2)
    (tmp_path / "tmp" / f"osprey-control-context-{SESSION_ID}.json").write_text(
        json.dumps({"target": "live", "generation": 1, "writes": 0})
    )

    hook_runner_raw(
        "osprey_control_context.py",
        "",
        {},
        config_path=make_config(ARMED_BOTH),
        cwd=tmp_path,
        hook_input_extra={"hook_event_name": "UserPromptSubmit", "session_id": SESSION_ID},
    )

    (record,) = _debug_records(tmp_path)
    assert record["status"] == "error"
    assert record["detail"] == "exception=TypeError"


def test_the_trace_stays_off_until_it_is_asked_for(
    tmp_path, hook_runner_raw, make_config, monkeypatch
):
    """A debug facility that files on every deployment is not a debug facility."""
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    (tmp_path / ".claude" / "hooks").mkdir(parents=True)
    _record(tmp_path, "va", 2)

    hook_runner_raw(
        "osprey_control_context.py",
        "",
        {},
        config_path=make_config(ARMED_BOTH),
        cwd=tmp_path,
        hook_input_extra={"hook_event_name": "SessionStart", "session_id": SESSION_ID},
    )

    assert not (tmp_path / ".claude" / "hooks" / "hook_debug.jsonl").exists()
