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
import os

import pytest

from osprey_connectors.write_target import KIND_BANNERS
from osprey_connectors_tests._harness import (
    state_dir_under,
    write_control_context,
    write_server_report,
)

SESSION_ID = "4f1c2a7e-0000-4000-8000-0000000000aa"

#: A switch-capable deployment armed for both of its targets.
ARMED_BOTH = {
    "control_system": {
        "type": "epics",
        "writes": "allowed",
        "connector": {
            "epics": {"prefix": "RING:"},
            "virtual_accelerator": {"prefix": "VA:"},
        },
    }
}

#: The same deployment whose login tier never writes.
TIER_NEVER = {
    "control_system": {
        "type": "epics",
        "writes": "never",
        "connector": {
            "epics": {"prefix": "RING:"},
            "virtual_accelerator": {"prefix": "VA:"},
        },
    }
}

#: A deployment that runs the live stand-in beside its ring and its simulator.
WITH_STANDIN = {
    "control_system": {
        "type": "epics",
        "writes": "allowed",
        "connector": {
            "epics": {"prefix": "RING:"},
            "virtual_accelerator": {"prefix": "VA:"},
            "live_standin": {"prefix": "SI:"},
        },
    }
}

REAL = f'"{KIND_BANNERS["real"]}"'
REHEARSAL = f'"{KIND_BANNERS["rehearsal"]}"'
SIMULATOR = f'"{KIND_BANNERS["simulator"]}"'

#: The same two-machine deployment whose login tier caps writes at asked.
TIER_ASKED = {
    "control_system": {
        "type": "epics",
        "writes": "asked",
        "connector": {
            "epics": {"prefix": "RING:"},
            "virtual_accelerator": {"prefix": "VA:"},
        },
    }
}

#: The line that says what each position means for the next write.
POSITIONS = (
    'positions: off="the next write is refused until the chip moves the switch" '
    'asked="a person approves the next write" '
    'allowed="the login\'s switch position approves the next write; limits still apply"'
)

#: The machines line of the two-machine deployment.
MACHINES = f"machines: live={REAL} va={SIMULATOR}"


def _agent_data(repo_root):
    return repo_root / "var" / "agent_data"


def _record(repo_root, target, generation, posture=None, switch=None):
    write_control_context(
        _agent_data(repo_root),
        target=target,
        generation=generation,
        posture=posture,
        switch=switch,
    )


def _publish_standin(repo_root, machine_kind):
    """A live controls server (this process) publishing the stand-in's kind."""
    write_server_report(
        _agent_data(repo_root),
        os.getpid(),
        targets={"standin": {"machine_kind": machine_kind}},
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
        "--- osprey control-context ---\n"
        "target: va  gen: 2\n"
        "writes: live=off va=off\n"
        f"{POSITIONS}\n"
        f"{MACHINES}"
    )


def test_a_machine_at_a_position_reads_it_and_the_others_off(tmp_path, run):
    _record(tmp_path, "va", 2, switch={"va": "asked"})

    assert run("SessionStart") == (
        "--- osprey control-context ---\n"
        "target: va  gen: 2\n"
        "writes: live=off va=asked\n"
        f"{POSITIONS}\n"
        f"{MACHINES}"
    )


@pytest.mark.parametrize("target,other", [("live", "va"), ("va", "live")])
@pytest.mark.parametrize(
    "position,config,word",
    [
        (None, ARMED_BOTH, "off"),
        ("asked", ARMED_BOTH, "asked"),
        ("allowed", ARMED_BOTH, "allowed"),
        ("allowed", TIER_NEVER, "refused:tier_never"),
    ],
)
def test_each_target_reads_each_of_the_four_words(
    tmp_path, run, target, other, position, config, word
):
    _record(tmp_path, "va", 2, switch={target: position} if position else {})

    block = run("SessionStart", config=config)
    other_word = "refused:tier_never" if config is TIER_NEVER else "off"
    words = {target: word, other: other_word}
    assert f"writes: live={words['live']} va={words['va']}\n" in block


def test_a_tier_capped_at_asked_reads_a_stored_allowed_as_asked(tmp_path, run):
    """The word is the verdict's position, never the stored one."""
    _record(tmp_path, "va", 2, switch={"live": "allowed", "va": "asked"})

    assert "writes: live=asked va=asked\n" in run("SessionStart", config=TIER_ASKED)


def test_a_legacy_stored_on_reads_as_asked(tmp_path, run):
    """The retired word is read as the narrowest arming position, and never echoed."""
    _record(tmp_path, "va", 2, switch={"va": "on"})

    block = run("SessionStart")
    assert "writes: live=off va=asked\n" in block
    assert "=on" not in block


def test_a_compaction_describes_it_again(tmp_path, run):
    """A rebuilt context may have lost the block; a start of any source restates it."""
    _record(tmp_path, "va", 2)
    run("SessionStart", source="startup")

    assert run("SessionStart", source="compact") is not None


def test_an_old_sandbox_narrowing_is_not_a_state_word(tmp_path, run):
    """The switch decides; the retired ``posture`` field is not read."""
    _record(tmp_path, "va", 2, posture={"live": "sandbox"}, switch={"live": "asked", "va": "asked"})

    block = run("SessionStart")
    assert "writes: live=asked va=asked\n" in block
    assert "sandbox" not in block
    assert "armed" not in block


def test_a_tier_that_never_writes_reads_as_refused_not_off(tmp_path, run):
    """The chip cannot lift a tier: the block names the reason, not a switch state."""
    _record(tmp_path, "va", 2, switch={"live": "asked", "va": "allowed"})

    block = run("SessionStart", config=TIER_NEVER)
    assert "writes: live=refused:tier_never va=refused:tier_never\n" in block


def test_an_unreadable_record_says_writes_are_refused_never_off(tmp_path, run):
    """An unreadable switch is not a switch someone turned off."""
    _record(tmp_path, "va", 2, switch={"va": "allowed"})
    record = state_dir_under(_agent_data(tmp_path)) / "control_context.json"
    record.write_text("{not json", encoding="utf-8")

    block = run("SessionStart")
    assert block == (
        "--- osprey control-context ---\n"
        "target: unknown  gen: unknown\n"
        "writes: live=refused:record_unavailable va=refused:record_unavailable\n"
        f"{POSITIONS}\n"
        f"{MACHINES}"
    )
    assert "=off" not in block
    assert "=asked" not in block
    assert "=allowed" not in block


def test_the_stand_in_reads_as_the_rehearsal_only_when_published_so(tmp_path, run):
    _record(tmp_path, "standin", 1)
    _publish_standin(tmp_path, "rehearsal")

    block = run("SessionStart", config=WITH_STANDIN)
    assert block.endswith(f"machines: live={REAL} standin={REHEARSAL} va={SIMULATOR}")


def test_a_stand_in_not_stood_up_reads_as_the_real_machine(tmp_path, run):
    """No live report, or a report that does not say rehearsal: never "nothing moves"."""
    _record(tmp_path, "standin", 1)

    assert run("SessionStart", config=WITH_STANDIN).endswith(
        f"machines: live={REAL} standin={REAL} va={SIMULATOR}"
    )

    _publish_standin(tmp_path, "real")
    block = run("SessionStart", config=WITH_STANDIN)
    assert block.endswith(f"machines: live={REAL} standin={REAL} va={SIMULATOR}")
    # The simulator's banner also says "nothing moves"; the stand-in must not.
    assert f"standin={REHEARSAL}" not in block
    assert block.count("nothing moves") == 1


def test_a_stand_in_coming_up_under_the_session_is_marked(tmp_path, run):
    _record(tmp_path, "standin", 1)
    run("SessionStart", config=WITH_STANDIN)
    _publish_standin(tmp_path, "rehearsal")

    block = run("UserPromptSubmit", config=WITH_STANDIN)
    assert block.startswith("--- osprey control-context (changed) ---\n")
    assert f"standin={REHEARSAL} (was {REAL})" in block


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
        "writes: live=off va=off\n"
        f"{POSITIONS}\n"
        f"{MACHINES}"
    )
    assert run("UserPromptSubmit") is None, "reported once, not on every prompt after it"


def test_a_position_moved_under_the_session_is_marked_on_its_target(tmp_path, run):
    _record(tmp_path, "live", 3)
    run("SessionStart")
    _record(tmp_path, "live", 3, switch={"live": "asked"})

    assert run("UserPromptSubmit") == (
        "--- osprey control-context (changed) ---\n"
        "target: live  gen: 3\n"
        "writes: live=asked (was off) va=off\n"
        f"{POSITIONS}\n"
        f"{MACHINES}"
    )


def test_a_widening_under_the_session_is_marked_with_the_position_it_left(tmp_path, run):
    _record(tmp_path, "live", 3, switch={"live": "asked"})
    run("SessionStart")
    _record(tmp_path, "live", 3, switch={"live": "allowed"})

    assert "writes: live=allowed (was asked) va=off\n" in run("UserPromptSubmit")


def test_a_turn_off_under_the_session_is_marked_on_its_target(tmp_path, run):
    _record(tmp_path, "live", 3, switch={"live": "allowed"})
    run("SessionStart")
    _record(tmp_path, "live", 3, switch={})

    assert run("UserPromptSubmit").endswith(
        f"writes: live=off (was allowed) va=off\n{POSITIONS}\n{MACHINES}"
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
    """Harness state, not a message: fixed keys, defining its words, asking nothing."""
    _record(tmp_path, "va", 2)
    block = run("SessionStart")

    assert block.count("\n") == 4
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
