"""The stamp a rendered `channel_write` approval leaves for the controls server.

Approval is enforced out here, in a PreToolUse hook, before the MCP server is
called at all. The server's earliest possible observation of the session's
target is therefore its own tool entry — which is *after* the human clicked, and
a target switch can land while they are still deciding. Nothing the server reads
on its own covers that window.

So the hook writes down the binding it rendered the prompt against, beside the
target state, keyed by the write payload — the only thing that provably crosses
the gap between a hook process and an MCP tool call. The server reads it back and
refuses a write whose approval was granted on a different session.

Two properties matter and are pinned here: the stamp carries the SAME record the
`Target:` line was rendered from (a stamp describing a different read would be a
second opinion about identity), and the key the hook files it under is the key
the server looks it up by (the derivation is stated twice — hooks run outside the
osprey venv — so the two spellings are compared directly).
"""

from __future__ import annotations

import json
import os
import time

import pytest

#: A PID on the synthesized ancestor chain that is not this process.
OWNER_PPID = 424242

LIVE_ENDPOINT = "pva://live-gw.example.org:5075"
VA_ENDPOINT = "pva://127.0.0.1:5074"

SERVER_PID = 5150

#: The tool name Claude Code passes for the controls server's write tool.
CHANNEL_WRITE = "mcp__controls__channel_write"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def approval(hook_module):
    """The `osprey_approval` module, imported through the test seam."""
    return hook_module("osprey_approval")


@pytest.fixture
def reader(approval):
    """The reader module the hook actually bound at import time."""
    module = approval._target_state
    assert module is not None, "the approval hook did not bind osprey_target_state"
    return module


@pytest.fixture
def state_dir(tmp_path, reader, monkeypatch):
    """Point the reader — and therefore the stamp writer — at a temp directory."""
    directory = tmp_path / "control_target"
    directory.mkdir()
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: str(directory))
    return directory


@pytest.fixture
def resolvable_session(reader, monkeypatch):
    """Make the synthesized owner chain resolve, with every PID alive."""
    monkeypatch.setattr(reader, "ancestor_pids", lambda *a, **k: [os.getpid(), OWNER_PPID, 300])
    monkeypatch.setattr(reader, "_is_process_alive", lambda pid: True)


def state_record(server_pid=SERVER_PID, owner_ppid=OWNER_PPID, target="va", generation=4):
    """A well-formed state record, with per-target display metadata."""
    return {
        "target": target,
        "generation": generation,
        "server_pid": server_pid,
        "owner_ppid": owner_ppid,
        "targets": {
            "live": {"label": "Storage ring", "endpoint": LIVE_ENDPOINT, "real_machine": True},
            "va": {"label": "Virtual accelerator", "endpoint": VA_ENDPOINT, "real_machine": False},
        },
        "children": [],
    }


def write_state(directory, **kwargs):
    """Write one state file into *directory* and hand back its path."""
    record = state_record(**kwargs)
    path = directory / f"target_state_{record['server_pid']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def write_payload(channel="TEST:PV", value=42.0, verification_level=None):
    """The tool input Claude Code hands the hook for a one-channel write."""
    payload = {"operations": [{"channel": channel, "value": value}]}
    if verification_level is not None:
        payload["verification_level"] = verification_level
    return payload


def hook_input_for(tool_input, tool_name=CHANNEL_WRITE):
    return {"tool_name": tool_name, "tool_input": tool_input, "cwd": os.getcwd()}


def stamps_in(directory):
    """Every stamp file in *directory*, parsed, newest name first."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("write_approval_*.json"))
    ]


# ---------------------------------------------------------------------------
# what the prompt renders is what the stamp records
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_channel_write_ask_stamps_the_binding_it_rendered(
    approval, state_dir, resolvable_session
):
    """The ask carries the target line; the stamp carries the same record."""
    write_state(state_dir, target="live", generation=7)
    tool_input = write_payload()

    output = approval.build_approval_output(
        "Channel write: TEST:PV=42.0", hook_input_for(tool_input)
    )

    assert (
        f"Target: LIVE MACHINE ({LIVE_ENDPOINT})"
        in (output["hookSpecificOutput"]["permissionDecisionReason"])
    )
    stamps = stamps_in(state_dir)
    assert len(stamps) == 1
    stamp = stamps[0]
    assert stamp["target"] == "live"
    assert stamp["generation"] == 7
    assert stamp["server_pid"] == SERVER_PID
    assert stamp["tool"] == "channel_write"


@pytest.mark.unit
def test_the_stamp_is_filed_under_the_key_the_server_looks_it_up_by(
    approval, state_dir, resolvable_session
):
    """One derivation, stated twice: the two spellings must produce one key.

    The hook cannot import the server's module — it runs outside that venv — so
    the key algorithm is written on both sides. Comparing them here is what
    keeps that duplication from silently drifting into two different keys, which
    would disable the cross-check without failing anything.
    """
    from osprey.mcp_server.control_system.tools import channel_write as tool

    write_state(state_dir, target="va")
    tool_input = write_payload(verification_level="readback")

    approval.build_approval_output("Channel write", hook_input_for(tool_input))

    server_key = tool._approval_stamp_key(tool_input["operations"], "readback")
    expected = state_dir / f"{tool.APPROVAL_STAMP_PREFIX}{server_key}{tool.APPROVAL_STAMP_SUFFIX}"
    assert expected.exists(), "the server would look this approval up under a name nothing wrote"
    assert approval.write_approval_key(tool_input) == server_key


@pytest.mark.unit
def test_an_unresolvable_session_stamps_an_unpublished_binding(
    approval, state_dir, resolvable_session
):
    """A prompt that could not name the target still records what it showed.

    The operator was told the target is unknown. That is a claim, and the server
    has to be able to tell it apart from "a target was published" — so the stamp
    exists and its binding is null, rather than the stamp being absent (which
    means "no comparison").
    """
    output = approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert (
        "Target: deployment baseline (state unavailable)"
        in (output["hookSpecificOutput"]["permissionDecisionReason"])
    )
    stamps = stamps_in(state_dir)
    assert len(stamps) == 1
    assert stamps[0]["target"] is None
    assert stamps[0]["generation"] is None
    assert stamps[0]["server_pid"] is None


# ---------------------------------------------------------------------------
# what is NOT stamped
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_other_tools_are_not_stamped(approval, state_dir, resolvable_session):
    """Only a write binds itself to a target; every other ask leaves nothing."""
    write_state(state_dir, target="live")

    approval.build_approval_output(
        "Tool: queue_start", hook_input_for({"foo": "bar"}, tool_name="mcp__bluesky__queue_start")
    )

    assert stamps_in(state_dir) == []


@pytest.mark.unit
def test_the_legacy_single_channel_payload_is_not_stamped(approval, state_dir, resolvable_session):
    """A payload shape the tool does not accept cannot be correlated with a call.

    The hook still renders these (it describes them for the human), but the MCP
    tool takes ``operations`` only — so there is no argument list both sides
    would derive the same key from, and a stamp keyed on a guess would be worse
    than none.
    """
    write_state(state_dir, target="live")

    approval.build_approval_output(
        "Channel write: TEST:PV=1.0", hook_input_for({"channel": "TEST:PV", "value": 1.0})
    )

    assert stamps_in(state_dir) == []


@pytest.mark.unit
def test_an_unwritable_state_directory_still_renders_the_prompt(
    approval, reader, resolvable_session, monkeypatch, tmp_path
):
    """Fail-open: no stamp is a missed cross-check, a lost prompt is a lost gate."""
    missing = tmp_path / "nope" / "control_target"
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: str(missing / "\0bad"))

    output = approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert "OSPREY APPROVAL REQUIRED" in output["hookSpecificOutput"]["permissionDecisionReason"]
    assert output["hookSpecificOutput"]["permissionDecision"] == "ask"


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_expired_stamps_are_swept_when_a_new_one_is_written(
    approval, state_dir, resolvable_session
):
    """The directory must not grow one file per write for the life of a project."""
    write_state(state_dir, target="live")
    stale = state_dir / "write_approval_deadbeef.json"
    stale.write_text("{}", encoding="utf-8")
    expired = time.time() - approval.WRITE_APPROVAL_TTL_S - 60
    os.utime(stale, (expired, expired))

    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert not stale.exists()
    assert len(stamps_in(state_dir)) == 1


@pytest.mark.unit
def test_a_stamp_never_looks_like_a_server_state_file(approval, state_dir, resolvable_session):
    """The reader globs ``target_state_*.json``; a stamp must never match it.

    A stamp picked up as a state file would be a record with no target at all
    answering for a live session, and the sweeper on the writer's side would
    treat it as a dead server's leftovers.
    """
    write_state(state_dir, target="live")

    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert list(state_dir.glob("target_state_*.json")) == [
        state_dir / f"target_state_{SERVER_PID}.json"
    ]
    assert approval._target_line(hook_input_for(write_payload())).startswith("Target: LIVE MACHINE")
