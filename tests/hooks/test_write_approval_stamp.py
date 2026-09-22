"""The stamp a rendered `channel_write` approval leaves for the controls server.

Approval is enforced out here, in a PreToolUse hook, before the MCP server is
called at all. The server's earliest possible observation of the deployment's
target is therefore its own tool entry — which is *after* the human clicked, and
a target switch can land while they are still deciding. Nothing the server reads
on its own covers that window.

So the hook writes down the binding it rendered the prompt against, beside the
control-context record, keyed by the write payload — the only thing that
provably crosses the gap between a hook process and an MCP tool call. The server
reads it back and refuses a write whose approval was granted against a different
`(target, generation)`.

Three properties matter and are pinned here: the stamp carries the SAME record
the `Target:` line was rendered from (a stamp describing a different read would
be a second opinion about identity); the name the hook files it under is the
name the server looks it up by (the derivation is stated twice — hooks run
outside the osprey venv — so the two spellings are exercised against each
other); and the stamp says WHICH audit session rendered it, so two sessions
sharing one agent-data root neither overwrite nor cross-check each other's
approvals.
"""

from __future__ import annotations

import json
import logging
import os
import time

import pytest

from tests._control_context_fixtures import (
    state_dir_under,
    write_control_context,
    write_server_report,
)

LIVE_ENDPOINT = "pva://live-gw.example.org:5075"
VA_ENDPOINT = "pva://127.0.0.1:5074"

#: Per-target display metadata, as a live controls server publishes it. The
#: `Target:` line is rendered from this and from nothing else, so a test that
#: wants a named machine has to put a LIVE report on disk beside the record.
TARGET_METADATA = {
    "live": {"label": "LIVE MACHINE", "endpoint": LIVE_ENDPOINT, "real_machine": True},
    "va": {"label": "Virtual accelerator", "endpoint": VA_ENDPOINT, "real_machine": False},
}

#: The tool name Claude Code passes for the controls server's write tool.
CHANNEL_WRITE = "mcp__controls__channel_write"

SESSION_A = "web-terminal-session-a"
SESSION_B = "kernel:9f3c1a2b"


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
def root(tmp_path, monkeypatch):
    """A deployment's agent-data root, stamped so every reader resolves to it.

    Stamping ``OSPREY_AGENT_DATA_ROOT`` rather than patching one module's
    resolver is what keeps the two halves of this suite on one directory: the
    hook resolves the stamp path through `osprey_target_state`, and the server
    resolves the same path through `target_state.state_dir()` — two modules,
    one variable. Patching either alone leaves the other reading the
    developer's own deployment.
    """
    directory = tmp_path / "agent_data"
    state_dir_under(directory).mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(directory))
    return directory


@pytest.fixture
def state_dir(root):
    """The directory the record, the reports and the stamps share.

    One directory per identity below the root, asked of
    :func:`~tests._control_context_fixtures.state_dir_under` rather than joined
    here: a suite that spelled the hops itself would keep provisioning a
    directory no reader resolves, and an empty stamp listing is the only
    symptom that leaves behind.
    """
    return state_dir_under(root)


@pytest.fixture
def session(monkeypatch):
    """Run as one audit session, the way a web-terminal child is stamped."""
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_A)
    return SESSION_A


@pytest.fixture
def no_session(monkeypatch):
    """Run with no audit session at all, the way a bare ``claude`` does."""
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)


def publish(root, target="live", generation=4, *, with_server=True):
    """Put a deployment on disk: a record, plus a LIVE server naming the targets."""
    write_control_context(root, target=target, generation=generation)
    if with_server:
        write_server_report(
            root,
            os.getpid(),
            applied_target=target,
            applied_generation=generation,
            targets=TARGET_METADATA,
        )


#: Distinguishes "the caller said nothing about confirmation" from an explicit
#: ``confirm=False``, which is a different write and must key differently.
_UNSET = object()


def write_payload(channel="TEST:PV", value=42.0, confirm=_UNSET):
    """The tool input Claude Code hands the hook for a one-channel write.

    ``confirm`` is omitted by default, which is how the agent calls the tool
    when it leaves the decision to the deployment.
    """
    payload = {"operations": [{"channel": channel, "value": value}]}
    if confirm is not _UNSET:
        payload["confirm"] = confirm
    return payload


def hook_input_for(tool_input, tool_name=CHANNEL_WRITE):
    return {"tool_name": tool_name, "tool_input": tool_input, "cwd": os.getcwd()}


def stamp_files(directory):
    """Every stamp file in *directory*, by name."""
    return sorted(path.name for path in directory.glob("write_approval_*.json"))


def stamps_in(directory):
    """Every stamp file in *directory*, parsed, by name order."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("write_approval_*.json"))
    ]


def reason_of(output):
    return output["hookSpecificOutput"]["permissionDecisionReason"]


# ---------------------------------------------------------------------------
# what the prompt renders is what the stamp records
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("session")
def test_a_channel_write_ask_stamps_the_binding_it_rendered(approval, root, state_dir):
    """The ask carries the target line; the stamp carries the same record."""
    publish(root, target="live", generation=7)

    output = approval.build_approval_output(
        "Channel write: TEST:PV=42.0", hook_input_for(write_payload())
    )

    assert f"Target: LIVE MACHINE ({LIVE_ENDPOINT})" in reason_of(output)
    stamps = stamps_in(state_dir)
    assert len(stamps) == 1
    assert stamps[0]["target"] == "live"
    assert stamps[0]["generation"] == 7
    assert stamps[0]["tool"] == "channel_write"


@pytest.mark.usefixtures("session")
def test_the_stamp_names_the_audit_session_that_rendered_it(approval, root, state_dir):
    """``session`` is the third field of the key, and it comes off the environment.

    The server compares it with its own `posture_session()`, so a stamp that
    named nothing — or named a process instead of a session — would let one
    session's approval vouch for another's write.
    """
    publish(root, target="va", generation=2)

    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert stamps_in(state_dir)[0]["session"] == SESSION_A


@pytest.mark.usefixtures("no_session")
def test_a_session_less_render_stamps_a_null_session(approval, root, state_dir):
    """A bare ``claude`` has no audit session, and the stamp says so plainly.

    ``None`` is a claim of its own — "nothing can attribute this stamp" — and
    the server matches it only from a process that is equally unattributed.
    """
    publish(root, target="va", generation=2)

    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert stamps_in(state_dir)[0]["session"] is None


@pytest.mark.usefixtures("session")
def test_the_stamp_carries_no_process_identity(approval, root, state_dir):
    """No pid in the stamp: the context is the deployment's, not a process tree's.

    The pid that used to sit here was the one field of the old stamp that a
    reader could not resolve — the control-context record carries no server
    pid — so it was written null on every stamp and attributed nothing.
    """
    publish(root)

    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert "server_pid" not in stamps_in(state_dir)[0]


@pytest.mark.usefixtures("root", "session")
def test_an_unpublished_deployment_stamps_an_unpublished_binding(approval, state_dir):
    """A prompt that could not name the target still records what it showed.

    The operator was told the target is unknown. That is a claim, and the server
    has to be able to tell it apart from "a target was published" — so the stamp
    exists and its binding is null, rather than the stamp being absent (which
    means "no comparison").
    """
    output = approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert "Target: deployment baseline (state unavailable)" in reason_of(output)
    stamps = stamps_in(state_dir)
    assert len(stamps) == 1
    assert stamps[0]["target"] is None
    assert stamps[0]["generation"] is None
    assert stamps[0]["session"] == SESSION_A


# ---------------------------------------------------------------------------
# one name, derived twice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confirm", [_UNSET, True, False])
@pytest.mark.usefixtures("state_dir", "session")
def test_the_stamp_is_filed_under_the_name_the_server_looks_it_up_by(approval, root, confirm):
    """One derivation, stated twice: the two spellings must find one file.

    The hook cannot import the server's module — it runs outside that venv — so
    the name algorithm is written on both sides. Driving the server's own reader
    against a file only the hook wrote is what keeps that duplication from
    silently drifting into two different names, which would disable the
    cross-check without failing anything.

    Every value of ``confirm`` is driven, including its omission: the hook reads
    the field out of ``tool_input`` and the server is handed the tool's own
    parameter default, so "absent" has to hash the same on both sides or the
    ordinary write — the one that leaves confirmation to the deployment — is the
    one whose approval window silently stops being checked.
    """
    from osprey.mcp_server.control_system.tools import channel_write as tool

    publish(root, target="va", generation=3)
    tool_input = write_payload(confirm=confirm)
    # What the tool's own signature hands `_approval_stamp_key` when the agent
    # leaves `confirm` out of the call.
    server_confirm = None if confirm is _UNSET else confirm

    approval.build_approval_output("Channel write", hook_input_for(tool_input))

    found, binding = tool._read_approval_stamp(tool_input["operations"], server_confirm)
    assert found, "the server would look this approval up under a name nothing wrote"
    assert binding == ("va", 3)


def test_the_file_name_carries_the_session_as_well_as_the_payload(
    approval, root, state_dir, monkeypatch
):
    """Two sessions, one payload, two files — neither overwrites the other.

    Sharing an agent-data root is ordinary (two windows on one checkout), and
    an identical write is exactly what two operators doing the same thing
    produce. Keying only on the payload would give the second render's stamp to
    the first session's server, which is the one comparison that must never be
    made against somebody else's approval.
    """
    publish(root, target="live", generation=5)

    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_A)
    approval.build_approval_output("Channel write", hook_input_for(write_payload()))
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_B)
    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert len(stamp_files(state_dir)) == 2
    assert {stamp["session"] for stamp in stamps_in(state_dir)} == {SESSION_A, SESSION_B}


@pytest.mark.usefixtures("no_session")
def test_session_less_siblings_share_one_file(approval, root, state_dir):
    """Two unattributed renders of the same write collide, and that is benign.

    Nothing distinguishes one session-less process from another — that is what
    having no audit session means — so their stamps share a name. The colliding
    payloads are equal by construction (the name is the payload's own hash and
    the binding comes from one shared record), so the survivor says exactly what
    the overwritten one said.
    """
    publish(root, target="live", generation=5)

    approval.build_approval_output("Channel write", hook_input_for(write_payload()))
    first = stamp_files(state_dir)
    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert stamp_files(state_dir) == first
    assert len(first) == 1


def test_the_confirmation_setting_is_part_of_the_approval_identity(approval):
    """Three different writes, three different keys.

    ``confirm`` decides whether the machine is read back after the value goes
    out, so a prompt approved with one setting must not vouch for a call made
    with another. If the field dropped out of the hash these three would
    collide, and nothing else in the suite would notice.
    """
    keys = {
        approval.write_approval_key(write_payload(confirm=confirm))
        for confirm in (_UNSET, True, False)
    }

    assert len(keys) == 3
    assert None not in keys


# ---------------------------------------------------------------------------
# whose stamp is it — the server's side of the key
# ---------------------------------------------------------------------------


def test_a_server_claims_only_stamps_from_its_own_session(monkeypatch):
    """`_stamp_is_ours` is an equality on the audit session, and nothing looser."""
    from osprey.mcp_server.control_system.tools import channel_write as tool

    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_A)

    assert tool._stamp_is_ours({"session": SESSION_A})
    assert not tool._stamp_is_ours({"session": SESSION_B})
    assert not tool._stamp_is_ours({"session": None})
    assert not tool._stamp_is_ours({})
    assert not tool._stamp_is_ours("not a stamp")


def test_a_session_less_server_claims_only_session_less_stamps(monkeypatch):
    """With nothing to attribute by, "ours" is the other unattributed renders.

    That is the collision the file name cannot break, and it is safe in the one
    direction that matters: a stamp naming a session is never claimed by a
    process that has none.
    """
    from osprey.mcp_server.control_system.tools import channel_write as tool

    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)

    assert tool._stamp_is_ours({"session": None})
    assert not tool._stamp_is_ours({"session": SESSION_A})


def test_a_stamp_from_another_session_is_not_compared_against(
    approval, root, state_dir, monkeypatch
):
    """A misfiled stamp is ignored, not obeyed.

    The name already separates the two sessions, so the server normally never
    sees the other's file at all. This pins the second line of defence: the
    payload is the authority on whose approval it records, and a stamp whose
    session is not ours costs this call its comparison rather than producing a
    wrong one.
    """
    from osprey.mcp_server.control_system.tools import channel_write as tool

    publish(root, target="live", generation=5)
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_B)
    tool_input = write_payload()
    approval.build_approval_output("Channel write", hook_input_for(tool_input))
    stamp = next(state_dir.glob("write_approval_*.json"))

    # The other session's server, reading under its own name.
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_A)
    ours = state_dir / tool._approval_stamp_name(
        tool._approval_stamp_key(tool_input["operations"], None)
    )
    assert ours != stamp
    stamp.rename(ours)

    assert tool._read_approval_stamp(tool_input["operations"], None) == (False, None)


@pytest.mark.usefixtures("state_dir")
def test_a_key_miss_is_reported_when_this_session_has_other_stamps(
    approval, root, monkeypatch, caplog
):
    """The one failure mode of a two-party hash that nothing else would surface.

    If the hook is stamping for this session and none of its stamps is the one
    this call asked for, the two derivations disagree and every approval-window
    check goes quiet without a single failure.
    """
    from osprey.mcp_server.control_system.tools import channel_write as tool

    publish(root, target="live", generation=5)
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_A)
    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.tools.channel_write"):
        found, _binding = tool._read_approval_stamp(
            write_payload(channel="OTHER:PV")["operations"], None
        )

    assert found is False
    assert "deriving different keys" in caplog.text


@pytest.mark.usefixtures("state_dir", "no_session")
def test_a_session_less_server_does_not_report_a_key_miss(approval, root, caplog):
    """With no session, "this server's other stamps" is a question with no answer.

    Every session-less process on the checkout files under the same name, so a
    stamp found beside a miss is as likely a sibling's as this process's own —
    and a warning that told an operator to re-run ``osprey build`` on that
    evidence would be wrong most of the time.
    """
    from osprey.mcp_server.control_system.tools import channel_write as tool

    publish(root, target="live", generation=5)
    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.tools.channel_write"):
        found, _binding = tool._read_approval_stamp(
            write_payload(channel="OTHER:PV")["operations"], None
        )

    assert found is False
    assert caplog.text == ""


# ---------------------------------------------------------------------------
# what is NOT stamped
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("session")
def test_other_tools_are_not_stamped(approval, root, state_dir):
    """Only a write binds itself to a target; every other ask leaves nothing."""
    publish(root)

    approval.build_approval_output(
        "Tool: queue_start", hook_input_for({"foo": "bar"}, tool_name="mcp__bluesky__queue_start")
    )

    assert stamps_in(state_dir) == []


@pytest.mark.usefixtures("session")
def test_the_legacy_single_channel_payload_is_not_stamped(approval, root, state_dir):
    """A payload shape the tool does not accept cannot be correlated with a call.

    The hook still renders these (it describes them for the human), but the MCP
    tool takes ``operations`` only — so there is no argument list both sides
    would derive the same key from, and a stamp keyed on a guess would be worse
    than none.
    """
    publish(root)

    approval.build_approval_output(
        "Channel write: TEST:PV=1.0", hook_input_for({"channel": "TEST:PV", "value": 1.0})
    )

    assert stamps_in(state_dir) == []


@pytest.mark.usefixtures("session")
def test_an_unwritable_state_directory_still_renders_the_prompt(
    approval, reader, monkeypatch, tmp_path
):
    """Fail-open: no stamp is a missed cross-check, a lost prompt is a lost gate."""
    missing = tmp_path / "nope" / "control_target"
    monkeypatch.setattr(reader, "resolve_state_dir", lambda hook_input=None: str(missing / "\0bad"))

    output = approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert "OSPREY APPROVAL REQUIRED" in reason_of(output)
    assert output["hookSpecificOutput"]["permissionDecision"] == "ask"


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("session")
def test_expired_stamps_are_swept_when_a_new_one_is_written(approval, root, state_dir):
    """The directory must not grow one file per write for the life of a project."""
    publish(root)
    stale = state_dir / "write_approval_deadbeef_cafe.json"
    stale.write_text("{}", encoding="utf-8")
    expired = time.time() - approval.WRITE_APPROVAL_TTL_S - 60
    os.utime(stale, (expired, expired))

    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert not stale.exists()
    assert len(stamps_in(state_dir)) == 1


@pytest.mark.usefixtures("session")
def test_a_stamp_is_neither_the_record_nor_a_server_report(approval, root, state_dir):
    """The readers glob ``server_*.json`` and name ``control_context.json``.

    A stamp picked up as a report would be a server with no pid answering for a
    live deployment, and one picked up as the record would be a control context
    with no target at all.
    """
    publish(root, target="live", generation=5)

    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    assert [path.name for path in state_dir.glob("server_*.json")] == [f"server_{os.getpid()}.json"]
    assert len(stamp_files(state_dir)) == 1
    assert approval._target_line(hook_input_for(write_payload())).startswith("Target: LIVE MACHINE")


# ---------------------------------------------------------------------------
# the stamp a rendered `queue_start` approval leaves for the queue tool
# ---------------------------------------------------------------------------
#
# A start carries at most a lane id, so the thing the human approved is the
# queue they were SHOWN — a list that any later add, move or remove replaces.
# The prompt therefore writes down the queue token it listed, the queue tool
# quotes it back as `expected_plan_queue_uid`, and the bridge refuses a start
# whose queue has moved since. The three properties pinned below are the ones
# that make that binding trustworthy: one file per lane (parallel tool calls
# render both prompts before either tool runs, so a shared file would let the
# survivor unbind the other lane's start); a render that showed NO queue nulls
# every token of the session, so a previous prompt's token cannot outlive the
# prompt that replaced it; and a session-less render stamps nothing at all.

#: The tool name Claude Code passes for the Bluesky queue's start tool.
QUEUE_START = "mcp__bluesky__queue_start"

#: A two-lane deployment, which is the only shape in which a start is an
#: ADDRESS: each lane drives its own machine, so each needs its own stamp.
TWO_LANE_CONFIG = {
    "services": {
        "bluesky": {"port": 60101, "target": "live"},
        "bluesky_va": {"port": 60102, "target": "va"},
    }
}

#: A lane the rendered config publishes no port for — the bridge that was never
#: given an address, as distinct from one that did not answer.
UNADDRESSABLE_LANE_CONFIG = {
    "services": {
        "bluesky": {"port": 60101, "target": "live"},
        "bluesky_va": {"target": "va"},
    }
}


def queue_snapshot(uid="uid-1"):
    """`GET /queue` as the bridge answers it, carrying the queue's token."""
    return {
        "status": {"manager_state": "idle", "plan_queue_uid": uid},
        "items": [],
        "running_item": None,
    }


def queue_stamps_in(directory):
    """Every queue-start stamp in *directory*, parsed, keyed by file name."""
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in directory.glob("queue_start_approval_*.json")
    }


def queue_stamp_name(approval, session, lane):
    """The name the hook files a stamp for *session*/*lane* under."""
    return f"queue_start_approval_{approval.write_approval_session_slug(session)}_{lane}.json"


@pytest.fixture
def bridge(monkeypatch):
    """An address for lane 1. No test dials it — every fetch is patched out."""
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", "http://127.0.0.1:60101")


def render_start(approval, monkeypatch, *, snapshot, lane=None, config=None):
    """Render a start prompt whose bridge answer is pinned to *snapshot*.

    The listing itself is stubbed out: these tests are about the stamp, and the
    lines under it are pinned in `test_approval_queue_enrichment.py` against a
    real HTTP server. Stubbing the fetch is also what keeps them offline — the
    ports these lanes publish have nothing behind them.
    """
    monkeypatch.setattr(approval, "_queue_snapshot", lambda base_url: snapshot)
    monkeypatch.setattr(approval, "_queue_item_lines", lambda snapshot, base_url: [])
    tool_input = {"lane": lane} if lane else {}
    return approval._describe_queue_start(
        tool_input,
        config if config is not None else {},
        hook_input_for(tool_input, QUEUE_START),
        read_record=lambda: None,
    )


@pytest.mark.usefixtures("root", "session", "bridge")
def test_a_queue_start_render_stamps_the_lane_and_the_queue_token(approval, state_dir, monkeypatch):
    """The prompt listed a queue; the stamp says which lane's, and at which token."""
    lines = render_start(approval, monkeypatch, snapshot=queue_snapshot("uid-1"))

    stamps = queue_stamps_in(state_dir)
    name = queue_stamp_name(approval, SESSION_A, "bluesky")
    assert list(stamps) == [name]
    assert stamps[name]["lane"] == "bluesky"
    assert stamps[name]["plan_queue_uid"] == "uid-1"
    assert stamps[name]["ts"] == pytest.approx(time.time(), abs=60)
    assert "EVERY pending item" in "\n".join(lines)


@pytest.mark.usefixtures("root", "session", "bridge")
def test_two_lanes_prompted_in_one_turn_leave_one_queue_start_stamp_each(
    approval, state_dir, monkeypatch
):
    """Parallel tool calls render both prompts before either tool runs.

    One file per session would let whichever prompt rendered second overwrite
    the first lane's token, and that lane's start would then quote a token from
    a queue nobody showed its approver.
    """
    render_start(
        approval,
        monkeypatch,
        snapshot=queue_snapshot("uid-1"),
        lane="bluesky",
        config=TWO_LANE_CONFIG,
    )
    render_start(
        approval,
        monkeypatch,
        snapshot=queue_snapshot("uid-2"),
        lane="bluesky_va",
        config=TWO_LANE_CONFIG,
    )

    stamps = queue_stamps_in(state_dir)
    assert sorted(stamps) == sorted(
        [
            queue_stamp_name(approval, SESSION_A, "bluesky"),
            queue_stamp_name(approval, SESSION_A, "bluesky_va"),
        ]
    )
    assert {payload["lane"]: payload["plan_queue_uid"] for payload in stamps.values()} == {
        "bluesky": "uid-1",
        "bluesky_va": "uid-2",
    }


@pytest.mark.usefixtures("root", "session", "bridge")
def test_an_unreachable_bridge_nulls_every_queue_start_stamp_of_the_session(
    approval, state_dir, monkeypatch
):
    """A prompt that showed no queue must leave no token behind it.

    Both lanes are nulled, not only the one being rendered: what the approver
    is being shown right now is no queue at all, and a surviving token on the
    other lane would be a binding no prompt ever made.
    """
    render_start(
        approval,
        monkeypatch,
        snapshot=queue_snapshot("uid-1"),
        lane="bluesky",
        config=TWO_LANE_CONFIG,
    )
    render_start(
        approval,
        monkeypatch,
        snapshot=queue_snapshot("uid-2"),
        lane="bluesky_va",
        config=TWO_LANE_CONFIG,
    )

    lines = render_start(
        approval, monkeypatch, snapshot=None, lane="bluesky", config=TWO_LANE_CONFIG
    )

    stamps = queue_stamps_in(state_dir)
    assert len(stamps) == 2
    assert {payload["plan_queue_uid"] for payload in stamps.values()} == {None}
    assert {payload["lane"] for payload in stamps.values()} == {"bluesky", "bluesky_va"}
    assert "the bridge could not be reached" in "\n".join(lines)


@pytest.mark.usefixtures("root", "session", "bridge")
def test_a_start_naming_no_lane_nulls_every_queue_start_stamp_of_the_session(
    approval, state_dir, monkeypatch
):
    """The multi-lane no-lane return shows no queue either — and is the one a
    session reaches without the bridge ever being asked."""
    render_start(
        approval,
        monkeypatch,
        snapshot=queue_snapshot("uid-1"),
        lane="bluesky",
        config=TWO_LANE_CONFIG,
    )

    lines = render_start(
        approval, monkeypatch, snapshot=queue_snapshot("uid-9"), config=TWO_LANE_CONFIG
    )

    assert [payload["plan_queue_uid"] for payload in queue_stamps_in(state_dir).values()] == [None]
    assert "Queue contents: not shown" in "\n".join(lines)


@pytest.mark.usefixtures("root", "session", "bridge")
def test_an_unaddressable_lane_nulls_every_queue_start_stamp_of_the_session(
    approval, state_dir, monkeypatch
):
    """A config that publishes no port for the lane lists nothing, so it binds
    nothing."""
    render_start(
        approval,
        monkeypatch,
        snapshot=queue_snapshot("uid-1"),
        lane="bluesky",
        config=UNADDRESSABLE_LANE_CONFIG,
    )

    lines = render_start(
        approval,
        monkeypatch,
        snapshot=queue_snapshot("uid-9"),
        lane="bluesky_va",
        config=UNADDRESSABLE_LANE_CONFIG,
    )

    assert [payload["plan_queue_uid"] for payload in queue_stamps_in(state_dir).values()] == [None]
    assert "publishes no port" in "\n".join(lines)


@pytest.mark.usefixtures("root", "bridge")
def test_nulling_leaves_another_sessions_queue_start_stamps_alone(approval, state_dir, monkeypatch):
    """Two sessions share one agent-data root. One's empty queue is not the
    other's."""
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_A)
    render_start(approval, monkeypatch, snapshot=queue_snapshot("uid-a"))

    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_B)
    render_start(approval, monkeypatch, snapshot=queue_snapshot("uid-b"))
    render_start(approval, monkeypatch, snapshot=None)

    stamps = queue_stamps_in(state_dir)
    assert stamps[queue_stamp_name(approval, SESSION_A, "bluesky")]["plan_queue_uid"] == "uid-a"
    assert stamps[queue_stamp_name(approval, SESSION_B, "bluesky")]["plan_queue_uid"] is None


@pytest.mark.usefixtures("root", "no_session", "bridge")
def test_a_session_less_render_writes_no_queue_start_stamp(approval, state_dir, monkeypatch):
    """The write stamp's `anon` collision is benign because that name is the
    write's own hash. A queue token is nobody's in particular, so an
    unattributed process leaves none at all rather than one every other
    unattributed process would read as its own."""
    render_start(approval, monkeypatch, snapshot=queue_snapshot("uid-1"))

    assert queue_stamps_in(state_dir) == {}


@pytest.mark.usefixtures("root", "session", "bridge")
def test_expired_queue_start_stamps_are_swept_when_a_new_one_is_written(
    approval, state_dir, monkeypatch
):
    """Queue-start stamps share the write stamps' TTL and their pruning."""
    stale = state_dir / "queue_start_approval_deadbeefdeadbeef_bluesky_va.json"
    stale.write_text("{}", encoding="utf-8")
    expired = time.time() - approval.WRITE_APPROVAL_TTL_S - 60
    os.utime(stale, (expired, expired))

    render_start(approval, monkeypatch, snapshot=queue_snapshot("uid-1"))

    assert not stale.exists()
    assert len(queue_stamps_in(state_dir)) == 1


@pytest.mark.usefixtures("session", "bridge")
def test_a_queue_start_render_leaves_write_approval_stamps_alone(
    approval, root, state_dir, monkeypatch
):
    """Two stamp kinds, two prefixes, one directory: neither the prune nor the
    null pass may reach across, or an approved write loses the binding a start
    never had anything to do with."""
    publish(root, target="live", generation=5)
    approval.build_approval_output("Channel write", hook_input_for(write_payload()))

    render_start(approval, monkeypatch, snapshot=queue_snapshot("uid-1"))
    render_start(approval, monkeypatch, snapshot=None)

    assert len(stamps_in(state_dir)) == 1
    assert stamps_in(state_dir)[0]["generation"] == 5
