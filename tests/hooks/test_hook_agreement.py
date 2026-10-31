"""The two write hooks agree: ``osprey_writes_check`` denies exactly where ``osprey_approval`` defers.

Both hooks ask the stdlib twin the same two questions about a call (which
machine it writes to, and the write verdict on that machine), so the deny and
the missing prompt cannot part company. A defer emits no decision, and is only
safe because the deny behind it is guaranteed; a deny with a prompt in front of
it would ask the human to approve a call the next hook refuses.

The lane-addressed queue tools are outside the equivalence: ``writes_check``
carves them out by lane, and the approval hook defers a start only on its own
lane's verdict (pinned in ``test_approval_hook.py``).

Every stage-1 tool is also pinned here end to end: a call the verdict allows is
asked and leaves the ``approval_ask_<id>.json`` stamp its in-tool record check
reads, and a call it refuses defers and leaves none. The in-chat route answers
"ask" under ``permission_mode`` ``default`` and leaves the same stamp, which
``require_approval_record`` accepts.

Hooks run as subprocesses through ``hook_runner``, the way Claude Code runs
them, against a record written under the agent-data root the hook derives.
"""

from __future__ import annotations

import json

import pytest
import yaml

from osprey_connectors.control_context import _record_path_under
from osprey_connectors_tests._harness import (
    pin_identity,
    state_dir_under,
    write_control_context,
)

#: A real-machine deployment with a simulator beside it, whose tier allows
#: writes: every refusal over it comes from the switch or the record.
TWO_TARGETS = {
    "type": "epics",
    "writes": "allowed",
    "connector": {"epics": {"prefix": "RING:"}, "virtual_accelerator": {"prefix": "VA:"}},
}

#: The same deployment with no tier stated, which is a tier that never writes.
NO_TIER = {
    "type": "epics",
    "connector": {"epics": {"prefix": "RING:"}, "virtual_accelerator": {"prefix": "VA:"}},
}

FACILITY_SERVER_TOOL = "mcp__magnets__set_current"
FACILITY_EXTRA = "mcp__orbit__apply_correction"

PREFIXES = [
    "mcp__controls__",
    "mcp__python__",
    "mcp__phoebus__",
    "mcp__bluesky__",
    "mcp__magnets__",
    "mcp__orbit__",
]

#: The hook_config a render writes for these servers: framework write tools, a
#: whole-server facility matcher, a declared facility write tool, and the queue
#: tools a plan lane addresses.
HOOK_CONFIG = {
    "server_prefixes": PREFIXES,
    "approval_prefixes": PREFIXES,
    "write_tools": [
        "mcp__controls__channel_write",
        "mcp__python__execute",
        "mcp__phoebus__phoebus_drive",
        "mcp__bluesky__queue_add",
        "mcp__bluesky__queue_start",
        "mcp__magnets__.*",
        FACILITY_EXTRA,
    ],
    "control_system_write_tools": [FACILITY_EXTRA],
    "lane_addressed_tools": ["queue_add", "queue_start", "queue_stop", "queue_remove"],
}

CHANNEL_WRITE_INPUT = {"operations": [{"channel": "RING:QF:SP", "value": 1.5}]}
PHOEBUS_DRIVE_INPUT = {"widget": "QF_setpoint", "verb": "set", "value": "1.5"}

#: Every write tool ``writes_check`` gates on the session or the real machine,
#: as ``(tool name, tool input)``.
GATED_CALLS = [
    pytest.param("mcp__controls__channel_write", CHANNEL_WRITE_INPUT, id="channel_write"),
    pytest.param(
        "mcp__python__execute",
        {"code": "write_channel('RING:QF:SP', 1.5)", "execution_mode": "readwrite"},
        id="execute",
    ),
    pytest.param("mcp__phoebus__phoebus_drive", PHOEBUS_DRIVE_INPUT, id="phoebus_drive"),
    pytest.param(FACILITY_SERVER_TOOL, {"value": 1.5}, id="facility-server"),
    pytest.param(FACILITY_EXTRA, {"value": 1.5}, id="facility-extra"),
]

_CORRUPT = "corrupt"

#: The deployments the matrix runs over, as ``(section, record)``: ``record``
#: is ``None`` for no record at all, :data:`_CORRUPT` for one that cannot be
#: read, or ``(target, switched-on machines)``.
DEPLOYMENTS = [
    pytest.param(TWO_TARGETS, None, id="no-record"),
    pytest.param(TWO_TARGETS, _CORRUPT, id="unreadable-record"),
    *(
        pytest.param(TWO_TARGETS, (target, on), id=f"{target}-on-{'+'.join(on) or 'none'}")
        for target in ("live", "va")
        for on in ((), ("live",), ("va",), ("live", "va"))
    ),
    pytest.param(NO_TIER, ("live", ("live", "va")), id="no-tier-everything-on"),
]

APPROVAL_ALWAYS = {"enabled": True, "default_policy": "always"}


def _config(tmp_path, section):
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump({"approval": APPROVAL_ALWAYS, "control_system": section}))
    return path


def _decision(result):
    if result is None:
        return None
    return result["hookSpecificOutput"]["permissionDecision"]


def _write_record(root, record):
    if record is None:
        return
    if record == _CORRUPT:
        path = _record_path_under(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        return
    target, on = record
    write_control_context(
        root, target=target, generation=3, switch=dict.fromkeys(on, "on") if on else None
    )


@pytest.mark.parametrize(("section", "record"), DEPLOYMENTS)
@pytest.mark.parametrize(("tool_name", "tool_input"), GATED_CALLS)
def test_writes_check_denies_exactly_where_the_approval_hook_defers(
    tmp_path, hook_runner, section, record, tool_name, tool_input
):
    """Over every gated tool, target, switch and unplaceable record: deny iff defer.

    The approval policy asks for everything, so every call this hook does not
    defer gets a decision; only a defer leaves none.
    """
    config = _config(tmp_path, section)
    _write_record(tmp_path / "var" / "agent_data", record)

    def run(hook):
        return _decision(
            hook_runner(
                hook,
                tool_name,
                tool_input,
                config_path=config,
                cwd=tmp_path,
                hook_config=HOOK_CONFIG,
            )
        )

    denied = run("osprey_writes_check.py") == "deny"
    approval = run("osprey_approval.py")

    assert denied == (approval is None), (
        f"writes_check {'denied' if denied else 'allowed'} but approval answered {approval!r}"
    )
    if not denied:
        assert approval == "ask"


# ---------------------------------------------------------------------------
# Stage-1 tools: an allowed call leaves the stamp, a refused one defers without
# ---------------------------------------------------------------------------

#: Every stage-1 tool, as ``(tool name, tool input, short name)``.
STAGE1_CALLS = [
    pytest.param(
        "mcp__controls__channel_write", CHANNEL_WRITE_INPUT, "channel_write", id="channel_write"
    ),
    pytest.param("mcp__bluesky__queue_start", {}, "queue_start", id="queue_start"),
    pytest.param("mcp__bluesky__queue_add", {"draft_revision": 3}, "queue_add", id="queue_add"),
    pytest.param(
        "mcp__phoebus__phoebus_drive", PHOEBUS_DRIVE_INPUT, "phoebus_drive", id="phoebus_drive"
    ),
]

SESSION_ID = "4f1c2a7e-0000-4000-8000-0000000000aa"


@pytest.fixture
def stamped_root(tmp_path, monkeypatch):
    """A stamped agent-data root and a pinned identity, shared by hook and test.

    The hook child resolves the stamp directory from both variables; the
    in-process ``target_state.state_dir()`` the record check reads resolves it
    from the same two, so the stamp is looked up where it was filed.
    """
    root = tmp_path / "agent_data"
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    pin_identity(monkeypatch)
    state_dir_under(root).mkdir(parents=True)
    return root


def _ask(tmp_path, hook_runner, tool_name, tool_input, tool_use_id, permission_mode="default"):
    return hook_runner(
        "osprey_approval.py",
        tool_name,
        tool_input,
        config_path=_config(tmp_path, TWO_TARGETS),
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
        hook_input_extra={
            "hook_event_name": "PreToolUse",
            "tool_use_id": tool_use_id,
            "session_id": SESSION_ID,
            "permission_mode": permission_mode,
        },
    )


def _ask_stamp(root, tool_use_id):
    path = state_dir_under(root) / f"approval_ask_{tool_use_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@pytest.mark.parametrize(("tool_name", "tool_input", "short_name"), STAGE1_CALLS)
def test_a_stage1_call_the_verdict_allows_is_asked_and_stamped(
    tmp_path, hook_runner, stamped_root, tool_name, tool_input, short_name
):
    """The machine switched on: an ask, and the stamp the tool's record check reads."""
    write_control_context(stamped_root, target="live", switch={"live": "on"})

    result = _ask(tmp_path, hook_runner, tool_name, tool_input, f"toolu_allowed_{short_name}")

    assert _decision(result) == "ask"
    stamp = _ask_stamp(stamped_root, f"toolu_allowed_{short_name}")
    assert stamp is not None
    assert stamp["tool"] == tool_name
    assert stamp["session_id"] == SESSION_ID
    assert stamp["approver"] == "human"


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "short_name"),
    [param for param in STAGE1_CALLS if param.id != "queue_add"],
)
def test_a_stage1_call_the_verdict_refuses_defers_with_no_stamp(
    tmp_path, hook_runner, stamped_root, tool_name, tool_input, short_name
):
    """The machine switched off: no decision and no stamp, so the tool refuses too."""
    write_control_context(stamped_root, target="live")

    result = _ask(tmp_path, hook_runner, tool_name, tool_input, f"toolu_refused_{short_name}")

    assert _decision(result) is None
    assert _ask_stamp(stamped_root, f"toolu_refused_{short_name}") is None


def test_an_add_the_verdict_refuses_keeps_its_prompt(tmp_path, hook_runner, stamped_root):
    """The one stage-1 tool never deferred: an add the verdict refuses stages tokenless.

    ``writes_check`` carves ``queue_add`` out and nothing refuses it in-tool,
    so a defer would remove its only gate; it is asked, and stamped, as ever.
    """
    write_control_context(stamped_root, target="live")

    result = _ask(
        tmp_path, hook_runner, "mcp__bluesky__queue_add", {"draft_revision": 3}, "toolu_add_off"
    )

    assert _decision(result) == "ask"
    assert _ask_stamp(stamped_root, "toolu_add_off") is not None


# ---------------------------------------------------------------------------
# The in-chat route leaves the record the tool accepts
# ---------------------------------------------------------------------------


def _record_check(tool, tool_use_id):
    from osprey.audit.call import call_scope
    from osprey.mcp_server.approval_record import require_approval_record

    with call_scope(tool_use_id, None):
        return require_approval_record(tool, {"approval": APPROVAL_ALWAYS})


def test_an_in_chat_ask_leaves_the_stamp_the_record_check_accepts(
    tmp_path, hook_runner, stamped_root
):
    """``permission_mode`` default, the hook answers "ask": the tool finds its record.

    In chat the card is the operator session's ``can_use_tool``, reached only
    after this hook answered "ask"; the tool then runs under the same tool-use
    id and reads the stamp this hook filed for it.
    """
    write_control_context(stamped_root, target="live", switch={"live": "on"})

    result = _ask(
        tmp_path, hook_runner, "mcp__controls__channel_write", CHANNEL_WRITE_INPUT, "toolu_chat"
    )

    assert _decision(result) == "ask"
    assert _ask_stamp(stamped_root, "toolu_chat")["permission_mode"] == "default"
    assert _record_check("channel_write", "toolu_chat") is None


def test_a_deferred_call_leaves_no_record_for_the_tool_to_accept(
    tmp_path, hook_runner, stamped_root
):
    """A defer files nothing, so a static "ask" after it cannot vouch for the call."""
    from osprey.mcp_server.approval_record import APPROVAL_MISSING

    write_control_context(stamped_root, target="live")

    result = _ask(
        tmp_path, hook_runner, "mcp__controls__channel_write", CHANNEL_WRITE_INPUT, "toolu_deferred"
    )

    assert _decision(result) is None
    refusal = _record_check("channel_write", "toolu_deferred")
    assert refusal is not None
    assert refusal.error_type == APPROVAL_MISSING


# ---------------------------------------------------------------------------
# Governed render: the position changes the prompt, never the agreement
# ---------------------------------------------------------------------------

GOVERNED_HOOK_CONFIG = {
    **HOOK_CONFIG,
    "governed_tools": ["mcp__controls__channel_write", "mcp__python__execute"],
}


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        pytest.param("mcp__controls__channel_write", CHANNEL_WRITE_INPUT, id="channel_write"),
        pytest.param(
            "mcp__python__execute",
            {"code": "write_channel('RING:QF:SP', 1.5)", "execution_mode": "readwrite"},
            id="execute",
        ),
    ],
)
@pytest.mark.parametrize(
    ("position", "expected"), [(None, None), ("asked", "ask"), ("allowed", "allow")]
)
def test_a_governed_write_is_denied_exactly_where_it_defers(
    tmp_path, hook_runner, tool_name, tool_input, position, expected
):
    config = _config(tmp_path, TWO_TARGETS)
    write_control_context(
        tmp_path / "var" / "agent_data",
        target="va",
        generation=3,
        switch={"va": position} if position else None,
    )

    def run(hook):
        return _decision(
            hook_runner(
                hook,
                tool_name,
                tool_input,
                config_path=config,
                cwd=tmp_path,
                hook_config=GOVERNED_HOOK_CONFIG,
            )
        )

    assert (run("osprey_writes_check.py") == "deny") == (position is None)
    assert run("osprey_approval.py") == expected
