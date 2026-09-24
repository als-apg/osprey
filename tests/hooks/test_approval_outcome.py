"""The answer to an approval prompt, recorded from the events that follow it.

The harness has no hook event for a human "No". So the approval hook leaves an
ask stamp keyed by the call's ``tool_use_id`` when it asks, and reads the
answer off what the harness does emit: ``PostToolUse`` / ``PostToolUseFailure``
for that id means the call ran (``approved``), and a stamp of the same
conversation still unconsumed at ``Stop`` / ``StopFailure`` means it did not
(``denied``). Both outcome branches must print nothing — the harness's flow is
not theirs to change.

Every end-to-end case runs the real hook script as a subprocess, the way the
harness does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._control_context_fixtures import pin_identity, state_dir_under

HOOK = "osprey_approval.py"
CHANNEL_WRITE = "mcp__controls__channel_write"
CHANNEL_READ = "mcp__controls__channel_read"
WRITE_INPUT = {"operations": [{"channel": "TEST:PV", "value": 1.0}]}

HOOK_CONFIG = {
    "server_prefixes": ["mcp__controls__"],
    "approval_prefixes": ["mcp__controls__"],
}

APPROVAL_CONFIG = {
    "approval": {
        "enabled": True,
        "default_policy": "always",
        "tools": {"channel_write": "always", "channel_read": "skip"},
    },
    "control_system": {"writes_enabled": True},
}

CONVERSATION = "conv-1111"
OTHER_CONVERSATION = "conv-2222"


@pytest.fixture
def deployment(tmp_path, monkeypatch, make_config):
    """A config, an agent-data root and one pinned identity for both halves."""
    identity = pin_identity(monkeypatch)
    root = tmp_path / "agent_data"
    state = state_dir_under(root)
    state.mkdir(parents=True)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    config = make_config(APPROVAL_CONFIG)

    class _Deployment:
        state_dir = state
        config_path = config
        cwd = tmp_path
        who = identity

        @staticmethod
        def ledger() -> list[dict]:
            paths = list(Path(tmp_path).rglob("hook_approval.jsonl"))
            records: list[dict] = []
            for path in paths:
                records.extend(
                    json.loads(line) for line in path.read_text().splitlines() if line.strip()
                )
            return records

        @staticmethod
        def ask_stamps() -> list[Path]:
            return sorted(state.glob("approval_ask_*.json"))

    return _Deployment


def _event(event, tool_use_id, session=CONVERSATION, **extra):
    payload = {"hook_event_name": event, "tool_use_id": tool_use_id, "session_id": session}
    payload.update(extra)
    return payload


def _ask(hook_runner, deployment, tool_use_id, session=CONVERSATION):
    result = hook_runner(
        HOOK,
        CHANNEL_WRITE,
        WRITE_INPUT,
        config_path=deployment.config_path,
        cwd=deployment.cwd,
        hook_config=HOOK_CONFIG,
        hook_input_extra=_event(
            "PreToolUse", tool_use_id, session=session, permission_mode="default"
        ),
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "ask"
    return result


def _after(hook_runner, deployment, event, tool_use_id, session=CONVERSATION, tool=CHANNEL_WRITE):
    return hook_runner(
        HOOK,
        tool,
        WRITE_INPUT,
        config_path=deployment.config_path,
        cwd=deployment.cwd,
        hook_config=HOOK_CONFIG,
        tool_response={"content": "ok"},
        hook_input_extra=_event(event, tool_use_id, session=session),
    )


def _stop(hook_runner, deployment, event="Stop", session=CONVERSATION):
    return hook_runner(
        HOOK,
        None,
        None,
        config_path=deployment.config_path,
        cwd=deployment.cwd,
        hook_config=HOOK_CONFIG,
        hook_input_extra={"hook_event_name": event, "session_id": session},
    )


def test_an_ask_leaves_a_stamp_keyed_by_the_tool_use_id(hook_runner, deployment):
    _ask(hook_runner, deployment, "toolu_ask1")

    (stamp_path,) = deployment.ask_stamps()
    assert stamp_path.name == "approval_ask_toolu_ask1.json"
    stamp = json.loads(stamp_path.read_text())
    assert stamp["tool_use_id"] == "toolu_ask1"
    assert stamp["tool"] == CHANNEL_WRITE
    assert stamp["session_id"] == CONVERSATION
    assert stamp["approver"] == "human"
    assert stamp["permission_mode"] == "default"

    (ask,) = deployment.ledger()
    assert ask["decision"] == "ask"
    assert ask["tool_use_id"] == "toolu_ask1"


def test_post_tool_use_files_approved_and_consumes_the_stamp(hook_runner, deployment):
    _ask(hook_runner, deployment, "toolu_ok")
    assert _after(hook_runner, deployment, "PostToolUse", "toolu_ok") is None

    ask, approved = deployment.ledger()
    assert approved["decision"] == "approved"
    assert approved["reason"] == "executed"
    assert approved["subject"] == CHANNEL_WRITE
    assert approved["tool_use_id"] == ask["tool_use_id"] == "toolu_ok"
    assert approved["detail"] == "approver=human mode=default"
    assert deployment.ask_stamps() == []


def test_post_tool_use_failure_files_approved_too(hook_runner, deployment):
    _ask(hook_runner, deployment, "toolu_failed")
    assert _after(hook_runner, deployment, "PostToolUseFailure", "toolu_failed") is None

    decisions = [record["decision"] for record in deployment.ledger()]
    assert decisions == ["ask", "approved"]
    assert deployment.ask_stamps() == []


def test_stop_files_denied_for_this_conversations_stamps_only(hook_runner, deployment):
    _ask(hook_runner, deployment, "toolu_mine")
    _ask(hook_runner, deployment, "toolu_theirs", session=OTHER_CONVERSATION)

    assert _stop(hook_runner, deployment) is None

    denied = [record for record in deployment.ledger() if record["decision"] == "denied"]
    assert [record["tool_use_id"] for record in denied] == ["toolu_mine"]
    assert denied[0]["reason"] == "not_executed"
    assert denied[0]["subject"] == CHANNEL_WRITE
    assert [path.name for path in deployment.ask_stamps()] == ["approval_ask_toolu_theirs.json"]


def test_stop_failure_closes_asks_too(hook_runner, deployment):
    _ask(hook_runner, deployment, "toolu_errored")
    assert _stop(hook_runner, deployment, event="StopFailure") is None

    assert [record["decision"] for record in deployment.ledger()] == ["ask", "denied"]
    assert deployment.ask_stamps() == []


def test_a_call_never_asked_files_nothing(hook_runner, deployment):
    assert _after(hook_runner, deployment, "PostToolUse", "toolu_never", tool=CHANNEL_READ) is None
    assert _stop(hook_runner, deployment) is None
    assert deployment.ledger() == []


@pytest.mark.parametrize(
    "payload",
    [
        _event("PostToolUse", "toolu_quiet"),
        _event("PostToolUseFailure", "toolu_quiet"),
        {"hook_event_name": "Stop", "session_id": CONVERSATION},
        {"hook_event_name": "StopFailure", "session_id": CONVERSATION},
    ],
)
def test_the_outcome_branches_print_nothing(hook_runner, hook_runner_raw, deployment, payload):
    _ask(hook_runner, deployment, "toolu_quiet")
    returncode, stdout, _stderr = hook_runner_raw(
        HOOK,
        CHANNEL_WRITE,
        WRITE_INPUT,
        config_path=deployment.config_path,
        cwd=deployment.cwd,
        hook_config=HOOK_CONFIG,
        hook_input_extra=payload,
    )
    assert returncode == 0
    assert stdout == ""


def test_a_dispatch_run_names_the_policy_as_approver(hook_runner, deployment, monkeypatch):
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    _ask(hook_runner, deployment, "toolu_dispatch")
    (stamp_path,) = deployment.ask_stamps()
    assert json.loads(stamp_path.read_text())["approver"] == "dispatch_policy"

    _after(hook_runner, deployment, "PostToolUse", "toolu_dispatch")
    approved = deployment.ledger()[-1]
    assert approved["detail"].startswith("approver=dispatch_policy ")


def test_the_middleware_reads_a_stamp_only_the_hook_wrote(hook_runner, deployment):
    """The hook writes, the MCP audit middleware reads, under one agent-data env."""
    from osprey.mcp_server import audit_middleware

    _ask(hook_runner, deployment, "toolu_cross")

    assert audit_middleware._approval_answered("toolu_cross") == ("human", "default")
    assert audit_middleware._approval_answered("toolu_absent") is None
    # Read, never consumed: the hook's PostToolUse owns the stamp.
    assert [path.name for path in deployment.ask_stamps()] == ["approval_ask_toolu_cross.json"]


def test_the_ask_stamp_spelling_is_the_middlewares(hook_module):
    from osprey.mcp_server import audit_middleware

    approval = hook_module("osprey_approval")
    assert approval.APPROVAL_ASK_PREFIX == audit_middleware.APPROVAL_ASK_PREFIX
    assert approval.APPROVAL_ASK_SUFFIX == audit_middleware.APPROVAL_ASK_SUFFIX


def test_ask_stamps_keep_their_own_ttl(hook_module, tmp_path):
    """Pruning an ask stamp uses the ask TTL, not the write-approval hour."""
    import os
    import time

    approval = hook_module("osprey_approval")
    two_hours_old = tmp_path / "approval_ask_toolu_old.json"
    two_hours_old.write_text("{}")
    stamp = time.time() - 7200
    os.utime(two_hours_old, (stamp, stamp))

    approval._prune_approval_stamps(
        str(tmp_path), approval.APPROVAL_ASK_PREFIX, ttl_s=approval.APPROVAL_ASK_TTL_S
    )
    assert two_hours_old.exists()

    approval._prune_approval_stamps(str(tmp_path), approval.APPROVAL_ASK_PREFIX)
    assert not two_hours_old.exists()
