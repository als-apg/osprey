"""Stage 2 of ``osprey_writes_check``: which machine a write tool writes to.

A framework write tool writes to the session's target, and follows that
target's switch. A facility write tool — every tool of a whole-server
``writes_check`` custom server, and every ``control_system.write_tools`` entry —
follows the real machine's switch, whatever the session points at: the
deployment declared it, and only ``live``'s switch covers it. A refusal is worded
by the twin's ``refusal_phrase`` and names no config key.

The hook is run as a subprocess through ``hook_runner``, the way Claude Code
runs it, against a record written under the agent-data root the hook derives
from its own ``cwd`` (or the stamped one), so the root anchor and the read are
both real.
"""

from __future__ import annotations

import json

import pytest
import yaml

import osprey.templates.claude_code.claude.hooks.osprey_target_state as twin
from osprey.agent_runner.write_tools import read_only_disallowed_tools
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    pin_identity,
    state_dir_under,
    write_control_context,
    write_payload,
)

#: A switch-capable deployment whose tier allows writes, so every refusal below
#: can only have come from the switch, the owner or the record.
TWO_TARGETS = {
    "type": "epics",
    "writes": "allowed",
    "connector": {
        "epics": {"prefix": "RING:"},
        "virtual_accelerator": {"prefix": "VA:"},
    },
}

#: A simulator-only deployment: it has no ``live`` target at all.
SIMULATOR_ONLY = {"type": "virtual_accelerator", "writes": "allowed"}

#: A facility-custom server gated whole, and a declared write tool, as the
#: render writes them into hook_config.
FACILITY_SERVER_TOOL = "mcp__magnets__set_current"
FACILITY_SERVER_READ = "mcp__magnets__read_current"
FACILITY_EXTRA = "mcp__orbit__apply_correction"
HOOK_CONFIG = {
    "server_prefixes": [
        "mcp__controls__",
        "mcp__python__",
        "mcp__bluesky__",
        "mcp__magnets__",
        "mcp__orbit__",
    ],
    "write_tools": [
        "mcp__controls__channel_write",
        "mcp__bluesky__queue_add",
        "mcp__bluesky__queue_start",
        "mcp__python__execute",
        "mcp__magnets__.*",
        FACILITY_EXTRA,
    ],
    "control_system_write_tools": [FACILITY_EXTRA],
    "lane_addressed_tools": ["queue_add", "queue_start", "queue_stop", "queue_remove"],
}

SESSION_KEY = "4f1c2a7e-0000-4000-8000-000000000001"


@pytest.fixture(autouse=True)
def stamped_root(tmp_path, monkeypatch):
    """A clean verdict environment, stamped at the root the hook would derive."""
    root = agent_data_root(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    isolate_verdict_environment(monkeypatch, root)
    return root


def agent_data_root(repo_root):
    """The agent-data root the hook DERIVES from *repo_root* when unstamped."""
    return repo_root / "var" / "agent_data"


def record_path(repo_root):
    """Where the record lands under *repo_root*, directory created."""
    directory = state_dir_under(agent_data_root(repo_root))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "control_context.json"


def write_record(repo_root, target, on=(), posture=None):
    """The session on *target*, with the switch on for each of *on*."""
    write_control_context(
        agent_data_root(repo_root),
        target=target,
        generation=3,
        posture=posture,
        switch=dict.fromkeys(on, "on") if on else None,
    )


def call(tmp_path, hook_runner, config, tool, tool_input=None):
    """Run the hook against one call under :data:`HOOK_CONFIG`."""
    return hook_runner(
        "osprey_writes_check.py",
        tool,
        tool_input if tool_input is not None else {"value": 1.5},
        config_path=config,
        cwd=tmp_path,
        hook_config=HOOK_CONFIG,
    )


def channel_write(tmp_path, hook_runner, config):
    return call(
        tmp_path,
        hook_runner,
        config,
        "mcp__controls__channel_write",
        {"operations": [{"channel": "RING:QF:SP", "value": 1.5}]},
    )


def reason_of(result):
    """The operator-facing refusal text, asserting there was one."""
    assert result is not None, "expected a deny, got a pass-through"
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    reason = output["permissionDecisionReason"]
    for word in ("writes_enabled", "control_system.", "profile.yml", "osprey build"):
        assert word not in reason, word
    return reason


def deny_text(reason, kind=None):
    return "\U0001f512 WRITES OFF — " + twin.refusal_phrase(reason, kind)


# ---------------------------------------------------------------------------
# framework write tools follow the session's target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("target", "kind"), [("live", "real"), ("va", "simulator")])
def test_a_framework_tool_is_refused_while_its_target_is_off(
    tmp_path, hook_runner, make_config, target, kind
):
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, target)

    assert reason_of(channel_write(tmp_path, hook_runner, config)) == deny_text("switch_off", kind)


@pytest.mark.parametrize("target", ["live", "va"])
def test_a_framework_tool_is_allowed_while_its_target_is_on(
    tmp_path, hook_runner, make_config, target
):
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, target, on=[target])

    assert channel_write(tmp_path, hook_runner, config) is None


def test_another_target_switched_on_does_not_arm_the_session(tmp_path, hook_runner, make_config):
    """The session on `live` with only the simulator on is refused, for the real machine."""
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "live", on=["va"])

    assert reason_of(channel_write(tmp_path, hook_runner, config)) == deny_text(
        "switch_off", "real"
    )


def test_a_narrowed_target_is_off_with_its_switch_on(tmp_path, hook_runner, make_config):
    """A recorded `sandbox` for the target refuses as `switch_off`."""
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "live", on=["live"], posture={"live": "sandbox"})

    assert reason_of(channel_write(tmp_path, hook_runner, config)) == deny_text(
        "switch_off", "real"
    )


def test_no_record_under_a_stamped_root_is_off(tmp_path, hook_runner, make_config):
    """Nobody switched anything on: the baseline, and its switch is off."""
    config = make_config({"control_system": TWO_TARGETS})

    assert reason_of(channel_write(tmp_path, hook_runner, config)) == deny_text(
        "switch_off", "real"
    )


def test_a_corrupt_record_under_a_stamped_root_refuses(tmp_path, hook_runner, make_config):
    """A record that cannot be read places the call nowhere, and that refuses."""
    config = make_config({"control_system": TWO_TARGETS})
    record_path(tmp_path).write_text('{"schema": ', encoding="utf-8")

    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    assert reason == deny_text("target_unknown")


def test_readonly_execution_passes_with_every_switch_off(tmp_path, hook_runner, make_config):
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "live")

    result = call(
        tmp_path,
        hook_runner,
        config,
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readonly"},
    )

    assert result is None


# ---------------------------------------------------------------------------
# facility write tools follow `live`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", [FACILITY_SERVER_TOOL, FACILITY_EXTRA])
def test_a_facility_tool_is_refused_while_live_is_off_with_the_session_on_va(
    tmp_path, hook_runner, make_config, tool
):
    """The session's own switch is on; the real machine's is not."""
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "va", on=["va"])

    reason = reason_of(call(tmp_path, hook_runner, config, tool))

    assert reason == deny_text("switch_off", "real")


@pytest.mark.parametrize("tool", [FACILITY_SERVER_TOOL, FACILITY_EXTRA])
def test_a_facility_tool_is_allowed_while_live_is_on_with_the_session_on_va(
    tmp_path, hook_runner, make_config, tool
):
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "va", on=["live"])

    assert call(tmp_path, hook_runner, config, tool) is None


def test_a_whole_server_facility_server_refuses_its_reads_while_live_is_off(
    tmp_path, hook_runner, make_config
):
    """Nothing in the render tells such a server's reads from its writes."""
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "va", on=["va"])

    reason = reason_of(call(tmp_path, hook_runner, config, FACILITY_SERVER_READ))

    assert reason == deny_text("switch_off", "real")


@pytest.mark.parametrize("tool", [FACILITY_SERVER_TOOL, FACILITY_EXTRA])
def test_a_facility_tool_is_always_refused_on_a_deployment_with_no_live_target(
    tmp_path, hook_runner, make_config, tool
):
    config = make_config({"control_system": SIMULATOR_ONLY})
    write_record(tmp_path, "va", on=["va", "live"])

    assert reason_of(call(tmp_path, hook_runner, config, tool)) == deny_text("target_unknown")


def test_a_framework_tool_named_in_write_tools_stays_a_framework_tool(
    tmp_path, hook_runner, make_config
):
    """An extra naming a framework tool was not appended, so it follows the session."""
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "va", on=["va"])

    assert channel_write(tmp_path, hook_runner, config) is None


# ---------------------------------------------------------------------------
# the classifier, in-process
# ---------------------------------------------------------------------------


@pytest.fixture
def classify(tmp_path, monkeypatch):
    """``hook_write_target`` under :data:`HOOK_CONFIG`, for one tool and input."""
    path = tmp_path / "hook_config_classify.json"
    path.write_text(json.dumps(HOOK_CONFIG))
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(path))

    def run(config, tool, tool_input=None, short=None):
        hook_input = {"tool_name": tool, "tool_input": tool_input or {}, "cwd": str(tmp_path)}
        short = short or tool.split("__", 2)[-1]
        return twin.hook_write_target(config, tool, short, hook_input)

    return run


def test_the_classifier_places_every_kind_of_call(tmp_path, classify):
    config = {"control_system": TWO_TARGETS}
    write_record(tmp_path, "va")

    assert classify(config, "mcp__controls__channel_read") is None
    assert classify(config, "mcp__python__execute", {"execution_mode": "readonly"}) is None
    assert classify(config, "mcp__controls__channel_write") == "va"
    assert classify(config, FACILITY_SERVER_TOOL) == "live"
    assert classify(config, FACILITY_EXTRA) == "live"


def test_the_classifier_places_a_lane_tool_on_its_lane(classify):
    """Lane-addressed tools write to the machine their lane serves."""
    two_lanes = {
        "control_system": TWO_TARGETS,
        "services": {"bluesky_live": {"target": "live"}, "bluesky_va": {"target": "va"}},
    }
    one_lane = {"control_system": TWO_TARGETS, "services": {"bluesky": {}}}

    assert classify(two_lanes, "mcp__bluesky__queue_start", {"lane": "bluesky_va"}) == "va"
    assert classify(two_lanes, "mcp__bluesky__queue_start", {"lane": "bluesky_live"}) == "live"
    assert classify(one_lane, "mcp__bluesky__queue_start", {}) == "live"


def test_an_unplaceable_write_is_unresolved_not_a_read(tmp_path, classify):
    two_lanes = {
        "control_system": TWO_TARGETS,
        "services": {"bluesky_live": {"target": "live"}, "bluesky_va": {"target": "va"}},
    }
    record_path(tmp_path).write_text('{"schema": ', encoding="utf-8")

    assert classify(two_lanes, "mcp__bluesky__queue_start", {}) is twin.UNRESOLVED
    assert classify(two_lanes, "mcp__bluesky__queue_start", {"lane": "nope"}) is twin.UNRESOLVED
    assert classify(two_lanes, "mcp__controls__channel_write") is twin.UNRESOLVED


# ---------------------------------------------------------------------------
# the owner: dispatch runs
# ---------------------------------------------------------------------------


def test_a_dispatch_run_with_no_owner_is_unscheduled(
    tmp_path, hook_runner, make_config, monkeypatch
):
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "live", on=["live"])
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")

    assert reason_of(channel_write(tmp_path, hook_runner, config)) == deny_text(
        "owner_unscheduled", "real"
    )


def test_a_dispatch_run_for_this_login_follows_its_switch(
    tmp_path, hook_runner, make_config, monkeypatch
):
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "live", on=["live"])
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    monkeypatch.setenv("OSPREY_CONTROL_OWNER", pin_identity(monkeypatch))

    assert channel_write(tmp_path, hook_runner, config) is None


# ---------------------------------------------------------------------------
# the simulator lane with the real machine off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["mcp__bluesky__queue_add", "mcp__bluesky__queue_start"])
def test_the_hook_never_denies_a_lane_tool_with_the_real_machine_off(
    tmp_path, hook_runner, make_config, tool
):
    """The lane carve-out: the tool decides on its lane, so the hook does not."""
    config = make_config(
        {
            "control_system": TWO_TARGETS,
            "services": {"bluesky_live": {"target": "live"}, "bluesky_va": {"target": "va"}},
        }
    )
    write_record(tmp_path, "live", on=["va"])

    result = hook_runner(
        "osprey_writes_check.py",
        tool,
        {"plan": "count", "lane": "bluesky_va"},
        config_path=config,
        cwd=tmp_path,
        hook_config={
            "server_prefixes": ["mcp__bluesky__"],
            "write_tools": ["mcp__bluesky__queue_add", "mcp__bluesky__queue_start"],
            "lane_addressed_tools": ["queue_add", "queue_start"],
        },
    )

    assert result is None


# ---------------------------------------------------------------------------
# posture unknown — exactly one cell
# ---------------------------------------------------------------------------

POSTURE_UNKNOWN_TEXT = (
    "\U0001f512 WRITE STATE UNKNOWN — no control-context record was found "
    "where this hook looks, so the write state set on the control-target "
    "chip in the header cannot be read.\n\n"
    "Writes stay refused until the controls MCP server is running; "
    "config.yml is not the gate here."
)


def test_posture_unknown_fires_unstamped_with_no_record(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The one cell where an empty answer proves nothing."""
    config = make_config({"control_system": TWO_TARGETS})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

    assert reason_of(channel_write(tmp_path, hook_runner, config)) == POSTURE_UNKNOWN_TEXT


def test_posture_unknown_outranks_a_tier_that_never_writes(
    tmp_path, hook_runner, make_config, monkeypatch
):
    config = make_config({"control_system": {"type": "epics", "writes": "never"}})
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

    assert reason_of(channel_write(tmp_path, hook_runner, config)) == POSTURE_UNKNOWN_TEXT


def test_a_record_alone_lifts_posture_unknown(tmp_path, hook_runner, make_config, monkeypatch):
    """Unstamped, a readable record in the derived directory is the evidence."""
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "live", on=["live"])
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

    assert channel_write(tmp_path, hook_runner, config) is None


def test_a_corrupt_record_under_a_derived_root_is_posture_unknown(
    tmp_path, hook_runner, make_config, monkeypatch
):
    config = make_config({"control_system": TWO_TARGETS})
    write_payload(record_path(tmp_path), {"schema": 99, "target": "live", "generation": 3})
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

    assert reason_of(channel_write(tmp_path, hook_runner, config)) == POSTURE_UNKNOWN_TEXT


def test_a_deployed_services_staged_config_still_finds_the_record(
    tmp_path, hook_runner, monkeypatch
):
    """The worker shape: a staged config, an identity of its own, and no stamp."""
    pin_identity(monkeypatch, "dispatch-worker-1")
    render = tmp_path / "build"
    render.mkdir()
    config = render / "config.yml"
    config.write_text(
        yaml.dump(
            {"project_root": "/home/runner/work/osprey/osprey/stack", "control_system": TWO_TARGETS}
        )
    )
    write_record(tmp_path, "live", on=["live"])
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

    assert channel_write(render, hook_runner, config) is None


def test_a_readonly_run_still_refuses_before_stage_two(
    tmp_path, hook_runner, make_config, monkeypatch
):
    config = make_config({"control_system": TWO_TARGETS})
    write_record(tmp_path, "live", on=["live"])
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    assert reason_of(channel_write(tmp_path, hook_runner, config)) == deny_text("readonly_run")


# ---------------------------------------------------------------------------
# the headless guard still blocks today's facility producers
# ---------------------------------------------------------------------------


def test_the_headless_guard_still_blocks_both_facility_producers(tmp_path):
    """A read-only headless run is denied both facility producers, unchanged."""
    (tmp_path / "config.yml").write_text(
        "claude_code:\n"
        "  servers:\n"
        "    magnets:\n"
        "      hooks: {pre_tool_use: [writes_check]}\n"
        "      permissions: {allow: [set_current]}\n"
    )
    hooks = tmp_path / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "hook_config.json").write_text(json.dumps(HOOK_CONFIG))

    blocked = read_only_disallowed_tools(tmp_path)

    assert "mcp__magnets" in blocked
    assert FACILITY_EXTRA in blocked
