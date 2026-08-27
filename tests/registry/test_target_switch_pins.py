"""Gate wiring for ``control_target_set``: approval-gated, never kill-switched.

The target switch is the one control-system tool whose gate must survive the
writes kill switch. A deployment that may write to neither of its targets is
precisely the one that most needs to move a session between the simulator and
the machine it only reads — and the kill switch renders every
writes-check-gated tool into ``permissions.deny``, where a call is blocked
before any hook or refusal of the tool's own ever runs.

Write posture is per connector type, so a render reaches that all-off state
only when no target may write. The render below gets there the way most
deployments do: it pins ``control_system.writes_enabled: false`` and writes no
per-type block, so both targets inherit the false. A deployment armed on one
target and not the other renders no static deny at all, which is a weaker
version of the same requirement — the pins here hold the strict case.

So this file pins three things that are easy to break by adding one hook to one
list:

1. a writes-off render does NOT deny ``mcp__controls__control_target_set``
   (while it still denies ``channel_write``, so the pin proves the kill switch
   ran at all);
2. the tool IS in the read-only side-effecting set, so a headless read-only
   query cannot call it — the switch mutates session state and the tool refuses
   such a run itself, but ``disallowed_tools`` is the layer that does not
   depend on the tool being reached;
3. the registry entry carries the approval hook and not the writes-check hook.

A fourth pin lives here for the same reason: the approval hook that renders the
switch prompt is deployed standalone and cannot import the framework, so it
carries literal copies of the lane keys and target names. Those copies are
checked against their originals below — drift there is silent, and it degrades a
safety prompt rather than breaking a build.
"""

from __future__ import annotations

import json

import pytest
import yaml

from osprey.registry.mcp import resolve_servers

TOOL = "control_target_set"
MATCHER = f"mcp__controls__{TOOL}"

ROSTER_TOOL = "control_target"
ROSTER_MATCHER = f"mcp__controls__{ROSTER_TOOL}"


def _controls() -> dict:
    """The resolved "controls" server dict, in its public/registered form."""
    servers = resolve_servers(
        {},
        {"project_root": "/tmp/test-project", "current_python_env": "/usr/bin/python3"},
    )
    matches = [s for s in servers if s["name"] == "controls"]
    assert len(matches) == 1, "expected exactly one resolved controls server"
    return matches[0]


def _hook_commands(server: dict, matcher: str) -> list[str]:
    rules = [r for r in server["hooks_pre"] if r["matcher"] == matcher]
    assert rules, f"no PreToolUse rule for {matcher!r}"
    return [h["command"] for rule in rules for h in rule["hooks"]]


# ---------------------------------------------------------------------------
# (c) registry wiring
# ---------------------------------------------------------------------------


def test_the_switch_tool_is_approval_gated() -> None:
    """It prompts the operator, and it is in ``ask`` rather than ``allow``."""
    controls = _controls()

    assert TOOL in controls["permissions_ask"]
    assert TOOL not in controls["permissions_allow"]
    assert any("osprey_approval.py" in c for c in _hook_commands(controls, MATCHER))


def test_the_switch_tool_is_never_writes_check_gated() -> None:
    """The kill switch must not be able to block a target switch.

    ``osprey_writes_check.py`` on this matcher would put the tool in
    ``permissions.deny`` on every writes-off deployment (see
    :func:`test_a_writes_off_render_does_not_deny_the_switch_tool`), which is
    the silent-denial failure this pin exists to catch.
    """
    commands = _hook_commands(_controls(), MATCHER)

    assert not any("osprey_writes_check.py" in c for c in commands), (
        f"{MATCHER} must never be writes-check gated — a writes-off deployment "
        f"would deny the switch outright"
    )


def test_the_roster_tool_is_a_silent_read() -> None:
    """``control_target`` reports; it never acts, so it never prompts.

    An approval prompt in front of a read that opens no socket and spawns
    nothing would train operators to click through prompts that precede no
    motion — and this is the tool an agent is meant to call BEFORE proposing a
    switch, so a prompt here would tax exactly the careful path.
    """
    controls = _controls()

    assert ROSTER_TOOL in controls["permissions_allow"]
    assert ROSTER_TOOL not in controls["permissions_ask"]
    assert ROSTER_MATCHER not in {r["matcher"] for r in controls["hooks_pre"]}


def test_the_roster_tool_is_callable_in_a_read_only_run() -> None:
    """It must survive the read-only floor, or the safe path is the blocked one.

    Membership is exact, not by prefix: ``mcp__controls__control_target_set``
    IS in this set and shares the roster's name as a prefix.
    """
    from osprey.agent_runner.write_tools import _registry_side_effect_tools

    side_effecting = _registry_side_effect_tools()

    assert ROSTER_MATCHER not in side_effecting
    assert MATCHER in side_effecting, "the switch must still be side-effecting"


# ---------------------------------------------------------------------------
# (b) read-only side-effecting set
# ---------------------------------------------------------------------------


def test_the_switch_tool_is_side_effecting_for_read_only_runs() -> None:
    """A headless read-only run may not call it.

    Derived from the registry walk rather than a hand-maintained list: the tool
    is in ``permissions_ask``, and the classifier treats every ask entry as
    side-effecting. Pinned here because that derivation is the only thing
    keeping the two in step.
    """
    from osprey.agent_runner.write_tools import _registry_side_effect_tools

    assert MATCHER in _registry_side_effect_tools()


def test_the_switch_tool_is_not_in_the_writes_kill_switch_list() -> None:
    """It is side-effecting, but it is not a hardware write.

    ``write_tools`` is the kill switch's own list, rendered from the
    writes-check matchers; the switch tool being in it would be the same
    silent-denial bug from the other direction.
    """
    from osprey.agent_runner.write_tools import _FALLBACK_WRITE_TOOLS

    assert MATCHER not in _FALLBACK_WRITE_TOOLS


# ---------------------------------------------------------------------------
# (a) the rendered writes-off deny list
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_writes_off_render_does_not_deny_the_switch_tool(tmp_path) -> None:
    """End-to-end: a real project rendered with writes off keeps the switch.

    Rendered rather than reasoned about, because the deny list is produced by
    ``build_claude_code_context``'s kill-switch pass walking the registry — the
    exact walk a new hook entry would sweep the tool into.
    """
    from osprey.cli.templates.manager import TemplateManager
    from osprey.utils.config_writer import config_update_fields

    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="target-switch-deny",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config_update_fields(project_dir / "config.yml", {"control_system.writes_enabled": False})
    manager.regenerate_claude_code(project_dir)

    settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
    permissions = settings["permissions"]

    # The kill switch ran: the pure-write tool is denied.
    assert "mcp__controls__channel_write" in permissions["deny"]
    # ... and the switch survived it, prompt intact.
    assert MATCHER not in permissions["deny"]
    assert MATCHER in permissions["ask"]

    # The roster is a silent read on the same render: allowed, never denied,
    # never prompted. It is the tool an agent calls before proposing a switch,
    # so a writes-off deployment losing it would cost the careful path.
    assert ROSTER_MATCHER in permissions["allow"]
    assert ROSTER_MATCHER not in permissions["deny"]
    assert ROSTER_MATCHER not in permissions["ask"]

    # The deployed hook config agrees: the switch is not a kill-switch tool.
    hook_config = json.loads((project_dir / ".claude" / "hooks" / "hook_config.json").read_text())
    assert MATCHER not in hook_config.get("write_tools", [])
    assert ROSTER_MATCHER not in hook_config.get("write_tools", [])

    # And the rendered project really did have writes off on EVERY target, so
    # the assertions above describe the all-off render and not a mixed one (a
    # mixed render denies nothing statically, which would pass these pins for
    # the wrong reason). Posture is per connector type, so that means the flat
    # key is false and no connector block overrides it for a type.
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    assert config["control_system"]["writes_enabled"] is False
    for block in (config["control_system"].get("connector") or {}).values():
        if isinstance(block, dict):
            assert block.get("writes_enabled") is not True


# ---------------------------------------------------------------------------
# (d) the approval hook's standalone copies of the framework's lane constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_approval_hooks_lane_literals_match_the_frameworks() -> None:
    """The hook names lanes with literals; this is what keeps them true.

    ``osprey_approval.py`` is deployed standalone, into projects that may run
    against a different osprey install than the one that rendered them, so it
    cannot import ``bluesky_bridge_connection`` — it spells the lane keys out.
    A lane key added or renamed upstream would otherwise leave the approval
    prompt describing a deployment shape that no longer exists: a second lane it
    cannot see reads to an approver as a single-lane deployment, which is the
    prompt that says nothing about which machine a plan would run on.

    The per-lane bridge-URL variable is pinned the same way: the hook derives
    it as ``<LANE>_BRIDGE_URL`` from the lane key, which has to stay the name
    ``lane_env_prefix`` builds, or a lane's queue listing would be fetched from
    the port the build wrote while the framework talks to the override.
    """
    from osprey.bluesky_bridge_connection import LANE_KEYS, LANE_ONE, lane_env_prefix
    from tests.hooks.conftest import import_hook

    hook = import_hook("osprey_approval")

    assert hook._LANE_KEYS == LANE_KEYS
    assert hook._LANE_ONE == LANE_ONE
    for lane in LANE_KEYS:
        assert f"{lane.upper()}_BRIDGE_URL" == f"{lane_env_prefix(lane)}_BRIDGE_URL"


@pytest.mark.unit
def test_the_approval_hooks_target_literals_match_the_frameworks() -> None:
    """The same drift guard for the target vocabulary the lane map is built on.

    The hook restates ``resolve_baseline_target`` in stdlib terms to give a lane
    with no declared target the deployment baseline — the substitution the host
    and the bridge both make. That restatement is only correct while these three
    literals are the framework's own: rename the connector type and the hook
    would call a simulator deployment 'live', which is the one direction a wrong
    answer must never go.
    """
    from osprey.mcp_server.control_system.target_state import TARGET_LIVE, TARGET_VA
    from osprey_connectors import types as connector_types
    from tests.hooks.conftest import import_hook

    hook = import_hook("osprey_approval")

    assert hook._TARGET_VA == TARGET_VA
    assert hook._TARGET_LIVE == TARGET_LIVE
    assert hook._VIRTUAL_ACCELERATOR_TYPE == connector_types.VIRTUAL_ACCELERATOR
