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

Two further groups of pin live here for the same reason: the approval hook that
renders the switch prompt, and the ``osprey_target_state`` reader every deployed
hook imports, are both standalone and cannot import the framework — so they
carry literal copies of the lane keys, the target names and the connector types,
and the reader carries a copy of the target-to-type derivation itself. Those
copies are checked against their originals below, constants and derivations
alike. Drift there is silent, and it degrades a safety prompt rather than
breaking a build.
"""

from __future__ import annotations

import copy
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
    and the bridge both make. That restatement is only correct while these
    literals are the framework's own: rename a connector type and the hook would
    call a simulator or a stand-in deployment 'live', which is the one direction
    a wrong answer must never go.
    """
    from osprey.mcp_server.control_system.target_state import (
        TARGET_LIVE,
        TARGET_STANDIN,
        TARGET_VA,
    )
    from osprey_connectors import types as connector_types
    from tests.hooks.conftest import import_hook

    hook = import_hook("osprey_approval")

    assert hook._TARGET_VA == TARGET_VA
    assert hook._TARGET_LIVE == TARGET_LIVE
    assert hook._TARGET_STANDIN == TARGET_STANDIN
    assert hook._VIRTUAL_ACCELERATOR_TYPE == connector_types.VIRTUAL_ACCELERATOR
    assert hook._LIVE_STANDIN_TYPE == connector_types.LIVE_STANDIN
    assert hook._BASELINE_TARGETS == {
        connector_types.VIRTUAL_ACCELERATOR: TARGET_VA,
        connector_types.LIVE_STANDIN: TARGET_STANDIN,
    }


# ---------------------------------------------------------------------------
# (e) the target-state reader's standalone copies of the target vocabulary
# ---------------------------------------------------------------------------


def _reader():
    """The ``osprey_target_state`` module the approval hook itself bound.

    Taken off the hook rather than imported directly: the reader is a library
    beside the hooks rather than a hook, so it has no seam of its own, and the
    hook's own binding is the copy every deployed hook actually reads through.
    """
    from tests.hooks.conftest import import_hook

    reader = import_hook("osprey_approval")._target_state
    assert reader is not None, "the approval hook did not bind osprey_target_state"
    return reader


@pytest.mark.unit
def test_the_readers_target_literals_match_the_frameworks() -> None:
    """Every hook that names a target names it through these literals.

    ``osprey_target_state`` is the stdlib reader the deployed hooks and the
    status line import; it runs outside the osprey venv and so spells the target
    vocabulary out rather than importing ``osprey_connectors.types``. A target
    added upstream and not restated here is a machine the hooks cannot name: the
    stand-in would fall through ``target_type`` to ``None`` and be described
    from the deployment-wide key, which is a posture for a machine nobody
    identified.
    """
    from osprey_connectors import types as connector_types

    reader = _reader()

    assert reader.TARGET_LIVE == connector_types.TARGET_LIVE
    assert reader.TARGET_VA == connector_types.TARGET_VA
    assert reader.TARGET_STANDIN == connector_types.TARGET_STANDIN
    assert reader.CONTROL_TARGETS == connector_types.CONTROL_TARGETS


@pytest.mark.unit
def test_the_readers_connector_type_literals_match_the_frameworks() -> None:
    """The type names the target-to-type step and the never-live sets are built on.

    The two tuples are the ones ``_live_type`` excludes on both sides of its
    derivation. Renaming ``live_standin`` upstream without renaming it here
    would make the stand-in's block look like an ordinary candidate for
    ``live`` — the one direction a wrong answer must never go.
    """
    from osprey_connectors import types as connector_types

    reader = _reader()

    assert reader.MOCK_TYPE == connector_types.MOCK
    assert reader.VIRTUAL_ACCELERATOR_TYPE == connector_types.VIRTUAL_ACCELERATOR
    assert reader.LIVE_STANDIN_TYPE == connector_types.LIVE_STANDIN
    assert reader.SIMULATED_TYPES == connector_types._SIMULATED_TYPES
    assert reader.STANDIN_TYPES == connector_types.STANDIN_TYPES


#: The shape the stand-in exists for: the facility's own ``epics`` block and the
#: soft IOC standing beside it. ``live`` stays derivable through all three
#: baselines precisely because the stand-in is skipped as a candidate.
_STANDIN_BESIDE_LIVE_CONNECTOR = {
    "epics": {"address_list": "10.0.0.1"},
    "virtual_accelerator": {"port": 5064},
    "live_standin": {"port": 5074},
}

#: A deployment whose only block is the stand-in: ``live`` names nothing, and
#: the framework raises where the reader answers ``None``.
_STANDIN_ONLY_CONNECTOR = {"live_standin": {"port": 5074}}


def _framework_target_type(section: dict, target: str) -> str | None:
    """``resolve_target``, with its refusal spelled the way the reader spells it.

    The reader answers ``None`` wherever the framework raises — it must never
    raise into a hook — so agreement is checked against that translation rather
    than against the exception itself.
    """
    from osprey_connectors.types import resolve_target

    try:
        return resolve_target(section, target)
    except ValueError:
        return None


@pytest.mark.unit
@pytest.mark.parametrize(
    "connector",
    [
        pytest.param(_STANDIN_BESIDE_LIVE_CONNECTOR, id="standin-beside-the-live-block"),
        pytest.param(_STANDIN_ONLY_CONNECTOR, id="standin-is-the-only-block"),
    ],
)
@pytest.mark.parametrize("baseline_type", ["epics", "virtual_accelerator", "live_standin"])
def test_the_reader_derives_targets_exactly_as_the_framework_does(connector, baseline_type) -> None:
    """Literal equality is not enough: the DERIVATIONS have to agree too.

    Pinning the constants catches a rename; it does not catch the reader
    treating a stand-in baseline as a live one, or counting the stand-in's block
    among the candidates ``live`` falls back to. Both failures are silent — the
    hook goes on rendering a target line, naming the wrong machine — so each
    baseline type is driven through both copies and the answers compared.
    """
    from osprey_connectors import types as connector_types

    reader = _reader()
    section = {"type": baseline_type, "connector": connector}

    assert reader._baseline_target(section) == connector_types.baseline_target(section)
    assert reader._live_type(section) == _framework_target_type(section, "live")
    for target in connector_types.CONTROL_TARGETS:
        assert reader.target_type(section, target) == _framework_target_type(section, target)


@pytest.mark.unit
def test_the_reader_reaches_the_standin_only_where_a_deployment_has_one() -> None:
    """``session_types`` restates ``configured_targets``, not the vocabulary.

    The reachable set is what ``most_restrictive_posture`` ANDs over, so a
    ``standin`` slot on a deployment carrying no ``live_standin`` block would
    fold in a posture for a soft IOC nobody stood up.
    """
    from osprey_connectors import types as connector_types

    reader = _reader()
    without = {
        "type": "epics",
        "connector": {"epics": {"address_list": "10.0.0.1"}, "virtual_accelerator": {"port": 5064}},
    }
    with_standin = {"type": "epics", "connector": _STANDIN_BESIDE_LIVE_CONNECTOR}
    # No simulator block, so this one is not in the switching world at all.
    # Pinned because whether a stand-in alone puts a deployment there is the
    # framework's ruling, and the reader has to make the same one.
    standin_but_no_va = {
        "type": "epics",
        "connector": {"epics": {"address_list": "10.0.0.1"}, "live_standin": {"port": 5074}},
    }

    for section in (without, with_standin, standin_but_no_va):
        assert sorted(reader.session_types(section)) == sorted(
            connector_types.session_posture(section)
        )
    assert "standin" not in reader.session_types(without)
    assert reader.session_types(with_standin)["standin"] == connector_types.LIVE_STANDIN


# ---------------------------------------------------------------------------
# (f) SC-5: widening the vocabulary must not touch a two-target render
# ---------------------------------------------------------------------------
#
# ``standin`` joined ``CONTROL_TARGETS`` as a third target, and every enumerator
# that used to loop that constant now loops ``configured_targets``. So a
# deployment that configures no ``live_standin`` block must enumerate, and
# render, exactly as it did before the third name existed. The pins below hold
# the settings.json half of that: the permissions Claude Code is launched with
# are rendered once, from ``session_posture``, and a phantom ``standin`` slot
# there is not a cosmetic extra key — it carries a write posture. The
# deployment-wide ``writes_enabled`` reaches an unconfigured target through
# ``target_writes_enabled``'s fallback, so a phantom slot on the config used
# below would report ``standin: True`` beside two disarmed real targets, turn an
# all-off render into a *mixed* one, and pull every writes-check-gated tool out
# of ``permissions.deny`` — the kill switch, lifted by a machine nobody stood
# up.
#
# That is exactly the config these renders use: the deployment-wide key is
# ``true`` and both configured types disarm themselves. It is the shape in which
# a phantom target changes the bytes, which is what makes the byte-identity pin
# sensitive rather than decorative — and the third render proves that sensitivity
# rather than asserting it.


def _two_target_connector(default_connector: dict) -> dict:
    """VA baseline beside the facility's own ``epics`` block, no stand-in.

    Both types disarm themselves so the render is all-off while the
    deployment-wide key stays ``true`` — see the section note above.
    """
    connector = {
        "virtual_accelerator": copy.deepcopy(default_connector["virtual_accelerator"]),
        "epics": copy.deepcopy(default_connector["epics"]),
    }
    connector["virtual_accelerator"]["writes_enabled"] = False
    connector["epics"]["writes_enabled"] = False
    return connector


def _with_standin(connector: dict) -> dict:
    """The same deployment plus a stand-in block, shaped like the VA's.

    Copied from the ``virtual_accelerator`` block rather than written out: the
    stand-in is a VA standing in for the live machine, so its block is that
    block, and copying it keeps this fixture true as the block gains keys. It
    states no posture of its own, so it inherits the deployment-wide ``true`` —
    which is what makes the third render differ from the two-target one.
    """
    connector = copy.deepcopy(connector)
    connector["live_standin"] = copy.deepcopy(connector["virtual_accelerator"])
    del connector["live_standin"]["writes_enabled"]
    return connector


def _non_path_strings(node) -> list[str]:
    """Every string in a rendered settings.json that is not a filesystem path.

    A raw substring search over the rendered file cannot answer "does this
    render name the stand-in", because every hook command carries the absolute
    path of the checkout that produced it and a checkout may be named anything
    at all — the worktree this feature was built in is literally called
    ``standin-third-target``. Strings containing a path separator are dropped
    (hook commands, the status line, and any path-shaped permission entry) and
    everything else — permission matchers, hook matchers, env keys and values,
    and the object keys themselves — is searched.
    """
    if isinstance(node, dict):
        found: list[str] = []
        for key, value in node.items():
            found.extend(_non_path_strings(key))
            found.extend(_non_path_strings(value))
        return found
    if isinstance(node, list):
        return [s for item in node for s in _non_path_strings(item)]
    if isinstance(node, str):
        return [] if "/" in node else [node]
    return []


@pytest.fixture(scope="module")
def standin_settings_renders(tmp_path_factory) -> dict:
    """Three settings.json renders of one project, differing only as noted.

    Rendered through ``regenerate_claude_code`` into the *same* project
    directory, so the byte comparison is a comparison of the render and not of
    two directories: every absolute path the template writes is identical across
    the three.

    * ``two`` — the two-target deployment, current vocabulary.
    * ``two_pre_standin`` — the same config with ``CONTROL_TARGETS``
      monkeypatched back to ``[live, va]``, the vocabulary as it stood before
      the stand-in existed. This is the "before" side of SC-5.
    * ``three`` — the same config plus a ``live_standin`` block: the render a
      phantom ``standin`` slot on the two-target config would have produced.
    """
    from osprey.cli.templates.manager import TemplateManager
    from osprey_connectors import types as connector_types

    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="standin-settings-render",
        output_dir=tmp_path_factory.mktemp("standin-settings-render"),
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    base = yaml.safe_load((project_dir / "config.yml").read_text())
    default_connector = base["control_system"]["connector"]

    def render(connector: dict) -> tuple[dict, str]:
        config = copy.deepcopy(base)
        section = config["control_system"]
        section["type"] = "virtual_accelerator"
        section["writes_enabled"] = True
        section["connector"] = connector
        (project_dir / "config.yml").write_text(yaml.dump(config))
        manager.regenerate_claude_code(project_dir)
        return section, (project_dir / ".claude" / "settings.json").read_text()

    two_connector = _two_target_connector(default_connector)
    three_connector = _with_standin(two_connector)

    two_section, two = render(two_connector)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            connector_types,
            "CONTROL_TARGETS",
            [connector_types.TARGET_LIVE, connector_types.TARGET_VA],
        )
        # The patch has to bite, or the "before" render is the "after" render
        # under another name and the pin below passes having proved nothing.
        # ``configured_targets`` reads the module global at call time, so it
        # does — asserted here rather than assumed.
        assert connector_types.configured_targets(
            {"type": "virtual_accelerator", "connector": three_connector}
        ) == [connector_types.TARGET_LIVE, connector_types.TARGET_VA], (
            "configured_targets did not follow the patched CONTROL_TARGETS — it "
            "captured the vocabulary at import time, and this pin cannot compare "
            "the two vocabularies"
        )
        _, two_pre_standin = render(two_connector)
    finally:
        monkeypatch.undo()

    three_section, three = render(three_connector)

    return {
        "two_section": two_section,
        "three_section": three_section,
        "two": two,
        "two_pre_standin": two_pre_standin,
        "three": three,
    }


@pytest.mark.unit
def test_a_two_target_deployment_enumerates_exactly_its_two_targets() -> None:
    """No ``standin`` slot where no ``live_standin`` block was configured.

    The posture map is the render's input, so this is what the two render pins
    below are downstream of — stated on its own because a phantom slot is worth
    catching without paying for a project render.
    """
    from osprey_connectors.types import (
        TARGET_LIVE,
        TARGET_STANDIN,
        TARGET_VA,
        configured_targets,
        session_posture,
    )

    connector = {
        "virtual_accelerator": {"timeout": 5.0},
        "epics": {"address_list": "10.0.0.1"},
    }
    two = {"type": "virtual_accelerator", "writes_enabled": True, "connector": connector}
    three = {**two, "connector": {**connector, "live_standin": {"timeout": 5.0}}}

    assert configured_targets(two) == [TARGET_LIVE, TARGET_VA]
    assert set(session_posture(two)) == {TARGET_LIVE, TARGET_VA}
    # The mirror: the same deployment WITH a stand-in block does get the slot,
    # so the absence above is the config's doing and not a target that fell out
    # of the enumeration everywhere.
    assert configured_targets(three) == [TARGET_LIVE, TARGET_VA, TARGET_STANDIN]
    assert set(session_posture(three)) == {TARGET_LIVE, TARGET_VA, TARGET_STANDIN}


@pytest.mark.slow
def test_a_two_target_settings_render_names_no_standin(standin_settings_renders) -> None:
    """SC-5, the readable half: the rendered permissions never say ``standin``."""
    two = json.loads(standin_settings_renders["two"])

    named = [s for s in _non_path_strings(two) if "standin" in s]

    assert not named, f"a two-target render named the stand-in: {named}"


@pytest.mark.slow
def test_a_two_target_settings_render_is_identical_under_the_old_vocabulary(
    standin_settings_renders,
) -> None:
    """SC-5, the pin: adding ``standin`` to the vocabulary changed no bytes here.

    The "before" side is the same config, in the same directory, rendered with
    ``CONTROL_TARGETS`` monkeypatched back to ``[live, va]``. Byte equality is
    the whole claim — a deployment that gained no target gained no render
    change, down to key order and whitespace.
    """
    renders = standin_settings_renders

    assert renders["two"] == renders["two_pre_standin"], (
        "the settings.json render of a two-target deployment changed when "
        "'standin' joined CONTROL_TARGETS"
    )


@pytest.mark.slow
def test_a_configured_standin_does_reach_the_settings_render(
    standin_settings_renders,
) -> None:
    """The other side of that pin: the comparison is sensitive, not vacuous.

    Adding the ``live_standin`` block — and nothing else — makes the render
    differ, because the stand-in inherits the deployment-wide
    ``writes_enabled: true`` while both configured types disarm themselves, so
    the deployment stops being all-off and becomes mixed. That difference is
    precisely what a phantom ``standin`` slot would have produced on the
    two-target config, so seeing it here is what makes the byte-identity above
    worth asserting.
    """
    from osprey_connectors.types import TARGET_LIVE, TARGET_STANDIN, TARGET_VA, session_posture

    renders = standin_settings_renders
    posture = session_posture(renders["three_section"])

    assert posture[TARGET_STANDIN] is True
    assert set(posture) == {TARGET_LIVE, TARGET_VA, TARGET_STANDIN}
    assert renders["three"] != renders["two"], (
        "a configured stand-in changed nothing in the render — the byte-identity "
        "pin for the two-target case is then insensitive and proves nothing"
    )
