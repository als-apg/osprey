"""Tests for the osprey_writes_check hook.

The hook refuses a write for either of two independent reasons, in order: the
session's own sandbox posture (``OSPREY_EXECUTION_MODE``), and the deployment's
write posture for the control-system target the session is pointed at. Read-only
tools and readonly python always pass through, and so do Bluesky queue tools at
the second stage, which a lane addresses rather than the session target.
"""

import json
import os

import pytest

from osprey_connectors.types import session_posture, target_writes_enabled


@pytest.mark.unit
def test_writes_disabled_blocks_channel_write(tmp_path, hook_runner, make_config):
    """Writes disabled blocks channel_write tool."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_writes_enabled_allows_channel_write(tmp_path, hook_runner, make_config):
    """Writes enabled allows channel_write through."""
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_writes_disabled_blocks_python_write_mode(tmp_path, hook_runner, make_config):
    """Writes disabled blocks python_execute in write mode."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "write"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_writes_disabled_allows_python_readonly(tmp_path, hook_runner, make_config):
    """Writes disabled still allows python_execute in readonly mode."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_writes_disabled_allows_python_missing_execution_mode(tmp_path, hook_runner, make_config):
    """Writes disabled allows python_execute when execution_mode is omitted.

    The server defaults execution_mode to "readonly", so when the agent omits
    the parameter (relying on the server default), the hook must treat it as
    readonly rather than blocking the call.
    """
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "print(42)"},  # no execution_mode — server defaults to "readonly"
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through (treated as readonly)


@pytest.mark.unit
def test_writes_disabled_allows_channel_read(tmp_path, hook_runner, make_config):
    """Writes disabled does not affect channel_read (read-only tool)."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_non_osprey_tools_pass_through(tmp_path, hook_runner, make_config):
    """Non-osprey tools are not affected by the writes check hook."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "some_other_tool",
        {"param": "value"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Not an osprey tool, passes through


@pytest.mark.unit
def test_deny_message_includes_reason(tmp_path, hook_runner, make_config):
    """Deny decision includes an informative message."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "permissionDecisionReason" in output
    assert "WRITES DISABLED" in output["permissionDecisionReason"]


# -- Config edge cases (gap fill) --


@pytest.mark.unit
def test_missing_config_file_denies(tmp_path, hook_runner):
    """A config.yml that does not exist arms nothing (fail-closed).

    ``load_osprey_config()`` hands back ``{}`` for a missing file, and a config
    that states no write posture at all is not a config that writes. This is the
    safe default for a safety-critical system — fail-closed, not fail-open.
    """
    # Point to a non-existent config path
    nonexistent_config = tmp_path / "nonexistent" / "config.yml"

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=nonexistent_config,
        cwd=tmp_path,
    )

    # Missing config → writes_enabled=False → deny (fail-closed)
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_missing_writes_enabled_key_denies(tmp_path, hook_runner, make_config):
    """Config exists but expresses no write posture anywhere → deny.

    Neither the deployment-wide key nor any per-connector block says anything
    about writes, and silence is not permission. Intentionally fail-closed, and
    the shape ``test_no_posture_stated_denies`` below pins on a session whose
    target actually resolves.
    """
    config = make_config({"control_system": {"type": "mock"}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    # Missing writes_enabled key → defaults to False → deny
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


# -- Dynamic write_tools via hook_config --


@pytest.mark.unit
def test_custom_write_tool_blocked_via_hook_config(tmp_path, hook_runner, make_config):
    """A custom tool listed in hook_config write_tools is blocked when writes disabled."""
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__custom__write_thing",
        {"param": "value"},
        config_path=config,
        cwd=tmp_path,
        hook_config={"write_tools": ["mcp__custom__write_thing"]},
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_custom_write_tool_allowed_when_writes_enabled(tmp_path, hook_runner, make_config):
    """A custom tool in write_tools is allowed through when writes are enabled."""
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__custom__write_thing",
        {"param": "value"},
        config_path=config,
        cwd=tmp_path,
        hook_config={"write_tools": ["mcp__custom__write_thing"]},
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_fallback_defaults_when_no_hook_config(tmp_path, hook_runner, make_config):
    """Without hook_config, falls back to the 2 framework default write tools."""
    config = make_config({"control_system": {"writes_enabled": False}})

    # The default tool mcp__controls__channel_write should still be blocked
    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
        # No hook_config — uses fallback defaults
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
@pytest.mark.parametrize(
    "stdin",
    ["", "{nope", "[]", "[1,2,3]"],
    ids=["empty", "invalid-json", "wrong-shape", "wrong-shape-truthy"],
)
def test_malformed_stdin_fails_open(tmp_path, hook_runner_raw, stdin):
    """Stdin the hook cannot use lets the tool through instead of blocking it.

    The four shapes are a closed pipe, a truncated write, and two payloads
    whose JSON is valid but is not the expected object — one falsy (``[]``)
    and one truthy (``[1,2,3]``). The truthy one is the sharper case: it
    survives an emptiness check, so only a shape check keeps it out. A
    PreToolUse hook fails open by exiting 0 and printing no decision —
    anything else would turn a bad payload into a denied tool call.
    """
    returncode, stdout, stderr = hook_runner_raw(
        "osprey_writes_check.py",
        tool_name=None,
        tool_input=None,
        cwd=tmp_path,
        stdin_override=stdin,
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "Traceback" not in stderr


# -- Session posture (OSPREY_EXECUTION_MODE) --
#
# A web-terminal session switched to the sandbox posture launches its agent with
# ``OSPREY_EXECUTION_MODE=readonly``; the hooks inherit it. The posture is a
# property of *this terminal session*, not of the deployment, so the hook answers
# it from the environment alone — ahead of config.yml, and in a vocabulary that
# never sends the operator to edit a config file that is not the gate.


@pytest.mark.unit
def test_posture_readonly_denies_channel_write_despite_writes_enabled(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The whole point: the posture outranks a deployment that permits writes.

    ``writes_enabled: true`` is exactly the configuration under which this hook
    would otherwise allow the call, and for a mixed read/write kernel it is the
    renderer's ``allow`` that puts the tool in front of the agent at all. The
    deny here is the only thing standing between a sandboxed session and a
    control-system write.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_posture_readonly_denies_python_readwrite_despite_writes_enabled(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A readwrite execution is a write, and the sandbox posture refuses it."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "readwrite"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_posture_message_names_the_posture_not_writes_enabled(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Two-vocabulary rule: nothing is wrong with the deployment config.

    Mirror of ``test_readonly_refusal_message_does_not_blame_deployment`` on the
    connector side. A posture refusal that mentions ``writes_enabled`` sends the
    operator off to flip a config key that will not lift the refusal — the
    session's own posture is the gate, and the control-target chip in the header
    is where it moves.

    Stage 1 names no target: the session-wide posture is answered from the
    environment ahead of any config I/O, so the scope half of the message is
    empty here and pinned with a target by
    ``test_writes_check_per_target.py``.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "WRITES OFF" in reason
    assert "writes_enabled" not in reason
    assert "WRITES DISABLED" not in reason
    assert "control-target chip in the header" in reason
    assert "terminal card" not in reason
    # The two sentences the operator needs, verbatim.
    assert "this session refuses control-system writes." in reason
    assert (
        "Turn writes back on from the control-target chip in the header; "
        "config.yml is not the gate here." in reason
    )


@pytest.mark.unit
def test_posture_readonly_still_allows_python_readonly(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Readonly execution is precisely what a sandboxed session is *for*.

    The posture check sits behind the execute-readonly early exit, so a readonly
    run keeps passing through — sandboxing a session must not cost the agent the
    ability to look at the machine.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_posture_does_not_affect_non_write_tools(tmp_path, hook_runner, make_config, monkeypatch):
    """Reads stay reads: the posture branch is behind the write-tool filter."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_no_posture_var_leaves_the_writes_enabled_allow_intact(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """With no posture set, the hook behaves exactly as it did before.

    The unset case is the overwhelmingly common one — every CLI session and
    every deployment that never touches the header chip — so it is pinned
    rather than left to the other tests to imply.
    """
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
@pytest.mark.parametrize("value", ["readwrite", "READONLY", "", "sandbox", "true"])
def test_posture_is_a_value_comparison_not_a_presence_check(
    tmp_path, hook_runner, make_config, monkeypatch, value
):
    """Only the exact ``readonly`` string sandboxes the session.

    Same semantics as the executor's posture clamp and ``is_readonly_run``: a
    presence check would sandbox a session on ``readwrite`` — the *writes*
    posture — and on every stale or mistyped value besides.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", value)
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through — not the sandbox posture


@pytest.mark.unit
def test_posture_deny_survives_an_unreadable_config(tmp_path, hook_runner_raw, monkeypatch):
    """A broken config.yml must not cost the posture deny.

    This hook fails *open*: an uncaught exception exits non-zero with no JSON on
    stdout and the tool proceeds. The posture branch therefore has to reach its
    deny without depending on anything that can raise — the config read, PyYAML,
    or the debug logger, which is pointed at a config that is not parseable and
    at a project directory with no ``.claude/hooks`` to append to.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")  # force the logging path to run
    broken = tmp_path / "config.yml"
    broken.write_text("control_system: [unclosed\n\t\tnot: yaml")

    returncode, stdout, stderr = hook_runner_raw(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=broken,
        cwd=tmp_path,
    )

    assert returncode == 0
    assert "Traceback" not in stderr
    decision = json.loads(stdout.strip().split("\n")[-1])
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "WRITES OFF" in decision["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.unit
def test_posture_deny_survives_an_absent_config(tmp_path, hook_runner, monkeypatch):
    """No config.yml at all is still a valid, posture-specific deny."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=tmp_path / "nonexistent" / "config.yml",
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "WRITES OFF" in result["hookSpecificOutput"]["permissionDecisionReason"]


# -- Deployment posture, per target (stage 2) --
#
# Write posture is a property of the machine a call would reach, not of the
# deployment as a whole, so stage 2 asks `osprey_target_state` for the posture
# of the target THIS session is pointed at. The rules it applies are the same
# ones `osprey_connectors.types.target_writes_enabled` applies on the framework
# side, which is what the parity table below states literally: for every config
# shape, the hook's decision is the resolver's answer for the same section and
# the same target.


def _state_dir(repo_root):
    """The state directory the reader derives from a repo root."""
    directory = repo_root / "var" / "agent_data" / "control_target"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_session_state(repo_root, target):
    """Write a state file this pytest process genuinely owns.

    ``owner_ppid`` is this process, which IS on the ancestor chain of the hook
    subprocess ``hook_runner`` spawns, and ``server_pid`` is this process, which
    is alive by definition — so the reader's real parentage and liveness rules
    select this record without any seam being replaced.
    """
    record = {
        "target": target,
        "generation": 3,
        "server_pid": os.getpid(),
        "owner_ppid": os.getpid(),
        "targets": {
            "live": {
                "label": "Storage ring",
                "endpoint": "pva://live-gw.example.org:5075",
                "real_machine": True,
                "probe_channel": "RING:BEAM:CURRENT",
            },
            "va": {
                "label": "Virtual accelerator",
                "endpoint": "pva://127.0.0.1:5074",
                "real_machine": False,
                "probe_channel": "VA:BEAM:CURRENT",
            },
        },
        "children": [],
    }
    path = _state_dir(repo_root) / f"target_state_{os.getpid()}.json"
    path.write_text(json.dumps(record), encoding="utf-8")


def _channel_write(tmp_path, hook_runner, config, **kwargs):
    """Run the hook against a `channel_write` call and hand back its decision."""
    return hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "RING:QF:SP", "value": 1.5}]},
        config_path=config,
        cwd=tmp_path,
        **kwargs,
    )


#: A deployment armed for its simulator and NOT for its ring — the shape the
#: per-target key exists for, reused by several tests below.
DISARMED_LIVE = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {"epics": {"writes_enabled": False}},
}

#: ``(section, session_target)`` for every config shape stage 2 must answer.
#: ``None`` as the target means no state file is written at all, so the session
#: target is unidentifiable and the posture both targets agree on is the answer.
POSTURE_SHAPES = [
    ({"type": "mock", "writes_enabled": True}, "live"),
    ({"type": "mock", "writes_enabled": False}, "live"),
    ({"type": "epics", "writes_enabled": True}, "live"),
    (
        {
            "type": "epics",
            "writes_enabled": False,
            "connector": {"virtual_accelerator": {"writes_enabled": True}},
        },
        "va",
    ),
    (DISARMED_LIVE, "live"),
    (DISARMED_LIVE, None),
    (
        {
            "type": "virtual_accelerator",
            "writes_enabled": True,
            "connector": {
                "epics": {"writes_enabled": False},
                "doocs": {"writes_enabled": False},
            },
        },
        "live",
    ),
    (
        {
            "type": "doocs",
            "writes_enabled": False,
            "connector": {"doocs": {"writes_enabled": True}},
        },
        "live",
    ),
    ({"type": "epics"}, "live"),
    (
        {
            "type": "virtual_accelerator",
            "writes_enabled": False,
            "connector": {"virtual_accelerator": {"writes_enabled": True}},
        },
        None,
    ),
    (
        {
            "type": "mock",
            "writes_enabled": True,
            "connector": {"epics": {"writes_enabled": False}},
        },
        None,
    ),
    (
        {
            "type": "epics",
            "writes_enabled": True,
            "connector": {
                "epics": {"writes_enabled": False},
                "virtual_accelerator": {"port": 5074},
            },
        },
        None,
    ),
    (
        {
            "type": "epics",
            "writes_enabled": False,
            "connector": {"epics": {"writes_enabled": True}, "virtual_accelerator": {}},
        },
        None,
    ),
    ({"type": "epics", "connector": {"epics": {"writes_enabled": True}}}, "live"),
]

POSTURE_SHAPE_IDS = [
    "mock-global-armed",
    "hello-world-global-unarmed",
    "single-target-epics-global-armed",
    "armed-va-only",
    "disarmed-live-on-live",
    "disarmed-live-no-state-file",
    "two-live-blocks-underivable",
    "doocs-baseline-armed-by-block",
    "no-posture-stated",
    "va-armed-only-no-state-file",
    "mock-carrying-one-live-block-no-state-file",
    "switch-capable-mixed-no-state-file",
    "empty-va-block-is-not-switch-capable",
    "posture-stated-only-in-a-connector-block",
]


@pytest.mark.unit
@pytest.mark.parametrize(("section", "target"), POSTURE_SHAPES, ids=POSTURE_SHAPE_IDS)
def test_hook_decision_matches_the_framework_resolver(
    tmp_path, hook_runner, make_config, section, target
):
    """The hook and `target_writes_enabled` answer one deployment identically.

    Two implementations of the tri-state rules ship in this repo — the framework
    resolver, and the stdlib copy the hooks import — and a deployment described
    two ways is a deployment whose safety claim is unverifiable. So the expected
    value here is computed from the resolver rather than written out per shape.

    A session with no resolvable target answers the posture every target it
    could REACH agrees on, which is the resolver ANDed over `session_posture` —
    both targets on a switch-capable deployment, and the one type
    `control_system.type` builds on every other. The three no-state-file shapes
    at the end of the list are the ones where the two sets differ: a simulator
    deployment with no live block, a mock carrying one live block, and a
    switch-capable render whose targets disagree.

    The last two shapes pin the branches the stdlib copy restates and nothing
    else in the list exercises: an EMPTY connector block, which leaves a
    deployment out of the two-target world even though both targets name a
    type, and a deployment that states its posture only inside a connector
    block, which is the section that has said something despite carrying no
    deployment-wide key at all. Both are armed here, so a copy that lost either
    branch answers unarmed and the parity assertion fails.
    """
    # Arrange
    config = make_config({"control_system": section})
    if target is not None:
        _write_session_state(tmp_path, target)

    if target is None:
        expected_armed = all(session_posture(section).values())
    else:
        expected_armed = target_writes_enabled(section, target)

    # Act
    result = _channel_write(tmp_path, hook_runner, config)

    # Assert
    allowed = result is None
    assert allowed is expected_armed


@pytest.mark.unit
def test_disarmed_live_denies_on_live_and_names_the_connector_block(
    tmp_path, hook_runner, make_config
):
    """The refusal names the key that would actually lift it.

    A deployment whose global flag is `true` and whose live block is `false` is
    exactly the case where naming `control_system.writes_enabled` would send the
    operator to flip a key that is already true and changes nothing.
    """
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})
    _write_session_state(tmp_path, "live")

    # Act
    result = _channel_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    reason = output["permissionDecisionReason"]
    assert "control_system.connector.epics.writes_enabled: true" in reason
    assert "the active target (live)" in reason
    assert "Set control_system.writes_enabled" not in reason


@pytest.mark.unit
def test_disarmed_live_still_allows_the_simulator(tmp_path, hook_runner, make_config):
    """The same deployment, one target over: the virtual accelerator is armed.

    The mirror of the test above, and the whole point of the per-target key — a
    facility whose baseline is a live machine can arm writes on its simulator
    without arming the ring.
    """
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})
    _write_session_state(tmp_path, "va")

    # Act
    result = _channel_write(tmp_path, hook_runner, config)

    # Assert
    assert result is None


@pytest.mark.unit
def test_no_posture_stated_denies(tmp_path, hook_runner, make_config):
    """A config that expresses no posture anywhere still refuses.

    The one shape where this hook and `osprey_approval` deliberately disagree:
    `writes_posture` answers `None` — silence, not a refusal — and approval
    falls through to its normal prompt, while this hook keeps the fail-closed
    reading it has always had for a config with nothing to say about writes.
    """
    # Arrange
    config = make_config({"control_system": {"type": "epics"}})
    _write_session_state(tmp_path, "live")

    # Act
    result = _channel_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


# -- Stage ordering and the two skips --


@pytest.mark.unit
def test_sandbox_posture_denies_a_queue_tool_on_an_armed_deployment(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Stage 1 covers every write call, arming tools included.

    Queue tools skip the per-target check because a lane addresses them, not the
    session target — but that skip sits BEHIND the posture branch. A sandboxed
    session that could still arm a Bluesky lane would be a sandbox in name only.
    """
    # Arrange
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"type": "epics", "writes_enabled": True}})
    _write_session_state(tmp_path, "live")

    # Act
    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__bluesky__queue_add",
        {"plan": "count", "lane": "bluesky_live"},
        config_path=config,
        cwd=tmp_path,
        hook_config={
            "write_tools": ["mcp__bluesky__queue_add"],
            "server_prefixes": ["mcp__bluesky__"],
            "lane_addressed_tools": ["queue_add"],
        },
    )

    # Assert
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "WRITES OFF" in output["permissionDecisionReason"]


@pytest.mark.unit
def test_queue_tools_skip_the_target_posture_check(tmp_path, hook_runner, make_config):
    """A lane-addressed tool passes stage 2 on a deployment that is not armed.

    `channel_write` on this same config and this same session is denied (see the
    parity table); the queue tool is not, because the lane it binds to — and the
    bridge that refuses for it — is not the session's target.
    """
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})
    _write_session_state(tmp_path, "live")

    # Act
    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__bluesky__queue_start",
        {"lane": "bluesky_live"},
        config_path=config,
        cwd=tmp_path,
        hook_config={
            "write_tools": ["mcp__bluesky__queue_start"],
            "server_prefixes": ["mcp__bluesky__"],
            "lane_addressed_tools": ["queue_start"],
        },
    )

    # Assert
    assert result is None


@pytest.mark.unit
def test_a_tool_not_listed_as_lane_addressed_is_gated_by_stage_two(
    tmp_path, hook_runner, make_config
):
    """An unlisted tool stays subject to the target posture.

    The carve-out is data from `hook_config.json`, so its absence is the case a
    render predating the key — or one whose file could not be read — lands in.
    That must fail TOWARDS gating: the same call that passes when the render
    declares it lane-addressed is denied when nothing does.
    """
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})
    _write_session_state(tmp_path, "live")

    # Act
    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__bluesky__queue_start",
        {"lane": "bluesky_live"},
        config_path=config,
        cwd=tmp_path,
        hook_config={
            "write_tools": ["mcp__bluesky__queue_start"],
            "server_prefixes": ["mcp__bluesky__"],
        },
    )

    # Assert
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_stage_two_fails_closed_when_the_config_is_not_a_mapping(tmp_path, hook_runner):
    """Anything stage 2 cannot get through resolves to NOT ARMED.

    A `config.yml` holding a YAML *list* parses fine and then has no
    `control_system` to ask for — the shape a hand-edit produces. Everywhere
    else this hook fails open, which for a hook that only enriches is right and
    for the one deciding whether a write reaches a machine is not; stage 2 is
    the deliberate exception, and this is the pin on it.
    """
    # Arrange
    config = tmp_path / "config.yml"
    config.write_text("- control_system\n- writes_enabled\n")
    _write_session_state(tmp_path, "live")

    # Act
    result = _channel_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "control_system.writes_enabled: true" in output["permissionDecisionReason"]


@pytest.mark.unit
def test_a_render_without_the_state_reader_is_not_armed(tmp_path, hook_module, monkeypatch):
    """No `osprey_target_state` sibling means NOT ARMED, on any config.

    The invariant this holds up is shared with `osprey_approval`: that hook is
    allowed to *defer* — emit no prompt at all — only where a deny from this one
    is guaranteed, so the two must agree that a render missing the reader is
    unarmed. `osprey_target_state` is where the posture rules live as well as the
    target lookup, so a render without it cannot answer stage 2 at all, and the
    config below is armed precisely so the answer cannot come from the config.
    """
    # Arrange
    hook = hook_module("osprey_writes_check")
    config = tmp_path / "config.yml"
    config.write_text("control_system:\n  type: epics\n  writes_enabled: true\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setattr(hook, "_target_state", None)

    # Act
    armed, refusal_keys, target, refusal = hook._deployment_posture({})

    # Assert
    assert armed is False
    assert refusal_keys == ["control_system.writes_enabled"]
    assert target is None
    # A missing reader is the deployment's problem, not the session's: the
    # refusal has to keep naming a config key rather than the header chip.
    assert refusal == hook._REFUSAL_DEPLOYMENT


@pytest.mark.unit
def test_an_unidentifiable_session_names_the_unarmed_block_and_not_the_global_key(
    tmp_path, hook_runner, make_config
):
    """No state file, and the refusal still names a key that would lift it.

    `DISARMED_LIVE` has `control_system.writes_enabled: true` already, so naming
    the deployment-wide key here would send the operator to flip a key that is
    set to the value being asked for and whose flip changes nothing. The
    deployment renders no switch — there is no `virtual_accelerator` block — so
    `live` is the one target a session here can hold, and its block is the one
    thing that would arm it.
    """
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})

    # Act
    result = _channel_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "control_system.connector.epics.writes_enabled: true" in reason
    assert "control_system.writes_enabled: true" not in reason
    assert "could not be identified" in reason


@pytest.mark.unit
def test_an_unidentifiable_session_names_only_the_targets_that_are_unarmed(
    tmp_path, hook_runner, make_config
):
    """Both targets reachable, one already armed: only the other one is named.

    A switch-capable deployment whose simulator is armed and whose machine is
    not refuses an unidentifiable session, because the answer is the posture
    every reachable target agrees on. The simulator's key is already `true`
    there, so naming it would be the same wrong instruction as naming an
    already-true deployment-wide key.
    """
    # Arrange
    config = make_config(
        {
            "control_system": {
                "type": "epics",
                "writes_enabled": False,
                "connector": {
                    "epics": {"address_list": "10.0.0.1"},
                    "virtual_accelerator": {"writes_enabled": True},
                },
            }
        }
    )

    # Act
    result = _channel_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "control_system.connector.epics.writes_enabled: true" in reason
    assert "virtual_accelerator" not in reason


@pytest.mark.unit
def test_a_cloned_python_server_keeps_its_readonly_executions(tmp_path, hook_runner, make_config):
    """A readonly execution on an `extends` clone is not a write, and passes.

    The carve-out is per TOOL, and a clone renames only the server prefix. Read
    off the full name it would apply to `mcp__python__execute` alone, and every
    readonly analysis on a cloned python server would be refused on a deployment
    that arms nothing.
    """
    # Arrange
    config = make_config({"control_system": {"type": "epics", "writes_enabled": False}})

    # Act
    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__pyva__execute",
        {"code": "print(caget('RING:BEAM:CURRENT'))", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config={
            "write_tools": ["mcp__pyva__execute"],
            "server_prefixes": ["mcp__pyva__"],
        },
    )

    # Assert
    assert result is None
