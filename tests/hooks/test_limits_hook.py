"""Tests for the osprey_limits hook.

This hook validates channel write values against configured safety limits
using the LimitsValidator. When limits are violated, the hook blocks the write.
When the validator is disabled or unavailable, writes pass through.

NOTE: LimitsValidator.from_config() reads the channel database from a JSON file
specified by control_system.limits_checking.database_path. Tests must create
a proper database file for validation to work.
"""

import importlib
import io
import json
import os
import sys
import types

import pytest
import yaml

from osprey_connectors.types import most_restrictive_limits_posture, target_limits_posture


def _make_limits_config(tmp_path, channels_db, enabled=True, allow_unlisted=False):
    """Create config.yml + channel_limits.json for a deployment-wide posture.

    The single-block shape, which is what every deployment had before per-type
    blocks existed: one mock connector, writes armed, and the two limits leaves
    stated deployment-wide. `_limits_config` writes it, so the sections these
    tests run against and the ones the per-type tests below run against are
    written by one thing.
    """
    return _limits_config(
        tmp_path,
        {
            "type": "mock",
            "writes_enabled": True,
            "limits_checking": {
                "enabled": enabled,
                "allow_unlisted_channels": allow_unlisted,
            },
        },
        channels_db,
    )


@pytest.mark.unit
def test_limits_violation_blocks_write(tmp_path, hook_runner):
    """Write exceeding channel limits is blocked."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 999.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_valid_value_passes(tmp_path, hook_runner):
    """Write within channel limits passes through."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        allow_unlisted=True,
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_limits_disabled_passes_through(tmp_path, hook_runner):
    """When limits_checking.enabled is false, all writes pass."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        enabled=False,
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 999999.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Allowed through (limits disabled)


@pytest.mark.unit
def test_non_write_tools_pass(tmp_path, hook_runner):
    """Non-write tools (channel_read, etc.) are not checked by limits hook."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_read",
        {"channels": ["TEST:PV"]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Read tools pass through


@pytest.mark.unit
def test_unlisted_channel_blocked_by_default(tmp_path, hook_runner):
    """Unlisted channels are blocked when allow_unlisted_channels is false (default)."""
    config = _make_limits_config(
        tmp_path,
        {"KNOWN:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
        allow_unlisted=False,
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "UNKNOWN:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_non_writable_channel_blocked(tmp_path, hook_runner):
    """Channel marked writable=false is blocked."""
    config = _make_limits_config(
        tmp_path,
        {"READONLY:PV": {"min_value": 0.0, "max_value": 100.0, "writable": False}},
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "READONLY:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_multiple_operations_any_violation_blocks(tmp_path, hook_runner):
    """If any operation in a batch violates limits, the entire batch is blocked."""
    config = _make_limits_config(
        tmp_path,
        {
            "PV:A": {"min_value": 0.0, "max_value": 100.0, "writable": True},
            "PV:B": {"min_value": 0.0, "max_value": 10.0, "writable": True},
        },
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {
            "operations": [
                {"channel": "PV:A", "value": 50.0},  # OK
                {"channel": "PV:B", "value": 999.0},  # Violation
            ]
        },
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


# -- Edge cases (gap fill) --


@pytest.mark.unit
def test_value_at_exact_maximum_passes(tmp_path, hook_runner):
    """Value exactly equal to max_value should pass validation."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 100.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Exact boundary should pass


@pytest.mark.unit
def test_value_at_exact_minimum_passes(tmp_path, hook_runner):
    """Value exactly equal to min_value should pass validation."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 0.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Exact boundary should pass


@pytest.mark.unit
def test_single_write_form_supported(tmp_path, hook_runner):
    """Single-write form (channel + value, not operations array) is validated."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
    )

    # Single-write form: channel + value at top level (no operations array)
    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"channel": "TEST:PV", "value": 999.0},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_single_write_form_valid_passes(tmp_path, hook_runner):
    """Single-write form with valid value passes through."""
    config = _make_limits_config(
        tmp_path,
        {"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}},
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"channel": "TEST:PV", "value": 50.0},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None  # Valid single-write passes


@pytest.mark.unit
def test_step_size_blocks_when_current_value_unreadable(tmp_path, hook_runner):
    """max_step validation blocks writes when current channel value can't be read.

    When max_step is configured, the validator tries to read the current channel
    value to verify the step size. In the hook context (no live control system),
    this read fails, and the validator blocks the write for safety — it can't
    confirm the step size is within bounds, so it fails closed.
    """
    config = _make_limits_config(
        tmp_path,
        {
            "TEST:PV": {
                "min_value": 0.0,
                "max_value": 100.0,
                "writable": True,
                "max_step": 5.0,
            }
        },
    )

    result = hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )

    # max_step requires reading current value → fails → deny (fail-closed)
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "step size" in result["hookSpecificOutput"]["permissionDecisionReason"].lower()


@pytest.mark.unit
@pytest.mark.parametrize(
    "stdin",
    ["", "{nope", "[]", "[1,2,3]"],
    ids=["empty", "invalid-json", "wrong-shape", "wrong-shape-truthy"],
)
def test_malformed_stdin_fails_open(tmp_path, hook_runner_raw, stdin):
    """Unusable stdin allows the write through rather than denying it.

    Limits validation is fail-closed once it has a channel and a value, but it
    never gets that far here: a closed pipe, a truncated write or a non-object
    payload — falsy (``[]``) or truthy (``[1,2,3]``) — leaves nothing to
    validate, so the hook exits 0 with no decision. The truthy payload is the
    one an emptiness check lets through, so it has to be rejected on shape.
    """
    returncode, stdout, stderr = hook_runner_raw(
        "osprey_limits.py",
        tool_name=None,
        tool_input=None,
        cwd=tmp_path,
        stdin_override=stdin,
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "Traceback" not in stderr


# -- Per-target limits posture --
#
# The limits posture is a property of the machine a write would reach, not of
# the deployment as a whole: `control_system.connector.<type>.limits_checking`
# overrides the deployment-wide `control_system.limits_checking` block whole, so
# a facility can refuse unlisted channels on its ring and allow them on its
# virtual accelerator. The hook therefore resolves the session's target before
# it builds a validator, and the tests below state that in the same shape
# `test_writes_check_hook.py` states the write posture: the expected decision is
# computed from the framework resolver rather than written out per config.


#: The one channel the limits database lists, and one it does not. The unlisted
#: write is what the `allow_unlisted_channels` leaf decides, which is the leaf a
#: per-type block exists to differ on.
KNOWN_DB = {"KNOWN:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}}
UNLISTED_CHANNEL = "UNKNOWN:PV"


def _limits_config(tmp_path, section, channels_db=None):
    """Write `config.yml` carrying *section*, plus the limits database.

    `database_path` stays deployment-wide — a deployment mounts one limits file
    — so it is injected into the section's own `limits_checking` block whatever
    per-type blocks the shape carries. It states no posture, so the section a
    test hands in and the section written here resolve identically.
    """
    db_path = tmp_path / "channel_limits.json"
    db_path.write_text(json.dumps(KNOWN_DB if channels_db is None else channels_db))

    rendered = json.loads(json.dumps(section))  # deep copy; every value is JSON
    rendered.setdefault("limits_checking", {})["database_path"] = str(db_path)

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump({"control_system": rendered}))
    return config_path


def _state_dir(repo_root):
    """The state directory the reader derives from a repo root."""
    directory = repo_root / "var" / "agent_data" / "control_target"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_session_state(repo_root, target):
    """Write a state file this pytest process genuinely owns.

    `owner_ppid` is this process, which IS on the ancestor chain of the hook
    subprocess `hook_runner` spawns, and `server_pid` is this process, which is
    alive by definition — so the reader's real parentage and liveness rules
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
            },
            "va": {
                "label": "Virtual accelerator",
                "endpoint": "pva://127.0.0.1:5074",
                "real_machine": False,
            },
            "standin": {
                "label": "Live stand-in",
                "endpoint": "pva://127.0.0.1:5076",
                "real_machine": False,
            },
        },
        "children": [],
    }
    path = _state_dir(repo_root) / f"target_state_{os.getpid()}.json"
    path.write_text(json.dumps(record), encoding="utf-8")


def _unlisted_write(tmp_path, hook_runner, config):
    """Run the hook against a write to a channel the database does not list."""
    return hook_runner(
        "osprey_limits.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": UNLISTED_CHANNEL, "value": 50.0}]},
        config_path=config,
        cwd=tmp_path,
    )


#: A deployment strict on its ring and permissive on its simulator — the shape
#: the per-type block exists for, reused by several tests below. Both connector
#: blocks are non-empty, so the deployment renders the target switch and the
#: reachable set a targetless call folds over is both targets.
VA_PERMISSIVE = {
    "type": "epics",
    "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
    "connector": {
        "epics": {"port_host": "live-gw.example.org"},
        "virtual_accelerator": {
            "port_host": "127.0.0.1",
            "limits_checking": {"enabled": True, "allow_unlisted_channels": True},
        },
    },
}

#: A mock deployment carrying one connector block it never builds. `live`
#: resolves to that block, so a call that NAMES `live` reads it — while a call
#: with no target folds over the single connector `control_system.type` builds,
#: which is the mock, and so answers the deployment-wide posture instead.
STRAY_LIVE_BLOCK = {
    "type": "mock",
    "limits_checking": {"enabled": True, "allow_unlisted_channels": False},
    "connector": {"epics": {"limits_checking": {"enabled": True, "allow_unlisted_channels": True}}},
}

#: A per-type block that states one leaf. It overrides whole, so it answers
#: nothing: the validator is the failsafe one and every write is refused.
HALF_WRITTEN_VA_BLOCK = {
    "type": "epics",
    "limits_checking": {"enabled": True, "allow_unlisted_channels": True},
    "connector": {
        "epics": {"port_host": "live-gw.example.org"},
        "virtual_accelerator": {"limits_checking": {"enabled": True}},
    },
}

#: A deployment permissive deployment-wide and strict on its ring — the mirror
#: of `VA_PERMISSIVE`, and the shape whose refusal must name a per-type key.
LIVE_STRICT_BLOCK = {
    "type": "epics",
    "limits_checking": {"enabled": True, "allow_unlisted_channels": True},
    "connector": {
        "epics": {"limits_checking": {"enabled": True, "allow_unlisted_channels": False}},
    },
}

#: A deployment whose only limits line cannot be read: `enabled` is a string,
#: which is what environment expansion leaves behind when nothing set the
#: variable. It states no posture, so the block is incomplete and every write is
#: refused — the alternative, reading it as an unset `enabled`, would mean "no
#: limits checking configured" and wave the write through.
UNREADABLE_DEPLOYMENT_WIDE = {
    "type": "epics",
    "limits_checking": {"enabled": "${OSPREY_LIMITS_ENABLED}", "allow_unlisted_channels": False},
    "connector": {"epics": {"port_host": "live-gw.example.org"}},
}

#: `(section, session_target)` for every config shape the hook must answer.
#: `None` as the target means no state file is written at all, so the session
#: target is unidentifiable and the posture every reachable target agrees on is
#: the answer.
POSTURE_SHAPES = [
    (VA_PERMISSIVE, "va"),
    (VA_PERMISSIVE, "live"),
    (VA_PERMISSIVE, "standin"),
    (VA_PERMISSIVE, None),
    (STRAY_LIVE_BLOCK, "live"),
    (STRAY_LIVE_BLOCK, None),
    (HALF_WRITTEN_VA_BLOCK, "va"),
    (HALF_WRITTEN_VA_BLOCK, "live"),
    (HALF_WRITTEN_VA_BLOCK, None),
    (UNREADABLE_DEPLOYMENT_WIDE, "live"),
    (UNREADABLE_DEPLOYMENT_WIDE, None),
    (LIVE_STRICT_BLOCK, "live"),
    (LIVE_STRICT_BLOCK, "va"),
    (
        {"type": "mock", "limits_checking": {"enabled": True, "allow_unlisted_channels": True}},
        "live",
    ),
    (
        {"type": "mock", "limits_checking": {"enabled": False, "allow_unlisted_channels": False}},
        "live",
    ),
    ({"type": "mock"}, None),
]

POSTURE_SHAPE_IDS = [
    "permissive-va-on-va",
    "permissive-va-on-live",
    "permissive-va-on-standin",
    "permissive-va-no-state-file",
    "stray-live-block-on-live",
    "stray-live-block-no-state-file",
    "half-written-block-on-its-own-target",
    "half-written-block-one-target-over",
    "half-written-block-no-state-file",
    "unreadable-deployment-wide-on-live",
    "unreadable-deployment-wide-no-state-file",
    "strict-live-block-on-live",
    "strict-live-block-on-va",
    "mock-unresolvable-live-permissive",
    "limits-disabled",
    "no-posture-stated",
]


def _expected_allowed(posture):
    """Whether an unlisted write passes under *posture*, per the validator's rules.

    Three outcomes, in the order `LimitsValidator._from_posture` takes them: an
    incomplete block builds the failsafe validator and refuses everything;
    limits checking that is not explicitly on builds no validator at all and the
    write goes through; and otherwise an unlisted channel needs an explicit
    `True`, since a tri-state `None` is nobody's permission.
    """
    if posture.incomplete:
        return False
    if posture.enabled is not True:
        return True
    return posture.allow_unlisted is True


@pytest.mark.unit
@pytest.mark.parametrize(("section", "target"), POSTURE_SHAPES, ids=POSTURE_SHAPE_IDS)
def test_hook_decision_matches_the_framework_resolver(tmp_path, hook_runner, section, target):
    """The hook and the package resolvers answer one deployment identically.

    The hook cannot resolve a posture itself — the rules live in
    `osprey_connectors.types` and the validator applies them — so what this pins
    is that the hook asks the right QUESTION for the identity it holds: the
    session's target when the state file names one, and the fold across every
    reachable target when it does not. Both expectations are computed from the
    resolvers rather than written out per shape, so a hook that asked the
    deployment-wide question instead fails on every row where a per-type block
    differs from it.
    """
    # Arrange
    config = _limits_config(tmp_path, section)
    if target is not None:
        _write_session_state(tmp_path, target)

    if target is None:
        posture = most_restrictive_limits_posture(section)
    else:
        posture = target_limits_posture(section, target)

    # Act
    result = _unlisted_write(tmp_path, hook_runner, config)

    # Assert
    allowed = result is None
    assert allowed is _expected_allowed(posture)


@pytest.mark.unit
def test_no_state_file_refuses_when_the_only_limits_line_cannot_be_read(tmp_path, hook_runner):
    """The baseline branch every unswitched session takes must not fail open.

    With no state file the hook folds across every reachable target. Both leaf
    folds send an unreadable posture's `None` to `False`, which on its own reads
    as "checking off, nothing permitted" -- and a validator is only built when
    `enabled` is `True`, so a fold that dropped the incompleteness would build
    no validator and allow the write. Stated absolutely here rather than against
    the resolver, because the parity test above derives both sides from the very
    fold this pins.
    """
    # Arrange
    config = _limits_config(tmp_path, UNREADABLE_DEPLOYMENT_WIDE)

    # Act
    result = _unlisted_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_live_refusal_names_the_deployment_wide_key(tmp_path, hook_runner):
    """On `live`, a deployment that relaxed only its simulator still refuses.

    And the refusal names the key that would actually lift it: the ring's own
    connector block wrote no `limits_checking`, so the deployment-wide line is
    the one an operator would have to edit.
    """
    # Arrange
    config = _limits_config(tmp_path, VA_PERMISSIVE)
    _write_session_state(tmp_path, "live")

    # Act
    result = _unlisted_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert (
        "control_system.limits_checking.allow_unlisted_channels"
        in (output["permissionDecisionReason"])
    )


@pytest.mark.unit
def test_va_passes_where_the_same_deployment_refuses_on_live(tmp_path, hook_runner):
    """The same config, one target over: the simulator takes the unlisted write.

    The mirror of the test above, and the whole point of the per-type block — a
    facility whose ring refuses unlisted channels can still let an agent write
    freely against its virtual accelerator.
    """
    # Arrange
    config = _limits_config(tmp_path, VA_PERMISSIVE)
    _write_session_state(tmp_path, "va")

    # Act
    result = _unlisted_write(tmp_path, hook_runner, config)

    # Assert
    assert result is None


@pytest.mark.unit
def test_per_type_refusal_names_the_connector_block(tmp_path, hook_runner):
    """A refusal read from a connector block names that block, not the global key.

    A deployment permissive deployment-wide and strict on its ring is exactly
    the case where naming `control_system.limits_checking.allow_unlisted_channels`
    would send the operator to flip a key that is already `true` and changes
    nothing.
    """
    # Arrange
    config = _limits_config(tmp_path, LIVE_STRICT_BLOCK)
    _write_session_state(tmp_path, "live")

    # Act
    result = _unlisted_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "control_system.connector.epics.limits_checking.allow_unlisted_channels" in reason


@pytest.mark.unit
def test_removed_state_directory_takes_the_most_restrictive_posture(tmp_path, hook_runner):
    """A session whose target cannot be read is refused what any target refuses.

    The state directory existed and is gone — a swept `var/`, a server that
    never started, a deployment whose agent-data root was moved — so the hook
    cannot say which machine this write would reach. The simulator would take
    it and the ring would not, and a guess between them could be a guess in
    favour of hardware, so the write is refused and the refusal names the
    deployment-wide key: no per-type line decides a union.
    """
    # Arrange
    config = _limits_config(tmp_path, VA_PERMISSIVE)
    _write_session_state(tmp_path, "va")
    state_dir = _state_dir(tmp_path)
    for path in state_dir.iterdir():
        path.unlink()
    state_dir.rmdir()

    # Act
    result = _unlisted_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert (
        "control_system.limits_checking.allow_unlisted_channels"
        in (output["permissionDecisionReason"])
    )


@pytest.mark.unit
def test_stray_connector_block_does_not_answer_for_a_targetless_call(tmp_path, hook_runner):
    """A mock deployment answers the deployment-wide posture, stray block and all.

    `live` on a mock deployment resolves to the single connector block that
    could be a real machine, so a call NAMING `live` reads it. A call with no
    target must not: the reachable set on a deployment that renders no switch is
    the one connector `control_system.type` builds, which here is the mock. A
    fold over the stray block would hand an agent the relaxation written for a
    machine no session on this deployment can select.
    """
    # Arrange — no state file at all
    config = _limits_config(tmp_path, STRAY_LIVE_BLOCK)

    # Act
    result = _unlisted_write(tmp_path, hook_runner, config)

    # Assert
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_osprey_unimportable_allows_the_write(tmp_path, hook_module, monkeypatch, capsys):
    """With `osprey` off the interpreter's path the hook exits 0 and decides nothing.

    The hook's one fail-open direction, and the reason it is right: a hook
    running under a bare system `python3` has no validator to consult and no
    posture to read, so refusing would block every write on a deployment whose
    limits are enforced by the MCP tool anyway. Driven in-process because the
    import can only be made to fail from inside the interpreter — a hook
    subprocess started by this suite has the venv, and therefore `osprey`, on
    its path by construction.
    """
    # Arrange
    hook = hook_module("osprey_limits")
    monkeypatch.setitem(sys.modules, "osprey.connectors.control_system.limits_validator", None)
    # A `None` entry in `sys.modules` is what makes the hook's own import raise.
    # Asserted here so that a test which stopped exercising the branch cannot
    # pass by exiting 0 for some other reason.
    with pytest.raises(ImportError):
        importlib.import_module("osprey.connectors.control_system.limits_validator")
    payload = {
        "tool_name": "mcp__controls__channel_write",
        "tool_input": {"operations": [{"channel": UNLISTED_CHANNEL, "value": 50.0}]},
        "cwd": str(tmp_path),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    # Act
    with pytest.raises(SystemExit) as exit_info:
        hook.main()

    # Assert
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == ""


# -- Branches only reachable from inside the interpreter --
#
# Three of the hook's decisions cannot be driven through `hook_runner`: a render
# without the sibling target-state reader, that reader raising, and a framework
# older than the render it is answering. All three are exercised in-process,
# against a stand-in validator rather than the real one, because config loading
# is a process-wide singleton keyed on `CONFIG_FILE` at first use — a test that
# pointed it at its own file would either read a config an earlier test cached
# or pin its own for every test after it. The stand-in answers exactly the
# VA-permissive / live-strict deployment the subprocess tests above describe,
# and records which entry point the hook asked, which is the branch under test.

#: The key a deployment-wide answer names, and the one a per-type answer does.
DEPLOYMENT_WIDE_KEY = "control_system.limits_checking.allow_unlisted_channels"
VA_BLOCK_KEY = (
    "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels"
)

#: `(allow_unlisted, answering key)` for the VA-permissive / live-strict
#: deployment, per entry point the hook can ask. The simulator takes unlisted
#: writes; the ring, the fold across both, and the deployment-wide question an
#: older framework asks all refuse.
STAND_IN_POSTURES = {
    ("target", "va"): (True, VA_BLOCK_KEY),
    ("target", "live"): (False, DEPLOYMENT_WIDE_KEY),
    ("most_restrictive", None): (False, DEPLOYMENT_WIDE_KEY),
    ("deployment_wide", None): (False, DEPLOYMENT_WIDE_KEY),
}


def _stand_in_validator_class(calls, per_target_api=True):
    """A `LimitsValidator` stand-in that records which entry point was asked.

    With *per_target_api* false it exposes only the old no-arg `from_config` and
    no `from_config_most_restrictive` — the shape a service still running an
    older `osprey` image presents to a freshly rendered hook.
    """

    class _StandIn:
        def __init__(self, allow_unlisted, key):
            self._allow_unlisted = allow_unlisted
            self._key = key

        def validate(self, channel, value):
            if channel in KNOWN_DB or self._allow_unlisted:
                return
            raise ValueError(
                f"Channel '{channel}' not in limits database "
                f"('{self._key}' does not allow unlisted channels)"
            )

    def _built(entry, target):
        calls.append((entry, target))
        return _StandIn(*STAND_IN_POSTURES[(entry, target)])

    if per_target_api:

        def from_config(cls, *, connector_type=None, target=None):
            return _built("target", target)

        def from_config_most_restrictive(cls):
            return _built("most_restrictive", None)

        _StandIn.from_config = classmethod(from_config)
        _StandIn.from_config_most_restrictive = classmethod(from_config_most_restrictive)
    else:

        def old_from_config(cls):
            return _built("deployment_wide", None)

        _StandIn.from_config = classmethod(old_from_config)

    return _StandIn


def _run_in_process(hook, monkeypatch, capsys, tmp_path, validator_cls):
    """Drive `main()` in this interpreter against *validator_cls*.

    Returns `(exit_code, decision, stderr)`, where the decision is the hook's
    parsed JSON output or `None` when it wrote none (allowed through).
    """
    monkeypatch.setitem(
        sys.modules,
        "osprey.connectors.control_system.limits_validator",
        types.SimpleNamespace(LimitsValidator=validator_cls),
    )
    payload = {
        "tool_name": "mcp__controls__channel_write",
        "tool_input": {"operations": [{"channel": UNLISTED_CHANNEL, "value": 50.0}]},
        "cwd": str(tmp_path),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    with pytest.raises(SystemExit) as exit_info:
        hook.main()

    captured = capsys.readouterr()
    stdout = captured.out.strip()
    return exit_info.value.code, (json.loads(stdout) if stdout else None), captured.err


@pytest.mark.unit
def test_a_render_without_the_state_reader_takes_the_most_restrictive_posture(
    tmp_path, hook_module, monkeypatch, capsys
):
    """No sibling reader is a session with no identity, not a session to wave through.

    A project rendered before `osprey_target_state` existed has no such sibling,
    so the guarded import leaves `_target_state` as `None`. The hook still has a
    validator and still has to decide, and the posture it decides under is the
    one every reachable target agrees on — refusing what the ring refuses rather
    than exiting 0 and letting an unlisted write reach it unchecked.
    """
    # Arrange
    hook = hook_module("osprey_limits")
    monkeypatch.setattr(hook, "_target_state", None)
    calls = []

    # Act
    code, decision, _stderr = _run_in_process(
        hook, monkeypatch, capsys, tmp_path, _stand_in_validator_class(calls)
    )

    # Assert
    assert code == 0
    assert calls == [("most_restrictive", None)]
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert DEPLOYMENT_WIDE_KEY in decision["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.unit
def test_a_raising_state_reader_takes_the_most_restrictive_posture(
    tmp_path, hook_module, monkeypatch, capsys
):
    """The reader is documented never to raise; if it did, the write is still checked.

    Same branch as the missing reader, reached the other way. What it pins is
    that no failure of target IDENTITY can turn into a write nobody validated —
    an exception escaping `_session_target` would exit the hook non-zero with no
    decision, which the agent runtime reads as no opinion.
    """
    # Arrange
    hook = hook_module("osprey_limits")

    def _boom(hook_input=None):
        raise RuntimeError("state directory vanished mid-read")

    monkeypatch.setattr(hook._target_state, "read_session_target", _boom)
    calls = []

    # Act
    code, decision, stderr = _run_in_process(
        hook, monkeypatch, capsys, tmp_path, _stand_in_validator_class(calls)
    )

    # Assert
    assert code == 0
    assert calls == [("most_restrictive", None)]
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "Traceback" not in stderr


@pytest.mark.unit
def test_an_older_framework_falls_back_to_the_deployment_wide_block(
    tmp_path, hook_module, monkeypatch, capsys
):
    """A hook newer than the `osprey` it meets asks the question that framework can answer.

    `osprey build` renders hooks into the repo on the host, while a service
    keeps the image it was built with until the next `osprey up --build`. On
    such a deployment `from_config_most_restrictive` does not exist and
    `from_config` takes no `target`, and calling either would exit the hook
    non-zero with no decision — an unlisted write reaching the machine with no
    limits applied at all. The fallback is the deployment-wide block, which is
    exactly what that framework enforced when it was the whole story.
    """
    # Arrange
    hook = hook_module("osprey_limits")
    calls = []
    older = _stand_in_validator_class(calls, per_target_api=False)

    # Act
    code, decision, stderr = _run_in_process(hook, monkeypatch, capsys, tmp_path, older)

    # Assert
    assert code == 0
    assert calls == [("deployment_wide", None)]
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert DEPLOYMENT_WIDE_KEY in decision["hookSpecificOutput"]["permissionDecisionReason"]
    assert "Traceback" not in stderr
