"""Tests for the osprey_writes_check hook.

The hook refuses a write for either of two independent reasons, in order: a
deployment-wide read-only run (``OSPREY_EXECUTION_MODE``), and the write verdict
for the machine the call writes to, asked of the stdlib twin. Read-only tools
and readonly python always pass through, and so do the lane-addressed Bluesky
queue tools at the second stage, which decide on their lane's machine in-tool.

Which machine a call writes to, framework against facility tools, is pinned in
``test_writes_check_per_target``; this module pins the mechanics around it.
"""

import json

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as _TWIN
from osprey_connectors import posture_store
from osprey_connectors.types import baseline_target
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    write_control_context,
)

#: A section whose login tier allows writes, on the real machine.
ALLOWED = {"type": "epics", "writes": "allowed"}

#: The same section with the tier refusing writes.
NEVER = {"type": "epics", "writes": "never"}

#: The texts no refusal here may carry: a config key, or a config surface.
KEY_WORDS = ("writes_enabled", "control_system.", "profile.yml", "osprey build")


@pytest.fixture(autouse=True)
def stamped_root(tmp_path, monkeypatch):
    """A clean verdict environment with a stamped agent-data root.

    ``osprey_target_state.posture_unknown`` refuses an UNSTAMPED process whose
    derived directory holds no control-context record. That refusal has its own
    suite (``test_writes_check_per_target``), and it would otherwise answer
    every case here before the verdict was asked at all. The identity is pinned
    so the hook child and this process name the same directory.
    """
    root = tmp_path / "var" / "agent_data"
    root.mkdir(parents=True, exist_ok=True)
    isolate_verdict_environment(monkeypatch, root)
    return root


def _switch(root, target="live", on=True):
    """Record the session on *target*, with its switch on or (no switch) off."""
    write_control_context(root, target=target, switch={target: "on"} if on else None)


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


def _reason(result):
    """The operator-facing refusal text, asserting there was a deny."""
    assert result is not None, "expected a deny, got a pass-through"
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    return output["permissionDecisionReason"]


def _assert_names_no_key(reason):
    for word in KEY_WORDS:
        assert word not in reason, word


# -- framework tools follow the switch -------------------------------------


def test_the_switch_off_denies_channel_write(tmp_path, hook_runner, make_config, stamped_root):
    config = make_config({"control_system": ALLOWED})
    _switch(stamped_root, "live", on=False)

    reason = _reason(_channel_write(tmp_path, hook_runner, config))

    assert reason == "\U0001f512 WRITES OFF — " + _TWIN.refusal_text("switch_off", "real")
    _assert_names_no_key(reason)


def test_the_switch_on_allows_channel_write(tmp_path, hook_runner, make_config, stamped_root):
    config = make_config({"control_system": ALLOWED})
    _switch(stamped_root, "live")

    assert _channel_write(tmp_path, hook_runner, config) is None


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_a_readwrite_execution_follows_the_switch(
    tmp_path, hook_runner, make_config, stamped_root, on
):
    config = make_config({"control_system": ALLOWED})
    _switch(stamped_root, "live", on=on)

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "readwrite"},
        config_path=config,
        cwd=tmp_path,
    )

    assert (result is None) is on


def test_a_tier_that_never_writes_refuses_with_the_switch_on(
    tmp_path, hook_runner, make_config, stamped_root
):
    config = make_config({"control_system": NEVER})
    _switch(stamped_root, "live")

    reason = _reason(_channel_write(tmp_path, hook_runner, config))

    assert _TWIN.refusal_phrase("tier_never") in reason
    _assert_names_no_key(reason)


def test_the_legacy_writes_enabled_false_is_a_tier_that_never_writes(
    tmp_path, hook_runner, make_config, stamped_root
):
    config = make_config({"control_system": {"type": "epics", "writes_enabled": False}})
    _switch(stamped_root, "live")

    reason = _reason(_channel_write(tmp_path, hook_runner, config))

    assert _TWIN.refusal_phrase("tier_never") in reason


def test_a_missing_config_file_denies(tmp_path, hook_runner, stamped_root):
    """No config states no tier, and a tier nobody stated allows nothing."""
    _switch(stamped_root, "live")

    reason = _reason(_channel_write(tmp_path, hook_runner, tmp_path / "nonexistent" / "c.yml"))

    assert _TWIN.refusal_phrase("tier_never") in reason


def test_stage_two_fails_closed_when_the_config_is_not_a_mapping(
    tmp_path, hook_runner, stamped_root
):
    """A `config.yml` holding a YAML list parses and has no section to ask."""
    config = tmp_path / "config.yml"
    config.write_text("- control_system\n- writes_enabled\n")
    _switch(stamped_root, "live")

    reason = _reason(_channel_write(tmp_path, hook_runner, config))

    _assert_names_no_key(reason)


#: ``(section, record target or None)`` for the parity table.
VERDICT_SHAPES = [
    ({"type": "epics", "writes": "allowed"}, "live"),
    ({"type": "epics", "writes": "never"}, "live"),
    ({"type": "epics", "writes_enabled": True}, "live"),
    ({"type": "epics"}, "live"),
    (
        {
            "type": "epics",
            "writes": "allowed",
            "connector": {"epics": {"prefix": "RING:"}, "virtual_accelerator": {"port": 1}},
        },
        "va",
    ),
    ({"type": "virtual_accelerator", "writes": "allowed"}, "va"),
    ({"type": "virtual_accelerator", "writes": "allowed"}, "live"),
    ({"type": "epics", "writes": "allowed"}, None),
    ({"type": "epics", "writes": "allowed", "login_tiers": {"nobody": "allowed"}}, "live"),
]


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
@pytest.mark.parametrize(("section", "target"), VERDICT_SHAPES)
def test_the_hook_decides_as_the_package_verdict(
    tmp_path, hook_runner, make_config, stamped_root, section, target, on
):
    """The hook and ``posture_store.write_verdict`` answer one call identically.

    The expected value is the package's verdict for the same section, record
    and identity; with no record the session sits on the baseline.
    """
    config = make_config({"control_system": section})
    if target is not None:
        _switch(stamped_root, target, on=on)
    posture_store.invalidate_cache()
    expected = posture_store.write_verdict(section, target or baseline_target(section))

    result = _channel_write(tmp_path, hook_runner, config)

    assert (result is None) is expected.allowed
    if result is not None:
        assert _TWIN.refusal_phrase(expected.reason, expected.kind) in _reason(result)


# -- calls that are not writes --------------------------------------------


def test_python_readonly_passes_with_the_switch_off(tmp_path, hook_runner, make_config):
    config = make_config({"control_system": ALLOWED})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None


def test_python_without_an_execution_mode_is_readonly(tmp_path, hook_runner, make_config):
    """The server defaults execution_mode to "readonly", and so does the hook."""
    config = make_config({"control_system": ALLOWED})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "print(42)"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None


def test_channel_read_passes(tmp_path, hook_runner, make_config):
    config = make_config({"control_system": NEVER})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_read",
        {"channels": ["SR:CURRENT:RB"]},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None


def test_non_osprey_tools_pass_through(tmp_path, hook_runner, make_config):
    config = make_config({"control_system": NEVER})

    result = hook_runner(
        "osprey_writes_check.py",
        "some_other_tool",
        {"param": "value"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None


def test_a_cloned_python_server_keeps_its_readonly_executions(tmp_path, hook_runner, make_config):
    """The readonly carve-out is per TOOL, and a clone renames only the prefix."""
    config = make_config({"control_system": NEVER})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__pyva__execute",
        {"code": "print(caget('RING:BEAM:CURRENT'))", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
        hook_config={"write_tools": ["mcp__pyva__execute"], "server_prefixes": ["mcp__pyva__"]},
    )

    assert result is None


def test_the_fallback_write_tools_gate_channel_write(tmp_path, hook_runner, make_config):
    """Without a hook_config the framework floor is still gated."""
    config = make_config({"control_system": ALLOWED})

    _reason(_channel_write(tmp_path, hook_runner, config))


# -- stage 1: a read-only run ----------------------------------------------


def test_a_readonly_run_denies_with_the_switch_on(
    tmp_path, hook_runner, make_config, monkeypatch, stamped_root
):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": ALLOWED})
    _switch(stamped_root, "live")

    reason = _reason(_channel_write(tmp_path, hook_runner, config))

    assert reason == "\U0001f512 WRITES OFF — " + _TWIN.refusal_text("readonly_run")
    _assert_names_no_key(reason)


def test_a_readonly_run_denies_a_readwrite_execution(
    tmp_path, hook_runner, make_config, monkeypatch, stamped_root
):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": ALLOWED})
    _switch(stamped_root, "live")

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "caput('PV', 1.0)", "execution_mode": "readwrite"},
        config_path=config,
        cwd=tmp_path,
    )

    assert _TWIN.refusal_phrase("readonly_run") in _reason(result)


def test_a_readonly_run_still_allows_python_readonly(
    tmp_path, hook_runner, make_config, monkeypatch
):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": ALLOWED})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
    )

    assert result is None


def test_a_readonly_run_denies_a_queue_tool(
    tmp_path, hook_runner, make_config, monkeypatch, stamped_root
):
    """Stage 1 sits ahead of the lane carve-out."""
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": ALLOWED})
    _switch(stamped_root, "live")

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

    assert _TWIN.refusal_phrase("readonly_run") in _reason(result)


@pytest.mark.parametrize("value", ["readwrite", "READONLY", "", "sandbox", "true"])
def test_a_readonly_run_is_a_value_comparison_not_a_presence_check(
    tmp_path, hook_runner, make_config, monkeypatch, stamped_root, value
):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", value)
    config = make_config({"control_system": ALLOWED})
    _switch(stamped_root, "live")

    assert _channel_write(tmp_path, hook_runner, config) is None


def test_the_readonly_deny_survives_an_unreadable_config(tmp_path, hook_runner_raw, monkeypatch):
    """Stage 1 reaches its deny without anything that can raise."""
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
    assert (
        _TWIN.refusal_phrase("readonly_run")
        in decision["hookSpecificOutput"]["permissionDecisionReason"]
    )


# -- the lane carve-out ----------------------------------------------------


def test_queue_start_skips_stage_two(tmp_path, hook_runner, make_config, stamped_root):
    """A lane-addressed tool passes with the session's switch off."""
    config = make_config({"control_system": ALLOWED})
    _switch(stamped_root, "live", on=False)

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

    assert result is None


def test_a_tool_not_listed_as_lane_addressed_is_gated_by_stage_two(
    tmp_path, hook_runner, make_config, stamped_root
):
    """The carve-out is data; its absence fails TOWARDS gating."""
    config = make_config({"control_system": ALLOWED})
    _switch(stamped_root, "live", on=False)

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

    assert _TWIN.refusal_phrase("switch_off", "real") in _reason(result)


# -- in-process: the old render, the phrases, the audit record -------------


def test_a_render_without_the_twin_refuses(tmp_path, hook_module, monkeypatch):
    """No `osprey_target_state` sibling cannot answer stage 2, and refuses."""
    hook = hook_module("osprey_writes_check")
    config = tmp_path / "config.yml"
    config.write_text("control_system:\n  type: epics\n  writes: allowed\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setattr(hook, "_target_state", None)

    answer = hook._stage_two({}, "mcp__controls__channel_write", "channel_write")

    assert answer == (False, "record_unavailable", None, None)
    assert hook._text("record_unavailable") == _TWIN.refusal_text("record_unavailable")
    assert hook._text("readonly_run") == _TWIN.refusal_text("readonly_run")


def test_the_fallback_phrases_are_the_twins(hook_module):
    hook = hook_module("osprey_writes_check")

    assert hook._READONLY_RUN_PHRASE == _TWIN.refusal_phrase("readonly_run")
    assert hook._RECORD_UNAVAILABLE_PHRASE == _TWIN.refusal_phrase("record_unavailable")


@pytest.mark.parametrize("reason", _TWIN.REASONS)
def test_every_deny_ends_with_the_refusal_trailer(hook_module, monkeypatch, capsys, reason):
    """The writes-check deny is the twin's refusal text: phrase, then the trailer line."""
    hook = hook_module("osprey_writes_check")
    monkeypatch.setattr(hook, "_record_refusal", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit):
        hook._deny({}, "mcp__controls__channel_write", reason, target="live", kind="real")

    text = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecisionReason"]
    assert text == "\U0001f512 WRITES OFF — " + _TWIN.refusal_text(reason, "real")
    assert text.splitlines()[-1] == f"[osprey-refusal: {reason}]"
    _assert_names_no_key(text)


def test_the_posture_unknown_deny_is_the_record_unavailable_refusal(
    hook_module, monkeypatch, capsys
):
    """No record where the hook looks reads as an unreadable switch, in the shared words."""
    hook = hook_module("osprey_writes_check")
    recorded = []
    monkeypatch.setattr(
        hook, "_record_refusal", lambda *args, **kwargs: recorded.append((args, kwargs))
    )

    with pytest.raises(SystemExit) as exit_info:
        hook._deny_posture_unknown({}, "mcp__controls__channel_write")

    assert exit_info.value.code == 0
    text = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecisionReason"]
    assert text == "\U0001f512 WRITES OFF — " + _TWIN.refusal_text("record_unavailable")
    _assert_names_no_key(text)
    assert "config.yml" not in text
    assert not hasattr(hook, "_POSTURE_UNKNOWN_DENY_REASON")
    # The ledger still tells "could not be read" apart from "was turned off".
    (args, kwargs) = recorded[0]
    assert args[1:] == ("mcp__controls__channel_write", "posture")
    assert kwargs == {"detail": "posture unknown"}


def test_a_render_without_the_twin_still_ends_with_the_trailer(hook_module, monkeypatch):
    hook = hook_module("osprey_writes_check")
    monkeypatch.setattr(hook, "_target_state", None)

    assert hook._text("readonly_run") == _TWIN.refusal_text("readonly_run")
    assert hook._text("switch_off", "real") == _TWIN.refusal_text("record_unavailable")


def test_a_deny_records_posture_with_the_reason_and_target(hook_module, monkeypatch, capsys):
    """Every verdict refusal records reason `posture`, detail `reason=<r> target=<t>`."""
    hook = hook_module("osprey_writes_check")
    recorded = []
    monkeypatch.setattr(
        hook, "_record_refusal", lambda *args, **kwargs: recorded.append((args, kwargs))
    )

    with pytest.raises(SystemExit) as exit_info:
        hook._deny({}, "mcp__controls__channel_write", "switch_off", target="live", kind="real")

    assert exit_info.value.code == 0
    (args, kwargs) = recorded[0]
    assert args[1:] == ("mcp__controls__channel_write", "posture")
    assert kwargs == {"detail": "reason=switch_off target=live"}
    reason = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecisionReason"]
    assert reason == "\U0001f512 WRITES OFF — " + _TWIN.refusal_text("switch_off", "real")


def test_the_writes_disabled_audit_word_is_retired(hook_module):
    """One word for every refusal: the stage-2 `writes_disabled` word is gone."""
    hook = hook_module("osprey_writes_check")

    assert not hasattr(hook, "_WRITES_DISABLED_AUDIT_REASON")
    assert hook._POSTURE_DENY_AUDIT_REASON == "posture"
