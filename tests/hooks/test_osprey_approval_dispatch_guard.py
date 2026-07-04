"""Tests for the approval hook's dispatch-context guard.

In headless dispatch runs the worker sets ``OSPREY_DISPATCH_RUN=1``. The
approval hook must then never emit an explicit ``permissionDecision: allow``
— CLI hook aggregation is not deny-dominates, so an explicit allow would
override the dispatch worker's per-trigger allowlist hook. Ask output is
unaffected: it falls through to the worker's context-aware ``can_use_tool``
backstop, which decides.
"""

import pytest

DISPATCH_HOOK_CONFIG = {
    "server_prefixes": ["mcp__controls__"],
    "approval_prefixes": ["mcp__controls__"],
}

WRITE_INPUT = {"operations": [{"channel": "TEST:PV", "value": 1.0}]}


def _decision(result) -> str | None:
    if result is None:
        return None
    return result.get("hookSpecificOutput", {}).get("permissionDecision")


@pytest.mark.unit
def test_allow_path_emits_no_decision_in_dispatch_context(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """approval.enabled=false normally emits explicit allow — not under dispatch."""
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    config = make_config(
        {"approval": {"enabled": False}, "control_system": {"writes_enabled": True}}
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__archiver_read",
        {"channels": ["TEST:PV"]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DISPATCH_HOOK_CONFIG,
    )

    # No decision (empty/none output) — never an explicit allow
    assert _decision(result) is None


@pytest.mark.unit
def test_allow_path_unchanged_outside_dispatch_context(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Without the env var the explicit allow is preserved (web-terminal behavior)."""
    monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    config = make_config(
        {"approval": {"enabled": False}, "control_system": {"writes_enabled": True}}
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__archiver_read",
        {"channels": ["TEST:PV"]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DISPATCH_HOOK_CONFIG,
    )

    assert _decision(result) == "allow"


@pytest.mark.unit
def test_skip_policy_emits_no_decision_in_dispatch_context(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Per-tool policy 'skip' is the other explicit-allow path — also guarded."""
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    config = make_config(
        {
            "approval": {
                "enabled": True,
                "default_policy": "always",
                "tools": {"archiver_read": "skip"},
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__archiver_read",
        {"channels": ["TEST:PV"]},
        config_path=config,
        cwd=tmp_path,
        hook_config=DISPATCH_HOOK_CONFIG,
    )

    assert _decision(result) is None


@pytest.mark.unit
def test_ask_path_still_asks_in_dispatch_context(tmp_path, hook_runner, make_config, monkeypatch):
    """The approval ask-gate for writes is untouched by the dispatch guard."""
    monkeypatch.setenv("OSPREY_DISPATCH_RUN", "1")
    config = make_config(
        {
            "approval": {
                "enabled": True,
                "default_policy": "selective",
                "requires_approval": ["channel_write", "execute"],
            },
            "control_system": {"writes_enabled": True},
        }
    )

    result = hook_runner(
        "osprey_approval.py",
        "mcp__controls__channel_write",
        WRITE_INPUT,
        config_path=config,
        cwd=tmp_path,
        hook_config=DISPATCH_HOOK_CONFIG,
    )

    assert _decision(result) == "ask"
