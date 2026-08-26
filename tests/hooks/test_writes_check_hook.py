"""Tests for the osprey_writes_check hook.

This hook enforces the master writes kill switch (control_system.writes_enabled).
When disabled, it blocks channel_write and write-mode python_execute.
Read-only tools and readonly python always pass through.
"""

import json

import pytest


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
    """If config.yml doesn't exist, writes_enabled defaults to False (fail-closed).

    The hook's load_osprey_config() returns {} when the file is missing,
    and writes_enabled defaults to False, which blocks writes. This is the
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
    """Config exists but has no writes_enabled key → defaults to False (deny).

    The hook uses .get("writes_enabled", False), so a missing key is treated
    as writes disabled. This is intentionally fail-closed.
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
    session's own posture is the gate, and the terminal card is where it moves.
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
    assert "SANDBOX POSTURE" in reason
    assert "writes_enabled" not in reason
    assert "WRITES DISABLED" not in reason
    assert "terminal card" in reason
    # The two sentences the operator needs, verbatim.
    assert "this terminal session refuses control-system writes." in reason
    assert (
        "Switch the session to writes posture from the terminal card; "
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
    every deployment that never touches the terminal card — so it is pinned
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
    assert "SANDBOX POSTURE" in decision["hookSpecificOutput"]["permissionDecisionReason"]


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
    assert "SANDBOX POSTURE" in result["hookSpecificOutput"]["permissionDecisionReason"]


# -- Server-level coverage for facility-custom servers (``write_servers``) --
#
# The registry gives a facility-custom server that opts into ``writes_check`` a
# REGEX matcher (``mcp__<name>__.*``): its tool names are unknown at render
# time. That matcher lands in ``write_tools`` like any other, where the hook's
# exact-name membership test can never match a real call against it. The render
# therefore also names the server in ``write_servers``, and the hook covers it
# by prefix — but ONLY under the sandbox posture, which is the one gate that has
# no way to tell such a server's reads from its writes and refuses the whole
# server rather than guessing.

#: A custom server as the render describes it: the unmatchable regex in
#: ``write_tools``, the bare name in ``write_servers``.
_CUSTOM_SERVER_CONFIG = {
    "write_tools": ["mcp__sitectl__.*"],
    "write_servers": ["sitectl"],
}


@pytest.mark.unit
def test_readonly_posture_refuses_a_custom_server_tool_by_prefix(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The gap this key closes: an exact-name miss that is still a write.

    ``mcp__sitectl__set_mode`` is not in ``write_tools`` — only the regex that
    can never match it is — so before ``write_servers`` the sandbox posture let
    the call straight through. The server name is the unit the render can state
    honestly, and it is the same server-level shape the SDK disallow engine
    honors.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__sitectl__set_mode",
        {"mode": "injection"},
        config_path=config,
        cwd=tmp_path,
        hook_config=_CUSTOM_SERVER_CONFIG,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_readonly_posture_refusal_speaks_the_posture_vocabulary(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A server-level refusal is a posture refusal, word for word.

    It is reached from the posture and lifted by the posture, so it must not
    send the operator to ``config.yml`` — the same two-vocabulary rule the
    exact-name posture deny obeys, and the same message, not a second dialect
    for the operator to learn.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__sitectl__set_mode",
        {"mode": "injection"},
        config_path=config,
        cwd=tmp_path,
        hook_config=_CUSTOM_SERVER_CONFIG,
    )

    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "SANDBOX POSTURE" in reason
    assert "writes_enabled" not in reason
    assert "WRITES DISABLED" not in reason
    assert "terminal card" in reason


@pytest.mark.unit
def test_readonly_posture_refuses_a_custom_server_read_too(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Server-level means server-level: the reads go too, deliberately.

    Nothing in the render knows which of a custom server's tools write, so under
    the sandbox posture the honest answer for all of them is refusal. This is
    the one place the coverage is coarser than a framework server's, and it is
    confined to the posture path for exactly that reason.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__sitectl__get_status",
        {},
        config_path=config,
        cwd=tmp_path,
        hook_config=_CUSTOM_SERVER_CONFIG,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
@pytest.mark.parametrize("posture", [None, "readwrite", "READONLY", "", "sandbox"])
def test_write_servers_is_inert_off_the_readonly_path(
    tmp_path, hook_runner, make_config, monkeypatch, posture
):
    """Every posture that is not the sandbox posture exits 0 — including none.

    The prefix test is deliberately confined to the readonly path. On any other
    posture a custom server keeps behaving exactly as it did before this key
    existed, which is what makes the change safe to ship to deployments whose
    custom servers are mostly reads.
    """
    if posture is None:
        monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    else:
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", posture)
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__sitectl__set_mode",
        {"mode": "injection"},
        config_path=config,
        cwd=tmp_path,
        hook_config=_CUSTOM_SERVER_CONFIG,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_writes_off_does_not_reach_the_kill_switch_via_write_servers(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The regression this ordering exists to prevent.

    ``writes_enabled: false`` is the deployment kill switch. If the prefix test
    ran before it instead of only on the posture path, every read tool of every
    custom server would start refusing the moment an operator turned writes off
    — a deployment-wide outage dressed up as a safety win. The call falls
    through the exact-name miss and exits 0, kill switch or not.
    """
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__sitectl__get_status",
        {},
        config_path=config,
        cwd=tmp_path,
        hook_config=_CUSTOM_SERVER_CONFIG,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_write_servers_does_not_widen_to_a_server_it_does_not_name(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A listed server covers its own prefix and nothing else.

    ``mcp__sitectl_extra__`` shares a leading substring with ``sitectl`` and is a
    different server; the ``__`` terminator in the composed prefix is what keeps
    the two apart.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__sitectl_extra__set_mode",
        {"mode": "injection"},
        config_path=config,
        cwd=tmp_path,
        hook_config=_CUSTOM_SERVER_CONFIG,
    )

    assert result is None  # Allowed through


@pytest.mark.unit
def test_exact_name_write_tool_still_denies_under_writes_off_with_write_servers(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The pre-existing branches are untouched by the new key.

    A framework write tool is an exact-name hit, so it never reaches the prefix
    test at all and the kill switch answers it exactly as before.
    """
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    config = make_config({"control_system": {"writes_enabled": False}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
        hook_config={
            "write_tools": ["mcp__controls__channel_write", "mcp__sitectl__.*"],
            "write_servers": ["sitectl"],
        },
    )

    assert result is not None
    assert "WRITES DISABLED" in result["hookSpecificOutput"]["permissionDecisionReason"]


# -- Degrade: no server name is deployment-independent, so there is no floor --


@pytest.mark.unit
@pytest.mark.parametrize(
    "write_servers",
    [None, "sitectl", {"sitectl": True}, 7],
    ids=["absent", "string", "mapping", "number"],
)
def test_unusable_write_servers_degrades_to_no_coverage(
    tmp_path, hook_runner_raw, make_config, monkeypatch, write_servers
):
    """Absent or malformed ⇒ the empty list ⇒ exactly pre-feature behavior.

    Deliberate asymmetry with ``write_tools``, which degrades to a fail-closed
    floor: server names are deployment-specific, so no floor this hook could
    ship would name the right ones. The refusal is dropped rather than guessed
    at, and the WARNING is what keeps that from being silent.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})
    hook_config = {"write_tools": ["mcp__sitectl__.*"]}
    if write_servers is not None:
        hook_config["write_servers"] = write_servers

    returncode, stdout, stderr = hook_runner_raw(
        "osprey_writes_check.py",
        "mcp__sitectl__set_mode",
        {"mode": "injection"},
        config_path=config,
        cwd=tmp_path,
        hook_config=hook_config,
    )

    assert returncode == 0
    assert stdout.strip() == ""  # no decision — the call goes through
    assert "Traceback" not in stderr
    assert "WARNING" in stderr
    assert "write_servers" in stderr
    assert "osprey build" in stderr  # the remedy, named


@pytest.mark.unit
def test_one_malformed_entry_does_not_cost_the_whole_list_its_coverage(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A bad element is dropped; the good ones still refuse.

    ``_get_write_servers`` filters the list element by element, and that filter
    is load-bearing rather than tidy. Without it a non-string entry makes
    ``"mcp__" + entry`` raise inside ``_write_server_for``, whose bare ``except``
    exists to keep a hook from crashing a tool call — so the refusal for EVERY
    later entry is swallowed with it and a sandbox deny becomes a silent allow.

    The malformed entries come FIRST for exactly that reason: a filter-less
    loop never reaches ``sitectl``.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__sitectl__set_mode",
        {"mode": "injection"},
        config_path=config,
        cwd=tmp_path,
        hook_config={
            "write_tools": ["mcp__sitectl__.*"],
            "write_servers": [7, None, "", "sitectl"],
        },
    )

    assert result is not None, "a malformed entry silenced the whole list"
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "SANDBOX POSTURE" in result["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.unit
def test_a_render_that_names_write_servers_warns_about_nothing(
    tmp_path, hook_runner_raw, make_config, monkeypatch
):
    """A current render is silent — including one whose list is legitimately empty.

    ``write_servers: []`` is what every default build emits (all framework
    writes_check matchers are exact names), so an empty list must read as
    "nothing to cover", never as "your render is stale".
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    returncode, stdout, stderr = hook_runner_raw(
        "osprey_writes_check.py",
        "mcp__sitectl__set_mode",
        {"mode": "injection"},
        config_path=config,
        cwd=tmp_path,
        hook_config={"write_tools": ["mcp__sitectl__.*"], "write_servers": []},
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "WARNING" not in stderr


@pytest.mark.unit
def test_absent_hook_config_stays_on_the_write_tools_fallback_silently(
    tmp_path, hook_runner_raw, make_config, monkeypatch
):
    """No ``hook_config.json`` at all is the *file* degrade, not this key's.

    ``write_tools`` already answers that case with its fail-closed floor and no
    warning, and the render emits both keys together — so a config the hook
    cannot read at all says nothing about ``write_servers`` specifically, and a
    second warning there would fire on every framework write call.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(tmp_path / "no-such-hook-config.json"))
    config = make_config({"control_system": {"writes_enabled": True}})

    returncode, stdout, stderr = hook_runner_raw(
        "osprey_writes_check.py",
        "mcp__sitectl__set_mode",
        {"mode": "injection"},
        config_path=config,
        cwd=tmp_path,
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "WARNING" not in stderr


@pytest.mark.unit
def test_the_warning_does_not_corrupt_a_decision_on_stdout(
    tmp_path, hook_runner_raw, make_config, monkeypatch
):
    """The WARNING goes to stderr, so stdout stays parseable JSON.

    Claude Code reads a PreToolUse decision off stdout. A warning printed there
    would make a deny unreadable and fail the call open, which is the worst
    possible way to report a stale render.
    """
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    config = make_config({"control_system": {"writes_enabled": True}})

    returncode, stdout, stderr = hook_runner_raw(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "TEST:PV", "value": 1.0}]},
        config_path=config,
        cwd=tmp_path,
        hook_config={"write_tools": ["mcp__controls__channel_write"]},
    )

    assert returncode == 0
    decision = json.loads(stdout)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "SANDBOX POSTURE" in decision["hookSpecificOutput"]["permissionDecisionReason"]
