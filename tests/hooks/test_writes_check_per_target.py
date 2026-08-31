"""Stage 2 of ``osprey_writes_check`` against the per-(session, target) store.

A write is gated by two things that two different actions lift: the deployment's
own posture for the target, which moves in ``config.yml``, and the operator's
narrowing of ONE target for ONE session, which moves on the control-target chip
in the header. This module pins the fork between them — a refusal that names the
wrong control sends the operator to one that will not move — and the one cell
where the posture cannot be read at all.

The hook is run as a subprocess through ``hook_runner``, the way Claude Code
runs it, so the env stamps, the state file and the store are exercised exactly
as a real session presents them. The state file this module writes is genuinely
owned by the pytest process, which IS on the hook subprocess's ancestor chain,
so the reader's real parentage and liveness rules select it with no seam
replaced.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.unit

#: The store key a session carries. The canonical UUID shape the web terminal
#: writes; the ``operator-`` shape is exercised separately below.
SESSION_KEY = "4f1c2a7e-0000-4000-8000-000000000001"

#: A switch-capable deployment armed for BOTH of its targets, so that every
#: refusal below can only have come from the store.
ARMED_BOTH = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {
        "epics": {"prefix": "RING:"},
        "virtual_accelerator": {"prefix": "VA:"},
    },
}

#: The same deployment with its ring disarmed in config — the shape whose
#: refusal must keep naming a config key.
DISARMED_LIVE = {
    "type": "epics",
    "writes_enabled": True,
    "connector": {
        "epics": {"prefix": "RING:", "writes_enabled": False},
        "virtual_accelerator": {"prefix": "VA:"},
    },
}


def agent_data_root(repo_root):
    """The agent-data root the hook DERIVES from *repo_root* when unstamped."""
    return repo_root / "var" / "agent_data"


def state_dir(repo_root):
    """The control-target state directory, created."""
    directory = agent_data_root(repo_root) / "control_target"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_session_state(repo_root, target):
    """Write a state file this pytest process genuinely owns."""
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
        },
        "children": [],
    }
    path = state_dir(repo_root) / f"target_state_{os.getpid()}.json"
    path.write_text(json.dumps(record), encoding="utf-8")


def write_store(repo_root, payload):
    """Write the posture store where the hook will look for it."""
    path = state_dir(repo_root) / "session-postures.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def channel_write(tmp_path, hook_runner, config):
    """Run the hook against a ``channel_write`` call, return its decision."""
    return hook_runner(
        "osprey_writes_check.py",
        "mcp__controls__channel_write",
        {"operations": [{"channel": "RING:QF:SP", "value": 1.5}]},
        config_path=config,
        cwd=tmp_path,
    )


def reason_of(result):
    """The operator-facing refusal text, asserting there was one."""
    assert result is not None, "expected a deny, got a pass-through"
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    return output["permissionDecisionReason"]


# ---------------------------------------------------------------------------
# a narrowing refuses, in the posture vocabulary, naming the target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["live", "va"])
def test_a_narrowed_target_denies_and_names_the_chip_and_the_target(
    tmp_path, hook_runner, make_config, monkeypatch, target
):
    """The refusal the header chip lifts says so, and says which machine.

    Two-vocabulary rule: the config arms both targets, so naming
    ``writes_enabled`` here would send the operator to flip a key already set to
    the value being asked for. Naming no target would describe the session-wide
    sandbox, which this session is not in.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, target)
    write_store(tmp_path, {SESSION_KEY: {target: "sandbox"}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES OFF" in reason
    assert "control-target chip in the header" in reason
    assert f"to the {target} target" in reason
    assert "writes_enabled" not in reason
    assert "WRITES DISABLED" not in reason
    assert "terminal card" not in reason


def test_the_verbatim_sentences_of_the_per_target_refusal(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The whole message, pinned. It is the only place this rule is explained."""
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "va")
    write_store(tmp_path, {SESSION_KEY: {"va": "sandbox"}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert reason == (
        "\U0001f512 WRITES OFF — this session refuses control-system "
        "writes to the va target.\n\n"
        "Turn writes back on from the control-target chip in the header; "
        "config.yml is not the gate here."
    )


def test_a_narrowing_on_another_target_leaves_this_one_alone(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Per-TARGET is the whole point: sandboxing the ring must not stop the VA."""
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "va")
    write_store(tmp_path, {SESSION_KEY: {"live": "sandbox"}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_another_sessions_narrowing_is_not_this_sessions(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The store is keyed by session, and the key is the whole match."""
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "va")
    write_store(tmp_path, {"some-other-session": "sandbox"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_a_session_with_no_key_never_reads_the_store(tmp_path, hook_runner, make_config):
    """Every CLI session: nothing addressed it, so nothing narrowed it.

    The store below sandboxes both targets under the key a web session would
    carry. A reader that matched on the file rather than on the key would
    sandbox every terminal on the host the moment one operator narrowed one
    browser tab.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "live")
    write_store(tmp_path, {SESSION_KEY: "sandbox"})

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_the_legacy_bare_sandbox_narrows_the_session_target(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The shape the session-wide posture wrote before targets existed.

    It narrowed the whole session, so it still narrows whichever target the
    session is on: an upgrade must not lift a live narrowing.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "live")
    write_store(tmp_path, {SESSION_KEY: "sandbox"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES OFF" in reason
    assert "to the live target" in reason


def test_a_bare_writes_entry_is_not_a_narrowing(tmp_path, hook_runner, make_config, monkeypatch):
    """The writes posture is the ABSENCE of an entry, never a stored assertion.

    Nothing in this file may widen anything, so the value some writers record
    for "not narrowed" has to mean exactly what no entry at all means.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "live")
    write_store(tmp_path, {SESSION_KEY: "writes"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_an_operator_key_narrows_like_any_other(tmp_path, hook_runner, make_config, monkeypatch):
    """``operator-`` keys are RETAINED by every enforcement reader.

    Their drop-on-restore rule belongs to the web server's startup load alone; a
    hook that dropped them would ignore a narrowing that is live for the rest of
    that session's life.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "live")
    write_store(tmp_path, {"operator-console-1": {"live": "sandbox"}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", "operator-console-1")

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES OFF" in reason
    assert "to the live target" in reason


def test_an_unidentified_target_takes_the_most_restrictive_entry(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """No state file, but a live one belonging to nobody on our chain.

    The session cannot say which machine it is about, so any sandbox in its
    record refuses. The refusal names no target, because none was decided.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    # A live record owned by a PID that is on no ancestor chain of ours: enough
    # to make the derived directory credible, not enough to resolve a target.
    stranger = {
        "target": "live",
        "generation": 1,
        "server_pid": os.getpid(),
        "owner_ppid": 1,
        "targets": {"live": {"label": "Ring", "endpoint": "pva://", "real_machine": True}},
        "children": [],
    }
    (state_dir(tmp_path) / "target_state_stranger.json").write_text(
        json.dumps(stranger), encoding="utf-8"
    )
    write_store(tmp_path, {SESSION_KEY: {"va": "sandbox"}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert — the scope phrase is absent, because no target was decided
    assert reason == (
        "\U0001f512 WRITES OFF — this session refuses control-system "
        "writes.\n\n"
        "Turn writes back on from the control-target chip in the header; "
        "config.yml is not the gate here."
    )


# ---------------------------------------------------------------------------
# the deployment's own refusal keeps its own vocabulary
# ---------------------------------------------------------------------------


def test_an_unarmed_deployment_still_names_the_config_key(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The fork's other side: no narrowing, so config.yml IS the gate here."""
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})
    write_session_state(tmp_path, "live")
    write_store(tmp_path, {SESSION_KEY: {"va": "sandbox"}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES DISABLED" in reason
    assert "control_system.connector.epics.writes_enabled: true" in reason
    assert "WRITES OFF" not in reason
    assert "control-target chip" not in reason


def test_a_narrowing_wins_the_wording_over_an_unarmed_deployment(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Both gates shut at once, and the session's is the one to say.

    Arming ``control_system.connector.epics.writes_enabled`` would not lift this
    write; lifting the narrowing and arming the key both would. Naming the key
    alone sends the operator to a control that leaves them still refused, with
    nothing on screen to say why.
    """
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})
    write_session_state(tmp_path, "live")
    write_store(tmp_path, {SESSION_KEY: {"live": "sandbox"}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES OFF" in reason
    assert "to the live target" in reason
    assert "WRITES DISABLED" not in reason


# ---------------------------------------------------------------------------
# posture unknown — exactly one cell
# ---------------------------------------------------------------------------


def test_posture_unknown_fires_unstamped_with_no_live_record(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The one cell where an empty store proves nothing.

    A session key says a narrowing is possible; an unstamped root says the
    directory was guessed; no live record there says the guess was not confirmed.
    An unreadable posture is not a permissive one.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert reason == (
        "\U0001f512 WRITE STATE UNKNOWN — this session carries a posture key, "
        "but no live control-target state was found where this hook looks, so "
        "the write state set on the control-target chip in the header cannot "
        "be read.\n\n"
        "Writes stay refused until the controls MCP server is running; "
        "config.yml is not the gate here."
    )
    assert "writes_enabled" not in reason


def test_the_stamp_alone_lifts_posture_unknown(tmp_path, hook_runner, make_config, monkeypatch):
    """Stamped, an empty store means exactly what it says.

    The stamp is handed to the child by the same process that writes the store,
    so there is nothing left to be uncertain about — no live server is needed
    for the absence of a narrowing to be trustworthy.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(agent_data_root(tmp_path)))

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_a_live_record_alone_lifts_posture_unknown(tmp_path, hook_runner, make_config, monkeypatch):
    """Unstamped, a live record in the derived directory is the evidence.

    It is the ordinary shape: a session launched before the stamp existed, or a
    deployment that never moved ``agent_data.base_dir``. The derivation found a
    directory a running server writes to, so the store read there is the store.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "live")
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_posture_unknown_does_not_fire_without_a_session_key(tmp_path, hook_runner, make_config):
    """No key, no narrowing to miss.

    This is every CLI session on the host, and it is why the refusal is gated on
    the key rather than on the missing state: without one, "no state" is the
    baseline fallback the deployment ceiling already answers for.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_posture_unknown_outranks_an_unarmed_deployment(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """No config key would lift it, so no config key is named.

    Refused before the ceiling is consulted at all: telling an operator to arm
    ``writes_enabled`` here would be an instruction that changes nothing.
    """
    # Arrange
    config = make_config({"control_system": {"type": "epics", "writes_enabled": False}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITE STATE UNKNOWN" in reason
    assert "WRITES DISABLED" not in reason


def test_a_readonly_run_still_refuses_before_stage_two(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Stage 1 is untouched and still answers first, from the environment alone.

    Its message names no target on purpose: the session-wide posture is decided
    ahead of any config or state I/O, and resolving a target there would make
    that answer depend on the very reads it is placed before.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "va")
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES OFF" in reason
    assert "control-target chip in the header" in reason
    assert "target." not in reason


# ---------------------------------------------------------------------------
# the store never widens
# ---------------------------------------------------------------------------


def test_no_store_entry_can_arm_an_unarmed_deployment(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Narrowing-only, stated as a test.

    The store holds narrowings and nothing else — there is no value an operator
    or a hand-edit can put in it that lifts a deployment's own refusal.
    """
    # Arrange
    config = make_config({"control_system": {"type": "epics", "writes_enabled": False}})
    write_session_state(tmp_path, "live")
    write_store(tmp_path, {SESSION_KEY: {"live": "writes"}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES DISABLED" in reason


def test_an_unknown_leaf_is_dropped_rather_than_honoured(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A future-version or hand-edited value must not reach the decision.

    Dropping it is the safe direction here precisely because the store can only
    narrow: what survives the filter decides whether a real machine is written
    to, and nothing that survives it can widen anything.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "live")
    write_store(tmp_path, {SESSION_KEY: {"live": "locked"}})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_a_corrupt_store_does_not_wedge_the_write_path(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A file nobody can repair from the browser must not refuse everything.

    Losing narrowings an operator can set again is the lesser harm, and it is
    the one the canonical reader chose.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "live")
    (state_dir(tmp_path) / "session-postures.json").write_text('{"a": ', encoding="utf-8")
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_readonly_execution_is_still_allowed_under_a_narrowing(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Looking at the machine is exactly what a narrowed target is for."""
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_session_state(tmp_path, "live")
    write_store(tmp_path, {SESSION_KEY: "sandbox"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    result = hook_runner(
        "osprey_writes_check.py",
        "mcp__python__execute",
        {"code": "print(42)", "execution_mode": "readonly"},
        config_path=config,
        cwd=tmp_path,
    )

    # Assert
    assert result is None
