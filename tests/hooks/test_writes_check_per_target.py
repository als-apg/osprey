"""Stage 2 of ``osprey_writes_check`` against the control-context record.

A write is gated by two things that two different actions lift: the deployment's
own posture for the target, which moves in the build profile, and the operator's
narrowing of ONE target, which moves on the control-target chip in the header
and is recorded in the deployment's control-context record. This module pins the
fork between them — a refusal that names the wrong control sends the operator to
one that will not move — and the one cell where the posture cannot be read at
all.

The hook is run as a subprocess through ``hook_runner``, the way Claude Code
runs it, so the env stamps and the record are exercised exactly as a real
session presents them. The record this module writes lands under the agent-data
root the hook DERIVES from its own ``cwd``, so the root anchor and the read are
both real and no seam is replaced.

The narrowing is the deployment's, not a session's: there is one record per
deployment instance and every process that can find it is subject to what it
says. A session key is carried only so a write-approval stamp can say WHO
approved, and nothing here is keyed by it.
"""

from __future__ import annotations

import pytest

from tests._control_context_fixtures import write_control_context, write_payload

pytestmark = pytest.mark.unit

#: The audit id a web-terminal session carries. It keys nothing; it is set in
#: the tests that need a session to be identifiable at all.
SESSION_KEY = "4f1c2a7e-0000-4000-8000-000000000001"

#: A switch-capable deployment armed for BOTH of its targets, so that every
#: refusal below can only have come from the record.
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


def record_path(repo_root):
    """Where the record lands under *repo_root*, directory created."""
    directory = agent_data_root(repo_root) / "control_target"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "control_context.json"


def write_record(repo_root, target, posture=None):
    """Write the deployment's record where the hook will look for it."""
    write_control_context(agent_data_root(repo_root), target=target, generation=3, posture=posture)


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
    the value being asked for. Naming no target would describe a deployment-wide
    sandbox, which this deployment is not in.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_record(tmp_path, target, posture={target: "sandbox"})
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
    write_record(tmp_path, "va", posture={"va": "sandbox"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert reason == (
        "\U0001f512 WRITES OFF — this deployment refuses control-system "
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
    write_record(tmp_path, "va", posture={"live": "sandbox"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_a_narrowing_reaches_a_session_that_carries_no_key(tmp_path, hook_runner, make_config):
    """The narrowing is the deployment's, so a bare ``claude`` is subject to it.

    Nothing addressed this process and it carries no audit id, but it can read
    the record — and the record is what the operator narrowed. A reader that
    required a key here would leave every CLI session writing to a machine the
    operator had just taken writes off.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_record(tmp_path, "live", posture={"live": "sandbox"})

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES OFF" in reason
    assert "to the live target" in reason


def test_the_legacy_bare_sandbox_narrows_the_recorded_target(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The shape the session-wide posture wrote before targets existed.

    It narrowed everything, so it still narrows whichever target the deployment
    is on: an upgrade must not lift a live narrowing.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_record(tmp_path, "live", posture="sandbox")
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES OFF" in reason
    assert "to the live target" in reason


def test_a_bare_writes_entry_is_not_a_narrowing(tmp_path, hook_runner, make_config, monkeypatch):
    """The writes posture is the ABSENCE of an entry, never a stored assertion.

    Nothing in the record may widen anything, so the value some writers record
    for "not narrowed" has to mean exactly what no entry at all means.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_record(tmp_path, "live", posture="writes")
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


# ---------------------------------------------------------------------------
# the deployment's own refusal keeps its own vocabulary
# ---------------------------------------------------------------------------


def test_an_unarmed_deployment_still_names_the_config_key(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The fork's other side: no narrowing, so config.yml IS the gate here."""
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})
    write_record(tmp_path, "live", posture={"va": "sandbox"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES DISABLED" in reason
    assert "control_system.connector.epics.writes_enabled: true" in reason
    assert "WRITES OFF" not in reason
    assert "control-target chip" not in reason


def test_the_arming_line_names_the_profile_the_build_reads(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The key is only half the remedy; the other half is WHERE to write it.

    ``config.yml`` is regenerated by every ``osprey build``, so an operator who
    follows a refusal there watches the change disappear on the next build. The
    durable surface is the build profile's ``config:`` block, which is what the
    health checks already name for the settings they ask for.
    """
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})
    write_record(tmp_path, "live", posture={"va": "sandbox"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "profile.yml" in reason
    assert "osprey build" in reason
    assert "in config.yml" not in reason


def test_a_narrowing_wins_the_wording_over_an_unarmed_deployment(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Both gates shut at once, and the record's is the one to say.

    Arming ``control_system.connector.epics.writes_enabled`` would not lift this
    write; lifting the narrowing and arming the key both would. Naming the key
    alone sends the operator to a control that leaves them still refused, with
    nothing on screen to say why.
    """
    # Arrange
    config = make_config({"control_system": DISARMED_LIVE})
    write_record(tmp_path, "live", posture={"live": "sandbox"})
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


def test_posture_unknown_fires_unstamped_with_no_record(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """The one cell where an empty answer proves nothing.

    An unstamped root says the directory was guessed; no record there says the
    guess was not confirmed. An unreadable posture is not a permissive one.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert reason == (
        "\U0001f512 WRITE STATE UNKNOWN — no control-context record was found "
        "where this hook looks, so the write state set on the control-target "
        "chip in the header cannot be read.\n\n"
        "Writes stay refused until the controls MCP server is running; "
        "config.yml is not the gate here."
    )
    assert "writes_enabled" not in reason


def test_posture_unknown_fires_for_a_bare_claude_too(tmp_path, hook_runner, make_config):
    """No audit id is not evidence that nothing was narrowed.

    A bare ``claude`` and a dispatch worker carry no session key and no stamp,
    and before the first controls server publishes there is no record for them
    to read either. Refused until there is — and allowed on retry the moment the
    server writes one, which is the whole shape of this refusal.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITE STATE UNKNOWN" in reason
    assert "until the controls MCP server is running" in reason


def test_the_stamp_alone_lifts_posture_unknown(tmp_path, hook_runner, make_config, monkeypatch):
    """Stamped, an empty directory means exactly what it says.

    The stamp is handed to the child by the same process that writes the record,
    so there is nothing left to be uncertain about — no running server is needed
    for the absence of a narrowing to be trustworthy.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(agent_data_root(tmp_path)))

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_a_record_alone_lifts_posture_unknown(tmp_path, hook_runner, make_config, monkeypatch):
    """Unstamped, a readable record in the derived directory is the evidence.

    It is the ordinary shape: a session launched before the stamp existed, or a
    deployment that never moved ``agent_data.base_dir``. The derivation found
    the directory the owner writes to, so the posture read there is the posture.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_record(tmp_path, "live")
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

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

    Its message names no target on purpose: the read-only run is decided ahead
    of any config or record I/O, and resolving a target there would make that
    answer depend on the very reads it is placed before.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_record(tmp_path, "va")
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES OFF" in reason
    assert "control-target chip in the header" in reason
    assert "target." not in reason


# ---------------------------------------------------------------------------
# the record never widens
# ---------------------------------------------------------------------------


def test_no_recorded_posture_can_arm_an_unarmed_deployment(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Narrowing-only, stated as a test.

    The record holds narrowings and nothing else — there is no value an operator
    or a hand-edit can put in it that lifts a deployment's own refusal.
    """
    # Arrange
    config = make_config({"control_system": {"type": "epics", "writes_enabled": False}})
    write_record(tmp_path, "live", posture={"live": "writes"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITES DISABLED" in reason


def test_an_unknown_leaf_is_dropped_rather_than_honoured(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A future-version or hand-edited value must not reach the decision.

    Dropping it is the safe direction here precisely because the record can only
    narrow: what survives the filter decides whether a real machine is written
    to, and nothing that survives it can widen anything.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_record(tmp_path, "live", posture={"live": "locked"})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_a_posture_that_is_not_a_map_does_not_wedge_the_write_path(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """``posture`` degrades on its own, without taking the identity with it.

    The record's target and generation are all-or-nothing; the narrowings beside
    them are not. A posture field written as a list narrows nothing and leaves
    the rest of the record readable, which is what keeps a writer's schema slip
    from refusing every write on the deployment.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_record(tmp_path, "live", posture=["live"])
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_a_corrupt_record_under_a_stamped_root_does_not_wedge_the_write_path(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """A file nobody can repair from the browser must not refuse everything.

    Losing narrowings an operator can set again is the lesser harm, and it is
    the one the canonical reader chose. The stamp is what makes it safe: the
    process that handed it over is the one that writes the record, so an
    unreadable file there is corruption and not a directory that was guessed.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    record_path(tmp_path).write_text('{"schema": ', encoding="utf-8")
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(agent_data_root(tmp_path)))

    # Act / Assert
    assert channel_write(tmp_path, hook_runner, config) is None


def test_a_corrupt_record_under_a_derived_root_refuses(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Unstamped, corruption is indistinguishable from the wrong directory.

    Both halves of ``posture_unknown`` hold: nothing handed this process a root,
    and what it found where it guessed cannot be read. There is no evidence the
    guess was right, so the write waits for the server rather than proceeding.
    """
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_payload(record_path(tmp_path), {"schema": 99, "target": "live", "generation": 3})
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.delenv("OSPREY_AGENT_DATA_ROOT", raising=False)

    # Act
    reason = reason_of(channel_write(tmp_path, hook_runner, config))

    # Assert
    assert "WRITE STATE UNKNOWN" in reason


def test_readonly_execution_is_still_allowed_under_a_narrowing(
    tmp_path, hook_runner, make_config, monkeypatch
):
    """Looking at the machine is exactly what a narrowed target is for."""
    # Arrange
    config = make_config({"control_system": ARMED_BOTH})
    write_record(tmp_path, "live", posture="sandbox")
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
