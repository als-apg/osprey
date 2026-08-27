"""The hot/cold tables must classify the target-switch keys, and stay in lockstep.

Two tables tell an operator whether a `setup_patch` has landed or is still
waiting on a restart: ``_HOT_CHANGE_PATHS`` / ``_COLD_CHANGE_NOTES`` in the
setup tool, and the "Hot vs. Cold Changes" table in the setup-mode skill. They
are read by different audiences and edited at different times, which is exactly
how they drift — so the lockstep test below walks every key the tool classifies
and demands a matching skill row, rather than naming keys one at a time.

The `target_switch.*` keys are cold for one reason worth stating plainly: the
controls server reads them through a config it caches at launch, so patching
them changes nothing until the server restarts. Only per-call hook subprocesses
are hot. `control_system.type` stays cold too, but its note now has a job the
others do not have: telling the operator that changing control *target* is a
run-time tool call (`control_target_set`), not a config edit and a rebuild.

Write posture is per connector type, so the skill table carries a row for
`control_system.connector.<type>.writes_enabled` as well. Like the probe
channel, that key is per-type and so has no exact-match note in the tool — the
table is the only place an operator learns it exists.
"""

import pytest

from osprey.mcp_server.workspace.tools.setup import (
    _COLD_CHANGE_NOTES,
    _HOT_CHANGE_PATHS,
    _classify_change,
)
from tests.mcp_server.test_setup_patch_classification import SKILL_TEMPLATE

CONFIG = "config.yml"

#: The run-time switch. Named once here so a rename is a one-line edit.
SWITCH_TOOL = "control_target_set"

CONTROL_SYSTEM_TYPE = "control_system.type"

#: Keys added for the target switch that the tool classifies by exact path.
TARGET_SWITCH_KEYS = [
    "control_system.target_switch.drain_timeout_s",
    "control_system.target_switch.probe_interval_s",
    "control_system.target_switch.live_gateway_acknowledged",
]

#: The probe channel lives under `control_system.connector.<target>`, so its
#: path is per-target and cannot be an exact-match table entry. It is classified
#: the way every other `connector.*` key is: unlisted, generic cold note.
PROBE_CHANNEL_KEYS = [
    "control_system.connector.virtual_accelerator.probe_channel",
    "control_system.connector.epics.probe_channel",
]

#: Write posture is per connector type, so it has the same shape as the probe
#: channel above: one key per type, no exact-match note, carried by the skill
#: table alone. An operator who reads only the deployment-wide row would not
#: learn that the key exists.
PER_TYPE_WRITES_KEYS = [
    "control_system.connector.virtual_accelerator.writes_enabled",
    "control_system.connector.epics.writes_enabled",
]


def _skill_rows(key: str) -> list[str]:
    """Rows of the skill's hot/cold table that classify ``key``.

    Same matching rule as the existing classification test: a table row starts
    with the back-ticked key path, so a row for a different key never matches.
    """
    text = SKILL_TEMPLATE.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.startswith(f"| `{key}` |")]


@pytest.mark.parametrize("key_path", TARGET_SWITCH_KEYS)
def test_target_switch_keys_are_cold_not_hot(key_path):
    """Server-read keys are cold — the config is cached for the process."""
    assert key_path not in _HOT_CHANGE_PATHS.get(CONFIG, set())
    assert key_path in _COLD_CHANGE_NOTES[CONFIG]
    note = _classify_change(CONFIG, key_path)
    assert note.startswith("cold")
    assert "restart" in note.lower()


@pytest.mark.parametrize("key_path", TARGET_SWITCH_KEYS)
def test_target_switch_notes_say_what_reads_the_key(key_path):
    """A cold note that only says "restart" does not explain what is stale."""
    note = _classify_change(CONFIG, key_path)
    assert "read" in note.lower(), f"the note must say who reads the key: {note}"


def test_ack_note_frames_the_key_as_a_deliberate_operator_statement():
    """The acknowledgment is a claim about the real machine, not a tuning knob."""
    note = _classify_change(CONFIG, "control_system.target_switch.live_gateway_acknowledged")
    assert "operator" in note.lower()
    assert SWITCH_TOOL in note


@pytest.mark.parametrize("key_path", PROBE_CHANNEL_KEYS)
def test_probe_channel_keeps_the_generic_connector_classification(key_path):
    """`probe_channel` is a `connector.*` key: unlisted, generic cold note."""
    assert key_path not in _HOT_CHANGE_PATHS.get(CONFIG, set())
    assert key_path not in _COLD_CHANGE_NOTES[CONFIG]
    note = _classify_change(CONFIG, key_path)
    assert note.startswith("cold")
    assert "mcp server" in note.lower()


@pytest.mark.parametrize("key_path", PER_TYPE_WRITES_KEYS)
def test_per_type_write_posture_keeps_the_generic_connector_classification(key_path):
    """Per-type `writes_enabled` is a `connector.*` key: unlisted, generic note."""
    # Arrange / Act
    note = _classify_change(CONFIG, key_path)

    # Assert
    assert key_path not in _HOT_CHANGE_PATHS.get(CONFIG, set())
    assert key_path not in _COLD_CHANGE_NOTES[CONFIG]
    assert note.startswith("cold")


def test_skill_table_documents_the_per_type_write_posture():
    """The table must name the key that answers instead of the flat one.

    A per-type block is what makes a deployment armed on one machine and not
    the other, so a table that lists only `control_system.writes_enabled`
    describes a posture the deployment may not have.
    """
    # Arrange / Act
    rows = _skill_rows("control_system.connector.<type>.writes_enabled")

    # Assert
    assert len(rows) == 1, f"expected one per-type writes_enabled row, got {rows}"
    assert "| Cold |" in rows[0]
    assert "| Hot |" not in rows[0]


def test_control_system_type_stays_cold_and_points_at_the_run_time_switch():
    """Config sets the starting target; the switch is a tool call, not a rebuild."""
    assert CONTROL_SYSTEM_TYPE not in _HOT_CHANGE_PATHS.get(CONFIG, set())
    note = _classify_change(CONFIG, CONTROL_SYSTEM_TYPE)
    assert note.startswith("cold")
    assert "restart" in note.lower()
    assert SWITCH_TOOL in note, f"the note must name the run-time switch: {note}"


def test_skill_table_documents_the_run_time_switch():
    """The skill must not send an operator to a rebuild to change target."""
    text = SKILL_TEMPLATE.read_text(encoding="utf-8")
    assert SWITCH_TOOL in text

    rows = _skill_rows(CONTROL_SYSTEM_TYPE)
    assert len(rows) == 1, f"expected one `{CONTROL_SYSTEM_TYPE}` row, got {rows}"
    assert SWITCH_TOOL in rows[0]


def test_skill_table_documents_the_probe_channel_key():
    """The per-target probe channel has no exact-match note, so the table carries it."""
    rows = _skill_rows("control_system.connector.<target>.probe_channel")
    assert len(rows) == 1, f"expected one probe_channel row, got {rows}"
    assert "| Cold |" in rows[0]


@pytest.mark.parametrize("key_path", sorted(_HOT_CHANGE_PATHS[CONFIG]))
def test_every_hot_key_is_tabulated_hot_in_the_skill(key_path):
    """Lockstep: a key the tool calls hot must read Hot in the skill table."""
    rows = _skill_rows(key_path)
    assert len(rows) == 1, f"expected one skill row for {key_path}, got {rows}"
    assert "| Hot |" in rows[0]
    assert "| Cold |" not in rows[0]


@pytest.mark.parametrize("key_path", sorted(_COLD_CHANGE_NOTES[CONFIG]))
def test_every_noted_cold_key_is_tabulated_cold_in_the_skill(key_path):
    """Lockstep: a key with a bespoke cold note must read Cold in the skill table.

    Parametrizing over the table itself is the point — the next key someone adds
    to `_COLD_CHANGE_NOTES` is covered without anyone remembering to add a case.
    """
    rows = _skill_rows(key_path)
    assert len(rows) == 1, f"expected one skill row for {key_path}, got {rows}"
    assert "| Cold |" in rows[0]
    assert "| Hot |" not in rows[0]
