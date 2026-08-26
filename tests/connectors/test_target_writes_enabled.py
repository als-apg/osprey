"""Write posture, per connector type — the tri-state and what it refuses to do.

``control_system.writes_enabled`` used to be the whole answer: one flag for the
deployment, so a facility that wanted writes against its simulator had to arm
them everywhere. The posture is now per connector type, and the shape of the
per-type key is the part worth pinning: absent inherits the deployment-wide
flag, so a config that says nothing per type behaves exactly as it did before;
literally ``true`` arms that type; anything else leaves it unarmed *without*
inheriting, so a ``false`` written under a live block cannot be overridden by a
global ``true`` meant for the simulator.

The two functions are one answer reached from two identities — a connector holds
a TYPE, the roster and the hooks hold a TARGET — and the tests below state the
truth table for both rather than deriving one from the other.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey_connectors.types import (
    EPICS,
    MOCK,
    TARGET_LIVE,
    TARGET_VA,
    TYPE_WRITES_ENABLED_LEAF,
    VIRTUAL_ACCELERATOR,
    WRITES_ENABLED_KEY,
    any_target_writes_enabled,
    session_posture,
    target_writes_enabled,
    target_writes_enabled_key,
    type_writes_enabled,
    writes_enabled_key,
)

CUSTOM_TYPE = "mypackage.TangoConnector"


def _section(
    control_system_type: Any = ...,
    writes_enabled: Any = ...,
    connector: Any = ...,
) -> dict[str, Any]:
    """A ``control_system:`` section as the rendered config.yml carries it."""
    section: dict[str, Any] = {}
    if control_system_type is not ...:
        section["type"] = control_system_type
    if writes_enabled is not ...:
        section["writes_enabled"] = writes_enabled
    if connector is not ...:
        section["connector"] = connector
    return section


# ---------------------------------------------------------------------------
# The key names
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_posture_keys_are_spelled_the_way_the_config_spells_them():
    assert WRITES_ENABLED_KEY == "control_system.writes_enabled"
    assert TYPE_WRITES_ENABLED_LEAF == "writes_enabled"


@pytest.mark.unit
def test_a_type_names_its_own_block_key():
    """Every refusal in the framework spells the key through this one function."""
    assert writes_enabled_key(EPICS) == "control_system.connector.epics.writes_enabled"
    assert (
        writes_enabled_key(CUSTOM_TYPE)
        == "control_system.connector.mypackage.TangoConnector.writes_enabled"
    )


@pytest.mark.unit
def test_no_type_names_the_deployment_wide_key():
    """A caller holding no type has no block to name, and that key answered it."""
    assert writes_enabled_key(None) == WRITES_ENABLED_KEY
    assert writes_enabled_key("") == WRITES_ENABLED_KEY


@pytest.mark.unit
def test_a_targets_key_is_the_block_its_posture_was_read_from():
    """The key a refusal names must be the key that decided the refusal."""
    # Arrange
    section = _section(
        EPICS,
        writes_enabled=False,
        connector={"epics": {"port": 5064}, "virtual_accelerator": {"writes_enabled": True}},
    )

    # Act / Assert
    assert (
        target_writes_enabled_key(section, TARGET_VA)
        == "control_system.connector.virtual_accelerator.writes_enabled"
    )
    assert (
        target_writes_enabled_key(section, TARGET_LIVE)
        == "control_system.connector.epics.writes_enabled"
    )


@pytest.mark.unit
def test_an_unresolvable_target_names_the_key_it_inherits_from():
    """`live` on a deployment that never described its real machine, and a target
    that names nothing at all: both read the deployment-wide key, so both name it."""
    # Arrange
    section = _section(MOCK, writes_enabled=True)

    # Act / Assert
    assert target_writes_enabled_key(section, TARGET_LIVE) == WRITES_ENABLED_KEY
    assert target_writes_enabled_key(section, "not-a-target") == WRITES_ENABLED_KEY


# ---------------------------------------------------------------------------
# type_writes_enabled — the tri-state
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_per_type_true_arms_that_type_over_a_global_false():
    """The point of the feature: arm the simulator on a deployment that is off."""
    # Arrange
    section = _section(
        EPICS,
        writes_enabled=False,
        connector={"virtual_accelerator": {"writes_enabled": True}},
    )

    # Act / Assert
    assert type_writes_enabled(section, VIRTUAL_ACCELERATOR) is True
    assert type_writes_enabled(section, EPICS) is False


@pytest.mark.unit
def test_a_per_type_false_disarms_that_type_over_a_global_true():
    """An explicit per-type value is the answer; it never falls back."""
    # Arrange
    section = _section(
        EPICS,
        writes_enabled=True,
        connector={"epics": {"writes_enabled": False}},
    )

    # Act / Assert
    assert type_writes_enabled(section, EPICS) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [False, None, "true", "True", "false", "", 0, 1, [], {}],
    ids=[
        "false",
        "bare-key-none",
        "string-true",
        "string-True",
        "string-false",
        "empty-string",
        "zero",
        "one",
        "empty-list",
        "empty-mapping",
    ],
)
def test_any_present_value_that_is_not_the_bool_true_is_unarmed_and_hard(value: Any):
    """Not armed, and not inherited either — the global ``true`` cannot save it."""
    # Arrange
    section = _section(
        EPICS,
        writes_enabled=True,
        connector={"epics": {"writes_enabled": value}},
    )

    # Act / Assert
    assert type_writes_enabled(section, EPICS) is False


@pytest.mark.unit
@pytest.mark.parametrize("global_value", [True, False], ids=["global-true", "global-false"])
@pytest.mark.parametrize(
    "connector",
    [
        {"epics": {"timeout": 5.0}},
        {"virtual_accelerator": {"writes_enabled": True}},
        {"epics": None},
        {"epics": "yes"},
        {},
        None,
        "not-a-mapping",
        ...,
    ],
    ids=[
        "block-without-the-leaf",
        "block-for-another-type-only",
        "block-is-none",
        "block-is-not-a-mapping",
        "empty-connector-table",
        "connector-is-none",
        "connector-is-not-a-mapping",
        "no-connector-table",
    ],
)
def test_an_absent_per_type_key_inherits_the_deployment_wide_key(
    connector: Any, global_value: bool
):
    """Every shape of "this type has said nothing" reads the global key."""
    # Arrange
    section = _section(EPICS, writes_enabled=global_value, connector=connector)

    # Act / Assert
    assert type_writes_enabled(section, EPICS) is global_value


@pytest.mark.unit
@pytest.mark.parametrize(
    "global_value",
    [False, None, "true", 1, ...],
    ids=["false", "bare-key-none", "string-true", "one", "absent"],
)
def test_the_inherited_deployment_wide_key_is_itself_true_or_nothing(global_value: Any):
    """Inheriting reads the global key with the same strictness it always had."""
    # Arrange
    section = _section(EPICS, writes_enabled=global_value, connector={"epics": {}})

    # Act / Assert
    assert type_writes_enabled(section, EPICS) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "section", [None, "not-a-mapping", 0, []], ids=["none", "string", "zero", "list"]
)
def test_a_section_that_is_not_a_mapping_is_not_armed(section: Any):
    # Act / Assert
    assert type_writes_enabled(section, EPICS) is False


@pytest.mark.unit
def test_a_config_with_no_posture_anywhere_is_unarmed_for_every_type():
    """No key written at all is the shipped default, and it is off."""
    # Arrange
    section = _section(EPICS, connector={"epics": {"timeout": 5.0}, "virtual_accelerator": {}})

    # Act / Assert
    assert type_writes_enabled(section, EPICS) is False
    assert type_writes_enabled(section, VIRTUAL_ACCELERATOR) is False
    assert type_writes_enabled(section, MOCK) is False
    assert target_writes_enabled(section, TARGET_LIVE) is False
    assert target_writes_enabled(section, TARGET_VA) is False


@pytest.mark.unit
def test_a_dotted_custom_type_is_one_key_and_not_a_path():
    """``mypackage.TangoConnector`` names one block; the dots are part of it."""
    # Arrange
    section = _section(
        CUSTOM_TYPE,
        writes_enabled=False,
        connector={
            CUSTOM_TYPE: {"writes_enabled": True},
            "mypackage": {"TangoConnector": {"writes_enabled": False}},
        },
    )

    # Act / Assert
    assert type_writes_enabled(section, CUSTOM_TYPE) is True


@pytest.mark.unit
def test_a_dotted_custom_type_with_no_block_of_its_own_inherits():
    """The nested lookalike is a different key and contributes nothing."""
    # Arrange
    section = _section(
        CUSTOM_TYPE,
        writes_enabled=True,
        connector={"mypackage": {"TangoConnector": {"writes_enabled": False}}},
    )

    # Act / Assert
    assert type_writes_enabled(section, CUSTOM_TYPE) is True


@pytest.mark.unit
def test_the_posture_does_not_read_the_environment(monkeypatch: pytest.MonkeyPatch):
    """A read-only run is the caller's AND, not this resolver's business.

    The resolver reports what the config describes so that a lint, a persona and
    a connector agree about it; the process-level refusal is applied on top by
    whoever is about to write.
    """
    # Arrange
    section = _section(EPICS, connector={"epics": {"writes_enabled": True}})
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    # Act / Assert
    assert type_writes_enabled(section, EPICS) is True
    assert target_writes_enabled(section, TARGET_LIVE) is True


@pytest.mark.unit
def test_asking_about_the_posture_does_not_mutate_the_section():
    # Arrange
    section = _section(EPICS, writes_enabled=False, connector={"epics": {"writes_enabled": True}})
    before = {
        "type": EPICS,
        "writes_enabled": False,
        "connector": {"epics": {"writes_enabled": True}},
    }

    # Act
    type_writes_enabled(section, VIRTUAL_ACCELERATOR)
    target_writes_enabled(section, TARGET_VA)

    # Assert
    assert section == before


# ---------------------------------------------------------------------------
# target_writes_enabled — the same answer, reached from a session target
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_va_reads_the_virtual_accelerator_block_and_live_reads_the_epics_one():
    # Arrange
    section = _section(
        EPICS,
        writes_enabled=False,
        connector={
            "epics": {"writes_enabled": False},
            "virtual_accelerator": {"writes_enabled": True},
        },
    )

    # Act / Assert
    assert target_writes_enabled(section, TARGET_VA) is True
    assert target_writes_enabled(section, TARGET_LIVE) is False


@pytest.mark.unit
def test_live_on_an_epics_baseline_reads_the_epics_block():
    # Arrange
    section = _section(
        EPICS,
        writes_enabled=False,
        connector={"epics": {"writes_enabled": True}},
    )

    # Act / Assert
    assert target_writes_enabled(section, TARGET_LIVE) is True


@pytest.mark.unit
def test_a_va_baseline_arms_its_own_target_without_arming_live():
    """The shape a VA deployment ships in: the simulator armed, hardware not."""
    # Arrange
    section = _section(
        VIRTUAL_ACCELERATOR,
        connector={
            "virtual_accelerator": {"writes_enabled": True},
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
        },
    )

    # Act / Assert
    assert target_writes_enabled(section, TARGET_VA) is True
    assert target_writes_enabled(section, TARGET_LIVE) is False


@pytest.mark.unit
def test_a_va_baseline_with_no_live_block_still_answers_live_from_the_global_key():
    """``live`` is underivable here, so the deployment-wide key is the answer."""
    # Arrange
    section = _section(
        VIRTUAL_ACCELERATOR,
        writes_enabled=True,
        connector={"virtual_accelerator": {"writes_enabled": False}},
    )

    # Act / Assert
    assert target_writes_enabled(section, TARGET_VA) is False
    assert target_writes_enabled(section, TARGET_LIVE) is True


@pytest.mark.unit
@pytest.mark.parametrize("global_value", [True, False], ids=["global-true", "global-false"])
def test_live_on_a_mock_deployment_answers_the_deployment_wide_key(global_value: bool):
    """Parity: a mock deployment never had a second target, so it keeps the flag."""
    # Arrange
    section = _section(MOCK, writes_enabled=global_value, connector={"mock": {"noise_level": 0.0}})

    # Act / Assert
    assert target_writes_enabled(section, TARGET_LIVE) is global_value


@pytest.mark.unit
@pytest.mark.parametrize("global_value", [True, False], ids=["global-true", "global-false"])
@pytest.mark.parametrize(
    "target",
    [None, "", "LIVE", "Va", "epics", "virtual_accelerator", 0],
    ids=["none", "blank", "wrong-case-live", "wrong-case-va", "a-type", "another-type", "zero"],
)
def test_an_unknown_target_answers_the_deployment_wide_key(target: Any, global_value: bool):
    """No type means no per-type block to consult; it does not mean armed."""
    # Arrange
    section = _section(
        EPICS,
        writes_enabled=global_value,
        connector={"epics": {"writes_enabled": not global_value}},
    )

    # Act / Assert
    assert target_writes_enabled(section, target) is global_value


@pytest.mark.unit
@pytest.mark.parametrize("target", [TARGET_LIVE, TARGET_VA], ids=["live", "va"])
def test_a_section_that_is_not_a_mapping_is_not_armed_for_any_target(target: str):
    # Act / Assert
    assert target_writes_enabled(None, target) is False
    assert target_writes_enabled("not-a-mapping", target) is False


# ---------------------------------------------------------------------------
# Speaking about the deployment's targets without holding one
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_posture_names_both_targets_only_where_the_switch_renders():
    """Without the switch a session has one target: the deployment baseline."""
    switchable = _section(
        EPICS,
        writes_enabled=False,
        connector={
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"writes_enabled": True},
        },
    )
    assert session_posture(switchable) == {TARGET_LIVE: False, TARGET_VA: True}
    assert session_posture(_section(VIRTUAL_ACCELERATOR)) == {TARGET_VA: False}
    assert session_posture(_section(MOCK, writes_enabled=True)) == {TARGET_LIVE: True}
    assert session_posture("not a mapping") == {TARGET_LIVE: False}


@pytest.mark.unit
def test_a_non_switchable_baseline_answers_the_built_type_not_the_live_derivation():
    """A mock deployment with a stray armed epics block builds a mock connector."""
    section = _section(MOCK, writes_enabled=False, connector={"epics": {"writes_enabled": True}})
    assert target_writes_enabled(section, TARGET_LIVE) is True
    assert session_posture(section) == {TARGET_LIVE: False}
    assert any_target_writes_enabled(section) is False


@pytest.mark.unit
def test_the_union_does_not_let_a_phantom_live_inherit_the_global_key():
    """Every real lane says ``false``; a global ``true`` must not arm the union."""
    # Arrange
    section = _section(
        VIRTUAL_ACCELERATOR,
        writes_enabled=True,
        connector={"virtual_accelerator": {"writes_enabled": False}},
    )

    # Act / Assert
    assert target_writes_enabled(section, TARGET_LIVE) is True
    assert any_target_writes_enabled(section) is False


@pytest.mark.unit
@pytest.mark.parametrize("global_value", [True, False])
def test_the_union_keeps_single_flag_parity_where_nothing_is_said_per_type(global_value: bool):
    """A deployment with only the deployment-wide key answers that key."""
    assert any_target_writes_enabled(_section(MOCK, writes_enabled=global_value)) is global_value
    assert any_target_writes_enabled(_section(EPICS, writes_enabled=global_value)) is global_value


@pytest.mark.unit
def test_the_union_is_true_when_one_reachable_target_is_armed():
    section = _section(
        EPICS,
        writes_enabled=False,
        connector={
            "virtual_accelerator": {"writes_enabled": True},
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
        },
    )
    assert any_target_writes_enabled(section) is True
    assert any_target_writes_enabled(_section()) is False
