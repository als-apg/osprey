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

import json
from typing import Any

import pytest

from osprey_connectors import session_store
from osprey_connectors.control_system import base as connector_base
from osprey_connectors.control_system.base import ChannelWriteResult, WriteOutcome
from osprey_connectors.factory import ConnectorFactory, isolated_connector_registries
from osprey_connectors.types import (
    EPICS,
    LIVE_STANDIN,
    MOCK,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    TYPE_WRITES_ENABLED_LEAF,
    VIRTUAL_ACCELERATOR,
    WRITES_ENABLED_KEY,
    any_target_writes_enabled,
    configured_targets,
    session_posture,
    switch_capable,
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
def test_a_deployment_with_no_standin_block_gets_no_standin_posture():
    """The vocabulary grew a third target; this deployment did not.

    The posture is published to a permissions render, a roster and a lint. A
    ``standin`` key here would be a machine nobody stood up, and armed or not
    from the deployment-wide flag rather than from anything about a stand-in.
    """
    # Arrange
    section = _section(
        EPICS,
        writes_enabled=True,
        connector={
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"writes_enabled": False},
        },
    )

    # Act
    posture = session_posture(section)

    # Assert
    assert TARGET_STANDIN not in posture
    assert posture == {TARGET_LIVE: True, TARGET_VA: False}


@pytest.mark.unit
def test_a_standin_baseline_is_switch_capable_and_does_not_raise():
    """The baseline is resolved, not looked up among ``live`` and ``va``.

    A ``live_standin`` deployment is baselined on a third target, and asking
    whether it can switch must answer rather than raise — every caller of
    :func:`session_posture` is downstream of this predicate.
    """
    # Arrange
    section = _section(
        LIVE_STANDIN,
        connector={
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"writes_enabled": True},
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert switch_capable(section) is True


@pytest.mark.unit
def test_a_standin_beside_a_simulator_is_switch_capable_without_a_live_block():
    """Two configured targets are the switching world, whichever two they are.

    A facility rehearsing on its stand-in beside the simulator, with no live
    machine authored yet, has exactly the multi-target world the switch — and
    the popover's per-target posture toggles behind it — exist for. ``live``
    stays underivable and simply is not on the roster.
    """
    # Arrange
    section = _section(
        LIVE_STANDIN,
        connector={
            "virtual_accelerator": {"writes_enabled": True},
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert switch_capable(section) is True
    assert configured_targets(section) == [TARGET_VA, TARGET_STANDIN]


@pytest.mark.unit
def test_a_va_baseline_beside_a_standin_is_switch_capable():
    """The same two-machine world, baselined on the simulator."""
    # Arrange
    section = _section(
        VIRTUAL_ACCELERATOR,
        connector={
            "virtual_accelerator": {"writes_enabled": True},
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert switch_capable(section) is True
    assert configured_targets(section) == [TARGET_VA, TARGET_STANDIN]


@pytest.mark.unit
@pytest.mark.parametrize(
    "section",
    [
        pytest.param(_section(EPICS, connector={"epics": {"gateways": {}}}), id="live-only"),
        pytest.param(
            _section(VIRTUAL_ACCELERATOR, connector={"virtual_accelerator": {"port": 5064}}),
            id="va-only",
        ),
        pytest.param(
            _section(LIVE_STANDIN, connector={LIVE_STANDIN: {"port": 5074}}), id="standin-only"
        ),
        pytest.param(_section(MOCK), id="bare-mock"),
    ],
)
def test_a_single_target_render_is_not_switch_capable(section):
    """One configured target is nowhere to switch to."""
    assert switch_capable(section) is False


@pytest.mark.unit
def test_the_live_and_va_pair_is_still_switch_capable():
    """The original two-target shape answers as it always did."""
    # Arrange
    section = _section(
        EPICS,
        connector={
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"port": 5064},
        },
    )

    # Act / Assert
    assert switch_capable(section) is True


@pytest.mark.unit
def test_a_mock_carrying_other_blocks_is_still_not_switch_capable():
    """The baseline-consistency guard survives the target count.

    A ``mock`` deployment that happens to carry an ``epics`` and a
    ``virtual_accelerator`` block enumerates two targets, but its baseline
    resolves to a machine its own type never selected — treating it as
    switchable would point a session at a real machine on the strength of a
    stray block.
    """
    # Arrange
    section = _section(
        MOCK,
        connector={
            "mock": {},
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"port": 5064},
        },
    )

    # Act / Assert
    assert switch_capable(section) is False


@pytest.mark.unit
def test_a_standin_baseline_posture_names_three_targets_in_vocabulary_order():
    """The baseline is among them, in the constant's order rather than first.

    The order this dict is built in is the order the rendered ``settings.json``
    lists the targets in, so it is the vocabulary's and never the baseline's:
    a deployment that gained no target must gain no reordering either.
    """
    # Arrange
    section = _section(
        LIVE_STANDIN,
        writes_enabled=False,
        connector={
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"writes_enabled": True},
            LIVE_STANDIN: {"writes_enabled": True},
        },
    )

    # Act
    posture = session_posture(section)

    # Assert
    assert list(posture) == [TARGET_LIVE, TARGET_VA, TARGET_STANDIN]
    assert posture == {TARGET_LIVE: False, TARGET_VA: True, TARGET_STANDIN: True}


@pytest.mark.unit
def test_a_deployment_that_configured_all_three_gets_all_three():
    """The stand-in is a machine of its own, with a posture of its own."""
    # Arrange
    section = _section(
        EPICS,
        writes_enabled=False,
        connector={
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"writes_enabled": True},
            LIVE_STANDIN: {"writes_enabled": True},
        },
    )

    # Act / Assert
    assert session_posture(section) == {
        TARGET_LIVE: False,
        TARGET_VA: True,
        TARGET_STANDIN: True,
    }


# ---------------------------------------------------------------------------
# configured_targets — what this deployment has, not what the vocabulary knows
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_configured_targets_are_the_baseline_and_every_block_behind_one():
    # Arrange
    section = _section(
        EPICS,
        connector={
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"writes_enabled": True},
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert configured_targets(section) == [TARGET_LIVE, TARGET_VA, TARGET_STANDIN]


@pytest.mark.unit
def test_a_standin_baseline_keeps_the_vocabulary_order():
    """Its own target is in the list, where :data:`CONTROL_TARGETS` puts it."""
    # Arrange
    section = _section(
        LIVE_STANDIN,
        connector={
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"writes_enabled": True},
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert configured_targets(section) == [TARGET_LIVE, TARGET_VA, TARGET_STANDIN]


@pytest.mark.unit
def test_a_va_baseline_enumerates_exactly_as_it_did_before_the_third_target():
    """The shape SC-5 pins: no stand-in block, so nothing about it may change."""
    # Arrange
    section = _section(
        VIRTUAL_ACCELERATOR,
        connector={
            "epics": {"gateways": {"read_only": {"address": "gw"}}},
            "virtual_accelerator": {"writes_enabled": True},
        },
    )

    # Act / Assert
    assert configured_targets(section) == [TARGET_LIVE, TARGET_VA]


@pytest.mark.unit
@pytest.mark.parametrize(
    "standin_block",
    [{}, None, "not-a-mapping", 0, [], ...],
    ids=["empty-mapping", "none", "not-a-mapping", "zero", "empty-list", "no-block"],
)
def test_a_target_without_a_usable_block_is_not_configured(standin_block: Any):
    """That block is what a connector is configured from; an empty one is nothing."""
    # Arrange
    connector: dict[str, Any] = {"epics": {"gateways": {"read_only": {"address": "gw"}}}}
    if standin_block is not ...:
        connector[LIVE_STANDIN] = standin_block
    section = _section(EPICS, connector=connector)

    # Act / Assert
    assert configured_targets(section) == [TARGET_LIVE]


@pytest.mark.unit
def test_a_live_that_does_not_resolve_is_not_a_configured_target():
    """``resolve_target`` refuses to guess a real machine, and a refusal is no slot."""
    # Arrange
    section = _section(
        VIRTUAL_ACCELERATOR,
        writes_enabled=True,
        connector={"virtual_accelerator": {"writes_enabled": False}},
    )

    # Act / Assert
    assert configured_targets(section) == [TARGET_VA]


@pytest.mark.unit
def test_the_baseline_is_configured_even_with_no_block_of_its_own():
    """A session is on the connector ``control_system.type`` builds regardless."""
    # Act / Assert
    assert configured_targets(_section(MOCK)) == [TARGET_LIVE]
    assert configured_targets(_section(MOCK, connector={})) == [TARGET_LIVE]
    assert configured_targets(_section(VIRTUAL_ACCELERATOR)) == [TARGET_VA]
    assert configured_targets(_section(LIVE_STANDIN)) == [TARGET_STANDIN]


@pytest.mark.unit
@pytest.mark.parametrize(
    "section", [None, "not-a-mapping", 0, [], {}], ids=["none", "string", "zero", "list", "empty"]
)
def test_a_section_that_is_not_a_mapping_still_has_its_baseline(section: Any):
    """Never raises and never empty: the one target such a deployment is on."""
    # Act / Assert
    assert configured_targets(section) == [TARGET_LIVE]


@pytest.mark.unit
def test_asking_which_targets_are_configured_does_not_mutate_the_section():
    # Arrange
    section = _section(EPICS, connector={"epics": {"port": 5064}})
    before = {"type": EPICS, "connector": {"epics": {"port": 5064}}}

    # Act
    configured_targets(section)

    # Assert
    assert section == before


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


# ---------------------------------------------------------------------------
# The connector's reference monitor
# ---------------------------------------------------------------------------
#
# Everything above is the deployment's own posture: what a config says, read
# from a config. The monitor ANDs two live terms with it — the readonly run and
# the operator's per-(session, target) narrowing — and the second is the reason
# this section exists. A target flipped to read-only from the control-target
# chip has to refuse the very next write on a session that is already running,
# so the store is read on every call rather than cached with the config.
#
# The tests drive ``connector.write_channel`` rather than the ``_writes_enabled``
# property, because the property is only half of what an operator meets: the
# other half is the refusal message, which forks four ways and is the only thing
# telling them which of the four refused and where to lift it.

SESSION_KEY = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

#: A switch-capable deployment with live (epics) and the VA both armed, and the
#: stand-in inheriting the deployment-wide ``false``.
ARMED_SECTION: dict[str, Any] = {
    "type": EPICS,
    "writes_enabled": False,
    "connector": {
        "epics": {"gateways": {"read_only": {"address": "gw"}}, "writes_enabled": True},
        "virtual_accelerator": {"host": "localhost", "writes_enabled": True},
        "live_standin": {"prefix": "S:", "writes_enabled": True},
    },
}


class _FakeConnector(connector_base.ControlSystemConnector):
    """A connector that records the writes the monitor let through.

    The abstract signature verbatim (``confirm`` included): a stand-in whose
    signature has drifted from the base class is wrapped by the same guard but
    exercises a call shape no real connector has.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, Any]] = []

    async def connect(self, config: dict[str, Any]) -> None: ...
    async def disconnect(self) -> None: ...

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        raise NotImplementedError

    async def read_multiple_channels(self, channel_addresses, timeout=None):
        raise NotImplementedError

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        self.writes.append((channel_address, value))
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED,
        )

    async def write_multiple_channels(self, operations, timeout=None):
        return [await self.write_channel(addr, val) for addr, val in operations]

    async def subscribe(self, channel_address, callback):
        raise NotImplementedError

    async def unsubscribe(self, channel_address):
        raise NotImplementedError

    async def get_metadata(self, channel_address):
        raise NotImplementedError

    async def validate_channel(self, channel_address) -> bool:
        return True


def _built(connector_type: str | None, control_target: str | None) -> _FakeConnector:
    """A connector stamped the way ``ConnectorFactory`` stamps one."""
    connector = _FakeConnector()
    connector._connector_type = connector_type
    connector._control_target = control_target
    return connector


@pytest.fixture
def deployment(monkeypatch):
    """Install a ``control_system:`` section as the connector's config reader.

    Both spellings the deployment half uses are answered from the one section —
    the dotted deployment-wide key an unstamped connector reads, and the whole
    section a stamped one keys its type on — so a test states the config once.
    """

    def _install(section: dict[str, Any]) -> None:
        def _get_config_value(key: str, default: Any = None) -> Any:
            if key == "control_system":
                return section
            if key == WRITES_ENABLED_KEY:
                return section.get("writes_enabled", default)
            return default

        monkeypatch.setattr("osprey_connectors.config.get_config_value", _get_config_value)

    return _install


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A scratch posture store, and the two stamps that make it readable.

    The session key starts UNSET even though the root is stamped: a store the
    session key does not address is a store nobody reads, and several tests
    below are exactly that case. ``.addressed()`` stamps the key.
    """
    root = tmp_path / "agent_data"
    directory = root / session_store.STATE_DIR_NAME
    directory.mkdir(parents=True)
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv(connector_base.POSTURE_SESSION_ENV_VAR, raising=False)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    # The executor's run-level pin, cleared like the mode: it is ANDed into
    # every store answer, so a stamp inherited from the environment this suite
    # runs in would refuse writes in tests that are about the store alone.
    monkeypatch.delenv(session_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    session_store.invalidate_cache()

    class _Store:
        key = SESSION_KEY
        path = directory / session_store.STORE_FILENAME

        @staticmethod
        def addressed() -> None:
            """Stamp the session key, so the store is this session's to read."""
            monkeypatch.setenv(connector_base.POSTURE_SESSION_ENV_VAR, SESSION_KEY)

        @classmethod
        def narrow(cls, **targets: str) -> None:
            """Record ``{target: posture}`` for this session key."""
            cls.write({SESSION_KEY: dict(targets)})

        @classmethod
        def write(cls, payload: Any) -> None:
            cls.path.write_text(json.dumps(payload), encoding="utf-8")
            session_store.invalidate_cache()

    yield _Store
    session_store.invalidate_cache()


class TestTheConnectorReferenceMonitor:
    """``ceiling ∧ not is_readonly_run() ∧ (store entry ≠ sandbox)``, per write."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_narrowed_target_refuses_while_another_target_writes(self, deployment, store):
        """The point of a per-target posture: one machine, not the session.

        Both connectors are the same armed type on the same session. Only the
        target differs, and only the narrowed one is refused.
        """
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        narrowed = _built(EPICS, TARGET_STANDIN)
        untouched = _built(EPICS, TARGET_VA)

        # Act
        refused = await narrowed.write_channel("S:CORR:1:SP", 0.5)
        allowed = await untouched.write_channel("VA:CORR:1:SP", 0.5)

        # Assert
        assert refused.outcome is WriteOutcome.REFUSED
        assert refused.refusal_reason == "WRITES_DISABLED"
        assert narrowed.writes == []
        assert allowed.outcome is WriteOutcome.CONFIRMED
        assert untouched.writes == [("VA:CORR:1:SP", 0.5)]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_refusal_names_the_target_and_the_chip(self, deployment, store):
        """A refusal that names no way out is a dead end.

        Nothing is wrong with the deployment here — it arms this type — so the
        message must not send anyone to a config file, and it must name both
        the machine that was narrowed and the surface that narrowed it.
        """
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        message = result.error_message
        assert TARGET_STANDIN in message
        assert "writes are off" in message
        assert "control-target chip in the header" in message
        assert "writes_enabled" not in message
        assert "resubmit" not in message.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_narrowing_lands_on_a_connector_that_is_already_writing(
        self, deployment, store
    ):
        """The live half: no respawn, no reconnect, no config edit.

        The deployment posture is process-cached and could not do this. The
        store term is read on every call precisely so a flip reaches a session
        that is already mid-conversation.
        """
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        before = await connector.write_channel("S:CORR:1:SP", 0.5)
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        after = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert before.outcome is WriteOutcome.CONFIRMED
        assert after.outcome is WriteOutcome.REFUSED
        assert connector.writes == [("S:CORR:1:SP", 0.5)]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_every_write_of_a_batch_is_refused_with_the_same_story(self, deployment, store):
        """The multi-write guard forks the same four ways, per operation."""
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        results = await connector.write_multiple_channels([("S:A:SP", 1.0), ("S:B:SP", 2.0)])

        # Assert
        assert [r.outcome for r in results] == [WriteOutcome.REFUSED] * 2
        assert all("control-target chip in the header" in r.error_message for r in results)
        assert connector.writes == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_without_a_session_key_the_store_is_not_consulted(self, deployment, store):
        """Nothing addressed this session, so nothing narrowed it.

        The entry in the file belongs to a session key this process does not
        carry. A process that read it anyway would let one operator's narrowing
        refuse writes for a CLI run, a dispatch worker or another operator.
        """
        # Arrange
        deployment(ARMED_SECTION)
        store.narrow(standin=session_store.POSTURE_SANDBOX, va=session_store.POSTURE_SANDBOX)
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert result.outcome is WriteOutcome.CONFIRMED
        assert connector.writes == [("S:CORR:1:SP", 0.5)]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_an_unstamped_target_takes_the_most_restrictive_entry(self, deployment, store):
        """A connector that cannot say which machine it writes to gets the floor.

        Granting it the most permissive answer would let a narrowing be walked
        around by whichever build site forgot to name its target.
        """
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        connector = _built(EPICS, None)

        # Act / Assert — one narrowed entry anywhere is enough
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        assert (await connector.write_channel("A:SP", 1.0)).outcome is WriteOutcome.REFUSED

        # Act / Assert — and nothing narrowed leaves the deployment in charge
        store.write({SESSION_KEY: {}})
        assert (await connector.write_channel("A:SP", 1.0)).outcome is WriteOutcome.CONFIRMED

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_unstamped_refusal_says_it_could_not_name_a_target(self, deployment, store):
        """Naming a target it does not have would be a lie an operator acts on."""
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        connector = _built(EPICS, None)

        # Act
        message = (await connector.write_channel("A:SP", 1.0)).error_message

        # Assert
        assert "at least one control target" in message
        assert "control-target chip in the header" in message

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_store_can_only_narrow(self, deployment, store):
        """An unarmed deployment stays unarmed however the store is spelled.

        ``writes`` is dropped on parse rather than honoured, so the store holds
        narrowings and nothing else — but the connector must be safe even if a
        hand-edited file gets past that filter.
        """
        # Arrange
        deployment({"type": EPICS, "writes_enabled": False, "connector": {"epics": {}}})
        store.addressed()
        store.narrow(standin=session_store.POSTURE_WRITES, va=session_store.POSTURE_WRITES)
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert result.outcome is WriteOutcome.REFUSED
        assert writes_enabled_key(EPICS) in result.error_message

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_narrowing_the_deployment_already_refuses_names_the_deployment(
        self, deployment, store
    ):
        """Last resort: the chip is named only when flipping it would help.

        Sending an operator to a chip whose other position the deployment
        refuses anyway costs them a round trip and teaches them the message
        cannot be trusted.
        """
        # Arrange
        deployment({"type": EPICS, "writes_enabled": False, "connector": {"epics": {}}})
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        message = (await connector.write_channel("S:CORR:1:SP", 0.5)).error_message

        # Assert
        assert writes_enabled_key(EPICS) in message
        assert "chip" not in message

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_readonly_run_still_wins_the_wording(self, deployment, store, monkeypatch):
        """Two live terms can hold at once; the one the operator cannot lift
        from the chip is the one worth naming."""
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert result.outcome is WriteOutcome.REFUSED
        assert "readonly execution mode" in result.error_message


class TestTheMonitorAndTheStoreRuleAgree:
    """The connector restates the store clause; the two answers stay identical.

    ``session_store.effective_writes`` is the canonical spelling of
    ``ceiling ∧ not readonly ∧ store``, but the connector cannot call it for the
    ceiling: its deployment half is keyed on the connector TYPE, which is not
    the ceiling that function derives for a caller holding only a target. The
    store clause is therefore restated in ``base._session_store_permits``, and
    this table is what keeps the restatement honest.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.parametrize("target", [TARGET_LIVE, TARGET_VA, TARGET_STANDIN, None])
    @pytest.mark.parametrize(
        "narrowed",
        [
            {},
            {TARGET_STANDIN: session_store.POSTURE_SANDBOX},
            {TARGET_VA: session_store.POSTURE_SANDBOX},
            dict.fromkeys((TARGET_LIVE, TARGET_VA, TARGET_STANDIN), session_store.POSTURE_SANDBOX),
        ],
        ids=["nothing", "standin", "va", "everything"],
    )
    @pytest.mark.parametrize("addressed", [True, False], ids=["addressed", "unaddressed"])
    async def test_the_two_answers_match(self, deployment, store, target, narrowed, addressed):
        # Arrange
        deployment(ARMED_SECTION)
        store.write({SESSION_KEY: narrowed})
        if addressed:
            store.addressed()
        connector = _built(EPICS, target)

        # Act
        canonical = session_store.effective_writes(
            ARMED_SECTION,
            SESSION_KEY if addressed else None,
            target,
            connector_type=EPICS,
        )
        result = await connector.write_channel("A:SP", 1.0)

        # Assert
        assert (result.outcome is WriteOutcome.CONFIRMED) is canonical


class TestTheFactoryBuiltConnector:
    """The stamps the factory really applies, through the real factory path."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_target_unstamped_connector_still_reads_its_types_posture(
        self, deployment, store
    ):
        """The mixed config SC pins: deployment-wide ``true``, this type ``false``.

        The target stamp indexes the STORE and nothing else, so a build site
        that names no target loses none of the deployment half — a connector
        built without one is refused by its own type's block exactly as a
        stamped one would be, and the refusal names that block.
        """
        # Arrange
        section = {
            "type": EPICS,
            "writes_enabled": True,
            "connector": {"epics": {"writes_enabled": False}},
        }
        deployment(section)

        # Act
        with isolated_connector_registries(clear=True):
            ConnectorFactory.register_control_system(EPICS, _FakeConnector)
            connector = await ConnectorFactory.create_control_system_connector(section)
        result = await connector.write_channel("SR:CORR:1:SP", 0.5)

        # Assert
        assert connector._connector_type == EPICS
        assert connector._control_target is None
        assert result.outcome is WriteOutcome.REFUSED
        assert writes_enabled_key(EPICS) in result.error_message
        assert connector.writes == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_named_target_is_what_the_store_is_indexed_by(self, deployment, store):
        """End to end: the factory's stamp is the key the narrowing was filed
        under, so an operator's flip and a connector's refusal meet."""
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)

        # Act
        with isolated_connector_registries(clear=True):
            ConnectorFactory.register_control_system(EPICS, _FakeConnector)
            connector = await ConnectorFactory.create_control_system_connector(
                ARMED_SECTION, control_target=TARGET_STANDIN
            )
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert connector._control_target == TARGET_STANDIN
        assert result.outcome is WriteOutcome.REFUSED
        assert "control-target chip in the header" in result.error_message


class TestOneStoreReadPerWrite:
    """A write asks the store once, and the refusal reuses that answer.

    The monitor decides on the store, and the refusal it builds names one of
    four causes on the strength of the same fact. Asking twice would let a
    narrowing lifted in between produce a refusal that blames the chip for a
    write the deployment turned down — a message that sends the operator to
    flip something that was not the gate.
    """

    @staticmethod
    def _counted(monkeypatch) -> list[tuple[str | None, str | None]]:
        calls: list[tuple[str | None, str | None]] = []
        real = session_store.store_permits

        def _counting(session_key, target):
            calls.append((session_key, target))
            return real(session_key, target)

        monkeypatch.setattr(session_store, "store_permits", _counting)
        return calls

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_refused_write_reads_the_store_once(self, deployment, store, monkeypatch):
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        calls = self._counted(monkeypatch)
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert — refused, named the chip, and asked once
        assert result.outcome is WriteOutcome.REFUSED
        assert "control-target chip in the header" in result.error_message
        assert calls == [(SESSION_KEY, TARGET_STANDIN)]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_refused_batch_reads_the_store_once_for_the_whole_batch(
        self, deployment, store, monkeypatch
    ):
        """Not once per operation: one verdict refused them all, and every
        result has to tell the same story about why."""
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        calls = self._counted(monkeypatch)
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        results = await connector.write_multiple_channels(
            [("S:A:SP", 1.0), ("S:B:SP", 2.0), ("S:C:SP", 3.0)]
        )

        # Assert
        assert len(results) == 3
        assert all("control-target chip in the header" in r.error_message for r in results)
        assert calls == [(SESSION_KEY, TARGET_STANDIN)]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_readonly_run_reads_the_store_not_at_all(self, deployment, store, monkeypatch):
        """The readonly term refuses first and its wording owes the store
        nothing, so a sandboxed run does no file work per write."""
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        calls = self._counted(monkeypatch)
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert result.outcome is WriteOutcome.REFUSED
        assert calls == []


class TestALaunchPinnedRunSaysSoInsteadOfBlamingTheChip:
    """The wording fork for a run the LAUNCH pin refused, not the live store.

    ``store_permits`` ANDs the launch pin into its answer, so a widen that
    lands while a narrow-launched run is in flight leaves the monitor with one
    ``False`` and two possible stories. Told the wrong one, an operator is sent
    to a chip that already reads ``writes`` — the flagship FR15 path producing
    a message that describes a state nobody can observe. So the refusal asks
    the pin separately and names the run.
    """

    @staticmethod
    def _widened(store) -> None:
        """A store that narrows nothing: the operator has already flipped back."""
        store.addressed()
        store.write({SESSION_KEY: {}})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_widen_under_a_running_script_names_the_run_not_the_chip(
        self, deployment, store, monkeypatch
    ):
        # Arrange — launched while standin was narrowed; widened since.
        deployment(ARMED_SECTION)
        self._widened(store)
        monkeypatch.setenv(
            session_store.LAUNCH_POSTURE_ENV_VAR,
            session_store.launch_posture_stamp(TARGET_STANDIN, session_store.POSTURE_SANDBOX),
        )
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert — same outcome and reason as every other refusal here
        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "WRITES_DISABLED"
        # ...and a story about the RUN, with a remedy that exists
        assert f"launched while writes were off for '{TARGET_STANDIN}'" in result.error_message
        assert "not to one already in flight" in result.error_message
        assert "Re-run the script" in result.error_message
        # The message an operator could not act on is exactly what must be gone.
        assert "Turn writes back on" not in result.error_message

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_fail_closed_pin_does_not_claim_anyone_narrowed_anything(
        self, deployment, store, monkeypatch
    ):
        """``*=sandbox`` is what the executor stamps when it could resolve nothing.

        A store it could not read and a target it could not name both land here,
        and neither is a decision an operator made. Reporting it as one would
        send them looking for a narrowing that does not exist.
        """
        # Arrange
        deployment(ARMED_SECTION)
        self._widened(store)
        monkeypatch.setenv(
            session_store.LAUNCH_POSTURE_ENV_VAR,
            session_store.launch_posture_stamp(None, session_store.POSTURE_SANDBOX),
        )
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert result.outcome is WriteOutcome.REFUSED
        assert "most restrictive write state" in result.error_message
        assert "neither its control target nor this session's write state" in result.error_message
        assert "could be resolved" in result.error_message
        assert "Re-run the script" in result.error_message
        assert "chip" not in result.error_message

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_a_live_narrowing_still_reads_as_a_live_narrowing(
        self, deployment, store, monkeypatch
    ):
        """The fork must not swallow the case it sits in front of.

        Launched open, narrowed since: this is the immediate-narrowing half of
        FR15, and the chip IS the remedy.
        """
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        monkeypatch.setenv(
            session_store.LAUNCH_POSTURE_ENV_VAR,
            session_store.launch_posture_stamp(TARGET_STANDIN, session_store.POSTURE_WRITES),
        )
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert result.outcome is WriteOutcome.REFUSED
        assert "control-target chip in the header" in result.error_message
        assert "Re-run the script" not in result.error_message

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_the_fork_costs_no_extra_store_read(self, deployment, store, monkeypatch):
        """The pin is one environment read, so the per-write memo is untouched."""
        # Arrange
        deployment(ARMED_SECTION)
        self._widened(store)
        monkeypatch.setenv(
            session_store.LAUNCH_POSTURE_ENV_VAR,
            session_store.launch_posture_stamp(TARGET_STANDIN, session_store.POSTURE_SANDBOX),
        )
        calls = TestOneStoreReadPerWrite._counted(monkeypatch)
        connector = _built(EPICS, TARGET_STANDIN)

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert result.outcome is WriteOutcome.REFUSED
        assert calls == [(SESSION_KEY, TARGET_STANDIN)]


class _OverridingConnector(_FakeConnector):
    """A connector that answers the posture itself.

    Overriding ``_writes_enabled`` is an established seam — several connectors
    and their test doubles do it — and the write guard has to keep honouring
    the override rather than reaching past it to the base implementation.
    """

    armed = True

    @property
    def _writes_enabled(self) -> bool:
        return self.armed


class TestAnOverriddenPostureStillGates:
    """The guard asks the property, not the base class's implementation of it."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_an_override_that_arms_writes_is_honoured(self, deployment, store):
        """Even against a store and a deployment that would both refuse: the
        subclass said yes, and the subclass owns the answer."""
        # Arrange
        deployment({"type": EPICS, "writes_enabled": False, "connector": {"epics": {}}})
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        connector = _OverridingConnector()
        connector._connector_type = EPICS
        connector._control_target = TARGET_STANDIN

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert result.outcome is WriteOutcome.CONFIRMED
        assert connector.writes == [("S:CORR:1:SP", 0.5)]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_an_override_that_refuses_still_gets_the_right_wording(self, deployment, store):
        """The override sets no memo, so the refusal reads the store itself.

        Which is the pre-memo behaviour, kept deliberately: a subclass cannot
        be expected to fill in bookkeeping the base class invented.
        """
        # Arrange
        deployment(ARMED_SECTION)
        store.addressed()
        store.narrow(standin=session_store.POSTURE_SANDBOX)
        connector = _OverridingConnector()
        connector.armed = False
        connector._connector_type = EPICS
        connector._control_target = TARGET_STANDIN

        # Act
        result = await connector.write_channel("S:CORR:1:SP", 0.5)

        # Assert
        assert result.outcome is WriteOutcome.REFUSED
        assert "control-target chip in the header" in result.error_message
        assert connector.writes == []
