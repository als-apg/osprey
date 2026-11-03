"""Which targets a deployment configures, and whether that is enough to switch.

``configured_targets`` is the enumeration every roster, prober, render and
state file walks: the deployment's baseline, plus each other target whose
connector type resolves here *and* has a non-empty block under
``control_system.connector``. ``switch_capable`` is one rule over it — two
configured targets, whichever two. ``target_configured`` is the per-target
half both read, and the deployment layer's deploy refusals
(``tier_limits_database_targets``) read the same list, so these cases pin what
a deployment is judged to *have*.

The cases are the ones the write-posture epic carried out of the retired
per-type posture tests: the subjects outlived the posture. The write ceiling
is a tier key (``control_system.writes``) now, so the per-type blocks here
carry leaves a rendered profile actually spells — ``timeout_s`` is what the
shipped preset authors on the simulator's block — and never an arming leaf.

The last section pins what the build's normaliser leaves behind. It removes
the retired per-type arming leaf and leaves a block that leaf was the only key
of in place, as ``{}``, "for the render's checks"
(``build_profile_document.normalize_retired_write_keys``). An empty block is
not a configured one, so a target that existed only because its block armed
writes is not on the roster after the rewrite.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey_connectors.types import (
    EPICS,
    IN_PROCESS,
    LIVE_STANDIN,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    VIRTUAL_ACCELERATOR,
    configured_targets,
    switch_capable,
    target_configured,
)

#: A simulator block as the shipped preset authors it — a real leaf, so the
#: block is configured, and nothing in it is about writes.
VA_BLOCK: dict[str, Any] = {"timeout_s": 5.0}

#: A live block that names where the machine is reached.
EPICS_BLOCK: dict[str, Any] = {"gateways": {"read_only": {"address": "gw"}}}


def _section(
    control_system_type: Any = ...,
    writes: Any = ...,
    connector: Any = ...,
) -> dict[str, Any]:
    """A ``control_system:`` section as the rendered config.yml carries it."""
    section: dict[str, Any] = {}
    if control_system_type is not ...:
        section["type"] = control_system_type
    if writes is not ...:
        section["writes"] = writes
    if connector is not ...:
        section["connector"] = connector
    return section


# ---------------------------------------------------------------------------
# switch_capable — two configured targets, whichever two
# ---------------------------------------------------------------------------


def test_a_standin_baseline_is_switch_capable_and_does_not_raise():
    """The baseline is resolved, not looked up among ``live`` and ``va``.

    A ``live_standin`` deployment is baselined on a third target, and asking
    whether it can switch must answer rather than raise — every roster and
    render is downstream of this predicate.
    """
    # Arrange
    section = _section(
        LIVE_STANDIN,
        connector={
            "epics": EPICS_BLOCK,
            "virtual_accelerator": VA_BLOCK,
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert switch_capable(section) is True


def test_a_standin_beside_a_simulator_is_switch_capable_without_a_live_block():
    """Two configured targets are the switching world, whichever two they are.

    A facility rehearsing on its stand-in beside the simulator, with no live
    machine authored yet, has exactly the multi-target world the switch — and
    the per-target toggles behind it — exist for. ``live`` stays underivable
    and simply is not on the roster.
    """
    # Arrange
    section = _section(
        LIVE_STANDIN,
        connector={
            "virtual_accelerator": VA_BLOCK,
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert switch_capable(section) is True
    assert configured_targets(section) == [TARGET_VA, TARGET_STANDIN]


def test_a_va_baseline_beside_a_standin_is_switch_capable():
    """The same two-machine world, baselined on the simulator."""
    # Arrange
    section = _section(
        VIRTUAL_ACCELERATOR,
        connector={
            "virtual_accelerator": VA_BLOCK,
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert switch_capable(section) is True
    assert configured_targets(section) == [TARGET_VA, TARGET_STANDIN]


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
        pytest.param(_section(), id="no-type"),
    ],
)
def test_a_single_target_render_is_not_switch_capable(section):
    """One configured target is nowhere to switch to."""
    assert switch_capable(section) is False


def test_the_live_and_va_pair_is_still_switch_capable():
    """The original two-target shape answers as it always did."""
    # Arrange
    section = _section(
        EPICS,
        connector={
            "epics": EPICS_BLOCK,
            "virtual_accelerator": {"port": 5064},
        },
    )

    # Act / Assert
    assert switch_capable(section) is True


def test_the_in_process_simulator_carrying_a_live_block_is_switch_capable():
    """The simulator in process baselines on ``va``, and an ``epics`` block is ``live``.

    Two configured targets are a switchable deployment, as the served
    simulator beside a live block is.
    """
    # Arrange
    section = _section(
        VIRTUAL_ACCELERATOR,
        connector={
            "epics": EPICS_BLOCK,
            "virtual_accelerator": {"serving": IN_PROCESS},
        },
    )

    # Act / Assert
    assert switch_capable(section) is True


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
        pytest.param(_section(), id="no-type"),
        pytest.param(
            _section(
                EPICS,
                connector={
                    "epics": EPICS_BLOCK,
                    "virtual_accelerator": {"port": 5064},
                },
            ),
            id="live-and-va",
        ),
        pytest.param(
            _section(
                VIRTUAL_ACCELERATOR,
                connector={
                    "virtual_accelerator": VA_BLOCK,
                    LIVE_STANDIN: {"port": 5074},
                },
            ),
            id="va-and-standin",
        ),
        pytest.param(
            _section(
                LIVE_STANDIN,
                connector={
                    "epics": EPICS_BLOCK,
                    "virtual_accelerator": VA_BLOCK,
                    LIVE_STANDIN: {"port": 5074},
                },
            ),
            id="standin-baseline-three-blocks",
        ),
        pytest.param(
            _section(
                VIRTUAL_ACCELERATOR,
                connector={
                    "epics": EPICS_BLOCK,
                    "virtual_accelerator": {"serving": IN_PROCESS},
                },
            ),
            id="in-process-va-and-epics",
        ),
    ],
)
def test_switch_capable_is_two_configured_targets(section):
    """A resolvable section is switch-capable exactly when it configures two targets."""
    assert switch_capable(section) is (len(configured_targets(section)) >= 2)


@pytest.mark.parametrize("section", [None, [], "epics"])
def test_a_section_that_is_not_a_mapping_is_not_switch_capable(section):
    """Only a ``control_system:`` mapping can configure a target."""
    assert switch_capable(section) is False


# ---------------------------------------------------------------------------
# configured_targets — what this deployment has, not what the vocabulary knows
# ---------------------------------------------------------------------------


def test_the_configured_targets_are_the_baseline_and_every_block_behind_one():
    # Arrange
    section = _section(
        EPICS,
        connector={
            "epics": EPICS_BLOCK,
            "virtual_accelerator": VA_BLOCK,
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert configured_targets(section) == [TARGET_LIVE, TARGET_VA, TARGET_STANDIN]


def test_a_standin_baseline_keeps_the_vocabulary_order():
    """Its own target is in the list, where :data:`CONTROL_TARGETS` puts it."""
    # Arrange
    section = _section(
        LIVE_STANDIN,
        connector={
            "epics": EPICS_BLOCK,
            "virtual_accelerator": VA_BLOCK,
            LIVE_STANDIN: {"port": 5074},
        },
    )

    # Act / Assert
    assert configured_targets(section) == [TARGET_LIVE, TARGET_VA, TARGET_STANDIN]


def test_a_va_baseline_enumerates_exactly_as_it_did_before_the_third_target():
    """The shape SC-5 pins: no stand-in block, so nothing about it may change."""
    # Arrange
    section = _section(
        VIRTUAL_ACCELERATOR,
        connector={
            "epics": EPICS_BLOCK,
            "virtual_accelerator": VA_BLOCK,
        },
    )

    # Act / Assert
    assert configured_targets(section) == [TARGET_LIVE, TARGET_VA]


@pytest.mark.parametrize(
    "standin_block",
    [{}, None, "not-a-mapping", 0, [], ...],
    ids=["empty-mapping", "none", "not-a-mapping", "zero", "empty-list", "no-block"],
)
def test_a_target_without_a_usable_block_is_not_configured(standin_block: Any):
    """That block is what a connector is configured from; an empty one is nothing."""
    # Arrange
    connector: dict[str, Any] = {"epics": EPICS_BLOCK}
    if standin_block is not ...:
        connector[LIVE_STANDIN] = standin_block
    section = _section(EPICS, connector=connector)

    # Act / Assert
    assert configured_targets(section) == [TARGET_LIVE]


def test_a_live_that_does_not_resolve_is_not_a_configured_target():
    """``resolve_target`` refuses to guess a real machine, and a refusal is no slot."""
    # Arrange
    section = _section(VIRTUAL_ACCELERATOR, connector={"virtual_accelerator": VA_BLOCK})

    # Act / Assert
    assert configured_targets(section) == [TARGET_VA]


def test_the_baseline_is_configured_even_with_no_block_of_its_own():
    """A deployment is on the connector ``control_system.type`` builds regardless."""
    # Act / Assert
    assert configured_targets(_section()) == [TARGET_VA]
    assert configured_targets(_section(connector={})) == [TARGET_VA]
    assert configured_targets(_section(VIRTUAL_ACCELERATOR)) == [TARGET_VA]
    assert configured_targets(_section(LIVE_STANDIN)) == [TARGET_STANDIN]


@pytest.mark.parametrize(
    "section", [None, "not-a-mapping", 0, [], {}], ids=["none", "string", "zero", "list", "empty"]
)
def test_a_section_that_is_not_a_mapping_still_has_its_baseline(section: Any):
    """Never raises and never empty: the one target such a deployment is on."""
    # Act / Assert
    assert configured_targets(section) == [TARGET_VA]


def test_asking_which_targets_are_configured_does_not_mutate_the_section():
    # Arrange
    section = _section(EPICS, connector={"epics": {"port": 5064}})
    before = {"type": EPICS, "connector": {"epics": {"port": 5064}}}

    # Act
    configured_targets(section)

    # Assert
    assert section == before


# ---------------------------------------------------------------------------
# After the tier rewrite — the block the normaliser emptied
# ---------------------------------------------------------------------------


def test_a_block_the_tier_rewrite_emptied_is_left_as_an_empty_mapping_and_not_configured():
    """The normaliser drops the retired arming leaf and leaves ``{}`` where the block was.

    A pre-tier profile often wrote a simulator block for one reason: to arm
    writes on it. The build's rewrite removes that leaf and writes the tier key
    beside the section instead; the block it emptied stays in the document as
    ``{}`` rather than being pruned. This pins what the enumeration makes of
    that shape: an empty mapping configures nothing, so the target is not on
    the roster, the deployment is no longer switch-capable, and the deploy
    refusal that reads ``configured_targets`` never asks about it. The tier key
    is a leaf of the section, not a block, so it configures no target either.
    """
    # Arrange — the rewrite's output: tier key written, the emptied block left behind
    rewritten = _section(
        EPICS,
        writes="allowed",
        connector={"epics": EPICS_BLOCK, "virtual_accelerator": {}},
    )

    # Act / Assert
    assert target_configured(rewritten, TARGET_VA) is False
    assert configured_targets(rewritten) == [TARGET_LIVE]
    assert switch_capable(rewritten) is False


def test_a_baseline_whose_own_block_the_rewrite_emptied_is_still_configured():
    """The baseline needs no block, so emptying its block costs it nothing.

    A simulator deployment whose only per-type block armed writes is, after
    the rewrite, a simulator deployment with ``{}`` under its own type and the
    tier key beside it — and it is on ``va`` exactly as it was with no block
    at all.
    """
    # Arrange
    rewritten = _section(
        VIRTUAL_ACCELERATOR, writes="allowed", connector={"virtual_accelerator": {}}
    )

    # Act / Assert
    assert configured_targets(rewritten) == [TARGET_VA]
    assert switch_capable(rewritten) is False
