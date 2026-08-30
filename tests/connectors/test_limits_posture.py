"""Limits posture, per connector type — the value, its key, and "strict".

``control_system.limits_checking`` used to be one block for the whole
deployment, so a deployment running a live machine beside a virtual accelerator
could not hold ``allow_unlisted_channels: false`` for the one and ``true`` for
the other. The posture is now per connector type, and the piece pinned here is
the value that carries it: :class:`~osprey_connectors.types.LimitsPosture`
holds the two leaves *together with the key that answered them*, so a refusal
sends an operator to the line they actually have to edit rather than to a
deployment-wide key some per-type block overrides.

``strict`` is defined here and nowhere else in the codebase: limits checking on
*and* unlisted channels refused. The tri-state matters — ``None`` is "the
deployment never said", which is not the same as "the deployment said no", and
only an explicit ``False`` makes a target strict.

The resolvers that *build* a posture from a config section are pinned in the
sibling test classes in this module.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest

from osprey_connectors import types as connector_types
from osprey_connectors.types import (
    CONTROL_TARGETS,
    EPICS,
    LIMITS_CHECKING_LEAF,
    LIMITS_LEAVES,
    LIVE_STANDIN,
    MOCK,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    VIRTUAL_ACCELERATOR,
    LimitsPosture,
    incomplete_limits_blocks,
    most_restrictive_limits_posture,
    target_limits_posture,
    type_limits_posture,
)

CUSTOM_TYPE = "mypackage.TangoConnector"

#: Leaf values a limits block may carry that no reader can turn into a boolean.
#: ``"${X}"`` is the one that matters most: environment expansion always yields
#: strings, so an unset variable reaches the resolvers looking exactly like this.
UNREADABLE_LEAVES = ["true", "false", 1, 0, None, [], {}, "${X}"]

ENABLED_LEAF, ALLOW_UNLISTED_LEAF = LIMITS_LEAVES


def _section(limits_checking: Any = ..., connector: Any = ...) -> dict[str, Any]:
    """A ``control_system:`` section as the rendered config.yml carries it.

    Each key is present only when given, so a test can state the difference
    between a deployment that wrote a block and one that never had one.
    """
    section: dict[str, Any] = {}
    if limits_checking is not ...:
        section[LIMITS_CHECKING_LEAF] = limits_checking
    if connector is not ...:
        section["connector"] = connector
    return section


def _block(enabled: Any = ..., allow_unlisted: Any = ...) -> dict[str, Any]:
    """A ``limits_checking:`` block, each leaf present only when given.

    Leaf *presence* is what decides whether a per-type block is complete, so
    omitting a leaf here is how a test writes an incomplete block.
    """
    block: dict[str, Any] = {}
    if enabled is not ...:
        block[ENABLED_LEAF] = enabled
    if allow_unlisted is not ...:
        block[ALLOW_UNLISTED_LEAF] = allow_unlisted
    return block


class TestLimitsPosture:
    """The posture value: its key spelling, ``strict``, and its immutability."""

    def test_leaf_constants_name_the_block_and_its_two_leaves(self) -> None:
        """The block name and the leaves a block must state are spelled once."""
        assert LIMITS_CHECKING_LEAF == "limits_checking"
        assert LIMITS_LEAVES == ("enabled", "allow_unlisted_channels")
        assert isinstance(LIMITS_LEAVES, tuple)

    def test_leaves_exclude_database_path(self) -> None:
        """``database_path`` stays deployment-wide, so it is not a block leaf.

        Compose mounts one limits database for the deployment; a per-type block
        that omits the path is complete, not incomplete.
        """
        assert "database_path" not in LIMITS_LEAVES

    @pytest.mark.parametrize("leaf", LIMITS_LEAVES)
    def test_key_names_the_per_type_block_when_a_type_answered(self, leaf: str) -> None:
        """A posture read from a connector block names that block's key."""
        posture = LimitsPosture(
            enabled=True, allow_unlisted=True, connector_type="virtual_accelerator"
        )
        assert posture.key(leaf) == (
            f"control_system.connector.virtual_accelerator.limits_checking.{leaf}"
        )

    @pytest.mark.parametrize("leaf", LIMITS_LEAVES)
    def test_key_names_the_deployment_wide_block_without_a_type(self, leaf: str) -> None:
        """No type answered means the deployment-wide block answered."""
        posture = LimitsPosture(enabled=True, allow_unlisted=False, connector_type=None)
        assert posture.key(leaf) == f"control_system.limits_checking.{leaf}"

    def test_key_keeps_a_dotted_custom_type_whole(self) -> None:
        """A custom connector's dotted module path is one key, never a path."""
        posture = LimitsPosture(enabled=True, allow_unlisted=False, connector_type=CUSTOM_TYPE)
        assert posture.key("enabled") == (
            "control_system.connector.mypackage.TangoConnector.limits_checking.enabled"
        )

    def test_key_treats_an_empty_type_as_no_type(self) -> None:
        """An empty type names nothing, so the deployment-wide key answers.

        Same reading :func:`~osprey_connectors.types.writes_enabled_key` gives a
        caller holding no type: a key with an empty segment in it would send an
        operator to a line that cannot exist.
        """
        posture = LimitsPosture(enabled=None, allow_unlisted=None, connector_type="")
        assert posture.key("enabled") == "control_system.limits_checking.enabled"

    @pytest.mark.parametrize(
        ("enabled", "allow_unlisted", "expected"),
        [
            (True, False, True),
            (True, True, False),
            (True, None, False),
            (False, False, False),
            (False, True, False),
            (False, None, False),
            (None, False, False),
            (None, True, False),
            (None, None, False),
        ],
    )
    def test_strict_truth_table(
        self, enabled: bool | None, allow_unlisted: bool | None, expected: bool
    ) -> None:
        """Strict is checking on *and* unlisted channels explicitly refused.

        ``None`` counts as not-strict on both leaves: a deployment that never
        stated a posture has not refused anything, and reading silence as a
        refusal would call a target strict on a guarantee nobody wrote down.
        """
        posture = LimitsPosture(enabled=enabled, allow_unlisted=allow_unlisted, connector_type=None)
        assert posture.strict is expected

    def test_strict_is_unaffected_by_the_answering_type(self) -> None:
        """Which block answered changes the key, never the posture."""
        deployment_wide = LimitsPosture(enabled=True, allow_unlisted=False, connector_type=None)
        per_type = LimitsPosture(enabled=True, allow_unlisted=False, connector_type="epics")
        assert deployment_wide.strict is per_type.strict is True

    def test_an_incomplete_block_is_never_strict(self) -> None:
        """A block missing a leaf answers ``None``, which is not strict."""
        posture = LimitsPosture(
            enabled=None,
            allow_unlisted=None,
            connector_type="virtual_accelerator",
            incomplete=("allow_unlisted_channels",),
        )
        assert posture.strict is False

    def test_incomplete_defaults_to_empty(self) -> None:
        """A posture built without the field describes a well-formed block."""
        posture = LimitsPosture(enabled=True, allow_unlisted=True, connector_type=None)
        assert posture.incomplete == ()

    def test_incomplete_names_the_missing_leaves(self) -> None:
        """The field carries leaf names, so a refusal can quote them."""
        posture = LimitsPosture(
            enabled=None,
            allow_unlisted=None,
            connector_type="epics",
            incomplete=("enabled", "allow_unlisted_channels"),
        )
        assert posture.incomplete == ("enabled", "allow_unlisted_channels")

    def test_posture_is_frozen(self) -> None:
        """The posture a refusal quotes cannot be edited after it was resolved."""
        posture = LimitsPosture(enabled=True, allow_unlisted=False, connector_type=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            posture.enabled = False  # type: ignore[misc]

    def test_postures_compare_by_value(self) -> None:
        """Two resolutions of the same config are the same posture."""
        assert LimitsPosture(True, False, "epics") == LimitsPosture(True, False, "epics")
        assert LimitsPosture(True, False, "epics") != LimitsPosture(True, False, None)

    def test_public_symbols_are_importable_from_types(self) -> None:
        """The posture and its constants are part of the module's public surface.

        :mod:`osprey_connectors.types` publishes no ``__all__`` today — every
        public name in it is reachable by import, and the writes family beside
        this one is consumed that way. The assertion is written so that adding
        one later has to include these names rather than silently drop them.
        """
        exported = getattr(connector_types, "__all__", None)
        for name in ("LimitsPosture", "LIMITS_CHECKING_LEAF", "LIMITS_LEAVES"):
            assert hasattr(connector_types, name)
            if exported is not None:
                assert name in exported


class TestTypeLimitsPosture:
    """Resolving one connector *type*'s posture out of a config section.

    The whole-block rule lives here: a per-type ``limits_checking`` block that
    is present states both leaves and then answers alone, with no leaf borrowed
    from the deployment-wide block; a block that is absent — or is not a
    mapping, at any level — leaves the deployment-wide block answering, which
    is the compatibility story for every deployment that never wrote one.
    """

    # ------------------------------------------------------------------
    # No per-type block: the deployment-wide block answers
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("deployment_wide", "expected"),
        [
            (_block(True, True), (True, True)),
            (_block(True, False), (True, False)),
            (_block(False, True), (False, True)),
            (_block(False, False), (False, False)),
            (_block(enabled=True), (True, None)),
            (_block(enabled=False), (False, None)),
            (_block(allow_unlisted=True), (None, True)),
            (_block(allow_unlisted=False), (None, False)),
            (_block(), (None, None)),
        ],
    )
    def test_a_type_with_no_block_of_its_own_reads_the_deployment_wide_block(
        self, deployment_wide: dict[str, Any], expected: tuple[bool | None, bool | None]
    ) -> None:
        """Both leaves are tri-state there: unset stays ``None``, never guessed.

        A deployment that says nothing per type keeps exactly the posture it had
        when the deployment-wide block was the only one there was.
        """
        section = _section(deployment_wide, connector={EPICS: {"timeout": 5.0}})
        posture = type_limits_posture(section, EPICS)
        assert (posture.enabled, posture.allow_unlisted) == expected
        assert posture.connector_type is None
        assert posture.incomplete == ()

    def test_the_deployment_wide_answer_names_the_deployment_wide_key(self) -> None:
        """Carrying no type is what makes the refusal name the editable line."""
        posture = type_limits_posture(_section(_block(True, False)), EPICS)
        assert posture.key(ENABLED_LEAF) == "control_system.limits_checking.enabled"
        assert posture.key(ALLOW_UNLISTED_LEAF) == (
            "control_system.limits_checking.allow_unlisted_channels"
        )

    def test_a_leaf_the_deployment_wide_block_never_carried_is_not_incomplete(self) -> None:
        """Only a *per-type* block has to state both leaves to answer.

        Deployment-wide silence on one leaf is the tri-state, not a malformed
        block: it is the shape every deployment predating per-type blocks has,
        and reading it as incomplete would block every write on the fleet.
        """
        posture = type_limits_posture(_section(_block(enabled=True)), EPICS)
        assert posture == LimitsPosture(True, None, None)
        assert posture.incomplete == ()

    def test_a_type_the_deployment_never_configured_reads_the_deployment_wide_block(
        self,
    ) -> None:
        """A stray block for some other type does not answer for this one."""
        section = _section(
            _block(True, False),
            connector={EPICS: {LIMITS_CHECKING_LEAF: _block(True, True)}},
        )
        assert type_limits_posture(section, MOCK) == LimitsPosture(True, False, None)

    # ------------------------------------------------------------------
    # Garbage counts as absent, and nothing raises
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "connector",
        [
            ...,
            {},
            {VIRTUAL_ACCELERATOR: {LIMITS_CHECKING_LEAF: _block(False, True)}},
            {EPICS: {}},
            {EPICS: {"timeout": 5.0}},
            {EPICS: "epics"},
            {EPICS: None},
            "epics",
            ["epics"],
            None,
        ],
    )
    def test_anything_that_is_not_a_per_type_block_counts_as_absent(self, connector: Any) -> None:
        """A block has to be a mapping at every level to answer for its type.

        A string, a list or a bare ``limits_checking:`` states no posture, and
        reading one as a block would either raise on a write path or invent a
        posture out of a typo. Falling through to the deployment-wide block
        leaves the deployment the posture it actually wrote down.
        """
        section = _section(_block(True, False), connector=connector)
        assert type_limits_posture(section, EPICS) == LimitsPosture(True, False, None)

    @pytest.mark.parametrize("limits_checking", [None, "true", ["enabled"], 5])
    def test_a_per_type_block_that_is_not_a_mapping_is_unreadable(
        self, limits_checking: Any
    ) -> None:
        """It does not inherit: this type said something nobody can read.

        Falling through to the deployment-wide block here would hand a machine
        the posture some other line wrote, on exactly the type whose own block
        is unreadable — the one case where inheritance is a guess rather than
        compatibility. Both leaves incomplete instead, so a reader blocks.
        """
        section = _section(
            _block(True, True),
            connector={EPICS: {LIMITS_CHECKING_LEAF: limits_checking}},
        )
        assert type_limits_posture(section, EPICS) == LimitsPosture(
            None, None, EPICS, LIMITS_LEAVES
        )

    @pytest.mark.parametrize("section", [None, "control_system", ["control_system"], 5, True])
    def test_a_section_that_is_not_a_mapping_states_nothing(self, section: Any) -> None:
        """Never raises: a section nobody can read is a deployment that said nothing."""
        assert type_limits_posture(section, EPICS) == LimitsPosture(None, None, None)

    def test_a_deployment_that_wrote_no_block_at_all_states_nothing(self) -> None:
        """Silence one level up, which is what a fresh deployment has."""
        assert type_limits_posture(_section(), EPICS) == LimitsPosture(None, None, None)

    @pytest.mark.parametrize("limits_checking", [None, "true", ["enabled"], 5])
    def test_a_deployment_wide_block_that_is_not_a_mapping_is_unreadable(
        self, limits_checking: Any
    ) -> None:
        """A block written and unreadable is not the same as no block.

        ``limits_checking: 'true'`` and a bare ``limits_checking:`` are a
        deployment that tried to say something; reading them as silence would
        leave checking off on a config whose author believed it was on. Both
        leaves incomplete, so a reader that must act blocks.
        """
        posture = type_limits_posture(_section(limits_checking), EPICS)
        assert posture == LimitsPosture(None, None, None, LIMITS_LEAVES)

    # ------------------------------------------------------------------
    # A complete per-type block answers alone
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("connector_type", [EPICS, VIRTUAL_ACCELERATOR, CUSTOM_TYPE])
    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize("allow_unlisted", [True, False])
    def test_a_complete_block_answers_alone_and_names_its_own_key(
        self, connector_type: str, enabled: bool, allow_unlisted: bool
    ) -> None:
        """The override is whole: the deployment-wide block is not consulted.

        The deployment-wide block below states the opposite of both leaves, so
        a resolver that inherited either half would be caught here — for a
        built-in type and for a custom connector's dotted module path alike.
        """
        section = _section(
            _block(not enabled, not allow_unlisted),
            connector={connector_type: {LIMITS_CHECKING_LEAF: _block(enabled, allow_unlisted)}},
        )
        posture = type_limits_posture(section, connector_type)
        assert posture == LimitsPosture(enabled, allow_unlisted, connector_type)
        assert posture.key(ENABLED_LEAF) == (
            f"control_system.connector.{connector_type}.limits_checking.enabled"
        )

    def test_a_complete_block_answers_where_no_deployment_wide_block_exists(self) -> None:
        """A per-type block is a posture on its own, not a modifier of one."""
        section = _section(connector={EPICS: {LIMITS_CHECKING_LEAF: _block(True, False)}})
        assert type_limits_posture(section, EPICS) == LimitsPosture(True, False, EPICS)

    def test_a_per_type_block_can_be_strict_where_the_deployment_is_not(self) -> None:
        """The whole point of the feature, read through :attr:`LimitsPosture.strict`."""
        section = _section(
            _block(True, True),
            connector={EPICS: {LIMITS_CHECKING_LEAF: _block(True, False)}},
        )
        assert type_limits_posture(section, EPICS).strict is True
        assert type_limits_posture(section, VIRTUAL_ACCELERATOR).strict is False

    def test_a_dotted_custom_type_is_one_key_and_not_a_path(self) -> None:
        """``mypackage.TangoConnector`` names one block, never two nested ones.

        A connector table that happens to nest the dots is not this type's
        block, so the deployment-wide posture answers rather than a mapping
        that only looks like a match after splitting a key nobody split.
        """
        section = _section(
            _block(True, False),
            connector={
                "mypackage": {"TangoConnector": {LIMITS_CHECKING_LEAF: _block(False, True)}}
            },
        )
        assert type_limits_posture(section, CUSTOM_TYPE) == LimitsPosture(True, False, None)

    # ------------------------------------------------------------------
    # Leaf values: only the literal booleans state a posture
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("value", UNREADABLE_LEAVES)
    def test_a_per_type_leaf_that_cannot_be_read_is_an_incomplete_block(self, value: Any) -> None:
        """A leaf nobody can read is a block that failed to state it.

        Not merely an unstated posture. ``enabled`` unset means "this
        deployment configured no limits checking", so a block that wrote
        ``enabled: 'true'`` — meaning to switch checking on — would have every
        write waved through if the value were read as plain silence. Naming it
        incomplete is what makes a reader block instead.
        """
        section = _section(
            _block(True, True),
            connector={EPICS: {LIMITS_CHECKING_LEAF: _block(value, value)}},
        )
        posture = type_limits_posture(section, EPICS)
        assert posture == LimitsPosture(None, None, EPICS, LIMITS_LEAVES)
        assert posture.strict is False

    @pytest.mark.parametrize("value", UNREADABLE_LEAVES)
    def test_a_deployment_wide_leaf_that_cannot_be_read_is_an_incomplete_block(
        self, value: Any
    ) -> None:
        """Same reading at the deployment-wide level, so the two cannot drift.

        Omitting a leaf is legal here and writing one unreadably is not: the
        deployment-wide block inherits nothing, but a line it did write and
        nobody can read is the same hazard in either scope.
        """
        assert type_limits_posture(_section(_block(value, value)), EPICS) == LimitsPosture(
            None, None, None, LIMITS_LEAVES
        )

    @pytest.mark.parametrize("value", UNREADABLE_LEAVES)
    @pytest.mark.parametrize("leaf", LIMITS_LEAVES)
    def test_one_unreadable_leaf_makes_the_whole_block_answer_nothing(
        self, leaf: str, value: Any
    ) -> None:
        """The readable half is dropped too, in either scope.

        A block written half in one spelling and half in another is a config
        nobody has finished; keeping the readable leaf would hand a caller a
        posture assembled out of one line the operator wrote and one they
        cannot have meant.
        """
        block = {**_block(True, False), leaf: value}
        deployment_wide = type_limits_posture(_section(block), EPICS)
        assert deployment_wide == LimitsPosture(None, None, None, (leaf,))
        per_type = type_limits_posture(
            _section(_block(True, True), connector={EPICS: {LIMITS_CHECKING_LEAF: block}}), EPICS
        )
        assert per_type == LimitsPosture(None, None, EPICS, (leaf,))

    def test_an_unexpanded_environment_variable_is_unreadable(self) -> None:
        """Environment expansion yields strings, so this is the shape it reaches us in.

        A deployment that wired ``enabled`` to a variable nothing set must not
        end up with limits checking silently off.
        """
        section = _section(_block("${OSPREY_LIMITS_ENABLED}", "${OSPREY_ALLOW_UNLISTED}"))
        posture = type_limits_posture(section, EPICS)
        assert posture.incomplete == LIMITS_LEAVES
        assert posture.enabled is None

    # ------------------------------------------------------------------
    # An incomplete per-type block answers nothing, and says what is missing
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("connector_type", [EPICS, CUSTOM_TYPE])
    @pytest.mark.parametrize(
        ("block", "missing"),
        [
            (_block(enabled=True), ("allow_unlisted_channels",)),
            (_block(enabled=False), ("allow_unlisted_channels",)),
            (_block(allow_unlisted=True), ("enabled",)),
            (_block(allow_unlisted=False), ("enabled",)),
            (_block(), ("enabled", "allow_unlisted_channels")),
        ],
    )
    def test_a_block_missing_a_leaf_answers_nothing_and_names_what_is_missing(
        self, connector_type: str, block: dict[str, Any], missing: tuple[str, ...]
    ) -> None:
        """No half-answer and no inheritance: the block is present, so it answers.

        The deployment-wide block below is permissive on both leaves. Borrowing
        either half would hand a deployment a posture it never wrote, on a
        block it half-wrote — so the posture states ``None`` twice and carries
        the missing leaf names for the refusal to quote.
        """
        section = _section(
            _block(True, True),
            connector={connector_type: {LIMITS_CHECKING_LEAF: block}},
        )
        posture = type_limits_posture(section, connector_type)
        assert posture == LimitsPosture(None, None, connector_type, missing)
        assert posture.strict is False

    def test_missing_leaves_are_listed_in_leaf_order(self) -> None:
        """So a refusal reads the same way whichever leaf was dropped."""
        section = _section(connector={EPICS: {LIMITS_CHECKING_LEAF: {"database_path": "/x.db"}}})
        assert type_limits_posture(section, EPICS).incomplete == LIMITS_LEAVES

    def test_an_incomplete_block_still_names_its_own_key(self) -> None:
        """The operator has to be sent to the block they half-wrote."""
        section = _section(
            _block(True, False),
            connector={EPICS: {LIMITS_CHECKING_LEAF: _block(enabled=True)}},
        )
        posture = type_limits_posture(section, EPICS)
        assert posture.key(ALLOW_UNLISTED_LEAF) == (
            "control_system.connector.epics.limits_checking.allow_unlisted_channels"
        )

    def test_database_path_alongside_both_leaves_is_a_complete_block(self) -> None:
        """The path is deployment-wide, so carrying one per type breaks nothing."""
        block = {**_block(True, False), "database_path": "/limits.db"}
        section = _section(connector={EPICS: {LIMITS_CHECKING_LEAF: block}})
        assert type_limits_posture(section, EPICS) == LimitsPosture(True, False, EPICS)

    # ------------------------------------------------------------------
    # A caller holding no type at all
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("connector_type", [None, ""])
    def test_a_caller_with_no_type_reads_the_deployment_wide_block(
        self, connector_type: str | None
    ) -> None:
        """No type is not "some type": it is the deployment-wide question.

        The validator built without a type and the fallback a target that
        resolves to none gets both arrive here, and both must read the block
        they have always read rather than a per-type block for a machine the
        caller never named.
        """
        section = _section(
            _block(True, False),
            connector={EPICS: {LIMITS_CHECKING_LEAF: _block(False, True)}},
        )
        posture = type_limits_posture(section, connector_type)
        assert posture == LimitsPosture(True, False, None)
        assert posture.key(ENABLED_LEAF) == "control_system.limits_checking.enabled"

    # ------------------------------------------------------------------
    # Reading a posture changes nothing
    # ------------------------------------------------------------------

    def test_resolving_does_not_mutate_the_section(self) -> None:
        """Resolvers read a shared, once-loaded config; none of them may write to it."""
        section = _section(
            _block(True, False),
            connector={EPICS: {LIMITS_CHECKING_LEAF: _block(enabled=True)}},
        )
        before = copy.deepcopy(section)
        for connector_type in (EPICS, MOCK, CUSTOM_TYPE, None):
            type_limits_posture(section, connector_type)
        assert section == before


def _va_baseline_deployment() -> dict[str, Any]:
    """A virtual-accelerator deployment that also names its facility's machine.

    Deployment-wide strict, with the virtual accelerator relaxed in a block of
    its own — the configuration the feature exists for. ``live`` is derived from
    the single non-simulated connector block, so all three targets resolve.
    """
    return {
        "type": VIRTUAL_ACCELERATOR,
        LIMITS_CHECKING_LEAF: _block(True, False),
        "connector": {
            EPICS: {"gateway_address": "live.example"},
            VIRTUAL_ACCELERATOR: {
                "gateway_address": "va.example",
                LIMITS_CHECKING_LEAF: _block(True, True),
            },
        },
    }


def _standin_deployment() -> dict[str, Any]:
    """An EPICS deployment running the live stand-in beside its own machine.

    ``live_standin`` is a connector type of its own, so its block is where the
    stand-in's posture is written and it never answers for ``live``.
    """
    return {
        "type": EPICS,
        LIMITS_CHECKING_LEAF: _block(True, False),
        "connector": {
            EPICS: {"gateway_address": "live.example"},
            VIRTUAL_ACCELERATOR: {"gateway_address": "va.example"},
            LIVE_STANDIN: {
                "gateway_address": "standin.example",
                LIMITS_CHECKING_LEAF: _block(True, True),
            },
        },
    }


def _mock_deployment() -> dict[str, Any]:
    """A mock deployment: no real machine, so ``live`` does not resolve at all."""
    return {
        "type": MOCK,
        LIMITS_CHECKING_LEAF: _block(True, False),
        "connector": {MOCK: {"channel_count": 12}},
    }


class TestTargetLimitsPosture:
    """Resolving one session *target*'s posture out of a config section.

    The target half of the family: a roster row, a hook, the executor and the
    tool layer carry a target rather than a connector type, and they must read
    the same posture the connector itself runs under. So the target is resolved
    to its type first and :func:`type_limits_posture` answers from there — one
    posture per machine, never one per holder.

    A target that does not resolve answers the deployment-wide block, which is
    the reading :func:`~osprey_connectors.types.target_writes_enabled` already
    takes: there is no per-type block to consult because there is no type, and
    a deployment that never had a second target keeps the posture it wrote.
    """

    # ------------------------------------------------------------------
    # A VA-baseline deployment with the simulator relaxed
    # ------------------------------------------------------------------

    def test_va_reads_the_virtual_accelerator_block(self) -> None:
        """The relaxation is written under the simulator, so the simulator has it."""
        posture = target_limits_posture(_va_baseline_deployment(), TARGET_VA)
        assert posture == LimitsPosture(True, True, VIRTUAL_ACCELERATOR)
        assert posture.key(ALLOW_UNLISTED_LEAF) == (
            "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels"
        )
        assert posture.strict is False

    def test_live_keeps_the_deployment_wide_posture(self) -> None:
        """``live`` resolves to ``epics`` here, which wrote no block of its own.

        The simulator's relaxation must not reach the facility's machine, and
        the refusal an operator sees on ``live`` has to name the deployment-wide
        line, because that is the line that answered.
        """
        posture = target_limits_posture(_va_baseline_deployment(), TARGET_LIVE)
        assert posture == LimitsPosture(True, False, None)
        assert posture.key(ALLOW_UNLISTED_LEAF) == (
            "control_system.limits_checking.allow_unlisted_channels"
        )
        assert posture.strict is True

    def test_standin_with_no_block_of_its_own_keeps_the_deployment_wide_posture(self) -> None:
        """A stand-in nobody wrote a block for is not relaxed by the simulator's."""
        posture = target_limits_posture(_va_baseline_deployment(), TARGET_STANDIN)
        assert posture == LimitsPosture(True, False, None)
        assert posture.key(ENABLED_LEAF) == "control_system.limits_checking.enabled"

    # ------------------------------------------------------------------
    # A deployment running the live stand-in
    # ------------------------------------------------------------------

    def test_standin_reads_its_own_connector_block(self) -> None:
        """``live_standin`` is a type of its own, so it has a posture of its own."""
        posture = target_limits_posture(_standin_deployment(), TARGET_STANDIN)
        assert posture == LimitsPosture(True, True, LIVE_STANDIN)
        assert posture.key(ENABLED_LEAF) == (
            "control_system.connector.live_standin.limits_checking.enabled"
        )

    @pytest.mark.parametrize("target", [TARGET_LIVE, TARGET_VA])
    def test_the_standin_block_answers_for_no_other_target(self, target: str) -> None:
        """Standing up the stand-in relaxes the stand-in and nothing else."""
        posture = target_limits_posture(_standin_deployment(), target)
        assert posture == LimitsPosture(True, False, None)
        assert posture.strict is True

    def test_an_unknown_target_answers_the_deployment_wide_block(self) -> None:
        """No type means no per-type block, so the deployment-wide one answers."""
        posture = target_limits_posture(_standin_deployment(), "labatory")
        assert posture == LimitsPosture(True, False, None)
        assert posture.key(ENABLED_LEAF) == "control_system.limits_checking.enabled"

    @pytest.mark.parametrize("target", [None, "", "LIVE", 5, ["live"]])
    def test_a_target_that_is_not_a_target_never_raises(self, target: Any) -> None:
        """A holder reading a posture must not be the thing that crashes on a typo."""
        assert target_limits_posture(_standin_deployment(), target) == LimitsPosture(
            True, False, None
        )

    # ------------------------------------------------------------------
    # A mock deployment, where ``live`` names no machine at all
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("target", [TARGET_LIVE, TARGET_VA, TARGET_STANDIN])
    def test_a_mock_deployment_answers_the_deployment_wide_block_everywhere(
        self, target: str
    ) -> None:
        """``live`` does not resolve here and the other two wrote no block.

        Parity with :func:`~osprey_connectors.types.target_writes_enabled`:
        refusing here would take the posture away from every deployment that
        never had a second target, rather than protecting anything.
        """
        posture = target_limits_posture(_mock_deployment(), target)
        assert posture == LimitsPosture(True, False, None)
        assert posture.key(ENABLED_LEAF) == "control_system.limits_checking.enabled"

    def test_a_mock_deployment_with_no_block_at_all_states_nothing(self) -> None:
        """Silence resolves to silence, on a target that resolves and one that does not."""
        section = _section(connector={MOCK: {"channel_count": 12}})
        assert target_limits_posture(section, TARGET_LIVE) == LimitsPosture(None, None, None)
        assert target_limits_posture(section, TARGET_VA) == LimitsPosture(None, None, None)

    def test_a_mock_deployment_reads_a_stray_block_only_where_the_target_resolves(self) -> None:
        """A single stray ``epics`` block is what ``live`` derives from here.

        Deliberately the same derivation
        :func:`~osprey_connectors.types.resolve_target` performs for write
        posture, so a holder that carries a target reads one machine's posture
        and not two. The reachable-set resolver is where such a deployment's
        stray block is ruled out, because there no session can select it.
        """
        section = {
            "type": MOCK,
            LIMITS_CHECKING_LEAF: _block(True, False),
            "connector": {
                MOCK: {"channel_count": 12},
                EPICS: {LIMITS_CHECKING_LEAF: _block(True, True)},
            },
        }
        assert target_limits_posture(section, TARGET_LIVE) == LimitsPosture(True, True, EPICS)
        assert target_limits_posture(section, TARGET_VA) == LimitsPosture(True, False, None)

    # ------------------------------------------------------------------
    # Incomplete blocks and malformed sections
    # ------------------------------------------------------------------

    def test_a_target_whose_block_is_incomplete_answers_nothing(self) -> None:
        """The whole-block rule reaches the target half unchanged."""
        section = _va_baseline_deployment()
        section["connector"][VIRTUAL_ACCELERATOR][LIMITS_CHECKING_LEAF] = _block(enabled=True)
        posture = target_limits_posture(section, TARGET_VA)
        assert posture == LimitsPosture(None, None, VIRTUAL_ACCELERATOR, (ALLOW_UNLISTED_LEAF,))
        assert posture.strict is False

    @pytest.mark.parametrize("section", [None, "control_system", ["control_system"], {}])
    def test_a_section_that_states_nothing_answers_nothing(self, section: Any) -> None:
        """Never raises, whichever target is asked about."""
        for target in CONTROL_TARGETS:
            assert target_limits_posture(section, target) == LimitsPosture(None, None, None)

    def test_resolving_a_target_does_not_mutate_the_section(self) -> None:
        """Resolvers read a shared, once-loaded config; none of them may write to it."""
        section = _standin_deployment()
        before = copy.deepcopy(section)
        for target in [*CONTROL_TARGETS, "labatory", None]:
            target_limits_posture(section, target)
        assert section == before


class TestMostRestrictive:
    """The posture that holds across every target a session here can select.

    What a caller with no target of its own has to assume: the stdlib hook when
    the session's target cannot be read, and any reader that must answer before
    a target is chosen. The reachable set is exactly the one
    :func:`~osprey_connectors.types.session_posture` walks — the configured
    targets when the deployment renders the switch, and otherwise the single
    connector ``control_system.type`` builds, read by *type*. A speculative loop
    over every target in the vocabulary would let the unresolvable-target
    fallback answer for machines no session here reaches.

    The answer is derived rather than read, so both leaves are definite: a
    ``None`` among the inputs counts as not-``True`` on both, and the key stays
    the deployment-wide one because no single per-type line answers for a union.
    """

    # ------------------------------------------------------------------
    # The feature's own configuration
    # ------------------------------------------------------------------

    def test_one_strict_target_makes_the_answer_strict(self) -> None:
        """A relaxed simulator does not relax what a targetless caller may assume.

        ``live`` is strict here and ``va`` is not, so the union refuses unlisted
        channels: the caller that does not know which machine it is about to
        touch must not be handed the permissive half.
        """
        posture = most_restrictive_limits_posture(_va_baseline_deployment())
        assert posture == LimitsPosture(True, False, None)
        assert posture.strict is True

    def test_all_permissive_targets_answer_permissive(self) -> None:
        """The union is not a fail-closed constant: it reports what was written."""
        section = _va_baseline_deployment()
        section[LIMITS_CHECKING_LEAF] = _block(True, True)
        posture = most_restrictive_limits_posture(section)
        assert posture == LimitsPosture(True, True, None)
        assert posture.strict is False

    def test_the_answer_names_the_deployment_wide_keys(self) -> None:
        """No per-type line answers for a union, so none is named.

        Sending an operator to one target's block for a posture that came from
        every target would have them edit a line that changes one machine and
        not the answer.
        """
        posture = most_restrictive_limits_posture(_va_baseline_deployment())
        assert posture.connector_type is None
        assert posture.key(ENABLED_LEAF) == "control_system.limits_checking.enabled"
        assert posture.key(ALLOW_UNLISTED_LEAF) == (
            "control_system.limits_checking.allow_unlisted_channels"
        )

    def test_a_configured_standin_is_part_of_the_reachable_set(self) -> None:
        """Every configured target counts, not just the two the switch is named for."""
        section = _standin_deployment()
        section[LIMITS_CHECKING_LEAF] = _block(True, True)
        assert most_restrictive_limits_posture(section) == LimitsPosture(True, True, None)
        section["connector"][LIVE_STANDIN][LIMITS_CHECKING_LEAF] = _block(True, False)
        assert most_restrictive_limits_posture(section) == LimitsPosture(True, False, None)

    # ------------------------------------------------------------------
    # A deployment with no per-type block at all
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("deployment_wide", [_block(True, False), _block(True, True)])
    def test_no_per_type_block_answers_the_deployment_wide_posture(
        self, deployment_wide: dict[str, Any]
    ) -> None:
        """The compatibility story: every target reads one block, so the union is it."""
        section = _va_baseline_deployment()
        section[LIMITS_CHECKING_LEAF] = deployment_wide
        del section["connector"][VIRTUAL_ACCELERATOR][LIMITS_CHECKING_LEAF]
        expected = type_limits_posture(section, EPICS)
        assert most_restrictive_limits_posture(section) == expected

    # ------------------------------------------------------------------
    # ``enabled`` is a union, ``allow_unlisted`` an intersection
    # ------------------------------------------------------------------

    def test_enabled_is_true_when_any_target_checks_limits(self) -> None:
        """Checking is on for the caller as soon as one reachable machine checks.

        The two leaves fold in opposite directions on purpose: limits checking
        being on somewhere is a constraint that may apply, while permission to
        write unlisted channels only holds where every machine grants it.
        """
        section = _va_baseline_deployment()
        section[LIMITS_CHECKING_LEAF] = _block(False, True)
        section["connector"][VIRTUAL_ACCELERATOR][LIMITS_CHECKING_LEAF] = _block(True, True)
        assert most_restrictive_limits_posture(section) == LimitsPosture(True, True, None)

    def test_enabled_is_false_when_no_target_checks_limits(self) -> None:
        """A deployment nobody checks limits on is reported as such."""
        section = _va_baseline_deployment()
        section[LIMITS_CHECKING_LEAF] = _block(False, True)
        section["connector"][VIRTUAL_ACCELERATOR][LIMITS_CHECKING_LEAF] = _block(False, True)
        assert most_restrictive_limits_posture(section) == LimitsPosture(False, True, None)

    @pytest.mark.parametrize(
        "va_block", [_block(True, ...), _block(True, None), _block(True, "yes")]
    )
    def test_a_target_that_states_nothing_counts_as_not_permitted(
        self, va_block: dict[str, Any]
    ) -> None:
        """``None`` is not ``True``, whether it came from silence or from garbage.

        An incomplete block, a leaf YAML read as ``None`` and a quoted string
        all leave one reachable target with no stated permission, and a union
        that granted permission anyway would be inventing it.
        """
        section = _va_baseline_deployment()
        section[LIMITS_CHECKING_LEAF] = _block(True, True)
        section["connector"][VIRTUAL_ACCELERATOR][LIMITS_CHECKING_LEAF] = va_block
        posture = most_restrictive_limits_posture(section)
        assert posture == LimitsPosture(True, False, None, (ALLOW_UNLISTED_LEAF,))
        assert posture.strict is True

    def test_a_deployment_that_stated_nothing_permits_nothing(self) -> None:
        """Both leaves fold to a definite ``False``: silence grants no permission."""
        assert most_restrictive_limits_posture(_section()) == LimitsPosture(False, False, None)

    @pytest.mark.parametrize("value", UNREADABLE_LEAVES)
    def test_a_deployment_whose_only_posture_is_unreadable_permits_nothing(
        self, value: Any
    ) -> None:
        """An incomplete posture counts as not-``True`` on both leaves.

        The union must not report checking as *on* because a line meant to
        switch it on could not be read — the caller would be told a guarantee
        holds that the runtime is about to refuse every write over.
        """
        section = {
            "type": VIRTUAL_ACCELERATOR,
            LIMITS_CHECKING_LEAF: _block(value, value),
            "connector": {VIRTUAL_ACCELERATOR: {"gateway_address": "va.example"}},
        }
        assert most_restrictive_limits_posture(section) == LimitsPosture(
            False, False, None, LIMITS_LEAVES
        )

    def test_one_unreadable_target_makes_the_union_strict(self) -> None:
        """A reachable machine nobody can read a posture for grants no permission."""
        section = _va_baseline_deployment()
        section[LIMITS_CHECKING_LEAF] = _block(True, True)
        section["connector"][VIRTUAL_ACCELERATOR][LIMITS_CHECKING_LEAF] = _block(True, "${X}")
        posture = most_restrictive_limits_posture(section)
        assert posture == LimitsPosture(True, False, None, (ALLOW_UNLISTED_LEAF,))
        assert posture.strict is True

    # ------------------------------------------------------------------
    # Deployments that do not render the switch
    # ------------------------------------------------------------------

    def test_a_mock_deployment_ignores_a_stray_live_block(self) -> None:
        """No switch here, so the built connector's own posture is the whole answer.

        ``live`` derives to ``epics`` on this section, but no session on a mock
        deployment ever reaches it — reading its relaxation would publish a
        posture the runtime does not share. The baseline is read by *type*, so
        the stray block is not consulted at all.
        """
        section = {
            "type": MOCK,
            LIMITS_CHECKING_LEAF: _block(True, False),
            "connector": {
                MOCK: {"channel_count": 12},
                EPICS: {LIMITS_CHECKING_LEAF: _block(True, True)},
            },
        }
        posture = most_restrictive_limits_posture(section)
        assert posture == LimitsPosture(True, False, None)
        assert posture.strict is True

    def test_a_mock_deployment_reads_its_own_per_type_block(self) -> None:
        """The baseline type's block still answers where the deployment wrote one."""
        section = {
            "type": MOCK,
            LIMITS_CHECKING_LEAF: _block(True, False),
            "connector": {MOCK: {LIMITS_CHECKING_LEAF: _block(True, True)}},
        }
        assert most_restrictive_limits_posture(section) == LimitsPosture(True, True, None)

    def test_a_va_only_deployment_reads_its_own_block(self) -> None:
        """One configured target and no switch: that target's posture is the answer."""
        section = {
            "type": VIRTUAL_ACCELERATOR,
            LIMITS_CHECKING_LEAF: _block(True, False),
            "connector": {VIRTUAL_ACCELERATOR: {LIMITS_CHECKING_LEAF: _block(True, True)}},
        }
        assert most_restrictive_limits_posture(section) == LimitsPosture(True, True, None)

    # ------------------------------------------------------------------
    # Malformed sections
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("section", [None, "control_system", ["control_system"], {}, 5])
    def test_a_section_that_states_nothing_never_raises(self, section: Any) -> None:
        """A caller with no target is often the one with no good config either."""
        assert most_restrictive_limits_posture(section) == LimitsPosture(False, False, None)

    # ------------------------------------------------------------------
    # Incompleteness travels through the fold
    # ------------------------------------------------------------------

    def test_an_incomplete_reachable_posture_makes_the_fold_incomplete(self) -> None:
        """Dropping this is the worst failure available in this function.

        Both leaf folds send an incomplete posture's ``None`` to ``False``,
        which reads as "checking off, and nothing permitted" — and a validator
        built from a posture whose ``enabled`` is not ``True`` is no validator
        at all. So a fold that forgot the incompleteness would hand *no*
        enforcement to the one caller that does not know which machine it is
        about to touch, on a deployment whose only limits line cannot be read.
        """
        section = {"type": EPICS, LIMITS_CHECKING_LEAF: _block("true", False)}
        assert type_limits_posture(section, EPICS).incomplete == (ENABLED_LEAF,)
        assert most_restrictive_limits_posture(section).incomplete == (ENABLED_LEAF,)

    def test_the_fold_unions_incomplete_leaves_in_leaf_order(self) -> None:
        """Different reachable machines may fail to state different leaves."""
        section = _va_baseline_deployment()
        section[LIMITS_CHECKING_LEAF] = _block("true", True)
        section["connector"][VIRTUAL_ACCELERATOR][LIMITS_CHECKING_LEAF] = _block(enabled=True)
        assert most_restrictive_limits_posture(section).incomplete == LIMITS_LEAVES

    def test_a_deployment_whose_blocks_are_all_well_formed_folds_complete(self) -> None:
        """The fold does not invent incompleteness either."""
        assert most_restrictive_limits_posture(_va_baseline_deployment()).incomplete == ()

    def test_resolving_the_union_does_not_mutate_the_section(self) -> None:
        """Resolvers read a shared, once-loaded config; none of them may write to it."""
        section = _standin_deployment()
        before = copy.deepcopy(section)
        most_restrictive_limits_posture(section)
        assert section == before


def _missing(connector_type: str, leaf: str) -> str:
    """The line the build refusal is expected to carry for one missing leaf."""
    return (
        f"control_system.connector.{connector_type}.limits_checking.{leaf} is missing; "
        "a per-type limits block must state both enabled and allow_unlisted_channels"
    )


def _not_a_block(connector_type: str | None, value: Any) -> str:
    """The line the build refusal is expected to carry for a non-mapping block."""
    key = LimitsPosture(None, None, connector_type).key(ENABLED_LEAF).rsplit(".", 1)[0]
    return (
        f"{key} is {value!r}, not a block; a limits block is a mapping stating "
        "enabled and allow_unlisted_channels"
    )


def _unreadable(connector_type: str | None, leaf: str, value: Any) -> str:
    """The line the build refusal is expected to carry for one unreadable leaf.

    Spelled from :meth:`~osprey_connectors.types.LimitsPosture.key` so the test
    pins the sentence rather than restating how a key is built.
    """
    key = LimitsPosture(None, None, connector_type).key(leaf)
    return (
        f"{key} is {value!r}, not a literal true or false; a limits leaf that "
        "cannot be read states no posture and blocks every write as a failsafe"
    )


class TestIncompleteBlocks:
    """The half-written per-type blocks a render carries, named for a refusal.

    The build and ``osprey validate`` refuse a profile that writes one leaf of a
    ``limits_checking`` block, because half a block is a posture nobody stated:
    :func:`~osprey_connectors.types.type_limits_posture` answers ``None`` twice
    for it and a write path falls back to a failsafe. This resolver is the
    render-side half of that refusal — it reads the config the build actually
    produced, so a spelling the profile layer could not classify still gets
    caught before it ships.

    It is a lint and not a posture: it answers about a *section*, never raises,
    and says nothing at all about a deployment whose blocks are well-formed.
    """

    # ------------------------------------------------------------------
    # Nothing to report
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "section",
        [
            _va_baseline_deployment(),
            _standin_deployment(),
            _mock_deployment(),
            _section(_block(True, False)),
            _section(_block(True, False), connector={EPICS: {"gateway_address": "x"}}),
            _section(connector={EPICS: {LIMITS_CHECKING_LEAF: _block(True, False)}}),
        ],
    )
    def test_well_formed_and_absent_blocks_report_nothing(self, section: Any) -> None:
        """A complete block and no block at all are both legal configs."""
        assert incomplete_limits_blocks(section) == []

    def test_a_block_carrying_both_leaves_and_a_path_is_complete(self) -> None:
        """``database_path`` is deployment-wide, so carrying one per type is legal."""
        block = {**_block(True, False), "database_path": "/limits.db"}
        section = _section(connector={EPICS: {LIMITS_CHECKING_LEAF: block}})
        assert incomplete_limits_blocks(section) == []

    @pytest.mark.parametrize("value", UNREADABLE_LEAVES)
    def test_an_unreadable_per_type_leaf_is_reported(self, value: Any) -> None:
        """The same reading the resolver takes, so the two cannot disagree.

        Such a block blocks every write at runtime, which is the safe direction
        but a poor way to find out. Refusing the build is where the config is
        still cheap to fix.
        """
        section = _section(connector={EPICS: {LIMITS_CHECKING_LEAF: _block(value, value)}})
        assert incomplete_limits_blocks(section) == [
            _unreadable(EPICS, ENABLED_LEAF, value),
            _unreadable(EPICS, ALLOW_UNLISTED_LEAF, value),
        ]

    @pytest.mark.parametrize("value", UNREADABLE_LEAVES)
    def test_an_unreadable_deployment_wide_leaf_is_reported(self, value: Any) -> None:
        """The hazard is the leaf, not the block it sits in.

        A deployment-wide ``enabled`` nobody can read is exactly the shape an
        unexpanded ``${VAR}`` arrives in, and the deployment-wide block is where
        most deployments write their only limits posture.
        """
        assert incomplete_limits_blocks(_section(_block(enabled=value))) == [
            _unreadable(None, ENABLED_LEAF, value)
        ]

    def test_an_unreadable_leaf_line_says_what_it_found_and_what_it_costs(self) -> None:
        """The refusal quotes the value, so an operator sees why it did not read."""
        section = _section(_block(enabled="${OSPREY_LIMITS_ENABLED}"))
        assert incomplete_limits_blocks(section) == [
            "control_system.limits_checking.enabled is '${OSPREY_LIMITS_ENABLED}', "
            "not a literal true or false; a limits leaf that cannot be read states "
            "no posture and blocks every write as a failsafe"
        ]

    @pytest.mark.parametrize("value", [None, "true", ["enabled"], 5])
    def test_a_per_type_block_that_is_not_a_mapping_is_reported_once(self, value: Any) -> None:
        """One line naming the block, because there are no leaves to name."""
        section = _section(connector={EPICS: {LIMITS_CHECKING_LEAF: value}})
        assert incomplete_limits_blocks(section) == [_not_a_block(EPICS, value)]

    @pytest.mark.parametrize("value", [None, "true", ["enabled"], 5])
    def test_a_deployment_wide_block_that_is_not_a_mapping_is_reported_once(
        self, value: Any
    ) -> None:
        """The block is written and unreadable in either scope alike."""
        assert incomplete_limits_blocks(_section(value)) == [_not_a_block(None, value)]

    def test_a_bare_limits_checking_line_is_reported_rather_than_ignored(self) -> None:
        """The shape a half-typed block actually has in YAML."""
        assert incomplete_limits_blocks(_section(None)) == [
            "control_system.limits_checking is None, not a block; a limits block "
            "is a mapping stating enabled and allow_unlisted_channels"
        ]

    def test_a_leaf_the_deployment_wide_block_never_carried_is_not_reported(self) -> None:
        """Only a per-type block has to state both leaves to answer at all.

        Deployment-wide silence on a leaf is the shape every deployment
        predating per-type blocks has; refusing it would refuse the fleet.
        """
        assert incomplete_limits_blocks(_section(_block(enabled=True))) == []
        assert incomplete_limits_blocks(_section(_block())) == []

    def test_the_deployment_wide_block_is_reported_before_the_connector_blocks(self) -> None:
        """One refusal, read top-down the way the config is written."""
        section = _section(
            _block(enabled="true"),
            connector={EPICS: {LIMITS_CHECKING_LEAF: _block(enabled=True)}},
        )
        assert incomplete_limits_blocks(section) == [
            _unreadable(None, ENABLED_LEAF, "true"),
            _missing(EPICS, ALLOW_UNLISTED_LEAF),
        ]

    # ------------------------------------------------------------------
    # One line per missing leaf
    # ------------------------------------------------------------------

    def test_a_block_missing_allow_unlisted_names_that_key(self) -> None:
        """The refusal quotes the key an operator has to add, verbatim."""
        section = _section(
            connector={VIRTUAL_ACCELERATOR: {LIMITS_CHECKING_LEAF: _block(enabled=True)}}
        )
        assert incomplete_limits_blocks(section) == [
            "control_system.connector.virtual_accelerator.limits_checking."
            "allow_unlisted_channels is missing; a per-type limits block must state "
            "both enabled and allow_unlisted_channels"
        ]

    def test_a_block_missing_enabled_names_that_key(self) -> None:
        """Either leaf alone is half a block, so either one is refused."""
        section = _section(connector={EPICS: {LIMITS_CHECKING_LEAF: _block(allow_unlisted=False)}})
        assert incomplete_limits_blocks(section) == [_missing(EPICS, ENABLED_LEAF)]

    def test_a_block_missing_both_leaves_names_both_in_leaf_order(self) -> None:
        """One line per leaf, so an operator can add them without re-running the build."""
        section = _section(
            connector={EPICS: {LIMITS_CHECKING_LEAF: {"database_path": "/limits.db"}}}
        )
        assert incomplete_limits_blocks(section) == [
            _missing(EPICS, ENABLED_LEAF),
            _missing(EPICS, ALLOW_UNLISTED_LEAF),
        ]

    def test_an_empty_block_is_incomplete_rather_than_absent(self) -> None:
        """A written ``limits_checking:`` with nothing under it is a half-written block."""
        section = _section(_block(True, False), connector={EPICS: {LIMITS_CHECKING_LEAF: _block()}})
        assert incomplete_limits_blocks(section) == [
            _missing(EPICS, ENABLED_LEAF),
            _missing(EPICS, ALLOW_UNLISTED_LEAF),
        ]

    def test_every_connector_type_is_walked(self) -> None:
        """One build refusal lists every half-written block the render carries."""
        section = _section(
            connector={
                EPICS: {LIMITS_CHECKING_LEAF: _block(True, False)},
                VIRTUAL_ACCELERATOR: {LIMITS_CHECKING_LEAF: _block(enabled=True)},
                LIVE_STANDIN: {LIMITS_CHECKING_LEAF: _block(allow_unlisted=True)},
            }
        )
        assert incomplete_limits_blocks(section) == [
            _missing(VIRTUAL_ACCELERATOR, ALLOW_UNLISTED_LEAF),
            _missing(LIVE_STANDIN, ENABLED_LEAF),
        ]

    def test_a_dotted_custom_type_is_named_whole(self) -> None:
        """The connector table's key is the type, dots and all, never a path."""
        section = _section(connector={CUSTOM_TYPE: {LIMITS_CHECKING_LEAF: _block(enabled=True)}})
        assert incomplete_limits_blocks(section) == [
            "control_system.connector.mypackage.TangoConnector.limits_checking."
            "allow_unlisted_channels is missing; a per-type limits block must state "
            "both enabled and allow_unlisted_channels"
        ]

    def test_the_reported_key_is_the_one_the_resolver_reads(self) -> None:
        """Refusal and runtime name one line, because both spell it from the posture."""
        section = _section(connector={EPICS: {LIMITS_CHECKING_LEAF: _block(enabled=True)}})
        posture = type_limits_posture(section, EPICS)
        assert posture.incomplete == (ALLOW_UNLISTED_LEAF,)
        assert incomplete_limits_blocks(section)[0].startswith(
            posture.key(ALLOW_UNLISTED_LEAF) + " is missing;"
        )

    # ------------------------------------------------------------------
    # Malformed sections: a lint never raises
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "connector",
        [
            {EPICS: "epics"},
            {EPICS: None},
            {EPICS: ["limits_checking"]},
            "epics",
            ["epics"],
            None,
            {},
        ],
    )
    def test_anything_that_is_not_a_block_is_not_an_incomplete_block(self, connector: Any) -> None:
        """A block has to be a mapping to be half-written.

        Reporting a string as a missing leaf would send an operator to add a
        leaf under a line that is not a block at all; the deployment simply has
        no per-type posture there, which the resolver already reads as absent.
        """
        assert incomplete_limits_blocks(_section(_block(True, False), connector=connector)) == []

    @pytest.mark.parametrize("section", [None, "control_system", ["control_system"], 5, {}])
    def test_a_section_that_is_not_a_section_reports_nothing(self, section: Any) -> None:
        """Pure over the rendered section, and never the thing that fails a build."""
        assert incomplete_limits_blocks(section) == []

    def test_linting_does_not_mutate_the_section(self) -> None:
        """Resolvers read a shared, once-loaded config; none of them may write to it."""
        section = _section(
            _block(True, False),
            connector={
                EPICS: {LIMITS_CHECKING_LEAF: _block(enabled=True)},
                VIRTUAL_ACCELERATOR: {"gateway_address": "va.example"},
            },
        )
        before = copy.deepcopy(section)
        incomplete_limits_blocks(section)
        assert section == before
