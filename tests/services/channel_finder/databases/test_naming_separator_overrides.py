"""The tree walk behind ``_collect_separator_overrides``.

``build_channels_from_selections`` formats a name from selections alone -- it
never looks a selection up in the tree. The one thing it *does* read from the
tree is the ``_separator`` metadata, collected by walking down the selections in
``_hierarchical_naming._collect_separator_overrides``. That walk has its own
navigation rules, separate from the recursion in ``_build_channel_map``, and
they decide how far down the tree the collection gets before it gives up.

``test_naming_characterization`` pins the outcome of that walk for selections
that match the tree cleanly. This module pins the rest of it: which selections
make the walk skip a level, which make it stop early, and what the two
instance-container fallbacks accept -- a selection that no ``_expansion``
generates still resolves through a container whose *key* matches the level name
case-insensitively, while a branch with no container at all ends the walk and
silently drops every override below it.

Every assertion is on an emitted channel name or on the returned override
mapping. ``_collect_separator_overrides`` is called directly because its return
value is the thing under test, and most cases pair it with the
``build_channels_from_selections`` name that consumes that mapping -- so the
walk is pinned through the public API as well as through its own return value.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest
from tests.services.channel_finder.conftest import _document, range_instances_doc

if TYPE_CHECKING:
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )


# ---------------------------------------------------------------------------
# Documents for shapes the shared fixture family does not cover: an instance
# level whose container is present in one branch and absent in the sibling, a
# container whose key is not upper-case, and an optional tree level with a
# ``_separator`` sitting *below* it.
# ---------------------------------------------------------------------------


def _paired_container_doc() -> dict[str, Any]:
    """One branch with a ``DEVICE`` container, one without, both carrying ``X``.

    ``system(tree) -> device(instances, optional) -> signal(tree) ->
    suffix(tree)``. ``X`` sets ``"_separator": "_"`` in *both* branches, so the
    only difference between them is whether the walk survives the ``device``
    level long enough to reach it.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "instances", "optional": True},
            {"name": "signal", "type": "tree"},
            {"name": "suffix", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}:{suffix}",
        tree={
            "SR": {
                "DEVICE": {
                    "_expansion": {"_type": "range", "_pattern": "D{}", "_range": [1, 2]},
                    "X": {"_separator": "_", "RB": {}},
                }
            },
            "BR": {"X": {"_separator": "_", "RB": {}}},
        },
    )


def _mixed_case_container_doc() -> dict[str, Any]:
    """An instance container keyed ``Device`` rather than ``DEVICE``."""
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "instances"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}",
        tree={
            "SR": {
                "Device": {
                    "_separator": "@",
                    "_expansion": {"_type": "range", "_pattern": "D{}", "_range": [1, 2]},
                    "X": {},
                }
            }
        },
    )


def _optional_skip_doc() -> dict[str, Any]:
    """An optional tree level with an override on the node *after* it.

    ``system(tree) -> subdevice(tree, optional) -> signal(tree) ->
    suffix(tree)``. ``Heartbeat`` sits at ``signal`` and carries the override,
    so it is only reached when the walk skips the omitted ``subdevice`` level
    instead of stopping at it. ``PSU`` is the sibling that does occupy the
    optional level and sets nothing.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "subdevice", "type": "tree", "optional": True},
            {"name": "signal", "type": "tree"},
            {"name": "suffix", "type": "tree"},
        ],
        naming_pattern="{system}:{subdevice}:{signal}:{suffix}",
        tree={
            "SR": {
                "Heartbeat": {"_separator": "_", "RB": {}},
                "PSU": {"Voltage": {"RB": {}}},
            }
        },
    )


def _crossing_instances_doc() -> dict[str, Any]:
    """An override on a node whose next level is ``instances``, not ``tree``."""
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "instances"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}",
        tree={
            "SR": {
                "_separator": "~",
                "DEVICE": {
                    "_expansion": {"_type": "range", "_pattern": "D{}", "_range": [1, 2]},
                    "X": {},
                },
            }
        },
    )


@pytest.fixture
def paired_container_db(
    hier_db_factory: Callable[..., HierarchicalChannelDatabase],
) -> HierarchicalChannelDatabase:
    return hier_db_factory(_paired_container_doc(), "paired_container.json")


@pytest.fixture
def mixed_case_container_db(
    hier_db_factory: Callable[..., HierarchicalChannelDatabase],
) -> HierarchicalChannelDatabase:
    return hier_db_factory(_mixed_case_container_doc(), "mixed_case_container.json")


@pytest.fixture
def optional_skip_db(
    hier_db_factory: Callable[..., HierarchicalChannelDatabase],
) -> HierarchicalChannelDatabase:
    return hier_db_factory(_optional_skip_doc(), "optional_skip.json")


@pytest.fixture
def crossing_instances_db(
    hier_db_factory: Callable[..., HierarchicalChannelDatabase],
) -> HierarchicalChannelDatabase:
    return hier_db_factory(_crossing_instances_doc(), "crossing_instances.json")


@pytest.fixture
def instance_separator_db(
    hier_db_factory: Callable[..., HierarchicalChannelDatabase],
) -> HierarchicalChannelDatabase:
    """``range_instances`` with an override on the ``BPM`` container only.

    ``QUAD`` has a container of its own and no override, which is what tells a
    collected separator apart from a leaked one.
    """
    doc = range_instances_doc()
    doc["tree"]["SR"]["BPM"]["DEVICE"]["_separator"] = "#"
    return hier_db_factory(doc, "instance_separator.json")


class TestInstanceContainerFallback:
    """Resolving an ``instances`` selection when no ``_expansion`` generates it."""

    def test_a_selection_no_expansion_generates_still_finds_the_container(
        self, instance_separator_db
    ) -> None:
        """The container key is matched directly once the expansion search fails.

        ``B01`` is generated by ``DEVICE``'s range and resolves through the
        expansion search; ``BOGUS`` is generated by nothing, and resolves
        through the container key instead. Both reach the same node, so both
        pick up its ``#``.
        """
        assert instance_separator_db._collect_separator_overrides(
            {"system": "SR", "family": "BPM", "device": "B01", "signal": "X"}
        ) == {("device", "signal"): "#"}
        assert instance_separator_db._collect_separator_overrides(
            {"system": "SR", "family": "BPM", "device": "BOGUS", "signal": "X"}
        ) == {("device", "signal"): "#"}

        assert instance_separator_db.build_channels_from_selections(
            {"system": "SR", "family": "BPM", "device": "BOGUS", "signal": "X"}
        ) == ["SR:BPM:BOGUS#X"]

    def test_the_container_key_is_matched_case_insensitively(self, mixed_case_container_db) -> None:
        """``Device`` matches the ``device`` level; the comparison upper-cases both."""
        assert mixed_case_container_db._collect_separator_overrides(
            {"system": "SR", "device": "NOPE", "signal": "X"}
        ) == {("device", "signal"): "@"}
        assert mixed_case_container_db.build_channels_from_selections(
            {"system": "SR", "device": "NOPE", "signal": "X"}
        ) == ["SR:NOPE@X"]

    def test_a_container_that_sets_no_override_contributes_nothing(
        self, instance_separator_db
    ) -> None:
        """``QUAD``'s container is found by the same fallback and yields ``{}``.

        Reaching a node is not the same as collecting from it: the fallback
        resolves ``BOGUS`` under ``QUAD`` too, and ``QUAD`` simply has no
        ``_separator`` for it to pick up.
        """
        assert (
            instance_separator_db._collect_separator_overrides(
                {"system": "SR", "family": "QUAD", "device": "BOGUS", "signal": "Setpoint"}
            )
            == {}
        )
        assert instance_separator_db.build_channels_from_selections(
            {"system": "SR", "family": "QUAD", "device": "BOGUS", "signal": "Setpoint"}
        ) == ["SR:QUAD:BOGUS:Setpoint"]


class TestNavigationStopsAtAMissingContainer:
    """A branch with no container for the ``instances`` level ends the walk."""

    def test_a_branch_without_a_container_loses_every_override_below_it(
        self, paired_container_db
    ) -> None:
        """The same ``X`` node, reached in one branch and not in the other.

        ``SR`` has a ``DEVICE`` container, so the walk passes through it and
        reaches ``X``'s ``_``. ``BR`` has none -- neither an expansion that
        generates ``D1`` nor a child keyed ``DEVICE`` -- so the walk stops at
        the ``device`` level and ``X`` is never visited, even though ``BR``'s
        own ``X`` carries an identical override.
        """
        assert paired_container_db._collect_separator_overrides(
            {"system": "SR", "device": "D1", "signal": "X", "suffix": "RB"}
        ) == {("signal", "suffix"): "_"}
        assert paired_container_db.build_channels_from_selections(
            {"system": "SR", "device": "D1", "signal": "X", "suffix": "RB"}
        ) == ["SR:D1:X_RB"]

        assert (
            paired_container_db._collect_separator_overrides(
                {"system": "BR", "device": "D1", "signal": "X", "suffix": "RB"}
            )
            == {}
        )
        assert paired_container_db.build_channels_from_selections(
            {"system": "BR", "device": "D1", "signal": "X", "suffix": "RB"}
        ) == ["BR:D1:X:RB"]

    def test_overrides_collected_before_the_stop_are_kept(self, hier_db_factory) -> None:
        """Stopping discards what is below, not what is already collected."""
        doc = _paired_container_doc()
        doc["tree"]["BR"]["_separator"] = "-"
        db = hier_db_factory(doc, "paired_container_root_sep.json")
        assert db._collect_separator_overrides(
            {"system": "BR", "device": "D1", "signal": "X", "suffix": "RB"}
        ) == {("system", "signal"): "-"}

    def test_omitting_the_optional_instance_level_walks_past_the_gap(
        self, paired_container_db
    ) -> None:
        """The same ``BR`` branch reaches ``X`` once ``device`` is left out.

        ``device`` is optional, so omitting it skips the level rather than
        stopping at it -- which is what makes the stop above attributable to the
        missing container and not to the branch shape.
        """
        assert paired_container_db._collect_separator_overrides(
            {"system": "BR", "signal": "X", "suffix": "RB"}
        ) == {("signal", "suffix"): "_"}
        assert paired_container_db.build_channels_from_selections(
            {"system": "BR", "signal": "X", "suffix": "RB"}
        ) == ["BR:X_RB"]


class TestSkippedOptionalLevels:
    """An optional level with no usable selection is skipped, and the walk goes on."""

    def test_an_omitted_optional_level_is_skipped(self, optional_skip_db) -> None:
        assert optional_skip_db._collect_separator_overrides(
            {"system": "SR", "signal": "Heartbeat", "suffix": "RB"}
        ) == {("signal", "suffix"): "_"}
        assert optional_skip_db.build_channels_from_selections(
            {"system": "SR", "signal": "Heartbeat", "suffix": "RB"}
        ) == ["SR:Heartbeat_RB"]

    def test_an_empty_optional_selection_is_skipped_the_same_way(self, optional_skip_db) -> None:
        """``""`` and ``[]`` both reduce to "no selection here" for the walk.

        The two diverge afterwards, in the Cartesian product rather than in the
        walk: an empty string leaves an empty component, while an empty list
        collapses the product to no channels at all -- with the override
        collected either way.
        """
        for empty in ("", []):
            assert optional_skip_db._collect_separator_overrides(
                {"system": "SR", "subdevice": empty, "signal": "Heartbeat", "suffix": "RB"}
            ) == {("signal", "suffix"): "_"}

        assert optional_skip_db.build_channels_from_selections(
            {"system": "SR", "subdevice": "", "signal": "Heartbeat", "suffix": "RB"}
        ) == ["SR:Heartbeat_RB"]
        assert (
            optional_skip_db.build_channels_from_selections(
                {"system": "SR", "subdevice": [], "signal": "Heartbeat", "suffix": "RB"}
            )
            == []
        )

    def test_occupying_the_optional_level_reaches_a_different_subtree(
        self, optional_skip_db
    ) -> None:
        """``PSU`` sets no override, so the sibling's ``_`` does not follow it."""
        assert (
            optional_skip_db._collect_separator_overrides(
                {"system": "SR", "subdevice": "PSU", "signal": "Voltage", "suffix": "RB"}
            )
            == {}
        )
        assert optional_skip_db.build_channels_from_selections(
            {"system": "SR", "subdevice": "PSU", "signal": "Voltage", "suffix": "RB"}
        ) == ["SR:PSU:Voltage:RB"]


class TestStoppedWalks:
    """A required level the walk cannot resolve ends it, keeping what came before."""

    def test_a_missing_required_level_stops_the_walk(self, separator_override_db) -> None:
        """``SR``'s override survives; ``BPM``'s, one level further down, does not.

        ``build_channels_from_selections`` refuses a document-level selection
        this incomplete, so the collection is observed directly -- the walk is
        reachable in this state through any caller that collects overrides
        before validating the selections.
        """
        assert separator_override_db._collect_separator_overrides({"system": "SR"}) == {
            ("system", "device"): "-"
        }
        assert separator_override_db._collect_separator_overrides(
            {"system": "SR", "device": "BPM"}
        ) == {("system", "device"): "-", ("device", "signal"): "_"}

    def test_an_empty_required_selection_stops_the_walk(self, separator_override_db) -> None:
        for empty in ("", []):
            assert separator_override_db._collect_separator_overrides(
                {"system": "SR", "device": empty, "signal": "X", "suffix": "RB"}
            ) == {("system", "device"): "-"}

        # ``device`` contributes no component, so the surviving (system, device)
        # override never applies and the pattern's own separators are used.
        assert separator_override_db.build_channels_from_selections(
            {"system": "SR", "device": "", "signal": "X", "suffix": "RB"}
        ) == ["SR:X:RB"]

    def test_a_tree_selection_absent_from_the_node_stops_the_walk(
        self, separator_override_db
    ) -> None:
        """An unknown value is not an error -- it is the end of the walk.

        The name is still built, so the override collected above the unknown
        level is applied to it: ``SR-NOPE`` keeps ``SR``'s dash while ``BPM``'s
        underscore, which the walk never reached, is absent.
        """
        assert separator_override_db._collect_separator_overrides(
            {"system": "SR", "device": "NOPE", "signal": "X", "suffix": "RB"}
        ) == {("system", "device"): "-"}
        assert separator_override_db.build_channels_from_selections(
            {"system": "SR", "device": "NOPE", "signal": "X", "suffix": "RB"}
        ) == ["SR-NOPE:X:RB"]

    def test_an_unknown_first_level_collects_nothing_at_all(self, separator_override_db) -> None:
        assert (
            separator_override_db._collect_separator_overrides(
                {"system": "NOPE", "device": "BPM", "signal": "X", "suffix": "RB"}
            )
            == {}
        )
        assert separator_override_db.build_channels_from_selections(
            {"system": "NOPE", "device": "BPM", "signal": "X", "suffix": "RB"}
        ) == ["NOPE:BPM:X:RB"]


class TestWhereAnOverrideBinds:
    """Which level pair an override lands on, and which selection it is read from."""

    def test_an_override_binds_past_an_intervening_instances_level(
        self, crossing_instances_db
    ) -> None:
        """The next *tree* level is the target, so ``device`` is stepped over.

        Characterized as-is, not endorsed: the resulting ``(system, signal)``
        pair can only apply when ``device`` contributes no component, so for
        every channel this document actually emits the override is inert -- the
        expansion agrees, emitting ``SR:D1:X`` with the pattern's colon.
        """
        assert crossing_instances_db._collect_separator_overrides(
            {"system": "SR", "device": "D1", "signal": "X"}
        ) == {("system", "signal"): "~"}
        assert crossing_instances_db.build_channels_from_selections(
            {"system": "SR", "device": "D1", "signal": "X"}
        ) == ["SR:D1:X"]
        assert list(crossing_instances_db._build_channel_map()) == ["SR:D1:X", "SR:D2:X"]

    def test_a_multi_value_selection_is_walked_by_its_first_value_only(
        self, separator_override_db
    ) -> None:
        """Characterized as-is, not endorsed: one branch decides the whole product.

        The walk reduces ``["BPM", "QUAD"]`` to ``"BPM"`` and collects ``BPM``'s
        underscore, then the product applies it to every combination -- so the
        ``QUAD`` channel is emitted with a separator ``QUAD`` does not set, and
        does not match the name the expansion gives it.
        """
        built = separator_override_db.build_channels_from_selections(
            {"system": "SR", "device": ["BPM", "QUAD"], "signal": "X", "suffix": "RB"}
        )
        assert built == ["SR-BPM_X:RB", "SR-QUAD_X:RB"]
        assert built[1] not in separator_override_db.channel_map

        # Reversing the order moves the underscore, and the same two selections
        # then render with the pattern's colon instead.
        assert separator_override_db.build_channels_from_selections(
            {"system": "SR", "device": ["QUAD", "BPM"], "signal": "X", "suffix": "RB"}
        ) == ["SR-QUAD:X:RB", "SR-BPM:X:RB"]
