"""Characterization of ``_hierarchical_naming`` -- what it does today, not what it should do.

This module is a safety net, not a specification. Every assertion below was
derived by running the current implementation and writing down its answer; none
of it was derived from the docstrings in
``osprey.services.channel_finder.databases._hierarchical_naming``. Several of the
behaviours pinned here are, in the judgement of the author, wrong. They are
pinned anyway, with a comment saying so, because the point of the file is to make
a *behaviour-preserving* change to that module detectable -- and a test that
asserts the intended behaviour cannot tell a preserved bug from a fixed one.

Three surprises are characterized explicitly, each in its own section below:

1. ``_is_leaf: true`` combined with dict children means different things under a
   required level and under an optional level; under the optional one the
   children are silently dropped.
2. An all-remaining-optional tail whose next level is ``tree``-typed never
   produces the bare parent channel, while the ``instances``-typed counterpart
   does.
3. ``get_statistics`` buckets by raw tree key, so a tree using ``_channel_part``
   reports ``channel_count: 0`` for systems that do have channels.

Nothing here asserts on coverage, on private call counts, or on anything else
that a pure deletion of unreachable code inside the module would perturb. The
observable surface is the emitted channel names, the ``channel_map`` payloads,
and the return value of ``build_channels_from_selections``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.services.channel_finder.conftest import (
    _document,
    hybrid_expansion_doc,
    range_instances_doc,
    separator_override_doc,
    tree_only_doc,
)

if TYPE_CHECKING:
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )


# ---------------------------------------------------------------------------
# The full channel inventory, one entry per shape in the shared fixture family.
#
# Lists are ordered, because the emission order is itself current behaviour:
# it follows the document's key order through the recursion, and a change to it
# would mean the traversal changed.
# ---------------------------------------------------------------------------

EXPECTED_CHANNELS: dict[str, list[str]] = {
    "tree_only_db": [
        "SR:BPM:X",
        "SR:BPM:Y",
        "SR:QUAD:Setpoint",
        "BR",
    ],
    "range_instances_db": [
        "SR:BPM:B01:X",
        "SR:BPM:B01:Y",
        "SR:BPM:B02:X",
        "SR:BPM:B02:Y",
        "SR:BPM:B03:X",
        "SR:BPM:B03:Y",
        "SR:QUAD:Q1:Setpoint",
        "SR:QUAD:Q1:Readback",
        "SR:QUAD:Q2:Setpoint",
        "SR:QUAD:Q2:Readback",
    ],
    "list_instances_db": [
        "RF:MAIN:Power",
        "RF:MAIN:Phase",
        "RF:BACKUP:Power",
        "RF:BACKUP:Phase",
        "RF:TEST:Power",
        "RF:TEST:Phase",
    ],
    "optional_tree_level_db": [
        "SR:PS:PSU:Voltage",
        "SR:PS:PSU:Current",
        "SR:PS:Heartbeat",
    ],
    # BR has no DEVICE container and contributes nothing at all -- not even its
    # own ``Status`` child, which never becomes a channel.
    "optional_instance_level_db": [
        "SR:D1:X",
        "SR:D2:X",
    ],
    "hybrid_expansion_db": [
        "SR:CH-1:Voltage",
        "SR:CH-1:Current",
        "SR:CH-2:Voltage",
        "SR:CH-2:Current",
        "SR:CH-3:Voltage",
        "SR:CH-3:Current",
        "SR:PS:Status",
    ],
    # BPM's scalar X/Y children are invisible to the dict guards, so BPM itself
    # is the channel and the scalars produce nothing.
    "scalar_leaves_db": [
        "SR:BPM",
        "SR:QUAD:Setpoint",
    ],
    "separator_override_db": [
        "SR-BPM_X:RB",
        "SR-BPM_X:SP",
        "SR-QUAD:Setpoint:RB",
    ],
    "channel_part_db": [
        "MAG:PS:Current",
        "RACK:Temp",
    ],
    "leaf_with_children_db": [
        "SR:PS:SIGNAL-Y",
        "SR:PS:SIGNAL-Y:RB",
        "SR:PS:SIGNAL-Y:SP",
    ],
    # Same SIGNAL-Y node as above, one level made optional: RB/SP are gone.
    # See the asymmetry section for the full statement of this.
    "optional_leaf_with_children_db": [
        "SR:PS:SIGNAL-Y",
        "SR:PS:PSU:Voltage",
    ],
    "all_optional_next_instances_db": [
        "SR",
        "SR:X",
        "SR:Y",
        "BR:D1:Z",
        "BR:D2:Z",
    ],
    # No bare "SR" here, unlike the instances counterpart above.
    "all_optional_next_tree_db": [
        "SR:PSU:Voltage",
        "SR:PSU:Current",
    ],
}


#: Channels that ``_build_channel_map`` emits but ``build_channels_from_selections``
#: cannot reproduce from their own stored path, because the path stops before a
#: *required* pattern level. Empty for every other shape -- see
#: ``TestBuildChannelsFromSelections`` for the full statement.
NON_REPLAYABLE_CHANNELS: dict[str, set[str]] = {name: set() for name in EXPECTED_CHANNELS} | {
    # BR has no children at all, so it terminates at ``system``.
    "tree_only_db": {"BR"},
    # BPM's children are scalars, invisible to the dict guards.
    "scalar_leaves_db": {"SR:BPM"},
    # SIGNAL-Y is a channel in its own right at the ``signal`` level, with
    # ``suffix`` left unfilled.
    "leaf_with_children_db": {"SR:PS:SIGNAL-Y"},
}


@pytest.fixture(params=sorted(EXPECTED_CHANNELS))
def any_shape(request) -> tuple[str, HierarchicalChannelDatabase]:
    """Every shape in the shared fixture family, one at a time."""
    return request.param, request.getfixturevalue(request.param)


class TestChannelInventory:
    """The complete set of channel names each shape expands to."""

    @pytest.mark.parametrize(("fixture_name", "expected"), sorted(EXPECTED_CHANNELS.items()))
    def test_build_channel_map_emits_exactly_these_channels(
        self, request, fixture_name: str, expected: list[str]
    ) -> None:
        db = request.getfixturevalue(fixture_name)
        assert list(db._build_channel_map()) == expected

    def test_loaded_channel_map_matches_a_fresh_expansion(self, any_shape) -> None:
        """``load_database`` stores the same map a fresh call builds."""
        _, db = any_shape
        assert db.channel_map == db._build_channel_map()

    def test_expansion_is_repeatable(self, any_shape) -> None:
        """Expanding twice yields an equal map -- no accumulated state."""
        _, db = any_shape
        assert db._build_channel_map() == db._build_channel_map()

    def test_every_entry_is_keyed_by_its_own_channel_name(self, any_shape) -> None:
        """Each value carries ``channel`` (equal to the key) and a ``path`` dict."""
        _, db = any_shape
        for name, info in db._build_channel_map().items():
            assert info["channel"] == name
            assert isinstance(info["path"], dict)
            assert set(info) == {"channel", "path"}


class TestChannelPaths:
    """The ``path`` payload stored alongside each channel name."""

    def test_tree_only_paths(self, tree_only_db) -> None:
        paths = {k: v["path"] for k, v in tree_only_db._build_channel_map().items()}
        assert paths == {
            "SR:BPM:X": {"system": "SR", "device": "BPM", "signal": "X"},
            "SR:BPM:Y": {"system": "SR", "device": "BPM", "signal": "Y"},
            "SR:QUAD:Setpoint": {"system": "SR", "device": "QUAD", "signal": "Setpoint"},
            # BR terminates early; the levels it never reached are simply absent
            # from the path rather than present as empty strings.
            "BR": {"system": "BR"},
        }

    def test_optional_tree_level_paths_omit_the_skipped_level(self, optional_tree_level_db) -> None:
        paths = {k: v["path"] for k, v in optional_tree_level_db._build_channel_map().items()}
        assert paths["SR:PS:PSU:Voltage"] == {
            "system": "SR",
            "device": "PS",
            "subdevice": "PSU",
            "signal": "Voltage",
        }
        # Heartbeat skipped ``subdevice`` and was re-assigned to ``signal``.
        assert paths["SR:PS:Heartbeat"] == {
            "system": "SR",
            "device": "PS",
            "signal": "Heartbeat",
        }

    def test_channel_part_paths_record_the_emitted_part_not_the_tree_key(
        self, channel_part_db
    ) -> None:
        """The path stores ``_channel_part`` values, so the tree key is unrecoverable.

        Characterized as-is, not endorsed: ``Magnets`` and ``Building-1`` are the
        keys a caller would navigate with, but neither appears in any path. The
        empty ``_channel_part`` is stored as an empty string, which is what makes
        ``get_statistics`` mis-bucket (see the section on that below).
        """
        paths = {k: v["path"] for k, v in channel_part_db._build_channel_map().items()}
        assert paths == {
            "MAG:PS:Current": {"system": "MAG", "device": "PS", "signal": "Current"},
            "RACK:Temp": {"system": "", "device": "RACK", "signal": "Temp"},
        }

    def test_all_optional_next_instances_paths(self, all_optional_next_instances_db) -> None:
        paths = {
            k: v["path"] for k, v in all_optional_next_instances_db._build_channel_map().items()
        }
        assert paths == {
            "SR": {"system": "SR"},
            "SR:X": {"system": "SR", "signal": "X"},
            "SR:Y": {"system": "SR", "signal": "Y"},
            "BR:D1:Z": {"system": "BR", "device": "D1", "signal": "Z"},
            "BR:D2:Z": {"system": "BR", "device": "D2", "signal": "Z"},
        }

    def test_colliding_channel_names_collapse_to_the_last_writer(self, hier_db_factory) -> None:
        """Two tree positions naming the same channel leave only the later path.

        Characterized as-is, not endorsed: ``channel_map`` is a plain dict keyed
        by the emitted name, so a document whose naming pattern omits a
        distinguishing level silently loses channels. Nothing warns.
        """
        db = hier_db_factory(
            _document(
                levels=[
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                ],
                naming_pattern="{system}:{signal}",
                tree={"SR": {"A": {"Temp": {}}, "B": {"Temp": {}}}},
            ),
            "collision.json",
        )
        channel_map = db._build_channel_map()
        assert list(channel_map) == ["SR:Temp"]
        assert channel_map["SR:Temp"]["path"] == {
            "system": "SR",
            "device": "B",
            "signal": "Temp",
        }


class TestLeafWithChildrenAsymmetry:
    """Finding 1: ``_is_leaf: true`` + dict children behaves differently by level kind.

    The same node -- ``_is_leaf: true`` with ``RB``/``SP`` dict children -- emits
    three channels under a required level and one under an optional level. Under
    the optional level the optional-skip branch fires before the leaf branch, the
    node is re-assigned to the *next* level, every level is thereby consumed, and
    the children are never visited.

    Characterized as-is, not endorsed: the drop is silent and there is no
    document-level signal that ``RB``/``SP`` will not appear.
    """

    def test_required_level_emits_the_node_and_recurses_into_its_children(
        self, leaf_with_children_db
    ) -> None:
        channel_map = leaf_with_children_db._build_channel_map()
        assert list(channel_map) == [
            "SR:PS:SIGNAL-Y",
            "SR:PS:SIGNAL-Y:RB",
            "SR:PS:SIGNAL-Y:SP",
        ]
        # The node occupies ``signal``; its children occupy the next tree level.
        assert channel_map["SR:PS:SIGNAL-Y"]["path"] == {
            "system": "SR",
            "device": "PS",
            "signal": "SIGNAL-Y",
        }
        assert channel_map["SR:PS:SIGNAL-Y:RB"]["path"] == {
            "system": "SR",
            "device": "PS",
            "signal": "SIGNAL-Y",
            "suffix": "RB",
        }

    def test_optional_level_drops_the_children_and_shifts_the_node_one_level_down(
        self, optional_leaf_with_children_db
    ) -> None:
        channel_map = optional_leaf_with_children_db._build_channel_map()
        assert list(channel_map) == ["SR:PS:SIGNAL-Y", "SR:PS:PSU:Voltage"]
        # SIGNAL-Y landed on ``suffix``, not on the ``subdevice`` level it was
        # written under -- that is the level skip, and it is why nothing is left
        # for RB/SP to occupy.
        assert optional_leaf_with_children_db._build_channel_map()["SR:PS:SIGNAL-Y"]["path"] == {
            "system": "SR",
            "device": "PS",
            "suffix": "SIGNAL-Y",
        }

    def test_the_two_shapes_disagree_about_the_same_node(
        self, leaf_with_children_db, optional_leaf_with_children_db
    ) -> None:
        """Stated directly: RB/SP survive under the required level and not the optional one."""
        required = set(leaf_with_children_db._build_channel_map())
        optional = set(optional_leaf_with_children_db._build_channel_map())

        assert {"SR:PS:SIGNAL-Y:RB", "SR:PS:SIGNAL-Y:SP"} <= required
        assert not {c for c in optional if c.endswith((":RB", ":SP"))}
        # The node's own channel is emitted either way.
        assert "SR:PS:SIGNAL-Y" in required & optional

    def test_optional_skip_at_the_final_level_emits_only_the_parent(self, hier_db_factory) -> None:
        """With no next level to skip into, the child expands in place and vanishes.

        Characterized as-is, not endorsed: ``Beat`` contributes nothing; the only
        channel is its parent ``SR``.
        """
        db = hier_db_factory(
            _document(
                levels=[
                    {"name": "system", "type": "tree"},
                    {"name": "sub", "type": "tree", "optional": True},
                ],
                naming_pattern="{system}:{sub}",
                tree={"SR": {"Beat": {"_is_leaf": True}}},
            ),
            "optional_skip_last.json",
        )
        channel_map = db._build_channel_map()
        assert list(channel_map) == ["SR"]
        assert channel_map["SR"]["path"] == {"system": "SR"}


class TestAllRemainingOptionalTail:
    """Finding 2: the ``tree`` arm of the all-remaining-optional check never yields a leaf.

    When every level from the current position on is optional, a node with an
    ``instances``-typed next level is treated as a leaf if it has no matching
    container -- so ``SR`` emits a bare ``SR`` channel. The ``tree``-typed
    counterpart re-tests the same "has dict children" predicate that a node must
    already have passed to reach that point, so it can only answer "not a leaf".

    Characterized as-is, not endorsed. The assertions below pin the *consequence*
    -- no bare parent channel on the tree side -- rather than the arm's return
    value, and hold whether or not the unreachable-True branch still exists.
    """

    def test_instances_tail_emits_the_bare_parent_channel(
        self, all_optional_next_instances_db
    ) -> None:
        channels = list(all_optional_next_instances_db._build_channel_map())
        assert "SR" in channels
        assert channels == ["SR", "SR:X", "SR:Y", "BR:D1:Z", "BR:D2:Z"]

    def test_tree_tail_never_emits_the_bare_parent_channel(self, all_optional_next_tree_db) -> None:
        channels = list(all_optional_next_tree_db._build_channel_map())
        assert "SR" not in channels
        assert channels == ["SR:PSU:Voltage", "SR:PSU:Current"]

    def test_leaf_detection_answers_for_each_tail_kind(
        self, all_optional_next_instances_db, all_optional_next_tree_db
    ) -> None:
        """The leaf verdicts behind the two inventories above.

        ``SR`` in the instances document has no ``DEVICE`` container, so it is a
        leaf; ``BR`` has one, so it is not. In the tree document ``SR`` is never
        a leaf, and neither is ``PSU`` -- only childless nodes are.
        """
        instances_tree = all_optional_next_instances_db.tree
        assert all_optional_next_instances_db._is_leaf_node(instances_tree["SR"], 1) is True
        assert all_optional_next_instances_db._is_leaf_node(instances_tree["BR"], 1) is False

        tree_tree = all_optional_next_tree_db.tree
        assert all_optional_next_tree_db._is_leaf_node(tree_tree["SR"], 1) is False
        assert all_optional_next_tree_db._is_leaf_node(tree_tree["SR"]["PSU"], 2) is False
        assert all_optional_next_tree_db._is_leaf_node(tree_tree["SR"]["PSU"]["Voltage"], 2) is True

    def test_nodes_with_only_scalar_children_are_leaves(self, scalar_leaves_db) -> None:
        """Scalars are invisible to the dict guards, so ``BPM`` is childless here."""
        assert scalar_leaves_db._is_leaf_node(scalar_leaves_db.tree["SR"]["BPM"], 1) is True
        assert list(scalar_leaves_db._build_channel_map()) == ["SR:BPM", "SR:QUAD:Setpoint"]


class TestStatisticsBucketing:
    """Finding 3: ``get_statistics`` buckets by raw tree key, not by emitted name."""

    def test_channel_part_systems_report_zero_channels(self, channel_part_db) -> None:
        """Characterized as-is, not endorsed: both counts are wrong.

        ``total_channels`` is 2 and both channels belong to one of the two listed
        systems, yet each system reports ``channel_count: 0``. The bucketing
        compares the stored path value (``MAG``, ``""``) against the tree key
        (``Magnets``, ``Building-1``) and never matches.
        """
        stats = channel_part_db.get_statistics()
        assert stats["total_channels"] == 2
        assert stats["systems"] == [
            {"name": "Magnets", "channel_count": 0},
            {"name": "Building-1", "channel_count": 0},
        ]

    def test_bucketing_is_correct_when_no_channel_part_is_used(self, tree_only_db) -> None:
        """The contrast case: without ``_channel_part`` the counts do add up."""
        stats = tree_only_db.get_statistics()
        assert stats["total_channels"] == 4
        assert stats["systems"] == [
            {"name": "SR", "channel_count": 3},
            {"name": "BR", "channel_count": 1},
        ]


class TestChannelPartRendering:
    """``_channel_part`` overrides and the separator cleanup they trigger."""

    def test_explicit_part_replaces_the_tree_key_in_the_name(self, channel_part_db) -> None:
        assert list(channel_part_db._build_channel_map()) == ["MAG:PS:Current", "RACK:Temp"]

    def test_empty_part_drops_the_component_and_the_leading_separator(
        self, channel_part_db
    ) -> None:
        """``Building-1`` contributes nothing, and the resulting ``:RACK:Temp`` is cleaned."""
        assert "RACK:Temp" in channel_part_db._build_channel_map()

    def test_empty_part_at_a_middle_level_collapses_the_doubled_separator(
        self, hier_db_factory
    ) -> None:
        doc = tree_only_doc()
        doc["tree"]["SR"]["BPM"]["_channel_part"] = ""
        db = hier_db_factory(doc, "mid_empty_part.json")
        # ``SR`` + empty + ``X`` would read ``SR::X``; the cleanup collapses it.
        assert list(db._build_channel_map()) == ["SR:X", "SR:Y", "SR:QUAD:Setpoint", "BR"]

    def test_two_empty_parts_leave_the_name_starting_at_the_first_real_component(
        self, hier_db_factory
    ) -> None:
        doc = tree_only_doc()
        doc["tree"]["SR"]["_channel_part"] = ""
        doc["tree"]["SR"]["BPM"]["_channel_part"] = ""
        db = hier_db_factory(doc, "two_empty_parts.json")
        assert list(db._build_channel_map()) == ["X", "Y", "QUAD:Setpoint", "BR"]

    def test_literal_pattern_text_is_kept_and_disappears_with_a_skipped_level(
        self, hier_db_factory
    ) -> None:
        """A literal prefix belongs to its placeholder: skip the level, lose the literal."""
        db = hier_db_factory(
            _document(
                levels=[
                    {"name": "system", "type": "tree"},
                    {"name": "sub", "type": "tree", "optional": True},
                    {"name": "signal", "type": "tree"},
                ],
                naming_pattern="{system}:S{sub}:{signal}",
                tree={"SR": {"Beat": {"_is_leaf": True}, "PSU": {"V": {}}}},
            ),
            "literal_prefix.json",
        )
        assert list(db._build_channel_map()) == ["SR:Beat", "SR:SPSU:V"]


class TestSeparatorOverrides:
    """``_separator`` metadata, in the tree expansion and in the selections path."""

    def test_overrides_apply_per_subtree_and_do_not_leak_to_siblings(
        self, separator_override_db
    ) -> None:
        """``SR`` sets ``-`` and ``BPM`` sets ``_``; ``QUAD`` keeps the pattern's ``:``."""
        assert list(separator_override_db._build_channel_map()) == [
            "SR-BPM_X:RB",
            "SR-BPM_X:SP",
            "SR-QUAD:Setpoint:RB",
        ]

    def test_selections_reproduce_the_expansion_names(self, separator_override_db) -> None:
        """The two collection paths agree for this document."""
        assert separator_override_db.build_channels_from_selections(
            {"system": "SR", "device": "BPM", "signal": "X", "suffix": ["RB", "SP"]}
        ) == ["SR-BPM_X:RB", "SR-BPM_X:SP"]
        assert separator_override_db.build_channels_from_selections(
            {"system": "SR", "device": "QUAD", "signal": "Setpoint", "suffix": "RB"}
        ) == ["SR-QUAD:Setpoint:RB"]

    def test_separator_on_the_tree_root_itself_is_ignored(self, hier_db_factory) -> None:
        """Only nodes below the root can carry an override.

        Characterized as-is, not endorsed: the root's ``_separator`` is dropped
        without complaint because the guard requires a non-zero level index.
        """
        doc = separator_override_doc()
        doc["tree"]["_separator"] = "@"
        db = hier_db_factory(doc, "root_separator.json")
        assert list(db._build_channel_map()) == [
            "SR-BPM_X:RB",
            "SR-BPM_X:SP",
            "SR-QUAD:Setpoint:RB",
        ]

    def test_separator_with_no_later_tree_level_is_ignored(self, hier_db_factory) -> None:
        """An override binds to the next ``tree`` level; with none, it is dropped."""
        db = hier_db_factory(
            _document(
                levels=[
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "instances"},
                ],
                naming_pattern="{system}:{device}",
                tree={
                    "SR": {
                        "_separator": "~",
                        "DEVICE": {
                            "_expansion": {
                                "_type": "range",
                                "_pattern": "D{}",
                                "_range": [1, 2],
                            }
                        },
                    }
                },
            ),
            "separator_no_tree_level.json",
        )
        assert list(db._build_channel_map()) == ["SR:D1", "SR:D2"]

    def test_separator_on_an_instance_container_binds_to_the_following_tree_level(
        self, hier_db_factory
    ) -> None:
        doc = range_instances_doc()
        doc["tree"]["SR"]["BPM"]["DEVICE"]["_separator"] = "#"
        db = hier_db_factory(doc, "instances_separator.json")
        channels = list(db._build_channel_map())
        assert channels[:2] == ["SR:BPM:B01#X", "SR:BPM:B01#Y"]
        # QUAD, which sets no override, is untouched.
        assert "SR:QUAD:Q1:Setpoint" in channels
        assert db.build_channels_from_selections(
            {"system": "SR", "family": "BPM", "device": "B01", "signal": "X"}
        ) == ["SR:BPM:B01#X"]

    def test_separator_on_a_hybrid_expansion_node_applies_to_every_instance(
        self, hier_db_factory
    ) -> None:
        doc = hybrid_expansion_doc()
        doc["tree"]["SR"]["CH"]["_separator"] = "%"
        db = hier_db_factory(doc, "hybrid_separator.json")
        assert list(db._build_channel_map()) == [
            "SR:CH-1%Voltage",
            "SR:CH-1%Current",
            "SR:CH-2%Voltage",
            "SR:CH-2%Current",
            "SR:CH-3%Voltage",
            "SR:CH-3%Current",
            "SR:PS:Status",
        ]
        assert db.build_channels_from_selections(
            {"system": "SR", "device": "CH-2", "signal": "Voltage"}
        ) == ["SR:CH-2%Voltage"]


class TestBuildChannelsFromSelections:
    """The selections API, which has two production callers and so is load-bearing."""

    def test_single_values_and_lists_form_a_cartesian_product(self, tree_only_db) -> None:
        assert tree_only_db.build_channels_from_selections(
            {"system": "SR", "device": "BPM", "signal": ["X", "Y"]}
        ) == ["SR:BPM:X", "SR:BPM:Y"]
        assert tree_only_db.build_channels_from_selections(
            {"system": ["SR", "BR"], "device": ["BPM", "QUAD"], "signal": "X"}
        ) == ["SR:BPM:X", "SR:QUAD:X", "BR:BPM:X", "BR:QUAD:X"]

    def test_a_missing_required_level_raises_and_names_the_level(self, tree_only_db) -> None:
        with pytest.raises(ValueError, match="Required level 'signal' is missing"):
            tree_only_db.build_channels_from_selections({"system": "SR", "device": "BPM"})

    def test_an_omitted_optional_level_becomes_an_empty_component(
        self, optional_tree_level_db
    ) -> None:
        assert optional_tree_level_db.build_channels_from_selections(
            {"system": "SR", "device": "PS", "signal": "Heartbeat"}
        ) == ["SR:PS:Heartbeat"]
        assert optional_tree_level_db.build_channels_from_selections(
            {"system": "SR", "device": "PS", "subdevice": "", "signal": "Heartbeat"}
        ) == ["SR:PS:Heartbeat"]
        assert optional_tree_level_db.build_channels_from_selections(
            {"system": "SR", "device": "PS", "subdevice": "PSU", "signal": "Voltage"}
        ) == ["SR:PS:PSU:Voltage"]

    def test_an_empty_list_anywhere_collapses_the_whole_product(self, tree_only_db) -> None:
        """Characterized as-is, not endorsed: an empty selection returns ``[]`` silently.

        A required level given ``[]`` or ``None`` is not the same as omitting it
        -- omitting raises, supplying an empty container returns no channels and
        no explanation.
        """
        assert (
            tree_only_db.build_channels_from_selections(
                {"system": "SR", "device": "BPM", "signal": []}
            )
            == []
        )
        assert (
            tree_only_db.build_channels_from_selections(
                {"system": "SR", "device": "BPM", "signal": None}
            )
            == []
        )

    def test_an_empty_string_for_a_required_level_drops_the_component(self, tree_only_db) -> None:
        """Characterized as-is, not endorsed: a required level accepts ``""`` and vanishes."""
        assert tree_only_db.build_channels_from_selections(
            {"system": "SR", "device": "BPM", "signal": ""}
        ) == ["SR:BPM"]

    def test_an_empty_list_for_an_optional_level_still_collapses_the_product(
        self, optional_tree_level_db
    ) -> None:
        """Unlike ``""`` and omission, ``[]`` on an optional level yields nothing."""
        assert (
            optional_tree_level_db.build_channels_from_selections(
                {"system": "SR", "device": "PS", "subdevice": [], "signal": "Heartbeat"}
            )
            == []
        )

    def test_levels_absent_from_the_naming_pattern_are_ignored(self, hier_db_factory) -> None:
        db = hier_db_factory(
            _document(
                levels=[
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                ],
                naming_pattern="{system}:{signal}",
                tree={"SR": {"BPM": {"X": {}}}},
            ),
            "subset_pattern.json",
        )
        # ``device`` is required by the hierarchy but not by the pattern, so
        # omitting it does not raise.
        assert db.build_channels_from_selections({"system": "SR", "signal": "X"}) == ["SR:X"]

    def test_unknown_selection_keys_are_ignored(self, tree_only_db) -> None:
        assert tree_only_db.build_channels_from_selections(
            {"system": "SR", "device": "BPM", "signal": "X", "not_a_level": "Z"}
        ) == ["SR:BPM:X"]

    def test_values_absent_from_the_tree_are_rendered_anyway(self, tree_only_db) -> None:
        """Characterized as-is, not endorsed: selections are never validated.

        The method is a name formatter, not a lookup. It happily returns a
        channel name that corresponds to nothing in the tree and is not a key of
        ``channel_map``.
        """
        built = tree_only_db.build_channels_from_selections(
            {"system": "NOPE", "device": "NOPE", "signal": "NOPE"}
        )
        assert built == ["NOPE:NOPE:NOPE"]
        assert built[0] not in tree_only_db.channel_map

    def test_channel_part_trees_require_the_emitted_part_not_the_tree_key(
        self, channel_part_db
    ) -> None:
        """Characterized as-is, not endorsed: navigating by tree key produces a bad name.

        ``_build_channel_map`` maps ``Magnets`` to ``MAG``, but this method does
        no such translation -- passing the key a caller would have browsed with
        yields a channel that does not exist.
        """
        assert channel_part_db.build_channels_from_selections(
            {"system": "MAG", "device": "PS", "signal": "Current"}
        ) == ["MAG:PS:Current"]

        bogus = channel_part_db.build_channels_from_selections(
            {"system": "Magnets", "device": "Power Supplies", "signal": "Current"}
        )
        assert bogus == ["Magnets:Power Supplies:Current"]
        assert bogus[0] not in channel_part_db.channel_map

    def test_expanded_instances_are_selectable_by_their_generated_names(
        self, hybrid_expansion_db, range_instances_db
    ) -> None:
        assert hybrid_expansion_db.build_channels_from_selections(
            {"system": "SR", "device": "CH-2", "signal": "Voltage"}
        ) == ["SR:CH-2:Voltage"]
        assert range_instances_db.build_channels_from_selections(
            {"system": "SR", "family": "BPM", "device": "B02", "signal": ["X", "Y"]}
        ) == ["SR:BPM:B02:X", "SR:BPM:B02:Y"]

    def test_literal_pattern_text_is_applied_to_selections(self, hier_db_factory) -> None:
        db = hier_db_factory(
            _document(
                levels=[
                    {"name": "sector", "type": "tree"},
                    {"name": "device", "type": "tree"},
                ],
                naming_pattern="S{sector}:F{device}",
                tree={"01": {"A": {}}, "02": {"B": {}}},
            ),
            "literal_selections.json",
        )
        assert list(db._build_channel_map()) == ["S01:FA", "S02:FB"]
        assert db.build_channels_from_selections({"sector": ["01", "02"], "device": "A"}) == [
            "S01:FA",
            "S02:FA",
        ]

    @pytest.mark.parametrize("fixture_name", sorted(EXPECTED_CHANNELS))
    def test_replaying_each_stored_path_reproduces_its_channel_name(
        self, request, fixture_name: str
    ) -> None:
        """Feeding a channel's own ``path`` back in returns that channel name.

        This is the round-trip that ties the two entry points together. It holds
        for every channel whose path covers all required pattern levels --
        including the ones carrying empty strings or omitting *optional* levels
        -- and never returns a different name. The exceptions are listed in
        :data:`NON_REPLAYABLE_CHANNELS` and covered by the test below.
        """
        db = request.getfixturevalue(fixture_name)
        truncated = NON_REPLAYABLE_CHANNELS[fixture_name]
        replayed = {
            name: db.build_channels_from_selections(info["path"])
            for name, info in db._build_channel_map().items()
            if name not in truncated
        }
        assert replayed == {name: [name] for name in replayed}

    @pytest.mark.parametrize("fixture_name", sorted(EXPECTED_CHANNELS))
    def test_paths_that_stop_before_a_required_level_cannot_be_replayed(
        self, request, fixture_name: str
    ) -> None:
        """Characterized as-is, not endorsed: some emitted channels are unreproducible.

        A node that terminates early -- because it has no children, or because
        ``_is_leaf`` made it a channel mid-hierarchy -- gets a path that omits a
        *required* level. ``_build_channel_map`` is happy to emit such a channel,
        but replaying its own path through ``build_channels_from_selections``
        raises, because that method insists every required pattern level be
        supplied. The two entry points therefore disagree about which channels
        exist, and :data:`NON_REPLAYABLE_CHANNELS` is that disagreement in full.
        """
        db = request.getfixturevalue(fixture_name)
        channel_map = db._build_channel_map()
        expected = NON_REPLAYABLE_CHANNELS[fixture_name]

        raised = set()
        for name, info in channel_map.items():
            try:
                db.build_channels_from_selections(info["path"])
            except ValueError:
                raised.add(name)

        assert raised == expected
