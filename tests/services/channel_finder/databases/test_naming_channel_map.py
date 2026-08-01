"""Recursion contracts of ``_hierarchical_naming._build_channel_map``.

This file is deliberately narrow. ``test_naming_characterization`` already pins
the emitted channel list and ``channel_map`` payload for every shape in the
shared fixture family, the ``_is_leaf: true``-with-children asymmetry, the
all-remaining-optional tail, and ``_separator`` collection at both depths. None
of that is repeated here.

What is added are three contracts of the recursion itself that no inventory
assertion reaches:

1. **Recursion depth is bounded by the level count.** ``expand_tree`` is never
   entered below ``len(hierarchy_levels)``, and a node handed to it at exactly
   that depth is always treated as a leaf -- whatever children it has. This is
   why the second ``if level_idx >= len(self.hierarchy_levels): return`` guard,
   the one *after* the leaf block, can never fire: reaching it requires a
   non-leaf node at a depth where every node is a leaf.
2. **The ``instances`` container lookup** matches by case-insensitive key, skips
   non-dict values, skips name matches that carry no ``_expansion``, and
   navigates *into* the container -- so siblings of the container are dropped.
3. **A leaf with dict children and no ``tree`` level left** does not stop after
   emitting its own channel; it falls through and is processed again as an
   ``instances`` level at the same position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.services.channel_finder.conftest import _document, leaf_with_children_doc

if TYPE_CHECKING:
    from collections.abc import Callable

    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )


#: A two-instance range, reused by every ``instances``-level document below.
RANGE_EXPANSION = {"_type": "range", "_pattern": "D{}", "_range": [1, 2]}

#: ``system(tree) -> device(instances) -> signal(tree)``.
INSTANCE_LEVELS = [
    {"name": "system", "type": "tree"},
    {"name": "device", "type": "instances"},
    {"name": "signal", "type": "tree"},
]

#: The same three levels with ``device`` optional, which is what makes
#: ``_hierarchical_loading._validate_instance_level`` return early on a first
#: branch that has no container and leave later branches unvalidated.
OPTIONAL_INSTANCE_LEVELS = [
    {"name": "system", "type": "tree"},
    {"name": "device", "type": "instances", "optional": True},
    {"name": "signal", "type": "tree"},
]

INSTANCE_PATTERN = "{system}:{device}:{signal}"


def _record_leaf_calls(db: HierarchicalChannelDatabase) -> list[tuple[int, bool]]:
    """Wrap ``db._is_leaf_node`` so every call records ``(level_idx, verdict)``.

    The wrapper is installed on the instance only, and the returned list fills
    in during the next ``_build_channel_map()`` call. ``_build_channel_map``
    consults ``_is_leaf_node`` exactly once per ``expand_tree`` entry, so the
    recorded indices are the recursion depths the traversal visited.
    """
    calls: list[tuple[int, bool]] = []
    unbound = type(db)._is_leaf_node

    def spy(node: dict, level_idx: int) -> bool:
        verdict = unbound(db, node, level_idx)
        calls.append((level_idx, verdict))
        return verdict

    db._is_leaf_node = spy
    return calls


class TestRecursionDepthIsBounded:
    """No node is ever expanded past the last hierarchy level.

    ``_build_channel_map`` carries two identical ``level_idx >=
    len(hierarchy_levels)`` returns, one inside the leaf block and one after it.
    The tests here pin the invariant that makes the second one dead code: at that
    depth ``_is_leaf_node`` answers True unconditionally, so the leaf block runs
    and returns, and control never reaches the trailing guard.
    """

    def test_traversal_never_recurses_below_the_last_level(self, leaf_with_children_db) -> None:
        """The deepest ``expand_tree`` entry sits at exactly ``len(levels)``.

        ``leaf_with_children_db`` is the shape that goes deepest: ``SIGNAL-Y``
        emits at the ``signal`` level and then re-enters recursion with
        ``RB``/``SP`` bound to ``suffix``, the last level, so those calls arrive
        with ``level_idx == 4``.
        """
        calls = _record_leaf_calls(leaf_with_children_db)
        leaf_with_children_db._build_channel_map()

        n_levels = len(leaf_with_children_db.hierarchy_levels)
        assert calls, "the traversal never ran"
        assert max(idx for idx, _ in calls) == n_levels
        assert not [idx for idx, _ in calls if idx > n_levels]

    def test_every_node_at_the_last_level_is_treated_as_a_leaf(self, leaf_with_children_db) -> None:
        """At full depth the verdict is True for every node the traversal sees."""
        calls = _record_leaf_calls(leaf_with_children_db)
        leaf_with_children_db._build_channel_map()

        n_levels = len(leaf_with_children_db.hierarchy_levels)
        at_full_depth = [verdict for idx, verdict in calls if idx >= n_levels]
        assert at_full_depth, "no node reached the last level"
        assert all(at_full_depth)

    def test_children_below_the_last_level_are_dropped(self, hier_db_factory) -> None:
        """A node at full depth stops even when it has dict children.

        ``RB`` here owns a ``DEEP`` child, but ``suffix`` was the last level, so
        the traversal emits ``...:RB`` and returns without visiting it.

        Characterized as-is, not endorsed: the drop is silent.
        """
        doc = leaf_with_children_doc()
        doc["tree"]["SR"]["PS"]["SIGNAL-Y"]["RB"]["DEEP"] = {"_description": "one too far"}
        db = hier_db_factory(doc, "below_last_level.json")

        assert list(db._build_channel_map()) == [
            "SR:PS:SIGNAL-Y",
            "SR:PS:SIGNAL-Y:RB",
            "SR:PS:SIGNAL-Y:SP",
        ]

    def test_a_node_with_children_is_a_leaf_once_the_levels_run_out(
        self, leaf_with_children_db
    ) -> None:
        """The direct statement of the invariant, on the only shape that could break it.

        A node with dict children and no ``_is_leaf`` marker is the one input
        that fails the first two leaf criteria. Asked about it at full depth,
        ``_is_leaf_node`` still answers True -- which is what leaves the trailing
        ``level_idx >= len(levels)`` guard unreachable.
        """
        n_levels = len(leaf_with_children_db.hierarchy_levels)
        node = leaf_with_children_db.tree["SR"]

        assert "_is_leaf" not in node
        assert any(isinstance(v, dict) for k, v in node.items() if not k.startswith("_"))
        assert leaf_with_children_db._is_leaf_node(node, n_levels) is True
        assert leaf_with_children_db._is_leaf_node(node, 0) is False


class TestInstanceContainerLookup:
    """How the ``instances`` arm finds the container that carries ``_expansion``."""

    @pytest.mark.parametrize("container_key", ["DEVICE", "device", "DeViCe"])
    def test_the_container_key_is_matched_case_insensitively(
        self, hier_db_factory: Callable[..., HierarchicalChannelDatabase], container_key: str
    ) -> None:
        """Any casing of the level name names the container."""
        db = hier_db_factory(
            _document(
                levels=INSTANCE_LEVELS,
                naming_pattern=INSTANCE_PATTERN,
                tree={"SR": {container_key: {"_expansion": RANGE_EXPANSION, "X": {}}}},
            ),
            f"container_{container_key}.json",
        )
        assert list(db._build_channel_map()) == ["SR:D1:X", "SR:D2:X"]

    def test_a_non_dict_value_with_the_container_name_is_skipped(self, hier_db_factory) -> None:
        """The scalar ``DEVICE`` does not shadow the real container."""
        db = hier_db_factory(
            _document(
                levels=INSTANCE_LEVELS,
                naming_pattern=INSTANCE_PATTERN,
                tree={
                    "SR": {
                        "DEVICE": 7,
                        "device": {"_expansion": RANGE_EXPANSION, "X": {}},
                    }
                },
            ),
            "scalar_shadow.json",
        )
        assert list(db._build_channel_map()) == ["SR:D1:X", "SR:D2:X"]

    def test_siblings_of_the_container_are_never_visited(self, hier_db_factory) -> None:
        """Expansion navigates *into* the container, abandoning everything beside it.

        Characterized as-is, not endorsed: ``Status`` sits next to the container
        and produces no channel, with no document-level signal that it will not.
        """
        db = hier_db_factory(
            _document(
                levels=INSTANCE_LEVELS,
                naming_pattern=INSTANCE_PATTERN,
                tree={
                    "SR": {
                        "Status": {},
                        "DEVICE": {"_expansion": RANGE_EXPANSION, "X": {}},
                    }
                },
            ),
            "sibling_outside_container.json",
        )
        channels = list(db._build_channel_map())
        assert channels == ["SR:D1:X", "SR:D2:X"]
        assert "SR:Status" not in channels

    def test_a_container_without_an_expansion_yields_nothing(self, hier_db_factory) -> None:
        """A name match with no ``_expansion`` leaves the branch empty.

        A *required* instance level cannot reach this: the loader rejects the
        document. The level is optional here so validation stops at ``AA`` --
        the first branch, which has no container at all -- and never inspects
        ``BB``. ``CC`` is the control: the empty branch does not disturb it.
        """
        db = hier_db_factory(
            _document(
                levels=OPTIONAL_INSTANCE_LEVELS,
                naming_pattern=INSTANCE_PATTERN,
                tree={
                    "AA": {"_description": "no container at all", "Status": {}},
                    "BB": {"DEVICE": {"X": {}}},
                    "CC": {"DEVICE": {"_expansion": RANGE_EXPANSION, "X": {}}},
                },
            ),
            "container_without_expansion.json",
        )
        assert list(db._build_channel_map()) == ["CC:D1:X", "CC:D2:X"]

    def test_the_scan_continues_past_a_name_match_that_has_no_expansion(
        self, hier_db_factory
    ) -> None:
        """The lookup stops at the first name match *carrying* ``_expansion``.

        ``BB`` holds two keys that both match ``device`` case-insensitively.
        ``DEVICE`` comes first and has no ``_expansion``, so the scan keeps
        going and expands from ``device`` instead.
        """
        db = hier_db_factory(
            _document(
                levels=OPTIONAL_INSTANCE_LEVELS,
                naming_pattern=INSTANCE_PATTERN,
                tree={
                    "AA": {"_description": "no container at all", "Status": {}},
                    "BB": {
                        "DEVICE": {"Y": {}},
                        "device": {"_expansion": RANGE_EXPANSION, "X": {}},
                    },
                },
            ),
            "scan_past_name_match.json",
        )
        channels = list(db._build_channel_map())
        assert channels == ["BB:D1:X", "BB:D2:X"]
        # ``Y`` lived in the container that was scanned past, so it is gone.
        assert not [c for c in channels if c.endswith(":Y")]


class TestLeafWithChildrenAndNoTreeLevelLeft:
    """The third exit from the leaf-with-children block: no ``tree`` level remains.

    ``test_naming_characterization`` pins the two exits that return -- children
    re-assigned to a later tree level, and the optional-level skip. When no
    ``tree`` level is left at or after the current position, the block instead
    falls *through*, and the node is processed a second time by the level
    handling below it.
    """

    def test_the_node_emits_its_channel_and_is_expanded_again_as_instances(
        self, hier_db_factory
    ) -> None:
        """``PS`` is both a channel and the parent of the ``num`` instances.

        ``num`` is the only level after ``device`` and it is ``instances``-typed,
        so the leaf block finds no next tree level and does not return. Control
        reaches the ``instances`` arm at the same ``level_idx``, which finds
        ``NUM`` and expands it.
        """
        db = hier_db_factory(
            _document(
                levels=[
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "tree"},
                    {"name": "num", "type": "instances"},
                ],
                naming_pattern="{system}:{device}:{num}",
                tree={"SR": {"PS": {"_is_leaf": True, "NUM": {"_expansion": RANGE_EXPANSION}}}},
            ),
            "leaf_no_tree_level_left.json",
        )
        channel_map = db._build_channel_map()

        assert list(channel_map) == ["SR:PS", "SR:PS:D1", "SR:PS:D2"]
        # The node's own channel stops at ``device``; the instances carry ``num``.
        assert channel_map["SR:PS"]["path"] == {"system": "SR", "device": "PS"}
        assert channel_map["SR:PS:D1"]["path"] == {
            "system": "SR",
            "device": "PS",
            "num": "D1",
        }
