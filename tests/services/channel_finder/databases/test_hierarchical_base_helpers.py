"""Direct-call tests for the pure helpers on ``_HierarchicalBase``.

The helpers exercised here are the cross-cutting utilities the hierarchical
mixins share: instance-name expansion, value coercion, and node/expansion
counting. Every one of them is pure -- it reads only its arguments -- so the
tests call them straight, with literal nodes for the ``staticmethod`` /
``classmethod`` helpers and a loaded database only where the helper is bound to
an instance. Each case pins the *emitted value*, including the fallback arms
that a schema-valid document never reaches through the loader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from osprey.services.channel_finder.databases._hierarchical_base import _HierarchicalBase

if TYPE_CHECKING:
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )


# ---------------------------------------------------------------------------
# _get_instance_names
# ---------------------------------------------------------------------------


def test_get_instance_names_range(tree_only_db: HierarchicalChannelDatabase) -> None:
    """``_type: range`` formats ``_pattern`` over the inclusive ``_range``."""
    names = tree_only_db._get_instance_names(
        {"_type": "range", "_pattern": "{:02d}", "_range": [1, 3]}
    )
    assert names == ["01", "02", "03"]


def test_get_instance_names_list(tree_only_db: HierarchicalChannelDatabase) -> None:
    """``_type: list`` returns ``_instances`` verbatim."""
    assert tree_only_db._get_instance_names({"_type": "list", "_instances": ["A", "B"]}) == [
        "A",
        "B",
    ]


@pytest.mark.parametrize(
    "expansion_def",
    [
        pytest.param({}, id="no-type"),
        pytest.param({"_type": "sequence"}, id="unknown-type"),
        pytest.param({"_type": None, "_instances": ["A"]}, id="null-type-with-instances"),
    ],
)
def test_get_instance_names_unknown_type_is_empty(
    tree_only_db: HierarchicalChannelDatabase, expansion_def: dict
) -> None:
    """An expansion whose ``_type`` matches neither arm expands to nothing.

    The final ``return []`` is the fallback: an unrecognised ``_type`` yields no
    instances rather than raising, and any sibling keys (``_instances`` here)
    are ignored because the dispatch never reaches their arm.
    """
    assert tree_only_db._get_instance_names(expansion_def) == []


# ---------------------------------------------------------------------------
# _ensure_list / _get_single_value
# ---------------------------------------------------------------------------


def test_ensure_list_passes_a_list_through_unchanged(
    tree_only_db: HierarchicalChannelDatabase,
) -> None:
    """A list is returned as the *same object*, not a copy."""
    original = ["X", "Y"]
    result = tree_only_db._ensure_list(original)
    assert result is original


def test_ensure_list_maps_none_to_empty(tree_only_db: HierarchicalChannelDatabase) -> None:
    """``None`` becomes ``[]`` -- an absent selection contributes no values."""
    assert tree_only_db._ensure_list(None) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("BPM", ["BPM"], id="str"),
        pytest.param(0, [0], id="falsy-int"),
        pytest.param(False, [False], id="false"),
    ],
)
def test_ensure_list_wraps_a_scalar(
    tree_only_db: HierarchicalChannelDatabase, value: object, expected: list
) -> None:
    """Any non-list, non-``None`` value is wrapped -- including falsy scalars."""
    assert tree_only_db._ensure_list(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(["A", "B"], "A", id="list-takes-first"),
        pytest.param([], None, id="empty-list-is-none"),
        pytest.param("A", "A", id="scalar-passthrough"),
        pytest.param(None, None, id="none-passthrough"),
    ],
)
def test_get_single_value(
    tree_only_db: HierarchicalChannelDatabase, value: object, expected: object
) -> None:
    """The list arm collapses to the first element, or ``None`` when empty."""
    assert tree_only_db._get_single_value(value) == expected


# ---------------------------------------------------------------------------
# _child_keys
# ---------------------------------------------------------------------------


def test_child_keys_drops_metadata_and_underscored_keys() -> None:
    """Known metadata keys and any other ``_``-prefixed key are not children."""
    node = {
        "_description": "Storage Ring",
        "_expansion": {"_type": "range"},
        "_channel_part": "SR",
        "_is_leaf": True,
        "_separator": "-",
        "_future_metadata": "ignored",
        "BPM": {},
        "QUAD": {},
    }
    assert _HierarchicalBase._child_keys(node) == ["BPM", "QUAD"]


def test_child_keys_of_a_childless_node_is_empty() -> None:
    """A node carrying only metadata has no children at all."""
    assert _HierarchicalBase._child_keys({"_description": "leaf"}) == []


# ---------------------------------------------------------------------------
# _expansion_instance_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expansion", "expected"),
    [
        pytest.param({"_range": [1, 8]}, 8, id="inclusive-range"),
        pytest.param({"_range": [3, 3]}, 1, id="single-element-range"),
        pytest.param({"_range": [5, 1]}, 0, id="inverted-range-floors-at-zero"),
        pytest.param({}, 1, id="empty-descriptor-defaults-to-one"),
    ],
)
def test_expansion_instance_count(expansion: dict, expected: int) -> None:
    """The count is the inclusive width of ``_range``, floored at zero.

    An empty descriptor falls back to the built-in ``[1, 1]`` default, so a node
    that declares ``_expansion`` without a range still multiplies by one rather
    than collapsing its subtree to zero channels.
    """
    assert _HierarchicalBase._expansion_instance_count(expansion) == expected


# ---------------------------------------------------------------------------
# _count_channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node",
    [
        pytest.param("SR:BPM:X", id="str"),
        pytest.param(42, id="int"),
        pytest.param(None, id="none"),
        pytest.param(["SR"], id="list"),
    ],
)
def test_count_channels_of_a_non_dict_is_zero(node: object) -> None:
    """A non-dict argument counts as nothing rather than raising.

    ``_count_channels`` recurses only into dict values, but it is also called
    directly on caller-supplied nodes; the guard keeps a scalar or a list from
    reaching the ``.items()`` call below it.
    """
    assert _HierarchicalBase._count_channels(node) == 0


def test_count_channels_counts_scalar_leaf_values() -> None:
    """Children whose value is a scalar each count as one channel.

    ``{"X": 1, "Y": 2}`` has no dict children, so the recursion never fires --
    the two scalars are counted by the ``else`` arm instead, and the node is not
    additionally counted as a childless leaf because ``_child_keys`` still sees
    ``X`` and ``Y``.
    """
    assert _HierarchicalBase._count_channels({"X": 1, "Y": 2}) == 2


def test_count_channels_mixes_scalar_and_dict_children() -> None:
    """A scalar sibling adds one on top of the recursion into a dict child."""
    node = {"_description": "Quadrupole", "units": "A", "Setpoint": {"_description": "sp"}}
    # "units" -> 1 (scalar arm); "Setpoint" -> 1 (childless dict).
    assert _HierarchicalBase._count_channels(node) == 2


def test_count_channels_of_a_metadata_only_node_is_one() -> None:
    """A node with no children is itself the channel."""
    assert _HierarchicalBase._count_channels({"_description": "Beam current"}) == 1


def test_count_channels_multiplies_dict_children_by_their_expansion() -> None:
    """A dict child carrying ``_expansion`` multiplies its own subtree count."""
    node = {
        "BPM": {
            "_expansion": {"_type": "range", "_range": [1, 4]},
            "X": {},
            "Y": {},
        },
    }
    assert _HierarchicalBase._count_channels(node) == 8


def test_count_channels_on_a_loaded_tree(scalar_leaves_db: HierarchicalChannelDatabase) -> None:
    """The scalar arm is reached through a real loaded document, not just literals.

    ``scalar_leaves`` holds ``SR.BPM = {"X": 1, "Y": 2}`` (two scalars) and
    ``SR.QUAD.Setpoint = {"units": "A"}`` (one scalar), which is the arithmetic
    the tree preview reports for that shape.
    """
    assert _HierarchicalBase._count_channels(scalar_leaves_db.tree) == 3
