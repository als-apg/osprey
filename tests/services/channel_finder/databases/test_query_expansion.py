"""Expansion-side behaviour of ``_hierarchical_query``.

Covers the level-name accessor and the instance-expansion half of the query
mixin: how an ``instances`` level turns into a list of options, what each
expansion ``_type`` contributes to the option descriptions, and what happens at
a branch where the level's container is absent or carries no ``_expansion``.

Tree navigation itself (``_navigate_to_node`` / ``_extract_tree_options``) is
characterised elsewhere; here ``get_options_at_level`` is only the public seam
that reaches ``_get_expansion_options`` and ``_expand_instances``.

Loaded databases and the ``hier_db_factory`` come from the package ``conftest``
as fixtures. The two documents built locally exist for shapes no shared fixture
provides -- a *required* instance level whose container is missing (or
expansion-less) in a sibling branch that validation never walks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.services.channel_finder.conftest import _document

if TYPE_CHECKING:
    from collections.abc import Callable

    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )


def _sibling_branch_missing_container_doc() -> dict:
    """A *required* instance level whose container exists in only one branch.

    ``_validate_instance_level`` refuses to load a document whose required
    instance level has no container -- but ``_find_level_container`` walks only
    the *first* tree child it finds. Listing ``SR`` (which owns ``DEVICE``)
    before ``BR`` (which does not) therefore loads cleanly, while navigating to
    ``BR`` reaches the "no key matches the level name" fall-through in
    ``_get_expansion_options`` without the level being optional.

    ``BR`` contributes no channels: the instances arm of ``_build_channel_map``
    finds no expansion definition there either.

    Channels: ``SR:D1:X``, ``SR:D2:X``.
    """
    return _document(
        levels=[
            {"name": "system", "type": "tree"},
            {"name": "device", "type": "instances"},
            {"name": "signal", "type": "tree"},
        ],
        naming_pattern="{system}:{device}:{signal}",
        tree={
            "SR": {
                "_description": "Storage Ring",
                "DEVICE": {
                    "_expansion": {"_type": "range", "_pattern": "D{}", "_range": [1, 2]},
                    "X": {"_description": "Horizontal position"},
                },
            },
            "BR": {"_description": "Booster - no DEVICE container", "Status": {}},
        },
    )


def _sibling_branch_container_without_expansion_doc() -> dict:
    """A sibling branch whose container matches the level name but has no expansion.

    Same first-branch-only validation loophole as
    :func:`_sibling_branch_missing_container_doc`, one step further along: ``BR``
    *does* own a ``DEVICE`` key, so ``_get_expansion_options`` matches it on
    name and then fails the ``"_expansion" in value`` guard -- the loop keeps
    scanning and still falls through to the empty result. A document where the
    *first* branch looked like this would be rejected outright with
    "container missing '_expansion' definition".

    Channels: ``SR:D1:X``, ``SR:D2:X``.
    """
    doc = _sibling_branch_missing_container_doc()
    doc["tree"]["BR"] = {
        "_description": "Booster - DEVICE container, no expansion",
        "DEVICE": {"X": {"_description": "Horizontal position"}},
    }
    return doc


# ---------------------------------------------------------------------------
# get_hierarchy_definition
# ---------------------------------------------------------------------------


def test_get_hierarchy_definition_returns_level_names_in_order(
    tree_only_db: HierarchicalChannelDatabase,
) -> None:
    """The accessor reports the declared level names, in declaration order."""
    assert tree_only_db.get_hierarchy_definition() == ["system", "device", "signal"]


def test_get_hierarchy_definition_includes_instance_levels(
    range_instances_db: HierarchicalChannelDatabase,
) -> None:
    """Instance-typed levels are reported alongside tree levels, undistinguished."""
    assert range_instances_db.get_hierarchy_definition() == [
        "system",
        "family",
        "device",
        "signal",
    ]


def test_get_hierarchy_definition_returns_a_defensive_copy(
    tree_only_db: HierarchicalChannelDatabase,
) -> None:
    """Callers get a fresh list; mutating it leaves the database untouched."""
    first = tree_only_db.get_hierarchy_definition()
    assert first is not tree_only_db.hierarchy_levels

    first.append("intruder")

    assert tree_only_db.hierarchy_levels == ["system", "device", "signal"]
    assert tree_only_db.get_hierarchy_definition() == ["system", "device", "signal"]


# ---------------------------------------------------------------------------
# _expand_instances: the two expansion types
# ---------------------------------------------------------------------------


def test_list_expansion_returns_instances_in_declaration_order(
    list_instances_db: HierarchicalChannelDatabase,
) -> None:
    """``_type: "list"`` yields ``_instances`` verbatim, order preserved.

    Descriptions are empty: unlike the range arm, the list arm has no index to
    describe an instance by.
    """
    options = list_instances_db.get_options_at_level("unit", {"system": "RF"})

    assert options == [
        {"name": "MAIN", "description": ""},
        {"name": "BACKUP", "description": ""},
        {"name": "TEST", "description": ""},
    ]


def test_list_expansion_options_match_the_expanded_channel_names(
    list_instances_db: HierarchicalChannelDatabase,
) -> None:
    """Every offered list instance really does appear in the flat channel map."""
    options = list_instances_db.get_options_at_level("unit", {"system": "RF"})

    for option in options:
        assert f"RF:{option['name']}:Power" in list_instances_db.channel_map


def test_range_expansion_numbers_its_descriptions(
    range_instances_db: HierarchicalChannelDatabase,
) -> None:
    """``_type: "range"`` formats ``_pattern`` per index and describes each by index."""
    options = range_instances_db.get_options_at_level("device", {"system": "SR", "family": "BPM"})

    assert options == [
        {"name": "B01", "description": "Instance 1"},
        {"name": "B02", "description": "Instance 2"},
        {"name": "B03", "description": "Instance 3"},
    ]


def test_range_expansion_is_read_per_container_not_per_level(
    range_instances_db: HierarchicalChannelDatabase,
) -> None:
    """A second branch at the same level expands under its own pattern and range."""
    options = range_instances_db.get_options_at_level("device", {"system": "SR", "family": "QUAD"})

    assert [option["name"] for option in options] == ["Q1", "Q2"]


# ---------------------------------------------------------------------------
# _get_expansion_options: branches with no usable container
# ---------------------------------------------------------------------------


def test_optional_instance_level_without_container_yields_no_options(
    optional_instance_level_db: HierarchicalChannelDatabase,
) -> None:
    """An optional instance level offers nothing in a branch that omits its container."""
    assert optional_instance_level_db.get_options_at_level("device", {"system": "BR"}) == []


def test_optional_instance_level_still_expands_where_the_container_exists(
    optional_instance_level_db: HierarchicalChannelDatabase,
) -> None:
    """The empty result is branch-local: the sibling branch expands normally."""
    options = optional_instance_level_db.get_options_at_level("device", {"system": "SR"})

    assert options == [
        {"name": "D1", "description": "Instance 1"},
        {"name": "D2", "description": "Instance 2"},
    ]


def test_required_instance_level_yields_no_options_in_an_unvalidated_branch(
    hier_db_factory: Callable[..., HierarchicalChannelDatabase],
) -> None:
    """A required level offers nothing where the container is missing.

    Validation only walks the first tree branch, so this shape loads even though
    ``BR`` has no ``DEVICE`` container -- the level being required is no
    guarantee that every branch can expand it.
    """
    db = hier_db_factory(_sibling_branch_missing_container_doc(), "sibling_missing.json")

    assert db.get_options_at_level("device", {"system": "BR"}) == []
    assert [option["name"] for option in db.get_options_at_level("device", {"system": "SR"})] == [
        "D1",
        "D2",
    ]


def test_instance_container_without_expansion_yields_no_options(
    hier_db_factory: Callable[..., HierarchicalChannelDatabase],
) -> None:
    """A name-matched container with no ``_expansion`` is skipped, not raised on."""
    db = hier_db_factory(
        _sibling_branch_container_without_expansion_doc(), "sibling_no_expansion.json"
    )

    assert db.get_options_at_level("device", {"system": "BR"}) == []


def test_branches_without_expansion_contribute_no_channels(
    hier_db_factory: Callable[..., HierarchicalChannelDatabase],
) -> None:
    """The empty option list is consistent with the map: such branches yield nothing."""
    db = hier_db_factory(_sibling_branch_missing_container_doc(), "sibling_missing_map.json")

    assert sorted(db.channel_map) == ["SR:D1:X", "SR:D2:X"]


def test_local_documents_are_rebuilt_per_call() -> None:
    """Mutating a built document does not leak into the next build.

    :func:`_sibling_branch_container_without_expansion_doc` derives itself by
    overwriting part of its sibling's result, so freshness is load-bearing here
    rather than incidental.
    """
    mutated = _sibling_branch_missing_container_doc()
    mutated["tree"]["BR"]["Status"] = {"_description": "touched"}

    assert _sibling_branch_missing_container_doc()["tree"]["BR"]["Status"] == {}
    assert "DEVICE" not in _sibling_branch_missing_container_doc()["tree"]["BR"]
