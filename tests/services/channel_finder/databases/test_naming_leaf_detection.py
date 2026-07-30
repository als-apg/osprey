"""Unit contracts for the naming mixin's per-node decisions.

``test_naming_characterization`` pins what ``_hierarchical_naming`` *emits*: the
channel inventory of every fixture shape, the stored paths, and the return value
of ``build_channels_from_selections``. This module works one level down, on the
three decisions those inventories are assembled from:

* ``_build_channel_with_separators``' pattern-completeness guard, which raises
  ``KeyError`` for a placeholder with no value in the path;
* ``_is_leaf_node``'s explicit ``_is_leaf`` marker and its "all levels consumed"
  arm, asserted as verdicts rather than through their emitted consequences;
* ``_get_channel_part``'s three-way lookup.

It closes with the one ``build_channels_from_selections`` behaviour the
characterization suite does not reach: a list-valued selection collects its
separator overrides from the first value only and then applies them to all of
them.
"""

from __future__ import annotations

import re

import pytest
from tests.services.channel_finder.conftest import leaf_with_children_doc


class TestPatternCompletenessGuard:
    """``_build_channel_with_separators`` raises when a placeholder has no value.

    The guard is documented on the method (``Raises: KeyError``) but no *document*
    can reach it. ``_hierarchical_loading._validate_naming_pattern`` rejects any
    placeholder that is not a hierarchy level, and after that check both callers
    supply a value for every remaining placeholder: ``build_channels_from_selections``
    zips one selection list per pattern level, and ``_build_channel_map`` fills the
    missing ones with ``""``. What the guard actually catches is a naming pattern
    that drifts *after* load, away from the levels it was validated against.
    """

    def test_a_pattern_level_absent_from_the_path_raises_naming_the_level(
        self, tree_only_db
    ) -> None:
        with pytest.raises(KeyError) as excinfo:
            tree_only_db._build_channel_with_separators({"system": "SR", "device": "BPM"}, {})

        assert excinfo.value.args[0] == "signal"

    def test_an_empty_string_is_a_value_and_does_not_raise(self, tree_only_db) -> None:
        """The guard tests membership, not truthiness -- that is what lets optional
        levels reach the method as ``""`` and be dropped from the name instead."""
        assert (
            tree_only_db._build_channel_with_separators(
                {"system": "SR", "device": "BPM", "signal": ""}, {}
            )
            == "SR:BPM"
        )

    def test_pattern_and_levels_agree_on_a_loaded_database(self, tree_only_db) -> None:
        """Why neither caller can reach the guard: the two sets coincide after load."""
        placeholders = set(re.findall(r"\{(\w+)\}", tree_only_db.naming_pattern))
        assert placeholders == set(tree_only_db._get_pattern_levels())

    def test_a_pattern_drifting_past_load_raises_from_the_selections_api(
        self, tree_only_db
    ) -> None:
        """A placeholder added after validation is dropped by ``_get_pattern_levels``
        -- so no selection list is built for it and the path reaches the guard short."""
        tree_only_db.naming_pattern = "{system}:{device}:{signal}:{revision}"

        with pytest.raises(KeyError) as excinfo:
            tree_only_db.build_channels_from_selections(
                {"system": "SR", "device": "BPM", "signal": "X"}
            )

        assert excinfo.value.args[0] == "revision"

    def test_the_same_drift_raises_from_the_tree_expansion(self, tree_only_db) -> None:
        """``_build_channel_map`` fills only pattern levels, so it too falls short."""
        tree_only_db.naming_pattern = "{system}:{device}:{signal}:{revision}"

        with pytest.raises(KeyError) as excinfo:
            tree_only_db._build_channel_map()

        assert excinfo.value.args[0] == "revision"


class TestExplicitLeafMarker:
    """``_is_leaf: true`` and the "all levels consumed" arm, as leaf verdicts.

    The characterization suite asserts what a marked node *emits*; these assert
    that the marker is what decides it. ``SIGNAL-Y`` sits at a required, non-final
    level and has dict children, so every other arm of ``_is_leaf_node`` answers
    "not a leaf" -- removing the marker flips both the verdict and the inventory.
    """

    def test_the_marker_makes_a_node_with_children_a_leaf_mid_hierarchy(
        self, leaf_with_children_db
    ) -> None:
        node = leaf_with_children_db.tree["SR"]["PS"]["SIGNAL-Y"]
        assert leaf_with_children_db._is_leaf_node(node, 2) is True

    def test_without_the_marker_the_node_is_not_a_leaf_and_loses_its_own_channel(
        self, hier_db_factory
    ) -> None:
        doc = leaf_with_children_doc()
        del doc["tree"]["SR"]["PS"]["SIGNAL-Y"]["_is_leaf"]
        db = hier_db_factory(doc, "leaf_marker_removed.json")

        assert db._is_leaf_node(db.tree["SR"]["PS"]["SIGNAL-Y"], 2) is False
        # The suffix variants survive; only the bare ``SR:PS:SIGNAL-Y`` is gone.
        assert list(db.channel_map) == ["SR:PS:SIGNAL-Y:RB", "SR:PS:SIGNAL-Y:SP"]

    def test_a_false_marker_leaves_the_childless_rule_in_charge(
        self, leaf_with_children_db
    ) -> None:
        """The marker can only force ``True``; ``_is_leaf: false`` is not a veto."""
        childless = {"_is_leaf": False, "_description": "Marked not-a-leaf, but has no children"}
        assert leaf_with_children_db._is_leaf_node(childless, 0) is True

    def test_a_consumed_hierarchy_makes_even_a_parent_node_a_leaf(self, tree_only_db) -> None:
        """``BPM`` has dict children either way; only the level index differs."""
        node = tree_only_db.tree["SR"]["BPM"]
        assert tree_only_db._is_leaf_node(node, len(tree_only_db.hierarchy_levels)) is True
        assert tree_only_db._is_leaf_node(node, 1) is False


class TestChannelPartLookup:
    """``_get_channel_part``'s three-way lookup, read off the node directly."""

    def test_an_explicit_part_replaces_the_tree_key(self, channel_part_db) -> None:
        assert channel_part_db._get_channel_part({"_channel_part": "MAG"}, "Magnets") == "MAG"

    def test_an_empty_part_is_honoured_rather_than_falling_back(self, channel_part_db) -> None:
        """The key is read by presence, not truthiness -- an empty override means
        "contribute nothing to the name", which is not the same as "not set"."""
        assert channel_part_db._get_channel_part({"_channel_part": ""}, "Building-1") == ""

    def test_a_node_without_the_key_uses_the_tree_key(self, channel_part_db) -> None:
        assert channel_part_db._get_channel_part({"_description": "Rack"}, "RACK") == "RACK"


class TestSelectionsWithListValues:
    """One ``build_channels_from_selections`` behaviour the inventories cannot show."""

    def test_a_separator_override_from_the_first_value_is_applied_to_all_of_them(
        self, separator_override_db
    ) -> None:
        """Characterized as-is, not endorsed: overrides leak across a list selection.

        ``_collect_separator_overrides`` navigates with a single value per level, so
        a list-valued ``device`` is resolved to its first entry -- ``BPM``, which
        carries ``"_separator": "_"``. The whole Cartesian product is then rendered
        with that separator, including ``QUAD``, which carries none. The same
        selection made on its own gets the pattern's ``:``, and the tree expansion
        agrees with the single-value answer.
        """
        assert separator_override_db.build_channels_from_selections(
            {"system": "SR", "device": ["BPM", "QUAD"], "signal": "X", "suffix": "RB"}
        ) == ["SR-BPM_X:RB", "SR-QUAD_X:RB"]

        # QUAD alone keeps the pattern separator between ``device`` and ``signal``.
        assert separator_override_db.build_channels_from_selections(
            {"system": "SR", "device": "QUAD", "signal": "Setpoint", "suffix": "RB"}
        ) == ["SR-QUAD:Setpoint:RB"]
        assert "SR-QUAD:Setpoint:RB" in separator_override_db.channel_map
