"""Selections resolve to the names the tree expansion emits.

``_build_channel_map`` decides which channels exist and how each is spelled.
``build_channels_from_selections`` must spell every real leaf a selection
reaches the same way, one combination at a time: channel parts below an
``instances`` level or an expanded instance, and the separators of each branch
of a multi-value selection. ``build_existing_channels`` then keeps the built
names that are channels and counts the combinations that hold none.
"""

from __future__ import annotations

from tests.services.channel_finder.conftest import (
    _document,
    channel_part_doc,
    hybrid_expansion_doc,
    range_instances_doc,
    separator_override_doc,
    tree_only_doc,
)


class TestChannelPartsBelowInstances:
    """A ``_channel_part`` below an instance is applied, as the expansion applies it."""

    def test_a_part_below_an_instances_level_is_applied(self, hier_db_factory) -> None:
        doc = range_instances_doc()
        doc["tree"]["SR"]["BPM"]["DEVICE"]["X"]["_channel_part"] = "XPOS"
        db = hier_db_factory(doc, "part_below_instances.json")
        assert "SR:BPM:B01:XPOS" in db.channel_map

        assert db.build_channels_from_selections(
            {"system": "SR", "family": "BPM", "device": "B01", "signal": "X"}
        ) == ["SR:BPM:B01:XPOS"]

    def test_a_part_below_an_expanded_tree_instance_is_applied(self, hier_db_factory) -> None:
        doc = hybrid_expansion_doc()
        doc["tree"]["SR"]["CH"]["Voltage"]["_channel_part"] = "V"
        db = hier_db_factory(doc, "part_below_expansion.json")
        assert "SR:CH-2:V" in db.channel_map

        assert db.build_channels_from_selections(
            {"system": "SR", "device": "CH-2", "signal": "Voltage"}
        ) == ["SR:CH-2:V"]


class TestEachCombinationIsWalkedOnItsOwn:
    """A multi-value selection is resolved per combination, not by its first value."""

    def test_each_branch_keeps_its_own_separator(self, hier_db_factory) -> None:
        db = hier_db_factory(separator_override_doc(), "separator_per_branch.json")

        built = db.build_channels_from_selections(
            {
                "system": "SR",
                "device": ["BPM", "QUAD"],
                "signal": ["X", "Setpoint"],
                "suffix": "RB",
            }
        )

        assert built == [
            "SR-BPM_X:RB",
            "SR-BPM_Setpoint:RB",
            "SR-QUAD:X:RB",
            "SR-QUAD:Setpoint:RB",
        ]
        assert [name for name in built if name in db.channel_map] == [
            "SR-BPM_X:RB",
            "SR-QUAD:Setpoint:RB",
        ]

    def test_a_level_outside_the_pattern_is_walked_for_every_value(self, hier_db_factory) -> None:
        db = hier_db_factory(
            _document(
                levels=[
                    {"name": "system", "type": "tree"},
                    {"name": "device", "type": "tree"},
                    {"name": "signal", "type": "tree"},
                ],
                naming_pattern="{system}:{signal}",
                tree={
                    "SR": {
                        "BPM": {"X": {"_channel_part": "BX"}},
                        "QUAD": {"X": {"_channel_part": "QX"}},
                    }
                },
            ),
            "navigation_only_list.json",
        )
        assert list(db.channel_map) == ["SR:BX", "SR:QX"]

        assert db.build_channels_from_selections(
            {"system": "SR", "device": ["BPM", "QUAD"], "signal": "X"}
        ) == ["SR:BX", "SR:QX"]


class TestBuildExistingChannels:
    """The bounded answer both callers of the selections API return."""

    def test_only_existing_addresses_are_listed(self, hier_db_factory) -> None:
        db = hier_db_factory(tree_only_doc(), "existing_only.json")

        answer = db.build_existing_channels(
            {"system": "SR", "device": ["BPM", "QUAD"], "signal": ["X", "Y", "Setpoint"]}
        )

        assert answer == {
            "channels": ["SR:BPM:X", "SR:BPM:Y", "SR:QUAD:Setpoint"],
            "total": 3,
            "invalid_count": 3,
            "invalid_examples": [
                {"system": "SR", "device": "BPM", "signal": "Setpoint"},
                {"system": "SR", "device": "QUAD", "signal": "X"},
                {"system": "SR", "device": "QUAD", "signal": "Y"},
            ],
        }

    def test_combinations_of_one_address_list_it_once_and_none_is_invalid(
        self, hier_db_factory
    ) -> None:
        """``Magnets`` is the tree key and ``MAG`` the part it emits: one address."""
        db = hier_db_factory(channel_part_doc(), "duplicate_address.json")

        answer = db.build_existing_channels(
            {"system": ["Magnets", "MAG"], "device": ["Power Supplies", "PS"], "signal": "Current"}
        )

        # Three of the four combinations spell the address; values the walk cannot
        # follow stay as given, so ``PS`` under either system is the address's part.
        assert answer == {
            "channels": ["MAG:PS:Current"],
            "total": 1,
            "invalid_count": 1,
            "invalid_examples": [
                {"system": "MAG", "device": "Power Supplies", "signal": "Current"}
            ],
        }

    def test_a_repeated_value_is_one_combination(self, hier_db_factory) -> None:
        db = hier_db_factory(tree_only_doc(), "repeated_value.json")

        answer = db.build_existing_channels(
            {"system": "SR", "device": "BPM", "signal": ["X", "X", "Nope", "Nope"]}
        )

        assert answer["channels"] == ["SR:BPM:X"]
        assert answer["invalid_count"] == 1
        assert answer["invalid_examples"] == [{"system": "SR", "device": "BPM", "signal": "Nope"}]

    def test_at_most_ten_examples_are_listed(self, hier_db_factory) -> None:
        db = hier_db_factory(tree_only_doc(), "example_cap.json")

        answer = db.build_existing_channels(
            {"system": "SR", "device": "BPM", "signal": [f"N{i}" for i in range(25)]}
        )

        assert answer["channels"] == []
        assert answer["total"] == 0
        assert answer["invalid_count"] == 25
        assert len(answer["invalid_examples"]) == 10
        assert answer["invalid_examples"][0] == {"system": "SR", "device": "BPM", "signal": "N0"}
