"""Tests for the dock-layout preset rules (`web_terminal/presets.py`).

One module answers "is this value a preset" and "what is left of it once the
panels unknown now are pruned" for the profile build, the terminal's config
loader and the open-arrangement op, so a layout refused in one place is refused
in every place for the same reason. One of those reasons is OSPREY's own, not
the layout grammar's: a tile shows one panel (``one_panel_per_tile``).
"""

from __future__ import annotations

import pytest

from osprey.interfaces.web_terminal.dock_layout import layout_refusal
from osprey.interfaces.web_terminal.presets import (
    ONE_PANEL_PER_TILE_TEXT,
    crowded_tile,
    names_terminal,
    preset_refusal,
    prune_preset_layout,
)


class TestPresetRefusal:
    @pytest.mark.parametrize(
        "layout",
        [
            pytest.param({"tabs": ["ariel"]}, id="one-tile"),
            pytest.param({"tabs": ["ariel"], "showing": "ariel", "size": 2}, id="showing-and-size"),
            pytest.param(
                {"split": "row", "items": [{"tabs": ["ariel"]}, {"tabs": ["okf"], "size": 3}]},
                id="split",
            ),
            pytest.param(
                {
                    "split": "column",
                    "items": [
                        {"tabs": ["ariel"]},
                        {"split": "row", "items": [{"tabs": ["okf"]}, {"tabs": ["lattice"]}]},
                    ],
                },
                id="nested",
            ),
        ],
    )
    def test_a_collection_layout_over_panel_ids_is_a_preset(self, layout):
        assert preset_refusal(layout) is None

    @pytest.mark.parametrize(
        ("layout", "fragment"),
        [
            pytest.param(["ariel", "okf"], "retired list-of-panel-ids form", id="list-form"),
            pytest.param("ariel", "must be a layout", id="string"),
            pytest.param(None, "must be a layout", id="none"),
            pytest.param({"slot": 0}, "not_layout", id="not-a-layout"),
            pytest.param({"tabs": []}, "empty_tabs", id="empty-tabs"),
            pytest.param({"tile": "t1", "tabs": ["ariel"]}, "unknown_field", id="tile-id"),
            pytest.param(
                {"tabs": ["ariel"], "showing": "okf"}, "showing_not_in_tabs", id="showing"
            ),
            pytest.param({"tabs": ["ariel"], "size": -1}, "bad_size", id="size"),
            pytest.param({"tabs": [0]}, "not member indexes", id="indexes"),
            pytest.param(
                {"split": "row", "items": [{"tabs": ["ariel"]}, {"tabs": [1]}]},
                "not member indexes",
                id="nested-index",
            ),
            pytest.param({"tabs": ["terminal"]}, "names the terminal", id="terminal"),
            pytest.param(
                {"split": "row", "items": [{"tabs": ["ariel"]}, {"tabs": ["terminal"]}]},
                "names the terminal",
                id="nested-terminal",
            ),
        ],
    )
    def test_anything_else_is_refused_with_its_reason(self, layout, fragment):
        refusal = preset_refusal(layout)
        assert refusal is not None and fragment in refusal, refusal

    @pytest.mark.parametrize(
        ("layout", "where"),
        [
            pytest.param({"tabs": ["ariel", "okf"], "showing": "okf"}, "layout.tabs", id="root"),
            pytest.param(
                {
                    "split": "column",
                    "items": [
                        {"tabs": ["ariel"]},
                        {"split": "row", "items": [{"tabs": ["okf"]}, {"tabs": ["a", "b"]}]},
                    ],
                },
                "layout.items[1].items[1].tabs",
                id="nested",
            ),
        ],
    )
    def test_a_tile_holding_several_panels_is_refused_with_where_it_is(self, layout, where):
        """A valid layout, by the grammar, that crowds a tile is not a preset."""
        assert layout_refusal(layout, form="collection") is None
        assert preset_refusal(layout) == (
            f"puts several panels in one tile ({where}): {ONE_PANEL_PER_TILE_TEXT}"
        )

    def test_the_layout_rules_come_before_the_one_panel_rule(self):
        """A layout that breaks a grammar rule is refused for that, crowded or not."""
        refusal = preset_refusal({"tabs": ["ariel", "okf"], "showing": "lattice"})
        assert refusal is not None and "showing_not_in_tabs" in refusal

    def test_the_terminal_rule_comes_before_the_one_panel_rule(self):
        refusal = preset_refusal({"tabs": ["terminal", "ariel"]})
        assert refusal is not None and "names the terminal" in refusal


class TestCrowdedTile:
    @pytest.mark.parametrize(
        "layout",
        [
            pytest.param({"tile": "t1", "tabs": ["terminal"], "showing": "terminal"}, id="dock"),
            pytest.param({"tabs": [0]}, id="collection"),
            pytest.param({"split": "row", "items": [{"tabs": ["a"]}, {"tabs": ["b"]}]}, id="split"),
        ],
    )
    def test_none_when_every_tile_holds_one_panel(self, layout):
        assert crowded_tile(layout) is None

    def test_names_the_first_crowded_tile_in_tree_order_in_either_form(self):
        dock = {
            "split": "row",
            "items": [
                {"tile": "t1", "tabs": ["terminal"], "showing": "terminal", "size": 0.4},
                {
                    "split": "column",
                    "size": 0.6,
                    "items": [
                        {"tile": "t2", "tabs": ["a", "b"], "showing": "b", "size": 0.5},
                        {"tile": "t3", "tabs": ["c", "d"], "showing": "c", "size": 0.5},
                    ],
                },
            ],
        }
        assert layout_refusal(dock) is None
        assert crowded_tile(dock) == "layout.items[1].items[0].tabs"
        collection = {"split": "row", "items": [{"tabs": [0]}, {"tabs": [1, 2]}]}
        assert crowded_tile(collection) == "layout.items[1].tabs"


class TestNamesTerminal:
    def test_finds_the_terminal_at_any_depth_and_in_malformed_trees(self):
        assert names_terminal({"tabs": ["terminal"]})
        assert names_terminal({"split": "row", "items": [{"tabs": ["a"]}, {"tabs": ["terminal"]}]})
        assert names_terminal({"items": [{"tabs": ["terminal"]}]})  # no split field, still read
        assert not names_terminal({"tabs": ["a", "b"]})
        assert not names_terminal(["terminal"])
        assert not names_terminal(None)


class TestPrunePresetLayout:
    def test_a_pruned_panel_takes_its_tile_and_a_one_item_split_collapses(self):
        layout = {"split": "row", "items": [{"tabs": ["ghost"]}, {"tabs": ["ariel"], "size": 2}]}
        assert prune_preset_layout(layout, lambda k: k != "ghost") == {
            "tabs": ["ariel"],
            "showing": "ariel",
        }

    def test_survivors_keep_their_weights_which_normalise_to_fractions(self):
        layout = {
            "split": "row",
            "items": [{"tabs": ["ariel"], "size": 3}, {"tabs": ["ghost"]}, {"tabs": ["okf"]}],
        }
        assert prune_preset_layout(layout, lambda k: k != "ghost") == {
            "split": "row",
            "items": [
                {"tabs": ["ariel"], "showing": "ariel", "size": 0.75},
                {"tabs": ["okf"], "showing": "okf", "size": 0.25},
            ],
        }

    def test_nothing_left_is_none(self):
        layout = {"split": "row", "items": [{"tabs": ["ghost"]}, {"tabs": ["gone"]}]}
        assert prune_preset_layout(layout, lambda k: False) is None

    def test_a_whole_preset_comes_back_normalised_and_unchanged_in_shape(self):
        layout = {
            "split": "column",
            "items": [{"tabs": ["okf"], "showing": "okf"}, {"tabs": ["lattice"]}],
        }
        assert prune_preset_layout(layout, lambda k: True) == {
            "split": "column",
            "items": [
                {"tabs": ["okf"], "showing": "okf", "size": 0.5},
                {"tabs": ["lattice"], "showing": "lattice", "size": 0.5},
            ],
        }
