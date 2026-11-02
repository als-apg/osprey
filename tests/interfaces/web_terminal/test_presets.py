"""Tests for the dock-layout preset rules (`web_terminal/presets.py`).

One module answers "is this value a preset" and "what is left of it once the
panels unknown now are pruned" for the profile build, the terminal's config
loader and the open-arrangement op, so a layout refused in one place is refused
in every place for the same reason.
"""

from __future__ import annotations

import pytest

from osprey.interfaces.web_terminal.presets import (
    names_terminal,
    preset_refusal,
    prune_preset_layout,
)


class TestPresetRefusal:
    @pytest.mark.parametrize(
        "layout",
        [
            pytest.param({"tabs": ["ariel"]}, id="one-tile"),
            pytest.param({"tabs": ["ariel", "okf"], "showing": "okf", "size": 2}, id="tabs"),
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
            pytest.param({"tabs": [0, 1]}, "not member indexes", id="indexes"),
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


class TestNamesTerminal:
    def test_finds_the_terminal_at_any_depth_and_in_malformed_trees(self):
        assert names_terminal({"tabs": ["terminal"]})
        assert names_terminal({"split": "row", "items": [{"tabs": ["a"]}, {"tabs": ["terminal"]}]})
        assert names_terminal({"items": [{"tabs": ["terminal"]}]})  # no split field, still read
        assert not names_terminal({"tabs": ["a", "b"]})
        assert not names_terminal(["terminal"])
        assert not names_terminal(None)


class TestPrunePresetLayout:
    def test_a_pruned_tab_leaves_its_tile_and_showing_falls_back(self):
        layout = {"tabs": ["ariel", "ghost"], "showing": "ghost", "size": 2}
        assert prune_preset_layout(layout, lambda k: k != "ghost") == {
            "tabs": ["ariel"],
            "showing": "ariel",
        }

    def test_an_emptied_tile_leaves_and_a_one_item_split_collapses(self):
        layout = {"split": "row", "items": [{"tabs": ["ghost"]}, {"tabs": ["ariel", "okf"]}]}
        assert prune_preset_layout(layout, lambda k: k != "ghost") == {
            "tabs": ["ariel", "okf"],
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
            "items": [{"tabs": ["ariel", "okf"], "showing": "okf"}, {"tabs": ["lattice"]}],
        }
        assert prune_preset_layout(layout, lambda k: True) == {
            "split": "column",
            "items": [
                {"tabs": ["ariel", "okf"], "showing": "okf", "size": 0.5},
                {"tabs": ["lattice"], "showing": "lattice", "size": 0.5},
            ],
        }
