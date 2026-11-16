"""Health: the state of a channel, of a thing and of a roll-up node."""

from __future__ import annotations

import time

import pytest

from osprey.facility.panels.health import channel_state, rollup, rule_channels, worst

PUMP = "XR01:IP1:State-Sts"
SUPPLY = "XR01:PS1:On-Sts"


def test_a_channel_without_a_rule_is_unknown(panel_graph):
    assert channel_state(panel_graph, "XR01:QF1:Current-RB", 3.0) == "unknown"
    assert channel_state(panel_graph, "XR99:NOPE:Nothing", 1) == "unknown"


def test_a_missing_value_is_unknown(panel_graph):
    assert channel_state(panel_graph, SUPPLY, None) == "unknown"


def test_a_value_no_row_names_is_unknown(panel_graph):
    assert channel_state(panel_graph, SUPPLY, 7) == "unknown"


def test_a_matching_row_gives_its_state(panel_graph):
    assert channel_state(panel_graph, SUPPLY, 1) == "normal"
    assert channel_state(panel_graph, SUPPLY, 0) == "warn"
    assert channel_state(panel_graph, SUPPLY, "0") == "warn"


def test_numbers_compare_as_numbers(panel_graph):
    assert channel_state(panel_graph, SUPPLY, 1.0) == "normal"
    assert channel_state(panel_graph, SUPPLY, "1.0") == "normal"
    assert channel_state(panel_graph, SUPPLY, 0.0) == "warn"


def test_sources_that_contradict_give_disagree(panel_graph):
    assert channel_state(panel_graph, PUMP, 0) == "disagree"


def test_sources_that_agree_give_that_state(panel_graph):
    assert channel_state(panel_graph, PUMP, 1) == "normal"


def test_the_state_does_not_depend_on_the_basis(panel_graph):
    # every supply row is inferred, the pump rows mix stated and inferred
    assert {r.basis for r in panel_graph.rule(SUPPLY)} == {"inferred"}
    assert {r.basis for r in panel_graph.rule(PUMP)} == {"stated", "inferred"}
    assert channel_state(panel_graph, SUPPLY, 1) == "normal"
    assert channel_state(panel_graph, PUMP, 1) == "normal"


def test_a_hidden_channel_is_unknown(panel_graph):
    assert channel_state(panel_graph, "XR02:SAF2:Door-Sts", 1) == "unknown"
    assert channel_state(panel_graph, "XR02:SAF1:Trip-Sts", 1) == "unknown"


def test_not_equal_rows_match_any_other_value():
    from osprey.facility.panels.health import _matches

    assert _matches("!=0", "5")
    assert _matches("!=0", "0.0") is False
    assert _matches("!=0", "x")


def test_worst_orders_the_states():
    assert worst(["disagree", "normal"]) == "disagree"
    assert worst(["disagree", "bad"]) == "bad"
    assert worst(["normal", "unknown"]) == "unknown"
    assert worst(["unknown", "disagree"]) == "disagree"
    assert worst(["warn", "disagree"]) == "warn"
    assert worst(["bad", "warn"]) == "bad"
    assert worst(["normal"]) == "normal"


def test_worst_of_nothing_is_none():
    assert worst([]) is None


def test_a_device_reads_its_own_rule_channels(panel_graph):
    assert rule_channels(panel_graph, "XR01:IP1") == [PUMP]
    assert rollup(panel_graph, "XR01:IP1", {PUMP: 1}) == "normal"
    assert rollup(panel_graph, "XR01:IP1", {PUMP: 0}) == "disagree"


def test_a_thing_without_a_rule_channel_contributes_nothing(panel_graph):
    assert rule_channels(panel_graph, "XR01:QF1") == []
    assert rollup(panel_graph, "XR01:QF1", {"XR01:QF1:Current-RB": 1}) is None
    assert rollup(panel_graph, "XR02:SHIELD1", {}) is None


def test_a_place_reads_everything_within_it_at_any_depth(panel_graph):
    assert sorted(rule_channels(panel_graph, "XR/XR01")) == sorted([PUMP, SUPPLY])
    assert sorted(rule_channels(panel_graph, "XR")) == sorted([PUMP, SUPPLY])
    assert rollup(panel_graph, "XR/XR01", {PUMP: 1, SUPPLY: 1}) == "normal"
    assert rollup(panel_graph, "XR/XR01", {PUMP: 1, SUPPLY: 0}) == "warn"
    assert rollup(panel_graph, "XR", {PUMP: 0, SUPPLY: 0}) == "warn"


def test_a_missing_value_in_a_place_reads_unknown(panel_graph):
    assert rollup(panel_graph, "XR/XR01", {PUMP: 1}) == "unknown"
    assert rollup(panel_graph, "XR/XR01", {}) == "unknown"


def test_a_channel_less_thing_leaves_the_place_unchanged(panel_graph):
    values = {PUMP: 1, SUPPLY: 1}
    before = rollup(panel_graph, "XR/XR01", values)
    # the rack holds three chassis and a place channel, none with a rule
    assert rule_channels(panel_graph, "XR/XR01/RK1") == []
    assert rollup(panel_graph, "XR/XR01/RK1", {"XR01:CH1:Temp-RB": 5}) is None
    assert rollup(panel_graph, "XR/XR01", values) == before == "normal"


def test_a_place_with_nothing_to_read_gives_none(panel_graph):
    assert rollup(panel_graph, "XR/XR02", {}) is None


def test_a_group_reads_its_members(panel_graph):
    assert rule_channels(panel_graph, "XR01-VAC", kind="group") == [PUMP]
    assert rollup(panel_graph, "XR01-VAC", {PUMP: 1}, kind="group") == "normal"
    assert rollup(panel_graph, "XR01-VAC", {PUMP: 0}, kind="group") == "disagree"


def test_an_excluded_thing_is_never_read(panel_graph):
    trip = "XR02:SAF1:Trip-Sts"
    door = "XR02:SAF2:Door-Sts"
    assert trip not in rule_channels(panel_graph, "XR02-VAC", kind="group")
    assert door not in rule_channels(panel_graph, "XR")
    assert rollup(panel_graph, "XR02-VAC", {trip: 1, door: 1}, kind="group") is None
    with pytest.raises(KeyError):
        rule_channels(panel_graph, "XR02:SAF1")


def test_a_roll_up_does_not_read_values_outside_its_channels(panel_graph):
    class Strict(dict):
        def __getitem__(self, key):
            raise AssertionError(key)

        def get(self, key, default=None):
            assert key in {PUMP, SUPPLY}, key
            return super().get(key, default)

    rollup(panel_graph, "XR", Strict({PUMP: 1, SUPPLY: 1}))


def test_a_roll_up_over_a_large_place_is_linear():
    from types import SimpleNamespace

    from osprey.facility.panels.tables import HealthRule

    count = 20_000
    rule = [HealthRule("c", "1", "normal", None, "stated", None, None, None)]
    things = [SimpleNamespace(id=f"D{i}", kind="device") for i in range(count)]
    place = SimpleNamespace(id="P", kind="place")

    class Big:
        def thing(self, *_args, **_kwargs):
            return place

        def within(self, *_args, **_kwargs):
            return things

        def channels(self, thing_id, *_args):
            return [SimpleNamespace(name=f"{thing_id}:V")] if thing_id != "P" else []

        def rule(self, *_args):
            return rule

    values = {f"D{i}:V": 1 for i in range(count)}
    started = time.perf_counter()
    assert rollup(Big(), "P", values) == "normal"
    assert time.perf_counter() - started < 2
