"""Health: the state of a channel, of a thing and of a roll-up node."""

from __future__ import annotations

import time

import pytest

from osprey.facility.panels.health import (
    channel_state,
    rollup,
    row_matches,
    rule_channels,
    unsupported,
    worst,
)

PLACE_XR02 = ["XR02:IP1:Pressure-RB", "XR02:QD1:Current-RB", "XR02:QF1:Current-RB"]
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


@pytest.mark.parametrize(
    ("row", "value", "expected"),
    [
        ("0", 0, True),
        ("0", "0.0", True),
        ("OPEN", "OPEN", True),
        ("OPEN", "open", False),
        ("!=0", 5, True),
        ("!=0", "0.0", False),
        ("!=0", "x", False),
        ("<5", 4.9, True),
        ("<5", 5, False),
        ("<=5", "5", True),
        (">5", 5, False),
        (">5", "6", True),
        (">=5", 5.0, True),
        (">=-2.5", -2, True),
        (">5", "x", False),
        ("<5", True, False),
        ("bit=1", 5, True),
        ("bit0=1", 5.0, True),
        ("bit1=1", "5", False),
        ("bit1=0", "5", True),
        ("bit2=1", 5, True),
        ("bit2=1", 5.5, False),
        ("bit2=1", "x", False),
        ("bit=0", 4, True),
    ],
)
def test_a_row_value_grammar(row, value, expected):
    assert row_matches(row, value) is expected


@pytest.mark.parametrize("row", ["expr:a>3", "expr:", "<x", "bit=x", "bitx=1", ">=", "!="])
def test_an_unsupported_row_is_never_matched(row):
    for value in (0, 1, 3, 5, "expr:a>3", "<x", "bit=x", "", "x"):
        assert row_matches(row, value) is False


def test_an_unsupported_row_is_not_compared_as_text(panel_graph):
    # the only row of this channel is unsupported: unknown, even for its own text
    assert channel_state(panel_graph, "XR02:BEND1:Current-RB", "expr:a>3") == "unknown"
    assert channel_state(panel_graph, "XR02:BEND1:Current-RB", 5) == "unknown"


def test_mixed_rows_evaluate_only_the_supported_ones(panel_graph):
    assert channel_state(panel_graph, "XR02:QF1:Current-RB", 2) == "normal"
    assert channel_state(panel_graph, "XR02:QF1:Current-RB", 9) == "unknown"


def test_comparison_rows_on_a_channel(panel_graph):
    name = "XR02:IP1:Pressure-RB"
    assert channel_state(panel_graph, name, 7) == "bad"
    assert channel_state(panel_graph, name, 0.5) == "warn"
    assert channel_state(panel_graph, name, 3) == "unknown"
    assert channel_state(panel_graph, name, "n/a") == "unknown"


def test_bit_rows_on_a_channel(panel_graph):
    name = "XR02:QD1:Current-RB"
    assert channel_state(panel_graph, name, 3) == "disagree"  # bit 0 normal, bit 1 warn
    assert channel_state(panel_graph, name, "1") == "normal"
    assert channel_state(panel_graph, name, 2.0) == "warn"
    assert channel_state(panel_graph, name, 4) == "unknown"


def test_unsupported_counts_forms_with_digits_replaced(panel_graph):
    assert unsupported(panel_graph) == {"expr:a>N": 2}


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
    assert set(PLACE_XR02) <= set(rule_channels(panel_graph, "XR"))
    assert rollup(panel_graph, "XR/XR01", {PUMP: 1, SUPPLY: 1}) == "normal"
    assert rollup(panel_graph, "XR/XR01", {PUMP: 1, SUPPLY: 0}) == "warn"
    assert rollup(panel_graph, "XR", {PUMP: 1, SUPPLY: 1}) == "unknown"  # XR02 values missing
    sector = {"XR02:IP1:Pressure-RB": 0.5, "XR02:QD1:Current-RB": 2, "XR02:QF1:Current-RB": 2}
    assert rollup(panel_graph, "XR", {PUMP: 1, SUPPLY: 1, **sector}) == "warn"


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
    assert rollup(panel_graph, "XR02:SHIELD1", {}) is None


def test_a_group_reads_its_members(panel_graph):
    assert rule_channels(panel_graph, "XR01-VAC", kind="group") == [PUMP]
    assert rollup(panel_graph, "XR01-VAC", {PUMP: 1}, kind="group") == "normal"
    assert rollup(panel_graph, "XR01-VAC", {PUMP: 0}, kind="group") == "disagree"


def test_an_excluded_thing_is_never_read(panel_graph):
    trip = "XR02:SAF1:Trip-Sts"
    door = "XR02:SAF2:Door-Sts"
    assert trip not in rule_channels(panel_graph, "XR02-VAC", kind="group")
    assert door not in rule_channels(panel_graph, "XR")
    pressure = "XR02:IP1:Pressure-RB"
    assert rule_channels(panel_graph, "XR02-VAC", kind="group") == [pressure]
    values = {trip: 1, door: 1, pressure: 7}
    assert rollup(panel_graph, "XR02-VAC", values, kind="group") == "bad"
    assert rollup(panel_graph, "XR02-VAC", {trip: 1, door: 1}, kind="group") == "unknown"
    with pytest.raises(KeyError):
        rule_channels(panel_graph, "XR02:SAF1")


def test_a_roll_up_does_not_read_values_outside_its_channels(panel_graph):
    class Strict(dict):
        def __getitem__(self, key):
            raise AssertionError(key)

        def get(self, key, default=None):
            assert key in {PUMP, SUPPLY, *PLACE_XR02, "XR02:BEND1:Current-RB"}, key
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
