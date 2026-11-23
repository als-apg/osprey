"""Health: the state of a channel, of a thing and of a roll-up node."""

from __future__ import annotations

import time

import pytest

from osprey_panels.health import (
    channel_state,
    rollup,
    row_matches,
    rule_channels,
    unsupported,
    worst,
)

PLACE_XR02 = ["XR02-IP1:Pressure-RB", "XR02-QD1:Current-RB", "XR02-QF1:Current-RB"]
PUMP = "XR01-IP1:State-Sts"
SUPPLY = "XR01-PS1:On-Sts"


def test_a_channel_without_a_rule_is_unknown(panel_graph):
    assert channel_state(panel_graph, "XR01-QF1:Current-RB", 3.0) == "unknown"
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
    assert channel_state(panel_graph, "XR02-SAF2:Door-Sts", 1) == "unknown"
    assert channel_state(panel_graph, "XR02-SAF1:Trip-Sts", 1) == "unknown"


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
    assert channel_state(panel_graph, "XR02-BEND1:Current-RB", "expr:a>3") == "unknown"
    assert channel_state(panel_graph, "XR02-BEND1:Current-RB", 5) == "unknown"


def test_mixed_rows_evaluate_only_the_supported_ones(panel_graph):
    assert channel_state(panel_graph, "XR02-QF1:Current-RB", 2) == "normal"
    assert channel_state(panel_graph, "XR02-QF1:Current-RB", 9) == "unknown"


def test_comparison_rows_on_a_channel(panel_graph):
    name = "XR02-IP1:Pressure-RB"
    assert channel_state(panel_graph, name, 7) == "bad"
    assert channel_state(panel_graph, name, 0.5) == "warn"
    assert channel_state(panel_graph, name, 3) == "unknown"
    assert channel_state(panel_graph, name, "n/a") == "unknown"


def test_bit_rows_on_a_channel(panel_graph):
    name = "XR02-QD1:Current-RB"
    assert channel_state(panel_graph, name, 3) == "disagree"  # bit 0 normal, bit 1 warn
    assert channel_state(panel_graph, name, "1") == "normal"
    assert channel_state(panel_graph, name, 2.0) == "warn"
    assert channel_state(panel_graph, name, 4) == "unknown"


def test_unsupported_counts_forms_with_digits_replaced(panel_graph):
    assert unsupported(panel_graph) == {"expr:a>N": 2, "bitN=N": 1}


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
    assert rollup(panel_graph, "XR01:QF1", {"XR01-QF1:Current-RB": 1}) is None
    assert rollup(panel_graph, "XR02:SHIELD1", {}) is None


def test_a_place_reads_everything_within_it_at_any_depth(panel_graph):
    assert sorted(rule_channels(panel_graph, "XR/XR01")) == sorted([PUMP, SUPPLY])
    assert set(PLACE_XR02) <= set(rule_channels(panel_graph, "XR"))
    assert rollup(panel_graph, "XR/XR01", {PUMP: 1, SUPPLY: 1}) == "normal"
    assert rollup(panel_graph, "XR/XR01", {PUMP: 1, SUPPLY: 0}) == "warn"
    assert rollup(panel_graph, "XR", {PUMP: 1, SUPPLY: 1}) == "unknown"  # XR02 values missing
    sector = {"XR02-IP1:Pressure-RB": 0.5, "XR02-QD1:Current-RB": 2, "XR02-QF1:Current-RB": 2}
    assert rollup(panel_graph, "XR", {PUMP: 1, SUPPLY: 1, **sector}) == "warn"


def test_a_missing_value_in_a_place_reads_unknown(panel_graph):
    assert rollup(panel_graph, "XR/XR01", {PUMP: 1}) == "unknown"
    assert rollup(panel_graph, "XR/XR01", {}) == "unknown"


def test_a_channel_less_thing_leaves_the_place_unchanged(panel_graph):
    values = {PUMP: 1, SUPPLY: 1}
    before = rollup(panel_graph, "XR/XR01", values)
    # the rack holds three chassis and a place channel, none with a rule
    assert rule_channels(panel_graph, "XR/XR01/RK1") == []
    assert rollup(panel_graph, "XR/XR01/RK1", {"XR01-CH1:Temp-RB": 5}) is None
    assert rollup(panel_graph, "XR/XR01", values) == before == "normal"


def test_a_place_with_nothing_to_read_gives_none(panel_graph):
    assert rollup(panel_graph, "XR02:SHIELD1", {}) is None


def test_a_group_reads_its_members(panel_graph):
    assert rule_channels(panel_graph, "XR01-VAC", kind="group") == [PUMP]
    assert rollup(panel_graph, "XR01-VAC", {PUMP: 1}, kind="group") == "normal"
    assert rollup(panel_graph, "XR01-VAC", {PUMP: 0}, kind="group") == "disagree"


def test_an_excluded_thing_is_never_read(panel_graph):
    trip = "XR02-SAF1:Trip-Sts"
    door = "XR02-SAF2:Door-Sts"
    assert trip not in rule_channels(panel_graph, "XR02-VAC", kind="group")
    assert door not in rule_channels(panel_graph, "XR")
    pressure = "XR02-IP1:Pressure-RB"
    assert rule_channels(panel_graph, "XR02-VAC", kind="group") == [pressure]
    values = {trip: 1, door: 1, pressure: 7}
    assert rollup(panel_graph, "XR02-VAC", values, kind="group") == "bad"
    assert rollup(panel_graph, "XR02-VAC", {trip: 1, door: 1}, kind="group") == "unknown"
    with pytest.raises(KeyError):
        rule_channels(panel_graph, "XR02:SAF1")


def test_a_roll_up_does_not_read_values_outside_its_channels(panel_graph):
    class Recording(dict):
        read: list[str]

        def __getitem__(self, key):
            self.read.append(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            self.read.append(key)
            return super().get(key, default)

    values = Recording({PUMP: 1, SUPPLY: 1})
    values.read = []
    rollup(panel_graph, "XR", values)
    assert values.read
    assert set(values.read) <= {PUMP, SUPPLY, *PLACE_XR02, "XR02-BEND1:Current-RB"}


def test_a_roll_up_over_a_large_place_is_linear():
    from types import SimpleNamespace

    from osprey_panels.tables import HealthRule

    count = 20_000
    rule = [HealthRule("c", "1", "normal", None, "stated", None, None, None)]
    things = [SimpleNamespace(id=f"D{i}", kind="device") for i in range(count)]
    place = SimpleNamespace(id="P", kind="place")

    class Big:
        def thing(self, *_args, **_kwargs):
            return place

        def contents(self, *_args, **_kwargs):
            return things

        def channels(self, thing_id, *_args):
            return [SimpleNamespace(name=f"{thing_id}:V")] if thing_id != "P" else []

        def rule(self, *_args):
            return rule

    values = {f"D{i}:V": 1 for i in range(count)}
    started = time.perf_counter()
    assert rollup(Big(), "P", values) == "normal"
    assert time.perf_counter() - started < 2


BIG = 2**53 + 1
NAN = float("nan")
INF = float("inf")


class _Odd:
    def __str__(self):
        raise RuntimeError("no text")


@pytest.mark.parametrize(
    "value",
    [NAN, INF, -INF, "nan", "inf", 10**400, -(10**400), None, "", [], [1], object(), _Odd()],
)
def test_no_live_value_can_raise(panel_graph, value):
    names = [PUMP, SUPPLY, *PLACE_XR02, "XR02-BEND1:Current-RB"]
    for name in names:
        assert channel_state(panel_graph, name, value) in {
            "unknown",
            "normal",
            "warn",
            "bad",
            "disagree",
        }
    assert rollup(panel_graph, "XR", dict.fromkeys(names, value)) is not None
    for row in ("bit=1", "bit3=0", ">5", "<=5", "!=0", "0", "OPEN"):
        assert row_matches(row, value) in {True, False}


def test_a_big_integer_keeps_its_precision():
    assert row_matches("bit0=1", BIG) is True
    assert row_matches("bit0=1", str(BIG)) is True
    assert row_matches("bit0=1", BIG - 1) is False
    assert row_matches(">9007199254740992", BIG) is True
    assert row_matches("<=9007199254740992", BIG) is False
    assert row_matches("!=9007199254740993", BIG) is False
    assert row_matches(f"{BIG}", BIG) is True
    assert row_matches("9007199254740992", BIG) is False


@pytest.mark.parametrize("row", ["!=0", "<5", "<=5", ">-5", ">=-5", "bit=0", "bit=1", "bit3=1"])
@pytest.mark.parametrize("value", [NAN, INF, -INF, "nan", "inf", "-inf", "Infinity"])
def test_a_non_finite_value_matches_no_operator_row(row, value):
    assert row_matches(row, value) is False


def test_a_non_finite_channel_is_unknown(panel_graph):
    for value in (NAN, INF, -INF, "nan"):
        assert channel_state(panel_graph, "XR02-IP1:Pressure-RB", value) == "unknown"
        assert channel_state(panel_graph, "XR02-QD1:Current-RB", value) == "unknown"


def test_a_bit_row_needs_zero_or_one():
    for row in ("bit=2", "bit1=7"):
        assert row_matches(row, 5) is False
        assert row_matches(row, 2) is False


def test_the_bit_form_tolerates_the_spaces_the_comparisons_do():
    assert row_matches("bit1= 1", 2) is True
    assert row_matches("bit1 =1", 2) is True
    assert row_matches("bit1 = 1", 2) is True
    assert row_matches(">= 5", 5) is True
    assert row_matches("> 5", 6) is True


@pytest.mark.parametrize(
    ("row_value", "expected"),
    [
        ("0", True),
        ("!=0", True),
        ("bit1=1", True),
        (">=5", True),
        ("expr:a>3", False),
        ("bit1=2", False),
        ("<x", False),
    ],
)
def test_supported_says_whether_the_engine_can_evaluate_a_row(row_value, expected):
    from osprey_panels.health import supported

    assert supported(row_value) is expected


@pytest.mark.parametrize(
    ("row_value", "expected"),
    [
        ("0", ("literal", None, 0, None, None)),
        (" ON ", ("literal", None, None, None, None)),
        ("2.5", ("literal", None, 2.5, None, None)),
        (">=5", ("compare", ">=", 5, None, None)),
        ("!= 0", ("compare", "!=", 0, None, None)),
        ("bit=1", ("bit", None, None, 0, 1)),
        ("bit3=0", ("bit", None, None, 3, 0)),
        ("bit1=2", None),
        ("expr:a>3", None),
        ("<inf", None),
    ],
)
def test_form_names_the_grammar_the_engine_evaluates(row_value, expected):
    from osprey_panels.health import form, supported

    found = form(row_value)
    assert (found is not None) is supported(row_value)
    if expected is None:
        assert found is None
    else:
        assert (found.kind, found.op, found.number, found.bit, found.wanted) == expected
        assert found.text == row_value.strip()


def test_disagree_has_one_definition_the_others_call(panel_graph):
    import importlib
    from pathlib import Path

    from osprey_panels import health, layout

    describe = importlib.import_module("osprey_panels.describe")

    assert health.disagree({"bad", "normal"}) and not health.disagree(["bad", "bad"])
    assert health.disagrees(panel_graph, "XR01-IP1:State-Sts")  # value 0: bad and normal
    assert not health.disagrees(panel_graph, "XR02-IP1:Pressure-RB")
    for module in (describe, layout):
        source = Path(module.__file__).read_text()
        assert "disagree(" in source or "disagrees(" in source
        assert "len(states" not in source and "states_a != states_b" not in source
