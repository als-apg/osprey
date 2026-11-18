"""The channel-fault table: which roles each fault names and what each hook answers."""

from __future__ import annotations

import math

import pytest

from osprey_connectors.simulation.channel_faults import (
    _DISCONNECTED,
    _FROZEN,
    CHANNEL_FAULTS,
    STUCK,
)


def test_the_table_holds_the_three_words() -> None:
    assert sorted(CHANNEL_FAULTS) == sorted([STUCK, _FROZEN, _DISCONNECTED])
    assert (STUCK, _FROZEN, _DISCONNECTED) == ("stuck", "frozen", "disconnected")


@pytest.mark.parametrize(
    ("word", "roles"),
    [
        (STUCK, {"setpoint"}),
        (_FROZEN, {"readback", "none"}),
        (_DISCONNECTED, {"setpoint", "readback", "none"}),
    ],
)
def test_each_fault_names_its_roles(word: str, roles: set[str]) -> None:
    assert CHANNEL_FAULTS[word].roles == roles


@pytest.mark.parametrize(
    ("word", "holds"), [(STUCK, True), (_FROZEN, False), (_DISCONNECTED, True)]
)
def test_the_write_hook_holds_stuck_and_disconnected_writes(word: str, holds: bool) -> None:
    assert CHANNEL_FAULTS[word].write is holds


def test_stuck_changes_no_reading() -> None:
    assert CHANNEL_FAULTS[STUCK].read is None


@pytest.mark.parametrize(("value_type", "snapshot"), [("float", 1.5), ("string", "alpha")])
def test_a_frozen_reading_reads_its_activation_value(value_type: str, snapshot: object) -> None:
    read = CHANNEL_FAULTS[_FROZEN].read
    assert read is not None
    assert read(value_type, snapshot) == snapshot


def test_a_disconnected_float_reads_nan() -> None:
    read = CHANNEL_FAULTS[_DISCONNECTED].read
    assert read is not None
    assert math.isnan(read("float", 1.5))


@pytest.mark.parametrize(
    ("value_type", "snapshot"), [("string", "alpha"), ("enum", "ON"), ("int", 3)]
)
def test_a_disconnected_non_float_reads_its_activation_value(
    value_type: str, snapshot: object
) -> None:
    read = CHANNEL_FAULTS[_DISCONNECTED].read
    assert read is not None
    assert read(value_type, snapshot) == snapshot


@pytest.mark.parametrize(
    ("word", "condition"), [(STUCK, None), (_FROZEN, None), (_DISCONNECTED, "udf")]
)
def test_only_disconnected_reports_a_condition(word: str, condition: str | None) -> None:
    assert CHANNEL_FAULTS[word].severity == condition
