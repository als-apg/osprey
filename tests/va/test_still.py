"""The still precondition: a lane fails before measuring when a reading it compares moves."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from tests.va._still import assert_readings_still


class _View:
    """A view that lists ``still``; the addresses in ``moving`` keep motion under it."""

    def __init__(self, *, lists_still: bool = True, moving: frozenset[str] = frozenset()) -> None:
        self.lists_still = lists_still
        self.moving = moving

    def scenarios(self) -> tuple[dict[str, str], ...]:
        names = ("nominal", "still") if self.lists_still else ("nominal",)
        return tuple({"name": name} for name in names)

    def motion_envelope(self, address: str, active: Sequence[str] = ()) -> float:
        return 1.0e-6 if address in self.moving or "still" not in active else 0.0


def test_still_readings_pass() -> None:
    assert_readings_still(_View(), ["BPM:X", "BPM:Y"], ["still"])


def test_a_moving_reading_fails_naming_it() -> None:
    with pytest.raises(AssertionError, match="BPM:X"):
        assert_readings_still(_View(moving=frozenset({"BPM:X"})), ["BPM:X", "BPM:Y"], ["still"])


def test_an_active_set_without_still_fails() -> None:
    with pytest.raises(AssertionError, match="not one that names 'still'"):
        assert_readings_still(_View(), ["BPM:Y"], ["nominal"])


def test_a_view_without_the_still_scenario_fails() -> None:
    with pytest.raises(AssertionError, match="lists no 'still' scenario"):
        assert_readings_still(_View(lists_still=False), ["BPM:Y"], ["still"])
