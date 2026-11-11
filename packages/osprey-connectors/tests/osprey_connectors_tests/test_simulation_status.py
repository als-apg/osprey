"""A served physics model's status text: only ``ok`` and ``no deck`` are not an error."""

from __future__ import annotations

import pytest

from osprey_connectors.simulation.status import STATUS_NO_DECK, STATUS_OK, is_error


def test_the_two_texts_that_are_not_an_error_are_ok_and_no_deck() -> None:
    assert {STATUS_OK, STATUS_NO_DECK} == {"ok", "no deck"}


@pytest.mark.parametrize(
    ("status", "error"),
    [("ok", False), ("no deck", False), ("closed orbit lost", True), ("", True)],
    ids=["ok", "no-deck", "engine-error", "empty"],
)
def test_only_ok_and_no_deck_are_not_an_error(status: str, error: bool) -> None:
    assert is_error(status) is error
