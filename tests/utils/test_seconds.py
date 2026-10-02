"""What ``osprey.utils.seconds`` accepts as a duration, with and without zero."""

from __future__ import annotations

import pytest

from osprey.utils.seconds import non_negative_seconds, positive_seconds


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(12, 12.0, id="int"),
        pytest.param(2.5, 2.5, id="float"),
        pytest.param("2.5", 2.5, id="numeric-string"),
        pytest.param(" 90 ", 90.0, id="padded-string"),
    ],
)
def test_a_positive_finite_number_is_read_as_seconds(value, expected):
    seconds = positive_seconds(value)
    assert seconds == expected
    assert type(seconds) is float


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="None"),
        pytest.param(True, id="True"),
        pytest.param(False, id="False"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param("inf", id="inf-string"),
        pytest.param("1e400", id="string-parsing-to-inf"),
        pytest.param(10**400, id="10**400"),
        pytest.param("abc", id="non-numeric-string"),
        pytest.param("", id="empty-string"),
        pytest.param([60], id="list"),
    ],
)
def test_anything_else_is_not_a_number_of_seconds(value):
    assert positive_seconds(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0, 0.0, id="zero"),
        pytest.param("0", 0.0, id="zero-string"),
        pytest.param(12, 12.0, id="int"),
        pytest.param(2.5, 2.5, id="float"),
        pytest.param("2.5", 2.5, id="numeric-string"),
    ],
)
def test_zero_or_a_positive_finite_number_is_a_bound(value, expected):
    seconds = non_negative_seconds(value)
    assert seconds == expected
    assert type(seconds) is float


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="None"),
        pytest.param(True, id="True"),
        pytest.param(False, id="False"),
        pytest.param(-1, id="negative"),
        pytest.param("-0.5", id="negative-string"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param("1e400", id="string-parsing-to-inf"),
        pytest.param(10**400, id="10**400"),
        pytest.param("abc", id="non-numeric-string"),
        pytest.param([60], id="list"),
    ],
)
def test_anything_else_is_not_a_bound(value):
    assert non_negative_seconds(value) is None
