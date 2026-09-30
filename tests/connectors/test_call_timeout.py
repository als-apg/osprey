"""The one call-bound checker every bounding control-system connector reads."""

import pytest

from osprey_connectors.control_system.call_timeout import DEFAULT_TIMEOUT_S, call_timeout_s


def test_an_absent_timeout_s_is_the_default():
    assert call_timeout_s({}, "tango") == 5.0 == DEFAULT_TIMEOUT_S


@pytest.mark.parametrize(("declared", "expected"), [(2, 2.0), (0.25, 0.25)])
def test_a_positive_number_is_returned_as_a_float(declared, expected):
    result = call_timeout_s({"timeout_s": declared}, "tango")
    assert result == expected
    assert type(result) is float


@pytest.mark.parametrize(("declared", "expected"), [("5", 5.0), ("0.25", 0.25)])
def test_a_numeric_string_is_parsed(declared, expected):
    result = call_timeout_s({"timeout_s": declared}, "tango")
    assert result == expected
    assert type(result) is float


@pytest.mark.parametrize(
    "bad",
    [
        0,
        -1,
        "five",
        "0",
        "nan",
        "",
        True,
        False,
        None,
        [5],
        float("nan"),
        float("inf"),
        float("-inf"),
        10**400,
    ],
)
def test_an_unusable_value_is_refused_naming_the_key_and_the_value(bad):
    with pytest.raises(ValueError) as refused:
        call_timeout_s({"timeout_s": bad}, "tango")
    message = str(refused.value)
    assert "control_system.connector.tango.timeout_s" in message
    assert repr(bad) in message


def test_the_old_spelling_is_not_read():
    assert call_timeout_s({"timeout": 0.5}, "tango") == DEFAULT_TIMEOUT_S


def test_a_block_with_no_type_names_a_placeholder():
    with pytest.raises(ValueError, match=r"control_system\.connector\.<type>\.timeout_s"):
        call_timeout_s({"timeout_s": 0}, None)
