"""Unit tests for :mod:`osprey.config_guards`.

The guard is the one definition of "positive integer" for authored config, and
its refusal is the one message every surface that refuses spells. The tests
below pin both: the predicate's treatment of ``bool`` (an ``int`` subclass a
config author never means), and the refusal naming the dotted key for a parsed
value and the variable for a value that arrives as text.
"""

from __future__ import annotations

import pytest

from osprey.config_guards import is_positive_int, require_positive_int, require_positive_int_str


class TestIsPositiveInt:
    @pytest.mark.parametrize("value", [1, 2, 8181, 65535])
    def test_positive_integers_pass(self, value: object) -> None:
        assert is_positive_int(value) is True

    @pytest.mark.parametrize("value", [0, -1, -8181])
    def test_zero_and_negatives_fail(self, value: object) -> None:
        assert is_positive_int(value) is False

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_is_not_an_integer_here(self, value: object) -> None:
        """``bool`` subclasses ``int``; ``port: true`` must never read as port 1."""
        assert is_positive_int(value) is False

    @pytest.mark.parametrize("value", [None, "1", 1.0, [1], {"port": 1}, object()])
    def test_non_integers_fail(self, value: object) -> None:
        assert is_positive_int(value) is False


class TestRequirePositiveInt:
    def test_absent_key_yields_the_default(self) -> None:
        assert require_positive_int(None, 30, "services.qmd.interval") == 30

    @pytest.mark.parametrize("value", [1, 8181])
    def test_a_usable_value_is_returned(self, value: int) -> None:
        assert require_positive_int(value, 30, "services.qmd.port") == value

    @pytest.mark.parametrize("bad", [0, -1, "8181", 8181.0, [8181]])
    def test_unusable_values_are_refused_by_key(self, bad: object) -> None:
        with pytest.raises(ValueError) as excinfo:
            require_positive_int(bad, 30, "services.qmd.port")
        assert "services.qmd.port must be a positive integer" in str(excinfo.value)
        assert repr(bad) in str(excinfo.value)

    def test_true_is_refused_rather_than_read_as_one(self) -> None:
        with pytest.raises(ValueError, match=r"services\.qmd\.port"):
            require_positive_int(True, 30, "services.qmd.port")


class TestRequirePositiveIntStr:
    def test_unset_variable_yields_the_default(self) -> None:
        assert require_positive_int_str(None, 50, "OSPREY_BLUESKY_MAX_RUNS") == 50

    @pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("7", 7), (" 200 ", 200)])
    def test_a_parsable_positive_value_is_returned(self, raw: str, expected: int) -> None:
        assert require_positive_int_str(raw, 50, "OSPREY_BLUESKY_MAX_RUNS") == expected

    @pytest.mark.parametrize("raw", ["0", "-1", "", "abc", "1.5", "true"])
    def test_unusable_text_is_refused_by_variable_name(self, raw: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            require_positive_int_str(raw, 50, "OSPREY_BLUESKY_MAX_RUNS")
        assert "OSPREY_BLUESKY_MAX_RUNS must be a positive integer" in str(excinfo.value)
        assert repr(raw) in str(excinfo.value)


def test_both_forms_spell_the_same_refusal() -> None:
    """One contract: the sentence does not change with how the value arrived."""
    with pytest.raises(ValueError) as from_value:
        require_positive_int(0, 1, "NAME")
    with pytest.raises(ValueError) as from_text:
        require_positive_int_str("0", 1, "NAME")
    assert str(from_value.value) == "NAME must be a positive integer, got 0"
    assert str(from_text.value) == "NAME must be a positive integer, got '0'"
