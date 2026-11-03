"""Unit tests for :mod:`osprey.config_guards`.

The guard is the one definition of "positive integer" for authored config, and
its refusal is the one message every surface that refuses spells. The tests
below pin both: the predicate's treatment of ``bool`` (an ``int`` subclass a
config author never means), and the refusal naming the dotted key for a parsed
value and the variable for a value that arrives as text.
"""

from __future__ import annotations

import pytest

from osprey.config_guards import (
    is_positive_int,
    require_absolute_path,
    require_positive_int,
    require_positive_int_str,
)


class TestIsPositiveInt:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(1, True, id="positive-one"),
            pytest.param(2, True, id="positive-two"),
            pytest.param(8181, True, id="positive-port"),
            pytest.param(65535, True, id="positive-max-port"),
            pytest.param(0, False, id="zero"),
            pytest.param(-1, False, id="negative-one"),
            pytest.param(-8181, False, id="negative-port"),
            # ``bool`` subclasses ``int``; ``port: true`` must never read as port 1.
            pytest.param(True, False, id="bool-true"),
            pytest.param(False, False, id="bool-false"),
            pytest.param(None, False, id="none"),
            pytest.param("1", False, id="numeric-string"),
            pytest.param(1.0, False, id="float"),
            pytest.param([1], False, id="list"),
            pytest.param({"port": 1}, False, id="mapping"),
            pytest.param(object(), False, id="arbitrary-object"),
        ],
    )
    def test_only_a_positive_integer_passes(self, value: object, expected: bool) -> None:
        assert is_positive_int(value) is expected


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


class TestRequireAbsolutePath:
    def test_absent_key_yields_none(self) -> None:
        assert require_absolute_path(None, "services.qmd.models_dir") is None

    def test_an_absolute_path_is_stripped_and_returned(self) -> None:
        """YAML indentation is not a reason to refuse a path."""
        assert require_absolute_path(" /srv/qmd-models ", "services.qmd.models_dir") == (
            "/srv/qmd-models"
        )

    @pytest.mark.parametrize("bad", ["", "   ", "models", "./models", "~/models", 7, True])
    def test_unusable_values_are_refused_by_key(self, bad: object) -> None:
        with pytest.raises(ValueError) as excinfo:
            require_absolute_path(bad, "services.qmd.models_dir")
        assert "services.qmd.models_dir" in str(excinfo.value)
        assert repr(bad) in str(excinfo.value)


def test_both_forms_spell_the_same_refusal() -> None:
    """One contract: the sentence does not change with how the value arrived."""
    with pytest.raises(ValueError) as from_value:
        require_positive_int(0, 1, "NAME")
    with pytest.raises(ValueError) as from_text:
        require_positive_int_str("0", 1, "NAME")
    assert str(from_value.value) == "NAME must be a positive integer, got 0"
    assert str(from_text.value) == "NAME must be a positive integer, got '0'"
