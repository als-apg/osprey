"""Tests for the safe AST expression evaluator."""

import math

import pytest

from osprey.simulation.expressions import (
    ExpressionError,
    compile_expression,
    evaluate,
    extract_channel_refs,
)


def _eval(source, channels=None):
    channels = channels or {}
    return evaluate(compile_expression(source), lambda name: channels[name])


class TestAllowedExpressions:
    """Allowed grammar evaluates correctly."""

    def test_arithmetic_operators(self):
        assert _eval("1 + 2 * 3") == 7.0
        assert _eval("10 / 4") == 2.5
        assert _eval("2 ** 3 - 1") == 7.0
        assert _eval("(1 + 2) * 3") == 9.0

    def test_unary_minus(self):
        assert _eval("-5 + 2") == -3.0
        assert _eval("-(2 + 3)") == -5.0

    def test_float_and_int_literals(self):
        assert _eval("3.5e-7") == pytest.approx(3.5e-7)
        assert _eval("42") == 42.0

    def test_functions(self):
        assert _eval("abs(-4.2)") == pytest.approx(4.2)
        assert _eval("min(3, 1, 2)") == 1.0
        assert _eval("max(0.0, -5.0)") == 0.0
        assert _eval("sqrt(16)") == 4.0
        assert _eval("exp(0)") == 1.0
        assert _eval("exp(1)") == pytest.approx(math.e)

    def test_channel_reference(self):
        assert _eval("ch('PV:A') * 2", {"PV:A": 21.0}) == 42.0

    def test_nested(self):
        result = _eval(
            "max(0.0, 98.5 - 0.85 * abs(ch('PV:SP') - 42.0)) * ch('PV:STATUS')",
            {"PV:SP": 28.4, "PV:STATUS": 1.0},
        )
        assert result == pytest.approx(98.5 - 0.85 * 13.6)


class TestRejectedExpressions:
    """Everything outside the tiny grammar is rejected at compile time."""

    @pytest.mark.parametrize(
        "source",
        [
            "1 < 2",  # comparison
            "1 if 2 else 3",  # conditional
            "x + 1",  # bare name
            "__import__('os')",  # unknown function
            "f(1)",  # unknown function
            "(1).real",  # attribute access
            "[1, 2]",  # list literal
            "{'a': 1}",  # dict literal
            "'abc'",  # bare string literal
            "ch(PV)",  # non-string ch() argument
            "ch('A', 'B')",  # ch() with two args
            "ch()",  # ch() with no args
            "min()",  # function with no args
            "max(1, b=2)",  # keyword arguments
            "abs.__call__(1)",  # attribute call
            "1 and 2",  # boolean op
            "5 % 2",  # modulo not allowed
            "~1",  # bitwise unary
            "lambda: 1",  # lambda
            "True",  # bool literal
        ],
    )
    def test_rejected(self, source):
        with pytest.raises(ExpressionError):
            compile_expression(source)

    def test_syntax_error(self):
        with pytest.raises(ExpressionError, match="syntax"):
            compile_expression("1 +")


class TestChannelRefs:
    """Channel reference extraction."""

    def test_extract_refs(self):
        node = compile_expression("ch('PV:A') + ch('PV:B') * abs(ch('PV:A'))")
        assert extract_channel_refs(node) == {"PV:A", "PV:B"}

    def test_no_refs(self):
        node = compile_expression("1 + 2")
        assert extract_channel_refs(node) == set()
