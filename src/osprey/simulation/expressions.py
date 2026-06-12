"""Safe arithmetic expression evaluation for simulation machine files.

Expressions are parsed with :mod:`ast` and evaluated by walking the tree —
``eval()`` is never used. The grammar is intentionally tiny: int/float
literals, ``+ - * / **``, unary minus, parentheses, the functions ``abs``,
``min``, ``max``, ``sqrt``, ``exp``, and ``ch('PV:NAME')`` channel references.
Everything else is rejected at parse time.
"""

import ast
import math
from collections.abc import Callable

__all__ = ["ExpressionError", "compile_expression", "evaluate", "extract_channel_refs"]


class ExpressionError(ValueError):
    """Raised when an expression is syntactically or semantically invalid."""


_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}

_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "exp": math.exp,
}

_CHANNEL_FUNC = "ch"


def compile_expression(source: str) -> ast.expr:
    """Parse and validate an expression, returning its AST body.

    Args:
        source: Expression string, e.g. ``"max(0.0, 98.5 - ch('PV:X'))"``.

    Returns:
        The validated AST expression node, ready for :func:`evaluate`.

    Raises:
        ExpressionError: If the expression has invalid syntax or contains
            disallowed constructs.
    """
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Invalid expression syntax {source!r}: {exc.msg}") from exc
    _validate(tree.body, source)
    return tree.body


def extract_channel_refs(node: ast.expr) -> set[str]:
    """Return the set of channel names referenced via ``ch(...)`` calls."""
    refs: set[str] = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == _CHANNEL_FUNC
            and sub.args
            and isinstance(sub.args[0], ast.Constant)
            and isinstance(sub.args[0].value, str)
        ):
            refs.add(sub.args[0].value)
    return refs


def evaluate(node: ast.expr, resolver: Callable[[str], float]) -> float:
    """Evaluate a validated expression node.

    Node invariants (numeric literals, allowed operators/functions,
    string-literal ``ch()`` arguments) are enforced by
    :func:`compile_expression` at parse time and are not re-checked here.

    Args:
        node: AST node previously returned by :func:`compile_expression`.
        resolver: Callable mapping a channel name to its numeric value;
            invoked for every ``ch('NAME')`` reference.

    Returns:
        The numeric result of the expression.
    """
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS[type(node.op)]
        return float(op(evaluate(node.left, resolver), evaluate(node.right, resolver)))
    if isinstance(node, ast.UnaryOp):
        operand = evaluate(node.operand, resolver)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == _CHANNEL_FUNC:
            return float(resolver(node.args[0].value))
        func = _FUNCTIONS[node.func.id]
        return float(func(*[evaluate(arg, resolver) for arg in node.args]))
    raise ExpressionError(f"Cannot evaluate node type {type(node).__name__!r}")


def _validate(node: ast.AST, source: str) -> None:
    """Recursively validate that a node only uses the allowed grammar."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(
                f"Only int/float literals are allowed, got {node.value!r} in {source!r}"
            )
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY_OPS:
            raise ExpressionError(
                f"Operator {type(node.op).__name__!r} is not allowed in {source!r}"
            )
        _validate(node.left, source)
        _validate(node.right, source)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise ExpressionError(
                f"Unary operator {type(node.op).__name__!r} is not allowed in {source!r}"
            )
        _validate(node.operand, source)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError(f"Only simple function calls are allowed in {source!r}")
        if node.keywords:
            raise ExpressionError(f"Keyword arguments are not allowed in {source!r}")
        if node.func.id == _CHANNEL_FUNC:
            if (
                len(node.args) != 1
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
            ):
                raise ExpressionError(
                    f"ch() requires a single string-literal channel name in {source!r}"
                )
            return
        if node.func.id not in _FUNCTIONS:
            raise ExpressionError(
                f"Function {node.func.id!r} is not allowed in {source!r}. "
                f"Allowed: {sorted(_FUNCTIONS)} and {_CHANNEL_FUNC}()"
            )
        if not node.args:
            raise ExpressionError(f"{node.func.id}() requires at least one argument in {source!r}")
        for arg in node.args:
            _validate(arg, source)
        return
    raise ExpressionError(f"Disallowed syntax {type(node).__name__!r} in expression {source!r}")
