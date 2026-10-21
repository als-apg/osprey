"""The model write token never reaches text, asserted against the syntax tree.

This one check stays static: it must cover every line that could log,
print, warn, format or raise the token, which no behavioural test can. It
scans the serving package and the entrypoint the token enters the process
through.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

VIRTUAL_ACCELERATOR = (
    Path(__file__).resolve().parents[2] / "src/osprey/services/virtual_accelerator"
)
SERVING = VIRTUAL_ACCELERATOR / "serving"
ENTRYPOINT = VIRTUAL_ACCELERATOR / "entrypoint.py"

#: A reference to the model write token, by every name it has in the serving
#: package: the runner's constructor argument and the attribute it is kept
#: on, the attribute the surface keeps it on, and the name a write receives
#: it under.
_TOKEN_NAMES = frozenset({"model_write_token", "_model_write_token", "_write_token", "token"})

#: Calls whose arguments reach a log, a terminal or a warning.
_OUTPUT_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)


def _mentions_token(node: ast.AST) -> bool:
    """Whether the token can reach the text *node* emits.

    A token that is only the test of a conditional expression chooses which
    text is emitted and never joins it, so it is not a mention.
    """
    tests = {
        id(each)
        for conditional in ast.walk(node)
        if isinstance(conditional, ast.IfExp)
        for each in ast.walk(conditional.test)
    }
    return any(
        id(each) not in tests
        and (
            (isinstance(each, ast.Name) and each.id in _TOKEN_NAMES)
            or (isinstance(each, ast.Attribute) and each.attr in _TOKEN_NAMES)
        )
        for each in ast.walk(node)
    )


def _token_leaks(tree: ast.AST) -> list[str]:
    """Every place in ``tree`` that would put the token into text.

    That is a logging, ``print`` or ``warnings.warn`` call, a ``format``
    call, an f-string, a ``%`` interpolation or a raised exception, any of
    which mentions the token. Passing the token on as an argument to another
    call is none of these, which is how the model surface receives it.
    """
    leaks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            emits = (isinstance(func, ast.Name) and func.id == "print") or (
                isinstance(func, ast.Attribute) and func.attr in _OUTPUT_METHODS | {"format"}
            )
        else:
            emits = isinstance(node, (ast.JoinedStr, ast.Raise)) or (
                isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
            )
        if emits and _mentions_token(node):
            leaks.append(ast.unparse(node))
    return leaks


@pytest.fixture(scope="module")
def serving_modules() -> list[Path]:
    modules = [*sorted(SERVING.glob("*.py")), ENTRYPOINT]
    assert ENTRYPOINT.is_file()
    return modules


def test_the_model_write_token_is_never_logged(serving_modules: list[Path]) -> None:
    """Nor printed, warned, formatted into text or raised, by any module of
    the serving package: the token is the one thing that gates a model write,
    and a log is readable by far more people than the write is allowed to."""
    leaks = {
        module.name: _token_leaks(ast.parse(module.read_text(encoding="utf-8")))
        for module in serving_modules
    }
    assert {name: found for name, found in leaks.items() if found} == {}


@pytest.mark.parametrize(
    "leak",
    [
        "LOG.info('armed with %s', self._model_write_token)",
        "print(model_write_token)",
        "warnings.warn(f'token {model_write_token}')",
        "text = 'token %s' % self._model_write_token",
        "raise ValueError(model_write_token)",
        "text = '{}'.format(self._model_write_token)",
        "LOG.info('got %s', request.token)",
    ],
)
def test_the_leak_check_catches_a_leak(leak: str) -> None:
    """The check above is only as good as its detector: each of these would
    put the token into text, and each is caught."""
    assert _token_leaks(ast.parse(leak))


def test_a_token_that_only_chooses_the_text_is_not_a_leak() -> None:
    """A conditional's test picks which text is emitted; the token joins none of it."""
    chooses = ast.parse("print('armed' if model_write_token else 'disabled')")
    assert _token_leaks(chooses) == []
    # The same token inside a branch is emitted text, and is caught.
    emits = ast.parse("print(model_write_token if model_write_token else 'disabled')")
    assert _token_leaks(emits)


def test_the_leak_check_lets_the_token_be_passed_on() -> None:
    """Handing the token to the object that checks it is not a leak."""
    passed_on = ast.parse(
        "surface = ModelSurface.for_view(model_write_token=self._model_write_token)"
    )
    assert _token_leaks(passed_on) == []
