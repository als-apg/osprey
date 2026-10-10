"""One definition decides what an environment-variable name is.

``osprey_connectors.connection.ENV_NAME_RE`` is it. The checkers that read it (a connection
block's ``auth.*_env``, the telemetry collector's ``auth.token_env``, the executor's
``child_env_passthrough``, a build profile's ``env:`` names and ``bind_env``) therefore cannot
disagree about an edge case.
"""

import ast
from pathlib import Path

import pytest

from osprey_connectors.connection import ENV_NAME_RE

REPO = Path(__file__).resolve().parent.parent
ROOTS = ("src", "packages", "scripts")
#: Below the module count of ``src`` alone (978) and above that of the other roots together
#: (82), so a walk that misses ``src`` or finds nothing fails.
MODULE_FLOOR = 500
_NAME_CLASS = "[A-Za-z_][A-Za-z0-9_]*"
#: Code generated from a schema, never edited by hand: its identifier patterns validate record
#: names, not environment-variable names, and cannot import from the connectors package.
GENERATED = ("src/osprey/facility/schema/_generated/",)
WHOLE_NAME_PATTERNS = frozenset(
    start + _NAME_CLASS + end for start in ("^", "\\A") for end in ("$", "\\Z")
)


@pytest.mark.parametrize("name", ["A", "_", "_PRIVATE", "OTLP_TOKEN", "no_proxy", "mixedCase_1"])
def test_a_variable_name_matches(name):
    assert ENV_NAME_RE.match(name)
    assert ENV_NAME_RE.fullmatch(name)


@pytest.mark.parametrize(
    "value",
    ["", "1A", "A-B", "A B", "A=B", "${A}", "A\n", "\nA", "A\nB", " A", "A "],
    ids=[
        "empty",
        "leading-digit",
        "dash",
        "space",
        "equals",
        "reference",
        "trailing-newline",
        "leading-newline",
        "embedded-newline",
        "leading-space",
        "trailing-space",
    ],
)
def test_anything_else_does_not_match_either_way(value):
    """``match`` and ``fullmatch`` agree, so no caller can pick the looser one."""
    assert ENV_NAME_RE.match(value) is None
    assert ENV_NAME_RE.fullmatch(value) is None


def _is_re_compile(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "compile"
        and isinstance(func.value, ast.Name)
        and func.value.id == "re"
    )


def test_the_connectors_package_holds_the_only_whole_name_pattern():
    modules = [path for root in ROOTS for path in sorted((REPO / root).rglob("*.py"))]
    assert len(modules) >= MODULE_FLOOR
    modules = [
        path for path in modules if not path.relative_to(REPO).as_posix().startswith(GENERATED)
    ]

    hits = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and _is_re_compile(node)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in WHOLE_NAME_PATTERNS
            ):
                hits.append(f"{path.relative_to(REPO).as_posix()}:{node.lineno}")

    defining = sorted({hit.rsplit(":", 1)[0] for hit in hits})
    assert defining == ["packages/osprey-connectors/src/osprey_connectors/connection.py"], (
        "an environment-variable name pattern is compiled outside the connectors package: "
        + ", ".join(hits)
        + "; import ENV_NAME_RE from osprey_connectors.connection instead"
    )
