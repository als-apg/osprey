"""Every ``error_type`` an MCP tool can emit is one the error-guidance hook knows.

The PostToolUse hook (``osprey_error_guidance.py``) classes each error envelope
by looking its ``error_type`` up in ``ERROR_CLASS_MAP``, and a value that is
not in that table falls through to "Internal" — "report verbatim, have an
operator check the server logs". For a bridge that is merely unreachable, a
lookup that merely missed, or a refusal the agent must not work around, that
is the wrong protocol. So the map has to know every value the servers emit,
and this test is the producer-side check that it does.

The scan is static: it walks every module under ``src/osprey/mcp_server`` and
collects the ``error_type`` from

- every ``make_error(...)`` call, positional or keyword, whether the value is
  a string literal or a module-level constant (resolved across the package,
  since ``REASON_*`` and ``ERROR_*`` names are imported between modules), and
  the default of any function parameter named ``error_type``;
- every ``"error_type": ...`` key in a dict literal (a hand-rolled envelope);
- every ``error_type = "..."`` assignment (a class attribute on an error
  family such as ``GraphStoreError``, or a local picked before the call);
- the keys of the Bluesky queue tools' ``_REFUSAL_HINTS`` table — the codes a
  bridge refusal is relayed under verbatim, which no call site spells.

A value the scan cannot resolve (``exc.error_type``, a ``code`` read from a
response body) is skipped; the families behind those are pinned by their own
tests (``test_graph_server_context.py`` for the graph store) or by the
``_REFUSAL_HINTS`` keys.
"""

from __future__ import annotations

import ast
import pathlib
from collections import defaultdict

import pytest

import osprey
from tests.mcp_server.conftest import hook_error_class_map

_MCP_SERVER_ROOT = pathlib.Path(osprey.__file__).parent / "mcp_server"

#: The Bluesky queue tools relay a bridge refusal under the code the bridge
#: sent; this table's keys are the vocabulary those codes come from.
_RELAY_TABLE_NAME = "_REFUSAL_HINTS"


def _string_constants(trees: dict[pathlib.Path, ast.Module]) -> dict[str, set[str]]:
    """Every module-level ``NAME = "literal"`` across the package, by name."""
    table: dict[str, set[str]] = defaultdict(set)
    for tree in trees.values():
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    table[target.id].add(node.value.value)
    return table


def _resolve(node: ast.AST | None, constants: dict[str, set[str]]) -> set[str]:
    """The string value(s) an expression can take, or empty when it is dynamic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        return set(constants.get(node.id, ()))
    return set()


def _error_type_argument(call: ast.Call) -> ast.AST | None:
    """The expression passed as ``error_type`` to a ``make_error`` call."""
    for keyword in call.keywords:
        if keyword.arg == "error_type":
            return keyword.value
    return call.args[0] if call.args else None


def _is_make_error(call: ast.Call) -> bool:
    func = call.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    return name == "make_error"


def emitted_error_types() -> dict[str, list[str]]:
    """Scan the MCP server package for every ``error_type`` value it can emit.

    Returns:
        Each value mapped to the ``file:line`` sites that emit it.
    """
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(_MCP_SERVER_ROOT.rglob("*.py"))
    }
    constants = _string_constants(trees)
    sites: dict[str, list[str]] = defaultdict(list)

    def record(values: set[str], path: pathlib.Path, lineno: int) -> None:
        for value in values:
            sites[value].append(f"{path.relative_to(_MCP_SERVER_ROOT)}:{lineno}")

    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_make_error(node):
                record(_resolve(_error_type_argument(node), constants), path, node.lineno)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == "error_type":
                        record(_resolve(value, constants), path, node.lineno)
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                if "error_type" in names:
                    record(_resolve(node.value, constants), path, node.lineno)
                if _RELAY_TABLE_NAME in names and isinstance(node.value, ast.Dict):
                    for key in node.value.keys:
                        record(_resolve(key, constants), path, node.lineno)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                args = node.args
                for arg, default in zip(
                    args.kwonlyargs + args.args[len(args.args) - len(args.defaults) :],
                    args.kw_defaults + args.defaults,
                    strict=True,
                ):
                    if arg.arg == "error_type" and default is not None:
                        record(_resolve(default, constants), path, node.lineno)
    return dict(sites)


@pytest.fixture(scope="module")
def emitted() -> dict[str, list[str]]:
    return emitted_error_types()


@pytest.mark.unit
def test_scan_finds_the_known_emitters(emitted):
    """A scanner that silently finds nothing would pass the check below.

    These anchors cover each way a value can be spelled: a literal, a constant
    imported from another module, a function-parameter default, a class
    attribute, a hand-rolled envelope key, and a relayed bridge code.
    """
    assert len(emitted["validation_error"]) > 50  # the workhorse literal
    assert emitted["target_switch_refused"]  # ERROR_REFUSED, as a parameter default
    assert emitted["unknown_bluesky_lane"]  # REASON_UNKNOWN_LANE, imported from lanes.py
    assert emitted["not_configured"]  # GraphNotConfigured.error_type, a class attribute
    assert emitted["interrupted_item_in_queue"]  # a bridge code, via _REFUSAL_HINTS
    assert emitted["session_target_mismatch"]  # a constant used as a _REFUSAL_HINTS key


@pytest.mark.unit
def test_every_emitted_error_type_is_a_hook_class_key(emitted):
    """No tool emits an ``error_type`` the guidance hook would class as Internal by default."""
    known = hook_error_class_map()
    unmapped = {value: sites for value, sites in emitted.items() if value not in known}
    assert not unmapped, "error_type values not in ERROR_CLASS_MAP (the hook classes these as " + (
        "Internal — 'check the server logs' — whatever they mean):\n"
        + "\n".join(
            f"  {value!r}: {', '.join(sites[:3])}{' …' if len(sites) > 3 else ''}"
            for value, sites in sorted(unmapped.items())
        )
    )
