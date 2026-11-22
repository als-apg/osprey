"""Every host path enters generated Python source as one literal via ``repr``.

Each generator below writes a script a subprocess runs. A folder name holding a
quote or ending in a backslash must still compile and bind the exact path. A new
generator that embeds a path joins the ``GENERATORS`` list.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from osprey.mcp_server.workspace.execution.sandbox_executor import _create_sandbox_wrapper
from osprey.services.bluesky_bridge.plan_validation import _render_dry_run_script
from osprey.services.python_executor.execution.wrapper import (
    WRITES_LEDGER_FILENAME,
    ExecutionWrapper,
)

HOSTILE_FOLDER_NAMES = ['a"b', "ends-in\\", "it's"]


def _sandbox(folder: Path) -> tuple[str, dict[str, Path]]:
    source = _create_sandbox_wrapper("x = 1", folder, folder, folder, secret_roots=())
    return source, {"_execution_dir": folder}


def _python_executor(folder: Path) -> tuple[str, dict[str, Path]]:
    source = ExecutionWrapper(execution_mode="readwrite").create_wrapper("pass", folder)
    return source, {
        "_execution_dir": folder,
        "_write_ledger_path": folder / WRITES_LEDGER_FILENAME,
    }


def _plan_dry_run(folder: Path) -> tuple[str, dict[str, Path]]:
    result_path = folder / "result.json"
    source = _render_dry_run_script(
        plan_path=folder,
        result_path=result_path,
        plan_name="p",
        sample_args={},
        fallback_movable_names=(),
        fallback_readable_names=(),
        point_count_key="k",
        inner_timeout=1.0,
    )
    return source, {"_PLAN_PATH": folder, "_RESULT_PATH": result_path}


GENERATORS: list[Callable[[Path], tuple[str, dict[str, Path]]]] = [
    _sandbox,
    _python_executor,
    _plan_dry_run,
]


def _bound_string(tree: ast.AST, name: str) -> str:
    """The string a module-level-or-nested ``name = ...`` binds, unwrapping ``Path(...)``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        assert isinstance(value, ast.Constant), f"{name} is not bound to a literal"
        return value.value
    raise AssertionError(f"no assignment to {name} in the generated source")


@pytest.mark.parametrize("generator", GENERATORS, ids=lambda g: g.__name__.lstrip("_"))
@pytest.mark.parametrize("folder_name", HOSTILE_FOLDER_NAMES)
def test_generated_source_binds_the_exact_path(
    tmp_path: Path, folder_name: str, generator: Callable[[Path], tuple[str, dict[str, Path]]]
) -> None:
    folder = tmp_path / folder_name
    folder.mkdir()
    source, bindings = generator(folder)

    compile(source, f"<{generator.__name__}>", "exec")
    tree = ast.parse(source)
    for name, expected in bindings.items():
        assert _bound_string(tree, name) == str(expected)
