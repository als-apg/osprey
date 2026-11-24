"""Tests fake a module's own ``subprocess`` / ``shutil`` name, never the stdlib module.

After a module does ``import subprocess``, ``<module>.subprocess`` *is* the
stdlib module object. A patch spelled through it — or through the stdlib
module directly — replaces the function for the whole process: every thread,
every background server and every other module in the worker sees the fake
while it is active. The scoped form rebinds the importing module's own name to
a stand-in (``tests/_scoped_stdlib.py``), so only that module sees the fake.

The test-tree walk refuses a ``patch`` / ``patch.object`` / ``patch.multiple``
/ ``setattr`` / ``delattr`` call whose target resolves to the stdlib module
itself, in each spelling it can take:

* a string naming an attribute of the module, ``"<dotted>.subprocess.run"``
  or ``"subprocess.run"``;
* an attribute access ending in the module name as the object argument,
  ``<module>.subprocess``;
* the bare module, ``subprocess``, or a name it was imported as.

The scoped helper only reaches a module that looks the stdlib name up at call
time, so the source walk pins that for ``src/`` and ``packages/*/src``: the
module imports ``subprocess`` / ``shutil`` at its top, and no function default
or class attribute binds one of their callables when the module is imported.
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import threading
from pathlib import Path

import pytest

from tests import _scoped_stdlib
from tests._scoped_stdlib import ChildProcessRefused, no_child_process, scoped
from tests._tree_scan import product_sources

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The stdlib modules whose functions a test fakes through a module's own name.
STDLIB_MODULES = frozenset({"subprocess", "shutil"})

#: Callees that replace an attribute, by their last name.
_PATCHERS = frozenset({"patch", "object", "multiple", "setattr", "delattr"})

_WIDE_TARGET = re.compile(rf"^(?:[\w.]+\.)?(?:{'|'.join(sorted(STDLIB_MODULES))})(?:\.\w+)+$")

_HELPER = "tests/_scoped_stdlib.py"


def _test_sources() -> list[tuple[str, str]]:
    """Every Python file of the framework's and each package's test tree, with its text."""
    roots = [REPO_ROOT / "tests", *sorted((REPO_ROOT / "packages").glob("*/tests"))]
    return [
        (path.relative_to(REPO_ROOT).as_posix(), path.read_text(encoding="utf-8"))
        for root in roots
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _stdlib_names(tree: ast.Module) -> frozenset[str]:
    """Every name *tree* binds to ``subprocess`` / ``shutil``, aliases included."""
    names = set(STDLIB_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                alias.asname
                for alias in node.names
                if alias.asname and alias.name.rpartition(".")[2] in STDLIB_MODULES
            )
    return frozenset(names)


def _target(node: ast.Call) -> ast.expr | None:
    """The patcher's target: its first positional argument, else its ``target=`` keyword."""
    if node.args:
        return node.args[0]
    return next((kw.value for kw in node.keywords if kw.arg == "target"), None)


def _wide_target(node: ast.expr, stdlib_names: frozenset[str]) -> bool:
    """Whether *node*, the target of a patcher, names the stdlib module itself."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(_WIDE_TARGET.match(node.value))
    if isinstance(node, ast.Name):
        return node.id in stdlib_names
    if isinstance(node, ast.Attribute):
        return node.attr in STDLIB_MODULES
    return False


def wide_patches(source: str) -> list[int]:
    """The lines of *source* that patch ``subprocess`` / ``shutil`` for the whole process."""
    tree = ast.parse(source)
    stdlib_names = _stdlib_names(tree)
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _callee(node) in _PATCHERS
        and (target := _target(node)) is not None
        and _wide_target(target, stdlib_names)
    )


def _binds_a_stdlib_callable(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in STDLIB_MODULES
        and callable(getattr(importlib.import_module(node.value.id), node.attr, None))
    )


def import_time_lookups(source: str) -> list[int]:
    """The lines of *source* that fix a ``subprocess`` / ``shutil`` lookup out of a scoped fake's reach.

    A function-local import reads ``sys.modules`` instead of the module's own
    name; a function default or class attribute holds the callable the module
    saw when it was imported.
    """
    tree = ast.parse(source)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            defaults = [*args.defaults, *(d for d in args.kw_defaults if d is not None)]
            lines.extend(d.lineno for d in defaults if _binds_a_stdlib_callable(d))
            if not isinstance(node, ast.Lambda):
                lines.extend(
                    inner.lineno
                    for inner in ast.walk(node)
                    if isinstance(inner, ast.Import)
                    and any(alias.name in STDLIB_MODULES for alias in inner.names)
                    or isinstance(inner, ast.ImportFrom)
                    and inner.module in STDLIB_MODULES
                )
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
                if value is not None and _binds_a_stdlib_callable(value):
                    lines.append(stmt.lineno)
    return sorted(set(lines))


def _sites(sources: list[tuple[str, str]], finder) -> list[str]:
    return [f"{path}:{line}" for path, text in sources for line in finder(text)]


def test_no_test_patches_subprocess_or_shutil_for_the_whole_process() -> None:
    sites = _sites(_test_sources(), wide_patches)
    assert sites == [], (
        "These patch the stdlib module itself, so the fake reaches the whole process. "
        f"Fake the module's own name with {_HELPER} instead:\n" + "\n".join(sites)
    )


def test_product_code_looks_subprocess_and_shutil_up_when_it_calls_them() -> None:
    sites = _sites(product_sources(REPO_ROOT), import_time_lookups)
    assert sites == [], (
        "Import subprocess / shutil at the top of the module and look the function up "
        f"where it is called, so a scoped fake from {_HELPER} reaches it:\n" + "\n".join(sites)
    )


@pytest.mark.parametrize(
    "source",
    [
        'patch("osprey.cli.app_cmd.subprocess.run")\n',
        'mock.patch("osprey.cli.app_cmd.shutil.which", return_value=None)\n',
        'monkeypatch.setattr("osprey.cli.app_cmd.subprocess.run", fake)\n',
        'patch("subprocess.run")\n',
        'monkeypatch.setattr("shutil.which", fake)\n',
        'monkeypatch.setattr(container_lifecycle.subprocess, "run", fake)\n',
        'patch.object(osprey.cli.app_cmd.shutil, "which")\n',
        "patch.multiple(app_cmd.subprocess, run=fake)\n",
        'monkeypatch.setattr(subprocess, "run", fake)\n',
        'setattr(shutil, "which", fake)\n',
        'patch.object(subprocess, "Popen")\n',
        'import shutil as shutil_mod\nmonkeypatch.setattr(shutil_mod, "copytree", fake)\n',
        'from os import path\nimport subprocess as sp\npatch.object(sp, "run")\n',
        'patch(target="subprocess.run")\n',
        'monkeypatch.setattr(target=shutil, name="which", value=fake)\n',
    ],
)
def test_each_wide_spelling_is_refused(source: str) -> None:
    assert wide_patches(source) == [source.count("\n")]


@pytest.mark.parametrize(
    "source",
    [
        'patch("osprey.cli.app_cmd.subprocess", new=stand_in)\n',
        'monkeypatch.setattr(app_cmd, "subprocess", stand_in)\n',
        'patch("osprey.cli.app_cmd._run")\n',
        'monkeypatch.setattr(writer.os, "fsync", fake)\n',
        'patch("osprey.cli.subprocess_tools.run")\n',
    ],
)
def test_a_scoped_patch_is_accepted(source: str) -> None:
    assert wide_patches(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    import subprocess\n",
        "def f():\n    from shutil import which\n",
        "def f(run=subprocess.run):\n    pass\n",
        "def f(*, which=shutil.which):\n    pass\n",
        "class C:\n    run = subprocess.run\n",
    ],
)
def test_each_import_time_lookup_is_refused(source: str) -> None:
    assert import_time_lookups(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\n\ndef f(run=None):\n    return (run or subprocess.run)()\n",
        'GUARD = "import subprocess"\n',
        "def f(timeout=subprocess.DEVNULL):\n    pass\n",
    ],
)
def test_a_call_time_lookup_is_accepted(source: str) -> None:
    assert import_time_lookups(source) == []


def test_a_scoped_fake_reaches_only_its_module(monkeypatch) -> None:
    fake_run = object()
    stand_in = scoped(monkeypatch, _scoped_stdlib, "importlib", run=fake_run)

    assert _scoped_stdlib.importlib.run is fake_run
    assert _scoped_stdlib.importlib.import_module is importlib.import_module
    assert not hasattr(importlib, "run")
    assert stand_in is _scoped_stdlib.importlib


def test_no_child_process_refuses_and_records_a_child_in_its_context_only() -> None:
    outside: list[int] = []
    go = threading.Event()

    def spawn_outside() -> None:
        go.wait()
        outside.append(subprocess.run(["true"], check=False).returncode)

    # Started before the context opens, so it runs in a context of its own.
    other = threading.Thread(target=spawn_outside)
    other.start()
    try:
        with no_child_process() as started:
            with pytest.raises(ChildProcessRefused):
                subprocess.run(["true"], check=False)
            go.set()
            other.join()
    finally:
        go.set()
        other.join()

    assert [event for event, _args in started] == ["subprocess.Popen"]
    assert outside == [0]
