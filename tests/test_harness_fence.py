"""The agent SDK is imported only inside the harness adapter.

``src/osprey/agent_runner/`` is the one package that imports the agent SDK;
every other shipped module reaches the agent through it. Two checks hold that
line. Ruff's banned-api rule (``TID251``, configured in ``pyproject.toml``)
refuses the import on every lint run, in the pre-commit hook and in an editor.
The source scan here refuses it under pytest as well, and also sees the two
dynamic spellings ruff cannot: ``importlib.import_module`` and ``__import__``
called with a literal module name.

The scope is the shipped trees, ``src/`` and ``packages/``. The adapter is the
one exemption, and it is a directory rather than a list of modules. Tests and
developer scripts live outside the shipped trees and drive the SDK directly.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SDK = "claude_agent_sdk"
SHIPPED_TREES = ("src", "packages")
FENCE = REPO_ROOT / "src" / "osprey" / "agent_runner"
_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})


def _names_sdk(module: str | None) -> bool:
    return module is not None and (module == SDK or module.startswith(f"{SDK}."))


def _dynamic_target(call: ast.Call) -> str | None:
    """The literal module name a dynamic import call loads, if it has one."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name not in _DYNAMIC_IMPORTERS or not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def sdk_import_lines(source: str) -> list[int]:
    """Line numbers in *source* that import the agent SDK, in any spelling."""
    if SDK not in source:
        return []
    lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(_names_sdk(a.name) for a in node.names):
            lines.add(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and _names_sdk(node.module):
            lines.add(node.lineno)
        elif isinstance(node, ast.Call) and _names_sdk(_dynamic_target(node)):
            lines.add(node.lineno)
    return sorted(lines)


def _shipped_modules() -> list[Path]:
    return sorted(
        path
        for tree in SHIPPED_TREES
        for path in (REPO_ROOT / tree).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _inside_fence(path: Path) -> bool:
    return path.is_relative_to(FENCE)


def test_no_shipped_module_outside_the_adapter_imports_the_sdk() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{line}"
        for path in _shipped_modules()
        if not _inside_fence(path)
        for line in sdk_import_lines(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"{SDK} is imported outside src/osprey/agent_runner/: {offenders}. "
        "Reach the agent through osprey.agent_runner."
    )


def test_the_adapter_is_where_the_sdk_is_imported() -> None:
    importers = [
        path
        for path in _shipped_modules()
        if _inside_fence(path) and sdk_import_lines(path.read_text(encoding="utf-8"))
    ]
    assert importers, f"no module under {FENCE.relative_to(REPO_ROOT)} imports {SDK}"


@pytest.mark.parametrize(
    "source",
    [
        "import claude_agent_sdk\n",
        "import claude_agent_sdk.types as sdk_types\n",
        "from claude_agent_sdk import query\n",
        "from claude_agent_sdk.types import TextBlock\n",
        "def run():\n    from claude_agent_sdk import query\n",
        "if TYPE_CHECKING:\n    from claude_agent_sdk import ClaudeAgentOptions\n",
        "importlib.import_module('claude_agent_sdk')\n",
        "__import__('claude_agent_sdk.types')\n",
    ],
    ids=[
        "import",
        "import-submodule",
        "from-import",
        "from-submodule",
        "function-local",
        "type-checking",
        "import-module",
        "dunder-import",
    ],
)
def test_the_scan_sees_every_spelling_of_the_import(source: str) -> None:
    last_line = source.rstrip("\n").count("\n") + 1
    assert sdk_import_lines(source) == [last_line]


@pytest.mark.parametrize(
    "source",
    [
        "from osprey.agent_runner import stream_query\n",
        "import claude_agent_sdk_extras\n",
        "from . import claude_agent_sdk\n",
        "MESSAGE = 'claude_agent_sdk is not installed'\n",
        "LOGGERS = ('claude_agent_sdk',)\n",
    ],
    ids=["adapter", "prefix-only", "relative", "message-text", "logger-name"],
)
def test_the_scan_passes_what_is_not_the_sdk(source: str) -> None:
    assert sdk_import_lines(source) == []


_PLANTED = """\
from __future__ import annotations

from typing import TYPE_CHECKING

import claude_agent_sdk.types
from claude_agent_sdk import query

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions


def run(options: ClaudeAgentOptions) -> object:
    import claude_agent_sdk

    return claude_agent_sdk, claude_agent_sdk.types, query, options
"""
_PLANTED_LINES = [5, 6, 9, 13]


def _ruff_banned_api_lines(filename: str) -> list[int]:
    """Lines ruff reports as TID251 for the planted source linted as *filename*."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--output-format=json",
            "--stdin-filename",
            filename,
            "-",
        ],
        cwd=REPO_ROOT,
        input=_PLANTED,
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return sorted(d["location"]["row"] for d in json.loads(result.stdout) if d["code"] == "TID251")


@pytest.mark.parametrize(
    "filename",
    [
        "src/osprey/utils/_planted.py",
        "src/osprey/agent_runner_extras/_planted.py",
        "packages/osprey-connectors/src/osprey_connectors/_planted.py",
    ],
    ids=["src", "adapter-name-prefix", "packages"],
)
def test_ruff_refuses_the_sdk_outside_the_adapter(filename: str) -> None:
    assert _ruff_banned_api_lines(filename) == _PLANTED_LINES


@pytest.mark.parametrize(
    "filename",
    [
        "src/osprey/agent_runner/_planted.py",
        "src/osprey/agent_runner/nested/_planted.py",
        "tests/_planted.py",
        "scripts/_planted.py",
    ],
    ids=["adapter", "adapter-subpackage", "tests", "scripts"],
)
def test_ruff_admits_the_sdk_in_the_adapter_and_outside_the_shipped_trees(filename: str) -> None:
    assert _ruff_banned_api_lines(filename) == []
