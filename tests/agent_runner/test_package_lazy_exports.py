"""Unit tests for the import cost of the :mod:`osprey.agent_runner` package.

The package is the agent harness adapter. Most of its modules drive an agent
through the agent SDK, but some serve callers that never drive one. The tests
here hold the invariant that importing the package, or any one module of it,
costs only that module: the agent SDK loads only when a name that needs it is
first read.

The import checks run in a fresh interpreter, not on the already-populated
``sys.modules`` of the test session.
"""

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import osprey.agent_runner

_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _modules_added_by_import(module: str) -> set[str]:
    """Return the module names a fresh import of ``module`` adds to ``sys.modules``.

    Args:
        module: Dotted name to import in a child interpreter.

    Returns:
        Every module name the import added, stdlib included.

    Raises:
        AssertionError: If the child interpreter fails to import the module.
    """
    code = (
        "import json, sys;"
        "before = set(sys.modules);"
        f"import {module};"
        "print(json.dumps(sorted(set(sys.modules) - before)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=_SRC),
        check=False,
    )
    assert result.returncode == 0, f"fresh import of {module} failed:\n{result.stderr}"
    return set(json.loads(result.stdout))


def test_importing_a_leaf_costs_only_that_leaf():
    """A stdlib-only module of the package loads no sibling and no agent SDK."""
    added = _modules_added_by_import("osprey.agent_runner.project_paths")

    assert not [name for name in added if name.split(".")[0] == "claude_agent_sdk"]
    assert "osprey.agent_runner.primitives" not in added


def test_public_names_are_exactly_the_lazy_map():
    """``__all__`` lists every lazily resolved name and nothing else."""
    import osprey.agent_runner

    assert sorted(osprey.agent_runner.__all__) == sorted(osprey.agent_runner._LAZY_EXPORTS)


@pytest.mark.parametrize(
    "module",
    [
        "osprey.agent_runner.tool_names",
        "osprey.cli.templates.claude_code",
        "osprey.deployment.web_terminals.artifacts",
    ],
)
def test_reading_a_tool_name_list_loads_no_agent_sdk(module):
    """The build and deploy layers read these lists and never start an agent."""
    added = _modules_added_by_import(module)

    assert not [name for name in added if name.split(".")[0] == "claude_agent_sdk"]


def test_importing_the_package_loads_no_module_of_it_and_no_agent_sdk():
    """The package root resolves its exports on access, so it imports no module."""
    added = _modules_added_by_import("osprey.agent_runner")

    assert not [name for name in added if name.split(".")[0] == "claude_agent_sdk"]
    assert not [name for name in added if name.startswith("osprey.agent_runner.")]


@pytest.mark.parametrize(
    "module",
    [
        "osprey.agent_runner.build_artifacts",
        "osprey.cli.templates.claude_code",
        "osprey.cli.scaffold_cmd",
        "osprey.cli.templates.manifest",
        "osprey.interfaces.web_terminal.scaffold_gallery_service",
    ],
)
def test_the_build_artifact_catalog_costs_no_agent_sdk(module: str) -> None:
    """The catalog's build-time consumers never drive an agent, so they load no agent SDK."""
    added = _modules_added_by_import(module)

    assert not [name for name in added if name.split(".")[0] == "claude_agent_sdk"]


@pytest.mark.parametrize(
    "module",
    ["osprey.agent_runner.claude_state", "osprey.agent_runner.artifact_resolve"],
)
def test_an_sdk_free_submodule_imports_without_the_sdk(module: str) -> None:
    """Deployment code and the container entrypoint read these modules without driving an agent."""
    added = _modules_added_by_import(module)

    assert not [name for name in added if name.split(".")[0] == "claude_agent_sdk"]
    assert "osprey.agent_runner.primitives" not in added


def test_every_public_name_is_its_defining_modules_object() -> None:
    """Each public name resolves to the very object its defining module holds."""
    import osprey.agent_runner as package

    assert set(package.__all__) == set(package._LAZY_EXPORTS)
    for name, leaf in package._LAZY_EXPORTS.items():
        defining = importlib.import_module(leaf, package.__name__)
        assert getattr(package, name) is getattr(defining, name), name


def test_an_unknown_name_is_an_attribute_error() -> None:
    """An unknown name raises, so ``from package import submodule`` falls back to the submodule."""
    import osprey.agent_runner as package

    with pytest.raises(AttributeError):
        package.no_such_name

    from osprey.agent_runner import clean_env

    assert clean_env is importlib.import_module("osprey.agent_runner.clean_env")


def test_importing_a_submodule_does_not_load_the_agent_sdk() -> None:
    """A module of the package costs only itself: no agent SDK, no primitives."""
    added = _modules_added_by_import("osprey.agent_runner.provider_env")

    assert "claude_agent_sdk" not in added
    assert "osprey.agent_runner.primitives" not in added


def test_the_typed_imports_and_the_lazy_map_name_the_same_objects() -> None:
    """The ``TYPE_CHECKING`` imports and the lazy map list the same names from the same modules."""
    tree = ast.parse(Path(osprey.agent_runner.__file__).read_text(encoding="utf-8"))
    typed: dict[str, str] = {}
    for node in tree.body:
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            continue
        for stmt in node.body:
            assert isinstance(stmt, ast.ImportFrom) and stmt.module is not None
            submodule = "." + stmt.module.removeprefix("osprey.agent_runner.")
            for alias in stmt.names:
                typed[alias.asname or alias.name] = submodule

    assert typed == osprey.agent_runner._LAZY_EXPORTS
    assert sorted(osprey.agent_runner.__all__) == sorted(osprey.agent_runner._LAZY_EXPORTS)
