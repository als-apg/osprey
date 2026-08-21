"""Both execution wrappers inject the one shared ``save_artifact()``.

The python executor and the visualization sandbox each generate a subprocess
script. A wrapper that re-inlined its own ``save_artifact()`` would compile and
run fine while quietly growing a second serializer — so the guard here is on
the generated text: the shared source must appear in both scripts verbatim.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from osprey.mcp_server.workspace.execution.sandbox_executor import _create_sandbox_wrapper
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey.stores.artifact_manifest import SAVE_ARTIFACT_SOURCE

pytestmark = pytest.mark.unit


def test_shared_source_is_valid_python():
    ast.parse(SAVE_ARTIFACT_SOURCE)
    assert "def save_artifact(" in SAVE_ARTIFACT_SOURCE


def test_python_executor_wrapper_injects_the_shared_source():
    script = ExecutionWrapper().create_wrapper("x = 1")

    assert SAVE_ARTIFACT_SOURCE.strip() in script
    assert script.count("def save_artifact(") == 1


def test_visualization_sandbox_injects_the_shared_source(tmp_path: Path):
    script = _create_sandbox_wrapper(
        "x = 1",
        execution_folder=tmp_path / "exec",
        workspace_root=tmp_path / "ws",
        project_root=tmp_path,
    )

    assert SAVE_ARTIFACT_SOURCE in script
    assert script.count("def save_artifact(") == 1
