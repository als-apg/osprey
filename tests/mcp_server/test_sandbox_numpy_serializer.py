"""The numpy branch of ``save_artifact()`` inside the sandboxed workspace executor.

The sandbox injects the same ``save_artifact()`` source as the python executor
(``osprey.stores.artifact_manifest.SAVE_ARTIFACT_SOURCE``), which delegates to
``artifact_store.serialize_object`` — pinned in
``tests/stores/test_numpy_serializer_parity.py``. What is specific here is the
sandbox itself: its import allow-list and patched ``open()`` must still let that
delegation work, so every test executes real user code through
``execute_sandbox_code`` the way the rest of the sandbox suite does.
"""

from __future__ import annotations

import textwrap
from unittest.mock import patch

import numpy as np
import pytest

from osprey.mcp_server.workspace.execution.sandbox_executor import execute_sandbox_code

pytestmark = pytest.mark.unit

NPY_MIME = "application/octet-stream"
NPY_TYPE = "file"


@pytest.fixture
def execution_folder(tmp_path):
    """Create a temp execution folder."""
    folder = tmp_path / "test_execution"
    folder.mkdir()
    return folder


@pytest.fixture
def workspace_root(tmp_path):
    """Create a temp workspace root."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "data").mkdir()
    return ws


async def _run(code: str, execution_folder, workspace_root):
    """Execute sandboxed user code and return the result."""
    with patch(
        "osprey.utils.workspace.resolve_workspace_root",
        return_value=workspace_root,
    ):
        return await execute_sandbox_code(
            code=textwrap.dedent(code),
            execution_folder=execution_folder,
            timeout=60,
        )


class TestNumpyBranch:
    async def test_ndarray_saves_as_npy(self, execution_folder, workspace_root):
        result = await _run(
            """\
            import numpy as np
            save_artifact(np.arange(12, dtype=np.uint16).reshape(3, 4), "Camera Frame")
            """,
            execution_folder,
            workspace_root,
        )

        assert result.success, result.stderr
        assert len(result.artifacts) == 1
        artifact = result.artifacts[0]
        assert artifact["path"].suffix == ".npy"
        assert artifact["artifact_type"] == NPY_TYPE
        assert artifact["mime_type"] == NPY_MIME

    async def test_saved_npy_round_trips(self, execution_folder, workspace_root):
        result = await _run(
            """\
            import numpy as np
            save_artifact(np.arange(12, dtype=np.uint16).reshape(3, 4), "Camera Frame")
            """,
            execution_folder,
            workspace_root,
        )

        assert result.success, result.stderr
        loaded = np.load(result.artifacts[0]["path"], allow_pickle=False)

        np.testing.assert_array_equal(loaded, np.arange(12, dtype=np.uint16).reshape(3, 4))
        assert loaded.dtype == np.uint16

    async def test_one_dimensional_array_round_trips(self, execution_folder, workspace_root):
        result = await _run(
            """\
            import numpy as np
            save_artifact(np.linspace(0.0, 1.0, 5), "Reading")
            """,
            execution_folder,
            workspace_root,
        )

        assert result.success, result.stderr
        assert result.artifacts[0]["path"].suffix == ".npy"
        np.testing.assert_array_equal(
            np.load(result.artifacts[0]["path"], allow_pickle=False),
            np.linspace(0.0, 1.0, 5),
        )

    async def test_object_dtype_falls_back_to_text(self, execution_folder, workspace_root):
        """``np.save`` cannot write object dtypes without pickle, which stays off."""
        result = await _run(
            """\
            import numpy as np
            save_artifact(np.array([{"a": 1}, None], dtype=object), "Weird")
            """,
            execution_folder,
            workspace_root,
        )

        assert result.success, result.stderr
        artifact = result.artifacts[0]
        assert artifact["path"].suffix == ".txt"
        assert artifact["artifact_type"] == "text"
        assert artifact["mime_type"] == "text/plain"


class TestExistingTypesUnchanged:
    """Scope guard: the numpy branch must not perturb any existing sniffing."""

    async def test_str_dict_and_bytes_keep_their_types(self, execution_folder, workspace_root):
        result = await _run(
            """\
            save_artifact("# Hello World", "String Test")
            save_artifact({"key": "value", "count": 42}, "Dict Test")
            save_artifact(b"binary content here", "Bytes Test")
            """,
            execution_folder,
            workspace_root,
        )

        assert result.success, result.stderr
        decisions = [
            (a["artifact_type"], a["path"].suffix, a["mime_type"]) for a in result.artifacts
        ]
        assert decisions == [
            ("markdown", ".md", "text/markdown"),
            ("json", ".json", "application/json"),
            ("binary", ".bin", "application/octet-stream"),
        ]
