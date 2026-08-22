"""Tests for the ``.npy`` branch of ``build_data_reader`` (_viz_common).

A ``channel_read`` image/waveform artifact is stored as a ``.npy`` file, so
passing its artifact id as ``data_source=`` to the plot tools must resolve to
that file and load it back as the original ndarray. The generated loader runs
inside the visualization sandbox, which has numpy but not ``osprey`` — so the
round-trip is exercised in a subprocess that mimics the sandbox preamble
instead of the test's own interpreter.
"""

import io
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

# Same names the viz preambles bind before the reader code runs.
_SANDBOX_PREAMBLE = "import numpy as np\nimport pandas as pd\n"


@pytest.fixture
def art_store(tmp_path: Path) -> Iterator:
    """Artifact store rooted in tmp_path, torn down so no state leaks."""
    from osprey.stores.artifact_store import initialize_artifact_store, reset_artifact_store

    store = initialize_artifact_store(workspace_root=tmp_path)
    yield store
    reset_artifact_store()


def _save_npy_artifact(store, array: np.ndarray, filename: str = "frame.npy") -> str:
    """Persist ``array`` as a .npy artifact, returning its artifact id."""
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    entry = store.save_file(
        file_content=buf.getvalue(),
        filename=filename,
        artifact_type="data",
        title="camera frame",
        mime_type="application/octet-stream",
        tool_source="channel_read",
    )
    return entry.id


def _run_reader_in_sandbox(data_source: str, tmp_path: Path) -> dict:
    """Run the generated reader in a fresh interpreter; return what it loaded."""
    from osprey.mcp_server.workspace.tools._viz_common import build_data_reader

    script = tmp_path / "sandbox_run.py"
    script.write_text(
        _SANDBOX_PREAMBLE
        + build_data_reader(data_source)
        + "\n"
        + "import json as _out_json\n"
        + "print('__RESULT__' + _out_json.dumps({\n"
        + "    'type': type(data).__name__,\n"
        + "    'shape': list(data.shape),\n"
        + "    'dtype': str(data.dtype),\n"
        + "    'values': data.tolist(),\n"
        + "}))\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"reader failed:\n{proc.stdout}\n{proc.stderr}"
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("__RESULT__")]
    assert marker, f"no result emitted:\n{proc.stdout}"
    result = json.loads(marker[0].removeprefix("__RESULT__"))
    result["stdout"] = proc.stdout
    return result


class TestResolveNpyArtifact:
    """A saved .npy artifact id resolves to its on-disk path."""

    def test_artifact_id_resolves_to_npy_path(self, art_store):
        from osprey.mcp_server.workspace.tools._viz_common import resolve_data_source

        array = np.arange(12, dtype=np.uint16).reshape(3, 4)
        artifact_id = _save_npy_artifact(art_store, array)

        resolved = Path(resolve_data_source(artifact_id))

        assert resolved.exists()
        assert resolved.suffix == ".npy"
        np.testing.assert_array_equal(np.load(resolved), array)

    def test_generated_code_embeds_the_resolved_path(self, art_store):
        """Resolution happens server-side: the emitted code carries an absolute path."""
        from osprey.mcp_server.workspace.tools._viz_common import build_data_reader

        artifact_id = _save_npy_artifact(art_store, np.zeros((2, 2)))

        code = build_data_reader(artifact_id)

        assert str(art_store.get_file_path(artifact_id)) in code
        assert ".npy'" in code


class TestNpyReaderRoundTrip:
    """The emitted loader returns the original ndarray inside the sandbox."""

    def test_two_dimensional_frame_round_trips(self, art_store, tmp_path: Path):
        """The FR3 case: a 2-D channel frame used as a plot data_source."""
        array = (np.arange(24, dtype=np.uint16) * 7).reshape(4, 6)
        artifact_id = _save_npy_artifact(art_store, array)

        result = _run_reader_in_sandbox(artifact_id, tmp_path)

        assert result["type"] == "ndarray"
        assert result["shape"] == [4, 6]
        assert result["dtype"] == "uint16"
        np.testing.assert_array_equal(np.array(result["values"], dtype=np.uint16), array)

    def test_one_dimensional_waveform_round_trips(self, tmp_path: Path):
        """An oversized waveform saved as .npy loads from a plain file path too."""
        array = np.linspace(-1.5, 1.5, 9, dtype=np.float64)
        fp = tmp_path / "waveform.npy"
        np.save(fp, array, allow_pickle=False)

        result = _run_reader_in_sandbox(str(fp), tmp_path)

        assert result["type"] == "ndarray"
        assert result["shape"] == [9]
        assert result["dtype"] == "float64"
        np.testing.assert_allclose(np.array(result["values"]), array)

    def test_three_dimensional_stack_keeps_its_shape(self, tmp_path: Path):
        """An RGB frame has no DataFrame equivalent — it must survive unflattened."""
        array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        fp = tmp_path / "rgb.npy"
        np.save(fp, array, allow_pickle=False)

        result = _run_reader_in_sandbox(str(fp), tmp_path)

        assert result["shape"] == [2, 3, 3]
        np.testing.assert_array_equal(np.array(result["values"], dtype=np.uint8), array)

    def test_summary_line_reports_shape_and_dtype(self, tmp_path: Path):
        """The shared summary print must not assume DataFrame columns."""
        fp = tmp_path / "summary.npy"
        np.save(fp, np.ones((2, 5), dtype=np.int32), allow_pickle=False)

        result = _run_reader_in_sandbox(str(fp), tmp_path)

        assert "data_source loaded: ndarray with shape (2, 5), dtype: int32" in result["stdout"]

    def test_missing_file_raises_before_load(self, tmp_path: Path):
        """The existing existence guard still fires for a .npy path."""
        from osprey.mcp_server.workspace.tools._viz_common import build_data_reader

        code = build_data_reader(str(tmp_path / "absent.npy"))
        namespace: dict = {}

        with pytest.raises(FileNotFoundError):
            exec(code, namespace)  # noqa: S102


class TestNpyReaderSandboxSafety:
    """The emitted .npy loader must clear the viz sandbox import whitelist."""

    def test_generated_code_has_no_disallowed_imports(self, tmp_path: Path):
        from osprey.mcp_server.workspace.execution.sandbox_executor import validate_sandbox_code
        from osprey.mcp_server.workspace.tools._viz_common import build_data_reader

        is_safe, violations = validate_sandbox_code(build_data_reader(str(tmp_path / "f.npy")))

        assert is_safe, violations


class TestExistingBranchesUnchanged:
    """The new branch must not shadow the formats that already worked."""

    def test_csv_still_yields_a_dataframe_with_columns(self, tmp_path: Path):
        import pandas as pd

        from osprey.mcp_server.workspace.tools._viz_common import build_data_reader

        fp = tmp_path / "data.csv"
        fp.write_text("a,b\n1,2\n3,4\n")

        namespace: dict = {"pd": pd}
        exec(build_data_reader(str(fp)), namespace)  # noqa: S102

        data = namespace["data"]
        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["a", "b"]
