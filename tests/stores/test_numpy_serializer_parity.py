"""The numpy branch in the two object serializers, and their parity.

Osprey serializes "save this object" twice, by design: once in-process
(``ArtifactStore._serialize_object``) and once as *source text* injected into
the python-executor subprocess (``ExecutionWrapper._get_save_artifact_injection``).
The two are duplicated on purpose and drift silently, so the numpy branch is
pinned on both sides here, plus a behavioural parity check — the same array
must reach the same ``.npy`` decision whichever path saved it.

Parity is asserted behaviourally (run both, compare the outcome), never by
matching source strings, because a string match would pass on a branch that is
present but broken.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey.stores.artifact_store import ArtifactStore, _serialize_object

pytestmark = pytest.mark.unit

ARRAY = np.arange(12, dtype=np.uint16).reshape(3, 4)

NPY_MIME = "application/octet-stream"
NPY_TYPE = "file"


@pytest.fixture(autouse=True)
def _no_server_launch(monkeypatch):
    """Saving an artifact tries to boot the gallery web server; don't."""
    import osprey.infrastructure.server_launcher as launcher

    monkeypatch.setattr(launcher, "ensure_artifact_server", lambda: None)


@pytest.fixture
def store(tmp_path, monkeypatch) -> ArtifactStore:
    """An ArtifactStore with a real repo root, so path anchors resolve."""
    repo_root = tmp_path / "repo"
    (repo_root / "build").mkdir(parents=True)
    config_path = repo_root / "build" / "config.yml"
    config_path.write_text(
        f"project_root: {repo_root}\nagent_data:\n  base_dir: state/agent\n",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.chdir(repo_root)
    return ArtifactStore(workspace_root=repo_root / "state" / "agent")


def _run_generated_save_artifact(exec_dir: Path, obj, **kwargs) -> dict:
    """Execute the *generated* ``save_artifact`` source and return its manifest entry.

    The wrapper emits source text that only ever runs inside the executor
    subprocess, so the only honest way to test it is to exec it — the same
    technique the p4p monkeypatch tests use.
    """
    exec_dir.mkdir(parents=True, exist_ok=True)
    source = ExecutionWrapper()._get_save_artifact_injection()
    namespace: dict = {"_execution_dir": exec_dir}
    exec(compile(source, "<generated-save-artifact>", "exec"), namespace)

    with contextlib.redirect_stdout(io.StringIO()):
        namespace["save_artifact"](obj, title=kwargs.pop("title", "Array"), **kwargs)

    manifest = json.loads((exec_dir / "artifacts" / "manifest.json").read_text())
    return manifest[-1]


# ---------------------------------------------------------------------------
# In-process serializer
# ---------------------------------------------------------------------------


class TestInProcessSerializer:
    def test_ndarray_saves_as_npy(self, store):
        entry = store.save_object(ARRAY, title="Camera frame")

        assert entry.filename.endswith(".npy")
        assert entry.mime_type == NPY_MIME
        assert entry.artifact_type == NPY_TYPE

    def test_saved_npy_round_trips(self, store):
        entry = store.save_object(ARRAY, title="Camera frame")

        loaded = np.load(store.get_file_path(entry.id), allow_pickle=False)

        np.testing.assert_array_equal(loaded, ARRAY)
        assert loaded.dtype == ARRAY.dtype

    def test_ndarray_never_reaches_the_repr_fallback(self):
        _content, artifact_type, filename, mime = _serialize_object(ARRAY, "frame")

        assert (artifact_type, Path(filename).suffix, mime) != ("text", ".txt", "text/plain")

    def test_one_dimensional_and_zero_dimensional_arrays_too(self, store):
        for array in (np.linspace(0.0, 1.0, 5), np.array(3.5)):
            entry = store.save_object(array, title="Reading")

            assert entry.filename.endswith(".npy")
            np.testing.assert_array_equal(
                np.load(store.get_file_path(entry.id), allow_pickle=False), array
            )

    def test_object_dtype_still_falls_through(self):
        """``np.save`` cannot write object dtypes without pickle, which stays off."""
        obj_array = np.array([{"a": 1}, None], dtype=object)

        _content, artifact_type, filename, mime = _serialize_object(obj_array, "weird")

        assert (artifact_type, Path(filename).suffix, mime) == ("text", ".txt", "text/plain")

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            ("# hello", ("markdown", ".md", "text/markdown")),
            ("<div>hi</div>", ("html", ".html", "text/html")),
            ({"a": 1}, ("json", ".json", "application/json")),
            ([1, 2, 3], ("json", ".json", "application/json")),
            (b"\x00\x01", ("binary", ".bin", "application/octet-stream")),
            (object(), ("text", ".txt", "text/plain")),
        ],
    )
    def test_existing_types_are_unchanged(self, obj, expected):
        """Scope guard: the numpy branch must not perturb any existing sniffing."""
        _content, artifact_type, filename, mime = _serialize_object(obj, "thing")

        assert (artifact_type, Path(filename).suffix, mime) == expected


# ---------------------------------------------------------------------------
# Generated (subprocess) serializer
# ---------------------------------------------------------------------------


class TestGeneratedSaveArtifact:
    def test_ndarray_saves_as_npy(self, tmp_path):
        entry = _run_generated_save_artifact(tmp_path, ARRAY)

        assert entry["filename"].endswith(".npy")
        assert entry["mime_type"] == NPY_MIME
        assert entry["artifact_type"] == NPY_TYPE

    def test_saved_npy_round_trips(self, tmp_path):
        entry = _run_generated_save_artifact(tmp_path, ARRAY)

        loaded = np.load(tmp_path / "artifacts" / entry["filename"], allow_pickle=False)

        np.testing.assert_array_equal(loaded, ARRAY)
        assert loaded.dtype == ARRAY.dtype
        assert entry["size_bytes"] == (tmp_path / "artifacts" / entry["filename"]).stat().st_size

    def test_object_dtype_still_falls_through(self, tmp_path):
        entry = _run_generated_save_artifact(tmp_path, np.array([{"a": 1}, None], dtype=object))

        assert entry["filename"].endswith(".txt")
        assert entry["artifact_type"] == "text"

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            ("# hello", ("markdown", ".md", "text/markdown")),
            ("<div>hi</div>", ("html", ".html", "text/html")),
            ({"a": 1}, ("json", ".json", "application/json")),
            ([1, 2, 3], ("json", ".json", "application/json")),
            (b"\x00\x01", ("binary", ".bin", "application/octet-stream")),
        ],
    )
    def test_existing_types_are_unchanged(self, tmp_path, obj, expected):
        entry = _run_generated_save_artifact(tmp_path, obj)

        assert (
            entry["artifact_type"],
            Path(entry["filename"]).suffix,
            entry["mime_type"],
        ) == expected


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


class TestSerializerParity:
    def test_both_serializers_make_the_same_npy_decision(self, store, tmp_path):
        in_process = store.save_object(ARRAY, title="Frame")
        generated = _run_generated_save_artifact(tmp_path / "exec", ARRAY, title="Frame")

        decision = (NPY_TYPE, ".npy", NPY_MIME)
        assert (
            in_process.artifact_type,
            Path(in_process.filename).suffix,
            in_process.mime_type,
        ) == decision
        assert (
            generated["artifact_type"],
            Path(generated["filename"]).suffix,
            generated["mime_type"],
        ) == decision

    def test_both_serializers_write_byte_identical_payloads(self, store, tmp_path):
        in_process = store.save_object(ARRAY, title="Frame")
        generated = _run_generated_save_artifact(tmp_path / "exec", ARRAY, title="Frame")

        assert (
            store.get_file_path(in_process.id).read_bytes()
            == (tmp_path / "exec" / "artifacts" / generated["filename"]).read_bytes()
        )
