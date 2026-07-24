"""Unit tests for :mod:`osprey.services.python_executor.services`.

Despite the module name, ``services.py`` is the file/notebook plumbing for the
Python executor: ``FileManager`` (execution-folder layout + Jupyter URL
generation), ``NotebookManager`` (attempt/final ``.ipynb`` generation), and the
``make_json_serializable`` / ``serialize_results_to_file`` result-serialisation
helpers that execution wrappers depend on.

The serialisation helpers are the load-bearing part: execution results contain
arbitrary scientific objects (numpy arrays, matplotlib figures, ``Path``s, sets,
complex numbers) that must degrade to a JSON-safe form without ever raising, and
must fall back gracefully when serialisation genuinely fails. Those contracts —
including the never-raise fallbacks — are pinned here. All file I/O is confined
to ``tmp_path``; nothing touches a container, a network, or a real Jupyter host.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import nbformat
import pytest

import osprey.services.python_executor.services as services_mod
from osprey.services.python_executor.models import PythonExecutionContext
from osprey.services.python_executor.services import (
    FileManager,
    NotebookManager,
    _is_matplotlib_figure,
    _serialize_matplotlib_figure,
    make_json_serializable,
    serialize_results_to_file,
    serialize_results_to_file_async,
)

# ---------------------------------------------------------------------------
# Test doubles for scientific objects (no numpy/matplotlib import required)
# ---------------------------------------------------------------------------


class _ArrayLike:
    """Stands in for a numpy array: has ``tolist`` (and ``item``, checked later)."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return list(self._data)

    def item(self):  # pragma: no cover - tolist path wins for multi-element
        return self._data[0]


class _ScalarLike:
    """Stands in for a numpy scalar: only ``item``, no ``tolist``."""

    def item(self):
        return 7


class _FrameLike:
    """Stands in for a pandas object: ``to_dict`` + ``index``."""

    def __init__(self, mapping):
        self._mapping = mapping
        self.index = list(mapping)

    def to_dict(self):
        return dict(self._mapping)


class _FakeFigure:
    """Minimal matplotlib ``Figure`` stand-in (class name must be ``Figure``)."""

    __name__ = "Figure"

    def __init__(self, axes=None):
        self._axes = axes or []

    def savefig(self, *a, **k):  # pragma: no cover - presence-only for detection
        pass

    def get_axes(self):
        return self._axes

    def get_size_inches(self):
        return _ArrayLike([6.4, 4.8])

    def get_dpi(self):
        return 100


# Rename the class so ``type(obj).__name__ == "Figure"`` as the detector requires.
_FakeFigure.__name__ = "Figure"


class TestFileManagerInit:
    def test_base_dir_uses_agent_data_dir(self, tmp_path):
        fm = FileManager({"agent_data_dir": str(tmp_path)})
        assert fm.base_dir == (tmp_path / "executed_scripts").resolve()

    def test_base_dir_defaults_when_missing(self):
        fm = FileManager({})
        # Falls back to the "_agent_data" relative default (resolved to abs).
        assert fm.base_dir.name == "executed_scripts"
        assert fm.base_dir.parent.name == "_agent_data"


class TestCreateExecutionFolder:
    def test_creates_folder_structure(self, tmp_path):
        fm = FileManager({"agent_data_dir": str(tmp_path)})
        ctx = fm.create_execution_folder("sensor_analysis")

        assert ctx.is_initialized
        assert ctx.folder_path.is_dir()
        assert ctx.attempts_folder.is_dir()
        assert ctx.attempts_folder.parent == ctx.folder_path
        # Folder name embeds the caller-supplied label and the fixed prefix.
        assert ctx.folder_path.name.startswith("execution_")
        assert "sensor_analysis" in ctx.folder_path.name
        # Year-month bucket sits directly under the base dir.
        assert ctx.folder_path.parent.parent == fm.base_dir

    def test_distinct_calls_produce_distinct_folders(self, tmp_path):
        fm = FileManager({"agent_data_dir": str(tmp_path)})
        a = fm.create_execution_folder()
        b = fm.create_execution_folder()
        assert a.folder_path != b.folder_path

    def test_permission_failure_does_not_abort_creation(self, tmp_path, monkeypatch):
        fm = FileManager({"agent_data_dir": str(tmp_path)})

        def boom(*a, **k):
            raise OSError("cannot chmod")

        # chmod failures are logged and swallowed; the folder must still exist.
        monkeypatch.setattr(services_mod.os, "chmod", boom)
        ctx = fm.create_execution_folder()
        assert ctx.folder_path.is_dir()


class TestCreateJupyterUrl:
    def test_url_under_base_dir_uses_default_port(self, tmp_path):
        fm = FileManager({"agent_data_dir": str(tmp_path)})
        target = fm.base_dir / "2026-07" / "execution_x"
        url = fm._create_jupyter_url(target)
        assert url.startswith("http://localhost:8088/lab/tree/executed_scripts/")
        assert url.endswith("2026-07/execution_x")

    def test_url_honours_configured_read_container_port(self, tmp_path):
        fm = FileManager(
            {
                "agent_data_dir": str(tmp_path),
                "service_configs": {"jupyter": {"containers": {"read": {"port_host": 9999}}}},
            }
        )
        url = fm._create_jupyter_url(fm.base_dir / "nb")
        assert "http://localhost:9999/" in url

    def test_path_outside_base_dir_returns_file_uri(self, tmp_path):
        fm = FileManager({"agent_data_dir": str(tmp_path)})
        outside = Path("/some/absolute/elsewhere")
        assert fm._create_jupyter_url(outside) == outside.as_uri()


class TestSaveResults:
    def test_writes_results_json(self, tmp_path):
        fm = FileManager({"agent_data_dir": str(tmp_path)})
        out = fm.save_results({"mean": 42}, tmp_path)
        assert out == tmp_path / "results.json"
        assert json.loads(out.read_text()) == {"mean": 42}

    def test_raises_runtimeerror_on_serialisation_failure(self, tmp_path, monkeypatch):
        fm = FileManager({"agent_data_dir": str(tmp_path)})

        def failed(results, file_path):
            return {"success": False, "error": "disk full"}

        monkeypatch.setattr(services_mod, "serialize_results_to_file", failed)
        with pytest.raises(RuntimeError, match="disk full"):
            fm.save_results({"x": 1}, tmp_path)


class TestMakeJsonSerializable:
    def test_datetime_becomes_isoformat(self):
        dt = datetime(2026, 7, 21, 12, 30, 0)
        assert make_json_serializable(dt) == dt.isoformat()

    def test_array_like_becomes_list(self):
        assert make_json_serializable(_ArrayLike([1, 2, 3])) == [1, 2, 3]

    def test_scalar_like_becomes_item(self):
        assert make_json_serializable(_ScalarLike()) == 7

    def test_path_becomes_string(self):
        assert make_json_serializable(Path("/tmp/x")) == "/tmp/x"

    def test_set_is_wrapped_with_type_tag(self):
        result = make_json_serializable({1, 2, 3})
        assert result["_type"] == "set"
        assert set(result["items"]) == {1, 2, 3}

    def test_complex_is_decomposed(self):
        result = make_json_serializable(complex(1, -2))
        assert result == {"real": 1.0, "imag": -2.0, "_type": "complex"}

    def test_frame_like_uses_to_dict(self):
        result = make_json_serializable(_FrameLike({"a": 1, "b": 2}))
        assert result == {"a": 1, "b": 2}

    def test_unknown_object_falls_back_to_string_record(self):
        class Weird:
            def __repr__(self):
                return "<weird>"

        result = make_json_serializable(Weird())
        assert result["type"] == "Weird"
        assert result["_serialization_note"] == "converted_to_string"
        assert "weird" in result["value"]

    def test_circular_reference_returns_failure_record_without_raising(self):
        d: dict = {}
        d["self"] = d
        result = make_json_serializable(d)
        assert result["_serialization_failed"] is True
        assert result["_original_type"] == "dict"
        assert "_error" in result

    def test_nested_structure_is_recursively_serialised(self):
        payload = {"arr": _ArrayLike([1, 2]), "when": datetime(2026, 1, 1), "p": Path("/a")}
        result = make_json_serializable(payload)
        assert result["arr"] == [1, 2]
        assert result["when"] == "2026-01-01T00:00:00"
        assert result["p"] == "/a"


class TestMatplotlibHelpers:
    def test_is_matplotlib_figure_true_for_figure_like(self):
        assert _is_matplotlib_figure(_FakeFigure()) is True

    def test_is_matplotlib_figure_false_for_plain_object(self):
        assert _is_matplotlib_figure(object()) is False

    def test_serialize_empty_figure(self):
        result = _serialize_matplotlib_figure(_FakeFigure(axes=[]))
        assert result["_type"] == "matplotlib_figure"
        assert result["axes"] == []
        assert result["figure_size"] == [6.4, 4.8]
        assert result["dpi"] == 100

    def test_serialize_figure_error_is_captured(self):
        class Broken:
            def get_axes(self):
                raise ValueError("no axes")

        result = _serialize_matplotlib_figure(Broken())
        assert result["_type"] == "matplotlib_figure"
        assert "Failed to serialize figure" in result["_error"]


class TestSerializeResultsToFile:
    def test_success_writes_file_and_reports_metadata(self, tmp_path):
        target = tmp_path / "results.json"
        meta = serialize_results_to_file({"k": "v"}, str(target))
        assert meta["success"] is True
        assert meta["file_path"] == str(target)
        assert json.loads(target.read_text()) == {"k": "v"}

    def test_serialisation_error_writes_fallback(self, tmp_path, monkeypatch):
        target = tmp_path / "results.json"

        def boom(_results):
            raise ValueError("cannot serialise")

        monkeypatch.setattr(services_mod, "make_json_serializable", boom)
        meta = serialize_results_to_file({"k": "v"}, str(target))
        assert meta["success"] is False
        assert meta["error"] == "cannot serialise"
        assert meta["error_type"] == "ValueError"
        # A minimal fallback record is still persisted.
        assert meta["fallback_saved"] is True
        fallback = json.loads(target.read_text())
        assert fallback["_serialization_failed"] is True

    def test_unwritable_path_records_error_and_fallback_error(self, tmp_path):
        # Directory does not exist -> both the primary and fallback writes fail,
        # but the function must return metadata rather than raise.
        target = tmp_path / "missing_dir" / "results.json"
        meta = serialize_results_to_file({"k": "v"}, str(target))
        assert meta["success"] is False
        assert meta["error"] is not None
        assert "fallback_error" in meta


class TestSerializeResultsToFileAsync:
    async def test_async_success_writes_file(self, tmp_path):
        target = tmp_path / "results.json"
        meta = await serialize_results_to_file_async({"k": 1}, str(target))
        assert meta["success"] is True
        assert json.loads(target.read_text()) == {"k": 1}

    async def test_async_serialisation_error_writes_fallback(self, tmp_path, monkeypatch):
        target = tmp_path / "results.json"

        def boom(_results):
            raise ValueError("nope")

        monkeypatch.setattr(services_mod, "make_json_serializable", boom)
        meta = await serialize_results_to_file_async({"k": 1}, str(target))
        assert meta["success"] is False
        assert meta["fallback_saved"] is True
        assert json.loads(target.read_text())["_serialization_failed"] is True


class TestNotebookManager:
    def _context(self, tmp_path) -> PythonExecutionContext:
        fm = FileManager({"agent_data_dir": str(tmp_path)})
        return fm.create_execution_folder("nb_test")

    def test_create_attempt_notebook_writes_and_tracks(self, tmp_path):
        nm = NotebookManager({"agent_data_dir": str(tmp_path)})
        ctx = self._context(tmp_path)

        path = nm.create_attempt_notebook(ctx, code="print(1)", stage="execution")
        assert path.exists()
        assert path.parent == ctx.attempts_folder
        # Attempt is recorded on the context with a sequential number.
        assert len(ctx.notebook_attempts) == 1
        assert ctx.notebook_attempts[0].attempt_number == 1
        # File is a valid notebook containing the code cell.
        nb = nbformat.read(path, as_version=4)
        assert any("print(1)" in c.source for c in nb.cells)

    def test_attempt_numbers_increment(self, tmp_path):
        nm = NotebookManager({"agent_data_dir": str(tmp_path)})
        ctx = self._context(tmp_path)
        p1 = nm.create_attempt_notebook(ctx, code="a=1", stage="execution")
        p2 = nm.create_attempt_notebook(ctx, code="b=2", stage="execution")
        assert p1 != p2
        assert [a.attempt_number for a in ctx.notebook_attempts] == [1, 2]

    def test_attempt_notebook_includes_error_context(self, tmp_path):
        nm = NotebookManager({"agent_data_dir": str(tmp_path)})
        ctx = self._context(tmp_path)
        path = nm.create_attempt_notebook(
            ctx, code="x", stage="execution", error_context="BOOM traceback"
        )
        nb = nbformat.read(path, as_version=4)
        assert any("BOOM traceback" in c.source for c in nb.cells)

    def test_create_final_notebook_without_results(self, tmp_path):
        nm = NotebookManager({"agent_data_dir": str(tmp_path)})
        ctx = self._context(tmp_path)
        path = nm.create_final_notebook(ctx, code="print('done')")
        assert path == ctx.folder_path / "notebook.ipynb"
        nb = nbformat.read(path, as_version=4)
        assert any("Successful Execution" in c.source for c in nb.cells)
        assert any("print('done')" in c.source for c in nb.cells)

    def test_create_final_notebook_with_results_adds_results_section(self, tmp_path):
        nm = NotebookManager({"agent_data_dir": str(tmp_path)})
        ctx = self._context(tmp_path)
        path = nm.create_final_notebook(ctx, code="x=1", results={"mean": 42})
        nb = nbformat.read(path, as_version=4)
        assert any("Execution Results" in c.source for c in nb.cells)
        assert any("results.json" in c.source for c in nb.cells)

    def test_create_final_notebook_with_error_context(self, tmp_path):
        nm = NotebookManager({"agent_data_dir": str(tmp_path)})
        ctx = self._context(tmp_path)
        path = nm.create_final_notebook(ctx, code="x=1", error_context="ValueError: bad")
        nb = nbformat.read(path, as_version=4)
        assert any("Failed Execution" in c.source for c in nb.cells)
        assert any("ValueError: bad" in c.source for c in nb.cells)
