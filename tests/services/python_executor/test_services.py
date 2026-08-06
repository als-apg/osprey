"""Unit tests for :mod:`osprey.services.python_executor.services`.

``services.py`` holds the ``make_json_serializable`` /
``serialize_results_to_file`` result-serialisation helpers that execution
wrappers depend on.

The helpers are load-bearing: execution results contain arbitrary scientific
objects (numpy arrays, matplotlib figures, ``Path``s, sets, complex numbers)
that must degrade to a JSON-safe form without ever raising, and must fall back
gracefully when serialisation genuinely fails. Those contracts — including the
never-raise fallbacks — are pinned here. All file I/O is confined to
``tmp_path``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import osprey.services.python_executor.services as services_mod
from osprey.services.python_executor.services import (
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
