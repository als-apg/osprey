"""Tests for the ``.mat`` MML loader.

The ``.mat`` files are written in each test with ``scipy.io.savemat`` (v5), so
they carry exactly the MATLAB shapes under test: cells mixing arrays and
structs, empty chars, and function-handle-shaped structs. Every decoded body
must survive a strict JSON dump, which is the contract of the canonical writer.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pytest
from scipy.io import savemat
from scipy.io.matlab import MatlabFunction, mat_struct

from osprey.services.mml.loaders import LoadedInput
from osprey.services.mml.loaders.mat import decode, load_mat
from osprey.services.mml.normalize import normalize_family


def _cell(*items: object) -> np.ndarray:
    """A MATLAB cell row holding ``items``."""
    cell = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        cell[index] = item
    return cell


def _strict_dump(value: object) -> str:
    """Dump the way the canonical writer does: no NaN, no non-JSON types."""
    return json.dumps(value, allow_nan=False, sort_keys=True)


def _ao() -> dict:
    """An AO struct with the shapes the decoder has to handle."""
    return {
        "BPM": {
            "FamilyName": "BPM",
            "Cells": _cell(np.array([1.0, 2.0, 3.0]), {"Units": "mm", "Gain": 2.5}),
            "Mixed": _cell(np.array([[1, 2], [3, 4]]), "text", np.array([], dtype=float)),
            "Empty": "",
            "Rows": np.array(["SR01  ", "SR02  "]),
            "DeviceList": np.array([[1, 1], [1, 2]]),
            "Monitor": {
                "ChannelNames": np.array(["SR01:BPM:X", "SR02:BPM:X"]),
                "HW2PhysicsFcn": {
                    "matlabroot": "/opt/matlab",
                    "separator": "/",
                    "sentinel": np.array([0], dtype=np.uint8),
                    "function_handle": {"function": "bpm2phys", "file": ""},
                },
            },
        }
    }


@pytest.fixture
def mat_path(tmp_path: Path) -> Path:
    """A v5 ``.mat`` carrying AO and AD."""
    path = tmp_path / "ring.mat"
    savemat(str(path), {"AO": _ao(), "AD": {"Machine": "QUOKKA", "Energy": 1.9}})
    return path


class TestLoadMat:
    """Reading a whole ``.mat`` export."""

    def test_returns_loaded_input_with_ao_and_ad(self, mat_path: Path):
        """AO and AD variables land in ``ao`` and ``ad``; the rest is fixed."""
        loaded = load_mat(mat_path)
        assert isinstance(loaded, LoadedInput)
        assert set(loaded.ao) == {"BPM"}
        assert loaded.ad == {"Machine": "QUOKKA", "Energy": 1.9}
        assert loaded.export is None
        assert loaded.system_keyed is False
        assert loaded.source == mat_path

    def test_decoded_body_passes_a_strict_json_dump(self, mat_path: Path):
        """Nothing numpy- or scipy-typed survives decoding."""
        loaded = load_mat(mat_path)
        _strict_dump(loaded.ao)
        _strict_dump(loaded.ad)

    def test_cell_with_array_and_struct(self, mat_path: Path):
        """A cell holding a numeric array and a struct decodes element by element."""
        body = load_mat(mat_path).ao["BPM"]
        assert body["Cells"] == [[1.0, 2.0, 3.0], {"Units": "mm", "Gain": 2.5}]
        assert body["Mixed"] == [[[1, 2], [3, 4]], "text", []]

    def test_empty_char_is_empty_string(self, mat_path: Path):
        """An empty char decodes as ``""``."""
        assert load_mat(mat_path).ao["BPM"]["Empty"] == ""

    def test_char_matrix_rows_are_deblanked(self, mat_path: Path):
        """A padded char matrix becomes one trailing-blank-free string per row."""
        assert load_mat(mat_path).ao["BPM"]["Rows"] == ["SR01", "SR02"]

    def test_numeric_matrix_is_nested_list(self, mat_path: Path):
        """A numeric matrix decodes via ``tolist``."""
        assert load_mat(mat_path).ao["BPM"]["DeviceList"] == [[1, 1], [1, 2]]

    def test_function_handle_struct_folds_after_normalisation(self, mat_path: Path):
        """The handle-shaped struct is raw here and folds in the normaliser."""
        body = load_mat(mat_path).ao["BPM"]
        handle = body["Monitor"]["HW2PhysicsFcn"]
        assert handle["function_handle"] == {"function": "bpm2phys", "file": ""}
        normalized = normalize_family(body)
        assert normalized["Monitor"]["HW2PhysicsFcn"] == {"$fn": "bpm2phys", "file": None}

    def test_lowercase_variable_names(self, tmp_path: Path):
        """``ao`` and ``ad`` are accepted as well as ``AO`` and ``AD``."""
        path = tmp_path / "lower.mat"
        savemat(str(path), {"ao": {"HCM": {"FamilyName": "HCM"}}, "ad": {"Machine": "X"}})
        loaded = load_mat(path)
        assert loaded.ao == {"HCM": {"FamilyName": "HCM"}}
        assert loaded.ad == {"Machine": "X"}

    def test_absent_ad_is_none(self, tmp_path: Path):
        """A file without an AD variable yields ``ad=None``."""
        path = tmp_path / "noad.mat"
        savemat(str(path), {"AO": {"HCM": {"FamilyName": "HCM"}}})
        assert load_mat(path).ad is None

    def test_missing_ao_is_refused(self, tmp_path: Path):
        """A file without AO raises a ClickException naming the file."""
        path = tmp_path / "noao.mat"
        savemat(str(path), {"other": 1})
        with pytest.raises(click.ClickException, match="noao.mat"):
            load_mat(path)


class TestV73Refusal:
    """HDF5-based MAT-files are refused before scipy touches them."""

    @pytest.mark.parametrize("version_and_endian", [b"\x00\x02IM", b"\x02\x00MI"])
    def test_v73_header_is_refused_with_save_hint(self, tmp_path: Path, version_and_endian: bytes):
        """The refusal names the file and the ``save('-v7', ...)`` fix."""
        path = tmp_path / "hdf5.mat"
        text = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Mon Jan  1 00:00:00 2026 HDF5"
        header = text.ljust(116, b" ") + b"\x00" * 8 + version_and_endian
        path.write_bytes(header + b"\x89HDF\r\n\x1a\n" + b"\x00" * 256)
        with pytest.raises(click.ClickException) as excinfo:
            load_mat(path)
        message = excinfo.value.message
        assert "save('-v7', ...)" in message
        assert "hdf5.mat" in message


class TestDecode:
    """Individual decoder arms."""

    def test_matlab_function_wrapping_struct_decodes_to_dict(self):
        """A 0-d MatlabFunction holding a struct decodes to a dict, not a list."""
        inner = mat_struct()
        inner._fieldnames = ["matlabroot", "separator", "sentinel", "function_handle"]
        inner.matlabroot = "/opt/matlab"
        inner.separator = "/"
        inner.sentinel = np.array([], dtype=np.uint8)
        handle = mat_struct()
        handle._fieldnames = ["function", "file"]
        handle.function = "amp2k"
        handle.file = ""
        inner.function_handle = handle
        wrapper = MatlabFunction(np.empty((), dtype=object))
        wrapper[()] = inner

        decoded = decode(wrapper)

        assert isinstance(decoded, dict)
        assert decoded["function_handle"] == {"function": "amp2k", "file": ""}
        assert decoded["sentinel"] == []
        _strict_dump(decoded)

    def test_numpy_scalars_become_python_scalars(self):
        """numpy scalars are converted so the dump and bool rules see Python types."""
        assert type(decode(np.int64(3))) is int
        assert type(decode(np.float32(1.5))) is float
        assert type(decode(np.bool_(True))) is bool
        assert type(decode(np.str_("x"))) is str

    def test_object_matrix_keeps_its_shape(self):
        """A 2-D cell decodes to nested lists of its shape."""
        cell = np.empty((2, 2), dtype=object)
        cell[0, 0], cell[0, 1] = "a", np.int64(1)
        cell[1, 0], cell[1, 1] = np.array([1.0]), np.array("b")
        assert decode(cell) == [["a", 1], [[1.0], "b"]]

    def test_zero_size_numeric_array_is_empty_list(self):
        """Any zero-size numeric array decodes as ``[]``."""
        assert decode(np.zeros((0, 3))) == []
