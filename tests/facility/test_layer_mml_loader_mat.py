"""Tests for the ``.mat`` MML loader.

The ``.mat`` files are written in each test with ``scipy.io.savemat`` (v5), so
they carry exactly the MATLAB shapes under test: cells mixing arrays and
structs, empty chars, and function-handle-shaped structs. Every decoded body
must survive a strict JSON dump, which is the contract of the canonical writer.

The lattice lane is the exception: a deck is a whole ring, so those tests read
the committed synthetic export rather than writing a ring by hand, and the
numbers they hold the loaded ring to come from that export's own fingerprint.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pytest
from scipy.io import savemat
from scipy.io.matlab import MatlabFunction, mat_struct

from osprey.facility.layers.mml.loaders import LoadedInput
from osprey.facility.layers.mml.loaders.mat import decode, load_lattice, load_mat
from osprey.facility.layers.mml.normalize import normalize_family
from tests.templates.mml_export_contract import VA_LATTICE_KEYS

SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "mml" / "synthetic"
SYNTHETIC_LATTICE = SYNTHETIC / "quokka.sr.lattice.mat"
MISMATCHED_LATTICE = SYNTHETIC / "mismatched.lattice.mat"


def _cell(*items: object) -> np.ndarray:
    """A MATLAB cell row holding ``items``."""
    cell = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        cell[index] = item
    return cell


def _lattice_facts() -> dict:
    """The exported fingerprint facts the loaded ring is held to."""
    va = json.loads((SYNTHETIC / "quokka.sr.va.json").read_text(encoding="utf-8"))
    return {key: va["lattice"][key] for key in VA_LATTICE_KEYS}


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


class TestLatticeInput:
    """A ``.mat`` whose top-level variable is ``THERING`` is a lattice, not an export."""

    def test_lattice_mat_carries_its_path_and_no_families(self):
        """The loader recognises the deck and leaves AO and AD empty."""
        loaded = load_mat(SYNTHETIC_LATTICE)
        assert loaded.lattice == SYNTHETIC_LATTICE
        assert loaded.ao == {}
        assert loaded.ad is None
        assert loaded.export is None
        assert loaded.system_keyed is False
        assert loaded.source == SYNTHETIC_LATTICE

    def test_an_ao_export_is_not_a_lattice(self, mat_path: Path):
        """Every other ``.mat`` stays an AO/AD export."""
        assert load_mat(mat_path).lattice is None

    def test_ao_beside_thering_is_refused(self, tmp_path: Path):
        """One file cannot be both, so neither half is dropped in silence."""
        path = tmp_path / "both.mat"
        ring = _cell({"FamName": "DR", "Length": 1.0, "PassMethod": "DriftPass"})
        savemat(str(path), {"AO": {"HCM": {"FamilyName": "HCM"}}, "THERING": ring})
        with pytest.raises(click.ClickException, match="both.mat"):
            load_mat(path)

    def test_loading_the_lattice_keeps_the_ringparam_and_matlab_indices(self):
        """Every index the export's numbers were taken at survives the load."""
        facts = _lattice_facts()
        ring = load_lattice(SYNTHETIC_LATTICE)
        assert len(ring) == facts["elements"]
        assert ring[facts["ringparam_indices"] - 1].tag == "RingParam"
        assert ring.energy == pytest.approx(facts["energy_gev"] * 1e9)

    def test_loading_the_mismatched_lattice_gives_the_renamed_family(self):
        """The counter-example deck loads; only its family names differ."""
        names = [element.FamName for element in load_lattice(MISMATCHED_LATTICE)]
        assert len(names) == _lattice_facts()["elements"]
        assert "QFX" in names
        assert "QF1" not in names

    def test_a_line_whose_bends_nearly_cancel_loads_at_periodicity_one(self, tmp_path: Path):
        """Bend angles summing to a rounding residue never set the periodicity."""
        path = tmp_path / "line.mat"
        angles = (0.1, 0.2, -0.3)
        assert sum(angles) != 0.0
        bends = [
            {
                "FamName": f"B{index}",
                "Class": "Bend",
                "Length": 1.0,
                "BendingAngle": angle,
                "Energy": 1.9e9,
                "PassMethod": "BndMPoleSymplectic4Pass",
            }
            for index, angle in enumerate(angles)
        ]
        savemat(str(path), {"THERING": _cell(*bends)})
        assert load_lattice(path).periodicity == 1

    def test_a_deck_that_is_not_a_ring_is_refused_by_name(self, tmp_path: Path):
        """A ``THERING`` pyAT cannot build raises a message naming the file."""
        path = tmp_path / "broken.mat"
        savemat(str(path), {"THERING": _cell({"FamName": "DR", "PassMethod": "DriftPass"})})
        with pytest.raises(click.ClickException, match="broken.mat"):
            load_lattice(path)
