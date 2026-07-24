"""Tests for the MATLAB Middle Layer -> Channel Finder JSON converter.

`MMLConverter` accumulates per-system MML dicts and writes them to JSON. The
behaviours worth pinning are the structure validation, the channel counting
(which skips whitespace padding and the ``setup``/``pyat`` metadata subtrees,
and recurses into nested fields), and the round-tripping through disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.channel_finder.utils.mml_converter import MMLConverter


def _sr_data() -> dict:
    """A small SR system: one flat family and one nested (subfield) family."""
    return {
        "BPM": {
            "Monitor": {
                # Trailing padding names should not be counted.
                "ChannelNames": ["SR01:BPM:X", "SR01:BPM:Y", "   "],
            },
        },
        "HCM": {
            "Setpoint": {
                # Nested field -> counted via _count_nested_channels recursion.
                "X": {"ChannelNames": ["SR01:HCM:XSet"]},
            },
            # setup subtree is metadata, must be skipped by the nested counter.
            "setup": {"DeviceList": [[1, 1]]},
        },
    }


class TestAddSystem:
    def test_rejects_non_dict(self):
        conv = MMLConverter()
        with pytest.raises(ValueError, match="must be a dictionary"):
            conv.add_system("SR", ["not", "a", "dict"])  # type: ignore[arg-type]

    def test_stores_data_verbatim(self):
        conv = MMLConverter()
        data = _sr_data()
        conv.add_system("SR", data)
        assert conv.data["SR"] is data


class TestChannelCounting:
    def test_counts_flat_and_nested_skipping_padding(self):
        conv = MMLConverter()
        # SR01:BPM:X, SR01:BPM:Y (padding "   " dropped) + SR01:HCM:XSet == 3
        assert conv._count_channels(_sr_data()) == 3

    def test_nested_counter_skips_setup_and_pyat(self):
        conv = MMLConverter()
        family = {
            "setup": {"ChannelNames": ["should:not:count"]},
            "pyAT": {"ChannelNames": ["also:not"]},
            "Field": {"Sub": {"ChannelNames": ["counts:1", "counts:2"]}},
        }
        assert conv._count_nested_channels(family) == 2

    def test_ignores_non_dict_family_values(self):
        conv = MMLConverter()
        # A scalar metadata value at family level must not blow up counting.
        assert conv._count_channels({"_meta": "SR ring", "BPM": _sr_data()["BPM"]}) == 2


class TestSaveJson:
    def test_creates_parent_dirs_and_round_trips(self, tmp_path: Path):
        conv = MMLConverter()
        conv.add_system("SR", _sr_data())
        out = tmp_path / "nested" / "dir" / "sr.json"
        conv.save_json(str(out))

        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["SR"]["BPM"]["Monitor"]["ChannelNames"][0] == "SR01:BPM:X"

    def test_indent_argument_is_applied(self, tmp_path: Path):
        conv = MMLConverter()
        conv.add_system("SR", {"BPM": {"Monitor": {"ChannelNames": ["a"]}}})
        out = tmp_path / "sr.json"
        conv.save_json(str(out), indent=4)
        text = out.read_text()
        # 4-space indentation implies a line beginning with exactly four spaces.
        assert "\n    " in text


class TestConvertAndSave:
    def test_multi_system_convert_and_save(self, tmp_path: Path):
        out = tmp_path / "facility.json"
        MMLConverter.convert_and_save(
            {
                "SR": {"BPM": {"Monitor": {"ChannelNames": ["SR:BPM:1"]}}},
                "BR": {"BPM": {"Monitor": {"ChannelNames": ["BR:BPM:1"]}}},
            },
            str(out),
        )
        loaded = json.loads(out.read_text())
        assert set(loaded) == {"SR", "BR"}
        assert loaded["BR"]["BPM"]["Monitor"]["ChannelNames"] == ["BR:BPM:1"]


class TestConvertFromPythonFile:
    def test_loads_variable_and_derives_system_name(self, tmp_path: Path):
        # Write a Python module exposing an MML variable, mirroring real usage.
        module = tmp_path / "MML_ao_250413_SR.py"
        module.write_text("MML_ao_SR = {'BPM': {'Monitor': {'ChannelNames': ['SR:BPM:1']}}}\n")
        out = tmp_path / "sr.json"

        MMLConverter.convert_from_python_file(str(module), "MML_ao_SR", str(out))

        loaded = json.loads(out.read_text())
        # System name is the trailing token of the variable name ("SR").
        assert list(loaded) == ["SR"]
        assert loaded["SR"]["BPM"]["Monitor"]["ChannelNames"] == ["SR:BPM:1"]
