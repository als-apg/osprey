"""Tests for the family-keyed JSON loader.

Each JSON form in the wild has a case here: flat, wrapped, system-keyed, and
the dialect spellings the shipped exporter and older exporters produce. The
excerpts are inline so each test shows the exact bytes it pins; a last group
reads the committed fixtures end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from osprey.services.mml.loaders import LoadedInput
from osprey.services.mml.loaders.json_any import load_json

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mml"

_BPM = {
    "FamilyName": "BPM",
    "DeviceList": [[1, 1], [1, 2]],
    "X": {"ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"]},
}


def _write(tmp_path: Path, body: object, name: str = "export.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(body))
    return path


def _write_text(tmp_path: Path, text: str, name: str = "export.json") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


class TestFlatForm:
    """A top-level map of families."""

    def test_flat_family_map(self, tmp_path):
        """A flat file is one family map and is not system-keyed."""
        path = _write(tmp_path, {"BPM": _BPM})
        loaded = load_json(path)
        assert isinstance(loaded, LoadedInput)
        assert loaded.ao == {"BPM": _BPM}
        assert loaded.system_keyed is False
        assert loaded.ad is None
        assert loaded.export is None
        assert loaded.source == path

    def test_family_by_tango_names_only(self, tmp_path):
        """A field carrying only TangoNames makes its parent a family."""
        body = {"MAG": {"Setpoint": {"TangoNames": ["sr/mag/1"]}}}
        loaded = load_json(_write(tmp_path, body))
        assert loaded.ao == body
        assert loaded.system_keyed is False

    def test_family_by_setup_device_list(self, tmp_path):
        """A DeviceList inside setup makes a family."""
        body = {"HCM": {"setup": {"DeviceList": [[1, 1]]}}}
        assert load_json(_write(tmp_path, body)).ao == body

    def test_accepts_str_path(self, tmp_path):
        """The path may be given as a string."""
        path = _write(tmp_path, {"BPM": _BPM})
        loaded = load_json(str(path))
        assert loaded.source == path


class TestWrappedForm:
    """A single ``ao``/``AO`` wrapper around the family map."""

    @pytest.mark.parametrize("key", ["ao", "AO"])
    def test_wrapper_is_unwrapped(self, tmp_path, key):
        """Either spelling of the wrapper key is removed."""
        loaded = load_json(_write(tmp_path, {key: {"BPM": _BPM}}))
        assert loaded.ao == {"BPM": _BPM}
        assert loaded.system_keyed is False

    def test_wrapper_holding_systems(self, tmp_path):
        """A wrapper around a system-keyed map is unwrapped and system-keyed."""
        loaded = load_json(_write(tmp_path, {"AO": {"SR": {"BPM": _BPM}}}))
        assert loaded.ao == {"SR": {"BPM": _BPM}}
        assert loaded.system_keyed is True

    def test_family_named_ao_is_not_a_wrapper(self, tmp_path):
        """A family that happens to be named AO stays a family."""
        body = {"AO": _BPM}
        loaded = load_json(_write(tmp_path, body))
        assert loaded.ao == body
        assert loaded.system_keyed is False

    def test_wrapped_ad_is_carried(self, tmp_path):
        """An ``ad``/``AD`` sibling of the wrapper becomes the AD."""
        body = {"ao": {"BPM": _BPM}, "AD": {"SubMachine": "SR"}}
        loaded = load_json(_write(tmp_path, body))
        assert loaded.ao == {"BPM": _BPM}
        assert loaded.ad == {"SubMachine": "SR"}


class TestSystemKeyedForm:
    """Top-level keys are systems, each holding families."""

    def test_system_keyed_sets_flag(self, tmp_path):
        """Every non-underscore top-level value being a family map sets system_keyed."""
        body = {
            "RING": {"_description": "ring", "BPM": _BPM},
            "BOOST": {"BPM": _BPM},
        }
        loaded = load_json(_write(tmp_path, body))
        assert loaded.system_keyed is True
        assert loaded.ao == body

    def test_system_description_is_kept(self, tmp_path):
        """A system's underscore metadata passes through untouched."""
        body = {"RING": {"_description": "Main ring", "BPM": _BPM}}
        loaded = load_json(_write(tmp_path, body))
        assert loaded.ao["RING"]["_description"] == "Main ring"


class TestDialect:
    """Spellings found in real exports."""

    def test_bare_non_finite_tokens_become_strings(self, tmp_path):
        """Bare NaN, Infinity and -Infinity become the normaliser's strings."""
        text = (
            '{"HCM": {"DeviceList": [[1, 1]], "Setpoint": {"ChannelNames": ["A"],'
            ' "Tolerance": NaN, "Range": [-Infinity, Infinity]}}}'
        )
        loaded = load_json(_write_text(tmp_path, text))
        setpoint = loaded.ao["HCM"]["Setpoint"]
        assert setpoint["Tolerance"] == "NaN"
        assert setpoint["Range"] == ["-Inf", "Inf"]

    def test_quoted_non_finite_strings_pass_through(self, tmp_path):
        """Quoted inf strings are left for the normaliser."""
        text = '{"QF": {"DeviceList": [[1, 1]], "Range": ["-inf", "inf"], "Position": ["NaN"]}}'
        loaded = load_json(_write_text(tmp_path, text))
        assert loaded.ao["QF"]["Range"] == ["-inf", "inf"]
        assert loaded.ao["QF"]["Position"] == ["NaN"]

    def test_typo_keys_pass_through(self, tmp_path):
        """A misspelled key and its function-handle record are not touched."""
        record = {"function_handle": {"function": "amp2k", "type": "simple", "file": ""}}
        body = {
            "QF": {
                "DeviceList": [[1, 1]],
                "Setpoint": {"ChannelNames": "RG:QF:SP", "HW2PhysicSDcn": record},
            }
        }
        loaded = load_json(_write(tmp_path, body))
        assert loaded.ao == body

    def test_cell_of_cells_passes_through(self, tmp_path):
        """Nested lists (cell arrays of cells) are kept as exported."""
        body = {"BEND": {"DeviceList": [[1, 1]], "Monitor": {"ChannelNames": [["A", "B"], ["C"]]}}}
        assert load_json(_write(tmp_path, body)).ao == body

    def test_unknown_top_level_keys_do_not_raise(self, tmp_path):
        """A stray non-family key beside real families is kept, not refused."""
        body = {"BPM": _BPM, "Mystery": 42, "Notes": {"text": "hi"}}
        loaded = load_json(_write(tmp_path, body))
        assert loaded.ao == body
        assert loaded.system_keyed is False

    def test_json_booleans_are_kept(self, tmp_path):
        """JSON booleans are decoded as bool; bool-to-int is the normaliser's job."""
        body = {"BPM": {**_BPM, "Status": [True, False]}}
        assert load_json(_write(tmp_path, body)).ao["BPM"]["Status"] == [True, False]


class TestExportBlockAndPairedAd:
    """The ``_export`` block and the ``<stem>.ad.json`` sibling."""

    def test_export_block_is_carried(self, tmp_path):
        """A top-level _export block becomes LoadedInput.export and leaves ao."""
        export = {"exporter": "mml_export 1.0.0", "submachine": "RING"}
        loaded = load_json(_write(tmp_path, {"_export": export, "BPM": _BPM}))
        assert loaded.export == export
        assert loaded.ao == {"BPM": _BPM}

    def test_sibling_ad_is_loaded(self, tmp_path):
        """<stem>.ad.json beside <stem>.ao.json is read as the AD."""
        ao_path = _write(tmp_path, {"BPM": _BPM}, name="m.ring.ao.json")
        _write_text(tmp_path, '{"SubMachine": "RING", "Energy": NaN}', name="m.ring.ad.json")
        loaded = load_json(ao_path)
        assert loaded.ad == {"SubMachine": "RING", "Energy": "NaN"}

    def test_no_sibling_ad(self, tmp_path):
        """Without a sibling AD file, ad is None."""
        loaded = load_json(_write(tmp_path, {"BPM": _BPM}, name="m.ring.ao.json"))
        assert loaded.ad is None

    def test_ad_not_paired_for_plain_json_name(self, tmp_path):
        """Only a .ao.json name pairs; export.json never picks up an AD."""
        _write(tmp_path, {"SubMachine": "RING"}, name="export.ad.json")
        loaded = load_json(_write(tmp_path, {"BPM": _BPM}, name="export.json"))
        assert loaded.ad is None


class TestRefusals:
    """The loader refuses only inputs with no family at all."""

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"_export": {"exporter": "x"}},
            {"Machine": "Quokka", "Energy": 2.4},
            {"ao": {"Machine": "Quokka"}},
            [1, 2, 3],
        ],
    )
    def test_no_family_raises_naming_file(self, tmp_path, body):
        """No family anywhere raises ClickException naming the file."""
        path = _write(tmp_path, body, name="nothing.json")
        with pytest.raises(click.ClickException) as excinfo:
            load_json(path)
        assert "nothing.json" in excinfo.value.message

    def test_invalid_json_raises_naming_file(self, tmp_path):
        """Undecodable JSON raises ClickException naming the file."""
        path = _write_text(tmp_path, "{not json", name="broken.json")
        with pytest.raises(click.ClickException) as excinfo:
            load_json(path)
        assert "broken.json" in excinfo.value.message


class TestCommittedFixtures:
    """The synthetic fixtures load in the form their README names."""

    def test_dialect_fixture_is_system_keyed(self):
        """dialect/ is system-keyed with RING and BOOST."""
        loaded = load_json(FIXTURES / "dialect" / "export.json")
        assert loaded.system_keyed is True
        assert {"RING", "BOOST"} <= set(loaded.ao)
        assert loaded.ao["RING"]["HCM"]["Setpoint"]["Tolerance"] == "NaN"
        assert loaded.ao["RING"]["HCM"]["Monitor"]["Range"] == [-1, "Inf"]

    def test_wrapped_fixture_is_unwrapped(self):
        """wrapped/ loses its ao wrapper."""
        loaded = load_json(FIXTURES / "wrapped" / "export.json")
        assert "ao" not in loaded.ao
        assert "BPM" in loaded.ao
        assert loaded.system_keyed is False

    def test_paired_fixture_loads_ad_and_export(self):
        """paired/ picks up its AD sibling and its _export block."""
        loaded = load_json(FIXTURES / "paired" / "quokka.ring.ao.json")
        assert loaded.ad is not None
        assert loaded.ad["SubMachine"] == "RING"
        assert loaded.export is not None
        assert loaded.export["machine"] == "Quokka"
        assert "_export" not in loaded.ao
        assert loaded.system_keyed is False

    @pytest.mark.parametrize("name", ["tango", "dualkey", "casedup"])
    def test_flat_fixtures(self, name):
        """The remaining synthetic fixtures load without refusal."""
        loaded = load_json(FIXTURES / name / "export.json")
        assert loaded.ao
