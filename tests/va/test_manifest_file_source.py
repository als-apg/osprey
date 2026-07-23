"""Tests for the file-backed channel source (``loaders.load_manifest_file``).

This is the facility-neutral seam: a facility that does not use the built-in
generated manifest supplies a ``{"channels": [...]}`` JSON file carrying the
same per-channel schema ``build_manifest()`` produces. Pure-python and
file-level -- no softioc, no lattice, no CA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator.manifest import (
    MANIFEST_CHANNEL_KEYS,
    ManifestFileError,
    load_manifest_file,
)


def _channel_entry(address: str, **overrides) -> dict:
    """One schema-complete manifest channel entry. Hierarchy defaults use a
    three-part-address facility's convention: the experiment rides in
    ``ring`` and unused levels are empty strings."""
    entry = {
        "address": address,
        "ring": "ZZEXP",
        "system": "DIAG",
        "family": "CAM",
        "device": "01",
        "field": "EXPOSURE",
        "subfield": "RB",
        "partition": "static-noisy",
        "record_type": "ai",
        "noise": True,
    }
    entry.update(overrides)
    return entry


def _write_manifest(tmp_path, channels) -> Path:
    path = tmp_path / "channels_manifest.json"
    path.write_text(json.dumps({"channels": channels}))
    return path


class TestLoadManifestFile:
    def test_loads_channels_from_valid_file(self, tmp_path):
        channels = [
            _channel_entry("ZZEXP:DIAG:CAM:01:EXPOSURE:RB"),
            _channel_entry("ZZEXP:DIAG:CAM:01:EXPOSURE:SP", subfield="SP", partition="sp-echo"),
        ]
        loaded = load_manifest_file(_write_manifest(tmp_path, channels))
        assert loaded == channels

    def test_three_part_addresses_load_without_grammar_constraint(self, tmp_path):
        # The seam imposes no address grammar: a facility whose real PV names
        # are three-part strings loads through the same call, carrying its
        # identity in the hierarchy keys rather than the address text.
        channels = [
            _channel_entry(
                "ZZEXP:JET:PRESSURE",
                system="JET",
                family="TARGET",
                field="PRESSURE",
            )
        ]
        loaded = load_manifest_file(_write_manifest(tmp_path, channels))
        assert loaded[0]["address"] == "ZZEXP:JET:PRESSURE"

    def test_missing_file_raises_named_error(self, tmp_path):
        with pytest.raises(ManifestFileError, match="not found"):
            load_manifest_file(tmp_path / "absent.json")

    def test_invalid_json_raises_named_error(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json")
        with pytest.raises(ManifestFileError, match="not valid JSON"):
            load_manifest_file(path)

    def test_top_level_must_carry_a_channels_list(self, tmp_path):
        path = tmp_path / "shapeless.json"
        path.write_text(json.dumps({"pvs": []}))
        with pytest.raises(ManifestFileError, match="'channels' list"):
            load_manifest_file(path)

    def test_channel_missing_schema_keys_is_named_in_the_error(self, tmp_path):
        incomplete = _channel_entry("ZZEXP:DIAG:CAM:02:EXPOSURE:RB")
        del incomplete["partition"], incomplete["noise"]
        with pytest.raises(ManifestFileError, match="noise, partition"):
            load_manifest_file(_write_manifest(tmp_path, [incomplete]))

    def test_duplicate_address_raises(self, tmp_path):
        dup = _channel_entry("ZZEXP:DIAG:CAM:03:EXPOSURE:RB")
        with pytest.raises(ManifestFileError, match="duplicate address"):
            load_manifest_file(_write_manifest(tmp_path, [dup, dict(dup)]))

    def test_empty_address_raises(self, tmp_path):
        with pytest.raises(ManifestFileError, match="empty address"):
            load_manifest_file(_write_manifest(tmp_path, [_channel_entry("")]))

    def test_schema_keys_match_build_records_consumption(self):
        # The documented schema is exactly what ioc/records.py's
        # build_records() reads off each channel dict -- keep the frozen set
        # honest against the factory's field accesses.
        assert MANIFEST_CHANNEL_KEYS == {
            "address",
            "ring",
            "system",
            "family",
            "device",
            "field",
            "subfield",
            "partition",
            "record_type",
            "noise",
        }
