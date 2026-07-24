"""Tests for entrypoint.py's facility-neutral source configuration.

Exercises the resolution helpers directly (not ``main()``, which also needs
a real ``machine.json`` and softioc) -- same shape as
``test_entrypoint_fault_env.py``. Each helper reads ``os.environ`` itself,
so tests set env vars via ``monkeypatch``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator import entrypoint


class TestResolveChannelsFile:
    """Backs VA_CHANNELS_FILE -- the file-backed channel source selector."""

    def test_unset_means_builtin_manifest(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VA_CHANNELS_FILE", raising=False)
        assert entrypoint._resolve_channels_file(tmp_path) is None

    def test_empty_means_builtin_manifest(self, monkeypatch, tmp_path):
        # The compose passthrough sends "" when the host var is absent --
        # empty must behave exactly like unset.
        monkeypatch.setenv("VA_CHANNELS_FILE", "")
        assert entrypoint._resolve_channels_file(tmp_path) is None

    def test_relative_path_resolves_against_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VA_CHANNELS_FILE", "channels_manifest.json")
        assert entrypoint._resolve_channels_file(tmp_path) == tmp_path / "channels_manifest.json"

    def test_absolute_path_is_kept(self, monkeypatch, tmp_path):
        absolute = tmp_path / "elsewhere" / "manifest.json"
        monkeypatch.setenv("VA_CHANNELS_FILE", str(absolute))
        assert entrypoint._resolve_channels_file(tmp_path / "data") == absolute


class TestResolveLatticeMode:
    """Backs VA_LATTICE -- whether PhysicsBridge (hence PyAT) is constructed."""

    def test_default_is_builtin_for_builtin_manifest(self, monkeypatch):
        monkeypatch.delenv("VA_LATTICE", raising=False)
        assert entrypoint._resolve_lattice_mode(None) == entrypoint.LATTICE_BUILTIN

    def test_default_is_none_for_file_backed_manifest(self, monkeypatch):
        monkeypatch.delenv("VA_LATTICE", raising=False)
        assert entrypoint._resolve_lattice_mode(Path("/x/manifest.json")) == entrypoint.LATTICE_NONE

    def test_explicit_builtin_overrides_file_backed_default(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "builtin")
        assert (
            entrypoint._resolve_lattice_mode(Path("/x/manifest.json")) == entrypoint.LATTICE_BUILTIN
        )

    def test_explicit_none_overrides_builtin_default(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "none")
        assert entrypoint._resolve_lattice_mode(None) == entrypoint.LATTICE_NONE

    def test_empty_behaves_like_unset(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "")
        assert entrypoint._resolve_lattice_mode(None) == entrypoint.LATTICE_BUILTIN

    def test_value_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "NONE")
        assert entrypoint._resolve_lattice_mode(None) == entrypoint.LATTICE_NONE

    def test_unknown_value_is_fatal(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "als")
        with pytest.raises(SystemExit, match="VA_LATTICE"):
            entrypoint._resolve_lattice_mode(None)


class TestParameterizedDataLoads:
    """The file-backed path reads facility data from the mount, never from
    the bundled tutorial files; the no-arg defaults keep the historical
    bundled-template reads byte-identical for the built-in path."""

    def test_boot_values_from_explicit_machine_json(self, tmp_path):
        machine = tmp_path / "machine.json"
        machine.write_text(
            json.dumps(
                {
                    "channels": {
                        "ZZEXP:LASER:ENERGY:RB": {"value": 1.25},
                        "ZZEXP:LASER:MODE:RB": {"expr": "derived, no value"},
                    }
                }
            )
        )
        assert entrypoint._load_boot_values(machine) == {"ZZEXP:LASER:ENERGY:RB": 1.25}

    def test_drive_limits_from_explicit_limits_file(self, tmp_path):
        limits = tmp_path / "channel_limits.json"
        limits.write_text(
            json.dumps(
                {
                    "defaults": {"writable": False},
                    "ZZEXP:JET:STAGE:SP": {
                        "writable": True,
                        "min_value": -5.0,
                        "max_value": 5.0,
                    },
                    "ZZEXP:LOCKED:DOWN:SP": {"min_value": 0, "max_value": 1},
                }
            )
        )
        # The non-writable default suppresses the second entry; only the
        # explicitly writable one yields a clamp band.
        assert entrypoint._load_drive_limits(limits) == {"ZZEXP:JET:STAGE:SP": (-5.0, 5.0)}

    def test_no_arg_defaults_still_read_the_bundled_data(self):
        # The ALS regression half of the seam: calling with no argument must
        # keep returning the bundled tutorial data (non-empty, known shape).
        assert entrypoint._load_boot_values()
        assert entrypoint._load_drive_limits()
