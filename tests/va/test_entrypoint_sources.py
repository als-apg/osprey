"""Tests for entrypoint.py's facility-neutral source configuration.

Exercises the resolution helpers directly (not ``main()``, which also needs
a real ``machine.json`` and softioc) -- same shape as
``test_entrypoint_fault_env.py``. Each helper reads ``os.environ`` itself,
so tests set env vars via ``monkeypatch``.

The one thing neither helper will do is choose a namespace on its own. A
container that picks the framework's bundled demo channels when it was told
nothing serves addresses that belong to one particular facility, under
whatever name the deployment gave it -- and from a client's side that is
indistinguishable from serving the facility. So the demo namespace is a
committed file like any other manifest, reachable only by being named, and
the tests below pin both halves: the refusal, and that naming it still
yields exactly the namespace the generator produces.
"""

from __future__ import annotations

import json

import pytest

from osprey.services.virtual_accelerator import entrypoint
from osprey.services.virtual_accelerator.manifest.build import build_manifest
from osprey.services.virtual_accelerator.manifest.loaders import load_manifest_file
from osprey.services.virtual_accelerator.manifest.paths import MANIFEST_OUTPUT


class TestResolveChannelsFile:
    """Backs VA_CHANNELS_FILE -- the channel source, which is required."""

    def test_unset_is_fatal(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VA_CHANNELS_FILE", raising=False)
        with pytest.raises(SystemExit) as excinfo:
            entrypoint._resolve_channels_file(tmp_path)
        assert "VA_CHANNELS_FILE" in str(excinfo.value)

    def test_empty_is_fatal_too(self, monkeypatch, tmp_path):
        # The compose passthrough sends "" when the host var is absent --
        # empty must behave exactly like unset, and unset is a refusal.
        monkeypatch.setenv("VA_CHANNELS_FILE", "")
        with pytest.raises(SystemExit, match="VA_CHANNELS_FILE"):
            entrypoint._resolve_channels_file(tmp_path)

    def test_refusal_names_both_ways_out(self, monkeypatch, tmp_path):
        """The message has to leave an operator somewhere to go.

        Two audiences reach it, and each needs a different sentence: a project
        deployment whose build should have written the pointer, and a hand-run
        container that wanted the demo. So the refusal names ``osprey build``
        for the first and the packaged manifest's own resolved path -- plus
        the lattice that goes with it -- for the second.
        """
        monkeypatch.delenv("VA_CHANNELS_FILE", raising=False)
        with pytest.raises(SystemExit) as excinfo:
            entrypoint._resolve_channels_file(tmp_path)
        message = str(excinfo.value)
        assert "osprey build" in message
        assert str(MANIFEST_OUTPUT) in message
        assert f"VA_LATTICE={entrypoint.LATTICE_BUILTIN}" in message
        # And it says what it will not do, so the refusal reads as a decision
        # rather than as a missing default.
        assert "never falls back" in message

    def test_relative_path_resolves_against_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VA_CHANNELS_FILE", "channels_manifest.json")
        assert entrypoint._resolve_channels_file(tmp_path) == tmp_path / "channels_manifest.json"

    def test_absolute_path_is_kept(self, monkeypatch, tmp_path):
        absolute = tmp_path / "elsewhere" / "manifest.json"
        monkeypatch.setenv("VA_CHANNELS_FILE", str(absolute))
        assert entrypoint._resolve_channels_file(tmp_path / "data") == absolute


class TestResolveLatticeMode:
    """Backs VA_LATTICE -- whether PhysicsBridge (hence PyAT) is constructed."""

    def test_default_is_no_lattice(self, monkeypatch):
        # The only model this process could build unasked is the framework's
        # own tutorial ring, which is the physics half of the same
        # substitution the channel source refuses. So it is asked for.
        monkeypatch.delenv("VA_LATTICE", raising=False)
        assert entrypoint._resolve_lattice_mode() == entrypoint.LATTICE_NONE

    def test_explicit_builtin_is_honoured(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "builtin")
        assert entrypoint._resolve_lattice_mode() == entrypoint.LATTICE_BUILTIN

    def test_explicit_none_is_honoured(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "none")
        assert entrypoint._resolve_lattice_mode() == entrypoint.LATTICE_NONE

    def test_empty_behaves_like_unset(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "")
        assert entrypoint._resolve_lattice_mode() == entrypoint.LATTICE_NONE

    def test_value_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "BUILTIN")
        assert entrypoint._resolve_lattice_mode() == entrypoint.LATTICE_BUILTIN

    def test_unknown_value_is_fatal(self, monkeypatch):
        monkeypatch.setenv("VA_LATTICE", "als")
        with pytest.raises(SystemExit, match="VA_LATTICE"):
            entrypoint._resolve_lattice_mode()


class TestPackagedDemoManifest:
    """The demo namespace the refusal points at, and what naming it buys.

    Removing the implicit fallback only holds if the explicit route lands in
    the same place: the tutorial quick-start (``scripts/va/run_va.sh``) and
    the image boot gate both name this file, and both are written against the
    channel set the generator produces. A committed manifest that had drifted
    from ``build_manifest()`` would move them off it silently -- same script,
    same readiness line, a different machine.
    """

    def test_the_packaged_manifest_is_the_generated_namespace(self):
        assert load_manifest_file(MANIFEST_OUTPUT) == build_manifest()["channels"]

    def test_naming_it_resolves_to_that_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VA_CHANNELS_FILE", str(MANIFEST_OUTPUT))
        assert entrypoint._resolve_channels_file(tmp_path) == MANIFEST_OUTPUT


class TestParameterizedDataLoads:
    """Every boot reads facility data from the mount, never from the bundled
    tutorial files. The no-arg defaults are no longer a boot path -- ``main()``
    always passes the mounted files -- but they remain the bundled-template
    reads the build-time and record-factory tests are written against, so they
    are pinned here too."""

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
        # Calling with no argument must keep returning the bundled tutorial
        # data (non-empty, known shape) for the callers that still do.
        assert entrypoint._load_boot_values()
        assert entrypoint._load_drive_limits()
