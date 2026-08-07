"""End-to-end wiring of the build's virtual-accelerator manifest step.

``tests/va/test_build_time_manifest.py`` covers the generator itself; this
module covers the two things only a real ``osprey build`` can show: that the
generated files land where the container's existing bind mount will find them,
and that the ``.env`` keys pointing at them appear (and disappear) together
with the files.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
import yaml

from osprey.services.virtual_accelerator.manifest.build import (
    LIMITS_FILENAME,
    MANIFEST_FILENAME,
)
from osprey.services.virtual_accelerator.manifest.loaders import load_manifest_file
from osprey.utils.dotenv import parse_dotenv_file


def _bundle_data_dir() -> Path:
    import osprey

    return Path(osprey.__file__).parent / "templates" / "apps" / "control_assistant" / "data"


def _write_profile(profile_dir: Path) -> Path:
    """A profile sourcing its own copy of the bundled control-assistant tree.

    ``hierarchical`` resolves to tier 3, the tier whose three paradigm
    databases agree — the shape the generator can actually build from.
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_bundle_data_dir(), profile_dir / "data")

    profile = {
        "name": "VA Build Step Test",
        "data_bundle": "control_assistant",
        "data": "data",
        "provider": "cborg",
        "model": "haiku",
        "channel_finder_mode": "hierarchical",
    }
    path = profile_dir / "profile.yml"
    path.write_text(yaml.dump(profile, default_flow_style=False))
    return path


def _invoke_build(profile_path: Path, output_dir: Path, project_name: str):
    from click.testing import CliRunner

    from osprey.cli.main import cli

    output_dir.mkdir(parents=True, exist_ok=True)
    return CliRunner().invoke(
        cli,
        [
            "build",
            project_name,
            str(profile_path),
            "--output-dir",
            str(output_dir),
            "--skip-deps",
            "--skip-lifecycle",
        ],
    )


def _build(profile_path: Path, output_dir: Path, project_name: str = "va-proj") -> Path:
    result = _invoke_build(profile_path, output_dir, project_name)
    assert result.exit_code == 0, (
        f"build failed (exit={result.exit_code})\n"
        f"--- output ---\n{result.output}\n"
        f"--- exception ---\n{result.exception}"
    )
    return output_dir / project_name


@pytest.fixture(autouse=True)
def detected_provider_key(monkeypatch):
    """Keep the build's provider-credential summary from warning about a miss."""
    monkeypatch.setenv("CBORG_API_KEY", "test-key")


class TestGeneratedFromProfileData:
    def test_manifest_and_limits_land_in_the_mounted_directory(self, tmp_path):
        profile_path = _write_profile(tmp_path / "profile")

        project_dir = _build(profile_path, tmp_path / "out")

        # data/simulation/ is the directory the VA compose file already
        # bind-mounts, so neither file needs a compose change to be reachable.
        simulation = project_dir / "data" / "simulation"
        assert (simulation / MANIFEST_FILENAME).is_file()
        assert (simulation / LIMITS_FILENAME).is_file()

    def test_manifest_loads_through_the_container_reader(self, tmp_path):
        profile_path = _write_profile(tmp_path / "profile")

        project_dir = _build(profile_path, tmp_path / "out")

        channels = load_manifest_file(project_dir / "data" / "simulation" / MANIFEST_FILENAME)
        assert channels, "generated manifest served no channels"

    def test_env_points_the_container_at_the_generated_manifest(self, tmp_path):
        profile_path = _write_profile(tmp_path / "profile")

        project_dir = _build(profile_path, tmp_path / "out")

        env = parse_dotenv_file(project_dir / ".env")
        assert env["VA_CHANNELS_FILE"] == MANIFEST_FILENAME
        # Without this the entrypoint defaults a file-backed source to
        # lattice "none" and the PyAT physics bridge is never built.
        assert env["VA_LATTICE"] == "builtin"

    def test_facility_data_edit_reaches_the_served_channel_set(self, tmp_path):
        profile_dir = tmp_path / "profile"
        profile_path = _write_profile(profile_dir)
        machine_json = profile_dir / "data" / "simulation" / "machine.json"
        machine = json.loads(machine_json.read_text())
        machine["channels"]["SR:VAC:GAUGE:SR99:PRESSURE:RB"] = {
            "value": 1e-9,
            "units": "Torr",
            "description": "Facility-added gauge",
        }
        machine_json.write_text(json.dumps(machine, indent=2))

        project_dir = _build(profile_path, tmp_path / "out")

        channels = load_manifest_file(project_dir / "data" / "simulation" / MANIFEST_FILENAME)
        assert "SR:VAC:GAUGE:SR99:PRESSURE:RB" in {c["address"] for c in channels}

    def test_limits_copy_matches_the_project_limits(self, tmp_path):
        profile_path = _write_profile(tmp_path / "profile")

        project_dir = _build(profile_path, tmp_path / "out")

        # The entrypoint reads drive limits from beside the manifest; they
        # must be the same limits the project ships, or the accelerator
        # enforces bounds the project never declared.
        assert (project_dir / "data" / "simulation" / LIMITS_FILENAME).read_bytes() == (
            project_dir / "data" / LIMITS_FILENAME
        ).read_bytes()


class TestSkippedWithoutParadigmDatabases:
    """A tree that can't back a manifest wires nothing at all."""

    def test_neither_files_nor_env_keys_appear(self, tmp_path):
        profile_dir = tmp_path / "profile"
        profile_path = _write_profile(profile_dir)
        shutil.rmtree(profile_dir / "data" / "channel_databases" / "tiers")

        project_dir = _build(profile_path, tmp_path / "out")

        env = parse_dotenv_file(project_dir / ".env")
        assert "VA_CHANNELS_FILE" not in env
        assert "VA_LATTICE" not in env
        assert not (project_dir / "data" / "simulation" / MANIFEST_FILENAME).exists()

    def test_the_build_still_succeeds(self, tmp_path):
        profile_dir = tmp_path / "profile"
        profile_path = _write_profile(profile_dir)
        (profile_dir / "data" / LIMITS_FILENAME).unlink()

        # Profile data is user-owned: a tree the generator can't use is not a
        # build failure, it just leaves the container's package fallback.
        project_dir = _build(profile_path, tmp_path / "out")
        assert (project_dir / "config.yml").is_file()


class TestCorruptFacilityDataFailsTheBuild:
    """Absent databases fall back; unreadable ones must not.

    Falling back on a corrupt tree would hand the operator a virtual
    accelerator serving the framework's bundled channel namespace while they
    believe they are driving their own facility's — for a control system that
    is a worse outcome than a failed build.
    """

    def test_error_names_the_offending_file(self, tmp_path, caplog):
        profile_dir = tmp_path / "profile"
        profile_path = _write_profile(profile_dir)
        corrupt = profile_dir / "data" / "channel_databases" / "tiers" / "tier3" / "in_context.json"
        corrupt.write_text('{"channels": [ truncated')

        with caplog.at_level(logging.ERROR):
            result = _invoke_build(profile_path, tmp_path / "out", "va-proj")

        assert result.exit_code != 0
        # The operator gets the path to go fix, not a bare KeyError from
        # inside a database parser. Asserted on the log record rather than
        # result.output, which rich line-wraps long paths in.
        assert str(corrupt) in caplog.text
        assert "Build error" in caplog.text
        assert "Unexpected error" not in caplog.text
