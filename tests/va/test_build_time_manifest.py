"""Tests for build-time channel-manifest generation from a project's data tree.

``osprey build`` is the only stage where a project's three paradigm channel
databases are still on disk (``tiers/`` is pruned from the built project and
the container mounts no databases), so this is where a facility's own
namespace becomes the channel set the virtual accelerator serves. Everything
here is pure filesystem work -- no container, no EPICS.

The skip path matters as much as the happy one: a data tree that ships no
paradigm databases, or whose edits made them disagree, must leave the build
alone and let the container keep its package fallback.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator.manifest import loaders
from osprey.services.virtual_accelerator.manifest.build import (
    LIMITS_FILENAME,
    MANIFEST_FILENAME,
    build_manifest,
    prepare_project_manifest,
    write_project_manifest,
)
from osprey.services.virtual_accelerator.manifest.paths import (
    DEFAULT_TIER,
    PACKAGE_PATHS,
    ManifestPaths,
)

# The files a data tree must carry for a manifest to be generated from it,
# relative to its data root. Copied (rather than the whole 2 MB bundle) so a
# test can knock one out and watch the gate close.
_SOURCE_FILES = (
    f"channel_databases/tiers/tier{DEFAULT_TIER}/hierarchical.json",
    f"channel_databases/tiers/tier{DEFAULT_TIER}/in_context.json",
    f"channel_databases/tiers/tier{DEFAULT_TIER}/middle_layer.json",
    "simulation/machine.json",
    "machine_state_channels.json",
    "channel_limits.json",
)


def _facility_tree(root: Path) -> Path:
    """Copy the bundled sources into ``root`` as a standalone facility tree."""
    for relative in _SOURCE_FILES:
        source = PACKAGE_PATHS.data_root / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return root


@pytest.fixture(scope="module")
def facility_tree(tmp_path_factory) -> Path:
    """An unedited copy of the bundled tree, shared by the read-only tests."""
    return _facility_tree(tmp_path_factory.mktemp("facility_data"))


@pytest.fixture
def editable_tree(tmp_path) -> Path:
    """A per-test copy, for the tests that edit the facility's data."""
    return _facility_tree(tmp_path / "data")


class TestManifestPaths:
    """The object that replaced the module-level package path constants."""

    def test_package_paths_anchor_the_bundled_tree(self):
        assert PACKAGE_PATHS.data_root.name == "data"
        assert PACKAGE_PATHS.data_root.parent.name == "control_assistant"
        assert PACKAGE_PATHS.tier == DEFAULT_TIER

    def test_bundled_tree_carries_every_source(self):
        assert PACKAGE_PATHS.missing_sources() == []

    def test_tier_selects_the_paradigm_subdirectory(self, facility_tree):
        tier1 = ManifestPaths(data_root=facility_tree, tier=1)
        assert tier1.hierarchical_db.parent.name == "tier1"
        assert tier1.in_context_db.parent.name == "tier1"
        assert tier1.middle_layer_db.parent.name == "tier1"

    def test_missing_sources_names_what_the_tree_lacks(self, editable_tree):
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        paths.middle_layer_db.unlink()
        paths.machine_json.unlink()

        assert paths.missing_sources() == [paths.middle_layer_db, paths.machine_json]

    def test_default_argument_is_the_package_tree(self):
        # Every runtime caller (the container entrypoint, the lattice
        # inventory, the strengths loader) still calls these with no
        # arguments; they must keep reading the bundled tree.
        assert loaders.load_hierarchical_channels() == loaders.load_hierarchical_channels(
            PACKAGE_PATHS
        )
        assert loaders.load_in_context_addresses() == loaders.load_in_context_addresses(
            PACKAGE_PATHS
        )
        assert (
            loaders.load_machine_state_candidate_addresses()
            == loaders.load_machine_state_candidate_addresses(PACKAGE_PATHS)
        )


class TestNonProfileBehaviorUnchanged:
    """A build that sources the bundled tree must produce today's manifest.

    The channel set itself is pinned by ``tests/va/test_manifest.py``'s
    measured ``EXPECTED_TOTAL`` / ``EXPECTED_RING_COUNTS`` against the
    no-argument ``build_manifest()`` — the call every runtime caller makes.
    What this class adds is that the default itself is the package tree: see
    ``TestManifestPaths.test_default_argument_is_the_package_tree``.
    """

    def test_metadata_records_the_tier_that_was_expanded(self):
        assert build_manifest()["_metadata"]["source_tier"] == DEFAULT_TIER


class TestPreparedFromFacilityTree:
    def test_copy_of_the_bundle_reproduces_the_bundle_manifest(self, facility_tree):
        prepared = prepare_project_manifest(facility_tree, DEFAULT_TIER)

        assert prepared is not None
        assert prepared.manifest == build_manifest()

    def test_limits_source_points_into_the_facility_tree(self, facility_tree):
        prepared = prepare_project_manifest(facility_tree, DEFAULT_TIER)

        assert prepared.limits_source == facility_tree / LIMITS_FILENAME

    def test_data_edit_is_reflected_in_the_generated_channel_set(self, editable_tree):
        baseline = prepare_project_manifest(editable_tree, DEFAULT_TIER)
        machine_json = editable_tree / "simulation" / "machine.json"
        machine = json.loads(machine_json.read_text())
        machine["channels"]["SR:VAC:GAUGE:SR99:PRESSURE:RB"] = {
            "value": 1e-9,
            "units": "Torr",
            "description": "Facility-added gauge",
        }
        machine_json.write_text(json.dumps(machine, indent=2))

        edited = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        addresses = {c["address"] for c in edited.manifest["channels"]}
        assert "SR:VAC:GAUGE:SR99:PRESSURE:RB" in addresses
        assert edited.manifest["_metadata"]["machine_json_novel_addresses"] == [
            "SR:VAC:GAUGE:SR99:PRESSURE:RB"
        ]
        assert (
            edited.manifest["_metadata"]["total_channels"]
            == baseline.manifest["_metadata"]["total_channels"] + 1
        )


class TestSkipGate:
    """No paradigm databases (or no limits) means: generate nothing at all."""

    def test_tree_without_paradigm_databases_skips(self, editable_tree):
        shutil.rmtree(editable_tree / "channel_databases")

        assert prepare_project_manifest(editable_tree, DEFAULT_TIER) is None

    def test_tree_without_the_requested_tier_skips(self, facility_tree):
        # The bundle ships tier 1 and tier 3; a build resolving some other
        # tier has no databases to expand.
        assert prepare_project_manifest(facility_tree, 2) is None

    def test_tree_without_drive_limits_skips(self, editable_tree):
        (editable_tree / LIMITS_FILENAME).unlink()

        # Limits and manifest ship together or not at all: a manifest without
        # limits is an accelerator that accepts any setpoint.
        assert prepare_project_manifest(editable_tree, DEFAULT_TIER) is None

    def test_tree_without_machine_json_skips(self, editable_tree):
        (editable_tree / "simulation" / "machine.json").unlink()

        assert prepare_project_manifest(editable_tree, DEFAULT_TIER) is None

    def test_skipping_writes_nothing(self, editable_tree):
        before = sorted(p.relative_to(editable_tree) for p in editable_tree.rglob("*"))
        (editable_tree / LIMITS_FILENAME).unlink()

        prepare_project_manifest(editable_tree, DEFAULT_TIER)

        after = sorted(p.relative_to(editable_tree) for p in editable_tree.rglob("*"))
        assert after == [p for p in before if p != Path(LIMITS_FILENAME)]


class TestParadigmMismatchDegradesToFallback:
    """Facility data is user-owned: disagreeing databases warn, never abort."""

    def test_edited_tree_warns_and_falls_back(self, editable_tree, caplog):
        in_context = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER).in_context_db
        db = json.loads(in_context.read_text())
        dropped = db["channels"].pop()
        in_context.write_text(json.dumps(db, indent=2))

        with caplog.at_level(logging.WARNING):
            prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared is None
        assert dropped["address"] in caplog.text
        assert "built-in channel set" in caplog.text

    def test_build_manifest_itself_still_raises(self, editable_tree):
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        db = json.loads(paths.in_context_db.read_text())
        db["channels"].pop()
        paths.in_context_db.write_text(json.dumps(db, indent=2))

        # The build step chooses to degrade; the generator itself does not
        # silently reconcile a broken namespace.
        with pytest.raises(loaders.ParadigmMismatchError):
            build_manifest(paths)


class TestWriteProjectManifest:
    def test_both_files_land_in_the_mounted_simulation_directory(self, facility_tree, tmp_path):
        prepared = prepare_project_manifest(facility_tree, DEFAULT_TIER)
        project_data = tmp_path / "project" / "data"
        project_data.mkdir(parents=True)

        manifest_path = write_project_manifest(prepared, project_data)

        # data/simulation/ is the directory the container already bind-mounts,
        # so neither file needs a compose change.
        assert manifest_path == project_data / "simulation" / MANIFEST_FILENAME
        assert manifest_path.is_file()
        assert (project_data / "simulation" / LIMITS_FILENAME).is_file()

    def test_written_manifest_loads_through_the_container_reader(self, facility_tree, tmp_path):
        prepared = prepare_project_manifest(facility_tree, DEFAULT_TIER)
        project_data = tmp_path / "data"
        project_data.mkdir()

        manifest_path = write_project_manifest(prepared, project_data)

        # This is the exact call the entrypoint makes on VA_CHANNELS_FILE.
        channels = loaders.load_manifest_file(manifest_path)
        assert len(channels) == prepared.manifest["_metadata"]["total_channels"]

    def test_limits_copy_prefers_the_built_project_tree(self, facility_tree, tmp_path):
        prepared = prepare_project_manifest(facility_tree, DEFAULT_TIER)
        project_data = tmp_path / "data"
        project_data.mkdir()
        # Stands in for a facility overlay landing on the project's limits.
        (project_data / LIMITS_FILENAME).write_text('{"channels": {}}\n')

        write_project_manifest(prepared, project_data)

        assert (project_data / "simulation" / LIMITS_FILENAME).read_text() == '{"channels": {}}\n'

    def test_limits_copy_falls_back_to_the_prepared_source(self, facility_tree, tmp_path):
        prepared = prepare_project_manifest(facility_tree, DEFAULT_TIER)
        project_data = tmp_path / "data"
        project_data.mkdir()

        write_project_manifest(prepared, project_data)

        assert (project_data / "simulation" / LIMITS_FILENAME).read_bytes() == (
            facility_tree / LIMITS_FILENAME
        ).read_bytes()
