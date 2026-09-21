"""Tests for build-time channel-manifest generation from a project's data tree.

``osprey build`` is the only stage where a project's three paradigm channel
databases are still on disk (``tiers/`` is pruned from the built project and
the container mounts no databases), so this is where a facility's own
namespace becomes the channel set the virtual accelerator serves. Everything
here is pure filesystem work -- no container, no EPICS.

The partial tree matters as much as the whole one: a project's namespace is
whatever paradigm databases it staged, and a manifest is built from that
subset. There is no fallback left to take -- a tree that names no channels at
all backs no manifest, and a build deploying a virtual accelerator on it
refuses (``tests/cli/test_build_va_manifest_honesty.py``) rather than letting
the container serve the framework's bundled tutorial namespace.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.services.virtual_accelerator.manifest import classify, loaders
from osprey.services.virtual_accelerator.manifest.build import (
    LIMITS_FILENAME,
    MANIFEST_FILENAME,
    CorruptChannelSourcesError,
    NoChannelSourcesError,
    build_manifest,
    manifest_gap_reason,
    prepare_project_manifest,
    write_project_manifest,
)
from osprey.services.virtual_accelerator.manifest.paths import (
    DEFAULT_TIER,
    PACKAGE_PATHS,
    ManifestPaths,
)
from tests._graph_index import build_index_from_ttl, default_index_path

# The tiered paradigm databases the manifest expands, derived by subtracting
# ``graph`` from the paradigm registry: a graph store is seeded from the corpus
# TTL and ships no tier database, so it contributes no source file here (see
# the exemption comment in manifest/build.py). Registering a file-backed
# paradigm adds its database to this list without an edit.
_PARADIGM_DB_FILES = tuple(
    f"channel_databases/tiers/tier{DEFAULT_TIER}/{name}.json"
    for name in sorted(set(VALID_CHANNEL_FINDER_MODES) - {"graph"})
)

# The files a data tree must carry for a manifest to be generated from it,
# relative to its data root. Copied (rather than the whole 2 MB bundle) so a
# test can knock one out and watch the gate close.
_SOURCE_FILES = (
    *_PARADIGM_DB_FILES,
    "simulation/machine.json",
    "machine_state_channels.json",
    "channel_limits.json",
)

# What a tree adds to those to serve a virtual accelerator: the deck, and the
# bindings derived against it. They come as a pair -- bindings without their
# deck are a missing source, and a tree carrying neither partitions by the
# no-bindings fallback -- so a copy that is to reproduce the bundle's own
# manifest carries both, while the tests that watch the fallback carry
# neither.
_SIMULATION_MODEL_FILES = (
    "simulation/lattice.json",
    "simulation/va_bindings.json",
)


#: A hierarchical database levelled the way another facility levels one --
#: the shipped worked example, five levels with no ring/field/subfield among
#: them. Every partition rule and every manifest identity key is read off
#: those names, so this tree describes a hierarchy the classifier cannot be
#: evaluated against.
_FOREIGN_LEVELS_DB = (
    PACKAGE_PATHS.data_root / "channel_databases" / "examples" / "hierarchical_jlab_style.json"
)

#: A database levelled exactly the way the classifier reads one, whose tokens
#: belong to no partition rule. Readable, classifiable, and classified as
#: static-noisy throughout -- which is a different outcome from the one above
#: and is reported differently.
_FOREIGN_TOKEN_DB = {
    "hierarchy": {
        "levels": [
            {"name": "ring", "type": "tree"},
            {"name": "system", "type": "tree"},
            {"name": "family", "type": "tree"},
            {"name": "device", "type": "instances"},
            {"name": "field", "type": "tree"},
            {"name": "subfield", "type": "tree"},
        ],
        "naming_pattern": "{ring}:{system}:{family}:{device}:{field}:{subfield}",
    },
    "tree": {
        "ZZLINAC": {
            "PWR": {
                "KLYSTRON": {
                    "DEVICE": {
                        "_expansion": {"_type": "list", "_instances": ["01", "02"]},
                        "POWER": {"CTRL": {}, "MEAS": {}},
                    }
                }
            }
        }
    },
}


def _facility_tree(root: Path, *, simulation_model: bool = False) -> Path:
    """Copy the bundled sources into ``root`` as a standalone facility tree.

    Args:
        root: Where the copy lands; becomes the tree's data root.
        simulation_model: Also copy the deck and the bindings derived against
            it, which is what makes the copy serve the same virtual
            accelerator the bundle does. Left out by default so a test can
            watch the no-bindings fallback partition a tree.
    """
    sources = (*_SOURCE_FILES, *(_SIMULATION_MODEL_FILES if simulation_model else ()))
    for relative in sources:
        source = PACKAGE_PATHS.data_root / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return root


@pytest.fixture(scope="module")
def facility_tree(tmp_path_factory) -> Path:
    """An unedited copy of the bundled tree, shared by the read-only tests.

    Serves no virtual accelerator: the tests on this fixture are the ones that
    watch a tree with no bindings partition by the fallback.
    """
    return _facility_tree(tmp_path_factory.mktemp("facility_data"))


@pytest.fixture(scope="module")
def served_facility_tree(tmp_path_factory) -> Path:
    """The same copy, carrying the bundle's own deck and bindings too.

    What a facility hands a container: the whole tree, so the accelerator the
    copy serves is the accelerator the bundle serves.
    """
    return _facility_tree(tmp_path_factory.mktemp("served_data"), simulation_model=True)


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
        paths.machine_json.unlink()

        assert paths.missing_sources() == [paths.machine_json]

    def test_an_absent_paradigm_database_is_not_a_missing_source(self, editable_tree):
        """It is a namespace the project did not stage, which is its own answer."""
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        paths.middle_layer_db.unlink()

        assert paths.missing_sources() == []
        assert paths.staged_paradigms == ("hierarchical", "in_context")
        assert paths.absent_paradigms == ("middle_layer",)

    def test_staged_paradigms_is_empty_when_the_tier_has_no_databases(self, editable_tree):
        shutil.rmtree(editable_tree / "channel_databases")
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)

        assert paths.staged_paradigms == ()
        assert len(paths.absent_paradigms) == 3

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
    def test_copy_of_the_bundle_reproduces_the_bundle_manifest(self, served_facility_tree):
        prepared = prepare_project_manifest(served_facility_tree, DEFAULT_TIER)

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


class TestStagedSubset:
    """Any non-empty subset of staged databases yields a manifest from those."""

    def test_one_staged_database_backs_a_manifest(self, editable_tree):
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        paths.in_context_db.unlink()
        paths.middle_layer_db.unlink()

        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared is not None
        assert prepared.manifest["_metadata"]["source_paradigms"] == ["hierarchical"]
        assert prepared.manifest["_metadata"]["absent_paradigms"] == [
            "in_context",
            "middle_layer",
        ]

    def test_a_subset_serves_the_same_channels_the_whole_tree_does(self, editable_tree):
        """The databases describe one namespace, so dropping copies of it keeps it."""
        whole = prepare_project_manifest(editable_tree, DEFAULT_TIER)
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        paths.in_context_db.unlink()
        paths.middle_layer_db.unlink()

        subset = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert subset.manifest["channels"] == whole.manifest["channels"]

    def test_without_the_hierarchical_database_channels_carry_no_identity_keys(self, editable_tree):
        """The cost of the subset, stated in the manifest rather than hidden."""
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        paths.hierarchical_db.unlink()

        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared.manifest["_metadata"]["source_paradigms"] == [
            "in_context",
            "middle_layer",
        ]
        sample = prepared.manifest["channels"][0]
        assert sample["ring"] == "" and sample["family"] == ""
        assert sample["partition"] == classify.PARTITION_STATIC_NOISY
        # The channel SET is still the project's own, which is the point.
        assert (
            prepared.manifest["_metadata"]["total_channels"]
            == (build_manifest()["_metadata"]["total_channels"])
        )

    def test_a_foreign_levelled_database_yields_a_manifest_naming_what_it_lacks(
        self, editable_tree
    ):
        """A tree levelled some other way is a namespace, not a broken file."""
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        paths.in_context_db.unlink()
        paths.middle_layer_db.unlink()
        shutil.copy2(_FOREIGN_LEVELS_DB, paths.hierarchical_db)

        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared is not None
        metadata = prepared.manifest["_metadata"]
        reason = metadata["unclassified_reason"]
        assert reason.startswith("levels system/family/sector/device/pv lack ")
        for level in ("ring", "field", "subfield"):
            assert level in reason
        assert metadata["setpoint_count"] == 0
        assert set(metadata["by_partition"]) == {classify.PARTITION_STATIC_NOISY}
        # Classified nothing means exactly that: no channel carries an
        # identity key derived from a level this classifier cannot read.
        assert all(channel["ring"] == "" for channel in prepared.manifest["channels"])
        assert all(channel["subfield"] == "" for channel in prepared.manifest["channels"])
        # And the addresses are still the facility's own.
        assert "MQS1L02.S" in {channel["address"] for channel in prepared.manifest["channels"]}

    def test_a_six_level_database_of_foreign_tokens_classifies_and_says_so(self, editable_tree):
        """The levels are readable, the tokens match no rule: still a manifest."""
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        paths.in_context_db.unlink()
        paths.middle_layer_db.unlink()
        paths.hierarchical_db.write_text(json.dumps(_FOREIGN_TOKEN_DB))

        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared is not None
        metadata = prepared.manifest["_metadata"]
        # Nothing was unreadable, so no reason is recorded -- the tokens
        # simply matched no rule.
        assert "unclassified_reason" not in metadata
        assert metadata["setpoint_count"] == 0
        assert set(metadata["by_partition"]) == {classify.PARTITION_STATIC_NOISY}
        assert set(metadata["by_ring"]) == {"ZZLINAC"}
        served = {c["address"]: c for c in prepared.manifest["channels"]}
        assert served["ZZLINAC:PWR:KLYSTRON:01:POWER:CTRL"]["family"] == "KLYSTRON"

    def test_build_manifest_refuses_a_tree_that_stages_nothing(self, editable_tree):
        shutil.rmtree(editable_tree / "channel_databases")
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)

        with pytest.raises(NoChannelSourcesError):
            build_manifest(paths)


class TestSkipGate:
    """No channel databases at all (or no limits) means: generate nothing."""

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


class TestParadigmMismatchYieldsNoManifest:
    """Databases that contradict each other describe no namespace to serve."""

    def test_edited_tree_warns_and_yields_nothing(self, editable_tree, caplog):
        in_context = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER).in_context_db
        db = json.loads(in_context.read_text())
        dropped = db["channels"].pop()
        in_context.write_text(json.dumps(db, indent=2))

        with caplog.at_level(logging.WARNING):
            prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared is None
        assert dropped["address"] in caplog.text
        # The old escape hatch, gone: nothing here offers the bundled channel
        # set to a project whose own databases disagree.
        assert "built-in channel set" not in caplog.text
        assert manifest_gap_reason(editable_tree, DEFAULT_TIER).endswith(
            "describe different channel sets"
        )

    def test_build_manifest_itself_still_raises(self, editable_tree):
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        db = json.loads(paths.in_context_db.read_text())
        db["channels"].pop()
        paths.in_context_db.write_text(json.dumps(db, indent=2))

        # The build step chooses to degrade; the generator itself does not
        # silently reconcile a broken namespace.
        with pytest.raises(loaders.ParadigmMismatchError):
            build_manifest(paths)


#: Valid JSON in a shape no paradigm parser accepts -- a dict keyed by
#: address where the parsers expect a list of channel entries. This is what a
#: hand-edited or half-migrated database looks like: the file is there, the
#: build has every reason to believe it holds the facility's channels, and
#: reading it raises from deep inside a parser.
_SCHEMA_INVALID_DB = '{"channels": {"FACILITY:TIER:SRC": {"description": "profile"}}}\n'

#: A body no parser can get past at all -- truncated JSON. Every paradigm
#: parser rejects the schema-invalid one above too; this one is for tests
#: that want the refusal to come from the JSON decoder rather than from a
#: parser's shape check, whichever parser reads it.
_UNPARSEABLE_DB = '{"channels": [\n'


def _corrupt(path: Path, body: str = _SCHEMA_INVALID_DB) -> Path:
    """Overwrite a paradigm database with a body that cannot be loaded."""
    path.write_text(body)
    return path


class TestCorruptDatabaseDegrades:
    """A staged database that cannot be READ is not a database that is ABSENT.

    An unreadable one contributes nothing and is named as broken; the manifest
    is still built from the databases that are left. Only when nothing usable
    remains does the tree back no manifest -- and then the refusal names the
    files to repair rather than files to add.
    """

    def test_a_corrupt_database_contributes_nothing_and_the_others_still_build(
        self, editable_tree, caplog
    ):
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        _corrupt(paths.in_context_db)

        with caplog.at_level(logging.WARNING):
            prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared is not None
        metadata = prepared.manifest["_metadata"]
        # The census counts the databases that fed it, and only those.
        assert metadata["source_paradigms"] == ["hierarchical", "middle_layer"]
        assert metadata["absent_paradigms"] == []
        assert metadata["total_channels"] > 0
        assert "FACILITY:TIER:SRC" not in {c["address"] for c in prepared.manifest["channels"]}
        # And the degrade is on the record, with the file and one line of why.
        (corrupt,) = metadata["corrupt_paradigms"]
        assert corrupt["paradigm"] == "in_context"
        assert corrupt["path"] == str(paths.in_context_db.relative_to(editable_tree))
        assert corrupt["detail"]
        assert str(paths.in_context_db) in caplog.text

    def test_a_corrupt_database_is_never_counted_as_absent(self, editable_tree):
        """The two are different remedies: repair this file, or ship that one."""
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        _corrupt(paths.in_context_db)
        paths.middle_layer_db.unlink()

        metadata = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest["_metadata"]

        assert metadata["absent_paradigms"] == ["middle_layer"]
        assert [c["paradigm"] for c in metadata["corrupt_paradigms"]] == ["in_context"]

    def test_a_corrupt_hierarchical_database_costs_the_identity_keys(self, editable_tree):
        """It is the one paradigm carrying a hierarchy path, so losing it shows."""
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        _corrupt(paths.hierarchical_db)

        metadata = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest["_metadata"]

        assert "hierarchical" not in metadata["source_paradigms"]
        assert metadata["setpoint_count"] == 0

    def test_a_shape_invalid_middle_layer_database_is_corrupt_not_seeds_only(self, editable_tree):
        """Valid JSON in the wrong shape is a corrupt source, not an empty facility.

        The middle-layer parser used to skip any system or family that was not
        a mapping and read such a file as zero channels; on a tier that stages
        only that database, the build then fell through to the scenario seeds
        while the build fact said "N channels from its middle_layer database".
        The body here is the flat paradigm's list of channels, which is what a
        half-migrated file looks like.
        """
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        paths.hierarchical_db.unlink()
        paths.in_context_db.unlink()
        _corrupt(paths.middle_layer_db, '{"channels": [{"address": "FACILITY:TIER:SRC"}]}\n')

        assert prepare_project_manifest(editable_tree, DEFAULT_TIER) is None
        with pytest.raises(CorruptChannelSourcesError) as excinfo:
            build_manifest(paths)

        assert "middle_layer" in str(excinfo.value)

    def test_every_database_corrupt_backs_no_manifest(self, editable_tree, caplog):
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        for database in paths.paradigm_databases.values():
            _corrupt(database, _UNPARSEABLE_DB)

        with caplog.at_level(logging.WARNING):
            assert prepare_project_manifest(editable_tree, DEFAULT_TIER) is None

        assert "could not be read" in caplog.text

    def test_the_refusal_for_every_database_corrupt_names_them_as_unreadable(self, editable_tree):
        """Distinct wording from the absent case: these files exist and are broken."""
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        for database in paths.paradigm_databases.values():
            _corrupt(database, _UNPARSEABLE_DB)

        reason = manifest_gap_reason(editable_tree, DEFAULT_TIER)

        assert "present and could not be read" in reason
        assert "are all absent" not in reason
        assert "name no channels" not in reason
        for paradigm, database in paths.paradigm_databases.items():
            assert paradigm in reason
            assert str(database.relative_to(editable_tree)) in reason

    def test_build_manifest_itself_raises_when_nothing_is_readable(self, editable_tree):
        """The generator refuses; it never returns an empty namespace as an answer."""
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        for database in paths.paradigm_databases.values():
            _corrupt(database, _UNPARSEABLE_DB)

        with pytest.raises(CorruptChannelSourcesError) as excinfo:
            build_manifest(paths)

        assert not isinstance(excinfo.value, NoChannelSourcesError)
        assert "present and could not be read" in str(excinfo.value)


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


#: A lattice with the shape the copy cares about: valid JSON, and
#: byte-identical on the other side. Nothing in the generator parses it, so the
#: smallest well-formed document is the whole requirement. The bindings beside
#: it are parsed -- they are the partition -- so they are a real document.
_LATTICE_TEXT = '{"name": "test ring", "elements": []}\n'

#: The digest a hand-written document stamps. The manifest generator never
#: opens the lattice, so what it holds only has to be a digest's shape.
_LATTICE_DIGEST = "0" * 64


def _linear(gain: float) -> dict:
    return {"kind": "linear", "gain": gain, "offset": 0.0}


def _strength(setpoint: str, readback: str | None, *, element: str = "Q1") -> dict:
    """A written binding: a magnet current onto one element's ``PolynomB[1]``.

    ``readback`` of ``None`` is the family that serves setpoint and readback on
    one address, which the schema spells ``same_as_setpoint``.
    """
    return {
        "kind": "strength",
        "family": "quad_a",
        "setpoint_address": setpoint,
        "readback_address": readback,
        "readback": "identity" if readback is not None else "same_as_setpoint",
        "element": element,
        "attribute": "PolynomB",
        "index": 1,
        "slices": [{"element": element, "weight": 1.0}],
        "owner": "quad_a",
        "calibration": _linear(0.01),
        "monitor_inverse": None,
        "nominal": 100.0,
        "energy_scaling": "brho",
        "energy_table": None,
    }


def _monitor(address: str, *, axis: str = "x", element: str = "BPM1") -> dict:
    """A read binding: one transverse axis of one orbit monitor."""
    return {
        "kind": "monitor",
        "family": "mon_a",
        "setpoint_address": address,
        "readback_address": None,
        "readback": "inverse",
        "element": element,
        "attribute": axis,
        "index": None,
        "slices": [{"element": element, "weight": 1.0}],
        "owner": "mon_a",
        "calibration": _linear(1.0e-3),
        "monitor_inverse": _linear(1.0e3),
        "nominal": None,
        "energy_scaling": "none",
        "energy_table": None,
    }


def _bindings_text(*bindings: dict) -> str:
    """A whole bindings document carrying *bindings*, as the emit lane writes one."""
    document = {
        "system": "StorageRing",
        "energy_gev": 2.0,
        "lattice_sha256": _LATTICE_DIGEST,
        "bindings": list(bindings),
    }
    return json.dumps(document, indent=2) + "\n"


#: The document a tree gets when the test only cares that it carries one: a
#: valid document binding nothing, so no address is coupled by it.
_BINDINGS_TEXT = _bindings_text()


def _with_simulation_model(
    tree: Path, *, lattice: bool = True, bindings: bool = True, document: str | None = None
) -> Path:
    """Give *tree* the files that say it serves a virtual accelerator."""
    paths = ManifestPaths(data_root=tree, tier=DEFAULT_TIER)
    paths.lattice_json.parent.mkdir(parents=True, exist_ok=True)
    if lattice:
        paths.lattice_json.write_text(_LATTICE_TEXT)
    if bindings:
        paths.va_bindings.write_text(_BINDINGS_TEXT if document is None else document)
    return tree


class TestSimulationModelSources:
    """The lattice and the bindings travel with the manifest that names them.

    A container is handed one directory, so a model left behind in the source
    tree is a model the virtual accelerator cannot load. The copies are
    byte-for-byte, which is what keeps the lattice digest recorded at emit time
    equal to the digest of the file the container reads.
    """

    def test_a_tree_carrying_lattice_and_bindings_backs_a_manifest(self, editable_tree):
        _with_simulation_model(editable_tree)
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)

        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared is not None
        assert prepared.model_sources == (paths.lattice_json, paths.va_bindings)

    def test_bindings_without_a_lattice_refuse_the_build(self, editable_tree):
        _with_simulation_model(editable_tree, lattice=False)

        assert prepare_project_manifest(editable_tree, DEFAULT_TIER) is None
        assert "simulation/lattice.json" in manifest_gap_reason(editable_tree, DEFAULT_TIER)

    def test_a_lattice_nothing_is_bound_to_is_not_a_source(self, editable_tree):
        """Without bindings no channel reaches the ring, so none of it ships."""
        _with_simulation_model(editable_tree, bindings=False)

        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared is not None
        assert prepared.model_sources == ()

    def test_lattice_and_bindings_land_beside_the_manifest(self, editable_tree, tmp_path):
        _with_simulation_model(editable_tree)
        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)
        project_data = tmp_path / "project" / "data"
        project_data.mkdir(parents=True)

        manifest_path = write_project_manifest(prepared, project_data)

        built = ManifestPaths(data_root=project_data, tier=DEFAULT_TIER)
        assert built.lattice_json.parent == manifest_path.parent
        assert built.lattice_json.is_file()
        assert built.va_bindings.is_file()

    def test_the_lattice_and_bindings_copies_are_byte_identical(self, editable_tree, tmp_path):
        _with_simulation_model(editable_tree)
        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)
        project_data = tmp_path / "project" / "data"
        project_data.mkdir(parents=True)

        write_project_manifest(prepared, project_data)

        built = ManifestPaths(data_root=project_data, tier=DEFAULT_TIER)
        source = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)
        assert built.lattice_json.read_bytes() == source.lattice_json.read_bytes()
        assert built.va_bindings.read_bytes() == source.va_bindings.read_bytes()

    def test_a_tree_serving_no_lattice_copies_neither_file(self, facility_tree, tmp_path):
        prepared = prepare_project_manifest(facility_tree, DEFAULT_TIER)
        project_data = tmp_path / "data"
        project_data.mkdir()

        write_project_manifest(prepared, project_data)

        built = ManifestPaths(data_root=project_data, tier=DEFAULT_TIER)
        assert not built.lattice_json.exists()
        assert not built.va_bindings.exists()

    def test_writing_into_the_source_tree_leaves_the_lattice_alone(self, editable_tree):
        """Destination and source are one file when a tree is built in place."""
        _with_simulation_model(editable_tree)
        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)
        paths = ManifestPaths(data_root=editable_tree, tier=DEFAULT_TIER)

        write_project_manifest(prepared, editable_tree)

        assert paths.lattice_json.read_text() == _LATTICE_TEXT
        assert paths.va_bindings.read_text() == _BINDINGS_TEXT


# --- the partition rule ------------------------------------------------------


def _by_address(manifest: dict) -> dict[str, dict]:
    return {channel["address"]: channel for channel in manifest["channels"]}


def _a_stated_pair(manifest: dict) -> tuple[str, str]:
    """A setpoint and its readback, as the unedited tree itself states them.

    Read off the manifest rather than spelled out, so nothing here carries one
    facility's address vocabulary: whichever device field the bundled tree
    happens to state first is as good as any other for the rule under test.
    """
    halves: dict[tuple[str, ...], dict[str, str]] = {}
    for channel in manifest["channels"]:
        key = (
            channel["ring"],
            channel["system"],
            channel["family"],
            channel["device"],
            channel["field"],
        )
        halves.setdefault(key, {})[channel["subfield"]] = channel["address"]
    for group in sorted(halves.values(), key=lambda g: sorted(g.values())):
        if "SP" in group and "RB" in group:
            return group["SP"], group["RB"]
    raise AssertionError("the bundled tree states no setpoint/readback pair")


def _an_unpaired_analog(manifest: dict) -> str:
    """An address the unedited tree serves static-noisy, for a monitor binding."""
    for channel in sorted(manifest["channels"], key=lambda c: c["address"]):
        if channel["partition"] == classify.PARTITION_STATIC_NOISY and channel["record_type"] == (
            classify.RECORD_TYPE_ANALOG
        ):
            return channel["address"]
    raise AssertionError("the bundled tree serves no static-noisy analog channel")


class TestTheBindingsAreThePyatCoupledPartition:
    """What a write steers the beam with is what the tree's bindings say it is.

    No address text, no hierarchy token and no family list decides this
    partition, so the same generator serves a facility whose magnets are
    called anything at all.
    """

    def test_a_tree_carrying_no_bindings_couples_nothing(self, facility_tree):
        prepared = prepare_project_manifest(facility_tree, DEFAULT_TIER)

        metadata = prepared.manifest["_metadata"]
        assert classify.PARTITION_PYAT_COUPLED not in metadata["by_partition"]
        assert metadata["partition_source"] == "none"

    def test_a_bound_setpoint_and_its_readback_become_the_coupled_pair(self, editable_tree):
        baseline = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest
        setpoint, readback = _a_stated_pair(baseline)
        _with_simulation_model(
            editable_tree, document=_bindings_text(_strength(setpoint, readback))
        )

        served = _by_address(prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest)

        assert served[setpoint]["partition"] == classify.PARTITION_PYAT_COUPLED
        assert served[readback]["partition"] == classify.PARTITION_PYAT_COUPLED
        assert served[setpoint]["subfield"] == "SP"
        assert served[readback]["subfield"] == "RB"

    def test_the_pair_is_keyed_on_the_setpoint_address_alone(self, editable_tree):
        """The bindings pair the two halves, so the bindings supply the key.

        A hierarchy path cannot be trusted to agree with a document that is
        free to serve a readback anywhere in the namespace, so the identity
        keys are emptied and the setpoint's own address carries the pair --
        the key ``serving/pvdb`` matches the two records on.
        """
        baseline = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest
        setpoint, readback = _a_stated_pair(baseline)
        _with_simulation_model(
            editable_tree, document=_bindings_text(_strength(setpoint, readback))
        )

        served = _by_address(prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest)

        for half in (served[setpoint], served[readback]):
            assert half["device"] == setpoint
            assert half["ring"] == half["system"] == half["family"] == half["field"] == ""
            assert half["record_type"] == classify.RECORD_TYPE_ANALOG

    def test_the_coupled_setpoints_are_exactly_the_documents_own(self, editable_tree):
        """The equality the deployed model is built against."""
        from osprey.services.virtual_accelerator.bindings import parse_bindings, setpoints

        baseline = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest
        setpoint, readback = _a_stated_pair(baseline)
        monitor = _an_unpaired_analog(baseline)
        text = _bindings_text(_strength(setpoint, readback), _monitor(monitor))
        _with_simulation_model(editable_tree, document=text)

        manifest = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest

        document = parse_bindings(json.loads(text))
        assert classify.pyat_coupled_setpoint_addresses(manifest["channels"]) == set(
            setpoints(document)
        )

    def test_a_monitor_is_served_under_the_axis_it_reads(self, editable_tree):
        baseline = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest
        monitor = _an_unpaired_analog(baseline)
        _with_simulation_model(editable_tree, document=_bindings_text(_monitor(monitor, axis="y")))

        served = _by_address(prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest)

        assert served[monitor]["partition"] == classify.PARTITION_PYAT_COUPLED
        assert served[monitor]["subfield"] == "Y"
        assert served[monitor]["device"] == monitor
        # A monitor is read, never written: it is not one of the setpoints.
        assert monitor not in classify.setpoint_addresses(
            prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest["channels"]
        )

    def test_one_address_carrying_both_halves_emits_one_setpoint(self, editable_tree):
        """``same_as_setpoint``: there is no second channel, so there is no RB."""
        baseline = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest
        setpoint, readback = _a_stated_pair(baseline)
        _with_simulation_model(editable_tree, document=_bindings_text(_strength(setpoint, None)))

        served = _by_address(prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest)

        assert served[setpoint]["subfield"] == "SP"
        assert served[setpoint]["partition"] == classify.PARTITION_PYAT_COUPLED
        # The tree's own readback channel is still served, just not coupled.
        assert served[readback]["partition"] == classify.PARTITION_STATIC_NOISY

    def test_a_bound_pair_stops_being_an_echo_pair(self, editable_tree):
        """The partitions are exclusive: an address coupled is not echoed."""
        baseline = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest
        setpoint, readback = _a_stated_pair(baseline)
        assert _by_address(baseline)[setpoint]["partition"] == classify.PARTITION_SP_ECHO
        _with_simulation_model(
            editable_tree, document=_bindings_text(_strength(setpoint, readback))
        )

        manifest = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest

        counts = manifest["_metadata"]["by_partition"]
        assert counts[classify.PARTITION_PYAT_COUPLED] == 2
        assert counts[classify.PARTITION_SP_ECHO] == (
            baseline["_metadata"]["by_partition"][classify.PARTITION_SP_ECHO] - 2
        )

    def test_metadata_names_the_document_that_claimed_the_partition(self, editable_tree):
        _with_simulation_model(editable_tree)

        prepared = prepare_project_manifest(editable_tree, DEFAULT_TIER)

        assert prepared.manifest["_metadata"]["partition_source"] == "simulation/va_bindings.json"

    def test_an_address_no_database_named_still_arrives_from_the_bindings(self, editable_tree):
        """The coupled entries ARE the document, whatever the databases listed.

        Normally nothing to report -- one emit run writes both files -- so the
        discrepancy is recorded rather than swallowed.
        """
        novel = "ZZRING:ZZSYS:ZZFAM:99:ZZFIELD:SP"
        _with_simulation_model(editable_tree, document=_bindings_text(_strength(novel, None)))

        manifest = prepare_project_manifest(editable_tree, DEFAULT_TIER).manifest

        served = _by_address(manifest)
        assert served[novel]["partition"] == classify.PARTITION_PYAT_COUPLED
        assert manifest["_metadata"]["bindings_novel_addresses"] == [novel]

    def test_a_refused_document_stops_the_manifest_by_name(self, editable_tree):
        from osprey.services.virtual_accelerator.bindings import BindingsError

        _with_simulation_model(editable_tree, document='{"system": "SR"}\n')

        with pytest.raises(BindingsError, match="va_bindings.json"):
            prepare_project_manifest(editable_tree, DEFAULT_TIER)


class TestSetpointEchoPairsFromAHierarchyPath:
    """Without bindings, a pair is the SP/RB halves of one device field."""

    def test_every_echo_channel_is_one_half_of_a_stated_pair(self, facility_tree):
        manifest = prepare_project_manifest(facility_tree, DEFAULT_TIER).manifest

        halves: dict[tuple, set[str]] = {}
        for channel in manifest["channels"]:
            if channel["partition"] != classify.PARTITION_SP_ECHO:
                continue
            assert channel["subfield"] in ("SP", "RB")
            key = (
                channel["ring"],
                channel["system"],
                channel["family"],
                channel["device"],
                channel["field"],
            )
            halves.setdefault(key, set()).add(channel["subfield"])
        assert halves, "the bundled tree states setpoint/readback pairs"
        assert all(group == {"SP", "RB"} for group in halves.values())

    def test_a_setpoint_is_echoed_only_where_the_tree_states_a_readback(self, facility_tree):
        """Every writable channel of this tree is paired; none is left dangling.

        The serving layer refuses an echo setpoint with nothing to echo into,
        so the generator must never make one.
        """
        manifest = prepare_project_manifest(facility_tree, DEFAULT_TIER).manifest

        echoes = {
            channel["address"]
            for channel in manifest["channels"]
            if channel["partition"] == classify.PARTITION_SP_ECHO
        }
        setpoints = classify.setpoint_addresses(manifest["channels"])
        assert setpoints <= echoes

    def test_a_status_flag_is_not_a_readback(self, facility_tree):
        """It shares no subfield vocabulary with a setpoint, so it pairs with none."""
        manifest = prepare_project_manifest(facility_tree, DEFAULT_TIER).manifest

        flags = [
            channel
            for channel in manifest["channels"]
            if channel["record_type"] == classify.RECORD_TYPE_BINARY
        ]
        assert flags, "the bundled tree carries status flags"
        assert all(channel["partition"] == classify.PARTITION_STATIC_NOISY for channel in flags)


# --- a tree with no hierarchy path -------------------------------------------

#: A middle-layer database in the shape a real MML export has: one list of
#: addresses per (family, field), the field's direction stated by its
#: ``MemberOf`` tags, and its quantity by ``HWUnits``. The second setpoint
#: group (``SPKL``) is NSLS-II's shape: a magnet states its kick angle beside
#: its current, so a family alone does not say which readback a setpoint
#: echoes into.
_MML_DB = {
    "ZZSR": {
        "QUADA": {
            "setup": {"DeviceList": [[1, 1], [1, 2]], "CommonNames": ["Q1", "Q2"]},
            "Setpoint": {
                "MemberOf": ["Magnet", "Setpoint"],
                "HWUnits": "A",
                "ChannelNames": ["ZZ-QUADA{1}Cur:Sp", "ZZ-QUADA{2}Cur:Sp"],
            },
            "Monitor": {
                "MemberOf": ["Magnet", "Monitor"],
                "HWUnits": "A",
                "ChannelNames": ["ZZ-QUADA{1}Cur:Am", "ZZ-QUADA{2}Cur:Am"],
            },
            "SPKL": {
                "MemberOf": ["Magnet", "Setpoint"],
                "HWUnits": "T*m^-1",
                "ChannelNames": ["ZZ-QUADA{1}K:Sp", "ZZ-QUADA{2}K:Sp"],
            },
            "RBKL": {
                "MemberOf": ["Magnet", "Readback"],
                "HWUnits": "T*m^-1",
                "ChannelNames": ["ZZ-QUADA{1}K:Rb", "ZZ-QUADA{2}K:Rb"],
            },
        },
        "SWITCHA": {
            "setup": {"DeviceList": [[1, 1], [1, 2]], "CommonNames": ["S1", "S2"]},
            "OnControl": {
                "MemberOf": ["Magnet", "Control"],
                "HWUnits": "",
                "ChannelNames": ["ZZ-SWITCHA{1}On:Sp", "ZZ-SWITCHA{2}On:Sp"],
            },
            "Fault": {
                "MemberOf": ["Magnet", "Monitor"],
                "HWUnits": "",
                "ChannelNames": ["ZZ-SWITCHA{1}Flt:Am", "ZZ-SWITCHA{2}Flt:Am"],
            },
        },
        "AMBIGA": {
            "setup": {"DeviceList": [[1, 1], [1, 2]], "CommonNames": ["A1", "A2"]},
            "Setpoint": {
                "MemberOf": ["Magnet", "Setpoint"],
                "HWUnits": "A",
                "ChannelNames": ["ZZ-AMBIGA{1}Cur:Sp", "ZZ-AMBIGA{2}Cur:Sp"],
            },
            "Monitor": {
                "MemberOf": ["Magnet", "Monitor"],
                "HWUnits": "A",
                "ChannelNames": ["ZZ-AMBIGA{1}Cur:Am", "ZZ-AMBIGA{2}Cur:Am"],
            },
            "Readback": {
                "MemberOf": ["Magnet", "Monitor"],
                "HWUnits": "A",
                "ChannelNames": ["ZZ-AMBIGA{1}Cur:Rb", "ZZ-AMBIGA{2}Cur:Rb"],
            },
        },
        "SHIFTA": {
            "setup": {"DeviceList": [[1, 1], [1, 2], [1, 3]], "CommonNames": ["H1", "H2", "H3"]},
            "Setpoint": {
                "MemberOf": ["Magnet", "Setpoint"],
                "HWUnits": "A",
                "ChannelNames": ["", "ZZ-SHIFTA{2}Cur:Sp", "ZZ-SHIFTA{3}Cur:Sp"],
            },
            "Monitor": {
                "MemberOf": ["Magnet", "Monitor"],
                "HWUnits": "A",
                "ChannelNames": [
                    "ZZ-SHIFTA{1}Cur:Am",
                    "ZZ-SHIFTA{2}Cur:Am",
                    "ZZ-SHIFTA{1}Cur:Am",
                ],
            },
        },
        "MONA": {
            "setup": {"DeviceList": [[1, 1]], "CommonNames": ["M1"]},
            "Monitor": {
                "MemberOf": ["Monitor"],
                "HWUnits": "mm",
                "ChannelNames": ["ZZ-MONA{1}Pos:X"],
            },
        },
    }
}

_MML_SETPOINTS = ("ZZ-QUADA{1}Cur:Sp", "ZZ-QUADA{2}Cur:Sp")
_MML_READBACKS = ("ZZ-QUADA{1}Cur:Am", "ZZ-QUADA{2}Cur:Am")
_MML_KICK_SETPOINT = "ZZ-QUADA{1}K:Sp"
_MML_MONITOR = "ZZ-MONA{1}Pos:X"
#: The two halves of a family that states no hardware units on either.
_MML_UNITLESS_WRITE = ("ZZ-SWITCHA{1}On:Sp", "ZZ-SWITCHA{2}On:Sp")
_MML_UNITLESS_READ = ("ZZ-SWITCHA{1}Flt:Am", "ZZ-SWITCHA{2}Flt:Am")
#: A setpoint two read-voted groups of the same quantity and count could answer.
_MML_TWICE_ANSWERED_SETPOINT = "ZZ-AMBIGA{1}Cur:Sp"
_MML_TWICE_ANSWERED_READS = ("ZZ-AMBIGA{1}Cur:Am", "ZZ-AMBIGA{1}Cur:Rb")
#: A family whose setpoint list is blank at its first device and whose monitor
#: list repeats that device's address at its third: equal counts, places that
#: no longer line up, and exactly one device both fields state.
_MML_SHIFTED_PAIR = ("ZZ-SHIFTA{2}Cur:Sp", "ZZ-SHIFTA{2}Cur:Am")
_MML_SHIFTED_LONE_SETPOINT = "ZZ-SHIFTA{3}Cur:Sp"
_MML_SHIFTED_LONE_MONITOR = "ZZ-SHIFTA{1}Cur:Am"


@pytest.fixture(scope="module")
def middle_layer_tree(tmp_path_factory) -> Path:
    """A data tree staging a middle-layer database and nothing else.

    The shape every MML-emitted tree has: no hierarchical database, so no
    channel carries a hierarchy path and every partition, record type and
    noise flag is read off the middle-layer database's own signal groups.
    """
    root = tmp_path_factory.mktemp("mml_data")
    paths = ManifestPaths(data_root=root, tier=DEFAULT_TIER)
    paths.middle_layer_db.parent.mkdir(parents=True, exist_ok=True)
    paths.middle_layer_db.write_text(json.dumps(_MML_DB))
    paths.machine_json.parent.mkdir(parents=True, exist_ok=True)
    paths.machine_json.write_text(json.dumps({"name": "zz", "channels": {}}))
    paths.machine_state_channels.write_text("{}")
    paths.channel_limits.write_text(json.dumps({"_version": "1.0"}))
    return root


class TestPartitionsWithoutAHierarchyPath:
    """A middle-layer database states its pairs by signal group, not by path."""

    def test_a_write_voted_group_pairs_with_its_read_voted_sibling(self, middle_layer_tree):
        served = _by_address(prepare_project_manifest(middle_layer_tree, DEFAULT_TIER).manifest)

        for setpoint, readback in zip(_MML_SETPOINTS, _MML_READBACKS, strict=True):
            assert served[setpoint]["partition"] == classify.PARTITION_SP_ECHO
            assert served[readback]["partition"] == classify.PARTITION_SP_ECHO
            assert served[setpoint]["subfield"] == "SP"
            assert served[readback]["subfield"] == "RB"
            assert served[setpoint]["device"] == served[readback]["device"] == setpoint

    def test_a_setpoint_measuring_another_quantity_is_not_paired(self, middle_layer_tree):
        """Same family, different units: not the readback of this setpoint.

        Echoing a current onto a kick-angle readback would publish a reading
        the facility never claimed, so the group is left unpaired instead.
        """
        served = _by_address(prepare_project_manifest(middle_layer_tree, DEFAULT_TIER).manifest)

        assert served[_MML_KICK_SETPOINT]["partition"] == classify.PARTITION_STATIC_NOISY

    def test_a_group_stating_no_units_pairs_with_nothing(self, middle_layer_tree):
        """Two groups that state no units do not thereby measure the same thing.

        An absent unit is the absence of a statement, so a write-voted control
        and a read-voted flag stay apart: echoing the write onto the flag
        would publish a reading the facility never claimed.
        """
        served = _by_address(prepare_project_manifest(middle_layer_tree, DEFAULT_TIER).manifest)

        for address in _MML_UNITLESS_WRITE + _MML_UNITLESS_READ:
            assert served[address]["partition"] == classify.PARTITION_STATIC_NOISY
            assert served[address]["subfield"] == ""

    def test_a_setpoint_two_read_groups_could_answer_is_not_paired(self, middle_layer_tree):
        """Two candidates of one quantity and count identify neither as the pair."""
        served = _by_address(prepare_project_manifest(middle_layer_tree, DEFAULT_TIER).manifest)

        assert served[_MML_TWICE_ANSWERED_SETPOINT]["partition"] == classify.PARTITION_STATIC_NOISY
        for address in _MML_TWICE_ANSWERED_READS:
            assert served[address]["partition"] == classify.PARTITION_STATIC_NOISY

    def test_a_pair_is_the_device_both_fields_state(self, middle_layer_tree):
        """The device position pairs the halves, not their place in the list.

        A field that leaves a device blank and a sibling that repeats one
        device's address keep the same number of addresses while their places
        no longer line up; only the device both fields state is a pair, and
        the halves that stand alone are served unpaired.
        """
        setpoint, readback = _MML_SHIFTED_PAIR
        served = _by_address(prepare_project_manifest(middle_layer_tree, DEFAULT_TIER).manifest)

        assert served[setpoint]["partition"] == classify.PARTITION_SP_ECHO
        assert served[readback]["partition"] == classify.PARTITION_SP_ECHO
        assert served[setpoint]["device"] == served[readback]["device"] == setpoint
        assert served[_MML_SHIFTED_LONE_SETPOINT]["partition"] == classify.PARTITION_STATIC_NOISY
        assert served[_MML_SHIFTED_LONE_MONITOR]["partition"] == classify.PARTITION_STATIC_NOISY

    def test_every_channel_is_analog(self, middle_layer_tree):
        """No hierarchy path means no record-type grammar to read one off."""
        manifest = prepare_project_manifest(middle_layer_tree, DEFAULT_TIER).manifest

        assert {c["record_type"] for c in manifest["channels"]} == {classify.RECORD_TYPE_ANALOG}

    def test_noise_is_the_address_read_vote(self, middle_layer_tree):
        """A measured address jitters; one that reports a write does not."""
        served = _by_address(prepare_project_manifest(middle_layer_tree, DEFAULT_TIER).manifest)

        assert served[_MML_MONITOR]["noise"] is True
        assert served[_MML_READBACKS[0]]["noise"] is True
        assert served[_MML_SETPOINTS[0]]["noise"] is False
        assert served[_MML_KICK_SETPOINT]["noise"] is False

    def test_an_unpaired_monitor_is_static_noisy(self, middle_layer_tree):
        served = _by_address(prepare_project_manifest(middle_layer_tree, DEFAULT_TIER).manifest)

        assert served[_MML_MONITOR]["partition"] == classify.PARTITION_STATIC_NOISY
        assert served[_MML_MONITOR]["subfield"] == ""

    def test_a_bound_address_leaves_the_echo_pair_for_the_model(self, middle_layer_tree, tmp_path):
        """The bindings win over the database's own pairing, on every tree."""
        root = tmp_path / "bound"
        shutil.copytree(middle_layer_tree, root)
        _with_simulation_model(
            root,
            document=_bindings_text(_strength(_MML_SETPOINTS[0], _MML_READBACKS[0])),
        )

        served = _by_address(prepare_project_manifest(root, DEFAULT_TIER).manifest)

        assert served[_MML_SETPOINTS[0]]["partition"] == classify.PARTITION_PYAT_COUPLED
        assert served[_MML_READBACKS[0]]["partition"] == classify.PARTITION_PYAT_COUPLED
        # The sibling device is untouched: one binding claims one device.
        assert served[_MML_SETPOINTS[1]]["partition"] == classify.PARTITION_SP_ECHO


# --- the re-exported facility trees ------------------------------------------

#: Where the SERVED trees are looked for. A served tree is a data root
#: carrying ``simulation/va_bindings.json``; the lane below runs for every
#: such tree under the MML fixtures, and says so rather than passing on
#: nothing when the fixtures carry none.
_MML_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "mml"


def _served_fixture_trees() -> list[Path]:
    return sorted(
        path.parent.parent for path in _MML_FIXTURE_ROOT.rglob("simulation/va_bindings.json")
    )


@pytest.mark.skipif(
    not _served_fixture_trees(),
    reason="no data root under tests/fixtures/mml carries simulation/va_bindings.json",
)
@pytest.mark.parametrize(
    "tree", _served_fixture_trees() or [None], ids=lambda p: getattr(p, "name", "none")
)
class TestReExportedFacilityTrees:
    """The rule against the real thing, on whichever facilities ship a tree.

    Facility-agnostic by construction: every name here is read out of the tree
    under test, so adding a third facility's fixture adds a third case and no
    code.
    """

    def test_the_coupled_setpoints_are_the_documents_own(self, tree):
        from osprey.services.virtual_accelerator.bindings import load_bindings, setpoints

        paths = ManifestPaths(data_root=tree, tier=DEFAULT_TIER)
        prepared = prepare_project_manifest(tree, DEFAULT_TIER)

        assert prepared is not None
        document = load_bindings(paths.va_bindings)
        assert classify.pyat_coupled_setpoint_addresses(prepared.manifest["channels"]) == set(
            setpoints(document)
        )

    def test_every_entry_is_analog_and_noise_follows_the_read_vote(self, tree):
        manifest = prepare_project_manifest(tree, DEFAULT_TIER).manifest

        assert {c["record_type"] for c in manifest["channels"]} == {classify.RECORD_TYPE_ANALOG}
        for channel in manifest["channels"]:
            if channel["subfield"] == "SP":
                assert channel["noise"] is False, channel["address"]
            elif channel["subfield"] == "RB":
                assert channel["noise"] is True, channel["address"]

    def test_the_document_names_itself_as_the_partition_source(self, tree):
        manifest = prepare_project_manifest(tree, DEFAULT_TIER).manifest

        assert manifest["_metadata"]["partition_source"] == "simulation/va_bindings.json"


# --- the knowledge-graph source ---------------------------------------------

_TTL_PREAMBLE = """\
@prefix narad_p: <https://narad.example.org/property/> .
@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .
"""


def _binding(name: str, address: str, predicate: str | None) -> str:
    """Render one corpus channel binding with the given direction predicate."""
    direction = f" ;\n    narad_p:{predicate} narad_sem:{name}_signal" if predicate else ""
    return f'<https://narad.example.org/binding/{name}> narad_p:fullPv "{address}"{direction} .\n'


#: How a graph-mode project spells the search index the roster reads: the
#: ``services.graphdb.index_path`` default, which is what every absence about
#: it puts in front of an operator.
_INDEX_SPELLING = "./data/channel_databases/graph.duckdb"

#: Two settable channels, three readable ones -- both directions have to reach
#: the manifest, because membership is the roster's whole answer.
_SMALL_CORPUS = _TTL_PREAMBLE + "".join(
    (
        _binding("hcm_sp", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"),
        _binding("hcm_rb", "SR:MAG:HCM:01:CURRENT:RB", "readsSignal"),
        _binding("rf_sp", "SR:RF:CAV:01:VOLTAGE:SP", "writesSignal"),
        _binding("bpm_x", "SR:DIAG:BPM:01:POSITION:X", "readsSignal"),
        _binding("temp", "SR:VAC:PUMP:01:TEMPERATURE:RB", "readsSignal"),
    )
)


def _graph_tree(
    root: Path, corpus: str | None = _SMALL_CORPUS, *, index: bool = True
) -> tuple[Path, dict]:
    """A graph-mode facility tree: per-tree sources plus a corpus, no databases.

    The manifest is built from the channel roster, and on this paradigm the
    roster reads the search index a build derives from the corpus -- so the
    tree stages both, exactly as a rendered project holds both. ``index=False``
    stages the corpus alone, for the cases about a tree nothing derived.

    Returns the data root and the config a graph-mode render resolves the
    corpus from -- ``ttl_path`` spelled relative, as a project writes it.
    """
    (root / "simulation").mkdir(parents=True)
    (root / "simulation" / "machine.json").write_text(json.dumps({"channels": {}}))
    (root / "machine_state_channels.json").write_text(json.dumps({"_comment": "empty"}))
    (root / LIMITS_FILENAME).write_text("{}\n")
    config = {
        "channel_finder": {"pipeline_mode": "graph"},
        "services": {"graphdb": {"ttl_path": "./facility.ttl"}},
        "config_dir": str(root),
    }
    if corpus is not None:
        (root / "facility.ttl").write_text(corpus)
        if index:
            build_index_from_ttl(root / "facility.ttl", config)
    return root, config


@pytest.fixture(autouse=True)
def _cold_roster_cache():
    """Every test reads its own corpus cold; none inherits another's parse."""
    import osprey.channel_roster as channel_roster

    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


class TestGraphSourcedManifest:
    """A graph-mode tree gets its channel set from the knowledge-graph corpus."""

    def test_every_corpus_binding_becomes_a_channel_both_directions(self, tmp_path):
        """Membership is the corpus's fullPv set: writes and reads alike."""
        root, config = _graph_tree(tmp_path / "data")

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        assert prepared is not None
        addresses = [c["address"] for c in prepared.manifest["channels"]]
        assert addresses == sorted(
            [
                "SR:MAG:HCM:01:CURRENT:SP",
                "SR:MAG:HCM:01:CURRENT:RB",
                "SR:RF:CAV:01:VOLTAGE:SP",
                "SR:DIAG:BPM:01:POSITION:X",
                "SR:VAC:PUMP:01:TEMPERATURE:RB",
            ]
        )

    def test_metadata_names_the_corpus_as_the_source(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data")

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        metadata = prepared.manifest["_metadata"]
        assert metadata["source_paradigms"] == ["graph"]
        # The configured spelling, which an operator can retype and edit. The
        # roster reads the search index, so that is the file named here.
        assert metadata["source_corpus"] == _INDEX_SPELLING
        # Graph mode stages no tier database by design: nothing is "absent",
        # nothing is corrupt, and no reader is owed either clause.
        assert metadata["absent_paradigms"] == []
        assert metadata["corrupt_paradigms"] == []

    def test_census_is_honest_about_what_the_graph_cannot_say(self, tmp_path):
        """No hierarchy identity keys are invented, so nothing is pyat-coupled.

        The corpus states membership, direction and -- for the one pair the
        roster vouches for, ``HCM:01:CURRENT:SP``/``:RB`` -- a readback, but
        not the hierarchy path the identity keys are read from. The pair is
        served as setpoint-echo, keyed on nothing but itself; every other
        entry lands pathless in the static-noisy partition -- exactly what a
        database tree without its hierarchical paradigm gets -- and the
        setpoint census counts the pair rather than guessing further.
        """
        root, config = _graph_tree(tmp_path / "data")

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        metadata = prepared.manifest["_metadata"]
        assert metadata["by_partition"] == {
            classify.PARTITION_SP_ECHO: 2,
            classify.PARTITION_STATIC_NOISY: 3,
        }
        assert metadata["setpoint_count"] == 1
        for channel in prepared.manifest["channels"]:
            for key in ("ring", "system", "family", "field"):
                assert channel[key] == ""
            if channel["partition"] == classify.PARTITION_STATIC_NOISY:
                assert channel["device"] == "" and channel["subfield"] == ""
            else:
                assert channel["partition"] == classify.PARTITION_SP_ECHO
                assert channel["device"] == "SR:MAG:HCM:01:CURRENT:SP"
                assert channel["subfield"] == ("SP" if channel["address"].endswith(":SP") else "RB")

    def test_duplicate_addresses_in_the_corpus_collapse_to_one_channel(self, tmp_path):
        """The manifest is a namespace: two bindings sharing one fullPv are one channel."""
        corpus = _TTL_PREAMBLE + "".join(
            (
                _binding("first", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"),
                _binding("second", "SR:MAG:HCM:01:CURRENT:SP", "readsSignal"),
            )
        )
        root, config = _graph_tree(tmp_path / "data", corpus=corpus)

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        assert prepared.manifest["_metadata"]["total_channels"] == 1
        assert [c["address"] for c in prepared.manifest["channels"]] == ["SR:MAG:HCM:01:CURRENT:SP"]

    def test_scenario_seed_union_applies_to_the_graph_source_too(self, tmp_path):
        """machine.json's novel addresses ride along, flagged as such."""
        root, config = _graph_tree(tmp_path / "data")
        (root / "simulation" / "machine.json").write_text(
            json.dumps({"channels": {"SR:VAC:GAUGE:99:PRESSURE:RB": {"value": 1e-9}}})
        )

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        metadata = prepared.manifest["_metadata"]
        assert metadata["machine_json_novel_addresses"] == ["SR:VAC:GAUGE:99:PRESSURE:RB"]
        assert metadata["total_channels"] == 6

    def test_written_graph_manifest_loads_through_the_container_reader(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data")
        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)
        project_data = tmp_path / "project" / "data"
        project_data.mkdir(parents=True)

        manifest_path = write_project_manifest(prepared, project_data)

        channels = loaders.load_manifest_file(manifest_path)
        assert len(channels) == 5
        assert (project_data / "simulation" / LIMITS_FILENAME).is_file()

    def test_without_a_config_a_graph_tree_still_backs_no_manifest(self, tmp_path):
        """Existing callers pass no config and must keep today's answer."""
        root, _ = _graph_tree(tmp_path / "data")

        assert prepare_project_manifest(root, DEFAULT_TIER) is None

    def test_a_non_graph_config_keeps_the_paradigm_rules(self, tmp_path):
        """A database-mode project never has its manifest read off a corpus."""
        root, config = _graph_tree(tmp_path / "data")
        config["channel_finder"]["pipeline_mode"] = "hierarchical"

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert "no channel database is staged" in reason


def _device(name: str, *bindings: str) -> str:
    """Render one corpus device grouping the named bindings."""
    objects = ", ".join(f"<https://narad.example.org/binding/{b}>" for b in bindings)
    return f"<https://narad.example.org/device/{name}> narad_p:hasBinding {objects} .\n"


def _bound(name: str, address: str, predicate: str, field: str, device: str = "BEND:0") -> str:
    """A binding carrying the ``bindingId`` whose field token the device grouping pairs on."""
    return (
        f'<https://narad.example.org/binding/{name}> narad_p:fullPv "{address}" ;\n'
        f"    narad_p:{predicate} narad_sem:{name}_signal ;\n"
        f'    narad_p:bindingId "narad:binding:als:SR:{device}:{field}:val" .\n'
    )


#: A facility whose addresses carry no ``:SP``/``:RB`` grammar at all: the one
#: pair here is stated by the corpus's device grouping, and the two leftover
#: channels (a golden setpoint nobody reports, a beam-current monitor) are not.
_STATED_PAIR_CORPUS = _TTL_PREAMBLE + "".join(
    (
        _bound("bend_sp", "SR01C___B______AC00", "writesSignal", "Setpoint"),
        _bound("bend_mon", "SR01C___B______AM00", "readsSignal", "Monitor"),
        _bound("bend_golden", "SR01C:BEND:Setpoint:Golden", "writesSignal", "SetpointGolden"),
        _device("bend", "bend_sp", "bend_mon", "bend_golden"),
        _bound("dcct", "SR01C___T______AM00", "readsSignal", "Monitor", device="DCCT:0"),
        _device("dcct", "dcct"),
    )
)


class TestGraphStatedPairs:
    """A pair the corpus states is served as a setpoint-echo pair, nothing more invented."""

    def test_a_stated_pair_becomes_an_sp_echo_pair_keyed_on_the_setpoint(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data", corpus=_STATED_PAIR_CORPUS)

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        by_address = {c["address"]: c for c in prepared.manifest["channels"]}
        setpoint = by_address["SR01C___B______AC00"]
        readback = by_address["SR01C___B______AM00"]
        assert setpoint["partition"] == readback["partition"] == classify.PARTITION_SP_ECHO
        assert setpoint["subfield"] == "SP"
        assert readback["subfield"] == "RB"
        # The pair shares exactly one identity key -- the setpoint's own
        # address -- and the other four stay as empty as on a pathless entry:
        # the graph states no hierarchy path, and none is invented.
        for channel in (setpoint, readback):
            assert channel["device"] == "SR01C___B______AC00"
            assert [channel[key] for key in ("ring", "system", "family", "field")] == [
                "",
                "",
                "",
                "",
            ]
        assert setpoint["record_type"] == readback["record_type"] == classify.RECORD_TYPE_ANALOG
        assert setpoint["noise"] is False and readback["noise"] is False

    def test_everything_the_corpus_leaves_unpaired_stays_pathless_static_noisy(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data", corpus=_STATED_PAIR_CORPUS)

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        by_address = {c["address"]: c for c in prepared.manifest["channels"]}
        for address in ("SR01C:BEND:Setpoint:Golden", "SR01C___T______AM00"):
            channel = by_address[address]
            assert channel["partition"] == classify.PARTITION_STATIC_NOISY
            for key in ("ring", "system", "family", "device", "field", "subfield"):
                assert channel[key] == ""
        metadata = prepared.manifest["_metadata"]
        assert metadata["by_partition"] == {
            classify.PARTITION_SP_ECHO: 2,
            classify.PARTITION_STATIC_NOISY: 2,
        }
        assert metadata["setpoint_count"] == 1
        assert metadata["total_channels"] == 4

    def test_the_written_manifest_loads_and_the_container_pairs_the_echo(self, tmp_path):
        """The shape the IOC reads: through the file loader, then the pvdb pairing.

        ``build_serving_pvdb`` is what pairs the two halves on their identity
        keys, and it is the one consumer that refuses an echo setpoint with
        nothing to echo into -- so it is the contract this manifest has to
        satisfy, proven here on the build host rather than at container boot.
        """
        from osprey.services.virtual_accelerator.serving.pvdb import build_serving_pvdb

        root, config = _graph_tree(tmp_path / "data", corpus=_STATED_PAIR_CORPUS)
        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)
        project_data = tmp_path / "project" / "data"
        project_data.mkdir(parents=True)

        manifest_path = write_project_manifest(prepared, project_data)

        channels = loaders.load_manifest_file(manifest_path)
        assert len(channels) == 4
        records = build_serving_pvdb(channels)
        assert records.setpoint_readbacks == {"SR01C___B______AC00": "SR01C___B______AM00"}
        assert set(records.static_noisy) == {"SR01C:BEND:Setpoint:Golden", "SR01C___T______AM00"}

    def test_a_readback_two_setpoints_claim_pairs_with_neither(self, tmp_path):
        """An ambiguous pair is served static-noisy on both sides, never half an echo."""
        from osprey.services.virtual_accelerator.serving.pvdb import build_serving_pvdb

        corpus = _TTL_PREAMBLE + "".join(
            (
                _bound("sp_a", "SR01C___B______AC00", "writesSignal", "Setpoint"),
                _bound("sp_b", "SR01C___B______AC01", "writesSignal", "Setpoint", device="BEND:1"),
                _bound("mon", "SR01C___B______AM00", "readsSignal", "Monitor"),
                _device("bend0", "sp_a", "mon"),
                _bound("mon_dup", "SR01C___B______AM00", "readsSignal", "Monitor", device="BEND:1"),
                _device("bend1", "sp_b", "mon_dup"),
            )
        )
        root, config = _graph_tree(tmp_path / "data", corpus=corpus)

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        metadata = prepared.manifest["_metadata"]
        assert metadata["by_partition"] == {classify.PARTITION_STATIC_NOISY: 3}
        assert metadata["setpoint_count"] == 0
        # And the IOC's contract holds on it: no echo setpoint left without a readback.
        assert build_serving_pvdb(prepared.manifest["channels"]).setpoint_readbacks == {}

    def test_a_readback_that_is_itself_a_setpoint_pairs_with_nothing(self, tmp_path):
        """A chain ``A -> B -> C`` states no clean pair; both links are dropped."""
        corpus = _TTL_PREAMBLE + "".join(
            (
                _bound("a", "SR:A", "writesSignal", "Setpoint"),
                _bound("b_mon", "SR:B", "readsSignal", "Monitor"),
                _device("dev_ab", "a", "b_mon"),
                _bound("b_sp", "SR:B", "writesSignal", "Setpoint", device="BEND:1"),
                _bound("c", "SR:C", "readsSignal", "Monitor", device="BEND:1"),
                _device("dev_bc", "b_sp", "c"),
            )
        )
        root, config = _graph_tree(tmp_path / "data", corpus=corpus)

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        metadata = prepared.manifest["_metadata"]
        assert metadata["by_partition"] == {classify.PARTITION_STATIC_NOISY: 3}
        assert metadata["setpoint_count"] == 0

    def test_a_readback_is_emitted_once_beside_its_setpoint(self, tmp_path):
        """The manifest is a namespace: the readback half is one channel, not two."""
        root, config = _graph_tree(tmp_path / "data", corpus=_STATED_PAIR_CORPUS)

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        addresses = [c["address"] for c in prepared.manifest["channels"]]
        assert addresses == sorted(addresses)
        assert len(set(addresses)) == len(addresses) == 4


class TestStagedDatabasesWinOverTheGraph:
    def test_a_staged_paradigm_database_keeps_priority(self, tmp_path, facility_tree):
        """The graph is consulted only when the tree stages no database at all."""
        root, config = _graph_tree(tmp_path / "data")
        db = root / f"channel_databases/tiers/tier{DEFAULT_TIER}/hierarchical.json"
        db.parent.mkdir(parents=True)
        shutil.copy2(
            facility_tree / f"channel_databases/tiers/tier{DEFAULT_TIER}/hierarchical.json", db
        )

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        metadata = prepared.manifest["_metadata"]
        assert metadata["source_paradigms"] == ["hierarchical"]
        assert "source_corpus" not in metadata

    def test_a_database_manifest_never_carries_a_corpus_key(self, facility_tree):
        prepared = prepare_project_manifest(facility_tree, DEFAULT_TIER)

        assert "source_corpus" not in prepared.manifest["_metadata"]


class TestGraphYieldsNothing:
    """The refusal names the corpus, never the absent database files."""

    def test_an_unbuilt_index_backs_no_manifest_and_is_named(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data", corpus=None)

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert _INDEX_SPELLING in reason
        assert "is not there" in reason
        # The remedy is the one that puts the file there, not a hunt for it.
        assert "osprey knowledge build-index" in reason
        assert "are all absent" not in reason

    def test_an_unreadable_index_backs_no_manifest_and_is_named(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data", index=False)
        index_path = default_index_path(root)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_bytes(b"not a database at all {{{")

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert _INDEX_SPELLING in reason
        assert "could not be read" in reason
        # Never conflated with the paradigm wordings: the operator repairs the
        # corpus, not database files that were never part of graph mode.
        assert "are all absent" not in reason
        assert "channel database" not in reason

    def test_an_empty_corpus_backs_no_manifest_and_is_named(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data", corpus=_TTL_PREAMBLE)

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert _INDEX_SPELLING in reason
        assert "declares no channels" in reason

    def test_a_graph_tree_missing_its_scenario_seed_is_named(self, tmp_path):
        """The corpus enumerates channels, but the per-tree sources still ship."""
        root, config = _graph_tree(tmp_path / "data")
        (root / "simulation" / "machine.json").unlink()

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert "missing simulation/machine.json" in reason

    def test_a_graph_tree_missing_its_drive_limits_is_named(self, tmp_path):
        """Limits and manifest ship together on the graph path too."""
        root, config = _graph_tree(tmp_path / "data")
        (root / LIMITS_FILENAME).unlink()

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert f"missing {LIMITS_FILENAME}" in reason

    def test_a_graph_tree_missing_its_machine_state_list_is_named(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data")
        (root / "machine_state_channels.json").unlink()

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert "missing machine_state_channels.json" in reason

    def test_graph_mode_naming_no_corpus_key_is_named_by_its_keys(self, tmp_path):
        """Graph mode with no ttl_path at all: the remedy is the config keys."""
        root, _ = _graph_tree(tmp_path / "data")
        config = {"channel_finder": {"pipeline_mode": "graph"}, "config_dir": str(root)}

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert "services.graphdb.ttl_path" in reason
        assert "services.graphdb.uri" in reason
        assert "are all absent" not in reason

    def test_an_unreadable_scenario_seed_raises_naming_the_file(self, tmp_path):
        """Same rule as the paradigm path: a broken per-tree source stops the build."""
        from osprey.errors import BuildProfileError

        root, config = _graph_tree(tmp_path / "data")
        (root / "simulation" / "machine.json").write_text("not json {")

        with pytest.raises(BuildProfileError) as excinfo:
            prepare_project_manifest(root, DEFAULT_TIER, config=config)

        assert "machine.json" in str(excinfo.value)
