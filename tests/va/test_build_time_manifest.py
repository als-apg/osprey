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

#: A body no parser can get past at all. The schema-invalid one above is
#: rejected by the flat and hierarchical parsers but reads as an empty tree to
#: the middle-layer one, which navigates any nesting it is given -- so a test
#: that needs EVERY staged database to be unreadable truncates the JSON
#: instead.
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


# --- the knowledge-graph source ---------------------------------------------

_TTL_PREAMBLE = """\
@prefix narad_p: <https://narad.example.org/property/> .
@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .
"""


def _binding(name: str, address: str, predicate: str | None) -> str:
    """Render one corpus channel binding with the given direction predicate."""
    direction = f" ;\n    narad_p:{predicate} narad_sem:{name}_signal" if predicate else ""
    return f'<https://narad.example.org/binding/{name}> narad_p:fullPv "{address}"{direction} .\n'


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


def _graph_tree(root: Path, corpus: str | None = _SMALL_CORPUS) -> tuple[Path, dict]:
    """A graph-mode facility tree: per-tree sources plus a corpus, no databases.

    Returns the data root and the config a graph-mode render resolves the
    corpus from -- ``ttl_path`` spelled relative, as a project writes it.
    """
    (root / "simulation").mkdir(parents=True)
    (root / "simulation" / "machine.json").write_text(json.dumps({"channels": {}}))
    (root / "machine_state_channels.json").write_text(json.dumps({"_comment": "empty"}))
    (root / LIMITS_FILENAME).write_text("{}\n")
    if corpus is not None:
        (root / "facility.ttl").write_text(corpus)
    config = {
        "channel_finder": {"pipeline_mode": "graph"},
        "services": {"graphdb": {"ttl_path": "./facility.ttl"}},
        "config_dir": str(root),
    }
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
        # The configured spelling, which an operator can retype and edit.
        assert metadata["source_corpus"] == "./facility.ttl"
        # Graph mode stages no tier database by design: nothing is "absent",
        # nothing is corrupt, and no reader is owed either clause.
        assert metadata["absent_paradigms"] == []
        assert metadata["corrupt_paradigms"] == []

    def test_census_is_honest_about_what_the_graph_cannot_say(self, tmp_path):
        """No hierarchy identity keys are invented, so nothing is pyat-coupled.

        The corpus states membership and direction, not the hierarchy path the
        identity keys are read from. Every entry lands pathless in the
        static-noisy partition -- exactly what a database tree without its
        hierarchical paradigm gets -- and the setpoint census says 0 rather
        than guessing from address grammar.
        """
        root, config = _graph_tree(tmp_path / "data")

        prepared = prepare_project_manifest(root, DEFAULT_TIER, config=config)

        metadata = prepared.manifest["_metadata"]
        assert metadata["by_partition"] == {classify.PARTITION_STATIC_NOISY: 5}
        assert metadata["setpoint_count"] == 0
        for channel in prepared.manifest["channels"]:
            assert channel["partition"] == classify.PARTITION_STATIC_NOISY
            for key in ("ring", "system", "family", "device", "field", "subfield"):
                assert channel[key] == ""

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

    def test_a_missing_corpus_backs_no_manifest_and_is_named(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data", corpus=None)

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert "./facility.ttl" in reason
        assert "is not there" in reason
        assert "are all absent" not in reason

    def test_an_unreadable_corpus_backs_no_manifest_and_is_named(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data", corpus="not turtle at all {{{\n")

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert "./facility.ttl" in reason
        assert "could not be read" in reason
        # Never conflated with the paradigm wordings: the operator repairs the
        # corpus, not database files that were never part of graph mode.
        assert "are all absent" not in reason
        assert "channel database" not in reason

    def test_an_empty_corpus_backs_no_manifest_and_is_named(self, tmp_path):
        root, config = _graph_tree(tmp_path / "data", corpus=_TTL_PREAMBLE)

        assert prepare_project_manifest(root, DEFAULT_TIER, config=config) is None
        reason = manifest_gap_reason(root, DEFAULT_TIER, config=config)
        assert "./facility.ttl" in reason
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
