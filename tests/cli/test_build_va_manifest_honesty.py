"""What a build says about the channel set its virtual accelerator will serve.

A project's accelerator serves the PROJECT's channels. It is built from
whatever paradigm channel databases the project's data tree stages at the tier
being built, whichever subset that is, and when the tree names no channels at
all the build refuses. The outcome this file exists to make impossible is the
third one that used to happen silently: a container serving the framework's
bundled demo namespace while its operators read their own facility's name on
it.

So there are exactly two outcomes, and both are stated:

* a tree staging one or more channel databases yields a manifest, and a fact
  names which databases fed it and which the tree did not stage;
* a tree staging none, with a virtual accelerator deployed, refuses the build
  and names what is missing.

The machine-state reconciliation counts ride along as a second fact. Both are
said once per ``(data root, tier)``, not once per persona render, and neither
is said at all when the deployment runs no virtual accelerator.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_cmd import _report_va_manifest_outcome, _SharedRenderInputs
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.build_profile_schema import VAConfig
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.templates.manager import TemplateManager
from osprey.errors import BuildProfileError
from osprey.services.virtual_accelerator.manifest.build import (
    LIMITS_FILENAME,
    prepare_project_manifest,
)
from osprey.services.virtual_accelerator.manifest.paths import (
    DEFAULT_TIER,
    PACKAGE_PATHS,
    ManifestPaths,
)

#: The sentence that must no longer exist anywhere in a build's output.
_DEAD_FALLBACK_SENTENCE = "built-in demo namespace"


@pytest.fixture(autouse=True)
def _plain_reporter():
    """Print facts without color, so an assertion reads the words alone."""
    previous = install_reporter(PhaseReporter(color=False))
    yield
    install_reporter(previous)


def _facility_tree(root: Path) -> Path:
    """Copy the bundled sources into ``root`` as a standalone facility tree."""
    paths = ManifestPaths(data_root=PACKAGE_PATHS.data_root, tier=DEFAULT_TIER)
    sources = [*paths.required_sources, paths.channel_limits]
    for source in sources:
        destination = root / source.relative_to(PACKAGE_PATHS.data_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return root


@pytest.fixture(scope="module")
def whole_tree(tmp_path_factory) -> Path:
    """A tree staging every paradigm database, as the bundle does."""
    return _facility_tree(tmp_path_factory.mktemp("whole") / "data")


@pytest.fixture(scope="module")
def partial_tree(tmp_path_factory) -> Path:
    """A tree staging ONE paradigm database, as the exemplar's does."""
    root = _facility_tree(tmp_path_factory.mktemp("partial") / "data")
    paths = ManifestPaths(data_root=root, tier=DEFAULT_TIER)
    paths.in_context_db.unlink()
    paths.middle_layer_db.unlink()
    return root


def _printed(capsys: pytest.CaptureFixture[str]) -> str:
    """Everything printed so far, whitespace collapsed (the console wraps)."""
    return " ".join(capsys.readouterr().out.split())


def _shared(repo_root: Path) -> _SharedRenderInputs:
    """The render inputs one build shares across the deployment and its personas."""
    return _SharedRenderInputs(
        repo_root=repo_root,
        build_dir=repo_root / "build",
        runtime_root=None,
        project_deps=[],
        skip_deps=True,
        manager=TemplateManager(),
        va_manifests={},
        va_reported=set(),
    )


def _profile(name: str = "deployment", *, data: str | None = None, va: bool = True) -> BuildProfile:
    """A profile that deploys a virtual accelerator, optionally with a ``data:`` tree."""
    return BuildProfile(
        name=name,
        data=data,
        virtual_accelerator=VAConfig() if va else None,
    )


def _report(shared, profile, tree: Path, prepared: Any) -> None:
    """Run the reporting step for one render."""
    _report_va_manifest_outcome(
        shared, profile, data_root=tree, tier=DEFAULT_TIER, prepared=prepared
    )


# --- a tree that stages channel databases -----------------------------------


def test_a_partial_tree_backs_a_manifest_from_what_it_stages(partial_tree):
    """One database is a namespace: the exemplar's shape, and it must build."""
    prepared = prepare_project_manifest(partial_tree, DEFAULT_TIER)

    assert prepared is not None
    metadata = prepared.manifest["_metadata"]
    assert metadata["source_paradigms"] == ["hierarchical"]
    assert metadata["absent_paradigms"] == ["in_context", "middle_layer"]
    assert metadata["total_channels"] > 0


def test_a_partial_trees_fact_names_the_databases_that_fed_it(partial_tree, tmp_path, capsys):
    prepared = prepare_project_manifest(partial_tree, DEFAULT_TIER)

    _report(_shared(tmp_path), _profile(), partial_tree, prepared)

    printed = _printed(capsys)
    assert "from its hierarchical channel database(s)" in printed
    assert "Not staged at that tier: in_context and middle_layer." in printed
    assert _DEAD_FALLBACK_SENTENCE not in printed
    # Told what will be served, not sent to a path to retype.
    assert str(partial_tree) not in printed


def test_a_whole_trees_fact_names_all_three_and_claims_nothing_absent(whole_tree, tmp_path, capsys):
    prepared = prepare_project_manifest(whole_tree, DEFAULT_TIER)

    _report(_shared(tmp_path), _profile(), whole_tree, prepared)

    printed = _printed(capsys)
    assert "from its hierarchical, in_context and middle_layer channel database(s)" in printed
    assert "Not staged" not in printed


def test_the_reconciliation_fact_carries_the_three_counts(whole_tree, tmp_path, capsys):
    prepared = prepare_project_manifest(whole_tree, DEFAULT_TIER)
    reconciliation = prepared.manifest["_metadata"]["machine_state_reconciliation"]

    _report(_shared(tmp_path), _profile(), whole_tree, prepared)

    printed = _printed(capsys)
    assert f"{reconciliation['candidates_checked']} checked" in printed
    assert f"{len(reconciliation['valid'])} valid" in printed
    assert f"{len(reconciliation['invalid'])} invalid" in printed


def test_both_facts_are_said_once_per_tree_across_personas(whole_tree, tmp_path, capsys):
    """The deployment and its personas share one tree, so one pair of facts."""
    shared = _shared(tmp_path)
    prepared = prepare_project_manifest(whole_tree, DEFAULT_TIER)

    for name in ("deployment", "persona-a", "persona-b", "persona-c"):
        _report(shared, _profile(name), whole_tree, prepared)

    printed = _printed(capsys)
    assert printed.count("Virtual-accelerator channel set built") == 1
    assert printed.count("machine-state channels reconciled") == 1


def test_a_second_tree_is_its_own_pair_of_facts(whole_tree, partial_tree, tmp_path, capsys):
    """The key is the tree, so a persona that moved its data tree reports again."""
    shared = _shared(tmp_path)

    _report(shared, _profile(), whole_tree, prepare_project_manifest(whole_tree, DEFAULT_TIER))
    _report(
        shared,
        _profile("persona"),
        partial_tree,
        prepare_project_manifest(partial_tree, DEFAULT_TIER),
    )

    assert _printed(capsys).count("Virtual-accelerator channel set built") == 2


def test_the_fact_separates_seeded_addresses_from_database_ones(tmp_path, capsys):
    """A scenario seed is a different kind of source, so it gets its own clause."""
    root = _facility_tree(tmp_path / "seeded" / "data")
    machine_json = root / "simulation" / "machine.json"
    machine = json.loads(machine_json.read_text())
    machine["channels"]["SR:VAC:GAUGE:SR99:PRESSURE:RB"] = {
        "value": 1e-9,
        "units": "Torr",
        "description": "An address that exists in no channel database",
    }
    machine_json.write_text(json.dumps(machine, indent=2))
    prepared = prepare_project_manifest(root, DEFAULT_TIER)

    _report(_shared(tmp_path), _profile(), root, prepared)

    printed = _printed(capsys)
    assert "plus 1 address(es) seeded only by simulation/machine.json" in printed
    # The database count excludes the seeded one rather than absorbing it.
    from_databases = prepared.manifest["_metadata"]["total_channels"] - 1
    assert f"{from_databases} channel(s) from its" in printed


def test_a_whole_trees_fact_claims_no_seeded_addresses(whole_tree, tmp_path, capsys):
    """The clause appears only when there is something to declare."""
    _report(
        _shared(tmp_path),
        _profile(),
        whole_tree,
        prepare_project_manifest(whole_tree, DEFAULT_TIER),
    )

    assert "seeded only by" not in _printed(capsys)


def test_without_a_hierarchical_database_the_fact_states_the_cost(tmp_path, capsys):
    """The absence that changes what the accelerator can DO is spelled out."""
    root = _facility_tree(tmp_path / "pathless" / "data")
    ManifestPaths(data_root=root, tier=DEFAULT_TIER).hierarchical_db.unlink()
    prepared = prepare_project_manifest(root, DEFAULT_TIER)

    _report(_shared(tmp_path), _profile(), root, prepared)

    printed = _printed(capsys)
    assert "carry no identity keys" in printed
    assert "serves 0 setpoints" in printed
    assert "static-noisy" in printed
    # And the claim is true of the manifest it describes.
    assert prepared.manifest["_metadata"]["setpoint_count"] == 0


def test_a_hierarchical_trees_fact_states_no_degradation(whole_tree, tmp_path, capsys):
    _report(
        _shared(tmp_path),
        _profile(),
        whole_tree,
        prepare_project_manifest(whole_tree, DEFAULT_TIER),
    )

    assert "carry no identity keys" not in _printed(capsys)


# --- a tree staging a database it cannot read -------------------------------

#: Valid JSON in a shape no paradigm parser accepts: the file is there and the
#: build has every reason to believe it holds the facility's channels.
_SCHEMA_INVALID_DB = '{"channels": {"FACILITY:TIER:SRC": {"description": "profile"}}}\n'

#: A body no parser can get past at all: the schema-invalid one above reads as
#: an empty tree to the middle-layer parser, which navigates any nesting it is
#: given, so a test that needs EVERY staged database unreadable truncates the
#: JSON instead.
_UNPARSEABLE_DB = '{"channels": [\n'


def test_a_corrupt_database_degrades_and_the_fact_names_the_file(tmp_path, capsys):
    """One broken database out of three is a degraded namespace, not a dead one.

    The manifest is built from the databases that are left, and the fact says
    so: the count belongs to the sources it names, and the file that could not
    be read is named as unreadable rather than silently missing from the list.
    """
    root = _facility_tree(tmp_path / "corrupt" / "data")
    paths = ManifestPaths(data_root=root, tier=DEFAULT_TIER)
    paths.in_context_db.write_text(_SCHEMA_INVALID_DB)

    prepared = prepare_project_manifest(root, DEFAULT_TIER)
    _report(_shared(tmp_path), _profile(), root, prepared)

    printed = _printed(capsys)
    assert "from its hierarchical and middle_layer channel database(s)" in printed
    assert "Staged but unreadable, contributing no channels: in_context" in printed
    assert str(paths.in_context_db.relative_to(root)) in printed
    # Named as broken, never as one the project did not stage.
    assert "Not staged" not in printed
    assert prepared.manifest["_metadata"]["source_paradigms"] == ["hierarchical", "middle_layer"]


def test_a_readable_trees_fact_claims_nothing_unreadable(whole_tree, tmp_path, capsys):
    """The clause appears only when there is something to declare."""
    _report(
        _shared(tmp_path),
        _profile(),
        whole_tree,
        prepare_project_manifest(whole_tree, DEFAULT_TIER),
    )

    assert "Staged but unreadable" not in _printed(capsys)


def test_a_tree_whose_every_database_is_unreadable_refuses_as_corrupt(tmp_path):
    """Nothing usable is left, and the refusal sends the operator to the files.

    The wording is deliberately not the absent-paradigms one: these databases
    were shipped and are broken, so the remedy is repairing them rather than
    staging something that was never there.
    """
    root = _facility_tree(tmp_path / "unreadable" / "data")
    paths = ManifestPaths(data_root=root, tier=DEFAULT_TIER)
    for database in paths.paradigm_databases.values():
        database.write_text(_UNPARSEABLE_DB)

    assert prepare_project_manifest(root, DEFAULT_TIER) is None
    with pytest.raises(BuildProfileError) as excinfo:
        _report(_shared(tmp_path), _profile(), root, None)

    message = str(excinfo.value)
    assert "present and could not be read" in message
    assert "are all absent" not in message
    for paradigm, database in paths.paradigm_databases.items():
        assert paradigm in message
        assert str(database.relative_to(root)) in message


# --- a tree that stages none ------------------------------------------------


def test_a_tree_with_no_channel_databases_refuses_the_build(tmp_path, capsys):
    """No fallback exists any more, so the only honest answer is to stop."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    with pytest.raises(BuildProfileError) as excinfo:
        _report(_shared(tmp_path), _profile(data="data"), data_root, None)

    message = str(excinfo.value)
    assert "no channel database is staged at tier 3" in message
    assert "hierarchical" in message and "in_context" in message and "middle_layer" in message
    assert "virtual_accelerator" in message
    assert _DEAD_FALLBACK_SENTENCE not in _printed(capsys)


def test_a_bundled_tree_with_no_channel_databases_refuses_too(tmp_path, capsys):
    """The refusal does not turn on `data:`. A deployed accelerator needs channels."""
    data_root = tmp_path / "bundled" / "data"
    data_root.mkdir(parents=True)

    with pytest.raises(BuildProfileError):
        _report(_shared(tmp_path), _profile(), data_root, None)


def test_a_tree_missing_its_scenario_seed_refuses_and_names_the_file(partial_tree, tmp_path):
    """Databases alone are not a tree: what is missing is named, not guessed at."""
    root = _facility_tree(tmp_path / "seedless" / "data")
    (root / "simulation" / "machine.json").unlink()

    assert prepare_project_manifest(root, DEFAULT_TIER) is None
    with pytest.raises(BuildProfileError) as excinfo:
        _report(_shared(tmp_path), _profile(), root, None)

    assert "simulation/machine.json" in str(excinfo.value)


def test_a_staged_but_empty_database_refuses(tmp_path):
    """A file that exists but names nothing is not a namespace either.

    The gate one layer down asks whether the database FILE is there. Without
    this, an empty one would sail past it and hand a deployed accelerator a
    manifest with no channels in it.
    """
    root = _facility_tree(tmp_path / "empty" / "data")
    paths = ManifestPaths(data_root=root, tier=DEFAULT_TIER)
    paths.hierarchical_db.unlink()
    paths.middle_layer_db.unlink()
    # Well-formed and empty, not malformed: the real header with no channels
    # under it, which is what a facility that has not filled its database in
    # yet ships.
    database = json.loads(paths.in_context_db.read_text())
    database["channels"] = {}
    paths.in_context_db.write_text(json.dumps(database, indent=2))
    (root / "simulation" / "machine.json").write_text(json.dumps({"channels": {}}))

    assert paths.staged_paradigms == ("in_context",)
    assert prepare_project_manifest(root, DEFAULT_TIER) is None
    with pytest.raises(BuildProfileError) as excinfo:
        _report(_shared(tmp_path), _profile(), root, None)

    assert "name no channels" in str(excinfo.value)
    assert "in_context" in str(excinfo.value)


def test_a_tree_missing_its_drive_limits_refuses(tmp_path):
    """Limits and manifest ship together: a manifest alone accepts any setpoint."""
    root = _facility_tree(tmp_path / "limitless" / "data")
    (root / LIMITS_FILENAME).unlink()

    assert prepare_project_manifest(root, DEFAULT_TIER) is None
    with pytest.raises(BuildProfileError) as excinfo:
        _report(_shared(tmp_path), _profile(), root, None)

    assert LIMITS_FILENAME in str(excinfo.value)


# --- nothing said when no accelerator is deployed ---------------------------


def test_a_deployment_without_a_virtual_accelerator_says_nothing(whole_tree, tmp_path, capsys):
    """The manifest is prepared for every build; only a deployed VA is reported on."""
    _report(_shared(tmp_path), _profile(va=False), whole_tree, None)

    assert _printed(capsys) == ""


def test_a_deployment_without_a_virtual_accelerator_is_not_refused(tmp_path, capsys):
    """A tree with no channels is only a problem for a build that deploys one."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    _report(_shared(tmp_path), _profile(data="data", va=False), data_root, None)

    assert _printed(capsys) == ""


def test_an_attached_project_says_nothing(tmp_path, capsys):
    """It deploys no services of its own, so the virtual accelerator is its host's."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    profile = BuildProfile(
        name="attached",
        data="data",
        deploy_services=False,
        virtual_accelerator=VAConfig(),
    )

    _report(_shared(tmp_path), profile, data_root, None)

    assert _printed(capsys) == ""


# --- the lattice the manifest earns ----------------------------------------


def _wired_env(tmp_path: Path, manifest: dict) -> dict[str, str]:
    """Run the build's one write outside ``build/`` over *manifest*, and read it."""
    from osprey.cli.build_cmd import _wire_build_derived_env
    from osprey.deployment.compose_generator import COMPOSE_ENV_FILENAME
    from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME
    from osprey.utils.dotenv import parse_dotenv_file

    repo = tmp_path / "repo"
    simulation = repo / "build" / "data" / "simulation"
    simulation.mkdir(parents=True)
    (simulation / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    _wire_build_derived_env(repo, repo / "build")

    return parse_dotenv_file(repo / COMPOSE_ENV_FILENAME)


def test_the_none_lattice_spelling_matches_the_containers_own():
    """A respelled constant that drifts would boot the container on the wrong mode."""
    from osprey.cli.build_cmd import _VA_LATTICE_NONE
    from osprey.services.virtual_accelerator import entrypoint

    assert _VA_LATTICE_NONE == entrypoint.LATTICE_NONE


def test_a_manifest_with_lattice_channels_keeps_the_builtin_lattice(whole_tree, tmp_path):
    """The bundle's case: pyat-coupled channels, so there is a model to run."""
    from osprey.utils.dotenv import VA_LATTICE_DEFAULT

    prepared = prepare_project_manifest(whole_tree, DEFAULT_TIER)
    assert prepared.manifest["_metadata"]["by_partition"].get("pyat-coupled")

    env = _wired_env(tmp_path, prepared.manifest)

    assert env["VA_LATTICE"] == VA_LATTICE_DEFAULT


def test_a_manifest_without_lattice_channels_gets_no_lattice(tmp_path):
    """Nothing for the built-in model to steer, so it is not asserted over it.

    The physics half of the fallback this feature removed: a lattice behind a
    namespace it does not describe.
    """
    from osprey.cli.build_cmd import _VA_LATTICE_NONE

    manifest = {"_metadata": {"by_partition": {"static-noisy": 9}}, "channels": []}

    env = _wired_env(tmp_path, manifest)

    assert env["VA_LATTICE"] == _VA_LATTICE_NONE


def test_an_unreadable_census_claims_no_lattice(tmp_path):
    """A lattice is the claim that needs evidence, so absence of it answers no."""
    from osprey.cli.build_cmd import _VA_LATTICE_NONE

    env = _wired_env(tmp_path, {"channels": []})

    assert env["VA_LATTICE"] == _VA_LATTICE_NONE
    assert env["VA_CHANNELS_FILE"]


# --- the whole build, on the framework's own gold-standard repo -------------


@pytest.fixture(scope="module")
def built_exemplar(tmp_path_factory) -> Path:
    """One real ``osprey build`` of the exemplar repo, which stages ONE database.

    The end-to-end shape of the ruling: the framework's own reference
    deployment carries a ``data:`` tree with a single paradigm database staged
    at the tier it builds, runs a virtual accelerator, and must come out of a
    build serving its OWN channels. Module-scoped because the build is the
    expensive part and both assertions below read the same run.
    """
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command
    from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

    repo = build_exemplar_repo(
        tmp_path_factory.mktemp("exemplar") / EXEMPLAR_DIRNAME, seed_env=True
    )
    previous = Path.cwd()
    os.chdir(repo)
    try:
        result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)
    assert result.exit_code == 0, result.output
    return repo


def test_the_exemplar_builds_and_serves_its_own_channels(built_exemplar):
    """Its manifest is generated from its own tree, not the bundled tutorial one."""
    from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME

    manifest = json.loads(
        (built_exemplar / "build" / "data" / "simulation" / MANIFEST_FILENAME).read_text()
    )
    bundled = prepare_project_manifest(PACKAGE_PATHS.data_root, DEFAULT_TIER)

    assert manifest["_metadata"]["source_paradigms"] == ["hierarchical"]
    assert manifest["channels"]
    assert {c["address"] for c in manifest["channels"]} != {
        c["address"] for c in bundled.manifest["channels"]
    }


def test_the_exemplar_build_leaves_the_manifest_env_set(built_exemplar):
    """No build path leaves the pointer unset, so the built-in default is unreachable."""
    from osprey.deployment.compose_generator import COMPOSE_ENV_FILENAME
    from osprey.utils.dotenv import parse_dotenv_file

    env = parse_dotenv_file(built_exemplar / COMPOSE_ENV_FILENAME)

    assert env.get("VA_CHANNELS_FILE")


def test_a_va_deploying_repo_with_no_channel_databases_fails_the_build(tmp_path_factory, caplog):
    """The refusal reaches a real `osprey build`, not just the helper.

    Every other refusal test here calls the reporting step directly. This one
    pins the wiring: that `_render_project` calls it at all, and that a build
    which cannot serve the project's channels stops with a non-zero exit
    instead of rendering a deployment around the framework's demo namespace.
    """
    import shutil as _shutil

    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command
    from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

    repo = build_exemplar_repo(tmp_path_factory.mktemp("dbless") / EXEMPLAR_DIRNAME, seed_env=True)
    _shutil.rmtree(repo / "data" / "channel_databases")

    previous = Path.cwd()
    os.chdir(repo)
    try:
        with caplog.at_level(logging.ERROR):
            result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)

    assert result.exit_code != 0
    # The refusal reaches an operator through the build logger, which is where
    # every other build error is spelled; stdout carries the phase card alone.
    assert "no channel database is staged at tier 3" in caplog.text
    for paradigm in ("hierarchical", "in_context", "middle_layer"):
        assert paradigm in caplog.text
    # And it stopped BEFORE writing a deployment around a namespace it cannot serve.
    assert not (repo / "build" / "config.yml").is_file()


def test_the_exemplar_build_derives_its_lattice_from_its_own_channels(built_exemplar):
    """It EARNS `builtin`: 8 of its 9 channels are pyat-coupled, so a model applies.

    The value is derived, not asserted, and here the derivation says yes. What
    would have been wrong is claiming the lattice without checking, which is
    what a manifest of pure telemetry gets caught by in
    ``test_a_manifest_without_lattice_channels_gets_no_lattice``.
    """
    from osprey.deployment.compose_generator import COMPOSE_ENV_FILENAME
    from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME
    from osprey.utils.dotenv import VA_LATTICE_DEFAULT, parse_dotenv_file

    manifest = json.loads(
        (built_exemplar / "build" / "data" / "simulation" / MANIFEST_FILENAME).read_text()
    )
    env = parse_dotenv_file(built_exemplar / COMPOSE_ENV_FILENAME)

    assert manifest["_metadata"]["by_partition"]["pyat-coupled"] > 0
    assert env["VA_LATTICE"] == VA_LATTICE_DEFAULT


# --- a graph-mode tree, served from its knowledge graph ---------------------

_GRAPH_CORPUS = """\
@prefix narad_p: <https://narad.example.org/property/> .
@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .
<https://narad.example.org/binding/hcm_sp> narad_p:fullPv "SR:MAG:HCM:01:CURRENT:SP" ;
    narad_p:writesSignal narad_sem:hcm_signal .
<https://narad.example.org/binding/hcm_rb> narad_p:fullPv "SR:MAG:HCM:01:CURRENT:RB" ;
    narad_p:readsSignal narad_sem:hcm_signal .
<https://narad.example.org/binding/bpm_x> narad_p:fullPv "SR:DIAG:BPM:01:POSITION:X" ;
    narad_p:readsSignal narad_sem:bpm_signal .
"""


def _graph_repo(root: Path, *, corpus: str | None = _GRAPH_CORPUS) -> Path:
    """A graph-mode deployment repo: a corpus and the per-tree sources, no databases."""
    from tests.fixtures.lifecycle_repo import FACILITY_ONTOLOGY_JSON

    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    (data / "simulation").mkdir(parents=True)
    (data / "simulation" / "machine.json").write_text(json.dumps({"channels": {}}))
    (data / "machine_state_channels.json").write_text(json.dumps({"_comment": "empty"}))
    (data / LIMITS_FILENAME).write_text("{}\n")
    (data / "facility_knowledge").mkdir()
    # The bundle's config names a compiled ontology under data/; the exemplar's
    # table satisfies it without this test growing a vocabulary of its own.
    (data / "facility_ontology.json").write_text(FACILITY_ONTOLOGY_JSON)
    if corpus is not None:
        (data / "facility.ttl").write_text(corpus)
    (root / "profile.yml").write_text(
        "name: Graph VA\n"
        "app_template: control_assistant\n"
        "provider: anthropic\n"
        "channel_finder_mode: graph\n"
        "data: data\n"
        "virtual_accelerator:\n"
        "  port: 5064\n"
        "config:\n"
        "  services.graphdb.ttl_path: ./data/facility.ttl\n"
    )
    return root


def _graph_config(root: Path) -> dict[str, Any]:
    """The rendered-config shape the deferred graph manifest step consults."""
    return {
        "channel_finder": {"pipeline_mode": "graph"},
        "services": {"graphdb": {"ttl_path": "./data/facility.ttl"}},
        "config_dir": str(root),
    }


@pytest.fixture(autouse=True)
def _cold_roster_cache():
    """Every test resolves its own corpus cold; none inherits another's parse."""
    import osprey.channel_roster as channel_roster

    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def test_a_graph_mode_repo_deploys_a_va_and_the_fact_names_the_corpus(tmp_path_factory, capsys):
    """The whole build: a knowledge graph is a channel source, and it is said.

    A graph-mode facility stages no paradigm database at all -- its channels
    live in the corpus the graph store is seeded from -- and a build deploying
    a virtual accelerator on it used to refuse as if the facility had no
    channels. Now it serves them, and the fact names the corpus rather than
    claiming database files that were never part of graph mode.
    """
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command
    from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME

    repo = _graph_repo(tmp_path_factory.mktemp("graph") / "repo")

    previous = Path.cwd()
    os.chdir(repo)
    try:
        result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)

    assert result.exit_code == 0, result.output
    printed = " ".join(result.output.split())
    assert "knowledge-graph corpus (./data/facility.ttl)" in printed
    assert "3 channel(s)" in printed
    # The honest gain and cost: the one pair the roster vouches for is served
    # as a setpoint echo, and everything else is static-noisy -- no identity
    # keys and no lattice.
    assert "The corpus pairs 1 setpoint(s) with a readback" in printed
    assert "served as setpoint-echo channels, every other channel as static-noisy" in printed
    assert "serves 0 setpoints" not in printed
    assert _DEAD_FALLBACK_SENTENCE not in printed

    manifest = json.loads((repo / "build" / "data" / "simulation" / MANIFEST_FILENAME).read_text())
    assert manifest["_metadata"]["source_paradigms"] == ["graph"]
    assert manifest["_metadata"]["source_corpus"] == "./data/facility.ttl"
    assert {c["address"] for c in manifest["channels"]} == {
        "SR:MAG:HCM:01:CURRENT:SP",
        "SR:MAG:HCM:01:CURRENT:RB",
        "SR:DIAG:BPM:01:POSITION:X",
    }
    from osprey.utils.dotenv import parse_dotenv_file

    env = parse_dotenv_file(repo / ".env")
    assert env["VA_CHANNELS_FILE"] == MANIFEST_FILENAME
    # Nothing pyat-coupled to steer, so the built-in lattice is not asserted.
    assert env["VA_LATTICE"] == "none"


def test_a_graph_manifests_fact_names_the_corpus_not_databases(tmp_path, capsys):
    """The reporting step alone, for the wording the build test reads end to end."""
    root = _graph_repo(tmp_path / "repo")
    prepared = prepare_project_manifest(root / "data", DEFAULT_TIER, config=_graph_config(root))

    _report(_shared(tmp_path), _profile(data="data"), root / "data", prepared)

    printed = _printed(capsys)
    assert "knowledge-graph corpus (./data/facility.ttl)" in printed
    assert "channel database(s)" not in printed
    assert "Not staged" not in printed
    assert "The corpus pairs 1 setpoint(s) with a readback" in printed
    # The pair is a real setpoint, so the all-static-noisy degradation
    # sentence would be false here and is not printed over it.
    assert "carries no hierarchy identity keys" not in printed
    assert "serves 0 setpoints" not in printed


def test_a_graph_corpus_pairing_nothing_still_states_the_cost(tmp_path, capsys):
    """A corpus whose device grouping states no pair gets the degradation sentence.

    The claim is read off the manifest's census, not the source: with no
    setpoint served, saying so is the honest fact, and the pairing sentence
    (which would read ``pairs 0 setpoint(s)``) is not printed at all.
    """
    corpus = (
        "@prefix narad_p: <https://narad.example.org/property/> .\n"
        "@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .\n"
        '<https://narad.example.org/binding/dcct> narad_p:fullPv "SR01C___T______AM00" ;\n'
        "    narad_p:readsSignal narad_sem:dcct_signal .\n"
        '<https://narad.example.org/binding/bend_sp> narad_p:fullPv "SR01C___B______AC00" ;\n'
        "    narad_p:writesSignal narad_sem:bend_signal .\n"
    )
    root = _graph_repo(tmp_path / "repo", corpus=corpus)
    prepared = prepare_project_manifest(root / "data", DEFAULT_TIER, config=_graph_config(root))

    _report(_shared(tmp_path), _profile(data="data"), root / "data", prepared)

    printed = _printed(capsys)
    assert "2 channel(s)" in printed
    assert "The knowledge graph carries no hierarchy identity keys" in printed
    assert "serves 0 setpoints" in printed
    assert "The corpus pairs" not in printed
    assert prepared.manifest["_metadata"]["setpoint_count"] == 0


def test_a_graph_repo_with_an_unreadable_corpus_refuses_naming_it(tmp_path):
    """Distinct from both the absent-paradigms and unreadable-databases refusals."""
    root = _graph_repo(tmp_path / "repo", corpus="not turtle {{{\n")
    config = _graph_config(root)

    assert prepare_project_manifest(root / "data", DEFAULT_TIER, config=config) is None
    with pytest.raises(BuildProfileError) as excinfo:
        _report_va_manifest_outcome(
            _shared(tmp_path),
            _profile(data="data"),
            data_root=root / "data",
            tier=DEFAULT_TIER,
            prepared=None,
            config=config,
        )

    message = str(excinfo.value)
    assert "./data/facility.ttl" in message
    assert "could not be read" in message
    assert "are all absent" not in message
    assert "channel database" not in message


def test_a_graph_repo_with_an_empty_corpus_refuses_naming_it(tmp_path):
    root = _graph_repo(
        tmp_path / "repo",
        corpus="@prefix narad_p: <https://narad.example.org/property/> .\n",
    )
    config = _graph_config(root)

    assert prepare_project_manifest(root / "data", DEFAULT_TIER, config=config) is None
    with pytest.raises(BuildProfileError) as excinfo:
        _report_va_manifest_outcome(
            _shared(tmp_path),
            _profile(data="data"),
            data_root=root / "data",
            tier=DEFAULT_TIER,
            prepared=None,
            config=config,
        )

    message = str(excinfo.value)
    assert "./data/facility.ttl" in message
    assert "declares no channels" in message


def test_a_graph_repo_with_an_unreadable_corpus_fails_a_real_build(tmp_path_factory, caplog):
    """The refusing path through the CLI itself, not just the reporting helper.

    The deferred graph check runs after the render, so this pins that the
    refusal still stops a real ``osprey build`` before anything is published:
    no ``build/`` tree, and no manifest env keys written into ``.env``.
    """
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command
    from osprey.utils.dotenv import parse_dotenv_file

    repo = _graph_repo(tmp_path_factory.mktemp("graph-bad") / "repo", corpus="not turtle {{{\n")

    previous = Path.cwd()
    os.chdir(repo)
    try:
        with caplog.at_level(logging.ERROR):
            result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)

    assert result.exit_code != 0
    assert "./data/facility.ttl" in caplog.text
    assert "could not be read" in caplog.text
    assert "are all absent" not in caplog.text
    # Refused before the swap published anything.
    assert not (repo / "build" / "config.yml").is_file()
    env_path = repo / ".env"
    env = parse_dotenv_file(env_path) if env_path.is_file() else {}
    assert "VA_CHANNELS_FILE" not in env
    assert "VA_LATTICE" not in env
