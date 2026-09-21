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
from collections.abc import Sequence
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
from tests._graph_index import build_index_from_ttl, default_index_path

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


def _without_model(root: Path) -> Path:
    """Take the ring and the bindings back out of a copied tree.

    The bundle stages a model, so :func:`_facility_tree` copies one: the
    sources it walks are the tree's ``required_sources``, and those include the
    ring and the bindings exactly when the tree carries bindings. That matters
    to every case about how a channel got classified, because bindings claim
    the pyat-coupled partition outright -- a tree carrying them is never asked
    what its hierarchy could have derived.

    So the two shapes are named rather than assumed. A tree WITH a model is the
    bundle's own and the one a facility ends up with; a tree without one is the
    only shape in which the hierarchy is the classifier, which is what the
    degradation facts below are about.
    """
    paths = ManifestPaths(data_root=root, tier=DEFAULT_TIER)
    for path in (paths.va_bindings, paths.lattice_json):
        if path.is_file():
            path.unlink()
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
        graph_indexes={},
        graph_facts_reported=set(),
    )


def _profile(name: str = "deployment", *, data: str = "data", va: bool = True) -> BuildProfile:
    """A profile that deploys a virtual accelerator from its own ``data:`` tree.

    Every profile carries one — ``data:`` is required and is the build's only
    source of the project's ``data/`` — so the key is spelled by default here
    rather than passed by each test.
    """
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
    """The absence that changes what the accelerator can DO is spelled out.

    Asked of a tree with no model in it, because that is the only tree where
    the hierarchy is what classifies a channel. Bindings claim the pyat-coupled
    partition themselves (see the sibling below), so on a tree carrying them
    the hierarchical database's identity keys are not what pairing rests on and
    there is no cost to state.
    """
    root = _without_model(_facility_tree(tmp_path / "pathless" / "data"))
    ManifestPaths(data_root=root, tier=DEFAULT_TIER).hierarchical_db.unlink()
    prepared = prepare_project_manifest(root, DEFAULT_TIER)

    _report(_shared(tmp_path), _profile(), root, prepared)

    printed = _printed(capsys)
    assert "carry no identity keys" in printed
    assert "serves 0 setpoints" in printed
    assert "static-noisy" in printed
    # And the claim is true of the manifest it describes.
    assert prepared.manifest["_metadata"]["setpoint_count"] == 0


def test_bindings_pair_the_channels_a_missing_hierarchy_could_not(tmp_path, capsys):
    """The same tree, with its model: the cost is not stated because it is not paid.

    What a channel is driven as comes from the bindings when the tree has them
    -- they name the element behind each address, which no hierarchy can infer
    -- and the hierarchical database's identity keys are a fallback for a tree
    that ships no model. So the absence reported above is reported only when it
    costs something, and a tree whose model answers the question is not told
    about a database it did not need.
    """
    root = _facility_tree(tmp_path / "modelled" / "data")
    ManifestPaths(data_root=root, tier=DEFAULT_TIER).hierarchical_db.unlink()
    prepared = prepare_project_manifest(root, DEFAULT_TIER)
    metadata = prepared.manifest["_metadata"]

    _report(_shared(tmp_path), _profile(), root, prepared)

    printed = _printed(capsys)
    assert metadata["partition_source"] == "simulation/va_bindings.json"
    assert metadata["setpoint_count"] > 0
    assert metadata["by_partition"]["pyat-coupled"] > 0
    assert "carry no identity keys" not in printed
    assert "serves 0 setpoints" not in printed


#: The shipped worked example of a database levelled the way another facility
#: levels one: five levels, none of them the ring/field/subfield the partition
#: rules read.
_FOREIGN_LEVELS_DB = (
    PACKAGE_PATHS.data_root / "channel_databases" / "examples" / "hierarchical_jlab_style.json"
)

#: Levelled exactly as the rules read one; every token belongs to some other
#: machine. Readable and classified -- as static-noisy, throughout.
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


def _single_database_tree(root: Path) -> ManifestPaths:
    """A tree staging the hierarchical database alone, so the caller can swap it.

    And staging no model, so what the swapped database is levelled like is what
    decides how its channels are driven. With bindings in the tree they would
    decide it instead, and a database levelled for another machine would be
    read past in silence rather than reported.
    """
    paths = ManifestPaths(data_root=_facility_tree(root), tier=DEFAULT_TIER)
    paths.in_context_db.unlink()
    paths.middle_layer_db.unlink()
    _without_model(paths.data_root)
    return paths


def test_a_foreign_levelled_database_is_reported_not_refused(tmp_path, capsys):
    """A valid file levelled another way used to be called unreadable."""
    paths = _single_database_tree(tmp_path / "levels" / "data")
    shutil.copy2(_FOREIGN_LEVELS_DB, paths.hierarchical_db)

    prepared = prepare_project_manifest(paths.data_root, DEFAULT_TIER)
    _report(_shared(tmp_path), _profile(), paths.data_root, prepared)

    printed = _printed(capsys)
    assert "not levelled the way the partition rules read it" in printed
    assert "levels system/family/sector/device/pv lack ring, field, subfield" in printed
    assert "serves 0 setpoints" in printed


def test_a_foreign_token_database_is_reported_with_the_tokens_it_saw(tmp_path, capsys):
    """Levels the rules can read, tokens no rule matched: the tokens are named."""
    paths = _single_database_tree(tmp_path / "tokens" / "data")
    paths.hierarchical_db.write_text(json.dumps(_FOREIGN_TOKEN_DB))

    prepared = prepare_project_manifest(paths.data_root, DEFAULT_TIER)
    _report(_shared(tmp_path), _profile(), paths.data_root, prepared)

    printed = _printed(capsys)
    assert "matched no partition rule (top-level tokens seen: ZZLINAC)" in printed
    assert "serves 0 setpoints" in printed
    assert "not levelled the way" not in printed


def test_a_foreign_database_beside_a_model_costs_nothing_and_says_nothing(tmp_path, capsys):
    """With bindings in the tree, how the database is levelled stops deciding.

    The bindings name the element behind each address, so the channels they
    name are driven whatever the database's levels look like, and the ones they
    do not name are static-noisy either way. Nothing is mis-driven, so nothing
    is reported -- which also means a facility that staged a database levelled
    for another machine hears about it only if it ships no model.
    """
    paths = _single_database_tree(tmp_path / "levels-modelled" / "data")
    shutil.copy2(_FOREIGN_LEVELS_DB, paths.hierarchical_db)
    for source in (PACKAGE_PATHS.va_bindings, PACKAGE_PATHS.lattice_json):
        shutil.copy2(source, paths.data_root / "simulation" / source.name)

    prepared = prepare_project_manifest(paths.data_root, DEFAULT_TIER)
    _report(_shared(tmp_path), _profile(), paths.data_root, prepared)

    printed = _printed(capsys)
    assert prepared.manifest["_metadata"]["partition_source"] == "simulation/va_bindings.json"
    assert "not levelled the way" not in printed
    assert "serves 0 setpoints" not in printed


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

#: A body no parser can get past at all -- truncated JSON. Every paradigm
#: parser rejects the schema-invalid one above too; this one makes the refusal
#: come from the JSON decoder rather than from a parser's shape check.
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

    Asked of a tree with no model, which is where the staged databases are the
    only source of channels. The sibling below is the other tree, and it does
    not refuse.
    """
    root = _without_model(_facility_tree(tmp_path / "empty" / "data"))
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


def test_an_empty_database_beside_a_model_is_credited_with_the_models_channels(tmp_path, capsys):
    """Current behaviour, pinned because it reads as the thing this file forbids.

    The same empty database, on a tree that stages the bundle's bindings: the
    build does not refuse, because the bindings name addresses and those become
    the manifest. The channels are real and the accelerator can serve them.
    What is not right is the sentence said about them -- they are credited to
    "its in_context channel database(s)", the one file in the tree that named
    nothing -- and the machine-state reconciliation turns up invalid
    candidates, which is the tree telling the operator the two halves disagree.

    Asserted as it behaves today, not as it should. The fix is a build_cmd.py
    change (attribute channels to what actually produced them, and reconsider
    whether a database naming nothing should still be called a source) and is
    reported rather than made here.
    """
    root = _facility_tree(tmp_path / "empty-modelled" / "data")
    paths = ManifestPaths(data_root=root, tier=DEFAULT_TIER)
    paths.hierarchical_db.unlink()
    paths.middle_layer_db.unlink()
    database = json.loads(paths.in_context_db.read_text())
    database["channels"] = {}
    paths.in_context_db.write_text(json.dumps(database, indent=2))
    (root / "simulation" / "machine.json").write_text(json.dumps({"channels": {}}))

    prepared = prepare_project_manifest(root, DEFAULT_TIER)
    _report(_shared(tmp_path), _profile(), root, prepared)

    from osprey.services.virtual_accelerator.bindings import load_bindings, setpoints
    from osprey.services.virtual_accelerator.manifest.classify import (
        pyat_coupled_setpoint_addresses,
    )

    printed = _printed(capsys)
    metadata = prepared.manifest["_metadata"]
    total = metadata["total_channels"]

    assert prepared is not None
    assert metadata["partition_source"] == "simulation/va_bindings.json"
    # Every channel in the manifest came from the bindings: nothing else named one.
    assert set(metadata["by_partition"]) == {"pyat-coupled"}
    assert metadata["by_partition"]["pyat-coupled"] == total
    assert pyat_coupled_setpoint_addresses(prepared.manifest["channels"]) == set(
        setpoints(load_bindings(paths.va_bindings))
    )
    # And the fact credits the empty file, which is the part that is wrong.
    assert f"{total} channel(s) from its in_context channel database(s)" in printed
    assert metadata["machine_state_reconciliation"]["invalid"]


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


def _wired_env(
    tmp_path: Path, *, model: Sequence[str] = (), partition_source: str | None = None
) -> dict[str, str]:
    """Run the build's one write outside the output zone, and read what it wrote.

    *model* names the model files the published tree stages beside its
    manifest, and *partition_source* what that manifest says claimed its
    pyat-coupled partition -- defaulting to the bindings when the tree stages
    them. Together they say in one line which tree a test is describing: a tree
    whose bindings both claimed its channels and are there to be mounted serves
    the lattice they tie those channels to, and anything less serves none.
    """
    from osprey.cli.build_cmd import _wire_build_derived_env
    from osprey.deployment.compose_generator import COMPOSE_ENV_FILENAME
    from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME
    from osprey.utils.dotenv import parse_dotenv_file

    bindings_name = ManifestPaths(data_root=Path("unused")).va_bindings.name
    if partition_source is None:
        partition_source = f"simulation/{bindings_name}" if bindings_name in model else "none"

    repo = tmp_path / "repo"
    simulation = repo / "build" / "data" / "simulation"
    simulation.mkdir(parents=True)
    (simulation / MANIFEST_FILENAME).write_text(
        json.dumps({"_metadata": {"partition_source": partition_source}, "channels": []}),
        encoding="utf-8",
    )
    for name in model:
        (simulation / name).write_text("{}", encoding="utf-8")

    _wire_build_derived_env(repo, repo / "build")

    return parse_dotenv_file(repo / COMPOSE_ENV_FILENAME)


def test_the_none_lattice_spelling_matches_the_containers_own():
    """A respelled constant that drifts would boot the container on the wrong mode."""
    from osprey.cli.build_cmd import _VA_LATTICE_NONE
    from osprey.services.virtual_accelerator import entrypoint

    assert _VA_LATTICE_NONE == entrypoint.LATTICE_NONE


def test_a_tree_that_stages_bindings_serves_its_lattice_by_name(tmp_path):
    """The name, not a mode: the entrypoint looks that file up in the tree."""
    paths = ManifestPaths(data_root=tmp_path / "unused")

    env = _wired_env(tmp_path, model=[paths.va_bindings.name, paths.lattice_json.name])

    assert env["VA_LATTICE"] == paths.lattice_json.name


def test_a_tree_that_stages_no_bindings_names_no_lattice(tmp_path):
    """Nothing ties the channel set to a ring, so none is asserted over it.

    The physics half of the fallback this feature removed: a lattice behind a
    namespace it does not describe.
    """
    from osprey.cli.build_cmd import _VA_LATTICE_NONE

    env = _wired_env(tmp_path)

    assert env["VA_LATTICE"] == _VA_LATTICE_NONE
    assert env["VA_CHANNELS_FILE"]


def test_a_lattice_without_bindings_is_not_a_served_lattice(tmp_path):
    """A ring no address reaches moves nothing, so the bindings are the evidence."""
    from osprey.cli.build_cmd import _VA_LATTICE_NONE

    paths = ManifestPaths(data_root=tmp_path / "unused")

    env = _wired_env(tmp_path, model=[paths.lattice_json.name])

    assert env["VA_LATTICE"] == _VA_LATTICE_NONE


def test_a_census_no_bindings_claimed_is_not_a_served_lattice(tmp_path):
    """The graph-sourced tree's case: readback pairs stated, nothing coupled.

    Its roster states which channels pair and no bindings at all, so its
    channels reach no model and a lattice beside them would boot one that
    nothing drives.
    """
    from osprey.cli.build_cmd import _VA_LATTICE_NONE

    paths = ManifestPaths(data_root=tmp_path / "unused")

    env = _wired_env(
        tmp_path,
        model=[paths.va_bindings.name, paths.lattice_json.name],
        partition_source="none",
    )

    assert env["VA_LATTICE"] == _VA_LATTICE_NONE


def test_bindings_the_render_did_not_stage_are_named_and_not_served(tmp_path, caplog):
    """A manifest partitioned by a document the mount does not carry."""
    import logging

    from osprey.cli.build_cmd import _VA_LATTICE_NONE

    with caplog.at_level(logging.WARNING):
        env = _wired_env(tmp_path, partition_source="simulation/va_bindings.json")

    assert env["VA_LATTICE"] == _VA_LATTICE_NONE
    assert "simulation/va_bindings.json" in caplog.text


def test_a_lattice_the_render_did_not_stage_is_named_and_not_served(tmp_path, caplog):
    """The other half of the same pair: bindings staged, the ring they name absent.

    The derived value is the lattice file's name, so a tree missing that file
    would be handed a pointer to nothing and refuse at boot. Both halves are
    checked on the tree the container mounts, and the absent one is named.
    """
    import logging

    from osprey.cli.build_cmd import _VA_LATTICE_NONE

    paths = ManifestPaths(data_root=tmp_path / "unused")

    with caplog.at_level(logging.WARNING):
        env = _wired_env(tmp_path, model=[paths.va_bindings.name])

    assert env["VA_LATTICE"] == _VA_LATTICE_NONE
    assert paths.lattice_json.name in caplog.text


def test_the_bundles_own_tree_stages_its_model(whole_tree):
    """What the framework ships, so the served case above is the demo's today.

    The bundle carries a ring and the bindings that tie its channels to it, so
    a build over the demo tree names that ring rather than ``none`` -- the
    served branch, not the unserved one. Both files are asked for, because the
    derivation requires the pair: bindings alone describe a model the tree
    cannot build, and a ring alone is one no address reaches.

    It is also why every tree copied from the bundle here carries a model
    unless :func:`_without_model` takes it back out: ``required_sources`` names
    the pair once the bindings are staged.
    """
    paths = ManifestPaths(data_root=whole_tree, tier=DEFAULT_TIER)

    assert paths.va_bindings.is_file()
    assert paths.lattice_json.is_file()
    assert set(paths.required_sources) >= {paths.va_bindings, paths.lattice_json}
    assert paths.missing_sources() == []


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
    # The refusal under test is the database paradigms' — the graph paradigm
    # the preset ships answers from its corpus and stages no database.
    profile = repo / "profile.yml"
    profile.write_text(
        profile.read_text().replace(
            "channel_finder_mode: graph", "channel_finder_mode: hierarchical"
        )
    )
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


def test_the_exemplar_build_names_the_lattice_its_own_tree_stages(built_exemplar):
    """The value is derived from the published tree, and here it says no model.

    The exemplar's data tree carries no bindings, so the build names no lattice
    over a channel set nothing would steer. What would be wrong is claiming one
    without checking the tree the container is about to mount.
    """
    from osprey.cli.build_cmd import _VA_LATTICE_NONE
    from osprey.deployment.compose_generator import COMPOSE_ENV_FILENAME
    from osprey.utils.dotenv import parse_dotenv_file

    paths = ManifestPaths(data_root=built_exemplar / "build" / "data")
    env = parse_dotenv_file(built_exemplar / COMPOSE_ENV_FILENAME)

    assert not paths.va_bindings.is_file()
    assert env["VA_LATTICE"] == _VA_LATTICE_NONE


def test_the_exemplar_render_carries_its_write_bands_at_the_data_root(built_exemplar):
    """The mount is the whole data root, and the model reads its bands from there.

    A lattice-backed boot builds its variables from ``channel_limits.json`` at
    the tree's root, one level above the served directory. The render has to
    carry it there or the container mounts a tree its model cannot read.
    """
    paths = ManifestPaths(data_root=built_exemplar / "build" / "data")

    assert paths.channel_limits.is_file()


# --- a graph-mode tree, served from its knowledge graph ---------------------

#: How a graph-mode project spells the search index the roster reads: the
#: ``services.graphdb.index_path`` default, which is what the manifest's own
#: metadata records and every refusal about it names.
_INDEX_SPELLING = "./data/channel_databases/graph.duckdb"

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


def _graph_repo(root: Path, *, corpus: str | None = _GRAPH_CORPUS, index: bool = False) -> Path:
    """A graph-mode deployment repo: a corpus and the per-tree sources, no databases.

    ``index`` writes the search index the roster reads beside the corpus, for
    the tests that call the manifest step directly. A test that runs the whole
    build leaves it off: the build derives its own index into the render, which
    is the path being exercised.
    """
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
        if index:
            build_index_from_ttl(data / "facility.ttl", _graph_config(root))
    (root / "profile.yml").write_text(
        "name: Graph VA\n"
        "provider: anthropic\n"
        "channel_finder_mode: graph\n"
        "data: data\n"
        "virtual_accelerator:\n"
        "  port: 5064\n"
        "config:\n"
        "  services.graphdb.ttl_path: ./data/facility.ttl\n"
        # The stores this repo's enabled servers dial: the graph the channel
        # finder answers from, and the logbook database the ariel server and
        # its agents read. Each is spelled with the compose directory its
        # fragment is located by. The posture keys every deployment must state
        # follow.
        "  services.graphdb.path: ./services/graphdb\n"
        "  services.postgresql.path: ./services/postgresql\n"
        "  services.postgresql.database_name: ariel\n"
        "  services.postgresql.username: ariel\n"
        "  deployed_services: [graphdb, postgresql]\n"
        "  control_system.type: mock\n"
        "  archiver.type: mock\n"
        "  claude_code.telemetry.enabled: false\n"
        "  hooks.debug: false\n"
        "  system.timezone: UTC\n"
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
    # The file the channel set was actually built from: the roster reads the
    # search index the build derived, and the fact names what it read.
    assert f"channel search index ({_INDEX_SPELLING})" in printed
    assert "knowledge-graph corpus" not in printed
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
    assert manifest["_metadata"]["source_corpus"] == _INDEX_SPELLING
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
    root = _graph_repo(tmp_path / "repo", index=True)
    prepared = prepare_project_manifest(root / "data", DEFAULT_TIER, config=_graph_config(root))

    _report(_shared(tmp_path), _profile(data="data"), root / "data", prepared)

    printed = _printed(capsys)
    assert _INDEX_SPELLING in printed
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
    root = _graph_repo(tmp_path / "repo", corpus=corpus, index=True)
    prepared = prepare_project_manifest(root / "data", DEFAULT_TIER, config=_graph_config(root))

    _report(_shared(tmp_path), _profile(data="data"), root / "data", prepared)

    printed = _printed(capsys)
    assert "2 channel(s)" in printed
    assert "The knowledge graph carries no hierarchy identity keys" in printed
    assert "serves 0 setpoints" in printed
    assert "The corpus pairs" not in printed
    assert prepared.manifest["_metadata"]["setpoint_count"] == 0


def test_a_graph_repo_with_an_unreadable_index_refuses_naming_it(tmp_path):
    """Distinct from both the absent-paradigms and unreadable-databases refusals."""
    root = _graph_repo(tmp_path / "repo")
    index_path = default_index_path(root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(b"not a database {{{")
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
    assert _INDEX_SPELLING in message
    assert "could not be read" in message
    assert "are all absent" not in message
    assert "channel database" not in message


def test_a_graph_repo_with_an_empty_corpus_refuses_naming_it(tmp_path):
    root = _graph_repo(
        tmp_path / "repo",
        corpus="@prefix narad_p: <https://narad.example.org/property/> .\n",
        index=True,
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
    assert _INDEX_SPELLING in message
    assert "declares no channels" in message


def test_a_graph_repo_with_an_unreadable_corpus_fails_a_real_build(tmp_path_factory, caplog):
    """The refusing path through the CLI itself, not just the reporting helper.

    The deferred graph check runs after the render, so this pins that the
    refusal still stops a real ``osprey build`` before anything is published:
    no ``build/`` tree, and no manifest env keys written into ``.env``.

    A corpus the build cannot parse costs the index rather than the render, so
    what the refusal names is the index that was never written -- and the
    remedy it carries is the one that would write it.
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
    assert _INDEX_SPELLING in caplog.text
    assert "is not there" in caplog.text
    assert "osprey knowledge build-index" in caplog.text
    assert "are all absent" not in caplog.text
    # Refused before the swap published anything.
    assert not (repo / "build" / "config.yml").is_file()
    env_path = repo / ".env"
    env = parse_dotenv_file(env_path) if env_path.is_file() else {}
    assert "VA_CHANNELS_FILE" not in env
    assert "VA_LATTICE" not in env


# --- the same facts, about a tree a facility harvested --------------------

# Every case above describes a tree assembled here, from the bundle's own
# sources or from a literal written into this file. These describe the tree an
# operator ends up with after running the MML chain and `osprey build` over a
# real export: the one shape of project that reaches this code with a channel
# set, a ring and the bindings between them all written by the same harvest.
#
# The recipe that produces it lives beside its own assertions in
# test_mml_build_recipes.py and is imported rather than rebuilt, so the chain
# is spelled once and both files read the same published tree.
from tests.cli.test_mml_build_recipes import (  # noqa: E402
    LIMITS_FILE,
    PACKAGED_DATA,
    SERVED,
    served_manifest,
    served_repo,  # noqa: F401  (pytest resolves it by name, not by reference)
)


@pytest.fixture(scope="module")
def harvested(served_repo):  # noqa: F811  (the imported fixture, by pytest's own name)
    """The recipe's published tree, under the name this file reads it by."""
    return served_repo


def _collapsed(text: str) -> str:
    """One line, whitespace collapsed -- the phase reporter wraps its facts."""
    return " ".join(text.split())


def test_a_harvested_trees_fact_names_the_database_the_harvest_wrote(harvested):
    """The channel set is the harvest's, and the fact says which file backs it.

    The same sentence the bundled trees above are held to, said about a tree
    whose one staged database was written minutes earlier by ``mml emit``: it
    names the paradigm that fed the manifest, names the two the harvest did
    not write, and claims no channel the tree does not hold.
    """
    printed = _collapsed(harvested["build"])
    total = served_manifest(harvested["repo"])["_metadata"]["total_channels"]

    assert f"{total} channel(s) from its middle_layer channel database(s)" in printed
    assert "Not staged at that tier: hierarchical and in_context" in printed
    assert _DEAD_FALLBACK_SENTENCE not in printed


def test_the_reconciliation_fact_rides_along_on_a_harvested_tree(harvested):
    """The machine-state list the harvest emitted is checked against that set.

    Both facts are said once per tree, so a harvested tree gets the second one
    too -- and on a tree where one harvest wrote both documents, every
    candidate the list names is an address the manifest serves.
    """
    printed = _collapsed(harvested["build"])
    listed = json.loads(
        (harvested["repo"] / "data" / "machine_state_channels.json").read_text(encoding="utf-8")
    )
    # The document is address -> entry, with the provenance stamp and the
    # note it opens on spelled as underscore keys.
    checked = len([key for key in listed if not key.startswith("_")])

    assert checked
    assert f"{checked} checked, {checked} valid, 0 invalid" in printed


def test_the_facility_bands_survive_the_lane_and_the_build(harvested):
    """A band the facility authored is still its own after the whole chain.

    ``channel_limits.json`` is the one document of the served tree that is
    shared: the deployment's own bands are in it before the harvest runs, and
    the virtual-accelerator lane states the bands of the channels it bound by
    merging into that file rather than replacing it. The build then copies the
    merged file beside the manifest. So the invariant is asked of the end of
    the chain, where it can actually fail: every entry the project carried
    before the harvest is still there, unchanged, in what the container will
    read -- and the lane's own bands are an addition to it.
    """
    before = json.loads((PACKAGED_DATA / LIMITS_FILE).read_text(encoding="utf-8"))
    published = json.loads((harvested["repo"] / SERVED / LIMITS_FILE).read_text(encoding="utf-8"))

    assert {key: published.get(key) for key in before} == before
    assert published.keys() > before.keys()


def test_the_published_bands_sit_at_the_root_and_beside_the_manifest(harvested):
    """Both readers find the same file: the model's, and the IOC's clamp.

    The model resolves the bands from the data root it is mounted at, and the
    IOC reads them from beside the manifest it serves. One build writes both,
    and a difference between them would clamp a write at one value while the
    model believed another.
    """
    data_root = harvested["repo"] / "build" / "data"

    assert (data_root / LIMITS_FILE).read_bytes() == (
        data_root / "simulation" / LIMITS_FILE
    ).read_bytes()
