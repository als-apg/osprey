"""The install recipes an MML import ends in, run end to end.

An import is only finished when ``osprey build`` accepts the facility it
wrote, so each recipe here is the literal sequence an operator types --
``init``, ``facility import mml``, the ``osprey set`` lines, ``validate``,
``build`` -- and the assertions are the claims that sequence makes:

* **hello-world, middle layer.** The export enters the facility description
  past the stop ``osprey facility import mml`` makes over the preset's
  authored record sources, so the build writes the middle-layer index and its
  DuckDB copy from the imported groups. The export reads one corrector
  readback for two setpoints, which the import pairs with neither, so the
  build runs clean. The rendered config binds the index and its copy, and
  ``run_sql`` answers from the copy. The set line never spells
  ``channel_finder.pipelines.*``: those keys are build-derived and ``validate``
  refuses a profile that states them.
* **control-assistant, from a 2.0 export.** The recipe over an export that
  carries a model, once per supported tree. ``osprey facility import mml``
  stops over the preset's authored record sources and prints one ``rm`` line
  per file, the recipe removes exactly those, installs the tree's reviewed
  ``imported/mml/mapping.yaml`` and imports. The import then lists, as ``rm``
  lines, the demo scenarios it leaves stale -- each names a channel or model
  the imported facility does not have -- and the recipe removes exactly
  those, since the build stops while one is left. The seeded ``limits.yaml``
  holds each band as the export states it, so the build then stops
  ``seed-invalid`` while a setpoint starts outside its band; ``facility
  validate`` names each one, and the recipe widens exactly the records those
  lines name before it builds. The claims are about what ``osprey build``
  then published: the simulator view serves the import's channels and model,
  and a second build changes no byte of it.

Every number here is read off a real run of the real verbs. The chain is cheap
enough (seconds) to drive once per recipe, so nothing about the rendered tree
is restated from a plan.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from tests.fixtures.mml._trees import names

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "mml"
PACKAGED_FACILITY = _REPO_ROOT / "src/osprey/templates/facilities/example"

#: The export the hello-world recipe imports: a paired ``ao``/``ad`` synthetic machine.
SOURCE = FIXTURES / "paired"
AO_INPUT = "quokka.ring.ao.json"
AD_INPUT = "quokka.ring.ad.json"

#: Where the build writes the middle-layer index and its DuckDB copy, relative
#: to the render.
INDEX_PATH = "data/channel_finder/middle_layer.json"
INDEX_DUCKDB_PATH = "data/channel_finder/middle_layer.duckdb"

#: The middle-layer paradigm card: the paradigm, the subagent and its server.
#: No ``channel_finder.pipelines.*`` key -- see this module's docstring.
MIDDLE_LAYER_SETTINGS = (
    "channel_finder_mode=middle_layer",
    "agents=[channel-finder]",
    "config.claude_code.servers.channel-finder.enabled=true",
)

#: The supported trees ``osprey build`` is claimed to accept, from the registry.
BUILT_TREES = names("builds")

LIMITS_FILE = "channel_limits.json"

#: The simulator view the build writes and the virtual accelerator serves,
#: relative to the repo, and the two files of it these recipes read.
SIMULATOR_VIEW = "build/data/simulator"
ADDRESSES_FILE = "addresses.json"
SERVED_MODELS_FILE = "served_models.json"


#: The slots a facility scenario names a channel or a model in.
_RESOLVING_SLOTS = ("overrides", "faults", "archiver", "couple", "noise")


def _resolving_scenarios() -> tuple[str, ...]:
    """The preset's facility scenarios that name a channel or a model, as repo paths.

    Read off the packaged preset, so a demo scenario added later is counted
    without a name being typed here. A scenario stating none of these slots
    names nothing an import can take away. A scenario's folder of attached
    files follows its file, as the import lists it.
    """
    found: list[str] = []
    for path in sorted((PACKAGED_FACILITY / "scenarios").glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if any(slot in document for slot in _RESOLVING_SLOTS):
            found.append(f"data/facility/scenarios/{path.name}")
            if path.with_suffix("").is_dir():
                found.append(f"data/facility/scenarios/{path.stem}/")
    return tuple(found)


DEMO_RESOLVING_SCENARIOS = _resolving_scenarios()

#: The facility description of a deployment, and the limits file an import
#: seeds into it, both relative to the repo.
FACILITY_DIR = "data/facility"
FACILITY_LIMITS = f"{FACILITY_DIR}/limits.yaml"

#: The first line of the stop ``facility import mml`` prints over authored
#: record sources; one ``rm`` line per file follows it.
AUTHORED_PRESENT = "import mml: authored-present: "

#: The header line ``facility import mml`` prints over the scenario files a
#: clean import leaves stale; one ``  rm`` line per file follows it.
STALE_SCENARIOS = "these scenario files name channels that no longer exist:"

#: The line a build stage prints for a setpoint that starts outside its band:
#: the address, the nominal it starts at, the side it lies on and the edge of
#: the limits record it lies beyond.
SEED_INVALID = re.compile(
    r"^facility: seed-invalid: channel (?P<address>.+?) — nominal (?P<nominal>\S+) lies "
    r"(?P<side>above|below) `(?P<edge>min_value|max_value)` \S+; "
    r"fix: .*widen the limits record$"
)

#: The first words of every line the response check prints.
RESPONSE_CHECK = "response check "

#: The first words of every note a written view prints on a clean run.
VIEW_NOTE = "  view "


def invoke(runner: CliRunner, *args: str) -> Result:
    """Run one ``osprey`` verb and insist it succeeded.

    A recipe is a chain: once a step fails every later assertion is about a
    tree nobody built, so each call carries its own output into the failure.
    """
    result = runner.invoke(cli, list(args), catch_exceptions=False)
    assert result.exit_code == 0, f"osprey {' '.join(args)} failed:\n{result.output}"
    return result


def stage_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy the fixture export somewhere writable and return its ``ao`` file.

    Out of the repo's ``tests/fixtures`` tree so nothing a recipe runs can
    write back into committed fixture data, and out of the deployment repo so
    the export is not mistaken for something the deployment ships.
    """
    inputs = tmp_path_factory.mktemp("mml-export")
    for name in (AO_INPUT, AD_INPUT):
        shutil.copy(SOURCE / name, inputs / name)
    return inputs / AO_INPUT


def facility_mapping(fixture: Path) -> Path:
    """The reviewed mapping ``facility import mml`` reads, committed beside a tree."""
    from osprey.facility.layers.mml.mapping import MAPPING_FILE

    return fixture / MAPPING_FILE


def clear_authored(runner: CliRunner, repo: Path, exports: Sequence[str]) -> tuple[str, ...]:
    """Obey the stop ``facility import mml`` makes over a preset's authored sources.

    The verb stops before it reads an export while an authored record source is
    present, and prints the ``rm`` line of each. Every path removed here was
    named by one of those lines, one path per line.

    Returns:
        The removed paths, in the order they were printed.
    """
    stopped = runner.invoke(
        cli, ["facility", "import", "mml", *exports, "--repo", str(repo)], catch_exceptions=False
    )
    assert stopped.exit_code == 1, stopped.output
    lines = stopped.stderr.splitlines()
    assert lines and lines[0].startswith(AUTHORED_PRESENT), stopped.stderr
    assert all(line.startswith("rm ") for line in lines[1:]), stopped.stderr

    removed: list[str] = []
    for line in lines[1:]:
        (named,) = remove_named(repo, line)
        removed.append(named)
    return tuple(removed)


def import_facility(runner: CliRunner, repo: Path, exports: Sequence[str], mapping: Path) -> Result:
    """Install the reviewed mapping and write the exports as the mml layer's sources.

    Every export of a facility goes in one call, as the verb takes them.
    """
    from osprey.facility.layers.mml.mapping import MAPPING_FILE

    target = repo / FACILITY_DIR / MAPPING_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mapping, target)
    return invoke(runner, "facility", "import", "mml", *exports, "--repo", str(repo))


def imported_probe(repo: Path) -> str:
    """The first readback by address of the import's channels.

    The rule the build's served-probe stop names its remedy by: a channel whose
    role is ``readback`` or states none.
    """
    channels = yaml.safe_load(
        (repo / FACILITY_DIR / "imported/mml/channels.yaml").read_text(encoding="utf-8")
    )
    return min(
        str(channel["id"]) for channel in channels if channel.get("role", "readback") == "readback"
    )


def remove_stale_scenarios(repo: Path, imported: Result) -> tuple[str, ...]:
    """Remove exactly the scenario files a clean ``facility import mml`` listed.

    Nothing here decides what to delete: every path removed was named by one
    ``  rm`` line under the list's header (``rm -r`` for a scenario's folder of
    attached files), one path per line.

    Returns:
        The removed paths, in the order they were printed; empty when the
        import listed none.
    """
    lines = imported.stderr.splitlines()
    if STALE_SCENARIOS not in lines:
        return ()
    listed = lines[lines.index(STALE_SCENARIOS) + 1 :]
    assert listed and all(line.startswith("  rm ") for line in listed), imported.stderr

    removed: list[str] = []
    for line in listed:
        (named,) = remove_named(repo, line.strip())
        removed.append(named)
    return tuple(removed)


def seed_stops(stderr: str) -> dict[str, tuple[str, float]]:
    """Every ``seed-invalid`` line of *stderr*: address -> the edge and the stated nominal."""
    stops: dict[str, tuple[str, float]] = {}
    for line in stderr.splitlines():
        match = SEED_INVALID.match(line)
        if match is None:
            continue
        expected_edge = "max_value" if match["side"] == "above" else "min_value"
        assert match["edge"] == expected_edge, line
        stops[match["address"]] = (match["edge"], float(match["nominal"]))
    return stops


def widen_limits_record(limits: Path, address: str, edge: str, nominal: float) -> None:
    """Move one edge of one limits record out to hold *nominal*; change no other line.

    The stop states the nominal to six significant digits, so the edge is set to
    the whole number at or beyond it rather than to the stated digits.
    """
    value = float(math.ceil(nominal) if edge == "max_value" else math.floor(nominal))
    lines = limits.read_text(encoding="utf-8").split("\n")
    starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith("- address:") and yaml.safe_load(line[2:])["address"] == address
    ]
    assert len(starts) == 1, f"{address} has {len(starts)} limits records in {limits}"
    end = next(
        (index for index in range(starts[0] + 1, len(lines)) if not lines[index].startswith("  ")),
        len(lines),
    )
    edges = [index for index in range(starts[0] + 1, end) if lines[index].startswith(f"  {edge}:")]
    assert len(edges) == 1, f"{address} states {edge} {len(edges)} times in {limits}"
    lines[edges[0]] = f"  {edge}: {value}"
    limits.write_text("\n".join(lines), encoding="utf-8")


def apply_seed_invalid_remedies(
    runner: CliRunner, repo: Path, *, responses: Sequence[str] | None = None
) -> tuple[str, ...]:
    """Widen the limits record of every ``seed-invalid`` stop the tree prints.

    ``osprey facility validate`` runs the build's own stages and writes nothing,
    so its stderr is where the stops are read. Each line names one address, and
    exactly that record of the deployment's ``data/facility/limits.yaml`` is
    widened to hold the nominal the line states. The verb then runs once more,
    render included, and must print the tree's build warnings and response-check
    lines and nothing else; a note a written view prints about its own index is that
    view's fact, asserted by its own tests, and is not read here.

    Args:
        runner: The CLI runner.
        repo: The deployment repo.
        responses: The lines the clean run prints, in order (build warnings, then
            response-check lines); when omitted, every line it prints must be a
            response-check line.

    Returns:
        The widened addresses, in the order the stops were printed.
    """
    where = ["facility", "validate", "--repo", str(repo)]
    stopped = runner.invoke(cli, where, catch_exceptions=False)
    stops = seed_stops(stopped.stderr)
    if stops:
        assert stopped.exit_code == 1, stopped.output
        assert len(stops) == len(stopped.stderr.splitlines()), stopped.stderr
    for address, (edge, nominal) in stops.items():
        widen_limits_record(repo / FACILITY_LIMITS, address, edge, nominal)

    clean = runner.invoke(cli, where, catch_exceptions=False)
    assert clean.exit_code == 0, clean.output
    printed = [line for line in clean.stderr.splitlines() if not line.startswith(VIEW_NOTE)]
    if responses is None:
        assert all(line.startswith(RESPONSE_CHECK) for line in printed), clean.stderr
    else:
        assert printed == list(responses), clean.stderr
    return tuple(stops)


def expected_seed_stops(tree: str) -> frozenset[str]:
    """The setpoints a fixture tree's build stops on once its exports are imported.

    The synthetic tree plants one corrector outside its band; the other trees'
    stops are the records their build case widens. Both are read from the
    seed-once module, so a wiring change that removes a stop fails there too.
    """
    from tests.facility.test_mml_layer_seed_once import OUTSIDE, WIDENED

    if tree == "synthetic":
        return frozenset({OUTSIDE})
    return frozenset(WIDENED.get(tree, {}))


def expected_response_lines(tree: str) -> tuple[str, ...]:
    """The lines a fixture tree's clean ``facility validate`` prints: its build warnings, then its response-check lines."""
    from tests.facility.test_response_check import (
        NSLS2_LINES,
        SPEAR3_LINE,
        SPEAR3_WRAPPED_LINES,
        SYNTHETIC_LINE,
    )

    return {
        "nsls2": (NSLS2_LINES[0], NSLS2_LINES[1]),
        "spear3": (*SPEAR3_WRAPPED_LINES, SPEAR3_LINE),
        "synthetic": (SYNTHETIC_LINE,),
    }[tree]


def build_past_the_seed_stops(
    runner: CliRunner, repo: Path, *, responses: Sequence[str] | None = None
) -> tuple[Result, tuple[str, ...], Result]:
    """Run ``osprey build`` into its ``seed-invalid`` stops, remedy them and build.

    Returns:
        The stopped build (or the clean one, on a tree with no stop), the
        addresses whose limits records were widened, and the build that passed.
    """
    arguments = ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"]
    stopped = runner.invoke(cli, arguments, catch_exceptions=False)
    remedied = apply_seed_invalid_remedies(runner, repo, responses=responses)
    if not remedied:
        assert stopped.exit_code == 0, stopped.output
        return stopped, remedied, stopped
    assert stopped.exit_code != 0, stopped.output
    return stopped, remedied, invoke(runner, *arguments)


def simulator_view(repo: Path) -> dict[str, dict]:
    """The simulator view's addresses and served models, as the container reads them."""
    view = repo / SIMULATOR_VIEW
    return {
        name: json.loads((view / name).read_text(encoding="utf-8"))
        for name in (ADDRESSES_FILE, SERVED_MODELS_FILE)
    }


def published(repo: Path) -> dict[str, bytes]:
    """Every file of the simulator view the build published, by its path under the view."""
    view = repo / SIMULATOR_VIEW
    return {
        path.relative_to(view).as_posix(): path.read_bytes()
        for path in sorted(view.rglob("*"))
        if path.is_file()
    }


def remove_named(repo: Path, line: str) -> list[str]:
    """Delete exactly the paths *line* names, and report them.

    Run as the operator would: the line is a shell command, so it is split as
    one. Nothing else in the tree is touched, which is what makes the
    assertions about what survived the refusal mean anything.
    """
    named = [word for word in shlex.split(line) if word not in ("rm", "-r")]
    assert named, line
    for relative in named:
        target = repo / relative
        assert target.exists(), f"{relative} was named but is not there"
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return named


def rendered_config(repo: Path) -> dict:
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def middle_layer_repo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The hello-world middle-layer recipe, driven once.

    Module-scoped: the recipe is one story, and re-running an ``osprey build``
    per assertion would say nothing the first one did not.
    """
    pytest.importorskip("duckdb")

    runner = CliRunner()
    export = stage_export(tmp_path_factory)
    repo = tmp_path_factory.mktemp("recipe-middle-layer") / "demo"

    invoke(runner, "init", str(repo), "--preset", "hello-world", "--no-git")
    exports = [str(export)]
    clear_authored(runner, repo, exports)
    import_facility(runner, repo, exports, facility_mapping(SOURCE))
    invoke(runner, "set", "--repo", str(repo), *MIDDLE_LAYER_SETTINGS)

    validate = invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    build = invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    return {
        "repo": repo,
        "validate": validate.output,
        "build": build.output,
    }


@pytest.fixture(scope="module", params=BUILT_TREES)
def served_repo(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """The control-assistant recipe over a 2.0 export, driven once per supported tree.

    The exports enter the facility description through ``facility import
    mml``, past the stop it makes over the preset's authored sources. The
    import lists the demo scenarios it leaves stale; one build runs while they
    are still there, and the recipe then removes exactly those. The first
    build after that runs into the ``seed-invalid`` stops the imported tree
    carries; ``build_past_the_seed_stops`` applies their remedy.

    The build then runs twice. The second run is what says the published tree
    is a function of the import and not of the run that wrote it.
    """
    fixture = FIXTURES / request.param
    exports = sorted(str(path) for path in fixture.glob("*.ao.json"))
    assert exports, f"{request.param} commits no export"

    runner = CliRunner()
    repo = tmp_path_factory.mktemp(f"recipe-served-{request.param}") / "demo"

    invoke(runner, "init", str(repo), "--preset", "control-assistant", "--no-git")
    cleared = clear_authored(runner, repo, exports)
    imported = import_facility(runner, repo, exports, facility_mapping(fixture))
    invoke(
        runner,
        "set",
        "--repo",
        str(repo),
        f"config.control_system.connector.virtual_accelerator.probe_channel={imported_probe(repo)}",
    )
    scenario_stop = runner.invoke(
        cli,
        ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
        catch_exceptions=False,
    )
    stale = remove_stale_scenarios(repo, imported)
    invoke(runner, "set", "--repo", str(repo), *MIDDLE_LAYER_SETTINGS)

    validate = invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    stopped, remedied, build = build_past_the_seed_stops(
        runner, repo, responses=expected_response_lines(request.param)
    )
    first = published(repo)
    first_view = simulator_view(repo)
    invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    return {
        "fixture": request.param,
        "repo": repo,
        "cleared": cleared,
        "scenario_stop": scenario_stop,
        "stale_scenarios": stale,
        "stopped": stopped,
        "remedied": remedied,
        "validate": validate.output,
        "build": build.output,
        "first": first,
        "first_view": first_view,
    }


class TestHelloWorldMiddleLayer:
    def test_validate_and_build_accept_the_recipe(self, middle_layer_repo: dict) -> None:
        assert "Profile is valid" in middle_layer_repo["validate"]
        assert (middle_layer_repo["repo"] / "build" / "config.yml").is_file()

    def test_the_build_binds_the_index_and_database_it_writes(
        self, middle_layer_repo: dict
    ) -> None:
        database = rendered_config(middle_layer_repo["repo"])["channel_finder"]["pipelines"][
            "middle_layer"
        ]["database"]
        build = middle_layer_repo["repo"] / "build"

        assert (database["path"], database["duckdb_path"]) == (INDEX_PATH, INDEX_DUCKDB_PATH)
        assert (build / INDEX_PATH).is_file()
        assert (build / INDEX_DUCKDB_PATH).is_file()

    def test_run_sql_answers_from_the_database_the_build_writes(
        self, middle_layer_repo: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import osprey.utils.config as config
        from osprey.mcp_server.channel_finder_middle_layer.server_context import (
            initialize_cf_ml_context,
            reset_cf_ml_context,
        )
        from osprey.mcp_server.channel_finder_middle_layer.tools.run_sql import run_sql
        from osprey.utils.workspace import reset_config_cache

        def reset() -> None:
            reset_cf_ml_context()
            reset_config_cache()
            config._default_config = None
            config._default_configurable = None
            config._config_cache.clear()

        monkeypatch.setenv("OSPREY_CONFIG", str(middle_layer_repo["repo"] / "build" / "config.yml"))
        reset()
        try:
            context = initialize_cf_ml_context()
            answer = json.loads(
                getattr(run_sql, "fn", run_sql)(
                    sql="SELECT count(DISTINCT channel_name) AS n FROM channels"
                )
            )
            channels = len(context.database.channel_map)
        finally:
            reset()

        assert channels > 0
        assert answer["rows"] == [{"n": channels}]

    def test_the_profile_states_no_build_derived_pipeline_key(
        self, middle_layer_repo: dict
    ) -> None:
        """The recipe's set line stops at the paradigm; the build derives the rest."""
        profile = (middle_layer_repo["repo"] / "profile.yml").read_text(encoding="utf-8")

        assert "channel_finder.pipelines" not in profile
        assert "channel_finder.pipeline_mode" not in profile
        assert (
            rendered_config(middle_layer_repo["repo"])["channel_finder"]["pipeline_mode"]
            == "middle_layer"
        )


class TestServedFromATwoZeroExport:
    """What ``osprey build`` publishes when the import brought a model.

    Everything asserted here is read back off the published tree: the recipe's
    claim is that an operator who typed these lines gets a container mounting
    their own machine, and the only evidence for it is the files the build
    actually wrote.
    """

    def test_a_supported_tree_is_claimed_to_build(self) -> None:
        # Every case below is parametrised over the registry, so a registry
        # that stopped claiming a build would empty them all silently rather
        # than fail.
        assert BUILT_TREES, "no supported tree is claimed to build"

    def test_the_view_serves_the_imports_channels_and_model(self, served_repo: dict) -> None:
        """The container reads the import's addresses and serves a physics model."""
        view = served_repo["first_view"]

        assert view[ADDRESSES_FILE]["channels"]
        assert [name for name in view[SERVED_MODELS_FILE]["models"] if name != "texture"]

    def test_a_second_build_changes_no_published_byte(self, served_repo: dict) -> None:
        """The served tree is a function of the import, not of the run.

        An operator rebuilding for an unrelated reason must not hand the IOC a
        different machine, so every file of the simulator view is compared
        whole against the first build's.
        """
        repo = served_repo["repo"]

        assert published(repo) == served_repo["first"]
        assert simulator_view(repo) == served_repo["first_view"]


class TestTheFacilityImportOfATwoZeroExport:
    """What ``facility import mml`` asks of the preset, and what the build then stops on.

    Read off the ``served_repo`` recipe above.
    """

    def test_the_import_stops_over_every_authored_record_source_of_the_preset(
        self, served_repo: dict
    ) -> None:
        """One ``rm`` line per file, and never ``classes.yaml``.

        The preset ships no ``classes.yaml``, so the import seeds the tree's
        own classes and the build reaches the limits records rather than
        stopping on a class nothing declares.
        """
        cleared = served_repo["cleared"]
        facility = served_repo["repo"] / FACILITY_DIR

        assert cleared == tuple(sorted(cleared))
        assert f"{FACILITY_DIR}/records/channels.yaml" in cleared
        assert all(path.startswith(f"{FACILITY_DIR}/") for path in cleared)
        assert [path for path in cleared if path.startswith(f"{FACILITY_DIR}/scenarios/")] == []
        assert not (PACKAGED_FACILITY / "classes.yaml").exists()
        assert (facility / "classes.yaml").is_file()
        assert (facility / "imported" / "mml" / "channels.yaml").is_file()

    def test_the_import_lists_the_stale_demo_scenarios_and_the_build_stops_until_they_are_gone(
        self, served_repo: dict
    ) -> None:
        """The listed files are the demo's channel-naming ones, and only they stop the build.

        ``nominal.yaml`` names nothing the import takes away, so it is never
        listed and stays.
        """
        listed = served_repo["stale_scenarios"]
        names = {Path(path).stem for path in listed}
        scenario_stop = served_repo["scenario_stop"]
        stops = [
            line for line in scenario_stop.stderr.splitlines() if line.startswith("facility: ")
        ]

        assert listed == DEMO_RESOLVING_SCENARIOS
        assert scenario_stop.exit_code != 0
        assert stops and " scenario " in stops[0], scenario_stop.output
        assert any(f" scenario {name} " in stops[0] for name in names), stops[0]
        assert not [
            line
            for line in served_repo["stopped"].stderr.splitlines()
            if line.startswith("facility: ") and " scenario " in line
        ]
        assert (served_repo["repo"] / FACILITY_DIR / "scenarios" / "nominal.yaml").is_file()

    @pytest.mark.parametrize("served_repo", ["spear3", "synthetic"], indirect=True)
    def test_the_build_stops_on_each_setpoint_outside_its_band_until_it_is_widened(
        self, served_repo: dict
    ) -> None:
        """The stops are the tree's own, and the remedy touched exactly those records.

        ``osprey build`` stops on the first of them; ``facility validate``
        prints them all, and that is the set the remedy was read from.
        """
        expected = expected_seed_stops(served_repo["fixture"])
        stopped = served_repo["stopped"]
        named = set(seed_stops(stopped.stderr))

        assert expected, f"{served_repo['fixture']} plants no stop to pass"
        assert stopped.exit_code != 0
        assert named and named <= expected
        assert len(served_repo["remedied"]) == len(expected)
        assert set(served_repo["remedied"]) == expected

    @pytest.mark.parametrize("served_repo", ["nsls2"], indirect=True)
    def test_a_tree_whose_export_starts_every_setpoint_inside_its_band_builds_at_once(
        self, served_repo: dict
    ) -> None:
        """No seed stop, and the remedy widens nothing."""
        stopped = served_repo["stopped"]

        assert not expected_seed_stops(served_repo["fixture"])
        assert stopped.exit_code == 0, stopped.output
        assert not seed_stops(stopped.stderr)
        assert served_repo["remedied"] == ()

    @pytest.mark.parametrize("served_repo", ["synthetic"], indirect=True)
    def test_the_synthetic_harvest_stops_on_its_one_planted_corrector(
        self, served_repo: dict
    ) -> None:
        """One line, naming the corrector the export starts outside its own ``Range``."""
        (address,) = expected_seed_stops("synthetic")
        lines = [
            line
            for line in served_repo["stopped"].stderr.splitlines()
            if line.startswith("facility: ")
        ]

        assert lines == [
            f"facility: seed-invalid: channel {address} — nominal 1.5 lies above `max_value` 1; "
            "fix: move the operating point inside [min_value, max_value], or widen the limits "
            "record"
        ]
        assert served_repo["remedied"] == (address,)
        limits = yaml.safe_load((served_repo["repo"] / FACILITY_LIMITS).read_text(encoding="utf-8"))
        (record,) = [row for row in limits["records"] if row["address"] == address]
        assert record == {
            "address": address,
            "min_value": -1.0,
            "max_value": 2.0,
            "writable": True,
        }

    def test_the_remedy_changes_no_other_line_of_the_seeded_limits(
        self, served_repo: dict, tmp_path: Path
    ) -> None:
        """A fresh import of the same exports differs only in the widened edges."""
        from osprey.facility.layers.mml.importer import import_mml
        from osprey.facility.layers.mml.mapping import MAPPING_FILE
        from osprey.facility.layers.mml.seed import HEADER

        fixture = FIXTURES / served_repo["fixture"]
        fresh = tmp_path / FACILITY_DIR
        (fresh / MAPPING_FILE).parent.mkdir(parents=True)
        shutil.copyfile(facility_mapping(fixture), fresh / MAPPING_FILE)
        import_mml(sorted(fixture.glob("*.ao.json")), fresh)

        seeded = (fresh / "limits.yaml").read_text(encoding="utf-8").splitlines()
        remedied = (served_repo["repo"] / FACILITY_LIMITS).read_text(encoding="utf-8").splitlines()

        assert remedied[0] == HEADER
        assert len(remedied) == len(seeded)
        changed = [index for index, line in enumerate(seeded) if line != remedied[index]]
        assert len(changed) == len(served_repo["remedied"])
        assert all(
            seeded[index].lstrip().startswith(("min_value:", "max_value:")) for index in changed
        )


class TestTheBandsOfAHarvestedTree:
    """The bands ``osprey build`` publishes on the tree the import leaves.

    They read the ``served_repo`` build above rather than driving the chain again.
    """

    def test_the_served_bands_are_the_facility_limits_records(self, served_repo: dict) -> None:
        """What the container reads is the limits view of the facility description.

        The build writes ``channel_limits.json`` from ``data/facility/limits.yaml``,
        so the bands are the records the import seeded and the remedy widened: one entry per
        record, with that record's bounds, and no entry the records do not hold.
        """
        from osprey.facility.views.limits import limits_document

        repo = served_repo["repo"]
        records = yaml.safe_load((repo / FACILITY_LIMITS).read_text(encoding="utf-8"))["records"]
        facility = json.loads((repo / "build" / "facility.json").read_text(encoding="utf-8"))
        bands = json.loads((repo / "build" / "data" / LIMITS_FILE).read_text(encoding="utf-8"))

        assert records
        assert bands == limits_document(facility)
        assert sorted(key for key in bands if not key.startswith("_")) == sorted(
            record["address"] for record in records
        )
        for record in records:
            for bound in ("min_value", "max_value"):
                assert bands[record["address"]].get(bound) == record.get(bound)
