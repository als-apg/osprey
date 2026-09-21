"""The three install recipes an MML harvest ends in, run end to end.

A harvest is only finished when ``osprey build`` accepts what ``osprey mml
emit`` wrote, so each recipe here is the literal sequence the install skill
tells an operator to type -- ``init``, the chain, one ``osprey set`` line,
``validate``, ``build`` -- and the assertions are the claims that sequence
makes:

* **hello-world, middle layer.** The emitted channel database is the one the
  rendered config binds, and a ``--duckdb`` emit is what puts ``duckdb_path``
  beside it. The set line never spells ``channel_finder.pipelines.*``: those
  keys are build-derived and ``validate`` refuses a profile that states them.
* **hello-world, graph.** The same emit feeds the other paradigm through a
  single ``services.graphdb.ttl_path`` key, with no channel-finder wiring.
* **control-assistant.** The preset ships demo material emit would contradict,
  so the first ``emit`` refuses with one ``rm`` line naming the pages and tier
  databases, and this recipe removes exactly what that line names and nothing
  else. The ring the preset shipped is not refused but taken: a 1.0 export
  describes no machine, so the tree is left serving none, while the preset
  nobody harvested onto keeps and serves its own. Its demo scenarios stay,
  because the machine that resolves them stays too. Afterwards the flat database,
  the tier-3 copy and the built copy are one file -- the assertion that goes
  red if the build's tier materializer ever overwrites the emitted database
  with preset material -- and the demo knowledge pages are gone from the
  bundle index rather than merely unlinked.
* **control-assistant, from a 2.0 export.** The same recipe over an export that
  carries a virtual accelerator, which is the only harvest that ends in a tree
  the build can serve a model from. It runs one verb further and one refusal
  further still -- the demo's own machine documents, which only a harvest
  carrying a machine replaces -- and the claims it makes are about what ``osprey build`` then
  published: the manifest is partitioned by the harvest's own bindings, the
  ring and the bindings reach the served directory byte for byte, and the
  ``.env`` names that ring by its file name.

Every number here is read off a real run of the real verbs. The chain is cheap
enough (seconds) to drive once per recipe, so nothing about the rendered tree
is restated from a plan.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.main import cli

pytest.importorskip("linkml_runtime")

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "mml"
PACKAGED_DATA = _REPO_ROOT / "src" / "osprey" / "templates" / "apps" / "control_assistant" / "data"
PACKAGED_KNOWLEDGE = PACKAGED_DATA / "facility_knowledge"

#: The export every recipe harvests: a paired ``ao``/``ad`` synthetic machine.
SOURCE = FIXTURES / "paired"
AO_INPUT = "quokka.ring.ao.json"
AD_INPUT = "quokka.ring.ad.json"

#: ``facility.token`` of that fixture's committed mapping, which names the
#: corpus (``data/Quokka.ttl``) and the ontology schema.
TOKEN = "Quokka"

#: The container-name prefix the recipes state. Lowercase on purpose: it
#: reaches Docker object names, which the token's capital would not survive.
PREFIX = "quokka"

BUNDLE_PATH = "data/facility_knowledge"
ONTOLOGY_PATH = "data/facility_ontology.json"
DATABASE_PATH = "data/channel_databases/middle_layer.json"
DUCKDB_PATH = "data/channel_databases/middle_layer.duckdb"
TIERED_DATABASE = "data/channel_databases/tiers/tier3/middle_layer.json"

#: Tier the middle-layer paradigm derives when no profile pins one.
EXPECTED_TIER = 3

#: The tier databases the control-assistant preset ships that are not emit's
#: own ``tier3/middle_layer.json``; each must be named by the refusal.
DEMO_TIER_SIBLINGS = (
    "data/channel_databases/tiers/tier1/in_context.json",
    "data/channel_databases/tiers/tier3/hierarchical.json",
    "data/channel_databases/tiers/tier3/in_context.json",
)

#: Read off the packaged bundle rather than listed here: a demo page directory
#: added to the preset must show up in the refusal without editing this test.
DEMO_KNOWLEDGE_DIRS = tuple(
    f"{BUNDLE_PATH}/{path.name}" for path in sorted(PACKAGED_KNOWLEDGE.iterdir()) if path.is_dir()
)

#: The middle-layer paradigm card: the paradigm, the subagent and its server,
#: and the three facility paths the emitted artifacts landed at. No
#: ``channel_finder.pipelines.*`` key -- see this module's docstring.
MIDDLE_LAYER_SETTINGS = (
    "channel_finder_mode=middle_layer",
    "agents=[channel-finder]",
    "config.claude_code.servers.channel-finder.enabled=true",
    f"config.facility_knowledge.bundle_path={BUNDLE_PATH}",
    f"config.facility.ontology={ONTOLOGY_PATH}",
    f"config.facility.prefix={PREFIX}",
)

#: The graph paradigm card: the corpus path replaces the whole channel-finder
#: wiring, because a graph deployment serves its channels from the seeded
#: store rather than a database file.
GRAPH_SETTINGS = (
    "channel_finder_mode=graph",
    f"config.services.graphdb.ttl_path=./data/{TOKEN}.ttl",
    f"config.facility_knowledge.bundle_path={BUNDLE_PATH}",
    f"config.facility.ontology={ONTOLOGY_PATH}",
    f"config.facility.prefix={PREFIX}",
)


#: The fixture exports that carry a virtual accelerator, discovered rather than
#: listed: a 2.0 export files its machine in a ``*.va.json`` sibling, so a
#: directory holding one is a harvest that ends in a tree the build can serve a
#: model from. A 2.0 re-export committed later joins the recipe below without a
#: name being typed here.
TWO_ZERO_TREES = tuple(
    sorted(
        directory.name
        for directory in FIXTURES.iterdir()
        if directory.is_dir() and any(directory.glob("*.va.json"))
    )
)

#: What the build derives into the deployment's ``.env`` once it has published
#: a manifest: the manifest's own name inside the mount, and the ring the tree
#: ties that channel set to.
MANIFEST_KEY = "VA_CHANNELS_FILE"
LATTICE_KEY = "VA_LATTICE"

MANIFEST_FILE = "channel_manifest.json"
LATTICE_FILE = "lattice.json"
BINDINGS_FILE = "va_bindings.json"
LIMITS_FILE = "channel_limits.json"

#: What ``VA_LATTICE`` says when the tree carries no model to steer.
LATTICE_NONE = "none"


def _packaged_scenarios() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The preset's scenario bundles, split by whether they name a channel.

    Read off the packaged preset through the command's own reading of a
    bundle, so a demo scenario added later lands on the right side of the split
    without a name being typed here. A bundle naming channels is one a harvest
    leaves nothing to resolve; a bundle naming none has nothing to go stale.
    """
    from osprey.cli.mml_cmd import _scenario_channels

    named: list[str] = []
    plain: list[str] = []
    for bundle in sorted((PACKAGED_DATA / "simulation" / "scenarios").iterdir()):
        if not bundle.is_dir():
            continue
        (named if _scenario_channels(bundle) else plain).append(bundle.name)
    return tuple(named), tuple(plain)


DEMO_CHANNELLED_SCENARIOS, DEMO_PLAIN_SCENARIOS = _packaged_scenarios()

#: Where the build publishes the served tree, relative to the repo.
SERVED = "build/data/simulation"


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


def harvest(runner: CliRunner, repo: Path, export: Path) -> None:
    """Import the export into *repo* and give it the fixture's checked mapping.

    Stands in for the reviewed ``osprey mml map --init`` pass: the mapping
    committed beside the fixture is what an operator reaches after filling and
    checking the skeleton, and it is what the recipes emit from.
    """
    invoke(runner, "mml", "import", str(export), "--repo", str(repo))
    shutil.copy(SOURCE / "mapping.yaml", repo / "data" / "mml" / "mapping.yaml")


def emit(runner: CliRunner, repo: Path, *args: str) -> Result:
    """Run ``osprey mml emit`` without judging its exit code."""
    return runner.invoke(cli, ["mml", "emit", "--repo", str(repo), *args], catch_exceptions=False)


def drive_emit(runner: CliRunner, repo: Path) -> tuple[Result, tuple[tuple[str, ...], ...]]:
    """Run ``emit`` until it accepts the tree, obeying each refusal as written.

    A preset that ships demo material is asked about it in stages -- the
    knowledge and tier material one pre-flight refuses over, the
    virtual-accelerator documents another -- so an operator following the
    instructions runs the verb again after each ``rm`` line. Nothing here
    decides what to delete: every path removed was named by the refusal that
    printed it, which is what makes the rounds evidence about the refusals
    rather than about this helper.

    Returns:
        The accepting run, and what each refusal named, in order.
    """
    rounds: list[tuple[str, ...]] = []
    for _ in range(5):
        result = emit(runner, repo)
        if result.exit_code == 0:
            return result, tuple(rounds)
        rounds.append(tuple(remove_named(repo, rm_line(result.output))))
    raise AssertionError(f"emit never accepted the tree; it refused over {rounds}")


def served_settings(prefix: str) -> tuple[str, ...]:
    """The middle-layer paradigm card, spelled for one fixture's own facility.

    Derived from the card above rather than retyped, so a key added there
    reaches every recipe that states the paradigm.
    """
    return tuple(
        f"config.facility.prefix={prefix}" if line.startswith("config.facility.prefix=") else line
        for line in MIDDLE_LAYER_SETTINGS
    )


def facility_prefix(fixture: Path) -> str:
    """The container-name prefix a fixture's reviewed mapping implies.

    Read off the mapping rather than listed per fixture: the token names the
    facility, and lowercase is what survives Docker object names.
    """
    document = yaml.safe_load((fixture / "mapping.yaml").read_text(encoding="utf-8"))
    return str(document["facility"]["token"]).lower()


def env_values(repo: Path) -> dict[str, str]:
    """The deployment ``.env`` the build appended its derived keys to."""
    from osprey.utils.dotenv import parse_dotenv_file

    return parse_dotenv_file(repo / ".env")


def served_manifest(repo: Path) -> dict:
    """The channel manifest the build published, as the container will read it."""
    import json

    return json.loads((repo / SERVED / MANIFEST_FILE).read_text(encoding="utf-8"))


def published(repo: Path) -> dict[str, bytes]:
    """Every file the build published into the served directory, by name."""
    directory = repo / SERVED
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def rm_line(output: str) -> str:
    """The single ``rm`` line a refusal prints, as one line."""
    lines = [line for line in output.splitlines() if line.startswith("rm ")]
    assert len(lines) == 1, f"expected exactly one rm line:\n{output}"
    return lines[0]


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rendered_config(repo: Path) -> dict:
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


def index_targets(index: Path) -> set[str]:
    """Every link target the bundle index lists."""
    return set(re.findall(r"\]\(([^)]+)\)", index.read_text(encoding="utf-8")))


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
    harvest(runner, repo, export)
    emitted = emit(runner, repo, "--duckdb")
    assert emitted.exit_code == 0, emitted.output
    invoke(runner, "set", "--repo", str(repo), *MIDDLE_LAYER_SETTINGS)

    validate = invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    build = invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    return {
        "repo": repo,
        "emit": emitted.output,
        "validate": validate.output,
        "build": build.output,
    }


@pytest.fixture(scope="module")
def graph_repo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The hello-world graph recipe, driven once."""
    runner = CliRunner()
    export = stage_export(tmp_path_factory)
    repo = tmp_path_factory.mktemp("recipe-graph") / "demo"

    invoke(runner, "init", str(repo), "--preset", "hello-world", "--no-git")
    harvest(runner, repo, export)
    emitted = emit(runner, repo)
    assert emitted.exit_code == 0, emitted.output
    invoke(runner, "set", "--repo", str(repo), *GRAPH_SETTINGS)

    validate = invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    build = invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    return {
        "repo": repo,
        "validate": validate.output,
        "build": build.output,
    }


@pytest.fixture(scope="module")
def control_assistant_repo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The control-assistant recipe, including the refusal it must pass through."""
    runner = CliRunner()
    export = stage_export(tmp_path_factory)
    repo = tmp_path_factory.mktemp("recipe-control-assistant") / "demo"

    invoke(runner, "init", str(repo), "--preset", "control-assistant", "--no-git")
    harvest(runner, repo, export)

    refused = emit(runner, repo)
    removed = remove_named(repo, rm_line(refused.output))

    # The first refusal is kept whole above, as the evidence the demo-material
    # cases read; what the preset is asked about after it is driven the way an
    # operator would, one refusal at a time.
    emitted, rounds = drive_emit(runner, repo)
    invoke(runner, "set", "--repo", str(repo), *MIDDLE_LAYER_SETTINGS)

    validate = invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    build = invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    return {
        "repo": repo,
        "refused": refused,
        "removed": removed,
        "rounds": rounds,
        "emit": emitted.output,
        "validate": validate.output,
        "build": build.output,
    }


@pytest.fixture(scope="module")
def demo_repo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The control-assistant preset built as it ships, with no harvest at all.

    The other side of the rule the recipe above pins: what displaces a
    deployment's ring is a harvest that describes no machine, so a deployment
    nobody harvested onto keeps the ring the preset shipped and serves it.
    """
    runner = CliRunner()
    repo = tmp_path_factory.mktemp("recipe-demo") / "demo"

    invoke(runner, "init", str(repo), "--preset", "control-assistant", "--no-git")
    validate = invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    build = invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    return {"repo": repo, "validate": validate.output, "build": build.output}


@pytest.fixture(scope="module", params=TWO_ZERO_TREES)
def served_repo(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """The control-assistant recipe over a 2.0 export, driven once per fixture.

    The same verbs as the recipe above, over the one kind of export that ends
    in a tree with a machine in it: ``import`` picks up the deck, the machine
    and the response matrix beside the file it is handed, the reviewed mapping
    committed with the fixture answers the block, and ``emit`` writes the ring
    and the bindings the build then publishes.

    The build runs twice. The second run is what says the published tree is a
    function of the harvest and not of the run that wrote it.
    """
    fixture = FIXTURES / request.param
    exports = sorted(str(path) for path in fixture.glob("*.ao.json"))
    assert exports, f"{request.param} commits no export"

    runner = CliRunner()
    repo = tmp_path_factory.mktemp(f"recipe-served-{request.param}") / "demo"

    invoke(runner, "init", str(repo), "--preset", "control-assistant", "--no-git")
    invoke(runner, "mml", "import", *exports, "--repo", str(repo))
    shutil.copy(fixture / "mapping.yaml", repo / "data" / "mml" / "mapping.yaml")

    emitted, rounds = drive_emit(runner, repo)
    invoke(runner, "set", "--repo", str(repo), *served_settings(facility_prefix(fixture)))

    validate = invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    build = invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")
    first = published(repo)
    first_env = env_values(repo)
    invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    return {
        "fixture": request.param,
        "repo": repo,
        "rounds": rounds,
        "emit": emitted.output,
        "validate": validate.output,
        "build": build.output,
        "first": first,
        "first_env": first_env,
    }


class TestHelloWorldMiddleLayer:
    def test_validate_and_build_accept_the_recipe(self, middle_layer_repo: dict) -> None:
        assert "Profile is valid" in middle_layer_repo["validate"]
        assert (middle_layer_repo["repo"] / "build" / "config.yml").is_file()

    def test_the_build_binds_the_emitted_database(self, middle_layer_repo: dict) -> None:
        database = rendered_config(middle_layer_repo["repo"])["channel_finder"]["pipelines"][
            "middle_layer"
        ]["database"]

        assert database["path"] == DATABASE_PATH
        assert (middle_layer_repo["repo"] / "build" / DATABASE_PATH).is_file()

    def test_a_duckdb_emit_puts_duckdb_path_in_the_rendered_config(
        self, middle_layer_repo: dict
    ) -> None:
        database = rendered_config(middle_layer_repo["repo"])["channel_finder"]["pipelines"][
            "middle_layer"
        ]["database"]

        assert database["duckdb_path"] == DUCKDB_PATH
        assert (middle_layer_repo["repo"] / DUCKDB_PATH).is_file()

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

    def test_the_build_resolves_to_tier_three(self, middle_layer_repo: dict) -> None:
        assert f"tier {EXPECTED_TIER}" in middle_layer_repo["build"]


class TestHelloWorldGraph:
    def test_validate_and_build_accept_the_recipe(self, graph_repo: dict) -> None:
        assert "Profile is valid" in graph_repo["validate"]
        assert (graph_repo["repo"] / "build" / "config.yml").is_file()

    def test_the_build_binds_the_emitted_corpus(self, graph_repo: dict) -> None:
        config = rendered_config(graph_repo["repo"])

        assert config["services"]["graphdb"]["ttl_path"] == f"./data/{TOKEN}.ttl"
        assert (graph_repo["repo"] / "build" / "data" / f"{TOKEN}.ttl").is_file()

    def test_the_graph_paradigm_renders_no_middle_layer_pipeline(self, graph_repo: dict) -> None:
        """One key chose the paradigm, so the file-database wiring must be absent."""
        pipelines = (rendered_config(graph_repo["repo"]).get("channel_finder") or {}).get(
            "pipelines"
        ) or {}

        assert "middle_layer" not in pipelines


class TestControlAssistant:
    def test_the_first_emit_refuses_naming_every_demo_artifact(
        self, control_assistant_repo: dict
    ) -> None:
        refused = control_assistant_repo["refused"]

        assert refused.exit_code != 0
        assert "Traceback" not in refused.output
        line = rm_line(refused.output)
        for demo in (*DEMO_KNOWLEDGE_DIRS, *DEMO_TIER_SIBLINGS):
            assert demo in line
        assert TIERED_DATABASE not in line

    def test_the_refusal_names_the_demo_page_directories_whole(
        self, control_assistant_repo: dict
    ) -> None:
        """Whole directories, so no demo sub-index outlives the pages under it."""
        named = control_assistant_repo["removed"]
        under_bundle = {item for item in named if item.startswith(f"{BUNDLE_PATH}/")}

        assert under_bundle == set(DEMO_KNOWLEDGE_DIRS)

    def test_removing_exactly_those_lets_the_emit_through(
        self, control_assistant_repo: dict
    ) -> None:
        repo = control_assistant_repo["repo"]

        assert "Profile is valid" in control_assistant_repo["validate"]
        assert (repo / DATABASE_PATH).is_file()
        assert (repo / "data" / f"{TOKEN}.ttl").is_file()

    def test_the_emitted_database_survives_the_build_unchanged(
        self, control_assistant_repo: dict
    ) -> None:
        """Flat, tiered and built copy are one file.

        The build's tier materializer copies ``tiers/tier3/<paradigm>.json``
        over the flat database. Emit dual-writes both, so the three agree --
        and this goes red the moment the materializer puts preset material
        where the harvest's own database belongs.
        """
        repo = control_assistant_repo["repo"]
        digests = {
            relative: sha256(repo / relative)
            for relative in (DATABASE_PATH, TIERED_DATABASE, f"build/{DATABASE_PATH}")
        }

        assert len(set(digests.values())) == 1, digests

    def test_the_build_materializes_the_tier_three_benchmark_queries(
        self, control_assistant_repo: dict
    ) -> None:
        assert (control_assistant_repo["repo"] / "build/data/benchmarks/queries.json").is_file()
        assert f"tier {EXPECTED_TIER}" in control_assistant_repo["build"]

    def test_the_bundle_index_lists_only_what_the_harvest_wrote(
        self, control_assistant_repo: dict
    ) -> None:
        """The demo pages are gone from the index, not merely unlinked from disk."""
        index = control_assistant_repo["repo"] / BUNDLE_PATH / "index.md"

        assert index_targets(index) == {"/facility.md", "/families/"}

    def test_every_advertised_concept_resolves_to_a_page(
        self, control_assistant_repo: dict
    ) -> None:
        from osprey.services.facility_knowledge.okf.bundle import OKFBundle

        bundle = control_assistant_repo["repo"] / BUNDLE_PATH
        concepts = OKFBundle(bundle).list_concepts()

        assert concepts
        for entry in concepts:
            assert (bundle / f"{entry.concept_id}.md").is_file(), entry.concept_id

    def test_a_one_zero_harvest_takes_the_presets_own_ring_out_of_the_tree(
        self, control_assistant_repo: dict
    ) -> None:
        """A 1.0 export describes no machine, so the tree is left serving none.

        The harvest re-answers the channel set, and the preset's ring answers
        the demo's: left in place it would be served over the facility's own
        addresses, a model of one machine reached through the names of
        another. So emit takes the deck and the bindings with it, names what it
        removed, and the build derives its lattice from the tree it is about to
        mount -- which now carries none.
        """
        repo = control_assistant_repo["repo"]
        emitted = " ".join(control_assistant_repo["emit"].split())

        for name in (LATTICE_FILE, BINDINGS_FILE):
            assert not (repo / "data" / "simulation" / name).exists(), name
            assert f"data/simulation/{name}" in emitted, name
        assert env_values(repo)[LATTICE_KEY] == LATTICE_NONE

    def test_the_build_names_no_lattice_twice_over(self, control_assistant_repo: dict) -> None:
        """Both lines of the build agree, because the tree gives one answer.

        Two questions are being answered, both called "is a lattice served".
        The stand-in gate asks it of the env chain -- what the profile and the
        rendered compose say -- and the env writer asks it of the tree about to
        be mounted. A harvested tree carries no ring for either to find, so an
        operator reading the build is told the same thing twice instead of
        being left to pick.
        """
        printed = " ".join(control_assistant_repo["build"].split())

        assert f"{LATTICE_KEY}={LATTICE_NONE}: no model to displace" in printed
        assert f"serves the lattice {LATTICE_FILE}" not in printed
        assert "serves no lattice" in printed

    def test_the_demo_scenarios_survive_because_the_demo_machine_does(
        self, control_assistant_repo: dict
    ) -> None:
        """A scenario is judged by the machine that will resolve it, not by the database.

        This harvest writes no machine -- a 1.0 export describes none -- so the
        deployment goes on serving the preset's own ``machine.json``, and every
        demo scenario still resolves against it. The channel database beside it
        is the facility's now, and says nothing about whether a scenario boots.
        So the recipe is refused once, over the pages and tier databases, and
        every demo bundle is still in the tree.
        """
        repo = control_assistant_repo["repo"]
        scenarios = repo / "data" / "simulation" / "scenarios"

        assert control_assistant_repo["rounds"] == ()
        assert {path.name for path in scenarios.iterdir() if path.is_dir()} == set(
            DEMO_CHANNELLED_SCENARIOS + DEMO_PLAIN_SCENARIOS
        )


class TestTheDemoNobodyHarvestedOnto:
    """The preset built as it ships: its ring is its own, and it keeps it."""

    def test_the_preset_still_serves_the_ring_it_shipped(self, demo_repo: dict) -> None:
        repo = demo_repo["repo"]
        packaged = PACKAGED_DATA / "simulation"
        assert (packaged / LATTICE_FILE).is_file(), "the preset ships no ring to keep"

        for name in (LATTICE_FILE, BINDINGS_FILE):
            assert (repo / "data" / "simulation" / name).read_bytes() == (
                packaged / name
            ).read_bytes(), name
        assert env_values(repo)[LATTICE_KEY] == LATTICE_FILE

    def test_the_demo_scenarios_are_all_still_there(self, demo_repo: dict) -> None:
        scenarios = demo_repo["repo"] / "data" / "simulation" / "scenarios"

        assert {path.name for path in scenarios.iterdir() if path.is_dir()} == set(
            DEMO_CHANNELLED_SCENARIOS + DEMO_PLAIN_SCENARIOS
        )


class TestServedFromATwoZeroExport:
    """What ``osprey build`` publishes when the harvest brought a machine.

    The chain's last verb hands the build a tree carrying a ring, the bindings
    that tie the harvested channels to it, and the bands those channels are
    driven within. Everything asserted here is read back off that published
    tree: the recipe's claim is that an operator who typed these lines gets a
    container mounting their own machine, and the only evidence for it is the
    files the build actually wrote.
    """

    def test_a_two_zero_export_is_committed(self) -> None:
        # Every case below is parametrised over the discovery, so a fixture
        # tree that stopped carrying a machine would empty them all silently
        # rather than fail.
        assert TWO_ZERO_TREES, "no fixture export carries a *.va.json sibling"

    def test_the_recipe_passes_through_three_refusals(self, served_repo: dict) -> None:
        """The demo's machine and its scenarios are refused, each on its own terms.

        The pages and tier databases go in the first ``rm`` line. The demo's
        own machine description and machine-state list go in the second,
        because they are found by a different pre-flight -- the one that asks
        what the virtual-accelerator lane can vouch for on this tree -- and
        what it asks of them is this command's provenance stamp. The demo's
        scenarios go in the third, and one at a time: a scenario is refused for
        naming a channel this harvest does not serve, which is a question about
        the channel set and can only be asked once that set is built. The
        demo's bindings carry the stamp (they were emitted), so they are
        replaced without being named, and the saved ring never carries one at
        all: it is a plain pyAT document, vouched for by the digest the
        bindings record.
        """
        rounds = served_repo["rounds"]

        assert len(rounds) == 3, rounds
        assert set(rounds[1]) == {
            "data/simulation/machine.json",
            "data/machine_state_channels.json",
        }
        assert set(rounds[2]) == {
            f"data/simulation/scenarios/{name}" for name in DEMO_CHANNELLED_SCENARIOS
        }

    def test_the_build_publishes_a_manifest_its_own_tree_backs(self, served_repo: dict) -> None:
        """Nothing the harvested tree needs is missing, and one database fed it.

        Asked of the source tree, which is the one the generator read. The
        built tree is where the answer is published, not where it is checked:
        a build prunes ``tiers/``, so the paradigm database the manifest was
        expanded from is deliberately not in the tree the container mounts --
        which is the whole reason the manifest has to ship.
        """
        from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths

        repo = served_repo["repo"]
        harvested = ManifestPaths(data_root=repo / "data")
        metadata = served_manifest(repo)["_metadata"]

        assert harvested.staged_paradigms == ("middle_layer",)
        assert harvested.missing_sources() == []
        assert metadata["source_paradigms"] == ["middle_layer"]
        assert sorted(metadata["absent_paradigms"]) == ["hierarchical", "in_context"]
        assert not ManifestPaths(data_root=repo / "build" / "data").tier_dir.exists()

    def test_the_manifest_is_partitioned_by_the_harvests_own_bindings(
        self, served_repo: dict
    ) -> None:
        """Every knob the model moves is one the harvest tied to an element.

        Not a count: the two sets are compared whole, so a binding onto an
        address the manifest does not serve, or a coupled channel no binding
        drives, is a difference rather than a number that still matches.
        """
        from osprey.services.virtual_accelerator.bindings import load_bindings, setpoints
        from osprey.services.virtual_accelerator.manifest.classify import (
            pyat_coupled_setpoint_addresses,
        )

        repo = served_repo["repo"]
        manifest = served_manifest(repo)
        bound = set(setpoints(load_bindings(repo / SERVED / BINDINGS_FILE)))

        assert bound
        assert manifest["_metadata"]["partition_source"] == f"simulation/{BINDINGS_FILE}"
        assert pyat_coupled_setpoint_addresses(manifest["channels"]) == bound

    def test_the_setpoints_no_binding_drives_are_exactly_the_echoes(
        self, served_repo: dict
    ) -> None:
        """The writable channels outnumber the driven ones, and the rest echo.

        A harvested namespace holds setpoints with no element behind them --
        a septum current, a cavity's drive. They stay writable and their
        readback follows them, which is the sp-echo partition; what they must
        never be is silently coupled to the ring. So the difference between
        "writable" and "driven" is named rather than tolerated.
        """
        from osprey.services.virtual_accelerator.bindings import load_bindings, setpoints
        from osprey.services.virtual_accelerator.manifest.classify import (
            PARTITION_SP_ECHO,
            SETPOINT_SUBFIELD,
            setpoint_addresses,
        )

        repo = served_repo["repo"]
        channels = served_manifest(repo)["channels"]
        bound = set(setpoints(load_bindings(repo / SERVED / BINDINGS_FILE)))
        echoes = {
            channel["address"]
            for channel in channels
            if channel["partition"] == PARTITION_SP_ECHO
            and channel["subfield"] == SETPOINT_SUBFIELD
        }

        assert echoes
        assert setpoint_addresses(channels) - bound == echoes

    def test_the_ring_and_the_bindings_reach_the_served_tree_byte_for_byte(
        self, served_repo: dict
    ) -> None:
        """Re-serialising either would describe a ring the provenance disowns.

        The bindings record the digest of the ring as emitted, so the copy the
        container mounts has to be those bytes and not an equivalent document.
        """
        repo = served_repo["repo"]

        for name in (LATTICE_FILE, BINDINGS_FILE):
            assert (repo / SERVED / name).read_bytes() == (
                repo / "data" / "simulation" / name
            ).read_bytes(), name

    def test_the_env_names_the_ring_the_harvest_emitted(self, served_repo: dict) -> None:
        """A file name, not a mode and not a path: the entrypoint looks it up.

        Both derived keys are names resolved inside the container's data mount,
        so a path here would point outside the tree that was just published.
        """
        env = served_repo["first_env"]

        assert env[LATTICE_KEY] == LATTICE_FILE
        assert env[MANIFEST_KEY] == MANIFEST_FILE
        assert "/" not in env[LATTICE_KEY] and "/" not in env[MANIFEST_KEY]

    def test_a_second_build_changes_no_published_byte(self, served_repo: dict) -> None:
        """The served tree is a function of the harvest, not of the run.

        An operator rebuilding for an unrelated reason must not hand the IOC a
        different machine, so every file under the mounted directory -- the
        manifest included -- is compared whole against the first build's.
        """
        repo = served_repo["repo"]

        assert published(repo) == served_repo["first"]
        assert env_values(repo) == served_repo["first_env"]
