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
  so the first ``emit`` refuses with one ``rm`` line; this recipe removes
  exactly what that line names and nothing else. Afterwards the flat database,
  the tier-3 copy and the built copy are one file -- the assertion that goes
  red if the build's tier materializer ever overwrites the emitted database
  with preset material -- and the demo knowledge pages are gone from the
  bundle index rather than merely unlinked.
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

    emitted = emit(runner, repo)
    assert emitted.exit_code == 0, emitted.output
    invoke(runner, "set", "--repo", str(repo), *MIDDLE_LAYER_SETTINGS)

    validate = invoke(runner, "validate", "--repo", str(repo), "--drift=warn")
    build = invoke(runner, "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle")

    return {
        "repo": repo,
        "refused": refused,
        "removed": removed,
        "emit": emitted.output,
        "validate": validate.output,
        "build": build.output,
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
