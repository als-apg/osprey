"""The build path a multi-user web-terminal deployment takes on a hello-world base.

This pins the recipe the setup skill hands an operator who starts from
``hello-world`` and adopts ``control-assistant``'s web-terminal stack: splice the
block in, strip what the base cannot serve, write the login files, prune the
shared logins and the panels, pull the knowledge bundle, validate, build. Scripted end
to end so the reference stays executable — a step that stops working fails here
rather than in an interview.

Several of those steps exist only because a file authored against
``control-assistant`` carries facts about that base which ``hello-world`` does
not have. Each is a line of the recipe, and each is why the naive splice of the
web-terminal block alone does not build:

* **Tier floor keys.** ``claude_code.permissions.deny`` and
  ``web.config_panel.enabled`` sit outside the block in the preset; without them
  the shared ``logbook`` and ``knowledge`` cards resolve as privileged and
  validation refuses.
* **Logins before validation.** The block names a default tier and its shared
  cards by name. Until ``osprey scaffold tiers`` has written ``tiers/<n>.yml``
  for each, validation refuses and names the verb that fixes it.
* **Panels the host lacks.** The write-armed tiers name ``events`` and
  ``bluesky``, which need service blocks this deployment has no reason to
  deploy, so the whole ``web_panels`` key comes out of those two files.
* **Apps nothing serves.** ``logbook`` opens the ``ariel`` panel, which is
  served by an ARIEL service a ``hello-world`` base does not declare
  (``services: {}``). It is dropped whole — file and ``shared:`` name — and with
  it goes the only reason the host would offer ``ariel``.
* **A graph store the base does not run.** ``knowledge`` finds channels in a
  graph store; ``hello-world`` deploys none, so the app is switched to the
  file-based channel finder and loses its graph-reading subagent and server.
* **Demo content in the block.** The roster arrives holding demo logins, and
  ``landing.notices`` names a markdown file only ``control-assistant`` ships.
  Nothing reports the missing file, so the recipe removes the key, and the demo
  logins give way to the facility's own people.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from tests.cli._lifecycle_build import rendered_config

#: The preset whose web-terminal stack the hello-world repo adopts.
SOURCE_PRESET = "control-assistant"

#: Every tier that preset's roster names. All are written; none is dropped.
TIER_NAMES = ("admin", "readonly", "readwrite")

#: Every app that preset's ``shared:`` list names. All are written; the recipe
#: prunes afterwards.
APP_NAMES = ("channel-finder", "knowledge", "logbook")

#: Apps this deployment keeps. ``knowledge`` opens the bundle pulled below.
KEPT_APPS = ("channel-finder", "knowledge")

#: Apps dropped because nothing on this base serves them.
DROPPED_APPS = ("logbook",)

#: Tiers whose file names panels this deployment cannot deploy.
PANEL_PRUNED_TIERS = ("readwrite", "admin")

#: Apps that read channels from a graph store this base does not run.
GRAPH_PRUNED_APPS = ("knowledge",)

#: The single dotted key holding the roster, the shared cards and the
#: web-terminal wiring. Spliced whole.
BLOCK_KEY = "modules.web_terminals"

#: Keys the preset sets *outside* the block that the block's shared cards
#: nonetheless depend on. Copied from the same authored config, so a preset that
#: changes their values changes what this recipe splices.
FLOOR_KEYS = ("claude_code.permissions.deny", "web.config_panel.enabled")

#: Roster entries that arrive inside the spliced block and are demo logins, not
#: people at the facility adopting it.
DEMO_LOGINS = ("alice", "bob", "carol")

#: The facility's own roster, which replaces the demo logins.
FACILITY_ROSTER = [
    {"name": "controlroom", "index": 0, "tier": "readonly", "display_name": "Control Room"},
]

#: Top-level panels the host must offer, once the recipe has pruned. It is
#: exactly the set of ``default_panel`` values the surviving apps name — ``okf``
#: for ``knowledge`` and ``channel-finder`` for ``channel-finder``, and nothing
#: else, because ``logbook`` (whose default panel was ``ariel``) is gone. A card
#: landing on a panel the host does not offer fails the build, so a panel here
#: is a panel the deployment must serve.
HOST_WEB_PANELS = ["channel-finder", "okf"]

#: Where the pulled knowledge bundle lands, relative to the repo root.
BUNDLE_PATH = "data/facility/knowledge"

#: Directory ``osprey build`` seeds in the *source* zone, one per roster entry.
CONTEXT_DIRNAME = "web-terminal-context"


def invoke(runner: CliRunner, *args: str) -> Result:
    """Run one ``osprey`` verb and insist it succeeded.

    The recipe is a chain: a step that fails makes every later assertion
    meaningless, so each call carries its own output into the failure message.
    """
    result = runner.invoke(cli, list(args))
    assert result.exit_code == 0, f"osprey {' '.join(args)} failed:\n{result.output}"
    return result


def read_profile(repo_root: Path) -> Any:
    from osprey.utils.config_writer import load_config_document

    return load_config_document(repo_root / "profile.yml")


def layer_file(repo_root: Path, kind: str, name: str) -> Path:
    return repo_root / kind / f"{name}.yml"


def splice_web_terminal_stack(repo_root: Path) -> None:
    """Paste the preset's web-terminal block and its floor keys into *repo_root*.

    Read through the profile writer rather than a plain YAML round-trip: the
    profile is a hand-edited document whose comments are the operator's, and the
    interview tells them to edit it afterwards.

    Two things arrive inside the block that this base cannot carry, so they come
    straight back out: ``landing.notices``, which names a file only the source
    preset ships, and the demo logins on the roster, which the facility's own
    people replace.
    """
    from osprey.cli.build_profile_resolve import preset_authored_config
    from osprey.utils.config_writer import load_config_document, save_config_document

    authored = preset_authored_config(SOURCE_PRESET)

    block = authored[BLOCK_KEY]
    del block["landing"]["notices"]
    block["users"] = [user for user in block["users"] if user["name"] not in DEMO_LOGINS] + [
        dict(user) for user in FACILITY_ROSTER
    ]

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)
    document["config"][BLOCK_KEY] = block
    for key in FLOOR_KEYS:
        document["config"][key] = authored[key]
    save_config_document(profile_path, document)


def drop_unserved_apps(repo_root: Path) -> None:
    """Remove every app this base has nothing behind, whole.

    Three deletions go together: the app file, the ``shared:`` name, and — once a
    build has seeded it — the per-card context directory. A ``shared:`` name left
    behind fails validation with the remedy to scaffold the app back.
    """
    from osprey.utils.config_writer import load_config_document, save_config_document

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)

    block = document["config"][BLOCK_KEY]
    block["shared"] = [name for name in block["shared"] if name not in DROPPED_APPS]

    save_config_document(profile_path, document)

    for name in DROPPED_APPS:
        layer_file(repo_root, "tiers", name).unlink()
        shutil.rmtree(repo_root / CONTEXT_DIRNAME / name, ignore_errors=True)


def drop_unhostable_panels(repo_root: Path) -> None:
    """Remove the whole ``web_panels`` key from the write-armed tier files.

    The key, not its items: a tier that adds no panels is spelled by not carrying
    the key at all, which is what the setup skill tells the operator to write
    and therefore what this pins.
    """
    from osprey.utils.config_writer import load_config_document, save_config_document

    for name in PANEL_PRUNED_TIERS:
        path = layer_file(repo_root, "tiers", name)
        document = load_config_document(path)
        del document["web_panels"]
        save_config_document(path, document)


def drop_graph_store_readers(repo_root: Path) -> None:
    """Point the graph-reading apps at the file-based channel finder.

    ``hello-world`` deploys no graph store, and a ``graph`` channel-finder mode
    without one fails the build. The subagent that queries the graph and the
    server it dials leave in the same edit.
    """
    from osprey.utils.config_writer import load_config_document, save_config_document

    for name in GRAPH_PRUNED_APPS:
        path = layer_file(repo_root, "tiers", name)
        document = load_config_document(path)
        document["channel_finder_mode"] = "hierarchical"
        document["agents"] = [
            agent for agent in document["agents"] if agent != "facility-knowledge-graph"
        ]
        document["config"]["claude_code.servers.graph.enabled"] = False
        save_config_document(path, document)


def select_host_panels(repo_root: Path) -> None:
    """Offer exactly the panels the surviving apps land on."""
    from osprey.utils.config_writer import load_config_document, save_config_document

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)
    document["web_panels"] = HOST_WEB_PANELS
    save_config_document(profile_path, document)


def emitted_names(repo_root: Path, kind: str) -> list[str]:
    return sorted(path.stem for path in (repo_root / kind).glob("*.yml"))


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Drive the whole recipe once and hand the tests its repo and its output.

    Module-scoped because the chain is one story, not five: re-running an
    ``osprey build`` per assertion would say nothing the first one did not.
    """
    runner = CliRunner()
    repo_root = tmp_path_factory.mktemp("deployments") / "demo"

    invoke(runner, "init", str(repo_root), "--preset", "hello-world", "--no-git")

    splice_web_terminal_stack(repo_root)

    # The spliced block names tiers and apps the repo does not have yet.
    # Recorded rather than asserted through `invoke`, so the test that owns the
    # claim is the one that reports it.
    spliced = runner.invoke(cli, ["validate", "--repo", str(repo_root), "--drift=warn"])

    tiers = invoke(runner, "scaffold", "tiers", "--repo", str(repo_root), "--from", SOURCE_PRESET)
    emitted = {"tiers": emitted_names(repo_root, "tiers"), "apps": emitted_names(repo_root, "apps")}

    scaffolded = runner.invoke(cli, ["validate", "--repo", str(repo_root), "--drift=warn"])

    drop_unserved_apps(repo_root)
    drop_unhostable_panels(repo_root)
    drop_graph_store_readers(repo_root)
    select_host_panels(repo_root)

    pull = invoke(
        runner, "scaffold", "pull", "--repo", str(repo_root), f"{SOURCE_PRESET}:{BUNDLE_PATH}"
    )

    settings = invoke(
        runner,
        "set",
        "--repo",
        str(repo_root),
        f"config.facility_knowledge.bundle_path={BUNDLE_PATH}",
    )

    validate = invoke(runner, "validate", "--repo", str(repo_root), "--drift=warn")
    build = invoke(runner, "build", "--repo", str(repo_root), "--skip-deps", "--skip-lifecycle")

    return {
        "repo_root": repo_root,
        "spliced_validate": spliced,
        "scaffolded_validate": scaffolded,
        "emitted": emitted,
        "scaffold": tiers.output,
        "pull": pull.output,
        "set": settings.output,
        "validate": validate.output,
        "build": build.output,
    }


def test_the_spliced_profile_refuses_until_its_tiers_and_apps_exist(
    built_repo: dict[str, Any],
) -> None:
    """The splice alone does not validate, and the refusal names the way out.

    The block names a default tier and three shared cards that have no file in
    this repo yet; the remedy for the cards is ``osprey scaffold tiers``.
    """
    result: Result = built_repo["spliced_validate"]

    assert result.exit_code != 0, result.output
    assert "osprey scaffold tiers" in result.output, result.output
    assert "tiers/readonly.yml" in result.output, result.output


def test_the_profile_validates_once_the_tiers_and_apps_are_written(
    built_repo: dict[str, Any],
) -> None:
    """From the scaffold verb onward validation is green, before any
    pruning — which is what makes "validate after every change" true for every
    later step instead of only the last one."""
    result: Result = built_repo["scaffolded_validate"]

    assert result.exit_code == 0, result.output


def test_the_recipe_writes_every_tier_and_app_the_preset_names(
    built_repo: dict[str, Any],
) -> None:
    """The scaffold verb has no selection flag: every deployment starts with
    all of them, and none lands in ``apps/``."""
    assert built_repo["emitted"] == {
        "tiers": sorted((*TIER_NAMES, *APP_NAMES)),
        "apps": [],
    }, built_repo["scaffold"]


def test_an_app_nothing_serves_leaves_nothing_behind(built_repo: dict[str, Any]) -> None:
    """Dropping ``logbook`` is three deletions, and this checks all three.

    A ``hello-world`` base declares no services, so the ARIEL server behind the
    logbook card does not exist. Half a deletion is worse than none: a
    ``shared:`` name with no file fails validation, and an orphaned context
    directory makes every later build warn.
    """
    repo_root: Path = built_repo["repo_root"]
    block = read_profile(repo_root)["config"][BLOCK_KEY]

    for name in DROPPED_APPS:
        assert not layer_file(repo_root, "tiers", name).exists(), name
        assert not (repo_root / CONTEXT_DIRNAME / name).exists(), name
        assert name not in block["shared"], name


def test_the_profile_carries_no_render_catalog(built_repo: dict[str, Any]) -> None:
    """Tiers and apps live in their own files; the profile names them and
    states nothing about them, neither nested nor as dotted keys."""
    profile = read_profile(built_repo["repo_root"])

    assert "renders" not in profile["config"][BLOCK_KEY]
    assert [key for key in profile["config"] if key.startswith(f"{BLOCK_KEY}.renders")] == []


def test_the_spliced_block_carries_no_demo_content(built_repo: dict[str, Any]) -> None:
    """``landing.notices`` names a file this base does not ship, and nothing
    reports it: validate and build both pass while every rendered config carries
    the dead path. The demo logins leave in the same edit."""
    block = read_profile(built_repo["repo_root"])["config"][BLOCK_KEY]

    assert "notices" not in block["landing"]
    assert [user["name"] for user in block["users"] if user["name"] in DEMO_LOGINS] == []


def test_every_written_file_parses_as_the_framework_reads_it(built_repo: dict[str, Any]) -> None:
    """The emission's own check, run against what actually landed on disk —
    including the files the recipe edited after the scaffold verbs wrote them."""
    from osprey.cli.profile_cmd import _parsed_layer_deltas

    repo_root: Path = built_repo["repo_root"]
    texts = {
        name: layer_file(repo_root, "tiers", name).read_text(encoding="utf-8")
        for name in TIER_NAMES
    } | {
        name: layer_file(repo_root, "tiers", name).read_text(encoding="utf-8") for name in KEPT_APPS
    }

    parsed = dict(_parsed_layer_deltas(texts, require_mapping=True))

    assert set(parsed) == set(TIER_NAMES) | set(KEPT_APPS)


def test_the_pruned_tiers_carry_no_web_panels_key(built_repo: dict[str, Any]) -> None:
    repo_root: Path = built_repo["repo_root"]

    for name in PANEL_PRUNED_TIERS:
        text = layer_file(repo_root, "tiers", name).read_text(encoding="utf-8")
        assert "web_panels" not in yaml.safe_load(text), name


def test_no_kept_app_reads_a_graph_store(built_repo: dict[str, Any]) -> None:
    repo_root: Path = built_repo["repo_root"]

    for name in KEPT_APPS:
        app = yaml.safe_load(layer_file(repo_root, "tiers", name).read_text(encoding="utf-8"))
        assert app.get("channel_finder_mode") != "graph", name


def test_the_host_offers_exactly_the_panels_the_apps_land_on(
    built_repo: dict[str, Any],
) -> None:
    """Validation does not run this check; the build does, one step later.

    So the minimum host list is derived rather than chosen: it is the set of
    ``default_panel`` values the surviving apps name. Anything beyond it is a
    tab the deployment has taken on the job of serving.
    """
    repo_root: Path = built_repo["repo_root"]

    landed_on = set()
    for kind, names in (("tiers", TIER_NAMES), ("tiers", KEPT_APPS)):
        for name in names:
            layer = yaml.safe_load(layer_file(repo_root, kind, name).read_text(encoding="utf-8"))
            if layer.get("default_panel") is not None:
                landed_on.add(layer["default_panel"])

    assert landed_on == set(HOST_WEB_PANELS)
    assert read_profile(repo_root)["web_panels"] == HOST_WEB_PANELS


def test_the_build_renders_a_project_for_every_tier_and_kept_app(
    built_repo: dict[str, Any],
) -> None:
    renders = len(TIER_NAMES) + len(KEPT_APPS)
    assert f"{renders} render(s)" in built_repo["build"], built_repo["build"]


def test_the_build_seeds_one_context_directory_per_roster_entry(
    built_repo: dict[str, Any],
) -> None:
    """``osprey build`` writes into the source zone as well as ``build/``.

    Operators keep this repo in version control, so the per-user context
    directories are seeded with a ``.gitkeep`` rather than left for the first
    person who needs one. They are the reason deleting a roster entry is not
    finished when the profile stops naming it.
    """
    repo_root: Path = built_repo["repo_root"]
    roster = {user["name"] for user in read_profile(repo_root)["config"][BLOCK_KEY]["users"]}

    seeded = {path.name for path in (repo_root / CONTEXT_DIRNAME).iterdir() if path.is_dir()}

    assert roster == {user["name"] for user in FACILITY_ROSTER}
    assert seeded == roster, built_repo["build"]
    for name in seeded:
        assert (repo_root / CONTEXT_DIRNAME / name / ".gitkeep").exists(), name


def test_the_rendered_config_points_at_the_pulled_bundle(built_repo: dict[str, Any]) -> None:
    """The second of the two `osprey set` keys, carried through the render."""
    repo_root: Path = built_repo["repo_root"]

    rendered = rendered_config(repo_root)

    assert rendered["facility_knowledge"]["bundle_path"] == BUNDLE_PATH


def test_the_pulled_bundle_is_structure_without_documents(built_repo: dict[str, Any]) -> None:
    """A pull without ``--with-content`` brings the indexes and no concepts, so
    the deployment starts with a knowledge base its facility has yet to write."""
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    repo_root: Path = built_repo["repo_root"]

    bundle_root = repo_root / BUNDLE_PATH

    # An empty result means nothing unless the structure arrived: a pull that
    # copied no file at all would satisfy the concept count just as well.
    assert len(list(bundle_root.rglob("index.md"))) == 6, built_repo["pull"]

    bundle = OKFBundle(root=bundle_root)

    assert bundle.list_concepts() == []
