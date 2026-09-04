"""The build path a multi-user web-terminal deployment takes on a hello-world base.

This pins the recipe the build interview hands an operator who starts from
``hello-world`` and adopts ``control-assistant``'s web-terminal stack: splice the
block in, strip what the base cannot serve, set the prefix, emit the persona
deltas, prune the catalog and the panels, pull the knowledge bundle, validate,
build. Scripted end to end so the reference stays executable — a step that stops
working fails here rather than in an interview.

Several of those steps exist only because a delta authored against
``control-assistant`` carries facts about that base which ``hello-world`` does
not have. Each is a line of the recipe, and each is why the naive splice of the
web-terminal block alone does not build:

* **Tier floor keys.** ``claude_code.permissions.deny`` and
  ``web.config_panel.enabled`` sit outside the block in the preset; without them
  the shared ``logbook`` and ``knowledge`` cards resolve as privileged and
  validation refuses.
* **facility.prefix.** The preset names one and ``hello-world`` does not, so the
  web container names render as ``-nginx`` and validation refuses. The recipe
  sets it early, which is what lets validation pass from the splice onward.
* **Panels the host lacks.** The write-armed deltas name ``events`` and
  ``bluesky``, which need service blocks this deployment has no reason to
  deploy, so the whole ``web_panels`` key comes out of those two files.
* **Personas nothing serves.** ``logbook`` opens the ``ariel`` panel, which is
  served by an ARIEL service a ``hello-world`` base does not declare
  (``services: {}``). It is dropped whole — file, catalog keys, roster entry —
  and with it goes the only reason the host would offer ``ariel``.
* **A catalog stated twice.** ``osprey scaffold personas`` appends dotted
  ``modules.web_terminals.personas.*`` keys, which win over the nested
  ``personas:`` mapping inside the spliced block. The mapping is deleted so each
  catalog fact is stated once.
* **Demo content in the block.** The roster arrives holding demo logins, and
  ``landing.notices`` names a markdown file only ``control-assistant`` ships.
  Nothing reports the missing file, so the recipe removes the key.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.main import cli

#: The preset whose web-terminal stack the hello-world repo adopts.
PERSONA_PRESET = "control-assistant"

#: Every persona that preset catalogs, in the order it writes them. All five are
#: emitted; the recipe prunes afterwards.
PERSONA_NAMES = ("readonly", "readwrite", "admin", "logbook", "knowledge")

#: Personas this deployment keeps. ``readonly`` is what ``default_persona``
#: names, ``knowledge`` opens the bundle pulled below, and the two write-armed
#: tiers stand in for roles an operator asked for by name.
KEPT_PERSONAS = ("readonly", "readwrite", "admin", "knowledge")

#: Personas dropped because nothing on this base serves them.
DROPPED_PERSONAS = ("logbook",)

#: The single dotted key holding the roster, the personas catalog and the
#: web-terminal wiring. Spliced whole.
BLOCK_KEY = "modules.web_terminals"

#: Keys the preset sets *outside* the block that the block's shared cards
#: nonetheless depend on. Copied from the same authored config, so a preset that
#: changes their values changes what this recipe splices.
FLOOR_KEYS = ("claude_code.permissions.deny", "web.config_panel.enabled")

#: Roster entries that arrive inside the spliced block and are demo logins, not
#: people at the facility adopting it.
DEMO_LOGINS = ("alice", "bob", "carol")

#: Personas whose delta names panels this deployment cannot deploy.
PANEL_PRUNED_PERSONAS = ("readwrite", "admin")

#: Top-level panels the host must offer, once the recipe has pruned. It is
#: exactly the set of ``default_panel`` values the surviving deltas name — ``okf``
#: for ``knowledge``, and nothing else, because ``logbook`` (whose default panel
#: was ``ariel``) is gone. A persona landing on a panel the host does not offer
#: fails the build with "Unknown default_panel", so a panel here is a panel the
#: deployment must serve.
HOST_WEB_PANELS = ["okf"]

#: Where the pulled knowledge bundle lands, relative to the repo root.
BUNDLE_PATH = "data/facility_knowledge"

#: The prefix the web container names are built from.
FACILITY_PREFIX = "demo"

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


def splice_web_terminal_stack(repo_root: Path) -> None:
    """Paste the preset's web-terminal block and its floor keys into *repo_root*.

    Read through the profile writer rather than a plain YAML round-trip: the
    profile is a hand-edited document whose comments are the operator's, and the
    interview tells them to edit it afterwards.

    Two things arrive inside the block that this base cannot carry, so they come
    straight back out: ``landing.notices``, which names a file only the source
    preset ships, and the demo logins on the roster.
    """
    from osprey.cli.build_profile_resolve import preset_authored_config
    from osprey.utils.config_writer import load_config_document, save_config_document

    authored = preset_authored_config(PERSONA_PRESET)

    block = authored[BLOCK_KEY]
    del block["landing"]["notices"]
    block["users"] = [user for user in block["users"] if user["name"] not in DEMO_LOGINS]

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)
    document["config"][BLOCK_KEY] = block
    for key in FLOOR_KEYS:
        document["config"][key] = authored[key]
    save_config_document(profile_path, document)


def drop_dead_catalog_mapping(repo_root: Path) -> None:
    """Delete the nested ``personas:`` mapping the dotted keys have superseded.

    ``osprey scaffold personas`` appends ``modules.web_terminals.personas.<p>.<k>``
    keys under ``config:``. The deeper spelling wins, so the mapping still inside
    the spliced block states every catalog fact a second time, with the source
    preset's values. The recipe removes it so each fact is stated once.
    """
    from osprey.utils.config_writer import load_config_document, save_config_document

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)
    del document["config"][BLOCK_KEY]["personas"]
    save_config_document(profile_path, document)


def drop_unserved_personas(repo_root: Path) -> None:
    """Remove every persona this base has nothing behind, whole.

    Four deletions go together, and the recipe says so: the delta file, the
    dotted catalog keys, the roster entry, and — once a build has seeded it —
    the per-user context directory. Leaving any one behind breaks a later build
    or leaves it warning about a user who is not on the roster.
    """
    from osprey.utils.config_writer import load_config_document, save_config_document

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)

    block = document["config"][BLOCK_KEY]
    block["users"] = [
        user for user in block["users"] if user.get("persona") not in DROPPED_PERSONAS
    ]
    for key in list(document["config"]):
        if key.startswith(f"{BLOCK_KEY}.personas.") and key.split(".")[3] in DROPPED_PERSONAS:
            del document["config"][key]

    save_config_document(profile_path, document)

    for name in DROPPED_PERSONAS:
        (repo_root / "personas" / f"{name}.yml").unlink()
        shutil.rmtree(repo_root / CONTEXT_DIRNAME / name, ignore_errors=True)


def drop_unhostable_panels(repo_root: Path) -> None:
    """Remove the whole ``web_panels`` key from the write-armed persona deltas.

    The key, not its items: a delta that selects no panels is spelled by not
    carrying the key at all, which is what the interview tells the operator to
    write and therefore what this pins.
    """
    from osprey.utils.config_writer import load_config_document, save_config_document

    for name in PANEL_PRUNED_PERSONAS:
        path = repo_root / "personas" / f"{name}.yml"
        document = load_config_document(path)
        del document["web_panels"]
        save_config_document(path, document)


def select_host_panels(repo_root: Path) -> None:
    """Offer exactly the panels the surviving deltas land on."""
    from osprey.utils.config_writer import load_config_document, save_config_document

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)
    document["web_panels"] = HOST_WEB_PANELS
    save_config_document(profile_path, document)


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

    invoke(runner, "set", "--repo", str(repo_root), f"config.facility.prefix={FACILITY_PREFIX}")

    # The recipe claims validation is green from here on, before a single
    # persona has been emitted. Recorded rather than asserted through `invoke`,
    # so the test that owns the claim is the one that reports it.
    spliced = runner.invoke(cli, ["validate", "--repo", str(repo_root), "--drift=warn"])

    personas = invoke(
        runner, "scaffold", "personas", "--repo", str(repo_root), "--from", PERSONA_PRESET
    )
    emitted = sorted(path.stem for path in (repo_root / "personas").glob("*.yml"))

    drop_dead_catalog_mapping(repo_root)
    drop_unserved_personas(repo_root)
    drop_unhostable_panels(repo_root)
    select_host_panels(repo_root)

    pull = invoke(
        runner, "scaffold", "pull", "--repo", str(repo_root), f"{PERSONA_PRESET}:{BUNDLE_PATH}"
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
        "emitted": emitted,
        "personas": personas.output,
        "pull": pull.output,
        "set": settings.output,
        "validate": validate.output,
        "build": build.output,
    }


def test_the_spliced_profile_validates_before_any_persona_is_emitted(
    built_repo: dict[str, Any],
) -> None:
    """``facility.prefix`` is set with the splice, not at the end.

    That ordering is the whole point: it makes "validate after every change"
    true for every later step instead of only the last one.
    """
    result: Result = built_repo["spliced_validate"]

    assert result.exit_code == 0, result.output


def test_the_recipe_emits_one_delta_file_per_persona(built_repo: dict[str, Any]) -> None:
    """The emission has no selection flag: every deployment starts with all five."""
    assert built_repo["emitted"] == sorted(PERSONA_NAMES), built_repo["personas"]


def test_a_persona_nothing_serves_leaves_nothing_behind(built_repo: dict[str, Any]) -> None:
    """Dropping ``logbook`` is four deletions, and this checks all four.

    A ``hello-world`` base declares no services, so the ARIEL server behind the
    logbook card does not exist. Half a deletion is worse than none: an orphaned
    catalog key fails the build, and an orphaned context directory makes every
    later build warn.
    """
    repo_root: Path = built_repo["repo_root"]
    profile = read_profile(repo_root)
    block = profile["config"][BLOCK_KEY]

    for name in DROPPED_PERSONAS:
        assert not (repo_root / "personas" / f"{name}.yml").exists(), name
        assert not (repo_root / CONTEXT_DIRNAME / name).exists(), name
        assert [user for user in block["users"] if user.get("persona") == name] == [], name
        assert [
            key for key in profile["config"] if key.startswith(f"{BLOCK_KEY}.personas.{name}.")
        ] == [], name


def test_the_catalog_states_each_persona_once(built_repo: dict[str, Any]) -> None:
    """The dotted keys win, so the nested mapping beside them is dead text."""
    profile = read_profile(built_repo["repo_root"])

    assert "personas" not in profile["config"][BLOCK_KEY]

    repointed = {
        key.split(".")[3] for key in profile["config"] if key.startswith(f"{BLOCK_KEY}.personas.")
    }

    assert repointed == set(KEPT_PERSONAS)


def test_the_spliced_block_carries_no_demo_content(built_repo: dict[str, Any]) -> None:
    """``landing.notices`` names a file this base does not ship, and nothing
    reports it: validate and build both pass while every persona config renders
    the dead path. The demo logins leave in the same edit."""
    block = read_profile(built_repo["repo_root"])["config"][BLOCK_KEY]

    assert "notices" not in block["landing"]
    assert [user["name"] for user in block["users"] if user["name"] in DEMO_LOGINS] == []


def test_every_emitted_delta_parses_as_the_framework_reads_it(built_repo: dict[str, Any]) -> None:
    """The emission's own check, run against what actually landed on disk —
    including the two files the recipe edited after the emission wrote them."""
    from osprey.cli.profile_cmd import _parsed_persona_deltas

    repo_root: Path = built_repo["repo_root"]
    texts = {
        name: (repo_root / "personas" / f"{name}.yml").read_text(encoding="utf-8")
        for name in KEPT_PERSONAS
    }

    parsed = dict(_parsed_persona_deltas(texts, require_mapping=True))

    assert set(parsed) == set(KEPT_PERSONAS)


def test_the_pruned_deltas_carry_no_web_panels_key(built_repo: dict[str, Any]) -> None:
    repo_root: Path = built_repo["repo_root"]

    for name in PANEL_PRUNED_PERSONAS:
        text = (repo_root / "personas" / f"{name}.yml").read_text(encoding="utf-8")
        assert "web_panels" not in yaml.safe_load(text), name


def test_the_host_offers_exactly_the_panels_the_deltas_land_on(
    built_repo: dict[str, Any],
) -> None:
    """Validation does not run this check; the build does, one step later.

    So the minimum host list is derived rather than chosen: it is the set of
    ``default_panel`` values the surviving deltas name. Anything beyond it is a
    tab the deployment has taken on the job of serving.
    """
    repo_root: Path = built_repo["repo_root"]

    landed_on = set()
    for name in KEPT_PERSONAS:
        delta = yaml.safe_load((repo_root / "personas" / f"{name}.yml").read_text(encoding="utf-8"))
        if delta.get("default_panel") is not None:
            landed_on.add(delta["default_panel"])

    assert landed_on == set(HOST_WEB_PANELS)
    assert read_profile(repo_root)["web_panels"] == HOST_WEB_PANELS


def test_the_build_renders_a_project_for_every_surviving_persona(
    built_repo: dict[str, Any],
) -> None:
    assert f"{len(KEPT_PERSONAS)} persona render(s)" in built_repo["build"], built_repo["build"]


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

    assert seeded == roster, built_repo["build"]
    for name in seeded:
        assert (repo_root / CONTEXT_DIRNAME / name / ".gitkeep").exists(), name


def test_the_rendered_config_points_at_the_pulled_bundle(built_repo: dict[str, Any]) -> None:
    """The second of the two `osprey set` keys, carried through the render."""
    repo_root: Path = built_repo["repo_root"]

    rendered = yaml.safe_load((repo_root / "build" / "config.yml").read_text(encoding="utf-8"))

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
