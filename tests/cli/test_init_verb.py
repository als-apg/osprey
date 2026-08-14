"""``osprey init`` — the one path a deployment repo comes into existence by.

The gold-standard repo these tests measure against is the hand-authored
exemplar (``tests/fixtures/lifecycle_repo.py``). Exemplar-first: it was written
by hand, and ``init`` is made to reproduce it.

What "reproduce" can mean is deliberately scoped, because the exemplar pins two
different kinds of thing:

* The files ``init`` AUTHORS — the anchored three-zone ``.gitignore``, the
  README explaining the zones, the ``ci-extra.yml`` include point — are
  compared BYTE FOR BYTE. Those are the repo's own shell, and the exemplar is
  their specification.
* The ``data/`` tree is compared on SHAPE alone — ~2 MB of packaged bundle
  against the exemplar's ~20 hand-authored files, which exist to pin paths
  rather than content. Every path the exemplar names must be present.
* The files ``init`` RENDERS from packaged material — ``profile.yml``,
  ``personas/*.yml``, ``.env.example``, ``triggers.yml`` — are compared BYTE
  FOR BYTE too. That is the full TR-4 contract: the packaged prose is the
  exemplar's, down to where it wraps, so a preset edit that drifts from the
  specification fails here rather than shipping.
"""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.init_cmd import (
    CI_EMITTED_PATHS,
    PRESERVED_BY_FORCE,
    REPO_VERIFY_PATH,
    WRITE_ONCE_FILES,
    init,
)
from osprey.cli.main import cli
from osprey.cli.profile_conventions import unknown_root_entries
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, exemplar_source_files

#: The exemplar's display name — what ``init`` derives from the directory it
#: was given (``als-exemplar`` → ``Als Exemplar``). Spelled out here rather than
#: re-derived, so this file states what the emission should produce instead of
#: reproducing the rule and agreeing with itself either way.
EXEMPLAR_NAME = "Als Exemplar"

#: Files ``init`` authors itself, and therefore the ones the exemplar specifies
#: down to the byte.
AUTHORED_FILES = (".gitignore", "README.md", "ci-extra.yml")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _no_harvested_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own exported API keys out of every repo these tests make.

    ``init`` seeds ``.env`` from ``os.environ`` by design, so without this a run
    on a machine with ``ANTHROPIC_API_KEY`` exported writes a real key into
    ``tmp_path`` — and the tests that assert ``.env`` is absent would pass or
    fail depending on whose laptop they run on. The seeding itself is tested
    explicitly, by putting a fake key back.
    """
    from osprey.cli.templates.scaffolding import provider_api_key_entries

    for entry in provider_api_key_entries():
        monkeypatch.delenv(entry["var"], raising=False)


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give ``git commit`` an identity, for machines and CI images that have none.

    ``init`` degrades to a note rather than failing when the commit cannot be
    made, which is right for an operator and useless for a test that wants to
    know the commit happened.
    """
    monkeypatch.setenv("GIT_AUTHOR_NAME", "OSPREY")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "osprey@example.org")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "OSPREY")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "osprey@example.org")


def run_init(runner: CliRunner, *args: str):
    """Invoke ``osprey init`` and return the result, with tracebacks preserved."""
    return runner.invoke(init, list(args), catch_exceptions=False)


def init_exemplar(runner: CliRunner, target: Path, *extra: str):
    """The canonical invocation — the whole command line the exemplar corresponds to.

    Nothing is set beyond the preset: the deployment's name comes from the
    directory, which is the design (one repo is one deployment, and its
    directory name IS the deployment name).
    """
    return run_init(runner, str(target), "--preset", "control-assistant", *extra)


def git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` and return its stdout."""
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


# ---------------------------------------------------------------------------
# The emitted shape
# ---------------------------------------------------------------------------


@pytest.fixture
def exemplar_repo(runner: CliRunner, tmp_path: Path) -> Path:
    """A repo ``osprey init`` made, at the exemplar's name and directory."""
    target = tmp_path / EXEMPLAR_DIRNAME
    result = init_exemplar(runner, target)
    assert result.exit_code == 0, result.output
    return target


@pytest.mark.parametrize("name", AUTHORED_FILES)
def test_authored_files_match_the_exemplar_byte_for_byte(exemplar_repo: Path, name: str) -> None:
    assert (
        exemplar_repo.joinpath(name).read_text(encoding="utf-8")
        == (exemplar_source_files(with_ci=False)[name])
    )


def test_every_source_path_the_exemplar_names_is_emitted(exemplar_repo: Path) -> None:
    """Shape, not bytes: the exemplar's data tree pins paths, not the whole bundle."""
    missing = sorted(
        relative
        for relative in exemplar_source_files(with_ci=False)
        if not exemplar_repo.joinpath(relative).exists()
    )
    assert missing == []


# ---------------------------------------------------------------------------
# TR-4 / SC-8
# ---------------------------------------------------------------------------
#
# Both tests below are the whole requirement, asserted in full. The prose they
# police — `profile.yml`, `personas/*.yml`, `.env.example` and `triggers.yml` —
# is none of it init's own: it comes from the bundled preset's comments, the
# header `build_profile_emit` renders, the project env template and the bundled
# trigger file, so any of those can reintroduce a retired verb.

#: Strings naming a retired command. Any of them in an artifact a fresh
#: deployment ships is an instruction to its operator to run something
#: that does not exist.
RETIRED_VERB_STRINGS = (
    "osprey deploy ",
    "osprey profile ",
    "osprey build <",
    "config set-control-system",
    "set-epics-gateway",
    "claude regen",
    "profile try",
)


def _emitted_source_files(repo: Path) -> list[Path]:
    """Text files a fresh repo ships, minus the verbatim-copied ``data/`` tree.

    ``data/`` is facility content that a build only ever reads; it is copied
    byte for byte from the bundle and carries no command prose (checked — it
    has zero retired-verb hits today). Excluding it keeps this criterion about
    the artifacts init actually renders.
    """
    return [
        path
        for path in sorted(repo.rglob("*"))
        if path.is_file()
        and ".git" not in path.parts
        and "data" not in path.relative_to(repo).parts[:1]
    ]


def test_no_emitted_artifact_names_a_retired_verb(exemplar_repo: Path) -> None:
    """SC-8: nothing a fresh deployment ships may tell its operator to run a dead verb."""
    offenders = [
        f"{path.relative_to(exemplar_repo)}: {verb!r}"
        for path in _emitted_source_files(exemplar_repo)
        for verb in RETIRED_VERB_STRINGS
        if verb in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


#: Every file init RENDERS rather than authors or copies verbatim — the whole
#: set the prose rewrite has to reach. All five are listed because this tuple is
#: the only thing that gates them: a file left out here is a file whose emitter
#: could drift from the exemplar forever without anything going red.
RENDERED_FILES = (
    "profile.yml",
    "personas/readonly.yml",
    "personas/readwrite.yml",
    ".env.example",
    "triggers.yml",
)


def test_rendered_files_match_the_exemplar_byte_for_byte(exemplar_repo: Path) -> None:
    """TR-4 in full: init emits the exemplar, down to the byte, for what it renders."""
    expected = exemplar_source_files(with_ci=False)
    differing = [
        name
        for name in RENDERED_FILES
        if exemplar_repo.joinpath(name).read_text(encoding="utf-8") != expected[name]
    ]
    assert differing == []


def test_source_lives_at_the_repo_root(exemplar_repo: Path) -> None:
    """No ``profile/`` nesting: the repo root IS the profile root."""
    assert (exemplar_repo / "profile.yml").is_file()
    assert not (exemplar_repo / "profile").exists()


def test_state_zone_is_created_empty(exemplar_repo: Path) -> None:
    """``var/`` exists and holds nothing — not even a ``.gitkeep``.

    A marker file would be git-ignored along with the directory, and would be
    the one thing ``osprey reset``'s wipe had to work around.
    """
    for relative in ("var/agent_data", "var/audit"):
        state_dir = exemplar_repo / relative
        assert state_dir.is_dir()
        assert list(state_dir.iterdir()) == []


def test_build_zone_is_not_created(exemplar_repo: Path) -> None:
    """``build/`` appears on the first build. Nothing derived exists before one."""
    assert not (exemplar_repo / "build").exists()


def test_no_unknown_root_entries(exemplar_repo: Path) -> None:
    """SC-9: a freshly init'ed repo builds without a single layout warning."""
    assert unknown_root_entries(exemplar_repo) == []


def test_emitted_profile_resolves(exemplar_repo: Path) -> None:
    """The profile the loader gets back is the one that was written.

    This is what makes ``image_source``'s single home a fact rather than a
    convention: a profile stating it in both places is rejected by the parser,
    so a resolving profile has exactly one.
    """
    from osprey.cli.build_profile import resolve_build_profile

    resolved, profile_dir = resolve_build_profile(exemplar_repo / "profile.yml", None)
    resolved.validate(profile_dir)
    assert resolved.name == EXEMPLAR_NAME
    assert profile_dir == exemplar_repo


def test_persona_renders_land_in_the_build_zone(exemplar_repo: Path) -> None:
    """A persona render is build output, and its ``project`` is the repo's name.

    ``project`` must equal ``project_path``'s basename — the invariant that
    lands each render where the catalog mounts it. ``osprey build`` names a
    persona render ``<repo>-<delta stem>`` under ``build/`` and the catalog
    derives the same name from the same repo, so the two meet by construction
    rather than by anyone keeping them in step.
    """
    profile = yaml.safe_load((exemplar_repo / "profile.yml").read_text(encoding="utf-8"))
    catalog = profile["config"]["modules.web_terminals"]["personas"]

    assert sorted(catalog) == ["readonly", "readwrite"]
    for persona_name, entry in catalog.items():
        assert entry["project"] == f"{EXEMPLAR_DIRNAME}-{persona_name}"
        assert entry["project_path"] == f"build/{EXEMPLAR_DIRNAME}-{persona_name}"
        assert entry["build_profile"] == f"personas/{persona_name}.yml"
        assert Path(entry["project_path"]).name == entry["project"]
        assert (exemplar_repo / entry["build_profile"]).is_file()


def test_data_key_names_the_repos_own_tree(exemplar_repo: Path) -> None:
    profile = yaml.safe_load((exemplar_repo / "profile.yml").read_text(encoding="utf-8"))
    assert profile["data"] == "data"
    assert profile["provenance"]["preset"] == "control-assistant"


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def test_git_status_is_clean_from_birth(exemplar_repo: Path) -> None:
    """SC-1: the anchored .gitignore plus the initial commit leave nothing to report."""
    assert git(exemplar_repo, "status", "--porcelain") == ""
    assert git(exemplar_repo, "log", "--oneline").strip().endswith("Initial deployment")
    assert git(exemplar_repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"


def test_generated_zones_are_ignored_not_merely_absent(exemplar_repo: Path) -> None:
    """The zones are ignored by pattern, so a build's output never dirties git."""
    (exemplar_repo / "build").mkdir()
    (exemplar_repo / "build" / "config.yml").write_text("rendered\n", encoding="utf-8")
    (exemplar_repo / "var" / "agent_data" / "memory.json").write_text("{}\n", encoding="utf-8")
    (exemplar_repo / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")

    assert git(exemplar_repo, "status", "--porcelain") == ""


def test_no_git_skips_the_repository(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / EXEMPLAR_DIRNAME
    result = init_exemplar(runner, target, "--no-git")

    assert result.exit_code == 0, result.output
    assert not (target / ".git").exists()
    assert "--no-git" in result.output


def test_enclosing_git_repository_is_left_alone(runner: CliRunner, tmp_path: Path) -> None:
    """The clone-your-empty-repository-first workflow: git is already the operator's."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    target = tmp_path / EXEMPLAR_DIRNAME

    result = init_exemplar(runner, target)

    assert result.exit_code == 0, result.output
    assert not (target / ".git").exists()
    assert "left alone" in result.output
    # Nothing was committed into somebody else's history — the enclosing repo
    # still has no HEAD at all.
    no_head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=tmp_path, capture_output=True
    )
    assert no_head.returncode != 0


def test_a_machine_without_git_still_gets_a_complete_repo(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """git is a nicety here, not a dependency: every failure degrades to a note.

    The deployment is complete without a commit — the source zone is on disk
    and buildable — so a machine with no git on PATH, or a git with no
    configured identity, must not turn a successful materialization into a
    failed command. Simulated at the ``subprocess.run`` boundary, since the
    alternative is a test that only runs on machines that happen to lack git.
    """

    def no_git(*args: object, **kwargs: object):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    # `init_cmd` imports subprocess inside the function, so the name resolves
    # off the stdlib module at call time — patching it there is what the code
    # actually reads; there is no module attribute on init_cmd to patch.
    monkeypatch.setattr(subprocess, "run", no_git)
    target = tmp_path / EXEMPLAR_DIRNAME

    result = init_exemplar(runner, target)

    assert result.exit_code == 0, result.output
    assert "No git available" in result.output
    assert (target / "profile.yml").is_file()
    assert not (target / ".git").exists()


def test_a_git_that_fails_mid_sequence_degrades_to_a_note(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commit is the step most likely to fail in the wild — no user.email set."""
    real_run = subprocess.run

    def failing_commit(args, **kwargs: object):
        if isinstance(args, list) and "commit" in args:
            return subprocess.CompletedProcess(args, 128, "", "Author identity unknown\n")
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failing_commit)
    target = tmp_path / EXEMPLAR_DIRNAME

    result = init_exemplar(runner, target)

    assert result.exit_code == 0, result.output
    assert "nothing is committed yet" in result.output
    # `git init` still ran, so the operator can commit once they set an identity.
    assert (target / ".git").is_dir()


def test_in_place_into_an_empty_clone(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No DIR: the deployment is written into the directory the operator stands in."""
    target = tmp_path / "cloned-empty"
    target.mkdir()
    subprocess.run(["git", "init", "--quiet", str(target)], check=True)
    monkeypatch.chdir(target)

    result = run_init(runner, "--preset", "control-assistant")

    assert result.exit_code == 0, result.output
    assert (target / "profile.yml").is_file()
    assert (target / "var" / "agent_data").is_dir()
    # The clone's own git is left alone — the operator commits when ready.
    assert "left alone" in result.output


def test_in_place_refuses_a_directory_that_is_not_empty(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "notes.txt").write_text("mine\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = run_init(runner, "--preset", "control-assistant")

    assert result.exit_code == 2
    assert "not empty" in result.output
    assert not (tmp_path / "profile.yml").exists()


# ---------------------------------------------------------------------------
# Where a repo may and may not be created
# ---------------------------------------------------------------------------


def test_refuses_to_nest_inside_another_deployment(exemplar_repo: Path, runner: CliRunner) -> None:
    """One repo is exactly one deployment; a nested one is discovered by accident."""
    result = init_exemplar(runner, exemplar_repo / "nested")

    assert result.exit_code == 2
    assert "inside another one" in result.output
    assert str(exemplar_repo) in result.output
    assert not (exemplar_repo / "nested").exists()


def test_refuses_to_nest_inside_a_repo_whose_zone_is_held_aside(
    exemplar_repo: Path, runner: CliRunner
) -> None:
    """A repo mid-replacement is still one repo, and still may not be nested in.

    It reaches the guard as a FAILED lookup — there is no profile.yml up there
    for the moment — so a uniform "not found means nothing above us" would build
    a second deployment inside a facility's interrupted one, at the moment they
    are least able to notice.
    """
    from osprey.cli.repo_resolver import HELD_SOURCE_ZONE_DIRNAME

    stash = exemplar_repo / HELD_SOURCE_ZONE_DIRNAME
    stash.mkdir()
    (exemplar_repo / "profile.yml").rename(stash / "profile.yml")

    result = init_exemplar(runner, exemplar_repo / "nested")

    assert result.exit_code == 2
    assert "inside another one" in result.output
    assert str(exemplar_repo) in result.output
    # …and it says why the parent looks empty, or the refusal reads as wrong.
    assert "held aside" in result.output
    assert not (exemplar_repo / "nested").exists()
    # The guard refuses; it does not repair. Nothing moved.
    assert (stash / "profile.yml").is_file()


def test_refuses_an_existing_directory_that_is_someone_elses(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "not-ours"
    target.mkdir()
    (target / "thesis.tex").write_text("\\documentclass{article}\n", encoding="utf-8")

    result = init_exemplar(runner, target)

    assert result.exit_code == 2
    assert "not an OSPREY deployment repo" in result.output
    assert (target / "thesis.tex").is_file()
    assert not (target / "profile.yml").exists()


def test_refuses_an_existing_deployment_without_force(
    exemplar_repo: Path, runner: CliRunner
) -> None:
    result = init_exemplar(runner, exemplar_repo)

    assert result.exit_code == 2
    assert "--force" in result.output


#: Content written into each survivor before the re-materialization, and looked
#: for afterwards. Distinctive so that a file re-created with fresh content
#: reads as destroyed rather than as preserved.
SENTINEL = "kept-across-force\n"


def _seed_survivor(repo: Path, entry: str) -> Path:
    """Put :data:`SENTINEL` where ``entry`` lives; return the file to check.

    A directory entry gets a file inside it — ``var/`` surviving means the
    agent's memory surviving, not the empty directory being re-created. ``.git``
    is a directory too, and its own contents are git's, so it gets the same
    treatment rather than being special-cased into a weaker assertion.
    """
    relative = f"{entry.rstrip('/')}/sentinel.txt" if entry.endswith("/") else entry
    if entry == ".git":
        relative = ".git/sentinel.txt"
    survivor = repo / relative
    survivor.parent.mkdir(parents=True, exist_ok=True)
    survivor.write_text(SENTINEL, encoding="utf-8")
    return survivor


def test_every_file_a_rendered_repo_ships_is_in_a_named_category(
    runner: CliRunner, tmp_path: Path
) -> None:
    """No file `init` writes may sit outside the three categories.

    This is the structural guard behind the ``--force`` promise, and it is the
    test that would have caught the defect that produced it: the CI pair was in
    NO category — neither replaced nor write-once — while the prose filed it
    under "safe", so threading ``--force`` into the emission looked harmless.

    Run against a repo with deploy coordinates so the CI pair is actually on
    disk; the fixture repo has no ``deploy:`` block and would not exercise it.
    """
    from osprey.cli.profile_cmd import MATERIALIZED_SOURCE_ENTRIES

    override = tmp_path / "deploy.yml"
    override.write_text(DEPLOY_COORDINATES, encoding="utf-8")
    target = tmp_path / "categorised"
    assert (
        run_init(runner, str(target), "--preset", "hello-world", "-O", str(override)).exit_code == 0
    )

    top_level = {*MATERIALIZED_SOURCE_ENTRIES, *WRITE_ONCE_FILES, ".env"}
    uncategorised = []
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(target)
        # `.git` is git's own, and `var/` is created empty — neither is content
        # this command writes, so neither is a category's to claim.
        if relative.parts[0] in {".git", "var"}:
            continue
        if relative.parts[0] in top_level or relative.as_posix() in CI_EMITTED_PATHS:
            continue
        uncategorised.append(relative.as_posix())

    assert uncategorised == []


def test_ci_emitted_paths_match_the_scaffolding_engine(runner: CliRunner) -> None:
    """The CI category's paths are the engine's, spelled locally to keep TR-2.

    ``deploy_scaffold`` drags the build-profile chain in with it, so the paths
    cannot be imported at module scope. Cross-checked here instead, which is
    where a rename of either would otherwise go unnoticed until an operator
    found an un-preserved pipeline.
    """
    from osprey.cli.deploy_scaffold import CI_OUTPUT_NAMES

    assert set(CI_EMITTED_PATHS) == {*CI_OUTPUT_NAMES.values(), "/".join(REPO_VERIFY_PATH)}


def test_force_replaces_the_source_zone_and_nothing_else(
    exemplar_repo: Path, runner: CliRunner
) -> None:
    """Re-materializing costs source edits. It can never cost anything else.

    Driven off :data:`~osprey.cli.init_cmd.PRESERVED_BY_FORCE`, which is itself
    DERIVED from the tables driving the writes — so the promise an operator
    reads, the code implementing it, and this proof are one object rather than
    three that agree today. A new write-once file joins all three at once.
    """
    survivors = {entry: _seed_survivor(exemplar_repo, entry) for entry in PRESERVED_BY_FORCE}
    # The source zone, which --force is entitled to replace. The marker has to
    # be a string the emitted profile's own prose cannot contain, or the
    # assertion below reads a re-materialized file as the operator's edit.
    replaced_marker = "REPLACED-BY-FORCE-MARKER"
    (exemplar_repo / "profile.yml").write_text(f"name: {replaced_marker}\n", encoding="utf-8")
    (exemplar_repo / "data" / "stale.json").write_text("{}\n", encoding="utf-8")

    result = init_exemplar(runner, exemplar_repo, "--force")

    assert result.exit_code == 0, result.output
    destroyed = sorted(
        entry
        for entry, survivor in survivors.items()
        if not survivor.is_file() or survivor.read_text(encoding="utf-8") != SENTINEL
    )
    assert destroyed == []
    # …and the source zone really was re-materialized, so the test cannot pass
    # by the command having done nothing at all.
    assert not (exemplar_repo / "data" / "stale.json").exists()
    assert replaced_marker not in (exemplar_repo / "profile.yml").read_text(encoding="utf-8")


def test_force_never_regenerates_a_hand_written_pipeline(runner: CliRunner, tmp_path: Path) -> None:
    """The CI files are `osprey scaffold ci`'s to regenerate, and it has its own --force.

    Exercised with deploy coordinates present, so the emission path actually
    runs — the survival test above covers a repo whose profile declares none,
    where there is nothing to emit and nothing to overwrite.
    """
    override = tmp_path / "deploy.yml"
    override.write_text(DEPLOY_COORDINATES, encoding="utf-8")
    target = tmp_path / "ci-deployment"
    assert (
        run_init(runner, str(target), "--preset", "hello-world", "-O", str(override)).exit_code == 0
    )

    hand_written = target / ".gitlab-ci.yml"
    hand_written.write_text("# mine, no marker\nbuild-it: {}\n", encoding="utf-8")

    result = run_init(
        runner, str(target), "--preset", "hello-world", "-O", str(override), "--force"
    )

    assert result.exit_code == 0, result.output
    assert hand_written.read_text(encoding="utf-8") == "# mine, no marker\nbuild-it: {}\n"
    assert "NOT written" in result.output
    # The remedy names the verb that OWNS regeneration. Pointing an operator
    # who just ran `init --force` back at --force would send them round a loop
    # that cannot terminate, since this command never forces a CI file.
    assert "osprey scaffold ci --force" in result.output


def test_the_force_help_names_every_entry_it_replaces(runner: CliRunner) -> None:
    """The other half of the promise: what --force DOES replace, named in full.

    Read from the materializer's own tuple, so a sixth entry added there without
    a matching word in the help text is caught here rather than by an operator
    losing a file the flag never warned them about.
    """
    from osprey.cli.profile_cmd import MATERIALIZED_SOURCE_ENTRIES

    help_text = runner.invoke(init, ["--help"], catch_exceptions=False).output
    unmentioned = sorted(name for name in MATERIALIZED_SOURCE_ENTRIES if name not in help_text)
    assert unmentioned == []


def test_an_empty_directory_needs_no_force(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / EXEMPLAR_DIRNAME
    target.mkdir()

    result = init_exemplar(runner, target)

    assert result.exit_code == 0, result.output
    assert (target / "profile.yml").is_file()


def test_a_failed_run_leaves_no_directory_behind(runner: CliRunner, tmp_path: Path) -> None:
    """A retry must not be refused because the first attempt half-created the target."""
    target = tmp_path / "doomed"

    result = run_init(runner, str(target), "--preset", "no-such-preset")

    assert result.exit_code != 0
    assert not target.exists()


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_env_is_seeded_from_the_shell(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    target = tmp_path / EXEMPLAR_DIRNAME

    result = init_exemplar(runner, target)

    assert result.exit_code == 0, result.output
    env_text = (target / ".env").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=sk-ant-not-a-real-key" in env_text
    # The banner names the verb that ran, so a reader can account for the value.
    assert "`osprey init`" in env_text
    assert stat.S_IMODE((target / ".env").stat().st_mode) == 0o600


def test_a_shell_with_no_keys_writes_no_env(exemplar_repo: Path) -> None:
    """An empty secrets file reads as a configured one. Absence is the honest state."""
    assert not (exemplar_repo / ".env").exists()
    assert (exemplar_repo / ".env.example").is_file()


def test_unused_exported_keys_are_reported_not_dropped_silently(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-not-a-real-key")

    result = init_exemplar(runner, tmp_path / EXEMPLAR_DIRNAME)

    assert result.exit_code == 0, result.output
    assert "OPENAI_API_KEY" in result.output
    assert "Not seeded" in result.output


# ---------------------------------------------------------------------------
# Baked overrides, presets, CI
# ---------------------------------------------------------------------------


def test_set_pairs_are_baked_into_the_emitted_profile(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / EXEMPLAR_DIRNAME
    result = init_exemplar(runner, target, "--set", "model=sonnet")

    assert result.exit_code == 0, result.output
    profile = yaml.safe_load((target / "profile.yml").read_text(encoding="utf-8"))
    assert profile["model"] == "sonnet"


def test_override_files_are_baked_into_the_emitted_profile(
    runner: CliRunner, tmp_path: Path
) -> None:
    override = tmp_path / "overlay.yml"
    override.write_text("provider: openai\n", encoding="utf-8")
    target = tmp_path / EXEMPLAR_DIRNAME

    result = init_exemplar(runner, target, "-O", str(override))

    assert result.exit_code == 0, result.output
    profile = yaml.safe_load((target / "profile.yml").read_text(encoding="utf-8"))
    assert profile["provider"] == "openai"


def test_list_presets_needs_no_target(runner: CliRunner) -> None:
    result = run_init(runner, "--list-presets")

    assert result.exit_code == 0
    assert "control-assistant" in result.output


def test_missing_preset_points_at_the_list(runner: CliRunner, tmp_path: Path) -> None:
    result = run_init(runner, str(tmp_path / "nowhere"))

    assert result.exit_code == 2
    assert "--list-presets" in result.output


DEPLOY_COORDINATES = """\
deploy:
  ci: gitlab
  image_source: local
  host:
    name: appsdev2
    fqdn: appsdev2.example.org
    user: operator
    project_path: /home/operator/deployments/als-exemplar
"""


def test_ci_pipeline_and_health_check_are_emitted_together(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A pipeline that invokes a health check that was never written is worse than neither."""
    override = tmp_path / "deploy.yml"
    override.write_text(DEPLOY_COORDINATES, encoding="utf-8")
    target = tmp_path / "ci-deployment"

    result = run_init(runner, str(target), "--preset", "hello-world", "-O", str(override))

    assert result.exit_code == 0, result.output
    assert (target / ".gitlab-ci.yml").is_file()
    verify = target / "scripts" / "verify.sh"
    assert verify.is_file()
    assert stat.S_IMODE(verify.stat().st_mode) & stat.S_IXUSR
    assert unknown_root_entries(target) == []


def test_no_deploy_block_emits_no_pipeline(exemplar_repo: Path) -> None:
    """Presets ship the deploy block commented out; a repo from one is complete without it."""
    assert not (exemplar_repo / ".gitlab-ci.yml").exists()
    assert not (exemplar_repo / "scripts").exists()


# ---------------------------------------------------------------------------
# --up
# ---------------------------------------------------------------------------


def test_up_flags_require_up(runner: CliRunner, tmp_path: Path) -> None:
    result = run_init(runner, str(tmp_path / "demo"), "--preset", "hello-world", "-d")

    assert result.exit_code == 2
    assert "--up" in result.output


def test_the_chain_refuses_a_flag_the_target_verb_does_not_declare() -> None:
    """A renamed flag on ``up`` must break the chain loudly, not fall back quietly.

    Judged on the flag NAME, independently of whether the operator passed it:
    a guard that only fired on a set flag would stay silent through the run
    that introduced the rename and then, later, start a deployment in the wrong
    mode for whoever first passed ``--dev``.
    """
    from osprey.cli.init_cmd import _forwarded

    @click.command("up")
    @click.option("-d", "--detach", "detached", is_flag=True)
    def renamed_dev(detached: bool) -> None:  # no --dev, and no --repo
        """Stands in for an `up` that renamed --dev out from under the chain."""

    with pytest.raises(click.ClickException) as refusal:
        _forwarded(renamed_dev, Path("/repo"), {"detached": False, "dev": False})

    assert "dev" in str(refusal.value)
    # A verb declaring no --repo is fine — the chain runs from inside the repo.
    assert _forwarded(renamed_dev, Path("/repo"), {"detached": True}) == {"detached": True}


def test_the_chain_finds_both_verbs_on_the_group() -> None:
    """``--up`` chains ``build`` then ``up`` through the CLI group, by name.

    The lookup is the whole mechanism, so the lookup is what is asserted.
    Checking it indirectly through a run would make the outcome depend on a
    full render succeeding, which is the acceptance walk's question rather
    than a unit test's.
    """
    import click as _click

    ctx = _click.Context(cli)
    assert cli.get_command(ctx, "build") is not None
    assert cli.get_command(ctx, "up") is not None


def test_up_reports_a_missing_verb_rather_than_failing_obscurely(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    """A verb the chain cannot find is named, not a traceback.

    Driven by hiding one of the two verbs, because they both exist now. The
    repo is still created — that part succeeded — and the operator is told
    which verb is missing.
    """
    from osprey.cli.main import LazyGroup

    real = LazyGroup.get_command
    monkeypatch.setattr(
        LazyGroup,
        "get_command",
        lambda self, ctx, name: None if name == "up" else real(self, ctx, name),
    )

    target = tmp_path / "chained"
    result = runner.invoke(
        cli,
        ["init", str(target), "--preset", "hello-world", "--up", "-d"],
        catch_exceptions=False,
    )

    assert (target / "profile.yml").is_file()
    assert result.exit_code != 0
    assert "--up cannot" in result.output
    assert "osprey up" in result.output


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_init_is_registered_on_the_cli(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "init" in result.output


def test_init_carries_no_repo_flag(runner: CliRunner) -> None:
    """``init`` CREATES repos, so it is repo-free: there is none to point at."""
    from osprey.cli.repo_resolver import is_repo_free

    assert is_repo_free("osprey init")
    assert "--repo" not in {param.opts[0] for param in init.params if param.opts}


def test_importing_the_module_stays_off_the_heavy_chain() -> None:
    """TR-2: ``osprey --help`` must not pay for the build-profile import chain.

    Measured by importing the module in a fresh interpreter and asking which
    other modules came with it, rather than by grepping the source for import
    statements — the latter passes for a module that pulls the chain in
    transitively.
    """
    probe = (
        "import sys, osprey.cli.init_cmd;"
        "print(any(name.startswith('osprey.cli.build_profile') for name in sys.modules),"
        "'osprey.cli.profile_cmd' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert completed.stdout.split() == ["False", "False"]
