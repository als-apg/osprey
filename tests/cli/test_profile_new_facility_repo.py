"""The facility repo ``osprey profile new`` creates around the profile.

``osprey profile new <name>`` is the first command a facility ever runs, and
what it produces is a whole repository: the profile it owns, the render target
beside it, the two repo-root files the facility owns from creation onward, and —
as soon as the profile says where to deploy — the generated pipeline. This module
pins that shape.

The profile directory's own contents are ``test_profile_new.py``'s subject; here
it is one entry in a tree. What matters at this level is which files exist, which
of them are regenerated and which are the facility's forever, and that every
degradation — no git, no deployment coordinates — still leaves a complete repo
behind.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_profile_resolve import FACILITY_BUILD_DIRNAME, FACILITY_PROFILE_DIRNAME
from osprey.cli.deploy_scaffold import VERIFY_OUTPUT_PATH, scaffold_deploy_files
from osprey.cli.deploy_scaffold_templates import CI_MARKER
from osprey.cli.profile_cmd import profile

#: A complete ``deploy:`` block, layered in with ``-O``. Presets ship the block
#: commented out, so this is how a repo gets a pipeline at creation time.
DEPLOY_OVERRIDE = """\
deploy:
  ci: gitlab
  image_source: local
  host:
    name: demo-deploy
    fqdn: demo-deploy.example.org
    user: osprey
    project_path: /opt/demo-facility
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _new(runner: CliRunner, repo: Path, preset: str = "hello-world", *extra: str):
    return runner.invoke(profile, ["new", str(repo), "--preset", preset, *extra])


def _deploy_override(tmp_path: Path) -> str:
    path = tmp_path / "deploy.yml"
    path.write_text(DEPLOY_OVERRIDE, encoding="utf-8")
    return str(path)


def _entries(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


def _untracked(repo: Path) -> set[str]:
    """Every path git would offer to add, as exact paths.

    Exact rather than a substring search over the output: ``profile/.env`` is a
    prefix of ``profile/.env.example``, and a test that cannot tell those apart
    would pass with the secrets file staged for commit.
    """
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    assert status.returncode == 0, status.stderr
    return {line[3:] for line in status.stdout.splitlines()}


# ---------------------------------------------------------------------------
# The emitted tree
# ---------------------------------------------------------------------------


def test_emits_the_whole_repo(runner: CliRunner, tmp_path: Path) -> None:
    """One command, one repo: nothing here is left for the operator to assemble."""
    repo = tmp_path / "demo-facility"

    result = _new(runner, repo)

    assert result.exit_code == 0, result.output
    assert _entries(repo) == [".git", ".gitignore", "build", "ci-extra.yml", "profile"]
    assert (repo / FACILITY_PROFILE_DIRNAME / "profile.yml").is_file()
    assert (repo / FACILITY_BUILD_DIRNAME).is_dir()
    assert _entries(repo / FACILITY_BUILD_DIRNAME) == [], "build/ is a render target, not content"


def test_the_repo_is_the_one_the_build_and_scaffold_verbs_recognize(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The layout is not this command's private convention — the same two
    functions that make ``osprey build`` and ``osprey deploy scaffold``
    position-independent have to find the repo they were given."""
    from osprey.cli.build_profile_resolve import facility_repo_root, resolve_build_output_dir
    from osprey.cli.deploy_scaffold import find_facility_repo_root

    repo = tmp_path / "demo-facility"
    assert _new(runner, repo).exit_code == 0
    profile_dir = repo / FACILITY_PROFILE_DIRNAME

    assert facility_repo_root(profile_dir) == repo
    assert find_facility_repo_root(profile_dir) == repo
    assert resolve_build_output_dir(None, profile_dir) == repo / FACILITY_BUILD_DIRNAME


def test_the_facility_name_comes_from_the_repo_not_the_profile_directory(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Every facility's profile directory is called ``profile/``, so the name has
    to be read off the repo the operator named — otherwise every facility in the
    world would be called "Profile"."""
    import yaml

    repo = tmp_path / "demo-facility"
    assert _new(runner, repo).exit_code == 0

    parsed = yaml.safe_load((repo / FACILITY_PROFILE_DIRNAME / "profile.yml").read_text())
    assert parsed["name"] == "Demo Facility"


def test_set_name_still_wins_over_the_repo_directory(runner: CliRunner, tmp_path: Path) -> None:
    import yaml

    repo = tmp_path / "demo-facility"
    assert _new(runner, repo, "hello-world", "--set", "name=ALS Control").exit_code == 0

    parsed = yaml.safe_load((repo / FACILITY_PROFILE_DIRNAME / "profile.yml").read_text())
    assert parsed["name"] == "ALS Control"


# ---------------------------------------------------------------------------
# .gitignore — anchored, or it swallows files nobody meant
# ---------------------------------------------------------------------------


def test_gitignore_patterns_are_anchored_to_the_repo_root(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The foot-gun this file exists to avoid: an unanchored ``build/`` or
    ``.env`` matches a same-named path ANYWHERE in the tree, so a facility that
    later adds one under its data tree finds it silently untracked.

    Asserted through git itself rather than by reading the patterns: what
    matters is what git does with them.
    """
    repo = tmp_path / "demo-facility"
    assert _new(runner, repo).exit_code == 0

    rendered = repo / FACILITY_BUILD_DIRNAME / "rendered"
    rendered.mkdir()
    (rendered / "config.yml").write_text("{}\n", encoding="utf-8")
    (repo / FACILITY_PROFILE_DIRNAME / ".env").write_text("KEY=secret\n", encoding="utf-8")
    # The same two names elsewhere in the tree, where they are ordinary facility
    # content. Under `tools/` rather than under the profile, whose own
    # `.gitignore` covers `.env` by name from the inside.
    tools = repo / "tools" / "build"
    tools.mkdir(parents=True)
    (tools / "make.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "tools" / ".env").write_text("not a secret\n", encoding="utf-8")

    untracked = _untracked(repo)

    assert f"{FACILITY_BUILD_DIRNAME}/rendered/config.yml" not in untracked
    assert f"{FACILITY_PROFILE_DIRNAME}/.env" not in untracked
    # An unanchored `build/` or `.env` would have swallowed both of these.
    assert "tools/build/make.sh" in untracked
    assert "tools/.env" in untracked


def test_the_profiles_secrets_are_ignored_from_the_repo_root(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The profile carries a ``.gitignore`` of its own; the repo's must cover the
    same secrets independently, because either file can be edited."""
    repo = tmp_path / "demo-facility"
    assert _new(runner, repo).exit_code == 0
    (repo / FACILITY_PROFILE_DIRNAME / ".gitignore").unlink()
    (repo / FACILITY_PROFILE_DIRNAME / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")

    assert f"{FACILITY_PROFILE_DIRNAME}/.env" not in _untracked(repo)


# ---------------------------------------------------------------------------
# git init, and doing without it
# ---------------------------------------------------------------------------


def test_git_init_leaves_a_repo_with_everything_untracked(
    runner: CliRunner, tmp_path: Path
) -> None:
    repo = tmp_path / "demo-facility"

    assert _new(runner, repo).exit_code == 0

    assert (repo / ".git").is_dir()
    untracked = _untracked(repo)
    assert ".gitignore" in untracked
    assert "ci-extra.yml" in untracked
    assert f"{FACILITY_PROFILE_DIRNAME}/profile.yml" in untracked
    assert not any(path.startswith(f"{FACILITY_BUILD_DIRNAME}/") for path in untracked)


def test_a_no_git_environment_still_gets_the_whole_repo(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git is how the repo is meant to be kept, not how it is built. Without it
    the command says so and finishes."""
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    repo = tmp_path / "demo-facility"

    result = _new(runner, repo)

    assert result.exit_code == 0, result.output
    assert "skipped `git init`" in result.output
    assert not (repo / ".git").exists()
    assert _entries(repo) == [".gitignore", "build", "ci-extra.yml", "profile"]
    assert (repo / FACILITY_PROFILE_DIRNAME / "profile.yml").is_file()


def test_an_existing_git_repository_is_left_alone(runner: CliRunner, tmp_path: Path) -> None:
    repo = tmp_path / "demo-facility"
    repo.mkdir()
    assert subprocess.run(["git", "init", "--quiet", str(repo)], check=False).returncode == 0
    marker = repo / ".git" / "osprey-was-here"
    marker.write_text("original\n", encoding="utf-8")

    result = _new(runner, repo, "hello-world", "--force")

    assert result.exit_code == 0, result.output
    assert marker.read_text(encoding="utf-8") == "original\n"
    assert "Existing git repository" in result.output


# ---------------------------------------------------------------------------
# The generated deployment files, and the repo that has no coordinates yet
# ---------------------------------------------------------------------------


def test_no_deploy_block_degrades_to_a_repo_without_a_pipeline(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Every preset ships ``deploy:`` commented out, so this is what a fresh
    repo normally looks like: complete, minus the two generated files, and told
    exactly how to get them."""
    repo = tmp_path / "demo-facility"

    result = _new(runner, repo)

    assert result.exit_code == 0, result.output
    assert not (repo / ".gitlab-ci.yml").exists()
    assert not repo.joinpath(*VERIFY_OUTPUT_PATH).exists()
    assert "No CI pipeline yet" in result.output
    assert "osprey deploy scaffold" in result.output
    # The include point is written anyway: it is the facility's file, not the
    # pipeline's, and it has to predate the pipeline that includes it.
    assert (repo / "ci-extra.yml").is_file()


def test_a_declared_deploy_block_emits_the_pipeline_at_creation(
    runner: CliRunner, tmp_path: Path
) -> None:
    repo = tmp_path / "demo-facility"

    result = _new(runner, repo, "hello-world", "-O", _deploy_override(tmp_path))

    assert result.exit_code == 0, result.output
    pipeline = repo / ".gitlab-ci.yml"
    assert pipeline.is_file()
    assert CI_MARKER in pipeline.read_text(encoding="utf-8")
    verify = repo.joinpath(*VERIFY_OUTPUT_PATH)
    assert verify.is_file()
    assert verify.stat().st_mode & 0o777 == 0o755, "the health check is meant to be run by hand"
    assert _entries(repo) == [
        ".git",
        ".gitignore",
        ".gitlab-ci.yml",
        "build",
        "ci-extra.yml",
        "profile",
    ]


def test_scaffold_re_emission_over_a_fresh_repo_changes_nothing(
    runner: CliRunner, tmp_path: Path
) -> None:
    """One engine, two entry points. If creation and re-emission could disagree,
    the operator's first ``osprey deploy scaffold`` would produce a diff.
    """
    repo = tmp_path / "demo-facility"
    assert _new(runner, repo, "hello-world", "-O", _deploy_override(tmp_path)).exit_code == 0

    again = scaffold_deploy_files(repo)

    assert [emitted.action for emitted in again] == ["unchanged", "unchanged"]


def test_the_ci_extra_file_matches_the_hand_built_reference(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``tests/deployment/goldens/`` is the hand-built reference deployment, and
    ``ci-extra.yml`` is the one file in it this command emits rather than the
    scaffolding engine. The golden was written for a facility called "Demo
    Facility", which is what a repo of that name is named after — so for this
    one repo name the two files are byte-identical.
    """
    golden = (
        Path(__file__).resolve().parents[1] / "deployment" / "goldens" / "ci-extra.yml"
    ).read_text(encoding="utf-8")
    repo = tmp_path / "demo-facility"
    assert _new(runner, repo).exit_code == 0

    assert (repo / "ci-extra.yml").read_text(encoding="utf-8") == golden


def test_the_pipeline_includes_the_ci_extra_file_that_was_written_beside_it(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The two halves of the CI surface have to agree: the generated pipeline
    includes ``ci-extra.yml`` by name, and nothing generates that file but this
    command."""
    import yaml

    repo = tmp_path / "demo-facility"
    assert _new(runner, repo, "hello-world", "-O", _deploy_override(tmp_path)).exit_code == 0

    pipeline = yaml.safe_load((repo / ".gitlab-ci.yml").read_text())
    included = {entry.get("local") for entry in pipeline["include"] if isinstance(entry, dict)}
    assert "ci-extra.yml" in included
    assert yaml.safe_load((repo / "ci-extra.yml").read_text()) is not None, (
        "an empty include target is not parseable YAML"
    )


# ---------------------------------------------------------------------------
# Re-running over an existing repo
# ---------------------------------------------------------------------------


def test_an_existing_repo_is_refused_without_force(runner: CliRunner, tmp_path: Path) -> None:
    repo = tmp_path / "demo-facility"
    assert _new(runner, repo).exit_code == 0
    (repo / "ci-extra.yml").write_text("my-job: {}\n", encoding="utf-8")

    result = _new(runner, repo)

    assert result.exit_code == 2
    assert "already exists" in result.output
    assert (repo / "ci-extra.yml").read_text(encoding="utf-8") == "my-job: {}\n"


def test_force_re_materializes_the_profile_but_never_the_facilitys_own_files(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``--force`` is about the profile. ``ci-extra.yml`` and the repo's
    ``.gitignore`` belong to the facility from the moment they exist, and CI jobs
    somebody wrote must not be a casualty of re-materializing a preset."""
    repo = tmp_path / "demo-facility"
    assert _new(runner, repo).exit_code == 0
    (repo / "ci-extra.yml").write_text("my-job:\n  script: [true]\n", encoding="utf-8")
    (repo / ".gitignore").write_text("/build/\n# my own rule\n*.bak\n", encoding="utf-8")
    (repo / FACILITY_PROFILE_DIRNAME / "profile.yml").write_text("name: Edited\n", encoding="utf-8")

    result = _new(runner, repo, "hello-world", "--force")

    assert result.exit_code == 0, result.output
    assert "Edited" not in (repo / FACILITY_PROFILE_DIRNAME / "profile.yml").read_text()
    assert (repo / "ci-extra.yml").read_text(encoding="utf-8") == "my-job:\n  script: [true]\n"
    assert "# my own rule" in (repo / ".gitignore").read_text(encoding="utf-8")


def test_force_refuses_a_directory_that_is_not_a_facility_repo(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A mistyped target may be somebody's work. ``--force`` is not a licence to
    write a repo into it."""
    repo = tmp_path / "not-a-repo"
    repo.mkdir()
    (repo / "keepme.txt").write_text("mine\n", encoding="utf-8")

    result = _new(runner, repo, "hello-world", "--force")

    assert result.exit_code == 2
    assert "not a facility repo" in result.output
    assert _entries(repo) == ["keepme.txt"]


def test_force_populates_an_empty_directory(runner: CliRunner, tmp_path: Path) -> None:
    repo = tmp_path / "demo-facility"
    repo.mkdir()

    result = _new(runner, repo, "hello-world", "--force")

    assert result.exit_code == 0, result.output
    assert (repo / FACILITY_PROFILE_DIRNAME / "profile.yml").is_file()


def test_a_failed_materialization_leaves_no_repo_root_behind(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Fail-before-mutating, at the repo level: a bad ``--set`` must not leave an
    empty directory that the corrected re-run then refuses."""
    repo = tmp_path / "demo-facility"

    result = _new(runner, repo, "hello-world", "--set", "tier=2")

    assert result.exit_code == 2
    assert not repo.exists()
