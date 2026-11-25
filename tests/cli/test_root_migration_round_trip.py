"""A previous-release deployment on the retired root layout migrates, by either route, to one tree.

The tree is the previous release's repo (:func:`_previous_release_repo`: its
root entries and its facility tree at their retired paths), extended here with
what a deployment in service also holds and the shared fixture does not: a
generated CI pair, a generated systemd unit and boot hook, an operator script
and service, a ``.dockerignore`` line and a profile value naming retired paths,
certificates under ``data/.runtime`` and canopy staging left at the root. It is
committed to its own git repository.

Every command that reads the source zone refuses it before writing anything.
The printed block, run through ``sh``, and ``osprey init --revalidate`` end in
the same files and the same staged history, and the result builds. A clash
refuses by both routes, naming every pair; a merge into a successor that
already holds other files keeps them all. ``osprey init --force`` over the
migrated tree replaces only the entries it ships and keeps every other file
under ``agent/`` and ``ops/``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.deploy_scaffold import scaffold_systemd_unit
from osprey.cli.init_cmd import init
from osprey.cli.main import cli
from osprey.cli.profile_cmd import MATERIALIZED_SOURCE_ENTRIES
from osprey.cli.repo_resolver import HELD_SOURCE_ZONE_DIRNAME
from osprey.repo_retired import RetiredRootRefused, find_retired_root, refuse_retired_layout
from osprey.utils.workspace import IMAGE_DIR_NAME
from osprey_facility.retired import find_retired
from tests._builds import run_build
from tests.cli.test_init_revalidate import _previous_release_repo
from tests.cli.test_root_retired_refusal import REFUSING
from tests.fixtures.lifecycle_repo import exemplar_source_files

pytestmark = [pytest.mark.slow]

#: The repo's directory name, as the previous release's fixture names it.
PROJECT = "als-exemplar"

#: The facility tree, relative to the deployment.
FACILITY = "data/facility"

#: The generated CI pipeline and its health check, at their retired paths.
CI_FILES = {".gitlab-ci.yml": ".gitlab-ci.yml", "ops/scripts/verify.sh": "scripts/verify.sh"}

#: The generated pipeline names the current paths; the previous release's
#: pipeline named the retired ones.
PIPELINE_SPELLINGS = (
    ("ops/ci-extra.yml", "ci-extra.yml"),
    ("ops/scripts/verify.sh", "scripts/verify.sh"),
)

#: The pipeline's include, before and after the migration.
RETIRED_INCLUDE = "- local: ci-extra.yml"
MIGRATED_INCLUDE = "- local: ops/ci-extra.yml"

#: The generated unit and boot hook: where the scaffolder writes them now, and
#: where the previous release kept them.
SCAFFOLDED = {
    "ops/osprey.service": "osprey.service",
    "ops/scripts/osprey-boot-hook.sh": "scripts/osprey-boot-hook.sh",
}

#: Operator files at retired paths, with no counterpart in the fixture.
OPERATOR_FILES = {
    "scripts/private/token.txt": "not for the image\n",
    "services/worker/Dockerfile": "FROM python:3.12\n",
    "data/code_review/reviewer.md": "# The reviewer's prompt.\n",
}

#: A certificate set as a lane minted it under the data tree.
CERTIFICATES = (
    "data/.runtime/bluesky_curve/bridge/proxy.key_secret",
    "data/.runtime/bluesky_curve/queueserver/publisher.key",
)

#: Canopy staging a previous build left at the root.
CANOPY = (
    "canopy_payload/index.json",
    "canopy-1.0-py3-none-any.whl",
    "pydmconverter-1.0-py3-none-any.whl",
)

#: A ``.dockerignore`` naming a path under a moved entry, before and after.
DOCKERIGNORE = "scripts/private/\n"
MIGRATED_DOCKERIGNORE = "ops/scripts/private/\n"

#: A profile value naming a retired path, added under ``config:``, and the
#: value the migration writes.
PROFILE_VALUE = "  code_review:\n    prompt: ./data/code_review/reviewer.md\n"
MIGRATED_PROFILE_VALUE = "    prompt: ./agent/code_review/reviewer.md\n"

#: The previous release's profile names a CANOPY home screen, and its repo
#: holds no application and so no screen library: no build of it passes with
#: the key, migrated or not.
HOME_SCREEN = "  web.panels.canopy.home_screen: facility/overview\n"

#: The first and last lines of the refusal: the block a user pastes.
REFUSAL_OPENS = "# The deployment repo at "
REFUSAL_CLOSES = "# or run: osprey init --revalidate"

#: What revalidate's passes that are not about the layout write, so the only
#: files the printed block alone leaves different from revalidate.
NON_LAYOUT_PREFIXES = (".osprey/preset-base/", "personas/", "tiers/")  # retired-spelling
NON_LAYOUT_FILES = ("profile.yml",)

#: What a build writes beside its own render that a migration never touches,
#: and git's own directory.
NOT_SOURCES = (".git", "build", "var")

#: Tracked moves the migration stages as renames: retired path, current path.
RENAMES = (
    ("ci-extra.yml", "ops/ci-extra.yml"),
    ("osprey.service", "ops/osprey.service"),
    ("scripts/osprey-boot-hook.sh", "ops/scripts/osprey-boot-hook.sh"),
    ("services/worker/Dockerfile", "ops/services/worker/Dockerfile"),
    ("data/code_review/reviewer.md", "agent/code_review/reviewer.md"),
)

#: What both routes leave for the operator: the host steps and the files whose
#: inner paths nothing rewrites.
HOST_STEPS = (
    "# Host steps, once the files moved:",
    "osprey build",
    "systemctl --user daemon-reload",
    "systemctl --user enable --now osprey.service",
    "crontab -e, replacing the osprey-boot-hook lines with:",
    "# Check by hand; the paths inside these are not rewritten:",
    "ops/ci-extra.yml",
    "ops/services/worker/Dockerfile",
    "ops/scripts/ (operator scripts)",
)

#: The entries ``init --force`` replaces under the two groups, by design.
REPLACED_GROUP_ENTRIES = ("agent/context", "agent/watchdogs", "agent/code_review")

#: A git identity and no developer config (signing, templates, hooks).
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "OSPREY",
    "GIT_COMMITTER_NAME": "OSPREY",
    "GIT_AUTHOR_EMAIL": "osprey@example.org",
    "GIT_COMMITTER_EMAIL": "osprey@example.org",
}


@pytest.fixture(scope="module")
def _git_env() -> Iterator[None]:
    """Hold :data:`GIT_ENV` for every git a module fixture or test runs."""
    with pytest.MonkeyPatch.context() as patch:
        for name, value in GIT_ENV.items():
            patch.setenv(name, value)
        yield


def _git(root: Path, *args: str) -> str:
    # No detached auto-gc: it repacks the objects a test is copying.
    return subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


def _commit_all(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _extend(repo: Path) -> None:
    """Add to the previous release's repo what a deployment in service also holds."""
    ci = exemplar_source_files(with_ci=True)
    pipeline = ci[".gitlab-ci.yml"]
    for current, retired in PIPELINE_SPELLINGS:
        assert current in pipeline, current
        pipeline = pipeline.replace(current, retired)
    assert RETIRED_INCLUDE in pipeline
    ci[".gitlab-ci.yml"] = pipeline
    _write(repo, {old: ci[new] for new, old in CI_FILES.items()})
    # The scaffolder writes the current paths; the previous release kept the
    # same files at the retired ones.
    scaffold_systemd_unit(repo)
    for new, old in SCAFFOLDED.items():
        (repo / new).rename(repo / old)
    shutil.rmtree(repo / "ops")
    _write(repo, OPERATOR_FILES)
    _write(repo, {rel: f"minted {rel}\n" for rel in CERTIFICATES})
    _write(repo, dict.fromkeys(CANOPY, "staged\n"))
    _write(repo, {".dockerignore": DOCKERIGNORE})
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert text.count("\nconfig:\n") == 1
    assert text.count(HOME_SCREEN) == 1
    text = text.replace(HOME_SCREEN, "").replace("\nconfig:\n", f"\nconfig:\n{PROFILE_VALUE}")
    profile.write_text(text, encoding="utf-8")


def _retired_tree(parent: Path) -> Path:
    """The extended previous-release repo, committed to its own repository."""
    repo = _previous_release_repo(parent / PROJECT)
    _extend(repo)
    _git(repo, "init", "-q")
    _commit_all(repo, "previous release, in service")
    return repo


def _copy(repo: Path, parent: Path) -> Path:
    return Path(shutil.copytree(repo, parent / PROJECT, symlinks=True))


def _sources(repo: Path) -> dict[str, bytes]:
    """Every file of the deployment a migration may touch, with its bytes."""
    files: dict[str, bytes] = {}
    for path in sorted(repo.rglob("*")):
        relative = path.relative_to(repo)
        if relative.parts[0] in NOT_SOURCES or not path.is_file():
            continue
        files[relative.as_posix()] = path.read_bytes()
    return files


def _snapshot(repo: Path) -> dict[str, bytes | None]:
    """Every path in the deployment, git's directory and the state zone included."""
    return {
        path.relative_to(repo).as_posix(): (
            path.read_bytes() if path.is_file() and not path.is_symlink() else None
        )
        for path in sorted(repo.rglob("*"))
    }


def _written_by_non_layout_passes(rel: str) -> bool:
    if rel in NON_LAYOUT_FILES or rel.startswith(NON_LAYOUT_PREFIXES):
        return True
    parts = rel.split("/")
    return len(parts) == 4 and parts[:2] == ["agent", "context"] and parts[3] == ".gitkeep"


def _refusal_message(repo: Path) -> str:
    with pytest.raises(RetiredRootRefused) as refusal:
        refuse_retired_layout(repo)
    return refusal.value.format_message()


def _refusal_block(output: str) -> str:
    """The refusal's lines, from its first comment through the alternative command."""
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(REFUSAL_OPENS))
    end = lines.index(REFUSAL_CLOSES, start)
    return "\n".join(lines[start : end + 1]) + "\n"


def _assert_refused(repo: Path, output: str, exit_code: int) -> None:
    assert exit_code == 1, output
    assert REFUSAL_OPENS + str(repo.resolve()) in output, output
    assert f"# So does the facility tree at {(repo / FACILITY).resolve()}" in output, output
    assert REFUSAL_CLOSES in output, output
    assert "Traceback" not in output, output
    assert not (repo / "build").exists()


def _normalised(output: str) -> list[str]:
    return [" ".join(line.split()) for line in output.splitlines()]


def _assert_names_host_steps(repo: Path, output: str) -> None:
    lines = _normalised(output)
    for step in HOST_STEPS:
        assert any(step in line for line in lines), (step, output)
    unit = f"cp {(repo / 'ops' / 'osprey.service').resolve()} ~/.config/systemd/user/"
    assert any(unit in line for line in lines), output
    hook = str((repo / "ops" / "scripts" / "osprey-boot-hook.sh").resolve())
    assert any(line.startswith("# @reboot ") and hook in line for line in lines), output


@pytest.fixture(scope="module")
def retired(tmp_path_factory: pytest.TempPathFactory, _git_env: None) -> Path:
    """The extended previous-release repo every test copies; never changed itself."""
    return _retired_tree(tmp_path_factory.mktemp("retired"))


@dataclass(frozen=True)
class Migrated:
    """A retired-layout deployment taken through refusal, ``--revalidate`` and a build.

    Attributes:
        repo: The deployment.
        refusal: What the build before the migration printed.
        revalidate: What ``--revalidate`` printed.
        sources: The deployment's files just after the migration.
        tracked: ``git status`` just after the migration.
        built: What the build after the migration printed.
        built_code: That build's exit code.
    """

    repo: Path
    refusal: str
    revalidate: str
    sources: dict[str, bytes]
    tracked: str
    built: str
    built_code: int


@pytest.fixture(scope="module")
def revalidated(
    retired: Path, tmp_path_factory: pytest.TempPathFactory, _git_env: None
) -> Migrated:
    """A copy migrated by ``osprey init --revalidate``, then built."""
    repo = _copy(retired, tmp_path_factory.mktemp("revalidated"))
    refused = run_build(repo)
    _assert_refused(repo, refused.output, refused.exit_code)
    result = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    sources = _sources(repo)
    tracked = _git(repo, "status", "--porcelain=v1", "-uall")
    built = run_build(repo)
    return Migrated(
        repo=repo,
        refusal=refused.output,
        revalidate=result.output,
        sources=sources,
        tracked=tracked,
        built=built.output,
        built_code=built.exit_code,
    )


# ── (a) every command refuses before any write ──────────────────────────────


_CASES = [
    pytest.param(args, id=" ".join(args[:4]))
    for invocations in REFUSING.values()
    for args in invocations
]


@pytest.mark.parametrize("args", _CASES)
def test_every_command_refuses_before_any_write(
    retired: Path, tmp_path: Path, args: list[str]
) -> None:
    repo = _copy(retired, tmp_path)
    export = tmp_path / "export.csv"
    export.write_text("address\nSR:NEW:CHANNEL\n", encoding="utf-8")
    values = {
        "repo": str(repo),
        "tier": str(sorted((repo / "tiers").glob("*.yml"))[0]),
        "export": str(export),
        "out": str(tmp_path / "render"),
    }
    expected = _refusal_message(repo)
    before = _snapshot(repo)

    result = CliRunner().invoke(cli, [arg.format(**values) for arg in args])

    assert result.exit_code == 1, result.output
    assert result.output.count(expected) == 1, result.output
    assert "Traceback" not in result.output
    assert _snapshot(repo) == before
    assert not (tmp_path / "render").exists()


def test_the_refusal_covers_root_and_facility_together(retired: Path) -> None:
    block = _refusal_message(retired).splitlines()

    for old, new in (
        ("rules", "agent/rules"),
        ("web-terminal-context", "agent/context"),
        ("services", "ops/services"),
        ("scripts", "ops/scripts"),
        ("osprey.service", "ops/osprey.service"),
        ("ci-extra.yml", "ops/ci-extra.yml"),
        ("data/code_review", "agent/code_review"),
        ("data/.runtime", "var/runtime"),
    ):
        assert any(line.endswith(f"mv {old} {new}") for line in block), (old, block)
    assert any(f" {FACILITY}/simulation/" in line for line in block), block
    assert any(line.endswith(" .dockerignore") for line in block), block
    assert any(line.endswith(" .gitlab-ci.yml") for line in block), block
    assert any(line.endswith(" profile.yml") for line in block), block
    for rel in ("canopy_payload", *CANOPY[1:]):
        assert any(line.endswith(f"-- {rel}") for line in block), (rel, block)


# ── (b) both routes, one tree, which builds ─────────────────────────────────


def test_the_revalidated_tree_holds_nothing_retired(revalidated: Migrated) -> None:
    repo = revalidated.repo

    assert find_retired_root(repo).empty
    assert find_retired(repo / FACILITY).empty
    for gone in ("rules", "web-terminal-context", "services", "scripts", "osprey.service"):
        assert not (repo / gone).exists(), gone
    for gone in ("ci-extra.yml", "data/code_review", "data/.runtime", *CANOPY):
        assert not (repo / gone).exists(), gone
    assert not (repo / "canopy_payload").exists()
    for rel in CERTIFICATES:
        moved = repo / "var" / "runtime" / rel.removeprefix("data/.runtime/")
        assert moved.read_text(encoding="utf-8") == f"minted {rel}\n", rel
    profile = (repo / "profile.yml").read_text(encoding="utf-8")
    assert MIGRATED_PROFILE_VALUE in profile
    assert "data/code_review" not in profile
    pipeline = (repo / ".gitlab-ci.yml").read_text(encoding="utf-8")
    assert MIGRATED_INCLUDE in pipeline, pipeline
    assert RETIRED_INCLUDE not in pipeline, pipeline


def test_the_migrated_tree_builds(revalidated: Migrated) -> None:
    assert revalidated.built_code == 0, revalidated.built
    assert "Traceback" not in revalidated.built
    assert find_retired_root(revalidated.repo).empty


def test_the_moves_keep_their_history(revalidated: Migrated, tmp_path: Path) -> None:
    staged = _git(revalidated.repo, "diff", "--cached", "--name-status", "-M")
    for old, new in RENAMES:
        assert f"\t{old}\t{new}\n" in staged, (old, staged)
    rule = next(rel for rel in sorted(revalidated.sources) if rel.startswith("agent/rules/"))

    repo = _copy(revalidated.repo, tmp_path)
    _git(repo, "commit", "-q", "-m", "current root layout")
    log = _git(repo, "log", "--follow", "--oneline", "--", rule)
    assert len(log.splitlines()) == 2, (rule, log)


def test_the_printed_block_ends_where_revalidate_does(
    revalidated: Migrated, retired: Path, tmp_path: Path, _git_env: None
) -> None:
    repo = _copy(retired, tmp_path)
    refused = run_build(repo)
    _assert_refused(repo, refused.output, refused.exit_code)

    subprocess.run(
        ["sh", "-e"],
        input=_refusal_block(refused.output),
        text=True,
        check=True,
        capture_output=True,
        timeout=120,
    )
    # The block moves the files and says to run the command again; revalidate
    # then has no move left, only the passes that are not about the layout.
    assert find_retired_root(repo).empty
    assert find_retired(repo / FACILITY).empty
    blocked = _sources(repo)
    differ = sorted(
        rel
        for rel in blocked.keys() | revalidated.sources.keys()
        if blocked.get(rel) != revalidated.sources.get(rel)
    )
    # What only the non-layout passes write: the preset base, the tier files in
    # their current and retired directories, the profile they upgrade, and
    # empty per-tier context dirs.
    assert all(_written_by_non_layout_passes(rel) for rel in differ), differ
    profile = blocked["profile.yml"].decode("utf-8")
    assert MIGRATED_PROFILE_VALUE in profile
    assert "data/code_review" not in profile
    result = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert _sources(repo) == revalidated.sources
    assert _git(repo, "status", "--porcelain=v1", "-uall") == revalidated.tracked


def test_a_second_run_is_a_no_op(revalidated: Migrated, tmp_path: Path, _git_env: None) -> None:
    repo = _copy(revalidated.repo, tmp_path)
    before = _snapshot(repo)

    result = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "nothing to rewrite" in result.output
    assert "Host steps" not in result.output
    assert _snapshot(repo) == before


# ── (c) a clash refuses by both routes; a merge keeps every file ────────────


def test_a_clash_refuses_by_both_routes_naming_every_file(
    retired: Path, tmp_path: Path, _git_env: None
) -> None:
    repo = _copy(retired, tmp_path)
    rule = sorted(p.name for p in (repo / "rules").iterdir() if p.is_file())[0]
    clashes = (
        (f"rules/{rule}", f"agent/rules/{rule}"),
        ("ci-extra.yml", "ops/ci-extra.yml"),
        ("services/worker/Dockerfile", "ops/services/worker/Dockerfile"),
    )
    _write(repo, {new: "# Kept here too.\n" for _old, new in clashes})
    _commit_all(repo, "copies at the current paths")
    before = _snapshot(repo)

    built = run_build(repo)
    revalidate = CliRunner().invoke(init, [str(repo), "--revalidate"])

    assert built.exit_code == 1, built.output
    assert revalidate.exit_code != 0, revalidate.output
    assert isinstance(revalidate.exception, SystemExit), revalidate.exception
    assert "Traceback" not in built.output + revalidate.output
    root = repo.resolve()
    refusal = _refusal_block(built.output).splitlines()
    named = " ".join(revalidate.output.split())
    for old, new in clashes:
        line = f"# {root / old} and {root / new} both exist: keep one, delete the other."
        assert line in refusal, (old, built.output)
        assert f"{old} and {new}" in named, (old, revalidate.output)
    assert _snapshot(repo) == before


def test_a_merge_into_a_successor_keeps_every_file(
    retired: Path, tmp_path: Path, _git_env: None
) -> None:
    repo = _copy(retired, tmp_path)
    old_rules = sorted(
        p.relative_to(repo / "rules").as_posix() for p in (repo / "rules").rglob("*") if p.is_file()
    )
    _write(
        repo, {"agent/rules/current.md": "# Already moved.\n", "ops/scripts/current.sh": "true\n"}
    )
    _commit_all(repo, "files already at the current paths")

    result = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    rules = sorted(
        p.relative_to(repo / "agent" / "rules").as_posix()
        for p in (repo / "agent" / "rules").rglob("*")
        if p.is_file()
    )
    assert rules == sorted([*old_rules, "current.md"])
    for rel in ("current.sh", "verify.sh", "osprey-boot-hook.sh", "private/token.txt"):
        assert (repo / "ops" / "scripts" / rel).is_file(), rel
    assert not (repo / "rules").exists()
    assert not (repo / "scripts").exists()
    assert find_retired_root(repo).empty


# ── (d) the .dockerignore follows the move; both outputs name the hand steps ─


def test_the_migrated_dockerignore_keeps_the_moved_entry_out_of_the_image(
    revalidated: Migrated,
) -> None:
    repo = revalidated.repo
    assert revalidated.built_code == 0, revalidated.built
    assert (repo / ".dockerignore").read_text(encoding="utf-8") == MIGRATED_DOCKERIGNORE
    assert (repo / "ops" / "scripts" / "private" / "token.txt").is_file()

    contexts = sorted(path for path in (repo / "build" / IMAGE_DIR_NAME).iterdir() if path.is_dir())
    assert contexts, revalidated.built
    for context in contexts:
        assert (context / "ops" / "scripts" / "verify.sh").is_file(), context
        assert not (context / "ops" / "scripts" / "private").exists(), context
        assert not (context / "scripts").exists(), context
        assert not (context / "var").exists(), context
        assert not (context / "data" / ".runtime").exists(), context
        assert not (context / "canopy_payload").exists(), context
        assert not list(context.glob("*.whl")), context


def test_both_outputs_name_the_host_steps_and_the_files_to_check(
    revalidated: Migrated,
) -> None:
    _assert_names_host_steps(revalidated.repo, _refusal_block(revalidated.refusal))
    steps = revalidated.revalidate[revalidated.revalidate.index("Left for you to do by hand") :]
    _assert_names_host_steps(revalidated.repo, steps)


# ── (e) init --force over the migrated tree ─────────────────────────────────


def test_force_keeps_operator_content_and_replaces_its_own_entries(
    revalidated: Migrated, tmp_path: Path, _git_env: None
) -> None:
    repo = _copy(revalidated.repo, tmp_path)
    for entry in REPLACED_GROUP_ENTRIES:
        assert entry in MATERIALIZED_SOURCE_ENTRIES, entry
    _write(
        repo,
        {
            "agent/rules/operator.md": "# The operator's own rule.\n",
            "agent/skills/operator-skill/SKILL.md": "# The operator's own skill.\n",
            "ops/project/notes.md": "operator notes\n",
            **{
                f"{entry}/operator-note.md": "replaced by design\n"
                for entry in REPLACED_GROUP_ENTRIES
            },
        },
    )
    replaced = tuple(entry for entry in MATERIALIZED_SOURCE_ENTRIES if "/" in entry)

    def kept(rel: str) -> bool:
        return rel.startswith(("agent/", "ops/")) and not any(
            rel == entry or rel.startswith(f"{entry}/") for entry in replaced
        )

    before = {rel: data for rel, data in _sources(repo).items() if kept(rel)}
    assert "agent/rules/operator.md" in before
    assert "ops/osprey.service" in before
    assert "ops/services/worker/Dockerfile" in before

    result = CliRunner().invoke(
        cli, ["init", str(repo), "--preset", "demo", "--force"], catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    after = _sources(repo)
    assert {rel: after.get(rel) for rel in before} == before
    for entry in REPLACED_GROUP_ENTRIES:
        assert not (repo / entry / "operator-note.md").exists(), entry
    assert (repo / "agent" / "context").is_dir()
    assert (repo / "agent" / "watchdogs").is_dir()
    assert (repo / "agent" / "code_review").is_dir()
    assert not (repo / HELD_SOURCE_ZONE_DIRNAME).exists()
    assert not any(path.name == HELD_SOURCE_ZONE_DIRNAME for path in repo.rglob("*"))
    assert find_retired_root(repo).empty
