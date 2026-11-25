"""A demo deployment on the retired facility layout migrates to what a fresh one builds.

The retired layout is laid from the table below, written out here rather than
read from :func:`osprey_facility.source_tree.retired_moves`, so the test holds
the declaration to the layout it replaced instead of to itself. Each old-layout
deployment is a fresh ``osprey init`` of the demo with every source moved back
to its old path and every authored ``deck:`` value naming the old decks
directory, committed to its own git repository.

``osprey build`` refuses such a deployment and prints the commands that move
it. ``osprey init --revalidate`` migrates one copy; the build that follows
writes the same facility file, benchmark dataset and views, byte for byte, as
the build of a fresh demo. A second copy, migrated by running the printed
commands through ``sh``, ends in the same tree as the revalidated one.
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

from osprey.cli.init_cmd import init
from osprey_facility.retired import find_retired
from tests._builds import init_project, run_build

pytestmark = [pytest.mark.slow]

#: The project name, and each repo's directory name: equal names give equal
#: build outputs.
PROJECT = "demo"

#: The facility tree, relative to a deployment.
FACILITY = "data/facility"

#: Every retired path of the facility tree and where its source lives now,
#: relative to the facility root.
R1 = (
    ("classes.yaml", "vocabulary/classes.yaml"),
    ("signals.yaml", "vocabulary/signals.yaml"),
    ("documents.yaml", "provenance/documents.yaml"),
    ("policy.yaml", "provenance/policy.yaml"),
    ("fixes.yaml", "provenance/fixes.yaml"),
    ("models.yaml", "simulation/models.yaml"),
    ("seeds.yaml", "simulation/seeds.yaml"),
    ("decks", "simulation/decks"),
    ("measurement", "simulation/measurement"),
    ("scenarios", "simulation/scenarios"),
    ("benchmarks/cross_paradigm/queries", "benchmarks/channel_finder"),
)

#: The directories only the current layout has; the retired one has none of them.
CURRENT_ONLY = ("vocabulary", "provenance", "simulation", "benchmarks/channel_finder")

#: An authored ``deck:`` value in the current layout, and in the retired one.
DECK_NOW = "deck: simulation/decks/"
DECK_RETIRED = "deck: decks/"

#: The first and last lines of the refusal: the block a user pastes.
REFUSAL_OPENS = "# The facility tree at "
REFUSAL_CLOSES = "# or run: osprey init --revalidate"

#: What a build writes beside its own render that a migration never touches,
#: and git's own directory.
NOT_SOURCES = (".git", "build", "var")

#: A file a build writes that is not byte-reproducible: a database imported
#: from a view whose own bytes are compared.
UNREPRODUCIBLE_SUFFIX = ".duckdb"

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
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


def _facility(repo: Path) -> Path:
    return repo / FACILITY


def _deck_lines(models: Path) -> int:
    """How many lines of *models* name a deck at its current path: one edit each."""
    return sum(DECK_NOW in line for line in models.read_text(encoding="utf-8").splitlines())


def _retire_decks(models: Path) -> None:
    """Point every authored ``deck:`` value in *models* at the retired decks directory."""
    text = models.read_text(encoding="utf-8")
    assert DECK_NOW in text, models
    models.write_text(text.replace(DECK_NOW, DECK_RETIRED), encoding="utf-8")


def _old_layout(parent: Path) -> Path:
    """A demo deployment on the retired layout, committed to its own repository."""
    repo = init_project(parent, "demo", PROJECT)
    facility = _facility(repo)
    for old, new in R1:
        if (facility / new).exists():
            (facility / old).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(facility / new), str(facility / old))
    for directory in sorted(CURRENT_ONLY, key=lambda path: path.count("/"), reverse=True):
        if (facility / directory).is_dir():
            (facility / directory).rmdir()
    _retire_decks(facility / "models.yaml")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "retired layout")
    return repo


def _sources(repo: Path) -> dict[str, bytes]:
    """Every file of the deployment a migration may touch, with its bytes."""
    files: dict[str, bytes] = {}
    for path in sorted(repo.rglob("*")):
        relative = path.relative_to(repo)
        if relative.parts[0] in NOT_SOURCES or not path.is_file():
            continue
        files[relative.as_posix()] = path.read_bytes()
    return files


def _rendered(repo: Path) -> dict[str, bytes]:
    """The facility file and every file under the render's ``data/``, with their bytes."""
    build = repo / "build"
    files = {"facility.json": (build / "facility.json").read_bytes()}
    for path in sorted((build / "data").rglob("*")):
        if path.is_file() and path.suffix != UNREPRODUCIBLE_SUFFIX:
            files[path.relative_to(build).as_posix()] = path.read_bytes()
    return files


def _refusal_block(output: str) -> str:
    """The refusal's lines, from its first comment through the alternative command."""
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(REFUSAL_OPENS))
    end = lines.index(REFUSAL_CLOSES, start)
    return "\n".join(lines[start : end + 1]) + "\n"


def _assert_refused(repo: Path, output: str, exit_code: int) -> None:
    assert exit_code == 1, output
    assert REFUSAL_OPENS + str(_facility(repo).resolve()) in output, output
    assert REFUSAL_CLOSES in output, output
    assert "Traceback" not in output, output
    assert not (repo / "build").exists()


@pytest.fixture(scope="module")
def fresh(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """What the build of a fresh demo deployment renders."""
    repo = init_project(tmp_path_factory.mktemp("fresh"), "demo", PROJECT)
    result = run_build(repo)
    assert result.exit_code == 0, result.output
    return _rendered(repo)


@dataclass(frozen=True)
class Migrated:
    """An old-layout deployment taken through refusal, ``--revalidate`` and a build.

    Attributes:
        repo: The deployment.
        refusal: What the build before the migration printed.
        refusal_code: That build's exit code.
        sources: The deployment's files just after the migration.
        tracked: ``git status`` just after the migration.
        built: What the build after the migration printed.
        built_code: That build's exit code.
    """

    repo: Path
    refusal: str
    refusal_code: int
    sources: dict[str, bytes]
    tracked: str
    built: str
    built_code: int


@pytest.fixture(scope="module")
def revalidated(tmp_path_factory: pytest.TempPathFactory, _git_env: None) -> Migrated:
    """An old-layout deployment migrated by ``osprey init --revalidate``, then built."""
    repo = _old_layout(tmp_path_factory.mktemp("revalidated"))
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
        refusal_code=refused.exit_code,
        sources=sources,
        tracked=tracked,
        built=built.output,
        built_code=built.exit_code,
    )


def test_the_old_layout_holds_no_current_path(tmp_path: Path, _git_env: None) -> None:
    facility = _facility(_old_layout(tmp_path))

    for directory in CURRENT_ONLY:
        assert not (facility / directory).exists(), directory
    for old in ("classes.yaml", "models.yaml", "seeds.yaml", "decks", "scenarios"):
        assert (facility / old).exists(), old
    assert (facility / "benchmarks" / "cross_paradigm" / "queries").is_dir()
    assert DECK_NOW not in (facility / "models.yaml").read_text(encoding="utf-8")


def test_build_refuses_the_old_layout_with_every_move(revalidated: Migrated) -> None:
    output = revalidated.refusal.splitlines()
    facility = _facility(revalidated.repo)
    moved = [(old, new) for old, new in R1 if (facility / new).exists()]
    assert len(moved) >= 8, moved
    for old, new in moved:
        assert f"git mv {FACILITY}/{old} {FACILITY}/{new}" in output, (old, output)
    edits = [line for line in output if line.startswith("perl -pi -e 's{decks/}")]
    assert len(edits) == _deck_lines(facility / "simulation" / "models.yaml") > 0, output


def test_the_revalidated_build_renders_what_a_fresh_demo_does(
    revalidated: Migrated, fresh: dict[str, bytes]
) -> None:
    assert revalidated.built_code == 0, revalidated.built
    rendered = _rendered(revalidated.repo)

    assert rendered["facility.json"] == fresh["facility.json"]
    assert any(name.startswith("data/benchmarks/") for name in fresh)
    assert any(name.startswith("data/simulator/") for name in fresh)
    assert sorted(rendered) == sorted(fresh)
    differing = [name for name in fresh if rendered[name] != fresh[name]]
    assert differing == []


def test_stale_deck_values_alone_are_refused_with_the_edit(tmp_path: Path) -> None:
    repo = init_project(tmp_path, "demo", PROJECT)
    models = _facility(repo) / "simulation" / "models.yaml"
    decks = _deck_lines(models)
    _retire_decks(models)

    result = run_build(repo)

    _assert_refused(repo, result.output, result.exit_code)
    block = _refusal_block(result.output)
    edits = [line for line in block.splitlines() if line.startswith("perl -pi -e ")]
    assert len(edits) == decks > 0, block
    assert all(line.endswith(" simulation/models.yaml") for line in edits), block
    assert not [line for line in block.splitlines() if " mv " in f" {line}"], block

    subprocess.run(["sh"], input=block, text=True, check=True, timeout=120)

    assert DECK_RETIRED not in models.read_text(encoding="utf-8")
    assert find_retired(_facility(repo)).empty


def test_the_printed_commands_end_where_revalidate_does(
    revalidated: Migrated, tmp_path: Path, _git_env: None
) -> None:
    repo = _old_layout(tmp_path)
    refused = run_build(repo)
    _assert_refused(repo, refused.output, refused.exit_code)

    subprocess.run(["sh"], input=_refusal_block(refused.output), text=True, check=True, timeout=120)

    assert _sources(repo) == revalidated.sources
    assert _git(repo, "status", "--porcelain=v1", "-uall") == revalidated.tracked
