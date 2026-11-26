"""The printed migration block and ``osprey init --revalidate`` leave the same tree.

A repo-scoped verb on a retired-layout repository refuses and prints a block a
user pastes into a shell. Pasted whole into ``sh``, with the installed
``osprey`` first on ``PATH``, that block leaves the same tree that
``osprey init --revalidate`` leaves on an identical copy: the same bytes in
every file, the same index (``git ls-files -s``) and the same worktree status,
in the deployment and in any other repository the migration touches. Both
trees build.

The block's own lines are held to that on a third copy, pasted without the
closing command: they alone leave nothing at a retired path and bring every
file ``--revalidate`` leaves at a moved path, so the closing command cannot
stand in for a move line that is missing or wrong.

Three repositories are taken both ways: the previous release (pre-epic, its
facility tree at ``data/facility`` with the old facility slots); the same
release with its rendered sources under a non-default in-repo ``data:`` root;
and the previous release whose facility tree is a repository of its own, where
the commands for that repository run by both routes. A refusal that only names
files to place by hand prints no command to run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey import repo_layout
from osprey.cli.init_cmd import init
from osprey.repo_retired import RetiredRootRefused, find_retired_root, refuse_retired_layout
from osprey_facility.errors import _PLACE_THESE, _revalidate_command
from osprey_facility.retired import find_retired
from tests._builds import run_build
from tests.cli.test_init_revalidate import _previous_release_repo
from tests.cli.test_root_migration_round_trip import (
    GIT_ENV,
    PROJECT,
    _commit_all,
    _extend,
    _git,
    _retired_tree,
)

pytestmark = [pytest.mark.slow]

#: The installed console script the printed block's closing command runs.
OSPREY = Path(sys.executable).parent / "osprey"

#: The facility tree in the previous release, and where the migration puts it.
RETIRED_FACILITY = "data/facility"
FACILITY = "facility"

#: The non-default in-repo data root, and what it holds in the previous release.
DATA_ROOT = "content"
RENDERED_SOURCES = ("facility", "canopy", "landing")

#: Where revalidate's commands for another repository start, in its output.
ELSEWHERE_STEPS = "Run these commands to move them:"


@pytest.fixture(scope="module")
def _git_env() -> Iterator[None]:
    """Hold :data:`GIT_ENV` for every git a module fixture or test runs."""
    with pytest.MonkeyPatch.context() as patch:
        for name, value in GIT_ENV.items():
            patch.setenv(name, value)
        yield


@dataclass(frozen=True)
class Routes:
    """One retired-layout repository taken by each route.

    Attributes:
        block: The copy the pasted refusal block migrated.
        moved: The copy the block's lines migrated, pasted without the
            closing command.
        revalidated: The copy ``osprey init --revalidate`` migrated.
        refusal: What the refusing verb printed on the block's copy.
        revalidate: What ``--revalidate`` printed.
        others: Paths, relative to each copy, of the other repositories inside it.
    """

    block: Path
    moved: Path
    revalidated: Path
    refusal: str
    revalidate: str
    others: tuple[str, ...]


def _shell_env() -> dict[str, str]:
    return {
        **os.environ,
        **GIT_ENV,
        "PATH": f"{OSPREY.parent}{os.pathsep}{os.environ['PATH']}",
    }


def _sh(script: str, cwd: Path) -> None:
    """Run *script* under ``sh -e`` from *cwd*, as a user pastes it."""
    cwd.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        ["sh", "-e"],
        input=script,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=_shell_env(),
        timeout=600,
    )
    assert done.returncode == 0, (script, done.stdout, done.stderr)


def _copy(source: Path, parent: Path) -> Path:
    """*source* under *parent*, at the same directory name."""
    parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copytree(source, parent / PROJECT, symlinks=True))


def _elsewhere_steps(output: str) -> str:
    """The commands ``--revalidate`` printed for other repositories, as one script."""
    lines = output.splitlines()
    script: list[str] = []
    for start, line in enumerate(lines):
        if not line.rstrip().endswith(ELSEWHERE_STEPS):
            continue
        for step in lines[start + 1 :]:
            if not step.strip() or step.lstrip().startswith("#"):
                break
            script.append(step.strip())
    return "\n".join(script) + "\n" if script else ""


def _refused(repo: Path) -> str:
    """What a build of the retired-layout *repo* prints: the refusal, and nothing built."""
    refused = run_build(repo)
    assert refused.exit_code == 1, refused.output
    assert "Traceback" not in refused.output, refused.output
    assert not (repo / "build").exists()
    return refused.output


def _without_closing_command(refusal: str, repo: Path) -> str:
    """The refusal block for *repo* up to, and without, its closing command."""
    lines = refusal.splitlines()
    assert lines[-1] == _revalidate_command(repo.resolve()), refusal
    return "\n".join(lines[:-1]) + "\n"


def _by_both_routes(source: Path, parent: Path, others: tuple[str, ...] = ()) -> Routes:
    """Take a copy of *source* through each route, and one through the block's lines alone."""
    block = _copy(source, parent / "block")
    moved = _copy(source, parent / "moved")
    revalidated = _copy(source, parent / "revalidated")

    refusal = _refused(block)
    _sh(refusal, parent / "elsewhere")
    _sh(_without_closing_command(_refused(moved), moved), parent / "elsewhere")

    result = CliRunner().invoke(init, [str(revalidated), "--revalidate"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    steps = _elsewhere_steps(result.output)
    assert bool(steps) == bool(others), result.output
    if steps:
        _sh(steps, parent / "elsewhere")
    return Routes(block, moved, revalidated, refusal, result.output, others)


def _files(repo: Path) -> dict[str, bytes | str]:
    """Every path under *repo* but git's own directories: file bytes, or a link target."""
    found: dict[str, bytes | str] = {}
    for path in sorted(repo.rglob("*")):
        relative = path.relative_to(repo)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            found[relative.as_posix()] = os.readlink(path)
        elif path.is_file():
            found[relative.as_posix()] = path.read_bytes()
    return found


def _repositories(routes: Routes) -> list[tuple[Path, Path]]:
    """Each repository of the block's copy beside its counterpart in the revalidated one."""
    pairs = [(routes.block, routes.revalidated)]
    pairs += [(routes.block / rel, routes.revalidated / rel) for rel in routes.others]
    return pairs


def _previous_release(parent: Path) -> Path:
    """The previous release in service, committed: facility tree at ``data/facility``."""
    return _retired_tree(parent)


def _in_repo_data_root(parent: Path) -> Path:
    """The previous release with its rendered sources under ``content/``, named by ``data:``."""
    repo = _previous_release_repo(parent / PROJECT)
    _extend(repo)
    for name in RENDERED_SOURCES:
        old = repo / "data" / name
        if old.exists():
            (repo / DATA_ROOT).mkdir(exist_ok=True)
            old.rename(repo / DATA_ROOT / name)
    assert (repo / DATA_ROOT / "facility").is_dir()
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    text = text.replace("./data/facility", f"./{DATA_ROOT}/facility")
    text = text.replace(" data/facility", f" {DATA_ROOT}/facility")
    profile.write_text(f"{text.rstrip()}\ndata: {DATA_ROOT}\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _commit_all(repo, "previous release, rendered sources under content/")
    return repo


def _facility_repository(parent: Path) -> Path:
    """The previous release whose facility tree at ``data/facility`` is its own repository."""
    repo = _previous_release_repo(parent / PROJECT)
    _extend(repo)
    tree = repo / RETIRED_FACILITY
    _git(tree, "init", "-q")
    _commit_all(tree, "the operator's facility tree")
    _git(repo, "init", "-q")
    _commit_all(repo, "previous release, facility tree in its own repository")
    return repo


#: Each fixture: its builder, and the other repositories it holds once migrated.
FIXTURES: dict[str, tuple[Callable[[Path], Path], tuple[str, ...]]] = {
    "previous-release": (_previous_release, ()),
    "in-repo-data-root": (_in_repo_data_root, ()),
    "facility-in-another-repository": (_facility_repository, (FACILITY,)),
}


@pytest.fixture(scope="module", params=sorted(FIXTURES))
def routes(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, _git_env: None
) -> Routes:
    """One fixture, taken by both routes."""
    build, others = FIXTURES[request.param]
    source = build(tmp_path_factory.mktemp(f"{request.param}-source"))
    return _by_both_routes(source, tmp_path_factory.mktemp(request.param), others)


def test_the_refusal_closes_on_the_command_that_finishes_it(routes: Routes) -> None:
    lines = routes.refusal.splitlines()
    assert lines[-1] == _revalidate_command(routes.block.resolve()), routes.refusal


def test_both_routes_leave_the_same_files(routes: Routes) -> None:
    for block, revalidated in _repositories(routes):
        assert _files(block) == _files(revalidated), block


def test_both_routes_leave_the_same_index_and_status(routes: Routes) -> None:
    for block, revalidated in _repositories(routes):
        for args in (("ls-files", "-s"), ("status", "--porcelain=v1", "-uall")):
            assert _git(block, *args) == _git(revalidated, *args), (block, args)


def test_both_routes_finish_the_migration(routes: Routes) -> None:
    for repo in (routes.block, routes.revalidated):
        assert find_retired_root(repo).empty, repo
        assert find_retired(repo / FACILITY).empty, repo
        assert not (repo / "data").exists(), repo
        assert not (repo / DATA_ROOT).exists(), repo


def _moved_paths(repo: Path) -> set[str]:
    """Every file of *repo* at or below a path a declared root move brings."""
    successors = [move.new for move in repo_layout.retired_moves()]
    return {
        path
        for path in _files(repo)
        if any(path == new or path.startswith(f"{new}/") for new in successors)
    }


def test_the_move_lines_alone_leave_nothing_at_a_retired_path(routes: Routes) -> None:
    repo = routes.moved
    assert find_retired_root(repo).empty, repo
    assert find_retired(repo / FACILITY).empty, repo
    assert not (repo / "data").exists(), repo
    assert not (repo / DATA_ROOT).exists(), repo


def test_the_move_lines_alone_bring_every_file_revalidate_leaves_at_a_moved_path(
    routes: Routes,
) -> None:
    # The rewrite after the moves may drop a file; it brings none to a moved path.
    expected = _moved_paths(routes.revalidated)
    assert expected, routes.revalidate
    assert expected - _moved_paths(routes.moved) == set()


def test_both_trees_build(routes: Routes, _git_env: None) -> None:
    for repo in (routes.block, routes.revalidated):
        built = run_build(repo)
        assert built.exit_code == 0, built.output
        assert "Traceback" not in built.output, built.output


def test_a_leftover_only_refusal_prints_no_command(tmp_path: Path, _git_env: None) -> None:
    repo = _copy(_previous_release(tmp_path / "source"), tmp_path / "migrated")
    migrated = CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)
    assert migrated.exit_code == 0, migrated.output
    (repo / "data").mkdir()
    (repo / "data" / "notes.md").write_text("the operator's notes\n", encoding="utf-8")

    with pytest.raises(RetiredRootRefused) as refused:
        refuse_retired_layout(repo)

    message = refused.value.format_message()
    lines = message.splitlines()
    assert lines[-1] == _PLACE_THESE, message
    assert "data/notes.md" in message, message
    assert not any("osprey init --revalidate" in line for line in lines), message
    assert not any(line.strip() and not line.startswith("#") for line in lines), message
