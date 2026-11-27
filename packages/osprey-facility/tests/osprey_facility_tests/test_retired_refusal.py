"""The refusal of a facility tree still laid out at retired paths.

A read of such a tree stops before anything is read, and what it prints is a
shell script: run from any directory, it moves every source to its current
path, so the tree holds nothing retired and builds, then runs
``osprey init --revalidate <repo>`` to finish the migration. The scripts run
here with a stand-in ``osprey`` that records how it was called. Old trees are
laid out from literal paths here, never from the declaration.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import types
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.simulation.engines import ENGINE_GROUP
from osprey_facility.build import build_facility
from osprey_facility.errors import (
    _PLACE_THESE,
    MOVE_HEADER,
    PASTE_HEADER,
    FacilityError,
    RetiredLayoutRefused,
    _revalidate_command,
)
from osprey_facility.layers.list import import_list
from osprey_facility.retired import (
    LineDelete,
    LineEdit,
    Move,
    find_retired,
    move_script,
    place_moves,
    refuse_retired,
    render_script,
    repository_root,
)
from osprey_facility.sources import load_sources
from osprey_facility.validate import run_stages
from osprey_facility_tests._harness import REPO_ROOT

DECK = REPO_ROOT / "src/osprey/templates/facilities/example/simulation/decks/LTB.json"

#: The engine the models name: one this module registers, so a build here asks
#: nothing of the engines the environment happens to install.
ENGINE = "deck_reader"

#: One model's entry, its ``deck`` value written in each style an author may use.
MODELS = {
    "block": f"# the ring\n- name: ring\n  engine: {ENGINE}\n  deck: decks/ring.json\n",
    "first-key": f"- deck: decks/ring.json\n  name: ring\n  engine: {ENGINE}\n",
    "quoted": f'- name: ring\n  engine: {ENGINE}\n  deck: "decks/ring.json"\n',
}


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.invalid")


@pytest.fixture
def decks_read(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Register engine ``ENGINE``, which reads the deck a build hands it; the decks read.

    The engine parses the file, so a build passes only when the model's ``deck``
    names a deck that is there. Every other entry-point group reads as installed.
    """
    read: list[Path] = []

    def prepare(deck: Path, *args: Any, **kwargs: Any) -> types.SimpleNamespace:
        json.loads(Path(deck).read_text(encoding="utf-8"))
        read.append(Path(deck))
        return types.SimpleNamespace(solve="open", length_m=1.0)

    def start_values(deck: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        json.loads(Path(deck).read_text(encoding="utf-8"))
        return {}

    def unused(*args: Any, **kwargs: Any) -> None:
        return None

    module = types.ModuleType(f"{__name__}.{ENGINE}")
    for name in ("locate", "describe", "build"):
        setattr(module, name, unused)
    module.prepare = prepare  # type: ignore[attr-defined]
    module.start_values = start_values  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    engines = metadata.EntryPoints(
        [metadata.EntryPoint(name=ENGINE, value=module.__name__, group=ENGINE_GROUP)]
    )
    installed = metadata.entry_points

    def entry_points(**params: Any) -> Any:
        return engines if params.get("group") == ENGINE_GROUP else installed(**params)

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    return read


def _write(root: Path, relative: str, text: str = "x\n") -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _old_tree(root: Path, models: str = MODELS["block"]) -> Path:
    _write(root, "identity.yaml", "code: demo\n")
    _write(root, "classes.yaml", "- class: Widget\n  parent: Magnet\n")
    _write(root, "models.yaml", models)
    (root / "decks").mkdir(parents=True)
    shutil.copy(DECK, root / "decks/ring.json")
    return root


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def _repository(root: Path, *, commit: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    if commit:
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "seed")
    return root


def _refusal(machine: Path, content: Path | None = None) -> RetiredLayoutRefused:
    with pytest.raises(RetiredLayoutRefused) as caught:
        refuse_retired(machine, content)
    return caught.value


def _run(message: str, elsewhere: Path) -> list[str]:
    """Run *message* under ``sh -e`` from *elsewhere*; each ``osprey`` call it made."""
    elsewhere.mkdir(exist_ok=True)
    bin_dir = elsewhere / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = elsewhere / "osprey-calls"
    stand_in = bin_dir / "osprey"
    stand_in.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n', encoding="utf-8")
    stand_in.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    done = subprocess.run(
        ["sh", "-e"],
        input=message,
        cwd=elsewhere,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    return calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


# --- when it refuses --------------------------------------------------------------------


def test_a_current_tree_is_not_refused(tmp_path: Path) -> None:
    _write(tmp_path, "vocabulary/classes.yaml")
    _write(tmp_path, "simulation/models.yaml", "- name: ring\n  deck: simulation/decks/r.json\n")

    refuse_retired(tmp_path)


def test_a_missing_tree_is_not_refused(tmp_path: Path) -> None:
    refuse_retired(tmp_path / "absent")


def test_the_refusal_holds_the_layout_it_found(tmp_path: Path) -> None:
    _old_tree(tmp_path)

    stop = _refusal(tmp_path)

    assert isinstance(stop, FacilityError)
    assert stop.layout == find_retired(tmp_path)
    assert stop.machine_dir == tmp_path.resolve()
    assert stop.content_dir is None


def test_reading_the_sources_refuses(tmp_path: Path) -> None:
    _old_tree(tmp_path)

    with pytest.raises(RetiredLayoutRefused):
        load_sources(tmp_path)


def test_running_the_stages_refuses_on_a_retired_content_path(tmp_path: Path) -> None:
    machine = tmp_path / "machine"
    content = tmp_path / "content"
    _write(machine, "identity.yaml", "code: demo\n")
    _write(content, "benchmarks/cross_paradigm/queries/tree_queries.json", "[]\n")

    with pytest.raises(RetiredLayoutRefused) as caught:
        run_stages(machine, project_name="demo", content_dir=content, extra_checks=())

    assert [move.root for move in caught.value.layout.moves] == [content]


def test_building_refuses(tmp_path: Path) -> None:
    _old_tree(tmp_path)

    with pytest.raises(RetiredLayoutRefused):
        build_facility(tmp_path, project_name="demo", extra_checks=())


def test_importing_a_channel_list_refuses_and_writes_nothing(tmp_path: Path) -> None:
    facility = _old_tree(tmp_path / "facility")
    listing = _write(tmp_path, "channels.csv", "SR:A:X\n")

    with pytest.raises(RetiredLayoutRefused):
        import_list(listing, facility)

    assert not (facility / "imported").exists()


# --- what it prints ---------------------------------------------------------------------


def test_the_message_is_comments_then_the_script(tmp_path: Path) -> None:
    _old_tree(tmp_path)
    root = tmp_path.resolve()

    lines = _refusal(tmp_path).message.splitlines()

    assert lines == [
        f"# The facility tree at {root} still holds sources at retired paths; nothing was read.",
        PASTE_HEADER,
        f"cd {root}",
        "mkdir -p vocabulary",
        "mv classes.yaml vocabulary/classes.yaml",
        "mkdir -p simulation",
        "mv models.yaml simulation/models.yaml",
        "mv decks simulation/decks",
        "perl -pi -e 's{decks/}{simulation/decks/} if $. == 4' simulation/models.yaml",
        f"osprey init --revalidate {root.parent}",
    ]


def test_the_closing_command_names_the_repo_given(tmp_path: Path) -> None:
    _old_tree(tmp_path / "facility")
    repo = tmp_path / "a repo"

    with pytest.raises(RetiredLayoutRefused) as caught:
        refuse_retired(tmp_path / "facility", repo=repo)

    assert caught.value.message.splitlines()[-1] == _revalidate_command(repo.resolve())
    assert caught.value.message.splitlines()[-1] == f"osprey init --revalidate '{repo.resolve()}'"


def test_in_a_repository_the_closing_command_names_its_top(tmp_path: Path) -> None:
    deployment = tmp_path / "deployment"
    facility = _old_tree(deployment / "facility")
    _repository(deployment)

    assert _refusal(facility).message.splitlines()[-1] == (
        f"osprey init --revalidate {deployment.resolve()}"
    )


def test_it_prints_alone_with_no_error_prefix(tmp_path: Path) -> None:
    _old_tree(tmp_path)
    stop = _refusal(tmp_path)
    stream = io.StringIO()

    stop.show(stream)

    assert stream.getvalue() == stop.message + "\n"


def test_tracked_sources_move_with_git_from_the_repository_top(tmp_path: Path) -> None:
    deployment = tmp_path / "deployment"
    facility = _old_tree(deployment / "data/facility")
    _repository(deployment)

    script = _refusal(facility).script

    assert script[0] == f"cd {deployment.resolve()}"
    assert "git mv data/facility/classes.yaml data/facility/vocabulary/classes.yaml" in script
    assert "git mv data/facility/decks data/facility/simulation/decks" in script
    assert "mkdir -p data/facility/vocabulary" in script
    assert "mkdir -p data/facility/simulation" in script
    assert script.index("mkdir -p data/facility/simulation") == (
        script.index("git mv data/facility/models.yaml data/facility/simulation/models.yaml") - 1
    )
    assert (
        "perl -pi -e 's{decks/}{simulation/decks/} if $. == 4' data/facility/simulation/models.yaml"
        in script
    )


def test_untracked_sources_in_a_repository_move_with_mv(tmp_path: Path) -> None:
    deployment = _repository(tmp_path / "deployment", commit=False)
    facility = _old_tree(deployment / "data/facility")

    script = _refusal(facility).script

    assert script[0] == f"cd {deployment.resolve()}"
    assert "mv data/facility/classes.yaml data/facility/vocabulary/classes.yaml" in script
    assert not [line for line in script if line.startswith("git ")]


def test_outside_a_repository_the_script_starts_at_the_tree_itself(tmp_path: Path) -> None:
    facility = _old_tree(tmp_path / "facility")

    script = _refusal(facility).script

    assert repository_root(facility) is None
    assert script[0] == f"cd {facility.resolve()}"
    assert "mv classes.yaml vocabulary/classes.yaml" in script


def test_without_git_every_move_is_a_plain_mv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment = tmp_path / "deployment"
    facility = _old_tree(deployment / "data/facility")
    _repository(deployment)
    monkeypatch.setenv("PATH", str(tmp_path / "no-binaries"))

    script = move_script(find_retired(facility))

    assert script[0] == f"cd {facility.resolve()}"
    assert "mv classes.yaml vocabulary/classes.yaml" in script


def test_paths_are_shell_quoted(tmp_path: Path) -> None:
    facility = _old_tree(tmp_path / "a facility")

    script = _refusal(facility).script

    assert script[0] == f"cd '{facility.resolve()}'"


def test_a_mounted_repository_gets_its_own_block(tmp_path: Path) -> None:
    machine = _old_tree(tmp_path / "machine-repo")
    _repository(machine)
    deployment = tmp_path / "deployment"
    content = deployment / "data/facility"
    _write(content, "benchmarks/cross_paradigm/queries/tree_queries.json", "[]\n")
    _repository(deployment)

    stop = _refusal(machine, content)

    lines = stop.message.splitlines()
    assert lines[0] == (
        f"# The facility tree at {machine.resolve()} (content at {content.resolve()})"
        " still holds sources at retired paths; nothing was read."
    )
    headers = [line for line in stop.script if line.startswith("cd ")]
    assert headers == [f"cd {machine.resolve()}", f"cd {deployment.resolve()}"]
    assert (
        "git mv data/facility/benchmarks/cross_paradigm/queries"
        " data/facility/benchmarks/channel_finder"
    ) in stop.script


def test_a_clash_is_a_comment_with_no_command(tmp_path: Path) -> None:
    _write(tmp_path, "classes.yaml")
    _write(tmp_path, "vocabulary/classes.yaml")
    root = tmp_path.resolve()

    stop = _refusal(tmp_path)

    assert stop.script == ()
    assert stop.message.splitlines()[1:] == [
        f"# {root / 'classes.yaml'} and {root / 'vocabulary/classes.yaml'} both exist:"
        " keep one, delete the other.",
        _PLACE_THESE,
    ]


def test_a_clash_beside_moves_prints_the_moves_but_no_command(tmp_path: Path) -> None:
    _old_tree(tmp_path)
    _write(tmp_path, "vocabulary/classes.yaml")

    lines = _refusal(tmp_path).message.splitlines()

    assert lines[1] == MOVE_HEADER
    assert "mv models.yaml simulation/models.yaml" in lines
    assert lines[-1] == _PLACE_THESE
    assert not [line for line in lines if line.startswith("osprey ")]


# --- running what it prints -------------------------------------------------------------


@pytest.mark.parametrize("style", sorted(MODELS), ids=lambda value: value)
def test_running_the_printed_script_leaves_a_tree_that_builds(
    tmp_path: Path, style: str, decks_read: list[Path]
) -> None:
    deployment = tmp_path / "deployment"
    facility = _old_tree(deployment / "data/facility", MODELS[style])
    _repository(deployment)

    calls = _run(_refusal(facility).message, tmp_path / "elsewhere")

    assert calls == [f"init --revalidate {deployment.resolve()}"]
    assert find_retired(facility).empty
    assert not (facility / "decks").exists()
    assert "deck: " in (facility / "simulation/models.yaml").read_text(encoding="utf-8")
    assert "simulation/decks/ring.json" in (facility / "simulation/models.yaml").read_text(
        encoding="utf-8"
    )
    build_facility(facility, project_name="demo", extra_checks=())
    assert decks_read == [facility / "simulation/decks/ring.json"]
    status = _git(deployment, "status", "--porcelain")
    assert "R  data/facility/classes.yaml -> data/facility/vocabulary/classes.yaml" in status


def test_running_the_printed_script_merges_into_an_existing_successor(
    tmp_path: Path, decks_read: list[Path]
) -> None:
    deployment = tmp_path / "deployment"
    facility = _old_tree(deployment / "data/facility")
    _write(facility, "decks/sub/extra.json", "{}\n")
    _write(facility, "simulation/decks/other.json", "{}\n")
    _repository(deployment)
    layout = find_retired(facility)
    assert [move.merge for move in layout.moves if move.old.startswith("decks")] == [True, True]

    _run(_refusal(facility).message, tmp_path / "elsewhere")

    assert find_retired(facility).empty
    assert not (facility / "decks").exists()
    assert (facility / "simulation/decks/sub/extra.json").is_file()
    assert (facility / "simulation/decks/other.json").is_file()
    build_facility(facility, project_name="demo", extra_checks=())
    assert decks_read == [facility / "simulation/decks/ring.json"]


def test_running_the_printed_script_untracked_outside_a_repository(
    tmp_path: Path, decks_read: list[Path]
) -> None:
    facility = _old_tree(tmp_path / "facility")
    _write(facility, "benchmarks/cross_paradigm/queries/tree_queries.json", "[]\n")

    _run(_refusal(facility).message, tmp_path / "elsewhere")

    assert find_retired(facility).empty
    assert not (facility / "benchmarks/cross_paradigm").exists()
    build_facility(facility, project_name="demo", extra_checks=())
    assert decks_read == [facility / "simulation/decks/ring.json"]


def test_running_the_printed_script_moves_a_mounted_second_repository(
    tmp_path: Path, decks_read: list[Path]
) -> None:
    machine = _old_tree(tmp_path / "machine-repo", MODELS["first-key"])
    _repository(machine)
    deployment = tmp_path / "deployment"
    content = deployment / "data/facility"
    _write(content, "benchmarks/cross_paradigm/queries/tree_queries.json", "[]\n")
    _repository(deployment)

    _run(_refusal(machine, content).message, tmp_path / "elsewhere")

    assert find_retired(machine, content).empty
    assert "R  classes.yaml -> vocabulary/classes.yaml" in _git(machine, "status", "--porcelain")
    assert "data/facility/benchmarks/channel_finder/tree_queries.json" in _git(
        deployment, "status", "--porcelain"
    )
    build_facility(machine, project_name="demo", content_dir=content, extra_checks=())
    assert decks_read == [machine / "simulation/decks/ring.json"]


# --- the script for any tree --------------------------------------------------------------


def test_the_rendered_script_moves_edits_and_deletes_in_one_block(tmp_path: Path) -> None:
    root = _repository(tmp_path / "repo", commit=False)
    _write(root, "old/a.txt")
    _write(root, "conf.yaml", "first\nkeep: old/a.txt\n")
    _write(root, "stale/x.txt")
    _write(root, "gone/deep/y.txt")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed")
    moves, _ = place_moves(root, [("old", "group/new")])
    edit = LineEdit(file=root / "conf.yaml", line=2, old="old/a.txt", new="group/new/a.txt")

    script = render_script(moves, [(root, edit)], [(root, "stale"), (root, "gone/deep")])

    assert script == [
        f"cd {root.resolve()}",
        "mkdir -p group",
        "git mv old group/new",
        "perl -pi -e 's{old/a\\.txt}{group/new/a.txt} if $. == 2' conf.yaml",
        "git rm -r -q -- stale || rm -r -- stale",
        "git rm -r -q -- gone/deep || rm -r -- gone/deep",
        "rmdir gone 2>/dev/null || true",
    ]
    _run("\n".join(script) + "\n", tmp_path / "elsewhere")
    assert (root / "group/new/a.txt").is_file()
    assert (root / "conf.yaml").read_text(encoding="utf-8") == "first\nkeep: group/new/a.txt\n"
    assert not (root / "stale").exists()
    assert not (root / "gone").exists()


def test_a_move_inside_a_nested_repository_runs_in_that_repository(tmp_path: Path) -> None:
    outer = _repository(tmp_path / "outer", commit=False)
    _write(outer, "keep.txt")
    nested = _repository(outer / "old/tree", commit=False)
    _write(nested, "a.txt")
    _write(nested, "untracked.txt")
    _git(nested, "add", "a.txt")
    _git(nested, "commit", "-q", "-m", "seed")
    _git(outer, "add", "keep.txt")
    _git(outer, "commit", "-q", "-m", "seed")
    moves = [
        Move(root=outer, old="old/tree", new="tree", is_dir=True, merge=False),
        Move(root=outer / "tree", old="a.txt", new="group/a.txt", is_dir=False, merge=False),
        Move(
            root=outer / "tree", old="untracked.txt", new="group/b.txt", is_dir=False, merge=False
        ),
    ]

    script = render_script(moves, (), ())

    assert script == [
        f"cd {outer.resolve()}",
        "mv old/tree tree",
        "mkdir -p tree/group",
        "git -C tree mv a.txt group/a.txt",
        "mv tree/untracked.txt tree/group/b.txt",
        "rmdir old 2>/dev/null || true",
    ]
    _run("\n".join(script) + "\n", tmp_path / "elsewhere")
    assert "R  a.txt -> group/a.txt" in _git(outer / "tree", "status", "--porcelain")
    assert (outer / "tree/group/b.txt").is_file()


def test_a_tracked_delete_git_refuses_is_removed_all_the_same(tmp_path: Path) -> None:
    root = _repository(tmp_path / "repo", commit=False)
    _write(root, "stale/x.txt", "as committed\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed")
    _write(root, "stale/x.txt", "changed since\n")

    script = render_script((), (), [(root, "stale")])
    _run("\n".join(script) + "\n", tmp_path / "elsewhere")

    assert not (root / "stale").exists()


def test_a_rendered_line_delete_removes_a_line_whatever_its_ending(tmp_path: Path) -> None:
    file = tmp_path / "f.txt"
    file.write_bytes(b"keep\r\ngone: 1\r\nlast\n")
    deletes = [
        (tmp_path, LineDelete(file=file, line=2, text="gone: 1")),
        (tmp_path, LineDelete(file=file, line=3, text="not this")),
    ]

    _run("\n".join(render_script((), (), (), deletes)) + "\n", tmp_path / "elsewhere")

    assert file.read_bytes() == b"keep\r\nlast\n"


def test_a_rendered_line_edit_takes_its_text_literally(tmp_path: Path) -> None:
    _write(tmp_path, "f.txt", "a.b x $HOME @x {y}\naxb\n")
    edit = LineEdit(file=tmp_path / "f.txt", line=1, old="a.b x $HOME @x {y}", new="$v @w {z}\\")

    _run("\n".join(render_script((), [(tmp_path, edit)], ())) + "\n", tmp_path / "elsewhere")

    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "$v @w {z}\\\naxb\n"


def test_untracked_deletes_outside_a_repository_use_rm(tmp_path: Path) -> None:
    _write(tmp_path, "tree/stale.txt")

    script = render_script((), (), [(tmp_path / "tree", "stale.txt")])

    assert script == [f"cd {(tmp_path / 'tree').resolve()}", "rm -r -- stale.txt"]


def test_nothing_to_render_is_an_empty_script() -> None:
    assert render_script((), (), ()) == []
