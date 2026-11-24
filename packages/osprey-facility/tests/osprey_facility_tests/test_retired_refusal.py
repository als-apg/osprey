"""The refusal of a facility tree still laid out at retired paths.

A read of such a tree stops before anything is read, and what it prints is a
shell script: run from any directory, it moves every source to its current
path, so the tree holds nothing retired and builds. Old trees are laid out
from literal paths here, never from the declaration.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from osprey_facility.build import build_facility
from osprey_facility.errors import FacilityError, RetiredLayoutRefused
from osprey_facility.layers.list import import_list
from osprey_facility.retired import find_retired, move_script, refuse_retired, repository_root
from osprey_facility.sources import load_sources
from osprey_facility.validate import run_stages
from osprey_facility_tests._harness import REPO_ROOT

DECK = REPO_ROOT / "src/osprey/templates/facilities/example/simulation/decks/LTB.json"

#: One model's entry, its ``deck`` value written in each style an author may use.
MODELS = {
    "block": "# the ring\n- name: ring\n  engine: pyat\n  deck: decks/ring.json\n",
    "first-key": "- deck: decks/ring.json\n  name: ring\n  engine: pyat\n",
    "quoted": '- name: ring\n  engine: pyat\n  deck: "decks/ring.json"\n',
}


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.invalid")


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


def _run(message: str, elsewhere: Path) -> None:
    elsewhere.mkdir(exist_ok=True)
    done = subprocess.run(
        ["sh", "-e"], input=message, cwd=elsewhere, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


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
        "# Move them to their current paths, then run the command again:",
        f"cd {root}",
        "mkdir -p vocabulary simulation",
        "mv classes.yaml vocabulary/classes.yaml",
        "mv models.yaml simulation/models.yaml",
        "mv decks simulation/decks",
        "perl -pi -e 's{decks/}{simulation/decks/} if $. == 4' simulation/models.yaml",
        "# or run: osprey init --revalidate",
    ]


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
    assert "mkdir -p data/facility/vocabulary data/facility/simulation" in script
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
    assert stop.message.splitlines()[-2:] == [
        f"# {root / 'classes.yaml'} and {root / 'vocabulary/classes.yaml'} both exist:"
        " keep one, delete the other.",
        "# or run: osprey init --revalidate",
    ]


# --- running what it prints -------------------------------------------------------------


@pytest.mark.parametrize("style", sorted(MODELS), ids=lambda value: value)
def test_running_the_printed_script_leaves_a_tree_that_builds(tmp_path: Path, style: str) -> None:
    deployment = tmp_path / "deployment"
    facility = _old_tree(deployment / "data/facility", MODELS[style])
    _repository(deployment)

    _run(_refusal(facility).message, tmp_path / "elsewhere")

    assert find_retired(facility).empty
    assert not (facility / "decks").exists()
    assert "deck: " in (facility / "simulation/models.yaml").read_text(encoding="utf-8")
    assert "simulation/decks/ring.json" in (facility / "simulation/models.yaml").read_text(
        encoding="utf-8"
    )
    build_facility(facility, project_name="demo", extra_checks=())
    status = _git(deployment, "status", "--porcelain")
    assert "R  data/facility/classes.yaml -> data/facility/vocabulary/classes.yaml" in status


def test_running_the_printed_script_merges_into_an_existing_successor(tmp_path: Path) -> None:
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


def test_running_the_printed_script_untracked_outside_a_repository(tmp_path: Path) -> None:
    facility = _old_tree(tmp_path / "facility")
    _write(facility, "benchmarks/cross_paradigm/queries/tree_queries.json", "[]\n")

    _run(_refusal(facility).message, tmp_path / "elsewhere")

    assert find_retired(facility).empty
    assert not (facility / "benchmarks/cross_paradigm").exists()
    build_facility(facility, project_name="demo", extra_checks=())


def test_running_the_printed_script_moves_a_mounted_second_repository(tmp_path: Path) -> None:
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
