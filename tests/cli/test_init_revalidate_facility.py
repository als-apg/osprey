"""``osprey init --revalidate`` moves a facility tree off its retired paths.

Each repo here is a fresh ``hello-world`` deployment whose facility tree is put
back into the retired layout: root files at the top, ``scenarios/`` and
``decks/`` beside them, the benchmark queries under ``cross_paradigm/``, and
authored ``deck:`` values naming ``decks/``. The rewrite moves what lies in the
deployment's own repository and prints the commands for what does not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from osprey_facility.retired import find_retired
from tests._builds import init_project

#: The authored models file in the retired layout: a header, block-style,
#: first-key and quoted flow-style ``deck:`` values, and comments on the lines
#: the rewrite edits.
_OLD_MODELS = (
    "# Machine models the simulator runs, one per lattice deck.\n"
    "# Each deck path is relative to the facility root.\n"
    "\n"
    "- name: SR   # the storage ring\n"
    "  engine: pyat\n"
    "  deck: decks/sr.json   # exported from the lattice\n"
    "- deck: decks/booster.json\n"
    "  name: BR\n"
    "  engine: pyat\n"
    '- {name: BTS, engine: pyat, deck: "decks/bts.json"}  # transfer line\n'
)

_NEW_MODELS = (
    "# Machine models the simulator runs, one per lattice deck.\n"
    "# Each deck path is relative to the facility root.\n"
    "\n"
    "- name: SR   # the storage ring\n"
    "  engine: pyat\n"
    "  deck: simulation/decks/sr.json   # exported from the lattice\n"
    "- deck: simulation/decks/booster.json\n"
    "  name: BR\n"
    "  engine: pyat\n"
    '- {name: BTS, engine: pyat, deck: "simulation/decks/bts.json"}  # transfer line\n'
)

#: Machine facts in the retired layout, relative to the machine root.
_OLD_MACHINE = {
    "classes.yaml": "# The device classes.\nBPM: {}\n",
    "models.yaml": _OLD_MODELS,
    "decks/sr.json": "{}\n",
    "decks/booster.json": "{}\n",
    "decks/bts.json": "{}\n",
    "scenarios/injection.yaml": "{}\n",
    "fixes.yaml": "[]\n",
}

#: The same facts where they live now.
_NEW_MACHINE = {
    "vocabulary/classes.yaml": "# The device classes.\nBPM: {}\n",
    "simulation/models.yaml": _NEW_MODELS,
    "simulation/decks/sr.json": "{}\n",
    "simulation/decks/booster.json": "{}\n",
    "simulation/decks/bts.json": "{}\n",
    "simulation/scenarios/injection.yaml": "{}\n",
    "provenance/fixes.yaml": "[]\n",
}

#: Deployment content in the retired layout, relative to the content root.
_OLD_CONTENT = {"benchmarks/cross_paradigm/queries/tree.json": "[]\n"}
_NEW_CONTENT = {"benchmarks/channel_finder/tree.json": "[]\n"}

_FACILITY = "data/facility"


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's git config (signing, templates) out of the rewrite's `git mv`."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "OSPREY")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "osprey@example.org")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "protocol.file.allow=always", *args],
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


def _retire(facility: Path) -> None:
    """Put the bundled hello-world facility tree back into the retired layout."""
    (facility / "simulation" / "seeds.yaml").rename(facility / "seeds.yaml")
    (facility / "simulation").rmdir()


def _plant(tmp_path: Path, *, git: bool) -> Path:
    """A hello-world deployment whose facility tree is in the retired layout."""
    root = init_project(tmp_path, "hello-world", "plant")
    facility = root / _FACILITY
    _retire(facility)
    _write(facility, {**_OLD_MACHINE, **_OLD_CONTENT})
    if git:
        _git(root, "init", "-q")
        _commit_all(root, "retired layout")
    return root


def _files(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def _revalidate(root: Path) -> object:
    return CliRunner().invoke(init, [str(root), "--revalidate"], catch_exceptions=False)


def _assert_current(facility: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        assert (facility / rel).read_text(encoding="utf-8") == text, rel
    assert find_retired(facility).empty


def test_tracked_moves_keep_their_history(tmp_path: Path) -> None:
    root = _plant(tmp_path, git=True)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    _assert_current(root / _FACILITY, {**_NEW_MACHINE, **_NEW_CONTENT})
    staged = _git(root, "diff", "--cached", "--name-status", "-M")
    for old, new in (
        ("classes.yaml", "vocabulary/classes.yaml"),
        ("seeds.yaml", "simulation/seeds.yaml"),
        ("decks/sr.json", "simulation/decks/sr.json"),
        ("benchmarks/cross_paradigm/queries/tree.json", "benchmarks/channel_finder/tree.json"),
    ):
        assert f"\t{_FACILITY}/{old}\t{_FACILITY}/{new}\n" in staged, staged
    _git(root, "commit", "-q", "-am", "current layout")
    log = _git(root, "log", "--follow", "--oneline", "--", f"{_FACILITY}/vocabulary/classes.yaml")
    assert len(log.splitlines()) == 2, log


def test_the_diff_reports_each_move(tmp_path: Path) -> None:
    root = _plant(tmp_path, git=True)

    output = _revalidate(root).output

    assert f"rename {_FACILITY}/classes.yaml => {_FACILITY}/vocabulary/classes.yaml" in output
    assert (
        f"rename {_FACILITY}/scenarios/injection.yaml => "
        f"{_FACILITY}/simulation/scenarios/injection.yaml"
    ) in output
    assert f"--- a/{_FACILITY}/models.yaml" in output
    assert f"+++ b/{_FACILITY}/simulation/models.yaml" in output
    assert "+  deck: simulation/decks/sr.json   # exported from the lattice" in output
    assert "Left for you to do by hand" not in output


def test_untracked_tree_is_renamed(tmp_path: Path) -> None:
    root = _plant(tmp_path, git=False)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    facility = root / _FACILITY
    _assert_current(facility, {**_NEW_MACHINE, **_NEW_CONTENT})
    for gone in ("classes.yaml", "models.yaml", "decks", "scenarios", "benchmarks/cross_paradigm"):
        assert not (facility / gone).exists(), gone


def test_moves_merge_into_an_existing_successor(tmp_path: Path) -> None:
    root = _plant(tmp_path, git=True)
    facility = root / _FACILITY
    _write(facility, {"simulation/scenarios/top-up.yaml": "{}\n"})
    _commit_all(root, "a current scenario")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (facility / "simulation" / "scenarios").iterdir()) == [
        "injection.yaml",
        "top-up.yaml",
    ]
    assert not (facility / "scenarios").exists()
    assert find_retired(facility).empty


def test_a_clash_refuses_before_anything_changes(tmp_path: Path) -> None:
    root = _plant(tmp_path, git=True)
    _write(root / _FACILITY, {"vocabulary/classes.yaml": "# Kept here too.\n"})
    before = _files(root)

    result = CliRunner().invoke(init, [str(root), "--revalidate"])

    assert result.exit_code != 0
    assert f"{_FACILITY}/classes.yaml" in result.output
    assert f"{_FACILITY}/vocabulary/classes.yaml" in result.output
    assert _files(root) == before


def test_deck_values_are_spliced_on_their_lines(tmp_path: Path) -> None:
    root = _plant(tmp_path, git=False)

    _revalidate(root)

    text = (root / _FACILITY / "simulation" / "models.yaml").read_text(encoding="utf-8")
    assert text == _NEW_MODELS


def test_a_current_tree_has_nothing_to_do(tmp_path: Path) -> None:
    root = init_project(tmp_path, "hello-world", "plant")
    before = _files(root)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert "nothing to rewrite" in result.output
    assert _files(root) == before


def test_a_second_run_is_a_no_op(tmp_path: Path) -> None:
    root = _plant(tmp_path, git=True)
    _revalidate(root)
    before = _files(root)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert "nothing to rewrite" in result.output
    assert _files(root) == before


def _mount(root: Path, machine: str, *, profile: str = "profile.yml") -> None:
    """Point the deployment at a machine root: ``machine:`` in *profile*."""
    path = root / profile
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.write_text(f"{text.rstrip()}\nmachine: {machine}\n".lstrip(), encoding="utf-8")


def _machine_repo(path: Path) -> Path:
    """A machine repository in the retired layout, committed."""
    _write(path, {"identity.yaml": "name: Plant\n", **_OLD_MACHINE})
    _git(path, "init", "-q")
    _commit_all(path, "machine facts")
    return path


def test_a_mounted_machine_is_left_for_the_operator(tmp_path: Path) -> None:
    machine = _machine_repo(tmp_path / "machine")
    root = init_project(tmp_path, "hello-world", "plant")
    shutil.rmtree(root / _FACILITY)
    _write(root / _FACILITY, _OLD_CONTENT)
    _mount(root, "../machine")
    _git(root, "init", "-q")
    _commit_all(root, "mounted")
    machine_before = _files(machine)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert _files(machine) == machine_before
    _assert_current(root / _FACILITY, _NEW_CONTENT)
    output = result.output
    assert "Left for you to do by hand" in output
    assert f"cd {machine.resolve()}" in output
    assert "git mv classes.yaml vocabulary/classes.yaml" in output
    assert "simulation/decks/" in output


def test_a_submodule_machine_root_is_left_for_the_operator(tmp_path: Path) -> None:
    source = _machine_repo(tmp_path / "machine-source")
    root = init_project(tmp_path, "hello-world", "plant")
    shutil.rmtree(root / _FACILITY)
    _git(root, "init", "-q")
    _git(root, "submodule", "add", "-q", str(source), "machine")
    _mount(root, "machine")
    _commit_all(root, "submodule machine")
    machine = root / "machine"
    machine_before = _files(machine)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert _files(machine) == machine_before
    assert "nothing to rewrite here" in result.output
    assert "Left for you to do by hand" in result.output
    assert f"cd {machine.resolve()}" in result.output


def test_a_machine_root_set_in_a_variant_is_found(tmp_path: Path) -> None:
    root = init_project(tmp_path, "hello-world", "plant")
    _write(root / "machine", {"identity.yaml": "name: Plant\n", **_OLD_MACHINE})
    _mount(root, "machine", profile="profiles/lab.yml")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    _assert_current(root / "machine", _NEW_MACHINE)
    assert "rename machine/classes.yaml => machine/vocabulary/classes.yaml" in result.output
