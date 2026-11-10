"""A machine repository builds on its own and mounted, to the same facility.

A machine repository versions one facility (``data/facility``) with the VA-only
profile it builds and serves by itself. A deployment mounts it, as a nested
clone or as a git submodule, and names its facility directory with
``machine: <mount>/data/facility``.

The shapes the docs describe hold on every PR:

- ``osprey validate`` passes the repository's own ``extends: hello-world``
  profile, and a fresh ``git clone`` of the repository builds;
- that standalone build and builds mounted at two paths write byte-equal
  facility files and decks, and every render starts the scenarios the machine
  marks ``start: true``;
- the repository's own top (``.git``, ``profile.yml``, ``build/``, ``var/``,
  ``.env``) reaches neither a render nor the staged image sources, and editing
  it moves neither the facility digest nor the profile hash, while a limit edit
  moves both;
- ``osprey facility validate`` stops on a deployment file restating a machine
  record and on a mount that holds no machine (an uninitialised submodule), and
  a standalone profile that states a start set beside the marks is refused.

Every repository here is ``git init``-ed, so its root holds ``.git`` the way a
real one does.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from osprey.cli.validate_cmd import validate
from osprey.facility.render import FACILITY_FILE, facility_digest
from osprey.utils.workspace import IMAGE_DIR_NAME, PROFILE_FILENAME
from osprey_connectors.workspace import BUILD_DIR_NAME
from tests.cli._lifecycle_build import invoke_build, profile_hash
from tests.fixtures.machine_repository import (
    FACILITY_PATH,
    MARKED_SCENARIO,
    REPOSITORY_FILES,
    add_repository_files,
    commit_all,
    git,
    git_init,
    make_machine_repository,
)

pytestmark = [pytest.mark.slow]

#: A mount as a nested clone, the machine repository's directory at the deployment top.
CLONE_MOUNT = "machine"

#: A mount as a git submodule, two levels down; its ``.git`` is a file.
SUBMODULE_MOUNT = "vendor/example-machine"

#: The start-set key, as a profile states it under ``config:``.
DEFAULT_SCENARIOS_KEY = "simulation.default_scenarios"

#: The secret the machine repository's working copy holds in its ``.env``.
SECRET = REPOSITORY_FILES[".env"].strip().split("=", 1)[1]

#: Entries of the machine repository's top that are never machine facts.
REPOSITORY_TOP_ENTRIES = {".git", PROFILE_FILENAME, "build", "var", ".env"}


# --- repositories ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def machine_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The committed machine repository, with ``build/``, ``var/`` and ``.env`` at its top."""
    return make_machine_repository(tmp_path_factory.mktemp("upstream") / "example-machine")


def _clone(source: Path, destination: Path) -> Path:
    git("clone", "-q", str(source), str(destination), cwd=destination.parent)
    return destination


def _deployment(parent: Path, name: str, source: Path, mount: str, *, submodule: bool) -> Path:
    """A committed deployment mounting *source* at *mount*.

    Its own ``data/facility`` is empty: a mounted deployment's facility directory
    holds only its content directories, and this one has none (a ``.gitkeep``
    keeps the directory in a clone, and dot-entries are not content). A nested clone
    keeps the working copy's ``.git``, ``build/``, ``var/`` and ``.env``; a
    submodule gets them written into its checkout after ``git submodule add``.
    """
    repo = parent / name
    (repo / "data" / "facility").mkdir(parents=True)
    (repo / "data" / "facility" / ".gitkeep").write_text("", encoding="utf-8")
    (repo / PROFILE_FILENAME).write_text(
        "extends: hello-world\n"
        "name: Mounted Deployment\n"
        f"project_name: {name}\n"
        "data: data\n"
        f"machine: {mount}/{FACILITY_PATH}\n",
        encoding="utf-8",
    )
    git_init(repo)
    if submodule:
        commit_all(repo, "the deployment")
        git("submodule", "add", "-q", str(source), mount, cwd=repo)
        commit_all(repo, "mount the machine")
        add_repository_files(repo / mount)
    else:
        shutil.copytree(source, repo / mount, symlinks=True)
        commit_all(repo, "the deployment")
    return repo


def _edit_config(repo: Path, key: str, value: Any) -> None:
    profile = repo / PROFILE_FILENAME
    data = yaml.safe_load(profile.read_text(encoding="utf-8"))
    data.setdefault("config", {})[key] = value
    profile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _facility_validate(repo: Path) -> Result:
    return CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])


# --- the three builds -----------------------------------------------------------------


@pytest.fixture(scope="module")
def built(machine_source: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A standalone build from a fresh clone, and builds mounted at two paths."""
    parent = tmp_path_factory.mktemp("builds")
    repos = {
        "standalone": _clone(machine_source, parent / "example-machine"),
        CLONE_MOUNT: _deployment(
            parent, "nested-clone", machine_source, CLONE_MOUNT, submodule=False
        ),
        SUBMODULE_MOUNT: _deployment(
            parent, "submodule", machine_source, SUBMODULE_MOUNT, submodule=True
        ),
    }
    for repo in repos.values():
        result = invoke_build(repo)
        assert result.exit_code == 0, result.output
    return repos


def _render_roots(repo: Path) -> list[Path]:
    """The deployment's render, each render's and each container image's."""
    build_dir = repo / BUILD_DIR_NAME
    tiers = sorted(path.parent for path in build_dir.glob("*-*/config.yml"))
    images = sorted(
        path.parent for path in (build_dir / IMAGE_DIR_NAME).glob(f"*/{BUILD_DIR_NAME}/config.yml")
    )
    return [build_dir, *tiers, *images]


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _start_set(render: Path) -> Any:
    config = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))
    if DEFAULT_SCENARIOS_KEY in config:
        return config[DEFAULT_SCENARIOS_KEY]
    return config["simulation"]["default_scenarios"]


def test_validate_passes_the_machine_repositorys_own_profile(
    machine_source: Path, tmp_path: Path
) -> None:
    clone = _clone(machine_source, tmp_path / "example-machine")

    result = CliRunner().invoke(validate, ["--repo", str(clone)])

    assert result.exit_code == 0, result.output


def test_the_standalone_clone_is_a_single_tree_with_no_working_copy_files(
    built: dict[str, Path], machine_source: Path
) -> None:
    clone = built["standalone"]

    assert (clone / ".git").is_dir()
    assert not (clone / ".env").exists()
    assert (machine_source / ".env").is_file()
    assert "machine:" not in (clone / PROFILE_FILENAME).read_text(encoding="utf-8")


def test_the_facility_file_is_byte_equal_standalone_and_mounted(built: dict[str, Path]) -> None:
    standalone = (built["standalone"] / BUILD_DIR_NAME / FACILITY_FILE).read_bytes()

    roots = {
        f"{name}:{root.relative_to(repo)}": (root / FACILITY_FILE).read_bytes() == standalone
        for name, repo in built.items()
        for root in _render_roots(repo)
    }

    assert len(roots) >= 2 * len(built)
    assert roots == dict.fromkeys(roots, True)
    for repo in built.values():
        assert str(repo).encode() not in standalone
        assert str(repo.resolve()).encode() not in standalone


def test_the_decks_are_byte_equal_standalone_and_mounted(
    built: dict[str, Path], machine_source: Path
) -> None:
    source = _files(machine_source / FACILITY_PATH / "decks")
    standalone = built["standalone"] / BUILD_DIR_NAME / "data"
    simulator = _files(standalone / "simulator" / "decks")

    assert source and simulator
    assert _files(standalone / "facility" / "decks") == source
    for name, repo in built.items():
        for root in _render_roots(repo):
            assert _files(root / "data" / "facility" / "decks") == source, (name, root)
            assert _files(root / "data" / "simulator" / "decks") == simulator, (name, root)


def test_every_render_starts_the_marked_scenario(built: dict[str, Path]) -> None:
    starts = {
        f"{name}:{root.relative_to(repo)}": _start_set(root)
        for name, repo in built.items()
        for root in _render_roots(repo)
    }

    assert starts == dict.fromkeys(starts, [MARKED_SCENARIO])


@pytest.mark.parametrize("mount", [CLONE_MOUNT, SUBMODULE_MOUNT])
def test_the_repository_top_reaches_neither_the_render_nor_the_staged_sources(
    built: dict[str, Path], machine_source: Path, mount: str
) -> None:
    repo = built[mount]
    top = repo / mount
    build_dir = repo / BUILD_DIR_NAME
    machine_facts = {
        rel: data
        for rel, data in _files(machine_source / FACILITY_PATH).items()
        if not rel.startswith(".git")
    }

    # The working copy at the mount carries every repository file.
    assert (top / ".git").exists()
    assert {entry for entry in REPOSITORY_TOP_ENTRIES if (top / entry).exists()} == (
        REPOSITORY_TOP_ENTRIES
    )
    assert (top / ".git").is_dir() is (mount == CLONE_MOUNT)

    # The render is the machine's facts beside the deployment's own files.
    rendered = machine_facts | _files(repo / "data" / "facility")
    for root in _render_roots(repo):
        assert _files(root / "data" / "facility") == rendered, root

    staged = build_dir / IMAGE_DIR_NAME / repo.name
    staged_mount = staged / mount
    assert sorted(path.name for path in staged_mount.iterdir()) == ["data"]
    assert sorted(path.name for path in (staged_mount / "data").iterdir()) == ["facility"]
    assert _files(staged_mount / "data" / "facility") == machine_facts
    assert not [path for path in build_dir.rglob(".git") if path != build_dir / ".git"]
    assert not [rel for rel, data in _files(build_dir).items() if SECRET.encode() in data]


# --- the digest and the profile hash, by direct calls -------------------------------


def _fingerprints(repo: Path, mount: str) -> tuple[str, str]:
    machine = repo / mount / FACILITY_PATH
    return (
        facility_digest(machine, content_dir=repo / "data" / "facility"),
        profile_hash(repo),
    )


@pytest.fixture(params=[CLONE_MOUNT, SUBMODULE_MOUNT])
def mounted(
    request: pytest.FixtureRequest, machine_source: Path, tmp_path: Path
) -> tuple[Path, str]:
    """A fresh, unbuilt deployment mounting the machine repository."""
    mount = request.param
    repo = _deployment(
        tmp_path, "deployment", machine_source, mount, submodule=mount == SUBMODULE_MOUNT
    )
    return repo, mount


def test_a_limit_edit_moves_the_digest_and_the_profile_hash(mounted: tuple[Path, str]) -> None:
    repo, mount = mounted
    before = _fingerprints(repo, mount)
    limits = repo / mount / FACILITY_PATH / "limits.yaml"
    text = limits.read_text(encoding="utf-8")
    assert "max_value: 12.0" in text
    limits.write_text(text.replace("max_value: 12.0", "max_value: 11.0", 1), encoding="utf-8")

    after = _fingerprints(repo, mount)

    assert after[0] != before[0]
    assert after[1] != before[1]


def _edit_git(top: Path) -> None:
    dot_git = top / ".git"
    if dot_git.is_dir():
        (dot_git / "info").mkdir(exist_ok=True)
        with (dot_git / "info" / "exclude").open("a", encoding="utf-8") as handle:
            handle.write("ignored-by-this-clone\n")
    else:
        dot_git.write_text(dot_git.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def _edit_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("edited\n", encoding="utf-8")


#: Edits at the mount's repository top; none of them is a machine fact.
REPOSITORY_TOP_EDITS = {
    ".git": _edit_git,
    "build": lambda top: _edit_file(top / "build" / "out.txt"),
    "build-new": lambda top: _edit_file(top / "build" / "data" / "facility" / "limits.yaml"),
    ".env": lambda top: _edit_file(top / ".env"),
    "var": lambda top: _edit_file(top / "var" / "state.txt"),
    "profile": lambda top: _edit_file(top / PROFILE_FILENAME),
}


@pytest.mark.parametrize("edit", sorted(REPOSITORY_TOP_EDITS))
def test_an_edit_at_the_repository_top_moves_neither(mounted: tuple[Path, str], edit: str) -> None:
    repo, mount = mounted
    before = _fingerprints(repo, mount)

    REPOSITORY_TOP_EDITS[edit](repo / mount)

    assert _fingerprints(repo, mount) == before


# --- stops ----------------------------------------------------------------------------


def test_a_mounted_deployment_validates(mounted: tuple[Path, str]) -> None:
    repo, _mount = mounted

    result = _facility_validate(repo)

    assert result.exit_code == 0, result.output


def test_a_restated_record_stops_validate(mounted: tuple[Path, str]) -> None:
    repo, mount = mounted
    record = Path("records") / "channels.yaml"
    restated = repo / "data" / "facility" / record
    restated.parent.mkdir(parents=True)
    shutil.copy2(repo / mount / FACILITY_PATH / record, restated)

    result = _facility_validate(repo)

    assert result.exit_code == 1, result.output
    assert result.stderr.startswith(
        f"facility: machine-restated: path data/facility/{record.as_posix()} — "
    )
    assert f"{mount}/{FACILITY_PATH}/{record.as_posix()}" in result.stderr


def test_an_uninitialised_submodule_stops_validate(machine_source: Path, tmp_path: Path) -> None:
    upstream = _deployment(tmp_path, "upstream", machine_source, SUBMODULE_MOUNT, submodule=True)
    # A clone without --recurse-submodules leaves the mount an empty directory.
    clone = _clone(upstream, tmp_path / "deployment")
    assert (clone / SUBMODULE_MOUNT).is_dir()
    assert not any((clone / SUBMODULE_MOUNT).iterdir())

    result = _facility_validate(clone)

    assert result.exit_code == 1, result.output
    assert result.stderr.startswith(
        f"facility: machine-missing: path {SUBMODULE_MOUNT}/{FACILITY_PATH}/identity.yaml — "
    )
    assert "git submodule update --init" in result.stderr


def test_a_stated_start_set_in_the_standalone_profile_is_refused(
    machine_source: Path, tmp_path: Path
) -> None:
    clone = _clone(machine_source, tmp_path / "example-machine")
    _edit_config(clone, DEFAULT_SCENARIOS_KEY, ["nominal"])

    result = _facility_validate(clone)

    assert result.exit_code == 1, result.output
    last = result.stderr.splitlines()[-1]
    assert last.startswith(f"facility: profile-invalid: path {DEFAULT_SCENARIOS_KEY} — ")
    assert MARKED_SCENARIO in last
    assert "start: true" in last
