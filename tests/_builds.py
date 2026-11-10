"""Real builds shared across test directories, used by ``tests/conftest.py``.

``built_demo`` is the one real ``osprey build --skip-deps`` of the
demo preset that the facility and build tests read. It is
session-scoped, and under xdist the workers of one run share it: the first
worker to ask builds it under the run's base temp directory, holding a lock, and
every later worker reads that build, so the build runs once per run whichever
groups its readers sit in. ``built_demo_watchdog_on`` is shared
the same way: a copy of that build with the preset's watchdog turned on in its
profile and the repo rebuilt in place.

``offline_build_env`` is for the tests that run a real ``osprey build`` with its
install: the environment additions under which that install reads uv's cache
and needs no index. It shows once per run that the cache holds what a build
installs, filling it if it does not, and each test passes the additions to its
own build call.

:class:`Restorable` is for the tests that start from a build and then edit or
rebuild it: the tree is made once and put back pristine, at the path it was
made at, for each test.
"""

from __future__ import annotations

import fcntl
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from click.testing import CliRunner, Result

from tests._scoped_temp import shared_dir


def init_project(parent: Path, preset: str, name: str, *sets: str) -> Path:
    """Run ``osprey init`` for a preset into ``parent/name``.

    Args:
        parent: The directory the repo is created in.
        preset: The preset name.
        name: The repo's directory name, which is the project name.
        sets: ``--set`` assignments, each ``KEY=VALUE``.

    Returns:
        The repo.
    """
    from osprey.cli.init_cmd import init

    repo = parent / name
    edits = [arg for assignment in sets for arg in ("--set", assignment)]
    result = CliRunner().invoke(init, [str(repo), "--preset", preset, "--no-git", *edits])
    assert result.exit_code == 0, result.output
    return repo


def init_machineless(parent: Path, name: str) -> Path:
    """Init a demo repo with no machine in it.

    Its facility tree holds the blank knowledge index the facility import seeds
    and nothing else, and its profile states the empty start set, the remedy of
    the stop the preset's start set meets once the file holds no scenario of it.

    Args:
        parent: The directory the repo is created in.
        name: The repo's directory name, which is the project name.

    Returns:
        The repo.
    """
    repo = init_project(parent, "demo", name)
    strip_machine(repo)
    return repo


def strip_machine(repo: Path) -> None:
    """Take the machine out of a demo repo.

    The facility tree is left holding the blank knowledge index the facility
    import seeds, and the profile states the empty start set.

    Args:
        repo: The repo.
    """
    from osprey.cli.main import cli
    from osprey.services.facility_knowledge.okf.index import _settle_bundle

    facility = repo / "data" / "facility"
    shutil.rmtree(facility)
    _settle_bundle(facility / "knowledge")
    stated = CliRunner().invoke(
        cli, ["set", "--repo", str(repo), "config.simulation.default_scenarios=[]"]
    )
    assert stated.exit_code == 0, stated.output


def run_build(repo: Path) -> Result:
    """Run ``osprey build --skip-deps`` on a repo.

    Args:
        repo: The repo.

    Returns:
        The CliRunner result.
    """
    from osprey.cli.build_cmd import build

    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps"])


class RenderOutputs(NamedTuple):
    """What ``render_facility_outputs`` wrote in one render.

    Attributes:
        render_dir: The render root it was handed. A render is staged before it
            moves into ``build/``, so this is the staging path.
        files: Each file it wrote, by its path relative to ``render_dir``, to
            the bytes it wrote.
    """

    render_dir: Path
    files: dict[str, bytes]


@dataclass
class BuiltProject:
    """A built repo and what ``render_facility_outputs`` wrote in each render.

    Attributes:
        repo: The repo.
        outputs: One entry per render, in render order. The bytes are taken as
            they are written, before the render moves into ``build/``.
        output: What the build printed, stdout and stderr together.
    """

    repo: Path
    outputs: list[RenderOutputs] = field(default_factory=list)
    output: str = ""

    @property
    def build_dir(self) -> Path:
        """The deployment's own render root."""
        from osprey_connectors.workspace import BUILD_DIR_NAME

        return self.repo / BUILD_DIR_NAME

    @property
    def facility_dir(self) -> Path:
        """The repo's ``data/facility`` directory."""
        return self.repo / "data" / "facility"

    @cached_property
    def facility_raw(self) -> bytes:
        """The bytes of ``build/facility.json``."""
        from osprey.facility.render import FACILITY_FILE

        return (self.build_dir / FACILITY_FILE).read_bytes()

    @cached_property
    def facility(self) -> dict[str, Any]:
        """``build/facility.json``, parsed."""
        document: dict[str, Any] = json.loads(self.facility_raw)
        return document


#: Under xdist, the file in the run's shared base temp directory that names the
#: built repo and holds its render outputs and what it printed.
SHARED_BUILD_RECORD = "built-ca.pickle"

#: Under xdist, the same for the build with the preset's watchdog on.
SHARED_WATCHDOG_BUILD_RECORD = "built-ca-watchdog-on.pickle"

#: Under xdist, the same for the build with orbit bump turned on.
SHARED_ORBIT_BUMP_BUILD_RECORD = "built-ca-orbit-bump.pickle"


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``lock_path`` for the block."""
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def build_demo(parent: Path, *sets: str) -> BuiltProject:
    """Initialise and build the demo preset under ``parent``.

    Args:
        parent: The directory the repo is created in.
        sets: ``--set`` assignments for the init, each ``KEY=VALUE``.

    Returns:
        The built repo and what each render's facility outputs were.
    """
    from osprey.facility import render

    repo = init_project(parent, "demo", "demo", *sets)
    built = BuiltProject(repo)
    real = render.render_facility_outputs

    def spy(
        render_dir: Path, doc: Any, rendered_config: Any, facility_dir: Path, **kwargs: Any
    ) -> list[Path]:
        written = real(render_dir, doc, rendered_config, facility_dir, **kwargs)
        built.outputs.append(
            RenderOutputs(
                render_dir,
                {path.relative_to(render_dir).as_posix(): path.read_bytes() for path in written},
            )
        )
        return written

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(render, "render_facility_outputs", spy)
        result = run_build(repo)
    assert result.exit_code == 0, result.output
    built.output = result.output
    return built


def _remove_tree(root: Path) -> None:
    """Remove *root*, write-protected entries included; a missing *root* is fine."""

    def writable_then_retry(function: Callable[[str], object], path: str, *_: object) -> None:
        os.chmod(path, 0o700)
        parent = os.path.dirname(path)
        os.chmod(parent, os.stat(parent).st_mode | 0o700)
        function(path)

    if root.exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(root, onexc=writable_then_retry)
        else:
            shutil.rmtree(root, onerror=writable_then_retry)


class Restorable:
    """A tree made once, then put back pristine at the path it was made at, per test.

    For a test that starts from a real ``osprey build`` and then edits or
    rebuilds the result: :meth:`restore` hands it the tree as the one build
    left it, without running that build again. The first call makes the tree
    and keeps a copy of it aside; every later call wipes the tree and copies
    that one back.

    The same path, never a moved copy, because a build at another path is not
    the moved bytes: a render records its repo's absolute path, and derives
    more from it than the path itself (the manifests' digests of the files
    that name it, the compose ``com.osprey.repo-id`` label). So every user
    shares the one path, and the trees they are handed are what a build
    there makes. That is also why it is for function-scoped users only: two
    holders alive at once would be handed the one directory.

    The first call runs *make* inside the first user's test, with that test's
    fixtures in force, as each user's own build ran before the tree was
    shared.

    Attributes:
        made: What *make* returned besides the tree, once made: a
            ``click.testing.Result`` for a build, so a test can read what the
            build printed.
    """

    def __init__(self, home: Path, make: Callable[[Path], tuple[Path, Any]]) -> None:
        """Hold *make* until the first :meth:`restore`.

        Args:
            home: An empty directory that lives as long as this object: the
                tree is made under ``home / "live"`` and its copy kept in
                ``home / "pristine"``. Made by the owning fixture's setup, so
                the scoped-temp plugin removes it when that fixture ends.
            make: Makes the tree in the directory it is handed and returns
                its root (inside that directory) and anything else to keep.
        """
        self._home = home
        self._make = make
        self._root: Path | None = None
        self._pristine = home / "pristine"
        self.made: Any = None

    def restore(self) -> Path:
        """The tree as *make* left it, at the path it was made at."""
        if self._root is None:
            live = self._home / "live"
            _remove_tree(live)  # what a failed make left
            live.mkdir()
            root, self.made = self._make(live)
            _remove_tree(self._pristine)  # what a failed copy left
            shutil.copytree(root, self._pristine, symlinks=True)
            self._root = root
            return root
        _remove_tree(self._root)
        shutil.copytree(self._pristine, self._root, symlinks=True)
        return self._root


def _built_once(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    record_name: str,
    make: Callable[[Path], Any],
) -> Any:
    """What ``make`` returns for a directory of its own, made once per run.

    Outside xdist the session calls ``make``. Under xdist the first worker to
    ask calls it in a directory every worker can read, holding a lock, and
    pickles the result as ``record_name`` in the run's base temp directory;
    every later worker reads that record.

    Args:
        tmp_path_factory: The session's temp path factory.
        name: The name the directory handed to ``make`` starts with.
        record_name: The record's file name.
        make: Makes the thing in the directory it is handed.

    Returns:
        What ``make`` returned.
    """
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        return make(tmp_path_factory.mktemp(name))
    shared = tmp_path_factory.getbasetemp().parent
    record = shared / record_name
    with _exclusive(shared / f"{record_name}.lock"):
        if record.is_file():
            return pickle.loads(record.read_bytes())
        made = make(shared_dir(tmp_path_factory, name))
        record.write_bytes(pickle.dumps(made))
        return made


@pytest.fixture(scope="session")
def built_demo(tmp_path_factory: pytest.TempPathFactory) -> BuiltProject:
    """The demo preset, initialised and built once per run.

    Tests read it and never write to it. Outside xdist the session builds it;
    under xdist the workers share one build in the run's base temp directory.
    """
    built: BuiltProject = _built_once(tmp_path_factory, "built-ca", SHARED_BUILD_RECORD, build_demo)
    return built


@pytest.fixture(scope="session")
def built_demo_orbit_bump(tmp_path_factory: pytest.TempPathFactory) -> BuiltProject:
    """The demo preset with orbit bump turned on at init, built once per run.

    The demo preset ships no application; this is the build its tests read.
    Tests read it and never write to it.
    """
    from tests.fixtures.lifecycle_repo import ENABLE_ORBIT_BUMP

    built: BuiltProject = _built_once(
        tmp_path_factory,
        "built-ca-orbit-bump",
        SHARED_ORBIT_BUMP_BUILD_RECORD,
        lambda parent: build_demo(parent, ENABLE_ORBIT_BUMP),
    )
    return built


#: The preset's one watchdog, as ``osprey init`` writes it into ``profile.yml``: off.
WATCHDOG_OFF = "  guard-circumvention:\n    engine: decision\n    enabled: false\n"


def build_watchdog_on(built: Path, parent: Path) -> Path:
    """Copy the built demo repo, turn its watchdog on, and rebuild it in place.

    The rebuild runs over the copy's disabled build, as an operator's rebuild
    after the profile edit runs over theirs.

    Args:
        built: The built demo repo, watchdog off.
        parent: The directory the copy is made in.

    Returns:
        The rebuilt copy.
    """
    from tests.cli._lifecycle_build import invoke_build

    repo = copy_build(built, parent)
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert text.count(WATCHDOG_OFF) == 1
    profile.write_text(
        text.replace(WATCHDOG_OFF, WATCHDOG_OFF.replace("false", "true")), encoding="utf-8"
    )
    rebuilt = invoke_build(repo)
    assert rebuilt.exit_code == 0, rebuilt.output
    return repo


@pytest.fixture(scope="session")
def built_demo_watchdog_on(
    tmp_path_factory: pytest.TempPathFactory, built_demo: BuiltProject
) -> Path:
    """``built_demo`` with ``guard-circumvention`` enabled, rebuilt once per run.

    Tests read it and never write to it; one that writes takes a ``copy_build``.
    """
    repo: Path = _built_once(
        tmp_path_factory,
        "built-ca-watchdog-on",
        SHARED_WATCHDOG_BUILD_RECORD,
        lambda parent: build_watchdog_on(built_demo.repo, parent),
    )
    return repo


#: What a real ``osprey build`` runs under so its install reads uv's cache alone.
OFFLINE_BUILD_ENV = {"UV_OFFLINE": "1"}

#: Under xdist, the file in the run's shared base temp directory that says uv's
#: cache was shown to hold what a build installs.
SHARED_CACHE_RECORD = "uv-cache-serves-the-build"

# Filling an empty cache downloads osprey's whole dependency tree.
_CACHE_FILL_TIMEOUT_S = 1800


def uv_path() -> str | None:
    """The uv a build installs with, found as the build finds it."""
    return os.environ.get("UV") or shutil.which("uv")


def install_osprey_into(venv: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    """Create *venv* afresh and install osprey into it as a build of this tree does.

    The requirement is the one ``osprey build`` resolves for this
    installation, under the same lock constraints that build applies
    (:func:`~osprey.cli.build_environment.source_lock_constraints`), so what
    this install needs from uv's cache is what a preset's build needs: a fill
    resolved without them could cache versions an offline build never asks for
    and miss the ones it does.
    """
    from osprey.cli.build_environment import (
        _pins_prerelease,
        _resolve_osprey_spec,
        source_lock_constraints,
    )

    uv = uv_path()
    assert uv is not None
    spec, _ = _resolve_osprey_spec("local")
    created = subprocess.run(
        [uv, "venv", str(venv), "--python", sys.executable, "--quiet", "--clear"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    command = [uv, "pip", "install", "--quiet", "-p", str(venv / "bin" / "python")]
    if _pins_prerelease(spec):
        command += ["--prerelease", "allow"]
    lock = source_lock_constraints(spec)
    if lock is not None:
        lock_file = venv / "osprey-lock-constraints.txt"
        lock_file.write_text(lock, encoding="utf-8")
        command += ["-c", str(lock_file)]
    return subprocess.run(
        [*command, spec],
        capture_output=True,
        text=True,
        env=dict(env),
        timeout=_CACHE_FILL_TIMEOUT_S,
    )


def fill_uv_cache(parent: Path) -> None:
    """Leave uv's cache able to serve a build's install with the network off.

    A cache that already can is left alone and nothing is fetched. One that
    cannot is filled by one install with the network on, then shown to serve.
    """
    offline = {**os.environ, **OFFLINE_BUILD_ENV}
    # The venv only proves the install, so it goes with the fill.
    with tempfile.TemporaryDirectory(prefix="uv-fill-", dir=parent) as scratch:
        venv = Path(scratch) / "venv"
        if install_osprey_into(venv, offline).returncode == 0:
            return
        filled = install_osprey_into(venv, os.environ)
        assert filled.returncode == 0, (
            "uv's cache does not hold what an osprey build installs, and it could not be "
            f"fetched:\n{filled.stdout}{filled.stderr}"
        )
        served = install_osprey_into(venv, offline)
        assert served.returncode == 0, (
            "uv's cache was filled and still does not serve an osprey build's install:\n"
            f"{served.stdout}{served.stderr}"
        )


@pytest.fixture(scope="session")
def offline_build_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Environment additions under which a real ``osprey build`` needs no index.

    The build's install is unchanged; uv reads every package from its cache,
    which this fixture fills once per run if it has to. Empty when uv is not
    installed: the build then installs with pip, which has no such mode.
    """
    if uv_path() is None:
        return {}
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        fill_uv_cache(tmp_path_factory.getbasetemp())
        return dict(OFFLINE_BUILD_ENV)
    shared = tmp_path_factory.getbasetemp().parent
    record = shared / SHARED_CACHE_RECORD
    with _exclusive(shared / f"{SHARED_CACHE_RECORD}.lock"):
        if not record.is_file():
            fill_uv_cache(tmp_path_factory.getbasetemp())
            record.write_text("served\n")
    return dict(OFFLINE_BUILD_ENV)


def copy_build(
    repo: Path,
    parent: Path,
    *,
    ignore: Callable[[str, list[str]], Iterable[str]] | None = None,
) -> Path:
    """Copy a built repo to ``parent / repo.name``, re-anchored so nothing in it names *repo*.

    A build records its repo's absolute path (``project_root`` in each
    config.yml, the manifests, the MCP and settings files); rewriting it makes
    the copy self-contained, so a test may write to the copy of a shared build.
    The rewrite is on bytes, so a file with internal offsets (a database) would
    be corrupted; a build carries the path in none. A build has no symlinks.

    The copy has NO image context: ``build/.image`` is left out. Start and
    deploy preflight (``render_images.verify_renders``) refuse it as a partial
    render, the snapshot seeder finds fewer targets, and nothing can be
    image-built from it. A test that needs any of that rebuilds the copy.

    Args:
        repo: The built repo.
        parent: The directory the copy is made in, under the repo's own name,
            which init made the project name.
        ignore: Further entries to leave out, as :func:`shutil.copytree` takes it.

    Returns:
        The copy.
    """
    from osprey.utils.workspace import IMAGE_DIR_NAME
    from osprey_connectors.workspace import BUILD_DIR_NAME

    repo = repo.resolve()
    dest = parent.resolve() / repo.name
    build_zone = repo / BUILD_DIR_NAME

    def skip(src: str, names: list[str]) -> set[str]:
        skipped = set(ignore(src, names)) if ignore is not None else set()
        if Path(src) == build_zone:
            skipped.add(IMAGE_DIR_NAME)
        return skipped

    shutil.copytree(repo, dest, ignore=skip)
    old, new = str(repo).encode(), str(dest).encode()
    rewritten = 0
    for path in dest.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            if old in data:
                path.write_bytes(data.replace(old, new))
                rewritten += 1
    assert rewritten, f"no file in {repo} names its path; the copy would not be its own"
    return dest
