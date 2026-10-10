"""Real builds shared across test directories, used by ``tests/conftest.py``.

``built_control_assistant`` is the one real ``osprey build --skip-deps`` of the
control-assistant preset that the facility and build tests read. It is
session-scoped, and under xdist the workers of one run share it: the first
worker to ask builds it under the run's base temp directory, holding a lock, and
every later worker reads that build, so the build runs once per run whichever
groups its readers sit in.

``offline_build_env`` is for the tests that run a real ``osprey build`` with its
install: the environment additions under which that install reads uv's cache
and needs no index. It shows once per run that the cache holds what a build
installs, filling it if it does not, and each test passes the additions to its
own build call.
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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from click.testing import CliRunner, Result


def init_project(parent: Path, preset: str, name: str) -> Path:
    """Run ``osprey init`` for a preset into ``parent/name``.

    Args:
        parent: The directory the repo is created in.
        preset: The preset name.
        name: The repo's directory name, which is the project name.

    Returns:
        The repo.
    """
    from osprey.cli.init_cmd import init

    repo = parent / name
    result = CliRunner().invoke(init, [str(repo), "--preset", preset, "--no-git"])
    assert result.exit_code == 0, result.output
    return repo


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
    """

    repo: Path
    outputs: list[RenderOutputs] = field(default_factory=list)

    @property
    def build_dir(self) -> Path:
        """The deployment's own render root."""
        from osprey.utils.workspace import BUILD_DIR_NAME

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
#: built repo and holds its render outputs.
SHARED_BUILD_RECORD = "built-ca.pickle"


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``lock_path`` for the block."""
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def build_control_assistant(parent: Path) -> BuiltProject:
    """Initialise and build the control-assistant preset under ``parent``.

    Args:
        parent: The directory the repo is created in.

    Returns:
        The built repo and what each render's facility outputs were.
    """
    from osprey.facility import render

    repo = init_project(parent, "control-assistant", "demo")
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
    return built


@pytest.fixture(scope="session")
def built_control_assistant(tmp_path_factory: pytest.TempPathFactory) -> BuiltProject:
    """The control-assistant preset, initialised and built once per run.

    Tests read it and never write to it. Outside xdist the session builds it;
    under xdist the workers share one build in the run's base temp directory.
    """
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        return build_control_assistant(tmp_path_factory.mktemp("built-ca"))
    shared = tmp_path_factory.getbasetemp().parent
    record = shared / SHARED_BUILD_RECORD
    with _exclusive(shared / f"{SHARED_BUILD_RECORD}.lock"):
        if record.is_file():
            repo, outputs = pickle.loads(record.read_bytes())
            return BuiltProject(repo, outputs)
        built = build_control_assistant(Path(tempfile.mkdtemp(prefix="built-ca-", dir=shared)))
        record.write_bytes(pickle.dumps((built.repo, built.outputs)))
        return built


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
    installation, so what this install needs from uv's cache is what a
    preset's build needs.
    """
    from osprey.cli.build_environment import _pins_prerelease, _resolve_osprey_spec

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
    return subprocess.run(
        [*command, spec],
        capture_output=True,
        text=True,
        env=dict(env),
        timeout=_CACHE_FILL_TIMEOUT_S,
    )


def fill_uv_cache(scratch: Path) -> None:
    """Leave uv's cache able to serve a build's install with the network off.

    A cache that already can is left alone and nothing is fetched. One that
    cannot is filled by one install with the network on, then shown to serve.
    """
    venv = scratch / "venv"
    offline = {**os.environ, **OFFLINE_BUILD_ENV}
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
        fill_uv_cache(tmp_path_factory.mktemp("uv-cache"))
        return dict(OFFLINE_BUILD_ENV)
    shared = tmp_path_factory.getbasetemp().parent
    record = shared / SHARED_CACHE_RECORD
    with _exclusive(shared / f"{SHARED_CACHE_RECORD}.lock"):
        if not record.is_file():
            fill_uv_cache(Path(tempfile.mkdtemp(prefix="uv-cache-", dir=shared)))
            record.write_text("served\n")
    return dict(OFFLINE_BUILD_ENV)
