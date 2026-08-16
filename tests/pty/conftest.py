"""The two pillars the pty harness stands on: a real repo, and a fake runtime.

Everything under ``tests/pty/`` drives OSPREY's own verbs on a real terminal and
reads what an operator would have seen. That rests on two things this module
provides:

**A deployment repo rendered by the real machinery.** Not a fixture-authored
tree — ``osprey init`` then ``osprey build``, the same two commands a user runs,
in a subprocess so nothing about the render leaks into the test process. The
result is cached under ``.cache/pty-exemplar/`` and keyed on the content of
everything that decides what a render looks like: the template data, the
scaffolder code that stamps it out, and the preset. Cold is a couple of
seconds and paid once per session; warm is a directory copy.

**A container runtime that is not there.** :mod:`tests.pty.stub_runtime` — a
``docker`` executable earlier on ``PATH`` than any real one, plus
``CONTAINER_RUNTIME=docker`` so the runtime probe pins to it. Every invocation
is logged; :class:`StubRuntime` reads the log back.

The exemplar is shared **read-only**. A start verb writes to the repo it starts
— it mints service tokens into ``.env``, generates key material under ``data/``,
and spools command output under ``var/`` — and the profile fingerprint notices,
so a second test starting the same directory is refused as stale. Use
``exemplar_copy`` for anything that runs a verb; the shared tree is for reading.

The preset is ``hello-world`` for one reason: it is the only bundled preset that
both deploys services and comes up against a stubbed runtime alone. The
control-assistant family reaches past the runtime to real network services (the
archiver's mongod, ARIEL's postgres) and blocks on them; the standalone presets
deploy no services at all, so there is no start to watch.

On top of those two, the fixtures every scenario module drives a verb with live
here as well: ``pty_env`` (the child environment), ``unhurried_runtime`` (a real
runtime's latency in front of the stub) and ``startable_repo`` (a writable copy
whose published ports are free on this host). They are shared by more than one
module, and a fixture pytest resolves by name is one no scenario module has to
import from another.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

#: The preset the exemplar is rendered from. See the module docstring for why
#: it is this one and not a richer profile.
EXEMPLAR_PRESET = "hello-world"

#: Directory name of the rendered repo inside its cache entry.
EXEMPLAR_DIRNAME = "exemplar"

#: A key baked into the render so a change to *this file's* render procedure
#: invalidates cached repos that were produced by the old one. Bump it whenever
#: the render steps or the seeded ``.env`` change.
RENDER_RECIPE_VERSION = "1"

#: The API key seeded into the rendered repo's ``.env``. A start verb refuses
#: outright without a secret store (``.env`` is the only one), so this is part
#: of the render, not of any one test. Obviously not a credential.
STUB_API_KEY = "sk-ant-stub-0000000000000000000000000000"

#: Everything whose content decides what a render looks like. The scaffolder
#: code is in the key beside its data: a template that is stamped out
#: differently renders differently, and a key that watched only the data would
#: serve a stale repo for the rest of the session. So the two verbs that drive
#: the render, the generator that writes the compose files, and the profile
#: machinery that resolves the preset are all in the key beside the templates —
#: each of them can change the rendered repo without a template changing at all.
#:
#: Targeted rather than "all of ``src/osprey``": a key over the whole package
#: would re-render on every unrelated edit in the tree, which on a working
#: checkout is every edit.
_KEY_INPUTS = (
    Path("src/osprey/templates"),
    Path("src/osprey/cli/templates"),
    Path("src/osprey/deployment/compose_generator.py"),
    Path("src/osprey/cli/init_cmd.py"),
    Path("src/osprey/cli/build_cmd.py"),
    Path("src/osprey/profiles"),
)

#: Directories never worth hashing — build artifacts of the interpreter, not
#: inputs to a render.
_KEY_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})


def repo_root() -> Path:
    """The OSPREY checkout these tests run against."""
    return Path(__file__).resolve().parents[2]


def preset_path(preset: str = EXEMPLAR_PRESET) -> Path:
    """The bundled preset file the exemplar is rendered from."""
    return repo_root() / "src" / "osprey" / "profiles" / "presets" / f"{preset}.yml"


def exemplar_cache_key(preset: str = EXEMPLAR_PRESET) -> str:
    """A content hash of everything that decides what the render looks like.

    Spelled as a content hash rather than as ``git rev-parse HEAD:<path>``
    because a working tree is the thing under test: a developer editing a
    template has not committed it yet, and a key that could not see the edit
    would hand every test in the session a repo rendered from the old one.
    Committed content hashes to the same value either way.

    Args:
        preset: Preset name, folded into the key with its file's content.

    Returns:
        A short hex digest, stable across processes and machines.
    """
    digest = hashlib.sha256()
    digest.update(f"recipe={RENDER_RECIPE_VERSION}\npreset={preset}\n".encode())
    digest.update(preset_path(preset).read_bytes())
    for relative in _KEY_INPUTS:
        root = repo_root() / relative
        entries = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in entries:
            if not path.is_file() or set(path.parts) & _KEY_SKIP_DIRS:
                continue
            name = path.relative_to(root) if root.is_dir() else path.name
            # NUL between the name and the bytes: without a separator the
            # concatenation is ambiguous — renaming a file and editing the one
            # before it can produce the same stream, and the key would then
            # serve a repo rendered from templates that no longer exist. The
            # executable bit is hashed with the content because a scaffolder's
            # mode bit survives the render into the repo it stamps out.
            digest.update(str(name).encode())
            digest.update(b"\0")
            digest.update(f"{path.stat().st_mode & 0o111:o}\0".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def exemplar_cache_dir(preset: str = EXEMPLAR_PRESET) -> Path:
    """Where a render of *preset* is cached.

    Stable across runs and inside the checkout (``.cache/`` is gitignored), so
    a CI lane can restore the parent directory wholesale: the key in the leaf
    name is what decides whether a restored entry is actually usable, so a
    coarse cache key on the CI side can never serve a stale repo.
    """
    return repo_root() / ".cache" / "pty-exemplar" / f"{preset}-{exemplar_cache_key(preset)}"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _cli(*args: str, cwd: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    """Run one osprey verb in a subprocess of this interpreter.

    A subprocess rather than ``CliRunner``: the render loads dotenv files into
    the process environment, installs logging, and changes directory, and a
    session-scoped fixture that did any of that in-process would hand every
    later test a polluted one. ``sys.executable -c`` rather than the console
    script so the render always runs the same interpreter — and therefore the
    same checkout — as the tests.
    """
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from osprey.cli.main import cli; sys.exit(cli())",
            *args,
        ],
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _render_env() -> dict[str, str]:
    """Environment for the render subprocesses.

    Every ``OSPREY_``/``COMPOSE_``/``DOCKER_`` variable the developer's shell
    happens to carry is dropped first. The render is cached under a key computed
    from repository content alone, so anything in the ambient environment that
    can change what is rendered — a pointer to another config, a compose project
    name, a daemon socket — would be a cache key the key does not have, and one
    session's shell would decide what every later session is handed.

    The stub runtime is then put on ``PATH``, even though a render does not talk
    to a runtime today. If one ever probes, it must reach the stub rather than
    the developer's daemon — a fixture that quietly started containers would be
    discovered the hard way.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OSPREY_", "COMPOSE_", "DOCKER_"))
    }
    env["CONTAINER_RUNTIME"] = "docker"
    env["PATH"] = f"{stub_runtime_dir()}{os.pathsep}{env.get('PATH', '')}"
    env["NO_COLOR"] = "1"
    return env


def render_exemplar(destination: Path, preset: str = EXEMPLAR_PRESET) -> Path:
    """Render one deployment repo into *destination* with the real verbs.

    ``init`` (no git — a commit needs an identity CI may not have), a seeded
    ``.env``, then ``build --skip-deps --skip-lifecycle``: the venv install and
    the profile's shell phases cost minutes and render nothing.

    Args:
        destination: Directory to create the repo in. Created if absent.
        preset: Bundled preset name.

    Returns:
        The rendered repo root.

    Raises:
        RuntimeError: When either verb fails, with its own output attached —
            a broken render is a broken harness, and swallowing the reason
            would leave every pty test failing for no visible cause.
    """
    destination.mkdir(parents=True, exist_ok=True)
    env = _render_env()

    result = _cli(
        "init", EXEMPLAR_DIRNAME, "--preset", preset, "--no-git", cwd=destination, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"osprey init failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )

    repo = destination / EXEMPLAR_DIRNAME
    (repo / ".env").write_text(f"ANTHROPIC_API_KEY={STUB_API_KEY}\n", encoding="utf-8")

    result = _cli("build", "--skip-deps", "--skip-lifecycle", cwd=repo, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"osprey build failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    return repo


def ensure_exemplar(preset: str = EXEMPLAR_PRESET) -> Path:
    """The cached render of *preset*, rendering it first if there is none.

    Built into a sibling scratch directory and moved into place, so a cache
    entry either does not exist or is complete: a half-rendered repo left by an
    interrupted session would otherwise be served to every later one.
    """
    cached = exemplar_cache_dir(preset)
    repo = cached / EXEMPLAR_DIRNAME
    if repo.is_dir():
        return repo

    staging = cached.with_name(f"{cached.name}.building-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        render_exemplar(staging, preset)
        cached.parent.mkdir(parents=True, exist_ok=True)
        try:
            staging.rename(cached)
        except OSError:
            # Another session won the race and put its own render here. Its
            # render is this render — same key, same inputs — so take theirs.
            if not repo.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return repo


# ---------------------------------------------------------------------------
# the stub runtime
# ---------------------------------------------------------------------------


def stub_runtime_dir() -> Path:
    """The directory holding the stub ``docker`` executable."""
    return Path(__file__).resolve().parent / "stub_runtime"


@dataclass(frozen=True)
class StubRuntime:
    """A configured stub runtime, and the log it writes.

    Attributes:
        directory: The directory to put first on ``PATH``.
        log_path: JSONL file the stub appends one record per invocation to.
        state_path: JSON file the stub keeps its "what is up" state in.
    """

    directory: Path
    log_path: Path
    state_path: Path

    def env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """*base* (default ``os.environ``) with the stub wired into it.

        ``CONTAINER_RUNTIME=docker`` is not decoration: it is read by
        :func:`osprey.deployment.runtime_helper._runtimes_to_try` ahead of
        config, so the probe never falls through to podman and finds a real
        one behind the stub.
        """
        env = dict(os.environ if base is None else base)
        env["PATH"] = f"{self.directory}{os.pathsep}{env.get('PATH', '')}"
        env["CONTAINER_RUNTIME"] = "docker"
        env["OSPREY_STUB_RUNTIME_LOG"] = str(self.log_path)
        env["OSPREY_STUB_RUNTIME_STATE"] = str(self.state_path)
        return env

    def invocations(self) -> list[dict[str, Any]]:
        """Every invocation the stub has answered, in order."""
        if not self.log_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def seams(self) -> list[str]:
        """The resolved seam of every invocation, in order (``"compose up"``)."""
        return [record["seam"] for record in self.invocations()]

    def unhandled(self) -> list[dict[str, Any]]:
        """Invocations the stub did not recognize — the coverage gap, if any."""
        return [record for record in self.invocations() if not record["handled"]]

    def reset(self) -> None:
        """Forget every invocation and every started container."""
        self.log_path.unlink(missing_ok=True)
        self.state_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers this directory's fixtures read.

    Declared here rather than in ``pyproject.toml`` because the marker is a
    contract with the ``stub_runtime`` fixture, and nothing outside this
    directory can use it.
    """
    config.addinivalue_line(
        "markers",
        "allow_unhandled_runtime: the test exercises the stub runtime's "
        "unhandled path on purpose, so the stub_runtime fixture must not fail "
        "it for the invocations it could not answer.",
    )


@pytest.fixture(scope="session")
def exemplar_preset() -> str:
    """The preset the shared exemplar is rendered from."""
    return EXEMPLAR_PRESET


@pytest.fixture(scope="session")
def exemplar_repo() -> Path:
    """The shared, rendered deployment repo. **Read-only.**

    Rendered once per session (cached across sessions) by the real ``init`` and
    ``build``. Anything that runs a verb against it must take an
    ``exemplar_copy`` first — see the module docstring.
    """
    return ensure_exemplar()


@pytest.fixture
def exemplar_copy(exemplar_repo: Path, tmp_path: Path) -> Path:
    """A private, writable copy of the exemplar for one test.

    The copy is what a start verb is allowed to scribble on: it mints tokens
    into ``.env``, writes key material under ``data/``, and spools output under
    ``var/``, all of which the profile fingerprint sees.
    """
    destination = tmp_path / exemplar_repo.name
    shutil.copytree(exemplar_repo, destination, symlinks=True)
    return destination


@pytest.fixture
def stub_runtime(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StubRuntime]:
    """A stub container runtime with a per-test invocation log.

    Teardown fails the test on any invocation the stub could not answer. That
    check belongs here rather than in each test: the stub exits 0 on an argv it
    does not know (see :mod:`tests.pty.stub_runtime._stub_main`), so a deploy
    that asked for something unscripted looks *successful* to everything above
    it, and a test that only asserted on rendered output would keep passing
    while measuring a deploy that half happened.

    A test that is *about* the unhandled path marks itself
    ``@pytest.mark.allow_unhandled_runtime``.
    """
    stub = StubRuntime(
        directory=stub_runtime_dir(),
        log_path=tmp_path / "stub-runtime.jsonl",
        state_path=tmp_path / "stub-runtime-state.json",
    )
    yield stub

    if "allow_unhandled_runtime" in request.keywords:
        return
    unhandled = stub.unhandled()
    if unhandled:
        pytest.fail(
            "the stub runtime was asked for invocations it cannot answer, so this test "
            "measured a deploy that only partly happened:\n  "
            + "\n  ".join(" ".join(record["argv"]) for record in unhandled)
            + "\nTeach tests/pty/stub_runtime/_stub_main.py the seam (handler + HANDLED_* "
            "set), or mark the test @pytest.mark.allow_unhandled_runtime if the gap is "
            "the point."
        )


@pytest.fixture
def pty_env(stub_runtime: StubRuntime) -> dict[str, str]:
    """The child's environment: stub runtime, a real terminal type, no color veto.

    Every ambient ``OSPREY_``/``COMPOSE_``/``DOCKER_`` variable is dropped for
    the same reason the exemplar render drops them — a developer's shell must
    not decide what the scenario deploys. ``COLUMNS``/``LINES`` go too: Rich
    reads them ahead of the pty's own window size, so leaving them in would let
    the runner's terminal set the width the screen is asserted at.

    The two secrets the scenarios drive with are cleared here and set back by
    the scenarios that want them, so an exported key on the developer's shell
    cannot arm or disarm a scenario silently.
    """
    base = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OSPREY_", "COMPOSE_", "DOCKER_"))
    }
    for key in ("NO_COLOR", "FORCE_COLOR", "COLUMNS", "LINES", "ANTHROPIC_API_KEY"):
        base.pop(key, None)
    env = stub_runtime.env(base)
    env["TERM"] = "xterm-256color"
    return env


#: Seconds of latency the unhurried runtime wrapper adds to each runtime call.
#: Chosen against the monitor's 0.25 s tick: a deploy whose runtime answers
#: instantly can finish a phase between two repaints, and a scenario about
#: repainting would then be asserting about a region that never got the chance.
RUNTIME_LATENCY = 0.4


@pytest.fixture
def unhurried_runtime(stub_runtime: StubRuntime, pty_env: dict[str, str], tmp_path: Path) -> None:
    """Give the stub runtime the latency a real one has.

    The stub answers in microseconds, which no container runtime does: a
    ``compose up`` against a daemon is a round trip measured in tenths of a
    second at best. Scenarios about the live region need that, because the
    region repaints on a 0.25 s tick and a phase that opens and closes inside
    one tick paints nothing at all — the harness would then be asserting about
    a region the deploy was simply too fast to draw.

    Spelled as a wrapper ahead of the stub on ``PATH`` rather than as a change
    to the stub, so the invocation log, the seams and the answers are exactly
    the ones every other pty test sees; the only difference is when they
    arrive.
    """
    directory = tmp_path / "unhurried-runtime"
    directory.mkdir()
    launcher = directory / "docker"
    launcher.write_text(
        f'#!/bin/sh\nsleep {RUNTIME_LATENCY}\nexec "{stub_runtime.directory}/docker" "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    pty_env["PATH"] = f"{directory}{os.pathsep}{pty_env['PATH']}"


def _free_host_ports(count: int) -> list[int]:
    """*count* distinct ports, none of them listened on as of now.

    Every probe socket is held open until the whole set has been allocated.
    Asking the kernel *count* times in a row and closing each probe before the
    next would let it hand back a port it has just released, and two services
    published on one port is a deploy that fails for a reason that has nothing
    to do with what the scenario is testing. Nothing binds these ports
    afterwards — the runtime is a stub — so releasing them at the end is safe;
    what matters is that they were distinct when they were chosen.
    """
    probes = []
    try:
        for _ in range(count):
            probe = socket.socket()
            probe.bind(("127.0.0.1", 0))
            probes.append(probe)
        return [int(probe.getsockname()[1]) for probe in probes]
    finally:
        for probe in probes:
            probe.close()


@pytest.fixture
def startable_repo(exemplar_copy: Path) -> Path:
    """An ``exemplar_copy`` whose published ports are free on this host.

    The start verb's host-port preflight is a real socket probe and refuses the
    deploy before touching the runtime when a port is taken — correctly, since
    it cannot tell a stub deploy from a real one. On a developer box something
    usually holds the exemplar's 5080 (a container runtime's own proxy, most
    often), and the scenarios below are about the terminal, not about port
    policy. So each published binding is moved to a free port in *this copy's*
    rendered compose file, which is the same edit ``osprey up``'s own refusal
    asks the operator for.

    Rendered output only: nothing in ``profile.yml`` changes, so the build
    fingerprint still matches and the start is not a stale one.
    """
    from osprey.deployment.container_lifecycle import as_built_compose_files, as_built_config_path
    from osprey.deployment.host_ports import parse_host_port_bindings
    from osprey.utils.config import load_project_config

    config = load_project_config(str(as_built_config_path(exemplar_copy)), wrap_errors=True)
    compose_files = [
        str(path) if os.path.isabs(str(path)) else str(exemplar_copy / str(path))
        for path in as_built_compose_files(config, exemplar_copy)
    ]
    bindings = parse_host_port_bindings(compose_files)
    assert bindings, (
        f"no published host ports were parsed out of {compose_files}; the rewrite below "
        f"would be a no-op and the scenarios would depend on this host's free ports"
    )
    for binding, port in zip(bindings, _free_host_ports(len(bindings)), strict=True):
        path = Path(binding.compose_file)
        text = path.read_text(encoding="utf-8")
        moved = text.replace(
            f"{binding.host_ip}:{binding.host_port}:", f"{binding.host_ip}:{port}:"
        )
        assert moved != text, f"could not move {binding.service}'s published port in {path}"
        path.write_text(moved, encoding="utf-8")
    return exemplar_copy
