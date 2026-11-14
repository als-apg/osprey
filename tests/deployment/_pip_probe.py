"""Run Dockerfile RUN bodies under a real ``sh`` with stub tools on ``PATH``.

The shipped image recipes install the pinned framework with plain ``pip`` in
one long ``RUN``. What that ``pip`` is invoked with, and whether a failing
install fails the layer, is a property of the shell body, not of any rendered
string, so the tests here execute the body under a real ``sh`` with ``pip``,
``uv`` and ``apt-get`` shadowed on ``PATH``. Both the shipped service recipes
(:mod:`tests.deployment.test_service_dockerfiles`) and the rendered project
template (:mod:`tests.cli.test_dockerfile_template`) probe their bodies here.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

#: A no-op ``apt-get``.
APT_NOOP = "#!/bin/sh\nexit 0\n"

#: ``uv pip …`` handed to the ``pip`` stub beside it, so one stub decides the
#: outcome whichever spelling the body uses.
UV_TO_PIP = '#!/bin/sh\n[ "$1" = pip ] && shift\nexec "$(dirname "$0")/pip" "$@"\n'


def stubbed_env(tmp_path: Path, stubs: dict[str, str], **extra: str) -> dict[str, str]:
    """``os.environ`` with *stubs* (name to script) first on ``PATH``, plus *extra*.

    The stubs are written executable into ``tmp_path / "bin"``.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for name, script in stubs.items():
        stub = stub_bin / name
        stub.write_text(script)
        stub.chmod(0o755)
    return dict(os.environ, PATH=f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}", **extra)


def run_body(body: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a RUN body under ``sh -c`` with *env*.

    Waited on until the body exits, with no wall-clock budget: every tool it
    calls is a stub, so it ends on its own however slowly a loaded host starts
    them, and the per-test timeout is the hang breaker.
    """
    return subprocess.run(["sh", "-c", body], capture_output=True, text=True, env=env)


def deps_sandbox(
    body: str, tmp_path: Path, staged: dict[str, str] | None = None
) -> tuple[str, Path]:
    """Point a deps-layer body at temp dirs; return ``(rewritten body, ctx)``.

    ``/tmp/deps-ctx`` becomes a build context holding a ``.dockerignore`` and
    any *staged* files (name to content) — the dev-staged manifests and lock
    constraints a ``--dev`` build COPYs there — the apt lists directory an
    empty temp dir, and ``/opt/osprey`` (where the project image writes its
    frozen constraints, and the lock-driven path keeps the lock) a temp dir
    of its own.
    """
    ctx = tmp_path / "deps-ctx"
    ctx.mkdir()
    (ctx / ".dockerignore").write_text("")
    for name, content in (staged or {}).items():
        (ctx / name).write_text(content)
    apt_lists = tmp_path / "apt-lists"
    apt_lists.mkdir()
    rewritten = (
        body.replace("/tmp/deps-ctx", str(ctx))
        .replace("/var/lib/apt/lists", str(apt_lists))
        .replace("/opt/osprey", str(tmp_path / "opt-osprey"))
    )
    return rewritten, ctx


def probe_wheel_body(body: str, tmp_path: Path, *, with_wheel: bool):
    """Execute a wheel-layer RUN body in a sandbox with a real shell.

    ``/tmp/ctx`` is rewritten to a temp dir (optionally holding a fake wheel),
    ``/opt/osprey`` to one holding the frozen constraints the project image's
    body reads, and a stub ``pip`` that always exits 1 shadows the real one on
    PATH, so the probe exercises the body's failure-propagation shape without
    a container. Returns ``(CompletedProcess, ctx_path)``.
    """
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / ".dockerignore").write_text("")
    if with_wheel:
        (ctx / "osprey_framework-0.0.0-py3-none-any.whl").write_text("")
    frozen = tmp_path / "opt-osprey"
    frozen.mkdir()
    (frozen / "constraints.txt").write_text("osprey-framework==0.0.0\n")
    env = stubbed_env(tmp_path, {"pip": "#!/bin/sh\nexit 1\n"})
    rewritten = body.replace("/tmp/ctx", str(ctx)).replace("/opt/osprey", str(frozen))
    return run_body(rewritten, env), ctx


def deps_pip_calls(
    body: str, tmp_path: Path, env: dict[str, str], staged: dict[str, str] | None = None
) -> list[list[str]]:
    """Every ``pip`` invocation the deps layer makes, in order, as argv lists.

    The body runs in :func:`deps_sandbox`'s temp dirs (holding *staged*),
    ``apt-get`` is a no-op and ``pip`` appends every invocation to a log and
    exits 0. *env* carries the build ARGs the body reads (``OSPREY_PIP_SPEC``
    or ``OSPREY_VERSION``, ``OSPREY_DEV``, ``OSPREY_PIP_PRE`` …).
    """
    rewritten, _ = deps_sandbox(body, tmp_path, staged)
    log = tmp_path / "pip-calls.log"
    pip = f'#!/bin/sh\nprintf "%s\\n" "$*" >> {shlex.quote(str(log))}\nexit 0\n'
    run_env = stubbed_env(
        tmp_path, {"apt-get": APT_NOOP, "pip": pip, "uv": UV_TO_PIP}, OSPREY_DEV="", PIP_NO_PROXY=""
    )
    run_env.update(env)
    result = run_body(rewritten, run_env)
    assert result.returncode == 0, (
        f"deps RUN failed under the stub:\n{result.stdout}\n{result.stderr}"
    )
    return [shlex.split(line) for line in log.read_text().splitlines()]


def primer_pip_argv(body: str, tmp_path: Path, env: dict[str, str]) -> list[str]:
    """The argv of the deps layer's framework (or optimizer) install, as ``pip`` received it.

    Runs the body through :func:`deps_pip_calls` and returns the argv of the
    first ``pip install`` naming ``osprey-framework`` or, in the optimizer's
    image, ``osprey-optimizer``.
    """
    calls = deps_pip_calls(body, tmp_path, env)
    primed = ("osprey-framework", "osprey-optimizer")
    primers = [c for c in calls if "install" in c and any(d in a for a in c for d in primed)]
    assert primers, f"no framework (or optimizer) install reached pip; calls: {calls}"
    return primers[0]


def wheel_pip_calls(body: str, tmp_path: Path, *, lock: bool) -> list[list[str]]:
    """Every ``pip`` invocation a wheel-layer body makes with a wheel staged.

    ``/tmp/ctx`` holds a stand-in framework wheel and ``/opt/osprey`` the
    frozen constraints the project image's body reads, plus — with *lock* —
    the ``lock-constraints.txt`` the lock-driven deps layer leaves there.
    ``pip`` logs each call and exits 0, and ``python`` is a no-op, so the
    framework modules the layer runs need not exist.
    """
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / ".osprey-layer-anchor").write_text("")
    (ctx / "osprey_framework-0.0.0-py3-none-any.whl").write_text("")
    opt = tmp_path / "opt-osprey"
    opt.mkdir()
    (opt / "constraints.txt").write_text("fwdep==1.0\n")
    if lock:
        (opt / "lock-constraints.txt").write_text("fwdep==1.0\n")
    log = tmp_path / "pip-calls.log"
    pip = f'#!/bin/sh\nprintf "%s\\n" "$*" >> {shlex.quote(str(log))}\nexit 0\n'
    env = stubbed_env(tmp_path, {"pip": pip, "python": "#!/bin/sh\nexit 0\n"})
    rewritten = body.replace("/tmp/ctx", str(ctx)).replace("/opt/osprey", str(opt))
    result = run_body(rewritten, env)
    assert result.returncode == 0, (
        f"wheel RUN failed under the stub:\n{result.stdout}{result.stderr}"
    )
    return [shlex.split(line) for line in log.read_text().splitlines()]


#: The one pin the stand-in lock carries; an install is held to the lock when a
#: constraints file it reads (``-c`` or ``PIP_CONSTRAINT`` / ``UV_CONSTRAINT``)
#: names it.
LOCK_MARKER = "lock-marker==1.0"


def lock_path_install_calls(
    body: str, tmp_path: Path, env: dict[str, str], *, source_build: bool = True
) -> list[tuple[bool, list[str]]]:
    """Run *body* as a source build would, and report each installer call.

    Everything a ``--dev`` build stages is in place — the lock constraints in
    the deps context and, as the deps layer leaves it, in ``/opt/osprey``;
    the base and extras manifests; a framework, optimizer and connectors
    wheel in ``/tmp/ctx`` — and ``pip`` (with ``uv pip`` handed to it) logs
    each call with whether it read the lock. With *source_build* false
    nothing is staged, as for a release build. Returns ``(held to the lock,
    argv)`` per call, in order.
    """
    staged_files = {
        "osprey-local-requirements.txt": "fwdep\n",
        "osprey-extra-requirements.txt": "extradep\n",
        "osprey-lock-constraints.txt": f"{LOCK_MARKER}\n",
    }
    sandbox = {
        "/tmp/deps-ctx": tmp_path / "deps-ctx",
        "/tmp/extras-ctx": tmp_path / "extras-ctx",
        "/tmp/ctx": tmp_path / "ctx",
        "/opt/osprey": tmp_path / "opt-osprey",
        "/var/lib/apt/lists": tmp_path / "apt-lists",
    }
    for directory in sandbox.values():
        directory.mkdir()
    (sandbox["/opt/osprey"] / "constraints.txt").write_text("fwdep==1.0\n")
    if source_build:
        for ctx in (sandbox["/tmp/deps-ctx"], sandbox["/tmp/extras-ctx"]):
            for name, content in staged_files.items():
                (ctx / name).write_text(content)
        (sandbox["/opt/osprey"] / "lock-constraints.txt").write_text(f"{LOCK_MARKER}\n")
        for wheel in ("osprey_framework", "osprey_optimizer", "osprey_connectors"):
            (sandbox["/tmp/ctx"] / f"{wheel}-0.0.0-py3-none-any.whl").write_text("")
    rewritten = body
    for path, directory in sandbox.items():
        rewritten = rewritten.replace(path, str(directory))
    log = tmp_path / "pip-calls.log"
    pip = (
        "#!/bin/sh\n"
        "held=0\n"
        f"for f in $PIP_CONSTRAINT $UV_CONSTRAINT; do grep -qs '{LOCK_MARKER}' \"$f\" && held=1; done\n"
        'prev=""\n'
        'for a in "$@"; do\n'
        f'  [ "$prev" = -c ] && grep -qs \'{LOCK_MARKER}\' "$a" && held=1\n'
        '  prev="$a"\n'
        "done\n"
        f'printf "%s %s\\n" "$held" "$*" >> {shlex.quote(str(log))}\n'
        "exit 0\n"
    )
    run_env = stubbed_env(
        tmp_path,
        {"apt-get": APT_NOOP, "pip": pip, "uv": UV_TO_PIP, "python": "#!/bin/sh\nexit 0\n"},
    )
    run_env.update(env)
    result = run_body(rewritten, run_env)
    assert result.returncode == 0, f"RUN failed under the stub:\n{result.stdout}{result.stderr}"
    if not log.exists():
        return []
    calls = []
    for line in log.read_text().splitlines():
        held, _, argv = line.partition(" ")
        calls.append((held == "1", shlex.split(argv)))
    return calls


def must_be_held_to_the_lock(argv: list[str]) -> bool:
    """Whether an installer call *argv* resolves third-party packages.

    Exempt: the uv bootstrap (uv is the installer, not an image package, and
    is removed before its layer ends), the ``--no-deps`` reinstall of the
    staged wheels (it resolves nothing), and every non-install verb.
    """
    if argv[:1] != ["install"]:
        return False
    if any(a.startswith("uv==") for a in argv):
        return False
    return "--no-deps" not in argv
