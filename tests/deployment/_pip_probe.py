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
    """Run a RUN body under ``sh -c`` with *env*."""
    return subprocess.run(["sh", "-c", body], capture_output=True, text=True, env=env, timeout=60)


def deps_sandbox(body: str, tmp_path: Path) -> tuple[str, Path]:
    """Point a deps-layer body at temp dirs; return ``(rewritten body, ctx)``.

    ``/tmp/deps-ctx`` becomes an empty build context holding only a
    ``.dockerignore``, and the apt lists directory an empty temp dir.
    """
    ctx = tmp_path / "deps-ctx"
    ctx.mkdir()
    (ctx / ".dockerignore").write_text("")
    apt_lists = tmp_path / "apt-lists"
    apt_lists.mkdir()
    rewritten = body.replace("/tmp/deps-ctx", str(ctx)).replace(
        "/var/lib/apt/lists", str(apt_lists)
    )
    return rewritten, ctx


def probe_wheel_body(body: str, tmp_path: Path, *, with_wheel: bool):
    """Execute a wheel-layer RUN body in a sandbox with a real shell.

    ``/tmp/ctx`` is rewritten to a temp dir (optionally holding a fake wheel)
    and a stub ``pip`` that always exits 1 shadows the real one on PATH, so the
    probe exercises the body's failure-propagation shape without a container.
    Returns ``(CompletedProcess, ctx_path)``.
    """
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / ".dockerignore").write_text("")
    if with_wheel:
        (ctx / "osprey_framework-0.0.0-py3-none-any.whl").write_text("")
    env = stubbed_env(tmp_path, {"pip": "#!/bin/sh\nexit 1\n"})
    return run_body(body.replace("/tmp/ctx", str(ctx)), env), ctx


def primer_pip_argv(body: str, tmp_path: Path, env: dict[str, str]) -> list[str]:
    """The argv of the deps layer's framework install, as ``pip`` received it.

    ``/tmp/deps-ctx`` and the apt lists directory are rewritten to temp dirs,
    ``apt-get`` is a no-op and ``pip`` appends every invocation to a log and
    exits 0. *env* carries the build ARGs the body reads (``OSPREY_PIP_SPEC``
    or ``OSPREY_VERSION``, ``OSPREY_DEV``, ``OSPREY_PIP_PRE`` …).

    Returns the argv of the first ``pip install`` naming ``osprey-framework``.
    """
    rewritten, _ = deps_sandbox(body, tmp_path)
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
    calls = [shlex.split(line) for line in log.read_text().splitlines()]
    primers = [c for c in calls if "install" in c and any("osprey-framework" in a for a in c)]
    assert primers, f"no framework install reached pip; calls: {calls}"
    return primers[0]
