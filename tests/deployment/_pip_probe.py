"""Run a Dockerfile deps-layer RUN body against a recording ``pip`` stub.

The shipped image recipes install the pinned framework with plain ``pip`` in
one long ``RUN``. What that ``pip`` is invoked with is a property of the shell
body, not of any rendered string, so the tests here execute the body under a
real ``sh`` with ``pip`` and ``apt-get`` shadowed on ``PATH`` and read back the
argv the primer install was given.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def primer_pip_argv(body: str, tmp_path: Path, env: dict[str, str]) -> list[str]:
    """The argv of the deps layer's framework install, as ``pip`` received it.

    ``/tmp/deps-ctx`` and the apt lists directory are rewritten to temp dirs,
    ``apt-get`` is a no-op and ``pip`` appends every invocation to a log and
    exits 0. *env* carries the build ARGs the body reads (``OSPREY_PIP_SPEC``
    or ``OSPREY_VERSION``, ``OSPREY_DEV``, ``OSPREY_PIP_PRE`` …).

    Returns the argv of the first ``pip install`` naming ``osprey-framework``.
    """
    ctx = tmp_path / "deps-ctx"
    ctx.mkdir()
    (ctx / ".dockerignore").write_text("")
    apt_lists = tmp_path / "apt-lists"
    apt_lists.mkdir()
    log = tmp_path / "pip-calls.log"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for name, script in (
        ("apt-get", "#!/bin/sh\nexit 0\n"),
        ("pip", f'#!/bin/sh\nprintf "%s\\n" "$*" >> {shlex.quote(str(log))}\nexit 0\n'),
    ):
        stub = stub_bin / name
        stub.write_text(script)
        stub.chmod(0o755)
    run_env = dict(
        os.environ,
        PATH=f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        OSPREY_DEV="",
        PIP_NO_PROXY="",
    )
    run_env.update(env)
    rewritten = body.replace("/tmp/deps-ctx", str(ctx)).replace(
        "/var/lib/apt/lists", str(apt_lists)
    )
    result = subprocess.run(["sh", "-c", rewritten], capture_output=True, text=True, env=run_env)
    assert result.returncode == 0, (
        f"deps RUN failed under the stub:\n{result.stdout}\n{result.stderr}"
    )
    calls = [shlex.split(line) for line in log.read_text().splitlines()]
    primers = [c for c in calls if "install" in c and any("osprey-framework" in a for a in c)]
    assert primers, f"no framework install reached pip; calls: {calls}"
    return primers[0]
