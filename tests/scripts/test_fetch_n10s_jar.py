"""Tests for the n10s jar fetch ci.yml's graph lanes run before their pytest step.

The fixture would SKIP the graph paradigm on a failed download, so the lanes
fetch the jar themselves and must red on a failure. Fake ``uv`` and ``curl`` on
``PATH`` stand in for the URL lookup and the download.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "fetch_n10s_jar.sh"
_URL = "https://example.invalid/releases/neosemantics-5.20.0.jar"


def _run(tmp_path: Path, *, curl_rc: int) -> tuple[int, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fakes = {
        "uv": f'echo "{_URL}"',
        # Write the -o target, then exit as told.
        "curl": f'while [ $# -gt 0 ]; do [ "$1" = -o ] && : > "$2"; shift; done; exit {curl_rc}',
    }
    for name, body in fakes.items():
        (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{body}\n")
        (bin_dir / name).chmod(0o755 | stat.S_IXUSR)
    github_env = tmp_path / "github_env"
    github_env.touch()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_ENV": str(github_env),
    }
    done = subprocess.run(["bash", str(_SCRIPT)], env=env, capture_output=True, check=False)
    return done.returncode, github_env.read_text()


def test_the_jar_path_reaches_the_later_steps(tmp_path: Path) -> None:
    code, exported = _run(tmp_path, curl_rc=0)
    assert code == 0
    assert exported == f"OSPREY_TEST_N10S_JAR={tmp_path / 'neosemantics-5.20.0.jar'}\n"


@pytest.mark.parametrize("curl_rc", [22, 28])
def test_a_failed_download_reds_the_step(tmp_path: Path, curl_rc: int) -> None:
    code, exported = _run(tmp_path, curl_rc=curl_rc)
    assert code == curl_rc
    assert exported == "", "a failed fetch must not hand a path to the fixture"
