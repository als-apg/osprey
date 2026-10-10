"""What ``offline_build_env`` promises the tests that run a real build."""

from __future__ import annotations

import os
import subprocess

import pytest

from tests._builds import install_osprey_into, uv_path

# Any request a subprocess makes goes to a port nothing listens on.
NO_ROUTE = dict.fromkeys(
    ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"),
    "http://127.0.0.1:9",
)
NO_ROUTE["NO_PROXY"] = NO_ROUTE["no_proxy"] = ""


@pytest.mark.slow
def test_a_build_installs_osprey_with_no_route_to_an_index(offline_build_env, tmp_path):
    if uv_path() is None:
        pytest.skip("uv is not installed; a build installs with pip, which has no cache-only mode")
    venv = tmp_path / "venv"
    installed = install_osprey_into(venv, {**os.environ, **offline_build_env, **NO_ROUTE})
    assert installed.returncode == 0, installed.stdout + installed.stderr
    probe = subprocess.run(
        [str(venv / "bin" / "python"), "-c", "import osprey"], capture_output=True, text=True
    )
    assert probe.returncode == 0, probe.stderr
