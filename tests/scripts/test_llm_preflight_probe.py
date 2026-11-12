"""Tests for the gateway pre-flight probe every model-spending lane in ci.yml runs.

A fake ``curl`` on ``PATH`` records the URL it was asked for and answers with a
given HTTP code, so the base-URL handling and the red/green outcome are checked
without a network.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "llm_preflight_probe.sh"


def _run(
    tmp_path: Path, *, code: str = "200", key: str = "k", base: str = "https://gw.example/"
) -> tuple[int, str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    seen = tmp_path / "url"
    # Record the POST target, write the -o body, print the -w code.
    curl = f"""#!/usr/bin/env bash
out=
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift ;;
    -X) printf '%s' "$3" > "{seen}"; shift ;;
  esac
  shift
done
echo 'gateway says no' > "$out"
printf '%s' "{code}"
"""
    (bin_dir / "curl").write_text(curl)
    (bin_dir / "curl").chmod(0o755 | stat.S_IXUSR)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "ALS_APG_API_KEY": key,
        "ALS_APG_BASE_URL": base,
        "OSPREY_E2E_FORCE_MODEL": "some-model",
    }
    done = subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, check=False
    )
    return done.returncode, done.stdout, seen.read_text() if seen.exists() else ""


@pytest.mark.parametrize(
    "base", ["https://gw.example", "https://gw.example/", "https://gw.example/v1"]
)
def test_the_probe_posts_to_the_base_url_override(tmp_path: Path, base: str) -> None:
    code, out, url = _run(tmp_path, base=base)
    assert code == 0, out
    assert url == "https://gw.example/v1/messages"
    assert "reachable (HTTP 200)" in out


@pytest.mark.parametrize("http", ["401", "503", "000"])
def test_a_non_200_answer_reds_the_step_with_the_body(tmp_path: Path, http: str) -> None:
    code, out, _ = _run(tmp_path, code=http)
    assert code == 1
    assert f"probe failed (HTTP {http})" in out
    assert "gateway says no" in out


def test_a_missing_key_reds_before_any_request(tmp_path: Path) -> None:
    code, out, url = _run(tmp_path, key="")
    assert code == 1
    assert "ALS_APG_API_KEY secret is not set" in out
    assert url == ""
