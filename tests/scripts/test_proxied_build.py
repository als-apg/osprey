"""Tests for the proxied-build proof ci.yml's dockerfile-e2e runs.

The template Dockerfiles bridge a facility's uppercase proxy variables into the
lowercase ones apt reads, and nothing about that bridge shows in a build's exit
status on a runner with open egress. ``scripts/ci/proxied_build.sh`` therefore
asserts on the proxy's ACCESS LOG, and each test here pins a way its steps could
survive as green while proving nothing. Fake ``docker`` and ``podman`` on
``PATH`` record their calls; ``podman build`` prints what the test hands it.
"""

from __future__ import annotations

import os
import socket
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "proxied_build.sh"

#: Logs each call; ``podman build`` prints $FAKE_BUILD_OUTPUT and exits $FAKE_BUILD_RC,
#: ``podman pull`` exits $FAKE_PULL_RC.
_FAKE_ENGINE = """#!/usr/bin/env bash
echo "${0##*/} $*" >> "$FAKE_ENGINE_LOG"
case "${0##*/} $1" in
  "podman build") printf '%s\\n' "${FAKE_BUILD_OUTPUT:-}"; exit "${FAKE_BUILD_RC:-0}" ;;
  "podman pull") exit "${FAKE_PULL_RC:-0}" ;;
esac
exit 0
"""


def _run(
    tmp_path: Path, subcommand: str, **env_vars: str
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for engine in ("docker", "podman"):
        (bin_dir / engine).write_text(_FAKE_ENGINE)
        (bin_dir / engine).chmod(0o755 | stat.S_IXUSR)
    log = tmp_path / "engine.log"
    log.write_text("")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_ENGINE_LOG": str(log),
        "OSPREY_PROXY_DIR": str(tmp_path / "proxy"),
        "PROBE_BASE_IMAGE": "busybox:test",
        "PROXY_BASE_IMAGE": "debian:test",
        **env_vars,
    }
    done = subprocess.run(
        ["bash", str(_SCRIPT), subcommand],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return done, [line.split() for line in log.read_text().splitlines()]


def _access_log(tmp_path: Path, *lines: str) -> None:
    (tmp_path / "proxy").mkdir(exist_ok=True)
    (tmp_path / "proxy" / "access.log").write_text("".join(f"{line}\n" for line in lines))


_APT = "CONNECT deb.debian.org:443 HTTP/1.1"
_PIP = "CONNECT files.pythonhosted.org:443 HTTP/1.1"


def test_the_script_runs_in_strict_mode() -> None:
    """These steps grep for evidence of absence, which is exactly the shape a
    swallowed pipeline status turns into a silent pass."""
    assert "\nset -euo pipefail\n" in _SCRIPT.read_text()


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ((_APT, _PIP), "both apt and pip traffic went through the proxy"),
        ((), "nothing was proxied at all"),
        # The apt half proves the BRIDGE: pip honours the uppercase names unaided.
        ((_PIP,), "no deb.debian.org request"),
        ((_APT,), "no pypi.org / files.pythonhosted.org request"),
    ],
)
def test_assert_reads_the_access_log(tmp_path: Path, lines: tuple[str, ...], message: str) -> None:
    _access_log(tmp_path, *lines)
    done, _ = _run(tmp_path, "assert")
    assert message in done.stdout, done.stdout
    assert (done.returncode == 0) is (len(lines) == 2)


def test_assert_fails_without_an_access_log(tmp_path: Path) -> None:
    done, calls = _run(tmp_path, "assert")
    assert done.returncode == 1
    assert ["docker", "logs", "osprey-ci-proxy"] in calls


def test_probe_passes_when_only_the_uppercase_value_reaches_run(tmp_path: Path) -> None:
    done, calls = _run(tmp_path, "probe", FAKE_BUILD_OUTPUT="HTTP_PROXY=http://127.0.0.1:8888")
    assert done.returncode == 0, done.stdout
    build = next(call for call in calls if call[:2] == ["podman", "build"])
    assert "--http-proxy=false" in build and "--pull=never" in build


@pytest.mark.parametrize(
    ("output", "rc", "message"),
    [
        # A runtime that supplies the lowercase name itself would satisfy the log
        # assertion with the Dockerfile's bridge deleted.
        (
            "HTTP_PROXY=http://127.0.0.1:8888\nhttp_proxy=http://127.0.0.1:8888",
            "0",
            "injects a lowercase proxy variable",
        ),
        ("NO-PROXY-ENV", "0", "never reached the RUN environment"),
        ("HTTP_PROXY=http://127.0.0.1:8888", "1", "the probe build failed (podman exit 1)"),
    ],
)
def test_probe_fails(tmp_path: Path, output: str, rc: str, message: str) -> None:
    done, _ = _run(tmp_path, "probe", FAKE_BUILD_OUTPUT=output, FAKE_BUILD_RC=rc)
    assert done.returncode == 1
    assert message in done.stdout, done.stdout


def test_probe_never_retries_the_build_itself(tmp_path: Path) -> None:
    """Only the pull is retried: the property under test is never retried into passing."""
    done, calls = _run(tmp_path, "probe", FAKE_BUILD_OUTPUT="NO-PROXY-ENV")
    assert done.returncode == 1
    assert [call[:2] for call in calls].count(["podman", "build"]) == 1


def test_build_passes_only_the_uppercase_proxy_args_to_podman(tmp_path: Path) -> None:
    """Uppercase in and nothing else, under podman: the lowercase spelling is what
    the Dockerfile must derive, and BuildKit would derive it for the Dockerfile."""
    done, calls = _run(tmp_path, "build", OSPREY_VERSION="9.9.9")
    assert done.returncode == 0, done.stderr
    assert [call[0] for call in calls] == ["podman"], "the build must run under podman only"
    build = calls[0]
    args = {build[i + 1] for i, arg in enumerate(build) if arg == "--build-arg"}
    assert args == {
        "HTTP_PROXY=http://127.0.0.1:8888",
        "HTTPS_PROXY=http://127.0.0.1:8888",
        # Required by the Dockerfile; OSPREY_DEV=1 lets an unreleased pin fall back.
        "OSPREY_VERSION=9.9.9",
        "OSPREY_DEV=1",
    }
    assert "--http-proxy=false" in build, "buildah would copy the host's proxy variables in"


def test_start_writes_an_info_level_config_and_waits_for_the_port(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = str(listener.getsockname()[1])
        done, calls = _run(tmp_path, "start", OSPREY_PROXY_PORT=port)
    assert done.returncode == 0, done.stdout
    conf = (tmp_path / "proxy" / "tinyproxy.conf").read_text().splitlines()
    # The request lines the assertion reads are logged at Info.
    assert {f"Port {port}", "LogLevel Info", 'LogFile "/proxy/access.log"'} <= set(conf)
    assert (tmp_path / "proxy" / "access.log").read_text() == ""
    run = calls[0]
    assert run[:6] == ["docker", "run", "-d", "--name", "osprey-ci-proxy", "--network"]
    assert f"{tmp_path / 'proxy'}:/proxy" in run and "debian:test" in run


def test_stop_never_fails(tmp_path: Path) -> None:
    done, calls = _run(tmp_path, "stop")
    assert done.returncode == 0
    assert calls == [["docker", "rm", "-f", "osprey-ci-proxy"]]
