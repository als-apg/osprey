"""Full-stack Docker proof of the optimizer deploy: the optimizer API and its
Redis job store, and nothing else.

Mirrors ``tests/e2e/test_bluesky_deploy.py``'s shape: a real ``osprey init`` +
``osprey build --dev`` + ``osprey up --dev`` on the ``hello-world`` preset, real
containers, real HTTP calls against the shipped artifacts. ``hello-world``
declares no services of its own, so the optimizer stack is the only thing
deployed beside the preset's telemetry, and its ``mock`` control system means
no run is started here: this proves the deployment, not an optimization.

Asserts, against the REAL deployed containers:
  * the API binds to 127.0.0.1 only, and the job store publishes nothing;
  * both containers reach ``healthy``, and a route backed by the job store
    answers, so the API reaches Redis over the internal network;
  * OPTIMIZER_LAUNCH_TOKEN was minted into the repo's ``.env``;
  * the image is the optimizer's own: it carries the unreleased optimizer
    package and its search stack, a CPU-only torch, and NOT the framework;
  * root resolves to the optimizer's control identity (``osprey-optimizer``),
    so every write it makes names it.

CONTAINER SAFETY: every docker invocation below names an exact container or
image. Teardown goes through ``osprey down``, then exact-named removal of this
project's own volumes (``tests/e2e/_volumes.py``).

Gating: needs Docker; skipped entirely without it. Runs in its own
``optimizer-deploy-e2e`` CI job, never in the fast lane.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.deployment.compose_generator import OPTIMIZER_CONTROL_IDENTITY, resolve_project_name
from tests.e2e._volumes import remove_project_volumes

# Pinned clear of the layout, as the bluesky deploy proof pins its bridge: a
# shared dev machine may hold the slot for something else.
OPTIMIZER_PORT = 18182
OPTIMIZER_URL = f"http://localhost:{OPTIMIZER_PORT}"
# A locally-running deployment routinely holds the default port block, and a
# bound-port collision aborts `up`. PORT_BASE moves this deployment's whole
# block off it, as tests/e2e/test_bluesky_deploy.py does.
PORT_BASE = 22400
# This module's own compose project name: two modules sharing one would adopt
# and tear down each other's containers.
PROJECT_NAME = "osprey-e2e-optimizer-deploy"
_PROJECT = resolve_project_name({"project_name": PROJECT_NAME})
OPTIMIZER_CONTAINER = f"{PROJECT_NAME}-optimizer"
REDIS_CONTAINER = f"{PROJECT_NAME}-optimizer-redis"
OPTIMIZER_IMAGE = f"{_PROJECT}-optimizer:local"

DEPLOY_UP_TIMEOUT_SEC = 1500
HEALTH_TIMEOUT_SEC = 240.0
CONTAINER_HEALTH_TIMEOUT_SEC = 240.0

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"]),
]


def _osprey() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


def _require(proc: subprocess.CompletedProcess, what: str) -> None:
    if proc.returncode != 0:
        pytest.fail(
            f"{what} failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )


@pytest.fixture(scope="module")
def deployed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Init + build + ``osprey up --dev`` an optimizer-only repo; tear down after."""
    osprey_bin = _osprey()
    base = tmp_path_factory.mktemp("optimizer_deploy_build")
    repo = base / PROJECT_NAME

    _require(
        _run(
            [
                str(osprey_bin),
                "init",
                str(repo),
                "--preset",
                "hello-world",
                "--no-git",
                "--set",
                f"port_base={PORT_BASE}",
                # The `optimizer.port` edit opts the project into the optimizer:
                # an `optimizer:` profile block is what `_inject_optimizer` keys on.
                "--set",
                f"optimizer.port={OPTIMIZER_PORT}",
            ],
            cwd=base,
            timeout=300,
        ),
        "osprey init",
    )
    _require(
        _run(
            [
                str(osprey_bin),
                "build",
                "--repo",
                str(repo),
                "--skip-deps",
                "--skip-lifecycle",
                "--dev",
            ],
            cwd=base,
            timeout=300,
        ),
        "osprey build",
    )
    env_path = repo / ".env"
    if not env_path.exists():
        shutil.copy(repo / ".env.example", env_path)

    # A fresh image, so the deployed service runs CURRENT source rather than a
    # cached <project>-optimizer:local. Exact-named image only.
    subprocess.run(["docker", "rmi", "-f", OPTIMIZER_IMAGE], capture_output=True, text=True)

    try:
        _require(
            _run([str(osprey_bin), "up", "-d", "--dev"], cwd=repo, timeout=DEPLOY_UP_TIMEOUT_SEC),
            "osprey up --dev",
        )
        _wait_for_health(f"{OPTIMIZER_URL}/health", HEALTH_TIMEOUT_SEC)
        yield repo
    finally:
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
        remove_project_volumes(_PROJECT)


def _wait_for_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {url} (last: {last_err})")


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120)


def _wait_for_container_health(container: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "(no status yet)"
    while time.monotonic() < deadline:
        proc = _docker("inspect", "--format", "{{.State.Health.Status}}", container)
        assert proc.returncode == 0, f"docker inspect {container} failed: {proc.stderr}"
        last = proc.stdout.strip()
        if last == "healthy":
            return
        time.sleep(2.0)
    raise AssertionError(f"{container} did not reach 'healthy' within {timeout:.0f}s ({last!r})")


def _get(path: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(f"{OPTIMIZER_URL}{path}", timeout=15.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@pytest.mark.usefixtures("deployed")
def test_the_api_binds_loopback_only() -> None:
    proc = _docker("port", OPTIMIZER_CONTAINER)
    assert proc.returncode == 0, f"docker port failed: {proc.stderr}"
    assert "127.0.0.1" in proc.stdout, proc.stdout
    assert "0.0.0.0" not in proc.stdout, proc.stdout


@pytest.mark.usefixtures("deployed")
def test_the_job_store_publishes_no_ports() -> None:
    proc = _docker("port", REDIS_CONTAINER)
    assert proc.returncode == 0, f"docker port failed: {proc.stderr}"
    assert not proc.stdout.strip(), f"the job store publishes host ports: {proc.stdout!r}"


@pytest.mark.usefixtures("deployed")
def test_both_containers_reach_healthy() -> None:
    for container in (OPTIMIZER_CONTAINER, REDIS_CONTAINER):
        _wait_for_container_health(container, CONTAINER_HEALTH_TIMEOUT_SEC)


@pytest.mark.usefixtures("deployed")
def test_the_api_reaches_its_job_store() -> None:
    """``GET /jobs`` reads Redis and answers 503 when it cannot."""
    status, body = _get("/jobs")
    assert status == 200, body
    assert body == []


def test_the_launch_token_was_minted(deployed: Path) -> None:
    from osprey_connectors.dotenv import parse_dotenv_file

    assert parse_dotenv_file(deployed / ".env").get("OPTIMIZER_LAUNCH_TOKEN")


@pytest.mark.usefixtures("deployed")
def test_the_image_is_the_optimizers_own() -> None:
    """The service stack and a CPU-only torch are in it; the framework is not."""
    probe = (
        "import importlib.util, sys, torch, osprey_optimizer.api.main; "
        "sys.exit(1 if torch.version.cuda else 2 if importlib.util.find_spec('osprey') else 0)"
    )
    proc = _docker("run", "--rm", "--entrypoint", "python", OPTIMIZER_IMAGE, "-c", probe)
    assert proc.returncode != 1, "the image carries a CUDA torch build"
    assert proc.returncode != 2, "the image carries the osprey framework"
    assert proc.returncode == 0, proc.stderr


@pytest.mark.usefixtures("deployed")
def test_root_runs_as_the_optimizer_identity() -> None:
    proc = _docker("exec", OPTIMIZER_CONTAINER, "getent", "passwd", "0")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split(":", 1)[0] == OPTIMIZER_CONTROL_IDENTITY
