"""Full-stack Docker integration test for the Bluesky scan bridge (task 2.14).

Mirrors ``tests/e2e/test_dispatch_deploy.py``'s shape (real ``osprey build`` +
``osprey deploy up --dev``, real container, real HTTP calls against the
shipped artifacts) but for the bluesky-bridge service: unlike the dispatch
worker, the bridge never shells out to an LLM, so this test needs no provider
API key — ``hello-world``'s bundled ``provider``/``model`` defaults are enough
for the build to succeed; nothing in the container talks to an LLM.

Uses the ``hello-world`` preset specifically (not ``control-assistant``):
control-assistant ships its own ``dispatch:`` block by default, which would
deploy the whole event-dispatcher/dispatch-worker stack (Node + Claude CLI
image, much slower to build) alongside bluesky for no reason — ``_inject_bluesky``
only *adds* to ``deployed_services``, it doesn't replace what a preset already
declares. ``hello-world`` declares no services of its own, so the bluesky
bridge ends up as the ONLY deployed service, exactly what this test needs.

Enables the bridge via ``--set bluesky.demo_runner=true`` (a plain CLI
override — no bundled preset ships ``bluesky:`` by default, since task 2.9's
generator is opt-in-via-profile, not default-on). ``demo_runner`` additionally
tells app.py's guarded startup hook (task 2.14a) to wire a real
``BlueskyPlanRunner`` against MOCK ophyd-async devices (``devices/mock.py``)
instead of the Phase 1 no-op ``FakePlanRunner`` — see that task for the actual
wiring; this test only asserts the end-to-end behavior it enables. Also pins
``bluesky.port`` (BRIDGE_PORT below) rather than relying on the 8090 default,
since this is a shared dev machine and 8090 collided with an unrelated
pre-existing process during development of this test.

Asserts, against the REAL deployed container:
  * the bridge binds to 127.0.0.1 (never 0.0.0.0) — ``docker port`` inspection.
  * BLUESKY_PROMOTE_TOKEN was minted into the project ``.env`` (task 2.10).
  * the built image contains the unreleased bluesky_bridge modules AND the
    bluesky-bridge extra (task 2.8's reviewer carry-forward) — NOT merely
    that the image builds, since a PyPI-based build would silently lack both
    and this must fail loudly rather than pass on stale code.
  * a demo ``count`` scan against mock devices (``det1``) promotes, runs to
    completion, and ``GET /runs/{id}/data`` returns the buffered rows.

CONTAINER SAFETY: every docker/podman invocation below names an exact
container/image — never a wildcard, never ``system prune``/``--volumes``.
Teardown goes through ``osprey deploy down`` (the shipped compose teardown
path), not a raw ``docker rm``/``rmi`` sweep.

Gating: needs Docker. Skipped entirely if unavailable. Lives in ``tests/e2e/``
(not ``tests/integration/``) so the fast lane (``pytest tests/ --ignore=tests/e2e``,
see ``ci_check.sh``/``premerge_check.sh``/ci.yml's ``test`` job) never collects
this ~6-minute real-container build+deploy; it runs instead in its own
``bluesky-deploy-e2e`` CI job (mirrors ``dispatch-deploy-e2e``'s setup, minus the
LLM secret/pre-flight probe this test doesn't need).
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

import pytest

# Deliberately NOT the bluesky-bridge default (8090): this is a shared dev
# machine with other long-running services, and 8090 was observed colliding
# with an unrelated pre-existing process during development of this test.
# Pinned via --set bluesky.port=... below rather than relying on the default.
BRIDGE_PORT = 18090
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"
BRIDGE_IMAGE = "osprey-bluesky-bridge:local"
# The fixture builds/deploys under this project name; the compose template
# renders the bridge container_name as ``<project>-bluesky-bridge``
# (services/bluesky/docker-compose.yml.j2), so derive it rather than hardcode a
# host-global name that breaks the moment the template is namespaced per-project.
PROJECT_NAME = "proj"
BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"

DEPLOY_UP_TIMEOUT_SEC = 600
HEALTH_TIMEOUT_SEC = 120.0
SCAN_TIMEOUT_SEC = 60.0

# Modules that only exist in THIS worktree's unreleased code — proves the
# --dev build actually baked in the local wheel, not a stale PyPI release
# (task 2.8 reviewer's carry-forward: assert content, not just "it builds").
_BLUESKY_BRIDGE_ONLY_MODULES = (
    "osprey.services.bluesky_bridge.plan_runner_bluesky",
    "osprey.services.bluesky_bridge.devices.mock",
    "osprey.services.bluesky_bridge.plans",
)

# Rerun only on AssertionError, which is what the genuinely-flaky failures raise
# (the container-startup health-wait in the fixture, and the HTTP-timing checks
# in the test bodies). Deterministic setup errors — a failed `osprey build`, a
# failed `deploy up`, a bound-port collision — surface via ``pytest.fail()`` as
# ``Failed``, not ``AssertionError``, so they now fail fast instead of burning a
# retry on a ~6-minute rebuild that would deterministically fail again.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"]),
]


def _find_osprey_console_script() -> Path:
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


@pytest.fixture(scope="module")
def deployed_bridge(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build + ``osprey deploy up --dev`` a bluesky-bridge-enabled project; tear down after."""
    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("scan_deploy_build")
    project_dir = base / PROJECT_NAME

    build = _run(
        [
            str(osprey_bin),
            "build",
            PROJECT_NAME,
            "--preset",
            "hello-world",
            "--set",
            "bluesky.demo_runner=true",
            "--set",
            f"bluesky.port={BRIDGE_PORT}",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(base),
            "--force",
        ],
        cwd=base,
        timeout=300,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    # Force a fresh image build so the deployed bridge runs CURRENT source —
    # `osprey deploy up` does not pass --build to compose, so it would
    # otherwise silently reuse a stale cached osprey-bluesky-bridge:local.
    # Exact-named image only.
    subprocess.run(["docker", "rmi", "-f", BRIDGE_IMAGE], capture_output=True, text=True)

    try:
        up = _run(
            [str(osprey_bin), "deploy", "up", "-d", "--dev"],
            cwd=project_dir,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey deploy up --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        yield project_dir
    finally:
        down = _run([str(osprey_bin), "deploy", "down"], cwd=project_dir, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey deploy down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )


def _wait_for_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:  # noqa: S310 - localhost
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {url} (last: {last_err})")


def _minted_token(project_dir: Path) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_PROMOTE_TOKEN")
    assert token, "BLUESKY_PROMOTE_TOKEN missing/empty in the project .env"
    return token


def _get(path: str) -> tuple[int, dict]:
    req = urllib.request.Request(f"{BRIDGE_URL}{path}", method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(path: str, body: dict, headers: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        f"{BRIDGE_URL}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bridge_binds_loopback_only(deployed_bridge: Path) -> None:
    """`docker port` must show 127.0.0.1, never 0.0.0.0 (task 2.8's fail-closed bind)."""
    proc = subprocess.run(
        ["docker", "port", BRIDGE_CONTAINER],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"docker port failed: {proc.stderr}"
    assert "127.0.0.1" in proc.stdout, f"expected a 127.0.0.1 bind, got: {proc.stdout!r}"
    assert "0.0.0.0" not in proc.stdout, f"bridge must never bind 0.0.0.0: {proc.stdout!r}"


def test_promote_token_was_minted(deployed_bridge: Path) -> None:
    """task 2.10: an unset BLUESKY_PROMOTE_TOKEN is auto-generated into .env."""
    token = _minted_token(deployed_bridge)
    assert len(token) >= 40  # secrets.token_urlsafe(32) -> ~43 url-safe chars


def test_image_contains_unreleased_bluesky_bridge_modules(deployed_bridge: Path) -> None:
    """Carry-forward from the 2.8 reviewer: assert CONTENT, not just "it builds".

    A PyPI-based (non---dev) build would lack both the unreleased
    bluesky_bridge submodules and the bluesky-bridge extra on the current
    release — this would still "build" (pip just installs the released
    package), so a green build alone is not proof of anything. Import each
    module inside the running container to prove the --dev wheel + extra
    both landed.
    """
    failures: list[str] = []
    for module in _BLUESKY_BRIDGE_ONLY_MODULES:
        proc = subprocess.run(
            ["docker", "exec", BRIDGE_CONTAINER, "python", "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            failures.append(f"{module}: {proc.stderr.strip()}")
    assert not failures, (
        "built image is missing unreleased bluesky_bridge modules or the "
        "bluesky-bridge extra deps (bluesky/ophyd-async) — was it built "
        "with --dev?\n  " + "\n  ".join(failures)
    )


def test_demo_scan_against_mock_devices_completes(deployed_bridge: Path) -> None:
    """End-to-end: launch -> promote -> poll -> read_run_data, against the real container."""
    token = _minted_token(deployed_bridge)

    status, body = _post(
        "/runs", {"plan_name": "count", "plan_args": {"detectors": ["det1"], "num": 3}}
    )
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    status, body = _post(f"/runs/{run_id}/promote", {}, headers={"X-Promote-Token": token})
    assert status == 200, f"promote failed: {status} {body}"

    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    last_status_body: dict = {}
    while time.monotonic() < deadline:
        _, last_status_body = _get(f"/runs/{run_id}")
        if last_status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(1.0)

    assert last_status_body.get("status") == "completed", (
        f"demo scan did not complete: {last_status_body}"
    )

    status, data_body = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data_body}"
    assert data_body["row_count"] == 3, f"expected 3 buffered rows: {data_body}"
    assert len(data_body["rows"]) == 3
