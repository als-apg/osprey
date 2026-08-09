"""Full-stack Docker proof of the MINIMUM bluesky deploy: bridge + RE manager
+ Redis, and nothing else.

Mirrors ``tests/e2e/test_dispatch_deploy.py``'s shape (real ``osprey build`` +
``osprey deploy up --dev``, real containers, real HTTP calls against the
shipped artifacts). Unlike the dispatch worker, nothing here shells out to an
LLM, so this test needs no provider API key — ``hello-world``'s bundled
``provider``/``model`` defaults are enough for the build to succeed.

**Scope, and how it differs from ``test_bluesky_queue_e2e.py``.** That module
is the full-stack proof: Virtual Accelerator, Tiled, the panels sidecar, real
scans driven to completion, aborts, restarts. THIS module deliberately deploys
the smallest thing the ``bluesky:`` profile block can produce — one bridge, one
``bluesky-queueserver`` RE Manager, one Redis — on the ``hello-world`` preset,
whose ``control_system.type`` is ``mock``. What that buys, and what nothing
else covers:

  * the queue stack boots at all with no VA, no Tiled and no sidecar to lean
    on (a missing ``depends_on`` guard or a template block that only renders
    when the VA is co-deployed would surface here first);
  * a BROWSE-ONLY deployment is a HEALTHY deployment — every container reaches
    ``healthy``, ``/health`` answers 200, and the capability record carries the
    honest ``can_execute: false`` / ``browse_only_connector`` rather than the
    service going red;
  * that deployment refuses to hold queue items it could never run.

``hello-world`` (not ``control-assistant``) is still the right preset for that:
control-assistant ships a ``dispatch:`` block, the VA, Tiled, the sidecar and
the web-terminal stack by default, all of which would be built and deployed for
no reason — ``_inject_bluesky`` only *adds* to ``deployed_services``, it does
not replace what a preset already declares. ``hello-world`` declares no
services of its own, so the bluesky stack ends up as the ONLY thing deployed.

The bluesky stack is opt-in via profile, so the build supplies a ``bluesky:``
block explicitly. It does so by pinning ``bluesky.port`` (BRIDGE_PORT below)
rather than by relying on the 8090 default, since this is a shared dev machine
and 8090 collided with an unrelated pre-existing process during development of
this test.

Asserts, against the REAL deployed containers:
  * the bridge binds to 127.0.0.1 (never 0.0.0.0) — ``docker port`` inspection.
  * BLUESKY_LAUNCH_TOKEN and the queueserver control-plane keypair were minted
    into the project ``.env``, and the manager's control socket answers.
  * the built image contains the unreleased bluesky modules — NOT merely that
    the image builds, since a PyPI-based build would silently lack them and
    this must fail loudly rather than pass on stale code. Importing them also
    exercises the bluesky/queueserver stack, which must be present for the
    manager's worker.
  * ``/health`` reports a browse-only capability, and enqueue is refused.

CONTAINER SAFETY: every docker/podman invocation below names an exact
container/image — never a wildcard, never ``system prune``/``--volumes``.
Teardown goes through ``osprey deploy down`` (the shipped compose teardown
path), not a raw ``docker rm``/``rmi`` sweep.

Gating: needs Docker. Skipped entirely if unavailable. Lives in ``tests/e2e/``
(not ``tests/integration/``) so the fast lane (``pytest tests/ --ignore=tests/e2e``,
see ``ci_check.sh``/``premerge_check.sh``/ci.yml's ``test`` job) never collects
this real-container build+deploy; it runs instead in its own
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
from typing import Any

import pytest

from osprey.deployment.compose_generator import resolve_project_name
from osprey.services.bluesky_bridge.queue_backend import REASON_BROWSE_ONLY_CONNECTOR

# Deliberately NOT the bluesky-bridge default (8090): this is a shared dev
# machine with other long-running services, and 8090 was observed colliding
# with an unrelated pre-existing process during development of this test.
# Pinned via --set bluesky.port=... below rather than relying on the default.
BRIDGE_PORT = 18090
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"
# ``hello-world`` deploys OpenObserve unconditionally (it is baked into
# ``deployed_services``, with no profile knob to drop it) on host port 5080,
# which a locally-running tutorial stack routinely holds — and a bound-port
# collision aborts `deploy up` before the bluesky containers it creates
# alongside it ever start. Moved to a high, unassigned port, same defensive
# convention as BRIDGE_PORT above. Nothing in this proof touches telemetry.
OPENOBSERVE_PORT = 25081
# The fixture builds/deploys under this project name; the compose template
# renders each container_name AND the bridge's locally-built image as
# ``<project>-<service>`` (services/bluesky/docker-compose.yml.j2), so
# derive them (via resolve_project_name, exactly as the template does) rather
# than hardcode host-global names that break the moment the template is
# namespaced per-project.
PROJECT_NAME = "proj"
BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"
QUEUESERVER_CONTAINER = f"{PROJECT_NAME}-bluesky-queueserver"
REDIS_CONTAINER = f"{PROJECT_NAME}-bluesky-redis"
BRIDGE_IMAGE = f"{resolve_project_name({'project_name': PROJECT_NAME})}-bluesky-bridge:local"

DEPLOY_UP_TIMEOUT_SEC = 900
HEALTH_TIMEOUT_SEC = 180.0
# The manager waits on Redis and imports the bluesky/ophyd stack before its
# control socket answers, hence the longer container-health budget.
CONTAINER_HEALTH_TIMEOUT_SEC = 180.0

# Modules that only exist in THIS worktree's unreleased code — proves the
# --dev build actually baked in the local wheel, not a stale PyPI release
# (task 2.8 reviewer's carry-forward: assert content, not just "it builds").
# ``queue_backend`` and ``qserver_startup`` are the queue stack's two halves:
# the bridge's client onto the RE manager, and the manager's own startup
# script. Importing them pulls in bluesky/ophyd-async AND
# bluesky-queueserver-api, all of which the deployed manager needs.
_BLUESKY_BRIDGE_ONLY_MODULES = (
    "osprey.services.bluesky_bridge.queue_backend",
    "osprey.services.bluesky_bridge.qserver_startup",
    "osprey.services.bluesky_bridge.devices.mock",
    "osprey.services.bluesky_bridge.plans",
)

# Rerun only on AssertionError, which is what the genuinely-flaky failures raise
# (the container-startup health-wait in the fixture, and the HTTP-timing checks
# in the test bodies). Deterministic setup errors — a failed `osprey build`, a
# failed `deploy up`, a bound-port collision — surface via ``pytest.fail()`` as
# ``Failed``, not ``AssertionError``, so they now fail fast instead of burning a
# retry on a multi-minute rebuild that would deterministically fail again.
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
    """Build + ``osprey deploy up --dev`` a bluesky-only project; tear down after."""
    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("scan_deploy_build")
    project_dir = base / PROJECT_NAME

    # Only top-level PROFILE keys go through `--set`; a `config.`-scoped key
    # needs an override file (a dotted `--set` would build a nested dict for
    # every segment and replace the whole `services:` block).
    override_path = base / "override.yml"
    override_path.write_text(
        f"config:\n  services.openobserve.port: {OPENOBSERVE_PORT}\n", encoding="utf-8"
    )

    build = _run(
        [
            str(osprey_bin),
            "build",
            PROJECT_NAME,
            "--preset",
            "hello-world",
            "--override",
            str(override_path),
            # The one --set is what opts this project into the bluesky stack at
            # all: a `bluesky:` profile block is what `_inject_bluesky` keys on.
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
    # otherwise silently reuse a stale cached <project>-bluesky-bridge:local.
    # Exact-named image only. (The queueserver deliberately has no `build:`
    # block and runs this same image, so one rmi covers both services.)
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


def _docker_inspect(container: str, fmt: str) -> str:
    proc = subprocess.run(
        ["docker", "inspect", "--format", fmt, container],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"docker inspect {container} failed: {proc.stderr}"
    return proc.stdout.strip()


def _wait_for_container_health(container: str, timeout: float) -> None:
    """Poll ``docker inspect .State.Health.Status`` until ``healthy``.

    The bridge's HTTP-readiness gate can pass while Docker still reports
    ``starting`` (the healthcheck runs only on its interval, after
    ``start_period``), so an instant equality assert would be racy — and the
    manager/Redis have no HTTP surface to wait on at all.
    """
    deadline = time.monotonic() + timeout
    last = "(no status yet)"
    while time.monotonic() < deadline:
        last = _docker_inspect(container, "{{.State.Health.Status}}")
        if last == "healthy":
            return
        time.sleep(2.0)
    raise AssertionError(
        f"{container} did not reach 'healthy' within {timeout:.0f}s (last status: {last!r})"
    )


def _env_value(project_dir: Path, key: str) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — nothing was minted"
    value = parse_dotenv_file(env_path).get(key)
    assert value, f"{key} missing/empty in the project .env"
    return value


def _get(path: str) -> tuple[int, Any]:
    req = urllib.request.Request(f"{BRIDGE_URL}{path}", method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _request(path: str, method: str, body: dict | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(  # noqa: S310
        f"{BRIDGE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _refusal_code(body: Any) -> Any:
    """The machine-readable code off a refusal body (``{"detail": {"code": ...}}``).

    Returns ``None`` for any other shape so an assertion reports the drifted
    body rather than a ``KeyError`` raised inside this helper.
    """
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail.get("code") if isinstance(detail, dict) else None


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


def test_queueserver_publishes_no_ports(deployed_bridge: Path) -> None:
    """The RE manager's control socket is never published to the host.

    Publishing it would put a second route to plan execution beside the
    bridge's launch-token gate, and that gate only means anything if it is the
    only way in. Redis, likewise, is on the internal network with no ports.
    """
    for container in (QUEUESERVER_CONTAINER, REDIS_CONTAINER):
        proc = subprocess.run(
            ["docker", "port", container], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0, f"docker port {container} failed: {proc.stderr}"
        assert not proc.stdout.strip(), (
            f"{container} publishes host ports it must not: {proc.stdout!r}"
        )


def test_queue_stack_containers_reach_healthy(deployed_bridge: Path) -> None:
    """Bridge, RE manager and Redis all reach ``healthy`` on a browse-only deploy.

    The manager's healthcheck is ``qserver ping``, which round-trips its own
    control socket and succeeds with the RE worker environment CLOSED — the
    correct reading here, where the environment is never opened because nothing
    can execute.
    """
    for container in (BRIDGE_CONTAINER, QUEUESERVER_CONTAINER, REDIS_CONTAINER):
        _wait_for_container_health(container, CONTAINER_HEALTH_TIMEOUT_SEC)


def test_launch_token_and_control_plane_keypair_were_minted(deployed_bridge: Path) -> None:
    """The deploy generates its own secrets.

    The launch token arms the queue; the CurveZMQ pair is what keeps the RE
    manager's control socket from being plaintext (the compose template's
    ``:?`` guards would abort the deploy outright if either half were missing,
    so a successful deploy already implies they exist — this asserts their
    SHAPE, which the guards cannot).
    """
    token = _env_value(deployed_bridge, "BLUESKY_LAUNCH_TOKEN")
    assert len(token) >= 40  # secrets.token_urlsafe(32) -> ~43 url-safe chars

    private_key = _env_value(deployed_bridge, "BLUESKY_QSERVER_ZMQ_PRIVATE_KEY")
    public_key = _env_value(deployed_bridge, "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY")
    # Z85-encoded 32-byte CurveZMQ keys are 40 characters.
    assert len(private_key) == 40, f"unexpected control-plane private key shape: {len(private_key)}"
    assert len(public_key) == 40, f"unexpected control-plane public key shape: {len(public_key)}"
    assert private_key != public_key


def test_image_contains_unreleased_bluesky_bridge_modules(deployed_bridge: Path) -> None:
    """Carry-forward from the 2.8 reviewer: assert CONTENT, not just "it builds".

    A PyPI-based (non---dev) build would lack the unreleased bluesky_bridge
    submodules — this would still "build" (pip just installs the released
    package), so a green build alone is not proof of anything. Import each
    module inside the running container to prove the --dev wheel landed; the
    imports also pull in the bluesky stack and bluesky-queueserver-api (both
    core dependencies), proving they are present for the RE manager's worker.
    """
    failures: list[str] = []
    for module in _BLUESKY_BRIDGE_ONLY_MODULES:
        proc = subprocess.run(
            ["docker", "exec", BRIDGE_CONTAINER, "python", "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            failures.append(f"{module}: {proc.stderr.strip()}")
    assert not failures, (
        "built image is missing unreleased bluesky_bridge modules or the "
        "bluesky/queueserver stack — was it built "
        "with --dev?\n  " + "\n  ".join(failures)
    )


def test_browse_only_deployment_is_healthy_and_says_so(deployed_bridge: Path) -> None:
    """``hello-world`` deploys on the mock connector: healthy, and honestly browse-only.

    ``status: "ok"`` is deliberately independent of ``can_execute`` — a
    deployment that cannot move hardware is not a broken deployment, and gating
    liveness on capability would make the container healthcheck flap for a
    configuration that is working exactly as configured.
    """
    status, body = _get("/health")
    assert status == 200, f"GET /health failed: {status} {body}"
    assert body["status"] == "ok", f"a browse-only bridge must still be ok: {body}"

    capability = body["capability"]
    assert capability["can_execute"] is False, (
        f"the mock connector cannot execute plans: {capability}"
    )
    assert capability["reason"] == REASON_BROWSE_ONLY_CONNECTOR, f"wrong reason: {capability}"
    assert "set-control-system virtual_accelerator" in capability["detail"], (
        f"the browse-only detail must name the command that flips it: {capability}"
    )


def test_plans_are_browsable_but_unqueueable(deployed_bridge: Path) -> None:
    """The catalog browses; the queue refuses to hold what it could never run.

    This is what "browse-only" means as a behavior rather than a label: plans
    can be listed and composed into the shared draft, and the enqueue that
    would turn a composed plan into pending work is refused with the capability
    record attached. Asserted on ``detail.code`` — a status-code-only check
    would keep passing while the refusal body drifted.
    """
    status, plans = _get("/plans")
    assert status == 200, f"GET /plans failed: {status} {plans}"
    assert any(p["name"] == "grid_scan" for p in plans), (
        f"the shipped catalog must be browsable here: {[p.get('name') for p in plans]}"
    )

    status, patched = _request(
        "/draft",
        "PATCH",
        {
            "plan_name": "grid_scan",
            "plan_args_patch": {
                "detectors": ["det1"],
                "axes": [{"setpoint": "motor1", "start": 0.0, "stop": 1.0, "num_points": 3}],
            },
            "client_id": "bluesky-deploy-e2e",
        },
    )
    assert status == 200, f"PATCH /draft failed: {status} {patched}"

    status, refusal = _request("/queue/items", "POST", {"draft_revision": patched["revision"]})
    assert status == 409, f"a browse-only enqueue must be refused: {status} {refusal}"
    assert _refusal_code(refusal) == REASON_BROWSE_ONLY_CONNECTOR, (
        f"wrong refusal code on a browse-only enqueue: {refusal}"
    )

    status, queue = _get("/queue")
    assert status == 200, f"GET /queue failed: {status} {queue}"
    assert queue["status"]["items_in_queue"] == 0, (
        f"a browse-only deployment is holding queue items: {queue}"
    )
