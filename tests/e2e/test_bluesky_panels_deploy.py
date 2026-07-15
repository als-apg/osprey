"""Full-stack Docker integration test for the Phase-6 "Operator Interfaces"
scan panels (task 4.3, bluesky-panels-deploy-e2e) -- the gold-standard proof
that the turn-key tutorial stack (Virtual Accelerator + Bluesky bridge +
co-deployed Tiled + the bluesky-panels sidecar + its three web panels) boots as
real containers and drives a real scan end to end through the sidecar.

Reuses ``tests/e2e/_orm_stack.py`` (the single source for FR11's VA-backed
turn-key deploy config): ``override_yaml()`` flips the tutorial's default
``control_system.type: mock`` to ``virtual_accelerator`` (a connector-mediated
scan only runs against a setpoint-tracking control system) and sets
``execution.execution_method: container`` (so ``BLUESKY_PROMOTE_TOKEN`` mints
safely on ``osprey deploy up`` -- see ``container_lifecycle.py``'s
``_local_exec_arming_unsafe``); ``build_args``/``find_osprey_console_script``
build the real project; ``select_correctors``/``select_bpms``/
``write_scan_env`` wire the substrate device env from the *built* project's
own ``data/channel_limits.json`` -- never a hardcoded preset channel. The one
thing ``_orm_stack.build_args``/``build_project_subprocess`` don't parameterize
is the bluesky-panels sidecar's port, so this module calls ``override_yaml``/
``build_args``/``find_osprey_console_script`` directly (mirroring what
``build_project_subprocess`` does internally) and appends one extra
``--set bluesky_panels.port=...`` override.

Plan discovery (test 4/5's headline): ``GET /plans`` through the sidecar's
read-proxy is scanned for a plan whose ``metadata.writes`` is ``True`` (the
only plans that actually drive a device -- ``count``/``scan``/``grid_scan``/
``orm``, the v1 built-ins, carry no ``PLAN_METADATA`` at all and are excluded
by construction, not by name); each candidate's ``GET /plans/{name}/source``
is then checked for ``validated: true``. The shipped exemplar catalog
currently has two such plans (``response_matrix``, ``grid_scan_nd`` --
see the project's "no bba/tune_scan" convention: only ``response_matrix``(orm)
+ n-d ``grid_scan`` ship as exemplars), but NEITHER name is hardcoded here:
``_build_minimal_plan_args`` maps the winning candidate's JSON ``schema`` by
FIELD SHAPE (``correctors``+``detectors``+``span_a``+``num`` vs.
``axes``+``detectors``) onto the derived corrector/BPM device names, so a
future third exemplar plan is picked up automatically as long as it matches
one of those two shapes, and is otherwise skipped rather than crashing the
discovery loop. This keeps the coupling to the shipped plan catalog minimal
-- the catalog is drafts, not a fixed contract this test should pin by name.

Per-function flaky, NOT module-level (mirrors ``test_va_substrate_equivalence
.py``'s documented convention): tests 1-5 accept one rerun on a genuinely
flaky HTTP/timing assertion. Both the sidecar's negative write-surface proof
(test 7: no ``/stop``, no unbounded ``POST /runs``) and the container-mutating
VA-stop test (test 6) stay STRICT with no flaky marker at all -- a
module-level ``flaky`` mark would silently sweep that safety proof into
lenient reruns, exactly the bug the va-substrate module docstring warns about.
``test_va_stopped_degrades_health`` (6) is DEFINED after the negative test 7
so that, in pytest's definition order, the container-mutating VA-stop test
runs LAST; it restores the VA container in a ``finally`` so a failed assertion
never leaves the stack degraded for a later run.

CONTAINER SAFETY: every docker/compose invocation below names an EXACT
resource (``<project>-bluesky-bridge``, ``<project>-virtual-accelerator``,
``<project>-bluesky-panels``, images ``osprey-bluesky-bridge:local``/
``osprey-va:local``/``osprey-bluesky-panels:local``) -- never ``system prune``,
never ``--volumes``, never a wildcard ``docker rm``/``rmi``. Teardown goes
through ``osprey deploy down`` (the shipped compose path), and the one
mid-test ``docker stop``/``docker start`` (test 6) names the VA container
exactly, restoring it before the fixture's own teardown runs.

Gating: needs Docker; the VA image builds natively for the host arch (PyAT/
softioc compile from source on Apple Silicon -- slow on a cold image cache).
Lives in ``tests/e2e/`` (never collected by the fast lane -- see ``ci_check.
sh``/ci.yml's ``test`` job's ``--ignore=tests/e2e``); runs in its own
advisory ``bluesky-panels-deploy-e2e`` CI job (mirrors ``bluesky-deploy-e2e``'s
gating: same-repo PRs only, no secrets -- neither the bridge nor the sidecar
shells out to an LLM).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.e2e import _orm_stack

# Distinct from every sibling e2e module's pinned bridge port (_orm_stack.py's
# 18102, test_bluesky_deploy.py's 18090, test_va_substrate_equivalence.py's
# 18099, test_tiled_roundtrip.py's 18101, test_bluesky_catalog_e2e.py's 18103,
# test_bluesky_sandbox_escape_e2e.py's 18105) so all these can run concurrently
# on a shared dev machine without a port collision.
BRIDGE_PORT = 18106
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

# The bluesky-panels sidecar's default host port (8095) is shared with the
# tutorial's own default -- pin a distinct one so this e2e never collides
# with a locally-running tutorial deploy on the same host.
BLUESKY_PANELS_PORT = 18095
BLUESKY_PANELS_URL = f"http://localhost:{BLUESKY_PANELS_PORT}"

VA_CA_PORT = _orm_stack.VA_CA_PORT

BRIDGE_IMAGE = _orm_stack.BRIDGE_IMAGE
VA_IMAGE = _orm_stack.VA_IMAGE
BLUESKY_PANELS_IMAGE = "osprey-bluesky-panels:local"

# The fixture builds/deploys under this project name; every compose template
# renders its container_name as ``<project>-<service>``, so derive them
# rather than hardcode host-global names that break the moment the templates
# are namespaced per-project.
PROJECT_NAME = "panels-proj"
BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"
VA_CONTAINER = f"{PROJECT_NAME}-virtual-accelerator"
BLUESKY_PANELS_CONTAINER = f"{PROJECT_NAME}-bluesky-panels"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
# The first-time native VA source build is slow (minutes); the sidecar +
# bridge + tiled all wait on it via depends_on/service_healthy chains.
DEPLOY_UP_TIMEOUT_SEC = 1200
HEALTH_TIMEOUT_SEC = 300.0
ROLLUP_TIMEOUT_SEC = 180.0
SCAN_TIMEOUT_SEC = 90.0
VA_DEGRADE_TIMEOUT_SEC = 60.0

# Module-level pytestmark deliberately carries NO `flaky` marker -- see the
# module docstring's "Per-function flaky" note. `flaky` is applied per
# function below to tests 1-6 only.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]


# ---------------------------------------------------------------------------
# Build/deploy scaffold
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


def _channel_limits(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8"))


def _bounds(limits: dict[str, Any], address: str) -> tuple[float, float]:
    entry = limits[address]
    return float(entry["min_value"]), float(entry["max_value"])


def _minted_token(project_dir: Path) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_PROMOTE_TOKEN")
    assert token, "BLUESKY_PROMOTE_TOKEN missing/empty in the project .env"
    return token


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


def _wait_for_rollup(url: str, expected: str, timeout: float) -> dict[str, Any]:
    """Poll ``url`` (a ``/health/full`` endpoint) until its ``rollup`` field
    equals ``expected``, returning the last body. Never raises on a probe
    failure mid-poll -- only on the overall timeout."""
    deadline = time.monotonic() + timeout
    last_body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:  # noqa: S310 - localhost
                if resp.status == 200:
                    last_body = json.loads(resp.read().decode("utf-8"))
                    if last_body.get("rollup") == expected:
                        return last_body
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            pass
        time.sleep(1.0)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for rollup=={expected!r} at {url} "
        f"(last body: {last_body})"
    )


def _wait_for_rollup_not(url: str, unexpected: str, timeout: float) -> dict[str, Any]:
    """Poll ``url`` until its ``rollup`` field DIFFERS from ``unexpected``."""
    deadline = time.monotonic() + timeout
    last_body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:  # noqa: S310 - localhost
                if resp.status == 200:
                    last_body = json.loads(resp.read().decode("utf-8"))
                    if last_body.get("rollup") != unexpected:
                        return last_body
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            pass
        time.sleep(1.0)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for rollup != {unexpected!r} at {url} "
        f"(last body: {last_body})"
    )


def _request(
    base: str, path: str, method: str, body: dict[str, Any] | None = None
) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(  # noqa: S310
        f"{base}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except ValueError:
                return resp.status, raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw.decode("utf-8"))
        except ValueError:
            return exc.code, raw.decode("utf-8", errors="replace")


def _sidecar_get(path: str) -> tuple[int, Any]:
    return _request(BLUESKY_PANELS_URL, path, "GET")


def _sidecar_post(path: str, body: dict[str, Any]) -> tuple[int, Any]:
    return _request(BLUESKY_PANELS_URL, path, "POST", body)


def _bridge_get(path: str) -> tuple[int, Any]:
    return _request(BRIDGE_URL, path, "GET")


def _bridge_post(path: str, body: dict[str, Any], token: str | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Promote-Token"] = token
    req = urllib.request.Request(  # noqa: S310
        f"{BRIDGE_URL}{path}", data=data, method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _get_html(path: str) -> tuple[int, str]:
    req = urllib.request.Request(f"{BLUESKY_PANELS_URL}{path}", method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _docker_port(container: str) -> str:
    proc = subprocess.run(["docker", "port", container], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"docker port {container} failed: {proc.stderr}"
    return proc.stdout


# ---------------------------------------------------------------------------
# Plan discovery (headline test's minimal-args builder) -- SHAPE-driven, not
# name-driven: see the module docstring.
# ---------------------------------------------------------------------------


def _build_minimal_plan_args(
    schema: dict[str, Any],
    correctors: dict[str, tuple[str, str]],
    bpms: dict[str, str],
    limits: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a minimal, valid ``plan_args`` for ``schema`` (a plan's JSON
    Schema from ``GET /plans``) using the derived corrector/BPM device names.

    Returns ``None`` when ``schema``'s shape isn't one this helper recognizes
    -- the caller then tries the next discovered candidate rather than
    guessing at an unfamiliar plan's parameters.
    """
    props = schema.get("properties", {})
    corrector_names = list(correctors.keys())
    bpm_names = list(bpms.keys())

    if {"correctors", "detectors", "span_a", "num"} <= props.keys():
        # response_matrix-shaped: sweep one corrector over a small bounded
        # current range, reading the BPMs.
        if not corrector_names or not bpm_names:
            return None
        return {
            "correctors": corrector_names[:1],
            "detectors": bpm_names[: min(2, len(bpm_names))],
            "span_a": 1.0,
            "num": 3,
        }

    if {"axes", "detectors"} <= props.keys():
        # grid_scan_nd-shaped: one axis (one corrector setpoint), one detector.
        if not corrector_names or not bpm_names:
            return None
        axis_name = corrector_names[0]
        sp_address, _rb_address = correctors[axis_name]
        lo, hi = _bounds(limits, sp_address)
        start = lo + 0.25 * (hi - lo)
        stop = lo + 0.75 * (hi - lo)
        return {
            "detectors": bpm_names[:1],
            "axes": [
                {"setpoint": axis_name, "start": start, "stop": stop, "num_points": 2},
            ],
        }

    return None


def _discover_writes_plan(
    correctors: dict[str, tuple[str, str]], bpms: dict[str, str], limits: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Discover a validated, writes-capable plan via the sidecar's read-proxy
    and build minimal ``plan_args`` for it.

    Candidates are every ``GET /plans`` entry whose ``metadata.writes`` is
    ``True`` (the v1 built-ins -- ``count``/``scan``/``grid_scan``/``orm`` --
    carry no ``PLAN_METADATA`` at all and are excluded by construction).
    Sorted by name for a deterministic pick across runs. Each candidate is
    checked against ``GET /plans/{name}/source``'s ``validated`` flag before
    its schema is used -- never assumed.
    """
    status, plans = _sidecar_get("/plans")
    assert status == 200, f"GET /plans (via sidecar) failed: {status} {plans}"
    assert isinstance(plans, list) and plans, f"no plans registered: {plans!r}"

    candidates = sorted(
        (
            p
            for p in plans
            if isinstance(p.get("metadata"), dict) and p["metadata"].get("writes") is True
        ),
        key=lambda p: p["name"],
    )
    assert candidates, f"no writes-capable plan found among {[p.get('name') for p in plans]}"

    for plan in candidates:
        name = plan["name"]
        src_status, source = _sidecar_get(f"/plans/{name}/source")
        if src_status != 200 or not isinstance(source, dict) or source.get("validated") is not True:
            continue
        args = _build_minimal_plan_args(plan["schema"], correctors, bpms, limits)
        if args is not None:
            return name, args

    raise AssertionError(
        f"no validated, writes-capable plan with a recognized schema shape found "
        f"among candidates: {[p['name'] for p in candidates]}"
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


class DeployedStack:
    """Everything the tests need about the one deployed project."""

    def __init__(
        self,
        project_dir: Path,
        correctors: dict[str, tuple[str, str]],
        bpms: dict[str, str],
        plan_name: str,
        plan_args: dict[str, Any],
    ):
        self.project_dir = project_dir
        self.correctors = correctors
        self.bpms = bpms
        self.plan_name = plan_name
        self.plan_args = plan_args


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedStack]:
    osprey_bin = _orm_stack.find_osprey_console_script()
    base = tmp_path_factory.mktemp("bluesky_panels_build")
    project_dir = base / PROJECT_NAME

    override_path = base / "override.yml"
    override_path.write_text(_orm_stack.override_yaml(), encoding="utf-8")

    # _orm_stack.build_args()/build_project_subprocess() don't parameterize
    # the bluesky-panels sidecar's port, so build the arg list directly (mirrors
    # what build_project_subprocess does internally) and append one extra
    # --set for it.
    args = _orm_stack.build_args(
        PROJECT_NAME,
        override_path=override_path,
        output_dir=base,
        bridge_port=BRIDGE_PORT,
        va_port=VA_CA_PORT,
    )
    args += ["--set", f"bluesky_panels.port={BLUESKY_PANELS_PORT}"]

    build = _run([str(osprey_bin), "build", *args], cwd=base, timeout=BUILD_TIMEOUT_SEC)
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    limits = _channel_limits(project_dir)
    correctors = _orm_stack.select_correctors(limits, count=1)
    bpms = _orm_stack.select_bpms(limits, count=2)
    # promote_token left unset: execution.execution_method=container (from
    # override_yaml()) makes _local_exec_arming_unsafe False, so `osprey
    # deploy up` auto-mints BLUESKY_PROMOTE_TOKEN safely -- no need to supply
    # one ourselves (unlike test_va_substrate_equivalence.py's local-exec
    # posture, which deliberately gates auto-minting off).
    _orm_stack.write_scan_env(project_dir, correctors=correctors, bpms=bpms)

    # Force fresh --dev builds so the deployed containers run CURRENT source
    # (osprey deploy up does not pass --build to compose, so it would
    # otherwise reuse stale cached images). Exact-named images only.
    # E2E_REUSE_IMAGES=1 skips this (dev-only fast local iteration); never
    # set it in CI, where a source change must always rebuild.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        for image in (BRIDGE_IMAGE, VA_IMAGE, BLUESKY_PANELS_IMAGE):
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)

    try:
        up = _run(
            [str(osprey_bin), "deploy", "up", "-d", "--dev"],
            cwd=project_dir,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey deploy up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        _wait_for_rollup(f"{BLUESKY_PANELS_URL}/health/full", "ok", ROLLUP_TIMEOUT_SEC)

        plan_name, plan_args = _discover_writes_plan(correctors, bpms, limits)

        yield DeployedStack(
            project_dir=project_dir,
            correctors=correctors,
            bpms=bpms,
            plan_name=plan_name,
            plan_args=plan_args,
        )
    finally:
        down = _run([str(osprey_bin), "deploy", "down"], cwd=project_dir, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey deploy down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )


# ---------------------------------------------------------------------------
# 1. Loopback binding
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_stack_boots_and_binds_loopback(deployed_stack: DeployedStack) -> None:
    for container in (BRIDGE_CONTAINER, BLUESKY_PANELS_CONTAINER, VA_CONTAINER):
        ports = _docker_port(container)
        assert "127.0.0.1" in ports, f"{container}: expected a 127.0.0.1 bind, got: {ports!r}"
        assert "0.0.0.0" not in ports, f"{container}: must never bind 0.0.0.0: {ports!r}"


# ---------------------------------------------------------------------------
# 2. Panels served
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_panels_served_200(deployed_stack: DeployedStack) -> None:
    for path in ("/plan/", "/results/", "/health-panel/"):
        status, body = _get_html(path)
        assert status == 200, f"GET {path} failed: {status}"
        assert "<html" in body.lower(), f"GET {path} did not return HTML: {body[:200]!r}"


# ---------------------------------------------------------------------------
# 3. Health rollup
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_health_full_rollup_healthy(deployed_stack: DeployedStack) -> None:
    status, body = _sidecar_get("/health/full")
    assert status == 200, f"GET /health/full failed: {status} {body}"
    assert body.get("rollup") == "ok", f"expected an 'ok' rollup: {body}"

    services = {s["name"]: s for s in body.get("services", [])}
    for name in ("bridge", "tiled", "va_ioc"):
        assert name in services, f"missing {name!r} in /health/full services: {body}"
        assert services[name]["status"] == "ok", f"{name} not ok: {services[name]}"

    # The VA IOC is Channel Access (raw TCP), never HTTP -- its detail text
    # must say so, distinct from the bridge/tiled HTTP probes.
    va_detail = services["va_ioc"]["detail"]
    assert "tcp" in va_detail.lower(), f"va_ioc detail doesn't read as a TCP probe: {va_detail!r}"
    for name in ("bridge", "tiled"):
        detail = services[name]["detail"]
        assert "200" in detail, f"{name} detail doesn't read as an HTTP probe: {detail!r}"


# ---------------------------------------------------------------------------
# 4. HEADLINE: scan via the sidecar's execute route
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_plan_via_sidecar_execute_completes(deployed_stack: DeployedStack) -> None:
    token = _minted_token(deployed_stack.project_dir)

    status, body = _sidecar_post(
        "/runs/execute",
        {"plan_name": deployed_stack.plan_name, "plan_args": deployed_stack.plan_args},
    )
    assert status == 200, f"POST /runs/execute failed: {status} {body}"
    assert token not in json.dumps(body), "promote token leaked into the sidecar execute response"
    assert body.get("status") != "writes_not_armed", (
        f"execute route reports writes not armed (expected armed): {body}"
    )
    run_id = body.get("run_id")
    assert run_id, f"no run_id in execute response: {body}"
    assert body.get("status") in ("running", "started", "completed"), (
        f"unexpected execute status: {body}"
    )

    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    last_status_body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _, last_status_body = _sidecar_get(f"/runs/{run_id}")
        assert token not in json.dumps(last_status_body), (
            "promote token leaked into the sidecar run-status response"
        )
        if last_status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(1.0)

    assert last_status_body.get("status") == "completed", (
        f"scan launched via the sidecar did not complete: {last_status_body}"
    )

    ds, data = _sidecar_get(f"/runs/{run_id}/data")
    assert ds == 200, f"GET /runs/{run_id}/data (via sidecar) failed: {ds} {data}"
    assert token not in json.dumps(data), "promote token leaked into the sidecar data response"
    assert data.get("row_count", 0) > 0, f"expected real rows: {data}"


# ---------------------------------------------------------------------------
# 5. Isolation: the same plan driven straight via the bridge
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_plan_direct_via_bridge(deployed_stack: DeployedStack) -> None:
    token = _minted_token(deployed_stack.project_dir)

    status, body = _bridge_post(
        "/runs", {"plan_name": deployed_stack.plan_name, "plan_args": deployed_stack.plan_args}
    )
    assert status == 200, f"POST /runs (bridge) failed: {status} {body}"
    run_id = body["id"]

    status, body = _bridge_post(f"/runs/{run_id}/promote", {}, token=token)
    assert status == 200, f"promote (bridge) failed: {status} {body}"

    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    last_status_body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _, last_status_body = _bridge_get(f"/runs/{run_id}")
        if last_status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(1.0)

    assert last_status_body.get("status") == "completed", (
        f"scan launched directly via the bridge did not complete: {last_status_body}"
    )

    status, data = _bridge_get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data (bridge) failed: {status} {data}"
    assert data.get("row_count", 0) > 0, f"expected real rows: {data}"


# ---------------------------------------------------------------------------
# 7. NEGATIVE (strict, no flaky): no stop / no unbounded create surfaced
# ---------------------------------------------------------------------------
# Placed before test 6 so the VA-stop test runs last among the container-
# mutating tests, while this negative proof (container-neutral) can run in
# whichever order pytest schedules it without affecting anything else.


def test_sidecar_exposes_no_stop_or_unbounded_create_route(
    deployed_stack: DeployedStack,
) -> None:
    # No /runs/{id}/stop at all -- the path template isn't registered on the
    # sidecar for any method, so it 404s regardless of verb.
    status, body = _request(BLUESKY_PANELS_URL, "/runs/not-a-real-run-id/stop", "GET")
    assert status == 404, f"expected no GET /runs/{{id}}/stop route, got {status}: {body}"
    status, body = _sidecar_post("/runs/not-a-real-run-id/stop", {})
    assert status == 404, f"expected no POST /runs/{{id}}/stop route, got {status}: {body}"

    # Bare POST /runs (unbounded intent-create) must not be exposed either --
    # GET /runs is registered (read-proxy), so a POST there hits Starlette's
    # 405 for a matched path/wrong method, not a 404.
    status, body = _sidecar_post("/runs", {"plan_name": "count", "plan_args": {}})
    assert status == 405, (
        f"sidecar must not expose unbounded POST /runs (only /runs/execute), got {status}: {body}"
    )


# ---------------------------------------------------------------------------
# 6. VA stopped degrades health (LAST among container-mutating tests)
# ---------------------------------------------------------------------------


def test_va_stopped_degrades_health(deployed_stack: DeployedStack) -> None:
    try:
        stop = subprocess.run(
            ["docker", "stop", VA_CONTAINER], capture_output=True, text=True, timeout=30
        )
        assert stop.returncode == 0, f"docker stop {VA_CONTAINER} failed: {stop.stderr}"

        body = _wait_for_rollup_not(
            f"{BLUESKY_PANELS_URL}/health/full", "ok", VA_DEGRADE_TIMEOUT_SEC
        )
        services = {s["name"]: s for s in body.get("services", [])}
        assert services["va_ioc"]["status"] == "unhealthy", f"va_ioc did not degrade: {body}"
        assert services["bridge"]["status"] == "ok", f"bridge unexpectedly degraded: {body}"
        assert services["tiled"]["status"] == "ok", f"tiled unexpectedly degraded: {body}"
        assert body.get("rollup") == "unhealthy", f"rollup did not degrade to unhealthy: {body}"
    finally:
        start = subprocess.run(
            ["docker", "start", VA_CONTAINER], capture_output=True, text=True, timeout=30
        )
        if start.returncode != 0:
            print(  # noqa: T201 - surface restore issues in CI logs
                f"docker start {VA_CONTAINER} failed: {start.stderr}"
            )
        else:
            try:
                _wait_for_rollup(f"{BLUESKY_PANELS_URL}/health/full", "ok", HEALTH_TIMEOUT_SEC)
            except AssertionError as exc:
                print(f"VA restore health-wait did not observe an 'ok' rollup: {exc}")  # noqa: T201
