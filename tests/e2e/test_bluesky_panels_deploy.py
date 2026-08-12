"""Full-stack Docker integration test for the Phase-6 "Operator Interfaces"
scan panel (task 4.3, bluesky-panels-deploy-e2e) -- the gold-standard proof
that the turn-key tutorial stack (Virtual Accelerator + Bluesky bridge + the
queueserver RE Manager + Redis + co-deployed Tiled + the bluesky-panels
sidecar + its web panel) boots as real containers and drives a real scan end
to end THROUGH THE SIDECAR, exactly as a browser would.

That last clause is this module's subject and the thing no sibling e2e covers:
the browser never talks to the bridge. Every call a panel makes goes through
the sidecar's relay, which rebuilds request headers from scratch and attaches
the launch token IT resolved in-process -- so the operator flow must work with
the browser holding no credential at all, and no sidecar response may ever
contain one. ``tests/e2e/test_bluesky_queue_e2e.py`` proves the queue stack
itself against the bridge; this proves the hop in front of it.

Reuses ``tests/e2e/_orm_stack.py`` (the single source for FR11's VA-backed
turn-key deploy config): ``build_args``/``find_osprey_console_script`` build
the real project and ``select_correctors``/``select_bpms``/``write_scan_env``
wire the substrate device env from the *built* project's own
``data/channel_limits.json`` -- never a hardcoded preset channel.
``override_yaml()`` still pins ``control_system.type: virtual_accelerator``
explicitly even though the preset now defaults to it (a connector-mediated
scan only runs against a setpoint-tracking control system; the shipped
default is asserted, not assumed, in ``test_bluesky_queue_e2e.py``). The one
thing ``_orm_stack.build_args``/``build_project_subprocess`` don't parameterize
is the bluesky-panels sidecar's port, so this module calls ``override_yaml``/
``build_args``/``find_osprey_console_script`` directly (mirroring what
``build_project_subprocess`` does internally) and appends one extra
``--set bluesky_panels.port=...`` override.

Plan discovery (test 3/4's headline): ``GET /plans`` through the sidecar's
read-proxy is scanned for a plan whose ``metadata.writes`` is ``True`` (the
only plans that actually drive a device); each candidate's
``GET /plans/{name}/source`` is then checked for ``validated: true``. The
shipped catalog currently has exactly two such plans (``orm``, ``grid_scan``
-- see the project's "no bba/tune_scan" convention: only ``orm`` + n-d
``grid_scan`` ship), but NEITHER name is hardcoded here:
``_build_minimal_plan_args`` maps the winning candidate's JSON ``schema`` by
FIELD SHAPE (``correctors``+``detectors``+``span_a``+``num`` vs.
``axes``+``detectors``) onto the derived corrector/BPM device names, so a
future third exemplar plan is picked up automatically as long as it matches
one of those two shapes, and is otherwise skipped rather than crashing the
discovery loop. This keeps the coupling to the shipped plan catalog minimal
-- the catalog is drafts, not a fixed contract this test should pin by name.

Per-function flaky, NOT module-level (mirrors ``test_va_substrate_equivalence
.py``'s documented convention): tests 1-4 accept one rerun on a genuinely
flaky HTTP/timing assertion. The sidecar's negative write-surface proof
(test 5: nothing POSTable under ``/runs`` at all) stays STRICT with no flaky
marker at all -- a module-level ``flaky`` mark would silently sweep that
safety proof into lenient reruns, exactly the bug the va-substrate module
docstring warns about.

CONTAINER SAFETY: every docker/compose invocation below names an EXACT
resource (``<project>-bluesky-bridge``, ``<project>-virtual-accelerator``,
``<project>-bluesky-panels``, images ``<project>-bluesky-bridge:local``/
``<project>-va:local``/``<project>-bluesky-panels:local``) -- never ``system prune``,
never ``--volumes``, never a wildcard ``docker rm``/``rmi``. Teardown goes
through ``osprey deploy down`` (the shipped compose path).

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

from tests.e2e import _orm_stack, _queue_drive
from tests.e2e._deploy_diagnostics import queue_stack_logs

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

# ariel-postgres (5432) and OpenObserve (5080) are deployed unconditionally by
# the control-assistant preset with no profile knob to drop them, and a
# locally-running tutorial deploy routinely holds both. A bound-port collision
# aborts `deploy up` before the containers this proof needs ever start, so both
# move to high, unassigned ports. Appended to _orm_stack's override rather than
# added there: only this module needs them moved, and _orm_stack's shape is
# shared with the render gate.
POSTGRES_PORT = 25434
OPENOBSERVE_PORT = 25083

# The fixture builds/deploys under this project name; every compose template
# renders its container_name and locally-built image as ``<project>-<service>``,
# so derive them rather than hardcode host-global names that break the moment
# the templates are namespaced per-project.
PROJECT_NAME = "panels-proj"
BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"
VA_CONTAINER = f"{PROJECT_NAME}-virtual-accelerator"
BLUESKY_PANELS_CONTAINER = f"{PROJECT_NAME}-bluesky-panels"

# Image tags derived the same way the service compose templates do
# (``resolve_project_name`` -> ``<project>-<service>:local``).
BRIDGE_IMAGE = _orm_stack.bridge_image(PROJECT_NAME)
VA_IMAGE = _orm_stack.va_image(PROJECT_NAME)
BLUESKY_PANELS_IMAGE = _orm_stack.panels_image(PROJECT_NAME)

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
# The first-time native VA source build is slow (minutes); the sidecar +
# bridge + tiled all wait on it via depends_on/service_healthy chains.
DEPLOY_UP_TIMEOUT_SEC = 1200
HEALTH_TIMEOUT_SEC = 300.0
SCAN_TIMEOUT_SEC = 90.0

# Module-level pytestmark deliberately carries NO `flaky` marker -- see the
# module docstring's "Per-function flaky" note. `flaky` is applied per
# function below to tests 1-4 only.
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
    token = env.get("BLUESKY_LAUNCH_TOKEN")
    assert token, "BLUESKY_LAUNCH_TOKEN missing/empty in the project .env"
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
        headers["X-Launch-Token"] = token
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
        # orm-shaped: sweep one corrector over a small bounded
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
        # grid_scan-shaped: one axis (one corrector setpoint), one detector.
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
    ``True`` -- the only plans that actually drive a device.
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
    # The two port moves go INSIDE _orm_stack's `config:` block, which its
    # top-level `dispatch: null` line closes -- appending at the end of the
    # string would nest them under `dispatch` instead. Splicing ahead of that
    # line keeps the indentation right; the assert makes a change to
    # _orm_stack's shape fail loudly here rather than silently produce an
    # override that does nothing.
    extra_config = (
        f"  services.postgresql.port_host: {POSTGRES_PORT}\n"
        f"  services.openobserve.port: {OPENOBSERVE_PORT}\n"
    )
    override_yaml = _orm_stack.override_yaml()
    assert "dispatch: null\n" in override_yaml, (
        "_orm_stack.override_yaml() no longer ends its config block with "
        "`dispatch: null` -- the port-move splice below needs a new anchor"
    )
    override_path.write_text(
        override_yaml.replace("dispatch: null\n", extra_config + "dispatch: null\n"),
        encoding="utf-8",
    )

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
    # launch_token left unset: `osprey deploy up` mints BLUESKY_LAUNCH_TOKEN
    # for every deployed service that declares it, so there is nothing to
    # supply ourselves here.
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
        _wait_for_health(f"{BLUESKY_PANELS_URL}/health", HEALTH_TIMEOUT_SEC)
        # HTTP readiness is not enqueue readiness -- the worker namespace an
        # enqueue validates against exists only once the RE worker environment
        # is open, and the bridge opens that off the readiness path. See
        # `_queue_drive.wait_for_worker_environment`.
        try:
            _queue_drive.wait_for_worker_environment(BRIDGE_URL)
        except AssertionError as exc:
            pytest.fail(f"{exc}\n{queue_stack_logs(_orm_stack.project_prefix(PROJECT_NAME))}")

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
    """The panel bundle serves, including its two deprecated aliases.

    ``/bluesky``, ``/plan`` and ``/results`` are ONE bundle mounted three times
    (the sidecar drives all of them from a single ``_BLUESKY_PANEL_DIR``
    constant), so each alias must serve identical bytes -- an alias that
    drifted into serving something else would be worse than a 404.
    """
    bodies: dict[str, str] = {}
    for path in ("/plan/", "/bluesky/", "/results/"):
        status, body = _get_html(path)
        assert status == 200, f"GET {path} failed: {status}"
        assert "<html" in body.lower(), f"GET {path} did not return HTML: {body[:200]!r}"
        bodies[path] = body

    for alias in ("/results/", "/plan/"):
        assert bodies[alias] == bodies["/bluesky/"], (
            f"{alias} must be a pure alias of /bluesky, serving the identical bundle"
        )


# ---------------------------------------------------------------------------
# 3. HEADLINE: a scan driven entirely through the sidecar's relays
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_plan_via_sidecar_queue_completes(deployed_stack: DeployedStack) -> None:
    """The whole operator flow through the sidecar, with NO credential in hand.

    This is the browser's view: compose the draft, enqueue it at the pinned
    revision, arm the queue, watch the run, read the results -- every call to
    the sidecar, never to the bridge. The launch token is resolved by the
    sidecar in-process and attached to each queue WRITE; this test never sends
    it and asserts, on every single response it reads, that the token has not
    come back out.
    """
    token = _minted_token(deployed_stack.project_dir)

    def _assert_no_token(label: str, payload: Any) -> None:
        assert token not in json.dumps(payload), f"launch token leaked into the sidecar {label}"

    # --- compose the draft through the sidecar's draft relay ---------------
    # Clear the shared draft first: a PATCH that re-states the current content
    # is a no-op that does NOT bump the revision, so a rerun (or a sibling test
    # staging the same plan) would pin an already-consumed revision and the
    # enqueue below would 409 draft_revision_already_launched.
    _request(BLUESKY_PANELS_URL, "/draft?client_id=panels-deploy-e2e", "DELETE")
    status, patched = _request(
        BLUESKY_PANELS_URL,
        "/draft",
        "PATCH",
        {
            "plan_name": deployed_stack.plan_name,
            "plan_args_patch": deployed_stack.plan_args,
            "client_id": "panels-deploy-e2e",
        },
    )
    assert status == 200, f"PATCH /draft (via sidecar) failed: {status} {patched}"
    _assert_no_token("draft response", patched)

    # --- enqueue at the pinned revision ------------------------------------
    status, enqueued = _sidecar_post("/queue/items", {"draft_revision": patched["revision"]})
    assert status == 200, f"POST /queue/items (via sidecar) failed: {status} {enqueued}"
    _assert_no_token("enqueue response", enqueued)
    run_id = enqueued.get("run_id")
    assert run_id, f"no run_id in the enqueue response: {enqueued}"

    # --- arm the queue: the sidecar supplies the token we never sent -------
    status, started = _sidecar_post("/queue/start", {})
    assert status == 200, (
        "the sidecar must attach the launch token it resolved in-process, so a "
        f"browser with no credential can still arm the queue: {status} {started}"
    )
    _assert_no_token("queue-start response", started)

    # --- watch it run, through the sidecar ---------------------------------
    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    last_status_body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _, queue_body = _sidecar_get("/queue")
        _assert_no_token("queue snapshot", queue_body)
        _, last_status_body = _sidecar_get(f"/runs/{run_id}")
        _assert_no_token("run-status response", last_status_body)
        if last_status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(1.0)

    assert last_status_body.get("status") == "completed", (
        f"scan driven through the sidecar did not complete: {last_status_body}"
    )

    ds, data = _sidecar_get(f"/runs/{run_id}/data")
    assert ds == 200, f"GET /runs/{run_id}/data (via sidecar) failed: {ds} {data}"
    _assert_no_token("data response", data)
    assert data.get("row_count", 0) > 0, f"expected real rows: {data}"


# ---------------------------------------------------------------------------
# 4. Isolation: the same plan driven straight via the bridge
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_plan_direct_via_bridge(deployed_stack: DeployedStack) -> None:
    """The same flow driven straight at the bridge, bypassing the sidecar.

    The isolation control for test 3: when that one fails, this says whether
    the sidecar's relay broke or the stack underneath it did. Here the caller
    holds the launch token itself -- the bridge gates the arming action, and
    the sidecar's only job is to hold that credential on the browser's behalf.
    """
    token = _minted_token(deployed_stack.project_dir)

    # Clear the shared draft first — the sidecar test stages these same args,
    # and a same-content PATCH is a no-op that keeps the consumed revision
    # (409 draft_revision_already_launched on the enqueue otherwise).
    _request(BRIDGE_URL, "/draft?client_id=panels-deploy-e2e-direct", "DELETE")
    status, patched = _request(
        BRIDGE_URL,
        "/draft",
        "PATCH",
        {
            "plan_name": deployed_stack.plan_name,
            "plan_args_patch": deployed_stack.plan_args,
            "client_id": "panels-deploy-e2e-direct",
        },
    )
    assert status == 200, f"PATCH /draft (bridge) failed: {status} {patched}"

    status, body = _bridge_post("/queue/items", {"draft_revision": patched["revision"]})
    assert status == 200, f"POST /queue/items (bridge) failed: {status} {body}"
    run_id = body["run_id"]

    status, body = _bridge_post("/queue/start", {}, token=token)
    assert status == 200, f"POST /queue/start (bridge) failed: {status} {body}"

    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    last_status_body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _, last_status_body = _bridge_get(f"/runs/{run_id}")
        if last_status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(1.0)

    assert last_status_body.get("status") == "completed", (
        f"scan enqueued directly via the bridge did not complete: {last_status_body}"
    )

    status, data = _bridge_get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data (bridge) failed: {status} {data}"
    assert data.get("row_count", 0) > 0, f"expected real rows: {data}"


# ---------------------------------------------------------------------------
# 5. NEGATIVE (strict, no flaky): the /runs surface is READ-ONLY on the sidecar
# ---------------------------------------------------------------------------
# Container-neutral: touches no container state, so it can run in whichever
# order pytest schedules it without affecting the scan tests above.


def test_sidecar_runs_surface_is_read_only(deployed_stack: DeployedStack) -> None:
    """Nothing under ``/runs`` is POSTable on the sidecar -- every write is a queue write.

    ``/runs`` used to carry the sidecar's one write (``POST /runs/launch``).
    That route is gone: execution is now the queue's, so the whole ``/runs``
    surface is the read-proxy's and nothing else. Proving the ABSENCE of every
    verb here — rather than "the launch route is the only one" — is a stronger
    invariant AND a simpler one to keep true.

    Method-code semantics are load-bearing in the assertions below: a path
    registered for GET but not POST returns Starlette's 405, while a path
    registered for nothing at all returns 404. Asserting the right one of the
    two is what makes each check say what it means.
    """
    # No /runs/{id}/stop at all -- the path template isn't registered on the
    # sidecar for any method, so it 404s regardless of verb. (The halt lives on
    # the queue surface: POST /queue/stop and POST /queue/abort.)
    status, body = _request(BLUESKY_PANELS_URL, "/runs/not-a-real-run-id/stop", "GET")
    assert status == 404, f"expected no GET /runs/{{id}}/stop route, got {status}: {body}"
    status, body = _sidecar_post("/runs/not-a-real-run-id/stop", {})
    assert status == 404, f"expected no POST /runs/{{id}}/stop route, got {status}: {body}"

    # The retired direct-execute route must not be back. The PATH still
    # matches a registered template — GET /runs/{run_id} binds "launch" as a
    # run id — so the router answers wrong-verb 405 here, not 404. That 405 is
    # exactly the proof wanted: the path resolves, and no POST is registered
    # on it.
    status, body = _sidecar_post("/runs/launch", {"plan_name": "grid_scan", "plan_args": {}})
    assert status == 405, (
        f"POST /runs/launch is retired; execution goes through the queue: {status} {body}"
    )
    status, body = _sidecar_post("/runs/not-a-real-run-id/launch", {})
    assert status == 404, (
        f"POST /runs/{{id}}/launch is retired; execution goes through the queue: {status} {body}"
    )

    # Bare POST /runs (unbounded intent-create) is not exposed either --
    # GET /runs IS registered (read-proxy), so a POST there hits Starlette's
    # 405 for a matched path/wrong method, not a 404. That 405 is therefore
    # also the positive control for this whole test: it proves the /runs prefix
    # is genuinely mounted, so the 404s above are real absences rather than the
    # sidecar having no /runs routes at all.
    status, body = _sidecar_post("/runs", {"plan_name": "grid_scan", "plan_args": {}})
    assert status == 405, (
        f"sidecar must not expose POST /runs, and GET /runs must exist: {status}: {body}"
    )

    # ... while the queue WRITE surface is present and relaying. A malformed
    # body reaches the bridge and earns its 422, which a missing route could
    # not produce -- so this distinguishes "relay wired" from "relay absent"
    # without enqueuing anything.
    status, body = _sidecar_post("/queue/items", {"not_a_draft_revision": True})
    assert status == 422, (
        f"POST /queue/items must relay to the bridge (expected its 422 on a bad "
        f"body), got {status}: {body}"
    )
