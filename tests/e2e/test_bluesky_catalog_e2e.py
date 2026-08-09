"""Real-container e2e for the layered plan catalog (task 1.6, closing
Phase 1 of the plan-catalog epic).

Mocked-client tests (``tests/services/bluesky_bridge/test_plan_loader_layered.py``,
``test_exemplar_plans.py``) only exercise OSPREY's half of the contract: that
the loader in this repo resolves layers and trust tiers correctly in-process.
They never prove that a *deployed* bridge container -- built from the shipped
image, reading its own filesystem layers -- actually serves the same catalog
over HTTP. This is the other half: it deploys a real bluesky-bridge container
and asserts the layered catalog (the shipped plans + an externally-injected
facility plan) is discoverable via ``GET /plans`` with correct
provenance/metadata, and that the browse-only surface around it is honest.

SCOPE, deliberately narrowed: this file proves DISCOVERY, not execution.
Execution has exactly one owner now -- ``tests/e2e/test_bluesky_queue_e2e.py``,
which deploys the whole queue stack (queueserver + Redis + Tiled + the Virtual
Accelerator) and drives real scans through arming, drain, abort and restart.
A facility-injected plan is a catalog entry like any other by the time it
reaches the queue, so re-proving execution here would mean standing up a second
VA-backed stack to re-test what that file already covers, at real wall-clock
cost. What this file uniquely proves is that a plan file dropped into a
facility layer is SERVED by a deployed container with the right provenance --
and that the deployment says plainly what it can and cannot do with it.

Uses the ``hello-world`` preset, whose ``control_system.type`` is ``mock``,
rather than the VA-backed stack ``tests/e2e/_orm_stack.py`` builds: the catalog
is connector-independent, so this skips the VA image's slow build entirely
(mirrors ``test_bluesky_deploy.py``'s identical rationale). A mock deployment
is BROWSE-ONLY, which is not a limitation here but part of the subject: plans
are discoverable and composable, the capability record says
``browse_only_connector`` and names the command that flips it, and the queue
refuses to hold work it could never run.

Container safety: every docker invocation below names an exact
container/image -- never a wildcard, never ``system prune``/``--volumes``.
Teardown goes through ``osprey deploy down``, matching every other e2e in
this directory.

Gating: needs Docker. Much lighter than the VA-backed e2e (no amd64
emulation) -- comparable to ``test_bluesky_deploy.py``'s build+deploy time.
Advisory CI lane (see ci.yml's ``bluesky-catalog-e2e`` job); run locally with
``E2E_REUSE_IMAGES=1`` set for fast iteration once the image cache is warm.
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

from osprey.services.bluesky_bridge.queue_backend import REASON_BROWSE_ONLY_CONNECTOR
from tests.e2e import _orm_stack

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

# Distinct from the sibling e2e modules' pinned ports (_orm_stack.py's 18102,
# test_bluesky_deploy.py's 18090, test_va_substrate_equivalence.py's 18099,
# test_tiled_roundtrip.py's 18101) so all five can run concurrently on a
# shared dev machine without a port collision.
BRIDGE_PORT = 18103
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 600
HEALTH_TIMEOUT_SEC = 120.0

# An ordinary, well-formed facility-tier plan. Its job here is to be FOUND --
# correct provenance, correct metadata, correct schema over HTTP -- so nothing
# below resolves its device names against a live worker.
_FACILITY_PLAN_SOURCE = '''"""Test-authored facility-tier plan for the layered plan catalog e2e
(tests/e2e/test_bluesky_catalog_e2e.py).

Not part of the shipped OSPREY package: this file is written to a throwaway
host directory and injected via `services.bluesky.plan_dir`
(BLUESKY_PLAN_DIRS), so the deployed bridge discovers it as a `facility`-tier
layer (plan_loader.py). Named `facility_probe` -- distinct from every
shipped/built-in plan name, so it never collides at the `GET /plans` merge.

Its parameter names are ordinary device-name strings; nothing here resolves
them against a live device, because this file proves DISCOVERY only (see the
module docstring). Execution of catalog plans -- facility-tier included -- is
`tests/e2e/test_bluesky_queue_e2e.py`'s subject.
"""

from __future__ import annotations

from typing import Any

from bluesky import plans as bp
from pydantic import BaseModel, Field, model_validator

PLAN_METADATA = {
    "name": "facility_probe",
    "description": "Probe scan: sweep one setpoint device, reading one detector at each point.",
    "category": "diagnostic",
    "required_devices": ["motor", "detector"],
    "writes": True,
}


class PARAMS(BaseModel):
    """Parameters for `facility_probe`: one setpoint swept over [start, stop]."""

    motor: str = Field(..., description="Setpoint device name to sweep.")
    detector: str = Field(..., description="Detector device name to read at each point.")
    start: float
    stop: float
    num: int = Field(..., ge=2, description="Number of evenly-spaced points.")

    @model_validator(mode="after")
    def _motor_and_detector_disjoint(self) -> "PARAMS":
        if self.motor == self.detector:
            raise ValueError(f"motor and detector must be distinct (got {self.motor!r} twice)")
        return self


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    """Wrap `bluesky.plans.scan`: move `motor` over `[start, stop]` in `num`
    steps, reading `detector` at each point."""
    motor = devices[params.motor]
    detector = devices[params.detector]
    return bp.scan([detector], motor, params.start, params.stop, num=params.num)
'''


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


def _get(path: str) -> tuple[int, Any]:
    req = urllib.request.Request(f"{BRIDGE_URL}{path}", method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _request(path: str, method: str, body: dict | None = None) -> tuple[int, Any]:
    """One request against the bridge, returning ``(status, parsed_body)``.

    Refusal bodies are what this module asserts on, so an ``HTTPError`` is a
    normal result here rather than an exception to propagate.
    """
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


@pytest.fixture(scope="module")
def deployed_catalog_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build + ``osprey deploy up --dev`` a bluesky-bridge project with one
    facility-injected plan file; tear down after.

    ``hello-world`` (mirrors ``test_bluesky_deploy.py``): no VA co-deploy, no
    LLM secret needed, no amd64-emulated image build. The preset's
    ``control_system.type`` is ``mock``, so this deployment is browse-only --
    which is part of what the tests below assert, not a gap in them. ``bluesky.plan_dir`` points at a throwaway
    host directory containing ``_FACILITY_PLAN_SOURCE`` -- the deploy wiring
    (Task 1.4) bind-mounts it read-only and sets ``BLUESKY_PLAN_DIRS``, so
    ``plan_loader.py`` scans it as a ``facility``-tier layer.
    """
    osprey_bin = _orm_stack.find_osprey_console_script()
    base = tmp_path_factory.mktemp("scan_catalog_build")
    plan_dir = tmp_path_factory.mktemp("scan_catalog_plans")
    (plan_dir / "facility_probe.py").write_text(_FACILITY_PLAN_SOURCE, encoding="utf-8")
    project_dir = base / "proj"

    build = subprocess.run(
        [
            str(osprey_bin),
            "build",
            "proj",
            "--preset",
            "hello-world",
            "--set",
            f"bluesky.port={BRIDGE_PORT}",
            "--set",
            f"bluesky.plan_dir={plan_dir}",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(base),
            "--force",
        ],
        cwd=str(base),
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_SEC,
        env={**os.environ, "CLAUDECODE": ""},
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    # Force a fresh --dev build so the deployed bridge runs CURRENT source
    # (osprey deploy up does not pass --build to compose, so it would
    # otherwise reuse a stale cached image). Exact-named image only.
    # E2E_REUSE_IMAGES=1 skips this for fast local iteration once the image
    # cache is warm; never set it in CI.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(
            ["docker", "rmi", "-f", _orm_stack.bridge_image("proj")], capture_output=True, text=True
        )

    try:
        up = subprocess.run(
            [str(osprey_bin), "deploy", "up", "-d", "--dev"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey deploy up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        yield project_dir
    finally:
        down = subprocess.run(
            [str(osprey_bin), "deploy", "down"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey deploy down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )


# ---------------------------------------------------------------------------
# Discovery: the layered catalog's provenance/metadata, over the real
# deployed HTTP API. Strict -- no @flaky -- since this is the core deliverable.
# ---------------------------------------------------------------------------


def test_plans_endpoint_shows_shipped_and_facility_provenance(
    deployed_catalog_stack: Path,
) -> None:
    """``GET /plans`` against the real container must show, in one response:

    - the shipped plans (``orm``, ``grid_scan``) with ``provenance ==
      "shipped"`` and non-null ``metadata`` (Task 1.5's in-image
      ``plans_core/`` files);
    - the externally-injected ``facility_probe`` plan with
      ``provenance == "facility"`` and its authored metadata round-tripped
      byte-for-byte through the loader's ``PLAN_METADATA`` parser.
    """
    status, plans = _get("/plans")
    assert status == 200, f"GET /plans failed: {status} {plans}"
    by_name = {p["name"]: p for p in plans}

    for shipped_name in ("orm", "grid_scan"):
        assert shipped_name in by_name, (
            f"{shipped_name!r} missing from GET /plans: {sorted(by_name)}"
        )
        entry = by_name[shipped_name]
        assert entry["provenance"] == "shipped", (
            f"{shipped_name!r}: expected provenance 'shipped', got {entry['provenance']!r}"
        )
        assert entry["metadata"] is not None, f"{shipped_name!r}: metadata is None"

    assert "facility_probe" in by_name, f"facility_probe missing from GET /plans: {sorted(by_name)}"
    facility_entry = by_name["facility_probe"]
    assert facility_entry["provenance"] == "facility", (
        "facility_probe: expected provenance 'facility' (injected via "
        f"services.bluesky.plan_dir/BLUESKY_PLAN_DIRS), got {facility_entry['provenance']!r}"
    )
    metadata = facility_entry["metadata"]
    assert metadata is not None, "facility_probe: metadata is None"
    assert metadata["name"] == "facility_probe"
    assert metadata["category"] == "diagnostic"
    assert metadata["required_devices"] == ["motor", "detector"]
    assert metadata["writes"] is True


# ---------------------------------------------------------------------------
# The browse-only surface around the catalog: composable, and honest about it.
#
# Replaces a launch->read round trip that used to run here on the removed
# demo-runner knob. Execution -- for facility-tier plans as much as any other
# -- now belongs to tests/e2e/test_bluesky_queue_e2e.py, which drives real
# scans against a real queue server. What is left here is the half that is
# genuinely about the catalog: a discovered plan can be composed, and the
# deployment tells the truth about what it will do with it.
# ---------------------------------------------------------------------------


def test_deployment_reports_browse_only_and_names_the_flip(
    deployed_catalog_stack: Path,
) -> None:
    """A mock deployment is HEALTHY and says plainly that it cannot execute.

    ``status: "ok"`` is deliberately independent of ``can_execute``: a
    browse-only deployment is a working deployment, and gating liveness on
    capability would flap the container healthcheck over a configuration doing
    exactly what it was told.
    """
    status, body = _get("/health")
    assert status == 200, f"GET /health failed: {status} {body}"
    assert body["status"] == "ok", f"a browse-only bridge must still be ok: {body}"

    capability = body["capability"]
    assert capability["can_execute"] is False, f"mock cannot execute plans: {capability}"
    assert capability["reason"] == REASON_BROWSE_ONLY_CONNECTOR, f"wrong reason: {capability}"
    assert "set-control-system virtual_accelerator" in capability["detail"], (
        f"the browse-only detail must name the command that flips it: {capability}"
    )


def test_a_facility_plan_is_composable_but_unqueueable(deployed_catalog_stack: Path) -> None:
    """The facility-injected plan reaches the draft, and stops at the queue.

    This is the discovery claim carried one step further than ``GET /plans``:
    the plan is not merely listed, its schema resolves well enough for the
    shared draft to accept real arguments for it -- which is what an operator
    or the agent would do first. The enqueue then refuses with the capability
    record attached, because a deployment that cannot execute must never HOLD
    queue items: an item sitting in a queue reads as work that will happen.

    Asserted on ``detail.code``; a status-code-only check would keep passing
    while the refusal body drifted.
    """
    status, patched = _request(
        "/draft",
        "PATCH",
        {
            "plan_name": "facility_probe",
            "plan_args_patch": {
                "motor": "motor1",
                "detector": "det1",
                "start": 0.0,
                "stop": 2.0,
                "num": 3,
            },
            "client_id": "catalog-e2e",
        },
    )
    assert status == 200, f"PATCH /draft failed for the facility plan: {status} {patched}"
    assert patched["plan_name"] == "facility_probe"

    status, refusal = _request("/queue/items", "POST", {"draft_revision": patched["revision"]})
    assert status == 409, f"a browse-only enqueue must be refused: {status} {refusal}"
    detail = refusal.get("detail") if isinstance(refusal, dict) else None
    assert isinstance(detail, dict) and detail.get("code") == REASON_BROWSE_ONLY_CONNECTOR, (
        f"wrong refusal code on a browse-only enqueue: {refusal}"
    )
    assert isinstance(detail.get("capability"), dict), (
        f"the refusal must carry the capability record the status surface publishes: {refusal}"
    )

    status, queue = _get("/queue")
    assert status == 200, f"GET /queue failed: {status} {queue}"
    assert queue["status"]["items_in_queue"] == 0, (
        f"a browse-only deployment is holding queue items: {queue}"
    )
