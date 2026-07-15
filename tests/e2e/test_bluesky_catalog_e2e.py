"""Real-container e2e for the layered plan catalog (task 1.6, closing
Phase 1 of the plan-catalog epic).

Mocked-client tests (``tests/services/bluesky_bridge/test_plan_loader_layered.py``,
``test_exemplar_plans.py``) only exercise OSPREY's half of the contract: that
the loader in this repo resolves layers and trust tiers correctly in-process.
They never prove that a *deployed* bridge container -- built from the shipped
image, reading its own filesystem layers -- actually serves the same catalog
over HTTP. This is the other half: it deploys a real bluesky-bridge container
and asserts the layered catalog (shipped exemplars + an externally-injected
facility plan + the built-ins) is discoverable via ``GET /plans`` with correct
provenance/metadata, and that a facility-injected plan file is not just
discoverable but actually executable end to end (launch -> promote -> read).

Uses the ``hello-world`` preset with ``services.bluesky.demo_runner=true``
(real bluesky RunEngine, MOCK ophyd-async devices -- ``devices/mock.py``, no
EPICS/Virtual Accelerator at all) rather than the VA-backed stack
``tests/e2e/_orm_stack.py`` builds -- this test's point is the plan catalog,
not accelerator physics, so it skips the VA image's amd64-emulated build
entirely (mirrors ``test_scan_deploy.py``'s identical rationale). The one
facility plan (``facility_probe``, below) is authored against the demo
scanner's fixed mock device names (``motor1``/``det1``) so it can actually
run against this stack.

Container safety: every docker invocation below names an exact
container/image -- never a wildcard, never ``system prune``/``--volumes``.
Teardown goes through ``osprey deploy down``, matching every other e2e in
this directory.

Gating: needs Docker. Much lighter than the VA-backed e2e (no amd64
emulation) -- comparable to ``test_scan_deploy.py``'s build+deploy time.
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

from tests.e2e import _orm_stack

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

# Distinct from the sibling e2e modules' pinned ports (_orm_stack.py's 18102,
# test_scan_deploy.py's 18090, test_va_substrate_equivalence.py's 18099,
# test_tiled_roundtrip.py's 18101) so all five can run concurrently on a
# shared dev machine without a port collision.
BRIDGE_PORT = 18103
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 600
HEALTH_TIMEOUT_SEC = 120.0
SCAN_TIMEOUT_SEC = 60.0

# Authored against `devices/mock.py`'s `build_devices()` defaults -- the
# demo scanner's fixed single motor/detector pair -- so this plan can
# actually run against the deployed stack (no BLUESKY_EPICS_MOTORS/_DETECTORS
# wiring applies to the demo-scanner branch; see app.py's `_lifespan`).
_FACILITY_PLAN_SOURCE = '''"""Test-authored facility-tier plan for the layered plan catalog e2e
(tests/e2e/test_scan_catalog_e2e.py).

Not part of the shipped OSPREY package: this file is written to a throwaway
host directory and injected via `services.bluesky.plan_dir`
(BLUESKY_PLAN_DIRS), so the deployed bridge discovers it as a `facility`-tier
layer (plan_loader.py). Named `facility_probe` -- distinct from every
shipped/built-in plan name, so it never collides at the `GET /plans` merge.

Authored against the demo scanner's mock device names (`devices/mock.py`'s
`build_devices()` defaults: a single `motor1`/`det1` pair), since this e2e
deploys with `services.bluesky.demo_runner=true` (no EPICS/Virtual
Accelerator) -- the point is proving an externally-injected plan file is
discoverable AND executable, not exercising accelerator physics.
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


def _minted_token(project_dir: Path) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_PROMOTE_TOKEN")
    assert token, "BLUESKY_PROMOTE_TOKEN missing/empty in the project .env"
    return token


def _get(path: str) -> tuple[int, Any]:
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


@pytest.fixture(scope="module")
def deployed_catalog_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build + ``osprey deploy up --dev`` a demo-scanner bluesky-bridge
    project with one facility-injected plan file; tear down after.

    ``hello-world`` + ``bluesky.demo_runner=true`` (mirrors
    ``test_scan_deploy.py``): no VA co-deploy, no LLM secret needed, no
    amd64-emulated image build. ``bluesky.plan_dir`` points at a throwaway
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
            "bluesky.demo_runner=true",
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
            ["docker", "rmi", "-f", _orm_stack.BRIDGE_IMAGE], capture_output=True, text=True
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

    - the shipped exemplars (``response_matrix``, ``grid_scan_nd``) with
      ``provenance == "shipped"`` and non-null ``metadata`` (Task 1.5's
      in-image ``plans_core/`` files);
    - the externally-injected ``facility_probe`` plan with
      ``provenance == "facility"`` and its authored metadata round-tripped
      byte-for-byte through the loader's ``PLAN_METADATA`` parser;
    - the built-in plans (``count``/``scan``/``grid_scan``/``orm``) still
      present (``plans.py``'s hand-built ``PlanSpec`` set, merged in by
      ``app.py``'s ``/plans`` route regardless of the directory catalog).
    """
    status, plans = _get("/plans")
    assert status == 200, f"GET /plans failed: {status} {plans}"
    by_name = {p["name"]: p for p in plans}

    for shipped_name in ("response_matrix", "grid_scan_nd"):
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

    for builtin_name in ("count", "scan", "grid_scan", "orm"):
        assert builtin_name in by_name, (
            f"built-in plan {builtin_name!r} missing from GET /plans: {sorted(by_name)}"
        )


# ---------------------------------------------------------------------------
# Scan-drive: the facility-injected plan is not just discoverable, it runs.
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
def test_facility_probe_launch_promote_read_round_trip(deployed_catalog_stack: Path) -> None:
    """Drive ``facility_probe`` -- the facility-injected plan (not a
    built-in) -- through the full deployed launch -> promote -> read path.

    ``motor1``/``det1`` are the demo scanner's fixed mock device names
    (``devices/mock.py``'s ``build_devices()`` defaults), which is exactly
    what ``facility_probe`` was authored against.
    """
    token = _minted_token(deployed_catalog_stack)

    plan_args = {"motor": "motor1", "detector": "det1", "start": 0.0, "stop": 2.0, "num": 3}
    status, body = _post("/runs", {"plan_name": "facility_probe", "plan_args": plan_args})
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    status, body = _post(f"/runs/{run_id}/promote", {}, headers={"X-Promote-Token": token})
    assert status == 200, f"promote failed: {status} {body}"

    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    status_body: dict = {}
    while time.monotonic() < deadline:
        _, status_body = _get(f"/runs/{run_id}")
        if status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(0.5)
    assert status_body.get("status") == "completed", (
        f"facility_probe scan did not complete within {SCAN_TIMEOUT_SEC:.0f}s "
        f"(status={status_body})"
    )

    status, data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    expected_rows = plan_args["num"]
    assert data["row_count"] == expected_rows, (
        f"expected {expected_rows} rows, got {data['row_count']}: {data}"
    )
    assert len(data["rows"]) == expected_rows
