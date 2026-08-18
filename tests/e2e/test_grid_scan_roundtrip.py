"""Real-container ``grid_scan`` round-trip e2e (task 1.4 / PROPOSAL.md FR7).

Crown re-architects scan-plan registration to a single gold-standard registry
whose shipped set is exactly ``{orm, grid_scan}`` (FR2) — ``grid_scan`` is the
shipped ``plans_core/grid_scan.py`` file, whose canonical ``PLAN_METADATA``
name is ``"grid_scan"`` (see that module's docstring).
Its *agentic* scenario is explicitly deferred (PROPOSAL.md's Out of Scope —
it waits on a separate VA physics enhancement), so this non-agentic HTTP-level
round trip is what keeps the plan itself verified end to end in the meantime
(Secondary Goal 2): ``GET /plans`` -> ``PATCH /draft`` -> ``POST /queue/items``
-> armed ``POST /queue/start`` -> poll -> ``GET /runs/{id}/data`` (the queue
flow, spelled once in ``tests/e2e/_queue_drive.py``), mirroring
``test_orm_roundtrip.py``'s pattern
(task 5.2) but driving ``grid_scan`` instead of ``orm``, and without that
test's model-oracle cross-check (``grid_scan`` has no physics model to
compare against — it wraps ``bluesky.plans.grid_scan`` generically).

Proves two things end to end:

  (a) the deployed plan produces a well-formed rectangular-grid result: a
      column for the swept corrector, a column for the read BPM, and exactly
      ``num_points`` rows (one axis, so the grid is 1-D here) — the product
      of each axis's ``num_points`` per ``plans_core/grid_scan.py``'s own
      contract.
  (b) the swept corrector's readback actually visits every distinct
      commanded grid point (not stuck at one value) -- the same class of
      corrector-echo regression ``test_orm_roundtrip.py`` guards against,
      applied to `grid_scan`'s ``bps.mv``-driven axis instead of ``orm``'s
      current sweep.

Reuses ``tests/e2e/_orm_stack.py`` (task 4.3's single source of this deploy
shape) for the build/deploy scaffold and the channel-limits-derived
corrector/BPM selection -- no hardcoded preset channel, no re-derived deploy
logic.

Container safety: every docker invocation below names an exact
container/image -- never a wildcard, never ``system prune``/``--volumes``.
Teardown goes through ``osprey down``, matching every other e2e in
this directory, followed by exact-named removal of this project's own volumes
(``tests/e2e/_volumes.py``): ``down`` keeps them by design, and a rerun must
not inherit their state.

Gating: needs Docker; the VA image builds natively for the host arch, so on
Apple Silicon PyAT/softioc compile from source (no prebuilt aarch64 wheels) --
slow (minutes) on a cold image cache. On CI this runs in the dedicated
``orm-roundtrip-e2e`` job, after ``test_orm_roundtrip.py`` and never
alongside it -- both stand up their own VA on CA port 5064. Run locally
with ``E2E_REUSE_IMAGES=1`` set for fast iteration once the image cache is
warm.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.services.bluesky_bridge.figure import rows_from_columnar
from tests.e2e import _orm_stack, _queue_drive
from tests.e2e._deploy_diagnostics import queue_stack_logs
from tests.e2e._volumes import remove_project_volumes

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    # dockerbuild: full VA/bridge/Tiled image build + deploy -- runs in the
    # dedicated orm-roundtrip-e2e CI job alongside test_orm_roundtrip.py,
    # never the shared e2e-tests lane (the marker->--ignore pairing is
    # enforced by tests/deployment/test_ci_workflow_wiring.py).
    pytest.mark.dockerbuild,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

# Distinct from every other e2e module's pinned bridge port (_orm_stack.py's
# 18102, test_bluesky_deploy.py's 18090, test_va_substrate_equivalence.py's
# 18099, test_tiled_roundtrip.py's 18101, test_bluesky_catalog_e2e.py's
# 18103, test_bluesky_sandbox_escape_e2e.py's 18105, test_bluesky_web_deploy.py's
# 18106) so this can run concurrently with any of them on a shared dev
# machine without a port collision.
BRIDGE_PORT = 18104

BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

#: Compose project this suite deploys under. Locally-built image tags follow
#: ``<project>-<service>``, so the forced image refresh below needs the same
#: name the build is given -- hence one constant rather than repeated literals.
PROJECT_NAME = "grid-scan-roundtrip"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 1200  # first-time native VA source build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0
# One corrector x 3 points x a 2-device (1 corrector + 1 BPM) bundle read per
# point -- generous headroom over a healthy run's expected few-seconds so
# this stays a meaningful "did it hang" gate rather than a flaky timing
# assertion (mirrors test_orm_roundtrip.py's SCAN_TIMEOUT_SEC rationale,
# scaled down for this test's much smaller device count).
SCAN_TIMEOUT_SEC = 120.0

# One axis, few points: keeps the run fast while still proving a real
# rectangular grid (not a degenerate single-point scan) -- 3 is the smallest
# value that lets the "every point visited" check (b) distinguish "grid
# actually swept" from "coincidentally saw the endpoints twice". Values stay
# well inside the corrector channel_limits band (+-12A, same band
# test_orm_roundtrip.py's SPAN_A documents) and grid_scan's own `ge=2`
# num_points floor.
AXIS_START_A = -3.0
AXIS_STOP_A = 3.0
NUM_POINTS = 3


def _get(path: str) -> tuple[int, Any]:
    return _queue_drive.request(BRIDGE_URL, path, "GET")


def _find_column(columns: list[str], device_name: str) -> str:
    """The event-data column for a device -- ophyd-async names a hinted
    child ``"<device>-<child>"``; match the device-name prefix rather than
    the exact key so this doesn't hardcode ophyd-async's internal
    child-attribute naming (mirrors test_va_substrate_equivalence.py's
    identical helper)."""
    for col in columns:
        if col == device_name or col.startswith(f"{device_name}-"):
            return col
    raise AssertionError(f"no column for device {device_name!r} in {columns!r}")


class DeployedGridScanStack:
    """Everything the round-trip test needs about the one deployment repo."""

    def __init__(self, repo: Path, corrector_name: str, bpm_name: str):
        self.repo = repo
        self.corrector_name = corrector_name
        self.bpm_name = bpm_name


@pytest.fixture(scope="module")
def deployed_grid_scan_stack(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DeployedGridScanStack]:
    base = tmp_path_factory.mktemp("grid_scan_roundtrip_build")
    # The deployment REPO: `osprey up` runs here, `.env` lives here, and the
    # render `osprey build` produced is `<repo>/build`.
    repo = _orm_stack.build_project_subprocess(
        PROJECT_NAME, output_dir=base, bridge_port=BRIDGE_PORT, timeout=BUILD_TIMEOUT_SEC
    )

    # The render's copy, not the operator-owned source under <repo>/data/ —
    # build/data is the file the deployed containers actually read.
    limits = _orm_stack.channel_limits(repo / "build")
    # A single corrector/BPM pair is all a 1-axis grid_scan needs -- unlike
    # the orm plan, grid_scan doesn't sweep every named corrector against
    # every named detector, so there is no benefit to _orm_stack's usual
    # DEFAULT_CORRECTOR_COUNT/DEFAULT_BPM_COUNT of 4.
    correctors = _orm_stack.select_correctors(limits, count=1)
    bpms = _orm_stack.select_bpms(limits, count=1)
    # Writes the repo root's `.env` — the deployment's whole secret store, and
    # the file `osprey up` refuses to start without.
    _orm_stack.write_substrate_env(repo, correctors=correctors, bpms=bpms)

    osprey_bin = _orm_stack.find_osprey_console_script()

    _orm_stack.force_image_rebuild(
        _orm_stack.va_image(PROJECT_NAME), _orm_stack.bridge_image(PROJECT_NAME)
    )

    try:
        up = subprocess.run(
            [str(osprey_bin), "up", "-d", "--dev"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _orm_stack.wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        # HTTP readiness is not enqueue readiness -- the worker namespace the
        # enqueue validates against exists only once the RE worker environment
        # is open, and the bridge opens that off the readiness path. See
        # `_queue_drive.wait_for_worker_environment`.
        try:
            _queue_drive.wait_for_worker_environment(BRIDGE_URL)
        except AssertionError as exc:
            pytest.fail(f"{exc}\n{queue_stack_logs(_orm_stack.project_prefix(PROJECT_NAME))}")
        yield DeployedGridScanStack(
            repo=repo,
            corrector_name=next(iter(correctors)),
            bpm_name=next(iter(bpms)),
        )
    finally:
        down = subprocess.run(
            [str(osprey_bin), "down"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        # `osprey down` keeps volumes by design; drop this project's own so a
        # rerun cannot inherit their state (see tests/e2e/_volumes.py).
        remove_project_volumes(_orm_stack.project_prefix(PROJECT_NAME))


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_grid_scan_roundtrip_produces_a_well_formed_grid(
    deployed_grid_scan_stack: DeployedGridScanStack,
) -> None:
    status, plans_body = _get("/plans")
    assert status == 200, f"GET /plans failed: {status} {plans_body}"
    plan_names = {p["name"] for p in plans_body}
    assert plan_names == {"orm", "grid_scan"}, (
        f"expected the shipped plan set to be exactly {{orm, grid_scan}} (FR2), got {plan_names}"
    )

    corrector_name = deployed_grid_scan_stack.corrector_name
    bpm_name = deployed_grid_scan_stack.bpm_name

    # Canonical grid_scan schema (plans_core/grid_scan.py's PARAMS): a
    # `readables` list, one `GridAxis` per swept dimension
    # (`setpoint`/`start`/`stop`/`num_points`), and `snake_axes`. A single
    # axis here -- the "n-dimensional" contract is exercised by
    # test_exemplar_plans.py's in-process 2-axis case; this e2e's job is the
    # real HTTP+container round trip, kept minimal to run fast.
    plan_args = {
        "readbacks": [bpm_name],
        "axes": [
            {
                "setpoint": corrector_name,
                "start": AXIS_START_A,
                "stop": AXIS_STOP_A,
                "num_points": NUM_POINTS,
            }
        ],
        "snake_axes": False,
    }

    token = _orm_stack.minted_launch_token(deployed_grid_scan_stack.repo)
    run_id, status_body = _queue_drive.run_plan(
        BRIDGE_URL,
        "grid_scan",
        plan_args,
        token=token,
        client_id="grid-scan-roundtrip-e2e",
        timeout=SCAN_TIMEOUT_SEC,
    )

    # No corrector-step hang: the poll above ran to a terminal status within a
    # bounded deadline (same regression class test_orm_roundtrip.py guards --
    # a corrector whose :RB never echoes its :SP blocks the bridge's
    # ConnectorSettable.set() settle-wait forever).
    assert status_body.get("status") == "completed", (
        f"grid_scan did not complete within {SCAN_TIMEOUT_SEC:.0f}s (status={status_body}) -- "
        "a corrector step whose :RB never echoes its :SP hangs exactly here, at the bridge's "
        "ConnectorSettable.set() settle-wait"
    )

    status, data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"

    # (a) a well-formed grid result: one row per grid point -- with a single
    # axis, that's simply num_points (the general contract is
    # prod(num_points across axes); see plans_core/grid_scan.py's docstring).
    assert data["row_count"] == NUM_POINTS, (
        f"expected {NUM_POINTS} rows (one per grid point), got {data['row_count']}: {data}"
    )

    columns = data["columns"]
    corrector_col = _find_column(columns, corrector_name)
    bpm_col = _find_column(columns, bpm_name)

    rows = rows_from_columnar(columns, data["rows"], data["row_count"]).rows
    assert len(rows) == NUM_POINTS

    corrector_values = [row[corrector_col] for row in rows]
    bpm_values = [row[bpm_col] for row in rows]

    assert all(v is not None for v in corrector_values), (
        f"corrector column {corrector_col!r} has a null reading: {corrector_values}"
    )
    assert all(v is not None for v in bpm_values), (
        f"detector column {bpm_col!r} has a null reading: {bpm_values}"
    )

    # (b) every distinct commanded grid point was actually visited -- not
    # stuck at one value, the corrector-echo regression this suite otherwise
    # guards against via the orm plan's sweep.
    distinct_values = {round(v, 3) for v in corrector_values}
    assert len(distinct_values) == NUM_POINTS, (
        f"expected {NUM_POINTS} distinct corrector readings (one per grid point), "
        f"got {sorted(distinct_values)} from {corrector_values} -- the corrector may be stuck "
        "at one value instead of stepping through the grid"
    )
    expected_values = {
        round(AXIS_START_A + i * (AXIS_STOP_A - AXIS_START_A) / (NUM_POINTS - 1), 3)
        for i in range(NUM_POINTS)
    }
    assert distinct_values == expected_values, (
        f"corrector readings {sorted(distinct_values)} don't match the commanded grid points "
        f"{sorted(expected_values)}"
    )
