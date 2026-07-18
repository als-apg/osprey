"""Real-container ``grid_scan`` round-trip e2e (task 1.4 / PROPOSAL.md FR7).

Crown re-architects scan-plan registration to a single gold-standard registry
whose shipped set is exactly ``{orm, grid_scan}`` (FR2) — ``grid_scan`` is the
shipped ``plans_core/grid_scan.py`` file, whose canonical ``PLAN_METADATA``
name is ``"grid_scan"`` (see that module's docstring).
Its *agentic* scenario is explicitly deferred (PROPOSAL.md's Out of Scope —
it waits on a separate VA physics enhancement), so this non-agentic HTTP-level
round trip is what keeps the plan itself verified end to end in the meantime
(Secondary Goal 2): ``GET /plans`` -> ``POST /runs`` -> launch -> poll ->
``GET /runs/{id}/data``, mirroring ``test_orm_roundtrip.py``'s pattern
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
Teardown goes through ``osprey deploy down``, matching every other e2e in
this directory.

Gating: needs Docker; the VA image builds natively for the host arch, so on
Apple Silicon PyAT/softioc compile from source (no prebuilt aarch64 wheels) --
slow (minutes) on a cold image cache. Also skipped on GitHub Actions runners,
which do not provision the real Docker VA+bridge+Tiled stack this test needs
(the other real-stack e2e siblings carry the same CI gate). Run locally
with ``E2E_REUSE_IMAGES=1`` set for fast iteration once the image cache is
warm.
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
    # Skipped on CI: needs a real Docker VA+bridge+Tiled+postgres stack, which
    # the default GitHub Actions runner does not provision (see
    # tests/e2e/README.md; the other real-stack e2e siblings carry the same
    # gate).
    pytest.mark.skipif(
        os.environ.get("GITHUB_ACTIONS") == "true",
        reason="needs a real Docker stack; not provisioned on CI runners",
    ),
]

# Distinct from every other e2e module's pinned bridge port (_orm_stack.py's
# 18102, test_bluesky_deploy.py's 18090, test_va_substrate_equivalence.py's
# 18099, test_tiled_roundtrip.py's 18101, test_bluesky_catalog_e2e.py's
# 18103, test_bluesky_sandbox_escape_e2e.py's 18105, test_bluesky_panels_deploy.py's
# 18106) so this can run concurrently with any of them on a shared dev
# machine without a port collision.
BRIDGE_PORT = 18104

BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

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


def _channel_limits(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8"))


def _minted_token(project_dir: Path) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_LAUNCH_TOKEN")
    assert token, (
        "BLUESKY_LAUNCH_TOKEN missing/empty in the project .env — the arming-safe "
        "execution.execution_method: container config (FR11) should auto-mint it"
    )
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
    """Everything the round-trip test needs about the one co-deployed project."""

    def __init__(self, project_dir: Path, corrector_name: str, bpm_name: str):
        self.project_dir = project_dir
        self.corrector_name = corrector_name
        self.bpm_name = bpm_name


@pytest.fixture(scope="module")
def deployed_grid_scan_stack(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DeployedGridScanStack]:
    base = tmp_path_factory.mktemp("grid_scan_roundtrip_build")
    project_dir = _orm_stack.build_project_subprocess(
        "grid-scan-roundtrip", output_dir=base, bridge_port=BRIDGE_PORT, timeout=BUILD_TIMEOUT_SEC
    )

    limits = _channel_limits(project_dir)
    # A single corrector/BPM pair is all a 1-axis grid_scan needs -- unlike
    # the orm plan, grid_scan doesn't sweep every named corrector against
    # every named detector, so there is no benefit to _orm_stack's usual
    # DEFAULT_CORRECTOR_COUNT/DEFAULT_BPM_COUNT of 4.
    correctors = _orm_stack.select_correctors(limits, count=1)
    bpms = _orm_stack.select_bpms(limits, count=1)
    _orm_stack.write_scan_env(project_dir, correctors=correctors, bpms=bpms)

    osprey_bin = _orm_stack.find_osprey_console_script()

    # Force fresh --dev builds so the deployed containers run CURRENT source
    # (osprey deploy up does not pass --build to compose, so it would
    # otherwise reuse a stale cached image). Exact-named images only.
    # E2E_REUSE_IMAGES=1 skips this for fast local iteration on the test
    # itself when the osprey source is unchanged; never set it in CI.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(["docker", "rmi", "-f", _orm_stack.VA_IMAGE], capture_output=True, text=True)
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
        yield DeployedGridScanStack(
            project_dir=project_dir,
            corrector_name=next(iter(correctors)),
            bpm_name=next(iter(bpms)),
        )
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
    # `detectors` list, one `GridAxis` per swept dimension
    # (`setpoint`/`start`/`stop`/`num_points`), and `snake_axes`. A single
    # axis here -- the "n-dimensional" contract is exercised by
    # test_exemplar_plans.py's in-process 2-axis case; this e2e's job is the
    # real HTTP+container round trip, kept minimal to run fast.
    plan_args = {
        "detectors": [bpm_name],
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

    status, body = _post("/runs", {"plan_name": "grid_scan", "plan_args": plan_args})
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    token = _minted_token(deployed_grid_scan_stack.project_dir)
    status, body = _post(f"/runs/{run_id}/launch", {}, headers={"X-Launch-Token": token})
    assert status == 200, f"launch failed: {status} {body}"

    # No corrector-step hang: poll to a terminal status within a bounded
    # deadline (same regression class test_orm_roundtrip.py guards -- a
    # corrector whose :RB never echoes its :SP blocks the bridge's
    # ConnectorSettable.set() settle-wait forever).
    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    status_body: dict = {}
    while time.monotonic() < deadline:
        _, status_body = _get(f"/runs/{run_id}")
        if status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(0.5)
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

    rows = [dict(zip(columns, values, strict=True)) for values in data["rows"]]
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
