"""Real-container ORM round-trip e2e (task 5.2 / PROPOSAL.md's headline
Functionality criteria for FR1/FR2/FR10/FR11).

Deploys the turn-key scan-stack config (task 4.3, ``tests/e2e/_orm_stack.py``
-- the single source of this deploy shape, also reused by the agentic
discovery e2e in 5.3/5.4), drives the real ``orm`` plan over the bridge's
HTTP API (``POST /runs`` -> launch -> poll -> ``GET /runs/{id}/data``), and
proves two things end to end:

  (a) the measured response matrix -- built via the SAME
      ``orm_analysis.build_response_matrix`` an MCP-side analysis step would
      use -- agrees with the independent ``lattice/response.py`` model oracle
      (mirrors task 5.1's in-process cross-check, but over the deployed
      HTTP+container stack rather than a direct ``PhysicsBridge`` call).
  (b) the scan reaches a terminal "completed" status within a bounded
      timeout -- i.e. no corrector step ever hangs the bridge's
      ``ConnectorSettable.set()`` settle-wait (``devices/connector.py`` --
      the connector-mediated device layer replacing the old direct-CA
      ``EpicsMotor``). A corrector readback that never leaves 0.0 (the FR10
      regression this deploy config's ``echo-pyat-coupled-sp-to-rb`` fix
      addresses) blocks exactly there, so a fixture-level timeout on this
      assertion IS the regression signature, not an incidentally slow test.

No physics fault is seeded on this stack (no ``VA_BPM_ERRORS``/
``VA_CORR_GAIN`` in the written ``.env`` -- see ``_orm_stack.write_scan_env``),
so every BPM/corrector carries the identity error state
(``PhysicsBridge.__init__``'s default). The measured/model
agreement is therefore bounded only by AT numerical-solve reproducibility and
the JSON/HTTP round trip, not a physical noise floor -- see ``MATCH_ATOL``.

No preset channel names are hardcoded: correctors and BPMs are derived from
the DEPLOYED project's own ``data/channel_limits.json`` via
``_orm_stack.select_correctors``/``select_bpms`` (restricted to the
pyat-coupled partition, exactly the class of device the ``orm`` plan and the
model oracle both operate on).

Container safety: every docker invocation below names an exact
container/image -- never a wildcard, never ``system prune``/``--volumes``.
Teardown goes through ``osprey deploy down``, matching every other e2e in
this directory.

Gating: needs Docker; the VA image builds natively for the host arch, so on
Apple Silicon PyAT/softioc compile from source (no prebuilt aarch64 wheels) --
slow (minutes) on a cold image cache. Advisory CI lane (see ci.yml); run
locally with ``E2E_REUSE_IMAGES=1`` set for fast iteration once the image
cache is warm.
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

import numpy as np
import pytest

from osprey.services.bluesky_bridge.orm_analysis import build_response_matrix
from tests.e2e import _orm_stack

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    # dockerbuild: full VA/bridge/Tiled image build + deploy -- runs in the
    # dedicated orm-roundtrip-e2e CI job, never the shared e2e-tests lane
    # (the marker->--ignore pairing is enforced by
    # tests/deployment/test_ci_workflow_wiring.py).
    pytest.mark.dockerbuild,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

BRIDGE_URL = f"http://localhost:{_orm_stack.BRIDGE_PORT}"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 1200  # first-time native VA source build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0
# 4 correctors x 5 points x an 8-device (4 corrector + 4 BPM) bundle read per
# point -- generous headroom over a healthy run's expected
# few-tens-of-seconds so this stays a meaningful "did it hang" gate rather
# than a flaky timing assertion.
SCAN_TIMEOUT_SEC = 240.0

# Within the corrector channel_limits band (+-12A) and the response model's
# documented linear regime (ORMParams' own docstring); NUM_POINTS >= 3 per
# the schema's `ge=3`.
SPAN_A = 5.0
NUM_POINTS = 5

# No VA_BPM_ERRORS/VA_CORR_GAIN are seeded on this stack (see module
# docstring) -- every device carries PhysicsBridge's identity error
# state, so there is no physical noise floor to size this against. The bound
# below is float round-trip/AT numerical-solve reproducibility margin, kept
# generous relative to task 5.1's probed in-process figure (4.9e-15 relative)
# to absorb the extra JSON/HTTP/container hop.
MATCH_RTOL = 1e-6
MATCH_ATOL = 1e-9  # meters


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


class DeployedOrmStack:
    """Everything the round-trip test needs about the one co-deployed project."""

    def __init__(
        self,
        project_dir: Path,
        correctors: dict[str, tuple[str, str]],
        bpms: dict[str, str],
    ):
        self.project_dir = project_dir
        self.correctors = correctors
        self.bpms = bpms


@pytest.fixture(scope="module")
def deployed_orm_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedOrmStack]:
    base = tmp_path_factory.mktemp("orm_roundtrip_build")
    project_dir = _orm_stack.build_project_subprocess(
        "orm-roundtrip", output_dir=base, timeout=BUILD_TIMEOUT_SEC
    )

    limits = _channel_limits(project_dir)
    correctors = _orm_stack.select_correctors(limits)
    bpms = _orm_stack.select_bpms(limits)
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
        yield DeployedOrmStack(project_dir=project_dir, correctors=correctors, bpms=bpms)
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
# Model oracle -- lattice/response.py's independent orbit_response, driven the
# same way the deployed orm plan sweeps (mirrors build_response_matrix's own
# degree-1-polyfit-over-the-sweep method, not a two-point finite difference,
# so a mismatch can only mean the two code paths disagree -- never an
# artifact of the model and measured fits taking a different shape).
# ---------------------------------------------------------------------------


def _corrector_famname(sp_address: str) -> str:
    """e.g. "SR:MAG:HCM:05:CURRENT:SP" -> "HCM05" (matches PhysicsBridge's
    on_setpoint / lattice/response.py's orbit_response FamName convention)."""
    _ring, _system, family, device, _field, _subfield = sp_address.split(":")
    return f"{family}{device}"


def _bpm_famname_axis(address: str) -> tuple[str, int]:
    """e.g. "SR:DIAG:BPM:07:POSITION:X" -> ("BPM07", 0); ":Y" -> axis 1
    (matches orbit_response's (x, y) tuple order)."""
    _ring, _system, family, device, _field, subfield = address.split(":")
    return f"{family}{device}", {"X": 0, "Y": 1}[subfield]


def _model_response_matrix(
    correctors: dict[str, tuple[str, str]],
    bpms: dict[str, str],
    currents: list[float],
) -> np.ndarray:
    from osprey.services.virtual_accelerator.lattice import orbit_response

    matrix = np.zeros((len(bpms), len(correctors)))
    for j, (_corr_name, (sp, _rb)) in enumerate(correctors.items()):
        fam = _corrector_famname(sp)
        readings = [orbit_response(fam, current) for current in currents]
        for i, (_bpm_name, addr) in enumerate(bpms.items()):
            bpm_fam, axis = _bpm_famname_axis(addr)
            values = [reading[bpm_fam][axis] for reading in readings]
            slope, _intercept = np.polyfit(currents, values, deg=1)
            matrix[i, j] = slope
    return matrix


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_orm_roundtrip_matches_model_with_no_corrector_hang(
    deployed_orm_stack: DeployedOrmStack,
) -> None:
    status, plans_body = _get("/plans")
    assert status == 200, f"GET /plans failed: {status} {plans_body}"
    plan_names = {p["name"] for p in plans_body}
    assert "orm" in plan_names, f"orm plan not discoverable via GET /plans: {plan_names}"

    correctors = deployed_orm_stack.correctors
    bpms = deployed_orm_stack.bpms
    plan_args = {
        "correctors": list(correctors),
        "detectors": list(bpms),
        "span_a": SPAN_A,
        "num": NUM_POINTS,
    }

    status, body = _post("/runs", {"plan_name": "orm", "plan_args": plan_args})
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    token = _minted_token(deployed_orm_stack.project_dir)
    status, body = _post(f"/runs/{run_id}/launch", {}, headers={"X-Launch-Token": token})
    assert status == 200, f"launch failed: {status} {body}"

    # (b) no corrector-step hang: poll to a terminal status within a bounded
    # deadline. A corrector whose :RB never echoes its :SP (the FR10
    # regression) blocks the bridge's ConnectorSettable.set() settle-wait
    # forever -- so a non-"completed" status here, after the deadline, IS the
    # failure this proves absent, not merely a slow run.
    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    status_body: dict = {}
    while time.monotonic() < deadline:
        _, status_body = _get(f"/runs/{run_id}")
        if status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(0.5)
    assert status_body.get("status") == "completed", (
        f"orm scan did not complete within {SCAN_TIMEOUT_SEC:.0f}s (status={status_body}) -- "
        "a corrector step whose :RB never echoes its :SP (the FR10 echo regression) hangs "
        "exactly here, at the bridge's ConnectorSettable.set() settle-wait"
    )

    status, data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    expected_rows = len(correctors) * NUM_POINTS
    assert data["row_count"] == expected_rows, (
        f"expected {expected_rows} rows (one per (corrector, current) point), "
        f"got {data['row_count']}: {data}"
    )

    columns = data["columns"]
    rows = [dict(zip(columns, values, strict=True)) for values in data["rows"]]
    measured = build_response_matrix(rows, correctors=list(correctors), detectors=list(bpms))

    # (a) matches the model oracle: the same symmetric-sweep currents the
    # deployed orm plan itself computes (plans.py's _orm_plan), so the model
    # is driven identically to how the plan drove the real stack.
    step = (2 * SPAN_A) / (NUM_POINTS - 1)
    currents = [-SPAN_A + i * step for i in range(NUM_POINTS)]
    model = _model_response_matrix(correctors, bpms, currents)

    assert np.allclose(measured, model, rtol=MATCH_RTOL, atol=MATCH_ATOL), (
        "measured ORM (driven over the deployed HTTP+container stack) does not match "
        f"the independent lattice/response.py model oracle within tolerance "
        f"(rtol={MATCH_RTOL}, atol={MATCH_ATOL}):\n"
        f"measured=\n{measured}\nmodel=\n{model}\n"
        f"max abs diff={float(np.max(np.abs(measured - model))):.3e}"
    )
