"""Phase 1 acceptance proof: the Virtual Accelerator + Bluesky bridge substrate,
co-deployed as real containers, is equivalent to (and honest about divergence
from) a real EPICS beamline (PROPOSAL.md's Risk-1/Station-2 gate).

One module-scoped build + ``osprey deploy up -d --dev`` co-deploys the
Virtual Accelerator (task 4.1) and the Bluesky bridge (task 2.9) wired to the
EPICS substrate scanner (task 2.3), with one sp-echo ``:SP`` pre-faulted
(task 3.1's ``VA_STUCK_SETPOINTS``) via task 4.2's env passthrough. Five
proofs then exercise the whole stack end to end:

  P1 co-deploy:     containers up, healthy, loopback-only, depends_on ordering held.
  P2 liveness:      the full manifest namespace is reachable over CA.
  P3 read-equiv:    a pyepics (host) read and an ophyd-async (bridge) read of
                     the same PV agree.
  P4 concurrent:    an EPICS-substrate ``scan`` plan runs to completion while a
                     concurrent host read observes the same PV consistently —
                     the loop-affinity falsifier (task 2.1).
  P5 honest divergence: a write to a pre-faulted ``:SP`` verifies (the SP
                     always latches its own readback), but an independent read
                     of the sibling ``:RB`` proves it never moved — and both
                     CA clients (host + bridge) agree on that frozen value.

No preset channel names are hardcoded: every address used below is derived
from the DEPLOYED project's own ``data/channel_limits.json`` (writable ⟺ a
``:SP`` address) restricted to sp-echo pairs (``classify_partition`` — a
write to a pyat-coupled ``:SP`` has ring-wide physics side effects, wrong for
an isolated fault/equivalence probe; sp-echo is a pure software echo, exactly
what P3-P5 need).

Container safety: every docker invocation below names an exact container/image
— never a wildcard, never ``system prune``/``--volumes``. The one forced
``docker rmi -f <image>`` (below) names an exact image, matching
``test_bluesky_deploy.py``'s precedent for forcing a fresh ``--dev`` build.
Teardown goes through ``osprey deploy down``, never a raw ``docker rm`` sweep.

Gating: needs Docker; the VA image builds natively for the host arch, so on
Apple Silicon PyAT/softioc compile from source (no prebuilt aarch64 wheels) —
slow (minutes) on a cold image cache. Lives in ``tests/e2e/`` (never
collected by the fast lane, see ``ci_check.sh``/ci.yml).

Markers: ``pytest.mark.flaky(reruns=1, only_rerun=[AssertionError])`` is
applied PER-FUNCTION to P1-P4 only, never at module level — P5 is the safety
proof and must stay strict (mirrors ``test_bluesky_write_refused_e2e``'s
strictness). A module-level ``flaky`` would silently sweep P5 into lenient
reruns, which is exactly the bug this convention exists to prevent.
"""

from __future__ import annotations

import asyncio
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "va" / "sweep_check.py"
# Out-of-process host-side CA op (see its module docstring): each P3-P5 host
# read/write runs in its own short-lived process so the libca CA-teardown
# assertion can never recur in this pytest process.
HOST_CA_OP_SCRIPT = Path(__file__).resolve().parent / "_va_host_ca_op.py"
# Must match _va_host_ca_op.RESULT_MARKER (kept as a local literal rather than
# imported -- tests/e2e is a package, so the helper is not on sys.path).
HOST_CA_RESULT_MARKER = "__HOST_CA_RESULT__"

# Channel Access port the Virtual Accelerator serves on. NOT freely
# overridable here: the Control Assistant preset's config.yml.j2 hardcodes
# `control_system.connector.virtual_accelerator.gateways.*.port: 5064` (it is
# not templated from `services.virtual_accelerator.port`) — so the host-side
# connector config below and the container's published port must both stay
# at this value, or the two silently drift apart.
VA_CA_PORT = 5064
VA_IMAGE = "osprey-va:local"
# The fixture builds/deploys under this project name; the compose templates
# render each service's container_name as ``<project>-<service>``
# (services/*/docker-compose.yml.j2), so derive them rather than hardcode
# host-global names that break the moment the templates are namespaced per-project.
PROJECT_NAME = "proj"
VA_CONTAINER = f"{PROJECT_NAME}-virtual-accelerator"

# Deliberately non-default (avoids colliding with test_bluesky_deploy.py's 18090
# on a shared dev machine — see that module's docstring).
BRIDGE_PORT = 18099
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"
BRIDGE_IMAGE = "osprey-bluesky-bridge:local"
BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"

# Device names wired into the bridge via BLUESKY_EPICS_MOTORS/_DETECTORS —
# arbitrary, resolved against explicit PV addresses (see _write_scan_env
# below), never a preset naming convention.
SCAN_MOTOR = "scan_motor"
P3_DETECTOR = "p3_det"
P4_DETECTOR = "p4_det"
P5_DETECTOR = "p5_det"

# The bridge's launch route (POST /runs/{id}/launch) fails closed on an unset
# BLUESKY_LAUNCH_TOKEN. `osprey deploy up` normally auto-mints one, but the
# control-assistant preset deploys with control_system.writes_enabled: true AND
# execution.execution_method: local, which deliberately gates auto-arming off
# (container_lifecycle._local_exec_arming_unsafe — a local unsandboxed agent
# could read the token and bypass the write gate). This e2e is a controlled
# test, not agent code, so it supplies its own token explicitly (the supported
# operator-provides-a-token path) rather than exercise that arming policy here.
LAUNCH_TOKEN = "e2e-substrate-equivalence-launch-token"

BUILD_TIMEOUT_SEC = 300
DEPLOY_UP_TIMEOUT_SEC = 1200  # first-time native VA source build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0
SWEEP_TIMEOUT_SEC = 120.0  # sweep()'s own connect deadline defaults to 45s
SCAN_TIMEOUT_SEC = 60.0
CONTAINER_HEALTH_TIMEOUT_SEC = 90.0  # docker healthcheck start_period + a few intervals
# Host CA op subprocess: process spawn + connector connect (name-server TCP) +
# one write/read round trip. The connector's own timeout is 5s (CONNECTOR_CONFIG).
HOST_CA_OP_TIMEOUT_SEC = 60.0

# Host-side connector config: points at the co-deployed VA over CA
# name-server/TCP mode — the one host<->container CA configuration proven to
# work across container runtimes (see probe/README.md). Mirrors
# tests/va/e2e/conftest.py's VA_GATEWAY_CONFIG/CONNECTOR_CONFIG exactly
# (duplicated locally rather than imported — this file owns nothing in that
# directory and must not couple to its collection hook).
_VA_GATEWAY = {"address": "localhost", "port": VA_CA_PORT, "use_name_server": True}
CONNECTOR_CONFIG: dict[str, Any] = {
    "type": "virtual_accelerator",
    "connector": {
        "virtual_accelerator": {
            "timeout": 5.0,
            "gateways": {"read_only": _VA_GATEWAY, "write_access": _VA_GATEWAY},
        }
    },
}

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]


# ---------------------------------------------------------------------------
# Build/deploy scaffold (mirrors tests/e2e/test_bluesky_deploy.py's shape)
# ---------------------------------------------------------------------------


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


def _channel_limits(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8"))


def _select_sp_echo_pairs(channel_limits: dict[str, Any], count: int) -> list[tuple[str, str]]:
    """Derive ``count`` disjoint sp-echo (``:SP``, ``:RB``) pairs from the
    deployed project's own channel_limits.json -- no hardcoded preset
    channels.

    A channel is writable (candidate ``:SP``) iff its channel_limits.json
    entry exists with that address ending ``:SP`` (the connector's own
    writability contract). Restricted to the sp-echo partition
    (``classify_partition``) rather than every writable ``:SP``: a
    pyat-coupled ``:SP`` write has ring-wide physics side effects (moves
    other BPMs via the lattice model), wrong for an isolated
    equivalence/fault probe -- sp-echo is a pure, isolated software copy
    (write SP, RB follows immediately, nothing else touched).
    """
    from osprey.services.virtual_accelerator.manifest import PARTITION_SP_ECHO, classify_partition

    keys = {k for k in channel_limits if not k.startswith("_") and k != "defaults"}
    sp_keys = sorted(k for k in keys if k.endswith(":SP"))

    pairs: list[tuple[str, str]] = []
    for sp in sp_keys:
        parts = sp.split(":")
        if len(parts) != 6:
            continue
        ring, system, family, device, field, subfield = parts
        path = {
            "ring": ring,
            "system": system,
            "family": family,
            "device": device,
            "field": field,
            "subfield": subfield,
        }
        if classify_partition(path) != PARTITION_SP_ECHO:
            continue
        rb = sp[:-3] + ":RB"
        if rb in keys:
            pairs.append((sp, rb))

    if len(pairs) < count:
        raise AssertionError(
            f"deployed project's channel_limits.json only yields {len(pairs)} sp-echo "
            f"pairs, need {count}"
        )
    return pairs[:count]


def _write_scan_env(project_dir: Path, pairs: dict[str, tuple[str, str]]) -> None:
    """Append task 4.2's contract env vars to the project ``.env`` -- BEFORE
    ``osprey deploy up`` (the bridge/VA compose templates pass these through
    from the project ``.env``, same mechanism as ``BLUESKY_LAUNCH_TOKEN``).
    """
    p3_sp, p3_rb = pairs["p3"]
    p4_sp, p4_rb = pairs["p4"]
    p5_sp, p5_rb = pairs["p5"]

    values = {
        # Supply the launch token ourselves — the preset's local-exec+writes
        # config gates auto-minting off (see LAUNCH_TOKEN above).
        "BLUESKY_LAUNCH_TOKEN": LAUNCH_TOKEN,
        "BLUESKY_EPICS_SUBSTRATE": "1",
        "BLUESKY_EPICS_MOTORS": f"{SCAN_MOTOR}={p4_sp}|{p4_rb}",
        "BLUESKY_EPICS_DETECTORS": (
            f"{P3_DETECTOR}={p3_rb},{P4_DETECTOR}={p4_rb},{P5_DETECTOR}={p5_rb}"
        ),
        "VA_STUCK_SETPOINTS": p5_sp,
    }

    env_path = project_dir / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    new_lines = "".join(f"{k}={v}\n" for k, v in values.items())
    env_path.write_text(existing + new_lines, encoding="utf-8")


class DeployedStack:
    """Everything the P1-P5 tests need about the one co-deployed project."""

    def __init__(
        self, project_dir: Path, pairs: dict[str, tuple[str, str]], limits: dict[str, Any]
    ):
        self.project_dir = project_dir
        self.pairs = pairs
        self.limits = limits

    def bounds(self, address: str) -> tuple[float, float]:
        entry = self.limits[address]
        return float(entry["min_value"]), float(entry["max_value"])


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedStack]:
    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("va_substrate_build")
    project_dir = base / PROJECT_NAME

    # Extends control-assistant (which already ships data/simulation/machine.json
    # + channel_limits.json) with the one flag it doesn't default to: the
    # control-system type. Written as a flat dotted-string key under `config:`
    # (matching the preset's own convention) rather than a `--set
    # config.control_system.type=...` CLI override -- `--set` builds a NESTED
    # dict for every dotted segment, which would replace the entire
    # `control_system:` block (wiping writes_enabled/limits_checking/connector
    # gateways) instead of overriding just the `type` field.
    # `dispatch: null` drops control-assistant's default event-dispatcher
    # stack (Node + Claude CLI image) -- irrelevant here and far slower to
    # build than the VA image already is.
    # `modules.web_terminals.enabled: false` scopes this deploy back to the VA +
    # bridge substrate: the control-assistant preset now ships the multi-user
    # web-terminal stack on by default, so an unqualified `deploy up` would also
    # auto-render both persona projects, build two web images, and start
    # nginx/web containers -- none of which this substrate-equivalence proof
    # exercises (that topology is covered by test_control_assistant_demo.py).
    override_path = base / "override.yml"
    override_path.write_text(
        "config:\n"
        "  control_system.type: virtual_accelerator\n"
        "  modules.web_terminals.enabled: false\n"
        "dispatch: null\n",
        encoding="utf-8",
    )

    build = _run(
        [
            str(osprey_bin),
            "build",
            PROJECT_NAME,
            "--preset",
            "control-assistant",
            "--override",
            str(override_path),
            "--set",
            f"virtual_accelerator.port={VA_CA_PORT}",
            "--set",
            f"bluesky.port={BRIDGE_PORT}",
            "--set",
            "bluesky.demo_runner=false",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(base),
            "--force",
        ],
        cwd=base,
        timeout=BUILD_TIMEOUT_SEC,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    limits = _channel_limits(project_dir)
    sp3, sp4, sp5 = _select_sp_echo_pairs(limits, count=3)
    pairs = {"p3": sp3, "p4": sp4, "p5": sp5}
    _write_scan_env(project_dir, pairs)

    # Force fresh --dev builds so the deployed containers run CURRENT source
    # (osprey deploy up does not pass --build to compose, so it would
    # otherwise reuse a stale cached image). Exact-named images only.
    # E2E_REUSE_IMAGES=1 skips this (dev-only: fast local iteration on the test
    # itself when the osprey source is unchanged; never set it in CI, where a
    # source change must always rebuild). The first-time native VA build is slow.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(["docker", "rmi", "-f", VA_IMAGE], capture_output=True, text=True)
        subprocess.run(["docker", "rmi", "-f", BRIDGE_IMAGE], capture_output=True, text=True)

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
        yield DeployedStack(project_dir=project_dir, pairs=pairs, limits=limits)
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


def _wait_for_container_health(container: str, timeout: float) -> None:
    """Poll ``docker inspect .State.Health.Status`` until ``healthy`` or timeout.

    The fixture's HTTP-readiness gate can pass while Docker still reports
    ``starting`` (the healthcheck runs only on its interval, after
    ``start_period``), so an instant equality assert is racy.
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


def _minted_token(project_dir: Path) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_LAUNCH_TOKEN")
    assert token, "BLUESKY_LAUNCH_TOKEN missing/empty in the project .env"
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


def _find_column(columns: list[str], device_name: str) -> int:
    """The event-data column for a device -- ophyd-async names a hinted
    child `"<device>-<child>"`; match the device-name prefix rather than the
    exact key so this doesn't hardcode ophyd-async's internal child-attribute
    naming."""
    for i, col in enumerate(columns):
        if col == device_name or col.startswith(f"{device_name}-"):
            return i
    raise AssertionError(f"no column for device {device_name!r} in {columns!r}")


def _docker_inspect(container: str, fmt: str) -> str:
    proc = subprocess.run(
        ["docker", "inspect", "--format", fmt, container],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"docker inspect {container} failed: {proc.stderr}"
    return proc.stdout.strip()


async def _run_scan(
    plan_name: str, plan_args: dict, project_dir: Path, timeout: float = SCAN_TIMEOUT_SEC
) -> tuple[str, dict]:
    """POST /runs -> launch -> poll to a terminal status. Returns (run_id, final_status_body)."""
    token = _minted_token(project_dir)
    status, body = _post("/runs", {"plan_name": plan_name, "plan_args": plan_args})
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    status, body = _post(f"/runs/{run_id}/launch", {}, headers={"X-Launch-Token": token})
    assert status == 200, f"launch failed: {status} {body}"

    deadline = time.monotonic() + timeout
    last_status_body: dict = {}
    while time.monotonic() < deadline:
        _, last_status_body = _get(f"/runs/{run_id}")
        if last_status_body.get("status") in ("completed", "error", "stopped"):
            break
        await asyncio.sleep(0.2)
    return run_id, last_status_body


def _host_ca_op_spec(
    project_dir: Path,
    *,
    read: str,
    write: dict[str, Any] | None = None,
    settle_read: bool = False,
) -> dict[str, Any]:
    """Build the JSON spec for one out-of-process host CA op (``_va_host_ca_op.py``).

    Carries the SAME ``get_config_value`` overrides the in-process connector
    used -- ``project_root`` and ``limits_checking.database_path`` point at the
    REAL deployed project so ``LimitsValidator`` enforces the REAL
    channel_limits.json this test selected channels from (these proofs write only
    LISTED sp-echo ``:SP`` channels, so limits are actually applied to them), and
    ``CONNECTOR_CONFIG`` is passed verbatim so the subprocess builds a REAL
    production ``VirtualAcceleratorConnector`` via ``ConnectorFactory`` under
    test-supplied config. Config keys these proofs don't exercise (e.g.
    write-verification ``default_level``/auto-tolerance) fall to code defaults --
    inert here, since every write passes ``verification_level="readback"``
    explicitly. The only thing that changes vs. an in-process connector is the
    process boundary -- required for CA-teardown safety (see ``_va_host_ca_op.py``).
    """
    overrides: dict[str, Any] = {
        "control_system.writes_enabled": True,
        "control_system.limits_checking.enabled": True,
        "control_system.limits_checking.database_path": "data/channel_limits.json",
        "control_system.limits_checking.allow_unlisted_channels": True,
        "project_root": str(project_dir),
    }
    return {
        "connector_config": CONNECTOR_CONFIG,
        "config_overrides": overrides,
        "read": read,
        "write": write,
        # sp-echo SP->RB propagation is async; poll the readback until it
        # reflects the write rather than race the echo (see _va_host_ca_op.py).
        "settle_read": settle_read,
    }


def _parse_host_ca_result(proc: subprocess.CompletedProcess) -> dict[str, Any]:
    """Extract the marker-prefixed JSON result line from a host CA op subprocess.

    Fails loudly (never silently passes) on a non-zero exit -- including a
    native SIGBUS (rc 138), which would mean the CA-teardown crash somehow
    reached the read/write path itself rather than being skipped.
    """
    if proc.returncode != 0:
        raise AssertionError(
            f"host CA op subprocess failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith(HOST_CA_RESULT_MARKER):
            return json.loads(line[len(HOST_CA_RESULT_MARKER) :])
    raise AssertionError(
        f"host CA op produced no {HOST_CA_RESULT_MARKER} result line:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def _run_host_ca_op(
    spec: dict[str, Any], timeout: float = HOST_CA_OP_TIMEOUT_SEC
) -> dict[str, Any]:
    """Run one host CA op to completion (blocking) and return its parsed result."""
    proc = subprocess.run(
        [sys.executable, str(HOST_CA_OP_SCRIPT), json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return _parse_host_ca_result(proc)


# ---------------------------------------------------------------------------
# P1: co-deploy — health, loopback binding, depends_on ordering
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_p1_co_deploy_health_binding_and_ordering(deployed_stack: DeployedStack) -> None:
    status, body = _get("/health")
    assert status == 200 and body.get("status") == "ok", f"bridge /health: {status} {body}"

    # A container serves HTTP /health (the fixture's readiness gate) before
    # Docker flips its healthcheck STATUS off "starting" (the healthcheck only
    # runs on its interval, after start_period) — so poll for "healthy" rather
    # than assert it the instant the fixture yields.
    _wait_for_container_health(VA_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)
    _wait_for_container_health(BRIDGE_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)

    va_ports = subprocess.run(
        ["docker", "port", VA_CONTAINER], capture_output=True, text=True, timeout=30
    )
    assert va_ports.returncode == 0, f"docker port {VA_CONTAINER} failed: {va_ports.stderr}"
    assert "127.0.0.1" in va_ports.stdout, f"VA CA port not on loopback: {va_ports.stdout!r}"
    assert "0.0.0.0" not in va_ports.stdout, (
        f"VA CA port must never bind 0.0.0.0: {va_ports.stdout!r}"
    )

    bridge_ports = subprocess.run(
        ["docker", "port", BRIDGE_CONTAINER], capture_output=True, text=True, timeout=30
    )
    assert bridge_ports.returncode == 0, (
        f"docker port {BRIDGE_CONTAINER} failed: {bridge_ports.stderr}"
    )
    assert "127.0.0.1" in bridge_ports.stdout, f"bridge not on loopback: {bridge_ports.stdout!r}"
    assert "0.0.0.0" not in bridge_ports.stdout, (
        f"bridge must never bind 0.0.0.0: {bridge_ports.stdout!r}"
    )

    # depends_on: condition: service_healthy (task 4.2) — the bridge container
    # cannot even start until the VA's healthcheck passes, so the VA's
    # StartedAt must precede the bridge's. RFC3339 UTC timestamps sort
    # lexicographically.
    va_started = _docker_inspect(VA_CONTAINER, "{{.State.StartedAt}}")
    bridge_started = _docker_inspect(BRIDGE_CONTAINER, "{{.State.StartedAt}}")
    assert va_started <= bridge_started, (
        f"expected the VA (depends_on: service_healthy) to start before the bridge — "
        f"VA StartedAt={va_started!r}, bridge StartedAt={bridge_started!r}"
    )


# ---------------------------------------------------------------------------
# P2: full-manifest CA liveness
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_p2_full_manifest_liveness(deployed_stack: DeployedStack) -> None:
    # Runs scripts/va/sweep_check.py as its OWN subprocess/CA client, exactly
    # as it's meant to be invoked against a host-published container (see its
    # module docstring) — never in-process here: this process also acts as an
    # async EPICS CA client via EPICSConnector (P3-P5), and mixing a
    # main-thread pyepics operation into a process that also drives CA off an
    # asyncio.to_thread() executor is a documented deadlock risk (see
    # tests/va/e2e/conftest.py's `_readiness_pv_served`).
    proc = subprocess.run(
        [sys.executable, str(SWEEP_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=SWEEP_TIMEOUT_SEC,
    )
    assert proc.returncode == 0, (
        f"full-manifest CA sweep failed:\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    # elapsed_s is informational only (see module docstring) — logged via the
    # sweep script's own "Connected: X/Y in Z.Zs" stdout line, never asserted.


# ---------------------------------------------------------------------------
# P3: read-equivalence — pyepics (host) vs ophyd-async (bridge)
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
async def test_p3_read_equivalence(deployed_stack: DeployedStack) -> None:
    sp, rb = deployed_stack.pairs["p3"]
    lo, hi = deployed_stack.bounds(sp)
    value = lo + 0.5 * (hi - lo)

    # Host side (pyepics), isolated in its own process: arrange a known,
    # non-default state (rather than comparing two never-written 0.0 defaults,
    # a degenerate proof), then read the sibling readback back. settle_read: the
    # SP->RB echo is asynchronous, so the host op polls the readback until it
    # reflects the write (bounded) instead of racing the echo (see
    # _va_host_ca_op.py) — the failure mode that a fixed no-wait read hits under
    # heavy load.
    host = _run_host_ca_op(
        _host_ca_op_spec(
            deployed_stack.project_dir,
            read=rb,
            write={"address": sp, "value": value},
            settle_read=True,
        )
    )
    assert host["write_success"] and host["write_verified"], (
        f"setup write to {sp} did not verify: {host}"
    )
    assert host["read_settled"], (
        f"host read of {rb} never settled to the written setpoint {value} "
        f"(last read {host['read_value']}) — sp-echo SP->RB propagation did not complete"
    )
    host_read = host["read_value"]

    # grid_scan is the catalog's minimal acquisition plan (`count` was dropped
    # with the trust-tiered registry): step the p4 scan motor through a 2-point
    # sweep and read the p3 detector at each point — the p3 pair itself is
    # never driven, so both rows sample the settled sp-echo value.
    m_sp, _ = deployed_stack.pairs["p4"]
    m_lo, m_hi = deployed_stack.bounds(m_sp)
    run_id, status_body = await _run_scan(
        "grid_scan",
        {
            "detectors": [P3_DETECTOR],
            "axes": [
                {
                    "setpoint": SCAN_MOTOR,
                    "start": m_lo + 0.25 * (m_hi - m_lo),
                    "stop": m_lo + 0.75 * (m_hi - m_lo),
                    "num_points": 2,
                }
            ],
        },
        deployed_stack.project_dir,
    )
    assert status_body.get("status") == "completed", (
        f"P3 read-equivalence scan did not complete: {status_body}"
    )

    status, data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    assert data["row_count"] == 2, f"expected one row per grid point: {data}"
    col = _find_column(data["columns"], P3_DETECTOR)
    bridge_value = data["rows"][0][col]
    assert bridge_value is not None, f"no value recorded for {P3_DETECTOR}: {data}"

    # sp-echo is a plain software copy — the host write should be exactly
    # reflected in both readers.
    assert abs(host_read - value) <= 1e-6, (
        f"host read of {rb} ({host_read}) does not match the written setpoint "
        f"({value}) — sp-echo should be an exact copy"
    )
    assert abs(host_read - bridge_value) <= 1e-6, (
        f"host (pyepics) read of {rb} = {host_read} != bridge (ophyd-async) read = {bridge_value}"
    )


# ---------------------------------------------------------------------------
# P4: concurrent scan + read — the loop-affinity falsifier
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
async def test_p4_concurrent_scan_and_read(deployed_stack: DeployedStack) -> None:
    sp, rb = deployed_stack.pairs["p4"]
    lo, hi = deployed_stack.bounds(sp)
    start = lo + 0.25 * (hi - lo)
    stop = lo + 0.75 * (hi - lo)
    num = 4

    token = _minted_token(deployed_stack.project_dir)
    status, body = _post(
        "/runs",
        {
            "plan_name": "grid_scan",
            "plan_args": {
                "detectors": [P4_DETECTOR],
                "axes": [{"setpoint": SCAN_MOTOR, "start": start, "stop": stop, "num_points": num}],
            },
        },
    )
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    status, body = _post(f"/runs/{run_id}/launch", {}, headers={"X-Launch-Token": token})
    assert status == 200, f"launch failed: {status} {body}"

    # Launch the host read in its OWN process immediately after launch, before
    # any polling sleep, so it genuinely overlaps the bridge's in-flight scan (a
    # wrong-loop/dead-monitor connect on the bridge side would stall the scan;
    # see module docstring and task 2.1). Isolating it in a subprocess is what
    # keeps the libca CA-teardown assertion from ever recurring in this process
    # (see _va_host_ca_op.py). Even if subprocess connect latency lands the read
    # after the scan settles, it still lands on a settled sp-echo step — which
    # the candidate set below accepts.
    read_proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(HOST_CA_OP_SCRIPT),
        json.dumps(_host_ca_op_spec(deployed_stack.project_dir, read=rb)),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # Own the child's lifecycle: unlike subprocess.run(timeout=...) (P3/P5),
    # asyncio.wait_for only cancels the await -- it does NOT kill the process --
    # and an early assertion below would skip communicate() entirely. Either
    # would orphan a CA-connecting subprocess (it self-terminates within the
    # connector's 5s timeout + os._exit, but a rerun under flaky() must not
    # inherit it). The finally reaps it on every non-consumed path.
    try:
        deadline = time.monotonic() + SCAN_TIMEOUT_SEC
        status_body: dict = {}
        while time.monotonic() < deadline:
            _, status_body = _get(f"/runs/{run_id}")
            if status_body.get("status") in ("completed", "error", "stopped"):
                break
            await asyncio.sleep(0.2)
        assert status_body.get("status") == "completed", f"P4 scan did not complete: {status_body}"

        stdout_b, stderr_b = await asyncio.wait_for(
            read_proc.communicate(), timeout=HOST_CA_OP_TIMEOUT_SEC
        )
        concurrent_value = _parse_host_ca_result(
            subprocess.CompletedProcess(
                args=[HOST_CA_OP_SCRIPT],
                returncode=read_proc.returncode or 0,
                stdout=stdout_b.decode(),
                stderr=stderr_b.decode(),
            )
        )["read_value"]
    finally:
        if read_proc.returncode is None:
            read_proc.kill()
            try:
                await asyncio.wait_for(read_proc.wait(), timeout=5.0)
            except (TimeoutError, ProcessLookupError):
                pass

    status, data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    assert data["row_count"] == num, f"expected {num} rows: {data}"
    col = _find_column(data["columns"], P4_DETECTOR)
    row_values = [row[col] for row in data["rows"]]
    assert len(row_values) == num and all(v is not None for v in row_values), (
        f"incomplete {P4_DETECTOR} column: {row_values}"
    )

    # The concurrent host read landed either before the first point settled
    # (the pristine 0.0 default) or during the settled window of whichever
    # point had most recently completed (sp-echo is a discrete, immediate
    # step -- never interpolated, never noisy) -- so it MUST match one of
    # these, never a value outside that set.
    candidates = [0.0, *row_values]
    assert any(abs(concurrent_value - c) <= 1e-6 for c in candidates), (
        f"concurrent host read of {rb} ({concurrent_value}) matched neither the "
        f"pristine default nor any scanned row {row_values}"
    )


# ---------------------------------------------------------------------------
# P5: honest divergence under a pre-faulted setpoint (STRICT — no flaky mark)
# ---------------------------------------------------------------------------


async def test_p5_honest_divergence_under_stuck_setpoint(deployed_stack: DeployedStack) -> None:
    sp, rb = deployed_stack.pairs["p5"]
    lo, hi = deployed_stack.bounds(sp)
    # Away from 0.0 (the RB's frozen initial value) and from the midpoints
    # P3/P4 use on their own disjoint pairs — irrelevant here, but keeps the
    # chosen value unambiguous against a stuck-at-zero readback.
    value = lo + 0.5 * (hi - lo)
    assert abs(value) > 1e-6

    # Host side (pyepics), isolated in its own process: write the pre-faulted SP,
    # then read the sibling RB back — one connect/write/read in one subprocess.
    host = _run_host_ca_op(
        _host_ca_op_spec(deployed_stack.project_dir, read=rb, write={"address": sp, "value": value})
    )
    # The SP always latches its own written value (records.py) even when stuck --
    # only the propagation to RB is dropped. write_channel's readback
    # verification re-reads the SAME channel it wrote (the SP), so a stuck-RB
    # fault is invisible to it: this MUST verify True.
    assert host["write_success"] is True, f"write to pre-faulted {sp} did not succeed: {host}"
    assert host["write_verified"] is True, (
        f"write to pre-faulted {sp} did not verify (SP always latches its own "
        f"readback regardless of the fault): {host}"
    )

    # Independent read of the SIBLING readback — this is where the fault is
    # honest: it must never have followed the SP.
    host_rb = host["read_value"]
    assert abs(host_rb - value) > 1e-6, (
        f"expected {rb} to diverge from the written setpoint {value} under "
        f"VA_STUCK_SETPOINTS, but it read {host_rb} — fault did not take effect"
    )

    # grid_scan replaces the dropped `count` builtin (see P3): drive the p4
    # scan motor, never the stuck p5 pair, and read the frozen p5 readback at
    # each of the 2 grid points.
    m_sp, _ = deployed_stack.pairs["p4"]
    m_lo, m_hi = deployed_stack.bounds(m_sp)
    run_id, status_body = await _run_scan(
        "grid_scan",
        {
            "detectors": [P5_DETECTOR],
            "axes": [
                {
                    "setpoint": SCAN_MOTOR,
                    "start": m_lo + 0.25 * (m_hi - m_lo),
                    "stop": m_lo + 0.75 * (m_hi - m_lo),
                    "num_points": 2,
                }
            ],
        },
        deployed_stack.project_dir,
    )
    assert status_body.get("status") == "completed", (
        f"P5 divergence scan did not complete: {status_body}"
    )

    status, data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    assert data["row_count"] == 2, f"expected one row per grid point: {data}"
    col = _find_column(data["columns"], P5_DETECTOR)
    bridge_rb = data["rows"][0][col]
    assert bridge_rb is not None, f"no value recorded for {P5_DETECTOR}: {data}"

    # Both independent CA clients (host pyepics, bridge ophyd-async) must
    # agree on the frozen value -- honest divergence, not a per-client one.
    assert abs(host_rb - bridge_rb) <= 1e-6, (
        f"host (pyepics) read of frozen {rb} = {host_rb} != bridge (ophyd-async) "
        f"read = {bridge_rb} — the two CA clients disagree on the stuck readback"
    )
