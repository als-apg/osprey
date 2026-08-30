"""Phase 1 acceptance proof: the Virtual Accelerator + Bluesky bridge substrate,
co-deployed as real containers, is equivalent to (and honest about divergence
from) a real EPICS beamline (PROPOSAL.md's Risk-1/Station-2 gate).

One module-scoped init + build + ``osprey up -d --dev`` co-deploys the
Virtual Accelerator (task 4.1) and the Bluesky bridge (task 2.9) wired to the
EPICS substrate scanner (task 2.3), with one sp-echo ``:SP`` pre-faulted
(task 3.1's ``VA_STUCK_SETPOINTS``) via task 4.2's env passthrough. Five
proofs then exercise the whole stack end to end:

  P1 co-deploy:     containers up, healthy, loopback-only, depends_on ordering held.
  P2 liveness:      the full manifest namespace is reachable over CA.
  P3 read-equiv:    a pyepics (host) read and an ophyd-async (bridge) read of
                     the same PV agree.
  P4 concurrent:    an EPICS-substrate ``grid_scan`` plan runs to completion
                     while a concurrent host read observes the same PV
                     consistently — the loop-affinity falsifier.
  P5 honest divergence: a write to a pre-faulted ``:SP`` is confirmed (the SP
                     always latches its own readback), but an independent read
                     of the sibling ``:RB`` proves it never moved — and both
                     CA clients (host + bridge) agree on that frozen value.

Plans reach the hardware through the bridge's QUEUE, not a direct-execute
route: P3/P4 stage the plan in the shared draft, enqueue that exact revision,
and arm ``POST /queue/start`` with the launch token (``PATCH /draft`` ->
``POST /queue/items`` -> ``POST /queue/start`` -> poll ``GET /runs/{id}``,
spelled once in ``tests/e2e/_queue_drive.py``). That is transport only — what
these proofs assert about the substrate is unchanged.

No preset channel names are hardcoded: every address used below is derived
from the deployment repo's own ``data/channel_limits.json`` — the same bytes
the build copies into the build zone for the deployed containers (writable ⟺ a
``:SP`` address) restricted to sp-echo pairs (``classify_partition`` — a
write to a pyat-coupled ``:SP`` has ring-wide physics side effects, wrong for
an isolated fault/equivalence probe; sp-echo is a pure software echo, exactly
what P3-P5 need).

Container safety: every docker invocation below names an exact container/image
— never a wildcard, never ``system prune``/``--volumes``. The one forced
``docker rmi -f <image>`` (below) names an exact image, matching
``test_bluesky_deploy.py``'s precedent for forcing a fresh ``--dev`` build.
Teardown goes through ``osprey down``, never a raw ``docker rm`` sweep,
followed by exact-named removal of this project's own volumes
(``tests/e2e/_volumes.py``): ``down`` keeps them by design, and a rerun must
not inherit their state.

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
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.compose_generator import resolve_project_name
from tests.e2e import _orm_stack, _queue_drive
from tests.e2e._deploy_diagnostics import dead_container_logs, queue_stack_logs
from tests.e2e._volumes import remove_project_volumes

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "va" / "sweep_check.py"
# Out-of-process host-side CA op (see its module docstring): each P3-P5 host
# read/write runs in its own short-lived process so the libca CA-teardown
# assertion can never recur in this pytest process.
HOST_CA_OP_SCRIPT = Path(__file__).resolve().parent / "_va_host_ca_op.py"
# Must match _va_host_ca_op.RESULT_MARKER (kept as a local literal rather than
# imported -- tests/e2e is a package, so the helper is not on sys.path).
HOST_CA_RESULT_MARKER = "__HOST_CA_RESULT__"


# Channel Access port the Virtual Accelerator serves on. An ephemeral free
# port, not 5064: this module already plumbs the one value everywhere it
# matters (`--set virtual_accelerator.port=` at init, and the name-server
# gateway below), and 5064 on a dev host belongs to whatever real deployment
# the operator is running — `port_layout.CA_DEFAULT_PORT` keeps VA instance 1
# there on purpose. The shipped 5064 default is covered at render level by
# tests/cli/test_va_default_config.py and tests/cli/test_rendered_va_block.py.
def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# import-time required because VA_CA_PORT seeds module-level constants
# (_VA_GATEWAY, the fixture's --set args) built while the module imports.
VA_CA_PORT = _reserve_free_port()
# The deployment repo's directory name IS the deployment's name; the compose
# templates render each service's container_name AND its locally-built image as
# ``<project>-<service>`` (services/*/docker-compose.yml.j2), so derive both
# (via resolve_project_name, exactly as the templates do) rather than hardcode
# host-global names that break the moment the templates are namespaced per-project.
PROJECT_NAME = "proj"
VA_CONTAINER = f"{PROJECT_NAME}-virtual-accelerator"
VA_IMAGE = f"{resolve_project_name({'project_name': PROJECT_NAME})}-va:local"

# Deliberately non-default (avoids colliding with test_bluesky_deploy.py's 18090
# on a shared dev machine — see that module's docstring).
BRIDGE_PORT = 18099
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"
BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"
BRIDGE_IMAGE = f"{resolve_project_name({'project_name': PROJECT_NAME})}-bluesky-bridge:local"

# Device names this suite authors into the worker's device file — arbitrary,
# resolved against explicit PV addresses (see _write_devices_file below), never
# a preset naming convention. Synthetic on purpose: this is the one lane that
# proves a device name need not BE its address, which is exactly what the
# ``settables``/``readables`` entries' ``setpoint``/``pv`` fields are for.
SCAN_MOTOR = "scan_motor"
P3_DETECTOR = "p3_det"
P4_DETECTOR = "p4_det"
P5_DETECTOR = "p5_det"

# The bridge's arming route (POST /queue/start) fails closed on an unset
# BLUESKY_LAUNCH_TOKEN. `osprey up` mints one for the deployed bluesky
# service, but this e2e supplies its own explicitly (the supported
# operator-provides-a-token path) so the test knows the token value up front
# and never has to read it back out of the repo's .env.
LAUNCH_TOKEN = "e2e-substrate-equivalence-launch-token"

# Identifies this suite as the draft's writer on every PATCH /draft frame. The
# draft is a single shared document, so a client id that names the writer is
# what makes a stray edit attributable.
_QUEUE_CLIENT_ID = "va-substrate-equivalence-e2e"

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


def _channel_limits(repo: Path) -> dict[str, Any]:
    """The deployment repo's own channel limits.

    ``osprey build`` copies ``<repo>/data`` into the build zone verbatim, so
    this file and the ``build/data/`` copy the bridge and the VA both read are
    the same bytes and name the same channels — but only this one exists before
    the build, which is when the plan devices have to be chosen and authored.
    """
    return json.loads((repo / "data" / "channel_limits.json").read_text(encoding="utf-8"))


def _select_sp_echo_pairs(channel_limits: dict[str, Any], count: int) -> list[tuple[str, str]]:
    """Derive ``count`` disjoint sp-echo (``:SP``, ``:RB``) pairs from the
    deployed render's own channel_limits.json -- no hardcoded preset
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


def _write_devices_file(repo: Path, pairs: dict[str, tuple[str, str]]) -> None:
    """Author this suite's plan devices at ``<repo>/data/bluesky_devices.yml``
    -- BETWEEN ``osprey init`` and ``osprey build``.

    The build copies ``<repo>/data`` into the build zone and stages the device
    file it finds there into ``build/services/bluesky/bluesky_devices.yml``,
    which the queueserver worker mounts. Written after the build, this file
    would be picked up by nothing; written before ``init``, it would break
    init's own copy of the preset's ``data/``.

    Assembled here rather than through
    ``osprey.services.bluesky_bridge.substrate_devices`` (which
    ``_orm_stack.write_devices_file`` delegates to) because THIS suite's whole
    point is synthetic device names: that producer names every device after its
    own address, and P3/P4/P5 have to stay addressable under names the
    equivalence assertions choose. The document SHAPE is still the product's --
    its key names are imported, not restated -- so a schema change breaks this
    lane rather than silently producing a file the worker skips.
    """
    from osprey.services.bluesky_bridge.devices._specs_from_file import (
        READABLES_KEY,
        SETTABLES_KEY,
    )

    p3_sp, p3_rb = pairs["p3"]
    p4_sp, p4_rb = pairs["p4"]
    p5_sp, p5_rb = pairs["p5"]

    document = {
        SETTABLES_KEY: [{"name": SCAN_MOTOR, "setpoint": p4_sp, "readback": p4_rb}],
        READABLES_KEY: [
            {"name": P3_DETECTOR, "pv": p3_rb},
            {"name": P4_DETECTOR, "pv": p4_rb},
            {"name": P5_DETECTOR, "pv": p5_rb},
        ],
    }

    devices_path = repo / "data" / "bluesky_devices.yml"
    devices_path.parent.mkdir(parents=True, exist_ok=True)
    devices_path.write_text(
        yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _write_env(repo: Path, pairs: dict[str, tuple[str, str]]) -> None:
    """Append this suite's contract env vars to the repo's ``.env`` -- BEFORE
    ``osprey up`` (the bridge/VA compose templates pass these through from the
    repo root's ``.env``).

    Creating that file is also what lets ``up`` run at all: the repo root's
    ``.env`` is the deployment's whole secret store, and ``up`` aborts when it
    is missing.

    Only two values, and neither is a device: the plan devices moved out of the
    environment and into the mounted device file (``_write_devices_file``), so
    what is left here is the launch token and the VA's stuck-channel fault.
    """
    _p5_sp, _p5_rb = pairs["p5"]

    values = {
        # Supply the launch token ourselves — the preset's local-exec+writes
        # config gates auto-minting off (see LAUNCH_TOKEN above).
        "BLUESKY_LAUNCH_TOKEN": LAUNCH_TOKEN,
        "VA_STUCK_SETPOINTS": _p5_sp,
    }

    env_path = repo / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    new_lines = "".join(f"{k}={v}\n" for k, v in values.items())
    env_path.write_text(existing + new_lines, encoding="utf-8")


class DeployedStack:
    """Everything the P1-P5 tests need about the one co-deployed repo."""

    def __init__(self, repo: Path, pairs: dict[str, tuple[str, str]], limits: dict[str, Any]):
        self.repo = repo
        self.pairs = pairs
        self.limits = limits

    def bounds(self, address: str) -> tuple[float, float]:
        entry = self.limits[address]
        return float(entry["min_value"]), float(entry["max_value"])


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedStack]:
    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("va_substrate_build")
    repo = base / PROJECT_NAME

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
    # web-terminal stack on by default, so an unqualified deploy would also
    # render both persona projects, build two web images, and start nginx/web
    # containers -- none of which this substrate-equivalence proof exercises
    # (that topology is covered by test_control_assistant_demo.py).
    # `_orm_stack.VA_ARCHIVER_CI_KNOBS` shrinks the archive the preset declares
    # to a CI-sized one -- this proof deploys the store (the preset's
    # `va_archiver:` block is what makes the VA's history real) but reads none
    # of its history, and seeding a tutorial-sized month costs every run.
    override_path = base / "override.yml"
    override_path.write_text(
        "config:\n"
        "  control_system.type: virtual_accelerator\n"
        "  modules.web_terminals.enabled: false\n"
        "dispatch: null\n" + _orm_stack.VA_ARCHIVER_CI_KNOBS,
        encoding="utf-8",
    )

    # Two steps, because the surface has two: `init` writes the repo's source
    # zone from the preset, `build` renders build/ from it. The repo directory
    # name IS the deployment name, so the container names above still hold.
    init = _run(
        [
            str(osprey_bin),
            "init",
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--override",
            str(override_path),
            "--set",
            f"virtual_accelerator.port={VA_CA_PORT}",
            "--set",
            f"bluesky.port={BRIDGE_PORT}",
            # This module's own thousand-port block (see
            # test_dispatch_deploy.py's 20700 note): everything not pinned
            # explicitly follows it instead of landing on a real deployment's
            # default 10000 block.
            "--set",
            "port_base=21500",
        ],
        cwd=base,
        timeout=BUILD_TIMEOUT_SEC,
    )
    if init.returncode != 0:
        pytest.fail(
            f"osprey init failed (rc={init.returncode}):\n"
            f"--- stdout ---\n{init.stdout}\n--- stderr ---\n{init.stderr}"
        )

    # STRICTLY between the two verbs -- see _write_devices_file. The pairs come
    # from the repo's own channel limits, the same bytes the build is about to
    # copy into the build zone.
    limits = _channel_limits(repo)
    sp3, sp4, sp5 = _select_sp_echo_pairs(limits, count=3)
    pairs = {"p3": sp3, "p4": sp4, "p5": sp5}
    _write_devices_file(repo, pairs)

    build = _run(
        [str(osprey_bin), "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        cwd=base,
        timeout=BUILD_TIMEOUT_SEC,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    _write_env(repo, pairs)

    # Force fresh --dev builds so the deployed containers run CURRENT source
    # (osprey up does not pass --build to compose, so it would otherwise reuse a
    # stale cached image). Exact-named images only.
    # E2E_REUSE_IMAGES=1 skips this (dev-only: fast local iteration on the test
    # itself when the osprey source is unchanged; never set it in CI, where a
    # source change must always rebuild). The first-time native VA build is slow.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(["docker", "rmi", "-f", VA_IMAGE], capture_output=True, text=True)
        subprocess.run(["docker", "rmi", "-f", BRIDGE_IMAGE], capture_output=True, text=True)

    try:
        up = _run(
            [str(osprey_bin), "up", "-d", "--dev"],
            cwd=repo,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}\n"
                f"--- containers that are not running ---\n{_dead_container_logs()}"
            )
        try:
            _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        except AssertionError as exc:
            pytest.fail(f"{exc}\n--- containers that are not running ---\n{_dead_container_logs()}")
        # HTTP readiness is not enqueue readiness -- the worker namespace the
        # enqueue validates against exists only once the RE worker environment
        # is open, and the bridge opens that off the readiness path. See
        # `_queue_drive.wait_for_worker_environment`. Its own diagnostic is the
        # two RUNNING containers that own the environment, which
        # `_dead_container_logs` skips by design.
        try:
            _queue_drive.wait_for_worker_environment(BRIDGE_URL)
        except AssertionError as exc:
            pytest.fail(f"{exc}\n{queue_stack_logs(_orm_stack.project_prefix(PROJECT_NAME))}")
        yield DeployedStack(repo=repo, pairs=pairs, limits=limits)
    finally:
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        # `osprey down` keeps volumes by design; drop this project's own so a
        # rerun cannot inherit their state (see tests/e2e/_volumes.py).
        remove_project_volumes(_orm_stack.project_prefix(PROJECT_NAME))


def _dead_container_logs() -> str:
    """Logs from every container of this deployment that is not running."""
    return dead_container_logs(resolve_project_name({"project_name": PROJECT_NAME}))


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


def _minted_token(repo: Path) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = repo / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    env = parse_dotenv_file(env_path)
    token = env.get("BLUESKY_LAUNCH_TOKEN")
    assert token, "BLUESKY_LAUNCH_TOKEN missing/empty in the deployment repo's .env"
    return token


def _get(path: str) -> tuple[int, dict]:
    req = urllib.request.Request(f"{BRIDGE_URL}{path}", method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
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
    plan_name: str, plan_args: dict, repo: Path, timeout: float = SCAN_TIMEOUT_SEC
) -> tuple[str, dict]:
    """Stage -> enqueue -> armed start -> poll. Returns (run_id, final_status_body).

    The three writes go through `_queue_drive`, the one copy of the queue flow.
    The poll stays here, and stays ``async``: P4 runs a host-side CA read
    concurrently with the run, so this module's loop must never be blocked
    while a plan is under way.
    """
    token = _minted_token(repo)
    run_id = _queue_drive.stage_and_enqueue(
        BRIDGE_URL, plan_name, plan_args, client_id=_QUEUE_CLIENT_ID
    )
    _queue_drive.start_queue(BRIDGE_URL, token)

    deadline = time.monotonic() + timeout
    last_status_body: dict = {}
    while time.monotonic() < deadline:
        _, last_status_body = _get(f"/runs/{run_id}")
        if last_status_body.get("status") in _queue_drive.TERMINAL_STATUSES:
            break
        await asyncio.sleep(0.2)
    return run_id, last_status_body


def _host_ca_op_spec(
    repo: Path,
    *,
    read: str,
    write: dict[str, Any] | None = None,
    settle_read: bool = False,
) -> dict[str, Any]:
    """Build the JSON spec for one out-of-process host CA op (``_va_host_ca_op.py``).

    Carries the SAME ``get_config_value`` overrides the in-process connector
    used -- ``project_root`` is the deployment repo and
    ``limits_checking.database_path`` names the RENDER's channel_limits.json, so
    ``LimitsValidator`` enforces the very file this test selected channels from
    (these proofs write only LISTED sp-echo ``:SP`` channels, so limits are
    actually applied to them). Spelled absolute rather than repo-relative: a
    relative ``database_path`` resolves against ``CONFIG_FILE``'s directory when
    that is set and against ``project_root`` otherwise, and this subprocess sets
    neither anchor to the render.

    The permissive half of the posture is spelled PER CONNECTOR TYPE, on
    ``virtual_accelerator`` -- the type this proof drives -- so the
    deployment-wide block keeps the strict posture a live machine deserves and
    this lane exercises the per-type override rather than relaxing everything.

    ``CONNECTOR_CONFIG`` is passed verbatim so the subprocess builds a REAL
    production ``VirtualAcceleratorConnector`` via ``ConnectorFactory`` under
    test-supplied config. Config keys these proofs don't exercise fall to code
    defaults -- inert here, since every write passes ``confirm=True`` explicitly
    rather than resolving the limits database's ``confirm`` field. The only
    thing that changes vs. an in-process connector is the
    process boundary -- required for CA-teardown safety (see ``_va_host_ca_op.py``).
    """
    overrides: dict[str, Any] = {
        "control_system.writes_enabled": True,
        "control_system.limits_checking.enabled": True,
        "control_system.limits_checking.database_path": str(
            repo / "build" / "data" / "channel_limits.json"
        ),
        # BOTH per-type leaves, deliberately: a per-type ``limits_checking``
        # block overrides the deployment-wide pair as a WHOLE, so one leaf on
        # its own is an incomplete block -- which resolves to the failsafe
        # validator, refuses every write, and turns this lane red. Spelled flat
        # and dotted like the rest of this map; the subprocess shim assembles
        # the nested ``control_system`` section the resolver reads.
        "control_system.connector.virtual_accelerator.limits_checking.enabled": True,
        "control_system.connector.virtual_accelerator.limits_checking"
        ".allow_unlisted_channels": True,
        "project_root": str(repo),
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
    # The sweep script defaults EPICS_CA_NAME_SERVERS to localhost:5064; this
    # stack serves CA on the module's ephemeral VA_CA_PORT, so the subprocess
    # must be told explicitly (the in-process connectors get it via
    # _VA_GATEWAY instead).
    proc = subprocess.run(
        [sys.executable, str(SWEEP_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=SWEEP_TIMEOUT_SEC,
        env={
            **os.environ,
            "EPICS_CA_NAME_SERVERS": f"localhost:{VA_CA_PORT}",
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
        },
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
            deployed_stack.repo,
            read=rb,
            write={"address": sp, "value": value},
            settle_read=True,
        )
    )
    assert host["write_outcome"] == "confirmed", f"setup write to {sp} was not confirmed: {host}"
    assert host["read_settled"], (
        f"host read of {rb} never settled to the written setpoint {value} "
        f"(last read {host['read_value']}) — sp-echo SP->RB propagation did not complete"
    )
    host_read = host["read_value"]

    # grid_scan is the catalog's minimal acquisition plan (`count` was dropped
    # with the trust-tiered registry): step the p4 scan setpoint through a 2-point
    # sweep and read the p3 readback at each point — the p3 pair itself is
    # never driven, so both rows sample the settled sp-echo value.
    m_sp, _ = deployed_stack.pairs["p4"]
    m_lo, m_hi = deployed_stack.bounds(m_sp)
    run_id, status_body = await _run_scan(
        "grid_scan",
        {
            "readbacks": [P3_DETECTOR],
            "axes": [
                {
                    "setpoint": SCAN_MOTOR,
                    "start": m_lo + 0.25 * (m_hi - m_lo),
                    "stop": m_lo + 0.75 * (m_hi - m_lo),
                    "num_points": 2,
                }
            ],
        },
        deployed_stack.repo,
    )
    assert status_body.get("status") == "completed", (
        f"P3 read-equivalence run did not complete: {status_body}"
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
# P4: concurrent run + read — the loop-affinity falsifier
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
async def test_p4_concurrent_scan_and_read(deployed_stack: DeployedStack) -> None:
    sp, rb = deployed_stack.pairs["p4"]
    lo, hi = deployed_stack.bounds(sp)
    start = lo + 0.25 * (hi - lo)
    stop = lo + 0.75 * (hi - lo)
    num = 4

    # Driven step by step rather than through `_run_scan`: the host read below
    # has to be spawned between the armed start and the first poll, so this
    # proof needs the start and the polling loop separated.
    token = _minted_token(deployed_stack.repo)
    run_id = _queue_drive.stage_and_enqueue(
        BRIDGE_URL,
        "grid_scan",
        {
            "readbacks": [P4_DETECTOR],
            "axes": [{"setpoint": SCAN_MOTOR, "start": start, "stop": stop, "num_points": num}],
        },
        client_id=_QUEUE_CLIENT_ID,
    )
    _queue_drive.start_queue(BRIDGE_URL, token)

    # Launch the host read in its OWN process immediately after the start, before
    # any polling sleep, so it genuinely overlaps the bridge's in-flight run (a
    # wrong-loop/dead-monitor connect on the bridge side would stall the run;
    # see module docstring and task 2.1). Isolating it in a subprocess is what
    # keeps the libca CA-teardown assertion from ever recurring in this process
    # (see _va_host_ca_op.py). Even if subprocess connect latency lands the read
    # after the run settles, it still lands on a settled sp-echo step — which
    # the candidate set below accepts.
    read_proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(HOST_CA_OP_SCRIPT),
        json.dumps(_host_ca_op_spec(deployed_stack.repo, read=rb)),
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
            if status_body.get("status") in _queue_drive.TERMINAL_STATUSES:
                break
            await asyncio.sleep(0.2)
        assert status_body.get("status") == "completed", f"P4 run did not complete: {status_body}"

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
        f"pristine default nor any row from the run {row_values}"
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
        _host_ca_op_spec(deployed_stack.repo, read=rb, write={"address": sp, "value": value})
    )
    # The SP always latches its own written value (records.py) even when stuck --
    # only the propagation to RB is dropped. write_channel confirms by re-reading
    # the SAME channel it wrote (the SP), so a stuck-RB fault is invisible to it:
    # the outcome MUST be `confirmed`.
    assert host["write_outcome"] == "confirmed", (
        f"write to pre-faulted {sp} was not confirmed (SP always latches its own "
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
    # scan setpoint, never the stuck p5 pair, and read the frozen p5 readback at
    # each of the 2 grid points.
    m_sp, _ = deployed_stack.pairs["p4"]
    m_lo, m_hi = deployed_stack.bounds(m_sp)
    run_id, status_body = await _run_scan(
        "grid_scan",
        {
            "readbacks": [P5_DETECTOR],
            "axes": [
                {
                    "setpoint": SCAN_MOTOR,
                    "start": m_lo + 0.25 * (m_hi - m_lo),
                    "stop": m_lo + 0.75 * (m_hi - m_lo),
                    "num_points": 2,
                }
            ],
        },
        deployed_stack.repo,
    )
    assert status_body.get("status") == "completed", (
        f"P5 divergence run did not complete: {status_body}"
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
