"""Pre-flight reachability, proved on the real VA-backed plan stack.

A plan that names a channel no IOC serves used to reach the RunEngine and fail
*mid-run*: the connector's read raised ``ConnectionError`` after its 5 s
timeout, with setpoints already applied and the plan half executed. The worker
plan wrappers now ask the connector, one message before the plan is even
built, whether every address the run declares answers, and refuse the run when
one does not (``osprey.services.bluesky_bridge.preflight.probe_before_motion``).

This module is that gate's end-to-end proof, on the deployed substrate rather
than against a stub connector: real containers, a real Virtual Accelerator
serving a real namespace, and a plan submitted through the queue exactly as an
operator submits one.

  R1 refused-before-motion: a plan naming an address the VA does not serve is
     refused; the refusal names the channel, the lane and the target; and an
     independent host-side Channel Access read of the swept corrector's ``:RB``
     proves the run moved nothing. STRICT — no rerun decorator (see Markers).
  R2 reachable plan unchanged: the same stack, the same corrector, a plan whose
     declared channels all answer, runs to completion and returns its rows. The
     gate refuses dead plans without refusing live ones.

WHAT IS NOT PROVED HERE, AND WHY. The third case worth having is the flaky
channel: one that fails its first probe and answers the retry, which the sweep
must NOT turn into a refused run (a loaded gateway dropping one probe is not a
dead machine). Producing that deterministically needs a channel whose *second*
answer differs from its first, inside a sub-second window — the retry runs as
soon as the first pass finishes — and the Virtual Accelerator offers no seam
for it: its namespace is fixed at startup, ``VA_STUCK_SETPOINTS`` freezes an
echo rather than a connection, and pausing the container would fail every
address at once with no control over when. Engineering it out of container
timing would produce a flaky test of a retry, which is worse than no test. So
that case stays at the seam where it IS deterministic — a stub connector whose
answer is scripted per call — in
``tests/services/bluesky_bridge/test_preflight.py``
(``test_flaky_address_answering_the_retry_is_not_reported``, and
``test_dead_address_is_reported_after_the_retry`` for its converse). This
module deliberately does not restate it.

Device-file route: the plan devices come from an AUTHORED
``<repo>/data/bluesky_devices.yml``, written between ``osprey init`` and
``osprey build`` (``_orm_stack.write_devices_file``, the same producer the
build's own turn-key derivation uses) and then given one hand-added entry for
the dead address. That is two proofs in one file: it is how the unserved
address gets into the worker's namespace at all, and it is the
authored-file-wins contract — R1 asserts the entry this suite added by hand
survives into the staged ``build/services/bluesky/bluesky_devices.yml`` the
worker mounts, which it only can if the authored file beat the roster-derived
derivation.

No preset channel name is hardcoded. The corrector and BPM are selected from
the deployment's own channel roster (``_orm_stack.roster_records`` ->
``select_correctors``/``select_bpms``), and the dead address is derived from
the selected BPM by moving its device index out of the range the facility has
— a plausible name in a real family that the manifest cannot serve — and
checked against the roster to be sure it is genuinely absent.

Container safety: every docker invocation below names an exact
container/image -- never a wildcard, never ``system prune``/``--volumes``.
Teardown goes through ``osprey down``, matching every other e2e in this
directory, followed by exact-named removal of this project's own volumes
(``tests/e2e/_volumes.py``): ``down`` keeps them by design, and a rerun must
not inherit their state.

Gating: needs Docker; the VA image builds natively for the host arch, so on
Apple Silicon PyAT/softioc compile from source (no prebuilt aarch64 wheels) --
slow (minutes) on a cold image cache. Lives in ``tests/e2e/`` (never collected
by the fast lane, see ``ci_check.sh``/ci.yml). Run locally with
``E2E_REUSE_IMAGES=1`` set for fast iteration once the image cache is warm.

Markers: ``e2e``/``slow``/docker-gated, matching
``test_va_substrate_equivalence.py``. No module-level ``flaky``: R1 is the
safety proof and a lenient rerun would let a refusal that fired for the wrong
reason pass on its second attempt — the rerun decorator is applied
per-function, to R2 only, exactly as that module applies it to P1-P4 and not
to P5. No ``xdist_group`` either, for the same reason its siblings carry none:
an ungrouped file goes to a single worker as a unit under both ``loadfile``
and ``loadgroup``, which is all this module needs (it owns its own bridge
port, its own thousand-port block, and an ephemeral CA port).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e import _orm_stack, _queue_drive
from tests.e2e._deploy_diagnostics import queue_stack_logs
from tests.e2e._volumes import remove_project_volumes

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    # harness_benchmark: the pre-flight refusal is a harness property -- the
    # roster decides, not the model-under-test.
    pytest.mark.harness_benchmark,
    # dockerbuild: full VA/bridge image build + deploy -- runs in the
    # dedicated orm-roundtrip-e2e CI job, never the shared e2e-tests lane
    # (the marker->--ignore pairing is enforced by
    # tests/deployment/test_ci_workflow_wiring.py).
    pytest.mark.dockerbuild,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

#: Compose project this suite deploys under. Container names and locally-built
#: image tags are both ``<project>-<service>``, so the forced image refresh and
#: the log dumps derive theirs from this one constant
#: (``_orm_stack.project_prefix``) rather than repeating a literal.
PROJECT_NAME = "preflight-probe"

#: Distinct from every other e2e module's pinned bridge port
#: (test_bluesky_deploy.py's 18090, test_va_substrate_equivalence.py's 18099,
#: test_tiled_roundtrip.py's 18101, _orm_stack.py's 18102,
#: test_bluesky_catalog_e2e.py's 18103, test_grid_scan_roundtrip.py's 18104,
#: test_bluesky_sandbox_escape_e2e.py's 18105, test_bluesky_web_deploy.py's
#: 18106) so this can run concurrently with any of them on a shared dev machine.
BRIDGE_PORT = 18107
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

#: This module's own thousand-port block (see test_dispatch_deploy.py's 20700
#: note): everything not pinned explicitly follows it instead of landing on a
#: real deployment's default 10000 block. 21200-22000 are taken by the other
#: e2e lanes.
PORT_BASE = 22100

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 1200  # first-time native VA source build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0

#: Deadline for one plan to reach a terminal status. Generous over both shapes
#: this module runs: a refused plan pays a probe timeout plus its retry on the
#: dead address (~10 s, bounded by ``preflight.PROBE_BUDGET_S``), and a healthy
#: two-point grid pays two corrector steps. Wide enough that a slow container
#: is not read as a hang, tight enough to still be a "did it hang" gate.
SCAN_TIMEOUT_SEC = 180.0

#: Out-of-process host-side CA read (see ``_va_host_ca_op.py``): each read runs
#: in its own short-lived process so the libca CA-teardown assertion can never
#: recur in this pytest process.
HOST_CA_OP_SCRIPT = Path(__file__).resolve().parent / "_va_host_ca_op.py"
#: Must match ``_va_host_ca_op.RESULT_MARKER`` (kept as a local literal rather
#: than imported -- tests/e2e is a package, so the helper is not on sys.path).
HOST_CA_RESULT_MARKER = "__HOST_CA_RESULT__"
#: Process spawn + connector connect (name-server TCP) + one read round trip.
HOST_CA_OP_TIMEOUT_SEC = 60.0

#: Host-side connector config: points at the co-deployed VA over CA
#: name-server/TCP mode -- the one host<->container CA configuration proven to
#: work across container runtimes (mirrors test_va_substrate_equivalence.py's
#: identical block, aimed at the port ``_orm_stack`` reserved for this deploy).
_VA_GATEWAY = {"address": "localhost", "port": _orm_stack.VA_CA_PORT, "use_name_server": True}
CONNECTOR_CONFIG: dict[str, Any] = {
    "type": "virtual_accelerator",
    "connector": {
        "virtual_accelerator": {
            "timeout": 5.0,
            "gateways": {"read_only": _VA_GATEWAY, "write_access": _VA_GATEWAY},
        }
    },
}

#: Device index for the unserved address. The demo facility's BPM devices run
#: 01-72, so 99 is a well-formed name in a real family that the VA manifest
#: cannot serve -- the shape of a typo or a decommissioned device, which is the
#: realistic way an operator meets this gate. Asserted absent from the roster
#: at authoring time rather than assumed.
DEAD_DEVICE_INDEX = "99"

#: Two grid points is the smallest real sweep: enough for the healthy plan to
#: produce one row per point, and enough for the refused plan to have had
#: somewhere to move to had the gate not stopped it.
NUM_POINTS = 2

#: Where in the corrector's own limits band the sweep runs. Off-centre on
#: purpose: the corrector's pristine readback is 0.0, mid-band, so a sweep at
#: 60-80% of the band is unambiguously somewhere the readback has not been --
#: which is what makes "the readback did not move" a statement with content.
AXIS_START_FRACTION = 0.6
AXIS_STOP_FRACTION = 0.8

#: Tolerance for comparing two CA reads of the same unmoved channel. The VA
#: serves doubles and nothing writes between the two reads, so this is float
#: formatting slack, not a physics tolerance (same value the sibling proofs
#: use for their sp-echo equality assertions).
READ_TOLERANCE = 1e-6

_QUEUE_CLIENT_ID = "preflight-probe-e2e"


def _get(path: str) -> tuple[int, Any]:
    return _queue_drive.request(BRIDGE_URL, path, "GET")


def _unserved_address(bpm_address: str, roster_addresses: frozenset[str]) -> str:
    """A plausible address in a real family that the deployment cannot serve.

    Built from a BPM readback the roster DID enumerate by moving its device
    index out of the facility's range, so every other segment -- ring, system,
    family, field, subfield -- is one the manifest genuinely uses. A name
    invented from nothing would prove less: it could be refused for being
    unparseable rather than for being unreachable.

    Args:
        bpm_address: A 6-part colon address the roster enumerated.
        roster_addresses: Every address the roster enumerated, to check the
            result against.

    Returns:
        The unserved address.

    Raises:
        AssertionError: If ``bpm_address`` does not follow the 6-part grammar,
            or if the result turns out to be a channel the facility has after
            all.
    """
    parts = bpm_address.split(":")
    assert len(parts) == 6, (
        f"expected a 6-part colon address to derive an unserved one from, got {bpm_address!r}"
    )
    parts[3] = DEAD_DEVICE_INDEX
    dead = ":".join(parts)
    assert dead not in roster_addresses, (
        f"{dead} is a channel this deployment actually has -- it cannot stand in for an "
        f"address no IOC serves. Pick a device index outside the facility's range."
    )
    return dead


def _add_unserved_readable(repo: Path, address: str) -> None:
    """Hand-add one readable naming ``address`` to the authored device file.

    The rest of the file is the product's own output
    (``_orm_stack.write_devices_file`` -> ``substrate_devices``), and this adds
    the one entry no producer would ever emit: a device whose channel does not
    exist. Loading and re-dumping rather than appending text, because the entry
    has to land under ``readables`` wherever that key sits; the generated
    header is carried across verbatim so the staged file still names the roster
    it is a projection of.
    """
    from osprey.cli.build_profile_schema import BlueskyConfig
    from osprey.services.bluesky_bridge.devices._specs_from_file import READABLES_KEY

    devices_path = repo / BlueskyConfig.devices_file
    raw = devices_path.read_text(encoding="utf-8")
    header = ""
    for line in raw.splitlines(keepends=True):
        if line.startswith("#") or not line.strip():
            header += line
        else:
            break

    document = yaml.safe_load(raw)
    document.setdefault(READABLES_KEY, []).append({"name": address, "pv": address})
    devices_path.write_text(
        header + f"# EDITED BY tests/e2e/test_preflight_probe.py: one readable naming {address},\n"
        f"# a channel this facility does not serve, was added by hand below.\n"
        + yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _host_read(repo: Path, address: str) -> float:
    """One host-side Channel Access read of ``address``, out of process.

    The independent observer this module's motion proof needs: the bridge is
    the thing under test, so "nothing moved" has to be checked by a CA client
    that is not it. Runs through the real production
    ``VirtualAcceleratorConnector`` built by ``ConnectorFactory``, isolated in
    its own process purely for CA-teardown safety (see ``_va_host_ca_op.py``).

    Read-only: no write posture is granted in the overrides, so this cannot
    move anything even by accident, which is the right shape for a witness.
    """
    spec = {
        "connector_config": CONNECTOR_CONFIG,
        "config_overrides": {"project_root": str(repo)},
        "read": address,
        "write": None,
        "settle_read": False,
    }
    proc = subprocess.run(
        [sys.executable, str(HOST_CA_OP_SCRIPT), json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=HOST_CA_OP_TIMEOUT_SEC,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"host CA read of {address} failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith(HOST_CA_RESULT_MARKER):
            return float(json.loads(line[len(HOST_CA_RESULT_MARKER) :])["read_value"])
    raise AssertionError(
        f"host CA read of {address} produced no {HOST_CA_RESULT_MARKER} result line:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


class DeployedPreflightStack:
    """Everything the two proofs need about the one co-deployed repo."""

    def __init__(
        self,
        repo: Path,
        corrector_setpoint: str,
        corrector_readback: str,
        bpm_name: str,
        dead_address: str,
    ):
        self.repo = repo
        self.corrector_setpoint = corrector_setpoint
        self.corrector_readback = corrector_readback
        self.bpm_name = bpm_name
        self.dead_address = dead_address

    def axis(self) -> tuple[float, float]:
        """The sweep's start/stop, inside the corrector's own limits band.

        Read from the deployment's ``channel_limits.json`` rather than
        hardcoded: the band is a property of THIS facility's corrector family,
        and a value outside it would be refused by the software limits layer
        for a reason that has nothing to do with reachability.
        """
        entry = _orm_stack.channel_limits(self.repo)[self.corrector_setpoint]
        low, high = float(entry["min_value"]), float(entry["max_value"])
        span = high - low
        return low + AXIS_START_FRACTION * span, low + AXIS_STOP_FRACTION * span


@pytest.fixture(scope="module")
def preflight_stack(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DeployedPreflightStack]:
    base = tmp_path_factory.mktemp("preflight_probe_build")

    # The plan devices are authored BETWEEN `init` and `build`: the build copies
    # <repo>/data into the build zone and stages the device file it finds there
    # for the queueserver worker, so a set written after the build would never
    # reach a container.
    correctors: dict[str, tuple[str, str]] = {}
    bpms: dict[str, str] = {}
    dead_address = ""

    def author_devices(repo: Path) -> None:
        nonlocal correctors, bpms, dead_address
        records = _orm_stack.roster_records(repo)
        # One of each is all a 1-axis grid_scan needs, and every device in this
        # file is a Channel Access connection the RE worker opens at startup --
        # a slice keeps the deploy fast without weakening either proof.
        correctors = _orm_stack.select_correctors(records, count=1)
        bpms = _orm_stack.select_bpms(records, count=1)
        _orm_stack.write_devices_file(repo, correctors=correctors, bpms=bpms)
        dead_address = _unserved_address(
            next(iter(bpms)), frozenset(record.address for record in records)
        )
        _add_unserved_readable(repo, dead_address)

    # The deployment REPO: `osprey up` runs here, `.env` lives here, and the
    # render `osprey build` produced is `<repo>/build`.
    repo = _orm_stack.build_project_subprocess(
        PROJECT_NAME,
        output_dir=base,
        bridge_port=BRIDGE_PORT,
        port_base=PORT_BASE,
        timeout=BUILD_TIMEOUT_SEC,
        pre_build=author_devices,
    )
    _orm_stack.assert_devices_authored(correctors, bpms)
    assert dead_address, "the pre_build hook never derived an unserved address"

    # The repo root's `.env` -- the deployment's whole secret store, and the
    # file `osprey up` refuses to start without.
    _orm_stack.seed_repo_env(repo)

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
        # `_queue_drive.wait_for_worker_environment`. That it opens AT ALL with
        # an unserved device in the file is itself part of the story: device
        # construction does no I/O, which is why a dead address is invisible
        # until something probes for it.
        try:
            _queue_drive.wait_for_worker_environment(BRIDGE_URL)
        except AssertionError as exc:
            pytest.fail(f"{exc}\n{queue_stack_logs(_orm_stack.project_prefix(PROJECT_NAME))}")
        setpoint, readback = next(iter(correctors.values()))
        yield DeployedPreflightStack(
            repo=repo,
            corrector_setpoint=setpoint,
            corrector_readback=readback,
            bpm_name=next(iter(bpms)),
            dead_address=dead_address,
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


# ---------------------------------------------------------------------------
# R1: a plan naming an unserved channel is refused before anything moves
#     (STRICT -- no flaky mark; see the module docstring's Markers note)
# ---------------------------------------------------------------------------


def test_a_plan_naming_an_unserved_channel_is_refused_before_motion(
    preflight_stack: DeployedPreflightStack,
) -> None:
    dead = preflight_stack.dead_address

    # The authored file won: the entry this suite added by hand is in the file
    # the worker mounts, not a roster-derived set that would never contain it.
    staged = (
        preflight_stack.repo / "build" / "services" / "bluesky" / "bluesky_devices.yml"
    ).read_text(encoding="utf-8")
    assert dead in staged, (
        f"the staged device file does not name {dead} -- the authored "
        f"<repo>/data/bluesky_devices.yml did not reach the build zone, so this proof would "
        f"be testing a device the worker never had rather than a channel no IOC serves"
    )

    # And the worker really built it. This is what separates the refusal under
    # test from a device-resolution failure: the name IS in the namespace a
    # plan resolves against, so anything that refuses the run below refused it
    # on reachability, not on a missing device.
    status, body = _get(f"/devices?prefix={dead}")
    assert status == 200, f"GET /devices failed: {status} {body}"
    assert [entry["name"] for entry in body["devices"]] == [dead], (
        f"the worker did not build a device named {dead}: {body}"
    )

    start, stop = preflight_stack.axis()
    before = _host_read(preflight_stack.repo, preflight_stack.corrector_readback)
    assert min(abs(before - start), abs(before - stop)) > READ_TOLERANCE, (
        f"{preflight_stack.corrector_readback} already reads {before}, which is one of the "
        f"points this sweep would command ({start}, {stop}) -- 'it did not move' would then "
        f"be true whether or not the gate fired"
    )

    token = _orm_stack.minted_launch_token(preflight_stack.repo)
    run_id, record = _queue_drive.run_plan(
        BRIDGE_URL,
        "grid_scan",
        {
            "readbacks": [dead],
            "axes": [
                {
                    "setpoint": preflight_stack.corrector_setpoint,
                    "start": start,
                    "stop": stop,
                    "num_points": NUM_POINTS,
                }
            ],
        },
        token=token,
        client_id=_QUEUE_CLIENT_ID,
        timeout=SCAN_TIMEOUT_SEC,
    )

    assert record.get("status") == "error", (
        f"a plan declaring the unserved channel {dead} was not refused (run {run_id}, "
        f"record {record}) -- without the pre-flight this run reaches the RunEngine and "
        f"fails mid-plan, with the corrector already stepped"
    )

    error = str(record.get("error", ""))
    assert "refusing plan 'grid_scan' before it moves anything" in error, (
        f"the refusal does not name the plan it refused: {error!r}"
    )
    identity = re.search(r"lane (\S+) \(target (\S+)\):", error)
    assert identity is not None, (
        f"the refusal names no lane/target -- an operator running several lanes cannot tell "
        f"which machine refused: {error!r}"
    )
    lane, target = identity.group(1), identity.group(2)
    assert (lane, target) == ("bluesky", "va"), (
        f"expected this single-lane virtual-accelerator deployment to identify itself as "
        f"lane 'bluesky' (target 'va'), got lane {lane!r} (target {target!r}): {error!r}"
    )
    assert f"1 declared channel did not respond within 5 s: {dead}" in error, (
        f"the refusal does not name the one channel that failed its probe, or does not quote "
        f"the bound the verdict rested on: {error!r}"
    )
    assert preflight_stack.corrector_setpoint not in error, (
        f"the refusal blames {preflight_stack.corrector_setpoint}, which the VA serves -- only "
        f"the addresses that actually failed a probe belong in it: {error!r}"
    )

    # Refused BEFORE motion, not after: no run was ever opened...
    assert "run_uid" not in record, (
        f"the refused plan opened a RunEngine run ({record}) -- the gate is supposed to fire "
        f"one message before the plan is built, so nothing should have started"
    )
    # ...and the corrector the sweep would have driven never moved. Read by an
    # independent host-side CA client, not by the bridge under test.
    after = _host_read(preflight_stack.repo, preflight_stack.corrector_readback)
    assert abs(after - before) <= READ_TOLERANCE, (
        f"{preflight_stack.corrector_readback} moved from {before} to {after} across a REFUSED "
        f"run -- the plan was stopped after it had already applied a setpoint, which is the "
        f"partially-executed run this gate exists to prevent"
    )


# ---------------------------------------------------------------------------
# R2: a fully reachable plan is unaffected
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_a_fully_reachable_plan_still_runs_to_completion(
    preflight_stack: DeployedPreflightStack,
) -> None:
    start, stop = preflight_stack.axis()
    token = _orm_stack.minted_launch_token(preflight_stack.repo)
    run_id, record = _queue_drive.run_plan(
        BRIDGE_URL,
        "grid_scan",
        {
            "readbacks": [preflight_stack.bpm_name],
            "axes": [
                {
                    "setpoint": preflight_stack.corrector_setpoint,
                    "start": start,
                    "stop": stop,
                    "num_points": NUM_POINTS,
                }
            ],
        },
        token=token,
        client_id=_QUEUE_CLIENT_ID,
        timeout=SCAN_TIMEOUT_SEC,
    )

    assert record.get("status") == "completed", (
        f"a plan whose declared channels are all served did not complete (run {run_id}, "
        f"record {record}) -- a pre-flight that refuses live plans is worse than the mid-run "
        f"failure it replaced"
    )

    status, data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    assert data["row_count"] == NUM_POINTS, (
        f"expected {NUM_POINTS} rows (one per grid point), got {data['row_count']}: {data}"
    )
