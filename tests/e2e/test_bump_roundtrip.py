"""Real-container ``orbit_bump_sweep`` round-trip e2e (PROPOSAL.md SC5).

``orbit_bump_sweep`` is the third shipped Bluesky plan, and the only one that
*solves* for what it writes: it measures each corrector's own response, then
computes the corrector currents that displace the beam at the target BPMs while
leaving the closure BPMs on the reference orbit. Every other proof of it runs
in-process — the unit tests against fake devices, and ``tests/va/
test_bump_crosscheck.py``'s cross-check of ``bump_analysis`` against the
``lattice/response.py`` model oracle. This is the one that runs it on a
deployed stack: the same plan file, loaded by the bridge's shipped-tier
catalog, driving real correctors over Channel Access through the queueserver,
with the orbit read back off a live soft IOC.

Deploys the turn-key scan-stack config (``tests/e2e/_orm_stack.py`` — the
single source of this deploy shape) and drives the plan over the bridge's queue
API (``PATCH /draft`` -> ``POST /queue/items`` -> armed ``POST /queue/start`` ->
poll -> ``GET /runs/{id}/data``, spelled once in ``tests/e2e/_queue_drive.py``),
proving five things end to end:

  (a) the run reaches a terminal ``completed`` status within a bounded timeout
      — i.e. no corrector step ever hangs the bridge's ``ConnectorSettable.
      set()`` settle-wait, the same regression class ``test_orm_roundtrip.py``
      guards.
  (b) the run's row set is EXACTLY ``baseline_reads + 2 * num`` rows. This is
      the plan's pinned row layout, and it is a real claim about this plan
      specifically: an ``orbit_bump_sweep`` run takes far more readings than it
      records — two probe reads per corrector, two convergence reads per step,
      two more per trim pass — and every one of them is taken off the record
      (``_snapshot``, no ``create``/``save`` pair). A probe or convergence read
      that leaked into an event document would show up here as extra rows, and
      would silently corrupt the positional step attribution the figure and any
      downstream analysis depend on.
  (c) the bump is a real, closed, local bump on the modelled machine: each
      recorded step sits inside ``±tolerance`` of what that step asked for at
      the target AND at every closure BPM; the full-amplitude step reaches the
      requested displacement; the ring outside the corrector span does not
      move, including at a monitor BPM the solve was never told to hold; and
      the terminal step brings the orbit and every corrector back to where the
      run found them.
  (d) ``GET /runs/{id}/figure`` serves the ``orbit_bump_sweep`` plan's OWN view
      of that run — the orbit-shift, band-residual and corrector-offset panels
      — rather than the generic fallback view.
  (e) that same figure survives the run leaving the bridge's memory: after the
      bridge container is restarted (``_orm_stack.restart_bridge``) the live
      row buffer is gone, the route can only answer from the Tiled catalog, and
      the figure it draws from there matches the one it drew live (mirrors
      ``test_orm_roundtrip.py``'s restart assertion).

No in-flight figure is asserted, unlike ``test_orm_roundtrip.py``'s. That test
runs a deliberately oversized 126-row scan; this one records seven rows, and
the plan's render draws nothing at all until ``baseline_reads`` baseline rows
plus one amplitude step are in hand (see ``_prepare``). The window in which a
partial figure exists is therefore a fraction of a short run — an assertion
that would be racing the run rather than testing the route.

No physics fault is seeded on this stack (no ``VA_BPM_ERRORS``/``VA_CORR_GAIN``
in the written ``.env`` — see ``_orm_stack.write_substrate_env``), so the VA's read
path is deterministic: the baseline reads return identical values and the
measured per-BPM sigma is exactly zero. That is legal and is not a skip — see
``bump_analysis.noise_floor_violations`` on why a zero sigma is a fact about
readback resolution rather than a fault — and it means the convergence band
here is exactly ``±TOLERANCE_M``, with no noise term taking over.

No preset channel names are hardcoded: correctors and BPMs are derived from the
render's own ``build/data/channel_limits.json`` — the limits database the
deployed containers actually read — via ``_orm_stack.select_correctors``/
``select_bpms``, and then chosen from those lists BY POSITION. See
``CORRECTOR_INDICES`` for why the positions are what they are.

Ports: every published port of this deployment is pinned to a value this module
owns — see the constants block below for the value per service and the sibling
pin each steps around. The WHOLE block is pinned, not only the three ports the
assertions read, because ``osprey up``'s preflight refuses to start when ANY
published port is taken: the archiver's Mongo, which nothing here reads, still
blocks the entire deploy if a tutorial stack on the same host holds its
default. Pinning all of them is what lets this module run beside an
already-deployed tutorial stack, and beside any sibling suite, without touching
or being blocked by anything it does not own.

Container safety: every docker invocation below names an exact
container/image — never a wildcard, never ``system prune``/``--volumes``. The
one restart names a single compose service inside this deployment's own
project. Teardown goes through ``osprey down``, matching every other e2e in
this directory.

Gating: needs Docker; the VA image builds natively for the host arch, so on
Apple Silicon PyAT/softioc compile from source (no prebuilt aarch64 wheels) --
slow (minutes) on a cold image cache. Runs on CI in the ``orm-roundtrip-e2e``
job, as the third of three sequential steps. Run locally with
``E2E_REUSE_IMAGES=1`` set for fast iteration once the image cache is warm.
"""

from __future__ import annotations

import math
import os
import shutil
import statistics
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from osprey.deployment.compose_generator import resolve_project_name
from osprey.services.bluesky_bridge.figure import rows_from_columnar
from tests.e2e import _orm_stack, _queue_drive
from tests.e2e._deploy_diagnostics import dead_container_logs, queue_stack_logs

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    # dockerbuild: full VA/bridge/Tiled image build + deploy -- runs in the
    # dedicated orm-roundtrip-e2e CI job alongside test_orm_roundtrip.py and
    # test_grid_scan_roundtrip.py, never the shared e2e-tests lane (the
    # marker->--ignore pairing is enforced by
    # tests/deployment/test_ci_workflow_wiring.py).
    pytest.mark.dockerbuild,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

# ---------------------------------------------------------------------------
# This module's own port block. Every value here is distinct from BOTH the
# preset defaults (8090 bridge / 8091 tiled / 8095 panels / 5064 VA / 5432
# postgres / 5080 openobserve / 27017 mongodb, all of which a tutorial stack
# deployed on the same host holds) and every sibling e2e module's pin. Those
# sibling pins, per service:
#
#   bridge      18090 test_bluesky_deploy · 18099 test_va_substrate_equivalence
#               18101 test_tiled_roundtrip · 18102 _orm_stack's default
#               18103 test_bluesky_catalog_e2e · 18104 test_grid_scan_roundtrip
#               18105 test_bluesky_sandbox_escape_e2e
#               18106 test_bluesky_web_deploy · 18108 test_bluesky_queue_e2e
#               18109 test_plan_stack_agentic
#               (18107 is test_nextcloud_talk_bridge_e2e's Nextcloud)
#               (no bluesky-web port: this deploy drops that sidecar entirely,
#               see EXTRA_CONFIG)
#   tiled       18191 test_bluesky_queue_e2e · 18192 test_tiled_roundtrip
#               18193 test_plan_stack_agentic
#   VA (CA)     15064 test_bluesky_queue_e2e · 15065 test_tiled_roundtrip
#               15066 test_plan_stack_agentic
#   postgres    25432 test_bluesky_queue_e2e · 25433 test_tiled_roundtrip
#               25434 test_bluesky_web_deploy · 25435 test_plan_stack_agentic
#   openobserve 25080 test_bluesky_queue_e2e · 25081 test_bluesky_deploy
#               25082 test_tiled_roundtrip · 25083 test_bluesky_web_deploy
#               25084 test_plan_stack_agentic
#   mongodb     27117 test_plan_stack_agentic
# ---------------------------------------------------------------------------
BRIDGE_PORT = 18110
TILED_PORT = 18194
VA_CA_PORT = 15067
POSTGRES_PORT = 25436
OPENOBSERVE_PORT = 25085
MONGODB_PORT = 27118
QMD_PORT = 28180  # preset default 8180 collides with any tutorial stack on the host

BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

#: Override-file additions on top of ``_orm_stack.override_yaml()``.
#:
#: The port moves cover every host port with no ``--set`` hook on
#: ``_orm_stack.init_args`` (which reaches only the bridge and VA ports).
#: ``bluesky``/``bluesky_web``/``va_archiver`` are top-level PROFILE keys,
#: not ``config:`` keys -- ``services.mongodb.port_host`` under ``config:`` is
#: silently discarded, because the archiver's Mongo port is owned by the
#: ``va_archiver`` block that renders it.
#:
#: ``bluesky_web: null`` drops the bluesky-web sidecar, for the same reason
#: ``override_yaml()`` already drops ``dispatch`` and the web terminals:
#: nothing here reads it, and it is a THIRD locally-built image compiled in
#: parallel with the VA and the bridge. Three concurrent source builds, each
#: resolving a large wheel set, is what an 8 GB Docker VM does not have the
#: memory for -- and a buildkit worker that gets killed reports only ``failed
#: to receive status: ... EOF``, which reads like a network fault rather than
#: the resource exhaustion it is. The panel surface has its own lane
#: (``test_bluesky_web_deploy.py``).
EXTRA_CONFIG: dict[str, Any] = {
    "config": {
        "services.postgresql.port_host": POSTGRES_PORT,
        "services.openobserve.port": OPENOBSERVE_PORT,
        "services.qmd.port": QMD_PORT,
    },
    "bluesky": {"tiled_port": TILED_PORT},
    "bluesky_web": None,
    "va_archiver": {"port_host": MONGODB_PORT},
}

#: Compose project this suite deploys under. Container names follow
#: ``<project>-<service>``, so the failure diagnostics and the bridge restart
#: below need the same name the build is given -- hence one constant rather
#: than repeated literals.
PROJECT_NAME = "bump-roundtrip"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 1200  # first-time native VA source build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0

# ---------------------------------------------------------------------------
# The bump's geometry
# ---------------------------------------------------------------------------
#
# Positions within the channel-limits-derived device lists, not device names --
# the deployed project's own channel_limits.json owns which devices exist.
# These are the SAME positions `tests/va/test_bump_crosscheck.py` picks (its
# lists come from `lattice.inventory.pyat_coupled_device_ids()`, which sorts by
# device id, and `select_correctors`/`select_bpms` sort by full address over
# the same pyat-coupled partition -- so position i is the same magnet either
# way), and they are picked for CONDITIONING rather than for aesthetics.
#
# The three correctors are clustered inside one sixth of the ring with enough
# betatron phase between them to close a bump: the probed response comes back
# well conditioned (condition number ~10) and the solve asks for single-digit
# amps, comfortably inside the +-12 A the correctors' channel_limits band
# allows. A tighter cluster is closer to a single combined kick and needs far
# more current for the same displacement. BPMs and correctors are colocated in
# this lattice, so the target BPM sits at the middle corrector -- the peak of
# the bump -- and the closure BPMs are spread around the rest of the ring, each
# well clear of the corrector span.
CORRECTOR_INDICES = (11, 12, 17)
TARGET_BPM_INDEX = 12
CLOSURE_BPM_INDICES = (0, 22, 31, 40, 49, 58)

#: A monitor BPM INSIDE the corrector span. Recorded at every point and never
#: part of the solve, it is what makes the "this is a bump" claim non-vacuous:
#: the beam has to actually be somewhere else across the span, not merely back
#: where it started at the handful of BPMs the solve constrained.
SPAN_MONITOR_BPM_INDEX = 14

#: A monitor BPM OUTSIDE the corrector span that the solve was never told to
#: hold. Closure at the six constrained BPMs is something the least-squares
#: solve was asked for; closure here is not, so it is the honest test of
#: whether the bump is local -- the same thing
#: ``test_bump_crosscheck.test_bump_closes_outside_the_corrector_span`` asserts
#: over every unconstrained BPM, applied here to one, because every device
#: named in this test is one more Channel Access connection the deployed worker
#: environment has to open before the queue can accept a plan.
CLOSURE_MONITOR_BPM_INDEX = 65

# ---------------------------------------------------------------------------
# The bump's scale
# ---------------------------------------------------------------------------

#: Probe dither half-amplitude, in amps. Large enough that the orbit it moves
#: is far above any readback resolution (tens of microns), small enough to stay
#: inside the quasi-linear window the corrector calibration was chosen to give
#: -- and inside the +-12 A channel_limits band, which the connector's
#: reference monitor enforces on every write.
PROBE_AMPLITUDE_A = 5.0

#: The displacement asked for at the target BPM, in meters -- the units the
#: deployed VA's BPM records carry (`serving/pvdb.py`'s analog channels serve
#: the full double; the six-digit `prec` there is display metadata only).
#: Sized so the solve lands at single-digit amps: measured in-process against
#: the same lattice, this corrector set needs at most 3.7 A for it.
TARGET_BUMP_M = 20e-6

#: The convergence band, in meters. 0.5% of the requested bump, and roughly 30x
#: the worst step residual this corrector set actually produces on this lattice
#: (~3e-9 m, measured in-process through the same `fit_probe_response` ->
#: `solve_offsets` path the plan runs). Deliberately not floored at the VA's
#: noise: this stack seeds no BPM errors, so the measured sigma is exactly zero
#: and `noise_floor_violations` correctly declines to draw a floor from it,
#: leaving the band exactly this value. A tolerance chosen at the noise floor
#: would therefore have been zero, which no band can satisfy.
TOLERANCE_M = 1e-7

#: Amplitude increments from zero to the requested bump. The plan visits
#: ``2 * num`` steps monodirectionally (up then back down), so this is the
#: smallest value that still gives an intermediate amplitude to check the
#: profile against -- the sweep visits 50%, 100%, 50%, 0%.
NUM_STEPS = 2

#: Reads taken before anything moves. The reference orbit and the noise
#: estimate the demand gate needs; ``ge=3`` in the plan's schema.
BASELINE_READS = 3

#: Correction passes allowed per amplitude step. The solve lands inside the
#: band on the first try here, so this is headroom rather than a working part
#: of the run -- but a run that needed it and was refused it would be a
#: misleading failure.
MAX_TRIM_ITERATIONS = 2

#: Wait after each move before reading. Small on purpose: the connector's
#: ``set()`` already blocks until the corrector's readback echoes its setpoint,
#: so this is settle margin on top of a completed write, not the write's own
#: timeout.
SETTLE_S = 0.2

#: The amplitude profile ``num=2`` monodirectional produces, spelled out rather
#: than imported from the plan's own ``_profile_scales``: the row-to-step
#: attribution every assertion below rests on is a CONTRACT of the plan, and a
#: test that re-derived it from the implementation could not notice the
#: implementation changing it.
EXPECTED_SCALES = (0.5, 1.0, 0.5, 0.0)

#: Rows a healthy run records: the baseline, then exactly one per amplitude
#: step. Every probe and convergence read is taken off the record.
EXPECTED_ROWS = BASELINE_READS + len(EXPECTED_SCALES)

#: Agreement bound for the magnitude cross-checks, as a fraction of the peak
#: bump amplitude -- the same relative bound
#: ``tests/va/test_bump_crosscheck.py`` holds its oracle comparison to, for the
#: same reason: the tolerable share of the bump that sextupole feed-down may
#: account for at these currents.
RELATIVE_BOUND = 1e-2

#: Noise multiplier the magnitude bounds floor at, so a BPM whose reading
#: scatters is never asked to agree to better than it can be read. Zero on this
#: clean stack (see the module docstring), which is why the relative term above
#: is the operative one -- but it is computed from the run's OWN baseline rows
#: rather than assumed, so a stack that did add read noise would widen these
#: bounds instead of failing under them.
NOISE_SIGMAS = 3.0

# 7 recorded rows, but far more device traffic than that: 3 correctors x (2
# probe moves + 2 off-record reads + 1 restore), then 4 steps x (1 three-
# corrector move + 2 off-record reads + 1 recorded row) -- each recorded row a
# 12-device bundle, each off-record read 7-9 devices, over Channel Access. A
# healthy point costs a second or two, so a
# healthy run here is a couple of minutes; this is several times that, which
# keeps it a meaningful "did it hang" gate (a hung settle-wait never returns at
# all) rather than a flaky timing assertion.
SCAN_TIMEOUT_SEC = 600.0

#: Grace window for the recorded rows to reach the pinned count AFTER the run
#: record reads "completed". The record is the queue item's status from the
#: manager's history; the rows are event documents on the bridge's document
#: plane. Same two-transport gap as the settled-figure wait below, and the two
#: facts can land in either order.
ROW_SETTLE_TIMEOUT_SEC = 30.0

#: How long the live figure may take to report itself settled AFTER the run
#: record reads "completed". These are two different facts: the record is the
#: queue item's status, while ``partial`` comes from the run's stop document
#: reaching the bridge's document plane, which the route reads instead of the
#: record on purpose.
SETTLED_FIGURE_TIMEOUT_SEC = 60.0

#: How long the restarted bridge may take to answer from Tiled. The health wait
#: inside ``restart_bridge`` proves only that the bridge process is back up;
#: whether the queueserver's ``TiledWriter`` has flushed this run to the
#: catalog is a separate, asynchronous fact, and this poll is what establishes
#: it.
TILED_FIGURE_TIMEOUT_SEC = 60.0

FIGURE_POLL_SEC = 1.0

#: Tolerance for live-vs-Tiled figure parity. Both figures are drawn by the
#: same ``render`` over the same measurements, so the only thing between them
#: is the storage round trip -- float64 through a Tiled table on one side,
#: float64 through JSON on the other, neither of which loses a bit. This is
#: insurance against a dtype narrowing somewhere in that path, not a real error
#: budget.
PARITY_RTOL = 1e-9
PARITY_ATOL = 1e-12

#: Panels the plan's render draws for THIS run, in the order it draws them. No
#: "Monitor response" panel: that one is built only when ``monitors`` were
#: requested, and this run requests none -- every device it reads is a BPM or a
#: corrector, both of which already have a panel that says something about
#: them.
EXPECTED_PANELS = (
    "Orbit shift across BPMs",
    "Band residual at constrained BPMs",
    "Corrector offsets",
)


def _dead_container_logs() -> str:
    """Logs from every container of this deployment that is not running."""
    return dead_container_logs(resolve_project_name({"project_name": PROJECT_NAME}))


def _get(path: str) -> tuple[int, Any]:
    return _queue_drive.request(BRIDGE_URL, path, "GET")


def _find_column(columns: list[str], device_name: str) -> str:
    """The event-data column for a device -- ophyd-async names a hinted child
    ``"<device>-<child>"``, so match the device-name prefix rather than the
    exact key (mirrors test_grid_scan_roundtrip.py's identical helper)."""
    for column in columns:
        if column == device_name or column.startswith(f"{device_name}-"):
            return column
    raise AssertionError(f"no column for device {device_name!r} in {columns!r}")


# ---------------------------------------------------------------------------
# The deployed stack
# ---------------------------------------------------------------------------


class DeployedBumpStack:
    """Everything the round-trip test needs about the one deployment repo.

    The four BPM roles are kept apart rather than merged into one list because
    the assertions are about the roles: a target is asked to move, a closure
    BPM to stay, the span monitor is expected to move without having been
    asked, and the closure monitor to stay without having been asked.
    """

    def __init__(
        self,
        repo: Path,
        correctors: list[str],
        target_bpm: str,
        closure_readbacks: list[str],
        span_monitor: str,
        closure_monitor: str,
    ):
        self.repo = repo
        self.correctors = correctors
        self.target_bpm = target_bpm
        self.closure_readbacks = closure_readbacks
        self.span_monitor = span_monitor
        self.closure_monitor = closure_monitor

    @property
    def monitors(self) -> list[str]:
        return [self.span_monitor, self.closure_monitor]

    @property
    def constrained_bpms(self) -> list[str]:
        """Targets then closure -- the row order the plan's solve, and every
        band assertion below, share."""
        return [self.target_bpm, *self.closure_readbacks]

    @property
    def all_bpms(self) -> list[str]:
        return [*self.constrained_bpms, *self.monitors]


def _horizontal_devices(limits: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The HCM corrector setpoints and BPM X readbacks of the deployed project.

    One plane only. This bump is horizontal: it is built from HCM correctors
    and verified on the BPMs' X axis, which is the one plane a single response
    fit can span. Mixing in a VCM would add a column that moves no X reading at
    all, leaving the fitted response rank-deficient -- which
    ``fit_probe_response`` correctly refuses, before any bump is solved.

    Both lists come from ``select_correctors``/``select_bpms``, so they are the
    deployed project's own channel_limits entries, restricted to the
    pyat-coupled partition (a write actually steers the beam through the AT
    lattice model), and sorted by address -- which for one family and one field
    is device-id order.
    """
    correctors = [
        address
        for address in _orm_stack.select_correctors(limits, count=None)
        if address.split(":")[2] == "HCM"
    ]
    bpms = [
        address
        for address in _orm_stack.select_bpms(limits, count=None)
        if address.endswith(":POSITION:X")
    ]
    return correctors, bpms


@pytest.fixture(scope="module")
def deployed_bump_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedBumpStack]:
    base = tmp_path_factory.mktemp("bump_roundtrip_build")
    # The deployment REPO: `osprey up` runs here, `.env` lives here, and the
    # render `osprey build` produced is `<repo>/build`.
    repo = _orm_stack.build_project_subprocess(
        PROJECT_NAME,
        output_dir=base,
        bridge_port=BRIDGE_PORT,
        va_port=VA_CA_PORT,
        timeout=BUILD_TIMEOUT_SEC,
        extra_config=EXTRA_CONFIG,
    )

    # The render's copy, not the operator-owned source under <repo>/data/:
    # control_system.limits_checking.database_path resolves against the
    # rendered config's directory, so build/data is the file the containers
    # actually read.
    limits = _orm_stack.channel_limits(repo / "build")
    available_correctors, available_bpms = _horizontal_devices(limits)

    bpm_indices = (
        TARGET_BPM_INDEX,
        *CLOSURE_BPM_INDICES,
        SPAN_MONITOR_BPM_INDEX,
        CLOSURE_MONITOR_BPM_INDEX,
    )
    assert len(available_correctors) > max(CORRECTOR_INDICES), (
        f"the deployed project yields {len(available_correctors)} horizontal correctors, "
        f"too few for this bump's geometry (needs position {max(CORRECTOR_INDICES)})"
    )
    assert len(available_bpms) > max(bpm_indices), (
        f"the deployed project yields {len(available_bpms)} horizontal BPM readbacks, "
        f"too few for this bump's geometry (needs position {max(bpm_indices)})"
    )

    stack = DeployedBumpStack(
        repo=repo,
        correctors=[available_correctors[index] for index in CORRECTOR_INDICES],
        target_bpm=available_bpms[TARGET_BPM_INDEX],
        closure_readbacks=[available_bpms[index] for index in CLOSURE_BPM_INDICES],
        span_monitor=available_bpms[SPAN_MONITOR_BPM_INDEX],
        closure_monitor=available_bpms[CLOSURE_MONITOR_BPM_INDEX],
    )

    # Only the devices this bump names reach the worker namespace: every one of
    # them is a Channel Access connection the RE worker environment has to open
    # before the queue will accept a plan, and the full 144-device inventory
    # would spend that on devices no assertion here reads.
    all_correctors = _orm_stack.select_correctors(limits, count=None)
    # Writes the repo root's `.env` — the deployment's whole secret store, and
    # the file `osprey up` refuses to start without.
    _orm_stack.write_substrate_env(
        repo,
        correctors={name: all_correctors[name] for name in stack.correctors},
        bpms={name: name for name in stack.all_bpms},
    )

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
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}\n"
                f"--- containers that are not running ---\n{_dead_container_logs()}"
            )
        try:
            _orm_stack.wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
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
        yield stack
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


# ---------------------------------------------------------------------------
# Reading figures off the deployed bridge
#
# These four helpers mirror `test_orm_roundtrip.py`'s rather than importing
# them: this suite's convention is that an e2e module owns its own scaffolding
# (see `_find_column` above, and `test_va_substrate_equivalence.py`'s copy of
# the same), and importing another TEST module would make this file's
# collection depend on that one's module-level fixtures and markers.
# ---------------------------------------------------------------------------


def _figure(run_id: str) -> tuple[int, Any]:
    """One ``GET /runs/{id}/figure``, returning ``(status, body)``.

    A shorter per-request timeout than `_queue_drive.request`'s 20 s default:
    this is called from 1 Hz polls with their own deadlines, and one hung
    request must not eat a whole poll budget by itself.
    """
    return _queue_drive.request(BRIDGE_URL, f"/runs/{run_id}/figure", "GET", timeout=10.0)


def _figure_summary(figure: dict[str, Any]) -> str:
    """A one-line digest of a figure response, for a failure message."""
    titles = [panel.get("title") for panel in figure.get("panels") or []]
    return (
        f"source={figure.get('source')!r} partial={figure.get('partial')!r} "
        f"reason={figure.get('reason')!r} panels={titles}"
    )


def _poll_figure(
    run_id: str,
    *,
    until: Callable[[dict[str, Any]], bool],
    what: str,
    timeout: float,
    give_up: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """Poll ``GET /runs/{id}/figure`` until *until* accepts a response body.

    Non-200 answers are retried rather than raised: a run whose first document
    has not landed yet is in neither the live buffer nor the Tiled catalog, and
    the route's honest answer for that is a 404.

    *give_up* names a condition under which waiting can no longer succeed --
    here, the live buffer outliving a restart. Returning a message from it
    fails right there rather than spending the rest of the deadline on a wait
    whose outcome is already decided.

    Every failure path raises ``AssertionError``, never ``pytest.fail``: this
    test's ``flaky`` marker reruns on ``AssertionError`` only, so a ``Failed``
    raised in here would quietly opt the whole test out of its one retry.
    """
    deadline = time.monotonic() + timeout
    last = "(no answer yet)"
    while time.monotonic() < deadline:
        status, body = _figure(run_id)
        if status == 200 and isinstance(body, dict):
            if give_up is not None:
                abandon = give_up(body)
                if abandon is not None:
                    raise AssertionError(abandon)
            if until(body):
                return body
            last = _figure_summary(body)
        else:
            last = f"HTTP {status}: {body}"
        time.sleep(FIGURE_POLL_SEC)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {what} on "
        f"GET /runs/{run_id}/figure (last seen: {last})"
    )


def _is_number(value: Any) -> bool:
    """True for a JSON number. Deliberately false for ``bool``, which is an
    ``int`` in Python but a flag on the wire -- ``decimated: true`` must be
    compared exactly, never within a tolerance."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _assert_json_close(live: Any, tiled: Any, *, path: str) -> None:
    """Assert two figure bodies are structurally identical, numbers to a tolerance.

    Walked generically rather than field by field so that the parity claim
    covers the WHOLE wire contract -- including any field a later change adds
    to a mark -- and so a mismatch reports the exact path (e.g.
    ``figure.panels[1].mark.series[0].points[3].y``) instead of a diff of two
    large dumps.
    """
    if isinstance(live, dict) and isinstance(tiled, dict):
        assert live.keys() == tiled.keys(), (
            f"{path}: the live and Tiled figures carry different fields "
            f"(live only: {sorted(live.keys() - tiled.keys())}, "
            f"Tiled only: {sorted(tiled.keys() - live.keys())})"
        )
        for key in live:
            _assert_json_close(live[key], tiled[key], path=f"{path}.{key}")
        return
    if isinstance(live, list) and isinstance(tiled, list):
        assert len(live) == len(tiled), (
            f"{path}: the live figure has {len(live)} entries here, the Tiled figure {len(tiled)}"
        )
        for index, (one, other) in enumerate(zip(live, tiled, strict=True)):
            _assert_json_close(one, other, path=f"{path}[{index}]")
        return
    if _is_number(live) or _is_number(tiled):
        # A drawn gap (null) on one side and a number on the other is a real
        # parity failure, and lands here rather than in the exact branch below.
        assert _is_number(live) and _is_number(tiled), (
            f"{path}: live={live!r} and Tiled={tiled!r} are not both numbers"
        )
        # Two NaNs are not a parity failure but a wire-contract one, and
        # `isclose` would report them as an unequal pair. The figure models
        # null every non-finite value at the drawing edge, so a NaN reaching a
        # client means that nulling was skipped -- on BOTH paths at once, which
        # a parity message would send someone hunting the wrong difference.
        both_nan = math.isnan(float(live)) and math.isnan(float(tiled))
        assert not both_nan, (
            f"both figures carry NaN at {path} -- no NaN should reach the wire; "
            "the figure models null non-finite values instead"
        )
        assert math.isclose(float(live), float(tiled), rel_tol=PARITY_RTOL, abs_tol=PARITY_ATOL), (
            f"{path}: live={live!r} vs Tiled={tiled!r} "
            f"(rel_tol={PARITY_RTOL}, abs_tol={PARITY_ATOL})"
        )
        return
    assert live == tiled, f"{path}: live={live!r} vs Tiled={tiled!r}"


def _assert_figures_agree(live: dict[str, Any], from_tiled: dict[str, Any]) -> None:
    """The two figures are the same figure, drawn from two stores.

    ``source`` is the one field that must differ -- it names which store
    answered -- so it is checked for its expected value and then set aside;
    everything else, panels and annotations and numbers alike, has to agree.
    """
    live_body = dict(live)
    tiled_body = dict(from_tiled)
    assert live_body.pop("source") == "live"
    assert tiled_body.pop("source") == "tiled"
    _assert_json_close(live_body, tiled_body, path="figure")


# ---------------------------------------------------------------------------
# Reading the run's own rows
# ---------------------------------------------------------------------------


def _column_values(rows: list[dict[str, Any]], columns: list[str], device: str) -> list[float]:
    """One device's reading from every row, as floats.

    A null reading is a failure rather than a gap: the plan aborts on a device
    that does not report, so a null here would mean a row was recorded for a
    point the plan believed it had read.
    """
    column = _find_column(columns, device)
    values = [row[column] for row in rows]
    assert all(value is not None for value in values), (
        f"device {device!r} (column {column!r}) has a null reading somewhere in this run: {values}"
    )
    return [float(value) for value in values]


class _Measured:
    """The run's recorded rows, resolved into what each assertion asks about.

    Rows carry no step number -- exactly as the plan's own render notes -- so
    steps are attributed by position: ``BASELINE_READS`` baseline rows, then
    one row per scale of the amplitude profile, in profile order. That
    attribution is only sound because the row COUNT was asserted first.
    """

    def __init__(self, stack: DeployedBumpStack, columns: list[str], rows: list[dict[str, Any]]):
        baseline_rows = rows[:BASELINE_READS]
        step_rows = rows[BASELINE_READS:]

        devices = [*stack.all_bpms, *stack.correctors]
        baseline = {name: _column_values(baseline_rows, columns, name) for name in devices}
        steps = {name: _column_values(step_rows, columns, name) for name in devices}

        #: Per-device reference: the mean of the baseline rows, which is what
        #: the plan itself measures every later offset against.
        self.reference = {name: statistics.fmean(values) for name, values in baseline.items()}
        #: Per-device sample standard deviation over the baseline rows -- the
        #: run's own measurement of its noise, on the same ddof=1 terms the
        #: plan uses.
        self.sigma = {name: statistics.stdev(values) for name, values in baseline.items()}
        #: Per-device orbit/current offset at each amplitude step.
        self.offsets = {
            name: [value - self.reference[name] for value in values]
            for name, values in steps.items()
        }

    def peak(self, devices: list[str]) -> float:
        """The largest excursion this run made anywhere among *devices*."""
        return max(abs(offset) for name in devices for offset in self.offsets[name])

    def bound(self, device: str, peak: float) -> float:
        """The magnitude-agreement bound at *device*:
        ``max(RELATIVE_BOUND * peak, NOISE_SIGMAS * sigma)``.

        The same form ``tests/va/test_bump_crosscheck.py`` uses: a share of the
        bump the lattice's nonlinearity may account for, floored at what the
        BPM can actually be read to.
        """
        return max(RELATIVE_BOUND * peak, NOISE_SIGMAS * self.sigma[device])


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_orbit_bump_sweep_roundtrip_closes_a_local_bump(
    deployed_bump_stack: DeployedBumpStack,
) -> None:
    stack = deployed_bump_stack

    status, plans_body = _get("/plans")
    assert status == 200, f"GET /plans failed: {status} {plans_body}"
    plan_names = {plan["name"] for plan in plans_body}
    assert "orbit_bump_sweep" in plan_names, (
        f"orbit_bump_sweep not discoverable via GET /plans: {plan_names}"
    )

    plan_args = {
        "correctors": stack.correctors,
        "targets": [{"readback": stack.target_bpm, "value": TARGET_BUMP_M}],
        "closure_readbacks": stack.closure_readbacks,
        "readbacks": stack.monitors,
        "mode": "relative",
        "sweep": "monodirectional",
        "num": NUM_STEPS,
        "baseline_reads": BASELINE_READS,
        "probe_amplitude": PROBE_AMPLITUDE_A,
        "tolerance": TOLERANCE_M,
        "max_trim_iterations": MAX_TRIM_ITERATIONS,
        # A step that cannot be trimmed into the band records what it reached
        # and the sweep continues, instead of the plan raising. The band
        # assertions below are then the gate, and they fail with the numbers
        # the run actually produced -- far more useful than a RuntimeError from
        # inside the plan naming one step.
        "best_effort": True,
        "settle_s": SETTLE_S,
        # No beam-current guard: this VA has no beam-current channel in its
        # pyat-coupled partition, and the plan's schema refuses half a guard
        # (a device with no threshold, or a threshold with no device).
        "monitors": [],
    }

    # The module fixture waits for the RE worker environment once, which covers
    # the first pass. This test restarts the bridge at the end, though, and the
    # bridge opens that environment off its readiness path -- so on a `flaky`
    # rerun the enqueue below IS an enqueue after a restart, which would
    # otherwise be refused 409 "not in the list of allowed plans" against a
    # still-empty worker namespace. Re-waiting costs one request when the
    # environment is already open.
    _queue_drive.wait_for_worker_environment(BRIDGE_URL)

    token = _orm_stack.minted_launch_token(stack.repo)
    run_id = _queue_drive.stage_and_enqueue(
        BRIDGE_URL, "orbit_bump_sweep", plan_args, client_id="bump-roundtrip-e2e"
    )
    _queue_drive.start_queue(BRIDGE_URL, token)
    status_body = _queue_drive.wait_for_terminal_status(
        BRIDGE_URL, run_id, timeout=SCAN_TIMEOUT_SEC
    )

    # --- (a) the run completed, and nothing hung -----------------------------
    assert status_body.get("status") == "completed", (
        f"orbit_bump_sweep did not complete within {SCAN_TIMEOUT_SEC:.0f}s "
        f"(status={status_body}) -- a corrector step whose :RB never echoes its :SP hangs "
        "exactly here, at the bridge's ConnectorSettable.set() settle-wait"
    )

    # --- (b) the pinned row layout -------------------------------------------
    # A short grace wait rather than an immediate read, for the same reason the
    # settled-figure poll below waits: the "completed" status and the rows ride
    # different transports (see ROW_SETTLE_TIMEOUT_SEC). Waiting until the
    # count reaches the pin keeps a late final event document from reading as a
    # missing row; a count ABOVE the pin exits immediately and fails the
    # equality below with the real number.
    deadline = time.monotonic() + ROW_SETTLE_TIMEOUT_SEC
    while True:
        status, data = _get(f"/runs/{run_id}/data")
        rows_landed = (
            status == 200 and isinstance(data, dict) and data.get("row_count", 0) >= EXPECTED_ROWS
        )
        if rows_landed or time.monotonic() >= deadline:
            break
        time.sleep(FIGURE_POLL_SEC)
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    assert data["truncated"] is False, (
        f"a {EXPECTED_ROWS}-row run should fit this route's default window: {data}"
    )
    assert data["row_count"] == EXPECTED_ROWS, (
        f"expected exactly {EXPECTED_ROWS} rows ({BASELINE_READS} baseline + one per "
        f"amplitude step of {list(EXPECTED_SCALES)}), got {data['row_count']} -- this plan "
        "takes two probe reads per corrector and two convergence reads per step OFF the "
        "record, so a count above this is one of those reads having been emitted as an "
        "event, which also shifts every step's row out of position"
    )

    columns = data["columns"]
    rows = rows_from_columnar(columns, data["rows"], data["row_count"]).rows
    assert len(rows) == EXPECTED_ROWS
    measured = _Measured(stack, columns, rows)

    # --- (c) it is a bump, it closed, and it came back ------------------------
    # Checked before the band assertions because it is what makes them mean
    # anything: an orbit that never moved sits inside every band trivially.
    full_scale = EXPECTED_SCALES.index(1.0)
    reached = measured.offsets[stack.target_bpm][full_scale]
    assert reached == pytest.approx(TARGET_BUMP_M, rel=RELATIVE_BOUND), (
        f"asked for a {TARGET_BUMP_M:.3e} m bump at {stack.target_bpm}, reached "
        f"{reached:.3e} m at the full-amplitude step"
    )

    # Taken over every recorded BPM, monitors included: the geometry pick (see
    # CORRECTOR_INDICES) puts the bump's peak AT the target BPM, so any BPM
    # carrying a larger excursion -- monitor or not -- is a bump that is not
    # where it was asked for. The message names the carrier so a monitor-driven
    # failure is self-explaining.
    peak = measured.peak(stack.all_bpms)
    peak_bpm = max(stack.all_bpms, key=lambda name: measured.peak([name]))
    assert peak == pytest.approx(TARGET_BUMP_M, rel=RELATIVE_BOUND), (
        f"the orbit excursion peaks at {peak:.3e} m (carried by {peak_bpm}), not at the "
        f"requested {TARGET_BUMP_M:.3e} m -- the bump is not where it was asked for"
    )

    # The span monitor is recorded and never solved for, so a displacement here
    # is the bump's body rather than anything the solve arranged. Bounding it
    # BELOW rather than above is the point: this is the assertion that fails if
    # the "bump" is really just a handful of BPMs held at zero.
    span_excursion = abs(measured.offsets[stack.span_monitor][full_scale])
    assert span_excursion > measured.bound(stack.span_monitor, peak), (
        f"the monitor BPM inside the corrector span ({stack.span_monitor}) moved only "
        f"{span_excursion:.3e} m at full amplitude, within the {peak:.3e} m bump's own "
        "closure bound -- the beam is not actually displaced across the span, so this run "
        "constrained a few BPMs rather than making a bump"
    )

    # The closure monitor is outside the span and equally unsolved-for, so its
    # staying put is the honest statement that the bump is LOCAL.
    for step, scale in enumerate(EXPECTED_SCALES):
        leaked = abs(measured.offsets[stack.closure_monitor][step])
        bound = measured.bound(stack.closure_monitor, peak)
        assert leaked <= bound, (
            f"step {step + 1} (scale {scale:+g}) leaked {leaked:.3e} m at "
            f"{stack.closure_monitor}, a BPM outside the corrector span that the solve was "
            f"never asked to hold, against a closure bound of {bound:.3e} m -- the bump is "
            "not closed"
        )

    # The terminal step commands the recorded working points back verbatim, so
    # every corrector must read where the run found it. A corrector left off
    # its working point is a machine the run did not put back. Bounded against
    # how far that corrector itself travelled during the sweep -- the same
    # relative form the orbit bounds use, in corrector units.
    terminal = len(EXPECTED_SCALES) - 1
    for corrector in stack.correctors:
        travelled = measured.peak([corrector])
        assert travelled > 0.0, (
            f"corrector {corrector} never left its working point during the sweep -- a bump "
            "that moved no current is not a bump, whatever the BPMs read"
        )
        residual = abs(measured.offsets[corrector][terminal])
        bound = measured.bound(corrector, travelled)
        assert residual <= bound, (
            f"corrector {corrector} ended the run {residual:.3e} A off the working point it "
            f"started on (against a bound of {bound:.3e} A over the {travelled:.3e} A it "
            "travelled) -- the terminal step commands those working points back verbatim"
        )

    # --- (c continued) every step landed inside the band ---------------------
    # Targets and closure BPMs alike, against what THAT step asked for. This is
    # the plan's own convergence contract, re-checked from the recorded rows.
    desired = {stack.target_bpm: TARGET_BUMP_M, **dict.fromkeys(stack.closure_readbacks, 0.0)}
    for step, scale in enumerate(EXPECTED_SCALES):
        for bpm in stack.constrained_bpms:
            asked = scale * desired[bpm]
            got = measured.offsets[bpm][step]
            assert abs(got - asked) <= TOLERANCE_M, (
                f"step {step + 1} of {len(EXPECTED_SCALES)} (scale {scale:+g}) sat "
                f"{abs(got - asked):.3e} m from what it asked for at {bpm}: reached "
                f"{got:.3e} m against {asked:.3e} m requested, outside the "
                f"{TOLERANCE_M:.3e} m tolerance the run was given"
            )

    # --- (d) the plan's own figure, served live ------------------------------
    # A short wait rather than an immediate read: the run RECORD reaching
    # "completed" is the queue item's status, while `partial` comes from the
    # stop document reaching the bridge's document plane. The route reads the
    # source and never the record, deliberately, so the two facts can land in
    # either order.
    settled = _poll_figure(
        run_id,
        until=lambda figure: figure["partial"] is False,
        what='the live figure to settle ("partial": false)',
        timeout=SETTLED_FIGURE_TIMEOUT_SEC,
    )
    assert settled["source"] == "live", (
        f"the completed run should still be served from the live buffer: {_figure_summary(settled)}"
    )
    assert settled["reason"] is None, (
        "the settled figure fell back to the default view instead of the orbit_bump_sweep "
        f"plan's own render: {_figure_summary(settled)}"
    )
    assert [panel["title"] for panel in settled["panels"]] == list(EXPECTED_PANELS), (
        f"unexpected panels on the settled figure: {_figure_summary(settled)}"
    )

    # --- (e) the same figure, after the run leaves the bridge's memory -------
    # Restarting the bridge drops its in-process row buffer and nothing else:
    # the queueserver owns the TiledWriter subscription, so this cannot lose
    # documents, and the catalog, Redis and the VA all keep running. The poll
    # below is the real readiness probe, since it waits for a SETTLED figure
    # rather than for any 200.
    _orm_stack.restart_bridge(PROJECT_NAME, bridge_url=BRIDGE_URL)

    from_tiled = _poll_figure(
        run_id,
        until=lambda figure: figure["source"] == "tiled" and figure["partial"] is False,
        what='the settled Tiled figure ("source": "tiled", "partial": false)',
        timeout=TILED_FIGURE_TIMEOUT_SEC,
        give_up=lambda figure: (
            "the restarted bridge is still serving this run from its live buffer, so it "
            "kept the state the restart was supposed to drop -- every Tiled assertion "
            f"below would be vacuous: {_figure_summary(figure)}"
            if figure["source"] == "live"
            else None
        ),
    )
    assert from_tiled["reason"] is None, (
        "the figure read back from Tiled fell back to the default view instead of the "
        "orbit_bump_sweep plan's own render -- the plan identity stamped on the start "
        f"document is what carries the run's plan across the restart: "
        f"{_figure_summary(from_tiled)}"
    )
    _assert_figures_agree(settled, from_tiled)
