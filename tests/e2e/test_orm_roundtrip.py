"""Real-container ORM round-trip e2e (task 5.2 / PROPOSAL.md's headline
Functionality criteria for FR1/FR2/FR10/FR11).

Deploys the turn-key plan-stack config (task 4.3, ``tests/e2e/_orm_stack.py``
-- the single source of this deploy shape, also reused by the agentic
discovery e2e in 5.3/5.4), drives the real ``orm`` plan over the bridge's
queue API (``PATCH /draft`` -> ``POST /queue/items`` -> armed
``POST /queue/start`` -> poll -> ``GET /runs/{id}/data``, spelled once in
``tests/e2e/_queue_drive.py``), and proves four things end to end:

  (a) the measured response matrix -- built via the SAME
      ``orm_analysis.build_response_matrix`` an MCP-side analysis step would
      use -- agrees with the independent ``lattice/response.py`` model oracle
      (mirrors task 5.1's in-process cross-check, but over the deployed
      HTTP+container stack rather than a direct ``PhysicsBridge`` call).
  (b) the run reaches a terminal "completed" status within a bounded
      timeout -- i.e. no corrector step ever hangs the bridge's
      ``ConnectorSettable.set()`` settle-wait (``devices/connector.py`` --
      the connector-mediated device layer replacing the old direct-CA
      ``EpicsMotor``). A corrector readback that never leaves 0.0 (the FR10
      regression this deploy config's ``echo-pyat-coupled-sp-to-rb`` fix
      addresses) blocks exactly there, so a fixture-level timeout on this
      assertion IS the regression signature, not an incidentally slow test.
  (c) ``GET /runs/{id}/figure`` serves the ``orm`` plan's OWN view of that run
      -- sweep traces while it is still filling in, then the fitted response
      matrix and the two anomaly-score panels once it settles -- computed over
      the run's FULL row set. This run is sized past ``get_run_data``'s
      100-row window on purpose (see ``NUM_POINTS``), so a figure route that
      quietly reused that window would draw short sweeps here and fail.
  (d) that same figure survives the run leaving the bridge's memory: after the
      bridge container is restarted (``_orm_stack.restart_bridge``) the live
      row buffer is gone, the route can only answer from the Tiled catalog,
      and the figure it draws from there matches the one it drew live.

No physics fault is seeded on this stack (no ``VA_BPM_ERRORS``/
``VA_CORR_GAIN`` in the written ``.env``),
so every BPM/corrector carries the identity error state
(``PhysicsBridge.__init__``'s default). The measured/model
agreement is therefore bounded only by AT numerical-solve reproducibility and
the JSON/HTTP round trip, not a physical noise floor -- see ``MATCH_ATOL``.

No preset channel names are hardcoded: correctors and BPMs are derived from
the deployment repo's own ``data/channel_limits.json`` -- the limits database
the build copies verbatim into the build zone, where the deployed containers
read it -- via ``_orm_stack.select_correctors``/``select_bpms`` (restricted to
the pyat-coupled partition, exactly the class of device the ``orm`` plan and
the model oracle both operate on). They reach the queueserver worker as the
device file this fixture authors before the build stages it, so the names a
plan here may address are exactly the names the worker registered.

Container safety: every docker invocation below names an exact
container/image -- never a wildcard, never ``system prune``/``--volumes``.
The one restart names a single compose service inside this deployment's own
project. Teardown goes through ``osprey down``, matching every other e2e in
this directory, followed by exact-named removal of this project's own volumes
(``tests/e2e/_volumes.py``): ``down`` keeps them by design, and a rerun must
not inherit their state.

Gating: needs Docker; the VA image builds natively for the host arch, so on
Apple Silicon PyAT/softioc compile from source (no prebuilt aarch64 wheels) --
slow (minutes) on a cold image cache. Runs on CI in the ``orm-roundtrip-e2e``
job, as the first of two sequential steps -- ``test_grid_scan_roundtrip.py``
follows it in the same job, so the two stacks never contend for 5064. Run
locally with ``E2E_REUSE_IMAGES=1`` set for fast iteration once the image cache
is warm.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from osprey.deployment.compose_generator import resolve_project_name
from osprey.services.bluesky_bridge.figure import rows_from_columnar
from osprey.services.bluesky_bridge.orm_analysis import build_response_matrix
from tests.e2e import _orm_stack, _queue_drive
from tests.e2e._deploy_diagnostics import dead_container_logs, queue_stack_logs
from tests.e2e._volumes import remove_project_volumes

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

#: Compose project this suite deploys under. Container names follow
#: ``<project>-<service>``, so the failure diagnostics below need the same name
#: the build is given -- hence one constant rather than two literals.
PROJECT_NAME = "orm-roundtrip"


def _dead_container_logs() -> str:
    """Logs from every container of this deployment that is not running."""
    return dead_container_logs(resolve_project_name({"project_name": PROJECT_NAME}))


BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 1200  # first-time native VA source build is slow (minutes)
HEALTH_TIMEOUT_SEC = 300.0

# Within the corrector channel_limits band (+-12A) and the response model's
# documented linear regime (ORMParams' own docstring); NUM_POINTS >= 3 per
# the schema's `ge=3`.
SPAN_A = 5.0

#: 6 correctors x 21 points = 126 rows, sized deliberately past the 100-row
#: default window of ``GET /runs/{id}/data``. The figure route computes over a
#: run's WHOLE row set -- a windowed ORM fit is silently wrong -- and this is
#: the run shape that can tell the two apart: read through the data window, the
#: last correctors' sweeps would come out short or missing entirely. Both stay
#: well under the 8-panel/12-series caps `plans_core/orm.py`'s render applies,
#: so every corrector gets its own trace panel and every BPM its own line.
CORRECTOR_COUNT = 6
NUM_POINTS = 21

# 6 correctors x 21 points x a 10-device (6 corrector + 4 BPM) bundle read per
# point. A healthy point costs a second or two, so a healthy run here is a few
# minutes; this is roughly fivefold headroom over that, which keeps it a
# meaningful "did it hang" gate (a hung settle-wait never returns at all)
# rather than a flaky timing assertion.
SCAN_TIMEOUT_SEC = 900.0

#: Interval for the in-flight figure poll. Shorter than the settled polls
#: because it is hunting for a window that closes: it has to land inside the
#: run rather than merely wait for a state that persists.
MIDFLIGHT_POLL_SEC = 0.5
FIGURE_POLL_SEC = 1.0

#: How long to wait for the run's FIRST recorded point. Much shorter than the
#: whole run on purpose: this wait is over as soon as one corrector has been
#: set and one bundle read, so a budget sized to the whole run would just be
#: the corrector-hang timeout spent under a less informative message.
MIDFLIGHT_TIMEOUT_SEC = 300.0

#: How long the live figure may take to report itself settled AFTER the run
#: record reads "completed". These are two different facts: the record is the
#: queue item's status, while ``partial`` comes from the run's stop document
#: reaching the bridge's document plane, which the route reads instead of the
#: record on purpose (see ``app.get_run_figure``).
SETTLED_FIGURE_TIMEOUT_SEC = 60.0

#: How long the restarted bridge may take to answer from Tiled. The health
#: wait inside ``restart_bridge`` proves only that the bridge process is back
#: up; whether the queueserver's ``TiledWriter`` has flushed this run to the
#: catalog is a separate, asynchronous fact, and this poll is what establishes
#: it (see that function's "WHAT THIS PROVES, AND WHAT IT DOES NOT").
TILED_FIGURE_TIMEOUT_SEC = 60.0

#: Tolerance for live-vs-Tiled figure parity. Both figures are drawn by the
#: same `render` over the same measurements, so the only thing between them is
#: the storage round trip -- float64 through a Tiled table on one side, float64
#: through JSON on the other, neither of which loses a bit. This is insurance
#: against a dtype narrowing somewhere in that path, not a real error budget.
PARITY_RTOL = 1e-9
PARITY_ATOL = 1e-12

# No VA_BPM_ERRORS/VA_CORR_GAIN are seeded on this stack (see module
# docstring) -- every device carries PhysicsBridge's identity error
# state, so there is no physical noise floor to size this against. The bound
# below is float round-trip/AT numerical-solve reproducibility margin, kept
# generous relative to task 5.1's probed in-process figure (4.9e-15 relative)
# to absorb the extra JSON/HTTP/container hop.
MATCH_RTOL = 1e-6
MATCH_ATOL = 1e-9  # meters


def _get(path: str) -> tuple[int, Any]:
    return _queue_drive.request(BRIDGE_URL, path, "GET")


class DeployedOrmStack:
    """Everything the round-trip test needs about the one deployment repo."""

    def __init__(
        self,
        repo: Path,
        correctors: dict[str, tuple[str, str]],
        bpms: dict[str, str],
    ):
        self.repo = repo
        self.correctors = correctors
        self.bpms = bpms


@pytest.fixture(scope="module")
def deployed_orm_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedOrmStack]:
    base = tmp_path_factory.mktemp("orm_roundtrip_build")

    # The plan devices are authored BETWEEN `init` and `build`: the build copies
    # <repo>/data into the build zone and stages the device file it finds there
    # for the queueserver worker, so a set written after the build would never
    # reach a container. Selected from the repo's own data/channel_limits.json —
    # the same bytes the build copies to build/data, and the only copy that
    # exists this early.
    correctors: dict[str, tuple[str, str]] = {}
    bpms: dict[str, str] = {}

    def author_devices(repo: Path) -> None:
        nonlocal correctors, bpms
        limits = _orm_stack.channel_limits(repo)
        correctors = _orm_stack.select_correctors(limits, CORRECTOR_COUNT)
        bpms = _orm_stack.select_bpms(limits)
        _orm_stack.write_devices_file(repo, correctors=correctors, bpms=bpms)

    # The deployment REPO: `osprey up` runs here, `.env` lives here, and the
    # render `osprey build` produced is `<repo>/build`.
    repo = _orm_stack.build_project_subprocess(
        PROJECT_NAME,
        output_dir=base,
        timeout=BUILD_TIMEOUT_SEC,
        pre_build=author_devices,
        # This module's own thousand-port block (see test_dispatch_deploy.py's
        # 20700 note): everything not pinned explicitly follows it instead of
        # landing on a real deployment's default 10000 block.
        port_base=21200,
    )
    _orm_stack.assert_devices_authored(correctors, bpms)

    # The repo root's `.env` — the deployment's whole secret store, and the file
    # `osprey up` refuses to start without.
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
        yield DeployedOrmStack(repo=repo, correctors=correctors, bpms=bpms)
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
# Reading figures off the deployed bridge
# ---------------------------------------------------------------------------


def _figure(run_id: str) -> tuple[int, Any]:
    """One ``GET /runs/{id}/figure``, returning ``(status, body)``.

    A shorter per-request timeout than `_queue_drive.request`'s 20 s default:
    this is called from 1 Hz polls with their own deadlines, and one hung
    request must not eat a whole poll budget by itself.
    """
    return _queue_drive.request(BRIDGE_URL, f"/runs/{run_id}/figure", "GET", timeout=10.0)


def _figure_summary(figure: dict[str, Any]) -> str:
    """A one-line digest of a figure response, for a failure message.

    Never the whole body: a settled figure is mostly heatmap numbers, and what
    a timed-out poll needs to report is which flags and which panels it kept
    seeing instead of the ones it wanted.
    """
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
    poll: float = FIGURE_POLL_SEC,
    give_up: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    """Poll ``GET /runs/{id}/figure`` until *until* accepts a response body.

    Non-200 answers are retried rather than raised: a run whose first document
    has not landed yet is in neither the live buffer nor the Tiled catalog, and
    the route's honest answer for that is a 404.

    *give_up* names the conditions under which waiting can no longer succeed --
    the run settling before any in-flight figure was seen, or the live buffer
    outliving a restart. Returning a message from it fails right there, with
    that message, rather than spending the rest of the deadline on a wait whose
    outcome is already decided.

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
        time.sleep(poll)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {what} on "
        f"GET /runs/{run_id}/figure (last seen: {last})"
    )


def _panel(figure: dict[str, Any], title: str) -> dict[str, Any]:
    """The one panel titled *title*, or an AssertionError naming what was there."""
    panels: list[dict[str, Any]] = figure["panels"]
    matches = [panel for panel in panels if panel["title"] == title]
    assert len(matches) == 1, (
        f"expected exactly one {title!r} panel, found {len(matches)} in "
        f"{[panel['title'] for panel in figure['panels']]}"
    )
    return matches[0]


def _sweep_panels(figure: dict[str, Any]) -> list[dict[str, Any]]:
    """The corrector-sweep trace panels.

    Keyed on the per-corrector title the plan builds (``f"{name} sweep"``) and
    not on the lines mark alone: the fit draws its own lines panel, "Response
    by BPM", whose series are correctors rather than BPMs. Matching every lines
    panel would sweep that one into the per-sweep assertions below, where it
    fails on a contract it was never meant to satisfy.
    """
    return [
        panel
        for panel in figure["panels"]
        if panel["mark"]["kind"] == "lines" and panel["title"].endswith(" sweep")
    ]


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
    ``figure.panels[6].mark.values[3][1]``) instead of a diff of two large
    dumps.
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
        "readbacks": list(bpms),
        "span_a": SPAN_A,
        "num": NUM_POINTS,
    }
    expected_rows = len(correctors) * NUM_POINTS

    # The module fixture waits for the RE worker environment once, which covers
    # the first pass. This test restarts the bridge at the end, though, and the
    # bridge opens that environment off its readiness path -- so on a `flaky`
    # rerun the enqueue below IS an enqueue after a restart, which would
    # otherwise be refused 409 "not in the list of allowed plans" against a
    # still-empty worker namespace. Re-waiting costs one request when the
    # environment is already open.
    _queue_drive.wait_for_worker_environment(BRIDGE_URL)

    token = _orm_stack.minted_launch_token(deployed_orm_stack.repo)
    run_id = _queue_drive.stage_and_enqueue(
        BRIDGE_URL, "orm", plan_args, client_id="orm-roundtrip-e2e"
    )
    _queue_drive.start_queue(BRIDGE_URL, token)

    # --- (c) the figure while the run is still filling in ------------------
    # One in-flight reading, taken at whatever moment the poll below first
    # finds a sweep drawn. Everything about that moment is a claim the panel
    # depends on: a figure exists before the run ends, it is the PLAN's figure
    # (`reason` is null -- a reason means the route fell back to the generic
    # view), it is served from the live buffer, and it says so by reporting
    # itself unsettled.
    try:
        midflight = _poll_figure(
            run_id,
            until=lambda figure: bool(figure["partial"]) and bool(_sweep_panels(figure)),
            what="an in-flight figure carrying at least one corrector sweep",
            timeout=MIDFLIGHT_TIMEOUT_SEC,
            poll=MIDFLIGHT_POLL_SEC,
            give_up=lambda figure: (
                None
                if figure["partial"]
                else (
                    f"the {expected_rows}-row run settled before any in-flight figure was "
                    f"observed (polled every {MIDFLIGHT_POLL_SEC}s) -- there is nothing wrong "
                    "with the route here, but this assertion is no longer testing what it "
                    "claims to; the run needs to be longer or the poll faster"
                )
            ),
        )
    except AssertionError as exc:
        # A run that never records a first row cannot produce an in-flight
        # figure either, so this wait is the first place the FR10 settle-wait
        # hang shows up -- earlier than the completion assertion below, and
        # with a message that would otherwise blame the figure route for it.
        # The run record is what tells the two apart, so it is attached here.
        record_status, record = _queue_drive.run_record(BRIDGE_URL, run_id)
        raise AssertionError(
            f"{exc}\n--- GET /runs/{run_id} (HTTP {record_status}) ---\n{record}\n"
            "If that record shows a run that never got going, this is the corrector "
            "settle-wait hang the completion assertion below names, not a figure-route "
            "failure."
        ) from exc
    assert midflight["source"] == "live", (
        f"an in-flight run must be served from the live row buffer: {_figure_summary(midflight)}"
    )
    assert midflight["partial"] is True, (
        f"an in-flight run must report itself unsettled: {_figure_summary(midflight)}"
    )
    assert midflight["reason"] is None, (
        "the in-flight figure fell back to the default view instead of the orm plan's own "
        f"render: {_figure_summary(midflight)}"
    )
    sweep_titles = [f"{name} sweep" for name in correctors]
    drawn = [panel["title"] for panel in _sweep_panels(midflight)]
    assert set(drawn) <= set(sweep_titles), (
        f"in-flight sweep panels {drawn} are not all correctors this run swept ({sweep_titles})"
    )
    assert all(
        series["points"] for panel in _sweep_panels(midflight) for series in panel["mark"]["series"]
    ), f"an in-flight sweep panel was drawn with no points on it: {_figure_summary(midflight)}"

    status_body = _queue_drive.wait_for_terminal_status(
        BRIDGE_URL, run_id, timeout=SCAN_TIMEOUT_SEC
    )

    # (b) no corrector-step hang: the poll above ran to a terminal status
    # within a bounded deadline. A corrector whose :RB never echoes its :SP
    # (the FR10 regression) blocks the bridge's ConnectorSettable.set()
    # settle-wait forever -- so a non-"completed" status here, after the
    # deadline, IS the failure this proves absent, not merely a slow run.
    assert status_body.get("status") == "completed", (
        f"orm run did not complete within {SCAN_TIMEOUT_SEC:.0f}s (status={status_body}) -- "
        "a corrector step whose :RB never echoes its :SP (the FR10 echo regression) hangs "
        "exactly here, at the bridge's ConnectorSettable.set() settle-wait"
    )

    # --- (c) the figure once the run has settled ---------------------------
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
        "the settled figure fell back to the default view instead of the orm plan's own "
        f"render: {_figure_summary(settled)}"
    )

    # Panel order is part of the wire contract the JS renderer and the MCP
    # projection both read: the fit leads, then every swept corrector's traces
    # in sweep order, demoted below it into their own section.
    assert [panel["title"] for panel in settled["panels"]] == [
        "Response matrix",
        "Response by BPM",
        "Corrector anomaly score",
        "BPM anomaly score",
        "Singular values",
        *sweep_titles,
    ], f"unexpected panels on the settled figure: {_figure_summary(settled)}"

    # THE point of the oversized run: a full sweep on every panel. Read through
    # get_run_data's 100-row window instead of the whole row set, the later
    # correctors' sweeps would be short here or missing altogether.
    for panel in _sweep_panels(settled):
        assert [series["label"] for series in panel["mark"]["series"]] == list(bpms), (
            f"{panel['title']}: one line per requested BPM was expected, got "
            f"{[series['label'] for series in panel['mark']['series']]}"
        )
        for series in panel["mark"]["series"]:
            assert len(series["points"]) == NUM_POINTS, (
                f"{panel['title']} / {series['label']}: {len(series['points'])} points, "
                f"expected {NUM_POINTS} -- on a {expected_rows}-row run this is the figure "
                "having been computed over a window of the rows rather than all of them, or "
                "a point whose corrector current was never recorded (a sample with no x is "
                "dropped from the trace)"
            )

    heatmap = _panel(settled, "Response matrix")["mark"]
    assert heatmap["kind"] == "heatmap", heatmap["kind"]
    assert heatmap["x_labels"] == list(correctors), (
        f"the heatmap's corrector axis is {heatmap['x_labels']}, expected every swept "
        f"corrector {list(correctors)} -- a missing column is a sweep the fit did not "
        "consider complete"
    )
    assert heatmap["y_labels"] == list(bpms), heatmap["y_labels"]
    assert len(heatmap["values"]) == len(bpms), (
        f"the heatmap has {len(heatmap['values'])} rows for {len(bpms)} BPMs"
    )
    assert all(len(row) == len(correctors) for row in heatmap["values"]), (
        f"the heatmap is ragged: row widths {[len(row) for row in heatmap['values']]}"
    )

    corrector_bars = _panel(settled, "Corrector anomaly score")["mark"]
    assert corrector_bars["kind"] == "bars", corrector_bars["kind"]
    assert corrector_bars["categories"] == list(correctors), corrector_bars["categories"]
    assert all(value is not None and math.isfinite(value) for value in corrector_bars["values"]), (
        f"the corrector anomaly scores are not all real numbers: {corrector_bars['values']}"
    )

    bpm_bars = _panel(settled, "BPM anomaly score")["mark"]
    assert bpm_bars["kind"] == "bars", bpm_bars["kind"]
    assert bpm_bars["categories"] == list(bpms), bpm_bars["categories"]
    assert all(value is not None and math.isfinite(value) for value in bpm_bars["values"]), (
        f"the BPM anomaly scores are not all real numbers: {bpm_bars['values']}"
    )

    # --- the recorded rows -------------------------------------------------
    # The default data window really is a window of this run, which is the
    # premise the sweep-length assertion above rests on.
    status, windowed = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {windowed}"
    assert windowed["row_count"] == expected_rows, (
        f"expected {expected_rows} rows (one per (corrector, current) point), "
        f"got {windowed['row_count']}: {windowed}"
    )
    assert windowed["truncated"] is True, (
        f"a {expected_rows}-row run should overflow this route's default window, but it "
        f"reported no truncation: {windowed['row_count']} rows, "
        f"{len(windowed['rows'])} returned"
    )
    assert len(windowed["rows"]) < windowed["row_count"], (
        f"the default window returned all {len(windowed['rows'])} rows of this run, so it "
        "no longer separates a windowed read from a whole-run one"
    )

    status, data = _get(f"/runs/{run_id}/data?max_rows={expected_rows}")
    assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
    assert data["truncated"] is False, f"asked for every row and still got a window: {data}"

    columns = data["columns"]
    rows = rows_from_columnar(columns, data["rows"], data["row_count"]).rows
    measured = build_response_matrix(rows, correctors=list(correctors), readbacks=list(bpms))

    # (a) matches the model oracle: the same symmetric-sweep currents the
    # deployed orm plan itself computes (plans_core/orm.py's build_plan), so
    # the model is driven identically to how the plan drove the real stack.
    # The plan kicks each corrector about its own pre-run working point; the
    # VA's correctors idle at 0 A, so here those kicks ARE these absolute
    # currents and the model needs no offset of its own.
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

    # The served heatmap carries that same physics. It comes from the figure
    # route's own fit (`sliced_response_matrix`, which slices each corrector's
    # sweep) rather than from `build_response_matrix` above, so agreement with
    # the model here is a second, independent path to the same slopes -- and
    # the statement that what an operator SEES is the measurement, not merely
    # a correctly shaped picture.
    drawn_matrix = np.array(heatmap["values"], dtype=float)
    assert np.allclose(drawn_matrix, model, rtol=MATCH_RTOL, atol=MATCH_ATOL), (
        "the response matrix the figure route DREW does not match the model oracle within "
        f"tolerance (rtol={MATCH_RTOL}, atol={MATCH_ATOL}):\n"
        f"drawn=\n{drawn_matrix}\nmodel=\n{model}\n"
        f"max abs diff={float(np.max(np.abs(drawn_matrix - model))):.3e}"
    )

    # --- (d) the same figure, after the run leaves the bridge's memory -----
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
        "the figure read back from Tiled fell back to the default view instead of the orm "
        f"plan's own render -- the plan identity stamped on the start document is what "
        f"carries the run's plan across the restart: {_figure_summary(from_tiled)}"
    )
    _assert_figures_agree(settled, from_tiled)
