"""Phase 2 acceptance proof: scan data survives a Bluesky bridge restart via
the co-deployed Tiled catalog (PROPOSAL.md's success criterion 11).

Deploys the queue stack -- bridge + ``bluesky-queueserver`` RE Manager + Redis
+ Tiled -- against the Virtual Accelerator, drives ONE scan to completion
through the queue (``PATCH /draft`` -> ``POST /queue/items`` -> token-gated
``POST /queue/start``), restarts ONLY the bridge container, and reads the same
run back through the same agent-facing endpoint (``GET /runs/{run_id}/data``).

The Virtual Accelerator is not incidental here. Plans execute in the
queueserver worker against real devices, so the ONLY connector that can produce
a run with data is one that can actually move something -- which is also why the
shipped ``control-assistant`` preset now defaults ``control_system.type`` to
``virtual_accelerator``. The mock connector would give a browse-only deployment
that refuses the enqueue outright, and there would be nothing to round-trip.

The live-row buffer is in-process: it dies with the bridge. So after the restart
the only thing that can make this read return anything is the durable Tiled
catalog, found via the ``osprey_run_id`` the enqueue path stamps into the queue
item's metadata (``queue_backend.RUN_ID_META_KEY``), which reaches the
RunEngine start document and is searched for by ``_from_tiled`` once the live
lookup misses. ``run_uid`` is the tell: it is ``null`` on the live path (the
bridge holds a buffer but never learned the uid the worker's RunEngine minted)
and POPULATED on the Tiled path (it is on the stored start document), so this
test asserts the read genuinely switched sources rather than merely returning
the same numbers.

This is why the pre-restart poll waits for ``status == "completed"``, not merely
"running": ``TiledWriter`` caches events and flushes only at the stop doc (or
its own ``batch_size`` cap), so restarting before completion would lose every
cached event -- a proof that would pass vacuously. (Relatedly, and deliberately
NOT asserted anywhere here: a bridge restarted MID-run legitimately has no live
buffer for that run until its next start document. That is correct behavior,
not a gap.)

Scope, versus ``tests/e2e/test_bluesky_queue_e2e.py``: that module is the broad
proof of the whole queue stack (capability, arming, serial drain, session
plans, abort, browse-only, network isolation) and it exercises this same
durability path as one stage among nine. THIS module is the narrow, strict
acceptance proof of criterion 11 alone -- one scan, one restart, byte-identical
read-back -- and keeps no flaky marker anywhere, so the criterion can never be
swept into a lenient rerun. Keep it that way if the two ever diverge.

Container safety: every docker invocation below names an exact container or
image -- never a wildcard, never ``system prune``/``--volumes``. Teardown
goes through ``osprey down``, never a raw ``docker rm`` sweep. The restart step
names the ``<project>-bluesky-bridge`` container only; ``osprey restart`` is
never used here because it bounces every service, including Tiled and the RE
manager, which would defeat the whole proof.

Markers: no ``pytest.mark.flaky`` anywhere in this module -- this file's
entire body IS the acceptance proof, so the round-trip assertions must stay
strict (mirrors ``test_va_substrate_equivalence.py``'s P5 convention: the
safety/acceptance proof is never swept into lenient reruns).
"""

from __future__ import annotations

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

from osprey.deployment.compose_generator import resolve_project_name

REPO_ROOT = Path(__file__).resolve().parents[2]

# Deliberately distinct from every sibling e2e module's pinned bridge port
# (test_bluesky_deploy.py's 18090, test_va_substrate_equivalence.py's 18099,
# _orm_stack.py's 18102, test_bluesky_queue_e2e.py's 18108) so they can run
# concurrently on a shared dev machine without a port collision.
BRIDGE_PORT = 18101
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"
TILED_PORT = 18192
# Channel Access port for this module's own Virtual Accelerator. The preset
# leaves the connector's gateway port UNSET so it follows
# services.virtual_accelerator.port -- moving the deployed soft-IOC's port is a
# one-place edit, and the connector follows it.
VA_CA_PORT = 15065
# ariel-postgres (5432) and OpenObserve (5080) are deployed unconditionally by
# the control-assistant preset with no profile knob to drop them, and a
# locally-running tutorial stack routinely holds both. A bound-port collision
# aborts `osprey up` before the bridge/Tiled containers it creates alongside
# them ever start, so both are moved to high, unassigned ports.
POSTGRES_PORT = 25433
OPENOBSERVE_PORT = 25082

# The deployment repo's directory name IS the deployment's name; the compose
# template renders each container_name AND the bridge's locally-built image as
# ``<project>-<service>`` (services/bluesky/docker-compose.yml.j2), so derive
# both (via resolve_project_name, exactly as the template does) rather than
# hardcode host-global names that break the moment the template is namespaced
# per-project.
PROJECT_NAME = "proj"
BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"
QUEUESERVER_CONTAINER = f"{PROJECT_NAME}-bluesky-queueserver"
TILED_CONTAINER = f"{PROJECT_NAME}-bluesky-tiled"
VA_CONTAINER = f"{PROJECT_NAME}-virtual-accelerator"
BRIDGE_IMAGE = f"{resolve_project_name({'project_name': PROJECT_NAME})}-bluesky-bridge:local"
VA_IMAGE = f"{resolve_project_name({'project_name': PROJECT_NAME})}-va:local"

# The scan: one corrector axis, one BPM detector, a handful of points. Small on
# purpose -- this proof is about durability, not about scan size.
SCAN_POINTS = 4

BUILD_TIMEOUT_SEC = 600
# The VA image compiles PyAT/softioc from source on a cold cache.
DEPLOY_UP_TIMEOUT_SEC = 2400
HEALTH_TIMEOUT_SEC = 420.0
CONTAINER_HEALTH_TIMEOUT_SEC = 240.0  # docker healthcheck start_period + a few intervals
SCAN_TIMEOUT_SEC = 420.0

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]


# ---------------------------------------------------------------------------
# Build/deploy scaffold (mirrors tests/e2e/test_va_substrate_equivalence.py's shape)
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


class DeployedStack:
    """The one deployed repo, plus the devices its scan will drive."""

    def __init__(
        self,
        repo: Path,
        correctors: dict[str, tuple[str, str]],
        bpms: dict[str, str],
        limits: dict[str, Any],
        token: str,
    ) -> None:
        self.repo = repo
        self.correctors = correctors
        self.bpms = bpms
        self.limits = limits
        self.token = token


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedStack]:
    """Build + co-deploy the VA, bridge, RE manager, Redis and Tiled."""
    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("tiled_roundtrip_build")
    repo = base / PROJECT_NAME

    # `dispatch: null` drops control-assistant's default event-dispatcher
    # stack (Node + Claude CLI image) -- irrelevant to this proof and far
    # slower to build than the VA/bridge/Tiled images already are. Only
    # top-level profile-key overrides go through -O/an override file (a flat
    # dotted `--set` would build a nested dict for `dispatch`, not null it out).
    #
    # modules.web_terminals.enabled: false drops the preset's per-persona
    # web-terminal stack (two persona images + nginx, all built locally):
    # nothing in this proof touches persona routing, and the deploy-up
    # credential preflight would otherwise abort for the personas' missing
    # LLM key. Persona coverage lives in the dedicated web-terminals lanes.
    # One dotted LEAF key on purpose -- overriding just `.enabled` leaves
    # the preset's `modules.web_terminals` siblings intact, whereas a nested
    # `modules:` mapping would wholesale-replace the subtree (same
    # convention as tests/e2e/_orm_stack.py).
    #
    # control_system.type is deliberately NOT set: the shipped preset already
    # defaults to virtual_accelerator, which is the connector this proof needs,
    # and pinning it here would hide a regression in that default.
    override_path = base / "override.yml"
    override_path.write_text(
        "dispatch: null\n"
        "config:\n"
        f"  services.postgresql.port_host: {POSTGRES_PORT}\n"
        f"  services.openobserve.port: {OPENOBSERVE_PORT}\n"
        "  modules.web_terminals.enabled: false\n",
        encoding="utf-8",
    )

    # bluesky.port/tiled_port/tiled_enabled and virtual_accelerator.port are
    # leaf scalars under top-level profile keys, so plain --set works (a dotted
    # --set is only unsafe for keys nested under an existing block you don't
    # want replaced wholesale, e.g. control_system.type -- not the case here).
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
            f"bluesky.port={BRIDGE_PORT}",
            "--set",
            "bluesky.tiled_enabled=true",
            "--set",
            f"bluesky.tiled_port={TILED_PORT}",
            "--set",
            f"virtual_accelerator.port={VA_CA_PORT}",
        ],
        cwd=base,
        timeout=BUILD_TIMEOUT_SEC,
    )
    if init.returncode != 0:
        pytest.fail(
            f"osprey init failed (rc={init.returncode}):\n"
            f"--- stdout ---\n{init.stdout}\n--- stderr ---\n{init.stderr}"
        )

    build = _run(
        [str(osprey_bin), "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
        cwd=base,
        timeout=BUILD_TIMEOUT_SEC,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )
    _seed_repo_env(repo)

    # Force a fresh --dev build so the deployed bridge container runs CURRENT
    # source (osprey up does not pass --build to compose, so it would
    # otherwise reuse a stale cached image). Exact-named images only.
    # E2E_REUSE_IMAGES=1 skips this (dev-only fast local iteration on the
    # test itself when the osprey source is unchanged; never set in CI).
    if not os.environ.get("E2E_REUSE_IMAGES"):
        for image in (BRIDGE_IMAGE, VA_IMAGE):
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)

    try:
        up = _run(
            [str(osprey_bin), "up", "-d", "--dev"],
            cwd=repo,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        _wait_for_container_health(BRIDGE_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)
        _wait_for_container_health(QUEUESERVER_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)
        _wait_for_container_health(TILED_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)

        # Device names come from the .env `osprey up` itself wrote
        # (_ensure_bluesky_substrate_env derives them from the render's own
        # channel_limits.json) -- never a hardcoded facility channel, and never a
        # second copy of that derivation living in this test.
        correctors = _parse_motors(_env_value(repo, "BLUESKY_EPICS_MOTORS"))
        bpms = _parse_detectors(_env_value(repo, "BLUESKY_EPICS_DETECTORS"))
        assert correctors and bpms, "up wired no scan devices into the repo's .env"

        yield DeployedStack(
            repo=repo,
            correctors=correctors,
            bpms=bpms,
            # The render's copy, not the source one the operator edits: it is the
            # file the deployed containers get, and the bounds below have to be
            # the bounds the bridge enforces.
            limits=json.loads(
                (repo / "build" / "data" / "channel_limits.json").read_text(encoding="utf-8")
            ),
            token=_env_value(repo, "BLUESKY_LAUNCH_TOKEN"),
        )
    finally:
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=600)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )


def _seed_repo_env(repo: Path) -> None:
    """Give the repo the ``.env`` ``osprey up`` refuses to start without.

    The repo root's ``.env`` is the deployment's whole secret store and the file
    every compose invocation is pointed at, so ``up`` aborts when it is absent.
    ``osprey init`` writes one only when the shell exports a key for the
    profile's provider, which this lane does not need — this is the ``cp
    .env.example .env`` the CLI itself recommends, done for the operator. The
    substrate values ``up`` mints (launch token, scan devices) are appended to
    whatever is here.
    """
    env_path = repo / ".env"
    if not env_path.exists():
        shutil.copy(repo / ".env.example", env_path)


def _env_value(repo: Path, key: str) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = repo / ".env"
    assert env_path.is_file(), f"no .env written at {env_path}"
    value = parse_dotenv_file(env_path).get(key)
    assert value, f"{key} missing/empty in the deployment repo's .env"
    return value


def _parse_motors(value: str) -> dict[str, tuple[str, str]]:
    """Parse ``BLUESKY_EPICS_MOTORS`` (``name=SP|RB,...``) back into a mapping."""
    out: dict[str, tuple[str, str]] = {}
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        name, _, spec = chunk.partition("=")
        sp, _, rb = spec.partition("|")
        out[name.strip()] = (sp.strip(), rb.strip())
    return out


def _parse_detectors(value: str) -> dict[str, str]:
    """Parse ``BLUESKY_EPICS_DETECTORS`` (``name=RB,...``) back into a mapping."""
    out: dict[str, str] = {}
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        name, _, rb = chunk.partition("=")
        out[name.strip()] = rb.strip()
    return out


def _wait_for_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:  # noqa: S310 - localhost
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
    ``start_period``), so an instant equality assert is racy. Also used after
    the bridge restart below, where there is no HTTP-readiness gate at all.
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


def _docker_inspect(container: str, fmt: str) -> str:
    proc = subprocess.run(
        ["docker", "inspect", "--format", fmt, container],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"docker inspect {container} failed: {proc.stderr}"
    return proc.stdout.strip()


def _request(
    path: str, method: str, body: dict | None = None, token: str | None = None
) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers: dict[str, str] = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Launch-Token"] = token
    req = urllib.request.Request(  # noqa: S310
        f"{BRIDGE_URL}{path}", data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _get(path: str) -> tuple[int, Any]:
    return _request(path, "GET")


# ---------------------------------------------------------------------------
# The round-trip proof (STRICT -- no flaky mark, see module docstring)
# ---------------------------------------------------------------------------


def test_tiled_roundtrip(deployed_stack: DeployedStack) -> None:
    # `deployed_stack` is requested for its side effect: it builds the project,
    # brings up the VA, bridge, RE manager, Redis and Tiled, and waits for them.

    # --- 1. this deployment must actually be able to execute ---------------
    # Not politeness either: on a browse-only deployment the enqueue below
    # would be refused and the whole round trip would have nothing to carry.
    # Asserting it here makes that failure mode say what it is.
    status, health = _get("/health")
    assert status == 200, f"GET /health failed: {status} {health}"
    assert health["capability"]["can_execute"] is True, (
        f"this proof needs an executable deployment: {health['capability']}"
    )

    # --- 2. compose the scan in the shared draft ---------------------------
    # One corrector axis swept across the middle half of its OWN
    # channel_limits band, reading one BPM. Device names and bounds both come
    # from the render, never from a hardcoded channel.
    axis_name = next(iter(deployed_stack.correctors))
    sp_address, _rb = deployed_stack.correctors[axis_name]
    entry = deployed_stack.limits[sp_address]
    lo, hi = float(entry["min_value"]), float(entry["max_value"])

    status, patched = _request(
        "/draft",
        "PATCH",
        {
            "plan_name": "grid_scan",
            "plan_args_patch": {
                "detectors": [next(iter(deployed_stack.bpms))],
                "axes": [
                    {
                        "setpoint": axis_name,
                        "start": lo + 0.375 * (hi - lo),
                        "stop": lo + 0.625 * (hi - lo),
                        "num_points": SCAN_POINTS,
                    }
                ],
            },
            "client_id": "tiled-roundtrip-e2e",
        },
    )
    assert status == 200, f"PATCH /draft failed: {status} {patched}"

    # --- 3. enqueue at the pinned revision, then arm the queue -------------
    # The two halves of what used to be one direct launch: `POST /queue/items`
    # takes plan_name/plan_args from the server-side draft snapshot AT this
    # revision (never from the request body) and mints the OSPREY run id;
    # `POST /queue/start` is the token-gated arming action that drains it.
    status, enqueued = _request("/queue/items", "POST", {"draft_revision": patched["revision"]})
    assert status == 200, f"POST /queue/items failed: {status} {enqueued}"
    run_id = enqueued["run_id"]

    status, started = _request("/queue/start", "POST", token=deployed_stack.token)
    assert status == 200, f"POST /queue/start failed: {status} {started}"

    # --- 4. poll to completion --------------------------------------------
    # Not politeness: TiledWriter caches events and flushes only at the stop
    # doc (or its own batch_size cap). Restarting before the run completes
    # would lose every cached event -- the poll is what makes the proof
    # meaningful (see module docstring).
    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    status_body: dict = {}
    while time.monotonic() < deadline:
        _, status_body = _get(f"/runs/{run_id}")
        if isinstance(status_body, dict) and status_body.get("status") in (
            "completed",
            "error",
            "stopped",
        ):
            break
        time.sleep(1.0)
    assert status_body.get("status") == "completed", f"scan did not complete: {status_body}"

    # --- 5. capture the pre-restart read; guard against a vacuous proof ----
    status, pre_data = _get(f"/runs/{run_id}/data?max_rows=1000")
    assert status == 200, f"GET /runs/{run_id}/data (pre-restart) failed: {status} {pre_data}"
    # The live buffer outlives the run, so this read MUST still be on the live
    # path -- `run_uid: null` is the tell. Without this the post-restart read
    # could be a second live read and the whole proof would say nothing about
    # Tiled at all.
    assert pre_data["run_uid"] is None, (
        f"expected the pre-restart read to come off the live buffer: {pre_data['run_uid']!r}"
    )
    pre_columns = pre_data["columns"]
    pre_rows = pre_data["rows"]
    pre_row_count = pre_data["row_count"]
    # Without this guard the whole proof would pass vacuously at 0 == 0 after
    # the restart -- a degraded/never-persisted writer would look identical
    # to a working one if both sides were simply empty.
    assert pre_row_count > 0, f"expected a non-zero pre-restart row count: {pre_data}"
    assert len(pre_rows) == pre_row_count, (
        f"pre-restart row_count ({pre_row_count}) does not match len(rows) "
        f"({len(pre_rows)}): {pre_data}"
    )

    # --- 6. restart ONLY the bridge; Tiled and the manager must stay up ----
    restart = subprocess.run(
        ["docker", "restart", BRIDGE_CONTAINER],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert restart.returncode == 0, f"docker restart {BRIDGE_CONTAINER} failed: {restart.stderr}"

    # --- 7. wait for the bridge to come back healthy -----------------------
    _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
    _wait_for_container_health(BRIDGE_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)

    # --- 8. read the SAME run id back through the SAME endpoint -----------
    # The live-row buffer died with the bridge process; the only thread back to
    # this run's data is osprey_run_id, carried in the queue item's metadata
    # into the RunEngine start doc and searched for by _from_tiled via
    # Key("start.osprey_run_id") == run_id.
    status, post_data = _get(f"/runs/{run_id}/data?max_rows=1000")
    assert status == 200, f"GET /runs/{run_id}/data (post-restart) failed: {status} {post_data}"

    # The source genuinely switched: Tiled has the stored start document and so
    # can report the RunEngine's own uid, which the live buffer never knew.
    assert post_data["run_uid"], (
        f"the post-restart read must come from Tiled (run_uid populated); a null "
        f"run_uid means it served a live buffer that should not exist: {post_data}"
    )

    assert post_data["columns"] == pre_columns, (
        f"columns diverged across the restart:\npre:  {pre_columns}\npost: {post_data['columns']}"
    )
    assert post_data["row_count"] == pre_row_count, (
        f"row_count diverged across the restart: pre={pre_row_count} post={post_data['row_count']}"
    )
    # Content, not just count: an identical row_count with different row
    # values would pass a length-only check.
    assert post_data["rows"] == pre_rows, (
        f"row content diverged across the restart:\npre:  {pre_rows}\npost: {post_data['rows']}"
    )
