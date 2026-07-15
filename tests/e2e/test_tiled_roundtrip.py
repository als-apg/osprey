"""Phase 2 acceptance proof: scan data survives a Bluesky bridge restart via
the co-deployed Tiled catalog (PROPOSAL.md's success criterion 11).

Deploys the bridge + Tiled with the mock-devices demo scanner
(``bluesky.demo_runner=true``, ``bluesky.tiled_enabled=true`` — no virtual
accelerator, no EPICS, no QEMU: the demo scanner runs a real bluesky
RunEngine against in-process mock devices), runs a scan to completion,
restarts ONLY the bridge container, and reads the same run back through the
same agent-facing endpoint (``GET /runs/{run_id}/data``). The in-memory run
registry and live-row buffer die with the bridge; the only thing that can
make the post-restart read return anything is the durable Tiled catalog,
found via ``osprey_run_id`` stamped into the RunEngine start doc (task 2.3)
and searched for by ``_from_tiled`` (task 3.2/3.3) after the registry lookup
misses.

This is why the pre-restart poll waits for ``status == "completed"``, not
merely "promoted" or "running": ``TiledWriter`` caches events and flushes
only at the stop doc (or its own ``batch_size`` cap), so restarting before
completion would lose every cached event while ``tiled_degraded`` still read
``False`` (Landmine 4 in the research brief) — a proof that would pass
vacuously.

Container safety: every docker invocation below names an exact container or
image — never a wildcard, never ``system prune``/``--volumes``. Teardown
goes through ``osprey deploy down``, never a raw ``docker rm`` sweep. The
restart step names the ``<project>-bluesky-bridge`` container only; ``osprey deploy restart``
is never used here because it bounces every service, including Tiled, which
would defeat the whole proof.

Markers: no ``pytest.mark.flaky`` anywhere in this module — this file's
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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Deliberately distinct from test_scan_deploy.py's 18090 and
# test_va_substrate_equivalence.py's 18099 so all three can run concurrently
# on a shared dev machine without a port collision.
BRIDGE_PORT = 18101
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"
BRIDGE_IMAGE = "osprey-bluesky-bridge:local"
# The fixture builds/deploys under this project name; the compose template
# renders each container_name as ``<project>-<service>``
# (services/bluesky/docker-compose.yml.j2), so derive them rather than hardcode
# host-global names that break the moment the template is namespaced per-project.
PROJECT_NAME = "proj"
BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"
TILED_CONTAINER = f"{PROJECT_NAME}-bluesky-tiled"

# The demo scanner's mock device factory (devices/mock.py's build_devices())
# defaults to a single "det1" MockDetector when the bridge's app.py lifespan
# hook wires it with no explicit motor/detector names.
DEMO_DETECTOR = "det1"

# The bridge's promote route (POST /runs/{id}/promote) fails closed on an
# unset BLUESKY_PROMOTE_TOKEN. The control-assistant preset deploys with
# control_system.writes_enabled: true AND execution.execution_method: local,
# which deliberately gates auto-arming off for that token
# (container_lifecycle._local_exec_arming_unsafe) — a local unsandboxed agent
# could otherwise read the token straight out of .env and bypass the write
# gate. This e2e is a controlled test, not agent code, so it supplies its own
# token explicitly (the supported operator-provides-a-token path) rather than
# exercise that arming policy here. BLUESKY_TILED_API_KEY, by contrast, is on
# the local-exec-safe allowlist (it grants catalog access only, no
# write-capable bridge route) and auto-mints regardless — no need to supply
# it ourselves.
PROMOTE_TOKEN = "e2e-tiled-roundtrip-promote-token"

BUILD_TIMEOUT_SEC = 300
DEPLOY_UP_TIMEOUT_SEC = 600
HEALTH_TIMEOUT_SEC = 300.0
CONTAINER_HEALTH_TIMEOUT_SEC = 90.0  # docker healthcheck start_period + a few intervals
SCAN_TIMEOUT_SEC = 60.0

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


def _write_promote_token(project_dir: Path) -> None:
    """Append BLUESKY_PROMOTE_TOKEN to the project .env BEFORE ``osprey deploy
    up`` -- the bridge compose template passes it through from the project
    .env, same mechanism as task 4.2's substrate-equivalence e2e."""
    env_path = project_dir / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    env_path.write_text(existing + f"BLUESKY_PROMOTE_TOKEN={PROMOTE_TOKEN}\n", encoding="utf-8")


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build + co-deploy the bridge and Tiled; yield the project directory."""
    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("tiled_roundtrip_build")
    project_dir = base / PROJECT_NAME

    # `dispatch: null` drops control-assistant's default event-dispatcher
    # stack (Node + Claude CLI image) -- irrelevant to this proof and far
    # slower to build than the bridge/Tiled images already are. Only
    # top-level profile-key overrides go through -O (a flat-dotted `--set`
    # would build a nested dict for `dispatch`, not null it out).
    #
    # services.postgresql.port_host: the control-assistant preset always
    # deploys an ariel-postgres service (config.yml.j2 bakes it into
    # deployed_services unconditionally, for ARIEL logbook search -- there is
    # no profile knob to drop it), hardcoded to host port 5432. This proof
    # never touches ariel/postgres at all, but a colliding local Postgres
    # (or another project's container already bound to 5432) would abort
    # `deploy up` before the bridge/Tiled containers it creates alongside it
    # ever start. Moved to a high, unassigned port -- same defensive
    # convention as this fixture's own BRIDGE_PORT choice above.
    override_path = base / "override.yml"
    override_path.write_text(
        "dispatch: null\nconfig:\n  services.postgresql.port_host: 15432\n", encoding="utf-8"
    )

    # bluesky.port/demo_runner/tiled_enabled are all leaf scalars under the
    # top-level `bluesky:` profile key, so plain --set works (a dotted --set
    # only becomes unsafe for keys nested under an existing block you don't
    # want replaced wholesale, e.g. control_system.type -- not the case
    # here: we don't touch control_system at all, so the preset's default
    # `control_system.type: mock` stands, which is all the demo scanner
    # needs -- no CA, no virtual accelerator).
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
            f"bluesky.port={BRIDGE_PORT}",
            "--set",
            "bluesky.demo_runner=true",
            "--set",
            "bluesky.tiled_enabled=true",
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

    _write_promote_token(project_dir)

    # Force a fresh --dev build so the deployed bridge container runs CURRENT
    # source (osprey deploy up does not pass --build to compose, so it would
    # otherwise reuse a stale cached image). Exact-named image only.
    # E2E_REUSE_IMAGES=1 skips this (dev-only fast local iteration on the
    # test itself when the osprey source is unchanged; never set in CI).
    if not os.environ.get("E2E_REUSE_IMAGES"):
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
        _wait_for_container_health(BRIDGE_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)
        _wait_for_container_health(TILED_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)
        yield project_dir
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


# ---------------------------------------------------------------------------
# The round-trip proof (STRICT -- no flaky mark, see module docstring)
# ---------------------------------------------------------------------------


def test_tiled_roundtrip(deployed_stack: Path) -> None:
    # `deployed_stack` is requested for its side effect: it builds the project,
    # brings up bridge + Tiled, and waits for both to report healthy.

    # --- 2. create + promote a scan --------------------------------------
    status, body = _post(
        "/runs", {"plan_name": "count", "plan_args": {"detectors": [DEMO_DETECTOR], "num": 3}}
    )
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    status, body = _post(f"/runs/{run_id}/promote", {}, headers={"X-Promote-Token": PROMOTE_TOKEN})
    assert status == 200, f"promote failed: {status} {body}"

    # --- 3. poll to completion --------------------------------------------
    # Not politeness: TiledWriter caches events and flushes only at the stop
    # doc (or its own batch_size cap). Restarting before the run completes
    # would lose every cached event while tiled_degraded still reads False --
    # the poll is what makes the proof meaningful (see module docstring).
    deadline = time.monotonic() + SCAN_TIMEOUT_SEC
    status_body: dict = {}
    while time.monotonic() < deadline:
        _, status_body = _get(f"/runs/{run_id}")
        if status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(0.2)
    assert status_body.get("status") == "completed", f"scan did not complete: {status_body}"

    # --- 4. capture the pre-restart read; guard against a vacuous proof --
    assert status_body.get("tiled_degraded") is False, (
        f"tiled_degraded must be False for a meaningful round-trip proof: {status_body}"
    )

    status, pre_data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data (pre-restart) failed: {status} {pre_data}"
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

    # --- 5. restart ONLY the bridge; Tiled must stay up -------------------
    restart = subprocess.run(
        ["docker", "restart", BRIDGE_CONTAINER],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert restart.returncode == 0, f"docker restart {BRIDGE_CONTAINER} failed: {restart.stderr}"

    # --- 6. wait for the bridge to come back healthy -----------------------
    _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
    _wait_for_container_health(BRIDGE_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)

    # --- 7. read the SAME run id back through the SAME endpoint -----------
    # The in-memory registry (and run.run_uid with it) died with the bridge
    # process; the only thread back to this run's data is osprey_run_id,
    # stamped into the RunEngine start doc by do_promote and searched for by
    # _from_tiled via Key("start.osprey_run_id") == run_id.
    status, post_data = _get(f"/runs/{run_id}/data")
    assert status == 200, f"GET /runs/{run_id}/data (post-restart) failed: {status} {post_data}"

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
