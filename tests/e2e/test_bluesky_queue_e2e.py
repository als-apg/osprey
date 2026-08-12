"""Real-container end-to-end proof of the whole Bluesky *queue* stack.

This is the acceptance instrument for the queue-backed scan stack: a fresh
``osprey build`` of the shipped ``control-assistant`` preset, deployed with
``osprey deploy up --dev``, driven through the surface an operator (or the
agent, or a panel) actually uses -- ``PATCH /draft`` -> ``POST /queue/items``
-> ``POST /queue/start`` -> ``GET /runs`` -> ``GET /runs/{id}/data`` -- against
real containers: the bluesky bridge, the ``bluesky-queueserver`` RE Manager,
its Redis, the co-deployed Tiled catalog, the Virtual Accelerator soft-IOC,
and the bluesky-panels sidecar.

Why this file exists at all: every other test of this stack drives a *mocked*
queue client. Those cover OSPREY's half of a two-party contract. The manager,
Redis durability, the CurveZMQ control socket, the document plane, and the
capability record's dependence on the deployed connector are the other half,
and only a real deploy exercises them.

**The preset is the subject, not a fixture detail.** ``control-assistant`` now
defaults ``control_system.type`` to ``virtual_accelerator``, so this module
deliberately does NOT override it (unlike ``tests/e2e/_orm_stack.py``, written
when ``mock`` was the default): stage 1 asserts the *shipped default* is the
executable one. The three ``--override`` entries and four ``--set`` ports are
host-hygiene only -- drop the event-dispatcher/web-terminal stacks this proof
never touches, and move every published port off the defaults so this can run
on a shared dev machine beside an already-deployed tutorial stack.

Stages, in order, each an independently-reportable test:

1. ``test_1_capability_*``       -- bridge ``/health`` and the sidecar's
   ``/bridge/health`` relay both report ``can_execute: true`` / ``executable``.
2. ``test_2_enqueue_*``          -- two grid scans from two draft revisions; a
   second enqueue of the SAME revision is refused ``draft_revision_already_launched``.
3. ``test_3_start_*``            -- ``POST /queue/start`` is token-gated
   (``launch_token_required``), the queue drains STRICTLY SERIALLY, and live
   rows accumulate at the poller's ~1 s cadence while a plan runs.
4. ``test_4_results_*``          -- both runs' data reads back off the live
   buffer (``run_uid: null``, the live path).
5. ``test_5_session_plan_*``     -- author -> validate (PASS + uploaded) ->
   stage in the draft -> enqueue -> drain; and a session plan whose validated
   bytes no longer match disk is refused ``session_plan_unvalidated``.
6. ``test_6_abort_*``            -- the emergency halt: a long scan is aborted
   with NO token, its record reaches ``stopped``, the manager returns to a
   startable state, and an abort with nothing running is ``nothing_running``.
7. ``test_7_restart_*``          -- restarting ONLY the bridge preserves queue
   and history (they live in Redis, not in the bridge), and the completed runs'
   data now serves off the DURABLE Tiled path (``run_uid`` populated).
8. ``test_8_mock_flip_*``        -- ``osprey config set-control-system mock`` +
   redeploy: every container still healthy, ``/health`` still 200 but
   ``can_execute: false`` / ``browse_only_connector``, and enqueue refused --
   a browse-only deployment never holds items it could never run.
9. ``test_9_security_*``         -- Redis is unreachable from ``osprey-network``;
   the queueserver CONTROL socket refuses a client without the server public
   key (the SOLE credential -- see below); and the document plane refuses a
   publisher with no client certificate.

**Refusals are asserted on ``detail.code``, never on the status code alone.**
Every refusal this stack emits is ``{"detail": {"code": <machine-readable>,
"detail": <sentence>, ...}}``; a status-code-only assertion passes while the
body drifts, which is exactly the class of regression this file is here to
catch.

**Ordering is load-bearing, and the fixture is module-scoped**: one build and
one deploy back every stage (a per-test deploy would take hours). Later stages
consume earlier ones' run ids through the module-level ``_S`` state object and
``pytest.skip`` when a prerequisite never happened, so a failure in stage 3
reports as one failure plus honest skips rather than eight cascading errors.
Stage 8 leaves the deployment on the mock connector on purpose -- it is the
last stage that needs an executable one, and stage 9's probes are connector-
independent.

**Not asserted, deliberately**: a bridge restart mid-run loses that run's live
rows until its next start document. That is CORRECT -- live rows are in-process
and the queue/history are the Redis-durable part -- so stage 7 restarts only
between runs and reads the completed runs back through Tiled.

**Control-socket credential model** (drives stage 9): the RE manager runs
``CURVE_SERVER=1`` with no authenticator and no client allowlist, while
``bluesky-queueserver-api``'s client authenticates with a FIXED keypair
compiled into the package. The server PUBLIC key is therefore the SOLE
credential for driving the queue -- which is why stage 9 probes the control
socket *without* it, and why that key must never reach panels, the sidecar, or
a log line.

CONTAINER SAFETY: every docker invocation below names an EXACT container,
image, or network belonging to this test's own project. Never a wildcard,
never ``system prune``, never ``volume rm``. Teardown goes through the shipped
``osprey deploy down``.

Gating: needs Docker. Lives in ``tests/e2e/`` so the fast lane
(``pytest tests/ --ignore=tests/e2e``) never collects this ~20-minute
build+deploy; run it by name --
``uv run pytest tests/e2e/test_bluesky_queue_e2e.py -v`` -- never with
``-m e2e`` (the marker selection leaks registries; see the repo convention).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from osprey.services.bluesky_bridge.queue_backend import (
    REASON_BROWSE_ONLY_CONNECTOR,
    REASON_EXECUTABLE,
)
from osprey.services.bluesky_bridge.session_upload import REASON_UNVALIDATED
from tests.e2e import _orm_stack

# ---------------------------------------------------------------------------
# Ports + names. Every published port is deliberately distinct from BOTH the
# preset defaults (8090/8091/8095/5064/5432/5080) and every sibling e2e
# module's pinned port (18090/18095/18099/18101/18102/18103/18105/18106), so
# this module can run on a shared dev machine beside an already-deployed
# tutorial stack without touching -- or being blocked by -- anything it does
# not own.
# ---------------------------------------------------------------------------
BRIDGE_PORT = 18108
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"
PANELS_PORT = 18096
PANELS_URL = f"http://localhost:{PANELS_PORT}"
TILED_PORT = 18191
VA_CA_PORT = 15064
POSTGRES_PORT = 25432
OPENOBSERVE_PORT = 25080

PROJECT_NAME = "queue-e2e"
BRIDGE_CONTAINER = f"{PROJECT_NAME}-bluesky-bridge"
QUEUESERVER_CONTAINER = f"{PROJECT_NAME}-bluesky-queueserver"
REDIS_CONTAINER = f"{PROJECT_NAME}-bluesky-redis"
TILED_CONTAINER = f"{PROJECT_NAME}-bluesky-tiled"
PANELS_CONTAINER = f"{PROJECT_NAME}-bluesky-panels"
VA_CONTAINER = f"{PROJECT_NAME}-virtual-accelerator"

# Every container this proof asserts healthy after the mock flip (stage 8).
_STACK_CONTAINERS = (
    BRIDGE_CONTAINER,
    QUEUESERVER_CONTAINER,
    REDIS_CONTAINER,
    TILED_CONTAINER,
    PANELS_CONTAINER,
    VA_CONTAINER,
)

BRIDGE_IMAGE = _orm_stack.bridge_image(PROJECT_NAME)
VA_IMAGE = _orm_stack.va_image(PROJECT_NAME)
PANELS_IMAGE = _orm_stack.panels_image(PROJECT_NAME)

BUILD_TIMEOUT_SEC = 600
# The VA image compiles PyAT/softioc from source on a cold cache, and the
# bridge/queueserver share an image that bakes in the whole bluesky stack.
DEPLOY_UP_TIMEOUT_SEC = 2400
HEALTH_TIMEOUT_SEC = 420.0
CONTAINER_HEALTH_TIMEOUT_SEC = 240.0
# Opening the RE worker environment connects every substrate device over
# Channel Access, which takes tens of seconds on a cold stack.
WORKER_ENV_TIMEOUT_SEC = 300.0
# One queued plan's wall-clock budget, and the whole-queue drain budget.
RUN_TIMEOUT_SEC = 420.0
DRAIN_TIMEOUT_SEC = 900.0
# How long a document published straight at the bridge's 0MQ proxy may take to
# show up on the read surface (proxy hop + the dispatcher's poll).
DOC_PLANE_ARRIVAL_TIMEOUT_SEC = 45.0

# Grid sizes, calibrated against the real VA rather than guessed: a
# connector-mediated grid_scan on this stack runs at ~18 points/second
# (measured: 2000 points in 112 s). That number is what these three constants
# are for.
#
#   SHORT          — drains in well under a second; used wherever only the
#                    fact of a completed run matters.
#   LIVE_SAMPLE    — must stay under way for TENS of seconds so ~1 s polling
#                    can actually observe rows accumulating. At 12 points the
#                    whole scan finished between two polls and stage 3 failed
#                    for a sampling artifact, not a product fault; 600 points
#                    is ~34 s, i.e. ~30 samples.
#   LONG           — must still be running when stage 6 aborts it, with room
#                    for the abort's own pause-poll composition. ~4 minutes.
#
# All three sweep a narrow band inside the corrector's own channel_limits,
# derived from the built project (never a hardcoded channel).
SHORT_POINTS = 4
LIVE_SAMPLE_POINTS = 600
LONG_POINTS = 4000

_LAUNCH_TOKEN_HEADER = "X-Launch-Token"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]


# ---------------------------------------------------------------------------
# Cross-stage state. Deliberately NOT a fixture: a fixture would have to be
# module-scoped to survive between tests anyway, and this makes the
# "stage N needs what stage M produced" dependency explicit at every
# `pytest.skip` call site.
# ---------------------------------------------------------------------------
@dataclass
class _Shared:
    run_short: str | None = None
    run_live: str | None = None
    run_session: str | None = None
    run_aborted: str | None = None
    # run id -> the rows read off the LIVE buffer, for the post-restart
    # Tiled comparison in stage 7.
    live_rows: dict[str, list[Any]] = field(default_factory=dict)
    live_columns: dict[str, list[str]] = field(default_factory=dict)
    session_plan_name: str | None = None


_S = _Shared()


@dataclass
class QueueStack:
    """Everything the stages need about the one deployed project."""

    project_dir: Path
    osprey_bin: Path
    correctors: dict[str, tuple[str, str]]
    bpms: dict[str, str]
    limits: dict[str, Any]
    token: str


# ---------------------------------------------------------------------------
# Process + HTTP helpers (same shapes as the sibling e2e modules)
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


def _request(
    base: str,
    path: str,
    method: str,
    body: dict[str, Any] | None = None,
    *,
    token: str | None = None,
    timeout: float = 20.0,
) -> tuple[int, Any]:
    """One HTTP call, returning ``(status, parsed_body)`` for 2xx AND 4xx/5xx.

    Refusals carry the body this suite asserts on, so an ``HTTPError`` is a
    normal result here, never an exception to propagate.

    ``body=None`` on a POST sends a genuinely EMPTY body (``b""``, so the
    request still carries ``Content-Length: 0``) with NO content-type -- which
    is what the bodyless routes take, and what proves ``POST /queue/abort``
    needs no parameters at all. Passing ``data=None`` instead would make urllib
    emit a POST with no length header, which is a different (and less
    representative) request.
    """
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers: dict[str, str] = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    elif method != "GET":
        data = b""
    if token:
        headers[_LAUNCH_TOKEN_HEADER] = token
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{base}{path}", data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except ValueError:
                return resp.status, raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw.decode("utf-8"))
        except ValueError:
            return exc.code, raw.decode("utf-8", errors="replace")


def _get(path: str, **kw: Any) -> tuple[int, Any]:
    return _request(BRIDGE_URL, path, "GET", **kw)


def _post(path: str, body: dict[str, Any] | None = None, **kw: Any) -> tuple[int, Any]:
    return _request(BRIDGE_URL, path, "POST", body, **kw)


def _sidecar_get(path: str, **kw: Any) -> tuple[int, Any]:
    return _request(PANELS_URL, path, "GET", **kw)


def _code_of(body: Any) -> Any:
    """The machine-readable refusal code off a bridge error body.

    Every refusal on this surface is ``{"detail": {"code": ..., ...}}``.
    Returns ``None`` rather than raising for any other shape, so an assertion
    failure reports the drifted body instead of a ``KeyError`` from the helper.
    """
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    if isinstance(detail, dict):
        return detail.get("code")
    return None


def _detail_of(body: Any) -> dict[str, Any]:
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, dict) else {}


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


def _docker_inspect(container: str, fmt: str) -> str:
    proc = subprocess.run(
        ["docker", "inspect", "--format", fmt, container],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"docker inspect {container} failed: {proc.stderr}"
    return proc.stdout.strip()


def _wait_for_container_health(container: str, timeout: float) -> None:
    """Poll ``docker inspect .State.Health.Status`` until ``healthy``.

    An HTTP-readiness gate can pass while Docker still reports ``starting``
    (the healthcheck only runs on its interval, after ``start_period``), so an
    instant equality assert would be racy.
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


def _env_value(project_dir: Path, key: str) -> str:
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path}"
    value = parse_dotenv_file(env_path).get(key)
    assert value, f"{key} missing/empty in the project .env"
    return value


def _parse_motors(value: str) -> dict[str, tuple[str, str]]:
    """Parse ``BLUESKY_EPICS_MOTORS`` (``name=SP|RB,...``) back into a mapping.

    Read from the .env that ``osprey deploy up`` itself wrote rather than
    re-derived here: the device names this test composes plans against are then
    exactly the ones the deployed worker registered, by construction.
    """
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


# ---------------------------------------------------------------------------
# Draft + queue helpers
# ---------------------------------------------------------------------------


def _grid_args(stack: QueueStack, num_points: int) -> dict[str, Any]:
    """Minimal ``grid_scan`` args: one corrector axis, one BPM detector.

    The sweep band is the middle half of the corrector's OWN
    ``channel_limits.json`` entry, so this never hardcodes a facility channel
    and never asks the reference monitor for a value outside its band.
    """
    axis_name = next(iter(stack.correctors))
    sp_address, _rb = stack.correctors[axis_name]
    entry = stack.limits[sp_address]
    lo, hi = float(entry["min_value"]), float(entry["max_value"])
    start = lo + 0.375 * (hi - lo)
    stop = lo + 0.625 * (hi - lo)
    return {
        "detectors": [next(iter(stack.bpms))],
        "axes": [
            {"setpoint": axis_name, "start": start, "stop": stop, "num_points": num_points},
        ],
    }


def _patch_draft(plan_name: str, plan_args: dict[str, Any]) -> int:
    """Replace the shared draft with ``plan_name``/``plan_args``; return the revision.

    ``expected_plan_name`` is deliberately NOT sent: this suite is the only
    writer, and a compare-and-set here would turn an unrelated draft edit into
    a confusing 409 rather than the plain overwrite the stages intend.
    """
    status, body = _request(
        BRIDGE_URL,
        "/draft",
        "PATCH",
        {"plan_name": plan_name, "plan_args_patch": plan_args, "client_id": "queue-e2e"},
    )
    assert status == 200, f"PATCH /draft failed: {status} {body}"
    assert body["plan_name"] == plan_name, f"draft did not take the plan name: {body}"
    revision = body["revision"]
    assert isinstance(revision, int), f"no integer revision on the PATCH response: {body}"
    return revision


def _draft_snapshot() -> dict[str, Any]:
    status, body = _get("/draft")
    assert status == 200, f"GET /draft failed: {status} {body}"
    return body


def _enqueue(revision: int, *, token: str | None = None) -> tuple[int, Any]:
    return _post("/queue/items", {"draft_revision": revision}, token=token)


def _enqueue_ok(revision: int, *, token: str | None = None) -> str:
    status, body = _enqueue(revision, token=token)
    assert status == 200, f"POST /queue/items failed: {status} {body}"
    run_id = body.get("run_id")
    assert run_id, f"no run_id in the enqueue response: {body}"
    assert body.get("revision") == revision, f"enqueue pinned a different revision: {body}"
    return str(run_id)


def _queue_snapshot() -> dict[str, Any]:
    status, body = _get("/queue")
    assert status == 200, f"GET /queue failed: {status} {body}"
    return body


def _run_record(run_id: str) -> tuple[int, Any]:
    return _get(f"/runs/{run_id}")


def _wait_for_run_status(
    run_id: str, wanted: tuple[str, ...], timeout: float, poll: float = 1.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        status, last = _run_record(run_id)
        if status == 200 and isinstance(last, dict) and last.get("status") in wanted:
            return last
        time.sleep(poll)
    raise AssertionError(
        f"run {run_id} never reached {wanted} within {timeout:.0f}s (last record: {last})"
    )


def _wait_for_manager_state(wanted: tuple[str, ...], timeout: float) -> str:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = _queue_snapshot()["status"].get("manager_state")
        if last in wanted:
            return str(last)
        time.sleep(1.0)
    raise AssertionError(f"manager never reached {wanted} within {timeout:.0f}s (last: {last!r})")


def _wait_for_worker_environment(timeout: float) -> dict[str, Any]:
    """Block until the manager reports an OPEN worker environment.

    Container health is not enqueue readiness. The bridge opens the RE worker
    environment in a background task deliberately excluded from readiness, and
    `POST /queue/items` validates against `plans_allowed` -- which the manager
    downloads from the worker only at that open. Enqueueing before it lands is
    refused 409 "not in the list of allowed plans", which reads like a
    permissions problem and is not one (the shipped permissions allow
    `[":.*"]`; the list was empty because the namespace was).

    `manager_state` is the wrong field to wait on -- it reads `idle` both
    before the environment has ever opened and after it is up.

    Spelled here rather than imported from `_queue_drive` for this module's
    standing reason: the acceptance instrument for the queue surface must not
    be written in terms of a helper that assumes that surface works.
    """
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = _queue_snapshot()["status"]
        if last.get("worker_environment_exists"):
            return dict(last)
        time.sleep(2.0)
    raise AssertionError(
        f"the RE worker environment never opened within {timeout:.0f}s -- the queue "
        f"cannot accept plans until it does (last manager status: {last!r})"
    )


# ---------------------------------------------------------------------------
# Fixture: one build, one deploy, for every stage
# ---------------------------------------------------------------------------


def _override_yaml() -> str:
    """Host hygiene and CI sizing ONLY -- never ``control_system.type``.

    ``dispatch: null`` drops the event-dispatcher stack (Node + Claude CLI
    image) and ``modules.web_terminals.enabled: false`` drops the per-persona
    web-terminal stack: neither is touched by this proof and both are slow to
    build (same convention as ``_orm_stack.override_yaml``). The two port keys
    move ariel-postgres and OpenObserve -- services the preset deploys
    unconditionally, with no profile knob -- off 5432/5080, which a locally
    running tutorial deploy routinely holds.

    ``_orm_stack.VA_ARCHIVER_CI_KNOBS`` shrinks the archive the preset's
    ``va_archiver:`` block declares (see the constant). It is a sizing override,
    not a behavioral one: the store and its recorder still deploy, still record,
    and still hold both tiers -- there is just far less seeded history to write
    first, none of which this proof reads. Nothing here touches what the stack
    IS, which is the property the docstring above is about: this module inherits
    ``control_system.type`` from the preset on purpose, so that a preset that
    stopped defaulting to the virtual accelerator would fail these stages rather
    than be papered over here.

    Written as flat dotted-string keys under ``config:`` (the preset's own
    convention): a ``--set`` would build a NESTED dict for every dotted segment
    and replace the whole ``services:`` block.
    """
    return (
        "dispatch: null\n"
        "config:\n"
        f"  services.postgresql.port_host: {POSTGRES_PORT}\n"
        f"  services.openobserve.port: {OPENOBSERVE_PORT}\n"
        "  modules.web_terminals.enabled: false\n" + _orm_stack.VA_ARCHIVER_CI_KNOBS
    )


def _drain_leftover_queue_items() -> None:
    """Empty the manager's PENDING queue before the stages start.

    The queue is Redis-backed, and that Redis lives in a compose NAMED VOLUME
    keyed on the project name. ``osprey deploy down`` removes containers and
    networks but NOT volumes, so a second run of this module against the same
    project name inherits whatever the previous run left queued -- including,
    by design, the plan stage 6 aborted (upstream requeues an interrupted plan;
    see `runs._queue_status`). Stage 2 asserts the queue holds exactly the two
    items it enqueued, so inherited work would fail it for a reason that has
    nothing to do with the code under test.

    Draining is the honest fix rather than making those assertions relative:
    the properties under test are what this stack does with the work THIS run
    submits, and starting from a known state is what makes "exactly two items"
    mean something. HISTORY is deliberately left alone -- it is append-only,
    every stage that reads it looks up its own run ids, and clearing it would
    throw away the very durability stage 7 depends on.

    Only pending items are removed, one by one through the bridge's own
    ``DELETE /queue/items/{uid}`` -- never by reaching into Redis, which would
    be a different (and untested) path from the one an operator has.
    """
    queue = _queue_snapshot()
    leftovers = queue["items"]
    if not leftovers:
        return
    print(  # noqa: T201 - surface inherited state in the run log
        f"[fixture] draining {len(leftovers)} queue item(s) left by an earlier run "
        f"(the Redis volume outlives `deploy down`)"
    )
    for item in leftovers:
        uid = item.get("item_uid")
        if not isinstance(uid, str):
            continue
        status, body = _request(BRIDGE_URL, f"/queue/items/{uid}", "DELETE")
        assert status == 200, f"could not drain leftover queue item {uid}: {status} {body}"

    remaining = _queue_snapshot()["status"]["items_in_queue"]
    assert remaining == 0, f"queue still holds {remaining} item(s) after draining leftovers"


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[QueueStack]:
    """Build + ``osprey deploy up --dev`` the whole queue stack; tear it down after."""
    osprey_bin = _orm_stack.find_osprey_console_script()
    base = tmp_path_factory.mktemp("bluesky_queue_e2e")
    project_dir = base / PROJECT_NAME

    override_path = base / "override.yml"
    override_path.write_text(_override_yaml(), encoding="utf-8")

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
            f"bluesky.tiled_port={TILED_PORT}",
            "--set",
            f"bluesky_panels.port={PANELS_PORT}",
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

    # Force fresh --dev builds so the deployed containers run CURRENT source
    # (`osprey deploy up` does not pass --build to compose, so it would
    # otherwise reuse a stale cached image). Exact-named images only.
    # E2E_REUSE_IMAGES=1 skips this for fast local iteration on the test
    # itself; never set it in CI, where a source change must always rebuild.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        for image in (BRIDGE_IMAGE, VA_IMAGE, PANELS_IMAGE):
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)

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
        _wait_for_health(f"{PANELS_URL}/health", HEALTH_TIMEOUT_SEC)
        _wait_for_container_health(QUEUESERVER_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)
        _wait_for_container_health(TILED_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)
        _wait_for_worker_environment(WORKER_ENV_TIMEOUT_SEC)

        _drain_leftover_queue_items()

        # Device names come from the .env `osprey deploy up` itself wrote
        # (_ensure_bluesky_substrate_env derives them from the built project's
        # own channel_limits.json) -- so the plans this test composes name
        # exactly the devices the deployed worker registered, and a change in
        # that derivation shows up here as a real failure rather than a
        # silently-diverging second copy of the same logic.
        correctors = _parse_motors(_env_value(project_dir, "BLUESKY_EPICS_MOTORS"))
        bpms = _parse_detectors(_env_value(project_dir, "BLUESKY_EPICS_DETECTORS"))
        assert correctors, "deploy up wrote no BLUESKY_EPICS_MOTORS -- no device to scan"
        assert bpms, "deploy up wrote no BLUESKY_EPICS_DETECTORS -- no device to read"

        limits = json.loads(
            (project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8")
        )

        yield QueueStack(
            project_dir=project_dir,
            osprey_bin=osprey_bin,
            correctors=correctors,
            bpms=bpms,
            limits=limits,
            token=_env_value(project_dir, "BLUESKY_LAUNCH_TOKEN"),
        )
    finally:
        down = _run([str(osprey_bin), "deploy", "down"], cwd=project_dir, timeout=600)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey deploy down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )


# ===========================================================================
# Stage 1 -- capability
# ===========================================================================


def test_1_capability_is_executable_on_the_shipped_preset(stack: QueueStack) -> None:
    """``/health`` reports an EXECUTABLE deployment -- with no connector override.

    The reason code is imported from ``queue_backend`` rather than spelled as a
    literal: the wire contract says consumers import these constants, and a
    hardcoded copy here would keep passing through a rename that broke every
    real consumer.
    """
    status, body = _get("/health")
    assert status == 200, f"GET /health failed: {status} {body}"
    assert body["status"] == "ok", f"bridge not ok: {body}"

    capability = body["capability"]
    assert capability["can_execute"] is True, (
        "the shipped control-assistant preset must deploy EXECUTABLE (it defaults "
        f"control_system.type to virtual_accelerator): {capability}"
    )
    assert capability["reason"] == REASON_EXECUTABLE, f"unexpected reason: {capability}"
    assert capability["detail"], "capability carries no operator-facing detail sentence"


def test_1_capability_relayed_verbatim_by_the_sidecar(stack: QueueStack) -> None:
    """The sidecar's ``/bridge/health`` relays the bridge's record UNCHANGED.

    ``/bridge/health``, not ``/health`` -- the sidecar keeps the latter for its
    own container healthcheck. A deep-equality assert (not a spot-check on
    ``can_execute``) is what makes this a relay test: the sidecar must not
    reshape, summarize, or enrich the record.
    """
    bridge_status, bridge_body = _get("/health")
    relay_status, relay_body = _sidecar_get("/bridge/health")

    assert relay_status == 200, f"GET /bridge/health failed: {relay_status} {relay_body}"
    assert relay_body["capability"] == bridge_body["capability"], (
        "the sidecar reshaped the capability record instead of relaying it:\n"
        f"bridge ({bridge_status}): {bridge_body['capability']}\n"
        f"sidecar: {relay_body['capability']}"
    )
    assert relay_body["capability"]["reason"] == REASON_EXECUTABLE


def test_1_capability_never_leaks_the_control_socket_credential(stack: QueueStack) -> None:
    """Neither health surface may carry the queueserver public key.

    That key is the SOLE credential for the control socket (see the module
    docstring), so "public" is a misnomer: anything that reaches a panel or a
    log must not contain it.
    """
    public_key = _env_value(stack.project_dir, "BLUESKY_QSERVER_ZMQ_PUBLIC_KEY")
    for label, (_status, body) in (
        ("bridge /health", _get("/health")),
        ("sidecar /bridge/health", _sidecar_get("/bridge/health")),
        ("bridge /queue", _get("/queue")),
    ):
        assert public_key not in json.dumps(body), (
            f"{label} leaked the queueserver control-socket public key"
        )


# ===========================================================================
# Stage 2 -- enqueue two scans from two draft revisions
# ===========================================================================


def test_2_enqueue_two_revisions_and_refuse_a_replay(stack: QueueStack) -> None:
    """Two scans from two draft revisions; the SAME revision cannot enqueue twice.

    The draft revision is the unit of "this exact plan, as the operator saw
    it": ``POST /queue/items`` takes ``plan_name``/``plan_args`` from the
    server-side snapshot AT that revision, never from the request body. Pinning
    one revision twice would put the same plan on hardware twice from a single
    human confirmation, so the second attempt is refused
    ``draft_revision_already_launched``.

    Both enqueues here are UNARMED (no token) on purpose: the queue is idle, so
    an added item just sits there until an armed start -- that is the designed
    compose-now/arm-later flow, and stage 3 owns the arming half.
    """
    revision_a = _patch_draft("grid_scan", _grid_args(stack, LIVE_SAMPLE_POINTS))
    _S.run_live = _enqueue_ok(revision_a)

    replay_status, replay_body = _enqueue(revision_a)
    assert replay_status == 409, (
        f"a second enqueue of revision {revision_a} must be refused: {replay_status} {replay_body}"
    )
    assert _code_of(replay_body) == "draft_revision_already_launched", (
        f"wrong refusal code on a replayed revision: {replay_body}"
    )
    assert _detail_of(replay_body).get("revision") == revision_a, (
        f"the refusal does not name the replayed revision: {replay_body}"
    )

    # A second, DIFFERENT revision enqueues normally -- the guard is per
    # revision, not a one-item-per-draft rule.
    revision_b = _patch_draft("grid_scan", _grid_args(stack, SHORT_POINTS))
    assert revision_b != revision_a, "editing the draft did not bump its revision"
    _S.run_short = _enqueue_ok(revision_b)

    queue = _queue_snapshot()
    queued_names = [item.get("name") for item in queue["items"]]
    assert queued_names == ["grid_scan", "grid_scan"], (
        f"expected exactly the two enqueued grid scans, got: {queue['items']}"
    )
    assert queue["status"]["items_in_queue"] == 2, f"queue status disagrees: {queue['status']}"

    # Both runs are visible as `pending` before anything is armed, and carry
    # the params they were enqueued with (unwrapped PARAMS fields, no envelope).
    for run_id in (_S.run_live, _S.run_short):
        status, record = _run_record(str(run_id))
        assert status == 200, f"GET /runs/{run_id} failed: {status} {record}"
        assert record["status"] == "pending", f"unexpected pre-start status: {record}"
        assert record["plan_name"] == "grid_scan"
        assert "axes" in record["plan_args"], f"plan_args lost the PARAMS fields: {record}"
        assert "run_uid" not in record, f"a pending run cannot have a RunEngine uid yet: {record}"


# ===========================================================================
# Stage 3 -- token-gated start, serial drain, live rows
# ===========================================================================


def test_3_start_requires_the_launch_token(stack: QueueStack) -> None:
    """``POST /queue/start`` without the token is refused ``launch_token_required``.

    Starting the queue is THE arming action -- qserver autostart stays disabled
    so the bridge originates every start. 403 (a token is configured, this
    request did not carry it) and 503 (no token configured at all) are both
    correct refusals; the code is what a client branches on, so it is the code
    this asserts.
    """
    if _S.run_live is None:
        pytest.skip("stage 2 never enqueued anything to start")

    status, body = _post("/queue/start")
    assert status in (403, 503), f"an unarmed start must be refused: {status} {body}"
    assert _code_of(body) == "launch_token_required", (
        f"wrong refusal code on an unarmed start: {body}"
    )

    # Negative control: the refusal changed nothing.
    queue = _queue_snapshot()
    assert queue["status"]["manager_state"] != "executing_queue", (
        f"the queue started despite the refusal: {queue['status']}"
    )
    assert queue["status"]["items_in_queue"] == 2, (
        f"the refused start disturbed the queue: {queue['status']}"
    )


def test_3_armed_start_drains_serially_with_live_rows(stack: QueueStack) -> None:
    """The armed start drains the queue STRICTLY SERIALLY, publishing live rows.

    Three things are proved in one drain, because they are only observable
    while a plan is actually under way:

    * the token-carrying start is accepted;
    * at most ONE item is ever running -- the queue server executes one plan at
      a time and OSPREY never fans out;
    * the document plane delivers rows to the bridge DURING the run, at the
      ~1 s cadence the panel freshness target assumes. Sampled off both
      surfaces that expose it: ``GET /runs/{id}``'s ``progress.rows_seen`` and
      ``GET /runs/{id}/data``'s live path (``run_uid: null``, ``partial: true``).

    ``progress`` is asserted only where it is PRESENT: an absent progress
    record is the honest answer for a denominator the estimator cannot know,
    and this test must never push the code toward fabricating one.
    """
    if _S.run_live is None or _S.run_short is None:
        pytest.skip("stage 2 never enqueued anything to start")

    status, body = _post("/queue/start", token=stack.token)
    assert status == 200, f"armed POST /queue/start failed: {status} {body}"
    assert body["started"] is True, f"start did not report started: {body}"

    # rows_seen samples keyed PER RUN: the drain moves from one plan to the
    # next, and the next one restarts its count from zero, so a single flat
    # list would show a decrease that says nothing about liveness.
    samples: dict[str, list[int]] = {}
    running_seen: set[str] = set()
    partial_live_read = False
    deadline = time.monotonic() + DRAIN_TIMEOUT_SEC

    while time.monotonic() < deadline:
        queue = _queue_snapshot()
        running_item = queue["running_item"]

        # Serial drain: the manager exposes ONE running item, never a list.
        if isinstance(running_item, dict) and running_item:
            uid = running_item.get("item_uid")
            if isinstance(uid, str):
                running_seen.add(uid)
            progress = running_item.get("progress")
            if isinstance(progress, dict):
                rows_seen = progress.get("rows_seen")
                assert isinstance(rows_seen, int), f"progress.rows_seen not an int: {progress}"
                samples.setdefault(str(uid), []).append(rows_seen)

        data_status, data = _get(f"/runs/{_S.run_live}/data")
        if data_status == 200 and data.get("partial") is True:
            partial_live_read = True
            assert data["run_uid"] is None, (
                f"the LIVE path must report run_uid null (Tiled populates it): {data}"
            )

        records = _get("/runs")[1]
        running_records = [r for r in records if r.get("status") == "running"]
        assert len(running_records) <= 1, (
            f"more than one run reported running -- the drain is not serial: {running_records}"
        )

        if queue["status"]["items_in_queue"] == 0 and not running_item:
            break
        time.sleep(1.0)
    else:
        raise AssertionError(
            f"the queue did not drain within {DRAIN_TIMEOUT_SEC:.0f}s: {_queue_snapshot()}"
        )

    for run_id in (_S.run_live, _S.run_short):
        record = _wait_for_run_status(str(run_id), ("completed",), RUN_TIMEOUT_SEC)
        assert record["status"] == "completed", f"run {run_id} did not complete: {record}"

    # At least one item was OBSERVED as the running item. Deliberately not
    # "both": a short plan can start and finish entirely between two ~1 s
    # polls, and a sampling artifact must not be reported as a seriality
    # failure. The real seriality proof is the `<= 1 running` assertion above,
    # which is checked on EVERY poll and cannot be missed by being too fast.
    assert running_seen, (
        "never observed any item as THE running item across the whole drain -- "
        "the queue reported no running plan while it was draining"
    )
    assert partial_live_read, (
        "never observed a partial live-buffer read while a plan was running -- the "
        "document plane delivered no rows to the bridge during the run"
    )
    advanced = [
        series for series in samples.values() if len(series) >= 2 and series[-1] > series[0]
    ]
    assert advanced, (
        "no run's live row count was ever seen to ADVANCE across ~1 s polls -- rows "
        f"are not reaching the bridge while a plan runs: {samples}"
    )


# ===========================================================================
# Stage 4 -- results retrieval, live-buffer path
# ===========================================================================


def test_4_results_read_back_off_the_live_buffer(stack: QueueStack) -> None:
    """Both completed runs serve their rows off the LIVE buffer.

    The buffer outlives the run (see ``live_rows.py``'s retention bound), so a
    completed run still reads on the live path -- ``run_uid: null``, and no
    ``partial`` flag now that the stop document has landed. The DURABLE Tiled
    path for these same two runs is stage 7's subject: it only becomes
    observable once the bridge restart throws the buffer away.

    The rows are stashed on ``_S`` so stage 7 can compare content, not merely
    counts: an identical row_count with different values would pass a
    length-only check.
    """
    for run_id in (_S.run_live, _S.run_short):
        if run_id is None:
            pytest.skip("stage 2/3 never produced a completed run")

        status, data = _get(f"/runs/{run_id}/data?max_rows=1000")
        assert status == 200, f"GET /runs/{run_id}/data failed: {status} {data}"
        assert data["run_uid"] is None, (
            f"a live-buffer read must report run_uid null: {data['run_uid']!r}"
        )
        assert "partial" not in data, (
            f"a completed run's buffer must not still be marked partial: {data}"
        )
        assert data["row_count"] > 0, f"no rows recorded for {run_id}: {data}"
        assert len(data["rows"]) == data["row_count"], (
            f"row_count disagrees with len(rows) for {run_id}: {data}"
        )
        assert data["columns"], f"no columns recorded for {run_id}: {data}"

        _S.live_rows[str(run_id)] = data["rows"]
        _S.live_columns[str(run_id)] = data["columns"]

    # The two runs asked for different point counts, so their row counts must
    # differ -- a guard against both reads accidentally serving one run.
    assert len(_S.live_rows[str(_S.run_live)]) != len(_S.live_rows[str(_S.run_short)]), (
        "both runs returned the same number of rows; the two reads may be serving one buffer"
    )


def test_4_unknown_run_data_is_404_not_an_empty_scan(stack: QueueStack) -> None:
    """A run neither source knows 404s -- never a 200 with an empty table.

    A 200-empty answer would make a nonexistent run indistinguishable from a
    valid scan that recorded nothing, which is how "the data is gone" gets read
    as "the scan produced nothing".
    """
    status, body = _get("/runs/definitely-not-a-real-run-id/data")
    assert status == 404, f"expected 404 for an unknown run, got {status}: {body}"


# ===========================================================================
# Stage 5 -- session plans
# ===========================================================================

# An orm-shaped session plan body, mirroring plans_core/orm.py's house style.
# NO `from __future__ import annotations`: `POST /plans/session` writes
# `f"PLAN_METADATA = {...}\n\n{body}"`, so the generated metadata assignment
# always precedes this text, and Python requires a __future__ import to be the
# file's first statement -- an inherent consequence of the metadata-prepending
# design, not a bug (PEP 585 makes the hints work natively anyway). Same
# constraint the sandbox-escape e2e documents.
_SESSION_PLAN_BODY = '''"""Session-authored plan for the queue e2e -- sweeps one corrector while
reading BPMs, mirroring plans_core/orm.py's shape. Exists to prove the
author -> validate -> upload -> enqueue -> execute path end to end against a
real queueserver worker."""

import logging
from typing import Any

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class PARAMS(BaseModel):
    correctors: list[str] = Field(..., min_length=1)
    detectors: list[str] = Field(..., min_length=1)
    span_a: float = Field(..., gt=0, le=10.0)
    num: int = Field(..., ge=3)

    @model_validator(mode="after")
    def _disjoint(self) -> "PARAMS":
        overlap = set(self.correctors) & set(self.detectors)
        if overlap:
            raise ValueError(f"correctors and detectors must be disjoint (overlap: {sorted(overlap)})")
        return self


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    correctors = [(name, devices[name]) for name in params.correctors]
    corrector_devices = [corrector for _, corrector in correctors]
    detector_devices = [devices[name] for name in params.detectors]
    step = (2 * params.span_a) / (params.num - 1)
    currents = [-params.span_a + i * step for i in range(params.num)]
    all_devices = corrector_devices + detector_devices

    @bpp.stage_decorator(all_devices)
    @bpp.run_decorator()
    def _sweep():
        for name, corrector in correctors:
            try:
                for current in currents:
                    yield from bps.mv(corrector, current)
                    yield from bps.trigger_and_read(all_devices)
            finally:
                try:
                    yield from bps.mv(corrector, 0.0)
                except Exception:
                    logger.warning("failed to restore corrector %s to 0", name, exc_info=True)

    return _sweep()
'''

_SESSION_PLAN_NAME = "queue_e2e_session_sweep"


def _session_plan_args(stack: QueueStack) -> dict[str, Any]:
    """Args for the session plan: one real corrector, one real BPM, a tiny sweep.

    One place, used by BOTH the validation dry run and the enqueue, so the
    bytes that were validated are exercised with the parameters they will
    actually run with.
    """
    return {
        "correctors": [next(iter(stack.correctors))],
        "detectors": [next(iter(stack.bpms))],
        "span_a": 1.0,
        "num": 3,
    }


def _author_session_plan(name: str, body: str) -> str:
    status, response = _post(
        "/plans/session",
        {
            "name": name,
            "description": "queue e2e session-authored corrector sweep",
            "category": "accelerator",
            "required_devices": ["correctors", "detectors"],
            "writes": True,
            "body": body,
        },
    )
    assert status == 200, f"POST /plans/session failed: {status} {response}"
    return str(response["content_hash"])


def test_5_session_plan_authored_validated_uploaded_and_executed(stack: QueueStack) -> None:
    """author -> validate (PASS + uploaded) -> stage in the draft -> enqueue -> drain.

    A session plan is the only plan tier that must be uploaded into the RE
    worker's namespace before it can run, and the upload happens as a
    consequence of a PASSING validation -- ``upload.uploaded`` on the validate
    response is the proof it reached a real manager, not merely that the bytes
    hashed clean.
    """
    _author_session_plan(_SESSION_PLAN_NAME, _SESSION_PLAN_BODY)

    # `sample_args` is not optional in practice: validation's third stage DRY
    # RUNS the plan, which instantiates its PARAMS model, so a plan with
    # required fields fails validation outright without them. They are the
    # same args the enqueue below uses, deliberately -- validating one shape
    # and running another would leave the dry run proving nothing about the
    # plan that actually executes.
    status, result = _post(
        "/plans/validate",
        {"name": _SESSION_PLAN_NAME, "sample_args": _session_plan_args(stack)},
        timeout=180.0,
    )
    assert status == 200, f"POST /plans/validate failed: {status} {result}"
    assert result["passed"] is True, f"session plan failed validation: {result['reasons']}"
    assert result["upload"]["uploaded"] is True, (
        f"a passing validation must upload the plan into the worker namespace: {result['upload']}"
    )
    _S.session_plan_name = _SESSION_PLAN_NAME

    # It is now a real catalog entry -- which is also what lets the draft
    # resolve its schema below.
    plans = _get("/plans")[1]
    assert _SESSION_PLAN_NAME in {p["name"] for p in plans}, (
        f"a validated session plan must appear in GET /plans: {[p['name'] for p in plans]}"
    )

    revision = _patch_draft(_SESSION_PLAN_NAME, _session_plan_args(stack))
    snapshot = _draft_snapshot()
    assert snapshot["draft"]["plan_name"] == _SESSION_PLAN_NAME, (
        f"the draft did not take the session plan: {snapshot}"
    )

    _S.run_session = _enqueue_ok(revision)

    start_status, start_body = _post("/queue/start", token=stack.token)
    assert start_status == 200, f"start failed for the session plan: {start_status} {start_body}"

    record = _wait_for_run_status(
        str(_S.run_session), ("completed", "error", "stopped"), RUN_TIMEOUT_SEC
    )
    assert record["status"] == "completed", f"session plan did not complete: {record}"
    assert record["plan_name"] == _SESSION_PLAN_NAME

    data_status, data = _get(f"/runs/{_S.run_session}/data")
    assert data_status == 200, f"session run produced no readable data: {data_status} {data}"
    assert data["row_count"] > 0, f"session run recorded no rows: {data}"


def test_5_session_plan_with_stale_validation_is_refused_at_enqueue(stack: QueueStack) -> None:
    """Re-authoring a validated session plan makes it UNVALIDATED again -- and unqueueable.

    The enqueue gate re-reads and re-hashes the file every time, so overwriting
    the file's bytes retires its passing record without any explicit
    invalidation step. Reached through the realistic route: pin the draft while
    the plan is still valid, THEN re-author, then enqueue -- exactly the race
    the gate's freshness exists for.

    Asserted on ``detail.code``: the refusal used to carry a bare-string detail,
    which made this very code unreachable at enqueue.
    """
    if _S.session_plan_name is None:
        pytest.skip("stage 5's authoring step never produced a validated session plan")

    revision = _patch_draft(_SESSION_PLAN_NAME, {**_session_plan_args(stack), "span_a": 2.0})

    # Same name, different bytes -> the recorded hash no longer matches disk.
    _author_session_plan(
        _SESSION_PLAN_NAME,
        _SESSION_PLAN_BODY.replace(
            "span_a: float = Field(..., gt=0, le=10.0)", "span_a: float = Field(..., gt=0, le=9.0)"
        ),
    )

    status, body = _enqueue(revision, token=stack.token)
    assert status == 409, f"an unvalidated session plan must be refused: {status} {body}"
    assert _code_of(body) == REASON_UNVALIDATED, (
        f"wrong refusal code for a stale validation record: {body}"
    )

    # Nothing landed in the queue.
    queue = _queue_snapshot()
    assert queue["status"]["items_in_queue"] == 0, (
        f"the refused enqueue left an item behind: {queue}"
    )


# ===========================================================================
# Stage 6 -- emergency abort
# ===========================================================================


def test_6_abort_halts_a_running_plan_without_a_token(stack: QueueStack) -> None:
    """The emergency halt stops a plan ALREADY moving hardware, ungated.

    ``POST /queue/stop`` halts the queue only AFTER the running item finishes;
    this is the other one. It is deliberately ungated -- no launch token, no
    capability check, no body -- because a halt that can be refused for a policy
    reason is a halt with a failure mode.

    The aborted run's record reads ``stopped``, not a status of its own: the
    projection collapses aborted/halted/stopped into one operator word for "a
    human stopped it". And the manager must come back to a startable state
    afterwards -- an abort that left the stack wedged would be a worse outcome
    than the run it stopped.

    WHERE THE ITEM GOES, which is the half a mocked test cannot see. Upstream
    does not discard an interrupted plan: it records the run in history AND
    pushes a copy back to the FRONT of the queue with its ``result``. So the
    queue an operator is left holding has the plan they just emergency-stopped
    at its head, and a naive start would put it straight back on the hardware.
    This test drives that whole sequence against the real manager: abort, read
    the record, watch the next armed start REFUSE
    (``interrupted_item_in_queue``), drop the item, and only then start
    normally. The first live run of this file found that gap -- the record read
    ``pending`` and the start went through.
    """
    revision = _patch_draft("grid_scan", _grid_args(stack, LONG_POINTS))
    _S.run_aborted = _enqueue_ok(revision)

    start_status, start_body = _post("/queue/start", token=stack.token)
    assert start_status == 200, f"could not start the long scan: {start_status} {start_body}"

    _wait_for_run_status(str(_S.run_aborted), ("running",), 180.0, poll=0.5)

    # NO token, no body -- the route declares no token header at all.
    abort_status, abort_body = _post("/queue/abort", timeout=180.0)
    assert abort_status == 200, f"the ungated abort was refused: {abort_status} {abort_body}"
    assert abort_body["aborted"] is True, f"abort did not report aborted: {abort_body}"
    assert "abort_pending" in abort_body, f"abort response omits abort_pending: {abort_body}"
    assert isinstance(abort_body.get("msg"), str), (
        f"abort must relay the manager's own sentence: {abort_body}"
    )

    record = _wait_for_run_status(str(_S.run_aborted), ("stopped", "completed", "error"), 240.0)
    assert record["status"] == "stopped", (
        f"an aborted run must project as 'stopped' (aborted/halted/stopped collapse "
        f"to one operator word): {record}"
    )

    state = _wait_for_manager_state(("idle",), 240.0)
    assert state == "idle", f"the manager did not return to a startable state: {state!r}"

    # The manager really did requeue it -- everything below is about that item.
    queue = _queue_snapshot()
    requeued = [
        item
        for item in queue["items"]
        if isinstance(item.get("result"), dict) and item["result"].get("exit_status")
    ]
    assert len(requeued) == 1, (
        f"expected exactly the aborted plan back in the queue, got: {queue['items']}"
    )
    assert requeued[0]["result"]["exit_status"] == "aborted"

    # An armed start must NOT re-run it. This is the safety property: an
    # emergency halt that a later start silently undoes is not a halt.
    status, body = _post("/queue/start", token=stack.token)
    assert status == 409, f"an armed start re-ran the plan a human just aborted: {status} {body}"
    assert _code_of(body) == "interrupted_item_in_queue", (
        f"wrong refusal code for a queue holding an aborted plan: {body}"
    )
    assert _detail_of(body).get("exit_status") == "aborted"
    assert _wait_for_manager_state(("idle",), 60.0) == "idle", (
        "the refused start still moved the manager"
    )

    # The documented way out works, and the queue starts normally again --
    # otherwise the refusal above would be a dead end rather than a decision
    # point. Removing it is the operator's explicit choice; the run stays
    # visible in /runs either way, because it is in history too.
    stranded_uid = requeued[0]["item_uid"]
    removed_status, removed_body = _request(BRIDGE_URL, f"/queue/items/{stranded_uid}", "DELETE")
    assert removed_status == 200, (
        f"could not drop the aborted item: {removed_status} {removed_body}"
    )

    still_listed = _run_record(str(_S.run_aborted))[1]
    assert still_listed["status"] == "stopped", (
        f"dropping the queue copy must not erase the run -- history still has it: {still_listed}"
    )

    revision_after = _patch_draft("grid_scan", _grid_args(stack, SHORT_POINTS))
    run_after = _enqueue_ok(revision_after)
    status, body = _post("/queue/start", token=stack.token)
    assert status == 200, f"the manager is not startable after the abort: {status} {body}"
    after = _wait_for_run_status(str(run_after), ("completed", "error", "stopped"), RUN_TIMEOUT_SEC)
    assert after["status"] == "completed", f"the post-abort run did not complete: {after}"


def test_6_abort_with_nothing_running_is_nothing_running(stack: QueueStack) -> None:
    """An abort with no plan under way is an honest 409, not a fake success.

    ``nothing_running`` is the truth -- nothing was stopped because there was
    nothing to stop -- and it is a different answer from ``abort_pause_timeout``
    (the Run Engine never paused, so the plan MAY STILL BE RUNNING). Collapsing
    the two would be the dangerous direction.
    """
    _wait_for_manager_state(("idle",), 240.0)

    status, body = _post("/queue/abort", timeout=120.0)
    assert status == 409, f"an abort with nothing running must 409: {status} {body}"
    assert _code_of(body) == "nothing_running", f"wrong refusal code for an idle abort: {body}"


# ===========================================================================
# Stage 7 -- bridge restart: Redis durability + the Tiled read path
# ===========================================================================


def test_7_bridge_restart_preserves_queue_and_history(stack: QueueStack) -> None:
    """Restarting ONLY the bridge keeps the queue and the run history.

    The bridge holds no queue state -- it is a facade over the RE manager, whose
    queue and history live in Redis. So a bridge that comes back must still see
    every run it saw before, without replaying anything.

    Only the bridge container is restarted, by exact name: ``osprey deploy
    restart`` would bounce Redis and the manager too and defeat the proof.
    """
    known_runs = [r for r in (_S.run_live, _S.run_short, _S.run_session) if r]
    if not known_runs:
        pytest.skip("no completed runs from the earlier stages to look for")

    before = _get("/runs?limit=100")[1]
    before_ids = {r["id"] for r in before}
    for run_id in known_runs:
        assert run_id in before_ids, f"{run_id} missing from /runs BEFORE the restart: {before_ids}"

    restart = subprocess.run(
        ["docker", "restart", BRIDGE_CONTAINER],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert restart.returncode == 0, f"docker restart {BRIDGE_CONTAINER} failed: {restart.stderr}"

    _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
    _wait_for_container_health(BRIDGE_CONTAINER, CONTAINER_HEALTH_TIMEOUT_SEC)

    queue = _queue_snapshot()
    assert queue["status"]["available"] is True, (
        f"the restarted bridge cannot read the manager: {queue['status']}"
    )

    after = _get("/runs?limit=100")[1]
    after_ids = {r["id"] for r in after}
    for run_id in known_runs:
        assert run_id in after_ids, (
            f"{run_id} vanished from /runs across the bridge restart -- the history is "
            f"not surviving in Redis: {after_ids}"
        )
    assert {r["id"] for r in before} <= after_ids, (
        "the restarted bridge reports fewer runs than it did before"
    )


def test_7_completed_run_data_serves_from_tiled_after_the_restart(stack: QueueStack) -> None:
    """Post-restart, the same run's data comes off the DURABLE Tiled path.

    The live-row buffer is in-process and died with the bridge, so this read can
    only be answered by Tiled -- which is exactly what makes ``run_uid``
    populated here (Tiled has the stored start document; the live buffer never
    knew the uid the worker's RunEngine minted). Content, not just counts, is
    compared against stage 4's live-path read: an identical row_count with
    different values would pass a length-only check.
    """
    run_id = _S.run_live
    if run_id is None or str(run_id) not in _S.live_rows:
        pytest.skip("stage 4 never captured a live-path read to compare against")

    status, data = _get(f"/runs/{run_id}/data?max_rows=1000")
    assert status == 200, f"GET /runs/{run_id}/data after the restart failed: {status} {data}"
    assert data["run_uid"], (
        f"the Tiled path must populate run_uid (null means it served a live buffer "
        f"that should not exist after a restart): {data}"
    )
    assert data["columns"] == _S.live_columns[str(run_id)], (
        f"columns diverged between the live and Tiled paths:\n"
        f"live:  {_S.live_columns[str(run_id)]}\ntiled: {data['columns']}"
    )
    assert data["rows"] == _S.live_rows[str(run_id)], (
        "row CONTENT diverged between the live buffer and Tiled -- the durable copy "
        "is not the same scan"
    )


# ===========================================================================
# Stage 8 -- flip to mock: browse-only
# ===========================================================================


def test_8_mock_flip_makes_the_deployment_browse_only(stack: QueueStack) -> None:
    """``set-control-system mock`` + redeploy -> healthy, but browse-only.

    Three claims, and the first is the one people get wrong: a browse-only
    deployment is a HEALTHY deployment. ``/health`` still answers 200 and every
    container still reports healthy; what changes is the capability record
    (``can_execute: false``, ``browse_only_connector``) and the fact that
    enqueue now refuses -- a deployment that cannot execute must never HOLD
    queue items, because an item sitting in a queue reads as work that will
    happen.

    Deliberately last among the functional stages: it leaves the deployment on
    the mock connector, and stage 9's probes do not care which connector is
    configured.
    """
    flip = _run(
        [str(stack.osprey_bin), "config", "set-control-system", "mock"],
        cwd=stack.project_dir,
        timeout=180,
    )
    assert flip.returncode == 0, f"set-control-system mock failed: {flip.stdout}\n{flip.stderr}"

    up = _run(
        [str(stack.osprey_bin), "deploy", "up", "-d", "--dev"],
        cwd=stack.project_dir,
        timeout=DEPLOY_UP_TIMEOUT_SEC,
    )
    assert up.returncode == 0, f"redeploy after the mock flip failed: {up.stdout}\n{up.stderr}"

    _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
    _wait_for_health(f"{PANELS_URL}/health", HEALTH_TIMEOUT_SEC)
    for container in _STACK_CONTAINERS:
        _wait_for_container_health(container, CONTAINER_HEALTH_TIMEOUT_SEC)

    status, body = _get("/health")
    assert status == 200, f"a browse-only bridge must still answer 200: {status} {body}"
    assert body["status"] == "ok", f"liveness must not track capability: {body}"

    capability = body["capability"]
    assert capability["can_execute"] is False, f"mock must not be executable: {capability}"
    assert capability["reason"] == REASON_BROWSE_ONLY_CONNECTOR, (
        f"wrong capability reason on the mock connector: {capability}"
    )
    assert "set-control-system virtual_accelerator" in capability["detail"], (
        f"the browse-only detail must name the command that flips it: {capability}"
    )


def test_8_browse_only_deployment_refuses_to_hold_queue_items(stack: QueueStack) -> None:
    """Enqueue is refused on a browse-only deployment, capability record attached."""
    plans = _get("/plans")[1]
    assert any(p["name"] == "grid_scan" for p in plans), (
        "the shipped catalog must still be BROWSABLE on a browse-only deployment "
        f"-- that is the whole point of the mode: {[p['name'] for p in plans]}"
    )

    # Measured as a DELTA, not against zero. The queue can legitimately carry
    # items from before the flip -- stage 6's abort deliberately leaves one
    # there (upstream requeues an interrupted plan; see `runs._queue_status`) --
    # and "holds nothing at all" would fail on that leftover while saying
    # nothing about the refusal under test. The property that matters is that
    # the browse-only deployment ACCEPTED nothing.
    before = _queue_snapshot()["status"]["items_in_queue"]

    revision = _patch_draft("grid_scan", _grid_args(stack, SHORT_POINTS))
    status, body = _enqueue(revision, token=stack.token)

    assert status == 409, f"a browse-only enqueue must be refused: {status} {body}"
    assert _code_of(body) == REASON_BROWSE_ONLY_CONNECTOR, (
        f"wrong refusal code on a browse-only enqueue: {body}"
    )
    capability = _detail_of(body).get("capability")
    assert isinstance(capability, dict) and capability.get("can_execute") is False, (
        f"the refusal must carry the capability record the status surface publishes: {body}"
    )

    queue = _queue_snapshot()
    assert queue["status"]["items_in_queue"] == before, (
        f"the refused enqueue added an item to a browse-only deployment's queue "
        f"({before} -> {queue['status']['items_in_queue']}): {queue}"
    )
    assert not any(
        item.get("kwargs", {}).get("axes", [{}])[0].get("num_points") == SHORT_POINTS
        for item in queue["items"]
        if isinstance(item, dict) and item.get("kwargs", {}).get("axes")
    ), f"the refused plan is sitting in the queue: {queue['items']}"


# ===========================================================================
# Stage 9 -- security probes
# ===========================================================================


def _project_network(name: str) -> str:
    """The full docker network name for this project's ``osprey-network``.

    Compose prefixes networks with the project name, and the exact prefix
    depends on how ``deploy up`` invoked compose -- so read it off a container
    that is only ever attached to that one network rather than guessing.
    """
    raw = _docker_inspect(name, "{{json .NetworkSettings.Networks}}")
    networks = list(json.loads(raw).keys())
    assert networks, f"{name} is attached to no network at all"
    return networks[0]


def test_9_security_redis_is_unreachable_from_osprey_network(stack: QueueStack) -> None:
    """Redis answers only on ``bluesky-internal``, never from ``osprey-network``.

    Redis holds the queue and history and has no authentication of its own; the
    only thing keeping it private is that it is attached to an ``internal:
    true`` network with the RE manager as its sole client. Probed from a
    throwaway container on ``osprey-network`` -- where the panels sidecar, Tiled
    and the VA live -- using this project's own bridge image (no pull, nothing
    new on the host).

    The positive control matters as much as the probe: the same connect from
    ``bluesky-internal`` MUST succeed, or a DNS-less "connection failed" would
    prove nothing about network isolation.
    """
    # Tiled sits on osprey-network only, so its single network is the one to probe from.
    osprey_network = _project_network(TILED_CONTAINER)
    internal_networks = json.loads(
        _docker_inspect(REDIS_CONTAINER, "{{json .NetworkSettings.Networks}}")
    )
    assert osprey_network not in internal_networks, (
        f"Redis is attached to {osprey_network} -- it must be on the internal network only: "
        f"{list(internal_networks)}"
    )
    redis_network = next(iter(internal_networks))

    probe = (
        "import socket,sys\n"
        "try:\n"
        "    socket.create_connection(('bluesky-redis', 6379), timeout=5).close()\n"
        "    print('CONNECTED')\n"
        "except Exception as exc:\n"
        "    print('REFUSED', type(exc).__name__, exc)\n"
    )

    outside = subprocess.run(
        ["docker", "run", "--rm", "--network", osprey_network, BRIDGE_IMAGE, "python", "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert "CONNECTED" not in outside.stdout, (
        f"Redis is reachable from {osprey_network}: {outside.stdout}\n{outside.stderr}"
    )

    inside = subprocess.run(
        ["docker", "run", "--rm", "--network", redis_network, BRIDGE_IMAGE, "python", "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert "CONNECTED" in inside.stdout, (
        "the positive control failed: Redis is not reachable even from "
        f"{redis_network}, so the negative probe above proves nothing.\n"
        f"{inside.stdout}\n{inside.stderr}"
    )


def test_9_security_control_socket_refuses_a_client_without_the_public_key(
    stack: QueueStack,
) -> None:
    """The queueserver CONTROL socket answers no one without the server public key.

    The manager runs ``CURVE_SERVER=1`` with NO authenticator and NO client
    allowlist, and ``bluesky-queueserver-api``'s client authenticates with a
    fixed keypair shipped in the package -- so the server PUBLIC key is the
    SOLE credential. A client without it never completes the CURVE handshake
    and gets no answer at all, which is what this probes: ``qserver ping``
    (the container's own healthcheck command) with the key stripped from the
    environment must NOT succeed.

    Run inside the queueserver container against its own loopback -- the
    strongest position an attacker could reach, and still not enough without
    the key. The unstripped ``qserver ping`` is the positive control; without
    it a broken CLI would look like a security guarantee.
    """
    armed = subprocess.run(
        ["docker", "exec", QUEUESERVER_CONTAINER, "sh", "-c", "timeout 30 qserver ping"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert armed.returncode == 0, (
        "the positive control failed: `qserver ping` WITH the public key did not "
        f"succeed, so the negative probe below proves nothing.\n{armed.stdout}\n{armed.stderr}"
    )

    bare = subprocess.run(
        [
            "docker",
            "exec",
            "--env",
            "QSERVER_ZMQ_PUBLIC_KEY=",
            QUEUESERVER_CONTAINER,
            "sh",
            "-c",
            "timeout 30 qserver ping",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert bare.returncode != 0, (
        "an unauthenticated 0MQ client reached the control socket -- the server "
        "public key is not actually gating it.\n"
        f"stdout: {bare.stdout}\nstderr: {bare.stderr}"
    )


def test_9_security_document_plane_rejects_an_uncertified_publisher(stack: QueueStack) -> None:
    """A publisher with no CURVE client certificate lands no documents.

    The bridge's 0MQ Proxy binds with ``ServerCurve`` and a PINNED directory of
    accepted client public keys -- never ``CURVE_ALLOW_ANY`` -- which is what
    stops a container on ``osprey-network`` from injecting forged run documents
    that would show up as a scan that never happened. Probed by publishing a
    start/stop pair carrying a fabricated ``osprey_run_id`` from an
    unencrypted publisher and asserting the bridge never buffers it.

    The positive control is what makes this more than "publishing to a socket is
    hard": the SAME documents, published from the SAME container with the
    queueserver's own client certificate, MUST land. Without it, a typo in the
    address or a Publisher that silently no-ops would read as a security
    guarantee.

    Note the pyzmq contract this rests on: the authenticator reads the client-
    certificate directory ONCE, at socket configure. Certificates written to a
    RUNNING bridge have no effect until it restarts -- which is why nothing here
    tries to add one, and why the positive control uses the certificate the
    deploy already installed.
    """
    forged_id = "forged-run-id-queue-e2e"
    certified_id = "certified-run-id-queue-e2e"

    def _publish(run_id: str, *, with_cert: bool) -> subprocess.CompletedProcess:
        # ClientCurve wraps the socket with the publisher's own secret
        # certificate plus the proxy's public one -- the two files the compose
        # template mounts at /app/curve in THIS container.
        curve = (
            "from bluesky.callbacks.zmq import ClientCurve\n"
            "curve_config = ClientCurve(\n"
            "    secret_path='/app/curve/publisher.key_secret',\n"
            "    server_public_key='/app/curve/proxy.key')\n"
            if with_cert
            else "curve_config = None\n"
        )
        # A COMPLETE run -- start, descriptor, one event, stop. A start alone
        # is not enough to be observable: the bridge's live-row recorder needs
        # an event before `GET /runs/{id}/data` has anything to serve, so a
        # partial sequence would make the positive control fail for a reason
        # that has nothing to do with certificates.
        script = (
            "from bluesky.callbacks.zmq import Publisher\n"
            "import time, uuid\n"
            f"{curve}"
            "p = Publisher('bluesky-bridge:5567', curve_config=curve_config)\n"
            # 0MQ SLOW JOINER: a PUB socket silently DROPS everything sent
            # before its connection to the proxy has finished handshaking, so a
            # publisher that constructs and immediately sends reaches nobody --
            # and the certified control then 'fails' for a reason that has
            # nothing to do with certificates. The real queueserver publisher
            # never meets this because it stays connected for the life of the
            # worker. Settle first.
            "time.sleep(3)\n"
            "run_uid = str(uuid.uuid4())\n"
            "desc_uid = str(uuid.uuid4())\n"
            f"p('start', {{'uid': run_uid, 'time': time.time(), 'scan_id': 1,\n"
            f"             'osprey_run_id': {run_id!r}}})\n"
            "p('descriptor', {'uid': desc_uid, 'run_start': run_uid,\n"
            "                 'time': time.time(), 'name': 'primary',\n"
            "                 'data_keys': {'probe_signal': {'dtype': 'number',\n"
            "                                                'shape': [],\n"
            "                                                'source': 'probe'}}})\n"
            "p('event', {'uid': str(uuid.uuid4()), 'descriptor': desc_uid, 'seq_num': 1,\n"
            "            'time': time.time(), 'data': {'probe_signal': 1.0},\n"
            "            'timestamps': {'probe_signal': time.time()}})\n"
            "p('stop', {'uid': str(uuid.uuid4()), 'run_start': run_uid, 'time': time.time(),\n"
            "           'exit_status': 'success', 'reason': '',\n"
            "           'num_events': {'primary': 1}})\n"
            "time.sleep(2)\n"
            "p.close()\n"
            "print('PUBLISHED')\n"
        )
        return subprocess.run(
            ["docker", "exec", QUEUESERVER_CONTAINER, "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )

    # --- the probe: no certificate at all ---------------------------------
    proc = _publish(forged_id, with_cert=False)
    assert "PUBLISHED" in proc.stdout, (
        "the forged-publisher probe never got as far as publishing, so it proves "
        f"nothing about the document plane:\n{proc.stdout}\n{proc.stderr}"
    )

    # --- the positive control: the deploy's own client certificate --------
    control = _publish(certified_id, with_cert=True)
    assert "PUBLISHED" in control.stdout, (
        f"the certified positive control failed to publish:\n{control.stdout}\n{control.stderr}"
    )

    # Wait for the CERTIFIED run to appear, bounded — the plane is a proxy plus
    # a polling dispatcher, so arrival is not instantaneous, but a plane that is
    # genuinely broken must make this test FAIL rather than hang.
    control_status, control_body = 0, None
    deadline = time.monotonic() + DOC_PLANE_ARRIVAL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        control_status, control_body = _get(f"/runs/{certified_id}/data")
        if control_status == 200:
            break
        time.sleep(1.0)
    assert control_status == 200, (
        "the positive control never reached the bridge within "
        f"{DOC_PLANE_ARRIVAL_TIMEOUT_SEC:.0f}s, so the refusal below proves nothing about "
        f"certificates: {control_status} {control_body}"
    )

    # Only now is the forged read meaningful: the certified run arriving proves
    # the plane is delivering, and gives the forged one every chance to have
    # arrived too if the socket had accepted it.

    status, body = _get(f"/runs/{forged_id}/data")
    assert status == 404, (
        f"the bridge accepted documents from an uncertified publisher: {status} {body}"
    )
