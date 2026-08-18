"""Real-container e2e for the layered plan catalog (task 1.6, closing
Phase 1 of the plan-catalog epic).

Mocked-client tests (``tests/services/bluesky_bridge/test_plan_loader_layered.py``,
``test_exemplar_plans.py``) only exercise OSPREY's half of the contract: that
the loader in this repo resolves layers and trust tiers correctly in-process.
They never prove that a *deployed* bridge container -- built from the shipped
image, reading its own filesystem layers -- actually serves the same catalog
over HTTP. This is the other half: it deploys a real bluesky-bridge container
and asserts the layered catalog (the shipped plans + an externally-injected
facility plan) is discoverable via ``GET /plans`` with correct
provenance/metadata, and that the browse-only surface around it is honest.

SCOPE, deliberately narrowed: this file proves DISCOVERY, not execution.
Execution has exactly one owner now -- ``tests/e2e/test_bluesky_queue_e2e.py``,
which deploys the whole queue stack (queueserver + Redis + Tiled + the Virtual
Accelerator) and drives real plan runs through arming, drain, abort and restart.
A facility-injected plan is a catalog entry like any other by the time it
reaches the queue, so re-proving execution here would mean standing up a second
VA-backed stack to re-test what that file already covers, at real wall-clock
cost. What this file uniquely proves is that a plan file dropped into a
facility layer is SERVED by a deployed container with the right provenance --
and that the deployment says plainly what it can and cannot do with it.

Uses the ``hello-world`` preset, whose ``control_system.type`` is ``mock``,
rather than the VA-backed stack ``tests/e2e/_orm_stack.py`` builds: the catalog
is connector-independent, so this skips the VA image's slow build entirely
(mirrors ``test_bluesky_deploy.py``'s identical rationale). A mock deployment
is BROWSE-ONLY, which is not a limitation here but part of the subject: plans
are discoverable and composable, the capability record says
``browse_only_connector`` and names the command that flips it, and the queue
refuses to hold work it could never run.

Container safety: every docker invocation below names an exact
container/image -- never a wildcard, never ``system prune``/``--volumes``.
Teardown goes through ``osprey down``, matching every other e2e in
this directory, followed by exact-named removal of this project's own volumes
(``tests/e2e/_volumes.py``): ``down`` keeps them by design, and a rerun must
not inherit their state.

Gating: needs Docker. Much lighter than the VA-backed e2e (no amd64
emulation) -- comparable to ``test_bluesky_deploy.py``'s build+deploy time.
Advisory CI lane (see ci.yml's ``bluesky-catalog-e2e`` job); run locally with
``E2E_REUSE_IMAGES=1`` set for fast iteration once the image cache is warm.
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

from osprey.services.bluesky_bridge.app import (
    PREVIEW_REASON_CATALOG_UNAVAILABLE,
    PREVIEW_REASON_FAILED,
    PREVIEW_REASON_PLAN_ERROR,
    PREVIEW_REASON_QUEUE_UNREACHABLE,
    PREVIEW_REASON_REFUSED,
    PREVIEW_REASON_TIMED_OUT,
    PREVIEW_REASON_UNKNOWN_PLAN,
)
from osprey.services.bluesky_bridge.plan_fields import (
    CHANNEL_ROLE_KEY,
    MOVABLE_ROLE,
    READABLE_ROLE,
)
from osprey.services.bluesky_bridge.queue_backend import (
    FLIP_COMMAND,
    REASON_BROWSE_ONLY_CONNECTOR,
)
from tests.e2e import _orm_stack
from tests.e2e._volumes import remove_project_volumes

# Every word the pre-flight is allowed to answer with. Imported rather than
# spelled, because the approval prompt branches on these exact literals.
_PREVIEW_REASONS = frozenset(
    {
        PREVIEW_REASON_UNKNOWN_PLAN,
        PREVIEW_REASON_CATALOG_UNAVAILABLE,
        PREVIEW_REASON_QUEUE_UNREACHABLE,
        PREVIEW_REASON_REFUSED,
        PREVIEW_REASON_TIMED_OUT,
        PREVIEW_REASON_FAILED,
        PREVIEW_REASON_PLAN_ERROR,
    }
)

# The nine keys every pre-flight answer carries, success or not.
_PREVIEW_KEYS = {
    "ok",
    "plan",
    "channels",
    "moves",
    "total_moves",
    "truncated",
    "move_cap",
    "reason",
    "detail",
}

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

# Distinct from the sibling e2e modules' pinned ports (_orm_stack.py's 18102,
# test_bluesky_deploy.py's 18090, test_va_substrate_equivalence.py's 18099,
# test_tiled_roundtrip.py's 18101) so all five can run concurrently on a
# shared dev machine without a port collision.
BRIDGE_PORT = 18103
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"

# OpenObserve ships with every preset and has no profile knob, so the deploy
# would otherwise publish 5080 -- which a locally running tutorial stack holds.
# `osprey up`'s host port preflight aborts the WHOLE deploy on one such clash,
# so leaving it on the default takes every test in this module down at fixture
# setup. Moved to default + 20000, and one past the queue e2e's 25080 so the
# two modules can deploy side by side.
OPENOBSERVE_PORT = 25081

# The repo directory name IS the deployment name, and the bridge compose
# template renders its locally-built image as ``<project>-bluesky-bridge:local``
# -- so one constant feeds both the `osprey init` path and the image tag.
PROJECT_NAME = "proj"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
DEPLOY_UP_TIMEOUT_SEC = 600
HEALTH_TIMEOUT_SEC = 120.0

# An ordinary, well-formed facility-tier plan. Its job here is to be FOUND --
# correct provenance, correct metadata, correct schema over HTTP -- so nothing
# below resolves its device names against a live worker.
_FACILITY_PLAN_SOURCE = '''"""Test-authored facility-tier plan for the layered plan catalog e2e
(tests/e2e/test_bluesky_catalog_e2e.py).

Not part of the shipped OSPREY package: this file is written to a throwaway
host directory and injected via `services.bluesky.plan_dir`
(BLUESKY_PLAN_DIRS), so the deployed bridge discovers it as a `facility`-tier
layer (plan_loader.py). Named `facility_probe` -- distinct from every
shipped/built-in plan name, so it never collides at the `GET /plans` merge.

Its parameter names are ordinary device-name strings; nothing here resolves
them against a live device, because this file proves DISCOVERY only (see the
module docstring). Execution of catalog plans -- facility-tier included -- is
`tests/e2e/test_bluesky_queue_e2e.py`'s subject.
"""

from __future__ import annotations

from typing import Any

from bluesky import plans as bp
from pydantic import BaseModel, Field, model_validator

from osprey.services.bluesky_bridge.plan_fields import MovableChannel, ReadableChannel

PLAN_METADATA = {
    "name": "facility_probe",
    "description": "Probe scan: sweep one setpoint device, reading one detector at each point.",
    "writes": True,
}


class PARAMS(BaseModel):
    """Parameters for `facility_probe`: one setpoint swept over [start, stop]."""

    motor: MovableChannel = Field(..., description="Setpoint device name to sweep.")
    detector: ReadableChannel = Field(..., description="Detector device name to read at each point.")
    start: float
    stop: float
    num: int = Field(..., ge=2, description="Number of evenly-spaced points.")

    @model_validator(mode="after")
    def _motor_and_detector_disjoint(self) -> "PARAMS":
        if self.motor == self.detector:
            raise ValueError(f"motor and detector must be distinct (got {self.motor!r} twice)")
        return self


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    """Wrap `bluesky.plans.scan`: move `motor` over `[start, stop]` in `num`
    steps, reading `detector` at each point."""
    motor = devices[params.motor]
    detector = devices[params.detector]
    return bp.scan([detector], motor, params.start, params.stop, num=params.num)
'''


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


def _request(path: str, method: str, body: dict | None = None) -> tuple[int, Any]:
    """One request against the bridge, returning ``(status, parsed_body)``.

    Refusal bodies are what this module asserts on, so an ``HTTPError`` is a
    normal result here rather than an exception to propagate.
    """
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(  # noqa: S310
        f"{BRIDGE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post_raw(path: str, data: bytes, content_type: str) -> tuple[int, Any]:
    """POST a body the JSON helpers cannot express (malformed, or not JSON at all).

    The pre-flight route promises one answer shape for EVERY request, including
    ones no well-behaved client would send, so a test of that promise has to be
    able to send them.
    """
    req = urllib.request.Request(  # noqa: S310
        f"{BRIDGE_URL}{path}", data=data, method="POST", headers={"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _assert_preview_shape(payload: Any, *, plan: str) -> None:
    """Every pre-flight answer, whatever happened, in one shape.

    The launch-approval gate reads ``ok`` and never a status code, so the keys
    have to be there on the failure paths too -- that is the property this
    helper exists to assert once and reuse.
    """
    assert isinstance(payload, dict), f"pre-flight answered with a non-object: {payload!r}"
    assert set(payload) == _PREVIEW_KEYS, f"pre-flight key set drifted: {sorted(payload)}"
    assert payload["plan"] == plan, f"pre-flight named a different plan: {payload}"
    assert isinstance(payload["ok"], bool)
    assert isinstance(payload["channels"], list)
    if payload["ok"]:
        assert payload["reason"] is None and payload["detail"] is None, (
            f"a successful pre-flight carries no failure words: {payload}"
        )
    else:
        assert payload["reason"] in _PREVIEW_REASONS, f"unknown pre-flight reason: {payload}"
        assert isinstance(payload["detail"], str) and payload["detail"], (
            f"a failed pre-flight must say why in a sentence: {payload}"
        )
        assert len(payload["detail"]) <= 2000, (
            "the detail lands verbatim in a human's approval prompt and must stay bounded"
        )
        assert payload["moves"] == [] and payload["total_moves"] == 0
        assert payload["truncated"] is False and payload["move_cap"] is None


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


@pytest.fixture(scope="module")
def deployed_catalog_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Init + build + ``osprey up --dev`` a bluesky-bridge repo with one
    facility-injected plan file; tear down after.

    ``hello-world`` (mirrors ``test_bluesky_deploy.py``): no VA co-deploy, no
    LLM secret needed, no amd64-emulated image build. The preset's
    ``control_system.type`` is ``mock``, so this deployment is browse-only --
    which is part of what the tests below assert, not a gap in them. ``bluesky.plan_dir`` points at a throwaway
    host directory containing ``_FACILITY_PLAN_SOURCE`` -- the deploy wiring
    (Task 1.4) bind-mounts it read-only and sets ``BLUESKY_PLAN_DIRS``, so
    ``plan_loader.py`` scans it as a ``facility``-tier layer.
    """
    osprey_bin = _orm_stack.find_osprey_console_script()
    base = tmp_path_factory.mktemp("plan_catalog_build")
    plan_dir = tmp_path_factory.mktemp("plan_catalog_plans")
    (plan_dir / "facility_probe.py").write_text(_FACILITY_PLAN_SOURCE, encoding="utf-8")
    # The deployment repo. Its directory name IS the deployment name, so the
    # image tag derived below (``proj-bluesky-bridge:local``) still holds.
    repo = base / PROJECT_NAME

    # Host hygiene only. Written as a flat dotted-string key under `config:`
    # (the preset's own convention) and passed as an override file rather than
    # a `--set`: a `--set` would build a NESTED dict for every dotted segment
    # and replace the whole `services:` block.
    override_path = base / "override.yml"
    override_path.write_text(
        f"config:\n  services.openobserve.port: {OPENOBSERVE_PORT}\n", encoding="utf-8"
    )

    # Two steps, because the surface has two: `init` writes the repo's source
    # zone from the preset plus these overrides, `build` renders build/ from it.
    init = _run(
        [
            str(osprey_bin),
            "init",
            str(repo),
            "--preset",
            "hello-world",
            "--no-git",
            "--override",
            str(override_path),
            "--set",
            f"bluesky.port={BRIDGE_PORT}",
            "--set",
            f"bluesky.plan_dir={plan_dir}",
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
        [str(osprey_bin), "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle", "--dev"],
        cwd=base,
        timeout=BUILD_TIMEOUT_SEC,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    # The repo root's .env is the deployment's whole secret store and the file
    # every compose invocation is pointed at, so `up` refuses to start without
    # one. `osprey init` writes it only when the shell exports a key for this
    # profile's provider, which this browse-only lane has no need of — this is
    # the `cp .env.example .env` the CLI itself recommends, done for the
    # operator.
    env_path = repo / ".env"
    if not env_path.exists():
        shutil.copy(repo / ".env.example", env_path)

    # Force a fresh --dev build so the deployed bridge runs CURRENT source
    # (osprey up does not pass --build to compose, so it would otherwise
    # reuse a stale cached image). Exact-named image only.
    # E2E_REUSE_IMAGES=1 skips this for fast local iteration once the image
    # cache is warm; never set it in CI.
    if not os.environ.get("E2E_REUSE_IMAGES"):
        subprocess.run(
            ["docker", "rmi", "-f", _orm_stack.bridge_image(PROJECT_NAME)],
            capture_output=True,
            text=True,
        )

    try:
        up = _run([str(osprey_bin), "up", "-d", "--dev"], cwd=repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        if up.returncode != 0:
            pytest.fail(
                f"osprey up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        yield repo
    finally:
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        # `osprey down` keeps volumes by design; drop this project's own so a
        # rerun cannot inherit their state (see tests/e2e/_volumes.py).
        remove_project_volumes(_orm_stack.project_prefix(PROJECT_NAME))


# ---------------------------------------------------------------------------
# Discovery: the layered catalog's provenance/metadata, over the real
# deployed HTTP API. Strict -- no @flaky -- since this is the core deliverable.
# ---------------------------------------------------------------------------


def test_plans_endpoint_shows_shipped_and_facility_provenance(
    deployed_catalog_stack: Path,
) -> None:
    """``GET /plans`` against the real container must show, in one response:

    - the shipped plans (``orm``, ``grid_scan``) with ``provenance ==
      "shipped"`` and non-null ``metadata`` (Task 1.5's in-image
      ``plans_core/`` files);
    - the externally-injected ``facility_probe`` plan with
      ``provenance == "facility"`` and its authored metadata round-tripped
      byte-for-byte through the loader's ``PLAN_METADATA`` parser.

    Every entry's metadata is asserted to be EXACTLY the three declared fields.
    That is the contract a consumer reads: ``category`` and ``required_devices``
    are retired, and a deployed container still publishing either would have a
    panel or an approval prompt telling an operator about a declaration nothing
    behind it honours any more.
    """
    status, plans = _get("/plans")
    assert status == 200, f"GET /plans failed: {status} {plans}"
    by_name = {p["name"]: p for p in plans}

    for shipped_name in ("orm", "grid_scan"):
        assert shipped_name in by_name, (
            f"{shipped_name!r} missing from GET /plans: {sorted(by_name)}"
        )
        entry = by_name[shipped_name]
        assert entry["provenance"] == "shipped", (
            f"{shipped_name!r}: expected provenance 'shipped', got {entry['provenance']!r}"
        )
        assert entry["metadata"] is not None, f"{shipped_name!r}: metadata is None"
        assert set(entry["metadata"]) == {"name", "description", "writes"}, (
            f"{shipped_name!r}: a deployed container is still publishing retired metadata "
            f"fields: {entry['metadata']}"
        )

    assert "facility_probe" in by_name, f"facility_probe missing from GET /plans: {sorted(by_name)}"
    facility_entry = by_name["facility_probe"]
    assert facility_entry["provenance"] == "facility", (
        "facility_probe: expected provenance 'facility' (injected via "
        f"services.bluesky.plan_dir/BLUESKY_PLAN_DIRS), got {facility_entry['provenance']!r}"
    )
    metadata = facility_entry["metadata"]
    assert metadata is not None, "facility_probe: metadata is None"
    assert metadata["name"] == "facility_probe"
    assert metadata["writes"] is True
    assert set(metadata) == {"name", "description", "writes"}


def test_the_served_schemas_carry_the_declared_channel_roles(
    deployed_catalog_stack: Path,
) -> None:
    """A deployed container publishes each plan's channel ROLES in its schema.

    The role declaration is what replaced guessing a channel's purpose from its
    parameter name, so it has to survive the trip a consumer actually makes:
    plan file -> loader -> ``PARAMS.model_json_schema()`` -> HTTP. Asserted on
    the exact JSON path, because a role that drifted into ``items`` would be
    invisible to every consumer while a presence-only check still passed.

    All three tiers are covered in one response, which is the point: the shipped
    plans, and the facility-injected file this module wrote itself -- whose
    single-channel fields (``MovableChannel``, not ``...Channels``) are the
    other annotation shape.
    """
    status, plans = _get("/plans")
    assert status == 200, f"GET /plans failed: {status} {plans}"
    by_name = {p["name"]: p for p in plans}

    orm = by_name["orm"]["schema"]["properties"]
    assert orm["correctors"][CHANNEL_ROLE_KEY] == MOVABLE_ROLE, (
        f"orm no longer declares its correctors movable over the wire: {orm['correctors']}"
    )
    assert orm["readbacks"][CHANNEL_ROLE_KEY] == READABLE_ROLE

    # grid_scan's movable is a field of the nested GridAxis model, which
    # pydantic emits under $defs and references from `axes.items`.
    grid = by_name["grid_scan"]["schema"]
    assert grid["properties"]["readbacks"][CHANNEL_ROLE_KEY] == READABLE_ROLE
    assert grid["$defs"]["GridAxis"]["properties"]["setpoint"][CHANNEL_ROLE_KEY] == MOVABLE_ROLE, (
        f"grid_scan's per-axis setpoint lost its movable role: {grid['$defs']['GridAxis']}"
    )

    facility = by_name["facility_probe"]["schema"]["properties"]
    assert facility["motor"][CHANNEL_ROLE_KEY] == MOVABLE_ROLE
    assert facility["detector"][CHANNEL_ROLE_KEY] == READABLE_ROLE
    # Negative control: an ordinary scalar parameter is not a channel.
    assert CHANNEL_ROLE_KEY not in facility["num"], (
        f"a plain number was annotated as a channel: {facility['num']}"
    )


# ---------------------------------------------------------------------------
# The pre-flight, on a deployment that cannot execute.
#
# The launch-approval gate reads this route before a human decides, so its
# ONLY promise is that it always answers in one shape. That promise is most
# load-bearing exactly here -- where no trajectory can be produced -- which is
# why a browse-only deployment is the honest place to prove it.
# ---------------------------------------------------------------------------


def test_preview_answers_in_one_shape_for_a_known_plan(deployed_catalog_stack: Path) -> None:
    """A known plan's pre-flight always answers 200 and always names its channels.

    Whether a trajectory comes back depends on a worker this deployment may not
    have; what does NOT depend on that is the shape, and the channel list -- the
    bridge's own reading of what the launch DECLARES it would move, which stays
    true and worth showing to an approver even when the moves behind it are
    unavailable.

    Both nesting shapes are covered: ``facility_probe``'s single-channel fields
    and ``grid_scan``'s movable buried in a list of ``GridAxis`` objects, which
    only a walk of the declared roles finds.
    """
    status, payload = _request(
        "/plans/facility_probe/preview",
        "POST",
        {"motor": "motor1", "detector": "det1", "start": 0.0, "stop": 2.0, "num": 3},
    )
    assert status == 200, f"the pre-flight must always answer 200: {status} {payload}"
    _assert_preview_shape(payload, plan="facility_probe")
    assert payload["channels"] == [
        {"channel": "motor1", "role": MOVABLE_ROLE},
        {"channel": "det1", "role": READABLE_ROLE},
    ], f"the declared channels are wrong or out of movable-first order: {payload['channels']}"

    status, payload = _request(
        "/plans/grid_scan/preview",
        "POST",
        {
            "readbacks": ["BPM1"],
            "axes": [{"setpoint": "COR1", "start": 0.0, "stop": 1.0, "num_points": 3}],
        },
    )
    assert status == 200, f"the pre-flight must always answer 200: {status} {payload}"
    _assert_preview_shape(payload, plan="grid_scan")
    assert payload["channels"] == [
        {"channel": "COR1", "role": MOVABLE_ROLE},
        {"channel": "BPM1", "role": READABLE_ROLE},
    ], f"the nested per-axis setpoint was not read as a declared movable: {payload['channels']}"


def test_preview_reports_an_unknown_plan_as_a_reason_not_a_404(
    deployed_catalog_stack: Path,
) -> None:
    """An unknown name is one more reason to have no trajectory, in the same shape.

    A 404 here would give the approval gate a second branch -- a status branch
    beside the ``ok`` branch -- for a case it must already handle.
    """
    status, payload = _request("/plans/no_such_plan_at_all/preview", "POST", {})

    assert status == 200, f"an unknown plan must not 404 on the pre-flight: {status} {payload}"
    _assert_preview_shape(payload, plan="no_such_plan_at_all")
    assert payload["ok"] is False
    assert payload["reason"] == PREVIEW_REASON_UNKNOWN_PLAN, f"wrong reason: {payload}"
    assert payload["channels"] == [], "there is no plan to declare channels for"


def test_preview_reports_a_body_that_is_not_parameters_as_a_plan_error(
    deployed_catalog_stack: Path,
) -> None:
    """Malformed and non-object bodies answer 200 too -- never FastAPI's own 422.

    The parameters are read raw for exactly this reason: a typed argument would
    have answered a bad body with a shape carrying no ``ok`` at all, which is
    the one answer this route promises never to give.
    """
    for label, data, content_type in (
        ("malformed JSON", b"{not json", "application/json"),
        ("a JSON array", b"[1, 2]", "application/json"),
        ("form-encoded", b"motor=motor1", "application/x-www-form-urlencoded"),
    ):
        status, payload = _post_raw("/plans/grid_scan/preview", data, content_type)
        assert status == 200, f"{label}: expected 200, got {status} {payload}"
        _assert_preview_shape(payload, plan="grid_scan")
        assert payload["reason"] == PREVIEW_REASON_PLAN_ERROR, f"{label}: {payload}"
        assert "JSON object" in payload["detail"], f"{label}: {payload}"


# ---------------------------------------------------------------------------
# The browse-only surface around the catalog: composable, and honest about it.
#
# Replaces a launch->read round trip that used to run here on the removed
# demo-runner knob. Execution -- for facility-tier plans as much as any other
# -- now belongs to tests/e2e/test_bluesky_queue_e2e.py, which drives real
# plan runs against a real queue server. What is left here is the half that is
# genuinely about the catalog: a discovered plan can be composed, and the
# deployment tells the truth about what it will do with it.
# ---------------------------------------------------------------------------


def test_deployment_reports_browse_only_and_names_the_flip(
    deployed_catalog_stack: Path,
) -> None:
    """A mock deployment is HEALTHY and says plainly that it cannot execute.

    ``status: "ok"`` is deliberately independent of ``can_execute``: a
    browse-only deployment is a working deployment, and gating liveness on
    capability would flap the container healthcheck over a configuration doing
    exactly what it was told.
    """
    status, body = _get("/health")
    assert status == 200, f"GET /health failed: {status} {body}"
    assert body["status"] == "ok", f"a browse-only bridge must still be ok: {body}"

    capability = body["capability"]
    assert capability["can_execute"] is False, f"mock cannot execute plans: {capability}"
    assert capability["reason"] == REASON_BROWSE_ONLY_CONNECTOR, f"wrong reason: {capability}"
    # Asserted against the bridge's own FLIP_COMMAND rather than a literal: the
    # subject is that the detail NAMES the flip command, and a copy of its
    # spelling here would pin whichever verb that constant happened to hold.
    assert FLIP_COMMAND in capability["detail"], (
        f"the browse-only detail must name the command that flips it: {capability}"
    )


def test_a_facility_plan_is_composable_but_unqueueable(deployed_catalog_stack: Path) -> None:
    """The facility-injected plan reaches the draft, and stops at the queue.

    This is the discovery claim carried one step further than ``GET /plans``:
    the plan is not merely listed, its schema resolves well enough for the
    shared draft to accept real arguments for it -- which is what an operator
    or the agent would do first. The enqueue then refuses with the capability
    record attached, because a deployment that cannot execute must never HOLD
    queue items: an item sitting in a queue reads as work that will happen.

    Asserted on ``detail.code``; a status-code-only check would keep passing
    while the refusal body drifted.
    """
    status, patched = _request(
        "/draft",
        "PATCH",
        {
            "plan_name": "facility_probe",
            "plan_args_patch": {
                "motor": "motor1",
                "detector": "det1",
                "start": 0.0,
                "stop": 2.0,
                "num": 3,
            },
            "client_id": "catalog-e2e",
        },
    )
    assert status == 200, f"PATCH /draft failed for the facility plan: {status} {patched}"
    assert patched["plan_name"] == "facility_probe"

    status, refusal = _request("/queue/items", "POST", {"draft_revision": patched["revision"]})
    assert status == 409, f"a browse-only enqueue must be refused: {status} {refusal}"
    detail = refusal.get("detail") if isinstance(refusal, dict) else None
    assert isinstance(detail, dict) and detail.get("code") == REASON_BROWSE_ONLY_CONNECTOR, (
        f"wrong refusal code on a browse-only enqueue: {refusal}"
    )
    assert isinstance(detail.get("capability"), dict), (
        f"the refusal must carry the capability record the status surface publishes: {refusal}"
    )

    status, queue = _get("/queue")
    assert status == 200, f"GET /queue failed: {status} {queue}"
    assert queue["status"]["items_in_queue"] == 0, (
        f"a browse-only deployment is holding queue items: {queue}"
    )
