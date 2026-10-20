"""Full-stack Docker proof of the trajectory service on the stand-in: its doors, its physics, its panel.

The deployment is the trajectory acceptance stack, rendered by
``tests/e2e/trajectory_acceptance/shape.py`` -- the one render the walk's
images come from -- and started with ``osprey up --dev``, so every image is
built here from the checked-out source. The service commands the live
stand-in on the host network; the archiver records the stand-in; one roster
user, ``alice``, is on one persona listing the ``trajectory`` panel. Before
the first test the fixture applies the acceptance scenario set to the
stand-in's state directory and sets the turns tracked to the calibrated window
through the setup route, so the stack the tests (and the stack-regression
replay that reuses this fixture) start from is the one the walk stages.

The trajectory container is driven over raw HTTP exactly as its two callers
reach it: the AGENT door (``/api/*``), which also demands the service's launch
token, and the PANEL door (``/panel/api/*``), the operator panel's. Both sit
behind the shared web gate, which admits a request by the operator secret it
carries and stamps the owner from the credential that matched -- never from
anything the request claims -- so every request sent with ``alice``'s minted
secret is owned by ``alice``.

What is pinned here and nowhere else:

* the stand-in serving as the service's control target, and the archiver
  recording the stand-in rather than the virtual accelerator beside it;
* the doors: a wrong launch token refused 403 ``launch_token_invalid``; on both
  doors a request without the secret refused, a body naming an owner refused
  400 ``owner_in_body``, a second holder refused 409 ``session_held``, the
  admitted owner read back through ``GET state``; an action's audit record
  landing under ``var/audit/trajectory/`` on the host;
* the physics the walk's rubric judges, as named data assertions on the served
  raw buffers, every threshold read from
  ``tests/e2e/trajectory_acceptance/sample_index.py``: the stored beam's
  pre-trigger share and its clearing by a cleaning chain, the RF-off decay,
  the RF-on survival over the whole window, the pilot tone's position spikes,
  the vertical trajectory the calibrated corrector kick moves, an applied step
  reducing it, and the published shot maximum covering the walk's bound.
  Stored charge is cleared by a cleaning chain before every shot a trajectory
  assertion reads, and the turns tracked are set up again after every
  scenario apply, which drops the session's writes;
* one panel smoke through ``alice``'s terminal: a real browser logs in through
  nginx with her login link, opens the panel the terminal proxies, opens a
  session and runs the selected chain; an in-container client starts the
  rendered ``trajectory`` MCP server and reads that shot's step, while an
  unstamped ``trajectory_shot`` is refused ``approval_missing``; the terminal
  refuses its token door without a secret and its panel door to the token;
* a continuous run ending by its lease once its heartbeats stop, and a
  selected reference surviving ``osprey down``, removal of the project's named
  volumes and a fresh ``osprey up`` (last, since it redeploys the stack).

The stand-in runs the VA image, built for ``linux/amd64``, and the acceptance
profile's timeouts are fixed for a host that runs it natively: an emulated
shot outlasts the connector's call timeout. On a Docker host of another
architecture the fixture fails before rendering, naming the host, rather than
letting every physics test time out one by one.

Stand-in machine state the physics needs -- the RF switch and the corrector
kick -- is written to the stand-in through the control-system connector the
deployment configures for it, the one route to the machine every caller uses.

Per-function flaky, NOT module-level: the health and session checks accept
one rerun on a timing assertion; the security negatives stay STRICT, so a
rerun can never sweep a refusal that failed to happen into a pass. The physics
and the panel hop are STRICT too: each of their tests acts on the session and
machine state the previous one left.

CONTAINER SAFETY: every docker/compose invocation names an EXACT resource --
the images the render names, the persona image ``<project>-operator:local``,
the containers ``<project>-web-alice`` and ``<project>-nginx`` -- never
``system prune``, never ``--volumes``, never a wildcard ``docker rm``/``rmi``.
Teardown goes through ``osprey down``, then exact-named removal of the
terminal tier's containers and of this project's own volumes
(``tests/e2e/_volumes.py``).

Gating: needs Docker and Playwright's chromium. Lives in ``tests/e2e/`` (never
collected by the fast lane) and is ``--ignore``'d by the shared e2e lane; runs
in its own ``trajectory-deploy-e2e`` CI job (same-repo PRs only, no secrets --
neither the service nor the terminal's agent reaches a model), which treats
any skip as a failure. One container stack per host: locally, never beside
another e2e stack, and never on the acceptance host, which runs the walk's
stack in the same port block.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import shutil
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e import _orm_stack
from tests.e2e._deploy_diagnostics import container_logs
from tests.e2e._volumes import remove_project_volumes
from tests.e2e.trajectory_acceptance import sample_index, shape

#: The compose project, the roster user and her persona: the acceptance
#: render's own, never restated.
PROJECT_NAME = shape.PROJECT_NAME
PROJECT = _orm_stack.project_prefix(PROJECT_NAME)
USER = shape.USER
PERSONA = shape.PERSONA
TRAJECTORY_CONTAINER = f"{PROJECT}-trajectory"
STANDIN_CONTAINER = f"{PROJECT}-live-standin"
WEB_CONTAINER = f"{PROJECT}-web-{USER}"
NGINX_CONTAINER = f"{PROJECT}-nginx"
#: ``osprey build`` renders a persona delta to ``build/<repo>-<persona>``, and
#: the persona image is tagged by that same name.
PERSONA_IMAGE = f"{PROJECT_NAME}-{PERSONA}:local"
#: The images the render names whose source this checkout changes: the project
#: image the trajectory service runs, the VA image the stand-in, the VA and the
#: archiver recorder run, and the bridge. Keys of ``resolve_image_defaults``.
REBUILT_IMAGE_KEYS = ("worker", "va", "bluesky_bridge")

#: The acceptance profile: the port block and the service's action timeout.
PROFILE = shape.profile()
PORT_BASE = int(PROFILE["port_base"])
ACTION_TIMEOUT_SEC = float(PROFILE["config"]["services.trajectory.action_timeout_s"])

#: The Docker architectures the stand-in's VA image runs on natively.
NATIVE_ARCHITECTURES = frozenset({"x86_64", "amd64"})

# The first-time native VA build, the project image and the persona image are
# slow on a cold cache.
DEPLOY_UP_TIMEOUT_SEC = 2400
# How long the container may take to answer /health at all ...
HEALTH_TIMEOUT_SEC = 300.0
# ... and, once it answers, how long until it reaches its control target.
REACHABLE_WITHIN_SEC = 30.0
#: How long one shot, step, setup or apply may take over HTTP: the service's
#: own action timeout, with room for the queue in front of it.
ACTION_HTTP_TIMEOUT_SEC = ACTION_TIMEOUT_SEC + 180.0

AGENT_PREFIX = "/api"
PANEL_PREFIX = "/panel/api"
DOORS = (AGENT_PREFIX, PANEL_PREFIX)

SECRET_HEADER = "X-Osprey-Terminal-Secret"
CLIENT_HEADER = "X-Trajectory-Client"

#: The trajectory panel as ``alice``'s terminal proxies it: the operator
#: bundle and its panel door, and the agent's token door.
TERMINAL_PANEL = "/panel/trajectory/panel/"
TERMINAL_PANEL_DOOR = "/panel/trajectory/panel/api"
TERMINAL_TOKEN_DOOR = "/panel/trajectory/api"
#: How long the terminal's agent may take to start and carry its panel token.
AGENT_START_TIMEOUT_SEC = 240.0
#: How long one browser expectation that waits on no action may take.
SETTLE_MS = 30_000
#: The panel's three raw plots, which a shot's buffers draw.
RAW_PLOTS = ("plot-raw-sum", "plot-raw-x", "plot-raw-y")

#: The model the trajectory service steers.
MODEL = sample_index.MODEL
#: Every calibrated threshold, each window turn placed on its plotted sample.
THRESHOLDS = sample_index.thresholds()
TRIGGER = int(THRESHOLDS["trigger_sample"])
TURNS_TRACKED = int(THRESHOLDS["turns_tracked"])

#: The declared chain steps, and the chains the tests run from them: the full
#: declared chain; a shot that keeps the stored charge it read (no cleaning);
#: and the cleaning chain alone, which empties the ring before a measurement.
FULL_CHAIN = ["arm", "inject", "clean", "readout"]
KEEP_STORED_CHAIN = ["arm", "inject", "readout"]
CLEANING_CHAIN = ["clean"]
#: The chain a continuous run reruns.
CONTINUOUS_STEPS = KEEP_STORED_CHAIN

#: The stand-in records the physics stages directly: the RF switch (0 off,
#: 1 on) and the turns-tracked setpoint the setup route moves.
RF_ON_SP = "SR:RF:ON:SP"
TURNS_SP = "SR:DIAG:TBT:TURNS:SP"
TURNS_RB = "SR:DIAG:TBT:TURNS:RB"
TBT_SUM_PROBE = "SR:DIAG:BPM:01:TBT:SUM"
RF_OFF, RF_ON = 0, 1
#: The raw buffer's plane order.
PLANE_X, PLANE_Y, PLANE_SUM = 0, 1, 2

#: The pilot tone spikes a monitor's positions on every tenth sample (the
#: engine's ``TONE_SPIKE_PERIOD``). On the vertical plane of an unkicked ring,
#: whose positions are the launch jitter alone, the spiked samples' rms stands
#: above the others' by an order of magnitude; with the tone off the two match.
TONE_SPIKE_RATIO_MIN = 3.0
TONE_OFF_RATIO_MAX = 2.0
#: The calibrated kick moves the first turn's vertical trajectory to an rms
#: well above the launch jitter alone (about twenty times it).
KICK_RMS_FACTOR_MIN = 4.0

#: How long an applied scenario may take to show inside the stand-in.
SCENARIO_VISIBLE_SEC = 30.0
#: From a write on the stand-in to that value being readable out of the
#: archive: the recorder's cadence plus one poll.
RECORDER_BUDGET_SEC = 30.0
RECORDER_POLL_SEC = 2.0
#: The step a stand-in setpoint is moved by to be seen in the archive.
ARCHIVED_STEP = 0.05

#: The continuous run's lease and heartbeat (``continuous.DEFAULT_LEASE_S`` /
#: ``DEFAULT_HEARTBEAT_S``), and how long heartbeats keep it alive -- past
#: two leases, so the run outliving a lease proves the heartbeats move it.
LEASE_SEC = 10.0
HEARTBEAT_SEC = 5.0
HEARTBEATS_FOR_SEC = 2 * LEASE_SEC + HEARTBEAT_SEC
REFERENCE_NAME = "kicked-and-corrected"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]


def _port(slot: str) -> int:
    from osprey.port_layout import default_port

    return default_port(slot, base=PORT_BASE)


#: ``alice``'s terminal (roster index 0), and the origin nginx publishes --
#: the one a browser logs in at and every mutating request must name.
WEB_URL = f"http://127.0.0.1:{_port('web')}"
WS_URL = f"ws://127.0.0.1:{_port('web')}"
ORIGIN = f"http://127.0.0.1:{_port('nginx')}"
#: The trajectory service, on the host network and bound to loopback.
TRAJECTORY_URL = f"http://127.0.0.1:{_port('trajectory')}"


# ---------------------------------------------------------------------------
# Build/deploy scaffold
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


def _minted(repo: Path, name: str) -> str:
    """A secret `osprey up` minted into the deployment repo's ``.env``."""
    from osprey_connectors.dotenv import parse_dotenv_file

    env_path = repo / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — {name} was not minted"
    value = parse_dotenv_file(env_path).get(name)
    assert value, f"{name} missing/empty in the deployment repo's .env"
    return value


def _request(
    path: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    base: str = TRAJECTORY_URL,
) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    all_headers = {"Content-Type": "application/json"} if data is not None else {}
    all_headers.update(headers or {})
    req = urllib.request.Request(f"{base}{path}", data=data, method=method, headers=all_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # localhost
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    text = raw.decode("utf-8", errors="replace")
    try:
        return status, json.loads(text)
    except ValueError:
        return status, text


def _code(body: Any) -> str | None:
    """The refusal code a door answered with: ``{"detail": {"code": ...}}``."""
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail.get("code") if isinstance(detail, dict) else None


class DeployedStack:
    def __init__(self, repo: Path, secret: str, launch_token: str, owner: str, osprey_bin: Path):
        self.repo = repo
        self.secret = secret
        self.launch_token = launch_token
        #: The roster user whose minted secret this stack presents — the owner
        #: the gate stamps on every admitted request.
        self.owner = owner
        self.osprey_bin = osprey_bin
        # What the ordered panel-hop tests hand on, one to the next.
        self.agent_env: dict[str, str] | None = None
        self.panel_page: Any = None
        self.panel_digest: str | None = None
        # What the ordered physics tests hand on, one to the next.
        self.client_id: str | None = None
        #: The turns tracked the stand-in rests at, read before anything wrote it.
        self.turns_rest: float | None = None
        self.rf_on_raw: Any = None
        self.kick_origin: float | None = None
        self.kicked_shot: dict[str, Any] | None = None
        self.applied_digests: list[str] = []
        self.corrected_shot: dict[str, Any] | None = None

    def headers(self, prefix: str, **extra: str) -> dict[str, str]:
        """The credentials a door's legitimate caller carries."""
        headers = {SECRET_HEADER: self.secret, **extra}
        if prefix == AGENT_PREFIX:
            headers["X-Launch-Token"] = self.launch_token
        return headers

    @property
    def config(self) -> dict[str, Any]:
        """The as-built config the deployment runs on."""
        return yaml.safe_load((self.repo / "build" / "config.yml").read_text(encoding="utf-8"))


def _logs() -> str:
    return container_logs(TRAJECTORY_CONTAINER)


def _terminal_logs() -> str:
    return container_logs(WEB_CONTAINER)


def _user_secret(repo: Path) -> str:
    """``alice``'s terminal secret, as ``osprey up`` minted it into ``.env``."""
    from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var

    return _minted(repo, terminal_secret_var(USER))


def _rebuilt_images(repo: Path) -> list[str]:
    """The exact image names this deploy builds from the checkout, as the render names them."""
    from osprey.deployment.compose_generator import resolve_image_defaults

    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    defaults = resolve_image_defaults(config)
    return [defaults[key] for key in REBUILT_IMAGE_KEYS] + [PERSONA_IMAGE]


def _remove_terminal_tier() -> None:
    """Exact-named removal of the terminal tier's containers; never an assertion."""
    for name in (WEB_CONTAINER, NGINX_CONTAINER):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=120)


def _release(stack: DeployedStack, prefix: str, client_id: str) -> None:
    status, body = _request(
        f"{prefix}/session",
        "DELETE",
        headers=stack.headers(prefix, **{CLIENT_HEADER: client_id}),
    )
    assert status == 200, f"release through {prefix} failed: {status} {body!r}"


def _open(stack: DeployedStack, prefix: str) -> dict[str, Any]:
    status, body = _request(f"{prefix}/session", "POST", headers=stack.headers(prefix))
    assert status == 200, f"open through {prefix} refused: {status} {body!r}\n{_logs()}"
    return body


def _active_inside_standin() -> str:
    """The active scenarios file as the stand-in container reads it."""
    result = subprocess.run(
        ["docker", "exec", STANDIN_CONTAINER, "cat", "/state/simulation/active_scenarios"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout if result.returncode == 0 else ""


def _sim_apply(stack: DeployedStack, *scenarios: str) -> None:
    """``osprey sim apply`` the scenarios, then wait until the stand-in serves them.

    ``--no-seed``: the apply only rewrites the active set in the stand-in's
    state directory and never reseeds the logbook or the archive. Once the
    container sees the file the stand-in rebuilds at the new set -- which drops
    every write made to it since. The file's content is no proof of that (an
    unchanged set reads the same before and after), so a turns tracked written
    off its rest value first marks the old build: the apply has taken once the
    stand-in reads its rest value again.
    """
    from osprey_connectors.simulation import DEFAULT_TICK_S
    from osprey_connectors.workspace import resolve_simulation_state_dir

    if stack.turns_rest is None:
        stack.turns_rest = _standin_read(stack, TURNS_RB)
    assert stack.turns_rest != TURNS_TRACKED, (
        f"the stand-in rests at {TURNS_TRACKED} turns tracked, so a rebuild cannot be told apart"
    )
    _standin_write(stack, TURNS_SP, TURNS_TRACKED)
    applied = _run(
        [str(stack.osprey_bin), "sim", "apply", *scenarios, "--no-seed"],
        cwd=stack.repo,
        timeout=120,
    )
    assert applied.returncode == 0, (
        f"osprey sim apply {' '.join(scenarios)} failed (rc={applied.returncode}):\n"
        f"--- stdout ---\n{applied.stdout}\n--- stderr ---\n{applied.stderr}"
    )
    written = (
        resolve_simulation_state_dir(stack.config, stack.repo) / "active_scenarios"
    ).read_text(encoding="utf-8")
    deadline = time.monotonic() + SCENARIO_VISIBLE_SEC
    while _active_inside_standin() != written:
        assert time.monotonic() < deadline, (
            f"{STANDIN_CONTAINER} never saw the active scenarios the host wrote ({written!r})"
        )
        time.sleep(0.2)
    while _standin_read(stack, TURNS_RB) != stack.turns_rest:
        assert time.monotonic() < deadline, (
            f"{STANDIN_CONTAINER} never rebuilt at the applied scenarios: {TURNS_RB} still "
            f"reads the {TURNS_TRACKED} written before the apply"
        )
        time.sleep(DEFAULT_TICK_S)


def _set_up(headers: dict[str, str], **settings: Any) -> dict[str, Any]:
    """Apply a setup through the panel door: the turns tracked, plus *settings*."""
    status, done = _request(
        f"{PANEL_PREFIX}/setup",
        "POST",
        body={"settings": {"turns_tracked": TURNS_TRACKED, **settings}},
        headers=headers,
        timeout=ACTION_HTTP_TIMEOUT_SEC,
    )
    assert status == 200, f"setup refused: {status} {done!r}\n{_logs()}"
    assert not done["unsettled"], f"the setup left settings unsettled: {done!r}"
    assert done["setup"] and not done["setup"]["void"], f"the setup did not take: {done!r}"
    assert done["setup"]["turns_tracked"] == TURNS_TRACKED, done["setup"]
    return done


def _switch_to_standin(stack: DeployedStack) -> None:
    """Move ``alice``'s control target to the stand-in, as her terminal's target chip does.

    The deployment's baseline is the virtual accelerator, and the service admits
    only owners whose control context names the target it drives; the switch
    lands in her record, which the service reads on every action.
    """
    status, body = _request(
        "/api/terminal/target",
        "POST",
        body={"target": "standin"},
        headers={SECRET_HEADER: stack.secret, "Origin": ORIGIN},
        base=WEB_URL,
    )
    assert status == 202, f"the switch to the stand-in was refused: {status} {body!r}"
    deadline = time.monotonic() + REACHABLE_WITHIN_SEC
    while True:
        status, grant = _request(
            f"{PANEL_PREFIX}/session", "POST", headers=stack.headers(PANEL_PREFIX)
        )
        if status == 200:
            _release(stack, PANEL_PREFIX, grant["client_id"])
            return
        assert _code(grant) == "control_target_mismatch", (
            f"open after the switch: {status} {grant!r}\n{_logs()}"
        )
        assert time.monotonic() < deadline, (
            f"{USER} is still not on the stand-in {REACHABLE_WITHIN_SEC:.0f}s after the switch"
        )
        time.sleep(1.0)


def _stage(stack: DeployedStack) -> None:
    """The owner on the stand-in, the acceptance scenario set applied to it, then
    the turns tracked set up."""
    _switch_to_standin(stack)
    _sim_apply(stack, *shape.ACTIVE_SET)
    grant = _open(stack, PANEL_PREFIX)
    try:
        _set_up(stack.headers(PANEL_PREFIX, **{CLIENT_HEADER: grant["client_id"]}))
    finally:
        _release(stack, PANEL_PREFIX, grant["client_id"])


def _docker_architecture() -> str:
    """The architecture this Docker host runs containers natively on."""
    result = subprocess.run(
        ["docker", "info", "--format", "{{.Architecture}}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedStack]:
    architecture = _docker_architecture()
    if architecture not in NATIVE_ARCHITECTURES:
        pytest.fail(
            f"this Docker host runs {architecture or 'an unknown architecture'}, so the "
            "linux/amd64 stand-in would run emulated, past the acceptance profile's fixed "
            "timeouts; run this module on an x86_64 Docker host"
        )
    osprey_bin = _orm_stack.find_osprey_console_script()
    base = tmp_path_factory.mktemp("trajectory_build")
    try:
        repo = shape.render(base)
    except shape.RenderError as exc:
        pytest.fail(str(exc))
    # Fresh --dev builds so the deployed containers run CURRENT source.
    _orm_stack.force_image_rebuild(*_rebuilt_images(repo))

    try:
        up = _run([str(osprey_bin), "up", "-d", "--dev"], cwd=repo, timeout=DEPLOY_UP_TIMEOUT_SEC)
        if up.returncode != 0:
            pytest.fail(
                f"osprey up -d --dev failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}\n{_logs()}"
            )
        try:
            _orm_stack.wait_for_health(f"{TRAJECTORY_URL}/health", HEALTH_TIMEOUT_SEC)
        except AssertionError as exc:
            pytest.fail(f"{exc}\n{_logs()}")
        try:
            _orm_stack.wait_for_health(f"{WEB_URL}/health", HEALTH_TIMEOUT_SEC)
        except AssertionError as exc:
            pytest.fail(f"{exc}\n{_terminal_logs()}")
        stack = DeployedStack(
            repo=repo,
            secret=_user_secret(repo),
            launch_token=_minted(repo, "TRAJECTORY_LAUNCH_TOKEN"),
            owner=USER,
            osprey_bin=osprey_bin,
        )
        _stage(stack)
        try:
            yield stack
        finally:
            _close_panel(stack)
    finally:
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
        _remove_terminal_tier()
        remove_project_volumes(PROJECT)


# ---------------------------------------------------------------------------
# 1. Health: the stand-in is the target, and serves
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
@pytest.mark.usefixtures("deployed_stack")
def test_health_reaches_the_standin_within_30s() -> None:
    """``/health`` is gate-exempt and names its target -- the stand-in -- reachable."""
    deadline = time.monotonic() + REACHABLE_WITHIN_SEC
    body: Any = None
    while time.monotonic() < deadline:
        status, body = _request("/health")
        if status == 200 and isinstance(body, dict) and body.get("target") == "reachable":
            break
        time.sleep(1.0)
    assert isinstance(body, dict) and body.get("target") == "reachable", (
        f"/health never reported target: reachable within {REACHABLE_WITHIN_SEC:.0f}s; "
        f"last: {body!r}\n{_logs()}"
    )
    assert body.get("control_target") == "standin", f"/health names another target: {body!r}"
    health = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Health.Status}}", STANDIN_CONTAINER],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert health.stdout.strip() == "healthy", (
        f"{STANDIN_CONTAINER} health: {health.stdout.strip() or health.stderr.strip()}\n"
        f"{container_logs(STANDIN_CONTAINER)}"
    )


# ---------------------------------------------------------------------------
# 2. Refusals (strict)
# ---------------------------------------------------------------------------


def test_agent_door_refuses_a_wrong_launch_token(deployed_stack: DeployedStack) -> None:
    headers = deployed_stack.headers(AGENT_PREFIX)
    headers["X-Launch-Token"] = "not-" + deployed_stack.launch_token
    for method, path in (("GET", "/state"), ("POST", "/session")):
        status, body = _request(f"{AGENT_PREFIX}{path}", method, headers=headers)
        assert status == 403, f"{method} {path} with a wrong token: {status} {body!r}"
        assert _code(body) == "launch_token_invalid", body


@pytest.mark.parametrize("prefix", DOORS)
def test_door_refuses_a_request_without_the_secret(
    deployed_stack: DeployedStack, prefix: str
) -> None:
    headers = deployed_stack.headers(prefix)
    del headers[SECRET_HEADER]
    for method, path in (("GET", "/state"), ("POST", "/session")):
        status, body = _request(f"{prefix}{path}", method, headers=headers)
        assert status == 401, f"{method} {prefix}{path} without the secret: {status} {body!r}"
    # The wrong secret is no better than none.
    headers[SECRET_HEADER] = "not-" + deployed_stack.secret
    status, body = _request(f"{prefix}/state", headers=headers)
    assert status == 401, f"GET {prefix}/state with a wrong secret: {status} {body!r}"


@pytest.mark.parametrize("prefix", DOORS)
def test_door_refuses_an_owner_in_the_body(deployed_stack: DeployedStack, prefix: str) -> None:
    status, body = _request(
        f"{prefix}/session",
        "POST",
        body={"owner": "someone-else"},
        headers=deployed_stack.headers(prefix),
    )
    assert status == 400, f"open naming an owner through {prefix}: {status} {body!r}"
    assert _code(body) == "owner_in_body", body
    status, state = _request(f"{prefix}/state", headers=deployed_stack.headers(prefix))
    assert status == 200 and state["holder"] is None, f"a refused open left a holder: {state!r}"


# ---------------------------------------------------------------------------
# 3. One holder, the admitted owner
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
@pytest.mark.parametrize(("first", "second"), [DOORS, DOORS[::-1]])
def test_second_holder_is_refused_and_state_names_the_admitted_owner(
    deployed_stack: DeployedStack, first: str, second: str
) -> None:
    grant = _open(deployed_stack, first)
    try:
        assert grant["owner"] == deployed_stack.owner, grant
        assert grant.get("client_id"), grant
        for prefix in DOORS:
            status, state = _request(f"{prefix}/state", headers=deployed_stack.headers(prefix))
            assert status == 200, f"GET {prefix}/state: {status} {state!r}"
            assert state["holder"]["owner"] == deployed_stack.owner, state
            assert state["caller"] == deployed_stack.owner, state
            assert state["holder"]["kind"] == ("agent" if first == AGENT_PREFIX else "panel"), state
            assert "client_id" not in state, f"state leaked the holder's client id: {state!r}"
            for secret in (deployed_stack.secret, deployed_stack.launch_token):
                assert secret not in json.dumps(state), "state echoed a credential"

        status, body = _request(f"{second}/session", "POST", headers=deployed_stack.headers(second))
        assert status == 409, f"second open through {second}: {status} {body!r}"
        assert _code(body) == "session_held", body
    finally:
        _release(deployed_stack, first, grant["client_id"])
    status, state = _request(f"{first}/state", headers=deployed_stack.headers(first))
    assert status == 200 and state["holder"] is None, f"release left a holder: {state!r}"


# ---------------------------------------------------------------------------
# 4. The audit record lands on the host
# ---------------------------------------------------------------------------


def _audit_records(repo: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((repo / "var" / "audit" / "trajectory").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_an_action_files_its_audit_record_under_var_audit_trajectory(
    deployed_stack: DeployedStack,
) -> None:
    from osprey.audit.envelope import DECISION_ALLOWED, SURFACE_TRAJECTORY_SERVICE

    def opens() -> list[dict[str, Any]]:
        return [
            r
            for r in _audit_records(deployed_stack.repo)
            if r.get("surface") == SURFACE_TRAJECTORY_SERVICE
            and r.get("subject") == "session_open"
            and r.get("decision") == DECISION_ALLOWED
        ]

    before = len(opens())
    grant = _open(deployed_stack, PANEL_PREFIX)
    _release(deployed_stack, PANEL_PREFIX, grant["client_id"])

    deadline = time.monotonic() + 10.0
    while len(opens()) <= before and time.monotonic() < deadline:
        time.sleep(0.5)
    new = opens()[before:]
    assert new, (
        f"no allowed session_open record under {deployed_stack.repo / 'var/audit/trajectory'}; "
        f"found {_audit_records(deployed_stack.repo)!r}"
    )
    assert new[-1].get("actor") == deployed_stack.owner, new[-1]
    raw = "".join(
        p.read_text(encoding="utf-8")
        for p in (deployed_stack.repo / "var" / "audit" / "trajectory").glob("*.jsonl")
    )
    for secret in (deployed_stack.secret, deployed_stack.launch_token):
        assert secret not in raw, "an audit record carries a credential"


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. The physics on the stand-in (STRICT, ordered: one held session, each test
#    acting on the machine the previous one left)
# ---------------------------------------------------------------------------


def _on_standin(stack: DeployedStack, work: Callable[[Any], Awaitable[Any]]) -> Any:
    """Run *work* with a connected connector to the stand-in, as the deployment configures it.

    The write guard re-reads the posture from the global config, so the
    deployment's rendered config is made the global one for the call only and
    the config singletons are reset around it.
    """
    from osprey_connectors import config as config_module
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors
    from osprey_connectors.types import LIVE_STANDIN

    async def run() -> Any:
        register_builtin_connectors()
        connector = await ConnectorFactory.create_control_system_connector(
            {**stack.config["control_system"], "type": LIVE_STANDIN}, control_target="standin"
        )
        try:
            return await work(connector)
        finally:
            await connector.disconnect()

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CONFIG_FILE", str(stack.repo / "build" / "config.yml"))
        patch.setattr(config_module, "_default_config", None)
        patch.setattr(config_module, "_default_configurable", None)
        return asyncio.run(run())


def _standin_write(stack: DeployedStack, address: str, value: float) -> None:
    """One checked write to the stand-in, confirmed by reading it back."""

    async def work(connector: Any) -> float:
        await connector.write_channel_checked(address, value)
        return float((await connector.read_channel(address)).value)

    read = _on_standin(stack, work)
    assert math.isclose(read, value, abs_tol=1e-9), f"{address} reads {read} after writing {value}"


def _standin_read(stack: DeployedStack, address: str) -> float:
    async def work(connector: Any) -> float:
        return float((await connector.read_channel(address)).value)

    return _on_standin(stack, work)


def _held(stack: DeployedStack) -> dict[str, str]:
    """The panel credentials plus the session the physics tests hold."""
    assert stack.client_id, "no session is held: an earlier physics test failed"
    return stack.headers(PANEL_PREFIX, **{CLIENT_HEADER: stack.client_id})


def _release_held(stack: DeployedStack) -> None:
    if stack.client_id is not None:
        client_id, stack.client_id = stack.client_id, None
        _release(stack, PANEL_PREFIX, client_id)


def _chain(stack: DeployedStack, steps: list[str]) -> dict[str, Any] | None:
    """Run *steps* as one panel chain of one shot; answers the shot a readout read."""
    status, done = _request(
        f"{PANEL_PREFIX}/shot",
        "POST",
        body={"steps": steps, "n_shots": 1},
        headers=_held(stack),
        timeout=ACTION_HTTP_TIMEOUT_SEC,
    )
    assert status == 200, f"chain {steps} refused: {status} {done!r}\n{_logs()}"
    assert done["steps_run"] == steps, done["steps_run"]
    shot = done["shot"]
    if "readout" not in steps:
        assert shot is None, f"a chain without a readout read a shot: {shot!r}"
        return None
    assert shot is not None and shot["shots_included"] == 1, (
        f"chain {steps} included no shot: {shot!r}\n{_logs()}\n{container_logs(STANDIN_CONTAINER)}"
    )
    assert shot["model"] == MODEL, shot["model"]
    return shot


def _measure(stack: DeployedStack) -> dict[str, Any]:
    """A shot into an emptied ring: the cleaning chain, then the full declared chain."""
    _chain(stack, CLEANING_CHAIN)
    shot = _chain(stack, FULL_CHAIN)
    assert shot is not None
    return shot


def _raw(stack: DeployedStack, shot: Mapping[str, Any]) -> Any:
    """The shot's served raw buffer: its one injection, ``(plane, monitor, sample)``."""
    from osprey.services.trajectory.raw import decode_raw

    req = urllib.request.Request(
        f"{TRAJECTORY_URL}{PANEL_PREFIX}/shots/{shot['shot_id']}/raw",
        headers=stack.headers(PANEL_PREFIX),
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # localhost
        buffer = decode_raw(resp.read())
    assert buffer.header.trigger_sample == TRIGGER, buffer.header
    n_injections, planes, _monitors, samples = buffer.header.shape
    assert n_injections == 1 and planes == 3, buffer.header
    assert samples > THRESHOLDS["last_index"], (
        f"the buffer holds {samples} samples; the {TURNS_TRACKED}-turn window ends at sample "
        f"{THRESHOLDS['last_index']}; {_turns_evidence(stack)}"
    )
    return buffer.samples[0]


def _turns_evidence(stack: DeployedStack) -> str:
    """What the stand-in holds now: its turns tracked and one monitor's series length."""
    series = TBT_SUM_PROBE

    async def work(connector: Any) -> str:
        turns = (await connector.read_channel(TURNS_RB)).value
        try:
            length: Any = len((await connector.read_channel(series)).value)
        except Exception as exc:  # evidence only
            length = repr(exc)
        return f"the stand-in reads {TURNS_RB} = {turns}, {series} holds {length} samples"

    try:
        return _on_standin(stack, work)
    except Exception as exc:  # evidence only
        return f"the stand-in could not be read: {exc!r}"


def _monitor(shot: Mapping[str, Any], index: int) -> str:
    order = shot["bpm_order"]
    return order[index] if index < len(order) else f"monitor #{index}"


def _rms(values: Sequence[float | None]) -> float:
    kept = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    assert kept, "no monitor read the plane"
    return math.sqrt(sum(v * v for v in kept) / len(kept))


def _spike_ratio(series: Any) -> float:
    """The rms of a position series on the tone's spike samples over the rms of the
    others, inside the tracked window."""
    import numpy as np

    from osprey.simulation.engines.pyat_faults import TONE_SPIKE_PERIOD

    index = np.arange(series.shape[-1])
    window = (index >= TRIGGER) & (index <= THRESHOLDS["last_index"])
    spiked = window & (index % TONE_SPIKE_PERIOD == 0)
    plain = window & (index % TONE_SPIKE_PERIOD != 0)
    return float(np.sqrt(np.nanmean(series[spiked] ** 2)) / np.sqrt(np.nanmean(series[plain] ** 2)))


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
def test_the_published_shot_maximum_covers_the_walks_bound_and_the_window_is_set_up(
    deployed_stack: DeployedStack,
) -> None:
    status, state = _request(f"{PANEL_PREFIX}/state", headers=deployed_stack.headers(PANEL_PREFIX))
    assert status == 200, f"GET state: {status} {state!r}"
    assert state["n_shots_max"] is not None and state["n_shots_max"] >= shape.N_SHOTS_MAX, (
        f"the service admits {state['n_shots_max']} shots per action; the walk asks for up to "
        f"{shape.N_SHOTS_MAX}"
    )
    setup = state["setup"]
    assert setup and not setup["void"], f"the staged setup does not hold: {setup!r}"
    assert setup["turns_tracked"] == TURNS_TRACKED, setup


def test_the_archiver_records_the_standin(
    deployed_stack: DeployedStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A setpoint moved on the stand-in alone comes back out of the archive.

    The virtual accelerator beside it is never written, so an archived sample
    carrying the new value can only have been recorded from the stand-in.
    """
    from datetime import UTC, datetime, timedelta

    from osprey.simulation.apply import archiver_store_config
    from osprey_connectors.archiver.mongodb_archiver_connector import MongoDBArchiverConnector

    store = archiver_store_config(deployed_stack.config, deployed_stack.repo)
    assert store is not None, "the deployment archives nothing in its store"
    monkeypatch.setenv(store["password_env"], store["password"])
    limits = json.loads(
        (deployed_stack.repo / "build" / "data" / "channel_limits.json").read_text(encoding="utf-8")
    )
    address = next(
        name
        for name in sorted(limits)
        if name.endswith(":SP") and limits[name]["writable"] and name != shape.KICK["record"]
    )
    original = _standin_read(deployed_stack, address)
    target = original + ARCHIVED_STEP
    written_at = datetime.now(UTC)
    _standin_write(deployed_stack, address, target)
    try:
        archive = MongoDBArchiverConnector()
        asyncio.run(
            archive.connect(
                {
                    "host": store["host"],
                    "port": store["port"],
                    "name": store["database"],
                    "collection": store["collection"],
                    "auth": {
                        "source": store["auth_source"],
                        "username": store["username"],
                        "password_env": store["password_env"],
                    },
                }
            )
        )
        deadline = time.monotonic() + RECORDER_BUDGET_SEC
        seen: list[float] = []
        while not any(abs(value - target) < 1e-6 for value in seen):
            assert time.monotonic() < deadline, (
                f"{address} was set to {target} on the stand-in but no archived sample carried "
                f"it within {RECORDER_BUDGET_SEC:.0f}s; the archive held {seen[-5:]}"
            )
            time.sleep(RECORDER_POLL_SEC)
            frame = asyncio.run(
                archive.get_data([address], written_at - timedelta(seconds=5), datetime.now(UTC))
            )
            seen = [float(v) for v in frame["value"]] if len(frame) else []
    finally:
        _standin_write(deployed_stack, address, original)


def test_the_stored_beam_reads_its_share_before_the_trigger(deployed_stack: DeployedStack) -> None:
    """A ring storing the scenario's charge, RF on: every monitor's mean sum before
    the trigger stands at least at the calibrated stored share of the peak after it."""
    _sim_apply(deployed_stack, *shape.ACTIVE_SET)
    grant = _open(deployed_stack, PANEL_PREFIX)
    assert grant["owner"] == deployed_stack.owner, grant
    deployed_stack.client_id = grant["client_id"]
    _set_up(_held(deployed_stack))
    _standin_write(deployed_stack, RF_ON_SP, RF_ON)

    shot = _chain(deployed_stack, KEEP_STORED_CHAIN)
    assert shot is not None
    sums = _raw(deployed_stack, shot)[PLANE_SUM]
    threshold = THRESHOLDS["stored"]["threshold"]
    short = {
        _monitor(shot, i): round(ratio, 4)
        for i, series in enumerate(sums)
        if (ratio := sample_index.stored_ratio(series, TRIGGER)) < threshold
    }
    assert not short, (
        f"pre-trigger/post-injection sum ratio below the stored threshold {threshold:.4f} "
        f"(stored share {THRESHOLDS['stored']['share']:.4f}) on {len(short)} of "
        f"{len(sums)} monitors: {short}"
    )


def test_a_cleaning_chain_clears_the_stored_beam(deployed_stack: DeployedStack) -> None:
    """After the cleaning chain the next shot reads an empty ring: every monitor's
    pre-trigger ratio falls below half the stored threshold."""
    _chain(deployed_stack, CLEANING_CHAIN)
    shot = _chain(deployed_stack, KEEP_STORED_CHAIN)
    assert shot is not None
    sums = _raw(deployed_stack, shot)[PLANE_SUM]
    ceiling = THRESHOLDS["stored"]["threshold"] / 2
    stored = {
        _monitor(shot, i): round(ratio, 4)
        for i, series in enumerate(sums)
        if (ratio := sample_index.stored_ratio(series, TRIGGER)) >= ceiling
    }
    assert not stored, (
        f"after the cleaning chain the pre-trigger/post-injection sum ratio stays at or above "
        f"{ceiling:.4f} on {len(stored)} of {len(sums)} monitors: {stored}"
    )


def test_with_the_rf_off_the_bunch_decays(deployed_stack: DeployedStack) -> None:
    """RF off: every monitor keeps more than the high share at the high turn and
    less than the low share at the low turn, of its first-turn sum."""
    rf_off = THRESHOLDS["rf_off"]
    high, low = rf_off["high"], rf_off["low"]
    _standin_write(deployed_stack, RF_ON_SP, RF_OFF)
    try:
        shot = _measure(deployed_stack)
        sums = _raw(deployed_stack, shot)[PLANE_SUM]
    finally:
        _standin_write(deployed_stack, RF_ON_SP, RF_ON)
    early = {
        _monitor(shot, i): round(share, 4)
        for i, series in enumerate(sums)
        if (share := sample_index.share_at(series, TRIGGER, high["turn"])) <= high["share_above"]
    }
    late = {
        _monitor(shot, i): round(share, 4)
        for i, series in enumerate(sums)
        if (share := sample_index.share_at(series, TRIGGER, low["turn"])) >= low["share_below"]
    }
    assert not early, (
        f"RF off: the sum at turn {high['turn']} (sample {high['index']}) is not above "
        f"{high['share_above']} of the first turn's on {early}"
    )
    assert not late, (
        f"RF off: the sum at turn {low['turn']} (sample {low['index']}) is not below "
        f"{low['share_below']} of the first turn's on {late}"
    )


def test_with_the_rf_on_the_bunch_survives_the_window(deployed_stack: DeployedStack) -> None:
    """RF on: every monitor keeps at least the survival share of its first-turn sum
    on every turn of the tracked window."""
    rf_on = THRESHOLDS["rf_on"]
    shot = _measure(deployed_stack)
    raw = _raw(deployed_stack, shot)
    deployed_stack.rf_on_raw = raw
    lost = {}
    for i, series in enumerate(raw[PLANE_SUM]):
        worst = min(
            sample_index.share_at(series, TRIGGER, turn)
            for turn in range(1, rf_on["last_turn"] + 1)
        )
        if worst < rf_on["share_at_least"]:
            lost[_monitor(shot, i)] = round(worst, 4)
    assert not lost, (
        f"RF on: the sum falls below {rf_on['share_at_least']} of the first turn's within "
        f"turns 1-{rf_on['last_turn']} (samples {rf_on['first_index']}-{rf_on['last_index']}) "
        f"on {lost}"
    )


def test_the_pilot_tone_spikes_the_positions(deployed_stack: DeployedStack) -> None:
    """With the pilot tone set up on, every monitor's vertical positions spike on the
    tone's samples; the RF-on shot before it, tone off, shows no such spikes."""
    plain = deployed_stack.rf_on_raw
    assert plain is not None, "the RF-on shot was not taken"
    _set_up(_held(deployed_stack), pilot_tone=1)
    try:
        shot = _measure(deployed_stack)
        toned = _raw(deployed_stack, shot)[PLANE_Y]
    finally:
        _set_up(_held(deployed_stack), pilot_tone=0)
    flat = {
        _monitor(shot, i): round(ratio, 2)
        for i, series in enumerate(toned)
        if (ratio := _spike_ratio(series)) < TONE_SPIKE_RATIO_MIN
    }
    assert not flat, (
        f"pilot tone on: the spiked-sample/other-sample vertical rms is below "
        f"{TONE_SPIKE_RATIO_MIN} on {flat}"
    )
    control = statistics.median(_spike_ratio(series) for series in plain[PLANE_Y])
    assert control <= TONE_OFF_RATIO_MAX, (
        f"pilot tone off: the median spiked-sample/other-sample vertical rms is {control:.2f}, "
        f"above {TONE_OFF_RATIO_MAX}: the spikes are not the tone's"
    )


def test_the_calibrated_kick_moves_the_vertical_trajectory(deployed_stack: DeployedStack) -> None:
    """The kick the walk stages raises the first turn's vertical rms well above the
    launch jitter of the same ring unkicked."""
    record, value_a = shape.KICK["record"], float(shape.KICK["value_a"])
    base = _measure(deployed_stack)
    origin = _standin_read(deployed_stack, record)
    deployed_stack.kick_origin = origin
    _standin_write(deployed_stack, record, origin + value_a)
    kicked = _measure(deployed_stack)
    deployed_stack.kicked_shot = kicked
    before, after = _rms(list(base["y"].values())), _rms(list(kicked["y"].values()))
    assert after >= KICK_RMS_FACTOR_MIN * before, (
        f"{shape.KICK['corrector']} raised by {value_a} A: the first turn's vertical rms went "
        f"from {before:.3e} m to {after:.3e} m, under {KICK_RMS_FACTOR_MIN}x"
    )
    assert not kicked["lost"], f"the kicked bunch is lost by {kicked['lost'][:5]}"


def test_an_applied_step_reduces_the_kicked_trajectory(deployed_stack: DeployedStack) -> None:
    kicked = deployed_stack.kicked_shot
    assert kicked is not None, "the kick was not taken"
    try:
        status, step = _request(f"{PANEL_PREFIX}/step", headers=_held(deployed_stack))
        assert status == 200, f"no step after the kicked shot: {status} {step!r}"
        assert step["shot_id"] == kicked["shot_id"], (step["shot_id"], kicked["shot_id"])
        assert not step["all_zero"], f"the step for the kick moves nothing: {step['reason']}"
        status, done = _request(
            f"{PANEL_PREFIX}/apply",
            "POST",
            body={"step_digest": step["digest"]},
            headers=_held(deployed_stack),
            timeout=ACTION_HTTP_TIMEOUT_SEC,
        )
        assert status == 200, f"apply refused: {status} {done!r}\n{_logs()}"
        assert not done["stopped"], done
        assert done["moved"], "the apply moved no corrector"
        deployed_stack.applied_digests.append(step["digest"])
        corrected = _measure(deployed_stack)
        deployed_stack.corrected_shot = corrected
        before, after = _rms(list(kicked["y"].values())), _rms(list(corrected["y"].values()))
        assert after < before, (
            f"the applied step left the first turn's vertical rms at {after:.3e} m, "
            f"not below the kicked {before:.3e} m"
        )
    finally:
        if deployed_stack.kick_origin is not None:
            _standin_write(deployed_stack, shape.KICK["record"], deployed_stack.kick_origin)
        _release_held(deployed_stack)


def test_the_apply_audit_line_names_the_admitted_owner(deployed_stack: DeployedStack) -> None:
    from osprey.audit.envelope import DECISION_ALLOWED, SURFACE_TRAJECTORY_SERVICE

    assert deployed_stack.applied_digests, "no step was applied"
    applies = [
        r
        for r in _audit_records(deployed_stack.repo)
        if r.get("surface") == SURFACE_TRAJECTORY_SERVICE
        and r.get("subject") == "apply"
        and r.get("decision") == DECISION_ALLOWED
    ]
    for digest in deployed_stack.applied_digests:
        lines = [r for r in applies if f"step_digest={digest}" in r.get("detail", "")]
        assert len(lines) == 1, f"apply of {digest}: {len(lines)} allowed records in {applies!r}"
        assert lines[0].get("actor") == deployed_stack.owner, lines[0]


def test_the_shot_lands_on_the_host_and_in_the_listing(deployed_stack: DeployedStack) -> None:
    shot = deployed_stack.corrected_shot
    assert shot is not None, "the corrected shot was not taken"
    path = deployed_stack.repo / "var" / "trajectory" / "shots" / MODEL / f"{shot['shot_id']}.json"
    assert path.is_file(), f"no shot file at {path}"
    assert json.loads(path.read_text(encoding="utf-8"))["shot_id"] == shot["shot_id"]

    status, listing = _request(
        f"{PANEL_PREFIX}/shots", headers=deployed_stack.headers(PANEL_PREFIX)
    )
    assert status == 200, f"GET shots: {status} {listing!r}"
    assert listing["model"] == MODEL, listing["model"]
    assert listing["shots"][0]["shot_id"] == shot["shot_id"], (
        "the newest listed shot is not the corrected one: "
        f"{[s['shot_id'] for s in listing['shots']]}"
    )


# ---------------------------------------------------------------------------
# 6. The panel smoke through alice's terminal: her browser, her agent's server,
#    the terminal's own refusals (STRICT, ordered: the browser's session spans them)
# ---------------------------------------------------------------------------


def _exec(*argv: str, timeout: float = 60) -> subprocess.CompletedProcess:
    """Run one command inside ``alice``'s terminal container, as the runtime uid."""
    return subprocess.run(
        ["docker", "exec", "-i", "--user", "1000", WEB_CONTAINER, *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _agent_pid() -> int | None:
    """The PID of the agent the terminal launched: the one process carrying the panel token."""
    found = _exec(
        "sh",
        "-c",
        "for d in /proc/[0-9]*; do "
        "if grep -qz '^OSPREY_PANEL_TOKEN=' $d/environ 2>/dev/null; then basename $d; fi; done",
    )
    pids = found.stdout.split() if found.returncode == 0 else []
    return int(pids[0]) if pids else None


def _process_env(pid: int) -> dict[str, str]:
    """The environment of *pid* inside the container, read off ``/proc``."""
    result = subprocess.run(
        ["docker", "exec", "--user", "1000", WEB_CONTAINER, "cat", f"/proc/{pid}/environ"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, f"could not read /proc/{pid}/environ: {result.stderr!r}"
    return dict(item.split("=", 1) for item in result.stdout.decode().split("\0") if "=" in item)


def _agent_env(stack: DeployedStack) -> dict[str, str]:
    """The environment of ``alice``'s agent, started by a terminal attach if none runs yet."""
    if stack.agent_env is not None:
        return stack.agent_env
    from websockets.sync.client import connect

    pid = _agent_pid()
    if pid is None:
        with connect(
            f"{WS_URL}/ws/terminal?mode=new",
            additional_headers={SECRET_HEADER: stack.secret},
            max_size=None,
            open_timeout=60,
        ) as socket:
            session_id = None
            deadline = time.monotonic() + 60
            while session_id is None and time.monotonic() < deadline:
                raw = socket.recv(timeout=max(1.0, deadline - time.monotonic()))
                if isinstance(raw, str) and raw.startswith("{"):
                    frame = json.loads(raw)
                    if frame.get("type") == "session_info":
                        session_id = frame["session_id"]
        assert session_id, f"the terminal never sent session_info\n{_terminal_logs()}"
        deadline = time.monotonic() + AGENT_START_TIMEOUT_SEC
        while (pid := _agent_pid()) is None:
            assert time.monotonic() < deadline, (
                f"no process in {WEB_CONTAINER} carries OSPREY_PANEL_TOKEN within "
                f"{AGENT_START_TIMEOUT_SEC:.0f}s of the attach\n{_terminal_logs()}"
            )
            time.sleep(2.0)
    stack.agent_env = _process_env(pid)
    assert stack.agent_env.get("OSPREY_PANEL_TOKEN"), "the agent carries an empty panel token"
    return stack.agent_env


class _OperatorBrowser:
    """One chromium tab of ``alice``'s, logged in through nginx."""

    def __init__(self, secret: str) -> None:
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch()
        self.context = self._browser.new_context()
        self.requests: list[str] = []
        self.context.on("request", lambda request: self.requests.append(request.url))
        self.page = self.context.new_page()
        token = urllib.parse.quote(secret, safe="")
        self.page.goto(f"{ORIGIN}/u/{USER}/?token={token}", wait_until="load")

    def close(self) -> None:
        for closing in (self.context.close, self._browser.close, self._playwright.stop):
            # Teardown only: a tab that is already closed is no finding.
            with contextlib.suppress(Exception):
                closing()


def _close_panel(stack: DeployedStack) -> None:
    if stack.panel_page is not None:
        stack.panel_page.close()
        stack.panel_page = None


def _panel(stack: DeployedStack) -> Any:
    """The operator's panel page, opened by the panel smoke."""
    assert stack.panel_page is not None, "the operator's panel was not opened"
    return stack.panel_page.page


def _session_control(page: Any, name: str) -> Any:
    """A session button where the operator presses it: the panel body's copy."""
    return page.locator(f'[data-view=session] [data-control="{name}"]')


def _holder(page: Any) -> Any:
    return page.locator("[data-fact=holder] .fact-value")


def test_the_operator_runs_a_shot_through_the_panel(deployed_stack: DeployedStack) -> None:
    """Logged in with her login link, ``alice`` opens a session in the panel her
    terminal proxies and runs the selected chain; its raw plots draw.

    The cookie the link exchanges for is the only credential the tab holds, and
    every request the bundle sends goes to the panel door below it.
    """
    from playwright.sync_api import expect

    # A physics test that failed part-way may have left its session held.
    _release_held(deployed_stack)
    operator = _OperatorBrowser(deployed_stack.secret)
    deployed_stack.panel_page = operator
    assert operator.context.cookies(), f"the login link issued no cookie\n{_terminal_logs()}"
    page = operator.page
    page.goto(f"{ORIGIN}/u/{USER}{TERMINAL_PANEL}", wait_until="load")
    expect(page.locator("[data-connection=connected]")).to_be_visible(timeout=SETTLE_MS)

    _session_control(page, "session-open").click()
    expect(_session_control(page, "session-end")).to_be_enabled(timeout=SETTLE_MS)
    expect(_holder(page)).to_have_text(f"panel of {USER} (this tab)", timeout=SETTLE_MS)
    run = page.locator('[data-control="run-selected"]')
    expect(run).to_be_enabled(timeout=SETTLE_MS)
    with page.expect_response(
        lambda r: r.url.endswith("/api/shot") and r.request.method == "POST",
        timeout=ACTION_HTTP_TIMEOUT_SEC * 1000,
    ) as answered:
        run.click()
    assert answered.value.ok, answered.value.text()
    shot_id = answered.value.json()["shot"]["shot_id"]
    for plot_id in RAW_PLOTS:
        expect(page.locator(f"#{plot_id} .main-svg").first).to_be_attached(timeout=SETTLE_MS)

    status, state = _request(f"{PANEL_PREFIX}/state", headers=deployed_stack.headers(PANEL_PREFIX))
    assert status == 200, f"GET state: {status} {state!r}"
    assert state["holder"]["owner"] == USER and state["holder"]["kind"] == "panel", state
    assert state["step"] and state["step"]["shot_id"] == shot_id, state["step"]
    deployed_stack.panel_digest = state["step"]["digest"]
    paths = [urllib.parse.urlsplit(url).path for url in operator.requests]
    assert any(path.startswith(f"/u/{USER}{TERMINAL_PANEL_DOOR}/") for path in paths), paths
    assert not [path for path in paths if path.startswith(f"/u/{USER}{TERMINAL_TOKEN_DOOR}/")], (
        f"the panel reached the agent's token door: {paths}"
    )


#: A stdio MCP client run INSIDE ``alice``'s terminal container.
#:
#: It starts the ``trajectory`` server exactly as the agent's ``.mcp.json``
#: renders it, in the agent's own environment (so with the agent's panel token
#: and the terminal's port), and speaks newline-delimited JSON-RPC to it:
#: initialize, the initialized notification, then one call of each tool. No
#: approval hook runs in front of it, so the shot carries no ask stamp. The
#: answer is one JSON line; it never carries the agent's environment.
_MCP_CLIENT = r"""
import glob, json, os, queue, subprocess, sys, threading, time

call_timeout = float(sys.argv[1])


def environ(pid):
    with open("/proc/%s/environ" % pid, "rb") as handle:
        raw = handle.read().decode("utf-8", "replace")
    return dict(item.split("=", 1) for item in raw.split("\0") if "=" in item)


agent, pid = None, None
for d in sorted(glob.glob("/proc/[0-9]*")):
    try:
        env = environ(os.path.basename(d))
    except OSError:
        continue
    if env.get("OSPREY_PANEL_TOKEN"):
        agent, pid = env, os.path.basename(d)
        break
out = {}
if agent is None:
    print(json.dumps({"error": "no process carries OSPREY_PANEL_TOKEN"}))
    sys.exit(0)
try:
    cwd = os.readlink("/proc/%s/cwd" % pid)
except OSError:
    cwd = None
candidates = []
if agent.get("OSPREY_CONFIG"):
    candidates.append(os.path.join(os.path.dirname(agent["OSPREY_CONFIG"]), ".mcp.json"))
if cwd:
    candidates.append(os.path.join(cwd, ".mcp.json"))
mcp_path = next((c for c in candidates if os.path.isfile(c)), None)
out["mcp_json"], out["candidates"] = mcp_path, candidates
if mcp_path is None:
    print(json.dumps(out))
    sys.exit(0)
with open(mcp_path) as handle:
    servers = json.load(handle)["mcpServers"]
out["servers"] = sorted(servers)
entry = servers.get("trajectory")
if entry is None:
    print(json.dumps(out))
    sys.exit(0)
out["command"] = [entry["command"], *entry.get("args", [])]
env = {**agent, **{k: str(v) for k, v in (entry.get("env") or {}).items()}}
proc = subprocess.Popen(
    out["command"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env,
    cwd=cwd or os.path.dirname(mcp_path),
    text=True,
    bufsize=1,
)
lines, err = queue.Queue(), []
threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout], daemon=True).start()
threading.Thread(target=lambda: [err.append(l) for l in proc.stderr], daemon=True).start()


def send(message):
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def answer(request_id, timeout):
    deadline = time.monotonic() + timeout
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            return {"timeout": timeout}
        try:
            line = lines.get(timeout=left)
        except queue.Empty:
            return {"timeout": timeout}
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get("id") == request_id:
            return message


def call(request_id, name, arguments):
    send({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
          "params": {"name": name, "arguments": arguments}})
    return answer(request_id, call_timeout)


send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {},
    "clientInfo": {"name": "osprey-e2e", "version": "0"}}})
out["initialize"] = answer(1, 120)
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
out["step"] = call(2, "trajectory_step", {})
out["shot"] = call(3, "trajectory_shot", {"n_shots": 1})
proc.stdin.close()
try:
    proc.wait(timeout=60)
except subprocess.TimeoutExpired:
    proc.kill()
out["stderr"] = "".join(err)[-4000:]
print(json.dumps(out))
"""


def _tool_payload(answer: dict[str, Any]) -> tuple[bool, Any]:
    """A ``tools/call`` answer's error flag and its JSON text content."""
    result = answer.get("result")
    assert isinstance(result, dict), f"the call was not answered with a result: {answer!r}"
    text = next(c["text"] for c in result["content"] if c.get("type") == "text")
    try:
        return bool(result.get("isError")), json.loads(text)
    except ValueError:
        return bool(result.get("isError")), text


def test_the_agents_server_reads_the_step_and_refuses_an_unstamped_shot(
    deployed_stack: DeployedStack,
) -> None:
    """The rendered server works through the token door and refuses without a stamp.

    ``trajectory_step`` reads the step the operator's shot left, while her tab
    still holds the session; ``trajectory_shot`` is put to a human by the
    preset's approval policy, and with no hook to ask one there is no stamp, so
    the server refuses ``approval_missing`` before it opens a session -- which
    is why the operator's session does not turn the call into ``session_held``.
    """
    assert deployed_stack.panel_digest, "the operator's shot left no step"
    _agent_env(deployed_stack)
    timeout = ACTION_HTTP_TIMEOUT_SEC
    ran = subprocess.run(
        ["docker", "exec", "-i", "--user", "1000", WEB_CONTAINER, "python3", "-c", _MCP_CLIENT]
        + [str(timeout)],
        capture_output=True,
        text=True,
        timeout=3 * timeout + 300,
    )
    assert ran.returncode == 0, (
        f"the in-container MCP client failed (rc={ran.returncode}):\n"
        f"--- stdout ---\n{ran.stdout}\n--- stderr ---\n{ran.stderr}"
    )
    out = json.loads(ran.stdout.strip().splitlines()[-1])
    assert "trajectory" in out.get("servers", []), (
        f"the agent's .mcp.json renders no trajectory server: {out!r}"
    )
    assert "result" in out["initialize"], f"the server did not initialize: {out!r}"

    failed, step = _tool_payload(out["step"])
    assert not failed, f"trajectory_step refused: {step!r}\n{out['stderr']}\n{_terminal_logs()}"
    assert step["digest"] == deployed_stack.panel_digest, (
        f"the server read step {step['digest']!r}, the panel's shot left {deployed_stack.panel_digest!r}"
    )

    failed, refusal = _tool_payload(out["shot"])
    assert failed, f"an unstamped trajectory_shot ran: {refusal!r}"
    assert isinstance(refusal, dict) and refusal.get("error_type") == "approval_missing", refusal

    status, state = _request(f"{PANEL_PREFIX}/state", headers=deployed_stack.headers(PANEL_PREFIX))
    assert status == 200, f"GET state: {status} {state!r}"
    assert state["holder"]["kind"] == "panel" and state["holder"]["owner"] == USER, (
        f"the refused shot moved the session: {state!r}"
    )
    assert state["step"] and state["step"]["digest"] == deployed_stack.panel_digest, state


def test_the_operator_ends_her_session(deployed_stack: DeployedStack) -> None:
    from playwright.sync_api import expect

    page = _panel(deployed_stack)
    _session_control(page, "session-end").click()
    expect(_holder(page)).to_have_text("nobody", timeout=SETTLE_MS)
    expect(_session_control(page, "session-open")).to_be_enabled(timeout=SETTLE_MS)
    _close_panel(deployed_stack)

    status, state = _request(f"{PANEL_PREFIX}/state", headers=deployed_stack.headers(PANEL_PREFIX))
    assert status == 200 and state["holder"] is None, f"the release left a holder: {state!r}"


def test_the_terminal_refuses_the_token_door_without_its_secret_and_the_panel_door_to_the_token(
    deployed_stack: DeployedStack,
) -> None:
    """At ``alice``'s terminal: the agent's token door admits her panel token (the
    control), and refuses a request with no credential or a wrong terminal
    secret; the operator's panel door and bundle refuse the panel token."""
    token = _agent_env(deployed_stack)["OSPREY_PANEL_TOKEN"]
    bearer = {"Authorization": f"Bearer {token}"}

    status, body = _request(f"{TERMINAL_TOKEN_DOOR}/state", headers=bearer, base=WEB_URL)
    assert status == 200 and body.get("caller") == USER, (
        f"the token door refused the agent's own panel token: {status} {body!r}\n{_terminal_logs()}"
    )

    for headers in ({}, {SECRET_HEADER: "not-" + deployed_stack.secret}):
        status, body = _request(f"{TERMINAL_TOKEN_DOOR}/state", headers=headers, base=WEB_URL)
        assert status == 401, f"token door with {sorted(headers)}: {status} {body!r}"

    for path in (f"{TERMINAL_PANEL_DOOR}/state", TERMINAL_PANEL):
        status, body = _request(path, headers=bearer, base=WEB_URL)
        assert status == 401, f"GET {path} with only the panel token: {status} {body!r}"
    status, body = _request(
        f"{TERMINAL_PANEL_DOOR}/session",
        "POST",
        headers={**bearer, "Origin": ORIGIN},
        base=WEB_URL,
    )
    assert status == 401, f"POST panel-door session with only the panel token: {status} {body!r}"
    status, state = _request(f"{PANEL_PREFIX}/state", headers=deployed_stack.headers(PANEL_PREFIX))
    assert status == 200 and state["holder"] is None, f"a refused open left a holder: {state!r}"


# ---------------------------------------------------------------------------
# 7. A continuous run ends by its lease (STRICT: it holds its own session)
# ---------------------------------------------------------------------------


class _EventTap:
    """The panel's state stream, read on a thread into a list of events."""

    def __init__(self, stack: DeployedStack) -> None:
        req = urllib.request.Request(
            f"{TRAJECTORY_URL}{PANEL_PREFIX}/events", headers=stack.headers(PANEL_PREFIX)
        )
        self._resp = urllib.request.urlopen(req, timeout=ACTION_HTTP_TIMEOUT_SEC)  # localhost
        self.events: list[dict[str, Any]] = []
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            for raw in self._resp:
                line = raw.decode("utf-8").strip()
                if line.startswith("data: "):
                    self.events.append(json.loads(line[len("data: ") :]))
        except Exception:  # the stream closes under the reader
            return

    def ended(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == "continuous_ended"]

    def close(self) -> None:
        self._resp.close()


def test_a_continuous_run_ends_when_its_heartbeats_stop(deployed_stack: DeployedStack) -> None:
    grant = _open(deployed_stack, PANEL_PREFIX)
    deployed_stack.client_id = grant["client_id"]
    tap = _EventTap(deployed_stack)
    try:
        status, run = _request(
            f"{PANEL_PREFIX}/continuous",
            "POST",
            body={"steps": CONTINUOUS_STEPS, "n_shots": 1},
            headers=_held(deployed_stack),
        )
        assert status == 200, f"continuous refused: {status} {run!r}\n{_logs()}"

        alive_until = time.monotonic() + HEARTBEATS_FOR_SEC
        while time.monotonic() < alive_until:
            # The service renews the lease no earlier than the beat was sent,
            # so the lease cannot lapse sooner than one lease after this.
            last_beat = time.monotonic()
            status, beat = _request(
                f"{PANEL_PREFIX}/continuous/heartbeat", "POST", headers=_held(deployed_stack)
            )
            assert status == 200, f"heartbeat: {status} {beat!r}"
            assert beat["continuous"] is not None, (
                f"the run ended while heartbeats moved its lease: {tap.ended()!r}\n{_logs()}"
            )
            time.sleep(HEARTBEAT_SEC / 2)
        assert not tap.ended(), f"the run ended while heartbeats moved its lease: {tap.ended()}"

        stopped_beating = last_beat
        deadline = stopped_beating + LEASE_SEC + ACTION_HTTP_TIMEOUT_SEC
        while True:
            status, state = _request(
                f"{PANEL_PREFIX}/state", headers=deployed_stack.headers(PANEL_PREFIX)
            )
            assert status == 200, f"GET state: {status} {state!r}"
            if state["continuous"] is None:
                break
            assert time.monotonic() < deadline, (
                f"the run outlived its lease by {time.monotonic() - stopped_beating:.0f}s"
            )
            time.sleep(0.5)
        lasted = time.monotonic() - stopped_beating
        assert lasted >= LEASE_SEC - 1.0, (
            f"the run ended {lasted:.1f}s after the last heartbeat, before its "
            f"{LEASE_SEC:g}s lease could lapse: {tap.ended()!r}\n{_logs()}"
        )

        settle = time.monotonic() + 10.0
        while not tap.ended() and time.monotonic() < settle:
            time.sleep(0.2)
        ended = tap.ended()
        assert [e["end"] for e in ended] == ["lease_expired"], ended
        status, beat = _request(
            f"{PANEL_PREFIX}/continuous/heartbeat", "POST", headers=_held(deployed_stack)
        )
        assert status == 200 and beat["continuous"] is None, (status, beat)
    finally:
        tap.close()
        _release_held(deployed_stack)


# ---------------------------------------------------------------------------
# 8. The reference survives a redeploy (LAST: it takes the stack down and up)
# ---------------------------------------------------------------------------


def test_the_selected_reference_survives_a_redeploy(deployed_stack: DeployedStack) -> None:
    shot = deployed_stack.corrected_shot
    assert shot is not None, "the corrected shot was not taken"
    grant = _open(deployed_stack, PANEL_PREFIX)
    deployed_stack.client_id = grant["client_id"]
    status, saved = _request(
        f"{PANEL_PREFIX}/references",
        "POST",
        body={"name": REFERENCE_NAME, "shot_id": shot["shot_id"]},
        headers=_held(deployed_stack),
    )
    assert status == 200, f"save reference: {status} {saved!r}"
    assert saved["shot_id"] == shot["shot_id"] and saved["sha256"], saved
    status, selected = _request(
        f"{PANEL_PREFIX}/references/select",
        "POST",
        body={"name": REFERENCE_NAME},
        headers=_held(deployed_stack),
        timeout=ACTION_HTTP_TIMEOUT_SEC,
    )
    assert status == 200, f"select reference: {status} {selected!r}"

    repo = deployed_stack.repo
    down = _run([str(deployed_stack.osprey_bin), "down"], cwd=repo, timeout=300)
    assert down.returncode == 0, f"osprey down failed:\n{down.stdout}\n{down.stderr}"
    remove_project_volumes(PROJECT)
    deployed_stack.client_id = None
    up = _run(
        [str(deployed_stack.osprey_bin), "up", "-d", "--dev"],
        cwd=repo,
        timeout=DEPLOY_UP_TIMEOUT_SEC,
    )
    assert up.returncode == 0, f"osprey up after down failed:\n{up.stdout}\n{up.stderr}\n{_logs()}"
    _orm_stack.wait_for_health(f"{TRAJECTORY_URL}/health", HEALTH_TIMEOUT_SEC)
    deployed_stack.secret = _user_secret(repo)

    status, refs = _request(
        f"{PANEL_PREFIX}/references", headers=deployed_stack.headers(PANEL_PREFIX)
    )
    assert status == 200, f"GET references after the redeploy: {status} {refs!r}\n{_logs()}"
    assert REFERENCE_NAME in refs["names"], refs
    assert refs["current"] == REFERENCE_NAME, refs
