"""Full-stack Docker proof of the deployed trajectory service, its two doors and its panel.

A multi-user control-assistant deployment with the virtual accelerator served
is built and started with ``osprey up``: one roster user, ``alice``, on one
persona that lists the ``trajectory`` panel, so the build renders her terminal
the panel tab and her agent the ``trajectory`` MCP server. The trajectory
container is then driven over raw HTTP exactly as its two callers reach it:

* the AGENT door (``/api/*``), which also demands the service's launch token;
* the PANEL door (``/panel/api/*``), the operator panel's.

Both doors sit behind the shared web gate, which admits a request by the
operator secret it carries and stamps the owner from the credential that
matched -- never from anything the request claims. In the multi-user shape the
service accepts each entitled roster user's minted terminal secret and names
that user, so every request this module sends with ``alice``'s secret is
owned by ``alice``.

What is pinned here and nowhere else: the deployed ``/health`` reaching its
target; the agent door's 403 ``launch_token_invalid`` on a wrong token; on both
doors, a request without the secret refused, a body naming an owner refused
400 ``owner_in_body``, a second holder refused 409 ``session_held``, and the
admitted owner read back through ``GET state``; and an action's audit record
landing under ``var/audit/trajectory/`` on the host.

Then the hop through ``alice``'s terminal, as the operator and the agent use
it: a real browser logs in through nginx with her login link, opens
``/panel/trajectory/panel/``, reads the live state, opens a session and takes a
shot through the panel door; an in-container client in her terminal starts
the rendered ``trajectory`` MCP server over stdio and reads that shot's step
with ``trajectory_step``, while ``trajectory_shot`` -- called with no approval
hook in front of it, so with no ask stamp -- is refused ``approval_missing``
before it opens a session; the browser then sends Stop while a shot is in
flight and releases. At the terminal, the agent's token door with no
terminal secret, and the panel door with nothing but the panel token, are
refused.

Then the deployed stack threads an errored ring end to end, in ORDER, one
stateful test building on the last: ``osprey sim apply first-turn-errors
--no-seed`` loads the shipped misalignment seed into the served virtual
accelerator; the first shot, two shots averaged so the served buffer's
timestamps must advance between them, is lost by BPM 36 on both; at most five
applied steps bring the first turn around every BPM; the apply's audit line
names the admitted owner; the shot lands as a file on the host and in the
panel's listing; a continuous run whose heartbeats stop ends by its lease; and
a selected reference survives ``osprey down``, removal of the project's named
volumes and a fresh ``osprey up`` (last, since it redeploys the stack).

The limits are the module's own and bound the correctors only: every vertical
corrector stays below the current at which the cold solve loses the closed
orbit, and no record names a command channel, so the preset's ``optional``
mode lets the shot's commands through while every corrector write is checked.

The VA image is built for x86_64. On a Docker host of another architecture it
runs emulated, far slower than natively, so there the deployment is given
service and connector timeouts long enough for an emulated shot; a native host
deploys with the shipped defaults.

Reuses ``tests/e2e/_orm_stack.py`` (the VA-backed control-assistant deploy
config: virtual-accelerator control system; dispatch, record archive and
trajectory service dropped). The lane turns web terminals back on with token
logins, rewrites the roster to its one user and persona after ``osprey init``,
and states the preset's ``trajectory:`` block back, on the host network, so
the trajectory service is the one service it deploys on the project image. The
terminal's agent is started so its panel token exists; the ``.env`` carries a
placeholder model key for it, and nothing contacts a model.

Per-function flaky, NOT module-level: the health/session checks accept one
rerun on a timing assertion; the security negatives (wrong token, no secret,
owner in the body, the terminal's door refusals, the unstamped shot) stay
STRICT with no flaky marker, so a rerun can never sweep a refusal that failed
to happen into a pass. The panel hop and the threading sequence are STRICT
too: each of their tests acts on the state the previous one left, so a rerun
would replay against a different session or ring.

CONTAINER SAFETY: every docker/compose invocation names an EXACT resource
(images ``<project>:local``, ``<project>-va:local``,
``<project>-bluesky-bridge:local``, the persona image
``<repo>-operator:local``; containers ``<project>-web-alice`` and
``<project>-nginx``) -- never ``system prune``, never ``--volumes``, never a
wildcard ``docker rm``/``rmi``. Teardown goes through ``osprey down``, then
exact-named removal of the terminal tier's containers and of this project's
own volumes (``tests/e2e/_volumes.py``).

Gating: needs Docker and Playwright's chromium. Lives in ``tests/e2e/`` (never
collected by the fast lane) and is ``--ignore``'d by the shared e2e lane; runs
in its own ``trajectory-deploy-e2e`` CI job (same-repo PRs only, no secrets --
neither the service nor the terminal's agent reaches a model), which treats
any skip as a failure. Locally, only one e2e stack may run per host at a time.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.e2e import _orm_stack
from tests.e2e._deploy_diagnostics import container_logs
from tests.e2e._volumes import remove_project_volumes

# Distinct from every sibling e2e module's pinned bridge port (18090, 18099,
# 18101-18110): `init_args` always serves the bluesky bridge, so it needs one.
BRIDGE_PORT = 18111

# This module's own thousand-port block: every companion port not pinned
# explicitly follows it instead of colliding with another e2e stack or a real
# deployment's default block.
PORT_BASE = 22200

# ariel-postgres, the telemetry gateway and the telemetry store are deployed by
# the preset with no profile knob to drop them; all three move to high,
# unassigned host ports so a locally-running deployment holding the defaults
# cannot abort `osprey up`.
POSTGRES_PORT = 25437
TELEMETRY_GATEWAY_PORT = 24331
TELEMETRY_STORE_PORT = 28136

PROJECT_NAME = "trajectory-proj"
PROJECT = _orm_stack.project_prefix(PROJECT_NAME)
TRAJECTORY_CONTAINER = f"{PROJECT}-trajectory"
VA_CONTAINER = f"{PROJECT}-virtual-accelerator"

#: The one roster user, and the persona she is on: a delta listing the
#: ``trajectory`` panel and nothing else, so the build renders her terminal
#: the panel tab and her agent the ``trajectory`` MCP server.
USER = "alice"
PERSONA = "operator"
#: ``osprey build`` renders a persona delta to ``build/<repo>-<persona>``, and
#: the persona image is tagged by that same name.
PERSONA_PROJECT = f"{PROJECT_NAME}-{PERSONA}"
WEB_CONTAINER = f"{PROJECT}-web-{USER}"
NGINX_CONTAINER = f"{PROJECT}-nginx"

# The trajectory service runs the PROJECT image; the VA, the bridge and the
# persona build their own. Exact tags, derived as the compose templates derive them.
PROJECT_IMAGE = f"{PROJECT}:local"
VA_IMAGE = _orm_stack.va_image(PROJECT_NAME)
BRIDGE_IMAGE = _orm_stack.bridge_image(PROJECT_NAME)
PERSONA_IMAGE = f"{PERSONA_PROJECT}:local"

#: A model key the terminal's agent starts with. The agent is started only so
#: the panel token it carries exists; nothing in the module contacts a model.
PLACEHOLDER_MODEL_KEY_LINE = "ANTHROPIC_API_KEY=placeholder-model-key\n"

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
# The first-time native VA build, the project image and the persona image are
# slow on a cold cache.
DEPLOY_UP_TIMEOUT_SEC = 2400
# How long the container may take to answer /health at all ...
HEALTH_TIMEOUT_SEC = 300.0
# ... and, once it answers, how long until it reaches its control target.
REACHABLE_WITHIN_SEC = 30.0

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

#: The errored ring the stack threads, and the seed it ships with.
SCENARIO = "first-turn-errors"
SHIPPED_SEED = 11
#: The BPM (1-based, in beam order from injection) the first shot is lost by at the latest.
LATEST_LOSS_BPM = 36
#: The steps applied after which every BPM must see the first turn.
MAX_STEPS = 5
#: Shots per acquisition: more than one, so a shot only completes when the
#: served buffer's timestamps advance between its members.
N_SHOTS = 2
#: The model the trajectory service steers.
MODEL = "SR"
#: Every vertical corrector's setpoint; the bound keeps each one below the
#: ~11.5 A at which the cold solve loses the closed orbit.
VCM_SETPOINT = re.compile(r"SR:MAG:VCM:\d+:CURRENT:SP")
VCM_BOUND_A = 11.0
#: The VA container's view of the deployment's active scenarios.
VA_ACTIVE_SCENARIOS = "/state/simulation/active_scenarios"
#: How long an applied scenario may take to show inside the container.
SCENARIO_VISIBLE_SEC = 30.0
#: How long one shot, step or apply may take over HTTP: the service's own
#: action timeout, with room for the queue in front of it.
ACTION_HTTP_TIMEOUT_SEC = 180.0

#: The Docker architectures the VA image runs on natively. Its recipe pins
#: ``linux/amd64``, so any other Docker host runs it under emulation.
VA_NATIVE_ARCHITECTURES = frozenset({"x86_64", "amd64"})
#: Under emulation one shot's pass holds the VA for minutes once the ring
#: keeps a closed orbit, and its Channel Access server answers nothing
#: meanwhile. The service then waits that long for the armed flags and for
#: each read, and an action of ``N_SHOTS`` such shots fits its action budget.
EMULATED_ARMED_TIMEOUT_SEC = 400.0
EMULATED_CALL_TIMEOUT_SEC = 90.0
EMULATED_ACTION_TIMEOUT_SEC = 3000.0
#: The continuous run's lease and heartbeat (``continuous.DEFAULT_LEASE_S`` /
#: ``DEFAULT_HEARTBEAT_S``), and how long heartbeats keep it alive -- past
#: two leases, so the run outliving a lease proves the heartbeats move it.
LEASE_SEC = 10.0
HEARTBEAT_SEC = 5.0
HEARTBEATS_FOR_SEC = 2 * LEASE_SEC + HEARTBEAT_SEC
#: Shots a continuous run may take: far more than the heartbeat window holds,
#: so only its lease can end it.
CONTINUOUS_MAX_SHOTS = 10_000
REFERENCE_NAME = "threaded"

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


@functools.cache
def _va_emulated() -> bool:
    """Whether this Docker host runs the VA image under emulation."""
    result = subprocess.run(
        ["docker", "info", "--format", "{{.Architecture}}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0 and result.stdout.strip() not in VA_NATIVE_ARCHITECTURES


def _timing_config() -> dict[str, float]:
    """The service and connector timeouts the VA's pace on this host needs.

    A natively served VA answers within the shipped defaults, so nothing is
    set; an emulated one gets the emulated timeouts.
    """
    if not _va_emulated():
        return {}
    return {
        "services.trajectory.armed_timeout_s": EMULATED_ARMED_TIMEOUT_SEC,
        "services.trajectory.action_timeout_s": EMULATED_ACTION_TIMEOUT_SEC,
        "control_system.connector.virtual_accelerator.timeout_s": EMULATED_CALL_TIMEOUT_SEC,
    }


def _action_http_timeout() -> float:
    """How long one action may take over HTTP on this host."""
    if not _va_emulated():
        return ACTION_HTTP_TIMEOUT_SEC
    return EMULATED_ACTION_TIMEOUT_SEC + ACTION_HTTP_TIMEOUT_SEC


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
        # What the ordered threading tests hand on, one to the next.
        self.client_id: str | None = None
        self.lost_per_shot: list[int | None] = []
        self.applied_digests: list[str] = []
        self.threaded_shot: dict[str, Any] | None = None
        self.reference_saved = False

    def headers(self, prefix: str, **extra: str) -> dict[str, str]:
        """The credentials a door's legitimate caller carries."""
        headers = {SECRET_HEADER: self.secret, **extra}
        if prefix == AGENT_PREFIX:
            headers["X-Launch-Token"] = self.launch_token
        return headers


def _logs() -> str:
    return container_logs(TRAJECTORY_CONTAINER)


def _terminal_logs() -> str:
    return container_logs(WEB_CONTAINER)


def _user_secret(repo: Path) -> str:
    """``alice``'s terminal secret, as ``osprey up`` minted it into ``.env``."""
    from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var

    return _minted(repo, terminal_secret_var(USER))


def _shape_repo(repo: Path) -> None:
    """One roster user on one persona delta, after ``osprey init`` materialized the preset's.

    ``osprey init`` renders one delta per preset persona and the preset's whole
    roster; the profile is rewritten here to the shape this lane deploys. The
    delta arms writes on the virtual accelerator alone, as the preset's
    read-write persona does, lists the ``trajectory`` panel and no other
    write-oriented one (the lane deploys no event dispatcher), and declares no
    ``extends`` -- a file in ``personas/`` inherits from the profile beside it.
    """
    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    terminals = profile["config"]["modules.web_terminals"]
    terminals["users"] = [{"name": USER, "index": 0, "persona": PERSONA}]
    terminals["personas"] = {PERSONA: {"build_profile": f"personas/{PERSONA}.yml"}}
    terminals["default_persona"] = PERSONA
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    personas = repo / "personas"
    for delta in personas.glob("*.yml"):
        delta.unlink()
    (personas / f"{PERSONA}.yml").write_text(
        f"name: {PROJECT_NAME} ({PERSONA})\n"
        "deploy_services: false\n"
        "web_panels:\n"
        "  - trajectory\n"
        "config:\n"
        "  control_system.writes_enabled: false\n"
        "  control_system.connector.epics.writes_enabled: false\n"
        "  control_system.connector.virtual_accelerator.writes_enabled: true\n"
        "  web.ui_mode: expert\n"
        "  modules.web_terminals.enabled: false\n",
        encoding="utf-8",
    )


def _remove_terminal_tier() -> None:
    """Exact-named removal of the terminal tier's containers; never an assertion."""
    for name in (WEB_CONTAINER, NGINX_CONTAINER):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=120)


def _author_corrector_limits(repo: Path) -> None:
    """Replace the repo's ``limits.yaml`` records with corrector-only ones.

    Written between ``osprey init`` and ``osprey build`` so the render
    enforces them. The shipped corrector records stay; every vertical
    corrector the facility wires gains a symmetric bound; nothing names a
    command channel, so the preset's ``optional`` mode lets commands pass.
    """
    facility = repo / "data" / "facility"
    limits_file = facility / "limits.yaml"
    limits = yaml.safe_load(limits_file.read_text(encoding="utf-8"))
    vcms = sorted(
        {
            match
            for path in facility.rglob("*.yaml")
            if path != limits_file
            for match in VCM_SETPOINT.findall(path.read_text(encoding="utf-8"))
        }
    )
    assert vcms, f"no vertical corrector setpoint found under {facility}"
    kept = [r for r in limits["records"] if ":MAG:" in r["address"] and r["address"] not in vcms]
    limits["records"] = kept + [
        {"address": vcm, "min_value": -VCM_BOUND_A, "max_value": VCM_BOUND_A} for vcm in vcms
    ]
    limits_file.write_text(yaml.safe_dump(limits, sort_keys=False), encoding="utf-8")


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedStack]:
    osprey_bin = _orm_stack.find_osprey_console_script()
    base = tmp_path_factory.mktemp("trajectory_build")
    repo = base / PROJECT_NAME

    args = _orm_stack.init_args(
        PROJECT_NAME,
        output_dir=base,
        bridge_port=BRIDGE_PORT,
        port_base=PORT_BASE,
        extra_config={
            "config": {
                "services.postgresql.port_host": POSTGRES_PORT,
                "services.telemetry_gateway.port": TELEMETRY_GATEWAY_PORT,
                "services.telemetry_store.port": TELEMETRY_STORE_PORT,
                # Web terminals back on, with token logins on loopback: the
                # roster and persona are rewritten after init (`_shape_repo`).
                "modules.web_terminals.enabled": True,
                "modules.web_terminals.image_source": "local",
                "modules.web_terminals.auth.method": "token",
                "modules.web_terminals.default_persona": PERSONA,
                "deploy.fqdn": "127.0.0.1",
                **_timing_config(),
            },
            "trajectory": {"network": "host"},
        },
    )
    init = _run([str(osprey_bin), "init", *args], cwd=base, timeout=BUILD_TIMEOUT_SEC)
    if init.returncode != 0:
        pytest.fail(
            f"osprey init failed (rc={init.returncode}):\n"
            f"--- stdout ---\n{init.stdout}\n--- stderr ---\n{init.stderr}"
        )
    _shape_repo(repo)
    _author_corrector_limits(repo)
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
    _orm_stack.seed_repo_env(repo)
    env_path = repo / ".env"
    with env_path.open("a", encoding="utf-8") as handle:
        handle.write(PLACEHOLDER_MODEL_KEY_LINE)
    os.chmod(env_path, 0o600)
    # Fresh --dev builds so the deployed containers run CURRENT source.
    _orm_stack.force_image_rebuild(PROJECT_IMAGE, VA_IMAGE, BRIDGE_IMAGE, PERSONA_IMAGE)

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


def _release(stack: DeployedStack, prefix: str, client_id: str) -> None:
    status, body = _request(
        f"{prefix}/session",
        "DELETE",
        headers=stack.headers(prefix, **{"X-Trajectory-Client": client_id}),
    )
    assert status == 200, f"release through {prefix} failed: {status} {body!r}"


def _open(stack: DeployedStack, prefix: str) -> dict[str, Any]:
    status, body = _request(f"{prefix}/session", "POST", headers=stack.headers(prefix))
    assert status == 200, f"open through {prefix} refused: {status} {body!r}\n{_logs()}"
    return body


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------


@pytest.mark.flaky(reruns=1, only_rerun=["AssertionError"])
@pytest.mark.usefixtures("deployed_stack")
def test_health_reaches_the_target_within_30s() -> None:
    """``/health`` is gate-exempt and names its target's reachability."""
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
    assert body.get("control_target"), f"/health names no control target: {body!r}"


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
# 5. The hop through alice's terminal: her browser, her agent's server, the
#    terminal's own refusals (STRICT, ordered: the browser's session spans them)
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
    """One chromium tab of ``alice``'s, logged in through nginx, on the trajectory panel."""

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
    """The operator's panel page, opened by the first panel-hop test."""
    assert stack.panel_page is not None, "the operator's panel was not opened"
    return stack.panel_page.page


def _fact(page: Any, view: str, name: str) -> Any:
    return page.locator(f"[data-view={view}] dt:text-is('{name}') + dd")


def _control(page: Any, name: str) -> Any:
    return page.locator(f"[data-control={name}]")


def test_the_operator_takes_a_shot_through_the_panel(deployed_stack: DeployedStack) -> None:
    """Logged in with her login link, ``alice`` reads the live state and shoots.

    The cookie the link exchanges for is the only credential the tab holds, and
    every request the bundle sends goes to the panel door below it.
    """
    from playwright.sync_api import expect

    operator = _OperatorBrowser(deployed_stack.secret)
    deployed_stack.panel_page = operator
    assert operator.context.cookies(), f"the login link issued no cookie\n{_terminal_logs()}"
    page = operator.page
    page.goto(f"{ORIGIN}/u/{USER}{TERMINAL_PANEL}", wait_until="load")
    expect(page.locator("[data-connection=connected]")).to_be_visible(timeout=SETTLE_MS)

    status, state = _request(f"{PANEL_PREFIX}/state", headers=deployed_stack.headers(PANEL_PREFIX))
    assert status == 200, f"GET state: {status} {state!r}"
    expect(_fact(page, "header", "Operator")).to_have_text(USER, timeout=SETTLE_MS)
    expect(_fact(page, "header", "Target")).to_have_text(str(state["target"]))
    expect(_fact(page, "header", "Model")).to_contain_text(MODEL)
    expect(_fact(page, "header", "Holder")).to_have_text("nobody")

    _control(page, "open").click()
    expect(_fact(page, "header", "Holder")).to_have_text(
        f"panel of {USER} (this tab)", timeout=SETTLE_MS
    )
    page.locator("input[name=n_shots]").fill(str(N_SHOTS))
    expect(_control(page, "shot")).to_be_enabled(timeout=SETTLE_MS)
    _control(page, "shot").click()
    digest = _fact(page, "step", "Digest")
    expect(digest).to_be_visible(timeout=_action_http_timeout() * 1000)
    expect(_fact(page, "status", "Included")).to_have_text(str(N_SHOTS))
    deployed_stack.panel_digest = digest.inner_text().strip()

    status, state = _request(f"{PANEL_PREFIX}/state", headers=deployed_stack.headers(PANEL_PREFIX))
    assert status == 200, f"GET state: {status} {state!r}"
    assert state["holder"]["owner"] == USER and state["holder"]["kind"] == "panel", state
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
    timeout = _action_http_timeout()
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
        f"the server read step {step['digest']!r}, the panel shows {deployed_stack.panel_digest!r}"
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


def _watch_until_running(watcher: Any, action: str) -> None:
    """Load the panel in *watcher* until its first snapshot shows *action* running.

    A tab's stream opens with the service's current snapshot, so a tab loaded
    while the action runs shows it running and offers Stop.
    """
    from playwright.sync_api import expect

    running = watcher.locator(f"p.running[data-running={action}]")
    deadline = time.monotonic() + SETTLE_MS / 1000
    while True:
        watcher.goto(f"{ORIGIN}/u/{USER}{TERMINAL_PANEL}", wait_until="load")
        expect(watcher.locator("[data-connection=connected]")).to_be_visible(timeout=SETTLE_MS)
        try:
            expect(running).to_be_visible(timeout=2_000)
            return
        except AssertionError:
            assert time.monotonic() < deadline, f"no tab ever saw the {action} running"


def test_the_operator_stops_a_shot_in_flight_and_releases(deployed_stack: DeployedStack) -> None:
    """Stop is offered to every admitted owner, holding or not: ``alice`` presses
    it in a second tab of hers, loaded while the shot runs, and the holding tab
    sees the shot end stopped, then releases."""
    from playwright.sync_api import expect

    page = _panel(deployed_stack)
    expect(_control(page, "shot")).to_be_enabled(timeout=SETTLE_MS)
    _control(page, "shot").click()
    watcher = deployed_stack.panel_page.context.new_page()
    _watch_until_running(watcher, "shot")
    expect(_fact(watcher, "header", "Holder")).to_have_text(f"panel of {USER}")
    _control(watcher, "stop").click()

    ended_ms = _action_http_timeout() * 1000
    expect(page.locator("[data-notice=stopped] strong").first).to_have_text(
        f"shot stopped by {USER}", timeout=ended_ms
    )
    expect(page.locator("[data-view=last-action] p")).to_have_text(
        "shot (stopped), 0 setting(s) moved", timeout=SETTLE_MS
    )
    expect(page.locator("p.running")).to_have_text("Nothing running", timeout=SETTLE_MS)

    _control(page, "release").click()
    expect(_fact(page, "header", "Holder")).to_have_text("nobody", timeout=SETTLE_MS)
    expect(_control(page, "open")).to_be_visible(timeout=SETTLE_MS)
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
# 6. Threading the errored ring (STRICT, ordered: each test builds on the last)
# ---------------------------------------------------------------------------


def _active_inside_va() -> str:
    """The active scenarios file as the VA container reads it."""
    result = subprocess.run(
        ["docker", "exec", VA_CONTAINER, "cat", VA_ACTIVE_SCENARIOS],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout if result.returncode == 0 else ""


def _sim_apply(stack: DeployedStack, *scenarios: str) -> None:
    """``osprey sim apply`` the scenarios, then wait until the VA serves them.

    ``--no-seed``: the apply only rewrites the active set and never reseeds
    the logbook or the archive. Once the container sees the file, two ticks
    are enough for the VA to rebuild at the new set.
    """
    from osprey_connectors.simulation import DEFAULT_TICK_S
    from osprey_connectors.workspace import resolve_simulation_state_dir

    applied = _run(
        [str(stack.osprey_bin), "sim", "apply", *scenarios, "--no-seed"],
        cwd=stack.repo,
        timeout=120,
    )
    assert applied.returncode == 0, (
        f"osprey sim apply {' '.join(scenarios)} failed (rc={applied.returncode}):\n"
        f"--- stdout ---\n{applied.stdout}\n--- stderr ---\n{applied.stderr}"
    )
    rendered = yaml.safe_load((stack.repo / "build" / "config.yml").read_text(encoding="utf-8"))
    written = (resolve_simulation_state_dir(rendered, stack.repo) / "active_scenarios").read_text(
        encoding="utf-8"
    )
    deadline = time.monotonic() + SCENARIO_VISIBLE_SEC
    while _active_inside_va() != written:
        assert time.monotonic() < deadline, (
            f"{VA_CONTAINER} never saw the active scenarios the host wrote ({written!r})"
        )
        time.sleep(0.2)
    time.sleep(2 * DEFAULT_TICK_S)


def _held(stack: DeployedStack) -> dict[str, str]:
    """The panel credentials plus the session the threading tests hold."""
    assert stack.client_id, "no session is held: an earlier threading test failed"
    return stack.headers(PANEL_PREFIX, **{CLIENT_HEADER: stack.client_id})


def _bpm_index(shot: dict[str, Any], bpm: str | None) -> int | None:
    """The 1-based BPM, in beam order, a shot is lost by; ``None`` for a full turn."""
    return None if bpm is None else shot["bpm_order"].index(bpm) + 1


def _lost_at(shot: dict[str, Any]) -> int | None:
    return _bpm_index(shot, shot["lost"][0] if shot["lost"] else None)


def _shoot(stack: DeployedStack) -> dict[str, Any]:
    status, shot = _request(
        f"{PANEL_PREFIX}/shot",
        "POST",
        body={"n_shots": N_SHOTS},
        headers=_held(stack),
        timeout=_action_http_timeout(),
    )
    assert status == 200, f"shot refused: {status} {shot!r}\n{_logs()}"
    assert shot["shots_included"] == N_SHOTS, (
        f"{N_SHOTS} shots asked, {shot['shots_included']} included: "
        f"{[(f['index'], f['reason']) for f in shot['shots_failed']]}\n"
        f"{_logs()}\n{container_logs(VA_CONTAINER)}"
    )
    assert shot["commands_fired"], f"the shot fired no command: {shot!r}"
    return shot


def test_the_first_shot_of_the_errored_ring_is_lost_by_bpm_36(
    deployed_stack: DeployedStack,
) -> None:
    _sim_apply(deployed_stack, SCENARIO)
    grant = _open(deployed_stack, PANEL_PREFIX)
    assert grant["owner"] == deployed_stack.owner, grant
    deployed_stack.client_id = grant["client_id"]

    shot = _shoot(deployed_stack)
    assert shot["model"] == MODEL, shot["model"]
    per_shot = [_bpm_index(shot, bpm) for bpm in shot["per_shot_lost_at"]]
    deployed_stack.lost_per_shot.append(_lost_at(shot))
    assert len(per_shot) == N_SHOTS, shot["per_shot_lost_at"]
    for k in per_shot:
        assert k is not None and k <= LATEST_LOSS_BPM, (
            f"{SCENARIO} with misalignment_seed {SHIPPED_SEED}: the uncorrected first turn "
            f"must be lost by BPM {LATEST_LOSS_BPM} on every shot; lost at {per_shot}"
        )


def test_applied_steps_bring_the_first_turn_around_within_five(
    deployed_stack: DeployedStack,
) -> None:
    assert deployed_stack.lost_per_shot, "the first shot was not taken"
    lost = deployed_stack.lost_per_shot
    shot: dict[str, Any] | None = None
    while lost[-1] is not None and len(deployed_stack.applied_digests) < MAX_STEPS:
        status, step = _request(f"{PANEL_PREFIX}/step", headers=_held(deployed_stack))
        assert status == 200, f"no step after a lost shot: {status} {step!r}"
        assert not step["all_zero"], f"the step for a lost turn moves nothing: {step['reason']}"
        status, done = _request(
            f"{PANEL_PREFIX}/apply",
            "POST",
            body={"step_digest": step["digest"]},
            headers=_held(deployed_stack),
            timeout=_action_http_timeout(),
        )
        assert status == 200, f"apply refused: {status} {done!r}\n{_logs()}"
        assert not done["stopped"], done
        assert done["moved"], "the apply moved no corrector"
        deployed_stack.applied_digests.append(step["digest"])
        shot = _shoot(deployed_stack)
        lost.append(_lost_at(shot))

    assert lost[-1] is None and shot is not None, (
        f"{SCENARIO} with misalignment_seed {SHIPPED_SEED}: after "
        f"{len(deployed_stack.applied_digests)} applied steps the first turn is still lost "
        f"(lost at BPM per shot: {lost})"
    )
    assert shot["lost"] == [] and shot["per_shot_lost_at"] == [None] * N_SHOTS, shot
    assert shot["last_bpm"] == shot["bpm_order"][-1], shot["last_bpm"]
    assert 1 <= len(deployed_stack.applied_digests) <= MAX_STEPS, lost
    deployed_stack.threaded_shot = shot


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
    shot = deployed_stack.threaded_shot
    assert shot is not None, "the ring was not threaded"
    path = deployed_stack.repo / "var" / "trajectory" / "shots" / MODEL / f"{shot['shot_id']}.json"
    assert path.is_file(), f"no shot file at {path}"
    assert json.loads(path.read_text(encoding="utf-8"))["shot_id"] == shot["shot_id"]

    status, listing = _request(
        f"{PANEL_PREFIX}/shots", headers=deployed_stack.headers(PANEL_PREFIX)
    )
    assert status == 200, f"GET shots: {status} {listing!r}"
    assert listing["model"] == MODEL, listing["model"]
    assert listing["shots"][0]["shot_id"] == shot["shot_id"], (
        "the newest listed shot is not the threaded one: "
        f"{[s['shot_id'] for s in listing['shots']]}"
    )

    status, saved = _request(
        f"{PANEL_PREFIX}/references",
        "POST",
        body={"name": REFERENCE_NAME, "shot_id": shot["shot_id"]},
        headers=_held(deployed_stack),
    )
    assert status == 200, f"save reference: {status} {saved!r}"
    assert saved["shot_id"] == shot["shot_id"] and saved["sha256"], saved
    deployed_stack.reference_saved = True


class _EventTap:
    """The panel's state stream, read on a thread into a list of events."""

    def __init__(self, stack: DeployedStack) -> None:
        req = urllib.request.Request(
            f"{TRAJECTORY_URL}{PANEL_PREFIX}/events", headers=stack.headers(PANEL_PREFIX)
        )
        self._resp = urllib.request.urlopen(req, timeout=_action_http_timeout())  # localhost
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
    assert deployed_stack.threaded_shot is not None, "the ring was not threaded"
    tap = _EventTap(deployed_stack)
    try:
        status, run = _request(
            f"{PANEL_PREFIX}/continuous",
            "POST",
            body={"n_shots": 1, "max_shots": CONTINUOUS_MAX_SHOTS},
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
        deadline = stopped_beating + LEASE_SEC + _action_http_timeout()
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


# ---------------------------------------------------------------------------
# 7. The reference survives a redeploy (LAST: it takes the stack down and up)
# ---------------------------------------------------------------------------


def test_the_selected_reference_survives_a_redeploy(deployed_stack: DeployedStack) -> None:
    assert deployed_stack.reference_saved, "no reference was saved"
    status, selected = _request(
        f"{PANEL_PREFIX}/references/select",
        "POST",
        body={"name": REFERENCE_NAME},
        headers=_held(deployed_stack),
        timeout=_action_http_timeout(),
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
