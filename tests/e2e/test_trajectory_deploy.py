"""Full-stack Docker proof of the deployed trajectory service's two doors.

A single-user control-assistant deployment with the virtual accelerator served
is built and started with ``osprey up``; the trajectory container it brings up
is then driven over raw HTTP exactly as its two callers reach it:

* the AGENT door (``/api/*``), which also demands the service's launch token;
* the PANEL door (``/panel/api/*``), the operator panel's.

Both doors sit behind the shared web gate, which admits a request by the
operator secret it carries and stamps the owner from the credential that
matched -- never from anything the request claims. In the single-user shape the
deployment secret is listed under the build account's name, so that account is
the owner every admitted request names.

What is pinned here and nowhere else: the deployed ``/health`` reaching its
target; the agent door's 403 ``launch_token_invalid`` on a wrong token; on both
doors, a request without the secret refused, a body naming an owner refused
400 ``owner_in_body``, a second holder refused 409 ``session_held``, and the
admitted owner read back through ``GET state``; and an action's audit record
landing under ``var/audit/trajectory/`` on the host. Opening a session is the
only writing action exercised -- nothing here moves a magnet or asserts beam
physics.

Reuses ``tests/e2e/_orm_stack.py`` (the VA-backed control-assistant deploy
config: virtual-accelerator control system, web terminals off -- the
single-user shape -- dispatch, record archive and trajectory service dropped).
The lane states the preset's ``trajectory:`` block back, on the host network,
so the trajectory service is the one service it deploys on the project image.

Per-function flaky, NOT module-level: the health/session checks accept one
rerun on a timing assertion; the security negatives (wrong token, no secret,
owner in the body) stay STRICT with no flaky marker, so a rerun can never
sweep a refusal that failed to happen into a pass.

CONTAINER SAFETY: every docker/compose invocation names an EXACT resource
(images ``<project>:local``, ``<project>-va:local``,
``<project>-bluesky-bridge:local``) -- never ``system prune``, never
``--volumes``, never a wildcard ``docker rm``/``rmi``. Teardown goes through
``osprey down`` followed by exact-named removal of this project's own volumes
(``tests/e2e/_volumes.py``).

Gating: needs Docker. Lives in ``tests/e2e/`` (never collected by the fast
lane) and is ``--ignore``'d by the shared e2e lane; runs in its own
``trajectory-deploy-e2e`` CI job (same-repo PRs only, no secrets -- the service
never reaches a model), which treats any skip as a failure. Locally, only one
e2e stack may run per host at a time.
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

# ariel-postgres and OpenObserve are deployed by the preset with no profile knob
# to drop them; both move to high, unassigned host ports so a locally-running
# deployment holding the defaults cannot abort `osprey up`.
POSTGRES_PORT = 25437
OPENOBSERVE_PORT = 25086

PROJECT_NAME = "trajectory-proj"
TRAJECTORY_CONTAINER = f"{_orm_stack.project_prefix(PROJECT_NAME)}-trajectory"

# The trajectory service runs the PROJECT image; the VA and the bridge build
# their own. Exact tags, derived as the compose templates derive them.
PROJECT_IMAGE = f"{_orm_stack.project_prefix(PROJECT_NAME)}:local"
VA_IMAGE = _orm_stack.va_image(PROJECT_NAME)
BRIDGE_IMAGE = _orm_stack.bridge_image(PROJECT_NAME)

BUILD_TIMEOUT_SEC = _orm_stack.BUILD_TIMEOUT_SEC
# The first-time native VA build and the project image are slow on a cold cache.
DEPLOY_UP_TIMEOUT_SEC = 1200
# How long the container may take to answer /health at all ...
HEALTH_TIMEOUT_SEC = 300.0
# ... and, once it answers, how long until it reaches its control target.
REACHABLE_WITHIN_SEC = 30.0

AGENT_PREFIX = "/api"
PANEL_PREFIX = "/panel/api"
DOORS = (AGENT_PREFIX, PANEL_PREFIX)

SECRET_HEADER = "X-Osprey-Terminal-Secret"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]


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
    from osprey.utils.dotenv import parse_dotenv_file

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
) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    all_headers = {"Content-Type": "application/json"} if data is not None else {}
    all_headers.update(headers or {})
    req = urllib.request.Request(
        f"{TRAJECTORY_URL}{path}", data=data, method=method, headers=all_headers
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # localhost
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


def _trajectory_port() -> int:
    from osprey.port_layout import default_port

    return default_port("trajectory", base=PORT_BASE)


TRAJECTORY_URL = f"http://127.0.0.1:{_trajectory_port()}"


class DeployedStack:
    def __init__(self, repo: Path, secret: str, launch_token: str, owner: str):
        self.repo = repo
        self.secret = secret
        self.launch_token = launch_token
        #: The build account the single-user roster lists the deployment
        #: secret under — the owner the gate stamps on every admitted request.
        self.owner = owner

    def headers(self, prefix: str, **extra: str) -> dict[str, str]:
        """The credentials a door's legitimate caller carries."""
        headers = {SECRET_HEADER: self.secret, **extra}
        if prefix == AGENT_PREFIX:
            headers["X-Launch-Token"] = self.launch_token
        return headers


def _logs() -> str:
    return container_logs(TRAJECTORY_CONTAINER)


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeployedStack]:
    from osprey.utils.identity import acting_identity

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
                "services.openobserve.port": OPENOBSERVE_PORT,
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
    # Fresh --dev builds so the deployed containers run CURRENT source.
    _orm_stack.force_image_rebuild(PROJECT_IMAGE, VA_IMAGE, BRIDGE_IMAGE)

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
        yield DeployedStack(
            repo=repo,
            secret=_minted(repo, "OSPREY_TERMINAL_SECRET"),
            launch_token=_minted(repo, "TRAJECTORY_LAUNCH_TOKEN"),
            # The build ran in this process's environment, so the account it
            # listed the secret under is the one this process resolves.
            owner=acting_identity(),
        )
    finally:
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}")
        remove_project_volumes(_orm_stack.project_prefix(PROJECT_NAME))


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
    assert status == 200 and state.get("owner") is None, f"a refused open left a holder: {state!r}"


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
            assert state["owner"] == deployed_stack.owner, state
            assert state["caller"] == deployed_stack.owner, state
            assert state["kind"] == ("agent" if first == AGENT_PREFIX else "panel"), state
            assert "client_id" not in state, f"state leaked the holder's client id: {state!r}"
            for secret in (deployed_stack.secret, deployed_stack.launch_token):
                assert secret not in json.dumps(state), "state echoed a credential"

        status, body = _request(f"{second}/session", "POST", headers=deployed_stack.headers(second))
        assert status == 409, f"second open through {second}: {status} {body!r}"
        assert _code(body) == "session_held", body
    finally:
        _release(deployed_stack, first, grant["client_id"])
    status, state = _request(f"{first}/state", headers=deployed_stack.headers(first))
    assert status == 200 and state["owner"] is None, f"release left a holder: {state!r}"


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
