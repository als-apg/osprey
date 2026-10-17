"""The agent's path end to end: MCP tool -> terminal proxy -> trajectory service.

The three trajectory tools run unmodified; their HTTP client reaches the real
web-terminal app in process, whose real panel proxy forwards to a real
trajectory service built with :func:`osprey.services.trajectory.app.create_app`.
Both hops are recorded as they leave their sender, so each row can say which
credential crossed which hop. The service's connector is the doors tests'
stand-in machine, so a write that reached the hardware is a listed write.

Every row carries ``no_auth_seam``: the suite-wide seam would set the operator
secret on the MCP client's request, and the agent never holds it.
"""

from __future__ import annotations

import json
import types
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from osprey.audit.envelope import DECISION_ALLOWED
from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, WebCredentials, get_web_credentials
from osprey.interfaces.web_terminal.app import (
    TRAJECTORY_LAUNCH_TOKEN_ENV,
    TRAJECTORY_PANEL_ID,
    UNIVERSAL_PANELS,
)
from osprey.interfaces.web_terminal.app import create_app as create_terminal_app
from osprey.mcp_server.http import reset_panel_token_latch
from osprey.mcp_server.trajectory import client
from osprey.mcp_server.trajectory.tools import loop
from osprey.services.trajectory.app import LAUNCH_TOKEN_ENV
from osprey.services.trajectory.app import create_app as create_service_app
from osprey.services.trajectory.startup import TrajectoryStartup
from osprey.services.trajectory.step import compute_step
from osprey.utils.identity import TERMINAL_USER_ENV
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn
from tests.services.trajectory.test_doors import (  # noqa: F401  (fixtures used by name)
    COMMANDS,
    HCM1,
    HCM2,
    TARGET,
    TOOL,
    VCM1,
    _deployment,
    _Machine,
    _ours,
    _shot,
    machine,
    records,
    state_dir,
)

pytestmark = pytest.mark.no_auth_seam

LAUNCH_TOKEN = "trajectory-launch-token-" + "t" * 32
BACKEND_URL = "http://127.0.0.1:8095"
_CALL_ID = "toolu_proxy1"
_OTHER_DIGEST = "e" * 64


class _Hops:
    """Every request each hop sent, as ``(method, path, headers)``."""

    def __init__(self) -> None:
        self.agent: list[tuple[str, str, dict[str, str]]] = []
        self.service: list[tuple[str, str, dict[str, str]]] = []

    @staticmethod
    def recorder(into: list[tuple[str, str, dict[str, str]]]):
        async def record(request: httpx.Request) -> None:
            into.append((request.method, request.url.path, dict(request.headers)))

        return record

    def service_paths(self, method: str | None = None) -> list[str]:
        return [path for m, path, _ in self.service if method is None or m == method]


def _declared() -> dict[str, Any]:
    return {
        "id": TRAJECTORY_PANEL_ID,
        "label": "TRAJECTORY",
        "url": BACKEND_URL,
        "configDefined": True,
    }


@pytest.fixture(autouse=True)
def _fresh_session():
    client.reset_session()
    reset_panel_token_latch()
    yield
    client.reset_session()
    reset_panel_token_latch()


@pytest.fixture(autouse=True)
def config(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Approval asks for both writing tools, as a rendered deployment does."""
    cfg: dict[str, Any] = {
        "approval": {
            "enabled": True,
            "tools": {"trajectory_shot": "ask", "trajectory_apply": "ask"},
        }
    }
    monkeypatch.setattr(loop, "_osprey_config", lambda: cfg)
    return cfg


@pytest.fixture(autouse=True)
def stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from osprey.mcp_server.control_system import target_state

    directory = tmp_path / "stamps"
    directory.mkdir()
    monkeypatch.setattr(target_state, "state_dir", lambda: directory)
    return directory


def _stamp(directory: Path, tool: str, **fields: str) -> None:
    """The approval hook's ask stamp for this call, as the hook writes it."""
    payload = {"tool_use_id": _CALL_ID, "tool": f"mcp__trajectory__{tool}", **fields}
    (directory / f"approval_ask_{_CALL_ID}.json").write_text(json.dumps(payload), encoding="utf-8")


async def _call(fn: Any, **kwargs: Any) -> Any:
    from osprey.audit.call import call_scope

    with call_scope(_CALL_ID, None):
        return await get_tool_fn(fn)(**kwargs)


@pytest.fixture
async def wired(
    tmp_path: Path,
    machine: _Machine,  # noqa: F811
    state_dir: Path,  # noqa: F811, ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[_Hops, FastAPI]]:
    """Boot terminal and service in this loop and route the MCP client through them."""
    hops = _Hops()
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
    monkeypatch.setenv(TRAJECTORY_LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)

    async with AsyncExitStack() as stack:
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace)},
            )
        )
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._load_panel_config",
                return_value=(set(UNIVERSAL_PANELS), [_declared()], None),
            )
        )
        terminal = create_terminal_app(shell_command="echo")
        await stack.enter_async_context(terminal.router.lifespan_context(terminal))
        credentials = get_web_credentials(terminal)

        # The service is its own process in a deployment: its token is in its
        # own environment, which the terminal (sharing this one) just emptied.
        monkeypatch.setenv(LAUNCH_TOKEN_ENV, LAUNCH_TOKEN)

        async def start() -> TrajectoryStartup:
            return TrajectoryStartup(
                target=TARGET,
                connector=machine,
                commands=COMMANDS,
                guarded_run_dir=Path("/unused"),
                tool=TOOL,
            )

        service = create_service_app(start=start)
        # The terminal's operator secret is alice's roster secret at the service.
        service.state.web_credentials = WebCredentials(
            operator_secret="service-own-secret-" + "s" * 32,
            panel_token="service-panel-token",
            roster_secrets=(credentials.operator_secret,),
            roster_owners=("alice",),
        )
        await stack.enter_async_context(service.router.lifespan_context(service))

        terminal.state.proxy_client = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=service),
                event_hooks={"request": [_Hops.recorder(hops.service)]},
            )
        )

        shim = types.ModuleType("httpx")
        shim.__dict__.update(httpx.__dict__)
        shim.AsyncClient = partial(  # type: ignore[attr-defined]
            httpx.AsyncClient,
            transport=httpx.ASGITransport(app=terminal),
            event_hooks={"request": [_Hops.recorder(hops.agent)]},
        )
        monkeypatch.setattr(client, "httpx", shim)
        monkeypatch.setenv(PANEL_TOKEN_ENV, credentials.panel_token)

        yield hops, service


async def _pending_step(service: FastAPI) -> Any:
    """Open the agent's session and leave a step pending under it."""
    client_id = await client.ensure_session()
    shot = _shot()
    step = compute_step(TOOL, shot, None, target=TARGET, client_id=client_id, bounds={})
    service.state.sessions.step.replace(step, shot)
    service.state.sessions.latest_shot_id = shot.shot_id
    return step


async def _the_step(service: FastAPI) -> str:
    """Leave a step pending and read it back through ``trajectory_step``."""
    step = await _pending_step(service)
    read = await _call(loop.trajectory_step)
    assert read["digest"] == step.digest
    return str(step.digest)


# -- the whole path ------------------------------------------------------------


async def test_shot_step_apply_reach_the_machine_as_the_terminals_owner(
    wired: tuple[_Hops, FastAPI],
    stamps: Path,
    machine: _Machine,  # noqa: F811
    records: list[dict[str, Any]],  # noqa: F811
) -> None:
    hops, service = wired

    _stamp(stamps, "trajectory_shot", approved_target=TARGET)
    # The service has no sequence module: the shot is refused there, after the
    # stamp checks passed and the session was opened through the proxy.
    with assert_raises_error(error_type="no_sequence_module"):
        await _call(loop.trajectory_shot, n_shots=1, approved_target=TARGET)
    (shot_line,) = [r for r in _ours(records) if r["subject"] == "shot"]
    assert shot_line["actor"] == "alice"

    digest = await _the_step(service)
    _stamp(stamps, "trajectory_apply", approved_target=TARGET, approved_step_digest=digest)
    done = await _call(
        loop.trajectory_apply,
        step_digest=digest,
        approved_target=TARGET,
        approved_step_digest=digest,
    )

    assert [moved["name"] for moved in done["moved"]] == [HCM1, HCM2, VCM1]
    assert [address for address, _ in machine.writes] == [HCM1, HCM2, VCM1]
    (line,) = [r for r in _ours(records) if r["subject"] == "apply"]
    assert line["actor"] == "alice"
    assert line["decision"] == DECISION_ALLOWED
    assert "owner=alice" in line["detail"]
    assert f"step_digest={digest}" in line["detail"]
    assert "/api/apply" in hops.service_paths("POST")


async def test_the_launch_token_crosses_only_the_proxy_to_service_hop(
    wired: tuple[_Hops, FastAPI],
) -> None:
    hops, service = wired

    await _the_step(service)

    assert hops.agent, "the MCP client reached the terminal"
    for _method, path, headers in hops.agent:
        assert path.startswith(client.BASE_PATH)
        assert LAUNCH_TOKEN not in " ".join(headers.values())
        assert LAUNCH_TOKEN_HEADER.lower() not in headers
    assert hops.service, "the proxy reached the service"
    for _method, path, headers in hops.service:
        assert path.startswith("/api/")
        assert headers[LAUNCH_TOKEN_HEADER.lower()] == LAUNCH_TOKEN


async def test_a_bad_panel_token_is_refused_at_the_terminal(
    wired: tuple[_Hops, FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    hops, _service = wired
    monkeypatch.setenv(PANEL_TOKEN_ENV, "not-the-panel-token-" + "x" * 24)
    reset_panel_token_latch()

    with assert_raises_error(error_type="trajectory_refused") as refused:
        await _call(loop.trajectory_step)

    assert refused["envelope"]["error_message"].startswith("HTTP 401")
    assert hops.agent, "the request reached the terminal"
    assert hops.service == [], "nothing reached the service"


# -- refused before any write ---------------------------------------------------


async def test_a_stamp_digest_other_than_the_calls_is_refused_before_the_write(
    wired: tuple[_Hops, FastAPI],
    stamps: Path,
    machine: _Machine,  # noqa: F811
) -> None:
    hops, service = wired
    digest = await _the_step(service)
    _stamp(stamps, "trajectory_apply", approved_target=TARGET, approved_step_digest=_OTHER_DIGEST)

    with assert_raises_error(error_type=loop.APPROVAL_MISSING):
        await _call(
            loop.trajectory_apply,
            step_digest=digest,
            approved_target=TARGET,
            approved_step_digest=digest,
        )

    assert machine.writes == []
    assert "/api/apply" not in hops.service_paths()


async def test_an_approved_digest_other_than_the_steps_is_refused_before_the_write(
    wired: tuple[_Hops, FastAPI],
    stamps: Path,
    machine: _Machine,  # noqa: F811
) -> None:
    hops, service = wired
    digest = await _the_step(service)
    _stamp(stamps, "trajectory_apply", approved_target=TARGET, approved_step_digest=_OTHER_DIGEST)

    with assert_raises_error(error_type=loop.APPROVAL_MISSING):
        await _call(
            loop.trajectory_apply,
            step_digest=digest,
            approved_target=TARGET,
            approved_step_digest=_OTHER_DIGEST,
        )

    assert machine.writes == []
    assert "/api/apply" not in hops.service_paths()


@pytest.mark.parametrize("tool", ["trajectory_shot", "trajectory_apply"])
async def test_an_approved_target_other_than_the_services_is_refused_before_the_write(
    wired: tuple[_Hops, FastAPI],
    stamps: Path,
    machine: _Machine,  # noqa: F811
    tool: str,
) -> None:
    hops, service = wired
    digest = await _the_step(service)
    other = "not-" + TARGET

    with assert_raises_error(error_type=loop.APPROVAL_MISSING) as refused:
        if tool == "trajectory_shot":
            _stamp(stamps, tool, approved_target=other)
            await _call(loop.trajectory_shot, approved_target=other)
        else:
            _stamp(stamps, tool, approved_target=other, approved_step_digest=digest)
            await _call(
                loop.trajectory_apply,
                step_digest=digest,
                approved_target=other,
                approved_step_digest=digest,
            )

    assert f"`{TARGET}`" in refused["envelope"]["error_message"]
    assert machine.writes == []
    assert hops.service_paths("POST") == ["/api/session"]
