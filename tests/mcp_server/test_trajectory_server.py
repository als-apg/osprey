"""The trajectory MCP server: its tools, its session, its stamp checks, its hop.

The HTTP boundary (:func:`osprey.mcp_server.trajectory.client._send`) is
replaced by a scripted fake that records every request, so each test states
exactly what the proxy answered and asserts exactly what was sent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from osprey.mcp_server.approval_stamp import approved_values_sha256
from osprey.mcp_server.trajectory import client
from osprey.mcp_server.trajectory.tools import loop
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

_CALL_ID = "toolu_traj1"
_TARGET = "va"
_DIGEST = "d" * 64
_VALUES = [{"name": "HCM1", "before": 0.5}, {"name": "HCM0", "before": -0.25}]


def _snapshot(**over: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "target": _TARGET,
        "caller": "alice",
        "holder": None,
        "queue_wait_s": 4.0,
        "action_timeout_s": 60.0,
        "action_budget_s": 75.0,
        "stop_budget_s": 15.0,
    }
    snap.update(over)
    return snap


def _refusal(code: str, detail: str = "refused", **rest: Any) -> dict[str, Any]:
    return {"detail": {"code": code, "detail": detail, **rest}}


class FakeProxy:
    """The proxy as the client sees it: scripted answers, recorded requests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: dict[tuple[str, str], list[tuple[int, Any]]] = {}

    def on(self, method: str, path: str, *answers: tuple[int, Any]) -> None:
        self.routes[(method, path)] = list(answers)

    def sent(self, method: str, path: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["method"] == method and c["path"] == path]

    async def __call__(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout,
    ) -> tuple[int, Any]:
        self.calls.append(
            {"method": method, "path": path, "json": json, "headers": headers, "timeout": timeout}
        )
        answers = self.routes.get((method, path))
        assert answers, f"unexpected request {method} {path}"
        return answers.pop(0) if len(answers) > 1 else answers[0]


@pytest.fixture(autouse=True)
def _fresh_session():
    client.reset_session()
    yield
    client.reset_session()


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> FakeProxy:
    fake = FakeProxy()
    fake.on("GET", "/state", (200, _snapshot()))
    fake.on("POST", "/session", (200, {"client_id": "c1", "owner": "alice", "target": _TARGET}))
    monkeypatch.setattr(client, "_send", fake)
    return fake


@pytest.fixture(autouse=True)
def config(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The rendered config the tools read; approval asks for both write tools."""
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

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(target_state, "state_dir", lambda: state)
    return state


def _stamp(state: Path, tool: str, **fields: str) -> None:
    payload = {"tool_use_id": _CALL_ID, "tool": f"mcp__trajectory__{tool}", **fields}
    (state / f"approval_ask_{_CALL_ID}.json").write_text(json.dumps(payload), encoding="utf-8")


async def _call(fn: Callable[..., Any], **kwargs: Any) -> Any:
    from osprey.audit.call import call_scope

    with call_scope(_CALL_ID, None):
        return await get_tool_fn(fn)(**kwargs)


def _shot_ok(proxy: FakeProxy) -> None:
    proxy.on("POST", "/shot", (200, {"shot_id": "s1", "commands_fired": []}))


# -- shot -------------------------------------------------------------------------


class TestShot:
    async def test_an_approved_shot_opens_the_session_and_shoots_under_it(self, proxy, stamps):
        _shot_ok(proxy)
        _stamp(stamps, "trajectory_shot", approved_target=_TARGET)

        result = await _call(loop.trajectory_shot, n_shots=3, approved_target=_TARGET)

        assert result["shot_id"] == "s1"
        (shot,) = proxy.sent("POST", "/shot")
        assert shot["json"] == {"n_shots": 3, "average": "mean"}
        assert shot["headers"] == {client.CLIENT_HEADER: "c1"}

    async def test_the_shot_is_bounded_by_the_snapshot_budgets_and_queue_wait(self, proxy, stamps):
        _shot_ok(proxy)
        _stamp(stamps, "trajectory_shot", approved_target=_TARGET)

        await _call(loop.trajectory_shot, approved_target=_TARGET)

        (shot,) = proxy.sent("POST", "/shot")
        # action_timeout_s + stop_budget_s + queue_wait_s
        assert shot["timeout"].read == pytest.approx(60.0 + 15.0 + 4.0)
        assert shot["timeout"].connect == client.CONNECT_TIMEOUT_S

    async def test_the_session_is_opened_once_and_reused(self, proxy, stamps):
        _shot_ok(proxy)
        _stamp(stamps, "trajectory_shot", approved_target=_TARGET)

        await _call(loop.trajectory_shot, approved_target=_TARGET)
        await _call(loop.trajectory_shot, approved_target=_TARGET)

        assert len(proxy.sent("POST", "/session")) == 1
        assert len(proxy.sent("POST", "/shot")) == 2

    async def test_a_shot_without_a_stamp_writes_nothing(self, proxy):
        _shot_ok(proxy)

        with assert_raises_error(error_type="approval_missing"):
            await _call(loop.trajectory_shot, approved_target=_TARGET)

        assert proxy.sent("POST", "/shot") == []
        assert proxy.sent("POST", "/session") == []

    async def test_a_shot_approved_for_another_target_writes_nothing(self, proxy, stamps):
        _shot_ok(proxy)
        _stamp(stamps, "trajectory_shot", approved_target="live")

        with assert_raises_error(error_type="approval_missing") as ctx:
            await _call(loop.trajectory_shot, approved_target="live")

        assert "`va`" in ctx["envelope"]["error_message"]
        assert proxy.sent("POST", "/shot") == []

    async def test_a_stamp_for_another_tool_does_not_admit_a_shot(self, proxy, stamps):
        _shot_ok(proxy)
        _stamp(stamps, "trajectory_apply", approved_target=_TARGET)

        with assert_raises_error(error_type="approval_missing"):
            await _call(loop.trajectory_shot, approved_target=_TARGET)

        assert proxy.sent("POST", "/shot") == []

    async def test_a_shot_asked_of_nobody_is_not_stamp_checked(self, proxy, config):
        config["approval"]["enabled"] = False
        _shot_ok(proxy)

        await _call(loop.trajectory_shot)

        assert len(proxy.sent("POST", "/shot")) == 1

    async def test_an_approval_switch_that_is_not_a_bool_refuses(self, proxy, config):
        config["approval"]["enabled"] = "false"
        _shot_ok(proxy)

        with assert_raises_error(error_type="approval_config_invalid"):
            await _call(loop.trajectory_shot)

        assert proxy.sent("POST", "/shot") == []

    async def test_a_stopped_shot_relays_the_commands_fired(self, proxy, stamps):
        proxy.on(
            "POST",
            "/shot",
            (409, _refusal("action_stopped", "stopped by bob", commands_fired=["arm"])),
        )
        _stamp(stamps, "trajectory_shot", approved_target=_TARGET)

        with assert_raises_error(error_type="action_stopped") as ctx:
            await _call(loop.trajectory_shot, approved_target=_TARGET)

        assert ctx["envelope"]["details"] == {"commands_fired": ["arm"]}


# -- session ----------------------------------------------------------------------


class TestSession:
    async def test_a_stale_same_owner_agent_session_is_taken_over(self, proxy, stamps):
        proxy.on("POST", "/session", (409, _refusal("session_held")))
        proxy.on("GET", "/state", (200, _snapshot(holder={"owner": "alice", "kind": "agent"})))
        proxy.on("POST", "/takeover", (200, {"client_id": "c2", "owner": "alice"}))
        _shot_ok(proxy)
        _stamp(stamps, "trajectory_shot", approved_target=_TARGET)

        await _call(loop.trajectory_shot, approved_target=_TARGET)

        assert len(proxy.sent("POST", "/takeover")) == 1
        assert proxy.sent("POST", "/shot")[0]["headers"] == {client.CLIENT_HEADER: "c2"}

    @pytest.mark.parametrize(
        "holder",
        [{"owner": "alice", "kind": "panel"}, {"owner": "bob", "kind": "agent"}],
        ids=["own-panel", "other-agent"],
    )
    async def test_a_session_anyone_else_holds_is_refused_not_taken(self, proxy, stamps, holder):
        proxy.on("POST", "/session", (409, _refusal("session_held", "held by bob")))
        proxy.on("GET", "/state", (200, _snapshot(holder=holder)))
        _shot_ok(proxy)
        _stamp(stamps, "trajectory_shot", approved_target=_TARGET)

        with assert_raises_error(error_type="session_held"):
            await _call(loop.trajectory_shot, approved_target=_TARGET)

        assert proxy.sent("POST", "/takeover") == []
        assert proxy.sent("POST", "/shot") == []

    async def test_a_session_taken_over_is_forgotten_and_reopened_next_call(self, proxy, stamps):
        proxy.on(
            "POST",
            "/shot",
            (409, _refusal("session_taken_over")),
            (200, {"shot_id": "s2"}),
        )
        _stamp(stamps, "trajectory_shot", approved_target=_TARGET)

        with assert_raises_error(error_type="session_taken_over"):
            await _call(loop.trajectory_shot, approved_target=_TARGET)
        assert client.cached_client_id() is None
        await _call(loop.trajectory_shot, approved_target=_TARGET)

        assert len(proxy.sent("POST", "/session")) == 2

    async def test_the_server_releases_its_session_at_exit(self, proxy, stamps):
        from osprey.mcp_server.trajectory.server import _lifespan, mcp

        proxy.on("DELETE", "/session", (200, {"client_id": "c1"}))
        _shot_ok(proxy)
        _stamp(stamps, "trajectory_shot", approved_target=_TARGET)

        async with _lifespan(mcp):
            await _call(loop.trajectory_shot, approved_target=_TARGET)

        (release,) = proxy.sent("DELETE", "/session")
        assert release["headers"] == {client.CLIENT_HEADER: "c1"}
        assert client.cached_client_id() is None

    async def test_a_server_that_holds_nothing_releases_nothing(self, proxy):
        from osprey.mcp_server.trajectory.server import _lifespan, mcp

        async with _lifespan(mcp):
            pass

        assert proxy.calls == []

    @pytest.mark.usefixtures("proxy")
    async def test_a_failed_release_never_raises(self, monkeypatch):
        from osprey.mcp_server.trajectory.server import _lifespan, mcp

        async def _down(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("terminal gone")

        await client.ensure_session()
        monkeypatch.setattr(client, "_send", _down)

        async with _lifespan(mcp):
            pass


# -- step ------------------------------------------------------------------------


class TestStep:
    async def test_the_step_is_read_without_a_session(self, proxy):
        proxy.on("GET", "/step", (200, {"digest": _DIGEST, "deltas": {"HCM0": 0.1}}))

        result = await get_tool_fn(loop.trajectory_step)()

        assert result["digest"] == _DIGEST
        assert proxy.sent("POST", "/session") == []

    async def test_no_pending_step_is_relayed(self, proxy):
        proxy.on("GET", "/step", (404, _refusal("no_step", "no step is pending")))

        with assert_raises_error(error_type="no_step"):
            await get_tool_fn(loop.trajectory_step)()


# -- apply -----------------------------------------------------------------------


class TestApply:
    def _apply_ok(self, proxy: FakeProxy) -> None:
        proxy.on("POST", "/apply", (200, {"moved": [], "stopped": False}))

    async def test_an_approved_step_is_applied_by_its_digest(self, proxy, stamps):
        self._apply_ok(proxy)
        _stamp(stamps, "trajectory_apply", approved_target=_TARGET, approved_step_digest=_DIGEST)

        await _call(
            loop.trajectory_apply,
            step_digest=_DIGEST,
            approved_target=_TARGET,
            approved_step_digest=_DIGEST,
        )

        (apply,) = proxy.sent("POST", "/apply")
        assert apply["json"] == {"step_digest": _DIGEST}
        assert apply["headers"] == {client.CLIENT_HEADER: "c1"}

    async def test_a_step_digest_other_than_the_approved_one_writes_nothing(self, proxy, stamps):
        self._apply_ok(proxy)
        _stamp(stamps, "trajectory_apply", approved_target=_TARGET, approved_step_digest=_DIGEST)

        with assert_raises_error(error_type="approval_missing"):
            await _call(
                loop.trajectory_apply,
                step_digest="e" * 64,
                approved_target=_TARGET,
                approved_step_digest=_DIGEST,
            )

        assert proxy.sent("POST", "/apply") == []

    async def test_an_apply_approved_for_another_target_writes_nothing(self, proxy, stamps):
        self._apply_ok(proxy)
        _stamp(stamps, "trajectory_apply", approved_target="live", approved_step_digest=_DIGEST)

        with assert_raises_error(error_type="approval_missing"):
            await _call(
                loop.trajectory_apply,
                step_digest=_DIGEST,
                approved_target="live",
                approved_step_digest=_DIGEST,
            )

        assert proxy.sent("POST", "/apply") == []

    async def test_an_approved_put_back_is_sent_as_model_and_values(self, proxy, stamps):
        self._apply_ok(proxy)
        sha = approved_values_sha256(_TARGET, "ring", _VALUES)
        _stamp(stamps, "trajectory_apply", approved_target=_TARGET, approved_values_sha256=sha)

        await _call(
            loop.trajectory_apply,
            model="ring",
            values=_VALUES,
            approved_target=_TARGET,
            approved_values_sha256=sha,
        )

        (apply,) = proxy.sent("POST", "/apply")
        assert apply["json"] == {"model": "ring", "values": _VALUES}

    async def test_a_put_back_other_than_the_approved_one_writes_nothing(self, proxy, stamps):
        self._apply_ok(proxy)
        sha = approved_values_sha256(_TARGET, "ring", _VALUES)
        _stamp(stamps, "trajectory_apply", approved_target=_TARGET, approved_values_sha256=sha)
        changed = [{"name": "HCM1", "before": 9.0}, {"name": "HCM0", "before": -0.25}]

        with assert_raises_error(error_type="approval_missing"):
            await _call(
                loop.trajectory_apply,
                model="ring",
                values=changed,
                approved_target=_TARGET,
                approved_values_sha256=sha,
            )

        assert proxy.sent("POST", "/apply") == []

    async def test_a_put_back_digest_bound_to_another_target_writes_nothing(self, proxy, stamps):
        self._apply_ok(proxy)
        sha = approved_values_sha256("live", "ring", _VALUES)
        _stamp(stamps, "trajectory_apply", approved_target=_TARGET, approved_values_sha256=sha)

        with assert_raises_error(error_type="approval_missing"):
            await _call(
                loop.trajectory_apply,
                model="ring",
                values=_VALUES,
                approved_target=_TARGET,
                approved_values_sha256=sha,
            )

        assert proxy.sent("POST", "/apply") == []

    async def test_a_step_apply_carrying_a_values_digest_writes_nothing(self, proxy, stamps):
        self._apply_ok(proxy)
        sha = approved_values_sha256(_TARGET, "ring", _VALUES)
        _stamp(stamps, "trajectory_apply", approved_target=_TARGET, approved_values_sha256=sha)

        with assert_raises_error(error_type="approval_missing"):
            await _call(
                loop.trajectory_apply,
                step_digest=_DIGEST,
                approved_target=_TARGET,
                approved_values_sha256=sha,
            )

        assert proxy.sent("POST", "/apply") == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"step_digest": _DIGEST, "model": "ring", "values": _VALUES},
            {"model": "ring"},
            {"values": _VALUES},
            {"model": "ring", "values": []},
        ],
        ids=["nothing", "both", "model-only", "values-only", "empty-values"],
    )
    async def test_a_call_that_is_neither_a_step_nor_a_put_back_is_refused(self, proxy, kwargs):
        with assert_raises_error(error_type="bad_request"):
            await _call(loop.trajectory_apply, **kwargs)

        assert proxy.calls == []

    async def test_a_failed_apply_relays_what_moved(self, proxy, stamps):
        moved = [{"name": "HCM0", "before": 0.1, "outcome": "written"}]
        proxy.on("POST", "/apply", (502, _refusal("apply_failed", "write failed", moved=moved)))
        _stamp(stamps, "trajectory_apply", approved_target=_TARGET, approved_step_digest=_DIGEST)

        with assert_raises_error(error_type="apply_failed") as ctx:
            await _call(
                loop.trajectory_apply,
                step_digest=_DIGEST,
                approved_target=_TARGET,
                approved_step_digest=_DIGEST,
            )

        assert ctx["envelope"]["details"] == {"moved": moved}


# -- the HTTP boundary -----------------------------------------------------------


class TestBoundary:
    @pytest.fixture
    def transport(self, monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_snapshot())

        real = httpx.AsyncClient

        def _client(**kwargs: Any) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(client.httpx, "AsyncClient", _client)
        monkeypatch.setattr(
            "osprey.mcp_server.http.web_terminal_url", lambda: "http://127.0.0.1:8087"
        )
        return seen

    async def test_requests_go_through_the_proxy_with_the_panel_token_only(
        self, transport, monkeypatch
    ):
        from osprey.mcp_server import http as mcp_http

        mcp_http.reset_panel_token_latch()
        monkeypatch.setenv("OSPREY_PANEL_TOKEN", "panel-tok")
        monkeypatch.setenv("TRAJECTORY_LAUNCH_TOKEN", "launch-tok")

        await client.get_state()

        (request,) = transport
        assert str(request.url) == "http://127.0.0.1:8087/panel/trajectory/api/state"
        assert request.headers["authorization"] == "Bearer panel-tok"
        assert "x-launch-token" not in request.headers
        assert "launch-tok" not in json.dumps(dict(request.headers))
        mcp_http.reset_panel_token_latch()

    async def test_an_unreachable_terminal_is_named(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        real = httpx.AsyncClient
        monkeypatch.setattr(
            client.httpx,
            "AsyncClient",
            lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
        )
        monkeypatch.setattr(
            "osprey.mcp_server.http.web_terminal_url", lambda: "http://127.0.0.1:8087"
        )

        with assert_raises_error(error_type="trajectory_unreachable"):
            await client.get_state()

    def test_the_proxys_plain_text_502_is_unreachable(self):
        with assert_raises_error(error_type="trajectory_unreachable"):
            client.relay_refusal(502, "Upstream connection failed: refused")

    def test_a_missing_launch_token_names_the_deployment_fix(self):
        with assert_raises_error(error_type="launch_token_invalid") as ctx:
            client.relay_refusal(403, _refusal("launch_token_invalid", "invalid launch token"))

        assert "TRAJECTORY_LAUNCH_TOKEN" in ctx["envelope"]["suggestions"][0]

    def test_the_spellings_match_the_service(self):
        from osprey.services.trajectory import app, sessions

        assert client.CLIENT_HEADER == app.CLIENT_HEADER
        assert client.AGENT_DOOR == app.AGENT_DOOR
        assert client.SESSION_HELD == sessions.SESSION_HELD
        assert client.NO_SESSION == sessions.NO_SESSION
        assert client.NOT_SESSION_HOLDER == sessions.NOT_SESSION_HOLDER
        assert client.SESSION_TAKEN_OVER == sessions.SESSION_TAKEN_OVER

    def test_the_agent_door_paths_are_panel_tier_routes(self):
        from osprey.interfaces.web_auth import PANEL_TIER_ROUTES

        routes = {(m, p) for m, p in PANEL_TIER_ROUTES}
        for method, path in [
            ("POST", "/session"),
            ("DELETE", "/session"),
            ("POST", "/takeover"),
            ("GET", "/state"),
            ("POST", "/shot"),
            ("GET", "/step"),
            ("POST", "/apply"),
        ]:
            assert (method, f"{client.BASE_PATH}{path}") in routes


# -- registration ----------------------------------------------------------------


class TestRegistration:
    def test_the_server_registers_exactly_the_three_tools(self):
        from osprey.mcp_server.trajectory.server import create_server
        from tests.mcp_server.conftest import registered_tool_names

        server = create_server()

        assert sorted(registered_tool_names(server)) == [
            "trajectory_apply",
            "trajectory_shot",
            "trajectory_step",
        ]

    def test_the_registry_row_reads_the_step_freely_and_asks_for_writes(self):
        from osprey.registry.mcp import FRAMEWORK_SERVERS

        row = FRAMEWORK_SERVERS["trajectory"]

        assert row.module == "osprey.mcp_server.trajectory"
        assert row.condition == "trajectory_panel_declared"
        assert row.permissions_allow == ["trajectory_step"]
        assert sorted(row.permissions_ask) == ["trajectory_apply", "trajectory_shot"]


# -- the proxy hop ---------------------------------------------------------------


class TestProxyHop:
    @pytest.mark.parametrize(
        ("panel_id", "path", "expected"),
        [
            ("trajectory", "api/shot", True),
            ("trajectory", "api/state", True),
            ("trajectory", "panel/api/state", True),
            ("trajectory", "panel/api/apply", True),
            ("trajectory", "panel/", False),
            ("trajectory", "panel/static/panel.js", False),
            ("trajectory", "", False),
            ("events", "api/shot", False),
            ("events", "panel/api/state", False),
        ],
    )
    def test_both_doors_are_the_trajectory_hop(self, panel_id, path, expected):
        from osprey.interfaces.web_terminal.routes import proxy

        assert proxy._is_trajectory_door_hop(panel_id, path) is expected

    def test_the_hop_has_a_connect_bound_and_no_read_bound(self):
        from osprey.interfaces.web_terminal.routes import proxy

        bounds = proxy._TRAJECTORY_DOOR_HOP_TIMEOUT
        assert bounds.read is None
        assert bounds.connect == 5.0

    async def test_the_request_carries_the_hop_bounds_when_given(self):
        from osprey.interfaces.web_terminal.routes import proxy

        seen: dict[str, Any] = {}

        class _Client:
            async def request(self, **kwargs: Any) -> str:
                seen.update(kwargs)
                return "resp"

        await proxy._request_no_redirect(
            _Client(),  # type: ignore[arg-type]
            method="POST",
            url="http://svc/api/shot",
            headers={},
            content=None,
            timeout=proxy._TRAJECTORY_DOOR_HOP_TIMEOUT,
        )

        assert seen["timeout"] is proxy._TRAJECTORY_DOOR_HOP_TIMEOUT
        assert seen["follow_redirects"] is False
