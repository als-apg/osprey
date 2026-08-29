"""Binding a queued Bluesky PLAN to the plan lane it will actually run on.

A **plan lane** is a whole Bluesky stack wired at render time to ONE control
target. A deployment renders one lane, or — when its build profile opted in —
two, one per target. The session, meanwhile, moves between targets at run time,
and only the HOST can see where it is: the session target lives in a state file
the controls MCP server writes, outside every bridge container.

So the host is what routes, and these tests pin the routing at the deployment
shape that decides it. The lane COUNT is the only branch point:

* **One lane** — the shape every deployment has shipped so far. There is nothing
  to route to, so a switched session is refused outright, exactly as it has been
  since the plan stack shipped (``test_single_lane_switch_refusal.py`` owns that
  contract in full; what is pinned here is that lanes did not change it).
* **Two lanes** — the switch becomes an ADDRESS. ``queue_add`` goes to the lane
  serving the session's target and reports which lane it bound the item to;
  ``queue_start`` must name that lane and is refused when it is no longer the
  active one — the mid-queue switch, which is the case this whole binding exists
  for. An item composed for one machine must never be started against the other.

**Halting is the exception to all of it.** Every decision above asks where the
SESSION is pointed. A halt asks where the HARDWARE is moving, which stops being
the same question the moment an operator switches targets mid-run — so
``stop_run`` and ``queue_stop`` are addressed to the lane that is actually
running, found by asking the bridges rather than by requiring a parameter
nobody should have to have kept. A halt that could be misdirected, or refused,
by a target switch is a halt with a failure mode.

The rest is pinned because nothing else pins it:

* the lane id rides on ``queue_add``'s result on EVERY deployment, single-lane
  included, where it is always ``"bluesky"`` — a field that appeared only
  sometimes is a field every consumer has to handle the absence of;
* the draft, and therefore the revision counter, is per lane: it is state held
  by a bridge process, so revision 4 exists on both lanes and means two
  different plans. The launch pin is ``(lane, revision)``, and naming a lane
  whose session has since moved is refused rather than silently redirected;
* the host composes ``active`` onto each lane's static capability record. The
  bridge publishes what a lane IS; the host is the only layer entitled to say
  which one the session is on — and one unreachable lane degrades to an
  ``error`` on its own entry rather than taking the board down with it.

Requests are intercepted at the ``httpx`` boundary rather than at the tool's own
HTTP helper, because the whole question here is WHICH BRIDGE a call reached —
which is a URL and a launch token, and a mock one layer higher would hide both.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from osprey.bluesky_bridge_connection import resolve_lane_bridge_urls
from osprey.mcp_server.bluesky import lanes as lanes_module
from osprey.mcp_server.bluesky.server_context import (
    initialize_server_context,
    reset_server_context,
)
from osprey.mcp_server.bluesky.tools import draft as draft_tools
from osprey.mcp_server.bluesky.tools import queue
from osprey.mcp_server.control_system import target_banner, target_state
from osprey.services.bluesky_bridge import queue_backend as qb
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

pytestmark = pytest.mark.unit

_CTX = "osprey.mcp_server.bluesky.server_context"

# Lane 1 resolves its URL from `bluesky.bridge_url`, every further lane from its
# own published port — so the two lanes are distinguishable by URL alone, which
# is what lets a test say which bridge answered.
_LANE_ONE_URL = "http://lane-one.test"
_VA_LANE_PORT = 10081
_VA_LANE_URL = f"http://127.0.0.1:{_VA_LANE_PORT}"

_LANE_ONE_TOKEN = "lane-one-launch-token"
_VA_LANE_TOKEN = "va-lane-launch-token"

# Connector type per deployment baseline, as `target_banner.resolve_baseline_target`
# reads it.
_BASELINE_TYPES = {"live": "epics", "va": "virtual_accelerator"}


# ---------------------------------------------------------------------------
# Fake bridges: one per lane, addressed by base URL
# ---------------------------------------------------------------------------
class _Bridges:
    """The lanes' HTTP endpoints, keyed by base URL.

    Stands in for the ``httpx`` module inside the MCP server's HTTP boundary, so
    every request goes through the real URL resolution and carries the real
    per-lane launch token header.
    """

    HTTPError = httpx.HTTPError

    def __init__(self, answers: dict[str, dict[str, Any]]) -> None:
        #: ``{base_url: {path: body}}`` — the body each lane answers with.
        self.answers = answers
        #: Every request, as ``(method, url, headers)``.
        self.calls: list[tuple[str, str, dict | None]] = []
        #: Base URLs of bridges that are DOWN — every request to one raises,
        #: the way an unreachable container does.
        self.down: set[str] = set()

    def _respond(self, method: str, url: str, headers: dict | None) -> SimpleNamespace:
        self.calls.append((method, url, headers))
        for base in self.down:
            if url.startswith(base):
                raise httpx.ConnectError(f"connection refused: {url}")
        for base, routes in self.answers.items():
            if not url.startswith(base):
                continue
            path = url[len(base) :]
            body = routes.get(path.split("?")[0])
            if body is None:
                return SimpleNamespace(status_code=404, json=lambda: {"detail": "no such route"})
            return SimpleNamespace(status_code=200, json=lambda body=body: body)
        raise AssertionError(f"request to an unknown bridge: {url}")

    def get(self, url: str, timeout: float | None = None, **kwargs: Any) -> SimpleNamespace:
        return self._respond("GET", url, kwargs.get("headers"))

    def post(self, url: str, timeout: float | None = None, **kwargs: Any) -> SimpleNamespace:
        return self._respond("POST", url, kwargs.get("headers"))

    def patch(self, url: str, timeout: float | None = None, **kwargs: Any) -> SimpleNamespace:
        return self._respond("PATCH", url, kwargs.get("headers"))

    def delete(self, url: str, timeout: float | None = None, **kwargs: Any) -> SimpleNamespace:
        return self._respond("DELETE", url, kwargs.get("headers"))

    # -- assertions helpers -------------------------------------------------
    def urls(self, path: str) -> list[str]:
        """Every request made to *path*, as full URLs."""
        return [url for _, url, _ in self.calls if url.endswith(path)]

    def token_for(self, path: str) -> str | None:
        """The launch token the (single) request to *path* carried."""
        matching = [headers for _, url, headers in self.calls if url.endswith(path)]
        assert len(matching) == 1, f"expected one request to {path}, got {len(matching)}"
        return (matching[0] or {}).get("X-Launch-Token")


def _capability(lane: str, lane_target: str) -> dict:
    """A lane's static capability record, exactly as its bridge publishes it."""
    return {
        "status": "ok",
        "capability": {
            "can_execute": True,
            "reason": "executable",
            "detail": "this deployment can execute plans",
            "lane": lane,
            "lane_target": lane_target,
        },
    }


def _queue_view(lane: str, *, running: bool) -> dict:
    """One lane's queue as the manager holds it — idle, or with a plan in motion."""
    if not running:
        return {"status": {"manager_state": "idle"}, "items": [], "running_item": None}
    return {
        "status": {"manager_state": "executing_queue", "running_item_uid": f"uid-{lane}"},
        "items": [],
        "running_item": {"item_uid": f"uid-{lane}", "name": "count"},
    }


def _lane_routes(
    lane: str, lane_target: str, *, revision: int, running: bool = False
) -> dict[str, Any]:
    """The handful of routes these tests exercise on one lane's bridge."""
    return {
        "/health": _capability(lane, lane_target),
        "/queue": _queue_view(lane, running=running),
        "/queue/items": {"run_id": f"run-on-{lane}", "item": {"item_uid": f"uid-{lane}"}},
        "/queue/start": {"started": True, "msg": f"started on {lane}"},
        "/queue/stop": {"stop_pending": True, "msg": f"stopping {lane}"},
        "/queue/abort": {"aborted": True, "abort_pending": False, "msg": f"aborted on {lane}"},
        "/draft": {"draft": {"plan_name": f"plan_on_{lane}"}, "revision": revision},
    }


# ---------------------------------------------------------------------------
# Deployment staging
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_bluesky_context():
    yield
    reset_server_context()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """Stage a rendered deployment of one or two lanes and arm the fake bridges.

    Returns a callable taking the baseline target and the second lane's target
    (``None`` for a single-lane deployment), plus an optional override of the
    second lane's config block — which is how the "renders a lane it cannot
    address" case is staged — and of the whole ``control_system`` section, which
    is how a per-target write posture is staged.
    """

    def _stage(
        baseline: str,
        second: str | None = None,
        *,
        second_block: dict | None = None,
        control_system: dict | None = None,
        running: str | None = None,
        revisions: tuple[int, int] = (1, 7),
    ) -> _Bridges:
        services: dict[str, Any] = {"bluesky": {"path": "./services/bluesky", "port": 10080}}
        if second is not None:
            # Both lanes carry `target:` on a two-lane deploy, which is what the
            # single-lane block has never had a reason to.
            services["bluesky"]["target"] = baseline
            services[f"bluesky_{second}"] = (
                second_block
                if second_block is not None
                else {"path": "./services/bluesky", "port": _VA_LANE_PORT, "target": second}
            )
        section = (
            control_system
            if control_system is not None
            else {"type": _BASELINE_TYPES[baseline], "writes_enabled": True}
        )
        config = {
            "control_system": section,
            "bluesky": {"bridge_url": _LANE_ONE_URL},
            "services": services,
        }

        monkeypatch.setattr(target_banner, "load_osprey_config", lambda: config)
        monkeypatch.setattr("osprey_connectors.workspace.load_osprey_config", lambda: config)

        def fake_get_config_value(key: str, default: Any = None, config_path: Any = None) -> Any:
            # The whole section, not only the deployment-wide flag: write posture
            # is resolved per control target out of `control_system.connector`,
            # so a stub that served one dotted key would answer "unarmed" for
            # every lane whatever this deployment says.
            return {
                "control_system": section,
                "control_system.writes_enabled": section.get("writes_enabled", False),
            }.get(key, default)

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)
        monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)

        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", _LANE_ONE_TOKEN)
        monkeypatch.setenv("BLUESKY_VA_LAUNCH_TOKEN", _VA_LANE_TOKEN)
        monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)
        monkeypatch.delenv("BLUESKY_VA_BRIDGE_URL", raising=False)

        answers = {
            _LANE_ONE_URL: _lane_routes(
                "bluesky", baseline, revision=revisions[0], running=running == "bluesky"
            )
        }
        if second is not None:
            second_key = f"bluesky_{second}"
            answers[_VA_LANE_URL] = _lane_routes(
                second_key, second, revision=revisions[1], running=running == second_key
            )
        bridges = _Bridges(answers)
        monkeypatch.setattr(f"{_CTX}.httpx", bridges)

        initialize_server_context()
        return bridges

    return _stage


def _session_on(target: str) -> None:
    """Write the state file a controls server owned by this session would write."""
    target_state.write_on_start(target)


def _switch_to(target: str, generation: int = 1) -> None:
    """Move an already-published session onto *target*, as a switch would."""
    target_state.publish_switch(target, generation=generation)


async def _add(revision: int = 1) -> dict:
    with_activity = get_tool_fn(queue.queue_add)
    return json.loads(await with_activity(draft_revision=revision))


async def _start(lane: str | None = None) -> dict:
    return json.loads(await get_tool_fn(queue.queue_start)(lane=lane))


async def _status() -> dict:
    return json.loads(await get_tool_fn(queue.queue_status)())


async def _stop_run() -> dict:
    from osprey.mcp_server.bluesky.tools import stop as stop_tools

    return json.loads(await get_tool_fn(stop_tools.stop_run)())


@pytest.fixture(autouse=True)
def _silence_activity(monkeypatch):
    """The agent-activity emit is a side channel; it reaches no bridge."""

    async def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr("osprey.mcp_server.bluesky.tools.queue.notify_agent_activity_async", _noop)
    monkeypatch.setattr("osprey.mcp_server.bluesky.tools.stop.notify_agent_activity_async", _noop)
    monkeypatch.setattr(draft_tools, "notify_agent_activity", lambda **kwargs: None)


# =========================================================================
# Single lane: everything the deployment shape has always done, unchanged
# =========================================================================


@pytest.mark.parametrize("baseline", ["live", "va"])
async def test_a_single_lane_add_reaches_the_only_bridge(deployment, baseline):
    bridges = deployment(baseline)

    await _add()

    assert bridges.urls("/queue/items") == [f"{_LANE_ONE_URL}/queue/items"]
    assert bridges.token_for("/queue/items") == _LANE_ONE_TOKEN


@pytest.mark.parametrize(("baseline", "session"), [("live", "va"), ("va", "live")])
async def test_a_single_lane_still_refuses_a_switched_session(deployment, baseline, session):
    """Task 3.9's blanket refusal, which two lanes supersede and one lane keeps."""
    bridges = deployment(baseline)
    _session_on(baseline)
    _switch_to(session)

    with assert_raises_error(error_type=qb.REASON_SESSION_TARGET_MISMATCH) as ctx:
        await _add(revision=7)

    assert bridges.calls == []
    assert ctx["envelope"]["details"]["session_target"] == session
    assert ctx["envelope"]["details"]["baseline_target"] == baseline


async def test_a_single_lane_start_needs_no_lane(deployment):
    """A deployment with one lane cannot ask an agent which lane it meant."""
    bridges = deployment("live")

    assert (await _start())["started"] is True
    assert bridges.urls("/queue/start") == [f"{_LANE_ONE_URL}/queue/start"]


async def test_a_single_lane_start_accepts_the_name_of_its_own_lane(deployment):
    bridges = deployment("live")

    await _start(lane="bluesky")

    assert bridges.urls("/queue/start") == [f"{_LANE_ONE_URL}/queue/start"]


async def test_a_single_lane_refuses_a_lane_it_does_not_render(deployment):
    """Refused, never answered from the one bridge it does have."""
    bridges = deployment("live")

    with assert_raises_error(error_type=lanes_module.REASON_UNKNOWN_LANE) as ctx:
        await _start(lane="bluesky_va")

    assert bridges.calls == []
    assert "bluesky_va" in ctx["envelope"]["error_message"]


async def test_a_single_lane_result_still_names_its_lane(deployment):
    """PINNED: the lane id rides on every deployment's result, not only two-lane ones."""
    deployment("live")

    assert (await _add())["lane"] == "bluesky"
    assert (await _start())["lane"] == "bluesky"


async def test_a_single_lane_status_is_the_bridges_own_answer(deployment):
    """No lane roster, no `active` — the shape single-lane consumers already read."""
    deployment("live")

    status = await _status()

    assert status == _capability("bluesky", "live")
    assert "lanes" not in status


# =========================================================================
# Two lanes: the switch is an address, not a refusal
# =========================================================================

_ROUTING = [
    pytest.param("live", None, _LANE_ONE_URL, "bluesky", _LANE_ONE_TOKEN, id="session-on-baseline"),
    pytest.param("va", "va", _VA_LANE_URL, "bluesky_va", _VA_LANE_TOKEN, id="session-switched"),
]


@pytest.mark.parametrize(("session", "switch", "url", "lane", "token"), _ROUTING)
async def test_queue_add_routes_to_the_lane_serving_the_session(
    deployment, session, switch, url, lane, token
):
    """The whole point: a switched session queues on the OTHER lane, not nowhere."""
    bridges = deployment("live", "va")
    if switch is not None:
        _session_on("live")
        _switch_to(switch)

    result = await _add()

    assert bridges.urls("/queue/items") == [f"{url}/queue/items"]
    # Per-lane token: one token honoured by both lanes would let a launch
    # approved against one machine be replayed against the other.
    assert bridges.token_for("/queue/items") == token
    assert result["lane"] == lane


async def test_queue_start_on_the_bound_lane_proceeds(deployment):
    bridges = deployment("live", "va")
    _session_on("live")
    _switch_to("va")

    bound = (await _add())["lane"]
    result = await _start(lane=bound)

    assert bound == "bluesky_va"
    assert bridges.urls("/queue/start") == [f"{_VA_LANE_URL}/queue/start"]
    assert bridges.token_for("/queue/start") == _VA_LANE_TOKEN
    assert result["started"] is True


async def test_queue_start_without_a_lane_is_refused_on_a_two_lane_deployment(deployment):
    """Two lanes make "start the queue" an ambiguous instruction about hardware."""
    bridges = deployment("live", "va")

    with assert_raises_error(error_type=lanes_module.REASON_LANE_REQUIRED) as ctx:
        await _start()

    assert bridges.calls == []
    assert "lane" in ctx["envelope"]["error_message"]


async def test_queue_start_naming_the_inactive_lane_is_refused(deployment):
    bridges = deployment("live", "va")
    _session_on("live")
    _switch_to("va")

    with assert_raises_error(error_type=lanes_module.REASON_LANE_MISMATCH) as ctx:
        await _start(lane="bluesky")

    assert bridges.calls == []
    message = ctx["envelope"]["error_message"]
    # Both lanes are named: the one asked for and the one the session is on.
    assert "bluesky" in message and "bluesky_va" in message
    assert ctx["envelope"]["details"]["active_lane"] == "bluesky_va"


async def test_a_switch_between_the_add_and_the_start_refuses_the_start(deployment):
    """The mid-queue switch — the case the whole binding exists for.

    The item was composed for, and queued on, the lane the session was on. A
    start issued after the session moved would drain that queue against a
    machine nobody chose it for, so the bound lane and the active lane are
    compared afresh at start time.
    """
    bridges = deployment("live", "va")
    _session_on("live")
    _switch_to("va")
    bound = (await _add())["lane"]

    _switch_to("live", generation=2)

    with assert_raises_error(error_type=lanes_module.REASON_LANE_MISMATCH) as ctx:
        await _start(lane=bound)

    assert bridges.urls("/queue/start") == []
    details = ctx["envelope"]["details"]
    assert details["active_lane"] == "bluesky"
    assert details["session_target"] == "live"


async def test_a_lane_refusal_carries_the_whole_lane_board(deployment):
    """A refusal a reader can act on without a second call."""
    bridges = deployment("live", "va")
    _session_on("live")
    _switch_to("va")

    with assert_raises_error(error_type=lanes_module.REASON_LANE_MISMATCH) as ctx:
        await _start(lane="bluesky")

    assert bridges.calls == []
    details = ctx["envelope"]["details"]
    assert details["lanes"] == [
        {"lane": "bluesky", "lane_target": "live", "active": False},
        {"lane": "bluesky_va", "lane_target": "va", "active": True},
    ]
    # Same `{code, detail, capability}` shape every other queue refusal has, so
    # a consumer branching on details.code needs no special case.
    assert details["code"] == lanes_module.REASON_LANE_MISMATCH
    assert details["capability"] == {
        "can_execute": False,
        "reason": lanes_module.REASON_LANE_MISMATCH,
        "detail": details["detail"],
    }


async def test_naming_a_lane_no_deployment_renders_is_refused(deployment):
    bridges = deployment("live", "va")

    with assert_raises_error(error_type=lanes_module.REASON_UNKNOWN_LANE) as ctx:
        await _start(lane="bluesky_live")

    assert bridges.calls == []
    assert "bluesky_live" in ctx["envelope"]["error_message"]


async def test_no_lane_serving_the_session_target_refuses_rather_than_guesses(deployment):
    """A misrendered pair — both lanes on one target — must not pick a machine."""
    bridges = deployment(
        "live",
        "va",
        second_block={"path": "./services/bluesky", "port": _VA_LANE_PORT, "target": "live"},
    )
    _session_on("live")
    _switch_to("va")

    with assert_raises_error(error_type=qb.REASON_SESSION_TARGET_MISMATCH) as ctx:
        await _add()

    assert bridges.calls == []
    assert ctx["envelope"]["details"]["active_lane"] is None


async def test_a_lane_this_deployment_cannot_address_is_a_refusal_not_a_crash(deployment):
    """`UnknownBlueskyLaneError` surfaces as a structured refusal, never a 500.

    A lane declared in the config but with no port is a lane nothing can be
    sent to. The connection module raises rather than falling back to lane 1 —
    that fallback is the wrong-machine bug the lane axis exists to prevent — and
    the tool layer has to turn that into an answer an agent can read.
    """
    bridges = deployment("live", "va", second_block={"path": "./services/bluesky", "target": "va"})
    _session_on("live")
    _switch_to("va")

    with assert_raises_error(error_type=lanes_module.REASON_UNKNOWN_LANE) as ctx:
        await _add()

    assert bridges.calls == []
    assert "bluesky_va" in ctx["envelope"]["error_message"]


# =========================================================================
# Write posture is per lane, because it is per control target
# =========================================================================

#: The deployment this whole section is about: a real machine as its baseline,
#: writes off deployment-wide, and the simulator armed by its own block. The
#: facility that wants its agent to run plans against the virtual accelerator
#: without arming the storage ring — which one deployment-wide flag could not
#: express.
_LIVE_MACHINE_VA_ARMED = {
    "type": "epics",
    "writes_enabled": False,
    "connector": {
        "epics": {"gateways": {"read_only": {"host": "gw-ro"}}},
        "virtual_accelerator": {"writes_enabled": True},
    },
}


async def test_a_start_on_the_armed_lane_passes_the_posture_gate(deployment):
    """The point of per-target posture: the simulator lane starts while live does not."""
    # Arrange
    bridges = deployment("live", "va", control_system=_LIVE_MACHINE_VA_ARMED)
    _session_on("live")
    _switch_to("va")

    # Act
    result = await _start(lane="bluesky_va")

    # Assert
    assert result["started"] is True
    assert bridges.urls("/queue/start") == [f"{_VA_LANE_URL}/queue/start"]
    assert bridges.token_for("/queue/start") == _VA_LANE_TOKEN


async def test_a_start_on_the_unarmed_lane_names_that_lanes_own_posture_key(deployment):
    """The other half of the same config: the live lane is refused, by its own key.

    Naming ``control_system.connector.epics.writes_enabled`` rather than the
    deployment-wide key is what keeps the refusal actionable — an operator sent
    to the deployment-wide key would arm the machine they deliberately left
    unarmed, and they would arm it to run a plan on the other one.
    """
    # Arrange
    bridges = deployment("live", "va", control_system=_LIVE_MACHINE_VA_ARMED)

    # Act
    with assert_raises_error(error_type="writes_disabled") as ctx:
        await _start(lane="bluesky")

    # Assert
    assert bridges.calls == []
    message = ctx["envelope"]["error_message"]
    assert "control_system.connector.epics.writes_enabled" in message
    assert "'bluesky'" in message and "'live'" in message
    assert all("control_system.writes_enabled" not in s for s in ctx["envelope"]["suggestions"])


async def test_a_start_with_no_lane_hears_lane_required_before_any_posture(deployment):
    """Gate ORDER on a two-lane deployment: the lane is bound before the posture.

    The live lane here is unarmed, so a tool that read the posture first would
    answer ``writes_disabled`` — for a lane the caller never named, chosen for
    them. There is no target to read a posture for until a lane is bound, and
    saying so is the only answer that does not guess which machine was meant.
    """
    # Arrange
    bridges = deployment("live", "va", control_system=_LIVE_MACHINE_VA_ARMED)

    # Act
    with assert_raises_error(error_type=lanes_module.REASON_LANE_REQUIRED) as ctx:
        await _start()

    # Assert
    assert bridges.calls == []
    assert "writes_enabled" not in ctx["envelope"]["error_message"]


async def test_a_readonly_session_is_refused_even_on_the_armed_lane(deployment, monkeypatch):
    """A read-only run refuses whatever the deployment armed, and says so.

    The refusal must not send the operator to a config key: the simulator IS
    armed here, so editing config would change nothing about why this was
    refused.
    """
    # Arrange
    bridges = deployment("live", "va", control_system=_LIVE_MACHINE_VA_ARMED)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    _session_on("live")
    _switch_to("va")

    # Act
    with assert_raises_error(error_type="writes_disabled") as ctx:
        await _start(lane="bluesky_va")

    # Assert
    assert bridges.calls == []
    assert "OSPREY_EXECUTION_MODE=readonly" in ctx["envelope"]["error_message"]
    assert all("profile.yml" not in s for s in ctx["envelope"]["suggestions"])


async def test_withdrawing_a_halt_is_allowed_while_any_rendered_lane_is_armed(deployment):
    """``queue_stop(cancel=True)`` gates on the union over rendered lanes' targets.

    A withdrawal is addressed to whichever lane is actually draining, and that
    lane is only known after the bridges have been asked — so the local gate
    that runs first can only ask whether any lane this deployment renders is
    armed. Here the simulator lane is, so the withdrawal proceeds and the
    per-lane check that remains is the launch token.
    """
    # Arrange
    bridges = deployment("live", "va", control_system=_LIVE_MACHINE_VA_ARMED)

    # Act
    await get_tool_fn(queue.queue_stop)(cancel=True)

    # Assert
    assert bridges.urls("/queue/stop") == [f"{_LANE_ONE_URL}/queue/stop"]
    assert bridges.token_for("/queue/stop") == _LANE_ONE_TOKEN


async def test_withdrawing_a_halt_is_refused_when_no_rendered_lane_is_armed(deployment):
    """Negative control for the union gate: with nothing armed anywhere it refuses.

    Without this the test above would pass on a tool that had dropped the
    posture check from the withdrawal path entirely.
    """
    # Arrange
    bridges = deployment("live", "va", control_system={"type": "epics", "writes_enabled": False})

    # Act
    with assert_raises_error(error_type="writes_disabled") as ctx:
        await get_tool_fn(queue.queue_stop)(cancel=True)

    # Assert
    assert bridges.calls == []
    assert "control_system.writes_enabled" in ctx["envelope"]["error_message"]


async def test_a_phantom_live_target_cannot_arm_a_withdrawal_on_a_va_deployment(deployment):
    """The union is over the RENDERED LANES' targets, not over both target names.

    A virtual-accelerator deployment renders one lane, serving ``va``, and has
    no live machine: nothing in its config names one, so ``live`` resolves to no
    connector type and would inherit the deployment-wide key. Unioning over both
    target names would let that phantom target answer ``true`` and arm a
    withdrawal on the only lane there is — the one the operator explicitly
    disarmed with its own block. The bridge's stop endpoint has no posture
    check, so this local gate is the whole defense.
    """
    # Arrange
    bridges = deployment(
        "va",
        control_system={
            "type": "virtual_accelerator",
            "writes_enabled": True,
            "connector": {"virtual_accelerator": {"writes_enabled": False}},
        },
    )

    # Act
    with assert_raises_error(error_type="writes_disabled"):
        await get_tool_fn(queue.queue_stop)(cancel=True)

    # Assert
    assert bridges.calls == []


# =========================================================================
# Halting follows the RUN, not the session
# =========================================================================


async def test_an_abort_reaches_the_lane_that_is_running_after_a_switch(deployment):
    """The plan started on one lane, the session moved to the other, stop_run works.

    Halting must never be gated behind — or misdirected by — a target switch:
    the hardware moving is the hardware moving, whatever the session has since
    selected. Nobody should have to switch back before they can stop a plan.
    """
    bridges = deployment("live", "va", running="bluesky")
    _session_on("live")
    _switch_to("va")

    result = await _stop_run()

    assert bridges.urls("/queue/abort") == [f"{_LANE_ONE_URL}/queue/abort"]
    assert result["aborted"] is True


async def test_a_queue_stop_reaches_the_lane_that_is_draining_after_a_switch(deployment):
    """`queue_stop` takes the same posture as the abort, for the same reason."""
    bridges = deployment("live", "va", running="bluesky")
    _session_on("live")
    _switch_to("va")

    await get_tool_fn(queue.queue_stop)()

    assert bridges.urls("/queue/stop") == [f"{_LANE_ONE_URL}/queue/stop"]


async def test_a_halt_with_nothing_running_anywhere_goes_to_the_active_lane(deployment):
    """No motion to find: the halt still lands somewhere and gets an honest answer."""
    bridges = deployment("live", "va")
    _session_on("live")
    _switch_to("va")

    await _stop_run()

    assert bridges.urls("/queue/abort") == [f"{_VA_LANE_URL}/queue/abort"]


async def test_a_downed_lane_does_not_stop_the_halt_finding_the_running_one(deployment):
    """An unreachable lane is skipped, never allowed to swallow the emergency stop."""
    bridges = deployment("live", "va", running="bluesky")
    bridges.down.add(_VA_LANE_URL)
    _session_on("live")
    _switch_to("va")

    assert (await _stop_run())["aborted"] is True
    assert bridges.urls("/queue/abort") == [f"{_LANE_ONE_URL}/queue/abort"]


# =========================================================================
# Drafts: per lane, and so are their revisions
# =========================================================================


async def test_the_draft_the_agent_reads_is_the_active_lanes_own(deployment):
    """Two lanes hold two drafts, with independent revision counters.

    Revision 1 on one lane and revision 1 on the other are different plans, so
    the draft tools have to address the lane the session is on — otherwise an
    agent could pin a revision on one machine's draft and queue it on the
    other's.
    """
    bridges = deployment("live", "va")

    on_baseline = json.loads(await get_tool_fn(draft_tools.get_draft)())
    assert on_baseline["revision"] == 1
    assert on_baseline["draft"]["plan_name"] == "plan_on_bluesky"

    _session_on("live")
    _switch_to("va")

    after_switch = json.loads(await get_tool_fn(draft_tools.get_draft)())
    assert after_switch["revision"] == 7
    assert after_switch["draft"]["plan_name"] == "plan_on_bluesky_va"

    assert bridges.urls("/draft") == [f"{_LANE_ONE_URL}/draft", f"{_VA_LANE_URL}/draft"]
    # Each result names the lane it came from — half of the launch pin.
    assert on_baseline["lane"] == "bluesky"
    assert after_switch["lane"] == "bluesky_va"


async def test_the_same_revision_number_on_both_lanes_is_refused_not_guessed(deployment):
    """`(lane, revision)` is the pin; the revision alone collides across lanes.

    Both bridges happen to be at revision 4. An agent that read revision 4 from
    lane 1, then had the session switched under it, must not silently queue the
    OTHER lane's revision 4 — a different plan, for a different machine, behind
    the same number.
    """
    bridges = deployment("live", "va", revisions=(4, 4))
    from_lane_one = json.loads(await get_tool_fn(draft_tools.get_draft)())
    assert from_lane_one == {
        "draft": {"plan_name": "plan_on_bluesky"},
        "revision": 4,
        "lane": "bluesky",
    }

    _session_on("live")
    _switch_to("va")

    with assert_raises_error(error_type=lanes_module.REASON_LANE_MISMATCH) as ctx:
        await get_tool_fn(queue.queue_add)(draft_revision=4, lane=from_lane_one["lane"])

    assert bridges.urls("/queue/items") == []
    assert ctx["envelope"]["details"]["active_lane"] == "bluesky_va"


async def test_an_add_that_names_the_active_lane_proceeds(deployment):
    """The pin is a check, not a second way to be refused."""
    bridges = deployment("live", "va", revisions=(4, 4))
    _session_on("live")
    _switch_to("va")

    result = await get_tool_fn(queue.queue_add)(draft_revision=4, lane="bluesky_va")

    assert json.loads(result)["lane"] == "bluesky_va"
    assert bridges.urls("/queue/items") == [f"{_VA_LANE_URL}/queue/items"]


# =========================================================================
# Capability surfacing: the host composes what the bridge cannot know
# =========================================================================


async def test_queue_status_marks_exactly_one_lane_active(deployment):
    bridges = deployment("live", "va")
    _session_on("live")
    _switch_to("va")

    status = await _status()

    assert [(lane["lane"], lane["active"]) for lane in status["lanes"]] == [
        ("bluesky", False),
        ("bluesky_va", True),
    ]
    assert status["active_lane"] == "bluesky_va"
    # Composed ONTO the bridge's record: its static fields survive untouched.
    active = next(lane for lane in status["lanes"] if lane["active"])
    assert active["capability"]["lane_target"] == "va"
    assert active["capability"]["can_execute"] is True
    assert active["capability"]["active"] is True
    # The top-level pair is the active lane's, so a reader that ignores the
    # roster still sees the deployment this session is actually on.
    assert status["capability"] == active["capability"]
    assert set(bridges.urls("/health")) == {f"{_LANE_ONE_URL}/health", f"{_VA_LANE_URL}/health"}


async def test_with_no_state_file_the_baselines_lane_is_the_active_one(deployment):
    """No switch has happened, so the session is on the baseline — lane 1's target."""
    deployment("live", "va")

    status = await _status()

    assert status["active_lane"] == "bluesky"
    assert [lane["active"] for lane in status["lanes"]] == [True, False]


async def test_no_lane_is_active_when_none_serves_the_session_target(deployment):
    """Zero active lanes is a legal answer, and it says can_execute false."""
    deployment(
        "live",
        "va",
        second_block={"path": "./services/bluesky", "port": _VA_LANE_PORT, "target": "live"},
    )
    _session_on("live")
    _switch_to("va")

    status = await _status()

    assert [lane["active"] for lane in status["lanes"]] == [False, False]
    assert status["active_lane"] is None
    assert status["capability"]["can_execute"] is False
    assert status["capability"]["reason"] == qb.REASON_SESSION_TARGET_MISMATCH


async def test_a_downed_inactive_lane_does_not_hide_the_board(deployment):
    """One lane's bad news is one entry's `error`, not the whole answer's failure."""
    bridges = deployment("live", "va")
    bridges.down.add(_VA_LANE_URL)

    status = await _status()

    (down,) = [lane for lane in status["lanes"] if lane["lane"] == "bluesky_va"]
    assert down["error"]
    assert "capability" not in down
    # The active lane is healthy and still reported, which is the whole point.
    assert status["active_lane"] == "bluesky"
    assert status["capability"]["can_execute"] is True


async def test_a_downed_active_lane_is_still_a_refusal(deployment):
    """An unreadable capability where the session IS must never read as executable."""
    bridges = deployment("live", "va")
    bridges.down.add(_LANE_ONE_URL)

    with assert_raises_error(error_type="bluesky_bridge_error"):
        await _status()


# =========================================================================
# The wire vocabulary
# =========================================================================


def test_the_lane_reason_codes_are_the_literal_wire_strings():
    """Host-only codes, but still a vocabulary consumers match on: pin the spelling."""
    assert (
        lanes_module.REASON_UNKNOWN_LANE,
        lanes_module.REASON_LANE_REQUIRED,
        lanes_module.REASON_LANE_MISMATCH,
    ) == ("unknown_bluesky_lane", "lane_required", "lane_mismatch")


# =========================================================================
# The sidecar's per-lane URL map
# =========================================================================


async def test_the_sidecar_publishes_a_lane_url_map_on_a_two_lane_deployment(deployment):
    """`app.state.bridge_urls` is what makes the read proxy's `?lane=` resolvable."""
    from osprey.interfaces.bluesky_web.app import app

    deployment("live", "va")
    try:
        async with app.router.lifespan_context(app):
            assert app.state.bridge_urls == {
                "bluesky": _LANE_ONE_URL,
                "bluesky_va": _VA_LANE_URL,
            }
            assert app.state.bridge_url == _LANE_ONE_URL
    finally:
        # The app is a module-level singleton; leaving the map behind would
        # make a later single-lane test read this one's deployment.
        app.state._state.pop("bridge_urls", None)


async def test_the_sidecar_publishes_no_lane_map_on_a_single_lane_deployment(deployment):
    """Byte-identical to what a single-lane sidecar has always published."""
    from osprey.interfaces.bluesky_web.app import app

    deployment("live")
    app.state._state.pop("bridge_urls", None)

    async with app.router.lifespan_context(app):
        assert not hasattr(app.state, "bridge_urls")
        assert app.state.bridge_url == _LANE_ONE_URL


def test_the_lane_url_map_is_empty_for_one_lane_and_populated_for_two(deployment):
    """The helper behind the wiring, asserted on the config rather than the app."""
    deployment("live")
    assert resolve_lane_bridge_urls() == {}

    deployment("live", "va")
    assert resolve_lane_bridge_urls() == {"bluesky": _LANE_ONE_URL, "bluesky_va": _VA_LANE_URL}
