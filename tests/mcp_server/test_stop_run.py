"""Unit tests for the ``stop_run`` MCP tool — the emergency abort.

``stop_run`` posts ``POST /queue/abort``: it stops the plan already moving
hardware, where ``queue_stop`` only halts the queue after the running item
finishes. Three properties carry the weight here and are asserted directly:

1. **Nothing gates it.** No ``control_system.writes_enabled`` re-read, no
   launch token, in any deployment posture — the same posture as a plain
   ``queue_stop``, and for the same reason: a halt that can be refused for a
   policy reason is a halt with a failure mode. The tests below put the process
   in the *most* refusable posture available (writes disabled, no token) and
   assert the HTTP call still goes out bare.
2. **Refusals are relayed verbatim.** The bridge's ``code`` becomes the
   envelope's ``error_type``, its sentence the ``error_message``, and the whole
   detail dict the ``details`` — nothing is reworded, and in particular an
   ``abort_pause_timeout`` must reach the agent still saying that nothing was
   aborted.
3. **The docstring cannot drift back.** ``test_stop_run_docstring_matches_the
   _capability_it_now_has`` is the inverse-drift pin (same shape as
   ``test_get_run_docstring_does_not_still_promise_retired_keys``): this tool
   spent a release documented as NOT FUNCTIONAL, and that wording next to a
   working abort is dangerous in the direction that costs the most — an agent
   told not to reach for the only tool that stops a moving plan.

The HTTP boundary (``_http_post_json``) is patched here so these run with no
Bluesky bridge process and no network.
"""

from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.bluesky import server_context
from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import stop
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.stop"

_ABORT_OK = {
    "aborted": True,
    "abort_pending": False,
    "paused_first": True,
    "manager_state": "idle",
    "msg": "plan aborted",
}


@pytest.fixture(autouse=True)
def _reset_bluesky_context():
    yield
    reset_server_context()


def _fn():
    return get_tool_fn(stop.stop_run)


def _locked_down(tmp_path, monkeypatch) -> None:
    """The most refusable posture a deployment can be in: writes disabled by
    the kill switch and no launch token anywhere. Every other Bluesky write
    tool refuses here; the abort must not."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({"control_system": {"writes_enabled": False}}))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    initialize_server_context()


async def test_stop_run_aborts_the_running_plan():
    with patch(f"{_MOD}._http_post_json", return_value=(200, _ABORT_OK)) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _fn()()

    assert post.call_args.args[0] == "/queue/abort"
    assert post.call_args.args[1] == {}
    data = extract_response_dict(result)
    assert data["aborted"] is True
    assert data["msg"] == "plan aborted"


async def test_stop_run_budgets_above_the_bridges_composed_abort():
    """The shared 15s HTTP timeout is shorter than the abort the bridge composes.

    The bridge's abort is ~10s of poll sleeps PLUS ~43 manager round trips, each
    bounded by the queueserver client's 2.0s recv timeout — ~96s worst case (the
    derivation is in ``stop._ABORT_TIMEOUT``). At 15s a slow-but-answering
    manager makes this tool report ``bluesky_bridge_unreachable`` while the
    bridge is mid-abort: the agent is told the bridge is down when the halt is
    actually in flight, and told to retry into a race.
    """
    with patch(f"{_MOD}._http_post_json", return_value=(200, _ABORT_OK)) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _fn()()

    assert post.call_args.kwargs["timeout"] == stop._ABORT_TIMEOUT
    assert stop._ABORT_TIMEOUT > 96.0, "must exceed the bridge's composed worst case"
    assert stop._ABORT_TIMEOUT > server_context._TIMEOUT


async def test_stop_run_takes_no_arguments():
    """A run id would only add a way for a halt to be refused: the queue server
    runs one plan at a time, so there is nothing to select. Pinned on the
    signature because an argument added later would be silently accepted and
    ignored, which reads to an agent as selection that does not exist."""
    import inspect

    signature = inspect.signature(get_tool_fn(stop.stop_run))
    assert list(signature.parameters) == []


async def test_stop_run_is_ungated_with_writes_off_and_no_token(tmp_path, monkeypatch):
    """The core safety property. Writes disabled + no token is exactly when an
    operator is most likely to want a run stopped, and the request must still
    go out — bare, with no arming header for the bridge to reject."""
    _locked_down(tmp_path, monkeypatch)

    with patch(f"{_MOD}._http_post_json", return_value=(200, _ABORT_OK)) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _fn()()

    assert extract_response_dict(result)["aborted"] is True
    assert post.call_count == 1
    # `_http_post_json(path, payload)` — the path and payload are positional, so
    # any header would have to arrive as the `headers` keyword. There is none,
    # on purpose. Asserted by NAME rather than as "no keywords at all": the call
    # legitimately carries a `timeout` budget (see
    # `test_stop_run_budgets_above_the_bridges_composed_abort`), and a
    # keyword-count assertion would conflate a transport budget with an arming
    # credential — the one thing this test exists to forbid.
    assert "headers" not in post.call_args.kwargs
    # `lane` joined `timeout` on the same footing: it is an ADDRESS (which plan
    # lane's bridge), not a credential, and on a single-lane deployment it is
    # None — the only lane there is. Same reasoning as the timeout budget: this
    # assertion forbids arming headers, not transport arguments.
    assert set(post.call_args.kwargs) == {"lane", "timeout"}
    assert post.call_args.kwargs["lane"] is None


async def test_stop_run_does_not_read_the_writes_kill_switch(tmp_path, monkeypatch):
    """Negative control for the property above, at the mechanism rather than
    the outcome: `queue._writes_enabled` is the shared re-read every armed tool
    performs, and the abort must never consult it — a tool that read it and
    ignored the answer would be one edit from gating a halt."""
    _locked_down(tmp_path, monkeypatch)

    with patch("osprey.mcp_server.bluesky.tools.queue._writes_enabled") as writes:
        with patch(f"{_MOD}._http_post_json", return_value=(200, _ABORT_OK)):
            with patch(f"{_MOD}.notify_agent_activity_async"):
                await _fn()()

    assert writes.call_count == 0


async def test_stop_run_relays_nothing_running_verbatim():
    refusal = {
        "detail": {
            "code": "nothing_running",
            "detail": "No plan is running, so there is nothing to abort (manager state 'idle').",
        }
    }
    with patch(f"{_MOD}._http_post_json", return_value=(409, refusal)):
        with assert_raises_error(error_type="nothing_running") as ctx:
            await _fn()()

    assert "nothing to abort" in ctx["envelope"]["error_message"]
    assert ctx["envelope"]["details"] == refusal["detail"]


async def test_stop_run_relays_an_abort_timeout_still_saying_nothing_stopped():
    """The refusal that must survive every hop unsoftened: the plan may
    still be running, and the bridge's sentence is the only thing that says so."""
    sentence = (
        "The Run Engine did not reach the paused state an abort requires within 10s, "
        "so NOTHING WAS ABORTED and the plan may still be running."
    )
    refusal = {"detail": {"code": "abort_pause_timeout", "detail": sentence}}
    with patch(f"{_MOD}._http_post_json", return_value=(503, refusal)):
        with assert_raises_error(error_type="abort_pause_timeout") as ctx:
            await _fn()()

    assert ctx["envelope"]["error_message"] == sentence
    assert "NOTHING WAS ABORTED" in ctx["envelope"]["error_message"]
    assert any("may still be running" in hint for hint in ctx["envelope"]["suggestions"])


async def test_stop_run_relays_an_unreachable_manager():
    refusal = {
        "detail": {"code": "manager_unreachable", "detail": "The queue server is not answering."}
    }
    with patch(f"{_MOD}._http_post_json", return_value=(503, refusal)):
        with assert_raises_error(error_type="manager_unreachable"):
            await _fn()()


async def test_stop_run_falls_back_when_the_bridge_sends_no_structured_detail():
    """An unhandled 500 or a proxy error page has no code to relay. Inventing
    one would put the agent on a vocabulary the bridge never used."""
    with patch(f"{_MOD}._http_post_json", return_value=(500, {"detail": "boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _fn()()

    assert "boom" in ctx["envelope"]["error_message"]


def test_stop_run_docstring_matches_its_actual_capability():
    """Inverse drift, in the direction that costs the most.

    Wording that documents this tool as NOT FUNCTIONAL, as it would be behind a
    410 route, tells an agent not to reach for the only tool that stops a moving
    plan — worse than silence, at the moment delay is most expensive. Positive
    halves pinned too, so a docstring merely emptied of those words fails as
    well.
    """
    doc = (stop.stop_run.__doc__ or "") + (stop.__doc__ or "")
    lowered = doc.lower()

    for retired in ("not functional", "retired", "stops nothing", "cannot stop", "dead tool"):
        assert retired not in lowered, (
            f"stop_run's documentation still carries {retired!r} — it now aborts the "
            f"running plan, and that wording steers an agent away from a working halt"
        )

    assert "abort" in lowered
    assert "/queue/abort" in doc, "the docstring must name the route it actually calls"
    assert "abort_pause_timeout" in doc, (
        "the failure that leaves a plan running must be documented, not just handled"
    )
    assert "ungated" in lowered, "the halt's ungated posture is the property an agent relies on"
