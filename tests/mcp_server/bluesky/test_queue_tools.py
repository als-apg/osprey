"""Tests for the Bluesky queue MCP tools: ``queue_status`` /
``queue_list`` / ``queue_add`` / ``queue_start`` / ``queue_stop``.

Three properties are load-bearing and are asserted directly, never inferred:

1. **The kill switch beats a valid token, with zero HTTP calls.**
   ``queue_start`` and ``queue_stop(cancel=True)`` re-read
   ``control_system.writes_enabled`` fresh from config before any network call,
   so a caller that bypassed the PreToolUse hook and holds a correct
   ``BLUESKY_LAUNCH_TOKEN`` is still refused. ``queue_add`` applies the same
   switch to the armed half of enqueue by WITHHOLDING the token header — the
   assertion there is on the header, not on the status code, because that is
   the mechanism.

2. **Refusal bodies, not just status codes.** Every bridge refusal is
   ``{"code", "detail", ...extras}``; these tools must relay the code and
   sentence verbatim and carry the extras (``capability``, ``manager_state``,
   ``revision``, ``plan``, ``item_left_behind``) through untouched. Tests
   assert the envelope's ``error_type`` equals the bridge's code and that
   ``details`` still holds the extras — a status-code-only assertion would
   pass while the body drifted.

3. **Halting is never gated.** A plain ``queue_stop`` works with writes
   disabled and no token at all.

The HTTP boundary (``..tools.queue._http_get_json`` / ``_http_post_json``) is
patched, so these exercise this module's gating, payload shaping and
error-envelope mapping against the queue wire contract with no bridge process.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import queue
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.queue"

_TOKEN = "genuinely-valid-token"


@pytest.fixture(autouse=True)
def _reset_bluesky_context():
    yield
    reset_server_context()


def _configure(tmp_path, monkeypatch, *, writes: bool | None, token: str | None) -> None:
    """Put the process in a deployment posture: writes on/off, token set/unset.

    ``writes=None`` writes no config.yml at all — the fail-closed case.
    """
    monkeypatch.chdir(tmp_path)
    if writes is not None:
        (tmp_path / "config.yml").write_text(
            yaml.dump({"control_system": {"writes_enabled": writes}})
        )
    if token is None:
        monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", token)
    initialize_server_context()


def _armed(tmp_path, monkeypatch) -> None:
    """The fully-armed posture: writes enabled and a valid token configured."""
    _configure(tmp_path, monkeypatch, writes=True, token=_TOKEN)


def _status_fn():
    return get_tool_fn(queue.queue_status)


def _list_fn():
    return get_tool_fn(queue.queue_list)


def _add_fn():
    return get_tool_fn(queue.queue_add)


def _start_fn():
    return get_tool_fn(queue.queue_start)


def _stop_fn():
    return get_tool_fn(queue.queue_stop)


def _refusal(code: str, detail: str, **extras) -> dict:
    """A bridge refusal body: the wire contract's nested ``detail`` dict."""
    return {"detail": {"code": code, "detail": detail, **extras}}


_BROWSE_ONLY = {
    "can_execute": False,
    "reason": "browse_only_connector",
    "detail": (
        "This deployment uses the mock connector, which cannot move hardware, so scan "
        "plans can be composed and validated but not executed. To execute plans, run "
        "`osprey config set-control-system virtual_accelerator` and redeploy."
    ),
}


# =========================================================================
# queue_status — capability, relayed whole
# =========================================================================


async def test_queue_status_reads_health_and_relays_the_capability_record(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    body = {"status": "ok", "capability": _BROWSE_ONLY}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _status_fn()()

    assert m.call_args.args[0] == "/health"
    assert extract_response_dict(result) == body


async def test_queue_status_does_not_gate_liveness_on_can_execute(tmp_path, monkeypatch):
    """Invariant (a): ``status: "ok"`` is independent of ``can_execute``.

    A browse-only deployment is a healthy deployment. The tool must return the
    bridge's 200 as a success, not convert a cannot-execute capability into an
    error envelope — the agent needs the ``detail`` sentence to relay.
    """
    _armed(tmp_path, monkeypatch)
    with patch(
        f"{_MOD}._http_get_json", return_value=(200, {"status": "ok", "capability": _BROWSE_ONLY})
    ):
        result = await _status_fn()()

    parsed = extract_response_dict(result)
    assert parsed["status"] == "ok"
    assert parsed["capability"]["can_execute"] is False
    assert "set-control-system virtual_accelerator" in parsed["capability"]["detail"]
    assert "error" not in parsed


async def test_queue_status_executable_deployment_passes_through(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    capability = {
        "can_execute": True,
        "reason": "executable",
        "detail": "Plans execute against the 'virtual_accelerator' connector.",
    }
    with patch(
        f"{_MOD}._http_get_json", return_value=(200, {"status": "ok", "capability": capability})
    ):
        result = await _status_fn()()
    assert extract_response_dict(result)["capability"] == capability


async def test_queue_status_non_200_is_an_error_that_says_treat_as_cannot_execute(
    tmp_path, monkeypatch
):
    """Invariant (b): a non-200 /health must be read as cannot-execute.

    The tool cannot answer the question, so it must not return something the
    agent could mistake for an answer — and its guidance has to say which way
    to fail.
    """
    _armed(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_get_json", return_value=(502, {"detail": "bad gateway"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _status_fn()()

    assert "bad gateway" in ctx["envelope"]["error_message"]
    assert any("unable to execute" in s for s in ctx["envelope"]["suggestions"])


# =========================================================================
# queue_list — the queue as the manager holds it
# =========================================================================


async def test_queue_list_reads_the_queue_route_and_relays_the_body(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    body = {
        "status": {"available": True, "manager_state": "idle", "items_in_queue": 1},
        "items": [{"item_uid": "u1", "name": "grid_scan", "kwargs": {"num": 3}}],
        "running_item": None,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _list_fn()()

    assert m.call_args.args[0] == "/queue"
    assert extract_response_dict(result) == body


async def test_queue_list_relays_indeterminate_progress_without_inventing_a_fraction(
    tmp_path, monkeypatch
):
    """``fraction: null`` means unknown; the tool must not normalise it to 0."""
    _armed(tmp_path, monkeypatch)
    body = {
        "status": {"available": True, "manager_state": "executing_queue"},
        "items": [],
        "running_item": {"item_uid": "u1", "progress": {"fraction": None, "rows_seen": 12}},
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _list_fn()()
    assert extract_response_dict(result)["running_item"]["progress"] == {
        "fraction": None,
        "rows_seen": 12,
    }


async def test_queue_list_relays_the_bridge_refusal_code_and_detail(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    body = _refusal("manager_unreachable", "The queue manager did not answer within 5s.")
    with patch(f"{_MOD}._http_get_json", return_value=(503, body)):
        with assert_raises_error(error_type="manager_unreachable") as ctx:
            await _list_fn()()

    envelope = ctx["envelope"]
    assert envelope["error_message"] == "The queue manager did not answer within 5s."
    assert envelope["details"]["code"] == "manager_unreachable"


# =========================================================================
# queue_add — enqueue is armed only sometimes
# =========================================================================


async def test_queue_add_posts_the_pinned_revision_with_the_token_when_armed(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    body = {"run_id": "abc123", "revision": 7, "item": {"item_uid": "u1"}}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _add_fn()(draft_revision=7)

    assert m.call_args.args[0] == "/queue/items"
    assert m.call_args.args[1] == {"draft_revision": 7}
    assert m.call_args.kwargs["headers"] == {"X-Launch-Token": _TOKEN}
    assert extract_response_dict(result) == body


async def test_queue_add_withholds_the_token_when_writes_are_disabled(tmp_path, monkeypatch):
    """The kill switch applied to enqueue: composing is allowed, arming is not.

    With writes off the request still goes out — an item on an idle queue moves
    nothing — but WITHOUT the launch token, so the bridge (which alone knows the
    manager's live state) permits it only while the queue is idle and refuses
    it the moment the queue is draining. Asserting on the absent header pins
    the mechanism; a status-code assertion would not.
    """
    _configure(tmp_path, monkeypatch, writes=False, token=_TOKEN)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=3)

    assert m.call_args.kwargs["headers"] is None


async def test_queue_add_missing_config_fails_closed_and_withholds_the_token(tmp_path, monkeypatch):
    """No config.yml at all is not "writes enabled" by omission."""
    _configure(tmp_path, monkeypatch, writes=None, token=_TOKEN)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=3)

    assert m.call_args.kwargs["headers"] is None


async def test_queue_add_without_a_configured_token_still_composes(tmp_path, monkeypatch):
    """An unarmed deployment may still build a queue; only starting it is gated."""
    _configure(tmp_path, monkeypatch, writes=True, token=None)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=3)

    m.assert_called_once()
    assert m.call_args.kwargs["headers"] is None


async def test_queue_add_armed_refusal_relays_code_manager_state_and_stranded_item(
    tmp_path, monkeypatch
):
    """The enqueue-while-running refusal, body and all.

    ``item_left_behind``/``item_uid`` say an unarmed item could NOT be
    withdrawn from an armed queue — the one extra a human must act on, so it
    has to survive the trip through this tool.
    """
    _armed(tmp_path, monkeypatch)
    body = _refusal(
        "launch_token_required",
        "the queue is running, starting, or set to autostart, so adding an item "
        "requires the launch token: missing or invalid launch token",
        manager_state="executing_queue",
        item_left_behind=True,
        item_uid="u9",
    )
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)):
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _add_fn()(draft_revision=7)

    details = ctx["envelope"]["details"]
    assert details["manager_state"] == "executing_queue"
    assert details["item_left_behind"] is True
    assert details["item_uid"] == "u9"
    assert ctx["envelope"]["error_message"] == body["detail"]["detail"]


async def test_queue_add_armed_refusal_names_the_kill_switch_when_writes_are_off(
    tmp_path, monkeypatch
):
    """Writes-off is WHY the token was withheld — say so without rewriting the code.

    The bridge's ``launch_token_required`` is relayed unchanged (that is what
    happened on the wire); the extra suggestion adds the piece the bridge could
    not know, so nobody goes hunting for a token that would change nothing.
    """
    _configure(tmp_path, monkeypatch, writes=False, token=_TOKEN)
    body = _refusal(
        "launch_token_required",
        "the queue is running, so adding an item requires the launch token",
        manager_state="executing_queue",
    )
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)):
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _add_fn()(draft_revision=7)

    envelope = ctx["envelope"]
    assert envelope["details"]["code"] == "launch_token_required"
    assert any("writes_enabled" in s for s in envelope["suggestions"])


async def test_queue_add_writes_enabled_refusal_does_not_blame_the_kill_switch(
    tmp_path, monkeypatch
):
    """Negative control for the hint above: with writes ON it must be ABSENT.

    Otherwise the previous test would pass on a tool that always appends it.
    """
    _armed(tmp_path, monkeypatch)
    body = _refusal("launch_token_required", "requires the launch token", manager_state="paused")
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)):
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _add_fn()(draft_revision=7)

    assert not any("writes_enabled" in s for s in ctx["envelope"]["suggestions"])


async def test_queue_add_stale_revision_relays_the_fresh_baseline(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    body = _refusal("stale_draft_revision", "the draft has moved on", revision=9)
    with patch(f"{_MOD}._http_post_json", return_value=(409, body)):
        with assert_raises_error(error_type="stale_draft_revision") as ctx:
            await _add_fn()(draft_revision=7)

    assert ctx["envelope"]["details"]["revision"] == 9
    assert any("get_draft" in s for s in ctx["envelope"]["suggestions"])


async def test_queue_add_already_launched_revision_points_at_a_draft_edit(tmp_path, monkeypatch):
    """The repeat-scan case: a consumed revision needs a NEW one, not a retry."""
    _armed(tmp_path, monkeypatch)
    body = _refusal("draft_revision_already_launched", "revision 7 was already queued", revision=7)
    with patch(f"{_MOD}._http_post_json", return_value=(409, body)):
        with assert_raises_error(error_type="draft_revision_already_launched") as ctx:
            await _add_fn()(draft_revision=7)

    assert any("set_draft" in s for s in ctx["envelope"]["suggestions"])


async def test_queue_add_unknown_device_points_at_the_device_list(tmp_path, monkeypatch):
    """The name is wrong, not the revision. Without its own hints this code
    falls back to "re-read the draft with get_draft", which sends the agent to
    re-read a draft that says exactly what it said before — so the hints must
    name list_devices, and the available set must survive to the caller."""
    _armed(tmp_path, monkeypatch)
    body = _refusal(
        "unknown_device",
        "plan 'grid_scan' referenced device 'COR9', which this worker did not build; "
        "available devices: ['BPM1', 'COR1']",
        plan="grid_scan",
        devices=["COR9"],
        available_devices=["BPM1", "COR1"],
    )
    with patch(f"{_MOD}._http_post_json", return_value=(400, body)):
        with assert_raises_error(error_type="unknown_device") as ctx:
            await _add_fn()(draft_revision=7)

    envelope = ctx["envelope"]
    assert envelope["details"]["available_devices"] == ["BPM1", "COR1"]
    assert any("list_devices" in s for s in envelope["suggestions"])
    assert not any("get_draft" in s for s in envelope["suggestions"])


@pytest.mark.parametrize("code", ["session_plan_unvalidated", "session_plan_not_in_namespace"])
async def test_queue_add_session_plan_refusal_names_the_offending_plan(tmp_path, monkeypatch, code):
    _armed(tmp_path, monkeypatch)
    body = _refusal(code, "orbit_sweep is not admissible", plan="orbit_sweep", reason=code)
    with patch(f"{_MOD}._http_post_json", return_value=(409, body)):
        with assert_raises_error(error_type=code) as ctx:
            await _add_fn()(draft_revision=7)

    assert ctx["envelope"]["details"]["plan"] == "orbit_sweep"
    assert any("validate_plan" in s for s in ctx["envelope"]["suggestions"])


async def test_queue_add_capability_refusal_relays_the_whole_capability_record(
    tmp_path, monkeypatch
):
    """A browse-only deployment never holds items — and the flip command must survive."""
    _armed(tmp_path, monkeypatch)
    body = _refusal(
        "browse_only_connector",
        "This deployment cannot execute plans.",
        capability=_BROWSE_ONLY,
    )
    with patch(f"{_MOD}._http_post_json", return_value=(409, body)):
        with assert_raises_error(error_type="browse_only_connector") as ctx:
            await _add_fn()(draft_revision=7)

    assert ctx["envelope"]["details"]["capability"] == _BROWSE_ONLY


async def test_queue_add_unstructured_error_body_falls_back_without_inventing_a_code(
    tmp_path, monkeypatch
):
    """No ``code`` on the wire means no code to relay — never a fabricated one."""
    _armed(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_post_json", return_value=(500, {"detail": "internal error"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _add_fn()(draft_revision=7)

    assert "internal error" in ctx["envelope"]["error_message"]
    assert "details" not in ctx["envelope"]


async def test_queue_add_emits_agent_activity_only_after_a_confirmed_enqueue(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    with patch(
        f"{_MOD}._http_post_json", return_value=(409, _refusal("stale_draft_revision", "no"))
    ):
        with patch(f"{_MOD}.notify_agent_activity_async") as notify:
            with assert_raises_error(error_type="stale_draft_revision"):
                await _add_fn()(draft_revision=7)
    notify.assert_not_called()

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "abc123"})):
        with patch(f"{_MOD}.notify_agent_activity_async") as notify:
            await _add_fn()(draft_revision=7)
    assert notify.call_args.kwargs["detail"] == "abc123"


# =========================================================================
# queue_start — the arming action
# =========================================================================


async def test_queue_start_writes_disabled_refuses_a_valid_token_with_zero_http_calls(
    tmp_path, monkeypatch
):
    """THE load-bearing safety proof for the queue surface. Do NOT relax this test.

    A hook-bypassed caller holding a correct BLUESKY_LAUNCH_TOKEN must not be
    able to start the queue while this deployment has writes disabled. The
    in-tool re-check must reject before ``_http_post_json`` is ever invoked —
    asserted directly, not inferred from the error type — and the refusal must
    not echo the token back.
    """
    _configure(tmp_path, monkeypatch, writes=False, token=_TOKEN)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _start_fn()()

    mock_post.assert_not_called()
    assert _TOKEN not in ctx["envelope"]["error_message"]


async def test_queue_start_missing_config_fails_closed(tmp_path, monkeypatch):
    """No config.yml is not "writes enabled" by omission, token notwithstanding."""
    _configure(tmp_path, monkeypatch, writes=None, token=_TOKEN)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled"):
            await _start_fn()()
    mock_post.assert_not_called()


async def test_queue_start_without_a_token_files_a_panel_start_request(tmp_path, monkeypatch):
    """The tokenless deployment posture: queue_start hands the decision to the human.

    In a deployed web terminal the launch token lives with the operator panels
    sidecar and never enters the agent's environment — so a tokenless
    queue_start must not dead-end in a refusal that reads like a config bug.
    It files ``POST /queue/start-request`` (no token header: the route arms
    nothing), and the result tells the agent exactly what happens next: the
    operator confirms in the queue panel. ``/queue/start`` is never called.
    """
    _configure(tmp_path, monkeypatch, writes=True, token=None)
    record = {
        "request_id": "abc123",
        "requested_by": "agent",
        "requested_at": "2026-08-12T20:00:00+00:00",
        "items_in_queue": 2,
    }
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"start_request": record})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _start_fn()()

    assert m.call_args.args[0] == "/queue/start-request"
    assert m.call_args.kwargs["headers"] is None
    body = extract_response_dict(result)
    assert body["started"] is False
    assert body["start_request"] == record
    assert "queue panel" in body["message"]


async def test_a_tokenless_start_request_relays_bridge_refusals(tmp_path, monkeypatch):
    """An empty queue refuses the request with the bridge's own code, so the
    agent hears "enqueue first", not a token problem."""
    _configure(tmp_path, monkeypatch, writes=True, token=None)
    body = _refusal("queue_empty", "the queue holds no items")
    with patch(f"{_MOD}._http_post_json", return_value=(409, body)):
        with assert_raises_error(error_type="queue_empty") as ctx:
            await _start_fn()()

    assert ctx["envelope"]["details"]["code"] == "queue_empty"


async def test_a_tokenless_start_still_respects_the_kill_switch(tmp_path, monkeypatch):
    """Writes disabled refuses BEFORE any start request is filed: a request the
    panel shows as confirmable must never exist on a writes-off deployment."""
    _configure(tmp_path, monkeypatch, writes=False, token=None)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled"):
            await _start_fn()()
    mock_post.assert_not_called()


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [("queue_start", {}), ("queue_stop", {"cancel": True})],
    ids=["queue_start", "queue_stop-cancel"],
)
async def test_arming_ops_report_the_kill_switch_before_the_missing_token(
    tmp_path, monkeypatch, tool, kwargs
):
    """Gate ORDER, not just gate presence: writes-off is reported first.

    With BOTH gates failing — writes disabled AND no token — the refusal must
    name ``writes_disabled``, never ``launch_token_required``. Both refuse, so
    no test that only checks "it refused" can tell the order; but the two send
    an operator to opposite places. Reporting the missing token would have them
    chase a credential that changes nothing while the kill switch is off, at
    the moment writes have deliberately been disabled.
    """
    _configure(tmp_path, monkeypatch, writes=False, token=None)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await get_tool_fn(getattr(queue, tool))(**kwargs)

    mock_post.assert_not_called()
    assert "writes_enabled" in ctx["envelope"]["error_message"]


async def test_queue_start_armed_posts_with_the_token(tmp_path, monkeypatch):
    """Contrast case: proves the refusals above are gated, not vacuous."""
    _armed(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True, "msg": ""})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _start_fn()()

    assert m.call_args.args[0] == "/queue/start"
    assert m.call_args.kwargs["headers"] == {"X-Launch-Token": _TOKEN}
    assert extract_response_dict(result)["started"] is True


async def test_queue_start_session_plan_refusal_names_the_blocking_plan(tmp_path, monkeypatch):
    """One stale session plan refuses the whole start — the agent needs to know which."""
    _armed(tmp_path, monkeypatch)
    body = _refusal(
        "session_plan_unvalidated",
        "orbit_sweep has no current passing validation",
        plan="orbit_sweep",
    )
    with patch(f"{_MOD}._http_post_json", return_value=(409, body)):
        with assert_raises_error(error_type="session_plan_unvalidated") as ctx:
            await _start_fn()()

    assert ctx["envelope"]["details"]["plan"] == "orbit_sweep"


async def test_queue_start_bridge_token_mismatch_relays_launch_token_required(
    tmp_path, monkeypatch
):
    """The bridge's own token verdict is relayed in the same vocabulary."""
    _armed(tmp_path, monkeypatch)
    body = _refusal(
        "launch_token_required", "starting the queue requires the launch token: invalid token"
    )
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)):
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _start_fn()()

    assert ctx["envelope"]["details"]["code"] == "launch_token_required"


async def test_queue_start_environment_unavailable_is_relayed_as_retryable(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    body = _refusal("environment_unavailable", "the worker environment could not be opened")
    with patch(f"{_MOD}._http_post_json", return_value=(503, body)):
        with assert_raises_error(error_type="environment_unavailable") as ctx:
            await _start_fn()()

    assert any("Retry" in s for s in ctx["envelope"]["suggestions"])


# =========================================================================
# queue_stop — halting is free, un-halting is armed
# =========================================================================


async def test_plain_queue_stop_works_with_writes_disabled_and_no_token(tmp_path, monkeypatch):
    """Halting is the safe direction and must never be blocked by the kill switch.

    The worst posture available — writes off, no token — must still reach the
    bridge, and must not smuggle a credential onto an ungated request.
    """
    _configure(tmp_path, monkeypatch, writes=False, token=None)
    with patch(
        f"{_MOD}._http_post_json", return_value=(200, {"stop_pending": True, "msg": ""})
    ) as m:
        result = await _stop_fn()()

    assert m.call_args.args[0] == "/queue/stop"
    assert m.call_args.args[1] == {"cancel": False}
    assert m.call_args.kwargs["headers"] is None
    assert extract_response_dict(result)["stop_pending"] is True


async def test_plain_queue_stop_does_not_even_consult_the_kill_switch(tmp_path, monkeypatch):
    """Ungated means ungated: a plain stop never reads writes_enabled at all.

    Stronger than "it happens to succeed with writes off" — this pins that no
    gate exists on the halting path, so a future change that starts consulting
    the kill switch here (and could then refuse a stop when config is
    unreadable) fails immediately rather than at the worst possible moment.
    """
    _configure(tmp_path, monkeypatch, writes=False, token=_TOKEN)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"stop_pending": True})):
        with patch(f"{_MOD}._writes_enabled") as writes_gate:
            await _stop_fn()()

    writes_gate.assert_not_called()


async def test_queue_stop_cancel_refused_when_writes_are_disabled_with_zero_http_calls(
    tmp_path, monkeypatch
):
    """Withdrawing a human's halt lets the queue drain toward hardware — same gate as start."""
    _configure(tmp_path, monkeypatch, writes=False, token=_TOKEN)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _stop_fn()(cancel=True)

    mock_post.assert_not_called()
    assert _TOKEN not in ctx["envelope"]["error_message"]


async def test_queue_stop_cancel_refused_without_a_token_with_zero_http_calls(
    tmp_path, monkeypatch
):
    _configure(tmp_path, monkeypatch, writes=True, token=None)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="launch_token_required"):
            await _stop_fn()(cancel=True)
    mock_post.assert_not_called()


async def test_queue_stop_cancel_armed_posts_the_token_and_the_cancel_flag(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    with patch(
        f"{_MOD}._http_post_json", return_value=(200, {"stop_pending": False, "msg": ""})
    ) as m:
        result = await _stop_fn()(cancel=True)

    assert m.call_args.args[1] == {"cancel": True}
    assert m.call_args.kwargs["headers"] == {"X-Launch-Token": _TOKEN}
    assert extract_response_dict(result)["stop_pending"] is False


async def test_queue_stop_relays_a_manager_refusal(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, writes=False, token=None)
    body = _refusal("queue_request_rejected", "no stop is pending")
    with patch(f"{_MOD}._http_post_json", return_value=(409, body)):
        with assert_raises_error(error_type="queue_request_rejected") as ctx:
            await _stop_fn()()

    assert ctx["envelope"]["error_message"] == "no stop is pending"


# =========================================================================
# Surface drift nets
# =========================================================================


def test_launch_run_module_is_retired():
    """``launch_run`` is gone: execution is the two-step queue flow now.

    A leftover module would keep registering a second, ungated write path
    alongside the queue.
    """
    with pytest.raises(ModuleNotFoundError):
        import osprey.mcp_server.bluesky.tools.launch  # noqa: F401


def test_every_refusal_code_this_module_handles_is_documented_for_the_agent():
    """Tool docstrings are the agent's operating manual — a code with hints but
    no docstring entry is remediation the agent will never read.

    ``stop_run`` (``tools/stop.py``) shares this module's relay and hint table
    rather than keeping a second copy, so it is part of the surface this table
    serves and its docstring counts here too.
    """
    from osprey.mcp_server.bluesky.tools import stop

    docs = " ".join(
        get_tool_fn(tool).__doc__ or ""
        for tool in (
            queue.queue_status,
            queue.queue_list,
            queue.queue_add,
            queue.queue_start,
            queue.queue_stop,
            stop.stop_run,
        )
    )
    undocumented = [code for code in queue._REFUSAL_HINTS if code not in docs]
    assert not undocumented, f"refusal codes handled but never explained: {undocumented}"
