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

import hashlib
import json
import logging
import re
import time
from unittest.mock import patch

import pytest
import yaml

from osprey.audit.posture import POSTURE_SESSION_ENV_VAR
from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import queue
from osprey.utils.owner_header import OWNER_HEADER
from osprey_connectors.identity import TERMINAL_USER_ENV
from osprey_connectors.posture_store import CONTROL_CONTEXT_TREE_ENV_VAR, CONTROL_OWNER_ENV_VAR
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.queue"

_TOKEN = "genuinely-valid-token"


@pytest.fixture(autouse=True)
def _reset_bluesky_context():
    yield
    reset_server_context()


def _configure(
    tmp_path,
    monkeypatch,
    *,
    writes: bool | None,
    token: str | None,
    control_system: dict | None = None,
) -> None:
    """Put the process in a deployment posture: writes on/off, token set/unset.

    ``writes=None`` writes no config.yml at all — the fail-closed case.
    ``control_system`` writes that whole section instead of the deployment-wide
    flag alone, which is how a per-connector-type posture is staged.
    """
    monkeypatch.chdir(tmp_path)
    section = control_system if control_system is not None else {"writes_enabled": writes}
    if control_system is not None or writes is not None:
        (tmp_path / "config.yml").write_text(yaml.dump({"control_system": section}))
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


def _headers_without_owner(m) -> dict[str, str]:
    """The headers one forwarded request carried, minus the owner stamp.

    Every POST to the bridge carries ``X-Osprey-Owner`` whenever the ladder can
    name an owner, and under test the process account always can. The launch
    token rows below are about the token and nothing else, so they compare the
    rest of the header dict exactly: a stray third header still fails them,
    while the owner half is pinned by the owner section at the end of the file.
    """
    headers = m.call_args.kwargs["headers"] or {}
    return {name: value for name, value in headers.items() if name != OWNER_HEADER}


def _owner_sent(m) -> str | None:
    """The owner one forwarded request named, or ``None`` when it named nobody."""
    return (m.call_args.kwargs["headers"] or {}).get(OWNER_HEADER)


def _as_terminal_user(monkeypatch, user: str) -> None:
    """Put the process in a terminal container: the roster account, no stamp over it."""
    monkeypatch.delenv(CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.delenv(CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.setenv(TERMINAL_USER_ENV, user)


def _as_owner_less_tree_container(monkeypatch, tmp_path) -> None:
    """Put the process in a container holding the state tree and owning nothing.

    The account such a container runs as names nobody whose chip anyone set, so
    the ladder answers NO_OWNER rather than falling through to that account.
    """
    monkeypatch.delenv(CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.setenv(CONTROL_CONTEXT_TREE_ENV_VAR, str(tmp_path / "control-context"))


def _refusal(code: str, detail: str, **extras) -> dict:
    """A bridge refusal body: the wire contract's nested ``detail`` dict."""
    return {"detail": {"code": code, "detail": detail, **extras}}


_BROWSE_ONLY = {
    "can_execute": False,
    "reason": "browse_only_connector",
    "detail": (
        "This deployment uses the mock connector, which cannot move hardware, so "
        "plans can be composed and validated but not executed. To execute plans, run "
        "`osprey set connector=virtual_accelerator` and redeploy."
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
    assert "osprey set connector=virtual_accelerator" in parsed["capability"]["detail"]
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
    assert _headers_without_owner(m) == {"X-Launch-Token": _TOKEN}
    assert extract_response_dict(result) == body


async def test_queue_add_withholds_the_token_when_writes_are_disabled(tmp_path, monkeypatch):
    """The kill switch applied to enqueue: composing is allowed, arming is not.

    With writes off the request still goes out — an item on an idle queue moves
    nothing — but WITHOUT the launch token, so the bridge (which alone knows the
    manager's live state) permits it only while the queue is idle and refuses
    it the moment the queue is draining. Asserting on the absent header pins
    the mechanism; a status-code assertion would not.

    The add SUCCEEDING is half the contract and is asserted too: an unarmed
    lane that could no longer compose a queue would have turned the
    compose-while-unarmed guarantee into a refusal nobody asked for.
    """
    _configure(tmp_path, monkeypatch, writes=False, token=_TOKEN)
    body = {"run_id": "r1", "revision": 3, "item": {"item_uid": "u1"}}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _add_fn()(draft_revision=3)

    assert _headers_without_owner(m) == {}
    assert extract_response_dict(result)["run_id"] == "r1"


async def test_queue_add_missing_config_fails_closed_and_withholds_the_token(tmp_path, monkeypatch):
    """No config.yml at all is not "writes enabled" by omission."""
    _configure(tmp_path, monkeypatch, writes=None, token=_TOKEN)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=3)

    assert _headers_without_owner(m) == {}


async def test_queue_add_without_a_configured_token_still_composes(tmp_path, monkeypatch):
    """An unarmed deployment may still build a queue; only starting it is gated."""
    _configure(tmp_path, monkeypatch, writes=True, token=None)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=3)

    m.assert_called_once()
    assert _headers_without_owner(m) == {}


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
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)) as m:
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _add_fn()(draft_revision=7)

    # The other half of the compose-while-unarmed contract: the same tokenless
    # request that an idle queue accepts is what a draining queue refuses.
    assert _headers_without_owner(m) == {}
    envelope = ctx["envelope"]
    assert envelope["details"]["code"] == "launch_token_required"
    assert envelope["details"]["manager_state"] == "executing_queue"
    assert any("writes_enabled" in s for s in envelope["suggestions"])


async def test_queue_add_names_the_bound_lanes_own_posture_key_in_the_withheld_hint(
    tmp_path, monkeypatch
):
    """The hint names the key that would arm THIS lane's machine, not the global one.

    A deployment that resolves its live target to a connector type has a
    per-type key to point at, and pointing at the deployment-wide one instead
    would tell an operator to arm every target in order to run a plan on one.
    """
    _configure(
        tmp_path,
        monkeypatch,
        writes=None,
        token=_TOKEN,
        control_system={
            "type": "epics",
            "writes_enabled": False,
            "connector": {"epics": {"gateways": {"read_only": {"host": "gw-ro"}}}},
        },
    )
    body = _refusal(
        "launch_token_required",
        "the queue is running, so adding an item requires the launch token",
        manager_state="executing_queue",
    )
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)):
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _add_fn()(draft_revision=7)

    suggestions = ctx["envelope"]["suggestions"]
    assert any("control_system.connector.epics.writes_enabled" in s for s in suggestions)
    assert any("'bluesky'" in s and "'live'" in s for s in suggestions)


async def test_queue_add_still_composes_in_a_readonly_session(tmp_path, monkeypatch):
    """A read-only run withholds the token; it does not stop a queue being built.

    The deployment is fully armed here, so the only thing keeping the token off
    this request is the sandbox posture — and composing onto an idle queue moves
    nothing, so it must still succeed.
    """
    _armed(tmp_path, monkeypatch)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    body = {"run_id": "r1", "revision": 3, "item": {"item_uid": "u1"}}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _add_fn()(draft_revision=3)

    assert _headers_without_owner(m) == {}
    assert extract_response_dict(result)["run_id"] == "r1"


async def test_queue_add_readonly_refusal_names_the_sandbox_posture_not_a_config_key(
    tmp_path, monkeypatch
):
    """A read-only session must not be told to edit a key that already says true.

    Writes ARE armed in this deployment, so the config-key hint would send an
    operator to change something that was never what withheld the token.
    """
    _armed(tmp_path, monkeypatch)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    body = _refusal(
        "launch_token_required",
        "the queue is running, so adding an item requires the launch token",
        manager_state="executing_queue",
    )
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)):
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _add_fn()(draft_revision=7)

    suggestions = ctx["envelope"]["suggestions"]
    assert any("OSPREY_EXECUTION_MODE=readonly" in s for s in suggestions)
    assert all("profile.yml" not in s for s in suggestions)


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
    """The repeat-run case: a consumed revision needs a NEW one, not a retry."""
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


async def test_queue_add_unknown_device_relays_a_capped_device_list(tmp_path, monkeypatch):
    """A worker whose namespace exceeds one page sends a count and a URL instead
    of the names. Both must survive into details untouched — the agent pages the
    names with list_devices, so the hint points there either way."""
    _armed(tmp_path, monkeypatch)
    body = _refusal(
        "unknown_device",
        "plan 'grid_scan' referenced device 'COR9', which this worker did not build; "
        "available devices: ['BPM1', 'BPM2', 'COR1'] (+2 more; full list via GET /devices)",
        plan="grid_scan",
        devices=["COR9"],
        available_count=5,
        available_devices_url="/devices",
    )
    with patch(f"{_MOD}._http_post_json", return_value=(400, body)):
        with assert_raises_error(error_type="unknown_device") as ctx:
            await _add_fn()(draft_revision=7)

    details = ctx["envelope"]["details"]
    assert details["available_count"] == 5
    assert details["available_devices_url"] == "/devices"
    assert "available_devices" not in details
    assert any("list_devices" in s for s in ctx["envelope"]["suggestions"])


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


async def test_queue_start_without_a_token_asks_the_bridge_and_relays_its_refusal(
    tmp_path, monkeypatch
):
    """A tokenless server still asks the bridge — it does not refuse locally.

    The bridge is the one authority on arming, so a deployment that withheld
    the launch token from this agent posts ``/queue/start`` anyway, with NO
    token header (never a header carrying ``None``, which would not survive
    the HTTP client), and relays the bridge's own ``launch_token_required``.
    """
    _configure(tmp_path, monkeypatch, writes=True, token=None)
    body = _refusal("launch_token_required", "starting the queue requires the launch token")
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)) as m:
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _start_fn()()

    assert m.call_args.args[0] == "/queue/start"
    assert _headers_without_owner(m) == {}
    assert ctx["envelope"]["details"]["code"] == "launch_token_required"


async def test_a_tokenless_start_still_respects_the_kill_switch(tmp_path, monkeypatch):
    """Writes disabled refuses before the bridge is reached at all, token or
    no token: the kill switch is checked ahead of every arming path."""
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
    assert _headers_without_owner(m) == {"X-Launch-Token": _TOKEN}
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
# queue_start — quoting back the queue the approver was shown
# =========================================================================
#
# The approval hook stamps the queue token it listed under the prompt; this
# tool quotes it back as `expected_plan_queue_uid` so the bridge can refuse a
# start whose queue moved in between. The helpers below write that file the way
# the HOOK writes it — its own file-name derivation spelled out again here —
# rather than through the tool's reader, so a drift between the two spellings
# fails these tests instead of silently disabling the binding in production.

_SESSION = "kernel:9f3c1a2b"

_QUEUE_LOGGER = "osprey.mcp_server.bluesky.tools.queue"


def _stamp_dir(tmp_path, monkeypatch):
    """Point the tool's stamp reader at a directory this test owns."""
    directory = tmp_path / "control_target"
    directory.mkdir(exist_ok=True)
    monkeypatch.setattr(queue.target_state, "state_dir", lambda: directory)
    return directory


def _stamp_name(session: str, lane: str) -> str:
    """The file name the approval hook files a start stamp under.

    Restated from ``osprey_approval.queue_start_approval_filename`` — a hashed
    session slug and a lane with everything outside ``[A-Za-z0-9_.-]``
    substituted away. The hook cannot be imported from this venv, so the
    derivation is spelled twice on purpose and exercised against the tool's own
    spelling here.
    """
    slug = hashlib.sha256(session.encode("utf-8")).hexdigest()[:16]
    return f"queue_start_approval_{slug}_{re.sub(r'[^A-Za-z0-9_.-]', '-', lane)}.json"


def _write_start_stamp(directory, session, lane, uid, *, age_s=0.0, payload_lane=None):
    """Write one queue-start stamp as the hook would have left it.

    ``payload_lane`` defaults to *lane* — the ordinary case, where the name and
    the payload agree. Passing a different one stages the collision the payload
    exists to catch: two lane ids whose names sanitise to the same file.
    """
    path = directory / _stamp_name(session, lane)
    path.write_text(
        json.dumps(
            {
                "lane": lane if payload_lane is None else payload_lane,
                "plan_queue_uid": uid,
                "ts": time.time() - age_s,
            }
        ),
        encoding="utf-8",
    )
    return path


async def test_queue_start_sends_the_uid_the_approval_prompt_stamped(tmp_path, monkeypatch):
    """The token the human's prompt listed rides on the start that follows it."""
    _armed(tmp_path, monkeypatch)
    monkeypatch.setenv(POSTURE_SESSION_ENV_VAR, _SESSION)
    _write_start_stamp(_stamp_dir(tmp_path, monkeypatch), _SESSION, "bluesky", "q7")

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True, "msg": ""})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()

    assert m.call_args.args[1] == {"expected_plan_queue_uid": "q7"}


async def test_queue_start_ignores_an_expired_stamp(tmp_path, monkeypatch):
    """An hour-old binding describes a queue nobody is looking at any more.

    The start still goes — an unbound start is what every deployment did before
    this existed — but it carries no token, so the bridge is never asked to
    compare against a list whose approval has aged out.
    """
    _armed(tmp_path, monkeypatch)
    monkeypatch.setenv(POSTURE_SESSION_ENV_VAR, _SESSION)
    _write_start_stamp(_stamp_dir(tmp_path, monkeypatch), _SESSION, "bluesky", "q7", age_s=7200.0)

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True, "msg": ""})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()

    assert m.call_args.args[1] == {}


async def test_a_stamp_naming_another_lane_is_not_quoted_and_warns_once(
    tmp_path, monkeypatch, caplog
):
    """The payload's lane is the truth, and a mismatch is reported, not guessed.

    Lane ids are substituted into the file name, so two configured lanes can
    land on one file. The stamp says which lane it was actually rendered for;
    when that is not the lane being started, quoting its token would bind this
    start to another lane's queue. One warning, naming the file and both lanes,
    is what tells an operator why a prompt-bound start went unbound.
    """
    _armed(tmp_path, monkeypatch)
    monkeypatch.setenv(POSTURE_SESSION_ENV_VAR, _SESSION)
    path = _write_start_stamp(
        _stamp_dir(tmp_path, monkeypatch), _SESSION, "bluesky", "q7", payload_lane="bluesky+va"
    )

    with caplog.at_level(logging.WARNING, logger=_QUEUE_LOGGER):
        with patch(
            f"{_MOD}._http_post_json", return_value=(200, {"started": True, "msg": ""})
        ) as m:
            with patch(f"{_MOD}.notify_agent_activity_async"):
                await _start_fn()()

    assert m.call_args.args[1] == {}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert path.name in message
    assert "bluesky+va" in message and "'bluesky'" in message


async def test_a_session_less_start_sends_no_uid(tmp_path, monkeypatch):
    """No audit session, no binding — and no quiet fallback to a shared file.

    Every unattributed process on a checkout would file under one name, so a
    token found there is as likely another window's queue as this one's. The
    hook stamps nothing for such a render; the tool must quote nothing either,
    even when some other process left a stamp behind.
    """
    _armed(tmp_path, monkeypatch)
    monkeypatch.delenv(POSTURE_SESSION_ENV_VAR, raising=False)
    directory = _stamp_dir(tmp_path, monkeypatch)
    (directory / "queue_start_approval_anon_bluesky.json").write_text(
        json.dumps({"lane": "bluesky", "plan_queue_uid": "q7", "ts": time.time()}),
        encoding="utf-8",
    )

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True, "msg": ""})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()

    assert m.call_args.args[1] == {}


async def test_a_null_or_missing_stamp_sends_no_uid(tmp_path, monkeypatch):
    """Both spellings of "the prompt showed no queue" mean: send nothing.

    The hook nulls the token on every render that listed no queue rather than
    deleting the file, so the tool meets a ``None`` and an absent file alike and
    has to read them the same way.
    """
    _armed(tmp_path, monkeypatch)
    monkeypatch.setenv(POSTURE_SESSION_ENV_VAR, _SESSION)
    directory = _stamp_dir(tmp_path, monkeypatch)
    _write_start_stamp(directory, _SESSION, "bluesky", None)

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True, "msg": ""})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()
    assert m.call_args.args[1] == {}

    (directory / _stamp_name(_SESSION, "bluesky")).unlink()
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True, "msg": ""})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()
    assert m.call_args.args[1] == {}


async def test_queue_start_relays_the_409_and_never_resends(tmp_path, monkeypatch):
    """A moved queue is a refusal for a human to answer, not a retry.

    The bridge's sentence and both uids are relayed verbatim, the remedy sends
    the agent back to re-read the queue, and the start is posted exactly once —
    a second attempt would re-arm a queue nobody approved.
    """
    _armed(tmp_path, monkeypatch)
    monkeypatch.setenv(POSTURE_SESSION_ENV_VAR, _SESSION)
    _write_start_stamp(_stamp_dir(tmp_path, monkeypatch), _SESSION, "bluesky", "q7")
    body = _refusal(
        "queue_changed_since_approval",
        "the queue has changed since it was approved: the start named queue 'q7', "
        "but the manager now holds 'q9'",
        plan_queue_uid="q9",
        expected_plan_queue_uid="q7",
    )

    with patch(f"{_MOD}._http_post_json", return_value=(409, body)) as m:
        with assert_raises_error(error_type="queue_changed_since_approval") as ctx:
            await _start_fn()()

    assert m.call_count == 1
    assert ctx["envelope"]["details"]["plan_queue_uid"] == "q9"
    assert ctx["envelope"]["details"]["expected_plan_queue_uid"] == "q7"
    assert any("queue_list" in s for s in ctx["envelope"]["suggestions"])


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
    assert _headers_without_owner(m) == {}
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
    assert _headers_without_owner(m) == {"X-Launch-Token": _TOKEN}
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
            queue.queue_remove,
            stop.stop_run,
        )
    )
    undocumented = [code for code in queue._REFUSAL_HINTS if code not in docs]
    assert not undocumented, f"refusal codes handled but never explained: {undocumented}"


# =========================================================================
# queue_remove — drop one pending item; the interrupted-item way out
# =========================================================================


def _remove_fn():
    return get_tool_fn(queue.queue_remove)


async def test_queue_remove_deletes_the_item_and_relays_the_body(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    body = {"removed": True, "item": {"item_uid": "u1", "name": "hysteresis_loop"}}
    with patch(f"{_MOD}._http_delete_json", return_value=(200, body)) as m:
        result = await _remove_fn()("u1")

    assert m.call_args.args[0] == "/queue/items/u1"
    # Omitted lane means the ACTIVE lane, exactly like every other queue read.
    assert m.call_args.kwargs["lane"] is None
    assert extract_response_dict(result) == body


async def test_queue_remove_passes_the_named_lane_through(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    body = {"removed": True, "item": None}
    with patch(f"{_MOD}._http_delete_json", return_value=(200, body)) as m:
        await _remove_fn()("u1", lane="bluesky2")

    assert m.call_args.kwargs["lane"] == "bluesky2"


async def test_queue_remove_url_encodes_the_uid(tmp_path, monkeypatch):
    """A uid is manager-minted and opaque — path-encode it, never trust it."""
    _armed(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_delete_json", return_value=(200, {"removed": True})) as m:
        await _remove_fn()("a/b c")

    assert m.call_args.args[0] == "/queue/items/a%2Fb%20c"


async def test_queue_remove_relays_the_bridge_refusal_code_and_detail(tmp_path, monkeypatch):
    """An unknown uid is the manager's refusal, relayed verbatim — the queue is
    unchanged and the hint sends the agent back to queue_list."""
    _armed(tmp_path, monkeypatch)
    body = _refusal("queue_request_rejected", "Item 'nope' is not in the queue.")
    with patch(f"{_MOD}._http_delete_json", return_value=(409, body)):
        with assert_raises_error(error_type="queue_request_rejected") as ctx:
            await _remove_fn()("nope")

    envelope = ctx["envelope"]
    assert envelope["error_message"] == "Item 'nope' is not in the queue."
    assert envelope["details"]["code"] == "queue_request_rejected"


async def test_queue_remove_is_ungated_by_writes_and_token(tmp_path, monkeypatch):
    """Removal must keep working with writes disabled and no token — it is the
    sole way past the interrupted-item start refusal, so gating it would trap a
    wedged queue exactly when the kill switch is on."""
    _configure(tmp_path, monkeypatch, writes=False, token=None)
    body = {"removed": True, "item": None}
    with patch(f"{_MOD}._http_delete_json", return_value=(200, body)) as m:
        result = await _remove_fn()("u1")

    assert extract_response_dict(result) == body
    # No launch token header on a removal — there is nothing to arm.
    assert "headers" not in m.call_args.kwargs


# =========================================================================
# the X-Osprey-Owner stamp — who queued this work
# =========================================================================


async def test_queue_add_names_the_terminal_user_as_the_owner(tmp_path, monkeypatch):
    """An add from a terminal container is attributed to that terminal's account.

    This is the whole point of the stamp: the bridge records the item against a
    person, so a narrowing set from that person's chip governs the plan when it
    runs. The assertion is on the exact header dict, which pins that the owner
    rides ALONGSIDE the launch token rather than in place of it.
    """
    _as_terminal_user(monkeypatch, "rosterbob")
    _armed(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=3)

    assert m.call_args.kwargs["headers"] == {
        "X-Launch-Token": _TOKEN,
        OWNER_HEADER: "rosterbob",
    }


async def test_a_tree_container_with_no_stamped_owner_sends_no_owner_header(tmp_path, monkeypatch):
    """Owner-less means the header is ABSENT, never present-and-empty.

    A blank or sentinel-shaped value would be read back as a name by nothing and
    warned about by the reader; the absent header is the ordinary owner-less
    case, and the bridge records the item at the ceiling.
    """
    _as_owner_less_tree_container(monkeypatch, tmp_path)
    _armed(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=3)

    assert m.call_args.kwargs["headers"] == {"X-Launch-Token": _TOKEN}


async def test_a_stamped_owner_beats_the_account_the_process_runs_as(tmp_path, monkeypatch):
    """A dispatch job's exported owner is the owner, not the account running it.

    The job runs as a service account nobody's chip belongs to; the stamp names
    the person the work is for, so it must win over the identity ladder.
    """
    monkeypatch.setenv(TERMINAL_USER_ENV, "service-account")
    monkeypatch.delenv(CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.setenv(CONTROL_OWNER_ENV_VAR, "alice")
    _armed(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=3)

    assert _owner_sent(m) == "alice"


async def test_queue_start_carries_the_owner(tmp_path, monkeypatch):
    """Starting a queue is attributed too: the run record names who released it."""
    _as_terminal_user(monkeypatch, "rosterbob")
    _armed(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()

    assert m.call_args.args[0] == "/queue/start"
    assert _owner_sent(m) == "rosterbob"


async def test_an_ungated_stop_still_carries_the_owner(tmp_path, monkeypatch):
    """The stamp is attribution, not a credential, so it rides an ungated request.

    A plain halt carries no launch token at all — the header dict exists only
    because of the owner — which pins that the stamp does not hitch a ride on
    the token decision.
    """
    _as_terminal_user(monkeypatch, "rosterbob")
    _configure(tmp_path, monkeypatch, writes=False, token=None)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"stop_pending": True})) as m:
        await _stop_fn()()

    assert m.call_args.kwargs["headers"] == {OWNER_HEADER: "rosterbob"}


async def test_an_owner_less_stop_sends_no_headers_at_all(tmp_path, monkeypatch):
    """Nothing to say means no header dict, not an empty one.

    An empty mapping would survive the HTTP client harmlessly, but it would also
    make "this request carries nothing" indistinguishable from "this request
    carries something the client dropped".
    """
    _as_owner_less_tree_container(monkeypatch, tmp_path)
    _configure(tmp_path, monkeypatch, writes=False, token=None)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"stop_pending": True})) as m:
        await _stop_fn()()

    assert m.call_args.kwargs["headers"] is None
