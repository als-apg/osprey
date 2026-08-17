"""Tests for the agent-activity emit on ``set_draft`` / ``clear_draft``.

A successful bridge PATCH/DELETE must fire exactly one
``notify_agent_activity(kind='panel', panel=<plans panel id>, ...)`` via
``anyio.to_thread.run_sync``; a rejected PATCH must fire none; and the emit
must never alter the tool result — including when the web terminal is down
(``notify_agent_activity`` swallows all exceptions).

The bridge HTTP boundary is mocked exactly like ``test_draft_tools.py``
(``_http_patch_json`` / ``_http_delete_json`` at the draft-module seam), and
``notify_agent_activity`` is patched at the same seam — draft.py imports it
into its own namespace (``from osprey.mcp_server.http import ...``), so the
patch must target ``osprey.mcp_server.bluesky.tools.draft``, not
``osprey.mcp_server.http``.

Later sections cover the emits on the other bluesky tool modules (``queue_stop``
and the authoring tools). Each is delimited by a ``#####`` banner, keeps its own
helpers and its own module-path constant, and names every test after the tool it
covers so a ``-k`` run selects the whole section.
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.bluesky.server_context import (
    initialize_server_context,
    reset_server_context,
)
from osprey.mcp_server.bluesky.tools import draft
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.draft"


def _set_fn():
    return get_tool_fn(draft.set_draft)


def _clear_fn():
    return get_tool_fn(draft.clear_draft)


@pytest.fixture(autouse=True)
def _reset_server_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initialize_server_context()
    yield
    reset_server_context()


_SET_RESP = {"revision": 1, "changed": ["plan_name"], "plan_name": "grid_scan"}
_CLEAR_RESP = {"revision": 2, "cleared": True}


# =========================================================================
# Successful set_draft → exactly one panel-kind emit with the plan name
# =========================================================================


async def test_set_draft_success_emits_one_panel_activity():
    with (
        patch(f"{_MOD}._http_patch_json", return_value=(200, _SET_RESP)),
        patch(f"{_MOD}.notify_agent_activity") as notify,
    ):
        result = await _set_fn()(plan_name="grid_scan")

    notify.assert_called_once_with(
        tool="set_draft", kind="panel", panel="bluesky", detail="grid_scan"
    )
    # Tool result is unchanged by the emit.
    assert extract_response_dict(result) == _SET_RESP


async def test_set_draft_panel_id_resolved_from_web_panels_config():
    """A facility that registered the panel (mount path ``/bluesky/``) under a
    non-canonical ``web.panels`` key gets that id in the emit — the id is
    config-resolved, never hardcoded."""
    config = {
        "web": {
            "panels": {
                "grafana": {"url": "http://localhost:9000", "path": "/d/scan/"},
                "operator-plan": {"url": "http://localhost:9000", "path": "/bluesky/"},
            }
        }
    }
    with (
        patch(f"{_MOD}._http_patch_json", return_value=(200, _SET_RESP)),
        patch(f"{_MOD}.notify_agent_activity") as notify,
        patch("osprey.utils.workspace.load_osprey_config", return_value=config),
    ):
        await _set_fn()(plan_name="grid_scan")

    assert notify.call_args.kwargs["panel"] == "operator-plan"


async def test_set_draft_panel_id_falls_back_to_a_pre_merge_plan_mount():
    """A deployment built before PLAN was folded into BLUESKY still registers a
    panel at ``/plan/``. The sidecar serves the merged bundle there, so the
    highlight must land on it rather than on a ``bluesky`` entry the config
    does not have."""
    config = {"web": {"panels": {"plan": {"url": "http://localhost:9000", "path": "/plan/"}}}}
    with (
        patch(f"{_MOD}._http_patch_json", return_value=(200, _SET_RESP)),
        patch(f"{_MOD}.notify_agent_activity") as notify,
        patch("osprey.utils.workspace.load_osprey_config", return_value=config),
    ):
        await _set_fn()(plan_name="grid_scan")

    assert notify.call_args.kwargs["panel"] == "plan"


async def test_set_draft_panel_id_prefers_the_current_mount_over_the_legacy_one():
    """A config.yml rebuilt in place can carry BOTH: the new registration and
    the deprecated alias left over from before the merge. The current spelling
    wins, or the rebuild would still highlight the retired entry."""
    config = {
        "web": {
            "panels": {
                "plan": {"url": "http://localhost:9000", "path": "/plan/"},
                "bluesky": {"url": "http://localhost:9000", "path": "/bluesky/"},
            }
        }
    }
    with (
        patch(f"{_MOD}._http_patch_json", return_value=(200, _SET_RESP)),
        patch(f"{_MOD}.notify_agent_activity") as notify,
        patch("osprey.utils.workspace.load_osprey_config", return_value=config),
    ):
        await _set_fn()(plan_name="grid_scan")

    assert notify.call_args.kwargs["panel"] == "bluesky"


# =========================================================================
# Successful clear_draft → one emit with detail='cleared'
# =========================================================================


async def test_clear_draft_success_emits_cleared_detail():
    with (
        patch(f"{_MOD}._http_delete_json", return_value=(200, _CLEAR_RESP)),
        patch(f"{_MOD}.notify_agent_activity") as notify,
    ):
        result = await _clear_fn()()

    notify.assert_called_once_with(
        tool="clear_draft", kind="panel", panel="bluesky", detail="cleared"
    )
    assert extract_response_dict(result) == _CLEAR_RESP


# =========================================================================
# Bridge 4xx on PATCH → NO emit, normal error result
# =========================================================================


async def test_set_draft_bridge_422_does_not_emit():
    body = {"detail": "unknown plan 'nope'"}
    with (
        patch(f"{_MOD}._http_patch_json", return_value=(422, body)),
        patch(f"{_MOD}.notify_agent_activity") as notify,
    ):
        with assert_raises_error(error_type="unknown_plan") as ctx:
            await _set_fn()(plan_name="nope")

    notify.assert_not_called()
    assert "unknown plan" in ctx["envelope"]["error_message"]


async def test_set_draft_no_draft_409_does_not_emit():
    body = {"code": "no_draft", "detail": "no draft exists"}
    with (
        patch(f"{_MOD}._http_patch_json", return_value=(409, body)),
        patch(f"{_MOD}.notify_agent_activity") as notify,
    ):
        with assert_raises_error(error_type="no_draft"):
            await _set_fn()(plan_args_patch={"num": 1})

    notify.assert_not_called()


# =========================================================================
# Web terminal down → tool result identical to the success case
# =========================================================================


def _dead_port() -> int:
    """A localhost port with nothing listening on it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def test_set_draft_result_identical_when_web_terminal_down(monkeypatch):
    """The REAL notify helper runs against a dead port; it swallows the
    connection failure, so the tool result matches the success case exactly."""
    monkeypatch.setenv("OSPREY_WEB_PORT", str(_dead_port()))
    with patch(f"{_MOD}._http_patch_json", return_value=(200, _SET_RESP)):
        result = await _set_fn()(plan_name="grid_scan")
    assert extract_response_dict(result) == _SET_RESP


async def test_clear_draft_result_identical_when_web_terminal_down(monkeypatch):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(_dead_port()))
    with patch(f"{_MOD}._http_delete_json", return_value=(200, _CLEAR_RESP)):
        result = await _clear_fn()()
    assert extract_response_dict(result) == _CLEAR_RESP


# #########################################################################
# queue_stop — the emit on the halting surface
#
# A stop and a withdrawal are opposite operations that share one tool and one
# success return, so the emit's ``detail`` is the only thing distinguishing
# them in the web terminal. These tests pin both strings exactly, and pin that
# the three refusals ahead of the mutation stay silent.
# #########################################################################

_QUEUE_MOD = "osprey.mcp_server.bluesky.tools.queue"

_QUEUE_STOP_TOKEN = "queue-stop-launch-token"
_STOP_RESP = {"stop_pending": True, "msg": "queue will stop after the running item"}
_WITHDRAWN_RESP = {"stop_pending": False, "msg": "pending stop withdrawn"}


def _queue_stop_fn():
    from osprey.mcp_server.bluesky.tools import queue

    return get_tool_fn(queue.queue_stop)


def _queue_stop_posture(tmp_path, monkeypatch, *, writes: bool, token: str | None) -> None:
    """Re-arm the bridge context for a queue_stop test: writes on/off, token set/unset."""
    (tmp_path / "config.yml").write_text(yaml.dump({"control_system": {"writes_enabled": writes}}))
    if token is None:
        monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", token)
    initialize_server_context()


async def test_queue_stop_plain_stop_emits_detail_stop(tmp_path, monkeypatch):
    """The ungated halt still reports itself: one run-kind emit, detail 'stop'."""
    _queue_stop_posture(tmp_path, monkeypatch, writes=False, token=None)
    with (
        patch(f"{_QUEUE_MOD}._http_post_json", return_value=(200, _STOP_RESP)),
        patch(f"{_QUEUE_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await _queue_stop_fn()()

    notify.assert_called_once_with("queue_stop", "run", detail="stop")
    assert extract_response_dict(result) == _STOP_RESP


async def test_queue_stop_withdrawal_emits_detail_stop_withdrawn(tmp_path, monkeypatch):
    """A withdrawal must never render as a stop — the operator reads this string
    to decide whether the queue is halting or draining toward hardware."""
    _queue_stop_posture(tmp_path, monkeypatch, writes=True, token=_QUEUE_STOP_TOKEN)
    with (
        patch(f"{_QUEUE_MOD}._http_post_json", return_value=(200, _WITHDRAWN_RESP)),
        patch(f"{_QUEUE_MOD}.notify_agent_activity_async") as notify,
    ):
        result = await _queue_stop_fn()(cancel=True)

    notify.assert_called_once_with("queue_stop", "run", detail="stop-withdrawn")
    assert extract_response_dict(result) == _WITHDRAWN_RESP


async def test_queue_stop_withdrawal_refused_for_writes_disabled_does_not_emit(
    tmp_path, monkeypatch
):
    """Refused before the mutation: nothing was withdrawn, so nothing is reported."""
    _queue_stop_posture(tmp_path, monkeypatch, writes=False, token=_QUEUE_STOP_TOKEN)
    with (
        patch(f"{_QUEUE_MOD}._http_post_json") as post,
        patch(f"{_QUEUE_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="writes_disabled"):
            await _queue_stop_fn()(cancel=True)

    post.assert_not_called()
    notify.assert_not_called()


async def test_queue_stop_withdrawal_refused_without_a_token_does_not_emit(tmp_path, monkeypatch):
    _queue_stop_posture(tmp_path, monkeypatch, writes=True, token=None)
    with (
        patch(f"{_QUEUE_MOD}._http_post_json") as post,
        patch(f"{_QUEUE_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="launch_token_required"):
            await _queue_stop_fn()(cancel=True)

    post.assert_not_called()
    notify.assert_not_called()


async def test_queue_stop_bridge_refusal_does_not_emit(tmp_path, monkeypatch):
    """The manager refused, so the queue's state is unchanged — no activity."""
    _queue_stop_posture(tmp_path, monkeypatch, writes=False, token=None)
    body = {"detail": {"code": "queue_request_rejected", "detail": "no stop is pending"}}
    with (
        patch(f"{_QUEUE_MOD}._http_post_json", return_value=(409, body)),
        patch(f"{_QUEUE_MOD}.notify_agent_activity_async") as notify,
    ):
        with assert_raises_error(error_type="queue_request_rejected"):
            await _queue_stop_fn()()

    notify.assert_not_called()


async def test_queue_stop_result_identical_when_web_terminal_down(tmp_path, monkeypatch):
    """The REAL notify helper against a dead port leaves the halt's result intact."""
    _queue_stop_posture(tmp_path, monkeypatch, writes=False, token=None)
    monkeypatch.setenv("OSPREY_WEB_PORT", str(_dead_port()))
    with patch(f"{_QUEUE_MOD}._http_post_json", return_value=(200, _STOP_RESP)):
        result = await _queue_stop_fn()()
    assert extract_response_dict(result) == _STOP_RESP


# #########################################################################
# write_plan / validate_plan — the emits on the authoring surface
#
# Both tools reach no hardware and are never kill-switch-gated, so the only
# thing separating an emit from silence is whether the plan file actually
# changed standing: write_plan mutates on any 200, validate_plan only on a
# PASS. A failed validation writes nothing the human's panel would re-render,
# so it must stay silent even though its HTTP call succeeded.
# #########################################################################

_AUTHORING_MOD = "osprey.mcp_server.bluesky.tools.authoring"

_WRITE_RESP = {"name": "tiny", "content_hash": "deadbeef"}
_PASS_RESP = {"passed": True, "reasons": [], "content_hash": "deadbeef", "upload": {}}
_FAIL_RESP = {"passed": False, "reasons": ["import of 'os' is not allowed"], "upload": {}}

_WRITE_ARGS = {
    "name": "tiny",
    "category": "accelerator",
    "required_devices": ["correctors"],
    "writes": False,
    "body": "def build_plan(devices, params):\n    yield\n",
}


def _write_plan_fn():
    from osprey.mcp_server.bluesky.tools import authoring

    return get_tool_fn(authoring.write_plan)


def _validate_plan_fn():
    from osprey.mcp_server.bluesky.tools import authoring

    return get_tool_fn(authoring.validate_plan)


async def test_write_plan_success_emits_one_plans_panel_activity():
    """Authoring changes what the BLUESKY panel lists, so the highlight lands
    on that panel — same id the draft emits resolve, not a second spelling."""
    with (
        patch(f"{_AUTHORING_MOD}._http_post_json", return_value=(200, _WRITE_RESP)),
        patch(f"{_AUTHORING_MOD}.notify_agent_activity") as notify,
    ):
        result = await _write_plan_fn()(**_WRITE_ARGS)

    notify.assert_called_once_with(tool="write_plan", kind="panel", panel="bluesky", detail="tiny")
    assert extract_response_dict(result) == _WRITE_RESP


async def test_write_plan_panel_id_resolved_from_web_panels_config():
    """The panel id is config-resolved for authoring too — a facility that
    registered the panel under its own key gets that key in the emit."""
    config = {"web": {"panels": {"operator-plan": {"path": "/bluesky/"}}}}
    with (
        patch(f"{_AUTHORING_MOD}._http_post_json", return_value=(200, _WRITE_RESP)),
        patch(f"{_AUTHORING_MOD}.notify_agent_activity") as notify,
        patch("osprey.utils.workspace.load_osprey_config", return_value=config),
    ):
        await _write_plan_fn()(**_WRITE_ARGS)

    assert notify.call_args.kwargs["panel"] == "operator-plan"


async def test_write_plan_rejected_does_not_emit():
    """Refused before the file was written — no plan exists to light up."""
    body = {"detail": "invalid plan name '1bad': must be a valid Python identifier"}
    with (
        patch(f"{_AUTHORING_MOD}._http_post_json", return_value=(400, body)),
        patch(f"{_AUTHORING_MOD}.notify_agent_activity") as notify,
    ):
        with assert_raises_error(error_type="plan_write_rejected"):
            await _write_plan_fn()(**{**_WRITE_ARGS, "name": "1bad"})

    notify.assert_not_called()


async def test_validate_plan_pass_emits_the_validated_detail():
    """A pass is what makes the plan loadable and enqueueable; the detail names
    the plan so the operator can tell which one just cleared."""
    with (
        patch(f"{_AUTHORING_MOD}._http_post_json", return_value=(200, _PASS_RESP)),
        patch(f"{_AUTHORING_MOD}.notify_agent_activity") as notify,
    ):
        result = await _validate_plan_fn()(name="tiny")

    notify.assert_called_once_with(
        tool="validate_plan", kind="panel", panel="bluesky", detail="validated tiny"
    )
    assert extract_response_dict(result) == _PASS_RESP


async def test_validate_plan_failure_does_not_emit():
    """A 200 carrying passed=false is still a non-event: the plan's standing is
    unchanged, so reporting it would announce a validation that did not happen."""
    with (
        patch(f"{_AUTHORING_MOD}._http_post_json", return_value=(200, _FAIL_RESP)),
        patch(f"{_AUTHORING_MOD}.notify_agent_activity") as notify,
    ):
        result = await _validate_plan_fn()(name="tiny")

    notify.assert_not_called()
    assert extract_response_dict(result) == _FAIL_RESP


async def test_validate_plan_unknown_name_does_not_emit():
    with (
        patch(
            f"{_AUTHORING_MOD}._http_post_json",
            return_value=(404, {"detail": "unknown session plan 'nope'"}),
        ),
        patch(f"{_AUTHORING_MOD}.notify_agent_activity") as notify,
    ):
        with assert_raises_error(error_type="unknown_session_plan"):
            await _validate_plan_fn()(name="nope")

    notify.assert_not_called()


async def test_validate_plan_bridge_error_does_not_emit():
    with (
        patch(f"{_AUTHORING_MOD}._http_post_json", return_value=(503, {"detail": "bridge down"})),
        patch(f"{_AUTHORING_MOD}.notify_agent_activity") as notify,
    ):
        with assert_raises_error(error_type="bluesky_bridge_error"):
            await _validate_plan_fn()(name="tiny")

    notify.assert_not_called()


async def test_write_plan_result_identical_when_web_terminal_down(monkeypatch):
    """The REAL notify helper against a dead port leaves the authored plan's
    result intact — the emit is best-effort, never load-bearing."""
    monkeypatch.setenv("OSPREY_WEB_PORT", str(_dead_port()))
    with patch(f"{_AUTHORING_MOD}._http_post_json", return_value=(200, _WRITE_RESP)):
        result = await _write_plan_fn()(**_WRITE_ARGS)
    assert extract_response_dict(result) == _WRITE_RESP


async def test_validate_plan_result_identical_when_web_terminal_down(monkeypatch):
    monkeypatch.setenv("OSPREY_WEB_PORT", str(_dead_port()))
    with patch(f"{_AUTHORING_MOD}._http_post_json", return_value=(200, _PASS_RESP)):
        result = await _validate_plan_fn()(name="tiny")
    assert extract_response_dict(result) == _PASS_RESP
