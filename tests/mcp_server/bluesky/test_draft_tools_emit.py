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
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

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
def _reset_scan_context(tmp_path, monkeypatch):
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

    notify.assert_called_once_with(tool="set_draft", kind="panel", panel="plan", detail="grid_scan")
    # Tool result is unchanged by the emit.
    assert extract_response_dict(result) == _SET_RESP


async def test_set_draft_panel_id_resolved_from_web_panels_config():
    """A facility that registered the plan panel (mount path ``/plan/``) under
    a non-canonical ``web.panels`` key gets that id in the emit — the id is
    config-resolved, never hardcoded."""
    config = {
        "web": {
            "panels": {
                "results": {"url": "http://localhost:9000", "path": "/results/"},
                "operator-plan": {"url": "http://localhost:9000", "path": "/plan/"},
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


# =========================================================================
# Successful clear_draft → one emit with detail='cleared'
# =========================================================================


async def test_clear_draft_success_emits_cleared_detail():
    with (
        patch(f"{_MOD}._http_delete_json", return_value=(200, _CLEAR_RESP)),
        patch(f"{_MOD}.notify_agent_activity") as notify,
    ):
        result = await _clear_fn()()

    notify.assert_called_once_with(tool="clear_draft", kind="panel", panel="plan", detail="cleared")
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
