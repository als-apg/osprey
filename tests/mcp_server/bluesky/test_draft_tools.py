"""Tests for the shared plan draft MCP tools (task 2.1):
``get_draft`` / ``set_draft`` / ``clear_draft``.

The real bridge draft module (`bluesky_bridge/draft.py`) is being built in
parallel — these tests mock the HTTP boundary
(``osprey.mcp_server.bluesky.tools.draft._http_get_json`` /
``_http_patch_json`` / ``_http_delete_json``) so they exercise only this
tool module's payload shaping and error-envelope mapping against the
PROPOSAL.md rev-4 bridge contract, with no bridge process needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import draft
from osprey.registry.mcp import FRAMEWORK_SERVERS
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.draft"


def _get_fn():
    return get_tool_fn(draft.get_draft)


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


# =========================================================================
# get_draft
# =========================================================================


async def test_get_draft_happy_path_returns_body():
    body = {
        "draft": {"plan_name": "grid_scan", "plan_args": {"num": 3}},
        "revision": 4,
    }
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)) as m:
        result = await _get_fn()()

    assert m.call_args.args[0] == "/draft"
    assert extract_response_dict(result) == body


async def test_get_draft_null_draft_still_carries_revision():
    body = {"draft": None, "revision": 7}
    with patch(f"{_MOD}._http_get_json", return_value=(200, body)):
        result = await _get_fn()()
    assert extract_response_dict(result) == body


async def test_get_draft_non_200_maps_to_generic_bridge_error():
    with patch(f"{_MOD}._http_get_json", return_value=(500, {"detail": "boom"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _get_fn()()
    assert "boom" in ctx["envelope"]["error_message"]


# =========================================================================
# set_draft — payload shaping
# =========================================================================


async def test_set_draft_plan_name_only_posts_client_id_and_plan_name():
    resp = {"revision": 1, "changed": ["plan_name"], "plan_name": "grid_scan"}
    with patch(f"{_MOD}._http_patch_json", return_value=(200, resp)) as m:
        result = await _set_fn()(plan_name="grid_scan")

    assert m.call_args.args[0] == "/draft"
    assert m.call_args.args[1] == {"client_id": "mcp-agent", "plan_name": "grid_scan"}
    assert extract_response_dict(result) == resp


async def test_set_draft_patch_only_posts_client_id_and_patch():
    resp = {"revision": 2, "changed": ["num"], "plan_name": "grid_scan"}
    with patch(f"{_MOD}._http_patch_json", return_value=(200, resp)) as m:
        result = await _set_fn()(plan_args_patch={"num": 5})

    assert m.call_args.args[1] == {
        "client_id": "mcp-agent",
        "plan_args_patch": {"num": 5},
    }
    assert extract_response_dict(result) == resp


async def test_set_draft_remove_only_posts_client_id_and_remove():
    resp = {"revision": 3, "changed": ["num"], "plan_name": "grid_scan"}
    with patch(f"{_MOD}._http_patch_json", return_value=(200, resp)) as m:
        result = await _set_fn()(remove=["num"])

    assert m.call_args.args[1] == {"client_id": "mcp-agent", "remove": ["num"]}
    assert extract_response_dict(result) == resp


async def test_set_draft_combined_arguments_all_present_in_payload():
    resp = {"revision": 4, "changed": ["plan_name", "num"], "plan_name": "grid_scan"}
    with patch(f"{_MOD}._http_patch_json", return_value=(200, resp)) as m:
        await _set_fn()(plan_name="grid_scan", plan_args_patch={"num": 2}, remove=["old_key"])

    assert m.call_args.args[1] == {
        "client_id": "mcp-agent",
        "plan_name": "grid_scan",
        "plan_args_patch": {"num": 2},
        "remove": ["old_key"],
    }


# =========================================================================
# set_draft — error mapping
# =========================================================================


async def test_set_draft_no_argument_errors_without_calling_bridge():
    with patch(f"{_MOD}._http_patch_json") as m:
        with assert_raises_error(error_type="set_draft_no_argument") as ctx:
            await _set_fn()()
    m.assert_not_called()
    assert "no argument" in ctx["envelope"]["error_message"].lower()


async def test_set_draft_no_draft_409_surfaces_bridge_message_and_hint():
    body = {"code": "no_draft", "detail": "no draft exists"}
    with patch(f"{_MOD}._http_patch_json", return_value=(409, body)):
        with assert_raises_error(error_type="no_draft") as ctx:
            await _set_fn()(plan_args_patch={"num": 1})

    assert "no draft exists" in ctx["envelope"]["error_message"]
    assert any("pass plan_name to create one" in s for s in ctx["envelope"]["suggestions"])


async def test_set_draft_other_409_without_no_draft_code_is_generic_bridge_error():
    """A 409 lacking code == 'no_draft' (e.g. expected_plan_name mismatch) must
    NOT be misclassified as the no-draft case."""
    body = {"detail": "plan_name mismatch"}
    with patch(f"{_MOD}._http_patch_json", return_value=(409, body)):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _set_fn()(plan_args_patch={"num": 1})
    assert "plan_name mismatch" in ctx["envelope"]["error_message"]


async def test_set_draft_unknown_plan_422_guides_validate_session_plan_first():
    body = {"detail": "unknown plan 'nope'"}
    with patch(f"{_MOD}._http_patch_json", return_value=(422, body)):
        with assert_raises_error(error_type="unknown_plan") as ctx:
            await _set_fn()(plan_name="nope")

    assert "unknown plan" in ctx["envelope"]["error_message"]
    assert any("validate the session plan first" in s for s in ctx["envelope"]["suggestions"])


async def test_set_draft_other_non_200_is_generic_bridge_error():
    with patch(f"{_MOD}._http_patch_json", return_value=(500, {"detail": "internal error"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _set_fn()(plan_name="grid_scan")
    assert "internal error" in ctx["envelope"]["error_message"]


# =========================================================================
# clear_draft — idempotence
# =========================================================================


async def test_clear_draft_happy_path():
    resp = {"revision": 5, "cleared": True}
    with patch(f"{_MOD}._http_delete_json", return_value=(200, resp)) as m:
        result = await _clear_fn()()

    assert m.call_args.args[0] == "/draft?client_id=mcp-agent"
    assert extract_response_dict(result) == resp


async def test_clear_draft_idempotent_when_no_draft_exists():
    """Bridge returns 200 no-op either way — clear_draft never errors here."""
    resp = {"revision": 5, "cleared": False}
    with patch(f"{_MOD}._http_delete_json", return_value=(200, resp)):
        result = await _clear_fn()()
    assert extract_response_dict(result) == resp


async def test_clear_draft_non_200_maps_to_generic_bridge_error():
    with patch(f"{_MOD}._http_delete_json", return_value=(503, {"detail": "unavailable"})):
        with assert_raises_error(error_type="bluesky_bridge_error") as ctx:
            await _clear_fn()()
    assert "unavailable" in ctx["envelope"]["error_message"]


# =========================================================================
# Registry: draft tools are silent-allow, no hooks_pre, permissions_ask
# unchanged (task 2.1's registry edit)
# =========================================================================


def test_draft_tools_are_silent_allow_no_hooks():
    bluesky_def = FRAMEWORK_SERVERS["bluesky"]
    for tool in ("get_draft", "set_draft", "clear_draft"):
        assert tool in bluesky_def.permissions_allow
        assert tool not in bluesky_def.permissions_ask

    by_matcher = {rule.matcher: rule for rule in bluesky_def.hooks_pre}
    for tool in ("get_draft", "set_draft", "clear_draft"):
        assert f"mcp__bluesky__{tool}" not in by_matcher


def test_permissions_ask_unchanged_by_draft_tools():
    bluesky_def = FRAMEWORK_SERVERS["bluesky"]
    assert bluesky_def.permissions_ask == [
        "launch_run",
        "stop_run",
        "write_plan",
        "validate_plan",
    ]
