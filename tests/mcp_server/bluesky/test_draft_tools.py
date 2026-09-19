"""Tests for the shared plan draft MCP tools:
``get_draft`` / ``set_draft`` / ``clear_draft``.

These tests mock the HTTP boundary
(``osprey.mcp_server.bluesky.tools.draft._http_get_json`` /
``_http_patch_json`` / ``_http_delete_json``) so they exercise only this
tool module's payload shaping and error-envelope mapping against the bridge's
draft contract, with no bridge process needed. The bridge side of that contract
is covered by ``tests/services/bluesky_bridge`` and, end to end, by
``tests/services/test_draft_roundtrip.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import draft
from osprey.registry.mcp import FRAMEWORK_SERVERS
from osprey.utils.owner_header import OWNER_HEADER
from osprey_connectors.identity import TERMINAL_USER_ENV
from osprey_connectors.posture_store import CONTROL_CONTEXT_TREE_ENV_VAR, CONTROL_OWNER_ENV_VAR
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MOD = "osprey.mcp_server.bluesky.tools.draft"


def _get_fn():
    return get_tool_fn(draft.get_draft)


def _set_fn():
    return get_tool_fn(draft.set_draft)


def _clear_fn():
    return get_tool_fn(draft.clear_draft)


def _as_terminal_user(monkeypatch, user: str) -> None:
    """Put the process in a terminal container: the roster account, no stamp over it."""
    monkeypatch.delenv(CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.delenv(CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.setenv(TERMINAL_USER_ENV, user)


def _as_owner_less_tree_container(monkeypatch, tmp_path) -> None:
    """Put the process in a container holding the state tree and owning nothing."""
    monkeypatch.delenv(CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.setenv(CONTROL_CONTEXT_TREE_ENV_VAR, str(tmp_path / "control-context"))


@pytest.fixture(autouse=True)
def _reset_server_context(tmp_path, monkeypatch):
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
# The owner on a draft write
#
# The draft is a surface the agent and the human share, so an edit made on it
# names its author the way every other write from this server does. Mirrors the
# ``queue_remove`` rows in ``test_queue_tools.py``.
# =========================================================================


async def test_a_draft_edit_names_who_made_it(monkeypatch):
    _as_terminal_user(monkeypatch, "rosterbob")
    resp = {"revision": 1, "changed": ["plan_name"], "plan_name": "grid_scan"}
    with patch(f"{_MOD}._http_patch_json", return_value=(200, resp)) as m:
        await _set_fn()(plan_name="grid_scan")

    assert m.call_args.kwargs["headers"] == {OWNER_HEADER: "rosterbob"}


async def test_a_draft_discard_names_who_discarded_it(monkeypatch):
    """Wiping the surface the human may be filling is the destructive draft
    write, and it is the one that most needs a name on it."""
    _as_terminal_user(monkeypatch, "rosterbob")
    with patch(
        f"{_MOD}._http_delete_json", return_value=(200, {"revision": 5, "cleared": True})
    ) as m:
        await _clear_fn()()

    assert m.call_args.kwargs["headers"] == {OWNER_HEADER: "rosterbob"}


async def test_an_owner_less_draft_write_sends_no_headers_at_all(tmp_path, monkeypatch):
    """Nothing to say means no header dict, not an empty one — the same
    absent-not-empty rule every other write from this server keeps."""
    _as_owner_less_tree_container(monkeypatch, tmp_path)
    with patch(f"{_MOD}._http_delete_json", return_value=(200, {"cleared": False})) as m:
        await _clear_fn()()

    assert m.call_args.kwargs["headers"] is None


async def test_a_draft_read_carries_no_owner(monkeypatch):
    """Reads name nobody: there is no change to attribute."""
    _as_terminal_user(monkeypatch, "rosterbob")
    with patch(f"{_MOD}._http_get_json", return_value=(200, {"draft": None, "revision": 1})) as m:
        await _get_fn()()

    assert "headers" not in m.call_args.kwargs


# =========================================================================
# Registry: draft tools are silent-allow, no hooks_pre, permissions_ask
# unchanged by the draft tools
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
        "queue_add",
        "queue_start",
        "queue_stop",
        "queue_remove",
        "stop_run",
        "write_plan",
        "validate_plan",
    ]
