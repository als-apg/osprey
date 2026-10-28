"""Stage 1 of approval on the Bluesky arming calls.

``queue_start`` and the armed ``queue_add`` run only on the record their
approval prompt left (``osprey.mcp_server.approval_record``), where the
deployment's approval policy asks. The order is the contract:

1. **The verdict first.** An off ``queue_start`` is refused for being off
   (``writes_disabled``, ``switch_off``), not for a missing record.
2. **Stage 1 next, before the bridge.** An on ``queue_start``, and a
   ``queue_add`` the verdict allows, with no record is refused
   ``approval_missing`` with zero HTTP calls.
3. **An unarmed add is not a write.** An off ``queue_add`` with no record is
   still added, unarmed, with the off phrase.

The HTTP boundary is patched; the stamps come from
``tests/_approval_stamp_fixture.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.approval_record import APPROVAL_CONFIG_INVALID, APPROVAL_MISSING
from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import queue
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    reset_verdict_state,
    write_control_context,
)
from tests._approval_stamp_fixture import ApprovalStamps, use_stamp_dir
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MOD = "osprey.mcp_server.bluesky.tools.queue"

_TOKEN = "genuinely-valid-token"

#: A tier that allows writes; the baseline ``live`` is real and the single
#: lane, declaring no target, serves it.
_ALLOWED_SECTION = {"type": "epics", "writes": "allowed"}

#: An approval policy that asks about every tool.
_ASKING = {"enabled": True, "default_policy": "always"}

_OFF_PHRASE = refusal_phrase("switch_off", "real")

_ADDED = {"run_id": "abc", "revision": 7, "item": {"item_uid": "u1"}}


@pytest.fixture(autouse=True)
def _reset_bluesky_context():
    yield
    reset_server_context()


@pytest.fixture(autouse=True)
def _clean_verdict_env(tmp_path, monkeypatch):
    isolate_verdict_environment(monkeypatch, tmp_path / "agent_data")
    yield
    reset_verdict_state()


@pytest.fixture(autouse=True)
def stamps(tmp_path, monkeypatch) -> ApprovalStamps:
    """A fresh stamp directory for every case, so no ambient stamp is read."""
    return use_stamp_dir(monkeypatch, tmp_path / "state")


def _configure(tmp_path, monkeypatch, *, on: bool, approval: dict | None = None) -> None:
    """A project whose approval policy asks, with the baseline switched *on* or off."""
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    config = {
        "control_system": _ALLOWED_SECTION,
        "approval": _ASKING if approval is None else approval,
    }
    (project / "config.yml").write_text(yaml.dump(config))
    monkeypatch.chdir(project)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    write_control_context(
        tmp_path / "agent_data", target="live", switch={"live": "on"} if on else None
    )
    reset_verdict_state()
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", _TOKEN)
    initialize_server_context()


def _launch_token(m) -> str | None:
    """The launch token the one forwarded request carried, or ``None``."""
    return (m.call_args.kwargs["headers"] or {}).get("X-Launch-Token")


# =========================================================================
# A missing record, for each tool
# =========================================================================


async def test_an_on_queue_start_without_a_record_is_refused_before_the_bridge(
    tmp_path, monkeypatch
):
    _configure(tmp_path, monkeypatch, on=True)

    with patch(f"{_MOD}._http_post_json") as m:
        with assert_raises_error(error_type=APPROVAL_MISSING) as ctx:
            await get_tool_fn(queue.queue_start)()

    m.assert_not_called()
    envelope = ctx["envelope"]
    assert envelope["error_message"].startswith("not run: ")
    assert "queue_start was refused before reaching the bridge" in envelope["error_message"]
    assert envelope["details"]["lane"] == "bluesky"


async def test_an_armed_queue_add_without_a_record_is_refused_before_the_bridge(
    tmp_path, monkeypatch
):
    _configure(tmp_path, monkeypatch, on=True)

    with patch(f"{_MOD}._http_post_json") as m:
        with assert_raises_error(error_type=APPROVAL_MISSING) as ctx:
            await get_tool_fn(queue.queue_add)(draft_revision=7)

    m.assert_not_called()
    assert "queue_add was refused before reaching the bridge" in (ctx["envelope"]["error_message"])


async def test_a_record_for_the_other_tool_does_not_start_the_queue(tmp_path, monkeypatch, stamps):
    """An add's record is not a start's: the stamp must name the tool called."""
    _configure(tmp_path, monkeypatch, on=True)

    with stamps.approved("queue_add"):
        with patch(f"{_MOD}._http_post_json") as m:
            with assert_raises_error(error_type=APPROVAL_MISSING):
                await get_tool_fn(queue.queue_start)()
    m.assert_not_called()


async def test_a_malformed_approval_switch_refuses_the_start(tmp_path, monkeypatch, stamps):
    _configure(tmp_path, monkeypatch, on=True, approval={"enabled": "yes"})

    with stamps.approved("queue_start"):
        with patch(f"{_MOD}._http_post_json") as m:
            with assert_raises_error(error_type=APPROVAL_CONFIG_INVALID):
                await get_tool_fn(queue.queue_start)()
    m.assert_not_called()


# =========================================================================
# The verdict comes first
# =========================================================================


async def test_an_off_queue_start_is_refused_for_being_off(tmp_path, monkeypatch):
    """No record either, yet the refusal is the verdict's: off is the reason."""
    _configure(tmp_path, monkeypatch, on=False)

    with patch(f"{_MOD}._http_post_json") as m:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await get_tool_fn(queue.queue_start)()

    m.assert_not_called()
    assert ctx["envelope"]["details"]["reason"] == "switch_off"
    assert ctx["envelope"]["error_message"].startswith(_OFF_PHRASE)


async def test_an_off_queue_add_without_a_record_is_added_unarmed(tmp_path, monkeypatch):
    """An unarmed add moves nothing, so stage 1 does not ask for its record."""
    _configure(tmp_path, monkeypatch, on=False)

    body = {**_ADDED, "armed": False}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await get_tool_fn(queue.queue_add)(draft_revision=7)

    assert _launch_token(m) is None
    refused = extract_response_dict(result)["writes_refused"]
    assert refused["reason"] == "switch_off"
    assert refused["message"] == f"Added without the launch token. {_OFF_PHRASE}"


# =========================================================================
# A record lets the call through
# =========================================================================


async def test_an_armed_queue_add_with_a_record_is_added_armed(tmp_path, monkeypatch, stamps):
    _configure(tmp_path, monkeypatch, on=True)

    body = {**_ADDED, "armed": True}
    with stamps.approved("queue_add"):
        with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
            with patch(f"{_MOD}.notify_agent_activity_async"):
                result = await get_tool_fn(queue.queue_add)(draft_revision=7)

    assert _launch_token(m) == _TOKEN
    payload = extract_response_dict(result)
    assert payload["armed"] is True
    assert "writes_refused" not in payload


async def test_an_on_queue_start_with_a_record_reaches_the_bridge(tmp_path, monkeypatch, stamps):
    _configure(tmp_path, monkeypatch, on=True)

    with stamps.approved("queue_start"):
        with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as m:
            with patch(f"{_MOD}.notify_agent_activity_async"):
                result = await get_tool_fn(queue.queue_start)()

    assert m.call_args.args[0] == "/queue/start"
    assert _launch_token(m) == _TOKEN
    assert extract_response_dict(result)["started"] is True


async def test_approval_off_needs_no_record(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, on=True, approval={"enabled": False})

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await get_tool_fn(queue.queue_start)()
    m.assert_called_once()
