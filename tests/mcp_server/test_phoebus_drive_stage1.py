"""``phoebus_drive`` follows its machine's switch, then stage 1 of approval.

Server-side the drive is checked in this order: the deployment offers it
(``phoebus.agent_access``), the session is on the baseline, the write verdict
for the baseline allows it, and, where the approval policy asks, the approval
prompt left its record for the call. Each refusal returns before the bridge is
reached, so the HTTP boundary is patched and asserted untouched.

The verdict reads a control-context record filed under ``tmp_path``; the
approval record is an ask stamp filed by :mod:`tests._approval_stamp_fixture`.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.approval_record import APPROVAL_CONFIG_INVALID, APPROVAL_MISSING
from osprey.mcp_server.phoebus.tools import bridge_tools
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    reset_verdict_state,
    write_control_context,
)
from tests._approval_stamp_fixture import ApprovalStamps, use_stamp_dir
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MOD = "osprey.mcp_server.phoebus.tools.bridge_tools"

#: A tier that allows writes, on a deployment whose baseline (``live``) is real.
_ALLOWED = {"type": "epics", "writes": "allowed"}

#: The armed posture an older config states: no ``writes`` key, a connector
#: leaf set true. The tier reads it as ``allowed``; the switch still decides.
_OLD_ARMED = {"type": "epics", "connector": {"epics": {"writes_enabled": True}}}

#: Approval that asks for every tool.
_ASKING = {"enabled": True, "default_policy": "always"}

_OFF_PHRASE = refusal_phrase("switch_off", "real")

_OK = (200, {"fired": True, "detail": "ok"})


@pytest.fixture(autouse=True)
def _clean_verdict_env(tmp_path, monkeypatch):
    """A pinned identity and a record root under tmp_path, and nothing ambient."""
    isolate_verdict_environment(monkeypatch, tmp_path / "agent_data")
    monkeypatch.setenv("PHOEBUS_REQUIRE_HANDLE", "0")
    monkeypatch.chdir(tmp_path)
    yield
    reset_verdict_state()


@pytest.fixture
def stamps(tmp_path, monkeypatch) -> ApprovalStamps:
    return use_stamp_dir(monkeypatch, tmp_path / "state")


def _deploy(tmp_path, monkeypatch, *, section=_ALLOWED, switch_on=True, approval=_ASKING):
    """Write a config offering the drive, and a record with the baseline's switch."""
    config: dict = {"phoebus": {"agent_access": "read_write"}, "control_system": section}
    if approval is not None:
        config["approval"] = approval
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(config))
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    write_control_context(
        tmp_path / "agent_data", target="live", switch={"live": "on"} if switch_on else None
    )
    reset_verdict_state()


def _drive():
    return get_tool_fn(bridge_tools.phoebus_drive)


# ── the switch ──────────────────────────────────────────────────────────────
async def test_switch_off_refuses_with_the_off_phrase(tmp_path, monkeypatch, stamps):
    _deploy(tmp_path, monkeypatch, switch_on=False)
    with patch(f"{_MOD}._http_post_drive") as post:
        with stamps.approved("phoebus_drive"):
            with assert_raises_error(error_type="writes_disabled") as ctx:
                await _drive()(widget="SetButton", verb="click")
    post.assert_not_called()
    envelope = ctx["envelope"]
    assert _OFF_PHRASE in envelope["error_message"]
    assert envelope["details"]["reason"] == "switch_off"
    assert envelope["details"]["target"] == "live"


async def test_tier_never_refuses_even_with_the_switch_on(tmp_path, monkeypatch, stamps):
    _deploy(tmp_path, monkeypatch, section={"type": "epics", "writes": "never"})
    with patch(f"{_MOD}._http_post_drive") as post:
        with stamps.approved("phoebus_drive"):
            with assert_raises_error(error_type="writes_disabled") as ctx:
                await _drive()(widget="SetButton", verb="click")
    post.assert_not_called()
    assert ctx["envelope"]["details"]["reason"] == "tier_never"


async def test_skew_switch_off_record_under_old_armed_config_refuses(tmp_path, monkeypatch, stamps):
    """An older config that armed the connector does not stand in for the switch."""
    _deploy(tmp_path, monkeypatch, section=_OLD_ARMED, switch_on=False)
    with patch(f"{_MOD}._http_post_drive") as post:
        with stamps.approved("phoebus_drive"):
            with assert_raises_error(error_type="writes_disabled") as ctx:
                await _drive()(widget="SetButton", verb="click")
    post.assert_not_called()
    assert _OFF_PHRASE in ctx["envelope"]["error_message"]
    assert ctx["envelope"]["details"]["reason"] == "switch_off"


async def test_old_armed_config_drives_with_the_switch_on(tmp_path, monkeypatch, stamps):
    _deploy(tmp_path, monkeypatch, section=_OLD_ARMED)
    with patch(f"{_MOD}._http_post_drive", return_value=_OK) as post:
        with stamps.approved("phoebus_drive"):
            await _drive()(widget="SetButton", verb="click")
    post.assert_called_once()


async def test_unreachable_verdict_refuses_record_unavailable(tmp_path, monkeypatch, stamps):
    _deploy(tmp_path, monkeypatch)
    with (
        patch(f"{_MOD}.posture_store.write_verdict", side_effect=RuntimeError("boom")),
        patch(f"{_MOD}._http_post_drive") as post,
        stamps.approved("phoebus_drive"),
        assert_raises_error(error_type="writes_disabled") as ctx,
    ):
        await _drive()(widget="SetButton", verb="click")
    post.assert_not_called()
    assert ctx["envelope"]["details"]["reason"] == "record_unavailable"


async def test_the_refusal_names_no_config_key(tmp_path, monkeypatch, stamps):
    _deploy(tmp_path, monkeypatch, switch_on=False)
    with stamps.approved("phoebus_drive"):
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _drive()(widget="SetButton", verb="click")
    text = json.dumps(ctx["envelope"])
    for banned in ("writes_enabled", "profile.yml", "config.yml", "osprey build", "redeploy"):
        assert banned not in text


# ── the approval record ─────────────────────────────────────────────────────
async def test_missing_record_refuses_where_approval_applies(tmp_path, monkeypatch, stamps):  # noqa: ARG001 - the stamp dir is empty
    _deploy(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_post_drive") as post:
        with assert_raises_error(error_type=APPROVAL_MISSING) as ctx:
            await _drive()(widget="SetButton", verb="click")
    post.assert_not_called()
    assert ctx["envelope"]["error_message"].startswith("not run:")


async def test_unstated_approval_asks_and_so_needs_a_record(tmp_path, monkeypatch, stamps):  # noqa: ARG001
    _deploy(tmp_path, monkeypatch, approval=None)
    with patch(f"{_MOD}._http_post_drive") as post:
        with assert_raises_error(error_type=APPROVAL_MISSING):
            await _drive()(widget="SetButton", verb="click")
    post.assert_not_called()


async def test_a_stamp_for_another_tool_is_no_record(tmp_path, monkeypatch, stamps):
    _deploy(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_post_drive") as post:
        with stamps.approved("queue_start"):
            with assert_raises_error(error_type=APPROVAL_MISSING):
                await _drive()(widget="SetButton", verb="click")
    post.assert_not_called()


async def test_a_stamped_drive_reaches_the_bridge(tmp_path, monkeypatch, stamps):
    _deploy(tmp_path, monkeypatch)
    with patch(f"{_MOD}._http_post_drive", return_value=_OK) as post:
        with stamps.approved("phoebus_drive", approver="dispatch_policy"):
            result = await _drive()(widget="SetButton", verb="click")
    post.assert_called_once()
    assert extract_response_dict(result)["status"] == "success"


@pytest.mark.parametrize(
    "approval",
    [
        {"enabled": False},
        {"enabled": True, "default_policy": "skip"},
        {"enabled": True, "default_policy": "always", "tools": {"phoebus_drive": "skip"}},
    ],
)
async def test_runs_unchanged_where_approval_does_not_apply(tmp_path, monkeypatch, approval):
    """No stamp, no call scope: a drive nobody is asked about runs as before."""
    _deploy(tmp_path, monkeypatch, approval=approval)
    with patch(f"{_MOD}._http_post_drive", return_value=_OK) as post:
        result = await _drive()(widget="SetButton", verb="click")
    post.assert_called_once()
    assert extract_response_dict(result)["fired"] is True


async def test_an_approval_switch_that_is_not_a_bool_refuses(tmp_path, monkeypatch, stamps):
    _deploy(tmp_path, monkeypatch, approval={"enabled": "yes"})
    with patch(f"{_MOD}._http_post_drive") as post:
        with stamps.approved("phoebus_drive"):
            with assert_raises_error(error_type=APPROVAL_CONFIG_INVALID):
                await _drive()(widget="SetButton", verb="click")
    post.assert_not_called()


# ── the order ───────────────────────────────────────────────────────────────
async def test_the_verdict_refuses_before_the_record_is_read(tmp_path, monkeypatch, stamps):  # noqa: ARG001
    """Off and unstamped: the call is refused by the switch, not by stage 1."""
    _deploy(tmp_path, monkeypatch, switch_on=False)
    with assert_raises_error(error_type="writes_disabled"):
        await _drive()(widget="SetButton", verb="click")


async def test_the_offer_refuses_before_the_verdict(tmp_path, monkeypatch):
    _deploy(tmp_path, monkeypatch, switch_on=False)
    config = yaml.safe_load((tmp_path / "config.yml").read_text())
    config["phoebus"]["agent_access"] = "read"
    (tmp_path / "config.yml").write_text(yaml.dump(config))
    with assert_raises_error(error_type="not_supported"):
        await _drive()(widget="SetButton", verb="click")


async def test_the_target_refuses_before_the_verdict(tmp_path, monkeypatch):
    _deploy(tmp_path, monkeypatch, switch_on=False)
    write_control_context(tmp_path / "agent_data", target="va")
    reset_verdict_state()
    with assert_raises_error(error_type="target_switched"):
        await _drive()(widget="SetButton", verb="click")


async def test_the_record_refuses_before_argument_validation(tmp_path, monkeypatch, stamps):  # noqa: ARG001
    _deploy(tmp_path, monkeypatch)
    with assert_raises_error(error_type=APPROVAL_MISSING):
        await _drive()(widget="0", verb="frobnicate")


async def test_a_stamped_drive_still_validates_its_arguments(tmp_path, monkeypatch, stamps):
    _deploy(tmp_path, monkeypatch)
    with stamps.approved("phoebus_drive"):
        with assert_raises_error(error_type="validation_error"):
            await _drive()(widget="0", verb="frobnicate")
