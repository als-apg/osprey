"""The write verdict in the Bluesky plan lane's arming gate.

The queue surface asks the write verdict
(``osprey_connectors.posture_store.write_verdict``) for the control target its
bound lane serves, fresh on every call, before anything is POSTed. Writes are
armed here the way a person arms them: a tier that allows writes and a record
whose switch is on for the machine. A refusal is the verdict's phrase
(``osprey_connectors.refusals.refusal_phrase``), which names who can lift it
and never a config key.

The properties asserted directly:

1. **An off machine refuses the arming calls.** ``queue_start`` refuses with
   zero HTTP calls; ``queue_add`` is still added, unarmed — the token is
   withheld and the result carries the off phrase.
2. **The verdict is per target.** A switch on for the simulator arms the lane
   serving it while the stand-in stays off, and the reverse.
3. **Every refusing term is the verdict's own.** Tier, unknown target (a
   degraded ``live`` lane), unknown owner, unreadable record, read-only run, a
   dispatch run with no owner: each refuses ``queue_start`` with its reason and
   phrase, and each withholds the token from ``queue_add``.
4. **A withdrawal follows the halted lane.** ``queue_stop(cancel=True)`` is
   given the verdict of the lane it is addressed to, and is refused when no
   rendered lane is found for it. A plain halt is never gated.

The bridge's own startup guard (``bluesky_bridge.validation``) is covered at
the end of this file; that part belongs to the bridge task.

The HTTP boundary is patched, so these exercise this module's gating and
refusal wording against the queue wire contract with no bridge process.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import queue
from osprey.utils.owner_header import OWNER_HEADER
from osprey_connectors import control_context, posture_store
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    reset_verdict_state,
    write_control_context,
)
from osprey_connectors_tests._harness._write_verdict_cases import _stage_tree
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MOD = "osprey.mcp_server.bluesky.tools.queue"

_TOKEN = "genuinely-valid-token"

#: A tier that allows writes. Baseline is ``live`` (``type: epics``); the lane
#: declares whichever target a test needs, ``standin`` or ``va`` (a simulator).
#: No controls server publishes a report here, so the verdict names ``standin``
#: the real machine (C-KIND): a bluesky process never vouches for a stand-in.
_ALLOWED_SECTION = {"type": "epics", "writes": "allowed"}

#: The old-model section the bridge startup-guard cases at the end still read.
#: Those cases belong to the bridge task, which moves them to the verdict.
_ARMED_SECTION = {
    "type": "epics",
    "writes_enabled": False,
    "connector": {
        "epics": {"prefix": "X:", "writes_enabled": True},
        "virtual_accelerator": {"host": "localhost", "writes_enabled": True},
        "live_standin": {"prefix": "S:", "writes_enabled": True},
    },
}

#: Words a refusal must never carry: config keys, files and build commands.
_BANNED = ("writes_enabled", "profile.yml", "config.yml", "osprey build", "redeploy")


@pytest.fixture(autouse=True)
def _reset_bluesky_context():
    yield
    reset_server_context()


@pytest.fixture(autouse=True)
def _clean_posture_env(tmp_path, monkeypatch):
    """No ambient verdict state: a pinned identity and a record root under tmp_path."""
    isolate_verdict_environment(monkeypatch, tmp_path / "agent_data")
    yield
    reset_verdict_state()


def _configure(
    tmp_path,
    monkeypatch,
    *,
    lane_target: str | None = None,
    section: dict | None = None,
    token: str | None = _TOKEN,
) -> None:
    """Render a project whose single plan lane declares *lane_target*."""
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    # No approval hook runs in front of these direct calls, so approval is off
    # and stage 1 asks for no record (test_queue_approval_record.py has it on).
    config: dict = {
        "control_system": _ALLOWED_SECTION if section is None else section,
        "approval": {"enabled": False},
    }
    if lane_target is not None:
        config["services"] = {"bluesky": {"target": lane_target}}
    (project / "config.yml").write_text(yaml.dump(config))
    monkeypatch.chdir(project)
    if token is None:
        monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", token)
    initialize_server_context()


def _switch(tmp_path, *on: str) -> Path:
    """Write the host record, pointed at the baseline, with *on* switched on."""
    path = write_control_context(
        tmp_path / "agent_data", target="live", switch=dict.fromkeys(on, "on")
    )
    reset_verdict_state()
    return path


def _switch_off(tmp_path, monkeypatch, target: str) -> Path:
    """A host record with *target* switched off, for the bridge startup-guard cases."""
    root = tmp_path / "agent_data"
    path = write_control_context(root, switch={target: "off"})
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    reset_verdict_state()
    return path


def _add_fn():
    return get_tool_fn(queue.queue_add)


def _start_fn():
    return get_tool_fn(queue.queue_start)


def _stop_fn():
    return get_tool_fn(queue.queue_stop)


def _headers_without_owner(m) -> dict[str, str]:
    """The headers one forwarded request carried, minus the owner stamp."""
    headers = m.call_args.kwargs["headers"] or {}
    return {name: value for name, value in headers.items() if name != OWNER_HEADER}


def _refusal(code: str, detail: str, **extras) -> dict:
    return {"detail": {"code": code, "detail": detail, **extras}}


def _assert_verdict_refusal(envelope: dict, reason: str, kind: str | None) -> None:
    """The envelope is the verdict's refusal: its reason, its phrase, no key."""
    assert envelope["error_type"] == "writes_disabled"
    assert envelope["details"]["reason"] == reason
    assert refusal_phrase(reason, kind) in envelope["error_message"]
    text = " ".join([envelope["error_message"], *envelope["suggestions"]])
    assert not any(word in text for word in _BANNED)
    assert _TOKEN not in text


async def _start_refused(reason: str, kind: str | None) -> None:
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _start_fn()()
    mock_post.assert_not_called()
    _assert_verdict_refusal(ctx["envelope"], reason, kind)


async def _add_unarmed(reason: str, kind: str | None):
    """Add a draft, asserting it went out unarmed with the verdict's phrase."""
    body = {"run_id": "r1", "revision": 3, "item": {"item_uid": "u1"}}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _add_fn()(draft_revision=3)
    assert _headers_without_owner(m) == {}
    refused = extract_response_dict(result)["writes_refused"]
    assert refused["reason"] == reason
    assert refusal_phrase(reason, kind) in refused["message"]
    return m


# =========================================================================
# An off machine refuses the arming calls
# =========================================================================


async def test_an_off_lane_target_refuses_queue_start_with_zero_http_calls(tmp_path, monkeypatch):
    """The switch is a kill switch for one machine, enforced at write time.

    The tier allows writes and the caller holds a valid launch token; the ONLY
    thing refusing is the switch, which is off for ``standin``.
    """
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path)

    await _start_refused("switch_off", "real")


async def test_the_start_refusal_names_the_lane_and_target_in_details(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path)

    with patch(f"{_MOD}._http_post_json"):
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _start_fn()()

    details = ctx["envelope"]["details"]
    assert details["target"] == "standin"
    assert details["lane"] == "bluesky"


async def test_an_off_lane_target_adds_unarmed_with_the_off_phrase(tmp_path, monkeypatch):
    """Composing still works; arming does not.

    An item added to an idle queue moves nothing, so an off switch must not
    turn enqueue into a refusal. It withholds the token, so the bridge refuses
    the add the moment the queue is draining, and says why in the result.
    """
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path)

    await _add_unarmed("switch_off", "real")


async def test_the_withheld_token_hint_carries_the_off_phrase(tmp_path, monkeypatch):
    """The bridge cannot know WHY the token was withheld; this server says so."""
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path)

    body = _refusal(
        "launch_token_required",
        "the queue is running, so adding an item requires the launch token",
        manager_state="executing_queue",
    )
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)):
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _add_fn()(draft_revision=7)

    hints = [s for s in ctx["envelope"]["suggestions"] if "withheld the launch token" in s]
    assert len(hints) == 1
    assert refusal_phrase("switch_off", "real") in hints[0]
    assert "'bluesky'" in hints[0]
    assert not any(word in hints[0] for word in _BANNED)


async def test_an_on_lane_target_adds_armed_and_starts(tmp_path, monkeypatch):
    """Contrast case: the refusals above are gated by the switch, not vacuous."""
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path, "standin")

    body = {"run_id": "abc", "revision": 7, "item": {"item_uid": "u1"}}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _add_fn()(draft_revision=7)
    assert _headers_without_owner(m) == {"X-Launch-Token": _TOKEN}
    assert "writes_refused" not in extract_response_dict(result)

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()
    assert _headers_without_owner(m) == {"X-Launch-Token": _TOKEN}


async def test_a_switch_turned_off_reaches_the_next_start(tmp_path, monkeypatch):
    """Asked fresh on every call: nothing is cached between two starts."""
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path, "standin")
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})):
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()

    _switch(tmp_path)
    await _start_refused("switch_off", "real")


# =========================================================================
# The verdict is per target
# =========================================================================


async def test_a_simulator_lane_runs_while_the_real_machine_is_off(tmp_path, monkeypatch):
    """The simulator-lane criterion: the real machine off, the simulator on.

    A switch names ONE machine, and the lane's own target indexes it: with only
    ``va`` on, a lane serving ``va`` adds armed and starts.
    """
    _configure(tmp_path, monkeypatch, lane_target="va")
    _switch(tmp_path, "va")

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r"})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=7)
    assert _headers_without_owner(m) == {"X-Launch-Token": _TOKEN}

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()
    assert _headers_without_owner(m) == {"X-Launch-Token": _TOKEN}


async def test_another_targets_switch_does_not_arm_this_lane(tmp_path, monkeypatch):
    """The converse: ``va`` on says nothing about the stand-in."""
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path, "va")

    await _start_refused("switch_off", "real")


# =========================================================================
# Every refusing term is the verdict's own
# =========================================================================


async def test_a_tier_that_allows_nothing_refuses_whatever_the_switch(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, section={"type": "epics", "writes": "never"})
    _switch(tmp_path, "live")

    await _start_refused("tier_never", "real")
    await _add_unarmed("tier_never", "real")


async def test_a_degraded_live_lane_refuses_target_unknown(tmp_path, monkeypatch):
    """A live lane writes only when its live connector block is configured.

    This deployment is a virtual-accelerator one; a lane declaring ``live`` has
    no real machine to write to, so the verdict cannot name its target.
    """
    _configure(
        tmp_path,
        monkeypatch,
        lane_target="live",
        section={"type": "virtual_accelerator", "writes": "allowed"},
    )
    write_control_context(tmp_path / "agent_data", target="va", switch={"va": "on", "live": "on"})
    reset_verdict_state()

    await _start_refused("target_unknown", None)


async def test_an_owner_missing_from_the_login_map_refuses_owner_unknown(tmp_path, monkeypatch):
    """Under a bound tree the owner must be named by the login map."""
    section = {"type": "epics", "writes": "allowed", "login_tiers": {"bob": "allowed"}}
    _configure(tmp_path, monkeypatch, section=section)
    _stage_tree(
        tmp_path,
        {"bind": "provisioned", "records": {"alice": {"status": "ok", "switch": {"live": "on"}}}},
        monkeypatch,
    )
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    reset_verdict_state()

    await _start_refused("owner_unknown", "real")
    await _add_unarmed("owner_unknown", "real")


async def test_an_owned_item_writes_through_its_owners_switch(tmp_path, monkeypatch):
    """Contrast case: the same owner, mapped, with the switch on, is armed."""
    section = {"type": "epics", "writes": "allowed", "login_tiers": {"alice": "allowed"}}
    _configure(tmp_path, monkeypatch, section=section)
    _stage_tree(
        tmp_path,
        {"bind": "provisioned", "records": {"alice": {"status": "ok", "switch": {"live": "on"}}}},
        monkeypatch,
    )
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    reset_verdict_state()

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()
    assert m.call_args.kwargs["headers"] == {"X-Launch-Token": _TOKEN, OWNER_HEADER: "alice"}


async def test_an_unreadable_record_refuses_record_unavailable(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    path = _switch(tmp_path, "live")
    path.write_text("{not json", encoding="utf-8")
    control_context.invalidate_cache()
    reset_verdict_state()

    await _start_refused("record_unavailable", "real")
    await _add_unarmed("record_unavailable", "real")


async def test_a_readonly_run_refuses_with_the_switch_on(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch)
    _switch(tmp_path, "live")
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    await _start_refused("readonly_run", "real")
    await _add_unarmed("readonly_run", "real")


@pytest.mark.parametrize("tree", [False, True], ids=["host", "bound-tree"])
async def test_an_owner_less_dispatch_run_refuses_owner_unscheduled(tmp_path, monkeypatch, tree):
    """A scheduled job runs for no login, so no switch covers its plans.

    ``queue_start`` refuses ``owner_unscheduled``; ``queue_add`` withholds the
    token and, naming nobody, carries no owner header.
    """
    _configure(tmp_path, monkeypatch)
    _switch(tmp_path, "live")
    if tree:
        _stage_tree(tmp_path, {"bind": "provisioned", "records": {}}, monkeypatch)
    monkeypatch.setenv(posture_store.DISPATCH_RUN_ENV_VAR, "1")
    reset_verdict_state()

    await _start_refused("owner_unscheduled", "real")
    m = await _add_unarmed("owner_unscheduled", "real")
    assert OWNER_HEADER not in (m.call_args.kwargs["headers"] or {})


# =========================================================================
# A withdrawal follows the halted lane
# =========================================================================


async def test_withdrawing_a_halt_is_refused_while_the_halted_lanes_machine_is_off(
    tmp_path, monkeypatch
):
    """Withdrawing a halt resumes motion, so it is given that lane's verdict."""
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path, "live")

    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _stop_fn()(cancel=True)

    mock_post.assert_not_called()
    _assert_verdict_refusal(ctx["envelope"], "switch_off", "real")


async def test_withdrawing_a_halt_posts_while_the_halted_lanes_machine_is_on(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path, "standin")

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"stop_pending": False})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _stop_fn()(cancel=True)

    assert m.call_args.args[1] == {"cancel": True}
    assert _headers_without_owner(m) == {"X-Launch-Token": _TOKEN}


async def test_withdrawing_a_halt_with_no_lane_found_is_refused(tmp_path, monkeypatch):
    """A halt lane no rendered lane carries has no machine to ask the verdict about."""
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _switch(tmp_path, "standin")

    async def _nowhere():
        return "no-such-lane"

    monkeypatch.setattr(queue, "resolve_halt_lane", _nowhere)
    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _stop_fn()(cancel=True)

    mock_post.assert_not_called()
    _assert_verdict_refusal(ctx["envelope"], "target_unknown", None)


async def test_a_plain_halt_is_never_gated_by_the_verdict(tmp_path, monkeypatch):
    """Stopping is not arming. An off machine must still be haltable."""
    _configure(tmp_path, monkeypatch, lane_target="standin", token=None)
    _switch(tmp_path)

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"stopped": True})):
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _stop_fn()(cancel=False)

    assert extract_response_dict(result)["stopped"] is True


# =========================================================================
# The bridge's own startup guard reads the same rule
# =========================================================================


def _patch_bridge_config(monkeypatch, *, section, lane_targets, limits_enabled, db_path):
    """Patch the config keys ``_assert_limits_readable_if_writable`` reads.

    The guard does its lookups through a function-body import of
    ``osprey_connectors.config``, so patching the module attribute is what takes
    effect — the same convention ``test_startup_assertion.py`` uses.

    Limits checking is per connector type and the guard resolves it out of the
    ``control_system`` SECTION, so ``limits_enabled`` is folded into the
    section's deployment-wide ``limits_checking`` block (on a copy, so the
    module-level section never carries one test's posture into the next)
    rather than answered behind a dotted key the guard no longer asks for.
    Only ``database_path`` stays a dotted lookup: it is deployment-wide.
    """
    section = {
        **section,
        "limits_checking": {
            **section.get("limits_checking", {}),
            "enabled": limits_enabled,
            "mode": "exclusive",
        },
    }

    def fake_get_config_value(key: str, default=None):
        if key == "control_system":
            return section
        if key == "control_system.type":
            return section.get("type", default)
        if key == "control_system.limits_checking.enabled":
            return limits_enabled
        if key == "control_system.limits_checking.database_path":
            return db_path
        if key.startswith("services.") and key.endswith(".target"):
            return lane_targets.get(key.split(".")[1], default)
        return default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", fake_get_config_value)


def test_the_bridge_startup_guard_refuses_a_writable_lane_with_no_limits_db(tmp_path, monkeypatch):
    """The unnarrowed baseline: this is the one posture that refuses startup."""
    from osprey.services.bluesky_bridge.validation import _assert_limits_readable_if_writable

    _patch_bridge_config(
        monkeypatch,
        section=_ARMED_SECTION,
        lane_targets={"bluesky": "standin"},
        limits_enabled=True,
        db_path=str(tmp_path / "missing.json"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        _assert_limits_readable_if_writable()
    assert "standin" in str(excinfo.value)


def test_the_bridge_startup_guard_does_not_read_the_switch(tmp_path, monkeypatch):
    """A lane switched off at startup still needs its limits database.

    The guard asks the write tier (``backstop_verdict``), which is target-blind
    and reads no record: a switch moves while the process runs, so a lane that
    is off now may be switched on after it starts and must have working limits
    enforcement from the start. A switch off therefore exempts nothing.
    """
    from osprey.services.bluesky_bridge.validation import _assert_limits_readable_if_writable

    _patch_bridge_config(
        monkeypatch,
        section=_ARMED_SECTION,
        lane_targets={"bluesky": "standin"},
        limits_enabled=True,
        db_path=str(tmp_path / "missing.json"),
    )
    _switch_off(tmp_path, monkeypatch, "standin")

    with pytest.raises(RuntimeError) as excinfo:
        _assert_limits_readable_if_writable()
    assert "standin" in str(excinfo.value)


def test_the_bridge_startup_guard_ignores_a_switch_of_another_target(tmp_path, monkeypatch):
    """Switching the stand-in off says nothing about the lane serving ``va``."""
    from osprey.services.bluesky_bridge.validation import _assert_limits_readable_if_writable

    _patch_bridge_config(
        monkeypatch,
        section=_ARMED_SECTION,
        lane_targets={"bluesky": "va"},
        limits_enabled=True,
        db_path=str(tmp_path / "missing.json"),
    )
    _switch_off(tmp_path, monkeypatch, "standin")

    with pytest.raises(RuntimeError):
        _assert_limits_readable_if_writable()
