"""The single-lane switch refusal: OC-1, the contract that makes the plan
stack safe to ship before a second lane exists.

Every deployment today renders exactly ONE Bluesky plan lane, wired at build
time to one control target. An agent session, meanwhile, can be switched to the
other target at run time. While the two disagree, a PLAN queued or started here
would run against a machine the session is not pointed at — so ``queue_add`` and
``queue_start`` refuse for as long as they disagree, and ``queue_stop`` never
does, because halting must stay reachable in every posture.

Three things are pinned, and each is pinned at the layer that owns it:

1. **Where the check runs.** The refusal is host-side, in the bluesky MCP
   tools, and lands BEFORE any HTTP call — asserted by the transport mock never
   being called, not merely by a status code. The bridge cannot make this call:
   it serves its lane's target and never learns the session's.
2. **Which way it fails.** Only a *readable* state file naming a *different*
   target refuses. Absent, corrupt, and dead-owner state all read as "on the
   baseline" and permit the operation — no switch has happened, so the lane's
   target is the session's target, and any other direction would break every
   deployment that never switches (today, all of them).
3. **That the reason code is one vocabulary.** The MCP tool module cannot
   import ``queue_backend`` (it pulls in ``bluesky-queueserver-api``), so the
   code string is spelled in both layers; these tests pin them equal, and pin
   the JS queue client's exhaustive capability set as carrying it too. A code
   one consumer has never seen is as useless to it as no code at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from osprey.mcp_server.bluesky.server_context import (
    initialize_server_context,
    reset_server_context,
)
from osprey.mcp_server.bluesky.tools import queue
from osprey.mcp_server.control_system import target_banner, target_state
from osprey.services.bluesky_bridge import queue_backend as qb
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.queue"
_TOKEN = "genuinely-valid-token"

# A PID above every platform's PID_MAX, so `os.kill(pid, 0)` reports it gone
# rather than hitting a real process the test machine happens to be running.
_DEAD_PID = 999_999_999

# The connector type each deployment baseline is rendered from. `mock` is in
# here on purpose: a mock deployment can never be switched in practice, but the
# comparison still runs on every queue_add, so it has to produce `live` rather
# than crash.
_BASELINE_TYPES = {"live": "epics", "va": "virtual_accelerator", "live-from-mock": "mock"}


@pytest.fixture(autouse=True)
def _reset_bluesky_context():
    yield
    reset_server_context()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """Stage a deployment: a baseline target, an armed posture, an empty state root.

    Returns a callable taking a key of :data:`_BASELINE_TYPES`, so a test names
    the baseline it wants ("live", "va") rather than the connector type that
    happens to imply it.
    """

    def _stage(baseline: str) -> None:
        control_system = {"type": _BASELINE_TYPES[baseline], "writes_enabled": True}

        # The baseline comes from the shared resolver reading the deployment
        # config; the kill switch is a separate, dotted read in the tool module.
        monkeypatch.setattr(
            target_banner, "load_osprey_config", lambda: {"control_system": control_system}
        )

        def fake_get_config_value(key: str, default: Any = None, config_path: Any = None) -> Any:
            return {"control_system.writes_enabled": True}.get(key, default)

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)

        # The state file lives under the shared data root; pointing that at
        # tmp_path is what keeps one test's session out of another's.
        monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)

        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", _TOKEN)
        initialize_server_context()

    return _stage


def _session_on(target: str) -> None:
    """Write the state file a controls server owned by this session would write."""
    target_state.write_on_start(target)


def _switch_to(target: str) -> None:
    """Move an already-published session onto *target*, as a switch would."""
    target_state.publish_switch(target, generation=1)


def _write_raw_state(body: str) -> Path:
    """Drop a state file this process owns with arbitrary bytes in it."""
    path = target_state.state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


async def _add(revision: int = 1):
    return await get_tool_fn(queue.queue_add)(draft_revision=revision)


async def _start():
    return await get_tool_fn(queue.queue_start)()


async def _stop(cancel: bool = False):
    return await get_tool_fn(queue.queue_stop)(cancel=cancel)


# =========================================================================
# The baseline session: the gate is transparent
# =========================================================================


@pytest.mark.parametrize("baseline", ["live", "va"])
async def test_queue_add_reaches_the_bridge_with_no_state_file(deployment, baseline):
    """No switch has ever happened, which is every deployment today."""
    deployment(baseline)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add()

    assert post.call_args.args[0] == "/queue/items"


@pytest.mark.parametrize("baseline", ["live", "va"])
async def test_queue_start_reaches_the_bridge_with_no_state_file(deployment, baseline):
    deployment(baseline)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start()

    assert post.call_args.args[0] == "/queue/start"


@pytest.mark.parametrize("baseline", ["live", "va"])
async def test_a_session_sitting_on_the_baseline_is_not_a_mismatch(deployment, baseline):
    """A published state file is not itself a refusal — only a differing one is."""
    deployment(baseline)
    _session_on(baseline)

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add()

    assert post.called


# =========================================================================
# The switched session: both armed operations refuse, before any HTTP call
# =========================================================================

_SWITCHED = [
    pytest.param("live", "va", id="live-baseline-switched-to-va"),
    pytest.param("va", "live", id="va-baseline-switched-to-live"),
]


@pytest.mark.parametrize(("baseline", "session"), _SWITCHED)
async def test_queue_add_refuses_while_the_session_is_switched(deployment, baseline, session):
    deployment(baseline)
    _session_on(baseline)
    _switch_to(session)

    with patch(f"{_MOD}._http_post_json") as post:
        with assert_raises_error(error_type=qb.REASON_SESSION_TARGET_MISMATCH) as ctx:
            await _add(revision=7)

    # The refusal is local: the bridge is never asked, so the pinned revision
    # is not spent and no item can be left behind.
    assert not post.called

    envelope = ctx["envelope"]
    # Both targets are named, so the operator is never left to infer which
    # machine the lane serves or which one they switched to.
    assert session in envelope["error_message"]
    assert baseline in envelope["error_message"]
    assert envelope["details"]["session_target"] == session
    assert envelope["details"]["baseline_target"] == baseline


@pytest.mark.parametrize(("baseline", "session"), _SWITCHED)
async def test_queue_start_refuses_while_the_session_is_switched(deployment, baseline, session):
    deployment(baseline)
    _session_on(baseline)
    _switch_to(session)

    with patch(f"{_MOD}._http_post_json") as post:
        with assert_raises_error(error_type=qb.REASON_SESSION_TARGET_MISMATCH) as ctx:
            await _start()

    assert not post.called
    assert session in ctx["envelope"]["error_message"]
    assert baseline in ctx["envelope"]["error_message"]


async def test_the_refusal_carries_a_capability_record_in_the_queue_wire_shape(deployment):
    """`{code, detail, capability}` — the shape every other queue refusal has.

    The bridge cannot compose this record (it never learns the session target),
    so the host-side server composes it. Consumers that branch on
    ``details.code`` and render ``capability.detail`` must need no special case.
    """
    deployment("live")
    _session_on("live")
    _switch_to("va")

    with patch(f"{_MOD}._http_post_json"):
        with assert_raises_error(error_type=qb.REASON_SESSION_TARGET_MISMATCH) as ctx:
            await _add()

    details = ctx["envelope"]["details"]
    assert details["code"] == qb.REASON_SESSION_TARGET_MISMATCH
    assert details["capability"] == {
        "can_execute": False,
        "reason": qb.REASON_SESSION_TARGET_MISMATCH,
        "detail": details["detail"],
    }
    assert details["detail"] == ctx["envelope"]["error_message"]


async def test_the_refusal_names_the_way_back(deployment):
    """The suggestions have to be actionable, not just accurate."""
    deployment("live")
    _session_on("live")
    _switch_to("va")

    with patch(f"{_MOD}._http_post_json"):
        with assert_raises_error(error_type=qb.REASON_SESSION_TARGET_MISMATCH) as ctx:
            await _start()

    suggestions = " ".join(ctx["envelope"]["suggestions"])
    assert "control_target_set(target='live')" in suggestions
    assert "control-system tools" in suggestions


async def test_halting_is_never_gated_by_the_session_target(deployment):
    """`queue_stop` stays reachable while switched, like it does with writes off."""
    deployment("live")
    _session_on("live")
    _switch_to("va")

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"stop_pending": True})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _stop()

    assert post.call_args.args[0] == "/queue/stop"


# =========================================================================
# Unreadable state is the baseline, never a refusal
# =========================================================================


async def test_a_corrupt_state_file_reads_as_the_baseline(deployment):
    deployment("live")
    _write_raw_state("{not json at all")

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add()

    assert post.called


async def test_a_state_file_naming_an_unknown_target_reads_as_the_baseline(deployment):
    """A record whose ``target`` is not one of the two names is not an answer."""
    deployment("live")
    _session_on("live")
    record = json.loads(target_state.state_file_path().read_text(encoding="utf-8"))
    record["target"] = "somewhere-else"
    target_state.state_file_path().write_text(json.dumps(record), encoding="utf-8")

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add()

    assert post.called


async def test_a_state_file_left_by_a_dead_server_is_ignored(deployment, tmp_path):
    """Nobody is on `va`; the file is the residue of a server that died."""
    deployment("live")
    stale = tmp_path / target_state.STATE_DIR_NAME / f"target_state_{_DEAD_PID}.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(
        json.dumps({"target": "va", "generation": 3, "server_pid": _DEAD_PID, "owner_ppid": 1}),
        encoding="utf-8",
    )

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add()

    assert post.called


async def test_a_mock_deployment_resolves_to_live_and_still_compares(deployment):
    """The comparison must not crash on a deployment that cannot be switched.

    `mock` is not a target; the baseline it implies is `live`, so a state file
    naming `va` is a mismatch there like anywhere else.
    """
    deployment("live-from-mock")
    _session_on("live")
    _switch_to("va")

    with patch(f"{_MOD}._http_post_json") as post:
        with assert_raises_error(error_type=qb.REASON_SESSION_TARGET_MISMATCH) as ctx:
            await _add()

    assert not post.called
    assert "live" in ctx["envelope"]["error_message"]


# =========================================================================
# One vocabulary across three layers
# =========================================================================


def test_the_tool_modules_reason_code_is_the_queue_backend_constant():
    """The two spellings the import invariant forces apart, pinned together."""
    assert queue.REASON_SESSION_TARGET_MISMATCH == qb.REASON_SESSION_TARGET_MISMATCH


def test_the_reason_has_an_entry_in_the_shared_refusal_hint_table():
    """So a lane-aware bridge relaying the same code answers the same way."""
    hints = queue._REFUSAL_HINTS[qb.REASON_SESSION_TARGET_MISMATCH]
    assert hints and all(isinstance(hint, str) and hint for hint in hints)


def test_the_reason_is_in_the_js_queue_clients_capability_vocabulary():
    """The panel classifies on an exhaustive set; a code missing from it falls
    through to the generic error branch and loses its capability rendering."""
    from osprey.interfaces import bluesky_web

    source = (
        Path(bluesky_web.__file__).parent / "panels" / "bluesky" / "draft-client.js"
    ).read_text(encoding="utf-8")
    _, _, after = source.partition("const DEPLOYMENT_CAPABILITY_CODES = new Set([")
    vocabulary, _, _ = after.partition("]);")

    assert f"'{qb.REASON_SESSION_TARGET_MISMATCH}'" in vocabulary
