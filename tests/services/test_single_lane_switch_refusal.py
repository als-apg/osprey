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
2. **Which way it fails.** Only a *readable* control-context record naming a
   *different* target refuses. An absent record, a corrupt one, and one naming
   a target this build has no name for all read as "on the baseline" and permit
   the operation — nothing readable says a switch happened, so the lane's target
   is the recorded control target, and any other direction would break every deployment
   that never switches (today, all of them). A record whose OWNER is dead is not
   in that set: one record describes the whole deployment, and the machine it
   names is where the deployment is whether or not the process that last wrote
   it is still running.
3. **That the reason code is one vocabulary.** The MCP tool module cannot
   import ``queue_backend`` (it pulls in ``bluesky-queueserver-api``), so the
   code string is spelled in both layers; these tests pin them equal, and pin
   the JS queue client's exhaustive capability set as carrying it too. A code
   one consumer has never seen is as useless to it as no code at all.
"""

from __future__ import annotations

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
from osprey_connectors import control_context, posture_store
from tests._control_context_fixtures import owner, write_control_context
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
        # config; the tool module reads the write posture separately.
        monkeypatch.setattr(
            target_banner, "load_osprey_config", lambda: {"control_system": control_system}
        )

        def fake_get_config_value(key: str, default: Any = None, config_path: Any = None) -> Any:
            # The whole section, not only the deployment-wide flag: write
            # posture is resolved per control target out of
            # `control_system.connector`, so a stub serving one dotted key would
            # answer "unarmed" for every lane whatever this deployment says.
            return {
                "control_system": control_system,
                "control_system.writes_enabled": True,
            }.get(key, default)

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)

        # The control-context record lives under the agent-data root; stamping
        # that at tmp_path is what keeps one test's deployment out of another's,
        # and out of whatever record this machine happens to be carrying.
        monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        control_context.invalidate_cache()

        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", _TOKEN)
        initialize_server_context()
        return tmp_path

    yield _stage
    control_context.invalidate_cache()


def _deployment_on(root: Path, target: str, generation: int = 0, **kwargs: Any) -> Path:
    """Point the deployment's control-context record at *target*.

    One record per deployment, and the banner reads its ``target`` — so this is
    the whole of what "the deployment is on X" means to every consumer below.
    """
    return write_control_context(root, target=target, generation=generation, **kwargs)


def _switch_to(root: Path, target: str) -> Path:
    """Move an already-published deployment onto *target*, as a switch would."""
    return _deployment_on(root, target, generation=1)


def _write_raw_record(root: Path, body: str) -> Path:
    """Drop a control-context record with arbitrary bytes in it."""
    path = control_context.record_path_under(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    control_context.invalidate_cache()
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
async def test_queue_add_reaches_the_bridge_with_no_record(deployment, baseline):
    """No switch has ever happened, which is every deployment today."""
    deployment(baseline)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add()

    assert post.call_args.args[0] == "/queue/items"


@pytest.mark.parametrize("baseline", ["live", "va"])
async def test_queue_start_reaches_the_bridge_with_no_record(deployment, baseline):
    deployment(baseline)
    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start()

    assert post.call_args.args[0] == "/queue/start"


@pytest.mark.parametrize("baseline", ["live", "va"])
async def test_a_session_sitting_on_the_baseline_is_not_a_mismatch(deployment, baseline):
    """A published record is not itself a refusal — only a differing one is."""
    root = deployment(baseline)
    _deployment_on(root, baseline)

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
    root = deployment(baseline)
    _deployment_on(root, baseline)
    _switch_to(root, session)

    with patch(f"{_MOD}._http_post_json") as post:
        with assert_raises_error(error_type=qb.REASON_CONTROL_TARGET_MISMATCH) as ctx:
            await _add(revision=7)

    # The refusal is local: the bridge is never asked, so the pinned revision
    # is not spent and no item can be left behind.
    assert not post.called

    envelope = ctx["envelope"]
    # Both targets are named, so the operator is never left to infer which
    # machine the lane serves or which one they switched to.
    assert session in envelope["error_message"]
    assert baseline in envelope["error_message"]
    assert envelope["details"]["control_target"] == session
    assert envelope["details"]["baseline_target"] == baseline


@pytest.mark.parametrize(("baseline", "session"), _SWITCHED)
async def test_queue_start_refuses_while_the_session_is_switched(deployment, baseline, session):
    root = deployment(baseline)
    _deployment_on(root, baseline)
    _switch_to(root, session)

    with patch(f"{_MOD}._http_post_json") as post:
        with assert_raises_error(error_type=qb.REASON_CONTROL_TARGET_MISMATCH) as ctx:
            await _start()

    assert not post.called
    assert session in ctx["envelope"]["error_message"]
    assert baseline in ctx["envelope"]["error_message"]


async def test_the_refusal_carries_a_capability_record_in_the_queue_wire_shape(deployment):
    """`{code, detail, capability}` — the shape every other queue refusal has.

    The bridge cannot compose this record (it never learns the recorded control
    target), so the host-side server composes it. Consumers that branch on
    ``details.code`` and render ``capability.detail`` must need no special case.
    """
    root = deployment("live")
    _deployment_on(root, "live")
    _switch_to(root, "va")

    with patch(f"{_MOD}._http_post_json"):
        with assert_raises_error(error_type=qb.REASON_CONTROL_TARGET_MISMATCH) as ctx:
            await _add()

    details = ctx["envelope"]["details"]
    assert details["code"] == qb.REASON_CONTROL_TARGET_MISMATCH
    assert details["capability"] == {
        "can_execute": False,
        "reason": qb.REASON_CONTROL_TARGET_MISMATCH,
        "detail": details["detail"],
    }
    assert details["detail"] == ctx["envelope"]["error_message"]


async def test_the_refusal_names_the_way_back(deployment):
    """The suggestions have to be actionable, not just accurate."""
    root = deployment("live")
    _deployment_on(root, "live")
    _switch_to(root, "va")

    with patch(f"{_MOD}._http_post_json"):
        with assert_raises_error(error_type=qb.REASON_CONTROL_TARGET_MISMATCH) as ctx:
            await _start()

    suggestions = " ".join(ctx["envelope"]["suggestions"])
    assert "control_target_set(target='live')" in suggestions
    assert "control-system tools" in suggestions


async def test_halting_is_never_gated_by_the_control_target(deployment):
    """`queue_stop` stays reachable while switched, like it does with writes off."""
    root = deployment("live")
    _deployment_on(root, "live")
    _switch_to(root, "va")

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"stop_pending": True})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _stop()

    assert post.call_args.args[0] == "/queue/stop"


# =========================================================================
# An unreadable record is the baseline, never a refusal
# =========================================================================


async def test_a_corrupt_record_reads_as_the_baseline(deployment):
    root = deployment("live")
    _write_raw_record(root, "{not json at all")

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add()

    assert post.called


async def test_a_record_naming_an_unknown_target_reads_as_the_baseline(deployment):
    """A record whose ``target`` is not one of the known names is not an answer."""
    root = deployment("live")
    _deployment_on(root, "somewhere-else")

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"run_id": "r1"})) as post:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add()

    assert post.called


async def test_a_record_whose_owner_is_dead_still_names_the_deployments_target(deployment):
    """The one case in this section that is NOT read as the baseline.

    A per-server state file left by a dead server used to be residue, and
    residue was ignored. There is one record now and it belongs to the
    deployment rather than to any process: the machine it names is where this
    deployment is pointed, and the death of whoever last held the pen does not
    move the plan lane back onto a target nobody selected. So this refuses,
    exactly as a live-owned record naming ``va`` does.
    """
    root = deployment("live")
    _deployment_on(root, "va", generation=3, owned_by=owner(pid=_DEAD_PID))

    with patch(f"{_MOD}._http_post_json") as post:
        with assert_raises_error(error_type=qb.REASON_CONTROL_TARGET_MISMATCH) as ctx:
            await _add()

    assert not post.called
    assert ctx["envelope"]["details"]["control_target"] == "va"


async def test_a_mock_deployment_resolves_to_live_and_still_compares(deployment):
    """The comparison must not crash on a deployment that cannot be switched.

    `mock` is not a target; the baseline it implies is `live`, so a record
    naming `va` is a mismatch there like anywhere else.
    """
    root = deployment("live-from-mock")
    _deployment_on(root, "live")
    _switch_to(root, "va")

    with patch(f"{_MOD}._http_post_json") as post:
        with assert_raises_error(error_type=qb.REASON_CONTROL_TARGET_MISMATCH) as ctx:
            await _add()

    assert not post.called
    assert "live" in ctx["envelope"]["error_message"]


# =========================================================================
# One vocabulary across three layers
# =========================================================================


def test_the_tool_modules_reason_code_is_the_queue_backend_constant():
    """The two spellings the import invariant forces apart, pinned together."""
    assert queue.REASON_CONTROL_TARGET_MISMATCH == qb.REASON_CONTROL_TARGET_MISMATCH


def test_the_reason_has_an_entry_in_the_shared_refusal_hint_table():
    """So a lane-aware bridge relaying the same code answers the same way."""
    hints = queue._REFUSAL_HINTS[qb.REASON_CONTROL_TARGET_MISMATCH]
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

    assert f"'{qb.REASON_CONTROL_TARGET_MISMATCH}'" in vocabulary
