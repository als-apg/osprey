"""The session posture store's term in the Bluesky plan lane's arming gate.

The queue surface has always re-read the DEPLOYMENT's write posture for the
control target its bound lane serves, fresh from config, before anything
reaches the network. This file pins the second term that now ANDs into that
answer: the per-(session, target) narrowing an operator sets from the header
chip, read through ``osprey_connectors.session_store.effective_writes``.

Three properties, all asserted directly:

1. **The store only narrows.** A sandboxed lane target refuses ``queue_start``
   with zero HTTP calls and withholds the launch token from ``queue_add``,
   even though the deployment's own config arms every target here. The
   converse — an unarmed ceiling that the store says nothing about — still
   refuses, so nothing in the store can widen a deployment.
2. **The narrowing is per target, not per session.** A session that sandboxed
   ``standin`` still queues and starts on a lane serving ``va``: the entry
   names one machine, and the lane's own target is what indexes it.
3. **No session key means the store is not consulted.** A process nobody
   stamped ``OSPREY_POSTURE_SESSION`` into behaves exactly as it did before
   this store existed, whatever the file happens to hold.

The bridge's own startup guard (``bluesky_bridge.validation``) reads the same
rule for the lane it IS, and is covered here too — same term, same store, and
keeping the two next to each other is what makes a divergence visible.

The HTTP boundary is patched, so these exercise this module's gating and
refusal wording against the queue wire contract with no bridge process.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.bluesky.server_context import initialize_server_context, reset_server_context
from osprey.mcp_server.bluesky.tools import queue
from osprey_connectors import session_store
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

pytestmark = pytest.mark.unit

_MOD = "osprey.mcp_server.bluesky.tools.queue"

_TOKEN = "genuinely-valid-token"
_SESSION = "0b6f2f7c-3d1e-4a2b-9c8d-1f2e3a4b5c6d"

#: A deployment that arms EVERY target at the ceiling, so the only thing that
#: can refuse in these tests is the session store. Baseline is ``live``
#: (``type: epics``); the lane declares whichever target a test needs.
_ARMED_SECTION = {
    "type": "epics",
    "writes_enabled": False,
    "connector": {
        "epics": {"prefix": "X:", "writes_enabled": True},
        "virtual_accelerator": {"host": "localhost", "writes_enabled": True},
        "live_standin": {"prefix": "S:", "writes_enabled": True},
    },
}


@pytest.fixture(autouse=True)
def _reset_bluesky_context():
    yield
    reset_server_context()


@pytest.fixture(autouse=True)
def _clean_posture_env(monkeypatch):
    """No ambient posture: every test states its own session key and root."""
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv(session_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    session_store.invalidate_cache()
    yield
    session_store.invalidate_cache()


def _configure(
    tmp_path,
    monkeypatch,
    *,
    lane_target: str | None = None,
    section: dict | None = None,
    token: str | None = _TOKEN,
) -> None:
    """Render a project whose single plan lane declares *lane_target*.

    The shipped control-assistant render declares ``services.bluesky.target:
    standin``, which is why the store term is exercised by the DEFAULT
    deployment and not only by a two-lane one.
    """
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    config: dict = {"control_system": _ARMED_SECTION if section is None else section}
    if lane_target is not None:
        config["services"] = {"bluesky": {"target": lane_target}}
    (project / "config.yml").write_text(yaml.dump(config))
    monkeypatch.chdir(project)
    if token is None:
        monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", token)
    initialize_server_context()


def _narrow(tmp_path, monkeypatch, entry, *, session_key: str | None = _SESSION) -> Path:
    """Write the posture store and stamp the anchors a session child carries.

    *entry* is the store value for :data:`_SESSION` — the per-target object the
    header chip writes. ``session_key=None`` stamps the root but no session,
    which is the "nothing addressed this session" case.
    """
    root = tmp_path / "agent_data"
    path = root / session_store.STATE_DIR_NAME / session_store.STORE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({_SESSION: entry}), encoding="utf-8")
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    if session_key is None:
        monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    else:
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", session_key)
    session_store.invalidate_cache()
    return path


def _add_fn():
    return get_tool_fn(queue.queue_add)


def _start_fn():
    return get_tool_fn(queue.queue_start)


def _stop_fn():
    return get_tool_fn(queue.queue_stop)


def _refusal(code: str, detail: str, **extras) -> dict:
    return {"detail": {"code": code, "detail": detail, **extras}}


# =========================================================================
# The store narrows the bound lane's target
# =========================================================================


async def test_a_sandboxed_lane_target_refuses_queue_start_with_zero_http_calls(
    tmp_path, monkeypatch
):
    """The header chip is a kill switch for one machine, enforced at write time.

    The deployment arms ``standin`` here and the caller holds a valid launch
    token; the ONLY thing refusing is the operator's per-target narrowing for
    this session. It must refuse before ``_http_post_json`` is invoked, exactly
    as the config-level posture does — a narrowing that only took effect on the
    next respawn would leave the session it was set from still able to start
    the queue.
    """
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _start_fn()()

    mock_post.assert_not_called()
    message = ctx["envelope"]["error_message"]
    assert "header chip" in message
    assert "standin" in message
    assert _TOKEN not in message


async def test_the_start_refusal_names_the_session_posture_not_a_config_key(tmp_path, monkeypatch):
    """Sending an operator to config.yml here would be a wrong instruction.

    ``control_system.connector.live_standin.writes_enabled`` already says
    ``true``; editing it, rebuilding and redeploying would change nothing. The
    refusal must point at the surface that actually holds the narrowing.
    """
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

    with patch(f"{_MOD}._http_post_json"):
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _start_fn()()

    envelope = ctx["envelope"]
    suggestions = envelope["suggestions"]
    assert any("header chip" in s for s in suggestions)
    assert not any("writes_enabled: true" in s for s in suggestions)
    assert "read-only" in envelope["error_message"] or "read-only" in " ".join(suggestions)


async def test_a_sandboxed_lane_target_withholds_the_launch_token_from_queue_add(
    tmp_path, monkeypatch
):
    """Composing still works; arming does not — the same split the config gate has.

    An item added to an idle queue moves nothing, so the narrowing must not turn
    enqueue into a refusal. What it must do is withhold the token, so the bridge
    refuses the add the moment the queue is actually draining.
    """
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

    body = {"run_id": "r1", "revision": 3, "item": {"item_uid": "u1"}}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            result = await _add_fn()(draft_revision=3)

    assert m.call_args.kwargs["headers"] is None
    assert extract_response_dict(result)["run_id"] == "r1"


async def test_the_withheld_token_hint_names_the_session_posture_and_the_target(
    tmp_path, monkeypatch
):
    """The bridge cannot know WHY the token was withheld; this server says so.

    The bridge's own sentence is relayed untouched — chasing a different token
    is the natural next step and it is the wrong one here, so the added hint has
    to name the header chip and the target rather than a config key.
    """
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

    body = _refusal(
        "launch_token_required",
        "the queue is running, so adding an item requires the launch token",
        manager_state="executing_queue",
    )
    with patch(f"{_MOD}._http_post_json", return_value=(403, body)):
        with assert_raises_error(error_type="launch_token_required") as ctx:
            await _add_fn()(draft_revision=7)

    suggestions = ctx["envelope"]["suggestions"]
    assert any("header chip" in s and "standin" in s for s in suggestions)
    assert not any("profile.yml" in s for s in suggestions)


# =========================================================================
# The narrowing is per target
# =========================================================================


async def test_an_armed_lane_on_another_target_still_queues(tmp_path, monkeypatch):
    """A store entry names ONE machine, and the lane's own target indexes it.

    Sandboxing the stand-in must not unarm the virtual accelerator: a session
    that narrowed the machine it is not running plans on has said nothing about
    the one it is.
    """
    _configure(tmp_path, monkeypatch, lane_target="va")
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

    body = {"run_id": "abc", "revision": 7, "item": {"item_uid": "u1"}}
    with patch(f"{_MOD}._http_post_json", return_value=(200, body)) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _add_fn()(draft_revision=7)

    assert m.call_args.kwargs["headers"] == {"X-Launch-Token": _TOKEN}


async def test_an_armed_lane_on_another_target_still_starts(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, lane_target="va")
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()

    assert m.call_args.kwargs["headers"] == {"X-Launch-Token": _TOKEN}


# =========================================================================
# The store never widens, and is not consulted without a session
# =========================================================================


async def test_an_unarmed_ceiling_still_refuses_with_an_empty_store(tmp_path, monkeypatch):
    """Nothing in this file can arm a target the deployment left unarmed.

    The store holds narrowings and nothing else; the deployment ceiling is read
    exactly as it was before the store existed.
    """
    unarmed = {
        "type": "epics",
        "writes_enabled": False,
        "connector": {"live_standin": {"prefix": "S:", "writes_enabled": False}},
    }
    _configure(tmp_path, monkeypatch, lane_target="standin", section=unarmed)
    _narrow(tmp_path, monkeypatch, {})

    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _start_fn()()

    mock_post.assert_not_called()
    assert any(
        "control_system.connector.live_standin.writes_enabled" in s
        for s in ctx["envelope"]["suggestions"]
    )


async def test_no_session_key_leaves_the_store_unconsulted(tmp_path, monkeypatch):
    """A process nobody stamped is a process nothing narrowed.

    The store here holds a sandbox entry for the very target this lane serves,
    and it must have no effect: without ``OSPREY_POSTURE_SESSION`` there is no
    session this entry could be about.
    """
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"}, session_key=None)

    with patch(f"{_MOD}._http_post_json", return_value=(200, {"started": True})) as m:
        with patch(f"{_MOD}.notify_agent_activity_async"):
            await _start_fn()()

    assert m.call_args.kwargs["headers"] == {"X-Launch-Token": _TOKEN}


async def test_the_store_narrows_the_halt_withdrawal_union_too(tmp_path, monkeypatch):
    """Withdrawing a halt resumes motion, so the union gate reads the store too.

    ``queue_stop(cancel=True)`` asks whether ANY rendered lane is armed. On this
    single-lane deployment the only lane serves ``standin``, which the session
    narrowed — so the withdrawal is refused before the network, exactly as an
    unarmed deployment's is.

    The refusal names the header chip rather than the deployment-wide config
    key, and getting there means asking about the RENDERED LANES' targets. The
    baseline of this deployment is ``live`` (``type: epics``) while its only
    lane serves ``standin``, so a wording that fell back to the baseline would
    read an entry about a machine no plan here ever runs on.
    """
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

    with patch(f"{_MOD}._http_post_json") as mock_post:
        with assert_raises_error(error_type="writes_disabled") as ctx:
            await _stop_fn()(cancel=True)

    mock_post.assert_not_called()
    assert "header chip" in ctx["envelope"]["error_message"]


async def test_a_plain_halt_is_never_gated_by_the_store(tmp_path, monkeypatch):
    """Stopping is not arming. A narrowed session must still be able to halt."""
    _configure(tmp_path, monkeypatch, lane_target="standin")
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

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
    ``osprey.utils.config``, so patching the module attribute is what takes
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
            "allow_unlisted_channels": False,
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

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)


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


def test_the_bridge_startup_guard_reads_the_session_store_too(tmp_path, monkeypatch):
    """A lane the session narrowed cannot write, so it needs no limits database.

    The guard is fail-OPEN by construction: it refuses only the one posture
    where this process could actually move hardware without limits enforcement.
    A narrowed target is not that posture, and the term is the same
    ``effective_writes`` call the MCP surface makes — a bridge that answered
    differently from the tool addressing it would be the divergence this whole
    store exists to prevent.
    """
    from osprey.services.bluesky_bridge.validation import _assert_limits_readable_if_writable

    _patch_bridge_config(
        monkeypatch,
        section=_ARMED_SECTION,
        lane_targets={"bluesky": "standin"},
        limits_enabled=True,
        db_path=str(tmp_path / "missing.json"),
    )
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

    _assert_limits_readable_if_writable()


def test_the_bridge_startup_guard_ignores_a_narrowing_of_another_target(tmp_path, monkeypatch):
    """Narrowing the stand-in says nothing about the lane serving ``va``."""
    from osprey.services.bluesky_bridge.validation import _assert_limits_readable_if_writable

    _patch_bridge_config(
        monkeypatch,
        section=_ARMED_SECTION,
        lane_targets={"bluesky": "va"},
        limits_enabled=True,
        db_path=str(tmp_path / "missing.json"),
    )
    _narrow(tmp_path, monkeypatch, {"standin": "sandbox"})

    with pytest.raises(RuntimeError):
        _assert_limits_readable_if_writable()
