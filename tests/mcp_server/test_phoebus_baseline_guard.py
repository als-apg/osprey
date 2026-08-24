"""Phoebus stays pinned to the deployment baseline, and says so.

The Phoebus bridge addresses one running Phoebus product whose PV context was
fixed when that product started, so a session-level control-system target
switch does not move it. Two behaviours follow, both covered here:

* ``phoebus_drive`` refuses while the session target differs from the
  deployment baseline — a write must never land on a target the session left;
* the four read tools prepend one informational line naming both targets while
  switched, and change nothing at all while on the baseline.

Both strings come from ``osprey.mcp_server.control_system.target_banner``,
whose resolution rules (owner-ppid ownership, ambiguity, corrupt state) are
unit-tested here as well because the HealthRuntime row will reuse them.

No Phoebus product, bridge, or network: the HTTP boundary is patched the way
``test_phoebus_tools`` does. The target-state directory is redirected into
``tmp_path`` so no test can see (or write) real session state.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from osprey.mcp_server.control_system import target_banner, target_state
from osprey.mcp_server.phoebus.tools import bridge_tools, databrowser_tools
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

_BRIDGE_MOD = "osprey.mcp_server.phoebus.tools.bridge_tools"
_DB_MOD = "osprey.mcp_server.phoebus.tools.databrowser_tools"

#: A PID that cannot name a running process: the maximum a 32-bit ``pid_t``
#: holds, which no kernel hands out. ``os.kill(pid, 0)`` on it raises
#: ProcessLookupError, so ``is_process_alive`` reports it dead.
_DEAD_PID = 2_147_483_646

#: A parent PID belonging to nobody — stands in for another session's server.
_FOREIGN_PPID = 999_999


# ── fixtures / helpers ──────────────────────────────────────────────────────
@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """Redirect the target-state directory into ``tmp_path``.

    ``target_state.state_dir()`` resolves through
    ``resolve_shared_data_root``; rebinding that one name keeps the real
    deployment's ``var/agent_data`` invisible to these tests in both
    directions.
    """
    root = tmp_path / "agent_data"
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    return root / target_state.STATE_DIR_NAME


def write_state(state_dir, *, target, owner_ppid, server_pid, raw=None):
    """Write one state file. ``raw`` replaces the record wholesale (corruption)."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{target_state.STATE_FILE_PREFIX}{server_pid}.json"
    if raw is not None:
        path.write_text(raw)
        return path
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": 1,
                "server_pid": server_pid,
                "owner_ppid": owner_ppid,
                "targets": {
                    "live": {"label": "live machine", "endpoint": "gw:5064", "real_machine": True},
                    "va": {"label": "virtual accelerator", "endpoint": "localhost:5074"},
                },
                "children": [],
            }
        )
    )
    return path


def own_ppid():
    """This process's parent — the Claude Code process a real server records.

    Used instead of monkeypatching ``os.getppid``: the resolver's whole job is
    to match a record against the *real* parent of the process reading it, and
    a test that patches that lookup away stops testing the match.
    """
    return os.getppid()


def own_pid():
    """A PID that is certainly alive: this test process."""
    return os.getpid()


def set_config(tmp_path, monkeypatch, control_system=None):
    """Point OSPREY_CONFIG at a config.yml declaring *control_system*.

    Also chdir into ``tmp_path`` so the workspace paths the phoebus tools
    resolve (plot dir, snapshot dir) land there and not in the repo.
    """
    config: dict = {}
    if control_system is not None:
        config["control_system"] = control_system
    config_file = tmp_path / "osprey_config.yml"
    config_file.write_text(yaml.dump(config))
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    monkeypatch.chdir(tmp_path)
    return config_file


@pytest.fixture
def switched(tmp_path, monkeypatch, state_root):
    """Baseline ``live`` (EPICS deployment) with the session switched to ``va``."""
    set_config(tmp_path, monkeypatch, {"type": "epics"})
    write_state(state_root, target="va", owner_ppid=own_ppid(), server_pid=own_pid())


@pytest.fixture
def on_baseline(tmp_path, monkeypatch, state_root):
    """Baseline ``live`` with no session state at all — nothing to announce."""
    set_config(tmp_path, monkeypatch, {"type": "epics"})


def bridge_fn(name):
    return get_tool_fn(getattr(bridge_tools, name))


# ── helper: baseline resolution ─────────────────────────────────────────────
@pytest.mark.parametrize(
    ("control_system", "expected"),
    [
        ({"type": "virtual_accelerator"}, target_state.TARGET_VA),
        ({"type": "epics"}, target_state.TARGET_LIVE),
        ({"type": "mock"}, target_state.TARGET_LIVE),
        ({}, target_state.TARGET_LIVE),
        (None, target_state.TARGET_LIVE),
    ],
)
def test_baseline_target_follows_the_shared_resolver(
    tmp_path, monkeypatch, control_system, expected
):
    """Only a virtual-accelerator deployment has a ``va`` baseline."""
    set_config(tmp_path, monkeypatch, control_system)
    assert target_banner.resolve_baseline_target() == expected


# ── helper: session resolution ──────────────────────────────────────────────
def test_session_target_absent_state_is_baseline(state_root):
    """No state directory at all — the switch has never run."""
    assert target_banner.resolve_session_target("live") == "live"
    assert not target_banner.resolve_target_situation().switched


def test_session_target_matching_owner_is_read(state_root):
    write_state(state_root, target="va", owner_ppid=own_ppid(), server_pid=own_pid())
    assert target_banner.resolve_session_target("live") == "va"


def test_session_target_ignores_another_sessions_record(state_root):
    """A live server owned by a different Claude Code process is not ours."""
    write_state(state_root, target="va", owner_ppid=_FOREIGN_PPID, server_pid=own_pid())
    assert target_banner.resolve_session_target("live") == "live"


def test_session_target_ambiguous_ownership_is_baseline(state_root):
    """Two live records claiming the same parent: no answer, not a guess."""
    write_state(state_root, target="va", owner_ppid=own_ppid(), server_pid=own_pid())
    # PID 1 always exists; os.kill(1, 0) raises PermissionError, which
    # is_process_alive deliberately counts as ALIVE.
    write_state(state_root, target="va", owner_ppid=own_ppid(), server_pid=1)
    assert target_banner.resolve_session_target("live") == "live"


def test_session_target_ignores_dead_server(state_root):
    """Residue of a server that died is not a target anyone is on."""
    write_state(state_root, target="va", owner_ppid=own_ppid(), server_pid=_DEAD_PID)
    assert target_banner.resolve_session_target("live") == "live"


def test_session_target_corrupt_state_is_baseline(state_root):
    write_state(state_root, target=None, owner_ppid=None, server_pid=own_pid(), raw="{not json")
    assert target_banner.resolve_session_target("live") == "live"


def test_session_target_unknown_target_name_is_baseline(state_root):
    write_state(state_root, target="banana", owner_ppid=own_ppid(), server_pid=own_pid())
    assert target_banner.resolve_session_target("live") == "live"


def test_situation_reports_both_targets(tmp_path, monkeypatch, state_root):
    set_config(tmp_path, monkeypatch, {"type": "virtual_accelerator"})
    write_state(state_root, target="live", owner_ppid=own_ppid(), server_pid=own_pid())
    situation = target_banner.resolve_target_situation()
    assert (situation.baseline_target, situation.session_target) == ("va", "live")
    assert situation.switched is True


# ── helper: rendering ───────────────────────────────────────────────────────
def test_pinned_line_is_none_on_baseline():
    situation = target_banner.TargetSituation(session_target="live", baseline_target="live")
    assert target_banner.baseline_pinned_line("Phoebus", situation) is None


def test_pinned_line_names_baseline_then_session():
    situation = target_banner.TargetSituation(session_target="va", baseline_target="live")
    assert target_banner.baseline_pinned_line("Phoebus", situation) == (
        "Phoebus is pinned to the deployment baseline (live); the session target is va"
    )


def test_pinned_line_takes_the_subject_from_the_caller():
    """HealthRuntime renders its own row from the same helper."""
    situation = target_banner.TargetSituation(session_target="live", baseline_target="va")
    line = target_banner.baseline_pinned_line("HealthRuntime", situation)
    assert line.startswith("HealthRuntime is pinned to the deployment baseline (va)")


def test_refusal_is_none_on_baseline():
    situation = target_banner.TargetSituation(session_target="va", baseline_target="va")
    assert target_banner.baseline_refusal("Phoebus", "Driving a widget", situation) is None


def test_refusal_names_both_targets_and_the_way_back():
    situation = target_banner.TargetSituation(session_target="va", baseline_target="live")
    message, suggestions = target_banner.baseline_refusal(
        "Phoebus", "Driving a Phoebus widget", situation
    )
    assert "deployment baseline (live)" in message
    assert "session target is va" in message
    assert "Driving a Phoebus widget" in message
    assert any("control_target_set(target='live')" in s for s in suggestions)


@pytest.mark.parametrize("line", [None, ""])
def test_prepend_line_leaves_the_payload_untouched(line):
    assert target_banner.prepend_line(line, '{"a": 1}') == '{"a": 1}'


def test_prepend_line_puts_the_line_first():
    assert target_banner.prepend_line("note", '{"a": 1}') == 'note\n{"a": 1}'


# ── phoebus_drive: refusal ──────────────────────────────────────────────────
async def test_drive_refuses_while_switched(switched):
    with patch(f"{_BRIDGE_MOD}._http_post_drive") as post:
        with assert_raises_error(error_type="target_switched") as ctx:
            await bridge_fn("phoebus_drive")(widget="SetButton", verb="click")
    # The refusal is a decision, not a failed round trip: nothing was sent.
    post.assert_not_called()
    message = ctx["envelope"]["error_message"]
    assert "deployment baseline (live)" in message and "session target is va" in message
    assert any("control_target_set(target='live')" in s for s in ctx["envelope"]["suggestions"])


async def test_drive_refusal_precedes_argument_validation(switched):
    """An operator on the wrong target learns that, not that their verb is bad."""
    with assert_raises_error(error_type="target_switched"):
        await bridge_fn("phoebus_drive")(widget="0", verb="frobnicate")


async def test_drive_proceeds_on_baseline(on_baseline):
    with patch(
        f"{_BRIDGE_MOD}._http_post_drive", return_value=(200, {"fired": True, "detail": "ok"})
    ) as post:
        result = await bridge_fn("phoebus_drive")(widget="SetButton", verb="click")
    post.assert_called_once()
    assert json.loads(result)["fired"] is True


async def test_drive_proceeds_when_state_is_corrupt(tmp_path, monkeypatch, state_root):
    """Unreadable state means "unknown", and unknown must not block an operator."""
    set_config(tmp_path, monkeypatch, {"type": "epics"})
    write_state(state_root, target=None, owner_ppid=None, server_pid=own_pid(), raw="}{")
    with patch(
        f"{_BRIDGE_MOD}._http_post_drive", return_value=(200, {"fired": True, "detail": "ok"})
    ):
        result = await bridge_fn("phoebus_drive")(widget="SetButton", verb="click")
    assert json.loads(result)["status"] == "success"


async def test_drive_proceeds_when_another_session_switched(tmp_path, monkeypatch, state_root):
    set_config(tmp_path, monkeypatch, {"type": "epics"})
    write_state(state_root, target="va", owner_ppid=_FOREIGN_PPID, server_pid=own_pid())
    with patch(
        f"{_BRIDGE_MOD}._http_post_drive", return_value=(200, {"fired": True, "detail": "ok"})
    ):
        result = await bridge_fn("phoebus_drive")(widget="SetButton", verb="click")
    assert json.loads(result)["status"] == "success"


# ── read tools: the informational line ──────────────────────────────────────
_EXPECTED_LINE = "Phoebus is pinned to the deployment baseline (live); the session target is va"


def _split_label(result):
    """Return ``(first_line, parsed_json)`` for a labelled tool result."""
    line, _, payload = result.partition("\n")
    return line, json.loads(payload)


async def _call_perceive():
    body = {"display": {"name": "demo"}, "widgets": []}
    with patch(f"{_BRIDGE_MOD}._http_get_json", return_value=(200, body)):
        return await bridge_fn("phoebus_perceive")(display="active")


async def _call_perceive_region():
    body = {"display": {"name": "demo"}, "widgets": []}
    with patch(f"{_BRIDGE_MOD}._http_get_json", return_value=(200, body)):
        return await bridge_fn("phoebus_perceive_region")(x=0, y=0, w=10, h=10)


async def _call_snapshot(tmp_path):
    entry = MagicMock()
    entry.to_tool_response.return_value = {"status": "success", "artifact_id": "abc"}
    store = MagicMock()
    store.save_data.return_value = entry
    with (
        patch(f"{_BRIDGE_MOD}._snapshot_dir", return_value=tmp_path),
        patch(f"{_BRIDGE_MOD}._http_get_bytes", return_value=(200, {}, b"\x89PNG\r\n\x1a\nx")),
        patch("osprey.stores.artifact_store.get_artifact_store", return_value=store),
    ):
        return await bridge_fn("phoebus_snapshot")(widget="Setpoint")


async def _call_snapshot_without_artifact_store(tmp_path):
    """The fallback success path must carry the same label as the artifact one."""
    with (
        patch(f"{_BRIDGE_MOD}._snapshot_dir", return_value=tmp_path),
        patch(f"{_BRIDGE_MOD}._http_get_bytes", return_value=(200, {}, b"\x89PNG\r\n\x1a\nx")),
        patch("osprey.stores.artifact_store.get_artifact_store", side_effect=RuntimeError("no")),
    ):
        return await bridge_fn("phoebus_snapshot")(widget="Setpoint")


async def _call_open_databrowser(tmp_path):
    with patch(f"{_DB_MOD}._http_post_open", return_value=(200, {"id": "d-1", "ready": True})):
        return await get_tool_fn(databrowser_tools.phoebus_open_databrowser)(channels=["SR:DCCT"])


_READ_CALLS = {
    "phoebus_perceive": lambda tmp_path: _call_perceive(),
    "phoebus_perceive_region": lambda tmp_path: _call_perceive_region(),
    "phoebus_snapshot": _call_snapshot,
    "phoebus_snapshot_fallback": _call_snapshot_without_artifact_store,
    "phoebus_open_databrowser": _call_open_databrowser,
}


@pytest.mark.parametrize("tool", sorted(_READ_CALLS))
async def test_read_tools_prepend_the_line_while_switched(tmp_path, tool, switched):
    result = await _READ_CALLS[tool](tmp_path)
    line, payload = _split_label(result)
    assert line == _EXPECTED_LINE
    assert payload["status"] == "success"


@pytest.mark.parametrize("tool", sorted(_READ_CALLS))
async def test_read_tools_add_nothing_on_baseline(tmp_path, tool, on_baseline):
    """On the baseline the result is exactly the JSON it has always been."""
    result = await _READ_CALLS[tool](tmp_path)
    assert result.startswith("{")
    assert "pinned to the deployment baseline" not in result
    json.loads(result)  # still a single parseable JSON document
