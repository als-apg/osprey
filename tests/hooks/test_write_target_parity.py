"""The write placement: one in the package, one in the hook twin, pinned together.

``osprey_writes_check`` places every hooked call with the stdlib twin's
``hook_write_target`` before it judges it; the chat's approval card names
the machine before the hook has run, through
:func:`osprey_connectors.write_target.write_target_for_tool`. Both are driven
here over the same render, hook_config, record and call, for every write tool
the render and the fallback list name, so the card never names a machine the
hook would not.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_hook_log as hook_log
import osprey.templates.claude_code.claude.hooks.osprey_target_state as twin
from osprey_connectors import posture_store
from osprey_connectors.write_target import (
    FALLBACK_WRITE_TOOLS,
    UNRESOLVED,
    short_tool_name,
    write_target_for_tool,
)
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    state_dir_under,
    write_control_context,
)

TWO_TARGETS = {
    "type": "epics",
    "writes": "allowed",
    "connector": {"epics": {"prefix": "RING:"}, "virtual_accelerator": {"prefix": "VA:"}},
}

FACILITY_SERVER_TOOL = "mcp__magnets__set_current"
DECLARED_WRITE = "mcp__plc__set_value"
DECLARED_READ = "mcp__plc__get_value"
FACILITY_EXTRA = "mcp__orbit__apply_correction"
APP_WRITE = "mcp__knob__set_knob"

HOOK_CONFIG: dict[str, Any] = {
    "server_prefixes": [
        "mcp__controls__",
        "mcp__python__",
        "mcp__bluesky__",
        "mcp__magnets__",
        "mcp__orbit__",
        "mcp__phoebus__",
        "mcp__plc__",
        "mcp__knob__",
    ],
    "write_tools": [
        "mcp__controls__channel_write",
        "mcp__bluesky__queue_add",
        "mcp__bluesky__queue_start",
        "mcp__python__execute",
        "mcp__python__execute_file",
        "mcp__phoebus__phoebus_drive",
        "mcp__magnets__.*",
        DECLARED_WRITE,
        FACILITY_EXTRA,
        APP_WRITE,
    ],
    "control_system_write_tools": [FACILITY_EXTRA],
    "facility_write_tools": [DECLARED_WRITE, "mcp__magnets__.*", APP_WRITE],
    "app_approvals": {APP_WRITE: {"app": "knob"}},
    "lane_addressed_tools": ["queue_add", "queue_start", "queue_stop", "queue_remove"],
}

#: The same render from before the ``facility_write_tools`` key: both sides read
#: the whole-server matchers in ``write_tools`` as the facility set.
NO_FACILITY_KEY: dict[str, Any] = {
    k: v for k, v in HOOK_CONFIG.items() if k != "facility_write_tools"
}

#: A hook_config the render wrote with no write list: both sides fall back.
NO_WRITE_LIST: dict[str, Any] = {"server_prefixes": HOOK_CONFIG["server_prefixes"]}

CONFIGS: dict[str, dict[str, Any]] = {
    "two-lanes": {
        "control_system": TWO_TARGETS,
        "services": {"bluesky_live": {"target": "live"}, "bluesky_va": {"target": "va"}},
    },
    "one-lane": {"control_system": TWO_TARGETS, "services": {"bluesky": {}}},
    "lane-serves-nothing-known": {
        "control_system": TWO_TARGETS,
        "services": {"bluesky_standin": {"target": "elsewhere"}},
    },
    "simulator-only": {"control_system": {"type": "virtual_accelerator", "writes": "allowed"}},
    "standin-baseline": {
        "control_system": {"type": "epics", "connector": {"live_standin": {}}},
    },
}

#: (id, tool, tool_input) — every write tool, a read, and the execute modes.
CALLS: list[tuple[str, str, dict[str, Any]]] = [
    ("channel-write", "mcp__controls__channel_write", {"value": 1.5}),
    ("channel-read", "mcp__controls__channel_read", {}),
    ("execute-readonly", "mcp__python__execute", {"execution_mode": "readonly"}),
    ("execute-modeless", "mcp__python__execute", {"code": "1"}),
    ("execute-readwrite", "mcp__python__execute", {"execution_mode": "readwrite"}),
    ("execute-file", "mcp__python__execute_file", {"path": "a.py"}),
    ("pyaml-measure", "mcp__python__pyaml_measure", {}),
    ("phoebus-drive", "mcp__phoebus__phoebus_drive", {}),
    ("facility-server", FACILITY_SERVER_TOOL, {}),
    ("facility-server-read", "mcp__magnets__read_current", {}),
    ("facility-extra", FACILITY_EXTRA, {}),
    ("declared-write", DECLARED_WRITE, {}),
    ("declared-read", DECLARED_READ, {}),
    ("app-write", APP_WRITE, {}),
    ("queue-start-no-lane", "mcp__bluesky__queue_start", {}),
    ("queue-start-va", "mcp__bluesky__queue_start", {"lane": "bluesky_va"}),
    ("queue-start-live", "mcp__bluesky__queue_start", {"lane": "bluesky_live"}),
    ("queue-start-lane-one", "mcp__bluesky__queue_start", {"lane": "bluesky"}),
    ("queue-start-unknown-lane", "mcp__bluesky__queue_start", {"lane": "nope"}),
    ("queue-add-standin", "mcp__bluesky__queue_add", {"lane": "bluesky_standin"}),
    ("not-mcp", "Bash", {"command": "ls"}),
]

#: How the session's record stands: a target, absent, or corrupt.
RECORDS = ["va", "live", "standin", "missing", "corrupt"]


def _repo(tmp_path: Path) -> Path:
    return tmp_path / "repo"


def _root(tmp_path: Path) -> Path:
    return _repo(tmp_path) / "var" / "agent_data"


def _set_record(root: Path, record: str) -> None:
    if record == "missing":
        return
    if record == "corrupt":
        directory = state_dir_under(root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "control_context.json").write_text('{"schema": ', encoding="utf-8")
        return
    write_control_context(root, target=record, generation=3)


def _both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    hook_config: dict[str, Any],
    tool: str,
    tool_input: dict[str, Any],
) -> tuple[Any, Any]:
    """``(package, twin)`` placement of one call, UNRESOLVED read as one word."""
    path = tmp_path / "hook_config.json"
    path.write_text(json.dumps(hook_config), encoding="utf-8")
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(path))
    hook_log._hook_config_cache = None
    hook_input = {"tool_name": tool, "tool_input": tool_input, "cwd": str(_repo(tmp_path))}
    prefixes = [*hook_config.get("server_prefixes", ())]
    short = hook_log.short_tool_name(tool, prefixes)
    assert short_tool_name(tool, prefixes) == short
    package = write_target_for_tool(config, tool, tool_input, hook_config=hook_config)
    hooked = twin.hook_write_target(config, tool, short, hook_input)
    return (
        "<unresolved>" if package is UNRESOLVED else package,
        "<unresolved>" if hooked is twin.UNRESOLVED else hooked,
    )


@pytest.fixture
def stamped(tmp_path, monkeypatch) -> Path:
    root = _root(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    isolate_verdict_environment(monkeypatch, root)
    return root


@pytest.mark.parametrize(
    ("config_id", "record", "call"),
    list(itertools.product(CONFIGS, RECORDS, CALLS)),
    ids=[f"{c}-{r}-{call[0]}" for c, r, call in itertools.product(CONFIGS, RECORDS, CALLS)],
)
def test_both_place_every_call_alike(tmp_path, monkeypatch, stamped, config_id, record, call):
    _id, tool, tool_input = call
    _set_record(stamped, record)

    package, hooked = _both(
        tmp_path, monkeypatch, CONFIGS[config_id], HOOK_CONFIG, tool, tool_input
    )

    assert package == hooked


@pytest.mark.parametrize("record", RECORDS)
@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("mcp__controls__channel_write", {}),
        ("mcp__phoebus__phoebus_drive", {}),
        ("mcp__python__pyaml_measure", {}),
        ("mcp__bluesky__queue_start", {"lane": "bluesky_va"}),
        (FACILITY_SERVER_TOOL, {}),
    ],
)
def test_both_fall_back_alike_when_the_render_names_no_write_list(
    tmp_path, monkeypatch, stamped, record, tool, tool_input
):
    _set_record(stamped, record)

    package, hooked = _both(
        tmp_path, monkeypatch, CONFIGS["two-lanes"], NO_WRITE_LIST, tool, tool_input
    )

    assert package == hooked


@pytest.mark.parametrize("record", ["va", "missing"])
def test_an_unstamped_process_with_no_record_cannot_place_a_session_write(
    tmp_path, monkeypatch, record
):
    """No stamp and no record: "nobody switched" and "looked in the wrong place" look
    alike, so neither side names a machine. A record found through the bind places it."""
    bind = tmp_path / "bind"
    bind.mkdir()
    isolate_verdict_environment(monkeypatch, _root(tmp_path))
    monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(bind))
    if record != "missing":
        (bind / "control_context.json").write_text(
            json.dumps({"schema": 1, "target": record, "generation": 1}), encoding="utf-8"
        )

    package, hooked = _both(
        tmp_path,
        monkeypatch,
        CONFIGS["two-lanes"],
        HOOK_CONFIG,
        "mcp__controls__channel_write",
        {},
    )

    assert package == hooked == ("<unresolved>" if record == "missing" else record)


def test_the_placement_is_the_sessions_machine_not_a_gateway_role(tmp_path, monkeypatch, stamped):
    """A framework write follows the session's machine; a facility write the real one."""
    write_control_context(stamped, target="va", generation=3)
    config = CONFIGS["two-lanes"]

    assert _both(
        tmp_path, monkeypatch, config, HOOK_CONFIG, "mcp__controls__channel_write", {}
    ) == (
        "va",
        "va",
    )
    assert _both(tmp_path, monkeypatch, config, HOOK_CONFIG, FACILITY_SERVER_TOOL, {}) == (
        "live",
        "live",
    )


@pytest.mark.parametrize("hook_config", [HOOK_CONFIG, NO_FACILITY_KEY], ids=["keyed", "pre-key"])
@pytest.mark.parametrize("record", RECORDS)
@pytest.mark.parametrize("call", CALLS, ids=[c[0] for c in CALLS])
def test_both_read_the_facility_set_alike(
    tmp_path, monkeypatch, stamped, hook_config, record, call
):
    _id, tool, tool_input = call
    _set_record(stamped, record)

    package, hooked = _both(
        tmp_path, monkeypatch, CONFIGS["two-lanes"], hook_config, tool, tool_input
    )

    assert package == hooked


def test_a_declaring_server_sends_its_declared_tool_to_live_and_its_read_nowhere(
    tmp_path, monkeypatch, stamped
):
    write_control_context(stamped, target="va", generation=3)
    config = CONFIGS["two-lanes"]

    assert _both(tmp_path, monkeypatch, config, HOOK_CONFIG, DECLARED_WRITE, {}) == (
        "live",
        "live",
    )
    assert _both(tmp_path, monkeypatch, config, HOOK_CONFIG, DECLARED_READ, {}) == (None, None)
    assert _both(tmp_path, monkeypatch, config, HOOK_CONFIG, "mcp__magnets__read_current", {}) == (
        "live",
        "live",
    )


def test_an_application_write_follows_the_sessions_machine(tmp_path, monkeypatch, stamped):
    """An application tool is in the facility write set, yet writes where the session points."""
    write_control_context(stamped, target="va", generation=3)

    assert _both(tmp_path, monkeypatch, CONFIGS["two-lanes"], HOOK_CONFIG, APP_WRITE, {}) == (
        "va",
        "va",
    )


#: The shipped virtual-accelerator preset's section, with writes allowed.
VA_PRESET = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {"serving": "in_process"}},
    "writes": "allowed",
}


@pytest.mark.parametrize("record", ["va", "missing"])
def test_an_application_write_on_the_va_preset_is_not_refused_target_unknown(
    tmp_path, monkeypatch, stamped, record
):
    """A simulator-only deployment has no real machine to send an application write to."""
    _set_record(stamped, record)

    package, hooked = _both(
        tmp_path, monkeypatch, {"control_system": VA_PRESET}, HOOK_CONFIG, APP_WRITE, {}
    )

    assert package == hooked == "va"
    assert twin.write_verdict(VA_PRESET, hooked).reason != "target_unknown"


def test_without_the_key_a_whole_server_matcher_is_still_facility(tmp_path, monkeypatch, stamped):
    """A render from before the key keeps sending its whole-server servers to `live`."""
    write_control_context(stamped, target="va", generation=3)

    assert _both(
        tmp_path, monkeypatch, CONFIGS["two-lanes"], NO_FACILITY_KEY, FACILITY_SERVER_TOOL, {}
    ) == ("live", "live")


def test_the_fallback_write_lists_match():
    assert list(FALLBACK_WRITE_TOOLS) == hook_log.FALLBACK_WRITE_TOOLS


def test_the_lane_keys_match():
    from osprey_connectors import write_target

    assert write_target._LANE_KEYS == twin._LANE_KEYS
    assert write_target._LANE_ADDRESSED_KEY == twin._LANE_ADDRESSED_KEY
    assert write_target._FACILITY_EXTRAS_KEY == twin._FACILITY_EXTRAS_KEY
    assert write_target._FACILITY_WRITES_KEY == twin._FACILITY_WRITES_KEY
    assert write_target._APP_APPROVALS_KEY == twin._APP_APPROVALS_KEY


@pytest.mark.usefixtures("stamped")
def test_a_failure_while_placing_is_unresolved_never_a_read(monkeypatch):
    from osprey_connectors import write_target

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("no")

    monkeypatch.setattr(write_target.control_context, "read_record_status", boom)

    placed = write_target_for_tool(
        CONFIGS["two-lanes"], "mcp__controls__channel_write", {}, hook_config=HOOK_CONFIG
    )

    assert placed is UNRESOLVED
