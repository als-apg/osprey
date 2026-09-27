"""Tests for the shared hook-logging library ``osprey_hook_log``.

This module is imported by other hooks rather than invoked as a script, so it is
tested by direct import. It carries module-level caches
(``_hook_config_cache``, ``_osprey_config_cache``, ``_debug_from_config``) that
must be reset between tests to keep the unit lane serial-safe. Coverage here
targets the config loaders, stdin/project-dir resolution, and the dual-sink
``log_hook`` gate; ``_is_debug_enabled`` is exercised elsewhere.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

import osprey.templates.claude_code.claude.hooks.osprey_hook_log as hook_log
from osprey.utils import workspace
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR


@pytest.fixture(autouse=True)
def _reset_caches_and_env(monkeypatch):
    """Reset module caches and the env vars the loaders read, before and after."""
    for var in ("OSPREY_HOOK_CONFIG", "OSPREY_CONFIG", "OSPREY_HOOK_DEBUG", "CLAUDE_PROJECT_DIR"):
        monkeypatch.delenv(var, raising=False)
    hook_log._hook_config_cache = None
    hook_log._osprey_config_cache = None
    hook_log._debug_from_config = None
    yield
    hook_log._hook_config_cache = None
    hook_log._osprey_config_cache = None
    hook_log._debug_from_config = None


# ---------------------------------------------------------------------------
# load_hook_config
# ---------------------------------------------------------------------------


def test_load_hook_config_reads_env_var_path(tmp_path, monkeypatch):
    cfg = tmp_path / "hook_config.json"
    cfg.write_text(json.dumps({"approval_prefixes": ["mcp__controls__"]}))
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(cfg))

    assert hook_log.load_hook_config() == {"approval_prefixes": ["mcp__controls__"]}


def test_load_hook_config_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(tmp_path / "does-not-exist.json"))
    assert hook_log.load_hook_config() == {}


def test_load_hook_config_unparseable_returns_empty(tmp_path, monkeypatch):
    bad = tmp_path / "hook_config.json"
    bad.write_text("{not valid json")
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(bad))
    assert hook_log.load_hook_config() == {}


def test_load_hook_config_caches(tmp_path, monkeypatch):
    cfg = tmp_path / "hook_config.json"
    cfg.write_text(json.dumps({"a": 1}))
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(cfg))

    first = hook_log.load_hook_config()
    cfg.unlink()  # cache must survive the file disappearing
    assert hook_log.load_hook_config() == first == {"a": 1}


# ---------------------------------------------------------------------------
# get_hook_input
# ---------------------------------------------------------------------------


def test_get_hook_input_parses_stdin(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"tool_name": "channel_write"})))
    assert hook_log.get_hook_input() == {"tool_name": "channel_write"}


def test_get_hook_input_empty_stdin_returns_empty(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert hook_log.get_hook_input() == {}


def test_get_hook_input_malformed_stdin_returns_empty(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json{{{"))
    assert hook_log.get_hook_input() == {}


@pytest.mark.parametrize("payload", ["[1, 2, 3]", '"str"', "42", "true"])
def test_get_hook_input_truthy_non_dict_returns_empty(payload, monkeypatch):
    """Valid JSON that parses to a truthy non-dict must still yield {}.

    Consumers gate on ``if not hook_input`` and then call ``.get()``. A truthy
    non-dict clears that gate and would raise ``AttributeError``, exiting
    non-zero — which makes fail-open PreToolUse hooks block the tool call.
    """
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    assert hook_log.get_hook_input() == {}


@pytest.mark.parametrize("payload", ["[]", "0"])
def test_get_hook_input_falsy_non_dict_returns_empty(payload, monkeypatch):
    """Falsy non-dicts already short-circuit in consumers; pin them anyway."""
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    assert hook_log.get_hook_input() == {}


# ---------------------------------------------------------------------------
# get_project_dir
# ---------------------------------------------------------------------------


def test_get_project_dir_prefers_env_var(monkeypatch):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/from/env")
    assert hook_log.get_project_dir({"cwd": "/from/input"}) == "/from/env"


def test_get_project_dir_falls_back_to_cwd():
    assert hook_log.get_project_dir({"cwd": "/from/input"}) == "/from/input"


def test_get_project_dir_empty_when_nothing_set():
    assert hook_log.get_project_dir({}) == ""


# ---------------------------------------------------------------------------
# load_osprey_config
# ---------------------------------------------------------------------------


def test_load_osprey_config_reads_project_config(tmp_path):
    (tmp_path / "config.yml").write_text(yaml.dump({"hooks": {"debug": True}}))
    cfg = hook_log.load_osprey_config({"cwd": str(tmp_path)})
    assert cfg == {"hooks": {"debug": True}}


def test_load_osprey_config_env_override_wins(tmp_path, monkeypatch):
    custom = tmp_path / "custom.yml"
    custom.write_text(yaml.dump({"marker": "from-env"}))
    monkeypatch.setenv("OSPREY_CONFIG", str(custom))
    # cwd has no config.yml, but OSPREY_CONFIG points to the custom one.
    assert hook_log.load_osprey_config({"cwd": "/nonexistent"}) == {"marker": "from-env"}


def test_load_osprey_config_missing_returns_empty(tmp_path):
    assert hook_log.load_osprey_config({"cwd": str(tmp_path)}) == {}


def test_load_osprey_config_caches(tmp_path):
    (tmp_path / "config.yml").write_text(yaml.dump({"x": 1}))
    first = hook_log.load_osprey_config({"cwd": str(tmp_path)})
    (tmp_path / "config.yml").unlink()
    assert hook_log.load_osprey_config({"cwd": str(tmp_path)}) == first == {"x": 1}


# ---------------------------------------------------------------------------
# agent_data_base_dir
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"agent_data": None},
        {"agent_data": "x"},
        {"agent_data": {}},
        {"agent_data": {"base_dir": ""}},
        {"agent_data": {"base_dir": "custom/agent_data"}},
        {"agent_data": {"base_dir": "/abs/root"}},
    ],
)
def test_agent_data_base_dir_matches_the_framework_reader(config):
    assert hook_log.agent_data_base_dir(config) == workspace.agent_data_base_dir(config)


def test_agent_data_fallback_literal_is_the_framework_default():
    source = Path(hook_log.__file__).read_text()
    assert '_DEFAULT_AGENT_DATA_ROOT = "' + DEFAULT_AGENT_DATA_BASE_DIR + '"' in source


def test_repo_agent_data_root_follows_a_relocated_base_dir(tmp_path):
    (tmp_path / "config.yml").write_text("agent_data:\n  base_dir: relocated/agent_data\n")

    root = hook_log.repo_agent_data_root({"cwd": str(tmp_path)})

    assert root == str(tmp_path / "relocated/agent_data")


def test_repo_agent_data_root_defaults_without_a_config(tmp_path):
    root = hook_log.repo_agent_data_root({"cwd": str(tmp_path)})

    assert root == str(tmp_path / DEFAULT_AGENT_DATA_BASE_DIR)


def test_an_absolute_base_dir_is_not_re_anchored(tmp_path, monkeypatch):
    absolute = str(tmp_path / "abs")
    config = {"agent_data": {"base_dir": absolute}}

    assert hook_log.agent_data_root_at("/anchor", config) == absolute

    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    subdirs = hook_log.agent_data_subdirs({"cwd": str(tmp_path / "elsewhere")}, "notebooks")

    assert subdirs == [(tmp_path / "abs" / "notebooks").resolve()]


def test_agent_data_subdirs_covers_both_anchors_once(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    build = repo / "build"
    build.mkdir(parents=True)
    (build / "config.yml").write_text("agent_data:\n  base_dir: relocated/agent_data\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(build / "config.yml"))

    zoned = hook_log.agent_data_subdirs({"cwd": str(build)}, "notebooks")

    assert zoned == [
        (repo / "relocated/agent_data/notebooks").resolve(),
        (build / "relocated/agent_data/notebooks").resolve(),
    ]

    hook_log._osprey_config_cache = None
    (repo / "config.yml").write_text("agent_data:\n  base_dir: relocated/agent_data\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(repo / "config.yml"))

    flat = hook_log.agent_data_subdirs({"cwd": str(repo)}, "notebooks")

    assert flat == [(repo / "relocated/agent_data/notebooks").resolve()]


# ---------------------------------------------------------------------------
# log_hook — dual-sink gate
# ---------------------------------------------------------------------------


def _hooks_dir(tmp_path):
    d = tmp_path / ".claude" / "hooks"
    d.mkdir(parents=True)
    return d


def test_log_hook_noop_when_debug_disabled(tmp_path, capsys, monkeypatch):
    _hooks_dir(tmp_path)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    # Debug not enabled (no OSPREY_HOOK_DEBUG, no config).

    hook_log.log_hook("osprey_test", {"tool_name": "channel_write"}, status="ok")

    assert capsys.readouterr().err == ""
    assert not (tmp_path / ".claude" / "hooks" / "hook_debug.jsonl").exists()


def test_log_hook_writes_stderr_and_jsonl_when_enabled(tmp_path, capsys, monkeypatch):
    _hooks_dir(tmp_path)
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))

    hook_log.log_hook(
        "osprey_test",
        {"tool_name": "channel_write", "tool_use_id": "abc123"},
        status="blocked",
        detail="writes disabled",
    )

    err = capsys.readouterr().err
    assert "[osprey_test]" in err
    assert "status=blocked" in err
    assert "writes disabled" in err

    jsonl = tmp_path / ".claude" / "hooks" / "hook_debug.jsonl"
    record = json.loads(jsonl.read_text().strip())
    assert record["hook"] == "osprey_test"
    assert record["tool"] == "channel_write"
    assert record["status"] == "blocked"
    assert record["tool_use_id"] == "abc123"
    assert record["detail"] == "writes disabled"


def test_log_hook_appends_multiple_records(tmp_path, monkeypatch):
    _hooks_dir(tmp_path)
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))

    hook_log.log_hook("h", {"tool_name": "a"}, status="ok")
    hook_log.log_hook("h", {"tool_name": "b"}, status="ok")

    lines = (tmp_path / ".claude" / "hooks" / "hook_debug.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(x)["tool"] for x in lines] == ["a", "b"]


def test_log_hook_survives_missing_log_dir(tmp_path, capsys, monkeypatch):
    """No .claude/hooks dir -> the JSONL write is best-effort and swallowed, but
    the stderr line still prints and nothing raises."""
    monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))  # no .claude/hooks created

    hook_log.log_hook("h", {"tool_name": "a"}, status="ok")  # must not raise

    assert "[h]" in capsys.readouterr().err
    assert not (tmp_path / ".claude" / "hooks" / "hook_debug.jsonl").exists()
