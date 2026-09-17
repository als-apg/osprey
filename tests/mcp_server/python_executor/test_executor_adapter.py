"""Unit tests for the MCP execution adapter (executor.py).

Tests the adapter module in isolation with mocked executors.
Pattern: monkeypatch.chdir(tmp_path) -> write config.yml -> mock deps -> call adapter.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from osprey.mcp_server.python_executor.executor import (
    ExecutionResult,
    _collect_figures,
    _create_execution_folder,
    _load_limits_validator,
    _read_config,
    _read_execution_metadata,
    execute_code,
    resolve_agent_interpreter,
)
from osprey_connectors.config import DEFAULT_EXECUTION_TIMEOUT_SECONDS, ConfigBuilder


@pytest.fixture(autouse=True)
def _reset_all_config_caches(monkeypatch):
    """Reset ALL config caches before each test.

    Prior test modules set the ConfigBuilder singleton via
    get_config_builder(set_as_default=True).  We must clear
    _default_config, _default_configurable, and _config_cache
    before each test so the adapter reads from the test's own
    config.yml via monkeypatch.chdir(tmp_path).
    """
    from osprey.utils.workspace import reset_config_cache

    reset_config_cache()

    import osprey.utils.config as _cfg

    monkeypatch.setattr(_cfg, "_default_config", None)
    monkeypatch.setattr(_cfg, "_default_configurable", None)
    # Save and clear the cache dict; monkeypatch restores it on teardown
    saved_cache = _cfg._config_cache.copy()
    _cfg._config_cache.clear()

    yield

    reset_config_cache()
    _cfg._config_cache.clear()
    _cfg._config_cache.update(saved_cache)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path, overrides=None):
    """Write a config.yml with execution infrastructure settings."""
    config = {
        "control_system": {
            "type": "mock",
            "limits_checking": {"enabled": False},
        },
        "execution": {
            "execution_method": "subprocess",
        },
        "python_executor": {
            "execution_timeout_seconds": 300,
        },
    }
    if overrides:
        _deep_merge(config, overrides)
    (tmp_path / "config.yml").write_text(yaml.dump(config))
    return config


def _deep_merge(base, overrides):
    """Recursively merge overrides into base dict."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("configured", ["subprocess", "local", "container"])
def test_config_resolves_every_accepted_method_to_subprocess(tmp_path, monkeypatch, configured):
    """Every accepted execution_method value resolves to the subprocess backend."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"execution": {"execution_method": configured}})
    config = _read_config()
    assert config["execution_method"] == "subprocess"


def test_config_defaults_execution_method_to_subprocess(tmp_path, monkeypatch):
    """When execution_method missing, defaults to 'subprocess'."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({}))
    config = _read_config()
    assert config["execution_method"] == "subprocess"


def test_config_rejects_unknown_execution_method(tmp_path, monkeypatch):
    """An unrecognised execution_method is a hard config error, not a silent default."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"execution": {"execution_method": "kubernetes"}})
    with pytest.raises(ValueError, match="execution.execution_method"):
        _read_config()


def test_config_reads_timeout(tmp_path, monkeypatch):
    """Adapter reads python_executor.execution_timeout_seconds."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"python_executor": {"execution_timeout_seconds": 120}})
    config = _read_config()
    assert config["timeout"] == 120


def test_config_timeout_default(tmp_path, monkeypatch):
    """When timeout config absent, the adapter falls back to the shared default."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({}))
    config = _read_config()
    assert config["timeout"] == DEFAULT_EXECUTION_TIMEOUT_SECONDS


def test_config_timeout_default_matches_the_builder_default(tmp_path, monkeypatch):
    """Both ends of the executor timeout read the same constant."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({}))
    builder = ConfigBuilder(config_path=str(tmp_path / "config.yml"))

    assert (
        builder._get_python_executor_config()["execution_timeout_seconds"]
        == _read_config()["timeout"]
        == DEFAULT_EXECUTION_TIMEOUT_SECONDS
    )


# ---------------------------------------------------------------------------
# Agent-code interpreter resolution
# ---------------------------------------------------------------------------


def test_agent_interpreter_prefers_project_venv(tmp_path):
    """Agent code runs in the project's own venv when the project ships one."""
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n")

    assert resolve_agent_interpreter(tmp_path) == venv_python


def test_agent_interpreter_falls_back_to_sys_executable(tmp_path):
    """Without a project venv, agent code runs in the interpreter running OSPREY."""
    assert resolve_agent_interpreter(tmp_path) == Path(sys.executable)


def test_agent_interpreter_defaults_to_resolved_project_root(tmp_path, monkeypatch):
    """Called with no argument, the helper resolves the project root itself."""
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n")

    monkeypatch.setattr(
        "osprey.mcp_server.python_executor.executor._resolve_project_root",
        lambda: tmp_path,
    )
    assert resolve_agent_interpreter() == venv_python


async def test_subprocess_spawned_with_resolved_interpreter(tmp_path, monkeypatch):
    """The resolved interpreter is the binary actually handed to the subprocess."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)

    project_root = tmp_path / "project"
    venv_python = project_root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        "osprey.mcp_server.python_executor.executor._resolve_project_root",
        lambda: project_root,
    )

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0

    with patch(
        "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ) as mock_spawn:
        await execute_code("print(42)", "readonly", "test")

    assert mock_spawn.call_args.args[0] == str(venv_python)


# ---------------------------------------------------------------------------
# Execution folder
# ---------------------------------------------------------------------------


def test_execution_folder_created(tmp_path, monkeypatch):
    """Adapter creates timestamped folder in _agent_data/data/python_executions/."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    folder = _create_execution_folder()
    assert folder.exists()
    assert folder.parent.name == "python_executions"
    assert (folder / "figures").exists()


# ---------------------------------------------------------------------------
# Limits validator
# ---------------------------------------------------------------------------


def test_limits_validator_loaded_and_passed(tmp_path, monkeypatch):
    """LimitsValidator.from_config() is called when loading the validator."""
    monkeypatch.chdir(tmp_path)
    # Write config with limits enabled + a limits database
    limits_db = tmp_path / "channel_limits.json"
    limits_db.write_text(
        json.dumps({"TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}})
    )
    _write_config(
        tmp_path,
        {
            "control_system": {
                "limits_checking": {
                    "enabled": True,
                    "database_path": str(limits_db),
                    "allow_unlisted_channels": False,
                    "on_violation": "error",
                }
            }
        },
    )
    # Force ConfigBuilder to use this test's config.yml
    from osprey.utils.config import get_config_builder

    get_config_builder(config_path=str(tmp_path / "config.yml"), set_as_default=True)

    validator = _load_limits_validator(target=None)
    assert validator is not None
    assert "TEST:PV" in validator.limits


def test_limits_validator_disabled_gracefully(tmp_path, monkeypatch):
    """When limits_checking.enabled=false, returns None."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    validator = _load_limits_validator(target=None)
    assert validator is None


# ---------------------------------------------------------------------------
# Wrapper / monkeypatch
# ---------------------------------------------------------------------------


def test_wrapper_includes_monkeypatch_when_validator_present(tmp_path, monkeypatch):
    """ExecutionWrapper.create_wrapper() output contains monkeypatch code when validator present."""
    monkeypatch.chdir(tmp_path)
    limits_db = tmp_path / "channel_limits.json"
    limits_db.write_text(
        json.dumps({"TEST:PV": {"min_value": 0, "max_value": 100, "writable": True}})
    )
    _write_config(
        tmp_path,
        {
            "control_system": {
                "limits_checking": {
                    "enabled": True,
                    "database_path": str(limits_db),
                    "allow_unlisted_channels": False,
                    "on_violation": "error",
                }
            }
        },
    )
    # Force ConfigBuilder to use this test's config.yml
    from osprey.utils.config import get_config_builder

    get_config_builder(config_path=str(tmp_path / "config.yml"), set_as_default=True)

    validator = _load_limits_validator(target=None)
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    wrapper = ExecutionWrapper(limits_validator=validator)
    wrapped = wrapper.create_wrapper("print('hello')", tmp_path)
    assert "_checked_ca_put" in wrapped
    assert "LimitsValidator" in wrapped


def test_wrapper_omits_monkeypatch_when_no_validator(tmp_path, monkeypatch):
    """Wrapper output has no monkeypatch when validator is None."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    wrapper = ExecutionWrapper(limits_validator=None)
    wrapped = wrapper.create_wrapper("print('hello')", tmp_path)
    assert "_checked_ca_put" not in wrapped


# ---------------------------------------------------------------------------
# Executor dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("configured", ["subprocess", "local", "container"])
async def test_every_configured_method_runs_the_subprocess_backend(
    tmp_path, monkeypatch, configured
):
    """No accepted config value reaches anything but asyncio.create_subprocess_exec."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"execution": {"execution_method": configured}})

    # Mock the subprocess to avoid actually running code
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"42\n", b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch(
        "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ) as mock_spawn:
        result = await execute_code("print(42)", "readonly", "test")

    mock_spawn.assert_awaited_once()
    assert result.execution_method_used == "subprocess"
    assert result.stdout == "42\n"


async def test_deprecated_container_method_still_executes(tmp_path, monkeypatch, caplog):
    """A legacy 'container' config warns once but still runs the code."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"execution": {"execution_method": "container"}})

    import osprey.utils.config as _cfg

    monkeypatch.setattr(_cfg, "_container_method_warned", False)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
    mock_proc.returncode = 0

    with (
        # Logger name is "CONFIG" (src/osprey/utils/config.py) — NOT the module
        # path. Naming the wrong logger leaves CONFIG at its inherited level, so
        # the record is filtered at emit whenever an earlier test has raised the
        # root level, and caplog.text comes back empty.
        caplog.at_level("WARNING", logger="CONFIG"),
        patch(
            "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ),
    ):
        result = await execute_code("print('ok')", "readonly", "test")

    assert result.execution_method_used == "subprocess"
    assert "deprecated" in caplog.text


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


async def test_timeout_from_config(tmp_path, monkeypatch):
    """Adapter reads the configured timeout and applies it to the subprocess wait."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"python_executor": {"execution_timeout_seconds": 42}})

    assert _read_config()["timeout"] == 42

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0

    with (
        patch(
            "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ),
        patch(
            "osprey.mcp_server.python_executor.executor.asyncio.wait_for",
            new_callable=AsyncMock,
            return_value=(b"", b""),
        ) as mock_wait_for,
    ):
        await execute_code("pass", "readonly", "test")

    assert mock_wait_for.call_args.kwargs.get("timeout") == 42


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


def test_result_dataclass_populated():
    """ExecutionResult has correct fields when constructed."""
    result = ExecutionResult(
        success=True,
        stdout="hello",
        stderr="",
        figures=[Path("/tmp/fig.png")],
        execution_method_used="subprocess",
        execution_time_seconds=1.5,
    )
    assert result.success is True
    assert result.stdout == "hello"
    assert len(result.figures) == 1
    assert result.execution_method_used == "subprocess"
    assert result.execution_time_seconds == 1.5


def test_result_dataclass_defaults():
    """ExecutionResult defaults are sensible."""
    result = ExecutionResult(success=False, stdout="", stderr="error")
    assert result.figures == []
    assert result.execution_method_used == "subprocess"
    assert result.execution_time_seconds is None
    assert result.error_message is None


# ---------------------------------------------------------------------------
# Error handling (no in-process fallback)
# ---------------------------------------------------------------------------


async def test_subprocess_error_returns_failure_result(tmp_path, monkeypatch):
    """When subprocess fails, adapter returns failure result (no fallback)."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)

    with patch(
        "osprey.mcp_server.python_executor.executor._execute_via_local",
        new_callable=AsyncMock,
        side_effect=OSError("subprocess failed"),
    ):
        result = await execute_code("print('hi')", "readonly", "error test")

    assert result.success is False
    assert result.error_message is not None


# ---------------------------------------------------------------------------
# Figure collection
# ---------------------------------------------------------------------------


def test_figure_collection_from_execution_folder(execution_folder):
    """After execution, adapter scans figures/ dir and returns figure paths."""
    # Create some figure files
    (execution_folder / "figures" / "figure_01.png").write_bytes(b"PNG")
    (execution_folder / "figures" / "figure_02.png").write_bytes(b"PNG")

    figures = _collect_figures(execution_folder)
    assert len(figures) == 2
    assert all(f.suffix == ".png" for f in figures)


def test_figure_collection_empty_folder(execution_folder):
    """Empty execution folder returns no figures."""
    figures = _collect_figures(execution_folder)
    assert figures == []


def test_figure_collection_multiple_formats(execution_folder):
    """Collects PNG, JPG, JPEG, and SVG files."""
    (execution_folder / "figures" / "plot.png").write_bytes(b"PNG")
    (execution_folder / "figures" / "photo.jpg").write_bytes(b"JPG")
    (execution_folder / "figures" / "diagram.svg").write_text("<svg/>")

    figures = _collect_figures(execution_folder)
    assert len(figures) == 3


# ---------------------------------------------------------------------------
# Execution metadata reading
# ---------------------------------------------------------------------------


def test_read_execution_metadata(execution_folder):
    """Reads execution_metadata.json from execution folder."""
    metadata = {"success": True, "stdout": "hello", "stderr": ""}
    (execution_folder / "execution_metadata.json").write_text(json.dumps(metadata))

    result = _read_execution_metadata(execution_folder)
    assert result == metadata


def test_read_execution_metadata_missing(execution_folder):
    """Returns None when execution_metadata.json doesn't exist."""
    result = _read_execution_metadata(execution_folder)
    assert result is None


def test_read_execution_metadata_invalid_json(execution_folder):
    """Returns None when execution_metadata.json is invalid JSON."""
    (execution_folder / "execution_metadata.json").write_text("not json{{{")

    result = _read_execution_metadata(execution_folder)
    assert result is None


# ---------------------------------------------------------------------------
# Invalid config
# ---------------------------------------------------------------------------


async def test_invalid_execution_method_returns_failure_result(tmp_path, monkeypatch):
    """An unknown execution_method surfaces as a failed execution, not a fallback run."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"execution": {"execution_method": "unknown_method"}})

    with patch(
        "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
    ) as mock_spawn:
        result = await execute_code("print('ok')", "readonly", "test")

    mock_spawn.assert_not_awaited()
    assert result.success is False
    assert result.execution_method_used == "subprocess"
    assert "unknown_method" in result.error_message


# ---------------------------------------------------------------------------
# Execution mode reaches the sandbox
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["readonly", "readwrite"])
async def test_execution_mode_exported_to_sandbox_env(tmp_path, monkeypatch, mode):
    """The declared mode is a runtime property of the subprocess, not just a
    pre-execution gate: the wrapper, osprey.runtime and the connectors all read
    ``OSPREY_EXECUTION_MODE`` to refuse writes in a readonly run."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0

    with patch(
        "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ) as mock_spawn:
        await execute_code("print(42)", mode, "test")

    assert mock_spawn.call_args.kwargs["env"]["OSPREY_EXECUTION_MODE"] == mode


async def test_wrapper_built_with_execution_mode(tmp_path, monkeypatch):
    """The wrapper is told the mode so it can emit the readonly guard."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0

    seen = {}
    from osprey.services.python_executor.execution import wrapper as wrapper_module

    real_init = wrapper_module.ExecutionWrapper.__init__

    def spy_init(self, *args, **kwargs):
        seen.update(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(wrapper_module.ExecutionWrapper, "__init__", spy_init)

    with patch(
        "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ):
        await execute_code("print(42)", "readonly", "test")

    assert seen.get("execution_mode") == "readonly"


# ---------------------------------------------------------------------------
# A sandbox that finished its script but never exited
# ---------------------------------------------------------------------------
#
# The wrapper persists the execution record before the interpreter shuts down.
# A child that then wedges in a library's shutdown hook is killed at the
# timeout — but its script ran, and may have written to the machine. That run
# is reported from its record, never as an empty-handed timeout.


def _hanging_proc():
    async def never_returns():
        await asyncio.sleep(3600)

    proc = AsyncMock()
    proc.communicate = AsyncMock(side_effect=never_returns)
    proc.returncode = None
    return proc


async def test_completed_script_is_reported_when_the_sandbox_fails_to_exit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"python_executor": {"execution_timeout_seconds": 1}})
    folder = tmp_path / "exec"
    folder.mkdir()
    monkeypatch.setattr(
        "osprey.mcp_server.python_executor.executor._create_execution_folder", lambda: folder
    )
    (folder / "execution_metadata.json").write_text(
        json.dumps({"success": True, "stdout": "Before: 0.0\nAfter: 0.1\n", "stderr": ""})
    )

    proc = _hanging_proc()
    with patch(
        "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=proc,
    ):
        result = await execute_code("print('x')", "readwrite", "test")

    proc.kill.assert_called_once()
    assert result.success is True
    assert result.failure_kind is None
    assert result.stdout == "Before: 0.0\nAfter: 0.1\n"
    assert "did not exit" in result.stderr


async def test_a_script_still_running_at_the_timeout_is_a_timeout(tmp_path, monkeypatch):
    """No record on disk means the script itself never finished: the kill is the verdict."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"python_executor": {"execution_timeout_seconds": 1}})
    folder = tmp_path / "exec"
    folder.mkdir()
    monkeypatch.setattr(
        "osprey.mcp_server.python_executor.executor._create_execution_folder", lambda: folder
    )

    proc = _hanging_proc()
    with patch(
        "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=proc,
    ):
        result = await execute_code("while True: pass", "readonly", "test")

    proc.kill.assert_called_once()
    assert result.success is False
    assert result.failure_kind == "timeout"
    assert result.stdout == ""
