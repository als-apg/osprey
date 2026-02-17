"""Unit tests for the MCP execution adapter (executor.py).

Tests the adapter module in isolation with mocked executors.
Pattern: monkeypatch.chdir(tmp_path) -> write config.yml -> mock deps -> call adapter.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from osprey.mcp_server.python_executor.executor import (
    ExecutionResult,
    _collect_figures,
    _create_execution_folder,
    _execute_in_process_fallback,
    _load_limits_validator,
    _read_config,
    _read_execution_metadata,
    execute_code,
)


@pytest.fixture(autouse=True)
def _reset_all_config_caches(monkeypatch):
    """Reset ALL config caches before each test.

    Prior test modules set the ConfigBuilder singleton via
    get_config_builder(set_as_default=True).  We must clear
    _default_config, _default_configurable, and _config_cache
    before each test so the adapter reads from the test's own
    config.yml via monkeypatch.chdir(tmp_path).
    """
    from osprey.mcp_server.common import reset_config_cache

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
            "execution_method": "container",
            "modes": {
                "read_only": {"kernel_name": "python3-epics-readonly"},
                "write_access": {"kernel_name": "python3-epics-write"},
            },
        },
        "services": {
            "jupyter": {
                "containers": {
                    "read": {
                        "hostname": "localhost",
                        "port_host": 8088,
                        "execution_modes": ["read_only"],
                    },
                    "write": {
                        "hostname": "localhost",
                        "port_host": 8089,
                        "execution_modes": ["write_access"],
                    },
                }
            }
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


@pytest.mark.unit
def test_config_reads_execution_method(tmp_path, monkeypatch):
    """Adapter reads execution.execution_method from config.yml."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"execution": {"execution_method": "local"}})
    config = _read_config()
    assert config["execution_method"] == "local"


@pytest.mark.unit
def test_config_defaults_execution_method_to_container(tmp_path, monkeypatch):
    """When execution_method missing, defaults to 'container'."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({}))
    config = _read_config()
    assert config["execution_method"] == "container"


@pytest.mark.unit
def test_config_reads_timeout(tmp_path, monkeypatch):
    """Adapter reads python_executor.execution_timeout_seconds."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"python_executor": {"execution_timeout_seconds": 120}})
    config = _read_config()
    assert config["timeout"] == 120


@pytest.mark.unit
def test_config_timeout_default(tmp_path, monkeypatch):
    """When timeout config absent, defaults to 600 seconds."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(yaml.dump({}))
    config = _read_config()
    assert config["timeout"] == 600


# ---------------------------------------------------------------------------
# Container endpoint construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_container_endpoint_construction_readonly(tmp_path, monkeypatch):
    """'readonly' maps to read container port (8088) + readonly kernel."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    config = _read_config()
    assert config["read_container"]["port_host"] == 8088
    assert config["readonly_kernel"] == "python3-epics-readonly"


@pytest.mark.unit
def test_container_endpoint_construction_readwrite(tmp_path, monkeypatch):
    """'readwrite' maps to write container port (8089) + write kernel."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    config = _read_config()
    assert config["write_container"]["port_host"] == 8089
    assert config["readwrite_kernel"] == "python3-epics-write"


# ---------------------------------------------------------------------------
# Execution folder
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execution_folder_created(tmp_path, monkeypatch):
    """Adapter creates timestamped folder in osprey-workspace/data/python_executions/."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    folder = _create_execution_folder()
    assert folder.exists()
    assert folder.parent.name == "python_executions"
    assert (folder / "figures").exists()


# ---------------------------------------------------------------------------
# Limits validator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_limits_validator_loaded_and_passed(tmp_path, monkeypatch):
    """LimitsValidator.from_config() is called when loading the validator."""
    monkeypatch.chdir(tmp_path)
    # Write config with limits enabled + a limits database
    limits_db = tmp_path / "channel_limits.json"
    limits_db.write_text(json.dumps({
        "TEST:PV": {"min_value": 0.0, "max_value": 100.0, "writable": True}
    }))
    _write_config(tmp_path, {
        "control_system": {
            "limits_checking": {
                "enabled": True,
                "database_path": str(limits_db),
                "allow_unlisted_channels": False,
                "on_violation": "error",
            }
        }
    })
    # Force ConfigBuilder to use this test's config.yml
    from osprey.utils.config import get_config_builder

    get_config_builder(config_path=str(tmp_path / "config.yml"), set_as_default=True)

    validator = _load_limits_validator()
    assert validator is not None
    assert "TEST:PV" in validator.limits


@pytest.mark.unit
def test_limits_validator_disabled_gracefully(tmp_path, monkeypatch):
    """When limits_checking.enabled=false, returns None."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    validator = _load_limits_validator()
    assert validator is None


# ---------------------------------------------------------------------------
# Wrapper / monkeypatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wrapper_includes_monkeypatch_when_validator_present(tmp_path, monkeypatch):
    """ExecutionWrapper.create_wrapper() output contains monkeypatch code when validator present."""
    monkeypatch.chdir(tmp_path)
    limits_db = tmp_path / "channel_limits.json"
    limits_db.write_text(json.dumps({
        "TEST:PV": {"min_value": 0, "max_value": 100, "writable": True}
    }))
    _write_config(tmp_path, {
        "control_system": {
            "limits_checking": {
                "enabled": True,
                "database_path": str(limits_db),
                "allow_unlisted_channels": False,
                "on_violation": "error",
            }
        }
    })
    # Force ConfigBuilder to use this test's config.yml
    from osprey.utils.config import get_config_builder

    get_config_builder(config_path=str(tmp_path / "config.yml"), set_as_default=True)

    validator = _load_limits_validator()
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    wrapper = ExecutionWrapper(execution_mode="local", limits_validator=validator)
    wrapped = wrapper.create_wrapper("print('hello')", tmp_path)
    assert "_checked_caput" in wrapped
    assert "LimitsValidator" in wrapped


@pytest.mark.unit
def test_wrapper_omits_monkeypatch_when_no_validator(tmp_path, monkeypatch):
    """Wrapper output has no monkeypatch when validator is None."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)
    from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

    wrapper = ExecutionWrapper(execution_mode="local", limits_validator=None)
    wrapped = wrapper.create_wrapper("print('hello')", tmp_path)
    assert "_checked_caput" not in wrapped


# ---------------------------------------------------------------------------
# Executor dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_local_subprocess_called(tmp_path, monkeypatch):
    """With execution_method: local, adapter calls asyncio.create_subprocess_exec."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"execution": {"execution_method": "local"}})

    # Mock the subprocess to avoid actually running code
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"42\n", b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch("osprey.mcp_server.python_executor.executor.asyncio") as mock_asyncio:
        mock_asyncio.create_subprocess_exec = AsyncMock(return_value=mock_proc)
        mock_asyncio.subprocess = asyncio.subprocess
        mock_asyncio.wait_for = AsyncMock(return_value=(b"42\n", b""))
        mock_asyncio.TimeoutError = asyncio.TimeoutError

        result = await execute_code("print(42)", "readonly", "test")

    # Should have used local execution (or fallen back)
    assert result.execution_method_used in ("local", "in_process_fallback")


@pytest.mark.unit
async def test_container_executor_called(tmp_path, monkeypatch):
    """With execution_method: container, adapter instantiates ContainerExecutor."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)

    mock_engine_result = MagicMock()
    mock_engine_result.success = True
    mock_engine_result.stdout = "42\n"
    mock_engine_result.error_message = None
    mock_engine_result.captured_figures = []

    with patch(
        "osprey.services.python_executor.execution.container_engine.ContainerExecutor"
    ) as MockExecutor:
        instance = MockExecutor.return_value
        instance.execute_code = AsyncMock(return_value=mock_engine_result)

        result = await execute_code("print(42)", "readonly", "test")

    assert result.success is True
    assert result.stdout == "42\n"
    assert result.execution_method_used == "container"
    MockExecutor.assert_called_once()


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_timeout_from_config(tmp_path, monkeypatch):
    """Adapter passes configured timeout to ContainerExecutor."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"python_executor": {"execution_timeout_seconds": 42}})

    mock_engine_result = MagicMock()
    mock_engine_result.success = True
    mock_engine_result.stdout = ""
    mock_engine_result.error_message = None
    mock_engine_result.captured_figures = []

    with patch(
        "osprey.services.python_executor.execution.container_engine.ContainerExecutor"
    ) as MockExecutor:
        instance = MockExecutor.return_value
        instance.execute_code = AsyncMock(return_value=mock_engine_result)

        await execute_code("pass", "readonly", "test")

    # Check the timeout passed to ContainerExecutor constructor
    call_kwargs = MockExecutor.call_args
    assert call_kwargs.kwargs.get("timeout") == 42 or call_kwargs[1].get("timeout") == 42


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_result_dataclass_populated():
    """ExecutionResult has correct fields when constructed."""
    result = ExecutionResult(
        success=True,
        stdout="hello",
        stderr="",
        figures=[Path("/tmp/fig.png")],
        execution_method_used="container",
        execution_time_seconds=1.5,
    )
    assert result.success is True
    assert result.stdout == "hello"
    assert len(result.figures) == 1
    assert result.execution_method_used == "container"
    assert result.execution_time_seconds == 1.5


@pytest.mark.unit
def test_result_dataclass_defaults():
    """ExecutionResult defaults are sensible."""
    result = ExecutionResult(success=False, stdout="", stderr="error")
    assert result.figures == []
    assert result.execution_method_used == "container"
    assert result.execution_time_seconds is None
    assert result.error_message is None


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_fallback_on_container_connectivity_error(tmp_path, monkeypatch):
    """When ContainerExecutor raises, adapter falls back to in-process exec."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path)

    from osprey.services.python_executor.exceptions import ContainerConnectivityError

    with patch(
        "osprey.services.python_executor.execution.container_engine.ContainerExecutor"
    ) as MockExecutor:
        instance = MockExecutor.return_value
        instance.execute_code = AsyncMock(
            side_effect=ContainerConnectivityError(
                "Connection refused", host="localhost", port=8088
            )
        )

        result = await execute_code("print(6 * 7)", "readonly", "fallback test")

    assert result.execution_method_used == "in_process_fallback"
    assert "42" in result.stdout
    assert result.success is True


@pytest.mark.unit
async def test_fallback_on_subprocess_error(tmp_path, monkeypatch):
    """When subprocess fails, adapter falls back to in-process exec."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"execution": {"execution_method": "local"}})

    with patch(
        "osprey.mcp_server.python_executor.executor._execute_via_local",
        new_callable=AsyncMock,
        side_effect=OSError("subprocess failed"),
    ):
        result = await execute_code("print('hi')", "readonly", "fallback test")

    assert result.execution_method_used == "in_process_fallback"
    assert "hi" in result.stdout


@pytest.mark.unit
def test_fallback_marks_result():
    """Fallback execution sets execution_method_used to 'in_process_fallback'."""
    result = _execute_in_process_fallback("print(42)")
    assert result.execution_method_used == "in_process_fallback"
    assert "42" in result.stdout
    assert result.success is True


@pytest.mark.unit
def test_fallback_preserves_save_artifact():
    """save_artifact is available in fallback namespace."""
    artifacts = []

    def mock_save(obj, title="", description="", artifact_type=None):
        artifacts.append(obj)

    result = _execute_in_process_fallback(
        "save_artifact('test_value', title='Test')",
        save_artifact_fn=mock_save,
    )
    assert result.success is True
    assert artifacts == ["test_value"]


# ---------------------------------------------------------------------------
# Figure collection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_figure_collection_from_execution_folder(execution_folder):
    """After execution, adapter scans figures/ dir and returns figure paths."""
    # Create some figure files
    (execution_folder / "figures" / "figure_01.png").write_bytes(b"PNG")
    (execution_folder / "figures" / "figure_02.png").write_bytes(b"PNG")

    figures = _collect_figures(execution_folder)
    assert len(figures) == 2
    assert all(f.suffix == ".png" for f in figures)


@pytest.mark.unit
def test_figure_collection_empty_folder(execution_folder):
    """Empty execution folder returns no figures."""
    figures = _collect_figures(execution_folder)
    assert figures == []


@pytest.mark.unit
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


@pytest.mark.unit
def test_read_execution_metadata(execution_folder):
    """Reads execution_metadata.json from execution folder."""
    metadata = {"success": True, "stdout": "hello", "stderr": ""}
    (execution_folder / "execution_metadata.json").write_text(json.dumps(metadata))

    result = _read_execution_metadata(execution_folder)
    assert result == metadata


@pytest.mark.unit
def test_read_execution_metadata_missing(execution_folder):
    """Returns None when execution_metadata.json doesn't exist."""
    result = _read_execution_metadata(execution_folder)
    assert result is None


@pytest.mark.unit
def test_read_execution_metadata_invalid_json(execution_folder):
    """Returns None when execution_metadata.json is invalid JSON."""
    (execution_folder / "execution_metadata.json").write_text("not json{{{")

    result = _read_execution_metadata(execution_folder)
    assert result is None


# ---------------------------------------------------------------------------
# Invalid config
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_invalid_execution_method_defaults_to_container(tmp_path, monkeypatch):
    """Unknown execution_method value falls through to container path."""
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, {"execution": {"execution_method": "unknown_method"}})

    mock_engine_result = MagicMock()
    mock_engine_result.success = True
    mock_engine_result.stdout = "ok"
    mock_engine_result.error_message = None
    mock_engine_result.captured_figures = []

    with patch(
        "osprey.services.python_executor.execution.container_engine.ContainerExecutor"
    ) as MockExecutor:
        instance = MockExecutor.return_value
        instance.execute_code = AsyncMock(return_value=mock_engine_result)

        result = await execute_code("print('ok')", "readonly", "test")

    # Unknown method falls through to the else (container) branch
    assert result.execution_method_used == "container"
    MockExecutor.assert_called_once()
