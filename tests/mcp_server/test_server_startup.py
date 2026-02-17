"""Tests for MCP server shared utilities and DataContext.

Covers: config loading, make_error format, DataContext
save/list/get operations, config bridge (create_server primes ConfigBuilder).
"""

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_config_builder_globals():
    """Reset ConfigBuilder module globals so each test starts clean.

    load_osprey_config() now delegates to get_config_builder(), which uses
    its own per-path cache and default singleton.  Both must be cleared
    between tests to prevent cross-contamination.
    """
    import osprey.utils.config as cfg_mod

    orig_default = cfg_mod._default_config
    orig_configurable = cfg_mod._default_configurable
    orig_cache = cfg_mod._config_cache.copy()

    yield

    cfg_mod._default_config = orig_default
    cfg_mod._default_configurable = orig_configurable
    cfg_mod._config_cache = orig_cache


@pytest.mark.unit
async def test_load_osprey_config(tmp_path, monkeypatch):
    """load_osprey_config reads config.yml from working directory."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text("control_system:\n  type: mock\n  writes_enabled: true\n")

    from osprey.mcp_server.common import load_osprey_config

    config = load_osprey_config()
    assert config["control_system"]["type"] == "mock"
    assert config["control_system"]["writes_enabled"] is True


@pytest.mark.unit
async def test_load_osprey_config_from_env(tmp_path, monkeypatch):
    """load_osprey_config reads from OSPREY_CONFIG env var."""
    config_file = tmp_path / "custom_config.yml"
    config_file.write_text("control_system:\n  type: epics\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))

    from osprey.mcp_server.common import load_osprey_config

    config = load_osprey_config()
    assert config["control_system"]["type"] == "epics"


@pytest.mark.unit
async def test_load_osprey_config_missing_file(tmp_path, monkeypatch):
    """load_osprey_config returns empty dict when config.yml is missing."""
    monkeypatch.chdir(tmp_path)
    # Ensure no OSPREY_CONFIG env var
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    from osprey.mcp_server.common import load_osprey_config

    config = load_osprey_config()
    assert isinstance(config, dict)
    assert config == {}


@pytest.mark.unit
async def test_load_osprey_config_resolves_env_vars(tmp_path, monkeypatch):
    """load_osprey_config resolves ${VAR:-default} env var placeholders."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "control_system:\n"
        "  type: ${CS_TYPE:-mock}\n"
        "  host: ${CS_HOST:-localhost}\n"
    )

    # CS_TYPE not set → should use default "mock"
    monkeypatch.delenv("CS_TYPE", raising=False)
    # CS_HOST set → should resolve
    monkeypatch.setenv("CS_HOST", "epics-server.lbl.gov")

    from osprey.mcp_server.common import load_osprey_config

    config = load_osprey_config()
    assert config["control_system"]["type"] == "mock"
    assert config["control_system"]["host"] == "epics-server.lbl.gov"


@pytest.mark.unit
async def test_make_error_format():
    """make_error produces standard error JSON format."""
    from osprey.mcp_server.common import make_error

    error = make_error(
        error_type="test_error",
        error_message="something went wrong",
        suggestions=["try again", "check logs"],
    )
    assert error["error"] is True
    assert error["error_type"] == "test_error"
    assert error["error_message"] == "something went wrong"
    assert len(error["suggestions"]) == 2


@pytest.mark.unit
async def test_make_error_no_suggestions():
    """make_error with no suggestions returns empty list."""
    from osprey.mcp_server.common import make_error

    error = make_error(
        error_type="simple_error",
        error_message="oops",
    )
    assert error["error"] is True
    assert error["suggestions"] == []


# ---------------------------------------------------------------------------
# DataContext tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_data_context_save(tmp_path, monkeypatch):
    """DataContext.save() creates data file with _osprey_metadata and updates index."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.data_context import DataContext

    ctx = DataContext(workspace_root=tmp_path / "osprey-workspace")
    entry = ctx.save(
        tool="channel_read",
        data={"value": 42},
        description="test output",
        summary={"channels_read": 1},
        access_details={"data_structure": "test"},
        data_type="channel_values",
    )

    # Verify data file exists and has correct structure
    data_file = Path(entry.data_file)
    assert data_file.exists()
    content = json.loads(data_file.read_text())
    assert "_osprey_metadata" in content
    assert content["_osprey_metadata"]["tool"] == "channel_read"
    assert content["_osprey_metadata"]["description"] == "test output"
    assert content["data"]["value"] == 42

    # Verify index file
    index_file = tmp_path / "osprey-workspace" / "data_context.json"
    assert index_file.exists()
    index = json.loads(index_file.read_text())
    assert index["entry_count"] == 1
    assert index["entries"][0]["tool"] == "channel_read"


@pytest.mark.unit
async def test_data_context_creates_directory(tmp_path, monkeypatch):
    """DataContext.save() creates the data directory if missing."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.data_context import DataContext

    ctx = DataContext(workspace_root=tmp_path / "osprey-workspace")
    entry = ctx.save(
        tool="test_tool",
        data={"test": True},
        description="test",
        summary={},
        access_details={},
        data_type="test",
    )

    data_file = Path(entry.data_file)
    assert data_file.exists()
    assert (tmp_path / "osprey-workspace" / "data").is_dir()


@pytest.mark.unit
async def test_data_context_list_entries(tmp_path, monkeypatch):
    """DataContext.list_entries() returns all entries with filtering."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.data_context import DataContext

    ctx = DataContext(workspace_root=tmp_path / "osprey-workspace")
    ctx.save(
        tool="archiver_read",
        data={},
        description="entry 1",
        summary={},
        access_details={},
        data_type="timeseries",
    )
    ctx.save(
        tool="channel_read",
        data={},
        description="entry 2",
        summary={},
        access_details={},
        data_type="channel_values",
    )
    ctx.save(
        tool="archiver_read",
        data={},
        description="entry 3",
        summary={},
        access_details={},
        data_type="timeseries",
    )

    # No filter — all entries
    assert len(ctx.list_entries()) == 3

    # Filter by tool
    assert len(ctx.list_entries(tool_filter="archiver_read")) == 2
    assert len(ctx.list_entries(tool_filter="channel_read")) == 1

    # Filter by data type
    assert len(ctx.list_entries(data_type_filter="timeseries")) == 2

    # last_n
    assert len(ctx.list_entries(last_n=1)) == 1
    assert ctx.list_entries(last_n=1)[0].description == "entry 3"


@pytest.mark.unit
async def test_data_context_get_entry(tmp_path, monkeypatch):
    """DataContext.get_entry() looks up by ID."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.data_context import DataContext

    ctx = DataContext(workspace_root=tmp_path / "osprey-workspace")
    e1 = ctx.save(
        tool="t1", data={}, description="first", summary={}, access_details={}, data_type="test"
    )
    e2 = ctx.save(
        tool="t2", data={}, description="second", summary={}, access_details={}, data_type="test"
    )

    assert ctx.get_entry(e1.id).description == "first"
    assert ctx.get_entry(e2.id).description == "second"
    assert ctx.get_entry(999) is None


@pytest.mark.unit
async def test_data_context_to_tool_response(tmp_path, monkeypatch):
    """DataContextEntry.to_tool_response() returns the compact format."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.data_context import DataContext

    ctx = DataContext(workspace_root=tmp_path / "osprey-workspace")
    entry = ctx.save(
        tool="archiver_read",
        data={"big": "payload"},
        description="Test archiver data",
        summary={"channels_queried": 1, "total_rows": 100},
        access_details={"columns": ["SR:CURRENT:RB"]},
        data_type="timeseries",
    )

    resp = entry.to_tool_response()
    assert resp["status"] == "success"
    assert resp["context_entry_id"] == entry.id
    assert resp["description"] == "Test archiver data"
    assert resp["summary"]["channels_queried"] == 1
    assert "data_file" in resp
    assert "hint" in resp


@pytest.mark.unit
async def test_data_context_reload_from_disk(tmp_path, monkeypatch):
    """A new DataContext instance loads the index from a previous session."""
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.data_context import DataContext

    ws = tmp_path / "osprey-workspace"

    # First session
    ctx1 = DataContext(workspace_root=ws)
    ctx1.save(
        tool="t1",
        data={"a": 1},
        description="session 1",
        summary={},
        access_details={},
        data_type="test",
    )

    # Second session (new instance, same workspace)
    ctx2 = DataContext(workspace_root=ws)
    assert len(ctx2.list_entries()) == 1
    assert ctx2.list_entries()[0].description == "session 1"

    # New save in second session gets next ID
    e2 = ctx2.save(
        tool="t2",
        data={"b": 2},
        description="session 2",
        summary={},
        access_details={},
        data_type="test",
    )
    assert e2.id == 2
    assert len(ctx2.list_entries()) == 2


# ---------------------------------------------------------------------------
# Config bridge tests (create_server primes ConfigBuilder from OSPREY_CONFIG)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _reset_config_globals():
    """Reset ConfigBuilder module globals so each test starts clean."""
    import osprey.utils.config as cfg_mod

    orig_default = cfg_mod._default_config
    orig_configurable = cfg_mod._default_configurable
    orig_cache = cfg_mod._config_cache.copy()

    yield

    cfg_mod._default_config = orig_default
    cfg_mod._default_configurable = orig_configurable
    cfg_mod._config_cache = orig_cache


@pytest.mark.unit
@pytest.mark.usefixtures("_reset_config_globals")
async def test_create_server_primes_config_builder(tmp_path, monkeypatch):
    """create_server() primes ConfigBuilder when OSPREY_CONFIG is set."""
    config_file = tmp_path / "config.yml"
    config_file.write_text("control_system:\n  type: mock\n  writes_enabled: false\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.control_system.server import create_server

    create_server()

    from osprey.utils.config import get_config_builder

    builder = get_config_builder()
    assert builder.raw_config["control_system"]["type"] == "mock"


@pytest.mark.unit
@pytest.mark.usefixtures("_reset_config_globals")
async def test_create_server_works_without_osprey_config(tmp_path, monkeypatch):
    """create_server() still works when OSPREY_CONFIG is not set."""
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)

    # Provide a config.yml in cwd so ConfigBuilder doesn't raise
    config_file = tmp_path / "config.yml"
    config_file.write_text("control_system:\n  type: mock\n")

    from osprey.mcp_server.control_system.server import create_server

    # Should not raise
    server = create_server()
    assert server is not None
