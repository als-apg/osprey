"""Tests for MCP server shared utilities and DataContext.

Covers: config loading, make_error format, DataContext
save/list/get operations, config bridge (create_server primes ConfigBuilder).
"""

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

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    assert config["control_system"]["type"] == "mock"
    assert config["control_system"]["writes_enabled"] is True


@pytest.mark.unit
async def test_load_osprey_config_from_env(tmp_path, monkeypatch):
    """load_osprey_config reads from OSPREY_CONFIG env var."""
    config_file = tmp_path / "custom_config.yml"
    config_file.write_text("control_system:\n  type: epics\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    assert config["control_system"]["type"] == "epics"


@pytest.mark.unit
async def test_load_osprey_config_missing_file(tmp_path, monkeypatch):
    """load_osprey_config returns empty dict when config.yml is missing."""
    monkeypatch.chdir(tmp_path)
    # Ensure no OSPREY_CONFIG env var
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    assert isinstance(config, dict)
    assert config == {}


@pytest.mark.unit
async def test_load_osprey_config_resolves_env_vars(tmp_path, monkeypatch):
    """load_osprey_config resolves ${VAR:-default} env var placeholders."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "control_system:\n  type: ${CS_TYPE:-mock}\n  host: ${CS_HOST:-localhost}\n"
    )

    # CS_TYPE not set → should use default "mock"
    monkeypatch.delenv("CS_TYPE", raising=False)
    # CS_HOST set → should resolve
    monkeypatch.setenv("CS_HOST", "epics-server.lbl.gov")

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    assert config["control_system"]["type"] == "mock"
    assert config["control_system"]["host"] == "epics-server.lbl.gov"


@pytest.mark.unit
async def test_make_error_format():
    """make_error produces standard error JSON format."""
    from osprey.mcp_server.errors import make_error

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
    from osprey.mcp_server.errors import make_error

    error = make_error(
        error_type="simple_error",
        error_message="oops",
    )
    assert error["error"] is True
    assert error["suggestions"] == []


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
