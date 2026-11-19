"""Tests for MCP server shared utilities and startup wiring.

Covers: config loading, make_error format, and the config bridge
(create_server installs the process config).
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_config_builder_globals():
    """Restore the config module's process state so each test starts clean.

    loaded_or_empty() reads the process config, which the lazy default loads
    and caches; the process state is put back between tests to prevent
    cross-contamination.
    """
    import osprey_connectors.config as cfg_mod

    orig_default = cfg_mod._installed
    orig_configurable = cfg_mod._last_lazy
    orig_cache = cfg_mod._loads.copy()

    yield

    cfg_mod._installed = orig_default
    cfg_mod._last_lazy = orig_configurable
    cfg_mod._loads = orig_cache


async def test_loaded_or_empty(tmp_path, monkeypatch):
    """loaded_or_empty reads config.yml from working directory."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "control_system:\n  type: virtual_accelerator\n  connector:\n    virtual_accelerator:\n      serving: in_process\n  writes_enabled: true\n"
    )

    from osprey.utils.config_paths import loaded_or_empty

    config = loaded_or_empty()
    assert config["control_system"]["type"] == "virtual_accelerator"
    assert config["control_system"]["writes_enabled"] is True


async def test_loaded_or_empty_from_env(tmp_path, monkeypatch):
    """loaded_or_empty reads from OSPREY_CONFIG env var."""
    config_file = tmp_path / "custom_config.yml"
    config_file.write_text("control_system:\n  type: epics\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))

    from osprey.utils.config_paths import loaded_or_empty

    config = loaded_or_empty()
    assert config["control_system"]["type"] == "epics"


async def test_loaded_or_empty_missing_file(tmp_path, monkeypatch):
    """loaded_or_empty returns empty dict when config.yml is missing."""
    monkeypatch.chdir(tmp_path)
    # Ensure no OSPREY_CONFIG env var
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    from osprey.utils.config_paths import loaded_or_empty

    config = loaded_or_empty()
    assert isinstance(config, dict)
    assert config == {}


async def test_loaded_or_empty_resolves_env_vars(tmp_path, monkeypatch):
    """loaded_or_empty resolves ${VAR:-default} env var placeholders."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "control_system:\n  type: ${CS_TYPE:-epics}\n  host: ${CS_HOST:-localhost}\n"
    )

    # CS_TYPE not set → should use default "epics"
    monkeypatch.delenv("CS_TYPE", raising=False)
    # CS_HOST set → should resolve
    monkeypatch.setenv("CS_HOST", "epics-server.example.com")

    from osprey.utils.config_paths import loaded_or_empty

    config = loaded_or_empty()
    assert config["control_system"]["type"] == "epics"
    assert config["control_system"]["host"] == "epics-server.example.com"


async def test_make_error_format():
    """make_error raises fastmcp ToolError with the standard envelope as message."""
    import json

    import pytest as _pytest
    from fastmcp.exceptions import ToolError

    from osprey.mcp_server.errors import make_error

    with _pytest.raises(ToolError) as exc_info:
        make_error(
            error_type="test_error",
            error_message="something went wrong",
            suggestions=["try again", "check logs"],
        )
    envelope = json.loads(str(exc_info.value))
    assert envelope["error"] is True
    assert envelope["error_type"] == "test_error"
    assert envelope["error_message"] == "something went wrong"
    assert len(envelope["suggestions"]) == 2


async def test_make_error_no_suggestions():
    """make_error with no suggestions raises with empty suggestions list."""
    import json

    import pytest as _pytest
    from fastmcp.exceptions import ToolError

    from osprey.mcp_server.errors import make_error

    with _pytest.raises(ToolError) as exc_info:
        make_error(
            error_type="simple_error",
            error_message="oops",
        )
    envelope = json.loads(str(exc_info.value))
    assert envelope["error"] is True
    assert envelope["suggestions"] == []


# ---------------------------------------------------------------------------
# Config bridge tests (create_server installs the config OSPREY_CONFIG names)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _reset_config_globals():
    """Restore the config module's process state so each test starts clean."""
    import osprey_connectors.config as cfg_mod

    orig_default = cfg_mod._installed
    orig_configurable = cfg_mod._last_lazy
    orig_cache = cfg_mod._loads.copy()

    yield

    cfg_mod._installed = orig_default
    cfg_mod._last_lazy = orig_configurable
    cfg_mod._loads = orig_cache


@pytest.mark.usefixtures("_reset_config_globals")
async def test_create_server_primes_config_builder(tmp_path, monkeypatch):
    """create_server() installs the process config when OSPREY_CONFIG is set."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "control_system:\n  type: virtual_accelerator\n  connector:\n    virtual_accelerator:\n      serving: in_process\n  writes_enabled: false\n"
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
    monkeypatch.chdir(tmp_path)

    from osprey.mcp_server.control_system.server import create_server

    create_server()

    from osprey_connectors.config import current_config

    assert current_config().get("control_system.type") == "virtual_accelerator"


@pytest.mark.usefixtures("_reset_config_globals")
async def test_create_server_works_without_osprey_config(tmp_path, monkeypatch):
    """create_server() still works when OSPREY_CONFIG is not set."""
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)

    # Provide a config.yml in cwd so the lazy load does not raise
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "control_system:\n  type: virtual_accelerator\n  connector:\n    virtual_accelerator:\n      serving: in_process\n"
    )

    from osprey.mcp_server.control_system.server import create_server

    # Should not raise
    server = create_server()
    assert server is not None
