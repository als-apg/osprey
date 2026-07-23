"""Tests for ConfigBuilder's ``load_env`` behavior.

The default-construction ``.env`` → ``os.environ`` passthrough is load-bearing:
it feeds the ``${VAR}`` references Claude Code expands in ``.mcp.json`` at MCP
server launch time. These tests pin both halves of the contract — default
construction still loads ``.env``, and ``load_env=False`` leaves the environment
untouched.
"""

from __future__ import annotations

from osprey.utils.config import ConfigBuilder

_CONFIG_YML = "project_root: /test/project\n"


def _write_project(tmp_path, env_body: str) -> str:
    """Write a minimal config.yml + .env into ``tmp_path`` and return the config path."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(_CONFIG_YML)
    (tmp_path / ".env").write_text(env_body)
    return str(config_file)


def test_default_construction_loads_dotenv(tmp_path, monkeypatch):
    """Default construction loads ``.env`` into ``os.environ`` — the load-bearing
    passthrough that ``.mcp.json`` ${VAR} refs depend on."""
    monkeypatch.delenv("OSPREY_LOAD_ENV_PROBE", raising=False)
    config_path = _write_project(tmp_path, "OSPREY_LOAD_ENV_PROBE=from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    ConfigBuilder(config_path)

    import os

    assert os.environ.get("OSPREY_LOAD_ENV_PROBE") == "from-dotenv"


def test_load_env_false_leaves_environ_untouched(tmp_path, monkeypatch):
    """``load_env=False`` skips the ``.env`` load entirely, so a variable defined
    only in ``.env`` never reaches ``os.environ``."""
    monkeypatch.delenv("OSPREY_LOAD_ENV_PROBE", raising=False)
    config_path = _write_project(tmp_path, "OSPREY_LOAD_ENV_PROBE=from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    ConfigBuilder(config_path, load_env=False)

    import os

    assert "OSPREY_LOAD_ENV_PROBE" not in os.environ


def test_load_env_false_does_not_override_existing_environ(tmp_path, monkeypatch):
    """With ``load_env=False`` an existing ``os.environ`` value is preserved even
    when ``.env`` would override it (default construction uses override=True)."""
    monkeypatch.setenv("OSPREY_LOAD_ENV_PROBE", "from-environ")
    config_path = _write_project(tmp_path, "OSPREY_LOAD_ENV_PROBE=from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    ConfigBuilder(config_path, load_env=False)

    import os

    assert os.environ["OSPREY_LOAD_ENV_PROBE"] == "from-environ"


def test_config_still_loads_with_load_env_false(tmp_path, monkeypatch):
    """Skipping ``.env`` loading must not affect config parsing itself."""
    config_path = _write_project(tmp_path, "OSPREY_LOAD_ENV_PROBE=from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    builder = ConfigBuilder(config_path, load_env=False)

    assert builder.raw_config["project_root"] == "/test/project"
