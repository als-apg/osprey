"""Tests for OSPREY_HOOK_DEBUG propagation through the full chain.

Covers:
- build_clean_env reading hooks.debug from config
- Lifespan setting OSPREY_CONFIG and resetting config cache
- osprey_hook_log._is_debug_enabled fallback to config.yml
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from osprey.interfaces.web_terminal.operator_session import build_clean_env


# ---------------------------------------------------------------------------
# build_clean_env — hooks.debug propagation
# ---------------------------------------------------------------------------


class TestBuildCleanEnvHookDebug:
    def test_sets_debug_from_config(self, tmp_path, monkeypatch):
        """config with hooks.debug: true → OSPREY_HOOK_DEBUG=1 in env."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": True}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path))
        assert env.get("OSPREY_HOOK_DEBUG") == "1"

    def test_skips_debug_when_false(self, tmp_path, monkeypatch):
        """config with hooks.debug: false → no env var."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": False}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path))
        assert "OSPREY_HOOK_DEBUG" not in env

    def test_skips_debug_when_no_hooks_section(self, tmp_path, monkeypatch):
        """config without hooks section → no env var."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"web_terminal": {"port": 8087}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path))
        assert "OSPREY_HOOK_DEBUG" not in env

    def test_respects_existing_env_var(self, tmp_path, monkeypatch):
        """Existing OSPREY_HOOK_DEBUG env var is not overridden."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": False}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")

        env = build_clean_env(project_cwd=str(tmp_path))
        assert env["OSPREY_HOOK_DEBUG"] == "1"

    def test_uses_osprey_config_env_var(self, tmp_path, monkeypatch):
        """OSPREY_CONFIG env var takes precedence over project_cwd for config lookup."""
        custom_config = tmp_path / "custom" / "config.yml"
        custom_config.parent.mkdir()
        custom_config.write_text(yaml.dump({"hooks": {"debug": True}}))

        project_config = tmp_path / "project" / "config.yml"
        project_config.parent.mkdir()
        project_config.write_text(yaml.dump({"hooks": {"debug": False}}))

        monkeypatch.setenv("OSPREY_CONFIG", str(custom_config))
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path / "project"))
        assert env.get("OSPREY_HOOK_DEBUG") == "1"


# ---------------------------------------------------------------------------
# Lifespan — OSPREY_CONFIG and config cache reset
# ---------------------------------------------------------------------------


class TestLifespanOspreyConfig:
    def test_lifespan_sets_osprey_config_env(self, tmp_path, monkeypatch):
        """Verify OSPREY_CONFIG is set from project_cwd before launcher calls."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": True}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.interfaces.web_terminal.app import create_app

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path / "ws")},
        ):
            app = create_app(shell_command="echo", project_dir=str(tmp_path))
            from fastapi.testclient import TestClient

            with TestClient(app):
                assert os.environ.get("OSPREY_CONFIG") == str(config_file)

        # Clean up env var set by the lifespan
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    def test_lifespan_does_not_override_existing_osprey_config(self, tmp_path, monkeypatch):
        """Existing OSPREY_CONFIG is respected."""
        (tmp_path / "config.yml").write_text(yaml.dump({}))
        monkeypatch.setenv("OSPREY_CONFIG", "/custom/config.yml")

        from osprey.interfaces.web_terminal.app import create_app

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path / "ws")},
        ):
            app = create_app(shell_command="echo", project_dir=str(tmp_path))
            from fastapi.testclient import TestClient

            with TestClient(app):
                assert os.environ.get("OSPREY_CONFIG") == "/custom/config.yml"

    def test_config_cache_reset_after_osprey_config_set(self, tmp_path, monkeypatch):
        """Verify stale config cache is cleared so launchers see fresh config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": True}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.mcp_server.common import _config_cache, reset_config_cache

        # Simulate stale cache from web_cmd.py pre-lifespan call
        import osprey.mcp_server.common as common_mod

        common_mod._config_cache = {"stale": True}

        from osprey.interfaces.web_terminal.app import create_app

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path / "ws")},
        ):
            app = create_app(shell_command="echo", project_dir=str(tmp_path))
            from fastapi.testclient import TestClient

            with TestClient(app):
                # Cache should have been reset and re-populated (not stale)
                assert common_mod._config_cache is None or common_mod._config_cache != {
                    "stale": True
                }

        # Clean up
        reset_config_cache()
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    def test_lifespan_hooks_env_populated(self, tmp_path, monkeypatch):
        """Verify hooks_env gets OSPREY_HOOK_DEBUG=1 when config enables it."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": True}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.interfaces.web_terminal.app import create_app
        from osprey.mcp_server.common import reset_config_cache

        reset_config_cache()

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path / "ws")},
        ):
            app = create_app(shell_command="echo", project_dir=str(tmp_path))
            from fastapi.testclient import TestClient

            with TestClient(app):
                assert app.state.hooks_env.get("OSPREY_HOOK_DEBUG") == "1"

        # Clean up
        reset_config_cache()
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)


# ---------------------------------------------------------------------------
# osprey_hook_log._is_debug_enabled
# ---------------------------------------------------------------------------


class TestIsDebugEnabled:
    @pytest.fixture(autouse=True)
    def _reset_module_cache(self):
        """Reset the module-level cache between tests."""
        import osprey.templates.claude_code.claude.hooks.osprey_hook_log as hook_mod

        hook_mod._debug_from_config = None
        yield
        hook_mod._debug_from_config = None

    def test_env_var_returns_true(self, monkeypatch):
        monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")

        from osprey.templates.claude_code.claude.hooks.osprey_hook_log import _is_debug_enabled

        assert _is_debug_enabled({"cwd": "/tmp"}) is True

    def test_no_env_no_config_returns_false(self, monkeypatch):
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.templates.claude_code.claude.hooks.osprey_hook_log import _is_debug_enabled

        assert _is_debug_enabled({"cwd": "/nonexistent"}) is False

    def test_config_fallback_returns_true(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": True}}))
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.templates.claude_code.claude.hooks.osprey_hook_log import _is_debug_enabled

        assert _is_debug_enabled({"cwd": str(tmp_path)}) is True

    def test_config_fallback_false(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": False}}))
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.templates.claude_code.claude.hooks.osprey_hook_log import _is_debug_enabled

        assert _is_debug_enabled({"cwd": str(tmp_path)}) is False

    def test_caches_result(self, tmp_path, monkeypatch):
        """Second call doesn't re-read file."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": True}}))
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        import osprey.templates.claude_code.claude.hooks.osprey_hook_log as hook_mod

        hook_input = {"cwd": str(tmp_path)}

        # First call reads config
        assert hook_mod._is_debug_enabled(hook_input) is True
        assert hook_mod._debug_from_config is True

        # Delete config file — second call should still return True from cache
        config_file.unlink()
        assert hook_mod._is_debug_enabled(hook_input) is True

    def test_uses_osprey_config_env_var(self, tmp_path, monkeypatch):
        """OSPREY_CONFIG env var overrides cwd-based config path."""
        custom_config = tmp_path / "custom_config.yml"
        custom_config.write_text(yaml.dump({"hooks": {"debug": True}}))
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
        monkeypatch.setenv("OSPREY_CONFIG", str(custom_config))

        from osprey.templates.claude_code.claude.hooks.osprey_hook_log import _is_debug_enabled

        # cwd has no config.yml, but OSPREY_CONFIG points to one
        assert _is_debug_enabled({"cwd": "/nonexistent"}) is True

    def test_empty_cwd_returns_false(self, monkeypatch):
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.templates.claude_code.claude.hooks.osprey_hook_log import _is_debug_enabled

        assert _is_debug_enabled({}) is False
