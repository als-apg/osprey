"""Tests for hook debug configuration and API endpoints.

Covers:
- build_clean_env NOT propagating OSPREY_HOOK_DEBUG (hooks read config.yml directly)
- Lifespan setting OSPREY_CONFIG and resetting config cache
- osprey_hook_log._is_debug_enabled fallback to config.yml
- Hook debug API endpoints (debug-status, debug-log)
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
import yaml

from osprey.interfaces.web_terminal.operator_session import build_clean_env

# ---------------------------------------------------------------------------
# build_clean_env — OSPREY_HOOK_DEBUG is NOT propagated
# ---------------------------------------------------------------------------


class TestBuildCleanEnvHookDebug:
    def test_does_not_set_debug_from_config(self, tmp_path, monkeypatch):
        """config with hooks.debug: true should NOT set OSPREY_HOOK_DEBUG in env."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": True}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path))
        assert "OSPREY_HOOK_DEBUG" not in env

    def test_no_debug_when_false(self, tmp_path, monkeypatch):
        """config with hooks.debug: false -> no env var."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": False}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path))
        assert "OSPREY_HOOK_DEBUG" not in env

    def test_no_debug_when_no_hooks_section(self, tmp_path, monkeypatch):
        """config without hooks section -> no env var."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"web_terminal": {"port": 8087}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.delenv("OSPREY_HOOK_DEBUG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path))
        assert "OSPREY_HOOK_DEBUG" not in env

    def test_manual_env_var_still_passes_through(self, tmp_path, monkeypatch):
        """Manually set OSPREY_HOOK_DEBUG passes through (not stripped)."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": False}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.setenv("OSPREY_HOOK_DEBUG", "1")

        env = build_clean_env(project_cwd=str(tmp_path))
        assert env["OSPREY_HOOK_DEBUG"] == "1"


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

        # Simulate stale cache from web_cmd.py pre-lifespan call
        from osprey.utils import config as config_module
        from osprey.utils.workspace import reset_config_cache

        config_module._config_cache["stale_path"] = "stale_value"

        from osprey.interfaces.web_terminal.app import create_app

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path / "ws")},
        ):
            app = create_app(shell_command="echo", project_dir=str(tmp_path))
            from fastapi.testclient import TestClient

            with TestClient(app):
                # Stale cache entry should have been cleared by lifespan
                assert "stale_path" not in config_module._config_cache

        # Clean up
        reset_config_cache()
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    def test_lifespan_hooks_env_is_empty(self, tmp_path, monkeypatch):
        """Verify hooks_env stays empty (no OSPREY_HOOK_DEBUG propagation)."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": True}}))
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.interfaces.web_terminal.app import create_app
        from osprey.utils.workspace import reset_config_cache

        reset_config_cache()

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path / "ws")},
        ):
            app = create_app(shell_command="echo", project_dir=str(tmp_path))
            from fastapi.testclient import TestClient

            with TestClient(app):
                assert app.state.hooks_env == {}

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

        # Delete config file -- second call should still return True from cache
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


# ---------------------------------------------------------------------------
# Hook Debug API Endpoints
# ---------------------------------------------------------------------------


class TestHookDebugEndpoints:
    @pytest.fixture()
    def _app(self, tmp_path, monkeypatch):
        """Create a test app with config.yml in tmp_path."""
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        from osprey.interfaces.web_terminal.app import create_app
        from osprey.utils.workspace import reset_config_cache

        reset_config_cache()

        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": True}}))

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path / "ws")},
        ):
            app = create_app(
                config_path=str(config_file),
                shell_command="echo",
                project_dir=str(tmp_path),
            )
            yield app

        reset_config_cache()
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    def test_hook_debug_status_returns_enabled(self, _app):
        from fastapi.testclient import TestClient

        with TestClient(_app) as client:
            resp = client.get("/api/hooks/debug-status")
            assert resp.status_code == 200
            assert resp.json()["enabled"] is True

    def test_hook_debug_status_returns_disabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        from osprey.interfaces.web_terminal.app import create_app
        from osprey.utils.workspace import reset_config_cache

        reset_config_cache()

        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": False}}))

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path / "ws")},
        ):
            app = create_app(
                config_path=str(config_file),
                shell_command="echo",
                project_dir=str(tmp_path),
            )
            from fastapi.testclient import TestClient

            with TestClient(app) as client:
                resp = client.get("/api/hooks/debug-status")
                assert resp.status_code == 200
                assert resp.json()["enabled"] is False

        reset_config_cache()
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

    def test_hook_debug_log_returns_entries(self, _app, tmp_path):
        # Write some JSONL log entries
        log_dir = tmp_path / ".claude" / "hooks"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "hook_debug.jsonl"
        entries = [
            {"ts": "2026-03-02T10:00:00Z", "hook": "PreToolUse", "tool": "Bash", "status": "allowed"},
            {"ts": "2026-03-02T10:00:01Z", "hook": "PreToolUse", "tool": "Write", "status": "blocked", "detail": "safety check"},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries))

        from fastapi.testclient import TestClient

        with TestClient(_app) as client:
            resp = client.get("/api/hooks/debug-log?limit=50")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["entries"]) == 2
            # Most recent first (reversed)
            assert data["entries"][0]["tool"] == "Write"
            assert data["entries"][1]["tool"] == "Bash"

    def test_hook_debug_log_empty_when_no_file(self, _app):
        from fastapi.testclient import TestClient

        with TestClient(_app) as client:
            resp = client.get("/api/hooks/debug-log")
            assert resp.status_code == 200
            assert resp.json()["entries"] == []

    def test_hook_debug_toggle_via_config_patch(self, _app, tmp_path):
        from fastapi.testclient import TestClient

        with TestClient(_app) as client:
            # Toggle debug off
            resp = client.patch(
                "/api/config",
                json={"updates": {"hooks.debug": False}},
            )
            assert resp.status_code == 200

            # Verify it took effect
            resp = client.get("/api/hooks/debug-status")
            assert resp.json()["enabled"] is False

            # Toggle back on
            resp = client.patch(
                "/api/config",
                json={"updates": {"hooks.debug": True}},
            )
            assert resp.status_code == 200

            resp = client.get("/api/hooks/debug-status")
            assert resp.json()["enabled"] is True
