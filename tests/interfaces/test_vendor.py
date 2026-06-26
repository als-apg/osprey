"""Tests for vendor helpers and offline-mode switching."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osprey.interfaces.vendor import asset_cdn_url, is_offline, vendor_url


class TestIsOffline:
    def test_defaults_to_false(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch("osprey.utils.workspace.load_osprey_config", return_value={}):
            assert is_offline() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_env_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("OSPREY_OFFLINE", val)
        assert is_offline() is True

    @pytest.mark.parametrize("val", ["0", "false", "FALSE", "no", "off"])
    def test_env_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("OSPREY_OFFLINE", val)
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"offline": True},
        ):
            # Env var wins: falsy env overrides config.yml's truthy setting
            assert is_offline() is False

    def test_config_yml_truthy_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"offline": True},
        ):
            assert is_offline() is True

    def test_empty_env_falls_through_to_config(self, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "")
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"offline": True},
        ):
            assert is_offline() is True

    def test_config_load_failure_returns_false(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            side_effect=RuntimeError("boom"),
        ):
            assert is_offline() is False


class TestAssetCdnUrl:
    def test_known_names(self):
        assert asset_cdn_url("xterm.js").startswith("https://cdn.jsdelivr.net/")
        assert asset_cdn_url("Plotly.js").startswith("https://cdn.plot.ly/")
        assert asset_cdn_url("KaTeX JS").endswith("katex.min.js")

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown vendor asset name"):
            asset_cdn_url("does-not-exist")


class TestVendorUrl:
    def test_offline_returns_local_path(self, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        assert vendor_url("xterm.js", "/static/vendor/xterm.min.js") == (
            "/static/vendor/xterm.min.js"
        )

    def test_online_returns_cdn_url(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch("osprey.utils.workspace.load_osprey_config", return_value={}):
            url = vendor_url("xterm.js", "/static/vendor/xterm.min.js")
        assert url.startswith("https://cdn.jsdelivr.net/")
        assert url.endswith("xterm.min.js")

    def test_online_unknown_name_raises(self, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch("osprey.utils.workspace.load_osprey_config", return_value={}):
            with pytest.raises(KeyError):
                vendor_url("nope", "/static/vendor/nope.js")

    def test_offline_unknown_name_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        assert vendor_url("nope", "/local/path.js") == "/local/path.js"


class TestWebTerminalRendering:
    """End-to-end: GET the web_terminal index route in both modes and
    assert the rendered HTML contains the expected src values."""

    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.web_terminal.app import create_app

        ws = tmp_path / "_agent_data"
        ws.mkdir()
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(ws)},
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                yield c

    def test_cdn_mode_uses_jsdelivr(self, client, monkeypatch):
        monkeypatch.delenv("OSPREY_OFFLINE", raising=False)
        with patch("osprey.utils.workspace.load_osprey_config", return_value={}):
            resp = client.get("/")
        assert resp.status_code == 200
        body = resp.text
        assert "cdn.jsdelivr.net" in body
        assert 'src="/static/vendor/xterm.min.js"' not in body

    def test_offline_mode_uses_local_paths(self, client, monkeypatch):
        monkeypatch.setenv("OSPREY_OFFLINE", "1")
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.text
        assert "/static/vendor/xterm.min.js" in body
        assert "cdn.jsdelivr.net" not in body
