"""Tests for data-driven web server launcher (registry.web + server_launcher)."""

from __future__ import annotations

from unittest.mock import patch

from osprey.registry.web import FRAMEWORK_WEB_SERVERS


class TestMakeConfigReader:
    """Tests for _make_config_reader using catalog entries."""

    def test_defaults_when_section_empty(self):
        """Config reader returns default host/port when config section is empty."""
        from osprey.infrastructure.server_launcher import _make_config_reader

        reader = _make_config_reader(FRAMEWORK_WEB_SERVERS["channel_finder"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={},
        ):
            host, port = reader()
        assert host == "127.0.0.1"
        assert port == 8092

    def test_custom_values_with_web_subkey(self):
        """Config reader navigates config_web_subkey correctly."""
        from osprey.infrastructure.server_launcher import _make_config_reader

        reader = _make_config_reader(FRAMEWORK_WEB_SERVERS["channel_finder"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={
                "channel_finder": {"web": {"host": "0.0.0.0", "port": 9999}},
            },
        ):
            host, port = reader()
        assert host == "0.0.0.0"
        assert port == 9999

    def test_flat_config_key(self):
        """Config reader works for servers without config_web_subkey."""
        from osprey.infrastructure.server_launcher import _make_config_reader

        reader = _make_config_reader(FRAMEWORK_WEB_SERVERS["artifact"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={"artifact_server": {"host": "10.0.0.1", "port": 7777}},
        ):
            host, port = reader()
        assert host == "10.0.0.1"
        assert port == 7777


class TestMakeAutoLaunchChecker:
    """Tests for _make_auto_launch_checker using catalog entries."""

    def test_require_section_missing(self):
        """Auto-launch returns False when require_section=True and section is absent."""
        from osprey.infrastructure.server_launcher import _make_auto_launch_checker

        checker = _make_auto_launch_checker(FRAMEWORK_WEB_SERVERS["channel_finder"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={},
        ):
            assert checker() is False

    def test_require_section_present(self):
        """Auto-launch returns True when section exists (default auto_launch=True)."""
        from osprey.infrastructure.server_launcher import _make_auto_launch_checker

        checker = _make_auto_launch_checker(FRAMEWORK_WEB_SERVERS["channel_finder"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={"channel_finder": {"pipeline_mode": "in_context"}},
        ):
            assert checker() is True

    def test_no_require_section(self):
        """Auto-launch returns True even when section is empty if require_section=False."""
        from osprey.infrastructure.server_launcher import _make_auto_launch_checker

        checker = _make_auto_launch_checker(FRAMEWORK_WEB_SERVERS["artifact"])
        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={},
        ):
            assert checker() is True


class TestBackwardCompatAliases:
    """Named ensure_* functions remain importable."""

    def test_ensure_channel_finder_server_exists(self):
        from osprey.infrastructure.server_launcher import ensure_channel_finder_server

        assert callable(ensure_channel_finder_server)

    def test_ensure_artifact_server_exists(self):
        from osprey.infrastructure.server_launcher import ensure_artifact_server

        assert callable(ensure_artifact_server)

    def test_ensure_ariel_server_exists(self):
        from osprey.infrastructure.server_launcher import ensure_ariel_server

        assert callable(ensure_ariel_server)
