"""Tests for Channel Finder server launcher registration."""

from __future__ import annotations

from unittest.mock import patch


class TestChannelFinderLauncher:
    """Tests for the channel finder server launcher functions."""

    def test_config_reader_defaults(self):
        """Config reader returns default host/port when section is empty."""
        from osprey.infrastructure.server_launcher import _channel_finder_config

        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={},
        ):
            host, port = _channel_finder_config()
        assert host == "127.0.0.1"
        assert port == 8092

    def test_config_reader_custom(self):
        """Config reader returns custom host/port from config."""
        from osprey.infrastructure.server_launcher import _channel_finder_config

        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={
                "channel_finder": {"web": {"host": "0.0.0.0", "port": 9999}},
            },
        ):
            host, port = _channel_finder_config()
        assert host == "0.0.0.0"
        assert port == 9999

    def test_auto_launch_false_when_no_section(self):
        """Auto-launch returns False when no channel_finder config section."""
        from osprey.infrastructure.server_launcher import _channel_finder_auto_launch

        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={},
        ):
            assert _channel_finder_auto_launch() is False

    def test_auto_launch_true_when_section_exists(self):
        """Auto-launch returns True when channel_finder section exists."""
        from osprey.infrastructure.server_launcher import _channel_finder_auto_launch

        with patch(
            "osprey.infrastructure.server_launcher.load_osprey_config",
            return_value={"channel_finder": {"pipeline_mode": "in_context"}},
        ):
            assert _channel_finder_auto_launch() is True

    def test_ensure_function_exists(self):
        """ensure_channel_finder_server() is importable."""
        from osprey.infrastructure.server_launcher import ensure_channel_finder_server

        assert callable(ensure_channel_finder_server)
