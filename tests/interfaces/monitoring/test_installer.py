"""Tests for the monitoring stack binary installer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.interfaces.monitoring.installer import INSTALL_DIR, _platform_arch, resolve_binary


class TestPlatformArch:
    """Test platform/architecture detection."""

    @patch("osprey.interfaces.monitoring.installer.sys")
    @patch("osprey.interfaces.monitoring.installer.platform")
    def test_darwin_arm64(self, mock_platform, mock_sys):
        mock_sys.platform = "darwin"
        mock_platform.machine.return_value = "arm64"
        os_name, arch = _platform_arch()
        assert os_name == "darwin"
        assert arch == "arm64"

    @patch("osprey.interfaces.monitoring.installer.sys")
    @patch("osprey.interfaces.monitoring.installer.platform")
    def test_linux_x86_64(self, mock_platform, mock_sys):
        mock_sys.platform = "linux"
        mock_platform.machine.return_value = "x86_64"
        os_name, arch = _platform_arch()
        assert os_name == "linux"
        assert arch == "amd64"

    @patch("osprey.interfaces.monitoring.installer.sys")
    @patch("osprey.interfaces.monitoring.installer.platform")
    def test_linux_aarch64(self, mock_platform, mock_sys):
        mock_sys.platform = "linux"
        mock_platform.machine.return_value = "aarch64"
        os_name, arch = _platform_arch()
        assert os_name == "linux"
        assert arch == "arm64"


class TestResolveBinary:
    """Test binary resolution logic."""

    @patch("osprey.interfaces.monitoring.installer.shutil.which")
    def test_prefers_path_over_installed(self, mock_which, tmp_path):
        """Binary on PATH should be preferred over ~/.osprey install."""
        path_binary = tmp_path / "prometheus"
        path_binary.touch()
        mock_which.return_value = str(path_binary)

        result = resolve_binary("prometheus")
        assert result == path_binary

    @patch("osprey.interfaces.monitoring.installer.shutil.which")
    def test_falls_back_to_installed(self, mock_which, tmp_path):
        """When not on PATH, fall back to ~/.osprey/monitoring/bin/."""
        mock_which.return_value = None

        # Create a fake installed binary
        install_dir = tmp_path / "bin"
        install_dir.mkdir()
        binary = install_dir / "prometheus"
        binary.touch()

        with patch("osprey.interfaces.monitoring.installer.INSTALL_DIR", install_dir):
            result = resolve_binary("prometheus")
            assert result == binary

    @patch("osprey.interfaces.monitoring.installer.shutil.which")
    def test_returns_none_when_not_found(self, mock_which, tmp_path):
        """Returns None when binary is not found anywhere."""
        mock_which.return_value = None

        with patch("osprey.interfaces.monitoring.installer.INSTALL_DIR", tmp_path / "nonexistent"):
            result = resolve_binary("prometheus")
            assert result is None
