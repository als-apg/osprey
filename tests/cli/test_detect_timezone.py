"""Tests for _detect_system_timezone() in scaffolding module."""

from unittest.mock import patch

from osprey.cli.templates.scaffolding import _detect_system_timezone


class TestDetectSystemTimezone:
    """Test automatic timezone detection."""

    def test_returns_string_or_none(self):
        """Detection returns a string timezone name or None."""
        result = _detect_system_timezone()
        assert result is None or isinstance(result, str)

    def test_detected_timezone_is_valid(self):
        """If a timezone is detected, it is a valid IANA timezone."""
        result = _detect_system_timezone()
        if result is not None:
            from zoneinfo import ZoneInfo

            # Should not raise
            ZoneInfo(result)

    def test_returns_none_when_no_localtime(self):
        """Returns None when /etc/localtime doesn't exist."""
        import pathlib

        with patch.object(pathlib.Path, "is_symlink", return_value=False):
            with patch.object(pathlib.Path, "exists", return_value=False):
                result = _detect_system_timezone()
                assert result is None

    def test_parses_zoneinfo_symlink(self):
        """Parses timezone from /etc/localtime symlink target."""
        import pathlib

        with patch.object(pathlib.Path, "is_symlink", return_value=True):
            with patch.object(
                pathlib.Path,
                "resolve",
                return_value=pathlib.Path("/usr/share/zoneinfo/America/Chicago"),
            ):
                result = _detect_system_timezone()
                assert result == "America/Chicago"
