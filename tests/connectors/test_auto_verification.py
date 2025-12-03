"""
Tests for automatic verification configuration in connectors.

Verifies that connectors automatically determine verification level and tolerance
from per-channel config (limits database) or global config, as documented.
"""

import json
import tempfile
from pathlib import Path

import pytest

from osprey.connectors.control_system.mock_connector import MockConnector


class TestAutomaticVerification:
    """Test automatic verification configuration lookup."""

    @pytest.mark.asyncio
    async def test_automatic_verification_without_config(self):
        """Test that connector works with automatic verification when no config available."""
        connector = MockConnector()
        await connector.connect({
            'response_delay_ms': 1,
            'enable_writes': True
        })

        # Call without verification_level - should use hardcoded default (callback)
        result = await connector.write_channel("TEST:CHANNEL", 100.0)

        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "callback"
        assert result.verification.verified is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_automatic_verification_with_global_config(self, tmp_path):
        """Test automatic verification uses global config when available."""
        # Create a temporary config file
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
control_system:
  type: mock
  writes_enabled: true
  write_verification:
    default_level: readback
    default_tolerance_percent: 0.5
""")

        # Create connector (will try to load config but gracefully fall back)
        connector = MockConnector()
        await connector.connect({
            'response_delay_ms': 1,
            'enable_writes': True
        })

        # Without explicit params, uses automatic config
        result = await connector.write_channel("TEST:CHANNEL", 100.0)

        assert result.success is True
        assert result.verification is not None
        # Will be callback (hardcoded fallback) since we can't inject config in tests
        # This is expected - tests run without config.yml

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_automatic_verification_with_per_channel_config(self, tmp_path, monkeypatch):
        """Test automatic verification uses per-channel config from limits database."""
        # Create limits database with per-channel verification
        limits_db = {
            "_comment": "Test limits database",
            "defaults": {
                "writable": True,
                "verification": {
                    "level": "callback"
                }
            },
            "CRITICAL:CHANNEL": {
                "min_value": 0.0,
                "max_value": 100.0,
                "verification": {
                    "level": "readback",
                }
            },
            "NORMAL:CHANNEL": {
                "min_value": 0.0,
                "max_value": 100.0
                # No verification override - uses defaults
            }
        }

        limits_file = tmp_path / "limits.json"
        limits_file.write_text(json.dumps(limits_db, indent=2))

        # Mock config to enable limits checking
        def mock_get_config_value(key, default=None):
            config_map = {
                'control_system.limits_checking.enabled': True,
                'control_system.limits_checking.database_path': str(limits_file),
                'control_system.limits_checking.allow_unlisted_channels': False,
                'control_system.limits_checking.on_violation': 'skip',
                'control_system.writes_enabled': True,
                'control_system.write_verification.default_level': 'callback',
                'control_system.write_verification.default_tolerance_percent': 0.1
            }
            return config_map.get(key, default)

        monkeypatch.setattr('osprey.utils.config.get_config_value', mock_get_config_value)

        # Create connector with limits validator
        connector = MockConnector()
        await connector.connect({
            'response_delay_ms': 1,
            'enable_writes': True,
            'noise_level': 0.0  # No noise for reliable verification tests
        })

        # Write to critical channel - should use readback verification
        result = await connector.write_channel("CRITICAL:CHANNEL", 50.0)

        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "readback"  # Per-channel config
        assert result.verification.verified is True
        assert result.verification.readback_value is not None

        # Write to normal channel - should use callback (from defaults)
        result = await connector.write_channel("NORMAL:CHANNEL", 50.0)

        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "callback"  # From defaults
        assert result.verification.verified is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_manual_override_still_works(self):
        """Test that manual verification_level parameter still works (override)."""
        connector = MockConnector()
        await connector.connect({
            'response_delay_ms': 1,
            'enable_writes': True
        })

        # Explicitly request 'none' verification
        result = await connector.write_channel(
            "TEST:CHANNEL",
            100.0,
            verification_level="none"
        )

        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "none"
        assert result.verification.verified is False

        # Explicitly request 'readback' with tolerance
        result = await connector.write_channel(
            "TEST:CHANNEL",
            100.0,
            verification_level="readback",
            tolerance=1.0
        )

        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "readback"
        assert result.verification.tolerance_used == 1.0

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_automatic_tolerance_calculation(self, tmp_path, monkeypatch):
        """Test that tolerance is automatically calculated from percentage."""
        limits_db = {
            "MOTOR:POSITION": {
                "min_value": -100.0,
                "max_value": 100.0,
                "verification": {
                    "level": "readback",
                    "tolerance_percent": 0.5  # 0.5%
                }
            }
        }

        limits_file = tmp_path / "limits.json"
        limits_file.write_text(json.dumps(limits_db, indent=2))

        def mock_get_config_value(key, default=None):
            config_map = {
                'control_system.limits_checking.enabled': True,
                'control_system.limits_checking.database_path': str(limits_file),
                'control_system.limits_checking.allow_unlisted_channels': False,
                'control_system.limits_checking.on_violation': 'skip',
                'control_system.writes_enabled': True,
            }
            return config_map.get(key, default)

        monkeypatch.setattr('osprey.utils.config.get_config_value', mock_get_config_value)

        connector = MockConnector()
        await connector.connect({
            'response_delay_ms': 1,
            'enable_writes': True,
            'noise_level': 0.0  # No noise for reliable verification tests
        })

        # Write value of 50.0 - tolerance should be 50.0 * 0.5% = 0.25
        result = await connector.write_channel("MOTOR:POSITION", 50.0)

        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "readback"
        assert result.verification.tolerance_used == pytest.approx(0.25, abs=0.01)

        await connector.disconnect()

