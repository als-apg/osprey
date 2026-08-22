"""
Tests for automatic verification configuration in connectors.

Verifies that connectors automatically determine verification level and tolerance
from per-channel config (limits database) or global config, as documented.
"""

import json
from unittest.mock import patch

import pytest

from osprey.connectors.control_system.mock_connector import MockConnector


def _config_with_writes_enabled(key, default=None):
    if key == "control_system.writes_enabled":
        return True
    return default


class TestAutomaticVerification:
    """Test automatic verification configuration lookup."""

    @pytest.mark.asyncio
    async def test_automatic_verification_without_config(self):
        """Test that connector works with automatic verification when no config available."""
        connector = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await connector.connect({"response_delay_ms": 1})

            # Call without verification_level - should use hardcoded default (callback)
            result = await connector.write_channel("TEST:CHANNEL", 100.0)

            assert result.success is True
            assert result.verification is not None
            assert result.verification.level == "callback"
            assert result.verification.verified is True

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_automatic_verification_with_per_channel_config(self, tmp_path, monkeypatch):
        """Test automatic verification uses per-channel config from limits database."""
        # Create limits database with per-channel verification
        limits_db = {
            "_comment": "Test limits database",
            "defaults": {"writable": True, "verification": {"level": "callback"}},
            "CRITICAL:CHANNEL": {
                "min_value": 0.0,
                "max_value": 100.0,
                "verification": {
                    "level": "readback",
                },
            },
            "NORMAL:CHANNEL": {
                "min_value": 0.0,
                "max_value": 100.0,
                # No verification override - uses defaults
            },
        }

        limits_file = tmp_path / "limits.json"
        limits_file.write_text(json.dumps(limits_db, indent=2))

        # Mock config to enable limits checking
        def mock_get_config_value(key, default=None):
            config_map = {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": str(limits_file),
                "control_system.limits_checking.allow_unlisted_channels": False,
                "control_system.limits_checking.on_violation": "skip",
                "control_system.writes_enabled": True,
                "control_system.write_verification.default_level": "callback",
                "control_system.write_verification.default_tolerance_percent": 0.1,
            }
            return config_map.get(key, default)

        monkeypatch.setattr("osprey.utils.config.get_config_value", mock_get_config_value)

        # Create connector with limits validator
        connector = MockConnector()
        await connector.connect(
            {
                "response_delay_ms": 1,
                "noise_level": 0.0,  # No noise for reliable verification tests
            }
        )

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
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await connector.connect({"response_delay_ms": 1})

            # Explicitly request 'none' verification
            result = await connector.write_channel("TEST:CHANNEL", 100.0, verification_level="none")

            assert result.success is True
            assert result.verification is not None
            assert result.verification.level == "none"
            assert result.verification.verified is False

            # Explicitly request 'readback' with tolerance
            result = await connector.write_channel(
                "TEST:CHANNEL", 100.0, verification_level="readback", tolerance=1.0
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
                "verification": {"level": "readback", "tolerance_percent": 0.5},  # 0.5%
            }
        }

        limits_file = tmp_path / "limits.json"
        limits_file.write_text(json.dumps(limits_db, indent=2))

        def mock_get_config_value(key, default=None):
            config_map = {
                "control_system.limits_checking.enabled": True,
                "control_system.limits_checking.database_path": str(limits_file),
                "control_system.limits_checking.allow_unlisted_channels": False,
                "control_system.limits_checking.on_violation": "skip",
                "control_system.writes_enabled": True,
            }
            return config_map.get(key, default)

        monkeypatch.setattr("osprey.utils.config.get_config_value", mock_get_config_value)

        connector = MockConnector()
        await connector.connect(
            {
                "response_delay_ms": 1,
                "noise_level": 0.0,  # No noise for reliable verification tests
            }
        )

        # Write value of 50.0 - tolerance should be 50.0 * 0.5% = 0.25
        result = await connector.write_channel("MOTOR:POSITION", 50.0)

        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "readback"
        assert result.verification.tolerance_used == pytest.approx(0.25, abs=0.01)

        await connector.disconnect()


def _limits_config(limits_file, **extra):
    """Config map that enables limits checking against ``limits_file``."""
    config_map = {
        "control_system.limits_checking.enabled": True,
        "control_system.limits_checking.database_path": str(limits_file),
        "control_system.limits_checking.allow_unlisted_channels": False,
        "control_system.limits_checking.on_violation": "skip",
        "control_system.writes_enabled": True,
    }
    config_map.update(extra)

    def get_config_value(key, default=None):
        return config_map.get(key, default)

    return get_config_value


def _write_limits_db(tmp_path, limits_db):
    limits_file = tmp_path / "limits.json"
    limits_file.write_text(json.dumps(limits_db, indent=2))
    return limits_file


class TestVerificationPrecedence:
    """Pin the four-layer resolution order documented on ``write_channel``.

    1. the channel's own ``verification`` entry, 2. the limits database
    ``defaults.verification`` block, 3. global ``control_system.write_verification``
    config, 4. the hard-coded ``("callback", None)`` fallback.
    """

    @pytest.mark.asyncio
    async def test_channel_entry_beats_defaults_block(self, tmp_path, monkeypatch):
        """Layer 1 wins over layer 2: a channel entry overrides the defaults block."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "verification": {"level": "none"}},
                "PICKY:CHANNEL": {
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "verification": {"level": "readback", "tolerance_absolute": 0.5},
                },
                "PLAIN:CHANNEL": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        monkeypatch.setattr("osprey.utils.config.get_config_value", _limits_config(limits_file))

        connector = MockConnector()
        await connector.connect({"response_delay_ms": 1, "noise_level": 0.0})

        assert connector._get_verification_config("PICKY:CHANNEL", 10.0) == ("readback", 0.5)
        # The channel without its own entry falls through to the defaults block.
        assert connector._get_verification_config("PLAIN:CHANNEL", 10.0) == ("none", None)

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_defaults_block_beats_global_config(self, tmp_path, monkeypatch):
        """Layer 2 wins over layer 3: the limits database outranks config.yml."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {
                    "writable": True,
                    "verification": {"level": "readback", "tolerance_percent": 1.0},
                },
                "ANY:CHANNEL": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            _limits_config(
                limits_file,
                **{
                    "control_system.write_verification.default_level": "none",
                    "control_system.write_verification.default_tolerance_percent": 25.0,
                },
            ),
        )

        connector = MockConnector()
        await connector.connect({"response_delay_ms": 1, "noise_level": 0.0})

        level, tolerance = connector._get_verification_config("ANY:CHANNEL", 40.0)
        assert level == "readback"
        assert tolerance == pytest.approx(0.4)  # 1 % of 40.0, not the global 25 %

        await connector.disconnect()

    def test_global_config_used_without_limits_database(self, monkeypatch):
        """Layer 3: with no limits database, config.yml supplies level and tolerance."""
        config_map = {
            "control_system.write_verification.default_level": "readback",
            "control_system.write_verification.default_tolerance_percent": 0.5,
        }
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda key, default=None: config_map.get(key, default),
        )

        connector = MockConnector()
        assert connector._limits_validator is None

        level, tolerance = connector._get_verification_config("TEST:CHANNEL", 100.0)
        assert level == "readback"
        assert tolerance == pytest.approx(0.5)  # 0.5 % of 100.0

    def test_hardcoded_fallback_when_config_unavailable(self, monkeypatch):
        """Layer 4: no limits database and no config leaves the safe built-in default."""

        def no_config(key, default=None):
            raise FileNotFoundError("no config.yml in this project")

        monkeypatch.setattr("osprey.utils.config.get_config_value", no_config)

        connector = MockConnector()
        assert connector._get_verification_config("TEST:CHANNEL", 100.0) == ("callback", None)

    @pytest.mark.asyncio
    async def test_explicit_level_overrides_every_layer(self, tmp_path, monkeypatch):
        """An explicit ``verification_level`` outranks all four layers."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "verification": {"level": "callback"}},
                "STRICT:CHANNEL": {
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "verification": {"level": "readback"},
                },
            },
        )
        monkeypatch.setattr("osprey.utils.config.get_config_value", _limits_config(limits_file))

        connector = MockConnector()
        await connector.connect({"response_delay_ms": 1, "noise_level": 0.0})

        # Omitted: the channel entry decides.
        omitted = await connector.write_channel("STRICT:CHANNEL", 50.0)
        assert omitted.verification.level == "readback"

        # Explicit: the caller decides, database entry notwithstanding.
        explicit = await connector.write_channel("STRICT:CHANNEL", 50.0, verification_level="none")
        assert explicit.verification.level == "none"

        await connector.disconnect()


class TestBatchVerificationResolution:
    """``write_multiple_channels`` resolves per channel instead of pinning "callback"."""

    @pytest.mark.asyncio
    async def test_batch_resolves_defaults_block_per_channel(self, tmp_path, monkeypatch):
        """An omitted level lets every channel in the batch resolve its own.

        This is the ``osprey.runtime.write_channels`` parity change: batches used to
        arrive at the connector pinned at "callback".
        """
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "verification": {"level": "readback"}},
                "BATCH:CH1": {"min_value": 0.0, "max_value": 100.0},
                "BATCH:CH2": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        monkeypatch.setattr("osprey.utils.config.get_config_value", _limits_config(limits_file))

        connector = MockConnector()
        await connector.connect({"response_delay_ms": 1, "noise_level": 0.0})

        results = await connector.write_multiple_channels(
            [("BATCH:CH1", 10.0), ("BATCH:CH2", 20.0)]
        )

        assert [r.channel_address for r in results] == ["BATCH:CH1", "BATCH:CH2"]
        for result in results:
            assert result.verification is not None
            assert result.verification.level == "readback"

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_batch_forwards_explicit_level_to_every_channel(self, tmp_path, monkeypatch):
        """An explicit level is forwarded unchanged, overriding the database."""
        limits_file = _write_limits_db(
            tmp_path,
            {
                "defaults": {"writable": True, "verification": {"level": "readback"}},
                "BATCH:CH1": {"min_value": 0.0, "max_value": 100.0},
                "BATCH:CH2": {"min_value": 0.0, "max_value": 100.0},
            },
        )
        monkeypatch.setattr("osprey.utils.config.get_config_value", _limits_config(limits_file))

        connector = MockConnector()
        await connector.connect({"response_delay_ms": 1, "noise_level": 0.0})

        results = await connector.write_multiple_channels(
            [("BATCH:CH1", 10.0), ("BATCH:CH2", 20.0)], verification_level="none"
        )

        for result in results:
            assert result.verification is not None
            assert result.verification.level == "none"

        await connector.disconnect()
