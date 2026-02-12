"""
Unit tests for channel write verification system.

Tests the three-tier verification approach:
- none: Fast write, no verification
- callback: Control system confirms processing
- readback: Full verification with readback comparison
"""

import pytest

from osprey.connectors.control_system.base import (
    ChannelWriteResult,
    WriteVerification,
)
from osprey.connectors.control_system.mock_connector import MockConnector


class TestMockConnectorVerification:
    """Test verification levels in MockConnector."""

    @pytest.fixture
    async def connector(self):
        """Create and connect MockConnector."""
        conn = MockConnector()
        await conn.connect(
            {
                "response_delay_ms": 1,  # Fast for testing
                "noise_level": 0.001,  # Low noise
                "enable_writes": True,
            }
        )
        yield conn
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_verification_none(self, connector):
        """Test 'none' verification level - no verification performed."""
        result = await connector.write_channel("TEST:CHANNEL", 100.0, verification_level="none")

        assert isinstance(result, ChannelWriteResult)
        assert result.success is True
        assert result.channel_address == "TEST:CHANNEL"
        assert result.value_written == 100.0
        assert result.verification is not None
        assert result.verification.level == "none"
        assert result.verification.verified is False  # No verification performed
        assert result.verification.readback_value is None

    @pytest.mark.asyncio
    async def test_verification_callback(self, connector):
        """Test 'callback' verification level - simulated callback confirmation."""
        result = await connector.write_channel("TEST:CHANNEL", 100.0, verification_level="callback")

        assert isinstance(result, ChannelWriteResult)
        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "callback"
        assert result.verification.verified is True  # Mock always succeeds
        assert "callback" in result.verification.notes.lower()

    @pytest.mark.asyncio
    async def test_verification_readback_success(self, connector):
        """Test 'readback' verification level - successful verification."""
        result = await connector.write_channel(
            "TEST:CHANNEL",
            100.0,
            verification_level="readback",
            tolerance=1.0,  # Larger tolerance to account for mock noise (0.1% noise on 100 = ~0.1, but can be larger)
        )

        assert isinstance(result, ChannelWriteResult)
        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "readback"
        assert result.verification.verified is True  # Within tolerance
        assert result.verification.readback_value is not None
        assert result.verification.tolerance_used == 1.0

        # Check readback value is close to written value
        diff = abs(result.verification.readback_value - 100.0)
        assert diff <= 1.0  # Within tolerance

    @pytest.mark.asyncio
    async def test_verification_readback_with_percentage_tolerance(self, connector):
        """Test readback verification with percentage-based tolerance."""
        # Write a value
        written_value = 1000.0
        # noise_level=0.001 → sigma = 1000 * 0.001 = 1.0
        # Use 1% tolerance (= 10 sigma) to avoid flaky failures
        tolerance_percent = 1.0
        absolute_tolerance = written_value * tolerance_percent / 100.0  # = 10.0

        result = await connector.write_channel(
            "TEST:CHANNEL",
            written_value,
            verification_level="readback",
            tolerance=absolute_tolerance,
        )

        assert result.success is True
        assert result.verification.verified is True
        # Mock adds small noise, should be within 10.0
        diff = abs(result.verification.readback_value - written_value)
        assert diff <= absolute_tolerance

    @pytest.mark.asyncio
    async def test_writes_disabled_returns_failure(self, connector):
        """Test that writes fail when connector has writes disabled."""
        # Reconfigure connector with writes disabled
        await connector.disconnect()
        await connector.connect({"response_delay_ms": 1, "enable_writes": False})  # Disable writes

        result = await connector.write_channel("TEST:CHANNEL", 100.0, verification_level="callback")

        assert result.success is False
        assert result.verification is not None
        assert result.verification.verified is False
        assert "disabled" in result.verification.notes.lower()
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_invalid_verification_level_raises_error(self, connector):
        """Test that invalid verification level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid verification_level"):
            await connector.write_channel("TEST:CHANNEL", 100.0, verification_level="invalid_level")


class TestVerificationDataModels:
    """Test verification data model structures."""

    def test_write_verification_model(self):
        """Test WriteVerification dataclass structure."""
        verif = WriteVerification(
            level="readback",
            verified=True,
            readback_value=100.5,
            tolerance_used=0.1,
            notes="Readback confirmed",
        )

        assert verif.level == "readback"
        assert verif.verified is True
        assert verif.readback_value == 100.5
        assert verif.tolerance_used == 0.1
        assert verif.notes == "Readback confirmed"

    def test_channel_write_result_model(self):
        """Test ChannelWriteResult dataclass structure."""
        result = ChannelWriteResult(
            channel_address="TEST:CHANNEL",
            value_written=100.0,
            success=True,
            verification=WriteVerification(
                level="callback", verified=True, notes="IOC callback confirmed"
            ),
        )

        assert result.channel_address == "TEST:CHANNEL"
        assert result.value_written == 100.0
        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "callback"
        assert result.verification.verified is True

    def test_channel_write_result_without_verification(self):
        """Test ChannelWriteResult can be created without verification."""
        result = ChannelWriteResult(
            channel_address="TEST:CHANNEL", value_written=100.0, success=True, verification=None
        )

        assert result.success is True
        assert result.verification is None


class TestVerificationTolerance:
    """Test tolerance calculation strategies."""

    def test_absolute_tolerance(self):
        """Test absolute tolerance comparison."""
        written_value = 100.0
        readback_value = 100.05
        tolerance = 0.1  # Absolute

        diff = abs(readback_value - written_value)
        verified = diff <= tolerance

        assert verified is True
        assert diff == pytest.approx(0.05, rel=1e-9)

    def test_percentage_tolerance(self):
        """Test percentage-based tolerance calculation."""
        written_value = 1000.0
        tolerance_percent = 0.1  # 0.1%
        absolute_tolerance = written_value * tolerance_percent / 100.0

        assert absolute_tolerance == 1.0

        # Test cases
        assert abs(1000.5 - written_value) <= absolute_tolerance  # Pass
        assert abs(1001.0 - written_value) <= absolute_tolerance  # Pass
        assert abs(1001.5 - written_value) > absolute_tolerance  # Fail

    def test_scale_adaptive_tolerance(self):
        """Test that percentage tolerance adapts to value scale."""
        tolerance_percent = 0.1  # 0.1% (one per mil)

        # Large value (MHz scale)
        large_value = 500.5  # MHz
        large_tolerance = large_value * tolerance_percent / 100.0
        assert large_tolerance == pytest.approx(0.5005, rel=1e-6)

        # Small value (mA scale)
        small_value = 5.0  # mA
        small_tolerance = small_value * tolerance_percent / 100.0
        assert small_tolerance == pytest.approx(0.005, rel=1e-6)

        # Both are 0.1% of their respective values
        assert (large_tolerance / large_value) * 100 == pytest.approx(0.1, rel=1e-6)
        assert (small_tolerance / small_value) * 100 == pytest.approx(0.1, rel=1e-6)
