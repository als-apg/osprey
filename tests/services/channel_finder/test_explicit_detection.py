"""
Unit tests for explicit channel detection feature.

Tests the optimization that detects when users provide explicit PV addresses
in their queries, allowing pipelines to skip search/navigation.

Covers:
- Detection of explicit addresses vs search terms
- Smart decision on whether additional search is needed
- Validation modes (strict, lenient, skip)
- Integration with all three pipelines
"""

from unittest.mock import MagicMock, patch

import pytest

from osprey.services.channel_finder.core.base_pipeline import (
    BasePipeline,
)
from osprey.services.channel_finder.core.models import (
    ExplicitChannelDetectionOutput,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_database():
    """Mock database with a few known channels."""
    db = MagicMock()
    # Known channels in database
    known_channels = [
        "MAG:HCM:H01:CURRENT:SP",
        "MAG:VCM:V01:CURRENT:SP",
        "BR:DCCT:CURRENT:RB",
    ]
    db.validate_channel = lambda ch: ch in known_channels
    db.get_channel = lambda ch: {"channel": ch, "address": ch} if ch in known_channels else None
    return db


@pytest.fixture
def mock_pipeline(mock_database):
    """Create a mock pipeline instance for testing."""

    # Create a concrete implementation for testing
    class TestPipeline(BasePipeline):
        def pipeline_name(self):
            return "Test Pipeline"

        async def process_query(self, query):
            pass

        def get_statistics(self):
            return {}

    # Mock the config loading to avoid needing config.yml
    with patch("osprey.utils.config.get_config_value", return_value="lenient"):
        pipeline = TestPipeline(mock_database, {"provider": "test", "model_id": "test"})
    return pipeline


# ============================================================
# Tests: Explicit Channel Detection
# ============================================================


class TestExplicitChannelDetection:
    """Test the LLM-based explicit channel detection."""

    @pytest.mark.asyncio
    async def test_detect_explicit_single_address(self, mock_pipeline):
        """Test detection of a single explicit PV address."""
        # Mock the LLM response
        mock_response = ExplicitChannelDetectionOutput(
            has_explicit_addresses=True,
            channel_addresses=["SC:HCM1:SP"],
            needs_additional_search=False,
            reasoning="Detected explicit PV address 'SC:HCM1:SP'",
        )

        with patch(
            "osprey.services.channel_finder.llm.get_chat_completion",
            return_value=mock_response,
        ):
            result = await mock_pipeline._detect_explicit_channels("Set the SC:HCM1:SP pv to 4.6")

        assert result.has_explicit_addresses is True
        assert len(result.channel_addresses) == 1
        assert "SC:HCM1:SP" in result.channel_addresses
        assert result.needs_additional_search is False

    @pytest.mark.asyncio
    async def test_detect_explicit_multiple_addresses(self, mock_pipeline):
        """Test detection of multiple explicit PV addresses."""
        mock_response = ExplicitChannelDetectionOutput(
            has_explicit_addresses=True,
            channel_addresses=["MAG:VCM:V01:CURRENT:SP", "MAG:HCM:H02:CURRENT:SP"],
            needs_additional_search=False,
            reasoning="Detected 2 explicit PV addresses",
        )

        with patch(
            "osprey.services.channel_finder.llm.get_chat_completion",
            return_value=mock_response,
        ):
            result = await mock_pipeline._detect_explicit_channels(
                "Get MAG:VCM:V01:CURRENT:SP and MAG:HCM:H02:CURRENT:SP"
            )

        assert result.has_explicit_addresses is True
        assert len(result.channel_addresses) == 2
        assert result.needs_additional_search is False

    @pytest.mark.asyncio
    async def test_detect_no_explicit_addresses(self, mock_pipeline):
        """Test when query has no explicit addresses (needs search)."""
        mock_response = ExplicitChannelDetectionOutput(
            has_explicit_addresses=False,
            channel_addresses=[],
            needs_additional_search=True,
            reasoning="No explicit channel addresses detected",
        )

        with patch(
            "osprey.services.channel_finder.llm.get_chat_completion",
            return_value=mock_response,
        ):
            result = await mock_pipeline._detect_explicit_channels(
                "Find the horizontal corrector magnet setpoint"
            )

        assert result.has_explicit_addresses is False
        assert len(result.channel_addresses) == 0
        assert result.needs_additional_search is True

    @pytest.mark.asyncio
    async def test_detect_mixed_explicit_and_search(self, mock_pipeline):
        """Test when query has both explicit addresses AND search terms."""
        mock_response = ExplicitChannelDetectionOutput(
            has_explicit_addresses=True,
            channel_addresses=["BR:HCM1"],
            needs_additional_search=True,
            reasoning="Has explicit 'BR:HCM1' but also needs search for 'all vertical corrector setpoints'",
        )

        with patch(
            "osprey.services.channel_finder.llm.get_chat_completion",
            return_value=mock_response,
        ):
            result = await mock_pipeline._detect_explicit_channels(
                "Get BR:HCM1 and all vertical corrector setpoints"
            )

        assert result.has_explicit_addresses is True
        assert len(result.channel_addresses) == 1
        assert result.needs_additional_search is True


# ============================================================
# Tests: Validation Modes
# ============================================================


class TestValidationModes:
    """Test the three validation modes: strict, lenient, skip."""

    def test_validation_strict_mode_all_valid(self, mock_pipeline):
        """Test strict mode with all channels in database."""
        mock_pipeline.explicit_validation_mode = "strict"

        valid, invalid = mock_pipeline._validate_explicit_channels(
            ["MAG:HCM:H01:CURRENT:SP", "MAG:VCM:V01:CURRENT:SP"]
        )

        assert len(valid) == 2
        assert len(invalid) == 0
        assert "MAG:HCM:H01:CURRENT:SP" in valid
        assert "MAG:VCM:V01:CURRENT:SP" in valid

    def test_validation_strict_mode_some_invalid(self, mock_pipeline):
        """Test strict mode with some channels not in database."""
        mock_pipeline.explicit_validation_mode = "strict"

        valid, invalid = mock_pipeline._validate_explicit_channels(
            [
                "MAG:HCM:H01:CURRENT:SP",  # in database
                "SC:CUSTOM:PV1:SP",  # NOT in database
                "MAG:VCM:V01:CURRENT:SP",  # in database
            ]
        )

        assert len(valid) == 2
        assert len(invalid) == 1
        assert "MAG:HCM:H01:CURRENT:SP" in valid
        assert "MAG:VCM:V01:CURRENT:SP" in valid
        assert "SC:CUSTOM:PV1:SP" in invalid

    def test_validation_lenient_mode_includes_all(self, mock_pipeline):
        """Test lenient mode includes all channels (warns for unknown)."""
        mock_pipeline.explicit_validation_mode = "lenient"

        valid, invalid = mock_pipeline._validate_explicit_channels(
            [
                "MAG:HCM:H01:CURRENT:SP",  # in database
                "SC:CUSTOM:PV1:SP",  # NOT in database
                "UNKNOWN:PV:ADDR",  # NOT in database
            ]
        )

        # Lenient mode: all go to valid list, invalid is empty
        assert len(valid) == 3
        assert len(invalid) == 0
        assert "MAG:HCM:H01:CURRENT:SP" in valid
        assert "SC:CUSTOM:PV1:SP" in valid
        assert "UNKNOWN:PV:ADDR" in valid

    def test_validation_skip_mode_no_validation(self, mock_pipeline):
        """Test skip mode accepts all without validation."""
        mock_pipeline.explicit_validation_mode = "skip"

        # Even with obviously fake addresses, skip mode accepts all
        valid, invalid = mock_pipeline._validate_explicit_channels(
            ["FAKE:ADDRESS:1", "ANOTHER:FAKE:2", "NOT:REAL:3"]
        )

        assert len(valid) == 3
        assert len(invalid) == 0
        assert "FAKE:ADDRESS:1" in valid

    def test_validation_default_mode_is_lenient(self, mock_database):
        """Test that default validation mode is lenient."""

        # Create pipeline without setting explicit_validation_mode
        class TestPipeline(BasePipeline):
            def pipeline_name(self):
                return "Test"

            async def process_query(self, query):
                pass

            def get_statistics(self):
                return {}

        with patch("osprey.utils.config.get_config_value", return_value="lenient") as mock_config:
            pipeline = TestPipeline(mock_database, {"provider": "test", "model_id": "test"})
            assert pipeline.explicit_validation_mode == "lenient"
            # Verify get_config_value was called with the right parameter
            mock_config.assert_called_once_with(
                "channel_finder.explicit_validation_mode", "lenient"
            )


# ============================================================
# Tests: Configuration Integration
# ============================================================


class TestConfigurationIntegration:
    """Test that validation mode is properly loaded from config."""

    def test_loads_validation_mode_from_config(self, mock_database):
        """Test that explicit_validation_mode is loaded from config."""

        class TestPipeline(BasePipeline):
            def pipeline_name(self):
                return "Test"

            async def process_query(self, query):
                pass

            def get_statistics(self):
                return {}

        # Test each mode
        for mode in ["strict", "lenient", "skip"]:
            with patch("osprey.utils.config.get_config_value", return_value=mode) as mock_config:
                pipeline = TestPipeline(mock_database, {"provider": "test", "model_id": "test"})
                assert pipeline.explicit_validation_mode == mode
                # Verify config was loaded with correct key and default
                mock_config.assert_called_once_with(
                    "channel_finder.explicit_validation_mode", "lenient"
                )

    def test_validation_mode_affects_behavior(self, mock_pipeline):
        """Test that changing validation mode changes behavior."""
        test_addresses = ["MAG:HCM:H01:CURRENT:SP", "UNKNOWN:PV"]

        # Strict mode: rejects unknown
        mock_pipeline.explicit_validation_mode = "strict"
        valid, invalid = mock_pipeline._validate_explicit_channels(test_addresses)
        assert len(valid) == 1
        assert len(invalid) == 1

        # Lenient mode: includes all
        mock_pipeline.explicit_validation_mode = "lenient"
        valid, invalid = mock_pipeline._validate_explicit_channels(test_addresses)
        assert len(valid) == 2
        assert len(invalid) == 0

        # Skip mode: includes all without validation
        mock_pipeline.explicit_validation_mode = "skip"
        valid, invalid = mock_pipeline._validate_explicit_channels(test_addresses)
        assert len(valid) == 2
        assert len(invalid) == 0


# ============================================================
# Tests: Edge Cases
# ============================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_channel_list(self, mock_pipeline):
        """Test validation with empty channel list."""
        mock_pipeline.explicit_validation_mode = "lenient"
        valid, invalid = mock_pipeline._validate_explicit_channels([])

        assert len(valid) == 0
        assert len(invalid) == 0

    @pytest.mark.asyncio
    async def test_detection_with_special_characters(self, mock_pipeline):
        """Test detection handles special characters in PV names."""
        mock_response = ExplicitChannelDetectionOutput(
            has_explicit_addresses=True,
            channel_addresses=["SR01C:BPM-1:X-POSITION"],
            needs_additional_search=False,
            reasoning="Detected PV with hyphens and colons",
        )

        with patch(
            "osprey.services.channel_finder.llm.get_chat_completion",
            return_value=mock_response,
        ):
            result = await mock_pipeline._detect_explicit_channels("Read SR01C:BPM-1:X-POSITION")

        assert result.has_explicit_addresses is True
        assert "SR01C:BPM-1:X-POSITION" in result.channel_addresses

    def test_validation_preserves_order(self, mock_pipeline):
        """Test that validation preserves channel order."""
        mock_pipeline.explicit_validation_mode = "skip"
        channels = ["FIRST:PV", "SECOND:PV", "THIRD:PV"]

        valid, _ = mock_pipeline._validate_explicit_channels(channels)

        assert valid == channels  # Order preserved


# ============================================================
# Tests: Build Result Helper
# ============================================================


class TestBuildResultHelper:
    """Test the shared _build_result helper method."""

    def test_build_result_with_known_channels(self, mock_pipeline):
        """Test building result with channels in database."""
        result = mock_pipeline._build_result(
            "test query", ["MAG:HCM:H01:CURRENT:SP", "BR:DCCT:CURRENT:RB"]
        )

        assert result.query == "test query"
        assert result.total_channels == 2
        assert len(result.channels) == 2
        assert result.channels[0].channel == "MAG:HCM:H01:CURRENT:SP"
        assert result.channels[0].address == "MAG:HCM:H01:CURRENT:SP"

    def test_build_result_with_unknown_channels(self, mock_pipeline):
        """Test building result with channels not in database (lenient mode)."""
        result = mock_pipeline._build_result("test query", ["UNKNOWN:PV:1", "FAKE:PV:2"])

        assert result.total_channels == 2
        # Channels not in database still included (used as-is for explicit addresses)
        assert len(result.channels) == 2
        assert result.channels[0].channel == "UNKNOWN:PV:1"
        assert result.channels[0].address == "UNKNOWN:PV:1"
        assert result.channels[0].description is None

    def test_build_result_empty_list(self, mock_pipeline):
        """Test building result with no channels."""
        result = mock_pipeline._build_result("test query", [])

        assert result.total_channels == 0
        assert len(result.channels) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
