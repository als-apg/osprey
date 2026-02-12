"""
Unit tests for pipeline _build_result methods.

Tests the simplified _build_result implementations in hierarchical and middle_layer
pipelines to ensure they correctly handle channel/address/description fields.
"""

from unittest.mock import MagicMock

import pytest

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_hierarchical_db():
    """Mock hierarchical database."""
    db = MagicMock()
    known_channels = ["CTRL:DEV:D01:Temperature", "CTRL:DEV:D02:Voltage"]
    db.validate_channel = lambda ch: ch in known_channels
    db.get_channel = lambda ch: {"channel": ch, "path": {}} if ch in known_channels else None
    return db


@pytest.fixture
def mock_middle_layer_db():
    """Mock middle layer database."""
    db = MagicMock()
    known_channels = ["SR01C:BPM1:X", "SR01C:BPM1:Y"]
    # Middle layer DB returns reconstructed description
    db.validate_channel = lambda ch: ch in known_channels
    db.get_channel = lambda ch: (
        {"channel": ch, "address": ch, "description": "SR:BPM:Monitor"}
        if ch in known_channels
        else None
    )
    return db


@pytest.fixture
def hierarchical_pipeline(mock_hierarchical_db):
    """Create hierarchical pipeline with mock database."""
    from osprey.services.channel_finder.pipelines.hierarchical.pipeline import (
        HierarchicalPipeline,
    )

    # Create a test pipeline instance with minimal mocking
    class TestHierarchicalPipeline(HierarchicalPipeline):
        def __init__(self, database):
            # Skip BasePipeline.__init__ to avoid config loading
            self.database = database
            self.model_config = {"provider": "test", "model_id": "test"}
            self.explicit_validation_mode = "lenient"

    return TestHierarchicalPipeline(mock_hierarchical_db)


@pytest.fixture
def middle_layer_pipeline(mock_middle_layer_db):
    """Create middle layer pipeline with mock database."""
    from osprey.services.channel_finder.pipelines.middle_layer.pipeline import (
        MiddleLayerPipeline,
    )

    # Create a test pipeline instance with minimal mocking
    class TestMiddleLayerPipeline(MiddleLayerPipeline):
        def __init__(self, database):
            # Skip BasePipeline.__init__ to avoid config loading
            self.database = database
            self.model_config = {"provider": "test", "model_id": "test"}
            self.explicit_validation_mode = "lenient"

    return TestMiddleLayerPipeline(mock_middle_layer_db)


# ============================================================
# Tests: Hierarchical Pipeline _build_result
# ============================================================


class TestHierarchicalBuildResult:
    """Test hierarchical pipeline's simplified _build_result method."""

    def test_build_result_with_valid_channels(self, hierarchical_pipeline):
        """Test building result with channels from hierarchical database."""
        channels = ["CTRL:DEV:D01:Temperature", "CTRL:DEV:D02:Voltage"]

        result = hierarchical_pipeline._build_result("test query", channels)

        assert result.query == "test query"
        assert result.total_channels == 2
        assert len(result.channels) == 2

        # For hierarchical DB: channel == address, description == None
        assert result.channels[0].channel == "CTRL:DEV:D01:Temperature"
        assert result.channels[0].address == "CTRL:DEV:D01:Temperature"
        assert result.channels[0].description is None

        assert result.channels[1].channel == "CTRL:DEV:D02:Voltage"
        assert result.channels[1].address == "CTRL:DEV:D02:Voltage"
        assert result.channels[1].description is None

    def test_build_result_with_explicit_channels(self, hierarchical_pipeline):
        """Test building result with explicit addresses (lenient mode)."""
        # In lenient mode, explicit addresses that aren't in DB still get returned
        channels = ["EXPLICIT:CHANNEL:ADDR"]

        result = hierarchical_pipeline._build_result("test query", channels)

        assert result.total_channels == 1
        assert result.channels[0].channel == "EXPLICIT:CHANNEL:ADDR"
        assert result.channels[0].address == "EXPLICIT:CHANNEL:ADDR"
        assert result.channels[0].description is None

    def test_build_result_empty_list(self, hierarchical_pipeline):
        """Test building result with no channels."""
        result = hierarchical_pipeline._build_result("test query", [])

        assert result.total_channels == 0
        assert len(result.channels) == 0
        assert "0 channels" in result.processing_notes

    def test_build_result_preserves_order(self, hierarchical_pipeline):
        """Test that channel order is preserved."""
        channels = [
            "CTRL:DEV:D03:Voltage",
            "CTRL:DEV:D01:Temperature",
            "CTRL:DEV:D02:Voltage",
        ]

        result = hierarchical_pipeline._build_result("test query", channels)

        assert result.channels[0].channel == "CTRL:DEV:D03:Voltage"
        assert result.channels[1].channel == "CTRL:DEV:D01:Temperature"
        assert result.channels[2].channel == "CTRL:DEV:D02:Voltage"


# ============================================================
# Tests: Middle Layer Pipeline _build_result
# ============================================================


class TestMiddleLayerBuildResult:
    """Test middle layer pipeline's simplified _build_result method."""

    def test_build_result_with_valid_channels(self, middle_layer_pipeline):
        """Test building result with channels from middle layer database."""
        channels = ["SR01C:BPM1:X", "SR01C:BPM1:Y"]

        result = middle_layer_pipeline._build_result("test query", channels)

        assert result.query == "test query"
        assert result.total_channels == 2
        assert len(result.channels) == 2

        # For middle layer DB: channel == address, description == None (not meaningful)
        assert result.channels[0].channel == "SR01C:BPM1:X"
        assert result.channels[0].address == "SR01C:BPM1:X"
        assert result.channels[0].description is None

        assert result.channels[1].channel == "SR01C:BPM1:Y"
        assert result.channels[1].address == "SR01C:BPM1:Y"
        assert result.channels[1].description is None

    def test_build_result_with_explicit_channels(self, middle_layer_pipeline):
        """Test building result with explicit addresses (lenient mode)."""
        # In lenient mode, explicit addresses that aren't in DB still get returned
        channels = ["CUSTOM:PV:ADDR"]

        result = middle_layer_pipeline._build_result("test query", channels)

        assert result.total_channels == 1
        assert result.channels[0].channel == "CUSTOM:PV:ADDR"
        assert result.channels[0].address == "CUSTOM:PV:ADDR"
        assert result.channels[0].description is None

    def test_build_result_empty_list(self, middle_layer_pipeline):
        """Test building result with no channels."""
        result = middle_layer_pipeline._build_result("test query", [])

        assert result.total_channels == 0
        assert len(result.channels) == 0
        assert "0 channels" in result.processing_notes

    def test_build_result_preserves_order(self, middle_layer_pipeline):
        """Test that channel order is preserved."""
        channels = ["SR01C:BPM1:Y", "SR01C:BPM1:X"]

        result = middle_layer_pipeline._build_result("test query", channels)

        assert result.channels[0].channel == "SR01C:BPM1:Y"
        assert result.channels[1].channel == "SR01C:BPM1:X"

    def test_build_result_database_description_not_used(
        self, middle_layer_pipeline, mock_middle_layer_db
    ):
        """Test that database's reconstructed description is NOT used."""
        channels = ["SR01C:BPM1:X"]

        # Verify database HAS a description field (but it's just the path)
        channel_data = mock_middle_layer_db.get_channel("SR01C:BPM1:X")
        assert channel_data is not None
        assert "description" in channel_data
        assert channel_data["description"] == "SR:BPM:Monitor"  # Reconstructed path

        # But _build_result should NOT use it (it's not meaningful)
        result = middle_layer_pipeline._build_result("test query", channels)
        assert result.channels[0].description is None


# ============================================================
# Tests: Consistency Between Pipelines
# ============================================================


class TestPipelineConsistency:
    """Test that both pipelines handle similar cases consistently."""

    def test_both_set_address_equal_to_channel(self, hierarchical_pipeline, middle_layer_pipeline):
        """Both pipelines should set address = channel."""
        h_result = hierarchical_pipeline._build_result("test", ["CHANNEL1"])
        m_result = middle_layer_pipeline._build_result("test", ["CHANNEL2"])

        assert h_result.channels[0].channel == h_result.channels[0].address
        assert m_result.channels[0].channel == m_result.channels[0].address

    def test_both_set_description_to_none(self, hierarchical_pipeline, middle_layer_pipeline):
        """Both pipelines should set description = None."""
        h_result = hierarchical_pipeline._build_result("test", ["CHANNEL1"])
        m_result = middle_layer_pipeline._build_result("test", ["CHANNEL2"])

        assert h_result.channels[0].description is None
        assert m_result.channels[0].description is None

    def test_both_handle_empty_list(self, hierarchical_pipeline, middle_layer_pipeline):
        """Both pipelines should handle empty channel lists."""
        h_result = hierarchical_pipeline._build_result("test", [])
        m_result = middle_layer_pipeline._build_result("test", [])

        assert h_result.total_channels == 0
        assert m_result.total_channels == 0
        assert len(h_result.channels) == 0
        assert len(m_result.channels) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
