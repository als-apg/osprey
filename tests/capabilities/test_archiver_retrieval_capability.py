"""Tests for native ArchiverRetrievalCapability.

Tests class attributes, error classification, context class, and guides.
"""

from datetime import datetime, timedelta

from osprey.base.errors import ErrorSeverity
from osprey.capabilities.archiver_retrieval import (
    ArchiverConnectionError,
    ArchiverDataContext,
    ArchiverDataError,
    ArchiverDependencyError,
    ArchiverRetrievalCapability,
    ArchiverTimeoutError,
)


class TestArchiverRetrievalCapabilityAttributes:
    """Test capability class attributes."""

    def test_name(self):
        assert ArchiverRetrievalCapability.name == "archiver_retrieval"

    def test_provides(self):
        assert "ARCHIVER_DATA" in ArchiverRetrievalCapability.provides

    def test_requires(self):
        requires = ArchiverRetrievalCapability.requires
        # Requires CHANNEL_ADDRESSES and TIME_RANGE
        req_types = [r if isinstance(r, str) else r[0] for r in requires]
        assert "CHANNEL_ADDRESSES" in req_types
        assert "TIME_RANGE" in req_types

    def test_has_description(self):
        assert ArchiverRetrievalCapability.description
        assert len(ArchiverRetrievalCapability.description) > 0


class TestArchiverRetrievalErrorClassification:
    """Test classify_error returns correct severity for each error type."""

    def test_timeout_is_retriable(self):
        exc = ArchiverTimeoutError("Request timed out")
        result = ArchiverRetrievalCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.RETRIABLE
        assert "suggestions" in result.metadata

    def test_connection_error_is_critical(self):
        exc = ArchiverConnectionError("Cannot connect to archiver")
        result = ArchiverRetrievalCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL
        assert "suggestions" in result.metadata

    def test_data_error_is_replanning(self):
        exc = ArchiverDataError("Unexpected data format")
        result = ArchiverRetrievalCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.REPLANNING
        assert "suggestions" in result.metadata

    def test_dependency_error_is_replanning(self):
        exc = ArchiverDependencyError("Missing CHANNEL_ADDRESSES")
        result = ArchiverRetrievalCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.REPLANNING

    def test_unknown_error_is_critical(self):
        exc = RuntimeError("Unexpected error")
        result = ArchiverRetrievalCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL


class TestArchiverDataContext:
    """Test ArchiverDataContext data class."""

    def test_context_type(self):
        assert ArchiverDataContext.CONTEXT_TYPE == "ARCHIVER_DATA"

    def test_context_category(self):
        assert ArchiverDataContext.CONTEXT_CATEGORY == "COMPUTATIONAL_DATA"

    def test_create_context(self):
        now = datetime.now()
        timestamps = [now + timedelta(seconds=i) for i in range(5)]
        ctx = ArchiverDataContext(
            timestamps=timestamps,
            precision_ms=1000,
            time_series_data={"SR:CURRENT:RB": [400.0, 400.1, 400.2, 400.3, 400.4]},
            available_channels=["SR:CURRENT:RB"],
        )
        assert len(ctx.timestamps) == 5
        assert len(ctx.available_channels) == 1
        assert ctx.precision_ms == 1000

    def test_get_access_details(self):
        now = datetime.now()
        timestamps = [now + timedelta(seconds=i) for i in range(5)]
        ctx = ArchiverDataContext(
            timestamps=timestamps,
            precision_ms=1000,
            time_series_data={"SR:CURRENT:RB": [400.0, 400.1, 400.2, 400.3, 400.4]},
            available_channels=["SR:CURRENT:RB"],
        )
        details = ctx.get_access_details("test_key")
        assert details["total_points"] == 5
        assert details["channel_count"] == 1
        assert "CRITICAL_ACCESS_PATTERNS" in details

    def test_get_summary_downsamples(self):
        """Test that get_summary downsamples large datasets."""
        now = datetime.now()
        # Create dataset larger than max_samples (10)
        n = 50
        timestamps = [now + timedelta(seconds=i) for i in range(n)]
        ctx = ArchiverDataContext(
            timestamps=timestamps,
            precision_ms=1000,
            time_series_data={"SR:CURRENT:RB": [float(i) for i in range(n)]},
            available_channels=["SR:CURRENT:RB"],
        )
        summary = ctx.get_summary()
        assert "WARNING" in summary
        assert "DOWNSAMPLED" in summary["WARNING"]
        # Should have channel data with fewer than n sample points
        channel_summary = summary["channel_data"]["SR:CURRENT:RB"]
        assert len(channel_summary["sample_values"]) <= 10

    def test_get_summary_small_dataset(self):
        """Test that get_summary doesn't downsample small datasets."""
        now = datetime.now()
        timestamps = [now + timedelta(seconds=i) for i in range(3)]
        ctx = ArchiverDataContext(
            timestamps=timestamps,
            precision_ms=1000,
            time_series_data={"SR:CURRENT:RB": [400.0, 400.1, 400.2]},
            available_channels=["SR:CURRENT:RB"],
        )
        summary = ctx.get_summary()
        channel_summary = summary["channel_data"]["SR:CURRENT:RB"]
        assert len(channel_summary["sample_values"]) == 3


class TestArchiverRetrievalGuides:
    """Test orchestrator and classifier guides."""

    def test_orchestrator_guide_exists(self):
        cap = ArchiverRetrievalCapability()
        guide = cap._create_orchestrator_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0

    def test_classifier_guide_exists(self):
        cap = ArchiverRetrievalCapability()
        guide = cap._create_classifier_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0
