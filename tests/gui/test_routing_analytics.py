"""
Tests for Routing Analytics

Tests metrics tracking, statistics generation, time-series data,
and analytics dashboard functionality.
"""

# Import from local module
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from osprey.interfaces.pyqt.routing_analytics import (
    RoutingAnalytics,
    RoutingMetric,
)

sys.path.insert(0, str(Path(__file__).parent))
from routing_test_base import RoutingTestBase

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_analytics_path(tmp_path):
    """Create temporary analytics file path."""
    return tmp_path / "test_analytics.json"


@pytest.fixture
def analytics_system(temp_analytics_path):
    """Create routing analytics system with temp storage."""
    return RoutingAnalytics(
        max_history=100,
        enable_persistence=True,
        persistence_path=temp_analytics_path,
    )


@pytest.fixture
def analytics_system_no_persist():
    """Create routing analytics system without persistence."""
    return RoutingAnalytics(
        max_history=100,
        enable_persistence=False,
    )


# ============================================================================
# RoutingMetric Tests
# ============================================================================


class TestRoutingMetric:
    """Test RoutingMetric dataclass."""

    def test_create_basic_metric(self):
        """Test creating basic routing metric."""
        metric = RoutingMetric(
            timestamp=time.time(),
            query="test query",
            project_selected="project1",
            confidence=0.9,
            routing_time_ms=50.0,
            cache_hit=False,
            mode="automatic",
        )

        assert metric.query == "test query"
        assert metric.project_selected == "project1"
        assert metric.confidence == 0.9
        assert metric.routing_time_ms == 50.0
        assert metric.cache_hit is False
        assert metric.mode == "automatic"
        assert metric.success is True
        assert metric.error is None

    def test_create_metric_with_cache_hit(self):
        """Test creating metric with cache hit."""
        metric = RoutingMetric(
            timestamp=time.time(),
            query="cached query",
            project_selected="project1",
            confidence=0.95,
            routing_time_ms=5.0,
            cache_hit=True,
            mode="automatic",
        )

        assert metric.cache_hit is True
        assert metric.routing_time_ms == 5.0

    def test_create_metric_with_failure(self):
        """Test creating metric with failure."""
        metric = RoutingMetric(
            timestamp=time.time(),
            query="failed query",
            project_selected="project1",
            confidence=0.5,
            routing_time_ms=100.0,
            cache_hit=False,
            mode="automatic",
            success=False,
            error="Routing failed",
        )

        assert metric.success is False
        assert metric.error == "Routing failed"

    def test_create_metric_with_alternatives(self):
        """Test creating metric with alternative projects."""
        metric = RoutingMetric(
            timestamp=time.time(),
            query="test query",
            project_selected="project1",
            confidence=0.8,
            routing_time_ms=50.0,
            cache_hit=False,
            mode="automatic",
            alternative_projects=["project2", "project3"],
        )

        assert metric.alternative_projects == ["project2", "project3"]


# ============================================================================
# RoutingAnalytics Initialization Tests
# ============================================================================


class TestRoutingAnalyticsInit:
    """Test RoutingAnalytics initialization."""

    def test_init_default(self, temp_analytics_path):
        """Test initialization with default parameters."""
        system = RoutingAnalytics(persistence_path=temp_analytics_path)

        assert system.max_history == 1000
        assert system.enable_persistence is True
        assert system.persistence_path == temp_analytics_path

    def test_init_custom_params(self, temp_analytics_path):
        """Test initialization with custom parameters."""
        system = RoutingAnalytics(
            max_history=500,
            enable_persistence=False,
            persistence_path=temp_analytics_path,
        )

        assert system.max_history == 500
        assert system.enable_persistence is False

    def test_init_creates_empty_structures(self, analytics_system):
        """Test initialization creates empty data structures."""
        assert len(analytics_system._metrics) == 0
        assert len(analytics_system._project_stats) == 0
        assert len(analytics_system._query_patterns) == 0


# ============================================================================
# Metrics Recording Tests
# ============================================================================


class TestMetricsRecording:
    """Test metrics recording functionality."""

    def test_record_basic_routing(self, analytics_system):
        """Test recording basic routing decision."""
        analytics_system.record_routing(
            query="What is the weather?",
            project_selected="weather_project",
            confidence=0.9,
            routing_time_ms=50.0,
        )

        assert len(analytics_system._metrics) == 1
        metric = analytics_system._metrics[0]
        assert metric.query == "What is the weather?"
        assert metric.project_selected == "weather_project"
        assert metric.confidence == 0.9

    def test_record_with_cache_hit(self, analytics_system):
        """Test recording routing with cache hit."""
        analytics_system.record_routing(
            query="cached query",
            project_selected="project1",
            confidence=0.95,
            routing_time_ms=5.0,
            cache_hit=True,
        )

        metric = analytics_system._metrics[0]
        assert metric.cache_hit is True

    def test_record_manual_routing(self, analytics_system):
        """Test recording manual routing decision."""
        analytics_system.record_routing(
            query="manual query",
            project_selected="project1",
            confidence=1.0,
            routing_time_ms=0.0,
            mode="manual",
        )

        metric = analytics_system._metrics[0]
        assert metric.mode == "manual"

    def test_record_with_reasoning(self, analytics_system):
        """Test recording routing with reasoning."""
        analytics_system.record_routing(
            query="test query",
            project_selected="project1",
            confidence=0.8,
            routing_time_ms=50.0,
            reasoning="High keyword match",
        )

        metric = analytics_system._metrics[0]
        assert metric.reasoning == "High keyword match"

    def test_record_with_alternatives(self, analytics_system):
        """Test recording routing with alternative projects."""
        analytics_system.record_routing(
            query="test query",
            project_selected="project1",
            confidence=0.8,
            routing_time_ms=50.0,
            alternative_projects=["project2", "project3"],
        )

        metric = analytics_system._metrics[0]
        assert metric.alternative_projects == ["project2", "project3"]

    def test_record_failed_routing(self, analytics_system):
        """Test recording failed routing."""
        analytics_system.record_routing(
            query="failed query",
            project_selected="project1",
            confidence=0.5,
            routing_time_ms=100.0,
            success=False,
            error="No matching project",
        )

        metric = analytics_system._metrics[0]
        assert metric.success is False
        assert metric.error == "No matching project"

    def test_max_history_enforcement(self, temp_analytics_path):
        """Test max history limit is enforced."""
        system = RoutingAnalytics(
            max_history=5,
            enable_persistence=False,
            persistence_path=temp_analytics_path,
        )

        # Use base class helper
        base = RoutingTestBase()
        metrics = base.verify_max_history_enforcement(
            system=system,
            max_history=5,
            add_item_func=lambda i: system.record_routing(
                query=f"query {i}",
                project_selected="project1",
                confidence=0.8,
                routing_time_ms=50.0,
            ),
            get_items_func=lambda: system._metrics,
            item_count=10,
        )
        # Should keep most recent
        assert metrics[-1].query == "query 9"

    def test_project_stats_update(self, analytics_system):
        """Test project statistics are updated."""
        analytics_system.record_routing(
            query="query1", project_selected="proj1", confidence=0.9, routing_time_ms=50.0
        )
        analytics_system.record_routing(
            query="query2", project_selected="proj1", confidence=0.8, routing_time_ms=60.0
        )

        stats = analytics_system._project_stats["proj1"]
        assert stats["count"] == 2
        assert abs(stats["total_confidence"] - 1.7) < 0.01  # Floating point tolerance
        assert stats["total_time_ms"] == 110.0

    def test_query_pattern_tracking(self, analytics_system):
        """Test query patterns are tracked."""
        analytics_system.record_routing(
            query="What is the weather?",
            project_selected="weather_proj",
            confidence=0.9,
            routing_time_ms=50.0,
        )

        assert "weather query" in analytics_system._query_patterns
        assert len(analytics_system._query_patterns["weather query"]) == 1


# ============================================================================
# Summary Statistics Tests
# ============================================================================


class TestSummaryStatistics:
    """Test summary statistics generation."""

    def test_summary_empty(self, analytics_system):
        """Test summary with no metrics."""
        summary = analytics_system.get_summary()

        assert summary.total_queries == 0
        assert summary.unique_queries == 0
        assert summary.project_usage == {}
        assert summary.avg_confidence == 0.0
        assert summary.cache_hit_rate == 0.0

    def test_summary_basic(self, analytics_system):
        """Test basic summary statistics."""
        # Add some metrics
        for i in range(5):
            analytics_system.record_routing(
                query=f"query {i}",
                project_selected="project1",
                confidence=0.8,
                routing_time_ms=50.0,
            )

        summary = analytics_system.get_summary()

        assert summary.total_queries == 5
        assert summary.unique_queries == 5
        assert summary.project_usage["project1"] == 5
        assert summary.avg_confidence == 0.8

    def test_summary_with_duplicates(self, analytics_system):
        """Test summary with duplicate queries."""
        # Add duplicate queries
        for _ in range(3):
            analytics_system.record_routing(
                query="same query",
                project_selected="project1",
                confidence=0.9,
                routing_time_ms=50.0,
            )

        summary = analytics_system.get_summary()

        assert summary.total_queries == 3
        assert summary.unique_queries == 1

    def test_summary_multiple_projects(self, analytics_system):
        """Test summary with multiple projects."""
        analytics_system.record_routing(
            query="query1", project_selected="proj1", confidence=0.9, routing_time_ms=50.0
        )
        analytics_system.record_routing(
            query="query2", project_selected="proj2", confidence=0.8, routing_time_ms=60.0
        )
        analytics_system.record_routing(
            query="query3", project_selected="proj1", confidence=0.85, routing_time_ms=55.0
        )

        summary = analytics_system.get_summary()

        assert summary.project_usage["proj1"] == 2
        assert summary.project_usage["proj2"] == 1

    def test_summary_cache_hit_rate(self, analytics_system):
        """Test cache hit rate calculation."""
        # 3 cache hits out of 5 queries
        for i in range(5):
            analytics_system.record_routing(
                query=f"query {i}",
                project_selected="project1",
                confidence=0.9,
                routing_time_ms=50.0,
                cache_hit=(i < 3),
            )

        summary = analytics_system.get_summary()

        assert summary.cache_hit_rate == 0.6  # 3/5

    def test_summary_manual_vs_automatic(self, analytics_system):
        """Test manual vs automatic routing counts."""
        analytics_system.record_routing(
            query="auto1",
            project_selected="proj1",
            confidence=0.9,
            routing_time_ms=50.0,
            mode="automatic",
        )
        analytics_system.record_routing(
            query="auto2",
            project_selected="proj1",
            confidence=0.9,
            routing_time_ms=50.0,
            mode="automatic",
        )
        analytics_system.record_routing(
            query="manual1",
            project_selected="proj1",
            confidence=1.0,
            routing_time_ms=0.0,
            mode="manual",
        )

        summary = analytics_system.get_summary()

        assert summary.manual_vs_automatic["automatic"] == 2
        assert summary.manual_vs_automatic["manual"] == 1

    def test_summary_time_range_filter(self, analytics_system):
        """Test summary with time range filter."""
        # Add old metric
        old_time = time.time() - 7200  # 2 hours ago
        analytics_system._metrics.append(
            RoutingMetric(
                timestamp=old_time,
                query="old query",
                project_selected="proj1",
                confidence=0.8,
                routing_time_ms=50.0,
                cache_hit=False,
                mode="automatic",
            )
        )

        # Add recent metric
        analytics_system.record_routing(
            query="recent query",
            project_selected="proj1",
            confidence=0.9,
            routing_time_ms=50.0,
        )

        # Get summary for last hour
        summary = analytics_system.get_summary(time_range_hours=1.0)

        assert summary.total_queries == 1  # Only recent query


# ============================================================================
# Project Statistics Tests
# ============================================================================


class TestProjectStatistics:
    """Test project-specific statistics."""

    def test_project_stats_no_data(self, analytics_system):
        """Test project stats with no data."""
        stats = analytics_system.get_project_stats("unknown_proj")

        assert stats["count"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["avg_routing_time_ms"] == 0.0
        assert stats["cache_hit_rate"] == 0.0
        assert stats["failure_rate"] == 0.0

    def test_project_stats_basic(self, analytics_system):
        """Test basic project statistics."""
        # Add metrics for project
        for i in range(5):
            analytics_system.record_routing(
                query=f"query {i}",
                project_selected="proj1",
                confidence=0.8,
                routing_time_ms=50.0,
            )

        stats = analytics_system.get_project_stats("proj1")

        assert stats["count"] == 5
        assert stats["avg_confidence"] == 0.8
        assert stats["avg_routing_time_ms"] == 50.0

    def test_project_stats_with_cache_hits(self, analytics_system):
        """Test project stats with cache hits."""
        # 2 cache hits out of 5
        for i in range(5):
            analytics_system.record_routing(
                query=f"query {i}",
                project_selected="proj1",
                confidence=0.9,
                routing_time_ms=50.0,
                cache_hit=(i < 2),
            )

        stats = analytics_system.get_project_stats("proj1")

        assert stats["cache_hit_rate"] == 0.4  # 2/5

    def test_project_stats_with_failures(self, analytics_system):
        """Test project stats with failures."""
        # 1 failure out of 3
        for i in range(3):
            analytics_system.record_routing(
                query=f"query {i}",
                project_selected="proj1",
                confidence=0.8,
                routing_time_ms=50.0,
                success=(i != 1),
            )

        stats = analytics_system.get_project_stats("proj1")

        assert abs(stats["failure_rate"] - 1 / 3) < 0.01


# ============================================================================
# Time Series Tests
# ============================================================================


class TestTimeSeries:
    """Test time-series data generation."""

    def test_time_series_empty(self, analytics_system):
        """Test time series with no data."""
        series = analytics_system.get_time_series_data("queries", time_range_hours=24.0)

        assert len(series) == 0

    def test_time_series_queries(self, analytics_system):
        """Test time series for query count."""
        # Add metrics
        for i in range(5):
            analytics_system.record_routing(
                query=f"query {i}",
                project_selected="proj1",
                confidence=0.8,
                routing_time_ms=50.0,
            )

        series = analytics_system.get_time_series_data("queries", time_range_hours=24.0)

        assert len(series) > 0
        # Each entry is (datetime, value)
        assert all(isinstance(dt, datetime) for dt, val in series)
        assert all(isinstance(val, (int, float)) for dt, val in series)

    def test_time_series_confidence(self, analytics_system):
        """Test time series for confidence."""
        analytics_system.record_routing(
            query="query1", project_selected="proj1", confidence=0.9, routing_time_ms=50.0
        )

        series = analytics_system.get_time_series_data("confidence", time_range_hours=24.0)

        assert len(series) > 0

    def test_time_series_routing_time(self, analytics_system):
        """Test time series for routing time."""
        analytics_system.record_routing(
            query="query1", project_selected="proj1", confidence=0.9, routing_time_ms=50.0
        )

        series = analytics_system.get_time_series_data("routing_time", time_range_hours=24.0)

        assert len(series) > 0

    def test_time_series_cache_hits(self, analytics_system):
        """Test time series for cache hits."""
        analytics_system.record_routing(
            query="query1",
            project_selected="proj1",
            confidence=0.9,
            routing_time_ms=50.0,
            cache_hit=True,
        )

        series = analytics_system.get_time_series_data("cache_hits", time_range_hours=24.0)

        assert len(series) > 0


# ============================================================================
# Query Pattern Tests
# ============================================================================


class TestQueryPatterns:
    """Test query pattern analysis."""

    def test_pattern_extraction(self, analytics_system):
        """Test pattern extraction for various query types."""
        # Use base class helper
        base = RoutingTestBase()
        base.verify_pattern_extraction(
            analytics_system,
            test_cases=[
                ("What is the weather today?", "weather query"),
                ("Show MPS status", "mps query"),
                ("Check system status", "status query"),
                ("Display the data", "display query"),
            ],
        )

    def test_get_query_patterns(self, analytics_system):
        """Test getting query patterns."""
        # Add queries with patterns
        analytics_system.record_routing(
            query="What is the weather?",
            project_selected="weather_proj",
            confidence=0.9,
            routing_time_ms=50.0,
        )
        analytics_system.record_routing(
            query="Show weather data",
            project_selected="weather_proj",
            confidence=0.85,
            routing_time_ms=50.0,
        )
        analytics_system.record_routing(
            query="MPS status",
            project_selected="mps_proj",
            confidence=0.8,
            routing_time_ms=50.0,
        )

        patterns = analytics_system.get_query_patterns(limit=10)

        assert len(patterns) >= 2
        # Each pattern is (pattern, count, most_common_project, avg_confidence)
        assert all(len(p) == 4 for p in patterns)

    def test_query_patterns_sorted_by_count(self, analytics_system):
        """Test query patterns are sorted by frequency."""
        # Add more weather queries than MPS
        for _ in range(3):
            analytics_system.record_routing(
                query="weather query",
                project_selected="weather_proj",
                confidence=0.9,
                routing_time_ms=50.0,
            )

        analytics_system.record_routing(
            query="MPS status",
            project_selected="mps_proj",
            confidence=0.8,
            routing_time_ms=50.0,
        )

        patterns = analytics_system.get_query_patterns(limit=10)

        # Weather pattern should be first (more frequent)
        assert patterns[0][0] == "weather query"


# ============================================================================
# Persistence Tests
# ============================================================================


class TestPersistence:
    """Test metrics persistence."""

    def test_save_and_load_metrics(self, temp_analytics_path):
        """Test saving and loading metrics."""
        # Create system and add metrics
        system1 = RoutingAnalytics(
            max_history=100,
            enable_persistence=True,
            persistence_path=temp_analytics_path,
        )

        system1.record_routing(
            query="test query",
            project_selected="proj1",
            confidence=0.9,
            routing_time_ms=50.0,
        )

        # Create new system with same path
        system2 = RoutingAnalytics(
            max_history=100,
            enable_persistence=True,
            persistence_path=temp_analytics_path,
        )

        # Should load previous metrics
        assert len(system2._metrics) == 1
        assert system2._metrics[0].query == "test query"

    def test_persistence_disabled(self, temp_analytics_path):
        """Test system works with persistence disabled."""
        system = RoutingAnalytics(
            max_history=100,
            enable_persistence=False,
            persistence_path=temp_analytics_path,
        )

        system.record_routing(
            query="test",
            project_selected="proj1",
            confidence=0.9,
            routing_time_ms=50.0,
        )

        # Use base class helper
        base = RoutingTestBase()
        base.verify_persistence_disabled(system, temp_analytics_path)

    def test_export_metrics(self, analytics_system, tmp_path):
        """Test exporting metrics to file."""
        # Add some metrics
        analytics_system.record_routing(
            query="test query",
            project_selected="proj1",
            confidence=0.9,
            routing_time_ms=50.0,
        )

        export_path = tmp_path / "export.json"
        analytics_system.export_metrics(export_path)

        # Use base class helper
        base = RoutingTestBase()
        data = base.verify_export_success(
            export_path,
            expected_keys=["metrics", "project_stats"],
        )
        assert len(data["metrics"]) == 1

    def test_export_creates_directory(self, analytics_system, tmp_path):
        """Test export creates parent directories."""
        # Use base class helper
        base = RoutingTestBase()
        base.verify_export_creates_directory(
            analytics_system,
            tmp_path,
            export_method="export_metrics",
        )


# ============================================================================
# Clear Metrics Tests
# ============================================================================


class TestClearMetrics:
    """Test clearing metrics."""

    def test_clear_metrics(self, analytics_system):
        """Test clearing all metrics."""
        # Add some data
        analytics_system.record_routing(
            query="test",
            project_selected="proj1",
            confidence=0.9,
            routing_time_ms=50.0,
        )

        assert len(analytics_system._metrics) > 0

        # Use base class helper
        base = RoutingTestBase()
        base.verify_clear_operation(
            analytics_system,
            clear_method="clear_metrics",
            check_funcs=[
                lambda: analytics_system._metrics,
                lambda: analytics_system._project_stats,
                lambda: analytics_system._query_patterns,
            ],
        )


# ============================================================================
# Integration Tests
# ============================================================================


class TestRoutingAnalyticsIntegration:
    """Integration tests for routing analytics."""

    def test_full_analytics_cycle(self, analytics_system):
        """Test complete analytics cycle."""
        # Record various routing decisions
        analytics_system.record_routing(
            query="What is the weather?",
            project_selected="weather_proj",
            confidence=0.9,
            routing_time_ms=50.0,
            cache_hit=False,
            mode="automatic",
        )

        analytics_system.record_routing(
            query="Show MPS status",
            project_selected="mps_proj",
            confidence=0.85,
            routing_time_ms=60.0,
            cache_hit=False,
            mode="automatic",
        )

        analytics_system.record_routing(
            query="What is the weather?",  # Duplicate
            project_selected="weather_proj",
            confidence=0.95,
            routing_time_ms=5.0,
            cache_hit=True,
            mode="automatic",
        )

        # Get summary
        summary = analytics_system.get_summary()

        assert summary.total_queries == 3
        assert summary.unique_queries == 2
        assert summary.cache_hit_rate > 0
        assert len(summary.project_usage) == 2

    def test_persistence_across_sessions(self, temp_analytics_path):
        """Test metrics persist across sessions."""
        # Session 1
        system1 = RoutingAnalytics(
            max_history=100,
            enable_persistence=True,
            persistence_path=temp_analytics_path,
        )

        for i in range(5):
            system1.record_routing(
                query=f"query {i}",
                project_selected="proj1",
                confidence=0.9,
                routing_time_ms=50.0,
            )

        # Session 2
        system2 = RoutingAnalytics(
            max_history=100,
            enable_persistence=True,
            persistence_path=temp_analytics_path,
        )

        # Should have metrics from session 1
        assert len(system2._metrics) == 5
        summary = system2.get_summary()
        assert summary.total_queries == 5

    def test_multiple_projects_analytics(self, analytics_system):
        """Test analytics for multiple projects."""
        # Add metrics for different projects
        for i in range(3):
            analytics_system.record_routing(
                query=f"weather {i}",
                project_selected="weather_proj",
                confidence=0.9,
                routing_time_ms=50.0,
            )

        for i in range(2):
            analytics_system.record_routing(
                query=f"mps {i}",
                project_selected="mps_proj",
                confidence=0.85,
                routing_time_ms=60.0,
            )

        # Check project stats
        weather_stats = analytics_system.get_project_stats("weather_proj")
        mps_stats = analytics_system.get_project_stats("mps_proj")

        assert weather_stats["count"] == 3
        assert mps_stats["count"] == 2
        assert weather_stats["avg_confidence"] == 0.9
        assert mps_stats["avg_confidence"] == 0.85
