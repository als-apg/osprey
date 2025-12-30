"""
Tests for Routing Cache

Tests caching functionality, similarity matching, TTL expiration,
LRU eviction, and advanced invalidation strategies.
"""

# Import from local module
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from osprey.interfaces.pyqt.routing_cache import (
    CachedRoutingDecision,
    CacheStatistics,
    RoutingCache,
)

sys.path.insert(0, str(Path(__file__).parent))
from routing_test_base import RoutingTestBase

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cache():
    """Create routing cache with default settings."""
    return RoutingCache(
        max_size=10,
        ttl_seconds=60.0,
        similarity_threshold=0.85,
        enable_advanced_invalidation=False,  # Disable for simpler tests
    )


@pytest.fixture
def cache_with_advanced():
    """Create routing cache with advanced invalidation."""
    with patch("osprey.interfaces.pyqt.routing_cache.AdvancedCacheInvalidationManager"):
        return RoutingCache(
            max_size=10,
            ttl_seconds=60.0,
            similarity_threshold=0.85,
            enable_advanced_invalidation=True,
        )


# ============================================================================
# CachedRoutingDecision Tests
# ============================================================================


class TestCachedRoutingDecision:
    """Test CachedRoutingDecision dataclass."""

    def test_create_basic_decision(self):
        """Test creating basic cached decision."""
        decision = CachedRoutingDecision(
            project_name="project1",
            confidence=0.9,
            reasoning="High match",
            alternative_projects=["project2"],
            timestamp=time.time(),
        )

        assert decision.project_name == "project1"
        assert decision.confidence == 0.9
        assert decision.reasoning == "High match"
        assert decision.hit_count == 0

    def test_increment_hit_count(self):
        """Test incrementing hit count."""
        decision = CachedRoutingDecision(
            project_name="project1",
            confidence=0.9,
            reasoning="test",
            alternative_projects=[],
            timestamp=time.time(),
        )

        decision.hit_count += 1
        assert decision.hit_count == 1


# ============================================================================
# CacheStatistics Tests
# ============================================================================


class TestCacheStatistics:
    """Test CacheStatistics dataclass."""

    def test_create_statistics(self):
        """Test creating cache statistics."""
        stats = CacheStatistics()

        assert stats.total_queries == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStatistics(total_queries=10, cache_hits=7, cache_misses=3)

        assert stats.hit_rate == 0.7
        assert stats.miss_rate == 0.3

    def test_zero_queries(self):
        """Test statistics with zero queries."""
        stats = CacheStatistics(total_queries=0)

        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 0.0


# ============================================================================
# Cache Initialization Tests
# ============================================================================


class TestCacheInit:
    """Test cache initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        cache = RoutingCache()

        assert cache.max_size == 100
        assert cache.ttl_seconds == 3600.0
        assert cache.similarity_threshold == 0.85

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        cache = RoutingCache(
            max_size=50,
            ttl_seconds=1800.0,
            similarity_threshold=0.9,
            enable_advanced_invalidation=False,
        )

        assert cache.max_size == 50
        assert cache.ttl_seconds == 1800.0
        assert cache.similarity_threshold == 0.9


# ============================================================================
# Cache Get/Put Tests
# ============================================================================


class TestCacheGetPut:
    """Test cache get and put operations."""

    def test_put_and_get_exact_match(self, cache):
        """Test putting and getting with exact match."""
        cache.put(
            query="What is the weather?",
            enabled_projects=["project1", "project2"],
            project_name="project1",
            confidence=0.9,
            reasoning="Weather query",
            alternative_projects=["project2"],
        )

        result = cache.get("What is the weather?", ["project1", "project2"])

        assert result is not None
        assert result.project_name == "project1"
        assert result.confidence == 0.9
        assert result.hit_count == 1

    def test_get_miss(self, cache):
        """Test cache miss."""
        result = cache.get("unknown query", ["project1"])

        assert result is None
        assert cache.stats.cache_misses == 1

    def test_get_similar_match(self, cache):
        """Test getting with similar query."""
        cache.put(
            query="what is the weather",
            enabled_projects=["project1"],
            project_name="project1",
            confidence=0.9,
            reasoning="test",
            alternative_projects=[],
        )

        # Similar query (different case, punctuation)
        result = cache.get("What is the weather?", ["project1"])

        assert result is not None
        assert result.project_name == "project1"

    def test_different_enabled_projects(self, cache):
        """Test cache miss with different enabled projects."""
        cache.put(
            query="test query",
            enabled_projects=["project1"],
            project_name="project1",
            confidence=0.9,
            reasoning="test",
            alternative_projects=[],
        )

        # Same query but different enabled projects
        result = cache.get("test query", ["project1", "project2"])

        assert result is None  # Should miss due to different context


# ============================================================================
# Query Normalization Tests
# ============================================================================


class TestQueryNormalization:
    """Test query normalization."""

    def test_normalize_case(self, cache):
        """Test case normalization."""
        normalized = cache._normalize_query("What Is The Weather?")
        assert normalized == "what is the weather"

    def test_normalize_whitespace(self, cache):
        """Test whitespace normalization."""
        normalized = cache._normalize_query("what  is   the    weather")
        assert normalized == "what is the weather"

    def test_normalize_punctuation(self, cache):
        """Test punctuation removal."""
        normalized = cache._normalize_query("what is the weather?!.,;:")
        assert normalized == "what is the weather"


# ============================================================================
# Similarity Calculation Tests
# ============================================================================


class TestSimilarityCalculation:
    """Test similarity calculation."""

    def test_identical_queries(self, cache):
        """Test similarity of identical queries."""
        similarity = cache._calculate_similarity("test query", "test query")
        assert similarity == 1.0

    def test_completely_different(self, cache):
        """Test similarity of completely different queries."""
        similarity = cache._calculate_similarity("weather forecast", "mps status")
        assert similarity < 0.5

    def test_partial_overlap(self, cache):
        """Test similarity with partial overlap."""
        similarity = cache._calculate_similarity("what is the weather", "what is the temperature")
        assert 0.0 < similarity < 1.0

    def test_empty_queries(self, cache):
        """Test similarity of empty queries."""
        similarity = cache._calculate_similarity("", "")
        assert similarity == 1.0


# ============================================================================
# TTL Expiration Tests
# ============================================================================


class TestTTLExpiration:
    """Test TTL-based expiration."""

    def test_expired_entry(self):
        """Test expired cache entry."""
        cache = RoutingCache(max_size=10, ttl_seconds=0.1, enable_advanced_invalidation=False)

        cache.put(
            query="test",
            enabled_projects=["project1"],
            project_name="project1",
            confidence=0.9,
            reasoning="test",
            alternative_projects=[],
        )

        # Wait for expiration
        time.sleep(0.2)

        result = cache.get("test", ["project1"])
        assert result is None  # Should be expired

    def test_not_expired(self, cache):
        """Test non-expired entry."""
        cache.put(
            query="test",
            enabled_projects=["project1"],
            project_name="project1",
            confidence=0.9,
            reasoning="test",
            alternative_projects=[],
        )

        result = cache.get("test", ["project1"])
        assert result is not None

    def test_remove_expired(self):
        """Test removing expired entries."""
        cache = RoutingCache(max_size=10, ttl_seconds=0.1, enable_advanced_invalidation=False)

        # Add entries
        for i in range(5):
            cache.put(
                query=f"query{i}",
                enabled_projects=["project1"],
                project_name="project1",
                confidence=0.9,
                reasoning="test",
                alternative_projects=[],
            )

        # Wait for expiration
        time.sleep(0.2)

        removed = cache.remove_expired()
        assert removed == 5


# ============================================================================
# LRU Eviction Tests
# ============================================================================


class TestLRUEviction:
    """Test LRU eviction."""

    def test_eviction_on_max_size(self, cache):
        """Test eviction when max size reached."""
        # Fill cache to max
        for i in range(10):
            cache.put(
                query=f"query{i}",
                enabled_projects=["project1"],
                project_name="project1",
                confidence=0.9,
                reasoning="test",
                alternative_projects=[],
            )

        assert len(cache._cache) == 10
        assert cache.stats.evictions == 0

        # Add one more - should evict oldest
        cache.put(
            query="query10",
            enabled_projects=["project1"],
            project_name="project1",
            confidence=0.9,
            reasoning="test",
            alternative_projects=[],
        )

        assert len(cache._cache) == 10
        assert cache.stats.evictions == 1

    def test_lru_ordering(self, cache):
        """Test LRU ordering on access."""
        # Add entries
        cache.put(
            query="query1",
            enabled_projects=["project1"],
            project_name="project1",
            confidence=0.9,
            reasoning="test",
            alternative_projects=[],
        )
        cache.put(
            query="query2",
            enabled_projects=["project1"],
            project_name="project1",
            confidence=0.9,
            reasoning="test",
            alternative_projects=[],
        )

        # Access query1 (should move to end)
        cache.get("query1", ["project1"])

        # Fill cache and trigger eviction
        for i in range(3, 12):
            cache.put(
                query=f"query{i}",
                enabled_projects=["project1"],
                project_name="project1",
                confidence=0.9,
                reasoning="test",
                alternative_projects=[],
            )

        # query1 should still be in cache (was accessed recently)
        # query2 should have been evicted (oldest)
        assert cache.get("query1", ["project1"]) is not None


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Test cache statistics."""

    def test_hit_statistics(self, cache):
        """Test hit statistics tracking."""
        cache.put(
            query="test",
            enabled_projects=["project1"],
            project_name="project1",
            confidence=0.9,
            reasoning="test",
            alternative_projects=[],
        )

        # Hit
        cache.get("test", ["project1"])
        assert cache.stats.cache_hits == 1
        assert cache.stats.total_queries == 1

        # Miss
        cache.get("unknown", ["project1"])
        assert cache.stats.cache_misses == 1
        assert cache.stats.total_queries == 2

    def test_get_statistics(self, cache):
        """Test getting statistics."""
        stats = cache.get_statistics()

        assert isinstance(stats, CacheStatistics)
        assert stats.total_entries == 0


# ============================================================================
# Clear Cache Tests
# ============================================================================


class TestClearCache:
    """Test clearing cache."""

    def test_clear(self, cache):
        """Test clearing cache."""
        # Add entries
        for i in range(5):
            cache.put(
                query=f"query{i}",
                enabled_projects=["project1"],
                project_name="project1",
                confidence=0.9,
                reasoning="test",
                alternative_projects=[],
            )

        assert len(cache._cache) > 0

        # Use base class helper
        base = RoutingTestBase()
        base.verify_clear_operation(
            cache,
            clear_method="clear",
            check_funcs=[
                lambda: cache._cache,
                lambda: cache.stats.total_entries,
            ],
        )


# ============================================================================
# Advanced Invalidation Tests
# ============================================================================


class TestAdvancedInvalidation:
    """Test advanced invalidation features."""

    def test_invalidate_project(self, cache_with_advanced):
        """Test invalidating by project."""
        # Mock the invalidation manager
        cache_with_advanced.invalidation_manager.invalidate_project = Mock(
            return_value=["key1", "key2"]
        )

        # Add some entries to cache
        cache_with_advanced._cache["key1"] = Mock()
        cache_with_advanced._cache["key2"] = Mock()

        count = cache_with_advanced.invalidate_project("project1")

        assert count == 2
        assert "key1" not in cache_with_advanced._cache
        assert "key2" not in cache_with_advanced._cache

    def test_invalidate_capability(self, cache_with_advanced):
        """Test invalidating by capability."""
        cache_with_advanced.invalidation_manager.invalidate_capability = Mock(return_value=["key1"])

        cache_with_advanced._cache["key1"] = Mock()

        count = cache_with_advanced.invalidate_capability("capability1")

        assert count == 1
        assert "key1" not in cache_with_advanced._cache

    def test_invalidate_pattern(self, cache_with_advanced):
        """Test invalidating by pattern."""
        cache_with_advanced.invalidation_manager.invalidate_pattern = Mock(
            return_value=["key1", "key2"]
        )

        cache_with_advanced._cache["key1"] = Mock()
        cache_with_advanced._cache["key2"] = Mock()

        count = cache_with_advanced.invalidate_pattern("route:*")

        assert count == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestCacheIntegration:
    """Integration tests for routing cache."""

    def test_full_cache_cycle(self, cache):
        """Test complete cache lifecycle."""
        # Put
        cache.put(
            query="What is the weather?",
            enabled_projects=["weather", "general"],
            project_name="weather",
            confidence=0.95,
            reasoning="Weather-specific query",
            alternative_projects=["general"],
        )

        # Get exact match
        result = cache.get("What is the weather?", ["weather", "general"])
        assert result is not None
        assert result.project_name == "weather"
        assert result.hit_count == 1

        # Get similar match
        result = cache.get("what is the weather", ["weather", "general"])
        assert result is not None
        assert result.hit_count == 2

        # Get with different context (miss)
        result = cache.get("What is the weather?", ["weather"])
        assert result is None

    def test_cache_performance(self, cache):
        """Test cache performance with many entries."""
        # Add many entries
        for i in range(100):
            cache.put(
                query=f"query {i}",
                enabled_projects=["project1"],
                project_name="project1",
                confidence=0.9,
                reasoning="test",
                alternative_projects=[],
            )

        # Cache should maintain max size
        assert len(cache._cache) == cache.max_size

        # Should still be able to get recent entries
        result = cache.get("query 99", ["project1"])
        assert result is not None

    def test_mixed_operations(self, cache):
        """Test mixed cache operations."""
        # Add entries
        cache.put("query1", ["proj1"], "proj1", 0.9, "test", [])
        cache.put("query2", ["proj1"], "proj1", 0.8, "test", [])

        # Get some
        cache.get("query1", ["proj1"])
        cache.get("unknown", ["proj1"])

        # Clear
        cache.clear()

        # Verify cleared
        assert len(cache._cache) == 0

        # Can still add after clear
        cache.put("query3", ["proj1"], "proj1", 0.9, "test", [])
        assert len(cache._cache) == 1
