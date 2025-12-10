"""
Routing Cache for Multi-Project Router

This module provides caching functionality for routing decisions to improve
performance by avoiding redundant LLM calls for similar queries.

Key Features:
- Query similarity matching using simple text comparison
- Time-based expiration with adaptive TTL
- LRU eviction for memory management
- Cache statistics tracking
- Configurable cache size and TTL
- Advanced invalidation strategies (event-driven, probabilistic)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
from collections import OrderedDict
import re

from osprey.utils.logger import get_logger
from osprey.interfaces.pyqt.advanced_cache_invalidation import (
    AdvancedCacheInvalidationManager
)

logger = get_logger("routing_cache")


@dataclass
class CachedRoutingDecision:
    """Cached routing decision with metadata."""
    project_name: str
    confidence: float
    reasoning: str
    alternative_projects: List[str]
    timestamp: float
    hit_count: int = 0
    original_query: str = ""


@dataclass
class CacheStatistics:
    """Statistics about cache performance."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_entries: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_misses / self.total_queries


class RoutingCache:
    """
    Cache for routing decisions with similarity-based matching.
    
    Uses simple text normalization and similarity to match queries.
    Implements LRU eviction and time-based expiration.
    """
    
    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 3600.0,  # 1 hour default
        similarity_threshold: float = 0.85,
        enable_advanced_invalidation: bool = True
    ):
        """Initialize routing cache.
        
        Args:
            max_size: Maximum number of entries to cache.
            ttl_seconds: Time-to-live for cache entries in seconds.
            similarity_threshold: Minimum similarity score (0-1) for cache hit.
            enable_advanced_invalidation: Enable advanced invalidation strategies.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        
        # OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CachedRoutingDecision] = OrderedDict()
        
        # Statistics
        self.stats = CacheStatistics()
        
        # Advanced invalidation manager
        self.advanced_invalidation_enabled = enable_advanced_invalidation
        if enable_advanced_invalidation:
            self.invalidation_manager = AdvancedCacheInvalidationManager(
                base_ttl=ttl_seconds,
                enable_adaptive_ttl=True,
                enable_probabilistic_expiration=True,
                enable_event_driven=True
            )
            # Register invalidation listener
            self.invalidation_manager.add_invalidation_listener(self._on_invalidation)
        else:
            self.invalidation_manager = None
        
        logger.info(
            f"Initialized RoutingCache: max_size={max_size}, "
            f"ttl={ttl_seconds}s, similarity_threshold={similarity_threshold}, "
            f"advanced_invalidation={enable_advanced_invalidation}"
        )
    
    def get(
        self,
        query: str,
        enabled_projects: List[str]
    ) -> Optional[CachedRoutingDecision]:
        """Get cached routing decision for query.
        
        Args:
            query: User query to look up.
            enabled_projects: List of currently enabled project names.
            
        Returns:
            CachedRoutingDecision if found and valid, None otherwise.
        """
        self.stats.total_queries += 1
        
        # Normalize query for comparison
        normalized_query = self._normalize_query(query)
        
        # Create cache key including enabled projects context
        projects_key = ",".join(sorted(enabled_projects))
        
        # Try exact match first
        exact_key = self._create_cache_key(normalized_query, projects_key)
        if exact_key in self._cache:
            cached = self._cache[exact_key]
            
            # Check if expired
            if self._is_expired(cached, exact_key):
                logger.debug(f"Cache entry expired for query: {query[:50]}...")
                del self._cache[exact_key]
                self.stats.cache_misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(exact_key)
            cached.hit_count += 1
            self.stats.cache_hits += 1
            
            # Update access metadata for advanced invalidation
            if self.advanced_invalidation_enabled and self.invalidation_manager:
                self.invalidation_manager.update_access(exact_key)
            
            logger.debug(
                f"Cache HIT (exact): {query[:50]}... → {cached.project_name} "
                f"(hits: {cached.hit_count})"
            )
            return cached
        
        # Try similarity matching
        similar_entry = self._find_similar_entry(
            normalized_query,
            projects_key,
            enabled_projects
        )
        
        if similar_entry:
            cached, similarity = similar_entry
            cached.hit_count += 1
            self.stats.cache_hits += 1
            
            logger.debug(
                f"Cache HIT (similar {similarity:.2f}): {query[:50]}... → "
                f"{cached.project_name} (hits: {cached.hit_count})"
            )
            return cached
        
        # Cache miss
        self.stats.cache_misses += 1
        logger.debug(f"Cache MISS: {query[:50]}...")
        return None
    
    def put(
        self,
        query: str,
        enabled_projects: List[str],
        project_name: str,
        confidence: float,
        reasoning: str,
        alternative_projects: List[str],
        capabilities: List[str] = None
    ):
        """Store routing decision in cache.
        
        Args:
            query: User query.
            enabled_projects: List of enabled project names.
            project_name: Selected project name.
            confidence: Routing confidence score.
            reasoning: Routing reasoning.
            alternative_projects: Alternative project options.
            capabilities: List of capabilities used (for advanced invalidation).
        """
        # Normalize query
        normalized_query = self._normalize_query(query)
        projects_key = ",".join(sorted(enabled_projects))
        cache_key = self._create_cache_key(normalized_query, projects_key)
        
        # Create cached decision
        cached = CachedRoutingDecision(
            project_name=project_name,
            confidence=confidence,
            reasoning=reasoning,
            alternative_projects=alternative_projects,
            timestamp=time.time(),
            hit_count=0,
            original_query=query
        )
        
        # Check if we need to evict
        if len(self._cache) >= self.max_size and cache_key not in self._cache:
            # Remove oldest entry (LRU)
            evicted_key, evicted_value = self._cache.popitem(last=False)
            self.stats.evictions += 1
            logger.debug(
                f"Cache eviction: {evicted_value.original_query[:50]}... "
                f"(hits: {evicted_value.hit_count})"
            )
        
        # Store in cache
        self._cache[cache_key] = cached
        self.stats.total_entries = len(self._cache)
        
        # Register with advanced invalidation manager
        if self.advanced_invalidation_enabled and self.invalidation_manager:
            self.invalidation_manager.register_cache_entry(
                cache_key,
                project_name,
                capabilities or []
            )
        
        logger.debug(
            f"Cache PUT: {query[:50]}... → {project_name} "
            f"(total entries: {self.stats.total_entries})"
        )
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self.stats.total_entries = 0
        logger.info("Cache cleared")
    
    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics.
        
        Returns:
            CacheStatistics object.
        """
        self.stats.total_entries = len(self._cache)
        return self.stats
    
    def remove_expired(self) -> int:
        """Remove expired entries from cache.
        
        Returns:
            Number of entries removed.
        """
        expired_keys = []
        
        for key, cached in self._cache.items():
            if self._is_expired(cached, key):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired cache entries")
        
        self.stats.total_entries = len(self._cache)
        return len(expired_keys)
    
    def invalidate_project(self, project_id: str) -> int:
        """Invalidate all cache entries for a project.
        
        Args:
            project_id: Project ID to invalidate.
            
        Returns:
            Number of entries invalidated.
        """
        if not self.advanced_invalidation_enabled or not self.invalidation_manager:
            logger.warning("Advanced invalidation not enabled")
            return 0
        
        keys_to_invalidate = self.invalidation_manager.invalidate_project(project_id)
        
        # Remove from cache
        for key in keys_to_invalidate:
            if key in self._cache:
                del self._cache[key]
        
        self.stats.total_entries = len(self._cache)
        return len(keys_to_invalidate)
    
    def invalidate_capability(self, capability_name: str) -> int:
        """Invalidate cache entries using a specific capability.
        
        Args:
            capability_name: Capability name to invalidate.
            
        Returns:
            Number of entries invalidated.
        """
        if not self.advanced_invalidation_enabled or not self.invalidation_manager:
            logger.warning("Advanced invalidation not enabled")
            return 0
        
        keys_to_invalidate = self.invalidation_manager.invalidate_capability(capability_name)
        
        # Remove from cache
        for key in keys_to_invalidate:
            if key in self._cache:
                del self._cache[key]
        
        self.stats.total_entries = len(self._cache)
        return len(keys_to_invalidate)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "route:project:*").
            
        Returns:
            Number of entries invalidated.
        """
        if not self.advanced_invalidation_enabled or not self.invalidation_manager:
            logger.warning("Advanced invalidation not enabled")
            return 0
        
        keys_to_invalidate = self.invalidation_manager.invalidate_pattern(pattern)
        
        # Remove from cache
        for key in keys_to_invalidate:
            if key in self._cache:
                del self._cache[key]
        
        self.stats.total_entries = len(self._cache)
        return len(keys_to_invalidate)
    
    # Private methods
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison.
        
        Args:
            query: Raw query string.
            
        Returns:
            Normalized query string.
        """
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation at end
        normalized = normalized.strip().rstrip('?!.,;:')
        
        return normalized
    
    def _create_cache_key(self, normalized_query: str, projects_key: str) -> str:
        """Create cache key from normalized query and projects.
        
        Args:
            normalized_query: Normalized query string.
            projects_key: Comma-separated sorted project names.
            
        Returns:
            Cache key string.
        """
        return f"{normalized_query}|{projects_key}"
    
    def _is_expired(self, cached: CachedRoutingDecision, cache_key: str = "") -> bool:
        """Check if cache entry is expired.
        
        Args:
            cached: Cached routing decision.
            cache_key: Cache key (for advanced invalidation).
            
        Returns:
            True if expired, False otherwise.
        """
        age = time.time() - cached.timestamp
        
        # Use advanced invalidation if enabled
        if self.advanced_invalidation_enabled and self.invalidation_manager and cache_key:
            # Get adaptive TTL
            metadata = self.invalidation_manager.event_driven.get_metadata(cache_key) if self.invalidation_manager.event_driven else None
            if metadata:
                ttl = metadata.adaptive_ttl
            else:
                ttl = self.ttl_seconds
            
            # Check probabilistic early expiration
            expiry_time = cached.timestamp + ttl
            return self.invalidation_manager.should_refresh(cache_key, expiry_time, cached.timestamp)
        
        # Fallback to simple TTL
        return age > self.ttl_seconds
    
    def _on_invalidation(self, cache_key: str):
        """Handle invalidation event.
        
        Args:
            cache_key: Cache key being invalidated.
        """
        logger.debug(f"Invalidation event for key: {cache_key}")
    
    def _find_similar_entry(
        self,
        normalized_query: str,
        projects_key: str,
        enabled_projects: List[str]
    ) -> Optional[Tuple[CachedRoutingDecision, float]]:
        """Find similar cache entry using text similarity.
        
        Args:
            normalized_query: Normalized query to match.
            projects_key: Projects context key.
            enabled_projects: List of enabled projects.
            
        Returns:
            Tuple of (CachedRoutingDecision, similarity_score) if found,
            None otherwise.
        """
        best_match = None
        best_similarity = 0.0
        
        for key, cached in self._cache.items():
            # Skip if expired
            if self._is_expired(cached):
                continue
            
            # Extract query and projects from key
            cached_query, cached_projects = key.split('|', 1)
            
            # Must have same enabled projects context
            if cached_projects != projects_key:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(normalized_query, cached_query)
            
            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = cached
        
        if best_match:
            return (best_match, best_similarity)
        
        return None
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries.
        
        Uses simple word-based Jaccard similarity.
        
        Args:
            query1: First query.
            query2: Second query.
            
        Returns:
            Similarity score between 0 and 1.
        """
        # Split into words
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        # Calculate Jaccard similarity
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)