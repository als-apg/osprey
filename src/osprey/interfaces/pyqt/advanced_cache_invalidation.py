"""
Advanced Cache Invalidation for Multi-Project Router

This module provides intelligent cache invalidation strategies beyond simple TTL,
including event-driven invalidation, adaptive TTL, and probabilistic early expiration.

Key Features:
- Event-driven invalidation (config changes, capability updates)
- Adaptive TTL based on usage patterns
- Probabilistic early expiration (XFetch algorithm)
- Pattern-based invalidation
- Cache warming hints
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict

from osprey.utils.logger import get_logger

logger = get_logger("advanced_cache_invalidation")


@dataclass
class CacheEntryMetadata:
    """Metadata for cache entry to support advanced invalidation."""
    key: str
    project_id: str
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    base_ttl: float = 3600.0
    adaptive_ttl: float = 3600.0
    dependencies: Set[str] = field(default_factory=set)  # Capability names, config keys, etc.


class AdaptiveTTLStrategy:
    """
    Calculates adaptive TTL based on usage patterns.
    
    Hot entries (frequently accessed) get longer TTL.
    Cold entries (rarely accessed) get shorter TTL.
    """
    
    def __init__(
        self,
        base_ttl: float = 3600.0,
        hot_threshold: int = 100,
        warm_threshold: int = 10,
        hot_multiplier: float = 4.0,
        warm_multiplier: float = 2.0
    ):
        """Initialize adaptive TTL strategy.
        
        Args:
            base_ttl: Base TTL in seconds (default: 1 hour).
            hot_threshold: Access count threshold for hot entries.
            warm_threshold: Access count threshold for warm entries.
            hot_multiplier: TTL multiplier for hot entries.
            warm_multiplier: TTL multiplier for warm entries.
        """
        self.base_ttl = base_ttl
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold
        self.hot_multiplier = hot_multiplier
        self.warm_multiplier = warm_multiplier
        
        logger.info(
            f"Initialized AdaptiveTTLStrategy: base={base_ttl}s, "
            f"hot_threshold={hot_threshold}, warm_threshold={warm_threshold}"
        )
    
    def calculate_ttl(
        self,
        access_count: int,
        last_access: float,
        created_at: float
    ) -> float:
        """Calculate adaptive TTL based on usage patterns.
        
        Args:
            access_count: Number of times entry has been accessed.
            last_access: Timestamp of last access.
            created_at: Timestamp when entry was created.
            
        Returns:
            Adaptive TTL in seconds.
        """
        # Hot entries: frequently accessed
        if access_count >= self.hot_threshold:
            ttl = self.base_ttl * self.hot_multiplier
            logger.debug(f"Hot entry: access_count={access_count}, TTL={ttl}s")
            return ttl
        
        # Warm entries: moderately accessed
        elif access_count >= self.warm_threshold:
            ttl = self.base_ttl * self.warm_multiplier
            logger.debug(f"Warm entry: access_count={access_count}, TTL={ttl}s")
            return ttl
        
        # Cold entries: rarely accessed
        else:
            # Consider recency - recently created entries get base TTL
            age = time.time() - created_at
            if age < self.base_ttl * 0.1:  # Less than 10% of base TTL old
                return self.base_ttl
            
            # Older cold entries get reduced TTL
            ttl = self.base_ttl * 0.5
            logger.debug(f"Cold entry: access_count={access_count}, TTL={ttl}s")
            return ttl


class ProbabilisticEarlyExpiration:
    """
    Implements probabilistic early expiration to prevent cache stampede.
    
    Uses XFetch algorithm: entries are probabilistically refreshed before
    expiration to avoid thundering herd problem.
    """
    
    def __init__(self, beta: float = 1.0):
        """Initialize probabilistic early expiration.
        
        Args:
            beta: Beta parameter for XFetch algorithm (default: 1.0).
                  Higher values = more aggressive early expiration.
        """
        self.beta = beta
        logger.info(f"Initialized ProbabilisticEarlyExpiration: beta={beta}")
    
    def should_refresh_early(
        self,
        current_time: float,
        expiry_time: float,
        last_access: float
    ) -> bool:
        """Determine if entry should be refreshed before expiration.
        
        Uses XFetch algorithm to probabilistically refresh entries
        before they expire, preventing cache stampede.
        
        Args:
            current_time: Current timestamp.
            expiry_time: When entry expires.
            last_access: When entry was last accessed.
            
        Returns:
            True if entry should be refreshed early.
        """
        delta = expiry_time - current_time
        
        # Already expired
        if delta <= 0:
            return True
        
        # XFetch algorithm
        # Probability increases as we approach expiration
        try:
            # XFetch: refresh if -beta * log(random) * delta > (current_time - last_access)
            # This means entries are more likely to refresh as they approach expiration
            # and when they haven't been accessed recently
            xfetch_value = -self.beta * math.log(random.random()) * delta
            gap = current_time - last_access
            should_refresh = xfetch_value < gap
            
            if should_refresh:
                logger.debug(
                    f"Early refresh triggered: delta={delta:.1f}s, "
                    f"xfetch={xfetch_value:.3f}, gap={gap:.3f}"
                )
            
            return should_refresh
            
        except (ValueError, ZeroDivisionError):
            # Edge case: if random.random() returns 0 or other math errors
            return False


class EventDrivenInvalidator:
    """
    Handles event-driven cache invalidation.
    
    Invalidates cache entries when:
    - Project configuration changes
    - Capabilities are updated
    - Dependencies change
    """
    
    def __init__(self):
        """Initialize event-driven invalidator."""
        self.metadata: Dict[str, CacheEntryMetadata] = {}
        self.project_entries: Dict[str, Set[str]] = defaultdict(set)
        self.capability_entries: Dict[str, Set[str]] = defaultdict(set)
        
        # Event listeners
        self.invalidation_listeners: List[Callable[[str], None]] = []
        
        logger.info("Initialized EventDrivenInvalidator")
    
    def register_entry(
        self,
        cache_key: str,
        project_id: str,
        capabilities: List[str],
        base_ttl: float = 3600.0
    ):
        """Register a cache entry for event-driven invalidation.
        
        Args:
            cache_key: Cache key.
            project_id: Project ID.
            capabilities: List of capability names used.
            base_ttl: Base TTL for this entry.
        """
        metadata = CacheEntryMetadata(
            key=cache_key,
            project_id=project_id,
            base_ttl=base_ttl,
            adaptive_ttl=base_ttl,
            dependencies=set(capabilities)
        )
        
        self.metadata[cache_key] = metadata
        self.project_entries[project_id].add(cache_key)
        
        for capability in capabilities:
            self.capability_entries[capability].add(cache_key)
        
        logger.debug(
            f"Registered cache entry: {cache_key} "
            f"(project={project_id}, capabilities={len(capabilities)})"
        )
    
    def update_access(self, cache_key: str):
        """Update access metadata for a cache entry.
        
        Args:
            cache_key: Cache key.
        """
        if cache_key in self.metadata:
            metadata = self.metadata[cache_key]
            metadata.access_count += 1
            metadata.last_access = time.time()
    
    def get_metadata(self, cache_key: str) -> Optional[CacheEntryMetadata]:
        """Get metadata for a cache entry.
        
        Args:
            cache_key: Cache key.
            
        Returns:
            CacheEntryMetadata or None if not found.
        """
        return self.metadata.get(cache_key)
    
    def on_config_change(self, project_id: str) -> Set[str]:
        """Invalidate cache entries when project configuration changes.
        
        Args:
            project_id: Project ID that changed.
            
        Returns:
            Set of cache keys to invalidate.
        """
        keys_to_invalidate = self.project_entries.get(project_id, set()).copy()
        
        logger.info(
            f"Config change for project '{project_id}': "
            f"invalidating {len(keys_to_invalidate)} cache entries"
        )
        
        # Notify listeners
        for key in keys_to_invalidate:
            self._notify_invalidation(key)
        
        # Clean up metadata
        for key in keys_to_invalidate:
            self._remove_entry(key)
        
        return keys_to_invalidate
    
    def on_capability_update(self, capability_name: str) -> Set[str]:
        """Invalidate cache entries when a capability is updated.
        
        Args:
            capability_name: Name of capability that was updated.
            
        Returns:
            Set of cache keys to invalidate.
        """
        keys_to_invalidate = self.capability_entries.get(capability_name, set()).copy()
        
        logger.info(
            f"Capability '{capability_name}' updated: "
            f"invalidating {len(keys_to_invalidate)} cache entries"
        )
        
        # Notify listeners
        for key in keys_to_invalidate:
            self._notify_invalidation(key)
        
        # Clean up metadata
        for key in keys_to_invalidate:
            self._remove_entry(key)
        
        return keys_to_invalidate
    
    def invalidate_pattern(self, pattern: str) -> Set[str]:
        """Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "route:project_id:*").
            
        Returns:
            Set of cache keys invalidated.
        """
        keys_to_invalidate = set()
        
        # Simple pattern matching (could be enhanced with regex)
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            keys_to_invalidate = {
                key for key in self.metadata.keys()
                if key.startswith(prefix)
            }
        else:
            # Exact match
            if pattern in self.metadata:
                keys_to_invalidate = {pattern}
        
        logger.info(
            f"Pattern invalidation '{pattern}': "
            f"invalidating {len(keys_to_invalidate)} cache entries"
        )
        
        # Notify and clean up
        for key in keys_to_invalidate:
            self._notify_invalidation(key)
            self._remove_entry(key)
        
        return keys_to_invalidate
    
    def add_invalidation_listener(self, listener: Callable[[str], None]):
        """Add a listener for invalidation events.
        
        Args:
            listener: Callback function that receives cache key.
        """
        self.invalidation_listeners.append(listener)
    
    def _notify_invalidation(self, cache_key: str):
        """Notify listeners of cache invalidation.
        
        Args:
            cache_key: Cache key being invalidated.
        """
        for listener in self.invalidation_listeners:
            try:
                listener(cache_key)
            except Exception as e:
                logger.error(f"Error in invalidation listener: {e}")
    
    def _remove_entry(self, cache_key: str):
        """Remove entry from metadata tracking.
        
        Args:
            cache_key: Cache key to remove.
        """
        if cache_key not in self.metadata:
            return
        
        metadata = self.metadata[cache_key]
        
        # Remove from project entries
        if metadata.project_id in self.project_entries:
            self.project_entries[metadata.project_id].discard(cache_key)
        
        # Remove from capability entries
        for capability in metadata.dependencies:
            if capability in self.capability_entries:
                self.capability_entries[capability].discard(cache_key)
        
        # Remove metadata
        del self.metadata[cache_key]


class AdvancedCacheInvalidationManager:
    """
    Manages all advanced cache invalidation strategies.
    
    Combines:
    - Adaptive TTL
    - Probabilistic early expiration
    - Event-driven invalidation
    """
    
    def __init__(
        self,
        base_ttl: float = 3600.0,
        enable_adaptive_ttl: bool = True,
        enable_probabilistic_expiration: bool = True,
        enable_event_driven: bool = True
    ):
        """Initialize advanced cache invalidation manager.
        
        Args:
            base_ttl: Base TTL in seconds.
            enable_adaptive_ttl: Enable adaptive TTL strategy.
            enable_probabilistic_expiration: Enable probabilistic early expiration.
            enable_event_driven: Enable event-driven invalidation.
        """
        self.base_ttl = base_ttl
        
        # Initialize strategies
        self.adaptive_ttl = AdaptiveTTLStrategy(base_ttl=base_ttl) if enable_adaptive_ttl else None
        self.probabilistic = ProbabilisticEarlyExpiration() if enable_probabilistic_expiration else None
        self.event_driven = EventDrivenInvalidator() if enable_event_driven else None
        
        logger.info(
            f"Initialized AdvancedCacheInvalidationManager: "
            f"adaptive_ttl={enable_adaptive_ttl}, "
            f"probabilistic={enable_probabilistic_expiration}, "
            f"event_driven={enable_event_driven}"
        )
    
    def calculate_ttl(
        self,
        cache_key: str,
        access_count: int = 0,
        last_access: Optional[float] = None,
        created_at: Optional[float] = None
    ) -> float:
        """Calculate TTL for a cache entry.
        
        Args:
            cache_key: Cache key.
            access_count: Number of accesses.
            last_access: Last access timestamp.
            created_at: Creation timestamp.
            
        Returns:
            Calculated TTL in seconds.
        """
        if not self.adaptive_ttl:
            return self.base_ttl
        
        now = time.time()
        last_access = last_access or now
        created_at = created_at or now
        
        return self.adaptive_ttl.calculate_ttl(access_count, last_access, created_at)
    
    def should_refresh(
        self,
        cache_key: str,
        expiry_time: float,
        last_access: Optional[float] = None
    ) -> bool:
        """Determine if cache entry should be refreshed.
        
        Args:
            cache_key: Cache key.
            expiry_time: When entry expires.
            last_access: Last access timestamp.
            
        Returns:
            True if entry should be refreshed.
        """
        now = time.time()
        
        # Check if already expired
        if now >= expiry_time:
            return True
        
        # Check probabilistic early expiration
        if self.probabilistic:
            last_access = last_access or now
            return self.probabilistic.should_refresh_early(now, expiry_time, last_access)
        
        return False
    
    def register_cache_entry(
        self,
        cache_key: str,
        project_id: str,
        capabilities: List[str]
    ):
        """Register a cache entry for event-driven invalidation.
        
        Args:
            cache_key: Cache key.
            project_id: Project ID.
            capabilities: List of capabilities used.
        """
        if self.event_driven:
            self.event_driven.register_entry(
                cache_key,
                project_id,
                capabilities,
                self.base_ttl
            )
    
    def update_access(self, cache_key: str):
        """Update access metadata for a cache entry.
        
        Args:
            cache_key: Cache key.
        """
        if self.event_driven:
            self.event_driven.update_access(cache_key)
    
    def invalidate_project(self, project_id: str) -> Set[str]:
        """Invalidate all cache entries for a project.
        
        Args:
            project_id: Project ID.
            
        Returns:
            Set of invalidated cache keys.
        """
        if self.event_driven:
            return self.event_driven.on_config_change(project_id)
        return set()
    
    def invalidate_capability(self, capability_name: str) -> Set[str]:
        """Invalidate cache entries using a specific capability.
        
        Args:
            capability_name: Capability name.
            
        Returns:
            Set of invalidated cache keys.
        """
        if self.event_driven:
            return self.event_driven.on_capability_update(capability_name)
        return set()
    
    def invalidate_pattern(self, pattern: str) -> Set[str]:
        """Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match.
            
        Returns:
            Set of invalidated cache keys.
        """
        if self.event_driven:
            return self.event_driven.invalidate_pattern(pattern)
        return set()
    
    def add_invalidation_listener(self, listener: Callable[[str], None]):
        """Add a listener for invalidation events.
        
        Args:
            listener: Callback function.
        """
        if self.event_driven:
            self.event_driven.add_invalidation_listener(listener)