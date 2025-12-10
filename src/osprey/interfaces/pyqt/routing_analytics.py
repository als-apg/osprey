"""
Routing Analytics for Multi-Project Router

This module provides comprehensive analytics and metrics tracking for routing decisions,
enabling visibility into routing patterns, performance, and system usage.

Key Features:
- Routing decision tracking
- Project usage statistics
- Performance metrics (confidence, timing, cache hits)
- Query pattern analysis
- Time-series data collection
- Dashboard data generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import time
import json
from pathlib import Path

from osprey.utils.logger import get_logger

logger = get_logger("routing_analytics")


@dataclass
class RoutingMetric:
    """Single routing decision metric."""
    timestamp: float
    query: str
    project_selected: str
    confidence: float
    routing_time_ms: float
    cache_hit: bool
    mode: str  # "automatic" or "manual"
    reasoning: str = ""
    alternative_projects: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


@dataclass
class AnalyticsSummary:
    """Summary statistics for analytics dashboard."""
    total_queries: int
    unique_queries: int
    project_usage: Dict[str, int]
    avg_confidence: float
    cache_hit_rate: float
    avg_routing_time_ms: float
    failed_routings: int
    manual_vs_automatic: Dict[str, int]
    top_query_patterns: List[Tuple[str, str, float]]  # (pattern, project, confidence)
    time_range: Tuple[datetime, datetime]


class RoutingAnalytics:
    """
    Tracks and analyzes routing decisions for insights and optimization.
    
    Provides:
    - Real-time metrics collection
    - Historical data analysis
    - Dashboard data generation
    - Performance monitoring
    - Usage pattern detection
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        enable_persistence: bool = True,
        persistence_path: Optional[Path] = None
    ):
        """Initialize routing analytics.
        
        Args:
            max_history: Maximum number of metrics to keep in memory.
            enable_persistence: Whether to persist metrics to disk.
            persistence_path: Path to save metrics (default: _agent_data/routing_analytics.json).
        """
        self.logger = logger
        self.max_history = max_history
        self.enable_persistence = enable_persistence
        
        # Set persistence path
        if persistence_path:
            self.persistence_path = persistence_path
        else:
            self.persistence_path = Path.cwd() / '_agent_data' / 'routing_analytics.json'
        
        # Metrics storage
        self._metrics: List[RoutingMetric] = []
        self._project_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_confidence': 0.0,
            'total_time_ms': 0.0,
            'cache_hits': 0,
            'failures': 0
        })
        
        # Query pattern tracking
        self._query_patterns: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        # Load persisted data if available
        if self.enable_persistence:
            self._load_metrics()
        
        self.logger.info(
            f"Initialized RoutingAnalytics "
            f"(max_history={max_history}, persistence={enable_persistence})"
        )
    
    def record_routing(
        self,
        query: str,
        project_selected: str,
        confidence: float,
        routing_time_ms: float,
        cache_hit: bool = False,
        mode: str = "automatic",
        reasoning: str = "",
        alternative_projects: List[str] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a routing decision.
        
        Args:
            query: User query.
            project_selected: Selected project name.
            confidence: Routing confidence score.
            routing_time_ms: Time taken for routing decision.
            cache_hit: Whether decision came from cache.
            mode: Routing mode ("automatic" or "manual").
            reasoning: Routing reasoning.
            alternative_projects: Alternative project options.
            success: Whether routing succeeded.
            error: Error message if failed.
        """
        metric = RoutingMetric(
            timestamp=time.time(),
            query=query,
            project_selected=project_selected,
            confidence=confidence,
            routing_time_ms=routing_time_ms,
            cache_hit=cache_hit,
            mode=mode,
            reasoning=reasoning,
            alternative_projects=alternative_projects or [],
            success=success,
            error=error
        )
        
        # Add to metrics
        self._metrics.append(metric)
        
        # Maintain max history
        if len(self._metrics) > self.max_history:
            self._metrics.pop(0)
        
        # Update project stats
        stats = self._project_stats[project_selected]
        stats['count'] += 1
        stats['total_confidence'] += confidence
        stats['total_time_ms'] += routing_time_ms
        if cache_hit:
            stats['cache_hits'] += 1
        if not success:
            stats['failures'] += 1
        
        # Track query pattern
        pattern = self._extract_pattern(query)
        self._query_patterns[pattern].append((project_selected, confidence))
        
        # Persist if enabled
        if self.enable_persistence:
            self._save_metrics()
        
        self.logger.debug(
            f"Recorded routing: {query[:50]}... → {project_selected} "
            f"(confidence: {confidence:.2f}, time: {routing_time_ms:.0f}ms)"
        )
    
    def get_summary(
        self,
        time_range_hours: Optional[float] = None
    ) -> AnalyticsSummary:
        """Get analytics summary.
        
        Args:
            time_range_hours: Optional time range in hours (None = all time).
            
        Returns:
            AnalyticsSummary with statistics.
        """
        # Filter metrics by time range
        if time_range_hours:
            cutoff_time = time.time() - (time_range_hours * 3600)
            metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
        else:
            metrics = self._metrics
        
        if not metrics:
            return AnalyticsSummary(
                total_queries=0,
                unique_queries=0,
                project_usage={},
                avg_confidence=0.0,
                cache_hit_rate=0.0,
                avg_routing_time_ms=0.0,
                failed_routings=0,
                manual_vs_automatic={'automatic': 0, 'manual': 0},
                top_query_patterns=[],
                time_range=(datetime.now(), datetime.now())
            )
        
        # Calculate statistics
        total_queries = len(metrics)
        unique_queries = len(set(m.query for m in metrics))
        
        # Project usage
        project_usage = Counter(m.project_selected for m in metrics)
        
        # Average confidence
        avg_confidence = sum(m.confidence for m in metrics) / total_queries
        
        # Cache hit rate
        cache_hits = sum(1 for m in metrics if m.cache_hit)
        cache_hit_rate = cache_hits / total_queries if total_queries > 0 else 0.0
        
        # Average routing time
        avg_routing_time = sum(m.routing_time_ms for m in metrics) / total_queries
        
        # Failed routings
        failed_routings = sum(1 for m in metrics if not m.success)
        
        # Manual vs automatic
        mode_counts = Counter(m.mode for m in metrics)
        manual_vs_automatic = {
            'automatic': mode_counts.get('automatic', 0),
            'manual': mode_counts.get('manual', 0)
        }
        
        # Top query patterns
        top_patterns = self._get_top_query_patterns(limit=10)
        
        # Time range
        timestamps = [m.timestamp for m in metrics]
        time_range = (
            datetime.fromtimestamp(min(timestamps)),
            datetime.fromtimestamp(max(timestamps))
        )
        
        return AnalyticsSummary(
            total_queries=total_queries,
            unique_queries=unique_queries,
            project_usage=dict(project_usage),
            avg_confidence=avg_confidence,
            cache_hit_rate=cache_hit_rate,
            avg_routing_time_ms=avg_routing_time,
            failed_routings=failed_routings,
            manual_vs_automatic=manual_vs_automatic,
            top_query_patterns=top_patterns,
            time_range=time_range
        )
    
    def get_project_stats(self, project_name: str) -> Dict[str, Any]:
        """Get statistics for a specific project.
        
        Args:
            project_name: Name of project.
            
        Returns:
            Dictionary with project statistics.
        """
        stats = self._project_stats.get(project_name)
        if not stats or stats['count'] == 0:
            return {
                'count': 0,
                'avg_confidence': 0.0,
                'avg_routing_time_ms': 0.0,
                'cache_hit_rate': 0.0,
                'failure_rate': 0.0
            }
        
        return {
            'count': stats['count'],
            'avg_confidence': stats['total_confidence'] / stats['count'],
            'avg_routing_time_ms': stats['total_time_ms'] / stats['count'],
            'cache_hit_rate': stats['cache_hits'] / stats['count'],
            'failure_rate': stats['failures'] / stats['count']
        }
    
    def get_time_series_data(
        self,
        metric_name: str,
        time_range_hours: float = 24.0,
        bucket_size_minutes: int = 60
    ) -> List[Tuple[datetime, float]]:
        """Get time-series data for a metric.
        
        Args:
            metric_name: Name of metric ('queries', 'confidence', 'routing_time', 'cache_hits').
            time_range_hours: Time range in hours.
            bucket_size_minutes: Size of time buckets in minutes.
            
        Returns:
            List of (timestamp, value) tuples.
        """
        cutoff_time = time.time() - (time_range_hours * 3600)
        metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return []
        
        # Create time buckets
        bucket_size_seconds = bucket_size_minutes * 60
        min_time = min(m.timestamp for m in metrics)
        max_time = max(m.timestamp for m in metrics)
        
        buckets = defaultdict(list)
        
        # Assign metrics to buckets
        for metric in metrics:
            bucket_key = int((metric.timestamp - min_time) / bucket_size_seconds)
            buckets[bucket_key].append(metric)
        
        # Calculate values for each bucket
        time_series = []
        for bucket_key in sorted(buckets.keys()):
            bucket_metrics = buckets[bucket_key]
            bucket_time = datetime.fromtimestamp(
                min_time + (bucket_key * bucket_size_seconds)
            )
            
            if metric_name == 'queries':
                value = len(bucket_metrics)
            elif metric_name == 'confidence':
                value = sum(m.confidence for m in bucket_metrics) / len(bucket_metrics)
            elif metric_name == 'routing_time':
                value = sum(m.routing_time_ms for m in bucket_metrics) / len(bucket_metrics)
            elif metric_name == 'cache_hits':
                value = sum(1 for m in bucket_metrics if m.cache_hit) / len(bucket_metrics)
            else:
                value = 0.0
            
            time_series.append((bucket_time, value))
        
        return time_series
    
    def get_query_patterns(self, limit: int = 20) -> List[Tuple[str, int, str, float]]:
        """Get common query patterns.
        
        Args:
            limit: Maximum number of patterns to return.
            
        Returns:
            List of (pattern, count, most_common_project, avg_confidence) tuples.
        """
        pattern_stats = []
        
        for pattern, project_confidence_list in self._query_patterns.items():
            count = len(project_confidence_list)
            
            # Find most common project
            projects = [p for p, c in project_confidence_list]
            most_common_project = Counter(projects).most_common(1)[0][0]
            
            # Calculate average confidence for this pattern
            confidences = [c for p, c in project_confidence_list]
            avg_confidence = sum(confidences) / len(confidences)
            
            pattern_stats.append((pattern, count, most_common_project, avg_confidence))
        
        # Sort by count (descending)
        pattern_stats.sort(key=lambda x: x[1], reverse=True)
        
        return pattern_stats[:limit]
    
    def clear_metrics(self):
        """Clear all metrics."""
        self._metrics.clear()
        self._project_stats.clear()
        self._query_patterns.clear()
        
        if self.enable_persistence:
            self._save_metrics()
        
        self.logger.info("Cleared all routing metrics")
    
    def export_metrics(self, filepath: Path) -> bool:
        """Export metrics to JSON file.
        
        Args:
            filepath: Path to export file.
            
        Returns:
            True if successful.
        """
        try:
            data = {
                'metrics': [
                    {
                        'timestamp': m.timestamp,
                        'query': m.query,
                        'project_selected': m.project_selected,
                        'confidence': m.confidence,
                        'routing_time_ms': m.routing_time_ms,
                        'cache_hit': m.cache_hit,
                        'mode': m.mode,
                        'reasoning': m.reasoning,
                        'alternative_projects': m.alternative_projects,
                        'success': m.success,
                        'error': m.error
                    }
                    for m in self._metrics
                ],
                'project_stats': dict(self._project_stats),
                'exported_at': datetime.now().isoformat()
            }
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Exported {len(self._metrics)} metrics to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    # Private methods
    
    def _extract_pattern(self, query: str) -> str:
        """Extract query pattern for grouping.
        
        Args:
            query: User query.
            
        Returns:
            Pattern string.
        """
        # Simple pattern extraction - could be enhanced with NLP
        query_lower = query.lower()
        
        # Common patterns
        if 'weather' in query_lower:
            return "weather query"
        elif 'mps' in query_lower or 'machine protection' in query_lower:
            return "mps query"
        elif 'status' in query_lower:
            return "status query"
        elif 'show' in query_lower or 'display' in query_lower:
            return "display query"
        elif 'get' in query_lower or 'fetch' in query_lower:
            return "data retrieval"
        else:
            # Use first few words as pattern
            words = query_lower.split()[:3]
            return ' '.join(words) if words else "other"
    
    def _get_top_query_patterns(self, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Get top query patterns with their most common project and confidence.
        
        Args:
            limit: Maximum number of patterns.
            
        Returns:
            List of (pattern, project, avg_confidence) tuples.
        """
        pattern_stats = []
        
        for pattern, project_confidence_list in self._query_patterns.items():
            # Find most common project
            projects = [p for p, c in project_confidence_list]
            if not projects:
                continue
            
            most_common_project = Counter(projects).most_common(1)[0][0]
            
            # Calculate average confidence
            confidences = [c for p, c in project_confidence_list]
            avg_confidence = sum(confidences) / len(confidences)
            
            pattern_stats.append((pattern, most_common_project, avg_confidence))
        
        # Sort by frequency (number of occurrences)
        pattern_counts = {
            pattern: len(self._query_patterns[pattern])
            for pattern in self._query_patterns
        }
        pattern_stats.sort(key=lambda x: pattern_counts[x[0]], reverse=True)
        
        return pattern_stats[:limit]
    
    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save only recent metrics to avoid large files
            recent_metrics = self._metrics[-self.max_history:]
            
            data = {
                'metrics': [
                    {
                        'timestamp': m.timestamp,
                        'query': m.query,
                        'project_selected': m.project_selected,
                        'confidence': m.confidence,
                        'routing_time_ms': m.routing_time_ms,
                        'cache_hit': m.cache_hit,
                        'mode': m.mode,
                        'reasoning': m.reasoning,
                        'alternative_projects': m.alternative_projects,
                        'success': m.success,
                        'error': m.error
                    }
                    for m in recent_metrics
                ],
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            self.logger.warning(f"Failed to save metrics: {e}")
    
    def _load_metrics(self):
        """Load metrics from disk."""
        try:
            if not self.persistence_path.exists():
                return
            
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
            
            # Load metrics
            for m_data in data.get('metrics', []):
                metric = RoutingMetric(
                    timestamp=m_data['timestamp'],
                    query=m_data['query'],
                    project_selected=m_data['project_selected'],
                    confidence=m_data['confidence'],
                    routing_time_ms=m_data['routing_time_ms'],
                    cache_hit=m_data['cache_hit'],
                    mode=m_data['mode'],
                    reasoning=m_data.get('reasoning', ''),
                    alternative_projects=m_data.get('alternative_projects', []),
                    success=m_data.get('success', True),
                    error=m_data.get('error')
                )
                self._metrics.append(metric)
                
                # Rebuild project stats
                stats = self._project_stats[metric.project_selected]
                stats['count'] += 1
                stats['total_confidence'] += metric.confidence
                stats['total_time_ms'] += metric.routing_time_ms
                if metric.cache_hit:
                    stats['cache_hits'] += 1
                if not metric.success:
                    stats['failures'] += 1
                
                # Rebuild query patterns
                pattern = self._extract_pattern(metric.query)
                self._query_patterns[pattern].append(
                    (metric.project_selected, metric.confidence)
                )
            
            self.logger.info(f"Loaded {len(self._metrics)} metrics from {self.persistence_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load metrics: {e}")