"""
Conversation Context for Multi-Project Routing

This module provides conversation context tracking to improve routing decisions
by maintaining awareness of recent queries, topics, and project usage patterns.

Key Features:
- Recent query tracking with project associations
- Topic detection and continuity analysis
- Confidence boosting based on conversation flow
- Configurable context window size
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import time
from collections import Counter

from osprey.utils.logger import get_logger

logger = get_logger("conversation_context")


@dataclass
class QueryRecord:
    """Record of a single query in conversation history."""
    query: str
    project_name: str
    confidence: float
    timestamp: float
    reasoning: str = ""


@dataclass
class TopicInfo:
    """Information about detected conversation topic."""
    topic_project: str
    confidence: float
    query_count: int
    last_updated: float


class ConversationContext:
    """
    Tracks conversation context for improved routing decisions.
    
    Maintains a sliding window of recent queries and their routing decisions,
    detects conversation topics, and provides context-aware confidence boosting.
    """
    
    def __init__(
        self,
        max_history: int = 10,
        topic_threshold: int = 2,
        topic_decay_seconds: float = 300.0,  # 5 minutes
        confidence_boost: float = 0.2
    ):
        """Initialize conversation context.
        
        Args:
            max_history: Maximum number of queries to track.
            topic_threshold: Minimum queries to establish a topic.
            topic_decay_seconds: Time before topic becomes stale.
            confidence_boost: Confidence boost for topic continuity.
        """
        self.max_history = max_history
        self.topic_threshold = topic_threshold
        self.topic_decay_seconds = topic_decay_seconds
        self.confidence_boost = confidence_boost
        
        self._history: List[QueryRecord] = []
        self._current_topic: Optional[TopicInfo] = None
        
        logger.info(
            f"Initialized ConversationContext: max_history={max_history}, "
            f"topic_threshold={topic_threshold}, decay={topic_decay_seconds}s"
        )
    
    def add_query(
        self,
        query: str,
        project_name: str,
        confidence: float,
        reasoning: str = ""
    ):
        """Add a query to conversation history.
        
        Args:
            query: User query text.
            project_name: Project that handled the query.
            confidence: Routing confidence score.
            reasoning: Routing reasoning.
        """
        record = QueryRecord(
            query=query,
            project_name=project_name,
            confidence=confidence,
            timestamp=time.time(),
            reasoning=reasoning
        )
        
        self._history.append(record)
        
        # Maintain max history size
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        # Update topic detection
        self._update_topic()
        
        logger.debug(
            f"Added query to context: '{query[:50]}...' → {project_name} "
            f"(history size: {len(self._history)})"
        )
    
    def get_recent_queries(self, count: int = 5) -> List[QueryRecord]:
        """Get most recent queries.
        
        Args:
            count: Number of recent queries to return.
            
        Returns:
            List of recent QueryRecord objects.
        """
        return self._history[-count:] if self._history else []
    
    def get_last_project(self) -> Optional[str]:
        """Get the project used in the last query.
        
        Returns:
            Project name or None if no history.
        """
        if self._history:
            return self._history[-1].project_name
        return None
    
    def has_active_topic(self) -> bool:
        """Check if there's an active conversation topic.
        
        Returns:
            True if active topic exists and hasn't decayed.
        """
        if not self._current_topic:
            return False
        
        # Check if topic has decayed
        age = time.time() - self._current_topic.last_updated
        if age > self.topic_decay_seconds:
            logger.debug(f"Topic decayed after {age:.1f}s")
            self._current_topic = None
            return False
        
        return True
    
    def get_current_topic(self) -> Optional[TopicInfo]:
        """Get current conversation topic if active.
        
        Returns:
            TopicInfo or None if no active topic.
        """
        if self.has_active_topic():
            return self._current_topic
        return None
    
    def should_boost_confidence(self, project_name: str) -> bool:
        """Check if confidence should be boosted for a project.
        
        Args:
            project_name: Project to check.
            
        Returns:
            True if project matches current topic.
        """
        if not self.has_active_topic():
            return False
        
        return self._current_topic.topic_project == project_name
    
    def get_confidence_boost(self, project_name: str) -> float:
        """Get confidence boost amount for a project.
        
        Args:
            project_name: Project to check.
            
        Returns:
            Confidence boost value (0.0 if no boost).
        """
        if self.should_boost_confidence(project_name):
            return self.confidence_boost
        return 0.0
    
    def get_context_summary(self) -> str:
        """Get human-readable context summary.
        
        Returns:
            Summary string describing current context.
        """
        if not self._history:
            return "No conversation history"
        
        summary_parts = []
        summary_parts.append(f"History: {len(self._history)} queries")
        
        if self.has_active_topic():
            topic = self._current_topic
            summary_parts.append(
                f"Active topic: {topic.topic_project} "
                f"({topic.query_count} queries, {topic.confidence:.0%} confidence)"
            )
        else:
            summary_parts.append("No active topic")
        
        last_project = self.get_last_project()
        if last_project:
            summary_parts.append(f"Last project: {last_project}")
        
        return " | ".join(summary_parts)
    
    def clear(self):
        """Clear all conversation context."""
        self._history.clear()
        self._current_topic = None
        logger.info("Conversation context cleared")
    
    def get_project_usage_stats(self) -> Dict[str, int]:
        """Get project usage statistics from history.
        
        Returns:
            Dictionary mapping project names to usage counts.
        """
        if not self._history:
            return {}
        
        project_counts = Counter(record.project_name for record in self._history)
        return dict(project_counts)
    
    # Private methods
    
    def _update_topic(self):
        """Update topic detection based on recent history."""
        if len(self._history) < self.topic_threshold:
            # Not enough history to establish topic
            self._current_topic = None
            return
        
        # Analyze recent queries (last N queries)
        recent_window = min(5, len(self._history))
        recent_queries = self._history[-recent_window:]
        
        # Count project usage in recent window
        project_counts = Counter(record.project_name for record in recent_queries)
        
        # Find dominant project
        if project_counts:
            dominant_project, count = project_counts.most_common(1)[0]
            
            # Calculate topic confidence based on consistency
            topic_confidence = count / len(recent_queries)
            
            # Establish topic if dominant enough
            if count >= self.topic_threshold:
                self._current_topic = TopicInfo(
                    topic_project=dominant_project,
                    confidence=topic_confidence,
                    query_count=count,
                    last_updated=time.time()
                )
                
                logger.debug(
                    f"Topic detected: {dominant_project} "
                    f"({count}/{len(recent_queries)} queries, "
                    f"{topic_confidence:.0%} confidence)"
                )
            else:
                # No clear dominant topic
                self._current_topic = None
    
    def get_context_for_routing(self) -> Dict:
        """Get context information for routing prompt.
        
        Returns:
            Dictionary with context information for LLM routing.
        """
        context = {
            "has_history": len(self._history) > 0,
            "recent_queries": []
        }
        
        # Add recent queries
        for record in self.get_recent_queries(3):
            context["recent_queries"].append({
                "query": record.query,
                "project": record.project_name,
                "confidence": record.confidence
            })
        
        # Add topic information
        if self.has_active_topic():
            topic = self._current_topic
            context["active_topic"] = {
                "project": topic.topic_project,
                "confidence": topic.confidence,
                "query_count": topic.query_count
            }
        
        # Add last project
        last_project = self.get_last_project()
        if last_project:
            context["last_project"] = last_project
        
        return context