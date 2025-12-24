"""
Automated Feedback Integration for Multi-Project Router

This module provides automated integration of user feedback into routing decisions,
creating a continuous improvement loop through feedback-based adjustments and
model retraining.

Key Features:
- Automatic feedback-based route adjustment
- Feedback-driven confidence calibration
- Root cause analysis of negative feedback
- Sentiment trend analysis
- Pattern learning from corrections
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter

from osprey.utils.logger import get_logger

logger = get_logger("automated_feedback_integration")


@dataclass
class FeedbackAdjustment:
    """Feedback-based routing adjustment."""
    original_project: str
    adjusted_project: str
    confidence_adjustment: float
    reasoning: str
    feedback_count: int
    last_updated: float


@dataclass
class FeedbackPattern:
    """Learned pattern from user feedback."""
    pattern_id: str
    query_pattern: str
    correct_project: str
    confidence: float
    feedback_instances: int
    success_rate: float
    last_seen: float


@dataclass
class SentimentTrend:
    """Sentiment trend for a project."""
    project_id: str
    positive_count: int
    negative_count: int
    trend_direction: str  # "improving", "degrading", "stable"
    confidence: float
    period_start: float
    period_end: float


class FeedbackBasedRouteAdjuster:
    """
    Adjusts routing decisions based on accumulated user feedback.
    
    Provides:
    - Automatic route corrections based on feedback patterns
    - Confidence calibration from feedback history
    - Learning from repeated corrections
    """
    
    def __init__(
        self,
        min_feedback_threshold: int = 3,
        confidence_boost_factor: float = 0.15,
        confidence_penalty_factor: float = 0.10
    ):
        """Initialize route adjuster.
        
        Args:
            min_feedback_threshold: Minimum feedback instances to apply adjustment.
            confidence_boost_factor: Confidence boost for positive feedback.
            confidence_penalty_factor: Confidence penalty for negative feedback.
        """
        self.min_feedback_threshold = min_feedback_threshold
        self.confidence_boost_factor = confidence_boost_factor
        self.confidence_penalty_factor = confidence_penalty_factor
        
        # Learned adjustments
        self.adjustments: Dict[str, FeedbackAdjustment] = {}
        
        # Query-to-correction mapping
        self.corrections: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        # query -> [(correct_project, timestamp, confidence)]
        
        logger.info(
            f"Initialized FeedbackBasedRouteAdjuster: "
            f"threshold={min_feedback_threshold}"
        )
    
    def record_feedback(
        self,
        query: str,
        selected_project: str,
        user_feedback: str,
        correct_project: Optional[str] = None
    ):
        """Record user feedback for a routing decision.
        
        Args:
            query: User query.
            selected_project: Project that was selected.
            user_feedback: "correct" or "incorrect".
            correct_project: Correct project if feedback was incorrect.
        """
        if user_feedback == "incorrect" and correct_project:
            # Record correction
            self.corrections[query].append((
                correct_project,
                time.time(),
                1.0  # Initial confidence
            ))
            
            # Update adjustment if threshold met
            if len(self.corrections[query]) >= self.min_feedback_threshold:
                self._update_adjustment(query, selected_project, correct_project)
            
            logger.info(
                f"Recorded correction: {query[:50]}... "
                f"{selected_project} → {correct_project}"
            )
    
    def get_adjustment(
        self,
        query: str,
        base_project: str,
        base_confidence: float
    ) -> Tuple[str, float, str]:
        """Get routing adjustment based on feedback.
        
        Args:
            query: User query.
            base_project: Base routing decision project.
            base_confidence: Base routing confidence.
            
        Returns:
            Tuple of (adjusted_project, adjusted_confidence, reasoning).
        """
        # Check for exact query match
        if query in self.adjustments:
            adjustment = self.adjustments[query]
            
            if adjustment.feedback_count >= self.min_feedback_threshold:
                return (
                    adjustment.adjusted_project,
                    min(0.95, base_confidence + adjustment.confidence_adjustment),
                    adjustment.reasoning
                )
        
        # Check for similar query corrections
        similar_adjustment = self._find_similar_adjustment(query, base_project)
        if similar_adjustment:
            return similar_adjustment
        
        # No adjustment
        return (base_project, base_confidence, "")
    
    def get_confidence_calibration(
        self,
        project: str,
        base_confidence: float,
        feedback_history: List[str]
    ) -> float:
        """Calibrate confidence based on feedback history.
        
        Args:
            project: Project name.
            base_confidence: Base confidence score.
            feedback_history: List of recent feedback ("correct"/"incorrect").
            
        Returns:
            Calibrated confidence score.
        """
        if not feedback_history:
            return base_confidence
        
        # Calculate feedback ratio
        positive = sum(1 for f in feedback_history if f == "correct")
        total = len(feedback_history)
        feedback_ratio = positive / total
        
        # Adjust confidence
        if feedback_ratio > 0.8:
            # High success rate - boost confidence
            adjustment = self.confidence_boost_factor * (feedback_ratio - 0.8) / 0.2
            calibrated = min(1.0, base_confidence + adjustment)
        elif feedback_ratio < 0.5:
            # Low success rate - reduce confidence
            adjustment = self.confidence_penalty_factor * (0.5 - feedback_ratio) / 0.5
            calibrated = max(0.0, base_confidence - adjustment)
        else:
            # Moderate success rate - no adjustment
            calibrated = base_confidence
        
        logger.debug(
            f"Calibrated confidence for {project}: "
            f"{base_confidence:.2f} → {calibrated:.2f} "
            f"(feedback ratio: {feedback_ratio:.2f})"
        )
        
        return calibrated
    
    # Private methods
    
    def _update_adjustment(
        self,
        query: str,
        wrong_project: str,
        correct_project: str
    ):
        """Update routing adjustment for a query.
        
        Args:
            query: User query.
            wrong_project: Incorrectly selected project.
            correct_project: Correct project.
        """
        corrections = self.corrections[query]
        
        # Find most common correction
        projects = [p for p, t, c in corrections]
        most_common = Counter(projects).most_common(1)[0]
        
        if most_common[0] == correct_project:
            # Create or update adjustment
            self.adjustments[query] = FeedbackAdjustment(
                original_project=wrong_project,
                adjusted_project=correct_project,
                confidence_adjustment=0.20,
                reasoning=f"Learned from {len(corrections)} user correction(s)",
                feedback_count=len(corrections),
                last_updated=time.time()
            )
            
            logger.info(
                f"Updated adjustment for '{query[:50]}...': "
                f"{wrong_project} → {correct_project}"
            )
    
    def _find_similar_adjustment(
        self,
        query: str,
        base_project: str
    ) -> Optional[Tuple[str, float, str]]:
        """Find adjustment from similar queries.
        
        Args:
            query: User query.
            base_project: Base project.
            
        Returns:
            Tuple of (project, confidence, reasoning) or None.
        """
        query_words = set(query.lower().split())
        
        best_match = None
        best_similarity = 0.0
        
        for adj_query, adjustment in self.adjustments.items():
            if adjustment.feedback_count < self.min_feedback_threshold:
                continue
            
            # Calculate word overlap
            adj_words = set(adj_query.lower().split())
            overlap = query_words & adj_words
            similarity = len(overlap) / max(len(query_words), len(adj_words))
            
            if similarity > best_similarity and similarity > 0.6:
                best_similarity = similarity
                best_match = (
                    adjustment.adjusted_project,
                    0.85 * similarity,
                    f"Similar to corrected query (similarity: {similarity:.0%})"
                )
        
        return best_match


class RootCauseAnalyzer:
    """
    Analyzes root causes of negative feedback.
    
    Categorizes feedback issues and identifies patterns
    for targeted improvements.
    """
    
    # Issue categories
    CATEGORIES = {
        'incorrect_routing': ['wrong', 'incorrect', 'mistake', 'error'],
        'performance': ['slow', 'timeout', 'wait', 'delay'],
        'incomplete_response': ['missing', 'incomplete', 'partial'],
        'capability_mismatch': ['cannot', 'unable', 'not supported'],
        'other': []
    }
    
    def __init__(self):
        """Initialize root cause analyzer."""
        self.issue_counts: Dict[str, int] = defaultdict(int)
        self.project_issues: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        logger.info("Initialized RootCauseAnalyzer")
    
    def analyze_feedback(
        self,
        query: str,
        project: str,
        feedback_comment: str
    ) -> str:
        """Analyze negative feedback to determine root cause.
        
        Args:
            query: User query.
            project: Selected project.
            feedback_comment: User's feedback comment.
            
        Returns:
            Category string.
        """
        comment_lower = feedback_comment.lower()
        
        # Categorize based on keywords
        for category, keywords in self.CATEGORIES.items():
            if any(keyword in comment_lower for keyword in keywords):
                self.issue_counts[category] += 1
                self.project_issues[project][category] += 1
                
                logger.info(
                    f"Categorized feedback as '{category}' for project {project}"
                )
                
                return category
        
        # Default to 'other'
        self.issue_counts['other'] += 1
        self.project_issues[project]['other'] += 1
        return 'other'
    
    def get_top_issues(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get top issues across all projects.
        
        Args:
            limit: Maximum number of issues to return.
            
        Returns:
            List of (category, count) tuples.
        """
        return Counter(self.issue_counts).most_common(limit)
    
    def get_project_issues(self, project: str) -> Dict[str, int]:
        """Get issues for a specific project.
        
        Args:
            project: Project name.
            
        Returns:
            Dictionary of issue categories and counts.
        """
        return dict(self.project_issues.get(project, {}))


class SentimentTrendAnalyzer:
    """
    Analyzes sentiment trends over time.
    
    Tracks positive/negative feedback trends to identify
    improving or degrading project performance.
    """
    
    def __init__(self, trend_window_days: int = 7):
        """Initialize sentiment trend analyzer.
        
        Args:
            trend_window_days: Number of days for trend analysis.
        """
        self.trend_window_days = trend_window_days
        self.feedback_history: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        # project -> [(feedback_type, timestamp)]
        
        logger.info(
            f"Initialized SentimentTrendAnalyzer: "
            f"window={trend_window_days} days"
        )
    
    def record_feedback(self, project: str, feedback_type: str):
        """Record feedback for trend analysis.
        
        Args:
            project: Project name.
            feedback_type: "correct" or "incorrect".
        """
        self.feedback_history[project].append((feedback_type, time.time()))
        
        # Prune old feedback
        cutoff = time.time() - (self.trend_window_days * 86400)
        self.feedback_history[project] = [
            (f, t) for f, t in self.feedback_history[project]
            if t >= cutoff
        ]
    
    def get_trend(self, project: str) -> SentimentTrend:
        """Get sentiment trend for a project.
        
        Args:
            project: Project name.
            
        Returns:
            SentimentTrend object.
        """
        history = self.feedback_history.get(project, [])
        
        if not history:
            return SentimentTrend(
                project_id=project,
                positive_count=0,
                negative_count=0,
                trend_direction="stable",
                confidence=0.0,
                period_start=time.time(),
                period_end=time.time()
            )
        
        # Count positive/negative
        positive = sum(1 for f, t in history if f == "correct")
        negative = sum(1 for f, t in history if f == "incorrect")
        total = len(history)
        
        # Determine trend direction
        if total >= 10:  # Need minimum data
            # Compare recent vs older feedback
            mid_point = len(history) // 2
            recent = history[mid_point:]
            older = history[:mid_point]
            
            recent_positive_rate = sum(1 for f, t in recent if f == "correct") / len(recent)
            older_positive_rate = sum(1 for f, t in older if f == "correct") / len(older)
            
            if recent_positive_rate > older_positive_rate + 0.1:
                direction = "improving"
            elif recent_positive_rate < older_positive_rate - 0.1:
                direction = "degrading"
            else:
                direction = "stable"
        else:
            direction = "insufficient_data"
        
        # Calculate confidence
        confidence = positive / total if total > 0 else 0.0
        
        # Get time range
        timestamps = [t for f, t in history]
        period_start = min(timestamps) if timestamps else time.time()
        period_end = max(timestamps) if timestamps else time.time()
        
        return SentimentTrend(
            project_id=project,
            positive_count=positive,
            negative_count=negative,
            trend_direction=direction,
            confidence=confidence,
            period_start=period_start,
            period_end=period_end
        )


class AutomatedFeedbackIntegration:
    """
    Main class for automated feedback integration.
    
    Combines all feedback analysis components to provide
    comprehensive feedback-driven routing improvements.
    """
    
    def __init__(
        self,
        enable_route_adjustment: bool = True,
        enable_confidence_calibration: bool = True,
        enable_root_cause_analysis: bool = True,
        enable_sentiment_trends: bool = True
    ):
        """Initialize automated feedback integration.
        
        Args:
            enable_route_adjustment: Enable automatic route adjustments.
            enable_confidence_calibration: Enable confidence calibration.
            enable_root_cause_analysis: Enable root cause analysis.
            enable_sentiment_trends: Enable sentiment trend analysis.
        """
        self.enable_route_adjustment = enable_route_adjustment
        self.enable_confidence_calibration = enable_confidence_calibration
        self.enable_root_cause_analysis = enable_root_cause_analysis
        self.enable_sentiment_trends = enable_sentiment_trends
        
        # Components
        self.route_adjuster = FeedbackBasedRouteAdjuster() if enable_route_adjustment else None
        self.root_cause_analyzer = RootCauseAnalyzer() if enable_root_cause_analysis else None
        self.sentiment_analyzer = SentimentTrendAnalyzer() if enable_sentiment_trends else None
        
        logger.info(
            f"Initialized AutomatedFeedbackIntegration: "
            f"route_adjustment={enable_route_adjustment}, "
            f"calibration={enable_confidence_calibration}, "
            f"root_cause={enable_root_cause_analysis}, "
            f"sentiment={enable_sentiment_trends}"
        )
    
    def process_feedback(
        self,
        query: str,
        selected_project: str,
        confidence: float,
        user_feedback: str,
        correct_project: Optional[str] = None,
        feedback_comment: Optional[str] = None
    ):
        """Process user feedback through all enabled components.
        
        Args:
            query: User query.
            selected_project: Project that was selected.
            confidence: Routing confidence.
            user_feedback: "correct" or "incorrect".
            correct_project: Correct project if feedback was incorrect.
            feedback_comment: Optional user comment.
        """
        # Route adjustment
        if self.route_adjuster:
            self.route_adjuster.record_feedback(
                query,
                selected_project,
                user_feedback,
                correct_project
            )
        
        # Root cause analysis
        if self.root_cause_analyzer and user_feedback == "incorrect" and feedback_comment:
            self.root_cause_analyzer.analyze_feedback(
                query,
                selected_project,
                feedback_comment
            )
        
        # Sentiment trends
        if self.sentiment_analyzer:
            self.sentiment_analyzer.record_feedback(selected_project, user_feedback)
    
    def get_routing_adjustment(
        self,
        query: str,
        base_project: str,
        base_confidence: float
    ) -> Tuple[str, float, str]:
        """Get routing adjustment based on feedback.
        
        Args:
            query: User query.
            base_project: Base routing decision.
            base_confidence: Base confidence.
            
        Returns:
            Tuple of (adjusted_project, adjusted_confidence, reasoning).
        """
        if self.route_adjuster:
            return self.route_adjuster.get_adjustment(
                query,
                base_project,
                base_confidence
            )
        
        return (base_project, base_confidence, "")
    
    def get_insights(self) -> Dict[str, Any]:
        """Get comprehensive feedback insights.
        
        Returns:
            Dictionary with all feedback insights.
        """
        insights = {}
        
        # Top issues
        if self.root_cause_analyzer:
            insights['top_issues'] = self.root_cause_analyzer.get_top_issues()
        
        # Sentiment trends (for all projects with feedback)
        if self.sentiment_analyzer:
            trends = {}
            for project in self.sentiment_analyzer.feedback_history.keys():
                trends[project] = self.sentiment_analyzer.get_trend(project)
            insights['sentiment_trends'] = trends
        
        return insights