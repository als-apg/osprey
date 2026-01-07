"""
Routing Feedback System for Multi-Project Router

This module provides user feedback collection and learning mechanisms for routing decisions,
enabling continuous improvement through user corrections and pattern analysis.

Key Features:
- Feedback collection (positive/negative)
- Correction tracking
- Pattern analysis and learning
- Confidence adjustment based on feedback
- Persistent storage
- Query similarity matching
"""

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from osprey.interfaces.pyqt.gui_utils import get_gui_data_dir
from osprey.utils.logger import get_logger

logger = get_logger("routing_feedback")


@dataclass
class FeedbackRecord:
    """Single feedback record."""

    timestamp: float
    query: str
    selected_project: str
    confidence: float
    user_feedback: str  # "correct" or "incorrect"
    correct_project: str | None = None  # If incorrect, what should it be
    reasoning: str = ""
    session_id: str | None = None


@dataclass
class FeedbackPattern:
    """Learned pattern from feedback."""

    query_pattern: str
    correct_project: str
    confidence: float
    feedback_count: int
    last_updated: float


class RoutingFeedback:
    """
    Collects and learns from user feedback on routing decisions.

    Provides:
    - Feedback recording (correct/incorrect)
    - Pattern learning from corrections
    - Confidence adjustments
    - Query similarity matching
    - Persistent storage
    """

    def __init__(
        self,
        max_history: int = 1000,
        enable_persistence: bool = True,
        persistence_path: Path | None = None,
        learning_threshold: int = 2,  # Min feedback count to establish pattern
    ):
        """Initialize routing feedback system.

        Args:
            max_history: Maximum feedback records to keep.
            enable_persistence: Whether to persist feedback to disk.
            persistence_path: Path to save feedback (default: _agent_data/routing_feedback.json).
            learning_threshold: Minimum feedback count to establish a learned pattern.
        """
        self.logger = logger
        self.max_history = max_history
        self.enable_persistence = enable_persistence
        self.learning_threshold = learning_threshold

        # Set persistence path
        if persistence_path:
            self.persistence_path = persistence_path
        else:
            # For GUI context without a specific project config, use framework-level data directory
            # This avoids requiring a config.yml when running from the framework directory
            try:
                from osprey.utils.config import get_agent_dir

                agent_data_dir = Path(get_agent_dir("routing_feedback"))
                self.persistence_path = agent_data_dir.parent / "routing_feedback.json"
            except FileNotFoundError:
                # Fallback: use framework-relative GUI data directory
                # This works regardless of CWD, user, or host
                self.persistence_path = get_gui_data_dir() / "routing_feedback.json"

        # Feedback storage
        self._feedback_records: list[FeedbackRecord] = []

        # Learned patterns from feedback
        self._learned_patterns: dict[str, FeedbackPattern] = {}

        # Query-to-project corrections
        self._corrections: dict[str, list[tuple[str, str]]] = defaultdict(
            list
        )  # query -> [(project, timestamp)]

        # Project statistics
        self._project_feedback: dict[str, dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "incorrect": 0}
        )

        # Load persisted data if available
        if self.enable_persistence:
            self._load_feedback()

        self.logger.info(
            f"Initialized RoutingFeedback "
            f"(max_history={max_history}, learning_threshold={learning_threshold})"
        )

    def record_feedback(
        self,
        query: str,
        selected_project: str,
        confidence: float,
        user_feedback: str,
        correct_project: str | None = None,
        reasoning: str = "",
        session_id: str | None = None,
    ):
        """Record user feedback on a routing decision.

        Args:
            query: User query.
            selected_project: Project that was selected.
            confidence: Routing confidence.
            user_feedback: "correct" or "incorrect".
            correct_project: If incorrect, the correct project.
            reasoning: Routing reasoning.
            session_id: Optional session identifier.
        """
        record = FeedbackRecord(
            timestamp=time.time(),
            query=query,
            selected_project=selected_project,
            confidence=confidence,
            user_feedback=user_feedback,
            correct_project=correct_project,
            reasoning=reasoning,
            session_id=session_id,
        )

        # Add to records
        self._feedback_records.append(record)

        # Maintain max history
        if len(self._feedback_records) > self.max_history:
            self._feedback_records.pop(0)

        # Update project statistics
        if user_feedback == "correct":
            self._project_feedback[selected_project]["correct"] += 1
        else:
            self._project_feedback[selected_project]["incorrect"] += 1

        # Track corrections
        if user_feedback == "incorrect" and correct_project:
            self._corrections[query].append((correct_project, time.time()))

            # Update learned patterns
            self._update_learned_patterns(query, correct_project)

        # Persist if enabled
        if self.enable_persistence:
            self._save_feedback()

        self.logger.info(
            f"Recorded feedback: {query[:50]}... → {selected_project} ({user_feedback})"
        )

    def get_routing_adjustment(
        self, query: str, base_project: str, base_confidence: float
    ) -> tuple[str, float, str]:
        """Get routing adjustment based on learned feedback.

        Args:
            query: User query.
            base_project: Base routing decision project.
            base_confidence: Base routing confidence.

        Returns:
            Tuple of (adjusted_project, adjusted_confidence, reasoning).
        """
        # Check for exact query match in corrections
        if query in self._corrections:
            corrections = self._corrections[query]
            if len(corrections) >= self.learning_threshold:
                # Get most common correction
                projects = [p for p, t in corrections]
                most_common = Counter(projects).most_common(1)[0][0]

                return (
                    most_common,
                    0.95,  # High confidence from user feedback
                    f"Learned from {len(corrections)} user correction(s)",
                )

        # Check for pattern match
        pattern = self._extract_pattern(query)
        if pattern in self._learned_patterns:
            learned = self._learned_patterns[pattern]
            if learned.feedback_count >= self.learning_threshold:
                return (
                    learned.correct_project,
                    learned.confidence,
                    f"Learned pattern from {learned.feedback_count} feedback(s)",
                )

        # Check for similar queries
        similar_adjustment = self._find_similar_query_adjustment(query)
        if similar_adjustment:
            return similar_adjustment

        # No adjustment needed
        return (base_project, base_confidence, "")

    def get_project_feedback_stats(self, project_name: str) -> dict[str, any]:
        """Get feedback statistics for a project.

        Args:
            project_name: Name of project.

        Returns:
            Dictionary with feedback statistics.
        """
        stats = self._project_feedback.get(project_name, {"correct": 0, "incorrect": 0})
        total = stats["correct"] + stats["incorrect"]

        if total == 0:
            return {
                "total_feedback": 0,
                "correct_count": 0,
                "incorrect_count": 0,
                "accuracy_rate": 0.0,
            }

        return {
            "total_feedback": total,
            "correct_count": stats["correct"],
            "incorrect_count": stats["incorrect"],
            "accuracy_rate": stats["correct"] / total,
        }

    def get_learned_patterns(self) -> list[FeedbackPattern]:
        """Get all learned patterns.

        Returns:
            List of FeedbackPattern objects.
        """
        return list(self._learned_patterns.values())

    def get_correction_suggestions(self, query: str) -> list[tuple[str, int]]:
        """Get correction suggestions for a query.

        Args:
            query: User query.

        Returns:
            List of (project, count) tuples sorted by frequency.
        """
        if query not in self._corrections:
            return []

        projects = [p for p, t in self._corrections[query]]
        return Counter(projects).most_common()

    def clear_feedback(self):
        """Clear all feedback data."""
        self._feedback_records.clear()
        self._learned_patterns.clear()
        self._corrections.clear()
        self._project_feedback.clear()

        if self.enable_persistence:
            self._save_feedback()

        self.logger.info("Cleared all feedback data")

    def export_feedback(self, filepath: Path) -> bool:
        """Export feedback to JSON file.

        Args:
            filepath: Path to export file.

        Returns:
            True if successful.
        """
        try:
            data = {
                "feedback_records": [
                    {
                        "timestamp": r.timestamp,
                        "query": r.query,
                        "selected_project": r.selected_project,
                        "confidence": r.confidence,
                        "user_feedback": r.user_feedback,
                        "correct_project": r.correct_project,
                        "reasoning": r.reasoning,
                        "session_id": r.session_id,
                    }
                    for r in self._feedback_records
                ],
                "learned_patterns": {
                    pattern: {
                        "query_pattern": p.query_pattern,
                        "correct_project": p.correct_project,
                        "confidence": p.confidence,
                        "feedback_count": p.feedback_count,
                        "last_updated": p.last_updated,
                    }
                    for pattern, p in self._learned_patterns.items()
                },
                "project_feedback": dict(self._project_feedback),
                "exported_at": datetime.now().isoformat(),
            }

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.info(
                f"Exported {len(self._feedback_records)} feedback records to {filepath}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to export feedback: {e}")
            return False

    # Private methods

    def _extract_pattern(self, query: str) -> str:
        """Extract query pattern for learning.

        Args:
            query: User query.

        Returns:
            Pattern string.
        """
        # Simple pattern extraction - could be enhanced with NLP
        query_lower = query.lower()

        # Common patterns
        if "weather" in query_lower:
            return "weather_query"
        elif "mps" in query_lower or "machine protection" in query_lower:
            return "mps_query"
        elif "status" in query_lower:
            return "status_query"
        elif "show" in query_lower or "display" in query_lower:
            return "display_query"
        elif "data" in query_lower:
            return "data_query"
        else:
            # Use first few words as pattern
            words = query_lower.split()[:3]
            return "_".join(words) if words else "other"

    def _update_learned_patterns(self, query: str, correct_project: str):
        """Update learned patterns from feedback.

        Args:
            query: User query.
            correct_project: Correct project for this query.
        """
        pattern = self._extract_pattern(query)

        if pattern in self._learned_patterns:
            # Update existing pattern
            learned = self._learned_patterns[pattern]

            # If same project, increase confidence
            if learned.correct_project == correct_project:
                learned.feedback_count += 1
                learned.confidence = min(0.99, learned.confidence + 0.05)
                learned.last_updated = time.time()
            else:
                # Different project - need more feedback to change
                if learned.feedback_count <= 2:
                    # Replace with new project
                    learned.correct_project = correct_project
                    learned.feedback_count = 1
                    learned.confidence = 0.7
                    learned.last_updated = time.time()
        else:
            # Create new pattern
            self._learned_patterns[pattern] = FeedbackPattern(
                query_pattern=pattern,
                correct_project=correct_project,
                confidence=0.7,
                feedback_count=1,
                last_updated=time.time(),
            )

        self.logger.debug(f"Updated learned pattern: {pattern} → {correct_project}")

    def _find_similar_query_adjustment(self, query: str) -> tuple[str, float, str] | None:
        """Find adjustment based on similar queries.

        Args:
            query: User query.

        Returns:
            Tuple of (project, confidence, reasoning) or None.
        """
        # Simple similarity: check if query contains words from corrected queries
        query_words = set(query.lower().split())

        best_match = None
        best_similarity = 0.0

        for corrected_query, corrections in self._corrections.items():
            if len(corrections) < self.learning_threshold:
                continue

            corrected_words = set(corrected_query.lower().split())

            # Calculate word overlap
            overlap = query_words & corrected_words
            similarity = len(overlap) / max(len(query_words), len(corrected_words))

            if similarity > best_similarity and similarity > 0.5:
                best_similarity = similarity
                projects = [p for p, t in corrections]
                most_common = Counter(projects).most_common(1)[0][0]
                best_match = (
                    most_common,
                    0.8 * similarity,  # Scale confidence by similarity
                    f"Similar to corrected query (similarity: {similarity:.0%})",
                )

        return best_match

    def _save_feedback(self):
        """Save feedback to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Save only recent feedback to avoid large files
            recent_feedback = self._feedback_records[-self.max_history :]

            data = {
                "feedback_records": [
                    {
                        "timestamp": r.timestamp,
                        "query": r.query,
                        "selected_project": r.selected_project,
                        "confidence": r.confidence,
                        "user_feedback": r.user_feedback,
                        "correct_project": r.correct_project,
                        "reasoning": r.reasoning,
                        "session_id": r.session_id,
                    }
                    for r in recent_feedback
                ],
                "learned_patterns": {
                    pattern: {
                        "query_pattern": p.query_pattern,
                        "correct_project": p.correct_project,
                        "confidence": p.confidence,
                        "feedback_count": p.feedback_count,
                        "last_updated": p.last_updated,
                    }
                    for pattern, p in self._learned_patterns.items()
                },
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.persistence_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to save feedback: {e}")

    def _load_feedback(self):
        """Load feedback from disk."""
        try:
            if not self.persistence_path.exists():
                return

            with open(self.persistence_path) as f:
                data = json.load(f)

            # Load feedback records
            for r_data in data.get("feedback_records", []):
                record = FeedbackRecord(
                    timestamp=r_data["timestamp"],
                    query=r_data["query"],
                    selected_project=r_data["selected_project"],
                    confidence=r_data["confidence"],
                    user_feedback=r_data["user_feedback"],
                    correct_project=r_data.get("correct_project"),
                    reasoning=r_data.get("reasoning", ""),
                    session_id=r_data.get("session_id"),
                )
                self._feedback_records.append(record)

                # Rebuild statistics
                if record.user_feedback == "correct":
                    self._project_feedback[record.selected_project]["correct"] += 1
                else:
                    self._project_feedback[record.selected_project]["incorrect"] += 1

                # Rebuild corrections
                if record.user_feedback == "incorrect" and record.correct_project:
                    self._corrections[record.query].append(
                        (record.correct_project, record.timestamp)
                    )

            # Load learned patterns
            for pattern, p_data in data.get("learned_patterns", {}).items():
                self._learned_patterns[pattern] = FeedbackPattern(
                    query_pattern=p_data["query_pattern"],
                    correct_project=p_data["correct_project"],
                    confidence=p_data["confidence"],
                    feedback_count=p_data["feedback_count"],
                    last_updated=p_data["last_updated"],
                )

            self.logger.info(
                f"Loaded {len(self._feedback_records)} feedback records and "
                f"{len(self._learned_patterns)} learned patterns from {self.persistence_path}"
            )

        except Exception as e:
            self.logger.warning(f"Failed to load feedback: {e}")
