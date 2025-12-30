"""
Tests for Routing Feedback System

Tests feedback collection, pattern learning, correction tracking,
and confidence adjustments based on user feedback.
"""

# Import from local module
import sys
import time
from pathlib import Path

import pytest

from osprey.interfaces.pyqt.routing_feedback import (
    FeedbackPattern,
    FeedbackRecord,
    RoutingFeedback,
)

sys.path.insert(0, str(Path(__file__).parent))
from routing_test_base import RoutingTestBase

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_feedback_path(tmp_path):
    """Create temporary feedback file path."""
    return tmp_path / "test_feedback.json"


@pytest.fixture
def feedback_system(temp_feedback_path):
    """Create routing feedback system with temp storage."""
    return RoutingFeedback(
        max_history=100,
        enable_persistence=True,
        persistence_path=temp_feedback_path,
        learning_threshold=2,
    )


@pytest.fixture
def feedback_system_no_persist():
    """Create routing feedback system without persistence."""
    return RoutingFeedback(
        max_history=100,
        enable_persistence=False,
        learning_threshold=2,
    )


# ============================================================================
# FeedbackRecord Tests
# ============================================================================


class TestFeedbackRecord:
    """Test FeedbackRecord dataclass."""

    def test_create_basic_record(self):
        """Test creating basic feedback record."""
        record = FeedbackRecord(
            timestamp=time.time(),
            query="test query",
            selected_project="project1",
            confidence=0.9,
            user_feedback="correct",
        )

        assert record.query == "test query"
        assert record.selected_project == "project1"
        assert record.confidence == 0.9
        assert record.user_feedback == "correct"
        assert record.correct_project is None
        assert record.reasoning == ""
        assert record.session_id is None

    def test_create_correction_record(self):
        """Test creating correction feedback record."""
        record = FeedbackRecord(
            timestamp=time.time(),
            query="test query",
            selected_project="project1",
            confidence=0.9,
            user_feedback="incorrect",
            correct_project="project2",
            reasoning="Wrong project selected",
            session_id="session123",
        )

        assert record.user_feedback == "incorrect"
        assert record.correct_project == "project2"
        assert record.reasoning == "Wrong project selected"
        assert record.session_id == "session123"


# ============================================================================
# FeedbackPattern Tests
# ============================================================================


class TestFeedbackPattern:
    """Test FeedbackPattern dataclass."""

    def test_create_pattern(self):
        """Test creating feedback pattern."""
        pattern = FeedbackPattern(
            query_pattern="weather_query",
            correct_project="weather_project",
            confidence=0.85,
            feedback_count=5,
            last_updated=time.time(),
        )

        assert pattern.query_pattern == "weather_query"
        assert pattern.correct_project == "weather_project"
        assert pattern.confidence == 0.85
        assert pattern.feedback_count == 5
        assert isinstance(pattern.last_updated, float)


# ============================================================================
# RoutingFeedback Initialization Tests
# ============================================================================


class TestRoutingFeedbackInit:
    """Test RoutingFeedback initialization."""

    def test_init_default(self, temp_feedback_path):
        """Test initialization with default parameters."""
        system = RoutingFeedback(persistence_path=temp_feedback_path)

        assert system.max_history == 1000
        assert system.enable_persistence is True
        assert system.learning_threshold == 2
        assert system.persistence_path == temp_feedback_path

    def test_init_custom_params(self, temp_feedback_path):
        """Test initialization with custom parameters."""
        system = RoutingFeedback(
            max_history=500,
            enable_persistence=False,
            persistence_path=temp_feedback_path,
            learning_threshold=3,
        )

        assert system.max_history == 500
        assert system.enable_persistence is False
        assert system.learning_threshold == 3

    def test_init_creates_empty_structures(self, feedback_system):
        """Test initialization creates empty data structures."""
        assert len(feedback_system._feedback_records) == 0
        assert len(feedback_system._learned_patterns) == 0
        assert len(feedback_system._corrections) == 0
        assert len(feedback_system._project_feedback) == 0


# ============================================================================
# Feedback Recording Tests
# ============================================================================


class TestFeedbackRecording:
    """Test feedback recording functionality."""

    def test_record_correct_feedback(self, feedback_system):
        """Test recording correct feedback."""
        feedback_system.record_feedback(
            query="What is the weather?",
            selected_project="weather_project",
            confidence=0.9,
            user_feedback="correct",
        )

        assert len(feedback_system._feedback_records) == 1
        record = feedback_system._feedback_records[0]
        assert record.query == "What is the weather?"
        assert record.selected_project == "weather_project"
        assert record.user_feedback == "correct"

    def test_record_incorrect_feedback(self, feedback_system):
        """Test recording incorrect feedback with correction."""
        feedback_system.record_feedback(
            query="Show me MPS status",
            selected_project="wrong_project",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="mps_project",
            reasoning="MPS query should go to MPS project",
        )

        assert len(feedback_system._feedback_records) == 1
        record = feedback_system._feedback_records[0]
        assert record.user_feedback == "incorrect"
        assert record.correct_project == "mps_project"
        assert record.reasoning == "MPS query should go to MPS project"

    def test_record_with_session_id(self, feedback_system):
        """Test recording feedback with session ID."""
        feedback_system.record_feedback(
            query="test query",
            selected_project="project1",
            confidence=0.8,
            user_feedback="correct",
            session_id="session_abc123",
        )

        record = feedback_system._feedback_records[0]
        assert record.session_id == "session_abc123"

    def test_max_history_enforcement(self, temp_feedback_path):
        """Test max history limit is enforced."""
        system = RoutingFeedback(
            max_history=5,
            enable_persistence=False,
            persistence_path=temp_feedback_path,
        )

        # Use base class helper
        base = RoutingTestBase()
        records = base.verify_max_history_enforcement(
            system=system,
            max_history=5,
            add_item_func=lambda i: system.record_feedback(
                query=f"query {i}",
                selected_project="project1",
                confidence=0.8,
                user_feedback="correct",
            ),
            get_items_func=lambda: system._feedback_records,
            item_count=10,
        )
        # Should keep most recent
        assert records[-1].query == "query 9"

    def test_project_statistics_update(self, feedback_system):
        """Test project statistics are updated."""
        feedback_system.record_feedback(
            query="query1", selected_project="proj1", confidence=0.9, user_feedback="correct"
        )
        feedback_system.record_feedback(
            query="query2", selected_project="proj1", confidence=0.8, user_feedback="correct"
        )
        feedback_system.record_feedback(
            query="query3", selected_project="proj1", confidence=0.7, user_feedback="incorrect"
        )

        stats = feedback_system._project_feedback["proj1"]
        assert stats["correct"] == 2
        assert stats["incorrect"] == 1

    def test_corrections_tracking(self, feedback_system):
        """Test corrections are tracked."""
        query = "Show MPS status"
        feedback_system.record_feedback(
            query=query,
            selected_project="wrong_proj",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="mps_proj",
        )

        assert query in feedback_system._corrections
        assert len(feedback_system._corrections[query]) == 1
        assert feedback_system._corrections[query][0][0] == "mps_proj"


# ============================================================================
# Pattern Learning Tests
# ============================================================================


class TestPatternLearning:
    """Test pattern learning from feedback."""

    def test_pattern_extraction(self, feedback_system):
        """Test pattern extraction for various query types."""
        # Use base class helper
        base = RoutingTestBase()
        base.verify_pattern_extraction(
            feedback_system,
            test_cases=[
                ("What is the weather today?", "weather_query"),
                ("Show MPS status", "mps_query"),
                ("Check system status", "status_query"),
                ("Display the data", "display_query"),
                ("Some random query here", "some_random_query"),
            ],
        )

    def test_learned_pattern_creation(self, feedback_system):
        """Test learned pattern is created from feedback."""
        feedback_system.record_feedback(
            query="What is the weather?",
            selected_project="wrong_proj",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj",
        )

        assert "weather_query" in feedback_system._learned_patterns
        pattern = feedback_system._learned_patterns["weather_query"]
        assert pattern.correct_project == "weather_proj"
        assert pattern.feedback_count == 1

    def test_learned_pattern_reinforcement(self, feedback_system):
        """Test learned pattern is reinforced with consistent feedback."""
        # First correction
        feedback_system.record_feedback(
            query="What is the weather?",
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj",
        )

        # Second correction for same pattern
        feedback_system.record_feedback(
            query="Show me weather data",
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj",
        )

        pattern = feedback_system._learned_patterns["weather_query"]
        assert pattern.feedback_count == 2
        assert pattern.confidence > 0.7  # Should increase

    def test_learned_pattern_conflict(self, feedback_system):
        """Test learned pattern handles conflicting feedback."""
        # First correction
        feedback_system.record_feedback(
            query="What is the weather?",
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj1",
        )

        # Conflicting correction
        feedback_system.record_feedback(
            query="Show weather",
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj2",
        )

        # With low feedback count, should replace
        pattern = feedback_system._learned_patterns["weather_query"]
        assert pattern.correct_project == "weather_proj2"


# ============================================================================
# Routing Adjustment Tests
# ============================================================================


class TestRoutingAdjustment:
    """Test routing adjustment based on feedback."""

    def test_no_adjustment_without_feedback(self, feedback_system):
        """Test no adjustment when no feedback exists."""
        project, confidence, reasoning = feedback_system.get_routing_adjustment(
            query="new query",
            base_project="base_proj",
            base_confidence=0.8,
        )

        assert project == "base_proj"
        assert confidence == 0.8
        assert reasoning == ""

    def test_adjustment_from_exact_match(self, feedback_system):
        """Test adjustment from exact query match."""
        query = "Show MPS status"

        # Record multiple corrections for same query
        for _ in range(3):
            feedback_system.record_feedback(
                query=query,
                selected_project="wrong",
                confidence=0.7,
                user_feedback="incorrect",
                correct_project="mps_proj",
            )

        # Should get adjustment
        project, confidence, reasoning = feedback_system.get_routing_adjustment(
            query=query,
            base_project="wrong",
            base_confidence=0.7,
        )

        assert project == "mps_proj"
        assert confidence == 0.95  # High confidence from user feedback
        assert "correction" in reasoning.lower()

    def test_adjustment_from_pattern(self, feedback_system):
        """Test adjustment from learned pattern."""
        # Build pattern with multiple feedbacks
        feedback_system.record_feedback(
            query="What is the weather?",
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj",
        )
        feedback_system.record_feedback(
            query="Show weather data",
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj",
        )

        # Different weather query should use pattern
        project, confidence, reasoning = feedback_system.get_routing_adjustment(
            query="Display weather information",
            base_project="wrong",
            base_confidence=0.7,
        )

        assert project == "weather_proj"
        assert "pattern" in reasoning.lower()

    def test_adjustment_from_similar_query(self, feedback_system):
        """Test adjustment from similar query."""
        # Record corrections for a query with unique words to avoid pattern matching
        for _ in range(3):
            feedback_system.record_feedback(
                query="show temperature sensor data readings",
                selected_project="wrong",
                confidence=0.7,
                user_feedback="incorrect",
                correct_project="sensor_proj",
            )

        # Similar query should get adjustment
        project, confidence, reasoning = feedback_system.get_routing_adjustment(
            query="display temperature sensor data values",
            base_project="wrong",
            base_confidence=0.7,
        )

        # Should get adjustment (either from pattern or similarity)
        assert project == "sensor_proj"
        # Reasoning should mention either pattern or similarity
        assert "pattern" in reasoning.lower() or "similar" in reasoning.lower()

    def test_learning_threshold_respected(self, temp_feedback_path):
        """Test learning threshold is respected."""
        system = RoutingFeedback(
            max_history=100,
            enable_persistence=False,
            persistence_path=temp_feedback_path,
            learning_threshold=3,
        )

        query = "test query"

        # Record only 2 corrections (below threshold of 3)
        for _ in range(2):
            system.record_feedback(
                query=query,
                selected_project="wrong",
                confidence=0.7,
                user_feedback="incorrect",
                correct_project="correct_proj",
            )

        # Should not get adjustment (below threshold)
        project, confidence, reasoning = system.get_routing_adjustment(
            query=query,
            base_project="wrong",
            base_confidence=0.7,
        )

        assert project == "wrong"  # No adjustment


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Test feedback statistics."""

    def test_project_stats_no_feedback(self, feedback_system):
        """Test project stats with no feedback."""
        stats = feedback_system.get_project_feedback_stats("unknown_proj")

        assert stats["total_feedback"] == 0
        assert stats["correct_count"] == 0
        assert stats["incorrect_count"] == 0
        assert stats["accuracy_rate"] == 0.0

    def test_project_stats_with_feedback(self, feedback_system):
        """Test project stats with feedback."""
        # Record mixed feedback
        for i in range(7):
            feedback_system.record_feedback(
                query=f"query{i}",
                selected_project="proj1",
                confidence=0.8,
                user_feedback="correct" if i < 5 else "incorrect",
            )

        stats = feedback_system.get_project_feedback_stats("proj1")

        assert stats["total_feedback"] == 7
        assert stats["correct_count"] == 5
        assert stats["incorrect_count"] == 2
        assert abs(stats["accuracy_rate"] - 5 / 7) < 0.01

    def test_get_learned_patterns(self, feedback_system):
        """Test getting all learned patterns."""
        # Create some patterns
        feedback_system.record_feedback(
            query="What is the weather?",
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj",
        )
        feedback_system.record_feedback(
            query="Show MPS status",
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="mps_proj",
        )

        patterns = feedback_system.get_learned_patterns()
        assert len(patterns) >= 2
        assert all(isinstance(p, FeedbackPattern) for p in patterns)

    def test_get_correction_suggestions(self, feedback_system):
        """Test getting correction suggestions."""
        query = "test query"

        # Record multiple corrections
        feedback_system.record_feedback(
            query=query,
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="proj1",
        )
        feedback_system.record_feedback(
            query=query,
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="proj1",
        )
        feedback_system.record_feedback(
            query=query,
            selected_project="wrong",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="proj2",
        )

        suggestions = feedback_system.get_correction_suggestions(query)

        assert len(suggestions) == 2
        assert suggestions[0][0] == "proj1"  # Most common
        assert suggestions[0][1] == 2  # Count
        assert suggestions[1][0] == "proj2"
        assert suggestions[1][1] == 1

    def test_get_correction_suggestions_no_data(self, feedback_system):
        """Test correction suggestions with no data."""
        suggestions = feedback_system.get_correction_suggestions("unknown query")
        assert suggestions == []


# ============================================================================
# Persistence Tests
# ============================================================================


class TestPersistence:
    """Test feedback persistence."""

    def test_save_and_load_feedback(self, temp_feedback_path):
        """Test saving and loading feedback."""
        # Create system and add feedback
        system1 = RoutingFeedback(
            max_history=100,
            enable_persistence=True,
            persistence_path=temp_feedback_path,
        )

        system1.record_feedback(
            query="test query",
            selected_project="proj1",
            confidence=0.9,
            user_feedback="correct",
        )

        # Create new system with same path
        system2 = RoutingFeedback(
            max_history=100,
            enable_persistence=True,
            persistence_path=temp_feedback_path,
        )

        # Should load previous feedback
        assert len(system2._feedback_records) == 1
        assert system2._feedback_records[0].query == "test query"

    def test_persistence_disabled(self, temp_feedback_path):
        """Test system works with persistence disabled."""
        system = RoutingFeedback(
            max_history=100,
            enable_persistence=False,
            persistence_path=temp_feedback_path,
        )

        system.record_feedback(
            query="test",
            selected_project="proj1",
            confidence=0.9,
            user_feedback="correct",
        )

        # Use base class helper
        base = RoutingTestBase()
        base.verify_persistence_disabled(system, temp_feedback_path)

    def test_export_feedback(self, feedback_system, tmp_path):
        """Test exporting feedback to file."""
        # Add some feedback
        feedback_system.record_feedback(
            query="test query",
            selected_project="proj1",
            confidence=0.9,
            user_feedback="correct",
        )

        export_path = tmp_path / "export.json"
        feedback_system.export_feedback(export_path)

        # Use base class helper
        base = RoutingTestBase()
        data = base.verify_export_success(
            export_path,
            expected_keys=["feedback_records", "learned_patterns"],
        )
        assert len(data["feedback_records"]) == 1

    def test_export_feedback_creates_directory(self, feedback_system, tmp_path):
        """Test export creates parent directories."""
        # Use base class helper
        base = RoutingTestBase()
        base.verify_export_creates_directory(
            feedback_system,
            tmp_path,
            export_method="export_feedback",
        )


# ============================================================================
# Clear Feedback Tests
# ============================================================================


class TestClearFeedback:
    """Test clearing feedback data."""

    def test_clear_feedback(self, feedback_system):
        """Test clearing all feedback."""
        # Add some data
        feedback_system.record_feedback(
            query="test",
            selected_project="proj1",
            confidence=0.9,
            user_feedback="correct",
        )

        assert len(feedback_system._feedback_records) > 0

        # Use base class helper
        base = RoutingTestBase()
        base.verify_clear_operation(
            feedback_system,
            clear_method="clear_feedback",
            check_funcs=[
                lambda: feedback_system._feedback_records,
                lambda: feedback_system._learned_patterns,
                lambda: feedback_system._corrections,
                lambda: feedback_system._project_feedback,
            ],
        )


# ============================================================================
# Integration Tests
# ============================================================================


class TestRoutingFeedbackIntegration:
    """Integration tests for routing feedback."""

    def test_full_feedback_cycle(self, feedback_system):
        """Test complete feedback cycle."""
        query = "What is the weather?"

        # Initial routing (wrong)
        feedback_system.record_feedback(
            query=query,
            selected_project="wrong_proj",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj",
        )

        # Record more corrections
        feedback_system.record_feedback(
            query=query,
            selected_project="wrong_proj",
            confidence=0.7,
            user_feedback="incorrect",
            correct_project="weather_proj",
        )

        # Now should get adjustment
        project, confidence, reasoning = feedback_system.get_routing_adjustment(
            query=query,
            base_project="wrong_proj",
            base_confidence=0.7,
        )

        assert project == "weather_proj"
        assert confidence > 0.7
        assert reasoning != ""

    def test_multiple_projects_learning(self, feedback_system):
        """Test learning for multiple projects."""
        # Weather project
        for _ in range(3):
            feedback_system.record_feedback(
                query="weather query",
                selected_project="wrong",
                confidence=0.7,
                user_feedback="incorrect",
                correct_project="weather_proj",
            )

        # MPS project
        for _ in range(3):
            feedback_system.record_feedback(
                query="MPS status",
                selected_project="wrong",
                confidence=0.7,
                user_feedback="incorrect",
                correct_project="mps_proj",
            )

        # Check both patterns learned
        patterns = feedback_system.get_learned_patterns()
        pattern_projects = {p.correct_project for p in patterns}

        assert "weather_proj" in pattern_projects
        assert "mps_proj" in pattern_projects

    def test_persistence_across_sessions(self, temp_feedback_path):
        """Test feedback persists across sessions."""
        # Session 1
        system1 = RoutingFeedback(
            max_history=100,
            enable_persistence=True,
            persistence_path=temp_feedback_path,
        )

        for _ in range(3):
            system1.record_feedback(
                query="weather query",
                selected_project="wrong",
                confidence=0.7,
                user_feedback="incorrect",
                correct_project="weather_proj",
            )

        # Session 2
        system2 = RoutingFeedback(
            max_history=100,
            enable_persistence=True,
            persistence_path=temp_feedback_path,
        )

        # Should have learned pattern from session 1
        project, confidence, reasoning = system2.get_routing_adjustment(
            query="weather query",
            base_project="wrong",
            base_confidence=0.7,
        )

        assert project == "weather_proj"
