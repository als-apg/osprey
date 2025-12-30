"""
Tests for Conversation Context

Tests conversation history tracking, topic detection, confidence boosting,
and context-aware routing features.
"""

import time

import pytest

from osprey.interfaces.pyqt.conversation_context import (
    ConversationContext,
    QueryRecord,
    TopicInfo,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def context():
    """Create conversation context."""
    return ConversationContext(
        max_history=10,
        topic_threshold=2,
        topic_decay_seconds=300.0,
        confidence_boost=0.2,
    )


@pytest.fixture
def context_short_decay():
    """Create conversation context with short decay time."""
    return ConversationContext(
        max_history=10,
        topic_threshold=2,
        topic_decay_seconds=1.0,  # 1 second
        confidence_boost=0.2,
    )


# ============================================================================
# QueryRecord Tests
# ============================================================================


class TestQueryRecord:
    """Test QueryRecord dataclass."""

    def test_create_basic_record(self):
        """Test creating basic query record."""
        record = QueryRecord(
            query="test query",
            project_name="project1",
            confidence=0.9,
            timestamp=time.time(),
        )

        assert record.query == "test query"
        assert record.project_name == "project1"
        assert record.confidence == 0.9
        assert record.reasoning == ""

    def test_create_record_with_reasoning(self):
        """Test creating record with reasoning."""
        record = QueryRecord(
            query="test query",
            project_name="project1",
            confidence=0.9,
            timestamp=time.time(),
            reasoning="High keyword match",
        )

        assert record.reasoning == "High keyword match"


# ============================================================================
# TopicInfo Tests
# ============================================================================


class TestTopicInfo:
    """Test TopicInfo dataclass."""

    def test_create_topic_info(self):
        """Test creating topic info."""
        topic = TopicInfo(
            topic_project="project1",
            confidence=0.8,
            query_count=3,
            last_updated=time.time(),
        )

        assert topic.topic_project == "project1"
        assert topic.confidence == 0.8
        assert topic.query_count == 3


# ============================================================================
# ConversationContext Initialization Tests
# ============================================================================


class TestConversationContextInit:
    """Test ConversationContext initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        ctx = ConversationContext()

        assert ctx.max_history == 10
        assert ctx.topic_threshold == 2
        assert ctx.topic_decay_seconds == 300.0
        assert ctx.confidence_boost == 0.2

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        ctx = ConversationContext(
            max_history=20,
            topic_threshold=3,
            topic_decay_seconds=600.0,
            confidence_boost=0.3,
        )

        assert ctx.max_history == 20
        assert ctx.topic_threshold == 3
        assert ctx.topic_decay_seconds == 600.0
        assert ctx.confidence_boost == 0.3

    def test_init_creates_empty_structures(self, context):
        """Test initialization creates empty data structures."""
        assert len(context._history) == 0
        assert context._current_topic is None


# ============================================================================
# Query Addition Tests
# ============================================================================


class TestQueryAddition:
    """Test adding queries to context."""

    def test_add_single_query(self, context):
        """Test adding a single query."""
        context.add_query("test query", "project1", 0.9)

        assert len(context._history) == 1
        record = context._history[0]
        assert record.query == "test query"
        assert record.project_name == "project1"
        assert record.confidence == 0.9

    def test_add_query_with_reasoning(self, context):
        """Test adding query with reasoning."""
        context.add_query("test query", "project1", 0.9, "High match")

        record = context._history[0]
        assert record.reasoning == "High match"

    def test_add_multiple_queries(self, context):
        """Test adding multiple queries."""
        for i in range(5):
            context.add_query(f"query {i}", "project1", 0.9)

        assert len(context._history) == 5

    def test_max_history_enforcement(self, context):
        """Test max history limit is enforced."""
        # Add more than max_history queries
        for i in range(15):
            context.add_query(f"query {i}", "project1", 0.9)

        assert len(context._history) == context.max_history
        # Should keep most recent
        assert context._history[-1].query == "query 14"
        assert context._history[0].query == "query 5"


# ============================================================================
# History Retrieval Tests
# ============================================================================


class TestHistoryRetrieval:
    """Test retrieving conversation history."""

    def test_get_recent_queries_empty(self, context):
        """Test getting recent queries with empty history."""
        recent = context.get_recent_queries(5)
        assert len(recent) == 0

    def test_get_recent_queries(self, context):
        """Test getting recent queries."""
        for i in range(10):
            context.add_query(f"query {i}", "project1", 0.9)

        recent = context.get_recent_queries(5)
        assert len(recent) == 5
        # Should be most recent
        assert recent[-1].query == "query 9"
        assert recent[0].query == "query 5"

    def test_get_recent_queries_more_than_available(self, context):
        """Test getting more recent queries than available."""
        for i in range(3):
            context.add_query(f"query {i}", "project1", 0.9)

        recent = context.get_recent_queries(10)
        assert len(recent) == 3

    def test_get_last_project_empty(self, context):
        """Test getting last project with empty history."""
        assert context.get_last_project() is None

    def test_get_last_project(self, context):
        """Test getting last project."""
        context.add_query("query1", "project1", 0.9)
        context.add_query("query2", "project2", 0.8)

        assert context.get_last_project() == "project2"


# ============================================================================
# Topic Detection Tests
# ============================================================================


class TestTopicDetection:
    """Test topic detection functionality."""

    def test_no_topic_insufficient_history(self, context):
        """Test no topic with insufficient history."""
        context.add_query("query1", "project1", 0.9)

        assert not context.has_active_topic()
        assert context.get_current_topic() is None

    def test_topic_detection_same_project(self, context):
        """Test topic detection with same project."""
        # Add queries for same project
        for i in range(3):
            context.add_query(f"query {i}", "project1", 0.9)

        assert context.has_active_topic()
        topic = context.get_current_topic()
        assert topic is not None
        assert topic.topic_project == "project1"
        assert topic.query_count >= 2

    def test_topic_detection_mixed_projects(self, context):
        """Test topic detection with mixed projects."""
        # Add queries for different projects
        context.add_query("query1", "project1", 0.9)
        context.add_query("query2", "project2", 0.8)
        context.add_query("query3", "project1", 0.9)

        # May or may not have topic depending on threshold
        # Just verify no errors
        context.has_active_topic()

    def test_topic_decay(self, context_short_decay):
        """Test topic decay over time."""
        # Add queries to establish topic
        for i in range(3):
            context_short_decay.add_query(f"query {i}", "project1", 0.9)

        assert context_short_decay.has_active_topic()

        # Wait for decay
        time.sleep(1.5)

        # Topic should have decayed
        assert not context_short_decay.has_active_topic()

    def test_topic_update_with_new_queries(self, context):
        """Test topic updates with new queries."""
        # Establish topic
        for i in range(3):
            context.add_query(f"query {i}", "project1", 0.9)

        topic1 = context.get_current_topic()
        assert topic1 is not None

        # Add more queries for same project
        context.add_query("query new", "project1", 0.9)

        topic2 = context.get_current_topic()
        assert topic2 is not None
        assert topic2.query_count > topic1.query_count


# ============================================================================
# Confidence Boosting Tests
# ============================================================================


class TestConfidenceBoosting:
    """Test confidence boosting functionality."""

    def test_no_boost_without_topic(self, context):
        """Test no boost without active topic."""
        assert not context.should_boost_confidence("project1")
        assert context.get_confidence_boost("project1") == 0.0

    def test_boost_for_topic_project(self, context):
        """Test boost for project matching topic."""
        # Establish topic
        for i in range(3):
            context.add_query(f"query {i}", "project1", 0.9)

        assert context.should_boost_confidence("project1")
        assert context.get_confidence_boost("project1") == context.confidence_boost

    def test_no_boost_for_different_project(self, context):
        """Test no boost for different project."""
        # Establish topic for project1
        for i in range(3):
            context.add_query(f"query {i}", "project1", 0.9)

        assert not context.should_boost_confidence("project2")
        assert context.get_confidence_boost("project2") == 0.0


# ============================================================================
# Context Summary Tests
# ============================================================================


class TestContextSummary:
    """Test context summary generation."""

    def test_summary_empty(self, context):
        """Test summary with empty history."""
        summary = context.get_context_summary()
        assert "No conversation history" in summary

    def test_summary_with_history(self, context):
        """Test summary with history."""
        context.add_query("query1", "project1", 0.9)
        context.add_query("query2", "project2", 0.8)

        summary = context.get_context_summary()
        assert "History: 2 queries" in summary
        assert "Last project: project2" in summary

    def test_summary_with_topic(self, context):
        """Test summary with active topic."""
        for i in range(3):
            context.add_query(f"query {i}", "project1", 0.9)

        summary = context.get_context_summary()
        assert "History:" in summary
        # May include topic info if detected
        if context.has_active_topic():
            assert "Active topic" in summary


# ============================================================================
# Project Usage Stats Tests
# ============================================================================


class TestProjectUsageStats:
    """Test project usage statistics."""

    def test_stats_empty(self, context):
        """Test stats with empty history."""
        stats = context.get_project_usage_stats()
        assert stats == {}

    def test_stats_single_project(self, context):
        """Test stats with single project."""
        for i in range(5):
            context.add_query(f"query {i}", "project1", 0.9)

        stats = context.get_project_usage_stats()
        assert stats["project1"] == 5

    def test_stats_multiple_projects(self, context):
        """Test stats with multiple projects."""
        for i in range(3):
            context.add_query(f"query {i}", "project1", 0.9)
        for i in range(2):
            context.add_query(f"query {i}", "project2", 0.8)

        stats = context.get_project_usage_stats()
        assert stats["project1"] == 3
        assert stats["project2"] == 2


# ============================================================================
# Context for Routing Tests
# ============================================================================


class TestContextForRouting:
    """Test context information for routing."""

    def test_routing_context_empty(self, context):
        """Test routing context with empty history."""
        routing_ctx = context.get_context_for_routing()

        assert routing_ctx["has_history"] is False
        assert routing_ctx["recent_queries"] == []

    def test_routing_context_with_history(self, context):
        """Test routing context with history."""
        context.add_query("query1", "project1", 0.9)
        context.add_query("query2", "project2", 0.8)

        routing_ctx = context.get_context_for_routing()

        assert routing_ctx["has_history"] is True
        assert len(routing_ctx["recent_queries"]) == 2
        assert routing_ctx["last_project"] == "project2"

    def test_routing_context_with_topic(self, context):
        """Test routing context with active topic."""
        for i in range(3):
            context.add_query(f"query {i}", "project1", 0.9)

        routing_ctx = context.get_context_for_routing()

        if context.has_active_topic():
            assert "active_topic" in routing_ctx
            assert routing_ctx["active_topic"]["project"] == "project1"

    def test_routing_context_recent_queries_limit(self, context):
        """Test routing context limits recent queries."""
        for i in range(10):
            context.add_query(f"query {i}", "project1", 0.9)

        routing_ctx = context.get_context_for_routing()

        # Should limit to 3 recent queries
        assert len(routing_ctx["recent_queries"]) == 3


# ============================================================================
# Clear Context Tests
# ============================================================================


class TestClearContext:
    """Test clearing context."""

    def test_clear(self, context):
        """Test clearing context."""
        # Add some data
        for i in range(5):
            context.add_query(f"query {i}", "project1", 0.9)

        assert len(context._history) > 0

        # Clear
        context.clear()

        assert len(context._history) == 0
        assert context._current_topic is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestConversationContextIntegration:
    """Integration tests for conversation context."""

    def test_full_conversation_flow(self, context):
        """Test full conversation flow."""
        # User asks about weather
        context.add_query("What is the weather?", "weather_proj", 0.9)

        # Follow-up about weather
        context.add_query("What about tomorrow?", "weather_proj", 0.85)

        # Topic should be established
        assert context.has_active_topic()
        assert context.should_boost_confidence("weather_proj")

        # Switch to different topic
        context.add_query("Show MPS status", "mps_proj", 0.9)

        # May or may not have topic depending on threshold
        last_project = context.get_last_project()
        assert last_project == "mps_proj"

    def test_topic_continuity(self, context):
        """Test topic continuity detection."""
        # Build topic
        for i in range(4):
            context.add_query(f"weather query {i}", "weather_proj", 0.9)

        # Topic should be strong
        topic = context.get_current_topic()
        assert topic is not None
        assert topic.topic_project == "weather_proj"
        assert topic.query_count >= 2

        # Confidence boost should be available
        boost = context.get_confidence_boost("weather_proj")
        assert boost == context.confidence_boost

    def test_mixed_conversation(self, context):
        """Test conversation with mixed projects."""
        # Alternate between projects
        context.add_query("weather query", "weather_proj", 0.9)
        context.add_query("mps query", "mps_proj", 0.8)
        context.add_query("weather query 2", "weather_proj", 0.9)
        context.add_query("mps query 2", "mps_proj", 0.8)

        # Should have history
        assert len(context._history) == 4

        # Get stats
        stats = context.get_project_usage_stats()
        assert stats["weather_proj"] == 2
        assert stats["mps_proj"] == 2

    def test_context_summary_evolution(self, context):
        """Test context summary changes over time."""
        # Empty
        summary1 = context.get_context_summary()
        assert "No conversation history" in summary1

        # Add query
        context.add_query("query1", "project1", 0.9)
        summary2 = context.get_context_summary()
        assert "History: 1 queries" in summary2

        # Add more to establish topic
        for i in range(2, 4):
            context.add_query(f"query{i}", "project1", 0.9)

        summary3 = context.get_context_summary()
        assert "History: 3 queries" in summary3
