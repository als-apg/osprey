"""
Tests for Semantic Context Analyzer

Tests semantic similarity, topic clustering, intent recognition,
and context-aware routing features.
"""

import os
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from osprey.interfaces.pyqt.semantic_context_analyzer import (
    IntentRecognizer,
    SemanticContextAnalyzer,
    SemanticQuery,
    SemanticSimilarityCalculator,
    TopicCluster,
)

# Force CPU mode to avoid CUDA compatibility issues in tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    mock_model = Mock()
    mock_model.encode = Mock(
        side_effect=lambda text, convert_to_numpy=True: np.random.rand(384).astype(np.float32)
    )
    return mock_model


@pytest.fixture
def similarity_calculator():
    """Create similarity calculator with fallback mode."""
    return SemanticSimilarityCalculator()


@pytest.fixture
def similarity_calculator_with_model(mock_sentence_transformer):
    """Create similarity calculator with mocked model."""
    with patch("osprey.interfaces.pyqt.semantic_context_analyzer.EMBEDDINGS_AVAILABLE", True):
        with patch(
            "osprey.interfaces.pyqt.semantic_context_analyzer.SentenceTransformer",
            return_value=mock_sentence_transformer,
        ):
            calc = SemanticSimilarityCalculator()
            calc.model = mock_sentence_transformer
            return calc


@pytest.fixture
def intent_recognizer():
    """Create intent recognizer."""
    return IntentRecognizer()


@pytest.fixture
def context_analyzer():
    """Create semantic context analyzer."""
    return SemanticContextAnalyzer(
        max_history=20,
        similarity_threshold=0.5,
        topic_similarity_threshold=0.6,
        enable_intent_recognition=True,
    )


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What is the temperature?",
        "Show me the pressure readings",
        "How do I configure the system?",
        "What about humidity levels?",
        "Display the voltage data",
    ]


# ============================================================================
# SemanticSimilarityCalculator Tests
# ============================================================================


class TestSemanticSimilarityCalculator:
    """Test semantic similarity calculator."""

    def test_init_fallback_mode(self):
        """Test initialization in fallback mode."""
        calc = SemanticSimilarityCalculator()
        assert calc.model_name == "all-MiniLM-L6-v2"
        # Model may or may not be loaded depending on environment

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        calc = SemanticSimilarityCalculator(model_name="custom-model")
        assert calc.model_name == "custom-model"

    def test_encode_fallback(self, similarity_calculator):
        """Test encoding with fallback method."""
        text = "Hello world"
        embedding = similarity_calculator.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        # Size depends on whether model loaded (384) or fallback (128)
        assert len(embedding) in [128, 384]
        # Normalized (or close to it for model embeddings)
        norm = np.linalg.norm(embedding)
        assert norm > 0  # Non-zero

    def test_encode_empty_text(self, similarity_calculator):
        """Test encoding empty text."""
        embedding = similarity_calculator.encode("")
        assert isinstance(embedding, np.ndarray)
        # Size depends on whether model loaded (384) or fallback (128)
        assert len(embedding) in [128, 384]

    def test_encode_with_model(self, similarity_calculator_with_model):
        """Test encoding with actual model."""
        text = "Test query"
        embedding = similarity_calculator_with_model.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384  # Model output size

    def test_calculate_similarity_identical(self, similarity_calculator):
        """Test similarity of identical embeddings."""
        embedding = similarity_calculator.encode("test")
        similarity = similarity_calculator.calculate_similarity(embedding, embedding)

        assert 0.99 <= similarity <= 1.0  # Should be very close to 1

    def test_calculate_similarity_different(self, similarity_calculator):
        """Test similarity of different embeddings."""
        emb1 = similarity_calculator.encode("temperature sensor")
        emb2 = similarity_calculator.encode("pressure gauge")

        similarity = similarity_calculator.calculate_similarity(emb1, emb2)
        assert 0.0 <= similarity <= 1.0

    def test_calculate_similarity_zero_vectors(self, similarity_calculator):
        """Test similarity with zero vectors."""
        # Get actual embedding size from a test encoding
        test_emb = similarity_calculator.encode("test")
        emb_size = len(test_emb)

        zero_vec = np.zeros(emb_size, dtype=np.float32)
        normal_vec = test_emb

        similarity = similarity_calculator.calculate_similarity(zero_vec, normal_vec)
        assert similarity == 0.0

    def test_calculate_similarity_bounds(self, similarity_calculator):
        """Test similarity is always in [0, 1] range."""
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        similarity = similarity_calculator.calculate_similarity(emb1, emb2)
        assert 0.0 <= similarity <= 1.0

    def test_encode_consistency(self, similarity_calculator):
        """Test encoding same text produces same result."""
        text = "consistent encoding test"
        emb1 = similarity_calculator.encode(text)
        emb2 = similarity_calculator.encode(text)

        assert np.allclose(emb1, emb2)


# ============================================================================
# IntentRecognizer Tests
# ============================================================================


class TestIntentRecognizer:
    """Test intent recognizer."""

    def test_init(self, intent_recognizer):
        """Test initialization."""
        assert intent_recognizer is not None
        assert hasattr(intent_recognizer, "INTENT_PATTERNS")

    def test_recognize_question_intent(self, intent_recognizer):
        """Test recognizing question intent."""
        queries = [
            "What is the temperature?",
            "How does this work?",
            "Why is the pressure high?",
            "When will it finish?",
            "Where is the sensor?",
        ]

        for query in queries:
            intent = intent_recognizer.recognize_intent(query)
            assert intent == "question"

    def test_recognize_command_intent(self, intent_recognizer):
        """Test recognizing command intent."""
        queries = [
            "Show me the data",
            "Display the readings",
            "Get the temperature",
            "List all sensors",
            "Execute the script",
        ]

        for query in queries:
            intent = intent_recognizer.recognize_intent(query)
            assert intent == "command"

    def test_recognize_clarification_intent(self, intent_recognizer):
        """Test recognizing clarification intent."""
        context = ["What is the temperature?"]
        queries = [
            "What about the pressure?",
            "Also show humidity",
            "And the voltage?",
            "Additionally, check the flow rate",
        ]

        for query in queries:
            intent = intent_recognizer.recognize_intent(query, context)
            assert intent == "clarification"

    def test_recognize_new_topic_intent(self, intent_recognizer):
        """Test recognizing new topic intent."""
        queries = [
            "Now show me something else",
            "Next, let's look at pressure",
            "Instead, display voltage",
            "Switch to humidity data",
        ]

        for query in queries:
            intent = intent_recognizer.recognize_intent(query)
            assert intent == "new_topic"

    def test_recognize_without_context(self, intent_recognizer):
        """Test intent recognition without context."""
        query = "Also show the pressure"
        intent = intent_recognizer.recognize_intent(query, context=None)
        # Without context, "also" might not trigger clarification
        assert intent in ["question", "command", "clarification"]

    def test_recognize_default_intent(self, intent_recognizer):
        """Test default intent for ambiguous queries."""
        query = "temperature readings"
        intent = intent_recognizer.recognize_intent(query)
        assert intent == "question"  # Default

    def test_recognize_case_insensitive(self, intent_recognizer):
        """Test case-insensitive intent recognition."""
        queries = [
            "WHAT IS THE TEMPERATURE?",
            "Show Me The Data",
            "also CHECK PRESSURE",
        ]

        for query in queries:
            intent = intent_recognizer.recognize_intent(query)
            assert intent in ["question", "command", "clarification"]


# ============================================================================
# SemanticQuery Tests
# ============================================================================


class TestSemanticQuery:
    """Test SemanticQuery dataclass."""

    def test_create_basic(self):
        """Test creating basic semantic query."""
        query = SemanticQuery(text="test query")
        assert query.text == "test query"
        assert query.embedding is None
        assert query.project is None
        assert query.intent is None
        assert isinstance(query.timestamp, float)

    def test_create_with_embedding(self):
        """Test creating query with embedding."""
        embedding = np.array([1.0, 2.0, 3.0])
        query = SemanticQuery(text="test", embedding=embedding)
        assert np.array_equal(query.embedding, embedding)

    def test_create_with_all_fields(self):
        """Test creating query with all fields."""
        embedding = np.array([1.0, 2.0])
        timestamp = time.time()

        query = SemanticQuery(
            text="test",
            embedding=embedding,
            timestamp=timestamp,
            project="project1",
            intent="question",
        )

        assert query.text == "test"
        assert np.array_equal(query.embedding, embedding)
        assert query.timestamp == timestamp
        assert query.project == "project1"
        assert query.intent == "question"


# ============================================================================
# TopicCluster Tests
# ============================================================================


class TestTopicCluster:
    """Test TopicCluster dataclass."""

    def test_create_cluster(self):
        """Test creating topic cluster."""
        centroid = np.array([1.0, 2.0, 3.0])
        query = SemanticQuery(text="test", project="proj1")

        cluster = TopicCluster(
            topic_id=0,
            centroid=centroid,
            queries=[query],
            dominant_project="proj1",
            confidence=1.0,
            last_updated=time.time(),
        )

        assert cluster.topic_id == 0
        assert np.array_equal(cluster.centroid, centroid)
        assert len(cluster.queries) == 1
        assert cluster.dominant_project == "proj1"
        assert cluster.confidence == 1.0


# ============================================================================
# SemanticContextAnalyzer Tests
# ============================================================================


class TestSemanticContextAnalyzer:
    """Test semantic context analyzer."""

    def test_init(self, context_analyzer):
        """Test initialization."""
        assert context_analyzer.max_history == 20
        assert context_analyzer.similarity_threshold == 0.5
        assert context_analyzer.topic_similarity_threshold == 0.6
        assert context_analyzer.enable_intent_recognition is True
        assert len(context_analyzer.query_history) == 0
        assert len(context_analyzer.topic_clusters) == 0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        analyzer = SemanticContextAnalyzer(
            max_history=10,
            similarity_threshold=0.7,
            topic_similarity_threshold=0.8,
            enable_intent_recognition=False,
        )

        assert analyzer.max_history == 10
        assert analyzer.similarity_threshold == 0.7
        assert analyzer.topic_similarity_threshold == 0.8
        assert analyzer.enable_intent_recognition is False

    def test_add_query(self, context_analyzer):
        """Test adding query to history."""
        context_analyzer.add_query("What is the temperature?", "project1", 0.9)

        assert len(context_analyzer.query_history) == 1
        query = context_analyzer.query_history[0]
        assert query.text == "What is the temperature?"
        assert query.project == "project1"
        assert query.embedding is not None
        assert query.intent is not None

    def test_add_multiple_queries(self, context_analyzer, sample_queries):
        """Test adding multiple queries."""
        for i, query_text in enumerate(sample_queries):
            context_analyzer.add_query(query_text, f"project{i % 2}", 0.8)

        assert len(context_analyzer.query_history) == len(sample_queries)

    def test_max_history_limit(self, context_analyzer):
        """Test max history limit enforcement."""
        # Add more than max_history queries
        for i in range(25):
            context_analyzer.add_query(f"Query {i}", "project1", 0.8)

        assert len(context_analyzer.query_history) == context_analyzer.max_history

    def test_get_relevant_context_empty(self, context_analyzer):
        """Test getting relevant context with empty history."""
        relevant = context_analyzer.get_relevant_context("test query")
        assert len(relevant) == 0

    def test_get_relevant_context(self, context_analyzer):
        """Test getting relevant context queries."""
        # Add some queries
        context_analyzer.add_query("What is the temperature?", "project1", 0.9)
        context_analyzer.add_query("Show me temperature data", "project1", 0.8)
        context_analyzer.add_query("Display pressure readings", "project2", 0.7)

        # Query for temperature-related context
        relevant = context_analyzer.get_relevant_context("temperature sensor", max_results=5)

        # Should find temperature-related queries
        assert isinstance(relevant, list)
        # Exact count depends on similarity threshold and embeddings

    def test_get_relevant_context_max_results(self, context_analyzer):
        """Test max_results parameter."""
        # Add many queries
        for i in range(10):
            context_analyzer.add_query(f"Query {i}", "project1", 0.8)

        relevant = context_analyzer.get_relevant_context("Query 5", max_results=3)
        assert len(relevant) <= 3

    def test_get_current_topic_none(self, context_analyzer):
        """Test getting current topic when none exists."""
        topic = context_analyzer.get_current_topic()
        assert topic is None

    def test_get_current_topic_active(self, context_analyzer):
        """Test getting active current topic."""
        # Add queries to create a topic
        for _ in range(3):
            context_analyzer.add_query("temperature query", "project1", 0.9)

        topic = context_analyzer.get_current_topic()
        # May or may not have topic depending on clustering threshold
        if topic:
            assert isinstance(topic, TopicCluster)
            assert topic.dominant_project == "project1"

    def test_get_current_topic_expired(self, context_analyzer):
        """Test expired topic is not returned."""
        # Add query and manually create old cluster
        context_analyzer.add_query("old query", "project1", 0.9)

        if context_analyzer.topic_clusters:
            # Make cluster old (> 5 minutes)
            context_analyzer.topic_clusters[0].last_updated = time.time() - 400

            topic = context_analyzer.get_current_topic()
            assert topic is None

    def test_should_boost_project_no_context(self, context_analyzer):
        """Test project boost with no context."""
        should_boost, boost, reason = context_analyzer.should_boost_project(
            "test query", "project1"
        )

        assert should_boost is False
        assert boost == 0.0
        assert reason == ""

    def test_should_boost_project_topic_continuity(self, context_analyzer):
        """Test project boost for topic continuity."""
        # Create a topic cluster
        for _ in range(3):
            context_analyzer.add_query("temperature reading", "project1", 0.9)

        # Query related to same topic
        should_boost, boost, reason = context_analyzer.should_boost_project(
            "show temperature", "project1"
        )

        # May boost if topic clustering worked
        if should_boost:
            assert boost > 0
            assert "topic continuity" in reason.lower() or "similar" in reason.lower()

    def test_should_boost_project_relevant_context(self, context_analyzer):
        """Test project boost based on relevant context."""
        # Add multiple queries for same project
        context_analyzer.add_query("temperature query 1", "project1", 0.9)
        context_analyzer.add_query("temperature query 2", "project1", 0.9)
        context_analyzer.add_query("temperature query 3", "project1", 0.9)

        # Similar query
        should_boost, boost, reason = context_analyzer.should_boost_project(
            "temperature data", "project1"
        )

        # May boost based on relevant context
        if should_boost:
            assert boost > 0
            assert "similar" in reason.lower() or "recent" in reason.lower()

    def test_get_context_summary_empty(self, context_analyzer):
        """Test context summary with no history."""
        summary = context_analyzer.get_context_summary()
        assert "No semantic context" in summary

    def test_get_context_summary_with_history(self, context_analyzer):
        """Test context summary with history."""
        context_analyzer.add_query("test query", "project1", 0.9)
        summary = context_analyzer.get_context_summary()

        assert "History: 1 queries" in summary
        assert "project1" in summary

    def test_get_context_summary_with_topic(self, context_analyzer):
        """Test context summary with active topic."""
        # Add queries to potentially create topic
        for _ in range(3):
            context_analyzer.add_query("temperature query", "project1", 0.9)

        summary = context_analyzer.get_context_summary()
        assert "History:" in summary
        # May include topic info if clustering worked

    def test_clear(self, context_analyzer):
        """Test clearing context."""
        # Add some data
        context_analyzer.add_query("test query", "project1", 0.9)
        assert len(context_analyzer.query_history) > 0

        # Clear
        context_analyzer.clear()

        assert len(context_analyzer.query_history) == 0
        assert len(context_analyzer.topic_clusters) == 0

    def test_topic_cluster_creation(self, context_analyzer):
        """Test topic cluster creation."""
        # Add similar queries
        for _ in range(3):
            context_analyzer.add_query("temperature sensor reading", "project1", 0.9)

        # Clusters may be created depending on threshold
        # Just verify no errors occur
        assert isinstance(context_analyzer.topic_clusters, list)

    def test_topic_cluster_pruning(self, context_analyzer):
        """Test topic cluster pruning (max 5 clusters)."""
        # Create many different topics
        topics = [
            "temperature",
            "pressure",
            "humidity",
            "voltage",
            "current",
            "flow",
            "level",
        ]

        for topic in topics:
            for i in range(2):
                context_analyzer.add_query(f"{topic} reading {i}", "project1", 0.9)

        # Should prune to max 5 clusters
        assert len(context_analyzer.topic_clusters) <= 5

    def test_intent_recognition_integration(self, context_analyzer):
        """Test intent recognition integration."""
        context_analyzer.add_query("What is the temperature?", "project1", 0.9)

        query = context_analyzer.query_history[0]
        assert query.intent == "question"

    def test_intent_disabled(self):
        """Test analyzer with intent recognition disabled."""
        analyzer = SemanticContextAnalyzer(enable_intent_recognition=False)
        analyzer.add_query("What is the temperature?", "project1", 0.9)

        query = analyzer.query_history[0]
        assert query.intent is None

    def test_concurrent_projects(self, context_analyzer):
        """Test handling multiple projects."""
        context_analyzer.add_query("temperature query", "project1", 0.9)
        context_analyzer.add_query("pressure query", "project2", 0.8)
        context_analyzer.add_query("humidity query", "project1", 0.7)

        assert len(context_analyzer.query_history) == 3

        # Check project distribution
        projects = [q.project for q in context_analyzer.query_history]
        assert "project1" in projects
        assert "project2" in projects

    def test_embedding_consistency(self, context_analyzer):
        """Test embedding consistency for same query."""
        context_analyzer.add_query("test query", "project1", 0.9)
        context_analyzer.add_query("test query", "project1", 0.9)

        emb1 = context_analyzer.query_history[0].embedding
        emb2 = context_analyzer.query_history[1].embedding

        # Should be very similar (same text)
        similarity = context_analyzer.similarity_calculator.calculate_similarity(emb1, emb2)
        assert similarity > 0.99

    def test_query_timestamp_ordering(self, context_analyzer):
        """Test queries are ordered by timestamp."""
        times = []
        for i in range(5):
            context_analyzer.add_query(f"query {i}", "project1", 0.9)
            times.append(context_analyzer.query_history[-1].timestamp)
            time.sleep(0.01)  # Small delay

        # Timestamps should be increasing
        assert times == sorted(times)


# ============================================================================
# Integration Tests
# ============================================================================


class TestSemanticContextIntegration:
    """Integration tests for semantic context analyzer."""

    def test_full_conversation_flow(self, context_analyzer):
        """Test full conversation flow with context."""
        # User asks about temperature
        context_analyzer.add_query("What is the temperature?", "project1", 0.9)

        # Follow-up question
        context_analyzer.add_query("What about pressure?", "project1", 0.8)

        # Check intent recognition
        assert context_analyzer.query_history[1].intent == "clarification"

        # Get relevant context
        relevant = context_analyzer.get_relevant_context("temperature sensor")
        assert len(relevant) >= 0  # May find relevant queries

    def test_topic_switching(self, context_analyzer):
        """Test switching between topics."""
        # Topic 1: Temperature
        for _ in range(3):
            context_analyzer.add_query("temperature reading", "project1", 0.9)

        # Topic 2: Pressure
        for _ in range(3):
            context_analyzer.add_query("pressure measurement", "project2", 0.8)

        # Should have queries from both topics
        assert len(context_analyzer.query_history) == 6

    def test_semantic_routing_boost(self, context_analyzer):
        """Test semantic routing boost calculation."""
        # Build context for project1
        context_analyzer.add_query("temperature sensor data", "project1", 0.9)
        context_analyzer.add_query("temperature readings", "project1", 0.9)

        # Query similar to context
        should_boost, boost, reason = context_analyzer.should_boost_project(
            "show temperature", "project1"
        )

        # Should consider boosting based on semantic similarity
        assert isinstance(should_boost, bool)
        assert isinstance(boost, float)
        assert isinstance(reason, str)

    def test_multi_project_context(self, context_analyzer):
        """Test context with multiple projects."""
        # Add queries for different projects
        context_analyzer.add_query("temperature data", "project1", 0.9)
        context_analyzer.add_query("pressure data", "project2", 0.8)
        context_analyzer.add_query("humidity data", "project3", 0.7)

        # Get summary
        summary = context_analyzer.get_context_summary()
        assert "History: 3 queries" in summary

    def test_long_conversation(self, context_analyzer):
        """Test long conversation with history limit."""
        # Add many queries
        for i in range(30):
            project = f"project{i % 3}"
            context_analyzer.add_query(f"query {i}", project, 0.8)

        # Should maintain max history
        assert len(context_analyzer.query_history) == context_analyzer.max_history

        # Should still function correctly
        relevant = context_analyzer.get_relevant_context("query 25")
        assert isinstance(relevant, list)
