"""
Semantic Context Analyzer for Multi-Project Router

This module provides semantic similarity-based context analysis using embeddings
to improve conversation-aware routing beyond simple keyword matching.

Key Features:
- Sentence embeddings for semantic similarity
- Topic modeling with clustering
- Intent recognition
- Multi-lingual support
- Context relevance scoring
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from osprey.utils.logger import get_logger

logger = get_logger("semantic_context_analyzer")

# Try to import sentence-transformers, fall back to simple similarity if not available
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. Install with: pip install sentence-transformers"
    )


@dataclass
class SemanticQuery:
    """Query with semantic embedding."""

    text: str
    embedding: np.ndarray | None = None
    timestamp: float = field(default_factory=time.time)
    project: str | None = None
    intent: str | None = None


@dataclass
class TopicCluster:
    """Detected topic cluster."""

    topic_id: int
    centroid: np.ndarray
    queries: list[SemanticQuery]
    dominant_project: str
    confidence: float
    last_updated: float


class SemanticSimilarityCalculator:
    """
    Calculates semantic similarity using sentence embeddings.

    Uses sentence-transformers for high-quality embeddings.
    Falls back to simple word overlap if not available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic similarity calculator.

        Args:
            model_name: Name of sentence-transformers model to use.
        """
        self.model_name = model_name
        self.model = None

        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence-transformers model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
                self.model = None

        if not self.model:
            logger.info("Using fallback word-overlap similarity")

    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector.

        Args:
            text: Text to encode.

        Returns:
            Embedding vector as numpy array.
        """
        if self.model:
            return self.model.encode(text, convert_to_numpy=True)
        else:
            # Fallback: simple hash-based fixed-size representation
            embedding_size = 128
            embedding = np.zeros(embedding_size, dtype=np.float32)

            words = text.lower().split()
            for word in words:
                # Use hash to map words to indices
                idx = hash(word) % embedding_size
                embedding[idx] += 1.0

            # Normalize to unit vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Similarity score between 0 and 1.
        """
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Clamp to [0, 1] range
        return float(max(0.0, min(1.0, (similarity + 1) / 2)))


class IntentRecognizer:
    """
    Recognizes user intent from queries.

    Classifies queries into intent categories:
    - question: Asking for information
    - command: Requesting an action
    - clarification: Following up on previous query
    - new_topic: Starting a new conversation topic
    """

    # Intent patterns (simple keyword-based, could be enhanced with ML)
    INTENT_PATTERNS = {
        "question": [
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "is",
            "are",
            "can",
            "could",
            "would",
        ],
        "command": ["show", "display", "get", "fetch", "list", "find", "search", "execute", "run"],
        "clarification": [
            "also",
            "and",
            "what about",
            "how about",
            "more",
            "another",
            "additionally",
        ],
        "new_topic": ["now", "next", "instead", "different", "change", "switch"],
    }

    def __init__(self):
        """Initialize intent recognizer."""
        logger.info("Initialized IntentRecognizer")

    def recognize_intent(self, query: str, context: list[str] | None = None) -> str:
        """Recognize intent from query.

        Args:
            query: User query.
            context: Previous queries for context.

        Returns:
            Intent category string.
        """
        query_lower = query.lower()

        # Check for clarification intent (requires context)
        if context and len(context) > 0:
            for pattern in self.INTENT_PATTERNS["clarification"]:
                if pattern in query_lower:
                    return "clarification"

        # Check for new topic intent
        for pattern in self.INTENT_PATTERNS["new_topic"]:
            if query_lower.startswith(pattern):
                return "new_topic"

        # Check for command intent
        for pattern in self.INTENT_PATTERNS["command"]:
            if query_lower.startswith(pattern):
                return "command"

        # Check for question intent
        for pattern in self.INTENT_PATTERNS["question"]:
            if pattern in query_lower:
                return "question"

        # Default to question
        return "question"


class SemanticContextAnalyzer:
    """
    Analyzes conversation context using semantic similarity.

    Provides:
    - Semantic similarity-based context matching
    - Topic clustering and detection
    - Intent-aware routing
    - Relevance-based context window
    """

    def __init__(
        self,
        max_history: int = 20,
        similarity_threshold: float = 0.5,
        topic_similarity_threshold: float = 0.6,
        enable_intent_recognition: bool = True,
    ):
        """Initialize semantic context analyzer.

        Args:
            max_history: Maximum queries to track.
            similarity_threshold: Minimum similarity for relevance.
            topic_similarity_threshold: Minimum similarity for same topic.
            enable_intent_recognition: Enable intent recognition.
        """
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.topic_similarity_threshold = topic_similarity_threshold
        self.enable_intent_recognition = enable_intent_recognition

        # Components
        self.similarity_calculator = SemanticSimilarityCalculator()
        self.intent_recognizer = IntentRecognizer() if enable_intent_recognition else None

        # History
        self.query_history: list[SemanticQuery] = []
        self.topic_clusters: list[TopicCluster] = []

        logger.info(
            f"Initialized SemanticContextAnalyzer: "
            f"max_history={max_history}, "
            f"similarity_threshold={similarity_threshold}"
        )

    def add_query(self, query: str, project: str, confidence: float = 0.0):
        """Add query to history with semantic analysis.

        Args:
            query: User query text.
            project: Project that handled the query.
            confidence: Routing confidence.
        """
        # Encode query
        embedding = self.similarity_calculator.encode(query)

        # Recognize intent
        intent = None
        if self.intent_recognizer:
            context_queries = [q.text for q in self.query_history[-3:]]
            intent = self.intent_recognizer.recognize_intent(query, context_queries)

        # Create semantic query
        semantic_query = SemanticQuery(
            text=query, embedding=embedding, project=project, intent=intent
        )

        # Add to history
        self.query_history.append(semantic_query)

        # Maintain max history
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)

        # Update topic clusters
        self._update_topic_clusters(semantic_query)

        logger.debug(
            f"Added query to semantic context: '{query[:50]}...' "
            f"(intent: {intent}, project: {project})"
        )

    def get_relevant_context(self, query: str, max_results: int = 5) -> list[SemanticQuery]:
        """Get relevant context queries based on semantic similarity.

        Args:
            query: Current query.
            max_results: Maximum number of relevant queries to return.

        Returns:
            List of relevant SemanticQuery objects, sorted by relevance.
        """
        if not self.query_history:
            return []

        # Encode current query
        query_embedding = self.similarity_calculator.encode(query)

        # Calculate similarities
        similarities = []
        for hist_query in self.query_history:
            if hist_query.embedding is not None:
                similarity = self.similarity_calculator.calculate_similarity(
                    query_embedding, hist_query.embedding
                )

                if similarity >= self.similarity_threshold:
                    similarities.append((similarity, hist_query))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top results
        relevant = [query for _, query in similarities[:max_results]]

        logger.debug(
            f"Found {len(relevant)} relevant context queries "
            f"(threshold: {self.similarity_threshold})"
        )

        return relevant

    def get_current_topic(self) -> TopicCluster | None:
        """Get current active topic cluster.

        Returns:
            TopicCluster if active topic exists, None otherwise.
        """
        if not self.topic_clusters:
            return None

        # Get most recent cluster
        recent_cluster = max(self.topic_clusters, key=lambda c: c.last_updated)

        # Check if still active (within last 5 minutes)
        age = time.time() - recent_cluster.last_updated
        if age < 300:  # 5 minutes
            return recent_cluster

        return None

    def should_boost_project(self, query: str, project: str) -> tuple[bool, float, str]:
        """Determine if project should get confidence boost.

        Args:
            query: Current query.
            project: Project to check.

        Returns:
            Tuple of (should_boost, boost_amount, reasoning).
        """
        # Check current topic
        current_topic = self.get_current_topic()
        if current_topic and current_topic.dominant_project == project:
            # Query is related to current topic
            query_embedding = self.similarity_calculator.encode(query)
            topic_similarity = self.similarity_calculator.calculate_similarity(
                query_embedding, current_topic.centroid
            )

            if topic_similarity >= self.topic_similarity_threshold:
                boost = 0.2 * topic_similarity  # Scale boost by similarity
                reasoning = f"Semantic topic continuity (similarity: {topic_similarity:.0%})"
                return True, boost, reasoning

        # Check relevant context
        relevant = self.get_relevant_context(query, max_results=3)
        if relevant:
            # Count how many relevant queries used this project
            project_count = sum(1 for q in relevant if q.project == project)
            if project_count >= 2:  # Majority
                boost = 0.15
                reasoning = f"Semantically similar to {project_count} recent queries"
                return True, boost, reasoning

        return False, 0.0, ""

    def get_context_summary(self) -> str:
        """Get human-readable context summary.

        Returns:
            Summary string.
        """
        if not self.query_history:
            return "No semantic context"

        parts = []
        parts.append(f"History: {len(self.query_history)} queries")

        current_topic = self.get_current_topic()
        if current_topic:
            parts.append(
                f"Active topic: {current_topic.dominant_project} "
                f"({len(current_topic.queries)} queries, "
                f"{current_topic.confidence:.0%} confidence)"
            )

        if self.query_history:
            last_query = self.query_history[-1]
            parts.append(f"Last: {last_query.project}")
            if last_query.intent:
                parts.append(f"Intent: {last_query.intent}")

        return " | ".join(parts)

    def clear(self):
        """Clear all context."""
        self.query_history.clear()
        self.topic_clusters.clear()
        logger.info("Cleared semantic context")

    # Private methods

    def _update_topic_clusters(self, query: SemanticQuery):
        """Update topic clusters with new query.

        Args:
            query: New semantic query.
        """
        if query.embedding is None:
            return

        # Find closest cluster
        closest_cluster = None
        closest_similarity = 0.0

        for cluster in self.topic_clusters:
            similarity = self.similarity_calculator.calculate_similarity(
                query.embedding, cluster.centroid
            )

            if similarity > closest_similarity:
                closest_similarity = similarity
                closest_cluster = cluster

        # Add to existing cluster or create new one
        if closest_cluster and closest_similarity >= self.topic_similarity_threshold:
            # Add to existing cluster
            closest_cluster.queries.append(query)
            closest_cluster.last_updated = time.time()

            # Update centroid (moving average)
            alpha = 0.3  # Weight for new query
            closest_cluster.centroid = (
                1 - alpha
            ) * closest_cluster.centroid + alpha * query.embedding

            # Update dominant project
            project_counts = defaultdict(int)
            for q in closest_cluster.queries:
                if q.project:
                    project_counts[q.project] += 1

            if project_counts:
                dominant = max(project_counts.items(), key=lambda x: x[1])
                closest_cluster.dominant_project = dominant[0]
                closest_cluster.confidence = dominant[1] / len(closest_cluster.queries)

            logger.debug(
                f"Added query to existing topic cluster {closest_cluster.topic_id} "
                f"(similarity: {closest_similarity:.2f})"
            )
        else:
            # Create new cluster
            new_cluster = TopicCluster(
                topic_id=len(self.topic_clusters),
                centroid=query.embedding.copy(),
                queries=[query],
                dominant_project=query.project or "unknown",
                confidence=1.0,
                last_updated=time.time(),
            )
            self.topic_clusters.append(new_cluster)

            logger.debug(f"Created new topic cluster {new_cluster.topic_id}")

        # Prune old clusters (keep last 5)
        if len(self.topic_clusters) > 5:
            self.topic_clusters.sort(key=lambda c: c.last_updated, reverse=True)
            self.topic_clusters = self.topic_clusters[:5]
