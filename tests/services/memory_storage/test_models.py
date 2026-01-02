"""Tests for memory_storage/models.py - Memory storage data models."""

from datetime import datetime

import pytest

from osprey.services.memory_storage.models import MemoryContent


class TestMemoryContent:
    """Tests for MemoryContent model."""

    def test_create_memory_content(self):
        """Test creating a valid memory content entry."""
        timestamp = datetime(2025, 1, 15, 14, 30)
        content = "User prefers morning meetings"

        memory = MemoryContent(timestamp=timestamp, content=content)

        assert memory.timestamp == timestamp
        assert memory.content == content

    def test_format_for_llm_basic(self):
        """Test basic LLM formatting."""
        timestamp = datetime(2025, 1, 15, 14, 30)
        content = "User prefers morning meetings"

        memory = MemoryContent(timestamp=timestamp, content=content)
        formatted = memory.format_for_llm()

        assert formatted == "[2025-01-15 14:30] User prefers morning meetings"

    def test_format_for_llm_complex_content(self):
        """Test LLM formatting with complex content."""
        timestamp = datetime(2025, 12, 23, 9, 15)
        content = "Working on project Alpha with team lead Sarah and focusing on ML models"

        memory = MemoryContent(timestamp=timestamp, content=content)
        formatted = memory.format_for_llm()

        assert "[2025-12-23 09:15]" in formatted
        assert "Working on project Alpha" in formatted
        assert content in formatted

    def test_format_for_llm_preserves_content(self):
        """Test that formatting preserves all content."""
        timestamp = datetime(2025, 6, 10, 16, 45)
        content = "Special characters: @#$%, numbers: 123, Unicode: café"

        memory = MemoryContent(timestamp=timestamp, content=content)
        formatted = memory.format_for_llm()

        assert content in formatted
        assert "@#$%" in formatted
        assert "café" in formatted

    def test_timestamp_format_consistency(self):
        """Test timestamp format is consistent across different times."""
        test_cases = [
            (datetime(2025, 1, 1, 0, 0), "[2025-01-01 00:00]"),
            (datetime(2025, 12, 31, 23, 59), "[2025-12-31 23:59]"),
            (datetime(2025, 6, 15, 12, 30), "[2025-06-15 12:30]"),
        ]

        for timestamp, expected_prefix in test_cases:
            memory = MemoryContent(timestamp=timestamp, content="test")
            formatted = memory.format_for_llm()
            assert formatted.startswith(expected_prefix)

    def test_pydantic_validation(self):
        """Test Pydantic validation of required fields."""
        from pydantic import ValidationError

        # Missing content should raise validation error
        with pytest.raises(ValidationError):
            MemoryContent(timestamp=datetime.now())

        # Missing timestamp should raise validation error
        with pytest.raises(ValidationError):
            MemoryContent(content="test content")

    def test_model_serialization(self):
        """Test model serialization to dict."""
        timestamp = datetime(2025, 1, 15, 14, 30)
        content = "Test content"

        memory = MemoryContent(timestamp=timestamp, content=content)
        data = memory.model_dump()

        assert "timestamp" in data
        assert "content" in data
        assert data["content"] == content
        assert data["timestamp"] == timestamp

    def test_model_deserialization(self):
        """Test model deserialization from dict."""
        data = {"timestamp": datetime(2025, 1, 15, 14, 30), "content": "Test content"}

        memory = MemoryContent(**data)

        assert memory.timestamp == data["timestamp"]
        assert memory.content == data["content"]

    def test_json_serialization(self):
        """Test JSON serialization with custom datetime."""
        timestamp = datetime(2025, 1, 15, 14, 30)
        content = "Test content"

        memory = MemoryContent(timestamp=timestamp, content=content)

        # Should be able to serialize to JSON
        json_str = memory.model_dump_json()
        assert isinstance(json_str, str)
        assert content in json_str

    def test_multiple_memories_different_timestamps(self):
        """Test creating multiple memories with different timestamps."""
        memories = [
            MemoryContent(timestamp=datetime(2025, 1, 1, 10, 0), content="First memory"),
            MemoryContent(timestamp=datetime(2025, 1, 2, 11, 0), content="Second memory"),
            MemoryContent(timestamp=datetime(2025, 1, 3, 12, 0), content="Third memory"),
        ]

        formatted = [m.format_for_llm() for m in memories]

        assert len(formatted) == 3
        assert "[2025-01-01 10:00] First memory" in formatted
        assert "[2025-01-02 11:00] Second memory" in formatted
        assert "[2025-01-03 12:00] Third memory" in formatted

    def test_empty_content(self):
        """Test memory with empty content string."""
        timestamp = datetime(2025, 1, 15, 14, 30)
        memory = MemoryContent(timestamp=timestamp, content="")

        assert memory.content == ""
        formatted = memory.format_for_llm()
        assert formatted == "[2025-01-15 14:30] "

    def test_whitespace_content(self):
        """Test memory with whitespace content."""
        timestamp = datetime(2025, 1, 15, 14, 30)
        content = "   spaces around   "
        memory = MemoryContent(timestamp=timestamp, content=content)

        # Content is preserved as-is
        assert memory.content == content
        formatted = memory.format_for_llm()
        assert content in formatted

    def test_multiline_content(self):
        """Test memory with multiline content."""
        timestamp = datetime(2025, 1, 15, 14, 30)
        content = "Line 1\nLine 2\nLine 3"
        memory = MemoryContent(timestamp=timestamp, content=content)

        formatted = memory.format_for_llm()
        assert "Line 1" in formatted
        assert "Line 2" in formatted
        assert "Line 3" in formatted
        assert formatted.startswith("[2025-01-15 14:30]")
