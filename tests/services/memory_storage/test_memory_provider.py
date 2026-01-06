"""Tests for User Memory Provider."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from osprey.data_management.providers import DataSourceContext
from osprey.data_management.request import DataSourceRequest, DataSourceRequester
from osprey.services.memory_storage.memory_provider import UserMemoryProvider
from osprey.services.memory_storage.models import MemoryContent
from osprey.state import UserMemories


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory storage manager."""
    manager = MagicMock()
    manager.get_all_memory_entries = MagicMock(return_value=[])
    return manager


@pytest.fixture
def memory_provider(mock_memory_manager):
    """Create UserMemoryProvider with mocked manager."""
    with patch(
        "osprey.services.memory_storage.memory_provider.get_memory_storage_manager",
        return_value=mock_memory_manager,
    ):
        return UserMemoryProvider()


@pytest.fixture
def sample_data_source_request():
    """Create a sample DataSourceRequest."""
    requester = DataSourceRequester(component_type="capability", component_name="test")
    return DataSourceRequest(user_id="user123", requester=requester)


class TestUserMemoryProviderInit:
    """Test UserMemoryProvider initialization."""

    @patch("osprey.services.memory_storage.memory_provider.get_memory_storage_manager")
    def test_initializes_with_memory_manager(self, mock_get_manager):
        """UserMemoryProvider should initialize with memory manager."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        provider = UserMemoryProvider()

        assert provider._memory_manager is mock_manager
        mock_get_manager.assert_called_once()


class TestUserMemoryProviderProperties:
    """Test UserMemoryProvider properties."""

    def test_name_property(self, memory_provider):
        """Provider should have correct name."""
        assert memory_provider.name == "core_user_memory"

    def test_context_type_property(self, memory_provider):
        """Provider should have correct context type."""
        assert memory_provider.context_type == "CORE_MEMORY_CONTEXT"

    def test_description_property(self, memory_provider):
        """Provider should have meaningful description."""
        description = memory_provider.description
        assert "memory" in description.lower()
        assert isinstance(description, str)
        assert len(description) > 0


class TestShouldRespond:
    """Test should_respond method."""

    def test_responds_when_user_id_present(self, memory_provider, sample_data_source_request):
        """should_respond should return True when user_id is present."""
        assert memory_provider.should_respond(sample_data_source_request) is True

    def test_does_not_respond_when_user_id_missing(self, memory_provider):
        """should_respond should return False when user_id is None."""
        requester = DataSourceRequester(component_type="capability", component_name="test")
        request = DataSourceRequest(user_id=None, requester=requester)

        assert memory_provider.should_respond(request) is False

    def test_responds_to_empty_string_user_id(self, memory_provider):
        """should_respond returns True for empty string (checks is not None)."""
        requester = DataSourceRequester(component_type="capability", component_name="test")
        request = DataSourceRequest(user_id="", requester=requester)

        # Empty string is not None, so should_respond returns True
        # (retrieve_data will handle empty user_id appropriately)
        assert memory_provider.should_respond(request) is True


class TestRetrieveData:
    """Test retrieve_data method."""

    @pytest.mark.asyncio
    async def test_retrieves_memory_for_valid_user(
        self, memory_provider, mock_memory_manager, sample_data_source_request
    ):
        """retrieve_data should retrieve memory entries for valid user."""
        # Mock memory entries
        mock_entries = [
            MemoryContent(timestamp=datetime.now(), content="Memory 1"),
            MemoryContent(timestamp=datetime.now(), content="Memory 2"),
        ]
        mock_memory_manager.get_all_memory_entries.return_value = mock_entries

        context = await memory_provider.retrieve_data(sample_data_source_request)

        assert context is not None
        assert isinstance(context, DataSourceContext)
        assert context.source_name == "core_user_memory"
        assert context.context_type == "CORE_MEMORY_CONTEXT"

        # Verify data structure
        assert isinstance(context.data, UserMemories)
        assert len(context.data.entries) == 2

    @pytest.mark.asyncio
    async def test_returns_none_when_no_user_id(self, memory_provider):
        """retrieve_data should return None when user_id is missing."""
        requester = DataSourceRequester(component_type="capability", component_name="test")
        request = DataSourceRequest(user_id=None, requester=requester)

        context = await memory_provider.retrieve_data(request)

        assert context is None

    @pytest.mark.asyncio
    async def test_returns_empty_context_for_no_memories(
        self, memory_provider, mock_memory_manager, sample_data_source_request
    ):
        """retrieve_data should return empty context when no memories exist."""
        mock_memory_manager.get_all_memory_entries.return_value = []

        context = await memory_provider.retrieve_data(sample_data_source_request)

        assert context is not None
        assert isinstance(context.data, UserMemories)
        assert len(context.data.entries) == 0
        assert context.metadata["entry_count"] == 0

    @pytest.mark.asyncio
    async def test_handles_storage_manager_exceptions(
        self, memory_provider, mock_memory_manager, sample_data_source_request
    ):
        """retrieve_data should handle storage manager exceptions gracefully."""
        mock_memory_manager.get_all_memory_entries.side_effect = Exception("Storage error")

        context = await memory_provider.retrieve_data(sample_data_source_request)

        # Should return None on error, not raise
        assert context is None

    @pytest.mark.asyncio
    async def test_converts_memory_entries_to_string_list(
        self, memory_provider, mock_memory_manager, sample_data_source_request
    ):
        """retrieve_data should convert MemoryContent objects to string list."""
        mock_entries = [
            MemoryContent(timestamp=datetime.now(), content="First memory"),
            MemoryContent(timestamp=datetime.now(), content="Second memory"),
        ]
        mock_memory_manager.get_all_memory_entries.return_value = mock_entries

        context = await memory_provider.retrieve_data(sample_data_source_request)

        assert context.data.entries == ["First memory", "Second memory"]

    @pytest.mark.asyncio
    async def test_includes_metadata_in_context(
        self, memory_provider, mock_memory_manager, sample_data_source_request
    ):
        """retrieve_data should include comprehensive metadata."""
        mock_entries = [MemoryContent(timestamp=datetime.now(), content="Memory")]
        mock_memory_manager.get_all_memory_entries.return_value = mock_entries

        context = await memory_provider.retrieve_data(sample_data_source_request)

        assert "user_id" in context.metadata
        assert "entry_count" in context.metadata
        assert "source_description" in context.metadata
        assert "is_core_provider" in context.metadata
        assert context.metadata["entry_count"] == 1
        assert context.metadata["is_core_provider"] is True

    @pytest.mark.asyncio
    async def test_warns_about_query_based_retrieval(
        self, memory_provider, mock_memory_manager, sample_data_source_request
    ):
        """retrieve_data should warn if query parameter is provided."""
        sample_data_source_request.query = "some query"
        mock_memory_manager.get_all_memory_entries.return_value = []

        with patch("osprey.services.memory_storage.memory_provider.logger") as mock_logger:
            await memory_provider.retrieve_data(sample_data_source_request)

            # Should log warning about query not being supported
            mock_logger.warning.assert_called_once()


class TestGetConfigRequirements:
    """Test get_config_requirements method."""

    def test_returns_memory_directory_requirement(self, memory_provider):
        """get_config_requirements should specify memory directory requirement."""
        requirements = memory_provider.get_config_requirements()

        assert "memory_directory" in requirements
        assert requirements["memory_directory"]["required"] is True
        assert requirements["memory_directory"]["type"] == "string"
        assert "config_path" in requirements["memory_directory"]


class TestHealthCheck:
    """Test health_check method."""

    @pytest.mark.asyncio
    async def test_passes_when_memory_manager_available(self, memory_provider):
        """health_check should pass when memory manager is available."""
        result = await memory_provider.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_fails_when_memory_manager_is_none(self):
        """health_check should fail when memory manager is None."""
        with patch(
            "osprey.services.memory_storage.memory_provider.get_memory_storage_manager",
            return_value=None,
        ):
            provider = UserMemoryProvider()
            result = await provider.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_basic_functionality(self, memory_provider):
        """health_check performs basic functionality check."""
        # The implementation checks if _memory_manager is not None
        # This is a simple availability check
        result = await memory_provider.health_check()

        # Should return True when manager exists
        assert result is True


class TestFormatForPrompt:
    """Test format_for_prompt method."""

    def test_formats_memory_with_entries(self, memory_provider):
        """format_for_prompt should format memory entries for LLM."""
        user_memories = UserMemories(entries=["Memory 1", "Memory 2", "Memory 3"])
        context = DataSourceContext(
            source_name="core_user_memory",
            context_type="CORE_MEMORY_CONTEXT",
            data=user_memories,
            metadata={"entry_count": 3},
            provider=memory_provider,
        )

        formatted = memory_provider.format_for_prompt(context)

        assert "🧠 User Memory" in formatted
        assert "(3 saved entries)" in formatted
        assert "• Memory 1" in formatted
        assert "• Memory 2" in formatted
        assert "• Memory 3" in formatted

    def test_formats_memory_without_entries_attribute(self, memory_provider):
        """format_for_prompt handles objects without entries attribute."""
        # Create a mock data object without entries attribute
        mock_data = MagicMock(spec=[])  # No attributes
        context = DataSourceContext(
            source_name="core_user_memory",
            context_type="CORE_MEMORY_CONTEXT",
            data=mock_data,
            metadata={"entry_count": 0},
            provider=memory_provider,
        )

        formatted = memory_provider.format_for_prompt(context)

        # Should still format header and show no entries
        assert "🧠 User Memory" in formatted
        assert "(No memory entries available)" in formatted

    def test_returns_empty_string_for_none_context(self, memory_provider):
        """format_for_prompt should return empty string for None context."""
        formatted = memory_provider.format_for_prompt(None)

        assert formatted == ""

    def test_returns_empty_string_for_context_without_data(self, memory_provider):
        """format_for_prompt should return empty string if context has no data."""
        context = DataSourceContext(
            source_name="core_user_memory",
            context_type="CORE_MEMORY_CONTEXT",
            data=None,
            metadata={},
            provider=memory_provider,
        )

        formatted = memory_provider.format_for_prompt(context)

        assert formatted == ""

    def test_formatted_output_is_well_structured(self, memory_provider):
        """format_for_prompt should produce well-structured output."""
        user_memories = UserMemories(entries=["Important info"])
        context = DataSourceContext(
            source_name="core_user_memory",
            context_type="CORE_MEMORY_CONTEXT",
            data=user_memories,
            metadata={"entry_count": 1},
            provider=memory_provider,
        )

        formatted = memory_provider.format_for_prompt(context)

        # Should have multiple lines
        lines = formatted.split("\n")
        assert len(lines) > 2

        # Should use bullet points for entries
        assert any("•" in line for line in lines)
