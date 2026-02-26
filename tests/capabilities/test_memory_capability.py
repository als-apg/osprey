"""Tests for MemoryOperationsCapability."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from osprey.base.errors import ErrorSeverity
from osprey.capabilities.memory import (
    ContentExtractionError,
    LLMCallError,
    MemoryCapabilityError,
    MemoryContentExtraction,
    MemoryContext,
    MemoryExtractionExample,
    MemoryFileError,
    MemoryOperation,
    MemoryOperationClassification,
    MemoryOperationsCapability,
    MemoryRetrievalError,
    UserIdNotAvailableError,
    _classify_memory_operation,
    _get_memory_extraction_prompt,
    _perform_memory_retrieve_operation,
    _perform_memory_save_operation,
)

# =============================================================================
# Test Memory Context
# =============================================================================


class TestMemoryContext:
    """Test MemoryContext class."""

    def test_memory_context_creation(self):
        """Test creating a memory context."""
        ctx = MemoryContext(
            memory_data={"test": "data"}, operation_type="save", operation_result="Success"
        )

        assert ctx.memory_data == {"test": "data"}
        assert ctx.operation_type == "save"
        assert ctx.operation_result == "Success"
        assert ctx.CONTEXT_TYPE == "MEMORY_CONTEXT"
        assert ctx.CONTEXT_CATEGORY == "CONTEXTUAL_KNOWLEDGE"

    def test_get_access_details(self):
        """Test get_access_details method."""
        ctx = MemoryContext(
            memory_data={"user_prefs": "value"}, operation_type="retrieve", operation_result="Found"
        )

        details = ctx.get_access_details("test_key")

        assert details["operation"] == "retrieve"
        assert "user_prefs" in details["data_keys"]
        assert "context.MEMORY_CONTEXT.test_key.memory_data" in details["access_pattern"]
        assert details["operation_result"] == "Found"

    def test_get_summary(self):
        """Test get_summary method."""
        ctx = MemoryContext(
            memory_data={"memories": ["item1", "item2"]},
            operation_type="retrieve",
            operation_result="Retrieved 2 entries",
        )

        summary = ctx.get_summary()

        assert summary["operation_type"] == "retrieve"
        assert summary["operation_result"] == "Retrieved 2 entries"
        assert summary["memory_data"]["memories"] == ["item1", "item2"]


# =============================================================================
# Test Memory Operation Classification
# =============================================================================


class TestMemoryOperationClassification:
    """Test MemoryOperation and related classification models."""

    def test_memory_operation_enum_values(self):
        """Test MemoryOperation enum has correct values."""
        assert MemoryOperation.SAVE.value == "save"
        assert MemoryOperation.RETRIEVE.value == "retrieve"

    def test_memory_operation_classification_model(self):
        """Test MemoryOperationClassification model."""
        classification = MemoryOperationClassification(
            operation="save", reasoning="User wants to save content"
        )

        assert classification.operation == "save"
        assert classification.reasoning == "User wants to save content"

    def test_memory_content_extraction_model(self):
        """Test MemoryContentExtraction model."""
        extraction = MemoryContentExtraction(
            content="Important info", found=True, explanation="Found user preference"
        )

        assert extraction.content == "Important info"
        assert extraction.found is True
        assert extraction.explanation == "Found user preference"


# =============================================================================
# Test Memory Extraction Example
# =============================================================================


class TestMemoryExtractionExample:
    """Test MemoryExtractionExample class."""

    def test_format_for_prompt(self):
        """Test format_for_prompt method."""
        messages = [HumanMessage(content="I prefer dark mode")]
        expected_output = MemoryContentExtraction(
            content="User prefers dark mode", found=True, explanation="UI preference"
        )

        example = MemoryExtractionExample(messages=messages, expected_output=expected_output)
        formatted = example.format_for_prompt()

        assert "Chat History:" in formatted
        assert "Expected Output:" in formatted
        assert "User prefers dark mode" in formatted
        assert "true" in formatted


# =============================================================================
# Test Exception Classes
# =============================================================================


class TestExceptionClasses:
    """Test memory-specific exception classes."""

    def test_memory_capability_error_inheritance(self):
        """Test base MemoryCapabilityError."""
        error = MemoryCapabilityError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_user_id_not_available_error(self):
        """Test UserIdNotAvailableError."""
        error = UserIdNotAvailableError("No user ID")
        assert isinstance(error, MemoryCapabilityError)
        assert str(error) == "No user ID"

    def test_content_extraction_error(self):
        """Test ContentExtractionError."""
        error = ContentExtractionError("Failed to extract")
        assert isinstance(error, MemoryCapabilityError)

    def test_memory_file_error(self):
        """Test MemoryFileError."""
        error = MemoryFileError("File write failed")
        assert isinstance(error, MemoryCapabilityError)

    def test_memory_retrieval_error(self):
        """Test MemoryRetrievalError."""
        error = MemoryRetrievalError("Retrieval failed")
        assert isinstance(error, MemoryCapabilityError)

    def test_llm_call_error(self):
        """Test LLMCallError."""
        error = LLMCallError("LLM call failed")
        assert isinstance(error, MemoryCapabilityError)


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestMemoryHelperFunctions:
    """Test memory helper functions."""

    def test_get_memory_extraction_prompt(self):
        """Test _get_memory_extraction_prompt function."""
        with patch("osprey.capabilities.memory.get_framework_prompts") as mock_prompts:
            mock_builder = MagicMock()
            mock_builder.build_prompt.return_value = "Test instructions"
            mock_prompts.return_value.get_memory_extraction_prompt_builder.return_value = (
                mock_builder
            )

            instructions = _get_memory_extraction_prompt()

            assert instructions == "Test instructions"
            mock_prompts.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_memory_operation_save(self):
        """Test classifying save operation."""
        mock_logger = MagicMock()

        with patch("osprey.capabilities.memory.get_chat_completion") as mock_completion:
            mock_completion.return_value = MemoryOperationClassification(
                operation="save", reasoning="User wants to save"
            )

            with patch("osprey.capabilities.memory.get_framework_prompts"):
                with patch("osprey.capabilities.memory.get_model_config"):
                    operation = await _classify_memory_operation("save this", mock_logger)

                    assert operation == MemoryOperation.SAVE

    @pytest.mark.asyncio
    async def test_classify_memory_operation_retrieve(self):
        """Test classifying retrieve operation."""
        mock_logger = MagicMock()

        with patch("osprey.capabilities.memory.get_chat_completion") as mock_completion:
            mock_completion.return_value = MemoryOperationClassification(
                operation="retrieve", reasoning="User wants to see memory"
            )

            with patch("osprey.capabilities.memory.get_framework_prompts"):
                with patch("osprey.capabilities.memory.get_model_config"):
                    operation = await _classify_memory_operation("show memory", mock_logger)

                    assert operation == MemoryOperation.RETRIEVE

    @pytest.mark.asyncio
    async def test_classify_memory_operation_invalid(self):
        """Test classification with invalid operation."""
        mock_logger = MagicMock()

        with patch("osprey.capabilities.memory.get_chat_completion") as mock_completion:
            mock_completion.return_value = MemoryOperationClassification(
                operation="invalid", reasoning="Unknown"
            )

            with patch("osprey.capabilities.memory.get_framework_prompts"):
                with patch("osprey.capabilities.memory.get_model_config"):
                    with pytest.raises(ContentExtractionError):
                        await _classify_memory_operation("test", mock_logger)

    @pytest.mark.asyncio
    async def test_perform_memory_save_operation_success(self):
        """Test successful memory save operation."""
        mock_logger = MagicMock()

        with patch("osprey.capabilities.memory.get_memory_storage_manager") as mock_mgr:
            mock_manager = MagicMock()
            mock_manager.add_memory_entry.return_value = True
            mock_mgr.return_value = mock_manager

            context = await _perform_memory_save_operation("test content", "user123", mock_logger)

            assert isinstance(context, MemoryContext)
            assert context.operation_type == "save"
            assert context.memory_data["saved_content"] == "test content"
            assert "timestamp" in context.memory_data

    @pytest.mark.asyncio
    async def test_perform_memory_save_operation_failure(self):
        """Test failed memory save operation."""
        mock_logger = MagicMock()

        with patch("osprey.capabilities.memory.get_memory_storage_manager") as mock_mgr:
            mock_manager = MagicMock()
            mock_manager.add_memory_entry.return_value = False
            mock_mgr.return_value = mock_manager

            with pytest.raises(MemoryFileError):
                await _perform_memory_save_operation("test", "user123", mock_logger)

    @pytest.mark.asyncio
    async def test_perform_memory_retrieve_operation(self):
        """Test memory retrieve operation."""
        mock_logger = MagicMock()
        mock_entries = [{"content": "memory1"}, {"content": "memory2"}]

        with patch("osprey.capabilities.memory.get_memory_storage_manager") as mock_mgr:
            mock_manager = MagicMock()
            mock_manager.get_all_memory_entries.return_value = mock_entries
            mock_mgr.return_value = mock_manager

            context = await _perform_memory_retrieve_operation("user123", mock_logger)

            assert isinstance(context, MemoryContext)
            assert context.operation_type == "retrieve"
            assert context.memory_data["memories"] == mock_entries
            assert "Retrieved 2 memory entries" in context.operation_result


# =============================================================================
# Test Capability Instance Methods
# =============================================================================


class TestMemoryCapabilityMigration:
    """Test MemoryOperationsCapability successfully migrated to instance method pattern."""

    def test_uses_instance_method_not_static(self):
        """Verify execute() migrated from @staticmethod to instance method."""
        execute_method = inspect.getattr_static(MemoryOperationsCapability, "execute")
        assert not isinstance(execute_method, staticmethod)

        sig = inspect.signature(MemoryOperationsCapability.execute)
        params = list(sig.parameters.keys())
        assert params == ["self"]

    def test_state_can_be_injected(self, mock_state, mock_step):
        """Verify capability instance can receive _state and _step injection."""
        capability = MemoryOperationsCapability()
        capability._state = mock_state
        capability._step = mock_step

        assert capability._state == mock_state
        assert capability._step == mock_step

    def test_has_langgraph_node_decorator(self):
        """Verify @capability_node decorator created langgraph_node attribute."""
        assert hasattr(MemoryOperationsCapability, "langgraph_node")
        assert callable(MemoryOperationsCapability.langgraph_node)


# =============================================================================
# Test Error Classification
# =============================================================================


class TestMemoryCapabilityErrorClassification:
    """Test error classification in MemoryOperationsCapability."""

    def test_classify_user_id_not_available_error(self):
        """Test classification of UserIdNotAvailableError."""
        exc = UserIdNotAvailableError("No user ID")
        classification = MemoryOperationsCapability.classify_error(exc, {})

        assert classification.severity == ErrorSeverity.CRITICAL
        assert "user ID not available" in classification.user_message

    def test_classify_content_extraction_error(self):
        """Test classification of ContentExtractionError."""
        exc = ContentExtractionError("Failed to extract")
        classification = MemoryOperationsCapability.classify_error(exc, {})

        assert classification.severity == ErrorSeverity.REPLANNING
        assert "clarification" in classification.user_message

    def test_classify_memory_retrieval_error(self):
        """Test classification of MemoryRetrievalError."""
        exc = MemoryRetrievalError("Retrieval failed")
        classification = MemoryOperationsCapability.classify_error(exc, {})

        assert classification.severity == ErrorSeverity.RETRIABLE
        assert "retrying" in classification.user_message

    def test_classify_memory_file_error(self):
        """Test classification of MemoryFileError."""
        exc = MemoryFileError("File error")
        classification = MemoryOperationsCapability.classify_error(exc, {})

        assert classification.severity == ErrorSeverity.RETRIABLE
        assert "retrying" in classification.user_message

    def test_classify_generic_error(self):
        """Test classification of generic errors."""
        exc = Exception("Generic error")
        classification = MemoryOperationsCapability.classify_error(exc, {})

        assert classification.severity == ErrorSeverity.RETRIABLE
        assert "Generic error" in classification.user_message


# =============================================================================
# Test Capability Execution
# =============================================================================


class TestMemoryCapabilityApprovalPath:
    """Test specific execution path that's critical for migration validation."""

    @pytest.mark.asyncio
    async def test_state_injection_in_approval_path(self, mock_state, mock_step, monkeypatch):
        """Test approved operation path validates state injection works."""
        from unittest.mock import MagicMock

        # Mock get_session_info
        monkeypatch.setattr(
            "osprey.capabilities.memory.get_session_info",
            MagicMock(return_value={"user_id": "test_user_123"}),
        )

        # Approved payload (bypasses complex internal logic)
        approved_payload = {"content": "Test memory content", "user_id": "test_user_123"}
        monkeypatch.setattr(
            "osprey.capabilities.memory.get_approval_resume_data",
            MagicMock(return_value=(True, approved_payload)),
        )

        mock_sm = MagicMock()
        mock_sm.store_context.return_value = {"context_data": {}}
        monkeypatch.setattr("osprey.capabilities.memory.StateManager", mock_sm)

        # Mock the save operation
        async def mock_save(content, user_id, logger):
            return MagicMock(success=True, memory_id="mem_123")

        monkeypatch.setattr("osprey.capabilities.memory._perform_memory_save_operation", mock_save)

        # Create instance and inject state/step
        capability = MemoryOperationsCapability()
        capability._state = mock_state
        capability._step = mock_step

        # Execute - validates self._state and self._step are accessible
        result = await capability.execute()

        assert isinstance(result, dict)
        assert "context_data" in result

    @pytest.mark.asyncio
    async def test_execute_without_user_id_fails(self, mock_state, mock_step, monkeypatch):
        """Test execution fails when user ID is not available."""
        # No approval resume data
        monkeypatch.setattr(
            "osprey.capabilities.memory.get_approval_resume_data",
            MagicMock(return_value=(False, None)),
        )

        # No user ID in session
        monkeypatch.setattr(
            "osprey.capabilities.memory.get_session_info", MagicMock(return_value={})
        )

        capability = MemoryOperationsCapability()
        capability._state = mock_state
        capability._step = mock_step

        with pytest.raises(UserIdNotAvailableError):
            await capability.execute()


# =============================================================================
# Test Orchestrator and Classifier Guides
# =============================================================================


class TestMemoryCapabilityGuides:
    """Test orchestrator and classifier guide creation."""

    def test_create_orchestrator_guide(self):
        """Test _create_orchestrator_guide method."""
        with patch("osprey.capabilities.memory.get_framework_prompts") as mock_prompts:
            mock_guide = MagicMock()
            mock_builder = MagicMock()
            mock_builder.get_orchestrator_guide.return_value = mock_guide
            mock_prompts.return_value.get_memory_extraction_prompt_builder.return_value = (
                mock_builder
            )

            capability = MemoryOperationsCapability()
            guide = capability._create_orchestrator_guide()

            assert guide == mock_guide

    def test_create_classifier_guide(self):
        """Test _create_classifier_guide method."""
        with patch("osprey.capabilities.memory.get_framework_prompts") as mock_prompts:
            mock_guide = MagicMock()
            mock_builder = MagicMock()
            mock_builder.get_classifier_guide.return_value = mock_guide
            mock_prompts.return_value.get_memory_extraction_prompt_builder.return_value = (
                mock_builder
            )

            capability = MemoryOperationsCapability()
            guide = capability._create_classifier_guide()

            assert guide == mock_guide
