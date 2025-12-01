"""Unit tests for BaseCapability helper methods (new simplified pattern)."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import Mock

import pytest

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.context import CapabilityContext


# Test context types (prefix with _ to avoid pytest collection warnings)
class _TestContext(CapabilityContext):
    """Test context for unit testing."""
    CONTEXT_TYPE: ClassVar[str] = "TEST_DATA"
    CONTEXT_CATEGORY: ClassVar[str] = "TEST"
    value: str

    def get_access_details(self, key: str) -> dict:
        """Get access details for testing."""
        return {"key": key, "value": self.value}

    def get_summary(self) -> dict:
        """Get summary for testing."""
        return {"value": self.value}


class _TestMultipleContext(CapabilityContext):
    """Another test context."""
    CONTEXT_TYPE: ClassVar[str] = "MULTIPLE_DATA"
    CONTEXT_CATEGORY: ClassVar[str] = "TEST"
    values: list[str]

    def get_access_details(self, key: str) -> dict:
        """Get access details for testing."""
        return {"key": key, "values": self.values}

    def get_summary(self) -> dict:
        """Get summary for testing."""
        return {"count": len(self.values)}


# ========================================
# Test get_required_contexts()
# ========================================


def test_get_required_contexts_simple(mock_state, mock_step, mock_registry):
    """Test simple context extraction with string requirements."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"
        requires = ["TEST_DATA"]

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    # Inject state (mimics decorator)
    instance._state = mock_state
    instance._step = mock_step

    # Mock the registry and context manager
    with mock_registry:
        contexts = instance.get_required_contexts()

    # Verify we got the expected context
    assert "TEST_DATA" in contexts
    assert isinstance(contexts["TEST_DATA"], _TestContext)
    assert contexts["TEST_DATA"].value == "test"


def test_get_required_contexts_with_cardinality(mock_state, mock_step, mock_registry):
    """Test context extraction with cardinality constraints."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"
        requires = [
            "TEST_DATA",
            ("MULTIPLE_DATA", "single")
        ]

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = mock_state
    instance._step = mock_step

    with mock_registry:
        contexts = instance.get_required_contexts()

    # Verify both contexts are present
    assert "TEST_DATA" in contexts
    assert "MULTIPLE_DATA" in contexts

    # Verify they're the correct types
    assert isinstance(contexts["TEST_DATA"], _TestContext)
    assert isinstance(contexts["MULTIPLE_DATA"], _TestMultipleContext)

    # Verify actual data
    assert contexts["TEST_DATA"].value == "test"
    assert contexts["MULTIPLE_DATA"].values == ["test1", "test2"]


def test_get_required_contexts_soft_mode(mock_state, mock_step, monkeypatch):
    """Test context extraction with soft constraint mode."""
    from unittest.mock import MagicMock

    class TestCap(BaseCapability):
        name = "test"
        description = "test"
        requires = ["TEST_DATA", "OPTIONAL_DATA"]

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = mock_state
    instance._step = mock_step

    # Mock registry
    mock_reg = MagicMock()
    mock_reg.context_types.TEST_DATA = "TEST_DATA"
    mock_reg.context_types.OPTIONAL_DATA = "OPTIONAL_DATA"

    # Mock context manager - only returns TEST_DATA (soft mode allows this)
    mock_cm = MagicMock()
    mock_cm.extract_from_step.return_value = {
        "TEST_DATA": _TestContext(value="test")
        # OPTIONAL_DATA is missing - soft mode should handle this
    }

    monkeypatch.setattr('osprey.registry.get_registry', lambda: mock_reg)
    monkeypatch.setattr('osprey.context.context_manager.ContextManager', lambda state: mock_cm)

    # Should not raise even if only one context exists
    contexts = instance.get_required_contexts(constraint_mode="soft")

    # At least one should be present (our mock returns TEST_DATA)
    assert len(contexts) > 0
    assert "TEST_DATA" in contexts

    # Verify its content
    assert isinstance(contexts["TEST_DATA"], _TestContext)
    assert contexts["TEST_DATA"].value == "test"


def test_get_required_contexts_no_requirements():
    """Test that empty requires returns empty dict."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"
        requires = []

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"test": "state"}
    instance._step = {"test": "step"}

    contexts = instance.get_required_contexts()
    assert contexts == {}


def test_get_required_contexts_without_state_injection():
    """Test that calling outside execute() raises clear error."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"
        requires = ["TEST_DATA"]

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    # Don't inject state

    with pytest.raises(RuntimeError, match="requires self._state"):
        instance.get_required_contexts()


def test_get_required_contexts_tuple_unpacking(mock_state, mock_step, mock_registry):
    """Test elegant tuple unpacking syntax matches order in requires field."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"
        requires = ["TEST_DATA", "MULTIPLE_DATA"]

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = mock_state
    instance._step = mock_step

    with mock_registry:
        # Test tuple unpacking (order matches requires field)
        test_data, multiple_data = instance.get_required_contexts()

        # Verify unpacking worked correctly
        assert isinstance(test_data, _TestContext)
        assert test_data.value == "test"
        assert isinstance(multiple_data, _TestMultipleContext)
        assert multiple_data.values == ["test1", "test2"]

        # Verify dict access still works (backward compatibility)
        contexts = instance.get_required_contexts()

        assert "TEST_DATA" in contexts
        assert "MULTIPLE_DATA" in contexts
        assert contexts["TEST_DATA"].value == "test"
        assert contexts["MULTIPLE_DATA"].values == ["test1", "test2"]


def test_get_required_contexts_single_value_unpacking(mock_state, mock_step, mock_registry):
    """Test tuple unpacking with single requirement."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"
        requires = ["TEST_DATA"]

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = mock_state
    instance._step = mock_step

    with mock_registry:
        # Single value unpacking requires trailing comma
        test_data, = instance.get_required_contexts()

    assert isinstance(test_data, _TestContext)
    assert test_data.value == "test"


# ========================================
# Test process_extracted_contexts()
# ========================================


def test_process_extracted_contexts_default():
    """Test that default implementation returns contexts unchanged."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    test_contexts = {"TEST": Mock()}

    result = instance.process_extracted_contexts(test_contexts)
    assert result is test_contexts  # Should return same object


def test_process_extracted_contexts_custom():
    """Test custom processing hook."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

        def process_extracted_contexts(self, contexts):
            """Flatten lists."""
            if isinstance(contexts.get("TEST"), list):
                contexts["TEST"] = [item for sublist in contexts["TEST"] for item in sublist]
            return contexts

    instance = TestCap()
    test_contexts = {"TEST": [[1, 2], [3, 4]]}

    result = instance.process_extracted_contexts(test_contexts)
    assert result["TEST"] == [1, 2, 3, 4]


# ========================================
# Test get_parameters()
# ========================================


def test_get_parameters_with_values():
    """Test getting parameters when they exist in step."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"test": "state"}
    instance._step = {
        "context_key": "test_key",
        "parameters": {
            "precision_ms": 500,
            "timeout": 30,
            "mode": "fast"
        }
    }

    params = instance.get_parameters()

    assert params == {"precision_ms": 500, "timeout": 30, "mode": "fast"}
    assert params.get("precision_ms") == 500
    assert params.get("timeout") == 30
    assert params.get("mode") == "fast"


def test_get_parameters_empty():
    """Test getting parameters when step has no parameters."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"test": "state"}
    instance._step = {"context_key": "test_key"}  # No parameters!

    params = instance.get_parameters()

    assert params == {}


def test_get_parameters_with_custom_default():
    """Test getting parameters with custom default value."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"test": "state"}
    instance._step = {"context_key": "test_key"}  # No parameters!

    custom_default = {"precision_ms": 1000, "timeout": 60}
    params = instance.get_parameters(default=custom_default)

    assert params == custom_default
    assert params.get("precision_ms") == 1000


def test_get_parameters_without_step():
    """Test error when called outside execute()."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    # Don't inject step

    with pytest.raises(RuntimeError, match="requires self._step"):
        instance.get_parameters()


def test_get_parameters_usage_pattern():
    """Test realistic usage pattern with fallback values."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"test": "state"}
    instance._step = {
        "context_key": "test_key",
        "parameters": {"precision_ms": 250}  # Only one param set
    }

    params = instance.get_parameters()

    # Can use get() with fallbacks for missing parameters
    precision = params.get("precision_ms", 1000)  # Uses provided value
    timeout = params.get("timeout", 30)  # Uses fallback

    assert precision == 250  # From parameters
    assert timeout == 30  # From fallback


# ========================================
# Test get_task_objective()
# ========================================


def test_get_task_objective_from_step():
    """Test getting task objective when it exists in step."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"task_current_task": "Global task"}
    instance._step = {
        "context_key": "test_key",
        "task_objective": "Specific step task"
    }

    task = instance.get_task_objective()

    # Should return step's task_objective
    assert task == "Specific step task"


def test_get_task_objective_fallback_to_current_task(monkeypatch):
    """Test fallback to current task when not in step."""
    from unittest.mock import MagicMock

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"task_current_task": "Global task"}
    instance._step = {"context_key": "test_key"}  # No task_objective!

    # Mock StateManager
    mock_sm = MagicMock()
    mock_sm.get_current_task.return_value = "Global task"
    monkeypatch.setattr('osprey.state.StateManager', mock_sm)

    task = instance.get_task_objective()

    # Should fall back to current task from state
    assert task == "Global task"
    mock_sm.get_current_task.assert_called_once()


def test_get_task_objective_with_custom_default():
    """Test using custom default instead of state fallback."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"task_current_task": "Global task"}
    instance._step = {"context_key": "test_key"}  # No task_objective!

    task = instance.get_task_objective(default="unknown task")

    # Should return custom default
    assert task == "unknown task"


def test_get_task_objective_without_state():
    """Test error when called outside execute()."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    # Don't inject state/step

    with pytest.raises(RuntimeError, match="requires self._step and self._state"):
        instance.get_task_objective()


def test_get_task_objective_common_patterns():
    """Test common usage patterns."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"task_current_task": "Find channels"}
    instance._step = {
        "context_key": "test_key",
        "task_objective": "Search for beam current channels"
    }

    # Pattern 1: Use directly
    task = instance.get_task_objective()
    assert task == "Search for beam current channels"

    # Pattern 2: Lowercase for search queries
    search_query = instance.get_task_objective().lower()
    assert search_query == "search for beam current channels"

    # Pattern 3: Use in logging
    log_msg = f"Starting: {instance.get_task_objective()}"
    assert log_msg == "Starting: Search for beam current channels"


# ========================================
# Test get_step_inputs()
# ========================================


def test_get_step_inputs_from_step():
    """Test getting inputs when they exist in step."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {}
    instance._step = {
        "context_key": "test_key",
        "inputs": [
            {"CHANNEL_ADDRESSES": "channels"},
            {"TIME_RANGE": "time_range"}
        ]
    }

    inputs = instance.get_step_inputs()

    # Should return step's inputs list
    assert inputs == [
        {"CHANNEL_ADDRESSES": "channels"},
        {"TIME_RANGE": "time_range"}
    ]


def test_get_step_inputs_empty_list():
    """Test getting inputs when step has no inputs."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {}
    instance._step = {"context_key": "test_key"}  # No inputs!

    inputs = instance.get_step_inputs()

    # Should return empty list as default
    assert inputs == []


def test_get_step_inputs_with_custom_default():
    """Test using custom default instead of empty list."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {}
    instance._step = {"context_key": "test_key"}  # No inputs!

    custom_default = [{"DEFAULT": "value"}]
    inputs = instance.get_step_inputs(default=custom_default)

    # Should return custom default
    assert inputs == custom_default


def test_get_step_inputs_none_value():
    """Test that None value in inputs returns default."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {}
    instance._step = {
        "context_key": "test_key",
        "inputs": None  # Explicitly None
    }

    inputs = instance.get_step_inputs()

    # Should return default (empty list) when inputs is None
    assert inputs == []


def test_get_step_inputs_without_step():
    """Test error when called outside execute()."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    # Don't inject step

    with pytest.raises(RuntimeError, match="requires self._step"):
        instance.get_step_inputs()


def test_get_step_inputs_common_patterns():
    """Test common usage patterns."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {}
    instance._step = {
        "context_key": "test_key",
        "inputs": [
            {"CHANNEL_ADDRESSES": "channels"},
            {"TIME_RANGE": "time"}
        ]
    }

    # Pattern 1: Use directly
    inputs = instance.get_step_inputs()
    assert len(inputs) == 2

    # Pattern 2: Check if any inputs exist
    has_inputs = bool(instance.get_step_inputs())
    assert has_inputs is True

    # Pattern 3: Count inputs for logging
    count = len(instance.get_step_inputs())
    assert count == 2


# ========================================
# Test store_output_context()
# ========================================


def test_store_output_context_simple(mock_state, mock_step, mock_registry):
    """Test single output storage."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = mock_state
    instance._step = mock_step

    test_context = _TestContext(value="test")

    with mock_registry as mock_reg:
        updates = instance.store_output_context(test_context)

    # Verify returned type
    assert isinstance(updates, dict)

    # Verify it contains context_data (this is what StateManager.store_context returns in our mock)
    assert "context_data" in updates


def test_store_output_context_without_context_type(monkeypatch):
    """Test error when context lacks CONTEXT_TYPE."""
    from unittest.mock import MagicMock

    class BadContext:
        """Context without CONTEXT_TYPE."""
        pass

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = {"test": "state"}
    instance._step = {"context_key": "test_key"}

    # Mock to prevent registry loading
    mock_reg = MagicMock()
    monkeypatch.setattr('osprey.registry.get_registry', lambda: mock_reg)

    bad_context = BadContext()

    with pytest.raises(AttributeError, match="CONTEXT_TYPE"):
        instance.store_output_context(bad_context)


def test_store_output_context_without_state():
    """Test error when called outside execute()."""

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    # Don't inject state

    test_context = _TestContext(value="test")

    with pytest.raises(RuntimeError, match="requires self._state"):
        instance.store_output_context(test_context)


def test_store_output_context_missing_context_key(mock_state, monkeypatch):
    """Test error when context_key missing from step."""
    from unittest.mock import MagicMock

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = mock_state
    instance._step = {}  # No context_key!

    # Mock to prevent registry loading
    mock_reg = MagicMock()
    mock_reg.context_types.TEST_DATA = "TEST_DATA"
    monkeypatch.setattr('osprey.registry.get_registry', lambda: mock_reg)

    test_context = _TestContext(value="test")

    with pytest.raises(ValueError, match="No context_key in step"):
        instance.store_output_context(test_context)


# ========================================
# Test store_output_contexts()
# ========================================


def test_store_output_contexts_multiple(mock_state, mock_step, monkeypatch):
    """Test multiple output storage with call verification."""
    from unittest.mock import MagicMock, call

    class TestCap(BaseCapability):
        name = "test"
        description = "test"

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = mock_state
    instance._step = mock_step

    context1 = _TestContext(value="test1")
    context2 = _TestMultipleContext(values=["test2", "test3"])

    # Mock registry with proper return values
    mock_reg = MagicMock()
    mock_reg.context_types.TEST_DATA = "TEST_DATA"
    mock_reg.context_types.MULTIPLE_DATA = "MULTIPLE_DATA"

    # Mock StateManager.store_context to return different updates for each call
    mock_store = MagicMock(side_effect=[
        {"context_data": {"TEST_DATA": "stored1"}},
        {"context_data": {"MULTIPLE_DATA": "stored2"}}
    ])

    monkeypatch.setattr('osprey.registry.get_registry', lambda: mock_reg)
    monkeypatch.setattr('osprey.state.StateManager.store_context', mock_store)

    updates = instance.store_output_contexts(context1, context2)

    # Verify the method was called twice (once for each context)
    assert mock_store.call_count == 2

    # Verify returned dict is merged from both calls
    assert isinstance(updates, dict)
    assert "context_data" in updates

    # Verify both contexts were stored (merged result)
    # The implementation merges dicts, so last one wins or they're combined
    assert updates["context_data"] is not None


def test_store_output_contexts_with_provides_validation(mock_state, mock_step, monkeypatch):
    """Test validation against provides field."""
    from unittest.mock import MagicMock

    class TestCap(BaseCapability):
        name = "test"
        description = "test"
        provides = ["TEST_DATA"]  # Only allow TEST_DATA

        async def execute(self) -> dict:
            return {}

    instance = TestCap()
    instance._state = mock_state
    instance._step = mock_step

    # Mock to prevent registry loading
    mock_reg = MagicMock()
    monkeypatch.setattr('osprey.registry.get_registry', lambda: mock_reg)

    wrong_context = _TestMultipleContext(values=["wrong"])

    with pytest.raises(ValueError, match="don't match provides"):
        instance.store_output_contexts(wrong_context)


# ========================================
# Test requires field validation
# ========================================


def test_requires_validation_invalid_cardinality():
    """Test that invalid cardinality raises error at initialization."""

    class BadCap1(BaseCapability):
        name = "bad"
        description = "bad"
        requires = [("DATA", "soft")]  # Wrong! soft is constraint_mode, not cardinality

        async def execute(self) -> dict:
            return {}

    # Error raised during instantiation (__init__)
    with pytest.raises(ValueError, match="Invalid cardinality 'soft'"):
        BadCap1()


def test_requires_validation_invalid_type():
    """Test that invalid type raises error at initialization."""

    class BadCap2(BaseCapability):
        name = "bad"
        description = "bad"
        requires = [123]  # Wrong type!

        async def execute(self) -> dict:
            return {}

    # Error raised during instantiation (__init__)
    with pytest.raises(ValueError, match="Invalid type"):
        BadCap2()


def test_requires_validation_invalid_tuple_length():
    """Test that invalid tuple length raises error."""

    class BadCap3(BaseCapability):
        name = "bad"
        description = "bad"
        requires = [("DATA", "single", "extra")]  # Too many elements!

        async def execute(self) -> dict:
            return {}

    # Error raised during instantiation (__init__)
    with pytest.raises(ValueError, match="Invalid tuple format"):
        BadCap3()


def test_requires_validation_valid_formats():
    """Test that valid requires formats work."""

    # String format
    class GoodCap1(BaseCapability):
        name = "good1"
        description = "good"
        requires = ["DATA"]

        async def execute(self) -> dict:
            return {}

    instance1 = GoodCap1()
    assert instance1.requires == ["DATA"]

    # Tuple with single
    class GoodCap2(BaseCapability):
        name = "good2"
        description = "good"
        requires = [("DATA", "single")]

        async def execute(self) -> dict:
            return {}

    instance2 = GoodCap2()
    assert instance2.requires == [("DATA", "single")]

    # Tuple with multiple
    class GoodCap3(BaseCapability):
        name = "good3"
        description = "good"
        requires = [("DATA", "multiple")]

        async def execute(self) -> dict:
            return {}

    instance3 = GoodCap3()
    assert instance3.requires == [("DATA", "multiple")]

    # Mixed
    class GoodCap4(BaseCapability):
        name = "good4"
        description = "good"
        requires = ["DATA1", ("DATA2", "single"), ("DATA3", "multiple")]

        async def execute(self) -> dict:
            return {}

    instance4 = GoodCap4()
    assert len(instance4.requires) == 3


# ========================================
# Test backward compatibility
# ========================================


@pytest.mark.asyncio
async def test_static_method_pattern_still_works(monkeypatch):
    """Test that old static method pattern still works."""
    from unittest.mock import MagicMock

    # Mock LangGraph functions
    monkeypatch.setattr('osprey.base.decorators.get_stream_writer', lambda: None)

    # Create proper mock state with execution plan
    mock_state = {
        "planning_execution_steps": [
            {
                "capability": "old_style",
                "context_key": "test_key",
                "task_objective": "Test"
            }
        ],
        "planning_current_step_index": 0,
        "execution_step_results": {}
    }

    # Mock StateManager to return the step
    mock_sm = MagicMock()
    mock_sm.get_current_step.return_value = mock_state["planning_execution_steps"][0]
    mock_sm.get_current_step_index.return_value = 0
    monkeypatch.setattr('osprey.state.StateManager', mock_sm)

    @capability_node
    class OldStyleCap(BaseCapability):
        name = "old_style"
        description = "test"

        @staticmethod
        async def execute(state: dict) -> dict:
            return {"test": "value"}

    result = await OldStyleCap.langgraph_node(mock_state)
    assert "test" in result
    assert result["test"] == "value"
    # Verify step progression
    assert result["planning_current_step_index"] == 1


@pytest.mark.asyncio
async def test_instance_method_pattern_works(monkeypatch):
    """Test that new instance method pattern works."""
    from unittest.mock import MagicMock

    # Mock LangGraph functions
    monkeypatch.setattr('osprey.base.decorators.get_stream_writer', lambda: None)

    # Create proper mock state with execution plan
    mock_state = {
        "planning_execution_steps": [
            {
                "capability": "new_style",
                "context_key": "test_key",
                "task_objective": "Test"
            }
        ],
        "planning_current_step_index": 0,
        "execution_step_results": {}
    }

    # Mock StateManager to return the step
    mock_sm = MagicMock()
    mock_sm.get_current_step.return_value = mock_state["planning_execution_steps"][0]
    mock_sm.get_current_step_index.return_value = 0
    monkeypatch.setattr('osprey.state.StateManager', mock_sm)

    @capability_node
    class NewStyleCap(BaseCapability):
        name = "new_style"
        description = "test"
        requires = []

        async def execute(self) -> dict:
            # Can access self._state and self._step
            assert self._state is not None
            assert self._step is not None
            # Verify step has expected structure
            assert "capability" in self._step
            assert "context_key" in self._step
            return {"test": "value"}

    result = await NewStyleCap.langgraph_node(mock_state)
    assert "test" in result
    assert result["test"] == "value"
    # Verify step progression
    assert result["planning_current_step_index"] == 1


# ========================================
# Fixtures
# ========================================


@pytest.fixture
def mock_state():
    """Create a mock agent state with complete execution plan."""
    return {
        "planning_execution_steps": [
            {
                "capability": "test",
                "context_key": "test_key",
                "task_objective": "Test objective"
            }
        ],
        "planning_current_step_index": 0,
        "context_data": {},
        "execution_step_results": {}
    }


@pytest.fixture
def mock_step():
    """Create a mock execution step."""
    return {
        "capability": "test",
        "context_key": "test_key",
        "task_objective": "Test objective"
    }


@pytest.fixture
def mock_registry(monkeypatch):
    """Mock the registry and context management for testing."""
    from unittest.mock import MagicMock
    from contextlib import contextmanager

    @contextmanager
    def mock_registry_context():
        # Mock registry
        mock_reg = MagicMock()
        mock_reg.context_types.TEST_DATA = "TEST_DATA"
        mock_reg.context_types.MULTIPLE_DATA = "MULTIPLE_DATA"
        mock_reg.context_types.OPTIONAL_DATA = "OPTIONAL_DATA"

        # Mock context manager
        mock_cm = MagicMock()
        mock_cm.extract_from_step.return_value = {
            "TEST_DATA": _TestContext(value="test"),
            "MULTIPLE_DATA": _TestMultipleContext(values=["test1", "test2"])
        }

        # Mock StateManager
        mock_sm = MagicMock()
        mock_sm.store_context.return_value = {"context_data": {}}

        # Mock the imports that happen inside the methods
        def mock_get_registry():
            return mock_reg

        # Patch at the point where they're imported (inside the methods)
        monkeypatch.setattr('osprey.registry.get_registry', mock_get_registry)
        monkeypatch.setattr('osprey.context.context_manager.ContextManager', lambda state: mock_cm)
        monkeypatch.setattr('osprey.state.StateManager', mock_sm)

        try:
            yield mock_reg
        finally:
            pass

    return mock_registry_context()

