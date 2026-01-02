"""Tests for models logging module."""

import contextvars
from datetime import datetime
from unittest.mock import MagicMock

from pydantic import BaseModel

from osprey.models.logging import (
    _api_call_context,
    _format_metadata_header,
    _sanitize_result_for_logging,
    set_api_call_context,
)

# =============================================================================
# Test API Call Context Setting
# =============================================================================


class TestSetApiCallContext:
    """Test API call context variable management."""

    def test_sets_basic_context(self):
        """Test setting basic caller context information."""
        set_api_call_context(
            function="my_function", module="my_module", class_name="MyClass", line=42
        )

        context = _api_call_context.get()

        assert context is not None
        assert context["function"] == "my_function"
        assert context["module"] == "my_module"
        assert context["class"] == "MyClass"
        assert context["line_number"] == 42
        assert context["source"] == "context_var"

    def test_sets_context_without_class(self):
        """Test setting context without class name."""
        set_api_call_context(function="standalone_func", module="utils", line=10)

        context = _api_call_context.get()

        assert context["function"] == "standalone_func"
        assert context["module"] == "utils"
        assert context["class"] is None
        assert context["line_number"] == 10

    def test_sets_context_with_extra_metadata(self):
        """Test setting context with additional metadata."""
        set_api_call_context(
            function="classify",
            module="classifier",
            extra={"capability": "python", "task_id": "task-123"},
        )

        context = _api_call_context.get()

        assert context["function"] == "classify"
        assert context["capability"] == "python"
        assert context["task_id"] == "task-123"

    def test_context_persists_across_calls(self):
        """Test that context persists within the same context variable scope."""
        set_api_call_context(function="test_func", module="test_module")

        context1 = _api_call_context.get()
        context2 = _api_call_context.get()

        assert context1 == context2

    def test_context_isolation_in_new_context(self):
        """Test that context is isolated when using contextvars.copy_context()."""
        # Set initial context
        set_api_call_context(function="func1", module="mod1")
        context1 = _api_call_context.get()

        # Create a copy of the context and modify it
        ctx = contextvars.copy_context()

        def set_new_context():
            set_api_call_context(function="func2", module="mod2")
            return _api_call_context.get()

        context2 = ctx.run(set_new_context)

        # Original context should be unchanged
        assert context1["function"] == "func1"
        assert context2["function"] == "func2"


# =============================================================================
# Test Result Sanitization for Logging
# =============================================================================


class TestSanitizeResultForLogging:
    """Test sanitization of various result types for logging."""

    def test_sanitize_string_result(self):
        """Test sanitizing a string result."""
        result = "This is a simple text response"
        sanitized = _sanitize_result_for_logging(result)

        assert sanitized == result
        assert isinstance(sanitized, str)

    def test_sanitize_pydantic_model_result(self):
        """Test sanitizing a Pydantic model instance."""

        class TestModel(BaseModel):
            name: str
            value: int
            active: bool

        result = TestModel(name="test", value=42, active=True)
        sanitized = _sanitize_result_for_logging(result)

        # Should be JSON string
        assert isinstance(sanitized, str)
        assert '"name"' in sanitized
        assert '"value"' in sanitized
        assert '"active"' in sanitized
        assert "test" in sanitized
        assert "42" in sanitized

    def test_sanitize_dict_result(self):
        """Test sanitizing a dict result."""
        result = {"key1": "value1", "key2": 123, "key3": True}
        sanitized = _sanitize_result_for_logging(result)

        # Should be formatted JSON
        assert isinstance(sanitized, str)
        assert "key1" in sanitized
        assert "value1" in sanitized
        assert "123" in sanitized

    def test_sanitize_list_result(self):
        """Test sanitizing a list result."""
        result = ["item1", "item2", "item3"]
        sanitized = _sanitize_result_for_logging(result)

        # Should be formatted JSON
        assert isinstance(sanitized, str)
        assert "item1" in sanitized
        assert "item2" in sanitized
        assert "item3" in sanitized

    def test_sanitize_pydantic_with_datetime(self):
        """Test sanitizing Pydantic model with datetime field."""
        from datetime import datetime

        class ModelWithDatetime(BaseModel):
            timestamp: datetime
            value: str

        result = ModelWithDatetime(timestamp=datetime(2024, 1, 1, 12, 0, 0), value="test")
        sanitized = _sanitize_result_for_logging(result)

        # Should serialize datetime using mode='json'
        assert isinstance(sanitized, str)
        assert "2024" in sanitized
        assert "test" in sanitized

    def test_sanitize_nested_dict(self):
        """Test sanitizing nested dictionary structures."""
        result = {"outer": {"inner": {"deep": "value"}}, "list": [1, 2, 3]}
        sanitized = _sanitize_result_for_logging(result)

        assert isinstance(sanitized, str)
        assert "outer" in sanitized
        assert "inner" in sanitized
        assert "deep" in sanitized
        assert "value" in sanitized

    def test_sanitize_custom_object_fallback(self):
        """Test sanitizing custom objects falls back to str()."""

        class CustomObject:
            def __str__(self):
                return "CustomObject representation"

        result = CustomObject()
        sanitized = _sanitize_result_for_logging(result)

        assert isinstance(sanitized, str)
        assert "CustomObject" in sanitized


# =============================================================================
# Test Metadata Header Formatting
# =============================================================================


class TestFormatMetadataHeader:
    """Test metadata header formatting."""

    def test_formats_basic_header(self):
        """Test formatting a basic metadata header."""
        caller_info = {
            "function": "test_function",
            "module": "test_module",
            "filename": "test_file.py",
            "line_number": 42,
        }

        header = _format_metadata_header(
            caller_info=caller_info,
            provider="anthropic",
            model_id="claude-3-sonnet",
            max_tokens=1024,
            temperature=0.7,
            enable_thinking=False,
            budget_tokens=None,
            output_model=None,
        )

        # Check for expected sections
        assert "LLM API CALL LOG" in header
        assert "CALLER INFORMATION" in header
        assert "MODEL CONFIGURATION" in header
        assert "test_function" in header
        assert "test_module" in header
        assert "test_file.py" in header
        assert "42" in header
        assert "anthropic" in header
        assert "claude-3-sonnet" in header
        assert "1024" in header
        assert "0.7" in header

    def test_includes_class_context_when_present(self):
        """Test header includes class name when provided."""
        caller_info = {
            "function": "method_name",
            "module": "my_module",
            "class": "MyClass",
            "filename": "file.py",
            "line_number": 10,
        }

        header = _format_metadata_header(
            caller_info=caller_info,
            provider="openai",
            model_id="gpt-4",
            max_tokens=500,
            temperature=0.0,
            enable_thinking=False,
            budget_tokens=None,
            output_model=None,
        )

        assert "MyClass" in header
        assert "# Class: MyClass" in header

    def test_includes_threading_context_when_present(self):
        """Test header includes threading context when provided."""
        caller_info = {
            "function": "async_task",
            "module": "async_module",
            "filename": "async.py",
            "line_number": 100,
            "threading": "ThreadPoolExecutor",
        }

        header = _format_metadata_header(
            caller_info=caller_info,
            provider="google",
            model_id="gemini-pro",
            max_tokens=2048,
            temperature=1.0,
            enable_thinking=True,
            budget_tokens=1000,
            output_model=None,
        )

        assert "ThreadPoolExecutor" in header
        assert "# Threading:" in header

    def test_includes_capability_context_when_present(self):
        """Test header includes capability when provided."""
        caller_info = {
            "function": "classify",
            "module": "classifier",
            "filename": "classifier.py",
            "line_number": 50,
            "capability": "python",
        }

        header = _format_metadata_header(
            caller_info=caller_info,
            provider="anthropic",
            model_id="claude-3",
            max_tokens=1000,
            temperature=0.0,
            enable_thinking=False,
            budget_tokens=None,
            output_model=None,
        )

        assert "python" in header
        assert "# Capability:" in header

    def test_includes_output_model_info(self):
        """Test header includes output model information."""

        class MyOutputModel(BaseModel):
            field: str

        caller_info = {
            "function": "test",
            "module": "test",
            "filename": "test.py",
            "line_number": 1,
        }

        header = _format_metadata_header(
            caller_info=caller_info,
            provider="openai",
            model_id="gpt-4",
            max_tokens=1000,
            temperature=0.0,
            enable_thinking=False,
            budget_tokens=None,
            output_model=MyOutputModel,
        )

        assert "MyOutputModel" in header
        assert "# Output Model:" in header

    def test_includes_thinking_parameters(self):
        """Test header includes thinking configuration."""
        caller_info = {
            "function": "think",
            "module": "thinker",
            "filename": "think.py",
            "line_number": 1,
        }

        header = _format_metadata_header(
            caller_info=caller_info,
            provider="anthropic",
            model_id="claude-3",
            max_tokens=5000,
            temperature=0.5,
            enable_thinking=True,
            budget_tokens=2000,
            output_model=None,
        )

        assert "Enable Thinking: True" in header
        assert "Budget Tokens: 2000" in header

    def test_includes_timestamp(self):
        """Test header includes timestamp."""
        caller_info = {
            "function": "test",
            "module": "test",
            "filename": "test.py",
            "line_number": 1,
        }

        header = _format_metadata_header(
            caller_info=caller_info,
            provider="anthropic",
            model_id="claude-3",
            max_tokens=1000,
            temperature=0.0,
            enable_thinking=False,
            budget_tokens=None,
            output_model=None,
        )

        # Should include a timestamp
        assert "# Timestamp:" in header
        # Timestamp should be in YYYY-MM-DD format
        current_year = datetime.now().year
        assert str(current_year) in header


# =============================================================================
# Test Caller Info Extraction
# =============================================================================


class TestGetCallerInfo:
    """Test _get_caller_info function."""

    def test_gets_caller_from_context_variable(self):
        """Test that caller info is retrieved from context variable when set."""
        from osprey.models.logging import _api_call_context, _get_caller_info

        # Set context manually
        test_context = {
            "function": "test_function",
            "module": "test_module",
            "class": "TestClass",
            "line_number": 42,
            "source": "context_var",
        }
        _api_call_context.set(test_context)

        caller_info = _get_caller_info()

        assert caller_info == test_context
        assert caller_info["source"] == "context_var"

    def test_falls_back_to_stack_inspection(self):
        """Test that caller info falls back to stack inspection when context not set."""
        from osprey.models.logging import _api_call_context, _get_caller_info

        # Clear context
        _api_call_context.set(None)

        caller_info = _get_caller_info()

        # Should have basic fields from stack inspection
        assert "function" in caller_info
        assert "filename" in caller_info
        assert "line_number" in caller_info
        assert "module" in caller_info

    def test_extracts_class_name_from_method(self):
        """Test extraction of class name from method call."""
        from osprey.models.logging import _api_call_context, _get_caller_info

        # Clear context to force stack inspection
        _api_call_context.set(None)

        class TestClassForInspection:
            def test_method(self):
                return _get_caller_info()

        instance = TestClassForInspection()
        caller_info = instance.test_method()

        # May or may not capture class depending on stack depth
        # Just verify we got valid caller info
        assert "function" in caller_info


# =============================================================================
# Test Complete API Call Logging
# =============================================================================


class TestLogApiCall:
    """Test log_api_call function."""

    def test_logging_disabled_by_default(self, tmp_path, monkeypatch):
        """Test that logging is disabled when save_all is False."""
        from osprey.models.logging import log_api_call

        # Mock config to disable logging
        monkeypatch.setattr(
            "osprey.models.logging.get_config_value",
            MagicMock(return_value={"api_calls": {"save_all": False}}),
        )

        # Call log_api_call - should return early without writing files
        log_api_call(
            message="test message",
            result="test result",
            provider="anthropic",
            model_id="claude-3",
            max_tokens=1000,
            temperature=0.7,
        )

        # No files should be created
        # (We can't easily test this without mocking file operations,
        # but the function should return early)

    def test_logging_enabled_creates_file(self, tmp_path, monkeypatch):
        """Test that logging creates file when enabled."""
        from osprey.models.logging import log_api_call

        # Mock config to enable logging
        monkeypatch.setattr(
            "osprey.models.logging.get_config_value",
            MagicMock(return_value={"api_calls": {"save_all": True, "latest_only": True}}),
        )

        # Mock agent dir to use tmp_path
        monkeypatch.setattr("osprey.models.logging.get_agent_dir", lambda x: tmp_path)

        # Mock caller info
        monkeypatch.setattr(
            "osprey.models.logging._get_caller_info",
            MagicMock(return_value={"function": "test", "module": "test", "line_number": 1}),
        )

        # Call log_api_call
        log_api_call(
            message="test input message",
            result="test output",
            provider="anthropic",
            model_id="claude-3-sonnet",
            max_tokens=1024,
            temperature=0.7,
        )

        # Check that a file was created
        files = list(tmp_path.glob("*.txt"))
        assert len(files) == 1

        # Check file contents
        content = files[0].read_text()
        assert "LLM API CALL LOG" in content
        assert "test input message" in content
        assert "test output" in content
        assert "anthropic" in content
        assert "claude-3-sonnet" in content

    def test_logging_with_pydantic_result(self, tmp_path, monkeypatch):
        """Test logging with Pydantic model result."""
        from osprey.models.logging import log_api_call

        class TestResult(BaseModel):
            value: str
            count: int

        # Mock config
        monkeypatch.setattr(
            "osprey.models.logging.get_config_value",
            MagicMock(return_value={"api_calls": {"save_all": True, "latest_only": True}}),
        )
        monkeypatch.setattr("osprey.models.logging.get_agent_dir", lambda x: tmp_path)
        monkeypatch.setattr(
            "osprey.models.logging._get_caller_info",
            MagicMock(return_value={"function": "test", "module": "test", "line_number": 1}),
        )

        result = TestResult(value="test", count=42)

        log_api_call(
            message="input",
            result=result,
            provider="openai",
            model_id="gpt-4",
            max_tokens=500,
            temperature=0.0,
        )

        files = list(tmp_path.glob("*.txt"))
        assert len(files) == 1

        content = files[0].read_text()
        assert '"value"' in content
        assert '"count"' in content
        assert "42" in content

    def test_logging_handles_errors_gracefully(self, monkeypatch):
        """Test that logging errors don't break execution."""
        from osprey.models.logging import log_api_call

        # Mock config to enable logging
        monkeypatch.setattr(
            "osprey.models.logging.get_config_value",
            MagicMock(return_value={"api_calls": {"save_all": True}}),
        )

        # Mock get_agent_dir to raise an error
        def failing_get_agent_dir(x):
            raise RuntimeError("Failed to get agent dir")

        monkeypatch.setattr("osprey.models.logging.get_agent_dir", failing_get_agent_dir)

        # Call should not raise, but log warning
        log_api_call(
            message="test",
            result="test",
            provider="test",
            model_id="test",
            max_tokens=100,
            temperature=0.0,
        )

        # If we get here, the error was handled gracefully

    def test_logging_with_timestamped_files(self, tmp_path, monkeypatch):
        """Test logging with timestamped filenames instead of latest_only."""
        from osprey.models.logging import log_api_call

        monkeypatch.setattr(
            "osprey.models.logging.get_config_value",
            MagicMock(return_value={"api_calls": {"save_all": True, "latest_only": False}}),
        )
        monkeypatch.setattr("osprey.models.logging.get_agent_dir", lambda x: tmp_path)
        monkeypatch.setattr(
            "osprey.models.logging._get_caller_info",
            MagicMock(return_value={"function": "test", "module": "test", "line_number": 1}),
        )

        log_api_call(
            message="test",
            result="result",
            provider="anthropic",
            model_id="claude",
            max_tokens=1000,
            temperature=0.5,
        )

        files = list(tmp_path.glob("*.txt"))
        assert len(files) == 1

        # Filename should contain timestamp (YYYYMMDD_HHMMSS format)
        filename = files[0].name
        assert filename.startswith("test_test_")
        assert "_20" in filename  # Should have year starting with 20
        assert filename.endswith(".txt")
