"""Tests for models logging module."""

import contextvars
import json
from datetime import datetime
from unittest.mock import MagicMock

from pydantic import BaseModel

from osprey.models.logging import (
    _api_call_body,
    _api_call_context,
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
# Test the api_call entry body
# =============================================================================


def _body(caller_info, **overrides):
    """An ``api_call`` body for *caller_info* with plain defaults."""
    kwargs = {
        "caller_info": caller_info,
        "message": "in",
        "result": "out",
        "provider": "anthropic",
        "model_id": "claude-3",
        "max_tokens": 1000,
        "temperature": 0.7,
        "enable_thinking": False,
        "budget_tokens": None,
        "output_model": None,
        "include_stack_trace": False,
    }
    kwargs.update(overrides)
    return _api_call_body(**kwargs)


class TestApiCallBody:
    """The body of one ``api_call`` entry."""

    def test_carries_caller_model_configuration_input_and_output(self):
        body = _body(
            {"function": "fn", "module": "mod", "filename": "a/b.py", "line_number": 42},
            message="the prompt",
            result="the answer",
        )

        assert body == {
            "caller": {"function": "fn", "module": "mod", "filename": "a/b.py", "line_number": 42},
            "provider": "anthropic",
            "model_id": "claude-3",
            "max_tokens": 1000,
            "temperature": 0.7,
            "enable_thinking": False,
            "budget_tokens": None,
            "output_model": None,
            "input": "the prompt",
            "output": "the answer",
        }

    def test_carries_class_threading_and_capability_when_present(self):
        body = _body(
            {
                "function": "fn",
                "module": "mod",
                "class": "MyClass",
                "threading": "asyncio.to_thread",
                "capability": "python",
                "source": "context_var",
            }
        )

        assert body["caller"]["class"] == "MyClass"
        assert body["caller"]["threading"] == "asyncio.to_thread"
        assert body["caller"]["capability"] == "python"
        assert "source" not in body["caller"]

    def test_names_the_output_model(self):
        class ResponseModel(BaseModel):
            answer: str

        assert _body({}, output_model=ResponseModel)["output_model"] == "ResponseModel"

    def test_carries_thinking_parameters(self):
        body = _body({}, enable_thinking=True, budget_tokens=5000)

        assert body["enable_thinking"] is True
        assert body["budget_tokens"] == 5000

    def test_the_output_is_the_sanitized_result(self):
        class Result(BaseModel):
            value: str
            count: int

        output = _body({}, result=Result(value="v", count=42))["output"]

        assert '"value": "v"' in output
        assert '"count": 42' in output

    def test_stack_and_skipped_frames_only_with_include_stack_trace(self):
        caller = {"function": "fn", "skipped_frames": ["wrapper (completion.py)"]}

        plain = _body(caller)
        traced = _body(caller, include_stack_trace=True)

        assert "stack" not in plain
        assert "skipped_frames" not in plain["caller"]
        assert traced["caller"]["skipped_frames"] == ["wrapper (completion.py)"]
        assert "test_stack_and_skipped_frames_only_with_include_stack_trace" in traced["stack"]

    def test_is_json_serialisable(self):
        body = _body({"function": "fn"}, result={"when": datetime(2026, 1, 1)})

        assert json.loads(json.dumps(body))["caller"] == {"function": "fn"}


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


# Note: a former test_extracts_class_name_from_method was removed — it was
# vacuous. _get_caller_info() unconditionally skips stack frames 0-2, so the
# method frame holding `self` (always frame 1 for a direct call) can never be
# selected, making the class-extraction branch unreachable from a test without
# refactoring production code. Its only assertion ("function" in caller_info)
# duplicates test_falls_back_to_stack_inspection above.


# =============================================================================
# Test recording a complete API call
# =============================================================================


def _enable(monkeypatch, **api_calls):
    """Point log_api_call at an ``api_calls`` config and a fixed caller."""
    monkeypatch.setattr(
        "osprey.models.logging.get_config_value",
        MagicMock(return_value={"api_calls": api_calls}),
    )
    monkeypatch.setattr(
        "osprey.models.logging._get_caller_info",
        MagicMock(return_value={"function": "test", "module": "test", "line_number": 1}),
    )


def _call(**overrides):
    from osprey.models.logging import log_api_call

    kwargs = {
        "message": "test input message",
        "result": "test output",
        "provider": "anthropic",
        "model_id": "claude-3-sonnet",
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    kwargs.update(overrides)
    log_api_call(**kwargs)


class TestLogApiCall:
    """log_api_call records one ``api_call`` entry per call, when enabled."""

    def test_an_enabled_call_writes_one_entry(self, monkeypatch, recorded_entries):
        _enable(monkeypatch, save_all=True)

        _call()

        entries = recorded_entries("api_call")
        assert len(entries) == 1
        body = entries[0].body
        assert body["input"] == "test input message"
        assert body["output"] == "test output"
        assert body["provider"] == "anthropic"
        assert body["model_id"] == "claude-3-sonnet"
        assert body["max_tokens"] == 1024
        assert body["caller"] == {"function": "test", "module": "test", "line_number": 1}

    def test_each_call_is_its_own_entry(self, monkeypatch, recorded_entries):
        _enable(monkeypatch, save_all=True)

        _call(message="first")
        _call(message="second")

        inputs = sorted(entry.body["input"] for entry in recorded_entries("api_call"))
        assert inputs == ["first", "second"]

    def test_a_disabled_call_writes_none(self, monkeypatch, recorded_entries, spool_zone):
        _enable(monkeypatch, save_all=False)

        _call()

        assert recorded_entries() == []
        assert not any(path.is_file() for path in spool_zone.root.rglob("*"))

    def test_disabled_by_default(self, monkeypatch, recorded_entries):
        _enable(monkeypatch)

        _call()

        assert recorded_entries() == []

    def test_the_record_off_switch_writes_none(self, monkeypatch, recorded_entries, spool_zone):
        from osprey.agent_record.contract import RECORD_OFF_ENV

        _enable(monkeypatch, save_all=True)
        monkeypatch.setenv(RECORD_OFF_ENV, "1")

        _call()

        assert recorded_entries() == []
        assert not any(path.is_file() for path in spool_zone.root.rglob("*"))

    def test_no_file_is_written_outside_the_record(self, monkeypatch, tmp_path):
        """The retired ``api_calls`` .txt files are gone; nothing lands in the cwd."""
        _enable(monkeypatch, save_all=True, latest_only=False)
        monkeypatch.chdir(tmp_path)

        _call()

        assert list(tmp_path.glob("*.txt")) == []

    def test_a_writer_failure_never_raises(self, monkeypatch, recorded_entries):
        _enable(monkeypatch, save_all=True)

        def failing_write(*_args, **_kwargs):
            raise RuntimeError("the writer broke")

        monkeypatch.setattr("osprey.models.logging.record_writer.write", failing_write)

        _call()

        assert recorded_entries() == []

    def test_a_caller_info_failure_never_raises(self, monkeypatch, recorded_entries):
        _enable(monkeypatch, save_all=True)
        monkeypatch.setattr(
            "osprey.models.logging._get_caller_info",
            MagicMock(side_effect=RuntimeError("no stack")),
        )

        _call()

        assert recorded_entries() == []
