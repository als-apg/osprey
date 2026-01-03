"""Tests for utility functions and helper classes in python_executor/models.py."""

from pathlib import Path

from osprey.services.python_executor.models import (
    ExecutionError,
    NotebookAttempt,
    NotebookType,
    PythonExecutionContext,
    preserve_once_set,
    validate_result_structure,
)


class TestPreserveOnceSet:
    """Tests for preserve_once_set reducer function."""

    def test_preserves_existing_value(self):
        """Test that existing value is preserved when new value is provided."""
        existing = "existing_value"
        new = "new_value"
        result = preserve_once_set(existing, new)
        assert result == "existing_value"

    def test_uses_new_when_existing_is_none(self):
        """Test that new value is used when existing is None."""
        existing = None
        new = "new_value"
        result = preserve_once_set(existing, new)
        assert result == "new_value"

    def test_preserves_when_new_is_none(self):
        """Test that existing value is preserved even when new is None."""
        existing = "existing_value"
        new = None
        result = preserve_once_set(existing, new)
        assert result == "existing_value"

    def test_both_none_returns_none(self):
        """Test that None is returned when both are None."""
        result = preserve_once_set(None, None)
        assert result is None

    def test_preserves_complex_objects(self):
        """Test preservation works with complex objects."""
        existing_dict = {"key": "value"}
        new_dict = {"different": "data"}
        result = preserve_once_set(existing_dict, new_dict)
        assert result == {"key": "value"}
        assert result is existing_dict  # Same object reference


class TestValidateResultStructure:
    """Tests for validate_result_structure AST validation function."""

    def test_valid_dict_literal_empty(self):
        """Test validation with empty dict literal."""
        code = "results = {}"
        assert validate_result_structure(code) is True

    def test_valid_dict_literal_with_values(self):
        """Test validation with dict literal containing values."""
        code = "results = {'key': 'value', 'number': 42}"
        assert validate_result_structure(code) is True

    def test_valid_dict_constructor_empty(self):
        """Test validation with dict() constructor."""
        code = "results = dict()"
        assert validate_result_structure(code) is True

    def test_valid_dict_constructor_with_kwargs(self):
        """Test validation with dict constructor with keyword arguments."""
        code = "results = dict(key='value', number=42)"
        assert validate_result_structure(code) is True

    def test_valid_dict_comprehension(self):
        """Test validation with dict comprehension."""
        code = "results = {k: v for k, v in items}"
        assert validate_result_structure(code) is True

    def test_valid_multiline_dict(self):
        """Test validation with multiline dict assignment."""
        code = """
results = {
    'key1': 'value1',
    'key2': 'value2',
    'nested': {'a': 1, 'b': 2}
}
"""
        assert validate_result_structure(code) is True

    def test_invalid_no_results_variable(self):
        """Test that code without 'results' variable fails."""
        code = "other_var = {}"
        assert validate_result_structure(code) is False

    def test_invalid_results_not_dict(self):
        """Test that non-dict assignment to results fails static check."""
        code = "results = None"
        assert validate_result_structure(code) is False

    def test_invalid_results_is_list(self):
        """Test that list assignment to results fails."""
        code = "results = []"
        assert validate_result_structure(code) is False

    def test_invalid_function_call_assignment(self):
        """Test that function call assignment can't be statically validated."""
        code = "results = some_function()"
        assert validate_result_structure(code) is False

    def test_invalid_variable_assignment(self):
        """Test that variable assignment can't be statically validated."""
        code = """
data = {}
results = data
"""
        assert validate_result_structure(code) is False

    def test_syntax_error_returns_false(self):
        """Test that syntax errors return False."""
        code = "results = {invalid syntax"
        assert validate_result_structure(code) is False

    def test_multiple_assignments_finds_dict(self):
        """Test that any dict-like assignment to results is found."""
        code = """
results = None  # First assignment (not dict)
results = {}    # Second assignment (dict) - should pass
"""
        assert validate_result_structure(code) is True

    def test_complex_code_with_results_dict(self):
        """Test validation in context of larger code block."""
        code = """
import pandas as pd
import numpy as np

data = pd.read_csv('file.csv')
analysis = data.describe()

results = {
    'mean': float(analysis['value'].mean()),
    'std': float(analysis['value'].std()),
    'count': len(data)
}

print("Analysis complete")
"""
        assert validate_result_structure(code) is True


class TestExecutionError:
    """Tests for ExecutionError dataclass."""

    def test_create_basic_error(self):
        """Test creating a basic execution error."""
        error = ExecutionError(error_type="execution", error_message="Division by zero")
        assert error.error_type == "execution"
        assert error.error_message == "Division by zero"
        assert error.failed_code is None
        assert error.attempt_number == 1
        assert error.stage == ""

    def test_create_complete_error(self):
        """Test creating error with all fields."""
        error = ExecutionError(
            error_type="syntax",
            error_message="Invalid syntax",
            failed_code="print(",
            traceback="Traceback...",
            attempt_number=2,
            stage="generation",
            analysis_issues=["Missing closing parenthesis"],
        )
        assert error.error_type == "syntax"
        assert error.attempt_number == 2
        assert error.stage == "generation"
        assert len(error.analysis_issues) == 1

    def test_to_prompt_text_basic(self):
        """Test basic prompt text formatting."""
        error = ExecutionError(
            error_type="execution",
            error_message="NameError: undefined_var",
            attempt_number=1,
            stage="execution",
        )
        text = error.to_prompt_text()
        assert "Attempt 1" in text
        assert "EXECUTION FAILED" in text
        assert "NameError: undefined_var" in text

    def test_to_prompt_text_with_code(self):
        """Test prompt text includes failed code."""
        error = ExecutionError(
            error_type="execution",
            error_message="Error occurred",
            failed_code="result = undefined_var + 10",
            attempt_number=1,
            stage="execution",
        )
        text = error.to_prompt_text()
        assert "Code that failed:" in text
        assert "```python" in text
        assert "undefined_var + 10" in text

    def test_to_prompt_text_with_analysis_issues(self):
        """Test prompt text includes analysis issues."""
        error = ExecutionError(
            error_type="analysis",
            error_message="Code failed analysis",
            attempt_number=1,
            stage="analysis",
            analysis_issues=["Missing imports", "Unsafe operation"],
        )
        text = error.to_prompt_text()
        assert "Issues Found:" in text
        assert "Missing imports" in text
        assert "Unsafe operation" in text

    def test_to_prompt_text_truncates_long_traceback(self):
        """Test that long tracebacks are truncated."""
        long_traceback = "Line\n" * 250  # Very long traceback (>1000 chars)
        error = ExecutionError(
            error_type="execution",
            error_message="Error",
            traceback=long_traceback,
            attempt_number=1,
            stage="execution",
        )
        text = error.to_prompt_text()
        # Should have truncated if traceback > 1000 chars
        if len(long_traceback) > 1000:
            assert "truncated" in text
            assert len(text) < len(long_traceback)  # Should be significantly shorter

    def test_to_prompt_text_complete(self):
        """Test complete prompt text with all components."""
        error = ExecutionError(
            error_type="execution",
            error_message="RuntimeError: Calculation failed",
            failed_code="result = risky_calculation()",
            traceback="Traceback (most recent call last):\n  File...",
            attempt_number=3,
            stage="execution",
            analysis_issues=["Risky operation detected"],
        )
        text = error.to_prompt_text()

        # Verify all components present
        assert "Attempt 3" in text
        assert "EXECUTION FAILED" in text
        assert "Code that failed:" in text
        assert "risky_calculation()" in text
        assert "**Error Type:** execution" in text  # Uses markdown bold
        assert "RuntimeError: Calculation failed" in text
        assert "Issues Found:" in text
        assert "Risky operation detected" in text
        assert "Traceback:" in text


class TestNotebookType:
    """Tests for NotebookType enum."""

    def test_enum_values(self):
        """Test all enum values are defined."""
        assert NotebookType.CODE_GENERATION_ATTEMPT.value == "code_generation_attempt"
        assert NotebookType.PRE_EXECUTION.value == "pre_execution"
        assert NotebookType.EXECUTION_ATTEMPT.value == "execution_attempt"
        assert NotebookType.FINAL_SUCCESS.value == "final_success"
        assert NotebookType.FINAL_FAILURE.value == "final_failure"

    def test_enum_members(self):
        """Test enum has expected members."""
        types = list(NotebookType)
        assert len(types) == 5
        assert NotebookType.CODE_GENERATION_ATTEMPT in types
        assert NotebookType.FINAL_SUCCESS in types


class TestNotebookAttempt:
    """Tests for NotebookAttempt dataclass."""

    def test_create_notebook_attempt(self):
        """Test creating a notebook attempt."""
        attempt = NotebookAttempt(
            notebook_type=NotebookType.FINAL_SUCCESS,
            attempt_number=1,
            stage="execution",
            notebook_path=Path("/path/to/notebook.ipynb"),
            notebook_link="http://jupyter/notebooks/notebook.ipynb",
        )
        assert attempt.notebook_type == NotebookType.FINAL_SUCCESS
        assert attempt.attempt_number == 1
        assert attempt.stage == "execution"
        assert attempt.notebook_path == Path("/path/to/notebook.ipynb")
        assert attempt.notebook_link == "http://jupyter/notebooks/notebook.ipynb"
        assert attempt.error_context is None

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        attempt = NotebookAttempt(
            notebook_type=NotebookType.CODE_GENERATION_ATTEMPT,
            attempt_number=2,
            stage="generation",
            notebook_path=Path("/tmp/gen.ipynb"),
            notebook_link="http://localhost/gen.ipynb",
            error_context="Generation failed",
            created_at="2025-12-23T10:00:00",
        )
        data = attempt.to_dict()

        assert data["notebook_type"] == "code_generation_attempt"
        assert data["attempt_number"] == 2
        assert data["stage"] == "generation"
        assert data["notebook_path"] == "/tmp/gen.ipynb"
        assert data["notebook_link"] == "http://localhost/gen.ipynb"
        assert data["error_context"] == "Generation failed"
        assert data["created_at"] == "2025-12-23T10:00:00"

    def test_to_dict_path_conversion(self):
        """Test that Path objects are converted to strings."""
        attempt = NotebookAttempt(
            notebook_type=NotebookType.FINAL_SUCCESS,
            attempt_number=1,
            stage="execution",
            notebook_path=Path("/absolute/path/notebook.ipynb"),
            notebook_link="http://link",
        )
        data = attempt.to_dict()
        assert isinstance(data["notebook_path"], str)
        assert data["notebook_path"] == "/absolute/path/notebook.ipynb"


class TestPythonExecutionContext:
    """Tests for PythonExecutionContext dataclass."""

    def test_create_empty_context(self):
        """Test creating an empty execution context."""
        context = PythonExecutionContext()
        assert context.folder_path is None
        assert context.folder_url is None
        assert context.attempts_folder is None
        assert context.context_file_path is None
        assert context.notebook_attempts == []

    def test_is_initialized_false_when_empty(self):
        """Test is_initialized returns False for empty context."""
        context = PythonExecutionContext()
        assert context.is_initialized is False

    def test_is_initialized_true_when_folder_set(self):
        """Test is_initialized returns True when folder_path is set."""
        context = PythonExecutionContext(folder_path=Path("/tmp/execution"))
        assert context.is_initialized is True

    def test_add_notebook_attempt(self):
        """Test adding a notebook attempt to context."""
        context = PythonExecutionContext()
        attempt = NotebookAttempt(
            notebook_type=NotebookType.FINAL_SUCCESS,
            attempt_number=1,
            stage="execution",
            notebook_path=Path("/path/notebook.ipynb"),
            notebook_link="http://link",
        )
        context.add_notebook_attempt(attempt)
        assert len(context.notebook_attempts) == 1
        assert context.notebook_attempts[0] == attempt

    def test_add_multiple_attempts(self):
        """Test adding multiple notebook attempts."""
        context = PythonExecutionContext()

        for i in range(3):
            attempt = NotebookAttempt(
                notebook_type=NotebookType.CODE_GENERATION_ATTEMPT,
                attempt_number=i + 1,
                stage="generation",
                notebook_path=Path(f"/path/notebook_{i}.ipynb"),
                notebook_link=f"http://link/{i}",
            )
            context.add_notebook_attempt(attempt)

        assert len(context.notebook_attempts) == 3
        assert context.notebook_attempts[0].attempt_number == 1
        assert context.notebook_attempts[2].attempt_number == 3

    def test_get_next_attempt_number_empty(self):
        """Test get_next_attempt_number returns 1 for empty context."""
        context = PythonExecutionContext()
        assert context.get_next_attempt_number() == 1

    def test_get_next_attempt_number_increments(self):
        """Test get_next_attempt_number increments correctly."""
        context = PythonExecutionContext()

        # First attempt
        assert context.get_next_attempt_number() == 1

        # Add first attempt
        attempt1 = NotebookAttempt(
            notebook_type=NotebookType.CODE_GENERATION_ATTEMPT,
            attempt_number=1,
            stage="generation",
            notebook_path=Path("/path1"),
            notebook_link="http://link1",
        )
        context.add_notebook_attempt(attempt1)

        # Should now be 2
        assert context.get_next_attempt_number() == 2

        # Add second attempt
        attempt2 = NotebookAttempt(
            notebook_type=NotebookType.EXECUTION_ATTEMPT,
            attempt_number=2,
            stage="execution",
            notebook_path=Path("/path2"),
            notebook_link="http://link2",
        )
        context.add_notebook_attempt(attempt2)

        # Should now be 3
        assert context.get_next_attempt_number() == 3

    def test_full_context_workflow(self):
        """Test complete context lifecycle."""
        # Initialize context
        context = PythonExecutionContext(
            folder_path=Path("/tmp/execution_123"),
            folder_url="http://jupyter/execution_123",
            attempts_folder=Path("/tmp/execution_123/attempts"),
            context_file_path=Path("/tmp/execution_123/context.json"),
        )

        assert context.is_initialized

        # Add generation attempt
        gen_attempt = NotebookAttempt(
            notebook_type=NotebookType.CODE_GENERATION_ATTEMPT,
            attempt_number=context.get_next_attempt_number(),
            stage="generation",
            notebook_path=context.folder_path / "gen_1.ipynb",
            notebook_link=f"{context.folder_url}/gen_1.ipynb",
        )
        context.add_notebook_attempt(gen_attempt)

        # Add execution attempt
        exec_attempt = NotebookAttempt(
            notebook_type=NotebookType.EXECUTION_ATTEMPT,
            attempt_number=context.get_next_attempt_number(),
            stage="execution",
            notebook_path=context.folder_path / "exec_1.ipynb",
            notebook_link=f"{context.folder_url}/exec_1.ipynb",
        )
        context.add_notebook_attempt(exec_attempt)

        # Add final success
        final_attempt = NotebookAttempt(
            notebook_type=NotebookType.FINAL_SUCCESS,
            attempt_number=context.get_next_attempt_number(),
            stage="final",
            notebook_path=context.folder_path / "final.ipynb",
            notebook_link=f"{context.folder_url}/final.ipynb",
        )
        context.add_notebook_attempt(final_attempt)

        # Verify complete workflow
        assert len(context.notebook_attempts) == 3
        assert context.notebook_attempts[0].notebook_type == NotebookType.CODE_GENERATION_ATTEMPT
        assert context.notebook_attempts[1].notebook_type == NotebookType.EXECUTION_ATTEMPT
        assert context.notebook_attempts[2].notebook_type == NotebookType.FINAL_SUCCESS
        assert context.get_next_attempt_number() == 4
