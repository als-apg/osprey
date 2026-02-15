"""Tests for code generator interface and factory.

This module tests the Protocol-based code generator interface, the factory
pattern implementation, and the basic generator extraction.
"""

import pytest

from osprey.services.python_executor import (
    BasicLLMCodeGenerator,
    CodeGenerator,
    PythonExecutionRequest,
    create_code_generator,
)
from osprey.services.python_executor.exceptions import CodeGenerationError
from osprey.services.python_executor.models import ExecutionError

# =============================================================================
# PROTOCOL TESTS
# =============================================================================


def test_basic_implements_protocol():
    """Verify basic generator implements the CodeGenerator protocol."""
    generator = BasicLLMCodeGenerator(model_config={})
    assert isinstance(generator, CodeGenerator)


def test_protocol_requires_generate_code_method():
    """Verify Protocol checks for generate_code method."""

    class ValidGenerator:
        async def generate_code(self, request, error_chain):
            return "code"

        def get_generation_metadata(self):
            return {}

    class InvalidGenerator:
        pass

    valid = ValidGenerator()
    invalid = InvalidGenerator()

    assert isinstance(valid, CodeGenerator)
    assert not isinstance(invalid, CodeGenerator)


# =============================================================================
# FACTORY TESTS
# =============================================================================


def test_factory_creates_basic_by_default():
    """Factory creates basic generator when not configured."""
    config = {}
    generator = create_code_generator(config)
    assert isinstance(generator, BasicLLMCodeGenerator)


def test_factory_creates_basic_explicitly():
    """Factory creates basic generator when explicitly configured."""
    config = {"execution": {"code_generator": "basic"}}
    generator = create_code_generator(config)
    assert isinstance(generator, BasicLLMCodeGenerator)


def test_factory_with_model_config_reference():
    """Factory can use model_config_name to reference models section."""
    # This would normally reference the models section in full config
    # For testing, we just verify the structure is accepted
    config = {
        "execution": {
            "code_generator": "basic",
            "generators": {"basic": {"model_config_name": "python_code_generator"}},
        }
    }
    generator = create_code_generator(config)
    assert isinstance(generator, BasicLLMCodeGenerator)


def test_factory_handles_unknown_generator():
    """Factory raises error for unknown generators."""
    config = {"execution": {"code_generator": "nonexistent"}}

    with pytest.raises(ValueError, match="not available"):
        create_code_generator(config)


def test_factory_passes_inline_config_to_generator():
    """Factory passes inline generator config to the generator."""
    inline_config = {"provider": "cborg", "model_id": "gpt-4", "temperature": 0.5}
    config = {"execution": {"code_generator": "basic", "generators": {"basic": inline_config}}}

    generator = create_code_generator(config)
    assert isinstance(generator, BasicLLMCodeGenerator)
    # Access model_config to trigger lazy loading
    assert generator.model_config == inline_config


# NOTE: Deprecation warning test removed - the factory no longer supports the old
# config structure. The new structure requires execution.code_generator configuration.


# =============================================================================
# REGISTRATION SYSTEM TESTS
# =============================================================================
# NOTE: The old programmatic registration API (register_generator, unregister_generator)
# has been replaced with the declarative registry system using CodeGeneratorRegistration.
# Custom generators are now registered via application registry.py files.
# See: osprey.registry.base.CodeGeneratorRegistration
# Tests for the registry system are in tests/registry/


# =============================================================================
# LEGACY GENERATOR TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_basic_generator_basic_generation(mock_llm_response):
    """Test basic generator produces code."""
    generator = BasicLLMCodeGenerator(model_config={})

    request = PythonExecutionRequest(
        user_query="Calculate 2 + 2",
        task_objective="Simple arithmetic",
        execution_folder_name="test",
    )

    # Mock the LLM response
    with mock_llm_response("result = 2 + 2"):
        code = await generator.generate_code(request, [])

    assert code
    assert isinstance(code, str)
    assert len(code) > 0


@pytest.mark.asyncio
async def test_basic_generator_cleans_markdown(mock_llm_response):
    """Test basic generator removes markdown code blocks."""
    generator = BasicLLMCodeGenerator(model_config={})

    request = PythonExecutionRequest(
        user_query="Test", task_objective="Test", execution_folder_name="test"
    )

    # Mock LLM returning code wrapped in markdown
    markdown_code = "```python\nresult = 42\n```"

    with mock_llm_response(markdown_code):
        code = await generator.generate_code(request, [])

    # Should have removed markdown wrapper
    assert code == "result = 42"
    assert "```" not in code


@pytest.mark.asyncio
async def test_basic_generator_handles_error_chain(mock_llm_response):
    """Test basic generator includes error feedback in prompt."""
    generator = BasicLLMCodeGenerator(model_config={})

    request = PythonExecutionRequest(
        user_query="Test", task_objective="Test", execution_folder_name="test"
    )

    error_chain = [
        ExecutionError(
            error_type="execution",
            error_message="NameError: name 'undefined_var' is not defined",
            attempt_number=1,
            stage="execution",
        ),
        ExecutionError(
            error_type="syntax",
            error_message="SyntaxError: invalid syntax",
            attempt_number=2,
            stage="generation",
        ),
    ]

    with mock_llm_response("result = 42"):
        code = await generator.generate_code(request, error_chain)

    assert code
    assert isinstance(code, str)


@pytest.mark.asyncio
async def test_basic_generator_raises_on_empty_response(mock_llm_response):
    """Test basic generator raises error on empty LLM response."""
    generator = BasicLLMCodeGenerator(model_config={})

    request = PythonExecutionRequest(
        user_query="Test", task_objective="Test", execution_folder_name="test"
    )

    with mock_llm_response(""):
        with pytest.raises(CodeGenerationError, match="empty code"):
            await generator.generate_code(request, [])


@pytest.mark.asyncio
async def test_basic_generator_handles_llm_exception(mock_llm_error):
    """Test basic generator converts LLM errors to CodeGenerationError."""
    generator = BasicLLMCodeGenerator(model_config={})

    request = PythonExecutionRequest(
        user_query="Test", task_objective="Test", execution_folder_name="test"
    )

    with mock_llm_error(Exception("LLM service unavailable")):
        with pytest.raises(CodeGenerationError, match="LLM code generation failed"):
            await generator.generate_code(request, [])


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_end_to_end_factory_and_generation(mock_llm_response):
    """Test complete flow from factory to code generation."""

    config = {"execution": {"code_generator": "basic", "generators": {"basic": {"model": "gpt-4"}}}}

    # Create generator via factory
    generator = create_code_generator(config)
    assert isinstance(generator, BasicLLMCodeGenerator)

    # Use generator to create code
    request = PythonExecutionRequest(
        user_query="Calculate mean", task_objective="Statistics", execution_folder_name="test"
    )

    with mock_llm_response("import numpy as np\nresult = np.mean([1, 2, 3])"):
        code = await generator.generate_code(request, [])

    assert "numpy" in code
    assert "mean" in code


# NOTE: Custom generator configuration is now tested through the registry system.
# See the generation/README.md for examples of registering custom generators
# via CodeGeneratorRegistration in application registry.py files.


# =============================================================================
# FIXTURES
# =============================================================================


class _MockChunk:
    """Mock LangChain AIMessageChunk with .content attribute."""

    def __init__(self, content: str):
        self.content = content


class _MockModel:
    """Mock LangChain model with async astream()."""

    def __init__(self, response_text: str):
        self._response = response_text

    async def astream(self, messages):
        yield _MockChunk(self._response)


class _MockErrorModel:
    """Mock LangChain model that raises on astream()."""

    def __init__(self, error):
        self._error = error

    async def astream(self, messages):
        raise self._error
        yield  # Make it a generator


@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock get_langchain_model to return specified response via astream."""
    from contextlib import contextmanager

    @contextmanager
    def _mock_response(response_text):
        def mock_get_model(**kwargs):
            return _MockModel(response_text)

        monkeypatch.setattr(
            "osprey.services.python_executor.generation.basic_generator.get_langchain_model",
            mock_get_model,
        )
        yield

    return _mock_response


@pytest.fixture
def mock_llm_error(monkeypatch):
    """Mock get_langchain_model to raise specified error on astream."""
    from contextlib import contextmanager

    @contextmanager
    def _mock_error(error):
        def mock_get_model(**kwargs):
            return _MockErrorModel(error)

        monkeypatch.setattr(
            "osprey.services.python_executor.generation.basic_generator.get_langchain_model",
            mock_get_model,
        )
        yield

    return _mock_error


# =============================================================================
# STRUCTURED ERROR CHAIN TESTS
# =============================================================================


def test_execution_error_to_prompt_text():
    """Verify ExecutionError formats nicely for prompts."""
    error = ExecutionError(
        error_type="execution",
        error_message="Division by zero",
        failed_code="result = 10 / 0",
        traceback="ZeroDivisionError: division by zero",
        attempt_number=1,
        stage="execution",
    )

    text = error.to_prompt_text()

    # Verify all components are in the formatted text
    assert "Attempt 1 - EXECUTION FAILED" in text
    assert "result = 10 / 0" in text
    assert "Division by zero" in text
    assert "ZeroDivisionError" in text
    assert "execution" in text.lower()  # Error type is in the text


def test_execution_error_with_analysis_issues():
    """Verify ExecutionError includes analysis issues in output."""
    error = ExecutionError(
        error_type="analysis",
        error_message="Static analysis failed",
        failed_code="def broken(\n    pass",
        analysis_issues=["Syntax error: Missing closing parenthesis", "Invalid indentation"],
        attempt_number=2,
        stage="analysis",
    )

    text = error.to_prompt_text()

    assert "Attempt 2 - ANALYSIS FAILED" in text
    assert "Issues Found:" in text
    assert "Missing closing parenthesis" in text
    assert "Invalid indentation" in text


def test_execution_error_truncates_long_traceback():
    """Verify ExecutionError truncates very long tracebacks."""
    long_traceback = "X" * 2000  # 2000 character traceback

    error = ExecutionError(
        error_type="execution",
        error_message="Error",
        traceback=long_traceback,
        attempt_number=1,
        stage="execution",
    )

    text = error.to_prompt_text()

    # Should be truncated
    assert "truncated" in text.lower()
    assert len(text) < len(long_traceback)


@pytest.mark.asyncio
async def test_basic_generator_receives_structured_errors(monkeypatch):
    """Verify BasicLLMCodeGenerator receives and processes ExecutionError objects."""
    # Track what prompt was built
    received_prompts = []

    class _CapturingModel:
        async def astream(self, messages):
            received_prompts.append(messages[0].content)
            yield _MockChunk("print('fixed code')")

    def mock_get_model(**kwargs):
        return _CapturingModel()

    monkeypatch.setattr(
        "osprey.services.python_executor.generation.basic_generator.get_langchain_model",
        mock_get_model,
    )

    generator = BasicLLMCodeGenerator(model_config={})

    # Create structured errors
    error_chain = [
        ExecutionError(
            error_type="execution",
            error_message="NameError: name 'x' is not defined",
            failed_code="result = x + 10",
            traceback="Traceback...\nNameError: name 'x' is not defined",
            attempt_number=1,
            stage="execution",
        )
    ]

    request = PythonExecutionRequest(
        user_query="Calculate something", task_objective="Do math", execution_folder_name="test"
    )

    await generator.generate_code(request, error_chain)

    # Verify prompt was built with structured error
    assert len(received_prompts) == 1
    prompt = received_prompts[0]

    # Verify structured error formatting is in the prompt
    assert "PREVIOUS ATTEMPT(S) FAILED" in prompt
    assert "result = x + 10" in prompt  # The failed code
    assert "NameError" in prompt  # The error
    assert "Attempt 1 - EXECUTION FAILED" in prompt  # Structured header
    assert "Generate IMPROVED code" in prompt  # Refinement instruction


@pytest.mark.asyncio
async def test_basic_generator_uses_to_prompt_text(monkeypatch):
    """Verify BasicLLMCodeGenerator calls to_prompt_text() method."""

    # Mock the LLM call
    def mock_get_model(**kwargs):
        return _MockModel("print('code')")

    monkeypatch.setattr(
        "osprey.services.python_executor.generation.basic_generator.get_langchain_model",
        mock_get_model,
    )

    # Spy on to_prompt_text calls
    calls = []
    original_to_prompt_text = ExecutionError.to_prompt_text

    def spy_to_prompt_text(self):
        calls.append(self)
        return original_to_prompt_text(self)

    ExecutionError.to_prompt_text = spy_to_prompt_text

    try:
        generator = BasicLLMCodeGenerator(model_config={})

        error_chain = [
            ExecutionError(
                error_type="execution",
                error_message="Test error",
                attempt_number=1,
                stage="execution",
            )
        ]

        request = PythonExecutionRequest(
            user_query="Test", task_objective="Test", execution_folder_name="test"
        )

        await generator.generate_code(request, error_chain)

        # Verify to_prompt_text was called for our error
        assert len(calls) == 1
        assert calls[0] == error_chain[0]

    finally:
        # Restore original method
        ExecutionError.to_prompt_text = original_to_prompt_text


@pytest.mark.asyncio
async def test_protocol_compatible_with_execution_error():
    """Verify custom generators can handle ExecutionError in protocol."""

    class CustomGenerator:
        async def generate_code(
            self, request: PythonExecutionRequest, error_chain: list[ExecutionError]
        ) -> str:
            # Should be able to use error chain
            if error_chain:
                # Access structured error data
                last_error = error_chain[-1]
                assert hasattr(last_error, "to_prompt_text")
                assert hasattr(last_error, "failed_code")
                return f"# Fixed: {last_error.error_message}"
            return "print('hello')"

        def get_generation_metadata(self):
            return {}

    # Verify it implements protocol
    generator = CustomGenerator()
    assert isinstance(generator, CodeGenerator)

    # Verify it works with ExecutionError
    error_chain = [
        ExecutionError(
            error_type="test",
            error_message="Test error",
            failed_code="broken code",
            attempt_number=1,
            stage="test",
        )
    ]

    request = PythonExecutionRequest(
        user_query="Test", task_objective="Test", execution_folder_name="test"
    )

    generated_code = await generator.generate_code(request, error_chain)
    assert "Fixed: Test error" in generated_code


def test_empty_error_chain_works():
    """Verify generators handle empty error chain correctly."""
    generator = BasicLLMCodeGenerator(model_config={})

    request = PythonExecutionRequest(
        user_query="Test", task_objective="Test", execution_folder_name="test"
    )

    # Should not raise with empty error chain
    prompt = generator._build_code_generation_prompt(request, [])

    # Should not contain error sections
    assert "PREVIOUS" not in prompt
    assert "FAILED" not in prompt


# =============================================================================
# BASIC GENERATOR CONFIG LOADING TESTS
# =============================================================================


class TestBasicGeneratorConfigLoading:
    """Test BasicLLMCodeGenerator YAML config file loading."""

    def test_uses_defaults_when_no_config_file(self):
        """Generator uses default prompts when no config file exists."""
        generator = BasicLLMCodeGenerator(model_config={})

        # Should use defaults
        assert generator.prompt_config["system_role"] == generator.DEFAULT_SYSTEM_ROLE
        assert generator.prompt_config["core_requirements"] == generator.DEFAULT_CORE_REQUIREMENTS
        assert generator.prompt_config["fallback_guidance"] == generator.DEFAULT_FALLBACK_GUIDANCE
        assert generator.prompt_config["_from_file"] is False

    def test_loads_config_from_yaml_file(self, tmp_path, monkeypatch):
        """Generator loads prompts from YAML config file."""

        # Create a config file
        config_content = """
system_role: "You are a custom code generator."
core_requirements: |
  CUSTOM REQUIREMENTS:
  1. Always add comments
  2. Use type hints
system_prompt_extensions: |
  DOMAIN SPECIFIC:
  - Use pandas for data
fallback_guidance: |
  CUSTOM GUIDANCE:
  - Be thorough
error_feedback:
  max_errors: 3
  header: "=== CUSTOM ERROR HEADER ==="
  footer: "Fix the issues above."
save_prompts: true
"""
        config_file = tmp_path / "basic_generator_config.yml"
        config_file.write_text(config_content)

        # Change to tmp directory so generator finds the config
        monkeypatch.chdir(tmp_path)

        generator = BasicLLMCodeGenerator(model_config={})

        assert generator.prompt_config["_from_file"] is True
        assert generator.prompt_config["system_role"] == "You are a custom code generator."
        assert "Always add comments" in generator.prompt_config["core_requirements"]
        assert "pandas for data" in generator.prompt_config["system_prompt_extensions"]
        assert "Be thorough" in generator.prompt_config["fallback_guidance"]
        assert generator.prompt_config["error_feedback"]["max_errors"] == 3
        assert generator._save_prompts is True

    def test_custom_config_path_via_model_config(self, tmp_path):
        """Generator can load config from custom path via model_config."""
        config_content = """
system_role: "Custom via path"
"""
        config_file = tmp_path / "custom_config.yml"
        config_file.write_text(config_content)

        generator = BasicLLMCodeGenerator(model_config={"basic_config_path": str(config_file)})

        assert generator.prompt_config["system_role"] == "Custom via path"
        assert generator.prompt_config["_from_file"] is True

    def test_falls_back_to_defaults_on_invalid_yaml(self, tmp_path, monkeypatch):
        """Generator falls back to defaults if YAML is invalid."""
        config_file = tmp_path / "basic_generator_config.yml"
        config_file.write_text("invalid: yaml: content: [")

        monkeypatch.chdir(tmp_path)

        generator = BasicLLMCodeGenerator(model_config={})

        # Should fall back to defaults
        assert generator.prompt_config["_from_file"] is False
        assert generator.prompt_config["system_role"] == generator.DEFAULT_SYSTEM_ROLE

    def test_prompt_config_used_in_prompt_building(self, tmp_path, monkeypatch):
        """Verify loaded config is actually used in prompt building."""
        config_content = """
system_role: "CUSTOM_ROLE_MARKER"
core_requirements: "CUSTOM_REQUIREMENTS_MARKER"
system_prompt_extensions: "CUSTOM_EXTENSIONS_MARKER"
"""
        config_file = tmp_path / "basic_generator_config.yml"
        config_file.write_text(config_content)
        monkeypatch.chdir(tmp_path)

        generator = BasicLLMCodeGenerator(model_config={})

        request = PythonExecutionRequest(
            user_query="Test", task_objective="Test", execution_folder_name="test"
        )

        prompt = generator._build_code_generation_prompt(request, [])

        assert "CUSTOM_ROLE_MARKER" in prompt
        assert "CUSTOM_REQUIREMENTS_MARKER" in prompt
        assert "CUSTOM_EXTENSIONS_MARKER" in prompt

    def test_error_feedback_config_used(self, tmp_path, monkeypatch):
        """Verify error feedback config is used when building prompts with errors."""
        config_content = """
error_feedback:
  max_errors: 1
  header: "CUSTOM_ERROR_HEADER"
  footer: "CUSTOM_ERROR_FOOTER"
"""
        config_file = tmp_path / "basic_generator_config.yml"
        config_file.write_text(config_content)
        monkeypatch.chdir(tmp_path)

        generator = BasicLLMCodeGenerator(model_config={})

        request = PythonExecutionRequest(
            user_query="Test", task_objective="Test", execution_folder_name="test"
        )

        error_chain = [
            ExecutionError(
                error_type="execution",
                error_message="Error 1",
                attempt_number=1,
                stage="execution",
            ),
            ExecutionError(
                error_type="execution",
                error_message="Error 2",
                attempt_number=2,
                stage="execution",
            ),
        ]

        prompt = generator._build_code_generation_prompt(request, error_chain)

        assert "CUSTOM_ERROR_HEADER" in prompt
        assert "CUSTOM_ERROR_FOOTER" in prompt
        # With max_errors=1, only the last error should be included
        assert "Error 2" in prompt

    def test_default_constants_are_valid_strings(self):
        """Verify default constants are properly formatted strings."""
        assert isinstance(BasicLLMCodeGenerator.DEFAULT_SYSTEM_ROLE, str)
        assert isinstance(BasicLLMCodeGenerator.DEFAULT_CORE_REQUIREMENTS, str)
        assert isinstance(BasicLLMCodeGenerator.DEFAULT_FALLBACK_GUIDANCE, str)
        assert isinstance(BasicLLMCodeGenerator.DEFAULT_ERROR_HEADER, str)
        assert isinstance(BasicLLMCodeGenerator.DEFAULT_ERROR_FOOTER, str)

        # Should not have leading/trailing whitespace issues
        assert not BasicLLMCodeGenerator.DEFAULT_CORE_REQUIREMENTS.startswith("\n")
        assert not BasicLLMCodeGenerator.DEFAULT_FALLBACK_GUIDANCE.startswith("\n")
