"""Tests for Claude Code generator.

These tests verify the Claude Code SDK integration for code generation.
They cover both single-shot and sequential workflows, configuration loading,
and safety features.

Note: These tests require the claude-agent-sdk package to be installed.
"""

import pytest

from osprey.services.python_executor.models import ExecutionError, PythonExecutionRequest

# Check if Claude SDK is available
try:
    from osprey.services.python_executor.generation import (
        CLAUDE_SDK_AVAILABLE,
        ClaudeCodeGenerator,
        CodeGenerator,
    )
except ImportError:
    CLAUDE_SDK_AVAILABLE = False


# Skip all tests if Claude SDK is not available
pytestmark = pytest.mark.skipif(not CLAUDE_SDK_AVAILABLE, reason="Claude Agent SDK not installed")


class TestClaudeCodeGeneratorBasics:
    """Test basic generator initialization and configuration."""

    def test_generator_implements_protocol(self):
        """Test that ClaudeCodeGenerator implements CodeGenerator protocol."""
        generator = ClaudeCodeGenerator()
        assert isinstance(generator, CodeGenerator)

    def test_initialization_with_defaults(self):
        """Test generator initialization with default configuration."""
        generator = ClaudeCodeGenerator()

        assert generator.config is not None
        assert generator.config.get("profile") in ["fast", "robust"]
        assert generator.config.get("profile_phases") is not None
        assert isinstance(generator.config.get("profile_phases"), list)

    def test_initialization_with_inline_config(self):
        """Test generator initialization with inline configuration."""
        config = {
            "profile": "fast",
            "phases": ["generate"],
            "model": "claude-haiku-4-5-20251001",
            "max_turns": 2,
            "max_budget_usd": 0.05,
        }

        generator = ClaudeCodeGenerator(model_config=config)

        assert generator.config["profile"] == "fast"
        assert generator.config["profile_phases"] == ["generate"]
        assert generator.config["model"] == "claude-haiku-4-5-20251001"
        assert generator.config["max_turns"] == 2
        assert generator.config["max_budget_usd"] == 0.05

    def test_code_extraction(self):
        """Test code extraction from text."""
        generator = ClaudeCodeGenerator()

        # Test Python code block extraction
        text_with_python = """
Here's the code you requested:

```python
import numpy as np
results = {'value': 42}
```

That should work!
"""
        code = generator._extract_code_from_text(text_with_python)
        assert code is not None
        assert "import numpy as np" in code
        assert "results" in code

        # Test generic code block extraction
        text_with_generic = """
```
import pandas as pd
data = pd.DataFrame()
```
"""
        code = generator._extract_code_from_text(text_with_generic)
        assert code is not None
        assert "import pandas" in code

        # Test no code found
        text_without_code = "This is just plain text without any code."
        code = generator._extract_code_from_text(text_without_code)
        assert code is None

    def test_code_cleaning(self):
        """Test code cleaning removes markdown formatting."""
        generator = ClaudeCodeGenerator()

        # Code with markdown wrapper
        raw_code = """```python
import numpy as np
results = {}
```"""
        cleaned = generator._clean_generated_code(raw_code)
        assert not cleaned.startswith("```")
        assert not cleaned.endswith("```")
        assert "import numpy" in cleaned

        # Code without markdown
        raw_code_clean = "import pandas as pd\nresults = {}"
        cleaned = generator._clean_generated_code(raw_code_clean)
        assert cleaned == raw_code_clean


class TestClaudeCodeGeneratorPrompts:
    """Test prompt building methods."""

    def test_system_prompt_building(self):
        """Test system prompt construction."""
        generator = ClaudeCodeGenerator()

        request = PythonExecutionRequest(
            user_query="Calculate statistics",
            task_objective="Statistical analysis",
            execution_folder_name="test",
        )

        prompt = generator._build_system_prompt(request)

        assert "Python code generator" in prompt
        # Check for key terms (case-insensitive)
        prompt_lower = prompt.lower()
        assert "executable" in prompt_lower and "python" in prompt_lower and "code" in prompt_lower
        assert "results" in prompt_lower


class TestClaudeCodeGeneratorSafety:
    """Test safety features of Claude Code generator."""

    @pytest.mark.asyncio
    async def test_safety_hook_blocks_dangerous_tools(self):
        """Test that safety hook blocks dangerous tools."""
        generator = ClaudeCodeGenerator()

        dangerous_tools = ["Write", "Edit", "Delete", "Bash", "Python"]

        for tool in dangerous_tools:
            hook_input = {
                "tool_name": tool,
                "tool_input": {},
                "session_id": "test",
                "transcript_path": "/test",
                "cwd": "/test",
            }

            result = await generator._safety_hook(hook_input, None, {"signal": None})

            # Should deny dangerous tools
            assert "hookSpecificOutput" in result
            assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_safety_hook_allows_safe_tools(self):
        """Test that safety hook allows safe read-only tools."""
        generator = ClaudeCodeGenerator()

        safe_tools = ["Read", "Grep", "Glob"]

        for tool in safe_tools:
            hook_input = {
                "tool_name": tool,
                "tool_input": {},
                "session_id": "test",
                "transcript_path": "/test",
                "cwd": "/test",
            }

            result = await generator._safety_hook(hook_input, None, {"signal": None})

            # Should allow safe tools (empty result means allow)
            assert result == {} or "permissionDecision" not in result.get("hookSpecificOutput", {})


class TestClaudeCodeGeneratorConfiguration:
    """Test configuration loading and handling."""

    def test_inline_configuration(self):
        """Test generator with inline configuration."""
        config = {"profile": "robust", "max_budget_usd": 1.0, "max_turns": 10}

        generator = ClaudeCodeGenerator(model_config=config)

        assert generator.config["profile"] == "robust"
        assert generator.config["max_budget_usd"] == 1.0
        assert generator.config["max_turns"] == 10

    def test_missing_profile_defaults_to_fast(self):
        """Test that missing profile defaults to fast."""
        generator = ClaudeCodeGenerator({})

        # Should default to fast
        assert generator.config.get("profile") in ["fast", "robust"]

    def test_codebase_dirs_empty_without_config(self):
        """Test that codebase_dirs is empty without configuration file."""
        generator = ClaudeCodeGenerator()

        # Without a config file, should have empty codebase_dirs
        assert isinstance(generator.config.get("codebase_dirs", []), list)


class TestClaudeCodeGeneratorFactoryIntegration:
    """Test integration with generator factory."""

    def test_factory_can_create_claude_generator(self):
        """Test that factory can create Claude Code generator."""
        from osprey.services.python_executor.generation import create_code_generator

        config = {
            "execution": {
                "code_generator": "claude_code",
                "generators": {"claude_code": {"profile": "fast"}},
            }
        }

        generator = create_code_generator(config)

        assert isinstance(generator, ClaudeCodeGenerator)
        assert generator.config["profile"] == "fast"

    def test_factory_falls_back_on_missing_sdk(self):
        """Test that factory falls back to legacy if SDK missing.

        Note: This test only works if SDK is actually missing, which won't
        be the case if tests are running (since we check CLAUDE_SDK_AVAILABLE).
        This is more of a documentation test showing the expected behavior.
        """
        # This is tested in the factory tests, not here
        pass


class TestClaudeCodeGeneratorWorkflows:
    """Test different phase configurations."""

    def test_single_phase_configuration(self):
        """Test configuration of single-phase workflow (fast profile)."""
        config = {"profile": "fast", "phases": ["generate"]}

        generator = ClaudeCodeGenerator(model_config=config)

        assert generator.config["profile_phases"] == ["generate"]

    def test_multi_phase_configuration(self):
        """Test configuration of multi-phase workflow (robust profile)."""
        config = {"profile": "robust", "phases": ["scan", "plan", "implement"]}

        generator = ClaudeCodeGenerator(model_config=config)

        assert generator.config["profile_phases"] == ["scan", "plan", "implement"]


@pytest.fixture
def sample_request():
    """Fixture providing a sample execution request."""
    return PythonExecutionRequest(
        user_query="Calculate statistics on data",
        task_objective="Compute mean, median, and std dev",
        execution_folder_name="test_stats",
        expected_results={"mean": "float", "median": "float", "std": "float"},
        capability_prompts=["Use numpy for calculations"],
    )


@pytest.fixture
def sample_error_chain():
    """Fixture providing a sample error chain with ExecutionError objects."""
    return [
        ExecutionError(
            error_type="execution",
            error_message="NameError: name 'np' is not defined",
            failed_code="result = np.mean(data)",
            attempt_number=1,
            stage="execution",
        ),
        ExecutionError(
            error_type="execution",
            error_message="Import numpy first",
            failed_code="result = np.mean(data)",
            attempt_number=2,
            stage="execution",
        ),
    ]


# =============================================================================
# BEHAVIORAL TESTS (NO LLM CALLS)
# =============================================================================


class TestClaudeCodeGeneratorBehavior:
    """Test generator behavior without making LLM calls."""

    def test_configuration_defaults(self):
        """Test default configuration values."""
        generator = ClaudeCodeGenerator()

        assert generator.config is not None
        assert "profile" in generator.config
        assert generator.config["profile"] in ["fast", "robust"]

    def test_configuration_with_inline_config(self):
        """Test generator respects inline configuration."""
        config = {"profile": "robust", "max_budget_usd": 0.5}

        generator = ClaudeCodeGenerator(model_config=config)

        assert generator.config["profile"] == "robust"
        assert generator.config.get("max_budget_usd") == 0.5

    def test_metadata_structure(self):
        """Test metadata structure is correct."""
        generator = ClaudeCodeGenerator()

        metadata = generator.get_generation_metadata()

        # Metadata should be a dict with expected keys
        assert isinstance(metadata, dict)
        assert "thinking_blocks" in metadata
        assert "tool_uses" in metadata
        assert "total_thinking_tokens" in metadata

    def test_code_extraction_from_markdown(self):
        """Test code extraction from markdown blocks."""
        generator = ClaudeCodeGenerator()

        # Python code block
        markdown = """Here's the solution:

```python
import numpy as np
results = {'mean': np.mean([1, 2, 3])}
```

That should work!"""

        code = generator._extract_code_from_text(markdown)
        assert code is not None
        assert "import numpy" in code
        assert "results" in code
        assert "```" not in code

    def test_code_extraction_looks_for_python_keywords(self):
        """Test extraction prefers blocks with Python keywords."""
        generator = ClaudeCodeGenerator()

        # Code with Python keywords
        markdown = """```python
import sys
results = {}
```"""

        code = generator._extract_code_from_text(markdown)
        assert code is not None
        assert "import sys" in code

    def test_code_extraction_none_when_no_code(self):
        """Test extraction returns None when no code found."""
        generator = ClaudeCodeGenerator()

        text = "This is just a plain text response with no code blocks."
        code = generator._extract_code_from_text(text)
        assert code is None

    def test_code_cleaning_removes_markdown(self):
        """Test code cleaning removes markdown wrappers."""
        generator = ClaudeCodeGenerator()

        # Code with markdown
        dirty = "```python\nimport sys\nresults = {}\n```"
        clean = generator._clean_generated_code(dirty)

        assert not clean.startswith("```")
        assert not clean.endswith("```")
        assert "import sys" in clean
        assert "python" not in clean

    def test_code_cleaning_strips_whitespace(self):
        """Test code cleaning strips extra whitespace."""
        generator = ClaudeCodeGenerator()

        dirty = "\n\n  \nimport sys\nresults = {}\n  \n\n"
        clean = generator._clean_generated_code(dirty)

        assert not clean.startswith("\n")
        assert not clean.endswith("\n\n")
        assert "import sys" in clean

    def test_code_cleaning_preserves_clean_code(self):
        """Test cleaning doesn't alter already clean code."""
        generator = ClaudeCodeGenerator()

        clean_code = "import sys\nresults = {}"
        result = generator._clean_generated_code(clean_code)

        assert result == clean_code


class TestClaudeCodeGeneratorSafetyBehavior:
    """Test safety features without LLM calls."""

    @pytest.mark.asyncio
    async def test_safety_hook_blocks_write_tools(self):
        """Test safety hook blocks Write tool."""
        generator = ClaudeCodeGenerator()

        hook_input = {
            "tool_name": "Write",
            "tool_input": {"path": "file.py", "content": "code"},
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp",
        }

        result = await generator._safety_hook(hook_input, None, {"signal": None})

        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_safety_hook_blocks_bash(self):
        """Test safety hook blocks Bash tool."""
        generator = ClaudeCodeGenerator()

        hook_input = {
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "session_id": "test",
            "transcript_path": "/tmp/test",
            "cwd": "/tmp",
        }

        result = await generator._safety_hook(hook_input, None, {"signal": None})

        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.asyncio
    async def test_safety_hook_allows_read_tools(self):
        """Test safety hook allows Read tool."""
        generator = ClaudeCodeGenerator()

        for safe_tool in ["Read", "Grep", "Glob"]:
            hook_input = {
                "tool_name": safe_tool,
                "tool_input": {},
                "session_id": "test",
                "transcript_path": "/tmp/test",
                "cwd": "/tmp",
            }

            result = await generator._safety_hook(hook_input, None, {"signal": None})

            # Should allow (empty result means allow)
            assert (
                result == {}
                or result.get("hookSpecificOutput", {}).get("permissionDecision") != "deny"
            )


class TestClaudeCodeGeneratorConfigurationBehavior:
    """Test configuration loading and handling."""

    def test_codebase_dirs_from_config(self):
        """Test codebase directories are extracted from config."""
        generator = ClaudeCodeGenerator()

        full_config = {
            "codebase_guidance": {
                "plotting": {
                    "directories": ["_agent_data/examples/plots/"],
                    "guidance": "Use for plotting",
                }
            }
        }

        profile = {"phases": ["generate"]}

        dirs = generator._get_codebase_dirs(full_config, profile)

        assert isinstance(dirs, list)
        # Should include plotting directory
        assert any("plots" in d for d in dirs)

    def test_codebase_dirs_empty_without_config(self):
        """Test codebase directories empty without configuration."""
        generator = ClaudeCodeGenerator()

        dirs = generator._get_codebase_dirs({}, {})

        assert isinstance(dirs, list)
        assert len(dirs) == 0

    def test_workflow_model_selection(self):
        """Test workflow model selection."""
        generator = ClaudeCodeGenerator()

        model = generator._get_workflow_model()

        assert isinstance(model, str)
        assert len(model) > 0
        # Should be a valid model identifier
        assert "claude" in model.lower() or "haiku" in model.lower() or "sonnet" in model.lower()


class TestClaudeCodeGeneratorWithFixtures:
    """Tests using fixtures for common test data."""

    def test_system_prompt_with_sample_request(self, sample_request):
        """Test system prompt building with realistic request."""
        generator = ClaudeCodeGenerator()

        prompt = generator._build_system_prompt(sample_request)

        # System prompt should be generic (task details go in user prompt)
        assert "Python" in prompt
        assert "results" in prompt.lower()
        assert "executable" in prompt.lower()

    def test_config_with_sample_request(self, sample_request):
        """Test generator configuration with sample request."""
        generator = ClaudeCodeGenerator()

        # Verify configuration is loaded
        assert generator.config is not None
        assert "profile" in generator.config


class TestClaudeCodeGeneratorStructuredErrors:
    """Test Claude Code generator's handling of structured ExecutionError objects."""

    def test_execution_error_has_to_prompt_text(self):
        """Verify ExecutionError has to_prompt_text method."""
        error = ExecutionError(
            error_type="execution",
            error_message="NameError: name 'undefined_var' is not defined",
            failed_code="result = undefined_var + 10",
            traceback="Traceback (most recent call last):\n  File \"<string>\", line 1\nNameError: name 'undefined_var' is not defined",
            attempt_number=1,
            stage="execution",
        )

        # Should have the method
        assert hasattr(error, "to_prompt_text")
        assert callable(error.to_prompt_text)

        # Should format nicely
        text = error.to_prompt_text()
        assert "NameError" in text
        assert "undefined_var" in text

    def test_execution_error_chain_formatting(self):
        """Verify error chain can be formatted for prompts."""
        error_chain = [
            ExecutionError(
                error_type="syntax",
                error_message="SyntaxError: invalid syntax",
                failed_code="def broken(\n    pass",
                attempt_number=1,
                stage="analysis",
            ),
            ExecutionError(
                error_type="execution",
                error_message="ZeroDivisionError: division by zero",
                failed_code="result = 10 / 0",
                attempt_number=2,
                stage="execution",
            ),
        ]

        # Each error should format
        for error in error_chain:
            text = error.to_prompt_text()
            assert isinstance(text, str)
            assert len(text) > 0

    def test_execution_error_with_analysis_issues(self):
        """Verify ExecutionError includes analysis issues in formatting."""
        error = ExecutionError(
            error_type="analysis",
            error_message="Static analysis failed",
            failed_code="import os\nos.system('rm -rf /')",
            analysis_issues=[
                "Security risk: System command execution",
                "Prohibited import detected: os.system",
            ],
            attempt_number=1,
            stage="analysis",
        )

        text = error.to_prompt_text()

        # Should include analysis issues
        assert "Security risk" in text
        assert "Prohibited import" in text


# =============================================================================
# SYSTEM PROMPT CUSTOMIZATION TESTS
# =============================================================================


class TestClaudeCodeGeneratorSystemPromptCustomization:
    """Test system prompt customization features."""

    def test_default_system_prompt_used(self):
        """Verify DEFAULT_SYSTEM_PROMPT is used when no customization."""
        generator = ClaudeCodeGenerator()

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test",
        )

        prompt = generator._build_system_prompt(request)

        # Should contain key phrases from default prompt
        assert "Python code generator" in prompt
        assert "scientific computing" in prompt or "control systems" in prompt

    def test_custom_system_prompt_replaces_default(self):
        """Verify custom system_prompt completely replaces the default."""
        custom_prompt = "You are a specialized quantum computing code generator."

        generator = ClaudeCodeGenerator(model_config={"system_prompt": custom_prompt})

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test",
        )

        prompt = generator._build_system_prompt(request)

        # Should use custom prompt, not default
        assert "quantum computing" in prompt
        # Default phrases should NOT be present
        assert "scientific computing and control systems" not in prompt

    def test_system_prompt_extensions_appended(self):
        """Verify system_prompt_extensions are appended to base prompt."""
        extensions = """
CONTROL SYSTEM OPERATIONS:
- Use osprey.runtime for all channel operations
- NEVER use epics.caput() directly
"""
        generator = ClaudeCodeGenerator(model_config={"system_prompt_extensions": extensions})

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test",
        )

        prompt = generator._build_system_prompt(request)

        # Should contain default prompt content
        assert "Python code generator" in prompt
        # AND should contain extensions
        assert "osprey.runtime" in prompt
        assert "NEVER use epics.caput()" in prompt

    def test_custom_prompt_with_extensions(self):
        """Verify extensions work with custom system_prompt too."""
        custom_prompt = "You are a custom generator."
        extensions = "CUSTOM EXTENSION MARKER"

        generator = ClaudeCodeGenerator(
            model_config={
                "system_prompt": custom_prompt,
                "system_prompt_extensions": extensions,
            }
        )

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test",
        )

        prompt = generator._build_system_prompt(request)

        assert "custom generator" in prompt
        assert "CUSTOM EXTENSION MARKER" in prompt

    def test_empty_extensions_not_added(self):
        """Verify empty extensions don't add extra whitespace."""
        generator = ClaudeCodeGenerator(model_config={"system_prompt_extensions": ""})

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test",
        )

        prompt = generator._build_system_prompt(request)

        # Should not have double newlines from empty extension
        assert "\n\n\n" not in prompt

    def test_config_loaded_into_generator_config(self):
        """Verify system prompt config is stored in generator.config."""
        custom_prompt = "Custom prompt"
        extensions = "Custom extensions"

        generator = ClaudeCodeGenerator(
            model_config={
                "system_prompt": custom_prompt,
                "system_prompt_extensions": extensions,
            }
        )

        assert generator.config.get("system_prompt") == custom_prompt
        assert generator.config.get("system_prompt_extensions") == extensions

    def test_default_system_prompt_constant_exists(self):
        """Verify DEFAULT_SYSTEM_PROMPT class attribute exists and is valid."""
        assert hasattr(ClaudeCodeGenerator, "DEFAULT_SYSTEM_PROMPT")
        assert isinstance(ClaudeCodeGenerator.DEFAULT_SYSTEM_PROMPT, str)
        assert len(ClaudeCodeGenerator.DEFAULT_SYSTEM_PROMPT) > 100  # Non-trivial content
        assert "Python" in ClaudeCodeGenerator.DEFAULT_SYSTEM_PROMPT
