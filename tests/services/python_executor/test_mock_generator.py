"""Tests for Mock Code Generator.

This module tests the MockCodeGenerator implementation, verifying that it
correctly implements the CodeGenerator protocol and provides all the expected
testing behaviors.

Test Coverage:
    - Protocol compliance
    - Static code generation
    - Code sequence generation
    - Predefined behaviors (success, errors, EPICS, security)
    - Error-aware generation
    - Call tracking and state management
    - Configuration handling
"""

import pytest

from osprey.services.python_executor.generation import CodeGenerator, MockCodeGenerator
from osprey.services.python_executor.models import PythonExecutionRequest, ExecutionError


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================

class TestMockGeneratorProtocol:
    """Test that MockCodeGenerator implements the CodeGenerator protocol."""

    def test_implements_code_generator_protocol(self):
        """MockCodeGenerator should implement CodeGenerator protocol."""
        generator = MockCodeGenerator()
        assert isinstance(generator, CodeGenerator)

    def test_has_generate_code_method(self):
        """MockCodeGenerator should have generate_code method."""
        generator = MockCodeGenerator()
        assert hasattr(generator, 'generate_code')
        assert callable(generator.generate_code)

    def test_accepts_model_config(self):
        """MockCodeGenerator should accept model_config parameter."""
        config = {"test": "value", "setting": 42}
        generator = MockCodeGenerator(model_config=config)
        assert generator.model_config == config


# =============================================================================
# STATIC CODE GENERATION TESTS
# =============================================================================

class TestStaticCodeGeneration:
    """Test static code generation mode."""

    @pytest.mark.asyncio
    async def test_set_code_returns_static_code(self):
        """set_code() should make generator return that code."""
        generator = MockCodeGenerator()
        test_code = "results = {'value': 42}"
        generator.set_code(test_code)

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code = await generator.generate_code(request, [])
        assert code == test_code

    @pytest.mark.asyncio
    async def test_static_code_persists_across_calls(self):
        """Static code should be returned on all calls."""
        generator = MockCodeGenerator()
        test_code = "results = {'test': 'persistent'}"
        generator.set_code(test_code)

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        # Multiple calls should return same code
        code1 = await generator.generate_code(request, [])
        code2 = await generator.generate_code(request, [])
        code3 = await generator.generate_code(request, ["error"])

        assert code1 == test_code
        assert code2 == test_code
        assert code3 == test_code

    @pytest.mark.asyncio
    async def test_unconfigured_generator_raises_error(self):
        """Generator without code configuration should raise ValueError."""
        generator = MockCodeGenerator()

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        with pytest.raises(ValueError, match="not configured"):
            await generator.generate_code(request, [])


# =============================================================================
# CODE SEQUENCE TESTS
# =============================================================================

class TestCodeSequenceGeneration:
    """Test code sequence generation mode."""

    @pytest.mark.asyncio
    async def test_set_code_sequence_returns_different_codes(self):
        """Code sequence should return different code on each call."""
        generator = MockCodeGenerator()
        sequence = [
            "results = {'attempt': 1}",
            "results = {'attempt': 2}",
            "results = {'attempt': 3}"
        ]
        generator.set_code_sequence(sequence)

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code1 = await generator.generate_code(request, [])
        code2 = await generator.generate_code(request, [])
        code3 = await generator.generate_code(request, [])

        assert code1 == sequence[0]
        assert code2 == sequence[1]
        assert code3 == sequence[2]

    @pytest.mark.asyncio
    async def test_sequence_repeats_last_code(self):
        """After sequence exhausted, should repeat last code."""
        generator = MockCodeGenerator()
        sequence = [
            "results = {'first': 1}",
            "results = {'last': 2}"
        ]
        generator.set_code_sequence(sequence)

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        # Call more times than sequence length
        codes = []
        for _ in range(5):
            code = await generator.generate_code(request, [])
            codes.append(code)

        assert codes[0] == sequence[0]
        assert codes[1] == sequence[1]
        assert codes[2] == sequence[1]  # Repeats last
        assert codes[3] == sequence[1]
        assert codes[4] == sequence[1]

    @pytest.mark.asyncio
    async def test_set_code_clears_sequence(self):
        """set_code() should clear any existing sequence."""
        generator = MockCodeGenerator()
        generator.set_code_sequence(["old1", "old2"])
        generator.set_code("new_static")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code1 = await generator.generate_code(request, [])
        code2 = await generator.generate_code(request, [])

        # Should return static code, not sequence
        assert code1 == "new_static"
        assert code2 == "new_static"

    @pytest.mark.asyncio
    async def test_set_sequence_clears_static(self):
        """set_code_sequence() should clear any static code."""
        generator = MockCodeGenerator()
        generator.set_code("old_static")
        generator.set_code_sequence(["seq1", "seq2"])

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code1 = await generator.generate_code(request, [])
        code2 = await generator.generate_code(request, [])

        # Should return sequence, not static
        assert code1 == "seq1"
        assert code2 == "seq2"


# =============================================================================
# PREDEFINED BEHAVIORS TESTS
# =============================================================================

class TestPredefinedBehaviors:
    """Test predefined behavior patterns."""

    @pytest.mark.asyncio
    async def test_behavior_success(self):
        """Success behavior should generate valid Python code."""
        generator = MockCodeGenerator(behavior="success")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code = await generator.generate_code(request, [])

        # Should be valid Python
        compile(code, '<string>', 'exec')

        # Should have results dictionary
        assert 'results' in code
        assert '{' in code

    @pytest.mark.asyncio
    async def test_behavior_syntax_error(self):
        """Syntax error behavior should generate invalid Python."""
        generator = MockCodeGenerator(behavior="syntax_error")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code = await generator.generate_code(request, [])

        # Should have syntax error
        with pytest.raises(SyntaxError):
            compile(code, '<string>', 'exec')

    @pytest.mark.asyncio
    async def test_behavior_runtime_error(self):
        """Runtime error behavior should generate code that fails execution."""
        generator = MockCodeGenerator(behavior="runtime_error")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code = await generator.generate_code(request, [])

        # Should compile (valid syntax)
        compile(code, '<string>', 'exec')

        # Should have division by zero
        assert '/ 0' in code

    @pytest.mark.asyncio
    async def test_behavior_missing_results(self):
        """Missing results behavior should generate code without results dict."""
        generator = MockCodeGenerator(behavior="missing_results")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code = await generator.generate_code(request, [])

        # Should be valid Python
        compile(code, '<string>', 'exec')

        # Should not have results = {...}
        assert 'results = {' not in code or 'results = {}' not in code

    @pytest.mark.asyncio
    async def test_behavior_channel_write(self):
        """Channel write behavior should generate code with write_channel."""
        generator = MockCodeGenerator(behavior="channel_write")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code = await generator.generate_code(request, [])

        # Should have runtime imports and write operations
        assert 'osprey.runtime' in code
        assert 'write_channel' in code
        assert 'results' in code

    @pytest.mark.asyncio
    async def test_behavior_channel_read(self):
        """Channel read behavior should generate code with read_channel only."""
        generator = MockCodeGenerator(behavior="channel_read")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code = await generator.generate_code(request, [])

        # Should have runtime imports and read operations
        assert 'osprey.runtime' in code
        assert 'read_channel' in code
        # Should NOT have write operations
        assert 'write_channel' not in code

    @pytest.mark.asyncio
    async def test_behavior_security_risk(self):
        """Security risk behavior should generate code with dangerous operations."""
        generator = MockCodeGenerator(behavior="security_risk")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code = await generator.generate_code(request, [])

        # Should have security-sensitive operations
        assert 'subprocess' in code or 'os.system' in code

    def test_unknown_behavior_raises_error(self):
        """Unknown behavior should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown behavior"):
            MockCodeGenerator(behavior="nonexistent_behavior")


# =============================================================================
# ERROR-AWARE GENERATION TESTS
# =============================================================================

class TestErrorAwareGeneration:
    """Test error-aware generation that adapts to feedback."""

    @pytest.mark.asyncio
    async def test_first_attempt_no_errors(self):
        """First attempt with no errors should return initial code."""
        generator = MockCodeGenerator(behavior="error_aware")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        code = await generator.generate_code(request, [])

        # Should be valid Python
        compile(code, '<string>', 'exec')
        assert 'results' in code

    @pytest.mark.asyncio
    async def test_adapts_to_import_error(self):
        """Should add imports when seeing import/name errors."""
        generator = MockCodeGenerator(behavior="error_aware")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        # Second attempt with import error
        error_chain = [
            ExecutionError(
                error_type="execution",
                error_message="NameError: name 'statistics' is not defined",
                attempt_number=1,
                stage="execution"
            )
        ]
        code = await generator.generate_code(request, error_chain)

        # Should have added imports
        assert 'import' in code
        assert 'results' in code

    @pytest.mark.asyncio
    async def test_adapts_to_zero_division_error(self):
        """Should fix division by zero errors."""
        generator = MockCodeGenerator(behavior="error_aware")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        error_chain = [
            ExecutionError(
                error_type="execution",
                error_message="ZeroDivisionError: division by zero",
                attempt_number=1,
                stage="execution"
            )
        ]
        code = await generator.generate_code(request, error_chain)

        # Should have zero check
        assert ('if' in code and '> 0' in code) or 'count > 0' in code

    @pytest.mark.asyncio
    async def test_adapts_to_syntax_error(self):
        """Should fix syntax errors."""
        generator = MockCodeGenerator(behavior="error_aware")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        error_chain = [
            ExecutionError(
                error_type="syntax",
                error_message="SyntaxError: invalid syntax",
                attempt_number=1,
                stage="generation"
            )
        ]
        code = await generator.generate_code(request, error_chain)

        # Should be valid Python now
        compile(code, '<string>', 'exec')

    @pytest.mark.asyncio
    async def test_generic_fix_for_unknown_errors(self):
        """Should provide generic fix for unknown error types."""
        generator = MockCodeGenerator(behavior="error_aware")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        error_chain = [
            ExecutionError(
                error_type="execution",
                error_message="SomeWeirdError: something went wrong",
                attempt_number=1,
                stage="execution"
            )
        ]
        code = await generator.generate_code(request, error_chain)

        # Should return simple working code
        compile(code, '<string>', 'exec')
        assert 'results' in code


# =============================================================================
# CALL TRACKING TESTS
# =============================================================================

class TestCallTracking:
    """Test call tracking and state management."""

    @pytest.mark.asyncio
    async def test_tracks_call_count(self):
        """Should track number of generate_code calls."""
        generator = MockCodeGenerator()
        generator.set_code("results = {}")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        assert generator.call_count == 0

        await generator.generate_code(request, [])
        assert generator.call_count == 1

        await generator.generate_code(request, [])
        assert generator.call_count == 2

        await generator.generate_code(request, [])
        assert generator.call_count == 3

    @pytest.mark.asyncio
    async def test_tracks_last_request(self):
        """Should track the last request received."""
        generator = MockCodeGenerator()
        generator.set_code("results = {}")

        request1 = PythonExecutionRequest(
            user_query="First query",
            task_objective="First task",
            execution_folder_name="test1"
        )

        request2 = PythonExecutionRequest(
            user_query="Second query",
            task_objective="Second task",
            execution_folder_name="test2"
        )

        await generator.generate_code(request1, [])
        assert generator.last_request == request1

        await generator.generate_code(request2, [])
        assert generator.last_request == request2

    @pytest.mark.asyncio
    async def test_tracks_last_error_chain(self):
        """Should track the last error chain received."""
        generator = MockCodeGenerator()
        generator.set_code("results = {}")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        error_chain1 = [
            ExecutionError(
                error_type="execution",
                error_message="Error 1",
                attempt_number=1,
                stage="execution"
            )
        ]
        error_chain2 = [
            ExecutionError(
                error_type="execution",
                error_message="Error 2",
                attempt_number=1,
                stage="execution"
            ),
            ExecutionError(
                error_type="execution",
                error_message="Error 3",
                attempt_number=2,
                stage="execution"
            )
        ]

        await generator.generate_code(request, error_chain1)
        assert generator.last_error_chain == error_chain1

        await generator.generate_code(request, error_chain2)
        assert generator.last_error_chain == error_chain2

    def test_reset_clears_call_tracking(self):
        """reset() should clear call tracking state."""
        generator = MockCodeGenerator()
        generator.set_code("results = {}")
        generator.call_count = 5
        generator.last_request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )
        generator.last_error_chain = [
            ExecutionError(
                error_type="execution",
                error_message="error",
                attempt_number=1,
                stage="execution"
            )
        ]

        generator.reset()

        assert generator.call_count == 0
        assert generator.last_request is None
        assert generator.last_error_chain == []

    def test_reset_preserves_code_config(self):
        """reset() should not clear code configuration."""
        generator = MockCodeGenerator()
        test_code = "results = {'preserved': True}"
        generator.set_code(test_code)

        generator.reset()

        # Code should still be configured
        assert generator.static_code == test_code


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMockGeneratorIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_realistic_retry_scenario(self):
        """Test realistic scenario: fail -> error feedback -> fix -> success."""
        generator = MockCodeGenerator()

        # First attempt fails (syntax error)
        generator.set_code_sequence([
            "def broken(",  # Syntax error
            "results = {'fixed': True}"  # Fixed version
        ])

        request = PythonExecutionRequest(
            user_query="Calculate statistics",
            task_objective="Compute mean",
            execution_folder_name="test"
        )

        # First call: get broken code
        code1 = await generator.generate_code(request, [])
        with pytest.raises(SyntaxError):
            compile(code1, '<string>', 'exec')

        # Second call: get fixed code
        error_chain = [
            ExecutionError(
                error_type="syntax",
                error_message="SyntaxError: invalid syntax",
                attempt_number=1,
                stage="generation"
            )
        ]
        code2 = await generator.generate_code(request, error_chain)
        compile(code2, '<string>', 'exec')  # Should work now

        # Verify tracking
        assert generator.call_count == 2
        assert generator.last_error_chain == error_chain

    @pytest.mark.asyncio
    async def test_error_aware_progressive_improvement(self):
        """Test error-aware mode improves with feedback."""
        generator = MockCodeGenerator(behavior="error_aware")

        request = PythonExecutionRequest(
            user_query="Test",
            task_objective="Test",
            execution_folder_name="test"
        )

        # First: no errors
        code1 = await generator.generate_code(request, [])

        # Second: with import error
        error1 = ExecutionError(
            error_type="execution",
            error_message="NameError: name 'x' is not defined",
            attempt_number=1,
            stage="execution"
        )
        code2 = await generator.generate_code(request, [error1])

        # Third: with division error
        error2 = ExecutionError(
            error_type="execution",
            error_message="ZeroDivisionError",
            attempt_number=2,
            stage="execution"
        )
        code3 = await generator.generate_code(request, [error2])

        # Each should be valid Python
        compile(code1, '<string>', 'exec')
        compile(code2, '<string>', 'exec')
        compile(code3, '<string>', 'exec')

        # Should have adapted
        assert 'import' in code2  # Added imports
        assert 'if' in code3 or 'check' in code3.lower()  # Added check


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_request():
    """Fixture providing a sample execution request."""
    return PythonExecutionRequest(
        user_query="Calculate mean of dataset",
        task_objective="Statistical analysis",
        execution_folder_name="test_analysis",
        expected_results={"mean": "float", "count": "int"}
    )


@pytest.fixture
def mock_generator():
    """Fixture providing a fresh mock generator."""
    return MockCodeGenerator()

