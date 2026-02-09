"""Integration tests for results validation through the pipeline.

Tests the validation of results dictionary through different stages:
1. Mock generator creating code without results
2. Static analysis catching and warning about it
3. Runtime validation detecting missing results (if execution available)
"""

from unittest.mock import Mock, patch

import pytest

from osprey.services.python_executor.generation import MockCodeGenerator
from osprey.services.python_executor.models import (
    PythonExecutionRequest,
    validate_result_structure,
)


class TestResultsValidationWithMockGenerator:
    """Integration tests using MockCodeGenerator to test validation."""

    def test_mock_generator_without_results_static_validation(self):
        """Mock generator can create code without results, static validation catches it."""
        # Configure mock generator to return code WITHOUT results
        code_without_results = """
import numpy as np

# Calculate some values
data = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(data)
std_value = np.std(data)

# Print but don't store in results
print(f"Mean: {mean_value}")
print(f"Std: {std_value}")
"""

        generator = MockCodeGenerator()
        generator.set_code(code_without_results)

        # Create a basic request
        request = PythonExecutionRequest(
            user_query="Calculate statistics",
            task_objective="Compute mean and std",
            execution_folder_name="test",
        )

        # Generate code (synchronous for mock)
        import asyncio

        generated_code = asyncio.run(generator.generate_code(request, []))

        # Verify the generated code
        assert generated_code == code_without_results

        # Static validation should return False
        assert validate_result_structure(generated_code) is False

    def test_mock_generator_with_dict_literal_results(self):
        """Mock generator creates code with dict literal results."""
        code_with_results = """
import numpy as np

data = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(data)

results = {
    "mean": float(mean_value),
    "count": len(data)
}
"""

        generator = MockCodeGenerator()
        generator.set_code(code_with_results)

        request = PythonExecutionRequest(
            user_query="Calculate mean", task_objective="Compute mean", execution_folder_name="test"
        )

        import asyncio

        generated_code = asyncio.run(generator.generate_code(request, []))

        # Static validation should pass
        assert validate_result_structure(generated_code) is True

    def test_mock_generator_with_dict_constructor_results(self):
        """Mock generator creates code with dict() constructor."""
        code_with_dict_constructor = """
import numpy as np

data = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(data)
std_value = np.std(data)

results = dict(
    mean=float(mean_value),
    std=float(std_value),
    count=len(data)
)
"""

        generator = MockCodeGenerator()
        generator.set_code(code_with_dict_constructor)

        request = PythonExecutionRequest(
            user_query="Calculate statistics",
            task_objective="Compute stats",
            execution_folder_name="test",
        )

        import asyncio

        generated_code = asyncio.run(generator.generate_code(request, []))

        # Static validation should pass
        assert validate_result_structure(generated_code) is True

    def test_mock_generator_with_results_from_function(self):
        """Mock generator creates code where results comes from function call."""
        code_with_function_call = """
import json

def get_results():
    return {"value": 42, "status": "success"}

# This can't be validated statically but should work at runtime
results = get_results()
"""

        generator = MockCodeGenerator()
        generator.set_code(code_with_function_call)

        request = PythonExecutionRequest(
            user_query="Get results", task_objective="Return results", execution_folder_name="test"
        )

        import asyncio

        generated_code = asyncio.run(generator.generate_code(request, []))

        # Static validation should return False (can't validate function return)
        # But this is OK - runtime validation will confirm it works
        assert validate_result_structure(generated_code) is False


class TestResultsValidationThroughAnalysis:
    """Test validation through the analysis node (without execution).

    Uses an empty configurable dict instead of get_full_configuration(),
    since StaticCodeAnalyzer gracefully falls back to defaults when
    registry is not initialized.
    """

    @staticmethod
    def _mock_approval_evaluator():
        """Create a mock approval evaluator that doesn't require config.yml."""
        evaluator = Mock()
        decision = Mock()
        decision.needs_approval = False
        decision.reasoning = "Test mode - no approval required"
        evaluator.evaluate = Mock(return_value=decision)
        return evaluator

    @pytest.mark.asyncio
    async def test_analysis_node_warns_on_missing_results(self, caplog):
        """Analysis node should warn (not fail) when results missing."""
        from osprey.services.python_executor.analysis.node import StaticCodeAnalyzer

        code_without_results = """
import numpy as np
data = np.array([1, 2, 3])
print(np.mean(data))
"""

        configurable = {"agent_control_defaults": {"epics_writes_enabled": False}}
        analyzer = StaticCodeAnalyzer(configurable)

        with patch(
            "osprey.approval.approval_manager.get_python_execution_evaluator",
            return_value=self._mock_approval_evaluator(),
        ):
            # This should complete successfully but log a warning
            with caplog.at_level("WARNING"):
                result = await analyzer.analyze_code(code_without_results, context=None)

        # Analysis should pass (no critical errors)
        assert result.passed is True

        # Should have logged a warning about missing results
        assert any(
            "does not appear to assign to 'results'" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_analysis_node_accepts_valid_results(self):
        """Analysis node should accept code with valid results."""
        from osprey.services.python_executor.analysis.node import StaticCodeAnalyzer

        code_with_results = """
import numpy as np
data = np.array([1, 2, 3])
mean_value = np.mean(data)

results = {
    "mean": float(mean_value),
    "count": len(data)
}
"""

        configurable = {"agent_control_defaults": {"epics_writes_enabled": False}}
        analyzer = StaticCodeAnalyzer(configurable)

        with patch(
            "osprey.approval.approval_manager.get_python_execution_evaluator",
            return_value=self._mock_approval_evaluator(),
        ):
            result = await analyzer.analyze_code(code_with_results, context=None)

        # Analysis should pass
        assert result.passed is True


class TestRuntimeValidationSimulation:
    """Test runtime validation without full execution.

    These tests simulate the runtime validation by directly checking
    what the wrapper would detect, without actually running code.
    """

    def test_runtime_validation_would_catch_missing_results(self):
        """Simulate runtime check for missing results."""
        # This simulates what happens in the execution wrapper

        # Code that doesn't create results
        code_namespace = {}
        exec(
            """
import numpy as np
data = np.array([1, 2, 3])
mean_value = np.mean(data)
print(mean_value)
""",
            code_namespace,
        )

        # Runtime check: is 'results' in globals?
        results_captured = "results" in code_namespace
        results_missing = not results_captured

        assert results_missing is True
        assert results_captured is False

    def test_runtime_validation_would_pass_with_results(self):
        """Simulate runtime check with valid results."""
        code_namespace = {}
        exec(
            """
import numpy as np
data = np.array([1, 2, 3])
mean_value = np.mean(data)

results = {
    "mean": float(mean_value),
    "count": len(data)
}
""",
            code_namespace,
        )

        # Runtime check
        results_captured = "results" in code_namespace
        results_missing = not results_captured

        assert results_missing is False
        assert results_captured is True

        # Verify it's actually a dict
        assert isinstance(code_namespace["results"], dict)
        assert "mean" in code_namespace["results"]

    def test_runtime_validation_detects_results_none(self):
        """Simulate runtime check when results is None."""
        code_namespace = {}
        exec(
            """
results = None
""",
            code_namespace,
        )

        # Runtime check
        results_captured = "results" in code_namespace
        results_is_none = code_namespace.get("results") is None

        assert results_captured is True
        assert results_is_none is True

    def test_runtime_validation_detects_function_returned_dict(self):
        """Runtime validation works even when results from function."""
        code_namespace = {}
        exec(
            """
def get_results():
    return {"value": 42}

results = get_results()
""",
            code_namespace,
        )

        # Runtime check passes even though static analysis couldn't validate
        results_captured = "results" in code_namespace

        assert results_captured is True
        assert isinstance(code_namespace["results"], dict)
        assert code_namespace["results"]["value"] == 42


@pytest.mark.skipif(
    True,  # Skip by default - only run when execution environment available
    reason="Requires execution environment (container or local Python)",
)
class TestFullExecutionValidation:
    """Full end-to-end tests with actual execution.

    These tests are skipped by default because they require:
    - Either a container execution environment
    - Or local Python execution with all dependencies

    To enable: Change skipif condition or run with specific marker.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_fails_on_missing_results(self):
        """Full pipeline should fail at execution when results missing."""
        from osprey.services.python_executor import PythonExecutorService

        # Create service
        service = PythonExecutorService()

        # Request with mock generator configured to not create results
        request = PythonExecutionRequest(
            user_query="Calculate mean without storing results",
            task_objective="Compute mean",
            execution_folder_name="test_no_results",
            approved_code="""
import numpy as np
data = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(data)
print(f"Mean: {mean_value}")
# Oops, forgot to create results!
""",
        )

        # This should raise CodeRuntimeError about missing results
        from osprey.services.python_executor.exceptions import CodeRuntimeError

        with pytest.raises(CodeRuntimeError) as exc_info:
            await service.ainvoke(request, config={"thread_id": "test"})

        # Verify error message
        assert "did not create required 'results' dictionary" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_full_pipeline_succeeds_with_results(self):
        """Full pipeline should succeed when results present."""
        from osprey.services.python_executor import PythonExecutorService

        service = PythonExecutorService()

        request = PythonExecutionRequest(
            user_query="Calculate mean and store results",
            task_objective="Compute mean",
            execution_folder_name="test_with_results",
            approved_code="""
import numpy as np
data = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(data)

results = {
    "mean": float(mean_value),
    "count": len(data)
}
""",
        )

        # This should succeed
        result = await service.ainvoke(request, config={"thread_id": "test"})

        assert result.execution_result.results is not None
        assert "mean" in result.execution_result.results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
